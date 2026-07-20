"""Continuous, resumable supervisor for explicit bounded task contracts."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from ai_team.core.autonomous_backlog import discover_next_task
from ai_team.core.cloud_resilience import (
    CloudModelRoute,
    CloudRecoveryState,
    LocalContinuitySettings,
    RetrySettings,
    SelectedCloudRouteProvider,
    classify_failure,
    create_resume_packet,
)
from ai_team.core.bounded_delivery import (
    BoundedDeliveryOptions,
    DeliveryLimits,
    TrustedTaskContract,
    load_trusted_task_contract,
    run_bounded_delivery,
)
from ai_team.core.ci_monitor import monitor_pull_request
from ai_team.core.external_qa import run_external_qa
from ai_team.core.github_executor import GitHubExecutionOptions, execute_github_action
from ai_team.core.project_loader import load_project
from ai_team.core.trusted_dev import TrustedDevSettings
from ai_team.providers.base import BaseProvider, redact_secrets


RECOVERABLE_STOP_REASONS = {
    "provider-network-error",
    "provider-quota-exhausted",
    "provider-timeout",
}
MAX_CONTRACTS = 256
MAX_CONTRACT_BYTES = 64_000
MAX_TOTAL_CONTRACT_BYTES = 1_000_000
MAX_PROVIDER_QUOTA_BACKOFF_SECONDS = 6 * 60 * 60


@dataclass(frozen=True)
class ContractEntry:
    path: Path
    contract: TrustedTaskContract
    task_sha: str


@dataclass(frozen=True)
class ContinuousBoundedOptions:
    project_path: Path
    contract_dir: Path
    provider_for_role: Callable[[str], BaseProvider]
    workspace_allowlist: list[str] | None
    report_dir: Path
    state_path: Path
    limits: DeliveryLimits = DeliveryLimits()
    once: bool = False
    interval_minutes: int = 15
    max_runtime_minutes: int | None = None
    github_execute: bool = False
    auto_merge: bool = False
    allow_unreviewed_development_merge: bool = False
    ci_wait_seconds: int = 900
    ci_poll_seconds: int = 10
    cloud_routes: tuple[CloudModelRoute, ...] = ()
    cloud_retry: RetrySettings = RetrySettings()
    local_continuity: LocalContinuitySettings = LocalContinuitySettings()
    trusted_dev: TrustedDevSettings = TrustedDevSettings()
    autonomous_product_loop: bool = False
    autonomous_scan_timeout_seconds: int = 180
    delivery_runner: Callable[[BoundedDeliveryOptions], dict[str, Any]] = run_bounded_delivery
    publisher: Callable[["ContinuousBoundedOptions", ContractEntry, dict[str, Any]], dict[str, Any]] | None = None
    sleeper: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic
    wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC)


def run_continuous_bounded_delivery(options: ContinuousBoundedOptions) -> dict[str, Any]:
    _validate_options(options)
    lock_path = options.state_path.with_suffix(options.state_path.suffix + ".lock")
    _acquire_lock(lock_path)
    started = options.monotonic()
    cycles = 0
    prior: dict[str, Any] = {}
    completed: set[str] = set()
    entries: list[ContractEntry] = []
    selected: ContractEntry | None = None
    try:
        while True:
            cycles += 1
            try:
                prior = _read_supervisor_state(options.state_path)
            except ValueError as exc:
                state = _supervisor_state(
                    {},
                    cycles,
                    "attention-required",
                    set(),
                    queue_size=0,
                    stop_reason="supervisor-state-invalid",
                    next_action="manual-review-required",
                    diagnostic=str(exc),
                )
                _write_json(_state_error_path(options.state_path), state)
                return state
            completed = {
                value
                for value in prior.get("completedTaskShas", [])
                if isinstance(value, str)
            }
            try:
                entries = discover_contracts(options.contract_dir)
            except (OSError, ValueError) as exc:
                state = _supervisor_state(
                    prior,
                    cycles,
                    "attention-required",
                    completed,
                    queue_size=0,
                    stop_reason="contract-queue-invalid",
                    next_action="manual-review-required",
                    diagnostic=str(exc),
                )
                _write_json(options.state_path, state)
                return state
            blocked_tasks = _blocked_tasks(entries, completed)
            selected = _next_ready_contract(entries, completed)
            if selected is None:
                pending_count = _pending_count(entries, completed)
                autonomous_backlog: dict[str, Any] | None = None
                if pending_count == 0 and options.autonomous_product_loop:
                    try:
                        loaded_project = load_project(
                            options.project_path,
                            allowlist=options.workspace_allowlist,
                        )
                        commands = loaded_project.profile.commands
                        baseline_commands = (
                            commands.lint,
                            commands.typecheck,
                            commands.test,
                            commands.build,
                        )
                        if not all(
                            isinstance(command, str) and command.strip()
                            for command in baseline_commands
                        ):
                            raise ValueError(
                                "autonomous product loop requires project lint, typecheck, test, and build commands"
                            )
                        autonomous_backlog = discover_next_task(
                            project_path=options.project_path,
                            contract_dir=options.contract_dir,
                            state_path=options.state_path.with_name("autonomous-product-backlog.json"),
                            provider=options.provider_for_role("product-manager"),
                            timeout_seconds=options.autonomous_scan_timeout_seconds,
                            project_validation_commands=tuple(
                                command
                                for command in (*baseline_commands, *commands.additional_validation)
                                if isinstance(command, str)
                            ),
                        )
                    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
                        autonomous_backlog = {
                            "status": "discovery-failed",
                            "diagnostic": str(redact_secrets(str(exc))),
                        }
                    if autonomous_backlog.get("status") == "task-created":
                        state = _supervisor_state(
                            prior,
                            cycles,
                            "planning-next-task",
                            completed,
                            queue_size=1,
                            next_action="run-autonomous-contract",
                            autonomous_backlog=autonomous_backlog,
                        )
                        _write_json(options.state_path, state)
                        continue
                state = _supervisor_state(
                    prior,
                    cycles,
                    "idle" if pending_count == 0 else "attention-required",
                    completed,
                    queue_size=pending_count,
                    stop_reason="dependency-resolution-stalled" if pending_count else None,
                    next_action="watch-contract-directory" if pending_count == 0 else "manual-review-required",
                    blocked_tasks=blocked_tasks,
                    autonomous_backlog=autonomous_backlog,
                )
                _write_json(options.state_path, state)
                if _must_stop(options, started):
                    return state
                options.sleeper(options.interval_minutes * 60)
                continue

            now = options.wall_clock()
            cloud_recovery: CloudRecoveryState | None = None
            selected_route: CloudModelRoute | None = None
            if options.cloud_routes:
                recovery_stage = _recovery_stage(prior, selected)
                cloud_recovery = CloudRecoveryState(
                    task_sha=selected.task_sha,
                    stage=recovery_stage,
                    routes=options.cloud_routes,
                    settings=options.cloud_retry,
                    payload=prior.get("cloudResilience"),
                )
                cloud_remaining = _cloud_recovery_remaining(prior, cloud_recovery, now)
                action, selected_route, next_time = cloud_recovery.next_action(now)
                if cloud_remaining > 0 or action == "cloud_waiting":
                    wait_until = next_time if action == "cloud_waiting" else _cloud_retry_time(cloud_recovery)
                    continuity = _continuity_for_waiting(
                        options, selected, cloud_recovery, prior, wait_until, task_state=None
                    ) if action == "cloud_waiting" else None
                    state = _supervisor_state(
                        prior,
                        cycles,
                        "cloud_waiting" if action == "cloud_waiting" else str(prior.get("status") or "retry_backoff"),
                        completed,
                        queue_size=_pending_count(entries, completed),
                        current=selected,
                        stop_reason="all-cloud-models-temporarily-unavailable" if action == "cloud_waiting" else "transient-provider-backoff",
                        next_action="provider-probe" if action == "cloud_waiting" else "retry-selected-cloud-model",
                        cloud_resilience=cloud_recovery.as_dict(),
                        continuity=continuity,
                    )
                    _write_json(options.state_path, state)
                    if _must_stop(options, started):
                        return state
                    options.sleeper(max(1, cloud_remaining if cloud_remaining > 0 else _seconds_until(wait_until, now)))
                    continue
                if action == "probe" and selected_route is not None:
                    # This is a cheap readiness probe only.  It does not create
                    # a worktree, run an agent, or consume task output.
                    if not cloud_recovery.probe_allowed(now):
                        next_probe = cloud_recovery.next_probe_budget_at(now)
                        state = _supervisor_state(
                            prior, cycles, "cloud_waiting", completed,
                            queue_size=_pending_count(entries, completed), current=selected,
                            stop_reason="provider-probe-budget-exhausted", next_action="provider-probe",
                            cloud_resilience=cloud_recovery.as_dict(),
                        )
                        _write_json(options.state_path, state)
                        if _must_stop(options, started):
                            return state
                        options.sleeper(max(1, _seconds_until(next_probe, now)))
                        continue
                    probe_provider = SelectedCloudRouteProvider(options.provider_for_role("engineer"), selected_route)
                    try:
                        probe_success = probe_provider.ready()
                    except Exception:
                        probe_success = False
                    cloud_recovery.record_probe(selected_route, success=probe_success, now=now)
                    if not probe_success:
                        _continuity_for_waiting(options, selected, cloud_recovery, prior, None, task_state=None)
                        state = _supervisor_state(
                            prior, cycles, "cloud_waiting", completed,
                            queue_size=_pending_count(entries, completed), current=selected,
                            stop_reason="all-cloud-models-temporarily-unavailable", next_action="provider-probe",
                            cloud_resilience=cloud_recovery.as_dict(),
                        )
                        _write_json(options.state_path, state)
                        if _must_stop(options, started):
                            return state
                        options.sleeper(max(1, _seconds_until(cloud_recovery.next_action(now)[2], now)))
                        continue

            remaining_backoff = _provider_backoff_remaining(prior, selected, now)
            if remaining_backoff > 0:
                state = _supervisor_state(
                    prior,
                    cycles,
                    "waiting-provider",
                    completed,
                    queue_size=_pending_count(entries, completed),
                    current=selected,
                    stop_reason="provider-quota-exhausted",
                    next_action="retry-after-provider-reset",
                    provider_backoff=prior.get("providerBackoff"),
                )
                _write_json(options.state_path, state)
                if _must_stop(options, started):
                    return state
                options.sleeper(remaining_backoff)
                continue

            task_dir = options.report_dir / "tasks" / f"{_slug(selected.contract.id)}-{selected.task_sha[:12]}"
            task_state = task_dir / "state.json"
            running = _supervisor_state(
                prior,
                cycles,
                "running",
                completed,
                queue_size=_pending_count(entries, completed),
                current=selected,
                next_action="bounded-delivery",
                cloud_resilience=cloud_recovery.as_dict() if cloud_recovery else None,
                blocked_tasks=blocked_tasks,
            )
            _write_json(options.state_path, running)
            try:
                result = options.delivery_runner(
                    BoundedDeliveryOptions(
                        project_path=options.project_path,
                        task_contract_path=selected.path,
                        provider_for_role=(
                            (lambda role: SelectedCloudRouteProvider(options.provider_for_role(role), selected_route)
                             if role == "engineer" and selected_route is not None else options.provider_for_role(role))
                            if cloud_recovery is not None else options.provider_for_role
                        ),
                        workspace_allowlist=options.workspace_allowlist,
                        report_dir=task_dir / "receipts",
                        state_path=task_state,
                        limits=options.limits,
                        trusted_dev=options.trusted_dev,
                    )
                )
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
                state = _supervisor_state(
                    running,
                    cycles,
                    "attention-required",
                    completed,
                    queue_size=_pending_count(entries, completed),
                    current=selected,
                    stop_reason="bounded-delivery-exception",
                    next_action="manual-review-required",
                    diagnostic=str(exc),
                )
                _write_json(options.state_path, state)
                return state
            task_checkpoint = _read_json(task_state)
            effective_result = task_checkpoint if result.get("status") == "already-completed" else result
            status = str(effective_result.get("status") or result.get("status") or "attention-required")
            if (
                status in {"completed", "already-completed"}
                and effective_result.get("taskSha") != selected.task_sha
            ):
                state = _supervisor_state(
                    running,
                    cycles,
                    "attention-required",
                    completed,
                    queue_size=_pending_count(entries, completed),
                    current=selected,
                    stop_reason="task-contract-changed-during-execution",
                    next_action="manual-review-required",
                    delivery_result=effective_result,
                )
                _write_json(options.state_path, state)
                return state
            if status not in {"completed", "already-completed"}:
                reason = str(effective_result.get("stopReason") or "bounded-delivery-failed")
                if cloud_recovery is not None and selected_route is not None:
                    classification = classify_failure(reason)
                    if classification == "transient_provider_error":
                        transition = cloud_recovery.record_failure(
                            selected_route,
                            reason=reason,
                            now=options.wall_clock(),
                            summary=str(effective_result.get("diagnostic") or ""),
                        )
                        task_checkpoint = _read_json(task_state)
                        waiting_state = str(transition["status"])
                        continuity = (
                            _continuity_for_waiting(
                                options, selected, cloud_recovery, prior,
                                _time_or_none(transition.get("nextRetryAt")), task_state=task_checkpoint,
                            )
                            if waiting_state == "cloud_waiting" else None
                        )
                        state = _supervisor_state(
                            running,
                            cycles,
                            waiting_state,
                            completed,
                            queue_size=_pending_count(entries, completed),
                            current=selected,
                            stop_reason=(
                                "all-cloud-models-temporarily-unavailable"
                                if waiting_state == "cloud_waiting" else reason
                            ),
                            next_action=(
                                "provider-probe" if waiting_state == "cloud_waiting"
                                else "retry-selected-cloud-model"
                            ),
                            delivery_result=effective_result,
                            cloud_resilience=cloud_recovery.as_dict(),
                            continuity=continuity,
                        )
                        _write_json(options.state_path, state)
                        if _must_stop(options, started):
                            return state
                        retry_at = _time_or_none(transition.get("nextRetryAt"))
                        options.sleeper(max(1, _seconds_until(retry_at, options.wall_clock())))
                        continue
                waiting = reason in RECOVERABLE_STOP_REASONS
                provider_backoff = (
                    _next_provider_quota_backoff(
                        prior,
                        selected,
                        str(effective_result.get("stage") or task_checkpoint.get("stage") or "unknown"),
                        options.interval_minutes * 60,
                        options.wall_clock(),
                    )
                    if reason == "provider-quota-exhausted"
                    else None
                )
                state = _supervisor_state(
                    running,
                    cycles,
                    "waiting-provider" if waiting else "attention-required",
                    completed,
                    queue_size=_pending_count(entries, completed),
                    current=selected,
                    stop_reason=reason,
                    next_action="retry-after-provider-reset" if waiting else "manual-review-required",
                    delivery_result=effective_result,
                    provider_backoff=provider_backoff,
                )
                _write_json(options.state_path, state)
                if not waiting or _must_stop(options, started):
                    return state
                options.sleeper(
                    int(provider_backoff["delaySeconds"])
                    if provider_backoff
                    else options.interval_minutes * 60
                )
                continue

            publication: dict[str, Any] | None = None
            if options.github_execute:
                publisher = options.publisher or publish_and_merge
                try:
                    publication = publisher(options, selected, effective_result)
                except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
                    publication = _publication_failure(
                        "publication-exception",
                        diagnostic=str(exc),
                    )
                if publication.get("success") is not True:
                    retryable = publication.get("retryable") is True
                    state = _supervisor_state(
                        running,
                        cycles,
                        "waiting-ci" if retryable else "attention-required",
                        completed,
                        queue_size=_pending_count(entries, completed),
                        current=selected,
                        stop_reason=str(publication.get("stopReason") or "publication-failed"),
                        next_action="retry-publication" if retryable else "manual-review-required",
                        delivery_result=effective_result,
                        publication=publication,
                    )
                    _write_json(options.state_path, state)
                    if not retryable or _must_stop(options, started):
                        return state
                    options.sleeper(options.interval_minutes * 60)
                    continue

            cleanup = (
                cleanup_completed_worktree(options, effective_result)
                if publication is not None and publication.get("success") is True
                else {"attempted": False, "success": True, "reason": "publication-not-completed"}
            )
            if publication is not None:
                publication = {**publication, "worktreeCleanup": cleanup}

            external_qa: dict[str, Any] | None = None
            if options.github_execute and publication is not None and publication.get("success") is True:
                # External payment QA is a human-only attestation gate.  The
                # runner builds its requirement without reading source env
                # files or running a command.
                project_profile_path = options.project_path / ".ai-team" / "project.yaml"
                if project_profile_path.is_file():
                    loaded_source = load_project(
                        options.project_path,
                        allowlist=options.workspace_allowlist,
                    )
                else:
                    loaded_source = None
                if loaded_source is not None and loaded_source.profile.external_qa.enabled:
                    source_revision = _git_head(options.project_path)
                    qa_result = run_external_qa(
                        loaded_source,
                        source_revision,
                        options.report_dir / "external-qa",
                    )
                    external_qa = qa_result.result
                    if qa_result.status == "review-required":
                        state = _supervisor_state(
                            running,
                            cycles,
                            "attention-required",
                            completed,
                            queue_size=_pending_count(entries, completed),
                            current=selected,
                            stop_reason="external-qa-human-attestation-required",
                            next_action="manual-review-required",
                            delivery_result=effective_result,
                            publication=publication,
                            external_qa=external_qa,
                        )
                        _write_json(options.state_path, state)
                        return state
            completed.add(selected.task_sha)
            state = _supervisor_state(
                running,
                cycles,
                "completed",
                completed,
                queue_size=_pending_count(entries, completed),
                current=selected,
                next_action="next-contract",
                delivery_result=effective_result,
                publication=publication,
                external_qa=external_qa,
            )
            _write_json(options.state_path, state)
            if _must_stop(options, started):
                return state
    except KeyboardInterrupt:
        try:
            latest = _read_supervisor_state(options.state_path)
        except (OSError, ValueError):
            latest = prior
        state = _supervisor_state(
            latest,
            cycles,
            "stopped",
            completed,
            queue_size=_pending_count(entries, completed),
            current=selected,
            stop_reason="operator-interrupted",
            next_action="resume-supervisor",
        )
        _write_json(options.state_path, state)
        return state
    finally:
        lock_path.unlink(missing_ok=True)


def discover_contracts(contract_dir: Path) -> list[ContractEntry]:
    if contract_dir.is_symlink():
        raise ValueError("trusted task contract directory must not be a symlink")
    root = contract_dir.resolve()
    if not root.is_dir():
        raise ValueError(f"trusted task contract directory does not exist: {root}")
    paths = sorted(root.glob("*.json"), key=lambda path: path.name)
    if len(paths) > MAX_CONTRACTS:
        raise ValueError(f"trusted task contract directory exceeds {MAX_CONTRACTS} files")
    entries: list[ContractEntry] = []
    ids: set[str] = set()
    total_bytes = 0
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"trusted task contract must be a regular non-symlink file: {path.name}")
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"trusted task contract escapes its directory: {path.name}") from exc
        size = resolved.stat().st_size
        if size > MAX_CONTRACT_BYTES:
            raise ValueError(f"trusted task contract exceeds {MAX_CONTRACT_BYTES} bytes: {path.name}")
        total_bytes += size
        if total_bytes > MAX_TOTAL_CONTRACT_BYTES:
            raise ValueError(f"trusted task contract queue exceeds {MAX_TOTAL_CONTRACT_BYTES} bytes")
        contract, task_sha = load_trusted_task_contract(resolved)
        if contract.id in ids:
            raise ValueError(f"duplicate trusted task id: {contract.id}")
        ids.add(contract.id)
        entries.append(ContractEntry(resolved, contract, task_sha))
    _validate_contract_dependencies(entries)
    return entries


def publish_and_merge(
    options: ContinuousBoundedOptions,
    entry: ContractEntry,
    result: dict[str, Any],
) -> dict[str, Any]:
    worktree_value = result.get("worktreePath")
    run_receipt_value = result.get("runReceipt")
    validation = result.get("validation")
    commit_sha = result.get("commitSha")
    if not (
        isinstance(worktree_value, str)
        and isinstance(run_receipt_value, str)
        and isinstance(validation, dict)
        and isinstance(validation.get("hash"), str)
        and isinstance(commit_sha, str)
        and result.get("taskSha") == entry.task_sha
    ):
        return _publication_failure("bounded-delivery-publication-evidence-missing")
    primary = load_project(options.project_path, allowlist=options.workspace_allowlist)
    if options.allow_unreviewed_development_merge and primary.profile.project.stage != "development":
        return _publication_failure("unreviewed-merge-requires-development-stage")
    require_review = not options.allow_unreviewed_development_merge
    branch = f"codex/auto-{_slug(entry.contract.id)}-{entry.task_sha[:8]}"
    repository = _repository_name(options.project_path)
    existing = _find_pull_request(options.project_path, repository, branch, commit_sha)
    if existing and existing.get("state") == "MERGED":
        synced = _sync_primary(options.project_path, primary.current_branch)
        return {
            "success": True,
            "prUrl": existing.get("url"),
            "commitSha": commit_sha,
            "syncedHead": synced,
            "resumedMergedPullRequest": True,
        }

    # A completed merge can legitimately be resumed after its disposable
    # worktree and local receipt have been cleaned up. Only an unmerged/open
    # publication still needs those local artifacts for push, PR, and CI gates.
    worktree = Path(worktree_value).resolve()
    run_receipt = Path(run_receipt_value).resolve()
    if not worktree.is_dir() or not run_receipt.is_file():
        return _publication_failure("bounded-delivery-publication-artifact-missing")
    loaded = load_project(worktree, allowlist=options.workspace_allowlist)

    validation_hash = str(validation["hash"])
    pr_url = str(existing.get("url")) if existing else ""
    pr_result: dict[str, Any] | None = None
    if not pr_url:
        pr = execute_github_action(
            loaded,
            GitHubExecutionOptions(
                action="pr",
                dry_run=False,
                validation_log_hash=validation_hash,
                receipt_path=run_receipt,
                test_evidence_hash=validation_hash,
                title=str(redact_secrets(entry.contract.title)),
                body=str(redact_secrets(_pull_request_body(entry, result))),
                base_branch=primary.current_branch,
                branch_name=branch,
            ),
        )
        pr_result = pr.as_dict()
        if not pr.success or not pr.pr_url:
            return _publication_failure("pull-request-creation-failed", githubResult=pr_result)
        pr_url = pr.pr_url

    ci = monitor_pull_request(
        project_root=worktree,
        repository=repository,
        pr_identifier=pr_url,
        report_dir=options.report_dir / "ci" / f"{_slug(entry.contract.id)}-{entry.task_sha[:12]}",
        wait_seconds=options.ci_wait_seconds,
        poll_seconds=options.ci_poll_seconds,
        require_approved_review=require_review,
    )
    if not ci.merge_ready:
        retryable = ci.status in {"pending", "timed_out", "no_required_checks"}
        return _publication_failure(
            "ci-not-merge-ready",
            retryable=retryable,
            prUrl=pr_url,
            ciStatus=ci.status,
            ciEvidence=str(ci.evidence_path),
            blockers=ci.evidence.get("blockers", []),
        )
    if not options.auto_merge:
        return _publication_failure("automatic-merge-disabled", prUrl=pr_url)

    merge = execute_github_action(
        loaded,
        GitHubExecutionOptions(
            action="merge",
            dry_run=False,
            validation_log_hash=validation_hash,
            receipt_path=run_receipt,
            test_evidence_hash=validation_hash,
            pr_identifier=pr_url,
            merge_method="squash",
            require_approved_review=require_review,
        ),
    )
    if not merge.success:
        return _publication_failure(
            "pull-request-merge-failed",
            prUrl=pr_url,
            ciEvidence=str(ci.evidence_path),
            githubResult=merge.as_dict(),
        )
    synced = _sync_primary(options.project_path, primary.current_branch)
    return {
        "success": True,
        "prUrl": pr_url,
        "commitSha": commit_sha,
        "ciStatus": ci.status,
        "ciEvidence": str(ci.evidence_path),
        "syncedHead": synced,
        "githubResult": merge.as_dict(),
        "prCreation": pr_result,
    }


def cleanup_completed_worktree(
    options: ContinuousBoundedOptions,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Remove only a clean, merged disposable worktree from this repository."""
    if not options.trusted_dev.cleanup_worktree_after_merge:
        return {"attempted": False, "success": True, "reason": "cleanup-disabled"}
    value = result.get("worktreePath")
    if not isinstance(value, str) or not value:
        return {"attempted": False, "success": False, "reason": "worktree-path-missing"}
    worktree = Path(value).resolve()
    if not worktree.exists():
        return {
            "attempted": False,
            "success": True,
            "reason": "merged-worktree-already-absent",
        }
    try:
        primary = load_project(options.project_path, allowlist=options.workspace_allowlist)
        loaded = load_project(worktree, allowlist=options.workspace_allowlist)
        if (
            worktree == primary.root
            or not loaded.is_disposable_worktree()
            or _git_common_directory(primary.root) != _git_common_directory(loaded.root)
        ):
            return {
                "attempted": False,
                "success": False,
                "reason": "worktree-identity-invalid",
            }
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=worktree,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if status.returncode != 0 or status.stdout.strip():
            return {
                "attempted": False,
                "success": False,
                "reason": "worktree-not-clean",
            }
        removed = subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=primary.root,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if removed.returncode != 0:
            return {
                "attempted": True,
                "success": False,
                "reason": "worktree-remove-failed",
                "diagnostic": str(redact_secrets(removed.stderr))[-2000:],
            }
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        return {
            "attempted": True,
            "success": False,
            "reason": "worktree-cleanup-exception",
            "diagnostic": str(redact_secrets(str(exc)))[:2000],
        }
    return {
        "attempted": True,
        "success": True,
        "reason": "merged-worktree-removed",
    }


def _git_common_directory(root: Path) -> Path:
    value = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if value.returncode != 0 or not value.stdout.strip():
        raise ValueError("cannot resolve Git common directory")
    path = Path(value.stdout.strip())
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise ValueError("cannot resolve source Git revision for external QA")
    return result.stdout.strip()


def _validate_options(options: ContinuousBoundedOptions) -> None:
    if options.interval_minutes < 1:
        raise ValueError("interval_minutes must be at least 1")
    if options.ci_wait_seconds < 1 or options.ci_poll_seconds < 1:
        raise ValueError("CI wait and poll values must be positive")
    if options.auto_merge and not options.github_execute:
        raise ValueError("automatic merge requires GitHub execution")
    if not options.once and (not options.github_execute or not options.auto_merge):
        raise ValueError("continuous bounded delivery requires PR execution and automatic merge")


def _must_stop(options: ContinuousBoundedOptions, started: float) -> bool:
    if options.once:
        return True
    return (
        options.max_runtime_minutes is not None
        and options.monotonic() - started >= options.max_runtime_minutes * 60
    )


def _supervisor_state(
    prior: dict[str, Any],
    cycle: int,
    status: str,
    completed: set[str],
    *,
    queue_size: int,
    current: ContractEntry | None = None,
    stop_reason: str | None = None,
    next_action: str,
    delivery_result: dict[str, Any] | None = None,
    publication: dict[str, Any] | None = None,
    external_qa: dict[str, Any] | None = None,
    diagnostic: str | None = None,
    provider_backoff: dict[str, Any] | None = None,
    cloud_resilience: dict[str, Any] | None = None,
    continuity: dict[str, Any] | None = None,
    blocked_tasks: list[dict[str, Any]] | None = None,
    autonomous_backlog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        # Version 2 adds cloudResilience and continuity while readers continue
        # to accept version-1 state that has only providerBackoff.
        "schemaVersion": 2,
        "revision": int(prior.get("revision") or 0) + 1,
        "updatedAt": datetime.now(UTC).isoformat(),
        "status": status,
        "cycleNumber": cycle,
        "queueSize": queue_size,
        "completedTaskShas": sorted(completed),
        "currentTask": (
            {
                "id": current.contract.id,
                "taskSha": current.task_sha,
                "contractPath": str(current.path),
                "dependsOn": list(current.contract.depends_on),
            }
            if current
            else None
        ),
        "stopReason": stop_reason,
        "nextAction": next_action,
        "blockedTasks": redact_secrets(
            blocked_tasks if blocked_tasks is not None else prior.get("blockedTasks", [])
        ),
        "delivery": _result_summary(delivery_result),
        "publication": publication,
        "externalQa": external_qa if external_qa is not None else prior.get("externalQa"),
        "diagnostic": diagnostic,
        "providerBackoff": provider_backoff,
        "cloudResilience": cloud_resilience,
        "continuity": continuity,
        "autonomousBacklog": redact_secrets(
            autonomous_backlog if autonomous_backlog is not None else prior.get("autonomousBacklog")
        ),
    }


def _recovery_stage(prior: dict[str, Any], current: ContractEntry) -> str:
    recovery = prior.get("cloudResilience")
    if isinstance(recovery, dict) and recovery.get("taskSha") == current.task_sha:
        stage = recovery.get("stage")
        if isinstance(stage, str) and stage:
            return stage
    delivery = prior.get("delivery")
    if isinstance(delivery, dict) and isinstance(delivery.get("stage"), str):
        return str(delivery["stage"])
    return "engineer"


def _cloud_recovery_remaining(prior: dict[str, Any], recovery: CloudRecoveryState, now: datetime) -> int:
    if prior.get("status") not in {"retry_backoff", "provider_fallback", "cloud_waiting"}:
        return 0
    retry_at = _cloud_retry_time(recovery)
    return _seconds_until(retry_at, now)


def _cloud_retry_time(recovery: CloudRecoveryState) -> datetime | None:
    circuit = recovery.circuits.get(recovery.current_route().key)
    if isinstance(circuit, dict):
        return _time_or_none(circuit.get("nextRetryAt")) or _time_or_none(circuit.get("nextProbeAt"))
    return None


def _seconds_until(value: datetime | None, now: datetime) -> int:
    if value is None:
        return 1
    return max(0, int((_as_utc(value) - _as_utc(now)).total_seconds() + 0.999999))


def _time_or_none(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return _as_utc(datetime.fromisoformat(value))
    except (TypeError, ValueError):
        return None


def _continuity_for_waiting(
    options: ContinuousBoundedOptions,
    entry: ContractEntry,
    recovery: CloudRecoveryState,
    prior: dict[str, Any],
    _next_retry_at: datetime | None,
    *,
    task_state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Create a recorder-only packet; a packet failure never kills supervision."""

    if not options.local_continuity.enabled:
        return {"status": "disabled", "repositoryModifications": "none"}
    try:
        checkpoint = task_state
        if checkpoint is None:
            task_dir = options.report_dir / "tasks" / f"{_slug(entry.contract.id)}-{entry.task_sha[:12]}"
            checkpoint = _read_json(task_dir / "state.json")
        snapshot_path = options.project_path.resolve()
        worktree_value = checkpoint.get("worktreePath") if isinstance(checkpoint, dict) else None
        if isinstance(worktree_value, str):
            candidate = Path(worktree_value).resolve()
            allowed_parent = options.project_path.resolve().parent
            if candidate.is_dir() and allowed_parent in candidate.parents and _same_git_repository(
                candidate, options.project_path.resolve()
            ):
                snapshot_path = candidate
        delivery = prior.get("delivery") if isinstance(prior.get("delivery"), dict) else {}
        recovered_receipts = checkpoint.get("receipts") if isinstance(checkpoint, dict) else []
        receipts = [str(value) for value in recovered_receipts if isinstance(value, str)] if isinstance(recovered_receipts, list) else []
        receipts.extend(
            str(value)
            for value in (delivery or {}).values()
            if isinstance(value, str) and ("receipt" in value.lower() or value.endswith(".json"))
        )
        packet = create_resume_packet(
            state_root=options.state_path.parent,
            project_path=snapshot_path,
            project_id=options.project_path.name,
            task_id=entry.contract.id,
            task_sha=entry.task_sha,
            task_title=entry.contract.title,
            task_state=checkpoint or {},
            supervisor_state=recovery,
            receipt_paths=receipts,
            continuity=options.local_continuity,
            now=options.wall_clock(),
        )
        return {"status": "completed", "repositoryModifications": "none", "resumePacket": packet}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"status": "deterministic-fallback-failed", "repositoryModifications": "none", "diagnostic": str(redact_secrets(str(exc)))}


def _same_git_repository(left: Path, right: Path) -> bool:
    def common(path: Path) -> Path | None:
        result = _run(["git", "rev-parse", "--git-common-dir"], path, 15)
        if result.returncode != 0 or not result.stdout.strip():
            return None
        value = Path(result.stdout.strip())
        return (path / value).resolve() if not value.is_absolute() else value.resolve()

    left_common = common(left)
    right_common = common(right)
    return left_common is not None and left_common == right_common


def _next_provider_quota_backoff(
    prior: dict[str, Any],
    current: ContractEntry,
    stage: str,
    base_delay_seconds: int,
    now: datetime,
) -> dict[str, Any]:
    previous = prior.get("providerBackoff")
    consecutive_failures = 1
    if (
        isinstance(previous, dict)
        and previous.get("taskSha") == current.task_sha
        and previous.get("stage") == stage
        and previous.get("stopReason") == "provider-quota-exhausted"
    ):
        previous_count = previous.get("consecutiveFailures")
        if isinstance(previous_count, int) and not isinstance(previous_count, bool):
            consecutive_failures = previous_count + 1
    exponent = min(consecutive_failures - 1, 20)
    maximum = max(base_delay_seconds, MAX_PROVIDER_QUOTA_BACKOFF_SECONDS)
    delay_seconds = min(base_delay_seconds * (2**exponent), maximum)
    retry_at = _as_utc(now) + timedelta(seconds=delay_seconds)
    return {
        "taskSha": current.task_sha,
        "stage": stage,
        "stopReason": "provider-quota-exhausted",
        "consecutiveFailures": consecutive_failures,
        "delaySeconds": delay_seconds,
        "nextRetryAt": retry_at.isoformat(),
    }


def _provider_backoff_remaining(
    prior: dict[str, Any],
    current: ContractEntry,
    now: datetime,
) -> int:
    if (
        prior.get("status") != "waiting-provider"
        or prior.get("stopReason") != "provider-quota-exhausted"
    ):
        return 0
    backoff = prior.get("providerBackoff")
    if not isinstance(backoff, dict) or backoff.get("taskSha") != current.task_sha:
        return 0
    retry_at = datetime.fromisoformat(str(backoff["nextRetryAt"]))
    remaining = (retry_at - _as_utc(now)).total_seconds()
    return max(0, int(remaining + 0.999999))


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("wall clock must be timezone-aware")
    return value.astimezone(UTC)


def _result_summary(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    return {
        key: result.get(key)
        for key in (
            "status",
            "stopReason",
            "taskSha",
            "commitSha",
            "worktreePath",
            "runReceipt",
            "executorReceipt",
            "statePath",
        )
    }


def _publication_failure(reason: str, *, retryable: bool = False, **evidence: Any) -> dict[str, Any]:
    return redact_secrets({
        "success": False,
        "retryable": retryable,
        "stopReason": reason,
        **evidence,
    })


def _pull_request_body(entry: ContractEntry, result: dict[str, Any]) -> str:
    return "\n".join((
        "## Trusted task",
        "",
        f"- Contract: `{entry.contract.id}`",
        f"- Task SHA: `{entry.task_sha}`",
        f"- Source: `{entry.contract.source_kind}:{entry.contract.source_reference}`",
        "",
        "## Bounded delivery evidence",
        "",
        f"- Commit: `{result.get('commitSha')}`",
        f"- Validation: `{(result.get('validation') or {}).get('hash')}`",
        "- PM, Architect, Engineer, QA and Reviewer receipts are retained by the control plane.",
        "- Production deployment, migration, seed, real payment and destructive operations were not executed.",
    ))


def _repository_name(cwd: Path) -> str:
    result = _run(["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"], cwd, 30)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("GitHub repository identity is unavailable")
    return result.stdout.strip()


def _find_pull_request(cwd: Path, repository: str, branch: str, commit_sha: str) -> dict[str, Any] | None:
    result = _run(
        [
            "gh", "pr", "list", "--repo", repository, "--head", branch, "--state", "all",
            "--limit", "1", "--json", "url,state,headRefOid",
        ],
        cwd,
        30,
    )
    if result.returncode != 0:
        raise RuntimeError("GitHub pull request lookup failed")
    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError("GitHub pull request lookup returned invalid JSON") from exc
    if not isinstance(payload, list) or not payload:
        return None
    item = payload[0]
    if not isinstance(item, dict) or item.get("headRefOid") != commit_sha:
        raise RuntimeError("existing pull request head does not match the attested commit")
    return item


def _sync_primary(project_path: Path, branch: str | None) -> str:
    if not branch:
        raise RuntimeError("primary project branch is unavailable")
    status = _run(["git", "status", "--porcelain"], project_path, 30)
    if status.returncode != 0:
        raise RuntimeError("unable to inspect primary project worktree status")
    if status.stdout.strip():
        raise RuntimeError("primary project worktree must be clean before fast-forward sync")
    current_branch = _run(["git", "branch", "--show-current"], project_path, 30)
    if current_branch.returncode != 0 or current_branch.stdout.strip() != branch:
        raise RuntimeError("primary project worktree is not on the expected branch")
    fetch = _run(["git", "fetch", "origin", branch], project_path, 120)
    if fetch.returncode != 0:
        raise RuntimeError("failed to fetch merged primary branch")
    ancestor = _run(["git", "merge-base", "--is-ancestor", "HEAD", f"origin/{branch}"], project_path, 30)
    if ancestor.returncode != 0:
        raise RuntimeError("primary branch cannot be fast-forwarded to the merged remote branch")
    merge = _run(["git", "merge", "--ff-only", f"origin/{branch}"], project_path, 120)
    if merge.returncode != 0:
        raise RuntimeError("primary branch fast-forward sync failed")
    head = _run(["git", "rev-parse", "HEAD"], project_path, 30)
    if head.returncode != 0:
        raise RuntimeError("unable to read synchronized primary HEAD")
    return head.stdout.strip()


def _slug(value: str) -> str:
    safe = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return "-".join(part for part in safe.split("-") if part)[:80] or "task"


def _validate_contract_dependencies(entries: list[ContractEntry]) -> None:
    by_id = {entry.contract.id: entry for entry in entries}
    for entry in entries:
        missing = [dependency for dependency in entry.contract.depends_on if dependency not in by_id]
        if missing:
            raise ValueError(
                f"trusted task contract {entry.contract.id} has unknown dependencies: {', '.join(missing)}"
            )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise ValueError("trusted task contract dependencies contain a cycle")
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency in by_id[task_id].contract.depends_on:
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in by_id:
        visit(task_id)


def _completed_contract_ids(entries: list[ContractEntry], completed: set[str]) -> set[str]:
    return {entry.contract.id for entry in entries if entry.task_sha in completed}


def _next_ready_contract(entries: list[ContractEntry], completed: set[str]) -> ContractEntry | None:
    completed_ids = _completed_contract_ids(entries, completed)
    return next(
        (
            entry
            for entry in entries
            if entry.task_sha not in completed
            and set(entry.contract.depends_on).issubset(completed_ids)
        ),
        None,
    )


def _blocked_tasks(entries: list[ContractEntry], completed: set[str]) -> list[dict[str, Any]]:
    completed_ids = _completed_contract_ids(entries, completed)
    return [
        {
            "id": entry.contract.id,
            "taskSha": entry.task_sha,
            "dependsOn": list(entry.contract.depends_on),
            "unmetDependencies": [
                dependency for dependency in entry.contract.depends_on if dependency not in completed_ids
            ],
        }
        for entry in entries
        if entry.task_sha not in completed
        and not set(entry.contract.depends_on).issubset(completed_ids)
    ]


def _pending_count(entries: list[ContractEntry], completed: set[str]) -> int:
    return sum(entry.task_sha not in completed for entry in entries)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_supervisor_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("continuous bounded delivery state is unreadable or invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("continuous bounded delivery state must be a JSON object")
    _validate_provider_backoff(value.get("providerBackoff"))
    _validate_cloud_resilience(value.get("cloudResilience"))
    return value


def _validate_cloud_resilience(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError("continuous bounded delivery cloud resilience must be an object")
    required = {"schemaVersion", "taskSha", "stage", "currentRoute", "routes", "circuits", "retryHistory", "probes", "automaticRecovery", "preferredRoute"}
    if set(value) != required:
        raise ValueError("continuous bounded delivery cloud resilience has invalid fields")
    if value.get("schemaVersion") != 1 or value.get("automaticRecovery") is not True:
        raise ValueError("continuous bounded delivery cloud resilience schema is invalid")
    if not isinstance(value.get("taskSha"), str) or len(value["taskSha"]) != 64:
        raise ValueError("continuous bounded delivery cloud resilience task SHA is invalid")
    if not isinstance(value.get("stage"), str) or not isinstance(value.get("currentRoute"), str):
        raise ValueError("continuous bounded delivery cloud resilience route is invalid")
    if not isinstance(value.get("routes"), list) or not isinstance(value.get("circuits"), dict):
        raise ValueError("continuous bounded delivery cloud resilience routes are invalid")


def _validate_provider_backoff(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError("continuous bounded delivery provider backoff must be an object")
    required = {
        "taskSha",
        "stage",
        "stopReason",
        "consecutiveFailures",
        "delaySeconds",
        "nextRetryAt",
    }
    if set(value) != required:
        raise ValueError("continuous bounded delivery provider backoff has invalid fields")
    task_sha = value.get("taskSha")
    if not (
        isinstance(task_sha, str)
        and len(task_sha) == 64
        and all(character in "0123456789abcdef" for character in task_sha)
    ):
        raise ValueError("continuous bounded delivery provider backoff task SHA is invalid")
    if value.get("stopReason") != "provider-quota-exhausted":
        raise ValueError("continuous bounded delivery provider backoff reason is invalid")
    stage = value.get("stage")
    if not (
        isinstance(stage, str)
        and 1 <= len(stage) <= 64
        and all(character.isalnum() or character in "-_" for character in stage)
    ):
        raise ValueError("continuous bounded delivery provider backoff stage is invalid")
    for field in ("consecutiveFailures", "delaySeconds"):
        item = value.get(field)
        if not isinstance(item, int) or isinstance(item, bool) or item < 1:
            raise ValueError(f"continuous bounded delivery provider backoff {field} is invalid")
    try:
        retry_at = datetime.fromisoformat(str(value.get("nextRetryAt")))
        _as_utc(retry_at)
    except (TypeError, ValueError) as exc:
        raise ValueError("continuous bounded delivery provider backoff retry time is invalid") from exc


def _state_error_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.error.json")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(redact_secrets(payload), indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _acquire_lock(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        existing = _read_json(path)
        pid = existing.get("pid")
        if isinstance(pid, int) and not _pid_exists(pid):
            path.unlink(missing_ok=True)
            return _acquire_lock(path)
        raise RuntimeError(f"continuous bounded delivery already running: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "createdAt": datetime.now(UTC).isoformat()}, handle)


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _run(args: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(args, 127, "", str(exc))
