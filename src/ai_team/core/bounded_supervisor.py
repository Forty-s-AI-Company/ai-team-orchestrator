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

from ai_team.core.bounded_delivery import (
    BoundedDeliveryOptions,
    DeliveryLimits,
    TrustedTaskContract,
    load_trusted_task_contract,
    run_bounded_delivery,
)
from ai_team.core.ci_monitor import monitor_pull_request
from ai_team.core.github_executor import GitHubExecutionOptions, execute_github_action
from ai_team.core.project_loader import load_project
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
            selected = next((entry for entry in entries if entry.task_sha not in completed), None)
            if selected is None:
                state = _supervisor_state(
                    prior,
                    cycles,
                    "idle",
                    completed,
                    queue_size=0,
                    next_action="watch-contract-directory",
                )
                _write_json(options.state_path, state)
                if _must_stop(options, started):
                    return state
                options.sleeper(options.interval_minutes * 60)
                continue

            remaining_backoff = _provider_backoff_remaining(
                prior,
                selected,
                options.wall_clock(),
            )
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
            )
            _write_json(options.state_path, running)
            try:
                result = options.delivery_runner(
                    BoundedDeliveryOptions(
                        project_path=options.project_path,
                        task_contract_path=selected.path,
                        provider_for_role=options.provider_for_role,
                        workspace_allowlist=options.workspace_allowlist,
                        report_dir=task_dir / "receipts",
                        state_path=task_state,
                        limits=options.limits,
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
            effective_result = _read_json(task_state) if result.get("status") == "already-completed" else result
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
                waiting = reason in RECOVERABLE_STOP_REASONS
                provider_backoff = (
                    _next_provider_quota_backoff(
                        prior,
                        selected,
                        str(effective_result.get("stage") or "unknown"),
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
            )
            _write_json(options.state_path, state)
            if _must_stop(options, started):
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
    worktree = Path(worktree_value).resolve()
    run_receipt = Path(run_receipt_value).resolve()
    if not worktree.is_dir() or not run_receipt.is_file():
        return _publication_failure("bounded-delivery-publication-artifact-missing")

    primary = load_project(options.project_path, allowlist=options.workspace_allowlist)
    if options.allow_unreviewed_development_merge and primary.profile.project.stage != "development":
        return _publication_failure("unreviewed-merge-requires-development-stage")
    require_review = not options.allow_unreviewed_development_merge
    loaded = load_project(worktree, allowlist=options.workspace_allowlist)
    branch = f"codex/auto-{_slug(entry.contract.id)}-{entry.task_sha[:8]}"
    repository = _repository_name(worktree)
    existing = _find_pull_request(worktree, repository, branch, commit_sha)
    if existing and existing.get("state") == "MERGED":
        synced = _sync_primary(options.project_path, primary.current_branch)
        return {
            "success": True,
            "prUrl": existing.get("url"),
            "commitSha": commit_sha,
            "syncedHead": synced,
            "resumedMergedPullRequest": True,
        }

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
    diagnostic: str | None = None,
    provider_backoff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
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
            }
            if current
            else None
        ),
        "stopReason": stop_reason,
        "nextAction": next_action,
        "delivery": _result_summary(delivery_result),
        "publication": publication,
        "diagnostic": diagnostic,
        "providerBackoff": provider_backoff,
    }


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
    return value


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
