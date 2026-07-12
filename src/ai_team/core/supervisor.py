from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from ai_team.core.fallback_policy import decide_fallback, state_timestamp
from ai_team.core.isolated_executor import run_in_disposable_worktree
from ai_team.core.orchestrator import Orchestrator
from ai_team.core.orchestrator import load_workflow
from ai_team.core.project_loader import LoadedProject, ProjectConfigError, load_project
from ai_team.providers.base import BaseProvider, ProviderErrorType, ProviderResult, redact_secrets


@dataclass(frozen=True)
class SupervisorOptions:
    project_path: Path
    provider: BaseProvider
    workflow: str = "project-analysis"
    run_mode: str = "create-only"
    dry_run: bool = True
    once: bool = False
    interval_minutes: int = 60
    max_runtime_minutes: int | None = None
    report_dir: Path = Path("reports/supervisor")
    workspace_allowlist: list[str] | None = None
    state_path: Path | None = None
    isolated_auto_commit: bool = False
    github_action: str | None = None
    github_execute: bool = False
    validation_log_hash: str | None = None
    test_evidence_hash: str | None = None


@dataclass(frozen=True)
class SupervisorRunSummary:
    report_paths: list[Path]
    completed_cycles: int
    stopped_reason: str


def run_supervisor(options: SupervisorOptions) -> SupervisorRunSummary:
    if options.interval_minutes < 1:
        raise ValueError("interval_minutes must be at least 1")

    started = datetime.now(UTC)
    deadline = (
        started + timedelta(minutes=options.max_runtime_minutes)
        if options.max_runtime_minutes is not None
        else None
    )
    report_paths: list[Path] = []
    completed_cycles = 0

    while True:
        report_paths.append(run_supervisor_cycle(options, cycle_number=completed_cycles + 1))
        completed_cycles += 1

        if options.once:
            return SupervisorRunSummary(report_paths, completed_cycles, "once")

        if deadline and datetime.now(UTC) >= deadline:
            return SupervisorRunSummary(report_paths, completed_cycles, "max-runtime")

        time.sleep(options.interval_minutes * 60)


def run_supervisor_cycle(options: SupervisorOptions, cycle_number: int) -> Path:
    started = datetime.now(UTC)
    state_path = _state_path(options)
    previous_state = _load_state(state_path)
    _write_state(
        state_path,
        {
            "schemaVersion": 1,
            "revision": int(previous_state.get("revision") or 0) + 1,
            "status": "running",
            "provider": options.provider.name,
            "workflow": options.workflow,
            "runMode": options.run_mode,
            "cycleNumber": cycle_number,
            "startedAt": started.isoformat(),
            "previous": _state_summary(previous_state),
        },
    )
    loaded: LoadedProject | None = None
    project_error: str | None = None

    try:
        loaded = load_project(options.project_path, allowlist=options.workspace_allowlist)
    except ProjectConfigError as exc:
        project_error = str(exc)

    stages: list[dict[str, Any]] = []
    stages.append(_stage("discovery", loaded is not None, {"projectError": project_error}))
    stages.append(_stage("quality-review", loaded is not None, _quality_review(loaded)))
    stages.append(_stage("triage", loaded is not None, _triage(loaded)))

    provider_result: ProviderResult | None = None
    provider_error_message: str | None = None
    isolated_details: dict[str, Any] | None = None
    if loaded is not None:
        try:
            workflow = load_workflow(options.workflow)
            if workflow.write_required and not options.dry_run:
                isolated = run_in_disposable_worktree(
                    source_project_path=options.project_path,
                    provider=options.provider,
                    workflow_name=options.workflow,
                    workspace_allowlist=options.workspace_allowlist,
                    receipt_dir=options.report_dir / "isolated",
                    worktree_parent=options.project_path.resolve().parent,
                    dry_run=False,
                    run_mode=options.run_mode,
                    keep_worktree=True,
                    auto_commit=options.isolated_auto_commit,
                    github_action=options.github_action,
                    github_execute=options.github_execute,
                    validation_log_hash=options.validation_log_hash,
                    test_evidence_hash=options.test_evidence_hash,
                )
                provider_result = isolated.workflow_result.provider_result
                isolated_details = {
                    "worktreePath": str(isolated.worktree_path),
                    "runReceipt": str(isolated.run_receipt),
                    "executorReceipt": str(isolated.executor_receipt),
                    "gitPolicy": isolated.git_policy,
                    "commitResult": isolated.commit_result,
                    "githubResult": isolated.github_result,
                }
            else:
                provider_result = Orchestrator(options.provider, max_retries=1).run(
                    loaded,
                    workflow_name=options.workflow,
                    dry_run=options.dry_run,
                    run_mode=options.run_mode,
                ).provider_result
            github_success = not isolated_details or not isolated_details.get("githubResult") or bool(
                isolated_details["githubResult"].get("success")
            )
            auto_cycle_success = (
                provider_result.success or provider_result.error_type == ProviderErrorType.EXTERNAL_REQUIRED
            ) and github_success
            stages.append(
                _stage(
                    "auto-cycle",
                    auto_cycle_success,
                    {
                        "provider": provider_result.provider,
                        "success": provider_result.success,
                        "errorType": provider_result.error_type,
                        "externalRequired": provider_result.data.get("externalRequired"),
                        "isolatedExecutor": isolated_details,
                    },
                )
            )
        except Exception as exc:
            provider_error_message = str(exc)
            stages.append(_stage("auto-cycle", False, {"message": str(exc)}))
    else:
        stages.append(_stage("auto-cycle", False, {"message": "project profile unavailable"}))

    fallback_decision = decide_fallback(provider_result, options.workflow)
    stages.append(
        _stage(
            "fallback-policy",
            True,
            {
                **fallback_decision.as_dict(),
                "provider": provider_result.provider if provider_result else options.provider.name,
                "runtimeProvider": (provider_result.data.get("runtimeProvider") if provider_result else None),
                "note": "Ollama fallback is never reported as Codex or Antigravity provider-native pass.",
            },
        )
    )
    stages.append(_stage("qa-handoff", True, _qa_handoff(provider_result)))
    stages.append(_stage("regression", True, _regression_plan(loaded)))
    stages.append(_stage("git-commit-evidence", True, _git_evidence(loaded)))

    completed = datetime.now(UTC)
    report = {
        "schemaVersion": 1,
        "cycleId": uuid4().hex,
        "cycleNumber": cycle_number,
        "startedAt": started.isoformat(),
        "completedAt": completed.isoformat(),
        "durationMs": int((completed - started).total_seconds() * 1000),
        "projectPath": str(options.project_path.resolve()),
        "workflow": options.workflow,
        "runMode": options.run_mode,
        "dryRun": options.dry_run,
        "provider": options.provider.name,
        "statePath": str(state_path),
        "resume": {
            "previousState": _state_summary(previous_state),
            "duplicateResumeSafe": True,
        },
        "stages": stages,
        "status": "completed" if all(stage["ok"] for stage in stages) else "attention_required",
    }

    options.report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = started.isoformat().replace(":", "").replace("+", "Z").replace(".", "")
    path = options.report_dir / f"{timestamp}-cycle-{cycle_number}-{uuid4().hex[:8]}.json"
    path.write_text(json.dumps(redact_secrets(report), indent=2, default=str), encoding="utf-8")
    _write_state(
        state_path,
        {
            "schemaVersion": 1,
            "revision": int(previous_state.get("revision") or 0) + 2,
            "status": report["status"],
            "provider": options.provider.name,
            "workflow": options.workflow,
            "runMode": options.run_mode,
            "cycleNumber": cycle_number,
            "startedAt": started.isoformat(),
            "completedAt": completed.isoformat(),
            "lastReportPath": str(path),
            "providerResult": _provider_state(provider_result),
            "fallbackPolicy": fallback_decision.as_dict(),
            "providerErrorMessage": provider_error_message,
            "nextAction": _next_action(provider_result, fallback_decision),
            "updatedAt": state_timestamp(),
        },
    )
    return path


def _stage(name: str, ok: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "ok": ok,
        "details": redact_secrets(details or {}),
    }


def _quality_review(loaded: LoadedProject | None) -> dict[str, Any]:
    if loaded is None:
        return {"message": "skipped"}
    return {
        "protectedBranch": loaded.is_branch_protected(),
        "disposableWorktree": loaded.is_disposable_worktree(),
        "allowDeploy": loaded.profile.safety.allow_deploy,
        "allowGitPush": loaded.profile.safety.allow_git_push,
        "allowDestructiveCommands": loaded.profile.safety.allow_destructive_commands,
    }


def _triage(loaded: LoadedProject | None) -> dict[str, Any]:
    if loaded is None:
        return {"autoExecutableTasks": []}
    if loaded.is_branch_protected():
        return {
            "autoExecutableTasks": [],
            "blockedReason": "protected branch requires disposable worktree for writes",
        }
    return {
        "autoExecutableTasks": ["project-analysis"],
        "writeTasksRequireDisposableWorktree": loaded.profile.safety.require_disposable_worktree_for_writes,
    }


def _qa_handoff(provider_result: ProviderResult | None) -> dict[str, Any]:
    if provider_result is None:
        return {"status": "not-created"}
    if provider_result.provider == "openhands" and provider_result.success:
        return {"status": "provider-native-ready", "provider": "openhands", "conversationId": provider_result.conversation_id}
    if provider_result.provider == "handsfreecode" and provider_result.success:
        return {
            "status": "provider-native-ready",
            "provider": "handsfreecode",
            "conversationId": provider_result.conversation_id,
            "taskId": provider_result.task_id,
            "runtimeProvider": provider_result.data.get("runtimeProvider"),
            "masqueradeAsCodexOrAntigravity": False,
        }
    if provider_result.error_type == ProviderErrorType.EXTERNAL_REQUIRED:
        return {"status": "external-required", "details": provider_result.data.get("externalRequired")}
    return {"status": "manual-review-required", "errorType": provider_result.error_type}


def _regression_plan(loaded: LoadedProject | None) -> dict[str, Any]:
    if loaded is None:
        return {"commands": []}
    commands = loaded.profile.commands.model_dump(exclude_none=True)
    return {
        "commands": commands,
        "execution": "planned-only",
        "note": "supervisor does not run product test commands unless a future policy explicitly enables it",
    }


def _git_evidence(loaded: LoadedProject | None) -> dict[str, Any]:
    if loaded is None:
        return {"available": False}
    status = _git_status(loaded.root)
    gh_available = shutil.which("gh") is not None
    return {
        "available": True,
        "branch": loaded.current_branch,
        "commitSha": loaded.commit_sha,
        "statusShort": status,
        "githubCliAvailable": gh_available,
        "pushPolicy": "blocked-by-default" if not loaded.profile.safety.allow_git_push else "policy-enabled",
        "mergePolicy": "external-required-branch-protection-and-review",
    }


def _git_status(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"
    return result.stdout.strip()


def _state_path(options: SupervisorOptions) -> Path:
    if options.state_path is not None:
        return options.state_path
    return options.report_dir / "state.json"


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "unreadable", "path": str(path)}
    return raw if isinstance(raw, dict) else {}


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(redact_secrets(payload), indent=2, default=str), encoding="utf-8")


def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
    if not state:
        return {"available": False}
    return {
        "available": True,
        "revision": state.get("revision"),
        "status": state.get("status"),
        "provider": state.get("provider"),
        "workflow": state.get("workflow"),
        "cycleNumber": state.get("cycleNumber"),
        "lastReportPath": state.get("lastReportPath"),
        "nextAction": state.get("nextAction"),
    }


def _provider_state(provider_result: ProviderResult | None) -> dict[str, Any]:
    if provider_result is None:
        return {"available": False}
    return {
        "available": True,
        "provider": provider_result.provider,
        "success": provider_result.success,
        "errorType": provider_result.error_type,
        "conversationId": provider_result.conversation_id,
        "taskId": provider_result.task_id,
        "runtimeProvider": provider_result.data.get("runtimeProvider"),
        "externalRequired": provider_result.data.get("externalRequired"),
    }


def _next_action(provider_result: ProviderResult | None, fallback_decision) -> str:
    if provider_result is None:
        return "manual-review"
    if provider_result.success:
        return "scheduled-discovery"
    if fallback_decision.quota_exhausted and fallback_decision.fallback_allowed:
        return "ollama-low-risk-fallback"
    if provider_result.error_type == ProviderErrorType.EXTERNAL_REQUIRED:
        return "external-required"
    return "manual-review"
