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

from ai_team.core.orchestrator import Orchestrator
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
    if loaded is not None:
        try:
            provider_result = Orchestrator(options.provider, max_retries=1).run(
                loaded,
                workflow_name=options.workflow,
                dry_run=options.dry_run,
                run_mode=options.run_mode,
            ).provider_result
            auto_cycle_success = provider_result.success or provider_result.error_type == ProviderErrorType.EXTERNAL_REQUIRED
            stages.append(
                _stage(
                    "auto-cycle",
                    auto_cycle_success,
                    {
                        "provider": provider_result.provider,
                        "success": provider_result.success,
                        "errorType": provider_result.error_type,
                        "externalRequired": provider_result.data.get("externalRequired"),
                    },
                )
            )
        except Exception as exc:
            stages.append(_stage("auto-cycle", False, {"message": str(exc)}))
    else:
        stages.append(_stage("auto-cycle", False, {"message": "project profile unavailable"}))

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
        "stages": stages,
        "status": "completed" if all(stage["ok"] for stage in stages) else "attention_required",
    }

    options.report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = started.isoformat().replace(":", "").replace("+", "Z").replace(".", "")
    path = options.report_dir / f"{timestamp}-cycle-{cycle_number}-{uuid4().hex[:8]}.json"
    path.write_text(json.dumps(redact_secrets(report), indent=2, default=str), encoding="utf-8")
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
        return {"status": "provider-native-ready", "conversationId": provider_result.conversation_id}
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
