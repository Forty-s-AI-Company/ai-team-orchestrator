from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_team.core.isolated_executor import run_in_disposable_worktree
from ai_team.providers.base import BaseProvider, ProviderErrorType, redact_secrets


@dataclass(frozen=True)
class TrustedTask:
    id: str
    title: str
    priority: int
    source: str
    risk: str
    instruction: str
    allowed_write_paths: list[str]
    validation_commands: list[str]
    auto_executable: bool


@dataclass(frozen=True)
class DeliveryOptions:
    project_path: Path
    provider: BaseProvider
    workspace_allowlist: list[str] | None
    report_dir: Path
    state_path: Path
    queue_path: Path
    once: bool = False
    interval_minutes: int = 60
    max_runtime_minutes: int | None = None
    github_execute: bool = False


def discover_trusted_tasks(project: Path) -> list[TrustedTask]:
    tasks: list[TrustedTask] = []
    eslint_config = project / "eslint.config.mjs"
    eslint_text = eslint_config.read_text(encoding="utf-8") if eslint_config.exists() else ""
    if (project / "coverage").exists() and "coverage/**" not in eslint_text:
        tasks.append(
            TrustedTask(
                id="lint-ignore-coverage",
                title="Exclude generated coverage artifacts from ESLint",
                priority=10,
                source="test-qa-discovery",
                risk="low",
                instruction=(
                    "Fix the ESLint configuration so generated coverage artifacts are ignored. "
                    "Modify only eslint.config.mjs. Preserve all existing ignores and behavior. "
                    "Do not change product code, dependencies, tests, payment, database, or deployment settings."
                ),
                allowed_write_paths=["eslint.config.mjs"],
                validation_commands=["npm run lint", "npm run typecheck", "npm run test"],
                auto_executable=True,
            )
        )

    todo = _run(["git", "grep", "-n", "-E", "TODO|FIXME|HACK", "--", "src", "tests"], project, timeout=30)
    if todo.returncode == 0 and todo.stdout.strip():
        tasks.append(
            TrustedTask(
                id="triage-source-todos",
                title="Triage source TODO and FIXME findings",
                priority=70,
                source="todo-discovery",
                risk="review",
                instruction="Triage source TODO/FIXME findings and propose bounded tasks; do not modify files.",
                allowed_write_paths=[],
                validation_commands=[],
                auto_executable=False,
            )
        )
    return sorted(tasks, key=lambda item: (item.priority, item.id))


def collect_discovery_evidence(project: Path) -> dict[str, Any]:
    history = _run(["git", "log", "-10", "--oneline"], project, timeout=30)
    diff_check = _run(["git", "diff", "--check"], project, timeout=30)
    readiness_files = [
        name for name in (
            "docs/production-readiness-review.md",
            "docs/production-go-live-checklist.md",
            "docs/live-commerce-mvp-report.md",
        )
        if (project / name).exists()
    ]
    return {
        "gitHistoryReview": {"ok": history.returncode == 0, "recentCommits": history.stdout.splitlines()[:10]},
        "codeReview": {"ok": diff_check.returncode == 0, "diffCheck": diff_check.stderr or diff_check.stdout},
        "testQaDiscovery": {
            "mode": "static-only-on-primary-worktree",
            "note": "Project scripts execute only inside the disposable worktree after trusted promotion.",
        },
        "productionReadinessReview": {"files": readiness_files, "externalItemsRemain": bool(readiness_files)},
    }


def run_delivery_supervisor(options: DeliveryOptions) -> dict[str, Any]:
    lock_path = options.state_path.with_suffix(options.state_path.suffix + ".lock")
    _acquire_lock(lock_path)
    try:
        started = time.monotonic()
        cycles = 0
        last: dict[str, Any] = {}
        while True:
            cycles += 1
            last = run_delivery_cycle(options, cycles)
            if options.once:
                break
            if options.max_runtime_minutes is not None and time.monotonic() - started >= options.max_runtime_minutes * 60:
                break
            if last.get("status") != "completed":
                time.sleep(max(1, options.interval_minutes) * 60)
        return {"cycles": cycles, "last": last}
    finally:
        lock_path.unlink(missing_ok=True)


def run_delivery_cycle(options: DeliveryOptions, cycle_number: int) -> dict[str, Any]:
    prior = _read_json(options.state_path)
    evidence = collect_discovery_evidence(options.project_path)
    tasks = discover_trusted_tasks(options.project_path)
    completed_ids = set(prior.get("completedTaskIds", []))
    queue = [task for task in tasks if task.id not in completed_ids]
    _write_json(
        options.queue_path,
        {"schemaVersion": 1, "discoveryEvidence": evidence, "tasks": [asdict(task) for task in queue]},
    )
    selected = next((task for task in queue if task.auto_executable), None)
    state: dict[str, Any] = {
        "schemaVersion": 1,
        "revision": int(prior.get("revision", 0)) + 1,
        "cycleNumber": cycle_number,
        "updatedAt": datetime.now(UTC).isoformat(),
        "queuePath": str(options.queue_path),
        "completedTaskIds": sorted(completed_ids),
        "discoveryEvidence": evidence,
    }
    if selected is None:
        state.update({"status": "idle", "nextAction": "scheduled-discovery", "currentTask": None})
        _write_json(options.state_path, state)
        return state

    state.update({"status": "executing", "currentTask": asdict(selected), "stage": "isolated-write"})
    _write_json(options.state_path, state)
    isolated = run_in_disposable_worktree(
        source_project_path=options.project_path,
        provider=options.provider,
        workflow_name="bug-fix-loop",
        workspace_allowlist=options.workspace_allowlist,
        receipt_dir=options.report_dir / "receipts",
        worktree_parent=options.project_path.parent,
        dry_run=False,
        run_mode="create-only",
        keep_worktree=True,
        auto_commit=True,
        commit_message=f"fix: {selected.title.lower()}",
        github_action="pr",
        github_execute=options.github_execute,
        github_branch=f"ai-team/{selected.id}",
        task_instruction=selected.instruction,
        allowed_write_paths=selected.allowed_write_paths,
        validation_commands=selected.validation_commands,
        require_validation=True,
    )
    result = isolated.workflow_result.provider_result
    github_ok = isolated.github_result is None or isolated.github_result.get("success") is True
    success = result.success and isolated.commit_result.get("committed") is True and github_ok
    if success:
        completed_ids.add(selected.id)
    quota = result.error_type == ProviderErrorType.RATE_LIMIT
    state.update(
        {
            "status": "completed" if success else ("waiting-quota" if quota else "attention-required"),
            "stage": "complete" if success else "paused",
            "completedTaskIds": sorted(completed_ids),
            "worktreePath": str(isolated.worktree_path),
            "runReceipt": str(isolated.run_receipt),
            "executorReceipt": str(isolated.executor_receipt),
            "provider": result.provider,
            "providerSuccess": result.success,
            "githubResult": isolated.github_result,
            "commitResult": isolated.commit_result,
            "nextAction": "scheduled-discovery" if success else ("resume-after-quota-reset" if quota else "review-failure"),
        }
    )
    _write_json(options.state_path, state)
    return state


def _run(args: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(args, cwd=cwd, check=False, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(args, 127, "", str(exc))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


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
        raise RuntimeError(f"delivery supervisor already running: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "createdAt": datetime.now(UTC).isoformat()}, handle)
