from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ai_team.core.git_policy import evaluate_git_action, inspect_candidate_files
from ai_team.core.github_executor import GitHubExecutionOptions, execute_github_action
from ai_team.core.orchestrator import Orchestrator, WorkflowRunResult, load_workflow
from ai_team.core.project_loader import LoadedProject, load_project
from ai_team.core.receipts import write_run_receipt
from ai_team.providers.base import BaseProvider, redact_secrets


@dataclass(frozen=True)
class IsolatedRunResult:
    worktree_path: Path
    workflow_result: WorkflowRunResult
    run_receipt: Path
    executor_receipt: Path
    git_policy: dict[str, Any]
    commit_result: dict[str, Any]
    github_result: dict[str, Any] | None = None


def run_in_disposable_worktree(
    source_project_path: str | Path,
    provider: BaseProvider,
    workflow_name: str,
    workspace_allowlist: list[str | Path] | None,
    receipt_dir: Path,
    worktree_parent: Path | None = None,
    dry_run: bool = False,
    run_mode: str = "create-only",
    keep_worktree: bool = True,
    auto_commit: bool = False,
    commit_message: str | None = None,
    github_action: str | None = None,
    github_execute: bool = False,
    github_branch: str | None = None,
    validation_log_hash: str | None = None,
    test_evidence_hash: str | None = None,
) -> IsolatedRunResult:
    source = load_project(source_project_path, allowlist=workspace_allowlist)
    workflow = load_workflow(workflow_name)
    if not workflow.write_required:
        raise ValueError("isolated executor is reserved for write-capable workflows")
    if source.is_disposable_worktree():
        raise ValueError("source project is already a disposable worktree; use ai-team run directly")

    worktree_path = create_disposable_worktree(source, worktree_parent=worktree_parent)
    executor_receipt: Path | None = None
    try:
        loaded = load_project(worktree_path, allowlist=workspace_allowlist)
        if loaded.is_branch_protected():
            raise ValueError("disposable worktree resolved to a protected branch")

        result = Orchestrator(provider=provider, max_retries=2).run(
            loaded,
            workflow_name=workflow_name,
            dry_run=dry_run,
            run_mode=run_mode,
        )
        changed_files = list_changed_files(loaded.root)
        file_check = inspect_candidate_files(loaded.root, changed_files)
        git_policy = evaluate_git_action(loaded, "commit", candidate_files=changed_files).as_dict()
        git_policy["changedFiles"] = changed_files
        git_policy["fileCheck"] = file_check
        commit_result = maybe_commit_changed_files(
            loaded,
            changed_files=changed_files,
            git_policy=git_policy,
            enabled=auto_commit and result.provider_result.success,
            commit_message=commit_message or default_commit_message(workflow_name),
        )
        if auto_commit and not result.provider_result.success:
            commit_result = {
                "attempted": False,
                "committed": False,
                "reason": "provider validation failed; commit denied",
            }
        run_receipt = write_run_receipt(
            loaded,
            result,
            receipt_dir,
            source_commit_sha=source.commit_sha,
        )
        github_result = maybe_execute_github_action(
            loaded,
            action=github_action,
            enabled=bool(github_action and commit_result.get("committed")),
            execute=github_execute,
            branch_name=github_branch,
            run_receipt=run_receipt,
            validation_log_hash=validation_log_hash,
            test_evidence_hash=test_evidence_hash,
        )
        executor_receipt = write_executor_receipt(
            source_project=source,
            worktree_project=loaded,
            result=result,
            run_receipt=run_receipt,
            receipt_dir=receipt_dir,
            git_policy=git_policy,
            commit_result=commit_result,
            github_result=github_result,
            keep_worktree=keep_worktree,
        )
        return IsolatedRunResult(
            worktree_path=worktree_path,
            workflow_result=result,
            run_receipt=run_receipt,
            executor_receipt=executor_receipt,
            git_policy=git_policy,
            commit_result=commit_result,
            github_result=github_result,
        )
    finally:
        if not keep_worktree and worktree_path.exists():
            remove_worktree(source.root, worktree_path)


def create_disposable_worktree(source: LoadedProject, worktree_parent: Path | None = None) -> Path:
    parent = (worktree_parent or source.project_dir.parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / f"{source.profile.project.name}-ai-write-{uuid4().hex[:8]}"
    _run_git(source.root, ["worktree", "add", "--detach", str(target), source.commit_sha or "HEAD"])
    return target.resolve()


def remove_worktree(repository_root: Path, worktree_path: Path) -> None:
    _run_git(repository_root, ["worktree", "remove", "--force", str(worktree_path)])


def list_changed_files(project_root: Path) -> list[str]:
    result = _run_git(project_root, ["status", "--porcelain"])
    files: list[str] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1].strip()
        if path:
            files.append(path)
    return files


def maybe_commit_changed_files(
    loaded_project: LoadedProject,
    changed_files: list[str],
    git_policy: dict[str, Any],
    enabled: bool,
    commit_message: str,
) -> dict[str, Any]:
    if not enabled:
        return {"attempted": False, "committed": False, "reason": "auto commit disabled"}
    if not changed_files:
        return {"attempted": True, "committed": False, "reason": "no changed files"}
    if not git_policy.get("allowed"):
        return {
            "attempted": True,
            "committed": False,
            "reason": "git policy denied commit",
            "policyReasons": git_policy.get("reasons", []),
        }

    _run_git(loaded_project.root, ["add", "--", *changed_files])
    staged_files = list_staged_files(loaded_project.root)
    staged_policy = evaluate_git_action(loaded_project, "commit", candidate_files=staged_files).as_dict()
    if not staged_policy.get("allowed"):
        _run_git(loaded_project.root, ["reset", "--", *staged_files])
        return {
            "attempted": True,
            "committed": False,
            "reason": "staged policy denied commit",
            "policyReasons": staged_policy.get("reasons", []),
            "stagedFiles": staged_files,
        }

    _run_git(loaded_project.root, ["commit", "-m", commit_message])
    commit_sha = _run_git(loaded_project.root, ["rev-parse", "HEAD"]).stdout.strip()
    loaded_project.commit_sha = commit_sha
    return {
        "attempted": True,
        "committed": True,
        "commitSha": commit_sha,
        "message": commit_message,
        "stagedFiles": staged_files,
    }


def list_staged_files(project_root: Path) -> list[str]:
    result = _run_git(project_root, ["diff", "--cached", "--name-only"])
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def default_commit_message(workflow_name: str) -> str:
    safe_workflow = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in workflow_name)
    return f"chore(ai-team): apply {safe_workflow}"


def maybe_execute_github_action(
    loaded_project: LoadedProject,
    action: str | None,
    enabled: bool,
    execute: bool,
    branch_name: str | None,
    run_receipt: Path,
    validation_log_hash: str | None,
    test_evidence_hash: str | None,
) -> dict[str, Any] | None:
    if not action:
        return None
    if not enabled:
        return {
            "attempted": False,
            "success": False,
            "reason": "GitHub action requires a committed isolated change",
            "action": action,
        }
    result = execute_github_action(
        loaded_project,
        GitHubExecutionOptions(
            action=action,
            dry_run=not execute,
            branch_name=branch_name,
            validation_log_hash=validation_log_hash,
            receipt_path=run_receipt,
            test_evidence_hash=test_evidence_hash,
        ),
    )
    return result.as_dict()


def write_executor_receipt(
    source_project: LoadedProject,
    worktree_project: LoadedProject,
    result: WorkflowRunResult,
    run_receipt: Path,
    receipt_dir: Path,
    git_policy: dict[str, Any],
    commit_result: dict[str, Any],
    github_result: dict[str, Any] | None,
    keep_worktree: bool,
) -> Path:
    receipt_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).isoformat()
    path = receipt_dir / f"{generated_at.replace(':', '').replace('+', 'Z').replace('.', '')}-isolated-{uuid4().hex[:8]}.json"
    payload = {
        "schemaVersion": 1,
        "generatedAt": generated_at,
        "sourceProjectPath": str(source_project.root),
        "sourceCommitSha": source_project.commit_sha,
        "worktreePath": str(worktree_project.root),
        "worktreeBranch": worktree_project.current_branch,
        "worktreeCommitSha": worktree_project.commit_sha,
        "workflow": result.workflow.name,
        "provider": result.provider_result.provider,
        "runMode": result.provider_result.data.get("runMode"),
        "dryRun": result.dry_run,
        "stages": result.stages,
        "runReceipt": str(run_receipt),
        "gitPolicy": git_policy,
        "commitResult": commit_result,
        "githubResult": github_result,
        "keepWorktree": keep_worktree,
        "validationResult": {
            "success": result.provider_result.success,
            "errorType": result.provider_result.error_type,
            "durationMs": result.duration_ms,
        },
    }
    path.write_text(json.dumps(redact_secrets(payload), indent=2, default=str), encoding="utf-8")
    return path


def _run_git(cwd: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
