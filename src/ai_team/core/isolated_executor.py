from __future__ import annotations

import json
import hashlib
import os
import shlex
import shutil
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
    task_instruction: str | None = None,
    allowed_write_paths: list[str] | None = None,
    validation_commands: list[str] | None = None,
    require_validation: bool = False,
    reuse_worktree_path: Path | None = None,
) -> IsolatedRunResult:
    source = load_project(source_project_path, allowlist=workspace_allowlist)
    workflow = load_workflow(workflow_name)
    if not workflow.write_required:
        raise ValueError("isolated executor is reserved for write-capable workflows")
    if source.is_disposable_worktree():
        raise ValueError("source project is already a disposable worktree; use ai-team run directly")

    reused_worktree = reuse_worktree_path is not None
    if reused_worktree:
        worktree_path = reuse_worktree_path.resolve()
        parent = (worktree_parent or source.project_dir.parent).resolve()
        try:
            worktree_path.relative_to(parent)
        except ValueError as exc:
            raise ValueError("reused worktree escapes the approved disposable worktree parent") from exc
        if worktree_path == source.root or not (worktree_path / ".git").is_file():
            raise ValueError("reused worktree must be an existing disposable linked worktree")
        if _git_common_dir(worktree_path) != _git_common_dir(source.root):
            raise ValueError("reused worktree does not belong to the source repository")
    else:
        worktree_path = create_disposable_worktree(source, worktree_parent=worktree_parent)
    executor_receipt: Path | None = None
    try:
        loaded = load_project(worktree_path, allowlist=workspace_allowlist)
        if loaded.is_branch_protected():
            raise ValueError("disposable worktree resolved to a protected branch")

        dependency_link = prepare_dependency_link(loaded.root, source.root)
        try:
            # The write provider must see the same controlled dependency tree
            # used by deterministic validation. This is also what makes local
            # framework documentation available before a Next.js edit.
            result = Orchestrator(provider=provider, max_retries=2).run(
                loaded,
                workflow_name=workflow_name,
                dry_run=dry_run,
                run_mode=run_mode,
                task_instruction=task_instruction,
            )
            changed_files = list_changed_files(loaded.root)
            file_check = inspect_candidate_files(loaded.root, changed_files)
            scope_check = inspect_write_scope(changed_files, allowed_write_paths)
            git_policy = evaluate_git_action(loaded, "commit", candidate_files=changed_files).as_dict()
            git_policy["changedFiles"] = changed_files
            git_policy["fileCheck"] = file_check
            git_policy["scopeCheck"] = scope_check
            if not scope_check["allowed"]:
                git_policy["allowed"] = False
                git_policy.setdefault("reasons", []).extend(scope_check["reasons"])
            validation_result = run_validation_commands(
                loaded.root,
                validation_commands or [],
                require_nonempty=require_validation,
                dependency_root=(
                    loaded.root
                    if (loaded.root / "node_modules").exists()
                    else source.root
                ),
            )
        finally:
            remove_dependency_link(dependency_link)
        effective_validation_hash = validation_log_hash or validation_result["hash"]
        effective_test_hash = test_evidence_hash or validation_result["hash"]
        try:
            commit_result = maybe_commit_changed_files(
                loaded,
                changed_files=changed_files,
                git_policy=git_policy,
                enabled=(
                    auto_commit
                    and result.provider_result.success
                    and bool(changed_files)
                    and validation_result["success"]
                ),
                commit_message=commit_message or default_commit_message(workflow_name),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            # Fail closed even if a future Git call in the commit path was not
            # converted to an explicit operation result yet.
            commit_result = _git_failure_result(
                stop_reason="git-subprocess-failed",
                operation="unknown",
                error=exc,
            )
        # Bounded delivery consumes this attested result rather than inferring
        # validation success from a commit alone.
        commit_result = {**commit_result, "validation": validation_result}
        if auto_commit and (
            not result.provider_result.success or not changed_files or not validation_result["success"]
        ):
            commit_result = {
                "attempted": False,
                "committed": False,
                "reason": "provider validation failed, produced no diff, or validation command failed; commit denied",
                "validation": validation_result,
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
            validation_log_hash=effective_validation_hash,
            test_evidence_hash=effective_test_hash,
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
            validation_result=validation_result,
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
        if not keep_worktree and not reused_worktree and worktree_path.exists():
            remove_worktree(source.root, worktree_path)


def create_disposable_worktree(source: LoadedProject, worktree_parent: Path | None = None) -> Path:
    parent = (worktree_parent or source.project_dir.parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / f"{source.profile.project.name}-ai-write-{uuid4().hex[:8]}"
    _run_git(source.root, ["worktree", "add", "--detach", str(target), source.commit_sha or "HEAD"])
    return target.resolve()


def _git_common_dir(project_root: Path) -> Path:
    output = _run_git(project_root, ["rev-parse", "--git-common-dir"]).stdout.strip()
    path = Path(output)
    return (project_root / path).resolve() if not path.is_absolute() else path.resolve()


def remove_worktree(repository_root: Path, worktree_path: Path) -> None:
    _run_git(repository_root, ["worktree", "remove", "--force", str(worktree_path)])


def list_changed_files(project_root: Path) -> list[str]:
    result = _run_git(project_root, ["status", "--porcelain", "--untracked-files=all"])
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


def inspect_write_scope(changed_files: list[str], allowed_write_paths: list[str] | None) -> dict[str, Any]:
    if not allowed_write_paths:
        return {"allowed": True, "allowedWritePaths": [], "reasons": []}
    normalized = {Path(item).as_posix().rstrip("/") for item in allowed_write_paths}
    outside = [
        item for item in changed_files
        if not any(Path(item).as_posix() == root or Path(item).as_posix().startswith(f"{root}/") for root in normalized)
    ]
    return {
        "allowed": not outside,
        "allowedWritePaths": sorted(normalized),
        "reasons": [f"changed file is outside trusted task scope: {item}" for item in outside],
    }


def prepare_dependency_link(worktree_root: Path, source_root: Path) -> Path | None:
    source_modules = source_root / "node_modules"
    target_modules = worktree_root / "node_modules"
    if not source_modules.is_dir():
        return None
    if target_modules.is_symlink():
        raise ValueError("disposable worktree dependency path must not be a pre-existing symlink")
    if target_modules.exists() and not target_modules.is_dir():
        raise ValueError("disposable worktree dependency path must be a directory")
    if target_modules.is_dir():
        # A provider self-check can create an incomplete cache-only node_modules
        # directory. Merge the trusted source dependencies into a private tree
        # so its mere existence cannot suppress dependency preparation.
        shutil.copytree(source_modules, target_modules, symlinks=True, dirs_exist_ok=True)
        return target_modules
    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(target_modules), str(source_modules)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return target_modules if result.returncode == 0 and target_modules.exists() else None
    if _requires_local_dependency_tree(worktree_root):
        try:
            # Next.js Turbopack rejects a top-level node_modules symlink that
            # resolves outside the project root. A private validation copy also
            # prevents generators such as Prisma from mutating primary deps.
            shutil.copytree(source_modules, target_modules, symlinks=True)
        except Exception:
            if target_modules.exists():
                shutil.rmtree(target_modules)
            raise
        return target_modules
    target_modules.symlink_to(source_modules, target_is_directory=True)
    return target_modules


def _requires_local_dependency_tree(project_root: Path) -> bool:
    manifest = project_root / "package.json"
    if not manifest.is_file() or manifest.is_symlink():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    dependencies = {
        **(payload.get("dependencies") if isinstance(payload.get("dependencies"), dict) else {}),
        **(payload.get("devDependencies") if isinstance(payload.get("devDependencies"), dict) else {}),
    }
    return "next" in dependencies


def remove_dependency_link(link: Path | None) -> None:
    if link is None or not link.exists():
        return
    if link.is_symlink():
        link.unlink()
    else:
        # The path is returned only when this module created it. On POSIX this
        # may be a private dependency copy; on Windows it may be a junction.
        shutil.rmtree(link)


def run_validation_commands(
    project_root: Path,
    commands: list[str],
    *,
    require_nonempty: bool = False,
    dependency_root: Path | None = None,
) -> dict[str, Any]:
    if require_nonempty and not commands:
        payload: dict[str, Any] = {
            "success": False,
            "commands": [],
            "reason": "trusted write tasks require at least one validation command",
        }
        payload["hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return payload
    if not commands:
        payload = {"success": True, "commands": [], "legacyNoValidation": True}
        payload["hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return payload
    results: list[dict[str, Any]] = []
    for command in commands:
        args = shlex.split(command, posix=False)
        executable = shutil.which(args[0])
        if executable:
            args[0] = executable
        try:
            completed = subprocess.run(
                args,
                cwd=project_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=900,
                env=_validation_env(dependency_root),
            )
            result = {
                "command": command,
                "returnCode": completed.returncode,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            }
        except FileNotFoundError as exc:
            result = {
                "command": command,
                "returnCode": 127,
                "stdout": "",
                "stderr": str(exc),
            }
        except subprocess.TimeoutExpired as exc:
            result = {
                "command": command,
                "returnCode": 124,
                "stdout": str(exc.stdout or "")[-4000:],
                "stderr": str(exc.stderr or exc)[-4000:],
            }
        results.append(result)
        if result["returnCode"] != 0:
            break
    payload = {"success": all(item["returnCode"] == 0 for item in results), "commands": results}
    if any(item["returnCode"] == 127 for item in results):
        payload.update({
            "kind": "execution-environment",
            "stopReason": "validation-command-unavailable",
        })
    elif any(item["returnCode"] == 124 for item in results):
        payload.update({
            "kind": "execution-environment",
            "stopReason": "validation-command-timeout",
        })
    payload["hash"] = hashlib.sha256(
        json.dumps(redact_secrets(payload), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return redact_secrets(payload)


def _validation_env(dependency_root: Path | None) -> dict[str, str]:
    allowed = {
        "APPDATA", "COMSPEC", "HOME", "LOCALAPPDATA", "PATHEXT", "PATH",
        "PROGRAMDATA", "SYSTEMDRIVE", "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE", "WINDIR",
    }
    env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    if dependency_root is not None:
        dependency_bin = dependency_root / "node_modules" / ".bin"
        env["PATH"] = f"{dependency_bin}{os.pathsep}{env.get('PATH', '')}"
    env["CI"] = "1"
    env["NODE_ENV"] = "test"
    return env


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
            "validationResult": {
                "success": False,
                "kind": "policy-validation",
                "stopReason": "git-policy-denied",
            },
        }

    identity_failure = _validate_git_identity(loaded_project.root)
    if identity_failure is not None:
        return identity_failure

    try:
        add_result = _run_git_attempt(loaded_project.root, ["add", "--", *changed_files])
    except (OSError, subprocess.SubprocessError) as exc:
        return _git_failure_result("git-add-failed", "add", error=exc)
    if add_result.returncode != 0:
        return _git_failure_result("git-add-failed", "add", completed=add_result)

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
            "validationResult": {
                "success": False,
                "kind": "policy-validation",
                "stopReason": "staged-git-policy-denied",
            },
        }

    try:
        commit = _run_git_attempt(loaded_project.root, ["commit", "-m", commit_message])
    except (OSError, subprocess.SubprocessError) as exc:
        return _git_failure_result("git-commit-failed", "commit", error=exc, staged_files=staged_files)
    if commit.returncode != 0:
        return _git_failure_result(
            "git-commit-failed",
            "commit",
            completed=commit,
            staged_files=staged_files,
        )

    commit_sha = _run_git(loaded_project.root, ["rev-parse", "HEAD"]).stdout.strip()
    loaded_project.commit_sha = commit_sha
    return {
        "attempted": True,
        "committed": True,
        "commitSha": commit_sha,
        "message": commit_message,
        "stagedFiles": staged_files,
        "validationResult": {
            "success": True,
            "kind": "git-commit",
            "stopReason": None,
        },
    }


def _validate_git_identity(project_root: Path) -> dict[str, Any] | None:
    missing: list[str] = []
    for key in ("user.name", "user.email"):
        try:
            result = _run_git_attempt(project_root, ["config", "--get", key])
        except (OSError, subprocess.SubprocessError) as exc:
            return _git_failure_result("git-identity-check-failed", "identity", error=exc)
        if result.returncode != 0 or not result.stdout.strip():
            missing.append(key)
    if missing:
        return _git_failure_result(
            "git-identity-missing",
            "identity",
            missing_fields=missing,
        )
    return None


def _git_failure_result(
    stop_reason: str,
    operation: str,
    *,
    completed: subprocess.CompletedProcess[str] | None = None,
    error: BaseException | None = None,
    missing_fields: list[str] | None = None,
    staged_files: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "attempted": True,
        "committed": False,
        "reason": stop_reason,
        "stopReason": stop_reason,
        "gitOperation": operation,
        "validationResult": {
            "success": False,
            "kind": "git-commit",
            "stopReason": stop_reason,
        },
    }
    if completed is not None:
        payload.update({
            "returnCode": completed.returncode,
            "stdout": _redacted_git_output(completed.stdout),
            "stderr": _redacted_git_output(completed.stderr),
        })
    if error is not None:
        error_stdout = getattr(error, "stdout", None)
        if error_stdout:
            payload["stdout"] = _redacted_git_output(error_stdout)
        payload["stderr"] = _redacted_git_output(getattr(error, "stderr", None) or error)
    if missing_fields:
        payload["missingIdentityFields"] = missing_fields
    if staged_files:
        payload["stagedFiles"] = staged_files
    return redact_secrets(payload)


def _redacted_git_output(value: object, limit: int = 4000) -> str:
    # Redact before truncation so a long value cannot move its identifying key
    # outside the retained tail and bypass pattern-based redaction.
    redacted = redact_secrets("" if value is None else str(value))
    return str(redacted)[-limit:]


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
    validation_result: dict[str, Any],
) -> Path:
    receipt_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).isoformat()
    path = receipt_dir / f"{generated_at.replace(':', '').replace('+', 'Z').replace('.', '')}-isolated-{uuid4().hex[:8]}.json"
    executor_validation = _executor_validation_result(
        provider_success=result.provider_result.success,
        command_validation=validation_result,
        commit_result=commit_result,
    )
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
        "providerSuccess": result.provider_result.success,
        "validationResult": executor_validation,
        "stopReason": executor_validation.get("stopReason"),
        "providerValidationResult": {
            "success": result.provider_result.success,
            "errorType": result.provider_result.error_type,
            "durationMs": result.duration_ms,
        },
        "commandValidationResult": validation_result,
    }
    path.write_text(json.dumps(redact_secrets(payload), indent=2, default=str), encoding="utf-8")
    return path


def _executor_validation_result(
    *,
    provider_success: bool,
    command_validation: dict[str, Any],
    commit_result: dict[str, Any],
) -> dict[str, Any]:
    if not provider_success:
        return {
            "success": False,
            "kind": "provider-execution",
            "stopReason": "provider-native-execution-failed",
        }
    if not command_validation.get("success"):
        return {
            "success": False,
            "kind": "deterministic-validation",
            "stopReason": "deterministic-validation-failed",
        }
    commit_validation = commit_result.get("validationResult")
    if isinstance(commit_validation, dict):
        return dict(commit_validation)
    return {"success": True, "kind": "executor", "stopReason": None}


def _run_git(cwd: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _run_git_attempt(cwd: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
