from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ai_team.core.github_gate import evaluate_github_action
from ai_team.core.git_policy import inspect_candidate_files
from ai_team.core.project_loader import LoadedProject
from ai_team.providers.base import redact_secrets


SAFE_BRANCH_RE = re.compile(r"[^A-Za-z0-9._/-]+")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
SECRET_SCAN_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"(?i)Bearer\s+[A-Za-z0-9_\-.]+"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password|hash[_-]?key|hash[_-]?iv)\s*[:=]\s*([^\s,;]+)"),
]


class CommandRunner(Protocol):
    def __call__(
        self,
        args: list[str],
        cwd: Path,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        ...


@dataclass(frozen=True)
class GitHubExecutionOptions:
    action: str
    dry_run: bool = True
    validation_log_hash: str | None = None
    receipt_path: Path | None = None
    test_evidence_hash: str | None = None
    title: str | None = None
    body: str | None = None
    base_branch: str | None = None
    branch_name: str | None = None
    pr_identifier: str | None = None
    merge_method: str = "squash"
    delete_branch: bool = False


@dataclass(frozen=True)
class GitHubExecutionResult:
    action: str
    dry_run: bool
    attempted: bool
    success: bool
    decision: dict[str, Any]
    branch: str | None = None
    pr_url: str | None = None
    validation_log_hash: str | None = None
    receipt_hash: str | None = None
    secret_scan_hash: str | None = None
    test_evidence_hash: str | None = None
    commands: list[dict[str, Any]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return redact_secrets(
            {
                "action": self.action,
                "dryRun": self.dry_run,
                "attempted": self.attempted,
                "success": self.success,
                "decision": self.decision,
                "branch": self.branch,
                "prUrl": self.pr_url,
                "validationLogHash": self.validation_log_hash,
                "receiptHash": self.receipt_hash,
                "secretScanHash": self.secret_scan_hash,
                "testEvidenceHash": self.test_evidence_hash,
                "commands": self.commands,
                "reasons": self.reasons,
            }
        )


def execute_github_action(
    loaded_project: LoadedProject,
    options: GitHubExecutionOptions,
    runner: CommandRunner | None = None,
) -> GitHubExecutionResult:
    action = _normalize_action(options.action)
    receipt_hash = _hash_file(options.receipt_path) if options.receipt_path else None
    try:
        changed_files = _changed_files_for_head(loaded_project.root)
        secret_scan = scan_commit_for_secrets(loaded_project.root, changed_files)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        scan_payload = {
            "changedFiles": [],
            "blocked": True,
            "reasons": [f"committed secret scan failed: {exc}"],
        }
        secret_scan = {
            **scan_payload,
            "hash": hashlib.sha256(json.dumps(scan_payload, sort_keys=True).encode("utf-8")).hexdigest(),
        }
    validation_hash = options.validation_log_hash
    test_hash = options.test_evidence_hash
    preflight_reasons = _preflight_reasons(action, validation_hash, receipt_hash, secret_scan["hash"], test_hash)
    preflight_reasons.extend(_receipt_reasons(loaded_project, options.receipt_path))
    decision = evaluate_github_action(
        loaded_project,
        action,
        dry_run=options.dry_run,
        validation_log_hash=validation_hash,
        receipt_hash=receipt_hash,
        secret_scan_hash=secret_scan["hash"],
        test_evidence_hash=test_hash,
    )
    if secret_scan["blocked"]:
        preflight_reasons.extend(secret_scan["reasons"])
    if preflight_reasons:
        return GitHubExecutionResult(
            action=action,
            dry_run=options.dry_run,
            attempted=False,
            success=False,
            decision=decision.as_dict(),
            validation_log_hash=validation_hash,
            receipt_hash=receipt_hash,
            secret_scan_hash=secret_scan["hash"],
            test_evidence_hash=test_hash,
            reasons=preflight_reasons,
        )
    if not decision.allowed:
        return GitHubExecutionResult(
            action=action,
            dry_run=options.dry_run,
            attempted=False,
            success=False,
            decision=decision.as_dict(),
            validation_log_hash=validation_hash,
            receipt_hash=receipt_hash,
            secret_scan_hash=secret_scan["hash"],
            test_evidence_hash=test_hash,
            reasons=decision.reasons,
        )
    branch = None
    if action in {"push", "pr"}:
        branch = (
            sanitize_branch_name(options.branch_name)
            if options.dry_run and options.branch_name
            else (loaded_project.current_branch if options.dry_run else ensure_push_branch(loaded_project, options.branch_name, runner=runner))
        )
        if options.dry_run and not branch:
            branch = default_branch_name(loaded_project)
    else:
        branch = loaded_project.current_branch
    runner = runner or _run_command
    if options.dry_run and action == "merge":
        merge_target = options.pr_identifier or branch
        if not merge_target:
            return GitHubExecutionResult(
                action=action,
                dry_run=True,
                attempted=False,
                success=False,
                decision=decision.as_dict(),
                branch=branch,
                validation_log_hash=validation_hash,
                receipt_hash=receipt_hash,
                secret_scan_hash=secret_scan["hash"],
                test_evidence_hash=test_hash,
                reasons=["merge dry-run requires a PR identifier or current branch"],
            )
        view_result = runner(
            [
                "gh",
                "pr",
                "view",
                merge_target,
                "--json",
                "mergeStateStatus,reviewDecision,isDraft,baseRefName,headRefName",
            ],
            loaded_project.root,
            60,
        )
        commands = [_command_dict(view_result)]
        reasons = _merge_gate_reasons(view_result)
        return GitHubExecutionResult(
            action=action,
            dry_run=True,
            attempted=False,
            success=not reasons,
            decision=decision.as_dict(),
            branch=branch,
            validation_log_hash=validation_hash,
            receipt_hash=receipt_hash,
            secret_scan_hash=secret_scan["hash"],
            test_evidence_hash=test_hash,
            commands=commands,
            reasons=reasons,
        )
    if options.dry_run:
        return GitHubExecutionResult(
            action=action,
            dry_run=True,
            attempted=False,
            success=True,
            decision=decision.as_dict(),
            branch=branch,
            validation_log_hash=validation_hash,
            receipt_hash=receipt_hash,
            secret_scan_hash=secret_scan["hash"],
            test_evidence_hash=test_hash,
        )

    commands: list[dict[str, Any]] = []
    if action == "push":
        commands.append(_command_dict(runner(["git", "push", "-u", "origin", f"HEAD:{branch}"], loaded_project.root, 60)))
        return _result_from_commands(action, options, decision.as_dict(), branch, None, commands, validation_hash, receipt_hash, secret_scan["hash"], test_hash)

    if action == "pr":
        commands.append(_command_dict(runner(["git", "push", "-u", "origin", f"HEAD:{branch}"], loaded_project.root, 60)))
        pr_args = [
            "gh",
            "pr",
            "create",
            "--title",
            options.title or f"AI Team automated update: {branch}",
            "--body",
            options.body or _default_pr_body(loaded_project, validation_hash, receipt_hash, secret_scan["hash"], test_hash),
            "--head",
            branch,
        ]
        if options.base_branch:
            pr_args.extend(["--base", options.base_branch])
        pr_result = runner(pr_args, loaded_project.root, 60)
        commands.append(_command_dict(pr_result))
        return _result_from_commands(
            action,
            options,
            decision.as_dict(),
            branch,
            _extract_pr_url(pr_result.stdout),
            commands,
            validation_hash,
            receipt_hash,
            secret_scan["hash"],
            test_hash,
        )

    merge_target = options.pr_identifier or branch
    if not merge_target:
        return GitHubExecutionResult(
            action=action,
            dry_run=False,
            attempted=False,
            success=False,
            decision=decision.as_dict(),
            validation_log_hash=validation_hash,
            receipt_hash=receipt_hash,
            secret_scan_hash=secret_scan["hash"],
            test_evidence_hash=test_hash,
            reasons=["merge requires pr_identifier or current branch"],
        )
    view_result = runner(
        [
            "gh",
            "pr",
            "view",
            merge_target,
            "--json",
            "mergeStateStatus,reviewDecision,isDraft,baseRefName,headRefName",
        ],
        loaded_project.root,
        60,
    )
    commands.append(_command_dict(view_result))
    merge_gate_reasons = _merge_gate_reasons(view_result)
    if merge_gate_reasons:
        return GitHubExecutionResult(
            action=action,
            dry_run=False,
            attempted=True,
            success=False,
            decision=decision.as_dict(),
            branch=branch,
            validation_log_hash=validation_hash,
            receipt_hash=receipt_hash,
            secret_scan_hash=secret_scan["hash"],
            test_evidence_hash=test_hash,
            commands=commands,
            reasons=merge_gate_reasons,
        )
    merge_args = ["gh", "pr", "merge", merge_target, f"--{options.merge_method}"]
    if options.delete_branch:
        merge_args.append("--delete-branch")
    commands.append(_command_dict(runner(merge_args, loaded_project.root, 120)))
    return _result_from_commands(action, options, decision.as_dict(), branch, None, commands, validation_hash, receipt_hash, secret_scan["hash"], test_hash)


def ensure_push_branch(
    loaded_project: LoadedProject,
    requested_branch: str | None = None,
    runner: CommandRunner | None = None,
) -> str:
    branch = sanitize_branch_name(requested_branch or default_branch_name(loaded_project))
    current = loaded_project.current_branch
    if current == branch:
        return branch
    runner = runner or _run_command
    runner(["git", "switch", "-c", branch], loaded_project.root, 30)
    loaded_project.current_branch = branch
    return branch


def default_branch_name(loaded_project: LoadedProject) -> str:
    short_sha = (loaded_project.commit_sha or "unknown")[:8]
    return sanitize_branch_name(f"ai-team/{loaded_project.profile.project.name}-{short_sha}")


def sanitize_branch_name(value: str) -> str:
    cleaned = SAFE_BRANCH_RE.sub("-", value.strip()).strip("/.-")
    if not cleaned:
        cleaned = "ai-team/update"
    if cleaned in {"main", "master"}:
        cleaned = f"ai-team/{cleaned}"
    return cleaned[:180]


def scan_commit_for_secrets(project_root: Path, changed_files: list[str]) -> dict[str, Any]:
    file_check = inspect_candidate_files(project_root, changed_files)
    reasons = list(file_check["reasons"])
    for relative in changed_files:
        blob = _read_head_blob(project_root, relative)
        if blob is None:
            continue
        for pattern in SECRET_SCAN_PATTERNS:
            if pattern.search(blob.decode("utf-8", errors="replace")[:1_000_000]):
                reasons.append(f"secret-like content in changed file: {relative}")
                break
    payload = {"changedFiles": changed_files, "blocked": bool(reasons), "reasons": reasons}
    return {
        **payload,
        "hash": hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest(),
    }


def _changed_files_for_head(project_root: Path) -> list[str]:
    result = _run_command(
        ["git", "diff-tree", "--root", "--no-commit-id", "--name-only", "-r", "HEAD"],
        project_root,
        30,
    )
    if result.returncode != 0:
        raise RuntimeError("unable to enumerate files in HEAD for secret scan")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _read_head_blob(project_root: Path, relative: str) -> bytes | None:
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=project_root,
        check=False,
        capture_output=True,
        timeout=30,
    )
    if result.returncode == 0:
        return result.stdout
    if not (project_root / relative).exists():
        return None
    raise RuntimeError(f"unable to read committed blob for secret scan: {relative}")


def _normalize_action(value: str) -> str:
    normalized = value.lower().strip()
    return "pr" if normalized == "pull-request" else normalized


def _preflight_reasons(
    action: str,
    validation_log_hash: str | None,
    receipt_hash: str | None,
    secret_scan_hash: str | None,
    test_evidence_hash: str | None,
) -> list[str]:
    reasons: list[str] = []
    if action in {"pr", "merge"} and not _is_sha256(validation_log_hash):
        reasons.append("valid SHA-256 validation log hash is required")
    if action in {"push", "pr", "merge"} and not receipt_hash:
        reasons.append("receipt hash is required")
    if action in {"push", "pr", "merge"} and not secret_scan_hash:
        reasons.append("secret scan hash is required")
    if action in {"pr", "merge"} and not _is_sha256(test_evidence_hash):
        reasons.append(f"{action} requires valid SHA-256 test evidence hash")
    return reasons


def _is_sha256(value: str | None) -> bool:
    return bool(value and SHA256_RE.fullmatch(value))


def _receipt_reasons(loaded_project: LoadedProject, receipt_path: Path | None) -> list[str]:
    if receipt_path is None or not receipt_path.is_file():
        return ["run receipt is missing"]
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ["run receipt is unreadable or invalid JSON"]
    if not isinstance(payload, dict):
        return ["run receipt must be a JSON object"]

    reasons: list[str] = []
    validation = payload.get("validationResult")
    if not isinstance(validation, dict) or validation.get("success") is not True:
        reasons.append("run receipt does not contain a successful validation result")
    if payload.get("projectPath") != str(loaded_project.root):
        reasons.append("run receipt project path does not match the disposable worktree")
    if payload.get("commitSha") != loaded_project.commit_sha:
        reasons.append("run receipt commit SHA does not match HEAD")

    source_sha = payload.get("sourceCommitSha")
    if source_sha and loaded_project.commit_sha:
        parent = _run_command(["git", "rev-parse", "HEAD^"], loaded_project.root, 30)
        if parent.returncode != 0 or parent.stdout.strip() != source_sha:
            reasons.append("run receipt source commit is not the direct parent of HEAD")
    return reasons


def _hash_file(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result_from_commands(
    action: str,
    options: GitHubExecutionOptions,
    decision: dict[str, Any],
    branch: str | None,
    pr_url: str | None,
    commands: list[dict[str, Any]],
    validation_hash: str | None,
    receipt_hash: str | None,
    secret_scan_hash: str | None,
    test_hash: str | None,
) -> GitHubExecutionResult:
    success = all(command["returnCode"] == 0 for command in commands)
    reasons = [] if success else ["one or more GitHub executor commands failed"]
    return GitHubExecutionResult(
        action=action,
        dry_run=options.dry_run,
        attempted=True,
        success=success,
        decision=decision,
        branch=branch,
        pr_url=pr_url,
        validation_log_hash=validation_hash,
        receipt_hash=receipt_hash,
        secret_scan_hash=secret_scan_hash,
        test_evidence_hash=test_hash,
        commands=commands,
        reasons=reasons,
    )


def _command_dict(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return redact_secrets(
        {
            "args": result.args,
            "returnCode": result.returncode,
            "stdout": result.stdout[:4000],
            "stderr": result.stderr[:4000],
        }
    )


def _run_command(args: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    if args[0] == "gh" and shutil.which("gh") is None:
        raise FileNotFoundError("GitHub CLI is not installed")
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


def _default_pr_body(
    loaded_project: LoadedProject,
    validation_hash: str | None,
    receipt_hash: str | None,
    secret_scan_hash: str | None,
    test_hash: str | None,
) -> str:
    return "\n".join(
        [
            "Automated AI Team update.",
            "",
            f"Project: {loaded_project.profile.project.name}",
            f"Source commit: {loaded_project.commit_sha}",
            f"Validation log hash: {validation_hash}",
            f"Receipt hash: {receipt_hash}",
            f"Secret scan hash: {secret_scan_hash}",
            f"Test evidence hash: {test_hash}",
        ]
    )


def _extract_pr_url(stdout: str) -> str | None:
    for token in stdout.split():
        if token.startswith("https://") and "/pull/" in token:
            return token.strip()
    return None


def _merge_gate_reasons(result: subprocess.CompletedProcess[str]) -> list[str]:
    if result.returncode != 0:
        return ["gh pr view failed before merge"]
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return ["gh pr view did not return valid JSON"]
    reasons: list[str] = []
    if payload.get("isDraft") is True:
        reasons.append("merge blocked because PR is draft")
    if payload.get("reviewDecision") != "APPROVED":
        reasons.append("merge requires approved review decision")
    if payload.get("mergeStateStatus") != "CLEAN":
        reasons.append("merge requires clean branch protection status")
    return reasons
