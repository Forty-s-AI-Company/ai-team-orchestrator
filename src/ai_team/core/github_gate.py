from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_team.core.git_policy import GitPolicyDecision, evaluate_git_action
from ai_team.core.project_loader import LoadedProject


@dataclass(frozen=True)
class GitHubGateDecision:
    action: str
    allowed: bool
    dry_run: bool
    external_required: bool = False
    reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "allowed": self.allowed,
            "dryRun": self.dry_run,
            "externalRequired": self.external_required,
            "reasons": self.reasons,
            "evidence": self.evidence,
        }


def evaluate_github_action(
    loaded_project: LoadedProject,
    action: str,
    dry_run: bool = True,
    validation_log_hash: str | None = None,
) -> GitHubGateDecision:
    normalized = action.lower().strip()
    if normalized == "pull-request":
        normalized = "pr"

    if normalized not in {"push", "pr", "merge"}:
        return GitHubGateDecision(
            action=normalized,
            allowed=False,
            dry_run=dry_run,
            reasons=[f"unsupported GitHub action: {action}"],
            evidence=_base_evidence(loaded_project, validation_log_hash),
        )

    git_policy_action = "push" if normalized == "push" else normalized
    git_policy = evaluate_git_action(loaded_project, git_policy_action)
    evidence = _base_evidence(loaded_project, validation_log_hash)
    evidence["gitPolicy"] = git_policy.as_dict()
    evidence["githubCliAvailable"] = shutil.which("gh") is not None

    reasons: list[str] = []
    external_required = False

    if not git_policy.allowed:
        reasons.extend(git_policy.reasons)
        external_required = external_required or git_policy.external_required

    if not evidence["githubCliAvailable"]:
        reasons.append("GitHub CLI is not installed or not on PATH")
        external_required = True
    elif not dry_run:
        auth = _gh_auth_status(loaded_project.root)
        evidence["githubAuth"] = auth
        if not auth["authenticated"]:
            reasons.append("GitHub CLI is not authenticated")
            external_required = True

    if normalized in {"pr", "merge"} and not validation_log_hash:
        reasons.append("validation log hash is required before PR or merge automation")

    if normalized == "merge":
        reasons.append("merge requires branch protection, review status, and explicit human-approved policy")
        external_required = True

    return GitHubGateDecision(
        action=normalized,
        allowed=dry_run and not reasons,
        dry_run=dry_run,
        external_required=external_required or not dry_run,
        reasons=reasons,
        evidence=evidence,
    )


def _base_evidence(loaded_project: LoadedProject, validation_log_hash: str | None) -> dict[str, Any]:
    return {
        "projectPath": str(loaded_project.root),
        "branch": loaded_project.current_branch,
        "commitSha": loaded_project.commit_sha,
        "protectedBranch": loaded_project.is_branch_protected(),
        "disposableWorktree": loaded_project.is_disposable_worktree(),
        "allowGitPush": loaded_project.profile.safety.allow_git_push,
        "validationLogHash": validation_log_hash,
    }


def _gh_auth_status(cwd: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["gh", "auth", "status"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"authenticated": False, "error": str(exc)}
    return {
        "authenticated": result.returncode == 0,
        "returnCode": result.returncode,
        "stdout": result.stdout[:2000],
        "stderr": result.stderr[:2000],
    }
