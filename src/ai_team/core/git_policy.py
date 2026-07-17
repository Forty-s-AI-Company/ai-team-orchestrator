from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_team.core.project_loader import LoadedProject


SECRET_PATTERNS = [
    re.compile(rb"sk-[A-Za-z0-9_\-]{10,}"),
    re.compile(rb"(?i)Bearer\s+[A-Za-z0-9_\-.]+"),
]

SECRET_ASSIGNMENT_PATTERN = re.compile(
    rb"(?i)(api[_-]?key|token|secret|password|hash[_-]?key|hash[_-]?iv)\s*[:=]\s*"
    rb"(?!(?:string|number|boolean|unknown|never|any|object|bigint|symbol)\b)([^\s,;]+)"
)
SAFE_SECRET_PLACEHOLDER_MARKERS = (
    b"dummy",
    b"example",
    b"fake",
    b"mock",
    b"placeholder",
    b"test",
)
RUNTIME_VALUE_PATTERN = re.compile(rb"[A-Za-z_$][A-Za-z0-9_$.]*(?:\([^\r\n]*\))?")

RUNTIME_ARTIFACT_MARKERS = {
    ".venv",
    "venv",
    "__pycache__",
    "reports",
    "logs",
    "receipts",
    ".hfc",
    ".pytest_cache",
}


@dataclass(frozen=True)
class GitPolicyDecision:
    action: str
    allowed: bool
    external_required: bool = False
    reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "allowed": self.allowed,
            "externalRequired": self.external_required,
            "reasons": self.reasons,
            "evidence": self.evidence,
        }


def evaluate_git_action(
    loaded_project: LoadedProject,
    action: str,
    candidate_files: list[str | Path] | None = None,
) -> GitPolicyDecision:
    normalized_action = action.lower().strip()
    reasons: list[str] = []
    evidence = {
        "branch": loaded_project.current_branch,
        "commitSha": loaded_project.commit_sha,
        "protectedBranch": loaded_project.is_branch_protected(),
        "disposableWorktree": loaded_project.is_disposable_worktree(),
        "allowGitPush": loaded_project.profile.safety.allow_git_push,
        "allowDeploy": loaded_project.profile.safety.allow_deploy,
    }

    file_check = inspect_candidate_files(loaded_project.root, candidate_files or [])
    evidence["fileCheck"] = file_check
    if file_check["blocked"]:
        reasons.extend(file_check["reasons"])

    if loaded_project.is_branch_protected():
        reasons.append("protected branch blocks automated git write actions")

    if normalized_action in {"add", "commit"}:
        if not loaded_project.is_disposable_worktree():
            reasons.append("git add/commit requires a disposable linked worktree")
        return GitPolicyDecision(
            action=normalized_action,
            allowed=not reasons,
            reasons=reasons,
            evidence=evidence,
        )

    if normalized_action == "push":
        if not loaded_project.is_disposable_worktree():
            reasons.append("push requires a disposable linked worktree")
        if not loaded_project.profile.safety.allow_git_push:
            reasons.append("project safety policy does not allow git push")
        return GitPolicyDecision(
            action=normalized_action,
            allowed=not reasons,
            external_required=not loaded_project.profile.safety.allow_git_push,
            reasons=reasons,
            evidence=evidence,
        )

    if normalized_action in {"pr", "pull-request"}:
        if not loaded_project.is_disposable_worktree():
            reasons.append("pull request creation requires a disposable linked worktree")
        if not loaded_project.profile.safety.allow_git_push:
            reasons.append("project safety policy does not allow git push required for pull request creation")
        return GitPolicyDecision(
            action=normalized_action,
            allowed=not reasons,
            external_required=not loaded_project.profile.safety.allow_git_push,
            reasons=reasons,
            evidence=evidence,
        )

    if normalized_action == "merge":
        return GitPolicyDecision(
            action=normalized_action,
            allowed=False,
            external_required=True,
            reasons=[
                "pull request and merge automation require GitHub authentication, branch protection checks, and reviewed receipts"
            ],
            evidence=evidence,
        )

    return GitPolicyDecision(
        action=normalized_action,
        allowed=False,
        reasons=[f"unsupported git action: {action}"],
        evidence=evidence,
    )


def inspect_candidate_files(project_root: Path, candidate_files: list[str | Path]) -> dict[str, Any]:
    reasons: list[str] = []
    inspected: list[str] = []
    for candidate in candidate_files:
        relative = Path(candidate)
        if relative.is_absolute():
            try:
                relative = relative.resolve().relative_to(project_root.resolve())
            except ValueError:
                reasons.append(f"candidate escapes project root: {candidate}")
                continue
        parts = {part.lower() for part in relative.parts}
        if parts.intersection(RUNTIME_ARTIFACT_MARKERS):
            reasons.append(f"runtime artifact path is blocked: {relative}")
            continue
        target = (project_root / relative).resolve()
        if _is_ignored(project_root, relative):
            reasons.append(f"ignored file is blocked: {relative}")
            continue
        if target.exists() and target.is_file():
            secret_check = _contains_candidate_secret(project_root, relative, target)
            if secret_check is None:
                reasons.append(f"candidate secret scan failed: {relative}")
                continue
            if secret_check:
                reasons.append(f"candidate appears to contain a secret: {relative}")
                continue
        inspected.append(str(relative))
    return {
        "blocked": bool(reasons),
        "reasons": reasons,
        "inspected": inspected,
    }


def _is_ignored(project_root: Path, relative: Path) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(relative)],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.returncode == 0


def _contains_candidate_secret(project_root: Path, relative: Path, path: Path) -> bool | None:
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", str(relative)],
        cwd=project_root,
        check=False,
        capture_output=True,
        timeout=10,
    )
    if tracked.returncode == 0:
        diff = subprocess.run(
            ["git", "diff", "HEAD", "--no-ext-diff", "--unified=0", "--", str(relative)],
            cwd=project_root,
            check=False,
            capture_output=True,
            timeout=10,
        )
        if diff.returncode != 0:
            return None
        data = b"\n".join(
            line[1:]
            for line in diff.stdout.splitlines()
            if line.startswith(b"+") and not line.startswith(b"+++")
        )[:1_000_000]
    elif tracked.returncode == 1:
        try:
            data = path.read_bytes()[:1_000_000]
        except OSError:
            return None
    else:
        return None
    return contains_secret_material(data)


def contains_secret_material(data: bytes) -> bool:
    """Return whether source bytes contain literal secret material.

    This is the single fail-closed classifier shared by candidate-file and
    committed-diff scans so publication cannot apply weaker or stale rules.
    """
    if any(pattern.search(data) for pattern in SECRET_PATTERNS):
        return True
    return any(_is_secret_assignment(match.group(2)) for match in SECRET_ASSIGNMENT_PATTERN.finditer(data))


def _is_secret_assignment(raw_value: bytes) -> bool:
    """Reject literal secret material while allowing references and explicit test placeholders."""
    value = raw_value.strip()
    quoted = len(value) >= 2 and value[:1] in {b'"', b"'", b"`"}
    normalized = value.strip(b'"\'`').lower()
    if any(marker in normalized for marker in SAFE_SECRET_PLACEHOLDER_MARKERS):
        return False
    runtime_value = value[1:-1].strip() if value.startswith(b"{") and value.endswith(b"}") else value
    if not quoted and RUNTIME_VALUE_PATTERN.fullmatch(runtime_value):
        return False
    return True
