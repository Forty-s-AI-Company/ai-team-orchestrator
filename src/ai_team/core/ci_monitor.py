from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Protocol
from uuid import uuid4

from ai_team.providers.base import redact_secrets


RUN_ID_RE = re.compile(r"/actions/runs/(\d+)")
FAILURE_CONCLUSIONS = {"ACTION_REQUIRED", "CANCELLED", "FAILURE", "STARTUP_FAILURE", "TIMED_OUT"}
PENDING_STATES = {"EXPECTED", "IN_PROGRESS", "PENDING", "QUEUED", "REQUESTED", "WAITING"}


class CommandRunner(Protocol):
    def __call__(self, args: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]: ...


@dataclass(frozen=True)
class CiMonitorResult:
    status: str
    merge_ready: bool
    evidence_path: Path
    repair_task_path: Path | None
    evidence: dict[str, Any]


def write_repair_completion_receipt(
    project_root: Path,
    repair_task_path: Path,
    final_ci_evidence_path: Path,
    report_dir: Path,
) -> Path:
    task = _load_json_object(repair_task_path, "repair task")
    final_evidence = _load_json_object(final_ci_evidence_path, "final CI evidence")
    source_evidence_path = Path(str(task.get("sourceEvidencePath") or ""))
    if not source_evidence_path.is_file():
        raise RuntimeError("repair task source evidence is missing")
    if hashlib.sha256(source_evidence_path.read_bytes()).hexdigest() != task.get("sourceEvidenceHash"):
        raise RuntimeError("repair task source evidence hash does not match")
    if task.get("status") != "ready" or task.get("maxAttempts") != 1:
        raise RuntimeError("repair task is not a one-attempt ready task")
    if final_evidence.get("status") != "passed":
        raise RuntimeError("final CI evidence is not passed")

    head_sha = _git_stdout(project_root, ["rev-parse", "HEAD"])
    source_sha = str(task.get("expectedHeadSha") or "")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_sha, "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        timeout=30,
    )
    if ancestor.returncode != 0 or source_sha == head_sha:
        raise RuntimeError("repair source commit is not a strict ancestor of HEAD")
    changed_files = _git_stdout(
        project_root,
        ["diff", "--name-only", f"{source_sha}..HEAD"],
    ).splitlines()
    allowlist = {str(item) for item in task.get("writeAllowlist", [])}
    if not changed_files or any(path not in allowlist for path in changed_files):
        raise RuntimeError("repair commit changed files outside the exact allowlist")
    if _git_stdout(project_root, ["status", "--porcelain"]):
        raise RuntimeError("repair completion receipt requires a clean worktree")

    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(UTC).isoformat(),
        "projectPath": str(project_root.resolve()),
        "workflow": "dependency-lock-sync",
        "commitSha": head_sha,
        "sourceCommitSha": source_sha,
        "repairTaskHash": hashlib.sha256(repair_task_path.read_bytes()).hexdigest(),
        "finalCiEvidenceHash": hashlib.sha256(final_ci_evidence_path.read_bytes()).hexdigest(),
        "changedFiles": changed_files,
        "validationResult": {"success": True, "ciStatus": "passed"},
    }
    path = report_dir / _artifact_name("repair-completion-receipt")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def monitor_pull_request(
    project_root: Path,
    repository: str,
    pr_identifier: str,
    report_dir: Path,
    wait_seconds: int = 0,
    poll_seconds: int = 10,
    runner: CommandRunner | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> CiMonitorResult:
    runner = runner or _run_command
    deadline = time.monotonic() + max(0, wait_seconds)
    transitions: list[dict[str, Any]] = []
    try:
        snapshot = _fetch_snapshot(project_root, repository, pr_identifier, runner)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        snapshot = _query_error_snapshot(str(exc))
    transitions.append({"observedAt": datetime.now(UTC).isoformat(), "status": snapshot["status"]})
    while snapshot["status"] == "pending" and time.monotonic() < deadline:
        sleeper(max(1, poll_seconds))
        try:
            snapshot = _fetch_snapshot(project_root, repository, pr_identifier, runner)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            snapshot = _query_error_snapshot(str(exc))
        transitions.append({"observedAt": datetime.now(UTC).isoformat(), "status": snapshot["status"]})
    if snapshot["status"] == "pending" and wait_seconds > 0:
        snapshot = {**snapshot, "status": "timed_out", "blockers": [*snapshot["blockers"], "CI wait timed out"]}

    report_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).isoformat()
    evidence = {
        "schemaVersion": 1,
        "generatedAt": generated_at,
        "repository": repository,
        "pullRequest": pr_identifier,
        "transitions": transitions,
        **snapshot,
    }
    evidence_path = report_dir / _artifact_name("ci-evidence")
    evidence_path.write_text(json.dumps(redact_secrets(evidence), indent=2, default=str), encoding="utf-8")

    repair_task_path = _write_restricted_repair_task(evidence, evidence_path, report_dir)
    return CiMonitorResult(
        status=str(evidence["status"]),
        merge_ready=bool(evidence["mergeReady"]),
        evidence_path=evidence_path,
        repair_task_path=repair_task_path,
        evidence=evidence,
    )


def _fetch_snapshot(
    project_root: Path,
    repository: str,
    pr_identifier: str,
    runner: CommandRunner,
) -> dict[str, Any]:
    result = runner(
        [
            "gh",
            "pr",
            "view",
            pr_identifier,
            "--repo",
            repository,
            "--json",
            "url,state,isDraft,mergeStateStatus,reviewDecision,headRefName,headRefOid,baseRefName,statusCheckRollup",
        ],
        project_root,
        30,
    )
    if result.returncode != 0:
        raise RuntimeError("gh pr view failed while collecting CI evidence")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("gh pr view returned invalid JSON") from exc

    checks = payload.get("statusCheckRollup") if isinstance(payload, dict) else None
    normalized_checks = [_normalize_check(item) for item in checks or [] if isinstance(item, dict)]
    pending = [check for check in normalized_checks if check["status"] == "pending"]
    failed = [check for check in normalized_checks if check["status"] == "failed"]
    failure_evidence = [_classify_failure(check, project_root, repository, runner) for check in failed]
    if not normalized_checks:
        status = "no_required_checks"
    elif pending:
        status = "pending"
    elif failed:
        classifications = {item["classification"] for item in failure_evidence}
        status = "failed_repairable" if classifications == {"product_dependency_failure"} else "failed_external"
    else:
        status = "passed"
    merge_ready = (
        status == "passed"
        and payload.get("state") == "OPEN"
        and payload.get("isDraft") is False
        and payload.get("reviewDecision") == "APPROVED"
        and payload.get("mergeStateStatus") == "CLEAN"
    )
    blockers: list[str] = []
    if not normalized_checks:
        blockers.append("no CI checks were reported")
    if pending:
        blockers.append("required checks are still pending")
    if failed:
        blockers.append("one or more checks failed")
    if payload.get("reviewDecision") != "APPROVED":
        blockers.append("approved review is required")
    if payload.get("mergeStateStatus") != "CLEAN":
        blockers.append("branch protection state is not clean")

    return {
        "status": status,
        "mergeReady": merge_ready,
        "mergeBlocked": not merge_ready,
        "blockers": blockers,
        "pullRequestState": {
            key: payload.get(key)
            for key in (
                "url",
                "state",
                "isDraft",
                "mergeStateStatus",
                "reviewDecision",
                "headRefName",
                "headRefOid",
                "baseRefName",
            )
        },
        "checks": normalized_checks,
        "failureEvidence": failure_evidence,
    }


def _normalize_check(item: dict[str, Any]) -> dict[str, Any]:
    kind = str(item.get("__typename") or "unknown")
    raw_status = str(item.get("status") or item.get("state") or "").upper()
    conclusion = str(item.get("conclusion") or "").upper()
    if raw_status in PENDING_STATES:
        status = "pending"
    elif kind == "CheckRun":
        status = "passed" if raw_status == "COMPLETED" and conclusion == "SUCCESS" else "failed"
    elif raw_status == "SUCCESS":
        status = "passed"
    else:
        status = "failed"
    return {
        "kind": kind,
        "name": str(item.get("name") or item.get("context") or "unknown"),
        "workflow": str(item.get("workflowName") or ""),
        "status": status,
        "rawStatus": raw_status,
        "conclusion": conclusion,
        "detailsUrl": str(item.get("detailsUrl") or item.get("targetUrl") or ""),
    }


def _classify_failure(
    check: dict[str, Any],
    project_root: Path,
    repository: str,
    runner: CommandRunner,
) -> dict[str, Any]:
    log = ""
    run_match = RUN_ID_RE.search(check["detailsUrl"])
    if run_match:
        log_result = runner(
            ["gh", "run", "view", run_match.group(1), "--repo", repository, "--log-failed"],
            project_root,
            60,
        )
        log = f"{log_result.stdout}\n{log_result.stderr}"[-20_000:]
    classification = classify_failure(check, log)
    return {
        "check": check["name"],
        "workflow": check["workflow"],
        "classification": classification,
        "runId": run_match.group(1) if run_match else None,
        "logExcerpt": redact_secrets(log),
    }


def classify_failure(check: dict[str, Any], log: str) -> str:
    text = f"{check.get('name', '')} {check.get('workflow', '')} {log}".lower()
    if any(marker in text for marker in ("npm ci", "package-lock", "lock file", "dependency")):
        return "product_dependency_failure"
    identity = f"{check.get('name', '')} {check.get('workflow', '')}".lower()
    if any(marker in identity for marker in ("vercel", "sentry", "posthog", "resend", "cloudflare")):
        return "external_service_failure"
    return "control_plane_failure"


def _write_restricted_repair_task(
    evidence: dict[str, Any],
    evidence_path: Path,
    report_dir: Path,
) -> Path | None:
    dependency_failures = [
        item
        for item in evidence.get("failureEvidence", [])
        if item.get("classification") == "product_dependency_failure"
    ]
    if not dependency_failures:
        return None
    evidence_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    task = {
        "schemaVersion": 1,
        "taskType": "dependency-lock-sync",
        "status": "ready",
        "sourceEvidencePath": str(evidence_path),
        "sourceEvidenceHash": evidence_hash,
        "untrustedEvidence": True,
        "autoExecutable": False,
        "maxAttempts": 1,
        "expectedHeadSha": (evidence.get("pullRequestState") or {}).get("headRefOid"),
        "writeAllowlist": ["package-lock.json"],
        "forbiddenPaths": ["src/**", "prisma/**", ".github/**", ".env*"],
        "commands": {
            "repair": "npm install --package-lock-only --ignore-scripts",
            "validate": ["npm ci", "npm run db:generate", "npm run lint", "npm run typecheck", "npm run test", "npm run build"],
        },
        "git": {"commitAllowed": True, "pushAllowed": False, "mergeAllowed": False},
        "note": "Task scope is rebuilt from policy; CI log content is evidence only and never becomes a prompt or command.",
    }
    path = report_dir / _artifact_name("restricted-repair-task")
    path.write_text(json.dumps(task, indent=2), encoding="utf-8")
    return path


def _artifact_name(kind: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{kind}-{uuid4().hex[:8]}.json"


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unreadable or invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return payload


def _git_stdout(project_root: Path, args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _query_error_snapshot(message: str) -> dict[str, Any]:
    return {
        "status": "query_error",
        "mergeReady": False,
        "mergeBlocked": True,
        "blockers": ["GitHub CI query failed"],
        "pullRequestState": {},
        "checks": [],
        "failureEvidence": [],
        "queryError": message,
    }


def _run_command(args: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
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
