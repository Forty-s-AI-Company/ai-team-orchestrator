"""Bounded Codex repair chain for watchdog restart loops.

The watchdog invokes this module only after its restart threshold is crossed.
Sol diagnoses with read-only access, Terra edits a disposable Git worktree,
deterministic project checks and a read-only Sol QA review gate the result, and
only then is the source branch fast-forwarded and pushed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from ai_team.core.isolated_executor import (
    prepare_dependency_link,
    remove_dependency_link,
    run_validation_commands,
)
from ai_team.core.project_loader import load_project
from ai_team.providers.base import redact_secrets


DIAGNOSIS_SCHEMA = "ai-team-watchdog-diagnosis/v1"
QA_SCHEMA = "ai-team-watchdog-qa/v1"
REPORT_SCHEMA = "ai-team-watchdog-ai-repair/v1"
MAX_MODEL_OUTPUT_BYTES = 128_000
MAX_PATCH_BYTES = 512_000
MAX_WRITE_PATHS = 12
MODEL_TIMEOUT_SECONDS = 1_200

REPOSITORY_PREFIXES = {
    "project": (
        "src/",
        "tests/",
        "scripts/",
        "playwright.config.",
    ),
    "orchestrator": (
        "src/",
        "tests/",
    ),
}


def run_watchdog_ai_repair(
    supervisor: dict[str, Any],
    *,
    project_path: Path,
    orchestrator_path: Path,
    report_dir: Path,
    codex_executable: str,
    diagnosis_model: str = "gpt-5.6-sol",
    repair_model: str = "gpt-5.6-terra",
    reasoning_effort: str = "high",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Diagnose, repair, validate, commit, and push one bounded recovery."""

    started = (now or datetime.now(UTC)).astimezone(UTC)
    project = project_path.resolve()
    orchestrator = orchestrator_path.resolve()
    reports = report_dir.resolve()
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "startedAt": started.isoformat(),
        "status": "running",
        "diagnosisModel": diagnosis_model,
        "repairModel": repair_model,
        "qaModel": diagnosis_model,
        "reasoningEffort": reasoning_effort,
        "supervisorEvidence": redact_secrets(supervisor),
    }
    worktree: Path | None = None
    dependency_link: Path | None = None
    source: Path | None = None
    try:
        _validate_repository(project)
        _validate_repository(orchestrator)
        evidence = _failure_evidence(supervisor, reports)
        diagnosis_result = _invoke_codex(
            codex_executable,
            model=diagnosis_model,
            reasoning_effort=reasoning_effort,
            root=orchestrator,
            write=False,
            prompt=_diagnosis_prompt(supervisor, evidence, project, orchestrator),
        )
        report["diagnosisCommand"] = _command_summary(diagnosis_result)
        diagnosis = _validate_diagnosis(_last_json_object(diagnosis_result.stdout))
        report["diagnosis"] = diagnosis
        if diagnosis["status"] != "repairable":
            raise ValueError(f"Sol diagnosis is not repairable: {diagnosis['summary']}")

        repository = diagnosis["repository"]
        source = project if repository == "project" else orchestrator
        _require_clean_development_branch(source)
        allowed_paths = _validate_write_paths(repository, diagnosis["allowedWritePaths"])
        base_sha = _git(source, "rev-parse", "HEAD").stdout.strip()
        branch = _git(source, "branch", "--show-current").stdout.strip()
        worktree = source.parent / f"{source.name}-watchdog-repair-{uuid4().hex[:10]}"
        _git(source, "worktree", "add", "--detach", str(worktree), base_sha)
        if repository == "project":
            dependency_link = prepare_dependency_link(worktree, source)

        repair_result = _invoke_codex(
            codex_executable,
            model=repair_model,
            reasoning_effort=reasoning_effort,
            root=worktree,
            write=True,
            prompt=_repair_prompt(diagnosis, allowed_paths),
        )
        report["repairCommand"] = _command_summary(repair_result)
        if _git(worktree, "rev-parse", "HEAD").stdout.strip() != base_sha:
            raise ValueError("Terra must not create Git commits during the repair stage")
        changed_files = _changed_files(worktree)
        if not changed_files:
            raise ValueError("Terra completed without producing a repair diff")
        outside = [path for path in changed_files if not _path_allowed(path, allowed_paths)]
        if outside:
            raise ValueError(f"Terra changed files outside diagnosed scope: {', '.join(outside)}")

        _git(worktree, "add", "--", *changed_files)
        patch = _git(worktree, "diff", "--cached", "--no-ext-diff", "--binary", "HEAD").stdout
        if len(patch.encode("utf-8")) > MAX_PATCH_BYTES:
            raise ValueError("repair patch exceeds the bounded QA evidence limit")
        candidate_hash = hashlib.sha256(patch.encode("utf-8", "replace")).hexdigest()
        validation = _run_deterministic_qa(repository, source, worktree)
        report["validation"] = validation
        if validation.get("success") is not True:
            raise ValueError("deterministic QA failed")
        _assert_candidate_unchanged(worktree, changed_files, candidate_hash, "deterministic QA")

        qa_result = _invoke_codex(
            codex_executable,
            model=diagnosis_model,
            reasoning_effort=reasoning_effort,
            root=worktree,
            write=False,
            prompt=_qa_prompt(diagnosis, validation, patch),
        )
        report["qaCommand"] = _command_summary(qa_result)
        qa = _validate_qa(_last_json_object(qa_result.stdout))
        report["qa"] = qa
        if qa["status"] != "passed" or qa["findings"]:
            raise ValueError("Codex QA rejected the repair")
        _assert_candidate_unchanged(worktree, changed_files, candidate_hash, "Codex QA")

        _git(worktree, "commit", "-m", f"fix(ai-team): {diagnosis['summary'][:72]}")
        repair_sha = _git(worktree, "rev-parse", "HEAD").stdout.strip()
        if _git(source, "rev-parse", "HEAD").stdout.strip() != base_sha:
            raise ValueError("source branch changed during automatic repair")
        if _git(source, "status", "--porcelain").stdout.strip():
            raise ValueError("source repository became dirty during automatic repair")
        _git(source, "merge", "--ff-only", repair_sha)
        _git(source, "push", "origin", f"HEAD:{branch}", timeout=180)
        report.update({
            "status": "passed",
            "repository": repository,
            "sourceRoot": str(source),
            "baseSha": base_sha,
            "repairSha": repair_sha,
            "branch": branch,
            "changedFiles": changed_files,
        })
        return _finish(report, reports, success=True, diagnostic="Sol diagnosis, Terra repair, and QA passed")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        report["status"] = "failed"
        report["error"] = str(redact_secrets(str(exc)))[:500]
        return _finish(report, reports, success=False, diagnostic=report["error"])
    finally:
        try:
            remove_dependency_link(dependency_link)
        except OSError:
            # Cleanup evidence is secondary to the already recorded repair
            # result; force-removing the worktree below gets another chance.
            pass
        if worktree is not None and source is not None and worktree.exists():
            subprocess.run(
                ["git", "-C", str(source), "worktree", "remove", "--force", str(worktree)],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )


def _diagnosis_prompt(
    supervisor: dict[str, Any],
    evidence: dict[str, Any],
    project: Path,
    orchestrator: Path,
) -> str:
    return "\n".join((
        "You are the read-only incident diagnostician for an autonomous development watchdog.",
        f"Product repository: {project}",
        f"Orchestrator repository: {orchestrator}",
        "Inspect both repositories and the bounded evidence below. Do not edit files or change external state.",
        "Choose exactly one repository for the smallest root-cause repair.",
        "Never request writes to .git, .env files, credentials, user systemd files, production deployment,",
        "database migrations/seeds, real payments, or destructive operations.",
        "Return JSON only without Markdown using this exact shape:",
        '{"schema":"ai-team-watchdog-diagnosis/v1","status":"repairable|unrepairable",'
        '"repository":"project|orchestrator","summary":"short Chinese summary",'
        '"rootCause":"concrete evidence-backed cause","repairInstruction":"bounded implementation instruction",'
        '"allowedWritePaths":["exact/relative/path"]}',
        f"SupervisorEvidence={json.dumps(redact_secrets(supervisor), ensure_ascii=False)}",
        f"FailureEvidence={json.dumps(redact_secrets(evidence), ensure_ascii=False)}",
    ))


def _repair_prompt(diagnosis: dict[str, Any], allowed_paths: list[str]) -> str:
    return "\n".join((
        "You are the repair engineer. Implement the diagnosed fix in this disposable Git worktree.",
        "You may edit only the exact allowlisted paths below. Do not commit, push, deploy, access secrets,",
        "run migrations/seeds, perform real payments, or modify external services.",
        "Add or update focused tests when an allowlisted test path permits it.",
        f"Diagnosis={json.dumps(diagnosis, ensure_ascii=False)}",
        f"AllowedWritePaths={json.dumps(allowed_paths, ensure_ascii=False)}",
        "When finished, provide a concise summary; the watchdog performs independent tests and review.",
    ))


def _qa_prompt(diagnosis: dict[str, Any], validation: dict[str, Any], patch: str) -> str:
    return "\n".join((
        "You are the read-only QA reviewer for an automatic watchdog repair.",
        "Review the exact untrusted patch and deterministic test evidence. Do not edit files or run external actions.",
        "Check security, correctness, performance, readability, and maintainability.",
        "Return JSON only without Markdown using exactly:",
        '{"schema":"ai-team-watchdog-qa/v1","status":"passed|failed","summary":"Chinese summary",'
        '"findings":["actionable finding"]}',
        f"Diagnosis={json.dumps(diagnosis, ensure_ascii=False)}",
        f"Validation={json.dumps(redact_secrets(validation), ensure_ascii=False)}",
        f"UntrustedPatchJson={json.dumps(patch, ensure_ascii=False)}",
    ))


def _invoke_codex(
    executable: str,
    *,
    model: str,
    reasoning_effort: str,
    root: Path,
    write: bool,
    prompt: str,
) -> subprocess.CompletedProcess[str]:
    args = [
        executable,
        "--ask-for-approval",
        "never",
        "exec",
        "--sandbox",
        "workspace-write" if write else "read-only",
        "--skip-git-repo-check",
        "--model",
        model,
        "--config",
        f'model_reasoning_effort="{reasoning_effort}"',
        "-",
    ]
    completed = subprocess.run(
        args,
        cwd=root,
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
        timeout=MODEL_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        stderr = str(redact_secrets(completed.stderr))[-2_000:]
        raise ValueError(f"Codex {model} failed with exit {completed.returncode}: {stderr}")
    if len(completed.stdout.encode("utf-8", "replace")) > MAX_MODEL_OUTPUT_BYTES:
        raise ValueError(f"Codex {model} output exceeds the bounded limit")
    return completed


def _validate_diagnosis(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != DIAGNOSIS_SCHEMA:
        raise ValueError("Sol diagnosis schema is invalid")
    if value.get("status") not in {"repairable", "unrepairable"}:
        raise ValueError("Sol diagnosis status is invalid")
    if value.get("repository") not in REPOSITORY_PREFIXES:
        raise ValueError("Sol diagnosis repository is invalid")
    for key in ("summary", "rootCause", "repairInstruction"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"Sol diagnosis {key} is required")
    if not isinstance(value.get("allowedWritePaths"), list):
        raise ValueError("Sol diagnosis allowedWritePaths must be a list")
    return value


def _validate_qa(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != QA_SCHEMA or value.get("status") not in {"passed", "failed"}:
        raise ValueError("Sol QA response is invalid")
    if not isinstance(value.get("summary"), str) or not isinstance(value.get("findings"), list):
        raise ValueError("Sol QA response fields are invalid")
    if not all(isinstance(item, str) for item in value["findings"]):
        raise ValueError("Sol QA findings must be strings")
    return value


def _validate_write_paths(repository: str, values: list[Any]) -> list[str]:
    if not values or len(values) > MAX_WRITE_PATHS:
        raise ValueError("diagnosed write scope must contain 1 to 12 paths")
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("diagnosed write paths must be strings")
        path = PurePosixPath(value.strip().replace("\\", "/"))
        normalized = path.as_posix()
        if path.is_absolute() or not normalized or ".." in path.parts or normalized.startswith(".git"):
            raise ValueError(f"unsafe diagnosed write path: {value}")
        if not any(normalized == prefix.rstrip("/") or normalized.startswith(prefix) for prefix in REPOSITORY_PREFIXES[repository]):
            raise ValueError(f"diagnosed write path is outside repair policy: {normalized}")
        result.append(normalized.rstrip("/"))
    return sorted(set(result))


def _path_allowed(path: str, allowed: list[str]) -> bool:
    normalized = PurePosixPath(path).as_posix()
    return any(normalized == root or normalized.startswith(f"{root}/") for root in allowed)


def _run_deterministic_qa(repository: str, source: Path, worktree: Path) -> dict[str, Any]:
    temp_root = Path.home() / ".local" / "state" / "ai-team" / "watchdog-test-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    os.chmod(temp_root, 0o700)
    if repository == "project":
        loaded = load_project(worktree, allowlist=[source.parent])
        commands = list(dict.fromkeys([
            loaded.profile.commands.lint,
            loaded.profile.commands.typecheck,
            loaded.profile.commands.test,
            loaded.profile.commands.build,
            *loaded.profile.commands.additional_validation,
        ]))
        commands = [f"/usr/bin/env TMPDIR={temp_root} {command}" for command in commands]
        database_url = os.environ.get("AI_TEAM_TEST_DATABASE_URL", "").strip()
        overrides = {"DATABASE_URL": database_url, "DIRECT_URL": database_url} if database_url else None
        return run_validation_commands(
            worktree,
            commands,
            require_nonempty=True,
            dependency_root=worktree,
            environment_overrides=overrides,
        )
    commands = [
        (
            f"/usr/bin/env TMPDIR={temp_root} PYTHONPATH={worktree / 'src'} "
            f"{source / '.venv/bin/python'} -m pytest -q"
        ),
        (
            f"/usr/bin/env TMPDIR={temp_root} RUFF_CACHE_DIR={temp_root / 'ruff-cache'} "
            f"{source / '.venv/bin/ruff'} check src tests"
        ),
    ]
    return run_validation_commands(worktree, commands, require_nonempty=True)


def _failure_evidence(supervisor: dict[str, Any], report_dir: Path) -> dict[str, Any]:
    external = supervisor.get("externalQa")
    receipt_value = external.get("receiptPath") if isinstance(external, dict) else None
    if not isinstance(receipt_value, str):
        return {}
    path = Path(receipt_value)
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(report_dir)
    except (OSError, ValueError):
        return {}
    if not resolved.is_file() or resolved.is_symlink() or resolved.stat().st_size > 1_000_000:
        return {}
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _changed_files(root: Path) -> list[str]:
    tracked = _git(root, "diff", "--name-only", "HEAD").stdout.splitlines()
    untracked = _git(root, "ls-files", "--others", "--exclude-standard").stdout.splitlines()
    return sorted({item.strip() for item in [*tracked, *untracked] if item.strip()})


def _assert_candidate_unchanged(
    root: Path,
    expected_files: list[str],
    expected_hash: str,
    stage: str,
) -> None:
    current_files = _changed_files(root)
    current_patch = _git(root, "diff", "--cached", "--no-ext-diff", "--binary", "HEAD").stdout
    unstaged_patch = _git(root, "diff", "--no-ext-diff", "--binary").stdout
    current_hash = hashlib.sha256(current_patch.encode("utf-8", "replace")).hexdigest()
    if current_files != expected_files or current_hash != expected_hash or unstaged_patch:
        raise ValueError(f"{stage} modified the candidate repair")


def _require_clean_development_branch(root: Path) -> None:
    if _git(root, "status", "--porcelain").stdout.strip():
        raise ValueError(f"automatic repair requires a clean source repository: {root}")
    branch = _git(root, "branch", "--show-current").stdout.strip()
    if not branch or branch in {"main", "master"}:
        raise ValueError("automatic repair requires a non-protected development branch")


def _validate_repository(root: Path) -> None:
    if not root.is_dir() or not (root / ".git").exists():
        raise ValueError(f"automatic repair repository is invalid: {root}")


def _git(root: Path, *args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        error = str(redact_secrets(completed.stderr))[-2_000:]
        raise ValueError(f"git {' '.join(args[:2])} failed: {error}")
    return completed


def _last_json_object(output: str) -> dict[str, Any]:
    stripped = output.strip()
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    for match in reversed(list(re.finditer(r"\{", stripped))):
        try:
            value = json.loads(stripped[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Codex returned no valid JSON object")


def _command_summary(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    output = completed.stdout.encode("utf-8", "replace")
    return {
        "exitCode": completed.returncode,
        "outputSha256": hashlib.sha256(output).hexdigest(),
        "outputBytes": len(output),
    }


def _finish(
    report: dict[str, Any],
    report_dir: Path,
    *,
    success: bool,
    diagnostic: str,
) -> dict[str, Any]:
    report["completedAt"] = datetime.now(UTC).isoformat()
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"watchdog-ai-repair-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    path.write_text(
        json.dumps(redact_secrets(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)
    return {
        "attempted": True,
        "success": success,
        "action": "codex-sol-terra-qa-repair",
        "diagnostic": diagnostic,
        "restarted": False,
        "reportPath": str(path),
        "repository": report.get("repository"),
        "repairSha": report.get("repairSha"),
        "changedFiles": report.get("changedFiles", []),
    }
