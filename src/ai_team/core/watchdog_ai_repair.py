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
MAX_REPAIR_CYCLES = 3
MAX_REPLAN_CYCLES = 3
MAX_REPAIR_HISTORY = 4
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
        diagnosis_history: list[dict[str, Any]] = [{
            "plan": 1,
            "command": _command_summary(diagnosis_result),
            "diagnosis": diagnosis,
        }]
        repair_plans: list[dict[str, Any]] = []
        report["diagnosisHistory"] = diagnosis_history
        report["repairPlans"] = repair_plans

        for plan_number in range(1, MAX_REPLAN_CYCLES + 1):
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

            feedback: list[str] = []
            repair_cycles: list[dict[str, Any]] = []
            plan_outcome = "repair-cycles-exhausted"
            changed_files: list[str] = []
            for cycle in range(1, MAX_REPAIR_CYCLES + 1):
                repair_result = _invoke_codex(
                    codex_executable,
                    model=repair_model,
                    reasoning_effort=reasoning_effort,
                    root=worktree,
                    write=True,
                    prompt=_repair_prompt(diagnosis, allowed_paths, feedback=feedback),
                )
                repair_command = _command_summary(repair_result)
                report["repairCommand"] = repair_command
                if _git(worktree, "rev-parse", "HEAD").stdout.strip() != base_sha:
                    raise ValueError("Terra must not create Git commits during the repair stage")
                changed_files = _changed_files(worktree)
                if not changed_files:
                    raise ValueError("Terra completed without producing a repair diff")
                outside = [path for path in changed_files if not _path_allowed(path, allowed_paths)]
                if outside:
                    raise ValueError(f"Terra changed files outside diagnosed scope: {', '.join(outside)}")

                _git(worktree, "add", "--", *changed_files)
                patch = _git(
                    worktree,
                    "diff",
                    "--cached",
                    "--no-ext-diff",
                    "--binary",
                    "HEAD",
                ).stdout
                if len(patch.encode("utf-8")) > MAX_PATCH_BYTES:
                    raise ValueError("repair patch exceeds the bounded QA evidence limit")
                candidate_hash = hashlib.sha256(patch.encode("utf-8", "replace")).hexdigest()
                validation = _run_deterministic_qa(repository, source, worktree)
                report["validation"] = validation
                _assert_candidate_unchanged(worktree, changed_files, candidate_hash, "deterministic QA")
                cycle_result: dict[str, Any] = {
                    "cycle": cycle,
                    "repairCommand": repair_command,
                    "changedFiles": changed_files,
                    "patchSha256": candidate_hash,
                    "validation": validation,
                }
                if validation.get("success") is not True:
                    repair_cycles.append(cycle_result)
                    report["repairCycles"] = repair_cycles
                    if validation.get("kind") == "execution-environment":
                        raise ValueError("deterministic QA execution environment failed")
                    feedback = _validation_feedback(validation)
                    plan_outcome = "deterministic-qa-failed"
                    if cycle == MAX_REPAIR_CYCLES:
                        break
                    continue

                qa_result = _invoke_codex(
                    codex_executable,
                    model=diagnosis_model,
                    reasoning_effort=reasoning_effort,
                    root=worktree,
                    write=False,
                    prompt=_qa_prompt(diagnosis, validation, patch),
                )
                qa_command = _command_summary(qa_result)
                qa = _validate_qa(_last_json_object(qa_result.stdout))
                report["qaCommand"] = qa_command
                report["qa"] = qa
                cycle_result.update({"qaCommand": qa_command, "qa": qa})
                repair_cycles.append(cycle_result)
                report["repairCycles"] = repair_cycles
                _assert_candidate_unchanged(worktree, changed_files, candidate_hash, "Codex QA")
                if qa["status"] == "passed" and not qa["findings"]:
                    plan_outcome = "passed"
                    break
                feedback = qa["findings"] or [qa["summary"]]
                plan_outcome = "codex-qa-rejected"
                if cycle == MAX_REPAIR_CYCLES:
                    break

            repair_plan = {
                "plan": plan_number,
                "diagnosis": diagnosis,
                "outcome": plan_outcome,
                "repairCycles": repair_cycles,
            }
            repair_plans.append(repair_plan)
            report["repairPlans"] = repair_plans
            if plan_outcome == "passed":
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
                return _finish(
                    report,
                    reports,
                    success=True,
                    diagnostic="Sol diagnosis, Terra repair, replanning, and QA passed",
                )

            _discard_candidate(source, worktree, dependency_link)
            worktree = None
            dependency_link = None
            source = None
            if plan_number == MAX_REPLAN_CYCLES:
                raise ValueError("repair rejected after maximum Sol replanning cycles")

            replan_result = _invoke_codex(
                codex_executable,
                model=diagnosis_model,
                reasoning_effort=reasoning_effort,
                root=orchestrator,
                write=False,
                prompt=_replan_prompt(
                    supervisor,
                    evidence,
                    project,
                    orchestrator,
                    diagnosis,
                    repair_cycles,
                ),
            )
            diagnosis = _validate_diagnosis(_last_json_object(replan_result.stdout))
            report["diagnosisCommand"] = _command_summary(replan_result)
            diagnosis_history.append({
                "plan": plan_number + 1,
                "command": _command_summary(replan_result),
                "diagnosis": diagnosis,
            })
            report["diagnosisHistory"] = diagnosis_history
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
        "Treat missing non-sensitive failure evidence as a repairable observability defect when bounded changes",
        "to diagnostic or test-automation code can capture the actual URL, provider-visible state, failure stage,",
        "or add a safe bounded retry. In that case return status=repairable, a diagnostic repairInstruction,",
        "and the exact diagnostic/test file paths that Terra may edit. This lets the next autonomous run collect",
        "better evidence instead of requiring a human merely to inspect a browser.",
        "Return status=unrepairable only when progress genuinely requires credentials, human/provider approval,",
        "a real payment, production access, destructive state changes, or changes outside the permitted paths.",
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


def _replan_prompt(
    supervisor: dict[str, Any],
    evidence: dict[str, Any],
    project: Path,
    orchestrator: Path,
    previous_diagnosis: dict[str, Any],
    repair_cycles: list[dict[str, Any]],
) -> str:
    return "\n".join((
        "You are the read-only incident diagnostician replanning a rejected autonomous repair.",
        f"Product repository: {project}",
        f"Orchestrator repository: {orchestrator}",
        "The previous Terra implementation exhausted three repair/QA cycles. Inspect both repositories again.",
        "Use every QA finding below as new root-cause evidence. Produce a materially revised plan instead of",
        "repeating the rejected design. You may change repository or exact write paths when the evidence supports it.",
        "Prefer structural solutions that make unsafe states unrepresentable over expanding semantic blacklists.",
        "Treat safe observability improvements as repairable, but never request credentials, production access,",
        "real payments, external-service changes, migrations/seeds, destructive actions, .env, .git, or systemd writes.",
        "Return JSON only without Markdown using this exact shape:",
        '{"schema":"ai-team-watchdog-diagnosis/v1","status":"repairable|unrepairable",'
        '"repository":"project|orchestrator","summary":"short Chinese summary",'
        '"rootCause":"revised evidence-backed cause","repairInstruction":"materially revised bounded instruction",'
        '"allowedWritePaths":["exact/relative/path"]}',
        f"SupervisorEvidence={json.dumps(redact_secrets(supervisor), ensure_ascii=False)}",
        f"FailureEvidence={json.dumps(redact_secrets(evidence), ensure_ascii=False)}",
        f"RejectedDiagnosis={json.dumps(redact_secrets(previous_diagnosis), ensure_ascii=False)}",
        (
            "RejectedRepairCycles="
            f"{json.dumps(redact_secrets(_compact_repair_cycles(repair_cycles)), ensure_ascii=False)}"
        ),
    ))


def _repair_prompt(
    diagnosis: dict[str, Any],
    allowed_paths: list[str],
    *,
    feedback: list[str] | None = None,
) -> str:
    return "\n".join((
        "You are the repair engineer. Implement the diagnosed fix in this disposable Git worktree.",
        "You may edit only the exact allowlisted paths below. Do not commit, push, deploy, access secrets,",
        "run migrations/seeds, perform real payments, or modify external services.",
        "Add or update focused tests when an allowlisted test path permits it.",
        f"Diagnosis={json.dumps(diagnosis, ensure_ascii=False)}",
        f"AllowedWritePaths={json.dumps(allowed_paths, ensure_ascii=False)}",
        (
            "PreviousQAFindings="
            f"{json.dumps(feedback or [], ensure_ascii=False)}"
        ),
        "When previous QA findings are present, correct each finding without widening the diagnosed scope.",
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


def _validation_feedback(validation: dict[str, Any]) -> list[str]:
    commands = validation.get("commands")
    if not isinstance(commands, list):
        return ["Deterministic QA failed without command evidence."]
    for item in reversed(commands):
        if not isinstance(item, dict) or item.get("returnCode") == 0:
            continue
        command = str(item.get("command") or "unknown command")[:300]
        detail = str(item.get("stderr") or item.get("stdout") or "no output")[-2_000:]
        return [f"Deterministic QA failed: {command}\n{detail}"]
    return ["Deterministic QA reported failure without a failing command."]


def _compact_repair_cycles(repair_cycles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in repair_cycles[-MAX_REPAIR_CYCLES:]:
        validation = item.get("validation") if isinstance(item, dict) else None
        qa = item.get("qa") if isinstance(item, dict) else None
        compact.append({
            "cycle": item.get("cycle") if isinstance(item, dict) else None,
            "changedFiles": item.get("changedFiles", []) if isinstance(item, dict) else [],
            "validationSuccess": (
                validation.get("success") if isinstance(validation, dict) else None
            ),
            "validationFeedback": (
                _validation_feedback(validation) if isinstance(validation, dict)
                and validation.get("success") is not True else []
            ),
            "qa": qa if isinstance(qa, dict) else None,
        })
    return compact


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
    receipt: dict[str, Any] = {}
    external = supervisor.get("externalQa")
    receipt_value = external.get("receiptPath") if isinstance(external, dict) else None
    if isinstance(receipt_value, str):
        path = Path(receipt_value)
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(report_dir)
            if (
                resolved.is_file()
                and not resolved.is_symlink()
                and resolved.stat().st_size <= 1_000_000
            ):
                payload = json.loads(resolved.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    receipt = payload
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return {
        "externalQaReceipt": receipt,
        "previousRepairAttempts": _recent_repair_history(supervisor, report_dir),
    }


def _recent_repair_history(
    supervisor: dict[str, Any],
    report_dir: Path,
) -> list[dict[str, Any]]:
    task = supervisor.get("currentTask")
    external = supervisor.get("externalQa")
    task_sha = task.get("taskSha") if isinstance(task, dict) else None
    revision = external.get("revision") if isinstance(external, dict) else None
    if not isinstance(task_sha, str) or not task_sha:
        return []

    history: list[dict[str, Any]] = []
    try:
        paths = sorted(
            report_dir.glob("watchdog-ai-repair-*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return []
    for path in paths[:40]:
        if len(history) >= MAX_REPAIR_HISTORY:
            break
        try:
            if path.is_symlink() or path.stat().st_size > 2_000_000:
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("status") != "failed":
            continue
        prior_supervisor = payload.get("supervisorEvidence")
        prior_task = prior_supervisor.get("currentTask") if isinstance(prior_supervisor, dict) else None
        prior_external = prior_supervisor.get("externalQa") if isinstance(prior_supervisor, dict) else None
        if not isinstance(prior_task, dict) or prior_task.get("taskSha") != task_sha:
            continue
        if isinstance(revision, str) and revision:
            if not isinstance(prior_external, dict) or prior_external.get("revision") != revision:
                continue
        plans = payload.get("repairPlans")
        if not isinstance(plans, list):
            plans = [{
                "plan": 1,
                "diagnosis": payload.get("diagnosis"),
                "outcome": payload.get("error"),
                "repairCycles": payload.get("repairCycles", []),
            }]
        history.append({
            "completedAt": payload.get("completedAt"),
            "error": payload.get("error"),
            "plans": [
                {
                    "plan": plan.get("plan"),
                    "diagnosis": plan.get("diagnosis"),
                    "outcome": plan.get("outcome"),
                    "repairCycles": _compact_repair_cycles(
                        plan.get("repairCycles") if isinstance(plan.get("repairCycles"), list) else []
                    ),
                }
                for plan in plans[-MAX_REPLAN_CYCLES:]
                if isinstance(plan, dict)
            ],
        })
    return history


def _discard_candidate(
    source: Path,
    worktree: Path,
    dependency_link: Path | None,
) -> None:
    remove_dependency_link(dependency_link)
    completed = subprocess.run(
        ["git", "-C", str(source), "worktree", "remove", "--force", str(worktree)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        error = str(redact_secrets(completed.stderr))[-2_000:]
        raise ValueError(f"failed to discard rejected repair worktree: {error}")


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
