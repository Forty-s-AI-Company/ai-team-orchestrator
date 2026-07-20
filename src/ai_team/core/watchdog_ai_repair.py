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
from typing import Any, Callable
from uuid import uuid4

from ai_team.core.isolated_executor import (
    prepare_dependency_link,
    remove_dependency_link,
    run_validation_commands,
)
from ai_team.core.project_loader import load_project
from ai_team.providers.antigravity import AntigravityProvider, AntigravitySettings
from ai_team.providers.base import ProviderRequest, redact_secrets


DIAGNOSIS_SCHEMA = "ai-team-watchdog-diagnosis/v1"
QA_SCHEMA = "ai-team-watchdog-qa/v1"
REPORT_SCHEMA = "ai-team-watchdog-ai-repair/v1"
MAX_MODEL_OUTPUT_BYTES = 128_000
MAX_PATCH_BYTES = 512_000
MAX_WRITE_PATHS = 12
MAX_REPAIR_CYCLES = 5
MAX_REPLAN_CYCLES = 5
MAX_REPAIR_HISTORY = 4
MODEL_TIMEOUT_SECONDS = 1_200
CURRENT_REPORT_NAME = "watchdog-ai-repair-current.json"

AGYReviewer = Callable[..., dict[str, Any]]

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


def _run_legacy_watchdog_ai_repair(
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
    antigravity_executable: str = "agy",
    antigravity_qa_model: str = "Gemini 3.1 Pro (High)",
    max_repair_cycles: int = MAX_REPAIR_CYCLES,
    agy_reviewer: AGYReviewer | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run the bounded Sol → Terra → AGY → Sol repair feedback loop."""

    if not 1 <= max_repair_cycles <= MAX_REPAIR_CYCLES:
        raise ValueError(f"max_repair_cycles must be between 1 and {MAX_REPAIR_CYCLES}")
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
        "agyQaModel": antigravity_qa_model,
        "initialReasoningEffort": reasoning_effort,
        "cycleLimit": max_repair_cycles,
        "activeCycle": 0,
        "activePhase": "initializing",
        "cycles": [],
        "supervisorEvidence": redact_secrets(supervisor),
    }
    worktree: Path | None = None
    dependency_link: Path | None = None
    source: Path | None = None
    try:
        _validate_repository(project)
        _validate_repository(orchestrator)
        evidence = _failure_evidence(supervisor, reports)
        diagnosis: dict[str, Any] | None = None
        feedback: list[str] = []
        cycles: list[dict[str, Any]] = report["cycles"]
        reviewer = agy_reviewer or _run_antigravity_qa

        for cycle_number in range(1, max_repair_cycles + 1):
            effort = _cycle_reasoning_effort(cycle_number, reasoning_effort)
            report.update({"activeCycle": cycle_number, "activePhase": "sol-diagnosis"})
            _write_current_report(report, reports)
            diagnosis_result = _invoke_codex(
                codex_executable,
                model=diagnosis_model,
                reasoning_effort=effort,
                root=orchestrator,
                write=False,
                prompt=(
                    _diagnosis_prompt(supervisor, evidence, project, orchestrator)
                    if diagnosis is None
                    else _replan_prompt(
                        supervisor,
                        evidence,
                        project,
                        orchestrator,
                        diagnosis,
                        cycles,
                    )
                ),
            )
            diagnosis = _validate_diagnosis(_last_json_object(diagnosis_result.stdout))
            cycle_result: dict[str, Any] = {
                "cycle": cycle_number,
                "reasoningEffort": effort,
                "diagnosisCommand": _command_summary(diagnosis_result),
                "diagnosis": diagnosis,
                "outcome": "diagnosed",
            }
            cycles.append(cycle_result)
            report["diagnosis"] = diagnosis
            _write_current_report(report, reports)
            if diagnosis["status"] != "repairable":
                cycle_result.update({
                    "outcome": "unrepairable",
                    "failureSummary": diagnosis["summary"],
                })
                return _defer(
                    report,
                    reports,
                    diagnostic=f"Sol 判定目前無法自動修復：{diagnosis['summary']}",
                )

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

            try:
                report["activePhase"] = "terra-repair"
                _write_current_report(report, reports)
                repair_result = _invoke_codex(
                    codex_executable,
                    model=repair_model,
                    reasoning_effort=effort,
                    root=worktree,
                    write=True,
                    prompt=_repair_prompt(diagnosis, allowed_paths, feedback=feedback),
                )
                cycle_result["repairCommand"] = _command_summary(repair_result)
                if _git(worktree, "rev-parse", "HEAD").stdout.strip() != base_sha:
                    raise ValueError("Terra must not create Git commits during the repair stage")
                changed_files = _changed_files(worktree)
                if not changed_files:
                    raise ValueError("Terra completed without producing a repair diff")
                outside = [path for path in changed_files if not _path_allowed(path, allowed_paths)]
                if outside:
                    raise ValueError(
                        f"Terra changed files outside diagnosed scope: {', '.join(outside)}"
                    )

                _git(worktree, "add", "--", *changed_files)
                patch = _git(
                    worktree, "diff", "--cached", "--no-ext-diff", "--binary", "HEAD"
                ).stdout
                if len(patch.encode("utf-8")) > MAX_PATCH_BYTES:
                    raise ValueError("repair patch exceeds the bounded QA evidence limit")
                candidate_hash = hashlib.sha256(patch.encode("utf-8", "replace")).hexdigest()
                cycle_result.update({
                    "changedFiles": changed_files,
                    "patchSha256": candidate_hash,
                })

                report["activePhase"] = "deterministic-qa"
                _write_current_report(report, reports)
                validation = _run_deterministic_qa(repository, source, worktree)
                cycle_result["validation"] = validation
                _assert_candidate_unchanged(
                    worktree, changed_files, candidate_hash, "deterministic QA"
                )
                validation_feedback: list[str] = []
                if validation.get("success") is not True:
                    validation_feedback = _validation_feedback(validation)
                    cycle_result.update({
                        "outcome": "deterministic-qa-failed",
                        "failureSummary": validation_feedback[0],
                    })
                    _write_current_report(report, reports)

                report["activePhase"] = "agy-qa"
                _write_current_report(report, reports)
                agy_qa = reviewer(
                    worktree=worktree,
                    diagnosis=diagnosis,
                    validation=validation,
                    patch=patch,
                    patch_sha=candidate_hash,
                    executable=antigravity_executable,
                    model=antigravity_qa_model,
                )
                cycle_result["agyQa"] = agy_qa
                _assert_candidate_unchanged(worktree, changed_files, candidate_hash, "Antigravity QA")

                report["activePhase"] = "sol-review"
                _write_current_report(report, reports)
                qa_result = _invoke_codex(
                    codex_executable,
                    model=diagnosis_model,
                    reasoning_effort=effort,
                    root=worktree,
                    write=False,
                    prompt=_qa_prompt(diagnosis, validation, agy_qa, patch),
                )
                sol_qa = _validate_qa(_last_json_object(qa_result.stdout))
                cycle_result.update({
                    "solReviewCommand": _command_summary(qa_result),
                    "solReview": sol_qa,
                })
                _assert_candidate_unchanged(
                    worktree, changed_files, candidate_hash, "Codex Sol review"
                )
                agy_passed = (
                    validation.get("success") is True
                    and agy_qa.get("status") == "passed"
                    and not agy_qa.get("findings")
                    and not agy_qa.get("blockers")
                )
                sol_passed = sol_qa["status"] == "passed" and not sol_qa["findings"]
                if agy_passed and sol_passed:
                    cycle_result["outcome"] = "passed"
                    report["activePhase"] = "commit-and-push"
                    _write_current_report(report, reports)
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
                        diagnostic="Sol 診斷、Terra 修正、AGY QA 與 Sol 複檢全部通過",
                    )

                feedback = [
                    *validation_feedback,
                    *[str(item) for item in agy_qa.get("findings", [])],
                    *[str(item) for item in agy_qa.get("blockers", [])],
                    *sol_qa["findings"],
                ] or [str(agy_qa.get("summary") or sol_qa["summary"])]
                cycle_result.update({
                    "outcome": "review-rejected",
                    "failureSummary": "；".join(feedback)[:2_000],
                })
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                feedback = [str(redact_secrets(str(exc)))[:2_000]]
                cycle_result.update({
                    "outcome": "cycle-error",
                    "failureSummary": feedback[0],
                })

            _write_current_report(report, reports)
            if source is not None and worktree is not None:
                _discard_candidate(source, worktree, dependency_link)
            worktree = dependency_link = source = None

        return _defer(
            report,
            reports,
            diagnostic=f"連續 {max_repair_cycles} 輪仍未通過，已記錄並暫緩此任務",
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        report["status"] = "failed"
        report["error"] = str(redact_secrets(str(exc)))[:500]
        return _finish(report, reports, success=False, diagnostic=report["error"])
    finally:
        try:
            remove_dependency_link(dependency_link)
        except OSError:
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


def _cycle_reasoning_effort(cycle_number: int, initial: str) -> str:
    return initial if cycle_number == 1 else "xhigh"


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
        "The previous repair cycle was rejected. Inspect both repositories again at xhigh reasoning.",
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


def _qa_prompt(
    diagnosis: dict[str, Any],
    validation: dict[str, Any],
    agy_qa: dict[str, Any],
    patch: str,
) -> str:
    return "\n".join((
        "You are the read-only QA reviewer for an automatic watchdog repair.",
        "Review the exact untrusted patch, deterministic tests, and independent AGY QA evidence.",
        "Do not edit files or run external actions. AGY rejection must be resolved or explicitly disproved.",
        "Check security, correctness, performance, readability, and maintainability.",
        "Return JSON only without Markdown using exactly:",
        '{"schema":"ai-team-watchdog-qa/v1","status":"passed|failed","summary":"Chinese summary",'
        '"findings":["actionable finding"]}',
        f"Diagnosis={json.dumps(diagnosis, ensure_ascii=False)}",
        f"Validation={json.dumps(redact_secrets(validation), ensure_ascii=False)}",
        f"AGYQA={json.dumps(redact_secrets(agy_qa), ensure_ascii=False)}",
        f"UntrustedPatchJson={json.dumps(patch, ensure_ascii=False)}",
    ))


def _run_antigravity_qa(
    *,
    worktree: Path,
    diagnosis: dict[str, Any],
    validation: dict[str, Any],
    patch: str,
    patch_sha: str,
    executable: str,
    model: str,
) -> dict[str, Any]:
    """Run AGY as a read-only, independently sandboxed repair reviewer."""

    commands = [
        str(item.get("command"))
        for item in validation.get("commands", [])
        if isinstance(item, dict) and isinstance(item.get("command"), str)
    ]
    changed_files = _changed_files(worktree)
    implementation_evidence = {
        "changedFiles": changed_files,
        "validation": {"success": validation.get("success") is True},
        "reviewEvidence": {
            "path": "/tmp/ai-team-review-evidence/patch.diff",
            "sha256": patch_sha,
            "bytes": len(patch.encode("utf-8")),
        },
    }
    prompt = "\n".join((
        f"Task: {diagnosis['summary']}",
        f"Instruction: {diagnosis['repairInstruction']}",
        "Acceptance Criteria: " + json.dumps([
            "修正符合 Sol 診斷的根因與範圍",
            "所有 deterministic QA 都通過",
            "沒有安全性、正確性、效能或可維護性阻擋項目",
        ], ensure_ascii=False),
        f"Allowed Write Paths: {json.dumps(changed_files, ensure_ascii=False)}",
        f"Validation Commands: {json.dumps(commands, ensure_ascii=False)}",
        "Change Policy: " + json.dumps({
            "schema_changes": False,
            "api_contract_changes": False,
            "migration_artifacts": False,
            "fixture_data": False,
        }),
        f"Implementation Evidence: {json.dumps(implementation_evidence, ensure_ascii=False)}",
    ))
    provider = AntigravityProvider(AntigravitySettings(
        executable=executable,
        status_args=["models"],
        quota_args=[],
        run_args=[
            "--model", model,
            "--print-timeout", "120s",
            "--mode", "plan",
            "--sandbox",
            "--print",
        ],
        timeout_seconds=45,
        run_timeout_seconds=MODEL_TIMEOUT_SECONDS,
        execution_enabled=True,
        prompt_max_chars=8192,
        read_only_sandbox_executable="/usr/bin/bwrap",
        allowed_models=(model,),
        allowed_reasoning_efforts=("high",),
    ))
    result = provider.run(ProviderRequest(
        workflow="watchdog-repair-qa",
        prompt=prompt,
        project_root=worktree,
        metadata={
            "boundedStage": "qa",
            "requestedModel": model,
            "reasoningEffort": "high",
            "reviewPatch": patch,
            "reviewPatchSha": patch_sha,
        },
        timeout_seconds=MODEL_TIMEOUT_SECONDS,
        run_mode="read-only",
    ))
    if not result.success:
        return {
            "schema": "ai-team-bounded-delivery/v1",
            "status": "failed",
            "summary": str(redact_secrets(result.content or "AGY QA 執行失敗"))[-2_000:],
            "findings": ["AGY QA 未產生可驗證的通過結果"],
            "tests": [],
            "blockers": [str(result.error_type or "unknown")],
            "provider": result.provider,
        }
    try:
        payload = json.loads(result.content)
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return {
        "schema": payload.get("schema"),
        "status": payload.get("status"),
        "summary": str(payload.get("summary") or "AGY QA 已完成"),
        "findings": payload.get("findings") if isinstance(payload.get("findings"), list) else [],
        "tests": payload.get("tests") if isinstance(payload.get("tests"), list) else [],
        "blockers": payload.get("blockers") if isinstance(payload.get("blockers"), list) else [],
        "provider": result.provider,
    }


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
        qa = (
            item.get("solReview", item.get("qa")) if isinstance(item, dict) else None
        )
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
            "agyQa": item.get("agyQa") if isinstance(item, dict) else None,
            "outcome": item.get("outcome") if isinstance(item, dict) else None,
            "failureSummary": item.get("failureSummary") if isinstance(item, dict) else None,
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
    if report.get("status") != "deferred":
        report["activePhase"] = "completed" if success else "failed"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"watchdog-ai-repair-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    _write_report_atomic(path, report)
    _write_report_atomic(report_dir / CURRENT_REPORT_NAME, report)
    return {
        "attempted": True,
        "success": success,
        "action": "codex-sol-terra-agy-qa-repair",
        "diagnostic": diagnostic,
        "restarted": False,
        "reportPath": str(path),
        "deferred": report.get("status") == "deferred",
        "deferredTaskSha": _report_task_sha(report),
        "repository": report.get("repository"),
        "repairSha": report.get("repairSha"),
        "changedFiles": report.get("changedFiles", []),
    }


def _defer(
    report: dict[str, Any],
    report_dir: Path,
    *,
    diagnostic: str,
) -> dict[str, Any]:
    report.update({
        "status": "deferred",
        "activePhase": "deferred",
        "deferReason": diagnostic,
    })
    # Deferral is a successful controller decision: the candidate was not
    # merged, the failure evidence was preserved, and the queue may continue.
    return _finish(report, report_dir, success=True, diagnostic=diagnostic)


def _report_task_sha(report: dict[str, Any]) -> str | None:
    supervisor = report.get("supervisorEvidence")
    task = supervisor.get("currentTask") if isinstance(supervisor, dict) else None
    value = task.get("taskSha") if isinstance(task, dict) else None
    return value if isinstance(value, str) and value else None


def _write_current_report(report: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_report_atomic(report_dir / CURRENT_REPORT_NAME, report)


def _write_report_atomic(path: Path, report: dict[str, Any]) -> None:
    payload = json.dumps(redact_secrets(report), ensure_ascii=False, indent=2) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
