from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from ai_team.core.bounded_delivery import (
    _validate_contract_validation_commands,
    load_trusted_task_contract,
)
from ai_team.core.external_qa import (
    HUMAN_ATTESTATION_REQUIRED,
    MANUAL_ATTESTATION_ONLY,
    SCHEMA as EXTERNAL_QA_MANUAL_REVIEW_SCHEMA,
)
from ai_team.core.project_loader import load_project


Runner = Callable[..., subprocess.CompletedProcess[str]]
AIRepairer = Callable[..., dict[str, Any]]
CONTRACT_COMMAND_DIAGNOSTIC = (
    "trusted task contract must run the project lint, typecheck, test, and build commands"
)


def requires_manual_review(supervisor: Any) -> bool:
    """Return whether a supervisor snapshot must never be repaired automatically.

    ``nextAction`` is the ordinary supervisor gate.  External QA also emits a
    fixed human-attestation receipt which remains authoritative while the
    supervisor is transiently running or resuming and its next action changes.
    Every receipt field is checked explicitly so malformed external QA data
    cannot suppress a legitimate automatic repair.
    """

    if not isinstance(supervisor, dict):
        return False
    if supervisor.get("nextAction") == "manual-review-required":
        return True
    external_qa = supervisor.get("externalQa")
    return (
        isinstance(external_qa, dict)
        and external_qa.get("schema") == EXTERNAL_QA_MANUAL_REVIEW_SCHEMA
        and external_qa.get("executionMode") == MANUAL_ATTESTATION_ONLY
        and external_qa.get("executionAttempted") is False
        and external_qa.get("status") == "review-required"
        and external_qa.get("reason") == HUMAN_ATTESTATION_REQUIRED
    )


@dataclass(frozen=True)
class AutoRepairOptions:
    enabled: bool = False
    project_path: Path | None = None
    contract_dir: Path | None = None
    backup_dir: Path | None = None
    max_attempts: int = 2
    ai_repair_enabled: bool = False
    orchestrator_path: Path | None = None
    ai_report_dir: Path | None = None
    revive_timer_name: str | None = None
    codex_executable: str = "codex"
    diagnosis_model: str = "gpt-5.6-sol"
    repair_model: str = "gpt-5.6-terra"
    reasoning_effort: str = "high"
    supervisor_state_path: Path | None = None
    antigravity_executable: str = "agy"
    antigravity_qa_model: str = "Gemini 3.1 Pro (High)"
    max_ai_repair_cycles: int = 5


def repair_key(supervisor: dict[str, Any], task_failure_reason: str | None) -> str:
    task = supervisor.get("currentTask")
    task_sha = task.get("taskSha") if isinstance(task, dict) else None
    backlog = supervisor.get("autonomousBacklog")
    backlog_task_sha = backlog.get("taskSha") if isinstance(backlog, dict) else None
    identity = {
        "taskSha": str(task_sha or "no-task"),
        "stopReason": str(supervisor.get("stopReason") or ""),
        "taskFailureReason": str(task_failure_reason or ""),
        "diagnostic": str(supervisor.get("diagnostic") or ""),
        "backlogTaskSha": str(backlog_task_sha or ""),
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"{identity['taskSha']}:{digest}"


def attempt_auto_repair(
    supervisor: dict[str, Any],
    *,
    alert_type: str,
    service_name: str,
    options: AutoRepairOptions,
    runner: Runner = subprocess.run,
    ai_repairer: AIRepairer | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Stop the supervisor, apply one deterministic repair, and restart on success."""

    if requires_manual_review(supervisor):
        return _result(
            False,
            False,
            "manual-review-required",
            "manual review is required; automatic repair is not permitted",
        )
    if not options.enabled:
        return _result(False, False, "disabled", "auto repair is disabled")
    if options.revive_timer_name and not _systemctl(runner, "stop", options.revive_timer_name):
        return _result(True, False, "stop-revive-timer", "failed to stop supervisor revive timer")
    stopped = _systemctl(runner, "stop", service_name)
    if not stopped:
        return _result(True, False, "stop-supervisor", "failed to stop supervisor safely")

    diagnostic = str(supervisor.get("diagnostic") or "")
    if diagnostic == CONTRACT_COMMAND_DIAGNOSTIC:
        repair = _repair_contract_commands(supervisor, options, now=now)
    elif alert_type == "idle-loop":
        repair = _invalidate_stalled_backlog(supervisor, options, now=now)
    elif alert_type in {"service-failed", "stale-state"}:
        repair = _result(True, True, "controlled-restart", "supervisor stopped for a clean restart")
    elif options.ai_repair_enabled:
        if options.project_path is None or options.orchestrator_path is None or options.ai_report_dir is None:
            repair = _result(True, False, "codex-ai-repair", "AI repair paths are not configured")
        else:
            if ai_repairer is None:
                from ai_team.core.watchdog_ai_repair import run_watchdog_ai_repair

                ai_repairer = run_watchdog_ai_repair
            repair = ai_repairer(
                supervisor,
                project_path=options.project_path,
                orchestrator_path=options.orchestrator_path,
                report_dir=options.ai_report_dir,
                codex_executable=options.codex_executable,
                diagnosis_model=options.diagnosis_model,
                repair_model=options.repair_model,
                reasoning_effort=options.reasoning_effort,
                antigravity_executable=options.antigravity_executable,
                antigravity_qa_model=options.antigravity_qa_model,
                max_repair_cycles=options.max_ai_repair_cycles,
                now=now,
            )
    else:
        return _result(
            True,
            False,
            "unsupported",
            "no deterministic repair is registered; supervisor remains stopped",
        )

    if repair["success"] is not True:
        return repair
    if repair.get("deferred") is True:
        deferred = _record_deferred_task(supervisor, repair, options, now=now)
        if deferred["success"] is not True:
            return {**repair, **deferred, "restarted": False}
        repair = {**repair, "deferredState": deferred}
    _systemctl(runner, "reset-failed", service_name)
    if not _systemctl(runner, "start", service_name):
        return {
            **repair,
            "success": False,
            "diagnostic": "repair passed but supervisor restart failed",
        }
    if options.revive_timer_name and not _systemctl(runner, "start", options.revive_timer_name):
        return {
            **repair,
            "success": False,
            "restarted": True,
            "diagnostic": "supervisor restarted but revive timer could not be restored",
        }
    return {**repair, "restarted": True}


def _invalidate_stalled_backlog(
    supervisor: dict[str, Any],
    options: AutoRepairOptions,
    *,
    now: datetime | None,
) -> dict[str, Any]:
    """Invalidate only the orphaned autonomous PM cache, then rescan safely."""

    supervisor_path = options.supervisor_state_path
    backlog = supervisor.get("autonomousBacklog")
    state_value = backlog.get("statePath") if isinstance(backlog, dict) else None
    if supervisor_path is None or not isinstance(state_value, str) or not state_value:
        return _result(True, False, "refresh-autonomous-backlog", "autonomous backlog state is missing")

    try:
        expected = supervisor_path.resolve(strict=True).with_name(
            "autonomous-product-backlog.json"
        )
    except OSError as exc:
        return _result(True, False, "refresh-autonomous-backlog", str(exc)[:300])
    candidate = Path(state_value)
    try:
        if candidate.is_symlink():
            raise ValueError("autonomous backlog state path must not be a symlink")
        resolved = candidate.resolve(strict=True)
        if resolved != expected or not resolved.is_file():
            raise ValueError("autonomous backlog state path is not trusted")
        if resolved.stat().st_size > 1_000_000:
            raise ValueError("autonomous backlog state exceeds size limit")
        state = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ValueError("autonomous backlog state must be a JSON object")
        expected_task_sha = backlog.get("taskSha") if isinstance(backlog, dict) else None
        if (
            state.get("outcome") != "task-created"
            or not isinstance(expected_task_sha, str)
            or state.get("taskSha") != expected_task_sha
        ):
            raise ValueError("autonomous backlog changed before repair")
        repaired_at = (now or datetime.now(UTC)).astimezone(UTC).isoformat()
        state.update({
            "updatedAt": repaired_at,
            "status": "rescan-required",
            "outcome": "rescan-required",
            "invalidatedTaskSha": expected_task_sha,
            "diagnostic": "orphaned terminal task cache invalidated by watchdog",
        })
        _write_bytes_atomic(
            resolved,
            (json.dumps(state, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _result(True, False, "refresh-autonomous-backlog", str(exc)[:300])
    return _result(
        True,
        True,
        "refresh-autonomous-backlog",
        "orphaned PM cache invalidated; supervisor will rescan",
    )


def _record_deferred_task(
    supervisor: dict[str, Any],
    repair: dict[str, Any],
    options: AutoRepairOptions,
    *,
    now: datetime | None,
) -> dict[str, Any]:
    path = options.supervisor_state_path
    task = supervisor.get("currentTask")
    task_sha = task.get("taskSha") if isinstance(task, dict) else None
    if path is None or not isinstance(task_sha, str) or not task_sha:
        return _result(True, False, "defer-task", "supervisor state or current task is missing")
    try:
        resolved = path.resolve(strict=True)
        if resolved.is_symlink() or not resolved.is_file() or resolved.stat().st_size > 2_000_000:
            raise ValueError("supervisor state is not a bounded regular file")
        state = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ValueError("supervisor state must be a JSON object")
        current = state.get("currentTask")
        if not isinstance(current, dict) or current.get("taskSha") != task_sha:
            raise ValueError("supervisor task changed during automatic repair")
        deferred_shas = [
            value for value in state.get("deferredTaskShas", [])
            if isinstance(value, str) and value
        ]
        if task_sha not in deferred_shas:
            deferred_shas.append(task_sha)
        prior_items = [
            item for item in state.get("deferredTasks", [])
            if isinstance(item, dict) and item.get("taskSha") != task_sha
        ]
        deferred_at = (now or datetime.now(UTC)).astimezone(UTC).isoformat()
        prior_items.append({
            "id": current.get("id"),
            "taskSha": task_sha,
            "reason": repair.get("diagnostic"),
            "reportPath": repair.get("reportPath"),
            "deferredAt": deferred_at,
            "attempts": options.max_ai_repair_cycles,
        })
        state.update({
            "revision": int(state.get("revision") or 0) + 1,
            "updatedAt": deferred_at,
            "status": "deferred",
            "deferredTaskShas": deferred_shas[-256:],
            "deferredTasks": prior_items[-256:],
            "stopReason": "repair-cycles-exhausted",
            "nextAction": "next-contract",
            "externalQa": None,
            "diagnostic": str(repair.get("diagnostic") or "")[:500],
        })
        _write_bytes_atomic(
            resolved,
            (json.dumps(state, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _result(True, False, "defer-task", str(exc)[:300])
    return _result(True, True, "defer-task", "failed task recorded and deferred")


def _repair_contract_commands(
    supervisor: dict[str, Any],
    options: AutoRepairOptions,
    *,
    now: datetime | None,
) -> dict[str, Any]:
    if options.project_path is None or options.contract_dir is None or options.backup_dir is None:
        return _result(True, False, "repair-contract", "auto repair paths are not configured")

    task = supervisor.get("currentTask")
    contract_value = task.get("contractPath") if isinstance(task, dict) else None
    if not isinstance(contract_value, str) or not contract_value:
        return _result(True, False, "repair-contract", "current task contract path is missing")

    contract_dir = options.contract_dir.resolve()
    contract_path = Path(contract_value)
    if contract_path.is_symlink():
        return _result(True, False, "repair-contract", "task contract must not be a symlink")
    try:
        contract_path = contract_path.resolve(strict=True)
        contract_path.relative_to(contract_dir)
    except (OSError, ValueError):
        return _result(True, False, "repair-contract", "task contract escapes the configured directory")
    if not contract_path.is_file() or contract_path.stat().st_size > 64_000:
        return _result(True, False, "repair-contract", "task contract is not a bounded regular file")

    try:
        raw = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _result(True, False, "repair-contract", "task contract is unreadable or invalid")
    if not isinstance(raw, dict):
        return _result(True, False, "repair-contract", "task contract must be a JSON object")

    original = contract_path.read_bytes()
    wrote_repair = False
    backup_path: Path | None = None
    try:
        project = load_project(options.project_path)
        commands = project.profile.commands
        baseline = (commands.lint, commands.typecheck, commands.test, commands.build)
        if not all(isinstance(command, str) and command.strip() for command in baseline):
            raise ValueError("project validation command profile is incomplete")
        canonical = list(dict.fromkeys([
            *(str(command).strip() for command in baseline),
            *commands.additional_validation,
        ]))
        repaired = {**raw, "validationCommands": canonical}
        encoded = json.dumps(
            repaired,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > 64_000:
            raise ValueError("repaired task contract exceeds size limit")
        backup_path = _write_backup(
            options.backup_dir,
            contract_path,
            at=now or datetime.now(UTC),
        )
        _write_bytes_atomic(contract_path, encoded)
        wrote_repair = True
        contract, _task_sha = load_trusted_task_contract(contract_path)
        _validate_contract_validation_commands(project, contract)
    except (OSError, ValueError) as exc:
        if wrote_repair:
            try:
                _write_bytes_atomic(contract_path, original)
            except OSError:
                return _result(
                    True,
                    False,
                    "repair-contract",
                    "validation failed and the original contract could not be restored",
                )
        return _result(True, False, "repair-contract", str(exc)[:300])

    return {
        **_result(True, True, "repair-contract", "canonical project validation commands restored"),
        "contractPath": str(contract_path),
        "backupPath": str(backup_path),
        "validationCommands": canonical,
    }


def _write_backup(backup_dir: Path, contract_path: Path, *, at: datetime) -> Path:
    if backup_dir.is_symlink():
        raise ValueError("repair backup directory must not be a symlink")
    root = backup_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("repair backup directory is invalid")
    timestamp = at.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup_path = root / f"{contract_path.name}.{timestamp}.bak"
    backup_path.write_bytes(contract_path.read_bytes())
    os.chmod(backup_path, 0o600)
    return backup_path


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.repair.tmp")
    try:
        temporary.write_bytes(payload)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _systemctl(runner: Runner, action: str, service_name: str) -> bool:
    try:
        completed = runner(
            ["systemctl", "--user", action, service_name],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _result(attempted: bool, success: bool, action: str, diagnostic: str) -> dict[str, Any]:
    return {
        "attempted": attempted,
        "success": success,
        "action": action,
        "diagnostic": diagnostic,
        "restarted": False,
    }
