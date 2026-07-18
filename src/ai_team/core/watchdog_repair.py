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
from ai_team.core.project_loader import load_project


Runner = Callable[..., subprocess.CompletedProcess[str]]
CONTRACT_COMMAND_DIAGNOSTIC = (
    "trusted task contract must run the project lint, typecheck, test, and build commands"
)


@dataclass(frozen=True)
class AutoRepairOptions:
    enabled: bool = False
    project_path: Path | None = None
    contract_dir: Path | None = None
    backup_dir: Path | None = None
    max_attempts: int = 2


def repair_key(supervisor: dict[str, Any], task_failure_reason: str | None) -> str:
    task = supervisor.get("currentTask")
    task_sha = task.get("taskSha") if isinstance(task, dict) else None
    identity = {
        "taskSha": str(task_sha or "no-task"),
        "stopReason": str(supervisor.get("stopReason") or ""),
        "taskFailureReason": str(task_failure_reason or ""),
        "diagnostic": str(supervisor.get("diagnostic") or ""),
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
    now: datetime | None = None,
) -> dict[str, Any]:
    """Stop the supervisor, apply one deterministic repair, and restart on success."""

    if not options.enabled:
        return _result(False, False, "disabled", "auto repair is disabled")
    stopped = _systemctl(runner, "stop", service_name)
    if not stopped:
        return _result(True, False, "stop-supervisor", "failed to stop supervisor safely")

    diagnostic = str(supervisor.get("diagnostic") or "")
    if diagnostic == CONTRACT_COMMAND_DIAGNOSTIC:
        repair = _repair_contract_commands(supervisor, options, now=now)
    elif alert_type in {"service-failed", "stale-state"}:
        repair = _result(True, True, "controlled-restart", "supervisor stopped for a clean restart")
    else:
        return _result(
            True,
            False,
            "unsupported",
            "no deterministic repair is registered; supervisor remains stopped",
        )

    if repair["success"] is not True:
        return repair
    _systemctl(runner, "reset-failed", service_name)
    if not _systemctl(runner, "start", service_name):
        return {
            **repair,
            "success": False,
            "diagnostic": "repair passed but supervisor restart failed",
        }
    return {**repair, "restarted": True}


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
