"""Fail-closed, deterministic staging-only external operations.

This module intentionally does not reuse the LLM executor or bounded delivery
contract.  It accepts a narrow contract and resolves every executable from a
constant allowlist, so a model cannot smuggle arbitrary shell through a task.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable
from uuid import uuid4

from ai_team.core.project_loader import LoadedProject, ProjectConfigError, load_project
from ai_team.providers.base import redact_secrets


SCHEMA = "ai-team-staging-operations/v1"
RECEIPT_SCHEMA = "ai-team-staging-operation-receipt/v1"
OPERATION_COMMANDS: dict[str, tuple[str, ...]] = {
    "database-migration": ("npm", "run", "db:migrate:deploy"),
    "database-seed": ("npm", "run", "db:seed"),
    "preview-deploy": ("vercel", "deploy", "--yes"),
}
OPERATION_POLICY_FLAG = {
    "database-migration": "allow_migration",
    "database-seed": "allow_seed",
    "preview-deploy": "allow_preview_deploy",
}


class StagingOperationsError(ValueError):
    """Raised for a staging-only operation that must stop fail-closed."""


@dataclass(frozen=True)
class StagingOperationsContract:
    id: str
    title: str
    source_kind: str
    source_reference: str
    environment: str
    deployment: str
    operations: tuple[str, ...]


@dataclass(frozen=True)
class CommandOutcome:
    operation: str
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class StagingOperationsResult:
    success: bool
    stop_reason: str | None
    validation_kind: str
    contract_sha: str | None
    receipt_path: Path
    outcomes: tuple[CommandOutcome, ...]


CommandRunner = Callable[[tuple[str, ...], Path, dict[str, str]], subprocess.CompletedProcess[str]]


def load_staging_operations_contract(path: Path) -> tuple[StagingOperationsContract, str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StagingOperationsError(f"staging contract must be readable JSON: {path}") from exc
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise StagingOperationsError(f"staging contract requires schema={SCHEMA}")

    source = raw.get("source")
    target = raw.get("target")
    if not isinstance(source, dict) or source.get("kind") not in {"github-issue", "trusted-contract"}:
        raise StagingOperationsError("staging contract source must be github-issue or trusted-contract")
    if not isinstance(target, dict):
        raise StagingOperationsError("staging contract requires a target mapping")
    values = {
        "id": raw.get("id"),
        "title": raw.get("title"),
        "source_kind": source.get("kind"),
        "source_reference": source.get("reference"),
        "environment": target.get("environment"),
        "deployment": target.get("deployment"),
    }
    if not all(isinstance(value, str) and value.strip() for value in values.values()):
        raise StagingOperationsError("staging contract requires non-empty id, title, source, and target values")
    if values["environment"].strip().lower() != "staging" or values["deployment"].strip().lower() != "preview":
        raise StagingOperationsError("staging contract target must be environment=staging and deployment=preview")

    operations_raw = raw.get("operations")
    if not isinstance(operations_raw, list) or not operations_raw:
        raise StagingOperationsError("staging contract requires a non-empty operations list")
    operations = tuple(item.strip() for item in operations_raw if isinstance(item, str) and item.strip())
    if len(operations) != len(operations_raw) or len(set(operations)) != len(operations):
        raise StagingOperationsError("staging contract operations must be non-empty, unique strings")
    unknown = sorted(set(operations) - set(OPERATION_COMMANDS))
    if unknown:
        raise StagingOperationsError(f"staging contract contains unsupported operation(s): {', '.join(unknown)}")

    normalized = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return (
        StagingOperationsContract(
            id=values["id"].strip(),
            title=values["title"].strip(),
            source_kind=values["source_kind"].strip(),
            source_reference=values["source_reference"].strip(),
            environment="staging",
            deployment="preview",
            operations=operations,
        ),
        hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    )


def run_staging_operations(
    project_path: Path,
    contract_path: Path,
    report_dir: Path,
    *,
    execute: bool = False,
    workspace_allowlist: list[str] | None = None,
    runner: CommandRunner | None = None,
) -> StagingOperationsResult:
    """Validate and optionally execute a fixed staging-only operation list.

    A receipt is emitted for every result, including malformed contracts and
    failed subprocesses.  ``execute=False`` is intentionally non-invasive and
    still validates the policy and target guards.
    """

    report_dir.mkdir(parents=True, exist_ok=True)
    loaded: LoadedProject | None = None
    contract: StagingOperationsContract | None = None
    contract_sha: str | None = None
    outcomes: list[CommandOutcome] = []
    validation_kind = "policy-validation"
    stop_reason: str | None = None
    success = False
    try:
        loaded = load_project(project_path, allowlist=workspace_allowlist)
        contract, contract_sha = load_staging_operations_contract(contract_path)
        _validate_policy(loaded, contract)
        _validate_staging_target(loaded, contract)
        if _working_tree_dirty(loaded.root):
            raise StagingOperationsError("project-worktree-dirty")
        validation_kind = "target-validation"
        if execute:
            validation_kind = "command-execution"
            command_runner = runner or _run_command
            for operation in contract.operations:
                command = OPERATION_COMMANDS[operation]
                completed = command_runner(command, loaded.root, _safe_command_environment(operation))
                outcome = CommandOutcome(
                    operation=operation,
                    command=command,
                    returncode=completed.returncode,
                    stdout=_bounded_redacted(completed.stdout),
                    stderr=_bounded_redacted(completed.stderr),
                )
                outcomes.append(outcome)
                if _working_tree_dirty(loaded.root):
                    raise StagingOperationsError("staging-operation-modified-project-worktree")
                if completed.returncode != 0:
                    raise StagingOperationsError(f"{operation}-command-failed")
        success = True
        validation_kind = "passed" if execute else "dry-run-validation"
    except (ProjectConfigError, StagingOperationsError, OSError, subprocess.TimeoutExpired) as exc:
        stop_reason = _stable_stop_reason(exc)
    receipt_path = _write_receipt(
        report_dir=report_dir,
        loaded=loaded,
        contract=contract,
        contract_sha=contract_sha,
        execute=execute,
        success=success,
        stop_reason=stop_reason,
        validation_kind=validation_kind,
        outcomes=outcomes,
    )
    return StagingOperationsResult(
        success=success,
        stop_reason=stop_reason,
        validation_kind=validation_kind,
        contract_sha=contract_sha,
        receipt_path=receipt_path,
        outcomes=tuple(outcomes),
    )


def _validate_policy(loaded: LoadedProject, contract: StagingOperationsContract) -> None:
    policy = loaded.profile.staging_operations
    if not policy.enabled:
        raise StagingOperationsError("staging-operations-disabled")
    if policy.environment.lower() != "staging":
        raise StagingOperationsError("staging-policy-environment-invalid")
    for operation in contract.operations:
        if not getattr(policy, OPERATION_POLICY_FLAG[operation]):
            raise StagingOperationsError(f"{operation}-not-authorized")
    # The legacy production flags must remain closed; this separate policy is
    # not a backdoor for production-capable workflows.
    safety = loaded.profile.safety
    if safety.allow_deploy or safety.allow_database_migration or safety.allow_database_seed:
        raise StagingOperationsError("production-safety-flags-must-remain-disabled")


def _validate_staging_target(loaded: LoadedProject, contract: StagingOperationsContract) -> None:
    policy = loaded.profile.staging_operations
    if contract.environment != "staging" or contract.deployment != "preview":
        raise StagingOperationsError("target-is-not-staging-preview")
    if "database-migration" in contract.operations or "database-seed" in contract.operations:
        if not policy.database_url_sha256:
            raise StagingOperationsError("staging-database-fingerprint-required")
        database_url = os.environ.get(policy.database_url_env)
        if not database_url:
            raise StagingOperationsError("staging-database-environment-missing")
        fingerprint = hashlib.sha256(database_url.encode("utf-8")).hexdigest()
        if not _constant_time_equal(fingerprint, policy.database_url_sha256):
            raise StagingOperationsError("staging-database-fingerprint-mismatch")
    if "preview-deploy" in contract.operations:
        preview_value = os.environ.get(policy.preview_environment_variable, "").strip().lower()
        if preview_value != "preview":
            raise StagingOperationsError("preview-environment-attestation-required")


def _run_command(command: tuple[str, ...], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )


def _safe_command_environment(operation: str) -> dict[str, str]:
    """Pass through secrets without logging them and force the minimal seed mode."""
    environment = dict(os.environ)
    # A caller cannot accidentally reuse a production bootstrap setting when
    # the only approved seed class is the minimal staging fixture.
    if operation == "database-seed":
        environment["SEED_MODE"] = "demo"
    # `vercel deploy` without `--prod` is the CLI's Preview command. Remove a
    # potentially inherited target override rather than trusting it.
    if operation == "preview-deploy":
        environment.pop("VERCEL_TARGET", None)
    return environment


def _working_tree_dirty(project_root: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise StagingOperationsError("project-worktree-status-unavailable") from exc
    if result.returncode != 0:
        raise StagingOperationsError("project-worktree-status-unavailable")
    return bool(result.stdout.strip())


def _constant_time_equal(left: str, right: str) -> bool:
    # hashlib.compare_digest is unavailable; this avoids a bespoke secret
    # comparison and keeps the value out of receipts and error messages.
    import hmac

    return hmac.compare_digest(left, right)


def _stable_stop_reason(exc: Exception) -> str:
    message = str(exc).strip()
    if message.endswith("-command-failed"):
        return message
    known = {
        "staging-operations-disabled",
        "staging-policy-environment-invalid",
        "production-safety-flags-must-remain-disabled",
        "staging-database-fingerprint-required",
        "staging-database-environment-missing",
        "staging-database-fingerprint-mismatch",
        "preview-environment-attestation-required",
        "target-is-not-staging-preview",
        "project-worktree-dirty",
        "project-worktree-status-unavailable",
        "staging-operation-modified-project-worktree",
    }
    if message in known or message.endswith("-not-authorized"):
        return message
    if isinstance(exc, subprocess.TimeoutExpired):
        return "staging-operation-timeout"
    return "staging-operation-validation-failed"


def _bounded_redacted(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    safe = redact_secrets(value or "")
    return safe[:4000] if isinstance(safe, str) else ""


def _write_receipt(
    *,
    report_dir: Path,
    loaded: LoadedProject | None,
    contract: StagingOperationsContract | None,
    contract_sha: str | None,
    execute: bool,
    success: bool,
    stop_reason: str | None,
    validation_kind: str,
    outcomes: list[CommandOutcome],
) -> Path:
    generated_at = datetime.now(UTC).isoformat()
    file_name = f"{generated_at.replace(':', '').replace('+', 'Z').replace('.', '')}-staging-operations-{uuid4().hex[:8]}.json"
    payload = {
        "schema": RECEIPT_SCHEMA,
        "generatedAt": generated_at,
        "operationExecutor": "deterministic-staging-operations",
        "providerExecution": {"success": True, "provider": "deterministic"},
        "projectPath": str(loaded.root) if loaded else None,
        "projectCommitSha": loaded.commit_sha if loaded else None,
        "contract": asdict(contract) if contract else None,
        "contractSha": contract_sha,
        "target": {"environment": "staging", "deployment": "preview"},
        "execute": execute,
        "commands": [
            {
                "operation": outcome.operation,
                "command": list(outcome.command),
                "commandSha256": hashlib.sha256("\0".join(outcome.command).encode("utf-8")).hexdigest(),
                "returnCode": outcome.returncode,
                "stdout": outcome.stdout,
                "stderr": outcome.stderr,
            }
            for outcome in outcomes
        ],
        "validationResult": {
            "success": success,
            "kind": validation_kind,
            "stopReason": stop_reason,
            "targetEnvironment": "staging",
            "targetDeployment": "preview",
            "databaseFingerprintValidated": bool(
                contract and any(item.startswith("database-") for item in contract.operations) and success
            ),
        },
    }
    path = report_dir / file_name
    path.write_text(json.dumps(redact_secrets(payload), indent=2, default=str), encoding="utf-8")
    return path
