"""Read-only PM discovery that turns a development gap into one bounded task.

The continuous supervisor owns execution.  This module only asks the PM for a
single next task when the queue is empty, validates the returned contract, and
atomically places it in the trusted contract directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_team.core.bounded_delivery import load_trusted_task_contract
from ai_team.providers.base import BaseProvider, ProviderRequest, redact_secrets


SCHEMA = "ai-team-autonomous-backlog/v1"
MAX_RESPONSE_BYTES = 48_000
TASK_ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]{2,78}$")


def discover_next_task(
    *,
    project_path: Path,
    contract_dir: Path,
    state_path: Path,
    provider: BaseProvider,
    timeout_seconds: int,
    project_validation_commands: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Generate at most one validated task for the current project revision."""

    revision = _project_revision(project_path)
    prior = _read_json(state_path)
    if prior.get("projectRevision") == revision and prior.get("outcome") in {
        "task-created",
        "no-safe-task",
    }:
        return {
            "status": "unchanged",
            "projectRevision": revision,
            "outcome": prior.get("outcome"),
            "statePath": str(state_path),
        }

    prompt = _discovery_prompt(revision)
    result = provider.run(
        ProviderRequest(
            workflow="autonomous-product-discovery",
            project_root=project_path,
            run_mode="run-agent",
            timeout_seconds=timeout_seconds,
            prompt=prompt,
            metadata={
                "role": "product-manager",
                "writeRequired": False,
                "writeAccess": False,
                "projectRevision": revision,
                # Antigravity natively validates this audited PM envelope;
                # the backlog-specific fields remain at the envelope's top
                # level so they cannot disappear inside an optional wrapper.
                "boundedStage": "pm",
            },
        )
    )
    if not result.success or result.provider in {"mock", ""}:
        return _persist(
            state_path,
            {
                "outcome": "provider-failed",
                "projectRevision": revision,
                "provider": result.provider,
                "errorType": str(result.error_type) if result.error_type else None,
            },
            status="provider-failed",
        )

    try:
        payload = _parse_payload(result.content)
    except ValueError as exc:
        return _persist(
            state_path,
            {
                "outcome": "invalid-response",
                "projectRevision": revision,
                "provider": result.provider,
                "diagnostic": str(exc),
            },
            status="invalid-response",
        )

    if payload["status"] == "ready":
        return _persist(
            state_path,
            {
                "outcome": "no-safe-task",
                "projectRevision": revision,
                "provider": result.provider,
                "summary": payload["summary"],
            },
            status="no-safe-task",
        )

    try:
        task_path, task_sha, task_id = _write_validated_contract(
            contract_dir,
            payload["contract"],
            revision=revision,
            project_validation_commands=project_validation_commands,
        )
    except (OSError, ValueError) as exc:
        return _persist(
            state_path,
            {
                "outcome": "contract-rejected",
                "projectRevision": revision,
                "provider": result.provider,
                "diagnostic": str(exc),
            },
            status="contract-rejected",
        )
    return _persist(
        state_path,
        {
            "outcome": "task-created",
            "projectRevision": revision,
            "provider": result.provider,
            "taskId": task_id,
            "taskSha": task_sha,
            "contractPath": str(task_path),
            "summary": payload["summary"],
        },
        status="task-created",
    )


def _discovery_prompt(revision: str) -> str:
    return "\n".join((
        "You are the read-only product manager for an autonomous development team.",
        f"Project Git revision: {revision}",
        "Inspect the repository, tests, recent commits, project docs, and existing product gaps.",
        "Choose exactly one small, high-value, independently testable next development task.",
        "The task must be safe for a disposable development worktree and must not require a human business decision.",
        "Never propose production deployment, live payments, secrets, real customer data, destructive actions, or external account changes.",
        "If the project is release-candidate ready or no safe/clear coding task exists, return status=ready.",
        "Return the provider-native PM envelope with schema=ai-team-bounded-delivery/v1, stage=pm,",
        "the runtime-supplied challenge, status=passed, and empty findings/tests/blockers arrays.",
        "Also provide backlogStatus=task|ready and a short Chinese summary at the top level.",
        "For status=task, contract must be a schemaVersion=1 trusted task contract with source.kind=trusted-contract,",
        "a lowercase hyphenated id beginning with auto-, no dependsOn, safe project-relative allowedWritePaths,",
        "and validationCommands containing npm run lint, npm run typecheck, npm run test, and npm run build.",
        "Use changePolicy only when the code task genuinely needs it; never request execution of migrations, seeds, deploys, or payments.",
    ))


def _parse_payload(content: str) -> dict[str, Any]:
    encoded = content.encode("utf-8")
    if len(encoded) > MAX_RESPONSE_BYTES:
        raise ValueError("autonomous PM response exceeds size limit")
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("autonomous PM did not return JSON") from exc
    if isinstance(value, dict) and value.get("schema") == "ai-team-bounded-delivery/v1":
        value = {
            "schema": SCHEMA,
            "status": value.get("backlogStatus"),
            "summary": value.get("summary"),
            "contract": value.get("contract"),
        }
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("autonomous PM returned an invalid schema")
    status = value.get("status")
    summary = value.get("summary")
    if status not in {"task", "ready"} or not isinstance(summary, str) or not summary.strip():
        raise ValueError("autonomous PM response is missing status or summary")
    if status == "ready":
        return {"status": status, "summary": summary.strip()}
    contract = value.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("autonomous PM task response is missing a contract")
    return {"status": status, "summary": summary.strip(), "contract": contract}


def _write_validated_contract(
    contract_dir: Path,
    raw_contract: dict[str, Any],
    *,
    revision: str,
    project_validation_commands: tuple[str, ...] = (),
) -> tuple[Path, str, str]:
    contract_dir.mkdir(parents=True, exist_ok=True)
    if contract_dir.is_symlink() or not contract_dir.is_dir():
        raise ValueError("autonomous contract directory is invalid")
    if raw_contract.get("dependsOn") not in (None, []):
        raise ValueError("autonomous tasks may not declare dependencies")

    sanitized = dict(raw_contract)
    task_id = _normalize_task_id(sanitized.get("id"), sanitized)
    sanitized["id"] = task_id
    sanitized["source"] = {
        "kind": "trusted-contract",
        "reference": f"autonomous project scan at {revision}",
    }
    sanitized["dependsOn"] = []
    if project_validation_commands:
        sanitized["validationCommands"] = _validated_project_commands(project_validation_commands)
    encoded = json.dumps(sanitized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > 64_000:
        raise ValueError("autonomous task contract exceeds size limit")
    fingerprint = hashlib.sha256(encoded).hexdigest()
    final_path = contract_dir / f"900-{task_id}-{fingerprint[:12]}.json"
    if final_path.exists():
        contract, task_sha = load_trusted_task_contract(final_path)
        return final_path, task_sha, contract.id

    temporary = contract_dir / f".{task_id}-{fingerprint[:12]}.tmp"
    try:
        temporary.write_bytes(encoded)
        contract, task_sha = load_trusted_task_contract(temporary)
        if contract.id != task_id or contract.depends_on:
            raise ValueError("autonomous task contract changed during validation")
        os.replace(temporary, final_path)
    finally:
        temporary.unlink(missing_ok=True)
    return final_path, task_sha, task_id


def _validated_project_commands(commands: tuple[str, ...]) -> list[str]:
    """Return unique trusted profile commands for an autonomous contract.

    The PM may suggest narrower commands, but the project profile is the
    authority for bounded delivery. Replacing model output here prevents a
    malformed autonomous contract from entering a systemd restart loop.
    """

    normalized: list[str] = []
    for command in commands:
        if not isinstance(command, str) or not command.strip():
            raise ValueError("project validation commands must be non-empty strings")
        value = command.strip()
        if value not in normalized:
            normalized.append(value)
    return normalized


def _normalize_task_id(raw_id: Any, contract: dict[str, Any]) -> str:
    """Make a model-proposed label safe for the trusted contract filename.

    IDs are bookkeeping only; the validated title and instruction remain the
    actual task content. A deterministic hash fallback also prevents one
    malformed label from stalling the autonomous loop.
    """
    raw = raw_id if isinstance(raw_id, str) else ""
    normalized = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    if normalized and not normalized.startswith("auto-"):
        normalized = f"auto-{normalized}"
    if TASK_ID_PATTERN.fullmatch(normalized):
        return normalized
    fingerprint = hashlib.sha256(
        json.dumps(contract, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    return f"auto-task-{fingerprint}"


def _project_revision(project_path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_path,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        result = None
    if result is not None and result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return "workspace-" + hashlib.sha256(str(project_path.resolve()).encode("utf-8")).hexdigest()[:16]


def _persist(state_path: Path, payload: dict[str, Any], *, status: str) -> dict[str, Any]:
    state = {
        "schemaVersion": 1,
        "updatedAt": datetime.now(UTC).isoformat(),
        "status": status,
        **redact_secrets(payload),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_suffix(state_path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    temporary.replace(state_path)
    return {**state, "statePath": str(state_path)}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}
