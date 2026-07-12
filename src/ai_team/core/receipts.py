from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ai_team.core.orchestrator import WorkflowRunResult
from ai_team.core.project_loader import LoadedProject
from ai_team.providers.base import redact_secrets


def write_run_receipt(
    loaded_project: LoadedProject,
    result: WorkflowRunResult,
    receipt_dir: Path,
) -> Path:
    receipt_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).isoformat()
    safe_workflow = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in result.workflow.name)
    timestamp = generated_at.replace(":", "").replace("+", "Z").replace(".", "")
    file_name = f"{timestamp}-{safe_workflow}-{uuid4().hex[:8]}.json"
    path = receipt_dir / file_name
    payload: dict[str, Any] = {
        "schemaVersion": 1,
        "generatedAt": generated_at,
        "projectPath": str(loaded_project.root),
        "branch": loaded_project.current_branch,
        "provider": result.provider_result.provider,
        "workflow": result.workflow.name,
        "stages": result.stages,
        "commitSha": loaded_project.commit_sha,
        "runMode": result.provider_result.data.get("runMode"),
        "startedAt": result.started_at.replace(microsecond=0).isoformat(),
        "completedAt": result.completed_at.replace(microsecond=0).isoformat(),
        "durationMs": result.duration_ms,
        "providerNative": {
            "ready": redact_secrets(result.provider_result.data.get("ready")),
            "conversationId": result.provider_result.conversation_id,
            "taskId": result.provider_result.task_id,
            "executionStatus": result.provider_result.data.get("executionStatus"),
            "runEndpointResult": redact_secrets(result.provider_result.data.get("runEndpointResult")),
            "externalRequired": redact_secrets(result.provider_result.data.get("externalRequired")),
        },
        "validationResult": {
            "success": result.provider_result.success,
            "dryRun": result.dry_run,
            "attempts": result.provider_result.attempts,
            "errorType": result.provider_result.error_type,
        },
        "providerContent": _safe_content(result.provider_result.content),
        "providerData": redact_secrets(result.provider_result.data),
    }
    path.write_text(json.dumps(redact_secrets(payload), indent=2, default=str), encoding="utf-8")
    return path


def _safe_content(content: str) -> str:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        redacted = redact_secrets(content)
    else:
        redacted = json.dumps(redact_secrets(parsed), default=str)
    if not isinstance(redacted, str):
        return ""
    return redacted[:4000]
