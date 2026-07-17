from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import BaseProvider, ProviderErrorType, ProviderRequest, ProviderResult, redact_secrets


@dataclass(frozen=True)
class HandsFreeCodeSettings:
    base_url: str = "http://127.0.0.1:31025"
    session_key_env: str = "HANDSFREECODE_SESSION_API_KEY"
    session_key_file: str | None = "~/.handsfreecode/session-api-key.txt"
    ready_path: str = "/ready"
    task_run_path: str = "/api/tasks/run"
    timeout_seconds: float = 30
    default_runtime_provider: str = "mock"


class HandsFreeCodeProvider(BaseProvider):
    name = "handsfreecode"

    def __init__(self, settings: HandsFreeCodeSettings | None = None, session_key: str | None = None) -> None:
        self.settings = settings or HandsFreeCodeSettings()
        self._session_key = session_key

    @property
    def session_key(self) -> str | None:
        return self._session_key or os.environ.get(self.settings.session_key_env) or self._session_key_from_file()

    def _session_key_from_file(self) -> str | None:
        if not self.settings.session_key_file:
            return None
        key_file = Path(os.path.expandvars(self.settings.session_key_file)).expanduser()
        if not key_file.exists() or not key_file.is_file():
            return None
        value = key_file.read_text(encoding="utf-8").strip()
        return value or None

    def ready(self) -> bool:
        diagnostics = self.diagnostics()
        return (
            diagnostics["ready"] is True
            and diagnostics.get("authConfigured") is True
            and diagnostics.get("sessionKeyPresent") is True
        )

    def diagnostics(self) -> dict[str, Any]:
        session_key_present = bool(self.session_key)
        result: dict[str, Any] = {
            "baseUrl": self.settings.base_url,
            "readyPath": self.settings.ready_path,
            "sessionKeyEnv": self.settings.session_key_env,
            "sessionKeyFileConfigured": bool(self.settings.session_key_file),
            "sessionKeyPresent": session_key_present,
            "failClosed": not session_key_present,
            "ready": False,
            "authConfigured": False,
            "providerNative": True,
            "errorType": None,
            "message": None,
        }
        try:
            request = urllib.request.Request(self._url(self.settings.ready_path), method="GET")
            with urllib.request.urlopen(request, timeout=5) as response:
                body = response.read().decode("utf-8", errors="replace")
                data = _decode_json(body)
                result["status"] = response.status
                result["ready"] = 200 <= response.status < 300
                if isinstance(data, dict):
                    result["authConfigured"] = bool(data.get("authConfigured"))
                    result["providers"] = redact_secrets(data.get("providers"))
                    result["response"] = redact_secrets(data)
        except urllib.error.HTTPError as exc:
            result["errorType"] = _http_error_type(exc.code)
            result["status"] = exc.code
            result["message"] = redact_secrets(exc.read().decode("utf-8", errors="replace"))
        except (TimeoutError, socket.timeout) as exc:
            result["errorType"] = ProviderErrorType.TIMEOUT
            result["message"] = str(exc)
        except urllib.error.URLError as exc:
            result["errorType"] = ProviderErrorType.NETWORK
            result["message"] = str(exc.reason)
        except Exception as exc:
            result["errorType"] = ProviderErrorType.UNKNOWN
            result["message"] = str(exc)
        return result

    def run(self, request: ProviderRequest) -> ProviderResult:
        try:
            headers = self._headers()
        except RuntimeError as exc:
            return ProviderResult(
                provider=self.name,
                success=False,
                error_type=ProviderErrorType.AUTH,
                content=str(exc),
                data={
                    "runMode": request.run_mode,
                    "externalRequired": {
                        "type": "session_key",
                        "env": self.settings.session_key_env,
                        "message": "Set a local HandsFreeCode session key before provider-native runs.",
                    },
                },
            )

        ready_result = self.diagnostics()
        if not ready_result.get("ready"):
            diagnostics_error = ready_result.get("errorType")
            error_type = (
                diagnostics_error
                if isinstance(diagnostics_error, ProviderErrorType)
                else ProviderErrorType.EXTERNAL_REQUIRED
            )
            return ProviderResult(
                provider=self.name,
                success=False,
                error_type=error_type,
                content=str(ready_result.get("message") or "HandsFreeCode is not ready"),
                data={
                    "runMode": request.run_mode,
                    "ready": redact_secrets(ready_result),
                    "externalRequired": {
                        "type": "handsfreecode_loopback",
                        "baseUrl": self.settings.base_url,
                        "message": "Start HandsFreeCode loopback before provider-native runs.",
                    },
                },
            )

        if request.run_mode == "read-only-agent":
            if self.settings.default_runtime_provider != "ollama":
                return ProviderResult(
                    provider=self.name,
                    success=False,
                    error_type=ProviderErrorType.EXTERNAL_REQUIRED,
                    content="read-only-agent requires the native Ollama runtime",
                    data={"runMode": request.run_mode, "runtimeMode": "run-agent", "writeAccess": False},
                )
            providers = ready_result.get("providers")
            if not isinstance(providers, dict) or providers.get("ollama") is not True:
                return ProviderResult(
                    provider=self.name,
                    success=False,
                    error_type=ProviderErrorType.EXTERNAL_REQUIRED,
                    content="read-only-agent requires Ollama to be ready",
                    data={
                        "runMode": request.run_mode,
                        "runtimeMode": "run-agent",
                        "writeAccess": False,
                        "ready": redact_secrets(ready_result),
                    },
                )

        payload = self._task_payload(request)
        http_request = urllib.request.Request(
            self._url(self.settings.task_run_path),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                http_request,
                timeout=request.timeout_seconds or self.settings.timeout_seconds,
            ) as response:
                response_body = response.read().decode("utf-8", errors="replace")
                data = _decode_json(response_body)
                ids = _extract_ids(data)
                success = 200 <= response.status < 300 and _response_success(data)
                error_type = _response_error_type(data) if not success else None
                return ProviderResult(
                    provider=self.name,
                    success=success,
                    error_type=error_type,
                    content=response_body,
                    data={
                        "status": response.status,
                        "runMode": request.run_mode,
                        "ready": redact_secrets(ready_result),
                        "conversationId": ids.get("conversationId"),
                        "taskId": ids.get("taskId"),
                        "executionStatus": _execution_status(data),
                        "runtimeProvider": _runtime_provider(data),
                        "runtimeMode": _runtime_mode(data, request),
                        "writeAccess": _write_access(data),
                        "tokenUsage": _token_usage(data),
                        "receiptPath": _receipt_path(data),
                        "response": redact_secrets(data),
                    },
                )
        except urllib.error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            return ProviderResult(
                provider=self.name,
                success=False,
                error_type=_http_error_type(exc.code),
                content=redact_secrets(response_body),
                data={"status": exc.code, "runMode": request.run_mode, "ready": redact_secrets(ready_result)},
            )
        except (TimeoutError, socket.timeout) as exc:
            return ProviderResult(
                provider=self.name,
                success=False,
                error_type=ProviderErrorType.TIMEOUT,
                content=str(exc),
                data={"runMode": request.run_mode},
            )
        except urllib.error.URLError as exc:
            return ProviderResult(
                provider=self.name,
                success=False,
                error_type=ProviderErrorType.NETWORK,
                content=str(exc.reason),
                data={"runMode": request.run_mode},
            )

    def _headers(self) -> dict[str, str]:
        key = self.session_key
        if not key:
            raise RuntimeError(f"missing {self.settings.session_key_env}; HandsFreeCode provider fails closed")
        return {
            "Content-Type": "application/json",
            "X-Session-API-Key": key,
        }

    def _url(self, path: str) -> str:
        base = self.settings.base_url.rstrip("/")
        suffix = path if path.startswith("/") else f"/{path}"
        return f"{base}{suffix}"

    def _task_payload(self, request: ProviderRequest) -> dict[str, Any]:
        write_access = bool(
            request.run_mode == "run-agent"
            and request.metadata.get("writeRequired") is True
            and request.metadata.get("writeAccess") is True
        )
        return {
            "projectPath": str(request.project_root),
            "workflow": request.workflow,
            "prompt": _safe_prompt_for_runtime(request.prompt),
            "provider": self.settings.default_runtime_provider,
            "mode": request.run_mode,
            "maxIterations": 2 if write_access else 1,
            "writeAccess": write_access,
            "metadata": {
                **request.metadata,
                "source": "ai-team-orchestrator",
                "dryRun": request.dry_run,
                "outerRunMode": request.run_mode,
            },
        }


def _decode_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _http_error_type(status: int) -> ProviderErrorType:
    if status in {401, 403}:
        return ProviderErrorType.AUTH
    if status == 429:
        return ProviderErrorType.RATE_LIMIT
    if status == 503:
        return ProviderErrorType.EXTERNAL_REQUIRED
    if status >= 500:
        return ProviderErrorType.NETWORK
    return ProviderErrorType.INVALID_RESPONSE


def _extract_ids(data: Any) -> dict[str, str | None]:
    if not isinstance(data, dict):
        return {"conversationId": None, "taskId": None}
    conversation_id = data.get("conversationId") or data.get("conversation_id")
    task_id = data.get("taskId") or data.get("task_id")
    return {
        "conversationId": str(conversation_id) if conversation_id else None,
        "taskId": str(task_id) if task_id else None,
    }


def _response_success(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    return str(data.get("status") or "").lower() == "completed"


def _response_error_type(data: Any) -> ProviderErrorType | None:
    if not isinstance(data, dict):
        return ProviderErrorType.INVALID_RESPONSE
    value = data.get("errorType") or data.get("error_type")
    if not value:
        return ProviderErrorType.UNKNOWN
    try:
        return ProviderErrorType(str(value))
    except ValueError:
        return ProviderErrorType.UNKNOWN


def _execution_status(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    value = data.get("status")
    return str(value) if value else None


def _runtime_provider(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    value = data.get("provider")
    return str(value) if value else None


def _receipt_path(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    value = data.get("receiptPath") or data.get("receipt_path")
    return str(value) if value else None


def _runtime_mode(data: Any, request: ProviderRequest) -> str:
    if isinstance(data, dict):
        value = data.get("runtimeMode") or data.get("runtime_mode")
        if value:
            return str(value)
    return "run-agent" if request.run_mode == "read-only-agent" else request.run_mode


def _write_access(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    return data.get("writeAccess") is True or data.get("write_access") is True


def _token_usage(data: Any) -> int:
    if not isinstance(data, dict):
        return 0
    value = data.get("tokenUsage") or data.get("token_usage")
    return value if isinstance(value, int) and value >= 0 else 0


def _safe_prompt_for_runtime(prompt: str) -> str:
    replacements = {
        "production deploy": "high-risk deployment",
        "real payment": "high-risk payment action",
        "destructive migration": "high-risk database migration",
    }
    safe_prompt = prompt
    for risky, replacement in replacements.items():
        safe_prompt = safe_prompt.replace(risky, replacement)
    return safe_prompt
