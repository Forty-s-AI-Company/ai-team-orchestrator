from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import BaseProvider, ProviderErrorType, ProviderRequest, ProviderResult, redact_secrets


@dataclass(frozen=True)
class OpenHandsSettings:
    base_url: str = "http://127.0.0.1:31024"
    session_key_env: str = "SESSION_API_KEY"
    session_key_file: str | None = None
    ready_path: str = "/ready"
    conversation_path: str = "/api/conversations"
    cancel_path_template: str = "/api/conversations/{task_id}/interrupt"
    timeout_seconds: float = 30
    host_workspace_root: str = "C:/Users/eden/Downloads/AI"
    container_workspace_root: str = "/projects"
    llm_model: str = "openai/gpt-5.5"
    llm_api_key_env: str = "OPENHANDS_LLM_API_KEY"
    llm_api_key: str = "placeholder-not-a-real-secret"


class OpenHandsProvider(BaseProvider):
    name = "openhands"

    def __init__(self, settings: OpenHandsSettings | None = None, session_key: str | None = None) -> None:
        self.settings = settings or OpenHandsSettings()
        self._session_key = session_key

    @property
    def session_key(self) -> str | None:
        if self._session_key:
            return self._session_key
        env_value = os.environ.get(self.settings.session_key_env)
        if env_value:
            return env_value
        if self.settings.session_key_file:
            return _read_secret_file(self.settings.session_key_file)
        return None

    def _url(self, path: str) -> str:
        base = self.settings.base_url.rstrip("/")
        suffix = path if path.startswith("/") else f"/{path}"
        return f"{base}{suffix}"

    def _headers(self) -> dict[str, str]:
        key = self.session_key
        if not key:
            raise RuntimeError(f"missing {self.settings.session_key_env}; OpenHands provider fails closed")
        return {
            "Content-Type": "application/json",
            "X-Session-API-Key": key,
        }

    def ready(self) -> bool:
        return self.diagnostics()["ready"] is True

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
            "errorType": None,
            "message": None,
        }
        try:
            request = urllib.request.Request(self._url(self.settings.ready_path), method="GET")
            with urllib.request.urlopen(request, timeout=5) as response:
                result["ready"] = 200 <= response.status < 300
                result["status"] = response.status
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
            )

        ready_result = self.diagnostics()
        if request.run_mode == "run-agent" and not self._resolved_llm_api_key(request.run_mode):
            return ProviderResult(
                provider=self.name,
                success=False,
                error_type=ProviderErrorType.EXTERNAL_REQUIRED,
                content=f"missing {self.settings.llm_api_key_env}; run-agent mode requires explicit LLM credentials",
                data={
                    "runMode": request.run_mode,
                    "ready": redact_secrets(ready_result),
                    "externalRequired": {
                        "type": "llm_credentials",
                        "env": self.settings.llm_api_key_env,
                        "message": "Set a local LLM credential before run-agent mode.",
                    },
                },
            )

        payload = self._conversation_payload(request)

        body = json.dumps(payload).encode("utf-8")
        http_request = urllib.request.Request(
            self._url(self.settings.conversation_path),
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(http_request, timeout=request.timeout_seconds) as response:
                response_body = response.read().decode("utf-8", errors="replace")
                data = _decode_json(response_body)
                ids = _extract_ids(data)
                run_endpoint_result = None
                success = 200 <= response.status < 300
                error_type = None
                if success and request.run_mode == "run-agent":
                    run_endpoint_result = self._run_conversation(ids.get("conversationId"), headers, request)
                    success = bool(run_endpoint_result.get("success"))
                    error_type = run_endpoint_result.get("errorType")

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
                        "executionStatus": ids.get("executionStatus"),
                        "tokenUsage": _token_usage(data, request.run_mode),
                        "runEndpointResult": redact_secrets(run_endpoint_result),
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
                data={"status": exc.code, "runMode": request.run_mode},
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
        except Exception as exc:
            return ProviderResult(
                provider=self.name,
                success=False,
                error_type=ProviderErrorType.UNKNOWN,
                content=str(exc),
                data={"runMode": request.run_mode},
            )

    def cancel(self, task_id: str) -> bool:
        try:
            headers = self._headers()
        except RuntimeError:
            return False

        safe_task_id = urllib.parse.quote(task_id, safe="")
        path = self.settings.cancel_path_template.format(task_id=safe_task_id)
        http_request = urllib.request.Request(
            self._url(path),
            data=b"{}",
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=10) as response:
                return 200 <= response.status < 300
        except Exception:
            return False

    def _conversation_payload(self, request: ProviderRequest) -> dict[str, Any]:
        llm_api_key = self._resolved_llm_api_key(request.run_mode)
        return {
            "workspace": {
                "kind": "LocalWorkspace",
                "working_dir": self._container_project_path(request.project_root),
            },
            "initial_message": {
                "role": "user",
                "run": False,
                "content": [
                    {
                        "type": "text",
                        "text": request.prompt,
                    }
                ],
            },
            "max_iterations": 1,
            "stuck_detection": True,
            "agent": {
                "kind": "Agent",
                "llm": {
                    "model": self.settings.llm_model,
                    "api_key": llm_api_key or self.settings.llm_api_key,
                    "usage_id": "ai-team-smoke",
                },
                "tools": [],
                "include_default_tools": [],
            },
            "tags": {
                "source": "ai-team",
                "workflow": _tag_value(request.workflow),
            },
        }

    def _run_conversation(
        self,
        conversation_id: str | None,
        headers: dict[str, str],
        request: ProviderRequest,
    ) -> dict[str, Any]:
        if not conversation_id:
            return {
                "success": False,
                "errorType": ProviderErrorType.INVALID_RESPONSE,
                "message": "OpenHands did not return a conversation id",
            }

        path = f"/api/conversations/{urllib.parse.quote(conversation_id, safe='')}/run"
        http_request = urllib.request.Request(
            self._url(path),
            data=b"{}",
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=request.timeout_seconds) as response:
                response_body = response.read().decode("utf-8", errors="replace")
                return {
                    "success": 200 <= response.status < 300,
                    "status": response.status,
                    "path": path,
                    "response": redact_secrets(_decode_json(response_body)),
                }
        except urllib.error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            return {
                "success": False,
                "status": exc.code,
                "path": path,
                "errorType": _http_error_type(exc.code),
                "response": redact_secrets(_decode_json(response_body)),
            }
        except (TimeoutError, socket.timeout) as exc:
            return {
                "success": False,
                "path": path,
                "errorType": ProviderErrorType.TIMEOUT,
                "message": str(exc),
            }
        except urllib.error.URLError as exc:
            return {
                "success": False,
                "path": path,
                "errorType": ProviderErrorType.NETWORK,
                "message": str(exc.reason),
            }

    def _resolved_llm_api_key(self, run_mode: str) -> str | None:
        env_value = os.environ.get(self.settings.llm_api_key_env)
        if env_value:
            return env_value
        if run_mode == "create-only":
            return self.settings.llm_api_key
        return None

    def _container_project_path(self, project_root: Path) -> str:
        host_root = Path(self.settings.host_workspace_root).expanduser().resolve()
        try:
            relative = project_root.resolve().relative_to(host_root)
        except ValueError:
            return project_root.as_posix()
        container_root = self.settings.container_workspace_root.rstrip("/")
        return f"{container_root}/{relative.as_posix()}"


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
    if status >= 500:
        return ProviderErrorType.NETWORK
    return ProviderErrorType.INVALID_RESPONSE


def _extract_ids(data: Any) -> dict[str, str | None]:
    if not isinstance(data, dict):
        return {"conversationId": None, "taskId": None, "executionStatus": None}

    conversation_id = data.get("id") or data.get("conversation_id") or data.get("conversationId")
    task_id = data.get("task_id") or data.get("taskId")
    execution_status = data.get("execution_status") or data.get("executionStatus")
    return {
        "conversationId": str(conversation_id) if conversation_id else None,
        "taskId": str(task_id) if task_id else None,
        "executionStatus": str(execution_status) if execution_status else None,
    }


def _tag_value(value: str) -> str:
    normalized = "".join(char.lower() for char in value if char.isalnum())
    return normalized[:64] or "workflow"


def _token_usage(data: Any, run_mode: str) -> int:
    if run_mode == "create-only":
        return 0
    if not isinstance(data, dict):
        return 0

    for key in ("total_tokens", "totalTokens", "tokens"):
        value = data.get(key)
        if isinstance(value, int):
            return value

    usage = data.get("usage") or data.get("stats")
    if isinstance(usage, dict):
        for key in ("total_tokens", "totalTokens", "tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                return value

    return 0


def _read_secret_file(path_value: str) -> str | None:
    expanded = os.path.expandvars(os.path.expanduser(path_value))
    path = Path(expanded)
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None
