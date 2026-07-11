from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from .base import BaseProvider, ProviderErrorType, ProviderRequest, ProviderResult, redact_secrets


@dataclass(frozen=True)
class OpenHandsSettings:
    base_url: str = "http://127.0.0.1:31024"
    session_key_env: str = "SESSION_API_KEY"
    ready_path: str = "/ready"
    conversation_path: str = "/api/v1/app-conversations"
    cancel_path_template: str = "/api/v1/app-conversations/{task_id}/stop"
    timeout_seconds: float = 30


class OpenHandsProvider(BaseProvider):
    name = "openhands"

    def __init__(self, settings: OpenHandsSettings | None = None, session_key: str | None = None) -> None:
        self.settings = settings or OpenHandsSettings()
        self._session_key = session_key

    @property
    def session_key(self) -> str | None:
        return self._session_key or os.environ.get(self.settings.session_key_env)

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

        payload: dict[str, Any] = {
            "title": f"AI Team: {request.workflow}",
            "initial_message": request.prompt,
            "project_path": str(request.project_root),
            "metadata": request.metadata,
        }

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
                return ProviderResult(
                    provider=self.name,
                    success=200 <= response.status < 300,
                    content=response_body,
                    data={"status": response.status, "response": redact_secrets(data)},
                )
        except urllib.error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            return ProviderResult(
                provider=self.name,
                success=False,
                error_type=_http_error_type(exc.code),
                content=redact_secrets(response_body),
                data={"status": exc.code},
            )
        except (TimeoutError, socket.timeout) as exc:
            return ProviderResult(
                provider=self.name,
                success=False,
                error_type=ProviderErrorType.TIMEOUT,
                content=str(exc),
            )
        except urllib.error.URLError as exc:
            return ProviderResult(
                provider=self.name,
                success=False,
                error_type=ProviderErrorType.NETWORK,
                content=str(exc.reason),
            )
        except Exception as exc:
            return ProviderResult(
                provider=self.name,
                success=False,
                error_type=ProviderErrorType.UNKNOWN,
                content=str(exc),
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
