from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import BaseProvider, ProviderErrorType, ProviderRequest, ProviderResult, redact_secrets


@dataclass(frozen=True)
class ProviderRouteAttempt:
    provider: str
    ready: bool
    success: bool | None = None
    error_type: ProviderErrorType | None = None
    details: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return redact_secrets(
            {
                "provider": self.provider,
                "ready": self.ready,
                "success": self.success,
                "errorType": self.error_type,
                "details": self.details or {},
            }
        )


class RouterProvider(BaseProvider):
    name = "auto"

    def __init__(self, providers: list[BaseProvider]) -> None:
        self.providers = providers

    def ready(self) -> bool:
        return any(_provider_ready(provider) for provider in self.providers)

    def diagnostics(self) -> dict[str, Any]:
        attempts = []
        for provider in self.providers:
            attempts.append(_provider_diagnostics(provider))
        return {
            "provider": self.name,
            "ready": any(item.get("ready") for item in attempts),
            "routes": attempts,
        }

    def run(self, request: ProviderRequest) -> ProviderResult:
        attempts: list[ProviderRouteAttempt] = []
        for provider in self.providers:
            if request.metadata.get("writeRequired") is True and provider.name not in {"codex", "antigravity"}:
                attempts.append(
                    ProviderRouteAttempt(
                        provider=provider.name,
                        ready=False,
                        details={"blockedReason": "write workflow requires an approved provider-native CLI"},
                    )
                )
                continue
            ready = _provider_ready(provider)
            if not ready:
                attempts.append(ProviderRouteAttempt(provider=provider.name, ready=False))
                continue
            result = provider.run(request)
            attempts.append(
                ProviderRouteAttempt(
                    provider=result.provider,
                    ready=True,
                    success=result.success,
                    error_type=result.error_type,
                    details={
                        "conversationId": result.conversation_id,
                        "taskId": result.task_id,
                        "runtimeProvider": result.data.get("runtimeProvider"),
                    },
                )
            )
            if result.success or result.error_type == ProviderErrorType.EXTERNAL_REQUIRED:
                return _with_route_data(result, attempts)
            if result.error_type in {ProviderErrorType.RATE_LIMIT, ProviderErrorType.TIMEOUT, ProviderErrorType.NETWORK}:
                continue
            return _with_route_data(result, attempts)

        return ProviderResult(
            provider=self.name,
            success=False,
            error_type=ProviderErrorType.EXTERNAL_REQUIRED,
            content="no provider route is available",
            data={"routeAttempts": [attempt.as_dict() for attempt in attempts]},
        )


def _provider_ready(provider: BaseProvider) -> bool:
    try:
        return provider.ready()
    except Exception:
        return False


def _provider_diagnostics(provider: BaseProvider) -> dict[str, Any]:
    diagnostics = getattr(provider, "diagnostics", None)
    if callable(diagnostics):
        try:
            value = diagnostics()
            if isinstance(value, dict):
                return redact_secrets(value)
        except Exception as exc:
            return {"provider": provider.name, "ready": False, "error": str(exc)}
    return {"provider": provider.name, "ready": _provider_ready(provider)}


def _with_route_data(result: ProviderResult, attempts: list[ProviderRouteAttempt]) -> ProviderResult:
    data = {
        **result.data,
        "selectedProvider": result.provider,
        "routeAttempts": [attempt.as_dict() for attempt in attempts],
    }
    return ProviderResult(
        provider=result.provider,
        success=result.success,
        content=result.content,
        error_type=result.error_type,
        attempts=result.attempts,
        data=data,
    )
