from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .base import BaseProvider, ProviderErrorType, ProviderRequest, ProviderResult, redact_secrets


MAX_SECONDARY_CONTENT_CHARS = 16_000


@dataclass(frozen=True)
class ProviderRouteAttempt:
    provider: str
    ready: bool
    model: str | None = None
    reasoning_effort: str | None = None
    success: bool | None = None
    error_type: ProviderErrorType | None = None
    details: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return redact_secrets(
            {
                "provider": self.provider,
                "ready": self.ready,
                "model": self.model,
                "reasoningEffort": self.reasoning_effort,
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
        last_result: ProviderResult | None = None
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
            last_result = result
            if (
                request.metadata.get("writeRequired") is True
                and result.error_type in {ProviderErrorType.NETWORK, ProviderErrorType.TIMEOUT}
            ):
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
            if result.success:
                return _with_route_data(result, attempts)
            if request.metadata.get("writeRequired") is True:
                # A write attempt is bound to the first approved provider and
                # disposable worktree. Never switch executors mid-change.
                return _with_route_data(result, attempts)
            if result.error_type in {
                ProviderErrorType.AUTH,
                ProviderErrorType.EXTERNAL_REQUIRED,
                ProviderErrorType.RATE_LIMIT,
                ProviderErrorType.TIMEOUT,
                ProviderErrorType.NETWORK,
            }:
                continue
            return _with_route_data(result, attempts)

        if last_result is not None:
            return _with_route_data(last_result, attempts)
        return ProviderResult(
            provider=self.name,
            success=False,
            error_type=ProviderErrorType.EXTERNAL_REQUIRED,
            content="no provider route is available",
            data={"routeAttempts": [attempt.as_dict() for attempt in attempts]},
        )

@dataclass(frozen=True)
class RouteTarget:
    provider: str
    model: str
    reasoning_effort: str

    def as_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "model": self.model,
            "reasoningEffort": self.reasoning_effort,
        }


@dataclass(frozen=True)
class RoleRoutingProfile:
    role: str
    primary: RouteTarget
    fallbacks: tuple[RouteTarget, ...] = ()
    secondary: RouteTarget | None = None
    allow_write: bool = False
    allow_read_only_agent: bool = False


class RoleRouterProvider(BaseProvider):
    """Route one declared team role through audited provider/model profiles."""

    name = "role-router"
    _FALLBACK_ERRORS = {
        ProviderErrorType.AUTH,
        ProviderErrorType.EXTERNAL_REQUIRED,
        ProviderErrorType.NETWORK,
        ProviderErrorType.RATE_LIMIT,
        ProviderErrorType.TIMEOUT,
    }

    def __init__(
        self,
        profile: RoleRoutingProfile,
        providers: dict[str, BaseProvider],
    ) -> None:
        self.profile = profile
        self.providers = providers
        # Orchestrator checks this capability instead of weakening the historical
        # HandsFreeCode-only read-only-agent contract.
        self.supports_read_only_agent = profile.allow_read_only_agent

    def ready(self) -> bool:
        return any(
            _provider_ready(self.providers[target.provider])
            for target in (self.profile.primary, *self.profile.fallbacks)
            if target.provider in self.providers
        )

    def ready_for_route(self, route: object) -> bool:
        """Probe only the exact supervisor-selected Engineer route."""
        target = _bounded_cloud_target(
            route,
            (self.profile.primary, *self.profile.fallbacks),
        )
        if target is None or self.profile.role != "engineer" or not self.profile.allow_write:
            return False
        provider = self.providers.get(target.provider)
        return provider is not None and _provider_ready(provider)

    def diagnostics(self) -> dict[str, Any]:
        targets = (self.profile.primary, *self.profile.fallbacks)
        return {
            "provider": self.name,
            "role": self.profile.role,
            "ready": self.ready(),
            "routes": [
                {
                    **target.as_dict(),
                    "diagnostics": _provider_diagnostics(self.providers[target.provider]),
                }
                for target in targets
                if target.provider in self.providers
            ],
        }

    def run(self, request: ProviderRequest) -> ProviderResult:
        write_required = request.metadata.get("writeRequired") is True
        required_provider = request.metadata.get("requiredProvider")
        if required_provider is not None and (
            not isinstance(required_provider, str)
            or required_provider != self.profile.primary.provider
        ):
            return self._policy_failure(
                "required provider must match the role profile primary route"
            )
        if request.run_mode == "read-only-agent" and (
            not self.profile.allow_read_only_agent
            or self.profile.primary.provider != "handsfreecode"
            or self.profile.fallbacks
            or self.profile.secondary is not None
        ):
            return self._policy_failure(
                "read-only-agent role routing requires an exclusive HandsFreeCode route"
            )
        if write_required and not self.profile.allow_write:
            return self._policy_failure("selected role is not permitted to execute write workflows")

        # Bounded delivery binds each audited role to one provider identity.
        # Preserve that provider's transient failure so the continuous
        # supervisor can classify and retry it instead of spending tokens on a
        # fallback that the stage provider gate must reject anyway.
        targets = (self.profile.primary,) if write_required or required_provider is not None else (
            self.profile.primary,
            *self.profile.fallbacks,
        )
        selected_cloud_route = request.metadata.get("boundedCloudRoute")
        if selected_cloud_route is not None:
            if not write_required or self.profile.role != "engineer":
                return self._policy_failure("bounded cloud route is allowed only for the Engineer write stage")
            target = _bounded_cloud_target(selected_cloud_route, (self.profile.primary, *self.profile.fallbacks))
            if target is None:
                return self._policy_failure("bounded cloud route is not allowlisted by the Engineer role profile")
            targets = (target,)
        attempts: list[ProviderRouteAttempt] = []
        selected_target: RouteTarget | None = None
        selected_result: ProviderResult | None = None
        last_target: RouteTarget | None = None
        last_result: ProviderResult | None = None
        for target in targets:
            provider = self.providers.get(target.provider)
            if provider is None:
                attempts.append(
                    ProviderRouteAttempt(
                        target.provider,
                        ready=False,
                        model=target.model,
                        reasoning_effort=target.reasoning_effort,
                        details={"blockedReason": "profile references an unavailable provider"},
                    )
                )
                continue
            ready = _provider_ready(provider)
            if not ready:
                attempts.append(
                    ProviderRouteAttempt(
                        target.provider,
                        ready=False,
                        model=target.model,
                        reasoning_effort=target.reasoning_effort,
                    )
                )
                if selected_cloud_route is not None:
                    diagnostics = _provider_diagnostics(provider)
                    unavailable = ProviderResult(
                        provider=target.provider,
                        success=False,
                        error_type=_diagnostic_error_type(diagnostics),
                        content="selected bounded provider route is not ready",
                        data={"providerDiagnostics": diagnostics},
                    )
                    return self._with_profile_data(unavailable, target, attempts)
                continue
            routed_request = _request_for_target(request, self.profile, target)
            result = provider.run(routed_request)
            last_target = target
            last_result = result
            attempts.append(
                ProviderRouteAttempt(
                    provider=result.provider,
                    ready=True,
                    model=target.model,
                    reasoning_effort=target.reasoning_effort,
                    success=result.success,
                    error_type=result.error_type,
                    details={
                        "conversationId": result.conversation_id,
                        "taskId": result.task_id,
                        "runtimeProvider": result.data.get("runtimeProvider"),
                    },
                )
            )
            if result.success:
                selected_target = target
                selected_result = result
                break
            if write_required:
                return self._with_profile_data(result, target, attempts)
            if result.error_type not in self._FALLBACK_ERRORS:
                return self._with_profile_data(result, target, attempts)

        if selected_target is None or selected_result is None:
            if last_target is not None and last_result is not None:
                return self._with_profile_data(last_result, last_target, attempts)
            return ProviderResult(
                provider=self.name,
                success=False,
                error_type=ProviderErrorType.EXTERNAL_REQUIRED,
                content="no role-aware provider route is available",
                data=self._routing_data(None, attempts),
            )

        secondary = self._run_secondary(request, selected_target)
        return self._with_profile_data(selected_result, selected_target, attempts, secondary)

    def _run_secondary(
        self,
        request: ProviderRequest,
        selected_target: RouteTarget,
    ) -> dict[str, Any] | None:
        target = self.profile.secondary
        if target is None or request.metadata.get("writeRequired") is True:
            return None
        if target == selected_target:
            return {**target.as_dict(), "success": True, "skipped": "same_as_selected_route"}
        provider = self.providers.get(target.provider)
        if provider is None or not _provider_ready(provider):
            return {**target.as_dict(), "success": False, "errorType": ProviderErrorType.EXTERNAL_REQUIRED}
        review_request = _request_for_target(
            replace(
                request,
                prompt=(
                    "Provide an independent, read-only second opinion for the following task. "
                    "Do not edit files or execute destructive actions.\n\n" + request.prompt
                ),
                metadata={**request.metadata, "writeRequired": False, "writeAccess": False},
            ),
            self.profile,
            target,
        )
        result = provider.run(review_request)
        content = str(redact_secrets(result.content))
        content_truncated = len(content) > MAX_SECONDARY_CONTENT_CHARS
        if content_truncated:
            content = content[:MAX_SECONDARY_CONTENT_CHARS]
        return redact_secrets(
            {
                **target.as_dict(),
                "success": result.success and not content_truncated,
                "errorType": (
                    ProviderErrorType.INVALID_RESPONSE
                    if content_truncated
                    else result.error_type
                ),
                "runtimeProvider": result.data.get("runtimeProvider"),
                "tokenUsage": result.data.get("tokenUsage", 0),
                "tokenUsageReported": result.data.get(
                    "tokenUsageReported",
                    "tokenUsage" in result.data,
                ),
                "content": content,
                "contentTruncated": content_truncated,
            }
        )

    def _with_profile_data(
        self,
        result: ProviderResult,
        target: RouteTarget,
        attempts: list[ProviderRouteAttempt],
        secondary: dict[str, Any] | None = None,
    ) -> ProviderResult:
        return replace(
            result,
            data={
                **result.data,
                **self._routing_data(target, attempts),
                "secondaryReview": secondary,
            },
        )

    def _routing_data(
        self,
        target: RouteTarget | None,
        attempts: list[ProviderRouteAttempt],
    ) -> dict[str, Any]:
        return {
            "role": self.profile.role,
            "routingProfile": self.profile.role,
            "selectedProvider": target.provider if target else None,
            "selectedModel": target.model if target else None,
            "reasoningEffort": target.reasoning_effort if target else None,
            "primaryRoute": self.profile.primary.as_dict(),
            "fallbackUsed": bool(target and target != self.profile.primary),
            "fallbackChain": [attempt.as_dict() for attempt in attempts[:-1]],
            "routeAttempts": [attempt.as_dict() for attempt in attempts],
        }

    def _policy_failure(self, message: str) -> ProviderResult:
        return ProviderResult(
            provider=self.name,
            success=False,
            error_type=ProviderErrorType.INVALID_RESPONSE,
            content=message,
            data={"role": self.profile.role, "routingProfile": self.profile.role},
        )


def _request_for_target(
    request: ProviderRequest,
    profile: RoleRoutingProfile,
    target: RouteTarget,
) -> ProviderRequest:
    return replace(
        request,
        metadata={
            **request.metadata,
            "role": profile.role,
            "routingProfile": profile.role,
            "requestedModel": target.model,
            "reasoningEffort": target.reasoning_effort,
        },
    )


def _bounded_cloud_target(value: object, targets: tuple[RouteTarget, ...]) -> RouteTarget | None:
    """Resolve a supervisor route only when it exactly matches profile policy."""

    if not isinstance(value, dict):
        return None
    provider = value.get("provider")
    model = value.get("model")
    reasoning = value.get("reasoningEffort")
    if not all(isinstance(item, str) for item in (provider, model, reasoning)):
        return None
    return next(
        (
            target
            for target in targets
            if target.provider == provider and target.model == model and target.reasoning_effort == reasoning
        ),
        None,
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


def _diagnostic_error_type(diagnostics: dict[str, Any]) -> ProviderErrorType:
    value = diagnostics.get("errorType")
    try:
        if value is not None:
            return ProviderErrorType(str(value))
    except ValueError:
        pass
    if diagnostics.get("failClosed") is True:
        return ProviderErrorType.AUTH
    return ProviderErrorType.EXTERNAL_REQUIRED


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
