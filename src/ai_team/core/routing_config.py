from __future__ import annotations

from typing import Any

from ai_team.core.orchestrator import WorkflowError
from ai_team.providers import RoleRoutingProfile, RouteTarget


ROLE_CHOICES = [
    "architect",
    "engineer",
    "product-manager",
    "project-analyst",
    "qa-engineer",
    "reviewer",
]
SUPPORTED_ROLE_PROVIDERS = {"codex", "antigravity", "handsfreecode"}


def load_role_profile(settings: dict[str, Any], role: str) -> RoleRoutingProfile:
    routing = settings.get("routing", {}) if isinstance(settings.get("routing"), dict) else {}
    roles = routing.get("roles", {}) if isinstance(routing.get("roles"), dict) else {}
    value = roles.get(role)
    if not isinstance(value, dict):
        raise WorkflowError(f"unknown or unconfigured routing role: {role}")
    fallback_values = value.get("fallbacks", [])
    if not isinstance(fallback_values, list):
        raise WorkflowError(f"routing role {role} fallbacks must be a list")
    profile = RoleRoutingProfile(
        role=role,
        primary=_route_target(value.get("primary"), role=role, field="primary"),
        fallbacks=tuple(
            _route_target(item, role=role, field=f"fallbacks[{index}]")
            for index, item in enumerate(fallback_values)
        ),
        secondary=(
            _route_target(value.get("secondary"), role=role, field="secondary")
            if value.get("secondary") is not None
            else None
        ),
        allow_write=value.get("allow_write") is True,
        allow_read_only_agent=value.get("allow_read_only_agent") is True,
    )
    if profile.allow_read_only_agent and (
        profile.primary.provider != "handsfreecode"
        or profile.fallbacks
        or profile.secondary is not None
    ):
        raise WorkflowError(
            f"routing role {role} read-only-agent must use HandsFreeCode exclusively"
        )
    for target in (
        profile.primary,
        *profile.fallbacks,
        *((profile.secondary,) if profile.secondary is not None else ()),
    ):
        _validate_profile_target(settings, role, target)
    return profile


def _route_target(value: object, *, role: str, field: str) -> RouteTarget:
    if not isinstance(value, dict):
        raise WorkflowError(f"routing role {role} requires a mapping for {field}")
    provider = value.get("provider")
    model = value.get("model")
    reasoning = value.get("reasoning_effort")
    if not isinstance(provider, str) or provider not in SUPPORTED_ROLE_PROVIDERS:
        raise WorkflowError(f"routing role {role} has unsupported provider in {field}")
    if not all(isinstance(item, str) and item.strip() for item in (model, reasoning)):
        raise WorkflowError(f"routing role {role} requires model and reasoning_effort in {field}")
    return RouteTarget(provider=provider, model=model, reasoning_effort=reasoning)


def _validate_profile_target(
    settings: dict[str, Any],
    role: str,
    target: RouteTarget,
) -> None:
    provider_settings = (
        settings.get(target.provider, {})
        if isinstance(settings.get(target.provider), dict)
        else {}
    )
    models = _string_list(provider_settings.get("allowed_models"))
    if target.model not in models:
        raise WorkflowError(
            f"routing role {role} model is not allowlisted for {target.provider}: {target.model}"
        )
    default_efforts = ["default"] if target.provider == "handsfreecode" else []
    efforts = _string_list(provider_settings.get("allowed_reasoning_efforts"), default_efforts)
    if target.reasoning_effort not in efforts:
        raise WorkflowError(
            f"routing role {role} reasoning effort is not allowlisted for "
            f"{target.provider}: {target.reasoning_effort}"
        )


def _string_list(value: object, fallback: list[str] | None = None) -> list[str]:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return fallback or []
