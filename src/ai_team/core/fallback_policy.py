from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ai_team.providers.base import ProviderErrorType, ProviderResult, redact_secrets


QUOTA_PATTERNS = [
    re.compile(r"(?i)usage limit"),
    re.compile(r"(?i)quota exceeded"),
    re.compile(r"(?i)resource_exhausted"),
    re.compile(r"(?i)too many requests"),
    re.compile(r"(?i)individual quota reached"),
    re.compile(r"(?i)try again at\s+(.+?)(?:\.|$)"),
    re.compile(r"(?i)reset time:\s*([0-9:\- ]+)(?:\s*\(local time\))?"),
    re.compile(r"(?i)resets in\s+([0-9hms ]+)"),
]

RESET_PATTERNS = [
    re.compile(r"(?i)try again at\s+(.+?)(?:\.|$)"),
    re.compile(r"(?i)reset time:\s*([0-9:\- ]+)(?:\s*\(local time\))?"),
    re.compile(r"(?i)resets in\s+([0-9hms ]+)"),
]

OLLAMA_ALLOWED_WORKFLOW_TERMS = {
    "project-analysis",
    "docs",
    "document",
    "documentation",
    "triage",
    "review",
    "report",
}

OLLAMA_BLOCKED_WORKFLOW_TERMS = {
    "bug-fix-loop",
    "implement",
    "write",
    "payment",
    "deploy",
    "migration",
    "payout",
    "settlement",
}


@dataclass(frozen=True)
class FallbackDecision:
    quota_exhausted: bool
    fallback_allowed: bool
    fallback_provider: str | None
    reset_time: str | None
    reason: str
    masquerade_as_provider: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "quotaExhausted": self.quota_exhausted,
            "fallbackAllowed": self.fallback_allowed,
            "fallbackProvider": self.fallback_provider,
            "resetTime": self.reset_time,
            "reason": self.reason,
            "masqueradeAsProvider": self.masquerade_as_provider,
        }


def decide_fallback(provider_result: ProviderResult | None, workflow_name: str) -> FallbackDecision:
    if provider_result is None:
        return FallbackDecision(False, False, None, None, "provider result unavailable")

    provider = provider_result.provider.lower()
    text = _decision_text(provider_result)
    quota_exhausted = provider_result.error_type == ProviderErrorType.RATE_LIMIT or _looks_like_quota(text)
    if provider not in {"codex", "antigravity"}:
        return FallbackDecision(
            quota_exhausted=False,
            fallback_allowed=False,
            fallback_provider=None,
            reset_time=None,
            reason=f"fallback is only evaluated for Codex or Antigravity, got {provider}",
        )

    if not quota_exhausted:
        return FallbackDecision(False, False, None, None, "quota exhaustion not detected")

    allowed, reason = _ollama_workflow_allowed(workflow_name)
    return FallbackDecision(
        quota_exhausted=True,
        fallback_allowed=allowed,
        fallback_provider="ollama" if allowed else None,
        reset_time=_extract_reset_time(text),
        reason=reason,
        masquerade_as_provider=False,
    )


def _decision_text(provider_result: ProviderResult) -> str:
    data = redact_secrets(provider_result.data)
    return f"{provider_result.content}\n{data}"


def _looks_like_quota(text: str) -> bool:
    return any(pattern.search(text) for pattern in QUOTA_PATTERNS)


def _extract_reset_time(text: str) -> str | None:
    for pattern in RESET_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None


def _ollama_workflow_allowed(workflow_name: str) -> tuple[bool, str]:
    lowered = workflow_name.lower()
    if any(term in lowered for term in OLLAMA_BLOCKED_WORKFLOW_TERMS):
        return False, "Ollama fallback is blocked for write, payment, deploy, migration, settlement, or payout work."
    if any(term in lowered for term in OLLAMA_ALLOWED_WORKFLOW_TERMS):
        return True, "Ollama fallback is limited to documentation, triage, review, report, and project analysis work."
    return False, "Workflow is not explicitly allowlisted for Ollama fallback."


def state_timestamp() -> str:
    return datetime.now(UTC).isoformat()
