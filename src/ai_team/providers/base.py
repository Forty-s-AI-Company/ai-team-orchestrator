from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class ProviderErrorType(StrEnum):
    AUTH = "auth"
    TIMEOUT = "timeout"
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    INVALID_RESPONSE = "invalid_response"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProviderRequest:
    workflow: str
    prompt: str
    project_root: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 30
    dry_run: bool = False


@dataclass(frozen=True)
class ProviderResult:
    provider: str
    success: bool
    content: str = ""
    error_type: ProviderErrorType | None = None
    attempts: int = 1
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def conversation_id(self) -> str | None:
        value = self.data.get("conversationId") or self.data.get("conversation_id")
        return str(value) if value else None

    @property
    def task_id(self) -> str | None:
        value = self.data.get("taskId") or self.data.get("task_id")
        return str(value) if value else None


class ProviderError(RuntimeError):
    def __init__(self, message: str, error_type: ProviderErrorType = ProviderErrorType.UNKNOWN) -> None:
        super().__init__(message)
        self.error_type = error_type


SECRET_PATTERNS = [
    re.compile(r"(?i)(session[_-]?api[_-]?key|api[_-]?key|token|secret|password)\s*[:=]\s*([^\s,;]+)"),
    re.compile(r"sk-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"(?i)Bearer\s+[A-Za-z0-9_\-.]+"),
]


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: redact_secrets(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if not isinstance(value, str):
        return value

    redacted = value
    for pattern in SECRET_PATTERNS:
        if pattern.pattern.startswith("(?i)(session"):
            redacted = pattern.sub(lambda match: f"{match.group(1)}=<redacted>", redacted)
        else:
            redacted = pattern.sub("<redacted>", redacted)
    return redacted


class BaseProvider(ABC):
    name = "base"

    @abstractmethod
    def ready(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def run(self, request: ProviderRequest) -> ProviderResult:
        raise NotImplementedError

    def cancel(self, task_id: str) -> bool:
        return False


class RetryingProvider(BaseProvider):
    def __init__(self, provider: BaseProvider, max_retries: int = 2, backoff_seconds: float = 0.05) -> None:
        self.provider = provider
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.name = provider.name

    def ready(self) -> bool:
        return self.provider.ready()

    def run(self, request: ProviderRequest) -> ProviderResult:
        attempts = 0
        last_result: ProviderResult | None = None
        for attempts in range(1, self.max_retries + 2):
            last_result = self.provider.run(request)
            if last_result.success:
                return ProviderResult(
                    provider=last_result.provider,
                    success=True,
                    content=last_result.content,
                    attempts=attempts,
                    data=last_result.data,
                )
            if last_result.error_type in {ProviderErrorType.AUTH, ProviderErrorType.INVALID_RESPONSE}:
                break
            time.sleep(self.backoff_seconds * attempts)

        assert last_result is not None
        return ProviderResult(
            provider=last_result.provider,
            success=False,
            content=last_result.content,
            error_type=last_result.error_type,
            attempts=attempts,
            data=last_result.data,
        )


class MockProvider(BaseProvider):
    name = "mock"

    def __init__(self, fail_times: int = 0, error_type: ProviderErrorType = ProviderErrorType.NETWORK) -> None:
        self.fail_times = fail_times
        self.error_type = error_type
        self.calls = 0

    def ready(self) -> bool:
        return True

    def run(self, request: ProviderRequest) -> ProviderResult:
        self.calls += 1
        if self.calls <= self.fail_times:
            return ProviderResult(
                provider=self.name,
                success=False,
                error_type=self.error_type,
                content="mock transient failure",
            )

        return ProviderResult(
            provider=self.name,
            success=True,
            content=f"mock completed {request.workflow}",
            data={"workflow": request.workflow, "dryRun": request.dry_run},
        )
