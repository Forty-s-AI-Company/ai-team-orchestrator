from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import BaseProvider, ProviderRequest, ProviderResult
from .cli_common import CliProviderSettings, build_diagnostics, cli_run_result


@dataclass(frozen=True)
class AntigravitySettings:
    executable: str = "antigravity"
    status_args: list[str] = field(default_factory=lambda: ["auth", "status"])
    quota_args: list[str] = field(default_factory=lambda: ["quota"])
    run_args: list[str] = field(default_factory=list)
    timeout_seconds: float = 45
    run_timeout_seconds: float = 180
    execution_enabled: bool = False

    def to_cli_settings(self) -> CliProviderSettings:
        return CliProviderSettings(
            executable=self.executable,
            status_args=self.status_args,
            quota_args=self.quota_args,
            run_args=self.run_args,
            timeout_seconds=self.timeout_seconds,
            run_timeout_seconds=self.run_timeout_seconds,
            execution_enabled=self.execution_enabled,
        )


class AntigravityProvider(BaseProvider):
    name = "antigravity"

    def __init__(self, settings: AntigravitySettings | None = None) -> None:
        self.settings = settings or AntigravitySettings()

    def ready(self) -> bool:
        return self.diagnostics().get("ready") is True

    def diagnostics(self) -> dict[str, Any]:
        return build_diagnostics(self.name, self.settings.to_cli_settings())

    def run(self, request: ProviderRequest) -> ProviderResult:
        return cli_run_result(self.name, self.settings.to_cli_settings(), request, prompt_arg_mode="append")
