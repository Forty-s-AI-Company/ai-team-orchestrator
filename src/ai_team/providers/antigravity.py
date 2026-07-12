from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    prompt_max_chars: int = 1200

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
        compact_request = replace(request, prompt=_compact_prompt(request.prompt, self.settings.prompt_max_chars))
        cli_settings = self.settings.to_cli_settings()
        cli_settings = replace(
            cli_settings,
            run_args=[*cli_settings.run_args, "--add-dir", str(request.project_root)],
        )
        return cli_run_result(self.name, cli_settings, compact_request, prompt_arg_mode="append")


def _compact_prompt(prompt: str, max_chars: int) -> str:
    values: dict[str, str] = {}
    for line in prompt.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip().lower()] = value.strip()
    normalized = (
        "Read-only AI Team task. "
        f"Project={values.get('project', 'unknown')}; "
        f"Root={values.get('root', 'unknown')}; "
        f"Workflow={values.get('workflow', 'unknown')}; "
        f"Stages={values.get('stages', 'inspect, review, report')}. "
        "Do not edit files, deploy, process payments, or run migrations. "
        "Return one compact JSON object with status, findings, tests, and blockers."
    )
    limit = max(200, max_chars)
    if len(normalized) <= limit:
        return normalized
    suffix = "\n[Prompt truncated by Antigravity compact mode]"
    return f"{normalized[: limit - len(suffix)].rstrip()}{suffix}"
