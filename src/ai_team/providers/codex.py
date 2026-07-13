from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from .base import BaseProvider, ProviderErrorType, ProviderRequest, ProviderResult
from .cli_common import CliProviderSettings, build_diagnostics, cli_run_result


@dataclass(frozen=True)
class CodexSettings:
    executable: str = "codex"
    status_args: list[str] = field(default_factory=lambda: ["--version"])
    quota_args: list[str] = field(default_factory=lambda: ["doctor", "--json"])
    run_args: list[str] = field(
        default_factory=lambda: ["exec", "--sandbox", "read-only", "--skip-git-repo-check"]
    )
    write_run_args: list[str] = field(
        default_factory=lambda: ["exec", "--sandbox", "danger-full-access", "--skip-git-repo-check"]
    )
    timeout_seconds: float = 45
    run_timeout_seconds: float = 180
    execution_enabled: bool = True

    def to_cli_settings(self, *, write_enabled: bool = False) -> CliProviderSettings:
        return CliProviderSettings(
            executable=self.executable,
            status_args=self.status_args,
            quota_args=self.quota_args,
            run_args=self.write_run_args if write_enabled else self.run_args,
            timeout_seconds=self.timeout_seconds,
            run_timeout_seconds=self.run_timeout_seconds,
            execution_enabled=self.execution_enabled,
        )


class CodexProvider(BaseProvider):
    name = "codex"

    def __init__(self, settings: CodexSettings | None = None) -> None:
        self.settings = settings or CodexSettings()

    def ready(self) -> bool:
        return self.diagnostics().get("ready") is True

    def diagnostics(self) -> dict[str, Any]:
        return build_diagnostics(self.name, self.settings.to_cli_settings())

    def run(self, request: ProviderRequest) -> ProviderResult:
        if request.workflow == "provider-smoke":
            return self._run_provider_smoke(request)

        write_enabled = request.metadata.get("writeRequired") is True
        if write_enabled:
            # Codex danger-full-access is only allowed inside disposable linked
            # worktrees. Primary repositories have a .git directory; linked
            # worktrees have a .git file pointing back to the source repo.
            git_marker = request.project_root / ".git"
            if not git_marker.is_file():
                return ProviderResult(
                    provider=self.name,
                    success=False,
                    content="trusted write requires a disposable linked worktree",
                )
        cli_settings = self.settings.to_cli_settings(write_enabled=write_enabled)
        cli_settings = replace(cli_settings, run_args=[*cli_settings.run_args, "-"])
        return cli_run_result(
            self.name,
            cli_settings,
            request,
            prompt_arg_mode="stdin",
        )

    def _run_provider_smoke(self, request: ProviderRequest) -> ProviderResult:
        """Verify native Codex execution without exposing the project workspace."""
        challenge = uuid4().hex
        prompt = (
            "Do not use tools or inspect any files. "
            "Return only strict JSON without Markdown fences using "
            "schema='ai-team-codex-smoke/v1', "
            f"challenge='{challenge}', provider='codex', status='ok'."
        )
        cli_settings = self.settings.to_cli_settings(write_enabled=False)
        cli_settings = replace(cli_settings, run_args=[*cli_settings.run_args, "-"])
        native_tmp = "/tmp" if os.name != "nt" else None
        with tempfile.TemporaryDirectory(prefix="ai-team-codex-smoke-", dir=native_tmp) as tmp:
            smoke_request = replace(request, prompt=prompt, project_root=Path(tmp))
            result = cli_run_result(
                self.name,
                cli_settings,
                smoke_request,
                prompt_arg_mode="stdin",
            )
        return _validate_smoke_response(result, challenge)


def _validate_smoke_response(result: ProviderResult, challenge: str) -> ProviderResult:
    data = {
        **result.data,
        "commandSucceeded": result.success,
        "responseValidated": False,
        "providerNative": True,
        "codexNativePass": False,
        "masqueradeAsProvider": False,
    }
    if not result.success:
        return replace(result, data=data)

    command = result.data.get("command") if isinstance(result.data, dict) else None
    stdout = command.get("stdout", "") if isinstance(command, dict) else ""
    try:
        payload = json.loads(str(stdout).strip())
    except json.JSONDecodeError:
        return ProviderResult(
            provider="codex",
            success=False,
            error_type=ProviderErrorType.INVALID_RESPONSE,
            content=str(stdout),
            attempts=result.attempts,
            data=data,
        )

    valid = bool(
        isinstance(payload, dict)
        and payload.get("schema") == "ai-team-codex-smoke/v1"
        and payload.get("challenge") == challenge
        and payload.get("provider") == "codex"
        and payload.get("status") == "ok"
    )
    return ProviderResult(
        provider="codex",
        success=valid,
        error_type=None if valid else ProviderErrorType.INVALID_RESPONSE,
        content=json.dumps(payload, separators=(",", ":")) if isinstance(payload, dict) else str(stdout),
        attempts=result.attempts,
        data={
            **data,
            "responseValidated": valid,
            "codexNativePass": valid,
            "responseSchema": payload.get("schema") if isinstance(payload, dict) else None,
        },
    )
