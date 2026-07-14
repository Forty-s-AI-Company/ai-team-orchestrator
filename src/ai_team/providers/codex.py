from __future__ import annotations

import json
import os
import re
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
    allowed_models: tuple[str, ...] = ()
    allowed_reasoning_efforts: tuple[str, ...] = ("low", "medium", "high", "xhigh")

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
        try:
            run_args = _apply_routing_options(
                cli_settings.run_args,
                request.metadata.get("requestedModel"),
                request.metadata.get("reasoningEffort"),
                self.settings,
            )
        except ValueError as exc:
            return ProviderResult(
                provider=self.name,
                success=False,
                error_type=ProviderErrorType.INVALID_RESPONSE,
                content=str(exc),
                data={
                    "providerNative": True,
                    "requestedModel": request.metadata.get("requestedModel"),
                    "reasoningEffort": request.metadata.get("reasoningEffort"),
                },
            )
        cli_settings = replace(cli_settings, run_args=run_args)
        cli_settings = replace(cli_settings, run_args=[*cli_settings.run_args, "-"])
        result = cli_run_result(
            self.name,
            cli_settings,
            request,
            prompt_arg_mode="stdin",
            stdout_only_content=True,
        )
        return _with_routing_metadata(result, request)

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
        try:
            routed_args = _apply_routing_options(
                cli_settings.run_args,
                request.metadata.get("requestedModel"),
                request.metadata.get("reasoningEffort"),
                self.settings,
            )
        except ValueError as exc:
            return ProviderResult(
                provider=self.name,
                success=False,
                error_type=ProviderErrorType.INVALID_RESPONSE,
                content=str(exc),
                data={"providerNative": True, "codexNativePass": False},
            )
        cli_settings = replace(cli_settings, run_args=[*routed_args, "-"])
        native_tmp = "/tmp" if os.name != "nt" else None
        with tempfile.TemporaryDirectory(prefix="ai-team-codex-smoke-", dir=native_tmp) as tmp:
            smoke_request = replace(request, prompt=prompt, project_root=Path(tmp))
            result = cli_run_result(
                self.name,
                cli_settings,
                smoke_request,
                prompt_arg_mode="stdin",
                stdout_only_content=True,
            )
        validated = _validate_smoke_response(result, challenge)
        return _with_routing_metadata(validated, request)


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

    try:
        payload = json.loads(result.content.strip())
    except json.JSONDecodeError:
        return ProviderResult(
            provider="codex",
            success=False,
            error_type=ProviderErrorType.INVALID_RESPONSE,
            content=result.content,
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
        content=json.dumps(payload, separators=(",", ":")) if isinstance(payload, dict) else result.content,
        attempts=result.attempts,
        data={
            **data,
            "responseValidated": valid,
            "codexNativePass": valid,
            "responseSchema": payload.get("schema") if isinstance(payload, dict) else None,
        },
    )


def _apply_routing_options(
    run_args: list[str],
    model: Any,
    reasoning_effort: Any,
    settings: CodexSettings,
) -> list[str]:
    """Add audited model controls without accepting arbitrary CLI arguments."""
    args = list(run_args)
    if model is not None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Codex routing model must be a non-empty string")
        if not settings.allowed_models or model not in settings.allowed_models:
            raise ValueError(f"Codex routing model is not allowlisted: {model}")
        args.extend(["--model", model])
    if reasoning_effort is not None:
        if not isinstance(reasoning_effort, str) or reasoning_effort not in settings.allowed_reasoning_efforts:
            raise ValueError(f"Codex reasoning effort is not allowlisted: {reasoning_effort}")
        args.extend(["--config", f'model_reasoning_effort="{reasoning_effort}"'])
    return args


def _with_routing_metadata(result: ProviderResult, request: ProviderRequest) -> ProviderResult:
    token_usage = _extract_token_usage(result)
    # Codex writes native progress and token diagnostics to stderr. Keep those
    # details in the bounded command evidence. cli_run_result already exposes
    # the complete generated stdout separately, so structured consumers are not
    # forced to parse a diagnostic string or a truncated evidence field.
    return replace(
        result,
        data={
            **result.data,
            "requestedModel": request.metadata.get("requestedModel"),
            "reasoningEffort": request.metadata.get("reasoningEffort"),
            "tokenUsage": token_usage if token_usage is not None else result.data.get("tokenUsage", 0),
            "tokenUsageReported": token_usage is not None,
        },
    )


def _extract_token_usage(result: ProviderResult) -> int | None:
    command = result.data.get("command") if isinstance(result.data, dict) else None
    stderr = command.get("stderr", "") if isinstance(command, dict) else ""
    match = re.search(r"(?im)^tokens used\s*\n\s*([0-9][0-9,]*)\s*$", str(stderr))
    return int(match.group(1).replace(",", "")) if match else None
