from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .base import BaseProvider, ProviderErrorType, ProviderRequest, ProviderResult
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
    diagnostics_cache_ttl_seconds: float = 30

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

    def __init__(
        self,
        settings: AntigravitySettings | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings or AntigravitySettings()
        self._monotonic = monotonic
        self._diagnostics_cache: dict[str, Any] | None = None
        self._diagnostics_cached_at: float | None = None
        self._diagnostics_started_at: float | None = None

    def ready(self) -> bool:
        return self.diagnostics().get("ready") is True

    def diagnostics(self) -> dict[str, Any]:
        now = self._monotonic()
        if self._cache_is_valid(now):
            return copy.deepcopy(self._diagnostics_cache)
        self._diagnostics_started_at = now
        diagnostics = build_diagnostics(self.name, self.settings.to_cli_settings())
        if diagnostics.get("ready") is True:
            self._diagnostics_cache = copy.deepcopy(diagnostics)
            self._diagnostics_cached_at = self._monotonic()
        return copy.deepcopy(diagnostics)

    def run(self, request: ProviderRequest) -> ProviderResult:
        budget = request.timeout_seconds if request.timeout_seconds is not None else self.settings.run_timeout_seconds
        started_at = self._monotonic()
        diagnostics = self.diagnostics()
        budget_started_at = self._diagnostics_started_at if self._cache_is_valid(self._monotonic()) else started_at
        remaining = budget - (self._monotonic() - (budget_started_at or started_at))
        if remaining <= 0:
            return _timeout_result(request, diagnostics, "deadline exhausted during diagnostics")

        challenge = uuid4().hex
        probe = _select_repository_probe(request.project_root) if request.workflow == "provider-smoke" else None
        prompt = _compact_prompt(
            request.prompt,
            self.settings.prompt_max_chars,
            challenge=challenge,
            probe_path=probe[0] if probe else None,
        )
        compact_request = replace(request, prompt=prompt, timeout_seconds=remaining)
        cli_settings = self.settings.to_cli_settings()
        cli_settings = replace(
            cli_settings,
            run_args=_bounded_run_args(cli_settings.run_args, request.project_root, remaining),
            run_timeout_seconds=min(cli_settings.run_timeout_seconds, remaining),
        )
        command_result = cli_run_result(
            self.name,
            cli_settings,
            compact_request,
            prompt_arg_mode="append",
            diagnostics_override=diagnostics,
        )
        return _validate_response(command_result, request, challenge, probe)

    def _cache_is_valid(self, now: float) -> bool:
        return bool(
            self._diagnostics_cache
            and self._diagnostics_cached_at is not None
            and now - self._diagnostics_cached_at <= self.settings.diagnostics_cache_ttl_seconds
        )


def _compact_prompt(
    prompt: str,
    max_chars: int,
    challenge: str,
    probe_path: str | None = None,
) -> str:
    values: dict[str, str] = {}
    for line in prompt.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip().lower()] = value.strip()
    if probe_path:
        normalized = (
            "Repository visibility smoke. Read the exact tracked file "
            f"'{probe_path}' and calculate its SHA-256 locally. Challenge={challenge}. "
            "Return only strict JSON with schema='ai-team-repository-smoke/v1', challenge, "
            "probe={path,sha256}, summary, findings=[], tests=[], blockers=[]. "
            "Do not use Markdown fences and do not edit files."
        )
    else:
        normalized = (
            "Read-only AI Team task. "
            f"Project={values.get('project', 'unknown')}; "
            f"Workflow={values.get('workflow', 'unknown')}; "
            f"Stages={values.get('stages', 'inspect, review, report')}; "
            f"Challenge={challenge}. Do not edit, deploy, process payments, or run migrations. "
            "Return only strict JSON with schema='ai-team-antigravity/v1', challenge, status, "
            "findings=[], tests=[], blockers=[]. Do not use Markdown fences."
        )
    limit = max(240, max_chars)
    if len(normalized) <= limit:
        return normalized
    suffix = " [truncated]"
    return f"{normalized[: limit - len(suffix)].rstrip()}{suffix}"


def _select_repository_probe(project_root: Path) -> tuple[str, str] | None:
    candidates = ["package.json", "pyproject.toml", "README.md"]
    result = subprocess.run(
        ["git", "ls-files", "--", *candidates],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    )
    tracked = {line.strip() for line in result.stdout.splitlines() if line.strip()} if result.returncode == 0 else set()
    for candidate in candidates:
        path = project_root / candidate
        if candidate in tracked and path.is_file():
            return candidate, hashlib.sha256(path.read_bytes()).hexdigest()
    return None


def _bounded_run_args(run_args: list[str], project_root: Path, remaining: float) -> list[str]:
    args = list(run_args)
    if "--print-timeout" in args:
        index = args.index("--print-timeout")
        if index + 1 < len(args):
            args[index + 1] = f"{max(1, int(remaining - 2))}s"
    insert_at = args.index("--print") if "--print" in args else len(args)
    args[insert_at:insert_at] = ["--add-dir", str(project_root)]
    return args


def _validate_response(
    result: ProviderResult,
    request: ProviderRequest,
    challenge: str,
    probe: tuple[str, str] | None,
) -> ProviderResult:
    base_data = {
        **result.data,
        "commandSucceeded": result.success,
        "responseValidated": False,
        "repositorySmokePassed": False,
        "providerNative": True,
        "antigravityNativePass": False,
        "masqueradeAsProvider": False,
    }
    if not result.success:
        return replace(result, data=base_data)
    try:
        payload = json.loads(result.content)
    except json.JSONDecodeError:
        return ProviderResult(
            provider=result.provider,
            success=False,
            error_type=ProviderErrorType.INVALID_RESPONSE,
            content=result.content,
            attempts=result.attempts,
            data=base_data,
        )
    valid = isinstance(payload, dict) and payload.get("challenge") == challenge
    expected_schema = "ai-team-repository-smoke/v1" if request.workflow == "provider-smoke" else "ai-team-antigravity/v1"
    valid = valid and payload.get("schema") == expected_schema
    valid = valid and all(isinstance(payload.get(key), list) for key in ("findings", "tests", "blockers"))
    repository_smoke_passed = False
    if valid and request.workflow == "provider-smoke":
        probe_payload = payload.get("probe")
        repository_smoke_passed = bool(
            probe
            and isinstance(probe_payload, dict)
            and probe_payload.get("path") == probe[0]
            and probe_payload.get("sha256") == probe[1]
        )
        valid = repository_smoke_passed
    elif valid:
        valid = isinstance(payload.get("status"), str)
    data = {
        **base_data,
        "responseValidated": valid,
        "repositorySmokePassed": repository_smoke_passed,
        "antigravityNativePass": valid,
        "responseSchema": payload.get("schema") if isinstance(payload, dict) else None,
    }
    return ProviderResult(
        provider=result.provider,
        success=valid,
        error_type=None if valid else ProviderErrorType.INVALID_RESPONSE,
        content=result.content,
        attempts=result.attempts,
        data=data,
    )


def _timeout_result(request: ProviderRequest, diagnostics: dict[str, Any], message: str) -> ProviderResult:
    return ProviderResult(
        provider="antigravity",
        success=False,
        error_type=ProviderErrorType.TIMEOUT,
        content=message,
        data={
            "runMode": request.run_mode,
            "diagnostics": diagnostics,
            "commandSucceeded": False,
            "responseValidated": False,
            "repositorySmokePassed": False,
            "providerNative": True,
            "antigravityNativePass": False,
            "masqueradeAsProvider": False,
        },
    )
