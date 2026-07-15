from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_team.core.fallback_policy import _extract_reset_time, _looks_like_quota

from .base import ProviderErrorType, ProviderRequest, ProviderResult, redact_secrets


SAFE_ENV_KEYS = {
    "APPDATA",
    "COMSPEC",
    "HOME",
    "LOCALAPPDATA",
    "PATHEXT",
    "PATH",
    "PROGRAMDATA",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "WINDIR",
}


@dataclass(frozen=True)
class CliProviderSettings:
    executable: str
    status_args: list[str] = field(default_factory=list)
    quota_args: list[str] = field(default_factory=list)
    run_args: list[str] = field(default_factory=list)
    timeout_seconds: float = 45
    run_timeout_seconds: float = 180
    execution_enabled: bool = True


@dataclass(frozen=True)
class CliCommandResult:
    available: bool
    return_code: int | None
    stdout: str
    stderr: str
    error: str | None = None

    @property
    def combined(self) -> str:
        return f"{self.stdout}\n{self.stderr}\n{self.error or ''}"


def executable_available(executable: str) -> bool:
    return _resolve_executable(executable) is not None


def run_cli_command(
    settings: CliProviderSettings,
    args: list[str],
    cwd: Path | None = None,
    input_text: str | None = None,
    timeout_seconds: float | None = None,
) -> CliCommandResult:
    resolved_executable = _resolve_executable(settings.executable)
    if not resolved_executable:
        return CliCommandResult(False, None, "", "", f"executable not found: {settings.executable}")

    try:
        completed = subprocess.run(
            [resolved_executable, *args],
            cwd=cwd,
            input=input_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds or settings.timeout_seconds,
            check=False,
            shell=False,
            env=_safe_env(),
        )
        return CliCommandResult(
            True,
            completed.returncode,
            str(redact_secrets(completed.stdout)),
            str(redact_secrets(completed.stderr)),
        )
    except subprocess.TimeoutExpired as exc:
        return CliCommandResult(
            True,
            None,
            str(redact_secrets(exc.stdout or "")),
            str(redact_secrets(exc.stderr or "")),
            f"timeout after {timeout_seconds or settings.timeout_seconds}s",
        )
    except OSError as exc:
        return CliCommandResult(True, None, "", "", str(exc))


def build_diagnostics(provider_name: str, settings: CliProviderSettings) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "provider": provider_name,
        "executable": settings.executable,
        "resolvedExecutable": _resolve_executable(settings.executable),
        "available": executable_available(settings.executable),
        "ready": False,
        "quotaExhausted": False,
        "resetTime": None,
        "errorType": None,
        "message": None,
    }
    if not diagnostics["available"]:
        diagnostics["errorType"] = ProviderErrorType.EXTERNAL_REQUIRED
        diagnostics["message"] = f"{settings.executable} CLI is not installed or not on PATH"
        return diagnostics

    commands = [settings.status_args]
    if settings.quota_args:
        commands.append(settings.quota_args)

    results: list[dict[str, Any]] = []
    quota_text_parts: list[str] = []
    for args in commands:
        if not args:
            continue
        result = run_cli_command(settings, args)
        results.append(_command_result_dict(args, result))
        quota_text_parts.append(result.combined)

    quota_text = "\n".join(quota_text_parts)
    status_ok = True
    if settings.status_args and results:
        status_ok = results[0].get("returnCode") == 0
    diagnostics["commands"] = results
    diagnostics["quotaExhausted"] = _looks_like_quota(quota_text)
    diagnostics["resetTime"] = _extract_reset_time(quota_text)
    diagnostics["ready"] = diagnostics["available"] and status_ok and not diagnostics["quotaExhausted"]
    if diagnostics["quotaExhausted"]:
        diagnostics["errorType"] = ProviderErrorType.RATE_LIMIT
        diagnostics["message"] = "quota exhausted"
    elif not status_ok:
        diagnostics["errorType"] = ProviderErrorType.EXTERNAL_REQUIRED
        diagnostics["message"] = "provider CLI status command failed"
    return redact_secrets(diagnostics)


def result_from_diagnostics(provider_name: str, diagnostics: dict[str, Any], request: ProviderRequest) -> ProviderResult | None:
    if not diagnostics.get("available"):
        return ProviderResult(
            provider=provider_name,
            success=False,
            error_type=ProviderErrorType.EXTERNAL_REQUIRED,
            content=str(diagnostics.get("message") or "CLI unavailable"),
            data={"runMode": request.run_mode, "externalRequired": diagnostics},
        )
    if diagnostics.get("quotaExhausted"):
        return ProviderResult(
            provider=provider_name,
            success=False,
            error_type=ProviderErrorType.RATE_LIMIT,
            content=f"quota exhausted; reset time: {diagnostics.get('resetTime')}",
            data={
                "runMode": request.run_mode,
                "quotaExhausted": True,
                "resetTime": diagnostics.get("resetTime"),
                "externalRequired": diagnostics,
            },
        )
    return None


def cli_run_result(
    provider_name: str,
    settings: CliProviderSettings,
    request: ProviderRequest,
    prompt_arg_mode: str = "append",
    diagnostics_override: dict[str, Any] | None = None,
    stdout_only_content: bool = False,
) -> ProviderResult:
    diagnostics = diagnostics_override or build_diagnostics(provider_name, settings)
    blocked = result_from_diagnostics(provider_name, diagnostics, request)
    if blocked is not None:
        return blocked
    if not settings.execution_enabled:
        return ProviderResult(
            provider=provider_name,
            success=False,
            error_type=ProviderErrorType.EXTERNAL_REQUIRED,
            content=f"{provider_name} execution is disabled by settings",
            data={"runMode": request.run_mode, "externalRequired": diagnostics},
        )
    if not settings.run_args:
        return ProviderResult(
            provider=provider_name,
            success=False,
            error_type=ProviderErrorType.EXTERNAL_REQUIRED,
            content=f"{provider_name} run command is not configured",
            data={"runMode": request.run_mode, "externalRequired": diagnostics},
        )

    args = [*settings.run_args]
    input_text = None
    if prompt_arg_mode == "stdin":
        input_text = request.prompt
    else:
        args.append(request.prompt)

    result = run_cli_command(
        settings,
        args,
        cwd=request.project_root,
        input_text=input_text,
        timeout_seconds=request.timeout_seconds or settings.run_timeout_seconds,
    )
    error_type = _error_type_from_command(result)
    success = result.return_code == 0 and error_type is None
    content = result.stdout.strip() if stdout_only_content and success else result.combined.strip()
    return ProviderResult(
        provider=provider_name,
        success=success,
        error_type=error_type,
        content=content,
        data={
            "runMode": request.run_mode,
            "diagnostics": diagnostics,
            "command": _command_result_dict(args, result),
            "quotaExhausted": error_type == ProviderErrorType.RATE_LIMIT,
            "resetTime": _extract_reset_time(result.combined),
            "masqueradeAsProvider": False,
        },
    )


def _safe_env() -> dict[str, str]:
    safe = {key: value for key, value in os.environ.items() if key.upper() in SAFE_ENV_KEYS}
    safe["PYTHONIOENCODING"] = "utf-8"
    safe["PYTHONUTF8"] = "1"
    return safe


def _resolve_executable(executable: str) -> str | None:
    explicit = Path(executable)
    if explicit.is_absolute() and explicit.exists():
        return str(explicit)
    if os.name == "nt" and not Path(executable).suffix:
        for suffix in (".cmd", ".exe", ".bat", ".ps1"):
            candidate = shutil.which(f"{executable}{suffix}")
            if candidate:
                return candidate
    return shutil.which(executable)


def _command_result_dict(args: list[str], result: CliCommandResult) -> dict[str, Any]:
    return redact_secrets(
        {
            "args": args,
            "available": result.available,
            "returnCode": result.return_code,
            "stdout": result.stdout[:4000],
            "stderr": result.stderr[:4000],
            "error": result.error,
        }
    )


def _error_type_from_command(result: CliCommandResult) -> ProviderErrorType | None:
    lowered = result.combined.lower()
    if "orchestrator_helper_" in lowered or "windows sandbox failed" in lowered:
        return ProviderErrorType.NETWORK
    if result.return_code == 0:
        return None
    if _looks_like_quota(result.combined):
        return ProviderErrorType.RATE_LIMIT
    if result.error and "timeout" in result.error.lower():
        return ProviderErrorType.TIMEOUT
    if "timeout" in result.combined.lower():
        return ProviderErrorType.TIMEOUT
    return ProviderErrorType.UNKNOWN
