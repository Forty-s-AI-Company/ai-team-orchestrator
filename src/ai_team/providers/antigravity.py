from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import tempfile
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
    allowed_models: tuple[str, ...] = ()
    allowed_reasoning_efforts: tuple[str, ...] = ("low", "medium", "high", "thinking")

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
        if request.workflow == "provider-smoke" and probe is None:
            return ProviderResult(
                provider=self.name,
                success=False,
                error_type=ProviderErrorType.INVALID_RESPONSE,
                content="provider-smoke requires a tracked, non-symlink manifest probe",
                data={
                    "runMode": request.run_mode,
                    "providerNative": True,
                    "antigravityNativePass": False,
                    "repositorySmokePassed": False,
                    "masqueradeAsProvider": False,
                },
            )
        if probe:
            native_tmp = "/tmp" if os.name != "nt" else None
            with tempfile.TemporaryDirectory(prefix="ai-team-antigravity-smoke-", dir=native_tmp) as tmp:
                smoke_root = Path(tmp)
                destination = smoke_root / probe[0]
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((request.project_root / probe[0]).read_bytes())
                return self._execute(request, diagnostics, remaining, challenge, probe, smoke_root)
        return self._execute(request, diagnostics, remaining, challenge, probe, request.project_root)

    def _execute(
        self,
        request: ProviderRequest,
        diagnostics: dict[str, Any],
        remaining: float,
        challenge: str,
        probe: tuple[str, str] | None,
        workspace_root: Path,
    ) -> ProviderResult:
        prompt = _compact_prompt(
            request.prompt,
            self.settings.prompt_max_chars,
            challenge=challenge,
            probe_path=probe[0] if probe else None,
            bounded_stage=request.metadata.get("boundedStage"),
        )
        compact_request = replace(
            request,
            prompt=prompt,
            project_root=workspace_root,
            timeout_seconds=remaining,
        )
        cli_settings = self.settings.to_cli_settings()
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
                data={
                    "providerNative": True,
                    "requestedModel": request.metadata.get("requestedModel"),
                    "reasoningEffort": request.metadata.get("reasoningEffort"),
                },
            )
        cli_settings = replace(
            cli_settings,
            run_args=_bounded_run_args(routed_args, workspace_root, remaining),
            run_timeout_seconds=min(cli_settings.run_timeout_seconds, remaining),
        )
        command_result = cli_run_result(
            self.name,
            cli_settings,
            compact_request,
            prompt_arg_mode="append",
            diagnostics_override=diagnostics,
        )
        result = _validate_response(command_result, request, challenge, probe)
        return replace(
            result,
            data={
                **result.data,
                "requestedModel": request.metadata.get("requestedModel"),
                "reasoningEffort": request.metadata.get("reasoningEffort"),
                "tokenUsage": result.data.get("tokenUsage", 0),
                "tokenUsageReported": "tokenUsage" in result.data,
            },
        )

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
    bounded_stage: Any = None,
) -> str:
    values: dict[str, str] = {}
    for line in prompt.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip().lower()] = value.strip()
    if isinstance(bounded_stage, str) and bounded_stage in {"pm", "architect", "qa"}:
        task = values.get("task", "unknown")[:120]
        instruction = values.get("instruction", "unknown")[:280]
        allowed_paths = _compact_json_array(values.get("allowed write paths", "[]"), max_chars=220)
        validation_commands = _compact_json_array(values.get("validation commands", "[]"), max_chars=280)
        implementation_evidence = _compact_implementation_evidence(
            values.get("implementation evidence", "{}")
        )
        stage_requirements = {
            "pm": "Include a non-empty acceptanceCriteria string array.",
            "architect": (
                "Include non-empty plan, allowedWritePaths, validationCommands arrays and "
                "schemaOrApiChange=false. Do not expand paths or commands beyond the task."
            ),
            "qa": "Report only evidence-backed diff findings. Use findings=[] when acceptance evidence passes.",
        }[bounded_stage]
        normalized = (
            f"Bounded read-only stage={bounded_stage}; Challenge={challenge}. "
            "Return JSON only: schema='ai-team-bounded-delivery/v1', challenge, stage, "
            "status='passed', findings=[], tests=[], blockers=[]. No Markdown. "
            "Forbidden: edit, shell, migrate, seed, deploy, payment, secrets, data deletion, schema/API changes. "
            f"{stage_requirements} Task={task}; Instruction={instruction}; "
            f"AllowedWritePaths={allowed_paths}; ValidationCommands={validation_commands}; "
            f"ImplementationEvidence={implementation_evidence}."
        )
    elif probe_path:
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
    limit = min(1600, max(240, max_chars))
    if len(normalized) <= limit:
        return normalized
    suffix = " [truncated]"
    return f"{normalized[: limit - len(suffix)].rstrip()}{suffix}"


def _compact_json_array(raw: str, *, max_chars: int) -> str:
    """Return a bounded JSON array without leaving a truncated JSON fragment."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return "[]"
    if not isinstance(value, list):
        return "[]"
    items = [item[:160] for item in value if isinstance(item, str)]
    while items:
        encoded = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) <= max_chars:
            return encoded
        items.pop()
    return "[]"


def _compact_implementation_evidence(raw: str) -> str:
    """Expose only the bounded facts a read-only QA stage needs to verify."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = {}
    if not isinstance(value, dict):
        value = {}
    changed_files = value.get("changedFiles")
    repairs = value.get("repairs")
    validation = value.get("validation")
    commit_sha = value.get("commitSha")
    summary = {
        "changedFileCount": len(changed_files) if isinstance(changed_files, list) else 0,
        "commitSha": commit_sha[:12] if isinstance(commit_sha, str) else None,
        "validationSuccess": validation.get("success") is True if isinstance(validation, dict) else False,
        "repairCount": len(repairs) if isinstance(repairs, list) else 0,
    }
    return json.dumps(summary, ensure_ascii=False, separators=(",", ":"))


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
        if candidate not in tracked or path.is_symlink() or not path.is_file():
            continue
        try:
            path.resolve().relative_to(project_root.resolve())
        except ValueError:
            continue
        if path.is_file():
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


def _apply_routing_options(
    run_args: list[str],
    model: Any,
    reasoning_effort: Any,
    settings: AntigravitySettings,
) -> list[str]:
    """Replace only the model value; never pass profile text as arbitrary flags."""
    args = list(run_args)
    if model is not None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Antigravity routing model must be a non-empty string")
        if not settings.allowed_models or model not in settings.allowed_models:
            raise ValueError(f"Antigravity routing model is not allowlisted: {model}")
        if "--model" in args:
            index = args.index("--model")
            if index + 1 >= len(args):
                raise ValueError("Antigravity base arguments contain an incomplete --model option")
            args[index + 1] = model
        else:
            args[0:0] = ["--model", model]
    if reasoning_effort is not None and (
        not isinstance(reasoning_effort, str)
        or reasoning_effort not in settings.allowed_reasoning_efforts
    ):
        raise ValueError(f"Antigravity reasoning effort is not allowlisted: {reasoning_effort}")
    if isinstance(model, str) and isinstance(reasoning_effort, str):
        expected = _reasoning_from_model_name(model)
        if expected is not None and expected != reasoning_effort:
            raise ValueError(
                "Antigravity reasoning effort does not match the allowlisted model name: "
                f"{model}"
            )
    return args


def _reasoning_from_model_name(model: str) -> str | None:
    suffixes = {
        "(Low)": "low",
        "(Medium)": "medium",
        "(High)": "high",
        "(Thinking)": "thinking",
    }
    return next((effort for suffix, effort in suffixes.items() if model.endswith(suffix)), None)


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
    bounded_stage = request.metadata.get("boundedStage")
    if isinstance(bounded_stage, str) and bounded_stage in {"pm", "architect", "qa"}:
        expected_schema = "ai-team-bounded-delivery/v1"
    else:
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
        if expected_schema == "ai-team-bounded-delivery/v1":
            valid = valid and payload.get("stage") == bounded_stage
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
