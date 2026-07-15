"""Persistent, bounded cloud-provider recovery for delivery supervisors.

This module intentionally does *not* execute delivery work.  It only owns the
small, serialisable state machine which decides whether a cloud model may be
called again.  That keeps retries recoverable across supervisor restarts and
prevents a local continuity helper from becoming an unbounded fallback
Engineer.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from ai_team.providers.base import BaseProvider, ProviderRequest, ProviderResult, redact_secrets


SCHEMA_VERSION = 1
MAX_RETRY_HISTORY = 64
MAX_PACKET_HISTORY = 8


@dataclass(frozen=True)
class CloudModelRoute:
    provider: str
    model: str
    reasoning_effort: str
    priority: int

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}:{self.reasoning_effort}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "reasoningEffort": self.reasoning_effort,
            "priority": self.priority,
        }


@dataclass(frozen=True)
class RetrySettings:
    max_attempts_per_model: int = 2
    initial_delay_seconds: int = 60
    multiplier: int = 2
    max_delay_seconds: int = 1800
    jitter_ratio: float = 0.15
    max_task_provider_attempts: int = 8
    max_provider_probes_per_hour: int = 4
    circuit_failure_threshold: int = 2
    circuit_cooldown_seconds: int = 1800
    circuit_max_cooldown_seconds: int = 14_400
    probe_interval_seconds: int = 900


@dataclass(frozen=True)
class LocalContinuitySettings:
    enabled: bool = True
    provider: str | None = None
    model: str | None = None
    command: tuple[str, ...] = ()
    timeout_seconds: int = 180
    max_output_tokens: int = 2000
    allow_repository_writes: bool = False
    allow_git_writes: bool = False
    allow_test_execution: bool = False
    allow_code_generation: bool = False
    deterministic_fallback: bool = True


TRANSIENT_STOP_REASONS = {
    "provider-quota-exhausted",
    "provider-rate-limit",
    "provider-timeout",
    "provider-network-error",
    "provider-capacity-unavailable",
    "provider-service-unavailable",
    "provider-temporary-upstream-failure",
}


def classify_failure(reason: str, *, error_summary: str = "") -> str:
    """Classify a redacted failure without collapsing all errors into quota."""

    value = f"{reason} {error_summary}".lower()
    if any(token in value for token in ("401", "403", "auth", "unauthoriz", "forbidden", "billing", "credit exhausted", "subscription")):
        return "account_or_hard_quota_error"
    if any(token in value for token in ("lint", "test", "build", "compile", "validation", "git-commit", "qa-review")):
        return "task_or_code_failure"
    if any(token in value for token in ("worktree", "disk", "repository lock", "permission", "state", "executable", "subprocess")):
        return "infrastructure_failure"
    if reason in TRANSIENT_STOP_REASONS or any(
        token in value
        for token in ("429", "rate limit", "quota", "capacity", "overloaded", "timeout", "connection reset", "temporary", "502", "503", "504")
    ):
        return "transient_provider_error"
    return "task_or_code_failure"


def default_engineer_routes() -> tuple[CloudModelRoute, ...]:
    return (
        CloudModelRoute("codex", "gpt-5.6-terra", "medium", 100),
        CloudModelRoute("codex", "gpt-5.6-sol", "medium", 80),
        CloudModelRoute("codex", "gpt-5.6-luna", "medium", 60),
    )


def load_resilience_settings(settings: dict[str, Any]) -> tuple[tuple[CloudModelRoute, ...], RetrySettings, LocalContinuitySettings]:
    providers = settings.get("providers", {}) if isinstance(settings.get("providers"), dict) else {}
    configured = providers.get("codex_engineer", {}) if isinstance(providers.get("codex_engineer"), dict) else {}
    raw_models = configured.get("models") if isinstance(configured.get("models"), list) else []
    routes: list[CloudModelRoute] = []
    for index, item in enumerate(raw_models):
        if not isinstance(item, dict):
            continue
        model, reasoning = item.get("name"), item.get("reasoning")
        priority = item.get("priority")
        if isinstance(model, str) and isinstance(reasoning, str) and isinstance(priority, int):
            routes.append(CloudModelRoute("codex", model, reasoning, priority))
    if not routes:
        routes = list(default_engineer_routes())
    routes.sort(key=lambda route: route.priority, reverse=True)

    retry_raw = configured.get("retry") if isinstance(configured.get("retry"), dict) else {}
    circuit_raw = configured.get("circuit_breaker") if isinstance(configured.get("circuit_breaker"), dict) else {}
    budget_raw = configured.get("budgets") if isinstance(configured.get("budgets"), dict) else {}
    retry = RetrySettings(
        max_attempts_per_model=_positive_int(retry_raw.get("max_attempts_per_model"), 2),
        initial_delay_seconds=_positive_int(retry_raw.get("initial_delay_seconds"), 60),
        multiplier=_positive_int(retry_raw.get("multiplier"), 2),
        max_delay_seconds=_positive_int(retry_raw.get("max_delay_seconds"), 1800),
        jitter_ratio=_ratio(retry_raw.get("jitter_ratio"), 0.15),
        max_task_provider_attempts=_positive_int(budget_raw.get("max_task_provider_attempts"), 8),
        max_provider_probes_per_hour=_positive_int(budget_raw.get("max_provider_probes_per_hour"), 4),
        circuit_failure_threshold=_positive_int(circuit_raw.get("failure_threshold"), 2),
        circuit_cooldown_seconds=_positive_int(circuit_raw.get("cooldown_seconds"), 1800),
        circuit_max_cooldown_seconds=_positive_int(circuit_raw.get("max_cooldown_seconds"), 14_400),
        probe_interval_seconds=_positive_int(circuit_raw.get("probe_interval_seconds"), 900),
    )
    local_raw = settings.get("local_continuity", {}) if isinstance(settings.get("local_continuity"), dict) else {}
    command = local_raw.get("command")
    local = LocalContinuitySettings(
        enabled=local_raw.get("enabled", True) is True,
        provider=_optional_string(local_raw.get("provider")),
        model=_optional_string(local_raw.get("model")),
        command=tuple(command) if isinstance(command, list) and all(isinstance(value, str) and value for value in command) else (),
        timeout_seconds=_positive_int(local_raw.get("timeout_seconds"), 180),
        max_output_tokens=_positive_int(local_raw.get("max_output_tokens"), 2000),
        allow_repository_writes=local_raw.get("allow_repository_writes", False) is True,
        allow_git_writes=local_raw.get("allow_git_writes", False) is True,
        allow_test_execution=local_raw.get("allow_test_execution", False) is True,
        allow_code_generation=local_raw.get("allow_code_generation", False) is True,
        deterministic_fallback=local_raw.get("deterministic_fallback", True) is True,
    )
    if any((local.allow_repository_writes, local.allow_git_writes, local.allow_test_execution, local.allow_code_generation)):
        raise ValueError("local_continuity is recorder-only and cannot enable repository writes, git writes, tests, or code generation")
    return tuple(routes), retry, local


class CloudRecoveryState:
    """A serialisable per-task, per-model circuit breaker state machine."""

    def __init__(
        self,
        *,
        task_sha: str,
        stage: str,
        routes: tuple[CloudModelRoute, ...],
        settings: RetrySettings,
        payload: dict[str, Any] | None = None,
        random_uniform: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self.task_sha = task_sha
        self.stage = stage
        self.routes = routes
        self.settings = settings
        self._random_uniform = random_uniform
        self.payload = payload if isinstance(payload, dict) else {}
        if self.payload.get("taskSha") != task_sha or self.payload.get("stage") != stage:
            self.payload = {}
        current = self.payload.get("currentRoute")
        self.current_key = current if isinstance(current, str) else routes[0].key
        self.circuits = self.payload.get("circuits") if isinstance(self.payload.get("circuits"), dict) else {}
        self.history = self.payload.get("retryHistory") if isinstance(self.payload.get("retryHistory"), list) else []
        self.probes = self.payload.get("probes") if isinstance(self.payload.get("probes"), list) else []

    def current_route(self) -> CloudModelRoute:
        return next((route for route in self.routes if route.key == self.current_key), self.routes[0])

    def next_action(self, now: datetime) -> tuple[str, CloudModelRoute | None, datetime | None]:
        """Return delivery, probe, or cloud_waiting without invoking a provider."""

        now = _utc(now)
        preferred = self.routes[0]
        preferred_circuit = self._circuit(preferred)
        if preferred_circuit["circuitState"] == "open" and _time(preferred_circuit.get("nextProbeAt")) <= now:
            self.current_key = preferred.key
            return "probe", preferred, now
        current = self.current_route()
        current_circuit = self._circuit(current)
        if current_circuit["circuitState"] == "open":
            if _time(current_circuit.get("nextProbeAt")) <= now:
                return "probe", current, now
            for route in self.routes:
                circuit = self._circuit(route)
                if circuit["circuitState"] != "open":
                    self.current_key = route.key
                    return "delivery", route, now
            next_probe = min(_time(self._circuit(route).get("nextProbeAt")) for route in self.routes)
            return "cloud_waiting", None, next_probe
        return "delivery", current, now

    def record_probe(self, route: CloudModelRoute, *, success: bool, now: datetime, summary: str = "") -> None:
        now = _utc(now)
        self.probes.append({"route": route.as_dict(), "at": now.isoformat(), "success": success})
        self.probes = self.probes[-MAX_RETRY_HISTORY:]
        circuit = self._circuit(route)
        if success:
            circuit.update({"circuitState": "closed", "cooldownUntil": None, "nextProbeAt": None})
            self._event("circuit_closed", route, now, "provider_probe_succeeded")
        else:
            self._open(route, now, "transient_provider_error", summary or "provider-probe-failed")
            self._event("provider_probe_failed", route, now, summary or "provider-probe-failed")

    def probe_allowed(self, now: datetime) -> bool:
        cutoff = _utc(now) - timedelta(hours=1)
        recent = [item for item in self.probes if _time(item.get("at")) >= cutoff]
        return len(recent) < self.settings.max_provider_probes_per_hour

    def next_probe_budget_at(self, now: datetime) -> datetime:
        cutoff = _utc(now) - timedelta(hours=1)
        recent = sorted(_time(item.get("at")) for item in self.probes if _time(item.get("at")) >= cutoff)
        if len(recent) < self.settings.max_provider_probes_per_hour:
            return _utc(now)
        return recent[0] + timedelta(hours=1)

    def record_failure(self, route: CloudModelRoute, *, reason: str, now: datetime, summary: str = "") -> dict[str, Any]:
        now = _utc(now)
        classification = classify_failure(reason, error_summary=summary)
        circuit = self._circuit(route)
        event = {
            "route": route.as_dict(), "at": now.isoformat(), "reason": reason,
            "classification": classification, "summary": redact_secrets(summary),
        }
        self.history.append(event)
        self.history = self.history[-MAX_RETRY_HISTORY:]
        if classification != "transient_provider_error":
            return {"status": "hard_blocked" if classification != "task_or_code_failure" else "failed", "classification": classification}

        attempts = int(circuit.get("attempts", 0)) + 1
        circuit.update({
            "attempts": attempts,
            "failureCount": int(circuit.get("failureCount", 0)) + 1,
            "lastFailureAt": now.isoformat(),
            "lastErrorCategory": classification,
        })
        total = sum(int(self._circuit(candidate).get("attempts", 0)) for candidate in self.routes)
        if total >= self.settings.max_task_provider_attempts:
            self._open(route, now, classification, reason)
            return {"status": "cloud_waiting", "classification": classification, "nextRetryAt": self.next_action(now)[2].isoformat()}
        if attempts < self.settings.max_attempts_per_model:
            delay = self._delay(attempts)
            retry_at = now + timedelta(seconds=delay)
            circuit["nextRetryAt"] = retry_at.isoformat()
            self._event("retry_backoff", route, now, reason)
            return {"status": "retry_backoff", "classification": classification, "nextRetryAt": retry_at.isoformat(), "delaySeconds": delay}

        self._open(route, now, classification, reason)
        for candidate in self.routes:
            if self._circuit(candidate)["circuitState"] != "open":
                self.current_key = candidate.key
                delay = self._delay(1)
                retry_at = now + timedelta(seconds=delay)
                self._circuit(candidate)["nextRetryAt"] = retry_at.isoformat()
                self._event("model_fallback_started", candidate, now, f"fallback-after:{route.model}")
                return {"status": "provider_fallback", "classification": classification, "nextRetryAt": retry_at.isoformat(), "delaySeconds": delay, "route": candidate.as_dict()}
        next_probe = min(_time(self._circuit(candidate).get("nextProbeAt")) for candidate in self.routes)
        self._event("cloud_waiting_started", route, now, reason)
        return {"status": "cloud_waiting", "classification": classification, "nextRetryAt": next_probe.isoformat()}

    def as_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": SCHEMA_VERSION,
            "taskSha": self.task_sha,
            "stage": self.stage,
            "currentRoute": self.current_key,
            "preferredRoute": self.routes[0].as_dict(),
            "routes": [route.as_dict() for route in self.routes],
            "circuits": self.circuits,
            "retryHistory": self.history,
            "probes": self.probes,
            "automaticRecovery": True,
        }

    def _circuit(self, route: CloudModelRoute) -> dict[str, Any]:
        current = self.circuits.get(route.key)
        if not isinstance(current, dict):
            current = {
                "failureCount": 0, "attempts": 0, "lastFailureAt": None,
                "lastErrorCategory": None, "circuitState": "closed",
                "cooldownUntil": None, "nextProbeAt": None, "nextRetryAt": None,
            }
            self.circuits[route.key] = current
        return current

    def _open(self, route: CloudModelRoute, now: datetime, category: str, reason: str) -> None:
        circuit = self._circuit(route)
        failures = max(1, int(circuit.get("failureCount", 1)))
        exponent = max(0, failures - self.settings.circuit_failure_threshold)
        cooldown = min(
            self.settings.circuit_cooldown_seconds * (2**exponent),
            self.settings.circuit_max_cooldown_seconds,
        )
        until = now + timedelta(seconds=cooldown)
        circuit.update({
            "circuitState": "open", "cooldownUntil": until.isoformat(), "nextProbeAt": until.isoformat(),
            "lastErrorCategory": category,
        })
        self._event("circuit_opened", route, now, reason)

    def _delay(self, attempt: int) -> int:
        base = min(self.settings.initial_delay_seconds * (self.settings.multiplier ** max(0, attempt - 1)), self.settings.max_delay_seconds)
        if self.settings.jitter_ratio <= 0:
            return base
        lower = 1 - self.settings.jitter_ratio
        upper = 1 + self.settings.jitter_ratio
        return max(1, round(base * self._random_uniform(lower, upper)))

    def _event(self, kind: str, route: CloudModelRoute, now: datetime, reason: str) -> None:
        self.history.append({"event": kind, "route": route.as_dict(), "at": now.isoformat(), "reason": reason})
        self.history = self.history[-MAX_RETRY_HISTORY:]


class SelectedCloudRouteProvider(BaseProvider):
    """Inject one supervisor-selected, allowlisted route into a role provider.

    The wrapped provider remains responsible for provider-native execution and
    receipts.  This wrapper carries no fallback behaviour itself, so a single
    write worktree can never silently switch providers mid-stage.
    """

    def __init__(self, provider: BaseProvider, route: CloudModelRoute) -> None:
        self.provider = provider
        self.route = route
        self.name = provider.name
        self.supports_read_only_agent = getattr(provider, "supports_read_only_agent", False)

    def ready(self) -> bool:
        return self.provider.ready()

    def run(self, request: ProviderRequest) -> ProviderResult:
        return self.provider.run(
            replace(
                request,
                metadata={
                    **request.metadata,
                    "boundedCloudRoute": self.route.as_dict(),
                },
            )
        )


def create_resume_packet(
    *,
    state_root: Path,
    project_path: Path,
    project_id: str | None = None,
    task_id: str,
    task_sha: str,
    task_title: str,
    task_state: dict[str, Any],
    supervisor_state: CloudRecoveryState,
    receipt_paths: list[str],
    continuity: LocalContinuitySettings,
    now: datetime,
) -> dict[str, str]:
    """Atomically persist deterministic recovery evidence outside the repository."""

    root = state_root / "continuity" / _safe_slug(task_id)
    root.mkdir(parents=True, exist_ok=True)
    repository = _git_summary(project_path)
    packet = redact_secrets({
        "schemaVersion": 1,
        "projectId": project_id or project_path.name,
        "runId": hashlib.sha256(f"{task_sha}:{now.isoformat()}".encode()).hexdigest()[:16],
        "taskId": task_id,
        "taskTitle": task_title,
        "role": "engineer",
        "dag": "bounded-autonomous-delivery",
        "currentStage": task_state.get("stage") or "engineer",
        "lastCompletedStage": _last_completed_stage(receipt_paths),
        "nextPendingStage": task_state.get("stage") or "engineer",
        "taskStatus": "cloud_waiting",
        "providerAttempts": supervisor_state.as_dict(),
        "errorClassifications": [item for item in supervisor_state.history if item.get("classification")],
        "retryHistory": supervisor_state.history,
        "cooldownUntil": _cooldown_summary(supervisor_state),
        "nextProbeAt": _next_probe(supervisor_state),
        "repository": repository,
        "worktreePath": task_state.get("worktreePath"),
        "branch": repository.get("branch"),
        "headCommit": repository.get("head"),
        "baseBranch": repository.get("branch"),
        "gitStatusSummary": repository.get("status"),
        "changedFiles": repository.get("changedFiles"),
        "stagedFiles": repository.get("stagedFiles"),
        "untrackedFiles": repository.get("untrackedFiles"),
        "diffStat": repository.get("diffStat"),
        "receiptPaths": receipt_paths,
        "artifacts": [],
        "testsAlreadyExecuted": (task_state.get("validation") or {}).get("commands", []),
        "testResults": task_state.get("validation"),
        "unresolvedIssues": ["all configured cloud Engineer models are temporarily unavailable"],
        "acceptanceCriteria": task_state.get("acceptanceCriteria", []),
        "pendingQa": True,
        "pendingReview": True,
        "pendingPr": True,
        "pendingCi": True,
        "pendingMerge": True,
        "importantDecisions": ["local continuity is recorder-only; it did not modify the repository"],
        "filesForNextEngineer": repository.get("changedFiles"),
        "warnings": ["verify worktree and HEAD against this packet before resuming"],
        "recommendedResumeAction": "probe preferred cloud model, reconcile Git fingerprint, then resume the pending engineer stage",
        "localContinuity": {
            "enabled": continuity.enabled,
            "mode": "recorder_only",
            "provider": continuity.provider,
            "model": continuity.model,
            "repositoryModifications": "none",
            "fallback": "deterministic",
        },
        "generatedAt": _utc(now).isoformat(),
    })
    local_result = _run_local_recorder(continuity, root, packet)
    packet["localContinuity"].update(local_result)
    stamp = _utc(now).strftime("%Y%m%dT%H%M%SZ")
    json_path = root / f"{stamp}-resume.json"
    markdown_path = root / f"{stamp}-resume.md"
    _atomic_json(json_path, packet)
    _atomic_text(markdown_path, _packet_markdown(packet))
    _trim_history(root)
    return {"json": str(json_path), "markdown": str(markdown_path), "mode": "deterministic"}


def _git_summary(project: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            result = subprocess.run(["git", *args], cwd=project, text=True, capture_output=True, timeout=15, check=False)
        except (OSError, subprocess.SubprocessError):
            return ""
        # Porcelain status uses the first two columns for index/worktree state.
        # Removing leading whitespace corrupts paths whose first status column
        # is blank (for example, `` M tests/example.ts``).
        return redact_secrets(result.stdout.rstrip("\r\n")) if result.returncode == 0 else ""

    porcelain = run("status", "--porcelain", "--untracked-files=all")
    lines = porcelain.splitlines()
    return {
        "path": str(project.resolve()), "branch": run("branch", "--show-current"), "head": run("rev-parse", "HEAD"),
        "status": "clean" if not lines else f"{len(lines)} changed entries",
        "changedFiles": [line[3:] for line in lines if len(line) > 3],
        "stagedFiles": run("diff", "--cached", "--name-only").splitlines(),
        "untrackedFiles": [line[3:] for line in lines if line.startswith("?? ")],
        "diffStat": run("diff", "--stat"),
    }


def _last_completed_stage(receipts: list[str]) -> str | None:
    stages = ("review", "qa", "engineer", "architect", "pm")
    completed: set[str] = set()
    for receipt in receipts:
        path = Path(receipt)
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue

        stage = payload.get("stage")
        validation = payload.get("validationResult")
        if (
            stage in stages
            and payload.get("providerSuccess") is True
            and isinstance(validation, dict)
            and validation.get("success") is True
            and payload.get("stopReason") in (None, "")
            and (stage != "engineer" or bool(payload.get("commitSha")))
        ):
            completed.add(stage)

    return next((stage for stage in stages if stage in completed), None)


def _cooldown_summary(state: CloudRecoveryState) -> dict[str, Any]:
    return {key: value.get("cooldownUntil") for key, value in state.circuits.items() if isinstance(value, dict) and value.get("cooldownUntil")}


def _next_probe(state: CloudRecoveryState) -> str | None:
    values = [_time(value.get("nextProbeAt")) for value in state.circuits.values() if isinstance(value, dict) and value.get("nextProbeAt")]
    return min(values).isoformat() if values else None


def _packet_markdown(packet: dict[str, Any]) -> str:
    return "\n".join((
        "# AI Team resume packet", "",
        f"- Task: `{packet['taskId']}`", f"- Current stage: `{packet['currentStage']}`",
        f"- Next provider probe: `{packet.get('nextProbeAt')}`", "- Local continuity: recorder-only / no repository modifications.",
        "", "## Recommended resume action", "", str(packet["recommendedResumeAction"]), "",
    ))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str))


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _trim_history(root: Path) -> None:
    for suffix in ("-resume.json", "-resume.md"):
        entries = sorted(root.glob(f"*{suffix}"))
        for path in entries[:-MAX_PACKET_HISTORY]:
            path.unlink(missing_ok=True)


def _run_local_recorder(
    settings: LocalContinuitySettings,
    state_root: Path,
    packet: dict[str, Any],
) -> dict[str, Any]:
    """Optionally ask a local recorder for a short summary inside a read-only sandbox.

    No repository directory is writable in the bubblewrap sandbox.  The output
    is informational only and is never interpreted as a stage completion.
    Missing bwrap, an invalid command, a timeout, or an invalid response simply
    keep the deterministic packet; continuity must never become a supervisor
    dependency.
    """

    if not settings.command:
        return {"execution": "not-configured", "fallback": "deterministic"}
    bwrap = shutil.which("bwrap")
    executable = shutil.which(settings.command[0])
    if not bwrap or not executable:
        return {"execution": "not-run", "fallback": "deterministic", "reason": "sandbox-or-command-unavailable"}
    # Only provide the already-redacted packet. The recorder has no tool or
    # filesystem capability for the product worktree and its stdout is capped.
    resolved_executable = str(Path(executable).resolve())
    sandbox_executable = "/run/ai-team-continuity-recorder"
    command = [bwrap, "--die-with-parent", "--unshare-all", "--unshare-net", "--new-session"]
    for runtime_path in ("/usr", "/bin", "/lib", "/lib64"):
        if Path(runtime_path).exists():
            command.extend(("--ro-bind", runtime_path, runtime_path))
    command.extend((
        "--dir", "/run", "--ro-bind", resolved_executable, sandbox_executable,
        "--bind", str(state_root), str(state_root), "--tmpfs", "/tmp",
        "--proc", "/proc", "--dev", "/dev", "--chdir", str(state_root),
        "--", sandbox_executable, *settings.command[1:],
    ))
    try:
        result = subprocess.run(
            command,
            input=json.dumps(packet, sort_keys=True),
            text=True,
            capture_output=True,
            timeout=settings.timeout_seconds,
            check=False,
            env={"PATH": "/usr/bin:/bin", "AI_TEAM_CONTINUITY_MODE": "recorder_only"},
        )
    except (OSError, subprocess.SubprocessError):
        return {"execution": "failed", "fallback": "deterministic"}
    if result.returncode != 0:
        return {"execution": "failed", "fallback": "deterministic"}
    summary = redact_secrets(result.stdout.strip())[: settings.max_output_tokens * 4]
    return {
        "execution": "completed", "fallback": "not-needed",
        "summary": summary, "repositoryModifications": "none",
    }


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _time(value: Any) -> datetime:
    if not isinstance(value, str):
        return datetime.min.replace(tzinfo=UTC)
    try:
        return _utc(datetime.fromisoformat(value))
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)


def _positive_int(value: Any, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else default


def _ratio(value: Any, default: float) -> float:
    return float(value) if isinstance(value, (int, float)) and 0 <= float(value) <= 0.5 else default


def _optional_string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _safe_slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value.lower()).strip("-")[:80] or "task"
