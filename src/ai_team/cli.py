from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from ai_team.core.orchestrator import Orchestrator, WorkflowError, load_workflow
from ai_team.core.git_policy import evaluate_git_action
from ai_team.core.github_executor import GitHubExecutionOptions, execute_github_action
from ai_team.core.ci_monitor import monitor_pull_request, write_repair_completion_receipt
from ai_team.core.delivery import DeliveryOptions, run_delivery_supervisor
from ai_team.core.bounded_delivery import BoundedDeliveryError, BoundedDeliveryOptions, DeliveryLimits, run_bounded_delivery
from ai_team.core.bounded_supervisor import ContinuousBoundedOptions, run_continuous_bounded_delivery
from ai_team.core.cloud_resilience import load_resilience_settings
from ai_team.core.isolated_executor import run_in_disposable_worktree
from ai_team.core.project_loader import ProjectConfigError, load_project
from ai_team.core.receipts import write_run_receipt
from ai_team.core.routing_config import ROLE_CHOICES, load_role_profile
from ai_team.core.staging_operations import run_staging_operations
from ai_team.core.supervisor import SupervisorOptions, run_supervisor
from ai_team.core.trusted_dev import (
    load_trusted_dev_settings,
    validate_trusted_dev_project,
)
from ai_team.core.watchdog import WatchdogThresholds, run_watchdog, send_windows_toast
from ai_team.core.watchdog_repair import AutoRepairOptions
from ai_team.providers import (
    AntigravityProvider,
    AntigravitySettings,
    CodexProvider,
    CodexSettings,
    HandsFreeCodeProvider,
    HandsFreeCodeSettings,
    MockProvider,
    OpenHandsProvider,
    OpenHandsSettings,
    RoleRouterProvider,
    RouterProvider,
    WriteSmokeProvider,
)
from ai_team.providers.base import redact_secrets


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml"


def validate_project(project: Path) -> None:
    if not project.exists():
        raise SystemExit(f"Project not found: {project}")
    if not project.is_dir():
        raise SystemExit(f"Target is not a directory: {project}")
    if not (project / ".git").exists():
        raise SystemExit(f"Target is not a Git repository: {project}")


def load_settings(path: Path = DEFAULT_SETTINGS_PATH) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def workspace_allowlist(settings: dict) -> list[str] | None:
    workspace = settings.get("workspace", {}) if isinstance(settings.get("workspace"), dict) else {}
    allowlist = workspace.get("allowlist")
    if not isinstance(allowlist, list):
        return None
    return [str(item) for item in allowlist if str(item).strip()]


def inspect_project(project_path: str) -> None:
    project = Path(project_path).resolve()
    validate_project(project)
    settings = load_settings()
    loaded = None
    profile_error = None
    try:
        loaded = load_project(project, allowlist=workspace_allowlist(settings))
    except ProjectConfigError as exc:
        profile_error = str(exc)

    print(f"Project name: {project.name}")
    print(f"Project path: {project}")
    print(f"Git repository: {(project / '.git').exists()}")
    print(f"package.json: {(project / 'package.json').exists()}")
    print(f"Prisma: {(project / 'prisma').exists()}")
    print(f".ai-team: {(project / '.ai-team').exists()}")
    if loaded:
        print("AI Team profile: valid")
        print(f"Root: {loaded.root}")
        print(f"Branch: {loaded.current_branch or 'unknown'}")
        print(f"Protected branch: {loaded.is_branch_protected()}")
    elif profile_error:
        print(f"AI Team profile: invalid ({profile_error})")


def init_project(project_path: str) -> None:
    project = Path(project_path).resolve()
    validate_project(project)

    ai_team_dir = project / ".ai-team"
    ai_team_dir.mkdir(exist_ok=True)

    project_yaml = ai_team_dir / "project.yaml"
    if project_yaml.exists():
        print(f"Project profile already exists, not overwritten: {project_yaml}")
        return

    content = f"""project:
  name: {project.name}
  root: "."
  stage: development

repository:
  protected_branches:
    - master
    - main

commands:
  install: npm install
  lint: npm run lint
  typecheck: npm run typecheck
  test: npm run test
  build: npm run build

safety:
  allow_git_push: false
  allow_deploy: false
  allow_database_migration: false
  allow_database_seed: false
  allow_destructive_commands: false
"""

    project_yaml.write_text(content, encoding="utf-8")
    print(f"AI team project profile created: {project_yaml}")


def validate_profile(project_path: str) -> None:
    settings = load_settings()
    loaded = load_project(project_path, allowlist=workspace_allowlist(settings))
    print("Project profile is valid")
    print(
        json.dumps(
            {
                "project": loaded.profile.project.name,
                "root": str(loaded.root),
                "branch": loaded.current_branch,
                "commitSha": loaded.commit_sha,
                "protectedBranch": loaded.is_branch_protected(),
                "disposableWorktree": loaded.is_disposable_worktree(),
                "allowGitPush": loaded.profile.safety.allow_git_push,
                "allowDeploy": loaded.profile.safety.allow_deploy,
            },
            indent=2,
        )
    )


def evaluate_git_policy(project_path: str, action: str, files: list[str] | None) -> None:
    settings = load_settings()
    loaded = load_project(project_path, allowlist=workspace_allowlist(settings))
    decision = evaluate_git_action(loaded, action, candidate_files=files or [])
    print(json.dumps(redact_secrets(decision.as_dict()), indent=2, default=str))
    if not decision.allowed:
        raise SystemExit(2)


def monitor_pr(
    project_path: str,
    repository: str,
    pr_identifier: str,
    report_dir: str | None,
    wait_seconds: int,
    poll_seconds: int,
) -> None:
    settings = load_settings()
    loaded = load_project(project_path, allowlist=workspace_allowlist(settings))
    result = monitor_pull_request(
        project_root=loaded.root,
        repository=repository,
        pr_identifier=pr_identifier,
        report_dir=Path(report_dir).resolve() if report_dir else REPO_ROOT / "reports" / "ci-monitor",
        wait_seconds=wait_seconds,
        poll_seconds=poll_seconds,
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "mergeReady": result.merge_ready,
                "evidencePath": str(result.evidence_path),
                "repairTaskPath": str(result.repair_task_path) if result.repair_task_path else None,
                "blockers": result.evidence.get("blockers", []),
                "failureEvidence": [
                    {
                        "check": item.get("check"),
                        "workflow": item.get("workflow"),
                        "classification": item.get("classification"),
                        "runId": item.get("runId"),
                    }
                    for item in result.evidence.get("failureEvidence", [])
                ],
            },
            indent=2,
            default=str,
        )
    )


def create_repair_receipt(
    project_path: str,
    repair_task_path: str,
    final_ci_evidence_path: str,
    report_dir: str | None,
) -> None:
    settings = load_settings()
    loaded = load_project(project_path, allowlist=workspace_allowlist(settings))
    path = write_repair_completion_receipt(
        loaded.root,
        Path(repair_task_path).resolve(),
        Path(final_ci_evidence_path).resolve(),
        Path(report_dir).resolve() if report_dir else REPO_ROOT / "reports" / "ci-monitor",
    )
    print(json.dumps({"receiptPath": str(path)}, indent=2))


def check_watchdog(
    supervisor_state: str,
    watchdog_state: str,
    alert_log: str,
    report_dir: str,
    service: str,
    repeat_count: int,
    restart_count: int,
    stale_minutes: int,
    cooldown_minutes: int,
    powershell_path: str,
    test_notification: bool,
    auto_repair: bool,
    project: str | None,
    contract_dir: str | None,
    repair_backup_dir: str | None,
    max_auto_repair_attempts: int,
) -> None:
    if test_notification:
        delivered = send_windows_toast(
            "AI Team 提醒測試",
            "Windows 桌面通知已成功啟用；這次測試不會呼叫任何 AI 模型。",
            powershell_path=powershell_path,
        )
        print(json.dumps({"status": "tested", "notificationDelivered": delivered}, indent=2))
        return
    if auto_repair and not all((project, contract_dir, repair_backup_dir)):
        raise ValueError(
            "watchdog --auto-repair requires --project, --contract-dir, and --repair-backup-dir"
        )

    result = run_watchdog(
        Path(supervisor_state).resolve(),
        Path(watchdog_state).resolve(),
        Path(alert_log).resolve(),
        service_name=service,
        report_dir=Path(report_dir).resolve(),
        thresholds=WatchdogThresholds(
            repeat_count=repeat_count,
            restart_count=restart_count,
            stale_seconds=stale_minutes * 60,
            cooldown_seconds=cooldown_minutes * 60,
        ),
        powershell_path=powershell_path,
        auto_repair=AutoRepairOptions(
            enabled=auto_repair,
            project_path=Path(project).resolve() if project else None,
            contract_dir=Path(contract_dir).resolve() if contract_dir else None,
            backup_dir=Path(repair_backup_dir).resolve() if repair_backup_dir else None,
            max_attempts=max_auto_repair_attempts,
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_openhands_provider(settings: dict) -> OpenHandsProvider:
    openhands = settings.get("openhands", {}) if isinstance(settings.get("openhands"), dict) else {}
    provider_settings = OpenHandsSettings(
        base_url=str(openhands.get("base_url") or "http://127.0.0.1:31024"),
        session_key_env=str(openhands.get("session_key_env") or "SESSION_API_KEY"),
        session_key_file=str(openhands.get("session_key_file") or "") or None,
        ready_path=str(openhands.get("ready_path") or "/ready"),
        conversation_path=str(openhands.get("conversation_path") or "/api/conversations"),
        cancel_path_template=str(
            openhands.get("cancel_path_template") or "/api/conversations/{task_id}/interrupt"
        ),
        timeout_seconds=float(openhands.get("timeout_seconds") or 30),
        host_workspace_root=str(openhands.get("host_workspace_root") or "C:/Users/eden/Downloads/AI"),
        container_workspace_root=str(openhands.get("container_workspace_root") or "/projects"),
        llm_model=str(openhands.get("llm_model") or "openai/gpt-5.5"),
        llm_api_key_env=str(openhands.get("llm_api_key_env") or "OPENHANDS_LLM_API_KEY"),
        llm_api_key=str(openhands.get("llm_api_key") or "placeholder-not-a-real-secret"),
    )
    return OpenHandsProvider(provider_settings)


def build_handsfreecode_provider(settings: dict) -> HandsFreeCodeProvider:
    handsfreecode = settings.get("handsfreecode", {}) if isinstance(settings.get("handsfreecode"), dict) else {}
    provider_settings = HandsFreeCodeSettings(
        base_url=str(handsfreecode.get("base_url") or "http://127.0.0.1:31025"),
        session_key_env=str(handsfreecode.get("session_key_env") or "HANDSFREECODE_SESSION_API_KEY"),
        session_key_file=str(handsfreecode.get("session_key_file") or "~/.handsfreecode/session-api-key.txt"),
        ready_path=str(handsfreecode.get("ready_path") or "/ready"),
        task_run_path=str(handsfreecode.get("task_run_path") or "/api/tasks/run"),
        timeout_seconds=float(handsfreecode.get("timeout_seconds") or 30),
        default_runtime_provider=str(handsfreecode.get("default_runtime_provider") or "mock"),
    )
    return HandsFreeCodeProvider(provider_settings)


def _string_list(value, fallback: list[str]) -> list[str]:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return fallback


def build_codex_provider(settings: dict) -> CodexProvider:
    codex = settings.get("codex", {}) if isinstance(settings.get("codex"), dict) else {}
    provider_settings = CodexSettings(
        executable=_resolve_codex_executable(str(codex.get("executable") or "codex")),
        status_args=_string_list(codex.get("status_args"), ["--version"]),
        quota_args=_string_list(codex.get("quota_args"), ["doctor", "--json"]),
        run_args=_string_list(codex.get("run_args"), ["exec", "--sandbox", "read-only", "--skip-git-repo-check"]),
        write_run_args=_string_list(
            codex.get("write_run_args"),
            ["exec", "--sandbox", "workspace-write", "--skip-git-repo-check"],
        ),
        timeout_seconds=float(codex.get("timeout_seconds") or 45),
        run_timeout_seconds=float(codex.get("run_timeout_seconds") or 180),
        execution_enabled=bool(codex.get("execution_enabled", True)),
        allowed_models=tuple(_string_list(codex.get("allowed_models"), [])),
        allowed_reasoning_efforts=tuple(
            _string_list(codex.get("allowed_reasoning_efforts"), ["low", "medium", "high", "xhigh"])
        ),
    )
    return CodexProvider(provider_settings)


def _resolve_codex_executable(configured: str) -> str:
    if configured != "auto-native":
        return configured
    extension_root = Path.home() / ".vscode" / "extensions"
    candidates = sorted(
        extension_root.glob("openai.chatgpt-*/bin/windows-x86_64/codex.exe"),
        reverse=True,
    )
    return str(candidates[0]) if candidates else "__codex_vscode_native_not_found__"


def build_antigravity_provider(settings: dict) -> AntigravityProvider:
    antigravity = settings.get("antigravity", {}) if isinstance(settings.get("antigravity"), dict) else {}
    provider_settings = AntigravitySettings(
        executable=str(antigravity.get("executable") or "antigravity"),
        status_args=_string_list(antigravity.get("status_args"), ["auth", "status"]),
        quota_args=_string_list(antigravity.get("quota_args"), ["quota"]),
        run_args=_string_list(antigravity.get("run_args"), []),
        timeout_seconds=float(antigravity.get("timeout_seconds") or 45),
        run_timeout_seconds=float(antigravity.get("run_timeout_seconds") or 180),
        execution_enabled=bool(antigravity.get("execution_enabled", False)),
        prompt_max_chars=int(antigravity.get("prompt_max_chars") or 1200),
        diagnostics_cache_ttl_seconds=float(antigravity.get("diagnostics_cache_ttl_seconds") or 30),
        read_only_sandbox_executable=(
            str(antigravity["read_only_sandbox_executable"])
            if isinstance(antigravity.get("read_only_sandbox_executable"), str)
            else None
        ),
        allowed_models=tuple(_string_list(antigravity.get("allowed_models"), [])),
        allowed_reasoning_efforts=tuple(
            _string_list(
                antigravity.get("allowed_reasoning_efforts"),
                ["low", "medium", "high", "thinking"],
            )
        ),
    )
    return AntigravityProvider(provider_settings)


def build_provider(provider_name: str, settings: dict, role: str | None = None):
    if role and provider_name != "auto":
        raise WorkflowError("--role requires --provider auto so routing decisions remain auditable")
    providers = {
        "codex": build_codex_provider(settings),
        "antigravity": build_antigravity_provider(settings),
        "handsfreecode": build_handsfreecode_provider(settings),
    }
    if role:
        return RoleRouterProvider(load_role_profile(settings, role), providers)
    if provider_name == "auto":
        # OpenHands remains available only by explicit request. Mock is a test
        # provider and must never masquerade as an automatic production route.
        return RouterProvider(list(providers.values()))
    if provider_name == "mock":
        return MockProvider()
    if provider_name == "write-smoke":
        return WriteSmokeProvider()
    if provider_name == "handsfreecode":
        return build_handsfreecode_provider(settings)
    if provider_name == "codex":
        return build_codex_provider(settings)
    if provider_name == "antigravity":
        return build_antigravity_provider(settings)
    return build_openhands_provider(settings)


def doctor() -> None:
    settings = load_settings()
    openhands_diagnostics = build_openhands_provider(settings).diagnostics()
    openhands = {
        "provider": "openhands",
        "ready": False,
        "externalRequired": False,
        "enabledByPolicy": False,
        "deprecated": True,
        "status": "disabled_by_policy",
        "explicitProviderDiagnostics": openhands_diagnostics,
    }
    handsfreecode = build_handsfreecode_provider(settings).diagnostics()
    codex = build_codex_provider(settings).diagnostics()
    antigravity = build_antigravity_provider(settings).diagnostics()
    auto = build_provider("auto", settings).diagnostics()
    print(
        json.dumps(
            {
                "settings": str(DEFAULT_SETTINGS_PATH),
                "openhands": openhands,
                "handsfreecode": handsfreecode,
                "codex": codex,
                "antigravity": antigravity,
                "auto": auto,
                "providerNative": {
                    "openhands": {
                        "ready": False,
                        "externalRequired": False,
                        "enabledByPolicy": False,
                        "deprecated": True,
                        "status": "disabled_by_policy",
                    },
                    "handsfreecode": {
                        "ready": handsfreecode.get("ready") is True
                        and handsfreecode.get("authConfigured") is True
                        and handsfreecode.get("sessionKeyPresent") is True,
                        "externalRequired": not (
                            handsfreecode.get("ready") is True
                            and handsfreecode.get("authConfigured") is True
                            and handsfreecode.get("sessionKeyPresent") is True
                        ),
                    },
                    "codex": {
                        "ready": codex.get("ready") is True,
                        "externalRequired": not codex.get("ready"),
                        "quotaExhausted": codex.get("quotaExhausted") is True,
                        "resetTime": codex.get("resetTime"),
                    },
                    "antigravity": {
                        "ready": antigravity.get("ready") is True,
                        "externalRequired": not antigravity.get("ready"),
                        "quotaExhausted": antigravity.get("quotaExhausted") is True,
                        "resetTime": antigravity.get("resetTime"),
                    },
                },
            },
            indent=2,
            default=str,
        )
    )


def run_workflow(
    project_path: str,
    workflow_name: str,
    provider_name: str,
    dry_run: bool,
    receipt_dir: str | None,
    mode: str,
    role: str | None = None,
) -> None:
    settings = load_settings()
    loaded = load_project(project_path, allowlist=workspace_allowlist(settings))
    load_workflow(workflow_name)
    provider = build_provider(provider_name, settings, role)
    result = Orchestrator(provider=provider, max_retries=2).run(
        loaded,
        workflow_name=workflow_name,
        dry_run=dry_run,
        timeout_seconds=None,
        run_mode=mode,
    )
    receipt_path = write_run_receipt(
        loaded,
        result,
        Path(receipt_dir).resolve() if receipt_dir else REPO_ROOT / "reports" / "receipts",
    )
    payload = {
        "workflow": result.workflow.name,
        "dryRun": result.dry_run,
        "stages": result.stages,
        "provider": result.provider_result.provider,
        "runMode": mode,
        "success": result.provider_result.success,
        "attempts": result.provider_result.attempts,
        "errorType": result.provider_result.error_type,
        "conversationId": result.provider_result.conversation_id,
        "taskId": result.provider_result.task_id,
        "content": _safe_stdout_content(result.provider_result.content),
        "data": redact_secrets(result.provider_result.data),
        "receiptPath": str(receipt_path),
    }
    print(json.dumps(redact_secrets(payload), indent=2, default=str))
    if not result.provider_result.success:
        raise SystemExit(2)


def run_isolated_workflow(
    project_path: str,
    workflow_name: str,
    provider_name: str,
    dry_run: bool,
    receipt_dir: str | None,
    mode: str,
    worktree_parent: str | None,
    keep_worktree: bool,
    auto_commit: bool,
    commit_message: str | None,
    github_action: str | None,
    github_execute: bool,
    github_branch: str | None,
    validation_log_hash: str | None,
    test_evidence_hash: str | None,
    role: str | None = None,
) -> None:
    settings = load_settings()
    provider = build_provider(provider_name, settings, role)
    result = run_in_disposable_worktree(
        source_project_path=project_path,
        provider=provider,
        workflow_name=workflow_name,
        workspace_allowlist=workspace_allowlist(settings),
        receipt_dir=Path(receipt_dir).resolve() if receipt_dir else REPO_ROOT / "reports" / "isolated",
        worktree_parent=Path(worktree_parent).resolve() if worktree_parent else None,
        dry_run=dry_run,
        run_mode=mode,
        keep_worktree=keep_worktree,
        auto_commit=auto_commit,
        commit_message=commit_message,
        github_action=github_action,
        github_execute=github_execute,
        github_branch=github_branch,
        validation_log_hash=validation_log_hash,
        test_evidence_hash=test_evidence_hash,
    )
    payload = {
        "workflow": result.workflow_result.workflow.name,
        "provider": result.workflow_result.provider_result.provider,
        "runMode": mode,
        "dryRun": dry_run,
        "success": result.workflow_result.provider_result.success,
        "errorType": result.workflow_result.provider_result.error_type,
        "worktreePath": str(result.worktree_path),
        "runReceipt": str(result.run_receipt),
        "executorReceipt": str(result.executor_receipt),
        "gitPolicy": result.git_policy,
        "commitResult": result.commit_result,
        "githubResult": result.github_result,
    }
    print(json.dumps(redact_secrets(payload), indent=2, default=str))
    if not result.workflow_result.provider_result.success or (
        result.github_result is not None and not result.github_result.get("success", False)
    ):
        raise SystemExit(2)


def evaluate_github_gate(
    project_path: str,
    action: str,
    dry_run: bool,
    validation_log_hash: str | None,
    receipt_path: str | None,
    test_evidence_hash: str | None,
    pr_identifier: str | None,
) -> None:
    settings = load_settings()
    loaded = load_project(project_path, allowlist=workspace_allowlist(settings))
    result = execute_github_action(
        loaded,
        GitHubExecutionOptions(
            action=action,
            dry_run=dry_run,
            validation_log_hash=validation_log_hash,
            receipt_path=Path(receipt_path).resolve() if receipt_path else None,
            test_evidence_hash=test_evidence_hash,
            pr_identifier=pr_identifier,
        ),
    )
    print(json.dumps(redact_secrets(result.as_dict()), indent=2, default=str))
    if not result.success:
        raise SystemExit(2)


def supervise(
    project_path: str,
    workflow_name: str,
    provider_name: str,
    dry_run: bool,
    mode: str,
    once: bool,
    interval_minutes: int,
    max_runtime_minutes: int | None,
    report_dir: str | None,
    state_path: str | None,
    isolated_auto_commit: bool,
    github_action: str | None,
    github_execute: bool,
    validation_log_hash: str | None,
    test_evidence_hash: str | None,
    delivery: bool,
    bounded_delivery: bool,
    task_contract: str | None,
    task_contract_dir: str | None,
    max_iterations: int,
    max_repair_attempts: int,
    max_token_usage: int,
    stage_timeout_seconds: int,
    auto_merge: bool,
    allow_unreviewed_development_merge: bool,
    ci_wait_seconds: int,
    ci_poll_seconds: int,
    trusted_dev_autopilot: bool,
    autonomous_product_loop: bool,
    role: str | None = None,
) -> None:
    settings = load_settings()
    selected_report_dir = Path(report_dir).resolve() if report_dir else REPO_ROOT / "reports" / "supervisor"
    if trusted_dev_autopilot and not bounded_delivery:
        raise WorkflowError("--trusted-dev-autopilot requires --bounded-delivery")
    if autonomous_product_loop and not (bounded_delivery and trusted_dev_autopilot and not once):
        raise WorkflowError(
            "--autonomous-product-loop requires continuous --bounded-delivery with --trusted-dev-autopilot"
        )
    if bounded_delivery:
        if delivery:
            raise WorkflowError("--bounded-delivery cannot be combined with legacy --delivery")
        if provider_name != "auto" or role is not None:
            raise WorkflowError("bounded delivery selects audited role routes internally; use --provider auto without --role")
        if mode != "create-only":
            raise WorkflowError("bounded delivery owns its stage modes; do not pass --mode run-agent")
        if isolated_auto_commit or github_action:
            raise WorkflowError("bounded delivery never accepts legacy auto-commit or GitHub action flags")
        if dry_run:
            raise WorkflowError("bounded delivery requires explicit --execute")
        resolved_project = Path(project_path).resolve()
        try:
            trusted_dev = load_trusted_dev_settings(
                settings,
                resolved_project,
                requested=trusted_dev_autopilot,
            )
        except ValueError as exc:
            raise WorkflowError(str(exc)) from None
        if trusted_dev.enabled:
            loaded = load_project(
                resolved_project,
                allowlist=workspace_allowlist(settings),
            )
            try:
                validate_trusted_dev_project(loaded)
            except ValueError as exc:
                raise WorkflowError(str(exc)) from None
        limits = DeliveryLimits(
            max_iterations=max_iterations,
            max_repair_attempts=max_repair_attempts,
            max_token_usage=max_token_usage,
            timeout_seconds=stage_timeout_seconds,
        )
        if trusted_dev.enabled:
            limits = DeliveryLimits(
                max_iterations=max(limits.max_iterations, trusted_dev.min_iterations),
                max_repair_attempts=max(
                    limits.max_repair_attempts,
                    trusted_dev.min_repair_attempts,
                ),
                max_token_usage=max(limits.max_token_usage, trusted_dev.min_token_usage),
                timeout_seconds=max(
                    limits.timeout_seconds,
                    trusted_dev.min_stage_timeout_seconds,
                ),
            )
        if min(limits.max_iterations, limits.max_repair_attempts, limits.max_token_usage, limits.timeout_seconds) < 1:
            raise WorkflowError("bounded delivery limits must all be positive")
        if not once:
            if task_contract:
                raise WorkflowError("continuous bounded delivery uses --task-contract-dir, not --task-contract")
            if not task_contract_dir:
                raise WorkflowError("continuous bounded delivery requires --task-contract-dir")
            selected_state_path = (
                Path(state_path).resolve()
                if state_path
                else selected_report_dir / "continuous-bounded-delivery-state.json"
            )
            try:
                cloud_routes, cloud_retry, local_continuity = load_resilience_settings(settings)
                result = run_continuous_bounded_delivery(
                    ContinuousBoundedOptions(
                        project_path=resolved_project,
                        contract_dir=Path(task_contract_dir).resolve(),
                        provider_for_role=lambda selected_role: build_provider("auto", settings, selected_role),
                        workspace_allowlist=workspace_allowlist(settings),
                        report_dir=selected_report_dir / "continuous-bounded-delivery",
                        state_path=selected_state_path,
                        limits=limits,
                        once=False,
                        interval_minutes=interval_minutes,
                        max_runtime_minutes=max_runtime_minutes,
                        github_execute=github_execute,
                        auto_merge=auto_merge,
                        allow_unreviewed_development_merge=allow_unreviewed_development_merge,
                        ci_wait_seconds=ci_wait_seconds,
                        ci_poll_seconds=ci_poll_seconds,
                        cloud_routes=cloud_routes,
                        cloud_retry=cloud_retry,
                        local_continuity=local_continuity,
                        trusted_dev=trusted_dev,
                        autonomous_product_loop=autonomous_product_loop,
                    )
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise WorkflowError(str(redact_secrets(str(exc)))) from None
            print(json.dumps(redact_secrets(result), indent=2, default=str))
            return
        if task_contract_dir:
            raise WorkflowError("one-shot bounded delivery accepts --task-contract only")
        if not task_contract:
            raise WorkflowError("bounded delivery requires --task-contract")
        if github_execute or auto_merge or allow_unreviewed_development_merge:
            raise WorkflowError("one-shot bounded delivery leaves GitHub publication to its caller")
        selected_state_path = Path(state_path).resolve() if state_path else selected_report_dir / "bounded-delivery-state.json"
        result = run_bounded_delivery(
            BoundedDeliveryOptions(
                project_path=resolved_project,
                task_contract_path=Path(task_contract).resolve(),
                provider_for_role=lambda selected_role: build_provider("auto", settings, selected_role),
                workspace_allowlist=workspace_allowlist(settings),
                report_dir=selected_report_dir / "bounded-delivery",
                state_path=selected_state_path,
                limits=limits,
                trusted_dev=trusted_dev,
            )
        )
        print(json.dumps(redact_secrets(result), indent=2, default=str))
        return
    provider = build_provider(provider_name, settings, role)
    if delivery:
        selected_state_path = Path(state_path).resolve() if state_path else selected_report_dir / "delivery-state.json"
        result = run_delivery_supervisor(
            DeliveryOptions(
                project_path=Path(project_path).resolve(),
                provider=provider,
                workspace_allowlist=workspace_allowlist(settings),
                report_dir=selected_report_dir,
                state_path=selected_state_path,
                queue_path=selected_report_dir / "trusted-task-queue.json",
                once=once,
                interval_minutes=interval_minutes,
                max_runtime_minutes=max_runtime_minutes,
                github_execute=github_execute,
            )
        )
        print(json.dumps(redact_secrets(result), indent=2, default=str))
        return
    summary = run_supervisor(
        SupervisorOptions(
            project_path=Path(project_path).resolve(),
            provider=provider,
            workflow=workflow_name,
            run_mode=mode,
            dry_run=dry_run,
            once=once,
            interval_minutes=interval_minutes,
            max_runtime_minutes=max_runtime_minutes,
            report_dir=selected_report_dir,
            state_path=Path(state_path).resolve() if state_path else None,
            workspace_allowlist=workspace_allowlist(settings),
            isolated_auto_commit=isolated_auto_commit,
            github_action=github_action,
            github_execute=github_execute,
            validation_log_hash=validation_log_hash,
            test_evidence_hash=test_evidence_hash,
        )
    )
    print(
        json.dumps(
            {
                "completedCycles": summary.completed_cycles,
                "stoppedReason": summary.stopped_reason,
                "reportPaths": [str(path) for path in summary.report_paths],
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-team",
        description="Reusable AI software team orchestrator",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a target project")
    inspect_parser.add_argument("project", nargs="?", default=".")

    init_parser = subparsers.add_parser("init", help="Create .ai-team project settings")
    init_parser.add_argument("project", nargs="?", default=".")

    validate_parser = subparsers.add_parser("validate", help="Validate .ai-team/project.yaml")
    validate_parser.add_argument("project", nargs="?", default=".")

    subparsers.add_parser("doctor", help="Check provider settings and provider-native loopback status")

    monitor_parser = subparsers.add_parser("pr-monitor", help="Collect PR checks and produce guarded CI evidence")
    monitor_parser.add_argument("project", nargs="?", default=".")
    monitor_parser.add_argument("--repo", required=True)
    monitor_parser.add_argument("--pr", required=True)
    monitor_parser.add_argument("--report-dir")
    monitor_parser.add_argument("--wait-seconds", type=int, default=0)
    monitor_parser.add_argument("--poll-seconds", type=int, default=10)

    repair_receipt_parser = subparsers.add_parser(
        "repair-receipt",
        help="Create an attested completion receipt for an exact-path CI repair",
    )
    repair_receipt_parser.add_argument("project", nargs="?", default=".")
    repair_receipt_parser.add_argument("--task-path", required=True)
    repair_receipt_parser.add_argument("--final-ci-evidence", required=True)
    repair_receipt_parser.add_argument("--report-dir")

    watchdog_parser = subparsers.add_parser(
        "watchdog",
        help="Detect repeated supervisor stalls and send a deduplicated Windows notification",
    )
    watchdog_parser.add_argument("--supervisor-state", required=True)
    watchdog_parser.add_argument("--watchdog-state", required=True)
    watchdog_parser.add_argument("--alert-log", required=True)
    watchdog_parser.add_argument("--report-dir", required=True)
    watchdog_parser.add_argument("--service", required=True)
    watchdog_parser.add_argument("--repeat-count", type=int, default=3)
    watchdog_parser.add_argument("--restart-count", type=int, default=3)
    watchdog_parser.add_argument("--stale-minutes", type=int, default=25)
    watchdog_parser.add_argument("--cooldown-minutes", type=int, default=30)
    watchdog_parser.add_argument("--powershell-path", default="powershell.exe")
    watchdog_parser.add_argument("--test-notification", action="store_true")
    watchdog_parser.add_argument("--auto-repair", action="store_true")
    watchdog_parser.add_argument("--project")
    watchdog_parser.add_argument("--contract-dir")
    watchdog_parser.add_argument("--repair-backup-dir")
    watchdog_parser.add_argument("--max-auto-repair-attempts", type=int, default=2)

    git_policy_parser = subparsers.add_parser("git-policy", help="Evaluate guarded git automation policy")
    git_policy_parser.add_argument("project", nargs="?", default=".")
    git_policy_parser.add_argument(
        "--action",
        choices=["add", "commit", "push", "pr", "pull-request", "merge"],
        required=True,
    )
    git_policy_parser.add_argument("--file", action="append", default=[])

    run_parser = subparsers.add_parser("run", help="Run a guarded workflow")
    run_parser.add_argument("project", nargs="?", default=".")
    run_parser.add_argument("--workflow", required=True)
    run_parser.add_argument(
        "--provider",
        choices=["auto", "mock", "openhands", "handsfreecode", "codex", "antigravity"],
        default="mock",
    )
    run_parser.add_argument(
        "--mode",
        choices=["create-only", "read-only-agent", "run-agent"],
        default="create-only",
    )
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--receipt-dir")
    run_parser.add_argument("--role", choices=ROLE_CHOICES)

    isolated_parser = subparsers.add_parser("isolated-run", help="Run a write workflow in a disposable worktree")
    isolated_parser.add_argument("project", nargs="?", default=".")
    isolated_parser.add_argument("--workflow", required=True)
    isolated_parser.add_argument(
        "--provider",
        choices=["auto", "mock", "write-smoke", "openhands", "handsfreecode", "codex", "antigravity"],
        default="mock",
    )
    isolated_parser.add_argument("--mode", choices=["create-only", "run-agent"], default="create-only")
    isolated_parser.add_argument("--dry-run", action="store_true")
    isolated_parser.add_argument("--receipt-dir")
    isolated_parser.add_argument("--worktree-parent")
    isolated_parser.add_argument("--remove-worktree", action="store_false", dest="keep_worktree")
    isolated_parser.set_defaults(keep_worktree=True)
    isolated_parser.add_argument("--auto-commit", action="store_true")
    isolated_parser.add_argument("--commit-message")
    isolated_parser.add_argument("--github-action", choices=["push", "pr", "pull-request", "merge"])
    isolated_parser.add_argument("--github-execute", action="store_true")
    isolated_parser.add_argument("--github-branch")
    isolated_parser.add_argument("--validation-log-hash")
    isolated_parser.add_argument("--test-evidence-hash")
    isolated_parser.add_argument("--role", choices=ROLE_CHOICES)

    github_parser = subparsers.add_parser("github-gate", help="Evaluate guarded GitHub automation policy")
    github_parser.add_argument("project", nargs="?", default=".")
    github_parser.add_argument("--action", choices=["push", "pr", "pull-request", "merge"], required=True)
    github_parser.add_argument("--execute", action="store_false", dest="dry_run")
    github_parser.set_defaults(dry_run=True)
    github_parser.add_argument("--validation-log-hash")
    github_parser.add_argument("--receipt-path")
    github_parser.add_argument("--test-evidence-hash")
    github_parser.add_argument("--pr-identifier")

    supervisor_parser = subparsers.add_parser("supervise", help="Run autonomous safe supervisor loop")
    supervisor_parser.add_argument("project", nargs="?", default=".")
    supervisor_parser.add_argument("--workflow", default="project-analysis")
    supervisor_parser.add_argument(
        "--provider",
        choices=["auto", "mock", "openhands", "handsfreecode", "codex", "antigravity"],
        default="mock",
    )
    supervisor_parser.add_argument("--role", choices=ROLE_CHOICES)
    supervisor_parser.add_argument("--mode", choices=["create-only", "run-agent"], default="create-only")
    supervisor_parser.add_argument("--dry-run", action="store_true", default=True)
    supervisor_parser.add_argument("--execute", action="store_false", dest="dry_run")
    supervisor_parser.add_argument("--once", action="store_true")
    supervisor_parser.add_argument("--interval-minutes", type=int, default=60)
    supervisor_parser.add_argument("--max-runtime-minutes", type=int)
    supervisor_parser.add_argument("--report-dir")
    supervisor_parser.add_argument("--state-path")
    supervisor_parser.add_argument("--auto-commit", action="store_true")
    supervisor_parser.add_argument("--github-action", choices=["push", "pr", "pull-request", "merge"])
    supervisor_parser.add_argument("--github-execute", action="store_true")
    supervisor_parser.add_argument("--validation-log-hash")
    supervisor_parser.add_argument("--test-evidence-hash")
    supervisor_parser.add_argument(
        "--delivery",
        action="store_true",
        help="Discover and execute trusted product tasks in disposable worktrees",
    )
    supervisor_parser.add_argument(
        "--bounded-delivery",
        action="store_true",
        help="Run explicit trusted tasks through the fail-closed multi-role delivery loop",
    )
    supervisor_parser.add_argument("--task-contract", help="Path to a schemaVersion=1 trusted task contract JSON file")
    supervisor_parser.add_argument(
        "--task-contract-dir",
        help="Directory of ordered schemaVersion=1 trusted task contracts for continuous bounded delivery",
    )
    supervisor_parser.add_argument("--max-iterations", type=int, default=2)
    supervisor_parser.add_argument("--max-repair-attempts", type=int, default=1)
    supervisor_parser.add_argument("--max-token-usage", type=int, default=120000)
    supervisor_parser.add_argument("--stage-timeout-seconds", type=int, default=180)
    supervisor_parser.add_argument("--auto-merge", action="store_true")
    supervisor_parser.add_argument(
        "--allow-unreviewed-development-merge",
        action="store_true",
        help="Allow CI-green merges without a human review only when project.stage=development",
    )
    supervisor_parser.add_argument("--ci-wait-seconds", type=int, default=900)
    supervisor_parser.add_argument("--ci-poll-seconds", type=int, default=10)
    supervisor_parser.add_argument(
        "--trusted-dev-autopilot",
        action="store_true",
        help="Enable allowlisted development-only long-run execution with Git checkpoints",
    )
    supervisor_parser.add_argument(
        "--autonomous-product-loop",
        action="store_true",
        help="When the trusted task queue is empty, let a read-only PM generate one validated next task",
    )

    staging_parser = subparsers.add_parser(
        "staging-operations",
        help="Run an explicit deterministic staging migration, seed, or Preview deployment",
    )
    staging_parser.add_argument("project", nargs="?", default=".")
    staging_parser.add_argument("--contract", required=True, help="Path to an ai-team-staging-operations/v1 contract")
    staging_parser.add_argument("--report-dir")
    staging_parser.add_argument("--execute", action="store_true", help="Execute fixed allowlisted commands after validation")

    return parser


def _safe_stdout_content(content: str) -> str:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        redacted = redact_secrets(content)
    else:
        redacted = json.dumps(redact_secrets(parsed), default=str)
    return redacted if isinstance(redacted, str) else ""


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "inspect":
            inspect_project(args.project)
        elif args.command == "init":
            init_project(args.project)
        elif args.command == "validate":
            validate_profile(args.project)
        elif args.command == "doctor":
            doctor()
        elif args.command == "pr-monitor":
            monitor_pr(
                args.project,
                args.repo,
                args.pr,
                args.report_dir,
                args.wait_seconds,
                args.poll_seconds,
            )
        elif args.command == "repair-receipt":
            create_repair_receipt(
                args.project,
                args.task_path,
                args.final_ci_evidence,
                args.report_dir,
            )
        elif args.command == "watchdog":
            check_watchdog(
                args.supervisor_state,
                args.watchdog_state,
                args.alert_log,
                args.report_dir,
                args.service,
                args.repeat_count,
                args.restart_count,
                args.stale_minutes,
                args.cooldown_minutes,
                args.powershell_path,
                args.test_notification,
                args.auto_repair,
                args.project,
                args.contract_dir,
                args.repair_backup_dir,
                args.max_auto_repair_attempts,
            )
        elif args.command == "git-policy":
            evaluate_git_policy(args.project, args.action, args.file)
        elif args.command == "run":
            run_workflow(
                args.project,
                args.workflow,
                args.provider,
                args.dry_run,
                args.receipt_dir,
                args.mode,
                args.role,
            )
        elif args.command == "isolated-run":
            run_isolated_workflow(
                args.project,
                args.workflow,
                args.provider,
                args.dry_run,
                args.receipt_dir,
                args.mode,
                args.worktree_parent,
                args.keep_worktree,
                args.auto_commit,
                args.commit_message,
                args.github_action,
                args.github_execute,
                args.github_branch,
                args.validation_log_hash,
                args.test_evidence_hash,
                args.role,
            )
        elif args.command == "github-gate":
            evaluate_github_gate(
                args.project,
                args.action,
                args.dry_run,
                args.validation_log_hash,
                args.receipt_path,
                args.test_evidence_hash,
                args.pr_identifier,
            )
        elif args.command == "supervise":
            supervise(
                args.project,
                args.workflow,
                args.provider,
                args.dry_run,
                args.mode,
                args.once,
                args.interval_minutes,
                args.max_runtime_minutes,
                args.report_dir,
                args.state_path,
                args.auto_commit,
                args.github_action,
                args.github_execute,
                args.validation_log_hash,
                args.test_evidence_hash,
                args.delivery,
                args.bounded_delivery,
                args.task_contract,
                args.task_contract_dir,
                args.max_iterations,
                args.max_repair_attempts,
                args.max_token_usage,
                args.stage_timeout_seconds,
                args.auto_merge,
                args.allow_unreviewed_development_merge,
                args.ci_wait_seconds,
                args.ci_poll_seconds,
                args.trusted_dev_autopilot,
                args.autonomous_product_loop,
                args.role,
            )
        elif args.command == "staging-operations":
            settings = load_settings()
            result = run_staging_operations(
                Path(args.project),
                Path(args.contract),
                Path(args.report_dir).resolve() if args.report_dir else REPO_ROOT / "reports" / "staging-operations",
                execute=args.execute,
                workspace_allowlist=workspace_allowlist(settings),
            )
            print(
                json.dumps(
                    {
                        "success": result.success,
                        "stopReason": result.stop_reason,
                        "validationKind": result.validation_kind,
                        "contractSha": result.contract_sha,
                        "receiptPath": str(result.receipt_path),
                    },
                    indent=2,
                )
            )
            if not result.success:
                raise SystemExit(2)
    except (BoundedDeliveryError, ProjectConfigError, WorkflowError) as exc:
        print(f"ai-team error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
