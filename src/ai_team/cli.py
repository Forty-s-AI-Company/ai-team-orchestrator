from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from ai_team.core.orchestrator import Orchestrator, WorkflowError, load_workflow
from ai_team.core.git_policy import evaluate_git_action
from ai_team.core.github_gate import evaluate_github_action
from ai_team.core.github_executor import GitHubExecutionOptions, execute_github_action
from ai_team.core.isolated_executor import run_in_disposable_worktree
from ai_team.core.project_loader import ProjectConfigError, load_project
from ai_team.core.receipts import write_run_receipt
from ai_team.core.supervisor import SupervisorOptions, run_supervisor
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
    RouterProvider,
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
        executable=str(codex.get("executable") or "codex"),
        status_args=_string_list(codex.get("status_args"), ["--version"]),
        quota_args=_string_list(codex.get("quota_args"), ["doctor", "--json"]),
        run_args=_string_list(codex.get("run_args"), ["exec", "--sandbox", "read-only", "--skip-git-repo-check"]),
        timeout_seconds=float(codex.get("timeout_seconds") or 45),
        run_timeout_seconds=float(codex.get("run_timeout_seconds") or 180),
        execution_enabled=bool(codex.get("execution_enabled", True)),
    )
    return CodexProvider(provider_settings)


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
    )
    return AntigravityProvider(provider_settings)


def build_provider(provider_name: str, settings: dict):
    if provider_name == "auto":
        return RouterProvider(
            [
                build_codex_provider(settings),
                build_antigravity_provider(settings),
                build_handsfreecode_provider(settings),
                build_openhands_provider(settings),
                MockProvider(),
            ]
        )
    if provider_name == "mock":
        return MockProvider()
    if provider_name == "handsfreecode":
        return build_handsfreecode_provider(settings)
    if provider_name == "codex":
        return build_codex_provider(settings)
    if provider_name == "antigravity":
        return build_antigravity_provider(settings)
    return build_openhands_provider(settings)


def doctor() -> None:
    settings = load_settings()
    openhands = build_openhands_provider(settings).diagnostics()
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
                        "ready": openhands.get("ready") is True and openhands.get("sessionKeyPresent") is True,
                        "externalRequired": not (
                            openhands.get("ready") is True and openhands.get("sessionKeyPresent") is True
                        ),
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
) -> None:
    settings = load_settings()
    loaded = load_project(project_path, allowlist=workspace_allowlist(settings))
    load_workflow(workflow_name)
    provider = build_provider(provider_name, settings)
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
    validation_log_hash: str | None,
    test_evidence_hash: str | None,
) -> None:
    settings = load_settings()
    provider = build_provider(provider_name, settings)
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
    if not result.workflow_result.provider_result.success:
        raise SystemExit(2)


def evaluate_github_gate(
    project_path: str,
    action: str,
    dry_run: bool,
    validation_log_hash: str | None,
    receipt_path: str | None,
    test_evidence_hash: str | None,
) -> None:
    settings = load_settings()
    loaded = load_project(project_path, allowlist=workspace_allowlist(settings))
    if dry_run:
        decision = evaluate_github_action(
            loaded,
            action,
            dry_run=True,
            validation_log_hash=validation_log_hash,
            receipt_hash=None,
            secret_scan_hash=None,
            test_evidence_hash=test_evidence_hash,
        )
        print(json.dumps(redact_secrets(decision.as_dict()), indent=2, default=str))
        if not decision.allowed:
            raise SystemExit(2)
        return

    result = execute_github_action(
        loaded,
        GitHubExecutionOptions(
            action=action,
            dry_run=False,
            validation_log_hash=validation_log_hash,
            receipt_path=Path(receipt_path).resolve() if receipt_path else None,
            test_evidence_hash=test_evidence_hash,
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
) -> None:
    settings = load_settings()
    provider = build_provider(provider_name, settings)
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
            report_dir=Path(report_dir).resolve() if report_dir else REPO_ROOT / "reports" / "supervisor",
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
    run_parser.add_argument("--mode", choices=["create-only", "run-agent"], default="create-only")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--receipt-dir")

    isolated_parser = subparsers.add_parser("isolated-run", help="Run a write workflow in a disposable worktree")
    isolated_parser.add_argument("project", nargs="?", default=".")
    isolated_parser.add_argument("--workflow", required=True)
    isolated_parser.add_argument(
        "--provider",
        choices=["auto", "mock", "openhands", "handsfreecode", "codex", "antigravity"],
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
    isolated_parser.add_argument("--validation-log-hash")
    isolated_parser.add_argument("--test-evidence-hash")

    github_parser = subparsers.add_parser("github-gate", help="Evaluate guarded GitHub automation policy")
    github_parser.add_argument("project", nargs="?", default=".")
    github_parser.add_argument("--action", choices=["push", "pr", "pull-request", "merge"], required=True)
    github_parser.add_argument("--execute", action="store_false", dest="dry_run")
    github_parser.set_defaults(dry_run=True)
    github_parser.add_argument("--validation-log-hash")
    github_parser.add_argument("--receipt-path")
    github_parser.add_argument("--test-evidence-hash")

    supervisor_parser = subparsers.add_parser("supervise", help="Run autonomous safe supervisor loop")
    supervisor_parser.add_argument("project", nargs="?", default=".")
    supervisor_parser.add_argument("--workflow", default="project-analysis")
    supervisor_parser.add_argument(
        "--provider",
        choices=["auto", "mock", "openhands", "handsfreecode", "codex", "antigravity"],
        default="mock",
    )
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
        elif args.command == "git-policy":
            evaluate_git_policy(args.project, args.action, args.file)
        elif args.command == "run":
            run_workflow(args.project, args.workflow, args.provider, args.dry_run, args.receipt_dir, args.mode)
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
                args.validation_log_hash,
                args.test_evidence_hash,
            )
        elif args.command == "github-gate":
            evaluate_github_gate(
                args.project,
                args.action,
                args.dry_run,
                args.validation_log_hash,
                args.receipt_path,
                args.test_evidence_hash,
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
            )
    except (ProjectConfigError, WorkflowError) as exc:
        print(f"ai-team error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
