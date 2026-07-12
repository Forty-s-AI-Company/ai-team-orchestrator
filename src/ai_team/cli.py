from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from ai_team.core.orchestrator import Orchestrator, WorkflowError, load_workflow
from ai_team.core.project_loader import ProjectConfigError, load_project
from ai_team.core.receipts import write_run_receipt
from ai_team.providers import MockProvider, OpenHandsProvider, OpenHandsSettings


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


def build_openhands_provider(settings: dict) -> OpenHandsProvider:
    openhands = settings.get("openhands", {}) if isinstance(settings.get("openhands"), dict) else {}
    provider_settings = OpenHandsSettings(
        base_url=str(openhands.get("base_url") or "http://127.0.0.1:31024"),
        session_key_env=str(openhands.get("session_key_env") or "SESSION_API_KEY"),
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


def doctor() -> None:
    settings = load_settings()
    provider = build_openhands_provider(settings)
    diagnostics = provider.diagnostics()
    print(
        json.dumps(
            {
                "settings": str(DEFAULT_SETTINGS_PATH),
                "openhands": diagnostics,
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
    provider = MockProvider() if provider_name == "mock" else build_openhands_provider(settings)
    timeout_seconds = provider.settings.timeout_seconds if isinstance(provider, OpenHandsProvider) else 30
    result = Orchestrator(provider=provider, max_retries=2).run(
        loaded,
        workflow_name=workflow_name,
        dry_run=dry_run,
        timeout_seconds=timeout_seconds,
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
        "content": result.provider_result.content,
        "data": result.provider_result.data,
        "receiptPath": str(receipt_path),
    }
    print(json.dumps(payload, indent=2, default=str))
    if not result.provider_result.success:
        raise SystemExit(2)


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

    subparsers.add_parser("doctor", help="Check provider settings and OpenHands loopback status")

    run_parser = subparsers.add_parser("run", help="Run a guarded workflow")
    run_parser.add_argument("project", nargs="?", default=".")
    run_parser.add_argument("--workflow", required=True)
    run_parser.add_argument("--provider", choices=["mock", "openhands"], default="mock")
    run_parser.add_argument("--mode", choices=["create-only", "run-agent"], default="create-only")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--receipt-dir")

    return parser


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
        elif args.command == "run":
            run_workflow(args.project, args.workflow, args.provider, args.dry_run, args.receipt_dir, args.mode)
    except (ProjectConfigError, WorkflowError) as exc:
        print(f"ai-team error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
