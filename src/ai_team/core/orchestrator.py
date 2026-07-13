from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import yaml

from ai_team.core.project_loader import LoadedProject
from ai_team.providers.base import BaseProvider, ProviderRequest, ProviderResult, RetryingProvider


class WorkflowError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkflowDefinition:
    name: str
    description: str
    stages: list[str]
    write_required: bool
    forbidden_actions: list[str]


@dataclass(frozen=True)
class WorkflowRunResult:
    workflow: WorkflowDefinition
    provider_result: ProviderResult
    dry_run: bool
    stages: list[str]
    started_at: datetime
    completed_at: datetime
    duration_ms: int


FORBIDDEN_ACTIONS = {"production_deploy", "real_payment", "destructive_migration"}


def load_workflow(name: str, workflow_dir: Path | None = None) -> WorkflowDefinition:
    root = workflow_dir or Path(__file__).resolve().parents[3] / "workflows"
    path = root / f"{name}.yaml"
    if not path.exists():
        raise WorkflowError(f"unknown workflow: {name}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise WorkflowError(f"workflow file must be a mapping: {path}")

    stages = data.get("stages")
    if not isinstance(stages, list) or not all(isinstance(item, str) for item in stages):
        raise WorkflowError(f"workflow stages must be a string list: {path}")

    forbidden = data.get("forbidden_actions", [])
    if not isinstance(forbidden, list) or not all(isinstance(item, str) for item in forbidden):
        raise WorkflowError(f"workflow forbidden_actions must be a string list: {path}")

    missing_forbidden = FORBIDDEN_ACTIONS.difference(forbidden)
    if missing_forbidden:
        raise WorkflowError(f"workflow must explicitly forbid: {', '.join(sorted(missing_forbidden))}")

    return WorkflowDefinition(
        name=str(data.get("name") or name),
        description=str(data.get("description") or ""),
        stages=stages,
        write_required=bool(data.get("write_required", False)),
        forbidden_actions=forbidden,
    )


class Orchestrator:
    def __init__(self, provider: BaseProvider, max_retries: int = 2) -> None:
        self.provider = RetryingProvider(provider, max_retries=max_retries)

    def run(
        self,
        loaded_project: LoadedProject,
        workflow_name: str,
        dry_run: bool = False,
        timeout_seconds: float | None = None,
        run_mode: str = "create-only",
        task_instruction: str | None = None,
    ) -> WorkflowRunResult:
        workflow = load_workflow(workflow_name)

        if run_mode not in {"create-only", "run-agent"}:
            raise WorkflowError(f"unsupported run mode: {run_mode}")

        if run_mode == "run-agent":
            loaded_project.assert_agent_run_allowed(workflow.name)

        if workflow.write_required and not dry_run:
            loaded_project.assert_write_allowed(workflow.name)
            if not loaded_project.profile.safety.allow_destructive_commands:
                pass

        prompt = build_workflow_prompt(
            loaded_project,
            workflow,
            dry_run=dry_run,
            task_instruction=task_instruction,
        )
        request = ProviderRequest(
            workflow=workflow.name,
            prompt=prompt,
            project_root=loaded_project.root,
            metadata={
                "project": loaded_project.profile.project.name,
                "branch": loaded_project.current_branch,
                "stages": workflow.stages,
                "dryRun": dry_run,
                "runMode": run_mode,
                "writeRequired": workflow.write_required and not dry_run,
            },
            timeout_seconds=timeout_seconds,
            dry_run=dry_run,
            run_mode=run_mode,
        )
        started_at = datetime.now(UTC)
        result = self.provider.run(request)
        completed_at = datetime.now(UTC)

        return WorkflowRunResult(
            workflow=workflow,
            provider_result=result,
            dry_run=dry_run,
            stages=workflow.stages,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=int((completed_at - started_at).total_seconds() * 1000),
        )


def build_workflow_prompt(
    loaded_project: LoadedProject,
    workflow: WorkflowDefinition,
    dry_run: bool,
    task_instruction: str | None = None,
) -> str:
    safety = loaded_project.profile.safety
    lines = [
        f"Project: {loaded_project.profile.project.name}",
        f"Root: {loaded_project.root}",
        f"Branch: {loaded_project.current_branch or 'unknown'}",
        f"Workflow: {workflow.name}",
        f"Stages: {', '.join(workflow.stages)}",
        f"Dry run: {dry_run}",
        "Forbidden actions: production deploy, real payment, destructive migration.",
        f"Safety allow_git_push: {safety.allow_git_push}",
        f"Safety allow_deploy: {safety.allow_deploy}",
        "Return a concise execution report with findings, planned changes, tests, and blockers.",
    ]
    if task_instruction:
        lines.extend(["Trusted task instruction:", task_instruction])
    return "\n".join(lines)
