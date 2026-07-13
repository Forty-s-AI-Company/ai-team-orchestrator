from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
import yaml

from ai_team.core.evidence import (
    EvidenceError,
    EvidencePolicy,
    ProjectEvidenceSnapshot,
    collect_project_evidence,
    parse_evidence_policy,
    validate_analysis_grounding,
)
from ai_team.core.project_loader import LoadedProject
from ai_team.providers.base import (
    BaseProvider,
    ProviderErrorType,
    ProviderRequest,
    ProviderResult,
    RetryingProvider,
)


class WorkflowError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkflowDefinition:
    name: str
    description: str
    stages: list[str]
    write_required: bool
    forbidden_actions: list[str]
    evidence: EvidencePolicy | None


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
SUPPORTED_RUN_MODES = {"create-only", "read-only-agent", "run-agent"}
READ_ONLY_AGENT_MODE = "read-only-agent"


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

    try:
        evidence = parse_evidence_policy(data.get("evidence"))
    except EvidenceError as exc:
        raise WorkflowError(str(exc)) from exc

    return WorkflowDefinition(
        name=str(data.get("name") or name),
        description=str(data.get("description") or ""),
        stages=stages,
        write_required=bool(data.get("write_required", False)),
        forbidden_actions=forbidden,
        evidence=evidence,
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

        if run_mode not in SUPPORTED_RUN_MODES:
            raise WorkflowError(f"unsupported run mode: {run_mode}")

        if run_mode == READ_ONLY_AGENT_MODE:
            if self.provider.name != "handsfreecode" and not getattr(
                self.provider.provider,
                "supports_read_only_agent",
                False,
            ):
                raise WorkflowError("read-only-agent is only supported by provider=handsfreecode")
            if workflow.write_required:
                raise WorkflowError("read-only-agent requires a workflow with write_required=false")
        elif run_mode == "run-agent":
            loaded_project.assert_agent_run_allowed(workflow.name)

        if workflow.write_required and not dry_run:
            loaded_project.assert_write_allowed(workflow.name)
            if not loaded_project.profile.safety.allow_destructive_commands:
                pass

        evidence_snapshot: ProjectEvidenceSnapshot | None = None
        if run_mode == READ_ONLY_AGENT_MODE:
            if workflow.evidence is None:
                raise WorkflowError("read-only-agent workflow must define a bounded evidence policy")
            try:
                evidence_snapshot = collect_project_evidence(loaded_project, workflow.evidence)
            except EvidenceError as exc:
                raise WorkflowError(f"read-only evidence collection failed: {exc}") from exc
            if not evidence_snapshot.manifest.get("fileCount"):
                raise WorkflowError("read-only evidence collection found no eligible files")

        prompt = build_workflow_prompt(
            loaded_project,
            workflow,
            dry_run=dry_run,
            task_instruction=task_instruction,
            evidence_snapshot=evidence_snapshot,
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
                "writeAccess": False if run_mode == READ_ONLY_AGENT_MODE else workflow.write_required and not dry_run,
                "evidenceManifest": evidence_snapshot.manifest if evidence_snapshot else None,
            },
            timeout_seconds=timeout_seconds,
            dry_run=dry_run,
            run_mode=run_mode,
        )
        started_at = datetime.now(UTC)
        result = self.provider.run(request)
        if evidence_snapshot is not None:
            provider_execution_validation = {
                "status": "passed" if result.success else "failed",
                "errorType": result.error_type,
            }
            evidence_validation = {
                "status": "passed" if evidence_snapshot.manifest.get("fileCount") else "failed",
                "fileCount": evidence_snapshot.manifest.get("fileCount", 0),
                "redactionCount": evidence_snapshot.manifest.get("redactionCount", 0),
            }
            grounding_validation = validate_analysis_grounding(
                result.content,
                evidence_snapshot,
                provider_success=result.success,
            )
            overall_success = (
                result.success
                and evidence_validation["status"] == "passed"
                and grounding_validation["status"] == "passed"
            )
            result = replace(
                result,
                success=overall_success,
                error_type=(
                    result.error_type
                    if result.error_type is not None or overall_success
                    else ProviderErrorType.INVALID_RESPONSE
                ),
                data={
                    **result.data,
                    "evidenceManifest": evidence_snapshot.manifest,
                    "providerExecutionValidation": provider_execution_validation,
                    "evidenceCollectionValidation": evidence_validation,
                    "analysisGroundingValidation": grounding_validation,
                },
            )
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
    evidence_snapshot: ProjectEvidenceSnapshot | None = None,
) -> str:
    safety = loaded_project.profile.safety
    report_instruction = (
        "Return only an evidence-backed technology summary, project facts with evidence paths, "
        "configured validation commands explicitly marked not run, unknowns, and policy blockers."
        if evidence_snapshot
        else "Return a concise execution report with findings, planned changes, tests, and blockers."
    )
    lines = [
        f"Project: {loaded_project.profile.project.name}",
        f"Root: {loaded_project.root}",
        f"Branch: {loaded_project.current_branch or 'unknown'}",
        f"Workflow: {workflow.name}",
        f"Stages: {', '.join(workflow.stages)}",
        f"Dry run: {dry_run}",
        "Read-only forbidden actions: migration, seed, deployment, data deletion, real payment, and secret operations.",
        "Do not provide commands, next steps, or recommendations for forbidden actions. "
        "They may only be reported as disallowed in a Policy Blockers section.",
        f"Safety allow_git_push: {safety.allow_git_push}",
        f"Safety allow_deploy: {safety.allow_deploy}",
        report_instruction,
    ]
    if task_instruction:
        lines.extend(["Trusted task instruction:", task_instruction])
    if evidence_snapshot:
        lines.extend(["", evidence_snapshot.prompt_section])
    return "\n".join(lines)
