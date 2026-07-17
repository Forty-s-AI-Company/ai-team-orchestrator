"""Fail-closed, multi-role delivery loop for explicit trusted tasks only."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from ai_team.core.isolated_executor import (
    IsolatedRunResult,
    list_changed_files,
    run_in_disposable_worktree,
)
from ai_team.core.project_loader import load_project
from ai_team.providers.base import BaseProvider, ProviderErrorType, ProviderRequest, ProviderResult, redact_secrets


SCHEMA = "ai-team-bounded-delivery/v1"
FORBIDDEN_PATTERNS = (
    r"\b(?:run|apply|execute)\s+(?:a\s+|the\s+)?(?:database\s+)?migrat(?:e|ion|ions)\b",
    r"\bproduction\s+(?:database\s+)?migrat(?:e|ion|ions)\b",
    r"\b(?:prisma\s+migrate|npm\s+run\s+[\w:-]*migrat[\w:-]*|pnpm\s+[\w:-]*migrat[\w:-]*|yarn\s+[\w:-]*migrat[\w:-]*)\b",
    r"\b(?:database\s+seeds?|seed(?:ing)?\s+(?:the\s+)?(?:data|database)|"
    r"(?:run|apply|execute|load|populate)\s+(?:the\s+)?(?:database\s+)?seeds?|"
    r"(?:npm|pnpm|yarn)\s+run\s+[\w:-]*seed[\w:-]*|prisma\s+db\s+seed)\b",
    r"\b(?:production\s+)?deploy(?:ment|ing)?\b",
    r"\b(?:real|live|production)\s+payments?\b",
    r"\b(?:process|execute|initiate|submit|make)\s+(?:an?\s+|the\s+)?"
    r"(?:(?:real|live|production)\s+)?payments?\b",
    r"\b(?:charge|capture|refund)\s+(?:an?\s+|the\s+)?"
    r"(?:(?:real|live|production)\s+)?(?:payment|card|funds?|transaction)\b",
    r"\b(?:read|write|expose|rotate|copy)\s+(?:a\s+)?(?:secret|credential|token|api[ _-]?key)\b",
    r"\b(?:delete|drop|truncate)\s+(?:data|database|records?|volume)\b",
    r"\bforce\s+push\b",
    r"\b(?:prisma\s+db\s+push|drizzle(?:-kit)?\s+push|git\s+push|gh\s+(?:pr|api))\b",
)
MIGRATION_ARTIFACT_PATTERN = re.compile(r"\bmigrat(?:e|ion|ions)\b")
SCHEMA_CHANGE_PATTERN = re.compile(r"\b(?:database\s+|prisma\s+)?schema\s+(?:change|changes|update|updates)\b")
API_CONTRACT_CHANGE_PATTERN = re.compile(r"\bapi\s+(?:contract\s+)?(?:change|changes|update|updates)\b")
FIXTURE_DATA_PATTERN = re.compile(r"\b(?:deterministic\s+)?(?:fixture|fake|sample|placeholder)\s+data\b")
ROLE_BY_STAGE = {"pm": "product-manager", "architect": "architect", "engineer": "engineer", "qa": "delivery-qa", "review": "reviewer"}
SECONDARY_PROVIDER_BY_STAGE = {"architect": "codex", "qa": "codex", "review": "antigravity"}
RECEIPT_FILE_PATTERN = re.compile(r"^(\d+)-(pm|architect|engineer|qa|review)\.json$")
REVIEW_EVIDENCE_PATH = "/tmp/ai-team-review-evidence/patch.diff"
MAX_REVIEW_PATCH_BYTES = 512_000


class BoundedDeliveryError(ValueError):
    """Raised when a task contract or a stage result is unsafe."""


class PolicyValidationError(BoundedDeliveryError):
    """Raised when otherwise structured stage output violates delivery policy."""


@dataclass(frozen=True)
class TaskChangePolicy:
    """Explicit, code-only product changes allowed by the trusted contract."""

    schema_changes: bool = False
    api_contract_changes: bool = False
    migration_artifacts: bool = False
    fixture_data: bool = False


@dataclass(frozen=True)
class TrustedTaskContract:
    id: str
    title: str
    source_kind: str
    source_reference: str
    instruction: str
    allowed_write_paths: tuple[str, ...]
    validation_commands: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    change_policy: TaskChangePolicy = TaskChangePolicy()


@dataclass(frozen=True)
class DeliveryLimits:
    max_iterations: int = 2
    max_repair_attempts: int = 1
    max_token_usage: int = 120_000
    timeout_seconds: int = 180


@dataclass(frozen=True)
class EngineeringAttempt:
    provider_result: ProviderResult
    worktree_path: Path
    changed_files: list[str]
    validation: dict[str, Any]
    commit_sha: str | None
    run_receipt: Path | None = None
    executor_receipt: Path | None = None


@dataclass(frozen=True)
class BoundedDeliveryOptions:
    project_path: Path
    task_contract_path: Path
    provider_for_role: Callable[[str], BaseProvider]
    workspace_allowlist: list[str] | None
    report_dir: Path
    state_path: Path
    limits: DeliveryLimits = DeliveryLimits()
    engineering_executor: Callable[[TrustedTaskContract, str, BaseProvider, int], EngineeringAttempt] | None = None


def load_trusted_task_contract(path: Path) -> tuple[TrustedTaskContract, str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BoundedDeliveryError(f"trusted task contract must be readable JSON: {path}") from exc
    if not isinstance(raw, dict) or raw.get("schemaVersion") != 1:
        raise BoundedDeliveryError("trusted task contract requires schemaVersion=1")
    source = raw.get("source")
    if not isinstance(source, dict) or source.get("kind") not in {"github-issue", "trusted-contract"}:
        raise BoundedDeliveryError("task source must be github-issue or trusted-contract")
    values = {
        "id": raw.get("id"), "title": raw.get("title"), "instruction": raw.get("instruction"),
        "source_kind": source.get("kind"), "source_reference": source.get("reference"),
    }
    if not all(isinstance(value, str) and value.strip() for value in values.values()):
        raise BoundedDeliveryError("task contract requires non-empty id, title, instruction, and source reference")
    paths = _string_list(raw.get("allowedWritePaths"), "allowedWritePaths")
    commands = _string_list(raw.get("validationCommands"), "validationCommands")
    depends_on = _optional_string_list(raw.get("dependsOn"), "dependsOn")
    change_policy = _load_change_policy(raw.get("changePolicy"))
    if not paths or not commands:
        raise BoundedDeliveryError("trusted write tasks require allowedWritePaths and validationCommands")
    contract = TrustedTaskContract(
        id=values["id"].strip(), title=values["title"].strip(), source_kind=values["source_kind"].strip(),
        source_reference=values["source_reference"].strip(), instruction=values["instruction"].strip(),
        allowed_write_paths=tuple(paths), validation_commands=tuple(commands),
        depends_on=tuple(depends_on), change_policy=change_policy,
    )
    if contract.id in contract.depends_on:
        raise BoundedDeliveryError("trusted task contract cannot depend on itself")
    _reject_forbidden(contract.instruction, "task instruction", contract)
    _validate_allowed_write_paths(contract.allowed_write_paths)
    _validate_change_policy_paths(contract)
    for command in contract.validation_commands:
        _reject_forbidden(command, "validation command", contract)
    normalized = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return contract, hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def run_bounded_delivery(options: BoundedDeliveryOptions) -> dict[str, Any]:
    contract, task_sha = load_trusted_task_contract(options.task_contract_path)
    project = load_project(options.project_path, allowlist=options.workspace_allowlist)
    _validate_contract_validation_commands(project, contract)
    prior = _read_json(options.state_path)
    context: dict[str, Any] = {
        "task": asdict(contract),
        "taskSha": task_sha,
        "receipts": [],
        "tokenUsage": 0,
        "resumedFrom": prior.get("status"),
    }
    try:
        context["receipts"] = _recover_receipt_paths(
            options.report_dir,
            task_sha,
            prior.get("receipts"),
        )
    except BoundedDeliveryError as exc:
        return _stop(options, context, str(exc), "receipt-integrity")
    if prior.get("status") == "completed" and prior.get("taskSha") == task_sha:
        return {
            "status": "already-completed",
            "taskSha": task_sha,
            "receipts": context["receipts"],
            "statePath": str(options.state_path),
        }
    try:
        receipt_payloads = _load_receipt_payloads(context["receipts"])
        context["tokenUsage"] = _recovered_token_usage(receipt_payloads)
        checkpoints = _recover_stage_checkpoints(receipt_payloads, contract)
    except BoundedDeliveryError as exc:
        return _stop(options, context, str(exc), "receipt-integrity")
    if context["tokenUsage"] > options.limits.max_token_usage:
        return _stop(options, context, "token-budget-exhausted", "policy-or-provider")
    _write_state(options, "running", "pm", context)
    try:
        pm_checkpoint = checkpoints.get("pm")
        pm = pm_checkpoint["evidence"] if pm_checkpoint else _run_stage(
            options, project.root, "pm", context, contract
        )
        acceptance = _required_strings(pm, "acceptanceCriteria")
        context["acceptanceCriteria"] = acceptance
        architect_checkpoint = checkpoints.get("architect")
        architect = architect_checkpoint["evidence"] if architect_checkpoint else _run_stage(
            options, project.root, "architect", context, contract
        )
        plan = _required_strings(architect, "plan")
        context["plan"] = plan
        context["planAllowedWritePaths"] = _required_strings(architect, "allowedWritePaths")

        repairs = _recover_repairs(prior, task_sha, tuple(context["planAllowedWritePaths"]))
        if repairs:
            context["repairs"] = repairs

        resumed_engineer = checkpoints.get("engineer")
        reusable_worktree: Path | None = None
        if resumed_engineer is not None:
            evidence = resumed_engineer["evidence"]
            reusable_worktree = _validated_resume_worktree(
                options,
                prior,
                task_sha=task_sha,
                allowed_write_paths=tuple(context["planAllowedWritePaths"]),
                expected_commit=evidence["commitSha"],
                expected_changed_files=evidence["changedFiles"],
                expected_run_receipt=evidence.get("runReceipt"),
                expected_executor_receipt=evidence.get("executorReceipt"),
                allow_dirty=False,
            )
            context.update({
                "worktreePath": str(reusable_worktree),
                "commitSha": evidence["commitSha"],
                "changedFiles": evidence["changedFiles"],
                "validation": evidence["validation"],
                "runReceipt": evidence.get("runReceipt"),
                "executorReceipt": evidence.get("executorReceipt"),
            })
            qa_checkpoint = checkpoints.get("qa")
            qa = qa_checkpoint["evidence"] if qa_checkpoint else _run_stage(
                options, reusable_worktree, "qa", context, contract
            )
            review_checkpoint = checkpoints.get("review") if qa_checkpoint else None
            review = review_checkpoint["evidence"] if review_checkpoint else _run_stage(
                options, reusable_worktree, "review", context, contract
            )
            findings = _findings(qa) + _findings(review)
            if not findings:
                return _complete(options, context)
            if not _findings_are_attributable(findings, tuple(context["planAllowedWritePaths"])):
                return _stop(options, context, "unattributed-or-out-of-scope-finding", "qa-review")
            if len(repairs) >= options.limits.max_repair_attempts:
                return _stop(options, context, "max-repair-attempts-reached", "qa-review")
            repairs.append({"findingSha": _sha(findings), "findings": findings})
            context["repairs"] = repairs
        elif options.engineering_executor is None and prior.get("taskSha") == task_sha and prior.get("worktreePath"):
            reusable_worktree = _validated_resume_worktree(
                options,
                prior,
                task_sha=task_sha,
                allowed_write_paths=tuple(context["planAllowedWritePaths"]),
                expected_commit=None,
                expected_changed_files=prior.get("changedFiles"),
                expected_run_receipt=prior.get("runReceipt"),
                expected_executor_receipt=prior.get("executorReceipt"),
                allow_dirty=True,
            )

        engineering_executor = options.engineering_executor or _default_engineering_executor(
            options, reusable_worktree=reusable_worktree
        )
        for iteration in range(1, options.limits.max_iterations + 1):
            instruction = _engineering_instruction(context, repairs)
            engineer = options.provider_for_role(ROLE_BY_STAGE["engineer"])
            attempt = engineering_executor(
                contract, instruction, engineer, iteration
            )
            failure = _engineering_failure(
                attempt,
                tuple(context["planAllowedWritePaths"]),
            )
            if not _add_tokens(options, context, attempt.provider_result) and failure is None:
                failure = ("token-budget-exhausted", "engineer", "policy-validation")
            _record_engineering_receipt(options, context, attempt, iteration, failure)
            # Retain the disposable worktree and validation evidence even when
            # this attempt fails. A bounded repair can then reuse the exact
            # worktree, while an attention-required state remains actionable.
            context.update({
                "worktreePath": str(attempt.worktree_path),
                "commitSha": attempt.commit_sha,
                "changedFiles": attempt.changed_files,
                "validation": attempt.validation,
                "runReceipt": str(attempt.run_receipt) if attempt.run_receipt else None,
                "executorReceipt": str(attempt.executor_receipt) if attempt.executor_receipt else None,
            })
            if failure is not None:
                reason, stage, _kind = failure
                if _is_repairable_validation_failure(failure, attempt, tuple(context["planAllowedWritePaths"])):
                    if len(repairs) >= options.limits.max_repair_attempts:
                        return _stop(options, context, "max-repair-attempts-reached", "validation")
                    repairs.append(_validation_repair_evidence(attempt))
                    context["repairs"] = repairs
                    continue
                return _stop(options, context, reason, stage)

            qa = _run_stage(options, attempt.worktree_path, "qa", context, contract)
            review = _run_stage(options, attempt.worktree_path, "review", context, contract)
            findings = _findings(qa) + _findings(review)
            if not findings:
                return _complete(options, context)
            if not _findings_are_attributable(findings, tuple(context["planAllowedWritePaths"])):
                return _stop(options, context, "unattributed-or-out-of-scope-finding", "qa-review")
            if len(repairs) >= options.limits.max_repair_attempts:
                return _stop(options, context, "max-repair-attempts-reached", "qa-review")
            repairs.append({"findingSha": _sha(findings), "findings": findings})
            context["repairs"] = repairs
        return _stop(options, context, "max-iterations-reached", "engineer")
    except BoundedDeliveryError as exc:
        return _stop(options, context, str(exc), "policy-or-provider")


def _run_stage(
    options: BoundedDeliveryOptions,
    root: Path,
    stage: str,
    context: dict[str, Any],
    contract: TrustedTaskContract,
) -> dict[str, Any]:
    provider = options.provider_for_role(ROLE_BY_STAGE[stage])
    expected = "antigravity" if stage in {"pm", "architect", "qa"} else "codex"
    review_patch = _review_patch_evidence(root, context) if stage in {"qa", "review"} else None
    request = ProviderRequest(
        workflow=f"bounded-delivery-{stage}", project_root=root, run_mode="run-agent",
        timeout_seconds=options.limits.timeout_seconds,
        prompt=_stage_prompt(stage, context, review_patch),
        metadata={
            "role": ROLE_BY_STAGE[stage], "writeRequired": False, "writeAccess": False,
            "taskSha": context["taskSha"], "boundedStage": stage,
            "requiredProvider": expected,
            "reviewPatch": review_patch["content"] if review_patch else None,
            "reviewPatchSha": review_patch["sha256"] if review_patch else None,
        },
    )
    before = _read_only_git_fingerprint(root)
    result = provider.run(request)
    if _read_only_git_fingerprint(root) != before:
        reason = "read-only-stage-modified-worktree"
        _write_receipt(
            options,
            context,
            stage,
            result,
            _validation_failure("read-only-integrity", reason),
        )
        raise BoundedDeliveryError(reason)
    if not _native_success(result, expected):
        reason = _provider_stop_reason(result)
        _write_receipt(options, context, stage, result, _validation_failure("provider-execution", reason))
        raise BoundedDeliveryError(reason)
    try:
        payload = json.loads(result.content)
    except json.JSONDecodeError as exc:
        reason = f"{stage} returned non-JSON content"
        _write_receipt(options, context, stage, result, _validation_failure("structured-output", reason))
        raise BoundedDeliveryError(reason) from exc
    try:
        _validate_stage_structure(stage, payload)
    except BoundedDeliveryError as exc:
        _write_receipt(
            options,
            context,
            stage,
            result,
            _validation_failure("structured-output", str(exc)),
        )
        raise
    try:
        _reject_forbidden(json.dumps(payload, ensure_ascii=False), f"{stage} output", contract)
        if stage == "architect":
            schema_or_api_change = payload.get("schemaOrApiChange")
            if not isinstance(schema_or_api_change, bool):
                raise PolicyValidationError("architect-schema-or-api-change-must-be-boolean")
            if schema_or_api_change and not _schema_or_api_change_allowed(contract):
                raise PolicyValidationError("architect-requires-product-decision")
            _validate_plan_scope(payload, contract)
    except PolicyValidationError as exc:
        _write_receipt(
            options,
            context,
            stage,
            result,
            _validation_failure("policy-validation", str(exc), primaryResult=payload),
        )
        raise
    try:
        expected_secondary = SECONDARY_PROVIDER_BY_STAGE.get(stage)
        secondary_payload: dict[str, Any] | None = None
        if expected_secondary is not None:
            failure_prefix = f"{stage}-{expected_secondary}-read-only-review"
            secondary = result.data.get("secondaryReview")
            if (
                not isinstance(secondary, dict)
                or secondary.get("success") is not True
                or secondary.get("provider") != expected_secondary
            ):
                raise BoundedDeliveryError(f"{failure_prefix}-failed")
            secondary_payload = _validate_secondary_review(
                secondary,
                stage,
                expected_secondary,
                contract,
            )
        if secondary_payload is not None:
            if secondary_payload["blockers"]:
                raise BoundedDeliveryError(
                    f"{stage}-{expected_secondary}-read-only-review-has-blockers"
                )
            if stage == "architect" and secondary_payload["findings"]:
                raise BoundedDeliveryError(
                    "architect-codex-read-only-review-has-findings"
                )
            if stage in {"qa", "review"}:
                payload = {
                    **payload,
                    "findings": [*payload["findings"], *secondary_payload["findings"]],
                    "tests": [*payload.get("tests", []), *secondary_payload["tests"]],
                }
    except PolicyValidationError as exc:
        _write_receipt(
            options,
            context,
            stage,
            result,
            _validation_failure("policy-validation", str(exc), primaryResult=payload),
        )
        raise
    except BoundedDeliveryError as exc:
        _write_receipt(
            options,
            context,
            stage,
            result,
            _validation_failure("secondary-provider-output", str(exc), primaryResult=payload),
        )
        raise
    if not _add_tokens(options, context, result):
        reason = "token-budget-exhausted"
        _write_receipt(
            options,
            context,
            stage,
            result,
            _validation_failure("policy-validation", reason, primaryResult=payload),
        )
        raise BoundedDeliveryError(reason)
    _write_receipt(options, context, stage, result, payload)
    return payload


def _default_engineering_executor(
    options: BoundedDeliveryOptions,
    *,
    reusable_worktree: Path | None = None,
) -> Callable[[TrustedTaskContract, str, BaseProvider, int], EngineeringAttempt]:

    def execute(contract: TrustedTaskContract, instruction: str, provider: BaseProvider, iteration: int) -> EngineeringAttempt:
        nonlocal reusable_worktree
        result: IsolatedRunResult = run_in_disposable_worktree(
            source_project_path=options.project_path, provider=provider, workflow_name="bug-fix-loop",
            workspace_allowlist=options.workspace_allowlist, receipt_dir=options.report_dir / "isolated",
            worktree_parent=options.project_path.resolve().parent, dry_run=False, run_mode="run-agent",
            keep_worktree=True, auto_commit=True, commit_message=f"fix: {contract.title}",
            task_instruction=instruction, allowed_write_paths=list(contract.allowed_write_paths),
            validation_commands=[*contract.validation_commands, "git diff --check"], require_validation=True,
            reuse_worktree_path=reusable_worktree,
        )
        reusable_worktree = result.worktree_path
        validation = result.commit_result.get("validation") or {"success": False}
        commit_validation = result.commit_result.get("validationResult")
        if isinstance(commit_validation, dict) and commit_validation.get("success") is False:
            validation = {**validation, **commit_validation}
        return EngineeringAttempt(
            provider_result=result.workflow_result.provider_result, worktree_path=result.worktree_path,
            changed_files=list(result.git_policy.get("changedFiles") or []), validation=validation,
            commit_sha=result.commit_result.get("commitSha"), run_receipt=result.run_receipt,
            executor_receipt=result.executor_receipt,
        )
    return execute


def _stage_prompt(
    stage: str,
    context: dict[str, Any],
    review_patch: dict[str, Any] | None = None,
) -> str:
    task = context["task"]
    change_policy = task.get("change_policy", {})
    schema_or_api_allowed = bool(
        change_policy.get("schema_changes") or change_policy.get("api_contract_changes")
    )
    evidence = {
        "acceptanceCriteria": context.get("acceptanceCriteria", []),
        "plan": context.get("plan", []),
        "allowedWritePaths": task["allowed_write_paths"],
        "validationCommands": task["validation_commands"],
        "changedFiles": context.get("changedFiles", []),
        "commitSha": context.get("commitSha"),
        "validation": context.get("validation", {"success": False, "reason": "validation evidence unavailable"}),
        "repairs": [item["findingSha"] for item in context.get("repairs", [])],
        "reviewEvidence": (
            {
                "path": REVIEW_EVIDENCE_PATH,
                "sha256": review_patch["sha256"],
                "bytes": review_patch["bytes"],
            }
            if review_patch
            else None
        ),
    }
    safe_evidence = redact_secrets(evidence)
    tests_shape = "tests=['evidence citation']" if stage in {"qa", "review"} else "tests=[]"
    return "\n".join((
        f"Bounded delivery stage: {stage}", f"Task SHA: {context['taskSha']}",
        f"Task: {redact_secrets(task['title'])}", f"Instruction: {redact_secrets(task['instruction'])}",
        f"Acceptance criteria: {json.dumps(safe_evidence['acceptanceCriteria'])}",
        f"Allowed write paths: {json.dumps(safe_evidence['allowedWritePaths'])}",
        f"Validation commands: {json.dumps(safe_evidence['validationCommands'])}",
        f"Change policy: {json.dumps(redact_secrets(change_policy))}",
        f"Implementation evidence: {json.dumps(safe_evidence)}",
        f"Return only JSON with schema='ai-team-bounded-delivery/v1', stage, status='passed', findings=[], {tests_shape}, blockers=[].",
        "Do not edit files, run shell commands, deploy, execute migrations or seeds, process payments, read secrets, or perform destructive actions.",
        "Schema/API code changes, migration artifacts, and fixture data are allowed only when their exact change-policy flag is true.",
        (
            "PM: include non-empty acceptanceCriteria. Architect: include non-empty plan, allowedWritePaths, "
            "validationCommands, and boolean schemaOrApiChange; it may be true only for the authorized change policy."
            if schema_or_api_allowed
            else "PM: include non-empty acceptanceCriteria. Architect: include non-empty plan, allowedWritePaths, validationCommands, schemaOrApiChange=false."
        ),
        "PM/architect: findings and blockers must be exactly []; do not restate required work as a finding or blocker.",
        "QA/review: evaluate every acceptance criterion; tests must cite non-empty validation or regression evidence; findings must be [] only when passed.",
        "QA/review: read the exact redacted patch from reviewEvidence.path with the read_file tool; do not use shell or command tools.",
        "QA/review: each failed finding must include path and message; blockers must be exactly [] for status='passed'.",
    ))


def _review_patch_evidence(root: Path, context: dict[str, Any]) -> dict[str, Any]:
    """Build exact, redacted diff evidence without exposing the primary worktree."""
    commit_sha = context.get("commitSha")
    changed_files = context.get("changedFiles")
    if (
        not isinstance(commit_sha, str)
        or re.fullmatch(r"[0-9a-f]{40}", commit_sha) is None
        or not isinstance(changed_files, list)
        or not changed_files
        or not all(isinstance(path, str) and path for path in changed_files)
    ):
        raise BoundedDeliveryError("review-patch-evidence-invalid")
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "show",
                "--format=",
                "--no-ext-diff",
                "--no-textconv",
                "--unified=3",
                commit_sha,
                "--",
                *changed_files,
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BoundedDeliveryError("review-patch-evidence-unavailable") from exc
    if completed.returncode != 0:
        raise BoundedDeliveryError("review-patch-evidence-unavailable")
    content = str(redact_secrets(completed.stdout)).strip()
    encoded = content.encode("utf-8")
    if not content or len(encoded) > MAX_REVIEW_PATCH_BYTES:
        raise BoundedDeliveryError("review-patch-evidence-out-of-bounds")
    return {
        "content": content,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
    }


def _engineering_failure(
    attempt: EngineeringAttempt,
    allowed_write_paths: tuple[str, ...],
) -> tuple[str, str, str] | None:
    if not _native_success(attempt.provider_result, "codex"):
        return _provider_stop_reason(attempt.provider_result), "engineer", "provider-execution"
    # Scope is a hard policy boundary. Check it before considering a failed
    # validation repair so out-of-scope writes can never enter the repair loop.
    if not _paths_within(attempt.changed_files, allowed_write_paths):
        return "engineering-diff-outside-allowed-paths", "engineer", "policy-validation"
    if not attempt.validation.get("success"):
        kind = attempt.validation.get("kind")
        reason = attempt.validation.get("stopReason")
        if isinstance(kind, str) and isinstance(reason, str):
            stage = "engineer" if kind in {"git-commit", "policy-validation"} else "validation"
            return reason, stage, kind
        return "deterministic-validation-failed", "validation", "deterministic-validation"
    if not attempt.commit_sha:
        return "engineering-commit-missing", "engineer", "commit-validation"
    return None


def _is_repairable_validation_failure(
    failure: tuple[str, str, str],
    attempt: EngineeringAttempt,
    allowed_write_paths: tuple[str, ...],
) -> bool:
    _reason, stage, kind = failure
    return (
        stage == "validation"
        and kind == "deterministic-validation"
        and bool(attempt.changed_files)
        and _paths_within(attempt.changed_files, allowed_write_paths)
    )


def _validation_repair_evidence(attempt: EngineeringAttempt) -> dict[str, Any]:
    failed_commands: list[dict[str, Any]] = []
    commands = attempt.validation.get("commands")
    if isinstance(commands, list):
        for item in commands:
            if not isinstance(item, dict) or item.get("returnCode") == 0:
                continue
            failed_commands.append({
                "command": str(redact_secrets(item.get("command", "")))[:500],
                "returnCode": item.get("returnCode"),
                # Keep the useful tail (compiler and test summaries usually
                # appear last), but never pass unbounded output back to a model.
                "stdout": str(redact_secrets(item.get("stdout", "")))[-2000:],
                "stderr": str(redact_secrets(item.get("stderr", "")))[-2000:],
            })
            if len(failed_commands) >= 4:
                break
    evidence = {
        "kind": "deterministic-validation",
        "changedFiles": attempt.changed_files,
        "failedCommands": failed_commands,
    }
    return {
        "kind": "deterministic-validation",
        "findingSha": _sha(evidence),
        "evidence": evidence,
    }


def _record_engineering_receipt(
    options: BoundedDeliveryOptions,
    context: dict[str, Any],
    attempt: EngineeringAttempt,
    iteration: int,
    failure: tuple[str, str, str] | None,
) -> None:
    evidence: dict[str, Any] = {
        "iteration": iteration, "changedFiles": attempt.changed_files, "validation": attempt.validation,
        "commitSha": attempt.commit_sha, "runReceipt": str(attempt.run_receipt) if attempt.run_receipt else None,
        "executorReceipt": str(attempt.executor_receipt) if attempt.executor_receipt else None,
    }
    if failure is not None:
        reason, _stage, kind = failure
        evidence.update({
            "validationError": reason,
            "stopReason": reason,
            "validation": {
                **attempt.validation,
                "success": False,
                "kind": kind,
                "stopReason": reason,
            },
        })
    _write_receipt(options, context, "engineer", attempt.provider_result, evidence)


def _write_receipt(options: BoundedDeliveryOptions, context: dict[str, Any], stage: str, result: ProviderResult, evidence: dict[str, Any]) -> None:
    options.report_dir.mkdir(parents=True, exist_ok=True)
    path = options.report_dir / f"{len(context['receipts']) + 1:02d}-{stage}.json"
    payload = redact_secrets({
        "schemaVersion": 1, "stage": stage, "generatedAt": datetime.now(UTC).isoformat(),
        "outerRunMode": "bounded-delivery", "runtimeMode": result.data.get("runtimeMode", "run-agent"),
        "writeAccess": stage == "engineer", "inputTaskSha": context["taskSha"], "taskSha": context["taskSha"],
        "provider": result.provider, "selectedModel": result.data.get("selectedModel"),
        "reasoningEffort": result.data.get("reasoningEffort"), "tokenUsage": result.data.get("tokenUsage", 0),
        "providerSuccess": result.success, "errorType": result.error_type,
        "validationResult": _receipt_validation(result, evidence),
        "stopReason": evidence.get("stopReason") or evidence.get("validationError"),
        "findingSha": _sha(evidence.get("findings", [])) if isinstance(evidence.get("findings"), list) else None,
        # Bind every post-engineering receipt to the exact commit it reviewed.
        # Engineering supplies the SHA in its own evidence; QA/review inherit
        # the already verified SHA from the bounded-delivery context.
        "commitSha": evidence.get("commitSha") or context.get("commitSha"),
        "evidence": evidence,
        "secondaryReview": result.data.get("secondaryReview"),
    })
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
    except FileExistsError as exc:
        raise BoundedDeliveryError("receipt-sequence-collision") from exc
    context["receipts"].append(str(path))


def _recover_receipt_paths(report_dir: Path, task_sha: str, state_receipts: Any) -> list[str]:
    if state_receipts is not None and (
        not isinstance(state_receipts, list)
        or not all(isinstance(item, str) and item for item in state_receipts)
    ):
        raise BoundedDeliveryError("receipt-state-invalid")
    root = report_dir.resolve()
    for item in state_receipts or []:
        path = Path(item).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise BoundedDeliveryError("receipt-path-outside-report-dir") from exc
        if not path.is_file():
            raise BoundedDeliveryError("receipt-referenced-by-state-is-missing")
    if not report_dir.exists():
        return []
    indexed: list[tuple[int, Path, str]] = []
    for path in report_dir.glob("*.json"):
        match = RECEIPT_FILE_PATTERN.fullmatch(path.name)
        if match is None:
            raise BoundedDeliveryError("receipt-file-name-invalid")
        indexed.append((int(match.group(1)), path.resolve(), match.group(2)))
    indexed.sort(key=lambda item: item[0])
    if [item[0] for item in indexed] != list(range(1, len(indexed) + 1)):
        raise BoundedDeliveryError("receipt-sequence-gap")
    recovered: list[str] = []
    for _index, path, stage in indexed:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BoundedDeliveryError("receipt-file-invalid") from exc
        if not isinstance(payload, dict) or payload.get("schemaVersion") != 1:
            raise BoundedDeliveryError("receipt-file-invalid")
        if payload.get("stage") != stage or payload.get("outerRunMode") != "bounded-delivery":
            raise BoundedDeliveryError("receipt-file-invalid")
        if payload.get("taskSha") != task_sha or payload.get("inputTaskSha") != task_sha:
            raise BoundedDeliveryError("receipt-task-sha-mismatch")
        recovered.append(str(path))
    return recovered


def _load_receipt_payloads(paths: list[str]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for item in paths:
        try:
            payload = json.loads(Path(item).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BoundedDeliveryError("receipt-file-invalid") from exc
        if not isinstance(payload, dict):
            raise BoundedDeliveryError("receipt-file-invalid")
        payloads.append(payload)
    return payloads


def _recovered_token_usage(receipts: list[dict[str, Any]]) -> int:
    total = 0
    for receipt in receipts:
        value = receipt.get("tokenUsage", 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise BoundedDeliveryError("receipt-token-usage-invalid")
        total += value
    return total


def _recover_stage_checkpoints(
    receipts: list[dict[str, Any]],
    contract: TrustedTaskContract,
) -> dict[str, dict[str, Any]]:
    """Recover only fully attested, dependency-ordered stage successes."""
    checkpoints: dict[str, dict[str, Any]] = {}
    for receipt in receipts:
        validation = receipt.get("validationResult")
        if not isinstance(validation, dict) or not isinstance(validation.get("success"), bool):
            raise BoundedDeliveryError("receipt-validation-result-invalid")
        provider_success = receipt.get("providerSuccess")
        if not isinstance(provider_success, bool):
            raise BoundedDeliveryError("receipt-provider-result-invalid")
        validation_success = validation["success"]
        stop_reason = receipt.get("stopReason")
        if validation_success and (not provider_success or stop_reason not in {None, ""}):
            raise BoundedDeliveryError("receipt-success-attestation-invalid")
        if not validation_success and (
            not isinstance(stop_reason, str)
            or not stop_reason
            or validation.get("stopReason") != stop_reason
        ):
            raise BoundedDeliveryError("receipt-failure-attestation-invalid")
        if not (provider_success and validation_success and stop_reason in {None, ""}):
            continue
        try:
            _accept_stage_checkpoint(checkpoints, receipt, contract)
        except (BoundedDeliveryError, KeyError, TypeError, ValueError) as exc:
            raise BoundedDeliveryError("receipt-checkpoint-invalid") from exc
    return checkpoints


def _accept_stage_checkpoint(
    checkpoints: dict[str, dict[str, Any]],
    receipt: dict[str, Any],
    contract: TrustedTaskContract,
) -> None:
    stage = receipt["stage"]
    expected_provider = "antigravity" if stage in {"pm", "architect", "qa"} else "codex"
    if receipt.get("provider") != expected_provider:
        raise BoundedDeliveryError("checkpoint provider mismatch")
    evidence = receipt.get("evidence")
    if not isinstance(evidence, dict):
        raise BoundedDeliveryError("checkpoint evidence missing")

    if stage == "engineer":
        if "architect" not in checkpoints:
            raise BoundedDeliveryError("engineer checkpoint has no architect dependency")
        validation = evidence.get("validation")
        commit_sha = evidence.get("commitSha")
        changed_files = evidence.get("changedFiles")
        if (
            not isinstance(validation, dict)
            or validation.get("success") is not True
            or not isinstance(commit_sha, str)
            or not commit_sha
            or receipt.get("commitSha") != commit_sha
            or not isinstance(changed_files, list)
            or not changed_files
            or not all(isinstance(item, str) and item for item in changed_files)
            or not _paths_within(
                changed_files,
                tuple(checkpoints["architect"]["evidence"]["allowedWritePaths"]),
            )
        ):
            raise BoundedDeliveryError("engineer checkpoint attestation invalid")
        checkpoints["engineer"] = receipt
        checkpoints.pop("qa", None)
        checkpoints.pop("review", None)
        return

    _validate_stage_structure(stage, evidence)
    _reject_forbidden(json.dumps(evidence, ensure_ascii=False), f"{stage} checkpoint", contract)
    if stage == "pm":
        checkpoints["pm"] = receipt
        for dependent in ("architect", "engineer", "qa", "review"):
            checkpoints.pop(dependent, None)
        return
    if stage == "architect":
        schema_or_api_change = evidence.get("schemaOrApiChange")
        if (
            "pm" not in checkpoints
            or not isinstance(schema_or_api_change, bool)
            or (schema_or_api_change and not _schema_or_api_change_allowed(contract))
        ):
            raise BoundedDeliveryError("architect checkpoint dependency invalid")
        _validate_plan_scope(evidence, contract)
        _validate_checkpoint_secondary(receipt, stage, "codex", contract)
        checkpoints["architect"] = receipt
        for dependent in ("engineer", "qa", "review"):
            checkpoints.pop(dependent, None)
        return
    if stage == "qa":
        engineer = checkpoints.get("engineer")
        if engineer is None or receipt.get("commitSha") != engineer.get("commitSha"):
            raise BoundedDeliveryError("qa checkpoint commit dependency invalid")
        _validate_checkpoint_secondary(receipt, stage, "codex", contract, allow_findings=True)
        checkpoints.pop("qa", None)
        checkpoints.pop("review", None)
        if not _findings(evidence):
            checkpoints["qa"] = receipt
        return
    if stage == "review":
        engineer = checkpoints.get("engineer")
        if (
            engineer is None
            or "qa" not in checkpoints
            or receipt.get("commitSha") != engineer.get("commitSha")
        ):
            return
        _validate_checkpoint_secondary(receipt, stage, "antigravity", contract, allow_findings=True)
        checkpoints.pop("review", None)
        if not _findings(evidence):
            checkpoints["review"] = receipt


def _validate_checkpoint_secondary(
    receipt: dict[str, Any],
    stage: str,
    provider: str,
    contract: TrustedTaskContract,
    *,
    allow_findings: bool = False,
) -> None:
    secondary = receipt.get("secondaryReview")
    if (
        not isinstance(secondary, dict)
        or secondary.get("success") is not True
        or secondary.get("provider") != provider
    ):
        raise BoundedDeliveryError("checkpoint secondary review invalid")
    payload = _validate_secondary_review(secondary, stage, provider, contract)
    if payload["blockers"] or (payload["findings"] and not allow_findings):
        raise BoundedDeliveryError("checkpoint secondary review did not approve")


def _recover_repairs(
    prior: dict[str, Any],
    task_sha: str,
    allowed_write_paths: tuple[str, ...],
) -> list[dict[str, Any]]:
    if prior.get("taskSha") != task_sha or prior.get("repairs") is None:
        return []
    repairs = prior.get("repairs")
    if not isinstance(repairs, list):
        raise BoundedDeliveryError("resume-repair-state-invalid")
    recovered: list[dict[str, Any]] = []
    for item in repairs:
        if not isinstance(item, dict) or not isinstance(item.get("findingSha"), str):
            raise BoundedDeliveryError("resume-repair-state-invalid")
        if isinstance(item.get("findings"), list):
            findings = _findings(item)
            if item["findingSha"] != _sha(findings) or not _findings_are_attributable(
                findings, allowed_write_paths
            ):
                raise BoundedDeliveryError("resume-repair-state-invalid")
        elif isinstance(item.get("evidence"), dict):
            evidence = item["evidence"]
            if evidence.get("kind") != "deterministic-validation" or item["findingSha"] != _sha(evidence):
                raise BoundedDeliveryError("resume-repair-state-invalid")
            changed_files = evidence.get("changedFiles")
            if not isinstance(changed_files, list) or not _paths_within(changed_files, allowed_write_paths):
                raise BoundedDeliveryError("resume-repair-state-invalid")
        else:
            raise BoundedDeliveryError("resume-repair-state-invalid")
        recovered.append(item)
    return recovered


def _validated_resume_worktree(
    options: BoundedDeliveryOptions,
    prior: dict[str, Any],
    *,
    task_sha: str,
    allowed_write_paths: tuple[str, ...],
    expected_commit: str | None,
    expected_changed_files: Any,
    expected_run_receipt: Any,
    expected_executor_receipt: Any,
    allow_dirty: bool,
) -> Path:
    try:
        if prior.get("taskSha") != task_sha or not isinstance(prior.get("worktreePath"), str):
            raise ValueError("state is not bound to this task and worktree")
        path = Path(prior["worktreePath"]).resolve()
        source = load_project(options.project_path, allowlist=options.workspace_allowlist)
        loaded = load_project(path, allowlist=options.workspace_allowlist)
        if path == source.root or not loaded.is_disposable_worktree() or loaded.is_branch_protected():
            raise ValueError("resume target is not a disposable writable worktree")
        if _git_common_directory(source.root) != _git_common_directory(loaded.root):
            raise ValueError("resume target belongs to another repository")
        changed_files = list_changed_files(loaded.root)
        if not _paths_within(changed_files, allowed_write_paths):
            raise ValueError("resume target contains out-of-scope changes")
        if not isinstance(expected_changed_files, list) or not all(
            isinstance(item, str) and item for item in expected_changed_files
        ):
            raise ValueError("resume state has invalid changed files")
        if not _paths_within(expected_changed_files, allowed_write_paths):
            raise ValueError("resume state contains out-of-scope changes")
        for key, value in (
            ("runReceipt", expected_run_receipt),
            ("executorReceipt", expected_executor_receipt),
        ):
            if prior.get(key) != value or not _is_attested_report_file(options.report_dir, value):
                raise ValueError(f"resume state has an invalid {key}")
        if expected_commit is not None:
            if prior.get("commitSha") != expected_commit or loaded.commit_sha != expected_commit or changed_files:
                raise ValueError("committed resume target is not clean at the attested commit")
            if prior.get("changedFiles") != expected_changed_files:
                raise ValueError("committed resume state does not match its receipt")
        elif not allow_dirty or sorted(changed_files) != sorted(expected_changed_files):
            raise ValueError("dirty resume target does not match its attested state")
        return path
    except (BoundedDeliveryError, OSError, subprocess.SubprocessError, ValueError) as exc:
        raise BoundedDeliveryError("resume-worktree-invalid") from exc


def _git_common_directory(root: Path) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise ValueError("cannot inspect Git common directory")
    value = Path(result.stdout.strip())
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _is_attested_report_file(report_dir: Path, value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    root = report_dir.resolve()
    path = Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path.is_file()


def _add_tokens(options: BoundedDeliveryOptions, context: dict[str, Any], result: ProviderResult) -> bool:
    tokens = result.data.get("tokenUsage", 0)
    context["tokenUsage"] += tokens if isinstance(tokens, int) and tokens >= 0 else 0
    return context["tokenUsage"] <= options.limits.max_token_usage


def _complete(options: BoundedDeliveryOptions, context: dict[str, Any]) -> dict[str, Any]:
    _write_state(options, "completed", "complete", context)
    return {"status": "completed", **context, "statePath": str(options.state_path)}


def _stop(options: BoundedDeliveryOptions, context: dict[str, Any], reason: str, stage: str) -> dict[str, Any]:
    _write_state(options, "attention-required", stage, context, reason)
    return {"status": "attention-required", "stopReason": reason, **context, "statePath": str(options.state_path)}


def _write_state(options: BoundedDeliveryOptions, status: str, stage: str, context: dict[str, Any], reason: str | None = None) -> None:
    options.state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = redact_secrets({"schemaVersion": 1, "status": status, "stage": stage, "stopReason": reason, **context})
    options.state_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _native_success(result: ProviderResult, expected_provider: str) -> bool:
    return result.success and result.provider == expected_provider and result.provider != "mock"


def _provider_stop_reason(result: ProviderResult) -> str:
    if result.provider == "mock":
        return "mock-provider-denied"
    if result.error_type == ProviderErrorType.RATE_LIMIT:
        return "provider-quota-exhausted"
    if result.error_type == ProviderErrorType.TIMEOUT:
        return "provider-timeout"
    if result.error_type == ProviderErrorType.NETWORK:
        return "provider-network-error"
    return "provider-native-execution-failed"


def _read_only_git_fingerprint(root: Path) -> str:
    """Fingerprint Git-visible state before and after every read-only stage."""
    outputs: list[bytes] = []
    for args in (
        ["git", "rev-parse", "--verify", "HEAD"],
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
    ):
        try:
            result = subprocess.run(
                args,
                cwd=root,
                check=False,
                capture_output=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BoundedDeliveryError("read-only-stage-git-inspection-failed") from exc
        if result.returncode != 0:
            raise BoundedDeliveryError("read-only-stage-git-inspection-failed")
        outputs.append(result.stdout)
    return hashlib.sha256(b"\0".join(outputs)).hexdigest()


def _required_strings(payload: dict[str, Any], key: str) -> list[str]:
    value = _string_list(payload.get(key), key)
    if not value:
        raise BoundedDeliveryError(f"{key} must be non-empty")
    return value


def _validate_plan_scope(payload: dict[str, Any], contract: TrustedTaskContract) -> None:
    if not _paths_within(_required_strings(payload, "allowedWritePaths"), contract.allowed_write_paths):
        raise PolicyValidationError("architect plan expands allowed write paths")
    if not set(_required_strings(payload, "validationCommands")).issubset(set(contract.validation_commands)):
        raise PolicyValidationError("architect plan expands validation commands")


def _validate_contract_validation_commands(project: Any, contract: TrustedTaskContract) -> None:
    baseline = [
        project.profile.commands.lint,
        project.profile.commands.typecheck,
        project.profile.commands.test,
        project.profile.commands.build,
    ]
    if not all(isinstance(command, str) and command.strip() for command in baseline):
        raise BoundedDeliveryError("project contract must define lint, typecheck, test, and build for bounded delivery")
    command_set = set(contract.validation_commands)
    baseline_set = set(baseline)
    if not baseline_set.issubset(command_set):
        raise BoundedDeliveryError("trusted task contract must run the project lint, typecheck, test, and build commands")
    allowed = baseline_set | set(project.profile.commands.additional_validation)
    if not command_set.issubset(allowed):
        raise BoundedDeliveryError("trusted task contract contains an undeclared additional validation command")


def _validate_allowed_write_paths(paths: tuple[str, ...]) -> None:
    for raw_path in paths:
        path = Path(raw_path)
        normalized = path.as_posix().rstrip("/")
        if not normalized or path.is_absolute() or ".." in path.parts or normalized in {".", ".git"}:
            raise BoundedDeliveryError("allowedWritePaths must be safe project-relative paths")
        if any(part in {".git", "node_modules", ".next", "coverage", "dist", "build", "logs", "receipts"} for part in path.parts):
            raise BoundedDeliveryError("allowedWritePaths contains a protected generated or Git path")
        name = path.name.lower()
        if name == ".env" or name.startswith(".env.") or any(token in name for token in ("credential", "session", "secret", "token")):
            raise BoundedDeliveryError("allowedWritePaths contains a sensitive path")


def _validate_change_policy_paths(contract: TrustedTaskContract) -> None:
    normalized = [Path(raw_path).as_posix().rstrip("/") for raw_path in contract.allowed_write_paths]
    if any(path == "prisma/schema.prisma" or path.endswith("/schema.prisma") for path in normalized):
        if not contract.change_policy.schema_changes:
            raise BoundedDeliveryError("schema write path requires changePolicy.schemaChanges")
    if any("migrations" in Path(path).parts for path in normalized):
        if not contract.change_policy.migration_artifacts:
            raise BoundedDeliveryError("migration write path requires changePolicy.migrationArtifacts")
    if any(any(part.lower() in {"fixture", "fixtures", "__fixtures__"} for part in Path(path).parts) for path in normalized):
        if not contract.change_policy.fixture_data:
            raise BoundedDeliveryError("fixture write path requires changePolicy.fixtureData")


def _validate_secondary_review(
    secondary: dict[str, Any],
    stage: str,
    provider: str,
    contract: TrustedTaskContract,
) -> dict[str, Any]:
    failure_prefix = f"{stage}-{provider}-read-only-review"
    content = secondary.get("content")
    if not isinstance(content, str):
        raise BoundedDeliveryError(f"{failure_prefix}-content-missing")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise BoundedDeliveryError(f"{failure_prefix}-is-not-structured") from exc
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA or payload.get("stage") != stage:
        raise BoundedDeliveryError(f"{failure_prefix}-schema-invalid")
    if (
        payload.get("status") != "passed"
        or not isinstance(payload.get("findings"), list)
        or not isinstance(payload.get("tests"), list)
        or not isinstance(payload.get("blockers"), list)
    ):
        raise BoundedDeliveryError(f"{failure_prefix}-not-approved")
    if stage in {"qa", "review"}:
        try:
            tests = _string_list(payload.get("tests"), "tests")
        except BoundedDeliveryError as exc:
            raise BoundedDeliveryError(f"{failure_prefix}-tests-missing") from exc
        if not tests:
            raise BoundedDeliveryError(f"{failure_prefix}-tests-missing")
    _findings(payload)
    _reject_forbidden_review_payload(
        payload,
        f"{stage} {provider} review",
        contract,
    )
    return payload


def _findings(payload: dict[str, Any]) -> list[dict[str, str]]:
    findings = payload.get("findings")
    if not isinstance(findings, list):
        raise BoundedDeliveryError("findings must be a list")
    normalized: list[dict[str, str]] = []
    for finding in findings:
        if not isinstance(finding, dict) or not isinstance(finding.get("path"), str) or not isinstance(finding.get("message"), str):
            raise BoundedDeliveryError("findings must include path and message")
        normalized.append({"path": finding["path"], "message": finding["message"]})
    return normalized


def _findings_are_attributable(findings: list[dict[str, str]], allowed: tuple[str, ...]) -> bool:
    return all(_paths_within([finding["path"]], allowed) for finding in findings)


def _paths_within(paths: list[str], allowed: tuple[str, ...]) -> bool:
    roots = tuple(Path(item).as_posix().rstrip("/") for item in allowed)
    normalized: list[str] = []
    for value in paths:
        if not isinstance(value, str):
            return False
        path = Path(value)
        if path.is_absolute() or not value or ".." in path.parts or "." in path.parts:
            return False
        normalized.append(path.as_posix().rstrip("/"))
    return all(
        any(path == root or path.startswith(f"{root}/") for root in roots)
        for path in normalized
    )


def _engineering_instruction(context: dict[str, Any], repairs: list[dict[str, Any]]) -> str:
    return json.dumps({"task": context["task"], "acceptanceCriteria": context["acceptanceCriteria"], "plan": context["plan"], "repairs": repairs}, ensure_ascii=False)


def _schema_or_api_change_allowed(contract: TrustedTaskContract) -> bool:
    return contract.change_policy.schema_changes or contract.change_policy.api_contract_changes


def _reject_forbidden(
    value: str,
    label: str,
    contract: TrustedTaskContract | None = None,
    *,
    allow_negated_review_evidence: bool = False,
) -> None:
    lowered = value.lower()
    if any(re.search(pattern, lowered) for pattern in FORBIDDEN_PATTERNS):
        raise PolicyValidationError(f"{label} contains a prohibited action or product-contract change")
    policy = contract.change_policy if contract is not None else TaskChangePolicy()
    if (
        MIGRATION_ARTIFACT_PATTERN.search(lowered)
        and not policy.migration_artifacts
        and not (
            allow_negated_review_evidence
            and _policy_mentions_are_only_absence_evidence(lowered, MIGRATION_ARTIFACT_PATTERN)
        )
    ):
        raise PolicyValidationError(f"{label} contains an unauthorized migration artifact")
    if (
        SCHEMA_CHANGE_PATTERN.search(lowered)
        and not policy.schema_changes
        and not (
            allow_negated_review_evidence
            and _policy_mentions_are_only_absence_evidence(lowered, SCHEMA_CHANGE_PATTERN)
        )
    ):
        raise PolicyValidationError(f"{label} contains an unauthorized schema change")
    if (
        API_CONTRACT_CHANGE_PATTERN.search(lowered)
        and not policy.api_contract_changes
        and not (
            allow_negated_review_evidence
            and _policy_mentions_are_only_absence_evidence(lowered, API_CONTRACT_CHANGE_PATTERN)
        )
    ):
        raise PolicyValidationError(f"{label} contains an unauthorized API contract change")
    if (
        FIXTURE_DATA_PATTERN.search(lowered)
        and not policy.fixture_data
        and not (
            allow_negated_review_evidence
            and _policy_mentions_are_only_absence_evidence(lowered, FIXTURE_DATA_PATTERN)
        )
    ):
        raise PolicyValidationError(f"{label} contains unauthorized fixture data")


def _reject_forbidden_review_payload(
    payload: Any,
    label: str,
    contract: TrustedTaskContract,
) -> None:
    """Validate every review string while allowing narrow absence assertions.

    Independent reviewers commonly cite scope evidence such as "no migration
    changes are present". That sentence is evidence that the deny-by-default
    policy held, not a request to create the artifact. Positive proposals,
    mixed statements, and prohibited actions still fail closed.
    """
    for value in _iter_string_values(payload):
        _reject_forbidden(
            value,
            label,
            contract,
            allow_negated_review_evidence=True,
        )


def _iter_string_values(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_string_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_string_values(item)


def _policy_mentions_are_only_absence_evidence(text: str, pattern: re.Pattern[str]) -> bool:
    sentence_boundaries = ".!?;\n。！？；"
    for match in pattern.finditer(text):
        start = max(text.rfind(boundary, 0, match.start()) for boundary in sentence_boundaries) + 1
        ends = [text.find(boundary, match.end()) for boundary in sentence_boundaries]
        end = min((position for position in ends if position >= 0), default=len(text))
        before = text[start:match.start()]
        after = text[match.end():end]
        english_absence = (
            re.search(r"\b(?:no|without)\b", before) is not None
            and (
                re.search(
                    r"\b(?:changes?|artifacts?|files?|paths?|data)\b.{0,40}\b(?:are|is|were|was)?\s*(?:present|included|added|modified|touched|created)\b",
                    after,
                )
                is not None
                or re.search(
                    r"^\s*(?:are|is|were|was)\s+(?:present|included|added|modified|touched|created)\b",
                    after,
                )
                is not None
            )
        )
        chinese_absence = (
            re.search(r"(?:未產生|未新增|未修改|未包含|沒有|不含|並未|無)", before) is not None
            and re.search(r"(?:變更|artifact|檔案|路徑|資料)", after) is not None
        )
        chinese_prohibition = (
            re.search(r"(?:不涉及|不得(?!不)|不可|禁止|無需)[^。！？；\n]{0,120}$", before)
            is not None
            and re.search(r"(?:但|卻|然而|不過|反而)", before) is None
        )
        if not english_absence and not chinese_absence and not chinese_prohibition:
            return False
    return True


def _validate_stage_structure(stage: str, payload: Any) -> None:
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA or payload.get("stage") != stage:
        raise BoundedDeliveryError(f"{stage} returned an invalid bounded-delivery schema")
    if (
        payload.get("status") != "passed"
        or not isinstance(payload.get("findings"), list)
        or not isinstance(payload.get("tests"), list)
        or not isinstance(payload.get("blockers"), list)
    ):
        raise BoundedDeliveryError(f"{stage} did not provide a passed, structured result")
    if payload["blockers"]:
        raise BoundedDeliveryError(f"{stage} returned blockers")
    if stage in {"pm", "architect"} and payload["findings"]:
        raise BoundedDeliveryError(f"{stage} returned findings")
    if stage == "pm":
        _required_strings(payload, "acceptanceCriteria")
    elif stage == "architect":
        _required_strings(payload, "plan")
        _required_strings(payload, "allowedWritePaths")
        _required_strings(payload, "validationCommands")
    elif stage in {"qa", "review"}:
        _findings(payload)
        _required_strings(payload, "tests")


def _validation_failure(kind: str, stop_reason: str, **evidence: Any) -> dict[str, Any]:
    return {
        **evidence,
        "validationError": stop_reason,
        "stopReason": stop_reason,
        "validation": {"success": False, "kind": kind, "stopReason": stop_reason},
    }


def _receipt_validation(result: ProviderResult, evidence: dict[str, Any]) -> dict[str, Any]:
    value = evidence.get("validation")
    validation = dict(value) if isinstance(value, dict) else {"success": result.success, "kind": "provider-output"}
    stop_reason = evidence.get("stopReason") or evidence.get("validationError")
    if isinstance(stop_reason, str) and stop_reason:
        validation["success"] = False
        validation.setdefault("kind", "stage-validation")
        validation["stopReason"] = stop_reason
    return validation


def _optional_string_list(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    return _string_list(value, label)


def _load_change_policy(value: Any) -> TaskChangePolicy:
    if value is None:
        return TaskChangePolicy()
    if not isinstance(value, dict):
        raise BoundedDeliveryError("changePolicy must be an object")
    allowed_keys = {
        "schemaChanges",
        "apiContractChanges",
        "migrationArtifacts",
        "fixtureData",
    }
    if set(value) - allowed_keys:
        raise BoundedDeliveryError("changePolicy contains unsupported fields")
    if not all(isinstance(item, bool) for item in value.values()):
        raise BoundedDeliveryError("changePolicy values must be booleans")
    policy = TaskChangePolicy(
        schema_changes=value.get("schemaChanges", False),
        api_contract_changes=value.get("apiContractChanges", False),
        migration_artifacts=value.get("migrationArtifacts", False),
        fixture_data=value.get("fixtureData", False),
    )
    if policy.migration_artifacts and not policy.schema_changes:
        raise BoundedDeliveryError("migrationArtifacts requires schemaChanges")
    return policy


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise BoundedDeliveryError(f"{label} must be a non-empty string list")
    normalized = [item.strip() for item in value]
    if len(set(normalized)) != len(normalized):
        raise BoundedDeliveryError(f"{label} must not contain duplicate values")
    return normalized


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()
