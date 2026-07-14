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

from ai_team.core.isolated_executor import IsolatedRunResult, run_in_disposable_worktree
from ai_team.core.project_loader import load_project
from ai_team.providers.base import BaseProvider, ProviderErrorType, ProviderRequest, ProviderResult, redact_secrets


SCHEMA = "ai-team-bounded-delivery/v1"
FORBIDDEN_PATTERNS = (
    r"\b(?:database\s+)?migrat(?:e|ion|ions)\b",
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
    r"\b(?:schema\s+change|api\s+contract\s+change)\b",
    r"\b(?:prisma\s+db\s+push|drizzle(?:-kit)?\s+push|git\s+push|gh\s+(?:pr|api))\b",
)
ROLE_BY_STAGE = {"pm": "product-manager", "architect": "architect", "engineer": "engineer", "qa": "delivery-qa", "review": "reviewer"}
SECONDARY_PROVIDER_BY_STAGE = {"architect": "codex", "qa": "codex", "review": "antigravity"}
RECEIPT_FILE_PATTERN = re.compile(r"^(\d+)-(pm|architect|engineer|qa|review)\.json$")


class BoundedDeliveryError(ValueError):
    """Raised when a task contract or a stage result is unsafe."""


class PolicyValidationError(BoundedDeliveryError):
    """Raised when otherwise structured stage output violates delivery policy."""


@dataclass(frozen=True)
class TrustedTaskContract:
    id: str
    title: str
    source_kind: str
    source_reference: str
    instruction: str
    allowed_write_paths: tuple[str, ...]
    validation_commands: tuple[str, ...]


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
    if not paths or not commands:
        raise BoundedDeliveryError("trusted write tasks require allowedWritePaths and validationCommands")
    contract = TrustedTaskContract(
        id=values["id"].strip(), title=values["title"].strip(), source_kind=values["source_kind"].strip(),
        source_reference=values["source_reference"].strip(), instruction=values["instruction"].strip(),
        allowed_write_paths=tuple(paths), validation_commands=tuple(commands),
    )
    _reject_forbidden(contract.instruction, "task instruction")
    _validate_allowed_write_paths(contract.allowed_write_paths)
    for command in contract.validation_commands:
        _reject_forbidden(command, "validation command")
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
    _write_state(options, "running", "pm", context)
    try:
        pm = _run_stage(options, project.root, "pm", context, contract)
        acceptance = _required_strings(pm, "acceptanceCriteria")
        context["acceptanceCriteria"] = acceptance
        architect = _run_stage(options, project.root, "architect", context, contract)
        plan = _required_strings(architect, "plan")
        context["plan"] = plan
        context["planAllowedWritePaths"] = _required_strings(architect, "allowedWritePaths")

        repairs: list[dict[str, Any]] = []
        for iteration in range(1, options.limits.max_iterations + 1):
            instruction = _engineering_instruction(context, repairs)
            engineer = options.provider_for_role(ROLE_BY_STAGE["engineer"])
            attempt = (options.engineering_executor or _default_engineering_executor(options))(
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
    request = ProviderRequest(
        workflow=f"bounded-delivery-{stage}", project_root=root, run_mode="run-agent",
        timeout_seconds=options.limits.timeout_seconds,
        prompt=_stage_prompt(stage, context),
        metadata={
            "role": ROLE_BY_STAGE[stage], "writeRequired": False, "writeAccess": False,
            "taskSha": context["taskSha"], "boundedStage": stage,
            "requiredProvider": expected,
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
        _reject_forbidden(json.dumps(payload, ensure_ascii=False), f"{stage} output")
        if stage == "architect":
            if payload.get("schemaOrApiChange") is not False:
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


def _default_engineering_executor(options: BoundedDeliveryOptions) -> Callable[[TrustedTaskContract, str, BaseProvider, int], EngineeringAttempt]:
    reusable_worktree: Path | None = None

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


def _stage_prompt(stage: str, context: dict[str, Any]) -> str:
    task = context["task"]
    evidence = {
        "acceptanceCriteria": context.get("acceptanceCriteria", []),
        "plan": context.get("plan", []),
        "allowedWritePaths": task["allowed_write_paths"],
        "validationCommands": task["validation_commands"],
        "changedFiles": context.get("changedFiles", []),
        "commitSha": context.get("commitSha"),
        "validation": context.get("validation", {"success": False, "reason": "validation evidence unavailable"}),
        "repairs": [item["findingSha"] for item in context.get("repairs", [])],
    }
    safe_evidence = redact_secrets(evidence)
    tests_shape = "tests=['evidence citation']" if stage in {"qa", "review"} else "tests=[]"
    return "\n".join((
        f"Bounded delivery stage: {stage}", f"Task SHA: {context['taskSha']}",
        f"Task: {redact_secrets(task['title'])}", f"Instruction: {redact_secrets(task['instruction'])}",
        f"Acceptance criteria: {json.dumps(safe_evidence['acceptanceCriteria'])}",
        f"Allowed write paths: {json.dumps(safe_evidence['allowedWritePaths'])}",
        f"Validation commands: {json.dumps(safe_evidence['validationCommands'])}",
        f"Implementation evidence: {json.dumps(safe_evidence)}",
        f"Return only JSON with schema='ai-team-bounded-delivery/v1', stage, status='passed', findings=[], {tests_shape}, blockers=[].",
        "Do not edit files, run shell commands, deploy, migrate, seed, process payments, read secrets, or propose schema/API changes.",
        "PM: include non-empty acceptanceCriteria. Architect: include non-empty plan, allowedWritePaths, validationCommands, schemaOrApiChange=false.",
        "PM/architect: findings and blockers must be exactly []; do not restate required work as a finding or blocker.",
        "QA/review: evaluate every acceptance criterion; tests must cite non-empty validation or regression evidence; findings must be [] only when passed.",
        "QA/review: each failed finding must include path and message; blockers must be exactly [] for status='passed'.",
    ))


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
    required = [
        project.profile.commands.lint,
        project.profile.commands.typecheck,
        project.profile.commands.test,
        project.profile.commands.build,
    ]
    if not all(isinstance(command, str) and command.strip() for command in required):
        raise BoundedDeliveryError("project contract must define lint, typecheck, test, and build for bounded delivery")
    if set(contract.validation_commands) != set(required):
        raise BoundedDeliveryError("trusted task contract must run exactly the project lint, typecheck, test, and build commands")


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


def _validate_secondary_review(
    secondary: dict[str, Any],
    stage: str,
    provider: str,
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
    _reject_forbidden(
        json.dumps(payload, ensure_ascii=False),
        f"{stage} {provider} review",
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
    return all(any(path == root or path.startswith(f"{root}/") for root in roots) for path in paths)


def _engineering_instruction(context: dict[str, Any], repairs: list[dict[str, Any]]) -> str:
    return json.dumps({"task": context["task"], "acceptanceCriteria": context["acceptanceCriteria"], "plan": context["plan"], "repairs": repairs}, ensure_ascii=False)


def _reject_forbidden(value: str, label: str) -> None:
    lowered = value.lower()
    if any(re.search(pattern, lowered) for pattern in FORBIDDEN_PATTERNS):
        raise PolicyValidationError(f"{label} contains a prohibited action or product-contract change")


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
