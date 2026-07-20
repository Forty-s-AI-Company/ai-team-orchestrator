"""Fail-closed staging external QA runner.

External QA is deliberately not part of disposable-worktree validation.  A
PayUni sandbox/browser test needs the source project's ignored ``.env.local``
and a reachable staging site.  This module therefore exposes one fixed
allowlisted command, gates it on an explicit sandbox flag, and writes only a
redacted receipt summary.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any

from ai_team.core.project_loader import LoadedProject
from ai_team.providers.base import SECRET_KEY_PATTERN, redact_secrets


SCHEMA = "ai-team-external-qa-receipt/v1"
ALLOWED_COMMANDS: dict[str, tuple[str, ...]] = {
    "npm run qa:payuni:sandbox": ("npm", "run", "qa:payuni:sandbox"),
}
MAX_OUTPUT_BYTES = 64_000
MAX_CHECK_DEPTH = 4
MAX_CHECK_FIELDS = 20
MAX_CHECK_ITEMS = 20
MAX_CHECK_STRING_CHARS = 300
TRUNCATED_DEPTH_MARKER = "<truncated: maximum depth>"
CALLBACK_TRADE_QUERY_PROVIDER_SIGNAL_FIELDS = (
    "tradeNotFound",
    "authentication",
    "invalidRequest",
    "processing",
    "providerRejection",
)
CALLBACK_TRADE_QUERY_SCALAR_FIELDS = (
    "attempt",
    "querySucceeded",
    "tradeStatus",
    "tradeNoPresent",
    "currentHttpsHostPath",
    "flowStage",
    "errorCategory",
)
CALLBACK_TRADE_QUERY_PROVIDER_RESULT_TYPES = frozenset(
    {"array", "boolean", "missing", "null", "number", "object", "string", "undefined"}
)
# These are PayUni's documented callback/query result names.  The receipt
# records names only, never provider values, and remains deliberately bounded.
CALLBACK_TRADE_QUERY_PROVIDER_RESULT_FIELDS = frozenset(
    {
        "Amt",
        "Auth",
        "AuthBank",
        "Card4No",
        "Card6No",
        "ECI",
        "EscrowType",
        "IP",
        "Inst",
        "InstEach",
        "InstFirst",
        "MerchantID",
        "MerchantOrderNo",
        "PayTime",
        "PaymentMethod",
        "PaymentType",
        "RespondCode",
        "RespondType",
        "Status",
        "TokenUseStatus",
        "TradeNo",
        "Version",
    }
)
MAX_CALLBACK_TRADE_QUERY_PROVIDER_RESULT_FIELDS = 12
LEGACY_CALLBACK_TRADE_QUERIES_RERUN_VERSION = 1
LEGACY_CALLBACK_TRADE_QUERIES_RERUN_REASON = "legacy-truncated-callback-trade-queries"
PROVIDER_RESULT_SHAPE_RERUN_VERSION = 2
PROVIDER_RESULT_SHAPE_RERUN_REASON = "missing-provider-result-shape"


@dataclass(frozen=True)
class ExternalQAResult:
    status: str
    result: dict[str, Any]
    receipt_path: Path | None = None


def run_external_qa(
    loaded: LoadedProject,
    revision: str,
    report_dir: Path,
    *,
    prior: dict[str, Any] | None = None,
    execute: bool = True,
    timeout_seconds: int = 1_200,
) -> ExternalQAResult:
    """Run the configured staging QA once for a source revision.

    ``not-configured`` is intentionally non-blocking: operators can add the
    sandbox flags/credentials to ``.env.local`` later without making every
    autonomous coding cycle fail.  Once explicitly enabled, any test failure
    is blocking and must be inspected before the task is marked complete.
    """

    config = loaded.profile.external_qa
    base: dict[str, Any] = {
        "schema": SCHEMA,
        "revision": revision,
        "environment": config.environment,
        "command": config.command,
        "reviewerRole": config.reviewer_role,
        "productionRequiresHumanApproval": config.production_requires_human_approval,
    }
    if not config.enabled:
        return ExternalQAResult("disabled", {**base, "status": "disabled"})
    if config.environment.lower() != "staging":
        return ExternalQAResult("blocked", {**base, "status": "blocked", "reason": "environment-not-staging"})
    command = ALLOWED_COMMANDS.get(config.command)
    if command is None:
        return ExternalQAResult("blocked", {**base, "status": "blocked", "reason": "command-not-allowlisted"})

    env_file = loaded.root / ".env.local"
    if not env_file.is_file():
        return ExternalQAResult("not-configured", {**base, "status": "not-configured", "reason": "env-file-missing"})
    if not _dotenv_flag_enabled(env_file, "PAYUNI_SANDBOX_QA_ENABLED"):
        return ExternalQAResult(
            "not-configured",
            {**base, "status": "not-configured", "reason": "PAYUNI_SANDBOX_QA_ENABLED-not-true"},
        )

    previous = prior if isinstance(prior, dict) else {}
    diagnostic_rerun: dict[str, Any] | None = None
    if config.run_once_per_revision and previous.get("revision") == revision:
        previous_status = str(previous.get("status") or "")
        if previous_status == "failed":
            if (
                not _provider_result_shape_rerun_was_performed(previous)
                and _has_callback_provider_result_missing_shape(previous)
            ):
                diagnostic_rerun = {
                    "version": PROVIDER_RESULT_SHAPE_RERUN_VERSION,
                    "reason": PROVIDER_RESULT_SHAPE_RERUN_REASON,
                }
            elif (
                not _legacy_callback_trade_queries_rerun_was_performed(previous)
                and _has_legacy_truncated_callback_trade_queries(previous)
            ):
                diagnostic_rerun = {
                    "version": LEGACY_CALLBACK_TRADE_QUERIES_RERUN_VERSION,
                    "reason": LEGACY_CALLBACK_TRADE_QUERIES_RERUN_REASON,
                }
        # A previous serializer truncated scalar callback evidence needed to
        # diagnose whether PayUni created or completed a trade. The current
        # bounded serializer preserves these values on this one retry.
        if previous_status in {"passed", "failed", "blocked"} and diagnostic_rerun is None:
            # Preserve the attested result. Replacing it with a small
            # ``already-run`` marker discarded the original Playwright error
            # and made watchdog diagnosis impossible. A failed revision stays
            # blocking until a repair creates a new Git revision.
            return ExternalQAResult(
                previous_status,
                {
                    **base,
                    **previous,
                    "status": previous_status,
                    "reason": "already-run-for-revision",
                    "receiptPath": previous.get("receiptPath"),
                },
                Path(previous["receiptPath"]) if isinstance(previous.get("receiptPath"), str) else None,
            )
    if not execute:
        return ExternalQAResult("ready", {**base, "status": "ready"})

    execution_base = base
    if diagnostic_rerun is not None:
        # This runner-owned marker is persisted in the replacement receipt so
        # even malformed or untrusted QA output containing the old truncation
        # marker can trigger at most one additional sandbox execution.
        execution_base = {
            **base,
            "diagnosticRerun": diagnostic_rerun,
        }

    started = datetime.now(UTC)
    # Do not let a supervisor process accidentally override the sandbox values
    # loaded by Node from the source project's .env.local.  Keep only process
    # plumbing; the app's own env file supplies DATABASE_URL and PayUni values.
    child_env = {
        key: value
        for key, value in os.environ.items()
        if key not in {
            "DATABASE_URL",
            "DIRECT_URL",
            "NEXT_PUBLIC_APP_URL",
            "PAYMENT_PROVIDER",
            "PAYUNI_ENV",
            "PAYUNI_API_BASE_URL",
            "PAYUNI_MERCHANT_ID",
            "PAYUNI_HASH_KEY",
            "PAYUNI_HASH_IV",
            "PAYUNI_WEBHOOK_SECRET",
            "PAYUNI_SANDBOX_QA_ENABLED",
            "PAYUNI_SANDBOX_REFUND_ENABLED",
            "PAYUNI_TEST_APP_URL",
            "PAYUNI_TEST_LIVE_PATH",
            "PAYUNI_TEST_CARD_NUMBER",
            "PAYUNI_TEST_EXPIRY",
            "PAYUNI_TEST_CVV",
        }
    }
    child_env["AI_TEAM_PROJECT_REVISION"] = revision
    try:
        completed = subprocess.run(
            list(command),
            cwd=loaded.root,
            env=child_env,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        result = {
            **execution_base,
            "status": "failed",
            "reason": "timeout",
            "startedAt": started.isoformat(),
            "completedAt": datetime.now(UTC).isoformat(),
        }
        receipt = _write_receipt(report_dir, result, "")
        return ExternalQAResult("failed", {**result, "receiptPath": str(receipt)}, receipt)
    except OSError as exc:
        result = {**execution_base, "status": "failed", "reason": "command-unavailable"}
        receipt = _write_receipt(report_dir, result, str(exc))
        return ExternalQAResult("failed", {**result, "receiptPath": str(receipt)}, receipt)

    output = ((completed.stdout or "") + (completed.stderr or ""))[-MAX_OUTPUT_BYTES:]
    parsed = _last_json_object(output)
    success = (
        completed.returncode == 0
        and parsed.get("success") is True
        and parsed.get("schema") == "celebratedeal-payuni-sandbox-qa/v1"
        and parsed.get("environment") == "sandbox"
        and isinstance(parsed.get("productionValidation"), dict)
        and parsed["productionValidation"].get("automatedChargeAllowed") is False
    )
    status = "passed" if success else "failed"
    result = {
        **execution_base,
        "status": status,
        "exitCode": completed.returncode,
        "startedAt": started.isoformat(),
        "completedAt": datetime.now(UTC).isoformat(),
        "outputSha256": hashlib.sha256(output.encode("utf-8", "replace")).hexdigest(),
        "providerChecks": _safe_checks(parsed),
        "error": str(parsed.get("error"))[:300] if isinstance(parsed.get("error"), str) else None,
    }
    receipt = _write_receipt(report_dir, result, output)
    return ExternalQAResult(status, {**result, "receiptPath": str(receipt)}, receipt)


def _dotenv_flag_enabled(path: Path, name: str) -> bool:
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() != name:
                continue
            return value.strip().strip("'\"").lower() == "true"
    except OSError:
        return False
    return False


def _last_json_object(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _has_legacy_truncated_callback_trade_queries(receipt: dict[str, Any]) -> bool:
    """Identify the old failed-receipt shape eligible for one diagnostic retry."""

    checks = receipt.get("providerChecks")
    if not isinstance(checks, dict):
        return False
    provider_checks = checks.get("providerChecks")
    if not isinstance(provider_checks, dict):
        return False
    callback_trade_queries = provider_checks.get("callbackTradeQueries")
    if not isinstance(callback_trade_queries, list) or not callback_trade_queries:
        return False
    # ``providerSignals`` was a nested object at the old depth boundary.  A
    # legacy receipt therefore has this exact marker in every recorded query;
    # mixed evidence is not an eligible legacy shape and stays cached.
    return all(
        isinstance(query, dict)
        and query.get("providerSignals") == TRUNCATED_DEPTH_MARKER
        for query in callback_trade_queries
    )


def _legacy_callback_trade_queries_rerun_was_performed(receipt: dict[str, Any]) -> bool:
    """Return whether this receipt is the bounded replacement diagnostic run."""

    rerun = receipt.get("diagnosticRerun")
    return (
        isinstance(rerun, dict)
        and rerun.get("version") == LEGACY_CALLBACK_TRADE_QUERIES_RERUN_VERSION
        and rerun.get("reason") == LEGACY_CALLBACK_TRADE_QUERIES_RERUN_REASON
    )


def _has_callback_provider_result_missing_shape(receipt: dict[str, Any]) -> bool:
    """Identify failed provider-result attempts that predate the v2 summary."""

    checks = receipt.get("providerChecks")
    if not isinstance(checks, dict):
        return False
    provider_checks = checks.get("providerChecks")
    if not isinstance(provider_checks, dict):
        return False
    callback_trade_queries = provider_checks.get("callbackTradeQueries")
    if not isinstance(callback_trade_queries, list):
        return False
    return any(
        isinstance(query, dict)
        and query.get("flowStage") == "provider-result"
        and "providerResultType" not in query
        for query in callback_trade_queries
    )


def _provider_result_shape_rerun_was_performed(receipt: dict[str, Any]) -> bool:
    """Return whether the one-time v2 provider-result diagnostic was run."""

    rerun = receipt.get("diagnosticRerun")
    return (
        isinstance(rerun, dict)
        and rerun.get("version") == PROVIDER_RESULT_SHAPE_RERUN_VERSION
        and rerun.get("reason") == PROVIDER_RESULT_SHAPE_RERUN_REASON
    )


def _safe_checks(parsed: dict[str, Any]) -> dict[str, Any] | None:
    checks = parsed.get("checks")
    if not isinstance(checks, dict):
        return None
    safe = _safe_check_value(checks, depth=0, path=())
    return safe if isinstance(safe, dict) else None


def _safe_check_value(value: Any, *, depth: int, path: tuple[str, ...]) -> Any:
    """Keep bounded, JSON-shaped provider evidence for a redacted receipt."""

    # Leaf evidence at the depth boundary remains safe under its existing type
    # and length limits.  The one exception below is deliberately narrow: the
    # nested PayUni classification object is necessary to diagnose a failed
    # callback, but it must not become a general depth-limit bypass.
    if path == ("providerChecks", "callbackTradeQueries", "providerSignals"):
        return _safe_callback_trade_query_provider_signals(value)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value[:MAX_CHECK_STRING_CHARS]
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else "<unsupported number>"
    if depth >= MAX_CHECK_DEPTH:
        return "<truncated: maximum depth>"
    if isinstance(value, dict):
        summary: dict[str, Any] = {}
        for key, item in islice(value.items(), MAX_CHECK_FIELDS):
            if not isinstance(key, str):
                continue
            safe_key = key[:MAX_CHECK_STRING_CHARS]
            # Detect sensitive names before truncating the key.  Otherwise a
            # deliberately long key ending in e.g. ``apiKey`` would evade the
            # receipt's normal key-based redaction once its suffix is dropped.
            if SECRET_KEY_PATTERN.search(key):
                summary[safe_key] = "<redacted>" if item else item
            else:
                summary[safe_key] = _safe_check_value(
                    item,
                    depth=depth + 1,
                    path=path + (key,),
                )
        return summary
    if isinstance(value, list):
        if path == ("providerChecks", "callbackTradeQueries"):
            return [
                _safe_callback_trade_query_attempt(item)
                if isinstance(item, dict)
                else _safe_check_value(item, depth=depth + 1, path=path)
                for item in value[:MAX_CHECK_ITEMS]
            ]
        return [
            _safe_check_value(item, depth=depth + 1, path=path)
            for item in value[:MAX_CHECK_ITEMS]
        ]
    return "<unsupported value>"


def _safe_callback_trade_query_provider_signals(value: Any) -> dict[str, bool]:
    """Serialize only the fixed boolean PayUni callback classifications."""

    if not isinstance(value, dict):
        return {}
    return {
        field: value[field]
        for field in CALLBACK_TRADE_QUERY_PROVIDER_SIGNAL_FIELDS
        if isinstance(value.get(field), bool)
    }


def _safe_callback_trade_query_attempt(value: dict[str, Any]) -> dict[str, Any]:
    """Serialize one PayUni callback query without relying on input order.

    This intentionally does not call the generic check serializer: a callback
    attempt can contain arbitrary provider data, while only these fixed
    diagnostics are useful and safe to persist.
    """

    summary: dict[str, Any] = {}
    for field in CALLBACK_TRADE_QUERY_SCALAR_FIELDS:
        item = value.get(field)
        if field == "attempt":
            if isinstance(item, int) and not isinstance(item, bool):
                summary[field] = item
        elif field in {"querySucceeded", "tradeNoPresent"}:
            if isinstance(item, bool):
                summary[field] = item
        elif field == "errorCategory":
            if item is None or isinstance(item, str):
                summary[field] = item if item is None else item[:MAX_CHECK_STRING_CHARS]
        elif isinstance(item, str):
            summary[field] = item[:MAX_CHECK_STRING_CHARS]

    provider_signals = _safe_callback_trade_query_provider_signals(value.get("providerSignals"))
    if "providerSignals" in value:
        summary["providerSignals"] = provider_signals

    provider_result_type = value.get("providerResultType")
    if isinstance(provider_result_type, str) and provider_result_type in CALLBACK_TRADE_QUERY_PROVIDER_RESULT_TYPES:
        summary["providerResultType"] = provider_result_type

    provider_result_fields = value.get("providerResultFields")
    if isinstance(provider_result_fields, list):
        safe_fields: list[str] = []
        for field in provider_result_fields:
            if (
                isinstance(field, str)
                and field in CALLBACK_TRADE_QUERY_PROVIDER_RESULT_FIELDS
                and field not in safe_fields
            ):
                safe_fields.append(field)
            if len(safe_fields) == MAX_CALLBACK_TRADE_QUERY_PROVIDER_RESULT_FIELDS:
                break
        summary["providerResultFields"] = safe_fields
    return summary


def _write_receipt(report_dir: Path, result: dict[str, Any], output: str) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        **result,
        "outputSha256": hashlib.sha256(output.encode("utf-8", "replace")).hexdigest(),
    }
    path = report_dir / f"external-qa-{result.get('revision', 'unknown')[:12]}.json"
    path.write_text(json.dumps(redact_secrets(receipt), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
