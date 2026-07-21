from __future__ import annotations

import base64
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from ai_team.core.watchdog_repair import (
    AIRepairer,
    AutoRepairOptions,
    attempt_auto_repair,
    repair_key,
    requires_manual_review,
)
from ai_team.core.telegram_notify import (
    TelegramSettings,
    load_telegram_settings,
    send_telegram_message,
)


@dataclass(frozen=True)
class WatchdogThresholds:
    repeat_count: int = 3
    restart_count: int = 3
    idle_count: int = 5
    stale_seconds: int = 25 * 60
    cooldown_seconds: int = 30 * 60
    repair_restart_grace_seconds: int = 2 * 60


Runner = Callable[..., subprocess.CompletedProcess[str]]
Notifier = Callable[[str, str], bool]
TelegramSender = Callable[..., bool]
INFORMATIONAL_ALERT_TYPES = {
    "task-deferred",
    "release-review-required",
    "provider-backoff",
}


def run_watchdog(
    supervisor_state_path: Path,
    watchdog_state_path: Path,
    alert_log_path: Path,
    *,
    service_name: str,
    report_dir: Path | None = None,
    thresholds: WatchdogThresholds = WatchdogThresholds(),
    now: datetime | None = None,
    runner: Runner = subprocess.run,
    notifier: Notifier | None = None,
    powershell_path: str = "powershell.exe",
    auto_repair: AutoRepairOptions = AutoRepairOptions(),
    ai_repairer: AIRepairer | None = None,
) -> dict[str, Any]:
    """Inspect one supervisor snapshot and emit at most one deduplicated alert."""

    _validate_thresholds(thresholds)
    _validate_auto_repair(auto_repair)
    checked_at = _as_utc(now or datetime.now(UTC))
    supervisor = _read_json(supervisor_state_path)
    previous = _read_json(watchdog_state_path, missing_ok=True)
    service = _service_status(service_name, runner)

    signature = _state_signature(supervisor)
    attention = _needs_attention(supervisor)
    repair_restart_grace = _awaiting_post_repair_heartbeat(
        previous,
        supervisor,
        service,
        checked_at,
        thresholds.repair_restart_grace_seconds,
    )
    same_signature_count = 0 if repair_restart_grace else (
        int(previous.get("sameSignatureCount") or 0) + 1
        if attention and previous.get("signature") == signature
        else (1 if attention else 0)
    )

    restart_total = _nonnegative_int(service.get("NRestarts"))
    previous_restart_total = _nonnegative_int(previous.get("lastRestartCount"))
    restart_delta = max(0, restart_total - previous_restart_total) if previous else 0
    supervisor_stale_seconds = _state_age_seconds(supervisor, checked_at)
    task_progress = _task_progress_evidence(report_dir, supervisor, checked_at)
    progress_age = task_progress.get("ageSeconds")
    stale_seconds = (
        min(supervisor_stale_seconds, progress_age)
        if isinstance(progress_age, int)
        else supervisor_stale_seconds
    )
    task_failure = _task_failure_evidence(report_dir, supervisor)
    idle_signature = _idle_stall_signature(supervisor)
    idle_stall_count = (
        0
        if repair_restart_grace or idle_signature is None
        else (
            _nonnegative_int(previous.get("idleStallCount")) + 1
            if previous.get("idleSignature") == idle_signature
            else 1
        )
    )

    alert = None
    if not repair_restart_grace:
        alert = _select_alert(
            supervisor,
            service,
            signature,
            same_signature_count,
            idle_signature,
            idle_stall_count,
            restart_delta,
            stale_seconds,
            task_failure,
            thresholds,
        ) or _select_lifecycle_alert(supervisor, previous)
    manual_review_required = requires_manual_review(supervisor)
    informational_alert = bool(
        alert and alert.get("type") in INFORMATIONAL_ALERT_TYPES
    )
    current_repair_key = (
        None
        if manual_review_required or informational_alert
        else repair_key(supervisor, task_failure["reason"])
    )
    previous_repair_attempts = (
        _nonnegative_int(previous.get("repairAttempts"))
        if current_repair_key is not None and previous.get("repairKey") == current_repair_key
        else 0
    )
    repair: dict[str, Any] | None = None
    repair_attempts = previous_repair_attempts
    if alert and auto_repair.enabled and not manual_review_required and not informational_alert:
        if previous_repair_attempts < auto_repair.max_attempts:
            repair_attempts += 1
            repair = attempt_auto_repair(
                supervisor,
                alert_type=alert["type"],
                service_name=service_name,
                options=auto_repair,
                runner=runner,
                ai_repairer=ai_repairer,
                now=checked_at,
            )
        else:
            repair = {
                "attempted": False,
                "success": False,
                "action": "attempt-limit-reached",
                "diagnostic": "maximum automatic repair attempts reached; supervisor remains stopped",
                "restarted": False,
                "exhausted": True,
            }
        alert = _repair_alert(alert, repair, repair_attempts, auto_repair.max_attempts)
    elif manual_review_required or not _repair_condition_present(supervisor, task_failure, restart_delta):
        current_repair_key = None
        repair_attempts = 0
    # A repair can spend several minutes in Sol/Terra/QA. Production calls use
    # the actual completion time so the restart grace does not expire while
    # the repair itself is still running. Tests that pass ``now`` remain fully
    # deterministic.
    recorded_at = checked_at if now is not None else datetime.now(UTC)
    delivered = False
    suppressed = False
    if alert:
        suppressed = _within_cooldown(previous, alert["key"], checked_at, thresholds.cooldown_seconds)
        if not suppressed:
            send = notifier or (
                lambda title, message: send_watchdog_notifications(
                    title,
                    message,
                    powershell_path=powershell_path,
                    runner=runner,
                )
            )
            delivered = send(alert["title"], alert["message"])
            _append_alert_log(alert_log_path, recorded_at, alert, delivered)

    state = {
        "schemaVersion": 1,
        "updatedAt": recorded_at.isoformat(),
        "signature": signature,
        "sameSignatureCount": same_signature_count,
        "idleSignature": idle_signature,
        "idleStallCount": idle_stall_count,
        "lastSupervisorUpdatedAt": supervisor.get("updatedAt"),
        "lastTaskProgressAt": task_progress.get("updatedAt"),
        "lastTaskProgressStage": task_progress.get("stage"),
        "lastRestartCount": restart_total,
        "lastTaskFailureReason": task_failure["reason"],
        "taskFailureCount": task_failure["count"],
        "taskReceiptCount": task_failure["receiptCount"],
        "lastDeferredTaskCount": _observed_list_count(
            supervisor.get("deferredTasks"),
            previous.get("lastDeferredTaskCount"),
            alert.get("type") if alert else None,
            "task-deferred",
        ),
        "lastReleaseReviewTaskCount": _observed_list_count(
            supervisor.get("releaseReviewTasks"),
            previous.get("lastReleaseReviewTaskCount"),
            alert.get("type") if alert else None,
            "release-review-required",
        ),
        "lastProviderBackoffKey": _observed_provider_backoff(
            supervisor,
            previous.get("lastProviderBackoffKey"),
            alert.get("type") if alert else None,
        ),
        "autoRepairEnabled": auto_repair.enabled,
        "repairKey": current_repair_key,
        "repairAttempts": repair_attempts,
        "lastRepairAt": (
            recorded_at.isoformat()
            if repair and repair.get("attempted") is True
            else previous.get("lastRepairAt")
        ),
        "lastRepair": repair if repair is not None else previous.get("lastRepair"),
        "lastAlertKey": (
            alert["key"]
            if alert and not suppressed
            else previous.get("lastAlertKey")
        ),
        "lastAlertAt": (
            recorded_at.isoformat()
            if alert and not suppressed
            else previous.get("lastAlertAt")
        ),
    }
    _write_json_atomic(watchdog_state_path, state)

    status = "ok"
    if repair is not None:
        if repair.get("success") is True:
            status = "repaired"
        elif repair.get("exhausted") is True:
            status = "repair-exhausted"
        else:
            status = "repair-failed"
    elif alert and not suppressed:
        status = "alerted"
    return {
        "status": status,
        "alertType": alert.get("type") if alert else None,
        "notificationDelivered": delivered,
        "notificationSuppressed": suppressed,
        "repairRestartGrace": repair_restart_grace,
        "sameSignatureCount": same_signature_count,
        "idleStallCount": idle_stall_count,
        "restartDelta": restart_delta,
        "staleSeconds": stale_seconds,
        "supervisorStaleSeconds": supervisor_stale_seconds,
        "taskProgressAgeSeconds": progress_age,
        "taskProgressStage": task_progress.get("stage"),
        "taskFailureReason": task_failure["reason"],
        "taskFailureCount": task_failure["count"],
        "taskReceiptCount": task_failure["receiptCount"],
        "repairAttempts": repair_attempts,
        "repair": repair,
        "service": service,
    }


def _repair_condition_present(
    supervisor: dict[str, Any], task_failure: dict[str, Any], restart_delta: int
) -> bool:
    return _needs_attention(supervisor) or bool(task_failure["reason"]) or restart_delta > 0


def _awaiting_post_repair_heartbeat(
    previous: dict[str, Any],
    supervisor: dict[str, Any],
    service: dict[str, str],
    now: datetime,
    grace_seconds: int,
) -> bool:
    """Avoid re-reading the stale failure snapshot immediately after repair.

    A successful repair starts the supervisor asynchronously. Until that
    process writes a new ``updatedAt`` heartbeat, a timer invocation can still
    see the pre-repair attention state and launch a duplicate repair. The
    grace is intentionally short and applies only while the service is active
    or starting; a failed/dead service is never hidden by it.
    """

    repair = previous.get("lastRepair")
    if not isinstance(repair, dict):
        return False
    if repair.get("success") is not True or repair.get("restarted") is not True:
        return False
    if service.get("ActiveState") not in {"active", "activating"}:
        return False
    if supervisor.get("updatedAt") != previous.get("lastSupervisorUpdatedAt"):
        return False
    value = previous.get("lastRepairAt")
    if not isinstance(value, str):
        return False
    try:
        repaired_at = _as_utc(datetime.fromisoformat(value))
    except ValueError:
        return False
    elapsed = (now - repaired_at).total_seconds()
    return 0 <= elapsed < grace_seconds


def _repair_alert(
    original: dict[str, str],
    repair: dict[str, Any],
    attempts: int,
    maximum: int,
) -> dict[str, str]:
    if repair.get("deferred") is True:
        return {
            "type": "auto-repair-deferred",
            "key": f"auto-repair-deferred:{original['key']}",
            "title": "AI Team 任務已暫緩並跳過",
            "message": (
                f"同一問題在 {attempts}/{maximum} 次自動修復內仍未通過；"
                "已保存報告並繼續其他工作，請稍後查看暫緩原因。"
            ),
        }
    if repair.get("success") is True:
        return {
            "type": "auto-repair-success",
            "key": f"auto-repair-success:{original['key']}:{attempts}",
            "title": "AI Team 已自動修復並重啟",
            "message": f"修復動作：{repair.get('action')}；嘗試 {attempts}/{maximum}。",
        }
    if repair.get("exhausted") is True:
        return {
            "type": "auto-repair-exhausted",
            "key": f"auto-repair-exhausted:{original['key']}",
            "title": "AI Team 自動修復已停止",
            "message": f"同一問題已達 {maximum} 次修復上限；Supervisor 保持停止，請查看紀錄。",
        }
    return {
        "type": "auto-repair-failed",
        "key": f"auto-repair-failed:{original['key']}:{attempts}",
        "title": "AI Team 自動修復未完成",
        "message": (
            f"修復動作：{repair.get('action')}；嘗試 {attempts}/{maximum}。"
            "Supervisor 已停止，下一輪會在上限內重試。"
        ),
    }


def send_windows_toast(
    title: str,
    message: str,
    *,
    powershell_path: str = "powershell.exe",
    runner: Runner = subprocess.run,
) -> bool:
    """Send a Windows toast without interpolating untrusted text into PowerShell."""

    title_b64 = base64.b64encode(title.encode("utf-8")).decode("ascii")
    message_b64 = base64.b64encode(message.encode("utf-8")).decode("ascii")
    script = f"""
$title = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{title_b64}'))
$message = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{message_b64}'))
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] > $null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType=WindowsRuntime] > $null
$template = [Windows.UI.Notifications.ToastTemplateType]::ToastText02
$xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent($template)
$nodes = $xml.GetElementsByTagName('text')
$null = $nodes.Item(0).AppendChild($xml.CreateTextNode($title))
$null = $nodes.Item(1).AppendChild($xml.CreateTextNode($message))
$toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('AI Team Watchdog').Show($toast)
""".strip()
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    try:
        completed = runner(
            [
                powershell_path,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-EncodedCommand",
                encoded,
            ],
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def send_watchdog_notifications(
    title: str,
    message: str,
    *,
    powershell_path: str = "powershell.exe",
    runner: Runner = subprocess.run,
    telegram_settings: TelegramSettings | None = None,
    telegram_sender: TelegramSender = send_telegram_message,
) -> bool:
    """Attempt every configured channel; a notification failure never stops repair."""

    settings = telegram_settings or load_telegram_settings()
    telegram_delivered = False
    if settings.configured:
        telegram_delivered = telegram_sender(title, message, settings=settings)
    toast_delivered = send_windows_toast(
        title,
        message,
        powershell_path=powershell_path,
        runner=runner,
    )
    return telegram_delivered or toast_delivered


def _select_alert(
    supervisor: dict[str, Any],
    service: dict[str, str],
    signature: str,
    same_signature_count: int,
    idle_signature: str | None,
    idle_stall_count: int,
    restart_delta: int,
    stale_seconds: int,
    task_failure: dict[str, Any],
    thresholds: WatchdogThresholds,
) -> dict[str, str] | None:
    task_id = _safe_label((supervisor.get("currentTask") or {}).get("id"), "尚未分配任務")
    reason = _safe_label(supervisor.get("stopReason") or supervisor.get("nextAction"), "原因未提供")

    if restart_delta >= thresholds.restart_count:
        return {
            "type": "restart-loop",
            "key": f"restart-loop:{task_id}",
            "title": "AI Team 疑似重啟迴圈",
            "message": f"1 分鐘內重啟 {restart_delta} 次。任務：{task_id}。請查看 watchdog 告警紀錄。",
        }
    if service.get("ActiveState") == "failed":
        return {
            "type": "service-failed",
            "key": "service-failed",
            "title": "AI Team 服務失敗",
            "message": "Supervisor 已進入 failed 狀態，請查看 systemctl 與 watchdog 告警紀錄。",
        }
    if idle_signature is not None and idle_stall_count >= thresholds.idle_count:
        return {
            "type": "idle-loop",
            "key": f"idle-loop:{idle_signature}",
            "title": "AI Team 有心跳但沒有推進",
            "message": (
                f"已連續 {idle_stall_count} 輪沒有任務、佇列或 Git 進度。"
                "系統將清除失效的 PM 任務快取並重新掃描。"
            ),
        }
    if task_failure["reason"] and task_failure["count"] >= thresholds.repeat_count:
        repeated_reason = _safe_label(task_failure["reason"], "原因未提供")
        return {
            "type": "repeated-task-receipts",
            "key": f"repeated-task-receipts:{task_id}:{repeated_reason}",
            "title": "AI Team 疑似隱性迴圈",
            "message": (
                f"同一任務已有 {task_failure['count']} 份相同失敗紀錄。"
                f"任務：{task_id}；原因：{repeated_reason}。"
            ),
        }
    if _needs_attention(supervisor) and same_signature_count >= thresholds.repeat_count:
        return {
            "type": "repeated-attention",
            "key": f"repeated-attention:{signature}",
            "title": "AI Team 任務連續卡住",
            "message": f"同一問題已連續出現 {same_signature_count} 次。任務：{task_id}；原因：{reason}。",
        }
    if stale_seconds >= thresholds.stale_seconds:
        return {
            "type": "stale-state",
            "key": f"stale-state:{signature}",
            "title": "AI Team 狀態太久沒更新",
            "message": f"狀態已 {stale_seconds // 60} 分鐘未更新。任務：{task_id}。",
        }
    return None


def _select_lifecycle_alert(
    supervisor: dict[str, Any],
    previous: dict[str, Any],
) -> dict[str, str] | None:
    deferred = supervisor.get("deferredTasks")
    deferred_count = _bounded_list_count(deferred)
    if deferred_count > _nonnegative_int(previous.get("lastDeferredTaskCount")):
        item = deferred[-1] if isinstance(deferred, list) and deferred else {}
        task_id = _safe_label(item.get("id") if isinstance(item, dict) else None, "未知任務")
        reason = _safe_label(
            item.get("reason") if isinstance(item, dict) else None,
            "已達自動修復上限",
        )
        return {
            "type": "task-deferred",
            "key": f"task-deferred:{task_id}",
            "title": "AI Team 有任務修不好，已先跳過",
            "message": f"任務：{task_id}；原因：{reason}。不阻塞其他開發，報告已保留。",
        }

    release_reviews = supervisor.get("releaseReviewTasks")
    release_count = _bounded_list_count(release_reviews)
    if release_count > _nonnegative_int(previous.get("lastReleaseReviewTaskCount")):
        item = release_reviews[-1] if isinstance(release_reviews, list) and release_reviews else {}
        task_id = _safe_label(item.get("id") if isinstance(item, dict) else None, "未知任務")
        reason = _safe_label(
            item.get("reason") if isinstance(item, dict) else None,
            "需要人工上線驗收",
        )
        return {
            "type": "release-review-required",
            "key": f"release-review-required:{task_id}",
            "title": "AI Team 有項目等待你驗收",
            "message": f"任務：{task_id}；{reason}。測試站開發仍會繼續。",
        }

    backoff = supervisor.get("providerBackoff")
    fingerprint = _provider_backoff_fingerprint(supervisor)
    if fingerprint and fingerprint != previous.get("lastProviderBackoffKey"):
        task = supervisor.get("currentTask")
        task_id = _safe_label(task.get("id") if isinstance(task, dict) else None, "未知任務")
        stage = _safe_label(backoff.get("stage") if isinstance(backoff, dict) else None, "未知階段")
        delay = _nonnegative_int(backoff.get("delaySeconds") if isinstance(backoff, dict) else 0)
        return {
            "type": "provider-backoff",
            "key": f"provider-backoff:{task_id}:{stage}",
            "title": "AI Team 模型供應商暫時不可用",
            "message": (
                f"任務：{task_id}；階段：{stage}。"
                f"系統將在約 {max(1, delay // 60)} 分鐘後自動重試。"
            ),
        }
    return None


def _bounded_list_count(value: Any) -> int:
    return min(len(value), 256) if isinstance(value, list) else 0


def _observed_list_count(
    value: Any,
    previous_value: Any,
    alert_type: str | None,
    observed_alert_type: str,
) -> int:
    current = _bounded_list_count(value)
    previous = _nonnegative_int(previous_value)
    repair_deferred_observed = (
        observed_alert_type == "task-deferred"
        and alert_type == "auto-repair-deferred"
    )
    if current <= previous or alert_type == observed_alert_type or repair_deferred_observed:
        return current
    return previous


def _provider_backoff_fingerprint(supervisor: dict[str, Any]) -> str | None:
    value = supervisor.get("providerBackoff")
    if not isinstance(value, dict) or not value:
        return None
    fields = (
        value.get("taskSha"),
        value.get("stage"),
        value.get("stopReason"),
        value.get("consecutiveFailures"),
        value.get("nextRetryAt"),
    )
    return "|".join(str(item or "") for item in fields)[:500]


def _observed_provider_backoff(
    supervisor: dict[str, Any],
    previous_value: Any,
    alert_type: str | None,
) -> str | None:
    current = _provider_backoff_fingerprint(supervisor)
    previous = previous_value if isinstance(previous_value, str) else None
    if current is None or current == previous or alert_type == "provider-backoff":
        return current
    return previous


def _state_signature(supervisor: dict[str, Any]) -> str:
    task = supervisor.get("currentTask")
    task_sha = task.get("taskSha") if isinstance(task, dict) else None
    values = (
        task_sha,
        supervisor.get("status"),
        supervisor.get("stopReason"),
        supervisor.get("nextAction"),
    )
    return "|".join(str(value or "") for value in values)


def _idle_stall_signature(supervisor: dict[str, Any]) -> str | None:
    """Identify an orphaned PM cache while the heartbeat remains healthy."""

    if (
        supervisor.get("status") != "idle"
        or _nonnegative_int(supervisor.get("queueSize")) != 0
        or supervisor.get("currentTask") is not None
        or supervisor.get("nextAction") != "watch-contract-directory"
    ):
        return None
    backlog = supervisor.get("autonomousBacklog")
    if not isinstance(backlog, dict):
        return None
    task_sha = backlog.get("taskSha")
    revision = backlog.get("projectRevision")
    if (
        backlog.get("status") != "unchanged"
        or backlog.get("outcome") != "task-created"
        or not isinstance(task_sha, str)
        or not task_sha
        or not isinstance(revision, str)
        or not revision
    ):
        return None
    return f"{revision[:40]}:{task_sha[:64]}"


def _needs_attention(supervisor: dict[str, Any]) -> bool:
    return (
        supervisor.get("status") == "attention-required"
        or requires_manual_review(supervisor)
    )


def _state_age_seconds(supervisor: dict[str, Any], now: datetime) -> int:
    value = supervisor.get("updatedAt")
    if not isinstance(value, str):
        return 0
    try:
        updated_at = _as_utc(datetime.fromisoformat(value))
    except ValueError:
        return 0
    return max(0, int((now - updated_at).total_seconds()))


def _task_progress_evidence(
    report_dir: Path | None,
    supervisor: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    """Return only bounded-delivery progress bound to the current task SHA.

    The heartbeat is a structured state transition written by bounded delivery
    itself.  A process merely existing, or writing arbitrary stdout, never
    counts as progress.
    """

    empty = {"updatedAt": None, "ageSeconds": None, "stage": None}
    task = supervisor.get("currentTask")
    if not isinstance(task, dict):
        return empty
    task_sha = task.get("taskSha")
    if not isinstance(task_sha, str) or re.fullmatch(r"[0-9a-f]{64}", task_sha) is None:
        return empty
    state_path = _current_task_state_path(report_dir, task_sha)
    if state_path is None:
        return empty
    try:
        state = _read_json(state_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return empty
    stage = state.get("stage")
    if (
        state.get("schemaVersion") != 1
        or state.get("taskSha") != task_sha
        or state.get("status") != "running"
        or stage not in {"pm", "architect", "engineer", "qa", "review"}
    ):
        return empty
    value = state.get("updatedAt")
    if not isinstance(value, str):
        return empty
    try:
        updated_at = _as_utc(datetime.fromisoformat(value))
    except ValueError:
        return empty
    # A damaged or manually forged future timestamp must not suppress stale
    # recovery indefinitely.  Five minutes only covers ordinary clock skew.
    if (updated_at - now).total_seconds() > 5 * 60:
        return empty
    return {
        "updatedAt": updated_at.isoformat(),
        "ageSeconds": max(0, int((now - updated_at).total_seconds())),
        "stage": stage,
    }


def _current_task_state_path(report_dir: Path | None, task_sha: str) -> Path | None:
    if report_dir is None or re.fullmatch(r"[0-9a-f]{64}", task_sha) is None:
        return None
    candidates: list[Path] = []
    for tasks_root in (
        report_dir / "continuous-bounded-delivery" / "tasks",
        report_dir / "tasks",
    ):
        if not tasks_root.is_dir() or tasks_root.is_symlink():
            continue
        try:
            resolved_root = tasks_root.resolve(strict=True)
        except OSError:
            continue
        for state_path in tasks_root.glob(f"*-{task_sha[:12]}/state.json"):
            if state_path.is_symlink() or state_path.parent.is_symlink():
                continue
            try:
                resolved = state_path.resolve(strict=True)
                resolved.relative_to(resolved_root)
            except (OSError, ValueError):
                continue
            if resolved.is_file() and resolved.stat().st_size <= 4_000_000:
                candidates.append(resolved)
    return candidates[0] if len(candidates) == 1 else None


def _task_failure_evidence(report_dir: Path | None, supervisor: dict[str, Any]) -> dict[str, Any]:
    empty = {"reason": None, "count": 0, "receiptCount": 0}
    task = supervisor.get("currentTask")
    if report_dir is None or not isinstance(task, dict):
        return empty
    task_sha = task.get("taskSha")
    if not isinstance(task_sha, str) or len(task_sha) < 12:
        return empty

    task_state_path = _current_task_state_path(report_dir, task_sha)
    if task_state_path is None:
        return empty
    task_root = task_state_path.parent.resolve()
    try:
        task_state = _read_json(task_state_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return empty
    receipts = task_state.get("receipts")
    if not isinstance(receipts, list):
        return empty

    receipt_reasons: list[str | None] = []
    receipt_count = 0
    # Bound filesystem work even if a damaged state file contains an oversized list.
    for value in receipts[-200:]:
        if not isinstance(value, str) or not value.endswith("-engineer.json"):
            continue
        receipt_path = Path(value)
        if receipt_path.is_symlink():
            continue
        try:
            resolved = receipt_path.resolve()
            resolved.relative_to(task_root)
        except (OSError, ValueError):
            continue
        if not resolved.is_file() or resolved.stat().st_size > 1_000_000:
            continue
        try:
            receipt = _read_json(resolved)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        receipt_count += 1
        validation = receipt.get("validationResult")
        reason = receipt.get("stopReason")
        if not isinstance(reason, str) and isinstance(validation, dict):
            reason = validation.get("stopReason")
        receipt_reasons.append(reason.strip() if isinstance(reason, str) and reason.strip() else None)

    # Only alert on the latest uninterrupted run of the same failure. A
    # successful receipt proves that the task has recovered and must clear old
    # failures, otherwise a repaired task would keep raising stale alerts.
    if not receipt_reasons or receipt_reasons[-1] is None:
        return {"reason": None, "count": 0, "receiptCount": receipt_count}
    reason = receipt_reasons[-1]
    count = 0
    for candidate in reversed(receipt_reasons):
        if candidate != reason:
            break
        count += 1
    return {"reason": reason, "count": count, "receiptCount": receipt_count}


def _service_status(service_name: str, runner: Runner) -> dict[str, str]:
    completed = runner(
        [
            "systemctl",
            "--user",
            "show",
            service_name,
            "--property=ActiveState",
            "--property=SubState",
            "--property=NRestarts",
            "--no-pager",
        ],
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    if completed.returncode != 0:
        return {"ActiveState": "unknown", "SubState": "unknown", "NRestarts": "0"}
    result = {"ActiveState": "unknown", "SubState": "unknown", "NRestarts": "0"}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in result:
            result[key] = value.strip()
    return result


def _within_cooldown(previous: dict[str, Any], key: str, now: datetime, cooldown_seconds: int) -> bool:
    if previous.get("lastAlertKey") != key:
        return False
    value = previous.get("lastAlertAt")
    if not isinstance(value, str):
        return False
    try:
        last_alert = _as_utc(datetime.fromisoformat(value))
    except ValueError:
        return False
    return (now - last_alert).total_seconds() < cooldown_seconds


def _append_alert_log(path: Path, now: datetime, alert: dict[str, str], delivered: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {
            "at": now.isoformat(),
            "type": alert["type"],
            "title": alert["title"],
            "message": alert["message"],
            "notificationDelivered": delivered,
        },
        ensure_ascii=False,
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    os.chmod(path, 0o600)


def _read_json(path: Path, *, missing_ok: bool = False) -> dict[str, Any]:
    if missing_ok and not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _validate_thresholds(thresholds: WatchdogThresholds) -> None:
    if min(
        thresholds.repeat_count,
        thresholds.restart_count,
        thresholds.idle_count,
        thresholds.stale_seconds,
        thresholds.cooldown_seconds,
        thresholds.repair_restart_grace_seconds,
    ) <= 0:
        raise ValueError("watchdog thresholds must be positive")


def _validate_auto_repair(options: AutoRepairOptions) -> None:
    if options.max_attempts <= 0:
        raise ValueError("watchdog auto repair attempts must be positive")
    if options.enabled and any(
        path is None for path in (options.project_path, options.contract_dir, options.backup_dir)
    ):
        raise ValueError("watchdog auto repair requires project, contract, and backup paths")
    if options.ai_repair_enabled and any(
        path is None
        for path in (options.project_path, options.orchestrator_path, options.ai_report_dir)
    ):
        raise ValueError(
            "watchdog AI repair requires project, orchestrator, and report paths"
        )
    if options.ai_repair_enabled and not options.enabled:
        raise ValueError("watchdog AI repair requires automatic repair to be enabled")
    if not 1 <= options.max_ai_repair_cycles <= 5:
        raise ValueError("watchdog AI repair cycles must be between 1 and 5")


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(str(value)))
    except (TypeError, ValueError):
        return 0


def _safe_label(value: Any, fallback: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return fallback
    return value.strip().replace("\n", " ")[:160]


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("watchdog timestamps must be timezone-aware")
    return value.astimezone(UTC)
