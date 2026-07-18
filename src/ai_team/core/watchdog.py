from __future__ import annotations

import base64
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from ai_team.core.watchdog_repair import AutoRepairOptions, attempt_auto_repair, repair_key


@dataclass(frozen=True)
class WatchdogThresholds:
    repeat_count: int = 3
    restart_count: int = 3
    stale_seconds: int = 25 * 60
    cooldown_seconds: int = 30 * 60


Runner = Callable[..., subprocess.CompletedProcess[str]]
Notifier = Callable[[str, str], bool]


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
    same_signature_count = (
        int(previous.get("sameSignatureCount") or 0) + 1
        if attention and previous.get("signature") == signature
        else (1 if attention else 0)
    )

    restart_total = _nonnegative_int(service.get("NRestarts"))
    previous_restart_total = _nonnegative_int(previous.get("lastRestartCount"))
    restart_delta = max(0, restart_total - previous_restart_total) if previous else 0
    stale_seconds = _state_age_seconds(supervisor, checked_at)
    task_failure = _task_failure_evidence(report_dir, supervisor)

    alert = _select_alert(
        supervisor,
        service,
        signature,
        same_signature_count,
        restart_delta,
        stale_seconds,
        task_failure,
        thresholds,
    )
    current_repair_key = repair_key(supervisor, task_failure["reason"])
    previous_repair_attempts = (
        _nonnegative_int(previous.get("repairAttempts"))
        if previous.get("repairKey") == current_repair_key
        else 0
    )
    repair: dict[str, Any] | None = None
    repair_attempts = previous_repair_attempts
    if alert and auto_repair.enabled:
        if previous_repair_attempts < auto_repair.max_attempts:
            repair_attempts += 1
            repair = attempt_auto_repair(
                supervisor,
                alert_type=alert["type"],
                service_name=service_name,
                options=auto_repair,
                runner=runner,
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
    elif not _repair_condition_present(supervisor, task_failure, restart_delta):
        current_repair_key = None
        repair_attempts = 0
    delivered = False
    suppressed = False
    if alert:
        suppressed = _within_cooldown(previous, alert["key"], checked_at, thresholds.cooldown_seconds)
        if not suppressed:
            send = notifier or (
                lambda title, message: send_windows_toast(
                    title,
                    message,
                    powershell_path=powershell_path,
                    runner=runner,
                )
            )
            delivered = send(alert["title"], alert["message"])
            _append_alert_log(alert_log_path, checked_at, alert, delivered)

    state = {
        "schemaVersion": 1,
        "updatedAt": checked_at.isoformat(),
        "signature": signature,
        "sameSignatureCount": same_signature_count,
        "lastSupervisorUpdatedAt": supervisor.get("updatedAt"),
        "lastRestartCount": restart_total,
        "lastTaskFailureReason": task_failure["reason"],
        "taskFailureCount": task_failure["count"],
        "taskReceiptCount": task_failure["receiptCount"],
        "autoRepairEnabled": auto_repair.enabled,
        "repairKey": current_repair_key,
        "repairAttempts": repair_attempts,
        "lastRepairAt": (
            checked_at.isoformat()
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
            checked_at.isoformat()
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
        "sameSignatureCount": same_signature_count,
        "restartDelta": restart_delta,
        "staleSeconds": stale_seconds,
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


def _repair_alert(
    original: dict[str, str],
    repair: dict[str, Any],
    attempts: int,
    maximum: int,
) -> dict[str, str]:
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


def _select_alert(
    supervisor: dict[str, Any],
    service: dict[str, str],
    signature: str,
    same_signature_count: int,
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


def _needs_attention(supervisor: dict[str, Any]) -> bool:
    return (
        supervisor.get("status") == "attention-required"
        or supervisor.get("nextAction") == "manual-review-required"
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


def _task_failure_evidence(report_dir: Path | None, supervisor: dict[str, Any]) -> dict[str, Any]:
    empty = {"reason": None, "count": 0, "receiptCount": 0}
    task = supervisor.get("currentTask")
    if report_dir is None or not isinstance(task, dict):
        return empty
    task_sha = task.get("taskSha")
    if not isinstance(task_sha, str) or len(task_sha) < 12:
        return empty

    task_state_paths: list[Path] = []
    for tasks_root in (
        report_dir / "continuous-bounded-delivery" / "tasks",
        report_dir / "tasks",
    ):
        if tasks_root.is_dir():
            task_state_paths.extend(tasks_root.glob(f"*-{task_sha[:12]}/state.json"))
    if len(task_state_paths) != 1:
        return empty

    task_state_path = task_state_paths[0]
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
        thresholds.stale_seconds,
        thresholds.cooldown_seconds,
    ) <= 0:
        raise ValueError("watchdog thresholds must be positive")


def _validate_auto_repair(options: AutoRepairOptions) -> None:
    if options.max_attempts <= 0:
        raise ValueError("watchdog auto repair attempts must be positive")
    if options.enabled and any(
        path is None for path in (options.project_path, options.contract_dir, options.backup_dir)
    ):
        raise ValueError("watchdog auto repair requires project, contract, and backup paths")


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
