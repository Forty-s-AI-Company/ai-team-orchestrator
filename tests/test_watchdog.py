from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_team.core.watchdog import WatchdogThresholds, run_watchdog


class WatchdogTests(unittest.TestCase):
    def test_repeated_attention_alerts_on_third_identical_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            supervisor = root / "supervisor.json"
            watcher = root / "watchdog.json"
            alerts = root / "alerts.log"
            now = datetime(2026, 7, 18, 10, tzinfo=UTC)
            _write_supervisor(supervisor, now, status="attention-required")
            notifications: list[tuple[str, str]] = []

            for offset in range(3):
                result = run_watchdog(
                    supervisor,
                    watcher,
                    alerts,
                    service_name="example.service",
                    now=now + timedelta(minutes=offset),
                    runner=_service_runner(10),
                    notifier=lambda title, message: notifications.append((title, message)) or True,
                )

            self.assertEqual(result["status"], "alerted")
            self.assertEqual(result["alertType"], "repeated-attention")
            self.assertEqual(len(notifications), 1)
            self.assertIn("連續卡住", notifications[0][0])
            self.assertIn("auto-example-task", notifications[0][1])

    def test_restart_delta_alerts_without_waiting_for_repeated_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            supervisor = root / "supervisor.json"
            watcher = root / "watchdog.json"
            alerts = root / "alerts.log"
            now = datetime(2026, 7, 18, 10, tzinfo=UTC)
            _write_supervisor(supervisor, now, status="running")
            run_watchdog(
                supervisor,
                watcher,
                alerts,
                service_name="example.service",
                now=now,
                runner=_service_runner(20),
                notifier=lambda _title, _message: True,
            )

            messages: list[str] = []
            result = run_watchdog(
                supervisor,
                watcher,
                alerts,
                service_name="example.service",
                now=now + timedelta(minutes=1),
                runner=_service_runner(24),
                notifier=lambda _title, message: messages.append(message) or True,
            )

            self.assertEqual(result["alertType"], "restart-loop")
            self.assertEqual(result["restartDelta"], 4)
            self.assertIn("重啟 4 次", messages[0])

    def test_stale_running_state_alerts_after_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            supervisor = root / "supervisor.json"
            now = datetime(2026, 7, 18, 10, tzinfo=UTC)
            _write_supervisor(supervisor, now - timedelta(minutes=26), status="running")

            result = run_watchdog(
                supervisor,
                root / "watchdog.json",
                root / "alerts.log",
                service_name="example.service",
                now=now,
                runner=_service_runner(5),
                notifier=lambda _title, _message: True,
            )

            self.assertEqual(result["alertType"], "stale-state")
            self.assertEqual(result["staleSeconds"], 26 * 60)

    def test_repeated_engineer_failure_receipts_alert_while_supervisor_is_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            supervisor = root / "supervisor.json"
            reports = root / "reports"
            now = datetime(2026, 7, 18, 10, tzinfo=UTC)
            _write_supervisor(supervisor, now, status="running")
            _write_task_failure_receipts(reports, count=3, reason="git-policy-denied")
            notifications: list[str] = []

            result = run_watchdog(
                supervisor,
                root / "watchdog.json",
                root / "alerts.log",
                service_name="example.service",
                report_dir=reports,
                now=now,
                runner=_service_runner(5),
                notifier=lambda _title, message: notifications.append(message) or True,
            )

            self.assertEqual(result["alertType"], "repeated-task-receipts")
            self.assertEqual(result["taskFailureReason"], "git-policy-denied")
            self.assertEqual(result["taskFailureCount"], 3)
            self.assertIn("3 份相同失敗紀錄", notifications[0])

    def test_successful_engineer_receipt_clears_historical_failure_streak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            supervisor = root / "supervisor.json"
            reports = root / "reports"
            now = datetime(2026, 7, 18, 10, tzinfo=UTC)
            _write_supervisor(supervisor, now, status="running")
            _write_task_failure_receipts(reports, count=4, reason="git-policy-denied")
            _append_successful_engineer_receipt(reports)
            notifications: list[str] = []

            result = run_watchdog(
                supervisor,
                root / "watchdog.json",
                root / "alerts.log",
                service_name="example.service",
                report_dir=reports,
                now=now,
                runner=_service_runner(5),
                notifier=lambda _title, message: notifications.append(message) or True,
            )

            self.assertEqual(result["status"], "ok")
            self.assertIsNone(result["taskFailureReason"])
            self.assertEqual(result["taskFailureCount"], 0)
            self.assertEqual(result["taskReceiptCount"], 5)
            self.assertEqual(notifications, [])

    def test_duplicate_alert_is_suppressed_during_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            supervisor = root / "supervisor.json"
            watcher = root / "watchdog.json"
            alerts = root / "alerts.log"
            now = datetime(2026, 7, 18, 10, tzinfo=UTC)
            _write_supervisor(supervisor, now, status="attention-required")
            notifications: list[str] = []

            for offset in range(4):
                result = run_watchdog(
                    supervisor,
                    watcher,
                    alerts,
                    service_name="example.service",
                    now=now + timedelta(minutes=offset),
                    runner=_service_runner(1),
                    notifier=lambda _title, message: notifications.append(message) or True,
                )

            self.assertEqual(result["status"], "ok")
            self.assertTrue(result["notificationSuppressed"])
            self.assertEqual(len(notifications), 1)
            self.assertEqual(len(alerts.read_text(encoding="utf-8").splitlines()), 1)

    def test_healthy_state_resets_repeated_attention_counter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            supervisor = root / "supervisor.json"
            watcher = root / "watchdog.json"
            now = datetime(2026, 7, 18, 10, tzinfo=UTC)
            _write_supervisor(supervisor, now, status="attention-required")
            for offset in range(2):
                run_watchdog(
                    supervisor,
                    watcher,
                    root / "alerts.log",
                    service_name="example.service",
                    now=now + timedelta(minutes=offset),
                    runner=_service_runner(1),
                    notifier=lambda _title, _message: True,
                )

            _write_supervisor(supervisor, now + timedelta(minutes=2), status="running")
            result = run_watchdog(
                supervisor,
                watcher,
                root / "alerts.log",
                service_name="example.service",
                now=now + timedelta(minutes=2),
                runner=_service_runner(1),
                notifier=lambda _title, _message: True,
            )

            self.assertEqual(result["sameSignatureCount"], 0)
            self.assertEqual(result["status"], "ok")

    def test_thresholds_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            supervisor = root / "supervisor.json"
            _write_supervisor(supervisor, datetime.now(UTC), status="running")
            with self.assertRaisesRegex(ValueError, "thresholds must be positive"):
                run_watchdog(
                    supervisor,
                    root / "watchdog.json",
                    root / "alerts.log",
                    service_name="example.service",
                    thresholds=WatchdogThresholds(repeat_count=0),
                    runner=_service_runner(0),
                )


def _write_supervisor(path: Path, updated_at: datetime, *, status: str) -> None:
    payload = {
        "updatedAt": updated_at.isoformat(),
        "status": status,
        "currentTask": {"id": "auto-example-task", "taskSha": "a" * 64},
        "stopReason": "publication-exception" if status == "attention-required" else None,
        "nextAction": "manual-review-required" if status == "attention-required" else "bounded-delivery",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_task_failure_receipts(report_dir: Path, *, count: int, reason: str) -> None:
    task_root = (
        report_dir
        / "continuous-bounded-delivery"
        / "tasks"
        / f"auto-example-task-{'a' * 12}"
    )
    receipts_dir = task_root / "receipts"
    receipts_dir.mkdir(parents=True)
    receipts: list[str] = []
    for number in range(1, count + 1):
        receipt = receipts_dir / f"{number:02d}-engineer.json"
        receipt.write_text(json.dumps({"stopReason": reason}), encoding="utf-8")
        receipts.append(str(receipt))
    (task_root / "state.json").write_text(json.dumps({"receipts": receipts}), encoding="utf-8")


def _append_successful_engineer_receipt(report_dir: Path) -> None:
    task_root = (
        report_dir
        / "continuous-bounded-delivery"
        / "tasks"
        / f"auto-example-task-{'a' * 12}"
    )
    state_path = task_root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    receipt = task_root / "receipts" / "99-engineer.json"
    receipt.write_text(json.dumps({"validationResult": {"success": True}}), encoding="utf-8")
    state["receipts"].append(str(receipt))
    state_path.write_text(json.dumps(state), encoding="utf-8")


def _service_runner(restarts: int):
    def run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=f"NRestarts={restarts}\nActiveState=active\nSubState=running\n",
            stderr="",
        )

    return run


if __name__ == "__main__":
    unittest.main()
