from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ai_team.core.watchdog import WatchdogThresholds, run_watchdog
from ai_team.core.watchdog_repair import (
    AutoRepairOptions,
    CONTRACT_COMMAND_DIAGNOSTIC,
    attempt_auto_repair,
)


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

    def test_successful_repair_waits_for_new_supervisor_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            supervisor = root / "supervisor.json"
            watcher = root / "watchdog.json"
            alerts = root / "alerts.log"
            now = datetime(2026, 7, 18, 10, tzinfo=UTC)
            old_heartbeat = now - timedelta(minutes=5)
            _write_supervisor(supervisor, old_heartbeat, status="attention-required")
            watcher.write_text(
                json.dumps(
                    {
                        "signature": (
                            f"{'a' * 64}|attention-required|publication-exception|"
                            "manual-review-required"
                        ),
                        "sameSignatureCount": 3,
                        "lastSupervisorUpdatedAt": old_heartbeat.isoformat(),
                        "lastRestartCount": 5,
                        "repairAttempts": 1,
                        "lastRepairAt": (now - timedelta(seconds=30)).isoformat(),
                        "lastRepair": {
                            "attempted": True,
                            "success": True,
                            "restarted": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            thresholds = WatchdogThresholds(
                repeat_count=1,
                restart_count=1,
                stale_seconds=60,
                cooldown_seconds=60,
                repair_restart_grace_seconds=120,
            )
            notifications: list[str] = []

            waiting = run_watchdog(
                supervisor,
                watcher,
                alerts,
                service_name="example.service",
                now=now,
                runner=_service_runner(9),
                notifier=lambda _title, message: notifications.append(message) or True,
                thresholds=thresholds,
            )

            self.assertEqual(waiting["status"], "ok")
            self.assertTrue(waiting["repairRestartGrace"])
            self.assertEqual(waiting["sameSignatureCount"], 0)
            self.assertIsNone(waiting["alertType"])
            self.assertEqual(notifications, [])

            # A changed heartbeat proves the restarted supervisor has observed
            # the repaired revision. A new attention state is actionable again.
            _write_supervisor(supervisor, now + timedelta(seconds=1), status="attention-required")
            observed = run_watchdog(
                supervisor,
                watcher,
                alerts,
                service_name="example.service",
                now=now + timedelta(seconds=1),
                runner=_service_runner(9),
                notifier=lambda _title, message: notifications.append(message) or True,
                thresholds=thresholds,
            )

            self.assertFalse(observed["repairRestartGrace"])
            self.assertEqual(observed["alertType"], "repeated-attention")
            self.assertEqual(len(notifications), 1)

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

    def test_restart_loop_repairs_contract_commands_and_restarts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _write_project(root)
            contracts = root / "contracts"
            contracts.mkdir()
            contract = contracts / "task.json"
            _write_incomplete_contract(contract)
            supervisor = root / "supervisor.json"
            now = datetime(2026, 7, 18, 10, tzinfo=UTC)
            _write_supervisor(
                supervisor,
                now,
                status="attention-required",
                stop_reason="bounded-delivery-exception",
                diagnostic=CONTRACT_COMMAND_DIAGNOSTIC,
                contract_path=contract,
                next_action="bounded-delivery",
            )
            runner = _RecordingServiceRunner(restarts=7)
            notifications: list[str] = []
            result: dict[str, object] = {}

            for offset in range(3):
                result = run_watchdog(
                    supervisor,
                    root / "watchdog.json",
                    root / "alerts.log",
                    service_name="example.service",
                    now=now + timedelta(minutes=offset),
                    runner=runner,
                    notifier=lambda title, _message: notifications.append(title) or True,
                    auto_repair=AutoRepairOptions(
                        enabled=True,
                        project_path=project,
                        contract_dir=contracts,
                        backup_dir=root / "backups",
                        max_attempts=2,
                    ),
                )

            self.assertEqual(result["status"], "repaired")
            self.assertEqual(result["alertType"], "auto-repair-success")
            self.assertTrue(result["repair"]["restarted"])
            repaired = json.loads(contract.read_text(encoding="utf-8"))
            self.assertEqual(
                repaired["validationCommands"],
                ["npm run lint", "npm run typecheck", "npm run test", "npm run build"],
            )
            self.assertEqual(len(list((root / "backups").glob("*.bak"))), 1)
            self.assertEqual(runner.actions.count("stop"), 1)
            self.assertEqual(runner.actions.count("reset-failed"), 1)
            self.assertEqual(runner.actions.count("start"), 1)
            self.assertEqual(notifications, ["AI Team 已自動修復並重啟"])

    def test_unknown_failure_stops_then_honors_repair_attempt_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _write_project(root)
            contracts = root / "contracts"
            contracts.mkdir()
            supervisor = root / "supervisor.json"
            now = datetime(2026, 7, 18, 10, tzinfo=UTC)
            _write_supervisor(
                supervisor,
                now,
                status="attention-required",
                stop_reason="unknown-failure",
                diagnostic="unknown diagnostic",
                next_action="bounded-delivery",
            )
            runner = _RecordingServiceRunner(restarts=2)
            options = AutoRepairOptions(
                enabled=True,
                project_path=project,
                contract_dir=contracts,
                backup_dir=root / "backups",
                max_attempts=1,
            )
            thresholds = WatchdogThresholds(
                repeat_count=1,
                restart_count=99,
                stale_seconds=3600,
                cooldown_seconds=1800,
            )

            first = run_watchdog(
                supervisor,
                root / "watchdog.json",
                root / "alerts.log",
                service_name="example.service",
                now=now,
                runner=runner,
                notifier=lambda _title, _message: True,
                thresholds=thresholds,
                auto_repair=options,
            )
            second = run_watchdog(
                supervisor,
                root / "watchdog.json",
                root / "alerts.log",
                service_name="example.service",
                now=now + timedelta(minutes=1),
                runner=runner,
                notifier=lambda _title, _message: True,
                thresholds=thresholds,
                auto_repair=options,
            )

            self.assertEqual(first["status"], "repair-failed")
            self.assertEqual(first["repair"]["action"], "unsupported")
            self.assertEqual(second["status"], "repair-exhausted")
            self.assertEqual(second["alertType"], "auto-repair-exhausted")
            self.assertEqual(runner.actions.count("stop"), 1)
            self.assertNotIn("start", runner.actions)

    def test_external_qa_manual_review_alerts_without_auto_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _write_project(root)
            orchestrator = root / "orchestrator"
            orchestrator.mkdir()
            contracts = root / "contracts"
            contracts.mkdir()
            supervisor = root / "supervisor.json"
            watcher = root / "watchdog.json"
            now = datetime(2026, 7, 18, 10, tzinfo=UTC)
            _write_supervisor(
                supervisor,
                now,
                status="attention-required",
                stop_reason="external-qa-failed",
            )
            watcher.write_text(
                json.dumps(
                    {
                        "repairKey": "stale-repair-key",
                        "repairAttempts": 2,
                        "lastRestartCount": 9,
                    }
                ),
                encoding="utf-8",
            )
            runner = _RecordingServiceRunner(restarts=9)
            calls: list[dict[str, object]] = []
            notifications: list[tuple[str, str]] = []

            def ai_repairer(_supervisor, **kwargs):
                calls.append(kwargs)
                return {
                    "attempted": True,
                    "success": True,
                    "action": "codex-sol-terra-qa-repair",
                    "diagnostic": "QA passed",
                    "restarted": False,
                }

            result = run_watchdog(
                supervisor,
                watcher,
                root / "alerts.log",
                service_name="example.service",
                now=now,
                runner=runner,
                notifier=lambda title, message: notifications.append((title, message)) or True,
                thresholds=WatchdogThresholds(
                    repeat_count=1,
                    restart_count=1,
                    stale_seconds=3600,
                    cooldown_seconds=1800,
                ),
                auto_repair=AutoRepairOptions(
                    enabled=True,
                    project_path=project,
                    contract_dir=contracts,
                    backup_dir=root / "backups",
                    ai_repair_enabled=True,
                    orchestrator_path=orchestrator,
                    ai_report_dir=root / "ai-reports",
                    revive_timer_name="example-revive.timer",
                    codex_executable="/usr/local/bin/codex",
                    diagnosis_model="gpt-5.6-sol",
                    repair_model="gpt-5.6-terra",
                    reasoning_effort="high",
                ),
                ai_repairer=ai_repairer,
            )

            self.assertEqual(result["status"], "alerted")
            self.assertEqual(result["alertType"], "repeated-attention")
            self.assertIsNone(result["repair"])
            self.assertEqual(result["repairAttempts"], 0)
            self.assertEqual(calls, [])
            self.assertEqual(runner.actions, ["show"])
            self.assertEqual(len(notifications), 1)
            state = json.loads(watcher.read_text(encoding="utf-8"))
            self.assertIsNone(state["repairKey"])
            self.assertEqual(state["repairAttempts"], 0)

    def test_external_qa_receipt_blocks_repair_through_transient_supervisor_states(self) -> None:
        cases = (
            ("running", "external-qa-human-attestation-required", "bounded-delivery"),
            ("operator-interrupted", "operator-interrupted", "resume-supervisor"),
        )
        for status, stop_reason, next_action in cases:
            with self.subTest(status=status, next_action=next_action), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                project = _write_project(root)
                orchestrator = root / "orchestrator"
                orchestrator.mkdir()
                contracts = root / "contracts"
                contracts.mkdir()
                supervisor = root / "supervisor.json"
                watcher = root / "watchdog.json"
                now = datetime(2026, 7, 18, 10, tzinfo=UTC)
                _write_supervisor(
                    supervisor,
                    now,
                    status=status,
                    stop_reason=stop_reason,
                    next_action=next_action,
                    external_qa=_manual_external_qa(),
                )
                watcher.write_text(
                    json.dumps({"repairKey": "stale-repair-key", "repairAttempts": 2}),
                    encoding="utf-8",
                )
                runner = _RecordingServiceRunner(restarts=0)
                ai_calls: list[dict[str, object]] = []

                result = run_watchdog(
                    supervisor,
                    watcher,
                    root / "alerts.log",
                    service_name="example.service",
                    now=now,
                    runner=runner,
                    notifier=lambda _title, _message: True,
                    thresholds=WatchdogThresholds(
                        repeat_count=1,
                        restart_count=99,
                        stale_seconds=3600,
                        cooldown_seconds=1800,
                    ),
                    auto_repair=AutoRepairOptions(
                        enabled=True,
                        project_path=project,
                        contract_dir=contracts,
                        backup_dir=root / "backups",
                        ai_repair_enabled=True,
                        orchestrator_path=orchestrator,
                        ai_report_dir=root / "ai-reports",
                        revive_timer_name="example-revive.timer",
                    ),
                    ai_repairer=lambda *_args, **kwargs: ai_calls.append(kwargs) or {
                        "attempted": True,
                        "success": True,
                        "action": "codex-sol-terra-qa-repair",
                        "diagnostic": "unexpected repair",
                        "restarted": False,
                    },
                )

                self.assertEqual(result["status"], "alerted")
                self.assertEqual(result["alertType"], "repeated-attention")
                self.assertIsNone(result["repair"])
                self.assertEqual(result["repairAttempts"], 0)
                self.assertEqual(ai_calls, [])
                self.assertEqual(runner.actions, ["show"])
                state = json.loads(watcher.read_text(encoding="utf-8"))
                self.assertIsNone(state["repairKey"])
                self.assertEqual(state["repairAttempts"], 0)

    def test_malformed_external_qa_does_not_block_auto_repair(self) -> None:
        malformed_receipts = (
            {key: value for key, value in _manual_external_qa().items() if key != "reason"},
            {**_manual_external_qa(), "executionAttempted": 0},
        )
        for external_qa in malformed_receipts:
            with self.subTest(external_qa=external_qa):
                runner = _RecordingServiceRunner(restarts=0)
                ai_calls: list[dict[str, object]] = []

                result = attempt_auto_repair(
                    {"nextAction": "bounded-delivery", "externalQa": external_qa},
                    alert_type="repeated-attention",
                    service_name="example.service",
                    options=AutoRepairOptions(
                        enabled=True,
                        ai_repair_enabled=True,
                        project_path=Path("project"),
                        orchestrator_path=Path("orchestrator"),
                        ai_report_dir=Path("reports"),
                    ),
                    runner=runner,
                    ai_repairer=lambda *_args, **kwargs: ai_calls.append(kwargs) or {
                        "attempted": True,
                        "success": True,
                        "action": "codex-sol-terra-qa-repair",
                        "diagnostic": "repair passed",
                        "restarted": False,
                    },
                )

                self.assertTrue(result["success"])
                self.assertEqual(len(ai_calls), 1)
                self.assertEqual(runner.actions, ["stop", "reset-failed", "start"])

    def test_failed_ai_repair_keeps_supervisor_and_revive_timer_stopped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _write_project(root)
            orchestrator = root / "orchestrator"
            orchestrator.mkdir()
            contracts = root / "contracts"
            contracts.mkdir()
            supervisor = root / "supervisor.json"
            now = datetime(2026, 7, 18, 10, tzinfo=UTC)
            _write_supervisor(
                supervisor,
                now,
                status="attention-required",
                next_action="bounded-delivery",
            )
            runner = _RecordingServiceRunner(restarts=9)

            result = run_watchdog(
                supervisor,
                root / "watchdog.json",
                root / "alerts.log",
                service_name="example.service",
                now=now,
                runner=runner,
                notifier=lambda _title, _message: True,
                thresholds=WatchdogThresholds(
                    repeat_count=1,
                    restart_count=1,
                    stale_seconds=3600,
                    cooldown_seconds=1800,
                ),
                auto_repair=AutoRepairOptions(
                    enabled=True,
                    project_path=project,
                    contract_dir=contracts,
                    backup_dir=root / "backups",
                    ai_repair_enabled=True,
                    orchestrator_path=orchestrator,
                    ai_report_dir=root / "ai-reports",
                    revive_timer_name="example-revive.timer",
                ),
                ai_repairer=lambda *_args, **_kwargs: {
                    "attempted": True,
                    "success": False,
                    "action": "codex-sol-terra-qa-repair",
                    "diagnostic": "QA rejected repair",
                    "restarted": False,
                },
            )

            self.assertEqual(result["status"], "repair-failed")
            self.assertEqual(
                runner.calls[-2:],
                [("stop", "example-revive.timer"), ("stop", "example.service")],
            )
            self.assertNotIn("start", runner.actions)

    def test_failed_service_gets_one_controlled_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _write_project(root)
            contracts = root / "contracts"
            contracts.mkdir()
            supervisor = root / "supervisor.json"
            now = datetime(2026, 7, 18, 10, tzinfo=UTC)
            _write_supervisor(supervisor, now, status="running")
            runner = _RecordingServiceRunner(restarts=1, active_state="failed", sub_state="failed")

            result = run_watchdog(
                supervisor,
                root / "watchdog.json",
                root / "alerts.log",
                service_name="example.service",
                now=now,
                runner=runner,
                notifier=lambda _title, _message: True,
                auto_repair=AutoRepairOptions(
                    enabled=True,
                    project_path=project,
                    contract_dir=contracts,
                    backup_dir=root / "backups",
                ),
            )

            self.assertEqual(result["status"], "repaired")
            self.assertEqual(result["repair"]["action"], "controlled-restart")
            self.assertEqual(runner.actions[-3:], ["stop", "reset-failed", "start"])

    def test_failed_contract_repair_restores_original_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = _write_project(root)
            contracts = root / "contracts"
            contracts.mkdir()
            contract = contracts / "task.json"
            _write_incomplete_contract(contract)
            payload = json.loads(contract.read_text(encoding="utf-8"))
            payload["allowedWritePaths"] = ["../outside.ts"]
            contract.write_text(json.dumps(payload), encoding="utf-8")
            original = contract.read_bytes()
            supervisor = root / "supervisor.json"
            now = datetime(2026, 7, 18, 10, tzinfo=UTC)
            _write_supervisor(
                supervisor,
                now,
                status="attention-required",
                stop_reason="bounded-delivery-exception",
                diagnostic=CONTRACT_COMMAND_DIAGNOSTIC,
                contract_path=contract,
                next_action="bounded-delivery",
            )
            runner = _RecordingServiceRunner(restarts=1)

            result = run_watchdog(
                supervisor,
                root / "watchdog.json",
                root / "alerts.log",
                service_name="example.service",
                now=now,
                runner=runner,
                notifier=lambda _title, _message: True,
                thresholds=WatchdogThresholds(
                    repeat_count=1,
                    restart_count=99,
                    stale_seconds=3600,
                    cooldown_seconds=1800,
                ),
                auto_repair=AutoRepairOptions(
                    enabled=True,
                    project_path=project,
                    contract_dir=contracts,
                    backup_dir=root / "backups",
                ),
            )

            self.assertEqual(result["status"], "repair-failed")
            self.assertEqual(contract.read_bytes(), original)
            self.assertEqual(runner.actions.count("stop"), 1)
            self.assertNotIn("start", runner.actions)

    def test_manual_review_direct_auto_repair_is_fail_closed_before_systemctl(self) -> None:
        runner = _RecordingServiceRunner(restarts=0)
        ai_calls: list[dict[str, object]] = []

        result = attempt_auto_repair(
            {"nextAction": "manual-review-required"},
            alert_type="repeated-attention",
            service_name="example.service",
            options=AutoRepairOptions(
                enabled=True,
                ai_repair_enabled=True,
                project_path=Path("project"),
                orchestrator_path=Path("orchestrator"),
                ai_report_dir=Path("reports"),
                revive_timer_name="example-revive.timer",
            ),
            runner=runner,
            ai_repairer=lambda *_args, **kwargs: ai_calls.append(kwargs) or {},
        )

        self.assertEqual(
            result,
            {
                "attempted": False,
                "success": False,
                "action": "manual-review-required",
                "diagnostic": "manual review is required; automatic repair is not permitted",
                "restarted": False,
            },
        )
        self.assertEqual(ai_calls, [])
        self.assertEqual(runner.calls, [])


def _write_supervisor(
    path: Path,
    updated_at: datetime,
    *,
    status: str,
    stop_reason: str | None = None,
    diagnostic: str | None = None,
    contract_path: Path | None = None,
    next_action: str | None = None,
    external_qa: dict[str, object] | None = None,
) -> None:
    payload = {
        "updatedAt": updated_at.isoformat(),
        "status": status,
        "currentTask": {
            "id": "auto-example-task",
            "taskSha": "a" * 64,
            **({"contractPath": str(contract_path)} if contract_path else {}),
        },
        "stopReason": (
            stop_reason
            if stop_reason is not None
            else ("publication-exception" if status == "attention-required" else None)
        ),
        "nextAction": (
            next_action
            if next_action is not None
            else ("manual-review-required" if status == "attention-required" else "bounded-delivery")
        ),
        "diagnostic": diagnostic,
        **({"externalQa": external_qa} if external_qa is not None else {}),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _manual_external_qa() -> dict[str, object]:
    return {
        "schema": "ai-team-external-qa-manual-review/v1",
        "revision": "a" * 40,
        "executionMode": "manual-attestation-only",
        "executionAttempted": False,
        "reviewerRole": "operator",
        "status": "review-required",
        "reason": "human-attestation-required",
    }


def _write_project(root: Path) -> Path:
    project = root / "project"
    project.mkdir()
    (project / ".git").mkdir()
    profile = project / ".ai-team" / "project.yaml"
    profile.parent.mkdir()
    profile.write_text(
        """project:
  name: watchdog-test
  root: "."
  stage: development
commands:
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
""",
        encoding="utf-8",
    )
    return project


def _write_incomplete_contract(path: Path) -> None:
    path.write_text(
        json.dumps({
            "schemaVersion": 1,
            "id": "auto-example-task",
            "title": "Example",
            "source": {"kind": "trusted-contract", "reference": "test"},
            "instruction": "Add safe tests.",
            "allowedWritePaths": ["tests/example.test.ts"],
            "validationCommands": ["npm run lint", "npm run test -- tests/example.test.ts"],
        }),
        encoding="utf-8",
    )


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


class _RecordingServiceRunner:
    def __init__(
        self,
        *,
        restarts: int,
        active_state: str = "active",
        sub_state: str = "running",
    ) -> None:
        self.restarts = restarts
        self.active_state = active_state
        self.sub_state = sub_state
        self.actions: list[str] = []
        self.calls: list[tuple[str, str]] = []

    def __call__(self, args, **_kwargs):
        action = args[2]
        self.actions.append(action)
        self.calls.append((action, args[3]))
        if action == "show":
            stdout = (
                f"NRestarts={self.restarts}\n"
                f"ActiveState={self.active_state}\n"
                f"SubState={self.sub_state}\n"
            )
        else:
            stdout = ""
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=stdout, stderr="")


if __name__ == "__main__":
    unittest.main()
