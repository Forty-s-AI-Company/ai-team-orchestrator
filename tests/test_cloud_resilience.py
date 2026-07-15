from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ai_team.core.cloud_resilience import (
    CloudRecoveryState,
    LocalContinuitySettings,
    RetrySettings,
    _run_local_recorder,
    classify_failure,
    create_resume_packet,
    default_engineer_routes,
)


class CloudResilienceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 7, 15, tzinfo=UTC)
        self.settings = RetrySettings(
            max_attempts_per_model=2,
            initial_delay_seconds=60,
            multiplier=2,
            max_delay_seconds=1800,
            jitter_ratio=0,
            max_task_provider_attempts=8,
            circuit_failure_threshold=2,
            circuit_cooldown_seconds=60,
            circuit_max_cooldown_seconds=3600,
            probe_interval_seconds=60,
        )

    def test_error_classification_keeps_hard_quota_and_code_failures_distinct(self) -> None:
        self.assertEqual(classify_failure("provider-quota-exhausted"), "transient_provider_error")
        self.assertEqual(classify_failure("provider failure", error_summary="HTTP 403 forbidden"), "account_or_hard_quota_error")
        self.assertEqual(classify_failure("deterministic-validation-failed"), "task_or_code_failure")
        self.assertEqual(classify_failure("worktree creation failed"), "infrastructure_failure")

    def test_terra_sol_luna_each_have_independent_circuit_and_cloud_wait(self) -> None:
        recovery = CloudRecoveryState(
            task_sha="a" * 64,
            stage="engineer",
            routes=default_engineer_routes(),
            settings=self.settings,
        )
        terra, sol, luna = default_engineer_routes()

        self.assertEqual(recovery.current_route(), terra)
        first = recovery.record_failure(terra, reason="provider-rate-limit", now=self.now)
        self.assertEqual(first["status"], "retry_backoff")
        second = recovery.record_failure(terra, reason="provider-rate-limit", now=self.now + timedelta(seconds=60))
        self.assertEqual(second["status"], "provider_fallback")
        self.assertEqual(recovery.current_route(), sol)
        recovery.record_failure(sol, reason="provider-capacity-unavailable", now=self.now + timedelta(seconds=120))
        third = recovery.record_failure(sol, reason="provider-capacity-unavailable", now=self.now + timedelta(seconds=180))
        self.assertEqual(third["status"], "provider_fallback")
        self.assertEqual(recovery.current_route(), luna)
        recovery.record_failure(luna, reason="provider-timeout", now=self.now + timedelta(seconds=240))
        waiting = recovery.record_failure(luna, reason="provider-timeout", now=self.now + timedelta(seconds=300))
        self.assertEqual(waiting["status"], "cloud_waiting")
        self.assertEqual(recovery.as_dict()["circuits"][terra.key]["circuitState"], "open")
        self.assertEqual(recovery.as_dict()["circuits"][sol.key]["circuitState"], "open")
        self.assertEqual(recovery.as_dict()["circuits"][luna.key]["circuitState"], "open")

    def test_preferred_terra_is_probed_and_resumed_after_cooldown(self) -> None:
        recovery = CloudRecoveryState(
            task_sha="b" * 64,
            stage="engineer",
            routes=default_engineer_routes(),
            settings=self.settings,
        )
        terra = default_engineer_routes()[0]
        recovery.record_failure(terra, reason="provider-rate-limit", now=self.now)
        recovery.record_failure(terra, reason="provider-rate-limit", now=self.now + timedelta(seconds=60))
        action, route, _ = recovery.next_action(self.now + timedelta(seconds=121))
        self.assertEqual(action, "probe")
        self.assertEqual(route, terra)
        recovery.record_probe(terra, success=True, now=self.now + timedelta(seconds=121))
        action, route, _ = recovery.next_action(self.now + timedelta(seconds=122))
        self.assertEqual(action, "delivery")
        self.assertEqual(route, terra)

    def test_provider_probe_budget_is_bounded_per_hour(self) -> None:
        settings = RetrySettings(max_provider_probes_per_hour=1)
        recovery = CloudRecoveryState(
            task_sha="e" * 64,
            stage="engineer",
            routes=default_engineer_routes(),
            settings=settings,
        )
        terra = default_engineer_routes()[0]
        recovery.record_probe(terra, success=False, now=self.now)
        self.assertFalse(recovery.probe_allowed(self.now + timedelta(minutes=1)))
        self.assertEqual(recovery.next_probe_budget_at(self.now + timedelta(minutes=1)), self.now + timedelta(hours=1))

    def test_task_attempt_budget_counts_the_current_failure(self) -> None:
        settings = RetrySettings(max_attempts_per_model=1, max_task_provider_attempts=1, jitter_ratio=0)
        recovery = CloudRecoveryState(
            task_sha="f" * 64,
            stage="engineer",
            routes=default_engineer_routes(),
            settings=settings,
        )
        result = recovery.record_failure(
            default_engineer_routes()[0], reason="provider-rate-limit", now=self.now
        )
        self.assertEqual(result["status"], "cloud_waiting")

    def test_local_recorder_sandbox_does_not_mount_root_or_home(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executable = root / "recorder"
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            settings = LocalContinuitySettings(command=(str(executable),))
            captured: list[str] = []

            def run(command, **_kwargs):
                captured.extend(command)
                return SimpleNamespace(returncode=0, stdout="summary")

            with patch("ai_team.core.cloud_resilience.shutil.which", side_effect=lambda value: "/usr/bin/bwrap" if value == "bwrap" else str(executable)), patch(
                "ai_team.core.cloud_resilience.subprocess.run", side_effect=run
            ):
                result = _run_local_recorder(settings, root, {"taskId": "task"})

            self.assertEqual(result["execution"], "completed")
            self.assertNotIn("/home", captured)
            self.assertNotIn("--ro-bind / /", " ".join(captured))

    def test_deterministic_resume_packet_never_modifies_repository(self) -> None:
        recovery = CloudRecoveryState(
            task_sha="c" * 64,
            stage="engineer",
            routes=default_engineer_routes(),
            settings=self.settings,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            source = project / "source.txt"
            source.write_text("unchanged", encoding="utf-8")
            packet = create_resume_packet(
                state_root=root / "state",
                project_path=project,
                task_id="safe-task",
                task_sha="c" * 64,
                task_title="safe task",
                task_state={"stage": "engineer", "worktreePath": str(project)},
                supervisor_state=recovery,
                receipt_paths=["/state/01-engineer.json"],
                continuity=LocalContinuitySettings(),
                now=self.now,
            )
            self.assertEqual(source.read_text(encoding="utf-8"), "unchanged")
            payload = json.loads(Path(packet["json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["localContinuity"]["repositoryModifications"], "none")
            self.assertTrue(Path(packet["markdown"]).is_file())

    def test_resume_packet_preserves_paths_and_only_marks_successful_stages_completed(self) -> None:
        recovery = CloudRecoveryState(
            task_sha="e" * 64,
            stage="engineer",
            routes=default_engineer_routes(),
            settings=self.settings,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            tracked = project / "tests/e2e/smoke.spec.ts"
            tracked.parent.mkdir(parents=True)
            tracked.write_text("before\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=project, check=True)
            subprocess.run(["git", "config", "user.name", "AI Team Test"], cwd=project, check=True)
            subprocess.run(["git", "config", "user.email", "ai-team@example.invalid"], cwd=project, check=True)
            subprocess.run(["git", "add", "tests/e2e/smoke.spec.ts"], cwd=project, check=True)
            subprocess.run(["git", "commit", "-qm", "test fixture"], cwd=project, check=True)
            tracked.write_text("after\n", encoding="utf-8")
            untracked = project / "src/app/api/security/csp-report/route.test.ts"
            untracked.parent.mkdir(parents=True)
            untracked.write_text("test\n", encoding="utf-8")

            receipts = root / "receipts"
            receipts.mkdir()
            pm = receipts / "01-pm.json"
            architect = receipts / "02-architect.json"
            engineer = receipts / "03-engineer.json"
            pm.write_text(json.dumps({
                "stage": "pm", "providerSuccess": True, "validationResult": {"success": True}, "stopReason": None,
            }), encoding="utf-8")
            architect.write_text(json.dumps({
                "stage": "architect", "providerSuccess": True, "validationResult": {"success": True}, "stopReason": None,
            }), encoding="utf-8")
            engineer.write_text(json.dumps({
                "stage": "engineer", "providerSuccess": False,
                "validationResult": {"success": False, "kind": "provider-execution"},
                "stopReason": "provider-quota-exhausted", "commitSha": None,
            }), encoding="utf-8")

            packet = create_resume_packet(
                state_root=root / "state",
                project_path=project,
                task_id="resume-integrity",
                task_sha="e" * 64,
                task_title="resume integrity",
                task_state={"stage": "engineer", "worktreePath": str(project)},
                supervisor_state=recovery,
                receipt_paths=[str(pm), str(architect), str(engineer)],
                continuity=LocalContinuitySettings(),
                now=self.now,
            )

            payload = json.loads(Path(packet["json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["lastCompletedStage"], "architect")
            self.assertEqual(payload["nextPendingStage"], "engineer")
            self.assertEqual(payload["changedFiles"], [
                "tests/e2e/smoke.spec.ts",
                "src/app/api/security/csp-report/route.test.ts",
            ])
            self.assertEqual(payload["filesForNextEngineer"], payload["changedFiles"])

    def test_resume_packet_redacts_secret_like_values(self) -> None:
        recovery = CloudRecoveryState(
            task_sha="d" * 64,
            stage="engineer",
            routes=default_engineer_routes(),
            settings=self.settings,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet = create_resume_packet(
                state_root=root / "state",
                project_path=root,
                task_id="secret-task",
                task_sha="d" * 64,
                task_title="api_key=not-for-packets",
                task_state={},
                supervisor_state=recovery,
                receipt_paths=[],
                continuity=LocalContinuitySettings(),
                now=self.now,
            )
            text = Path(packet["json"]).read_text(encoding="utf-8")
            self.assertNotIn("not-for-packets", text)
            self.assertIn("<redacted>", text)
