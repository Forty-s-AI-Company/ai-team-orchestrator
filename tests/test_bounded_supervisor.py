from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ai_team.core.bounded_delivery import load_trusted_task_contract
from ai_team.core.cloud_resilience import LocalContinuitySettings, RetrySettings, default_engineer_routes
from ai_team.core.bounded_supervisor import (
    ContinuousBoundedOptions,
    _sync_primary,
    cleanup_completed_worktree,
    discover_contracts,
    publish_and_merge,
    run_continuous_bounded_delivery,
)
from ai_team.core.trusted_dev import TrustedDevSettings
from ai_team.providers.base import BaseProvider, ProviderRequest, ProviderResult


class ContinuousBoundedSupervisorTests(unittest.TestCase):
    def test_merged_publication_resumes_after_disposable_artifacts_are_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts"
            contracts.mkdir()
            _write_contract(contracts / "001-task.json", "task")
            entry = discover_contracts(contracts)[0]
            result = {
                "status": "completed",
                "taskSha": entry.task_sha,
                "commitSha": "a" * 40,
                "worktreePath": str(root / "already-cleaned-worktree"),
                "runReceipt": str(root / "already-cleaned-receipt.json"),
                "validation": {"hash": "b" * 64},
            }
            primary = SimpleNamespace(
                root=root,
                current_branch="development",
                profile=SimpleNamespace(project=SimpleNamespace(stage="development")),
            )
            options = ContinuousBoundedOptions(
                project_path=root,
                contract_dir=contracts,
                provider_for_role=lambda _role: _NoopProvider(),
                workspace_allowlist=[tmp],
                report_dir=root / "reports",
                state_path=root / "state.json",
                once=False,
                github_execute=True,
                auto_merge=True,
                allow_unreviewed_development_merge=True,
            )

            with (
                patch("ai_team.core.bounded_supervisor.load_project", return_value=primary),
                patch("ai_team.core.bounded_supervisor._repository_name", return_value="example/project"),
                patch(
                    "ai_team.core.bounded_supervisor._find_pull_request",
                    return_value={"state": "MERGED", "url": "https://example.test/pull/1"},
                ),
                patch("ai_team.core.bounded_supervisor._sync_primary", return_value="c" * 40),
            ):
                publication = publish_and_merge(options, entry, result)

            self.assertTrue(publication["success"], publication)
            self.assertTrue(publication["resumedMergedPullRequest"])
            self.assertEqual(publication["prUrl"], "https://example.test/pull/1")

    def test_keyboard_interrupt_persists_resumable_stopped_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts"
            contracts.mkdir()
            state_path = root / "state.json"

            result = run_continuous_bounded_delivery(
                ContinuousBoundedOptions(
                    project_path=root,
                    contract_dir=contracts,
                    provider_for_role=lambda _role: _NoopProvider(),
                    workspace_allowlist=[tmp],
                    report_dir=root / "reports",
                    state_path=state_path,
                    once=False,
                    github_execute=True,
                    auto_merge=True,
                    sleeper=lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt()),
                )
            )

            self.assertEqual(result["status"], "stopped")
            self.assertEqual(result["stopReason"], "operator-interrupted")
            self.assertEqual(result["nextAction"], "resume-supervisor")
            self.assertFalse(state_path.with_suffix(".json.lock").exists())

    def test_cleanup_removes_only_clean_disposable_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "AI Team Test"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
            profile = root / ".ai-team" / "project.yaml"
            profile.parent.mkdir()
            profile.write_text(
                """project:
  name: cleanup-test
  stage: development
repository:
  protected_branches: [main, master]
safety:
  allow_git_push: false
  allow_deploy: false
  allow_database_migration: false
  allow_database_seed: false
  allow_destructive_commands: false
""",
                encoding="utf-8",
            )
            subprocess.run(["git", "add", ".ai-team/project.yaml"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "initial"], cwd=root, check=True)
            worktree = Path(tmp) / "worktree"
            subprocess.run(
                ["git", "worktree", "add", "--detach", str(worktree), "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            contracts = Path(tmp) / "contracts"
            contracts.mkdir()
            options = ContinuousBoundedOptions(
                project_path=root,
                contract_dir=contracts,
                provider_for_role=lambda _role: _NoopProvider(),
                workspace_allowlist=[tmp],
                report_dir=Path(tmp) / "reports",
                state_path=Path(tmp) / "state.json",
                trusted_dev=TrustedDevSettings(cleanup_worktree_after_merge=True),
            )

            result = cleanup_completed_worktree(
                options,
                {"worktreePath": str(worktree)},
            )

            self.assertTrue(result["success"], result)
            self.assertTrue(result["attempted"])
            self.assertFalse(worktree.exists())

    def test_cloud_fallback_creates_checkpoint_then_resumes_preferred_model(self) -> None:
        """A temporary model outage must not permanently block the queue."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts"
            contracts.mkdir()
            contract_path = _write_contract(contracts / "001-task.json", "task")
            task_sha = load_trusted_task_contract(contract_path)[1]
            now = [datetime(2026, 7, 15, tzinfo=UTC)]
            monotonic = [0.0]
            models: list[str] = []
            snapshots: list[dict[str, object]] = []
            finished = [False]
            state_path = root / "state.json"

            def delivery(delivery_options):
                selected = delivery_options.provider_for_role("engineer")
                models.append(selected.route.model)
                if len(models) <= 3:
                    return {
                        "status": "attention-required",
                        "stopReason": "provider-rate-limit",
                        "taskSha": task_sha,
                        "stage": "engineer",
                    }
                finished[0] = True
                return {"status": "completed", "taskSha": task_sha, "commitSha": "commit-task"}

            def sleep(seconds: float) -> None:
                snapshots.append(json.loads(state_path.read_text(encoding="utf-8")))
                now[0] += timedelta(seconds=seconds)
                monotonic[0] += seconds
                if finished[0]:
                    monotonic[0] = 100_000

            result = run_continuous_bounded_delivery(
                ContinuousBoundedOptions(
                    project_path=root,
                    contract_dir=contracts,
                    provider_for_role=lambda _role: _NoopProvider(),
                    workspace_allowlist=[tmp],
                    report_dir=root / "reports",
                    state_path=state_path,
                    once=False,
                    interval_minutes=1,
                    max_runtime_minutes=1_000,
                    github_execute=True,
                    auto_merge=True,
                    delivery_runner=delivery,
                    publisher=lambda _options, _entry, _result: {"success": True},
                    sleeper=sleep,
                    monotonic=lambda: monotonic[0],
                    wall_clock=lambda: now[0],
                    cloud_routes=default_engineer_routes(),
                    cloud_retry=RetrySettings(
                        max_attempts_per_model=1,
                        initial_delay_seconds=1,
                        multiplier=2,
                        max_delay_seconds=10,
                        jitter_ratio=0,
                        max_task_provider_attempts=8,
                        circuit_failure_threshold=1,
                        circuit_cooldown_seconds=10,
                        circuit_max_cooldown_seconds=10,
                        probe_interval_seconds=1,
                    ),
                    local_continuity=LocalContinuitySettings(),
                )
            )

            self.assertEqual(models, ["gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.6-terra"])
            waiting = next(item for item in snapshots if item["status"] == "cloud_waiting")
            self.assertEqual(waiting["stopReason"], "all-cloud-models-temporarily-unavailable")
            packet = waiting["continuity"]["resumePacket"]
            self.assertTrue(Path(packet["json"]).is_file())
            self.assertEqual(result["status"], "idle")

    def test_continuous_mode_runs_ordered_contracts_until_runtime_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts"
            contracts.mkdir()
            first = _write_contract(contracts / "001-first.json", "first")
            second = _write_contract(contracts / "002-second.json", "second")
            executed: list[str] = []
            published: list[str] = []

            def delivery(options):
                contract, task_sha = load_trusted_task_contract(options.task_contract_path)
                executed.append(contract.id)
                return {
                    "status": "completed",
                    "taskSha": task_sha,
                    "commitSha": f"commit-{contract.id}",
                }

            def publisher(options, entry, result):
                published.append(entry.contract.id)
                return {"success": True, "prUrl": f"https://example.test/{entry.contract.id}"}

            times = iter((0.0, 0.0, 61.0))
            result = run_continuous_bounded_delivery(
                ContinuousBoundedOptions(
                    project_path=root,
                    contract_dir=contracts,
                    provider_for_role=lambda _role: _NoopProvider(),
                    workspace_allowlist=[tmp],
                    report_dir=root / "reports",
                    state_path=root / "state.json",
                    once=False,
                    interval_minutes=1,
                    max_runtime_minutes=1,
                    github_execute=True,
                    auto_merge=True,
                    delivery_runner=delivery,
                    publisher=publisher,
                    monotonic=lambda: next(times),
                )
            )

            self.assertEqual(executed, ["first", "second"])
            self.assertEqual(published, ["first", "second"])
            self.assertEqual(result["status"], "completed")
            expected_shas = {
                load_trusted_task_contract(first)[1],
                load_trusted_task_contract(second)[1],
            }
            self.assertEqual(set(result["completedTaskShas"]), expected_shas)

    def test_dependency_backlog_runs_ready_backend_before_blocked_ui(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts"
            contracts.mkdir()
            ui = _write_contract(contracts / "001-ui.json", "ui", depends_on=["backend"])
            backend = _write_contract(contracts / "002-backend.json", "backend")
            state_path = root / "state.json"
            executed: list[str] = []
            snapshots: list[dict[str, object]] = []

            def delivery(options):
                snapshots.append(json.loads(state_path.read_text(encoding="utf-8")))
                contract, task_sha = load_trusted_task_contract(options.task_contract_path)
                executed.append(contract.id)
                return {"status": "completed", "taskSha": task_sha, "commitSha": f"commit-{contract.id}"}

            times = iter((0.0, 0.0, 61.0))
            result = run_continuous_bounded_delivery(
                ContinuousBoundedOptions(
                    project_path=root,
                    contract_dir=contracts,
                    provider_for_role=lambda _role: _NoopProvider(),
                    workspace_allowlist=[tmp],
                    report_dir=root / "reports",
                    state_path=state_path,
                    once=False,
                    interval_minutes=1,
                    max_runtime_minutes=1,
                    github_execute=True,
                    auto_merge=True,
                    delivery_runner=delivery,
                    publisher=lambda _options, _entry, _result: {"success": True},
                    monotonic=lambda: next(times),
                )
            )

            self.assertEqual(executed, ["backend", "ui"])
            self.assertEqual(snapshots[0]["blockedTasks"][0]["id"], "ui")
            self.assertEqual(snapshots[0]["blockedTasks"][0]["unmetDependencies"], ["backend"])
            self.assertEqual(result["status"], "completed")
            self.assertEqual(
                set(result["completedTaskShas"]),
                {load_trusted_task_contract(backend)[1], load_trusted_task_contract(ui)[1]},
            )

    def test_dependency_backlog_rejects_unknown_dependency_and_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            contracts = Path(tmp) / "unknown"
            contracts.mkdir()
            _write_contract(contracts / "001-ui.json", "ui", depends_on=["missing"])
            with self.assertRaisesRegex(ValueError, "unknown dependencies"):
                discover_contracts(contracts)

        with tempfile.TemporaryDirectory() as tmp:
            contracts = Path(tmp) / "cycle"
            contracts.mkdir()
            _write_contract(contracts / "001-a.json", "a", depends_on=["b"])
            _write_contract(contracts / "002-b.json", "b", depends_on=["a"])
            with self.assertRaisesRegex(ValueError, "contain a cycle"):
                discover_contracts(contracts)

    def test_attention_required_stops_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts"
            contracts.mkdir()
            _write_contract(contracts / "001-task.json", "task")
            published: list[str] = []

            result = run_continuous_bounded_delivery(
                ContinuousBoundedOptions(
                    project_path=root,
                    contract_dir=contracts,
                    provider_for_role=lambda _role: _NoopProvider(),
                    workspace_allowlist=[tmp],
                    report_dir=root / "reports",
                    state_path=root / "state.json",
                    once=False,
                    github_execute=True,
                    auto_merge=True,
                    delivery_runner=lambda _options: {
                        "status": "attention-required",
                        "stopReason": "architect-requires-product-decision",
                    },
                    publisher=lambda _options, entry, _result: published.append(entry.contract.id),
                )
            )

            self.assertEqual(result["status"], "attention-required")
            self.assertEqual(result["stopReason"], "architect-requires-product-decision")
            self.assertEqual(published, [])

    def test_transient_provider_failure_waits_and_retries_same_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts"
            contracts.mkdir()
            contract_path = _write_contract(contracts / "001-task.json", "task")
            task_sha = load_trusted_task_contract(contract_path)[1]
            attempts: list[str] = []
            sleeps: list[float] = []
            waiting_states: list[dict[str, object]] = []
            state_path = root / "state.json"
            now = [datetime(2026, 7, 15, tzinfo=UTC)]

            def delivery(_options):
                attempts.append(task_sha)
                if len(attempts) == 1:
                    return {
                        "status": "attention-required",
                        "stopReason": "provider-quota-exhausted",
                        "taskSha": task_sha,
                    }
                return {
                    "status": "completed",
                    "taskSha": task_sha,
                    "commitSha": "commit-task",
                }

            def sleep_and_capture(seconds: float) -> None:
                sleeps.append(seconds)
                waiting_states.append(json.loads(state_path.read_text(encoding="utf-8")))
                now[0] += timedelta(seconds=seconds)

            times = iter((0.0, 0.0, 61.0))
            result = run_continuous_bounded_delivery(
                ContinuousBoundedOptions(
                    project_path=root,
                    contract_dir=contracts,
                    provider_for_role=lambda _role: _NoopProvider(),
                    workspace_allowlist=[tmp],
                    report_dir=root / "reports",
                    state_path=state_path,
                    once=False,
                    interval_minutes=1,
                    max_runtime_minutes=1,
                    github_execute=True,
                    auto_merge=True,
                    delivery_runner=delivery,
                    publisher=lambda _options, _entry, _result: {"success": True},
                    sleeper=sleep_and_capture,
                    monotonic=lambda: next(times),
                    wall_clock=lambda: now[0],
                )
            )

            self.assertEqual(attempts, [task_sha, task_sha])
            self.assertEqual(sleeps, [60])
            self.assertEqual(waiting_states[0]["status"], "waiting-provider")
            self.assertEqual(waiting_states[0]["stopReason"], "provider-quota-exhausted")
            self.assertEqual(waiting_states[0]["nextAction"], "retry-after-provider-reset")
            self.assertEqual(
                waiting_states[0]["providerBackoff"],
                {
                    "taskSha": task_sha,
                    "stage": "unknown",
                    "stopReason": "provider-quota-exhausted",
                    "consecutiveFailures": 1,
                    "delaySeconds": 60,
                    "nextRetryAt": "2026-07-15T00:01:00+00:00",
                },
            )
            self.assertEqual(result["status"], "completed")
            self.assertIn(task_sha, result["completedTaskShas"])

    def test_repeated_quota_failures_use_persisted_exponential_backoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts"
            contracts.mkdir()
            contract_path = _write_contract(contracts / "001-task.json", "task")
            task_sha = load_trusted_task_contract(contract_path)[1]
            attempts = 0
            sleeps: list[float] = []
            now = [datetime(2026, 7, 15, tzinfo=UTC)]

            def delivery(_options):
                nonlocal attempts
                attempts += 1
                if attempts <= 3:
                    return {
                        "status": "attention-required",
                        "stopReason": "provider-quota-exhausted",
                        "taskSha": task_sha,
                    }
                return {
                    "status": "completed",
                    "taskSha": task_sha,
                    "commitSha": "commit-task",
                }

            def sleep_and_advance(seconds: float) -> None:
                sleeps.append(seconds)
                now[0] += timedelta(seconds=seconds)

            times = iter((0.0, 0.0, 0.0, 0.0, 61.0))
            result = run_continuous_bounded_delivery(
                ContinuousBoundedOptions(
                    project_path=root,
                    contract_dir=contracts,
                    provider_for_role=lambda _role: _NoopProvider(),
                    workspace_allowlist=[tmp],
                    report_dir=root / "reports",
                    state_path=root / "state.json",
                    once=False,
                    interval_minutes=1,
                    max_runtime_minutes=1,
                    github_execute=True,
                    auto_merge=True,
                    delivery_runner=delivery,
                    publisher=lambda _options, _entry, _result: {"success": True},
                    sleeper=sleep_and_advance,
                    monotonic=lambda: next(times),
                    wall_clock=lambda: now[0],
                )
            )

            self.assertEqual(attempts, 4)
            self.assertEqual(sleeps, [60, 120, 240])
            self.assertEqual(result["status"], "completed")

    def test_restart_honors_persisted_quota_backoff_before_provider_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts"
            contracts.mkdir()
            contract_path = _write_contract(contracts / "001-task.json", "task")
            task_sha = load_trusted_task_contract(contract_path)[1]
            now = datetime(2026, 7, 15, tzinfo=UTC)
            state_path = root / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "revision": 4,
                        "status": "waiting-provider",
                        "stopReason": "provider-quota-exhausted",
                        "completedTaskShas": [],
                        "currentTask": {"taskSha": task_sha},
                        "providerBackoff": {
                            "taskSha": task_sha,
                            "stage": "engineer",
                            "stopReason": "provider-quota-exhausted",
                            "consecutiveFailures": 2,
                            "delaySeconds": 120,
                            "nextRetryAt": (now + timedelta(seconds=120)).isoformat(),
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = run_continuous_bounded_delivery(
                ContinuousBoundedOptions(
                    project_path=root,
                    contract_dir=contracts,
                    provider_for_role=lambda _role: _NoopProvider(),
                    workspace_allowlist=[tmp],
                    report_dir=root / "reports",
                    state_path=state_path,
                    once=True,
                    delivery_runner=lambda _options: self.fail("provider ran before persisted retry time"),
                    sleeper=lambda _seconds: self.fail("--once must not sleep for persisted quota backoff"),
                    wall_clock=lambda: now,
                )
            )

            self.assertEqual(result["status"], "waiting-provider")
            self.assertEqual(result["providerBackoff"]["consecutiveFailures"], 2)
            self.assertEqual(result["providerBackoff"]["delaySeconds"], 120)

    def test_quota_backoff_resets_when_delivery_advances_to_another_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts"
            contracts.mkdir()
            contract_path = _write_contract(contracts / "001-task.json", "task")
            task_sha = load_trusted_task_contract(contract_path)[1]
            attempts = 0
            sleeps: list[float] = []
            now = [datetime(2026, 7, 15, tzinfo=UTC)]

            def delivery(_options):
                nonlocal attempts
                attempts += 1
                if attempts <= 3:
                    return {
                        "status": "attention-required",
                        "stage": "engineer" if attempts <= 2 else "qa",
                        "stopReason": "provider-quota-exhausted",
                        "taskSha": task_sha,
                    }
                return {
                    "status": "completed",
                    "taskSha": task_sha,
                    "commitSha": "commit-task",
                }

            def sleep_and_advance(seconds: float) -> None:
                sleeps.append(seconds)
                now[0] += timedelta(seconds=seconds)

            times = iter((0.0, 0.0, 0.0, 0.0, 61.0))
            result = run_continuous_bounded_delivery(
                ContinuousBoundedOptions(
                    project_path=root,
                    contract_dir=contracts,
                    provider_for_role=lambda _role: _NoopProvider(),
                    workspace_allowlist=[tmp],
                    report_dir=root / "reports",
                    state_path=root / "state.json",
                    once=False,
                    interval_minutes=1,
                    max_runtime_minutes=1,
                    github_execute=True,
                    auto_merge=True,
                    delivery_runner=delivery,
                    publisher=lambda _options, _entry, _result: {"success": True},
                    sleeper=sleep_and_advance,
                    monotonic=lambda: next(times),
                    wall_clock=lambda: now[0],
                )
            )

            self.assertEqual(attempts, 4)
            self.assertEqual(sleeps, [60, 120, 60])
            self.assertEqual(result["status"], "completed")

    def test_quota_backoff_uses_task_checkpoint_stage_when_runner_summary_omits_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts"
            contracts.mkdir()
            contract_path = _write_contract(contracts / "001-task.json", "task")
            task_sha = load_trusted_task_contract(contract_path)[1]

            def delivery(options):
                options.state_path.parent.mkdir(parents=True, exist_ok=True)
                options.state_path.write_text(
                    json.dumps({"status": "attention-required", "stage": "engineer"}),
                    encoding="utf-8",
                )
                return {
                    "status": "attention-required",
                    "stopReason": "provider-quota-exhausted",
                    "taskSha": task_sha,
                }

            result = run_continuous_bounded_delivery(
                ContinuousBoundedOptions(
                    project_path=root,
                    contract_dir=contracts,
                    provider_for_role=lambda _role: _NoopProvider(),
                    workspace_allowlist=[tmp],
                    report_dir=root / "reports",
                    state_path=root / "state.json",
                    once=True,
                    interval_minutes=1,
                    delivery_runner=delivery,
                    wall_clock=lambda: datetime(2026, 7, 15, tzinfo=UTC),
                )
            )

            self.assertEqual(result["status"], "waiting-provider")
            self.assertEqual(result["providerBackoff"]["stage"], "engineer")

    def test_delivery_exception_is_recorded_without_escaping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts"
            contracts.mkdir()
            _write_contract(contracts / "001-task.json", "task")

            result = run_continuous_bounded_delivery(
                ContinuousBoundedOptions(
                    project_path=root,
                    contract_dir=contracts,
                    provider_for_role=lambda _role: _NoopProvider(),
                    workspace_allowlist=[tmp],
                    report_dir=root / "reports",
                    state_path=root / "state.json",
                    once=True,
                    delivery_runner=lambda _options: (_ for _ in ()).throw(
                        RuntimeError("token = supervisor-secret-value")
                    ),
                )
            )

            self.assertEqual(result["status"], "attention-required")
            self.assertEqual(result["stopReason"], "bounded-delivery-exception")
            text = (root / "state.json").read_text(encoding="utf-8")
            self.assertNotIn("supervisor-secret-value", text)
            self.assertIn("<redacted>", text)

    def test_contract_sha_change_during_delivery_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts"
            contracts.mkdir()
            _write_contract(contracts / "001-task.json", "task")
            published: list[str] = []

            result = run_continuous_bounded_delivery(
                ContinuousBoundedOptions(
                    project_path=root,
                    contract_dir=contracts,
                    provider_for_role=lambda _role: _NoopProvider(),
                    workspace_allowlist=[tmp],
                    report_dir=root / "reports",
                    state_path=root / "state.json",
                    once=True,
                    github_execute=True,
                    auto_merge=True,
                    delivery_runner=lambda _options: {
                        "status": "completed",
                        "taskSha": "f" * 64,
                        "commitSha": "a" * 40,
                    },
                    publisher=lambda _options, entry, _result: published.append(entry.contract.id),
                )
            )

            self.assertEqual(result["status"], "attention-required")
            self.assertEqual(result["stopReason"], "task-contract-changed-during-execution")
            self.assertEqual(published, [])

    def test_completed_task_sha_is_not_run_again(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts"
            contracts.mkdir()
            contract_path = _write_contract(contracts / "001-task.json", "task")
            task_sha = load_trusted_task_contract(contract_path)[1]
            state_path = root / "state.json"
            state_path.write_text(
                json.dumps({"revision": 3, "completedTaskShas": [task_sha]}),
                encoding="utf-8",
            )

            result = run_continuous_bounded_delivery(
                ContinuousBoundedOptions(
                    project_path=root,
                    contract_dir=contracts,
                    provider_for_role=lambda _role: _NoopProvider(),
                    workspace_allowlist=[tmp],
                    report_dir=root / "reports",
                    state_path=state_path,
                    once=True,
                    delivery_runner=lambda _options: self.fail("completed task was rerun"),
                )
            )

            self.assertEqual(result["status"], "idle")

    def test_invalid_queue_preserves_completed_task_shas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts"
            contracts.mkdir()
            completed_sha = "a" * 64
            state_path = root / "state.json"
            state_path.write_text(
                json.dumps({"revision": 3, "completedTaskShas": [completed_sha]}),
                encoding="utf-8",
            )
            (contracts / "001-invalid.json").write_text("{not-json", encoding="utf-8")

            result = run_continuous_bounded_delivery(
                ContinuousBoundedOptions(
                    project_path=root,
                    contract_dir=contracts,
                    provider_for_role=lambda _role: _NoopProvider(),
                    workspace_allowlist=[tmp],
                    report_dir=root / "reports",
                    state_path=state_path,
                    once=True,
                )
            )

            self.assertEqual(result["status"], "attention-required")
            self.assertEqual(result["stopReason"], "contract-queue-invalid")
            self.assertEqual(result["completedTaskShas"], [completed_sha])

    def test_corrupt_supervisor_state_is_preserved_and_stops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts"
            contracts.mkdir()
            _write_contract(contracts / "001-task.json", "task")
            state_path = root / "state.json"
            state_path.write_text("{corrupt", encoding="utf-8")

            result = run_continuous_bounded_delivery(
                ContinuousBoundedOptions(
                    project_path=root,
                    contract_dir=contracts,
                    provider_for_role=lambda _role: _NoopProvider(),
                    workspace_allowlist=[tmp],
                    report_dir=root / "reports",
                    state_path=state_path,
                    once=True,
                    delivery_runner=lambda _options: self.fail("corrupt state must not rerun work"),
                )
            )

            self.assertEqual(result["status"], "attention-required")
            self.assertEqual(result["stopReason"], "supervisor-state-invalid")
            self.assertEqual(state_path.read_text(encoding="utf-8"), "{corrupt")
            self.assertTrue((root / "state.json.error.json").is_file())

    def test_invalid_persisted_provider_backoff_is_preserved_and_stops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts"
            contracts.mkdir()
            _write_contract(contracts / "001-task.json", "task")
            state_path = root / "state.json"
            original = json.dumps(
                {
                    "revision": 3,
                    "completedTaskShas": [],
                    "providerBackoff": {
                        "taskSha": "not-a-sha",
                        "stage": "engineer",
                        "stopReason": "provider-quota-exhausted",
                        "consecutiveFailures": 1,
                        "delaySeconds": 60,
                        "nextRetryAt": "2026-07-15T00:01:00+00:00",
                    },
                }
            )
            state_path.write_text(original, encoding="utf-8")

            result = run_continuous_bounded_delivery(
                ContinuousBoundedOptions(
                    project_path=root,
                    contract_dir=contracts,
                    provider_for_role=lambda _role: _NoopProvider(),
                    workspace_allowlist=[tmp],
                    report_dir=root / "reports",
                    state_path=state_path,
                    once=True,
                    delivery_runner=lambda _options: self.fail("invalid backoff state must not run work"),
                )
            )

            self.assertEqual(result["status"], "attention-required")
            self.assertEqual(result["stopReason"], "supervisor-state-invalid")
            self.assertEqual(state_path.read_text(encoding="utf-8"), original)
            self.assertTrue((root / "state.json.error.json").is_file())

    def test_state_redacts_publication_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts"
            contracts.mkdir()
            contract_path = _write_contract(contracts / "001-task.json", "task")
            task_sha = load_trusted_task_contract(contract_path)[1]
            state_path = root / "state.json"

            run_continuous_bounded_delivery(
                ContinuousBoundedOptions(
                    project_path=root,
                    contract_dir=contracts,
                    provider_for_role=lambda _role: _NoopProvider(),
                    workspace_allowlist=[tmp],
                    report_dir=root / "reports",
                    state_path=state_path,
                    once=True,
                    github_execute=True,
                    auto_merge=True,
                    delivery_runner=lambda _options: {
                        "status": "completed",
                        "taskSha": task_sha,
                        "commitSha": "abc",
                    },
                    publisher=lambda _options, _entry, _result: {
                        "success": True,
                        "token": "publication-secret-value",
                    },
                )
            )

            text = state_path.read_text(encoding="utf-8")
            self.assertNotIn("publication-secret-value", text)
            self.assertIn("<redacted>", text)

    def test_continuous_mode_requires_pr_execution_and_auto_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts"
            contracts.mkdir()
            with self.assertRaisesRegex(ValueError, "requires PR execution"):
                run_continuous_bounded_delivery(
                    ContinuousBoundedOptions(
                        project_path=root,
                        contract_dir=contracts,
                        provider_for_role=lambda _role: _NoopProvider(),
                        workspace_allowlist=[tmp],
                        report_dir=root / "reports",
                        state_path=root / "state.json",
                    )
                )

    @unittest.skipUnless(hasattr(Path, "symlink_to"), "symlinks unsupported")
    def test_contract_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts"
            contracts.mkdir()
            target = _write_contract(root / "outside.json", "outside")
            link = contracts / "001-link.json"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("symlink creation is unavailable")
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                discover_contracts(contracts)

    def test_oversized_contract_is_rejected_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            contracts = Path(tmp) / "contracts"
            contracts.mkdir()
            (contracts / "001-large.json").write_text("x" * 64_001, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "exceeds 64000 bytes"):
                discover_contracts(contracts)

    def test_publication_waives_review_only_for_development_and_keeps_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts"
            contracts.mkdir()
            contract_path = _write_contract(contracts / "001-task.json", "task")
            contract_payload = json.loads(contract_path.read_text(encoding="utf-8"))
            contract_payload["title"] = "api_key=publication-title-secret"
            contract_path.write_text(json.dumps(contract_payload), encoding="utf-8")
            contract, task_sha = load_trusted_task_contract(contract_path)
            entry = discover_contracts(contracts)[0]
            worktree = root / "worktree"
            worktree.mkdir()
            receipt = root / "run-receipt.json"
            receipt.write_text("{}", encoding="utf-8")
            result = {
                "status": "completed",
                "taskSha": task_sha,
                "commitSha": "a" * 40,
                "worktreePath": str(worktree),
                "runReceipt": str(receipt),
                "validation": {"success": True, "hash": "b" * 64},
            }
            primary = SimpleNamespace(
                current_branch="master",
                profile=SimpleNamespace(project=SimpleNamespace(stage="development")),
            )
            disposable = SimpleNamespace()
            github_options = []

            def github_action(_loaded, options):
                github_options.append(options)
                return SimpleNamespace(
                    success=True,
                    pr_url="https://example.test/pull/1" if options.action == "pr" else None,
                    as_dict=lambda: {"success": True, "action": options.action},
                )

            ci = SimpleNamespace(
                merge_ready=True,
                status="passed",
                evidence_path=root / "ci.json",
                evidence={"blockers": []},
            )
            options = ContinuousBoundedOptions(
                project_path=root,
                contract_dir=contracts,
                provider_for_role=lambda _role: _NoopProvider(),
                workspace_allowlist=[tmp],
                report_dir=root / "reports",
                state_path=root / "state.json",
                once=False,
                github_execute=True,
                auto_merge=True,
                allow_unreviewed_development_merge=True,
            )

            with (
                patch(
                    "ai_team.core.bounded_supervisor.load_project",
                    side_effect=[primary, disposable],
                ),
                patch(
                    "ai_team.core.bounded_supervisor._repository_name",
                    return_value="example/project",
                ),
                patch(
                    "ai_team.core.bounded_supervisor._find_pull_request",
                    return_value=None,
                ),
                patch(
                    "ai_team.core.bounded_supervisor.execute_github_action",
                    side_effect=github_action,
                ),
                patch(
                    "ai_team.core.bounded_supervisor.monitor_pull_request",
                    return_value=ci,
                ) as monitor,
                patch(
                    "ai_team.core.bounded_supervisor._sync_primary",
                    return_value="c" * 40,
                ),
            ):
                publication = publish_and_merge(options, entry, result)

            self.assertTrue(publication["success"], publication)
            self.assertEqual(contract.id, "task")
            self.assertEqual([item.action for item in github_options], ["pr", "merge"])
            self.assertEqual(github_options[0].validation_log_hash, "b" * 64)
            self.assertEqual(github_options[0].receipt_path, receipt)
            self.assertNotIn("publication-title-secret", github_options[0].title)
            self.assertIn("<redacted>", github_options[0].title)
            self.assertFalse(github_options[1].require_approved_review)
            self.assertFalse(monitor.call_args.kwargs["require_approved_review"])

    def test_unreviewed_publication_fails_closed_outside_development(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts"
            contracts.mkdir()
            _write_contract(contracts / "001-task.json", "task")
            entry = discover_contracts(contracts)[0]
            worktree = root / "worktree"
            worktree.mkdir()
            receipt = root / "run-receipt.json"
            receipt.write_text("{}", encoding="utf-8")
            result = {
                "status": "completed",
                "taskSha": entry.task_sha,
                "commitSha": "a" * 40,
                "worktreePath": str(worktree),
                "runReceipt": str(receipt),
                "validation": {"hash": "b" * 64},
            }
            primary = SimpleNamespace(
                current_branch="master",
                profile=SimpleNamespace(project=SimpleNamespace(stage="production")),
            )
            options = ContinuousBoundedOptions(
                project_path=root,
                contract_dir=contracts,
                provider_for_role=lambda _role: _NoopProvider(),
                workspace_allowlist=[tmp],
                report_dir=root / "reports",
                state_path=root / "state.json",
                once=True,
                github_execute=True,
                auto_merge=True,
                allow_unreviewed_development_merge=True,
            )

            with patch("ai_team.core.bounded_supervisor.load_project", return_value=primary):
                publication = publish_and_merge(options, entry, result)

            self.assertFalse(publication["success"])
            self.assertEqual(
                publication["stopReason"],
                "unreviewed-merge-requires-development-stage",
            )

    def test_primary_sync_fails_closed_when_status_inspection_fails(self) -> None:
        failed_status = SimpleNamespace(returncode=128, stdout="", stderr="secret=hidden")
        with patch("ai_team.core.bounded_supervisor._run", return_value=failed_status):
            with self.assertRaisesRegex(RuntimeError, "inspect primary"):
                _sync_primary(Path("/project"), "master")

    def test_primary_sync_fails_closed_on_unexpected_branch(self) -> None:
        results = iter(
            (
                SimpleNamespace(returncode=0, stdout="", stderr=""),
                SimpleNamespace(returncode=0, stdout="feature/task\n", stderr=""),
            )
        )
        with patch("ai_team.core.bounded_supervisor._run", side_effect=lambda *_args: next(results)):
            with self.assertRaisesRegex(RuntimeError, "expected branch"):
                _sync_primary(Path("/project"), "master")


class _NoopProvider(BaseProvider):
    def ready(self) -> bool:
        return True

    def run(self, request: ProviderRequest) -> ProviderResult:
        return ProviderResult(provider="noop", success=True)


def _write_contract(path: Path, task_id: str, *, depends_on: list[str] | None = None) -> Path:
    path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "id": task_id,
                "title": f"Task {task_id}",
                "source": {"kind": "trusted-contract", "reference": f"test:{task_id}"},
                "instruction": f"Update the documented behavior for {task_id}.",
                "allowedWritePaths": [f"docs/{task_id}.md"],
                "validationCommands": ["git diff --check"],
                **({"dependsOn": depends_on} if depends_on is not None else {}),
            }
        ),
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    unittest.main()
