from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ai_team.core.bounded_delivery import load_trusted_task_contract
from ai_team.core.bounded_supervisor import (
    ContinuousBoundedOptions,
    _sync_primary,
    discover_contracts,
    publish_and_merge,
    run_continuous_bounded_delivery,
)
from ai_team.providers.base import BaseProvider, ProviderRequest, ProviderResult


class ContinuousBoundedSupervisorTests(unittest.TestCase):
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


def _write_contract(path: Path, task_id: str) -> Path:
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
            }
        ),
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    unittest.main()
