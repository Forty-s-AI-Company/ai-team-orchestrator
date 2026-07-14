from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from ai_team.core.bounded_delivery import (
    BoundedDeliveryError,
    BoundedDeliveryOptions,
    DeliveryLimits,
    EngineeringAttempt,
    load_trusted_task_contract,
    run_bounded_delivery,
)
from ai_team.providers.base import BaseProvider, ProviderErrorType, ProviderRequest, ProviderResult


class BoundedDeliveryTests(unittest.TestCase):
    def test_fake_native_providers_complete_all_state_transitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _init_project(Path(tmp) / "project")
            contract_path = _write_contract(Path(tmp), "docs/safe.md")
            options = _options(Path(tmp), root, contract_path, _provider_for_role, _successful_engineering_attempt)

            result = run_bounded_delivery(options)

            self.assertEqual(result["status"], "completed", result)
            self.assertEqual(result["commitSha"], "fake-commit")
            receipts = [Path(path) for path in result["receipts"]]
            self.assertEqual([path.name for path in receipts], ["01-pm.json", "02-architect.json", "03-engineer.json", "04-qa.json", "05-review.json"])
            self.assertTrue(all(path.exists() for path in receipts))
            architect_receipt = json.loads(receipts[1].read_text(encoding="utf-8"))
            self.assertEqual(architect_receipt["secondaryReview"]["provider"], "codex")
            self.assertEqual(architect_receipt["outerRunMode"], "bounded-delivery")
            self.assertFalse(architect_receipt["writeAccess"])
            engineer_receipt = json.loads(receipts[2].read_text(encoding="utf-8"))
            self.assertTrue(engineer_receipt["writeAccess"])
            self.assertEqual(engineer_receipt["commitSha"], "fake-commit")
            self.assertNotIn("super-secret-value", receipts[0].read_text(encoding="utf-8"))
            qa_prompt_evidence = json.loads(receipts[3].read_text(encoding="utf-8"))["evidence"]
            self.assertEqual(qa_prompt_evidence["stage"], "qa")
            self.assertEqual(run_bounded_delivery(options)["status"], "already-completed")

    def test_mock_provider_can_never_count_as_a_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _init_project(Path(tmp) / "project")
            contract_path = _write_contract(Path(tmp), "docs/safe.md")
            result = run_bounded_delivery(
                _options(Path(tmp), root, contract_path, lambda _role: _MockSuccessProvider(), _successful_engineering_attempt)
            )
            self.assertEqual(result["status"], "attention-required")
            self.assertEqual(result["stopReason"], "mock-provider-denied")

    def test_out_of_scope_qa_finding_stops_without_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _init_project(Path(tmp) / "project")
            contract_path = _write_contract(Path(tmp), "docs/safe.md")

            def provider(role: str) -> BaseProvider:
                return _StageProvider(role, qa_findings=[{"path": "src/outside.ts", "message": "unrelated"}])

            result = run_bounded_delivery(_options(Path(tmp), root, contract_path, provider, _successful_engineering_attempt))
            self.assertEqual(result["status"], "attention-required")
            self.assertEqual(result["stopReason"], "unattributed-or-out-of-scope-finding")

    def test_engineering_diff_must_stay_within_the_architect_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _init_project(Path(tmp) / "project")
            contract_path = _write_contract(Path(tmp), "docs")

            def outside_plan(contract, instruction: str, provider: BaseProvider, iteration: int) -> EngineeringAttempt:
                attempt = _successful_engineering_attempt(contract, instruction, provider, iteration)
                return EngineeringAttempt(
                    provider_result=attempt.provider_result,
                    worktree_path=attempt.worktree_path,
                    changed_files=["docs/other.md"],
                    validation=attempt.validation,
                    commit_sha=attempt.commit_sha,
                )

            result = run_bounded_delivery(_options(Path(tmp), root, contract_path, _provider_for_role, outside_plan))
            self.assertEqual(result["status"], "attention-required")
            self.assertEqual(result["stopReason"], "engineering-diff-outside-allowed-paths")

    def test_provider_timeout_is_a_fail_closed_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _init_project(Path(tmp) / "project")
            contract_path = _write_contract(Path(tmp), "docs/safe.md")
            result = run_bounded_delivery(
                _options(Path(tmp), root, contract_path, lambda _role: _TimeoutProvider(), _successful_engineering_attempt)
            )
            self.assertEqual(result["status"], "attention-required")
            self.assertEqual(result["stopReason"], "provider-timeout")
            self.assertTrue((Path(tmp) / "reports" / "01-pm.json").exists())

    def test_read_only_stage_worktree_write_is_a_fail_closed_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _init_project(Path(tmp) / "project")
            contract_path = _write_contract(Path(tmp), "docs/safe.md")

            def provider(role: str) -> BaseProvider:
                return _WritingQaProvider(role)

            result = run_bounded_delivery(
                _options(Path(tmp), root, contract_path, provider, _successful_engineering_attempt)
            )

            self.assertEqual(result["status"], "attention-required")
            self.assertEqual(result["stopReason"], "read-only-stage-modified-worktree")
            qa_receipt = json.loads((Path(tmp) / "reports" / "04-qa.json").read_text())
            self.assertEqual(
                qa_receipt["evidence"]["validationError"],
                "read-only-stage-modified-worktree",
            )

    def test_architect_requires_a_structured_codex_second_opinion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _init_project(Path(tmp) / "project")
            contract_path = _write_contract(Path(tmp), "docs/safe.md")

            def provider(role: str) -> BaseProvider:
                return _ArchitectWithoutCodexProvider(role)

            result = run_bounded_delivery(_options(Path(tmp), root, contract_path, provider, _successful_engineering_attempt))
            self.assertEqual(result["status"], "attention-required")
            self.assertEqual(result["stopReason"], "architect-codex-read-only-review-failed")
            self.assertTrue((Path(tmp) / "reports" / "02-architect.json").exists())

    def test_qa_requires_a_structured_codex_second_opinion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _init_project(Path(tmp) / "project")
            contract_path = _write_contract(Path(tmp), "docs/safe.md")

            def provider(role: str) -> BaseProvider:
                return _QaWithoutCodexProvider(role)

            result = run_bounded_delivery(
                _options(Path(tmp), root, contract_path, provider, _successful_engineering_attempt)
            )

            self.assertEqual(result["status"], "attention-required")
            self.assertEqual(result["stopReason"], "qa-codex-read-only-review-failed")
            qa_receipt = json.loads((Path(tmp) / "reports" / "04-qa.json").read_text())
            self.assertFalse(qa_receipt["validationResult"]["success"])

    def test_secondary_qa_findings_enter_the_repair_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _init_project(Path(tmp) / "project")
            contract_path = _write_contract(Path(tmp), "docs/safe.md")

            def provider(role: str) -> BaseProvider:
                return _QaSecondaryFindingProvider(role)

            result = run_bounded_delivery(
                _options(Path(tmp), root, contract_path, provider, _successful_engineering_attempt)
            )

            self.assertEqual(result["status"], "attention-required")
            self.assertEqual(result["stopReason"], "max-repair-attempts-reached")
            self.assertEqual(len(result["repairs"]), 1)

    def test_retry_preserves_prior_receipts_after_an_attention_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _init_project(Path(tmp) / "project")
            contract_path = _write_contract(Path(tmp), "docs/safe.md")
            options = _options(Path(tmp), root, contract_path, lambda _role: _TimeoutProvider(), _successful_engineering_attempt)
            first = run_bounded_delivery(options)
            self.assertEqual(first["status"], "attention-required")
            second = run_bounded_delivery(options)
            self.assertEqual(second["receipts"], [str(Path(tmp) / "reports" / "01-pm.json"), str(Path(tmp) / "reports" / "02-pm.json")])

    def test_task_contract_rejects_prohibited_actions_before_a_provider_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_contract(Path(tmp), "docs/safe.md", instruction="run a database migration")
            with self.assertRaises(BoundedDeliveryError):
                load_trusted_task_contract(path)

    def test_task_contract_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_contract(Path(tmp), "../.env")
            with self.assertRaises(BoundedDeliveryError):
                load_trusted_task_contract(path)


class _StageProvider(BaseProvider):
    def __init__(self, role: str, qa_findings: list[dict[str, str]] | None = None) -> None:
        self.role = role
        self.qa_findings = qa_findings or []

    def ready(self) -> bool:
        return True

    def run(self, request: ProviderRequest) -> ProviderResult:
        stage = request.metadata["boundedStage"]
        payload: dict[str, object] = {
            "schema": "ai-team-bounded-delivery/v1",
            "stage": stage,
            "status": "passed",
            "findings": self.qa_findings if stage == "qa" else [],
            "tests": ["evidence reviewed"],
            "blockers": [],
        }
        if stage == "pm":
            payload["acceptanceCriteria"] = ["A safe documentation file is updated", "API_KEY=super-secret-value is redacted"]
        if stage == "architect":
            payload.update({
                "plan": ["Edit only docs/safe.md"],
                "allowedWritePaths": ["docs/safe.md"],
                "validationCommands": ["npm run lint", "npm run typecheck", "npm run test", "npm run build"],
                "schemaOrApiChange": False,
            })
        expected_provider = "antigravity" if self.role in {"product-manager", "architect", "delivery-qa"} else "codex"
        data: dict[str, object] = {"tokenUsage": 10, "selectedModel": "fake-model", "reasoningEffort": "high"}
        secondary_provider = {
            "architect": "codex",
            "qa": "codex",
            "review": "antigravity",
        }.get(stage)
        if secondary_provider is not None:
            data["secondaryReview"] = {
                "provider": secondary_provider,
                "success": True,
                "content": json.dumps({
                    "schema": "ai-team-bounded-delivery/v1", "stage": stage, "status": "passed",
                    "findings": [], "tests": ["independent review"], "blockers": [],
                }),
            }
        return ProviderResult(provider=expected_provider, success=True, content=json.dumps(payload), data=data)


class _MockSuccessProvider(_StageProvider):
    def __init__(self) -> None:
        super().__init__("product-manager")

    def run(self, request: ProviderRequest) -> ProviderResult:
        result = super().run(request)
        return ProviderResult(provider="mock", success=True, content=result.content, data=result.data)


class _ArchitectWithoutCodexProvider(_StageProvider):
    def run(self, request: ProviderRequest) -> ProviderResult:
        result = super().run(request)
        if request.metadata["boundedStage"] != "architect":
            return result
        return ProviderResult(provider=result.provider, success=True, content=result.content, data={**result.data, "secondaryReview": None})


class _QaWithoutCodexProvider(_StageProvider):
    def run(self, request: ProviderRequest) -> ProviderResult:
        result = super().run(request)
        if request.metadata["boundedStage"] != "qa":
            return result
        return ProviderResult(
            provider=result.provider,
            success=True,
            content=result.content,
            data={**result.data, "secondaryReview": None},
        )


class _QaSecondaryFindingProvider(_StageProvider):
    def run(self, request: ProviderRequest) -> ProviderResult:
        result = super().run(request)
        if request.metadata["boundedStage"] != "qa":
            return result
        secondary = {
            **result.data["secondaryReview"],
            "content": json.dumps(
                {
                    "schema": "ai-team-bounded-delivery/v1",
                    "stage": "qa",
                    "status": "passed",
                    "findings": [{"path": "docs/safe.md", "message": "secondary QA finding"}],
                    "tests": ["independent review"],
                    "blockers": [],
                }
            ),
        }
        return ProviderResult(
            provider=result.provider,
            success=True,
            content=result.content,
            data={**result.data, "secondaryReview": secondary},
        )


class _TimeoutProvider(BaseProvider):
    def ready(self) -> bool:
        return True

    def run(self, request: ProviderRequest) -> ProviderResult:
        return ProviderResult(provider="antigravity", success=False, error_type=ProviderErrorType.TIMEOUT, content="timeout")


class _WritingQaProvider(_StageProvider):
    def run(self, request: ProviderRequest) -> ProviderResult:
        if request.metadata["boundedStage"] == "qa":
            (request.project_root / "qa-must-not-write.txt").write_text("unexpected", encoding="utf-8")
        return super().run(request)


def _provider_for_role(role: str) -> BaseProvider:
    return _StageProvider(role)


def _successful_engineering_attempt(contract, instruction: str, provider: BaseProvider, iteration: int) -> EngineeringAttempt:
    result = ProviderResult(provider="codex", success=True, content="implementation complete", data={"tokenUsage": 20})
    return EngineeringAttempt(
        provider_result=result,
        worktree_path=Path("/tmp/fake-bounded-worktree"),
        changed_files=["docs/safe.md"],
        validation={"success": True, "commands": [{"command": "npm run lint", "returnCode": 0}]},
        commit_sha="fake-commit",
    )


def _options(base: Path, root: Path, contract_path: Path, provider_for_role, engineering_executor) -> BoundedDeliveryOptions:
    def execute_in_project(contract, instruction: str, provider: BaseProvider, iteration: int) -> EngineeringAttempt:
        attempt = engineering_executor(contract, instruction, provider, iteration)
        return EngineeringAttempt(
            provider_result=attempt.provider_result,
            worktree_path=root,
            changed_files=attempt.changed_files,
            validation=attempt.validation,
            commit_sha=attempt.commit_sha,
            run_receipt=attempt.run_receipt,
            executor_receipt=attempt.executor_receipt,
        )

    return BoundedDeliveryOptions(
        project_path=root,
        task_contract_path=contract_path,
        provider_for_role=provider_for_role,
        workspace_allowlist=[str(base)],
        report_dir=base / "reports",
        state_path=base / "state.json",
        limits=DeliveryLimits(max_iterations=2, max_repair_attempts=1, max_token_usage=1000, timeout_seconds=30),
        engineering_executor=execute_in_project,
    )


def _write_contract(base: Path, allowed_path: str, instruction: str = "Update the approved documentation only") -> Path:
    path = base / "task.json"
    path.write_text(json.dumps({
        "schemaVersion": 1,
        "id": "trusted-doc-update",
        "title": "Update approved documentation",
        "source": {"kind": "trusted-contract", "reference": "ops/2026-07-13/doc-update"},
        "instruction": instruction,
        "allowedWritePaths": [allowed_path],
        "validationCommands": ["npm run lint", "npm run typecheck", "npm run test", "npm run build"],
    }), encoding="utf-8")
    return path


def _init_project(root: Path) -> Path:
    root.mkdir()
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "AI Team Test"], cwd=root, check=True)
    profile = root / ".ai-team" / "project.yaml"
    profile.parent.mkdir()
    profile.write_text(
        """project:\n  name: sample\n  root: \".\"\n  stage: development\nrepository:\n  protected_branches: [master, main]\ncommands:\n  lint: npm run lint\n  typecheck: npm run typecheck\n  test: npm run test\n  build: npm run build\nsafety:\n  allow_git_push: true\n  allow_deploy: false\n  allow_database_migration: false\n  allow_database_seed: false\n  allow_destructive_commands: false\n""",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True)
    return root


if __name__ == "__main__":
    unittest.main()
