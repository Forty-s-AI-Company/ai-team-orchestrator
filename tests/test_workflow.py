from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ai_team.core.orchestrator import Orchestrator
from ai_team.core.orchestrator import WorkflowError
from ai_team.core.project_loader import ProjectConfigError
from ai_team.core.project_loader import load_project
from ai_team.core.receipts import write_run_receipt
from ai_team.providers import (
    BaseProvider,
    MockProvider,
    ProviderErrorType,
    ProviderRequest,
    ProviderResult,
)


class HandsFreeCodeReadOnlyStub(BaseProvider):
    name = "handsfreecode"

    def __init__(self, content: str | None = None) -> None:
        self.last_request: ProviderRequest | None = None
        self.content = content or (
            "Evidence-backed technology summary: Node.js, Next.js, and TypeScript are identified "
            "from package.json. Test coverage is unknown because no coverage report was provided."
        )

    def ready(self) -> bool:
        return True

    def run(self, request: ProviderRequest) -> ProviderResult:
        self.last_request = request
        return ProviderResult(
            provider=self.name,
            success=True,
            content=self.content,
            data={
                "runMode": request.run_mode,
                "runtimeMode": "run-agent",
                "writeAccess": False,
                "runtimeProvider": "ollama",
                "tokenUsage": 31,
            },
        )


def init_git_project(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    ai_team = root / ".ai-team"
    ai_team.mkdir()
    (ai_team / "project.yaml").write_text(
        """project:
  name: sample
  root: "."
  stage: development

repository:
  protected_branches:
    - master
    - main

commands:
  lint: npm run lint

safety:
  allow_git_push: false
  allow_deploy: false
  allow_database_migration: false
  allow_database_seed: false
  allow_destructive_commands: false
""",
        encoding="utf-8",
    )
    (root / "package.json").write_text(
        '{"dependencies":{"next":"1.0.0"},"devDependencies":{"typescript":"1.0.0"}}',
        encoding="utf-8",
    )
    (root / "tsconfig.json").write_text("{}", encoding="utf-8")
    ci = root / ".github" / "workflows" / "ci.yml"
    ci.parent.mkdir(parents=True)
    ci.write_text(
        "name: CI\njobs:\n  checks:\n    steps:\n      - run: npm run lint\n      - run: npm test\n",
        encoding="utf-8",
    )


class WorkflowTests(unittest.TestCase):
    def test_mock_workflow_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            loaded = load_project(root)
            result = Orchestrator(MockProvider()).run(
                loaded,
                workflow_name="project-analysis",
                dry_run=True,
            )
            self.assertTrue(result.provider_result.success)
            self.assertEqual(result.provider_result.data["runMode"], "create-only")

    def test_create_only_does_not_collect_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            loaded = load_project(root)
            with patch("ai_team.core.orchestrator.collect_project_evidence") as collector:
                result = Orchestrator(MockProvider()).run(
                    loaded,
                    workflow_name="project-analysis",
                    run_mode="create-only",
                )
            collector.assert_not_called()
            self.assertTrue(result.provider_result.success)

    def test_receipt_redacts_provider_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            loaded = load_project(root)
            provider = MockProvider()
            result = Orchestrator(provider).run(
                loaded,
                workflow_name="project-analysis",
                dry_run=True,
            )
            redaction_result = replace(
                result,
                provider_result=replace(
                    result.provider_result,
                    content="SESSION_API_KEY" + "=supersecret",
                    data={
                        "runMode": "run-agent",
                        "conversationId": "11111111-1111-4111-8111-111111111111",
                        "taskId": "task-123",
                        "executionStatus": "idle",
                        "ready": {"ready": True},
                    },
                ),
            )
            receipt = write_run_receipt(loaded, redaction_result, Path(tmp) / "receipts")
            content = receipt.read_text(encoding="utf-8")
            self.assertIn("project-analysis", content)
            self.assertIn("11111111-1111-4111-8111-111111111111", content)
            self.assertIn("task-123", content)
            self.assertIn("\"runMode\": \"run-agent\"", content)
            self.assertIn("durationMs", content)
            self.assertNotIn("supersecret", content)
            self.assertEqual(result.workflow.name, "project-analysis")
            self.assertIn("inspect", result.stages)

    def test_receipt_does_not_include_llm_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            loaded = load_project(root)
            result = Orchestrator(MockProvider()).run(
                loaded,
                workflow_name="project-analysis",
                dry_run=True,
            )
            redaction_result = replace(
                result,
                provider_result=replace(
                    result.provider_result,
                    data={
                        "runMode": "run-agent",
                        "response": {
                            "agent": {
                                "llm": {
                                    "api_key": "plain-local-llm-key",
                                }
                            }
                        },
                        "runEndpointResult": {"success": True},
                    },
                    content='{"api_key":"plain-local-llm-key"}',
                ),
            )
            receipt = write_run_receipt(loaded, redaction_result, Path(tmp) / "receipts")
            content = receipt.read_text(encoding="utf-8")
            self.assertNotIn("plain-local-llm-key", content)
            self.assertIn("<redacted>", content)

    def test_receipt_names_are_unique(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            loaded = load_project(root)
            result = Orchestrator(MockProvider()).run(
                loaded,
                workflow_name="project-analysis",
                dry_run=True,
            )
            receipt_dir = Path(tmp) / "receipts"
            first = write_run_receipt(loaded, result, receipt_dir)
            second = write_run_receipt(loaded, result, receipt_dir)
            self.assertNotEqual(first, second)
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())

    def test_write_workflow_dry_run_allowed_on_protected_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            loaded = load_project(root)
            loaded.current_branch = "master"
            result = Orchestrator(MockProvider()).run(
                loaded,
                workflow_name="bug-fix-loop",
                dry_run=True,
            )
            self.assertTrue(result.provider_result.success)

    def test_run_agent_rejected_on_primary_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            loaded = load_project(root)
            loaded.current_branch = "feature/test"
            with self.assertRaises(ProjectConfigError):
                Orchestrator(MockProvider()).run(
                    loaded,
                    workflow_name="project-analysis",
                    run_mode="run-agent",
                )

    def test_read_only_agent_allowed_on_protected_primary_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            loaded = load_project(root)
            loaded.current_branch = "master"
            provider = HandsFreeCodeReadOnlyStub()
            result = Orchestrator(provider).run(
                loaded,
                workflow_name="project-analysis",
                run_mode="read-only-agent",
            )
            self.assertTrue(result.provider_result.success)
            self.assertEqual(result.provider_result.data["runtimeMode"], "run-agent")
            self.assertFalse(result.provider_result.data["writeAccess"])
            self.assertIsNotNone(provider.last_request)
            assert provider.last_request is not None
            self.assertIn("BEGIN BOUNDED PROJECT EVIDENCE", provider.last_request.prompt)
            self.assertIn("package.json", provider.last_request.prompt)
            self.assertIn("Next.js", provider.last_request.prompt)
            self.assertEqual(
                result.provider_result.data["analysisGroundingValidation"]["status"],
                "passed",
            )

    def test_read_only_agent_rejects_write_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            loaded = load_project(root)
            with self.assertRaises(WorkflowError):
                Orchestrator(HandsFreeCodeReadOnlyStub()).run(
                    loaded,
                    workflow_name="bug-fix-loop",
                    run_mode="read-only-agent",
                )

    def test_read_only_agent_rejects_non_handsfreecode_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            loaded = load_project(root)
            with self.assertRaises(WorkflowError):
                Orchestrator(MockProvider()).run(
                    loaded,
                    workflow_name="project-analysis",
                    run_mode="read-only-agent",
                )

    def test_read_only_agent_does_not_call_provider_without_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            (root / "package.json").unlink()
            (root / "tsconfig.json").unlink()
            (root / ".github" / "workflows" / "ci.yml").unlink()
            loaded = load_project(root)
            provider = HandsFreeCodeReadOnlyStub()
            with self.assertRaises(WorkflowError):
                Orchestrator(provider).run(
                    loaded,
                    workflow_name="project-analysis",
                    run_mode="read-only-agent",
                )
            self.assertIsNone(provider.last_request)

    def test_read_only_agent_fails_closed_when_analysis_is_ungrounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            loaded = load_project(root)
            provider = HandsFreeCodeReadOnlyStub(
                "Node.js, Next.js, TypeScript, requirements.txt, and 30% test coverage."
            )
            result = Orchestrator(provider).run(
                loaded,
                workflow_name="project-analysis",
                run_mode="read-only-agent",
            )
            self.assertFalse(result.provider_result.success)
            self.assertEqual(result.provider_result.error_type, ProviderErrorType.INVALID_RESPONSE)
            self.assertEqual(
                result.provider_result.data["providerExecutionValidation"]["status"],
                "passed",
            )
            self.assertEqual(
                result.provider_result.data["analysisGroundingValidation"]["status"],
                "failed",
            )

    def test_read_only_agent_receipt_records_outer_and_runtime_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            loaded = load_project(root)
            result = Orchestrator(HandsFreeCodeReadOnlyStub()).run(
                loaded,
                workflow_name="project-analysis",
                run_mode="read-only-agent",
            )
            receipt = write_run_receipt(loaded, result, Path(tmp) / "receipts")
            content = receipt.read_text(encoding="utf-8")
            payload = json.loads(content)
            self.assertIn('"runMode": "read-only-agent"', content)
            self.assertIn('"runtimeMode": "run-agent"', content)
            self.assertIn('"runtimeProvider": "ollama"', content)
            self.assertIn('"tokenUsage": 31', content)
            self.assertIn("Evidence-backed technology summary", content)
            self.assertGreater(payload["evidenceManifest"]["fileCount"], 0)
            self.assertEqual(
                payload["validationResult"]["evidenceCollection"]["status"],
                "passed",
            )
            self.assertEqual(
                payload["validationResult"]["analysisGrounding"]["status"],
                "passed",
            )

    def test_unsupported_run_mode_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            loaded = load_project(root)
            with self.assertRaises(WorkflowError):
                Orchestrator(MockProvider()).run(
                    loaded,
                    workflow_name="project-analysis",
                    run_mode="bad-mode",
                )


if __name__ == "__main__":
    unittest.main()
