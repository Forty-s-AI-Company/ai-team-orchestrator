from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from ai_team.core.supervisor import SupervisorOptions, run_supervisor
from ai_team.providers import HandsFreeCodeProvider, HandsFreeCodeSettings, MockProvider, ProviderErrorType
from ai_team.providers.base import BaseProvider, ProviderRequest, ProviderResult


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
  test: npm run test

safety:
  allow_git_push: false
  allow_deploy: false
  allow_database_migration: false
  allow_database_seed: false
  allow_destructive_commands: false
""",
        encoding="utf-8",
    )


class SupervisorTests(unittest.TestCase):
    def test_supervisor_once_writes_structured_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            report_dir = Path(tmp) / "reports"
            summary = run_supervisor(
                SupervisorOptions(
                    project_path=root,
                    provider=MockProvider(),
                    once=True,
                    report_dir=report_dir,
                    workspace_allowlist=[tmp],
                )
            )
            self.assertEqual(summary.completed_cycles, 1)
            self.assertEqual(summary.stopped_reason, "once")
            self.assertEqual(len(summary.report_paths), 1)
            report = json.loads(summary.report_paths[0].read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "completed")
            self.assertIn("git-commit-evidence", [stage["name"] for stage in report["stages"]])

    def test_supervisor_reports_project_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            report_dir = Path(tmp) / "reports"
            summary = run_supervisor(
                SupervisorOptions(
                    project_path=root,
                    provider=MockProvider(),
                    once=True,
                    report_dir=report_dir,
                    workspace_allowlist=[tmp],
                )
            )
            report = json.loads(summary.report_paths[0].read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "attention_required")
            discovery = next(stage for stage in report["stages"] if stage["name"] == "discovery")
            self.assertFalse(discovery["ok"])
            self.assertIn("missing project profile", discovery["details"]["projectError"])

    def test_handsfreecode_supervisor_cycle_reports_provider_native_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            report_dir = Path(tmp) / "reports"
            summary = run_supervisor(
                SupervisorOptions(
                    project_path=root,
                    provider=_StaticProvider(
                        ProviderResult(
                            provider="handsfreecode",
                            success=True,
                            content="ok",
                            data={
                                "conversationId": "conv_test",
                                "taskId": "task_test",
                                "runtimeProvider": "mock",
                            },
                        )
                    ),
                    once=True,
                    report_dir=report_dir,
                    workspace_allowlist=[tmp],
                )
            )
            report = json.loads(summary.report_paths[0].read_text(encoding="utf-8"))
            qa = next(stage for stage in report["stages"] if stage["name"] == "qa-handoff")
            self.assertEqual(qa["details"]["provider"], "handsfreecode")
            self.assertEqual(qa["details"]["runtimeProvider"], "mock")
            self.assertFalse(qa["details"]["masqueradeAsCodexOrAntigravity"])

    def test_handsfreecode_unavailable_recovery_is_external_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            report_dir = Path(tmp) / "reports"
            state_path = Path(tmp) / "state.json"
            summary = run_supervisor(
                SupervisorOptions(
                    project_path=root,
                    provider=HandsFreeCodeProvider(
                        HandsFreeCodeSettings(base_url="http://127.0.0.1:9"),
                        session_key="test-session",
                    ),
                    once=True,
                    report_dir=report_dir,
                    state_path=state_path,
                    workspace_allowlist=[tmp],
                )
            )
            report = json.loads(summary.report_paths[0].read_text(encoding="utf-8"))
            auto_cycle = next(stage for stage in report["stages"] if stage["name"] == "auto-cycle")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertTrue(auto_cycle["ok"])
            self.assertEqual(auto_cycle["details"]["errorType"], "external_required")
            self.assertEqual(state["nextAction"], "external-required")

    def test_codex_quota_fallback_uses_ollama_without_masquerade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            summary = run_supervisor(
                SupervisorOptions(
                    project_path=root,
                    provider=_StaticProvider(
                        ProviderResult(
                            provider="codex",
                            success=False,
                            error_type=ProviderErrorType.RATE_LIMIT,
                            content="You've hit your usage limit. try again at Feb 23rd, 2026 9:01 PM.",
                        )
                    ),
                    workflow="project-analysis",
                    once=True,
                    report_dir=Path(tmp) / "reports",
                    workspace_allowlist=[tmp],
                )
            )
            report = json.loads(summary.report_paths[0].read_text(encoding="utf-8"))
            fallback = next(stage for stage in report["stages"] if stage["name"] == "fallback-policy")
            self.assertTrue(fallback["details"]["quotaExhausted"])
            self.assertTrue(fallback["details"]["fallbackAllowed"])
            self.assertEqual(fallback["details"]["fallbackProvider"], "ollama")
            self.assertFalse(fallback["details"]["masqueradeAsProvider"])

    def test_antigravity_quota_fallback_blocks_write_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            summary = run_supervisor(
                SupervisorOptions(
                    project_path=root,
                    provider=_StaticProvider(
                        ProviderResult(
                            provider="antigravity",
                            success=False,
                            error_type=ProviderErrorType.RATE_LIMIT,
                            content="Error: HTTP 429 Too Many Requests RESOURCE_EXHAUSTED Reset Time: 2026-07-12 08:00:00 (Local Time)",
                        )
                    ),
                    workflow="bug-fix-loop",
                    dry_run=True,
                    once=True,
                    report_dir=Path(tmp) / "reports",
                    workspace_allowlist=[tmp],
                )
            )
            report = json.loads(summary.report_paths[0].read_text(encoding="utf-8"))
            fallback = next(stage for stage in report["stages"] if stage["name"] == "fallback-policy")
            self.assertTrue(fallback["details"]["quotaExhausted"])
            self.assertFalse(fallback["details"]["fallbackAllowed"])
            self.assertIsNone(fallback["details"]["fallbackProvider"])
            self.assertFalse(fallback["details"]["masqueradeAsProvider"])

    def test_duplicate_resume_updates_state_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            state_path = Path(tmp) / "state.json"
            options = SupervisorOptions(
                project_path=root,
                provider=MockProvider(),
                once=True,
                report_dir=Path(tmp) / "reports",
                state_path=state_path,
                workspace_allowlist=[tmp],
            )
            first = run_supervisor(options)
            first_state = json.loads(state_path.read_text(encoding="utf-8"))
            second = run_supervisor(options)
            second_state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(first.completed_cycles, 1)
            self.assertEqual(second.completed_cycles, 1)
            self.assertGreater(second_state["revision"], first_state["revision"])
            self.assertEqual(second_state["nextAction"], "scheduled-discovery")

    def test_supervisor_state_redacts_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            state_path = Path(tmp) / "state.json"
            run_supervisor(
                SupervisorOptions(
                    project_path=root,
                    provider=_StaticProvider(
                        ProviderResult(
                            provider="codex",
                            success=False,
                            error_type=ProviderErrorType.EXTERNAL_REQUIRED,
                            data={"externalRequired": {"api_key": "plain-secret-value"}},
                        )
                    ),
                    once=True,
                    report_dir=Path(tmp) / "reports",
                    state_path=state_path,
                    workspace_allowlist=[tmp],
                )
            )
            state = state_path.read_text(encoding="utf-8")
            self.assertNotIn("plain-secret-value", state)
            self.assertIn("<redacted>", state)


class _StaticProvider(BaseProvider):
    name = "static"

    def __init__(self, result: ProviderResult) -> None:
        self.result = result
        self.name = result.provider

    def ready(self) -> bool:
        return self.result.success

    def run(self, request: ProviderRequest) -> ProviderResult:
        return self.result


if __name__ == "__main__":
    unittest.main()
