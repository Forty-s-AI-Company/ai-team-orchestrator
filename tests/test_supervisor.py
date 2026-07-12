from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from ai_team.core.supervisor import SupervisorOptions, run_supervisor
from ai_team.providers import MockProvider


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


if __name__ == "__main__":
    unittest.main()
