from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from ai_team.core.orchestrator import Orchestrator
from ai_team.core.project_loader import load_project
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

safety:
  allow_git_push: false
  allow_deploy: false
  allow_database_migration: false
  allow_database_seed: false
  allow_destructive_commands: false
""",
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
            self.assertEqual(result.workflow.name, "project-analysis")
            self.assertIn("inspect", result.stages)

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


if __name__ == "__main__":
    unittest.main()
