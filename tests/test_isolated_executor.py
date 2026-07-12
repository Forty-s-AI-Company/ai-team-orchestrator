from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from ai_team.core.github_gate import evaluate_github_action
from ai_team.core.isolated_executor import run_in_disposable_worktree
from ai_team.core.project_loader import load_project
from ai_team.providers import MockProvider


def init_committed_project(root: Path, allow_git_push: bool = False) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "AI Team Test"], cwd=root, check=True)
    (root / ".gitignore").write_text("reports/\nlogs/\n", encoding="utf-8")
    ai_team = root / ".ai-team"
    ai_team.mkdir()
    (ai_team / "project.yaml").write_text(
        f"""project:
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
  allow_git_push: {str(allow_git_push).lower()}
  allow_deploy: false
  allow_database_migration: false
  allow_database_seed: false
  allow_destructive_commands: false
""",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", ".gitignore", ".ai-team/project.yaml"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True)


class IsolatedExecutorTests(unittest.TestCase):
    def test_write_workflow_runs_in_disposable_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root)
            receipt_dir = Path(tmp) / "receipts"

            result = run_in_disposable_worktree(
                source_project_path=root,
                provider=MockProvider(),
                workflow_name="bug-fix-loop",
                workspace_allowlist=[tmp],
                receipt_dir=receipt_dir,
                worktree_parent=Path(tmp),
                keep_worktree=True,
            )

            self.assertTrue(result.workflow_result.provider_result.success)
            self.assertTrue(result.worktree_path.exists())
            self.assertTrue((result.worktree_path / ".git").is_file())
            self.assertTrue(result.run_receipt.exists())
            self.assertTrue(result.executor_receipt.exists())
            payload = json.loads(result.executor_receipt.read_text(encoding="utf-8"))
            self.assertEqual(payload["workflow"], "bug-fix-loop")
            self.assertTrue(payload["gitPolicy"]["allowed"], payload["gitPolicy"]["reasons"])

    def test_isolated_executor_rejects_read_only_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root)
            with self.assertRaises(ValueError):
                run_in_disposable_worktree(
                    source_project_path=root,
                    provider=MockProvider(),
                    workflow_name="project-analysis",
                    workspace_allowlist=[tmp],
                    receipt_dir=Path(tmp) / "receipts",
                    worktree_parent=Path(tmp),
                )

    def test_github_gate_push_without_policy_is_external_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root, allow_git_push=False)
            loaded = load_project(root, allowlist=[tmp])
            loaded.current_branch = "feature/test"

            decision = evaluate_github_action(loaded, "push", dry_run=True)

            self.assertFalse(decision.allowed)
            self.assertTrue(decision.external_required)
            self.assertIn("allow git push", " ".join(decision.reasons).lower())

    def test_github_gate_pr_requires_validation_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root, allow_git_push=True)
            loaded = load_project(root, allowlist=[tmp])
            loaded.current_branch = "feature/test"

            decision = evaluate_github_action(loaded, "pr", dry_run=True)

            self.assertFalse(decision.allowed)
            self.assertIn("validation log hash", " ".join(decision.reasons))

    def test_github_gate_merge_always_external_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root, allow_git_push=True)
            loaded = load_project(root, allowlist=[tmp])
            loaded.current_branch = "feature/test"

            decision = evaluate_github_action(loaded, "merge", dry_run=True, validation_log_hash="abc123")

            self.assertFalse(decision.allowed)
            self.assertTrue(decision.external_required)
            self.assertIn("branch protection", " ".join(decision.reasons))


if __name__ == "__main__":
    unittest.main()
