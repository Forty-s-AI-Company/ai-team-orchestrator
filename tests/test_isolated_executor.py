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
from ai_team.providers.base import BaseProvider, ProviderRequest, ProviderResult


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


def make_disposable_worktree_marker(root: Path) -> None:
    git_marker = root / ".git"
    git_marker_dir = root / ".git-dir"
    if git_marker.is_dir():
        git_marker.rename(git_marker_dir)
    git_marker.write_text(f"gitdir: {git_marker_dir.as_posix()}\n", encoding="utf-8")


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

    def test_auto_commit_commits_safe_changes_in_disposable_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root)

            result = run_in_disposable_worktree(
                source_project_path=root,
                provider=_WritingProvider("notes/change.md", "safe change\n"),
                workflow_name="bug-fix-loop",
                workspace_allowlist=[tmp],
                receipt_dir=Path(tmp) / "receipts",
                worktree_parent=Path(tmp),
                keep_worktree=True,
                auto_commit=True,
                commit_message="chore(ai-team): test safe change",
            )

            self.assertTrue(result.commit_result["committed"], result.commit_result)
            log = subprocess.run(
                ["git", "log", "-1", "--pretty=%s"],
                cwd=result.worktree_path,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(log.stdout.strip(), "chore(ai-team): test safe change")

    def test_auto_commit_blocks_secret_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root)

            result = run_in_disposable_worktree(
                source_project_path=root,
                provider=_WritingProvider("config.txt", "api_key = plain-secret-value\n"),
                workflow_name="bug-fix-loop",
                workspace_allowlist=[tmp],
                receipt_dir=Path(tmp) / "receipts",
                worktree_parent=Path(tmp),
                keep_worktree=True,
                auto_commit=True,
            )

            self.assertFalse(result.commit_result["committed"])
            self.assertIn("policy denied", result.commit_result["reason"])

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

    def test_github_gate_pr_dry_run_allowed_when_policy_inputs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root, allow_git_push=True)
            make_disposable_worktree_marker(root)
            loaded = load_project(root, allowlist=[tmp])
            loaded.current_branch = "feature/test"

            decision = evaluate_github_action(loaded, "pr", dry_run=True, validation_log_hash="abc123")

            self.assertTrue(decision.allowed, decision.reasons)
            self.assertFalse(decision.external_required)


class _WritingProvider(BaseProvider):
    name = "writing-test"

    def __init__(self, relative_path: str, content: str) -> None:
        self.relative_path = relative_path
        self.content = content

    def ready(self) -> bool:
        return True

    def run(self, request: ProviderRequest) -> ProviderResult:
        target = request.project_root / self.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.content, encoding="utf-8")
        return ProviderResult(
            provider=self.name,
            success=True,
            content="wrote file",
            data={"runMode": request.run_mode},
        )


if __name__ == "__main__":
    unittest.main()
