from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from ai_team.core.git_policy import evaluate_git_action
from ai_team.core.project_loader import load_project


def init_git_project(root: Path, allow_git_push: bool = False) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    (root / ".gitignore").write_text("ignored.txt\nreports/\nlogs/\n", encoding="utf-8")
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


def make_disposable_worktree(root: Path) -> None:
    git_marker = root / ".git"
    git_marker_dir = root / ".git-dir"
    if git_marker.is_dir():
        git_marker.rename(git_marker_dir)
    git_marker.write_text(f"gitdir: {git_marker_dir.as_posix()}\n", encoding="utf-8")


class GitPolicyTests(unittest.TestCase):
    def test_commit_denied_on_primary_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            loaded = load_project(root, allowlist=[tmp])
            loaded.current_branch = "feature/test"

            decision = evaluate_git_action(loaded, "commit", candidate_files=["README.md"])

            self.assertFalse(decision.allowed)
            self.assertIn("disposable linked worktree", " ".join(decision.reasons))

    def test_commit_allowed_on_disposable_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            make_disposable_worktree(root)
            (root / "README.md").write_text("ok\n", encoding="utf-8")
            loaded = load_project(root, allowlist=[tmp])
            loaded.current_branch = "feature/test"

            decision = evaluate_git_action(loaded, "commit", candidate_files=["README.md"])

            self.assertTrue(decision.allowed, decision.reasons)
            self.assertEqual(decision.evidence["fileCheck"]["inspected"], ["README.md"])

    def test_protected_branch_blocks_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            make_disposable_worktree(root)
            loaded = load_project(root, allowlist=[tmp])
            loaded.current_branch = "master"

            decision = evaluate_git_action(loaded, "commit")

            self.assertFalse(decision.allowed)
            self.assertIn("protected branch", " ".join(decision.reasons))

    def test_ignored_runtime_artifact_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            make_disposable_worktree(root)
            (root / "reports").mkdir()
            (root / "reports" / "run.json").write_text("{}\n", encoding="utf-8")
            loaded = load_project(root, allowlist=[tmp])
            loaded.current_branch = "feature/test"

            decision = evaluate_git_action(loaded, "commit", candidate_files=["reports/run.json"])

            self.assertFalse(decision.allowed)
            self.assertIn("runtime artifact", " ".join(decision.reasons))

    def test_secret_candidate_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            make_disposable_worktree(root)
            (root / "config.txt").write_text("api_key = super-secret-value\n", encoding="utf-8")
            loaded = load_project(root, allowlist=[tmp])
            loaded.current_branch = "feature/test"

            decision = evaluate_git_action(loaded, "commit", candidate_files=["config.txt"])

            self.assertFalse(decision.allowed)
            self.assertIn("secret", " ".join(decision.reasons))

    def test_typescript_secret_named_type_annotation_is_not_a_secret_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            make_disposable_worktree(root)
            (root / "component.tsx").write_text(
                "type Props = { csrfToken: string; apiKey?: string; };\n",
                encoding="utf-8",
            )
            loaded = load_project(root, allowlist=[tmp])
            loaded.current_branch = "feature/test"

            decision = evaluate_git_action(loaded, "commit", candidate_files=["component.tsx"])

            self.assertTrue(decision.allowed, decision.reasons)

    def test_tracked_candidate_scans_only_new_lines_for_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "AI Team Test"], cwd=root, check=True)
            candidate = root / "mfa.test.ts"
            candidate.write_text(
                "const secret = generateTotpSecret();\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "fixture"], cwd=root, check=True, capture_output=True)
            make_disposable_worktree(root)
            candidate.write_text(
                "const secret = generateTotpSecret();\n"
                "vi.stubEnv(\"CSRF_SECRET\", \"test-only-non-production-key-material\");\n",
                encoding="utf-8",
            )
            loaded = load_project(root, allowlist=[tmp])
            loaded.current_branch = "feature/test"

            decision = evaluate_git_action(loaded, "commit", candidate_files=["mfa.test.ts"])

            self.assertTrue(decision.allowed, decision.reasons)

    def test_tracked_candidate_still_blocks_new_secret_assignments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "AI Team Test"], cwd=root, check=True)
            candidate = root / "config.txt"
            candidate.write_text("safe=true\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "fixture"], cwd=root, check=True, capture_output=True)
            make_disposable_worktree(root)
            candidate.write_text(
                "safe=true\napi_key = super-secret-value\n",
                encoding="utf-8",
            )
            loaded = load_project(root, allowlist=[tmp])
            loaded.current_branch = "feature/test"

            decision = evaluate_git_action(loaded, "commit", candidate_files=["config.txt"])

            self.assertFalse(decision.allowed)
            self.assertIn("secret", " ".join(decision.reasons))

    def test_push_and_pr_require_external_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            make_disposable_worktree(root)
            loaded = load_project(root, allowlist=[tmp])
            loaded.current_branch = "feature/test"

            push = evaluate_git_action(loaded, "push")
            pr = evaluate_git_action(loaded, "pr")

            self.assertFalse(push.allowed)
            self.assertTrue(push.external_required)
            self.assertFalse(pr.allowed)
            self.assertTrue(pr.external_required)

    def test_pr_allowed_on_disposable_worktree_when_push_policy_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root, allow_git_push=True)
            make_disposable_worktree(root)
            loaded = load_project(root, allowlist=[tmp])
            loaded.current_branch = "feature/test"

            decision = evaluate_git_action(loaded, "pr")

            self.assertTrue(decision.allowed, decision.reasons)
            self.assertFalse(decision.external_required)


if __name__ == "__main__":
    unittest.main()
