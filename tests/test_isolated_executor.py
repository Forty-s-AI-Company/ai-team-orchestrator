from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_team.core.github_executor import GitHubExecutionOptions, execute_github_action, sanitize_branch_name
from ai_team.core.github_gate import evaluate_github_action
from ai_team.core.isolated_executor import (
    _validation_env,
    prepare_dependency_link,
    remove_dependency_link,
    run_in_disposable_worktree,
    run_test_database_bootstrap,
    run_validation_commands,
)
from ai_team.core.project_loader import load_project
from ai_team.core.trusted_dev import TestDatabaseSettings, TrustedDevSettings
from ai_team.providers import MockProvider
from ai_team.providers.base import BaseProvider, ProviderRequest, ProviderResult
from ai_team.providers.write_smoke import WriteSmokeProvider


VALIDATION_HASH = "a" * 64
TEST_HASH = "b" * 64


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


def write_valid_receipt(root: Path, loaded) -> Path:
    receipt = root / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "projectPath": str(loaded.root),
                "commitSha": loaded.commit_sha,
                "validationResult": {"success": True},
            }
        ),
        encoding="utf-8",
    )
    return receipt


class IsolatedExecutorTests(unittest.TestCase):
    def test_trusted_next_dependencies_are_reused_until_manifest_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            worktree = root / "worktree"
            source_module = source / "node_modules" / "sample" / "index.js"
            source_module.parent.mkdir(parents=True)
            source_module.write_text("source\n", encoding="utf-8")
            (source / "package.json").write_text("{}\n", encoding="utf-8")
            worktree.mkdir()
            (worktree / "package.json").write_text(
                json.dumps({"dependencies": {"next": "16.2.10"}}),
                encoding="utf-8",
            )

            prepared = prepare_dependency_link(worktree, source, reuse_existing=True)
            copied_module = prepared / "sample" / "index.js"
            copied_module.write_text("generated in worktree\n", encoding="utf-8")
            second = prepare_dependency_link(worktree, source, reuse_existing=True)

            self.assertEqual(second, prepared)
            self.assertEqual(
                copied_module.read_text(encoding="utf-8"),
                "generated in worktree\n",
            )

    def test_next_dependencies_are_copied_inside_disposable_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            worktree = root / "worktree"
            source_module = source / "node_modules" / "sample" / "index.js"
            source_module.parent.mkdir(parents=True)
            source_module.write_text("source\n", encoding="utf-8")
            worktree.mkdir()
            (worktree / "package.json").write_text(
                json.dumps({"dependencies": {"next": "16.2.10"}}),
                encoding="utf-8",
            )

            prepared = prepare_dependency_link(worktree, source)

            self.assertEqual(prepared, worktree / "node_modules")
            self.assertFalse(prepared.is_symlink())
            copied_module = prepared / "sample" / "index.js"
            copied_module.write_text("worktree\n", encoding="utf-8")
            self.assertEqual(source_module.read_text(encoding="utf-8"), "source\n")

            remove_dependency_link(prepared)
            self.assertFalse(prepared.exists())

    def test_partial_dependency_cache_cannot_suppress_next_dependency_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            worktree = root / "worktree"
            source_binary = source / "node_modules" / ".bin" / "eslint"
            source_binary.parent.mkdir(parents=True)
            source_binary.write_text("eslint\n", encoding="utf-8")
            cache = worktree / "node_modules" / ".vite" / "cache"
            cache.mkdir(parents=True)
            (worktree / "package.json").write_text(
                json.dumps({"dependencies": {"next": "16.2.10"}}),
                encoding="utf-8",
            )

            prepared = prepare_dependency_link(worktree, source)

            self.assertEqual(prepared, worktree / "node_modules")
            self.assertEqual((prepared / ".bin" / "eslint").read_text(encoding="utf-8"), "eslint\n")
            self.assertTrue(cache.exists())
            remove_dependency_link(prepared)
            self.assertFalse(prepared.exists())

    def test_provider_sees_controlled_dependencies_before_it_runs(self) -> None:
        class DependencyAwareProvider(BaseProvider):
            saw_dependencies = False

            def ready(self) -> bool:
                return True

            def run(self, request: ProviderRequest) -> ProviderResult:
                self.saw_dependencies = (request.project_root / "node_modules" / ".bin" / "eslint").is_file()
                return ProviderResult(provider="mock", success=True, content="dependency inspection complete")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root)
            (root / ".gitignore").write_text("reports/\nlogs/\nnode_modules/\n", encoding="utf-8")
            (root / "package.json").write_text(
                json.dumps({"dependencies": {"next": "16.2.10"}}),
                encoding="utf-8",
            )
            subprocess.run(["git", "add", ".gitignore", "package.json"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "add package manifest"], cwd=root, check=True, capture_output=True)
            eslint = root / "node_modules" / ".bin" / "eslint"
            eslint.parent.mkdir(parents=True)
            eslint.write_text("eslint\n", encoding="utf-8")
            provider = DependencyAwareProvider()

            result = run_in_disposable_worktree(
                source_project_path=root,
                provider=provider,
                workflow_name="bug-fix-loop",
                workspace_allowlist=[tmp],
                receipt_dir=Path(tmp) / "receipts",
                worktree_parent=Path(tmp),
                keep_worktree=True,
            )

            self.assertTrue(provider.saw_dependencies)
            self.assertFalse((result.worktree_path / "node_modules").exists())

    def test_validation_command_unavailable_is_fail_closed_environment_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_validation_commands(
                Path(tmp),
                ["definitely-not-an-ai-team-command --version"],
                require_nonempty=True,
            )

            self.assertFalse(result["success"])
            self.assertEqual(result["kind"], "execution-environment")
            self.assertEqual(result["stopReason"], "validation-command-unavailable")
            self.assertEqual(result["commands"][0]["returnCode"], 127)

    def test_validation_environment_does_not_force_node_mode(self) -> None:
        with patch.dict(os.environ, {"NODE_ENV": "production"}, clear=False):
            env = _validation_env(None)

        self.assertEqual(env["CI"], "1")
        self.assertNotIn("NODE_ENV", env)

    def test_test_database_bootstrap_rejects_non_loopback_target(self) -> None:
        settings = TestDatabaseSettings(
            enabled=True,
            bootstrap_commands=("npm run db:migrate:deploy",),
        )
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"AI_TEAM_TEST_DATABASE_URL": "postgresql://user:pass@db.example.com/app_dev"},
            clear=False,
        ), patch("ai_team.core.isolated_executor.run_validation_commands") as runner:
            result, environment = run_test_database_bootstrap(Path(tmp), settings)

        self.assertFalse(result["success"])
        self.assertEqual(result["stopReason"], "test-database-target-rejected")
        self.assertEqual(environment, {})
        runner.assert_not_called()

    def test_validation_out_of_scope_mutation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root)
            mutator = Path(tmp) / "mutate.py"
            mutator.write_text(
                "from pathlib import Path\n"
                "path = Path('.ai-team/project.yaml')\n"
                "path.write_text(path.read_text(encoding='utf-8') + '# mutated\\n', encoding='utf-8')\n",
                encoding="utf-8",
            )
            base_sha = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            result = run_in_disposable_worktree(
                source_project_path=root,
                provider=_WritingProvider("notes/change.md", "safe change\n"),
                workflow_name="bug-fix-loop",
                workspace_allowlist=[tmp],
                receipt_dir=Path(tmp) / "receipts",
                worktree_parent=Path(tmp),
                keep_worktree=True,
                auto_commit=True,
                allowed_write_paths=["notes/change.md"],
                validation_commands=[f"{sys.executable} {mutator}"],
                require_validation=True,
            )

            self.assertFalse(result.commit_result["committed"])
            self.assertEqual(result.commit_result["validationResult"]["kind"], "candidate-integrity")
            self.assertEqual(
                result.commit_result["validationResult"]["stopReason"],
                "validation-mutated-candidate",
            )
            self.assertIn(".ai-team/project.yaml", result.git_policy["changedFiles"])
            self.assertFalse(result.git_policy["scopeCheck"]["allowed"])
            self.assertEqual(
                subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=result.worktree_path,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                base_sha,
            )
            receipt = json.loads(result.executor_receipt.read_text(encoding="utf-8"))
            self.assertTrue(receipt["providerSuccess"])
            self.assertFalse(receipt["validationResult"]["success"])
            self.assertEqual(receipt["validationResult"]["kind"], "candidate-integrity")

    def test_validation_in_scope_mutation_still_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root)
            mutator = Path(tmp) / "mutate.py"
            mutator.write_text(
                "from pathlib import Path\n"
                "path = Path('notes/change.md')\n"
                "path.write_text(path.read_text(encoding='utf-8') + 'validation rewrite\\n', encoding='utf-8')\n",
                encoding="utf-8",
            )

            result = run_in_disposable_worktree(
                source_project_path=root,
                provider=_WritingProvider("notes/change.md", "safe change\n"),
                workflow_name="bug-fix-loop",
                workspace_allowlist=[tmp],
                receipt_dir=Path(tmp) / "receipts",
                worktree_parent=Path(tmp),
                keep_worktree=True,
                auto_commit=True,
                allowed_write_paths=["notes/change.md"],
                validation_commands=[f"{sys.executable} {mutator}"],
                require_validation=True,
            )

            self.assertFalse(result.commit_result["committed"])
            self.assertTrue(result.git_policy["scopeCheck"]["allowed"])
            self.assertEqual(
                result.commit_result["validationResult"],
                {
                    "success": False,
                    "kind": "candidate-integrity",
                    "stopReason": "validation-mutated-candidate",
                },
            )

    def test_non_next_dependencies_keep_lightweight_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            worktree = root / "worktree"
            (source / "node_modules").mkdir(parents=True)
            worktree.mkdir()
            (worktree / "package.json").write_text(
                json.dumps({"dependencies": {"react": "19.0.0"}}),
                encoding="utf-8",
            )

            prepared = prepare_dependency_link(worktree, source)

            if os.name == "nt":
                self.assertTrue(prepared.exists())
            else:
                self.assertTrue(prepared.is_symlink())
            remove_dependency_link(prepared)
            self.assertFalse(prepared.exists())

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

    def test_worktree_ready_callback_runs_before_provider_execution(self) -> None:
        class CallbackAwareProvider(BaseProvider):
            name = "callback-aware"

            def ready(self) -> bool:
                return True

            def run(self, request: ProviderRequest) -> ProviderResult:
                self.assert_ready(request.project_root)
                return ProviderResult(provider=self.name, success=True, content="ready")

            def assert_ready(self, root: Path) -> None:
                if callbacks != [root]:
                    raise AssertionError("worktree callback did not run before provider")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root)
            callbacks: list[Path] = []

            result = run_in_disposable_worktree(
                source_project_path=root,
                provider=CallbackAwareProvider(),
                workflow_name="bug-fix-loop",
                workspace_allowlist=[tmp],
                receipt_dir=Path(tmp) / "receipts",
                worktree_parent=Path(tmp),
                keep_worktree=True,
                on_worktree_ready=callbacks.append,
            )

            self.assertEqual(callbacks, [result.worktree_path])

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

    def test_auto_commit_missing_git_identity_fails_closed_with_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root)
            subprocess.run(["git", "config", "--unset-all", "user.name"], cwd=root, check=True)
            subprocess.run(["git", "config", "--unset-all", "user.email"], cwd=root, check=True)

            with patch.dict(
                os.environ,
                {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull},
                clear=False,
            ):
                result = run_in_disposable_worktree(
                    source_project_path=root,
                    provider=_WritingProvider("notes/change.md", "safe change\n"),
                    workflow_name="bug-fix-loop",
                    workspace_allowlist=[tmp],
                    receipt_dir=Path(tmp) / "receipts",
                    worktree_parent=Path(tmp),
                    keep_worktree=True,
                    auto_commit=True,
                )

            self._assert_git_failure_receipt(result, "git-identity-missing")
            self.assertEqual(
                result.commit_result["missingIdentityFields"],
                ["user.name", "user.email"],
            )

    @unittest.skipIf(os.name == "nt", "POSIX executable Git hook fixture")
    def test_auto_commit_hook_rejection_fails_closed_and_redacts_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root)
            hook = root / ".git" / "hooks" / "pre-commit"
            sensitive_line = "api_" + "key = hook-" + "secret-value"
            hook.write_text(
                f"#!/bin/sh\necho '{sensitive_line}' >&2\nexit 1\n",
                encoding="utf-8",
            )
            hook.chmod(0o700)

            result = run_in_disposable_worktree(
                source_project_path=root,
                provider=_WritingProvider("notes/change.md", "safe change\n"),
                workflow_name="bug-fix-loop",
                workspace_allowlist=[tmp],
                receipt_dir=Path(tmp) / "receipts",
                worktree_parent=Path(tmp),
                keep_worktree=True,
                auto_commit=True,
            )

            self._assert_git_failure_receipt(result, "git-commit-failed")
            receipt_text = result.executor_receipt.read_text(encoding="utf-8")
            self.assertNotIn("hook-secret-value", receipt_text)
            self.assertIn("<redacted>", receipt_text)

    def test_auto_commit_git_add_failure_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root)

            def fail_add(cwd: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
                if args and args[0] == "add":
                    return subprocess.CompletedProcess(
                        ["git", *args],
                        128,
                        "",
                        "api_" + "key = " + ("z" * 4100),
                    )
                return subprocess.run(
                    ["git", *args],
                    cwd=cwd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

            with patch("ai_team.core.isolated_executor._run_git_attempt", side_effect=fail_add):
                result = run_in_disposable_worktree(
                    source_project_path=root,
                    provider=_WritingProvider("notes/change.md", "safe change\n"),
                    workflow_name="bug-fix-loop",
                    workspace_allowlist=[tmp],
                    receipt_dir=Path(tmp) / "receipts",
                    worktree_parent=Path(tmp),
                    keep_worktree=True,
                    auto_commit=True,
                )

            self._assert_git_failure_receipt(result, "git-add-failed")
            receipt_text = result.executor_receipt.read_text(encoding="utf-8")
            self.assertNotIn("z" * 100, receipt_text)
            self.assertIn("<redacted>", receipt_text)

    def test_auto_commit_git_commit_failure_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root)

            def fail_commit(cwd: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
                if args and args[0] == "commit":
                    return subprocess.CompletedProcess(
                        ["git", *args],
                        1,
                        "",
                        "synthetic commit failure",
                    )
                return subprocess.run(
                    ["git", *args],
                    cwd=cwd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

            with patch("ai_team.core.isolated_executor._run_git_attempt", side_effect=fail_commit):
                result = run_in_disposable_worktree(
                    source_project_path=root,
                    provider=_WritingProvider("notes/change.md", "safe change\n"),
                    workflow_name="bug-fix-loop",
                    workspace_allowlist=[tmp],
                    receipt_dir=Path(tmp) / "receipts",
                    worktree_parent=Path(tmp),
                    keep_worktree=True,
                    auto_commit=True,
                )

            self._assert_git_failure_receipt(result, "git-commit-failed")
            self.assertEqual(result.commit_result["returnCode"], 1)
            self.assertEqual(result.commit_result["gitOperation"], "commit")

    def test_repair_can_reuse_the_same_disposable_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root)
            first = run_in_disposable_worktree(
                source_project_path=root,
                provider=_WritingProvider("notes/change.md", "first\n"),
                workflow_name="bug-fix-loop",
                workspace_allowlist=[tmp],
                receipt_dir=Path(tmp) / "receipts",
                worktree_parent=Path(tmp),
                keep_worktree=True,
                auto_commit=True,
            )
            second = run_in_disposable_worktree(
                source_project_path=root,
                provider=_WritingProvider("notes/change.md", "second\n"),
                workflow_name="bug-fix-loop",
                workspace_allowlist=[tmp],
                receipt_dir=Path(tmp) / "receipts",
                worktree_parent=Path(tmp),
                keep_worktree=True,
                auto_commit=True,
                reuse_worktree_path=first.worktree_path,
            )
            self.assertEqual(second.worktree_path, first.worktree_path)
            self.assertTrue(second.commit_result["committed"], second.commit_result)

    def test_write_smoke_auto_commit_and_pr_dry_run_records_all_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root, allow_git_push=True)

            result = run_in_disposable_worktree(
                source_project_path=root,
                provider=WriteSmokeProvider(),
                workflow_name="bug-fix-loop",
                workspace_allowlist=[tmp],
                receipt_dir=Path(tmp) / "receipts",
                worktree_parent=Path(tmp),
                keep_worktree=True,
                auto_commit=True,
                github_action="pr",
                github_branch="ai-team/write-smoke-test",
                validation_log_hash=VALIDATION_HASH,
                test_evidence_hash=TEST_HASH,
            )

            self.assertTrue(result.commit_result["committed"], result.commit_result)
            self.assertIsNotNone(result.github_result)
            self.assertTrue(result.github_result["success"], result.github_result)
            self.assertEqual(result.github_result["branch"], "ai-team/write-smoke-test")
            self.assertIsNotNone(result.github_result["receiptHash"])
            self.assertRegex(result.github_result["secretScanHash"], r"^[0-9a-f]{64}$")
            self.assertEqual(result.github_result["validationLogHash"], VALIDATION_HASH)
            self.assertEqual(result.github_result["testEvidenceHash"], TEST_HASH)
            run_receipt = json.loads(result.run_receipt.read_text(encoding="utf-8"))
            self.assertEqual(run_receipt["commitSha"], result.commit_result["commitSha"])
            self.assertNotEqual(run_receipt["sourceCommitSha"], run_receipt["commitSha"])

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

    def test_provider_failure_cannot_commit_or_trigger_github_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root, allow_git_push=True)

            result = run_in_disposable_worktree(
                source_project_path=root,
                provider=_WritingFailureProvider("notes/partial.md", "partial output\n"),
                workflow_name="bug-fix-loop",
                workspace_allowlist=[tmp],
                receipt_dir=Path(tmp) / "receipts",
                worktree_parent=Path(tmp),
                keep_worktree=True,
                auto_commit=True,
                github_action="pr",
                validation_log_hash=VALIDATION_HASH,
                test_evidence_hash=TEST_HASH,
            )

            self.assertFalse(result.workflow_result.provider_result.success)
            self.assertFalse(result.commit_result["committed"])
            self.assertIn("provider validation failed", result.commit_result["reason"])
            self.assertFalse(result.github_result["success"])
            self.assertFalse(result.github_result["attempted"])

    def test_provider_failure_skips_expensive_deterministic_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root)
            with patch("ai_team.core.isolated_executor.run_validation_commands") as validator:
                result = run_in_disposable_worktree(
                    source_project_path=root,
                    provider=_WritingFailureProvider("notes/partial.md", "partial output\n"),
                    workflow_name="bug-fix-loop",
                    workspace_allowlist=[tmp],
                    receipt_dir=Path(tmp) / "receipts",
                    worktree_parent=Path(tmp),
                    keep_worktree=True,
                    auto_commit=True,
                    validation_commands=["npm run build"],
                    require_validation=True,
                )

            validator.assert_not_called()
            validation = result.commit_result["validation"]
            self.assertEqual(validation["kind"], "provider-execution")
            self.assertTrue(validation["skippedDeterministicValidation"])

    def test_trusted_validation_failure_creates_reusable_git_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root)
            failing_validation = Path(tmp) / "fail-validation.py"
            failing_validation.write_text("raise SystemExit(1)\n", encoding="utf-8")
            result = run_in_disposable_worktree(
                source_project_path=root,
                provider=_WritingProvider("notes/change.md", "work in progress\n"),
                workflow_name="bug-fix-loop",
                workspace_allowlist=[tmp],
                receipt_dir=Path(tmp) / "receipts",
                worktree_parent=Path(tmp),
                keep_worktree=True,
                auto_commit=True,
                allowed_write_paths=["notes/change.md"],
                validation_commands=[f"{sys.executable} {failing_validation}"],
                require_validation=True,
                trusted_dev=TrustedDevSettings(
                    enabled=True,
                    checkpoint_on_validation_failure=True,
                ),
            )

            self.assertTrue(result.commit_result["committed"], result.commit_result)
            self.assertTrue(result.commit_result["checkpoint"])
            self.assertFalse(result.commit_result["validationPassed"])
            self.assertEqual(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=result.worktree_path,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout,
                "",
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

    def test_github_gate_merge_requires_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root, allow_git_push=True)
            make_disposable_worktree_marker(root)
            loaded = load_project(root, allowlist=[tmp])
            loaded.current_branch = "feature/test"

            decision = evaluate_github_action(loaded, "merge", dry_run=True, validation_log_hash="abc123")

            self.assertFalse(decision.allowed)
            self.assertFalse(decision.external_required)
            self.assertIn("receipt hash", " ".join(decision.reasons))

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

    def test_github_gate_merge_allowed_when_policy_inputs_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root, allow_git_push=True)
            make_disposable_worktree_marker(root)
            loaded = load_project(root, allowlist=[tmp])
            loaded.current_branch = "feature/test"

            decision = evaluate_github_action(
                loaded,
                "merge",
                dry_run=True,
                validation_log_hash="validation-hash",
                receipt_hash="receipt-hash",
                secret_scan_hash="secret-scan-hash",
                test_evidence_hash="test-hash",
            )

            self.assertTrue(decision.allowed, decision.reasons)

    def test_github_executor_push_dry_run_requires_receipt_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root, allow_git_push=True)
            make_disposable_worktree_marker(root)
            loaded = load_project(root, allowlist=[tmp])
            loaded.current_branch = "feature/test"

            result = execute_github_action(loaded, GitHubExecutionOptions(action="push", dry_run=True))

            self.assertFalse(result.success)
            self.assertIn("receipt hash", " ".join(result.reasons))

    def test_github_executor_pr_dry_run_records_evidence_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root, allow_git_push=True)
            make_disposable_worktree_marker(root)
            loaded = load_project(root, allowlist=[tmp])
            loaded.current_branch = "feature/test"
            receipt = write_valid_receipt(root, loaded)

            result = execute_github_action(
                loaded,
                GitHubExecutionOptions(
                    action="pr",
                    dry_run=True,
                    validation_log_hash=VALIDATION_HASH,
                    receipt_path=receipt,
                    test_evidence_hash=TEST_HASH,
                ),
            )

            self.assertTrue(result.success, result.reasons)
            self.assertFalse(result.attempted)
            self.assertIsNotNone(result.receipt_hash)
            self.assertIsNotNone(result.secret_scan_hash)

    def test_github_executor_blocks_review_waiver_outside_development(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root, allow_git_push=True)
            make_disposable_worktree_marker(root)
            loaded = load_project(root, allowlist=[tmp])
            loaded.profile.project.stage = "production"
            loaded.current_branch = "feature/test"
            receipt = write_valid_receipt(root, loaded)

            with (
                patch("ai_team.core.github_gate.shutil.which", return_value="/usr/bin/gh"),
                patch(
                    "ai_team.core.github_gate._gh_auth_status",
                    return_value={"authenticated": True},
                ),
            ):
                result = execute_github_action(
                    loaded,
                    GitHubExecutionOptions(
                        action="merge",
                        dry_run=True,
                        validation_log_hash=VALIDATION_HASH,
                        receipt_path=receipt,
                        test_evidence_hash=TEST_HASH,
                        pr_identifier="123",
                        require_approved_review=False,
                    ),
                )

            self.assertFalse(result.success)
            self.assertFalse(result.attempted)
            self.assertIn("development-stage", " ".join(result.reasons))

    def test_github_executor_scans_committed_blob_not_working_tree_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root, allow_git_push=True)
            make_disposable_worktree_marker(root)
            secret_file = root / "committed.txt"
            secret_file.write_text("api_key = should-never-pass-scan\n", encoding="utf-8")
            subprocess.run(["git", "add", "committed.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "unsafe fixture"], cwd=root, check=True, capture_output=True)
            secret_file.write_text("safe working tree copy\n", encoding="utf-8")
            loaded = load_project(root, allowlist=[tmp])
            loaded.current_branch = "feature/test"
            receipt = write_valid_receipt(root, loaded)

            result = execute_github_action(
                loaded,
                GitHubExecutionOptions(
                    action="pr",
                    dry_run=True,
                    validation_log_hash=VALIDATION_HASH,
                    receipt_path=receipt,
                    test_evidence_hash=TEST_HASH,
                ),
            )

            self.assertFalse(result.success)
            self.assertIn("secret-like content", " ".join(result.reasons))

    def test_github_executor_allows_runtime_csrf_token_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root, allow_git_push=True)
            source_sha = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
            ).stdout.strip()
            (root / "page.tsx").write_text(
                "const element = <Form csrfToken={csrf.token} />;\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "add", "page.tsx"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "runtime reference"], cwd=root, check=True, capture_output=True)
            make_disposable_worktree_marker(root)
            loaded = load_project(root, allowlist=[tmp])
            loaded.current_branch = "feature/test"
            receipt = write_valid_receipt(root, loaded)
            receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
            receipt_payload["sourceCommitSha"] = source_sha
            receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")

            result = execute_github_action(
                loaded,
                GitHubExecutionOptions(
                    action="pr",
                    dry_run=True,
                    validation_log_hash=VALIDATION_HASH,
                    receipt_path=receipt,
                    test_evidence_hash=TEST_HASH,
                ),
            )

            self.assertTrue(result.success, result.reasons)

    def test_github_executor_allows_explicit_test_token_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root, allow_git_push=True)
            source_sha = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
            ).stdout.strip()
            (root / "component.test.tsx").write_text(
                'const props = { csrfToken: "csrf-test-token" };\n',
                encoding="utf-8",
            )
            subprocess.run(["git", "add", "component.test.tsx"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "test placeholder"], cwd=root, check=True, capture_output=True)
            make_disposable_worktree_marker(root)
            loaded = load_project(root, allowlist=[tmp])
            loaded.current_branch = "feature/test"
            receipt = write_valid_receipt(root, loaded)
            receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
            receipt_payload["sourceCommitSha"] = source_sha
            receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")

            result = execute_github_action(
                loaded,
                GitHubExecutionOptions(
                    action="pr",
                    dry_run=True,
                    validation_log_hash=VALIDATION_HASH,
                    receipt_path=receipt,
                    test_evidence_hash=TEST_HASH,
                ),
            )

            self.assertTrue(result.success, result.reasons)

    def test_github_executor_does_not_flag_unchanged_fixture_secret(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root, allow_git_push=True)
            (root / "fixture.yml").write_text("JOB_SECRET: ci-test-fixture\n", encoding="utf-8")
            subprocess.run(["git", "add", "fixture.yml"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "fixture"], cwd=root, check=True, capture_output=True)
            source_sha = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
            ).stdout.strip()
            (root / "safe.md").write_text("safe change\n", encoding="utf-8")
            subprocess.run(["git", "add", "safe.md"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "safe"], cwd=root, check=True, capture_output=True)
            make_disposable_worktree_marker(root)
            loaded = load_project(root, allowlist=[tmp])
            loaded.current_branch = "feature/test"
            receipt = write_valid_receipt(root, loaded)
            receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
            receipt_payload["sourceCommitSha"] = source_sha
            receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")

            result = execute_github_action(
                loaded,
                GitHubExecutionOptions(action="push", dry_run=True, receipt_path=receipt),
            )

            self.assertTrue(result.success, result.reasons)

    def test_github_executor_push_execute_runs_git_push(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root, allow_git_push=True)
            make_disposable_worktree_marker(root)
            loaded = load_project(root, allowlist=[tmp])
            loaded.current_branch = "feature/test"
            receipt = write_valid_receipt(root, loaded)
            commands: list[list[str]] = []

            def fake_runner(args: list[str], cwd: Path, timeout: int):
                commands.append(args)
                return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

            with (
                patch("ai_team.core.github_gate.shutil.which", return_value="/usr/bin/gh"),
                patch(
                    "ai_team.core.github_gate._gh_auth_status",
                    return_value={"authenticated": True},
                ),
            ):
                result = execute_github_action(
                    loaded,
                    GitHubExecutionOptions(
                        action="push",
                        dry_run=False,
                        receipt_path=receipt,
                        branch_name="ai-team/test",
                    ),
                    runner=fake_runner,
                )

            self.assertTrue(result.success, result.reasons)
            self.assertIn(["git", "push", "-u", "origin", "HEAD:ai-team/test"], commands)

    def test_sanitize_branch_name_blocks_protected_names(self) -> None:
        self.assertEqual(sanitize_branch_name("master"), "ai-team/master")

    def test_github_executor_merge_execute_checks_pr_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root, allow_git_push=True)
            make_disposable_worktree_marker(root)
            loaded = load_project(root, allowlist=[tmp])
            loaded.current_branch = "feature/test"
            receipt = write_valid_receipt(root, loaded)
            commands: list[list[str]] = []

            def fake_runner(args: list[str], cwd: Path, timeout: int):
                commands.append(args)
                if args[:3] == ["gh", "pr", "view"]:
                    return subprocess.CompletedProcess(
                        args=args,
                        returncode=0,
                        stdout=json.dumps(
                            {
                                "mergeStateStatus": "CLEAN",
                                "reviewDecision": "APPROVED",
                                "isDraft": False,
                                "headRefOid": loaded.commit_sha,
                                "statusCheckRollup": [
                                    {
                                        "__typename": "CheckRun",
                                        "status": "COMPLETED",
                                        "conclusion": "SUCCESS",
                                    }
                                ],
                            }
                        ),
                        stderr="",
                    )
                return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

            with (
                patch("ai_team.core.github_gate.shutil.which", return_value="/usr/bin/gh"),
                patch(
                    "ai_team.core.github_gate._gh_auth_status",
                    return_value={"authenticated": True},
                ),
            ):
                result = execute_github_action(
                    loaded,
                    GitHubExecutionOptions(
                        action="merge",
                        dry_run=False,
                        validation_log_hash=VALIDATION_HASH,
                        receipt_path=receipt,
                        test_evidence_hash=TEST_HASH,
                        pr_identifier="123",
                    ),
                    runner=fake_runner,
                )

            self.assertTrue(result.success, result.reasons)
            self.assertIn(["gh", "pr", "merge", "123", "--squash"], commands)

    def test_github_executor_merge_blocks_unapproved_pr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root, allow_git_push=True)
            make_disposable_worktree_marker(root)
            loaded = load_project(root, allowlist=[tmp])
            loaded.current_branch = "feature/test"
            receipt = write_valid_receipt(root, loaded)

            def fake_runner(args: list[str], cwd: Path, timeout: int):
                return subprocess.CompletedProcess(
                    args=args,
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "mergeStateStatus": "CLEAN",
                            "reviewDecision": "REVIEW_REQUIRED",
                            "isDraft": False,
                            "headRefOid": loaded.commit_sha,
                            "statusCheckRollup": [
                                {
                                    "__typename": "CheckRun",
                                    "status": "COMPLETED",
                                    "conclusion": "SUCCESS",
                                }
                            ],
                        }
                    ),
                    stderr="",
                )

            with (
                patch("ai_team.core.github_gate.shutil.which", return_value="/usr/bin/gh"),
                patch(
                    "ai_team.core.github_gate._gh_auth_status",
                    return_value={"authenticated": True},
                ),
            ):
                result = execute_github_action(
                    loaded,
                    GitHubExecutionOptions(
                        action="merge",
                        dry_run=False,
                        validation_log_hash=VALIDATION_HASH,
                        receipt_path=receipt,
                        test_evidence_hash=TEST_HASH,
                        pr_identifier="123",
                    ),
                    runner=fake_runner,
                )

            self.assertFalse(result.success)
            self.assertIn("approved review", " ".join(result.reasons))

    def test_github_executor_can_explicitly_waive_review_for_development(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root, allow_git_push=True)
            make_disposable_worktree_marker(root)
            loaded = load_project(root, allowlist=[tmp])
            loaded.current_branch = "feature/test"
            receipt = write_valid_receipt(root, loaded)

            def fake_runner(args: list[str], cwd: Path, timeout: int):
                if args[:3] == ["gh", "pr", "view"]:
                    return subprocess.CompletedProcess(
                        args=args,
                        returncode=0,
                        stdout=json.dumps(
                            {
                                "mergeStateStatus": "CLEAN",
                                "reviewDecision": "",
                                "isDraft": False,
                                "headRefOid": loaded.commit_sha,
                                "statusCheckRollup": [
                                    {
                                        "__typename": "CheckRun",
                                        "status": "COMPLETED",
                                        "conclusion": "SUCCESS",
                                    }
                                ],
                            }
                        ),
                        stderr="",
                    )
                return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

            with (
                patch("ai_team.core.github_gate.shutil.which", return_value="/usr/bin/gh"),
                patch(
                    "ai_team.core.github_gate._gh_auth_status",
                    return_value={"authenticated": True},
                ),
            ):
                result = execute_github_action(
                    loaded,
                    GitHubExecutionOptions(
                        action="merge",
                        dry_run=False,
                        validation_log_hash=VALIDATION_HASH,
                        receipt_path=receipt,
                        test_evidence_hash=TEST_HASH,
                        pr_identifier="123",
                        require_approved_review=False,
                    ),
                    runner=fake_runner,
                )

            self.assertTrue(result.success, result.reasons)

    def test_github_executor_merge_blocks_pending_checks_and_stale_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_committed_project(root, allow_git_push=True)
            make_disposable_worktree_marker(root)
            loaded = load_project(root, allowlist=[tmp])
            loaded.current_branch = "feature/test"
            receipt = write_valid_receipt(root, loaded)

            def fake_runner(args: list[str], cwd: Path, timeout: int):
                return subprocess.CompletedProcess(
                    args=args,
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "mergeStateStatus": "CLEAN",
                            "reviewDecision": "APPROVED",
                            "isDraft": False,
                            "headRefOid": "f" * 40,
                            "statusCheckRollup": [
                                {
                                    "__typename": "CheckRun",
                                    "status": "IN_PROGRESS",
                                    "conclusion": "",
                                }
                            ],
                        }
                    ),
                    stderr="",
                )

            result = execute_github_action(
                loaded,
                GitHubExecutionOptions(
                    action="merge",
                    dry_run=True,
                    validation_log_hash=VALIDATION_HASH,
                    receipt_path=receipt,
                    test_evidence_hash=TEST_HASH,
                    pr_identifier="123",
                ),
                runner=fake_runner,
            )

            self.assertFalse(result.success)
            self.assertIn("head SHA", " ".join(result.reasons))
            self.assertIn("pending", " ".join(result.reasons))

    def _assert_git_failure_receipt(self, result, stop_reason: str) -> None:
        self.assertTrue(result.workflow_result.provider_result.success)
        self.assertTrue(result.run_receipt.exists())
        self.assertTrue(result.executor_receipt.exists())
        self.assertFalse(result.commit_result["committed"])
        self.assertEqual(result.commit_result["stopReason"], stop_reason)
        self.assertFalse(result.commit_result["validationResult"]["success"])
        self.assertEqual(result.commit_result["validationResult"]["kind"], "git-commit")
        payload = json.loads(result.executor_receipt.read_text(encoding="utf-8"))
        self.assertTrue(payload["providerSuccess"])
        self.assertFalse(payload["validationResult"]["success"])
        self.assertEqual(payload["validationResult"]["kind"], "git-commit")
        self.assertEqual(payload["validationResult"]["stopReason"], stop_reason)
        self.assertEqual(payload["stopReason"], stop_reason)


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


class _WritingFailureProvider(_WritingProvider):
    name = "writing-failure-test"

    def run(self, request: ProviderRequest) -> ProviderResult:
        super().run(request)
        return ProviderResult(
            provider=self.name,
            success=False,
            content="provider validation failed after partial write",
            data={"runMode": request.run_mode},
        )


if __name__ == "__main__":
    unittest.main()
