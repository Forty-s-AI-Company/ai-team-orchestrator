from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from ai_team.core.github_executor import GitHubExecutionOptions, execute_github_action, sanitize_branch_name
from ai_team.core.github_gate import evaluate_github_action
from ai_team.core.isolated_executor import run_in_disposable_worktree
from ai_team.core.project_loader import load_project
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

            result = execute_github_action(
                loaded,
                GitHubExecutionOptions(action="push", dry_run=False, receipt_path=receipt, branch_name="ai-team/test"),
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
