from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from ai_team.core.ci_monitor import classify_failure, monitor_pull_request, write_repair_completion_receipt


class CiMonitorTests(unittest.TestCase):
    def test_failed_dependency_check_creates_restricted_repair_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = _SequenceRunner(
                pr_payloads=[_pr_payload(check_status="COMPLETED", conclusion="FAILURE")],
                failed_log="npm ci failed: package.json and package-lock.json are not in sync",
            )

            result = monitor_pull_request(
                root,
                "example/project",
                "1",
                root / "reports",
                runner=runner,
            )

            self.assertEqual(result.status, "failed_repairable")
            self.assertFalse(result.merge_ready)
            self.assertIsNotNone(result.repair_task_path)
            task = json.loads(result.repair_task_path.read_text(encoding="utf-8"))
            self.assertEqual(task["writeAllowlist"], ["package-lock.json"])
            self.assertFalse(task["git"]["mergeAllowed"])
            self.assertFalse(task["autoExecutable"])
            self.assertEqual(task["expectedHeadSha"], "abc123")
            self.assertNotIn("package-lock.json are not in sync", json.dumps(task))

    def test_monitor_waits_for_pending_check_then_records_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = _SequenceRunner(
                pr_payloads=[
                    _pr_payload(check_status="IN_PROGRESS", conclusion=""),
                    _pr_payload(check_status="COMPLETED", conclusion="FAILURE"),
                ],
                failed_log="npm ci lock file mismatch",
            )
            sleeps: list[float] = []

            result = monitor_pull_request(
                root,
                "example/project",
                "1",
                root / "reports",
                wait_seconds=5,
                poll_seconds=1,
                runner=runner,
                sleeper=sleeps.append,
            )

            self.assertEqual(result.status, "failed_repairable")
            self.assertEqual(sleeps, [1])
            self.assertEqual(runner.pr_calls, 2)

    def test_passed_checks_still_require_review_and_clean_merge_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = _pr_payload(check_status="COMPLETED", conclusion="SUCCESS")
            runner = _SequenceRunner(pr_payloads=[payload])

            result = monitor_pull_request(root, "example/project", "1", root / "reports", runner=runner)

            self.assertEqual(result.status, "passed")
            self.assertFalse(result.merge_ready)
            self.assertIn("approved review is required", result.evidence["blockers"])

    def test_development_policy_can_explicitly_waive_human_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = _pr_payload(check_status="COMPLETED", conclusion="SUCCESS")
            payload["mergeStateStatus"] = "CLEAN"
            runner = _SequenceRunner(pr_payloads=[payload])

            result = monitor_pull_request(
                root,
                "example/project",
                "1",
                root / "reports",
                runner=runner,
                require_approved_review=False,
            )

            self.assertEqual(result.status, "passed")
            self.assertTrue(result.merge_ready)
            self.assertNotIn("approved review is required", result.evidence["blockers"])

    def test_failure_classification_covers_all_required_categories(self) -> None:
        self.assertEqual(
            classify_failure({"name": "quality", "workflow": "CI"}, "npm ci package-lock mismatch"),
            "product_dependency_failure",
        )
        self.assertEqual(
            classify_failure({"name": "Vercel", "workflow": ""}, "deployment failed"),
            "external_service_failure",
        )
        self.assertEqual(
            classify_failure({"name": "AI Team receipt", "workflow": "control"}, "invalid receipt"),
            "control_plane_failure",
        )

    def test_completion_receipt_binds_exact_repair_commit_and_passed_ci(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "package-lock.json").write_text('{"lockfileVersion":3}\n', encoding="utf-8")
            subprocess.run(["git", "add", "package-lock.json"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "base"], cwd=root, check=True, capture_output=True)
            base_sha = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
            ).stdout.strip()
            source_evidence = Path(tmp) / "failed.json"
            source_evidence.write_text('{"status":"failed_repairable"}\n', encoding="utf-8")
            task = Path(tmp) / "task.json"
            task.write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "maxAttempts": 1,
                        "expectedHeadSha": base_sha,
                        "sourceEvidencePath": str(source_evidence),
                        "sourceEvidenceHash": _hash(source_evidence),
                        "writeAllowlist": ["package-lock.json"],
                    }
                ),
                encoding="utf-8",
            )
            (root / "package-lock.json").write_text('{"lockfileVersion":3,"fixed":true}\n', encoding="utf-8")
            subprocess.run(["git", "add", "package-lock.json"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "fix lock"], cwd=root, check=True, capture_output=True)
            final_evidence = Path(tmp) / "passed.json"
            final_evidence.write_text('{"status":"passed"}\n', encoding="utf-8")

            receipt = write_repair_completion_receipt(root, task, final_evidence, Path(tmp) / "reports")

            payload = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(payload["sourceCommitSha"], base_sha)
            self.assertEqual(payload["changedFiles"], ["package-lock.json"])
            self.assertTrue(payload["validationResult"]["success"])


def _pr_payload(check_status: str, conclusion: str) -> dict:
    return {
        "url": "https://example.test/pull/1",
        "state": "OPEN",
        "isDraft": False,
        "mergeStateStatus": "UNSTABLE",
        "reviewDecision": "",
        "headRefName": "ai-team/test",
        "headRefOid": "abc123",
        "baseRefName": "master",
        "statusCheckRollup": [
            {
                "__typename": "CheckRun",
                "name": "quality",
                "workflowName": "CI",
                "status": check_status,
                "conclusion": conclusion,
                "detailsUrl": "https://github.com/example/project/actions/runs/123/job/456",
            }
        ],
    }


class _SequenceRunner:
    def __init__(self, pr_payloads: list[dict], failed_log: str = "") -> None:
        self.pr_payloads = pr_payloads
        self.failed_log = failed_log
        self.pr_calls = 0

    def __call__(self, args: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
        if args[:3] == ["gh", "pr", "view"]:
            index = min(self.pr_calls, len(self.pr_payloads) - 1)
            self.pr_calls += 1
            return subprocess.CompletedProcess(args, 0, json.dumps(self.pr_payloads[index]), "")
        if args[:3] == ["gh", "run", "view"]:
            return subprocess.CompletedProcess(args, 0, self.failed_log, "")
        return subprocess.CompletedProcess(args, 1, "", "unexpected command")


def _hash(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
