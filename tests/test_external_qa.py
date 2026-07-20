from __future__ import annotations

import concurrent.futures
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_team.core.external_qa import (
    HUMAN_ATTESTATION_REQUIRED,
    MANUAL_ATTESTATION_ONLY,
    SCHEMA,
    run_external_qa,
)
from ai_team.core.project_loader import load_project


class ExternalQATests(unittest.TestCase):
    def _project(self, enabled: bool = True) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / ".git").mkdir()
        (root / ".ai-team").mkdir()
        (root / ".ai-team" / "project.yaml").write_text(
            f"""project:
  name: sample
  root: .
external_qa:
  enabled: {str(enabled).lower()}
  command: npm run qa:payuni:sandbox
""",
            encoding="utf-8",
        )
        return root

    def test_enabled_policy_builds_fixed_human_review_requirement_without_io(self) -> None:
        root = self._project()
        env_file = root / ".env.local"
        env_contents = "PAYUNI_SANDBOX_QA_ENABLED=true\nsecret=do-not-read\n"
        env_file.write_text(env_contents, encoding="utf-8")
        report_dir = root / "reports"
        receipt = report_dir / "external-qa-aaaaaaaaaaaa.json"
        report_dir.mkdir()
        original_receipt = b'{"historic":"receipt"}\n'
        receipt.write_bytes(original_receipt)
        loaded = load_project(root)

        with (
            patch("ai_team.core.external_qa.Path.is_file", side_effect=AssertionError("environment access")),
            patch("ai_team.core.external_qa.Path.read_text", side_effect=AssertionError("environment access")),
            patch("ai_team.core.external_qa.Path.mkdir", side_effect=AssertionError("report write")),
            patch("ai_team.core.external_qa.Path.write_text", side_effect=AssertionError("report write")),
        ):
            result = run_external_qa(
                loaded,
                "a" * 40,
                report_dir,
                prior={"status": "passed", "checks": {"nested": [object()]}},
            )

        self.assertEqual(result.status, "review-required")
        self.assertIsNone(result.receipt_path)
        self.assertEqual(
            result.result,
            {
                "schema": SCHEMA,
                "revision": "a" * 40,
                "executionMode": MANUAL_ATTESTATION_ONLY,
                "executionAttempted": False,
                "reviewerRole": "delivery-qa",
                "status": "review-required",
                "reason": HUMAN_ATTESTATION_REQUIRED,
            },
        )
        self.assertEqual(env_file.read_text(encoding="utf-8"), env_contents)
        self.assertEqual(receipt.read_bytes(), original_receipt)
        self.assertEqual(sorted(path.name for path in report_dir.iterdir()), [receipt.name])
        for forbidden in ("command", "checks", "error", "host", "path", "provider"):
            self.assertNotIn(forbidden, result.result)

    def test_disabled_policy_is_non_executing_and_does_not_use_prior_evidence(self) -> None:
        root = self._project(enabled=False)
        loaded = load_project(root)

        result = run_external_qa(
            loaded,
            "b" * 40,
            root / "missing-reports",
            prior={"status": "failed", "revision": "b" * 40, "error": "untrusted"},
        )

        self.assertEqual(result.status, "disabled")
        self.assertFalse(result.result["executionAttempted"])
        self.assertNotIn("error", result.result)
        self.assertFalse((root / "missing-reports").exists())

    def test_malformed_prior_and_same_prefix_revisions_remain_independent(self) -> None:
        root = self._project()
        loaded = load_project(root)
        first = "c" * 12 + "1" * 28
        second = "c" * 12 + "2" * 28
        malicious_prior = {
            "status": "passed",
            "revision": first,
            "checks": {"x": [{"y": [{"z": object()}]}]},
            "provider": {"value": "forged"},
        }

        first_result = run_external_qa(loaded, first, root / "reports", prior=malicious_prior)
        second_result = run_external_qa(loaded, second, root / "reports", prior=malicious_prior)

        self.assertEqual(first_result.result["revision"], first)
        self.assertEqual(second_result.result["revision"], second)
        self.assertEqual(first_result.result["status"], "review-required")
        self.assertEqual(second_result.result["status"], "review-required")
        self.assertFalse((root / "reports").exists())

    def test_parallel_calls_cannot_share_or_overwrite_a_receipt(self) -> None:
        root = self._project()
        loaded = load_project(root)
        revisions = ["d" * 39 + str(index) for index in range(8)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(
                executor.map(lambda revision: run_external_qa(loaded, revision, root / "reports"), revisions)
            )

        self.assertEqual([result.result["revision"] for result in results], revisions)
        self.assertTrue(all(result.status == "review-required" for result in results))
        self.assertTrue(all(result.receipt_path is None for result in results))
        self.assertFalse((root / "reports").exists())


if __name__ == "__main__":
    unittest.main()
