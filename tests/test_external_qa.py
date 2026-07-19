from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_team.core.external_qa import (
    MAX_CHECK_FIELDS,
    MAX_CHECK_ITEMS,
    MAX_CHECK_STRING_CHARS,
    run_external_qa,
)
from ai_team.core.project_loader import load_project


class ExternalQATests(unittest.TestCase):
    def _project(self, enabled: bool = True) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
        (root / ".ai-team").mkdir()
        (root / ".ai-team" / "project.yaml").write_text(
            f"""project:\n  name: sample\n  root: .\nexternal_qa:\n  enabled: {str(enabled).lower()}\n  environment: staging\n  command: npm run qa:payuni:sandbox\n""",
            encoding="utf-8",
        )
        (root / ".env.local").write_text(
            "PAYUNI_SANDBOX_QA_ENABLED=true\n",
            encoding="utf-8",
        )
        return root

    def test_disabled_policy_does_not_execute(self) -> None:
        root = self._project(enabled=False)
        loaded = load_project(root)
        with patch("ai_team.core.external_qa.subprocess.run") as run:
            result = run_external_qa(loaded, "a" * 40, root / "reports")
        self.assertEqual(result.status, "disabled")
        run.assert_not_called()

    def test_success_receipt_contains_only_bounded_summary(self) -> None:
        root = self._project()
        loaded = load_project(root)
        payload = (
            '{"schema":"celebratedeal-payuni-sandbox-qa/v1",'
            '"success":true,"environment":"sandbox",'
            '"checks":{"browserCheckout":"passed"},'
            '"productionValidation":{"automatedChargeAllowed":false}}\n'
        )
        completed = subprocess.CompletedProcess(
            ["npm", "run", "qa:payuni:sandbox"], 0, payload, "secret-looking-stderr"
        )
        with patch("ai_team.core.external_qa.subprocess.run", return_value=completed):
            result = run_external_qa(loaded, "b" * 40, root / "reports")
        self.assertEqual(result.status, "passed")
        self.assertIsNotNone(result.receipt_path)
        receipt = result.receipt_path.read_text(encoding="utf-8")
        self.assertIn('"browserCheckout": "passed"', receipt)
        self.assertNotIn("secret-looking-stderr", receipt)

    def test_run_once_preserves_original_failure_evidence(self) -> None:
        root = self._project()
        loaded = load_project(root)
        receipt = root / "reports" / "external-qa-deadbeef.json"
        prior = {
            "schema": "ai-team-external-qa-receipt/v1",
            "revision": "c" * 40,
            "status": "failed",
            "exitCode": 1,
            "error": "page.waitForURL: Timeout 45000ms exceeded",
            "providerChecks": None,
            "receiptPath": str(receipt),
        }
        receipt.parent.mkdir()
        receipt.write_text(json.dumps(prior), encoding="utf-8")

        with patch("ai_team.core.external_qa.subprocess.run") as run:
            result = run_external_qa(
                loaded,
                "c" * 40,
                root / "reports",
                prior=prior,
            )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.result["reason"], "already-run-for-revision")
        self.assertEqual(result.result["error"], prior["error"])
        self.assertEqual(result.result["exitCode"], 1)
        run.assert_not_called()

    def test_timeout_provider_checks_keep_structured_payuni_diagnostics(self) -> None:
        root = self._project()
        loaded = load_project(root)
        payload = {
            "schema": "celebratedeal-payuni-sandbox-qa/v1",
            "success": False,
            "environment": "sandbox",
            "error": "page.waitForURL: Timeout 45000ms exceeded",
            "checks": {
                "providerChecks": {
                    "stage": "awaiting-callback",
                    "currentHttpsHostPath": "https://sandbox-api.payuni.com.tw/api/checkout/callback",
                    "confirmationDialog": {"visible": True, "confirmed": False},
                    "checkoutHttpStatus": 302,
                    "visiblePayUniStatus": "等待 PayUni 回傳付款結果",
                }
            },
            "productionValidation": {"automatedChargeAllowed": False},
        }
        completed = subprocess.CompletedProcess(
            ["npm", "run", "qa:payuni:sandbox"], 1, json.dumps(payload) + "\n", ""
        )

        with patch("ai_team.core.external_qa.subprocess.run", return_value=completed):
            result = run_external_qa(loaded, "d" * 40, root / "reports")

        self.assertEqual(result.status, "failed")
        provider_checks = result.result["providerChecks"]["providerChecks"]
        self.assertEqual(provider_checks["stage"], "awaiting-callback")
        self.assertEqual(
            provider_checks["currentHttpsHostPath"],
            "https://sandbox-api.payuni.com.tw/api/checkout/callback",
        )
        self.assertEqual(provider_checks["confirmationDialog"], {"visible": True, "confirmed": False})
        self.assertEqual(provider_checks["checkoutHttpStatus"], 302)
        self.assertEqual(provider_checks["visiblePayUniStatus"], "等待 PayUni 回傳付款結果")

    def test_provider_check_summary_is_bounded_and_redacted_in_receipt(self) -> None:
        root = self._project()
        loaded = load_project(root)
        long_sensitive_key = "ordinary-" * 40 + "apiKey"
        payload = {
            "schema": "celebratedeal-payuni-sandbox-qa/v1",
            "success": False,
            "environment": "sandbox",
            "checks": {
                "providerChecks": {
                    "apiKey": "sensitive-payuni-credential",
                    long_sensitive_key: "hidden-after-key-truncation",
                    "longValue": "x" * (MAX_CHECK_STRING_CHARS + 1),
                    "manyFields": {str(index): index for index in range(MAX_CHECK_FIELDS + 1)},
                    "items": list(range(MAX_CHECK_ITEMS + 1)),
                }
            },
            "productionValidation": {"automatedChargeAllowed": False},
        }
        completed = subprocess.CompletedProcess(
            ["npm", "run", "qa:payuni:sandbox"], 1, json.dumps(payload) + "\n", ""
        )

        with patch("ai_team.core.external_qa.subprocess.run", return_value=completed):
            result = run_external_qa(loaded, "e" * 40, root / "reports")

        provider_checks = result.result["providerChecks"]["providerChecks"]
        self.assertEqual(len(provider_checks["longValue"]), MAX_CHECK_STRING_CHARS)
        self.assertEqual(len(provider_checks["manyFields"]), MAX_CHECK_FIELDS)
        self.assertEqual(len(provider_checks["items"]), MAX_CHECK_ITEMS)
        receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["providerChecks"]["providerChecks"]["apiKey"], "<redacted>")
        self.assertEqual(
            receipt["providerChecks"]["providerChecks"][long_sensitive_key[:MAX_CHECK_STRING_CHARS]],
            "<redacted>",
        )
        self.assertNotIn("sensitive-payuni-credential", result.receipt_path.read_text(encoding="utf-8"))
        self.assertNotIn("hidden-after-key-truncation", result.receipt_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
