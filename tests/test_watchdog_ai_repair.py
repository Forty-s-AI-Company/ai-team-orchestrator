from __future__ import annotations

import unittest

from ai_team.core.watchdog_ai_repair import (
    _last_json_object,
    _path_allowed,
    _repair_prompt,
    _validation_feedback,
    _validate_write_paths,
)


class WatchdogAIRepairPolicyTests(unittest.TestCase):
    def test_accepts_bounded_project_paths(self) -> None:
        paths = _validate_write_paths(
            "project",
            ["scripts/payuni-sandbox-external-qa.mjs", "tests/payuni.test.ts"],
        )

        self.assertEqual(
            paths,
            ["scripts/payuni-sandbox-external-qa.mjs", "tests/payuni.test.ts"],
        )
        self.assertTrue(_path_allowed("tests/payuni.test.ts", paths))

    def test_rejects_secret_or_parent_escape_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside repair policy"):
            _validate_write_paths("project", [".env.local"])
        with self.assertRaisesRegex(ValueError, "unsafe diagnosed write path"):
            _validate_write_paths("orchestrator", ["../service.conf"])

    def test_extracts_strict_json_after_cli_progress(self) -> None:
        payload = _last_json_object(
            'progress line\n{"schema":"ai-team-watchdog-qa/v1","status":"passed"}'
        )

        self.assertEqual(payload["status"], "passed")

    def test_repair_prompt_carries_qa_findings_back_to_terra(self) -> None:
        prompt = _repair_prompt(
            {"summary": "repair"},
            ["scripts/check.mjs"],
            feedback=["Only catch the wait timeout; do not swallow click failures."],
        )

        self.assertIn("PreviousQAFindings", prompt)
        self.assertIn("do not swallow click failures", prompt)

    def test_validation_feedback_uses_the_failing_command(self) -> None:
        feedback = _validation_feedback({
            "commands": [
                {"command": "npm run lint", "returnCode": 0},
                {"command": "npm run test", "returnCode": 1, "stderr": "one test failed"},
            ]
        })

        self.assertIn("npm run test", feedback[0])
        self.assertIn("one test failed", feedback[0])


if __name__ == "__main__":
    unittest.main()
