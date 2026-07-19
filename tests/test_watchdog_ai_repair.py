from __future__ import annotations

import unittest

from ai_team.core.watchdog_ai_repair import (
    _last_json_object,
    _path_allowed,
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


if __name__ == "__main__":
    unittest.main()
