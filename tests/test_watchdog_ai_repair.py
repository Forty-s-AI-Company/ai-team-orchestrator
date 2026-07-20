from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_team.core.watchdog_ai_repair import (
    MAX_REPAIR_CYCLES,
    _compact_repair_cycles,
    _diagnosis_prompt,
    _last_json_object,
    _path_allowed,
    _recent_repair_history,
    _replan_prompt,
    _repair_prompt,
    _validation_feedback,
    _validate_write_paths,
)


class WatchdogAIRepairPolicyTests(unittest.TestCase):
    def test_diagnosis_prompt_routes_missing_evidence_to_bounded_observability_repair(self) -> None:
        prompt = _diagnosis_prompt(
            {},
            {},
            Path("/project"),
            Path("/orchestrator"),
        )

        self.assertIn("missing non-sensitive failure evidence", prompt)
        self.assertIn("status=repairable", prompt)
        self.assertIn("human/provider approval", prompt)

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

    def test_replan_prompt_carries_rejected_qa_into_a_new_sol_plan(self) -> None:
        prompt = _replan_prompt(
            {},
            {},
            Path("/project"),
            Path("/orchestrator"),
            {"summary": "old blacklist repair"},
            [{
                "cycle": 3,
                "changedFiles": ["scripts/check.mjs"],
                "validation": {"success": True},
                "qa": {
                    "status": "failed",
                    "summary": "unsafe blacklist",
                    "findings": ["Use a structural representation instead."],
                },
            }],
        )

        self.assertIn("materially revised plan", prompt)
        self.assertIn("Use a structural representation instead", prompt)
        self.assertIn("RejectedDiagnosis", prompt)

    def test_compact_repair_cycles_keeps_only_bounded_replanning_evidence(self) -> None:
        cycles = [
            {
                "cycle": index,
                "changedFiles": ["scripts/check.mjs"],
                "validation": {"success": True},
                "qa": {"status": "failed", "summary": f"failure {index}", "findings": []},
            }
            for index in range(1, MAX_REPAIR_CYCLES + 3)
        ]

        compact = _compact_repair_cycles(cycles)

        self.assertEqual(len(compact), MAX_REPAIR_CYCLES)
        self.assertEqual(compact[0]["cycle"], 3)
        self.assertEqual(compact[-1]["qa"]["summary"], "failure 5")

    def test_recent_repair_history_only_reuses_same_task_and_revision(self) -> None:
        supervisor = {
            "currentTask": {"taskSha": "task-sha"},
            "externalQa": {"revision": "revision-a"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matching = {
                "status": "failed",
                "completedAt": "2026-07-20T00:00:00+00:00",
                "error": "QA rejected",
                "supervisorEvidence": supervisor,
                "diagnosis": {"summary": "matching diagnosis"},
                "repairCycles": [{
                    "cycle": 3,
                    "validation": {"success": True},
                    "qa": {"status": "failed", "summary": "matching QA", "findings": []},
                }],
            }
            unrelated = {
                **matching,
                "supervisorEvidence": {
                    "currentTask": {"taskSha": "other-task"},
                    "externalQa": {"revision": "revision-a"},
                },
            }
            (root / "watchdog-ai-repair-1.json").write_text(
                json.dumps(matching),
                encoding="utf-8",
            )
            (root / "watchdog-ai-repair-2.json").write_text(
                json.dumps(unrelated),
                encoding="utf-8",
            )

            history = _recent_repair_history(supervisor, root)

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["error"], "QA rejected")
        self.assertEqual(
            history[0]["plans"][0]["repairCycles"][0]["qa"]["summary"],
            "matching QA",
        )

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
