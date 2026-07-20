from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_team.core.watchdog_ai_repair import (
    MAX_REPAIR_CYCLES,
    _agy_acceptance_criteria,
    _agy_blocking_findings,
    _blocking_review_findings,
    _cycle_reasoning_effort,
    _compact_repair_cycles,
    _constrain_diagnosis_to_contract,
    _diagnosis_prompt,
    _freeze_acceptance_contract,
    _last_json_object,
    _path_allowed,
    _qa_prompt,
    _recent_repair_history,
    _replan_prompt,
    _repair_prompt,
    _validation_feedback,
    _validate_diagnosis,
    _validate_patch_budget,
    _validate_qa,
    _validate_write_paths,
)


class WatchdogAIRepairPolicyTests(unittest.TestCase):
    def test_first_cycle_is_high_and_later_cycles_escalate_to_xhigh(self) -> None:
        self.assertEqual(MAX_REPAIR_CYCLES, 5)
        self.assertEqual(_cycle_reasoning_effort(1, "high"), "high")
        self.assertEqual(_cycle_reasoning_effort(2, "high"), "xhigh")
        self.assertEqual(_cycle_reasoning_effort(5, "high"), "xhigh")

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

        self.assertIn("frozen contract", prompt)
        self.assertIn("acceptance criteria, and exclusions are immutable", prompt)
        self.assertIn("Use a structural representation instead", prompt)
        self.assertIn("RejectedDiagnosis", prompt)

    def test_first_diagnosis_freezes_testable_acceptance_contract(self) -> None:
        diagnosis = _validate_diagnosis({
            "schema": "ai-team-watchdog-diagnosis/v1",
            "status": "repairable",
            "repository": "orchestrator",
            "summary": "避免修復紀錄重複",
            "rootCause": "current alias 被 glob 重複讀取",
            "repairInstruction": "排除 current alias 並補測試",
            "allowedWritePaths": ["src/ai_team/core/watchdog_ai_repair.py"],
            "acceptanceCriteria": [{
                "id": "AC-1",
                "requirement": "同一份修復結果只出現一次",
                "verification": "重複 current alias 的單元測試通過",
            }],
            "outOfScope": ["重寫 Git 發布協議"],
        })

        contract = _freeze_acceptance_contract(diagnosis)

        self.assertEqual(contract["repository"], "orchestrator")
        self.assertEqual(contract["acceptanceCriteria"][0]["id"], "AC-1")
        self.assertIn("重寫 Git 發布協議", contract["outOfScope"])
        self.assertEqual(contract["changeBudget"]["maxChangedFiles"], 5)
        self.assertEqual(len(contract["sha256"]), 64)

    def test_replan_cannot_expand_frozen_repository_paths_or_acceptance_rules(self) -> None:
        original = _validate_diagnosis({
            "schema": "ai-team-watchdog-diagnosis/v1",
            "status": "repairable",
            "repository": "orchestrator",
            "summary": "bounded repair",
            "rootCause": "root cause",
            "repairInstruction": "fix parser",
            "allowedWritePaths": ["src/ai_team/core/watchdog_ai_repair.py"],
            "acceptanceCriteria": [{
                "id": "AC-1",
                "requirement": "parser is total",
                "verification": "malformed input test passes",
            }],
        })
        contract = _freeze_acceptance_contract(original)
        expanded = _validate_diagnosis({
            **original,
            "allowedWritePaths": [
                "src/ai_team/core/watchdog_ai_repair.py",
                "src/ai_team/core/new_publication_protocol.py",
            ],
            "acceptanceCriteria": [
                *original["acceptanceCriteria"],
                {
                    "id": "AC-2",
                    "requirement": "replace the publication architecture",
                    "verification": "fault injection suite passes",
                },
            ],
        })

        constrained, follow_ups = _constrain_diagnosis_to_contract(expanded, contract)

        self.assertEqual(constrained["allowedWritePaths"], contract["allowedWritePaths"])
        self.assertEqual(constrained["acceptanceCriteria"], contract["acceptanceCriteria"])
        self.assertEqual(len(follow_ups), 2)
        self.assertTrue(all(item["allowedToBlock"] is False for item in follow_ups))

    def test_only_frozen_contract_failures_and_patch_regressions_can_block(self) -> None:
        contract = {
            "acceptanceCriteria": [{"id": "AC-1"}],
        }
        qa = _validate_qa({
            "schema": "ai-team-watchdog-qa/v1",
            "status": "failed",
            "summary": "mixed review",
            "findings": [
                {
                    "id": "contract",
                    "category": "acceptance-failure",
                    "acceptanceRuleId": "AC-1",
                    "introducedByCurrentPatch": False,
                    "severity": "medium",
                    "evidence": "AC-1 test still fails",
                    "action": "fix AC-1",
                },
                {
                    "id": "architecture",
                    "category": "architecture-improvement",
                    "acceptanceRuleId": None,
                    "introducedByCurrentPatch": False,
                    "severity": "high",
                    "evidence": "a transactional protocol would be stronger",
                    "action": "create a separate task",
                },
                {
                    "id": "regression",
                    "category": "patch-regression",
                    "acceptanceRuleId": None,
                    "introducedByCurrentPatch": True,
                    "severity": "high",
                    "evidence": "new malformed input crash",
                    "action": "handle malformed input",
                },
            ],
        })

        blocking = _blocking_review_findings(qa, contract)

        self.assertEqual([item["id"] for item in blocking], ["contract", "regression"])

    def test_unstructured_review_opinion_is_follow_up_not_a_moving_goalpost(self) -> None:
        qa = _validate_qa({
            "schema": "ai-team-watchdog-qa/v1",
            "status": "failed",
            "summary": "consider a rewrite",
            "findings": ["Replace the entire publication architecture."],
        })

        self.assertEqual(_blocking_review_findings(qa, {"acceptanceCriteria": []}), [])
        self.assertEqual(qa["findings"][0]["category"], "unverified")

    def test_review_prompt_explicitly_rejects_scope_creep(self) -> None:
        prompt = _qa_prompt(
            {"summary": "repair"},
            {"success": True},
            {"status": "passed"},
            "diff --git a/a b/a",
            acceptance_contract={"acceptanceCriteria": [{"id": "AC-1"}]},
        )

        self.assertIn("acceptance contract is frozen", prompt)
        self.assertIn("must not fail this repair", prompt)
        self.assertIn("introducedByCurrentPatch", prompt)

    def test_agy_receives_string_criteria_and_only_matching_rules_block(self) -> None:
        contract = {
            "acceptanceCriteria": [{
                "id": "AC-1",
                "requirement": "history is deduplicated",
                "verification": "focused test passes",
            }],
        }
        criteria = _agy_acceptance_criteria(contract)
        qa = {
            "providerExecutionSucceeded": True,
            "status": "failed",
            "findings": [
                "AC-1: focused test still fails",
                "Consider replacing the publication architecture",
            ],
            "blockers": [],
        }

        blocking = _agy_blocking_findings(qa, contract)

        self.assertEqual(criteria, ["AC-1: history is deduplicated; Verification: focused test passes"])
        self.assertEqual(blocking, ["AC-1: focused test still fails"])

    def test_patch_budget_stops_architecture_rewrites(self) -> None:
        with self.assertRaisesRegex(ValueError, "bounded limit is 5"):
            _validate_patch_budget([f"src/file-{index}.py" for index in range(6)], "+ok")
        oversized_patch = "\n".join(f"+line {index}" for index in range(501))
        with self.assertRaisesRegex(ValueError, "bounded limit is 500"):
            _validate_patch_budget(["src/file.py"], oversized_patch)

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
        self.assertEqual(compact[-1]["qa"]["summary"], "failure 7")

    def test_compact_replanning_evidence_excludes_non_blocking_findings(self) -> None:
        compact = _compact_repair_cycles([{
            "cycle": 1,
            "validation": {"success": True},
            "solReview": {
                "status": "failed",
                "summary": "mixed",
                "findings": ["blocking", "architecture"],
            },
            "blockingFindings": ["blocking"],
            "agyQa": {
                "status": "failed",
                "findings": ["AC-1: failed", "replace architecture"],
                "blockers": [],
            },
            "agyBlockingFindings": ["AC-1: failed"],
        }])

        self.assertEqual(compact[0]["qa"]["findings"], ["blocking"])
        self.assertEqual(compact[0]["agyQa"]["findings"], ["AC-1: failed"])

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
