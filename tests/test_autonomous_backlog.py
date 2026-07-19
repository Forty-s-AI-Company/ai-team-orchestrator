from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_team.core.autonomous_backlog import discover_next_task
from ai_team.core.bounded_delivery import load_trusted_task_contract
from ai_team.providers.base import BaseProvider, ProviderRequest, ProviderResult


class AutonomousBacklogTests(unittest.TestCase):
    def test_creates_one_validated_task_and_deduplicates_same_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contracts = root / "contracts"
            state_path = root / "backlog.json"
            provider = _DiscoveryProvider(_task_payload())

            first = discover_next_task(
                project_path=root,
                contract_dir=contracts,
                state_path=state_path,
                provider=provider,
                timeout_seconds=30,
            )

            self.assertEqual(first["status"], "task-created", first)
            task_path = Path(first["contractPath"])
            contract, task_sha = load_trusted_task_contract(task_path)
            self.assertEqual(contract.id, "auto-accessibility-smoke")
            self.assertEqual(task_sha, first["taskSha"])
            self.assertEqual(provider.calls, 1)

            second = discover_next_task(
                project_path=root,
                contract_dir=contracts,
                state_path=state_path,
                provider=provider,
                timeout_seconds=30,
            )

            self.assertEqual(second["status"], "unchanged", second)
            self.assertEqual(provider.calls, 1)

    def test_ready_response_is_persisted_without_a_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = discover_next_task(
                project_path=root,
                contract_dir=root / "contracts",
                state_path=root / "backlog.json",
                provider=_DiscoveryProvider({
                    "schema": "ai-team-autonomous-backlog/v1",
                    "status": "ready",
                    "summary": "測試站已具備目前可驗證的安全開發項目。",
                }),
                timeout_seconds=30,
            )

            self.assertEqual(result["status"], "no-safe-task", result)
            self.assertFalse((root / "contracts").exists())

    def test_marks_the_request_as_an_autonomous_product_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider = _DiscoveryProvider(_task_payload())
            result = discover_next_task(
                project_path=root,
                contract_dir=root / "contracts",
                state_path=root / "backlog.json",
                provider=provider,
                timeout_seconds=30,
            )

            self.assertEqual(result["status"], "task-created", result)
            self.assertEqual(provider.last_request.workflow, "autonomous-product-discovery")
            self.assertEqual(provider.last_request.metadata["boundedStage"], "pm")
            self.assertIn("changePolicy as a JSON object", provider.last_request.prompt)
            self.assertIn("never use a string, array, or null", provider.last_request.prompt)

    def test_accepts_native_pm_envelope_with_top_level_backlog_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = {
                "schema": "ai-team-bounded-delivery/v1",
                "stage": "pm",
                "status": "passed",
                "challenge": "runtime-owned",
                "findings": [],
                "tests": [],
                "blockers": [],
                "backlogStatus": "task",
                "summary": "補上可驗證的無障礙 smoke 測試。",
                "contract": _task_payload()["contract"],
            }

            result = discover_next_task(
                project_path=root,
                contract_dir=root / "contracts",
                state_path=root / "backlog.json",
                provider=_DiscoveryProvider(payload),
                timeout_seconds=30,
            )

            self.assertEqual(result["status"], "task-created", result)

    def test_rejects_generated_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = _task_payload()
            payload["contract"]["dependsOn"] = ["not-generated"]

            result = discover_next_task(
                project_path=root,
                contract_dir=root / "contracts",
                state_path=root / "backlog.json",
                provider=_DiscoveryProvider(payload),
                timeout_seconds=30,
            )

            self.assertEqual(result["status"], "contract-rejected", result)
            self.assertFalse(list((root / "contracts").glob("*.json")))

    def test_normalizes_a_model_task_identifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = _task_payload()
            payload["contract"]["id"] = "Accessibility Audit!"

            result = discover_next_task(
                project_path=root,
                contract_dir=root / "contracts",
                state_path=root / "backlog.json",
                provider=_DiscoveryProvider(payload),
                timeout_seconds=30,
            )

            self.assertEqual(result["status"], "task-created", result)
            contract, _ = load_trusted_task_contract(Path(result["contractPath"]))
            self.assertEqual(contract.id, "auto-accessibility-audit")

    def test_project_commands_replace_incomplete_model_validation_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = _task_payload()
            payload["contract"]["validationCommands"] = [
                "npm run lint",
                "npm run typecheck",
                "npm run test -- tests/one.test.ts",
            ]
            project_commands = (
                "npm run lint",
                "npm run typecheck",
                "npm run test",
                "npm run build",
            )

            result = discover_next_task(
                project_path=root,
                contract_dir=root / "contracts",
                state_path=root / "backlog.json",
                provider=_DiscoveryProvider(payload),
                timeout_seconds=30,
                project_validation_commands=project_commands,
            )

            self.assertEqual(result["status"], "task-created", result)
            contract, _ = load_trusted_task_contract(Path(result["contractPath"]))
            self.assertEqual(contract.validation_commands, project_commands)


class _DiscoveryProvider(BaseProvider):
    name = "antigravity"

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls = 0

    def ready(self) -> bool:
        return True

    def run(self, request: ProviderRequest) -> ProviderResult:
        self.calls += 1
        self.last_request = request
        return ProviderResult(
            provider=self.name,
            success=True,
            content=json.dumps(self.payload),
        )


def _task_payload() -> dict[str, object]:
    return {
        "schema": "ai-team-autonomous-backlog/v1",
        "status": "task",
        "summary": "補上可驗證的無障礙 smoke 測試。",
        "contract": {
            "schemaVersion": 1,
            "id": "auto-accessibility-smoke",
            "title": "補上無障礙 smoke 驗證",
            "source": {"kind": "trusted-contract", "reference": "placeholder"},
            "instruction": "Add an accessibility smoke test without external operations.",
            "allowedWritePaths": ["tests"],
            "validationCommands": [
                "npm run lint",
                "npm run typecheck",
                "npm run test",
                "npm run build",
            ],
            "changePolicy": {
                "schemaChanges": False,
                "apiContractChanges": False,
                "migrationArtifacts": False,
                "fixtureData": False,
            },
        },
    }


if __name__ == "__main__":
    unittest.main()
