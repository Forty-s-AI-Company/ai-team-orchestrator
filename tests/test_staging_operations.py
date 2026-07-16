from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_team.core.bounded_delivery import BoundedDeliveryError, load_trusted_task_contract
from ai_team.core.staging_operations import SCHEMA, load_staging_operations_contract, run_staging_operations


class StagingOperationsTests(unittest.TestCase):
    def test_contract_rejects_non_staging_or_non_preview_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_contract(Path(tmp), environment="production")
            with self.assertRaisesRegex(Exception, "environment=staging and deployment=preview"):
                load_staging_operations_contract(path)

    def test_contract_rejects_unknown_or_duplicate_operations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for operations in (["database-migration", "database-migration"], ["shell"]):
                with self.subTest(operations=operations):
                    path = _write_contract(Path(tmp), operations=operations)
                    with self.assertRaises(Exception):
                        load_staging_operations_contract(path)

    def test_legacy_bounded_delivery_still_rejects_migration_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bounded.json"
            path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "id": "unsafe",
                        "title": "Unsafe",
                        "source": {"kind": "trusted-contract", "reference": "test"},
                        "instruction": "Run a database migration",
                        "allowedWritePaths": ["docs/safe.md"],
                        "validationCommands": ["npm run lint"],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(BoundedDeliveryError):
                load_trusted_task_contract(path)

    def test_disabled_policy_fails_closed_before_running_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _init_project(Path(tmp) / "project")
            contract = _write_contract(Path(tmp), operations=["database-migration"])
            called = False

            def runner(*_args):
                nonlocal called
                called = True
                raise AssertionError("must not run")

            result = run_staging_operations(root, contract, Path(tmp) / "reports", execute=True, runner=runner)

            self.assertFalse(result.success)
            self.assertEqual(result.stop_reason, "staging-operations-disabled")
            self.assertFalse(called)
            receipt = _receipt(result.receipt_path)
            self.assertFalse(receipt["validationResult"]["success"])
            self.assertEqual(receipt["validationResult"]["stopReason"], "staging-operations-disabled")

    def test_database_fingerprint_mismatch_fails_closed_without_exposing_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"DATABASE_URL": "test-secret-database-url"}, clear=False):
            root = _init_project(Path(tmp) / "project", _staging_policy("different-value", migration=True))
            contract = _write_contract(Path(tmp), operations=["database-migration"])
            result = run_staging_operations(root, contract, Path(tmp) / "reports", execute=True)

            self.assertFalse(result.success)
            self.assertEqual(result.stop_reason, "staging-database-fingerprint-mismatch")
            self.assertNotIn("test-secret-database-url", result.receipt_path.read_text(encoding="utf-8"))

    def test_preview_deploy_requires_preview_environment_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"VERCEL_ENV": "production"}, clear=False):
            root = _init_project(Path(tmp) / "project", _staging_policy("unused", preview=True))
            contract = _write_contract(Path(tmp), operations=["preview-deploy"])
            result = run_staging_operations(root, contract, Path(tmp) / "reports", execute=True)

            self.assertFalse(result.success)
            self.assertEqual(result.stop_reason, "preview-environment-attestation-required")

    def test_dirty_product_worktree_is_rejected_before_external_command(self) -> None:
        database_url = "staging-only-database"
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"DATABASE_URL": database_url}, clear=False):
            root = _init_project(Path(tmp) / "project", _staging_policy(database_url, migration=True))
            (root / "unrelated.txt").write_text("user change", encoding="utf-8")
            contract = _write_contract(Path(tmp), operations=["database-migration"])
            result = run_staging_operations(root, contract, Path(tmp) / "reports", execute=True)

            self.assertFalse(result.success)
            self.assertEqual(result.stop_reason, "project-worktree-dirty")

    def test_seed_forces_minimal_demo_mode(self) -> None:
        database_url = "staging-only-database"
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"DATABASE_URL": database_url, "SEED_MODE": "production-bootstrap"}, clear=False
        ):
            root = _init_project(Path(tmp) / "project", _staging_policy(database_url, seed=True))
            contract = _write_contract(Path(tmp), operations=["database-seed"])
            captured_environment: dict[str, str] = {}

            def runner(command: tuple[str, ...], _cwd: Path, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
                captured_environment.update(environment)
                return subprocess.CompletedProcess(command, 0, "seeded", "")

            result = run_staging_operations(root, contract, Path(tmp) / "reports", execute=True, runner=runner)

            self.assertTrue(result.success)
            self.assertEqual(captured_environment["SEED_MODE"], "demo")

    def test_allowlisted_migration_seed_and_preview_are_executed_and_receipted(self) -> None:
        database_url = "staging-only-database"
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"DATABASE_URL": database_url, "VERCEL_ENV": "preview"}, clear=False
        ):
            root = _init_project(
                Path(tmp) / "project",
                _staging_policy(database_url, migration=True, seed=True, preview=True),
            )
            contract = _write_contract(
                Path(tmp), operations=["database-migration", "database-seed", "preview-deploy"]
            )
            commands: list[tuple[str, ...]] = []

            def runner(command: tuple[str, ...], _cwd: Path, _env: dict[str, str]) -> subprocess.CompletedProcess[str]:
                commands.append(command)
                return subprocess.CompletedProcess(command, 0, "done token=test-secret", "")

            result = run_staging_operations(root, contract, Path(tmp) / "reports", execute=True, runner=runner)

            self.assertTrue(result.success)
            self.assertEqual(
                commands,
                [
                    ("npm", "run", "db:migrate:deploy"),
                    ("npm", "run", "db:seed"),
                    ("vercel", "deploy", "--yes"),
                ],
            )
            receipt = _receipt(result.receipt_path)
            self.assertEqual(receipt["schema"], "ai-team-staging-operation-receipt/v1")
            self.assertEqual(receipt["target"], {"environment": "staging", "deployment": "preview"})
            self.assertTrue(receipt["validationResult"]["success"])
            self.assertTrue(receipt["validationResult"]["databaseFingerprintValidated"])
            self.assertNotIn("test-secret", result.receipt_path.read_text(encoding="utf-8"))

    def test_command_failure_is_receipted_as_failure(self) -> None:
        database_url = "staging-only-database"
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"DATABASE_URL": database_url}, clear=False):
            root = _init_project(Path(tmp) / "project", _staging_policy(database_url, migration=True))
            contract = _write_contract(Path(tmp), operations=["database-migration"])

            def runner(command: tuple[str, ...], _cwd: Path, _env: dict[str, str]) -> subprocess.CompletedProcess[str]:
                return subprocess.CompletedProcess(command, 19, "", "credential=must-not-appear")

            result = run_staging_operations(root, contract, Path(tmp) / "reports", execute=True, runner=runner)

            self.assertFalse(result.success)
            self.assertEqual(result.stop_reason, "database-migration-command-failed")
            receipt = _receipt(result.receipt_path)
            self.assertFalse(receipt["validationResult"]["success"])
            self.assertEqual(receipt["validationResult"]["kind"], "command-execution")
            self.assertNotIn("must-not-appear", result.receipt_path.read_text(encoding="utf-8"))


def _init_project(root: Path, staging_policy: str | None = None) -> Path:
    root.mkdir()
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    (root / ".ai-team").mkdir()
    policy = staging_policy or ""
    (root / ".ai-team" / "project.yaml").write_text(
        f"""project:
  name: sample
  root: \".\"
  stage: development
repository:
  protected_branches: [main, master]
commands:
  lint: npm run lint
safety:
  allow_git_push: false
  allow_deploy: false
  allow_database_migration: false
  allow_database_seed: false
  allow_destructive_commands: false
{policy}""",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", ".ai-team/project.yaml"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.name=AI Team Test", "-c", "user.email=test@example.invalid", "commit", "-m", "initial"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return root


def _staging_policy(database_url: str, *, migration: bool = False, seed: bool = False, preview: bool = False) -> str:
    digest = hashlib.sha256(database_url.encode("utf-8")).hexdigest()
    return f"""staging_operations:
  enabled: true
  environment: staging
  database_url_env: DATABASE_URL
  database_url_sha256: {digest}
  allow_migration: {str(migration).lower()}
  allow_seed: {str(seed).lower()}
  allow_preview_deploy: {str(preview).lower()}
  preview_environment_variable: VERCEL_ENV
"""


def _write_contract(
    root: Path,
    *,
    operations: list[str] | None = None,
    environment: str = "staging",
    deployment: str = "preview",
) -> Path:
    path = root / "staging-contract.json"
    path.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "id": "staging-smoke",
                "title": "Staging external smoke",
                "source": {"kind": "trusted-contract", "reference": "manual-approval"},
                "target": {"environment": environment, "deployment": deployment},
                "operations": operations or ["database-migration"],
            }
        ),
        encoding="utf-8",
    )
    return path


def _receipt(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
