from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ai_team.core.trusted_dev import (
    TestDatabaseSettings,
    load_trusted_dev_settings,
    validate_test_database_url,
    validate_trusted_dev_project,
)


class TrustedDevSettingsTests(unittest.TestCase):
    def test_requires_explicit_flag_and_exact_project_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            project.mkdir()
            settings = {
                "trusted_dev_autopilot": {
                    "enabled_projects": [str(project)],
                    "test_database": {
                        "enabled": True,
                        "bootstrap_commands": ["npm run db:migrate:deploy"],
                    },
                }
            }

            disabled = load_trusted_dev_settings(settings, project, requested=False)
            enabled = load_trusted_dev_settings(settings, project, requested=True)

            self.assertFalse(disabled.enabled)
            self.assertTrue(enabled.enabled)
            self.assertTrue(enabled.test_database.enabled)
            with self.assertRaisesRegex(ValueError, "not allowlisted"):
                load_trusted_dev_settings(
                    settings,
                    Path(tmp) / "another-project",
                    requested=True,
                )

    def test_database_guard_accepts_only_loopback_development_names(self) -> None:
        settings = TestDatabaseSettings()

        validate_test_database_url(
            "postgresql://user:password@127.0.0.1:5432/application_dev",
            settings,
        )
        for unsafe_url in (
            "postgresql://user:password@db.example.com/application_dev",
            "postgresql://user:password@localhost/application",
            "mysql://user:password@localhost/application_test",
        ):
            with self.subTest(url=unsafe_url), self.assertRaises(ValueError):
                validate_test_database_url(unsafe_url, settings)

    def test_project_guard_rejects_production_capabilities(self) -> None:
        safety = SimpleNamespace(
            allow_deploy=True,
            allow_database_migration=False,
            allow_database_seed=False,
            allow_destructive_commands=False,
            require_disposable_worktree_for_writes=True,
        )
        project = SimpleNamespace(
            profile=SimpleNamespace(
                project=SimpleNamespace(stage="development"),
                safety=safety,
            )
        )

        with self.assertRaisesRegex(ValueError, "remain disabled"):
            validate_trusted_dev_project(project)


if __name__ == "__main__":
    unittest.main()
