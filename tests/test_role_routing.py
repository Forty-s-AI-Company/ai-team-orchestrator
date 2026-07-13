from __future__ import annotations

import unittest

from ai_team.cli import build_parser, build_provider
from ai_team.core.orchestrator import WorkflowError
from ai_team.core.routing_config import load_role_profile
from ai_team.providers import RoleRouterProvider, RouterProvider


SETTINGS = {
    "routing": {
        "roles": {
            "engineer": {
                "primary": {
                    "provider": "codex",
                    "model": "gpt-5.6-terra",
                    "reasoning_effort": "high",
                },
                "fallbacks": [],
                "allow_write": True,
            }
        }
    },
    "codex": {
        "execution_enabled": False,
        "allowed_models": ["gpt-5.6-terra"],
        "allowed_reasoning_efforts": ["high"],
    },
    "antigravity": {"execution_enabled": False, "allowed_models": []},
    "handsfreecode": {"allowed_models": ["qwen2.5-coder:7b"]},
}


class RoleRoutingConfigurationTests(unittest.TestCase):
    def test_auto_provider_excludes_mock_and_openhands(self) -> None:
        provider = build_provider("auto", SETTINGS)

        self.assertIsInstance(provider, RouterProvider)
        self.assertEqual(
            [item.name for item in provider.providers],
            ["codex", "antigravity", "handsfreecode"],
        )

    def test_role_builds_auditable_role_router(self) -> None:
        provider = build_provider("auto", SETTINGS, role="engineer")

        self.assertIsInstance(provider, RoleRouterProvider)
        self.assertEqual(provider.profile.primary.model, "gpt-5.6-terra")
        self.assertEqual(provider.profile.primary.reasoning_effort, "high")
        self.assertTrue(provider.profile.allow_write)

    def test_role_with_explicit_provider_is_rejected(self) -> None:
        with self.assertRaises(WorkflowError):
            build_provider("codex", SETTINGS, role="engineer")

    def test_unknown_provider_in_profile_fails_closed(self) -> None:
        unsafe = {
            "routing": {
                "roles": {
                    "engineer": {
                        "primary": {
                            "provider": "mock",
                            "model": "pretend",
                            "reasoning_effort": "high",
                        }
                    }
                }
            }
        }

        with self.assertRaises(WorkflowError):
            load_role_profile(unsafe, "engineer")

    def test_run_cli_accepts_declared_role(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "/tmp/project",
                "--workflow",
                "project-analysis",
                "--provider",
                "auto",
                "--role",
                "architect",
            ]
        )

        self.assertEqual(args.role, "architect")


if __name__ == "__main__":
    unittest.main()
