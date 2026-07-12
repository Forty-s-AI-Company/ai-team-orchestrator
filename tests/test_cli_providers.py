from __future__ import annotations

import sys
import unittest
from pathlib import Path

from ai_team.providers import (
    AntigravityProvider,
    AntigravitySettings,
    CodexProvider,
    CodexSettings,
    ProviderErrorType,
    ProviderRequest,
)


class CliProviderTests(unittest.TestCase):
    def test_codex_quota_exhaustion_parses_reset_time(self) -> None:
        provider = CodexProvider(
            CodexSettings(
                executable=sys.executable,
                status_args=["--version"],
                quota_args=[
                    "-c",
                    "print(\"You've hit your usage limit. try again at Feb 23rd, 2026 9:01 PM.\")",
                ],
                run_args=["-c", "print('should-not-run')"],
            )
        )

        result = provider.run(_request())

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, ProviderErrorType.RATE_LIMIT)
        self.assertTrue(result.data["quotaExhausted"])
        self.assertIn("Feb 23rd, 2026 9:01 PM", str(result.data["resetTime"]))

    def test_antigravity_quota_exhaustion_parses_reset_time(self) -> None:
        provider = AntigravityProvider(
            AntigravitySettings(
                executable=sys.executable,
                status_args=["--version"],
                quota_args=[
                    "-c",
                    "print('Error: HTTP 429 Too Many Requests RESOURCE_EXHAUSTED Reset Time: 2026-07-12 08:00:00 (Local Time)')",
                ],
            )
        )

        result = provider.run(_request())

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, ProviderErrorType.RATE_LIMIT)
        self.assertTrue(result.data["quotaExhausted"])
        self.assertEqual(result.data["resetTime"], "2026-07-12 08:00:00")

    def test_antigravity_execution_disabled_is_external_required(self) -> None:
        provider = AntigravityProvider(
            AntigravitySettings(
                executable=sys.executable,
                status_args=["--version"],
                quota_args=[],
                run_args=["-c", "print('disabled')"],
                execution_enabled=False,
            )
        )

        result = provider.run(_request())

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, ProviderErrorType.EXTERNAL_REQUIRED)
        self.assertEqual(result.provider, "antigravity")

    def test_cli_provider_does_not_masquerade_as_other_provider(self) -> None:
        provider = CodexProvider(
            CodexSettings(
                executable=sys.executable,
                status_args=["--version"],
                quota_args=[],
                run_args=["-c", "import sys; print(sys.argv[-1])"],
            )
        )

        result = provider.run(_request(prompt="hello cli"))

        self.assertTrue(result.success)
        self.assertEqual(result.provider, "codex")
        self.assertFalse(result.data["masqueradeAsProvider"])
        self.assertIn("hello cli", result.content)

    def test_cli_status_failure_is_not_ready(self) -> None:
        provider = CodexProvider(
            CodexSettings(
                executable=sys.executable,
                status_args=["-c", "import sys; sys.exit(7)"],
                quota_args=[],
            )
        )

        diagnostics = provider.diagnostics()

        self.assertFalse(diagnostics["ready"])
        self.assertEqual(diagnostics["errorType"], ProviderErrorType.EXTERNAL_REQUIRED)


def _request(prompt: str = "hello") -> ProviderRequest:
    return ProviderRequest(workflow="project-analysis", prompt=prompt, project_root=Path.cwd())


if __name__ == "__main__":
    unittest.main()
