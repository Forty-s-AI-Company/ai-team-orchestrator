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
from ai_team.providers.cli_common import CliProviderSettings, run_cli_command


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

    def test_antigravity_individual_quota_resets_in_is_rate_limit(self) -> None:
        provider = AntigravityProvider(
            AntigravitySettings(
                executable=sys.executable,
                status_args=["--version"],
                quota_args=[],
                run_args=[
                    "-c",
                    "import sys; sys.stderr.write('Error: Individual quota reached. Resets in 1h51m33s.'); sys.exit(1)",
                ],
                execution_enabled=True,
            )
        )

        result = provider.run(_request())

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, ProviderErrorType.RATE_LIMIT)
        self.assertEqual(result.data["resetTime"], "1h51m33s")

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

    def test_cli_command_decodes_utf8_output_on_windows(self) -> None:
        result = run_cli_command(
            CliProviderSettings(executable=sys.executable),
            ["-c", "print('測試輸出')"],
        )

        self.assertEqual(result.return_code, 0)
        self.assertIn("測試輸出", result.stdout)

    def test_cli_run_uses_provider_run_timeout_when_request_timeout_is_unspecified(self) -> None:
        provider = CodexProvider(
            CodexSettings(
                executable=sys.executable,
                status_args=["--version"],
                quota_args=[],
                run_args=["-c", "import time; time.sleep(0.2); print('late')"],
                timeout_seconds=5,
                run_timeout_seconds=0.01,
            )
        )

        result = provider.run(_request())

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, ProviderErrorType.TIMEOUT)
        self.assertIn("timeout", str(result.data["command"]["error"]))

    def test_cli_timeout_stderr_is_classified_as_timeout(self) -> None:
        provider = AntigravityProvider(
            AntigravitySettings(
                executable=sys.executable,
                status_args=["--version"],
                quota_args=[],
                run_args=["-c", "import sys; sys.stderr.write('Error: timeout waiting for response'); sys.exit(1)"],
                execution_enabled=True,
            )
        )

        result = provider.run(_request())

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, ProviderErrorType.TIMEOUT)


def _request(prompt: str = "hello") -> ProviderRequest:
    return ProviderRequest(workflow="project-analysis", prompt=prompt, project_root=Path.cwd())


if __name__ == "__main__":
    unittest.main()
