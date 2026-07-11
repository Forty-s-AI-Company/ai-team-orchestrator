from __future__ import annotations

import os
import unittest
from pathlib import Path

from ai_team.providers import MockProvider, OpenHandsProvider, OpenHandsSettings, ProviderErrorType
from ai_team.providers.base import ProviderRequest, RetryingProvider, redact_secrets


class ProviderTests(unittest.TestCase):
    def test_openhands_fails_closed_without_session_key(self) -> None:
        old_value = os.environ.pop("SESSION_API_KEY", None)
        try:
            provider = OpenHandsProvider(OpenHandsSettings(base_url="http://127.0.0.1:31024"))
            result = provider.run(
                ProviderRequest(
                    workflow="project-analysis",
                    prompt="hello",
                    project_root=Path.cwd(),
                )
            )
            self.assertFalse(result.success)
            self.assertEqual(result.error_type, ProviderErrorType.AUTH)
        finally:
            if old_value is not None:
                os.environ["SESSION_API_KEY"] = old_value

    def test_provider_timeout_is_classified(self) -> None:
        provider = OpenHandsProvider(
            OpenHandsSettings(
                base_url="http://10.255.255.1:31024",
                timeout_seconds=0.001,
            ),
            session_key="test-session",
        )
        result = provider.run(
            ProviderRequest(
                workflow="project-analysis",
                prompt="hello",
                project_root=Path.cwd(),
                timeout_seconds=0.001,
            )
        )
        self.assertFalse(result.success)
        self.assertIn(result.error_type, {ProviderErrorType.TIMEOUT, ProviderErrorType.NETWORK})

    def test_retry_exhaustion(self) -> None:
        provider = RetryingProvider(MockProvider(fail_times=5), max_retries=2, backoff_seconds=0)
        result = provider.run(
            ProviderRequest(
                workflow="project-analysis",
                prompt="hello",
                project_root=Path.cwd(),
            )
        )
        self.assertFalse(result.success)
        self.assertEqual(result.attempts, 3)

    def test_secret_redaction(self) -> None:
        sample = "SESSION_API_KEY" + "=supersecret " + "Bearer " + "abc.def.ghi " + "sk-" + "test123456789"
        redacted = redact_secrets(sample)
        self.assertNotIn("supersecret", redacted)
        self.assertNotIn("abc.def.ghi", redacted)
        self.assertNotIn("sk-" + "test123456789", redacted)


if __name__ == "__main__":
    unittest.main()
