from __future__ import annotations

import unittest
from pathlib import Path

from ai_team.providers import ProviderErrorType, ProviderRequest, ProviderResult, RouterProvider
from ai_team.providers.base import BaseProvider


class RouterProviderTests(unittest.TestCase):
    def test_router_selects_first_ready_successful_provider(self) -> None:
        router = RouterProvider(
            [
                _StaticProvider("codex", ready=False, result=None),
                _StaticProvider("antigravity", ready=True, result=_result("antigravity", True)),
                _StaticProvider("handsfreecode", ready=True, result=_result("handsfreecode", True)),
            ]
        )

        result = router.run(_request())

        self.assertTrue(result.success)
        self.assertEqual(result.provider, "antigravity")
        self.assertEqual(result.data["selectedProvider"], "antigravity")
        self.assertEqual(result.data["routeAttempts"][0]["provider"], "codex")
        self.assertFalse(result.data["routeAttempts"][0]["ready"])

    def test_router_continues_after_rate_limit(self) -> None:
        router = RouterProvider(
            [
                _StaticProvider("codex", ready=True, result=_result("codex", False, ProviderErrorType.RATE_LIMIT)),
                _StaticProvider("handsfreecode", ready=True, result=_result("handsfreecode", True)),
            ]
        )

        result = router.run(_request())

        self.assertTrue(result.success)
        self.assertEqual(result.provider, "handsfreecode")
        self.assertEqual(result.data["routeAttempts"][0]["errorType"], ProviderErrorType.RATE_LIMIT)
        self.assertNotEqual(result.provider, "codex")

    def test_router_does_not_masquerade_local_fallback_as_cli_provider(self) -> None:
        router = RouterProvider(
            [
                _StaticProvider("codex", ready=True, result=_result("codex", False, ProviderErrorType.RATE_LIMIT)),
                _StaticProvider(
                    "handsfreecode",
                    ready=True,
                    result=ProviderResult(
                        provider="handsfreecode",
                        success=True,
                        content="ok",
                        data={"runtimeProvider": "ollama"},
                    ),
                ),
            ]
        )

        result = router.run(_request())

        self.assertEqual(result.provider, "handsfreecode")
        self.assertEqual(result.data["runtimeProvider"], "ollama")
        self.assertEqual(result.data["selectedProvider"], "handsfreecode")

    def test_router_falls_back_after_cli_quota_and_timeout(self) -> None:
        router = RouterProvider(
            [
                _StaticProvider("codex", ready=True, result=_result("codex", False, ProviderErrorType.RATE_LIMIT)),
                _StaticProvider(
                    "antigravity",
                    ready=True,
                    result=_result("antigravity", False, ProviderErrorType.TIMEOUT),
                ),
                _StaticProvider(
                    "handsfreecode",
                    ready=True,
                    result=ProviderResult(
                        provider="handsfreecode",
                        success=True,
                        content="ok",
                        data={"runtimeProvider": "ollama"},
                    ),
                ),
            ]
        )

        result = router.run(_request())

        self.assertTrue(result.success)
        self.assertEqual(result.provider, "handsfreecode")
        self.assertEqual(result.data["selectedProvider"], "handsfreecode")
        self.assertEqual(result.data["routeAttempts"][0]["errorType"], ProviderErrorType.RATE_LIMIT)
        self.assertEqual(result.data["routeAttempts"][1]["errorType"], ProviderErrorType.TIMEOUT)


class _StaticProvider(BaseProvider):
    def __init__(self, name: str, ready: bool, result: ProviderResult | None) -> None:
        self.name = name
        self._ready = ready
        self._result = result

    def ready(self) -> bool:
        return self._ready

    def run(self, request: ProviderRequest) -> ProviderResult:
        if self._result is None:
            return ProviderResult(provider=self.name, success=False, error_type=ProviderErrorType.EXTERNAL_REQUIRED)
        return self._result


def _request() -> ProviderRequest:
    return ProviderRequest(workflow="project-analysis", prompt="hello", project_root=Path.cwd())


def _result(provider: str, success: bool, error_type: ProviderErrorType | None = None) -> ProviderResult:
    return ProviderResult(provider=provider, success=success, error_type=error_type, content=provider)


if __name__ == "__main__":
    unittest.main()
