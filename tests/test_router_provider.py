from __future__ import annotations

import unittest
from pathlib import Path

from ai_team.providers import (
    ProviderErrorType,
    ProviderRequest,
    ProviderResult,
    RoleRouterProvider,
    RoleRoutingProfile,
    RouteTarget,
    RouterProvider,
)
from ai_team.providers.base import BaseProvider


class RouterProviderTests(unittest.TestCase):
    def test_engineer_route_probe_checks_only_selected_provider(self) -> None:
        router = RoleRouterProvider(
            RoleRoutingProfile(
                role="engineer",
                primary=RouteTarget("codex", "gpt-5.6-terra", "high"),
                fallbacks=(
                    RouteTarget("handsfreecode", "qwen2.5-coder:7b", "default"),
                ),
                allow_write=True,
            ),
            {
                "codex": _StaticProvider(
                    "codex",
                    ready=False,
                    result=_result("codex", False, ProviderErrorType.NETWORK),
                ),
                "handsfreecode": _StaticProvider(
                    "handsfreecode",
                    ready=True,
                    result=_result("handsfreecode", True),
                ),
            },
        )

        self.assertFalse(router.ready_for_route({
            "provider": "codex",
            "model": "gpt-5.6-terra",
            "reasoningEffort": "high",
        }))
        self.assertTrue(router.ready_for_route({
            "provider": "handsfreecode",
            "model": "qwen2.5-coder:7b",
            "reasoningEffort": "default",
        }))

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

    def test_router_preserves_last_native_failure_classification(self) -> None:
        router = RouterProvider(
            [
                _StaticProvider(
                    "codex",
                    ready=True,
                    result=_result("codex", False, ProviderErrorType.RATE_LIMIT),
                )
            ]
        )

        result = router.run(_request())

        self.assertEqual(result.provider, "codex")
        self.assertEqual(result.error_type, ProviderErrorType.RATE_LIMIT)

    def test_write_workflow_never_falls_back_to_local_or_mock_provider(self) -> None:
        router = RouterProvider(
            [
                _StaticProvider("codex", ready=True, result=_result("codex", False, ProviderErrorType.RATE_LIMIT)),
                _StaticProvider("handsfreecode", ready=True, result=_result("handsfreecode", True)),
                _StaticProvider("mock", ready=True, result=_result("mock", True)),
            ]
        )
        request = ProviderRequest(
            workflow="bug-fix-loop",
            prompt="write",
            project_root=Path.cwd(),
            metadata={"writeRequired": True},
        )

        result = router.run(request)

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, ProviderErrorType.RATE_LIMIT)
        self.assertEqual(len(result.data["routeAttempts"]), 1)
        self.assertEqual(result.data["routeAttempts"][0]["provider"], "codex")

    def test_role_router_records_model_reasoning_and_transient_fallback(self) -> None:
        codex = _StaticProvider(
            "codex",
            ready=True,
            result=_result("codex", False, ProviderErrorType.RATE_LIMIT),
        )
        antigravity = _RecordingProvider("antigravity", _result("antigravity", True))
        router = RoleRouterProvider(
            RoleRoutingProfile(
                role="product-manager",
                primary=RouteTarget("codex", "gpt-5.6-terra", "medium"),
                fallbacks=(RouteTarget("antigravity", "Gemini 3.5 Flash (High)", "high"),),
            ),
            {"codex": codex, "antigravity": antigravity},
        )

        result = router.run(_request())

        self.assertTrue(result.success)
        self.assertEqual(result.provider, "antigravity")
        self.assertEqual(result.data["role"], "product-manager")
        self.assertEqual(result.data["selectedModel"], "Gemini 3.5 Flash (High)")
        self.assertEqual(result.data["reasoningEffort"], "high")
        self.assertTrue(result.data["fallbackUsed"])
        self.assertEqual(antigravity.requests[0].metadata["requestedModel"], "Gemini 3.5 Flash (High)")

    def test_role_router_required_provider_preserves_primary_quota_failure(self) -> None:
        antigravity = _StaticProvider(
            "antigravity",
            ready=True,
            result=_result("antigravity", False, ProviderErrorType.RATE_LIMIT),
        )
        codex = _RecordingProvider("codex", _result("codex", True))
        router = RoleRouterProvider(
            RoleRoutingProfile(
                role="product-manager",
                primary=RouteTarget("antigravity", "Gemini 3.5 Flash (High)", "high"),
                fallbacks=(RouteTarget("codex", "gpt-5.6-terra", "medium"),),
            ),
            {"antigravity": antigravity, "codex": codex},
        )

        result = router.run(
            ProviderRequest(
                workflow="bounded-delivery-pm",
                prompt="read only",
                project_root=Path.cwd(),
                metadata={"writeRequired": False, "requiredProvider": "antigravity"},
            )
        )

        self.assertFalse(result.success)
        self.assertEqual(result.provider, "antigravity")
        self.assertEqual(result.error_type, ProviderErrorType.RATE_LIMIT)
        self.assertFalse(result.data["fallbackUsed"])
        self.assertEqual(len(result.data["routeAttempts"]), 1)
        self.assertEqual(codex.requests, [])

    def test_role_router_fails_closed_on_invalid_response(self) -> None:
        fallback = _RecordingProvider("handsfreecode", _result("handsfreecode", True))
        router = RoleRouterProvider(
            RoleRoutingProfile(
                role="reviewer",
                primary=RouteTarget("codex", "gpt-5.5", "high"),
                fallbacks=(RouteTarget("handsfreecode", "qwen2.5-coder:7b", "medium"),),
            ),
            {
                "codex": _StaticProvider(
                    "codex",
                    ready=True,
                    result=_result("codex", False, ProviderErrorType.INVALID_RESPONSE),
                ),
                "handsfreecode": fallback,
            },
        )

        result = router.run(_request())

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, ProviderErrorType.INVALID_RESPONSE)
        self.assertEqual(fallback.requests, [])

    def test_role_router_preserves_quota_failure_when_no_fallback_succeeds(self) -> None:
        router = RoleRouterProvider(
            RoleRoutingProfile(
                role="engineer",
                primary=RouteTarget("codex", "gpt-5.6-terra", "high"),
            ),
            {
                "codex": _StaticProvider(
                    "codex",
                    ready=True,
                    result=_result("codex", False, ProviderErrorType.RATE_LIMIT),
                )
            },
        )

        result = router.run(_request())

        self.assertEqual(result.provider, "codex")
        self.assertEqual(result.error_type, ProviderErrorType.RATE_LIMIT)
        self.assertEqual(result.data["selectedModel"], "gpt-5.6-terra")

    def test_role_router_never_falls_back_for_write_workflow(self) -> None:
        fallback = _RecordingProvider("antigravity", _result("antigravity", True))
        router = RoleRouterProvider(
            RoleRoutingProfile(
                role="engineer",
                primary=RouteTarget("codex", "gpt-5.6-terra", "high"),
                fallbacks=(RouteTarget("antigravity", "Gemini 3.1 Pro (High)", "high"),),
                allow_write=True,
            ),
            {
                "codex": _StaticProvider(
                    "codex",
                    ready=True,
                    result=_result("codex", False, ProviderErrorType.TIMEOUT),
                ),
                "antigravity": fallback,
            },
        )
        request = ProviderRequest(
            workflow="bug-fix-loop",
            prompt="write",
            project_root=Path.cwd(),
            metadata={"writeRequired": True},
        )

        result = router.run(request)

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, ProviderErrorType.TIMEOUT)
        self.assertEqual(fallback.requests, [])

    def test_secondary_review_is_forced_read_only_and_recorded(self) -> None:
        reviewer = _RecordingProvider("antigravity", _result("antigravity", True))
        router = RoleRouterProvider(
            RoleRoutingProfile(
                role="architect",
                primary=RouteTarget("codex", "gpt-5.6-sol", "high"),
                secondary=RouteTarget("antigravity", "Gemini 3.1 Pro (High)", "high"),
            ),
            {
                "codex": _StaticProvider("codex", ready=True, result=_result("codex", True)),
                "antigravity": reviewer,
            },
        )

        result = router.run(_request())

        self.assertTrue(result.data["secondaryReview"]["success"])
        self.assertEqual(result.data["secondaryReview"]["tokenUsage"], 0)
        self.assertFalse(result.data["secondaryReview"]["tokenUsageReported"])
        self.assertFalse(reviewer.requests[0].metadata["writeRequired"])
        self.assertFalse(reviewer.requests[0].metadata["writeAccess"])

    def test_secondary_review_preserves_structured_content_beyond_old_limit(self) -> None:
        structured = '{"schema":"example/v1","payload":"' + ("x" * 3000) + '"}'
        reviewer = _RecordingProvider(
            "codex",
            ProviderResult(provider="codex", success=True, content=structured),
        )
        router = RoleRouterProvider(
            RoleRoutingProfile(
                role="architect",
                primary=RouteTarget("antigravity", "Gemini 3.1 Pro (High)", "high"),
                secondary=RouteTarget("codex", "gpt-5.6-sol", "high"),
            ),
            {
                "antigravity": _StaticProvider(
                    "antigravity",
                    ready=True,
                    result=_result("antigravity", True),
                ),
                "codex": reviewer,
            },
        )

        result = router.run(_request())

        secondary = result.data["secondaryReview"]
        self.assertTrue(secondary["success"])
        self.assertFalse(secondary["contentTruncated"])
        self.assertEqual(secondary["content"], structured)

    def test_secondary_review_fails_closed_when_content_exceeds_bound(self) -> None:
        reviewer = _RecordingProvider(
            "codex",
            ProviderResult(provider="codex", success=True, content="x" * 16_001),
        )
        router = RoleRouterProvider(
            RoleRoutingProfile(
                role="architect",
                primary=RouteTarget("antigravity", "Gemini 3.1 Pro (High)", "high"),
                secondary=RouteTarget("codex", "gpt-5.6-sol", "high"),
            ),
            {
                "antigravity": _StaticProvider(
                    "antigravity",
                    ready=True,
                    result=_result("antigravity", True),
                ),
                "codex": reviewer,
            },
        )

        result = router.run(_request())

        secondary = result.data["secondaryReview"]
        self.assertFalse(secondary["success"])
        self.assertTrue(secondary["contentTruncated"])
        self.assertEqual(secondary["errorType"], ProviderErrorType.INVALID_RESPONSE)


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


class _RecordingProvider(_StaticProvider):
    def __init__(self, name: str, result: ProviderResult) -> None:
        super().__init__(name, ready=True, result=result)
        self.requests: list[ProviderRequest] = []

    def run(self, request: ProviderRequest) -> ProviderResult:
        self.requests.append(request)
        return super().run(request)


def _request() -> ProviderRequest:
    return ProviderRequest(workflow="project-analysis", prompt="hello", project_root=Path.cwd())


def _result(provider: str, success: bool, error_type: ProviderErrorType | None = None) -> ProviderResult:
    return ProviderResult(provider=provider, success=success, error_type=error_type, content=provider)


if __name__ == "__main__":
    unittest.main()
