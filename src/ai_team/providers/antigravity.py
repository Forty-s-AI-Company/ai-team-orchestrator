from __future__ import annotations

from .base import BaseProvider, ProviderErrorType, ProviderRequest, ProviderResult


class AntigravityProvider(BaseProvider):
    name = "antigravity"

    def ready(self) -> bool:
        return False

    def run(self, request: ProviderRequest) -> ProviderResult:
        return ProviderResult(
            provider=self.name,
            success=False,
            error_type=ProviderErrorType.UNKNOWN,
            content="Antigravity provider is not implemented in the OpenHands reset baseline.",
        )
