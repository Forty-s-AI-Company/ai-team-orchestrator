"""Human-only gate for external QA.

External payment QA is intentionally outside autonomous orchestration.  This
module builds a deterministic review requirement and does not execute commands,
read environment files, or create receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_team.core.project_loader import LoadedProject


SCHEMA = "ai-team-external-qa-manual-review/v1"
MANUAL_ATTESTATION_ONLY = "manual-attestation-only"
HUMAN_ATTESTATION_REQUIRED = "human-attestation-required"


@dataclass(frozen=True)
class ExternalQAResult:
    status: str
    result: dict[str, Any]
    receipt_path: Path | None = None


def run_external_qa(
    loaded: LoadedProject,
    revision: str,
    report_dir: Path,
    *,
    prior: dict[str, Any] | None = None,
    execute: bool = True,
    timeout_seconds: int = 1_200,
) -> ExternalQAResult:
    """Return the fixed human-attestation requirement for an enabled policy.

    Arguments retained for caller compatibility are deliberately ignored.  In
    particular, prior results cannot attest a revision and ``report_dir`` is
    never inspected or written.
    """

    del report_dir, prior, execute, timeout_seconds
    config = loaded.profile.external_qa
    base = {
        "schema": SCHEMA,
        "revision": revision,
        "executionMode": MANUAL_ATTESTATION_ONLY,
        "executionAttempted": False,
        "reviewerRole": config.reviewer_role,
    }
    if not config.enabled:
        return ExternalQAResult("disabled", {**base, "status": "disabled"})
    return ExternalQAResult(
        "review-required",
        {
            **base,
            "status": "review-required",
            "reason": HUMAN_ATTESTATION_REQUIRED,
        },
    )
