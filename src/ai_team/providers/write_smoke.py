from __future__ import annotations

from pathlib import Path
import subprocess

from .base import BaseProvider, ProviderErrorType, ProviderRequest, ProviderResult


SMOKE_WORKFLOW = "bug-fix-loop"
SMOKE_RELATIVE_PATH = Path("docs/ai-team-smoke/isolated-write-smoke.md")
SMOKE_CONTENT = """# AI Team Isolated Write Smoke

This file is generated only inside a disposable Git worktree.

- Purpose: validate guarded add, commit, push, and pull request gates.
- Scope: documentation-only smoke evidence.
- Production deploy: prohibited.
- Real payment: prohibited.
- Destructive migration: prohibited.
"""


class WriteSmokeProvider(BaseProvider):
    """Deterministic provider for validating the isolated write control path."""

    name = "write-smoke"

    def ready(self) -> bool:
        return True

    def run(self, request: ProviderRequest) -> ProviderResult:
        denial = _validate_request(request)
        if denial:
            return ProviderResult(
                provider=self.name,
                success=False,
                error_type=ProviderErrorType.EXTERNAL_REQUIRED,
                content=denial,
                data={"runMode": request.run_mode, "writePerformed": False},
            )

        target = (request.project_root / SMOKE_RELATIVE_PATH).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("x", encoding="utf-8") as stream:
                stream.write(SMOKE_CONTENT)
        except FileExistsError:
            return ProviderResult(
                provider=self.name,
                success=False,
                error_type=ProviderErrorType.EXTERNAL_REQUIRED,
                content="write-smoke target already exists; refusing to overwrite",
                data={"runMode": request.run_mode, "writePerformed": False},
            )
        return ProviderResult(
            provider=self.name,
            success=True,
            content="isolated documentation smoke written",
            data={
                "runMode": request.run_mode,
                "writePerformed": True,
                "relativePath": SMOKE_RELATIVE_PATH.as_posix(),
                "scope": "documentation-only",
            },
        )


def _validate_request(request: ProviderRequest) -> str | None:
    if request.workflow != SMOKE_WORKFLOW:
        return f"write-smoke only supports workflow '{SMOKE_WORKFLOW}'"
    if request.dry_run:
        return "write-smoke requires a non-dry-run isolated executor"
    if request.run_mode != "create-only":
        return "write-smoke only supports create-only mode"
    if not _is_linked_worktree(request.project_root):
        return "write-smoke requires a disposable linked worktree"
    return None


def _is_linked_worktree(project_root: Path) -> bool:
    if not (project_root / ".git").is_file():
        return False
    try:
        top_level = _git_path(project_root, "--show-toplevel")
        git_dir = _git_path(project_root, "--git-dir")
        common_dir = _git_path(project_root, "--git-common-dir")
    except (OSError, subprocess.SubprocessError):
        return False
    return top_level == project_root.resolve() and git_dir != common_dir


def _git_path(project_root: Path, argument: str) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", argument],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    )
    value = Path(result.stdout.strip())
    return (project_root / value).resolve() if not value.is_absolute() else value.resolve()
