from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class ProjectConfigError(ValueError):
    """Raised when a project profile is missing, malformed, or unsafe."""


class ProjectInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    root: str = "."
    stage: str = "development"

    @field_validator("name", "root", "stage")
    @classmethod
    def non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must not be empty")
        return value.strip()


class RepositoryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protected_branches: list[str] = Field(default_factory=lambda: ["main", "master"])


class CommandSet(BaseModel):
    model_config = ConfigDict(extra="allow")

    install: str | None = None
    lint: str | None = None
    typecheck: str | None = None
    test: str | None = None
    build: str | None = None
    additional_validation: list[str] = Field(default_factory=list)

    @field_validator("additional_validation")
    @classmethod
    def non_empty_additional_validation(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("additional validation commands must not be empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("additional validation commands must be unique")
        return normalized


class SafetyPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allow_git_push: bool = False
    allow_deploy: bool = False
    allow_database_migration: bool = False
    allow_database_seed: bool = False
    allow_destructive_commands: bool = False
    require_disposable_worktree_for_writes: bool = True


class ExternalQAConfig(BaseModel):
    """Policy for a human-only external QA review gate.

    ``enabled`` requires a reviewer attestation; it never authorizes the
    orchestrator to execute a payment, refund, browser, or provider command.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    execution_mode: Literal["manual-attestation-only"] = "manual-attestation-only"
    environment: str = "staging"
    command: str = "npm run qa:payuni:sandbox"
    run_once_per_revision: bool = True
    reviewer_role: str = "delivery-qa"
    production_requires_human_approval: bool = True
    trigger_paths: list[str] = Field(default_factory=list)

    @field_validator("environment", "command", "reviewer_role")
    @classmethod
    def non_empty_external_qa_value(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must not be empty")
        return value.strip()

    @field_validator("trigger_paths")
    @classmethod
    def safe_external_qa_trigger_paths(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            path = value.strip().replace("\\", "/")
            parts = tuple(part for part in path.split("/") if part)
            if not path or path.startswith("/") or ".." in parts or path.startswith(".git"):
                raise ValueError("external QA trigger paths must be safe project-relative prefixes")
            normalized.append(path)
        if len(normalized) > 128 or len(normalized) != len(set(normalized)):
            raise ValueError("external QA trigger paths must be unique and contain at most 128 items")
        return normalized


class StagingOperationsPolicy(BaseModel):
    """Explicit opt-in for deterministic non-production external operations.

    This policy is deliberately separate from ``SafetyPolicy``.  Enabling a
    staging operation must never turn on the corresponding production-capable
    project flag.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    environment: str = "staging"
    database_url_env: str = "DATABASE_URL"
    database_url_sha256: str | None = None
    allow_migration: bool = False
    allow_seed: bool = False
    allow_preview_deploy: bool = False
    preview_environment_variable: str = "VERCEL_ENV"

    @field_validator("environment", "database_url_env", "preview_environment_variable")
    @classmethod
    def non_empty_staging_value(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must not be empty")
        return value.strip()

    @field_validator("database_url_sha256")
    @classmethod
    def valid_database_fingerprint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
            raise ValueError("database_url_sha256 must be a SHA-256 hex digest")
        return normalized


class ProjectProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: ProjectInfo
    repository: RepositoryPolicy = Field(default_factory=RepositoryPolicy)
    commands: CommandSet = Field(default_factory=CommandSet)
    safety: SafetyPolicy = Field(default_factory=SafetyPolicy)
    external_qa: ExternalQAConfig = Field(default_factory=ExternalQAConfig)
    staging_operations: StagingOperationsPolicy = Field(default_factory=StagingOperationsPolicy)


class LoadedProject(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    profile: ProjectProfile
    config_path: Path
    project_dir: Path
    root: Path
    current_branch: str | None
    commit_sha: str | None

    def is_branch_protected(self) -> bool:
        if not self.current_branch:
            return False
        return self.current_branch in self.profile.repository.protected_branches

    def is_disposable_worktree(self) -> bool:
        git_marker = self.project_dir / ".git"
        return git_marker.is_file()

    def assert_write_allowed(self, workflow_name: str) -> None:
        if self.is_branch_protected():
            raise ProjectConfigError(
                f"workflow '{workflow_name}' requires writes but branch "
                f"'{self.current_branch}' is protected"
            )

        safety = self.profile.safety
        if safety.require_disposable_worktree_for_writes and not self.is_disposable_worktree():
            raise ProjectConfigError(
                f"workflow '{workflow_name}' requires a disposable git worktree for non-dry-run writes"
            )

        if workflow_name == "bug-fix-loop" and safety.allow_destructive_commands:
            raise ProjectConfigError("destructive commands are not allowed for bug-fix-loop")

    def assert_agent_run_allowed(self, workflow_name: str) -> None:
        if self.is_branch_protected():
            raise ProjectConfigError(
                f"workflow '{workflow_name}' run-agent mode is denied on protected branch "
                f"'{self.current_branch}'"
            )
        if not self.is_disposable_worktree():
            raise ProjectConfigError(
                f"workflow '{workflow_name}' run-agent mode requires a disposable git worktree"
            )


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProjectConfigError(f"cannot read project profile: {path}") from exc
    except yaml.YAMLError as exc:
        raise ProjectConfigError(f"invalid YAML in project profile: {path}") from exc

    if not isinstance(data, dict):
        raise ProjectConfigError("project profile must be a YAML mapping")
    return data


def _resolve_allowlist(project_dir: Path, explicit: list[str | Path] | None) -> list[Path]:
    values: list[str | Path] = []
    if explicit:
        values.extend(explicit)

    env_value = os.environ.get("AI_TEAM_WORKSPACE_ALLOWLIST")
    if env_value:
        values.extend(part for part in env_value.split(os.pathsep) if part.strip())

    if not values:
        values.append(project_dir.parent)

    return [Path(value).expanduser().resolve() for value in values]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _assert_under_allowlist(path: Path, allowlist: list[Path]) -> None:
    if any(_is_relative_to(path, allowed) for allowed in allowlist):
        return
    allowed_text = ", ".join(str(item) for item in allowlist)
    raise ProjectConfigError(f"project root escapes workspace allowlist: {path}; allowed: {allowed_text}")


def current_git_branch(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    branch = result.stdout.strip()
    return branch or None


def current_git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    commit = result.stdout.strip()
    return commit or None


def load_project(project_path: str | Path, allowlist: list[str | Path] | None = None) -> LoadedProject:
    project_dir = Path(project_path).expanduser().resolve()
    if not project_dir.exists():
        raise ProjectConfigError(f"project does not exist: {project_dir}")
    if not (project_dir / ".git").exists():
        raise ProjectConfigError(f"project is not a git repository: {project_dir}")

    config_path = project_dir / ".ai-team" / "project.yaml"
    if not config_path.exists():
        raise ProjectConfigError(f"missing project profile: {config_path}")

    try:
        profile = ProjectProfile.model_validate(_load_yaml(config_path))
    except ValidationError as exc:
        raise ProjectConfigError(str(exc)) from exc

    root_value = Path(profile.project.root)
    root = (project_dir / root_value).resolve() if not root_value.is_absolute() else root_value.resolve()
    resolved_allowlist = _resolve_allowlist(project_dir, allowlist)
    _assert_under_allowlist(root, resolved_allowlist)

    if not root.exists():
        raise ProjectConfigError(f"configured project root does not exist: {root}")

    return LoadedProject(
        profile=profile,
        config_path=config_path,
        project_dir=project_dir,
        root=root,
        current_branch=current_git_branch(root),
        commit_sha=current_git_commit(root),
    )
