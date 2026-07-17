from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ai_team.core.project_loader import LoadedProject


ALLOWED_TEST_DATABASE_COMMANDS = {"npm run db:migrate:deploy"}
LOOPBACK_DATABASE_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class TestDatabaseSettings:
    enabled: bool = False
    url_env: str = "AI_TEAM_TEST_DATABASE_URL"
    allowed_hosts: tuple[str, ...] = ("localhost", "127.0.0.1", "::1")
    required_database_suffixes: tuple[str, ...] = ("_test", "_dev")
    bootstrap_commands: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrustedDevSettings:
    enabled: bool = False
    checkpoint_on_validation_failure: bool = False
    preserve_dependency_tree: bool = False
    cleanup_worktree_after_merge: bool = False
    min_iterations: int = 2
    min_repair_attempts: int = 1
    min_token_usage: int = 120_000
    min_stage_timeout_seconds: int = 180
    test_database: TestDatabaseSettings = TestDatabaseSettings()


def load_trusted_dev_settings(
    settings: dict[str, Any],
    project_path: Path,
    *,
    requested: bool,
) -> TrustedDevSettings:
    """Load an explicit development-only autonomy profile.

    Merely having a configuration block never enables the mode. The caller
    must also pass ``--trusted-dev-autopilot`` and the project must be in the
    configured path allowlist.
    """

    if not requested:
        return TrustedDevSettings()
    raw = settings.get("trusted_dev_autopilot")
    if not isinstance(raw, dict):
        raise ValueError("trusted-dev-autopilot is not configured")
    projects = raw.get("enabled_projects")
    if not isinstance(projects, list) or not projects:
        raise ValueError("trusted-dev-autopilot requires enabled_projects")
    resolved_project = project_path.expanduser().resolve()
    enabled_projects = {
        Path(value).expanduser().resolve()
        for value in projects
        if isinstance(value, str) and value.strip()
    }
    if resolved_project not in enabled_projects:
        raise ValueError("project is not allowlisted for trusted-dev-autopilot")

    database_raw = raw.get("test_database")
    database = _test_database_settings(database_raw if isinstance(database_raw, dict) else {})
    return TrustedDevSettings(
        enabled=True,
        checkpoint_on_validation_failure=raw.get("checkpoint_on_validation_failure", True) is True,
        preserve_dependency_tree=raw.get("preserve_dependency_tree", True) is True,
        cleanup_worktree_after_merge=raw.get("cleanup_worktree_after_merge", True) is True,
        min_iterations=_positive_int(raw.get("min_iterations"), 6),
        min_repair_attempts=_positive_int(raw.get("min_repair_attempts"), 4),
        min_token_usage=_positive_int(raw.get("min_token_usage"), 360_000),
        min_stage_timeout_seconds=_positive_int(raw.get("min_stage_timeout_seconds"), 1_200),
        test_database=database,
    )


def validate_test_database_url(url: str, settings: TestDatabaseSettings) -> None:
    parsed = urlparse(url)
    database_name = parsed.path.lstrip("/")
    if parsed.scheme not in {"postgres", "postgresql"}:
        raise ValueError("trusted-dev test database must use PostgreSQL")
    if (parsed.hostname or "").lower() not in {host.lower() for host in settings.allowed_hosts}:
        raise ValueError("trusted-dev test database must use an allowlisted loopback host")
    if not database_name or not any(
        database_name.lower().endswith(suffix.lower())
        for suffix in settings.required_database_suffixes
    ):
        raise ValueError("trusted-dev database name must end in an approved development/test suffix")


def validate_trusted_dev_project(project: LoadedProject) -> None:
    safety = project.profile.safety
    if project.profile.project.stage != "development":
        raise ValueError("trusted-dev-autopilot requires project.stage=development")
    if any((
        safety.allow_deploy,
        safety.allow_database_migration,
        safety.allow_database_seed,
        safety.allow_destructive_commands,
    )):
        raise ValueError(
            "trusted-dev-autopilot requires deploy, production database, seed, and destructive permissions to remain disabled"
        )
    if not safety.require_disposable_worktree_for_writes:
        raise ValueError("trusted-dev-autopilot requires disposable worktrees for writes")


def _test_database_settings(raw: dict[str, Any]) -> TestDatabaseSettings:
    enabled = raw.get("enabled", False) is True
    url_env = raw.get("url_env", "AI_TEAM_TEST_DATABASE_URL")
    if not isinstance(url_env, str) or not url_env.strip():
        raise ValueError("trusted-dev test database url_env must be non-empty")
    hosts = _string_tuple(raw.get("allowed_hosts"), tuple(sorted(LOOPBACK_DATABASE_HOSTS)))
    if not hosts or any(host.lower() not in LOOPBACK_DATABASE_HOSTS for host in hosts):
        raise ValueError("trusted-dev test database hosts must be loopback-only")
    suffixes = _string_tuple(raw.get("required_database_suffixes"), ("_test", "_dev"))
    if not suffixes or any(not suffix.startswith("_") for suffix in suffixes):
        raise ValueError("trusted-dev database suffixes must be explicit underscore suffixes")
    commands = _string_tuple(raw.get("bootstrap_commands"), ())
    if any(command not in ALLOWED_TEST_DATABASE_COMMANDS for command in commands):
        raise ValueError("trusted-dev test database bootstrap command is not allowlisted")
    if enabled and not commands:
        raise ValueError("enabled trusted-dev test database requires bootstrap_commands")
    return TestDatabaseSettings(
        enabled=enabled,
        url_env=url_env.strip(),
        allowed_hosts=hosts,
        required_database_suffixes=suffixes,
        bootstrap_commands=commands,
    )


def _string_tuple(value: object, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return fallback
    if not isinstance(value, list) or not value:
        return ()
    normalized = tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    if len(normalized) != len(value) or len(set(normalized)) != len(normalized):
        raise ValueError("trusted-dev string lists must contain unique non-empty strings")
    return normalized


def _positive_int(value: object, fallback: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else fallback
