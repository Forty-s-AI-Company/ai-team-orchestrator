from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_team.core.delivery import DeliveryOptions, TrustedTask, run_delivery_cycle
from ai_team.cli import _resolve_codex_executable, build_parser
from ai_team.providers.base import BaseProvider, ProviderRequest, ProviderResult


class DeliveryTests(unittest.TestCase):
    def test_supervisor_delivery_flag_is_parsed(self) -> None:
        args = build_parser().parse_args(["supervise", "project", "--delivery"])
        self.assertTrue(args.delivery)

    def test_codex_auto_native_falls_back_without_extension(self) -> None:
        with patch("ai_team.cli.Path.home", return_value=Path("Z:/missing-home")):
            self.assertEqual(_resolve_codex_executable("auto-native"), "codex")

    def test_trusted_task_runs_in_disposable_worktree_and_persists_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            _init_project(root)
            task = TrustedTask("safe-doc", "Safe doc update", 10, "test", "low", "write safely", ["docs/safe.md"], ["git diff --check"], True)
            options = DeliveryOptions(root, _WritingProvider("docs/safe.md"), [tmp], Path(tmp) / "reports", Path(tmp) / "state.json", Path(tmp) / "queue.json", True)
            with patch("ai_team.core.delivery.discover_trusted_tasks", return_value=[task]):
                result = run_delivery_cycle(options, 1)
            self.assertEqual(result["status"], "completed", result)
            self.assertIn("safe-doc", result["completedTaskIds"])
            self.assertNotEqual(Path(result["worktreePath"]), root)
            self.assertTrue(result["commitResult"]["committed"])

    def test_completed_task_is_not_resumed_twice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            _init_project(root)
            task = TrustedTask("done", "Done", 1, "test", "low", "noop", ["docs/safe.md"], [], True)
            state = Path(tmp) / "state.json"
            state.write_text(json.dumps({"revision": 1, "completedTaskIds": ["done"]}), encoding="utf-8")
            options = DeliveryOptions(root, _WritingProvider("docs/safe.md"), [tmp], Path(tmp) / "reports", state, Path(tmp) / "queue.json", True)
            with patch("ai_team.core.delivery.discover_trusted_tasks", return_value=[task]):
                result = run_delivery_cycle(options, 2)
            self.assertEqual(result["status"], "idle")
            self.assertIsNone(result["currentTask"])

    def test_out_of_scope_write_is_not_committed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            _init_project(root)
            task = TrustedTask("scope", "Scope", 1, "test", "low", "write", ["docs/allowed.md"], ["git diff --check"], True)
            options = DeliveryOptions(root, _WritingProvider("src/outside.txt"), [tmp], Path(tmp) / "reports", Path(tmp) / "state.json", Path(tmp) / "queue.json", True)
            with patch("ai_team.core.delivery.discover_trusted_tasks", return_value=[task]):
                result = run_delivery_cycle(options, 1)
            self.assertEqual(result["status"], "attention-required")
            self.assertFalse(result["commitResult"]["committed"])


class _WritingProvider(BaseProvider):
    name = "delivery-test"

    def __init__(self, relative_path: str) -> None:
        self.relative_path = relative_path

    def ready(self) -> bool:
        return True

    def run(self, request: ProviderRequest) -> ProviderResult:
        target = request.project_root / self.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("safe\n", encoding="utf-8")
        return ProviderResult(provider=self.name, success=True, content="done")


def _init_project(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "AI Team Test"], cwd=root, check=True)
    profile = root / ".ai-team" / "project.yaml"
    profile.parent.mkdir()
    profile.write_text(
        """project:\n  name: sample\n  root: \".\"\n  stage: development\nrepository:\n  protected_branches: [master, main]\ncommands:\n  test: npm run test\nsafety:\n  allow_git_push: true\n  allow_deploy: false\n  allow_database_migration: false\n  allow_database_seed: false\n  allow_destructive_commands: false\n""",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
