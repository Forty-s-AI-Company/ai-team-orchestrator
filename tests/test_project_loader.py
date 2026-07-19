from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from ai_team.core.project_loader import ProjectConfigError, load_project


def init_git_project(root: Path, project_root_value: str = ".") -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    ai_team = root / ".ai-team"
    ai_team.mkdir()
    (ai_team / "project.yaml").write_text(
        f"""project:
  name: sample
  root: "{project_root_value}"
  stage: development

repository:
  protected_branches:
    - master
    - main

commands:
  lint: npm run lint

safety:
  allow_git_push: false
  allow_deploy: false
  allow_database_migration: false
  allow_database_seed: false
  allow_destructive_commands: false
""",
        encoding="utf-8",
    )


class ProjectLoaderTests(unittest.TestCase):
    def test_loads_valid_project_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            loaded = load_project(root)
            self.assertEqual(loaded.profile.project.name, "sample")
            self.assertEqual(loaded.root, root.resolve())

    def test_loads_normalized_additional_validation_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            profile = root / ".ai-team" / "project.yaml"
            profile.write_text(
                profile.read_text(encoding="utf-8").replace(
                    "  lint: npm run lint\n",
                    "  lint: npm run lint\n  additional_validation:\n    - '  npm run e2e:smoke  '\n",
                ),
                encoding="utf-8",
            )

            loaded = load_project(root)

            self.assertEqual(loaded.profile.commands.additional_validation, ["npm run e2e:smoke"])

    def test_loads_staging_external_qa_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            profile = root / ".ai-team" / "project.yaml"
            profile.write_text(
                profile.read_text(encoding="utf-8").replace(
                    "safety:\n",
                    "external_qa:\n  enabled: true\n  environment: staging\n  command: npm run qa:payuni:sandbox\n\nsafety:\n",
                ),
                encoding="utf-8",
            )

            loaded = load_project(root)

            self.assertTrue(loaded.profile.external_qa.enabled)
            self.assertEqual(loaded.profile.external_qa.environment, "staging")
            self.assertEqual(loaded.profile.external_qa.command, "npm run qa:payuni:sandbox")

    def test_rejects_duplicate_or_empty_additional_validation_commands(self) -> None:
        for values in (["npm run e2e:smoke", "npm run e2e:smoke"], [""]):
            with self.subTest(values=values), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "project"
                root.mkdir()
                init_git_project(root)
                profile = root / ".ai-team" / "project.yaml"
                rendered = "\n".join(f"    - {value!r}" for value in values)
                profile.write_text(
                    profile.read_text(encoding="utf-8").replace(
                        "  lint: npm run lint\n",
                        f"  lint: npm run lint\n  additional_validation:\n{rendered}\n",
                    ),
                    encoding="utf-8",
                )

                with self.assertRaises(ProjectConfigError):
                    load_project(root)

    def test_blocks_project_root_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root, project_root_value="../outside")
            with self.assertRaises(ProjectConfigError):
                load_project(root, allowlist=[root])

    def test_denies_write_on_protected_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            loaded = load_project(root)
            loaded.current_branch = "master"
            with self.assertRaises(ProjectConfigError):
                loaded.assert_write_allowed("bug-fix-loop")

    def test_denies_run_agent_on_protected_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            loaded = load_project(root)
            loaded.current_branch = "master"
            with self.assertRaises(ProjectConfigError):
                loaded.assert_agent_run_allowed("project-analysis")

    def test_denies_write_on_primary_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            loaded = load_project(root)
            loaded.current_branch = "feature/test"
            with self.assertRaises(ProjectConfigError):
                loaded.assert_write_allowed("bug-fix-loop")
            with self.assertRaises(ProjectConfigError):
                loaded.assert_agent_run_allowed("project-analysis")

    def test_allows_write_on_disposable_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            init_git_project(root)
            git_marker = root / ".git"
            git_marker_dir = root / ".git-dir"
            if git_marker.is_dir():
                git_marker.rename(git_marker_dir)
            git_marker.write_text(f"gitdir: {git_marker_dir.as_posix()}\n", encoding="utf-8")
            loaded = load_project(root)
            loaded.current_branch = "feature/test"
            loaded.assert_write_allowed("bug-fix-loop")
            loaded.assert_agent_run_allowed("project-analysis")


if __name__ == "__main__":
    unittest.main()
