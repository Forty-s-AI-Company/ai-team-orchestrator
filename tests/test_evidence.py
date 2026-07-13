from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_team.core.evidence import (
    EvidenceError,
    EvidencePolicy,
    collect_project_evidence,
    parse_evidence_policy,
    validate_analysis_grounding,
)
from ai_team.core.project_loader import LoadedProject, ProjectInfo, ProjectProfile


def loaded_project(root: Path) -> LoadedProject:
    profile = ProjectProfile(project=ProjectInfo(name="evidence-test"))
    config = root / ".ai-team" / "project.yaml"
    config.parent.mkdir(exist_ok=True)
    config.write_text("project:\n  name: evidence-test\n", encoding="utf-8")
    return LoadedProject(
        profile=profile,
        config_path=config,
        project_dir=root,
        root=root,
        current_branch="master",
        commit_sha="a" * 40,
    )


class EvidenceTests(unittest.TestCase):
    def test_sensitive_binary_and_symlink_content_is_never_collected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside_tmp:
            root = Path(tmp)
            outside = Path(outside_tmp) / "README-outside.md"
            outside.write_text("outside-secret-value", encoding="utf-8")
            (root / ".env").write_text("API_KEY=env-secret-value", encoding="utf-8")
            (root / "credentials.json").write_text('{"password":"credential-secret"}', encoding="utf-8")
            (root / "README-binary.txt").write_bytes(b"text\x00binary-secret")
            (root / "README-unknown.blob").write_text("unknown-secret-value", encoding="utf-8")
            (root / "README-link.md").symlink_to(outside)
            snapshot = collect_project_evidence(
                loaded_project(root),
                EvidencePolicy(include=("README*",)),
            )

            serialized = str(snapshot.manifest) + snapshot.prompt_section
            self.assertNotIn("env-secret-value", serialized)
            self.assertNotIn("credential-secret", serialized)
            self.assertNotIn("binary-secret", serialized)
            self.assertNotIn("outside-secret-value", serialized)
            self.assertNotIn("unknown-secret-value", serialized)
            self.assertEqual(snapshot.manifest["fileCount"], 0)
            self.assertGreater(snapshot.manifest["excludedReasonCounts"]["sensitive_path"], 0)
            self.assertGreater(snapshot.manifest["excludedReasonCounts"]["symlink"], 0)
            self.assertGreater(
                snapshot.manifest["excludedReasonCounts"]["unknown_or_binary_extension"],
                0,
            )

    def test_root_filename_pattern_does_not_match_nested_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "docs"
            nested.mkdir()
            (root / "README.md").write_text("root readme", encoding="utf-8")
            (nested / "README.md").write_text("nested readme", encoding="utf-8")
            snapshot = collect_project_evidence(
                loaded_project(root),
                EvidencePolicy(include=("README*",)),
            )
            self.assertEqual(
                [item["path"] for item in snapshot.manifest["files"]],
                ["README.md"],
            )

    def test_parent_traversal_pattern_is_rejected(self) -> None:
        with self.assertRaises(EvidenceError):
            parse_evidence_policy({"include": ["../outside.json"]})

    def test_non_allowlisted_pattern_is_rejected(self) -> None:
        with self.assertRaises(EvidenceError):
            parse_evidence_policy({"include": ["**/*.ts"]})

    def test_oversized_candidate_is_not_read_into_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("x" * 300, encoding="utf-8")
            snapshot = collect_project_evidence(
                loaded_project(root),
                EvidencePolicy(include=("README.md",), max_candidate_bytes=128),
            )
            self.assertEqual(snapshot.manifest["fileCount"], 0)
            self.assertEqual(snapshot.manifest["excludedReasonCounts"]["oversized_candidate"], 1)

    def test_per_file_and_total_limits_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("r" * 200, encoding="utf-8")
            (root / "package.json").write_text('{"name":"' + "p" * 180 + '"}', encoding="utf-8")
            snapshot = collect_project_evidence(
                loaded_project(root),
                EvidencePolicy(
                    include=("package.json", "README.md"),
                    per_file_max_bytes=100,
                    total_content_max_bytes=150,
                    max_candidate_bytes=512,
                ),
            )
            self.assertEqual(snapshot.manifest["contentBytes"], 150)
            self.assertTrue(snapshot.manifest["files"][0]["truncated"])
            self.assertTrue(snapshot.manifest["files"][1]["truncated"])

    def test_grounding_validation_rejects_unsupported_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "package.json").write_text(
                '{"name":"sample-app","version":"1.2.3","dependencies":{"next":"1"},'
                '"devDependencies":{"typescript":"1"}}',
                encoding="utf-8",
            )
            (root / "tsconfig.json").write_text("{}", encoding="utf-8")
            workflow = root / ".github" / "workflows" / "ci.yml"
            workflow.parent.mkdir(parents=True)
            workflow.write_text(
                "name: CI\njobs:\n  checks:\n    steps:\n      - run: npm run lint\n      - run: npm test\n",
                encoding="utf-8",
            )
            snapshot = collect_project_evidence(
                loaded_project(root),
                EvidencePolicy(include=("package.json", "tsconfig.json", ".github/workflows/*.yml")),
            )
            self.assertIn(
                "Copy detected package name and version exactly",
                snapshot.prompt_section,
            )
            passed = validate_analysis_grounding(
                "sample-app 1.2.3 uses Node.js, Next.js, and TypeScript from package.json. Coverage is unknown.",
                snapshot,
                provider_success=True,
            )
            negated_inference = validate_analysis_grounding(
                "sample-app 1.2.3 uses Node.js, Next.js, and TypeScript from package.json. "
                "No additional dependency is inferred from project structure. Coverage is unknown.",
                snapshot,
                provider_success=True,
            )
            unsupported_inference = validate_analysis_grounding(
                "sample-app 1.2.3 uses Node.js, Next.js, and TypeScript from package.json. "
                "Prettier is inferred from project structure. Coverage is unknown.",
                snapshot,
                provider_success=True,
            )
            failed = validate_analysis_grounding(
                "sample-app 1.2.3 uses Node.js, Next.js, TypeScript, requirements.txt, and 30% test coverage. "
                "Prettier is not in evidence, but inferred from project structure. "
                "Set up CI, update all dependencies, and add a database seeding script.\n"
                "### Planned Changes\nAdd more comprehensive tests and update README instructions to deploy to Vercel. "
                "The CI workflow ensures the codebase remains clean and functional.",
                snapshot,
                provider_success=True,
            )
            self.assertEqual(passed["status"], "passed")
            self.assertEqual(negated_inference["status"], "passed")
            self.assertIn(
                "inferred project claim is not evidence-backed",
                unsupported_inference["unsupportedClaims"],
            )
            self.assertEqual(failed["status"], "failed")
            for expected in (
                "requirements.txt",
                "coverage percentage",
                "CI setup contradicts the included workflow evidence",
                "dependency update recommendation lacks freshness evidence",
                "inferred project claim is not evidence-backed",
                "planned changes are not authorized by analysis evidence",
                "test expansion recommendation lacks coverage evidence",
                "CI outcome claimed without execution evidence",
                "forbidden database seed recommendation",
                "forbidden deployment recommendation",
            ):
                self.assertIn(expected, failed["unsupportedClaims"])
            self.assertEqual(
                snapshot.facts["ciWorkflows"][0]["commands"],
                ["npm run lint", "npm test"],
            )
            missing_package = validate_analysis_grounding(
                "Node.js, Next.js, and TypeScript from package.json. Coverage unknown.",
                snapshot,
                provider_success=True,
            )
            self.assertEqual(
                missing_package["missingExpectedFacts"],
                ["packageName", "packageVersion"],
            )

    def test_policy_blockers_and_not_run_commands_are_not_recommendations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "package.json").write_text(
                '{"dependencies":{"next":"1"},"devDependencies":{"typescript":"1"}}',
                encoding="utf-8",
            )
            (root / "tsconfig.json").write_text("{}", encoding="utf-8")
            snapshot = collect_project_evidence(
                loaded_project(root),
                EvidencePolicy(include=("package.json", "tsconfig.json")),
            )
            validation = validate_analysis_grounding(
                "Node.js, Next.js, TypeScript from package.json. Coverage unknown.\n"
                "### Configured Validation Commands Marked Not Run\n"
                "- `npm run db:migrate:deploy`: deploys migrations in CI.\n"
                "### Policy Blockers\n"
                "- Deployment: Disallowed by the project contract.",
                snapshot,
                provider_success=True,
            )
            self.assertEqual(validation["status"], "passed")
            self.assertEqual(validation["disallowedActionRecommendations"], [])


if __name__ == "__main__":
    unittest.main()
