from __future__ import annotations

import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_team.providers import (
    AntigravityProvider,
    AntigravitySettings,
    CodexProvider,
    CodexSettings,
    ProviderErrorType,
    ProviderRequest,
    WriteSmokeProvider,
)
from ai_team.providers.cli_common import CliProviderSettings, run_cli_command
from ai_team.providers.antigravity import _compact_prompt


class CliProviderTests(unittest.TestCase):
    def test_codex_workspace_write_requires_linked_worktree_marker(self) -> None:
        provider = CodexProvider(
            CodexSettings(
                executable=sys.executable,
                status_args=["--version"],
                quota_args=[],
                run_args=["-c", "print('read')"],
                write_run_args=["-c", "print('write')"],
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            denied = provider.run(
                ProviderRequest("bug-fix-loop", "task", root, metadata={"writeRequired": True})
            )
            (root / ".git").write_text("gitdir: test", encoding="utf-8")
            allowed = provider.run(
                ProviderRequest("bug-fix-loop", "task", root, metadata={"writeRequired": True})
            )

        self.assertFalse(denied.success)
        self.assertIn("disposable linked worktree", denied.content)
        self.assertTrue(allowed.success)
        self.assertIn("write", allowed.content)

    def test_codex_quota_exhaustion_parses_reset_time(self) -> None:
        provider = CodexProvider(
            CodexSettings(
                executable=sys.executable,
                status_args=["--version"],
                quota_args=[
                    "-c",
                    "print(\"You've hit your usage limit. try again at Feb 23rd, 2026 9:01 PM.\")",
                ],
                run_args=["-c", "print('should-not-run')"],
            )
        )

        result = provider.run(_request())

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, ProviderErrorType.RATE_LIMIT)
        self.assertTrue(result.data["quotaExhausted"])
        self.assertIn("Feb 23rd, 2026 9:01 PM", str(result.data["resetTime"]))

    def test_antigravity_quota_exhaustion_parses_reset_time(self) -> None:
        provider = AntigravityProvider(
            AntigravitySettings(
                executable=sys.executable,
                status_args=["--version"],
                quota_args=[
                    "-c",
                    "print('Error: HTTP 429 Too Many Requests RESOURCE_EXHAUSTED Reset Time: 2026-07-12 08:00:00 (Local Time)')",
                ],
            )
        )

        result = provider.run(_request())

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, ProviderErrorType.RATE_LIMIT)
        self.assertTrue(result.data["quotaExhausted"])
        self.assertEqual(result.data["resetTime"], "2026-07-12 08:00:00")

    def test_antigravity_individual_quota_resets_in_is_rate_limit(self) -> None:
        provider = AntigravityProvider(
            AntigravitySettings(
                executable=sys.executable,
                status_args=["--version"],
                quota_args=[],
                run_args=[
                    "-c",
                    "import sys; sys.stderr.write('Error: Individual quota reached. Resets in 1h51m33s.'); sys.exit(1)",
                ],
                execution_enabled=True,
            )
        )

        result = provider.run(_request())

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, ProviderErrorType.RATE_LIMIT)
        self.assertEqual(result.data["resetTime"], "1h51m33s")

    def test_antigravity_execution_disabled_is_external_required(self) -> None:
        provider = AntigravityProvider(
            AntigravitySettings(
                executable=sys.executable,
                status_args=["--version"],
                quota_args=[],
                run_args=["-c", "print('disabled')"],
                execution_enabled=False,
            )
        )

        result = provider.run(_request())

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, ProviderErrorType.EXTERNAL_REQUIRED)
        self.assertEqual(result.provider, "antigravity")

    def test_cli_provider_does_not_masquerade_as_other_provider(self) -> None:
        provider = CodexProvider(
            CodexSettings(
                executable=sys.executable,
                status_args=["--version"],
                quota_args=[],
                run_args=["-c", "import sys; print(sys.stdin.read())"],
            )
        )

        result = provider.run(_request(prompt="hello cli"))

        self.assertTrue(result.success)
        self.assertEqual(result.provider, "codex")
        self.assertFalse(result.data["masqueradeAsProvider"])
        self.assertIn("hello cli", result.content)

    def test_cli_status_failure_is_not_ready(self) -> None:
        provider = CodexProvider(
            CodexSettings(
                executable=sys.executable,
                status_args=["-c", "import sys; sys.exit(7)"],
                quota_args=[],
            )
        )

        diagnostics = provider.diagnostics()

        self.assertFalse(diagnostics["ready"])
        self.assertEqual(diagnostics["errorType"], ProviderErrorType.EXTERNAL_REQUIRED)

    def test_cli_command_decodes_utf8_output_on_windows(self) -> None:
        result = run_cli_command(
            CliProviderSettings(executable=sys.executable),
            ["-c", "print('測試輸出')"],
        )

        self.assertEqual(result.return_code, 0)
        self.assertIn("測試輸出", result.stdout)

    def test_cli_run_uses_provider_run_timeout_when_request_timeout_is_unspecified(self) -> None:
        provider = CodexProvider(
            CodexSettings(
                executable=sys.executable,
                status_args=["--version"],
                quota_args=[],
                run_args=["-c", "import time; time.sleep(0.2); print('late')"],
                timeout_seconds=5,
                run_timeout_seconds=0.01,
            )
        )

        result = provider.run(_request())

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, ProviderErrorType.TIMEOUT)
        self.assertIn("timeout", str(result.data["command"]["error"]))

    def test_cli_timeout_stderr_is_classified_as_timeout(self) -> None:
        provider = AntigravityProvider(
            AntigravitySettings(
                executable=sys.executable,
                status_args=["--version"],
                quota_args=[],
                run_args=["-c", "import sys; sys.stderr.write('Error: timeout waiting for response'); sys.exit(1)"],
                execution_enabled=True,
            )
        )

        result = provider.run(_request())

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, ProviderErrorType.TIMEOUT)

    def test_cli_success_output_containing_timeout_is_not_failure(self) -> None:
        provider = CodexProvider(
            CodexSettings(
                executable=sys.executable,
                status_args=["--version"],
                quota_args=[],
                run_args=["-c", "print('No timeout occurred')"],
                execution_enabled=True,
            )
        )

        result = provider.run(_request())

        self.assertTrue(result.success)
        self.assertIsNone(result.error_type)

    def test_antigravity_compact_prompt_enforces_length_limit(self) -> None:
        prompt = _compact_prompt(
            "Project: sample\nWorkflow: project-analysis\nStages: inspect, report\n" + "x" * 500,
            240,
            challenge="challenge-1",
        )

        self.assertLessEqual(len(prompt), 240)
        self.assertIn("challenge-1", prompt)

    def test_antigravity_return_code_zero_with_plain_text_is_invalid(self) -> None:
        provider = AntigravityProvider(
            AntigravitySettings(
                executable=sys.executable,
                status_args=["--version"],
                quota_args=[],
                run_args=["-c", "print('None')"],
                execution_enabled=True,
            )
        )

        result = provider.run(_request())

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, ProviderErrorType.INVALID_RESPONSE)
        self.assertFalse(result.data["responseValidated"])
        self.assertFalse(result.data["antigravityNativePass"])

    def test_antigravity_valid_challenge_json_is_native_pass(self) -> None:
        script = (
            "import json,re,sys; p=sys.argv[-1]; c=re.search(r'Challenge=([0-9a-f]+)',p).group(1); "
            "print(json.dumps({'schema':'ai-team-antigravity/v1','challenge':c,'status':'ok',"
            "'findings':[],'tests':[],'blockers':[]}))"
        )
        provider = AntigravityProvider(
            AntigravitySettings(
                executable=sys.executable,
                status_args=["--version"],
                quota_args=[],
                run_args=["-c", script],
                execution_enabled=True,
            )
        )

        result = provider.run(_request())

        self.assertTrue(result.success, result.content)
        self.assertTrue(result.data["responseValidated"])
        self.assertTrue(result.data["antigravityNativePass"])

    def test_antigravity_repository_smoke_validates_probe_hash(self) -> None:
        script = (
            "import hashlib,json,pathlib,re,sys; p=sys.argv[-1]; c=re.search(r'Challenge=([0-9a-f]+)',p).group(1); "
            "f=re.search(r\"tracked file '([^']+)'\",p).group(1); root=pathlib.Path(sys.argv[sys.argv.index('--add-dir')+1]); "
            "h=hashlib.sha256((root/f).read_bytes()).hexdigest(); print(json.dumps({"
            "'schema':'ai-team-repository-smoke/v1','challenge':c,'probe':{'path':f,'sha256':h},"
            "'summary':'visible','findings':[],'tests':[],'blockers':[]}))"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "package.json").write_text('{"name":"sample"}\n', encoding="utf-8")
            subprocess.run(["git", "add", "package.json"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True)
            provider = AntigravityProvider(
                AntigravitySettings(
                    executable=sys.executable,
                    status_args=["--version"],
                    quota_args=[],
                    run_args=["-c", script],
                    execution_enabled=True,
                )
            )

            result = provider.run(
                ProviderRequest(
                    workflow="provider-smoke",
                    prompt="Project: sample\nWorkflow: provider-smoke",
                    project_root=root,
                )
            )

        self.assertTrue(result.success, result.content)
        self.assertTrue(result.data["repositorySmokePassed"])
        self.assertTrue(result.data["antigravityNativePass"])

    def test_antigravity_ready_then_run_reuses_successful_diagnostics(self) -> None:
        diagnostics = {"provider": "antigravity", "ready": True, "quotaExhausted": False}
        provider = AntigravityProvider(
            AntigravitySettings(
                executable=sys.executable,
                run_args=["-c", "print('None')"],
                execution_enabled=True,
            )
        )
        with patch("ai_team.providers.antigravity.build_diagnostics", return_value=diagnostics) as mocked:
            self.assertTrue(provider.ready())
            provider.run(_request())

        self.assertEqual(mocked.call_count, 1)

    def test_antigravity_deadline_exhausted_by_diagnostics_skips_run(self) -> None:
        clock = [0.0]

        def monotonic() -> float:
            return clock[0]

        def diagnostics(*args, **kwargs):
            clock[0] = 2.0
            return {"provider": "antigravity", "ready": True, "quotaExhausted": False}

        provider = AntigravityProvider(
            AntigravitySettings(executable=sys.executable, run_args=["-c", "raise SystemExit(99)"], execution_enabled=True),
            monotonic=monotonic,
        )
        with patch("ai_team.providers.antigravity.build_diagnostics", side_effect=diagnostics):
            result = provider.run(
                ProviderRequest(
                    workflow="project-analysis",
                    prompt="Project: sample",
                    project_root=Path.cwd(),
                    timeout_seconds=1,
                )
            )

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, ProviderErrorType.TIMEOUT)
        self.assertNotIn("command", result.data)

    def test_write_smoke_rejects_primary_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)

            result = WriteSmokeProvider().run(
                ProviderRequest(
                    workflow="bug-fix-loop",
                    prompt="smoke",
                    project_root=root,
                    dry_run=False,
                )
            )

            self.assertFalse(result.success)
            self.assertFalse(result.data["writePerformed"])
            self.assertIn("disposable", result.content)

    def test_write_smoke_rejects_forged_git_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_git_dir = root / "fake-git-dir"
            fake_git_dir.mkdir()
            (root / ".git").write_text(f"gitdir: {fake_git_dir.as_posix()}\n", encoding="utf-8")

            result = WriteSmokeProvider().run(
                ProviderRequest(
                    workflow="bug-fix-loop",
                    prompt="smoke",
                    project_root=root,
                    dry_run=False,
                )
            )

            self.assertFalse(result.success)
            self.assertFalse((root / "docs/ai-team-smoke/isolated-write-smoke.md").exists())

    def test_write_smoke_rejects_dry_run_and_run_agent(self) -> None:
        provider = WriteSmokeProvider()
        root = Path.cwd()

        dry_run = provider.run(
            ProviderRequest(
                workflow="bug-fix-loop",
                prompt="smoke",
                project_root=root,
                dry_run=True,
            )
        )
        run_agent = provider.run(
            ProviderRequest(
                workflow="bug-fix-loop",
                prompt="smoke",
                project_root=root,
                dry_run=False,
                run_mode="run-agent",
            )
        )

        self.assertFalse(dry_run.success)
        self.assertFalse(run_agent.success)
        self.assertIn("create-only", run_agent.content)


def _request(prompt: str = "hello") -> ProviderRequest:
    return ProviderRequest(workflow="project-analysis", prompt=prompt, project_root=Path.cwd())


if __name__ == "__main__":
    unittest.main()
