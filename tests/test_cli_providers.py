from __future__ import annotations

import json
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
    ProviderResult,
    WriteSmokeProvider,
)
from ai_team.providers.cli_common import CliProviderSettings, run_cli_command
from ai_team.providers.antigravity import _compact_prompt, _read_only_sandbox_settings
from ai_team.providers.antigravity import _apply_routing_options as apply_antigravity_routing
from ai_team.providers.codex import _apply_routing_options as apply_codex_routing
from ai_team.providers.codex import _extract_token_usage


class CliProviderTests(unittest.TestCase):
    def test_codex_routing_adds_allowlisted_model_and_reasoning(self) -> None:
        settings = CodexSettings(allowed_models=("gpt-5.6-terra",))

        args = apply_codex_routing(["exec", "--sandbox", "read-only"], "gpt-5.6-terra", "high", settings)

        self.assertEqual(args[-4:], ["--model", "gpt-5.6-terra", "--config", 'model_reasoning_effort="high"'])

    def test_codex_routing_rejects_unknown_model_and_reasoning(self) -> None:
        settings = CodexSettings(allowed_models=("gpt-5.6-terra",))

        with self.assertRaises(ValueError):
            apply_codex_routing([], "lookalike-model", "high", settings)
        with self.assertRaises(ValueError):
            apply_codex_routing([], "gpt-5.6-terra", "unbounded", settings)

    def test_codex_token_usage_is_parsed_from_native_stderr(self) -> None:
        result = ProviderResult(
            provider="codex",
            success=True,
            data={"command": {"stderr": "model: gpt-5.6-terra\ntokens used\n7,044\n"}},
        )

        self.assertEqual(_extract_token_usage(result), 7044)

    def test_codex_success_content_excludes_native_stderr_diagnostics(self) -> None:
        provider = CodexProvider(
            CodexSettings(
                executable=sys.executable,
                status_args=["--version"],
                quota_args=[],
                run_args=[
                    "-c",
                    "import sys; print('{\"schema\":\"example/v1\"}'); "
                    "sys.stderr.write('native progress\\ntokens used\\n42\\n')",
                ],
            )
        )

        result = provider.run(_request())

        self.assertTrue(result.success)
        self.assertEqual(result.content, '{"schema":"example/v1"}')
        self.assertIn("native progress", result.data["command"]["stderr"])
        self.assertEqual(result.data["tokenUsage"], 42)

    def test_codex_structured_content_is_not_limited_by_command_evidence(self) -> None:
        provider = CodexProvider(
            CodexSettings(
                executable=sys.executable,
                status_args=["--version"],
                quota_args=[],
                run_args=[
                    "-c",
                    "import json; print(json.dumps({'schema':'example/v1','payload':'x'*5000}))",
                ],
            )
        )

        result = provider.run(_request())

        self.assertTrue(result.success)
        self.assertGreater(len(result.content), 5000)
        self.assertEqual(json.loads(result.content)["schema"], "example/v1")
        self.assertEqual(len(result.data["command"]["stdout"]), 4000)

    def test_antigravity_routing_replaces_only_allowlisted_model(self) -> None:
        settings = AntigravitySettings(allowed_models=("Gemini 3.1 Pro (High)",))

        args = apply_antigravity_routing(
            ["--model", "Gemini 3.5 Flash (Low)", "--print"],
            "Gemini 3.1 Pro (High)",
            "high",
            settings,
        )

        self.assertEqual(args, ["--model", "Gemini 3.1 Pro (High)", "--print"])

    def test_antigravity_routing_rejects_mismatched_reasoning_label(self) -> None:
        settings = AntigravitySettings(allowed_models=("Gemini 3.5 Flash (Low)",))

        with self.assertRaises(ValueError):
            apply_antigravity_routing([], "Gemini 3.5 Flash (Low)", "high", settings)

    def test_antigravity_bounded_delivery_prompt_has_stage_and_no_write_capability(self) -> None:
        prompt = _compact_prompt(
            "Task: Update a documented behavior\nInstruction: Edit the approved path only",
            1200,
            challenge="challenge-1",
            bounded_stage="qa",
        )

        self.assertIn("schema='ai-team-bounded-delivery/v1'", prompt)
        self.assertIn("stage=qa", prompt)
        self.assertIn("Forbidden: edit, shell", prompt)
        self.assertIn("Challenge=challenge-1", prompt)
        self.assertIn("tests=['evidence citation']", prompt)

    def test_antigravity_bounded_review_uses_delivery_schema(self) -> None:
        prompt = _compact_prompt(
            "Task: Review a bounded diff\nInstruction: Inspect only",
            1200,
            challenge="challenge-review",
            bounded_stage="review",
        )

        self.assertIn("schema='ai-team-bounded-delivery/v1'", prompt)
        self.assertIn("stage=review", prompt)
        self.assertIn("Verify every AcceptanceCriteria item", prompt)
        self.assertIn("tests=['evidence citation']", prompt)

    def test_antigravity_pm_prompt_forbids_restatement_findings(self) -> None:
        prompt = _compact_prompt(
            "Task: Define acceptance criteria\nInstruction: Analyze only",
            1200,
            challenge="challenge-pm",
            bounded_stage="pm",
        )

        self.assertIn("findings and blockers MUST be exactly []", prompt)
        self.assertIn("never restate required work as a finding or blocker", prompt)

    def test_antigravity_bounded_stage_requires_read_only_filesystem_sandbox(self) -> None:
        provider = AntigravityProvider(
            AntigravitySettings(
                executable=sys.executable,
                status_args=["--version"],
                quota_args=[],
                run_args=["-c", "print('must-not-run')"],
                execution_enabled=True,
            )
        )

        result = provider.run(
            ProviderRequest(
                workflow="bounded-delivery-qa",
                prompt="Task: inspect only",
                project_root=Path.cwd(),
                run_mode="run-agent",
                metadata={"boundedStage": "qa", "writeAccess": False},
            )
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, ProviderErrorType.INVALID_RESPONSE)
        self.assertIn("read-only filesystem sandbox", result.content)

    def test_antigravity_read_only_sandbox_mounts_root_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = Path(tmp) / "bwrap"
            sandbox.write_text("test", encoding="utf-8")
            sandbox.chmod(0o700)
            settings = CliProviderSettings(
                executable="/opt/agy",
                run_args=["--mode", "plan", "--sandbox", "--print"],
            )

            wrapped = _read_only_sandbox_settings(settings, str(sandbox), Path("/workspace"))

        self.assertIsNotNone(wrapped)
        assert wrapped is not None
        self.assertEqual(wrapped.executable, str(sandbox))
        self.assertEqual(
            wrapped.run_args,
            [
                "--die-with-parent",
                "--new-session",
                "--ro-bind",
                "/",
                "/",
                "--dev-bind",
                "/dev",
                "/dev",
                "--proc",
                "/proc",
                "--tmpfs",
                "/tmp",
                "--chdir",
                "/workspace",
                "/opt/agy",
                "--mode",
                "plan",
                "--sandbox",
                "--print",
            ],
        )

    def test_antigravity_bounded_qa_prompt_keeps_evidence_json_valid(self) -> None:
        evidence = json.dumps(
            {
                "acceptanceCriteria": [
                    "The safe path is updated",
                    "Lint succeeds",
                    "Tests succeed",
                    "Build succeeds",
                ],
                "allowedWritePaths": [f"src/very-long-path-{index}.tsx" for index in range(20)],
                "validationCommands": ["npm run lint", "npm run typecheck", "npm run test", "npm run build"],
                "changedFiles": ["src/component.tsx", "src/component.test.ts"],
                "commitSha": "a" * 40,
                "validation": {"success": True, "commands": {"npm run test": {"output": "x" * 2000}}},
                "repairs": [],
            }
        )
        prompt = _compact_prompt(
            "\n".join(
                (
                    "Task: Update a documented behavior",
                    "Instruction: Edit the approved path only",
                    'Acceptance criteria: ["The safe path is updated", "Lint succeeds", "Tests succeed", "Build succeeds"]',
                    f"Allowed write paths: {json.dumps([f'src/very-long-path-{index}.tsx' for index in range(20)])}",
                    'Validation commands: ["npm run lint", "npm run typecheck", "npm run test", "npm run build"]',
                    f"Implementation evidence: {evidence}",
                )
            ),
            1200,
            challenge="challenge-1",
            bounded_stage="qa",
        )

        acceptance = prompt.split("AcceptanceCriteria=", 1)[1].split("; AllowedWritePaths=", 1)[0]
        allowed = prompt.split("AllowedWritePaths=", 1)[1].split("; ValidationCommands=", 1)[0]
        commands = prompt.split("ValidationCommands=", 1)[1].split("; ImplementationEvidence=", 1)[0]
        compact_evidence = prompt.split("ImplementationEvidence=", 1)[1].removesuffix(".")

        self.assertEqual(
            json.loads(acceptance),
            ["The safe path is updated", "Lint succeeds", "Tests succeed", "Build succeeds"],
        )
        self.assertIsInstance(json.loads(allowed), list)
        self.assertEqual(json.loads(commands), ["npm run lint", "npm run typecheck", "npm run test", "npm run build"])
        self.assertEqual(
            json.loads(compact_evidence),
            {
                "changedFileCount": 2,
                "commitSha": "a" * 12,
                "validationSuccess": True,
                "repairCount": 0,
            },
        )
        self.assertNotIn("[truncated]", prompt)

    def test_codex_trusted_write_requires_linked_worktree_marker(self) -> None:
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

    def test_windows_sandbox_helper_failure_is_not_reported_as_success(self) -> None:
        provider = CodexProvider(
            CodexSettings(
                executable=sys.executable,
                status_args=["--version"],
                quota_args=[],
                run_args=["-c", "import sys; sys.stderr.write('windows sandbox failed: orchestrator_helper_incomplete')"],
            )
        )

        result = provider.run(_request())

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, ProviderErrorType.NETWORK)

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

    def test_codex_provider_smoke_uses_sterile_workspace_and_validates_challenge(self) -> None:
        script = (
            "import json,os,re,sys; "
            "p=sys.stdin.read(); c=re.search(r\"challenge='([0-9a-f]+)'\",p).group(1); "
            "print(json.dumps({'schema':'ai-team-codex-smoke/v1','challenge':c,"
            "'provider':'codex','status':'ok','saw_env':os.path.exists('.env')}))"
        )
        provider = CodexProvider(
            CodexSettings(
                executable=sys.executable,
                status_args=["--version"],
                quota_args=[],
                run_args=["-c", script],
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("TEST_SECRET=do-not-read", encoding="utf-8")
            result = provider.run(_request(root=root, workflow="provider-smoke"))

        self.assertTrue(result.success)
        self.assertTrue(result.data["providerNative"])
        self.assertTrue(result.data["codexNativePass"])
        self.assertTrue(result.data["responseValidated"])
        self.assertFalse(json.loads(result.content)["saw_env"])
        self.assertNotIn("do-not-read", str(result.data))

    def test_codex_provider_smoke_fails_closed_on_unvalidated_output(self) -> None:
        provider = CodexProvider(
            CodexSettings(
                executable=sys.executable,
                status_args=["--version"],
                quota_args=[],
                run_args=["-c", "print('not-json')"],
            )
        )

        result = provider.run(_request(workflow="provider-smoke"))

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, ProviderErrorType.INVALID_RESPONSE)
        self.assertFalse(result.data["codexNativePass"])

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

    def test_antigravity_accepts_one_valid_json_inside_text_envelope(self) -> None:
        script = (
            "import json,re,sys; p=sys.argv[-1]; c=re.search(r'Challenge=([0-9a-f]+)',p).group(1); "
            "v={'schema':'ai-team-antigravity/v1','challenge':c,'status':'ok',"
            "'findings':[],'tests':[{'id':'qa'}],'blockers':[]}; "
            "print('Evidence analysis follows.'); print('```json'); print(json.dumps(v)); print('```')"
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
        self.assertEqual(json.loads(result.content)["tests"], [{"id": "qa"}])
        self.assertNotIn("Evidence analysis", result.content)

    def test_antigravity_rejects_ambiguous_valid_json_envelope(self) -> None:
        script = (
            "import json,re,sys; p=sys.argv[-1]; c=re.search(r'Challenge=([0-9a-f]+)',p).group(1); "
            "v={'schema':'ai-team-antigravity/v1','challenge':c,'status':'ok',"
            "'findings':[],'tests':[],'blockers':[]}; print(json.dumps(v)); print(json.dumps(v))"
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

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, ProviderErrorType.INVALID_RESPONSE)
        self.assertFalse(result.data["responseValidated"])

    def test_antigravity_repository_smoke_validates_probe_hash(self) -> None:
        script = (
            "import hashlib,json,pathlib,re,sys; p=sys.argv[-1]; c=re.search(r'Challenge=([0-9a-f]+)',p).group(1); "
            "f=re.search(r\"tracked file '([^']+)'\",p).group(1); root=pathlib.Path(sys.argv[sys.argv.index('--add-dir')+1]); "
            "h=hashlib.sha256((root/f).read_bytes()).hexdigest(); print(json.dumps({"
            "'schema':'ai-team-repository-smoke/v1','challenge':c,'probe':{'path':f,'sha256':h},"
            "'summary':'visible','findings':[],'tests':[],'blockers':[],"
            "'saw_env':(root/'.env').exists()}))"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "test@example.local"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            (root / "package.json").write_text('{"name":"sample"}\n', encoding="utf-8")
            (root / ".env").write_text("TEST_SECRET=do-not-read\n", encoding="utf-8")
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
        self.assertFalse(json.loads(result.content)["saw_env"])
        self.assertNotIn("do-not-read", str(result.data))

    def test_antigravity_repository_smoke_fails_closed_without_safe_probe(self) -> None:
        provider = AntigravityProvider(
            AntigravitySettings(
                executable=sys.executable,
                status_args=["--version"],
                quota_args=[],
                run_args=["-c", "print('must-not-run')"],
                execution_enabled=True,
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            result = provider.run(_request(root=root, workflow="provider-smoke"))

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, ProviderErrorType.INVALID_RESPONSE)
        self.assertFalse(result.data["antigravityNativePass"])
        self.assertNotIn("must-not-run", result.content)

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


def _request(
    prompt: str = "hello",
    *,
    root: Path | None = None,
    workflow: str = "project-analysis",
) -> ProviderRequest:
    return ProviderRequest(workflow=workflow, prompt=prompt, project_root=root or Path.cwd())


if __name__ == "__main__":
    unittest.main()
