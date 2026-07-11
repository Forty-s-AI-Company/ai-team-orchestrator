from __future__ import annotations

import os
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ai_team.providers import MockProvider, OpenHandsProvider, OpenHandsSettings, ProviderErrorType
from ai_team.providers.base import ProviderRequest, RetryingProvider, redact_secrets


class ProviderTests(unittest.TestCase):
    def test_openhands_ready_success_fixture(self) -> None:
        server = _FakeOpenHandsServer()
        try:
            provider = OpenHandsProvider(
                OpenHandsSettings(base_url=server.base_url),
                session_key="test-session",
            )
            diagnostics = provider.diagnostics()
            self.assertTrue(diagnostics["ready"])
            self.assertEqual(diagnostics["status"], 200)
        finally:
            server.close()

    def test_openhands_conversation_created_fixture(self) -> None:
        server = _FakeOpenHandsServer()
        try:
            provider = OpenHandsProvider(
                OpenHandsSettings(
                    base_url=server.base_url,
                    host_workspace_root=str(Path.cwd().anchor or Path.cwd()),
                    container_workspace_root="/projects",
                ),
                session_key="test-session",
            )
            result = provider.run(
                ProviderRequest(
                    workflow="project-analysis",
                    prompt="hello",
                    project_root=Path.cwd(),
                )
            )
            self.assertTrue(result.success)
            self.assertEqual(result.conversation_id, "11111111-1111-4111-8111-111111111111")
            self.assertEqual(result.data["executionStatus"], "idle")
            self.assertEqual(server.last_post_path, "/api/conversations")
        finally:
            server.close()

    def test_openhands_fails_closed_without_session_key(self) -> None:
        old_value = os.environ.pop("SESSION_API_KEY", None)
        try:
            provider = OpenHandsProvider(OpenHandsSettings(base_url="http://127.0.0.1:31024"))
            result = provider.run(
                ProviderRequest(
                    workflow="project-analysis",
                    prompt="hello",
                    project_root=Path.cwd(),
                )
            )
            self.assertFalse(result.success)
            self.assertEqual(result.error_type, ProviderErrorType.AUTH)
        finally:
            if old_value is not None:
                os.environ["SESSION_API_KEY"] = old_value

    def test_provider_timeout_is_classified(self) -> None:
        provider = OpenHandsProvider(
            OpenHandsSettings(
                base_url="http://10.255.255.1:31024",
                timeout_seconds=0.001,
            ),
            session_key="test-session",
        )
        result = provider.run(
            ProviderRequest(
                workflow="project-analysis",
                prompt="hello",
                project_root=Path.cwd(),
                timeout_seconds=0.001,
            )
        )
        self.assertFalse(result.success)
        self.assertIn(result.error_type, {ProviderErrorType.TIMEOUT, ProviderErrorType.NETWORK})

    def test_openhands_unavailable_diagnostics(self) -> None:
        provider = OpenHandsProvider(
            OpenHandsSettings(base_url="http://127.0.0.1:9"),
            session_key="test-session",
        )
        diagnostics = provider.diagnostics()
        self.assertFalse(diagnostics["ready"])
        self.assertEqual(diagnostics["errorType"], ProviderErrorType.NETWORK)
        self.assertTrue(diagnostics["sessionKeyPresent"])

    def test_retry_exhaustion(self) -> None:
        provider = RetryingProvider(MockProvider(fail_times=5), max_retries=2, backoff_seconds=0)
        result = provider.run(
            ProviderRequest(
                workflow="project-analysis",
                prompt="hello",
                project_root=Path.cwd(),
            )
        )
        self.assertFalse(result.success)
        self.assertEqual(result.attempts, 3)

    def test_secret_redaction(self) -> None:
        sample = "SESSION_API_KEY" + "=supersecret " + "Bearer " + "abc.def.ghi " + "sk-" + "test123456789"
        redacted = redact_secrets(sample)
        self.assertNotIn("supersecret", redacted)
        self.assertNotIn("abc.def.ghi", redacted)
        self.assertNotIn("sk-" + "test123456789", redacted)

class _FakeOpenHandsHandler(BaseHTTPRequestHandler):
    server_version = "FakeOpenHands/1.0"

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/ready":
            self._send_json({"status": "ready"})
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or "0")
        raw_body = self.rfile.read(length).decode("utf-8")
        self.server.last_post_path = self.path  # type: ignore[attr-defined]
        self.server.last_post_body = json.loads(raw_body)  # type: ignore[attr-defined]
        if self.path != "/api/conversations":
            self.send_response(404)
            self.end_headers()
            return
        self._send_json(
            {
                "id": "11111111-1111-4111-8111-111111111111",
                "execution_status": "idle",
                "workspace": {"kind": "LocalWorkspace", "working_dir": "/projects/sample"},
            }
        )

    def _send_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _FakeOpenHandsServer:
    def __init__(self) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenHandsHandler)
        self.httpd.last_post_path = None  # type: ignore[attr-defined]
        self.httpd.last_post_body = None  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        host, port = self.httpd.server_address
        return f"http://{host}:{port}"

    @property
    def last_post_path(self) -> str | None:
        return self.httpd.last_post_path  # type: ignore[attr-defined]

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
