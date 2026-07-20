from __future__ import annotations

import json
import subprocess
import unittest
import urllib.error
from typing import Any

from ai_team.core.telegram_notify import (
    TelegramSettings,
    load_telegram_settings,
    send_telegram_message,
)
from ai_team.core.watchdog import send_watchdog_notifications


class _Response:
    def __init__(self, payload: dict[str, Any], *, status: int = 200) -> None:
        self._content = json.dumps(payload).encode("utf-8")
        self.status = status

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _limit: int) -> bytes:
        return self._content


class TelegramNotificationTests(unittest.TestCase):
    def test_loads_secrets_without_exposing_token_in_repr(self) -> None:
        settings = load_telegram_settings({
            "AI_TEAM_TELEGRAM_BOT_TOKEN": "123456789:abcdefghijklmnopqrstuvwxyz_ABCDE",
            "AI_TEAM_TELEGRAM_CHAT_ID": "-1001234567890",
            "AI_TEAM_TELEGRAM_THREAD_ID": "42",
        })

        self.assertTrue(settings.configured)
        self.assertEqual(settings.thread_id, 42)
        self.assertNotIn(settings.bot_token, repr(settings))

    def test_sends_plain_json_to_the_official_send_message_endpoint(self) -> None:
        requests: list[tuple[object, int]] = []
        settings = TelegramSettings(
            bot_token="123456789:abcdefghijklmnopqrstuvwxyz_ABCDE",
            chat_id="-1001234567890",
            thread_id=42,
        )

        def opener(request: object, *, timeout: int) -> _Response:
            requests.append((request, timeout))
            return _Response({"ok": True, "result": {"message_id": 1}})

        delivered = send_telegram_message(
            "AI Team 任務卡住",
            "同一問題已連續出現三次。",
            settings=settings,
            opener=opener,
        )

        self.assertTrue(delivered)
        request, timeout = requests[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertTrue(request.full_url.endswith("/sendMessage"))
        self.assertEqual(timeout, 15)
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["chat_id"], "-1001234567890")
        self.assertEqual(payload["message_thread_id"], 42)
        self.assertIn("AI Team 任務卡住", payload["text"])
        self.assertNotIn("parse_mode", payload)

    def test_missing_or_invalid_settings_never_attempt_network_io(self) -> None:
        calls: list[object] = []

        delivered = send_telegram_message(
            "title",
            "message",
            settings=TelegramSettings(bot_token="bad/token", chat_id="123"),
            opener=lambda request, **_kwargs: calls.append(request),
        )

        self.assertFalse(delivered)
        self.assertEqual(calls, [])

    def test_network_or_api_failures_are_non_fatal(self) -> None:
        settings = TelegramSettings(
            bot_token="123456789:abcdefghijklmnopqrstuvwxyz_ABCDE",
            chat_id="123456789",
        )

        def unavailable(_request: object, **_kwargs: object) -> _Response:
            raise urllib.error.URLError("offline")

        self.assertFalse(send_telegram_message(
            "title",
            "message",
            settings=settings,
            opener=unavailable,
        ))
        self.assertFalse(send_telegram_message(
            "title",
            "message",
            settings=settings,
            opener=lambda *_args, **_kwargs: _Response({"ok": False}),
        ))

    def test_dispatcher_attempts_telegram_even_when_windows_toast_fails(self) -> None:
        settings = TelegramSettings(
            bot_token="123456789:abcdefghijklmnopqrstuvwxyz_ABCDE",
            chat_id="123456789",
        )
        telegram_calls: list[tuple[str, str]] = []

        delivered = send_watchdog_notifications(
            "title",
            "message",
            telegram_settings=settings,
            telegram_sender=lambda title, message, **_kwargs: (
                telegram_calls.append((title, message)) or True
            ),
            runner=lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, "", "failed"),
        )

        self.assertTrue(delivered)
        self.assertEqual(telegram_calls, [("title", "message")])


if __name__ == "__main__":
    unittest.main()
