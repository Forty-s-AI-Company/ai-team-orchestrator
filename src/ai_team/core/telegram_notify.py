"""Fail-safe Telegram Bot notifications for unattended AI Team alerts."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


BOT_TOKEN_ENV = "AI_TEAM_TELEGRAM_BOT_TOKEN"
CHAT_ID_ENV = "AI_TEAM_TELEGRAM_CHAT_ID"
THREAD_ID_ENV = "AI_TEAM_TELEGRAM_THREAD_ID"
TELEGRAM_TEXT_LIMIT = 4096
TELEGRAM_RESPONSE_LIMIT = 65_536
TELEGRAM_TIMEOUT_SECONDS = 15

UrlOpener = Callable[..., Any]


@dataclass(frozen=True)
class TelegramSettings:
    """Secrets stay out of reports and the dataclass representation."""

    bot_token: str = field(default="", repr=False)
    chat_id: str = ""
    thread_id: int | None = None

    @property
    def configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)


def load_telegram_settings(
    environment: Mapping[str, str] | None = None,
) -> TelegramSettings:
    values = os.environ if environment is None else environment
    token = values.get(BOT_TOKEN_ENV, "").strip()
    chat_id = values.get(CHAT_ID_ENV, "").strip()
    thread_value = values.get(THREAD_ID_ENV, "").strip()
    thread_id: int | None = None
    if thread_value:
        try:
            candidate = int(thread_value)
        except ValueError:
            candidate = 0
        if candidate > 0:
            thread_id = candidate
    return TelegramSettings(bot_token=token, chat_id=chat_id, thread_id=thread_id)


def send_telegram_message(
    title: str,
    message: str,
    *,
    settings: TelegramSettings | None = None,
    opener: UrlOpener = urllib.request.urlopen,
    timeout_seconds: int = TELEGRAM_TIMEOUT_SECONDS,
) -> bool:
    """Send one plain-text alert without exposing the token in logs or argv."""

    configured = settings or load_telegram_settings()
    if not configured.configured or not _valid_settings(configured):
        return False
    text = _notification_text(title, message)
    payload: dict[str, Any] = {
        "chat_id": configured.chat_id,
        "text": text,
    }
    if configured.thread_id is not None:
        payload["message_thread_id"] = configured.thread_id
    request = urllib.request.Request(
        (
            "https://api.telegram.org/bot"
            f"{configured.bot_token}/sendMessage"
        ),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener(request, timeout=timeout_seconds) as response:
            content = response.read(TELEGRAM_RESPONSE_LIMIT + 1)
            if len(content) > TELEGRAM_RESPONSE_LIMIT:
                return False
            if getattr(response, "status", 200) != 200:
                return False
    except (OSError, TimeoutError, ValueError, urllib.error.URLError):
        return False
    try:
        result = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(result, dict) and result.get("ok") is True


def _valid_settings(settings: TelegramSettings) -> bool:
    token_valid = re.fullmatch(
        r"[0-9]{5,20}:[A-Za-z0-9_-]{20,100}",
        settings.bot_token,
    )
    chat_valid = re.fullmatch(
        r"(?:-?[0-9]{1,20}|@[A-Za-z0-9_]{5,32})",
        settings.chat_id,
    )
    return bool(token_valid and chat_valid)


def _notification_text(title: str, message: str) -> str:
    safe_title = title.strip().replace("\x00", "") or "AI Team 狀態提醒"
    safe_message = message.strip().replace("\x00", "") or "請查看 Watchdog 狀態。"
    text = f"🤖 CelebrateDeal AI Team\n\n{safe_title}\n{safe_message}"
    return text[:TELEGRAM_TEXT_LIMIT]
