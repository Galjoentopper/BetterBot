"""Telegram helper utilities for BetterBot notifications."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import requests

LOGGER = logging.getLogger(__name__)


def _build_url(token: str, method: str = "sendMessage") -> str:
    return f"https://api.telegram.org/bot{token}/{method}"


def _post(token: str, payload: Dict[str, Any], method: str = "sendMessage") -> None:
    url = _build_url(token, method)
    response = requests.post(url, json=payload, timeout=5)
    if response.status_code >= 400:
        LOGGER.error("Telegram API error: %s", response.text)
        response.raise_for_status()


def send_alert(message: str, *, token: Optional[str] = None, chat_id: Optional[str] = None) -> None:
    """Send a single alert message to Telegram."""

    token = token or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        LOGGER.warning("Telegram credentials missing; alert not sent: %s", message)
        return
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        _post(token, payload)
        LOGGER.info("Sent Telegram alert: %s", message)
    except requests.RequestException as exc:  # pragma: no cover - network interaction
        LOGGER.exception("Failed to send Telegram alert: %s", exc)


def send_report(report: str, *, token: Optional[str] = None, chat_id: Optional[str] = None) -> None:
    """Send a detailed report message to Telegram (uses Markdown)."""

    send_alert(report, token=token, chat_id=chat_id)


__all__ = ["send_alert", "send_report"]
