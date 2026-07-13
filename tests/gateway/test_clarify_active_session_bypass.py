"""Regression tests for clarify replies while a gateway session is busy."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource, build_session_key


class _ClarifyBypassAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="text")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "private"}


def _event(
    text="custom answer",
    *,
    message_type=MessageType.TEXT,
    media_urls=None,
    media_types=None,
):
    return MessageEvent(
        text=text,
        message_type=message_type,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="private",
            user_id="user1",
        ),
        message_id="msg1",
        media_urls=list(media_urls or []),
        media_types=list(media_types or []),
    )


def _clear_clarify_state():
    from tools import clarify_gateway as cm

    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


@pytest.fixture(autouse=True)
def _reset_clarify_state():
    _clear_clarify_state()
    yield
    _clear_clarify_state()


def _make_runner(transcription_result):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(
        multiplex_profiles=False,
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
        stt_enabled=True,
    )
    runner.session_store = None
    runner._is_user_authorized = lambda _source: True
    runner._enrich_message_with_transcription = AsyncMock(
        return_value=transcription_result,
    )
    return runner


def _session_key(event):
    return build_session_key(
        event.source,
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )


@pytest.mark.asyncio
async def test_active_session_routes_typed_choice_clarify_reply_to_runner_not_busy_queue():
    """Typed text must resolve a pending choice clarify even while the agent is busy.

    Telegram button clarifies keep the adapter session active while the agent
    thread blocks on ``wait_for_response``.  If the adapter only bypasses for
    entries already marked ``awaiting_text``, typed replies to the visible
    multi-choice prompt are handled as busy follow-ups and the clarify wait is
    never resolved.
    """
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    adapter = _ClarifyBypassAdapter()
    adapter._message_handler = AsyncMock(return_value="")
    adapter._busy_session_handler = AsyncMock(return_value=True)
    event = _event("None of those are valid options")
    session_key = build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get(
            "group_sessions_per_user", True
        ),
        thread_sessions_per_user=adapter.config.extra.get(
            "thread_sessions_per_user", False
        ),
    )
    adapter._active_sessions[session_key] = asyncio.Event()
    cm.register("clarify-1", session_key, "Pick one", ["A", "B"])

    await adapter.handle_message(event)

    adapter._message_handler.assert_awaited_once_with(event)
    adapter._busy_session_handler.assert_not_awaited()
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_pending_clarify_resolves_voice_reply_from_transcript(monkeypatch):
    from tools import clarify_gateway as cm

    event = _event(
        "voice_message_123.ogg",
        message_type=MessageType.VOICE,
        media_urls=["/tmp/voice_message_123.ogg"],
        media_types=["audio/ogg"],
    )
    session_key = _session_key(event)
    entry = cm.register("clarify-voice", session_key, "What should I do?", None)
    runner = _make_runner(('"use the local model"', ["use the local model"]))
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda *_args, **_kwargs: [],
    )

    result = await runner._handle_message(event)

    assert result == ""
    assert entry.response == "use the local model"
    runner._enrich_message_with_transcription.assert_awaited_once_with(
        "",
        ["/tmp/voice_message_123.ogg"],
    )


@pytest.mark.asyncio
async def test_pending_clarify_keeps_waiting_when_voice_transcription_fails(
    monkeypatch,
):
    from tools import clarify_gateway as cm

    event = _event(
        "voice_message_456.ogg",
        message_type=MessageType.VOICE,
        media_urls=["/tmp/voice_message_456.ogg"],
        media_types=["audio/ogg"],
    )
    session_key = _session_key(event)
    entry = cm.register("clarify-voice-failed", session_key, "What should I do?", None)
    runner = _make_runner(("[voice message could not be transcribed]", []))
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda *_args, **_kwargs: [],
    )

    result = await runner._handle_message(event)

    assert "transcribe" in result.lower()
    assert "text" in result.lower()
    assert entry.response is None
    assert cm.get_pending_for_session(session_key) is entry
