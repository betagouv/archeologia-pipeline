from __future__ import annotations

import threading

import pytest

from app.cancel_token import CancelToken


class TestCancelToken:
    def test_initial_state_is_not_cancelled(self):
        event = threading.Event()
        token = CancelToken(event)
        assert token.is_cancelled() is False

    def test_is_cancelled_after_event_set(self):
        event = threading.Event()
        token = CancelToken(event)
        event.set()
        assert token.is_cancelled() is True

    def test_is_cancelled_after_event_clear(self):
        event = threading.Event()
        event.set()
        token = CancelToken(event)
        assert token.is_cancelled() is True
        event.clear()
        assert token.is_cancelled() is False

    def test_multiple_tokens_share_same_event(self):
        event = threading.Event()
        token1 = CancelToken(event)
        token2 = CancelToken(event)
        assert token1.is_cancelled() is False
        assert token2.is_cancelled() is False
        event.set()
        assert token1.is_cancelled() is True
        assert token2.is_cancelled() is True

    def test_works_with_fresh_event(self):
        token = CancelToken(threading.Event())
        assert token.is_cancelled() is False
