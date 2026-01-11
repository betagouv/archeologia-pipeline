from __future__ import annotations

import threading


class CancelToken:
    def __init__(self, event: threading.Event):
        self._event = event

    def is_cancelled(self) -> bool:
        return bool(self._event.is_set())
