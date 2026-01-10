from __future__ import annotations

from typing import Protocol

from ..cancel_token import CancelToken
from ..progress_reporter import ProgressReporter
from ..run_context import RunContext


class ModeRunner(Protocol):
    def run(self, ctx: RunContext, reporter: ProgressReporter, cancel: CancelToken) -> None: ...
