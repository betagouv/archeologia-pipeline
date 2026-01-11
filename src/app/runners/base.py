from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol

from ..cancel_token import CancelToken
from ..progress_reporter import ProgressReporter
from ..run_context import RunContext

if TYPE_CHECKING:
    from ..structured_logger import StructuredLogger


class ModeRunner(Protocol):
    def run(
        self,
        ctx: RunContext,
        reporter: ProgressReporter,
        cancel: CancelToken,
        slog: Optional["StructuredLogger"] = None,
    ) -> None: ...
