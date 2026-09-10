from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol

from ..cancel_token import CancelToken
from ..progress_reporter import ProgressReporter
from ..run_context import RunContext

if TYPE_CHECKING:
    from ..structured_logger import StructuredLogger


class ModeRunner(Protocol):
    # Renvoie le verdict de la finalisation : True = succès, False = terminé en
    # erreur (ex. 0/N dalle produite), None = annulation ou runner legacy sans
    # verdict. Propagé par PipelineController jusqu'au bandeau de fin de l'UI.
    def run(
        self,
        ctx: RunContext,
        reporter: ProgressReporter,
        cancel: CancelToken,
        slog: Optional["StructuredLogger"] = None,
    ) -> Optional[bool]: ...
