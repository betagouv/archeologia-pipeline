from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from threading import Event


def create_cancellable_feedback(cancel_check: Callable[[], bool]) -> Optional["QgsProcessingFeedback"]:
    """
    Crée un QgsProcessingFeedback qui vérifie périodiquement si l'annulation a été demandée.
    
    Args:
        cancel_check: Fonction qui retourne True si l'annulation a été demandée.
        
    Returns:
        Un QgsProcessingFeedback configuré pour l'annulation, ou None si QGIS n'est pas disponible.
    """
    try:
        from qgis.core import QgsProcessingFeedback
    except ImportError:
        return None

    class CancellableFeedback(QgsProcessingFeedback):
        def __init__(self, check_cancel: Callable[[], bool]):
            super().__init__()
            self._check_cancel = check_cancel

        def isCanceled(self) -> bool:
            if self._check_cancel():
                self.cancel()
                return True
            return super().isCanceled()

        def setProgress(self, progress: float) -> None:
            if self._check_cancel():
                self.cancel()
            super().setProgress(progress)

    return CancellableFeedback(cancel_check)
