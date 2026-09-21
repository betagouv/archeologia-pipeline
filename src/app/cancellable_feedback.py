from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from threading import Event  # noqa: F401 — annotations différées
    from qgis.core import QgsProcessingFeedback  # noqa: F401 — F821 latent (AUDIT HYG-01)


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
            self._deadline: Optional[float] = None
            self._expired = False

        # -- Chien de garde (incident 2026-09-19) ------------------------
        # Un LAZ corrompu fait boucler PDAL sans fin : 10 h de silence dans
        # pdal:exportrastertin sur une dalle, sans erreur ni progression.
        # L'appelant arme une limite AUTOUR d'un algorithme et abandonne la
        # dalle si elle est franchie.

        def start_watchdog(self, timeout_s: float) -> None:
            """Arme une limite de temps pour l'algorithme qui suit (0 = aucune)."""
            self._deadline = time.monotonic() + float(timeout_s) if timeout_s else None
            self._expired = False

        def stop_watchdog(self) -> bool:
            """Désarme et dit si la limite avait été franchie."""
            self._deadline = None
            expired, self._expired = self._expired, False
            return expired

        def _watchdog_bit(self) -> bool:
            if self._deadline is None or time.monotonic() < self._deadline:
                return False
            # Volontairement PAS self.cancel() : l'annulation Qt est collante
            # et emporterait tout le reste du run. Ici seul l'algorithme en
            # cours rend la main ; stop_watchdog() remet le feedback à neuf
            # pour la dalle suivante.
            self._expired = True
            return True

        def isCanceled(self) -> bool:
            if self._check_cancel():
                self.cancel()
                return True
            if self._watchdog_bit():
                return True
            return super().isCanceled()

        def setProgress(self, progress: float) -> None:
            if self._check_cancel():
                self.cancel()
            super().setProgress(progress)

    return CancellableFeedback(cancel_check)
