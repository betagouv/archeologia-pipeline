"""Toast — message éphémère affiché en bas-centre d'un widget parent.

Utilisé pour expliquer le verrou MNT (étape 2) au lieu d'un blocage silencieux.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import Qt, QTimer
from qgis.PyQt.QtWidgets import QLabel


class Toast(QLabel):
    def __init__(self, parent, message: str, duration: int = 2800):
        super().__init__(message, parent)
        self.setObjectName("Toast")
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMaximumWidth(380)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.adjustSize()
        self._reposition()
        self.show()
        self.raise_()
        QTimer.singleShot(duration, self.close)

    def _reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        x = (parent.width() - self.width()) // 2
        y = parent.height() - self.height() - 48
        self.move(max(8, x), max(8, y))


def show_toast(parent, message: str, duration: int = 2800) -> Toast:
    return Toast(parent, message, duration)
