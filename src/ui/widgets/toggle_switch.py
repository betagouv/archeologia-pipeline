"""ToggleSwitch — interrupteur « pilule » (style iOS) sans image.

Bouton checkable peint à la main (piste arrondie + pastille glissante). Émet le
signal ``toggled(bool)`` hérité de :class:`QAbstractButton`, donc interchangeable
avec une ``QCheckBox`` côté logique (``setChecked`` / ``isChecked`` / ``toggled``).
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QRectF, QSize, Qt
from qgis.PyQt.QtGui import QColor, QPainter
from qgis.PyQt.QtWidgets import QAbstractButton

_TRACK_W = 40
_TRACK_H = 22
_MARGIN = 2


class ToggleSwitch(QAbstractButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(_TRACK_W, _TRACK_H)

    def sizeHint(self) -> QSize:  # noqa: N802 (signature Qt)
        return QSize(_TRACK_W, _TRACK_H)

    def paintEvent(self, _event):  # noqa: N802 (signature Qt)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)

        # Piste : bleu si activé, gris sinon (légèrement plus terne si désactivé).
        on = self.isChecked()
        if not self.isEnabled():
            track = QColor("#cfcfcf")
        elif on:
            track = QColor("#2b79c2")
        else:
            track = QColor("#b8b8b8")
        radius = _TRACK_H / 2.0
        p.setBrush(track)
        p.drawRoundedRect(QRectF(0, 0, _TRACK_W, _TRACK_H), radius, radius)

        # Pastille blanche, calée à droite si activé, à gauche sinon.
        d = _TRACK_H - 2 * _MARGIN
        x = (_TRACK_W - _MARGIN - d) if on else _MARGIN
        p.setBrush(QColor("#ffffff"))
        p.drawEllipse(QRectF(x, _MARGIN, d, d))
