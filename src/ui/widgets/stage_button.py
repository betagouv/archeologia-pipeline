"""StageButton — un stade de la frise (étape 1).

Carte affichant une icône SVG (teintée selon l'état) + libellé + sous-libellé.
Le rôle visuel reflète l'état du stade vis-à-vis du point d'entrée :

- ``entry``    : stade d'entrée choisi (accent bleu) ;
- ``executed`` : stade exécuté (après l'entrée) ;
- ``skipped``  : stade sauté (avant l'entrée, atténué).

Le badge « ENTRÉE » n'est PAS dessiné ici : c'est un overlay flottant géré par
la page (pour pouvoir chevaucher la bordure supérieure, ce que Qt ne permet pas
à un enfant interne).
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QPointF, Qt, pyqtSignal
from qgis.PyQt.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from qgis.PyQt.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from ..icons import colored_pixmap

_ROLE_COLORS = {"entry": "#1d5a96", "executed": "#000000", "skipped": "#5a5a5a"}
_ICON_SIZE = 26


class ArrowConnector(QWidget):
    """Flèche horizontale peinte (ligne + pointe) entre deux stades de la frise.

    Remplace l'ancien caractère « ▶ » par une vraie flèche : bleue quand le
    segment est actif (à droite du point d'entrée), grise sinon.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active = False
        self.setFixedWidth(28)

    def set_active(self, on: bool) -> None:
        on = bool(on)
        if on != self._active:
            self._active = on
            self.update()

    def paintEvent(self, _event):  # noqa: N802 (signature Qt)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        color = QColor("#2b79c2" if self._active else "#a0a0a0")
        cy = self.height() / 2.0
        left, right = 5.0, self.width() - 5.0
        head = 6.0  # longueur de la pointe
        pen = QPen(color, 2.0)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawLine(QPointF(left, cy), QPointF(right - head, cy))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(color))
        p.drawPolygon(QPolygonF([
            QPointF(right, cy),
            QPointF(right - head, cy - head * 0.62),
            QPointF(right - head, cy + head * 0.62),
        ]))


class StageButton(QFrame):
    clicked = pyqtSignal()

    def __init__(self, icon_name: str, label: str, sub: str, *, clickable: bool = True,
                 optional: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("StageButton")
        self._icon_name = icon_name
        self._clickable = clickable
        self.setProperty("role", "executed")
        self.setProperty("optional", bool(optional))
        self.setCursor(Qt.PointingHandCursor if clickable else Qt.ArrowCursor)
        self.setMinimumHeight(78)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 8, 10)
        layout.setSpacing(3)
        layout.setAlignment(Qt.AlignTop)

        self._glyph = QLabel()
        self._glyph.setObjectName("StageGlyph")
        self._glyph.setAlignment(Qt.AlignCenter)
        self._glyph.setFixedHeight(_ICON_SIZE + 2)
        self._name = QLabel(label)
        self._name.setObjectName("StageName")
        self._name.setAlignment(Qt.AlignCenter)
        self._name.setWordWrap(True)
        self._sub = QLabel(sub)
        self._sub.setObjectName("StageSub")
        self._sub.setAlignment(Qt.AlignCenter)
        self._sub.setWordWrap(True)

        layout.addWidget(self._glyph)
        layout.addWidget(self._name)
        layout.addWidget(self._sub)
        layout.addStretch(1)

        self.set_role("executed")

    def set_role(self, role: str) -> None:
        """role ∈ {'entry', 'executed', 'skipped'}."""
        self.setProperty("role", role)
        self._glyph.setPixmap(
            colored_pixmap(self._icon_name, _ROLE_COLORS.get(role, "#000000"), _ICON_SIZE)
        )
        for w in (self, self._name, self._sub):
            try:
                w.style().unpolish(w)
                w.style().polish(w)
            except Exception:
                pass

    def mousePressEvent(self, event):  # noqa: N802 (signature Qt)
        if self._clickable:
            self.clicked.emit()
        super().mousePressEvent(event)
