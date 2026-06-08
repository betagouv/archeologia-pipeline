"""Carte (#Card) du wizard V2 — conteneur « fieldset » façon QGroupBox.

L'en-tête (pastille numérotée #CardNum + titre #CardTitle + compteur optionnel
#CardCounter) chevauche le bord supérieur d'une bordure arrondie : la ligne
haute est interrompue derrière l'en-tête puis se prolonge à droite. Helper
partagé par toutes les pages d'étapes → en-tête identique sur tous les onglets.
"""
from __future__ import annotations

from typing import Tuple

from qgis.PyQt.QtCore import QEvent, QRectF, Qt
from qgis.PyQt.QtGui import QColor, QPainter, QPainterPath, QPen
from qgis.PyQt.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

_BORDER = "#c4c4c4"
_RADIUS = 6.0
_LINE_Y = 9.0      # ordonnée de la bordure haute (centre vertical de la pastille)
_GAP_PAD = 6.0     # marge horizontale de la coupure autour de l'en-tête


class _CardFrame(QFrame):
    """QFrame dont la bordure arrondie est peinte à la main, avec une coupure
    sous l'en-tête (titre posé sur la ligne, comme une légende de fieldset)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._header: QWidget | None = None

    def set_header(self, header: QWidget) -> None:
        self._header = header
        # La coupure de bordure suit la largeur de l'en-tête (qui change quand le
        # compteur est mis à jour) → on repeint dès que l'en-tête bouge/redimensionne.
        header.installEventFilter(self)

    def eventFilter(self, obj, event):  # noqa: N802 (signature Qt)
        if obj is self._header and event.type() in (
            QEvent.Type.Resize, QEvent.Type.Move, QEvent.Type.Show,
        ):
            self.update()
        return super().eventFilter(obj, event)

    def paintEvent(self, _event):  # noqa: N802 (signature Qt)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(_BORDER), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)

        left, right = 0.5, self.width() - 0.5
        bottom = self.height() - 0.5

        # Sans en-tête : bordure pleine arrondie (ex. carte de la frise).
        if self._header is None:
            p.drawRoundedRect(QRectF(left, 0.5, right - left, bottom - 0.5), _RADIUS, _RADIUS)
            return

        y = _LINE_Y + 0.5
        gx0 = self._header.x() - _GAP_PAD
        gx1 = self._header.x() + self._header.width() + _GAP_PAD

        # Tracé : on part juste après la coupure, on fait le tour, on revient
        # jusqu'au début de la coupure → la ligne haute « saute » l'en-tête.
        path = QPainterPath()
        path.moveTo(gx1, y)
        path.lineTo(right - _RADIUS, y)
        path.quadTo(right, y, right, y + _RADIUS)
        path.lineTo(right, bottom - _RADIUS)
        path.quadTo(right, bottom, right - _RADIUS, bottom)
        path.lineTo(left + _RADIUS, bottom)
        path.quadTo(left, bottom, left, bottom - _RADIUS)
        path.lineTo(left, y + _RADIUS)
        path.quadTo(left, y, left + _RADIUS, y)
        path.lineTo(max(gx0, left + _RADIUS), y)
        p.drawPath(path)


def build_card(title: str = "", num: str = "") -> Tuple[QFrame, QVBoxLayout]:
    """Retourne ``(carte, layout_contenu)``. Ajouter le contenu au layout.

    Quand ``title`` est fourni, ``carte.counter`` expose un :class:`QLabel`
    (compteur, ex. « 4 sur 6 sélectionnées ») à mettre à jour par la page ;
    il vaut ``None`` pour une carte sans titre.
    """
    card = _CardFrame()
    card.setObjectName("Card")
    card.counter = None  # type: ignore[attr-defined]
    outer = QVBoxLayout(card)
    outer.setSpacing(10)

    if not title:
        outer.setContentsMargins(12, 12, 12, 12)
        return card, outer

    # En-tête posé sur la bordure haute : marge haute nulle pour qu'il chevauche.
    outer.setContentsMargins(12, 0, 12, 12)
    header = QWidget()
    header.setObjectName("CardHeader")
    hl = QHBoxLayout(header)
    hl.setContentsMargins(0, 0, 0, 0)
    hl.setSpacing(7)
    if num:
        badge = QLabel(num)
        badge.setObjectName("CardNum")
        badge.setFixedSize(18, 18)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hl.addWidget(badge)
    tlabel = QLabel(title)
    tlabel.setObjectName("CardTitle")
    hl.addWidget(tlabel)
    counter = QLabel("")
    counter.setObjectName("CardCounter")
    hl.addWidget(counter)

    # Aligné à gauche (pas de stretch interne) → la largeur de l'en-tête = celle
    # de son contenu, donc la coupure de bordure ne couvre que le titre.
    outer.addWidget(header, 0, Qt.AlignmentFlag.AlignLeft)
    card.set_header(header)
    card.counter = counter  # type: ignore[attr-defined]
    return card, outer
