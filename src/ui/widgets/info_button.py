"""InfoButton — le seul bouton de fiche de l'interface (interface rationalisée, 2026-10-24).

Un « i » gras dans un rond de 24 px (le glyphe ⓘ dessinait un second cercle dans le
rond du bouton ; « i » seul depuis le 2026-10-25, comme les maquettes), posé à droite du nom de l'objet (image, structure,
modèle) ; il ouvre la fiche de cet objet et consomme son clic, il ne coche donc
jamais la carte qui le porte. Style ``#InfoButton`` dans ``theme/v2.qss``.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import QSize, Qt
from qgis.PyQt.QtWidgets import QToolButton

COTE = 24


class InfoButton(QToolButton):
    def __init__(self, tooltip: str = "Voir la fiche", parent=None):
        super().__init__(parent)
        self.setObjectName("InfoButton")
        self.setText("i")
        self.setFixedSize(QSize(COTE, COTE))
        self.setAutoRaise(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip(tooltip)
