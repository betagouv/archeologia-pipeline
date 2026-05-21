"""Carte (#Card) du wizard V2 — conteneur blanc bordé avec en-tête optionnel.

En-tête = pastille numérotée (#CardNum) + titre (#CardTitle), comme les ``.gb``
numérotées de la maquette. Helper partagé par toutes les pages d'étapes.
"""
from __future__ import annotations

from typing import Tuple

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout


def build_card(title: str = "", num: str = "") -> Tuple[QFrame, QVBoxLayout]:
    """Retourne ``(carte, layout_contenu)``. Ajouter le contenu au layout."""
    card = QFrame()
    card.setObjectName("Card")
    outer = QVBoxLayout(card)
    outer.setContentsMargins(12, 10, 12, 12)
    outer.setSpacing(10)
    if title:
        header = QHBoxLayout()
        header.setSpacing(8)
        if num:
            badge = QLabel(num)
            badge.setObjectName("CardNum")
            badge.setFixedSize(18, 18)
            badge.setAlignment(Qt.AlignCenter)
            header.addWidget(badge)
        tlabel = QLabel(title)
        tlabel.setObjectName("CardTitle")
        header.addWidget(tlabel)
        header.addStretch(1)
        outer.addLayout(header)
    return card, outer
