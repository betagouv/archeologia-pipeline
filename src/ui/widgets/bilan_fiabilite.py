"""Bilan de fiabilité de fin de run — barres par entité, du plus sûr au plus douteux.

Une rangée par couche de détections : le libellé de l'entité, une barre dont la
longueur est proportionnelle au nombre de détections (la plus fournie prend
toute la largeur) et dont les segments, du plus sûr au plus douteux, sont dans la
couleur de la couche déclinée par niveau — exactement les teintes de la légende
de QGIS (``profil_scores.teinte_niveau``) —, puis le total. Survoler une rangée
donne la phrase du bilan. Données : module pur :mod:`app.services.bilan_fiabilite`.
Compatible Qt5/Qt6 : énumérés scopés.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from qgis.PyQt.QtCore import QEvent, QRectF, QSize, Qt
from qgis.PyQt.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from qgis.PyQt.QtWidgets import QSizePolicy, QToolTip, QWidget

from ...app.services.bilan_fiabilite import LigneBilan
from .profil_scores import RGB, couleur_de_classe, teinte_niveau

_RANGEE = 22
_LARGEUR_LIBELLE = 190
_LARGEUR_TOTAL = 56
_BLEU_DEFAUT: RGB = (42, 120, 214)
_ENCRE = QColor("#2c2c2c")
_ENCRE_DOUCE = QColor("#5a5a5a")
_FOND_BARRE = QColor("#efefef")


def _nb(n: int) -> str:
    return f"{n:,}".replace(",", " ")


class BilanFiabiliteWidget(QWidget):
    """Les barres du bilan ; hauteur fixée par le nombre de rangées."""

    def __init__(self, lignes: Sequence[LigneBilan], parent=None):
        super().__init__(parent)
        self._lignes: List[LigneBilan] = list(lignes)
        self._bases: List[RGB] = [couleur_de_classe(ligne.couche) or _BLEU_DEFAUT for ligne in self._lignes]
        self.setObjectName("BilanFiabilite")
        h = len(self._lignes) * _RANGEE + 6
        self.setMinimumHeight(h)
        self.setMaximumHeight(h)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setToolTip("\n".join(ligne.phrase() for ligne in self._lignes))

    def sizeHint(self) -> QSize:  # noqa: N802 (signature Qt)
        return QSize(480, len(self._lignes) * _RANGEE + 6)

    def _rangee_sous(self, y: float) -> Optional[LigneBilan]:
        i = int((y - 3) // _RANGEE)
        return self._lignes[i] if 0 <= i < len(self._lignes) else None

    def event(self, ev) -> bool:  # noqa: N802 (signature Qt)
        if ev.type() == QEvent.Type.ToolTip:
            ligne = self._rangee_sous(ev.pos().y())
            if ligne is not None:
                QToolTip.showText(ev.globalPos(), ligne.phrase(), self)
                return True
        return super().event(ev)

    def paintEvent(self, _ev) -> None:  # noqa: N802 (signature Qt)
        if not self._lignes:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        police = QFont(self.font())
        police.setPointSizeF(max(7.0, self.font().pointSizeF() - 1))
        p.setFont(police)
        fm = QFontMetrics(police)
        maximum = max(ligne.total for ligne in self._lignes) or 1
        x0 = _LARGEUR_LIBELLE
        largeur_barre = max(40, self.width() - x0 - _LARGEUR_TOTAL - 8)
        for i, (ligne, base) in enumerate(zip(self._lignes, self._bases)):
            y = 3 + i * _RANGEE
            p.setPen(_ENCRE)
            libelle = fm.elidedText(ligne.label, Qt.TextElideMode.ElideRight, _LARGEUR_LIBELLE - 10)
            p.drawText(QRectF(0, y, _LARGEUR_LIBELLE - 10, _RANGEE), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, libelle)
            # fond de barre : la place de la plus fournie
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(_FOND_BARRE)
            p.drawRoundedRect(QRectF(x0, y + 4, largeur_barre, _RANGEE - 8), 3, 3)
            longueur = largeur_barre * ligne.total / maximum
            x = x0
            for cat, n in ligne.par_niveau():
                if n <= 0:
                    continue
                seg = longueur * n / ligne.total
                p.setBrush(teinte_niveau(base, cat))
                p.drawRect(QRectF(x, y + 4, seg, _RANGEE - 8))
                x += seg
            p.setPen(QPen(QColor("#ffffff"), 1))
            x = x0
            for cat, n in ligne.par_niveau()[:-1]:
                x += longueur * n / ligne.total if ligne.total else 0
                p.drawLine(QRectF(x, y + 4, 0, _RANGEE - 8).topLeft(), QRectF(x, y + 4, 0, _RANGEE - 8).bottomLeft())
            p.setPen(_ENCRE_DOUCE)
            p.drawText(QRectF(x0 + largeur_barre + 6, y, _LARGEUR_TOTAL, _RANGEE), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, _nb(ligne.total))
        p.end()


def legende_niveaux(lignes: Sequence[LigneBilan]) -> List[Tuple[str, str]]:
    """``[(catégorie, libellé), …]`` des niveaux présents, du plus sûr au plus douteux
    — pour une légende textuelle sous les barres."""
    from ...app.services.bilan_fiabilite import NIVEAUX_DU_PLUS_SUR
    from ...app.services.fiabilite import LABELS_FR

    presents = {c for ligne in lignes for c, _n in ligne.par_niveau()}
    return [(c, LABELS_FR.get(c, c)) for c in NIVEAUX_DU_PLUS_SUR if c in presents]
