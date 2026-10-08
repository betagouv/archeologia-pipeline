"""Bandeau « Appris sur » : carte des zones d'apprentissage d'une classe et liste des zones.

Posé en tête du bloc « Ce que le modèle a appris » de la fiche de classe (demande
utilisateur 2026-10-08 : une carte unique qui dit où la classe a été apprise et mesurée,
pas de carte dans « Par zone d'évaluation »). À gauche, une petite carte par pays
concerné (France, Irlande) : un disque par zone, dans la couleur de la couche, d'aire
proportionnelle aux objets annotés. À droite, les zones avec une barre et leurs
effectifs ; survoler une ligne entoure sa zone sur la carte, survoler un disque donne
son nom et ses effectifs. Données et géométrie : module pur
:mod:`app.services.carte_zones`. Compatible Qt5/Qt6 : énumérés scopés.
"""
from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

from qgis.PyQt.QtCore import QEvent, QPointF, QSize, Qt
from qgis.PyQt.QtGui import QColor, QPainter, QPainterPath, QPen
from qgis.PyQt.QtWidgets import QFrame, QHBoxLayout, QLabel, QSizePolicy, QToolTip, QVBoxLayout, QWidget

from ...app.services.carte_zones import (
    ZoneSituee,
    anneaux_du_pays,
    pays_presents,
    phrase_resume,
    projection,
    rayon,
    situer,
)
from .vignette import couleur_texte

RGB = Tuple[int, int, int]
_COTE = 150          # côté d'une carte de pays, px logiques
_ECART = 8
_TERRE = QColor("#eef0ea")
_TRAIT = QColor("#a3aaa3")


def _nb(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _effectifs(z: Any) -> str:
    morceaux = []
    if getattr(z, "objets", 0):
        morceaux.append(f"{_nb(z.objets)} objets")
    if getattr(z, "tuiles", 0):
        morceaux.append(f"{_nb(z.tuiles)} tuiles")
    return " · ".join(morceaux)


class CarteZonesWidget(QWidget):
    """Une carte par pays, les zones en disques ; ``surligner(nom)`` entoure une zone."""

    def __init__(self, zones: Sequence[ZoneSituee], data: Mapping[str, Any], couleur: RGB, parent=None):
        super().__init__(parent)
        self._zones = list(zones)
        self._pays = pays_presents(self._zones)
        self._anneaux = {p: anneaux_du_pays(data, p) for p in self._pays}
        self._couleur = tuple(int(v) for v in couleur)
        self._surligne: Optional[str] = None
        self.setObjectName("CarteZones")
        n = max(1, len(self._pays))
        self.setFixedSize(n * _COTE + (n - 1) * _ECART, _COTE)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)

    def sizeHint(self) -> QSize:  # noqa: N802 (signature Qt)
        return self.size()

    def surligner(self, nom: Optional[str]) -> None:
        if nom != self._surligne:
            self._surligne = nom
            self.update()

    def _disques(self) -> List[Tuple[ZoneSituee, float, float, float]]:
        """``(zone, x, y, rayon)`` en px du widget, les plus grosses d'abord."""
        maximum = max((z.objets for z in self._zones), default=0)
        out = []
        for i, pays in enumerate(self._pays):
            f, _l, _h = projection(self._anneaux[pays], pays, _COTE, _COTE)
            dx = i * (_COTE + _ECART)
            for z in self._zones:
                if z.pays == pays:
                    x, y = f(z.lon, z.lat)
                    out.append((z, x + dx, y, rayon(z.objets, maximum, _COTE)))
        return sorted(out, key=lambda t: -t[3])

    def event(self, ev) -> bool:  # noqa: N802 (signature Qt)
        if ev.type() == QEvent.Type.ToolTip:
            pos = ev.pos()
            for z, x, y, r in reversed(self._disques()):          # les petits sont dessus
                if (pos.x() - x) ** 2 + (pos.y() - y) ** 2 <= (r + 2) ** 2:
                    QToolTip.showText(ev.globalPos(), f"{z.nom} — {_effectifs(z)} annotés", self)
                    return True
            QToolTip.hideText()
            return True
        return super().event(ev)

    def paintEvent(self, _ev) -> None:  # noqa: N802 (signature Qt)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for i, pays in enumerate(self._pays):
            f, _l, _h = projection(self._anneaux[pays], pays, _COTE, _COTE)
            dx = i * (_COTE + _ECART)
            chemin = QPainterPath()
            for anneau in self._anneaux[pays]:
                pts = [QPointF(*f(lon, lat)) + QPointF(dx, 0) for lon, lat in anneau]
                if len(pts) > 2:
                    chemin.moveTo(pts[0])
                    for q in pts[1:]:
                        chemin.lineTo(q)
                    chemin.closeSubpath()
            p.setPen(QPen(_TRAIT, 0.8))
            p.setBrush(_TERRE)
            p.drawPath(chemin)
        fond = QColor(*self._couleur)
        fond.setAlpha(205)
        bord = QColor(couleur_texte(self._couleur))
        for z, x, y, r in self._disques():
            p.setPen(QPen(bord, 1))
            p.setBrush(fond)
            p.drawEllipse(QPointF(x, y), r, r)
        if self._surligne:
            for z, x, y, r in self._disques():
                if z.nom == self._surligne:
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.setPen(QPen(QColor("#ffffff"), 4))
                    p.drawEllipse(QPointF(x, y), r + 4, r + 4)
                    p.setPen(QPen(QColor("#1b1b1b"), 1.6))
                    p.drawEllipse(QPointF(x, y), r + 4, r + 4)
        p.end()


class _LigneZone(QWidget):
    """Une zone de la liste : nom, barre proportionnelle aux objets, effectifs. Le survol
    entoure la zone sur la carte."""

    def __init__(self, zone: Any, maximum: int, couleur: RGB, carte: Optional[CarteZonesWidget], parent=None):
        super().__init__(parent)
        self._nom = str(zone.nom)
        self._carte = carte
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        nom = QLabel(self._nom)
        nom.setObjectName("FicheTexte")
        nom.setWordWrap(True)
        piste = QWidget()
        piste.setFixedSize(110, 8)
        barre = QFrame(piste)
        barre.setObjectName("CarteZonesBarre")
        largeur = max(3, round(110 * (zone.objets or 0) / maximum)) if maximum else 3
        barre.setGeometry(0, 0, largeur, 8)
        barre.setStyleSheet("background: rgb(%d, %d, %d); border-radius: 2px;" % tuple(couleur))  # couleur de la couche
        chiffres = QLabel(_effectifs(zone))
        chiffres.setObjectName("FicheLegende")
        chiffres.setMinimumWidth(150)
        nom.setFixedWidth(260)        # colonne fixe : barres et chiffres restent près des noms
        lay.addWidget(nom)
        lay.addWidget(piste, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(chiffres)
        lay.addStretch(1)
        self.setToolTip(f"{self._nom} — {_effectifs(zone)} annotés")

    def enterEvent(self, ev) -> None:  # noqa: N802 (signature Qt)
        if self._carte is not None:
            self._carte.surligner(self._nom)
        super().enterEvent(ev)

    def leaveEvent(self, ev) -> None:  # noqa: N802 (signature Qt)
        if self._carte is not None:
            self._carte.surligner(None)
        super().leaveEvent(ev)


def bandeau_appris_sur(zones: Sequence[Any], data: Mapping[str, Any], couleur: RGB, parent=None) -> Optional[QWidget]:
    """Le bandeau « Appris sur » (carte + liste), ou ``None`` si aucune zone n'est située :
    la fiche garde alors sa liste de zones en texte."""
    situees = situer(zones, data)
    if not situees:
        return None
    cadre = QFrame(parent)
    cadre.setObjectName("CarteZonesBandeau")
    lay = QHBoxLayout(cadre)
    lay.setContentsMargins(10, 8, 10, 8)
    lay.setSpacing(14)
    carte = CarteZonesWidget(situees, data, couleur)
    lay.addWidget(carte, 0, Qt.AlignmentFlag.AlignTop)
    colonne = QVBoxLayout()
    colonne.setSpacing(4)
    titre = QLabel(f"Appris sur · {phrase_resume(zones)}")
    titre.setObjectName("FicheSousTitre")
    colonne.addWidget(titre)
    maximum = max((int(getattr(z, "objets", 0) or 0) for z in zones), default=0)
    for z in zones:
        colonne.addWidget(_LigneZone(z, maximum, couleur, carte))
    colonne.addStretch(1)
    lay.addLayout(colonne, 1)
    return cadre
