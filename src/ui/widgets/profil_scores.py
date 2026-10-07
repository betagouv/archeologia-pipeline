"""Profil des scores d'une classe — la figure derrière les quatre niveaux de fiabilité.

Barres empilées par bande de score (pas de 0,05) : les fausses détections de
l'évaluation en gris, les vraies dans la couleur de la classe déclinée par niveau
— exactement les teintes de la légende de QGIS (``fiabilite.STYLE_SPEC`` +
``color_palette.apply_confidence``). Les coupures des niveaux sont tracées, chaque
niveau est nommé sous l'axe avec sa part de vrais objets mesurée et son effectif.
On voit ainsi d'où viennent les coupures : sous le seuil presque tout est faux,
au-dessus de la dernière coupure presque tout est vrai.

Données : module pur :mod:`app.services.profil_scores`. Dessin au pinceau, aucune
dépendance ; net à toute densité d'écran (QPainter dessine en pixels logiques).
Compatible Qt5/Qt6 : énumérés scopés.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

from qgis.PyQt.QtCore import QRectF, Qt
from qgis.PyQt.QtGui import QColor, QFont, QPainter, QPen
from qgis.PyQt.QtWidgets import QSizePolicy, QWidget

from ...app.services.fiabilite import STYLE_SPEC, Categorie, pct, phrase_mesure
from ...app.services.profil_scores import Profil, profil_pour_classe

RGB = Tuple[int, int, int]
_BLEU_DEFAUT: RGB = (42, 120, 214)
_GRIS_FAUX = QColor("#d9d9d9")
_GRIS_SOUS_SEUIL = QColor("#9a9a9a")
_ENCRE = QColor("#2c2c2c")
_ENCRE_DOUCE = QColor("#5a5a5a")
_GRILLE = QColor("#e6e6e6")
_HAUTEUR = 224


def _teinte(base: RGB, categorie: str) -> QColor:
    """Couleur du niveau = couleur de la classe déclinée comme dans la légende."""
    repr_ = float(STYLE_SPEC.get(categorie, {}).get("repr", 0.5))
    try:
        from ...pipeline.cv.color_palette import apply_confidence

        r, g, b = apply_confidence(base, repr_)
        return QColor(int(r), int(g), int(b))
    except Exception:  # noqa: BLE001 — hors QGIS : déclinaison simple en clarté
        c = QColor(*base)
        return c.darker(100 + int((repr_ - 0.5) * 80)) if repr_ >= 0.5 else c.lighter(100 + int((0.5 - repr_) * 120))


def _pas_grille(maximum: int) -> int:
    """Un pas « rond » qui donne 2 à 4 lignes de grille."""
    if maximum <= 0:
        return 1
    brut = maximum / 3
    for exp in range(0, 7):          # 1, 2, 5, 10, 20, 50, 100… : le premier ≥ maximum/3
        for base in (1, 2, 5):
            pas = base * 10 ** exp
            if pas >= brut:
                return pas
    return max(1, int(brut))


def _nb(n: int) -> str:
    return f"{n:,}".replace(",", " ")


class ProfilScoresWidget(QWidget):
    """Barres vraies/fausses par bande de score, coupures et niveaux de la classe."""

    def __init__(self, profil: Profil, couleur_base: Optional[RGB] = None, parent=None):
        super().__init__(parent)
        self._p = profil
        self._base: RGB = couleur_base or _BLEU_DEFAUT
        self.setObjectName("ProfilScores")
        self.setMinimumHeight(_HAUTEUR)
        self.setMaximumHeight(_HAUTEUR)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lignes = [f"{_nb(profil.total)} détections de l'évaluation, par bande de score de 0,05."]
        for c in profil.categories:
            lignes.append(f"{c.label} : score ≥ {c.seuil:g} — {phrase_mesure(c)}".replace(".", ","))
        self.setToolTip("\n".join(lignes))

    # ------------------------------------------------------------------
    def paintEvent(self, _ev) -> None:  # noqa: N802 (signature Qt)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        police = QFont(self.font())
        police.setPointSizeF(max(7.0, self.font().pointSizeF() - 1))
        p.setFont(police)
        petite = QFont(police)
        petite.setPointSizeF(max(6.5, police.pointSizeF() - 1))

        prof = self._p
        bandes = prof.bandes
        w, h = self.width(), self.height()
        if not bandes or w < 120:
            p.setPen(_ENCRE_DOUCE)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Profil des scores indisponible")
            return

        gauche, droite, haut, bas = 44, 8, 24, 72
        xmin, xmax = bandes[0].lo, bandes[-1].hi
        # L'échelle se règle sur les bandes AU-DESSUS du seuil : sous le seuil, les
        # fausses détections sont dix à cent fois plus nombreuses et écraseraient
        # tout le reste (Enclos : 5 600 écartées pour 138 gardées). Les barres
        # écartées dépassent et sont rognées en haut ; leur effectif est écrit
        # sous l'axe.
        au_dessus = [b.total for b in bandes if b.hi > prof.seuil + 1e-9]
        maximum = max(au_dessus or [b.total for b in bandes]) or 1
        n_ecartees = prof.n_sous_seuil or sum(b.total for b in bandes if b.hi <= prof.seuil + 1e-9)
        largeur_trace = w - gauche - droite
        hauteur_trace = h - haut - bas

        def x(v: float) -> float:
            return gauche + (v - xmin) / (xmax - xmin) * largeur_trace

        def y(c: float) -> float:
            return haut + hauteur_trace * (1 - c / maximum)

        # — grille et axe des effectifs —
        pas = _pas_grille(maximum)
        p.setPen(QPen(_GRILLE, 1))
        valeur = 0
        while valeur <= maximum:
            yy = y(valeur)
            p.drawLine(QRectF(gauche, yy, largeur_trace, 0).topLeft(), QRectF(gauche, yy, largeur_trace, 0).topRight())
            p.setPen(_ENCRE_DOUCE)
            p.setFont(petite)
            libelle = "0" if valeur == 0 else (f"{valeur // 1000} k" if valeur >= 1000 else str(valeur))
            p.drawText(QRectF(0, yy - 8, gauche - 6, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, libelle)
            p.setPen(QPen(_GRILLE, 1))
            valeur += pas

        # — barres : fausses (gris) puis vraies (teinte du niveau) —
        p.setPen(Qt.PenStyle.NoPen)
        p.setClipRect(QRectF(gauche, haut, largeur_trace, hauteur_trace + 1))
        for b in bandes:
            x0, x1 = x(b.lo) + 1, x(b.hi) - 1
            if x1 <= x0:
                continue
            p.setBrush(_GRIS_FAUX)
            p.drawRect(QRectF(x0, y(b.fp), x1 - x0, y(0) - y(b.fp)))
            if b.tp > 0:
                cat = prof.categorie_de(b.lo)
                p.setBrush(_teinte(self._base, cat.categorie) if cat else _GRIS_SOUS_SEUIL)
                y_haut, y_bas = y(b.fp + b.tp), y(b.fp)
                if y_bas - y_haut > 2:
                    p.drawRoundedRect(QRectF(x0, y_haut, x1 - x0, y_bas - y_haut - 2), 2, 2)
        p.setClipping(False)

        # — axe des scores —
        p.setPen(QPen(QColor("#c4c4c4"), 1))
        p.drawLine(QRectF(gauche, y(0), largeur_trace, 0).topLeft(), QRectF(gauche, y(0), largeur_trace, 0).topRight())
        p.setFont(petite)
        p.setPen(_ENCRE_DOUCE)
        for v in (0.1, 0.3, 0.5, 0.7, 0.9):
            if xmin <= v <= xmax:
                p.drawText(QRectF(x(v) - 20, y(0) + 2, 40, 12), Qt.AlignmentFlag.AlignHCenter, f"{v:g}".replace(".", ","))

        # — coupures des niveaux, étiquettes en quinconce (0,29 et 0,35 se touchent) —
        p.setFont(police)
        for i, c in enumerate(prof.categories):
            xc = x(c.seuil)
            p.setPen(QPen(_ENCRE, 1))
            p.drawLine(QRectF(xc, haut - 4, 0, y(0) - haut + 4).topLeft(), QRectF(xc, haut - 4, 0, y(0) - haut + 4).bottomLeft())
            texte = ("seuil " if i == 0 else "") + f"{c.seuil:g}".replace(".", ",")
            p.drawText(QRectF(xc + 3, haut - 18 + (i % 2) * 11, 70, 12), Qt.AlignmentFlag.AlignLeft, texte)

        # — niveaux sous l'axe : nom, puis part mesurée et effectif. Une bande
        #   étroite (douteux 0,29–0,35) reçoit un rectangle de 76 px centré sur
        #   elle et descend d'une rangée, pour ne rien rogner ni chevaucher. —
        p.setFont(police)
        for i, c in enumerate(prof.categories):
            fin = prof.categories[i + 1].seuil if i + 1 < len(prof.categories) else xmax
            x0, x1 = x(c.seuil), x(min(fin, xmax))
            etroite = (x1 - x0) < 76
            cx = (x0 + x1) / 2
            largeur = max(x1 - x0, 76.0)
            rect_x = min(max(cx - largeur / 2, 0.0), w - largeur)
            decal = 26 if etroite else 0
            p.setPen(_ENCRE)
            p.setFont(police)
            p.drawText(QRectF(rect_x, y(0) + 14 + decal, largeur, 14), Qt.AlignmentFlag.AlignHCenter, c.label)
            p.setPen(_ENCRE_DOUCE)
            p.setFont(petite)
            mesure = pct(c.mesure)
            detail = f"{mesure} % · {_nb(c.n)}" if mesure is not None else f"{_nb(c.n)} dét."
            p.drawText(QRectF(rect_x, y(0) + 28 + decal, largeur, 12), Qt.AlignmentFlag.AlignHCenter, detail)
            if etroite:
                p.setPen(QPen(QColor("#c4c4c4"), 1))
                p.drawLine(QRectF(cx, y(0) + 12, 0, 14).topLeft(), QRectF(cx, y(0) + 12, 0, 14).bottomLeft())
        if x(prof.seuil) - x(xmin) > 40:
            p.setPen(_ENCRE_DOUCE)
            p.setFont(police)
            p.drawText(QRectF(x(xmin), y(0) + 14, x(prof.seuil) - x(xmin), 14), Qt.AlignmentFlag.AlignHCenter, "écartées")
            p.setFont(petite)
            p.drawText(QRectF(x(xmin), y(0) + 28, x(prof.seuil) - x(xmin), 12), Qt.AlignmentFlag.AlignHCenter, _nb(n_ecartees))
        p.end()


def couleur_de_classe(classe: str) -> Optional[RGB]:
    """Couleur de base de la classe (registre partagé) ; ``None`` hors QGIS."""
    try:
        from ...pipeline.cv.class_color_registry import color_for_class

        r, g, b = color_for_class(classe)
        return int(r), int(g), int(b)
    except Exception:  # noqa: BLE001
        return None


def figure_profil(
    model_dir: Optional[Path], classe: str, categories: Sequence[Categorie], parent=None
) -> Optional[ProfilScoresWidget]:
    """La figure prête à poser, ou ``None`` si l'évaluation n'est pas livrée avec le modèle."""
    if model_dir is None or not categories:
        return None
    profil = profil_pour_classe(Path(model_dir), classe, categories)
    if profil is None:
        return None
    return ProfilScoresWidget(profil, couleur_de_classe(classe), parent)
