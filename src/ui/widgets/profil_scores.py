"""Profil des scores d'une classe — la figure derrière les quatre niveaux de fiabilité.

Barres empilées par bande de score (pas de 0,05) : les fausses détections de
l'évaluation en gris, les vraies dans la couleur de la classe déclinée par niveau
— exactement les teintes de la légende de QGIS (``fiabilite.STYLE_SPEC`` +
``color_palette.apply_confidence``). Les coupures des niveaux sont tracées, chaque
niveau est nommé sous l'axe avec sa part de vrais objets mesurée et son effectif.
On voit ainsi d'où viennent les coupures : sous le seuil presque tout est faux,
au-dessus de la dernière coupure presque tout est vrai.

Depuis 2026-10-08 :

- le **seuil est mobile** (:meth:`ProfilScoresWidget.set_seuil`) : la ligne suit le
  seuil réglé par l'utilisateur, les niveaux se recalculent comme au run
  (``categories_effectives``) et :meth:`~ProfilScoresWidget.bilan` dit ce que ce
  seuil garde et écarte sur le banc ;
- une **ligne pointillée** marque le point d'équilibre précision-rappel de
  l'évaluation (``seuil_f1max``) : on voit que le seuil déployé est en dessous ;
- **survoler une barre** donne ses vraies et ses fausses ;
- **clic droit** : enregistrer ou copier l'image (rapport, présentation) ;
- un mode **mini** (``mini=True``) sans libellés de niveau, pour la carte d'entité
  (étape 3) et les petits multiples par zone d'évaluation.

Données : module pur :mod:`app.services.profil_scores`. Dessin au pinceau, aucune
dépendance ; net à toute densité d'écran (QPainter dessine en pixels logiques).
Compatible Qt5/Qt6 : énumérés scopés.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from qgis.PyQt.QtCore import QEvent, QRectF, Qt
from qgis.PyQt.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from qgis.PyQt.QtWidgets import (
    QApplication,
    QFileDialog,
    QMenu,
    QSizePolicy,
    QToolTip,
    QWidget,
)

from ...app.services.fiabilite import (
    STYLE_SPEC,
    Categorie,
    categories_effectives,
    pct,
    phrase_mesure,
)
from ...app.services.profil_scores import (
    Bande,
    Bilan,
    Profil,
    bilan_au_seuil,
    libelle_zone,
    profil_pour_classe,
    profils_par_zone,
)

RGB = Tuple[int, int, int]
_BLEU_DEFAUT: RGB = (42, 120, 214)
_GRIS_FAUX = QColor("#d9d9d9")
_GRIS_SOUS_SEUIL = QColor("#9a9a9a")
_ENCRE = QColor("#2c2c2c")
_ENCRE_DOUCE = QColor("#5a5a5a")
_GRILLE = QColor("#e6e6e6")
_SEUIL = QColor("#e8590c")            # ligne du seuil appliqué : jamais une couleur de classe
_COUPURE = QColor("#8a8a8a")          # les autres coupures, discrètes
_HAUTEUR = 224
_HAUTEUR_MINI = 96


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


#: Couleur d'un niveau à partir de la couleur de base d'une couche — partagé avec
#: le bilan de fin de run (``ui/widgets/bilan_fiabilite``).
teinte_niveau = _teinte


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


def _v(x: float) -> str:
    return f"{x:g}".replace(".", ",")


class ProfilScoresWidget(QWidget):
    """Barres vraies/fausses par bande de score, coupures et niveaux de la classe."""

    def __init__(
        self,
        profil: Profil,
        couleur_base: Optional[RGB] = None,
        parent=None,
        *,
        mini: bool = False,
    ):
        super().__init__(parent)
        self._p = profil
        self._base: RGB = couleur_base or _BLEU_DEFAUT
        self._mini = bool(mini)
        self._seuil: Optional[float] = None          # seuil mobile ; None = celui du modèle
        self._cats: Tuple[Categorie, ...] = profil.categories
        self.setObjectName("ProfilScoresMini" if mini else "ProfilScores")
        h = _HAUTEUR_MINI if mini else _HAUTEUR
        self.setMinimumHeight(h)
        self.setMaximumHeight(h)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._maj_infobulle()

    # ------------------------------------------------------------------
    # Seuil mobile et bilan
    # ------------------------------------------------------------------
    @property
    def profil(self) -> Profil:
        return self._p

    @property
    def seuil(self) -> float:
        return self._p.seuil if self._seuil is None else self._seuil

    def set_seuil(self, seuil: Optional[float]) -> None:
        """Déplace la ligne du seuil ; les niveaux suivent la règle du run
        (``categories_effectives``). ``None`` = retour au seuil du modèle."""
        self._seuil = None if seuil is None else float(seuil)
        self._cats = (
            self._p.categories if self._seuil is None
            else categories_effectives(self._p.categories, self._seuil)
        )
        self._maj_infobulle()
        self.update()

    def bilan(self) -> Bilan:
        """Ce que le seuil courant garde et écarte sur le banc."""
        return bilan_au_seuil(self._p.fines or self._p.bandes, self.seuil)

    def phrase_bilan(self) -> str:
        """Le bilan du seuil courant, dit par rapport au seuil du modèle."""
        reference = bilan_au_seuil(self._p.fines or self._p.bandes, self._p.seuil)
        # La case « Confiance » affiche deux décimales : un seuil de modèle à 0,245
        # y devient 0,24 — à moins d'une demi-décimale, c'est le seuil du modèle.
        if abs(self.seuil - self._p.seuil) < 0.005:
            return reference.phrase()
        return self.bilan().phrase(reference)

    def _maj_infobulle(self) -> None:
        p = self._p
        lignes = [f"{_nb(p.total)} détections de l'évaluation, par bande de score de 0,05."]
        if p.zone:
            lignes.insert(0, f"Zone d'évaluation : {p.zone}")
        for c in self._cats:
            lignes.append(f"{c.label} : score ≥ {_v(c.seuil)} — {phrase_mesure(c)}".replace(".", ","))
        if p.seuil_f1max is not None:
            lignes.append(
                f"Point d'équilibre précision-rappel de l'évaluation : {_v(p.seuil_f1max)} "
                "(ligne pointillée) — le seuil déployé est choisi en dessous : en prospection, "
                "une structure manquée ne se rattrape pas."
            )
        lignes.append("Survolez une barre pour ses effectifs ; clic droit pour enregistrer l'image.")
        self.setToolTip("\n".join(lignes))

    # ------------------------------------------------------------------
    # Géométrie partagée entre le dessin et l'infobulle
    # ------------------------------------------------------------------
    def _cadre(self):
        bandes = self._p.bandes
        w, h = self.width(), self.height()
        if self._mini:
            gauche, droite, haut, bas = 30, 6, 14, 18
        else:
            gauche, droite, haut, bas = 44, 8, 36, 72   # trois rangées d'étiquettes en haut
        xmin, xmax = bandes[0].lo, bandes[-1].hi
        seuil = self.seuil
        # L'échelle se règle sur les bandes AU-DESSUS du seuil du MODÈLE : sous le
        # seuil, les fausses détections sont dix à cent fois plus nombreuses et
        # écraseraient tout le reste (Enclos : 5 600 écartées pour 138 gardées).
        # Les barres écartées dépassent et sont rognées en haut ; leur effectif
        # est écrit sous l'axe. Un seuil mobile ne change pas l'échelle (la
        # figure ne saute pas quand on règle).
        au_dessus = [b.total for b in bandes if b.hi > self._p.seuil + 1e-9]
        maximum = max(au_dessus or [b.total for b in bandes]) or 1
        largeur_trace = w - gauche - droite
        hauteur_trace = h - haut - bas

        def x(v: float) -> float:
            return gauche + (v - xmin) / (xmax - xmin) * largeur_trace

        def y(c: float) -> float:
            return haut + hauteur_trace * (1 - c / maximum)

        return gauche, droite, haut, bas, xmin, xmax, seuil, maximum, largeur_trace, hauteur_trace, x, y

    def _bande_sous(self, px: float) -> Optional[Bande]:
        if not self._p.bandes or self.width() < 120:
            return None
        *_, xmin, xmax, _s, _m, _lt, _ht, x, _y = self._cadre()
        for b in self._p.bandes:
            if x(b.lo) <= px < x(b.hi):
                return b
        return None

    # ------------------------------------------------------------------
    # Événements : infobulle par barre, menu contextuel
    # ------------------------------------------------------------------
    def event(self, ev) -> bool:  # noqa: N802 (signature Qt)
        if ev.type() == QEvent.Type.ToolTip:
            b = self._bande_sous(ev.pos().x())
            if b is not None:
                part = b.part_vrais
                texte = (
                    f"Scores de {_v(b.lo)} à {_v(b.hi)} : {_nb(b.tp)} vraie{'s' if b.tp > 1 else ''}, "
                    f"{_nb(b.fp)} fausse{'s' if b.fp > 1 else ''}"
                    + (f" — {round(part * 100)} % de vrais objets" if part is not None else "")
                )
                if b.hi <= self.seuil + 1e-9:
                    texte += " · écartées par le seuil"
                QToolTip.showText(ev.globalPos(), texte, self)
                return True
        return super().event(ev)

    def contextMenuEvent(self, ev) -> None:  # noqa: N802 (signature Qt)
        menu = QMenu(self)
        enregistrer = menu.addAction("Enregistrer l'image…")
        copier = menu.addAction("Copier l'image")
        choix = menu.exec(ev.globalPos())
        if choix == enregistrer:
            self.enregistrer_image()
        elif choix == copier:
            QApplication.clipboard().setPixmap(self.image())

    def image(self, echelle: int = 2) -> QPixmap:
        """La figure rendue en double résolution, fond blanc (rapport, diaporama)."""
        pm = QPixmap(self.width() * echelle, self.height() * echelle)
        pm.setDevicePixelRatio(echelle)
        pm.fill(QColor("#ffffff"))
        self.render(pm)
        return pm

    def enregistrer_image(self) -> None:
        suffixe = f"_{libelle_zone(self._p.zone).lower().replace(' ', '_')}" if self._p.zone else ""
        defaut = Path.home() / f"profil_{self._p.classe}{suffixe}.png"
        chemin, _f = QFileDialog.getSaveFileName(
            self, "Enregistrer la figure", str(defaut), "Image PNG (*.png)"
        )
        if chemin:
            self.image().save(chemin, "PNG")

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
        w = self.width()
        if not bandes or w < 120:
            p.setPen(_ENCRE_DOUCE)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Profil des scores indisponible")
            return

        gauche, droite, haut, bas, xmin, xmax, seuil, maximum, largeur_trace, hauteur_trace, x, y = self._cadre()
        cats = self._cats
        n_ecartees = sum(b.total for b in (prof.fines or bandes) if b.hi <= seuil + 1e-9)
        mini = self._mini

        # — grille et axe des effectifs —
        pas = _pas_grille(maximum)
        p.setPen(QPen(_GRILLE, 1))
        valeur = 0
        while valeur <= maximum:
            yy = y(valeur)
            p.drawLine(QRectF(gauche, yy, largeur_trace, 0).topLeft(), QRectF(gauche, yy, largeur_trace, 0).topRight())
            if not mini or valeur in (0, pas):
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
                cat = _categorie_de(cats, b.lo)
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
                p.drawText(QRectF(x(v) - 20, y(0) + 2, 40, 12), Qt.AlignmentFlag.AlignHCenter, _v(v))

        # — point d'équilibre précision-rappel : pointillé, libellé à gauche de la
        #   ligne (les coupures sont libellées à droite de la leur) —
        if prof.seuil_f1max is not None and xmin <= prof.seuil_f1max <= xmax:
            xe = x(prof.seuil_f1max)
            p.setPen(QPen(_ENCRE_DOUCE, 1, Qt.PenStyle.DashLine))
            p.drawLine(QRectF(xe, haut - 4, 0, y(0) - haut + 4).topLeft(), QRectF(xe, haut - 4, 0, y(0) - haut + 4).bottomLeft())
            if not mini:
                # Rangée du haut, à elle seule : à gauche de sa ligne s'il y a la place,
                # sinon à droite (les coupures occupent les deux rangées suivantes).
                p.setFont(petite)
                texte = f"équilibre {_v(prof.seuil_f1max)}"
                if xe - gauche > 84:
                    p.drawText(QRectF(xe - 83, haut - 30, 80, 12), Qt.AlignmentFlag.AlignRight, texte)
                else:
                    p.drawText(QRectF(xe + 3, haut - 30, 80, 12), Qt.AlignmentFlag.AlignLeft, texte)

        # — coupures des niveaux en gris, étiquettes en quinconce (0,29 et 0,35 se
        #   touchent) ; le SEUIL APPLIQUÉ (première coupure) dans sa couleur propre,
        #   plus épais, sur un halo blanc : il doit se voir quelle que soit la
        #   couleur de la classe (constat utilisateur 2026-10-08). —
        p.setFont(petite if mini else police)
        for i, c in enumerate(cats):
            xc = x(c.seuil)
            if i == 0:
                p.setPen(QPen(QColor("#ffffff"), 5))
                p.drawLine(QRectF(xc, haut - 4, 0, y(0) - haut + 4).topLeft(), QRectF(xc, haut - 4, 0, y(0) - haut + 4).bottomLeft())
                p.setPen(QPen(_SEUIL, 2))
            else:
                p.setPen(QPen(_COUPURE, 1))
            p.drawLine(QRectF(xc, haut - 4, 0, y(0) - haut + 4).topLeft(), QRectF(xc, haut - 4, 0, y(0) - haut + 4).bottomLeft())
            if mini and i > 0:
                continue  # en mini, seule la ligne du seuil est libellée
            texte = ("seuil " if i == 0 else "") + _v(c.seuil)
            rect = QRectF(xc + 3, haut - (12 if mini else 18) + (0 if mini else (i % 2) * 11), 70, 12)
            if i == 0:
                grasse = QFont(p.font())
                grasse.setBold(True)
                p.setFont(grasse)
                p.setPen(_SEUIL)
            else:
                p.setPen(_COUPURE)
            p.drawText(rect, Qt.AlignmentFlag.AlignLeft, texte)
            p.setFont(petite if mini else police)
        if mini:
            p.end()
            return

        # — niveaux sous l'axe : nom, puis part mesurée et effectif. Une bande
        #   étroite (douteux 0,29–0,35) reçoit un rectangle de 76 px centré sur
        #   elle et descend d'une rangée, pour ne rien rogner ni chevaucher. —
        p.setFont(police)
        for i, c in enumerate(cats):
            fin = cats[i + 1].seuil if i + 1 < len(cats) else xmax
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
        if x(seuil) - x(xmin) > 40:
            p.setPen(_ENCRE_DOUCE)
            p.setFont(police)
            p.drawText(QRectF(x(xmin), y(0) + 14, x(seuil) - x(xmin), 14), Qt.AlignmentFlag.AlignHCenter, "écartées")
            p.setFont(petite)
            p.drawText(QRectF(x(xmin), y(0) + 28, x(seuil) - x(xmin), 12), Qt.AlignmentFlag.AlignHCenter, _nb(n_ecartees))
        p.end()


def _categorie_de(cats: Sequence[Categorie], score: float) -> Optional[Categorie]:
    courante = None
    for c in cats:
        if score >= c.seuil - 1e-9:
            courante = c
    return courante


def couleur_de_classe(classe: str) -> Optional[RGB]:
    """Couleur de base de la classe (registre partagé) ; ``None`` hors QGIS.

    ``classe`` est la **clé du registre**, c'est-à-dire le nom de la couche de
    détections : le nom de classe en temps normal, « classe — Modèle » en
    comparaison A/B (cf. ``model_orchestrator.layer_name_for_class``) — la même
    clé que ``layer_loader``, sinon la figure et la légende divergent.
    """
    try:
        from ...pipeline.cv.class_color_registry import color_for_class

        r, g, b = color_for_class(classe)
        return int(r), int(g), int(b)
    except Exception:  # noqa: BLE001
        return None


def figure_profil(
    model_dir: Optional[Path],
    classe: str,
    categories: Sequence[Categorie],
    parent=None,
    *,
    couleur: Optional[RGB] = None,
    mini: bool = False,
) -> Optional[ProfilScoresWidget]:
    """La figure prête à poser, ou ``None`` si l'évaluation n'est pas livrée.
    ``couleur`` : couleur de base déjà résolue (couche qualifiée en A/B) ; sinon
    celle du registre pour ``classe``."""
    if model_dir is None or not categories:
        return None
    profil = profil_pour_classe(Path(model_dir), classe, categories)
    if profil is None:
        return None
    return ProfilScoresWidget(profil, couleur or couleur_de_classe(classe), parent, mini=mini)


def figures_par_zone(
    model_dir: Optional[Path],
    classe: str,
    categories: Sequence[Categorie],
    parent=None,
    *,
    couleur: Optional[RGB] = None,
) -> List[Tuple[str, ProfilScoresWidget]]:
    """``[(libellé de zone, figure mini), …]`` — petits multiples par zone
    d'évaluation ; vide s'il y a moins de deux zones."""
    if model_dir is None or not categories:
        return []
    base = couleur or couleur_de_classe(classe)
    return [
        (libelle_zone(p.zone), ProfilScoresWidget(p, base, parent, mini=True))
        for p in profils_par_zone(Path(model_dir), classe, categories)
    ]
