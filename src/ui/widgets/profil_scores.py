"""Profil des scores d'une classe — la figure derrière les quatre niveaux de fiabilité.

Barres empilées par bande de score (pas de 0,05) : les fausses détections de
l'évaluation en gris, les vraies dans la couleur de la classe déclinée par niveau
— exactement les teintes de la légende de QGIS (``fiabilite.STYLE_SPEC`` +
``color_palette.apply_confidence``). Les coupures des niveaux sont tracées, chaque
niveau est nommé sous l'axe (avec sa part de vrais objets mesurée et son effectif
en mode complet). On voit ainsi d'où viennent les coupures : sous le seuil presque
tout est faux, au-dessus de la dernière coupure presque tout est vrai.

Depuis 2026-10-08 :

- le **seuil est mobile** (:meth:`ProfilScoresWidget.set_seuil`) : la ligne — orange,
  sur halo blanc, jamais une couleur de classe — suit le seuil réglé par
  l'utilisateur, les niveaux se recalculent comme au run (``categories_effectives``),
  **précision et rappel au banc** pour ce seuil s'écrivent en haut à droite de la
  figure (``profil_scores.precision_rappel``) et :meth:`~ProfilScoresWidget.phrase_bilan`
  dit ce que ce seuil change sur le banc (première ligne de l'infobulle) ;
- une **ligne pointillée** marque le point d'équilibre précision-rappel de
  l'évaluation (« équilibre (F1) ») : on voit que le seuil déployé est en dessous ;
- **survoler une barre** donne ses vraies et ses fausses ;
- **clic droit** : enregistrer ou copier l'image (rapport, présentation) ;
- un mode **mini** (``mini=True``), niveaux nommés sous l'axe sans mesure, pour la
  carte d'entité (étape 3) et les petits multiples par zone d'évaluation ;
- les **libellés sous l'axe ne se chevauchent jamais** : rangés par
  ``profil_scores.disposer_etiquettes`` (pur : première rangée où l'étiquette tient à
  droite de la précédente), la hauteur du widget suit le nombre de rangées
  (``resizeEvent``) et une étiquette décalée reçoit un tiret vers sa bande ;
- :func:`ligne_essai_seuil` : la ligne « Tester un seuil » des fiches (classe et ⓘ),
  qui déplace la ligne de la figure **sans toucher au seuil du traitement**.

Données : module pur :mod:`app.services.profil_scores`. Dessin au pinceau, aucune
dépendance ; net à toute densité d'écran (QPainter dessine en pixels logiques).
Compatible Qt5/Qt6 : énumérés scopés.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from qgis.PyQt.QtCore import QEvent, QRectF, Qt
from qgis.PyQt.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen, QPixmap
from qgis.PyQt.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
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
    disposer_etiquettes,
    libelle_zone,
    phrase_precision_rappel,
    precision_rappel,
    profil_pour_classe,
    profils_par_zone,
)
from .no_wheel import NoWheelDoubleSpinBox

RGB = Tuple[int, int, int]
_BLEU_DEFAUT: RGB = (42, 120, 214)
_GRIS_FAUX = QColor("#d9d9d9")
_GRIS_SOUS_SEUIL = QColor("#9a9a9a")
_ENCRE = QColor("#2c2c2c")
_ENCRE_DOUCE = QColor("#5a5a5a")
_GRILLE = QColor("#e6e6e6")
_TIRET = QColor("#c4c4c4")
_SEUIL = QColor("#e8590c")            # ligne du seuil appliqué : jamais une couleur de classe
_COUPURE = QColor("#8a8a8a")          # les autres coupures, discrètes

# Géométrie : marges (gauche, droite, haut), hauteur du tracé, puis sous l'axe une
# marge de 14 px et des rangées de libellés (nom + mesure en complet, nom seul en mini).
_MARGES = {False: (44, 8, 36), True: (30, 6, 14)}
_TRACE = {False: 116, True: 64}
_RANGEE = {False: 28, True: 12}
_BAS_FIXE = 16


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


class _Etiquette:
    """Un libellé sous l'axe : nom (+ détail en complet), centre de sa bande, rangée."""

    __slots__ = ("nom", "detail", "cx", "largeur", "rangee", "gauche", "couleur")

    def __init__(self, nom: str, detail: str, cx: float, largeur: float, couleur: QColor):
        self.nom, self.detail, self.cx, self.largeur, self.couleur = nom, detail, cx, largeur, couleur
        self.rangee, self.gauche = 0, 0.0


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
        self._rangees = 1
        self.setObjectName("ProfilScoresMini" if mini else "ProfilScores")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._fixer_hauteur(2)
        self._maj_infobulle()

    # ------------------------------------------------------------------
    # Seuil mobile, bilan, précision et rappel
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
        self._ajuster_rangees()
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

    def phrase_precision_rappel(self) -> str:
        """« précision 65 % · rappel 72 % » au seuil courant, sur le banc ; ``""`` sans donnée."""
        return phrase_precision_rappel(*precision_rappel(self._p, self.seuil))

    def _maj_infobulle(self) -> None:
        p = self._p
        lignes = [f"{_nb(p.total)} détections de l'évaluation, par bande de score de 0,05."]
        pr = self.phrase_precision_rappel()
        if pr:
            lignes.insert(0, f"Au seuil {_v(round(self.seuil, 3))} : {pr} (précision comptée sur les bandes, "
                             "rappel lu dans la table de l'évaluation).")
        if p.fines:
            lignes.insert(0, self.phrase_bilan())   # ce que le seuil courant garde et écarte sur le banc
        if p.zone:
            lignes.insert(0, f"Zone d'évaluation : {p.zone}")
        for c in self._cats:
            lignes.append(f"{c.label} : score ≥ {_v(c.seuil)} — {phrase_mesure(c)}".replace(".", ","))
        if p.seuil_f1max is not None:
            lignes.append(
                f"Point d'équilibre précision-rappel (F1 maximal) de l'évaluation : {_v(p.seuil_f1max)} "
                "(ligne pointillée) — le seuil déployé est choisi en dessous : en prospection, "
                "une structure manquée ne se rattrape pas."
            )
        lignes.append("Survolez une barre pour ses effectifs ; clic droit pour enregistrer l'image.")
        self.setToolTip("\n".join(lignes))

    # ------------------------------------------------------------------
    # Géométrie partagée entre le dessin, l'infobulle et la hauteur
    # ------------------------------------------------------------------
    def _polices(self) -> Tuple[QFont, QFont]:
        police = QFont(self.font())
        police.setPointSizeF(max(7.0, self.font().pointSizeF() - 1))
        petite = QFont(police)
        petite.setPointSizeF(max(6.5, police.pointSizeF() - 1))
        return police, petite

    def _hauteur(self, rangees: int) -> int:
        _g, _d, haut = _MARGES[self._mini]
        return haut + _TRACE[self._mini] + _BAS_FIXE + max(1, rangees) * _RANGEE[self._mini]

    def _fixer_hauteur(self, rangees: int) -> None:
        self._rangees = max(1, rangees)
        h = self._hauteur(self._rangees)
        self.setMinimumHeight(h)
        self.setMaximumHeight(h)

    def _cadre(self):
        bandes = self._p.bandes
        w = self.width()
        gauche, droite, haut = _MARGES[self._mini]
        bas = _BAS_FIXE + self._rangees * _RANGEE[self._mini]
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
        hauteur_trace = self.height() - haut - bas

        def x(v: float) -> float:
            return gauche + (v - xmin) / (xmax - xmin) * largeur_trace

        def y(c: float) -> float:
            return haut + hauteur_trace * (1 - c / maximum)

        return gauche, droite, haut, bas, xmin, xmax, seuil, maximum, largeur_trace, hauteur_trace, x, y

    def _etiquettes(self, cats: Optional[Sequence[Categorie]] = None, seuil: Optional[float] = None) -> List[_Etiquette]:
        """Les libellés sous l'axe, rangés sans chevauchement (``disposer_etiquettes``) —
        pour les niveaux courants, ou pour ``cats``/``seuil`` donnés (calcul de hauteur)."""
        if not self._p.bandes or self.width() < 120:
            return []
        gauche, _d, _h, _b, xmin, xmax, seuil_courant, _m, _lt, _ht, x, _y = self._cadre()
        if seuil is None:
            seuil = seuil_courant
        police, petite = self._polices()
        fm, fm_petite = QFontMetrics(police), QFontMetrics(petite)
        mini = self._mini
        cats = tuple(cats) if cats is not None else self._cats
        etiquettes: List[_Etiquette] = []
        if x(seuil) - x(xmin) > 24:
            n_ecartees = sum(b.total for b in (self._p.fines or self._p.bandes) if b.hi <= seuil + 1e-9)
            etiquettes.append(_Etiquette("écartées", "" if mini else _nb(n_ecartees),
                                         (x(xmin) + x(seuil)) / 2, 0.0, _ENCRE_DOUCE))
        for i, c in enumerate(cats):
            fin = cats[i + 1].seuil if i + 1 < len(cats) else xmax
            mesure = pct(c.mesure)
            detail = "" if mini else (f"{mesure} % · {_nb(c.n)}" if mesure is not None else f"{_nb(c.n)} dét.")
            etiquettes.append(_Etiquette(c.label.lower() if mini else c.label, detail,
                                         (x(c.seuil) + x(min(fin, xmax))) / 2, 0.0, _ENCRE))
        for e in etiquettes:
            l_nom = (fm_petite if mini else fm).horizontalAdvance(e.nom)
            l_detail = fm_petite.horizontalAdvance(e.detail) if e.detail else 0
            e.largeur = max(l_nom, l_detail) + (4 if mini else 6)
        for e, (rangee, gauche_x) in zip(etiquettes, disposer_etiquettes(
            [(e.cx, e.largeur) for e in etiquettes], float(self.width()), ecart=4.0,
        )):
            e.rangee, e.gauche = rangee, gauche_x
        return etiquettes

    def _ajuster_rangees(self) -> None:
        """La hauteur = le nombre de rangées nécessaires au PIRE seuil possible (balayage
        par pas de 0,05) à la largeur courante, pas au seuil courant : sinon la figure
        grandissait et rétrécissait en réglant le seuil, et tout ce qui est dessous
        (la ligne « Tester un seuil ») sautait (constat utilisateur 2026-10-08)."""
        candidats = [self._p.seuil, self.seuil] + [round(0.05 * k, 2) for k in range(1, 20)]
        rangees = 1
        for s in candidats:
            cats = self._p.categories if abs(s - self._p.seuil) < 1e-9 else categories_effectives(self._p.categories, s)
            etiquettes = self._etiquettes(cats, s)
            rangees = max(rangees, max((e.rangee for e in etiquettes), default=0) + 1)
        if rangees != self._rangees:
            self._fixer_hauteur(rangees)

    def resizeEvent(self, ev) -> None:  # noqa: N802 (signature Qt)
        super().resizeEvent(ev)
        self._ajuster_rangees()

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
        police, petite = self._polices()
        p.setFont(police)

        prof = self._p
        bandes = prof.bandes
        w = self.width()
        if not bandes or w < 120:
            p.setPen(_ENCRE_DOUCE)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Profil des scores indisponible")
            return

        gauche, droite, haut, bas, xmin, xmax, seuil, maximum, largeur_trace, hauteur_trace, x, y = self._cadre()
        cats = self._cats
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

        # — point d'équilibre précision-rappel (F1) : pointillé, libellé sur la
        #   rangée du haut, à gauche de la ligne s'il y a la place —
        if prof.seuil_f1max is not None and xmin <= prof.seuil_f1max <= xmax:
            xe = x(prof.seuil_f1max)
            p.setPen(QPen(_ENCRE_DOUCE, 1, Qt.PenStyle.DashLine))
            p.drawLine(QRectF(xe, haut - 4, 0, y(0) - haut + 4).topLeft(), QRectF(xe, haut - 4, 0, y(0) - haut + 4).bottomLeft())
            if not mini:
                p.setFont(petite)
                texte = f"équilibre (F1) {_v(prof.seuil_f1max)}"
                if xe - gauche > 104:
                    p.drawText(QRectF(xe - 103, haut - 30, 100, 12), Qt.AlignmentFlag.AlignRight, texte)
                else:
                    p.drawText(QRectF(xe + 3, haut - 30, 100, 12), Qt.AlignmentFlag.AlignLeft, texte)

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

        # — précision et rappel au seuil courant, en haut à droite du tracé, sur un
        #   fond blanc translucide pour rester lisibles au-dessus des barres —
        pr = self.phrase_precision_rappel()
        if pr:
            p.setFont(petite)
            fm = p.fontMetrics()
            largeur_pr = fm.horizontalAdvance(pr) + 8
            rect_pr = QRectF(gauche + largeur_trace - largeur_pr - 2, haut + 1, largeur_pr, 13)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 215))
            p.drawRoundedRect(rect_pr, 3, 3)
            p.setPen(_ENCRE)
            p.drawText(rect_pr, Qt.AlignmentFlag.AlignCenter, pr)

        # — libellés sous l'axe, rangés sans chevauchement ; une étiquette décalée
        #   d'une rangée reçoit un tiret vers le centre de sa bande —
        rangee_h = _RANGEE[mini]
        for e in self._etiquettes():
            y0 = y(0) + 14 + e.rangee * rangee_h
            if e.rangee > 0:
                p.setPen(QPen(_TIRET, 1))
                p.drawLine(QRectF(e.cx, y(0) + 12, 0, y0 - y(0) - 13).topLeft(), QRectF(e.cx, y(0) + 12, 0, y0 - y(0) - 13).bottomLeft())
            p.setPen(e.couleur)
            p.setFont(petite if mini else police)
            p.drawText(QRectF(e.gauche, y0, e.largeur, 14 if not mini else 11), Qt.AlignmentFlag.AlignHCenter, e.nom)
            if e.detail:
                p.setPen(_ENCRE_DOUCE)
                p.setFont(petite)
                p.drawText(QRectF(e.gauche, y0 + 14, e.largeur, 12), Qt.AlignmentFlag.AlignHCenter, e.detail)
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


def ligne_essai_seuil(
    figure: ProfilScoresWidget,
    autres: Sequence[ProfilScoresWidget] = (),
    seuil_initial: Optional[float] = None,
    parent=None,
) -> QWidget:
    """« Tester un seuil » : une case qui déplace la ligne de ``figure`` (et des
    ``autres``, p. ex. les petits multiples par zone), affiche précision et rappel au
    banc pour ce seuil, et un « ↺ » qui revient à ``seuil_initial`` (le seuil réglé à
    l'étape 3, ou celui du modèle). **Aucun effet sur le seuil du traitement** :
    c'est un essai, dans la fiche de classe comme dans la fiche ⓘ du modèle."""
    depart = float(seuil_initial if seuil_initial is not None else figure.profil.seuil)
    ligne = QWidget(parent)
    ligne.setObjectName("ProfilEssaiSeuil")
    lay = QHBoxLayout(ligne)
    lay.setContentsMargins(0, 2, 0, 0)
    lay.setSpacing(8)
    titre = QLabel("Tester un seuil")
    titre.setObjectName("FicheTexte")
    spin = NoWheelDoubleSpinBox()
    spin.setRange(0.0, 1.0)
    spin.setSingleStep(0.05)
    spin.setDecimals(2)
    spin.setFixedWidth(64)
    spin.setValue(depart)
    spin.setToolTip("Déplace la ligne du seuil sur la figure — sans changer le seuil du traitement")
    retour = QPushButton("↺")
    retour.setObjectName("EntityResetBtn")
    retour.setFlat(True)
    retour.setCursor(Qt.CursorShape.PointingHandCursor)
    retour.setToolTip(f"Revenir au seuil {_v(round(depart, 3))}")
    mesure = QLabel("")
    mesure.setObjectName("FicheTexte")
    # Largeur fixe : le texte ne fait pas glisser la note quand les chiffres changent.
    mesure.setMinimumWidth(QFontMetrics(mesure.font()).horizontalAdvance("précision 100 % · rappel 100 %") + 6)
    note = QLabel("essai sans effet sur le seuil du traitement")
    note.setObjectName("FicheLegende")

    def _appliquer(v: float) -> None:
        figure.set_seuil(v)
        for f in autres:
            f.set_seuil(v)
        mesure.setText(figure.phrase_precision_rappel())
        retour.setEnabled(abs(v - depart) > 1e-9)

    spin.valueChanged.connect(_appliquer)
    retour.clicked.connect(lambda *_: spin.setValue(depart))
    _appliquer(depart)
    for w in (titre, spin, retour, mesure):
        lay.addWidget(w)
    lay.addStretch(1)
    lay.addWidget(note)
    return ligne
