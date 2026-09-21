"""Fiche d'un produit de l'étape 2 — ce que l'archéologue lit avant de cocher.

Pendant, côté produits, de :mod:`class_info_dialog` : mêmes helpers de mise en
page, même gabarit (liste à gauche, fiche à droite), même visuel d'accès. Elle
répond à quatre questions que l'étape 2 ne savait pas poser :

1. **Qu'est-ce que je vais voir ?** — vignette du produit et lecture de l'image ;
2. **À quoi ça sert ?** — usage en prospection ;
3. **Qu'est-ce que ça ne montre pas ?** — les angles morts, dits franchement ;
4. **D'où ça sort ?** — méthode, réglages de l'étape 2 et sources.

Tous les produits sont toujours chargés ensemble : ouvrir la fiche du Sky-View
Factor et feuilleter jusqu'au Local Dominance est précisément la façon dont on
choisit entre deux indices.

Les données viennent du module PUR :mod:`app.services.indice_fiche` ; ce module
ne fait que des widgets Qt. Compatible Qt5/Qt6 : énumérés scopés.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QPixmap
from qgis.PyQt.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...app.services.indice_fiche import Comparaison, IndiceFiche
from ..widgets.vignette import pixmap_ajuste
# Helpers de mise en page de la fiche de classe : même gabarit, donc même code.
# Les extraire dans un module tiers ne ferait qu'ajouter un fichier pour quatre
# fabriques de QLabel.
from .class_info_dialog import _label, _puces, _separateur, _titre_bloc

_VIGNETTE_MAX = 320  # côté max de l'aperçu, en px logiques

#: Nom d'objet QSS par verdict. La case affiche le symbole de la source
#: (``-`` / ``o`` / ``+`` / ``++``) et non plus une pastille : c'est la
#: couleur qui porte la lecture en diagonale. Les formes étoilées se
#: colorent comme leur forme simple, l'astérisque suffit à dire la nuance.
_STYLE_VERDICT = {
    "-": "VerdictNon", "-*": "VerdictNon",
    "o": "VerdictPartiel",
    "+": "VerdictOui", "+*": "VerdictOui",
    "++": "VerdictFort",
}


# ----------------------------------------------------------------------
# Aperçu : la vignette du produit, navigable s'il y en a plusieurs
# ----------------------------------------------------------------------
class _Apercu(QWidget):
    """Visionneuse des vignettes d'un produit.

    Les chemins du JSON sont relatifs à ``data/`` : c'est ``racine`` qui les
    résout. Un fichier absent (vignette pas encore produite, plugin installé à
    la main) affiche un cadre d'attente, jamais une exception.
    """

    def __init__(self, fiche: IndiceFiche, racine: Optional[Path], parent=None):
        super().__init__(parent)
        self._fiche = fiche
        self._racine = Path(racine) if racine else None
        self._i = 0

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self._image = QLabel()
        self._image.setObjectName("FicheImage")
        self._image.setFixedSize(_VIGNETTE_MAX, _VIGNETTE_MAX)
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image.setWordWrap(True)
        lay.addWidget(self._image)

        barre = QHBoxLayout()
        barre.setSpacing(0)
        barre.addStretch(1)
        self._prev = QPushButton("‹")
        self._compteur = _label("", "FicheCompteur", wrap=False)
        self._next = QPushButton("›")
        for b, pas in ((self._prev, -1), (self._next, 1)):
            b.setObjectName("FicheNav")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setFixedWidth(24)
            b.clicked.connect(lambda _c=False, p=pas: self._decale(p))
        barre.addWidget(self._prev)
        barre.addWidget(self._compteur)
        barre.addWidget(self._next)
        lay.addLayout(barre)

        self._legende = _label("", "FicheLegende")
        lay.addWidget(self._legende)
        lay.addStretch(1)

        self._refresh()

    def _decale(self, pas: int) -> None:
        n = len(self._fiche.vignettes)
        if n:
            self._i = (self._i + pas) % n
        self._refresh()

    def _chemin(self) -> Optional[Path]:
        vs = self._fiche.vignettes
        if not vs or self._racine is None:
            return None
        p = self._racine / vs[self._i].image
        return p if p.is_file() else None

    def _refresh(self) -> None:
        vs = self._fiche.vignettes
        n = len(vs)
        v = vs[self._i] if n else None
        for w in (self._prev, self._next, self._compteur):
            w.setVisible(n > 1)
        self._compteur.setText(f"{self._i + 1} / {n}" if n > 1 else "")

        chemin = self._chemin()
        if chemin is None:
            self._image.setPixmap(QPixmap())
            self._image.setProperty("state", "vide")
            self._image.setText(
                "Illustration à produire\npour ce produit" if not n
                else "Vignette introuvable\ndans le dossier data/"
            )
        else:
            # −2 px : le cadre du QSS prend 1 px de chaque côté, viser la
            # taille du widget ferait rogner l'image d'autant.
            pix = pixmap_ajuste(
                str(chemin), _VIGNETTE_MAX - 2, dpr=self.devicePixelRatioF()
            )
            self._image.setProperty("state", "plein")
            if pix.isNull():
                self._image.setText("Vignette illisible")
            else:
                self._image.setText("")
                self._image.setPixmap(pix)
        self._image.style().unpolish(self._image)
        self._image.style().polish(self._image)

        # Sous l'image : d'où vient le cadre, et sous quelle licence s'il n'est
        # pas de nous. La légende du JSON est la phrase de lecture.
        lignes = [t for t in (v.legende if v else "", v.source if v else "") if t]
        if v and v.licence:
            lignes.append(v.licence)
        self._legende.setText(" · ".join(lignes))


# ----------------------------------------------------------------------
# Le corps d'une fiche
# ----------------------------------------------------------------------
class _CorpsFiche(QWidget):
    def __init__(self, fiche: IndiceFiche, racine: Optional[Path], parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 18)
        lay.setSpacing(12)

        # — en-tête : titre métier, puis l'identité technique en second —
        titre = QHBoxLayout()
        titre.setSpacing(10)
        titre.addWidget(_label(fiche.titre, "FicheTitre", wrap=False))
        titre.addWidget(_label(f"{fiche.tag} · {fiche.nom}", "FicheId", wrap=False))
        titre.addStretch(1)
        lay.addLayout(titre)

        if fiche.resume:
            lay.addWidget(_label(fiche.resume, "FicheResume"))

        # — aperçu + lecture de l'image côte à côte —
        haut = QHBoxLayout()
        haut.setSpacing(18)
        haut.addWidget(_Apercu(fiche, racine))
        colonne = QVBoxLayout()
        colonne.setSpacing(8)
        if fiche.lire:
            colonne.addWidget(_titre_bloc("Ce que montre l'image"))
            colonne.addWidget(_label(fiche.lire, "FicheTexte"))
        colonne.addStretch(1)
        haut.addLayout(colonne, 1)
        lay.addLayout(haut)

        # — blocs textuels —
        for titre_bloc, contenu in self._blocs(fiche):
            lay.addWidget(_separateur())
            lay.addWidget(_titre_bloc(titre_bloc))
            lay.addWidget(contenu)

        lay.addStretch(1)

    # -- fabrication des blocs ----------------------------------------
    @staticmethod
    def _parametres(fiche: IndiceFiche) -> QLabel:
        """Réglage, défaut et effet, un par ligne.

        Le défaut est donné avec son unité de terrain quand elle existe
        (« 10 px, soit 5 m ») : c'est le piège numéro un des indices RVT, dont
        les rayons sont en pixels et changent donc de portée avec la résolution
        du modèle d'altitude.
        """
        lignes: List[str] = []
        for p in fiche.parametres:
            tete = p.label or p.cle
            if p.defaut:
                tete = f"{tete} — {p.defaut}"
            lignes.append(f"•  {tete}")
            if p.sens:
                lignes.append(f"     {p.sens}")
        return _label("\n".join(lignes), "FicheTexte")

    @staticmethod
    def _references(fiche: IndiceFiche) -> QLabel:
        """Sources cliquables : une citation par puce, liée à son DOI/URL."""
        lignes = []
        for r in fiche.references:
            texte = r.citation.replace("&", "&amp;").replace("<", "&lt;")
            if r.url:
                lignes.append(f'•&nbsp;&nbsp;<a href="{r.url}">{texte}</a>')
            else:
                lignes.append(f"•&nbsp;&nbsp;{texte}")
        # Les citations des fiches sont traduites de l'anglais : le dire une
        # fois ici plutôt que d'accoler « (traduit) » à chacune d'elles.
        lignes.append(
            "<i>Les citations sont traduites de l'anglais par le plugin ; "
            "les noms d'algorithmes et de réglages restent en version originale.</i>"
        )
        lab = _label("<br>".join(lignes), "FicheTexte")
        lab.setTextFormat(Qt.TextFormat.RichText)
        lab.setOpenExternalLinks(True)
        return lab

    def _blocs(self, f: IndiceFiche):
        out = []
        if f.usage:
            out.append(("Dans quelle optique l'utiliser", _puces(f.usage)))
        if f.limites:
            # Même traitement visuel que le « Ne détecte pas » d'une classe :
            # c'est le bloc qu'on lit pour ne pas se tromper.
            out.append(("Ce que ce produit ne montre pas", _puces(f.limites, "FicheHorsCible")))
        if f.methode:
            out.append(("Comment c'est calculé", _puces(f.methode)))
        if f.parametres:
            out.append(("Réglages (étape 2 → Réglages avancés…)", self._parametres(f)))
        if f.references:
            out.append(("Sources", self._references(f)))
        if not f.est_complete:
            out.append((
                "Fiche incomplète",
                _label(
                    "Blocs à écrire pour ce produit : "
                    + ", ".join(f.manques)
                    + " (data/indices_fiches.json).",
                    "FicheManque",
                ),
            ))
        return out


# ----------------------------------------------------------------------
# Comparer les produits entre eux
# ----------------------------------------------------------------------
class _CorpsComparaison(QWidget):
    """Les tableaux comparatifs.

    Une fiche répond « que montre ce produit » ; ces tableaux répondent
    « lequel prendre », qui est la question qu'on se pose devant la liste
    cases à cocher de l'étape 2 et qu'aucune fiche prise seule ne résout.
    """

    def __init__(self, comparaison: Comparaison, fiches: Sequence[IndiceFiche], parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 18)
        lay.setSpacing(12)

        lay.addWidget(_label("Comparer les produits", "FicheTitre", wrap=False))
        if comparaison.note:
            lay.addWidget(_label(comparaison.note, "FicheResume"))
        if comparaison.legende:
            lay.addWidget(_label(
                "     ".join(f"{k}  {v}" for k, v in comparaison.legende),
                "FicheLegende",
            ))

        for tableau in comparaison.tableaux:
            lay.addWidget(_separateur())
            lay.addWidget(_titre_bloc(tableau.titre))
            lay.addWidget(self._grille(tableau, fiches))
            if tableau.source:
                lay.addWidget(_label(tableau.source, "FicheLegende"))
        lay.addStretch(1)

    @staticmethod
    def _grille(tableau, fiches) -> QWidget:
        hote = QWidget()
        g = QGridLayout(hote)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(3)

        for j, col in enumerate(tableau.colonnes):
            tete = _label(col.libelle, "FicheColonne")
            tete.setAlignment(
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom
            )
            tete.setFixedWidth(76)
            if col.aide:
                tete.setToolTip(col.aide)
            g.addWidget(tete, 0, j + 1)

        for i, fiche in enumerate(fiches, start=1):
            nom = _label(f"{fiche.tag}   {fiche.titre}", "FicheTexte", wrap=False)
            nom.setToolTip(fiche.nom)
            g.addWidget(nom, i, 0)
            for j, col in enumerate(tableau.colonnes):
                verdict = tableau.verdict(fiche.cle, col.cle)
                # Case vide = produit non évalué par la source, ou colonne
                # sans objet : un point, jamais un « non » qu'on n'a pas lu.
                case = _label(verdict or "\u00b7", wrap=False)
                case.setObjectName(_STYLE_VERDICT.get(verdict, "VerdictVide"))
                case.setAlignment(Qt.AlignmentFlag.AlignCenter)
                # L'aide de la colonne est le seul endroit qui donne le sens
                # du symbole : il change d'une colonne à l'autre (``++`` vaut
                # « excellent » partout, sauf en Complexité).
                case.setToolTip(
                    f"{fiche.titre} \u2014 {col.libelle} : {verdict}\n{col.aide}"
                    if verdict else
                    f"{fiche.titre} \u2014 {col.libelle} : "
                    "non évalué par la source"
                )
                g.addWidget(case, i, j + 1)

        g.setColumnStretch(0, 1)
        return hote


# ----------------------------------------------------------------------
# Dialog
# ----------------------------------------------------------------------
class IndiceInfoDialog(QDialog):
    """Les fiches des produits de l'étape 2, feuilletables."""

    def __init__(
        self,
        fiches: Sequence[IndiceFiche],
        racine: Optional[Path] = None,
        cle_active: str = "",
        parent=None,
        comparaison: Optional[Comparaison] = None,
    ):
        super().__init__(parent)
        self._fiches = list(fiches)
        self._racine = racine
        # La comparaison occupe la première ligne de la liste quand elle
        # existe ; les fiches sont alors décalées d'un rang (cf. _decalage).
        self._comparaison = (
            comparaison if (comparaison and not comparaison.est_vide) else None
        )
        self.setObjectName("IndiceInfoDialog")
        self.setWindowTitle("Produits de visualisation")
        self.setMinimumSize(760, 560)
        self.resize(940, 660)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        corps = QHBoxLayout()
        corps.setContentsMargins(0, 0, 0, 0)
        corps.setSpacing(0)

        # Liste de gauche : tous les produits, pour comparer avant de cocher.
        self._liste = QListWidget()
        self._liste.setObjectName("FicheListe")
        self._liste.setFixedWidth(210)
        if self._comparaison is not None:
            entree = QListWidgetItem("\u229e  Comparer les produits")
            entree.setToolTip(
                "Quel produit pour quelle forme, et ce que chacun sait faire"
            )
            self._liste.addItem(entree)
        for f in self._fiches:
            item = QListWidgetItem(f"{f.tag} · {f.titre}")
            item.setToolTip(f"{f.nom} — {f.famille}")
            self._liste.addItem(item)
        self._liste.currentRowChanged.connect(self._afficher)
        corps.addWidget(self._liste)

        self._zone = QScrollArea()
        self._zone.setObjectName("FicheScroll")
        self._zone.setWidgetResizable(True)
        self._zone.setFrameShape(QFrame.Shape.NoFrame)
        self._zone.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        corps.addWidget(self._zone, 1)
        root.addLayout(corps, 1)

        pied = QHBoxLayout()
        pied.setContentsMargins(14, 10, 14, 12)
        pied.addStretch(1)
        fermer = QPushButton("Fermer")
        fermer.setCursor(Qt.CursorShape.PointingHandCursor)
        fermer.clicked.connect(self.accept)
        pied.addWidget(fermer)
        root.addLayout(pied)

        if self._fiches:
            depart = next(
                (i for i, f in enumerate(self._fiches) if f.cle == cle_active), 0
            )
            self._liste.setCurrentRow(depart + self._decalage)
            self._afficher(depart + self._decalage)
        else:
            self._zone.setWidget(_label("Aucune fiche disponible.", "FicheManque"))

    @property
    def _decalage(self) -> int:
        """Nombre de lignes qui précèdent les fiches dans la liste."""
        return 1 if self._comparaison is not None else 0

    def _afficher(self, row: int) -> None:
        if self._comparaison is not None and row == 0:
            self._zone.setWidget(
                _CorpsComparaison(self._comparaison, self._fiches)
            )
            return
        i = row - self._decalage
        if not (0 <= i < len(self._fiches)):
            return
        self._zone.setWidget(_CorpsFiche(self._fiches[i], self._racine))


def ouvrir_fiche_indice(
    fiches: Sequence[IndiceFiche],
    racine: Optional[Path],
    cle_active: str = "",
    parent=None,
    comparaison: Optional[Comparaison] = None,
) -> None:
    """Ouvre les fiches en modal, positionnées sur ``cle_active``.

    ``comparaison`` ajoute les tableaux comparatifs en tête de liste ;
    absente ou vide, l'entrée n'apparaît simplement pas.
    """
    if not fiches:
        return
    IndiceInfoDialog(
        fiches, racine, cle_active, parent=parent, comparaison=comparaison
    ).exec()
