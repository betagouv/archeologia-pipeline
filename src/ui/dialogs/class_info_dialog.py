"""Fiche d'une structure détectable — ce que l'archéologue lit avant de cocher.

Répond à trois questions que l'étape 3 ne savait pas poser :

1. **À quoi ça ressemble ?** — une vignette RVT issue du corpus d'entraînement,
   avec bascule « relief seul » / « vérité terrain » ;
2. **Où et en quelle quantité le modèle l'a-t-il apprise ?** — corpus, zones
   nommées, tuiles et objets annotés par zone et par split ;
3. **Dans quelle optique s'en servir ?** — contexte de prospection, hors-cible
   explicite, fiabilité mesurée au banc, limites connues.

Les données viennent du bloc ``classes[].fiche`` du ``model_card.yaml``, lu par
le module PUR :mod:`app.services.class_fiche` ; ce module ne fait que des
widgets Qt. Une entité couverte par plusieurs classes (cible dérivée, ou
comparaison A/B) affiche une fiche par classe, sélectionnable à gauche.

Compatible Qt5/Qt6 : tous les énumérés sont scopés (``Qt.AlignmentFlag…``).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QPixmap
from qgis.PyQt.QtWidgets import (
    QDialog,
    QFrame,
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

from ...app.services.class_fiche import ClassFiche
from ...app.services.fiabilite import pct

_VIGNETTE_MAX = 320  # côté max de l'aperçu, en px logiques


# ----------------------------------------------------------------------
# Petits helpers de mise en page
# ----------------------------------------------------------------------
def _label(text: str, obj: str = "", *, wrap: bool = True) -> QLabel:
    lab = QLabel(text)
    if obj:
        lab.setObjectName(obj)
    lab.setWordWrap(wrap)
    lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
    return lab


def _titre_bloc(text: str) -> QLabel:
    return _label(text.upper(), "FicheBlocTitre", wrap=False)


def _puces(items: Sequence[str], obj: str = "FicheTexte") -> QLabel:
    """Liste à puces en un seul QLabel : moins de widgets, wrap correct."""
    return _label("\n".join(f"•  {t}" for t in items), obj)


def _nb(n: int) -> str:
    """Entier à la française : espace insécable fine tous les trois chiffres.

    ``format(n, ",")`` puis substitution — ``:n`` dépendrait de la locale du
    poste, qui n'est pas garantie française sous QGIS.
    """
    return f"{n:,}".replace(",", " ")


def _separateur() -> QFrame:
    line = QFrame()
    line.setObjectName("FicheSep")
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Plain)
    return line


# ----------------------------------------------------------------------
# Aperçu : une vignette, bascule relief / vérité terrain, navigation
# ----------------------------------------------------------------------
class _Apercu(QWidget):
    """Visionneuse des vignettes d'une fiche.

    Les chemins du ``model_card`` sont relatifs au dossier du modèle : c'est
    ``model_dir`` qui les résout. Un fichier absent (modèle installé à la main,
    vignette oubliée au packaging) affiche un cadre d'attente, jamais une
    exception ni une image cassée.
    """

    def __init__(self, fiche: ClassFiche, model_dir: Optional[Path], parent=None):
        super().__init__(parent)
        self._fiche = fiche
        self._dir = Path(model_dir) if model_dir else None
        self._i = 0
        self._annote = bool(fiche.vignettes and fiche.vignettes[0].annote)

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
        self._btn_relief = QPushButton("Relief seul")
        self._btn_verite = QPushButton("Vérité terrain")
        for b, annote in ((self._btn_relief, False), (self._btn_verite, True)):
            b.setObjectName("FicheToggle")
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _c=False, a=annote: self._set_annote(a))
            barre.addWidget(b)
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

    # -- état ----------------------------------------------------------
    def _set_annote(self, annote: bool) -> None:
        self._annote = annote
        self._refresh()

    def _decale(self, pas: int) -> None:
        n = len(self._fiche.vignettes)
        if n:
            self._i = (self._i + pas) % n
        self._refresh()

    # -- rendu ---------------------------------------------------------
    def _chemin(self) -> Optional[Path]:
        vs = self._fiche.vignettes
        if not vs or self._dir is None:
            return None
        v = vs[self._i]
        rel = v.annote if (self._annote and v.annote) else v.brut
        p = self._dir / rel
        return p if p.is_file() else None

    def _refresh(self) -> None:
        vs = self._fiche.vignettes
        n = len(vs)
        v = vs[self._i] if n else None
        a_verite = bool(v and v.annote)

        self._btn_relief.setChecked(not self._annote)
        self._btn_verite.setChecked(self._annote)
        self._btn_verite.setEnabled(a_verite)
        self._btn_relief.setEnabled(n > 0)
        for b in (self._prev, self._next, self._compteur):
            b.setVisible(n > 1)
        self._compteur.setText(f"{self._i + 1} / {n}" if n > 1 else "")

        chemin = self._chemin()
        if chemin is None:
            self._image.setPixmap(QPixmap())
            self._image.setProperty("state", "vide")
            self._image.setText(
                "Illustration à produire\npour cette classe"
                if not n else "Vignette introuvable\ndans le dossier du modèle"
            )
        else:
            pix = QPixmap(str(chemin))
            self._image.setProperty("state", "plein")
            if pix.isNull():
                self._image.setText("Vignette illisible")
            else:
                self._image.setText("")
                self._image.setPixmap(pix.scaled(
                    _VIGNETTE_MAX, _VIGNETTE_MAX,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))
        # Repolish : la propriété dynamique pilote le style du cadre.
        self._image.style().unpolish(self._image)
        self._image.style().polish(self._image)

        morceaux = [m for m in (v.zone if v else "", v.legende if v else "") if m]
        self._legende.setText(" — ".join(morceaux))


# ----------------------------------------------------------------------
# Le corps d'une fiche
# ----------------------------------------------------------------------
class _CorpsFiche(QWidget):
    def __init__(self, fiche: ClassFiche, model_dir: Optional[Path], parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 18)
        lay.setSpacing(12)

        # — en-tête —
        titre = QHBoxLayout()
        titre.setSpacing(10)
        titre.addWidget(_label(fiche.label, "FicheTitre", wrap=False))
        titre.addWidget(_label(fiche.nom, "FicheId", wrap=False))
        titre.addStretch(1)
        lay.addLayout(titre)

        if fiche.resume:
            lay.addWidget(_label(fiche.resume, "FicheResume"))

        # — aperçu + fiabilité côte à côte —
        haut = QHBoxLayout()
        haut.setSpacing(18)
        haut.addWidget(_Apercu(fiche, model_dir))
        colonne = QVBoxLayout()
        colonne.setSpacing(8)
        if fiche.reconnaitre:
            colonne.addWidget(_titre_bloc("Reconnaître"))
            colonne.addWidget(_label(fiche.reconnaitre, "FicheTexte"))
        colonne.addWidget(_titre_bloc("Contexte technique"))
        colonne.addWidget(_label(self._contexte(fiche), "FicheTexte"))
        if fiche.fiabilite:
            colonne.addWidget(_titre_bloc("Fiabilité mesurée au banc"))
            colonne.addWidget(_label(self._fiabilite(fiche), "FicheTexte"))
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
    def _contexte(f: ClassFiche) -> str:
        lignes = []
        if f.rvt_label:
            res = f" à {f.resolution_m:g} m".replace(".", ",") if f.resolution_m else ""
            lignes.append(f"Indice {f.rvt_label}{res}")
        if f.task_label:
            lignes.append(f"Sortie : {f.task_label.lower()}")
        if f.seuil is not None:
            lignes.append(f"Seuil déployé : {f.seuil:g}".replace(".", ","))
        if f.modele:
            lignes.append(f"Modèle : {f.modele}")
        if f.statut:
            lignes.append(f"Statut : {f.statut}")
        return "\n".join(lignes)

    @staticmethod
    def _fiabilite(f: ClassFiche) -> str:
        lignes = []
        for c in f.fiabilite:
            mesure = pct(c.mesure)
            suffixe = f" — {mesure} % de vrais objets mesurés sur {c.n}" if mesure is not None else ""
            lignes.append(f"{c.categorie.replace('_', ' ')} : score ≥ {c.seuil:g}{suffixe}".replace(".", ","))
        return "\n".join(lignes)

    @staticmethod
    def _entrainement(f: ClassFiche) -> str:
        e = f.entrainement
        if e is None:
            return ""
        lignes: List[str] = []
        if e.corpus:
            lignes.append(f"Corpus : {e.corpus}")
        if e.annotation:
            lignes.append(f"Annotation : {e.annotation}")
        if e.zones:
            lignes.append("")
            lignes.append("Zones d'apprentissage :")
            for z in e.zones:
                chiffres = []
                if z.tuiles:
                    chiffres.append(f"{_nb(z.tuiles)} tuiles")
                if z.objets:
                    chiffres.append(f"{_nb(z.objets)} objets")
                détail = f" — {', '.join(chiffres)}" if chiffres else ""
                lignes.append(f"•  {z.nom}{détail}")
        if e.splits:
            lignes.append("")
            parts = [
                f"{s.nom} {_nb(s.tuiles)} tuiles / {_nb(s.objets)} objets"
                for s in e.splits
            ]
            lignes.append("Répartition : " + "  ·  ".join(parts))
        if e.total_objets:
            lignes.append(
                f"Total : {_nb(e.total_objets)} objets annotés sur "
                f"{_nb(e.total_tuiles)} tuiles"
            )
        return "\n".join(lignes)

    def _blocs(self, f: ClassFiche):
        out = []
        txt = self._entrainement(f)
        if txt:
            out.append(("Ce que le modèle a appris", _label(txt, "FicheTexte")))
        elif not f.est_complete:
            out.append((
                "Ce que le modèle a appris",
                _label(
                    "Provenance des données d'entraînement non renseignée pour cette "
                    "classe (bloc fiche.entrainement du model_card.yaml).",
                    "FicheManque",
                ),
            ))
        if f.hors_cible:
            out.append(("Ne détecte pas", _puces(f.hors_cible, "FicheHorsCible")))
        if f.usage:
            out.append(("Dans quelle optique l'utiliser", _label(f.usage, "FicheTexte")))
        if f.limites:
            out.append(("Limites connues du modèle", _puces(f.limites)))
        return out


# ----------------------------------------------------------------------
# Dialog
# ----------------------------------------------------------------------
class ClassInfoDialog(QDialog):
    """Fiche(s) des classes qui portent une entité."""

    def __init__(
        self,
        fiches: Sequence[ClassFiche],
        model_dirs: Optional[dict] = None,
        titre: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._fiches = list(fiches)
        self._dirs = dict(model_dirs or {})
        self.setObjectName("ClassInfoDialog")
        self.setWindowTitle(titre or "Structure détectable")
        self.setMinimumSize(760, 560)
        self.resize(900, 640)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        corps = QHBoxLayout()
        corps.setContentsMargins(0, 0, 0, 0)
        corps.setSpacing(0)

        # Liste de gauche : seulement quand l'entité a plusieurs classes
        self._liste = QListWidget()
        self._liste.setObjectName("FicheListe")
        self._liste.setFixedWidth(200)
        for f in self._fiches:
            item = QListWidgetItem(f.label)
            item.setToolTip(f"{f.nom} — {f.modele}")
            self._liste.addItem(item)
        self._liste.currentRowChanged.connect(self._afficher)
        self._liste.setVisible(len(self._fiches) > 1)
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
            self._liste.setCurrentRow(0)
            self._afficher(0)
        else:
            self._zone.setWidget(_label(
                "Aucune classe installée ne porte cette entité.", "FicheManque"
            ))

    def _afficher(self, row: int) -> None:
        if not (0 <= row < len(self._fiches)):
            return
        f = self._fiches[row]
        self._zone.setWidget(_CorpsFiche(f, self._dirs.get(f.modele_id)))


def ouvrir_fiche_entite(
    fiches: Sequence[ClassFiche],
    model_dirs: Optional[dict],
    titre: str,
    parent=None,
) -> None:
    """Ouvre la fiche en modal. Rien à afficher → rien ne s'ouvre."""
    if not fiches:
        return
    ClassInfoDialog(fiches, model_dirs, titre, parent=parent).exec()
