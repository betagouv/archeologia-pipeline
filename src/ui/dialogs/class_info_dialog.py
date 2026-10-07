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
from typing import Callable, List, Optional, Sequence

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QPixmap
from qgis.PyQt.QtWidgets import (
    QGridLayout,
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
from ..widgets.vignette import FicheButton, pixmap_ajuste
from ...app.services.profil_scores import libelle_zone, zones_sans_objet
from ..widgets.profil_scores import figure_profil, figures_par_zone, ligne_essai_seuil

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


def _lignes_liste(titre: str, items: Sequence[str]) -> List[str]:
    """Un seul élément : « Titre : texte » (forme d'origine des fiches) ;
    plusieurs : une puce par élément, sans titre — la liste se lit seule."""
    if len(items) == 1:
        return [f"{titre} : {items[0]}"]
    return [f"•  {t}" for t in items]


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
        # Repolish : la propriété dynamique pilote le style du cadre.
        self._image.style().unpolish(self._image)
        self._image.style().polish(self._image)

        # Sous l'image, seulement le LIEU (décision utilisateur 2026-09-15) : la
        # légende du model_card est une note interne de relecture, pas un texte
        # d'interface.
        self._legende.setText(v.zone if v else "")


# ----------------------------------------------------------------------
# Le corps d'une fiche
# ----------------------------------------------------------------------
class _CorpsFiche(QWidget):
    def __init__(self, fiche: ClassFiche, model_dir: Optional[Path], parent=None,
                 ouvrir_modele: Optional[Callable[[], None]] = None,
                 couleur=None, seuil: Optional[float] = None, observe=None):
        """``couleur`` : couleur de base de la COUCHE (qualifiée en A/B), sinon celle
        du registre pour la classe ; ``seuil`` : seuil effectif réglé à l'étape 3,
        pour que la figure et son bilan montrent ce que le run appliquera."""
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
        if ouvrir_modele is not None:
            lien = FicheButton(f"Architecture, entraînement et métriques d'évaluation de « {fiche.modele} »")
            lien.setText("Fiche du modèle")
            lien.clicked.connect(lambda *_: ouvrir_modele())
            colonne.addWidget(lien, 0, Qt.AlignmentFlag.AlignLeft)
        if fiche.fiabilite:
            colonne.addWidget(_titre_bloc("Fiabilité mesurée au banc"))
            colonne.addWidget(_label(self._fiabilite(fiche, observe), "FicheTexte"))
            note = self._note_observation(observe)
            if note:
                colonne.addWidget(_label(note, "FicheLegende"))
        colonne.addStretch(1)
        haut.addLayout(colonne, 1)
        lay.addLayout(haut)

        # — profil des scores : la figure derrière les niveaux de fiabilité —
        figure = (figure_profil(model_dir, fiche.nom, fiche.fiabilite, couleur=couleur)
                  if fiche.fiabilite else None)
        if figure is not None:
            lay.addWidget(_separateur())
            lay.addWidget(_titre_bloc("Profil des scores à l'évaluation"))
            lay.addWidget(figure)
            legende = (
                "Vraies détections en couleur, fausses en gris, par bande de score de 0,05. "
                "Les niveaux commencent là où la part de vrais objets atteint 35, 60 et 85 %. "
                "La ligne pointillée est le point d'équilibre entre précision et rappel (F1) ; "
                "le seuil déployé est choisi en dessous. La case « Tester un seuil » est un essai, "
                "sans effet sur le seuil du traitement."
            )
            if seuil is not None:
                figure.set_seuil(seuil)
                if abs(seuil - figure.profil.seuil) > 1e-9:
                    legende += (f" Seuil réglé à {seuil:.2f} (modèle : "
                                f"{figure.profil.seuil:g}).").replace(".", ",")
            lay.addWidget(_label(legende, "FicheLegende"))
            # Petits multiples par zone d'évaluation : une classe sûre ici et
            # faible là se voit d'un coup d'œil (les linéaires surtout).
            zones = figures_par_zone(model_dir, fiche.nom, fiche.fiabilite, couleur=couleur)
            # « Tester un seuil » : la figure (et les zones) suivent, précision et
            # rappel au banc s'affichent — sans toucher au seuil du traitement.
            lay.addWidget(ligne_essai_seuil(figure, [f for _n, f in zones], seuil))
            if zones:
                lay.addWidget(_titre_bloc("Par zone d'évaluation"))
                exclues = zones_sans_objet(model_dir, fiche.nom) if model_dir is not None else []
                if exclues:
                    noms = ", ".join(libelle_zone(z) for z in exclues)
                    lay.addWidget(_label(
                        f"{len(exclues)} zone{'s' if len(exclues) > 1 else ''} sans objet annoté de cette classe "
                        f"({noms}) : on n'y compte que des fausses détections, rien à mesurer — non affichée"
                        f"{'s' if len(exclues) > 1 else ''}.",
                        "FicheLegende",
                    ))
                grille = QGridLayout()
                grille.setHorizontalSpacing(14)
                grille.setVerticalSpacing(6)
                for i, (nom_zone, fig) in enumerate(zones):
                    if seuil is not None:
                        fig.set_seuil(seuil)
                    cellule = QVBoxLayout()
                    cellule.setSpacing(1)
                    cellule.addWidget(_label(
                        f"{nom_zone} — {fig.profil.total:,} détections".replace(",", " "),
                        "FicheLegende",
                    ))
                    cellule.addWidget(fig)
                    grille.addLayout(cellule, i // 2, i % 2)
                lay.addLayout(grille)

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
            lignes.append(f"Seuil de confiance déployé : {f.seuil:g}".replace(".", ","))
        if f.modele:
            lignes.append(f"Modèle : {f.modele}")
        if f.statut:
            lignes.append(f"Statut : {f.statut}")
        return "\n".join(lignes)

    @staticmethod
    def _fiabilite(f: ClassFiche, observe=None) -> str:
        """Une ligne par niveau : la mesure du banc, puis « chez vous : … » dès
        qu'un verdict a été saisi à ce niveau (fiabilité observée, 2026-10-08)."""
        comptes = getattr(observe, "par_categorie", None) or {}
        lignes = []
        for c in f.fiabilite:
            mesure = pct(c.mesure)
            suffixe = f" — {mesure} % de vrais objets mesurés sur {c.n}" if mesure is not None else ""
            ligne = f"{c.label} : score ≥ {c.seuil:g}{suffixe}".replace(".", ",")
            compte = comptes.get(c.categorie)
            if compte is not None and compte.phrase():
                ligne += f" · chez vous : {compte.phrase()}"
            lignes.append(ligne)
        return "\n".join(lignes)

    @staticmethod
    def _note_observation(observe) -> str:
        if observe is None or not getattr(observe, "n_runs", 0):
            return ""
        n = observe.total_verifies
        if not n and not observe.total_a_revoir:
            return (f"Aucun verdict pour cette classe dans vos {observe.n_runs} run(s) connus : "
                    "renseignez le champ « validation » (oui / non / peut-être) dans QGIS.")
        return (f"« Chez vous » = vos verdicts (champ « validation » : oui / non / peut-être) sur "
                f"{observe.n_runs} run(s) connus, {n} vérification(s). Le banc est un plancher annoncé ; "
                "le terrain dit ce qu'il vaut ici.")

    @staticmethod
    def _entrainement(f: ClassFiche) -> str:
        e = f.entrainement
        if e is None:
            return ""
        lignes: List[str] = []
        if e.corpus:
            lignes.extend(_lignes_liste("Corpus", e.corpus))
        if e.annotation:
            lignes.extend(_lignes_liste("Annotation", e.annotation))
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
            # Un texte seul (forme d'origine) reste un paragraphe ; plusieurs
            # éléments = une puce chacun, comme « Ne détecte pas ».
            contenu = _puces(f.usage) if len(f.usage) > 1 else _label(f.usage[0], "FicheTexte")
            out.append(("Dans quelle optique l'utiliser", contenu))
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
        models: Optional[dict] = None,
        couleurs: Optional[dict] = None,
        seuils: Optional[dict] = None,
        observes: Optional[dict] = None,
    ):
        super().__init__(parent)
        self._fiches = list(fiches)
        self._dirs = dict(model_dirs or {})
        self._models = dict(models or {})  # nom → InstalledModel, pour « Fiche du modèle »
        self._couleurs = dict(couleurs or {})  # (modèle, classe) → couleur de la couche
        self._seuils = dict(seuils or {})      # (modèle, classe) → seuil effectif (étape 3)
        self._observes = dict(observes or {})  # (modèle, classe) → Observation (vos verdicts)
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
        model = self._models.get(f.modele_id)
        ouvrir = (lambda: self._ouvrir_modele(model)) if model is not None else None
        self._zone.setWidget(_CorpsFiche(
            f, self._dirs.get(f.modele_id), ouvrir_modele=ouvrir,
            couleur=self._couleurs.get((f.modele_id, f.nom)),
            seuil=self._seuils.get((f.modele_id, f.nom)),
            observe=self._observes.get((f.modele_id, f.nom)),
        ))

    def _ouvrir_modele(self, model) -> None:
        """Fiche ⓘ du modèle de la classe affichée, par-dessus celle-ci."""
        from .model_info_dialog import ModelInfoDialog  # différé, comme à l'étape 3
        ModelInfoDialog(model, parent=self).exec()


def ouvrir_fiche_entite(
    fiches: Sequence[ClassFiche],
    model_dirs: Optional[dict],
    titre: str,
    parent=None,
    models: Optional[dict] = None,
    couleurs: Optional[dict] = None,
    seuils: Optional[dict] = None,
    observes: Optional[dict] = None,
) -> None:
    """Ouvre la fiche en modal. Rien à afficher → rien ne s'ouvre. ``models``
    (nom → ``InstalledModel``) active le lien « Fiche du modèle » ; ``couleurs`` et
    ``seuils`` (clé ``(modèle, classe)``) alignent la figure sur la couche et le
    seuil du run."""
    if not fiches:
        return
    ClassInfoDialog(
        fiches, model_dirs, titre, parent=parent, models=models,
        couleurs=couleurs, seuils=seuils, observes=observes,
    ).exec()
