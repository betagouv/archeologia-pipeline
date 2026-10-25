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
widgets Qt. Une structure couverte par plusieurs classes (cible dérivée, ou
comparaison A/B) affiche une fiche par classe, choisie dans la liste en tête.

Depuis le 2026-10-24 la fiche se lit dans le **panneau latéral** de l'assistant
(:class:`~ui.widgets.fiche_panel.FichePanel`, 420 px) : :func:`corps_fiche_entite`
en fabrique le corps, vignette au-dessus du texte ; :class:`ClassInfoDialog` n'est
plus qu'une enveloppe pour le rendu hors écran (``dev/rendu_hors_ecran.py profil``).

Compatible Qt5/Qt6 : tous les énumérés sont scopés (``Qt.AlignmentFlag…``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QPixmap
from qgis.PyQt.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...app.services.class_fiche import ClassFiche
from ...app.services.fiabilite import pct
from ..icons import colored_icon
from ..widgets.vignette import (
    ARDOISE,
    couleur_texte,
    pastille,
    pixmap_ajuste,
    vignette_annotee_recoloree,
)
from ...app.services.profil_scores import libelle_zone, zones_sans_objet
from ..widgets.profil_scores import couleur_de_classe, figure_profil, figures_par_zone, ligne_essai_seuil
from ...app.services.carte_zones import charger as charger_zones_corpus
from ..widgets.carte_zones import bandeau_appris_sur

_ZONES_CORPUS: Optional[dict] = None


class _LienFiche(QPushButton):
    """Lien texte vers une autre fiche, à l'intérieur d'une fiche."""

    def __init__(self, tooltip: str = "", parent=None):
        super().__init__("", parent)
        self.setObjectName("FicheBtn")
        self.setFlat(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if tooltip:
            self.setToolTip(tooltip)


def _zones_corpus() -> dict:
    """``data/zones_corpus.json`` (emprises des zones, contours), lu une fois."""
    global _ZONES_CORPUS
    if _ZONES_CORPUS is None:
        _ZONES_CORPUS = charger_zones_corpus(Path(__file__).resolve().parents[3])
    return _ZONES_CORPUS

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

    def __init__(self, fiche: ClassFiche, model_dir: Optional[Path], parent=None, couleur=None):
        super().__init__(parent)
        self._fiche = fiche
        self._dir = Path(model_dir) if model_dir else None
        # Couleur de la couche : les contours de la vérité terrain la prennent
        # (au lieu du jaune des fichiers, demande utilisateur 2026-10-08).
        self._couleur = couleur
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
            v = self._fiche.vignettes[self._i]
            source = str(chemin)
            if self._annote and v.annote and v.brut and self._couleur is not None and self._dir is not None:
                source = vignette_annotee_recoloree(str(self._dir / v.brut), str(chemin), self._couleur)
            pix = pixmap_ajuste(
                source, _VIGNETTE_MAX - 2, dpr=self.devicePixelRatioF()
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
        base = couleur or couleur_de_classe(fiche.nom) or (42, 120, 214)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        # Liseré de 4 px dans la couleur de la classe, en tête du corps : la fiche du
        # modèle a le sien, ardoise (option A, 2026-10-08).
        lisere = QFrame()
        lisere.setObjectName("FicheLisere")
        lisere.setFixedHeight(4)
        lisere.setStyleSheet("background: rgb(%d, %d, %d);" % tuple(int(v) for v in base))
        outer.addWidget(lisere)
        corps = QWidget()
        outer.addWidget(corps)
        lay = QVBoxLayout(corps)
        lay.setContentsMargins(18, 14, 18, 18)
        lay.setSpacing(12)

        # — en-tête : étiquette de nature dans la couleur de la classe (celle de sa
        #   couche dans QGIS), puis le titre (option A, 2026-10-08) —
        nature = QHBoxLayout()
        nature.setSpacing(6)
        point = QLabel()
        point.setPixmap(pastille(base, 10, self.devicePixelRatioF()))
        etiquette = QLabel("STRUCTURE DÉTECTABLE")
        etiquette.setObjectName("FicheNature")
        etiquette.setStyleSheet(f"color: {couleur_texte(base)};")   # couleur de la classe : dynamique
        nature.addWidget(point)
        nature.addWidget(etiquette)
        nature.addStretch(1)
        lay.addLayout(nature)
        titre = QHBoxLayout()
        titre.setSpacing(10)
        titre.addWidget(_label(fiche.label, "FicheTitre", wrap=False))
        titre.addWidget(_label(fiche.nom, "FicheId", wrap=False))
        titre.addStretch(1)
        lay.addLayout(titre)

        if fiche.resume:
            lay.addWidget(_label(fiche.resume, "FicheResume"))

        # Ordre de lecture (demande utilisateur 2026-10-08), celui de quelqu'un qui
        # hésite à cocher l'entité : ce que c'est (vignette, reconnaître) → est-ce
        # que je la veux (dans quelle optique, ne détecte pas) → sur quoi elle a été
        # apprise → ce qu'elle vaut (fiabilité : texte, figure, essai de seuil,
        # zones, d'un seul tenant) → limites → contexte technique et fiche du modèle.
        blocs = dict(self._blocs(fiche, base))

        # — aperçu, puis reconnaître : empilés, la fiche se lit dans un panneau de 420 px —
        haut = QVBoxLayout()
        haut.setSpacing(10)
        haut.addWidget(_Apercu(fiche, model_dir, couleur=base))   # pleine largeur : la légende se replie
        colonne = QVBoxLayout()
        colonne.setSpacing(8)
        if fiche.reconnaitre:
            colonne.addWidget(_titre_bloc("Reconnaître"))
            colonne.addWidget(_label(fiche.reconnaitre, "FicheTexte"))
        # « Dans quelle optique » et « Ne détecte pas » occupent la colonne à côté de
        # la vignette : c'est la suite de la lecture, et la colonne ne reste pas vide.
        for titre_bloc in ("Dans quelle optique l'utiliser", "Ne détecte pas"):
            contenu = blocs.pop(titre_bloc, None)
            if contenu is not None:
                colonne.addWidget(_titre_bloc(titre_bloc))
                colonne.addWidget(contenu)
        haut.addLayout(colonne)
        lay.addLayout(haut)

        def _bloc(titre_bloc: str) -> None:
            contenu = blocs.pop(titre_bloc, None)
            if contenu is not None:
                lay.addWidget(_separateur())
                lay.addWidget(_titre_bloc(titre_bloc))
                lay.addWidget(contenu)

        _bloc("Ce que le modèle a appris")

        # — fiabilité : texte, puis la figure qui l'explique, l'essai de seuil et les
        #   zones, d'un seul tenant —
        if fiche.fiabilite:
            lay.addWidget(_separateur())
            lay.addWidget(_titre_bloc("Fiabilité mesurée au banc"))
            lay.addWidget(_label(self._fiabilite(fiche, observe), "FicheTexte"))
            note = self._note_observation(observe)
            if note:
                lay.addWidget(_label(note, "FicheLegende"))
        figure = (figure_profil(model_dir, fiche.nom, fiche.fiabilite, couleur=couleur)
                  if fiche.fiabilite else None)
        if figure is not None:
            lay.addWidget(_label("Profil des scores à l'évaluation", "FicheSousTitre"))
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
                                f"{figure.profil.seuil:g})").replace(".", ",") + "."
            lay.addWidget(_label(legende, "FicheLegende"))
            # Petits multiples par zone d'évaluation : une classe sûre ici et
            # faible là se voit d'un coup d'œil (les linéaires surtout).
            zones = figures_par_zone(model_dir, fiche.nom, fiche.fiabilite, couleur=couleur)
            # « Tester un seuil » : la figure (et les zones) suivent, précision et
            # rappel au banc s'affichent — sans toucher au seuil du traitement.
            lay.addWidget(ligne_essai_seuil(figure, [f for _n, f in zones], seuil))
            if zones:
                lay.addWidget(_label("Par zone d'évaluation", "FicheSousTitre"))
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
                    grille.addLayout(cellule, i, 0)   # une colonne : le panneau fait 420 px
                lay.addLayout(grille)

        _bloc("Limites connues du modèle")
        for titre_bloc in list(blocs):          # un bloc inattendu n'est jamais perdu
            _bloc(titre_bloc)

        # — contexte technique et fiche du modèle, en fin —
        lay.addWidget(_separateur())
        lay.addWidget(_titre_bloc("Contexte technique"))
        lay.addWidget(_label(self._contexte(fiche), "FicheTexte"))
        if ouvrir_modele is not None:
            lien = _LienFiche(f"Architecture, entraînement et métriques d'évaluation de « {fiche.modele} »")
            lien.setText("Fiche du modèle")
            lien.setObjectName("FicheLienModele")    # ardoise : la couleur des fiches de modèle
            lien.setIcon(colored_icon("modele", ARDOISE, 14, dpr=self.devicePixelRatioF()))
            lien.clicked.connect(lambda *_: ouvrir_modele())
            lay.addWidget(lien, 0, Qt.AlignmentFlag.AlignLeft)

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
            return (f"Aucun verdict pour cette classe dans vos {observe.n_runs} traitement(s) connu(s) : "
                    "renseignez le champ « validation » (oui / non / peut-être) dans QGIS.")
        return (f"« Chez vous » = vos verdicts (champ « validation » : oui / non / peut-être) sur "
                f"{observe.n_runs} traitement(s) connu(s), {n} vérification(s). Le banc est un plancher annoncé ; "
                "le terrain dit ce qu'il vaut ici.")

    @staticmethod
    def _entrainement(f: ClassFiche, avec_zones: bool = True) -> str:
        e = f.entrainement
        if e is None:
            return ""
        lignes: List[str] = []
        if e.corpus:
            lignes.extend(_lignes_liste("Corpus", e.corpus))
        if e.annotation:
            lignes.extend(_lignes_liste("Annotation", e.annotation))
        if e.zones and avec_zones:
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
                f"{s.nom} {_nb(s.tuiles)} dalles / {_nb(s.objets)} objets"
                for s in e.splits
            ]
            lignes.append("Répartition : " + "  ·  ".join(parts))
        if e.total_objets:
            lignes.append(
                f"Total : {_nb(e.total_objets)} objets annotés sur "
                f"{_nb(e.total_tuiles)} tuiles"
            )
        return "\n".join(lignes)

    def _blocs(self, f: ClassFiche, couleur=None):
        out = []
        # Bandeau « Appris sur » (carte + zones, 2026-10-08) en tête du bloc : il
        # remplace la liste des zones en texte ; sans zone située, la liste reste.
        bandeau = None
        if f.entrainement is not None and f.entrainement.zones:
            bandeau = bandeau_appris_sur(f.entrainement.zones, _zones_corpus(), couleur or (42, 120, 214))
        txt = self._entrainement(f, avec_zones=bandeau is None)
        if bandeau is not None:
            contenu = QWidget()
            pile = QVBoxLayout(contenu)
            pile.setContentsMargins(0, 0, 0, 0)
            pile.setSpacing(8)
            pile.addWidget(bandeau)
            if txt.strip():
                pile.addWidget(_label(txt.strip("\n"), "FicheTexte"))
            out.append(("Ce que le modèle a appris", contenu))
        elif txt:
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
# Feuilleter plusieurs fiches dans un seul corps
# ----------------------------------------------------------------------
class Feuilleteur(QWidget):
    """Un corps de fiche parmi plusieurs : une liste déroulante en tête dès qu'il y
    en a au moins deux, le corps choisi en dessous. Les corps sont construits à la
    demande (les figures se calculent à la construction) et conservés, cachés.

    ``entrees`` : ``[(libellé, infobulle, fabrique() -> QWidget), …]``.
    """

    change = pyqtSignal(int)

    def __init__(self, entrees, parent=None):
        super().__init__(parent)
        self._fabriques = [f for _l, _t, f in entrees]
        self._corps: dict = {}
        self._courant = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self._choix = QComboBox()
        self._choix.setObjectName("FicheChoix")
        for i, (libelle, infobulle, _f) in enumerate(entrees):
            self._choix.addItem(libelle)
            if infobulle:
                self._choix.setItemData(i, infobulle, Qt.ItemDataRole.ToolTipRole)
        self._choix.setVisible(len(entrees) > 1)
        self._choix.currentIndexChanged.connect(self._afficher)
        hote = QWidget()
        hl = QHBoxLayout(hote)
        hl.setContentsMargins(12, 8, 12, 0)
        hl.addWidget(self._choix, 1)
        hote.setVisible(len(entrees) > 1)
        lay.addWidget(hote)
        self._lay = lay
        if entrees:
            self._afficher(0)

    def selectionner(self, i: int) -> None:
        self._choix.setCurrentIndex(i)
        self._afficher(i)

    def _afficher(self, i: int) -> None:
        if not (0 <= i < len(self._fabriques)):
            return
        if self._courant is not None:
            self._corps[self._courant].hide()
        if i not in self._corps:
            w = self._fabriques[i]()
            self._corps[i] = w
            self._lay.addWidget(w)
        self._corps[i].show()
        self._courant = i
        self.change.emit(i)


class CorpsEntite(QWidget):
    """Hôte du corps d'une fiche de structure ; ``modele_demande(model)`` quand on
    clique « Fiche du modèle » — l'assistant l'ouvre dans l'onglet « Modèle »."""

    modele_demande = pyqtSignal(object)


def corps_fiche_entite(
    fiches: Sequence[ClassFiche],
    model_dirs: Optional[dict],
    titre: str,
    models: Optional[dict] = None,
    couleurs: Optional[dict] = None,
    seuils: Optional[dict] = None,
    observes: Optional[dict] = None,
) -> CorpsEntite:
    """Le corps de la fiche d'une structure (une fiche par classe qui la porte, à
    feuilleter). ``models`` (nom → ``InstalledModel``) active le lien « Fiche du
    modèle » ; ``couleurs`` et ``seuils`` (clé ``(modèle, classe)``) alignent la
    figure sur la couche et le seuil du traitement ; ``observes`` = vos verdicts."""
    del titre   # le panneau le porte
    hote = CorpsEntite()
    lay = QVBoxLayout(hote)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    fiches = list(fiches)
    dirs = dict(model_dirs or {})
    models = dict(models or {})
    couleurs = dict(couleurs or {})
    seuils = dict(seuils or {})
    observes = dict(observes or {})
    if not fiches:
        manque = _label("Aucun modèle installé ne couvre cette structure.", "FicheManque")
        manque.setContentsMargins(18, 16, 18, 16)
        lay.addWidget(manque)
        return hote

    def fabrique(f: ClassFiche):
        def _construire() -> QWidget:
            model = models.get(f.modele_id)
            ouvrir = (lambda m=model: hote.modele_demande.emit(m)) if model is not None else None
            return _CorpsFiche(
                f, dirs.get(f.modele_id), ouvrir_modele=ouvrir,
                couleur=couleurs.get((f.modele_id, f.nom)),
                seuil=seuils.get((f.modele_id, f.nom)),
                observe=observes.get((f.modele_id, f.nom)),
            )
        return _construire

    lay.addWidget(Feuilleteur([(f.label, f"{f.nom} — {f.modele}", fabrique(f)) for f in fiches]))
    return hote


# ----------------------------------------------------------------------
# Enveloppe hors écran : le corps dans un dialogue (dev/rendu_hors_ecran.py)
# ----------------------------------------------------------------------
class FicheDialog(QDialog):
    """Un corps de fiche dans sa fenêtre à part, non modale : c'est ainsi que l'assistant
    ouvre toute fiche depuis le 2026-10-25 (le panneau latéral du 2026-10-24 est retiré)."""

    def __init__(self, corps: QWidget, titre: str, nom_objet: str, parent=None,
                 taille=(900, 640)):
        super().__init__(parent)
        self.setObjectName(nom_objet)
        self.setWindowTitle(titre)
        self.setMinimumSize(560, 480)
        self.resize(*taille)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        zone = QScrollArea()
        zone.setObjectName("FicheScroll")
        zone.setWidgetResizable(True)
        zone.setFrameShape(QFrame.Shape.NoFrame)
        zone.setWidget(corps)
        root.addWidget(zone, 1)
        pied = QHBoxLayout()
        pied.setContentsMargins(14, 10, 14, 12)
        pied.addStretch(1)
        fermer = QPushButton("Fermer")
        fermer.setCursor(Qt.CursorShape.PointingHandCursor)
        fermer.clicked.connect(self.accept)
        pied.addWidget(fermer)
        root.addLayout(pied)


class ClassInfoDialog(FicheDialog):
    """Fiche(s) des classes qui portent une structure, en dialogue (hors écran)."""

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
        corps = corps_fiche_entite(
            fiches, model_dirs, titre, models=models, couleurs=couleurs, seuils=seuils, observes=observes,
        )
        # Nature dans le titre de la fenêtre (option A, 2026-10-08).
        super().__init__(corps, f"Structure · {titre}" if titre else "Structure détectable",
                         "ClassInfoDialog", parent)
        corps.modele_demande.connect(self._ouvrir_modele)

    def _ouvrir_modele(self, model) -> None:
        """Fiche ⓘ du modèle de la classe affichée, par-dessus celle-ci."""
        from .model_info_dialog import ModelInfoDialog  # différé

        ModelInfoDialog(model, parent=self).exec()
