"""Manuel intégré — la fenêtre d'aide du plugin.

Sommaire à gauche (chapitres et leurs sections), texte à droite, recherche dans
tout le manuel, historique Précédent / Suivant, zoom du texte, export PDF. Le
contenu vient du module PUR :mod:`app.services.aide` (``aide/*.md`` + Nouveautés
depuis ``metadata.txt``) ; ce module ne fait que des widgets Qt. Non modale : on
la garde ouverte à côté de l'assistant.

Rendu par ``QTextBrowser.setMarkdown`` (dialecte GitHub : tableaux, images, liens),
disponible dans Qt 5.15 (QGIS 3.34+) comme en Qt 6 (QGIS 4). Qt ne crée pas d'ancre
sur les titres : on parcourt les blocs de titre du document et on fait défiler
jusqu'au bloc dont le :func:`~app.services.aide.slug` est l'ancre demandée.

Depuis 2026-10-08 : la **recherche** liste les occurrences de tous les chapitres
dans le sommaire (chapitre › section — extrait) et saute dedans ; Entrée à nouveau
passe à l'occurrence suivante du chapitre ouvert ; vider le champ rend le sommaire.
**Précédent / Suivant** (Alt+← / Alt+→) rejouent la navigation ; **Ctrl+molette**,
Ctrl++ / Ctrl+- / Ctrl+0 règlent la taille du texte (la feuille de style est
exprimée en px : le zoom natif de QTextEdit, qui change la police du widget, n'y
ferait rien). À la **première ouverture après une mise à jour**, le manuel s'ouvre
sur les Nouveautés et le bouton « Aide » de l'assistant porte une pastille tant
qu'elles n'ont pas été affichées (réglage ``archeologia/aide/nouveautes_vues``).
Compatible Qt5/Qt6 : énumérés scopés.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from qgis.PyQt.QtCore import QSize, Qt, QUrl, pyqtSignal
from qgis.PyQt.QtGui import QColor, QDesktopServices, QImage, QKeySequence, QTextCursor, QTextDocument

# QShortcut a changé de module entre Qt5 (QtWidgets) et Qt6 (QtGui).
try:
    from qgis.PyQt.QtGui import QShortcut
except ImportError:  # pragma: no cover - Qt5
    from qgis.PyQt.QtWidgets import QShortcut
from qgis.PyQt.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextBrowser,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from ...app.services.aide import (
    CLE_NOUVEAUTES,
    Chapitre,
    Resultat,
    charger_chapitres,
    chapitre,
    est_image,
    rechercher,
    resoudre_cible,
    slug,
)
from ..icons import colored_icon

try:
    from ...app.plugin_metadata import get_plugin_version
except Exception:  # pragma: no cover - défensif (hors QGIS)
    def get_plugin_version() -> str:
        return ""

_ROLE_CIBLE = Qt.ItemDataRole.UserRole
_REGLAGE_NOUVEAUTES = "archeologia/aide/nouveautes_vues"
_ZOOM_MIN, _ZOOM_MAX, _ZOOM_PAS = 0.7, 2.0, 0.1

#: Feuille de style du document (sous-ensemble CSS de Qt rich text). Les tailles
#: suivent celles des fiches (#FicheTexte 11 px, #FicheTitre 15 px).
_CSS = """
h1 { font-size: 17px; font-weight: bold; color: #000000; margin-bottom: 6px; }
h2 { font-size: 13px; font-weight: bold; color: #000000; margin-top: 18px; }
h3 { font-size: 11px; font-weight: bold; color: #2c2c2c; margin-top: 12px; }
p, li { font-size: 11px; color: #2c2c2c; }
li { margin-bottom: 2px; }
a { color: #1d5a96; }
code { font-family: Consolas, "Courier New", monospace; font-size: 10px;
       background-color: #f0f0f0; }
pre { font-family: Consolas, "Courier New", monospace; font-size: 10px;
      background-color: #f0f0f0; margin: 6px 0; }
table { border-collapse: collapse; margin: 6px 0; }
th { font-size: 10px; color: #5a5a5a; background-color: #f0f0f0; padding: 3px 6px;
     border: 1px solid #dddddd; }
td { font-size: 11px; color: #2c2c2c; padding: 3px 6px; border: 1px solid #dddddd; }
blockquote { color: #8a5e18; margin-left: 0px; padding-left: 8px;
             border-left: 3px solid #e3c08a; }
hr { color: #e3e3e3; }
"""


# ----------------------------------------------------------------------
# Nouveautés vues : réglage QGIS (profil), repli QSettings hors QGIS
# ----------------------------------------------------------------------
def _reglages():
    try:
        from qgis.core import QgsSettings

        return QgsSettings()
    except Exception:  # noqa: BLE001 — hors QGIS (rendu hors écran, tests)
        from qgis.PyQt.QtCore import QSettings

        return QSettings("archeologia", "archeologia")


def nouveautes_non_lues() -> bool:
    """Vrai tant que les Nouveautés de la version installée n'ont pas été affichées."""
    version = get_plugin_version() or ""
    if not version or version == "?":          # metadata.txt illisible : pas de pastille
        return False
    try:
        return str(_reglages().value(_REGLAGE_NOUVEAUTES, "") or "") != version
    except Exception:  # noqa: BLE001
        return False


def marquer_nouveautes_lues() -> None:
    version = get_plugin_version() or ""
    if version and version != "?":
        try:
            _reglages().setValue(_REGLAGE_NOUVEAUTES, version)
        except Exception:  # noqa: BLE001
            pass


class _Texte(QTextBrowser):
    """QTextBrowser dont Ctrl+molette demande un zoom à la fenêtre (le zoom natif
    change la police du widget, sans effet sur une feuille de style en px)."""

    zoom_demande = pyqtSignal(int)

    def wheelEvent(self, ev) -> None:  # noqa: N802 (signature Qt)
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = ev.angleDelta().y()
            if delta:
                self.zoom_demande.emit(1 if delta > 0 else -1)
            ev.accept()
            return
        super().wheelEvent(ev)


class AideDialog(QDialog):
    """Le manuel, feuilletable : chapitres à gauche, texte à droite."""

    chapitre_affiche = pyqtSignal(str)   # clé du chapitre affiché (badge « Nouveautés »)

    def __init__(self, chapitres: Sequence[Chapitre], dossier: Path, parent=None):
        super().__init__(parent)
        self._chapitres = list(chapitres)
        self._dossier = Path(dossier)
        self._cle = ""
        self._ancre = ""
        self._historique: List[Tuple[str, str]] = []
        self._position = -1
        self._navigation = False          # vrai pendant Précédent / Suivant (pas d'empilement)
        self._zoom = 1.0
        self._recherche_courante = ""
        self._motif_courant = ""       # forme exacte (accents) du résultat ouvert
        self._mode_resultats = False
        self.setObjectName("AideDialog")
        version = get_plugin_version() or ""
        self.setWindowTitle(f"Manuel d'Archéolog'IA{' — v' + version if version else ''}")
        self.setMinimumSize(760, 520)
        self.resize(1000, 680)
        # Fenêtre à part entière (barre de titre, réduction) et non modale : le
        # manuel se lit à côté de l'assistant, pas à sa place.
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._barre())

        corps = QHBoxLayout()
        corps.setContentsMargins(0, 0, 0, 0)
        corps.setSpacing(0)
        self._sommaire = QTreeWidget()
        self._sommaire.setObjectName("AideSommaire")
        self._sommaire.setHeaderHidden(True)
        self._sommaire.setFixedWidth(250)
        self._sommaire.setIndentation(14)
        self._sommaire.currentItemChanged.connect(self._sur_selection)
        corps.addWidget(self._sommaire)

        self._texte = _Texte()
        self._texte.setObjectName("AideTexte")
        self._texte.setFrameShape(QFrame.Shape.NoFrame)
        self._texte.setOpenLinks(False)
        self._texte.setOpenExternalLinks(False)
        self._texte.setSearchPaths([str(self._dossier)])
        self._texte.document().setDefaultStyleSheet(self._css())
        self._texte.anchorClicked.connect(self._sur_lien)
        self._texte.zoom_demande.connect(self._zoomer)
        corps.addWidget(self._texte, 1)
        root.addLayout(corps, 1)

        # Comme dans un navigateur : Ctrl+F va au champ de recherche, F3 à l'occurrence
        # suivante, Alt+← / Alt+→ rejouent la navigation, Ctrl++ / Ctrl+- / Ctrl+0 zooment.
        QShortcut(QKeySequence(QKeySequence.StandardKey.Find), self, self._focus_recherche)
        QShortcut(QKeySequence(QKeySequence.StandardKey.FindNext), self, self._chercher)
        QShortcut(QKeySequence(QKeySequence.StandardKey.Back), self, self._precedent)
        QShortcut(QKeySequence(QKeySequence.StandardKey.Forward), self, self._suivant)
        QShortcut(QKeySequence(QKeySequence.StandardKey.ZoomIn), self, lambda: self._zoomer(1))
        QShortcut(QKeySequence(QKeySequence.StandardKey.ZoomOut), self, lambda: self._zoomer(-1))
        QShortcut(QKeySequence("Ctrl+0"), self, lambda: self._zoomer(0))

        self._remplir_sommaire()
        if self._chapitres:
            self.ouvrir(self._chapitres[0].cle)
        else:
            self._texte.setMarkdown(
                "# Manuel introuvable\n\nLe dossier `aide/` du plugin est vide ou absent."
            )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _bouton_icone(self, icone: str, infobulle: str, slot) -> QPushButton:
        b = QPushButton()
        b.setObjectName("GhostButton")
        b.setIcon(colored_icon(icone, "#2c2c2c", 14, dpr=self.devicePixelRatioF()))
        b.setIconSize(QSize(14, 14))
        b.setFixedWidth(30)
        b.setToolTip(infobulle)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.clicked.connect(slot)
        return b

    def _barre(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("AideBarre")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 8, 16, 8)
        lay.setSpacing(8)
        self._btn_precedent = self._bouton_icone("arrow-left", "Précédent (Alt+←)", self._precedent)
        self._btn_suivant = self._bouton_icone("arrow-right", "Suivant (Alt+→)", self._suivant)
        self._btn_precedent.setEnabled(False)
        self._btn_suivant.setEnabled(False)
        lay.addWidget(self._btn_precedent)
        lay.addWidget(self._btn_suivant)
        titre = QLabel("Manuel")
        titre.setObjectName("WizardTitle")
        lay.addWidget(titre)
        lay.addStretch(1)
        self._recherche = QLineEdit()
        self._recherche.setObjectName("AideRecherche")
        self._recherche.setPlaceholderText("Rechercher dans le manuel… (Ctrl+F)")
        self._recherche.setClearButtonEnabled(True)
        self._recherche.setFixedWidth(240)
        self._recherche.returnPressed.connect(self._chercher)
        self._recherche.textChanged.connect(self._sur_texte_recherche)
        lay.addWidget(self._recherche)
        suivant = QPushButton("Occurrence suivante")
        suivant.setObjectName("GhostButton")
        suivant.setToolTip("Occurrence suivante dans le chapitre ouvert (Entrée dans le champ, ou F3)")
        suivant.clicked.connect(self._chercher)
        lay.addWidget(suivant)
        pdf = QPushButton("Exporter en PDF")
        pdf.setObjectName("GhostButton")
        pdf.setToolTip("Le manuel complet, tous chapitres, dans un fichier PDF")
        pdf.clicked.connect(self._exporter_pdf)
        lay.addWidget(pdf)
        return bar

    def _remplir_sommaire(self) -> None:
        self._mode_resultats = False
        self._sommaire.blockSignals(True)
        try:
            self._sommaire.clear()
            for c in self._chapitres:
                racine = QTreeWidgetItem([c.titre])
                racine.setData(0, _ROLE_CIBLE, (c.cle, "", ""))
                racine.setToolTip(0, c.titre)
                for niveau, titre, s in c.sections:
                    if niveau == 2:
                        enfant = QTreeWidgetItem([titre])
                        enfant.setData(0, _ROLE_CIBLE, (c.cle, s, ""))
                        racine.addChild(enfant)
                self._sommaire.addTopLevelItem(racine)
        finally:
            self._sommaire.blockSignals(False)

    def _remplir_resultats(self, resultats: Sequence[Resultat]) -> None:
        """Le sommaire devient la liste des résultats : chapitre (n) › section — extrait."""
        self._mode_resultats = True
        self._sommaire.blockSignals(True)
        try:
            self._sommaire.clear()
            par_chapitre: dict = {}
            for r in resultats:
                par_chapitre.setdefault(r.cle, []).append(r)
            for cle, liste in par_chapitre.items():
                racine = QTreeWidgetItem([f"{liste[0].chapitre} ({len(liste)})"])
                racine.setData(0, _ROLE_CIBLE, (cle, liste[0].ancre, liste[0].motif))
                racine.setToolTip(0, liste[0].chapitre)
                for r in liste:
                    libelle = f"{r.section} — {r.extrait}" if r.section else r.extrait
                    enfant = QTreeWidgetItem([libelle])
                    enfant.setData(0, _ROLE_CIBLE, (r.cle, r.ancre, r.motif))
                    enfant.setToolTip(0, f"{r.chapitre} › {r.section}\n{r.extrait}" if r.section else r.extrait)
                    racine.addChild(enfant)
                racine.setExpanded(True)
                self._sommaire.addTopLevelItem(racine)
        finally:
            self._sommaire.blockSignals(False)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def ouvrir(self, cle: str, ancre: str = "") -> None:
        """Affiche le chapitre ``cle`` et fait défiler jusqu'au titre ``ancre``."""
        c = chapitre(self._chapitres, cle)
        if c is None:
            return
        if cle != self._cle:
            self._cle = cle
            # QTextEdit.setMarkdown n'a pas d'argument de dialecte : le défaut
            # est déjà GitHub (tableaux, listes à cocher, liens, images).
            self._texte.setMarkdown(c.markdown)
            self._texte.setExtraSelections([])
            self._texte.document().setDefaultStyleSheet(self._css())
            if cle == CLE_NOUVEAUTES:
                marquer_nouveautes_lues()
            self.chapitre_affiche.emit(cle)
        self._ancre = ancre
        self._ajuster_images()
        if not self._mode_resultats:
            self._selectionner(cle, ancre)
        if ancre:
            self._defiler_vers(ancre)
        else:
            self._texte.verticalScrollBar().setValue(0)
        self._empiler(cle, ancre)

    def _empiler(self, cle: str, ancre: str) -> None:
        if self._navigation:
            return
        if 0 <= self._position < len(self._historique) and self._historique[self._position] == (cle, ancre):
            return
        del self._historique[self._position + 1:]
        self._historique.append((cle, ancre))
        self._position = len(self._historique) - 1
        self._maj_boutons_historique()

    def _maj_boutons_historique(self) -> None:
        self._btn_precedent.setEnabled(self._position > 0)
        self._btn_suivant.setEnabled(self._position < len(self._historique) - 1)

    def _aller(self, position: int) -> None:
        if not (0 <= position < len(self._historique)):
            return
        self._position = position
        cle, ancre = self._historique[position]
        self._navigation = True
        try:
            self.ouvrir(cle, ancre)
        finally:
            self._navigation = False
        self._maj_boutons_historique()

    def _precedent(self) -> None:
        self._aller(self._position - 1)

    def _suivant(self) -> None:
        self._aller(self._position + 1)

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------
    def _css(self) -> str:
        """La feuille de style à l'échelle du zoom (toutes les tailles sont en px)."""
        if abs(self._zoom - 1.0) < 1e-9:
            return _CSS
        return re.sub(r"(\d+)px", lambda m: f"{max(1, round(int(m.group(1)) * self._zoom))}px", _CSS)

    def _zoomer(self, sens: int) -> None:
        """``sens`` +1 / −1 d'un cran, 0 = retour à la taille normale ; le chapitre est
        re-rendu (la feuille de style s'applique au rendu) à la même position."""
        nouveau = 1.0 if sens == 0 else min(_ZOOM_MAX, max(_ZOOM_MIN, round(self._zoom + sens * _ZOOM_PAS, 2)))
        if abs(nouveau - self._zoom) < 1e-9 or not self._cle:
            return
        self._zoom = nouveau
        barre = self._texte.verticalScrollBar()
        ratio = barre.value() / barre.maximum() if barre.maximum() else 0.0
        cle, ancre = self._cle, self._ancre
        self._cle = ""
        self._navigation = True
        try:
            self.ouvrir(cle, "")
        finally:
            self._navigation = False
        self._ancre = ancre
        barre.setValue(int(ratio * barre.maximum()))
        self._texte.setToolTip(f"Taille du texte : {round(self._zoom * 100)} % (Ctrl+molette, Ctrl+0 pour revenir)")

    def _ajuster_images(self) -> None:
        """Ramène chaque image à la largeur de lecture, lissée et nette.

        Qt affiche une image Markdown à sa taille en pixels et la redimensionne
        au dessin sans lissage : une capture de 980 px débordait, puis, ramenée
        à la largeur, pixelisait. Même discipline que ``pixmap_ajuste`` des
        vignettes : on rastérise nous-mêmes à ``largeur × dpr`` pixels physiques
        avec lissage, on pose le ratio d'écran et on enregistre le résultat comme
        ressource du document sous le nom de l'image. Un clic sur l'image l'ouvre
        en taille réelle (cf. ``_sur_lien``).
        """
        doc = self._texte.document()
        dpr = self.devicePixelRatioF()
        # Largeur disponible = zone de lecture moins le padding QSS (18 px de
        # chaque côté), les marges du document et le retrait d'une image placée
        # dans une liste (un niveau = indentWidth, 40 px) — sinon une barre
        # horizontale apparaît pour quelques pixels.
        base = self._texte.viewport().width() - 36 - 2 * doc.documentMargin() - 8
        bloc = doc.begin()
        while bloc.isValid():
            niveau = bloc.blockFormat().indent()
            if bloc.textList() is not None:
                niveau += bloc.textList().format().indent()
            largeur_max = max(200, int(base - niveau * doc.indentWidth()))
            it = bloc.begin()
            while not it.atEnd():
                frag = it.fragment()
                fmt = frag.charFormat()
                if fmt.isImageFormat():
                    img_fmt = fmt.toImageFormat()
                    nom = img_fmt.name()
                    image = QImage(str(self._dossier / nom))
                    if not image.isNull():
                        largeur = min(image.width(), largeur_max)
                        rendu = image.scaledToWidth(
                            max(1, round(largeur * dpr)),
                            Qt.TransformationMode.SmoothTransformation,
                        )
                        rendu.setDevicePixelRatio(dpr)
                        doc.addResource(
                            QTextDocument.ResourceType.ImageResource, QUrl(nom), rendu
                        )
                        img_fmt.setWidth(largeur)
                        img_fmt.setHeight(image.height() * largeur / image.width())
                        curseur = QTextCursor(doc)
                        curseur.setPosition(frag.position())
                        curseur.setPosition(
                            frag.position() + frag.length(), QTextCursor.MoveMode.KeepAnchor
                        )
                        curseur.setCharFormat(img_fmt)
                it += 1
            bloc = bloc.next()

    def _selectionner(self, cle: str, ancre: str) -> None:
        """Met le sommaire en phase sans redéclencher ``ouvrir``."""
        self._sommaire.blockSignals(True)
        try:
            for i in range(self._sommaire.topLevelItemCount()):
                racine = self._sommaire.topLevelItem(i)
                if racine.data(0, _ROLE_CIBLE)[0] != cle:
                    continue
                racine.setExpanded(True)
                cible = racine
                for j in range(racine.childCount()):
                    if racine.child(j).data(0, _ROLE_CIBLE)[1] == ancre:
                        cible = racine.child(j)
                self._sommaire.setCurrentItem(cible)
                self._sommaire.scrollToItem(cible)
                break
        finally:
            self._sommaire.blockSignals(False)

    def _bloc_titre(self, ancre: str):
        doc = self._texte.document()
        bloc = doc.begin()
        while bloc.isValid():
            if bloc.blockFormat().headingLevel() and slug(bloc.text()) == ancre:
                return bloc
            bloc = bloc.next()
        return None

    def _defiler_vers(self, ancre: str) -> None:
        bloc = self._bloc_titre(ancre)
        if bloc is not None:
            haut = self._texte.document().documentLayout().blockBoundingRect(bloc).top()
            self._texte.verticalScrollBar().setValue(int(haut))

    def _sur_selection(self, item: Optional[QTreeWidgetItem], _prev=None) -> None:
        if item is None:
            return
        cle, ancre, motif = item.data(0, _ROLE_CIBLE)
        self.ouvrir(cle, ancre)
        if motif:
            self._aller_au_motif(ancre, motif)

    def _aller_au_motif(self, ancre: str, motif: str) -> None:
        """Depuis le titre de la section (ou le début), la première occurrence du motif."""
        self._motif_courant = motif
        curseur = self._texte.textCursor()
        bloc = self._bloc_titre(ancre) if ancre else None
        curseur.setPosition(bloc.position() if bloc is not None else 0)
        self._texte.setTextCursor(curseur)
        if self._texte.find(motif):
            self._surligner(motif)
            c = self._texte.textCursor()
            c.clearSelection()
            self._texte.setTextCursor(c)
            self._texte.ensureCursorVisible()

    def _sur_lien(self, url: QUrl) -> None:
        """Lien externe → navigateur ; lien interne → chapitre et ancre."""
        if url.scheme() in ("http", "https", "mailto"):
            QDesktopServices.openUrl(url)
            return
        # setOpenLinks(False) : l'URL arrive telle qu'écrite dans le Markdown
        # (« etape-2-produits.md#reglages » ou « #reglages »), sans résolution.
        cible = url.toString()
        if est_image(cible):
            # Une capture liée à elle-même : taille réelle dans la visionneuse
            # du système, comme les courbes de la fiche ⓘ d'un modèle.
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._dossier / cible)))
            return
        cle, ancre = resoudre_cible(cible)
        self.ouvrir(cle or self._cle, ancre)

    # ------------------------------------------------------------------
    # Recherche
    # ------------------------------------------------------------------
    def _focus_recherche(self) -> None:
        self._recherche.setFocus()
        self._recherche.selectAll()

    def _sur_texte_recherche(self, texte: str) -> None:
        if not texte.strip() and self._mode_resultats:
            self._quitter_resultats()

    def _quitter_resultats(self) -> None:
        self._recherche_courante = ""
        self._motif_courant = ""
        self._texte.setExtraSelections([])
        self._remplir_sommaire()
        self._selectionner(self._cle, self._ancre)

    def _chercher(self) -> None:
        """Nouveau texte → résultats de tout le manuel dans le sommaire, et saut au
        premier ; même texte → occurrence suivante dans le chapitre ouvert."""
        texte = self._recherche.text().strip()
        if not texte:
            if self._mode_resultats:
                self._quitter_resultats()
            self._texte.setExtraSelections([])
            return
        if texte != self._recherche_courante:
            self._recherche_courante = texte
            resultats = rechercher(self._chapitres, texte)
            if not resultats:
                self._recherche.setToolTip(f"« {texte} » n'apparaît pas dans le manuel")
                if self._mode_resultats:
                    self._quitter_resultats()
                    self._recherche_courante = texte
                self._texte.setExtraSelections([])
                return
            self._recherche.setToolTip(f"{len(resultats)} occurrence(s) dans le manuel")
            self._remplir_resultats(resultats)
            self._sommaire.setCurrentItem(self._sommaire.topLevelItem(0).child(0))  # → _sur_selection
            return
        # Occurrence suivante : la forme exacte du résultat ouvert (« fiabilité » pour
        # « fiabilite » tapé) — QTextDocument.find tient compte des accents.
        motif = self._motif_courant or texte
        trouve = self._texte.find(motif)
        if not trouve:
            # Fin du chapitre : on repart du début (une seule fois).
            curseur = self._texte.textCursor()
            curseur.movePosition(QTextCursor.MoveOperation.Start)
            self._texte.setTextCursor(curseur)
            trouve = self._texte.find(motif)
        if not trouve:
            self._recherche.setToolTip(f"« {texte} » n'apparaît pas dans ce chapitre")
            self._texte.setExtraSelections([])
            return
        self._surligner(motif)
        # La sélection du curseur se peint PAR-DESSUS le surlignage (en bleu si le
        # widget a le focus, en gris pâle sinon) : on la retire, le jaune franc
        # marque seul l'occurrence courante ; la position reste à sa fin pour que
        # « Occurrence suivante » enchaîne.
        curseur = self._texte.textCursor()
        curseur.clearSelection()
        self._texte.setTextCursor(curseur)
        self._texte.ensureCursorVisible()

    def _surligner(self, texte: str) -> None:
        """Toutes les occurrences en jaune clair, la courante en jaune franc.

        La sélection seule ne se voit presque pas : le champ de recherche garde le
        focus et le widget peint alors sa sélection « inactive », gris pâle
        (constat utilisateur).
        """
        doc = self._texte.document()
        courant = self._texte.textCursor()
        selections = []
        curseur = QTextCursor(doc)
        while True:
            curseur = doc.find(texte, curseur)
            if curseur.isNull():
                break
            sel = QTextEdit.ExtraSelection()
            sel.cursor = curseur
            meme = (curseur.selectionStart() == courant.selectionStart()
                    and curseur.selectionEnd() == courant.selectionEnd())
            sel.format.setBackground(QColor("#ffd54a" if meme else "#fff3b0"))
            selections.append(sel)
        self._texte.setExtraSelections(selections)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _exporter_pdf(self) -> None:
        version = get_plugin_version() or ""
        defaut = f"manuel-archeologia{'-v' + version if version else ''}.pdf"
        chemin, _f = QFileDialog.getSaveFileName(
            self, "Exporter le manuel en PDF", str(Path.home() / defaut), "PDF (*.pdf)"
        )
        if not chemin:
            return
        try:
            from qgis.PyQt.QtPrintSupport import QPrinter

            doc = QTextDocument()
            doc.setDefaultStyleSheet(_CSS)
            # Chemins d'images relatifs à aide/ : le document les résout par son URL de base.
            doc.setBaseUrl(QUrl.fromLocalFile(str(self._dossier) + "/"))
            doc.setMarkdown("\n\n---\n\n".join(c.markdown for c in self._chapitres))
            imprimante = QPrinter(QPrinter.PrinterMode.HighResolution)
            imprimante.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
            imprimante.setOutputFileName(chemin)
            # PyQt6 nomme la méthode `print`, PyQt5 `print_` (mot réservé de Python 2).
            (getattr(doc, "print", None) or doc.print_)(imprimante)
        except Exception as exc:  # noqa: BLE001 — l'export est un confort, jamais un crash
            QMessageBox.warning(self, "Export PDF", f"L'export a échoué : {exc}")
            return
        QMessageBox.information(self, "Export PDF", f"Manuel enregistré :\n{chemin}")


# ----------------------------------------------------------------------
# Point d'entrée : une seule fenêtre, réutilisée
# ----------------------------------------------------------------------
_instance: Optional[AideDialog] = None


def ouvrir_aide(
    dossier: Path,
    parent=None,
    cle: str = "",
    ancre: str = "",
    metadata_path: Optional[Path] = None,
    nouveautes_si_non_lues: bool = False,
) -> AideDialog:
    """Ouvre (ou ramène au premier plan) le manuel, sur ``cle``/``ancre`` si donnés.

    ``dossier`` est ``<plugin>/aide`` ; ``metadata_path`` (``<plugin>/metadata.txt``)
    alimente le chapitre Nouveautés. La fenêtre est relue à chaque ouverture depuis
    le disque : un chapitre corrigé apparaît sans relancer QGIS.
    ``nouveautes_si_non_lues`` (bouton « Aide », menu Manuel) : à la première
    ouverture après une mise à jour, le chapitre Nouveautés prend le pas sur
    ``cle`` — un renvoi précis (journal → Dépannage) ne le demande pas.
    """
    global _instance
    dossier = Path(dossier)
    if metadata_path is None:
        metadata_path = dossier.parent / "metadata.txt"
    chapitres = charger_chapitres(dossier, metadata_path=metadata_path)
    if nouveautes_si_non_lues and nouveautes_non_lues() and chapitre(chapitres, CLE_NOUVEAUTES):
        cle, ancre = CLE_NOUVEAUTES, ""
    vivante = _instance is not None
    if vivante:
        try:
            _instance.isVisible()
        except RuntimeError:          # objet C++ détruit avec son parent
            vivante = False
    if not vivante:
        _instance = AideDialog(chapitres, dossier, parent=parent)
    else:
        _instance._chapitres = chapitres
        _instance._cle = ""
        _instance._recherche_courante = ""
        _instance._remplir_sommaire()
    if cle:
        _instance.ouvrir(cle, ancre)
    elif chapitres:
        _instance.ouvrir(chapitres[0].cle)
    _instance.show()
    _instance.raise_()
    _instance.activateWindow()
    return _instance
