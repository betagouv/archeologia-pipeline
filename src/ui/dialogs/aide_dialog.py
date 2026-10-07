"""Manuel intégré — la fenêtre d'aide du plugin.

Sommaire à gauche (chapitres et leurs sections), texte à droite, recherche dans le
chapitre, export PDF. Le contenu vient du module PUR :mod:`app.services.aide`
(``aide/*.md`` + Nouveautés depuis ``metadata.txt``) ; ce module ne fait que des
widgets Qt. Non modale : on la garde ouverte à côté de l'assistant.

Rendu par ``QTextBrowser.setMarkdown`` (dialecte GitHub : tableaux, images, liens),
disponible dans Qt 5.15 (QGIS 3.34+) comme en Qt 6 (QGIS 4). Qt ne crée pas d'ancre
sur les titres : on parcourt les blocs de titre du document et on fait défiler
jusqu'au bloc dont le :func:`~app.services.aide.slug` est l'ancre demandée.
Compatible Qt5/Qt6 : énumérés scopés.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from qgis.PyQt.QtCore import Qt, QUrl
from qgis.PyQt.QtGui import QDesktopServices, QImage, QTextCursor, QTextDocument
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
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from ...app.services.aide import (
    Chapitre,
    charger_chapitres,
    chapitre,
    est_image,
    resoudre_cible,
    slug,
)

try:
    from ...app.plugin_metadata import get_plugin_version
except Exception:  # pragma: no cover - défensif (hors QGIS)
    def get_plugin_version() -> str:
        return ""

_ROLE_CIBLE = Qt.ItemDataRole.UserRole

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


class AideDialog(QDialog):
    """Le manuel, feuilletable : chapitres à gauche, texte à droite."""

    def __init__(self, chapitres: Sequence[Chapitre], dossier: Path, parent=None):
        super().__init__(parent)
        self._chapitres = list(chapitres)
        self._dossier = Path(dossier)
        self._cle = ""
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

        self._texte = QTextBrowser()
        self._texte.setObjectName("AideTexte")
        self._texte.setFrameShape(QFrame.Shape.NoFrame)
        self._texte.setOpenLinks(False)
        self._texte.setOpenExternalLinks(False)
        self._texte.setSearchPaths([str(self._dossier)])
        self._texte.document().setDefaultStyleSheet(_CSS)
        self._texte.anchorClicked.connect(self._sur_lien)
        corps.addWidget(self._texte, 1)
        root.addLayout(corps, 1)

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
    def _barre(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("AideBarre")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 8, 16, 8)
        lay.setSpacing(8)
        titre = QLabel("Manuel")
        titre.setObjectName("WizardTitle")
        lay.addWidget(titre)
        lay.addStretch(1)
        self._recherche = QLineEdit()
        self._recherche.setObjectName("AideRecherche")
        self._recherche.setPlaceholderText("Rechercher dans ce chapitre…")
        self._recherche.setClearButtonEnabled(True)
        self._recherche.setFixedWidth(240)
        self._recherche.returnPressed.connect(self._chercher)
        lay.addWidget(self._recherche)
        suivant = QPushButton("Suivant")
        suivant.setObjectName("GhostButton")
        suivant.setToolTip("Occurrence suivante (Entrée dans le champ)")
        suivant.clicked.connect(self._chercher)
        lay.addWidget(suivant)
        pdf = QPushButton("Exporter en PDF")
        pdf.setObjectName("GhostButton")
        pdf.setToolTip("Le manuel complet, tous chapitres, dans un fichier PDF")
        pdf.clicked.connect(self._exporter_pdf)
        lay.addWidget(pdf)
        return bar

    def _remplir_sommaire(self) -> None:
        self._sommaire.clear()
        for c in self._chapitres:
            racine = QTreeWidgetItem([c.titre])
            racine.setData(0, _ROLE_CIBLE, (c.cle, ""))
            racine.setToolTip(0, c.titre)
            for niveau, titre, s in c.sections:
                if niveau == 2:
                    enfant = QTreeWidgetItem([titre])
                    enfant.setData(0, _ROLE_CIBLE, (c.cle, s))
                    racine.addChild(enfant)
            self._sommaire.addTopLevelItem(racine)

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
            self._texte.document().setDefaultStyleSheet(_CSS)
        self._ajuster_images()
        self._selectionner(cle, ancre)
        if ancre:
            self._defiler_vers(ancre)
        else:
            self._texte.verticalScrollBar().setValue(0)

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

    def _defiler_vers(self, ancre: str) -> None:
        doc = self._texte.document()
        bloc = doc.begin()
        while bloc.isValid():
            if bloc.blockFormat().headingLevel() and slug(bloc.text()) == ancre:
                haut = doc.documentLayout().blockBoundingRect(bloc).top()
                self._texte.verticalScrollBar().setValue(int(haut))
                return
            bloc = bloc.next()

    def _sur_selection(self, item: Optional[QTreeWidgetItem], _prev=None) -> None:
        if item is None:
            return
        cle, ancre = item.data(0, _ROLE_CIBLE)
        self.ouvrir(cle, ancre)

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

    def _chercher(self) -> None:
        texte = self._recherche.text().strip()
        if not texte:
            return
        if self._texte.find(texte):
            return
        # Fin du chapitre : on repart du début (une seule fois).
        curseur = self._texte.textCursor()
        curseur.movePosition(QTextCursor.MoveOperation.Start)
        self._texte.setTextCursor(curseur)
        if not self._texte.find(texte):
            self._recherche.setToolTip(f"« {texte} » n'apparaît pas dans ce chapitre")

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
            doc.print_(imprimante)
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
) -> AideDialog:
    """Ouvre (ou ramène au premier plan) le manuel, sur ``cle``/``ancre`` si donnés.

    ``dossier`` est ``<plugin>/aide`` ; ``metadata_path`` (``<plugin>/metadata.txt``)
    alimente le chapitre Nouveautés. La fenêtre est relue à chaque ouverture depuis
    le disque : un chapitre corrigé apparaît sans relancer QGIS.
    """
    global _instance
    dossier = Path(dossier)
    if metadata_path is None:
        metadata_path = dossier.parent / "metadata.txt"
    chapitres = charger_chapitres(dossier, metadata_path=metadata_path)
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
        _instance._remplir_sommaire()
    if cle:
        _instance.ouvrir(cle, ancre)
    elif chapitres:
        _instance.ouvrir(chapitres[0].cle)
    _instance.show()
    _instance.raise_()
    _instance.activateWindow()
    return _instance
