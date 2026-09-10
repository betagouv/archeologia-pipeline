"""Onglet « Visualisation » — mur visuel du catalogue France entière.

Parcourir les MNT / indices RVT déjà produits et les ouvrir dans QGIS sans
relancer de pipeline : rail des départements à gauche, mur de vignettes à
droite, une couche par clic. Le dialogue reste ouvert pendant ce temps — c'est
tout l'intérêt : on empile MNT + SVF + LD et on compare sur la carte.

Cet onglet ne diffuse que des **fonds** (MNT et indices de visualisation).
Les détections IA n'y figurent pas : elles sortent du pipeline, pas du catalogue.

L'état d'une carte est dérivé de la pile de couches de QGIS, jamais stocké dans
la carte : si l'utilisateur supprime la couche depuis le panneau Couches, la
carte repasse en « idle » (signal ``layersRemoved``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from qgis.PyQt.QtCore import QEvent, Qt, pyqtSignal
from qgis.PyQt.QtGui import QPixmap
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..app.services.visu_catalogue import (
    FAMILY_ORDER,
    Catalogue,
    CatalogItem,
    Department,
    filter_departments,
    load_catalogue,
)
from .widgets.indice_card import (
    CARD_W,
    STATE_IDLE,
    STATE_LOADED,
    STATE_LOADING,
    IndiceCard,
)

RAIL_W = 252
GRID_HSPACING = 14
GRID_VSPACING = 16

#: Emplacement du catalogue livré avec le plugin.
DEFAULT_CATALOGUE = Path("data") / "demo_catalogue" / "catalogue.json"


class _DeptItem(QWidget):
    """Une ligne du rail : pastille du code + nom + « N indices · maj AAAA-MM »."""

    def __init__(self, dept: Department, parent=None):
        super().__init__(parent)
        self.setObjectName("VisuRailItem")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(10)

        code = QLabel(dept.code)
        code.setObjectName("VisuRailCode")
        code.setAlignment(Qt.AlignmentFlag.AlignCenter)
        code.setFixedSize(30, 26)
        lay.addWidget(code)

        texts = QVBoxLayout()
        texts.setContentsMargins(0, 0, 0, 0)
        texts.setSpacing(1)
        name = QLabel(dept.name)
        name.setObjectName("VisuRailName")
        sub = QLabel(f"{dept.count} indices · maj {dept.updated or '—'}")
        sub.setObjectName("VisuRailSub")
        texts.addWidget(name)
        texts.addWidget(sub)
        lay.addLayout(texts, 1)
        self._parts = (self, code, name)

    def set_active(self, active: bool) -> None:
        """``QListWidget::item:selected`` ne traverse pas ``setItemWidget`` :
        c'est la ligne qui doit repropager son état à ses enfants stylés."""
        for w in self._parts:
            w.setProperty("state", "active" if active else "")
            w.style().unpolish(w)
            w.style().polish(w)


class VisualisationTab(QWidget):
    """Mur visuel du catalogue + ouverture des couches dans QGIS."""

    #: Nombre de couches ouvertes par l'onglet (pour le compteur d'onglet).
    layer_count_changed = pyqtSignal(int)

    def __init__(self, plugin_root: Path, iface=None, parent=None):
        super().__init__(parent)
        self._plugin_root = Path(plugin_root)
        self._iface = iface
        self._catalogue: Catalogue = Catalogue()
        self._catalogue_dir: Path = self._plugin_root / DEFAULT_CATALOGUE.parent
        self._dept: Optional[Department] = None
        self._family: Optional[str] = None
        self._cards: Dict[str, IndiceCard] = {}
        self._pixmaps: Dict[str, QPixmap] = {}
        # id de couche QGIS -> (code_departement, clé d'indice). N'y figurent que
        # les couches ajoutées par cet onglet : « Tout retirer » ne doit jamais
        # toucher aux autres couches du projet de l'utilisateur.
        self._layers: Dict[str, tuple] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())
        root.addWidget(self._build_liseret())
        root.addWidget(self._build_body(), 1)
        root.addWidget(self._build_footer())

        self._connect_project()
        self.reload_catalogue()

    # ------------------------------------------------------------------ UI

    def _build_header(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("VisuHeader")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 10, 16, 10)
        lay.setSpacing(12)

        texts = QVBoxLayout()
        texts.setContentsMargins(0, 0, 0, 0)
        texts.setSpacing(1)
        title = QLabel("Visualisation des indices")
        title.setObjectName("VisuTitle")
        sub = QLabel("Catalogue France entière · MNT & indices RVT prêts à consulter")
        sub.setObjectName("VisuSubtitle")
        texts.addWidget(title)
        texts.addWidget(sub)
        lay.addLayout(texts, 1)

        self._cat_stamp = QLabel("—")
        self._cat_stamp.setObjectName("VisuStamp")
        lay.addWidget(self._cat_stamp)

        self._refresh_btn = QPushButton("Actualiser le catalogue")
        self._refresh_btn.setObjectName("GhostButton")
        self._refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh_btn.clicked.connect(self.reload_catalogue)
        lay.addWidget(self._refresh_btn)
        return bar

    def _build_liseret(self) -> QWidget:
        self._liseret = QProgressBar()
        self._liseret.setObjectName("WizardProgress")
        self._liseret.setTextVisible(False)
        self._liseret.setFixedHeight(4)
        self._liseret.setRange(0, 0)          # indéterminé
        self._liseret.setVisible(False)
        return self._liseret

    def _build_body(self) -> QWidget:
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._build_rail())
        split.addWidget(self._build_wall())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([RAIL_W, 700])
        return split

    def _build_rail(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("VisuRail")
        panel.setMinimumWidth(RAIL_W)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filtrer un département…")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._refresh_rail)
        lay.addWidget(self._filter)

        self._rail_count = QLabel("—")
        self._rail_count.setObjectName("VisuRailCount")
        lay.addWidget(self._rail_count)

        self._rail = QListWidget()
        self._rail.setObjectName("VisuRailList")
        self._rail.setFrameShape(QFrame.Shape.NoFrame)
        self._rail.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._rail.currentItemChanged.connect(self._on_dept_changed)
        lay.addWidget(self._rail, 1)
        return panel

    def _build_wall(self) -> QWidget:
        wall = QWidget()
        lay = QVBoxLayout(wall)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(10)

        head = QHBoxLayout()
        head.setSpacing(10)
        self._wall_title = QLabel("—")
        self._wall_title.setObjectName("VisuWallTitle")
        self._wall_meta = QLabel("")
        self._wall_meta.setObjectName("VisuWallMeta")
        head.addWidget(self._wall_title)
        head.addWidget(self._wall_meta, 1)

        self._family_btns: Dict[Optional[str], QPushButton] = {}
        for label, fam in [("Tous", None)] + [(f, f) for f in FAMILY_ORDER]:
            b = QPushButton(label)
            b.setObjectName("VisuFamilyBtn")
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _checked, f=fam: self._set_family(f))
            self._family_btns[fam] = b
            head.addWidget(b)
        lay.addLayout(head)

        self._banner = QLabel()
        self._banner.setObjectName("VisuBanner")
        self._banner.setWordWrap(True)
        lay.addWidget(self._banner)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(GRID_HSPACING)
        self._grid.setVerticalSpacing(GRID_VSPACING)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._scroll.setWidget(self._grid_host)
        self._scroll.viewport().installEventFilter(self)
        lay.addWidget(self._scroll, 1)

        self._empty = QLabel("")
        self._empty.setObjectName("VisuEmpty")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setVisible(False)
        lay.addWidget(self._empty)
        return wall

    def _build_footer(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("VisuFooter")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 8, 16, 8)
        lay.setSpacing(10)

        self._footer_state = QLabel("Aucune couche ouverte dans QGIS")
        self._footer_state.setObjectName("VisuFooterState")
        lay.addWidget(self._footer_state, 1)

        self._clear_btn = QPushButton("Tout retirer")
        self._clear_btn.setObjectName("GhostButton")
        self._clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._clear_btn.clicked.connect(self._remove_all)
        self._clear_btn.setVisible(False)
        lay.addWidget(self._clear_btn)

        self._to_qgis_btn = QPushButton("Voir dans QGIS →")
        self._to_qgis_btn.setObjectName("VisuPrimaryBtn")
        self._to_qgis_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._to_qgis_btn.clicked.connect(self._show_qgis)
        lay.addWidget(self._to_qgis_btn)
        return bar

    # ------------------------------------------------------- catalogue

    def reload_catalogue(self) -> None:
        self._refresh_btn.setEnabled(False)
        self._refresh_btn.setText("Actualisation…")
        QApplication.processEvents()
        path = self._plugin_root / DEFAULT_CATALOGUE
        try:
            self._catalogue = load_catalogue(path)
            self._catalogue_dir = path.parent
            self._cat_stamp.setText(f"catalogue {self._catalogue.updated or '—'}")
        except Exception as exc:                       # noqa: BLE001 — jamais de grille muette
            self._catalogue = Catalogue()
            self._cat_stamp.setText("catalogue indisponible")
            self._show_empty(
                "Catalogue injoignable — vérifiez la connexion ou l'emplacement du "
                f"catalogue.\n({exc})")
        finally:
            self._refresh_btn.setEnabled(True)
            self._refresh_btn.setText("Actualiser le catalogue")
        self._refresh_rail()

    def _refresh_rail(self) -> None:
        covered = self._catalogue.covered
        shown = filter_departments(covered, self._filter.text())
        total = len(self._catalogue.departments) or len(covered)

        previous = self._dept.code if self._dept else None
        self._rail.blockSignals(True)
        self._rail.clear()
        for dept in shown:
            item = QListWidgetItem()
            widget = _DeptItem(dept)
            item.setSizeHint(widget.sizeHint())
            item.setData(Qt.ItemDataRole.UserRole, dept.code)
            self._rail.addItem(item)
            self._rail.setItemWidget(item, widget)
        self._rail.blockSignals(False)

        self._rail_count.setText(
            f"{len(covered)} départements couverts sur {total}"
            + ("" if len(shown) == len(covered) else f" · {len(shown)} affichés"))

        if not shown:
            self._show_empty("Aucun résultat.")
            return
        # On garde le département courant s'il survit au filtre ; sinon on ouvre
        # sur celui que le catalogue met en avant (le mieux pourvu), pas sur le
        # premier de l'alphabet — le mur doit être plein dès l'ouverture.
        wanted = previous or self._featured_code()
        target = 0
        if wanted:
            for i in range(self._rail.count()):
                if self._rail.item(i).data(Qt.ItemDataRole.UserRole) == wanted:
                    target = i
                    break
        self._rail.setCurrentRow(target)
        item = self._rail.item(target)
        if item is not None:
            self._rail.scrollToItem(
                item, QAbstractItemView.ScrollHint.PositionAtCenter)

    def _featured_code(self) -> Optional[str]:
        """Département d'ouverture : celui que le catalogue signale, sinon le mieux pourvu."""
        covered = self._catalogue.covered
        if not covered:
            return None
        for dept in covered:
            if dept.featured:
                return dept.code
        return max(covered, key=lambda d: d.count).code

    def _on_dept_changed(self, current: Optional[QListWidgetItem], previous=None) -> None:
        for item, active in ((previous, False), (current, True)):
            if item is None:
                continue
            widget = self._rail.itemWidget(item)
            if isinstance(widget, _DeptItem):
                widget.set_active(active)
        if current is None:
            return
        code = current.data(Qt.ItemDataRole.UserRole)
        self._dept = self._catalogue.by_code(code)
        self._family = None                    # le filtre repart à « Tous »
        self._rebuild_wall()

    def _set_family(self, family: Optional[str]) -> None:
        self._family = family
        self._rebuild_wall()

    # ------------------------------------------------------------- mur

    def _rebuild_wall(self) -> None:
        dept = self._dept
        for card in self._cards.values():
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()
        if dept is None:
            return

        res = f" · dalle {dept.resolution:g} m".replace(".", ",") if dept.resolution else ""
        self._wall_title.setText(dept.name)
        self._wall_meta.setText(
            f"{dept.count} indices · publiés {dept.updated or '—'}{res}")

        present = dept.families
        for fam, btn in self._family_btns.items():
            btn.setChecked(fam == self._family)
            n = dept.count if fam is None else len(dept.items_in_family(fam))
            btn.setText(("Tous" if fam is None else fam) + f" ({n})")
            btn.setEnabled(fam is None or fam in present)

        self._banner.setText(self._banner_text(dept))

        items = dept.items_in_family(self._family)
        if not items:
            self._show_empty("Aucun indice dans cette famille.")
            return
        self._empty.setVisible(False)
        self._scroll.setVisible(True)

        for it in items:
            info = it.info
            card = IndiceCard(it.key, info.metier, info.name)
            card.set_content(self._pixmap_for(it), info.sigle,
                             f"{it.size_go:g} Go".replace(".", ",") if it.size_go else "")
            card.open_requested.connect(self._open_indice)
            card.remove_requested.connect(self._remove_indice)
            self._cards[it.key] = card
        self._relayout_grid()
        self._sync_cards()

    def _banner_text(self, dept: Department) -> str:
        """Le bandeau annonce ce qui va RÉELLEMENT se passer au clic.

        Un catalogue de démonstration lit des rasters déjà calculés sur le poste :
        écrire « rien n'est téléchargé » serait vrai mais laisserait croire à une
        diffusion distante qui n'existe pas encore. On le dit, et on dit ce que
        le même geste fera en production.
        """
        geste = ("Utilisez « Afficher dans QGIS » pour ouvrir un indice, "
                 "derrière cette fenêtre.")
        if any(item.streamed for item in dept.items):
            return (f"<b>Lecture en flux</b> · {geste} Le raster est lu à la demande "
                    "— rien n'est téléchargé sur votre poste.")
        return (f"<b>Aperçu local</b> · {geste} Ce catalogue lit des rasters déjà "
                "calculés sur ce poste ; en diffusion, le même geste lira un COG "
                "distant sans rien télécharger.")

    def _pixmap_for(self, item: CatalogItem) -> QPixmap:
        """Vignette locale, mise en cache mémoire.

        ponytail: chargement synchrone — les vignettes sont des fichiers locaux
        livrés avec le plugin, donc instantanées. Si le catalogue passe à des
        vignettes distantes (contrat §2.7), c'est ici qu'il faudra un
        QNetworkAccessManager + un cache disque : 12 cartes = 12 requêtes, et
        l'UI ne doit jamais attendre dessus.
        """
        if not item.thumbnail:
            return QPixmap()
        if item.thumbnail not in self._pixmaps:
            self._pixmaps[item.thumbnail] = QPixmap(
                str(self._catalogue_dir / item.thumbnail))
        return self._pixmaps[item.thumbnail]

    def _relayout_grid(self) -> None:
        while self._grid.count():
            self._grid.takeAt(0)
        width = self._scroll.viewport().width()
        cols = max(1, width // (CARD_W + GRID_HSPACING))
        for i, card in enumerate(self._cards.values()):
            self._grid.addWidget(card, i // cols, i % cols)
        # Les colonnes se partagent la largeur : à 200 px fixes il resterait un
        # trou d'une largeur de carte à droite. La carte porte un plafond
        # (IndiceCard.MAX_W) pour qu'elles ne s'étalent pas sur un grand écran.
        for c in range(cols):
            self._grid.setColumnStretch(c, 1)
        for c in range(cols, cols + 8):        # purge d'une grille précédemment plus large
            self._grid.setColumnStretch(c, 0)

    def eventFilter(self, obj, event):  # noqa: N802 (signature Qt)
        """Le nombre de colonnes suit la largeur RÉELLE du viewport.

        Se contenter du ``resizeEvent`` de l'onglet ne suffit pas : au premier
        affichage le mur est construit avant que la zone de défilement ait sa
        taille définitive, et la grille resterait figée sur deux colonnes.
        """
        if obj is self._scroll.viewport() and event.type() == QEvent.Type.Resize:
            if self._cards:
                self._relayout_grid()
        return super().eventFilter(obj, event)

    def _show_empty(self, message: str) -> None:
        self._empty.setText(message)
        self._empty.setVisible(True)
        self._scroll.setVisible(False)

    # -------------------------------------------------- ouverture QGIS

    def _item_for(self, key: str) -> Optional[CatalogItem]:
        if self._dept is None:
            return None
        for it in self._dept.items:
            if it.key == key:
                return it
        return None

    def _layer_id_for(self, key: str) -> Optional[str]:
        if self._dept is None:
            return None
        wanted = (self._dept.code, key)
        for lid, ref in self._layers.items():
            if ref == wanted:
                return lid
        return None

    def _open_indice(self, key: str) -> None:
        from qgis.core import QgsProject, QgsRasterLayer

        item = self._item_for(key)
        if item is None or self._layer_id_for(key) is not None:
            return
        card = self._cards.get(key)
        dept = self._dept
        if card is not None:
            card.set_state(STATE_LOADING)
        self._liseret.setVisible(True)
        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        QApplication.processEvents()
        try:
            # ponytail: construction sur le thread principal. C'est délibéré —
            # un QgsRasterLayer n'est pas sûr à construire hors thread principal,
            # et les sources actuelles sont des VRT locaux (ouverture immédiate).
            # Pour de vraies sources distantes, la voie propre n'est pas un
            # QgsTask qui fabrique la couche, mais un QgsTask qui ne fait que
            # valider l'URL, la couche restant construite ici.
            name = f"{item.info.sigle} — {dept.name}" if dept else item.info.sigle
            layer = QgsRasterLayer(item.source, name, "gdal")
        finally:
            QApplication.restoreOverrideCursor()
            self._liseret.setVisible(False)

        if not layer.isValid():
            if card is not None:
                card.set_state(STATE_IDLE)
            self._warn(f"Impossible d'ouvrir {item.info.sigle} — {dept.name if dept else ''}. "
                       "La source est peut-être indisponible.")
            return

        first = not self._layers
        QgsProject.instance().addMapLayer(layer)
        self._layers[layer.id()] = (dept.code if dept else "", key)
        if card is not None:
            card.set_state(STATE_LOADED)

        if first:
            self._zoom_to(item, layer)
        if self._iface is not None:
            self._iface.mapCanvas().refresh()
        self._sync_footer()

    def _zoom_to(self, item: CatalogItem, layer) -> None:
        """Recadre sur la PREMIÈRE couche seulement, jamais ensuite.

        L'emprise du catalogue prime sur celle de la couche : une mosaïque VRT
        couvre tout le département alors que les dalles n'en remplissent qu'une
        part — se caler sur ``layer.extent()`` afficherait surtout du vide.
        """
        if self._iface is None:
            return
        canvas = self._iface.mapCanvas()
        try:
            if item.extent:
                from qgis.core import QgsCoordinateReferenceSystem, QgsRectangle
                rect = QgsRectangle(*[float(v) for v in item.extent])
                crs = QgsCoordinateReferenceSystem("EPSG:2154")
                if canvas.mapSettings().destinationCrs() != crs:
                    from qgis.core import QgsCoordinateTransform, QgsProject
                    tr = QgsCoordinateTransform(
                        crs, canvas.mapSettings().destinationCrs(), QgsProject.instance())
                    rect = tr.transformBoundingBox(rect)
                canvas.setExtent(rect)
            else:
                canvas.setExtent(layer.extent())
        except Exception:                        # noqa: BLE001 — un zoom raté n'annule pas l'ajout
            canvas.setExtent(layer.extent())

    def _remove_indice(self, key: str) -> None:
        from qgis.core import QgsProject

        lid = self._layer_id_for(key)
        if lid is None:
            return
        QgsProject.instance().removeMapLayer(lid)
        if self._iface is not None:
            self._iface.mapCanvas().refresh()

    def _remove_all(self) -> None:
        from qgis.core import QgsProject

        ids = list(self._layers)
        if ids:
            QgsProject.instance().removeMapLayers(ids)
        if self._iface is not None:
            self._iface.mapCanvas().refresh()

    def _show_qgis(self) -> None:
        window = self.window()
        if window is not None:
            window.showMinimized()

    def _warn(self, message: str) -> None:
        if self._iface is not None:
            self._iface.messageBar().pushWarning("Archéolog'IA", message)

    # ------------------------------------------------- synchro projet

    def _connect_project(self) -> None:
        try:
            from qgis.core import QgsProject
            QgsProject.instance().layersRemoved.connect(self._on_layers_removed)
        except Exception:                        # noqa: BLE001 — hors QGIS (aperçu isolé)
            pass

    def _on_layers_removed(self, layer_ids) -> None:
        """Une couche retirée depuis le panneau Couches doit rendre la carte cliquable."""
        touched = False
        for lid in layer_ids:
            if self._layers.pop(lid, None) is not None:
                touched = True
        if touched:
            self._sync_cards()
            self._sync_footer()

    def _sync_cards(self) -> None:
        for key, card in self._cards.items():
            card.set_state(STATE_LOADED if self._layer_id_for(key) else STATE_IDLE)

    def _sync_footer(self) -> None:
        n = len(self._layers)
        if n == 0:
            self._footer_state.setText("Aucune couche ouverte dans QGIS")
        else:
            pluriel = "s" if n > 1 else ""
            self._footer_state.setText(
                f"{n} couche{pluriel} ouverte{pluriel} dans QGIS · {self._source_mode()}")
        self._clear_btn.setVisible(n > 0)
        self.layer_count_changed.emit(n)

    def _source_mode(self) -> str:
        """Dit d'où viennent VRAIMENT les couches ouvertes.

        Le catalogue de démonstration pointe des fichiers locaux ; l'écran ne
        doit pas annoncer « flux distant » pendant ce temps. Le jour où le
        catalogue porte des URL de COG, ``streamed`` bascule et le libellé suit
        sans qu'on y retouche.
        """
        distant = False
        for code, key in self._layers.values():
            dept = self._catalogue.by_code(code)
            for item in (dept.items if dept else []):
                if item.key == key and item.streamed:
                    distant = True
        return "flux distant" if distant else "source locale"

    def cleanup(self) -> None:
        """Le plugin est déchargé : on oublie les couches, on n'y touche pas."""
        self._layers.clear()
        try:
            from qgis.core import QgsProject
            QgsProject.instance().layersRemoved.disconnect(self._on_layers_removed)
        except Exception:                        # noqa: BLE001
            pass
