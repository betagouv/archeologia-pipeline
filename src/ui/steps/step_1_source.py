"""Étape 1 — Source. Frise cliquable (= sélecteur de mode) + entrée/sortie.

La frise EST le sélecteur : chaque stade fixe le ``data_mode``. Les stades avant
le point d'entrée sont « sautés », celui d'entrée est mis en avant (badge
« ENTRÉE »), ceux d'après « exécutés ». La logique métier (modes, stades,
validation de chemin) vient du module pur :mod:`app.services.source_modes`.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from qgis.PyQt.QtCore import QPoint, Qt, QTimer, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...app.services.source_modes import (
    mode_info,
    normalize_vector_input,
    ordered_modes,
    path_state,
    pipeline_stages,
)
from ..icons import colored_pixmap
from ..widgets.card import build_card
from ..widgets.stage_button import ArrowConnector, StageButton

logger = logging.getLogger(__name__)

# Au-delà de ce nombre de dalles, on demande confirmation (téléchargement lourd).
_LARGE_SELECTION_THRESHOLD = 50


class SourcePage(QWidget):
    """Page « Source » : frise de stades + bandeau de mode + entrée/sortie."""

    mode_changed = pyqtSignal(str)
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = "ign_laz"
        self._source_values: dict = {}  # config_key -> texte (par mode)
        self._loading = False
        # État de la sélection des dalles sur le canevas (mode ign_laz).
        self._tile_tool = None
        self._grid_layer = None
        self._prev_map_tool = None
        self._msg_item = None
        self._validate_btn = None
        # Bandeau persistant « zoomez pour voir les dalles » + canevas connecté à
        # ``scaleChanged`` (déconnecté au teardown pour ne pas fuiter le signal).
        self._scale_msg_item = None
        self._scale_canvas = None
        self._build()
        self._apply_mode_ui()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        heading = QLabel("Où voulez-vous démarrer le pipeline ?")
        heading.setObjectName("WizardPageHeading")
        root.addWidget(heading)
        sub = QLabel(
            "Cliquez sur l'étape où vous voulez injecter vos données. "
            "Les étapes amont sont sautées."
        )
        sub.setObjectName("WizardPageSub")
        sub.setWordWrap(True)
        root.addWidget(sub)

        # ── Carte frise + bandeau ──
        # Numérotée « 1 » : la carte « Entrée & sortie » porte le « 2 », la
        # séquence visuelle doit être complète sur la page.
        self._frise_card, fv = build_card("Point d'entrée du pipeline", "1")
        frise_row = QHBoxLayout()
        frise_row.setContentsMargins(0, 12, 0, 0)  # place en haut pour le badge ENTRÉE
        frise_row.setSpacing(0)
        self._stage_buttons: dict = {}
        self._connectors: list = []  # (ArrowConnector, right_stage_id)
        for i, st in enumerate(pipeline_stages()):
            if i > 0:
                conn = ArrowConnector()
                frise_row.addWidget(conn)
                self._connectors.append((conn, st.id))
            btn = StageButton(
                st.icon, st.label, st.sub, clickable=(st.mode is not None), optional=st.optional
            )
            if st.mode is not None:
                btn.clicked.connect(lambda m=st.mode: self._on_mode_clicked(m))
            self._stage_buttons[st.id] = btn
            frise_row.addWidget(btn, 1)
        fv.addLayout(frise_row)
        fv.addWidget(self._build_banner())
        root.addWidget(self._frise_card)

        # Badge « ENTRÉE » flottant — overlay sur la bordure haute du stade d'entrée.
        self._entry_badge = QLabel("ENTRÉE", self._frise_card)
        self._entry_badge.setObjectName("StageBadge")
        self._entry_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._entry_badge.hide()

        # ── Carte entrée & sortie ──
        io_card, iv = build_card("Entrée & sortie", "2")
        self._source_label = QLabel("")
        self._source_label.setObjectName("FieldLabel")
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        self._source_edit = QLineEdit()
        self._source_edit.textChanged.connect(self._on_source_text_changed)
        self._browse_btn = QPushButton("Parcourir…")
        self._browse_btn.clicked.connect(self._browse_source)
        self._qgis_btn = QPushButton("Couche / groupe QGIS")
        self._qgis_btn.setObjectName("GhostButton")
        self._qgis_btn.setToolTip(
            "Sélectionner une couche ou un groupe de couches du projet QGIS "
            "(points, lignes ou polygones)"
        )
        self._qgis_btn.clicked.connect(self._pick_qgis_layer)
        self._dalles_btn = QPushButton("Sélectionner les dalles")
        self._dalles_btn.setObjectName("GhostButton")
        self._dalles_btn.setToolTip("Choisir les dalles IGN directement sur la carte")
        self._dalles_btn.clicked.connect(self._pick_dalles_on_canvas)
        row1.addWidget(self._source_edit, 1)
        row1.addWidget(self._browse_btn)
        row1.addWidget(self._qgis_btn)
        row1.addWidget(self._dalles_btn)
        iv.addWidget(self._source_label)
        iv.addLayout(row1)

        out_label = QLabel("Dossier de sortie")
        out_label.setObjectName("FieldLabel")
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("Chemin du dossier de sortie des résultats")
        self._output_edit.textChanged.connect(self._on_output_changed)
        self._out_browse_btn = QPushButton("Parcourir…")
        self._out_browse_btn.clicked.connect(self._browse_output)
        row2.addWidget(self._output_edit, 1)
        row2.addWidget(self._out_browse_btn)
        iv.addWidget(out_label)
        iv.addLayout(row2)

        root.addWidget(io_card)
        root.addStretch(1)

    def _build_banner(self) -> QWidget:
        self._banner = QFrame()
        self._banner.setObjectName("ModeBanner")
        layout = QHBoxLayout(self._banner)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)
        self._banner_icon = QLabel("")
        self._banner_icon.setObjectName("ModeBannerIcon")
        self._banner_icon.setFixedSize(28, 28)
        self._banner_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text = QVBoxLayout()
        text.setSpacing(1)
        # Titre + complément en deux QLabel stylés par le QSS (pas de couleur
        # en dur dans du rich text, qui échapperait au thème).
        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        self._banner_title = QLabel("")
        self._banner_title.setObjectName("ModeBannerTitle")
        self._banner_sub = QLabel("")
        self._banner_sub.setObjectName("ModeBannerSub")
        title_row.addWidget(self._banner_title)
        title_row.addWidget(self._banner_sub)
        title_row.addStretch(1)
        self._banner_desc = QLabel("")
        self._banner_desc.setObjectName("ModeBannerText")
        self._banner_desc.setWordWrap(True)
        text.addLayout(title_row)
        text.addWidget(self._banner_desc)
        layout.addWidget(self._banner_icon, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(text, 1)
        return self._banner

    # ------------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------------
    def _on_mode_clicked(self, mode: str) -> None:
        if mode == self._mode:
            return
        self._stash_current_source()
        self._mode = mode
        self._apply_mode_ui()
        if not self._loading:
            self.mode_changed.emit(mode)
            self.changed.emit()

    def _stash_current_source(self) -> None:
        self._source_values[mode_info(self._mode).config_key] = self._source_edit.text().strip()

    def _apply_mode_ui(self) -> None:
        info = mode_info(self._mode)
        entry = info.entry_stage
        for st in pipeline_stages():
            if st.id == entry:
                role = "entry"
            elif st.id > entry:
                role = "executed"
            else:
                role = "skipped"
            self._stage_buttons[st.id].set_role(role)
        for conn, right_id in self._connectors:
            conn.set_active(right_id >= entry)

        self._banner_icon.setPixmap(
            colored_pixmap(info.icon, "#ffffff", 20, dpr=self.devicePixelRatioF())
        )
        self._banner_title.setText(info.banner_label)
        self._banner_sub.setText(f"· {info.banner_sub}")
        self._banner_desc.setText(info.description)
        self._source_label.setText(info.source_label)
        self._source_edit.setPlaceholderText(info.placeholder)
        self._qgis_btn.setVisible(self._mode == "ign_laz")
        self._dalles_btn.setVisible(self._mode == "ign_laz")
        if self._mode != "ign_laz":
            # Changer de mode pendant une sélection en cours : on ferme proprement.
            self.cancel_dalles_selection_if_active()

        prev = self._loading
        self._loading = True
        try:
            self._source_edit.setText(self._source_values.get(info.config_key, ""))
        finally:
            self._loading = prev
        self._refresh_validation()
        QTimer.singleShot(0, self._reposition_entry_badge)

    def _reposition_entry_badge(self) -> None:
        """Place le badge « ENTRÉE » à cheval sur la bordure haute du stade d'entrée."""
        badge = getattr(self, "_entry_badge", None)
        if badge is None:
            return
        btn = self._stage_buttons.get(mode_info(self._mode).entry_stage)
        if btn is None:
            badge.hide()
            return
        badge.adjustSize()
        top_left = btn.mapTo(self._frise_card, QPoint(0, 0))
        x = top_left.x() + (btn.width() - badge.width()) // 2
        y = top_left.y() - badge.height() // 2
        badge.move(x, max(0, y))
        badge.raise_()
        badge.show()

    def resizeEvent(self, event):  # noqa: N802 (signature Qt)
        super().resizeEvent(event)
        # Appel synchrone : pendant un drag continu, le badge doit suivre la
        # bulle dans la même frame (le timer seul le faisait « traîner »).
        self._reposition_entry_badge()
        QTimer.singleShot(0, self._reposition_entry_badge)

    def showEvent(self, event):  # noqa: N802 (signature Qt)
        super().showEvent(event)
        QTimer.singleShot(0, self._reposition_entry_badge)

    # ------------------------------------------------------------------
    # Browse
    # ------------------------------------------------------------------
    def _browse_source(self) -> None:
        info = mode_info(self._mode)
        if info.is_file:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Sélectionner la zone d'étude (polygone ou points)",
                "",
                "Vecteurs (*.shp *.dbf *.geojson *.json *.gpkg *.txt);;Tous (*.*)",
            )
        else:
            path = QFileDialog.getExistingDirectory(self, "Sélectionner un dossier")
        if path:
            self._source_edit.setText(path)

    def _browse_output(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Sélectionner le dossier de sortie")
        if directory:
            self._output_edit.setText(directory)

    def _pick_qgis_layer(self) -> None:
        try:
            from qgis.core import QgsProject, QgsVectorFileWriter, QgsWkbTypes
            from qgis.PyQt.QtWidgets import QInputDialog
        except ImportError:
            QMessageBox.warning(self, "Erreur", "API QGIS non disponible.")
            return

        project = QgsProject.instance()
        # La résolution des dalles (union + filtre spatial) accepte toute
        # géométrie : points/lignes/polygones. Seules les tables sans
        # géométrie (ex. .dbf chargé seul) sont exclues.
        geom_types = (
            QgsWkbTypes.GeometryType.PointGeometry,
            QgsWkbTypes.GeometryType.LineGeometry,
            QgsWkbTypes.GeometryType.PolygonGeometry,
        )

        def _usable(lyr) -> bool:
            return lyr is not None and hasattr(lyr, "geometryType") and lyr.geometryType() in geom_types

        # Entrées proposées : (libellé, nom de base, couches). Les groupes du
        # panneau Couches d'abord (leurs couches seront unionnées), puis les
        # couches individuelles.
        entries = []
        for grp in project.layerTreeRoot().findGroups(True):
            grp_layers = [tl.layer() for tl in grp.findLayers() if _usable(tl.layer())]
            if grp_layers:
                entries.append(
                    (f"📁 {grp.name()}  ({len(grp_layers)} couches)", grp.name(), grp_layers)
                )
        for lyr in project.mapLayers().values():
            if _usable(lyr):
                entries.append((f"{lyr.name()}  ({lyr.featureCount()} entités)", lyr.name(), [lyr]))

        if not entries:
            QMessageBox.information(
                self,
                "Aucune couche vecteur",
                "Aucune couche vecteur avec géométrie n'est chargée dans le projet QGIS.",
            )
            return
        names = [label for label, _, _ in entries]
        chosen, ok = QInputDialog.getItem(
            self, "Couche ou groupe", "Zone d'étude :", names, 0, False
        )
        if not ok:
            return
        _, base_name, layers = entries[names.index(chosen)]

        # Couche unique déjà sur disque : on la pointe directement, pas de copie.
        if len(layers) == 1:
            p = normalize_vector_input(Path(layers[0].source().split("|")[0].strip()))
            if p.exists() and p.suffix.lower() in (".shp", ".geojson", ".json", ".gpkg"):
                self._source_edit.setText(str(p))
                return

        # Sinon (groupe, couche mémoire, couche distante) : export dans un seul
        # GeoPackage multi-couches — resolve_tiles_from_polygon unionne toutes
        # les couches du fichier, chacune avec son propre CRS.
        export_dir = self._plugin_root() / "data" / "temp_zones"
        export_dir.mkdir(parents=True, exist_ok=True)
        out = export_dir / f"zone_{re.sub(r'[^A-Za-z0-9_-]', '_', base_name)}.gpkg"
        try:
            out.unlink(missing_ok=True)  # sinon les anciennes couches s'empilent
        except OSError as e:
            QMessageBox.warning(
                self,
                "Fichier verrouillé",
                f"Impossible de remplacer {out.name} : {e}\n"
                "Déchargez cette couche du projet QGIS puis réessayez.",
            )
            return

        used_names = set()
        for i, lyr in enumerate(layers):
            # Deux couches homonymes dans le groupe s'écraseraient dans le GPKG.
            layer_name = lyr.name()
            while layer_name in used_names:
                layer_name = f"{lyr.name()}_{len(used_names)}"
            used_names.add(layer_name)

            save_options = QgsVectorFileWriter.SaveVectorOptions()
            save_options.driverName = "GPKG"
            save_options.layerName = layer_name
            if i:
                save_options.actionOnExistingFile = (
                    QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteLayer
                )
            error = QgsVectorFileWriter.writeAsVectorFormatV3(
                lyr, str(out), project.transformContext(), save_options
            )
            if error[0] != QgsVectorFileWriter.WriterError.NoError:
                QMessageBox.warning(
                    self,
                    "Erreur d'export",
                    f"Impossible d'exporter la couche « {lyr.name()} ».\n{error[1]}",
                )
                return
        self._source_edit.setText(str(out))

    # ------------------------------------------------------------------
    # Sélection des dalles IGN directement sur le canevas (mode ign_laz)
    # ------------------------------------------------------------------
    def _plugin_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def _pick_dalles_on_canvas(self) -> None:
        """Active l'outil de sélection des dalles sur la carte QGIS."""
        try:
            from qgis.core import Qgis
            from qgis.utils import iface

            from ..map_tools.grid_layer import (
                load_quadrillage_layer,
                zoom_to_france_metro_if_hidden,
            )
            from ..map_tools.tile_picker_tool import TilePickerMapTool
        except ImportError:
            QMessageBox.warning(self, "Erreur", "API QGIS non disponible.")
            return
        if iface is None or iface.mapCanvas() is None:
            QMessageBox.warning(self, "Erreur", "Canevas QGIS indisponible.")
            return
        if self._tile_tool is not None:
            # Auto-réparation : si l'outil est posé mais plus actif sur le canevas
            # (teardown précédent avorté), on nettoie l'état périmé puis on continue
            # — sinon le bouton resterait bloqué jusqu'à un redémarrage de QGIS.
            if iface.mapCanvas().mapTool() is self._tile_tool:
                return  # réellement actif
            logger.warning("Sélection dalles : état périmé (_tile_tool inactif) → nettoyage")
            self._teardown_dalles_selection()

        layer = load_quadrillage_layer(self._plugin_root())
        if layer is None:
            QMessageBox.warning(
                self,
                "Quadrillage introuvable",
                "Le quadrillage IGN (data/quadrillage_france/) est introuvable ou "
                "invalide. Placez le shapefile TA_diff_pkk_lidarhd_classe.shp (et son "
                "index .qix) dans ce dossier.",
            )
            return
        layer.removeSelection()
        zoom_to_france_metro_if_hidden(layer)  # cadre la métropole si la vue est perdue (U1)

        canvas = iface.mapCanvas()
        self._grid_layer = layer
        self._prev_map_tool = canvas.mapTool()
        tool = TilePickerMapTool(canvas, layer)
        tool.selection_changed.connect(self._on_tiles_selection_changed)
        tool.cancelled.connect(self._on_tiles_cancel)
        tool.too_zoomed_out.connect(self._on_too_zoomed_out)
        self._tile_tool = tool
        canvas.setMapTool(tool)

        # Barre de message QGIS : reste visible même si le dialogue est réduit.
        bar = iface.messageBar()
        widget = bar.createMessage(
            "Sélection des dalles IGN",
            "Cliquez une dalle (bascule), encadrez pour ajouter, Ctrl+encadré pour retirer. "
            "Échap pour annuler.",
        )
        self._validate_btn = QPushButton("Valider (0 dalle)", widget)
        self._validate_btn.clicked.connect(self._on_tiles_validate)
        clear_btn = QPushButton("Tout effacer", widget)
        clear_btn.clicked.connect(self._on_clear_selection)
        cancel_btn = QPushButton("Annuler", widget)
        cancel_btn.clicked.connect(self._on_tiles_cancel)
        widget.layout().addWidget(self._validate_btn)
        widget.layout().addWidget(clear_btn)
        widget.layout().addWidget(cancel_btn)
        bar.pushWidget(widget, Qgis.MessageLevel.Info)
        self._msg_item = widget

        # Bandeau persistant + suivi de l'échelle : tant que la grille est masquée
        # (vue trop dézoomée), un avertissement reste affiché et se met à jour ;
        # il disparaît dès que la grille redevient visible.
        self._scale_canvas = canvas
        canvas.scaleChanged.connect(self._on_canvas_scale_changed)
        self._refresh_scale_notice()

        self._dalles_btn.setEnabled(False)
        # Minimiser le dialogue du plugin pour dégager le canevas. NE PAS utiliser
        # lower() : sur Windows, abaisser une fenêtre possédée par la fenêtre QGIS
        # minimise QGIS lui-même. showMinimized() ne réduit que le dialogue.
        try:
            self.window().showMinimized()
        except Exception:
            pass

    def _on_tiles_selection_changed(self, n: int) -> None:
        if self._validate_btn is not None:
            from ...app.services.tile_selection import estimate_download_size

            est = estimate_download_size(n)
            suffix = f" {est}" if est else ""
            self._validate_btn.setText(f"Valider ({n} dalle{'s' if n > 1 else ''}{suffix})")

    def _on_clear_selection(self) -> None:
        if self._grid_layer is not None:
            self._grid_layer.removeSelection()
            self._on_tiles_selection_changed(0)

    def _on_too_zoomed_out(self) -> None:
        # Clic/encadré tenté alors que la grille est masquée (vue trop dézoomée).
        try:
            from qgis.core import Qgis
            from qgis.utils import iface

            if iface is not None:
                iface.messageBar().pushMessage(
                    "Zoomez davantage pour afficher et sélectionner les dalles",
                    level=Qgis.MessageLevel.Warning,
                    duration=3,
                )
        except Exception:
            pass

    def _on_canvas_scale_changed(self, *_args) -> None:
        # ``QgsMapCanvas.scaleChanged`` émet l'échelle ; on n'a besoin que de l'état.
        self._refresh_scale_notice()

    def _refresh_scale_notice(self) -> None:
        """Affiche / actualise / retire le bandeau « zoomez pour voir les dalles ».

        Persistant tant que la grille est masquée (vue trop dézoomée) ; le texte suit
        l'échelle courante ; il disparaît dès que la grille redevient visible. On ne
        re-pousse qu'à l'apparition (sinon ``setText`` sur l'item existant).
        """
        try:
            from qgis.core import Qgis
            from qgis.utils import iface

            from ...app.services.grid_view import grid_is_hidden
        except ImportError:
            return
        layer = self._grid_layer
        canvas = self._scale_canvas
        if iface is None or layer is None or canvas is None:
            return
        hidden = grid_is_hidden(
            layer.hasScaleBasedVisibility(), canvas.scale(), layer.minimumScale()
        )
        bar = iface.messageBar()
        if hidden:
            cur = f"{int(canvas.scale()):,}".replace(",", " ")
            req = f"{int(layer.minimumScale()):,}".replace(",", " ")
            text = f"Zoomez pour afficher les dalles — échelle 1:{cur} (requise ≤ 1:{req})"
            if self._scale_msg_item is None:
                item = bar.createMessage("Sélection des dalles", text)
                bar.pushWidget(item, Qgis.MessageLevel.Warning)
                self._scale_msg_item = item
            else:
                try:
                    self._scale_msg_item.setText(text)
                except Exception:
                    pass
        elif self._scale_msg_item is not None:
            try:
                bar.popWidget(self._scale_msg_item)
            except Exception:
                pass
            self._scale_msg_item = None

    def _on_tiles_validate(self) -> None:
        layer = self._grid_layer
        if layer is None:
            self._defer_teardown()
            return
        ids = layer.selectedFeatureIds()
        if not ids:
            QMessageBox.information(
                self, "Aucune dalle", "Sélectionnez au moins une dalle (clic ou encadré)."
            )
            return  # rester en mode sélection
        from qgis.core import QgsFeatureRequest

        from ...app.services.tile_selection import estimate_download_size, format_dalles_urls

        n = len(ids)
        if n > _LARGE_SELECTION_THRESHOLD:
            resp = QMessageBox.question(
                self,
                "Téléchargement volumineux",
                f"{n} dalles sélectionnées ({estimate_download_size(n)}). "
                "Lancer un tel téléchargement ?",
            )
            if resp != QMessageBox.StandardButton.Yes:
                return  # rester en mode sélection

        # Lecture des entités + écriture du fichier : si ça échoue (champ illisible,
        # dossier non inscriptible…), on NE déclenche PAS le teardown — la sélection
        # est conservée et « Annuler » reste la porte de sortie (qui réactive le
        # bouton). Sans ce garde-fou, un échec laissait l'outil/le bouton bloqués
        # jusqu'au redémarrage de QGIS.
        try:
            tiles = [
                (
                    str(f["nom_pkk"]) if f["nom_pkk"] else "",
                    str(f["url_telech"]) if f["url_telech"] else "",
                )
                for f in layer.getFeatures(QgsFeatureRequest().setFilterFids(ids))
            ]
            export_dir = self._plugin_root() / "data" / "temp_zones"
            export_dir.mkdir(parents=True, exist_ok=True)
            out_path = export_dir / "dalles_selection.txt"
            out_path.write_text(format_dalles_urls(tiles), encoding="utf-8")
        except Exception:
            logger.exception("Échec de l'enregistrement de la sélection (n=%s)", n)
            QMessageBox.warning(
                self,
                "Échec de l'enregistrement",
                "Impossible d'enregistrer la sélection des dalles. La sélection est "
                "conservée — réessayez ou cliquez « Annuler ».",
            )
            return  # rester en mode sélection
        try:
            from qgis.core import Qgis
            from qgis.utils import iface

            if iface is not None:
                iface.messageBar().pushMessage(
                    f"{n} dalle(s) enregistrée(s)", level=Qgis.MessageLevel.Success, duration=4
                )
        except Exception:
            pass
        self._defer_teardown()
        self._source_edit.setText(str(out_path))

    def _on_tiles_cancel(self) -> None:
        self._defer_teardown()

    def _defer_teardown(self) -> None:
        # Différé : on est dans le slot d'un bouton de la barre de message ; la
        # détruire de façon synchrone (popWidget) supprimerait l'émetteur courant.
        QTimer.singleShot(0, self._teardown_dalles_selection)

    def cancel_dalles_selection_if_active(self) -> None:
        """Ferme proprement une sélection en cours (fermeture/unload/run/mode).

        Synchrone (pas appelée depuis le slot d'un widget de la barre de message)
        → nécessaire pour un teardown immédiat à la fermeture/l'unload.
        """
        if (
            self._tile_tool is not None
            or self._grid_layer is not None
            or self._msg_item is not None
            or self._scale_msg_item is not None
            or self._scale_canvas is not None
        ):
            self._teardown_dalles_selection()

    def _teardown_dalles_selection(self) -> None:
        """Restaure l'outil-carte, retire la barre de message et la couche. Idempotent."""
        try:
            from qgis.utils import iface
        except ImportError:
            iface = None
        canvas = iface.mapCanvas() if iface is not None else None
        if canvas is not None and self._tile_tool is not None:
            if self._prev_map_tool is not None:
                canvas.setMapTool(self._prev_map_tool)
            else:
                canvas.unsetMapTool(self._tile_tool)
        if self._tile_tool is not None:
            try:
                self._tile_tool.cleanup()
            except Exception:
                pass
        if iface is not None and self._msg_item is not None:
            try:
                iface.messageBar().popWidget(self._msg_item)
            except Exception:
                pass
        # Déconnecter le suivi d'échelle (sinon le signal fuiterait d'une session de
        # sélection à l'autre) et retirer le bandeau persistant.
        if self._scale_canvas is not None:
            try:
                self._scale_canvas.scaleChanged.disconnect(self._on_canvas_scale_changed)
            except (TypeError, RuntimeError):
                pass  # déjà déconnecté / objet C++ détruit
        if iface is not None and self._scale_msg_item is not None:
            try:
                iface.messageBar().popWidget(self._scale_msg_item)
            except Exception:
                pass
        try:
            from ..map_tools.grid_layer import remove_quadrillage_layer

            remove_quadrillage_layer()
        except Exception:
            pass
        self._tile_tool = None
        self._grid_layer = None
        self._prev_map_tool = None
        self._msg_item = None
        self._validate_btn = None
        self._scale_msg_item = None
        self._scale_canvas = None
        # Ne ré-activer le bouton que hors lecture seule (run en cours).
        if not self._source_edit.isReadOnly():
            self._dalles_btn.setEnabled(True)
        # Restaurer le dialogue minimisé (showNormal dé-minimise ; raise le ramène).
        try:
            win = self.window()
            win.showNormal()
            win.raise_()
            win.activateWindow()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Validation (bordure : ok / warn / error)
    # ------------------------------------------------------------------
    def _on_source_text_changed(self) -> None:
        self._refresh_source_validation()
        if not self._loading:
            self._stash_current_source()
            self.changed.emit()

    def _on_output_changed(self) -> None:
        self._refresh_output_validation()
        if not self._loading:
            self.changed.emit()

    def _refresh_validation(self) -> None:
        self._refresh_source_validation()
        self._refresh_output_validation()

    def _refresh_source_validation(self) -> None:
        info = mode_info(self._mode)
        state = path_state(
            self._source_edit.text(), expect_dir=not info.is_file, valid_exts=info.valid_exts
        )
        self._set_state(self._source_edit, state)

    def _refresh_output_validation(self) -> None:
        state = path_state(self._output_edit.text(), expect_dir=True, allow_create=True)
        self._set_state(self._output_edit, state)

    @staticmethod
    def _set_state(edit: QLineEdit, state: str) -> None:
        edit.setProperty("state", state)
        edit.style().unpolish(edit)
        edit.style().polish(edit)

    # ------------------------------------------------------------------
    # Persistance
    # ------------------------------------------------------------------
    def current_mode(self) -> str:
        return self._mode

    def summary(self) -> str:
        """Résumé court pour le sous-libellé du rail (étape Source)."""
        return mode_info(self._mode).banner_label

    def set_readonly(self, ro: bool) -> None:
        """Verrouille la saisie pour consultation pendant un run (lecture seule).

        N'agit que sur ``setEnabled``/``setReadOnly`` (jamais ``setText``/…),
        donc n'émet aucun signal ``changed`` et ne déclenche pas l'autosave.
        """
        if ro:
            # Un run démarre : on referme une sélection de dalles en cours
            # AVANT de désactiver le bouton (sinon il resterait actif).
            self.cancel_dalles_selection_if_active()
        self._source_edit.setReadOnly(ro)
        self._output_edit.setReadOnly(ro)
        for btn in (self._browse_btn, self._qgis_btn, self._dalles_btn, self._out_browse_btn):
            btn.setEnabled(not ro)
        for btn in self._stage_buttons.values():
            btn.setEnabled(not ro)

    def load_from(self, config: dict) -> None:
        files = (config.get("app") or {}).get("files") or {}
        prev = self._loading
        self._loading = True
        try:
            for mode in ordered_modes():
                key = mode_info(mode).config_key
                self._source_values[key] = str(files.get(key) or "")
            self._mode = str(files.get("data_mode") or "ign_laz")
            if self._mode not in [s.mode for s in pipeline_stages() if s.mode]:
                self._mode = "ign_laz"
            self._output_edit.setText(str(files.get("output_dir") or ""))
            self._apply_mode_ui()
        finally:
            self._loading = prev

    def collect_into(self, config: dict) -> None:
        self._stash_current_source()
        files = config.setdefault("app", {}).setdefault("files", {})
        files["data_mode"] = self._mode
        for mode in ordered_modes():
            key = mode_info(mode).config_key
            files[key] = self._source_values.get(key, "")
        files["output_dir"] = self._output_edit.text().strip()
