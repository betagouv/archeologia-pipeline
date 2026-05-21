"""Étape 1 — Source. Frise cliquable (= sélecteur de mode) + entrée/sortie.

La frise EST le sélecteur : chaque stade fixe le ``data_mode``. Les stades avant
le point d'entrée sont « sautés », celui d'entrée est mis en avant (badge
« ENTRÉE »), ceux d'après « exécutés ». La logique métier (modes, stades,
validation de chemin) vient du module pur :mod:`app.services.source_modes`.
"""
from __future__ import annotations

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
    ordered_modes,
    path_state,
    pipeline_stages,
)
from ..icons import colored_pixmap
from ..widgets.card import build_card
from ..widgets.stage_button import StageButton


class SourcePage(QWidget):
    """Page « Source » : frise de stades + bandeau de mode + entrée/sortie."""

    mode_changed = pyqtSignal(str)
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = "ign_laz"
        self._source_values: dict = {}  # config_key -> texte (par mode)
        self._loading = False
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
        self._frise_card, fv = build_card()
        frise_row = QHBoxLayout()
        frise_row.setContentsMargins(0, 12, 0, 0)  # place en haut pour le badge ENTRÉE
        frise_row.setSpacing(0)
        self._stage_buttons: dict = {}
        self._connectors: list = []  # (QLabel, right_stage_id)
        for i, st in enumerate(pipeline_stages()):
            if i > 0:
                conn = QLabel("▶")
                conn.setObjectName("StageConnector")
                conn.setAlignment(Qt.AlignCenter)
                conn.setFixedWidth(20)
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
        self._entry_badge.setAlignment(Qt.AlignCenter)
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
        self._qgis_btn = QPushButton("Couche QGIS")
        self._qgis_btn.setObjectName("GhostButton")
        self._qgis_btn.setToolTip("Sélectionner une couche polygone du projet QGIS")
        self._qgis_btn.clicked.connect(self._pick_qgis_layer)
        row1.addWidget(self._source_edit, 1)
        row1.addWidget(self._browse_btn)
        row1.addWidget(self._qgis_btn)
        iv.addWidget(self._source_label)
        iv.addLayout(row1)

        out_label = QLabel("Dossier de sortie")
        out_label.setObjectName("FieldLabel")
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("Chemin du dossier de sortie des résultats")
        self._output_edit.textChanged.connect(self._on_output_changed)
        out_browse = QPushButton("Parcourir…")
        out_browse.clicked.connect(self._browse_output)
        row2.addWidget(self._output_edit, 1)
        row2.addWidget(out_browse)
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
        self._banner_icon.setAlignment(Qt.AlignCenter)
        text = QVBoxLayout()
        text.setSpacing(1)
        self._banner_title = QLabel("")
        self._banner_title.setObjectName("ModeBannerTitle")
        self._banner_desc = QLabel("")
        self._banner_desc.setObjectName("ModeBannerText")
        self._banner_desc.setWordWrap(True)
        text.addWidget(self._banner_title)
        text.addWidget(self._banner_desc)
        layout.addWidget(self._banner_icon, 0, Qt.AlignTop)
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
            conn.setProperty("active", right_id >= entry)
            conn.style().unpolish(conn)
            conn.style().polish(conn)

        self._banner_icon.setPixmap(colored_pixmap(info.icon, "#ffffff", 18))
        self._banner_title.setText(
            f"<b>{info.banner_label}</b>  ·  <span style='color:#5a5a5a;'>{info.banner_sub}</span>"
        )
        self._banner_desc.setText(info.description)
        self._source_label.setText(info.source_label)
        self._source_edit.setPlaceholderText(info.placeholder)
        self._qgis_btn.setVisible(self._mode == "ign_laz")

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
                "Sélectionner le polygone de la zone d'étude",
                "",
                "Vecteurs (*.shp *.geojson *.json *.gpkg *.txt);;Tous (*.*)",
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
        layers = [
            lyr
            for lyr in project.mapLayers().values()
            if hasattr(lyr, "geometryType") and lyr.geometryType() == QgsWkbTypes.PolygonGeometry
        ]
        if not layers:
            QMessageBox.information(
                self,
                "Aucune couche polygone",
                "Aucune couche polygone n'est chargée dans le projet QGIS.",
            )
            return
        names = [f"{lyr.name()}  ({lyr.featureCount()} entités)" for lyr in layers]
        chosen, ok = QInputDialog.getItem(self, "Couche polygone", "Couche :", names, 0, False)
        if not ok:
            return
        layer = layers[names.index(chosen)]

        source = layer.source().split("|")[0].strip()
        p = Path(source)
        if p.suffix.lower() == ".dbf" and p.with_suffix(".shp").exists():
            source = str(p.with_suffix(".shp"))
            p = Path(source)
        if p.exists() and p.suffix.lower() in (".shp", ".geojson", ".json", ".gpkg"):
            self._source_edit.setText(source)
            return

        export_dir = Path(__file__).resolve().parents[3] / "data" / "temp_zones"
        export_dir.mkdir(parents=True, exist_ok=True)
        tmp_shp = export_dir / f"{layer.name().replace(' ', '_')}.shp"
        save_options = QgsVectorFileWriter.SaveVectorOptions()
        save_options.driverName = "ESRI Shapefile"
        error = QgsVectorFileWriter.writeAsVectorFormatV3(
            layer, str(tmp_shp), project.transformContext(), save_options
        )
        if error[0] != QgsVectorFileWriter.NoError:
            QMessageBox.warning(
                self,
                "Erreur d'export",
                f"Impossible d'exporter la couche « {layer.name()} ».\n{error[1]}",
            )
            return
        self._source_edit.setText(str(tmp_shp))

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
