"""MainDialog avec Mode Simple / Expert, connecté au pipeline."""

import json
import logging
import sys
import threading
from pathlib import Path
from typing import Optional

from qgis.PyQt.QtCore import Qt, QObject, pyqtSignal
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QSizePolicy,
)

from ..config.config_manager import ConfigManager


class _QtLogEmitter(QObject):
    message = pyqtSignal(str)
    progress = pyqtSignal(int)
    stage = pyqtSignal(str)
    run_enabled = pyqtSignal(bool)
    load_layers = pyqtSignal(list, list, list)  # (vrt_paths, shapefile_paths, class_colors)


class QtLogHandler(logging.Handler):
    def __init__(self, emitter: _QtLogEmitter):
        super().__init__()
        self._emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        self._emitter.message.emit(msg)


# ──────────────────────────────────────────────
# Widgets utilitaires (repris de l'existant)
# ──────────────────────────────────────────────

class NoWheelSpinBox(QSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

    def wheelEvent(self, event):
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

    def wheelEvent(self, event):
        event.ignore()


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


# ──────────────────────────────────────────────
# Produits disponibles
# ──────────────────────────────────────────────

PRODUCTS = [
    ("MNT",     "MNT",     True,  False),
    ("DENSITE", "DENSITE", False, False),
    ("M_HS",    "M-HS",    False, True),
    ("SVF",     "SVF",     False, True),
    ("SLO",     "SLO",     False, True),
    ("LD",      "LD",      False, True),
    ("SLRM",    "SLRM",    False, True),
    ("VAT",     "VAT",     False, True),
]


# ──────────────────────────────────────────────
# Dialog principal
# ──────────────────────────────────────────────

class MainDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Archéolog'IA")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMinimizeButtonHint)

        # ── Infrastructure ──
        self._loading = False
        self._plugin_root = Path(__file__).resolve().parents[2]
        self._config_manager = ConfigManager(self._plugin_root)
        self._config = self._config_manager.load_last_ui_config()
        self._current_mode = None
        self._cancel_event = threading.Event()

        # Icône de la fenêtre = icône du plugin
        _icon_path = self._plugin_root / "data" / "icon.png"
        if _icon_path.exists():
            self.setWindowIcon(QIcon(str(_icon_path)))
        self.setMinimumSize(850, 650)
        self.resize(850, 700)

        # ── Logger ──
        self._logger = logging.getLogger("archeologia_pipeline")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

        self._log_emitter = _QtLogEmitter()
        self._log_emitter.message.connect(self._append_log)
        self._log_emitter.progress.connect(self._set_progress)
        self._log_emitter.stage.connect(self._set_stage)
        self._log_emitter.run_enabled.connect(self._set_run_enabled)
        self._log_emitter.load_layers.connect(self._load_layers_to_project)
        self._qt_log_handler = QtLogHandler(self._log_emitter)
        self._qt_log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        if not any(isinstance(h, QtLogHandler) for h in self._logger.handlers):
            self._logger.addHandler(self._qt_log_handler)

        # ── Layout racine ──
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        # ── Toggle Mode Simple / Expert + Config ──
        top_row = QWidget()
        top_layout = QHBoxLayout(top_row)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(QLabel("Mode :"))
        self.mode_combo = NoWheelComboBox()
        self.mode_combo.addItem("Simple", "simple")
        self.mode_combo.addItem("Expert", "expert")
        self.mode_combo.setToolTip(
            "Mode Simple : configuration minimale, les paramètres avancés utilisent les valeurs par défaut.\n"
            "Mode Expert : accès à tous les paramètres (résolutions, seuils, workers, etc.)."
        )
        self.mode_combo.currentIndexChanged.connect(self._apply_mode)
        top_layout.addWidget(self.mode_combo)
        top_layout.addStretch(1)

        self.load_config_btn = QPushButton("Charger config")
        self.load_config_btn.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_DialogOpenButton))
        self.load_config_btn.setToolTip("Charger un profil de configuration sauvegardé (.json)")
        self.load_config_btn.clicked.connect(self._load_config)
        top_layout.addWidget(self.load_config_btn)

        self.save_config_btn = QPushButton("Sauvegarder config")
        self.save_config_btn.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_DialogSaveButton))
        self.save_config_btn.setToolTip("Sauvegarder la configuration actuelle dans un fichier (.json)")
        self.save_config_btn.clicked.connect(self._save_config)
        top_layout.addWidget(self.save_config_btn)

        root_layout.addWidget(top_row)

        # ── Splitter : config (haut) / logs (bas) ──
        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.setChildrenCollapsible(False)

        # --- Partie haute : configuration (scrollable) ---
        self._config_scroll = config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        config_scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        config_scroll.setMinimumHeight(80)
        config_container = QWidget()
        self._config_layout = QVBoxLayout(config_container)
        self._config_layout.setContentsMargins(0, 0, 0, 0)
        self._config_layout.setSpacing(8)
        config_scroll.setWidget(config_container)

        self._build_sources_section()
        self._build_products_section()
        self._build_mnt_section()
        self._build_rvt_section()
        self._build_detection_section()
        self._build_performance_section()

        self._splitter.addWidget(config_scroll)

        # --- Partie basse : logs ---
        self._build_logs_section()

        root_layout.addWidget(self._splitter, 1)

        # ── Boutons d'action ──
        self._build_buttons_row(root_layout)

        # ── Titres de section en gras + bleu ──
        for group in self.findChildren(QGroupBox):
            self._make_bold_title(group)

        # ── Charger la configuration dans les widgets ──
        self._load_into_widgets()
        self._wire_autosave()
        self._apply_data_mode_state()

        # ── Scanner les modèles disponibles ──
        self._refresh_models()
        self._update_available_rvt_targets()
        self._apply_detection_state()

        self._refresh_path_validations()

        # ── Appliquer le mode initial ──
        self._apply_mode()

        # ── Validation initiale (après que tous les widgets soient visibles) ──
        self._validate_can_run()

        self._logger.info("Pipeline prêt à être utilisé")

    # ═════════════════════════════════════════════
    # SECTION : Sources de données
    # ═════════════════════════════════════════════

    def _build_sources_section(self):
        group = QGroupBox("Sources de données")
        self._sources_group = group
        layout = QVBoxLayout(group)

        # ── Étape 1 : Type de données ──
        step1 = QWidget()
        step1_layout = QHBoxLayout(step1)
        step1_layout.setContentsMargins(0, 0, 0, 0)
        step1_label = QLabel("<b>1.</b> Type de données :")
        step1_layout.addWidget(step1_label)
        self.data_mode_combo = NoWheelComboBox()
        self.data_mode_combo.addItem("Téléchargement IGN (à partir d'une zone)", "ign_laz")
        self.data_mode_combo.addItem("Nuages de points locaux (LAZ/LAS)", "local_laz")
        self.data_mode_combo.addItem("MNT déjà calculés (TIF/ASC)", "existing_mnt")
        self.data_mode_combo.addItem("Indices de visualisation existants (TIF)", "existing_rvt")
        self.data_mode_combo.setToolTip(
            "<b>Point d'entrée dans le pipeline</b><br><br>"
            "<table style='border-collapse:collapse; font-size:9pt;'>"
            "<tr>"
            "  <td style='padding:6px 12px; width:130px; background:#e8f6f3; border:1px solid #aaa; text-align:center;'>"
            "    <b>① Téléchargement</b><br><span style='color:#666;'>Zone → dalles LiDAR</span></td>"
            "  <td style='padding:3px 4px;'>→</td>"
            "  <td style='padding:6px 12px; width:130px; background:#eaf2f8; border:1px solid #aaa; text-align:center;'>"
            "    <b>② Nuages de points</b><br><span style='color:#666;'>LAZ / LAS</span></td>"
            "  <td style='padding:3px 4px;'>→</td>"
            "  <td style='padding:6px 12px; width:130px; background:#fef9e7; border:1px solid #aaa; text-align:center;'>"
            "    <b>③ MNT</b><br><span style='color:#666;'>TIF / ASC</span></td>"
            "  <td style='padding:3px 4px;'>→</td>"
            "  <td style='padding:6px 12px; width:130px; background:#fdedec; border:1px solid #aaa; text-align:center;'>"
            "    <b>④ Indices (RVT)</b><br><span style='color:#666;'>TIF (SVF, M-HS…)</span></td>"
            "  <td style='padding:3px 4px;'>→</td>"
            "  <td style='padding:6px 12px; width:130px; background:#f5eef8; border:1px solid #aaa; text-align:center;'>"
            "    <b>⑤ Détection IA</b></td>"
            "</tr>"
            "</table>"
            "<br>"
            "Ce choix détermine à quelle étape le pipeline démarre :<br>"
            "• <b>Téléchargement IGN</b> : pipeline complet depuis le téléchargement des dalles LiDAR<br>"
            "• <b>Nuages locaux</b> : commence à l'étape ②, avec vos fichiers LAZ/LAS<br>"
            "• <b>MNT déjà calculés</b> : commence à l'étape ③, le MNT existe déjà<br>"
            "• <b>Indices existants</b> : commence à l'étape ④, directement à la détection"
        )
        step1_layout.addWidget(self.data_mode_combo, 1)
        layout.addWidget(step1)

        # ── Description du mode sélectionné ──
        self._mode_description = QLabel("")
        self._mode_description.setWordWrap(True)
        self._mode_description.setStyleSheet("color: #666; font-style: italic; margin-left: 24px; margin-bottom: 4px;")
        layout.addWidget(self._mode_description)

        # ── Étape 2 : Source / entrée ──
        step2 = QWidget()
        step2_layout = QHBoxLayout(step2)
        step2_layout.setContentsMargins(0, 0, 0, 0)
        self._step2_label = QLabel("<b>2.</b> Zone d'étude (polygone) :")
        step2_layout.addWidget(self._step2_label)
        self.specific_source_edit = QLineEdit()
        self.specific_source_edit.setPlaceholderText("Fichier shapefile délimitant la zone d'étude (.shp)")
        step2_layout.addWidget(self.specific_source_edit, 1)
        specific_btn = QPushButton("Parcourir")
        specific_btn.clicked.connect(self._browse_specific_source)
        step2_layout.addWidget(specific_btn)
        self._qgis_layer_btn = QPushButton("Couche QGIS")
        self._qgis_layer_btn.setToolTip(
            "Sélectionner une couche polygone déjà chargée dans le projet QGIS"
        )
        self._qgis_layer_btn.clicked.connect(self._pick_qgis_polygon_layer)
        step2_layout.addWidget(self._qgis_layer_btn)
        layout.addWidget(step2)

        self.specific_source_label = self._step2_label

        # ── Étape 3 : Dossier de sortie ──
        step3 = QWidget()
        step3_layout = QHBoxLayout(step3)
        step3_layout.setContentsMargins(0, 0, 0, 0)
        step3_label = QLabel("<b>3.</b> Dossier de sortie :")
        step3_label.setToolTip("Le dossier dans lequel tous les résultats seront enregistrés.")
        step3_layout.addWidget(step3_label)
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Chemin vers le dossier de sortie des résultats")
        step3_layout.addWidget(self.output_dir_edit, 1)
        output_btn = QPushButton("Parcourir")
        output_btn.clicked.connect(self._browse_output_dir)
        step3_layout.addWidget(output_btn)
        layout.addWidget(step3)

        self._config_layout.addWidget(group)
        self._update_mode_description()

    # ═════════════════════════════════════════════
    # SECTION : MNT et Indices de visualisation (checkboxes) — toujours visible
    # ═════════════════════════════════════════════

    # Tooltips individuelles par produit
    _PRODUCT_TOOLTIPS = {
        "MNT": "Modèle Numérique de Terrain : représentation de l'altitude du sol.",
        "DENSITE": "Carte de densité des points LiDAR au sol.",
        "M_HS": "Multi-Hillshade : ombrage multi-directionnel qui révèle le micro-relief.",
        "SVF": "Sky-View Factor : fraction de ciel visible depuis chaque point.\nMet en évidence les creux et dépressions.",
        "SLO": "Slope (pente) : angle d'inclinaison du terrain en chaque point.",
        "LD": "Local Dominance : rapport de hauteur locale, révèle les structures en relief.",
        "SLRM": "Simple Local Relief Model : différence entre le MNT et un MNT lissé.\nIsoler les micro-reliefs.",
        "VAT": "Visualization for Archaeological Topography :\ncombinaison optimisée pour l'archéologie.",
    }

    def _build_products_section(self):
        group = QGroupBox("MNT et Indices de visualisation")
        group.setToolTip(
            "Sélectionnez les indices à calculer à partir du MNT.\n"
            "Survolez chaque case pour une description détaillée."
        )
        layout = QVBoxLayout(group)

        products_row = QWidget()
        products_layout = QHBoxLayout(products_row)
        products_layout.setContentsMargins(0, 0, 0, 0)

        self._product_cbs: dict[str, QCheckBox] = {}
        for key, label, default, _is_rvt in PRODUCTS:
            cb = QCheckBox(label)
            cb.setChecked(default)
            cb.setToolTip(self._PRODUCT_TOOLTIPS.get(key, ""))
            cb.toggled.connect(self._validate_can_run)
            self._product_cbs[key] = cb
            products_layout.addWidget(cb)
        products_layout.addStretch(1)
        layout.addWidget(products_row)

        self._products_group = group
        self._config_layout.addWidget(group)

    # ═════════════════════════════════════════════
    # SECTION : Paramètres des indices de visualisation (expert)
    # ═════════════════════════════════════════════

    def _build_rvt_section(self):
        self._rvt_group = QGroupBox("Paramètres des indices de visualisation")
        rvt_layout = QVBoxLayout(self._rvt_group)

        self.rvt_tabs = QTabWidget()

        # --- M-HS ---
        mdh_tab = QWidget()
        mdh_form = QFormLayout(mdh_tab)
        self.mdh_num_directions_spin = NoWheelSpinBox()
        self.mdh_num_directions_spin.setRange(1, 360)
        self.mdh_num_directions_spin.setValue(16)
        self.mdh_num_directions_spin.setToolTip("Nombre de directions d'éclairage simulées (défaut : 16)")
        self.mdh_sun_elevation_spin = NoWheelSpinBox()
        self.mdh_sun_elevation_spin.setRange(0, 90)
        self.mdh_sun_elevation_spin.setValue(35)
        self.mdh_sun_elevation_spin.setToolTip("Angle d'élévation du soleil en degrés (défaut : 35°)")
        self.mdh_ve_factor_spin = NoWheelSpinBox()
        self.mdh_ve_factor_spin.setRange(1, 100)
        self.mdh_ve_factor_spin.setValue(1)
        self.mdh_save_8bit_cb = QCheckBox("Sauver en 8bit")
        self.mdh_save_8bit_cb.setChecked(True)
        mdh_form.addRow("Nombre directions :", self.mdh_num_directions_spin)
        mdh_form.addRow("Élévation solaire (°) :", self.mdh_sun_elevation_spin)
        mdh_form.addRow("Facteur VE :", self.mdh_ve_factor_spin)
        mdh_form.addRow("", self.mdh_save_8bit_cb)
        self.rvt_tabs.addTab(mdh_tab, "M-HS")

        # --- SVF ---
        svf_tab = QWidget()
        svf_form = QFormLayout(svf_tab)
        self.svf_noise_remove_spin = NoWheelSpinBox()
        self.svf_noise_remove_spin.setRange(0, 9999)
        self.svf_noise_remove_spin.setValue(0)
        self.svf_num_directions_spin = NoWheelSpinBox()
        self.svf_num_directions_spin.setRange(1, 360)
        self.svf_num_directions_spin.setValue(16)
        self.svf_radius_spin = NoWheelSpinBox()
        self.svf_radius_spin.setRange(0, 100000)
        self.svf_radius_spin.setValue(10)
        self.svf_ve_factor_spin = NoWheelSpinBox()
        self.svf_ve_factor_spin.setRange(1, 100)
        self.svf_ve_factor_spin.setValue(1)
        self.svf_save_8bit_cb = QCheckBox("Sauver en 8bit")
        self.svf_save_8bit_cb.setChecked(True)
        svf_form.addRow("Suppression bruit :", self.svf_noise_remove_spin)
        svf_form.addRow("Nombre directions :", self.svf_num_directions_spin)
        svf_form.addRow("Rayon (px) :", self.svf_radius_spin)
        svf_form.addRow("Facteur VE :", self.svf_ve_factor_spin)
        svf_form.addRow("", self.svf_save_8bit_cb)
        self.rvt_tabs.addTab(svf_tab, "SVF")

        # --- Slope ---
        slope_tab = QWidget()
        slope_form = QFormLayout(slope_tab)
        self.slope_unit_combo = NoWheelComboBox()
        self.slope_unit_combo.addItem("Degrés", 0)
        self.slope_unit_combo.addItem("Pourcentage", 1)
        self.slope_ve_factor_spin = NoWheelSpinBox()
        self.slope_ve_factor_spin.setRange(1, 100)
        self.slope_ve_factor_spin.setValue(1)
        self.slope_save_8bit_cb = QCheckBox("Sauver en 8bit")
        self.slope_save_8bit_cb.setChecked(True)
        slope_form.addRow("Unité :", self.slope_unit_combo)
        slope_form.addRow("Facteur VE :", self.slope_ve_factor_spin)
        slope_form.addRow("", self.slope_save_8bit_cb)
        self.rvt_tabs.addTab(slope_tab, "Slope")

        # --- LD ---
        ld_tab = QWidget()
        ld_form = QFormLayout(ld_tab)
        self.ldo_angular_res_spin = NoWheelSpinBox()
        self.ldo_angular_res_spin.setRange(1, 360)
        self.ldo_angular_res_spin.setValue(15)
        self.ldo_min_radius_spin = NoWheelSpinBox()
        self.ldo_min_radius_spin.setRange(0, 100000)
        self.ldo_min_radius_spin.setValue(10)
        self.ldo_max_radius_spin = NoWheelSpinBox()
        self.ldo_max_radius_spin.setRange(0, 100000)
        self.ldo_max_radius_spin.setValue(20)
        self.ldo_observer_h_spin = NoWheelDoubleSpinBox()
        self.ldo_observer_h_spin.setDecimals(2)
        self.ldo_observer_h_spin.setRange(0.0, 10000.0)
        self.ldo_observer_h_spin.setSingleStep(0.1)
        self.ldo_observer_h_spin.setValue(1.7)
        self.ldo_ve_factor_spin = NoWheelSpinBox()
        self.ldo_ve_factor_spin.setRange(1, 100)
        self.ldo_ve_factor_spin.setValue(1)
        self.ldo_save_8bit_cb = QCheckBox("Sauver en 8bit")
        self.ldo_save_8bit_cb.setChecked(True)
        ld_form.addRow("Résolution angulaire (°) :", self.ldo_angular_res_spin)
        ld_form.addRow("Rayon min (px) :", self.ldo_min_radius_spin)
        ld_form.addRow("Rayon max (px) :", self.ldo_max_radius_spin)
        ld_form.addRow("Hauteur observateur (m) :", self.ldo_observer_h_spin)
        ld_form.addRow("Facteur VE :", self.ldo_ve_factor_spin)
        ld_form.addRow("", self.ldo_save_8bit_cb)
        self.rvt_tabs.addTab(ld_tab, "LD")

        # --- SLRM ---
        slrm_tab = QWidget()
        slrm_form = QFormLayout(slrm_tab)
        self.slrm_radius_spin = NoWheelSpinBox()
        self.slrm_radius_spin.setRange(1, 100000)
        self.slrm_radius_spin.setValue(20)
        self.slrm_ve_factor_spin = NoWheelSpinBox()
        self.slrm_ve_factor_spin.setRange(1, 100)
        self.slrm_ve_factor_spin.setValue(1)
        self.slrm_save_8bit_cb = QCheckBox("Sauver en 8bit")
        self.slrm_save_8bit_cb.setChecked(True)
        slrm_form.addRow("Rayon (px) :", self.slrm_radius_spin)
        slrm_form.addRow("Facteur VE :", self.slrm_ve_factor_spin)
        slrm_form.addRow("", self.slrm_save_8bit_cb)
        self.rvt_tabs.addTab(slrm_tab, "SLRM")

        # --- VAT ---
        vat_tab = QWidget()
        vat_form = QFormLayout(vat_tab)
        self.vat_terrain_type_combo = NoWheelComboBox()
        self.vat_terrain_type_combo.addItem("Général", 0)
        self.vat_terrain_type_combo.addItem("Plat", 1)
        self.vat_terrain_type_combo.addItem("Pentu", 2)
        self.vat_save_8bit_cb = QCheckBox("Sauver en 8bit")
        self.vat_save_8bit_cb.setChecked(True)
        vat_form.addRow("Type de terrain :", self.vat_terrain_type_combo)
        vat_form.addRow("", self.vat_save_8bit_cb)
        self.rvt_tabs.addTab(vat_tab, "VAT")

        rvt_layout.addWidget(self.rvt_tabs)

        # Marge tuiles (tuilage RVT)
        overlap_row = QWidget()
        overlap_layout = QHBoxLayout(overlap_row)
        overlap_layout.setContentsMargins(0, 0, 0, 0)
        overlap_layout.addWidget(QLabel("Marge tuiles :"))
        self.tile_overlap_spin = NoWheelSpinBox()
        self.tile_overlap_spin.setRange(0, 100)
        self.tile_overlap_spin.setValue(20)
        self.tile_overlap_spin.setToolTip(
            "Pourcentage de chevauchement entre les tuiles lors du calcul des indices.\n"
            "Évite les artefacts aux bords des tuiles."
        )
        overlap_layout.addWidget(self.tile_overlap_spin)
        overlap_layout.addWidget(QLabel("%"))
        overlap_layout.addStretch(1)
        rvt_layout.addWidget(overlap_row)

        self._reset_rvt_btn = QPushButton("Remettre par défaut")
        self._reset_rvt_btn.clicked.connect(self._reset_rvt_config)
        rvt_layout.addWidget(self._reset_rvt_btn)

        self._config_layout.addWidget(self._rvt_group)

    # ═════════════════════════════════════════════
    # SECTION : Paramètres MNT (expert, masqué si existing_mnt/rvt)
    # ═════════════════════════════════════════════

    def _build_mnt_section(self):
        self._mnt_group = QGroupBox("Paramètres MNT")
        mnt_layout = QVBoxLayout(self._mnt_group)

        # Filtre PDAL (concerne les nuages → étape pré-MNT)
        filter_row = QWidget()
        filter_form = QFormLayout(filter_row)
        filter_form.setContentsMargins(0, 0, 0, 0)
        self.filter_expression_edit = QLineEdit()
        self.filter_expression_edit.setPlaceholderText(
            "Ex: Classification = 2 OR Classification = 6"
        )
        self.filter_expression_edit.setToolTip(
            "Expression de filtrage PDAL appliquée aux nuages de points.\n"
            "Permet de ne garder que certaines classifications LiDAR.\n\n"
            "Appliqué avant le calcul du MNT."
        )
        filter_form.addRow("Filtre PDAL :", self.filter_expression_edit)
        mnt_layout.addWidget(filter_row)

        # Résolutions
        res_row = QWidget()
        res_layout = QHBoxLayout(res_row)
        res_layout.setContentsMargins(0, 0, 0, 0)
        res_layout.addWidget(QLabel("Résolution MNT :"))
        self.mnt_resolution_spin = NoWheelDoubleSpinBox()
        self.mnt_resolution_spin.setDecimals(2)
        self.mnt_resolution_spin.setRange(0.01, 100.0)
        self.mnt_resolution_spin.setSingleStep(0.1)
        self.mnt_resolution_spin.setValue(0.5)
        self.mnt_resolution_spin.setToolTip("Résolution du MNT en mètres. Plus petit = plus précis mais plus lourd.")
        res_layout.addWidget(self.mnt_resolution_spin)
        res_layout.addWidget(QLabel("m"))
        res_layout.addSpacing(20)
        res_layout.addWidget(QLabel("Résolution densité :"))
        self.density_resolution_spin = NoWheelDoubleSpinBox()
        self.density_resolution_spin.setDecimals(2)
        self.density_resolution_spin.setRange(0.01, 100.0)
        self.density_resolution_spin.setSingleStep(0.1)
        self.density_resolution_spin.setValue(1.0)
        self.density_resolution_spin.setToolTip("Résolution de la carte de densité en mètres.")
        res_layout.addWidget(self.density_resolution_spin)
        res_layout.addWidget(QLabel("m"))
        res_layout.addStretch(1)
        mnt_layout.addWidget(res_row)

        self._reset_mnt_btn = QPushButton("Remettre par défaut")
        self._reset_mnt_btn.clicked.connect(self._reset_mnt_config)
        mnt_layout.addWidget(self._reset_mnt_btn)

        self._config_layout.addWidget(self._mnt_group)

    # ═════════════════════════════════════════════
    # SECTION : Performance (expert)
    # ═════════════════════════════════════════════

    def _build_performance_section(self):
        self._performance_group = QGroupBox("Performance")
        perf_main_layout = QVBoxLayout(self._performance_group)

        perf_row = QWidget()
        perf_layout = QHBoxLayout(perf_row)
        perf_layout.setContentsMargins(0, 0, 0, 0)

        perf_layout.addWidget(QLabel("Workers :"))
        self.max_workers_spin = NoWheelSpinBox()
        self.max_workers_spin.setRange(1, 16)
        self.max_workers_spin.setValue(4)
        self.max_workers_spin.setToolTip(
            "Nombre de traitements parallèles.\n\n"
            "Repères selon votre RAM disponible :\n"
            "  • 8 Go RAM  →  2 à 4 workers\n"
            "  • 16 Go RAM →  4 à 8 workers\n"
            "  • 32 Go RAM →  8 à 12 workers\n\n"
            "Plus de workers = plus rapide, mais consomme plus de mémoire."
        )
        perf_layout.addWidget(self.max_workers_spin)

        self._workers_hint = QLabel("")
        self._workers_hint.setStyleSheet("color: #888; font-style: italic;")
        self.max_workers_spin.valueChanged.connect(self._update_workers_hint)
        perf_layout.addWidget(self._workers_hint)
        perf_layout.addStretch(1)
        perf_main_layout.addWidget(perf_row)
        self._update_workers_hint()

        self._reset_perf_btn = QPushButton("Remettre par défaut")
        self._reset_perf_btn.clicked.connect(self._reset_perf_config)
        perf_main_layout.addWidget(self._reset_perf_btn)

        self._config_layout.addWidget(self._performance_group)

    # ═════════════════════════════════════════════
    # SECTION : Détection automatique (ex Computer Vision)
    # ═════════════════════════════════════════════

    def _build_detection_section(self):
        group = QGroupBox("Détection automatique par IA")
        group_layout = QVBoxLayout(group)

        # Toggle activation
        self.detection_enabled_cb = QCheckBox("Activer la détection par intelligence artificielle")
        self.detection_enabled_cb.setToolTip(
            "Active l'analyse par intelligence artificielle des indices de visualisation\n"
            "pour détecter automatiquement des objets archéologiques (cratères, structures, etc.)."
        )
        self.detection_enabled_cb.toggled.connect(self._apply_detection_state)
        group_layout.addWidget(self.detection_enabled_cb)

        # Texte explicatif quand désactivé
        self._detection_hint = QLabel(
            "Activez pour détecter automatiquement des objets archéologiques "
            "(cratères, structures, etc.) à l'aide de l'intelligence artificielle."
        )
        self._detection_hint.setWordWrap(True)
        self._detection_hint.setStyleSheet("color: #888; font-style: italic; margin-left: 20px;")
        group_layout.addWidget(self._detection_hint)

        # --- Contenu détection (masqué si désactivé) ---
        self._detection_content = QWidget()
        det_layout = QVBoxLayout(self._detection_content)
        det_layout.setContentsMargins(0, 4, 0, 0)

        # Table multi-runs (simple: juste modèle; expert: + RVT cible + aire min)
        runs_group = QGroupBox("Modèles")
        runs_vlayout = QVBoxLayout(runs_group)
        runs_vlayout.setContentsMargins(4, 4, 4, 4)

        self.det_runs_table = QTableWidget(0, 4)
        self.det_runs_table.setHorizontalHeaderLabels(["Modèle", "Indice cible", "Aire min (m²)", ""])
        self.det_runs_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.det_runs_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.det_runs_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.det_runs_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        self.det_runs_table.setColumnWidth(3, 60)
        self.det_runs_table.setMaximumHeight(180)
        self.det_runs_table.setMinimumHeight(80)
        self.det_runs_table.verticalHeader().setVisible(False)
        self.det_runs_table.verticalHeader().setDefaultSectionSize(36)
        runs_vlayout.addWidget(self.det_runs_table)

        runs_btn_row = QWidget()
        runs_btn_layout = QHBoxLayout(runs_btn_row)
        runs_btn_layout.setContentsMargins(0, 0, 0, 0)
        self.det_add_run_btn = QPushButton("+ Ajouter un modèle")
        self.det_add_run_btn.clicked.connect(lambda: self._add_det_run_row())
        runs_btn_layout.addWidget(self.det_add_run_btn)
        runs_btn_layout.addStretch(1)
        self.det_refresh_btn = QPushButton("Actualiser")
        self.det_refresh_btn.clicked.connect(self._refresh_models)
        runs_btn_layout.addWidget(self.det_refresh_btn)
        runs_vlayout.addWidget(runs_btn_row)

        det_layout.addWidget(runs_group)

        # Classes à détecter
        classes_group = QGroupBox("Classes à détecter")
        classes_layout = QVBoxLayout(classes_group)
        self.det_classes_list = QListWidget()
        self.det_classes_list.setMaximumHeight(150)
        classes_layout.addWidget(self.det_classes_list)
        classes_btn_row = QWidget()
        classes_btn_layout = QHBoxLayout(classes_btn_row)
        classes_btn_layout.setContentsMargins(0, 0, 0, 0)
        self.det_select_all_btn = QPushButton("Tout sélectionner")
        self.det_select_all_btn.clicked.connect(self._select_all_classes)
        self.det_deselect_all_btn = QPushButton("Tout désélectionner")
        self.det_deselect_all_btn.clicked.connect(self._deselect_all_classes)
        classes_btn_layout.addWidget(self.det_select_all_btn)
        classes_btn_layout.addWidget(self.det_deselect_all_btn)
        classes_btn_layout.addStretch(1)
        classes_layout.addWidget(classes_btn_row)
        det_layout.addWidget(classes_group)

        # --- Section expert détection ---
        self._detection_expert_content = QWidget()
        det_expert_layout = QVBoxLayout(self._detection_expert_content)
        det_expert_layout.setContentsMargins(0, 8, 0, 0)

        # Seuils avec labels parlants
        thresholds_group = QGroupBox("Seuils de détection")
        thresholds_layout = QVBoxLayout(thresholds_group)

        # Confiance
        conf_row = QWidget()
        conf_layout = QHBoxLayout(conf_row)
        conf_layout.setContentsMargins(0, 0, 0, 0)
        conf_layout.addWidget(QLabel("Niveau de certitude :"))
        self.det_confidence_spin = NoWheelDoubleSpinBox()
        self.det_confidence_spin.setDecimals(2)
        self.det_confidence_spin.setRange(0.0, 1.0)
        self.det_confidence_spin.setSingleStep(0.05)
        self.det_confidence_spin.setValue(0.3)
        self.det_confidence_spin.setToolTip(
            "Seuil de confiance minimal pour retenir une détection.\n\n"
            "• 0.1 – 0.3 : Détection large (plus de résultats, mais plus de faux positifs)\n"
            "• 0.3 – 0.5 : Équilibré (recommandé)\n"
            "• 0.5 – 0.8 : Détection stricte (moins de résultats, mais plus fiables)\n"
            "• 0.8 – 1.0 : Très strict (uniquement les détections quasi certaines)"
        )
        conf_layout.addWidget(self.det_confidence_spin)

        self._conf_label = QLabel("")
        self._conf_label.setStyleSheet("color: #666; font-style: italic; margin-left: 8px;")
        self._conf_label.setMinimumWidth(140)
        conf_layout.addWidget(self._conf_label)
        self.det_confidence_spin.valueChanged.connect(self._update_confidence_label)
        conf_layout.addStretch(1)
        thresholds_layout.addWidget(conf_row)

        # IoU
        iou_row = QWidget()
        iou_layout = QHBoxLayout(iou_row)
        iou_layout.setContentsMargins(0, 0, 0, 0)
        iou_layout.addWidget(QLabel("Chevauchement max (IoU) :"))
        self.det_iou_spin = NoWheelDoubleSpinBox()
        self.det_iou_spin.setDecimals(2)
        self.det_iou_spin.setRange(0.0, 1.0)
        self.det_iou_spin.setSingleStep(0.05)
        self.det_iou_spin.setValue(0.5)
        self.det_iou_spin.setToolTip(
            "Seuil IoU (Intersection over Union) pour la suppression des doublons.\n\n"
            "Quand deux détections se chevauchent au-delà de ce seuil,\n"
            "seule la plus confiante est conservée.\n\n"
            "• 0.3 : Suppression agressive des doublons\n"
            "• 0.5 : Équilibré (recommandé)\n"
            "• 0.7 : Conserve plus de détections proches"
        )
        iou_layout.addWidget(self.det_iou_spin)
        iou_layout.addStretch(1)
        thresholds_layout.addWidget(iou_row)

        det_expert_layout.addWidget(thresholds_group)

        # Options de sortie
        output_group = QGroupBox("Options de sortie")
        output_layout = QVBoxLayout(output_group)
        self.det_generate_annotated_cb = QCheckBox("Générer des images annotées")
        self.det_generate_annotated_cb.setToolTip(
            "Produit des images montrant les détections superposées\n"
            "sur les indices de visualisation. Utile pour la vérification visuelle."
        )
        self.det_generate_shp_cb = QCheckBox("Générer des shapefiles")
        self.det_generate_shp_cb.setChecked(True)
        self.det_generate_shp_cb.setToolTip("Produit des shapefiles exploitables dans QGIS.")
        output_layout.addWidget(self.det_generate_annotated_cb)
        output_layout.addWidget(self.det_generate_shp_cb)
        det_expert_layout.addWidget(output_group)

        det_layout.addWidget(self._detection_expert_content)

        group_layout.addWidget(self._detection_content)

        self._detection_group = group
        self._config_layout.addWidget(group)

        self._update_confidence_label()
        self._apply_detection_state()

    # ═════════════════════════════════════════════
    # SECTION : Logs (redimensionnable via splitter)
    # ═════════════════════════════════════════════

    def _build_logs_section(self):
        logs_container = QWidget()
        logs_container.setObjectName("logs_container")
        logs_vlayout = QVBoxLayout(logs_container)
        logs_vlayout.setContentsMargins(0, 0, 0, 0)
        logs_vlayout.setSpacing(0)

        # Barre de titre distincte
        header = QWidget()
        header.setObjectName("logs_header")
        header.setStyleSheet("""
            QWidget#logs_header {
                background-color: #3c3f41;
                border-top: 2px solid #2b79c2;
            }
        """)
        header.setFixedHeight(28)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 0, 8, 0)
        header_layout.setSpacing(6)

        title_label = QLabel("📋  Journal d'exécution")
        title_label.setStyleSheet("color: #bbbbbb; font-weight: bold; font-size: 11px;")
        header_layout.addWidget(title_label)
        header_layout.addStretch(1)

        clear_btn = QPushButton("Effacer")
        clear_btn.setFixedHeight(20)
        clear_btn.setStyleSheet("""
            QPushButton {
                background: #555; color: #ccc; border: none;
                border-radius: 3px; padding: 0 8px; font-size: 10px;
            }
            QPushButton:hover { background: #666; }
        """)
        clear_btn.clicked.connect(lambda: self.logs_text.clear())
        header_layout.addWidget(clear_btn)

        logs_vlayout.addWidget(header)

        # Zone de texte avec fond sombre
        self.logs_text = QPlainTextEdit()
        self.logs_text.setReadOnly(True)
        self.logs_text.setPlaceholderText("Les messages du pipeline apparaîtront ici...")
        self.logs_text.setStyleSheet("""
            QPlainTextEdit {
                background-color: #2b2b2b;
                color: #d4d4d4;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 10px;
                border: none;
                border-top: 1px solid #444;
            }
        """)
        logs_vlayout.addWidget(self.logs_text)

        self._splitter.addWidget(logs_container)

        # Les logs prennent tout l'espace restant sous la config
        logs_container.setMinimumHeight(80)
        self._splitter.setStretchFactor(0, 0)  # config : taille naturelle
        self._splitter.setStretchFactor(1, 1)  # logs : prend l'espace restant
        self._splitter.setSizes([400, 200])     # taille initiale : config=600px, logs=200px

    # ═════════════════════════════════════════════
    # Boutons d'action en bas
    # ═════════════════════════════════════════════

    def _build_buttons_row(self, parent_layout):
        buttons_row = QWidget()
        buttons_layout = QHBoxLayout(buttons_row)
        buttons_layout.setContentsMargins(0, 0, 0, 0)

        # Action principale
        self.run_btn = QPushButton("Lancer le pipeline")
        self.run_btn.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_MediaPlay))
        self.run_btn.setMinimumHeight(36)
        self.run_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 4px 20px;"
            "  background-color: #27ae60; color: white; border-radius: 4px; }"
            "QPushButton:hover { background-color: #219a52; }"
            "QPushButton:disabled { background-color: #bdc3c7; color: #7f8c8d; }"
        )
        self.run_btn.clicked.connect(self._on_run_clicked)
        buttons_layout.addWidget(self.run_btn)

        self.cancel_btn = QPushButton("Annuler")
        self.cancel_btn.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_MediaStop))
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        buttons_layout.addWidget(self.cancel_btn)

        buttons_layout.addSpacing(20)

        self.stage_label = QLabel("")
        self.stage_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        buttons_layout.addWidget(self.stage_label, 0)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        buttons_layout.addWidget(self.progress_bar, 1)

        parent_layout.addWidget(buttons_row, 0)

        # Validation : desactiver Lancer si config incomplete
        self.output_dir_edit.textChanged.connect(self._validate_can_run)
        self.specific_source_edit.textChanged.connect(self._validate_can_run)
        self.data_mode_combo.currentIndexChanged.connect(self._validate_can_run)
        self._validate_can_run()

    # ═════════════════════════════════════════════
    # Logique Mode Simple / Expert
    # ═════════════════════════════════════════════

    def _apply_mode(self):
        """Applique la visibilité des sections selon le mode choisi.

        Ne modifie que la visibilité : les valeurs des widgets sont
        préservées pour qu'un switch Simple↔Expert soit transparent.
        Ré-applique les états dépendants (data mode, détection) pour
        garantir la cohérence.
        """
        is_expert = (self.mode_combo.currentData() == "expert")
        data_mode = self.data_mode_combo.currentData() or "ign_laz"

        # Paramètres MNT : expert uniquement ET si le data mode nécessite un calcul MNT
        needs_mnt = data_mode not in ("existing_mnt", "existing_rvt")
        self._mnt_group.setVisible(is_expert and needs_mnt)

        # RVT détaillé : expert uniquement ET si le data mode le permet
        needs_products = data_mode != "existing_rvt"
        self._rvt_group.setVisible(is_expert and needs_products)

        # Performance : expert uniquement
        self._performance_group.setVisible(is_expert)

        # Colonnes Indice cible et Aire min : toujours visibles (simple + expert)

        # Section expert détection : visible uniquement si expert ET détection activée
        det_enabled = self.detection_enabled_cb.isChecked()
        self._detection_expert_content.setVisible(is_expert and det_enabled)

    def _apply_detection_state(self):
        """Active/désactive le contenu détection selon le checkbox.

        Tient compte du mode Simple/Expert pour la partie expert.
        """
        enabled = self.detection_enabled_cb.isChecked()
        is_expert = (self.mode_combo.currentData() == "expert")
        self._detection_content.setVisible(enabled)
        self._detection_hint.setVisible(not enabled)
        self._detection_expert_content.setVisible(enabled and is_expert)

    def _validate_can_run(self):
        """Active le bouton Lancer uniquement si la configuration est complète.

        Vérifie : source, sortie, et au moins un indice coché.
        Met à jour le tooltip du bouton. Le style visuel des champs est géré
        par _refresh_path_validations.
        """
        has_source = bool(self.specific_source_edit.text().strip())
        has_output = bool(self.output_dir_edit.text().strip())

        data_mode = self.data_mode_combo.currentData() or "ign_laz"
        if data_mode == "existing_rvt":
            has_product = True
        else:
            mnt_visible = data_mode not in ("existing_mnt", "existing_rvt")
            has_product = any(
                cb.isChecked()
                for key, cb in self._product_cbs.items()
                if key not in ("MNT", "DENSITE") or mnt_visible
            )

        can_run = has_source and has_output and has_product
        self.run_btn.setEnabled(can_run)

        if can_run:
            self.run_btn.setToolTip("Lancer le pipeline avec la configuration actuelle")
        else:
            reasons = []
            if not has_source:
                reasons.append("• Renseignez la source de données (étape 2)")
            if not has_output:
                reasons.append("• Renseignez le dossier de sortie (étape 3)")
            if not has_product:
                reasons.append("• Cochez au moins un indice de visualisation")
            self.run_btn.setToolTip("Configuration incomplète :\n" + "\n".join(reasons))

    # ═════════════════════════════════════════════
    # Sauvegarde / Chargement de configuration (export/import JSON utilisateur)
    # ═════════════════════════════════════════════

    def _gather_config(self) -> dict:
        """Collecte toute la configuration actuelle dans un dictionnaire (pour export)."""
        # Synchroniser self._config depuis les widgets
        self._collect_config_from_widgets()
        return self._config

    def _apply_config(self, config: dict):
        """Applique un dictionnaire de configuration à l'interface (pour import)."""
        defaults = self._config_manager.default_config()
        self._config_manager._deep_update(defaults, config)
        self._config = defaults
        self._load_into_widgets()
        self._apply_data_mode_state()
        self._apply_detection_state()
        self._apply_mode()
        self._refresh_model_classes()
        self._refresh_path_validations()

    def _save_config(self):
        """Sauvegarde la configuration actuelle dans un fichier JSON (export utilisateur)."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Sauvegarder la configuration",
            "", "Configuration (*.json);;Tous (*.*)"
        )
        if path:
            if not path.endswith(".json"):
                path += ".json"
            config = self._gather_config()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

    def _load_config(self):
        """Charge une configuration depuis un fichier JSON (import utilisateur)."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Charger une configuration",
            "", "Configuration (*.json);;Tous (*.*)"
        )
        if path:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
            if "ui_config" in config and isinstance(config["ui_config"], dict):
                config = config["ui_config"]
            self._apply_config(config)

    # ═════════════════════════════════════════════
    # Persistance ConfigManager (last_ui_config.json — dernière config utilisée)
    # ═════════════════════════════════════════════

    def _load_into_widgets(self) -> None:
        self._loading = True
        try:
            files = (self._config.get("app") or {}).get("files") or {}

            self.output_dir_edit.setText(files.get("output_dir") or "")

            cv = self._config.get("computer_vision") or {}
            self.detection_enabled_cb.setChecked(bool(cv.get("enabled", False)))
            self.det_confidence_spin.setValue(float(cv.get("confidence_threshold", 0.3)))
            self.det_iou_spin.setValue(float(cv.get("iou_threshold", 0.5)))
            self.det_generate_annotated_cb.setChecked(bool(cv.get("generate_annotated_images", False)))
            self.det_generate_shp_cb.setChecked(bool(cv.get("generate_shapefiles", False)))

            # Charger les runs CV dans le tableau
            self.det_runs_table.setRowCount(0)
            runs = cv.get("runs") or []
            if isinstance(runs, list):
                for run in runs:
                    if isinstance(run, dict):
                        model = str(run.get("model") or "")
                        rvt = str(run.get("target_rvt") or "LD")
                        min_area = float(run.get("min_area_m2", 0.0))
                        if model:
                            self._add_det_run_row(model_name=model, target_rvt=rvt, min_area_m2=min_area)
            # Compat: ancien format mono-modèle
            if self.det_runs_table.rowCount() == 0:
                old_model = str(cv.get("selected_model") or "")
                old_rvt = str(cv.get("target_rvt") or "LD")
                if old_model:
                    self._add_det_run_row(model_name=old_model, target_rvt=old_rvt)

            processing = self._config.get("processing") or {}
            self.mnt_resolution_spin.setValue(float(processing.get("mnt_resolution", 0.5)))
            self.density_resolution_spin.setValue(float(processing.get("density_resolution", 1.0)))
            self.tile_overlap_spin.setValue(int(processing.get("tile_overlap", 20)))
            self.max_workers_spin.setValue(int(processing.get("max_workers", 4)))
            self.filter_expression_edit.setText(processing.get("filter_expression") or "")

            products = processing.get("products") or {}
            for pkey, _label, default, _is_rvt in PRODUCTS:
                self._product_cbs[pkey].setChecked(bool(products.get(pkey, default)))

            rvt = self._config.get("rvt_params") or {}
            mdh = rvt.get("mdh") or {}
            self.mdh_num_directions_spin.setValue(int(mdh.get("num_directions", 16)))
            self.mdh_sun_elevation_spin.setValue(int(mdh.get("sun_elevation", 35)))
            self.mdh_ve_factor_spin.setValue(int(mdh.get("ve_factor", 1)))
            self.mdh_save_8bit_cb.setChecked(bool(mdh.get("save_as_8bit", True)))

            svf = rvt.get("svf") or {}
            self.svf_noise_remove_spin.setValue(int(svf.get("noise_remove", 0)))
            self.svf_num_directions_spin.setValue(int(svf.get("num_directions", 16)))
            self.svf_radius_spin.setValue(int(svf.get("radius", 10)))
            self.svf_ve_factor_spin.setValue(int(svf.get("ve_factor", 1)))
            self.svf_save_8bit_cb.setChecked(bool(svf.get("save_as_8bit", True)))

            slope = rvt.get("slope") or {}
            unit = int(slope.get("unit", 0))
            idx_unit = self.slope_unit_combo.findData(unit)
            self.slope_unit_combo.setCurrentIndex(idx_unit if idx_unit >= 0 else 0)
            self.slope_ve_factor_spin.setValue(int(slope.get("ve_factor", 1)))
            self.slope_save_8bit_cb.setChecked(bool(slope.get("save_as_8bit", True)))

            ldo = rvt.get("ldo") or {}
            self.ldo_angular_res_spin.setValue(int(ldo.get("angular_res", 15)))
            self.ldo_min_radius_spin.setValue(int(ldo.get("min_radius", 10)))
            self.ldo_max_radius_spin.setValue(int(ldo.get("max_radius", 20)))
            self.ldo_observer_h_spin.setValue(float(ldo.get("observer_h", 1.7)))
            self.ldo_ve_factor_spin.setValue(int(ldo.get("ve_factor", 1)))
            self.ldo_save_8bit_cb.setChecked(bool(ldo.get("save_as_8bit", True)))

            slrm = rvt.get("slrm") or {}
            self.slrm_radius_spin.setValue(int(slrm.get("radius", 20)))
            self.slrm_ve_factor_spin.setValue(int(slrm.get("ve_factor", 1)))
            self.slrm_save_8bit_cb.setChecked(bool(slrm.get("save_as_8bit", True)))

            vat = rvt.get("vat") or {}
            terrain = int(vat.get("terrain_type", 0))
            idx_terrain = self.vat_terrain_type_combo.findData(terrain)
            self.vat_terrain_type_combo.setCurrentIndex(idx_terrain if idx_terrain >= 0 else 0)
            self.vat_save_8bit_cb.setChecked(bool(vat.get("save_as_8bit", True)))

            # data_mode chargé en dernier : son signal currentIndexChanged déclenche
            # _on_data_mode_changed_internal → _save_from_widgets. En le plaçant après
            # tous les autres widgets, on garantit que l'état est complet avant toute sauvegarde.
            mode = files.get("data_mode") or "ign_laz"
            idx_mode = self.data_mode_combo.findData(mode)
            self.data_mode_combo.setCurrentIndex(idx_mode if idx_mode >= 0 else 0)
        finally:
            self._loading = False

    def _collect_config_from_widgets(self) -> None:
        """Synchronise self._config depuis l'état actuel des widgets."""
        app = self._config.setdefault("app", {})
        files = app.setdefault("files", {})

        files["output_dir"] = self.output_dir_edit.text().strip()
        files["data_mode"] = self.data_mode_combo.currentData()

        mode = self.data_mode_combo.currentData() or self._current_mode or "ign_laz"
        key, _, _ = self._mode_mapping(str(mode))
        files[key] = self.specific_source_edit.text().strip()

        cv = self._config.setdefault("computer_vision", {})
        cv["enabled"] = self.detection_enabled_cb.isChecked()

        # Collecter les runs depuis le tableau
        runs = []
        for row in range(self.det_runs_table.rowCount()):
            model_combo = self.det_runs_table.cellWidget(row, 0)
            rvt_combo = self.det_runs_table.cellWidget(row, 1)
            area_spin = self.det_runs_table.cellWidget(row, 2)
            model_val = str(model_combo.currentData() or model_combo.currentText() or "") if isinstance(model_combo, QComboBox) else ""
            rvt_val = str(rvt_combo.currentData() or "LD") if isinstance(rvt_combo, QComboBox) else "LD"
            area_val = float(area_spin.value()) if isinstance(area_spin, QDoubleSpinBox) else 0.0
            if model_val:
                runs.append({"model": model_val, "target_rvt": rvt_val, "min_area_m2": area_val})
        cv["runs"] = runs

        # Compat: garder selected_model/target_rvt du premier run
        if runs:
            cv["selected_model"] = runs[0]["model"]
            cv["target_rvt"] = runs[0]["target_rvt"]
        else:
            cv["selected_model"] = ""
            cv["target_rvt"] = "LD"
        cv["confidence_threshold"] = float(self.det_confidence_spin.value())
        cv["iou_threshold"] = float(self.det_iou_spin.value())
        cv["generate_annotated_images"] = self.det_generate_annotated_cb.isChecked()
        cv["generate_shapefiles"] = self.det_generate_shp_cb.isChecked()
        cv["models_dir"] = str(self._plugin_root / "data" / "models")

        processing = self._config.setdefault("processing", {})
        processing["mnt_resolution"] = float(self.mnt_resolution_spin.value())
        processing["density_resolution"] = float(self.density_resolution_spin.value())
        processing["tile_overlap"] = int(self.tile_overlap_spin.value())
        processing["max_workers"] = int(self.max_workers_spin.value())
        processing["filter_expression"] = self.filter_expression_edit.text().strip()

        products = processing.setdefault("products", {})
        for pkey, _label, _default, _is_rvt in PRODUCTS:
            products[pkey] = self._product_cbs[pkey].isChecked()

        rvt = self._config.setdefault("rvt_params", {})
        mdh = rvt.setdefault("mdh", {})
        mdh["num_directions"] = int(self.mdh_num_directions_spin.value())
        mdh["sun_elevation"] = int(self.mdh_sun_elevation_spin.value())
        mdh["ve_factor"] = int(self.mdh_ve_factor_spin.value())
        mdh["save_as_8bit"] = self.mdh_save_8bit_cb.isChecked()

        svf = rvt.setdefault("svf", {})
        svf["noise_remove"] = int(self.svf_noise_remove_spin.value())
        svf["num_directions"] = int(self.svf_num_directions_spin.value())
        svf["radius"] = int(self.svf_radius_spin.value())
        svf["ve_factor"] = int(self.svf_ve_factor_spin.value())
        svf["save_as_8bit"] = self.svf_save_8bit_cb.isChecked()

        slope = rvt.setdefault("slope", {})
        slope["unit"] = int(self.slope_unit_combo.currentData())
        slope["ve_factor"] = int(self.slope_ve_factor_spin.value())
        slope["save_as_8bit"] = self.slope_save_8bit_cb.isChecked()

        ldo = rvt.setdefault("ldo", {})
        ldo["angular_res"] = int(self.ldo_angular_res_spin.value())
        ldo["min_radius"] = int(self.ldo_min_radius_spin.value())
        ldo["max_radius"] = int(self.ldo_max_radius_spin.value())
        ldo["observer_h"] = float(self.ldo_observer_h_spin.value())
        ldo["ve_factor"] = int(self.ldo_ve_factor_spin.value())
        ldo["save_as_8bit"] = self.ldo_save_8bit_cb.isChecked()

        slrm = rvt.setdefault("slrm", {})
        slrm["radius"] = int(self.slrm_radius_spin.value())
        slrm["ve_factor"] = int(self.slrm_ve_factor_spin.value())
        slrm["save_as_8bit"] = self.slrm_save_8bit_cb.isChecked()

        vat = rvt.setdefault("vat", {})
        vat["terrain_type"] = int(self.vat_terrain_type_combo.currentData())
        vat["save_as_8bit"] = self.vat_save_8bit_cb.isChecked()

    def _save_from_widgets(self) -> None:
        self._collect_config_from_widgets()
        self._config_manager.save_last_ui_config(self._config)

    def _sync_config_from_widgets(self) -> None:
        self._collect_config_from_widgets()

    def _save_specific_source_only(self) -> None:
        mode = self.data_mode_combo.currentData() or self._current_mode or "ign_laz"
        key, _, _ = self._mode_mapping(mode)
        app = self._config.setdefault("app", {})
        files = app.setdefault("files", {})
        files[key] = self.specific_source_edit.text().strip()
        self._config_manager.save_last_ui_config(self._config)

    # ═════════════════════════════════════════════
    # Autosave (chaque changement widget → last_ui_config.json)
    # ═════════════════════════════════════════════

    def _wire_autosave(self) -> None:
        self.output_dir_edit.textChanged.connect(self._on_any_changed)
        self.data_mode_combo.currentIndexChanged.connect(self._on_data_mode_changed_internal)
        self.specific_source_edit.textChanged.connect(self._on_specific_source_changed)

        self.mnt_resolution_spin.valueChanged.connect(self._on_any_changed)
        self.density_resolution_spin.valueChanged.connect(self._on_any_changed)
        self.tile_overlap_spin.valueChanged.connect(self._on_any_changed)
        self.max_workers_spin.valueChanged.connect(self._on_any_changed)
        self.filter_expression_edit.textChanged.connect(self._on_any_changed)

        self._product_cbs["MNT"].toggled.connect(self._on_any_changed)
        self._product_cbs["DENSITE"].toggled.connect(self._on_any_changed)

        self.mdh_num_directions_spin.valueChanged.connect(self._on_any_changed)
        self.mdh_sun_elevation_spin.valueChanged.connect(self._on_any_changed)
        self.mdh_ve_factor_spin.valueChanged.connect(self._on_any_changed)
        self.mdh_save_8bit_cb.toggled.connect(self._on_any_changed)

        self.svf_noise_remove_spin.valueChanged.connect(self._on_any_changed)
        self.svf_num_directions_spin.valueChanged.connect(self._on_any_changed)
        self.svf_radius_spin.valueChanged.connect(self._on_any_changed)
        self.svf_ve_factor_spin.valueChanged.connect(self._on_any_changed)
        self.svf_save_8bit_cb.toggled.connect(self._on_any_changed)

        self.slope_unit_combo.currentIndexChanged.connect(self._on_any_changed)
        self.slope_ve_factor_spin.valueChanged.connect(self._on_any_changed)
        self.slope_save_8bit_cb.toggled.connect(self._on_any_changed)

        self.ldo_angular_res_spin.valueChanged.connect(self._on_any_changed)
        self.ldo_min_radius_spin.valueChanged.connect(self._on_any_changed)
        self.ldo_max_radius_spin.valueChanged.connect(self._on_any_changed)
        self.ldo_observer_h_spin.valueChanged.connect(self._on_any_changed)
        self.ldo_ve_factor_spin.valueChanged.connect(self._on_any_changed)
        self.ldo_save_8bit_cb.toggled.connect(self._on_any_changed)

        self.slrm_radius_spin.valueChanged.connect(self._on_any_changed)
        self.slrm_ve_factor_spin.valueChanged.connect(self._on_any_changed)
        self.slrm_save_8bit_cb.toggled.connect(self._on_any_changed)

        self.vat_terrain_type_combo.currentIndexChanged.connect(self._on_any_changed)
        self.vat_save_8bit_cb.toggled.connect(self._on_any_changed)

        self.detection_enabled_cb.toggled.connect(self._on_det_enabled_changed)
        self.det_runs_table.cellChanged.connect(self._on_any_changed)
        self.det_runs_table.currentCellChanged.connect(self._on_det_run_selection_changed)
        self.det_confidence_spin.valueChanged.connect(self._on_any_changed)
        self.det_iou_spin.valueChanged.connect(self._on_any_changed)
        self.det_generate_annotated_cb.toggled.connect(self._on_any_changed)
        self.det_generate_shp_cb.toggled.connect(self._on_any_changed)

        for key, _label, _default, is_rvt in PRODUCTS:
            if is_rvt:
                self._product_cbs[key].toggled.connect(self._on_rvt_products_changed)

    def _on_any_changed(self) -> None:
        if self._loading:
            return
        self._refresh_path_validations()
        self._save_from_widgets()

    def _on_specific_source_changed(self) -> None:
        if self._loading:
            return
        self._save_specific_source_only()
        self._refresh_path_validations()

    def _on_rvt_products_changed(self) -> None:
        if self._loading:
            return
        self._update_available_rvt_targets()
        self._save_from_widgets()

    def _on_det_enabled_changed(self) -> None:
        if self._loading:
            return
        self._apply_detection_state()
        self._save_from_widgets()

    def _on_det_run_selection_changed(self) -> None:
        if self._loading:
            return
        self._refresh_model_classes()

    def closeEvent(self, event):
        self._save_specific_source_only()
        self._save_from_widgets()
        super().closeEvent(event)

    # ═════════════════════════════════════════════
    # Labels dynamiques
    # ═════════════════════════════════════════════

    def _update_confidence_label(self):
        val = self.det_confidence_spin.value()
        if val < 0.2:
            text = "Très permissif"
            color = "#c0392b"
        elif val < 0.35:
            text = "Détection large"
            color = "#e67e22"
        elif val < 0.55:
            text = "Équilibré"
            color = "#27ae60"
        elif val < 0.75:
            text = "Sélectif"
            color = "#2980b9"
        else:
            text = "Très strict"
            color = "#8e44ad"
        self._conf_label.setText(text)
        self._conf_label.setStyleSheet(f"color: {color}; font-style: italic; margin-left: 8px;")

    def _update_workers_hint(self):
        w = self.max_workers_spin.value()
        ram = w * 2
        self._workers_hint.setText(f"≈ {ram} Go RAM disponible nécessaire")

    def _update_mode_description(self):
        mode = self.data_mode_combo.currentData() or "ign_laz"
        descriptions = {
            "ign_laz": "Les dalles LiDAR seront téléchargées automatiquement depuis l'IGN pour la zone délimitée par votre polygone.",
            "local_laz": "Indiquez le dossier contenant vos fichiers de nuages de points (LAZ ou LAS).",
            "existing_mnt": "Indiquez le dossier contenant vos MNT déjà calculés (formats TIF ou ASC).",
            "existing_rvt": "Indiquez le dossier contenant vos indices de visualisation déjà calculés (format TIF).",
        }
        labels = {
            "ign_laz": "<b>2.</b> Zone d'étude (polygone) :",
            "local_laz": "<b>2.</b> Dossier des nuages :",
            "existing_mnt": "<b>2.</b> Dossier des MNT :",
            "existing_rvt": "<b>2.</b> Dossier des indices :",
        }
        placeholders = {
            "ign_laz": "Fichier shapefile délimitant la zone d'étude (.shp)",
            "local_laz": "Dossier contenant les fichiers LAZ/LAS",
            "existing_mnt": "Dossier contenant les fichiers TIF/ASC",
            "existing_rvt": "Dossier contenant les fichiers TIF",
        }
        self._mode_description.setText(descriptions.get(mode, ""))
        self._step2_label.setText(labels.get(mode, "<b>2.</b> Source :"))
        self.specific_source_edit.setPlaceholderText(placeholders.get(mode, ""))

    # ═════════════════════════════════════════════
    # Gestion des modes de données
    # ═════════════════════════════════════════════

    def _mode_mapping(self, mode: str) -> tuple:
        """Retourne (config_key, label_text, is_file) pour le mode donné."""
        if mode == "ign_laz":
            return "input_file", "Fichier dalles IGN (liste URLs):", True
        if mode == "local_laz":
            return "local_laz_dir", "Dossier nuages locaux (LAZ/LAS):", False
        if mode == "existing_mnt":
            return "existing_mnt_dir", "Dossier MNT existants (TIF/ASC):", False
        if mode == "existing_rvt":
            return "existing_rvt_dir", "Dossier indices RVT existants (TIF RVT):", False
        return "input_file", "Fichier dalles IGN (liste URLs):", True

    def _on_data_mode_changed_internal(self) -> None:
        if self._loading:
            return

        # Sauvegarder la source du mode précédent
        prev_mode = self._current_mode or "ign_laz"
        prev_key, _, _ = self._mode_mapping(str(prev_mode))
        app = self._config.setdefault("app", {})
        files = app.setdefault("files", {})
        files[prev_key] = self.specific_source_edit.text().strip()
        self._config_manager.save_last_ui_config(self._config)

        self._apply_data_mode_state()
        self._refresh_path_validations()
        self._save_from_widgets()

    def _apply_data_mode_state(self) -> None:
        mode = self.data_mode_combo.currentData()
        if mode is None:
            mode = "ign_laz"
        self._current_mode = mode

        # Restaurer la source correspondant au mode
        files = (self._config.get("app") or {}).get("files") or {}
        key, _, _ = self._mode_mapping(mode)
        self._loading = True
        try:
            self.specific_source_edit.setText(files.get(key) or "")
        finally:
            self._loading = False

        self._update_mode_description()

        # Bouton "Couche QGIS" uniquement en mode IGN
        self._qgis_layer_btn.setVisible(mode == "ign_laz")

        # Visibilité
        needs_products = mode != "existing_rvt"
        self._products_group.setVisible(needs_products)

        mnt_needed = mode not in ("existing_mnt", "existing_rvt")
        if "MNT" in self._product_cbs:
            self._product_cbs["MNT"].setVisible(mnt_needed)
        if "DENSITE" in self._product_cbs:
            self._product_cbs["DENSITE"].setVisible(mnt_needed)

        self._apply_mode()
        self._update_available_rvt_targets()
        # Masquer la colonne "Indice cible" en mode existing_rvt (un seul dossier, nom générique)
        is_existing_rvt = (mode == "existing_rvt")
        self.det_runs_table.setColumnHidden(1, is_existing_rvt)
        self._refresh_path_validations()

    def _on_data_mode_changed(self):
        """Slot pour le signal du combo data_mode (aussi appelé depuis _build_sources_section)."""
        self._on_data_mode_changed_internal()

    # ═════════════════════════════════════════════
    # Validation des chemins
    # ═════════════════════════════════════════════

    def _refresh_path_validations(self) -> None:
        self._set_lineedit_path_state(self.output_dir_edit, expect_dir=True, allow_create=True)
        mode = self.data_mode_combo.currentData() or self._current_mode or "ign_laz"
        _, _, is_file = self._mode_mapping(mode)
        self._set_lineedit_path_state(self.specific_source_edit, expect_dir=not is_file)

    def _set_lineedit_path_state(self, edit: QLineEdit, expect_dir: bool, allow_create: bool = False) -> None:
        text = (edit.text() or "").strip()
        if not text:
            edit.setStyleSheet("")
            return
        p = Path(text)
        ok = p.exists() and (p.is_dir() if expect_dir else p.is_file())
        if not ok and allow_create and expect_dir:
            ok = True
        if ok:
            edit.setStyleSheet("")
        else:
            edit.setStyleSheet("QLineEdit { background-color: #ffd6d6; }")

    # ═════════════════════════════════════════════
    # Gestion des modèles de détection
    # ═════════════════════════════════════════════

    def _get_available_models(self) -> list:
        """Retourne la liste des modèles disponibles: [(label, path), ...]"""
        p = self._plugin_root / "data" / "models"
        items: list = []
        if p.exists() and p.is_dir():
            for model_dir in p.iterdir():
                if not model_dir.is_dir():
                    continue
                weights_dir = model_dir / "weights"
                weights_file = None
                for candidate in ("best.onnx", "best.pt"):
                    f = weights_dir / candidate
                    if f.exists() and f.is_file():
                        weights_file = f
                        break
                if weights_file:
                    items.append((model_dir.name, str(weights_file)))
            if not items:
                for ext in ("*.pt", "*.onnx", "*.engine"):
                    for f in p.rglob(ext):
                        if f.is_file():
                            items.append((f.name, str(f)))
        return sorted({(label, path) for label, path in items}, key=lambda t: t[0].lower())

    def _get_available_rvt_keys(self) -> list:
        """Retourne les clés RVT cochées: [(key, label), ...]. Fallback: toutes si aucune cochée."""
        if (self.data_mode_combo.currentData() or "") == "existing_rvt":
            return [("RVT", "RVT (dossier fourni)")]
        checked = [
            (key, label) for key, label, _d, is_rvt in PRODUCTS
            if is_rvt and self._product_cbs.get(key, None) is not None and self._product_cbs[key].isChecked()
        ]
        if checked:
            return checked
        return [(key, label) for key, label, _d, is_rvt in PRODUCTS if is_rvt]

    def _get_model_preferred_rvt(self, model_path: str) -> Optional[str]:
        """Lit le fichier training_params.json du modèle et retourne le type RVT préféré
        (clé rvt.type), ou None si introuvable.
        """
        if not model_path:
            return None
        try:
            p = Path(model_path)
            model_dir = p.parent
            if model_dir.name == "weights":
                model_dir = model_dir.parent
            tp_file = model_dir / "training_params.json"
            if not tp_file.exists():
                return None
            with tp_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            rvt = (data or {}).get("rvt") or {}
            rvt_type = rvt.get("type")
            if isinstance(rvt_type, str) and rvt_type.strip():
                return rvt_type.strip()
        except Exception as e:
            self._logger.warning(f"Erreur lecture training_params.json pour {model_path}: {e}")
        return None

    def _apply_preferred_rvt_for_row(self, row: int) -> None:
        """Sélectionne automatiquement le RVT préféré (depuis training_params.json)
        pour la ligne donnée, si disponible dans la combo.
        """
        if row < 0 or row >= self.det_runs_table.rowCount():
            return
        model_combo = self.det_runs_table.cellWidget(row, 0)
        rvt_combo = self.det_runs_table.cellWidget(row, 1)
        if not isinstance(model_combo, QComboBox) or not isinstance(rvt_combo, QComboBox):
            return
        model_path = str(model_combo.currentData() or "")
        preferred = self._get_model_preferred_rvt(model_path)
        if not preferred:
            return
        idx = rvt_combo.findData(preferred)
        if idx >= 0 and rvt_combo.currentIndex() != idx:
            rvt_combo.setCurrentIndex(idx)

    def _get_model_preferred_min_area(self, model_path: str) -> Optional[float]:
        """Lit le fichier training_params.json du modèle et retourne l'aire minimale
        préférée en m² (clé `detection.min_area_m2`, ou `min_area_m2` à la racine),
        ou None si introuvable.
        """
        if not model_path:
            return None
        try:
            p = Path(model_path)
            model_dir = p.parent
            if model_dir.name == "weights":
                model_dir = model_dir.parent
            tp_file = model_dir / "training_params.json"
            if not tp_file.exists():
                return None
            with tp_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            detection = (data or {}).get("detection") or {}
            value = detection.get("min_area_m2")
            if value is None:
                value = (data or {}).get("min_area_m2")
            if value is None:
                return None
            return float(value)
        except Exception as e:
            self._logger.warning(f"Erreur lecture min_area_m2 depuis training_params.json pour {model_path}: {e}")
        return None

    def _apply_preferred_min_area_for_row(self, row: int) -> None:
        """Applique automatiquement l'aire minimale préférée (depuis training_params.json)
        sur la ligne donnée, si disponible.
        """
        if row < 0 or row >= self.det_runs_table.rowCount():
            return
        model_combo = self.det_runs_table.cellWidget(row, 0)
        area_spin = self.det_runs_table.cellWidget(row, 2)
        if not isinstance(model_combo, QComboBox) or not isinstance(area_spin, QDoubleSpinBox):
            return
        model_path = str(model_combo.currentData() or "")
        preferred = self._get_model_preferred_min_area(model_path)
        if preferred is None:
            return
        if area_spin.value() != preferred:
            area_spin.setValue(preferred)

    def _on_det_run_model_changed(self) -> None:
        """Quand l'utilisateur change le modèle d'une ligne, aligne automatiquement
        la combo RVT et l'aire minimale sur les valeurs préférées du modèle
        (d'après training_params.json).
        """
        if self._loading:
            return
        sender = self.sender()
        if sender is None:
            return
        for r in range(self.det_runs_table.rowCount()):
            if self.det_runs_table.cellWidget(r, 0) is sender:
                self._apply_preferred_rvt_for_row(r)
                self._apply_preferred_min_area_for_row(r)
                return

    def _add_det_run_row(self, model_name: str = "", target_rvt: Optional[str] = None, min_area_m2: Optional[float] = None) -> None:
        """Ajoute une ligne au tableau des runs détection.

        Si `target_rvt` vaut None, le RVT est déduit automatiquement depuis
        `training_params.json` du modèle sélectionné (clé `rvt.type`).
        Si `min_area_m2` vaut None, l'aire minimale est déduite automatiquement
        depuis `training_params.json` (clé `detection.min_area_m2`).
        """
        auto_rvt = target_rvt is None
        auto_area = min_area_m2 is None
        initial_area = float(min_area_m2) if min_area_m2 is not None else 0.0
        self._loading = True
        try:
            row = self.det_runs_table.rowCount()
            self.det_runs_table.insertRow(row)

            # Combo modèle
            model_combo = NoWheelComboBox()
            available_models = self._get_available_models()
            for label, path in available_models:
                model_combo.addItem(label, path)
            if model_name:
                for i in range(model_combo.count()):
                    data = str(model_combo.itemData(i) or "")
                    text = model_combo.itemText(i)
                    if model_name in (data, text) or text == model_name:
                        model_combo.setCurrentIndex(i)
                        break
            model_combo.currentIndexChanged.connect(self._on_any_changed)
            model_combo.currentIndexChanged.connect(self._refresh_model_classes)
            model_combo.currentIndexChanged.connect(self._on_det_run_model_changed)
            self.det_runs_table.setCellWidget(row, 0, model_combo)

            # Combo RVT
            rvt_combo = NoWheelComboBox()
            rvt_keys = self._get_available_rvt_keys()
            for key, label in rvt_keys:
                rvt_combo.addItem(label, key)
            if target_rvt:
                idx = rvt_combo.findData(target_rvt)
                if idx >= 0:
                    rvt_combo.setCurrentIndex(idx)
            rvt_combo.currentIndexChanged.connect(self._on_any_changed)
            self.det_runs_table.setCellWidget(row, 1, rvt_combo)

            # Spinbox aire min (m²)
            area_spin = NoWheelDoubleSpinBox()
            area_spin.setDecimals(0)
            area_spin.setRange(0.0, 100000.0)
            area_spin.setSingleStep(50.0)
            area_spin.setValue(initial_area)
            area_spin.setSuffix(" m²")
            area_spin.setToolTip("Aire minimale en m² (0 = pas de filtrage). Les détections plus petites seront supprimées.")
            area_spin.setMinimumWidth(90)
            area_spin.valueChanged.connect(self._on_any_changed)
            self.det_runs_table.setCellWidget(row, 2, area_spin)

            # Boutons actions (info + supprimer)
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(2, 0, 2, 0)
            actions_layout.setSpacing(2)

            info_btn = QPushButton("ℹ")
            info_btn.setFixedSize(24, 24)
            info_btn.setToolTip("Paramètres d'entraînement")
            info_btn.clicked.connect(self._on_row_info_clicked)

            del_btn = QPushButton("×")
            del_btn.setFixedSize(24, 24)
            del_btn.setToolTip("Supprimer ce modèle")
            del_btn.clicked.connect(self._on_row_delete_clicked)

            actions_layout.addWidget(info_btn)
            actions_layout.addWidget(del_btn)
            self.det_runs_table.setCellWidget(row, 3, actions_widget)
        finally:
            self._loading = False
        if auto_rvt:
            self._apply_preferred_rvt_for_row(row)
        if auto_area:
            self._apply_preferred_min_area_for_row(row)
        self._refresh_model_classes()

    def _find_row_for_sender(self) -> int:
        btn = self.sender()
        if btn is None:
            return -1
        parent = btn.parentWidget()
        for row in range(self.det_runs_table.rowCount()):
            if self.det_runs_table.cellWidget(row, 3) is parent:
                return row
        return -1

    def _on_row_delete_clicked(self) -> None:
        row = self._find_row_for_sender()
        if row < 0:
            return
        self.det_runs_table.blockSignals(True)
        try:
            self.det_runs_table.removeRow(row)
            new_count = self.det_runs_table.rowCount()
            if new_count > 0:
                self.det_runs_table.setCurrentCell(min(row, new_count - 1), 0)
            else:
                self.det_runs_table.setCurrentCell(-1, -1)
        finally:
            self.det_runs_table.blockSignals(False)
        self._refresh_model_classes()
        if not self._loading:
            self._save_from_widgets()

    def _on_row_info_clicked(self) -> None:
        row = self._find_row_for_sender()
        if row < 0:
            return
        self._show_model_training_info_for_row(row)

    def _show_model_training_info_for_row(self, row: int) -> None:
        combo = self.det_runs_table.cellWidget(row, 0)
        if not isinstance(combo, QComboBox):
            return
        model_path = str(combo.currentData() or "")
        model_name = combo.currentText() or ""
        if not model_path:
            QMessageBox.information(self, "Info modèle", "Aucun modèle sélectionné sur cette ligne.")
            return

        model_file = Path(model_path)
        if not model_file.exists():
            QMessageBox.warning(self, "Info modèle", f"Le fichier du modèle n'existe pas:\n{model_path}")
            return

        model_dir = model_file.parent
        if model_dir.name == "weights":
            model_dir = model_dir.parent

        training_params_file = model_dir / "training_params.json"
        params = {}
        if training_params_file.exists():
            try:
                with training_params_file.open("r", encoding="utf-8") as f:
                    params = json.load(f)
            except Exception as e:
                self._logger.warning(f"Erreur lecture training_params.json: {e}")

        config_json_file = model_dir / "config.json"
        config_data = {}
        if config_json_file.exists():
            try:
                with config_json_file.open("r", encoding="utf-8") as f:
                    config_data = json.load(f)
            except Exception:
                pass

        args_data = {}
        args_file = model_dir / "args.yaml"
        if args_file.exists():
            try:
                import yaml
                with args_file.open("r", encoding="utf-8") as f:
                    args_data = yaml.safe_load(f) or {}
            except Exception:
                pass

        if not params and not config_data and not args_data:
            QMessageBox.information(
                self, f"Information - {model_name}",
                f"Aucune information disponible pour ce modèle.\n\nFichiers attendus dans:\n{model_dir}"
            )
            return

        info_lines = [f"Modèle: {model_name}\n"]
        if "description" in params:
            info_lines.append(f"{params['description']}\n")
        if "creation_date" in params:
            info_lines.append(f"Date de création: {params['creation_date']}")
        info_lines.append("=" * 50)

        # ── Architecture ──────────────────────────────────────────────
        model_info = params.get("model", {})
        cfg_model = config_data.get("model", {})
        arch = model_info.get("architecture") or cfg_model.get("architecture") or args_data.get("model", "")
        task = model_info.get("task") or cfg_model.get("task") or args_data.get("task", "")
        encoder = model_info.get("encoder") or cfg_model.get("encoder", "")
        imgsz = model_info.get("imgsz") or cfg_model.get("resolution") or args_data.get("imgsz", "")
        variant = model_info.get("variant") or cfg_model.get("variant", "")
        num_classes = cfg_model.get("num_classes", "")
        if arch or task:
            info_lines.append("\nArchitecture:")
            if arch:
                arch_str = arch
                if variant:
                    arch_str += f" ({variant})"
                info_lines.append(f"  • Modèle: {arch_str}")
            if encoder:
                info_lines.append(f"  • Encoder: {encoder}")
            if task:
                _task_labels = {
                    "detect": "Détection (bounding box)",
                    "detection": "Détection (bounding box)",
                    "instance_segmentation": "Segmentation d'instances",
                    "semantic_segmentation": "Segmentation sémantique",
                    "segment": "Segmentation",
                }
                info_lines.append(f"  • Tâche: {_task_labels.get(task, task)}")
            if imgsz:
                info_lines.append(f"  • Taille d'image: {imgsz}×{imgsz} px")

        # ── Classes ───────────────────────────────────────────────────
        class_names = cfg_model.get("class_names", [])
        if not class_names:
            classes_file = model_dir / "classes.txt"
            if classes_file.exists():
                try:
                    class_names = [ln.strip() for ln in classes_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
                except Exception:
                    pass
        if class_names:
            nc = num_classes or len(class_names)
            info_lines.append(f"  • Classes ({nc}): {', '.join(class_names)}")

        # ── Indice RVT d'entraînement ─────────────────────────────────
        rvt_info = params.get("rvt", {})
        if rvt_info:
            rvt_type = rvt_info.get("type", "")
            rvt_params = rvt_info.get("params", {})
            _rvt_labels = {
                "LD": "Local Dominance (LD)",
                "SVF": "Sky View Factor (SVF)",
                "HS": "Hillshade (HS)",
                "M_HS": "Multi-Hillshade (M_HS)",
                "SLRM": "Simple Local Relief Model (SLRM)",
                "VAT": "Visualisation à teinte variable (VAT)",
                "SLOPE": "Pente (Slope)",
            }
            info_lines.append(f"\nIndice RVT d'entraînement: {_rvt_labels.get(rvt_type, rvt_type)}")
            if rvt_params:
                _param_labels = {
                    "angular_res": "Résolution angulaire",
                    "min_radius": "Rayon min",
                    "max_radius": "Rayon max",
                    "observer_h": "Hauteur observateur",
                    "ve_factor": "Facteur d'exagération",
                    "svf_n_dir": "Nb directions",
                    "svf_r_max": "Rayon max",
                    "svf_noise": "Filtre bruit",
                    "save_as_8bit": "Export 8 bits",
                }
                for k, v in rvt_params.items():
                    lbl = _param_labels.get(k, k)
                    info_lines.append(f"  • {lbl}: {v}")

        # ── MNT ───────────────────────────────────────────────────────
        mnt_info = params.get("mnt", {})
        if mnt_info:
            info_lines.append("\nMNT d'entraînement:")
            if "resolution" in mnt_info:
                info_lines.append(f"  • Résolution: {mnt_info['resolution']} m/px")
            if "filter_expression" in mnt_info:
                info_lines.append(f"  • Filtre classification LiDAR: {mnt_info['filter_expression']}")

        # ── Entraînement ──────────────────────────────────────────────
        training = config_data.get("training", {})
        if training:
            info_lines.append("\nEntraînement:")
            for k, lbl in [
                ("num_epochs", "Époques"),
                ("batch_size", "Batch size"),
                ("effective_batch_size", "Batch size effectif"),
                ("learning_rate", "Learning rate"),
                ("lr_encoder", "LR encoder"),
                ("lr_scheduler", "Scheduler LR"),
                ("weight_decay", "Weight decay"),
                ("use_amp", "Mixed precision (AMP)"),
                ("use_ema", "EMA"),
                ("early_stopping", "Early stopping"),
                ("early_stopping_patience", "Patience early stopping"),
                ("seed", "Seed"),
            ]:
                if k in training:
                    info_lines.append(f"  • {lbl}: {training[k]}")

        # ── Jeu de données ────────────────────────────────────────────
        dataset = config_data.get("dataset", {})
        if dataset:
            info_lines.append("\nJeu de données:")
            if "roboflow_project" in dataset:
                info_lines.append(f"  • Projet: {dataset['roboflow_project']}")
            if "roboflow_version" in dataset:
                info_lines.append(f"  • Version: {dataset['roboflow_version']}")

        # ── SAHI ──────────────────────────────────────────────────────
        sahi = args_data.get("sahi", {})
        if sahi:
            info_lines.append(f"\nInférence SAHI (slicing):")
            info_lines.append(f"  • Taille de tuile: {sahi.get('slice_height', 'N/A')}×{sahi.get('slice_width', 'N/A')} px")
            info_lines.append(f"  • Recouvrement: {sahi.get('overlap_ratio', 'N/A')}")

        # ── Clustering ────────────────────────────────────────────────
        clustering_cfgs = args_data.get("clustering", [])
        if clustering_cfgs:
            info_lines.append("\nClustering spatial (DBSCAN):")
            for cc in clustering_cfgs:
                targets = ", ".join(cc.get("target_classes", []))
                out_cls = cc.get("output_class_name", "")
                info_lines.append(f"  • {targets} → {out_cls}")
                info_lines.append(f"    Distance max (eps_m): {cc.get('eps_m', 'N/A')} m")
                info_lines.append(f"    Détections min par groupe: {cc.get('min_cluster_size', 'N/A')}")
                info_lines.append(f"    Confiance min: {cc.get('min_confidence', 'N/A')}")
                info_lines.append(f"    Géométrie de sortie: {cc.get('output_geometry', 'N/A')}")

        # ── Métriques d'évaluation ────────────────────────────────────
        eval_file = model_dir / "evaluation_results.json"
        if eval_file.exists():
            try:
                with eval_file.open("r", encoding="utf-8") as _f:
                    eval_data = json.load(_f)
                # Privilégier le split "test", sinon "valid"
                eval_split = eval_data.get("test") or eval_data.get("validation") or eval_data.get("valid")
                split_name = "test" if "test" in eval_data else "validation"
                if eval_split:
                    gm = eval_split.get("global_metrics", {})
                    info_lines.append(f"\nMétriques d'évaluation ({split_name}, {eval_split.get('num_images', '?')} images):")
                    if "mAP_50" in gm:
                        info_lines.append(f"  • mAP@50: {gm['mAP_50']:.3f}")
                    if "mAP_50_95" in gm:
                        info_lines.append(f"  • mAP@50:95: {gm['mAP_50_95']:.3f}")
                    if "precision" in gm:
                        info_lines.append(f"  • Précision: {gm['precision']:.3f}")
                    if "recall" in gm:
                        info_lines.append(f"  • Rappel: {gm['recall']:.3f}")
                    if "f1" in gm:
                        info_lines.append(f"  • F1: {gm['f1']:.3f}")
                    per_class = eval_split.get("per_class", {})
                    if per_class:
                        info_lines.append("  Par classe:")
                        for cls, cm in per_class.items():
                            f1 = cm.get("f1")
                            ap = cm.get("ap_50_95") or cm.get("ap_50")
                            parts_cls = []
                            if f1 is not None:
                                parts_cls.append(f"F1={f1:.3f}")
                            if ap is not None:
                                parts_cls.append(f"AP={ap:.3f}")
                            info_lines.append(f"    - {cls}: {', '.join(parts_cls)}")
            except Exception:
                pass

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Information - {model_name}")
        dialog.setMinimumWidth(560)
        dialog.setMinimumHeight(480)
        dialog_layout = QVBoxLayout(dialog)
        from qgis.PyQt.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        info_label = QLabel("\n".join(info_lines))
        info_label.setWordWrap(True)
        info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info_label.setContentsMargins(8, 8, 8, 8)
        scroll.setWidget(info_label)
        dialog_layout.addWidget(scroll)

        btn_layout = QHBoxLayout()
        open_folder_btn = QPushButton("Ouvrir le dossier")
        open_folder_btn.clicked.connect(lambda: self._open_model_folder(model_dir))
        btn_layout.addWidget(open_folder_btn)
        btn_layout.addStretch(1)
        close_btn = QPushButton("Fermer")
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(close_btn)
        dialog_layout.addLayout(btn_layout)
        dialog.exec()

    def _refresh_models(self) -> None:
        available_models = self._get_available_models()
        self._loading = True
        try:
            for row in range(self.det_runs_table.rowCount()):
                combo = self.det_runs_table.cellWidget(row, 0)
                if not isinstance(combo, QComboBox):
                    continue
                current = str(combo.currentData() or combo.currentText() or "")
                combo.clear()
                for label, path in available_models:
                    combo.addItem(label, path)
                if current:
                    idx = combo.findData(current)
                    if idx < 0:
                        idx = combo.findText(current)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
        finally:
            self._loading = False
        self._refresh_model_classes()

    def _update_available_rvt_targets(self) -> None:
        rvt_keys = self._get_available_rvt_keys()
        self._loading = True
        try:
            for row in range(self.det_runs_table.rowCount()):
                combo = self.det_runs_table.cellWidget(row, 1)
                if not isinstance(combo, QComboBox):
                    continue
                current = str(combo.currentData() or "")
                combo.clear()
                for key, label in rvt_keys:
                    combo.addItem(label, key)
                if current:
                    idx = combo.findData(current)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
        finally:
            self._loading = False

    @staticmethod
    def _load_classes_for_model_path(model_path: str) -> list:
        if not model_path:
            return []
        model_file = Path(model_path)
        if not model_file.exists():
            return []
        model_dir = model_file.parent
        if model_dir.name == "weights":
            model_dir = model_dir.parent

        class_names: list = []
        for candidate in (
            model_dir / "classes.txt",
            model_dir / "classes.txt.txt",
            model_dir / "class_names.txt",
            model_dir / "class_names.txt.txt",
        ):
            try:
                if candidate.exists() and candidate.is_file():
                    lines = [ln.strip() for ln in candidate.read_text(encoding="utf-8-sig").splitlines()]
                    class_names = [ln for ln in lines if ln]
                    if class_names:
                        break
            except Exception:
                continue

        if not class_names:
            try:
                for candidate in (model_dir / "classes.json", model_dir / "class_names.json"):
                    if candidate.exists() and candidate.is_file():
                        parsed = json.loads(candidate.read_text(encoding="utf-8"))
                        if isinstance(parsed, list):
                            class_names = [str(c) for c in parsed]
                        elif isinstance(parsed, dict):
                            class_names = [str(parsed[k]) for k in sorted(parsed.keys(), key=lambda x: int(x) if str(x).isdigit() else x)]
                        if class_names:
                            break
            except Exception:
                pass

        # Ajouter les classes de sortie de clustering depuis args.yaml
        args_file = model_dir / "args.yaml"
        if args_file.exists():
            try:
                import yaml
                with args_file.open("r", encoding="utf-8") as _f:
                    args_data = yaml.safe_load(_f) or {}
                for cluster_cfg in args_data.get("clustering", []):
                    out_cls = str(cluster_cfg.get("output_class_name") or "").strip()
                    if out_cls and out_cls not in class_names:
                        class_names.append(out_cls)
            except Exception:
                pass

        return list(dict.fromkeys(class_names))

    def _refresh_model_classes(self) -> None:
        # Sauvegarder l'état coché actuel avant de reconstruire
        previous_state: dict = {}
        for i in range(self.det_classes_list.count()):
            item = self.det_classes_list.item(i)
            if not (item.flags() & Qt.ItemFlag.ItemIsUserCheckable):
                continue
            key = str(item.data(Qt.ItemDataRole.UserRole) or "")
            if "\t" in key:
                previous_state[key] = item.checkState()

        self.det_classes_list.clear()
        for row in range(self.det_runs_table.rowCount()):
            combo = self.det_runs_table.cellWidget(row, 0)
            if not isinstance(combo, QComboBox):
                continue
            model_path = str(combo.currentData() or "")
            model_label = combo.currentText() or "?"
            classes = self._load_classes_for_model_path(model_path)

            header = QListWidgetItem(f"── {model_label} ──")
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            font = header.font()
            font.setBold(True)
            header.setFont(font)
            header.setBackground(self.det_classes_list.palette().alternateBase())
            header.setData(Qt.ItemDataRole.UserRole, f"__header__{model_label}")
            self.det_classes_list.addItem(header)

            if not classes:
                info = QListWidgetItem("  (aucune classe trouvée)")
                info.setFlags(Qt.ItemFlag.NoItemFlags)
                info.setData(Qt.ItemDataRole.UserRole, f"__info__{model_label}")
                self.det_classes_list.addItem(info)
            else:
                for class_name in classes:
                    item = QListWidgetItem(f"  {class_name}")
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    key = f"{model_label}\t{class_name}"
                    # Restaurer l'état précédent ; coché par défaut pour les nouvelles classes
                    state = previous_state.get(key, Qt.CheckState.Checked)
                    item.setCheckState(state)
                    item.setData(Qt.ItemDataRole.UserRole, key)
                    self.det_classes_list.addItem(item)

    def _select_all_classes(self) -> None:
        for i in range(self.det_classes_list.count()):
            item = self.det_classes_list.item(i)
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(Qt.CheckState.Checked)

    def _deselect_all_classes(self) -> None:
        for i in range(self.det_classes_list.count()):
            item = self.det_classes_list.item(i)
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(Qt.CheckState.Unchecked)

    def _get_selected_classes(self) -> dict:
        result: dict = {}
        for i in range(self.det_classes_list.count()):
            item = self.det_classes_list.item(i)
            if item.checkState() != Qt.CheckState.Checked:
                continue
            data = str(item.data(Qt.ItemDataRole.UserRole) or "")
            if "\t" not in data:
                continue
            model_label, class_name = data.split("\t", 1)
            result.setdefault(model_label, []).append(class_name)
        return result

    def _get_selected_classes_flat(self) -> list:
        selected = []
        for classes in self._get_selected_classes().values():
            selected.extend(classes)
        return list(dict.fromkeys(selected))

    def _open_model_folder(self, folder_path: Path) -> None:
        import os
        import subprocess
        folder_str = str(folder_path)
        if sys.platform == "win32":
            os.startfile(folder_str)
        elif sys.platform == "darwin":
            subprocess.run(["open", folder_str])
        else:
            subprocess.run(["xdg-open", folder_str])

    # ═════════════════════════════════════════════
    # Boutons Reset
    # ═════════════════════════════════════════════

    def _reset_mnt_config(self) -> None:
        defaults = self._config_manager.load()
        processing = defaults.get("processing") or {}
        self._loading = True
        try:
            self.mnt_resolution_spin.setValue(float(processing.get("mnt_resolution", 0.5)))
            self.density_resolution_spin.setValue(float(processing.get("density_resolution", 1.0)))
            self.filter_expression_edit.setText(
                processing.get("filter_expression", "")
            )
        finally:
            self._loading = False
        self._save_from_widgets()
        self._logger.info("Paramètres MNT remis par défaut")

    def _reset_rvt_config(self) -> None:
        defaults = self._config_manager.load()
        rvt = defaults.get("rvt_params") or {}
        self._loading = True
        try:
            mdh = rvt.get("mdh") or {}
            self.mdh_num_directions_spin.setValue(int(mdh.get("num_directions", 16)))
            self.mdh_sun_elevation_spin.setValue(int(mdh.get("sun_elevation", 35)))
            self.mdh_ve_factor_spin.setValue(int(mdh.get("ve_factor", 1)))
            self.mdh_save_8bit_cb.setChecked(bool(mdh.get("save_as_8bit", True)))

            svf = rvt.get("svf") or {}
            self.svf_noise_remove_spin.setValue(int(svf.get("noise_remove", 0)))
            self.svf_num_directions_spin.setValue(int(svf.get("num_directions", 16)))
            self.svf_radius_spin.setValue(int(svf.get("radius", 10)))
            self.svf_ve_factor_spin.setValue(int(svf.get("ve_factor", 1)))
            self.svf_save_8bit_cb.setChecked(bool(svf.get("save_as_8bit", True)))

            slope = rvt.get("slope") or {}
            idx_unit = self.slope_unit_combo.findData(int(slope.get("unit", 0)))
            self.slope_unit_combo.setCurrentIndex(idx_unit if idx_unit >= 0 else 0)
            self.slope_ve_factor_spin.setValue(int(slope.get("ve_factor", 1)))
            self.slope_save_8bit_cb.setChecked(bool(slope.get("save_as_8bit", True)))

            ldo = rvt.get("ldo") or {}
            self.ldo_angular_res_spin.setValue(int(ldo.get("angular_res", 15)))
            self.ldo_min_radius_spin.setValue(int(ldo.get("min_radius", 10)))
            self.ldo_max_radius_spin.setValue(int(ldo.get("max_radius", 20)))
            self.ldo_observer_h_spin.setValue(float(ldo.get("observer_h", 1.7)))
            self.ldo_ve_factor_spin.setValue(int(ldo.get("ve_factor", 1)))
            self.ldo_save_8bit_cb.setChecked(bool(ldo.get("save_as_8bit", True)))

            slrm = rvt.get("slrm") or {}
            self.slrm_radius_spin.setValue(int(slrm.get("radius", 20)))
            self.slrm_ve_factor_spin.setValue(int(slrm.get("ve_factor", 1)))
            self.slrm_save_8bit_cb.setChecked(bool(slrm.get("save_as_8bit", True)))

            vat = rvt.get("vat") or {}
            idx_terrain = self.vat_terrain_type_combo.findData(int(vat.get("terrain_type", 0)))
            self.vat_terrain_type_combo.setCurrentIndex(idx_terrain if idx_terrain >= 0 else 0)
            self.vat_save_8bit_cb.setChecked(bool(vat.get("save_as_8bit", True)))
        finally:
            self._loading = False
        self._save_from_widgets()
        self._logger.info("Paramètres RVT remis par défaut")

    def _reset_perf_config(self) -> None:
        defaults = self._config_manager.load()
        processing = defaults.get("processing") or {}
        self._loading = True
        try:
            self.max_workers_spin.setValue(int(processing.get("max_workers", 4)))
        finally:
            self._loading = False
        self._save_from_widgets()
        self._logger.info("Paramètres Performance remis par défaut")

    def _reset_det_config(self) -> None:
        defaults = self._config_manager.load()
        cv = defaults.get("computer_vision") or {}
        self._loading = True
        try:
            self.detection_enabled_cb.setChecked(bool(cv.get("enabled", False)))
            self.det_confidence_spin.setValue(float(cv.get("confidence_threshold", 0.3)))
            self.det_iou_spin.setValue(float(cv.get("iou_threshold", 0.5)))
            self.det_generate_annotated_cb.setChecked(bool(cv.get("generate_annotated_images", False)))
            self.det_generate_shp_cb.setChecked(bool(cv.get("generate_shapefiles", False)))
            self.det_runs_table.setRowCount(0)
        finally:
            self._loading = False
        self._apply_detection_state()
        self._save_from_widgets()
        self._logger.info("Paramètres Détection remis par défaut")

    # ═════════════════════════════════════════════
    # Logs
    # ═════════════════════════════════════════════

    def _append_log(self, msg: str) -> None:
        self.logs_text.appendPlainText(msg)
        sb = self.logs_text.verticalScrollBar()
        sb.setValue(sb.maximum())


    def _set_progress(self, value: int) -> None:
        self.progress_bar.setValue(int(value))

    def _set_stage(self, text: str) -> None:
        self.stage_label.setText(text)

    def _set_run_enabled(self, enabled: bool) -> None:
        self.run_btn.setEnabled(bool(enabled))
        self.cancel_btn.setEnabled(not bool(enabled))
        self.save_config_btn.setEnabled(bool(enabled))
        self.load_config_btn.setEnabled(bool(enabled))
        self._config_scroll.widget().setEnabled(bool(enabled))
        if enabled:
            self._cancel_event.clear()
            self.stage_label.setText("")
            self.progress_bar.setValue(0)

    # ═════════════════════════════════════════════
    # Lancement du pipeline
    # ═════════════════════════════════════════════

    def _on_run_clicked(self) -> None:
        if not self.run_btn.isEnabled():
            return

        self._sync_config_from_widgets()

        cv_cfg = self._config.setdefault("computer_vision", {})
        if cv_cfg.get("enabled", False):
            classes_by_model = self._get_selected_classes()
            path_to_label: dict = {}
            for row in range(self.det_runs_table.rowCount()):
                combo = self.det_runs_table.cellWidget(row, 0)
                if isinstance(combo, QComboBox):
                    path_to_label[str(combo.currentData() or "")] = combo.currentText() or ""
            runs = cv_cfg.get("runs") or []
            for run in runs:
                model_path = str(run.get("model") or "")
                model_label = path_to_label.get(model_path, model_path)
                run["selected_classes"] = classes_by_model.get(model_label, [])
            cv_cfg.pop("selected_classes", None)

        mode_label = self.data_mode_combo.currentText()
        output = self.output_dir_edit.text().strip() or "(non défini)"
        active_products = [label for key, label, _d, _r in PRODUCTS if self._product_cbs[key].isChecked()]
        det_status = "activée" if cv_cfg.get("enabled") else "désactivée"

        # Détecter les modèles sans aucune classe sélectionnée
        skipped_models = []
        if cv_cfg.get("enabled", False):
            runs = cv_cfg.get("runs") or []
            for run in runs:
                model_path = str(run.get("model") or "")
                model_label = path_to_label.get(model_path, model_path)
                if not run.get("selected_classes"):
                    skipped_models.append(model_label)

        warning_line = ""
        if skipped_models:
            names = "\n".join(f"  • {m}" for m in skipped_models)
            warning_line = f"\n\n⚠ Modèle(s) ignoré(s) (0 classe sélectionnée) :\n{names}"

        summary = (
            f"Mode : {mode_label}\n"
            f"Sortie : {output}\n"
            f"Produits : {', '.join(active_products) if active_products else '(aucun)'}\n"
            f"Détection IA : {det_status}"
            f"{warning_line}\n\n"
            "Lancer le pipeline ?"
        )
        reply = QMessageBox.question(
            self, "Confirmation", summary,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._config_manager.save_last_ui_config(self._config)
        self._cancel_event.clear()
        self._set_run_enabled(False)
        self._logger.info("Lancement du pipeline…")

        def worker():
            try:
                from ..app.cancel_token import CancelToken
                from ..app.pipeline_controller import PipelineController, file_logging
                from ..app.qt_progress_reporter import QtProgressReporter
                from ..app.run_context import build_run_context

                reporter = QtProgressReporter(self._logger, self._log_emitter)
                ctx = build_run_context(self._config)
                with file_logging(ctx.output_dir, reporter):
                    PipelineController().run(ctx=ctx, reporter=reporter, cancel=CancelToken(self._cancel_event))
            except Exception:
                self._logger.exception("Erreur pendant l'exécution du pipeline")
            finally:
                self._log_emitter.run_enabled.emit(True)

        threading.Thread(target=worker, daemon=True).start()

    def _on_cancel_clicked(self) -> None:
        if self.cancel_btn.isEnabled():
            self._cancel_event.set()
            self._logger.info("Annulation demandée...")

    # ═════════════════════════════════════════════
    # Chargement des couches QGIS
    # ═════════════════════════════════════════════

    def _load_layers_to_project(self, vrt_paths: list, shapefile_paths: list, class_colors: list = None) -> None:
        try:
            from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer, QgsRectangle
            from ..pipeline.cv.class_utils import BASE_COLOR_PALETTE, get_color_for_confidence

            project = QgsProject.instance()
            loaded_count = 0
            combined_extent = QgsRectangle()
            loaded_layers = []
            class_colors = class_colors or []

            global_color_map: dict = {}
            if class_colors and len(class_colors) == 1 and isinstance(class_colors[0], dict):
                global_color_map = class_colors[0]
                class_colors = []

            for vrt_path in vrt_paths:
                if not vrt_path:
                    continue
                vrt_path_str = str(vrt_path)
                parts = vrt_path_str.replace("\\", "/").split("/")
                layer_name = "index"
                for i, part in enumerate(parts):
                    if part == "tif" and i > 0:
                        layer_name = parts[i - 1]
                        break
                    elif part == "MNT":
                        layer_name = "MNT"
                        break

                existing_layers = project.mapLayersByName(layer_name)
                already_loaded = False
                for existing in existing_layers:
                    if existing.source() == vrt_path_str:
                        already_loaded = True
                        if combined_extent.isNull():
                            combined_extent = existing.extent()
                        else:
                            combined_extent.combineExtentWith(existing.extent())
                        break
                if already_loaded:
                    self._logger.info(f"Couche raster déjà présente: {layer_name}")
                    continue

                layer = QgsRasterLayer(vrt_path_str, layer_name, "gdal")
                if layer.isValid():
                    project.addMapLayer(layer)
                    loaded_layers.append(layer)
                    if combined_extent.isNull():
                        combined_extent = layer.extent()
                    else:
                        combined_extent.combineExtentWith(layer.extent())
                    loaded_count += 1
                    self._logger.info(f"Couche raster chargée: {layer_name}")
                else:
                    self._logger.warning(f"Impossible de charger le VRT: {vrt_path_str}")

            for shp_path in shapefile_paths:
                if not shp_path:
                    continue
                shp_path_str = str(shp_path)

                # Support GPKG: "chemin.gpkg|layername=class_name"
                if '|layername=' in shp_path_str:
                    _gpkg_path, layer_name = shp_path_str.split('|layername=', 1)
                    class_name = layer_name
                    ogr_source = shp_path_str
                else:
                    layer_name = Path(shp_path_str).stem
                    parts = layer_name.split("_")
                    class_name = "_".join(parts[2:]) if len(parts) >= 3 and parts[0] == "detections" else layer_name
                    ogr_source = shp_path_str

                color_idx = 0
                if global_color_map and class_name in global_color_map:
                    color_idx = global_color_map[class_name]
                elif global_color_map:
                    for cname, cidx in global_color_map.items():
                        if cname.lower() in class_name.lower():
                            color_idx = cidx
                            break
                else:
                    try:
                        import re
                        temp_layer = QgsVectorLayer(ogr_source, "temp", "ogr")
                        if temp_layer.isValid() and temp_layer.featureCount() > 0:
                            for feat in temp_layer.getFeatures():
                                conf_color_val = feat.attribute("conf_color")
                                if conf_color_val:
                                    match = re.match(r"color(\d+)_", str(conf_color_val))
                                    if match:
                                        color_idx = int(match.group(1))
                                break
                        del temp_layer
                    except Exception as e:
                        self._logger.warning(f"Impossible d'extraire conf_color de {layer_name}: {e}")

                existing_layers = project.mapLayersByName(layer_name)
                already_loaded = False
                for existing in existing_layers:
                    if existing.source() == ogr_source:
                        already_loaded = True
                        if combined_extent.isNull():
                            combined_extent = existing.extent()
                        else:
                            combined_extent.combineExtentWith(existing.extent())
                        break
                if already_loaded:
                    self._logger.info(f"Couche vecteur déjà présente: {layer_name}")
                    continue

                layer = QgsVectorLayer(ogr_source, layer_name, "ogr")
                if layer.isValid():
                    _is_cluster = layer.fields().indexFromName("nb_detect") >= 0
                    if _is_cluster:
                        self._apply_cluster_style(layer)
                    else:
                        self._apply_confidence_style_unified(layer, color_idx, get_color_for_confidence)
                    project.addMapLayer(layer)
                    loaded_layers.append(layer)
                    if combined_extent.isNull():
                        combined_extent = layer.extent()
                    else:
                        combined_extent.combineExtentWith(layer.extent())
                    loaded_count += 1
                    base_color = BASE_COLOR_PALETTE[color_idx % len(BASE_COLOR_PALETTE)]
                    self._logger.info(f"Couche vecteur chargée: {layer_name} (classe={class_name}, couleur={color_idx} RGB{base_color})")
                else:
                    self._logger.warning(f"Impossible de charger la couche: {ogr_source}")

            if loaded_count > 0:
                self._logger.info(f"{loaded_count} couche(s) ajoutée(s) au projet QGIS")
                if not combined_extent.isNull():
                    try:
                        from qgis.utils import iface
                        if iface and iface.mapCanvas():
                            combined_extent.scale(1.05)
                            iface.mapCanvas().setExtent(combined_extent)
                            iface.mapCanvas().refresh()
                            self._logger.info("Zoom sur l'étendue des résultats")
                    except Exception as zoom_err:
                        self._logger.warning(f"Impossible de zoomer: {zoom_err}")

        except Exception as e:
            self._logger.error(f"Erreur lors du chargement des couches: {e}")

    def _apply_cluster_style(self, layer) -> None:
        try:
            from qgis.core import (
                QgsFillSymbol, QgsSingleSymbolRenderer,
                QgsLinePatternFillSymbolLayer, QgsSimpleLineSymbolLayer,
            )
            from qgis.PyQt.QtGui import QColor

            symbol = QgsFillSymbol()
            symbol.deleteSymbolLayer(0)

            hatch1 = QgsLinePatternFillSymbolLayer()
            hatch1.setLineAngle(45)
            hatch1.setDistance(3.0)
            hatch1.setLineWidth(0.4)
            hatch1.setColor(QColor(0, 0, 0))
            symbol.appendSymbolLayer(hatch1)

            hatch2 = QgsLinePatternFillSymbolLayer()
            hatch2.setLineAngle(135)
            hatch2.setDistance(3.0)
            hatch2.setLineWidth(0.4)
            hatch2.setColor(QColor(0, 0, 0))
            symbol.appendSymbolLayer(hatch2)

            outline = QgsSimpleLineSymbolLayer()
            outline.setColor(QColor(0, 0, 0))
            outline.setWidth(0.6)
            symbol.appendSymbolLayer(outline)

            renderer = QgsSingleSymbolRenderer(symbol)
            layer.setRenderer(renderer)
            layer.triggerRepaint()
            self._logger.info(f"Style cluster appliqué à {layer.name()}")
        except Exception as e:
            self._logger.warning(f"Impossible d'appliquer le style cluster: {e}")

    def _apply_confidence_style_unified(self, layer, color_idx: int, get_color_for_confidence_fn) -> None:
        try:
            from qgis.core import (
                QgsCategorizedSymbolRenderer, QgsRendererCategory, QgsFillSymbol,
            )
            from ..pipeline.cv.class_utils import compute_confidence_bins

            # Aligne la symbologie dynamique QGIS sur le seuil configuré dans
            # l'UI (Paramètres avancés → Niveau de certitude). Si threshold > 0,
            # les tranches entièrement sous ce seuil sont omises et la première
            # est tronquée — même logique que la symbologie écrite dans le .qgs.
            min_conf = 0.0
            try:
                cv_cfg = (self._config or {}).get("computer_vision") or {}
                min_conf = float(cv_cfg.get("confidence_threshold", 0.0) or 0.0)
            except Exception:
                min_conf = 0.0

            bins = compute_confidence_bins(min_conf)
            categories = []
            for b in bins:
                r, g, bl = get_color_for_confidence_fn(color_idx, b['repr'])
                rgb_str = f'{r},{g},{bl}'
                symbol = QgsFillSymbol.createSimple({
                    'color': '0,0,0,0',
                    'outline_color': f'{rgb_str},255',
                    'outline_width': '0.6',
                    'outline_style': 'solid',
                })
                category = QgsRendererCategory(b['label'], symbol, b['label'])
                categories.append(category)

            if not categories:
                return
            renderer = QgsCategorizedSymbolRenderer('conf_bin', categories)
            layer.setRenderer(renderer)
            layer.triggerRepaint()
        except Exception as e:
            self._logger.warning(f"Impossible d'appliquer le style: {e}")

    # ═════════════════════════════════════════════
    # Helpers
    # ═════════════════════════════════════════════

    @staticmethod
    def _make_bold_title(group: QGroupBox):
        group.setStyleSheet(
            "QGroupBox#" + group.objectName() + "::title { font-weight: bold; color: #1a5276; }"
            if group.objectName() else
            "QGroupBox::title { font-weight: bold; color: #1a5276; }"
        )

    def _row_widget(self, edit: QLineEdit, button: QPushButton) -> QWidget:
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(edit, 1)
        l.addWidget(button, 0)
        return w

    def _browse_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Sélectionner le dossier de sortie")
        if directory:
            self.output_dir_edit.setText(directory)

    def _browse_specific_source(self):
        mode = self.data_mode_combo.currentData() or "ign_laz"
        if mode == "ign_laz":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Sélectionner le polygone de la zone d'étude",
                "", "Shapefile (*.shp);;GeoJSON (*.geojson *.json);;GeoPackage (*.gpkg);;Tous (*.*)"
            )
            if file_path:
                self.specific_source_edit.setText(file_path)
        else:
            directory = QFileDialog.getExistingDirectory(self, "Sélectionner un dossier")
            if directory:
                self.specific_source_edit.setText(directory)

    def _pick_qgis_polygon_layer(self):
        """Ouvre un dialogue listant les couches polygone du projet QGIS."""
        try:
            from qgis.core import QgsProject, QgsWkbTypes, QgsVectorFileWriter
        except ImportError:
            QMessageBox.warning(self, "Erreur", "API QGIS non disponible.")
            return

        project = QgsProject.instance()
        polygon_layers = []
        for layer in project.mapLayers().values():
            try:
                if hasattr(layer, "geometryType") and layer.geometryType() == QgsWkbTypes.PolygonGeometry:
                    polygon_layers.append(layer)
            except Exception:
                continue

        if not polygon_layers:
            QMessageBox.information(
                self,
                "Aucune couche polygone",
                "Aucune couche polygone n'est chargée dans le projet QGIS.\n\n"
                "Créez ou chargez une couche polygone délimitant votre zone d'étude, "
                "puis réessayez.",
            )
            return

        # Dialogue de sélection
        from qgis.PyQt.QtWidgets import QInputDialog
        names = [f"{l.name()}  ({l.featureCount()} entités)" for l in polygon_layers]
        chosen, ok = QInputDialog.getItem(
            self,
            "Sélectionner une couche polygone",
            "Couche :",
            names,
            0,
            False,
        )
        if not ok:
            return

        idx = names.index(chosen)
        layer = polygon_layers[idx]

        # Si la couche a un fichier source sur disque, l'utiliser directement
        source_path = layer.source().split("|")[0].strip()  # enlever |layername=... etc.
        source_p = Path(source_path)
        # QGIS peut pointer vers le .dbf ; vérifier si le .shp frère existe
        if source_p.suffix.lower() == ".dbf":
            shp_sibling = source_p.with_suffix(".shp")
            if shp_sibling.exists():
                source_path = str(shp_sibling)
                source_p = shp_sibling
        if source_p.exists() and source_p.suffix.lower() in (
            ".shp", ".geojson", ".json", ".gpkg",
        ):
            self.specific_source_edit.setText(source_path)
            self._logger.info(f"Couche QGIS sélectionnée : {layer.name()} → {source_path}")
            return

        self._logger.info(f"Source couche '{layer.name()}' non trouvée sur disque (source={layer.source()!r}), export nécessaire")

        # Couche mémoire / virtuelle → exporter en shapefile dans le dossier du plugin
        export_dir = self._plugin_root / "data" / "temp_zones"
        export_dir.mkdir(parents=True, exist_ok=True)
        tmp_shp = export_dir / f"{layer.name().replace(' ', '_')}.shp"

        self._logger.info(f"Export de la couche mémoire '{layer.name()}' vers {tmp_shp}")
        save_options = QgsVectorFileWriter.SaveVectorOptions()
        save_options.driverName = "ESRI Shapefile"
        error = QgsVectorFileWriter.writeAsVectorFormatV3(
            layer,
            str(tmp_shp),
            project.transformContext(),
            save_options,
        )
        if error[0] != QgsVectorFileWriter.NoError:
            QMessageBox.warning(
                self,
                "Erreur d'export",
                f"Impossible d'exporter la couche '{layer.name()}'.\n{error[1]}",
            )
            return

        self.specific_source_edit.setText(str(tmp_shp))
        self._logger.info(f"Couche QGIS exportée : {layer.name()} → {tmp_shp}")
