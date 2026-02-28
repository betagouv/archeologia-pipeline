import logging
import threading
import time
from pathlib import Path

from qgis.PyQt.QtCore import Qt, QObject, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDialog,
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
    QDoubleSpinBox,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QScrollArea,
    QVBoxLayout,
    QWidget,
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


class NoWheelSpinBox(QSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)

    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event) -> None:
        event.ignore()


# (config_key, ui_label, default_checked, is_rvt)
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


class MainDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Archeolog'IA pipeline")

        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMinimizeButtonHint)

        self.setMinimumSize(800, 600)
        self.resize(800, 600)

        self._loading = False
        self._plugin_root = Path(__file__).resolve().parents[2]
        self._config_manager = ConfigManager(self._plugin_root)
        self._config = self._config_manager.load()
        self._current_mode = None
        self._cancel_event = threading.Event()

        layout = QVBoxLayout(self)

        self._config_scroll = config_scroll = QScrollArea()
        config_scroll.setWidgetResizable(True)
        config_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        config_container = QWidget()
        config_layout = QVBoxLayout(config_container)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.setSpacing(8)
        config_scroll.setWidget(config_container)
        layout.addWidget(config_scroll, 0)

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

        sources_group = QGroupBox("Sources de données et chemins")
        self._sources_layout = QFormLayout(sources_group)
        self._sources_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self.output_dir_edit = QLineEdit()
        self.output_dir_btn = QPushButton("Parcourir")
        self.output_dir_btn.clicked.connect(self._browse_output_dir)
        self._sources_layout.addRow("Dossier de sortie:", self._row_widget(self.output_dir_edit, self.output_dir_btn))

        self.data_mode_combo = NoWheelComboBox()
        self.data_mode_combo.addItem("Données IGN (LAZ via URL)", "ign_laz")
        self.data_mode_combo.addItem("Nuages locaux (LAZ/LAS)", "local_laz")
        self.data_mode_combo.addItem("MNT existants (ASC/TIF)", "existing_mnt")
        self.data_mode_combo.addItem("Indices RVT existants (TIF RVT)", "existing_rvt")
        self._sources_layout.addRow("Source des données (mode):", self.data_mode_combo)

        self.specific_source_edit = QLineEdit()
        self.specific_source_btn = QPushButton("Parcourir")
        self.specific_source_btn.clicked.connect(self._browse_specific_source)
        self.specific_source_row = self._row_widget(self.specific_source_edit, self.specific_source_btn)
        self._sources_layout.addRow("Source spécifique:", self.specific_source_row)
        self.specific_source_label = self._sources_layout.labelForField(self.specific_source_row)

        config_layout.addWidget(sources_group)

        self._general_group = general_group = QGroupBox("Configuration générale")
        general_group_layout = QVBoxLayout(general_group)
        general_content = QWidget()
        general_layout = QVBoxLayout(general_content)
        general_layout.setContentsMargins(0, 0, 0, 0)

        self._resolutions_row = resolutions_row = QWidget()
        resolutions_layout = QHBoxLayout(resolutions_row)
        resolutions_layout.setContentsMargins(0, 0, 0, 0)

        resolutions_layout.addWidget(QWidget(), 0)

        self.mnt_resolution_spin = NoWheelDoubleSpinBox()
        self.mnt_resolution_spin.setDecimals(2)
        self.mnt_resolution_spin.setRange(0.01, 1000.0)
        self.mnt_resolution_spin.setSingleStep(0.1)

        self.density_resolution_spin = NoWheelDoubleSpinBox()
        self.density_resolution_spin.setDecimals(2)
        self.density_resolution_spin.setRange(0.01, 1000.0)
        self.density_resolution_spin.setSingleStep(0.1)

        self.tile_overlap_spin = NoWheelSpinBox()
        self.tile_overlap_spin.setRange(0, 100)

        resolutions_layout.addWidget(QWidget(), 0)
        resolutions_layout.addWidget(QWidget(), 0)

        resolutions_layout.addWidget(QLabel("MNT:"))
        resolutions_layout.addWidget(self.mnt_resolution_spin)
        resolutions_layout.addWidget(QLabel("m"))
        resolutions_layout.addSpacing(12)

        resolutions_layout.addWidget(QLabel("Densité:"))
        resolutions_layout.addWidget(self.density_resolution_spin)
        resolutions_layout.addWidget(QLabel("m"))
        resolutions_layout.addSpacing(12)

        resolutions_layout.addWidget(QLabel("Marge:"))
        resolutions_layout.addWidget(self.tile_overlap_spin)
        resolutions_layout.addWidget(QLabel("%"))
        resolutions_layout.addSpacing(12)

        resolutions_layout.addWidget(QLabel("Workers:"))
        self.max_workers_spin = NoWheelSpinBox()
        self.max_workers_spin.setRange(1, 16)
        self.max_workers_spin.setToolTip("Nombre de téléchargements/prétraitements parallèles (1-16)")
        resolutions_layout.addWidget(self.max_workers_spin)
        resolutions_layout.addStretch(1)

        general_layout.addWidget(resolutions_row)

        self._filter_widget = QWidget()
        filter_form = QFormLayout(self._filter_widget)
        filter_form.setContentsMargins(0, 0, 0, 0)
        self.filter_expression_edit = QLineEdit()
        filter_form.addRow("Expression filtre:", self.filter_expression_edit)
        general_layout.addWidget(self._filter_widget)

        self._pyramids_row = pyramids_row = QWidget()
        pyramids_layout = QHBoxLayout(pyramids_row)
        pyramids_layout.setContentsMargins(0, 0, 0, 0)
        pyramids_layout.addWidget(QLabel("Pyramides raster:"))
        self.pyramids_enabled_cb = QCheckBox("Activer")
        pyramids_layout.addWidget(self.pyramids_enabled_cb)
        pyramids_layout.addSpacing(12)
        pyramids_layout.addWidget(QLabel("Niveaux:"))
        self.pyramids_levels_edit = QLineEdit()
        self.pyramids_levels_edit.setPlaceholderText("2,4,8,16,32,64")
        pyramids_layout.addWidget(self.pyramids_levels_edit)
        pyramids_layout.addStretch(1)
        general_layout.addWidget(pyramids_row)

        self._products_row = products_row = QWidget()
        products_layout = QHBoxLayout(products_row)
        products_layout.setContentsMargins(0, 0, 0, 0)
        products_layout.addWidget(QLabel("Produits:"))

        self._product_cbs: dict[str, QCheckBox] = {}
        for key, label, _default, _is_rvt in PRODUCTS:
            cb = QCheckBox(label)
            self._product_cbs[key] = cb
            products_layout.addWidget(cb)
        products_layout.addStretch(1)
        general_layout.addWidget(products_row)

        # Aliases for backward compatibility
        self.product_mnt_cb = self._product_cbs["MNT"]
        self.product_densite_cb = self._product_cbs["DENSITE"]
        self.product_mhs_cb = self._product_cbs["M_HS"]
        self.product_svf_cb = self._product_cbs["SVF"]
        self.product_slo_cb = self._product_cbs["SLO"]
        self.product_ld_cb = self._product_cbs["LD"]
        self.product_slrm_cb = self._product_cbs["SLRM"]
        self.product_vat_cb = self._product_cbs["VAT"]

        self.reset_general_btn = QPushButton("Remettre par défaut")
        self.reset_general_btn.clicked.connect(self._reset_general_config)
        general_layout.addWidget(self.reset_general_btn)

        general_group_layout.addWidget(general_content)
        config_layout.addWidget(general_group)

        self._rvt_group = rvt_group = QGroupBox("Paramètres RVT")
        rvt_layout = QVBoxLayout(rvt_group)

        self.rvt_tabs = QTabWidget()

        mdh_tab = QWidget()
        mdh_form = QFormLayout(mdh_tab)
        self.mdh_num_directions_spin = NoWheelSpinBox()
        self.mdh_num_directions_spin.setRange(1, 360)
        self.mdh_num_directions_spin.setToolTip("Nombre de directions d'éclairage simulées (défaut : 16)")
        self.mdh_sun_elevation_spin = NoWheelSpinBox()
        self.mdh_sun_elevation_spin.setRange(0, 90)
        self.mdh_sun_elevation_spin.setToolTip("Angle d'élévation du soleil en degrés (défaut : 35°)")
        self.mdh_ve_factor_spin = NoWheelSpinBox()
        self.mdh_ve_factor_spin.setRange(1, 100)
        self.mdh_ve_factor_spin.setToolTip("Facteur d'exagération verticale (défaut : 1)")
        self.mdh_save_8bit_cb = QCheckBox("Sauver en 8bit")
        mdh_form.addRow("Nombre directions :", self.mdh_num_directions_spin)
        mdh_form.addRow("Élévation solaire (°) :", self.mdh_sun_elevation_spin)
        mdh_form.addRow("Facteur VE :", self.mdh_ve_factor_spin)
        mdh_form.addRow("", self.mdh_save_8bit_cb)
        self.rvt_tabs.addTab(mdh_tab, "M-HS")

        svf_tab = QWidget()
        svf_form = QFormLayout(svf_tab)
        self.svf_noise_remove_spin = NoWheelSpinBox()
        self.svf_noise_remove_spin.setRange(0, 9999)
        self.svf_noise_remove_spin.setToolTip("Suppression du bruit : 0 = désactivé (défaut : 0)")
        self.svf_num_directions_spin = NoWheelSpinBox()
        self.svf_num_directions_spin.setRange(1, 360)
        self.svf_num_directions_spin.setToolTip("Nombre de directions d'analyse (défaut : 16)")
        self.svf_radius_spin = NoWheelSpinBox()
        self.svf_radius_spin.setRange(0, 100000)
        self.svf_radius_spin.setToolTip("Rayon de recherche en pixels (défaut : 10)")
        self.svf_ve_factor_spin = NoWheelSpinBox()
        self.svf_ve_factor_spin.setRange(1, 100)
        self.svf_ve_factor_spin.setToolTip("Facteur d'exagération verticale (défaut : 1)")
        self.svf_save_8bit_cb = QCheckBox("Sauver en 8bit")
        svf_form.addRow("Suppression bruit :", self.svf_noise_remove_spin)
        svf_form.addRow("Nombre directions :", self.svf_num_directions_spin)
        svf_form.addRow("Rayon (px) :", self.svf_radius_spin)
        svf_form.addRow("Facteur VE :", self.svf_ve_factor_spin)
        svf_form.addRow("", self.svf_save_8bit_cb)
        self.rvt_tabs.addTab(svf_tab, "SVF")

        slope_tab = QWidget()
        slope_form = QFormLayout(slope_tab)
        self.slope_unit_combo = NoWheelComboBox()
        self.slope_unit_combo.addItem("Degrés", 0)
        self.slope_unit_combo.addItem("Pourcentage", 1)
        self.slope_ve_factor_spin = NoWheelSpinBox()
        self.slope_ve_factor_spin.setRange(1, 100)
        self.slope_ve_factor_spin.setToolTip("Facteur d'exagération verticale (défaut : 1)")
        self.slope_save_8bit_cb = QCheckBox("Sauver en 8bit")
        slope_form.addRow("Unité :", self.slope_unit_combo)
        slope_form.addRow("Facteur VE :", self.slope_ve_factor_spin)
        slope_form.addRow("", self.slope_save_8bit_cb)
        self.rvt_tabs.addTab(slope_tab, "Slope")

        ld_tab = QWidget()
        ld_form = QFormLayout(ld_tab)
        self.ldo_angular_res_spin = NoWheelSpinBox()
        self.ldo_angular_res_spin.setRange(1, 360)
        self.ldo_angular_res_spin.setToolTip("Pas angulaire en degrés (défaut : 15°)")
        self.ldo_min_radius_spin = NoWheelSpinBox()
        self.ldo_min_radius_spin.setRange(0, 100000)
        self.ldo_min_radius_spin.setToolTip("Rayon minimal de recherche en pixels (défaut : 10)")
        self.ldo_max_radius_spin = NoWheelSpinBox()
        self.ldo_max_radius_spin.setRange(0, 100000)
        self.ldo_max_radius_spin.setToolTip("Rayon maximal de recherche en pixels (défaut : 20)")
        self.ldo_observer_h_spin = NoWheelDoubleSpinBox()
        self.ldo_observer_h_spin.setDecimals(2)
        self.ldo_observer_h_spin.setRange(0.0, 10000.0)
        self.ldo_observer_h_spin.setSingleStep(0.1)
        self.ldo_observer_h_spin.setToolTip("Hauteur de l'observateur en mètres (défaut : 1.7 m)")
        self.ldo_ve_factor_spin = NoWheelSpinBox()
        self.ldo_ve_factor_spin.setRange(1, 100)
        self.ldo_ve_factor_spin.setToolTip("Facteur d'exagération verticale (défaut : 1)")
        self.ldo_save_8bit_cb = QCheckBox("Sauver en 8bit")
        ld_form.addRow("Résolution angulaire (°) :", self.ldo_angular_res_spin)
        ld_form.addRow("Rayon min (px) :", self.ldo_min_radius_spin)
        ld_form.addRow("Rayon max (px) :", self.ldo_max_radius_spin)
        ld_form.addRow("Hauteur observateur (m) :", self.ldo_observer_h_spin)
        ld_form.addRow("Facteur VE :", self.ldo_ve_factor_spin)
        ld_form.addRow("", self.ldo_save_8bit_cb)
        self.rvt_tabs.addTab(ld_tab, "LD")

        slrm_tab = QWidget()
        slrm_form = QFormLayout(slrm_tab)
        self.slrm_radius_spin = NoWheelSpinBox()
        self.slrm_radius_spin.setRange(1, 100000)
        self.slrm_radius_spin.setToolTip("Rayon du filtre de lissage en pixels (défaut : 20)")
        self.slrm_ve_factor_spin = NoWheelSpinBox()
        self.slrm_ve_factor_spin.setRange(1, 100)
        self.slrm_ve_factor_spin.setToolTip("Facteur d'exagération verticale (défaut : 1)")
        self.slrm_save_8bit_cb = QCheckBox("Sauver en 8bit")
        slrm_form.addRow("Rayon (px) :", self.slrm_radius_spin)
        slrm_form.addRow("Facteur VE :", self.slrm_ve_factor_spin)
        slrm_form.addRow("", self.slrm_save_8bit_cb)
        self.rvt_tabs.addTab(slrm_tab, "SLRM")

        vat_tab = QWidget()
        vat_form = QFormLayout(vat_tab)
        self.vat_terrain_type_combo = NoWheelComboBox()
        self.vat_terrain_type_combo.addItem("Général", 0)
        self.vat_terrain_type_combo.addItem("Plat", 1)
        self.vat_terrain_type_combo.addItem("Pentu", 2)
        self.vat_save_8bit_cb = QCheckBox("Sauver en 8bit")
        self.vat_terrain_type_combo.setToolTip("Type de terrain pour le calcul VAT (défaut : Général)")
        vat_form.addRow("Type de terrain :", self.vat_terrain_type_combo)
        vat_form.addRow("", self.vat_save_8bit_cb)
        self.rvt_tabs.addTab(vat_tab, "VAT")

        rvt_layout.addWidget(self.rvt_tabs)

        self.reset_rvt_btn = QPushButton("Remettre par défaut")
        self.reset_rvt_btn.clicked.connect(self._reset_rvt_config)
        rvt_layout.addWidget(self.reset_rvt_btn)

        config_layout.addWidget(rvt_group)

        cv_group = QGroupBox("Détection par Computer Vision")
        cv_group_layout = QVBoxLayout(cv_group)
        cv_content = QWidget()
        cv_layout = QVBoxLayout(cv_content)
        cv_layout.setContentsMargins(0, 0, 0, 0)

        self.cv_enabled_cb = QCheckBox("Activer la détection par computer vision")
        cv_layout.addWidget(self.cv_enabled_cb)

        cv_form = QFormLayout()

        # Table multi-runs (modèle + RVT cible par ligne)
        runs_group = QGroupBox("Modèles à exécuter")
        runs_vlayout = QVBoxLayout(runs_group)
        runs_vlayout.setContentsMargins(4, 4, 4, 4)

        self.cv_runs_table = QTableWidget(0, 4)
        self.cv_runs_table.setHorizontalHeaderLabels(["Modèle", "RVT cible", "Aire min (m²)", ""])
        self.cv_runs_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.cv_runs_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.cv_runs_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.cv_runs_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        self.cv_runs_table.setColumnWidth(3, 60)
        self.cv_runs_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.cv_runs_table.setMaximumHeight(300)
        self.cv_runs_table.setMinimumHeight(120)
        self.cv_runs_table.verticalHeader().setVisible(False)
        self.cv_runs_table.verticalHeader().setDefaultSectionSize(36)
        runs_vlayout.addWidget(self.cv_runs_table)

        runs_btn_row = QWidget()
        runs_btn_layout = QHBoxLayout(runs_btn_row)
        runs_btn_layout.setContentsMargins(0, 0, 0, 0)
        self.cv_add_run_btn = QPushButton("+ Ajouter")
        self.cv_add_run_btn.clicked.connect(self._add_cv_run_row)
        self.cv_refresh_models_btn = QPushButton("Actualiser modèles")
        self.cv_refresh_models_btn.clicked.connect(self._refresh_models)
        runs_btn_layout.addWidget(self.cv_add_run_btn)
        runs_btn_layout.addStretch(1)
        runs_btn_layout.addWidget(self.cv_refresh_models_btn)
        runs_vlayout.addWidget(runs_btn_row)

        cv_form.addRow(runs_group)

        # Compat: attributs internes pour le code qui référence encore les anciens combos
        self.cv_model_combo = None
        self.cv_target_rvt_combo = None

        classes_group = QGroupBox("Classes à détecter")
        classes_layout = QVBoxLayout(classes_group)
        self.cv_classes_list = QListWidget()
        self.cv_classes_list.setMaximumHeight(200)
        classes_layout.addWidget(self.cv_classes_list)
        classes_btn_row = QWidget()
        classes_btn_layout = QHBoxLayout(classes_btn_row)
        classes_btn_layout.setContentsMargins(0, 0, 0, 0)
        self.cv_classes_select_all_btn = QPushButton("Tout sélectionner")
        self.cv_classes_select_all_btn.clicked.connect(self._select_all_classes)
        self.cv_classes_deselect_all_btn = QPushButton("Tout désélectionner")
        self.cv_classes_deselect_all_btn.clicked.connect(self._deselect_all_classes)
        classes_btn_layout.addWidget(self.cv_classes_select_all_btn)
        classes_btn_layout.addWidget(self.cv_classes_deselect_all_btn)
        classes_btn_layout.addStretch(1)
        classes_layout.addWidget(classes_btn_row)
        cv_form.addRow(classes_group)

        thresholds_row = QWidget()
        thresholds_layout = QHBoxLayout(thresholds_row)
        thresholds_layout.setContentsMargins(0, 0, 0, 0)
        thresholds_layout.addWidget(QLabel("Confiance:"))
        self.cv_confidence_spin = NoWheelDoubleSpinBox()
        self.cv_confidence_spin.setDecimals(2)
        self.cv_confidence_spin.setRange(0.0, 1.0)
        self.cv_confidence_spin.setSingleStep(0.05)
        thresholds_layout.addWidget(self.cv_confidence_spin)
        thresholds_layout.addSpacing(12)
        thresholds_layout.addWidget(QLabel("IoU:"))
        self.cv_iou_spin = NoWheelDoubleSpinBox()
        self.cv_iou_spin.setDecimals(2)
        self.cv_iou_spin.setRange(0.0, 1.0)
        self.cv_iou_spin.setSingleStep(0.05)
        thresholds_layout.addWidget(self.cv_iou_spin)
        thresholds_layout.addStretch(1)
        cv_form.addRow("Seuils:", thresholds_row)

        cv_layout.addLayout(cv_form)

        output_group = QGroupBox("Options de sortie")
        output_layout = QVBoxLayout(output_group)
        self.cv_generate_annotated_cb = QCheckBox("Générer des images avec bounding boxes/polygones")
        self.cv_generate_shp_cb = QCheckBox("Générer des shapefiles")
        output_layout.addWidget(self.cv_generate_annotated_cb)
        output_layout.addWidget(self.cv_generate_shp_cb)
        cv_layout.addWidget(output_group)


        self.reset_cv_btn = QPushButton("Remettre par défaut")
        self.reset_cv_btn.clicked.connect(self._reset_cv_config)
        cv_layout.addWidget(self.reset_cv_btn)

        cv_group_layout.addWidget(cv_content)
        config_layout.addWidget(cv_group)
        config_layout.addStretch(1)

        logs_group = QGroupBox("Logs")
        logs_layout = QVBoxLayout(logs_group)
        self.logs_text = QPlainTextEdit()
        self.logs_text.setReadOnly(True)
        logs_layout.addWidget(self.logs_text)
        layout.addWidget(logs_group, 1)

        buttons_row = QWidget()
        buttons_layout = QHBoxLayout(buttons_row)
        buttons_layout.setContentsMargins(0, 0, 0, 0)

        self.run_btn = QPushButton("Lancer le pipeline")
        self.run_btn.clicked.connect(self._on_run_clicked)
        buttons_layout.addWidget(self.run_btn)

        self.cancel_btn = QPushButton("Annuler")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        buttons_layout.addWidget(self.cancel_btn)

        self.save_prefs_btn = QPushButton("Sauvegarder préférences")
        self.save_prefs_btn.clicked.connect(self._on_save_prefs_clicked)
        buttons_layout.addWidget(self.save_prefs_btn)

        self.clear_logs_btn = QPushButton("Nettoyer logs")
        self.clear_logs_btn.clicked.connect(self._clear_logs)
        buttons_layout.addWidget(self.clear_logs_btn)

        buttons_layout.addSpacing(12)

        self.stage_label = QLabel("")
        self.stage_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        buttons_layout.addWidget(self.stage_label, 0)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        buttons_layout.addWidget(self.progress_bar, 1)

        layout.addWidget(buttons_row, 0)

        self._clear_logs()
        self._load_into_widgets()
        self._wire_autosave()
        self._apply_data_mode_state()

        self._refresh_models()
        self._update_available_rvt_targets()
        self._apply_cv_enabled_state()

        self._refresh_path_validations()

        self._logger.info("Pipeline prêt à être utilisé")

    def closeEvent(self, event):
        self._save_specific_source_only()
        self._save_from_widgets()
        super().closeEvent(event)

    def _append_log(self, msg: str) -> None:
        self.logs_text.appendPlainText(msg)
        sb = self.logs_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _clear_logs(self) -> None:
        try:
            self.logs_text.clear()
        except Exception:
            pass

    def _set_progress(self, value: int) -> None:
        self.progress_bar.setValue(int(value))

    def _set_stage(self, text: str) -> None:
        self.stage_label.setText(text)

    def _set_run_enabled(self, enabled: bool) -> None:
        self.run_btn.setEnabled(bool(enabled))
        self.cancel_btn.setEnabled(not bool(enabled))
        self.save_prefs_btn.setEnabled(bool(enabled))
        self._config_scroll.widget().setEnabled(bool(enabled))
        if enabled:
            self._cancel_event.clear()
            self.stage_label.setText("")
            self.progress_bar.setValue(0)

    def _load_layers_to_project(self, vrt_paths: list, shapefile_paths: list, class_colors: list = None) -> None:
        """Charge les couches VRT et shapefiles dans le projet QGIS courant."""
        try:
            from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer, QgsRectangle
            from ..pipeline.cv.class_utils import BASE_COLOR_PALETTE, get_color_for_confidence

            project = QgsProject.instance()
            loaded_count = 0
            combined_extent = QgsRectangle()
            loaded_layers = []
            class_colors = class_colors or []

            # Détecter le mapping global {class_name: palette_index} (encodé comme dict dans la liste)
            global_color_map: dict = {}
            if class_colors and len(class_colors) == 1 and isinstance(class_colors[0], dict):
                global_color_map = class_colors[0]
                class_colors = []

            # Charger les VRT (couches raster)
            for vrt_path in vrt_paths:
                if not vrt_path:
                    continue
                vrt_path_str = str(vrt_path)
                # Extraire un nom lisible depuis le chemin
                # Ex: results/RVT/LD/tif/index.vrt -> "LD"
                parts = vrt_path_str.replace("\\", "/").split("/")
                layer_name = "index"
                for i, part in enumerate(parts):
                    if part == "tif" and i > 0:
                        layer_name = parts[i - 1]
                        break
                    elif part == "MNT":
                        layer_name = "MNT"
                        break

                # Vérifier si une couche avec ce nom et cette source existe déjà
                existing_layers = project.mapLayersByName(layer_name)
                already_loaded = False
                for existing in existing_layers:
                    if existing.source() == vrt_path_str:
                        already_loaded = True
                        # Utiliser l'étendue de la couche existante pour le zoom
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
                    # Combiner les étendues
                    if combined_extent.isNull():
                        combined_extent = layer.extent()
                    else:
                        combined_extent.combineExtentWith(layer.extent())
                    loaded_count += 1
                    self._logger.info(f"Couche raster chargée: {layer_name}")
                else:
                    self._logger.warning(f"Impossible de charger le VRT: {vrt_path_str}")

            # Charger les shapefiles (couches vecteur) avec style par classe
            # Utilise BASE_COLOR_PALETTE de class_utils.py (source unique de vérité)
            # L'index de couleur est extrait de l'attribut conf_color du shapefile (format: color{idx}_level)

            for shp_path in shapefile_paths:
                if not shp_path:
                    continue
                shp_path_str = str(shp_path)
                # Extraire le nom du fichier sans extension
                from pathlib import Path
                layer_name = Path(shp_path_str).stem
                
                # Extraire le nom de classe depuis le nom du fichier (ex: detections_LD_charbonnière -> charbonnière)
                # Format attendu: detections_{RVT}_{class_name}
                parts = layer_name.split("_")
                if len(parts) >= 3 and parts[0] == "detections":
                    class_name = "_".join(parts[2:])  # Rejoindre au cas où le nom contient des underscores
                else:
                    class_name = layer_name
                
                # Déterminer l'index de couleur pour cette classe
                color_idx = 0  # Fallback
                if global_color_map and class_name in global_color_map:
                    color_idx = global_color_map[class_name]
                elif global_color_map:
                    # Fallback: correspondance partielle
                    for cname, cidx in global_color_map.items():
                        if cname.lower() in layer_name.lower():
                            color_idx = cidx
                            break
                else:
                    # Ancien comportement: lire conf_color depuis le shapefile
                    try:
                        import re
                        temp_layer = QgsVectorLayer(shp_path_str, "temp", "ogr")
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

                # Vérifier si une couche avec ce nom et cette source existe déjà
                existing_layers = project.mapLayersByName(layer_name)
                already_loaded = False
                for existing in existing_layers:
                    if existing.source() == shp_path_str:
                        already_loaded = True
                        # Utiliser l'étendue de la couche existante pour le zoom
                        if combined_extent.isNull():
                            combined_extent = existing.extent()
                        else:
                            combined_extent.combineExtentWith(existing.extent())
                        break
                
                if already_loaded:
                    self._logger.info(f"Couche vecteur déjà présente: {layer_name}")
                    continue

                layer = QgsVectorLayer(shp_path_str, layer_name, "ogr")
                if layer.isValid():
                    # Appliquer le style avec la couleur correspondant à la classe (depuis BASE_COLOR_PALETTE)
                    self._apply_confidence_style_unified(layer, color_idx, get_color_for_confidence)
                    
                    project.addMapLayer(layer)
                    loaded_layers.append(layer)
                    # Combiner les étendues
                    if combined_extent.isNull():
                        combined_extent = layer.extent()
                    else:
                        combined_extent.combineExtentWith(layer.extent())
                    loaded_count += 1
                    base_color = BASE_COLOR_PALETTE[color_idx % len(BASE_COLOR_PALETTE)]
                    self._logger.info(f"Couche vecteur chargée: {layer_name} (classe={class_name}, couleur={color_idx} RGB{base_color})")
                else:
                    self._logger.warning(f"Impossible de charger le shapefile: {shp_path_str}")

            if loaded_count > 0:
                self._logger.info(f"✅ {loaded_count} couche(s) ajoutée(s) au projet QGIS")
                
                # Zoomer sur l'étendue combinée des couches chargées
                if not combined_extent.isNull():
                    try:
                        from qgis.utils import iface
                        if iface and iface.mapCanvas():
                            # Ajouter une marge de 5% autour de l'étendue
                            combined_extent.scale(1.05)
                            iface.mapCanvas().setExtent(combined_extent)
                            iface.mapCanvas().refresh()
                            self._logger.info("🔍 Zoom sur l'étendue des résultats")
                    except Exception as zoom_err:
                        self._logger.warning(f"Impossible de zoomer: {zoom_err}")

        except Exception as e:
            self._logger.error(f"Erreur lors du chargement des couches: {e}")

    def _apply_confidence_style_unified(self, layer, color_idx: int, get_color_for_confidence_fn) -> None:
        """Applique un style catégorisé par confiance avec BASE_COLOR_PALETTE.
        
        Args:
            layer: Couche QGIS
            color_idx: Index de la couleur de base (0-11) depuis BASE_COLOR_PALETTE
            get_color_for_confidence_fn: Fonction get_color_for_confidence de class_utils
        """
        try:
            from qgis.core import (
                QgsCategorizedSymbolRenderer,
                QgsRendererCategory,
                QgsFillSymbol,
            )

            categories = []
            # Confiances représentatives pour chaque intervalle: 0.1, 0.3, 0.5, 0.7, 0.9
            conf_bins_with_conf = [
                ('[0:0.2[', 0.1),
                ('[0.2:0.4[', 0.3),
                ('[0.4:0.6[', 0.5),
                ('[0.6:0.8[', 0.7),
                ('[0.8:1]', 0.9),
            ]

            for conf_bin, conf_value in conf_bins_with_conf:
                # Obtenir la couleur RGB depuis BASE_COLOR_PALETTE avec variation de luminosité
                r, g, b = get_color_for_confidence_fn(color_idx, conf_value)
                rgb_str = f'{r},{g},{b}'
                
                # Créer un symbole de contour (ligne) sans remplissage
                symbol = QgsFillSymbol.createSimple({
                    'color': '0,0,0,0',  # Transparent
                    'outline_color': f'{rgb_str},255',
                    'outline_width': '0.6',
                    'outline_style': 'solid',
                })
                
                category = QgsRendererCategory(conf_bin, symbol, conf_bin)
                categories.append(category)

            renderer = QgsCategorizedSymbolRenderer('conf_bin', categories)
            layer.setRenderer(renderer)
            layer.triggerRepaint()

        except Exception as e:
            self._logger.warning(f"Impossible d'appliquer le style: {e}")

    def _row_widget(self, edit: QLineEdit, button: QPushButton) -> QWidget:
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(edit, 1)
        l.addWidget(button, 0)
        return w

    def _wire_autosave(self) -> None:
        self.output_dir_edit.textChanged.connect(self._on_any_changed)
        self.data_mode_combo.currentIndexChanged.connect(self._on_data_mode_changed)
        self.specific_source_edit.textChanged.connect(self._on_specific_source_changed)
        # dossier modèles: embarqué dans le plugin (pas de champ UI)

        self.mnt_resolution_spin.valueChanged.connect(self._on_any_changed)
        self.density_resolution_spin.valueChanged.connect(self._on_any_changed)
        self.tile_overlap_spin.valueChanged.connect(self._on_any_changed)
        self.max_workers_spin.valueChanged.connect(self._on_any_changed)
        self.filter_expression_edit.textChanged.connect(self._on_any_changed)

        self.product_mnt_cb.toggled.connect(self._on_any_changed)
        self.product_densite_cb.toggled.connect(self._on_any_changed)
        # produits RVT : on utilise _on_rvt_products_changed pour recalculer les RVT cibles + sauvegarder

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

        self.cv_enabled_cb.toggled.connect(self._on_cv_enabled_changed)
        self.cv_runs_table.cellChanged.connect(self._on_any_changed)
        self.cv_runs_table.currentCellChanged.connect(self._on_cv_run_selection_changed)
        self.cv_confidence_spin.valueChanged.connect(self._on_any_changed)
        self.cv_iou_spin.valueChanged.connect(self._on_any_changed)
        self.cv_generate_annotated_cb.toggled.connect(self._on_any_changed)
        self.cv_generate_shp_cb.toggled.connect(self._on_any_changed)

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

    def _on_cv_enabled_changed(self) -> None:
        if self._loading:
            return
        self._apply_cv_enabled_state()
        self._save_from_widgets()

    def _apply_cv_enabled_state(self) -> None:
        enabled = self.cv_enabled_cb.isChecked()
        self.cv_runs_table.setEnabled(enabled)
        self.cv_add_run_btn.setEnabled(enabled)
        self.cv_refresh_models_btn.setEnabled(enabled)
        self.cv_confidence_spin.setEnabled(enabled)
        self.cv_iou_spin.setEnabled(enabled)
        self.cv_generate_annotated_cb.setEnabled(enabled)
        self.cv_generate_shp_cb.setEnabled(enabled)
        self.cv_classes_list.setEnabled(enabled)
        self.cv_classes_select_all_btn.setEnabled(enabled)
        self.cv_classes_deselect_all_btn.setEnabled(enabled)
        self.reset_cv_btn.setEnabled(enabled)

    def _get_available_models(self) -> list:
        """Retourne la liste des modèles disponibles: [(label, path), ...]"""
        p = self._plugin_root / "models"
        items: list[tuple[str, str]] = []
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
        checked = [
            (key, label) for key, label, _d, is_rvt in PRODUCTS
            if is_rvt and self._product_cbs.get(key, None) is not None and self._product_cbs[key].isChecked()
        ]
        if checked:
            return checked
        # Fallback: tous les RVT (mode existing_rvt par ex.)
        return [(key, label) for key, label, _d, is_rvt in PRODUCTS if is_rvt]

    def _add_cv_run_row(self, model_name: str = "", target_rvt: str = "LD", min_area_m2: float = 0.0) -> None:
        """Ajoute une ligne au tableau des runs CV avec des combos modèle, RVT et filtre aire."""
        self._loading = True
        try:
            row = self.cv_runs_table.rowCount()
            self.cv_runs_table.insertRow(row)

            # Combo modèle
            model_combo = NoWheelComboBox()
            available_models = self._get_available_models()
            for label, path in available_models:
                model_combo.addItem(label, path)
            # Sélectionner le modèle demandé
            if model_name:
                for i in range(model_combo.count()):
                    data = str(model_combo.itemData(i) or "")
                    text = model_combo.itemText(i)
                    if model_name in (data, text) or text == model_name:
                        model_combo.setCurrentIndex(i)
                        break
            model_combo.currentIndexChanged.connect(self._on_any_changed)
            model_combo.currentIndexChanged.connect(self._refresh_model_classes)
            self.cv_runs_table.setCellWidget(row, 0, model_combo)

            # Combo RVT
            rvt_combo = NoWheelComboBox()
            rvt_keys = self._get_available_rvt_keys()
            for key, label in rvt_keys:
                rvt_combo.addItem(label, key)
            # Sélectionner le RVT demandé
            if target_rvt:
                idx = rvt_combo.findData(target_rvt)
                if idx >= 0:
                    rvt_combo.setCurrentIndex(idx)
            rvt_combo.currentIndexChanged.connect(self._on_any_changed)
            self.cv_runs_table.setCellWidget(row, 1, rvt_combo)

            # Spinbox aire min (m²) — 0 = pas de filtrage
            area_spin = QDoubleSpinBox()
            area_spin.setDecimals(0)
            area_spin.setRange(0.0, 100000.0)
            area_spin.setSingleStep(50.0)
            area_spin.setValue(min_area_m2)
            area_spin.setSuffix(" m²")
            area_spin.setToolTip("Aire minimale en m² (0 = pas de filtrage). Les détections plus petites seront supprimées.")
            area_spin.setMinimumWidth(90)
            area_spin.valueChanged.connect(self._on_any_changed)
            self.cv_runs_table.setCellWidget(row, 2, area_spin)

            # Boutons actions (ℹ info + × supprimer)
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
            self.cv_runs_table.setCellWidget(row, 3, actions_widget)
        finally:
            self._loading = False

    def _find_row_for_sender(self) -> int:
        """Trouve la ligne du tableau correspondant au bouton cliqué (via parentWidget)."""
        btn = self.sender()
        if btn is None:
            return -1
        # Le bouton est dans un QWidget (actions_widget) qui est le cellWidget de la colonne 3
        parent = btn.parentWidget()
        for row in range(self.cv_runs_table.rowCount()):
            if self.cv_runs_table.cellWidget(row, 3) is parent:
                return row
        return -1

    def _on_row_delete_clicked(self) -> None:
        """Supprime la ligne du bouton × cliqué."""
        row = self._find_row_for_sender()
        if row < 0:
            return
        self.cv_runs_table.blockSignals(True)
        try:
            self.cv_runs_table.removeRow(row)
            new_count = self.cv_runs_table.rowCount()
            if new_count > 0:
                self.cv_runs_table.setCurrentCell(min(row, new_count - 1), 0)
            else:
                self.cv_runs_table.setCurrentCell(-1, -1)
        finally:
            self.cv_runs_table.blockSignals(False)
        self._refresh_model_classes()
        if not self._loading:
            self._save_from_widgets()

    def _on_row_info_clicked(self) -> None:
        """Affiche les infos du modèle de la ligne du bouton ℹ cliqué."""
        row = self._find_row_for_sender()
        if row < 0:
            return
        self._show_model_training_info_for_row(row)

    def _show_model_training_info_for_row(self, row: int) -> None:
        """Affiche les paramètres d'entraînement du modèle à la ligne donnée."""
        import json

        combo = self.cv_runs_table.cellWidget(row, 0)
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

        # Charger training_params.json
        params = {}
        if training_params_file.exists():
            try:
                with training_params_file.open("r", encoding="utf-8") as f:
                    params = json.load(f)
            except Exception as e:
                self._logger.warning(f"Erreur lecture training_params.json: {e}")

        # Charger config.json (modèles SMP)
        config_json_file = model_dir / "config.json"
        config_data = {}
        if config_json_file.exists():
            try:
                with config_json_file.open("r", encoding="utf-8") as f:
                    config_data = json.load(f)
            except Exception:
                pass

        # Charger args.yaml
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
                self,
                f"Information - {model_name}",
                "Aucune information disponible pour ce modèle.\n\n"
                f"Fichiers attendus dans:\n{model_dir}"
            )
            return

        info_lines = [f"Paramètres du modèle: {model_name}\n"]

        if "description" in params:
            info_lines.append(f"{params['description']}\n")
        if "creation_date" in params:
            info_lines.append(f"📅 Date de création: {params['creation_date']}")

        info_lines.append("=" * 50)

        # --- Section Architecture ---
        model_info = params.get("model", {})
        cfg_model = config_data.get("model", {})
        arch = model_info.get("architecture") or cfg_model.get("architecture") or args_data.get("model", "")
        task = model_info.get("task") or cfg_model.get("task") or args_data.get("task", "")
        encoder = model_info.get("encoder") or cfg_model.get("encoder", "")
        imgsz = model_info.get("imgsz") or args_data.get("imgsz", "")
        if arch or task:
            info_lines.append("\n🧠 Architecture:")
            if arch:
                info_lines.append(f"  • Modèle: {arch}")
            if encoder:
                info_lines.append(f"  • Encoder: {encoder}")
            if task:
                info_lines.append(f"  • Tâche: {task}")
            if imgsz:
                info_lines.append(f"  • Taille d'image: {imgsz}×{imgsz}")

        # Classes
        class_names = cfg_model.get("class_names", [])
        if not class_names:
            classes_file = model_dir / "classes.txt"
            if classes_file.exists():
                try:
                    class_names = [l.strip() for l in classes_file.read_text(encoding="utf-8").splitlines() if l.strip()]
                except Exception:
                    pass
        if class_names:
            info_lines.append(f"  • Classes ({len(class_names)}): {', '.join(class_names)}")

        # --- Section Entraînement (depuis config.json) ---
        training = config_data.get("training", {})
        if training:
            info_lines.append("\n📈 Entraînement:")
            if "num_epochs" in training:
                info_lines.append(f"  • Époques: {training['num_epochs']}")
            if "batch_size" in training:
                info_lines.append(f"  • Batch size: {training['batch_size']}")
            if "learning_rate" in training:
                info_lines.append(f"  • Learning rate: {training['learning_rate']}")
            if "loss_type" in training:
                info_lines.append(f"  • Loss: {training['loss_type']}")
            if "scheduler" in training:
                info_lines.append(f"  • Scheduler: {training['scheduler']}")

        # --- Section SAHI ---
        sahi = args_data.get("sahi", {})
        if sahi:
            info_lines.append("\n🔲 SAHI (slicing inférence):")
            info_lines.append(f"  • Taille slice: {sahi.get('slice_height', 'N/A')}×{sahi.get('slice_width', 'N/A')}")
            info_lines.append(f"  • Chevauchement: {sahi.get('overlap_ratio', 'N/A')}")

        # --- Section MNT ---
        if "mnt" in params:
            mnt = params["mnt"]
            info_lines.append("\n📊 Paramètres MNT:")
            info_lines.append(f"  • Résolution: {mnt.get('resolution', 'N/A')} m")
            if "filter_expression" in mnt:
                info_lines.append(f"  • Filtre: {mnt['filter_expression']}")

        # --- Section RVT ---
        if "rvt" in params:
            rvt = params["rvt"]
            rvt_type = rvt.get("type", "N/A")
            info_lines.append(f"\n🖼️ Paramètres RVT ({rvt_type}):")
            rvt_params = rvt.get("params", {})
            if rvt_type == "LD":
                info_lines.append(f"  • Résolution angulaire: {rvt_params.get('angular_res', 'N/A')}°")
                info_lines.append(f"  • Rayon min: {rvt_params.get('min_radius', 'N/A')} px")
                info_lines.append(f"  • Rayon max: {rvt_params.get('max_radius', 'N/A')} px")
                info_lines.append(f"  • Hauteur observateur: {rvt_params.get('observer_h', 'N/A')} m")
            elif rvt_type == "SVF":
                info_lines.append(f"  • Suppression bruit: {rvt_params.get('noise_remove', 'N/A')}")
                info_lines.append(f"  • Nombre directions: {rvt_params.get('num_directions', 'N/A')}")
                info_lines.append(f"  • Rayon: {rvt_params.get('radius', 'N/A')} px")
            elif rvt_type in ("M_HS", "M-HS"):
                info_lines.append(f"  • Nombre directions: {rvt_params.get('num_directions', 'N/A')}")
                info_lines.append(f"  • Élévation solaire: {rvt_params.get('sun_elevation', 'N/A')}°")
            elif rvt_type in ("SLO", "Slope"):
                unit_val = rvt_params.get('unit', 0)
                unit_str = "Degrés" if unit_val == 0 else "Pourcentage"
                info_lines.append(f"  • Unité: {unit_str}")
            elif rvt_type == "VAT":
                terrain_val = rvt_params.get('terrain_type', 0)
                terrain_str = ["Général", "Plat", "Pentu"][terrain_val] if terrain_val in [0, 1, 2] else "N/A"
                info_lines.append(f"  • Type de terrain: {terrain_str}")
            if "ve_factor" in rvt_params:
                info_lines.append(f"  • Facteur VE: {rvt_params['ve_factor']}")
            if "save_as_8bit" in rvt_params:
                info_lines.append(f"  • Sauvegarde 8bit: {'Oui' if rvt_params['save_as_8bit'] else 'Non'}")

        # --- Section Évaluation ---
        eval_file = model_dir / "evaluation_results.json"
        if eval_file.exists():
            try:
                with eval_file.open("r", encoding="utf-8") as f:
                    eval_data = json.load(f)
                test = eval_data.get("test", eval_data.get("val", {}))
                if test:
                    info_lines.append("\n📊 Métriques d'évaluation:")
                    if "miou_fg" in test:
                        info_lines.append(f"  • mIoU (foreground): {test['miou_fg']:.4f}")
                    if "mean_dice_fg" in test:
                        info_lines.append(f"  • Dice moyen (fg): {test['mean_dice_fg']:.4f}")
                    if "pixel_acc" in test:
                        info_lines.append(f"  • Pixel accuracy: {test['pixel_acc']:.4f}")
            except Exception:
                pass

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Information - {model_name}")
        dialog.setMinimumWidth(500)
        dialog_layout = QVBoxLayout(dialog)

        info_label = QLabel("\n".join(info_lines))
        info_label.setWordWrap(True)
        info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        dialog_layout.addWidget(info_label)

        btn_layout = QHBoxLayout()
        open_folder_btn = QPushButton("📂 Ouvrir le dossier des métriques")
        open_folder_btn.clicked.connect(lambda: self._open_model_folder(model_dir))
        btn_layout.addWidget(open_folder_btn)
        btn_layout.addStretch(1)
        close_btn = QPushButton("Fermer")
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(close_btn)
        dialog_layout.addLayout(btn_layout)

        dialog.exec()

    def _on_cv_run_selection_changed(self) -> None:
        """Met à jour les classes affichées quand on change de ligne dans le tableau."""
        if self._loading:
            return
        self._refresh_model_classes()

    def _refresh_models(self) -> None:
        """Actualise les combos modèle dans toutes les lignes du tableau."""
        available_models = self._get_available_models()
        self._loading = True
        try:
            for row in range(self.cv_runs_table.rowCount()):
                combo = self.cv_runs_table.cellWidget(row, 0)
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

    def _get_selected_run_model_path(self) -> str:
        """Retourne le chemin du modèle de la ligne sélectionnée dans le tableau."""
        row = self.cv_runs_table.currentRow()
        if row < 0:
            return ""
        combo = self.cv_runs_table.cellWidget(row, 0)
        if isinstance(combo, QComboBox):
            return str(combo.currentData() or "")
        return ""

    @staticmethod
    def _load_classes_for_model_path(model_path: str) -> list:
        """Charge les noms de classes depuis le dossier d'un modèle. Retourne une liste de str."""
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
                import json
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

        return list(dict.fromkeys(class_names))

    def _refresh_model_classes(self) -> None:
        """Charge les classes de tous les modèles du tableau, groupées par modèle."""
        self.cv_classes_list.clear()

        for row in range(self.cv_runs_table.rowCount()):
            combo = self.cv_runs_table.cellWidget(row, 0)
            if not isinstance(combo, QComboBox):
                continue
            model_path = str(combo.currentData() or "")
            model_label = combo.currentText() or "?"
            classes = self._load_classes_for_model_path(model_path)

            # En-tête du modèle (non-cochable, gras)
            header = QListWidgetItem(f"── {model_label} ──")
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            font = header.font()
            font.setBold(True)
            header.setFont(font)
            header.setBackground(self.cv_classes_list.palette().alternateBase())
            # Stocker le nom du modèle dans le data role pour le regroupement
            header.setData(Qt.ItemDataRole.UserRole, f"__header__{model_label}")
            self.cv_classes_list.addItem(header)

            if not classes:
                info = QListWidgetItem("  (aucune classe trouvée)")
                info.setFlags(Qt.ItemFlag.NoItemFlags)
                info.setData(Qt.ItemDataRole.UserRole, f"__info__{model_label}")
                self.cv_classes_list.addItem(info)
            else:
                for class_name in classes:
                    item = QListWidgetItem(f"  {class_name}")
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Checked)
                    # Stocker modèle + classe pour _get_selected_classes
                    item.setData(Qt.ItemDataRole.UserRole, f"{model_label}\t{class_name}")
                    self.cv_classes_list.addItem(item)

    def _open_model_folder(self, folder_path: Path) -> None:
        """Ouvre le dossier du modèle dans l'explorateur de fichiers."""
        import os
        import subprocess
        import sys
        
        folder_str = str(folder_path)
        
        if sys.platform == "win32":
            os.startfile(folder_str)
        elif sys.platform == "darwin":
            subprocess.run(["open", folder_str])
        else:
            subprocess.run(["xdg-open", folder_str])

    def _select_all_classes(self) -> None:
        for i in range(self.cv_classes_list.count()):
            item = self.cv_classes_list.item(i)
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(Qt.CheckState.Checked)

    def _deselect_all_classes(self) -> None:
        for i in range(self.cv_classes_list.count()):
            item = self.cv_classes_list.item(i)
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(Qt.CheckState.Unchecked)

    def _get_selected_classes(self) -> dict:
        """Retourne un dict {model_name: [class_names]} des classes cochées par modèle."""
        result: dict = {}
        for i in range(self.cv_classes_list.count()):
            item = self.cv_classes_list.item(i)
            if item.checkState() != Qt.CheckState.Checked:
                continue
            data = str(item.data(Qt.ItemDataRole.UserRole) or "")
            if "\t" not in data:
                continue
            model_label, class_name = data.split("\t", 1)
            result.setdefault(model_label, []).append(class_name)
        return result

    def _get_selected_classes_flat(self) -> list:
        """Retourne la liste plate de toutes les classes cochées (compat)."""
        selected = []
        for classes in self._get_selected_classes().values():
            selected.extend(classes)
        return list(dict.fromkeys(selected))

    def _get_model_name_from_path(self, model_path: str) -> str:
        """Extrait le nom du modèle depuis son chemin."""
        if not model_path:
            return ""
        model_file = Path(model_path)
        if model_file.name == "best.pt" or model_file.name == "best.onnx":
            # Structure: models/model_name/weights/best.pt
            if model_file.parent.name == "weights":
                return model_file.parent.parent.name
        return model_file.stem

    def _update_available_rvt_targets(self) -> None:
        """Met à jour les combos RVT dans toutes les lignes du tableau des runs (produits cochés uniquement)."""
        rvt_keys = self._get_available_rvt_keys()

        self._loading = True
        try:
            for row in range(self.cv_runs_table.rowCount()):
                combo = self.cv_runs_table.cellWidget(row, 1)
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

    def _on_data_mode_changed(self) -> None:
        if self._loading:
            return

        # Important: à ce moment-là, currentData() pointe déjà sur le *nouveau* mode.
        # On doit d'abord sauvegarder la valeur courante dans la clé du *mode précédent*.
        prev_mode = self._current_mode or "ign_laz"
        prev_key, _, _ = self._mode_mapping(str(prev_mode))
        app = self._config.setdefault("app", {})
        files = app.setdefault("files", {})
        files[prev_key] = self.specific_source_edit.text().strip()
        self._config_manager.save(self._config)

        self._apply_data_mode_state()
        self._refresh_path_validations()
        self._save_from_widgets()

    def _apply_data_mode_state(self) -> None:
        mode = self.data_mode_combo.currentData()
        if mode is None:
            mode = "ign_laz"
        self._current_mode = mode

        key, label_text, _ = self._mode_mapping(mode)
        if self.specific_source_label is not None:
            self.specific_source_label.setText(label_text)

        files = (self._config.get("app") or {}).get("files") or {}
        self._loading = True
        try:
            self.specific_source_edit.setText(files.get(key) or "")
        finally:
            self._loading = False

        self._apply_mode_visibility(mode)
        self._refresh_path_validations()

    def _apply_mode_visibility(self, mode: str) -> None:
        """Masque ou affiche les sections selon le mode sélectionné."""
        needs_laz = mode in ("ign_laz", "local_laz")
        needs_products = mode != "existing_rvt"
        needs_rvt_params = mode != "existing_rvt"

        # Résolutions MNT/densité, filtre PDAL, workers : uniquement modes LAZ
        self._resolutions_row.setVisible(needs_laz)
        self._filter_widget.setVisible(needs_laz)

        # Pyramides, produits et bouton reset : pas en existing_rvt
        self._pyramids_row.setVisible(needs_products)
        self._products_row.setVisible(needs_products)
        self.reset_general_btn.setVisible(needs_products)

        # Paramètres RVT : pas en existing_rvt
        self._rvt_group.setVisible(needs_rvt_params)

    def _refresh_path_validations(self) -> None:
        # Le dossier de sortie sera créé s'il n'existe pas, donc pas de validation rouge
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
        
        # Si allow_create=True et c'est un dossier, on accepte même s'il n'existe pas
        if not ok and allow_create and expect_dir:
            # Vérifier que le chemin parent existe ou est valide
            ok = True  # Le dossier sera créé automatiquement
        
        if ok:
            edit.setStyleSheet("")
        else:
            edit.setStyleSheet("QLineEdit { background-color: #ffd6d6; }")

    def _mode_mapping(self, mode: str) -> tuple[str, str, bool]:
        if mode == "ign_laz":
            return "input_file", "Fichier dalles IGN (liste URLs):", True
        if mode == "local_laz":
            return "local_laz_dir", "Dossier nuages locaux (LAZ/LAS):", False
        if mode == "existing_mnt":
            return "existing_mnt_dir", "Dossier MNT existants (TIF/ASC):", False
        if mode == "existing_rvt":
            return "existing_rvt_dir", "Dossier indices RVT existants (TIF RVT):", False
        return "input_file", "Fichier dalles IGN (liste URLs):", True

    def _load_into_widgets(self) -> None:
        self._loading = True
        try:
            files = (self._config.get("app") or {}).get("files") or {}

            self.output_dir_edit.setText(files.get("output_dir") or "")

            mode = files.get("data_mode") or "ign_laz"
            idx = self.data_mode_combo.findData(mode)
            self.data_mode_combo.setCurrentIndex(idx if idx >= 0 else 0)

            cv = self._config.get("computer_vision") or {}
            self.cv_enabled_cb.setChecked(bool(cv.get("enabled", False)))
            self.cv_confidence_spin.setValue(float(cv.get("confidence_threshold", 0.3)))
            self.cv_iou_spin.setValue(float(cv.get("iou_threshold", 0.5)))
            self.cv_generate_annotated_cb.setChecked(bool(cv.get("generate_annotated_images", False)))
            self.cv_generate_shp_cb.setChecked(bool(cv.get("generate_shapefiles", False)))

            # Charger les runs CV dans le tableau
            self.cv_runs_table.setRowCount(0)
            runs = cv.get("runs") or []
            if isinstance(runs, list):
                for run in runs:
                    if isinstance(run, dict):
                        model = str(run.get("model") or "")
                        rvt = str(run.get("target_rvt") or "LD")
                        min_area = float(run.get("min_area_m2", 0.0))
                        if model:
                            self._add_cv_run_row(model_name=model, target_rvt=rvt, min_area_m2=min_area)
            # Compat: ancien format mono-modèle (si runs est vide)
            if self.cv_runs_table.rowCount() == 0:
                old_model = str(cv.get("selected_model") or "")
                old_rvt = str(cv.get("target_rvt") or "LD")
                if old_model:
                    self._add_cv_run_row(model_name=old_model, target_rvt=old_rvt)

            processing = self._config.get("processing") or {}
            self.mnt_resolution_spin.setValue(float(processing.get("mnt_resolution", 0.5)))
            self.density_resolution_spin.setValue(float(processing.get("density_resolution", 1.0)))
            self.tile_overlap_spin.setValue(int(processing.get("tile_overlap", 20)))
            self.max_workers_spin.setValue(int(processing.get("max_workers", 4)))
            self.filter_expression_edit.setText(processing.get("filter_expression") or "")

            pyramids = (processing.get("pyramids") or {})
            self.pyramids_enabled_cb.setChecked(bool(pyramids.get("enabled", False)))
            levels = pyramids.get("levels", [2, 4, 8, 16, 32, 64])
            if isinstance(levels, (list, tuple)):
                try:
                    self.pyramids_levels_edit.setText(",".join([str(int(v)) for v in levels if str(v).strip()]))
                except Exception:
                    self.pyramids_levels_edit.setText("2,4,8,16,32,64")
            else:
                self.pyramids_levels_edit.setText("2,4,8,16,32,64")

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

            # La restauration des combos modèle/RVT se fait après _refresh_models/_update_available_rvt_targets
        finally:
            self._loading = False

    def _collect_config_from_widgets(self) -> None:
        """Synchronise self._config depuis l'état actuel des widgets (sans sauvegarder sur disque)."""
        app = self._config.setdefault("app", {})
        files = app.setdefault("files", {})

        files["output_dir"] = self.output_dir_edit.text().strip()
        files["data_mode"] = self.data_mode_combo.currentData()

        mode = self.data_mode_combo.currentData() or self._current_mode or "ign_laz"
        key, _, _ = self._mode_mapping(str(mode))
        files[key] = self.specific_source_edit.text().strip()

        cv = self._config.setdefault("computer_vision", {})
        cv["enabled"] = self.cv_enabled_cb.isChecked()

        # Collecter les runs depuis le tableau
        runs = []
        for row in range(self.cv_runs_table.rowCount()):
            model_combo = self.cv_runs_table.cellWidget(row, 0)
            rvt_combo = self.cv_runs_table.cellWidget(row, 1)
            area_spin = self.cv_runs_table.cellWidget(row, 2)
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
        cv["confidence_threshold"] = float(self.cv_confidence_spin.value())
        cv["iou_threshold"] = float(self.cv_iou_spin.value())
        cv["generate_annotated_images"] = self.cv_generate_annotated_cb.isChecked()
        cv["generate_shapefiles"] = self.cv_generate_shp_cb.isChecked()
        cv["models_dir"] = str(self._plugin_root / "models")

        # Nettoyage ancien format global
        cv.pop("size_filter", None)

        processing = self._config.setdefault("processing", {})
        processing["mnt_resolution"] = float(self.mnt_resolution_spin.value())
        processing["density_resolution"] = float(self.density_resolution_spin.value())
        processing["tile_overlap"] = int(self.tile_overlap_spin.value())
        processing["max_workers"] = int(self.max_workers_spin.value())
        processing["filter_expression"] = self.filter_expression_edit.text().strip()

        pyramids = processing.setdefault("pyramids", {})
        pyramids["enabled"] = self.pyramids_enabled_cb.isChecked()
        raw_levels = self.pyramids_levels_edit.text().strip()
        parsed_levels: list[int] = []
        if raw_levels:
            for part in raw_levels.replace(";", ",").split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    v = int(part)
                except Exception:
                    continue
                if v > 1:
                    parsed_levels.append(v)
        pyramids["levels"] = parsed_levels if parsed_levels else [2, 4, 8, 16, 32, 64]

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
        self._config_manager.save(self._config)

    def _sync_config_from_widgets(self) -> None:
        self._collect_config_from_widgets()

    def _save_specific_source_only(self) -> None:
        mode = self.data_mode_combo.currentData() or self._current_mode or "ign_laz"
        key, _, _ = self._mode_mapping(mode)

        app = self._config.setdefault("app", {})
        files = app.setdefault("files", {})
        files[key] = self.specific_source_edit.text().strip()

        self._config_manager.save(self._config)

    def _browse_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Sélectionner le dossier de sortie")
        if directory:
            self.output_dir_edit.setText(directory)

    def _browse_specific_source(self) -> None:
        mode = self.data_mode_combo.currentData() or "ign_laz"
        _, _, is_file = self._mode_mapping(mode)

        if is_file:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Sélectionner le fichier dalles IGN (liste URLs)",
                "",
                "Fichiers texte (*.txt *.csv);;Tous les fichiers (*.*)",
            )
            if file_path:
                self.specific_source_edit.setText(file_path)
        else:
            directory = QFileDialog.getExistingDirectory(self, "Sélectionner un dossier")
            if directory:
                self.specific_source_edit.setText(directory)

    def _reset_general_config(self) -> None:
        self._loading = True
        try:
            self.mnt_resolution_spin.setValue(0.5)
            self.density_resolution_spin.setValue(1.0)
            self.tile_overlap_spin.setValue(20)
            self.max_workers_spin.setValue(4)
            self.filter_expression_edit.setText(
                "Classification = 2 OR Classification = 6 OR Classification = 66 OR Classification = 67 OR Classification = 9"
            )

            self.pyramids_enabled_cb.setChecked(True)
            self.pyramids_levels_edit.setText("2,4,8,16,32,64")

            for pkey, _label, default, _is_rvt in PRODUCTS:
                self._product_cbs[pkey].setChecked(default)
        finally:
            self._loading = False
        self._save_from_widgets()
        self._logger.info("Configuration générale remise par défaut")

    def _reset_rvt_config(self) -> None:
        defaults = self._config_manager.default_config()
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
        finally:
            self._loading = False

        self._save_from_widgets()
        self._logger.info("Paramètres RVT remis par défaut")

    def _reset_cv_config(self) -> None:
        defaults = self._config_manager.default_config()
        cv = defaults.get("computer_vision") or {}

        self._loading = True
        try:
            self.cv_enabled_cb.setChecked(bool(cv.get("enabled", False)))
            self.cv_confidence_spin.setValue(float(cv.get("confidence_threshold", 0.3)))
            self.cv_iou_spin.setValue(float(cv.get("iou_threshold", 0.5)))
            self.cv_generate_annotated_cb.setChecked(bool(cv.get("generate_annotated_images", False)))
            self.cv_generate_shp_cb.setChecked(bool(cv.get("generate_shapefiles", False)))

            # Vider le tableau des runs
            self.cv_runs_table.setRowCount(0)
        finally:
            self._loading = False

        self._apply_cv_enabled_state()
        self._save_from_widgets()
        self._logger.info("Computer Vision remis par défaut")

    def _on_save_prefs_clicked(self) -> None:
        self._save_specific_source_only()
        self._save_from_widgets()
        self._logger.info("Préférences sauvegardées")

    def _on_run_clicked(self) -> None:
        if not self.run_btn.isEnabled():
            return

        self._sync_config_from_widgets()

        # Récupérer les classes sélectionnées depuis l'interface (pas de sauvegarde persistante)
        cv_cfg = self._config.setdefault("computer_vision", {})
        if cv_cfg.get("enabled", False):
            classes_by_model = self._get_selected_classes()  # {model_label: [classes]}
            # Construire un mapping model_path -> model_label depuis le tableau
            path_to_label: dict = {}
            for row in range(self.cv_runs_table.rowCount()):
                combo = self.cv_runs_table.cellWidget(row, 0)
                if isinstance(combo, QComboBox):
                    path_to_label[str(combo.currentData() or "")] = combo.currentText() or ""
            # Injecter selected_classes dans chaque run
            runs = cv_cfg.get("runs") or []
            for run in runs:
                model_path = str(run.get("model") or "")
                model_label = path_to_label.get(model_path, model_path)
                run["selected_classes"] = classes_by_model.get(model_label, [])
            # Compat: liste plate globale
            selected_classes = self._get_selected_classes_flat()
            cv_cfg["selected_classes"] = selected_classes
            if not selected_classes:
                QMessageBox.warning(
                    self,
                    "Avertissement",
                    "La Computer Vision est activée mais aucune classe n'est sélectionnée.\n\n"
                    "Veuillez sélectionner au moins une classe dans l'onglet Computer Vision, "
                    "ou désactiver la Computer Vision.",
                )
                return

        # Récapitulatif avant lancement
        mode_label = self.data_mode_combo.currentText()
        output = self.output_dir_edit.text().strip() or "(non défini)"
        active_products = [label for key, label, _d, _r in PRODUCTS if self._product_cbs[key].isChecked()]
        cv_status = "activée" if cv_cfg.get("enabled") else "désactivée"
        summary = (
            f"Mode : {mode_label}\n"
            f"Sortie : {output}\n"
            f"Produits : {', '.join(active_products) if active_products else '(aucun)'}\n"
            f"Computer Vision : {cv_status}\n\n"
            "Lancer le pipeline ?"
        )
        reply = QMessageBox.question(
            self, "Confirmation", summary,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

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

