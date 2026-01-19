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
    load_layers = pyqtSignal(list, list)  # (vrt_paths, shapefile_paths)


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
        self._cv_selected_model_from_config = ""
        self._cv_target_rvt_from_config = ""
        self._cancel_event = threading.Event()

        layout = QVBoxLayout(self)

        config_scroll = QScrollArea()
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

        general_group = QGroupBox("Configuration générale")
        general_group_layout = QVBoxLayout(general_group)
        general_content = QWidget()
        general_layout = QVBoxLayout(general_content)
        general_layout.setContentsMargins(0, 0, 0, 0)

        resolutions_row = QWidget()
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

        filter_form = QFormLayout()
        self.filter_expression_edit = QLineEdit()
        filter_form.addRow("Expression filtre:", self.filter_expression_edit)
        general_layout.addLayout(filter_form)

        pyramids_row = QWidget()
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

        products_row = QWidget()
        products_layout = QHBoxLayout(products_row)
        products_layout.setContentsMargins(0, 0, 0, 0)
        products_layout.addWidget(QLabel("Produits:"))

        self.product_mnt_cb = QCheckBox("MNT")
        self.product_densite_cb = QCheckBox("DENSITE")
        self.product_mhs_cb = QCheckBox("M-HS")
        self.product_svf_cb = QCheckBox("SVF")
        self.product_slo_cb = QCheckBox("SLO")
        self.product_ld_cb = QCheckBox("LD")
        self.product_vat_cb = QCheckBox("VAT")

        products_layout.addWidget(self.product_mnt_cb)
        products_layout.addWidget(self.product_densite_cb)
        products_layout.addWidget(self.product_mhs_cb)
        products_layout.addWidget(self.product_svf_cb)
        products_layout.addWidget(self.product_slo_cb)
        products_layout.addWidget(self.product_ld_cb)
        products_layout.addWidget(self.product_vat_cb)
        products_layout.addStretch(1)
        general_layout.addWidget(products_row)

        self.reset_general_btn = QPushButton("Remettre par défaut")
        self.reset_general_btn.clicked.connect(self._reset_general_config)
        general_layout.addWidget(self.reset_general_btn)

        general_group_layout.addWidget(general_content)
        config_layout.addWidget(general_group)

        rvt_group = QGroupBox("Paramètres RVT")
        rvt_layout = QVBoxLayout(rvt_group)

        self.rvt_tabs = QTabWidget()

        mdh_tab = QWidget()
        mdh_form = QFormLayout(mdh_tab)
        self.mdh_num_directions_spin = NoWheelSpinBox()
        self.mdh_num_directions_spin.setRange(1, 360)
        self.mdh_sun_elevation_spin = NoWheelSpinBox()
        self.mdh_sun_elevation_spin.setRange(0, 90)
        self.mdh_ve_factor_spin = NoWheelSpinBox()
        self.mdh_ve_factor_spin.setRange(1, 100)
        self.mdh_save_8bit_cb = QCheckBox("Sauver en 8bit")
        mdh_form.addRow("Nombre directions:", self.mdh_num_directions_spin)
        mdh_form.addRow("Élévation solaire:", self.mdh_sun_elevation_spin)
        mdh_form.addRow("Facteur VE:", self.mdh_ve_factor_spin)
        mdh_form.addRow("", self.mdh_save_8bit_cb)
        self.rvt_tabs.addTab(mdh_tab, "M-HS")

        svf_tab = QWidget()
        svf_form = QFormLayout(svf_tab)
        self.svf_noise_remove_spin = NoWheelSpinBox()
        self.svf_noise_remove_spin.setRange(0, 9999)
        self.svf_num_directions_spin = NoWheelSpinBox()
        self.svf_num_directions_spin.setRange(1, 360)
        self.svf_radius_spin = NoWheelSpinBox()
        self.svf_radius_spin.setRange(0, 100000)
        self.svf_ve_factor_spin = NoWheelSpinBox()
        self.svf_ve_factor_spin.setRange(1, 100)
        self.svf_save_8bit_cb = QCheckBox("Sauver en 8bit")
        svf_form.addRow("Suppression bruit:", self.svf_noise_remove_spin)
        svf_form.addRow("Nombre directions:", self.svf_num_directions_spin)
        svf_form.addRow("Rayon:", self.svf_radius_spin)
        svf_form.addRow("Facteur VE:", self.svf_ve_factor_spin)
        svf_form.addRow("", self.svf_save_8bit_cb)
        self.rvt_tabs.addTab(svf_tab, "SVF")

        slope_tab = QWidget()
        slope_form = QFormLayout(slope_tab)
        self.slope_unit_combo = NoWheelComboBox()
        self.slope_unit_combo.addItem("Degrés", 0)
        self.slope_unit_combo.addItem("Pourcentage", 1)
        self.slope_ve_factor_spin = NoWheelSpinBox()
        self.slope_ve_factor_spin.setRange(1, 100)
        self.slope_save_8bit_cb = QCheckBox("Sauver en 8bit")
        slope_form.addRow("Unité:", self.slope_unit_combo)
        slope_form.addRow("Facteur VE:", self.slope_ve_factor_spin)
        slope_form.addRow("", self.slope_save_8bit_cb)
        self.rvt_tabs.addTab(slope_tab, "Slope")

        ld_tab = QWidget()
        ld_form = QFormLayout(ld_tab)
        self.ldo_angular_res_spin = NoWheelSpinBox()
        self.ldo_angular_res_spin.setRange(1, 360)
        self.ldo_min_radius_spin = NoWheelSpinBox()
        self.ldo_min_radius_spin.setRange(0, 100000)
        self.ldo_max_radius_spin = NoWheelSpinBox()
        self.ldo_max_radius_spin.setRange(0, 100000)
        self.ldo_observer_h_spin = NoWheelDoubleSpinBox()
        self.ldo_observer_h_spin.setDecimals(2)
        self.ldo_observer_h_spin.setRange(0.0, 10000.0)
        self.ldo_observer_h_spin.setSingleStep(0.1)
        self.ldo_ve_factor_spin = NoWheelSpinBox()
        self.ldo_ve_factor_spin.setRange(1, 100)
        self.ldo_save_8bit_cb = QCheckBox("Sauver en 8bit")
        ld_form.addRow("Résolution angulaire:", self.ldo_angular_res_spin)
        ld_form.addRow("Rayon min:", self.ldo_min_radius_spin)
        ld_form.addRow("Rayon max:", self.ldo_max_radius_spin)
        ld_form.addRow("Hauteur observateur:", self.ldo_observer_h_spin)
        ld_form.addRow("Facteur VE:", self.ldo_ve_factor_spin)
        ld_form.addRow("", self.ldo_save_8bit_cb)
        self.rvt_tabs.addTab(ld_tab, "LD")

        vat_tab = QWidget()
        vat_form = QFormLayout(vat_tab)
        self.vat_terrain_type_combo = NoWheelComboBox()
        self.vat_terrain_type_combo.addItem("Général", 0)
        self.vat_terrain_type_combo.addItem("Plat", 1)
        self.vat_terrain_type_combo.addItem("Pentu", 2)
        self.vat_save_8bit_cb = QCheckBox("Sauver en 8bit")
        vat_form.addRow("Type de terrain:", self.vat_terrain_type_combo)
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

        model_row = QWidget()
        model_layout = QHBoxLayout(model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        self.cv_model_combo = NoWheelComboBox()
        self.cv_model_info_btn = QPushButton("ℹ")
        self.cv_model_info_btn.setFixedWidth(30)
        self.cv_model_info_btn.setToolTip("Afficher les paramètres d'entraînement du modèle")
        self.cv_model_info_btn.clicked.connect(self._show_model_training_info)
        self.cv_refresh_models_btn = QPushButton("Actualiser")
        self.cv_refresh_models_btn.clicked.connect(self._refresh_models)
        model_layout.addWidget(self.cv_model_combo, 1)
        model_layout.addWidget(self.cv_model_info_btn, 0)
        model_layout.addWidget(self.cv_refresh_models_btn, 0)
        cv_form.addRow("Modèle:", model_row)

        self.cv_target_rvt_combo = NoWheelComboBox()
        cv_form.addRow("RVT cible:", self.cv_target_rvt_combo)

        classes_group = QGroupBox("Classes à détecter")
        classes_layout = QVBoxLayout(classes_group)
        self.cv_classes_list = QListWidget()
        self.cv_classes_list.setMaximumHeight(100)
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
        self.cv_generate_annotated_cb = QCheckBox("Générer des images avec bounding boxes")
        self.cv_generate_shp_cb = QCheckBox("Générer des shapefiles")
        output_layout.addWidget(self.cv_generate_annotated_cb)
        output_layout.addWidget(self.cv_generate_shp_cb)
        cv_layout.addWidget(output_group)

        sahi_group = QGroupBox("Paramètres SAHI")
        sahi_form = QFormLayout(sahi_group)
        self.cv_slice_height_spin = NoWheelSpinBox()
        self.cv_slice_height_spin.setRange(1, 100000)
        self.cv_slice_width_spin = NoWheelSpinBox()
        self.cv_slice_width_spin.setRange(1, 100000)
        self.cv_overlap_spin = NoWheelDoubleSpinBox()
        self.cv_overlap_spin.setDecimals(2)
        self.cv_overlap_spin.setRange(0.0, 0.99)
        self.cv_overlap_spin.setSingleStep(0.05)
        sahi_form.addRow("Hauteur slice:", self.cv_slice_height_spin)
        sahi_form.addRow("Largeur slice:", self.cv_slice_width_spin)
        sahi_form.addRow("Chevauchement:", self.cv_overlap_spin)
        cv_layout.addWidget(sahi_group)

        size_filter_group = QGroupBox("Filtrage par taille des détections")
        size_filter_layout = QVBoxLayout(size_filter_group)
        size_filter_desc = QLabel(
            "Supprime du shapefile final les détections dont la plus grande dimension "
            "(largeur ou hauteur de la bounding box) dépasse le seuil défini. "
            "Utile pour éliminer les faux positifs de grande taille (ex: routes, parcelles)."
        )
        size_filter_desc.setWordWrap(True)
        size_filter_desc.setStyleSheet("color: #666; font-size: 11px; margin-bottom: 5px;")
        size_filter_layout.addWidget(size_filter_desc)
        self.cv_size_filter_enabled_cb = QCheckBox("Activer le filtrage par taille")
        size_filter_layout.addWidget(self.cv_size_filter_enabled_cb)
        size_filter_row = QWidget()
        size_filter_row_layout = QHBoxLayout(size_filter_row)
        size_filter_row_layout.setContentsMargins(0, 0, 0, 0)
        size_filter_row_layout.addWidget(QLabel("Taille max (plus grande dimension):"))
        self.cv_size_filter_max_spin = NoWheelDoubleSpinBox()
        self.cv_size_filter_max_spin.setDecimals(1)
        self.cv_size_filter_max_spin.setRange(1.0, 1000.0)
        self.cv_size_filter_max_spin.setSingleStep(5.0)
        self.cv_size_filter_max_spin.setValue(50.0)
        size_filter_row_layout.addWidget(self.cv_size_filter_max_spin)
        size_filter_row_layout.addWidget(QLabel("mètres"))
        size_filter_row_layout.addStretch(1)
        size_filter_layout.addWidget(size_filter_row)
        cv_layout.addWidget(size_filter_group)

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
        if enabled:
            self._cancel_event.clear()
            self.stage_label.setText("")
            self.progress_bar.setValue(0)

    def _load_layers_to_project(self, vrt_paths: list, shapefile_paths: list) -> None:
        """Charge les couches VRT et shapefiles dans le projet QGIS courant."""
        try:
            from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer

            project = QgsProject.instance()
            loaded_count = 0

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

                layer = QgsRasterLayer(vrt_path_str, layer_name, "gdal")
                if layer.isValid():
                    project.addMapLayer(layer)
                    loaded_count += 1
                    self._logger.info(f"Couche raster chargée: {layer_name}")
                else:
                    self._logger.warning(f"Impossible de charger le VRT: {vrt_path_str}")

            # Charger les shapefiles (couches vecteur) avec style par classe
            # Palettes identiques à conversion_shp.py
            palettes = [
                ['255,255,0', '255,204,0', '255,153,0', '255,102,0', '255,0,0'],      # Jaune -> Rouge
                ['204,229,255', '153,204,255', '102,178,255', '51,153,255', '0,102,204'],  # Bleus
                ['235,224,255', '204,179,255', '178,128,255', '153,77,255', '102,0,204'],  # Violets
                ['204,255,204', '153,255,153', '102,204,102', '51,153,51', '0,102,0'],     # Verts
                ['240,240,240', '200,200,200', '160,160,160', '120,120,120', '80,80,80'], # Gris
                ['255,235,204', '255,204,153', '230,170,115', '204,136,85', '153,85,34'], # Bruns
            ]

            for shp_idx, shp_path in enumerate(shapefile_paths):
                if not shp_path:
                    continue
                shp_path_str = str(shp_path)
                # Extraire le nom du fichier sans extension
                from pathlib import Path
                layer_name = Path(shp_path_str).stem

                layer = QgsVectorLayer(shp_path_str, layer_name, "ogr")
                if layer.isValid():
                    # Appliquer le style avec la palette correspondant à l'index de classe
                    self._apply_confidence_style(layer, palettes[shp_idx % len(palettes)])
                    
                    project.addMapLayer(layer)
                    loaded_count += 1
                    self._logger.info(f"Couche vecteur chargée: {layer_name} (palette {shp_idx % len(palettes)})")
                else:
                    self._logger.warning(f"Impossible de charger le shapefile: {shp_path_str}")

            if loaded_count > 0:
                self._logger.info(f"✅ {loaded_count} couche(s) ajoutée(s) au projet QGIS")

        except Exception as e:
            self._logger.error(f"Erreur lors du chargement des couches: {e}")

    def _apply_confidence_style(self, layer, palette: list) -> None:
        """Applique un style catégorisé par confiance avec la palette spécifiée."""
        try:
            from qgis.core import (
                QgsCategorizedSymbolRenderer,
                QgsRendererCategory,
                QgsSymbol,
                QgsFillSymbol,
                QgsSimpleLineSymbolLayer,
            )
            from qgis.PyQt.QtGui import QColor

            categories = []
            conf_bins = ['[0:0.2[', '[0.2:0.4[', '[0.4:0.6[', '[0.6:0.8[', '[0.8:1]']

            for i, (conf_bin, rgb) in enumerate(zip(conf_bins, palette)):
                # Créer un symbole de contour (ligne) sans remplissage
                symbol = QgsFillSymbol.createSimple({
                    'color': '0,0,0,0',  # Transparent
                    'outline_color': f'{rgb},255',
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

        self.vat_terrain_type_combo.currentIndexChanged.connect(self._on_any_changed)
        self.vat_save_8bit_cb.toggled.connect(self._on_any_changed)

        self.cv_enabled_cb.toggled.connect(self._on_cv_enabled_changed)
        self.cv_model_combo.currentIndexChanged.connect(self._on_any_changed)
        self.cv_model_combo.currentIndexChanged.connect(self._refresh_model_classes)
        self.cv_target_rvt_combo.currentIndexChanged.connect(self._on_any_changed)
        self.cv_confidence_spin.valueChanged.connect(self._on_any_changed)
        self.cv_iou_spin.valueChanged.connect(self._on_any_changed)
        self.cv_generate_annotated_cb.toggled.connect(self._on_any_changed)
        self.cv_generate_shp_cb.toggled.connect(self._on_any_changed)
        self.cv_slice_height_spin.valueChanged.connect(self._on_any_changed)
        self.cv_slice_width_spin.valueChanged.connect(self._on_any_changed)
        self.cv_overlap_spin.valueChanged.connect(self._on_any_changed)

        self.product_mhs_cb.toggled.connect(self._on_rvt_products_changed)
        self.product_svf_cb.toggled.connect(self._on_rvt_products_changed)
        self.product_slo_cb.toggled.connect(self._on_rvt_products_changed)
        self.product_ld_cb.toggled.connect(self._on_rvt_products_changed)
        self.product_vat_cb.toggled.connect(self._on_rvt_products_changed)

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
        self.cv_model_combo.setEnabled(enabled)
        self.cv_model_info_btn.setEnabled(enabled)
        self.cv_refresh_models_btn.setEnabled(enabled)
        self.cv_target_rvt_combo.setEnabled(enabled)
        self.cv_confidence_spin.setEnabled(enabled)
        self.cv_iou_spin.setEnabled(enabled)
        self.cv_generate_annotated_cb.setEnabled(enabled)
        self.cv_generate_shp_cb.setEnabled(enabled)
        self.cv_slice_height_spin.setEnabled(enabled)
        self.cv_slice_width_spin.setEnabled(enabled)
        self.cv_overlap_spin.setEnabled(enabled)

    def _refresh_models(self) -> None:
        p = self._plugin_root / "models"
        current = str(self.cv_model_combo.currentData() or "")

        items: list[tuple[str, str]] = []
        if p.exists() and p.is_dir():
            # Format old_src: models/<model_name>/weights/best.pt (+ args.yaml)
            for model_dir in p.iterdir():
                if not model_dir.is_dir():
                    continue
                weights_file = model_dir / "weights" / "best.pt"
                args_file = model_dir / "args.yaml"
                if weights_file.exists() and weights_file.is_file():
                    # args.yaml est présent dans old_src mais on ne le rend pas obligatoire
                    items.append((model_dir.name, str(weights_file)))

            # Fallback: si on a des poids directement dans models/ (ou autre structure)
            if not items:
                for ext in ("*.pt", "*.onnx", "*.engine"):
                    for f in p.rglob(ext):
                        if f.is_file():
                            items.append((f.name, str(f)))

        # Ordre déterministe
        items = sorted({(label, path) for label, path in items}, key=lambda t: t[0].lower())

        self._loading = True
        try:
            self.cv_model_combo.clear()
            for label, path in items:
                self.cv_model_combo.addItem(label, path)

            desired = str(current or self._cv_selected_model_from_config or "")
            if desired:
                idx = self.cv_model_combo.findData(desired)
                if idx >= 0:
                    self.cv_model_combo.setCurrentIndex(idx)
                else:
                    # Compat: si on a sauvegardé seulement le nom du modèle
                    idx = self.cv_model_combo.findText(desired)
                    if idx >= 0:
                        self.cv_model_combo.setCurrentIndex(idx)
        finally:
            self._loading = False
        
        self._refresh_model_classes()

    def _refresh_model_classes(self) -> None:
        """Charge les classes disponibles pour le modèle sélectionné."""
        self.cv_classes_list.clear()
        
        model_path = str(self.cv_model_combo.currentData() or "")
        if not model_path:
            return
        
        model_file = Path(model_path)
        if not model_file.exists():
            return
        
        model_dir = model_file.parent
        if model_dir.name == "weights":
            model_dir = model_dir.parent
        
        class_names = []
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
        
        cv_config = self._config.get("computer_vision") or {}
        selected_classes = cv_config.get("selected_classes") or []
        
        for class_name in class_names:
            item = QListWidgetItem(class_name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            if not selected_classes or class_name in selected_classes:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)
            self.cv_classes_list.addItem(item)

    def _show_model_training_info(self) -> None:
        """Affiche les paramètres d'entraînement du modèle sélectionné."""
        import json
        
        model_path = str(self.cv_model_combo.currentData() or "")
        model_name = self.cv_model_combo.currentText()
        
        if not model_path:
            QMessageBox.information(
                self,
                "Information modèle",
                "Aucun modèle sélectionné."
            )
            return
        
        model_file = Path(model_path)
        if not model_file.exists():
            QMessageBox.warning(
                self,
                "Information modèle",
                f"Le fichier du modèle n'existe pas:\n{model_path}"
            )
            return
        
        model_dir = model_file.parent
        if model_dir.name == "weights":
            model_dir = model_dir.parent
        
        training_params_file = model_dir / "training_params.json"
        
        if not training_params_file.exists():
            QMessageBox.information(
                self,
                f"Information - {model_name}",
                "Aucune information sur les paramètres d'entraînement n'est disponible pour ce modèle.\n\n"
                f"Fichier attendu:\n{training_params_file}"
            )
            return
        
        try:
            with training_params_file.open("r", encoding="utf-8") as f:
                params = json.load(f)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Erreur",
                f"Impossible de lire les paramètres d'entraînement:\n{e}"
            )
            return
        
        info_lines = [f"Paramètres d'entraînement du modèle: {model_name}\n"]
        
        if "description" in params:
            info_lines.append(f"{params['description']}\n")
        
        if "creation_date" in params:
            info_lines.append(f"📅 Date de création: {params['creation_date']}")
        
        info_lines.append("=" * 50)
        
        if "mnt" in params:
            mnt = params["mnt"]
            info_lines.append("\n📊 Paramètres MNT:")
            info_lines.append(f"  • Résolution: {mnt.get('resolution', 'N/A')} m")
            if "filter_expression" in mnt:
                info_lines.append(f"  • Filtre: {mnt['filter_expression']}")
        
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
            elif rvt_type == "M_HS" or rvt_type == "M-HS":
                info_lines.append(f"  • Nombre directions: {rvt_params.get('num_directions', 'N/A')}")
                info_lines.append(f"  • Élévation solaire: {rvt_params.get('sun_elevation', 'N/A')}°")
            elif rvt_type == "SLO" or rvt_type == "Slope":
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
            item.setCheckState(Qt.CheckState.Checked)

    def _deselect_all_classes(self) -> None:
        for i in range(self.cv_classes_list.count()):
            item = self.cv_classes_list.item(i)
            item.setCheckState(Qt.CheckState.Unchecked)

    def _get_selected_classes(self) -> list:
        selected = []
        for i in range(self.cv_classes_list.count()):
            item = self.cv_classes_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected.append(item.text())
        return selected

    def _update_available_rvt_targets(self) -> None:
        mapping = [
            ("M_HS", "M-HS"),
            ("SVF", "SVF"),
            ("SLO", "SLO"),
            ("LD", "LD"),
            ("VAT", "VAT"),
        ]

        enabled = {
            "M_HS": self.product_mhs_cb.isChecked(),
            "SVF": self.product_svf_cb.isChecked(),
            "SLO": self.product_slo_cb.isChecked(),
            "LD": self.product_ld_cb.isChecked(),
            "VAT": self.product_vat_cb.isChecked(),
        }

        current = self.cv_target_rvt_combo.currentData() or ""

        self._loading = True
        try:
            self.cv_target_rvt_combo.clear()
            for key, label in mapping:
                if enabled.get(key):
                    self.cv_target_rvt_combo.addItem(label, key)
            if self.cv_target_rvt_combo.count() == 0:
                for key, label in mapping:
                    self.cv_target_rvt_combo.addItem(label, key)

            desired = current or self._cv_target_rvt_from_config
            if desired:
                idx = self.cv_target_rvt_combo.findData(desired)
                if idx >= 0:
                    self.cv_target_rvt_combo.setCurrentIndex(idx)
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

        self._refresh_path_validations()

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

            self._cv_selected_model_from_config = cv.get("selected_model") or ""
            self._cv_target_rvt_from_config = cv.get("target_rvt") or ""

            sahi = cv.get("sahi") or {}
            self.cv_slice_height_spin.setValue(int(sahi.get("slice_height", 640)))
            self.cv_slice_width_spin.setValue(int(sahi.get("slice_width", 640)))
            self.cv_overlap_spin.setValue(float(sahi.get("overlap_ratio", 0.2)))

            size_filter = cv.get("size_filter") or {}
            self.cv_size_filter_enabled_cb.setChecked(bool(size_filter.get("enabled", False)))
            self.cv_size_filter_max_spin.setValue(float(size_filter.get("max_meters", 50.0)))

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
            self.product_mnt_cb.setChecked(bool(products.get("MNT", True)))
            self.product_densite_cb.setChecked(bool(products.get("DENSITE", True)))
            self.product_mhs_cb.setChecked(bool(products.get("M_HS", True)))
            self.product_svf_cb.setChecked(bool(products.get("SVF", True)))
            self.product_slo_cb.setChecked(bool(products.get("SLO", True)))
            self.product_ld_cb.setChecked(bool(products.get("LD", True)))
            self.product_vat_cb.setChecked(bool(products.get("VAT", False)))

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

            vat = rvt.get("vat") or {}
            terrain = int(vat.get("terrain_type", 0))
            idx_terrain = self.vat_terrain_type_combo.findData(terrain)
            self.vat_terrain_type_combo.setCurrentIndex(idx_terrain if idx_terrain >= 0 else 0)
            self.vat_save_8bit_cb.setChecked(bool(vat.get("save_as_8bit", True)))

            # La restauration des combos modèle/RVT se fait après _refresh_models/_update_available_rvt_targets
        finally:
            self._loading = False

    def _save_from_widgets(self) -> None:
        app = self._config.setdefault("app", {})
        files = app.setdefault("files", {})

        files["output_dir"] = self.output_dir_edit.text().strip()
        files["data_mode"] = self.data_mode_combo.currentData()

        mode = self.data_mode_combo.currentData() or self._current_mode or "ign_laz"
        key, _, _ = self._mode_mapping(str(mode))
        files[key] = self.specific_source_edit.text().strip()

        cv = self._config.setdefault("computer_vision", {})
        cv["enabled"] = self.cv_enabled_cb.isChecked()
        cv["selected_model"] = str(self.cv_model_combo.currentData() or "")
        cv["target_rvt"] = str(self.cv_target_rvt_combo.currentData() or "LD")
        cv["confidence_threshold"] = float(self.cv_confidence_spin.value())
        cv["iou_threshold"] = float(self.cv_iou_spin.value())
        cv["generate_annotated_images"] = self.cv_generate_annotated_cb.isChecked()
        cv["generate_shapefiles"] = self.cv_generate_shp_cb.isChecked()
        cv["models_dir"] = str(self._plugin_root / "models")

        sahi = cv.setdefault("sahi", {})
        sahi["slice_height"] = int(self.cv_slice_height_spin.value())
        sahi["slice_width"] = int(self.cv_slice_width_spin.value())
        sahi["overlap_ratio"] = float(self.cv_overlap_spin.value())

        size_filter = cv.setdefault("size_filter", {})
        size_filter["enabled"] = self.cv_size_filter_enabled_cb.isChecked()
        size_filter["max_meters"] = float(self.cv_size_filter_max_spin.value())

        cv["selected_classes"] = self._get_selected_classes()

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
        products["MNT"] = self.product_mnt_cb.isChecked()
        products["DENSITE"] = self.product_densite_cb.isChecked()
        products["M_HS"] = self.product_mhs_cb.isChecked()
        products["SVF"] = self.product_svf_cb.isChecked()
        products["SLO"] = self.product_slo_cb.isChecked()
        products["LD"] = self.product_ld_cb.isChecked()
        products["VAT"] = self.product_vat_cb.isChecked()

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

        vat = rvt.setdefault("vat", {})
        vat["terrain_type"] = int(self.vat_terrain_type_combo.currentData())
        vat["save_as_8bit"] = self.vat_save_8bit_cb.isChecked()

        self._config_manager.save(self._config)

    def _sync_config_from_widgets(self) -> None:
        app = self._config.setdefault("app", {})
        files = app.setdefault("files", {})

        files["output_dir"] = self.output_dir_edit.text().strip()
        files["data_mode"] = self.data_mode_combo.currentData()

        mode = self.data_mode_combo.currentData() or self._current_mode or "ign_laz"
        key, _, _ = self._mode_mapping(str(mode))
        files[key] = self.specific_source_edit.text().strip()

        cv = self._config.setdefault("computer_vision", {})
        cv["enabled"] = self.cv_enabled_cb.isChecked()
        cv["selected_model"] = str(self.cv_model_combo.currentData() or "")
        cv["target_rvt"] = str(self.cv_target_rvt_combo.currentData() or "LD")
        cv["confidence_threshold"] = float(self.cv_confidence_spin.value())
        cv["iou_threshold"] = float(self.cv_iou_spin.value())
        cv["generate_annotated_images"] = self.cv_generate_annotated_cb.isChecked()
        cv["generate_shapefiles"] = self.cv_generate_shp_cb.isChecked()
        cv["models_dir"] = str(self._plugin_root / "models")

        sahi = cv.setdefault("sahi", {})
        sahi["slice_height"] = int(self.cv_slice_height_spin.value())
        sahi["slice_width"] = int(self.cv_slice_width_spin.value())
        sahi["overlap_ratio"] = float(self.cv_overlap_spin.value())

        size_filter = cv.setdefault("size_filter", {})
        size_filter["enabled"] = self.cv_size_filter_enabled_cb.isChecked()
        size_filter["max_meters"] = float(self.cv_size_filter_max_spin.value())

        cv["selected_classes"] = self._get_selected_classes()

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
        products["MNT"] = self.product_mnt_cb.isChecked()
        products["DENSITE"] = self.product_densite_cb.isChecked()
        products["M_HS"] = self.product_mhs_cb.isChecked()
        products["SVF"] = self.product_svf_cb.isChecked()
        products["SLO"] = self.product_slo_cb.isChecked()
        products["LD"] = self.product_ld_cb.isChecked()
        products["VAT"] = self.product_vat_cb.isChecked()

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

        vat = rvt.setdefault("vat", {})
        vat["terrain_type"] = int(self.vat_terrain_type_combo.currentData())
        vat["save_as_8bit"] = self.vat_save_8bit_cb.isChecked()

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

            self.product_mnt_cb.setChecked(True)
            self.product_densite_cb.setChecked(False)
            self.product_mhs_cb.setChecked(False)
            self.product_svf_cb.setChecked(False)
            self.product_slo_cb.setChecked(False)
            self.product_ld_cb.setChecked(False)
            self.product_vat_cb.setChecked(False)
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

            self._cv_selected_model_from_config = cv.get("selected_model") or ""
            self._cv_target_rvt_from_config = cv.get("target_rvt") or "LD"

            sahi = cv.get("sahi") or {}
            self.cv_slice_height_spin.setValue(int(sahi.get("slice_height", 640)))
            self.cv_slice_width_spin.setValue(int(sahi.get("slice_width", 640)))
            self.cv_overlap_spin.setValue(float(sahi.get("overlap_ratio", 0.2)))
        finally:
            self._loading = False

        self._refresh_models()
        self._update_available_rvt_targets()
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
        self._cancel_event.clear()
        self._set_run_enabled(False)
        self._logger.info("Lancement du pipeline (stub)")

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

