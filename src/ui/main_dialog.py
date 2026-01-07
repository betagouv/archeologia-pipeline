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

        self._log_emitter = _QtLogEmitter()
        self._log_emitter.message.connect(self._append_log)
        self._log_emitter.progress.connect(self._set_progress)
        self._log_emitter.stage.connect(self._set_stage)
        self._log_emitter.run_enabled.connect(self._set_run_enabled)
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
        self.jpg_mhs_cb = QCheckBox("Exporter en JPG (+JGW)")
        mdh_form.addRow("Nombre directions:", self.mdh_num_directions_spin)
        mdh_form.addRow("Élévation solaire:", self.mdh_sun_elevation_spin)
        mdh_form.addRow("Facteur VE:", self.mdh_ve_factor_spin)
        mdh_form.addRow("", self.mdh_save_8bit_cb)
        mdh_form.addRow("", self.jpg_mhs_cb)
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
        self.jpg_svf_cb = QCheckBox("Exporter en JPG (+JGW)")
        svf_form.addRow("Suppression bruit:", self.svf_noise_remove_spin)
        svf_form.addRow("Nombre directions:", self.svf_num_directions_spin)
        svf_form.addRow("Rayon:", self.svf_radius_spin)
        svf_form.addRow("Facteur VE:", self.svf_ve_factor_spin)
        svf_form.addRow("", self.svf_save_8bit_cb)
        svf_form.addRow("", self.jpg_svf_cb)
        self.rvt_tabs.addTab(svf_tab, "SVF")

        slope_tab = QWidget()
        slope_form = QFormLayout(slope_tab)
        self.slope_unit_combo = NoWheelComboBox()
        self.slope_unit_combo.addItem("Degrés", 0)
        self.slope_unit_combo.addItem("Pourcentage", 1)
        self.slope_ve_factor_spin = NoWheelSpinBox()
        self.slope_ve_factor_spin.setRange(1, 100)
        self.slope_save_8bit_cb = QCheckBox("Sauver en 8bit")
        self.jpg_slo_cb = QCheckBox("Exporter en JPG (+JGW)")
        slope_form.addRow("Unité:", self.slope_unit_combo)
        slope_form.addRow("Facteur VE:", self.slope_ve_factor_spin)
        slope_form.addRow("", self.slope_save_8bit_cb)
        slope_form.addRow("", self.jpg_slo_cb)
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
        self.jpg_ld_cb = QCheckBox("Exporter en JPG (+JGW)")
        ld_form.addRow("Résolution angulaire:", self.ldo_angular_res_spin)
        ld_form.addRow("Rayon min:", self.ldo_min_radius_spin)
        ld_form.addRow("Rayon max:", self.ldo_max_radius_spin)
        ld_form.addRow("Hauteur observateur:", self.ldo_observer_h_spin)
        ld_form.addRow("Facteur VE:", self.ldo_ve_factor_spin)
        ld_form.addRow("", self.ldo_save_8bit_cb)
        ld_form.addRow("", self.jpg_ld_cb)
        self.rvt_tabs.addTab(ld_tab, "LD")

        vat_tab = QWidget()
        vat_form = QFormLayout(vat_tab)
        self.vat_terrain_type_combo = NoWheelComboBox()
        self.vat_terrain_type_combo.addItem("Général", 0)
        self.vat_terrain_type_combo.addItem("Plat", 1)
        self.vat_terrain_type_combo.addItem("Pentu", 2)
        self.vat_save_8bit_cb = QCheckBox("Sauver en 8bit")
        self.jpg_vat_cb = QCheckBox("Exporter en JPG (+JGW)")
        vat_form.addRow("Type de terrain:", self.vat_terrain_type_combo)
        vat_form.addRow("", self.vat_save_8bit_cb)
        vat_form.addRow("", self.jpg_vat_cb)
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
        self.cv_refresh_models_btn = QPushButton("Actualiser")
        self.cv_refresh_models_btn.clicked.connect(self._refresh_models)
        model_layout.addWidget(self.cv_model_combo, 1)
        model_layout.addWidget(self.cv_refresh_models_btn, 0)
        cv_form.addRow("Modèle:", model_row)

        self.cv_target_rvt_combo = NoWheelComboBox()
        cv_form.addRow("RVT cible:", self.cv_target_rvt_combo)

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
        self.filter_expression_edit.textChanged.connect(self._on_any_changed)

        self.product_mnt_cb.toggled.connect(self._on_any_changed)
        self.product_densite_cb.toggled.connect(self._on_any_changed)
        # produits RVT : on utilise _on_rvt_products_changed pour recalculer les RVT cibles + sauvegarder

        self.mdh_num_directions_spin.valueChanged.connect(self._on_any_changed)
        self.mdh_sun_elevation_spin.valueChanged.connect(self._on_any_changed)
        self.mdh_ve_factor_spin.valueChanged.connect(self._on_any_changed)
        self.mdh_save_8bit_cb.toggled.connect(self._on_any_changed)
        self.jpg_mhs_cb.toggled.connect(self._on_any_changed)

        self.svf_noise_remove_spin.valueChanged.connect(self._on_any_changed)
        self.svf_num_directions_spin.valueChanged.connect(self._on_any_changed)
        self.svf_radius_spin.valueChanged.connect(self._on_any_changed)
        self.svf_ve_factor_spin.valueChanged.connect(self._on_any_changed)
        self.svf_save_8bit_cb.toggled.connect(self._on_any_changed)
        self.jpg_svf_cb.toggled.connect(self._on_any_changed)

        self.slope_unit_combo.currentIndexChanged.connect(self._on_any_changed)
        self.slope_ve_factor_spin.valueChanged.connect(self._on_any_changed)
        self.slope_save_8bit_cb.toggled.connect(self._on_any_changed)
        self.jpg_slo_cb.toggled.connect(self._on_any_changed)

        self.ldo_angular_res_spin.valueChanged.connect(self._on_any_changed)
        self.ldo_min_radius_spin.valueChanged.connect(self._on_any_changed)
        self.ldo_max_radius_spin.valueChanged.connect(self._on_any_changed)
        self.ldo_observer_h_spin.valueChanged.connect(self._on_any_changed)
        self.ldo_ve_factor_spin.valueChanged.connect(self._on_any_changed)
        self.ldo_save_8bit_cb.toggled.connect(self._on_any_changed)
        self.jpg_ld_cb.toggled.connect(self._on_any_changed)

        self.vat_terrain_type_combo.currentIndexChanged.connect(self._on_any_changed)
        self.vat_save_8bit_cb.toggled.connect(self._on_any_changed)
        self.jpg_vat_cb.toggled.connect(self._on_any_changed)

        self.cv_enabled_cb.toggled.connect(self._on_cv_enabled_changed)
        self.cv_model_combo.currentIndexChanged.connect(self._on_any_changed)
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
        self._set_lineedit_path_state(self.output_dir_edit, expect_dir=True)

        mode = self.data_mode_combo.currentData() or self._current_mode or "ign_laz"
        _, _, is_file = self._mode_mapping(mode)
        self._set_lineedit_path_state(self.specific_source_edit, expect_dir=not is_file)

    def _set_lineedit_path_state(self, edit: QLineEdit, expect_dir: bool) -> None:
        text = (edit.text() or "").strip()
        if not text:
            edit.setStyleSheet("")
            return

        p = Path(text)
        ok = p.exists() and (p.is_dir() if expect_dir else p.is_file())
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
            self.cv_slice_height_spin.setValue(int(sahi.get("slice_height", 750)))
            self.cv_slice_width_spin.setValue(int(sahi.get("slice_width", 750)))
            self.cv_overlap_spin.setValue(float(sahi.get("overlap_ratio", 0.2)))

            processing = self._config.get("processing") or {}
            self.mnt_resolution_spin.setValue(float(processing.get("mnt_resolution", 0.5)))
            self.density_resolution_spin.setValue(float(processing.get("density_resolution", 1.0)))
            self.tile_overlap_spin.setValue(int(processing.get("tile_overlap", 20)))
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

            jpg = ((processing.get("output_formats") or {}).get("jpg") or {})
            self.jpg_mhs_cb.setChecked(bool(jpg.get("M_HS", True)))
            self.jpg_svf_cb.setChecked(bool(jpg.get("SVF", True)))
            self.jpg_slo_cb.setChecked(bool(jpg.get("SLO", True)))
            self.jpg_ld_cb.setChecked(bool(jpg.get("LD", True)))
            self.jpg_vat_cb.setChecked(bool(jpg.get("VAT", True)))

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

        processing = self._config.setdefault("processing", {})
        processing["mnt_resolution"] = float(self.mnt_resolution_spin.value())
        processing["density_resolution"] = float(self.density_resolution_spin.value())
        processing["tile_overlap"] = int(self.tile_overlap_spin.value())
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

        output_formats = processing.setdefault("output_formats", {})
        jpg = output_formats.setdefault("jpg", {})
        jpg["M_HS"] = self.jpg_mhs_cb.isChecked()
        jpg["SVF"] = self.jpg_svf_cb.isChecked()
        jpg["SLO"] = self.jpg_slo_cb.isChecked()
        jpg["LD"] = self.jpg_ld_cb.isChecked()
        jpg["VAT"] = self.jpg_vat_cb.isChecked()

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
        processing = defaults.get("processing") or {}
        jpg = ((processing.get("output_formats") or {}).get("jpg") or {})
        rvt = defaults.get("rvt_params") or {}

        self._loading = True
        try:
            mdh = rvt.get("mdh") or {}
            self.mdh_num_directions_spin.setValue(int(mdh.get("num_directions", 16)))
            self.mdh_sun_elevation_spin.setValue(int(mdh.get("sun_elevation", 35)))
            self.mdh_ve_factor_spin.setValue(int(mdh.get("ve_factor", 1)))
            self.mdh_save_8bit_cb.setChecked(bool(mdh.get("save_as_8bit", True)))
            self.jpg_mhs_cb.setChecked(bool(jpg.get("M_HS", True)))

            svf = rvt.get("svf") or {}
            self.svf_noise_remove_spin.setValue(int(svf.get("noise_remove", 0)))
            self.svf_num_directions_spin.setValue(int(svf.get("num_directions", 16)))
            self.svf_radius_spin.setValue(int(svf.get("radius", 10)))
            self.svf_ve_factor_spin.setValue(int(svf.get("ve_factor", 1)))
            self.svf_save_8bit_cb.setChecked(bool(svf.get("save_as_8bit", True)))
            self.jpg_svf_cb.setChecked(bool(jpg.get("SVF", True)))

            slope = rvt.get("slope") or {}
            unit = int(slope.get("unit", 0))
            idx_unit = self.slope_unit_combo.findData(unit)
            self.slope_unit_combo.setCurrentIndex(idx_unit if idx_unit >= 0 else 0)
            self.slope_ve_factor_spin.setValue(int(slope.get("ve_factor", 1)))
            self.slope_save_8bit_cb.setChecked(bool(slope.get("save_as_8bit", True)))
            self.jpg_slo_cb.setChecked(bool(jpg.get("SLO", True)))

            ldo = rvt.get("ldo") or {}
            self.ldo_angular_res_spin.setValue(int(ldo.get("angular_res", 15)))
            self.ldo_min_radius_spin.setValue(int(ldo.get("min_radius", 10)))
            self.ldo_max_radius_spin.setValue(int(ldo.get("max_radius", 20)))
            self.ldo_observer_h_spin.setValue(float(ldo.get("observer_h", 1.7)))
            self.ldo_ve_factor_spin.setValue(int(ldo.get("ve_factor", 1)))
            self.ldo_save_8bit_cb.setChecked(bool(ldo.get("save_as_8bit", True)))
            self.jpg_ld_cb.setChecked(bool(jpg.get("LD", True)))

            vat = rvt.get("vat") or {}
            terrain = int(vat.get("terrain_type", 0))
            idx_terrain = self.vat_terrain_type_combo.findData(terrain)
            self.vat_terrain_type_combo.setCurrentIndex(idx_terrain if idx_terrain >= 0 else 0)
            self.vat_save_8bit_cb.setChecked(bool(vat.get("save_as_8bit", True)))
            self.jpg_vat_cb.setChecked(bool(jpg.get("VAT", True)))
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
            self.cv_slice_height_spin.setValue(int(sahi.get("slice_height", 750)))
            self.cv_slice_width_spin.setValue(int(sahi.get("slice_width", 750)))
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

        self._save_specific_source_only()
        self._save_from_widgets()
        self._cancel_event.clear()
        self._set_run_enabled(False)
        self._logger.info("Lancement du pipeline (stub)")

        def worker():
            file_handler = None
            root_logger = None
            root_prev_level = None
            try:
                files = (self._config.get("app") or {}).get("files") or {}
                mode = files.get("data_mode")
                output_dir_str = (files.get("output_dir") or "").strip()

                if output_dir_str:
                    output_dir = Path(output_dir_str)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    log_path = output_dir / f"pipeline_log_{ts}.txt"
                    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
                    file_handler.setLevel(logging.INFO)
                    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
                    root_logger = logging.getLogger()
                    root_prev_level = root_logger.level
                    if root_prev_level > logging.INFO:
                        root_logger.setLevel(logging.INFO)
                    root_logger.addHandler(file_handler)
                    self._logger.info(f"Logs écrits dans: {log_path}")

                self._logger.info(f"Mode: {mode}")
                self._logger.info(f"Sortie: {output_dir_str}")

                processing_cfg = (self._config.get("processing") or {})
                products_cfg = (processing_cfg.get("products") or {})
                if not isinstance(products_cfg, dict):
                    products_cfg = {}
                cv_cfg = self._config.get("computer_vision") or {}
                if not isinstance(cv_cfg, dict):
                    cv_cfg = {}

                from ..pipeline.preflight import run_preflight

                if not run_preflight(
                    mode=str(mode),
                    cv_config=cv_cfg,
                    products=products_cfg,
                    log=lambda m: self._logger.info(m),
                ):
                    return

                if mode in ("ign_laz", "local_laz"):
                    from ..pipeline.ign.downloader import download_ign_dalles
                    from ..pipeline.ign.preprocess import prepare_merged_tiles

                    from ..pipeline.modes.local_laz import run_local_laz

                    input_file = (files.get("input_file") or "").strip()
                    local_dir_str = (files.get("local_laz_dir") or "").strip()
                    if not output_dir_str:
                        self._logger.error("Aucun dossier de sortie n'est configuré")
                        return

                    output_dir = Path(output_dir_str)

                    if mode == "ign_laz":
                        # Progress ranges (global %)
                        download_range = (0, 25)
                        merge_range = (25, 35)
                        products_range = (35, 95)
                        finalize_range = (95, 100)

                        if not input_file:
                            self._logger.error("Mode IGN sélectionné mais aucun fichier de liste d'URLs n'est configuré")
                            return
                        input_path = Path(input_file)
                        if not input_path.exists():
                            self._logger.error(f"Fichier dalles IGN introuvable: {input_path}")
                            return

                        result = download_ign_dalles(
                            input_file=input_path,
                            output_dir=output_dir,
                            log=lambda m: self._logger.info(m),
                            progress=lambda p: self._log_emitter.progress.emit(
                                int(download_range[0] + (download_range[1] - download_range[0]) * (int(p) / 100.0))
                            ),
                            stage=lambda s: self._log_emitter.stage.emit(str(s)),
                            cancel=lambda: self._cancel_event.is_set(),
                        )
                    else:
                        if not local_dir_str:
                            self._logger.error("Mode local_laz sélectionné mais aucun dossier nuages locaux n'est configuré")
                            return
                        local_dir = Path(local_dir_str)
                        self._log_emitter.stage.emit("Indexation des nuages locaux")
                        self._log_emitter.progress.emit(0)
                        result = run_local_laz(
                            local_laz_dir=local_dir,
                            output_dir=output_dir,
                            log=lambda m: self._logger.info(m),
                        )

                    processing = (self._config.get("processing") or {})
                    tile_overlap = processing.get("tile_overlap", 5)
                    try:
                        tile_overlap = float(tile_overlap)
                    except Exception:
                        tile_overlap = 5.0

                    self._log_emitter.stage.emit("Fusion (voisins + merge)")
                    if mode == "ign_laz":
                        self._log_emitter.progress.emit(merge_range[0])
                    else:
                        self._log_emitter.progress.emit(0)

                    merged_result = prepare_merged_tiles(
                        sorted_list_file=result.sorted_list_file,
                        dalles_dir=result.dalles_dir,
                        output_dir=output_dir,
                        tile_overlap_percent=tile_overlap,
                        log=lambda m: self._logger.info(m),
                        cancel=lambda: self._cancel_event.is_set(),
                    )

                    if mode == "ign_laz":
                        self._log_emitter.progress.emit(merge_range[1])

                    products = (processing.get("products") or {})
                    need_mnt = bool(products.get("MNT", True)) or any(
                        bool(products.get(k, False)) for k in ("M_HS", "SVF", "SLO", "LD", "VAT")
                    )

                    if need_mnt and merged_result.merged_files:
                        from ..pipeline.ign.products.mnt import create_terrain_model
                        from ..pipeline.ign.products.density import create_density_map
                        from ..pipeline.ign.products.indices import create_visualization_products
                        from ..pipeline.ign.products.crop import crop_final_products
                        from ..pipeline.ign.products.results import copy_final_products_to_results

                        mnt_resolution = processing.get("mnt_resolution", 0.5)
                        try:
                            mnt_resolution = float(mnt_resolution)
                        except Exception:
                            mnt_resolution = 0.5

                        filter_expression = processing.get(
                            "filter_expression",
                            "Classification = 2 OR Classification = 6 OR Classification = 66 OR Classification = 67 OR Classification = 9",
                        )

                        density_resolution = processing.get("density_resolution", 1.0)
                        try:
                            density_resolution = float(density_resolution)
                        except Exception:
                            density_resolution = 1.0

                        output_structure = processing.get("output_structure", {})
                        if not isinstance(output_structure, dict):
                            output_structure = {}
                        output_formats = processing.get("output_formats", {})
                        if not isinstance(output_formats, dict):
                            output_formats = {}

                        rvt_params = self._config.get("rvt_params") or {}
                        if not isinstance(rvt_params, dict):
                            rvt_params = {}

                        cv_config = self._config.get("computer_vision") or {}
                        if not isinstance(cv_config, dict):
                            cv_config = {}
                        cv_enabled = bool(cv_config.get("enabled", False))
                        cv_target_rvt = str(cv_config.get("target_rvt", "LD"))
                        cv_generate_shapefiles = bool(cv_config.get("generate_shapefiles", False))
                        cv_labels_dir = None
                        cv_shp_dir = None
                        cv_tif_transform_data = {}

                        self._log_emitter.stage.emit("Traitement des dalles")
                        if mode == "ign_laz":
                            self._log_emitter.progress.emit(products_range[0])
                        else:
                            self._log_emitter.progress.emit(0)

                        total_mnt = len(merged_result.merged_files)
                        for i, merged_path in enumerate(merged_result.merged_files, start=1):
                            if self._cancel_event.is_set():
                                self._logger.info("Annulation demandée")
                                break

                            if mode == "ign_laz":
                                frac = (i - 1) / max(1, total_mnt)
                                pct = int(round(products_range[0] + (products_range[1] - products_range[0]) * frac))
                                self._log_emitter.progress.emit(pct)
                            else:
                                pct = int(round(100.0 * (i - 1) / max(1, total_mnt)))
                                self._log_emitter.progress.emit(pct)

                            tile_name = merged_path.name.replace(".copc.laz", "").replace(".laz", "")
                            self._log_emitter.stage.emit(f"Traitement dalle {i}/{total_mnt}: {tile_name}")
                            mnt_res = create_terrain_model(
                                input_laz_path=merged_path,
                                temp_dir=output_dir / "temp",
                                current_tile_name=tile_name,
                                mnt_resolution=mnt_resolution,
                                tile_overlap_percent=tile_overlap,
                                filter_expression=str(filter_expression),
                                log=lambda m: self._logger.info(m),
                            )

                            products_cfg = products if isinstance(products, dict) else {}
                            if bool(products_cfg.get("DENSITE", False)):
                                create_density_map(
                                    input_laz_path=merged_path,
                                    temp_dir=output_dir / "temp",
                                    current_tile_name=tile_name,
                                    density_resolution=density_resolution,
                                    tile_overlap_percent=tile_overlap,
                                    filter_expression=str(filter_expression),
                                    log=lambda m: self._logger.info(m),
                                )

                            create_visualization_products(
                                temp_dir=output_dir / "temp",
                                current_tile_name=tile_name,
                                products=products_cfg,
                                rvt_params=rvt_params,
                                log=lambda m: self._logger.info(m),
                            )

                            cropped = crop_final_products(
                                temp_dir=output_dir / "temp",
                                current_tile_name=tile_name,
                                products=products_cfg,
                                rvt_params=rvt_params,
                                log=lambda m: self._logger.info(m),
                            )
                            if cropped:
                                export_info = copy_final_products_to_results(
                                    temp_dir=output_dir / "temp",
                                    output_dir=output_dir,
                                    current_tile_name=tile_name,
                                    products=products_cfg,
                                    output_structure=output_structure,
                                    output_formats=output_formats,
                                    rvt_params=rvt_params,
                                    pyramids_config=(processing.get("pyramids") or {}),
                                    log=lambda m: self._logger.info(m),
                                )

                                if cv_enabled and bool(products_cfg.get(cv_target_rvt, False)):
                                    created_by_product = (export_info or {}).get("created_jpgs_by_product") or {}
                                    created_jpgs = []
                                    if isinstance(created_by_product, dict) and cv_target_rvt in created_by_product:
                                        created_jpgs = created_by_product.get(cv_target_rvt) or []
                                    if not created_jpgs:
                                        created_jpgs = (export_info or {}).get("created_jpgs") or []
                                    tif_transform_data = (export_info or {}).get("tif_transform_data") or {}
                                    if isinstance(tif_transform_data, dict):
                                        cv_tif_transform_data.update(tif_transform_data)
                                    for jpg_path in created_jpgs:
                                        try:
                                            if jpg_path is None:
                                                continue

                                            if not str(jpg_path).lower().replace("\\", "/").endswith("/jpg"):
                                                # Robustesse: si on reçoit autre chose qu'un fichier JPG, on ignore
                                                pass

                                            from ..pipeline.cv.runner import run_cv_on_folder

                                            jpg_dir_path = Path(jpg_path).parent
                                            rvt_base_dir = jpg_dir_path.parent

                                            # Sécurité: ne lancer la CV que sur le RVT sélectionné
                                            if str(rvt_base_dir.name).upper() != str(cv_target_rvt).upper():
                                                continue

                                            if cv_labels_dir is None:
                                                cv_labels_dir = jpg_dir_path
                                            if cv_shp_dir is None:
                                                cv_shp_dir = rvt_base_dir / "shapefiles"

                                            run_cv_on_folder(
                                                jpg_dir=jpg_dir_path,
                                                cv_config=cv_config,
                                                target_rvt=cv_target_rvt,
                                                rvt_base_dir=rvt_base_dir,
                                                tif_transform_data=cv_tif_transform_data,
                                                single_jpg=Path(jpg_path),
                                                run_shapefile_dedup=False,
                                                log=lambda m: self._logger.info(m),
                                            )
                                        except Exception as e:
                                            self._logger.error(f"Erreur Computer Vision: {e}")

                            # After finishing this tile, bump progress a bit (monotonic)
                            if mode == "ign_laz":
                                frac_done = i / max(1, total_mnt)
                                pct_done = int(round(products_range[0] + (products_range[1] - products_range[0]) * frac_done))
                                self._log_emitter.progress.emit(pct_done)

                        if cv_enabled and cv_generate_shapefiles and cv_labels_dir is not None and cv_shp_dir is not None:
                            if mode == "ign_laz":
                                self._log_emitter.stage.emit("Finalisation (shapefiles)")
                                self._log_emitter.progress.emit(finalize_range[0])
                            try:
                                from ..pipeline.cv.runner import deduplicate_cv_shapefiles_final

                                deduplicate_cv_shapefiles_final(
                                    labels_dir=cv_labels_dir,
                                    shp_dir=cv_shp_dir,
                                    target_rvt=cv_target_rvt,
                                    cv_config=cv_config,
                                    tif_transform_data=cv_tif_transform_data,
                                    temp_dir=output_dir / "temp",
                                    crs="EPSG:2154",
                                    log=lambda m: self._logger.info(m),
                                )
                            except Exception as e:
                                self._logger.error(f"Erreur déduplication shapefiles CV: {e}")

                        self._log_emitter.progress.emit(100)

                    self._log_emitter.stage.emit("Terminé")
                    self._log_emitter.progress.emit(100)
                    if mode == "ign_laz":
                        self._logger.info(
                            f"Téléchargement IGN terminé: {result.downloaded} téléchargés, {result.skipped_existing} déjà présents (total {result.total}). Fichier trié: {result.sorted_list_file}"
                        )
                        self._logger.info(
                            f"Fusion IGN terminée: {len(merged_result.merged_files)} fichiers fusionnés. Dossier: {merged_result.merged_dir}"
                        )

                    if cv_enabled:
                        try:
                            from ..pipeline.modes.existing_rvt import run_existing_rvt

                            target_rvt = str((cv_config or {}).get("target_rvt", "LD"))
                            rvt_cfg = output_structure.get("RVT", {}) if isinstance(output_structure, dict) else {}
                            base_dir_name = str(rvt_cfg.get("base_dir", "RVT"))
                            type_dir_name = str(rvt_cfg.get(target_rvt, target_rvt))
                            generated_rvt_tif_dir = (output_dir / "results") / base_dir_name / type_dir_name / "tif"

                            if not generated_rvt_tif_dir.exists() or not generated_rvt_tif_dir.is_dir():
                                self._logger.error(
                                    f"Computer Vision demandée mais aucun dossier RVT/TIF trouvé: {generated_rvt_tif_dir}"
                                )
                            else:
                                self._log_emitter.stage.emit("Computer Vision (existing MNT)")
                                self._log_emitter.progress.emit(90)
                                run_existing_rvt(
                                    existing_rvt_dir=generated_rvt_tif_dir,
                                    output_dir=output_dir,
                                    cv_config=cv_config,
                                    output_structure=output_structure,
                                    log=lambda m: self._logger.info(m),
                                )
                                if cv_generate_shp:
                                    self._logger.info("Computer Vision (existing MNT): shapefiles générés")
                        except Exception as e:
                            self._logger.error(f"Erreur Computer Vision (existing MNT): {e}")

                    self._log_emitter.stage.emit("Terminé")
                    self._log_emitter.progress.emit(100)
                    self._logger.info(f"Mode existing_mnt terminé: {res.total} MNT traités")
                    return

                if mode == "existing_mnt":
                    from ..pipeline.modes.existing_mnt import run_existing_mnt

                    existing_mnt_dir_str = (files.get("existing_mnt_dir") or "").strip()
                    if not existing_mnt_dir_str:
                        self._logger.error("Mode existing_mnt sélectionné mais aucun dossier MNT n'est configuré")
                        return
                    if not output_dir_str:
                        self._logger.error("Aucun dossier de sortie n'est configuré")
                        return

                    output_dir = Path(output_dir_str)
                    output_dir.mkdir(parents=True, exist_ok=True)

                    processing_cfg = (self._config.get("processing") or {})
                    output_structure = processing_cfg.get("output_structure", {})
                    if not isinstance(output_structure, dict):
                        output_structure = {}

                    output_formats = processing_cfg.get("output_formats", {})
                    if not isinstance(output_formats, dict):
                        output_formats = {}

                    rvt_params = self._config.get("rvt_params") or {}
                    if not isinstance(rvt_params, dict):
                        rvt_params = {}

                    cv_config = self._config.get("computer_vision") or {}
                    if not isinstance(cv_config, dict):
                        cv_config = {}

                    self._log_emitter.stage.emit("Computer Vision (existing MNT)")
                    self._log_emitter.progress.emit(0)

                    res = run_existing_mnt(
                        existing_mnt_dir=Path(existing_mnt_dir_str),
                        output_dir=output_dir,
                        products=products_cfg,
                        output_structure=output_structure,
                        output_formats=output_formats,
                        pyramids_config=(processing_cfg.get("pyramids") or {}),
                        rvt_params=rvt_params,
                        log=lambda m: self._logger.info(m),
                    )

                    self._log_emitter.stage.emit("Terminé")
                    self._log_emitter.progress.emit(100)
                    self._logger.info(f"Mode existing_mnt terminé: {res.total} MNT traités")
                    return

                if mode == "existing_rvt":
                    from ..pipeline.modes.existing_rvt import run_existing_rvt

                    existing_rvt_dir_str = (files.get("existing_rvt_dir") or "").strip()
                    if not existing_rvt_dir_str:
                        self._logger.error("Mode existing_rvt sélectionné mais aucun dossier RVT n'est configuré")
                        return
                    if not output_dir_str:
                        self._logger.error("Aucun dossier de sortie n'est configuré")
                        return

                    output_dir = Path(output_dir_str)
                    output_dir.mkdir(parents=True, exist_ok=True)

                    processing_cfg = (self._config.get("processing") or {})
                    output_structure = processing_cfg.get("output_structure", {})
                    if not isinstance(output_structure, dict):
                        output_structure = {}

                    cv_config = self._config.get("computer_vision") or {}
                    if not isinstance(cv_config, dict):
                        cv_config = {}

                    self._log_emitter.stage.emit("Computer Vision (existing RVT)")
                    self._log_emitter.progress.emit(0)

                    res = run_existing_rvt(
                        existing_rvt_dir=Path(existing_rvt_dir_str),
                        output_dir=output_dir,
                        cv_config=cv_config,
                        output_structure=output_structure,
                        log=lambda m: self._logger.info(m),
                    )

                    self._log_emitter.stage.emit("Terminé")
                    self._log_emitter.progress.emit(100)
                    self._logger.info(f"Mode existing_rvt terminé: {res.total_images} images")
                    return

                self._log_emitter.stage.emit("Préparation")
                self._log_emitter.progress.emit(10)
                time.sleep(0.4)
                if self._cancel_event.is_set():
                    self._logger.info("Pipeline annulé")
                    return

                self._logger.info("(stub) Préparation...")
                self._log_emitter.stage.emit("Traitement")
                self._log_emitter.progress.emit(40)
                time.sleep(0.4)
                if self._cancel_event.is_set():
                    self._logger.info("Pipeline annulé")
                    return

                self._logger.info("(stub) Traitement...")
                self._log_emitter.stage.emit("Finalisation")
                self._log_emitter.progress.emit(80)
                time.sleep(0.4)
                if self._cancel_event.is_set():
                    self._logger.info("Pipeline annulé")
                    return

                self._log_emitter.progress.emit(100)
                self._logger.info("Pipeline terminé (stub)")
            except Exception:
                self._logger.exception("Erreur pendant l'exécution du pipeline (stub)")
            finally:
                try:
                    if file_handler is not None:
                        if root_logger is not None:
                            root_logger.removeHandler(file_handler)
                        file_handler.close()
                except Exception:
                    pass
                try:
                    if root_logger is not None and root_prev_level is not None:
                        root_logger.setLevel(root_prev_level)
                except Exception:
                    pass
                self._log_emitter.run_enabled.emit(True)

        threading.Thread(target=worker, daemon=True).start()

    def _on_cancel_clicked(self) -> None:
        if self.cancel_btn.isEnabled():
            self._cancel_event.set()
            self._logger.info("Annulation demandée...")

