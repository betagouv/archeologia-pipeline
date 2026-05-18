"""Adaptateur widget Qt ↔ dictionnaire de configuration.

Avant ce module, ``MainDialog`` portait deux méthodes d'environ 100
lignes chacune (``_load_into_widgets`` et ``_collect_config_from_widgets``)
qui mélangeaient :

- la **structure** du dict de config (``processing.products.MNT``,
  ``rvt_params.mdh.num_directions``…) ;
- l'accès direct à 50+ widgets ;
- des défauts par champ (``int(processing.get("max_workers", 4))``…).

Le moindre ajout de paramètre (ex. nouveau RVT ou nouveau hyperparam)
demandait de toucher 3 endroits dans le dialog plus la sauvegarde —
sans aucune garantie de cohérence schema/UI.

Ce module factorise la conversion. Il ne touche que les **champs
simples** (spinbox, checkbox, line edit, combobox via ``currentData``) :
- output_dir, data_mode + specific_source
- detection enabled / confidence / iou / annotated / shp
- processing (résolutions, overlap, workers, filter_expression)
- products (8 cases à cocher)
- rvt_params (mdh, svf, slope, ldo, slrm, vat)

La gestion du **tableau des runs CV** et des **classes sélectionnées
par modèle** reste dans ``MainDialog`` — elles dépendent de callbacks
spécifiques du dialog (création de lignes, signaux item-changed,
combobox dynamiques) qu'extraire ici aurait fragilisé la chaîne sans
gain de testabilité.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


# ----------------------------------------------------------------------
# Surface widgets : ce dont l'adaptateur a besoin
# ----------------------------------------------------------------------
@dataclass
class WidgetBag:
    """Refs aux widgets manipulés par :class:`ConfigWidgetAdapter`.

    On ne type pas en QtWidget pour rester compatible avec les tests
    qui injectent des mocks (les tests Qt headless sont coûteux et
    fragiles ; un mock minimal suffit pour vérifier la bijection
    config ↔ widgets).
    """

    # Mode d'affichage Simple/Expert (combobox UI uniquement)
    mode_combo: Any

    # Sources fichiers
    output_dir_edit: Any
    data_mode_combo: Any
    specific_source_edit: Any

    # Computer Vision (hors runs table — gérée par le dialog)
    detection_enabled_cb: Any
    det_confidence_spin: Any
    det_iou_spin: Any
    det_generate_annotated_cb: Any
    det_generate_shp_cb: Any

    # Processing
    mnt_resolution_spin: Any
    density_resolution_spin: Any
    tile_overlap_spin: Any
    max_workers_spin: Any
    filter_expression_edit: Any

    # Products (dict[code → QCheckBox])
    product_cbs: Dict[str, Any]

    # RVT params — MDH
    mdh_num_directions_spin: Any
    mdh_sun_elevation_spin: Any
    mdh_ve_factor_spin: Any
    mdh_save_8bit_cb: Any

    # RVT params — SVF
    svf_noise_remove_spin: Any
    svf_num_directions_spin: Any
    svf_radius_spin: Any
    svf_ve_factor_spin: Any
    svf_save_8bit_cb: Any

    # RVT params — Slope
    slope_unit_combo: Any
    slope_ve_factor_spin: Any
    slope_save_8bit_cb: Any

    # RVT params — LDO
    ldo_angular_res_spin: Any
    ldo_min_radius_spin: Any
    ldo_max_radius_spin: Any
    ldo_observer_h_spin: Any
    ldo_ve_factor_spin: Any
    ldo_save_8bit_cb: Any

    # RVT params — SLRM
    slrm_radius_spin: Any
    slrm_ve_factor_spin: Any
    slrm_save_8bit_cb: Any

    # RVT params — VAT
    vat_terrain_type_combo: Any
    vat_save_8bit_cb: Any


# ----------------------------------------------------------------------
# Mapping mode → clé de stockage du chemin spécifique
# ----------------------------------------------------------------------
# data_mode → (clé dans files{}, … les autres tuples sont gardés en
# dialog pour les libellés affichés). Cette table est petite et
# stable ; on la duplique ici pour rester autonome.
_MODE_KEY = {
    "ign_laz": "input_file",
    "local_laz": "local_laz_dir",
    "existing_mnt": "existing_mnt_dir",
    "existing_rvt": "existing_rvt_dir",
}


def _mode_specific_key(mode: str) -> str:
    return _MODE_KEY.get(mode, "input_file")


# ----------------------------------------------------------------------
# Codes produits (canonique, ordre stable pour les listes "active")
# ----------------------------------------------------------------------
_PRODUCT_CODES = ("MNT", "DENSITE", "M_HS", "SVF", "SLO", "LD", "SLRM", "VAT")
_PRODUCT_DEFAULTS = {
    "MNT": True,
    "DENSITE": False,
    "M_HS": False,
    "SVF": False,
    "SLO": False,
    "LD": False,
    "SLRM": False,
    "VAT": False,
}

# Filtre PDAL par défaut. Synchronisé avec
# :func:`app.run_context._build_processing_config`.
_DEFAULT_FILTER_EXPRESSION = (
    "Classification = 2 OR Classification = 6 OR Classification = 66 "
    "OR Classification = 67 OR Classification = 9"
)


# ----------------------------------------------------------------------
# Adaptateur
# ----------------------------------------------------------------------
class ConfigWidgetAdapter:
    """Adaptateur bijectif config dict ↔ widgets Qt.

    Méthodes ne touchant pas aux signaux Qt (le dialog reste responsable
    du ``blockSignals`` global pendant un :meth:`apply_to_widgets`).
    """

    def __init__(self, bag: WidgetBag):
        self._b = bag

    # ------------------------------------------------------------------
    # config → widgets
    # ------------------------------------------------------------------
    def apply_to_widgets(self, config: Mapping[str, Any]) -> None:
        """Pousse ``config`` dans les widgets."""
        b = self._b

        # UI / display_mode
        ui_cfg = config.get("ui") or {}
        display_mode = str(ui_cfg.get("display_mode") or "simple")
        idx_display = b.mode_combo.findData(display_mode)
        b.mode_combo.setCurrentIndex(idx_display if idx_display >= 0 else 0)

        # Files / data_mode + specific_source : data_mode chargé EN DERNIER
        # par :meth:`apply_data_mode` (son signal currentIndexChanged
        # déclenche _on_data_mode_changed côté dialog → _save_from_widgets ;
        # on veut que tout l'état soit restauré avant ce déclenchement). Le
        # dialog se charge de l'ordonnancement final ; ici on n'écrit ni le
        # data_mode ni la specific_source (cette dernière dépend du mode).
        files = (config.get("app") or {}).get("files") or {}
        b.output_dir_edit.setText(files.get("output_dir") or "")

        # CV
        cv = config.get("computer_vision") or {}
        b.detection_enabled_cb.setChecked(bool(cv.get("enabled", False)))
        b.det_confidence_spin.setValue(float(cv.get("confidence_threshold", 0.3)))
        b.det_iou_spin.setValue(float(cv.get("iou_threshold", 0.5)))
        b.det_generate_annotated_cb.setChecked(bool(cv.get("generate_annotated_images", False)))
        b.det_generate_shp_cb.setChecked(bool(cv.get("generate_shapefiles", False)))

        # Processing
        processing = config.get("processing") or {}
        b.mnt_resolution_spin.setValue(float(processing.get("mnt_resolution", 0.5)))
        b.density_resolution_spin.setValue(float(processing.get("density_resolution", 1.0)))
        b.tile_overlap_spin.setValue(int(processing.get("tile_overlap", 20)))
        b.max_workers_spin.setValue(int(processing.get("max_workers", 4)))
        # Si le filtre est vide (config legacy ou utilisateur ayant vidé
        # le champ), on affiche le défaut — sinon le pipeline filtrerait
        # silencieusement avec ce défaut sans que l'utilisateur le voie.
        filter_value = (processing.get("filter_expression") or "").strip()
        b.filter_expression_edit.setText(filter_value or _DEFAULT_FILTER_EXPRESSION)

        # Products
        products = processing.get("products") or {}
        for pkey in _PRODUCT_CODES:
            cb = b.product_cbs.get(pkey)
            if cb is not None:
                cb.setChecked(bool(products.get(pkey, _PRODUCT_DEFAULTS[pkey])))

        # RVT params
        rvt = config.get("rvt_params") or {}
        self._apply_mdh(rvt.get("mdh") or {})
        self._apply_svf(rvt.get("svf") or {})
        self._apply_slope(rvt.get("slope") or {})
        self._apply_ldo(rvt.get("ldo") or {})
        self._apply_slrm(rvt.get("slrm") or {})
        self._apply_vat(rvt.get("vat") or {})

    def apply_data_mode(self, config: Mapping[str, Any]) -> None:
        """Restaure ``data_mode`` + ``specific_source`` séparément.

        Voir docstring de :meth:`apply_to_widgets` : le dialog veut le
        déclencher en dernier pour orchestrer son flux d'événements.

        ``specific_source_edit`` est chargé ici (et pas dans
        :meth:`apply_to_widgets`) parce que la clé JSON dont sa valeur
        provient dépend du ``data_mode`` (``input_file`` / ``local_laz_dir``
        / ``existing_mnt_dir`` / ``existing_rvt_dir``). Symétrise
        :meth:`collect_into` qui écrit dans cette même clé mode-dépendante.
        """
        files = (config.get("app") or {}).get("files") or {}
        mode = files.get("data_mode") or "ign_laz"
        idx_mode = self._b.data_mode_combo.findData(mode)
        self._b.data_mode_combo.setCurrentIndex(idx_mode if idx_mode >= 0 else 0)
        self._b.specific_source_edit.setText(files.get(_mode_specific_key(mode)) or "")

    # ------------------------------------------------------------------
    # widgets → config
    # ------------------------------------------------------------------
    def collect_into(self, config: Dict[str, Any]) -> None:
        """Met à jour ``config`` (in-place) avec l'état actuel des widgets."""
        b = self._b

        # UI
        ui = config.setdefault("ui", {})
        ui["display_mode"] = b.mode_combo.currentData() or "simple"

        # Files
        app = config.setdefault("app", {})
        files = app.setdefault("files", {})
        files["output_dir"] = b.output_dir_edit.text().strip()
        mode = b.data_mode_combo.currentData() or "ign_laz"
        files["data_mode"] = mode
        files[_mode_specific_key(mode)] = b.specific_source_edit.text().strip()

        # CV
        cv = config.setdefault("computer_vision", {})
        cv["enabled"] = bool(b.detection_enabled_cb.isChecked())
        cv["confidence_threshold"] = float(b.det_confidence_spin.value())
        cv["iou_threshold"] = float(b.det_iou_spin.value())
        cv["generate_annotated_images"] = bool(b.det_generate_annotated_cb.isChecked())
        cv["generate_shapefiles"] = bool(b.det_generate_shp_cb.isChecked())

        # Processing
        processing = config.setdefault("processing", {})
        processing["mnt_resolution"] = float(b.mnt_resolution_spin.value())
        processing["density_resolution"] = float(b.density_resolution_spin.value())
        processing["tile_overlap"] = int(b.tile_overlap_spin.value())
        processing["max_workers"] = int(b.max_workers_spin.value())
        processing["filter_expression"] = b.filter_expression_edit.text().strip()

        products = processing.setdefault("products", {})
        for pkey in _PRODUCT_CODES:
            cb = b.product_cbs.get(pkey)
            if cb is not None:
                products[pkey] = bool(cb.isChecked())

        # RVT params
        rvt = config.setdefault("rvt_params", {})
        self._collect_mdh(rvt.setdefault("mdh", {}))
        self._collect_svf(rvt.setdefault("svf", {}))
        self._collect_slope(rvt.setdefault("slope", {}))
        self._collect_ldo(rvt.setdefault("ldo", {}))
        self._collect_slrm(rvt.setdefault("slrm", {}))
        self._collect_vat(rvt.setdefault("vat", {}))

    # ------------------------------------------------------------------
    # RVT params : un sous-bloc par produit (on évite la sur-abstraction
    # — chaque produit a son propre jeu de paramètres typés)
    # ------------------------------------------------------------------
    def _apply_mdh(self, mdh: Mapping[str, Any]) -> None:
        b = self._b
        b.mdh_num_directions_spin.setValue(int(mdh.get("num_directions", 16)))
        b.mdh_sun_elevation_spin.setValue(int(mdh.get("sun_elevation", 35)))
        b.mdh_ve_factor_spin.setValue(int(mdh.get("ve_factor", 1)))
        b.mdh_save_8bit_cb.setChecked(bool(mdh.get("save_as_8bit", True)))

    def _collect_mdh(self, mdh: Dict[str, Any]) -> None:
        b = self._b
        mdh["num_directions"] = int(b.mdh_num_directions_spin.value())
        mdh["sun_elevation"] = int(b.mdh_sun_elevation_spin.value())
        mdh["ve_factor"] = int(b.mdh_ve_factor_spin.value())
        mdh["save_as_8bit"] = bool(b.mdh_save_8bit_cb.isChecked())

    def _apply_svf(self, svf: Mapping[str, Any]) -> None:
        b = self._b
        b.svf_noise_remove_spin.setValue(int(svf.get("noise_remove", 0)))
        b.svf_num_directions_spin.setValue(int(svf.get("num_directions", 16)))
        b.svf_radius_spin.setValue(int(svf.get("radius", 10)))
        b.svf_ve_factor_spin.setValue(int(svf.get("ve_factor", 1)))
        b.svf_save_8bit_cb.setChecked(bool(svf.get("save_as_8bit", True)))

    def _collect_svf(self, svf: Dict[str, Any]) -> None:
        b = self._b
        svf["noise_remove"] = int(b.svf_noise_remove_spin.value())
        svf["num_directions"] = int(b.svf_num_directions_spin.value())
        svf["radius"] = int(b.svf_radius_spin.value())
        svf["ve_factor"] = int(b.svf_ve_factor_spin.value())
        svf["save_as_8bit"] = bool(b.svf_save_8bit_cb.isChecked())

    def _apply_slope(self, slope: Mapping[str, Any]) -> None:
        b = self._b
        unit = int(slope.get("unit", 0))
        idx_unit = b.slope_unit_combo.findData(unit)
        b.slope_unit_combo.setCurrentIndex(idx_unit if idx_unit >= 0 else 0)
        b.slope_ve_factor_spin.setValue(int(slope.get("ve_factor", 1)))
        b.slope_save_8bit_cb.setChecked(bool(slope.get("save_as_8bit", True)))

    def _collect_slope(self, slope: Dict[str, Any]) -> None:
        b = self._b
        slope["unit"] = int(b.slope_unit_combo.currentData())
        slope["ve_factor"] = int(b.slope_ve_factor_spin.value())
        slope["save_as_8bit"] = bool(b.slope_save_8bit_cb.isChecked())

    def _apply_ldo(self, ldo: Mapping[str, Any]) -> None:
        b = self._b
        b.ldo_angular_res_spin.setValue(int(ldo.get("angular_res", 15)))
        b.ldo_min_radius_spin.setValue(int(ldo.get("min_radius", 10)))
        b.ldo_max_radius_spin.setValue(int(ldo.get("max_radius", 20)))
        b.ldo_observer_h_spin.setValue(float(ldo.get("observer_h", 1.7)))
        b.ldo_ve_factor_spin.setValue(int(ldo.get("ve_factor", 1)))
        b.ldo_save_8bit_cb.setChecked(bool(ldo.get("save_as_8bit", True)))

    def _collect_ldo(self, ldo: Dict[str, Any]) -> None:
        b = self._b
        ldo["angular_res"] = int(b.ldo_angular_res_spin.value())
        ldo["min_radius"] = int(b.ldo_min_radius_spin.value())
        ldo["max_radius"] = int(b.ldo_max_radius_spin.value())
        ldo["observer_h"] = float(b.ldo_observer_h_spin.value())
        ldo["ve_factor"] = int(b.ldo_ve_factor_spin.value())
        ldo["save_as_8bit"] = bool(b.ldo_save_8bit_cb.isChecked())

    def _apply_slrm(self, slrm: Mapping[str, Any]) -> None:
        b = self._b
        b.slrm_radius_spin.setValue(int(slrm.get("radius", 20)))
        b.slrm_ve_factor_spin.setValue(int(slrm.get("ve_factor", 1)))
        b.slrm_save_8bit_cb.setChecked(bool(slrm.get("save_as_8bit", True)))

    def _collect_slrm(self, slrm: Dict[str, Any]) -> None:
        b = self._b
        slrm["radius"] = int(b.slrm_radius_spin.value())
        slrm["ve_factor"] = int(b.slrm_ve_factor_spin.value())
        slrm["save_as_8bit"] = bool(b.slrm_save_8bit_cb.isChecked())

    def _apply_vat(self, vat: Mapping[str, Any]) -> None:
        b = self._b
        terrain = int(vat.get("terrain_type", 0))
        idx_terrain = b.vat_terrain_type_combo.findData(terrain)
        b.vat_terrain_type_combo.setCurrentIndex(idx_terrain if idx_terrain >= 0 else 0)
        b.vat_save_8bit_cb.setChecked(bool(vat.get("save_as_8bit", True)))

    def _collect_vat(self, vat: Dict[str, Any]) -> None:
        b = self._b
        vat["terrain_type"] = int(b.vat_terrain_type_combo.currentData())
        vat["save_as_8bit"] = bool(b.vat_save_8bit_cb.isChecked())
