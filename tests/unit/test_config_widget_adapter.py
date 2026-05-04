"""Tests de :class:`ConfigWidgetAdapter`.

L'adaptateur est testable hors Qt : on injecte des mocks qui
implémentent l'API minimale (``setText/text``, ``setValue/value``,
``setChecked/isChecked``, ``findData/setCurrentIndex/currentData``).
La bijection est vérifiée par roundtrip : config → widgets → config.
"""
from __future__ import annotations

import pytest

# L'adaptateur n'importe que du Python pur — pas de qgis.PyQt. Le
# package ``ui`` a un __init__.py vide, donc l'import direct fonctionne.
from ui.config_widget_adapter import ConfigWidgetAdapter, WidgetBag


# ----------------------------------------------------------------------
# Mocks de widgets Qt
# ----------------------------------------------------------------------
class _LineEditMock:
    def __init__(self, text: str = ""):
        self._t = text

    def setText(self, value: str) -> None:
        self._t = value

    def text(self) -> str:
        return self._t


class _SpinMock:
    def __init__(self, value=0):
        self._v = value

    def setValue(self, value) -> None:
        self._v = value

    def value(self):
        return self._v


class _CheckMock:
    def __init__(self, checked: bool = False):
        self._c = checked

    def setChecked(self, value: bool) -> None:
        self._c = bool(value)

    def isChecked(self) -> bool:
        return self._c


class _ComboMock:
    """Combobox mocké : simule findData / setCurrentIndex / currentData
    via une liste de tuples ``(label, data)`` et un index courant.
    """

    def __init__(self, items: list, index: int = 0):
        self._items = list(items)  # [(label, data), …]
        self._idx = index

    def findData(self, data) -> int:
        for i, (_label, d) in enumerate(self._items):
            if d == data:
                return i
        return -1

    def setCurrentIndex(self, idx: int) -> None:
        if 0 <= idx < len(self._items):
            self._idx = idx

    def currentData(self):
        return self._items[self._idx][1] if self._items else None


# ----------------------------------------------------------------------
# Fixture : un bag complet de widgets vides
# ----------------------------------------------------------------------
def _make_bag() -> WidgetBag:
    return WidgetBag(
        mode_combo=_ComboMock([("Simple", "simple"), ("Expert", "expert")]),
        output_dir_edit=_LineEditMock(""),
        data_mode_combo=_ComboMock([
            ("IGN", "ign_laz"),
            ("Local", "local_laz"),
            ("MNT", "existing_mnt"),
            ("RVT", "existing_rvt"),
        ]),
        specific_source_edit=_LineEditMock(""),
        detection_enabled_cb=_CheckMock(),
        det_confidence_spin=_SpinMock(0.0),
        det_iou_spin=_SpinMock(0.0),
        det_generate_annotated_cb=_CheckMock(),
        det_generate_shp_cb=_CheckMock(),
        mnt_resolution_spin=_SpinMock(0.0),
        density_resolution_spin=_SpinMock(0.0),
        tile_overlap_spin=_SpinMock(0),
        max_workers_spin=_SpinMock(0),
        filter_expression_edit=_LineEditMock(""),
        product_cbs={
            "MNT": _CheckMock(), "DENSITE": _CheckMock(),
            "M_HS": _CheckMock(), "SVF": _CheckMock(),
            "SLO": _CheckMock(), "LD": _CheckMock(),
            "SLRM": _CheckMock(), "VAT": _CheckMock(),
        },
        mdh_num_directions_spin=_SpinMock(0),
        mdh_sun_elevation_spin=_SpinMock(0),
        mdh_ve_factor_spin=_SpinMock(0),
        mdh_save_8bit_cb=_CheckMock(),
        svf_noise_remove_spin=_SpinMock(0),
        svf_num_directions_spin=_SpinMock(0),
        svf_radius_spin=_SpinMock(0),
        svf_ve_factor_spin=_SpinMock(0),
        svf_save_8bit_cb=_CheckMock(),
        slope_unit_combo=_ComboMock([("Degrés", 0), ("Pourcent", 1)]),
        slope_ve_factor_spin=_SpinMock(0),
        slope_save_8bit_cb=_CheckMock(),
        ldo_angular_res_spin=_SpinMock(0),
        ldo_min_radius_spin=_SpinMock(0),
        ldo_max_radius_spin=_SpinMock(0),
        ldo_observer_h_spin=_SpinMock(0.0),
        ldo_ve_factor_spin=_SpinMock(0),
        ldo_save_8bit_cb=_CheckMock(),
        slrm_radius_spin=_SpinMock(0),
        slrm_ve_factor_spin=_SpinMock(0),
        slrm_save_8bit_cb=_CheckMock(),
        vat_terrain_type_combo=_ComboMock([("Plat", 0), ("Vallonné", 1)]),
        vat_save_8bit_cb=_CheckMock(),
    )


# ----------------------------------------------------------------------
# apply_to_widgets : bonnes valeurs poussées
# ----------------------------------------------------------------------
class TestApplyToWidgets:
    def test_output_dir_pushed(self):
        bag = _make_bag()
        adapter = ConfigWidgetAdapter(bag)
        adapter.apply_to_widgets({"app": {"files": {"output_dir": "/tmp/out"}}})
        assert bag.output_dir_edit.text() == "/tmp/out"

    def test_display_mode_pushed(self):
        bag = _make_bag()
        ConfigWidgetAdapter(bag).apply_to_widgets({"ui": {"display_mode": "expert"}})
        assert bag.mode_combo.currentData() == "expert"

    def test_processing_pushed(self):
        bag = _make_bag()
        ConfigWidgetAdapter(bag).apply_to_widgets({
            "processing": {
                "mnt_resolution": 0.25,
                "density_resolution": 2.5,
                "tile_overlap": 10,
                "max_workers": 8,
                "filter_expression": "Classification = 2",
            }
        })
        assert bag.mnt_resolution_spin.value() == 0.25
        assert bag.density_resolution_spin.value() == 2.5
        assert bag.tile_overlap_spin.value() == 10
        assert bag.max_workers_spin.value() == 8
        assert bag.filter_expression_edit.text() == "Classification = 2"

    def test_empty_filter_expression_shows_default(self):
        """Si la config a un filtre vide, l'UI doit afficher le défaut
        PDAL au lieu de laisser le champ vide (sinon l'utilisateur
        croit qu'aucun filtre n'est appliqué)."""
        bag = _make_bag()
        ConfigWidgetAdapter(bag).apply_to_widgets({"processing": {"filter_expression": ""}})
        assert "Classification = 2" in bag.filter_expression_edit.text()

    def test_products_pushed(self):
        bag = _make_bag()
        ConfigWidgetAdapter(bag).apply_to_widgets({
            "processing": {"products": {"MNT": True, "SVF": True, "M_HS": False}}
        })
        assert bag.product_cbs["MNT"].isChecked() is True
        assert bag.product_cbs["SVF"].isChecked() is True
        assert bag.product_cbs["M_HS"].isChecked() is False

    def test_products_use_defaults_when_missing(self):
        bag = _make_bag()
        # Aucune clé "products" → MNT défaut True, autres False.
        ConfigWidgetAdapter(bag).apply_to_widgets({})
        assert bag.product_cbs["MNT"].isChecked() is True
        assert bag.product_cbs["DENSITE"].isChecked() is False

    def test_rvt_mdh_pushed(self):
        bag = _make_bag()
        ConfigWidgetAdapter(bag).apply_to_widgets({
            "rvt_params": {"mdh": {"num_directions": 8, "sun_elevation": 45, "ve_factor": 2}}
        })
        assert bag.mdh_num_directions_spin.value() == 8
        assert bag.mdh_sun_elevation_spin.value() == 45
        assert bag.mdh_ve_factor_spin.value() == 2

    def test_rvt_slope_combo_via_finddata(self):
        bag = _make_bag()
        ConfigWidgetAdapter(bag).apply_to_widgets({"rvt_params": {"slope": {"unit": 1}}})
        assert bag.slope_unit_combo.currentData() == 1

    def test_data_mode_separate_method(self):
        """``apply_data_mode`` doit être appelé séparément (le dialog
        veut le déclencher en dernier pour orchestrer ses signaux)."""
        bag = _make_bag()
        adapter = ConfigWidgetAdapter(bag)
        adapter.apply_to_widgets({"app": {"files": {"data_mode": "local_laz"}}})
        # apply_to_widgets ne touche PAS data_mode.
        assert bag.data_mode_combo.currentData() == "ign_laz"
        adapter.apply_data_mode({"app": {"files": {"data_mode": "local_laz"}}})
        assert bag.data_mode_combo.currentData() == "local_laz"


# ----------------------------------------------------------------------
# collect_into : bonnes valeurs récupérées
# ----------------------------------------------------------------------
class TestCollectInto:
    def test_collects_output_dir(self):
        bag = _make_bag()
        bag.output_dir_edit.setText("/tmp/results")
        cfg: dict = {}
        ConfigWidgetAdapter(bag).collect_into(cfg)
        assert cfg["app"]["files"]["output_dir"] == "/tmp/results"

    def test_collects_data_mode_and_specific_source(self):
        bag = _make_bag()
        bag.data_mode_combo.setCurrentIndex(2)  # existing_mnt
        bag.specific_source_edit.setText("/path/mnt")
        cfg: dict = {}
        ConfigWidgetAdapter(bag).collect_into(cfg)
        assert cfg["app"]["files"]["data_mode"] == "existing_mnt"
        assert cfg["app"]["files"]["existing_mnt_dir"] == "/path/mnt"

    def test_collects_products(self):
        bag = _make_bag()
        bag.product_cbs["MNT"].setChecked(True)
        bag.product_cbs["SVF"].setChecked(True)
        cfg: dict = {}
        ConfigWidgetAdapter(bag).collect_into(cfg)
        prods = cfg["processing"]["products"]
        assert prods["MNT"] is True
        assert prods["SVF"] is True
        assert prods["DENSITE"] is False

    def test_collects_processing_floats_and_ints(self):
        bag = _make_bag()
        bag.mnt_resolution_spin.setValue(0.5)
        bag.tile_overlap_spin.setValue(15)
        bag.max_workers_spin.setValue(6)
        cfg: dict = {}
        ConfigWidgetAdapter(bag).collect_into(cfg)
        assert cfg["processing"]["mnt_resolution"] == 0.5
        assert cfg["processing"]["tile_overlap"] == 15
        assert cfg["processing"]["max_workers"] == 6


# ----------------------------------------------------------------------
# Roundtrip : config → widgets → config
# ----------------------------------------------------------------------
class TestRoundtrip:
    def test_full_roundtrip(self):
        """Une config complète appliquée puis recollectée doit être
        équivalente sur les champs gérés par l'adaptateur."""
        original = {
            "ui": {"display_mode": "expert"},
            "app": {"files": {"output_dir": "/tmp/x", "data_mode": "ign_laz", "input_file": ""}},
            "computer_vision": {
                "enabled": True,
                "confidence_threshold": 0.4,
                "iou_threshold": 0.6,
                "generate_annotated_images": True,
                "generate_shapefiles": False,
            },
            "processing": {
                "mnt_resolution": 0.25,
                "density_resolution": 1.5,
                "tile_overlap": 12,
                "max_workers": 6,
                "filter_expression": "Class=2",
                "products": {
                    "MNT": True, "DENSITE": True, "M_HS": False, "SVF": True,
                    "SLO": False, "LD": True, "SLRM": False, "VAT": True,
                },
            },
            "rvt_params": {
                "mdh": {"num_directions": 8, "sun_elevation": 30, "ve_factor": 2, "save_as_8bit": False},
                "svf": {"noise_remove": 1, "num_directions": 32, "radius": 25, "ve_factor": 1, "save_as_8bit": True},
                "slope": {"unit": 1, "ve_factor": 1, "save_as_8bit": True},
                "ldo": {"angular_res": 30, "min_radius": 5, "max_radius": 30, "observer_h": 1.5, "ve_factor": 1, "save_as_8bit": True},
                "slrm": {"radius": 15, "ve_factor": 1, "save_as_8bit": True},
                "vat": {"terrain_type": 1, "save_as_8bit": True},
            },
        }
        bag = _make_bag()
        adapter = ConfigWidgetAdapter(bag)
        adapter.apply_to_widgets(original)
        adapter.apply_data_mode(original)

        recollected: dict = {}
        adapter.collect_into(recollected)

        # Vérifier les champs gérés par l'adaptateur (pas d'invariant
        # bit-pour-bit sur les défauts CV "models_dir" qui sont posés
        # ailleurs ; on ne compare que ce que l'adaptateur écrit).
        assert recollected["ui"]["display_mode"] == "expert"
        assert recollected["app"]["files"]["output_dir"] == "/tmp/x"
        assert recollected["app"]["files"]["data_mode"] == "ign_laz"
        assert recollected["computer_vision"]["enabled"] is True
        assert recollected["computer_vision"]["confidence_threshold"] == 0.4
        assert recollected["processing"]["mnt_resolution"] == 0.25
        assert recollected["processing"]["products"]["MNT"] is True
        assert recollected["processing"]["products"]["VAT"] is True
        assert recollected["rvt_params"]["mdh"]["num_directions"] == 8
        assert recollected["rvt_params"]["svf"]["radius"] == 25
        assert recollected["rvt_params"]["slope"]["unit"] == 1
        assert recollected["rvt_params"]["vat"]["terrain_type"] == 1
