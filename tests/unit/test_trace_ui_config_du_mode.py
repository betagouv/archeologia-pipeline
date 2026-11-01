"""La trace complète (``metadata_<ts>.json``) ne garde que les réglages qui s'appliquent au mode (crash tests du
2026-11-01 : un run « MNT existants » traçait le dossier d'indices d'un ancien run fénétrange, la résolution de
densité et le filtre LAS du LiDAR, le seuil de couverture et les réglages de 9 indices non calculés)."""
from __future__ import annotations

from app.services.finalize_service import _ui_config_trace

PRODUITS = {"MNT": True, "DENSITE": False, "COUVERTURE": False, "LD": True, "VAT": True, "CVAT": True,
            "SVF": False, "HS": False}
RVT = {"ldo": {"min_radius_m": 5}, "vat": {"t": 0}, "cvat": {}, "svf": {"radius_m": 5}, "hs": {"az": 315},
       "inconnue": {"k": 1}}


def _ui(mode, **processing):
    return {
        "app": {"files": {"output_dir": "o", "data_mode": mode, "input_file": "zone.shp", "local_laz_dir": "laz",
                          "existing_mnt_dir": "mnt", "existing_rvt_dir": "D:/ancien/run/tif_test"}},
        "processing": {"mnt_resolution": 0.5, "density_resolution": 1.0, "filter_expression": "Classification = 2",
                       "coverage_threshold_percent": 30, "tile_overlap": 0, "max_workers": 2,
                       "products": dict(PRODUITS), **processing},
        "rvt_params": {k: dict(v) for k, v in RVT.items()},
        "computer_vision": {"enabled": True},
    }


def test_mnt_existant_ne_trace_que_ses_reglages():
    ui = _ui("existing_mnt")
    t = _ui_config_trace(ui)
    assert t["app"]["files"] == {"output_dir": "o", "data_mode": "existing_mnt", "existing_mnt_dir": "mnt"}
    assert set(t["processing"]) == {"tile_overlap", "max_workers", "products"}
    assert set(t["rvt_params"]) == {"ldo", "vat", "cvat", "inconnue"}     # section inconnue : gardée
    assert t["computer_vision"] == {"enabled": True}
    assert ui["app"]["files"]["existing_rvt_dir"] == "D:/ancien/run/tif_test"   # l'appelant n'est pas touché


def test_lidar_garde_ses_reglages_selon_les_produits():
    t = _ui_config_trace(_ui("ign_laz"))
    assert t["app"]["files"] == {"output_dir": "o", "data_mode": "ign_laz", "input_file": "zone.shp"}
    assert set(t["processing"]) == {"mnt_resolution", "filter_expression", "tile_overlap", "max_workers", "products"}
    t = _ui_config_trace(_ui("local_laz", products={**PRODUITS, "DENSITE": True, "COUVERTURE": True}))
    assert t["app"]["files"]["local_laz_dir"] == "laz" and "input_file" not in t["app"]["files"]
    assert {"density_resolution", "coverage_threshold_percent"} <= set(t["processing"])


def test_mode_inconnu_trace_tout():
    ui = _ui("")
    assert _ui_config_trace(ui)["app"] == ui["app"]
