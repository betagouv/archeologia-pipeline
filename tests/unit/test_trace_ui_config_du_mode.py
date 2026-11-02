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


def test_rvt_params_racine_ne_garde_que_les_produits_calcules(tmp_path, monkeypatch):
    """Relance du crash test de_sh (2026-11-02) : ``rvt_params`` à la RACINE de traitement.json et de
    metadata_*.json gardait les réglages des 13 indices alors que LD, VAT et CVAT seuls étaient calculés."""
    import json
    import time

    from app.services import finalize_service

    monkeypatch.setattr(finalize_service, "_collect_vrt_paths_and_build", lambda *a, **k: [])
    monkeypatch.setattr(finalize_service, "_build_coverage_polygons", lambda *a, **k: None)

    class _R:
        def __getattr__(self, _n):
            return lambda *a, **k: None

    out = tmp_path / "sortie"
    out.mkdir()
    rvt = {**{k: dict(v) for k, v in RVT.items()}, "city": {"terrain_type": 1}, "resolution_m": 1.0}
    finalize_service.finalize_pipeline(output_dir=out, cv_cfg={}, rvt_params=rvt, reporter=_R(),
                                       start_time=time.time(), tiles_processed=1,
                                       active_products=["MNT", "LD", "VAT", "CVAT"], ui_config=_ui("existing_mnt"))
    attendu = {"ldo", "vat", "cvat", "inconnue", "resolution_m"}
    trace = json.loads((out / "livrable" / "traitement.json").read_text(encoding="utf-8"))
    assert set(trace["rvt_params"]) == attendu
    assert trace["resolution_m"] == 1.0
    meta = json.loads(next((out / "technique" / "journaux").glob("metadata_*.json")).read_text(encoding="utf-8"))
    assert set(meta["rvt_params"]) == attendu
    assert set(rvt) > attendu                              # l'appelant n'est pas touché
