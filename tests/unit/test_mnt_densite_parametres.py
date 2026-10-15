"""Lot 0e-a (2026-10-15) : BOUNDS n'est pas un paramètre de pdal:exportrastertin
ni de pdal:density (QGIS 3.44.1 et 4.0.3 : FILTER_EXTENT) — Processing l'ignorait.
Retiré avec le calcul d'emprise qui ne servait qu'à lui : les sorties ne changent pas."""
from __future__ import annotations

from pathlib import Path

import pipeline.ign.pdal_validation as pdal_validation
import pipeline.ign.products.density as density_mod
import pipeline.ign.products.mnt as mnt_mod

TILE = "LHD_FXX_0500_6500"


def _laz(tmp_path: Path) -> Path:
    p = tmp_path / f"{TILE}_merged.laz"
    p.write_bytes(b"laz")
    return p


def _algo(recus: list):
    def algo(algo_id, params, **_k):
        recus.append((algo_id, dict(params)))
        Path(params["OUTPUT"]).write_bytes(b"tif")
    return algo


def test_mnt_n_envoie_que_des_parametres_de_l_algorithme(tmp_path, monkeypatch):
    recus: list = []
    monkeypatch.setattr(mnt_mod, "validate_las_or_laz_with_pdal", lambda p: (True, ""))
    monkeypatch.setattr(mnt_mod, "run_qgis_algorithm", _algo(recus))
    monkeypatch.setattr(mnt_mod, "assign_crs_if_missing", lambda p: None)
    laz = _laz(tmp_path)
    mnt_mod.create_terrain_model(
        input_laz_path=laz, temp_dir=tmp_path / "t", current_tile_name=TILE,
        mnt_resolution=0.5, tile_overlap_percent=20.0, filter_expression="Classification = 2",
    )
    assert recus == [("pdal:exportrastertin", {
        "INPUT": str(laz), "OUTPUT": str(tmp_path / "t" / f"{TILE}_MNT.tif"),
        "RESOLUTION": "0.5", "FILTER_EXPRESSION": "Classification = 2",
    })]


def test_densite_sans_bounds_ni_lecture_d_emprise(tmp_path, monkeypatch):
    recus: list = []
    monkeypatch.setattr(density_mod, "validate_las_or_laz_with_pdal", lambda p: (True, ""))
    monkeypatch.setattr(density_mod, "run_qgis_algorithm", _algo(recus))
    laz = _laz(tmp_path)
    density_mod.create_density_map(
        input_laz_path=laz, temp_dir=tmp_path / "t", current_tile_name=TILE,
        density_resolution=1.0, tile_overlap_percent=20.0, filter_expression="Classification = 2",
    )
    assert recus == [("pdal:density", {
        "INPUT": str(laz), "OUTPUT": str(tmp_path / "t" / f"{TILE}_densite.tif"),
        "RESOLUTION": "1.0", "FILTER_EXPRESSION": "Classification = 2",
    })]


def test_plus_de_lecture_d_emprise_pdal():
    # get_laz_bounds lançait un `pdal info --metadata` par dalle pour rien.
    assert not hasattr(pdal_validation, "get_laz_bounds")
    assert not hasattr(mnt_mod, "get_laz_bounds") and not hasattr(density_mod, "get_laz_bounds")
