"""Robustesse géoréférencement : une dalle MNT « dégénérée » (placeholder 1×1 px /
posée à l'origine du CRS) doit être ignorée AVANT traitement — sinon elle produit
une sortie 1×1 à (0,0) qui gonfle l'emprise du VRT de mosaïque (couches QGIS vides).

La boucle d'isolation est stubée pour appeler réellement ``_process_one_mnt`` ;
``TileSpec.from_raster`` est monkeypatché (pas de lecture GDAL réelle).
"""
from __future__ import annotations

from pathlib import Path

import pipeline.modes.existing_mnt as em
from pipeline.modes.existing_mnt import run_existing_mnt
from pipeline.tilespec import TileSpec


def _degenerate_spec(path):
    return TileSpec.from_values(
        source_path=Path(path),
        bounds=(-0.5, -0.5, 0.5, 0.5),
        pixel_size_x=1.0, pixel_size_y=-1.0,
        width_px=1, height_px=1, crs="EPSG:2154",
    )


def _fake_isolated_calls_process(items, process, *, cancel=None, on_failure=None):
    for i, it in enumerate(items, start=1):
        process(i, it)
    return 0, []


def test_degenerate_tile_skipped_and_reported(tmp_path, monkeypatch):
    mnt_dir = tmp_path / "mnt"
    mnt_dir.mkdir()
    (mnt_dir / "placeholder.tif").write_bytes(b"x")  # jamais lu : from_raster est stubé

    monkeypatch.setattr(
        "pipeline.batch.process_items_isolated", _fake_isolated_calls_process
    )
    monkeypatch.setattr(
        em.TileSpec, "from_raster",
        staticmethod(lambda p, **k: _degenerate_spec(p)),
    )

    logs: list[str] = []
    errors: list[str] = []
    res = run_existing_mnt(
        existing_mnt_dir=mnt_dir,
        output_dir=tmp_path / "out",
        products={}, output_structure={}, output_formats={}, rvt_params={},
        log=logs.append, error_log=errors.append,
    )

    assert res.total == 0  # dalle dégénérée non comptée comme produite
    assert any("placeholder.tif" in e and "dégénér" in e for e in errors)
