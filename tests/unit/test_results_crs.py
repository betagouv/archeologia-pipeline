"""Garantie de CRS à la frontière de production des rasters (results.py).

Régression visée : un raster sans CRS exploitable (``ENGCRS["unnamed"]`` émis par
pdal/rvt-qgis) atterrissait tel quel dans ``indices/**/tif/`` puis dans
``index.vrt`` — QGIS échouait alors à le charger (« Pas de transformation
disponible … code 4096 »). On exige désormais que ``copy_mnt_to_results``,
``copy_final_products_to_results`` et ``build_vrt_index`` ré-étiquettent
(EPSG:2154, *assignation* sans reprojection) tout TIF dépourvu de CRS, **sans
jamais écraser** un vrai CRS projeté/géographique déjà présent.
"""
from __future__ import annotations

import shutil

import pytest

from pipeline.tilespec import TileSpec
from pipeline.ign.products.results import (
    build_vrt_index,
    copy_final_products_to_results,
    copy_mnt_to_results,
)
from pipeline.ign.products.rvt_naming import get_rvt_source_and_dest_filenames

TILE = "LHD_FXX_0623_6864"


def _write_tif(path, *, crs, width=20, height=10, origin=(623000.0, 6864000.0), px=0.5):
    """Écrit un GeoTIFF minimal ; ``crs=None`` → raster sans CRS."""
    rasterio = pytest.importorskip("rasterio")
    import numpy as np
    from rasterio.transform import from_origin

    profile = dict(
        driver="GTiff", width=width, height=height, count=1, dtype="float32",
        transform=from_origin(origin[0], origin[1], px, px),
    )
    if crs is not None:
        profile["crs"] = crs
    with rasterio.open(str(path), "w", **profile) as ds:
        ds.write(np.zeros((1, height, width), dtype="float32"))


# --------------------------------------------------------------------------- #
#  copy_mnt_to_results                                                          #
# --------------------------------------------------------------------------- #

class TestCopyMntToResults:
    def test_stamps_crs_when_missing(self, tmp_path):
        _write_tif(tmp_path / "src.tif", crs=None)
        out = copy_mnt_to_results(
            temp_mnt_path=tmp_path / "src.tif",
            output_dir=tmp_path / "out",
            current_tile_name=TILE,
        )
        assert TileSpec.from_raster(out).crs == "EPSG:2154"

    def test_preserves_epsg2154(self, tmp_path):
        _write_tif(tmp_path / "src.tif", crs="EPSG:2154")
        out = copy_mnt_to_results(
            temp_mnt_path=tmp_path / "src.tif",
            output_dir=tmp_path / "out",
            current_tile_name=TILE,
        )
        assert TileSpec.from_raster(out).crs == "EPSG:2154"

    def test_does_not_clobber_other_projected_crs(self, tmp_path):
        _write_tif(tmp_path / "src.tif", crs="EPSG:32631", origin=(500000.0, 5400000.0))
        out = copy_mnt_to_results(
            temp_mnt_path=tmp_path / "src.tif",
            output_dir=tmp_path / "out",
            current_tile_name=TILE,
        )
        assert TileSpec.from_raster(out).crs == "EPSG:32631"


# --------------------------------------------------------------------------- #
#  copy_final_products_to_results                                               #
# --------------------------------------------------------------------------- #

class TestCopyFinalProducts:
    def _run(self, tmp_path, monkeypatch, *, src_crs):
        # Neutralise gdaladdo (pyramides) : non pertinent et dépendant du PATH.
        import pipeline.ign.products.results as results_mod
        monkeypatch.setattr(results_mod, "build_raster_pyramids", lambda *a, **k: True)

        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()
        out_dir = tmp_path / "out"
        # La source lue par la fonction porte le nom « dest » (cf. results.py).
        _src, dest_name = get_rvt_source_and_dest_filenames("MNT", TILE, "0623", "6864", {})
        _write_tif(temp_dir / dest_name, crs=src_crs)

        copy_final_products_to_results(
            temp_dir=temp_dir,
            output_dir=out_dir,
            current_tile_name=TILE,
            products={"MNT": True},
            output_structure={},
            output_formats={"tif": True},
            rvt_params={},
        )
        tifs = list((out_dir / "indices" / "MNT" / "tif").glob("*.tif"))
        assert len(tifs) == 1
        return tifs[0]

    def test_stamps_crs_on_copied_tif_when_missing(self, tmp_path, monkeypatch):
        out_tif = self._run(tmp_path, monkeypatch, src_crs=None)
        assert TileSpec.from_raster(out_tif).crs == "EPSG:2154"

    def test_preserves_real_crs(self, tmp_path, monkeypatch):
        out_tif = self._run(tmp_path, monkeypatch, src_crs="EPSG:32631")
        assert TileSpec.from_raster(out_tif).crs == "EPSG:32631"


# --------------------------------------------------------------------------- #
#  build_vrt_index                                                             #
# --------------------------------------------------------------------------- #

class TestBuildVrtIndex:
    def test_stamps_source_tifs_when_missing(self, tmp_path):
        folder = tmp_path / "tif"
        folder.mkdir()
        a, b = folder / "a.tif", folder / "b.tif"
        _write_tif(a, crs=None, origin=(623000.0, 6864000.0))
        _write_tif(b, crs=None, origin=(624000.0, 6864000.0))

        build_vrt_index(folder)

        assert TileSpec.from_raster(a).crs == "EPSG:2154"
        assert TileSpec.from_raster(b).crs == "EPSG:2154"

    @pytest.mark.skipif(
        shutil.which("gdalbuildvrt") is None, reason="gdalbuildvrt absent du PATH"
    )
    def test_built_vrt_inherits_epsg2154(self, tmp_path):
        folder = tmp_path / "tif"
        folder.mkdir()
        _write_tif(folder / "a.tif", crs=None, origin=(623000.0, 6864000.0))
        _write_tif(folder / "b.tif", crs=None, origin=(624000.0, 6864000.0))

        assert build_vrt_index(folder) is True
        vrt = folder / "index.vrt"
        assert vrt.exists()
        assert TileSpec.from_raster(vrt).crs == "EPSG:2154"

    def test_does_not_clobber_real_crs(self, tmp_path):
        folder = tmp_path / "tif"
        folder.mkdir()
        a = folder / "a.tif"
        _write_tif(a, crs="EPSG:32631", origin=(500000.0, 5400000.0))

        build_vrt_index(folder)

        assert TileSpec.from_raster(a).crs == "EPSG:32631"

    def test_skips_unreadable_source_gracefully(self, tmp_path):
        folder = tmp_path / "tif"
        folder.mkdir()
        good = folder / "good.tif"
        _write_tif(good, crs=None)
        (folder / "garbage.tif").write_bytes(b"not a tiff")

        # Best-effort : ne lève pas, étiquette quand même la source valide.
        build_vrt_index(folder)
        assert TileSpec.from_raster(good).crs == "EPSG:2154"
