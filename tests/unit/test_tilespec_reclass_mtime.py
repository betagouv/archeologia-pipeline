"""Idempotence mtime de ``reclass_rvt_nodata`` (revue adversariale v2, 2026-09-02).

La boucle finale de ``create_visualization_products`` applique la
reclassification à TOUTES les sorties, y compris celles servies du cache. Si
elle réécrit un raster déjà reclassé, son mtime « rajeunit » à chaque run →
``needs_refresh`` régénérerait les PNG publiés et ``purge_stale_cached_detections``
jetterait le cache d'inférence CV à chaque reprise à paramètres identiques.
Contrat : déjà reclassé + étiqueté → aucune écriture, mtime intact.
"""
from __future__ import annotations

import os

import pytest

from pipeline.tilespec import RVT_BYTE_NODATA, reclass_rvt_nodata


def _write_rasters(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    import numpy as np
    from rasterio.transform import from_origin

    transform = from_origin(623000.0, 6864000.0, 0.5, 0.5)
    dem = tmp_path / "dem.tif"
    with rasterio.open(
        str(dem), "w", driver="GTiff", width=8, height=8, count=1,
        dtype="float32", nodata=-9999.0, transform=transform, crs="EPSG:2154",
    ) as ds:
        arr = np.full((8, 8), 100.0, dtype="float32")
        arr[0, :] = -9999.0  # une ligne de NoData MNT
        ds.write(arr, 1)
    rvt = tmp_path / "rvt.tif"
    with rasterio.open(
        str(rvt), "w", driver="GTiff", width=8, height=8, count=1,
        dtype="uint8", transform=transform, crs="EPSG:2154",
    ) as ds:
        arr = np.full((8, 8), 128, dtype="uint8")
        arr[0, :] = 255  # NoData rvt-qgis sur la ligne NoData du MNT
        ds.write(arr, 1)
    return rvt, dem


class TestReclassMtimeIdempotence:
    def test_second_call_does_not_rewrite(self, tmp_path):
        rvt, dem = _write_rasters(tmp_path)
        assert reclass_rvt_nodata(rvt, dem) is True  # 1er appel : écrit + étiquette

        old = 1_600_000_000.0
        os.utime(rvt, (old, old))
        assert reclass_rvt_nodata(rvt, dem) is True  # 2e appel : no-op
        assert rvt.stat().st_mtime == pytest.approx(old)

    def test_first_call_tags_nodata(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        rvt, dem = _write_rasters(tmp_path)
        reclass_rvt_nodata(rvt, dem)
        with rasterio.open(str(rvt)) as ds:
            assert ds.nodata == RVT_BYTE_NODATA
