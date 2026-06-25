"""Tests de l'extraction des polygones « zones mal couvertes »."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("shapely")

from pipeline.coverage_polygons import (  # noqa: E402
    GPKG_LAYER_NAME,
    extract_low_coverage_polygons,
    write_low_coverage_gpkg,
)


def _write_coverage(path: Path, arr: np.ndarray) -> None:
    from rasterio.transform import from_origin

    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype="uint8", nodata=255,
        transform=from_origin(0.0, float(arr.shape[0]), 1.0, 1.0), crs="EPSG:2154",
    ) as ds:
        ds.write(arr, 1)


@pytest.fixture()
def coverage_tif(tmp_path: Path) -> Path:
    arr = np.full((60, 100), 80, dtype=np.uint8)   # fond bien couvert
    arr[10:20, 10:20] = 0      # zone mal couverte 10x10 = 100 m²
    arr[40:42, 40:42] = 5      # zone 2x2 = 4 m² (sous min_area 25)
    arr[50:55, 80:90] = 255    # nodata (hors zone d'étude)
    p = tmp_path / "index_cov.tif"
    _write_coverage(p, arr)
    return p


class TestExtract:
    def test_extrait_la_zone_et_filtre_le_bruit(self, coverage_tif: Path):
        polys = extract_low_coverage_polygons(coverage_tif, 30.0)
        assert len(polys) == 1
        assert polys[0].area == pytest.approx(100.0)

    def test_le_nodata_n_est_pas_extrait(self, coverage_tif: Path):
        # nodata=255 : exclu même avec un seuil très haut.
        polys = extract_low_coverage_polygons(coverage_tif, 90.0)
        # fond=80 < 90 → tout le raster valide sort, mais PAS la zone 255.
        total = sum(p.area for p in polys)
        assert total == pytest.approx(60 * 100 - 5 * 10)

    def test_seuil_respecte(self, coverage_tif: Path):
        polys_30 = extract_low_coverage_polygons(coverage_tif, 30.0)
        assert sum(p.area for p in polys_30) == pytest.approx(100.0)
        polys_40 = extract_low_coverage_polygons(coverage_tif, 40.0)
        assert sum(p.area for p in polys_40) == pytest.approx(100.0)  # 4 m² filtré

    def test_fusion_a_travers_les_blocs(self, coverage_tif: Path):
        # block_size=16 force la zone 10x10 (lignes 10-20) à chevaucher 2 blocs.
        polys = extract_low_coverage_polygons(coverage_tif, 30.0, block_size=16)
        assert len(polys) == 1
        assert polys[0].area == pytest.approx(100.0)

    def test_aucune_zone(self, tmp_path: Path):
        arr = np.full((10, 10), 90, dtype=np.uint8)
        p = tmp_path / "full.tif"
        _write_coverage(p, arr)
        assert extract_low_coverage_polygons(p, 30.0) == []


class TestWriteGpkg:
    def test_ecrit_le_gpkg(self, coverage_tif: Path, tmp_path: Path):
        gpd = pytest.importorskip("geopandas")
        polys = extract_low_coverage_polygons(coverage_tif, 30.0)
        out = write_low_coverage_gpkg(polys, tmp_path / "zones.gpkg")
        assert out is not None and out.exists()
        gdf = gpd.read_file(out, layer=GPKG_LAYER_NAME)
        assert len(gdf) == 1
        assert float(gdf["area_m2"].iloc[0]) == pytest.approx(100.0)

    def test_liste_vide_renvoie_none(self, tmp_path: Path):
        assert write_low_coverage_gpkg([], tmp_path / "zones.gpkg") is None
