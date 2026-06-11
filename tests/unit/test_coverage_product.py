"""Tests du wrapper produit COUVERTURE (lecture densité → écriture couverture)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from pipeline.ign.products.coverage import CoverageResult, create_coverage_map  # noqa: E402

TILE = "LHD_FXX_0624_6864"
NODATA = -9999.0


def _write_density(path: Path, arr: np.ndarray) -> None:
    from rasterio.transform import from_origin

    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype="float32", nodata=NODATA,
        transform=from_origin(624000.0, 6864000.0, 1.0, 1.0), crs="EPSG:2154",
    ) as ds:
        ds.write(arr.astype("float32"), 1)


@pytest.fixture()
def density_tif(tmp_path: Path) -> Path:
    arr = np.full((20, 20), 4.0)        # zone bien couverte
    arr[0:8, 0:8] = NODATA              # zone sans points (nodata PDAL)
    p = tmp_path / f"{TILE}_densite.tif"
    _write_density(p, arr)
    return p


class TestCreateCoverageMap:
    def test_cree_le_tif_attendu(self, density_tif: Path, tmp_path: Path):
        res = create_coverage_map(
            density_path=density_tif, temp_dir=tmp_path, current_tile_name=TILE
        )
        assert isinstance(res, CoverageResult)
        assert res.coverage_path == tmp_path / f"{TILE}_couverture.tif"
        assert res.coverage_path.exists()

    def test_contenu_et_georeferencement(self, density_tif: Path, tmp_path: Path):
        res = create_coverage_map(
            density_path=density_tif, temp_dir=tmp_path, current_tile_name=TILE
        )
        with rasterio.open(res.coverage_path) as ds, rasterio.open(density_tif) as src:
            assert ds.dtypes[0] == "uint8"
            assert ds.nodata == 255
            assert ds.transform == src.transform
            assert ds.crs == src.crs
            out = ds.read(1)
        assert out[15, 15] == 100   # cœur de la zone couverte
        assert out[2, 2] == 0       # cœur de la zone nodata → 0 % (pas 255)

    def test_idempotent(self, density_tif: Path, tmp_path: Path):
        res1 = create_coverage_map(
            density_path=density_tif, temp_dir=tmp_path, current_tile_name=TILE
        )
        stamp = res1.coverage_path.stat().st_mtime_ns
        res2 = create_coverage_map(
            density_path=density_tif, temp_dir=tmp_path, current_tile_name=TILE
        )
        assert res2.coverage_path.stat().st_mtime_ns == stamp

    def test_densite_absente_leve(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            create_coverage_map(
                density_path=tmp_path / "absent.tif", temp_dir=tmp_path,
                current_tile_name=TILE,
            )
