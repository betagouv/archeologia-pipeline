"""Produit COUVERTURE : % de couverture locale des points sol, par dalle.

Wrapper IO autour de :mod:`pipeline.coverage_math` : lit le raster densité de
la dalle (sortie ``pdal:density``), calcule le % de couverture (disque de
5 cellules) et écrit ``<dalle>_couverture.tif`` (uint8, NoData 255) dans le
dossier temporaire, même géoréférencement que la densité.

Backends IO : rasterio si disponible, sinon osgeo/GDAL (fourni par QGIS) —
même stratégie de repli que ``pipeline.tilespec``. Aucun import QGIS.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

from ...types import LogFn
from .rvt_naming import get_rvt_temp_filename

#: (array, nodata, géotransformée GDAL, crs_wkt)
_RasterData = Tuple[Any, Optional[float], Tuple[float, ...], Optional[str]]


@dataclass(frozen=True)
class CoverageResult:
    coverage_path: Path


def _read_density_rasterio(path: Path) -> Optional[_RasterData]:
    """Lecture via rasterio ; ``None`` si rasterio n'est pas installé."""
    try:
        import rasterio
    except ImportError:
        return None
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        nodata = None if ds.nodata is None else float(ds.nodata)
        transform = ds.transform.to_gdal()
        crs_wkt = ds.crs.to_wkt() if ds.crs else None
    return arr, nodata, transform, crs_wkt


def _read_density_gdal(path: Path) -> _RasterData:
    from osgeo import gdal

    ds = gdal.Open(str(path))
    if ds is None:
        raise IOError(f"Raster densité illisible: {path}")
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    nodata = band.GetNoDataValue()
    nodata = None if nodata is None else float(nodata)
    transform = ds.GetGeoTransform()
    crs_wkt = ds.GetProjection() or None
    ds = None
    return arr, nodata, transform, crs_wkt


def _write_coverage_rasterio(path: Path, arr, transform_gdal, crs_wkt) -> bool:
    """Écriture via rasterio ; ``False`` si rasterio n'est pas installé."""
    try:
        import rasterio
        from rasterio.transform import Affine
    except ImportError:
        return False
    profile = {
        "driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1],
        "count": 1, "dtype": "uint8", "nodata": 255,
        "transform": Affine.from_gdal(*transform_gdal), "compress": "deflate",
    }
    if crs_wkt:
        profile["crs"] = crs_wkt
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(arr, 1)
    return True


def _write_coverage_gdal(path: Path, arr, transform_gdal, crs_wkt) -> None:
    from osgeo import gdal

    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        str(path), int(arr.shape[1]), int(arr.shape[0]), 1, gdal.GDT_Byte,
        options=["COMPRESS=DEFLATE"],
    )
    ds.SetGeoTransform(transform_gdal)
    if crs_wkt:
        ds.SetProjection(crs_wkt)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(255)
    band.WriteArray(arr)
    band.FlushCache()
    ds = None


def create_coverage_map(
    *,
    density_path: Path,
    temp_dir: Path,
    current_tile_name: str,
    log: LogFn = lambda _: None,
) -> CoverageResult:
    """Calcule le raster COUVERTURE de la dalle depuis son raster densité.

    Idempotent (ne refait pas un TIF existant), comme ``create_density_map``.
    """
    output_path = temp_dir / get_rvt_temp_filename("COUVERTURE", current_tile_name, {})
    if output_path.exists():
        return CoverageResult(coverage_path=output_path)
    if not density_path.exists():
        raise FileNotFoundError(
            f"Raster densité introuvable pour la couverture: {density_path}"
        )

    data = _read_density_rasterio(density_path)
    if data is None:
        data = _read_density_gdal(density_path)
    arr, nodata, transform, crs_wkt = data

    from ...coverage_math import compute_coverage_percent

    coverage = compute_coverage_percent(arr, density_nodata=nodata)

    temp_dir.mkdir(parents=True, exist_ok=True)
    if not _write_coverage_rasterio(output_path, coverage, transform, crs_wkt):
        _write_coverage_gdal(output_path, coverage, transform, crs_wkt)
    log(f"Couverture calculée: {output_path.name}")
    return CoverageResult(coverage_path=output_path)
