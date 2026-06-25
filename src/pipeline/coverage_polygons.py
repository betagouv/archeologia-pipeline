"""Extraction des polygones « zones mal couvertes » depuis la mosaïque COUVERTURE.

Lit le raster (VRT ou TIF) **par blocs** (RAM bornée quelle que soit la taille
de la zone d'étude), vectorise les cellules sous le seuil, fusionne les
polygones contigus (les coutures de blocs/dalles disparaissent) et filtre le
bruit (< ``min_area_m2``, équivalent du ``v.clean rmarea`` de PCSAPS).

Backends raster : rasterio si disponible, sinon osgeo/GDAL (fourni par QGIS).
Géométrie : shapely (requis QGIS-side). Écriture GPKG : geopandas (différé).
Aucun import QGIS. Suppose des rasters non pivotés (géotransformée sans
rotation), vrai pour tous les produits du pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

from .coverage_math import COVERAGE_NODATA

#: Nom de la couche dans le GeoPackage (consommé par ui/layer_loader).
GPKG_LAYER_NAME = "zones_mal_couvertes"

DEFAULT_MIN_AREA_M2 = 25.0
DEFAULT_BLOCK_SIZE = 2048

#: (array, géotransformée GDAL du bloc)
_Block = Tuple[Any, Tuple[float, float, float, float, float, float]]


def _iter_blocks_rasterio(
    raster_path: Path, block_size: int
) -> Optional[Iterator[_Block]]:
    """Génère ``(array, geotransform GDAL)`` par bloc. ``None`` si rasterio absent."""
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError:
        return None

    def _gen() -> Iterator[_Block]:
        with rasterio.open(raster_path) as ds:
            for row in range(0, ds.height, block_size):
                for col in range(0, ds.width, block_size):
                    win = Window(
                        col, row,
                        min(block_size, ds.width - col),
                        min(block_size, ds.height - row),
                    )
                    yield ds.read(1, window=win), ds.window_transform(win).to_gdal()

    return _gen()


def _iter_blocks_gdal(raster_path: Path, block_size: int) -> Iterator[_Block]:
    from osgeo import gdal

    ds = gdal.Open(str(raster_path))
    if ds is None:
        raise IOError(f"Raster couverture illisible: {raster_path}")
    band = ds.GetRasterBand(1)
    gt = ds.GetGeoTransform()
    for row in range(0, ds.RasterYSize, block_size):
        for col in range(0, ds.RasterXSize, block_size):
            w = min(block_size, ds.RasterXSize - col)
            h = min(block_size, ds.RasterYSize - row)
            arr = band.ReadAsArray(col, row, w, h)
            block_gt = (
                gt[0] + col * gt[1], gt[1], gt[2],
                gt[3] + row * gt[5], gt[4], gt[5],
            )
            yield arr, block_gt
    ds = None


def _shapes_rasterio(mask, transform_gdal) -> Optional[List[Any]]:
    """Polygones shapely des pixels à 1 du masque. ``None`` si rasterio absent."""
    try:
        from rasterio import features
        from rasterio.transform import Affine
    except ImportError:
        return None
    from shapely.geometry import shape

    t = Affine.from_gdal(*transform_gdal)
    return [
        shape(geom)
        for geom, val in features.shapes(mask, mask=mask.astype(bool), transform=t)
        if val == 1
    ]


def _shapes_gdal(mask, transform_gdal) -> List[Any]:
    import shapely.wkb
    from osgeo import gdal, ogr

    h, w = mask.shape
    mem = gdal.GetDriverByName("MEM").Create("", int(w), int(h), 1, gdal.GDT_Byte)
    mem.SetGeoTransform(transform_gdal)
    band = mem.GetRasterBand(1)
    band.WriteArray(mask)
    drv = ogr.GetDriverByName("Memory")
    vds = drv.CreateDataSource("mem")
    layer = vds.CreateLayer("polys", srs=None)
    layer.CreateField(ogr.FieldDefn("val", ogr.OFTInteger))
    # maskBand = band : les pixels à 0 sont exclus, seuls les 1 sont vectorisés.
    gdal.Polygonize(band, band, layer, 0)
    geoms: List[Any] = []
    for feat in layer:
        g = feat.GetGeometryRef()
        if g is not None and feat.GetField("val") == 1:
            geoms.append(shapely.wkb.loads(bytes(g.ExportToWkb())))
    vds = None
    mem = None
    return geoms


def extract_low_coverage_polygons(
    raster_path: Path,
    threshold_percent: float,
    *,
    nodata: int = COVERAGE_NODATA,
    min_area_m2: float = DEFAULT_MIN_AREA_M2,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> List[Any]:
    """Polygones shapely des zones « < threshold_percent » (hors NoData).

    Fusionnés (``unary_union`` : les morceaux partageant une arête — y compris
    aux coutures de blocs — deviennent un seul polygone) puis filtrés par
    surface minimale et triés par surface décroissante.
    """
    import numpy as np
    from shapely.ops import unary_union

    blocks = _iter_blocks_rasterio(raster_path, block_size)
    if blocks is None:
        blocks = _iter_blocks_gdal(raster_path, block_size)

    raw: List[Any] = []
    for arr, gt in blocks:
        mask = ((arr < float(threshold_percent)) & (arr != nodata)).astype(np.uint8)
        if not mask.any():
            continue
        shapes = _shapes_rasterio(mask, gt)
        if shapes is None:
            shapes = _shapes_gdal(mask, gt)
        raw.extend(shapes)

    if not raw:
        return []
    merged = unary_union(raw)
    parts = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    parts = [g for g in parts if g.area >= float(min_area_m2)]
    parts.sort(key=lambda g: g.area, reverse=True)
    return parts


def write_low_coverage_gpkg(
    polygons: List[Any],
    gpkg_path: Path,
    *,
    layer_name: str = GPKG_LAYER_NAME,
    crs: str = "EPSG:2154",
) -> Optional[Path]:
    """Écrit les polygones en GeoPackage (geopandas, import différé).

    ``None`` si la liste est vide (pas de fichier créé).
    """
    if not polygons:
        return None
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        {"area_m2": [round(float(g.area), 1) for g in polygons]},
        geometry=list(polygons),
        crs=crs,
    )
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(gpkg_path), layer=layer_name, driver="GPKG")
    return gpkg_path
