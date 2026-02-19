"""
Utilitaires géospatiaux partagés : extraction de géotransform et création de world files.

Unifie la logique dupliquée entre cv/runner.py et ign/products/convert_tif_to_jpg.py.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def extract_tif_geotransform(
    tif_path: Path,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Extrait les 6 paramètres de géotransform d'un fichier TIF.

    Essaie dans l'ordre : rasterio → GDAL bindings → gdalinfo CLI.

    Returns:
        (pixel_width, row_rotation, x_origin, col_rotation, pixel_height, y_origin)
        ou un tuple de None si échec.
    """
    none6 = (None, None, None, None, None, None)

    # 1. rasterio
    try:
        import rasterio  # type: ignore

        with rasterio.open(str(tif_path)) as ds:
            t = ds.transform
            return (
                float(t.a),   # pixel_width
                float(t.b),   # row_rotation
                float(t.c),   # x_origin
                float(t.d),   # col_rotation
                float(t.e),   # pixel_height
                float(t.f),   # y_origin
            )
    except Exception:
        pass

    # 2. GDAL bindings
    try:
        from osgeo import gdal  # type: ignore

        ds = gdal.Open(str(tif_path))
        if ds is not None:
            gt = ds.GetGeoTransform()
            ds = None
            if gt and gt != (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
                # GDAL order: (x_origin, pixel_width, row_rotation, y_origin, col_rotation, pixel_height)
                # Reorder to: (pixel_width, row_rotation, x_origin, col_rotation, pixel_height, y_origin)
                return (float(gt[1]), float(gt[2]), float(gt[0]), float(gt[4]), float(gt[5]), float(gt[3]))
    except Exception:
        pass

    # 3. gdalinfo CLI
    try:
        import json
        import shutil
        import subprocess

        gdalinfo = shutil.which("gdalinfo")
        if gdalinfo:
            r = subprocess.run(
                [gdalinfo, "-json", str(tif_path)],
                capture_output=True,
                text=True,
            )
            if r.returncode == 0:
                parsed = json.loads(r.stdout or "{}")
                gt = parsed.get("geoTransform") if isinstance(parsed, dict) else None
                if isinstance(gt, (list, tuple)) and len(gt) >= 6:
                    # GDAL order: (x_origin, pixel_width, row_rotation, y_origin, col_rotation, pixel_height)
                    # Reorder to: (pixel_width, row_rotation, x_origin, col_rotation, pixel_height, y_origin)
                    return (float(gt[1]), float(gt[2]), float(gt[0]), float(gt[4]), float(gt[5]), float(gt[3]))
    except Exception:
        pass

    return none6


def extract_tif_transform_data(
    tif_path: Path,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Raccourci : extrait uniquement (pixel_width, pixel_height, x_origin, y_origin).

    Compatible avec l'ancien ``extract_tif_transform_data`` de ``cv/runner.py``.
    """
    pixel_width, _row_rot, x_origin, _col_rot, pixel_height, y_origin = extract_tif_geotransform(tif_path)
    return pixel_width, pixel_height, x_origin, y_origin


def write_world_file(
    image_path: Path,
    pixel_width: float,
    pixel_height: float,
    x_origin: float,
    y_origin: float,
    row_rotation: float = 0.0,
    col_rotation: float = 0.0,
) -> Optional[Path]:
    """
    Crée un fichier world (.jgw pour JPEG, .pgw pour PNG, .tfw pour TIFF).
    """
    ext_map = {
        ".jpg": ".jgw",
        ".jpeg": ".jgw",
        ".png": ".pgw",
        ".tif": ".tfw",
        ".tiff": ".tfw",
    }
    world_ext = ext_map.get(image_path.suffix.lower())
    if not world_ext:
        return None

    # Clamp negligible rotation values to zero to avoid gdalbuildvrt
    # rejecting files with "rotated geo transforms" due to floating-point noise.
    if abs(row_rotation) < 1e-10:
        row_rotation = 0.0
    if abs(col_rotation) < 1e-10:
        col_rotation = 0.0

    world_path = image_path.with_suffix(world_ext)
    with open(world_path, "w") as f:
        f.write(f"{pixel_width:.10f}\n")
        f.write(f"{row_rotation:.10f}\n")
        f.write(f"{col_rotation:.10f}\n")
        f.write(f"{pixel_height:.10f}\n")
        f.write(f"{x_origin:.10f}\n")
        f.write(f"{y_origin:.10f}\n")
    return world_path


def create_world_file_from_tif(input_tif_path: Path, output_image_path: Path) -> bool:
    """
    Crée un fichier world pour ``output_image_path`` en lisant le géoréférencement
    de ``input_tif_path``.

    Compatible avec l'ancien ``create_world_file_from_tif`` de ``convert_tif_to_jpg.py``.
    """
    pw, _rr, xo, _cr, ph, yo = extract_tif_geotransform(input_tif_path)
    if pw is None:
        return False
    result = write_world_file(
        output_image_path,
        pixel_width=pw,
        pixel_height=ph,
        x_origin=xo,
        y_origin=yo,
        row_rotation=_rr or 0.0,
        col_rotation=_cr or 0.0,
    )
    return result is not None
