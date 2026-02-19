from __future__ import annotations

import json
from dataclasses import dataclass
import math
from pathlib import Path
from shutil import which
import subprocess
from typing import Optional, Tuple

from .ign.pdal_validation import run_pdal_command_cancellable
from .types import CancelFn


@dataclass(frozen=True)
class CoordsResult:
    x_km: int
    y_km: int


def extract_xy_from_tile_name(tile_name: str) -> Tuple[str, str]:
    """Extrait les coordonnées (x, y) d'un nom de dalle type 'LHD_FXX_0840_6520_...'."""
    parts = tile_name.split("_")
    if len(parts) < 4:
        raise ValueError(f"Nom de dalle inattendu: {tile_name}")
    return parts[2], parts[3]


def extract_xy_from_filename(filename: str) -> Optional[CoordsResult]:
    try:
        name = Path(filename).name
        name_no_ext = Path(Path(name).stem).stem
        parts = name_no_ext.split("_")
        if len(parts) < 4:
            return None
        x = int(parts[2])
        y = int(parts[3])
        return CoordsResult(x_km=x, y_km=y)
    except Exception:
        return None


def _infer_xy_from_bounds(xmin: float, ymax: float) -> Optional[CoordsResult]:
    try:
        snap_tol_m = 5.0

        def _km_index(value_m: float) -> int:
            v = float(value_m)
            nearest_km = round(v / 1000.0)
            if abs(v - (nearest_km * 1000.0)) <= snap_tol_m:
                return int(nearest_km)
            return int(math.floor(v / 1000.0))

        x_km = _km_index(xmin)
        y_km = _km_index(ymax)
        return CoordsResult(x_km=x_km, y_km=y_km)
    except Exception:
        return None


def infer_xy_from_pdal(path: Path, *, cancel: Optional[CancelFn] = None) -> Optional[CoordsResult]:
    pdal = which("pdal")
    if not pdal:
        return None

    cmd = [pdal, "info", "--metadata", str(path)]
    try:
        r = run_pdal_command_cancellable(cmd, timeout_s=120, cancel=cancel)
    except Exception:
        return None

    if r.returncode != 0:
        return None

    try:
        parsed = json.loads(r.stdout or "")
    except Exception:
        return None

    minx = maxy = None

    def _walk(obj: object) -> None:
        nonlocal minx, maxy
        if isinstance(obj, dict):
            if {"minx", "maxy"}.issubset(obj.keys()):
                try:
                    minx = float(obj.get("minx"))
                    maxy = float(obj.get("maxy"))
                except Exception:
                    pass
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)

    _walk(parsed)

    if minx is None or maxy is None:
        return None

    return _infer_xy_from_bounds(minx, maxy)


def _get_raster_bounds(path: Path) -> Optional[Tuple[float, float, float, float]]:
    try:
        import rasterio  # type: ignore

        with rasterio.open(str(path)) as ds:
            b = ds.bounds
            return float(b.left), float(b.bottom), float(b.right), float(b.top)
    except Exception:
        pass

    try:
        from osgeo import gdal  # type: ignore

        ds = gdal.Open(str(path))
        if ds is None:
            return None
        gt = ds.GetGeoTransform()
        width = ds.RasterXSize
        height = ds.RasterYSize
        ds = None
        if not gt:
            return None

        xmin = gt[0]
        xres = gt[1]
        ymax = gt[3]
        yres = gt[5]
        xmax = xmin + width * xres
        ymin = ymax + height * yres
        return float(xmin), float(ymin), float(xmax), float(ymax)
    except Exception:
        pass

    # Fallback CLI: gdalinfo -json
    try:
        gdalinfo = which("gdalinfo")
        if not gdalinfo:
            return None

        r = subprocess.run([gdalinfo, "-json", str(path)], capture_output=True, text=True)
        if r.returncode != 0:
            return None

        parsed = json.loads(r.stdout or "{}")

        # Prefer cornerCoordinates when available
        cc = parsed.get("cornerCoordinates") if isinstance(parsed, dict) else None
        if isinstance(cc, dict):
            ul = cc.get("upperLeft")
            lr = cc.get("lowerRight")
            if isinstance(ul, (list, tuple)) and isinstance(lr, (list, tuple)) and len(ul) >= 2 and len(lr) >= 2:
                xmin = float(ul[0])
                ymax = float(ul[1])
                xmax = float(lr[0])
                ymin = float(lr[1])
                return xmin, ymin, xmax, ymax

        # Fallback: parse geoTransform + size
        gt = parsed.get("geoTransform") if isinstance(parsed, dict) else None
        size = parsed.get("size") if isinstance(parsed, dict) else None
        if isinstance(gt, (list, tuple)) and len(gt) >= 6 and isinstance(size, (list, tuple)) and len(size) >= 2:
            xmin = float(gt[0])
            xres = float(gt[1])
            ymax = float(gt[3])
            yres = float(gt[5])
            width = float(size[0])
            height = float(size[1])
            xmax = xmin + width * xres
            ymin = ymax + height * yres
            return float(xmin), float(ymin), float(xmax), float(ymax)
    except Exception:
        return None

    return None


def infer_xy_from_raster(path: Path) -> Optional[CoordsResult]:
    bounds = _get_raster_bounds(path)
    if not bounds:
        return None
    xmin, _ymin, _xmax, ymax = bounds
    return _infer_xy_from_bounds(xmin, ymax)


def infer_xy_from_world_file(jpg_path: Path) -> Optional[CoordsResult]:
    try:
        ext = jpg_path.suffix.lower()
        if ext not in {".jpg", ".jpeg"}:
            return None
        wld = jpg_path.with_suffix(".jgw")
        if not wld.exists():
            wld = jpg_path.with_suffix(".wld")
        if not wld.exists() or not wld.is_file():
            return None
        vals = [float(x.strip()) for x in wld.read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(vals) < 6:
            return None
        a = vals[0]
        e = vals[3]
        c = vals[4]
        f = vals[5]
        xmin = c - (a / 2.0)
        ymax = f - (e / 2.0)
        return _infer_xy_from_bounds(xmin, ymax)
    except Exception:
        return None


def infer_xy_from_file(path: Path, *, cancel: Optional[CancelFn] = None) -> Optional[CoordsResult]:
    ext = path.suffix.lower()

    if ext in {".laz", ".las"}:
        return infer_xy_from_pdal(path, cancel=cancel)

    if ext in {".tif", ".tiff", ".asc"}:
        return infer_xy_from_raster(path)

    if ext in {".jpg", ".jpeg"}:
        return infer_xy_from_world_file(path)

    return None
