from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple


LogFn = Callable[[str], None]


def _subprocess_kwargs_no_window() -> Dict[str, Any]:
    if os.name != "nt":
        return {}
    kwargs: Dict[str, Any] = {"creationflags": subprocess.CREATE_NO_WINDOW}
    try:
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0
        kwargs["startupinfo"] = si
    except Exception:
        pass
    return kwargs


def _extract_xy_from_tile_name(tile_name: str) -> Tuple[str, str]:
    parts = tile_name.split("_")
    if len(parts) < 4:
        raise ValueError(f"Nom de dalle inattendu: {tile_name}")
    return parts[2], parts[3]


def crop_final_products(
    *,
    temp_dir: Path,
    current_tile_name: str,
    products: Dict[str, bool],
    rvt_params: Dict[str, Any],
    log: LogFn = lambda _: None,
    gdalwarp_path: Optional[str] = None,
) -> Dict[str, Path]:
    temp_dir.mkdir(parents=True, exist_ok=True)

    x, y = _extract_xy_from_tile_name(current_tile_name)

    gdalwarp = gdalwarp_path or shutil.which("gdalwarp")  # type: ignore[name-defined]
    if not gdalwarp:
        raise FileNotFoundError("gdalwarp executable not found in PATH")

    target_xmin = int(x) * 1000
    target_xmax = (int(x) + 1) * 1000
    target_ymin = (int(y) - 1) * 1000
    target_ymax = int(y) * 1000

    xmin_r = str(target_xmin)
    ymin_r = str(target_ymin)
    xmax_r = str(target_xmax)
    ymax_r = str(target_ymax)

    vat_params = (rvt_params or {}).get("vat", {})
    terrain_type = str(vat_params.get("terrain_type", 0))
    blend_combination = str(vat_params.get("blend_combination", 0))
    preset_suffix = f"_T{terrain_type}_B{blend_combination}"

    all_files: Dict[str, Tuple[str, str]] = {
        "MNT": (f"{current_tile_name}_MNT.tif", f"LHD_FXX_{x}_{y}_MNT_A_0M50_LAMB93_IGN69.tif"),
        "M_HS": (f"{current_tile_name}_hillshade.tif", f"LHD_FXX_{x}_{y}_M-HS_A_LAMB93.tif"),
        "SVF": (f"{current_tile_name}_SVF.tif", f"LHD_FXX_{x}_{y}_SVF_A_LAMB93.tif"),
        "SLO": (f"{current_tile_name}_Slope.tif", f"LHD_FXX_{x}_{y}_SLO_A_LAMB93.tif"),
        "LD": (f"{current_tile_name}_LD.tif", f"LHD_FXX_{x}_{y}_LD_A_LAMB93.tif"),
        "SLRM": (f"{current_tile_name}_SLRM.tif", f"LHD_FXX_{x}_{y}_SLRM_A_LAMB93.tif"),
        "DENSITE": (f"{current_tile_name}_densite.tif", f"LHD_FXX_{x}_{y}_densite_A_LAMB93.tif"),
        "VAT": (f"{current_tile_name}_VAT{preset_suffix}.tif", f"LHD_FXX_{x}_{y}_VAT_A_LAMB93.tif"),
    }

    cropped: Dict[str, Path] = {}

    for product_name in ["MNT", "DENSITE", "M_HS", "SVF", "SLO", "LD", "SLRM", "VAT"]:
        if not products.get(product_name, False):
            continue

        src_name, dst_name = all_files[product_name]
        src_path = temp_dir / src_name
        dst_path = temp_dir / dst_name

        if dst_path.exists():
            cropped[product_name] = dst_path
            continue

        if not src_path.exists():
            continue

        # Compression spécifique pour MNT: LERC_ZSTD avec tolérance 1cm
        # (précision LiDAR HD: 0-10cm absolu, 0-5cm relatif)
        if product_name == "MNT":
            cmd = [
                gdalwarp,
                "-te",
                xmin_r,
                ymin_r,
                xmax_r,
                ymax_r,
                str(src_path),
                str(dst_path),
                "-of",
                "GTiff",
                "-co",
                "COMPRESS=LERC_ZSTD",
                "-co",
                "MAX_Z_ERROR=0.01",
            ]
        else:
            cmd = [
                gdalwarp,
                "-te",
                xmin_r,
                ymin_r,
                xmax_r,
                ymax_r,
                str(src_path),
                str(dst_path),
                "-of",
                "GTiff",
                "-co",
                "COMPRESS=ZSTD",
                "-co",
                "PREDICTOR=2",
            ]

        result = subprocess.run(cmd, capture_output=True, text=True, **_subprocess_kwargs_no_window())
        if result.returncode != 0:
            raise RuntimeError(f"Erreur gdalwarp ({product_name}): {result.stderr or result.stdout}")

        if dst_path.exists():
            cropped[product_name] = dst_path

    return cropped
