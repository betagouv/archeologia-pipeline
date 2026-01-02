from __future__ import annotations

import os
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..coords import extract_xy_from_filename, infer_xy_from_file
from ..ign.products.crop import crop_final_products
from ..ign.products.indices import create_visualization_products
from ..ign.products.results import copy_final_products_to_results


LogFn = Callable[[str], None]


@dataclass(frozen=True)
class ExistingMntResult:
    total: int


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


def _infer_tile_coords_from_mnt(mnt_path: Path, log: LogFn) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    inferred = infer_xy_from_file(mnt_path)
    if inferred is not None:
        x_str = f"{int(inferred.x_km):04d}"
        y_str = f"{int(inferred.y_km):04d}"
        tile_name = f"MNT_EXT_{x_str}_{y_str}"
        log(f"Coordonnées MNT déduites via metadata: x={x_str}, y={y_str} (tile={tile_name})")
        return x_str, y_str, tile_name

    xy = extract_xy_from_filename(mnt_path.name)
    if xy is not None:
        x_str = f"{int(xy.x_km):04d}"
        y_str = f"{int(xy.y_km):04d}"
        tile_name = mnt_path.stem
        log(f"Coordonnées MNT déduites du nom de fichier (fallback): x={x_str}, y={y_str}")
        return x_str, y_str, tile_name

    log(f"Impossible de déduire les coordonnées via metadata et nom: {mnt_path.stem}")
    return None, None, None


def run_existing_mnt(
    *,
    existing_mnt_dir: Path,
    output_dir: Path,
    products: Dict[str, bool],
    output_structure: Dict[str, Any],
    output_formats: Dict[str, Any],
    rvt_params: Dict[str, Any],
    log: LogFn = lambda _: None,
) -> ExistingMntResult:
    if not existing_mnt_dir.exists() or not existing_mnt_dir.is_dir():
        raise FileNotFoundError(f"Dossier MNT inexistant ou invalide: {existing_mnt_dir}")

    mnt_files = list(existing_mnt_dir.glob("*.tif")) + list(existing_mnt_dir.glob("*.tiff")) + list(existing_mnt_dir.glob("*.asc"))
    if not mnt_files:
        raise FileNotFoundError(f"Aucun fichier MNT (*.tif, *.tiff, *.asc) trouvé dans {existing_mnt_dir}")

    temp_dir = output_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    gdal_translate = shutil.which("gdal_translate")

    for mnt_path in mnt_files:
        x_str, y_str, tile_name = _infer_tile_coords_from_mnt(mnt_path, log)
        if not x_str or not y_str or not tile_name:
            raise ValueError(f"Impossible de déduire les coordonnées pour le MNT: {mnt_path.name}")

        current_tile_name = tile_name
        temp_mnt_path = temp_dir / f"{current_tile_name}_MNT.tif"

        if not temp_mnt_path.exists():
            if mnt_path.suffix.lower() in (".tif", ".tiff"):
                shutil.copy2(str(mnt_path), str(temp_mnt_path))
            elif mnt_path.suffix.lower() == ".asc":
                if not gdal_translate:
                    raise FileNotFoundError("gdal_translate executable not found in PATH")
                cmd = [gdal_translate, str(mnt_path), str(temp_mnt_path)]
                r = subprocess.run(cmd, capture_output=True, text=True, **_subprocess_kwargs_no_window())
                if r.returncode != 0:
                    raise RuntimeError(r.stderr or r.stdout)
            else:
                raise ValueError(f"Format MNT non supporté: {mnt_path.suffix}")

        create_visualization_products(
            temp_dir=temp_dir,
            current_tile_name=current_tile_name,
            products=products,
            rvt_params=rvt_params,
            log=log,
        )

        crop_final_products(
            temp_dir=temp_dir,
            current_tile_name=current_tile_name,
            products=products,
            rvt_params=rvt_params,
            log=log,
        )

        copy_final_products_to_results(
            temp_dir=temp_dir,
            output_dir=output_dir,
            current_tile_name=current_tile_name,
            products=products,
            output_structure=output_structure,
            output_formats=output_formats,
            rvt_params=rvt_params,
            log=log,
        )

    return ExistingMntResult(total=len(mnt_files))
