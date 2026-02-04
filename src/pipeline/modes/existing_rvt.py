from __future__ import annotations

import os
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

from ..cv.runner import extract_tif_transform_data
from ..coords import extract_xy_from_filename, infer_xy_from_file


LogFn = Callable[[str], None]
CancelCheckFn = Callable[[], bool]


@dataclass(frozen=True)
class ExistingRvtResult:
    total_images: int


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


def _convert_tif_to_jpg_with_world(input_tif: Path, output_jpg: Path) -> None:
    from ..ign.products.convert_tif_to_jpg import convert_tif_to_jpg

    output_jpg.parent.mkdir(parents=True, exist_ok=True)
    ok = bool(convert_tif_to_jpg(str(input_tif), str(output_jpg), 95, create_world_file=True, reference_tif_path=str(input_tif)))
    if not ok or not output_jpg.exists():
        raise RuntimeError(f"Échec conversion TIF->JPG: {input_tif}")


def _normalized_rvt_name(*, tif_path: Path, target_rvt: str) -> str:
    xy = infer_xy_from_file(tif_path)
    if xy is None:
        xy = extract_xy_from_filename(tif_path.name)
    if xy is None:
        return tif_path.name
    return f"LHD_FXX_{int(xy.x_km):04d}_{int(xy.y_km):04d}_{target_rvt}_A_LAMB93{tif_path.suffix}"


def run_existing_rvt(
    *,
    existing_rvt_dir: Path,
    output_dir: Path,
    cv_config: Dict[str, Any],
    output_structure: Dict[str, Any],
    log: LogFn = lambda _: None,
    cancel_check: CancelCheckFn | None = None,
) -> ExistingRvtResult:
    if not existing_rvt_dir.exists() or not existing_rvt_dir.is_dir():
        raise FileNotFoundError(f"Dossier RVT inexistant ou invalide: {existing_rvt_dir}")

    tif_files = sorted(list(existing_rvt_dir.glob("*.tif")) + list(existing_rvt_dir.glob("*.tiff")))
    if not tif_files:
        raise FileNotFoundError(f"Aucun fichier TIF/TIFF trouvé dans {existing_rvt_dir} pour le mode existing_rvt")

    if not bool((cv_config or {}).get("enabled", False)):
        raise RuntimeError("Mode existing_rvt sélectionné mais la computer vision est désactivée")

    target_rvt = str((cv_config or {}).get("target_rvt", "LD"))

    rvt_output_dir: Path | None = None
    try:
        rvt_cfg = output_structure.get("RVT", {}) if isinstance(output_structure, dict) else {}
        base_dir_name = str(rvt_cfg.get("base_dir", "RVT"))
        type_dir_name = str(rvt_cfg.get(target_rvt, target_rvt))
        rvt_output_dir = (output_dir / "results") / base_dir_name / type_dir_name
        rvt_output_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        rvt_output_dir = None

    if rvt_output_dir is not None:
        try:
            tif_out_dir = rvt_output_dir / "tif"
            tif_out_dir.mkdir(parents=True, exist_ok=True)
            for tif_path in tif_files:
                dest = tif_out_dir / tif_path.name
                if not dest.exists():
                    shutil.copy2(str(tif_path), str(dest))
        except Exception:
            pass

    jpg_output_dir = (rvt_output_dir / "jpg") if rvt_output_dir is not None else existing_rvt_dir
    jpg_output_dir.mkdir(parents=True, exist_ok=True)

    jpg_files: List[Path] = []
    tif_transform_data: Dict[str, Any] = {}
    kept_tif_names: set[str] = set()
    kept_jpg_names: set[str] = set()
    for tif_path in tif_files:
        effective_tif_path = tif_path
        if rvt_output_dir is not None:
            try:
                tif_out_dir = rvt_output_dir / "tif"
                tif_out_dir.mkdir(parents=True, exist_ok=True)
                normalized_name = _normalized_rvt_name(tif_path=tif_path, target_rvt=target_rvt)
                dest = tif_out_dir / normalized_name
                if dest.name != tif_path.name:
                    log(f"RVT: renommage (coords) {tif_path.name} -> {dest.name}")
                shutil.copy2(str(tif_path), str(dest))
                effective_tif_path = dest
                kept_tif_names.add(dest.name)
            except Exception:
                effective_tif_path = tif_path

        jpg_path = jpg_output_dir / (effective_tif_path.stem + ".jpg")
        if not jpg_path.exists():
            log(f"Conversion TIF->JPG (existing_rvt): {effective_tif_path.name} -> {jpg_path.name}")
            _convert_tif_to_jpg_with_world(effective_tif_path, jpg_path)
        jpg_files.append(jpg_path)
        kept_jpg_names.add(jpg_path.name)

        pixel_width, pixel_height, x_origin, y_origin = extract_tif_transform_data(effective_tif_path)
        if all(v is not None for v in (pixel_width, pixel_height, x_origin, y_origin)):
            tif_transform_data[jpg_path.stem] = (float(pixel_width), float(pixel_height), float(x_origin), float(y_origin))

    if rvt_output_dir is not None:
        try:
            tif_out_dir = rvt_output_dir / "tif"
            for p in tif_out_dir.glob("*.tif"):
                if p.name in kept_tif_names:
                    continue
                try:
                    size = int(p.stat().st_size)
                except Exception:
                    size = -1
                if size == 0 or p.stem.isdigit():
                    try:
                        p.unlink()
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            for p in jpg_output_dir.glob("*.jpg"):
                if p.name in kept_jpg_names:
                    continue
                if p.stem.isdigit():
                    try:
                        p.unlink()
                    except Exception:
                        pass
        except Exception:
            pass

    from ..cv.runner import deduplicate_cv_shapefiles_final, run_cv_on_folder
    from ..ign.products.results import build_vrt_index

    effective_cv_config = dict(cv_config or {})
    effective_cv_config["scan_all"] = True

    run_cv_on_folder(
        jpg_dir=jpg_output_dir,
        cv_config=effective_cv_config,
        target_rvt=target_rvt,
        rvt_base_dir=rvt_output_dir,
        tif_transform_data=tif_transform_data,
        run_shapefile_dedup=False,
        log=log,
        cancel_check=cancel_check,
    )

    # Créer le VRT AVANT de générer les shapefiles/projet QGIS
    # pour que le projet QGIS puisse référencer le VRT au lieu des TIF individuels
    if rvt_output_dir is not None:
        tif_out_dir = rvt_output_dir / "tif"
        if tif_out_dir.exists() and list(tif_out_dir.glob("*.tif")):
            build_vrt_index(tif_out_dir, pattern="*.tif", output_name="index.vrt", log=log)
        if jpg_output_dir.exists() and list(jpg_output_dir.glob("*.jpg")):
            build_vrt_index(jpg_output_dir, pattern="*.jpg", output_name="index.vrt", log=log)

    if bool(effective_cv_config.get("generate_shapefiles", False)) and rvt_output_dir is not None:
        deduplicate_cv_shapefiles_final(
            labels_dir=jpg_output_dir,
            shp_dir=rvt_output_dir / "shapefiles",
            target_rvt=target_rvt,
            cv_config=cv_config,
            tif_transform_data=tif_transform_data,
            crs="EPSG:2154",
            log=log,
        )

    return ExistingRvtResult(total_images=len(jpg_files))
