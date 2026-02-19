from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from ..geo_utils import extract_tif_transform_data
from ..coords import extract_xy_from_filename, infer_xy_from_file
from ..types import LogFn, CancelCheckFn


@dataclass(frozen=True)
class ExistingRvtResult:
    total_images: int


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


def _cleanup_orphans(directory: Path | None, glob_pattern: str, kept_names: set[str]) -> None:
    """Supprime les fichiers orphelins (vides ou à nom numérique) non produits par cette exécution."""
    if directory is None or not directory.exists():
        return
    try:
        for p in directory.glob(glob_pattern):
            if p.name in kept_names:
                continue
            is_empty = p.stat().st_size == 0
            if is_empty or p.stem.isdigit():
                try:
                    p.unlink()
                except Exception:
                    pass
    except Exception:
        pass


def run_existing_rvt(
    *,
    existing_rvt_dir: Path,
    output_dir: Path,
    cv_config: Dict[str, Any],
    output_structure: Dict[str, Any],
    log: LogFn = lambda _: None,
    cancel_check: CancelCheckFn | None = None,
    rvt_params: Dict[str, Any] | None = None,
) -> ExistingRvtResult:
    if not existing_rvt_dir.exists() or not existing_rvt_dir.is_dir():
        raise FileNotFoundError(f"Dossier RVT inexistant ou invalide: {existing_rvt_dir}")

    tif_files = sorted(list(existing_rvt_dir.glob("*.tif")) + list(existing_rvt_dir.glob("*.tiff")))
    if not tif_files:
        raise FileNotFoundError(f"Aucun fichier TIF/TIFF trouvé dans {existing_rvt_dir} pour le mode existing_rvt")

    cv_enabled = bool((cv_config or {}).get("enabled", False))
    target_rvt = str((cv_config or {}).get("target_rvt", "LD"))

    rvt_output_dir: Path | None = None
    try:
        from ..ign.products.rvt_naming import get_rvt_param_suffix
        rvt_cfg = output_structure.get("RVT", {}) if isinstance(output_structure, dict) else {}
        base_dir_name = str(rvt_cfg.get("base_dir", "RVT"))
        type_dir_base = str(rvt_cfg.get(target_rvt, target_rvt))
        param_suffix = get_rvt_param_suffix(target_rvt, rvt_params or {}) if rvt_params else ""
        type_dir_name = f"{type_dir_base}{param_suffix}" if param_suffix else type_dir_base
        rvt_output_dir = (output_dir / "results") / base_dir_name / type_dir_name
        rvt_output_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        rvt_output_dir = None

    jpg_output_dir = (rvt_output_dir / "jpg") if rvt_output_dir is not None else existing_rvt_dir
    jpg_output_dir.mkdir(parents=True, exist_ok=True)

    tif_out_dir = (rvt_output_dir / "tif") if rvt_output_dir is not None else None
    if tif_out_dir is not None:
        tif_out_dir.mkdir(parents=True, exist_ok=True)

    jpg_files: List[Path] = []
    tif_transform_data: Dict[str, Any] = {}
    kept_tif_names: set[str] = set()
    kept_jpg_names: set[str] = set()

    for tif_path in tif_files:
        if cancel_check is not None and cancel_check():
            log("Annulation demandée, arrêt du traitement RVT.")
            break

        effective_tif_path = tif_path
        if tif_out_dir is not None:
            try:
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

    # Nettoyage des fichiers orphelins (vides ou numériques) non produits par cette exécution
    _cleanup_orphans(tif_out_dir, "*.tif", kept_tif_names)
    _cleanup_orphans(jpg_output_dir, "*.jpg", kept_jpg_names)

    # Computer Vision (uniquement si activée)
    # La déduplication shapefile est gérée par run_cv_on_folder (run_shapefile_dedup=True).
    # La création des VRT est déléguée à finalize_pipeline() pour éviter le double travail.
    if cv_enabled and not (cancel_check is not None and cancel_check()):
        from ..cv.runner import run_cv_on_folder

        # cv_config est déjà un run_cfg unique (le caller boucle sur les runs)
        cv_config["scan_all"] = True

        run_cv_on_folder(
            jpg_dir=jpg_output_dir,
            cv_config=cv_config,
            target_rvt=target_rvt,
            rvt_base_dir=rvt_output_dir,
            tif_transform_data=tif_transform_data,
            run_shapefile_dedup=True,
            log=log,
            cancel_check=cancel_check,
        )

    return ExistingRvtResult(total_images=len(jpg_files))
