from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..geo_utils import extract_tif_transform_data
from ..coords import extract_xy_from_filename, get_raster_bounds, infer_xy_from_file
from ..ign.products.tile_splitter import IGN_TILE_SIZE_M
from ..output_paths import indice_base_dir, indice_tif_dir, indice_jpg_dir
from ..types import LogFn, CancelCheckFn


# Tolérance d'alignement sur la grille IGN 1 km (mètres). Même sémantique
# que dans existing_mnt.py.
_ALIGN_TOLERANCE_M = 50


@dataclass(frozen=True)
class ExistingRvtResult:
    total_images: int


def _classify_rvt_layout(
    bounds: Tuple[float, float, float, float],
    *,
    tile_size_m: int = IGN_TILE_SIZE_M,
    tol_m: float = _ALIGN_TOLERANCE_M,
) -> str:
    """Classifie l'emprise d'un raster RVT pour choisir le flux de traitement.

    Sémantique identique à :func:`_classify_mnt_layout` dans ``existing_mnt.py`` :

    - ``"large"``  : largeur ou hauteur > 1 km + tolérance → on applique SAHI
      directement sur le raster complet (slicing 640 × 640 en mémoire à
      l'inférence). La limite PIL ``MAX_IMAGE_PIXELS`` est désactivée
      globalement pour permettre la conversion TIF → PNG d'une grande image
      (voir ``convert_tif_to_png.py`` et ``computer_vision_onnx.py``).
    - ``"standard"`` : ≈ 1 km aligné sur la grille IGN → comportement d'origine.
    - ``"small"``  : < 1 km ou non aligné → conservé tel quel (nom d'origine
      préservé par ``_normalized_rvt_name``).
    """
    xmin, ymin, xmax, ymax = bounds
    width = xmax - xmin
    height = ymax - ymin

    if width > tile_size_m + tol_m or height > tile_size_m + tol_m:
        return "large"

    dim_ok = abs(width - tile_size_m) <= tol_m and abs(height - tile_size_m) <= tol_m
    align_ok = (
        abs(xmin - round(xmin / tile_size_m) * tile_size_m) <= tol_m
        and abs(ymax - round(ymax / tile_size_m) * tile_size_m) <= tol_m
    )
    if dim_ok and align_ok:
        return "standard"

    return "small"


def _convert_tif_to_png_with_world(input_tif: Path, output_png: Path) -> None:
    from ..ign.products.convert_tif_to_png import convert_tif_to_png

    output_png.parent.mkdir(parents=True, exist_ok=True)
    ok = bool(convert_tif_to_png(str(input_tif), str(output_png), create_world_file=True, reference_tif_path=str(input_tif)))
    if not ok or not output_png.exists():
        raise RuntimeError(f"Échec conversion TIF->PNG: {input_tif}")


def _normalized_rvt_name(*, tif_path: Path, target_rvt: str) -> str:
    # Vérifier l'emprise réelle de la tuile pour éviter les collisions
    # de noms quand les tuiles font moins de ~1 km (ex: 350x350 à 0.5 m/px = 175 m)
    bounds = get_raster_bounds(tif_path)
    if bounds is not None:
        xmin, ymin, xmax, ymax = bounds
        width_m = xmax - xmin
        height_m = ymax - ymin
        if width_m < 900 or height_m < 900 or width_m > 1100 or height_m > 1100:
            # Pas une dalle ~1 km standard → conserver le nom d'origine
            # pour éviter que plusieurs tuiles soient renommées identiquement
            return tif_path.name

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
    global_color_map: Dict[str, Any] | None = None,
    indices_folder_name: str | None = None,
) -> ExistingRvtResult:
    if not existing_rvt_dir.exists() or not existing_rvt_dir.is_dir():
        raise FileNotFoundError(f"Dossier RVT inexistant ou invalide: {existing_rvt_dir}")

    tif_files = sorted(list(existing_rvt_dir.glob("*.tif")) + list(existing_rvt_dir.glob("*.tiff")))
    if not tif_files:
        raise FileNotFoundError(f"Aucun fichier TIF/TIFF trouvé dans {existing_rvt_dir} pour le mode existing_rvt")

    cv_enabled = bool((cv_config or {}).get("enabled", False))
    target_rvt = str((cv_config or {}).get("target_rvt", "LD"))
    # indices_folder_name permet de forcer le nom du dossier indices/<X>/
    # (ex: "RVT" en mode existing_rvt UI où l'indice cible est inconnu)
    folder_name = indices_folder_name if indices_folder_name is not None else target_rvt

    rvt_output_dir: Path | None = None
    try:
        rvt_output_dir = indice_base_dir(output_dir, folder_name)
        rvt_output_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        rvt_output_dir = None

    jpg_output_dir = indice_jpg_dir(output_dir, folder_name) if rvt_output_dir is not None else existing_rvt_dir
    jpg_output_dir.mkdir(parents=True, exist_ok=True)

    tif_out_dir = indice_tif_dir(output_dir, folder_name) if rvt_output_dir is not None else None
    if tif_out_dir is not None:
        tif_out_dir.mkdir(parents=True, exist_ok=True)

    jpg_files: List[Path] = []
    tif_transform_data: Dict[str, Any] = {}
    kept_tif_names: set[str] = set()
    kept_jpg_names: set[str] = set()

    # ── Inspection des rasters RVT (pas de pré-découpage pour les grands) ──
    # Les rasters RVT larges (> 1 km) sont traités tels quels : on laisse
    # SAHI faire son slicing 640×640 à l'inférence. Cela évite les sous-dalles
    # NoData en bord de couverture et préserve la continuité des indices.
    # La limite PIL ``MAX_IMAGE_PIXELS`` est désactivée dans
    # ``convert_tif_to_png.py`` et ``computer_vision_onnx.py`` pour autoriser
    # les grandes emprises.
    for tif_path in tif_files:
        bounds = get_raster_bounds(tif_path)
        layout = _classify_rvt_layout(bounds) if bounds is not None else "standard"
        if layout == "large" and bounds is not None:
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            log(
                f"RVT {tif_path.name}: emprise ≈ {width:.0f} x {height:.0f} m → "
                f"SAHI assure le slicing à l'inférence (pas de pré-découpage)"
            )

    total_tif = len(tif_files)
    log(f"Traitement de {total_tif} fichiers TIF…")

    for idx, tif_path in enumerate(tif_files):
        if cancel_check is not None and cancel_check():
            log("Annulation demandée, arrêt du traitement RVT.")
            break

        effective_tif_path = tif_path
        if tif_out_dir is not None and tif_out_dir.resolve() != existing_rvt_dir.resolve():
            try:
                normalized_name = _normalized_rvt_name(tif_path=tif_path, target_rvt=target_rvt)
                dest = tif_out_dir / normalized_name
                if not dest.exists():
                    if dest.name != tif_path.name:
                        log(f"RVT: renommage (coords) {tif_path.name} -> {dest.name}")
                    shutil.copy2(str(tif_path), str(dest))
                effective_tif_path = dest
                kept_tif_names.add(dest.name)
            except Exception:
                effective_tif_path = tif_path
        elif tif_out_dir is not None:
            # existing_rvt_dir == tif_out_dir : les TIFs sont déjà au bon endroit
            kept_tif_names.add(tif_path.name)

        jpg_path = jpg_output_dir / (effective_tif_path.stem + ".png")
        if not jpg_path.exists():
            log(f"Conversion TIF->PNG (existing_rvt): {effective_tif_path.name} -> {jpg_path.name}")
            _convert_tif_to_png_with_world(effective_tif_path, jpg_path)
        jpg_files.append(jpg_path)
        kept_jpg_names.add(jpg_path.name)

        pixel_width, pixel_height, x_origin, y_origin = extract_tif_transform_data(effective_tif_path)
        if all(v is not None for v in (pixel_width, pixel_height, x_origin, y_origin)):
            tif_transform_data[jpg_path.stem] = (float(pixel_width), float(pixel_height), float(x_origin), float(y_origin))

        if total_tif > 100 and (idx + 1) % 500 == 0:
            log(f"  … {idx + 1}/{total_tif} TIF traités")

    # Nettoyage des fichiers orphelins (vides ou numériques) non produits par cette exécution
    _cleanup_orphans(tif_out_dir, "*.tif", kept_tif_names)
    _cleanup_orphans(jpg_output_dir, "*.png", kept_jpg_names)

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
            output_dir=output_dir,
            tif_transform_data=tif_transform_data,
            run_shapefile_dedup=True,
            global_color_map=global_color_map,
            log=log,
            cancel_check=cancel_check,
        )

    return ExistingRvtResult(total_images=len(jpg_files))
