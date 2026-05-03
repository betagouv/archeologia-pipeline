from __future__ import annotations

import os
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..coords import extract_xy_from_filename, get_raster_bounds, infer_xy_from_file
from ..subprocess_utils import subprocess_kwargs_no_window
from ..ign.products.crop import copy_products_without_crop, crop_final_products
from ..ign.products.indices import create_visualization_products
from ..ign.products.results import copy_final_products_to_results
from ..ign.products.tile_splitter import IGN_TILE_SIZE_M
from ..types import CancelCheckFn, LogFn


# Tolérance d'alignement sur la grille IGN 1 km (mètres). Un MNT dont
# les bornes s'écartent de plus de cette valeur du multiple de 1 km le plus
# proche est considéré "small" (emprise native préservée, pas de crop).
_ALIGN_TOLERANCE_M = 50


@dataclass(frozen=True)
class ExistingMntResult:
    total: int


# _subprocess_kwargs_no_window importé depuis subprocess_utils


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


def _large_tile_name_for(
    mnt_path: Path,
    bounds: Tuple[float, float, float, float],
) -> str:
    """Produit un ``current_tile_name`` pour un MNT traité d'un seul bloc.

    Le pipeline interne RVT attend un nom de la forme
    ``LHD_FXX_{x}_{y}_MNT.tif`` : plusieurs étages en aval
    (``copy_products_without_crop``, ``copy_final_products_to_results``,
    conversion shapefile) appellent :func:`extract_xy_from_tile_name` qui
    prend ``parts[2]`` / ``parts[3]``. Pour un MNT de grande emprise qui
    n'est pas une dalle IGN 1 km, on dérive donc x/y depuis le coin
    nord-ouest des bornes Lambert-93 (pareil que pour les tuiles IGN
    classiques), et on préserve le stem d'origine après coup pour
    garantir l'unicité du nom quand deux MNT différents partagent le même
    coin NW au kilomètre près.

    - Format final : ``LHD_FXX_{xxxx}_{yyyy}_EXT_{safe_stem}``.
    - Les caractères non alphanumériques du stem sont remplacés par ``_``
      pour rester compatible avec GDAL / QGIS et éviter les collisions
      entre tokens.
    - Un éventuel suffixe ``_MNT`` final du stem est retiré pour ne pas
      produire ``*_MNT_MNT.tif``.
    """
    import re

    from ..coords import _infer_xy_from_bounds  # type: ignore[attr-defined]

    xmin, _ymin, _xmax, ymax = bounds
    xy = _infer_xy_from_bounds(xmin, ymax)
    if xy is None:
        # Fallback robuste : arrondi vers le bas au km
        x_km = int(xmin // 1000)
        y_km = int(ymax // 1000)
    else:
        x_km = xy.x_km
        y_km = xy.y_km

    stem = mnt_path.stem
    if stem.lower().endswith("_mnt"):
        stem = stem[:-4]
    safe_stem = re.sub(r"[^0-9A-Za-z]+", "_", stem).strip("_")
    safe_stem = safe_stem or "EXT"

    return f"LHD_FXX_{x_km:04d}_{y_km:04d}_EXT_{safe_stem}"


def _classify_mnt_layout(
    bounds: Tuple[float, float, float, float],
    *,
    tile_size_m: int = IGN_TILE_SIZE_M,
    tol_m: float = _ALIGN_TOLERANCE_M,
) -> str:
    """Classifie l'emprise d'un MNT pour choisir le flux de traitement adapté.

    - ``"large"``  : largeur ou hauteur > 1 km + tolérance → on traite le
      raster d'un seul bloc (RVT calculés sur l'emprise complète, CV délégué
      à SAHI qui fera son propre slicing 640 × 640 à l'inférence). Pas de
      pré-découpage en sous-dalles 1 km : évite les bordures NoData inutiles
      et garantit la cohérence des indices RVT au travers de la scène.
    - ``"standard"`` : ≈ 1 km aligné sur la grille IGN → comportement d'origine
      (crop 1 km qui est essentiellement un no-op).
    - ``"small"``  : < 1 km ou non aligné → pas de crop, on conserve l'emprise
      native pour ne pas introduire de NoData autour.
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


def _copy_source_mnt_to_temp(
    *,
    source_path: Path,
    temp_mnt_path: Path,
    gdal_translate: Optional[str],
) -> None:
    """Matérialise le MNT source dans temp_dir sous le nom attendu par RVT."""
    if temp_mnt_path.exists():
        return
    suffix = source_path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        shutil.copy2(str(source_path), str(temp_mnt_path))
    elif suffix == ".asc":
        if not gdal_translate:
            raise FileNotFoundError("gdal_translate executable not found in PATH")
        cmd = [gdal_translate, str(source_path), str(temp_mnt_path)]
        r = subprocess.run(
            cmd, capture_output=True, text=True, **subprocess_kwargs_no_window()
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr or r.stdout)
    else:
        raise ValueError(f"Format MNT non supporté: {source_path.suffix}")


def _process_single_mnt_tile(
    *,
    current_tile_name: str,
    temp_dir: Path,
    output_dir: Path,
    products: Dict[str, bool],
    output_structure: Dict[str, Any],
    output_formats: Dict[str, Any],
    rvt_params: Dict[str, Any],
    skip_crop: bool,
    log: LogFn,
    cancel_check: CancelCheckFn | None,
) -> bool:
    """Exécute le flux RVT → crop (ou copie) → copie finale pour une dalle unique.

    Le TIF source doit déjà exister à ``temp_dir / {current_tile_name}_MNT.tif``.
    Retourne False si une annulation a été demandée en cours de route.
    """
    create_visualization_products(
        temp_dir=temp_dir,
        current_tile_name=current_tile_name,
        products=products,
        rvt_params=rvt_params,
        log=log,
    )

    if cancel_check is not None and cancel_check():
        log("Annulation demandée après création des indices RVT.")
        return False

    if skip_crop:
        copy_products_without_crop(
            temp_dir=temp_dir,
            current_tile_name=current_tile_name,
            products=products,
            rvt_params=rvt_params,
            log=log,
        )
    else:
        crop_final_products(
            temp_dir=temp_dir,
            current_tile_name=current_tile_name,
            products=products,
            rvt_params=rvt_params,
            log=log,
        )

    if cancel_check is not None and cancel_check():
        log("Annulation demandée après le crop/copie des produits.")
        return False

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
    return True


def run_existing_mnt(
    *,
    existing_mnt_dir: Path,
    output_dir: Path,
    products: Dict[str, bool],
    output_structure: Dict[str, Any],
    output_formats: Dict[str, Any],
    rvt_params: Dict[str, Any],
    log: LogFn = lambda _: None,
    cancel_check: CancelCheckFn | None = None,
) -> ExistingMntResult:
    if not existing_mnt_dir.exists() or not existing_mnt_dir.is_dir():
        raise FileNotFoundError(f"Dossier MNT inexistant ou invalide: {existing_mnt_dir}")

    mnt_files = list(existing_mnt_dir.glob("*.tif")) + list(existing_mnt_dir.glob("*.tiff")) + list(existing_mnt_dir.glob("*.asc"))
    if not mnt_files:
        raise FileNotFoundError(f"Aucun fichier MNT (*.tif, *.tiff, *.asc) trouvé dans {existing_mnt_dir}")

    from ..output_paths import intermediaires_dir
    temp_dir = intermediaires_dir(output_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    gdal_translate = shutil.which("gdal_translate")

    processed = 0
    for mnt_path in mnt_files:
        if cancel_check is not None and cancel_check():
            log("Annulation demandée, arrêt du traitement MNT.")
            break

        # 1) Inspecter l'emprise du MNT pour choisir le flux adapté
        bounds = get_raster_bounds(mnt_path)
        if bounds is not None:
            layout = _classify_mnt_layout(bounds)
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            log(
                f"MNT {mnt_path.name}: emprise ≈ {width:.0f} x {height:.0f} m → layout='{layout}'"
            )
        else:
            layout = "standard"
            log(f"MNT {mnt_path.name}: emprise inconnue → layout par défaut 'standard'")

        # 2) Cas LARGE: PAS de pré-découpage. On calcule les indices RVT sur
        #    le raster complet, puis le CV (SAHI) ira découper à l'inférence
        #    en tuiles 640×640. Cela évite les sous-dalles NoData en bord de
        #    couverture et préserve la cohérence des indices sur toute la scène.
        if layout == "large" and bounds is not None:
            current_tile_name = _large_tile_name_for(mnt_path, bounds)
            temp_mnt_path = temp_dir / f"{current_tile_name}_MNT.tif"
            _copy_source_mnt_to_temp(
                source_path=mnt_path,
                temp_mnt_path=temp_mnt_path,
                gdal_translate=gdal_translate,
            )
            log(
                f"MNT {mnt_path.name}: emprise > 1 km → RVT et CV sur le "
                f"raster complet (pas de pré-découpage, SAHI assure le slicing)"
            )
            ok = _process_single_mnt_tile(
                current_tile_name=current_tile_name,
                temp_dir=temp_dir,
                output_dir=output_dir,
                products=products,
                output_structure=output_structure,
                output_formats=output_formats,
                rvt_params=rvt_params,
                # Pas de crop 1 km sur un raster multi-km : on conserve l'emprise native.
                skip_crop=True,
                log=log,
                cancel_check=cancel_check,
            )
            if ok:
                processed += 1
            continue

        # 3) Cas STANDARD et SMALL: une seule dalle logique
        x_str, y_str, tile_name = _infer_tile_coords_from_mnt(mnt_path, log)
        if not x_str or not y_str or not tile_name:
            raise ValueError(f"Impossible de déduire les coordonnées pour le MNT: {mnt_path.name}")

        current_tile_name = tile_name
        temp_mnt_path = temp_dir / f"{current_tile_name}_MNT.tif"
        _copy_source_mnt_to_temp(
            source_path=mnt_path,
            temp_mnt_path=temp_mnt_path,
            gdal_translate=gdal_translate,
        )

        skip_crop = (layout == "small")
        if skip_crop:
            log(
                f"MNT {mnt_path.name}: emprise < 1 km (ou non alignée IGN) → "
                f"crop sauté, emprise native conservée."
            )

        ok = _process_single_mnt_tile(
            current_tile_name=current_tile_name,
            temp_dir=temp_dir,
            output_dir=output_dir,
            products=products,
            output_structure=output_structure,
            output_formats=output_formats,
            rvt_params=rvt_params,
            skip_crop=skip_crop,
            log=log,
            cancel_check=cancel_check,
        )
        if ok:
            processed += 1

    return ExistingMntResult(total=processed)
