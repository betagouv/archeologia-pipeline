from __future__ import annotations

import os
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ...cancellation import PipelineCancelled, check_cancelled
from ...coords import extract_xy_from_tile_name as _extract_xy_from_tile_name
from ...geo_utils import extract_tif_transform_data
from ...output_paths import indices_dir, indice_tif_dir, indice_base_dir
from ...subprocess_utils import run_subprocess_cancellable, subprocess_kwargs_no_window
from .rvt_naming import PRODUCT_ORDER, get_rvt_source_and_dest_filenames, get_rvt_folder_name
from ...types import CancelCheckFn, LogFn


# _extract_xy_from_tile_name importé depuis coords
# _subprocess_kwargs_no_window importé depuis subprocess_utils


def build_vrt_index(
    folder: Path,
    *,
    pattern: str = "*.tif",
    output_name: str = "index.vrt",
    log: LogFn = lambda _: None,
) -> bool:
    """
    Crée un fichier VRT (Virtual Raster) indexant tous les fichiers correspondant au pattern.
    Permet de charger toutes les dalles d'un coup dans QGIS.
    """
    try:
        gdalbuildvrt = shutil.which("gdalbuildvrt")
        if not gdalbuildvrt:
            log("gdalbuildvrt introuvable: création VRT ignorée")
            return False

        files = sorted(folder.glob(pattern))
        if not files:
            return False

        vrt_path = folder / output_name
        # gdalbuildvrt écrase toujours le VRT de sortie : on régénère systématiquement

        # Use -input_file_list to avoid Windows command line length limit (WinError 206)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            for file in files:
                f.write(str(file) + '\n')
            filelist_path = f.name
        
        try:
            cmd = [str(gdalbuildvrt), "-input_file_list", filelist_path, str(vrt_path)]
            r = subprocess.run(cmd, capture_output=True, text=True, **subprocess_kwargs_no_window())
        finally:
            try:
                os.unlink(filelist_path)
            except OSError:
                pass
        if r.returncode != 0:
            log(f"Échec gdalbuildvrt pour {folder.name}: {r.stderr or r.stdout}")
            return False
        log(f"VRT créé: {vrt_path.relative_to(folder.parent)}")
        return True
    except Exception as e:
        log(f"Erreur création VRT pour {folder.name}: {e}")
        return False


def build_raster_pyramids(
    raster_file: Path,
    *,
    levels: List[int] | None = None,
    log: LogFn = lambda _: None,
    cancel_check: CancelCheckFn | None = None,
) -> bool:
    try:
        gdaladdo = shutil.which("gdaladdo")
        if not gdaladdo:
            log("gdaladdo introuvable: génération pyramides ignorée")
            return False

        if levels is None or len(levels) == 0:
            levels = [2, 4, 8, 16, 32, 64]

        if not raster_file.exists():
            return False

        cmd = [
            str(gdaladdo),
            "-r",
            "average",
            str(raster_file),
            *[str(lvl) for lvl in levels],
        ]
        r = run_subprocess_cancellable(cmd, cancel=cancel_check)
        if r.returncode != 0:
            log(f"Échec gdaladdo (pyramides) pour {raster_file.name}: {r.stderr or r.stdout}")
            return False
        return True
    except PipelineCancelled:
        raise
    except Exception as e:
        log(f"Erreur génération pyramides (gdaladdo) pour {raster_file.name}: {e}")
        return False


def _convert_tif_to_png(
    input_tif: Path, output_png: Path, *, cancel_check: CancelCheckFn | None = None
) -> bool:
    try:
        from .convert_tif_to_png import convert_tif_to_png

        output_png.parent.mkdir(parents=True, exist_ok=True)
        ok = bool(
            convert_tif_to_png(
                str(input_tif),
                str(output_png),
                create_world_file=True,
                reference_tif_path=str(input_tif),
            )
        )
        return ok and output_png.exists()
    except PipelineCancelled:
        raise
    except Exception:
        try:
            gdal_translate = shutil.which("gdal_translate")
            if not gdal_translate:
                return False
            output_png.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                str(gdal_translate),
                "-of",
                "PNG",
                "-co",
                "WORLDFILE=YES",
                str(input_tif),
                str(output_png),
            ]
            run_subprocess_cancellable(cmd, cancel=cancel_check, output_path=output_png)
            return output_png.exists()
        except PipelineCancelled:
            raise
        except Exception:
            return False


def copy_mnt_to_results(
    *,
    temp_mnt_path: Path,
    output_dir: Path,
    current_tile_name: str,
    log: LogFn = lambda _: None,
) -> Path:
    x, y = _extract_xy_from_tile_name(current_tile_name)

    mnt_tif = indice_tif_dir(output_dir, "MNT")
    mnt_tif.mkdir(parents=True, exist_ok=True)

    output_name = f"LHD_FXX_{x}_{y}_MNT_A_0M50_LAMB93_IGN69.tif"
    out_path = mnt_tif / output_name

    if not temp_mnt_path.exists():
        raise FileNotFoundError(f"MNT source introuvable: {temp_mnt_path}")

    if not out_path.exists():
        shutil.copy2(str(temp_mnt_path), str(out_path))
        log(f"MNT copié: {out_path.relative_to(indices_dir(output_dir))}")

    return out_path


def copy_final_products_to_results(
    *,
    temp_dir: Path,
    output_dir: Path,
    current_tile_name: str,
    products: Dict[str, bool],
    output_structure: Dict[str, Any],
    output_formats: Dict[str, Any],
    rvt_params: Dict[str, Any],
    log: LogFn = lambda _: None,
    cancel_check: CancelCheckFn | None = None,
    name_suffix: str = "",
) -> Dict[str, Any]:
    x, y = _extract_xy_from_tile_name(current_tile_name)

    idx_dir = indices_dir(output_dir)
    idx_dir.mkdir(parents=True, exist_ok=True)

    # Générer les noms de fichiers avec paramètres pour invalider le cache
    all_products = list(PRODUCT_ORDER)
    source_files_cropped: Dict[str, str] = {}
    source_files_uncropped: Dict[str, str] = {}
    for product in all_products:
        uncropped, cropped = get_rvt_source_and_dest_filenames(
            product, current_tile_name, x, y, rvt_params, name_suffix=name_suffix
        )
        source_files_uncropped[product] = uncropped
        source_files_cropped[product] = cropped

    out_formats_tif = bool(output_formats.get("tif", True))
    jpg_cfg = output_formats.get("jpg", {}) if isinstance(output_formats.get("jpg", {}), dict) else {}

    pyramids_enabled = True
    pyramids_levels: List[int] = [2, 4, 8, 16, 32, 64]

    created_jpgs: List[Path] = []
    created_jpgs_by_product: Dict[str, List[Path]] = {}
    tif_transform_data: Dict[str, Tuple[float, float, float, float]] = {}

    for product_name in PRODUCT_ORDER:
        if not products.get(product_name, False):
            continue
        check_cancelled(cancel_check)

        cropped_name = source_files_cropped[product_name]
        uncropped_name = source_files_uncropped[product_name]
        input_path_cropped = temp_dir / cropped_name
        input_path_uncropped = temp_dir / uncropped_name

        # Dossier suffixé par les paramètres RVT (ex: indices/SVF_R10_D16_V1_N0/) :
        # deux exécutions de paramètres différents ne s'écrasent plus.
        # MNT/DENSITE → suffixe vide → dossier brut. Doit rester symétrique avec
        # resolve_rvt_tif_dir côté consommation CV (même rvt_params).
        folder_name = get_rvt_folder_name(product_name, rvt_params)
        base_dir = indice_base_dir(output_dir, folder_name)

        # Utiliser le nom du fichier croppé (sans extension) comme base
        output_base = cropped_name.replace(".tif", "")

        if out_formats_tif:
            tif_dir = base_dir / "tif"
            tif_dir.mkdir(parents=True, exist_ok=True)
            tif_path = tif_dir / f"{output_base}.tif"
            if input_path_cropped.exists() and not tif_path.exists():
                shutil.copy2(str(input_path_cropped), str(tif_path))
                log(f"TIF rogné copié: {tif_path.relative_to(idx_dir)}")
                if pyramids_enabled:
                    build_raster_pyramids(tif_path, levels=pyramids_levels, log=log, cancel_check=cancel_check)

        should_jpg = bool(jpg_cfg.get(product_name, False))
        if should_jpg:
            jpg_dir = base_dir / "png"
            jpg_dir.mkdir(parents=True, exist_ok=True)
            jpg_path = jpg_dir / f"{output_base}.png"
            if not input_path_uncropped.exists():
                log(
                    f"PNG demandé mais TIF source introuvable: {input_path_uncropped.relative_to(temp_dir)} (produit={product_name})"
                )
            elif jpg_path.exists():
                log(f"PNG déjà présent: {jpg_path.relative_to(idx_dir)}")
                created_jpgs.append(jpg_path)
                created_jpgs_by_product.setdefault(product_name, []).append(jpg_path)
            else:
                ok = _convert_tif_to_png(input_path_uncropped, jpg_path, cancel_check=cancel_check)
                if ok:
                    log(f"PNG créé: {jpg_path.relative_to(idx_dir)}")
                    created_jpgs.append(jpg_path)
                    created_jpgs_by_product.setdefault(product_name, []).append(jpg_path)
                    pixel_width, pixel_height, x_origin, y_origin = extract_tif_transform_data(input_path_uncropped)
                    if all(v is not None for v in (pixel_width, pixel_height, x_origin, y_origin)):
                        tif_transform_data[jpg_path.stem] = (
                            float(pixel_width),
                            float(pixel_height),
                            float(x_origin),
                            float(y_origin),
                        )
                else:
                    log(
                        f"Échec conversion TIF->PNG: {input_path_uncropped.relative_to(temp_dir)} -> {jpg_path.relative_to(idx_dir)}"
                    )

    return {
        "created_jpgs": created_jpgs,
        "created_jpgs_by_product": created_jpgs_by_product,
        "tif_transform_data": tif_transform_data,
    }
