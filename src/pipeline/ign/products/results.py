from __future__ import annotations

import os
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from ...cv.runner import extract_tif_transform_data


LogFn = Callable[[str], None]


def _extract_xy_from_tile_name(tile_name: str) -> Tuple[str, str]:
    parts = tile_name.split("_")
    if len(parts) < 4:
        raise ValueError(f"Nom de dalle inattendu: {tile_name}")
    return parts[2], parts[3]


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
        
        # Use -input_file_list to avoid Windows command line length limit (WinError 206)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            for file in files:
                f.write(str(file) + '\n')
            filelist_path = f.name
        
        try:
            cmd = [str(gdalbuildvrt), "-input_file_list", filelist_path, str(vrt_path)]
            r = subprocess.run(cmd, capture_output=True, text=True, **_subprocess_kwargs_no_window())
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
            *[str(l) for l in levels],
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, **_subprocess_kwargs_no_window())
        if r.returncode != 0:
            log(f"Échec gdaladdo (pyramides) pour {raster_file.name}: {r.stderr or r.stdout}")
            return False
        return True
    except Exception as e:
        log(f"Erreur génération pyramides (gdaladdo) pour {raster_file.name}: {e}")
        return False


def _convert_tif_to_jpg(input_tif: Path, output_jpg: Path) -> bool:
    try:
        from .convert_tif_to_jpg import convert_tif_to_jpg

        output_jpg.parent.mkdir(parents=True, exist_ok=True)
        ok = bool(
            convert_tif_to_jpg(
                str(input_tif),
                str(output_jpg),
                95,
                create_world_file=True,
                reference_tif_path=str(input_tif),
            )
        )
        return ok and output_jpg.exists()
    except Exception:
        try:
            gdal_translate = shutil.which("gdal_translate")
            if not gdal_translate:
                return False
            output_jpg.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                str(gdal_translate),
                "-of",
                "JPEG",
                "-co",
                "QUALITY=95",
                "-co",
                "WORLDFILE=YES",
                str(input_tif),
                str(output_jpg),
            ]
            subprocess.run(cmd, check=False, **_subprocess_kwargs_no_window())
            return output_jpg.exists()
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

    results_dir = output_dir / "results"
    mnt_tif_dir = results_dir / "MNT" / "tif"
    mnt_tif_dir.mkdir(parents=True, exist_ok=True)

    output_name = f"LHD_FXX_{x}_{y}_MNT_A_0M50_LAMB93_IGN69.tif"
    out_path = mnt_tif_dir / output_name

    if not temp_mnt_path.exists():
        raise FileNotFoundError(f"MNT source introuvable: {temp_mnt_path}")

    if not out_path.exists():
        shutil.copy2(str(temp_mnt_path), str(out_path))
        log(f"MNT copié: {out_path.relative_to(results_dir)}")

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
    pyramids_config: Dict[str, Any] | None = None,
    log: LogFn = lambda _: None,
) -> Dict[str, Any]:
    x, y = _extract_xy_from_tile_name(current_tile_name)

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    source_files_cropped: Dict[str, str] = {
        "MNT": f"LHD_FXX_{x}_{y}_MNT_A_0M50_LAMB93_IGN69.tif",
        "DENSITE": f"LHD_FXX_{x}_{y}_densite_A_LAMB93.tif",
        "M_HS": f"LHD_FXX_{x}_{y}_M-HS_A_LAMB93.tif",
        "SVF": f"LHD_FXX_{x}_{y}_SVF_A_LAMB93.tif",
        "SLO": f"LHD_FXX_{x}_{y}_SLO_A_LAMB93.tif",
        "LD": f"LHD_FXX_{x}_{y}_LD_A_LAMB93.tif",
        "SLRM": f"LHD_FXX_{x}_{y}_SLRM_A_LAMB93.tif",
        "VAT": f"LHD_FXX_{x}_{y}_VAT_A_LAMB93.tif",
    }

    vat_params = (rvt_params or {}).get("vat", {})
    terrain_type = str(vat_params.get("terrain_type", 0))
    blend_combination = str(vat_params.get("blend_combination", 0))
    preset_suffix = f"_T{terrain_type}_B{blend_combination}"

    source_files_uncropped: Dict[str, str] = {
        "MNT": f"{current_tile_name}_MNT.tif",
        "DENSITE": f"{current_tile_name}_densite.tif",
        "M_HS": f"{current_tile_name}_hillshade.tif",
        "SVF": f"{current_tile_name}_SVF.tif",
        "SLO": f"{current_tile_name}_Slope.tif",
        "LD": f"{current_tile_name}_LD.tif",
        "SLRM": f"{current_tile_name}_SLRM.tif",
        "VAT": f"{current_tile_name}_VAT{preset_suffix}.tif",
    }

    out_formats_tif = bool(output_formats.get("tif", True))
    jpg_cfg = output_formats.get("jpg", {}) if isinstance(output_formats.get("jpg", {}), dict) else {}

    pyramids_enabled = False
    pyramids_levels: List[int] = [2, 4, 8, 16, 32, 64]
    try:
        cfg = pyramids_config or {}
        pyramids_enabled = bool(cfg.get("enabled", False))
        raw_levels = cfg.get("levels", None)
        if isinstance(raw_levels, (list, tuple)):
            parsed = []
            for v in raw_levels:
                try:
                    iv = int(v)
                except Exception:
                    continue
                if iv > 1:
                    parsed.append(iv)
            if parsed:
                pyramids_levels = parsed
    except Exception:
        pyramids_enabled = False

    created_jpgs: List[Path] = []
    created_jpgs_by_product: Dict[str, List[Path]] = {}
    tif_transform_data: Dict[str, Tuple[float, float, float, float]] = {}

    for product_name in ["MNT", "DENSITE", "M_HS", "SVF", "SLO", "LD", "SLRM", "VAT"]:
        if not products.get(product_name, False):
            continue

        cropped_name = source_files_cropped[product_name]
        uncropped_name = source_files_uncropped[product_name]
        input_path_cropped = temp_dir / cropped_name
        input_path_uncropped = temp_dir / uncropped_name

        if product_name in ["MNT", "DENSITE"]:
            base_dir_name = str(output_structure.get(product_name, product_name))
            base_dir = results_dir / base_dir_name
        else:
            rvt_conf = output_structure.get("RVT", {}) if isinstance(output_structure.get("RVT", {}), dict) else {}
            rvt_base = str(rvt_conf.get("base_dir", "RVT"))
            rvt_subdir = str(rvt_conf.get(product_name, product_name))
            base_dir = results_dir / rvt_base / rvt_subdir

        if product_name == "MNT":
            output_base = f"LHD_FXX_{x}_{y}_MNT_A_0M50_LAMB93_IGN69"
        elif product_name == "DENSITE":
            output_base = f"LHD_FXX_{x}_{y}_densite_A_LAMB93"
        else:
            display_name = "M-HS" if product_name == "M_HS" else product_name
            output_base = f"LHD_FXX_{x}_{y}_{display_name}_A_LAMB93"

        if out_formats_tif:
            tif_dir = base_dir / "tif"
            tif_dir.mkdir(parents=True, exist_ok=True)
            tif_path = tif_dir / f"{output_base}.tif"
            if input_path_cropped.exists() and not tif_path.exists():
                shutil.copy2(str(input_path_cropped), str(tif_path))
                log(f"TIF rogné copié: {tif_path.relative_to(results_dir)}")
                if pyramids_enabled:
                    build_raster_pyramids(tif_path, levels=pyramids_levels, log=log)

        should_jpg = bool(jpg_cfg.get(product_name, False))
        if should_jpg:
            jpg_dir = base_dir / "jpg"
            jpg_dir.mkdir(parents=True, exist_ok=True)
            jpg_path = jpg_dir / f"{output_base}.jpg"
            if not input_path_uncropped.exists():
                log(
                    f"JPG demandé mais TIF source introuvable: {input_path_uncropped.relative_to(temp_dir)} (produit={product_name})"
                )
            elif jpg_path.exists():
                log(f"JPG déjà présent: {jpg_path.relative_to(results_dir)}")
                created_jpgs.append(jpg_path)
                created_jpgs_by_product.setdefault(product_name, []).append(jpg_path)
            else:
                ok = _convert_tif_to_jpg(input_path_uncropped, jpg_path)
                if ok:
                    log(f"JPG créé: {jpg_path.relative_to(results_dir)}")
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
                        f"Échec conversion TIF->JPG: {input_path_uncropped.relative_to(temp_dir)} -> {jpg_path.relative_to(results_dir)}"
                    )

    return {
        "created_jpgs": created_jpgs,
        "created_jpgs_by_product": created_jpgs_by_product,
        "tif_transform_data": tif_transform_data,
    }
