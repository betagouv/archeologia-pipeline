from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .cvat import compute_cvat
from .qgis_processing import run_qgis_algorithm
from .rvt_naming import get_rvt_temp_filename
from ...tilespec import reclass_rvt_nodata
from ...types import LogFn, format_params_line


@dataclass(frozen=True)
class IndicesResult:
    outputs: Dict[str, Path]


def _as_int(value: Any, default: int) -> int:
    """Convertit une valeur en int, gère la virgule française."""
    try:
        if isinstance(value, str):
            # Gérer la virgule française (ex: "10,5" -> 10)
            value = value.replace(",", ".")
        return int(float(value))
    except Exception:
        return int(default)


def _as_float(value: Any, default: float) -> float:
    """Convertit une valeur en float, gère la virgule française."""
    try:
        if isinstance(value, str):
            # Gérer la virgule française (ex: "1,7" -> 1.7)
            value = value.replace(",", ".")
        return float(value)
    except Exception:
        return float(default)


def _as_bool(value: Any, default: bool) -> bool:
    try:
        return bool(value)
    except Exception:
        return bool(default)


def create_visualization_products(
    *,
    temp_dir: Path,
    current_tile_name: str,
    products: Dict[str, bool],
    rvt_params: Dict[str, Any],
    log: LogFn = lambda _: None,
    feedback: Optional[Any] = None,
    context: Optional[Any] = None,
) -> IndicesResult:
    temp_dir.mkdir(parents=True, exist_ok=True)

    input_file = f"{current_tile_name}_MNT.tif"
    input_path = temp_dir / input_file
    if not input_path.exists():
        raise FileNotFoundError(f"MNT introuvable pour indices: {input_path}")

    outputs: Dict[str, Path] = {}

    if products.get("HS", False):
        hs = (rvt_params or {}).get("hs", {})
        sun_azimuth = _as_int(hs.get("sun_azimuth", 315), 315)
        sun_elevation = _as_int(hs.get("sun_elevation", 35), 35)
        ve_factor = _as_int(hs.get("ve_factor", 1), 1)
        save_as_8bit = _as_bool(hs.get("save_as_8bit", True), True)
        out = temp_dir / get_rvt_temp_filename("HS", current_tile_name, rvt_params)
        if not out.exists():
            log(format_params_line("RVT/HS", {
                "tile": current_tile_name,
                "sun_azimuth": sun_azimuth,
                "sun_elevation": sun_elevation,
                "ve_factor": ve_factor,
                "save_as_8bit": save_as_8bit,
            }))
            params = {
                "INPUT": str(input_path),
                "OUTPUT": str(out),
                "SUN_AZIMUTH": sun_azimuth,
                "SUN_ELEVATION": sun_elevation,
                "VE_FACTOR": ve_factor,
                "SAVE_AS_8BIT": save_as_8bit,
            }
            run_qgis_algorithm("rvt:rvt_hillshade", params, feedback=feedback, context=context)
        if out.exists():
            outputs["HS"] = out
        else:
            log(f"HS non créé: {out.name}")

    if products.get("M_HS", False):
        mdh = (rvt_params or {}).get("mdh", {})
        num_directions = _as_int(mdh.get("num_directions", 16), 16)
        if num_directions < 2:
            log(
                f"RVT multi-hillshade: NUM_DIRECTIONS={num_directions} invalide, utilisation de 16 (minimum=2)"
            )
            num_directions = 16
        sun_elevation = _as_int(mdh.get("sun_elevation", 35), 35)
        ve_factor = _as_int(mdh.get("ve_factor", 1), 1)
        save_as_8bit = _as_bool(mdh.get("save_as_8bit", True), True)
        out = temp_dir / get_rvt_temp_filename("M_HS", current_tile_name, rvt_params)
        if not out.exists():
            log(format_params_line("RVT/M_HS", {
                "tile": current_tile_name,
                "num_directions": num_directions,
                "sun_elevation": sun_elevation,
                "ve_factor": ve_factor,
                "save_as_8bit": save_as_8bit,
            }))
            params = {
                "INPUT": str(input_path),
                "OUTPUT": str(out),
                "NUM_DIRECTIONS": num_directions,
                "SAVE_AS_8BIT": save_as_8bit,
                "SUN_ELEVATION": sun_elevation,
                "VE_FACTOR": ve_factor,
            }
            run_qgis_algorithm("rvt:rvt_multi_hillshade", params, feedback=feedback, context=context)
        if out.exists():
            outputs["M_HS"] = out
        else:
            log(f"M_HS non créé: {out.name}")

    if products.get("SVF", False):
        svf = (rvt_params or {}).get("svf", {})
        num_directions = _as_int(svf.get("num_directions", 16), 16)
        if num_directions < 2:
            log(f"RVT SVF: NUM_DIRECTIONS={num_directions} invalide, utilisation de 16 (minimum=2)")
            num_directions = 16
        radius = _as_int(svf.get("radius", 10), 10)
        ve_factor = _as_int(svf.get("ve_factor", 1), 1)
        noise_remove = _as_int(svf.get("noise_remove", 0), 0)
        save_as_8bit = _as_bool(svf.get("save_as_8bit", True), True)
        out = temp_dir / get_rvt_temp_filename("SVF", current_tile_name, rvt_params)
        if not out.exists():
            log(format_params_line("RVT/SVF", {
                "tile": current_tile_name,
                "num_directions": num_directions,
                "radius": radius,
                "noise_remove": noise_remove,
                "ve_factor": ve_factor,
                "save_as_8bit": save_as_8bit,
            }))
            params = {
                "INPUT": str(input_path),
                "OUTPUT": str(out),
                "NOISE_REMOVE": noise_remove,
                "NUM_DIRECTIONS": num_directions,
                "RADIUS": radius,
                "SAVE_AS_8BIT": save_as_8bit,
                "VE_FACTOR": ve_factor,
            }
            run_qgis_algorithm("rvt:rvt_svf", params, feedback=feedback, context=context)
        if out.exists():
            outputs["SVF"] = out
        else:
            log(f"SVF non créé: {out.name}")

    if products.get("SLO", False):
        slope = (rvt_params or {}).get("slope", {})
        unit = _as_int(slope.get("unit", 0), 0)
        ve_factor = _as_int(slope.get("ve_factor", 1), 1)
        save_as_8bit = _as_bool(slope.get("save_as_8bit", True), True)
        out = temp_dir / get_rvt_temp_filename("SLO", current_tile_name, rvt_params)
        if not out.exists():
            log(format_params_line("RVT/SLO", {
                "tile": current_tile_name,
                "unit": "degrees" if unit == 0 else "percent",
                "ve_factor": ve_factor,
                "save_as_8bit": save_as_8bit,
            }))
            params = {
                "INPUT": str(input_path),
                "OUTPUT": str(out),
                "UNIT": unit,
                "VE_FACTOR": ve_factor,
                "SAVE_AS_8BIT": save_as_8bit,
            }
            run_qgis_algorithm("rvt:rvt_slope", params, feedback=feedback, context=context)
        if out.exists():
            outputs["SLO"] = out
        else:
            log(f"SLO non créé: {out.name}")

    if products.get("LD", False):
        ldo = (rvt_params or {}).get("ldo", {})
        angular_res = _as_int(ldo.get("angular_res", 15), 15)
        min_radius = _as_int(ldo.get("min_radius", 10), 10)
        max_radius = _as_int(ldo.get("max_radius", 20), 20)
        # OBSERVER_H: utiliser _as_float pour gérer la virgule française
        observer_h = _as_float(ldo.get("observer_h", 1.7), 1.7)
        # RVT LD requiert OBSERVER_H > 0
        if observer_h <= 0:
            log(f"LD: OBSERVER_H={observer_h} invalide (RVT requiert > 0), utilisation de 1.7")
            observer_h = 1.7
        ve_factor = _as_int(ldo.get("ve_factor", 1), 1)
        save_as_8bit = _as_bool(ldo.get("save_as_8bit", True), True)
        out = temp_dir / get_rvt_temp_filename("LD", current_tile_name, rvt_params)
        if not out.exists():
            # RVT QGIS: passer un entier si la valeur est entière, sinon float
            observer_h_param = int(observer_h) if observer_h == int(observer_h) else observer_h
            log(format_params_line("RVT/LD", {
                "tile": current_tile_name,
                "angular_res": angular_res,
                "min_radius": min_radius,
                "max_radius": max_radius,
                "observer_h_m": observer_h,
                "ve_factor": ve_factor,
                "save_as_8bit": save_as_8bit,
            }))
            params = {
                "INPUT": str(input_path),
                "OUTPUT": str(out),
                "ANGULAR_RES": angular_res,
                "MIN_RADIUS": min_radius,
                "MAX_RADIUS": max_radius,
                "OBSERVER_H": observer_h_param,
                "VE_FACTOR": ve_factor,
                "SAVE_AS_8BIT": save_as_8bit,
            }
            run_qgis_algorithm("rvt:rvt_ld", params, feedback=feedback, context=context)
        if out.exists():
            outputs["LD"] = out
        else:
            log(f"LD non créé: {out.name}")

    if products.get("SLRM", False):
        slrm = (rvt_params or {}).get("slrm", {})
        radius = _as_int(slrm.get("radius", 20), 20)
        ve_factor = _as_int(slrm.get("ve_factor", 1), 1)
        save_as_8bit = _as_bool(slrm.get("save_as_8bit", True), True)
        out = temp_dir / get_rvt_temp_filename("SLRM", current_tile_name, rvt_params)
        if not out.exists():
            log(format_params_line("RVT/SLRM", {
                "tile": current_tile_name,
                "radius": radius,
                "ve_factor": ve_factor,
                "save_as_8bit": save_as_8bit,
            }))
            params = {
                "INPUT": str(input_path),
                "OUTPUT": str(out),
                "RADIUS": radius,
                "VE_FACTOR": ve_factor,
                "SAVE_AS_8BIT": save_as_8bit,
            }
            run_qgis_algorithm("rvt:rvt_slrm", params, feedback=feedback, context=context)
        if out.exists():
            outputs["SLRM"] = out
        else:
            log(f"SLRM non créé: {out.name}")

    if products.get("VAT", False):
        vat = (rvt_params or {}).get("vat", {})
        terrain_type = _as_int(vat.get("terrain_type", 0), 0)
        blend_combination = _as_int(vat.get("blend_combination", 0), 0)
        save_as_8bit = _as_bool(vat.get("save_as_8bit", True), True)

        # Utiliser get_rvt_temp_filename pour le nom final
        standard_vat_tif = temp_dir / get_rvt_temp_filename("VAT", current_tile_name, rvt_params)
        # Le nom de base pour les outputs intermédiaires de rvt_blender
        vat_output_base = standard_vat_tif.with_suffix("").with_name(standard_vat_tif.stem + "_outputs")

        if not standard_vat_tif.exists():
            log(format_params_line("RVT/VAT", {
                "tile": current_tile_name,
                "terrain_type": {0: "general", 1: "flat", 2: "steep"}.get(terrain_type, str(terrain_type)),
                "blend_combination": blend_combination,
                "save_as_8bit": save_as_8bit,
            }))
            params = {
                "INPUT": str(input_path),
                "distance_units": "meters",
                "area_units": "m2",
                "ellipsoid": "EPSG:7019",
                "BLEND_COMBINATION": blend_combination,
                "TERRAIN_TYPE": terrain_type,
                "SAVE_AS_8BIT": save_as_8bit,
                "SAVE_AS_FLOAT": False,
                "OUTPUT": str(vat_output_base),
            }
            run_qgis_algorithm("rvt:rvt_blender", params, feedback=feedback, context=context)

            expected_tif = Path(str(vat_output_base) + "_8bit.tif")
            if expected_tif.exists():
                shutil.copy2(str(expected_tif), str(standard_vat_tif))
            else:
                candidates = sorted(temp_dir.glob(f"{vat_output_base.name}*.tif"))
                if candidates:
                    shutil.copy2(str(candidates[0]), str(standard_vat_tif))

        if standard_vat_tif.exists():
            outputs["VAT"] = standard_vat_tif

    if products.get("MSTP", False):
        mstp = (rvt_params or {}).get("mstp", {})
        local_scale_min = _as_int(mstp.get("local_scale_min", 3), 3)
        local_scale_max = _as_int(mstp.get("local_scale_max", 21), 21)
        local_scale_step = _as_int(mstp.get("local_scale_step", 2), 2)
        meso_scale_min = _as_int(mstp.get("meso_scale_min", 23), 23)
        meso_scale_max = _as_int(mstp.get("meso_scale_max", 203), 203)
        meso_scale_step = _as_int(mstp.get("meso_scale_step", 18), 18)
        # Ces repli 223/2023/180 sont les défauts de RVT (clé absente), PAS les
        # défauts du plugin (100/400/60, écrits par config_manager et l'UI). Ne
        # pas les « aligner » : rvt_naming.py replie sur les mêmes valeurs, et
        # les faire diverger ferait nommer le dossier autrement qu'il n'est
        # calculé (cf. l'invariant « même rvt_params des deux côtés »).
        broad_scale_min = _as_int(mstp.get("broad_scale_min", 223), 223)
        broad_scale_max = _as_int(mstp.get("broad_scale_max", 2023), 2023)
        broad_scale_step = _as_int(mstp.get("broad_scale_step", 180), 180)
        lightness = _as_float(mstp.get("lightness", 1.2), 1.2)
        ve_factor = _as_int(mstp.get("ve_factor", 1), 1)
        save_as_8bit = _as_bool(mstp.get("save_as_8bit", True), True)
        out = temp_dir / get_rvt_temp_filename("MSTP", current_tile_name, rvt_params)
        if not out.exists():
            log(format_params_line("RVT/MSTP", {
                "tile": current_tile_name,
                "local_scale": f"{local_scale_min}-{local_scale_max}/{local_scale_step}",
                "meso_scale": f"{meso_scale_min}-{meso_scale_max}/{meso_scale_step}",
                "broad_scale": f"{broad_scale_min}-{broad_scale_max}/{broad_scale_step}",
                "lightness": lightness,
                "ve_factor": ve_factor,
                "save_as_8bit": save_as_8bit,
            }))
            params = {
                "INPUT": str(input_path),
                "OUTPUT": str(out),
                "LOCAL_SCALE_MIN": local_scale_min,
                "LOCAL_SCALE_MAX": local_scale_max,
                "LOCAL_SCALE_STEP": local_scale_step,
                "MESO_SCALE_MIN": meso_scale_min,
                "MESO_SCALE_MAX": meso_scale_max,
                "MESO_SCALE_STEP": meso_scale_step,
                "BROAD_SCALE_MIN": broad_scale_min,
                "BROAD_SCALE_MAX": broad_scale_max,
                "BROAD_SCALE_STEP": broad_scale_step,
                "LIGHTNESS": lightness,
                "VE_FACTOR": ve_factor,
                "SAVE_AS_8BIT": save_as_8bit,
            }
            run_qgis_algorithm("rvt:rvt_mstp", params, feedback=feedback, context=context)
        if out.exists():
            outputs["MSTP"] = out
        else:
            log(f"MSTP non créé: {out.name}")

    if products.get("CVAT", False):
        cvat = (rvt_params or {}).get("cvat", {})
        save_as_8bit = _as_bool(cvat.get("save_as_8bit", True), True)
        out = temp_dir / get_rvt_temp_filename("CVAT", current_tile_name, rvt_params)
        if not out.exists():
            log(format_params_line("RVT/CVAT", {
                "tile": current_tile_name,
                "save_as_8bit": save_as_8bit,
            }))
            # CVAT n'est pas exposé via Processing : calcul in-process (cf. cvat.py).
            compute_cvat(
                input_path=input_path,
                output_path=out,
                save_as_8bit=save_as_8bit,
                log=log,
            )
        if out.exists():
            outputs["CVAT"] = out
        else:
            log(f"CVAT non créé: {out.name}")

    # Rendus 8 bits : séparer NoData réel (→255, étiqueté) et valides saturés
    # (→254) tant que le MNT est sous la main — après coup les deux classes de
    # 255 sont indiscernables (cf. tilespec.reclass_rvt_nodata). Sans cela, un
    # MNT à large NoData (dalle étrangère reprojetée) rend le sans-donnée en
    # blanc opaque, et masquer 255 en aval troue les zones saturées (talus).
    # Idempotent, best-effort, no-op sur un produit float.
    for key, out_path in outputs.items():
        try:
            reclass_rvt_nodata(out_path, input_path)
        except Exception as exc:
            log(f"Reclass NoData {key} ignoré ({exc!r})")

    return IndicesResult(outputs=outputs)
