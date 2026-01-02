from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .qgis_processing import run_qgis_algorithm


LogFn = Callable[[str], None]


@dataclass(frozen=True)
class IndicesResult:
    outputs: Dict[str, Path]


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_float(value: Any, default: float) -> float:
    try:
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

    if products.get("M_HS", False):
        out = temp_dir / f"{current_tile_name}_hillshade.tif"
        if not out.exists():
            mdh = (rvt_params or {}).get("mdh", {})
            num_directions = _as_int(mdh.get("num_directions", 16), 16)
            if num_directions < 2:
                log(
                    f"RVT multi-hillshade: NUM_DIRECTIONS={num_directions} invalide, utilisation de 16 (minimum=2)"
                )
                num_directions = 16
            params = {
                "INPUT": str(input_path),
                "OUTPUT": str(out),
                "NUM_DIRECTIONS": num_directions,
                "SAVE_AS_8BIT": _as_bool(mdh.get("save_as_8bit", True), True),
                "SUN_ELEVATION": _as_int(mdh.get("sun_elevation", 35), 35),
                "VE_FACTOR": _as_int(mdh.get("ve_factor", 1), 1),
            }
            run_qgis_algorithm("rvt:rvt_multi_hillshade", params, feedback=feedback, context=context)
        outputs["M_HS"] = out

    if products.get("SVF", False):
        out = temp_dir / f"{current_tile_name}_SVF.tif"
        if not out.exists():
            svf = (rvt_params or {}).get("svf", {})
            num_directions = _as_int(svf.get("num_directions", 16), 16)
            if num_directions < 2:
                log(f"RVT SVF: NUM_DIRECTIONS={num_directions} invalide, utilisation de 16 (minimum=2)")
                num_directions = 16
            params = {
                "INPUT": str(input_path),
                "OUTPUT": str(out),
                "NOISE_REMOVE": _as_int(svf.get("noise_remove", 0), 0),
                "NUM_DIRECTIONS": num_directions,
                "RADIUS": _as_int(svf.get("radius", 10), 10),
                "SAVE_AS_8BIT": _as_bool(svf.get("save_as_8bit", True), True),
                "VE_FACTOR": _as_int(svf.get("ve_factor", 1), 1),
            }
            run_qgis_algorithm("rvt:rvt_svf", params, feedback=feedback, context=context)
        outputs["SVF"] = out

    if products.get("SLO", False):
        out = temp_dir / f"{current_tile_name}_Slope.tif"
        if not out.exists():
            slope = (rvt_params or {}).get("slope", {})
            params = {
                "INPUT": str(input_path),
                "OUTPUT": str(out),
                "UNIT": _as_int(slope.get("unit", 0), 0),
                "VE_FACTOR": _as_int(slope.get("ve_factor", 1), 1),
                "SAVE_AS_8BIT": _as_bool(slope.get("save_as_8bit", True), True),
            }
            run_qgis_algorithm("rvt:rvt_slope", params, feedback=feedback, context=context)
        outputs["SLO"] = out

    if products.get("LD", False):
        out = temp_dir / f"{current_tile_name}_LD.tif"
        if not out.exists():
            ldo = (rvt_params or {}).get("ldo", {})
            params = {
                "INPUT": str(input_path),
                "OUTPUT": str(out),
                "ANGULAR_RES": _as_int(ldo.get("angular_res", 15), 15),
                "MIN_RADIUS": _as_int(ldo.get("min_radius", 10), 10),
                "MAX_RADIUS": _as_int(ldo.get("max_radius", 20), 20),
                "OBSERVER_H": _as_float(ldo.get("observer_h", 1.7), 1.7),
                "VE_FACTOR": _as_int(ldo.get("ve_factor", 1), 1),
                "SAVE_AS_8BIT": _as_bool(ldo.get("save_as_8bit", True), True),
            }
            run_qgis_algorithm("rvt:rvt_ld", params, feedback=feedback, context=context)
        outputs["LD"] = out

    if products.get("VAT", False):
        vat = (rvt_params or {}).get("vat", {})
        terrain_type = _as_int(vat.get("terrain_type", 0), 0)
        blend_combination = _as_int(vat.get("blend_combination", 0), 0)
        save_as_8bit = _as_bool(vat.get("save_as_8bit", True), True)

        preset_suffix = f"_T{terrain_type}_B{blend_combination}"
        vat_output_base = temp_dir / f"{current_tile_name}_VAT{preset_suffix}_outputs"
        standard_vat_tif = temp_dir / f"{current_tile_name}_VAT{preset_suffix}.tif"

        if not standard_vat_tif.exists():
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
                candidates = sorted(temp_dir.glob(f"{current_tile_name}_VAT{preset_suffix}_outputs*.tif"))
                if candidates:
                    shutil.copy2(str(candidates[0]), str(standard_vat_tif))

        if standard_vat_tif.exists():
            outputs["VAT"] = standard_vat_tif

    return IndicesResult(outputs=outputs)
