from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from ..pdal_validation import validate_las_or_laz_with_pdal
from .qgis_processing import run_qgis_algorithm


LogFn = Callable[[str], None]


@dataclass(frozen=True)
class TerrainModelResult:
    mnt_path: Path


def _extract_xy_from_tile_name(tile_name: str) -> Tuple[int, int]:
    parts = tile_name.split("_")
    if len(parts) < 4:
        raise ValueError(f"Nom de dalle inattendu: {tile_name}")
    return int(parts[2]), int(parts[3])


def create_terrain_model(
    *,
    input_laz_path: Path,
    temp_dir: Path,
    current_tile_name: str,
    mnt_resolution: float,
    tile_overlap_percent: float,
    filter_expression: str,
    log: LogFn = lambda _: None,
    feedback: Optional[Any] = None,
    context: Optional[Any] = None,
) -> TerrainModelResult:
    output_file = f"{current_tile_name}_MNT.tif"
    output_path = temp_dir / output_file

    if not input_laz_path.exists():
        raise FileNotFoundError(f"Fichier d'entrée MNT introuvable: {input_laz_path}")

    ok, msg = validate_las_or_laz_with_pdal(input_laz_path)
    if not ok:
        raise IOError(f"Fichier d'entrée MNT illisible/corrompu via PDAL: {input_laz_path} ({msg})")

    if output_path.exists():
        return TerrainModelResult(mnt_path=output_path)

    temp_dir.mkdir(parents=True, exist_ok=True)

    margin_percent = float(tile_overlap_percent) / 100.0
    margin_meters = 1000.0 * margin_percent

    x_km, y_km = _extract_xy_from_tile_name(current_tile_name)

    base_xmin = x_km * 1000
    base_xmax = (x_km + 1) * 1000
    base_ymin = (y_km - 1) * 1000
    base_ymax = y_km * 1000

    extended_xmin = base_xmin - margin_meters
    extended_xmax = base_xmax + margin_meters
    extended_ymin = base_ymin - margin_meters
    extended_ymax = base_ymax + margin_meters

    log(f"Création MNT avec marge {margin_percent * 100}%")

    parameters: Dict[str, Any] = {
        "INPUT": str(input_laz_path),
        "OUTPUT": str(output_path),
        "RESOLUTION": str(mnt_resolution),
        "FILTER_EXPRESSION": filter_expression,
        "BOUNDS": f"{extended_xmin},{extended_ymin},{extended_xmax},{extended_ymax}",
    }

    try:
        run_qgis_algorithm("pdal:exportrastertin", parameters, feedback=feedback, context=context)
    except Exception as e:
        log(f"Échec création MNT avec pdal:exportrastertin: {e}")
        log("Tentative de fallback avec pdal:exportraster...")
        run_qgis_algorithm("pdal:exportraster", parameters, feedback=feedback, context=context)

    if not output_path.exists():
        raise RuntimeError(f"Échec création MNT: fichier non créé: {output_path}")

    return TerrainModelResult(mnt_path=output_path)
