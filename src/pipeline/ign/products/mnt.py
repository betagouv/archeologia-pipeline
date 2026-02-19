from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from ..pdal_validation import get_laz_bounds, validate_las_or_laz_with_pdal
from .qgis_processing import run_qgis_algorithm
from ...types import LogFn


@dataclass(frozen=True)
class TerrainModelResult:
    mnt_path: Path


def _try_extract_xy_from_tile_name(tile_name: str) -> Optional[Tuple[int, int]]:
    """Tente d'extraire les coordonnées X,Y du nom de dalle. Retourne None si impossible."""
    parts = tile_name.split("_")
    if len(parts) < 4:
        return None
    try:
        return int(parts[2]), int(parts[3])
    except ValueError:
        return None


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
    # Marge toujours basée sur 1km (taille d'une dalle IGN standard)
    margin_meters = 1000.0 * margin_percent
    
    # Priorité 1: extraire les coordonnées du nom de la dalle (1km x 1km standard IGN)
    xy = _try_extract_xy_from_tile_name(current_tile_name)
    
    if xy is not None:
        x_km, y_km = xy
        base_xmin = float(x_km * 1000)
        base_xmax = float((x_km + 1) * 1000)
        base_ymin = float((y_km - 1) * 1000)
        base_ymax = float(y_km * 1000)
        log(f"Bounds dalle {current_tile_name}: {base_xmin},{base_ymin} -> {base_xmax},{base_ymax} (1000x1000m)")
    else:
        # Fallback: lire les bounds du LAZ et calculer le centre de la dalle 1km x 1km
        laz_bounds = get_laz_bounds(input_laz_path)
        if laz_bounds is None:
            raise ValueError(f"Impossible de déterminer les bounds pour {current_tile_name}: "
                           "coordonnées absentes du nom et bounds LAZ illisibles")
        raw_xmin, raw_ymin, raw_xmax, raw_ymax = laz_bounds
        # Calculer le centre du fichier LAZ
        center_x = (raw_xmin + raw_xmax) / 2.0
        center_y = (raw_ymin + raw_ymax) / 2.0
        # Déterminer la dalle 1km x 1km contenant ce centre
        x_km = int(center_x / 1000)
        y_km = int(center_y / 1000) + 1  # +1 car convention IGN: y_km est le bord supérieur
        base_xmin = float(x_km * 1000)
        base_xmax = float((x_km + 1) * 1000)
        base_ymin = float((y_km - 1) * 1000)
        base_ymax = float(y_km * 1000)
        log(f"Bounds dalle (via centre LAZ): {base_xmin},{base_ymin} -> {base_xmax},{base_ymax} (1000x1000m)")

    extended_xmin = base_xmin - margin_meters
    extended_xmax = base_xmax + margin_meters
    extended_ymin = base_ymin - margin_meters
    extended_ymax = base_ymax + margin_meters

    log(f"Création MNT avec marge {margin_percent * 100}% ({margin_meters}m)")

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
