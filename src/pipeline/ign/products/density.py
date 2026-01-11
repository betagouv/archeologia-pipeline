from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from ..pdal_validation import get_laz_bounds, validate_las_or_laz_with_pdal
from .qgis_processing import run_qgis_algorithm


LogFn = Callable[[str], None]


@dataclass(frozen=True)
class DensityResult:
    density_path: Path


def _extract_xy_from_tile_name(tile_name: str) -> Tuple[int, int]:
    parts = tile_name.split("_")
    if len(parts) < 4:
        raise ValueError(f"Nom de dalle inattendu: {tile_name}")
    return int(parts[2]), int(parts[3])


def create_density_map(
    *,
    input_laz_path: Path,
    temp_dir: Path,
    current_tile_name: str,
    density_resolution: float,
    tile_overlap_percent: float,
    filter_expression: str,
    log: LogFn = lambda _: None,
    feedback: Optional[Any] = None,
    context: Optional[Any] = None,
) -> DensityResult:
    output_file = f"{current_tile_name}_densite.tif"
    output_path = temp_dir / output_file

    if output_path.exists():
        return DensityResult(density_path=output_path)

    if not input_laz_path.exists():
        raise FileNotFoundError(f"Fichier d'entrée densité introuvable: {input_laz_path}")

    ok, msg = validate_las_or_laz_with_pdal(input_laz_path)
    if not ok:
        raise IOError(f"Fichier d'entrée densité illisible/corrompu via PDAL: {input_laz_path} ({msg})")

    temp_dir.mkdir(parents=True, exist_ok=True)

    margin_percent = float(tile_overlap_percent) / 100.0
    # Marge toujours basée sur 1km (taille d'une dalle IGN standard)
    margin_meters = 1000.0 * margin_percent
    
    # Lire les bounds réels du fichier LAZ (important pour les dalles fusionnées)
    laz_bounds = get_laz_bounds(input_laz_path)
    
    if laz_bounds is not None:
        base_xmin, base_ymin, base_xmax, base_ymax = laz_bounds
        # Arrondir aux km pour avoir des bounds propres
        base_xmin = float(int(base_xmin / 1000) * 1000)
        base_ymin = float(int(base_ymin / 1000) * 1000)
        base_xmax = float((int(base_xmax / 1000) + 1) * 1000)
        base_ymax = float((int(base_ymax / 1000) + 1) * 1000)
    else:
        # Fallback: calculer à partir du nom (pour une dalle simple de 1km)
        x_km, y_km = _extract_xy_from_tile_name(current_tile_name)
        base_xmin = float(x_km * 1000)
        base_xmax = float((x_km + 1) * 1000)
        base_ymin = float((y_km - 1) * 1000)
        base_ymax = float(y_km * 1000)

    extended_xmin = base_xmin - margin_meters
    extended_xmax = base_xmax + margin_meters
    extended_ymin = base_ymin - margin_meters
    extended_ymax = base_ymax + margin_meters

    log(f"Création densité avec marge {margin_percent * 100}% ({margin_meters}m)")

    parameters: Dict[str, Any] = {
        "INPUT": str(input_laz_path),
        "OUTPUT": str(output_path),
        "BOUNDS": f"{extended_xmin},{extended_ymin},{extended_xmax},{extended_ymax}",
        "RESOLUTION": str(density_resolution),
        "FILTER_EXPRESSION": filter_expression,
    }

    run_qgis_algorithm("pdal:density", parameters, feedback=feedback, context=context)

    if not output_path.exists():
        raise RuntimeError(f"Échec création densité: fichier non créé: {output_path}")

    return DensityResult(density_path=output_path)
