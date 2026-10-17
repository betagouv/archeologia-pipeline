from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..pdal_validation import validate_las_or_laz_with_pdal
from . import mnt as _mnt
from .qgis_processing import run_qgis_algorithm
from .results import needs_refresh
from ...types import LogFn


@dataclass(frozen=True)
class DensityResult:
    density_path: Path


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

    # Fraîcheur vis-à-vis du LAZ fusionné (cf. mnt.py) ; entrée absente +
    # sortie présente → cache gardé (comportement historique).
    if output_path.exists() and (
        not input_laz_path.exists() or not needs_refresh(input_laz_path, output_path)
    ):
        log(f"Densité réutilisée (cache intermédiaire) : {output_file}")
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
    # Emprise = celle des points du LAZ fusionné : pdal:density n'a pas de
    # paramètre BOUNDS (ignoré, retiré au lot 0e-a), cf. mnt.py.
    log(f"Création densité avec marge {margin_percent * 100}% ({margin_meters}m)")
    # Trace post-hoc des valeurs effectivement envoyées à PDAL.
    from ...types import format_params_line
    log(format_params_line("DENSITE", {
        "tile": current_tile_name,
        "resolution_m": density_resolution,
        "tile_overlap_pct": tile_overlap_percent,
        "filter": filter_expression,
    }))

    parameters: Dict[str, Any] = {
        "INPUT": str(input_laz_path),
        "OUTPUT": str(output_path),
        "RESOLUTION": str(density_resolution),
        "FILTER_EXPRESSION": filter_expression,
    }

    # Même grille calée que le MNT (mnt.emprise_alignee) : sans emprise, pdal:density cale la sienne sur le
    # premier point moins un demi-pixel, et le rognage au kilomètre la rééchantillonnait (couverture comprise).
    bornes = _mnt._bornes_du_nuage(input_laz_path)
    if bornes is not None:
        x0, x1, y0, y1 = _mnt.emprise_alignee(bornes, density_resolution)
        parameters["FILTER_EXTENT"] = f"{x0},{x1},{y0},{y1} [EPSG:2154]"
    else:
        log("⚠️ Densité : bornes du nuage illisibles — grille de PDAL sans calage (décalage possible d'un demi-pixel)")

    run_qgis_algorithm("pdal:density", parameters, feedback=feedback, context=context)

    if not output_path.exists():
        raise RuntimeError(f"Échec création densité: fichier non créé: {output_path}")

    return DensityResult(density_path=output_path)
