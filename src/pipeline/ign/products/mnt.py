from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..pdal_validation import validate_las_or_laz_with_pdal
from .qgis_processing import run_qgis_algorithm
from .results import needs_refresh
from ...tilespec import assign_crs_if_missing
from ...types import LogFn


# Chien de garde de l'étape MNT (incident 2026-09-19). Un LAZ corrompu fait
# boucler PDAL sans fin : 10 h de silence sur une dalle, sans erreur ni
# progression. Une dalle légitime se calcule en 1 à 2 min (mesuré sur un lot
# de 423) ; 2 h laissent 60× de marge pour une machine lente ou un gros
# fusionné, tout en bornant le pire cas. Réglable par appel si besoin.
MNT_ALGO_TIMEOUT_S = 2 * 3600


@dataclass(frozen=True)
class TerrainModelResult:
    mnt_path: Path


#: Calage de la grille du MNT et de la densité, inscrit dans la signature du cache (cache_guard) : un dossier calculé
#: avant ce calage refait une fois ses MNT et sa densité au lieu de les resservir (-2 : la densité suit, après -1).
GRILLE_MNT = "emprise-alignee-2"


def emprise_alignee(bornes: Tuple[float, float, float, float], resolution: float) -> Tuple[float, float, float, float]:
    """(xmin, xmax, ymin, ymax) des points, arrondis VERS L'EXTÉRIEUR au multiple de la résolution.

    Sans emprise, pdal:exportrastertin cale la grille d'un LAZ ordinaire (pas d'un COPC) sur le premier point moins
    un demi-pixel (815 999,75 pour un nuage qui commence à 816 000) : le rognage au kilomètre prenait alors, pour
    chaque pixel publié, l'un de ses voisins, d'un sens qui changeait avec le premier point (25 cm à 0,5 m, et une
    dalle décalée d'une colonne d'une relance à l'autre). Calée sur les multiples de la résolution, la grille est
    celle des dalles IGN, et le rognage ne fait plus que découper. Moins d'un pixel de plus que les points : pas de
    zone vide ajoutée au bord d'un lot.
    """
    xmin, xmax, ymin, ymax = (float(v) for v in bornes)
    r = float(resolution)
    eps = 1e-6
    return (math.floor(xmin / r + eps) * r, math.ceil(xmax / r - eps) * r,
            math.floor(ymin / r + eps) * r, math.ceil(ymax / r - eps) * r)


def _bornes_du_nuage(laz: Path) -> Optional[Tuple[float, float, float, float]]:
    """Bornes des points d'après l'en-tête du LAZ (``pdal info --metadata``, sans lire les points), ou None."""
    pdal = shutil.which("pdal")
    if not pdal:
        return None
    try:
        sortie = subprocess.run([pdal, "info", "--metadata", str(laz)], capture_output=True, text=True, timeout=120,
                                **({"creationflags": 0x08000000} if hasattr(subprocess, "STARTUPINFO") else {}))
        m = json.loads(sortie.stdout)["metadata"]
        return float(m["minx"]), float(m["maxx"]), float(m["miny"]), float(m["maxy"])
    except Exception:  # noqa: BLE001 — sans bornes, la grille de PDAL d'avant (le journal le dit)
        return None


def _run_under_watchdog(
    algorithm_id: str,
    parameters: Dict[str, Any],
    *,
    feedback: Optional[Any],
    context: Optional[Any],
    timeout_s: float,
    tile_name: str,
) -> None:
    """Lance un algorithme QGIS sous limite de temps.

    Le feedback annulable (``app.cancellable_feedback``) porte le chien de
    garde ; tout autre feedback (ou ``None``) passe en direct, sans limite.

    ponytail: le feedback est partagé par tout le run et la limite s'arme
    autour d'UN algorithme — valable parce que la boucle produits est
    séquentielle (``process_items_isolated``). La paralléliser demanderait
    un feedback par dalle.
    """
    start = getattr(feedback, "start_watchdog", None)
    stop = getattr(feedback, "stop_watchdog", None)
    if start is None or stop is None:
        run_qgis_algorithm(algorithm_id, parameters, feedback=feedback, context=context)
        return

    start(timeout_s)
    try:
        run_qgis_algorithm(algorithm_id, parameters, feedback=feedback, context=context)
    finally:
        # Le dépassement prime sur l'exception que QGIS lève en tuant le
        # sous-processus : c'est lui la cause, et son message est le seul
        # exploitable dans le journal.
        if stop():
            raise TimeoutError(
                f"{algorithm_id} n'a pas rendu la main en {timeout_s / 3600:.1f} h "
                f"sur {tile_name} — dalle abandonnée (entrée LAZ probablement "
                "corrompue : PDAL boucle sans fin sur une zone illisible)."
            )


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
    algo_timeout_s: float = MNT_ALGO_TIMEOUT_S,
) -> TerrainModelResult:
    output_file = f"{current_tile_name}_MNT.tif"
    output_path = temp_dir / output_file

    if not input_laz_path.exists():
        raise FileNotFoundError(f"Fichier d'entrée MNT introuvable: {input_laz_path}")

    ok, msg = validate_las_or_laz_with_pdal(input_laz_path)
    if not ok:
        raise IOError(f"Fichier d'entrée MNT illisible/corrompu via PDAL: {input_laz_path} ({msg})")

    # Fraîcheur vis-à-vis du LAZ fusionné (pas simple existence) : un
    # re-merge (jeu de voisins modifié, halo) re-matérialise le LAZ → le MNT
    # doit suivre, et la chaîne aval (RVT/PNG/cache CV) se ré-arme d'elle-même.
    if not needs_refresh(input_laz_path, output_path):
        log(f"MNT réutilisé (cache intermédiaire) : {output_file}")
        return TerrainModelResult(mnt_path=output_path)

    temp_dir.mkdir(parents=True, exist_ok=True)

    margin_percent = float(tile_overlap_percent) / 100.0
    # Marge toujours basée sur 1km (taille d'une dalle IGN standard)
    margin_meters = 1000.0 * margin_percent
    # L'emprise du raster est celle des points du LAZ fusionné, marge comprise :
    # pdal:exportrastertin n'a pas de paramètre BOUNDS (QGIS 3.44 et 4.0 : il
    # était ignoré, retiré au lot 0e-a) ; FILTER_EXTENT changerait l'emprise des
    # dalles de bord — décision 0e, pas un correctif silencieux.
    log(f"Création MNT avec marge {margin_percent * 100}% ({margin_meters}m)")
    # Trace post-hoc des valeurs effectivement envoyées à PDAL.
    from ...types import format_params_line
    log(format_params_line("MNT", {
        "tile": current_tile_name,
        "resolution_m": mnt_resolution,
        "tile_overlap_pct": tile_overlap_percent,
        "filter": filter_expression,
    }))

    parameters: Dict[str, Any] = {
        "INPUT": str(input_laz_path),
        "OUTPUT": str(output_path),
        "RESOLUTION": str(mnt_resolution),
        "FILTER_EXPRESSION": filter_expression,
    }
    bornes = _bornes_du_nuage(input_laz_path)
    if bornes is not None:
        x0, x1, y0, y1 = emprise_alignee(bornes, mnt_resolution)
        parameters["FILTER_EXTENT"] = f"{x0},{x1},{y0},{y1} [EPSG:2154]"
        log(f"MNT : grille calée sur {x0:.2f}, {y1:.2f} (multiples de {mnt_resolution} m)")
    else:
        log("⚠️ MNT : bornes du nuage illisibles — grille de PDAL sans calage (décalage possible d'un demi-pixel)")

    watchdog = dict(
        feedback=feedback, context=context, timeout_s=algo_timeout_s, tile_name=current_tile_name
    )
    try:
        _run_under_watchdog("pdal:exportrastertin", parameters, **watchdog)
    except TimeoutError as e:
        # Pas de fallback : pdal:exportraster relirait le MÊME LAZ et
        # rebouclerait le même temps. La dalle est perdue, le lot continue
        # (isolation par dalle dans le runner).
        log(f"⏱ {e}")
        raise
    except Exception as e:
        log(f"Échec création MNT avec pdal:exportrastertin: {e}")
        log("Tentative de fallback avec pdal:exportraster...")
        try:
            _run_under_watchdog("pdal:exportraster", parameters, **watchdog)
        except TimeoutError as e2:
            log(f"⏱ {e2}")
            raise

    if not output_path.exists():
        raise RuntimeError(f"Échec création MNT: fichier non créé: {output_path}")

    # pdal:exportraster* émet parfois le MNT en CRS local « unnamed » (dalles
    # LiDAR HD à CRS compound) ; on réaffirme EPSG:2154 (assignation, pas de
    # reprojection). Le CRS se propage ensuite fidèlement aux RVT/crop/VRT ;
    # sans ça il casse la transformation au zoom QGIS de finalisation.
    assigned = assign_crs_if_missing(output_path)
    if assigned:
        log(f"CRS absent/local sur le MNT → affecté {assigned} (assignation, sans reprojection)")

    return TerrainModelResult(mnt_path=output_path)
