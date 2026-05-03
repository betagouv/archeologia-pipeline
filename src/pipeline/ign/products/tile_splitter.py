"""Découpage d'un raster continu (MNT ou indice RVT) en sous-dalles 1 km alignées IGN.

Ce module existe pour absorber des jeux de données non-IGN (issus de LiDARs
locaux, drones, photogrammétrie) dont l'emprise couvre plusieurs km² sans
suivre la grille standard des dalles LHD_FXX_*. Les sous-dalles générées
réutilisent la convention IGN (xmin = x_km*1000, ymax = y_km*1000), ce qui
permet aux étapes suivantes du pipeline (RVT, CV, déduplication) de
fonctionner sans modification.

Deux usages principaux :

- :func:`split_mnt_to_ign_tiles` : découpe un MNT en sous-dalles 1 km (format
  de compression LERC_ZSTD, tolérance 1 cm). Destiné au mode ``existing_mnt``.
- :func:`split_rvt_raster_to_ign_tiles` : découpe un indice RVT (LD, SVF…)
  en sous-dalles 1 km nommées selon la convention IGN
  ``LHD_FXX_{x}_{y}_{RVT}_A_LAMB93.tif``. Destiné au mode ``existing_rvt``.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from ...subprocess_utils import subprocess_kwargs_no_window
from ...types import LogFn


# Taille standard des dalles IGN LiDAR HD (mètres).
IGN_TILE_SIZE_M = 1000


@dataclass(frozen=True)
class RasterSubTile:
    """Sous-dalle 1 km IGN-alignée générique (MNT ou RVT)."""
    x_km: int
    y_km: int
    x_str: str            # "0677"
    y_str: str            # "6256"
    tile_path: Path       # chemin TIF sur disque
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass(frozen=True)
class MntSubTile:
    """Représente une sous-dalle 1 km générée à partir d'un MNT plus large."""
    x_km: int
    y_km: int
    x_str: str            # "0677"
    y_str: str            # "6256"
    tile_name: str        # "MNT_EXT_0677_6256"
    tile_path: Path       # chemin TIF sur disque
    xmin: int
    ymin: int
    xmax: int
    ymax: int


def compute_ign_tiles_covering_bounds(
    bounds: Tuple[float, float, float, float],
    *,
    only_full_coverage: bool = False,
    tile_size_m: int = IGN_TILE_SIZE_M,
) -> List[Tuple[int, int]]:
    """Liste les couples (x_km, y_km) des dalles IGN 1 km intersectant un rectangle.

    Convention IGN : la dalle (x_km, y_km) couvre
    x ∈ [x_km * 1000, (x_km + 1) * 1000[ et y ∈ ](y_km - 1) * 1000, y_km * 1000].

    Args:
        bounds: (xmin, ymin, xmax, ymax) en mètres, projection Lambert-93.
        only_full_coverage: si True, on ne conserve que les dalles dont
            l'emprise est intégralement recouverte par ``bounds`` (utile pour
            éviter les sous-dalles avec bord de NoData).
        tile_size_m: taille d'une dalle (par défaut 1 000 m).

    Returns:
        Liste de couples (x_km, y_km) triés par X croissant puis Y croissant.
    """
    xmin, ymin, xmax, ymax = bounds

    x_km_start = int(math.floor(xmin / tile_size_m))
    x_km_end_excl = int(math.ceil(xmax / tile_size_m))
    y_km_start = int(math.floor(ymin / tile_size_m)) + 1
    y_km_end_excl = int(math.ceil(ymax / tile_size_m)) + 1

    tiles: List[Tuple[int, int]] = []
    for x_km in range(x_km_start, x_km_end_excl):
        for y_km in range(y_km_start, y_km_end_excl):
            txmin = x_km * tile_size_m
            txmax = (x_km + 1) * tile_size_m
            tymin = (y_km - 1) * tile_size_m
            tymax = y_km * tile_size_m

            if only_full_coverage:
                if txmin < xmin or txmax > xmax or tymin < ymin or tymax > ymax:
                    continue
            else:
                # exiger une intersection non-dégénérée
                if txmax <= xmin or txmin >= xmax or tymax <= ymin or tymin >= ymax:
                    continue
            tiles.append((x_km, y_km))

    return tiles


def split_raster_to_ign_tiles(
    *,
    raster_path: Path,
    output_dir: Path,
    bounds: Tuple[float, float, float, float],
    output_filename_fn: Callable[[int, int], str],
    log: LogFn = lambda _: None,
    gdalwarp_path: Optional[str] = None,
    only_full_coverage: bool = False,
    creation_options: Optional[List[str]] = None,
    description: str = "sous-dalle",
) -> List[RasterSubTile]:
    """Découpe un raster en sous-dalles 1 km alignées sur la grille IGN (version générique).

    Les sous-dalles sont écrites dans ``output_dir`` sous le nom donné par
    ``output_filename_fn(x_km, y_km)``. Les fichiers déjà présents avec une
    taille non nulle sont réutilisés (cache anti-rework).

    Args:
        raster_path: chemin du raster source.
        output_dir: dossier où écrire les sous-dalles.
        bounds: (xmin, ymin, xmax, ymax) du raster (mètres, Lambert-93).
        output_filename_fn: fonction ``(x_km, y_km) -> filename`` (sans chemin).
        creation_options: liste d'options ``-co`` pour ``gdalwarp``
            (par défaut : ``["COMPRESS=ZSTD", "PREDICTOR=2"]``).
        only_full_coverage: voir :func:`compute_ign_tiles_covering_bounds`.
        description: label utilisé dans les logs (ex. ``"sous-dalle MNT"``).
    """
    gdalwarp = gdalwarp_path or shutil.which("gdalwarp")
    if not gdalwarp:
        raise FileNotFoundError("gdalwarp executable not found in PATH")

    output_dir.mkdir(parents=True, exist_ok=True)

    tile_coords = compute_ign_tiles_covering_bounds(
        bounds, only_full_coverage=only_full_coverage
    )
    if not tile_coords:
        log(f"Aucune tuile 1 km IGN ne recouvre les bornes de {raster_path.name}: {bounds}")
        return []

    if creation_options is None:
        creation_options = ["COMPRESS=ZSTD", "PREDICTOR=2"]

    coverage_note = " (couverture complète uniquement)" if only_full_coverage else ""
    log(
        f"Découpage de {raster_path.name} en {len(tile_coords)} {description}(s) "
        f"1 km alignée(s) sur la grille IGN{coverage_note}"
    )

    sub_tiles: List[RasterSubTile] = []
    for x_km, y_km in tile_coords:
        x_str = f"{x_km:04d}"
        y_str = f"{y_km:04d}"
        tile_filename = output_filename_fn(x_km, y_km)
        tile_path = output_dir / tile_filename

        txmin = x_km * IGN_TILE_SIZE_M
        txmax = (x_km + 1) * IGN_TILE_SIZE_M
        tymin = (y_km - 1) * IGN_TILE_SIZE_M
        tymax = y_km * IGN_TILE_SIZE_M

        if tile_path.exists() and tile_path.stat().st_size > 0:
            log(f"  [cache] {tile_path.name}")
            sub_tiles.append(RasterSubTile(
                x_km=x_km, y_km=y_km, x_str=x_str, y_str=y_str,
                tile_path=tile_path,
                xmin=txmin, ymin=tymin, xmax=txmax, ymax=tymax,
            ))
            continue

        cmd = [
            gdalwarp,
            "-te", str(txmin), str(tymin), str(txmax), str(tymax),
            str(raster_path), str(tile_path),
            "-of", "GTiff",
        ]
        for opt in creation_options:
            cmd.extend(["-co", opt])
        cmd.append("-overwrite")

        r = subprocess.run(
            cmd, capture_output=True, text=True, **subprocess_kwargs_no_window()
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"gdalwarp (split) a échoué pour {tile_path.name}: "
                f"{r.stderr or r.stdout}"
            )

        if not tile_path.exists() or tile_path.stat().st_size == 0:
            log(f"  Sous-dalle vide ignorée: {tile_path.name}")
            continue

        log(f"  [ok] {tile_path.name} (emprise {txmin}->{txmax}, {tymin}->{tymax})")
        sub_tiles.append(RasterSubTile(
            x_km=x_km, y_km=y_km, x_str=x_str, y_str=y_str,
            tile_path=tile_path,
            xmin=txmin, ymin=tymin, xmax=txmax, ymax=tymax,
        ))

    return sub_tiles


def split_mnt_to_ign_tiles(
    *,
    mnt_path: Path,
    output_dir: Path,
    bounds: Tuple[float, float, float, float],
    log: LogFn = lambda _: None,
    gdalwarp_path: Optional[str] = None,
    only_full_coverage: bool = False,
    tile_prefix: str = "MNT_EXT",
) -> List[MntSubTile]:
    """Découpe un raster MNT en sous-dalles 1 km alignées sur la grille IGN.

    Les sous-dalles sont écrites dans ``output_dir`` au format
    ``{tile_prefix}_{x:04d}_{y:04d}_MNT.tif`` (nomenclature compatible avec
    ``_process_single_mnt_tile``). Les fichiers déjà présents avec une taille
    non nulle sont réutilisés (cache anti-rework).

    Les sous-dalles incluent automatiquement les bords NoData : ``gdalwarp``
    étend le raster à l'emprise exacte de la cellule 1 km, le LiDAR local ne
    couvrant généralement pas intégralement chaque dalle IGN.
    """
    def _naming(x_km: int, y_km: int) -> str:
        return f"{tile_prefix}_{x_km:04d}_{y_km:04d}_MNT.tif"

    raw_tiles = split_raster_to_ign_tiles(
        raster_path=mnt_path,
        output_dir=output_dir,
        bounds=bounds,
        output_filename_fn=_naming,
        creation_options=["COMPRESS=LERC_ZSTD", "MAX_Z_ERROR=0.01"],
        log=log,
        gdalwarp_path=gdalwarp_path,
        only_full_coverage=only_full_coverage,
        description="sous-dalle MNT",
    )

    return [
        MntSubTile(
            x_km=r.x_km, y_km=r.y_km, x_str=r.x_str, y_str=r.y_str,
            tile_name=f"{tile_prefix}_{r.x_str}_{r.y_str}",
            tile_path=r.tile_path,
            xmin=r.xmin, ymin=r.ymin, xmax=r.xmax, ymax=r.ymax,
        )
        for r in raw_tiles
    ]


def split_rvt_raster_to_ign_tiles(
    *,
    raster_path: Path,
    output_dir: Path,
    bounds: Tuple[float, float, float, float],
    target_rvt: str,
    log: LogFn = lambda _: None,
    gdalwarp_path: Optional[str] = None,
    only_full_coverage: bool = False,
) -> List[RasterSubTile]:
    """Découpe un raster d'indice RVT (LD, SVF, SLO...) en sous-dalles 1 km IGN.

    Les sous-dalles sont écrites au format
    ``LHD_FXX_{x:04d}_{y:04d}_{target_rvt}_A_LAMB93.tif`` afin d'être
    directement compatibles avec le reste du pipeline (CV, déduplication
    par dalle, conversion shapefile).
    """
    rvt_label = (target_rvt or "RVT").upper()
    if rvt_label == "M_HS":
        # Le nom affiché côté IGN utilise un tiret (cf. rvt_naming.py)
        rvt_filename_token = "M-HS"
    else:
        rvt_filename_token = rvt_label

    def _naming(x_km: int, y_km: int) -> str:
        return f"LHD_FXX_{x_km:04d}_{y_km:04d}_{rvt_filename_token}_A_LAMB93.tif"

    return split_raster_to_ign_tiles(
        raster_path=raster_path,
        output_dir=output_dir,
        bounds=bounds,
        output_filename_fn=_naming,
        creation_options=["COMPRESS=ZSTD", "PREDICTOR=2"],
        log=log,
        gdalwarp_path=gdalwarp_path,
        only_full_coverage=only_full_coverage,
        description=f"sous-dalle {rvt_label}",
    )
