from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from ..coords import extract_xy_from_filename, infer_xy_from_file


CancelFn = Callable[[], bool]


@dataclass(frozen=True)
class TileRecord:
    x_km: int
    y_km: int
    filename: str
    url: str


def extract_xy_from_ign_filename(filename: str) -> Optional[Tuple[int, int]]:
    """Extract (x_km, y_km) from IGN naming convention.

    Expected patterns include:
    - LHD_FXX_0946_6744_...(.laz/.copc.laz)

    Returns None if no coords can be parsed.
    """
    xy = extract_xy_from_filename(filename)
    if xy is None:
        return None
    return int(xy.x_km), int(xy.y_km)


def infer_xy_from_pdal_bounds(path: Path, *, cancel: Optional[CancelFn] = None) -> Optional[Tuple[int, int]]:
    """Infer tile (x_km, y_km) from PDAL bounds (km grid), using xmin/ymax.

    Uses: pdal info --metadata <file>

    Returns None if PDAL is missing, fails, or bounds cannot be located.
    """

    inferred = infer_xy_from_file(path, cancel=cancel)
    if inferred is None:
        return None
    return int(inferred.x_km), int(inferred.y_km)


def rename_file_with_inferred_coords(
    *,
    dalles_dir: Path,
    filename: str,
    x_km: int,
    y_km: int,
    log: Callable[[str], None] = lambda _: None,
) -> str:
    """Rename downloaded file to include inferred coords.

    Returns the new filename (or the original filename if rename fails).
    """

    dest = dalles_dir / filename
    if not dest.exists() or not dest.is_file():
        return filename

    new_name = f"IGN_EXT_{x_km:04d}_{y_km:04d}{dest.suffix}"
    new_dest = dalles_dir / new_name

    try:
        if new_dest.exists() and new_dest != dest:
            new_name = f"IGN_EXT_{x_km:04d}_{y_km:04d}_{dest.stem}{dest.suffix}"
            new_dest = dalles_dir / new_name

        if new_dest != dest:
            dest.rename(new_dest)
            log(f"Renommage (fallback coords): {filename} -> {new_name}")
        return new_name
    except Exception as e:
        log(f"Renommage ignoré (fallback coords) pour {filename}: {e}")
        return filename


def build_sorted_records_with_fallback(
    *,
    file_list: List[Tuple[str, str]],
    dalles_dir: Path,
    cancel: Optional[CancelFn] = None,
    log: Callable[[str], None] = lambda _: None,
) -> List[TileRecord]:
    """Build sortable tile records for IGN downloads.

    For each (filename,url):
    - try parsing coords from filename
    - if missing, infer via PDAL bounds from the downloaded file and rename it

    Returns a list of TileRecord sorted by (x_km, y_km).
    """

    records: List[TileRecord] = []

    for filename, url in file_list:
        if cancel is not None and cancel():
            log("Annulation demandée")
            break

        dest = dalles_dir / filename
        inferred = infer_xy_from_pdal_bounds(dest, cancel=cancel)
        xy = None
        if inferred is not None:
            xy = (int(inferred[0]), int(inferred[1]))
        else:
            xy = extract_xy_from_ign_filename(filename)

        if xy is None:
            log(f"Impossible d'inférer les coordonnées via metadata et nom: {filename}")
            continue

        if extract_xy_from_ign_filename(filename) is None:
            filename = rename_file_with_inferred_coords(
                dalles_dir=dalles_dir,
                filename=filename,
                x_km=int(xy[0]),
                y_km=int(xy[1]),
                log=log,
            )

        records.append(TileRecord(x_km=int(xy[0]), y_km=int(xy[1]), filename=filename, url=url))

    records.sort(key=lambda r: (r.x_km, r.y_km))
    return records
