from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

from ..coords import extract_xy_from_filename, infer_xy_from_file


LogFn = Callable[[str], None]


@dataclass(frozen=True)
class LocalLazResult:
    sorted_list_file: Path
    dalles_dir: Path
    total: int


def build_sorted_list_from_local_laz(
    *,
    local_dir: Path,
    sorted_list_file: Path,
    log: LogFn = lambda _: None,
) -> LocalLazResult:
    laz_files_set = set(local_dir.glob("*.laz")) | set(local_dir.glob("*.las"))
    laz_files = list(laz_files_set)
    if not laz_files:
        raise FileNotFoundError(f"Aucun fichier LAZ/LAS trouvé dans {local_dir}")

    temp_data: List[Tuple[int, int, str, str]] = []
    for path in laz_files:
        xy = infer_xy_from_file(path)
        if xy is None:
            xy = extract_xy_from_filename(path.name)
            if xy is None:
                continue
        x, y = int(xy.x_km), int(xy.y_km)
        temp_data.append((x, y, path.name, str(path)))

    if not temp_data:
        raise ValueError("Aucun nuage local avec coordonnées valides trouvé")

    temp_data.sort(key=lambda r: (r[0], r[1]))

    sorted_list_file.parent.mkdir(parents=True, exist_ok=True)
    with sorted_list_file.open("w", encoding="utf-8") as f:
        for _, _, name, full_path in temp_data:
            f.write(f"{name},{full_path}\n")

    log(f"Fichier trié généré (mode local_laz): {sorted_list_file}")

    return LocalLazResult(sorted_list_file=sorted_list_file, dalles_dir=local_dir, total=len(temp_data))


def run_local_laz(
    *,
    local_laz_dir: Path,
    output_dir: Path,
    log: LogFn = lambda _: None,
) -> LocalLazResult:
    if not local_laz_dir.exists() or not local_laz_dir.is_dir():
        raise FileNotFoundError(f"Dossier nuages locaux inexistant ou invalide: {local_laz_dir}")

    sorted_list_file = output_dir / "fichier_tri.txt"
    return build_sorted_list_from_local_laz(local_dir=local_laz_dir, sorted_list_file=sorted_list_file, log=log)
