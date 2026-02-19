from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..coords import extract_xy_from_filename, infer_xy_from_file
from .downloader import download_one
from .pdal_validation import (
    run_pdal_command_cancellable,
    validate_las_or_laz_with_pdal,
)
from ..types import LogFn, CancelFn


@dataclass(frozen=True)
class IgnPreprocessResult:
    merged_dir: Path
    temp_dir: Path
    merged_files: List[Path]


def _default_log(_: str) -> None:
    return


def _default_cancel() -> bool:
    return False


def calculate_neighbor_coordinates(x: str, y: str) -> List[Tuple[int, int, int]]:
    min_x = int(f"1{x}") - 10000
    min_y = int(f"1{y}") - 10000

    neighbors: List[Tuple[int, int, int]] = []
    place_dalle = 0

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            place_dalle += 1
            voisin_x = min_x + dx
            voisin_y = min_y + dy
            neighbors.append((voisin_x, voisin_y, place_dalle))

    return neighbors


def format_coordinate(coord: int) -> str:
    return f"{coord:04d}"


def find_neighbor_file(sorted_list_file: Path, voisin_x: int, voisin_y: int, log: LogFn = _default_log) -> Optional[Tuple[str, str]]:
    coord_x = format_coordinate(voisin_x)
    coord_y = format_coordinate(voisin_y)
    search_pattern = f"{coord_x}_{coord_y}"

    try:
        with sorted_list_file.open("r", encoding="utf-8") as f:
            for line in f:
                if search_pattern in line:
                    parts = line.strip().split(",", 1)
                    if len(parts) == 2:
                        return parts[0], parts[1]
    except FileNotFoundError:
        log(f"Fichier trié non trouvé: {sorted_list_file}")

    return None


def calculate_crop_bounds(voisin_x: int, voisin_y: int, place_dalle: int, margin_m: int) -> Dict[str, str]:
    bounds: Dict[str, str] = {}

    if place_dalle == 1:
        xnum = int(f"1{voisin_x:04d}") - 10000
        bounds["xmin"] = f"{xnum:04d}{(1000 - margin_m):03d}"
        xnum2 = int(f"1{voisin_x:04d}") - 10000 + 1
        bounds["xmax"] = f"{xnum2:04d}000"
        ynum = int(f"1{voisin_y:04d}") - 10000 - 1
        bounds["ymin"] = f"{ynum:04d}{(1000 - margin_m):03d}"
        bounds["ymax"] = f"{voisin_y:04d}000"

    elif place_dalle == 2:
        xnum = int(f"1{voisin_x:04d}") - 10000
        bounds["xmin"] = f"{xnum:04d}{(1000 - margin_m):03d}"
        xnum2 = int(f"1{voisin_x:04d}") - 10000 + 1
        bounds["xmax"] = f"{xnum2:04d}000"
        ynum = int(f"1{voisin_y:04d}") - 10000 - 1
        bounds["ymin"] = f"{ynum:04d}000"
        bounds["ymax"] = f"{voisin_y:04d}000"

    elif place_dalle == 3:
        xnum = int(f"1{voisin_x:04d}") - 10000
        bounds["xmin"] = f"{xnum:04d}{(1000 - margin_m):03d}"
        xnum2 = int(f"1{voisin_x:04d}") - 10000 + 1
        bounds["xmax"] = f"{xnum2:04d}000"
        ynum = int(f"1{voisin_y:04d}") - 10000 - 1
        bounds["ymin"] = f"{ynum:04d}000"
        ynum2 = int(f"1{voisin_y:04d}") - 10000 - 1
        bounds["ymax"] = f"{ynum2:04d}{margin_m:03d}"

    elif place_dalle == 4:
        bounds["xmin"] = f"{voisin_x:04d}000"
        xnum = int(f"1{voisin_x:04d}") - 10000 + 1
        bounds["xmax"] = f"{xnum:04d}000"
        ynum = int(f"1{voisin_y:04d}") - 10000 - 1
        bounds["ymin"] = f"{ynum:04d}{(1000 - margin_m):03d}"
        bounds["ymax"] = f"{voisin_y:04d}000"

    elif place_dalle == 5:
        bounds["xmin"] = f"{voisin_x:04d}000"
        xnum = int(f"1{voisin_x:04d}") - 10000 + 1
        bounds["xmax"] = f"{xnum:04d}000"
        ynum = int(f"1{voisin_y:04d}") - 10000 - 1
        bounds["ymin"] = f"{ynum:04d}000"
        ynum2 = int(f"1{voisin_y:04d}") - 10000 - 1
        bounds["ymax"] = f"{ynum2:04d}{margin_m:03d}"

    elif place_dalle == 6:
        xnum = int(f"1{voisin_x:04d}") - 10000
        bounds["xmin"] = f"{xnum:04d}000"
        bounds["xmax"] = f"{xnum:04d}{margin_m:03d}"
        ynum = int(f"1{voisin_y:04d}") - 10000 - 1
        bounds["ymin"] = f"{ynum:04d}{(1000 - margin_m):03d}"
        bounds["ymax"] = f"{voisin_y:04d}000"

    elif place_dalle == 7:
        xnum = int(f"1{voisin_x:04d}") - 10000
        bounds["xmin"] = f"{xnum:04d}000"
        bounds["xmax"] = f"{xnum:04d}{margin_m:03d}"
        ynum = int(f"1{voisin_y:04d}") - 10000 - 1
        bounds["ymin"] = f"{ynum:04d}000"
        bounds["ymax"] = f"{voisin_y:04d}000"

    else:
        xnum = int(f"1{voisin_x:04d}") - 10000
        bounds["xmin"] = f"{xnum:04d}000"
        bounds["xmax"] = f"{xnum:04d}{margin_m:03d}"
        ynum = int(f"1{voisin_y:04d}") - 10000 - 1
        bounds["ymin"] = f"{ynum:04d}000"
        ynum2 = int(f"1{voisin_y:04d}") - 10000 - 1
        bounds["ymax"] = f"{ynum2:04d}{margin_m:03d}"

    return bounds


def _pdal_exe() -> str:
    p = shutil.which("pdal")
    if not p:
        raise FileNotFoundError("pdal executable not found in PATH")
    return p


def crop_neighbor_tile(
    *,
    input_path: Path,
    output_path: Path,
    bounds: Dict[str, str],
    log: LogFn = _default_log,
    cancel: CancelFn = _default_cancel,
) -> bool:
    if cancel():
        return False

    if output_path.exists():
        ok, _ = validate_las_or_laz_with_pdal(output_path)
        if ok:
            return True
        try:
            output_path.unlink()
        except Exception:
            pass

    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline_config = {
        "pipeline": [
            {"type": "readers.las", "filename": str(input_path)},
            {
                "type": "filters.crop",
                "bounds": f"([{bounds['xmin']},{bounds['xmax']}],[{bounds['ymin']},{bounds['ymax']}])",
            },
            {"type": "writers.las", "filename": str(output_path), "compression": "laszip"},
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(pipeline_config, tf, indent=2)
        pipeline_file = tf.name

    try:
        cmd = [_pdal_exe(), "pipeline", pipeline_file]
        result = run_pdal_command_cancellable(cmd, cancel=cancel)
        if result.returncode != 0:
            log(f"Erreur PDAL crop (code {result.returncode})")
            if result.returncode == 3221225477:
                log("PDAL a crashé (0xC0000005). Conseil: relancez le pipeline avec moins de workers (ex: max_workers=2).")
            if result.stderr:
                log(result.stderr.strip())
            return False

        ok, msg = validate_las_or_laz_with_pdal(output_path)
        if not ok:
            log(f"Fichier voisin rogné invalide via PDAL: {output_path.name}")
            if msg:
                log(f"PDAL: {msg}")
            return False

        return True
    finally:
        try:
            Path(pipeline_file).unlink()
        except Exception:
            pass


def merge_tiles(
    *,
    central_path: Path,
    neighbor_paths: List[Path],
    output_path: Path,
    log: LogFn = _default_log,
    cancel: CancelFn = _default_cancel,
) -> bool:
    if cancel():
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        ok, _ = validate_las_or_laz_with_pdal(output_path)
        if ok:
            return True
        try:
            output_path.unlink()
        except Exception:
            pass

    ok_c, msg_c = validate_las_or_laz_with_pdal(central_path)
    if not ok_c:
        log(f"Fichier central invalide via PDAL: {central_path.name}")
        if msg_c:
            log(f"PDAL: {msg_c}")
        return False

    valid_files: List[Path] = [central_path]
    for p in neighbor_paths:
        if cancel():
            return False
        ok, _ = validate_las_or_laz_with_pdal(p)
        if ok:
            valid_files.append(p)

    if len(valid_files) <= 1:
        shutil.copy2(str(central_path), str(output_path))
        return True

    cmd = [_pdal_exe(), "merge"] + [str(p) for p in valid_files] + [str(output_path)]
    result = run_pdal_command_cancellable(cmd, cancel=cancel)
    if result.returncode != 0:
        log(f"Erreur PDAL merge (code {result.returncode})")
        log("💡 Conseil: réduisez max_workers dans config.json (ex: max_workers=1 ou 2) pour éviter les crashs mémoire.")
        if result.returncode in (3221225477, 3221226505):
            # 0xC0000005 = ACCESS_VIOLATION, 0xC0000409 = STACK_BUFFER_OVERRUN
            log("PDAL a crashé (erreur mémoire).")
        if result.stderr:
            log(result.stderr.strip())
        return False

    ok_out, msg_out = validate_las_or_laz_with_pdal(output_path)
    if not ok_out:
        log(f"Fichier fusionné invalide via PDAL: {output_path.name}")
        if msg_out:
            log(f"PDAL: {msg_out}")
        return False

    return True


@dataclass
class _PreprocessTask:
    """Tâche de prétraitement pour une dalle."""
    index: int
    total: int
    filename: str
    url: str
    tile_name: str
    x: str
    y: str
    central_path: Path
    temp_dir: Path
    merged_dir: Path
    margin_m: int
    dalles_dir: Path


@dataclass
class _PreprocessResult:
    """Résultat du prétraitement d'une dalle."""
    index: int
    tile_name: str
    merged_path: Optional[Path]
    success: bool
    error: Optional[str] = None


def _process_single_tile_preprocess(
    task: _PreprocessTask,
    file_index: Dict[str, Tuple[str, str]],
    log: LogFn,
    cancel: CancelFn,
) -> _PreprocessResult:
    """
    Traite le prétraitement d'une seule dalle (crop voisins + merge).
    Utilise uniquement PDAL (subprocess externe) → thread-safe.
    """
    try:
        if cancel():
            return _PreprocessResult(
                index=task.index,
                tile_name=task.tile_name,
                merged_path=None,
                success=False,
                error="Annulation demandée",
            )

        log(f"[Dalle {task.index}/{task.total}] Début prétraitement: {task.tile_name}")

        neighbors = calculate_neighbor_coordinates(task.x, task.y)

        cropped_neighbors: List[Path] = []
        for voisin_x, voisin_y, place_dalle in neighbors:
            if cancel():
                return _PreprocessResult(
                    index=task.index,
                    tile_name=task.tile_name,
                    merged_path=None,
                    success=False,
                    error="Annulation demandée",
                )

            # Recherche rapide via l'index dict
            coord_key = f"{format_coordinate(voisin_x)}_{format_coordinate(voisin_y)}"
            neigh = file_index.get(coord_key)
            if not neigh:
                continue

            neighbor_file, neighbor_url = neigh
            neighbor_input = task.dalles_dir / neighbor_file
            if not neighbor_input.exists():
                ok, _was_skipped = download_one(
                    neighbor_url,
                    neighbor_file,
                    task.dalles_dir,
                    log=lambda m: log(f"[{task.tile_name}] {m}"),
                    cancel=cancel,
                )
                if not ok or not neighbor_input.exists():
                    continue

            bounds = calculate_crop_bounds(voisin_x, voisin_y, place_dalle, task.margin_m)
            output_file = f"{task.tile_name}_neighbor_{place_dalle}.laz"
            neighbor_output = task.temp_dir / output_file

            if crop_neighbor_tile(
                input_path=neighbor_input,
                output_path=neighbor_output,
                bounds=bounds,
                log=lambda m: log(f"[{task.tile_name}] {m}"),
                cancel=cancel,
            ):
                cropped_neighbors.append(neighbor_output)

        merged_path = task.merged_dir / f"{task.tile_name}_merged.laz"
        if not merge_tiles(
            central_path=task.central_path,
            neighbor_paths=cropped_neighbors,
            output_path=merged_path,
            log=lambda m: log(f"[{task.tile_name}] {m}"),
            cancel=cancel,
        ):
            return _PreprocessResult(
                index=task.index,
                tile_name=task.tile_name,
                merged_path=None,
                success=False,
                error=f"Échec merge pour {task.tile_name}",
            )

        log(f"[Dalle {task.index}/{task.total}] Prétraitement terminé: {task.tile_name}")

        return _PreprocessResult(
            index=task.index,
            tile_name=task.tile_name,
            merged_path=merged_path,
            success=True,
        )

    except Exception as e:
        return _PreprocessResult(
            index=task.index,
            tile_name=task.tile_name,
            merged_path=None,
            success=False,
            error=str(e),
        )


def _build_file_index(sorted_list_file: Path) -> Dict[str, Tuple[str, str]]:
    """
    Construit un index dict {coord_key: (filename, url)} pour recherche O(1).
    coord_key = "XXXX_YYYY" (ex: "0948_6633")
    """
    index: Dict[str, Tuple[str, str]] = {}
    try:
        with sorted_list_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",", 1)
                if len(parts) != 2:
                    continue
                filename, url = parts[0].strip(), parts[1].strip()
                # Extraire les coordonnées du nom de fichier
                # Format attendu: LHD_FXX_XXXX_YYYY_...
                name_parts = filename.replace(".copc.laz", "").replace(".laz", "").split("_")
                if len(name_parts) >= 4:
                    coord_key = f"{name_parts[2]}_{name_parts[3]}"
                    index[coord_key] = (filename, url)
    except Exception:
        pass
    return index


StageFn = Callable[[str], None]


def _default_stage(_: str) -> None:
    return


def prepare_merged_tiles(
    *,
    sorted_list_file: Path,
    dalles_dir: Path,
    output_dir: Path,
    tile_overlap_percent: float,
    log: LogFn = _default_log,
    cancel: CancelFn = _default_cancel,
    max_workers: Optional[int] = None,
    stage: StageFn = _default_stage,
) -> IgnPreprocessResult:
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = temp_dir

    margin_m = max(0, min(999, int(round(1000.0 * float(tile_overlap_percent) / 100.0))))

    # Construire l'index pour recherche rapide des voisins
    file_index = _build_file_index(sorted_list_file)

    with sorted_list_file.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    total = len(lines)
    
    # Préparer les tâches
    tasks: List[_PreprocessTask] = []
    for idx, line in enumerate(lines, start=1):
        parts = line.split(",", 1)
        if len(parts) != 2:
            continue

        filename, url = parts[0].strip(), parts[1].strip()
        base_tile_name = filename.replace(".copc.laz", "").replace(".laz", "")

        central_path = dalles_dir / filename
        if not central_path.exists():
            raise FileNotFoundError(f"Fichier central introuvable: {central_path}")

        x, y = _extract_coordinates(filename, dalles_dir=dalles_dir, log=log, cancel=cancel)
        tile_name = base_tile_name
        if len(base_tile_name.split("_")) < 4:
            tile_name = f"LHD_FXX_{x}_{y}"

        tasks.append(_PreprocessTask(
            index=idx,
            total=total,
            filename=filename,
            url=url,
            tile_name=tile_name,
            x=x,
            y=y,
            central_path=central_path,
            temp_dir=temp_dir,
            merged_dir=merged_dir,
            margin_m=margin_m,
            dalles_dir=dalles_dir,
        ))

    # Configuration parallélisation
    # PDAL est un subprocess externe → thread-safe
    if max_workers is None:
        max_workers = min(4, max(1, os.cpu_count() or 1))
    
    log(f"Prétraitement parallèle: {max_workers} worker(s) pour {total} dalle(s)")

    # Lock pour synchroniser les logs et le compteur
    log_lock = threading.Lock()
    completed_count = [0]
    
    def thread_safe_log(msg: str) -> None:
        with log_lock:
            log(msg)

    def update_stage_count() -> None:
        with log_lock:
            completed_count[0] += 1
            stage(f"Fusion dalle {completed_count[0]}/{total}")

    # Exécution parallèle
    merged_files: List[Path] = []
    results_by_index: Dict[int, _PreprocessResult] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                _process_single_tile_preprocess,
                task,
                file_index,
                thread_safe_log,
                cancel,
            ): task
            for task in tasks
        }

        for future in as_completed(future_to_task):
            if cancel():
                executor.shutdown(wait=False, cancel_futures=True)
                break

            task = future_to_task[future]
            try:
                result = future.result()
                results_by_index[result.index] = result
                update_stage_count()
                
                if not result.success:
                    thread_safe_log(f"Échec prétraitement dalle {result.tile_name}: {result.error}")
            except Exception as e:
                thread_safe_log(f"Exception prétraitement dalle {task.tile_name}: {e}")

    # Reconstituer la liste dans l'ordre original
    for idx in sorted(results_by_index.keys()):
        result = results_by_index[idx]
        if result.success and result.merged_path is not None:
            merged_files.append(result.merged_path)
        elif result.error and "Annulation" not in (result.error or ""):
            raise RuntimeError(f"[Dalle {result.index}/{total}] {result.tile_name}: {result.error}")

    return IgnPreprocessResult(merged_dir=merged_dir, temp_dir=temp_dir, merged_files=merged_files)


def _extract_coordinates(
    filename: str,
    *,
    dalles_dir: Path,
    log: LogFn = _default_log,
    cancel: CancelFn = _default_cancel,
) -> Tuple[str, str]:
    inferred = infer_xy_from_file(dalles_dir / filename, cancel=cancel)
    if inferred is not None:
        x_str = f"{int(inferred.x_km):04d}"
        y_str = f"{int(inferred.y_km):04d}"
        log(f"Coordonnées inférées via metadata pour {filename}: x={x_str}, y={y_str}")
        return x_str, y_str

    xy = extract_xy_from_filename(filename)
    if xy is not None:
        return f"{int(xy.x_km):04d}", f"{int(xy.y_km):04d}"

    raise ValueError(f"Impossible d'extraire / inférer les coordonnées de: {filename}")
