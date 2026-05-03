"""
Gestion du runner ONNX externe (subprocess compilé via PyInstaller).

Extrait de runner.py pour améliorer la lisibilité.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from typing import TypedDict
except ImportError:  # Python < 3.8
    from typing_extensions import TypedDict

from ..geo_utils import write_world_file
from ..subprocess_utils import subprocess_kwargs_no_window
from ..types import LogFn, CancelCheckFn


class RunnerPayload(TypedDict, total=False):
    """Contrat JSON envoyé au runner ONNX externe via --config."""
    jpg_dir: str
    target_rvt: str
    rvt_base_dir: Optional[str]
    cv_config: Dict[str, Any]
    single_jpg: Optional[str]
    run_shapefile_dedup: bool
    tif_transform_data: Dict[str, Tuple[float, float, float, float]]
    global_color_map: Dict[str, int]


def find_external_cv_runner(log: Optional[LogFn] = None) -> Optional[Path]:
    """
    Trouve le runner ONNX externe.

    Args:
        log: Fonction de logging

    Returns:
        Chemin vers le runner ONNX ou None si non trouvé
    """
    plugin_root = Path(__file__).resolve().parents[3]

    if os.name == "nt":
        candidate = plugin_root / "data" / "third_party" / "cv_runner_onnx" / "windows" / "cv_runner_onnx.exe"
    else:
        candidate = plugin_root / "data" / "third_party" / "cv_runner_onnx" / "linux" / "cv_runner_onnx"

    try:
        if candidate.exists() and candidate.is_file():
            return candidate
        elif log:
            log(f"Computer Vision: runner ONNX non trouvé à {candidate}")
    except Exception as e:
        if log:
            log(f"Computer Vision: erreur vérification runner ONNX {candidate}: {e}")

    return None


def _parse_runner_stdout(
    line: str,
    log: LogFn,
) -> None:
    """Parse une ligne de stdout du runner externe et la log de façon lisible."""
    if line.startswith("progress="):
        try:
            parts = line.split()
            progress_part = parts[0].split("=")[1]
            current, total = progress_part.split("/")
            image_name = parts[1].split("=")[1] if len(parts) > 1 else ""
            status = parts[2].split("=")[1] if len(parts) > 2 else ""

            if status == "processing":
                log(f"Computer Vision: [{current}/{total}] Analyse de {image_name}...")
            elif status == "done":
                dets = parts[3].split("=")[1] if len(parts) > 3 else "0"
                mode = ""
                for p in parts[4:]:
                    if p.startswith("mode="):
                        mode = p.split("=")[1]
                        break
                mode_str = f" [{mode.upper()}]" if mode else ""
                log(f"Computer Vision: [{current}/{total}] {image_name} -> {dets} détection(s){mode_str}")
            elif status == "skipped":
                log(f"Computer Vision: [{current}/{total}] {image_name} (déjà traité)")
            else:
                log(f"[cv_runner] {line}")
        except Exception:
            log(f"[cv_runner] {line}")
    elif line.startswith("summary:"):
        try:
            parts = line.replace("summary:", "").strip().split()
            info = {p.split("=")[0]: p.split("=")[1] for p in parts}
            log(f"Computer Vision: Terminé - {info.get('success', '?')} images traitées, {info.get('total_detections', '?')} détections au total")
        except Exception:
            log(f"[cv_runner] {line}")
    elif line.startswith("images="):
        total_imgs = line.split("=")[1]
        log(f"Computer Vision: {total_imgs} images à analyser")
    elif line.startswith("model_path="):
        model = Path(line.split("=")[1]).name
        log(f"Computer Vision: Modèle -> {model}")
    elif line.startswith("class_names="):
        log(f"Computer Vision: Classes -> {line.split('=')[1]}")
    elif "ERROR" in line or "error" in line.lower():
        log(f"[cv_runner] {line}")
    elif line.startswith("seg_params="):
        try:
            params = line.split("=", 1)[1]
            log(f"Computer Vision: Paramètres segmentation -> {params}")
        except Exception:
            log(f"[cv_runner] {line}")
    elif line.startswith("legend_created="):
        log(f"Computer Vision: Légende créée")
    else:
        # Relayer les lignes non reconnues (logs internes, debug, etc.)
        log(f"[cv_runner] {line}")


def run_external_cv_runner(
    *,
    ext: Path,
    jpg_dir: Path,
    target_rvt: str,
    rvt_base_dir: Optional[Path],
    detection_dir: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    cv_config: Dict[str, Any],
    single_jpg: Optional[Path],
    run_shapefile_dedup: bool,
    tif_transform_data: Optional[Dict[str, Tuple[float, float, float, float]]],
    global_color_map: Optional[Dict[str, int]] = None,
    log: LogFn = lambda _: None,
    cancel_check: Optional[CancelCheckFn] = None,
) -> None:
    """
    Exécute le runner ONNX externe via subprocess et parse sa sortie en temps réel.

    Raises:
        RuntimeError: si le runner échoue ou est annulé.
    """
    payload: RunnerPayload = {
        "jpg_dir": str(jpg_dir),
        "target_rvt": target_rvt,
        "rvt_base_dir": str(rvt_base_dir) if rvt_base_dir else None,
        "detection_dir": str(detection_dir) if detection_dir else None,
        "raw_dir": str(raw_dir) if raw_dir else None,
        "cv_config": cv_config,
        "single_jpg": str(single_jpg) if single_jpg else None,
        "run_shapefile_dedup": bool(run_shapefile_dedup),
        "tif_transform_data": tif_transform_data or {},
        "global_color_map": global_color_map or {},
    }

    if bool((cv_config or {}).get("export_runner_config", False)):
        try:
            base = rvt_base_dir or jpg_dir.parent
            out_dir = Path(base) / "cv_runner_configs"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            suffix = single_jpg.stem if single_jpg else "folder"
            out_path = out_dir / f"cv_runner_{target_rvt}_{suffix}_{ts}.json"
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            log(f"Computer Vision: config runner exportée -> {out_path}")
        except Exception as e:
            log(f"Computer Vision: impossible d'exporter la config runner: {e}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(payload, f)
        cfg_path = Path(f.name)

    try:
        cmd = [str(ext), "--config", str(cfg_path)]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **subprocess_kwargs_no_window()
        )

        cancelled = False
        if process.stdout:
            for line in process.stdout:
                if cancel_check and cancel_check():
                    log("Computer Vision: Annulation demandée, arrêt du processus...")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    cancelled = True
                    break

                line = line.rstrip()
                if not line:
                    continue
                _parse_runner_stdout(line, log)

        if cancelled:
            raise RuntimeError("Computer Vision: Annulé par l'utilisateur")

        _, stderr = process.communicate()
        if stderr:
            for line in stderr.splitlines():
                if line.strip():
                    log(f"[cv_runner][stderr] {line}")

        if process.returncode != 0:
            raise RuntimeError(f"cv_runner failed (code={process.returncode})")

        # Déplacer les JSON/TXT du dossier source (jpg_dir) vers raw_dir
        # (fallback pour anciens binaires compilés qui ne connaissent pas raw_dir)
        if raw_dir is not None and raw_dir != jpg_dir:
            import shutil
            raw_dir.mkdir(parents=True, exist_ok=True)
            for label_file in list(jpg_dir.glob("*.txt")) + list(jpg_dir.glob("*.json")):
                dest = raw_dir / label_file.name
                if dest.exists():
                    continue  # Le nouveau runner a déjà écrit directement ici
                try:
                    shutil.move(str(label_file), str(dest))
                except Exception as e:
                    log(f"Computer Vision: impossible de déplacer {label_file.name} vers raw_detections: {e}")

        # Créer les fichiers world pour les images annotées générées par le cv_runner
        generate_annotated = bool((cv_config or {}).get("generate_annotated_images", False))
        if generate_annotated and tif_transform_data:
            base = detection_dir or rvt_base_dir or jpg_dir.parent
            annotated_dir = Path(base) / "annotated_images"
            if annotated_dir.exists():
                for annotated_img in annotated_dir.glob("*.png"):
                    stem = annotated_img.stem
                    if stem.endswith("_detections"):
                        original_stem = stem[:-11]
                    else:
                        original_stem = stem
                    transform = tif_transform_data.get(original_stem)
                    if transform and len(transform) == 4:
                        pixel_width, pixel_height, x_origin, y_origin = transform
                        world_path = write_world_file(
                            annotated_img, pixel_width, pixel_height, x_origin, y_origin
                        )
                        if world_path:
                            log(f"Fichier world créé: {world_path.name}")
    finally:
        try:
            cfg_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
