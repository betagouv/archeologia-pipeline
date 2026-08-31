"""
Gestion du runner ONNX externe (subprocess compilé via PyInstaller).

Extrait de runner.py pour améliorer la lisibilité.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

try:
    from typing import TypedDict
except ImportError:  # Python < 3.8
    from typing_extensions import TypedDict

from ..cancellation import PipelineCancelled
from ..geo_utils import write_world_file
from ..subprocess_utils import subprocess_kwargs_no_window
from ..types import LogFn, CancelCheckFn


ImageProgressFn = Callable[[int, int, str], None]
"""Callback invoqué quand le runner termine le traitement d'une image.

Signature : ``(index_1based, total, image_name)``. Sert au stepper UI
à actualiser la sous-progression d'un run CV.
"""

TileProgressFn = Callable[[int, int], None]
"""Callback invoqué à chaque ligne « SAHI: X/Y tuiles traitées » du runner.

Signature : ``(current, total)``. Sous-progression AU SEIN d'une image :
sur une grande dalle (centaines de tuiles SAHI, plusieurs minutes CPU),
c'est le seul signal qui bouge entre deux ``ImageProgressFn``.
"""

# « RF-DETR Seg SAHI: 10/144 tuiles traitées » / « SegFormer SAHI: 3/36 tuiles… »
# L'annonce du total (« SAHI: 144 tuiles », sans slash) ne matche pas.
_TILE_PROGRESS_RE = re.compile(r"SAHI: (\d+)/(\d+) tuiles")


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
            if log:
                # Traçabilité du binaire (audit 2026-08-31) : sans stamp, un exe
                # périmé a tourné 2 mois et demi sans que rien ne le dise.
                info_path = candidate.parent / "build_info.json"
                try:
                    import json as _json
                    _info = _json.loads(info_path.read_text(encoding="utf-8"))
                    log(f"Computer Vision: runner ONNX build {_info.get('date', '?')} "
                        f"(commit {_info.get('commit', '?')})")
                except Exception:
                    log("Computer Vision: runner ONNX SANS build_info.json — âge "
                        "inconnu, recompiler via dev/runner_onnx/build.py")
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
    image_progress: Optional[ImageProgressFn] = None,
    tile_progress: Optional[TileProgressFn] = None,
) -> Optional[int]:
    """Parse une ligne de stdout du runner externe et la log de façon lisible.

    ``image_progress``, si fourni, est invoqué pour chaque ligne
    ``progress=N/TOTAL`` quel que soit le status (processing, done,
    skipped). Permet au stepper UI de suivre la sous-progression sans
    parser à nouveau les logs textuels.

    ``tile_progress``, si fourni, est invoqué pour chaque ligne
    « SAHI: X/Y tuiles traitées » (sous-progression intra-image, cf.
    :data:`TileProgressFn`). La ligne reste relayée au log à l'identique.

    Retourne le ``total_detections`` de la ligne ``summary:`` (None pour
    toute autre ligne) — remonté jusqu'au narrateur pour annoncer
    « Détection terminée : N zones ».
    """
    if tile_progress is not None:
        m = _TILE_PROGRESS_RE.search(line)
        if m is not None:
            try:
                tile_progress(int(m.group(1)), int(m.group(2)))
            except Exception:
                pass

    if line.startswith("progress="):
        try:
            parts = line.split()
            progress_part = parts[0].split("=")[1]
            current, total = progress_part.split("/")
            image_name = parts[1].split("=")[1] if len(parts) > 1 else ""
            status = parts[2].split("=")[1] if len(parts) > 2 else ""

            # Le runner émet 2 lignes ``progress=`` par image
            # (status=processing au début, status=done à la fin). On ne
            # remonte qu'une seule fois la progression à l'UI — au début
            # de l'image, pour que le compteur "Image i/N" apparaisse
            # immédiatement. Les images déjà en cache sortent en
            # status=skipped — on remonte aussi pour faire avancer le
            # compteur.
            if image_progress is not None and status in ("processing", "skipped"):
                try:
                    image_progress(int(current), int(total), image_name)
                except Exception:
                    pass

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
            td = info.get("total_detections")
            if td is not None:
                try:
                    return int(td)
                except ValueError:
                    pass
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
        log("Computer Vision: Légende créée")
    else:
        # Relayer les lignes non reconnues (logs internes, debug, etc.)
        log(f"[cv_runner] {line}")
    return None


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
    image_progress: Optional[ImageProgressFn] = None,
    tile_progress: Optional[TileProgressFn] = None,
) -> Optional[int]:
    """
    Exécute le runner ONNX externe via subprocess et parse sa sortie en temps réel.

    Returns:
        Le ``total_detections`` annoncé par la ligne ``summary:`` du runner,
        ou None si elle est absente (ancien binaire, sortie tronquée).

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
        # stderr FUSIONNÉ dans stdout (AUDIT v2 CVPROC-01) : avec deux tubes
        # séparés, stderr n'était drainé qu'au communicate() final → dès ~64 Ko
        # de tracebacks côté binaire (modèle incompatible → traceback PAR
        # image), le binaire bloquait sur write(stderr) et le parent sur
        # readline(stdout) : deadlock définitif, Annuler inopérant. Fusionnées,
        # les lignes stderr sont relayées en direct par _parse_runner_stdout
        # (repli « [cv_runner] … »).
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **subprocess_kwargs_no_window()
        )

        cancelled = False
        total_detections: Optional[int] = None
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
                parsed = _parse_runner_stdout(
                    line, log, image_progress=image_progress, tile_progress=tile_progress
                )
                if parsed is not None:
                    total_detections = parsed

        if cancelled:
            raise PipelineCancelled()

        # stdout est à EOF (boucle ci-dessus) et stderr est fusionné dedans :
        # communicate() ne fait plus que récolter le code retour.
        process.communicate()

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
        return total_detections
    finally:
        try:
            cfg_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
