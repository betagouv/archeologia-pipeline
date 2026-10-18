"""
CLI du runner ONNX unifié pour l'inférence CV.
Ce runner utilise ONNX Runtime et supporte les modèles YOLO et RF-DETR exportés en ONNX.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Forcer l'encodage UTF-8 sur stdout/stderr pour Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Forcer UTF-8 pour GDAL/fiona (évite les SystemError avec noms de classes accentués)
import os as _os
_os.environ.setdefault("GDAL_FILENAME_IS_UTF8", "YES")
_os.environ.setdefault("SHAPE_ENCODING", "UTF-8")
_os.environ.setdefault("PYTHONIOENCODING", "utf-8")
_os.environ.setdefault("PGCLIENTENCODING", "UTF-8")
# Empêcher GDAL de charger les plugins DLL d'OSGeo4W (incompatibles avec le GDAL bundlé)
_os.environ["GDAL_DRIVER_PATH"] = "disable"


def _try_import_version(mod_name: str) -> str:
    try:
        mod = __import__(mod_name)
        v = getattr(mod, "__version__", None)
        return str(v) if v is not None else "unknown"
    except Exception as e:
        return f"MISSING ({e})"


def _try_import_ok(mod_name: str) -> str:
    try:
        __import__(mod_name)
        return "OK"
    except Exception as e:
        return f"MISSING ({e})"


def _print(msg: str) -> None:
    sys.stdout.write(str(msg) + "\n")
    sys.stdout.flush()  # Flush immédiatement pour le streaming en temps réel


def _resource_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parent


def _plugin_root() -> Path:
    """Retourne la racine du plugin."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parent.parent


def _ensure_src_on_sys_path() -> None:
    root = _resource_root()
    plugin_root = _plugin_root()

    candidates = [
        root / "src",
        plugin_root / "src",
    ]
    for c in candidates:
        try:
            if c.exists() and c.is_dir():
                if str(c) not in sys.path:
                    sys.path.insert(0, str(c))
                return
        except Exception:
            continue

    dev_src = plugin_root / "src"
    if dev_src.exists() and dev_src.is_dir() and str(dev_src) not in sys.path:
        sys.path.insert(0, str(dev_src))


def _resolve_model_path(cv_config: Dict[str, Any]) -> Path:
    """Résout le chemin vers le modèle ONNX."""
    selected_model = str((cv_config or {}).get("selected_model", "")).strip()
    if not selected_model:
        raise ValueError("Computer Vision activée mais aucun modèle sélectionné")

    model_path = Path(selected_model)
    
    # Si c'est un chemin vers un fichier .pt, chercher le .onnx correspondant
    if model_path.suffix.lower() == ".pt":
        onnx_path = model_path.with_suffix(".onnx")
        if onnx_path.exists():
            return onnx_path
        # Sinon, erreur explicite
        raise FileNotFoundError(
            f"Modèle ONNX non trouvé: {onnx_path}\n"
            f"Le fichier .pt existe mais pas le .onnx. "
            f"Exportez le modèle avec: python runner_onnx/export_to_onnx.py --model {model_path} --output {onnx_path}"
        )
    
    # Si c'est un chemin absolu vers un fichier ONNX existant
    if model_path.suffix.lower() == ".onnx" and model_path.exists() and model_path.is_file():
        return model_path
    
    # Chercher dans le dossier models
    models_dir = Path((cv_config or {}).get("models_dir", "models"))
    
    # Essayer différents chemins
    candidates = [
        models_dir / selected_model / "weights" / "best.onnx",
        models_dir / selected_model / "best.onnx",
        models_dir / f"{selected_model}.onnx",
        models_dir / selected_model,
    ]
    
    for candidate in candidates:
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".onnx":
            return candidate
    
    raise FileNotFoundError(
        f"Modèle ONNX non trouvé pour: {selected_model}\n"
        f"Chemins testés: {[str(c) for c in candidates]}"
    )


def _iter_images(jpg_dir: Path, single_jpg: Optional[Path], scan_all: bool) -> list[Path]:
    if single_jpg is not None:
        return [single_jpg]
    if scan_all:
        return sorted(jpg_dir.glob("*.png"))
    return sorted(jpg_dir.glob("*.png"))[:1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[cv_runner_onnx][%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    cfg_path = Path(args.config)
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))

    jpg_dir = Path(payload["jpg_dir"]).resolve()
    target_rvt = str(payload.get("target_rvt", "LD"))
    rvt_base_dir_raw = payload.get("rvt_base_dir")
    rvt_base_dir = Path(rvt_base_dir_raw).resolve() if rvt_base_dir_raw else jpg_dir.parent
    detection_dir_raw = payload.get("detection_dir")
    detection_dir = Path(detection_dir_raw).resolve() if detection_dir_raw else rvt_base_dir
    raw_dir_raw = payload.get("raw_dir")
    raw_dir = Path(raw_dir_raw).resolve() if raw_dir_raw else jpg_dir

    cv_config: Dict[str, Any] = payload.get("cv_config") or {}
    tif_transform_data: Dict[str, Any] = payload.get("tif_transform_data") or {}
    run_shapefile_dedup = bool(payload.get("run_shapefile_dedup", True))
    global_color_map: Dict[str, Any] = payload.get("global_color_map") or {}

    single_jpg_raw = payload.get("single_jpg")
    single_jpg = Path(single_jpg_raw).resolve() if single_jpg_raw else None

    enabled = bool(cv_config.get("enabled", False))
    if not enabled:
        _print("CV disabled")
        return 0

    # Afficher les versions des dépendances
    _print(f"dep:onnxruntime={_try_import_version('onnxruntime')}")
    _print(f"dep:numpy={_try_import_version('numpy')}")
    _print(f"dep:pillow={_try_import_version('PIL')}")
    _print(f"dep:sahi_lite=internal")

    generate_annotated_images = bool(cv_config.get("generate_annotated_images", False))
    generate_shapefiles = bool(cv_config.get("generate_shapefiles", False))

    # Vérifier les dépendances critiques
    try:
        import onnxruntime  # noqa: F401
    except Exception as e:
        _print(f"ERROR: onnxruntime missing ({e})")
        return 2

    try:
        import numpy  # noqa: F401
    except Exception as e:
        _print(f"ERROR: numpy missing ({e})")
        return 2

    annotated_output_dir = detection_dir / "annotated_images"
    shapefile_output_dir = detection_dir / "shapefiles"

    if generate_annotated_images:
        annotated_output_dir.mkdir(parents=True, exist_ok=True)
    if generate_shapefiles:
        shapefile_output_dir.mkdir(parents=True, exist_ok=True)

    _ensure_src_on_sys_path()

    from pipeline.cv import appareil as app_mod
    from pipeline.cv import computer_vision_onnx as cv_mod
    from pipeline.cv.cv_output import get_detection_output_path
    from pipeline.cv.runner_cache import image_deja_traitee

    shp_mod = None
    if generate_shapefiles and run_shapefile_dedup:
        try:
            from pipeline.cv import conversion_shp as shp_mod  # type: ignore
        except Exception as e:
            _print(f"WARN: shapefile deps not available ({e})")

    # Résoudre le chemin du modèle
    try:
        model_path = _resolve_model_path(cv_config)
    except Exception as e:
        _print(f"ERROR: {e}")
        return 1

    _print(f"model_type=onnx")
    _print(f"model_path={model_path}")
    
    # Charger les noms de classes depuis le dossier du modèle
    class_names = None
    class_colors = None
    try:
        # 2026-08-31 : l'ancien import visait load_class_colors_from_model, qui
        # n'a JAMAIS existé — l'import échouait en bloc, class_names restait None
        # (pas de légende, pas de mapping des seuils par classe). ModelProfile est
        # la source réelle (args.yaml.class_colors + classes.txt), comme le
        # fallback Python (runner_inference).
        from pipeline.cv.model_profile import ModelProfile
        _profile = ModelProfile.load(model_path)
        class_names = list(_profile.class_names) if _profile.class_names else None
        class_colors = list(_profile.class_colors) if _profile.class_colors else None
        if class_names:
            _print(f"class_names={class_names}")
        _print(f"class_colors_loaded={class_colors}")  # Toujours afficher, même si None
    except Exception as e:
        _print(f"WARN: impossible de charger les noms/couleurs de classes: {e}")

    # Seuils par classe (T1 audit 2026-08-31) : le payload les porte déjà dans
    # cv_config, mais le binaire ne les appliquait PAS — les seuils F1-max mesurés
    # ne valaient que sur le fallback Python. Même helper que runner_inference.
    confidence_per_class = None
    try:
        from pipeline.cv.class_utils import confidence_per_class_ids
        confidence_per_class = confidence_per_class_ids(cv_config, class_names, _print)
    except Exception as e:
        _print(f"WARN: seuils par classe non résolus: {e}")
    
    # Créer le fichier légende dans annotated_images
    if generate_annotated_images and class_names and annotated_output_dir:
        try:
            from pipeline.cv.cv_output import save_legend_file
            save_legend_file(str(annotated_output_dir), class_names, class_colors)
            _print(f"legend_created={annotated_output_dir}")
        except Exception as e:
            _print(f"WARN: impossible de créer la légende: {e}")

    # Défauts UNIFIÉS (audit 2026-08-31) : le binaire slicait à 750 quand tout le
    # reste de la chaîne disait 640, et chaque chemin d'entrée avait son seuil.
    from pipeline.cv.model_config import (
        DEFAULT_CONFIDENCE, DEFAULT_IOU, DEFAULT_SAHI_OVERLAP, DEFAULT_SAHI_SLICE,
    )
    confidence_threshold = float(cv_config.get("confidence_threshold", DEFAULT_CONFIDENCE))
    iou_threshold = float(cv_config.get("iou_threshold", DEFAULT_IOU))

    sahi_cfg = cv_config.get("sahi", {}) if isinstance(cv_config.get("sahi", {}), dict) else {}
    slice_height = int(sahi_cfg.get("slice_height", DEFAULT_SAHI_SLICE))
    slice_width = int(sahi_cfg.get("slice_width", DEFAULT_SAHI_SLICE))
    overlap_ratio = float(sahi_cfg.get("overlap_ratio", DEFAULT_SAHI_OVERLAP))

    # Charger les métadonnées du modèle pour les paramètres de segmentation
    model_meta = {}
    meta_path = model_path.with_suffix('.json')
    if meta_path.exists():
        try:
            model_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    is_segmentation = model_meta.get("model_type") in ("segformer", "smp") or model_meta.get("task") in ("semantic_segmentation", "instance_segmentation")
    use_sahi_meta = model_meta.get("use_sahi", True)

    if is_segmentation:
        bg_bias = float(model_meta.get("bg_bias", 0.0))
        model_conf = model_meta.get("confidence_threshold")
        effective_conf = float(model_conf) if model_conf is not None else confidence_threshold
        _print(f"seg_params=confidence_threshold={effective_conf} bg_bias={bg_bias} use_sahi={use_sahi_meta}")

    scan_all = bool(cv_config.get("scan_all", False))
    jpg_files = _iter_images(jpg_dir=jpg_dir, single_jpg=single_jpg, scan_all=scan_all)
    jpg_files = [p for p in jpg_files if p.exists()]

    _print(f"jpg_dir={jpg_dir}")
    _print(f"target_rvt={target_rvt}")
    _print(f"rvt_base_dir={rvt_base_dir}")
    
    total_images = len(jpg_files)
    _print(f"images={total_images}")

    success_count = 0
    processed_count = 0
    skipped_already_processed = 0
    total_detections = 0
    # Transmis par --config (cv_config du run) : sans lui, le binaire sautait
    # toute image au cache même quand le plugin demandait de la refaire.
    force_reprocess = bool(cv_config.get("force_reprocess", False))
    # Session ONNX créée une fois, à la première image à inférer : un run servi
    # entièrement par le cache n'en paie pas la création (lot 2).
    onnx_session = None
    session_tentee = False

    # Carte graphique (0.15.0, vague 3) : bloc posé par le plugin
    # (app/services/carte_graphique) — sonde sur la première image inférée, ou carte
    # déjà retenue sur ce poste (device_id). Windows seulement (DirectML).
    cg = cv_config.get("carte_graphique") if isinstance(cv_config.get("carte_graphique"), dict) else {}
    cg_demande = bool(cg.get("demande")) and sys.platform == "win32"
    sonde_a_faire = cg_demande and cg.get("device_id") is None
    courant = {"idx": 0}

    def _ouvrir_carte(device_id: int):
        s = cv_mod._load_onnx_model(str(model_path), providers=app_mod.providers_dml(device_id))
        return s if app_mod.session_sur_carte(s[0]) else None

    def _ouvrir_processeur():
        return cv_mod._load_onnx_model(str(model_path), providers=[app_mod.FOURNISSEUR_CPU])

    def _sur_repli(motif: str) -> None:
        _print(app_mod.ligne(app_mod.decision_repli(motif, courant["idx"])))

    def _detections(path: Path) -> list:
        try:
            return list((json.loads(path.read_text(encoding="utf-8")) or {}).get("detections") or [])
        except (OSError, ValueError):
            return []

    # Seuil que juge la sonde : celui du sidecar pour la segmentation sémantique (run_onnx_inference
    # remplace alors le seuil reçu), sinon le seuil par classe puis le seuil global.
    seuil_sonde = float(effective_conf) if is_segmentation else float(confidence_threshold)

    def _seuil_de(d: dict) -> float:
        if is_segmentation:
            return seuil_sonde
        try:
            return float((confidence_per_class or {}).get(int(d.get("class_id", -1)), confidence_threshold))
        except (TypeError, ValueError):
            return float(confidence_threshold)

    def _carte_retenue(session_processeur):
        """Carte déjà retenue sur ce poste : on l'ouvre sans sonder ; indisponible → processeur."""
        carte = str(cg.get("carte") or f"carte {cg['device_id']}")
        try:
            s = _ouvrir_carte(int(cg["device_id"]))
            motif = "le moteur est retombé sur le processeur"
        except Exception as e:  # noqa: BLE001 — pilote, mémoire, modèle refusé
            s, motif = None, f"{type(e).__name__}: {app_mod.court(e)}"
        if s is None:
            _print(app_mod.ligne(app_mod.decision_indisponible(carte, motif)))
            return session_processeur
        _print(app_mod.ligne(app_mod.decision_memorisee(int(cg["device_id"]), carte)))
        return app_mod.avec_repli(s, _ouvrir_processeur, _sur_repli)

    def _sonder(jpg_file: Path, image_name: str, secondes_processeur: float, session_processeur):
        """Rejoue la première image (déjà inférée sur le processeur, résultat gardé) sur
        chaque carte, dans un dossier provisoire ; rend la session à garder."""
        tmp = Path(tempfile.mkdtemp(prefix="sonde_carte_"))

        def inferer(sess):
            # Même chaîne que l'image processeur (le temps processeur comprend, lui, l'image
            # annotée si elle est demandée : écart faible, dans la marge des 80 %).
            t0 = time.perf_counter()
            cv_mod.run_onnx_inference(
                image_path=str(jpg_file), model_path=str(model_path), output_path=str(tmp / jpg_file.name),
                confidence_threshold=confidence_threshold, confidence_per_class=confidence_per_class,
                iou_threshold=iou_threshold, slice_height=slice_height, slice_width=slice_width,
                overlap_ratio=overlap_ratio, generate_annotated_images=False, jpg_folder_path=str(tmp),
                return_count=True, class_names=class_names, class_colors=class_colors, onnx_session=sess,
            )
            secondes = time.perf_counter() - t0
            sortie = tmp / f"{image_name}.json"
            if not sortie.exists():   # run_onnx_inference avale ses exceptions : sans JSON, la carte a planté
                raise RuntimeError("inférence sans résultat sur la carte (erreur du pilote, voir le journal)")
            return secondes, _detections(sortie)

        try:
            decision, session = app_mod.sonder(
                cg.get("adaptateurs") or [], secondes_processeur, _detections(raw_dir / f"{image_name}.json"),
                _seuil_de, _ouvrir_carte, inferer, lambda m: _print(f"[carte graphique] {m}"),
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        _print(app_mod.ligne(decision))
        if session is None:
            return session_processeur
        return app_mod.avec_repli(session, _ouvrir_processeur, _sur_repli)

    for idx, jpg_file in enumerate(jpg_files, 1):
        image_name = jpg_file.stem
        courant["idx"] = idx

        detection_output_path = get_detection_output_path(
            str(jpg_file),
            target_rvt,
            str(annotated_output_dir) if generate_annotated_images else None,
        )

        if image_deja_traitee(raw_dir, image_name, force_reprocess):
            skipped_already_processed += 1
            _print(f"progress={idx}/{total_images} image={jpg_file.name} status=skipped")
            continue

        processed_count += 1
        _print(f"progress={idx}/{total_images} image={jpg_file.name} status=processing")

        if not session_tentee:
            session_tentee = True
            onnx_session = cv_mod.charger_session_partagee(str(model_path), _print)
            if cg_demande and cg.get("device_id") is not None and onnx_session is not None:
                onnx_session = _carte_retenue(onnx_session)

        t_image = time.perf_counter()
        t_image_mur = time.time()
        try:
            ok, num_dets = cv_mod.run_onnx_inference(
                image_path=str(jpg_file),
                model_path=str(model_path),
                output_path=str(detection_output_path),
                confidence_threshold=confidence_threshold,
                confidence_per_class=confidence_per_class,
                iou_threshold=iou_threshold,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_ratio=overlap_ratio,
                generate_annotated_images=generate_annotated_images,
                annotated_output_dir=str(annotated_output_dir) if generate_annotated_images else None,
                jpg_folder_path=str(raw_dir),
                return_count=True,
                class_names=class_names,
                class_colors=class_colors,
                onnx_session=onnx_session,
            )
        except Exception as e:
            ok = False
            num_dets = 0
            _print(f"ERROR: run_onnx_inference failed for {jpg_file.name}: {e}")
            import traceback
            traceback.print_exc()
        secondes_image = time.perf_counter() - t_image

        # Sonde de la carte graphique : après la première image inférée sur le processeur, qu'elle
        # ait ou non des détections (« ok » est faux sans détection) — l'inférence a abouti si
        # son JSON vient d'être écrit (run_onnx_inference avale ses exceptions).
        if sonde_a_faire and onnx_session is not None:
            sortie_cpu = raw_dir / f"{image_name}.json"
            if sortie_cpu.exists() and sortie_cpu.stat().st_mtime >= t_image_mur - 1:
                sonde_a_faire = False
                onnx_session = _sonder(jpg_file, image_name, secondes_image, onnx_session)

        # Déterminer le mode d'inférence pour le log
        infer_mode = ""
        if is_segmentation:
            try:
                from PIL import Image as _PILImg
                with _PILImg.open(str(jpg_file)) as _tmp:
                    _iw, _ih = _tmp.size
                _mw = model_meta.get("image_size", 640)
                infer_mode = "sahi" if (use_sahi_meta and (_iw > _mw * 1.5 or _ih > _mw * 1.5)) else "direct"
            except Exception:
                infer_mode = "sahi" if use_sahi_meta else "direct"

        mode_suffix = f" mode={infer_mode}" if infer_mode else ""
        if ok:
            success_count += 1
            total_detections += num_dets
            _print(f"progress={idx}/{total_images} image={jpg_file.name} status=done detections={num_dets}{mode_suffix}")
        else:
            _print(f"progress={idx}/{total_images} image={jpg_file.name} status=done detections=0{mode_suffix}")

    if cg_demande and sonde_a_faire and processed_count > 0:
        # Option demandée, images traitées, mais aucune inférence processeur aboutie pour sonder.
        _print(app_mod.ligne(app_mod.decision_non_sondee(
            "aucune inférence aboutie sur le processeur dans ce traitement pour la comparer")))

    _print(f"summary: success={success_count} processed={processed_count} skipped={skipped_already_processed} total_detections={total_detections}")

    # Génération des shapefiles
    if generate_shapefiles and run_shapefile_dedup and shp_mod is not None:
        out_shp = shapefile_output_dir / f"detections_{target_rvt}.shp"
        create_fn = getattr(shp_mod, "create_shapefile_from_detections", None)
        if callable(create_fn):
            try:
                _raw_classes = cv_config.get("selected_classes")
                _selected_classes = _raw_classes if isinstance(_raw_classes, list) else None
                create_fn(
                    labels_dir=str(jpg_dir),
                    output_shapefile=str(out_shp),
                    tif_transform_data=tif_transform_data,
                    crs="EPSG:2154",
                    temp_dir=None,
                    class_names=class_names,
                    selected_classes=_selected_classes,
                    class_colors=class_colors,
                    global_color_map=global_color_map if global_color_map else None,
                )
            except Exception as e:
                _print(f"WARN: shapefile creation failed: {e}")

        dedup_fn = getattr(shp_mod, "deduplicate_shapefiles_final", None)
        if callable(dedup_fn):
            shp_paths = [str(p) for p in shapefile_output_dir.glob("*.shp")]
            if shp_paths:
                try:
                    min_area_m2 = float(cv_config.get("min_area_m2", 0.0))
                    dedup_fn(
                        labels_dir=str(jpg_dir),
                        shapefile_paths=shp_paths,
                        iou_threshold=0.1,
                        crs="EPSG:2154",
                        area_filter_enabled=min_area_m2 > 0,
                        area_filter_min_m2=min_area_m2,
                    )
                except Exception as e:
                    _print(f"WARN: shapefile dedup failed: {e}")

        _print(f"shapefiles_dir={shapefile_output_dir}")
        _print(f"detections_shp={out_shp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
