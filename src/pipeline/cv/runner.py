from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..geo_utils import extract_tif_transform_data, write_world_file as write_world_file_from_transform
from ..types import LogFn, CancelCheckFn
from .external_runner import find_external_cv_runner, run_external_cv_runner

# Re-export pour compatibilité (utilisé par preflight.py)
_find_external_cv_runner = find_external_cv_runner



def _get_model_slug(cv_config: Dict[str, Any]) -> str:
    """Retourne un slug court et sûr pour le nom du modèle (pour nommer les sous-dossiers)."""
    selected = cv_config.get("selected_model", "")
    if not selected:
        return "unknown"
    p = Path(selected)
    # Remonter au dossier modèle (weights/best.onnx -> model_name)
    model_dir = p.parent
    if model_dir.name == "weights":
        model_dir = model_dir.parent
    slug = model_dir.name or p.stem
    # Nettoyer pour un nom de dossier sûr
    slug = re.sub(r'[^\w\-.]', '_', slug)
    return slug or "model"


def _prepare_model_workdir(
    rvt_base_dir: Optional[Path],
    model_slug: str,
    log: LogFn,
) -> Path:
    """Crée le dossier raw_detections/ pour stocker les JSON/TXT d'inférence.

    Les images PNG restent dans indices/<RVT>/png/ et ne sont pas copiées.
    Retourne model_raw_dir.
    """
    model_raw_dir = (rvt_base_dir or Path(".")) / model_slug / "raw_detections"
    model_raw_dir.mkdir(parents=True, exist_ok=True)
    return model_raw_dir


def _has_cached_detection(raw_dir: Path, png_stem: str) -> bool:
    """Renvoie True si une détection (txt ou json) existe déjà pour ce PNG.

    Le runner ONNX externe comme le fallback Python écrivent
    ``{stem}.txt`` (format YOLO) et ``{stem}.json`` (payload complet) dans
    ``raw_detections/``. La présence de l'un des deux suffit à considérer
    l'image traitée — un run précédent peut n'avoir écrit que le .json si
    aucune détection n'a passé le seuil de confiance (fichier .txt vide
    possible). On considère aussi les fichiers vides (0 détection) comme
    un résultat légitime.
    """
    return (raw_dir / f"{png_stem}.txt").exists() or (raw_dir / f"{png_stem}.json").exists()


def _list_candidate_pngs(
    *,
    jpg_dir: Path,
    cv_config: Dict[str, Any],
    single_jpg: Optional[Path],
) -> list:
    """Liste les PNG que le runner va traiter, en respectant single_jpg/scan_all.

    Réplique le choix fait par :func:`_run_fallback_inference` pour que le
    short-circuit amont voit exactement le même périmètre que l'inférence.
    """
    if single_jpg is not None:
        return [single_jpg] if Path(single_jpg).exists() else []
    all_pngs = sorted(jpg_dir.glob("*.png"))
    scan_all = bool(cv_config.get("scan_all", False))
    return all_pngs if scan_all else all_pngs[:1]


def run_cv_on_folder(
    *,
    jpg_dir: Path,
    cv_config: Dict[str, Any],
    target_rvt: str,
    rvt_base_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    tif_transform_data: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    single_jpg: Optional[Path] = None,
    run_shapefile_dedup: bool = True,
    global_color_map: Optional[Dict[str, int]] = None,
    log: LogFn = lambda _: None,
    cancel_check: Optional[CancelCheckFn] = None,
) -> None:
    # ── Court-circuit si aucune classe sélectionnée ───────────────────
    _sel = (cv_config or {}).get("selected_classes")
    if isinstance(_sel, list) and len(_sel) == 0:
        log(f"Computer Vision: aucune classe sélectionnée pour ce run — inférence ignorée")
        return

    # ── Isolation par modèle ──────────────────────────────────────────
    # Chaque modèle écrit ses labels (.txt/.json), images annotées et
    # shapefiles dans un sous-dossier dédié pour éviter les collisions
    # quand plusieurs modèles ciblent le même RVT.
    model_slug = _get_model_slug(cv_config)

    # Déterminer le dossier de détections (nouvelle structure)
    if output_dir is not None:
        from ..output_paths import detection_model_dir
        effective_detection_dir = detection_model_dir(output_dir, model_slug)
        effective_detection_dir.mkdir(parents=True, exist_ok=True)
    else:
        effective_detection_dir = (rvt_base_dir or jpg_dir.parent) / model_slug
        effective_detection_dir.mkdir(parents=True, exist_ok=True)

    # Dossier raw_detections : stocke les JSON/TXT, les PNG restent dans jpg_dir (indices/)
    effective_rvt_base = effective_detection_dir.parent if output_dir is not None else (rvt_base_dir or jpg_dir.parent)
    effective_raw_dir = _prepare_model_workdir(effective_rvt_base, model_slug, log)

    # Générer le fichier classes.txt dans le dossier raw_detections du modèle
    try:
        from .class_utils import load_class_names_from_model, resolve_model_weights_path
        weights_path = resolve_model_weights_path(cv_config)
        if weights_path and weights_path.exists():
            class_names = load_class_names_from_model(weights_path)
            if class_names:
                classes_file = effective_raw_dir / "classes.txt"
                if not classes_file.exists():
                    if isinstance(class_names, dict):
                        sorted_names = [class_names[k] for k in sorted(class_names.keys())]
                        classes_file.write_text("\n".join(sorted_names), encoding="utf-8")
                    elif isinstance(class_names, (list, tuple)):
                        classes_file.write_text("\n".join(str(n) for n in class_names), encoding="utf-8")
                    log(f"Fichier classes.txt créé: {classes_file}")
    except Exception as e:
        log(f"Avertissement: impossible de créer classes.txt: {e}")

    # Log SAHI config (injectée depuis args.yaml du modèle par resolve_cv_runs)
    sahi_cfg = cv_config.get("sahi", {}) if isinstance(cv_config.get("sahi", {}), dict) else {}
    log(f"Computer Vision [{model_slug}]: SAHI slice={sahi_cfg.get('slice_height', 640)}×{sahi_cfg.get('slice_width', 640)}, overlap={sahi_cfg.get('overlap_ratio', 0.2)} (depuis args.yaml modèle)")

    # ── Short-circuit : détections déjà présentes dans raw_detections/ ───────
    # Si toutes les PNG ciblées ont déjà un .txt ou .json dans
    # ``effective_raw_dir`` et que l'utilisateur ne force pas le re-traitement,
    # on saute complètement l'inférence (externe + fallback) et on enchaîne
    # sur la génération des shapefiles. Cela évite de relancer le binaire
    # ONNX pour rien (qui ne sait pas toujours skipper lui-même) et permet
    # d'itérer rapidement sur les paramètres aval (confidence, clustering,
    # aire minimale, symbologie) sans refaire l'inférence.
    force_reprocess = bool(cv_config.get("force_reprocess", False))
    candidate_pngs = _list_candidate_pngs(
        jpg_dir=jpg_dir, cv_config=cv_config, single_jpg=single_jpg,
    )
    if not force_reprocess and candidate_pngs:
        missing = [p for p in candidate_pngs if not _has_cached_detection(effective_raw_dir, p.stem)]
        if not missing:
            log(
                f"Computer Vision [{model_slug}]: {len(candidate_pngs)} image(s) "
                f"déjà traitée(s) dans {effective_raw_dir.name}/ — inférence sautée"
            )
            if run_shapefile_dedup:
                shapefile_output_dir = effective_detection_dir / "shapefiles"
                deduplicate_cv_shapefiles_final(
                    labels_dir=effective_raw_dir,
                    png_dir=jpg_dir,
                    shp_dir=shapefile_output_dir,
                    target_rvt=target_rvt,
                    cv_config=cv_config,
                    tif_transform_data=tif_transform_data,
                    crs="EPSG:2154",
                    global_color_map=global_color_map,
                    log=log,
                )
            return
        else:
            already = len(candidate_pngs) - len(missing)
            if already > 0:
                log(
                    f"Computer Vision [{model_slug}]: {already}/{len(candidate_pngs)} "
                    f"image(s) déjà traitée(s), inférence uniquement sur {len(missing)} restante(s)"
                )

    # 1) Essayer le runner ONNX externe (compilé)
    ext = find_external_cv_runner(log=log)
    if ext is not None:
        log(f"Computer Vision: utilisation runner externe -> {ext}")
        try:
            # Le runner externe ne gère que l'inférence (pas les shapefiles).
            # La génération shapefile + post-processing global est faite côté
            # plugin Python (shapely disponible) après le retour du runner.
            run_external_cv_runner(
                ext=ext,
                jpg_dir=jpg_dir,
                target_rvt=target_rvt,
                rvt_base_dir=effective_rvt_base,
                detection_dir=effective_detection_dir,
                raw_dir=effective_raw_dir,
                cv_config=cv_config,
                single_jpg=single_jpg,
                run_shapefile_dedup=False,
                tif_transform_data=tif_transform_data,
                global_color_map=global_color_map,
                log=log,
                cancel_check=cancel_check,
            )
            # Générer les shapefiles côté plugin (avec shapely + post-processing)
            if run_shapefile_dedup:
                shapefile_output_dir = effective_detection_dir / "shapefiles"
                deduplicate_cv_shapefiles_final(
                    labels_dir=effective_raw_dir,
                    png_dir=jpg_dir,
                    shp_dir=shapefile_output_dir,
                    target_rvt=target_rvt,
                    cv_config=cv_config,
                    tif_transform_data=tif_transform_data,
                    crs="EPSG:2154",
                    global_color_map=global_color_map,
                    log=log,
                )
            return
        except Exception as e:
            # Si l'utilisateur a annulé, propager l'erreur sans fallback
            if "annul" in str(e).lower() or "cancel" in str(e).lower():
                raise
            log(f"Computer Vision: échec runner externe, fallback Python ONNX: {e}")

    # 2) Fallback : inférence ONNX en Python (onnxruntime)
    expected = (
        "data/third_party/cv_runner_onnx/windows/cv_runner_onnx.exe" if os.name == "nt" else "data/third_party/cv_runner_onnx/linux/cv_runner_onnx"
    )
    log(f"Computer Vision: runner externe absent (attendu: {expected})")
    log("Computer Vision: fallback interne ONNX -> src.pipeline.cv.computer_vision_onnx")

    enabled = bool((cv_config or {}).get("enabled", False))
    if not enabled:
        return

    _run_fallback_inference(
        jpg_dir=jpg_dir,
        raw_dir=effective_raw_dir,
        cv_config=cv_config,
        target_rvt=target_rvt,
        rvt_base_dir=effective_rvt_base,
        effective_detection_dir=effective_detection_dir,
        tif_transform_data=tif_transform_data,
        single_jpg=single_jpg,
        run_shapefile_dedup=run_shapefile_dedup,
        global_color_map=global_color_map,
        log=log,
        cancel_check=cancel_check,
    )


# ------------------------------------------------------------------ #
#  Fallback : inférence ONNX en Python (onnxruntime)                   #
# ------------------------------------------------------------------ #

def _run_fallback_inference(
    *,
    jpg_dir: Path,
    raw_dir: Optional[Path] = None,
    cv_config: Dict[str, Any],
    target_rvt: str,
    rvt_base_dir: Optional[Path] = None,
    effective_detection_dir: Optional[Path] = None,
    tif_transform_data: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    single_jpg: Optional[Path] = None,
    run_shapefile_dedup: bool = True,
    global_color_map: Optional[Dict[str, int]] = None,
    log: LogFn = lambda _: None,
    cancel_check: Optional[CancelCheckFn] = None,
) -> None:
    """Inférence ONNX image par image via computer_vision_onnx (fallback Python).

    jpg_dir  : dossier source contenant les PNG d'entrée (indices/<RVT>/png/)
    raw_dir  : dossier de sortie pour les JSON/TXT (detections/<model>/raw_detections/)
               Si None, utilise jpg_dir (comportement rétrocompat).
    """
    # raw_dir = dossier de sortie JSON/TXT ; par défaut = jpg_dir (rétrocompat)
    _raw_dir = raw_dir if raw_dir is not None else jpg_dir
    from . import computer_vision_onnx as cv_mod
    from .class_utils import (
        resolve_model_weights_path,
        load_class_names_from_model,
        load_class_colors_from_model,
    )
    from .cv_output import get_detection_output_path

    selected_model = cv_config.get("selected_model", "")
    if not selected_model:
        raise ValueError("Computer Vision activée mais aucun modèle sélectionné")

    confidence_threshold = float(cv_config.get("confidence_threshold", 0.3))
    iou_threshold = float(cv_config.get("iou_threshold", 0.5))
    generate_annotated_images = bool(cv_config.get("generate_annotated_images", False))
    generate_shapefiles = bool(cv_config.get("generate_shapefiles", False))

    sahi_config = cv_config.get("sahi", {}) if isinstance(cv_config.get("sahi", {}), dict) else {}
    slice_height = int(sahi_config.get("slice_height", 640))
    slice_width = int(sahi_config.get("slice_width", 640))
    overlap_ratio = float(sahi_config.get("overlap_ratio", 0.2))

    weights_path = resolve_model_weights_path(cv_config)
    if weights_path is None or not weights_path.exists():
        raise FileNotFoundError(f"Fichier de poids du modèle non trouvé: {weights_path}")

    class_names = load_class_names_from_model(weights_path)
    class_colors = load_class_colors_from_model(weights_path)
    log(f"Computer Vision: {len(class_names or [])} classes, couleurs={'oui' if class_colors else 'non'}")
    log(f"SAHI: slice={slice_height}×{slice_width}, overlap={overlap_ratio}")

    # Charger la session ONNX une seule fois pour toutes les images
    onnx_session = cv_mod._load_onnx_model(str(weights_path))
    log(f"Computer Vision: session ONNX chargée -> {weights_path.name}")

    # Charger les métadonnées du modèle pour afficher les paramètres de segmentation
    _model_meta = {}
    _meta_path = weights_path.with_suffix('.json')
    if _meta_path.exists():
        try:
            _model_meta = json.loads(_meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    _is_segmentation = _model_meta.get("model_type") in ("segformer", "smp") or _model_meta.get("task") in ("semantic_segmentation", "instance_segmentation")
    _use_sahi_meta = _model_meta.get("use_sahi", True)
    if _is_segmentation:
        _bg_bias = float(_model_meta.get("bg_bias", 0.0))
        _model_conf = _model_meta.get("confidence_threshold")
        _eff_conf = float(_model_conf) if _model_conf is not None else confidence_threshold
        log(f"Computer Vision: Paramètres segmentation -> confidence_threshold={_eff_conf} bg_bias={_bg_bias} use_sahi={_use_sahi_meta}")

    rvt_base = rvt_base_dir or jpg_dir.parent
    det_base = effective_detection_dir if effective_detection_dir is not None else rvt_base
    annotated_output_dir: Optional[Path] = None
    shapefile_output_dir: Optional[Path] = None

    if generate_annotated_images:
        annotated_output_dir = det_base / "annotated_images"
        annotated_output_dir.mkdir(parents=True, exist_ok=True)

    if generate_shapefiles:
        shapefile_output_dir = det_base / "shapefiles"
        shapefile_output_dir.mkdir(parents=True, exist_ok=True)

    if single_jpg is not None:
        jpg_files = [single_jpg]
        scan_all = False
    else:
        scan_all = bool(cv_config.get("scan_all", False))
        if scan_all:
            jpg_files = sorted(jpg_dir.glob("*.png"))
        else:
            jpg_files = sorted(jpg_dir.glob("*.png"))[:1]

    jpg_files = [p for p in jpg_files if p and Path(p).exists()]
    if not jpg_files:
        return

    force_reprocess = bool(cv_config.get("force_reprocess", False))
    success_count = 0
    skipped_already_processed = 0

    for jpg_file in jpg_files:
        if cancel_check and cancel_check():
            log("Computer Vision: Annulation demandée, arrêt de l'inférence...")
            raise RuntimeError("Computer Vision: Annulé par l'utilisateur")
        image_name = jpg_file.stem
        labels_txt = _raw_dir / f"{image_name}.txt"
        labels_json = _raw_dir / f"{image_name}.json"

        detection_output_path = get_detection_output_path(
            str(jpg_file),
            target_rvt,
            str(annotated_output_dir) if annotated_output_dir else None,
        )
        annotated_img = Path(detection_output_path)

        if not force_reprocess and (annotated_img.exists() or labels_txt.exists() or labels_json.exists()):
            skipped_already_processed += 1
            continue

        log(f"Inférence CV sur: {jpg_file.name} (SAHI: {slice_width}x{slice_height}, overlap={overlap_ratio})")

        ok = cv_mod.run_onnx_inference(
            image_path=str(jpg_file),
            model_path=str(weights_path),
            output_path=detection_output_path,
            confidence_threshold=confidence_threshold,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_ratio=overlap_ratio,
            generate_annotated_images=generate_annotated_images,
            annotated_output_dir=str(annotated_output_dir) if annotated_output_dir else None,
            iou_threshold=iou_threshold,
            jpg_folder_path=str(_raw_dir),
            class_names=class_names,
            class_colors=class_colors,
            onnx_session=onnx_session,
        )
        if ok:
            success_count += 1
            if generate_annotated_images and annotated_output_dir is not None:
                annotated_path = Path(detection_output_path)
                if annotated_path.exists() and tif_transform_data:
                    jpg_stem = jpg_file.stem
                    transform = tif_transform_data.get(jpg_stem)
                    if transform and len(transform) == 4:
                        pixel_width, pixel_height, x_origin, y_origin = transform
                        world_path = write_world_file_from_transform(annotated_path, pixel_width, pixel_height, x_origin, y_origin)
                        if world_path:
                            log(f"Fichier world créé: {world_path.name}")

    if scan_all and success_count == 0 and skipped_already_processed == len(jpg_files):
        return

    if run_shapefile_dedup and generate_shapefiles and shapefile_output_dir is not None:
        deduplicate_cv_shapefiles_final(
            labels_dir=_raw_dir,
            png_dir=jpg_dir,
            shp_dir=shapefile_output_dir,
            target_rvt=target_rvt,
            cv_config=cv_config,
            tif_transform_data=tif_transform_data,
            crs="EPSG:2154",
            global_color_map=global_color_map,
            log=log,
        )


# ------------------------------------------------------------------ #
#  Génération shapefiles + déduplication                               #
# ------------------------------------------------------------------ #

def deduplicate_cv_shapefiles_final(
    *,
    labels_dir: Path,
    png_dir: Optional[Path] = None,
    shp_dir: Path,
    target_rvt: str,
    cv_config: Optional[Dict[str, Any]] = None,
    tif_transform_data: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    global_color_map: Optional[Dict[str, int]] = None,
    temp_dir: Optional[Path] = None,
    crs: str = "EPSG:2154",
    log: LogFn = lambda _: None,
) -> None:
    from .conversion_shp import create_shapefile_from_detections, _filter_gpkg_by_min_area
    log("Computer Vision: conversion GeoPackage -> src.pipeline.cv.conversion_shp")

    class_names = None
    class_colors = None
    model_task = None
    clustering_configs = None
    postprocess_config = None
    try:
        from .class_utils import resolve_model_weights_path, load_class_names_from_model, load_class_colors_from_model
        from .model_config import load_clustering_config_from_model, load_postprocess_config_from_model
        if isinstance(cv_config, dict):
            weights_path = resolve_model_weights_path(cv_config)
            if weights_path and weights_path.exists():
                class_names = load_class_names_from_model(weights_path)
                class_colors = load_class_colors_from_model(weights_path)
                # Lire le type de tâche depuis les métadonnées du modèle
                meta_path = weights_path.with_suffix('.json')
                if meta_path.exists():
                    try:
                        import json as _json
                        _meta = _json.loads(meta_path.read_text(encoding='utf-8'))
                        model_task = _meta.get('task')
                        log(f"Computer Vision: tâche du modèle = {model_task}")
                    except Exception:
                        pass
                # Charger la configuration de clustering
                clustering_configs = load_clustering_config_from_model(weights_path)
                if clustering_configs:
                    log(f"Computer Vision: {len(clustering_configs)} config(s) de clustering chargée(s)")
                # Charger la configuration de post-traitement géométrique (merge/overlap)
                postprocess_config = load_postprocess_config_from_model(weights_path)
    except Exception as e:
        log(f"Computer Vision: impossible de récupérer les noms de classes depuis le modèle: {e}")

    # Filtrer les configs de clustering selon selected_classes :
    # une config n'est activée que si son output_class_name est sélectionné
    _raw_classes = (cv_config or {}).get("selected_classes")
    selected_classes = _raw_classes if isinstance(_raw_classes, list) else None
    if clustering_configs and selected_classes is not None:
        clustering_configs = [
            cc for cc in clustering_configs
            if str(cc.get("output_class_name") or "").strip() in selected_classes
        ]
        if not clustering_configs:
            log("Computer Vision: clustering désactivé (output_class_name non sélectionné)")
        else:
            log(f"Computer Vision: {len(clustering_configs)} config(s) de clustering actives après filtrage")

    shp_dir.mkdir(parents=True, exist_ok=True)

    # Générer les shapefiles par classe (le post-processing global
    # — fusion des polygones adjacents + suppression des superpositions —
    # est intégré directement dans create_shapefile_from_detections)
    out_shp = shp_dir / f"detections_{target_rvt}.gpkg"
    try:
        create_shapefile_from_detections(
            labels_dir=str(labels_dir),
            png_dir=str(png_dir) if png_dir is not None else None,
            output_shapefile=str(out_shp),
            tif_transform_data=tif_transform_data,
            crs=str(crs),
            temp_dir=str(temp_dir) if temp_dir is not None else None,
            class_names=class_names,
            selected_classes=selected_classes,
            class_colors=class_colors,
            global_color_map=global_color_map if global_color_map else None,
            model_task=model_task,
            clustering_configs=clustering_configs,
            postprocess_config=postprocess_config,
            min_confidence=float((cv_config or {}).get("confidence_threshold", 0.0) or 0.0),
        )
        qgs_root = shp_dir.parent if shp_dir.name.lower() in {"shapefiles", "shp"} else shp_dir
        qgs_path = qgs_root / "detections_validation.qgs"
        if qgs_path.exists():
            log(f"Computer Vision: projet QGIS généré -> {qgs_path}")
    except Exception as e:
        log(f"Computer Vision: génération shapefile/projet QGIS ignorée (erreur): {e}")

    # Filtrage par aire minimale (optionnel)
    min_area_m2 = float((cv_config or {}).get("min_area_m2", 0.0))
    if min_area_m2 > 0:
        gpkg_paths = [str(p) for p in shp_dir.glob("*.gpkg")]
        if gpkg_paths:
            try:
                _filter_gpkg_by_min_area(
                    gpkg_paths=gpkg_paths,
                    min_area_m2=min_area_m2,
                    crs=str(crs),
                )
            except Exception as e:
                log(f"Computer Vision: filtrage par aire ignoré (erreur): {e}")
