"""Génération des shapefiles à partir des détections brutes.

Encapsule l'étape « après inférence » qui transforme les ``.txt``/``.json``
écrits dans ``raw_detections/`` en GeoPackage géoréférencé. Délègue
l'essentiel à :mod:`conversion_shp` mais centralise la résolution des
métadonnées du modèle (classes, couleurs, clustering, post-traitement)
qui pilotent cette conversion.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..cancellation import PipelineCancelled, check_cancelled
from ..types import CancelCheckFn, LogFn


def deduplicate_cv_shapefiles_final(
    *,
    labels_dir: Path,
    png_dir: Optional[Path] = None,
    shp_dir: Path,
    target_rvt: str,
    output_dir: Optional[Path] = None,
    cv_config: Optional[Dict[str, Any]] = None,
    tif_transform_data: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    global_color_map: Optional[Dict[str, int]] = None,
    temp_dir: Optional[Path] = None,
    crs: str = "EPSG:2154",
    log: LogFn = lambda _: None,
    cancel_check: Optional[CancelCheckFn] = None,
) -> None:
    from .conversion_shp import create_shapefile_from_detections, _filter_gpkg_by_min_area
    check_cancelled(cancel_check)
    log("Computer Vision: conversion GeoPackage -> src.pipeline.cv.conversion_shp")

    class_names = None
    class_colors = None
    model_task = None
    clustering_configs = None
    postprocess_config = None
    try:
        from .model_config import resolve_model_weights_path
        from .model_profile import ModelProfile
        if isinstance(cv_config, dict):
            weights_path = resolve_model_weights_path(cv_config)
            if weights_path and weights_path.exists():
                profile = ModelProfile.load(weights_path)
                class_names = list(profile.class_names) if profile.class_names else None
                class_colors = list(profile.class_colors) if profile.class_colors else None
                model_task = profile.task
                if model_task is not None:
                    log(f"Computer Vision: tâche du modèle = {model_task}")
                if profile.clustering:
                    clustering_configs = [rule.to_dict() for rule in profile.clustering]
                    log(f"Computer Vision: {len(clustering_configs)} config(s) de clustering chargée(s)")
                postprocess_config = profile.postprocess.to_dict()
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

    # shp_dir n'est PLUS créé d'office : la sortie est routée par entité vers
    # detections/<entity_slug>/ (créés à l'écriture). shp_dir ne sert que de
    # repli legacy, créé paresseusement par conversion_shp s'il est réellement utilisé.

    # ── Routage entité-centré ──────────────────────────────────────────
    # Construit la table classe → GeoPackage d'entité à partir du découpage
    # ``entities`` du run (posé par l'orchestrateur). Chaque classe est écrite
    # dans detections/<entity_slug>/<entity_slug>.gpkg (vocabulaire utilisateur).
    # Repli (run sans 'entities', ex. config ancienne / mono-modèle) : un
    # GeoPackage modèle-centré sous detections/<model_slug>/, collecté par finalize.
    class_targets = None
    out_shp = shp_dir / f"detections_{target_rvt}.gpkg"
    entities = (cv_config or {}).get("entities") if isinstance(cv_config, dict) else None
    if output_dir is not None and entities:
        from ..output_paths import build_entity_class_targets
        # Routage classe → [(gpkg d'entité, nom_de_couche)] (helper pur testé) :
        # gère les classes partagées et la DUPLICATION renommée (ex. source
        # 'cratere_obus' → couche 'cratere_obus' dans « Trous d'obus » ET couche
        # renommée dans la dérivée « Zones d'extraction »).
        class_targets = build_entity_class_targets(output_dir, entities)
        _n_gpkg = len({gp for lst in class_targets.values() for gp, _ in lst})
        log(f"Computer Vision: routage par entité actif ({_n_gpkg} GeoPackage(s) d'entité)")
    elif output_dir is not None:
        from .runner_cache import get_model_slug
        from ..output_paths import detection_entity_dir
        slug = get_model_slug(cv_config or {})
        out_shp = detection_entity_dir(output_dir, slug) / f"{slug}.gpkg"
        out_shp.parent.mkdir(parents=True, exist_ok=True)
        log(f"Computer Vision: run sans 'entities' — repli modèle-centré detections/{slug}/")

    # Générer les shapefiles par classe (le post-processing global
    # — fusion des polygones adjacents + suppression des superpositions —
    # est intégré directement dans create_shapefile_from_detections)
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
            class_targets=class_targets,
            cancel_check=cancel_check,
        )
        qgs_root = shp_dir.parent if shp_dir.name.lower() in {"shapefiles", "shp"} else shp_dir
        qgs_path = qgs_root / "detections_validation.qgs"
        if qgs_path.exists():
            log(f"Computer Vision: projet QGIS généré -> {qgs_path}")
    except PipelineCancelled:
        raise
    except Exception as e:
        log(f"Computer Vision: génération shapefile/projet QGIS ignorée (erreur): {e}")

    # Filtrage par aire minimale (optionnel) — cible les GeoPackage réellement
    # écrits : ceux des entités si routage actif, sinon le GeoPackage de repli.
    min_area_m2 = float((cv_config or {}).get("min_area_m2", 0.0))
    if min_area_m2 > 0:
        if class_targets:
            _all_gpkgs = {gp for lst in class_targets.values() for gp, _ in lst}
            gpkg_paths = sorted(gp for gp in _all_gpkgs if Path(gp).exists())
        elif Path(out_shp).exists():
            gpkg_paths = [str(out_shp)]
        else:
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
