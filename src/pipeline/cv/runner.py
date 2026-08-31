"""Orchestration de l'inférence CV pour un dossier d'images.

Ce module est la porte d'entrée du pipeline CV. Pour chaque image PNG
candidate, il choisit le runner (externe compilé ou fallback Python),
gère le court-circuit cache, puis enchaîne sur la génération des
shapefiles.

Modules délégués :

- :mod:`runner_cache` — résolution des chemins, slug modèle, vérification
  des fichiers déjà produits.
- :mod:`runner_inference` — fallback Python ONNX (utilisé quand
  ``cv_runner_onnx`` n'est pas disponible).
- :mod:`runner_shapefiles` — production des GeoPackage à partir des
  ``.txt``/``.json`` de ``raw_detections/``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..cancellation import PipelineCancelled
from ..types import LogFn, CancelCheckFn
from .external_runner import (
    ImageProgressFn,
    TileProgressFn,
    find_external_cv_runner,
    run_external_cv_runner,
)
from .runner_cache import (
    get_model_slug,
    has_cached_detection,
    list_candidate_pngs,
    prepare_model_workdir,
    purge_stale_cached_detections,
)
from .runner_inference import run_fallback_inference
from .runner_shapefiles import deduplicate_cv_shapefiles_final

# Re-export pour compatibilité (utilisé par preflight.py)
_find_external_cv_runner = find_external_cv_runner


def _absolutize_models_dir(cv_config: Dict[str, Any]) -> Dict[str, Any]:
    """Résout ``cv_config['models_dir']`` en chemin absolu (racine du plugin).

    Le runner externe (subprocess) comme le fallback Python résolvent
    ``models_dir`` relativement à leur **CWD** — qui n'est PAS la racine du
    plugin (sous QGIS 4, le CWD est le dossier d'installation de QGIS). Un
    ``models_dir`` relatif (« data/models » par défaut) pointe alors à côté
    et le ``.onnx`` du modèle n'est pas trouvé. On le résout en absolu ici,
    une seule fois, pour tous les consommateurs en aval. Retourne une copie ;
    no-op si ``models_dir`` est absent ou déjà absolu.
    """
    if not isinstance(cv_config, dict):
        return cv_config
    raw = str(cv_config.get("models_dir") or "").strip()
    if not raw or Path(raw).is_absolute():
        return cv_config
    plugin_root = Path(__file__).resolve().parents[3]
    return {**cv_config, "models_dir": str((plugin_root / raw).resolve())}


def run_cv_on_folder(
    *,
    jpg_dir: Path,
    cv_config: Dict[str, Any],
    target_rvt: str,
    rvt_base_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    tif_transform_data: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    valid_region_bounds: Optional[List[Tuple[float, float, float, float]]] = None,
    single_jpg: Optional[Path] = None,
    run_shapefile_dedup: bool = True,
    global_color_map: Optional[Dict[str, int]] = None,
    log: LogFn = lambda _: None,
    cancel_check: Optional[CancelCheckFn] = None,
    image_progress: Optional[ImageProgressFn] = None,
    tile_progress: Optional[TileProgressFn] = None,
) -> Optional[int]:
    # Retour : total_detections du run (résumé du runner externe), None si
    # inconnu (fallback in-process, court-circuit, ancien binaire).
    # ``models_dir`` → absolu : indispensable pour que le runner externe
    # (subprocess) ET le fallback trouvent le .onnx quel que soit le CWD.
    cv_config = _absolutize_models_dir(cv_config)

    # Filet SAHI : ``resolve_cv_runs`` (cv_post_service / finalize) tourne avec
    # le ``models_dir`` RELATIF de la config ; sous QGIS (CWD ≠ racine plugin)
    # il échoue à résoudre le dossier modèle et omet ``cv_config["sahi"]`` → le
    # runner externe retombe alors sur le défaut 640 (mauvaise échelle), alors
    # que le fallback in-process lit ``profile.sahi`` (correct). ``models_dir``
    # étant désormais absolu, on relit le vrai SAHI de l'args.yaml du modèle
    # pour aligner binaire et fallback.
    try:
        from .model_config import resolve_sahi_config
        _sahi = resolve_sahi_config(cv_config)
        if _sahi is not None:
            cv_config = {**cv_config, "sahi": _sahi}
    except Exception as _e:  # ne jamais bloquer l'inférence sur ce filet
        log(f"Avertissement: résolution SAHI ignorée ({_e})")

    # ── Court-circuit si aucune classe sélectionnée ───────────────────
    _sel = (cv_config or {}).get("selected_classes")
    if isinstance(_sel, list) and len(_sel) == 0:
        log("Computer Vision: aucune classe sélectionnée pour ce run — inférence ignorée")
        return

    # ── Isolation par modèle ──────────────────────────────────────────
    # Chaque modèle écrit ses labels (.txt/.json), images annotées et
    # shapefiles dans un sous-dossier dédié pour éviter les collisions
    # quand plusieurs modèles ciblent le même RVT.
    model_slug = get_model_slug(cv_config)

    # Dossier TECHNIQUE du modèle (échafaudage non-livrable : raw_detections/ +
    # annotated_images/) sous detections/_technique/<model_slug>/. Les détections
    # vectorielles, elles, sont routées par ENTITÉ vers detections/<entity_slug>/
    # (cf. runner_shapefiles), pour coller au vocabulaire utilisateur.
    if output_dir is not None:
        from ..output_paths import detection_technique_dir, detection_technique_raw_dir
        effective_detection_dir = detection_technique_dir(output_dir, model_slug)
        effective_detection_dir.mkdir(parents=True, exist_ok=True)
        effective_raw_dir = detection_technique_raw_dir(output_dir, model_slug)
        effective_raw_dir.mkdir(parents=True, exist_ok=True)
    else:
        effective_detection_dir = (rvt_base_dir or jpg_dir.parent) / model_slug
        effective_detection_dir.mkdir(parents=True, exist_ok=True)
        effective_raw_dir = prepare_model_workdir(effective_detection_dir.parent, model_slug, log)

    # Base RVT (pour external_runner / fallback) — parent du dossier technique.
    effective_rvt_base = effective_detection_dir.parent if output_dir is not None else (rvt_base_dir or jpg_dir.parent)

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

    # Trace post-hoc des paramètres effectifs CV (resolution post
    # ModelProfile pour confidence_threshold).
    from ..types import format_params_line
    from .model_config import DEFAULT_CONFIDENCE
    _runtime_conf = float(cv_config.get("confidence_threshold", DEFAULT_CONFIDENCE) or DEFAULT_CONFIDENCE)
    _effective_conf = _runtime_conf
    _conf_source = "runtime"
    try:
        from .model_profile import ModelProfile
        _wp = (cv_config.get("weights_path") or cv_config.get("selected_model") or "")
        if _wp:
            _profile = ModelProfile.load(Path(_wp).parent if Path(_wp).suffix else Path(_wp))
            _effective_conf = _profile.effective_confidence_threshold(_runtime_conf)
            if abs(_effective_conf - _runtime_conf) > 1e-9:
                _conf_source = "metadata"
    except Exception:
        # ModelProfile.load peut échouer (fichiers manquants) — on
        # garde la valeur runtime sans bloquer l'inférence.
        pass

    _selected_classes = cv_config.get("selected_classes")
    if _selected_classes is None:
        _classes_str = "all"
    elif isinstance(_selected_classes, list):
        _classes_str = ",".join(_selected_classes) if _selected_classes else "(empty)"
    else:
        _classes_str = str(_selected_classes)

    log(format_params_line(f"CV/{model_slug}", {
        "model": model_slug,
        "target_rvt": target_rvt,
        "confidence": _effective_conf,
        "confidence_source": _conf_source,
        "iou_threshold": float(cv_config.get("iou_threshold", 0.5) or 0.5),
        "sahi_slice": f"{int(sahi_cfg.get('slice_height', 640))}x{int(sahi_cfg.get('slice_width', 640))}",
        "sahi_overlap": float(sahi_cfg.get("overlap_ratio", 0.2) or 0.2),
        "classes": _classes_str,
        "min_area_m2": float(cv_config.get("min_area_m2", 0) or 0),
    }))

    # ── Short-circuit : détections déjà présentes dans raw_detections/ ───────
    # Si toutes les PNG ciblées ont déjà un .txt ou .json dans
    # ``effective_raw_dir`` et que l'utilisateur ne force pas le re-traitement,
    # on saute complètement l'inférence (externe + fallback) et on enchaîne
    # sur la génération des shapefiles. Cela évite de relancer le binaire
    # ONNX pour rien (qui ne sait pas toujours skipper lui-même) et permet
    # d'itérer rapidement sur les paramètres aval (confidence, clustering,
    # aire minimale, symbologie) sans refaire l'inférence.
    force_reprocess = bool(cv_config.get("force_reprocess", False))
    candidate_pngs = list_candidate_pngs(
        jpg_dir=jpg_dir, cv_config=cv_config, single_jpg=single_jpg,
    )
    # Un cache plus ancien que son PNG a été calculé sur une autre géométrie
    # (ex. bascule rogné → à marge) : purgé pour forcer la ré-inférence — le
    # binaire externe saute par simple existence des fichiers, sans les dates.
    stale = purge_stale_cached_detections(effective_raw_dir, candidate_pngs)
    if stale:
        log(
            f"Computer Vision [{model_slug}]: {stale} cache(s) de détection "
            "plus ancien(s) que leur PNG — purgé(s), ré-inférence"
        )
    if not force_reprocess and candidate_pngs:
        missing = [p for p in candidate_pngs if not has_cached_detection(effective_raw_dir, p.stem)]
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
                    output_dir=output_dir,
                    cv_config=cv_config,
                    tif_transform_data=tif_transform_data,
                    valid_region_bounds=valid_region_bounds,
                    crs="EPSG:2154",
                    global_color_map=global_color_map,
                    log=log,
                    cancel_check=cancel_check,
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
            total_detections = run_external_cv_runner(
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
                image_progress=image_progress,
                tile_progress=tile_progress,
            )
            # Générer les shapefiles côté plugin (avec shapely + post-processing)
            if run_shapefile_dedup:
                shapefile_output_dir = effective_detection_dir / "shapefiles"
                deduplicate_cv_shapefiles_final(
                    labels_dir=effective_raw_dir,
                    png_dir=jpg_dir,
                    shp_dir=shapefile_output_dir,
                    target_rvt=target_rvt,
                    output_dir=output_dir,
                    cv_config=cv_config,
                    tif_transform_data=tif_transform_data,
                    valid_region_bounds=valid_region_bounds,
                    crs="EPSG:2154",
                    global_color_map=global_color_map,
                    log=log,
                    cancel_check=cancel_check,
                )
            return total_detections
        except PipelineCancelled:
            # Annulation utilisateur : propager sans tenter le fallback.
            raise
        except Exception as e:
            log(f"Computer Vision: échec runner externe, fallback Python ONNX: {e}")

    # 2) Fallback : inférence ONNX en Python (onnxruntime)
    expected = (
        "data/third_party/cv_runner_onnx/windows/cv_runner_onnx.exe" if os.name == "nt" else "data/third_party/cv_runner_onnx/linux/cv_runner_onnx"
    )
    log(f"Computer Vision: runner externe absent (attendu: {expected})")
    log("Computer Vision: fallback interne ONNX -> src.pipeline.cv.computer_vision_onnx")

    enabled = bool((cv_config or {}).get("enabled", False))
    if not enabled:
        return None

    # ponytail: le fallback in-process ne remonte ni tuiles ni total — chemin
    # rare (binaire absent), à câbler si l'usage sans binaire se généralise.
    run_fallback_inference(
        jpg_dir=jpg_dir,
        raw_dir=effective_raw_dir,
        cv_config=cv_config,
        target_rvt=target_rvt,
        rvt_base_dir=effective_rvt_base,
        output_dir=output_dir,
        effective_detection_dir=effective_detection_dir,
        tif_transform_data=tif_transform_data,
        valid_region_bounds=valid_region_bounds,
        single_jpg=single_jpg,
        run_shapefile_dedup=run_shapefile_dedup,
        global_color_map=global_color_map,
        log=log,
        cancel_check=cancel_check,
        image_progress=image_progress,
    )
