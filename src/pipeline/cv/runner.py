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
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..cancellation import DetectionImpossible, PipelineCancelled
from ..types import LogFn, CancelCheckFn
from .external_runner import (
    ImageProgressFn,
    TileProgressFn,
    find_external_cv_runner,
    run_external_cv_runner,
)
from .runner_cache import (
    aligner_cache_detections,
    cle_cache_detections,
    get_model_slug,
    has_cached_detection,
    list_candidate_pngs,
    prepare_model_workdir,
    purge_stale_cached_detections,
    version_moteur,
)
from .runner_inference import run_fallback_inference
from .runner_shapefiles import deduplicate_cv_shapefiles_final

# Re-export pour compatibilité (utilisé par preflight.py)
_find_external_cv_runner = find_external_cv_runner

MOTIF_MOTEUR_ABSENT = "moteur de détection absent de l'extension (cv_runner_onnx) — réinstallez l'extension"


def _onnxruntime_disponible() -> bool:
    """Le repli dans le processus de QGIS n'existe que si onnxruntime y est
    installé — ni QGIS 3.44 (OSGeo4W) ni QGIS 4.0 ne l'ont (vérifié 2026-10-15)."""
    import importlib.util
    return importlib.util.find_spec("onnxruntime") is not None


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


#: Pas de pixel (m) vraisemblable pour un MNT ou un indice ; hors de là, la fenêtre n'est pas mise à l'échelle.
GSD_PLAUSIBLE = (0.05, 50.0)


def _gsd_des_images(tif_transform_data, jpg_dir: Optional[Path]) -> Optional[float]:
    """Résolution (m/px) des images inférées : celle des TIF dont elles sont tirées (``tif_transform_data``, pixel
    pour pixel), sinon le fichier world du premier PNG. ``None`` si inconnue."""
    for t in (tif_transform_data or {}).values():
        try:
            if not (t and t[0]):
                continue
            pw, ph, x0, y0 = (float(v) for v in t[:4])
        except (TypeError, ValueError, IndexError):
            continue
        # Sans géotransformation, GDAL rend un pixel de 1 à l'origine (0, 0), axe vertical montant : pas une
        # résolution (audit 2026-10-31). Un pas hors de [5 cm ; 50 m] n'est pas métrique (degrés…).
        if ph > 0 or (x0, y0) == (0.0, 0.0) or not GSD_PLAUSIBLE[0] <= abs(pw) <= GSD_PLAUSIBLE[1]:
            return None
        return round(abs(pw), 6)
    try:
        for pgw in sorted(Path(jpg_dir).glob("*.pgw")):
            return round(abs(float(pgw.read_text(encoding="utf-8").split()[0])), 6)
    except (OSError, ValueError, IndexError, TypeError):
        pass
    return None


def remplissage_des_fenetres(pngs, sahi, gsd_m: Optional[float]) -> Optional[Dict[str, Any]]:
    """Images de ``pngs`` qui occupent moins de la moitié d'une fenêtre d'analyse (``sahi`` mis à l'échelle) :
    ``{n, total, part_moyenne (des seules images concernées), fenetre_px, fenetre_m}``, ``None`` s'il n'y en a pas.
    Déterministe (tailles lues dans l'en-tête des PNG), calculé une fois par run et par modèle : la cause de
    l'avertissement VISIBLE (``cv_post_service.avertissements_remplissage``) ; le détail par image reste au journal
    technique (``computer_vision_onnx.message_fenetre_peu_remplie``)."""
    from PIL import Image

    from .sahi_lite import part_de_fenetre

    sw, sh = int((sahi or {}).get("slice_width") or 0), int((sahi or {}).get("slice_height") or 0)
    if sw <= 0 or sh <= 0 or not pngs:
        return None
    parts = []
    for p in pngs:
        try:
            with Image.open(p) as im:
                parts.append(part_de_fenetre(im.size[0], im.size[1], sw, sh))
        except OSError:
            continue
    peu = [x for x in parts if x < 0.5]
    if not peu:
        return None
    return {"n": len(peu), "total": len(parts), "part_moyenne": sum(peu) / len(peu), "fenetre_px": sw,
            "fenetre_m": round(sw * gsd_m, 3) if gsd_m else None}


def _sahi_a_la_resolution(cv_config, tif_transform_data, jpg_dir, log):
    """``cv_config["sahi"]`` remis en mètres de terrain (``model_config.sahi_a_l_echelle``) et journalisé."""
    try:
        from .model_config import resolve_model_weights_path, sahi_a_l_echelle
        try:   # côté plugin seulement (le binaire n'appelle pas run_cv_on_folder) : source unique du GSD
            from ...app.services.indices_metres import gsd_du_modele as gsd_entrainement
        except ImportError:  # pragma: no cover - tests (src/ sur le chemin)
            from app.services.indices_metres import gsd_du_modele as gsd_entrainement
        sahi = cv_config.get("sahi")
        wp = resolve_model_weights_path(cv_config)
        if not isinstance(sahi, dict) or wp is None:
            return cv_config
        gsd_m, gsd_in = gsd_entrainement(wp), _gsd_des_images(tif_transform_data, jpg_dir)
        if gsd_m and gsd_in is None:
            log("⚠ Computer Vision : résolution des images inconnue ou invraisemblable (raster sans "
                "géoréférencement, ou en degrés) — fenêtre de détection laissée à la taille d'entraînement.")
        echelle = sahi_a_l_echelle(sahi, gsd_m, gsd_in)
        if echelle.get("borne"):
            # Fenêtre bornée (facteur 8 ou plancher de 32 px) : elle ne couvre PAS le terrain d'entraînement.
            ecart = max(gsd_in / gsd_m, gsd_m / gsd_in)
            log(f"⚠ Computer Vision : images à {gsd_in:g} m/px, modèle appris à {gsd_m:g} m/px → fenêtre de "
                f"détection bornée à {echelle['slice_width']} px ({echelle['slice_width'] * gsd_in:g} m de terrain "
                f"au lieu de {echelle['slice_modele_px'] * gsd_m:g} m) au lieu de {sahi.get('slice_width')} px : "
                "l'échelle d'entraînement n'est pas respectée"
                + (" — écart de plus de 8 fois : vérifiez la résolution et le système de coordonnées des images."
                   if ecart > 8 else "."))
        elif echelle != sahi:
            log(f"Computer Vision : images à {gsd_in:g} m/px, modèle appris à {gsd_m:g} m/px → fenêtre de "
                f"{echelle['slice_width']} px ({echelle['slice_width'] * gsd_in:g} m de terrain, comme à "
                f"l'entraînement) au lieu de {sahi.get('slice_width')} px, remise à la taille du modèle.")
        return {**cv_config, "sahi": echelle}
    except Exception as e:  # noqa: BLE001 — jamais bloquer l'inférence sur la mise à l'échelle
        log(f"Avertissement : mise à l'échelle de la fenêtre de détection ignorée ({e})")
        return cv_config


def run_cv_on_folder(
    *,
    jpg_dir: Path,
    cv_config: Dict[str, Any],
    target_rvt: str,
    rvt_base_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    tif_transform_data: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    valid_region_bounds: Optional[List[Tuple[float, float, float, float]]] = None,
    cell_bounds_by_stem: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    single_jpg: Optional[Path] = None,
    run_shapefile_dedup: bool = True,
    global_color_map: Optional[Dict[str, int]] = None,
    log: LogFn = lambda _: None,
    cancel_check: Optional[CancelCheckFn] = None,
    image_progress: Optional[ImageProgressFn] = None,
    tile_progress: Optional[TileProgressFn] = None,
    stats: Optional[Dict[str, Any]] = None,
    on_appareil: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Optional[int]:
    # ``on_appareil`` : décision du binaire sur l'appareil de détection (carte
    # graphique retenue / écartée / repli), relayée au journal dès qu'elle tombe.
    # ``stats`` : mesure du run déposée par le runner externe (images réellement
    # inférées, secondes) — cf. run_external_cv_runner ; le repli in-process ne
    # la remplit pas (il ne remonte ni tuiles ni total).
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
    # Fenêtre en mètres (2026-10-31) : remise à la résolution des images AVANT la clé du cache (qui la porte).
    cv_config = _sahi_a_la_resolution(cv_config, tif_transform_data, jpg_dir, log)

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

    # ── Clé du cache des détections (lot 0b, 2026-10) ─────────────────
    # raw_detections/ est indexé par nom d'image : la clé dit avec quoi il a
    # été calculé (fichiers du modèle, SAHI résolu, seuils, moteur). Clé
    # différente — ou absente : dossier d'une version précédente — → purge,
    # donc ré-inférence (le binaire saute toute image au cache).
    ext = find_external_cv_runner(log=log)
    if ext is None and not _onnxruntime_disponible():
        # Avant toute purge : un moteur manquant ne doit pas vider le cache.
        raise DetectionImpossible(MOTIF_MOTEUR_ABSENT)
    from .model_config import resolve_model_weights_path as _poids_du_modele
    _purgees = aligner_cache_detections(
        effective_raw_dir,
        cle_cache_detections(cv_config, _poids_du_modele(cv_config), version_moteur(ext)),
    )
    if _purgees:
        log(
            f"Computer Vision [{model_slug}]: {_purgees} image(s) en cache calculée(s) avec un "
            "autre modèle, d'autres seuils ou un autre moteur — purgée(s), ré-inférence"
        )

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
    if stats is not None:   # fenêtres surtout vides : remontées à l'écran par l'appelant, une fois par modèle
        try:
            stats["remplissage"] = remplissage_des_fenetres(
                candidate_pngs, cv_config.get("sahi"), _gsd_des_images(tif_transform_data, jpg_dir))
        except Exception as e:  # noqa: BLE001 — un avertissement ne bloque jamais la détection
            log(f"Avertissement : part des fenêtres remplies non calculée ({e})")
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
                    cell_bounds_by_stem=cell_bounds_by_stem,
                    crs=(cv_config or {}).get("crs") or "EPSG:2154",  # CRS du run (cf. existing_rvt)
                    global_color_map=global_color_map,
                    log=log,
                    cancel_check=cancel_check,
                )
            if stats is not None:
                stats["moteur"] = "cache"
            return
        else:
            already = len(candidate_pngs) - len(missing)
            if already > 0:
                log(
                    f"Computer Vision [{model_slug}]: {already}/{len(candidate_pngs)} "
                    f"image(s) déjà traitée(s), inférence uniquement sur {len(missing)} restante(s)"
                )

    # 1) Essayer le runner ONNX externe (compilé), trouvé avant la clé du cache
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
                stats=stats,
                on_appareil=on_appareil,
            )
        except PipelineCancelled:
            # Annulation utilisateur : propager sans tenter le fallback.
            raise
        except Exception as e:
            if not _onnxruntime_disponible():
                raise DetectionImpossible(f"moteur de détection en échec ({e})") from e
            log(f"Computer Vision: échec runner externe, fallback Python ONNX: {e}")
        else:
            # Le moteur a tourné : un échec de l'écriture qui suit (GeoPackage
            # verrouillé…) remonte tel quel, ce n'est ni le moteur ni un repli.
            if stats is not None:
                stats["moteur"] = "binaire"
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
                    cell_bounds_by_stem=cell_bounds_by_stem,
                    crs=(cv_config or {}).get("crs") or "EPSG:2154",  # CRS du run (cf. existing_rvt)
                    global_color_map=global_color_map,
                    log=log,
                    cancel_check=cancel_check,
                )
            return total_detections

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
    if stats is not None:
        stats["moteur"] = "python"
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
        cell_bounds_by_stem=cell_bounds_by_stem,
        single_jpg=single_jpg,
        run_shapefile_dedup=run_shapefile_dedup,
        global_color_map=global_color_map,
        log=log,
        cancel_check=cancel_check,
        image_progress=image_progress,
    )
