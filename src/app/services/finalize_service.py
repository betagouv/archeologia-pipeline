from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..progress_reporter import ProgressReporter
    from ..structured_logger import StructuredLogger

LogFn = Callable[[str], None]


def build_entity_grouping(
    runs: Optional[List[Dict[str, Any]]],
) -> "tuple[Dict[str, str], set]":
    """Depuis les runs, renvoie ``(entity_labels: slug→libellé, derived_slugs)``.

    ``derived_slugs`` = slugs des entités **dérivées** (zone + constituants) : seules
    elles forment un groupe de couches — dans le ``.qgs`` (``ui/qgs_writer``) **et** au
    chargement live (``layer_loader.load_result_layers``). Helper partagé par les deux
    chemins pour garantir le **même** regroupement.
    """
    entity_labels: Dict[str, str] = {}
    derived_slugs: set = set()
    for r in runs or []:
        if not isinstance(r, dict):
            continue
        for ent in (r.get("entities") or []):
            if not isinstance(ent, dict):
                continue
            slug = str(ent.get("slug") or "").strip()
            if not slug:
                continue
            entity_labels[slug] = str(ent.get("label") or slug)
            if ent.get("is_derived"):
                derived_slugs.add(slug)
    return entity_labels, derived_slugs


def build_min_confidence_by_slug(
    cv_runs: Optional[List[Dict[str, Any]]],
    *,
    model_slug_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
) -> Dict[str, float]:
    """Mappe chaque slug de couche → seuil de confiance du run qui la produit.

    La symbologie catégorisée du ``.qgs`` consolidé doit utiliser, par couche,
    le **même** ``min_confidence`` que celui ayant servi à binner le champ
    ``conf_bin`` à la conversion (``cv_config["confidence_threshold"]`` du run,
    cf. ``runner_shapefiles.deduplicate_cv_shapefiles_final``). Sinon les
    libellés de catégories (ex. ``[0.2:0.4[``) ne matchent pas les ``conf_bin``
    réels (ex. ``[0.3:0.4[``) et la tranche basse devient invisible.

    Les seuils étant **par entité** (réglages avancés), un unique seuil global
    ne convient pas : on indexe par slug d'entité (route entité-centrée
    ``detections/<slug>/<slug>.gpkg``), avec repli sur le slug de modèle pour
    les runs legacy sans ``entities``.

    Si un slug est alimenté par plusieurs runs, on conserve le **minimum** :
    la 1ʳᵉ tranche de légende couvre alors toutes les ``conf_bin`` présentes.

    ``model_slug_fn`` : injecté pour les tests ; par défaut, import différé de
    ``pipeline.cv.runner_cache.get_model_slug`` (évite de tirer QGIS/shapely
    sur le chemin entité, pur).
    """
    result: Dict[str, float] = {}

    def _put(slug: str, conf: float) -> None:
        slug = (slug or "").strip()
        if not slug:
            return
        prev = result.get(slug)
        result[slug] = conf if prev is None else min(prev, conf)

    for run in cv_runs or []:
        if not isinstance(run, dict):
            continue
        conf = float(run.get("confidence_threshold", 0.0) or 0.0)
        entities = run.get("entities") or []
        if entities:
            for ent in entities:
                if isinstance(ent, dict):
                    _put(str(ent.get("slug") or ""), conf)
        else:
            fn = model_slug_fn
            if fn is None:
                from ...pipeline.cv.runner_cache import get_model_slug as fn  # import différé
            _put(fn(run), conf)

    return result


def _collect_vrt_paths_and_build(idx_dir: Path, det_dir: Path, log: LogFn) -> List[str]:
    """Parcourt indices/ pour créer les index.vrt et retourne les chemins VRT."""
    try:
        from ...pipeline.ign.products.results import build_vrt_index
    except ImportError:
        from pipeline.ign.products.results import build_vrt_index

    vrt_paths: List[str] = []

    # VRT pour chaque dossier de produit TIF dans indices/
    if idx_dir.exists():
        for tif_dir in idx_dir.rglob("tif"):
            if not tif_dir.is_dir():
                continue
            vrt_path = tif_dir / "index.vrt"
            if list(tif_dir.glob("*.tif")):
                build_vrt_index(tif_dir, pattern="*.tif", output_name="index.vrt", log=log)
                if vrt_path.exists():
                    vrt_paths.append(str(vrt_path))

    return vrt_paths


def _collect_low_coverage_polygons(
    tif_dir: Path,
    threshold_percent: float,
    log: LogFn,
) -> list:
    """Polygones « zones mal couvertes » de la mosaïque COUVERTURE.

    Chemin nominal : l'``index.vrt`` construit juste avant (mosaïque complète,
    coutures de dalles fusionnées). Repli si le VRT manque (échec gdalbuildvrt) :
    extraction **dalle par dalle** — toutes les dalles sont couvertes, seules
    les zones à cheval sur deux dalles restent scindées (loggé).
    """
    try:
        from ...pipeline.coverage_polygons import extract_low_coverage_polygons
    except ImportError:  # pragma: no cover — fallback tests standalone
        from pipeline.coverage_polygons import extract_low_coverage_polygons

    raster = tif_dir / "index.vrt"
    if raster.exists():
        return extract_low_coverage_polygons(raster, threshold_percent)
    tifs = sorted(tif_dir.glob("*.tif"))
    if not tifs:
        return []
    if len(tifs) > 1:
        log(
            "Couverture : index.vrt absent — extraction dalle par dalle "
            "(les zones à cheval sur deux dalles ne sont pas fusionnées)"
        )
    polygons: list = []
    for tif in tifs:
        polygons.extend(extract_low_coverage_polygons(tif, threshold_percent))
    polygons.sort(key=lambda g: g.area, reverse=True)
    return polygons


def _build_coverage_polygons(
    idx_dir: Path,
    threshold_percent: float,
    log: LogFn,
) -> Optional[str]:
    """Vectorise les zones sous le seuil de ``indices/COUVERTURE/`` (si présent)
    vers ``indices/COUVERTURE/zones_mal_couvertes.gpkg``.

    Best-effort : toute erreur est loggée et renvoie ``None`` — la génération
    QA ne doit jamais avorter la finalisation (audit ROB).
    """
    tif_dir = idx_dir / "COUVERTURE" / "tif"
    if not tif_dir.is_dir():
        return None
    try:
        try:
            from ...pipeline.coverage_polygons import write_low_coverage_gpkg
        except ImportError:  # pragma: no cover — fallback tests standalone
            from pipeline.coverage_polygons import write_low_coverage_gpkg

        polygons = _collect_low_coverage_polygons(tif_dir, threshold_percent, log)
        if not polygons:
            log("Couverture : aucune zone sous le seuil — pas de polygones générés")
            return None
        gpkg = write_low_coverage_gpkg(
            polygons, idx_dir / "COUVERTURE" / "zones_mal_couvertes.gpkg"
        )
        if gpkg is None:
            return None
        log(f"Couverture : {len(polygons)} zone(s) mal couverte(s) → {gpkg.name}")
        return str(gpkg)
    except Exception as e:  # noqa: BLE001 — QA best-effort
        log(f"Note: polygones de couverture non générés ({e})")
        return None


def _list_gpkg_layers(gpkg_path: Path) -> List[str]:
    """Liste les couches d'un GeoPackage avec plusieurs méthodes de fallback."""
    # Méthode 1 : fiona
    try:
        import fiona
        return list(fiona.listlayers(str(gpkg_path)))
    except Exception:
        pass
    # Méthode 2 : osgeo.ogr (toujours disponible dans OSGeo4W)
    try:
        from osgeo import ogr
        ds = ogr.Open(str(gpkg_path))
        if ds is not None:
            layers = [ds.GetLayerByIndex(i).GetName() for i in range(ds.GetLayerCount())]
            ds = None
            return layers
    except Exception:
        pass
    # Méthode 3 : geopandas (lecture du fichier)
    try:
        import geopandas as gpd
        return gpd.list_layers(str(gpkg_path))["name"].tolist()
    except Exception:
        pass
    return []


def _gpkg_layer_feature_count(gpkg_path: Path, layer: str) -> int:
    """Nombre d'entités d'une couche GeoPackage, ou ``-1`` si indéterminable.

    Sert à écarter les couches **vides** à la collecte (ne pas charger/écrire une
    couche de détection sans entité → évite l'avertissement « CRS absent » sur une
    couche vide). En cas de doute (``-1``), l'appelant **conserve** la couche.
    """
    # osgeo.ogr (toujours présent dans OSGeo4W)
    try:
        from osgeo import ogr

        ds = ogr.Open(str(gpkg_path))
        if ds is not None:
            lyr = ds.GetLayerByName(layer)
            n = int(lyr.GetFeatureCount()) if lyr is not None else -1
            ds = None
            return n
    except Exception:
        pass
    # Repli geopandas/pyogrio
    try:
        import geopandas as gpd

        return int(len(gpd.read_file(str(gpkg_path), layer=layer)))
    except Exception:
        return -1


def _collect_shapefiles(det_dir: Path) -> List[str]:
    """Collecte les couches GeoPackage de détection CV (organisation entité-centrée).

    Parcourt ``detections/<entity_slug>/<entity_slug>.gpkg`` (et tout ``.gpkg``
    livrable) en **excluant** l'échafaudage technique ``detections/_technique/``
    (dumps d'inférence, GeoPackage modèle de repli vide) **et les couches vides**
    (0 entité — p.ex. vidées par le filtre d'aire minimale) : une couche vide ne
    doit être ni chargée ni écrite dans le ``.qgs``. Reste tolérant aux anciens
    layouts (``shapefiles/``) tant qu'ils ne sont pas sous ``_technique/``.
    """
    shapefile_paths: List[str] = []
    if not det_dir.exists():
        return shapefile_paths

    try:
        from ...pipeline.output_paths import DIR_TECHNIQUE
    except ImportError:
        from pipeline.output_paths import DIR_TECHNIQUE

    for gpkg_file in sorted(det_dir.rglob("*.gpkg")):
        # Exclure l'échafaudage technique (detections/_technique/…)
        if DIR_TECHNIQUE in gpkg_file.relative_to(det_dir).parts:
            continue
        layers = _list_gpkg_layers(gpkg_file)
        if layers:
            for layer in layers:
                # Sauter les couches vides (comptage == 0 ; -1 = inconnu → on garde).
                if _gpkg_layer_feature_count(gpkg_file, layer) == 0:
                    continue
                shapefile_paths.append(f"{gpkg_file}|layername={layer}")
        else:
            # Dernier recours : on inscrit le GPKG seul (nom de couche inconnu)
            shapefile_paths.append(str(gpkg_file))

    return shapefile_paths


def finalize_pipeline(
    *,
    output_dir: Path,
    cv_cfg: Dict[str, Any],
    rvt_params: Dict[str, Any],
    reporter: "ProgressReporter",
    slog: Optional["StructuredLogger"] = None,
    start_time: float,
    tiles_processed: int = 0,
    tiles_total: Optional[int] = None,
    active_products: Optional[List[str]] = None,
    extra_label: str = "",
    coverage_threshold_percent: float = 30.0,
    ui_config: Optional[Dict[str, Any]] = None,
    outcome: str = "success",
) -> None:
    """
    Finalisation commune à tous les runners :
    1. Création des index VRT (tif/)
    2. Collecte des shapefiles CV
    3. Chargement des couleurs de classes
    4. Logs de fin de pipeline
    5. Chargement des couches dans QGIS

    ``outcome`` (``success`` / ``cancelled`` / ``failed``) reflète l'issue
    RÉELLE du run : appelée depuis les ``finally`` des runners, cette
    fonction annonçait inconditionnellement « ✅ TERMINÉ AVEC SUCCÈS » même
    quand une exception fatale était en vol (AUDIT v2 ROB-14).
    ``tiles_total`` permet un décompte honnête (réussies/total) quand des
    éléments ont échoué ; défaut = ``tiles_processed``.
    """
    import time

    try:  # fallback standalone (tests : src/ sur le path)
        from ...pipeline.output_paths import indices_dir, detections_dir
    except ImportError:  # pragma: no cover
        from pipeline.output_paths import indices_dir, detections_dir
    from ..progress_reporter import report_busy, report_stage_id
    from ..progress_stages import Stage
    from ..user_narrator import create_user_narrator

    narrator = create_user_narrator(reporter)

    idx_dir = indices_dir(output_dir)
    det_dir = detections_dir(output_dir)
    def log(m: str) -> None:
        reporter.info(m)

    # 1. Création des index VRT
    # La finalisation occupe la bande 95→100 du plan (toutes phases CV/produits
    # terminées) : on entre à 95 pour garantir la continuité avec la phase
    # précédente, et la barre passe à 100 en sortie (load_layers fait).
    report_stage_id(reporter, Stage.FINALIZE)
    report_busy(reporter, False)  # garde-fou : sortir d'un éventuel mode indéterminé
    reporter.progress(95)
    reporter.stage("Création des index VRT")
    reporter.info("Création des fichiers VRT d’indexation...")
    narrator.finalize_start()
    vrt_paths = _collect_vrt_paths_and_build(idx_dir, det_dir, log)

    # Polygones QA « zones mal couvertes » (si le produit COUVERTURE a tourné).
    qa_paths: List[str] = []
    qa_gpkg = _build_coverage_polygons(idx_dir, coverage_threshold_percent, log)
    if qa_gpkg:
        qa_paths.append(qa_gpkg)

    # 2. Collecte des shapefiles CV (tous les runs)
    try:  # fallback standalone
        from ...pipeline.cv.class_utils import resolve_cv_runs
    except ImportError:  # pragma: no cover
        from pipeline.cv.class_utils import resolve_cv_runs
    cv_runs = resolve_cv_runs(cv_cfg or {})
    shapefile_paths: List[str] = _collect_shapefiles(det_dir)

    # 3. Les couleurs ne sont plus pré-calculées ici : chaque classe dérive sa
    # couleur de son nom via le registre partagé (pipeline.cv.class_color_registry),
    # à l'affichage comme à la génération (refonte couleurs 2026-06-12).

    # 3b. Le projet QGIS consolidé (detections_validation.qgs) n'est PLUS écrit ici.
    # L'API QGIS n'est pas thread-safe et finalize_pipeline tourne sur le thread worker ;
    # l'écriture XML à la main produisait des projets non relisables (CRS absent, couche
    # invalide). Le .qgs est désormais généré via l'API QGIS (QgsProject.write) sur le
    # thread principal, par ui/qgs_writer.write_validation_project, déclenché depuis
    # run_view._on_load_layers (même chemin que le chargement live, qui fonctionne).

    # 4. Génération du fichier metadata.json
    try:
        import json as _json
        import datetime as _dt

        meta = {
            "pipeline_version": "2.0",
            "date": _dt.datetime.now().isoformat(timespec="seconds"),
            "tiles_processed": tiles_processed,
            "active_products": active_products or [],
            "rvt_params": rvt_params or {},
            "cv_runs": [
                {
                    "model": r.get("selected_model", ""),
                    "target_rvt": r.get("target_rvt", ""),
                }
                for r in cv_runs
            ],
            # Correspondance entité (vocabulaire utilisateur) → dossier/fichier
            # livrable. Trace le lien entre ce que l'utilisateur a coché et où
            # les résultats atterrissent (detections/<slug>/<slug>.gpkg).
            "detections_entities": [
                {
                    "entity": ent.get("id", ""),
                    "label": ent.get("label", ""),
                    "slug": ent.get("slug", ""),
                    "folder": f"detections/{ent.get('slug', '')}",
                    "gpkg": f"detections/{ent.get('slug', '')}/{ent.get('slug', '')}.gpkg",
                    "is_derived": bool(ent.get("is_derived", False)),
                    "model": r.get("selected_model", ""),
                }
                for r in cv_runs
                for ent in (r.get("entities") or [])
            ],
            "structure": {
                "indices": str(idx_dir),
                "detections": str(det_dir),
            },
            "ui_config": ui_config or {},
        }
        meta_path = output_dir / "metadata.json"
        meta_path.write_text(_json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        reporter.info(f"Métadonnées enregistrées: {meta_path.name}")
    except Exception as _meta_e:
        reporter.info(f"Note: métadonnées non écrites ({_meta_e})")

    # 5. Logs de fin de pipeline — conditionnés par l'issue réelle (ROB-14).
    elapsed = time.time() - start_time
    products_list = active_products or []
    total = tiles_total if tiles_total is not None else tiles_processed
    success = outcome == "success"

    if slog:
        slog.end_pipeline(
            success=success,
            tiles_processed=tiles_processed,
            tiles_total=total,
            products=products_list,
        )
    # Annonce narrative à l'utilisateur (visible dans la fenêtre QGIS).
    # « ⏹ Traitement annulé » est émis par le runner après la finalisation —
    # rien à annoncer ici dans le cas cancelled.
    if outcome == "failed":
        narrator.pipeline_failed("erreur inattendue pendant le traitement")
    elif success:
        narrator.pipeline_complete(
            tiles_processed=tiles_processed,
            products=products_list,
            start_time=start_time,
        )
    if not slog:
        reporter.info("")
        reporter.info("════════════════════════════════════════════════════════════")
        if success:
            reporter.info("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
        elif outcome == "cancelled":
            reporter.info("⏹ PIPELINE ANNULÉ")
        else:
            reporter.info("❌ PIPELINE TERMINÉ AVEC ERREURS")
        reporter.info("════════════════════════════════════════════════════════════")
        reporter.info(f"  ⏱️ Durée totale : {elapsed:.1f}s")
        if extra_label:
            reporter.info(f"  📄 {extra_label} : {tiles_processed}")
        elif tiles_processed > 0:
            reporter.info(f"  📄 Dalles traitées : {tiles_processed}")
        reporter.info(f"  📦 Produits : {', '.join(products_list) if products_list else 'aucun'}")
        reporter.info("════════════════════════════════════════════════════════════")
        reporter.info("")

    # 5. Chargement des couches dans QGIS
    if vrt_paths or shapefile_paths or qa_paths:
        reporter.stage("Chargement des couches")
        reporter.info(f"Chargement de {len(vrt_paths)} VRT et {len(shapefile_paths)} shapefile(s) dans QGIS...")
        # ``colors_param`` conservé (signature de load_layers) mais désormais
        # ignoré côté UI : la couleur dérive du nom de classe via le registre.
        colors_param: list = []
        try:
            # 4ᵉ argument seulement si nécessaire : les reporters/fakes legacy
            # à 3 paramètres restent compatibles (aucun run sans COUVERTURE
            # ne change de signature d'appel).
            if qa_paths:
                reporter.load_layers(vrt_paths, shapefile_paths, colors_param, qa_paths)
            else:
                reporter.load_layers(vrt_paths, shapefile_paths, colors_param)
        except Exception as e:
            reporter.info(f"Note: Chargement des couches non disponible ({e})")

    # La barre ne saute à 100 % que sur un vrai succès (ROB-14) : après un
    # échec ou une annulation, une barre pleine contredirait le message.
    if success:
        reporter.stage("Terminé")
        reporter.progress(100)
    elif outcome == "cancelled":
        reporter.stage("Annulé")
    else:
        reporter.stage("Interrompu par une erreur")
