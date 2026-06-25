"""Chargement des couches résultat dans le projet QGIS (raster VRT + vecteurs).

Porté depuis ``main_dialog._load_layers_to_project`` pour être partagé par la
V2 (``run_view``). Fonctions QGIS-side : ne pas importer hors QGIS.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from ..app.progress_reporter import USER_INFO


def _apply_cluster_style(layer, logger: logging.Logger) -> None:
    """Hachures croisées noires + contour : style des zones de clustering."""
    try:
        from qgis.core import (
            QgsFillSymbol,
            QgsLinePatternFillSymbolLayer,
            QgsSimpleLineSymbolLayer,
            QgsSingleSymbolRenderer,
        )
        from qgis.PyQt.QtGui import QColor

        symbol = QgsFillSymbol()
        symbol.deleteSymbolLayer(0)
        for angle in (45, 135):
            hatch = QgsLinePatternFillSymbolLayer()
            hatch.setLineAngle(angle)
            hatch.setDistance(3.0)
            hatch.setLineWidth(0.4)
            hatch.setColor(QColor(0, 0, 0))
            symbol.appendSymbolLayer(hatch)
        outline = QgsSimpleLineSymbolLayer()
        outline.setColor(QColor(0, 0, 0))
        outline.setWidth(0.6)
        symbol.appendSymbolLayer(outline)
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))
        layer.triggerRepaint()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Impossible d'appliquer le style cluster: {e}")


def _apply_confidence_style(
    layer, base_color, confidence_threshold: float, logger: logging.Logger,
) -> None:
    """Symbologie catégorisée par tranche de confiance (contour coloré).

    ``base_color`` est la couleur de base RGB de la classe (registre) ; chaque
    tranche de confiance la décline en luminosité via ``apply_confidence``.
    """
    try:
        from qgis.core import (
            QgsCategorizedSymbolRenderer,
            QgsFillSymbol,
            QgsRendererCategory,
        )

        from ..pipeline.cv.class_utils import compute_confidence_bins
        from ..pipeline.cv.color_palette import apply_confidence

        bins = compute_confidence_bins(max(0.0, float(confidence_threshold or 0.0)))
        categories = []
        for b in bins:
            r, g, bl = apply_confidence(base_color, b["repr"])
            symbol = QgsFillSymbol.createSimple({
                "color": "0,0,0,0",
                "outline_color": f"{r},{g},{bl},255",
                "outline_width": "0.6",
                "outline_style": "solid",
            })
            categories.append(QgsRendererCategory(b["label"], symbol, b["label"]))
        if not categories:
            return
        layer.setRenderer(QgsCategorizedSymbolRenderer("conf_bin", categories))
        layer.triggerRepaint()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Impossible d'appliquer le style: {e}")


def _parse_gpkg_source(shp_path_str: str):
    """``(gpkg_path, layer_name, class_name)`` depuis ``path.gpkg|layername=X`` ou ``path.shp``.

    Pour un GPKG, ``layer_name`` = ``class_name`` = la table. Pour un shapefile legacy
    ``detections_<RVT>_<classe>``, ``class_name`` est extrait du stem.
    """
    if "|layername=" in shp_path_str:
        _gpkg, layer_name = shp_path_str.split("|layername=", 1)
        class_name = layer_name
    else:
        _gpkg = shp_path_str
        layer_name = Path(shp_path_str).stem
        parts = layer_name.split("_")
        class_name = (
            "_".join(parts[2:])
            if len(parts) >= 3 and parts[0] == "detections"
            else layer_name
        )
    return _gpkg, layer_name, class_name


def build_detection_vector_layer(
    ogr_source: str,
    layer_name: str,
    *,
    base_color,
    confidence_threshold: float,
    logger: logging.Logger,
):
    """Construit un ``QgsVectorLayer`` de détection valide, CRS fixé + symbologie appliquée.

    ``base_color`` : couleur de base RGB de la classe (résolue via le registre par
    l'appelant). Symbologie : cluster (hachures) si le champ ``nb_detect`` est
    présent, sinon catégorisée par tranche de confiance. **N'ajoute la couche à
    AUCUN projet** — l'appelant en est propriétaire. Source unique de vérité
    partagée par le chargement live (:func:`load_result_layers`) ET l'écriture du
    projet ``.qgs`` (``ui/qgs_writer.py``). ``None`` si la couche est invalide.
    """
    from qgis.core import QgsVectorLayer

    layer = QgsVectorLayer(ogr_source, layer_name, "ogr")
    if not layer.isValid():
        return None
    # Couche de détection vide (0 entité, p.ex. vidée par le filtre d'aire) : ne pas
    # la charger (évite l'avertissement « CRS absent » sur une couche sans donnée).
    # featureCount() == -1 = inconnu → on conserve (prudence).
    if layer.featureCount() == 0:
        logger.info(f"Couche détection vide ignorée: {layer_name}")
        return None
    _ensure_layer_crs(layer, logger)
    if layer.fields().indexFromName("nb_detect") >= 0:
        _apply_cluster_style(layer, logger)
    else:
        _apply_confidence_style(layer, base_color, confidence_threshold, logger)
    return layer


def apply_coverage_raster_symbology(layer, threshold_percent: float, logger: logging.Logger) -> None:
    """Pseudo-couleur du raster COUVERTURE : 0 % rouge opaque → seuil orange
    semi-transparent → 100 % entièrement transparent. Les lacunes « brillent »
    par-dessus le MNT, le reste s'efface. Partagée chargement live / ``.qgs``."""
    try:
        from qgis.core import (
            QgsColorRampShader,
            QgsRasterShader,
            QgsSingleBandPseudoColorRenderer,
        )
        from qgis.PyQt.QtGui import QColor

        thr = float(threshold_percent)
        items = [
            QgsColorRampShader.ColorRampItem(0.0, QColor(178, 24, 43, 255), "0 % (pas de points sol)"),
            QgsColorRampShader.ColorRampItem(thr, QColor(244, 165, 130, 180), f"{thr:.0f} % (seuil)"),
            QgsColorRampShader.ColorRampItem(100.0, QColor(255, 255, 255, 0), "100 % (bien couvert)"),
        ]
        shader_fn = QgsColorRampShader(0.0, 100.0)
        shader_fn.setColorRampType(QgsColorRampShader.Type.Interpolated)
        shader_fn.setClassificationMode(QgsColorRampShader.ClassificationMode.Continuous)
        shader_fn.setColorRampItemList(items)
        shader = QgsRasterShader()
        shader.setRasterShaderFunction(shader_fn)
        renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
        renderer.setClassificationMin(0.0)
        renderer.setClassificationMax(100.0)
        layer.setRenderer(renderer)
        layer.triggerRepaint()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Symbologie couverture non appliquée: {e}")


def low_coverage_layer_display_name(threshold_percent: float) -> str:
    """Nom d'affichage de la couche QA (partagé live / .qgs / réutilisation)."""
    return f"Zones mal couvertes (<{float(threshold_percent):.0f} %)"


def build_low_coverage_vector_layer(gpkg_path: str, threshold_percent: float, logger: logging.Logger):
    """Couche « Zones mal couvertes » stylée (hachures rouges 45°/135° + contour
    rouge, intérieur transparent), CRS fixé. N'ajoute la couche à AUCUN projet —
    l'appelant en est propriétaire. Partagée chargement live / ``.qgs``.
    ``None`` si invalide."""
    try:
        from qgis.core import (
            QgsFillSymbol,
            QgsLinePatternFillSymbolLayer,
            QgsSimpleLineSymbolLayer,
            QgsSingleSymbolRenderer,
            QgsVectorLayer,
        )
        from qgis.PyQt.QtGui import QColor

        from ..pipeline.coverage_polygons import GPKG_LAYER_NAME

        name = low_coverage_layer_display_name(threshold_percent)
        layer = QgsVectorLayer(f"{gpkg_path}|layername={GPKG_LAYER_NAME}", name, "ogr")
        if not layer.isValid():
            # Repli : GPKG mono-couche sans nom de couche connu.
            layer = QgsVectorLayer(str(gpkg_path), name, "ogr")
            if not layer.isValid():
                return None
        _ensure_layer_crs(layer, logger)
        red = QColor(200, 30, 30)
        symbol = QgsFillSymbol()
        symbol.deleteSymbolLayer(0)
        for angle in (45, 135):
            hatch = QgsLinePatternFillSymbolLayer()
            hatch.setLineAngle(angle)
            hatch.setDistance(3.0)
            hatch.setLineWidth(0.4)
            hatch.setColor(red)
            symbol.appendSymbolLayer(hatch)
        outline = QgsSimpleLineSymbolLayer()
        outline.setColor(red)
        outline.setWidth(0.6)
        symbol.appendSymbolLayer(outline)
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))
        layer.triggerRepaint()
        return layer
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Couche zones mal couvertes non construite: {e}")
        return None


def load_result_layers(
    vrt_paths: list,
    shapefile_paths: list,
    class_colors: Optional[list],
    logger: logging.Logger,
    confidence_threshold: float = 0.0,
    entity_labels: Optional[dict] = None,
    derived_slugs: Optional[set] = None,
    min_conf_by_slug: Optional[dict] = None,
    qa_paths: Optional[list] = None,
    coverage_threshold_percent: float = 30.0,
) -> None:
    """Charge les VRT (raster) et vecteurs (détections) dans le projet QGIS,
    applique la symbologie, puis zoome sur l'étendue combinée (déféré).

    ``derived_slugs`` (slugs d'entités dérivées) + ``entity_labels`` (slug→libellé)
    pilotent le **regroupement** des couches de détection : une couche dont le slug
    de dossier est dans ``derived_slugs`` est placée dans un groupe d'arbre nommé
    par son libellé (zone + constituants ensemble, ex. « Regroupement de cratères ») ;
    toutes les autres restent à plat. Miroir de ``ui/qgs_writer.write_validation_project``."""
    try:
        from qgis.core import (
            QgsProject,
            QgsRasterLayer,
            QgsRectangle,
        )

        from ..pipeline.cv.class_color_registry import color_for_class

        project = QgsProject.instance()
        root = project.layerTreeRoot()
        derived_slugs = derived_slugs or set()
        entity_labels = entity_labels or {}
        min_conf_by_slug = min_conf_by_slug or {}
        loaded_count = 0
        combined_extent = QgsRectangle()
        loaded_layers: List = []
        # ``class_colors`` n'est plus consulté (la couleur dérive du nom de classe
        # via le registre) ; le paramètre est conservé pour ne pas casser les
        # appelants existants.

        # ── Rasters (VRT d'indices) ──
        for vrt_path in vrt_paths:
            if not vrt_path:
                continue
            vrt_path_str = str(vrt_path)
            _vp = Path(vrt_path_str)
            rvt_type = _vp.parent.parent.name if _vp.parent.name == "tif" else _vp.parent.name
            layer_name = f"index_{rvt_type}" if rvt_type else "index"

            if _reuse_existing(project, layer_name, vrt_path_str, loaded_layers, combined_extent):
                logger.info(f"Couche raster déjà présente: {layer_name}")
                continue

            layer = QgsRasterLayer(vrt_path_str, layer_name, "gdal")
            if layer.isValid():
                _ensure_layer_crs(layer, logger)
                if rvt_type == "COUVERTURE":
                    apply_coverage_raster_symbology(layer, coverage_threshold_percent, logger)
                project.addMapLayer(layer)
                loaded_layers.append(layer)
                combined_extent = _combine(combined_extent, layer.extent())
                loaded_count += 1
                logger.info(f"Couche raster chargée: {layer_name}")
            else:
                logger.warning(f"Impossible de charger le VRT: {vrt_path_str}")

        # ── Vecteurs (détections : shapefile ou GPKG) ──
        for shp_path in shapefile_paths:
            if not shp_path:
                continue
            shp_path_str = str(shp_path)
            _gpkg, layer_name, class_name = _parse_gpkg_source(shp_path_str)
            ogr_source = shp_path_str
            # slug d'entité = dossier du GeoPackage (detections/<slug>/<slug>.gpkg)
            slug = Path(_gpkg).parent.name

            base_color = color_for_class(class_name)

            if _reuse_existing(project, layer_name, ogr_source, loaded_layers, combined_extent):
                logger.info(f"Couche vecteur déjà présente: {layer_name}")
                continue

            # Seuil par entité (= seuil du run qui a binné conf_bin) ; repli sur le
            # seuil global pour un slug absent de la map (run legacy / sécurité).
            layer_conf = min_conf_by_slug.get(slug, confidence_threshold)
            layer = build_detection_vector_layer(
                ogr_source, layer_name, base_color=base_color,
                confidence_threshold=layer_conf, logger=logger,
            )
            if layer is not None:
                # Entité dérivée → groupe d'arbre (zone + constituants ensemble) ;
                # sinon couche à plat (racine), comme le .qgs.
                if slug in derived_slugs:
                    label = entity_labels.get(slug, slug)
                    project.addMapLayer(layer, False)  # pas à la racine
                    # insertGroup(0) (et non addGroup, qui appende EN BAS) : le groupe
                    # se place EN HAUT de l'arbre, donc au-dessus des rasters MNT/indices.
                    grp = root.findGroup(label) or root.insertGroup(0, label)
                    grp.addLayer(layer)
                else:
                    project.addMapLayer(layer)
                loaded_layers.append(layer)
                combined_extent = _combine(combined_extent, layer.extent())
                loaded_count += 1
                logger.info(
                    f"Couche vecteur chargée: {layer_name} "
                    f"(classe={class_name}, RGB{base_color})"
                )
            else:
                logger.warning(f"Impossible de charger la couche: {ogr_source}")

        # ── Vecteur QA : zones mal couvertes (tout en haut de l'arbre) ──
        qa_name = low_coverage_layer_display_name(coverage_threshold_percent)
        for qa_path in (qa_paths or []):
            if not qa_path:
                continue
            qa_str = str(qa_path)
            if _reuse_existing(project, qa_name, qa_str, loaded_layers, combined_extent):
                logger.info(f"Couche QA déjà présente: {qa_name}")
                continue
            layer = build_low_coverage_vector_layer(qa_str, coverage_threshold_percent, logger)
            if layer is None:
                logger.warning(f"Impossible de charger la couche QA: {qa_str}")
                continue
            project.addMapLayer(layer, False)
            root.insertLayer(0, layer)  # au-dessus des groupes d'entités et des rasters
            loaded_layers.append(layer)
            combined_extent = _combine(combined_extent, layer.extent())
            loaded_count += 1
            logger.info(f"Couche QA chargée: {qa_name}")

        if loaded_count > 0:
            from ..app.user_narrator import _human_count
            logger.log(
                USER_INFO,
                f"📂 {_human_count(loaded_count, 'couche ajoutée', 'couches ajoutées')} au projet QGIS",
            )

        if loaded_layers:
            _deferred_zoom(loaded_layers, project, logger)

    except Exception as e:  # noqa: BLE001
        logger.error(f"Erreur lors du chargement des couches: {e}")


# ── helpers ─────────────────────────────────────────────────────────


def _combine(extent, new_extent):
    if extent.isNull():
        return new_extent
    extent.combineExtentWith(new_extent)
    return extent


def _ensure_layer_crs(layer, logger, fallback_authid: str = "EPSG:2154") -> None:
    """Affecte le CRS canonique du pipeline si la couche n'en a pas d'exploitable.

    Un CRS local « unnamed » (ex. MNT que PDAL a émis sans georéférencement
    reconnu) est invalide ou dépourvu de code d'autorité (``authid`` vide) côté
    QGIS. Comme tout le pipeline est en Lambert-93, on lui **affecte** EPSG:2154
    (sans reprojeter — les coordonnées sont déjà correctes), ce qui évite l'erreur
    « Pas de transformation disponible entre unnamed et EPSG:2154 » au zoom final.
    Garde-fou défensif : la correction de fond se fait à la source (``mnt.py``).
    """
    try:
        from qgis.core import QgsCoordinateReferenceSystem

        crs = layer.crs()
        if crs.isValid() and crs.authid():
            return  # CRS exploitable (code d'autorité présent) → ne pas toucher
        target = QgsCoordinateReferenceSystem(fallback_authid)
        if target.isValid():
            layer.setCrs(target)
            logger.warning(f"CRS absent/local sur « {layer.name()} » → affecté {fallback_authid}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Impossible de fixer le CRS de « {layer.name()} »: {e}")


def _reuse_existing(project, layer_name, source, loaded_layers, combined_extent) -> bool:
    """Si une couche de même nom + source existe déjà, la réutilise.

    Défensif : on force une relecture du datasource (``reload``) avant réutilisation.
    Si le fichier (VRT régénéré, GPKG réécrit) a changé sur disque depuis son
    chargement initial, QGIS afficherait sinon la version en mémoire (périmée).
    Best-effort — la purge au lancement (``purge_output_dir_layers``) reste la
    parade principale ; ceci couvre une couche chargée par un autre chemin.
    """
    for existing in project.mapLayersByName(layer_name):
        if existing.source() == source:
            try:
                existing.reload()
                existing.triggerRepaint()
            except Exception:  # noqa: BLE001 — relecture best-effort, jamais bloquante
                pass
            loaded_layers.append(existing)
            _combine(combined_extent, existing.extent())
            return True
    return False


def purge_output_dir_layers(output_dir, logger: logging.Logger) -> int:
    """Retire du projet QGIS les couches périmées d'un ``output_dir`` avant un re-run.

    À appeler sur le **thread principal, au lancement** du run : libère les datasets
    ``index_<produit>.vrt`` / GPKG encore détenus par QGIS depuis un run précédent dans le
    **même** dossier de sortie, AVANT que le worker ne régénère les VRT. Sans cela,
    QGIS réécrit sa version en mémoire (périmée) par-dessus le VRT fraîchement
    régénéré → les dalles ajoutées restent invisibles. Best-effort : toute erreur est
    loggée sans jamais bloquer le lancement. Retourne le nombre de couches retirées.

    La décision (quelles couches) est déléguée au helper pur testable
    ``app.services.layer_purge.select_layers_to_purge`` ; ici on ne fait que les
    appels QGIS (énumération + ``removeMapLayers``).
    """
    try:
        from qgis.core import QgsProject

        from ..app.services.layer_purge import select_layers_to_purge

        if not output_dir:
            return 0
        project = QgsProject.instance()
        sources = {lid: lyr.source() for lid, lyr in project.mapLayers().items()}
        to_remove = select_layers_to_purge(sources, str(output_dir))
        if to_remove:
            project.removeMapLayers(to_remove)  # libère chaque dataProvider/dataset
            logger.info(
                f"{len(to_remove)} couche(s) périmée(s) du run précédent retirée(s) "
                "avant régénération (même dossier de sortie)."
            )
        return len(to_remove)
    except Exception as e:  # noqa: BLE001 — purge best-effort, jamais bloquante
        logger.warning(f"Purge des couches périmées impossible: {e}")
        return 0


def _deferred_zoom(loaded_layers, project, logger) -> None:
    """Zoom différé (QTimer 0) : le canvas n'a pas encore synchronisé son CRS
    dans le même tour de boucle que addMapLayer."""
    try:
        from qgis.PyQt.QtCore import QTimer
        from qgis.utils import iface

        if not (iface and iface.mapCanvas()):
            logger.info("iface indisponible — zoom non effectué")
            return
        layers_for_zoom = list(loaded_layers)

        def _do_zoom():
            try:
                from qgis.core import QgsCoordinateTransform, QgsRectangle
                canvas = iface.mapCanvas()
                canvas_crs = canvas.mapSettings().destinationCrs()
                combined = QgsRectangle()
                combined.setMinimal()
                for lay in layers_for_zoom:
                    ext = lay.extent()
                    if ext.isNull() or ext.isEmpty():
                        continue
                    l_crs = lay.crs()
                    if l_crs.isValid() and canvas_crs.isValid() and l_crs != canvas_crs:
                        try:
                            xf = QgsCoordinateTransform(l_crs, canvas_crs, project)
                            ext = xf.transformBoundingBox(ext)
                        except Exception:
                            continue
                    if not (ext.isNull() or ext.isEmpty()):
                        combined.combineExtentWith(ext)
                if not (combined.isNull() or combined.isEmpty()):
                    combined.scale(1.05)
                    canvas.setExtent(combined)
                    canvas.refresh()
                    logger.log(USER_INFO, "Zoom sur l'étendue des résultats")
                    return
                iface.setActiveLayer(layers_for_zoom[0])
                iface.zoomToActiveLayer()
                logger.log(USER_INFO, "Zoom sur l'étendue des résultats")
            except Exception as ze:  # noqa: BLE001
                logger.warning(f"Impossible de zoomer: {ze}")

        QTimer.singleShot(0, _do_zoom)
    except Exception as zoom_err:  # noqa: BLE001
        logger.warning(f"Impossible de programmer le zoom: {zoom_err}")
