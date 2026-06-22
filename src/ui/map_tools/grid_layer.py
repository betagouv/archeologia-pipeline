"""Chargement / retrait de la couche quadrillage IGN sur le canevas QGIS.

Sépare la gestion de la couche (projet, symbologie, visibilité d'échelle, groupe)
de l'outil de sélection lui-même (:mod:`.tile_picker_tool`). Réutilise les motifs
de :mod:`ui.layer_loader` (provider ``ogr``, validation ``isValid``, repli CRS
EPSG:2154, groupes via le ``layerTreeRoot``, **``triggerRepaint`` + ``canvas.refresh``
après ajout/réutilisation** — sans quoi une couche ajoutée hors légende peut ne pas
être (re)dessinée immédiatement : « grille absente jusqu'au redémarrage »).

La couche est lourde (~490 k dalles) : on la masque tant qu'on n'est pas
suffisamment zoomé (``setScaleBasedVisibility``) pour ne jamais rendre toute la
France d'un coup, et on s'appuie sur l'index ``.qix`` pour les requêtes spatiales.
Les décisions pures (réutilisation, masquage à l'échelle, emprise métropole) vivent
dans :mod:`app.services.grid_view` (testables hors QGIS).
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

GRID_LAYER_NAME = "Quadrillage IGN LiDAR HD"
GRID_GROUP_NAME = "Quadrillage IGN"

# Échelle (dénominateur) la plus « dézoomée » à laquelle la grille reste visible.
# Au-delà (vue France entière), on la masque : rendre 490 k entités serait lent.
_MIN_VISIBLE_SCALE = 1_500_000.0


def _schedule_canvas_repaint(layer) -> None:
    """Invalide le cache de rendu de la couche et programme un rafraîchissement.

    Motif :func:`ui.layer_loader._deferred_zoom` : ``triggerRepaint`` (invalide le
    cache de *cette* couche) puis ``canvas.refresh`` différé (``QTimer 0`` — le
    canevas finalise son CRS/outil au tour de boucle suivant). Sans cela, une couche
    ajoutée via ``addMapLayer(..., addToLegend=False)`` peut ne pas être dessinée
    avant un déplacement/zoom ou un redémarrage de QGIS. Best-effort.
    """
    try:
        layer.triggerRepaint()
    except Exception:  # noqa: BLE001 — repaint best-effort, jamais bloquant
        pass
    try:
        from qgis.PyQt.QtCore import QTimer
        from qgis.utils import iface

        if iface is None or iface.mapCanvas() is None:
            return
        QTimer.singleShot(0, lambda: iface.mapCanvas().refresh())
    except Exception:  # noqa: BLE001
        logger.exception("Rafraîchissement du canevas impossible")


def load_quadrillage_layer(plugin_root: Path):
    """Charge (ou réutilise) la couche quadrillage et la renvoie, ou ``None``.

    Renvoie ``None`` si le fichier est introuvable ou la couche invalide —
    l'appelant (UI) se charge d'afficher un message à l'utilisateur. Réutilise une
    couche déjà chargée (évite de recharger ~490 k entités) **après l'avoir
    consolidée** : une couche présente au registre mais orpheline de l'arbre de
    couches n'est jamais dessinée (cause racine du bug « grille absente »).
    """
    from qgis.core import (
        QgsCoordinateReferenceSystem,
        QgsFillSymbol,
        QgsProject,
        QgsSingleSymbolRenderer,
        QgsVectorLayer,
    )

    # Import différé (module pur, pas de QGIS) — résolu au runtime côté plugin.
    from ...app.services.grid_view import decide_grid_reuse
    from ...pipeline.ign.quadrillage_paths import resolve_quadrillage_path

    path = resolve_quadrillage_path(plugin_root)
    logger.info("Quadrillage : path=%s exists=%s", path, path.exists())
    if not path.exists():
        logger.warning("Quadrillage introuvable : %s", path)
        return None

    project = QgsProject.instance()
    source = str(path)

    # Réutiliser si déjà chargée — mais valider d'abord (valide ? dans l'arbre ?).
    candidate = None
    for lyr in project.mapLayersByName(GRID_LAYER_NAME):
        if lyr.source().split("|")[0] == source:
            candidate = lyr
            break
    if candidate is not None:
        in_tree = project.layerTreeRoot().findLayer(candidate.id()) is not None
        decision = decide_grid_reuse(is_valid=candidate.isValid(), in_tree=in_tree)
        logger.info(
            "Quadrillage : couche existante isValid=%s in_tree=%s → %s",
            candidate.isValid(), in_tree, decision,
        )
        if decision == "reuse":
            _schedule_canvas_repaint(candidate)
            return candidate
        if decision == "readd":
            # Valide mais absente de l'arbre → recréer son nœud (sinon non dessinée).
            root = project.layerTreeRoot()
            group = root.findGroup(GRID_GROUP_NAME) or root.insertGroup(0, GRID_GROUP_NAME)
            group.addLayer(candidate)
            _schedule_canvas_repaint(candidate)
            return candidate
        # decision == "reload" : couche invalide → retirer puis recharger frais.
        remove_quadrillage_layer()

    layer = QgsVectorLayer(source, GRID_LAYER_NAME, "ogr")
    if not layer.isValid():
        logger.warning("Couche quadrillage invalide : %s", source)
        return None
    logger.info("Quadrillage : chargement frais crs=%s", layer.crs().authid())

    if not layer.crs().isValid():
        layer.setCrs(QgsCoordinateReferenceSystem("EPSG:2154"))

    # Symbologie : contour seul (remplissage transparent) → le fond reste visible.
    symbol = QgsFillSymbol.createSimple({
        "style": "no",                       # pas de remplissage
        "outline_color": "230,80,40,255",    # orange/rouge discret
        "outline_width": "0.2",
    })
    layer.setRenderer(QgsSingleSymbolRenderer(symbol))

    # Ne pas dessiner toute la France : visible seulement une fois zoomé.
    layer.setScaleBasedVisibility(True)
    layer.setMinimumScale(_MIN_VISIBLE_SCALE)
    layer.setMaximumScale(0.0)

    root = project.layerTreeRoot()
    group = root.findGroup(GRID_GROUP_NAME) or root.insertGroup(0, GRID_GROUP_NAME)
    project.addMapLayer(layer, False)
    group.addLayer(layer)
    _schedule_canvas_repaint(layer)
    return layer


def remove_quadrillage_layer() -> None:
    """Retire la couche quadrillage et son groupe (si vide). Idempotent."""
    try:
        from qgis.core import QgsProject
    except ImportError:
        return
    project = QgsProject.instance()
    for lyr in project.mapLayersByName(GRID_LAYER_NAME):
        project.removeMapLayer(lyr.id())
    root = project.layerTreeRoot()
    group = root.findGroup(GRID_GROUP_NAME)
    if group is not None and not group.children():
        root.removeChildNode(group)


def zoom_to_france_metro_if_hidden(layer) -> None:
    """Cadre la France métropolitaine si la grille est masquée (vue « perdue »).

    Différé (``QTimer.singleShot(0,…)``, motif ``layer_loader``) : le CRS du canevas
    n'est pas forcément synchronisé dans le même tour de boucle que l'ajout de la
    couche. Ne fait **rien** si l'utilisateur voit déjà des dalles (vue déjà zoomée
    sur sa zone) — on ne perturbe pas une vue de travail. Sinon, recadre sur la
    France métropolitaine (:data:`FRANCE_METRO_2154_BBOX`) ; à cette échelle la
    grille reste masquée (490 k dalles), l'utilisateur zoome ensuite sur sa zone
    (le bandeau persistant l'y invite, côté UI).

    L'emprise est exprimée en **EPSG:2154 explicite** (et non ``layer.crs()``) puis
    transformée vers le CRS du canevas → robuste quel que soit le CRS de la couche
    ou du projet.
    """
    try:
        from qgis.PyQt.QtCore import QTimer
        from qgis.utils import iface
    except ImportError:
        return
    if iface is None or iface.mapCanvas() is None:
        return

    def _do() -> None:
        try:
            from qgis.core import (
                QgsCoordinateReferenceSystem,
                QgsCoordinateTransform,
                QgsProject,
                QgsRectangle,
            )

            from ...app.services.grid_view import FRANCE_METRO_2154_BBOX, grid_is_hidden

            canvas = iface.mapCanvas()
            min_scale = layer.minimumScale()
            hidden = grid_is_hidden(layer.hasScaleBasedVisibility(), canvas.scale(), min_scale)
            logger.info(
                "Quadrillage zoom : échelle=%.0f minScale=%.0f masquée=%s",
                canvas.scale(), min_scale, hidden,
            )
            if not hidden:
                return  # déjà assez zoomé pour voir la grille → ne pas perturber la vue

            xmin, ymin, xmax, ymax = FRANCE_METRO_2154_BBOX
            target = QgsRectangle(xmin, ymin, xmax, ymax)  # exprimée en L93
            src_crs = QgsCoordinateReferenceSystem("EPSG:2154")
            canvas_crs = canvas.mapSettings().destinationCrs()
            if canvas_crs.isValid() and src_crs.isValid() and canvas_crs != src_crs:
                target = QgsCoordinateTransform(
                    src_crs, canvas_crs, QgsProject.instance()
                ).transformBoundingBox(target)
            canvas.setExtent(target)
            canvas.refresh()
            logger.info("Quadrillage zoom : recadré sur la France métropolitaine")
        except Exception:  # noqa: BLE001
            logger.exception("zoom_to_france_metro_if_hidden a échoué")

    QTimer.singleShot(0, _do)
