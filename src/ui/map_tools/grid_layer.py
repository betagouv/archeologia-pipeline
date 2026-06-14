"""Chargement / retrait de la couche quadrillage IGN sur le canevas QGIS.

Sépare la gestion de la couche (projet, symbologie, visibilité d'échelle, groupe)
de l'outil de sélection lui-même (:mod:`.tile_picker_tool`). Réutilise les motifs
de :mod:`ui.layer_loader` (provider ``ogr``, validation ``isValid``, repli CRS
EPSG:2154, groupes via le ``layerTreeRoot``).

La couche est lourde (~490 k dalles) : on la masque tant qu'on n'est pas
suffisamment zoomé (``setScaleBasedVisibility``) pour ne jamais rendre toute la
France d'un coup, et on s'appuie sur l'index ``.qix`` pour les requêtes spatiales.
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

# Emprise approximative de la France métropolitaine en Lambert-93 (EPSG:2154),
# pour recadrer la vue quand l'utilisateur est « perdu » (vue monde / ailleurs).
_FRANCE_METRO_2154 = (99000.0, 6046000.0, 1242000.0, 7110000.0)


def load_quadrillage_layer(plugin_root: Path):
    """Charge (ou réutilise) la couche quadrillage et la renvoie, ou ``None``.

    Renvoie ``None`` si le fichier est introuvable ou la couche invalide —
    l'appelant (UI) se charge d'afficher un message à l'utilisateur.
    """
    from qgis.core import (
        QgsCoordinateReferenceSystem,
        QgsFillSymbol,
        QgsProject,
        QgsSingleSymbolRenderer,
        QgsVectorLayer,
    )

    # Import différé (module pur, pas de QGIS) — résolu au runtime côté plugin.
    from ...pipeline.ign.quadrillage_paths import resolve_quadrillage_path

    path = resolve_quadrillage_path(plugin_root)
    if not path.exists():
        logger.warning("Quadrillage introuvable : %s", path)
        return None

    project = QgsProject.instance()
    source = str(path)

    # Réutiliser si déjà chargée (éviter de recharger ~490 k entités).
    for lyr in project.mapLayersByName(GRID_LAYER_NAME):
        if lyr.source().split("|")[0] == source:
            return lyr

    layer = QgsVectorLayer(source, GRID_LAYER_NAME, "ogr")
    if not layer.isValid():
        logger.warning("Couche quadrillage invalide : %s", source)
        return None

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


def zoom_to_france_if_lost(layer) -> None:
    """Recadre le canevas sur la France si la vue est trop dézoomée (grille masquée).

    Différé (``QTimer.singleShot(0,…)``, motif ``layer_loader``) : le CRS du
    canevas n'est pas forcément synchronisé dans le même tour de boucle que
    l'ajout de la couche. Ne fait rien si l'utilisateur est déjà assez zoomé
    pour voir des dalles (on ne perturbe pas sa vue).
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
            from qgis.core import QgsCoordinateTransform, QgsProject, QgsRectangle

            canvas = iface.mapCanvas()
            # Déjà assez zoomé pour voir la grille → ne pas toucher à la vue.
            if not (layer.hasScaleBasedVisibility() and canvas.scale() > layer.minimumScale()):
                return
            canvas_crs = canvas.mapSettings().destinationCrs()
            layer_crs = layer.crs()
            xmin, ymin, xmax, ymax = _FRANCE_METRO_2154
            target = QgsRectangle(xmin, ymin, xmax, ymax)
            if canvas_crs.isValid() and layer_crs.isValid() and canvas_crs != layer_crs:
                target = QgsCoordinateTransform(
                    layer_crs, canvas_crs, QgsProject.instance()
                ).transformBoundingBox(target)
            canvas.setExtent(target)
            canvas.refresh()
        except Exception:
            logger.exception("zoom_to_france_if_lost a échoué")

    QTimer.singleShot(0, _do)
