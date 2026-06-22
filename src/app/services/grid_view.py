"""Décisions pures pour la couche quadrillage IGN sur le canevas (sans QGIS).

La couche quadrillage (~490 k dalles) n'est rendue qu'une fois suffisamment zoomé
(``setScaleBasedVisibility``). L'outil-carte et la couche vivent dans
``src/ui/map_tools/`` (non collectés par pytest). On isole ici les **décisions**
testables hors QGIS :

- :func:`decide_grid_reuse` — réutiliser / ré-ajouter / recharger une couche déjà
  présente. Le cas « présente au registre mais absente de l'arbre » est la cause
  racine du bug « grille absente, réparée par redémarrage » (une couche orpheline
  de l'arbre n'est jamais dessinée).
- :func:`grid_is_hidden` — la grille est-elle masquée à l'échelle courante ? Sert
  aussi de critère « vue perdue » : si la grille est masquée à l'activation, on
  recadre sur la France métropolitaine (:data:`FRANCE_METRO_2154_BBOX`).

Module pur (aucun import QGIS) → collectable et testable hors QGIS.
"""
from __future__ import annotations

from typing import Tuple

# Emprise de la France métropolitaine en Lambert-93 (EPSG:2154) — (xmin, ymin, xmax,
# ymax). Source unique partagée avec ``grid_layer`` pour cadrer la métropole (le
# quadrillage IGN LiDAR HD est en L93 : pas de DOM-TOM, non représentables en L93).
FRANCE_METRO_2154_BBOX: Tuple[float, float, float, float] = (
    99000.0,
    6046000.0,
    1242000.0,
    7110000.0,
)


def decide_grid_reuse(*, is_valid: bool, in_tree: bool) -> str:
    """Que faire d'une couche quadrillage déjà présente au registre du projet ?

    - ``"reload"`` : couche invalide (source perdue/verrouillée) → la retirer puis
      recharger frais.
    - ``"readd"`` : couche valide mais **absente de l'arbre de couches** → recréer
      son nœud (sinon le canevas ne la dessine jamais).
    - ``"reuse"`` : couche valide et présente dans l'arbre → réutiliser telle quelle.
    """
    if not is_valid:
        return "reload"
    return "reuse" if in_tree else "readd"


def grid_is_hidden(scale_based: bool, canvas_scale: float, min_scale: float) -> bool:
    """La grille est-elle masquée à l'échelle courante (vue trop dézoomée) ?

    ``canvas_scale`` et ``min_scale`` sont des **dénominateurs** : plus grands =
    plus dézoomé. La grille est masquée au-delà du seuil (strictement) ; à
    l'égalité elle reste visible.
    """
    return scale_based and canvas_scale > min_scale
