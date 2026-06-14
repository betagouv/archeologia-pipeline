"""Résolveur de chemin du quadrillage IGN LiDAR HD (pur, sans QGIS/OGR).

Le quadrillage est la grille des dalles téléchargeables (un polygone par dalle,
attributs ``nom_pkk`` + ``url_telech``). C'est un shapefile lourd (~176 Mo,
~490 k entités) qui, **sans index spatial**, est pénible à manipuler de façon
interactive (chaque clic/rendu balaie toutes les entités). On l'accélère avec un
sidecar ``.qix`` (R-tree, ~2 Mo, cf. ``dev/build_quadrillage_index.py``) que
GDAL/OGR et QGIS utilisent automatiquement — sans changer le format livré.

Ce module fournit la **source de vérité unique** du chemin, partagée par
:mod:`pipeline.ign.tile_resolver` (intersection polygone) et l'outil UI de
sélection des dalles sur le canevas. Il renvoie le shapefile (notre artefact
livré). Un ``.gpkg`` équivalent reste accepté en option : s'il est présent, il
est préféré — la bascule est ainsi transparente si on régénère un jour la grille
dans ce format.

Aucun import QGIS/OGR ici : le module reste importable hors QGIS (et donc
collectable par pytest), et ``tile_resolver`` peut l'importer en intra-paquet
(``from .quadrillage_paths import ...``) sans dépendance croisée vers ``app``.
"""
from __future__ import annotations

from pathlib import Path

_QUADRILLAGE_DIR = Path("data") / "quadrillage_france"
_BASENAME = "TA_diff_pkk_lidarhd_classe"

QUADRILLAGE_GPKG_RELPATH = _QUADRILLAGE_DIR / f"{_BASENAME}.gpkg"
QUADRILLAGE_SHP_RELPATH = _QUADRILLAGE_DIR / f"{_BASENAME}.shp"


def resolve_quadrillage_path(plugin_root: Path) -> Path:
    """Chemin du quadrillage à utiliser, relatif à ``plugin_root``.

    Préfère le GeoPackage slim (léger + R-tree) s'il existe, sinon le shapefile
    legacy. Si aucun n'existe, renvoie le chemin ``.shp`` (la vérification
    d'existence et le message d'erreur restent à la charge de l'appelant).
    """
    gpkg = plugin_root / QUADRILLAGE_GPKG_RELPATH
    if gpkg.exists():
        return gpkg
    return plugin_root / QUADRILLAGE_SHP_RELPATH
