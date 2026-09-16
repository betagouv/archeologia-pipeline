"""Calcul du pourcentage de couverture locale des points sol (produit COUVERTURE).

Module **pur numpy** (aucun import QGIS/GDAL/rasterio) : testable standalone.

**Origine de la méthode.** Elle est publiée par Le Jeune, Dardignac & David
(2017), « Bercé avant la forêt. Premiers résultats des traitements
cartographiques et prospections lidar sur la forêt de Bercé (Sarthe) », *Revue
forestière française* LXIX-4-5, p. 519-543, doi 10.4267/2042/67877 (HAL
hal-03447369) : « la couverture lidar est ici définie par un pourcentage de
cellules raster documentée par au moins un point de mesure lidar au sol […] par
l'usage d'une fenêtre flottante circulaire de cinq cellules de diamètre », et
leurs cartes distinguent la « couverture lidar faible (< 30 %) » — d'où le seuil
par défaut du plugin. L'original tourne sous GRASS GIS.

Le code transpose nativement le bloc « Coverage » de ``PCSAPS.sh`` (Y. Le Jeune,
GRASS ``r.neighbors -c size=5``), script non publié dont le sigle n'a pas pu
être élucidé (recherche du 2026-09-16 : aucune occurrence, ni en ligne ni dans
les rapports disponibles — ne pas le développer au hasard). La citation
ci-dessus est la source à donner.

Pour chaque cellule : % de cellules contenant au moins un point dans un disque
de 5 cellules de diamètre, normalisé par le nombre réel de cellules de la
fenêtre — ce qui corrige le biais de bord du ``*100/13`` fixe du script.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

#: Diamètre (en cellules) du disque d'analyse — fixé par design (spec 2026-06-11).
DISC_DIAMETER_CELLS = 5

#: Valeur NoData du raster de couverture produit (uint8, % dans [0, 100]).
COVERAGE_NODATA = 255


def disc_offsets(diameter: int = DISC_DIAMETER_CELLS) -> List[Tuple[int, int]]:
    """Décalages ``(dy, dx)`` du disque : distance euclidienne ≤ rayon ``(d-1)//2``.

    ``diameter=5`` → rayon 2 → 13 cellules (le « /13 » de PCSAPS).
    """
    rad = (int(diameter) - 1) // 2
    return [
        (dy, dx)
        for dy in range(-rad, rad + 1)
        for dx in range(-rad, rad + 1)
        if dy * dy + dx * dx <= rad * rad
    ]


def compute_coverage_percent(
    density: np.ndarray,
    *,
    density_nodata: Optional[float] = None,
    diameter: int = DISC_DIAMETER_CELLS,
) -> np.ndarray:
    """Raster uint8 0–100 : % de cellules « avec points » dans le disque.

    - ``density`` : tableau 2D (compte de points par cellule, ``pdal:density``).
    - Cellule « avec points » = densité > 0. NoData/NaN comptent comme « sans
      points » (pas de retour sol à cet endroit — sémantique PCSAPS
      ``if(isnull(...), 0, ...)``) : à l'intérieur d'une dalle, l'absence de
      donnée EST l'information (couverture 0 %), le NoData de sortie n'apparaît
      que hors dalles (remplissage gdalwarp/VRT).
    - Normalisation par le nombre réel de cellules de la fenêtre (bords tronqués).
    - Accumulateurs uint8 (compte max = 13) : RAM bornée même sur un raster issu
      d'un gros LAZ fusionné.
    """
    if density.ndim != 2:
        raise ValueError(f"density doit être 2D, reçu shape={density.shape}")

    presence = density > 0  # NaN > 0 et nodata négatif > 0 → déjà False
    if density_nodata is not None:
        presence &= ~np.isclose(density, density_nodata)
    presence = presence.astype(np.uint8)

    h, w = presence.shape
    filled = np.zeros((h, w), dtype=np.uint8)
    total = np.zeros((h, w), dtype=np.uint8)

    for dy, dx in disc_offsets(diameter):
        dst_y = slice(max(0, -dy), h - max(0, dy))
        dst_x = slice(max(0, -dx), w - max(0, dx))
        src_y = slice(max(0, dy), h - max(0, -dy))
        src_x = slice(max(0, dx), w - max(0, -dx))
        filled[dst_y, dst_x] += presence[src_y, src_x]
        total[dst_y, dst_x] += 1

    out = np.full((h, w), COVERAGE_NODATA, dtype=np.uint8)
    ok = total > 0
    # Arrondi entier au plus proche ; uint16 suffit (13*100 + 6 < 65536).
    out[ok] = (
        (filled[ok].astype(np.uint16) * 100 + total[ok] // 2) // total[ok]
    ).astype(np.uint8)
    return out
