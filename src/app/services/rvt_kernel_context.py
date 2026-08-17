"""Contexte spatial requis par les noyaux RVT — diagnostic partagé.

**Le problème.** Chaque visualisation RVT lit un voisinage autour de chaque
pixel. Si le raster fourni au calcul ne s'étend pas d'au moins ce rayon autour
d'un pixel, RVT ne renonce pas : il **fabrique** le voisinage manquant en
repliant le raster sur lui-même (``np.pad(..., mode="symmetric")``, cf.
``rvt.vis.max_elevation_deviation``). La valeur produite est inventée, sans
aucun signal.

Deux manifestations selon la géométrie :

- **Traitement par dalle** (``ign_laz`` / ``local_laz``) : la donnée voisine
  existe, mais seule la marge (``tile_overlap``) est fournie au calcul. Un
  noyau plus large que la marge produit des **coutures visibles** entre dalles.
- **Raster unique** (``existing_mnt``) : il n'y a pas de voisin. Un noyau large
  face à un petit MNT rend le produit majoritairement fabriqué — et comme il
  n'y a qu'un raster, **aucune couture ne le signale**. Échec silencieux.

Ce module est la source unique du diagnostic. La table :data:`KERNEL_PARAMS`
est le miroir de ``rvt.tile._get_rvt_visualization_overlap`` — la réponse de
RVT lui-même à « combien de contexte faut-il par visualisation ». HS / M_HS /
SLO / VAT / CVAT n'y figurent pas : leurs noyaux sont des 3×3 (ou des
compositions de petits noyaux), jamais un problème aux échelles du plugin.

Pur (aucun import QGIS) : testable en standalone, cf.
``tests/unit/test_rvt_kernel_context.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

# Marge maximale que ``preprocess.prepare_merged_tiles`` sait découper chez les
# dalles voisines. Au-delà, le recouvrement demandé est silencieusement rogné.
_MAX_TILE_MARGIN_M = 999

# En dessous de cette fraction de l'emprise disposant d'un voisinage complet, un
# raster unique est signalé. Seuil délibérément bas : sur un raster isolé, une
# bande de bordure approximative est **inévitable** (il n'existe pas de donnée
# au-delà) et l'signaler serait du bruit. Ce qu'on veut attraper, c'est le
# produit majoritairement fabriqué, pas la bordure normale.
MIN_FULL_CONTEXT_RATIO = 0.5


@dataclass(frozen=True)
class KernelParam:
    """Le paramètre qui fixe le rayon de noyau d'un produit RVT."""

    section: str    # clé de section dans ``rvt_params``
    key: str        # clé du rayon dans cette section
    default: int    # défaut RVT, appliqué quand la clé est absente
    ui_label: str   # libellé du champ à l'étape 2 → message actionnable


# Miroir de rvt.tile._get_rvt_visualization_overlap. ⚠ La section de LD est
# « ldo », pas « ld » (cf. indices.py:192 et step_2_indices.py:462).
KERNEL_PARAMS: Dict[str, KernelParam] = {
    "SVF": KernelParam("svf", "radius", 10, "Rayon"),
    "LD": KernelParam("ldo", "max_radius", 20, "Rayon max"),
    "SLRM": KernelParam("slrm", "radius", 20, "Rayon"),
    "MSTP": KernelParam("mstp", "broad_scale_max", 2023, "Échelle large — rayon max"),
}

# Bandes d'échelle MSTP : (préfixe de clé, libellé, min, max, pas) par défaut.
_MSTP_BANDS: Tuple[Tuple[str, str, int, int, int], ...] = (
    ("local_scale", "Échelle locale", 3, 21, 2),
    ("meso_scale", "Échelle méso", 23, 203, 18),
    ("broad_scale", "Échelle large", 223, 2023, 180),
)


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _section(rvt_params: Optional[Mapping], name: str) -> Mapping:
    section = (rvt_params or {}).get(name, {})
    return section if isinstance(section, Mapping) else {}


def kernel_radius_px(product: str, rvt_params: Optional[Mapping]) -> Optional[int]:
    """Rayon de noyau de ``product``, en pixels — ``None`` s'il n'en a pas de réglable."""
    param = KERNEL_PARAMS.get(product)
    if param is None:
        return None
    return _as_int(_section(rvt_params, param.section).get(param.key, param.default), param.default)


def tile_margin_px(tile_overlap_percent: float, mnt_resolution: float) -> int:
    """Contexte disponible autour de chaque dalle, en pixels.

    Reproduit exactement le calcul de ``preprocess.prepare_merged_tiles`` (marge
    en pourcentage d'une dalle IGN de 1 km, plafonnée à 999 m) puis convertit en
    pixels. C'est la donnée voisine réellement fournie au calcul RVT — à
    condition que les dalles voisines existent : une dalle isolée (commande
    unique, bord d'emprise) n'en reçoit aucune.
    """
    try:
        resolution = float(mnt_resolution)
    except (TypeError, ValueError):
        return 0
    if resolution <= 0:
        return 0
    margin_m = max(0, min(_MAX_TILE_MARGIN_M, int(round(1000.0 * float(tile_overlap_percent) / 100.0))))
    return int(margin_m / resolution)


def full_context_ratio(width_px: int, height_px: int, radius_px: int) -> float:
    """Fraction de l'emprise dont le voisinage de rayon ``radius_px`` tient dedans.

    Les pixels situés à moins de ``radius_px`` d'un bord voient leur voisinage
    complété par symétrie ; seul le rectangle central en est exempt.
    """
    if width_px <= 0 or height_px <= 0:
        return 0.0
    inner_w = max(0, width_px - 2 * radius_px)
    inner_h = max(0, height_px - 2 * radius_px)
    return (inner_w * inner_h) / float(width_px * height_px)


def max_radius_for_ratio(width_px: int, height_px: int, min_ratio: float) -> int:
    """Plus grand rayon dont le voisinage reste complet sur ``min_ratio`` de l'emprise.

    Recherche dichotomique sur :func:`full_context_ratio`, qui est décroissante
    en ``radius_px`` — moins casse-gueule que la forme fermée du trinôme, et
    elle réutilise la définition au lieu de la dupliquer.
    """
    low, high = 0, max(width_px, height_px)
    while low < high:
        mid = (low + high + 1) // 2
        if full_context_ratio(width_px, height_px, mid) >= min_ratio:
            low = mid
        else:
            high = mid - 1
    return low


def _requested(products: Optional[Mapping], product: str) -> bool:
    return bool((products or {}).get(product, False))


def tiled_context_warnings(
    products: Optional[Mapping],
    rvt_params: Optional[Mapping],
    *,
    tile_overlap_percent: float,
    mnt_resolution: float,
) -> List[str]:
    """Produits dont le noyau dépasse la marge entre dalles (→ coutures visibles).

    Critère strict : la donnée voisine **existe**, ne pas la fournir au calcul
    est un défaut évitable, pas une fatalité de bordure.
    """
    margin = tile_margin_px(tile_overlap_percent, mnt_resolution)
    messages: List[str] = []
    for product, param in KERNEL_PARAMS.items():
        if not _requested(products, product):
            continue
        radius = kernel_radius_px(product, rvt_params)
        if radius is None or radius <= margin:
            continue
        messages.append(
            f"{product} : le noyau atteint {radius} px mais la marge entre dalles "
            f"n'en fournit que {margin} px — au-delà, RVT reconstruit le voisinage "
            f"par symétrie et les dalles ne se raccordent plus (coutures visibles). "
            f"Réduisez « {param.ui_label} (px) » à ≤ {margin} px, ou augmentez la "
            f"marge tuiles."
        )
    return messages


def raster_context_warnings(
    products: Optional[Mapping],
    rvt_params: Optional[Mapping],
    *,
    width_px: int,
    height_px: int,
) -> List[str]:
    """Produits dont le noyau rend un raster isolé majoritairement fabriqué.

    Critère souple (:data:`MIN_FULL_CONTEXT_RATIO`) : sur un raster sans voisin,
    une bordure approximative est inévitable. On ne signale que la dégénérescence.
    """
    messages: List[str] = []
    for product, param in KERNEL_PARAMS.items():
        if not _requested(products, product):
            continue
        radius = kernel_radius_px(product, rvt_params)
        if radius is None:
            continue
        ratio = full_context_ratio(width_px, height_px, radius)
        if ratio >= MIN_FULL_CONTEXT_RATIO:
            continue
        target = max_radius_for_ratio(width_px, height_px, MIN_FULL_CONTEXT_RATIO)
        messages.append(
            f"{product} : le noyau atteint {radius} px sur un raster de "
            f"{width_px}×{height_px} px — {round(ratio * 100)} % de l'emprise "
            f"seulement aura un voisinage complet, le reste est reconstruit par "
            f"symétrie. Réduisez « {param.ui_label} (px) » à ≤ {target} px."
        )
    return messages


def mstp_scale_errors(
    products: Optional[Mapping], rvt_params: Optional[Mapping]
) -> List[str]:
    """Combinaisons min/max/pas que ``rvt.vis.mstp`` refuse (il lève une exception).

    Sans ce garde-fou, la faute n'apparaît qu'au calcul de la première dalle —
    après le téléchargement et le MNT. Piège classique en réduisant le rayon max
    sans toucher au pas : ``max - min < pas`` est rejeté par RVT.
    """
    if not _requested(products, "MSTP"):
        return []
    section = _section(rvt_params, "mstp")
    messages: List[str] = []
    for prefix, label, d_min, d_max, d_step in _MSTP_BANDS:
        lo = _as_int(section.get(f"{prefix}_min", d_min), d_min)
        hi = _as_int(section.get(f"{prefix}_max", d_max), d_max)
        step = _as_int(section.get(f"{prefix}_step", d_step), d_step)
        if lo > hi:
            messages.append(
                f"MSTP / « {label} » : rayon min ({lo} px) supérieur au rayon max "
                f"({hi} px) — RVT refuse ce réglage."
            )
            continue
        if hi - lo < step:
            messages.append(
                f"MSTP / « {label} » : l'écart entre rayon min ({lo} px) et rayon max "
                f"({hi} px) est plus petit que le pas ({step} px) — RVT refuse ce "
                f"réglage. Réduisez le pas à ≤ {hi - lo} px."
            )
    return messages
