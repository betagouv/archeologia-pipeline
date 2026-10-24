"""Catalogue des produits raster + logique du verrou MNG (étape 2), pur/testable.

Le MNT est la base des indices RVT : cocher un indice RVT force MNT, et MNT ne
peut être décoché tant qu'un indice RVT est actif (au lieu d'un verrou
silencieux, l'UI affiche un toast explicatif renvoyé par :func:`toggle`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ProductInfo:
    key: str           # clé config (processing.products.<key>)
    tag: str           # libellé court (badge)
    full_name: str
    description: str
    is_rvt: bool       # True = indice de visualisation (dépend du MNT)
    metier: str = ""   # ce que l'image montre, pour l'utilisateur (« Structures en relief »)


_PRODUCTS: List[ProductInfo] = [
    ProductInfo("MNT", "MNT", "Modèle numérique de terrain", "Altitude du sol", False, metier="Relief nu"),
    ProductInfo("DENSITE", "Densité", "Densité de points", "Points LiDAR / m²", False, metier="Densité de points"),
    ProductInfo("COUVERTURE", "Couverture", "Couverture des points sol",
                "QA : zones où le MNT est interpolé", False, metier="Couverture du sol mesuré"),
    ProductInfo("HS", "HS", "Hillshade", "Ombrage simple (une direction)", True, metier="Ombrage simple"),
    ProductInfo("M_HS", "M-HS", "Multi-Hillshade", "Ombrage multi-directionnel", True, metier="Ombrage du relief"),
    ProductInfo("SVF", "SVF", "Sky-View Factor", "Révèle creux et dépressions", True, metier="Creux & dépressions"),
    ProductInfo("OPNS", "OPNS", "Openness", "Saillies OU creux, au choix", True, metier="Saillies ou creux"),
    ProductInfo("SLO", "SLO", "Slope", "Pente du terrain", True, metier="Pente du terrain"),
    ProductInfo("LD", "LD", "Local Dominance", "Structures en relief", True, metier="Structures en relief"),
    ProductInfo("SLRM", "SLRM", "Simple Local Relief", "Micro-reliefs isolés", True, metier="Micro-reliefs"),
    ProductInfo("VAT", "VAT", "Visualisation archéo", "Combinaison optimisée", True, metier="Vue archéo optimisée"),
    ProductInfo("MSTP", "MSTP", "Multi-Scale Topographic Position",
                "Position topographique multi-échelle (RGB)", True, metier="Bosses & cuvettes"),
    ProductInfo("CVAT", "CVAT", "Combined VAT",
                "VAT combiné (general + flat), terrains variés", True, metier="Vue archéo renforcée"),
    ProductInfo("PRISM", "PRISM", "Prismatic Openness",
                "Ouverture prismatique, micro-relief en couleurs (RGB)", True, metier="Ouverture prismatique"),
    ProductInfo("CRIM", "CRIM", "Color Relief Image Map",
                "Relief coloré par la pente, fond sable (RGB)", True, metier="Relief coloré"),
]
_BY_KEY = {p.key: p for p in _PRODUCTS}


def all_products() -> List[ProductInfo]:
    return list(_PRODUCTS)


def product(key: str) -> ProductInfo:
    return _BY_KEY[key]


def rvt_keys() -> List[str]:
    return [p.key for p in _PRODUCTS if p.is_rvt]


def base_keys() -> List[str]:
    return [p.key for p in _PRODUCTS if not p.is_rvt]


#: Les trois images cochées d'avance à une ouverture vierge (lot 3, 2026-10-24) : le
#: relief nu, les creux (SVF) et les reliefs (LD) — un premier regard utile, sans
#: pastille rouge dès l'ouverture. Une configuration enregistrée garde ses coches.
_RECOMMANDEES: List[str] = ["MNT", "SVF", "LD"]
_QUALITE: List[str] = ["DENSITE", "COUVERTURE"]


def recommended_keys() -> List[str]:
    """Section « Recommandé pour commencer » de l'étape 2 : MNT, SVF, LD."""
    return list(_RECOMMANDEES)


def quality_keys() -> List[str]:
    """Section « Qualité de la donnée » : densité et couverture des points sol."""
    return list(_QUALITE)


def other_rvt_keys() -> List[str]:
    """Section « Autres images du relief » : les RVT hors recommandées, dans l'ordre du catalogue."""
    return [p.key for p in _PRODUCTS if p.is_rvt and p.key not in _RECOMMANDEES]


def default_products() -> Dict[str, bool]:
    """Les trois images recommandées cochées, le reste décoché."""
    return {p.key: p.key in _RECOMMANDEES for p in _PRODUCTS}


# Produits dérivés du nuage de points LiDAR (densité = points/m², couverture =
# zones où le sol est mesuré) : impossibles quand l'entrée est un MNT/RVT déjà
# interpolé. Le MNT (entrée du mode) et les indices RVT (calculés depuis le MNT)
# restent disponibles en mode existing_mnt.
_POINT_CLOUD_PRODUCTS: List[str] = ["DENSITE", "COUVERTURE"]


def products_unavailable_in_mode(mode: str) -> List[str]:
    """Clés de produits incompatibles avec ``mode`` (à forcer décochés en UI).

    Sans nuage de points (``existing_mnt`` / ``existing_rvt``), les produits
    dérivés du nuage sont impossibles. Comme la carte « Modèle de base » est
    seulement *masquée* (pas réinitialisée) au changement de mode, une sélection
    héritée d'un run LAZ y persisterait et serait resérialisée dans la config
    (faux « produit demandé », avertissement « ignoré »). Cette liste pilote la
    purge dans ``IndicesPage.set_mode``.
    """
    if mode in ("existing_mnt", "existing_rvt"):
        return list(_POINT_CLOUD_PRODUCTS)
    return []


def requires_mnt(products: Dict[str, bool]) -> bool:
    """True si un indice RVT est actif (→ MNT requis)."""
    return any(products.get(k) for k in rvt_keys())


def count_selected(products: Dict[str, bool]) -> int:
    return sum(1 for p in _PRODUCTS if products.get(p.key))


def toggle(products: Dict[str, bool], key: str) -> Tuple[Dict[str, bool], Optional[str]]:
    """Bascule un produit. Retourne ``(nouveaux_produits, message_toast | None)``.

    - cocher un indice RVT force ``MNT`` ;
    - décocher ``MNT`` alors qu'un indice RVT est actif est **bloqué** : on
      renvoie l'état inchangé + un message expliquant quels indices l'exigent.
    """
    new = {p.key: bool(products.get(p.key, False)) for p in _PRODUCTS}
    new[key] = not new.get(key, False)

    if key in rvt_keys() and new[key]:
        new["MNT"] = True

    if key == "MNT" and not new["MNT"] and requires_mnt(new):
        active = [product(k).tag for k in rvt_keys() if new.get(k)]
        msg = (
            f"MNT est requis par {', '.join(active)} — décochez d'abord ces "
            "indices pour pouvoir retirer MNT."
        )
        unchanged = {p.key: bool(products.get(p.key, False)) for p in _PRODUCTS}
        return unchanged, msg

    return new, None
