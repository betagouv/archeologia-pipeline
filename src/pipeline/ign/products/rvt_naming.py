"""
Utilitaires pour la génération des noms de fichiers RVT avec paramètres.

Les noms de fichiers incluent les paramètres de configuration pour invalider
le cache quand les paramètres changent entre deux exécutions.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple


def _as_int(value: Any, default: int) -> int:
    """Convertit une valeur en int, gère la virgule française."""
    try:
        if isinstance(value, str):
            # Gérer la virgule française (ex: "10,5" -> 10)
            value = value.replace(",", ".")
        return int(float(value))
    except Exception:
        return int(default)


def _as_float(value: Any, default: float) -> float:
    """Convertit une valeur en float, gère la virgule française."""
    try:
        if isinstance(value, str):
            # Gérer la virgule française (ex: "1,7" -> 1.7)
            value = value.replace(",", ".")
        return float(value)
    except Exception:
        return float(default)


def get_rvt_param_suffix(product_name: str, rvt_params: Dict[str, Any]) -> str:
    """
    Génère le suffixe de paramètres pour un produit RVT donné.
    
    Args:
        product_name: Nom du produit (M_HS, SVF, SLO, LD, SLRM, VAT)
        rvt_params: Dictionnaire des paramètres RVT
    
    Returns:
        Suffixe à ajouter au nom de fichier (ex: "_R10_D16_V1_N0" pour SVF)
    """
    rvt_params = rvt_params or {}
    
    if product_name == "M_HS":
        mdh = rvt_params.get("mdh", {})
        num_directions = _as_int(mdh.get("num_directions", 16), 16)
        if num_directions < 2:
            num_directions = 16
        sun_elevation = _as_int(mdh.get("sun_elevation", 35), 35)
        ve_factor = _as_int(mdh.get("ve_factor", 1), 1)
        return f"_D{num_directions}_E{sun_elevation}_V{ve_factor}"
    
    elif product_name == "SVF":
        svf = rvt_params.get("svf", {})
        num_directions = _as_int(svf.get("num_directions", 16), 16)
        if num_directions < 2:
            num_directions = 16
        radius = _as_int(svf.get("radius", 10), 10)
        ve_factor = _as_int(svf.get("ve_factor", 1), 1)
        noise_remove = _as_int(svf.get("noise_remove", 0), 0)
        return f"_R{radius}_D{num_directions}_V{ve_factor}_N{noise_remove}"
    
    elif product_name == "SLO":
        slope = rvt_params.get("slope", {})
        unit = _as_int(slope.get("unit", 0), 0)
        ve_factor = _as_int(slope.get("ve_factor", 1), 1)
        return f"_U{unit}_V{ve_factor}"
    
    elif product_name == "LD":
        ldo = rvt_params.get("ldo", {})
        angular_res = _as_int(ldo.get("angular_res", 15), 15)
        min_radius = _as_int(ldo.get("min_radius", 10), 10)
        max_radius = _as_int(ldo.get("max_radius", 20), 20)
        # Utiliser _as_float pour gérer la virgule française
        observer_h = _as_float(ldo.get("observer_h", 1.7), 1.7)
        # RVT LD requiert OBSERVER_H > 0
        if observer_h <= 0:
            observer_h = 1.7
        ve_factor = _as_int(ldo.get("ve_factor", 1), 1)
        observer_h_str = str(observer_h).replace(".", "p")
        return f"_A{angular_res}_Rmin{min_radius}_Rmax{max_radius}_H{observer_h_str}_V{ve_factor}"
    
    elif product_name == "SLRM":
        slrm = rvt_params.get("slrm", {})
        radius = _as_int(slrm.get("radius", 20), 20)
        ve_factor = _as_int(slrm.get("ve_factor", 1), 1)
        return f"_R{radius}_V{ve_factor}"
    
    elif product_name == "VAT":
        vat = rvt_params.get("vat", {})
        terrain_type = _as_int(vat.get("terrain_type", 0), 0)
        blend_combination = _as_int(vat.get("blend_combination", 0), 0)
        return f"_T{terrain_type}_B{blend_combination}"
    
    # MNT, DENSITE: pas de paramètres
    return ""


def get_rvt_temp_filename(
    product_name: str,
    current_tile_name: str,
    rvt_params: Dict[str, Any],
) -> str:
    """
    Génère le nom de fichier temporaire pour un produit RVT.
    
    Args:
        product_name: Nom du produit (M_HS, SVF, SLO, LD, SLRM, VAT, MNT, DENSITE)
        current_tile_name: Nom de la dalle (ex: "LHD_FXX_0872_6904")
        rvt_params: Dictionnaire des paramètres RVT
    
    Returns:
        Nom de fichier avec paramètres (ex: "LHD_FXX_0872_6904_SVF_R10_D16_V1_N0.tif")
    """
    param_suffix = get_rvt_param_suffix(product_name, rvt_params)
    
    base_names = {
        "MNT": "MNT",
        "DENSITE": "densite",
        "M_HS": "hillshade",
        "SVF": "SVF",
        "SLO": "Slope",
        "LD": "LD",
        "SLRM": "SLRM",
        "VAT": "VAT",
    }
    
    base_name = base_names.get(product_name, product_name)
    return f"{current_tile_name}_{base_name}{param_suffix}.tif"


def get_all_rvt_temp_filenames(
    current_tile_name: str,
    rvt_params: Dict[str, Any],
) -> Dict[str, str]:
    """
    Génère tous les noms de fichiers temporaires RVT pour une dalle.
    
    Returns:
        Dict[product_name, filename]
    """
    products = ["MNT", "DENSITE", "M_HS", "SVF", "SLO", "LD", "SLRM", "VAT"]
    return {
        p: get_rvt_temp_filename(p, current_tile_name, rvt_params)
        for p in products
    }


def get_rvt_source_and_dest_filenames(
    product_name: str,
    current_tile_name: str,
    x: str,
    y: str,
    rvt_params: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Génère les noms de fichiers source (temp) et destination (cropped) pour un produit.
    
    Args:
        product_name: Nom du produit
        current_tile_name: Nom de la dalle
        x, y: Coordonnées extraites du nom de dalle
        rvt_params: Paramètres RVT
    
    Returns:
        Tuple (source_filename, dest_filename)
    """
    source_filename = get_rvt_temp_filename(product_name, current_tile_name, rvt_params)
    
    # Suffixe de paramètres pour invalider le cache si config change
    param_suffix = get_rvt_param_suffix(product_name, rvt_params)
    
    # Noms de destination avec paramètres pour invalider le cache
    if product_name == "MNT":
        dest_filename = f"LHD_FXX_{x}_{y}_MNT_A_0M50_LAMB93_IGN69.tif"
    elif product_name == "DENSITE":
        dest_filename = f"LHD_FXX_{x}_{y}_densite_A_LAMB93.tif"
    elif product_name == "M_HS":
        dest_filename = f"LHD_FXX_{x}_{y}_M-HS{param_suffix}_A_LAMB93.tif"
    else:
        dest_filename = f"LHD_FXX_{x}_{y}_{product_name}{param_suffix}_A_LAMB93.tif"
    
    return source_filename, dest_filename
