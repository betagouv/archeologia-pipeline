"""
Utilitaires pour la génération des noms de fichiers RVT avec paramètres.

Les noms de fichiers incluent les paramètres de configuration pour invalider
le cache quand les paramètres changent entre deux exécutions.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

#: Ordre canonique des produits raster — SOURCE UNIQUE pour les boucles
#: crop/copy/nommage (anti-bug « produit oublié dans une liste », cf.
#: avertissement run_context.py). Miroir app-side : ``run_context._ALL_PRODUCTS``
#: (qui ne peut pas l'importer : src/app doit rester importable sans tirer ce
#: package — un test verrouille la synchronisation des deux tuples).
PRODUCT_ORDER: Tuple[str, ...] = (
    "MNT", "DENSITE", "COUVERTURE", "HS", "M_HS", "SVF", "OPNS", "SLO", "LD", "SLRM",
    "VAT", "MSTP", "CVAT", "PRISM", "CRIM",
)


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


#: Bornes DURES des deux algorithmes RVT qui balayent l'horizon. ``rvt:rvt_svf``
#: et ``rvt:rvt_opns`` déclarent exactement les mêmes ``QgsProcessingParameterNumber``
#: (c'est le même noyau, ``rvt.vis.sky_view_factor``) : hors de ces intervalles,
#: Processing refuse le paramètre et la dalle échoue. Le RAYON n'est plus écrêté depuis qu'il se règle en
#: mètres (2026-10-31) : 5 m font 5 px à 1 m, le moteur intégré les calcule ; seul le secours Processing est
#: borné à 10-50 px, refusé avant le lancement (rvt_kernel_context.bornes_errors), jamais écrêté en silence.
HORIZON_RADIUS_RANGE: Tuple[int, int] = (1, 10 ** 6)
HORIZON_DIRECTIONS_RANGE: Tuple[int, int] = (8, 64)
HORIZON_NOISE_RANGE: Tuple[int, int] = (0, 3)  # enum : aucune / faible / moyenne / forte


def _clamp(value: int, bornes: Tuple[int, int]) -> int:
    lo, hi = bornes
    return max(lo, min(hi, value))


def _horizon_settings(rvt_params: Dict[str, Any], section: str) -> Dict[str, int]:
    """Réglages EFFECTIFS d'un produit à balayage d'horizon, bornes appliquées.

    SOURCE UNIQUE : le suffixe de dossier (ci-dessous) et l'appel à l'algorithme
    (``indices.py``) lisent tous deux cette fonction. Sans elle, un rayon de
    200 px serait ramené à 50 au calcul mais écrit ``_R200`` au nom du dossier :
    le nom mentirait sur l'image qu'il contient.

    ``save_as_8bit`` n'y figure pas : il ne change pas le nom du dossier et se
    lit côté appelant.
    """
    cfg = (rvt_params or {}).get(section) or {}
    if not isinstance(cfg, dict):
        cfg = {}
    return {
        "radius": _clamp(_as_int(cfg.get("radius", 10), 10), HORIZON_RADIUS_RANGE),
        "num_directions": _clamp(
            _as_int(cfg.get("num_directions", 16), 16), HORIZON_DIRECTIONS_RANGE
        ),
        "noise_remove": _clamp(
            _as_int(cfg.get("noise_remove", 0), 0), HORIZON_NOISE_RANGE
        ),
        "ve_factor": _as_int(cfg.get("ve_factor", 1), 1),
    }


#: Unité de pente envoyée pour « Pourcentage » : le code « percent » de l'enum UNIT de rvt:rvt_slope
#: (degree = 0, radian = 1, percent = 2).
PENTE_POURCENT = 2


def slope_unit(rvt_params: Dict[str, Any]) -> int:
    """Unité EFFECTIVE de la pente : 0 (degrés) ou 2 (pourcentage).

    Jusqu'à la 0.15.0, l'option « Pourcentage » envoyait 1, le code des RADIANS : une configuration qui porte 1
    est relue comme un pourcentage, ce que l'utilisateur avait demandé. SOURCE UNIQUE du nom de dossier et du
    calcul (``indices.py``), comme ``svf_settings``.
    """
    cfg = (rvt_params or {}).get("slope") or {}
    unite = _as_int(cfg.get("unit", 0), 0) if isinstance(cfg, dict) else 0
    return PENTE_POURCENT if unite in (1, PENTE_POURCENT) else 0


#: Défauts du MSTP : ceux de l'interface et de ConfigManager (échelle large 100/400/60, pas 223/2023/180 comme
#: rvt : un rayon de 2 023 px dépasse de loin la marge de tuilage). Repli unique quand une clé manque, pour le nom
#: du dossier ET le calcul ; app.services.rvt_kernel_context en garde un miroir (test de synchronisation).
MSTP_DEFAUTS: Dict[str, float] = {
    "local_scale_min": 3, "local_scale_max": 21, "local_scale_step": 2,
    "meso_scale_min": 23, "meso_scale_max": 203, "meso_scale_step": 18,
    "broad_scale_min": 100, "broad_scale_max": 400, "broad_scale_step": 60,
    "lightness": 0.9, "ve_factor": 1,
}
_PAS_MSTP = ("local_scale_step", "meso_scale_step", "broad_scale_step")


def mstp_settings(rvt_params: Dict[str, Any]) -> Dict[str, float]:
    """Réglages EFFECTIFS du MSTP (section ``mstp``), défauts de ``MSTP_DEFAUTS`` pour une clé absente."""
    cfg = (rvt_params or {}).get("mstp") or {}
    if not isinstance(cfg, dict):
        cfg = {}
    m = {k: (_as_float(cfg.get(k, d), d) if k == "lightness" else _as_int(cfg.get(k, d), int(d)))
         for k, d in MSTP_DEFAUTS.items()}
    # L'exagération n'est plus proposée (sans effet en arithmétique exacte, mais du bruit en flottant) : celle
    # d'une ancienne configuration est ignorée, sans quoi l'image changerait sous le même nom de dossier.
    m["ve_factor"] = 1
    return m


def svf_settings(rvt_params: Dict[str, Any]) -> Dict[str, int]:
    """Réglages effectifs du Sky-View Factor (section ``svf``)."""
    return _horizon_settings(rvt_params, "svf")


def opns_settings(rvt_params: Dict[str, Any]) -> Dict[str, int]:
    """Réglages effectifs de l'openness (section ``opns``) + son type.

    Mêmes bornes que le SVF : c'est le même balayage d'horizon, au signe du
    modèle d'altitude près.
    """
    cfg = (rvt_params or {}).get("opns") or {}
    if not isinstance(cfg, dict):
        cfg = {}
    reglages = _horizon_settings(rvt_params, "opns")
    # Un seul réglage de type, comme rvt-qgis : 0 = positive, 1 = négative.
    reglages["opns_type"] = 1 if _as_int(cfg.get("opns_type", 0), 0) == 1 else 0
    return reglages


#: Section de ``rvt_params`` de chaque indice (celle qui porte son ``save_as_8bit``).
SECTION_DU_PRODUIT: Dict[str, str] = {
    "HS": "hs", "M_HS": "mdh", "SVF": "svf", "OPNS": "opns", "SLO": "slope", "LD": "ldo", "SLRM": "slrm",
    "VAT": "vat", "MSTP": "mstp", "CVAT": "cvat", "PRISM": "prism", "CRIM": "crim",
}

#: Marque d'une sortie flottante (« Sauver en 8 bits » décoché). Absente en 8 bits, le défaut : les dossiers 8 bits
#: gardent leur nom. Sans elle, 8 bits et flottant partageaient un nom de fichier, et une relance resservait l'un
#: pour l'autre (le cache des intermédiaires ne compare que les dates).
SUFFIXE_FLOTTANT = "_F32"


def sortie_8bits(product_name: str, rvt_params: Dict[str, Any]) -> bool:
    """« Sauver en 8 bits » de l'indice (vrai par défaut)."""
    cfg = (rvt_params or {}).get(SECTION_DU_PRODUIT.get(product_name, ""), {})
    return bool(cfg.get("save_as_8bit", True)) if isinstance(cfg, dict) else True


def get_rvt_param_suffix(product_name: str, rvt_params: Dict[str, Any]) -> str:
    """Suffixe de paramètres d'un indice, ``SUFFIXE_FLOTTANT`` compris quand la sortie est flottante."""
    suffixe = _suffixe_reglages(product_name, rvt_params)
    if product_name in SECTION_DU_PRODUIT and not sortie_8bits(product_name, rvt_params):
        suffixe += SUFFIXE_FLOTTANT
    return suffixe


def _suffixe_reglages(product_name: str, rvt_params: Dict[str, Any]) -> str:
    """
    Génère le suffixe de paramètres pour un produit RVT donné.
    
    Args:
        product_name: Nom du produit (M_HS, SVF, SLO, LD, SLRM, VAT)
        rvt_params: Dictionnaire des paramètres RVT
    
    Returns:
        Suffixe à ajouter au nom de fichier (ex: "_R10_D16_V1_N0" pour SVF)
    """
    rvt_params = rvt_params or {}

    if product_name == "HS":
        hs = rvt_params.get("hs", {})
        sun_azimuth = _as_int(hs.get("sun_azimuth", 315), 315)
        sun_elevation = _as_int(hs.get("sun_elevation", 35), 35)
        ve_factor = _as_int(hs.get("ve_factor", 1), 1)
        return f"_Az{sun_azimuth}_E{sun_elevation}_V{ve_factor}"

    elif product_name == "M_HS":
        mdh = rvt_params.get("mdh", {})
        num_directions = _as_int(mdh.get("num_directions", 16), 16)
        if num_directions < 2:
            num_directions = 16
        # En 8 bits l'image compose trois éclairages fixes : le nombre de directions n'y change rien, le nom
        # ne le porte donc plus (constat 12 de l'audit ; il reste au nom de la sortie flottante, où il compte).
        if str(mdh.get("save_as_8bit", True)).strip().lower() not in ("false", "0", "non", "no"):
            num_directions = 16
        sun_elevation = _as_int(mdh.get("sun_elevation", 35), 35)
        ve_factor = _as_int(mdh.get("ve_factor", 1), 1)
        return f"_D{num_directions}_E{sun_elevation}_V{ve_factor}"
    
    elif product_name == "SVF":
        s = svf_settings(rvt_params)
        return (
            f"_R{s['radius']}_D{s['num_directions']}"
            f"_V{s['ve_factor']}_N{s['noise_remove']}"
        )
    
    elif product_name == "OPNS":
        o = opns_settings(rvt_params)
        # Le TYPE en tête : positive et négative sont deux images sans
        # rapport (convexités vs concavités), jamais dans le même dossier.
        type_token = "Neg" if o["opns_type"] == 1 else "Pos"
        return (
            f"_{type_token}_R{o['radius']}_D{o['num_directions']}"
            f"_V{o['ve_factor']}_N{o['noise_remove']}"
        )

    elif product_name == "SLO":
        slope = rvt_params.get("slope", {})
        ve_factor = _as_int(slope.get("ve_factor", 1), 1)
        return f"_U{slope_unit(rvt_params)}_V{ve_factor}"
    
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

    elif product_name == "MSTP":
        m = mstp_settings(rvt_params)
        # Pas de _ dans le token lightness (cohérent avec observer_h du LD).
        lightness_str = str(m["lightness"]).replace(".", "p")
        suffixe = (f"_L{m['local_scale_min']}-{m['local_scale_max']}_M{m['meso_scale_min']}-{m['meso_scale_max']}"
                   f"_B{m['broad_scale_min']}-{m['broad_scale_max']}_Li{lightness_str}")
        # Les pas changent l'image : ils vont au nom dès qu'ils quittent les défauts (aux défauts, le nom d'avant).
        if any(m[k] != MSTP_DEFAUTS[k] for k in _PAS_MSTP):
            suffixe += "_P{}-{}-{}".format(*(m[k] for k in _PAS_MSTP))
        return suffixe

    elif product_name == "PRISM":
        # Combinaison 1 du blender RVT. Comme pour le VAT, le préréglage de
        # terrain ne change pas la recette mais tous ses réglages sous-jacents :
        # il doit donc figurer au nom du dossier.
        prism = rvt_params.get("prism", {})
        terrain_type = _as_int(prism.get("terrain_type", 0), 0)
        return f"_T{terrain_type}"

    elif product_name == "CRIM":
        crim = rvt_params.get("crim", {})
        # La colormap est le paramètre visible du produit : deux colormaps
        # donnent deux images sans rapport, elles ne peuvent pas partager un
        # dossier. Les bornes de coupe suivent la même règle.
        colormap = str(crim.get("colormap", "OrRd")).strip() or "OrRd"
        colormap = "".join(c for c in colormap if c.isalnum() or c == "_")
        cut_min = _as_float(crim.get("min_colormap_cut", 0.0), 0.0)
        cut_max = _as_float(crim.get("max_colormap_cut", 1.0), 1.0)
        bornes = f"{cut_min}-{cut_max}".replace(".", "p")
        return f"_C{colormap}_Cut{bornes}"

    # MNT, DENSITE, COUVERTURE, CVAT (composition figée): pas de suffixe.
    return ""


def get_rvt_folder_name(product_name: str, rvt_params: Dict[str, Any]) -> str:
    """Nom du dossier d'indice = code produit + suffixe de paramètres RVT.

    Permet à deux exécutions de paramètres différents d'écrire dans des dossiers
    distincts (``indices/<CODE><suffixe>/``) plutôt que de s'écraser.

    Ex: ``"SVF"`` -> ``"SVF_R10_D16_V1_N0"``,
        ``"LD"`` -> ``"LD_A15_Rmin10_Rmax20_H1p7_V1"``.
    MNT / DENSITE (suffixe vide) renvoient le code brut (``"MNT"``, ``"DENSITE"``).

    NB : ``get_rvt_param_suffix`` applique les valeurs *par défaut* quand un
    paramètre est absent (dict vide → suffixe par défaut, pas ``""``). Il faut
    donc passer le *même* ``rvt_params`` à la création et à la consommation pour
    viser le même dossier.
    """
    return f"{product_name}{get_rvt_param_suffix(product_name, rvt_params)}"


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
        "COUVERTURE": "couverture",
        "HS": "HS",
        "M_HS": "hillshade",
        "SVF": "SVF",
        "OPNS": "OPNS",
        "SLO": "Slope",
        "LD": "LD",
        "SLRM": "SLRM",
        "VAT": "VAT",
        "MSTP": "MSTP",
        "CVAT": "CVAT",
        "PRISM": "PRISM",
        "CRIM": "CRIM",
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
    return {
        p: get_rvt_temp_filename(p, current_tile_name, rvt_params)
        for p in PRODUCT_ORDER
    }


def jeton_resolution(resolution_m: float) -> str:
    """Résolution du MNT au format du nom IGN : 0,5 m → ``0M50``, 1 m → ``1M00``, 0,25 m → ``0M25``."""
    cm = int(round(float(resolution_m) * 100))
    return f"{cm // 100}M{cm % 100:02d}"


def est_lambert93(crs: Any) -> bool:
    """Vrai pour EPSG:2154 et pour un CRS non dit (modes LiDAR : Lambert-93 par construction)."""
    return not crs or str(crs).strip().upper() == "EPSG:2154"


def prefixe_dalle(x: str, y: str, crs: Any = "EPSG:2154") -> str:
    """Préfixe des noms de dalle : ``LHD_FXX_{x}_{y}`` (nommage du LiDAR HD IGN) en EPSG:2154, nom d'avant à
    l'octet près ; sinon ``EPSG_<code>_{x}_{y}`` (``CRS_PROJETE_{x}_{y}`` sans code EPSG). Toujours DEUX jetons :
    x et y restent ``parts[2]``/``parts[3]`` pour tous les lecteurs (``coords.extract_xy_*``, halo, conversion)."""
    if est_lambert93(crs):
        return f"LHD_FXX_{x}_{y}"
    code = str(crs).strip().upper()
    if code.startswith("EPSG:") and code[5:].isdigit():
        return f"EPSG_{code[5:]}_{x}_{y}"
    return f"CRS_PROJETE_{x}_{y}"


def jeton_crs(crs: Any, *, mnt: bool = False) -> str:
    """``_LAMB93`` (``_LAMB93_IGN69`` pour le MNT) en EPSG:2154 ; rien sinon : le CRS est déjà au préfixe."""
    if not est_lambert93(crs):
        return ""
    return "_LAMB93_IGN69" if mnt else "_LAMB93"


def get_rvt_source_and_dest_filenames(
    product_name: str,
    current_tile_name: str,
    x: str,
    y: str,
    rvt_params: Dict[str, Any],
    *,
    name_suffix: str = "",
    mnt_resolution: float = 0.5,
    crs: Any = "EPSG:2154",
) -> Tuple[str, str]:
    """
    Génère les noms de fichiers source (temp) et destination (cropped) pour un produit.

    Args:
        product_name: Nom du produit
        current_tile_name: Nom de la dalle
        x, y: Coordonnées extraites du nom de dalle (cosmétiques)
        rvt_params: Paramètres RVT
        mnt_resolution: Résolution du MNT (m), écrite dans le nom du MNT publié.
        name_suffix: Suffixe d'unicité inséré avant ``.tif`` dans le nom de destination
            (ex. ``"_<uid>"``). Le placement venant des métadonnées, ``x``/``y`` ne
            servent qu'au préfixe cosmétique ``LHD_FXX_{x}_{y}`` ; pour des dalles
            sub-km partageant la même cellule km, ce suffixe évite l'écrasement des
            produits (le nom de destination est bâti depuis ``x``/``y`` seuls).
            ``""`` (défaut) → nom strictement identique au comportement historique.
        crs: CRS du run. EPSG:2154 (défaut) → noms historiques ``LHD_FXX_…_LAMB93`` ; autre CRS →
            ``EPSG_<code>_…`` sans Lambert-93 ni IGN69 (:func:`prefixe_dalle`, :func:`jeton_crs`).

    Returns:
        Tuple (source_filename, dest_filename)
    """
    source_filename = get_rvt_temp_filename(product_name, current_tile_name, rvt_params)

    # Suffixe de paramètres pour invalider le cache si config change
    param_suffix = get_rvt_param_suffix(product_name, rvt_params)

    # Noms de destination avec paramètres pour invalider le cache
    pre, sc = prefixe_dalle(x, y, crs), jeton_crs(crs)
    if product_name == "MNT":
        # Résolution RÉELLE au nom (figée à 0M50 jusqu'ici, même pour un MNT à 1 m) ; à 0,5 m, le nom d'avant.
        dest_filename = f"{pre}_MNT_A_{jeton_resolution(mnt_resolution)}{jeton_crs(crs, mnt=True)}{name_suffix}.tif"
    elif product_name == "DENSITE":
        dest_filename = f"{pre}_densite_A{sc}{name_suffix}.tif"
    elif product_name == "COUVERTURE":
        dest_filename = f"{pre}_couverture_A{sc}{name_suffix}.tif"
    elif product_name == "M_HS":
        dest_filename = f"{pre}_M-HS{param_suffix}_A{sc}{name_suffix}.tif"
    else:
        dest_filename = f"{pre}_{product_name}{param_suffix}_A{sc}{name_suffix}.tif"

    return source_filename, dest_filename
