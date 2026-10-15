"""Moteur d'indices intégré : les indices de visualisation calculés dans le plugin avec la bibliothèque
rvt embarquée (``rvt_embarque/`` : copie figée de celle de rvt-qgis 1.0.1, voir son NOTICE), au lieu des
algorithmes Processing de l'extension rvt-qgis.

Règle : chaque produit appelle les MÊMES fonctions de rvt, avec les MÊMES arguments, que l'algorithme
Processing qu'il remplace (``processing_provider/rvt_*.py`` de rvt-qgis 1.0.1). Tout écart est nommé là où il
est fait et prouvé au pixel près (``tests/unit/test_moteur_indices.py``, bancs N2).

Imports de numpy et de rvt différés : ce module s'importe sans eux (tests, préflight).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

#: « processing » : chaque produit qui a un algorithme Processing passe par l'extension rvt-qgis, comme avant
#: la 0.16.0 (diagnostic d'un écart ; l'extension est alors requise). CVAT et CRIM n'en ont pas.
VARIABLE_MOTEUR = "ARCHEO_INDICES_MOTEUR"

#: Produits calculés par le moteur intégré ; les autres passent encore par Processing (rvt-qgis).
PRODUITS_MOTEUR = frozenset({"CVAT", "CRIM", "LD"})

#: Produits sans algorithme Processing : toujours calculés ici, même forcé.
SANS_PROCESSING = frozenset({"CVAT", "CRIM"})


def par_le_moteur(produit: str) -> bool:
    """Vrai si ``produit`` est calculé ici, faux s'il passe par Processing (rvt-qgis)."""
    if produit in SANS_PROCESSING:
        return True
    if os.environ.get(VARIABLE_MOTEUR, "").strip().lower() == "processing":
        return False
    return produit in PRODUITS_MOTEUR


def chemin_reglages(nom: str) -> Path:
    """Fichier de réglages de rvt embarqué (``rvt_embarque/settings/``)."""
    return Path(__file__).resolve().parent / "rvt_embarque" / "settings" / nom


def moteur_disponible() -> str:
    """'' si la bibliothèque rvt embarquée se charge (numpy, scipy, matplotlib, GDAL), sinon la raison."""
    try:
        from .rvt_embarque import blend, default, vis  # noqa: F401
    except Exception as exc:  # noqa: BLE001 — toute raison est à montrer au préflight
        return f"{type(exc).__name__}: {exc}"
    return ""



#: Bornes des paramètres des algorithmes Processing de rvt-qgis 1.0.1 (minValue/maxValue des
#: QgsProcessingParameterNumber, et les enums). Hors de ces bornes Processing refusait le paramètre et la dalle
#: échouait : le moteur refuse de même (``hors_bornes``), en nommant le réglage. Un seul écart : LD.max_radius
#: commence à 1 (Processing acceptait 0 ; rvt rendait alors une image vide, que GDAL refusait d'écrire).
#: Relues dans l'extension voisine par tests/unit/test_moteur_indices.py.
VE = (-1000, 1000)
BORNES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "HS": {"sun_azimuth": (0, 360), "sun_elevation": (0, 90), "ve_factor": VE},
    "M_HS": {"num_directions": (2, 64), "sun_elevation": (0, 90), "ve_factor": VE},
    "SVF": {"radius": (10, 50), "num_directions": (8, 64), "noise_remove": (0, 3), "ve_factor": VE},
    "OPNS": {"radius": (10, 50), "num_directions": (8, 64), "noise_remove": (0, 3), "ve_factor": VE},
    "SLO": {"unit": (0, 2), "ve_factor": VE},
    "LD": {"angular_res": (4, 32), "min_radius": (0, 100), "max_radius": (1, 100), "observer_h": (0.5, 20),
           "ve_factor": VE},
    "SLRM": {"radius": (10, 50), "ve_factor": VE},
    "MSTP": {"lightness": (0.1, 5.0), "ve_factor": VE,
             **{f"{e}_scale_{k}": (1, 10 ** 9) for e in ("local", "meso", "broad") for k in ("min", "max", "step")}},
    "VAT": {"terrain_type": (0, 2), "blend_combination": (0, 2)},
    "PRISM": {"terrain_type": (0, 2)},
}


def hors_bornes(produit: str, valeurs: Dict[str, float]) -> str:
    """'' si chaque valeur est dans les bornes de Processing, sinon le message de refus."""
    for cle, v in valeurs.items():
        b = BORNES.get(produit, {}).get(cle)
        if b is not None and not (b[0] <= v <= b[1]):
            return f"{produit} : {cle}={v} hors des bornes de RVT ({b[0]} à {b[1]})"
    return ""


def _verifier(produit: str, **valeurs: float) -> None:
    raison = hors_bornes(produit, valeurs)
    if raison:
        raise ValueError(raison)


def _rvt():
    from .rvt_embarque import blend, blend_func, default, vis
    return blend, blend_func, default, vis


class Couches:
    """Le MNT d'une dalle, lu une fois pour tous ses produits calculés ici."""

    def __init__(self, chemin: Path, brut, res_x: float, res_y: float, no_data) -> None:
        self.chemin = Path(chemin)
        self._brut = brut
        self.res_x, self.res_y, self.no_data = res_x, res_y, no_data

    @classmethod
    def lire(cls, chemin: Path) -> "Couches":
        _, _, default, _ = _rvt()
        d = default.get_raster_arr(str(chemin))
        return cls(chemin, d["array"], d["resolution"][0], d["resolution"][1], d["no_data"])

    def brut(self):
        """Copie du MNT tel que lu : ce que reçoit un algorithme Processing, qui relit le fichier."""
        return self._brut.copy()


def dominance_locale(dem, min_rad=10, max_rad=20, rad_inc=1, angular_res=15, observer_height=1.7, ve_factor=1,
                     no_data=None):
    """rvt.vis.local_dominance, mêmes opérations dans le même ordre (mêmes types : identique sous numpy 1.x et
    2.x) ; seules changent la manière de décaler (une vue sur le MNT rembourré, |décalage| ≤ max_rad =
    rembourrage, au lieu de deux np.roll) et l'accumulation (np.copyto(where=…) au lieu de np.where + indexation).
    Mesuré ×3,3 à ×3,8 sur trois dalles (prototype run_indices.py:ld_vues)."""
    import numpy as np
    if no_data is not None:
        dem[dem == no_data] = np.nan
    dem = dem.astype(np.float32)
    rembourre = np.pad(array=dem, pad_width=max_rad, mode="edge") * ve_factor
    h, w = dem.shape
    n_dist = int((max_rad - min_rad) / rad_inc + 1)
    distances = np.arange(n_dist * rad_inc, step=rad_inc) + min_rad
    n_ang = int(359 / angular_res + 1)
    angles = np.arange(n_ang * angular_res, step=angular_res)
    norma = np.sum((observer_height / distances) * (2 * distances + rad_inc)) * n_ang
    n_shifts = distances.size * angles.size
    x_t = (np.outer(np.cos(np.deg2rad(angles)), distances)).reshape(n_shifts)
    y_t = (np.outer(np.sin(np.deg2rad(angles)), distances)).reshape(n_shifts)
    distances = (np.outer(np.ones(n_ang), distances)).reshape(n_shifts)
    dist_factor = 2 * distances + rad_inc
    coeur = rembourre[max_rad:max_rad + h, max_rad:max_rad + w]
    sortie = coeur * 0
    haut = coeur + observer_height
    for i in range(n_shifts):
        dy, dx = int(round(y_t[i])), int(round(x_t[i]))
        decale = rembourre[max_rad - dy:max_rad - dy + h, max_rad - dx:max_rad - dx + w]
        nouveau = (sortie + (haut - decale) / distances[i] * dist_factor[i]).astype(np.float32)
        np.copyto(sortie, nouveau, where=haut > decale)
    return sortie / norma


def _ecrire(c: Couches, out: Path, arr, e_type: int) -> None:
    """save_raster de rvt (LZW, géoréférencement du MNT, NoData NaN), comme les algorithmes Processing, dans un
    voisin temporaire basculé une fois complet : jamais de TIF partiel sous le nom final."""
    import numpy as np
    _, _, default, _ = _rvt()
    tmp = out.with_name(out.name + ".tmp")
    default.save_raster(src_raster_path=str(c.chemin), out_raster_path=str(tmp), out_raster_arr=arr,
                        no_data=np.nan, e_type=e_type)
    os.replace(tmp, out)


def _publier(c: Couches, out: Path, arr, huit_bits: bool, visualisation: str) -> None:
    """Sortie d'un algorithme rvt:* : float_to_8bit puis Byte, ou le flottant brut en Float32."""
    _, _, default, _ = _rvt()
    if huit_bits:
        _ecrire(c, out, default.DefaultValues().float_to_8bit(
            float_arr=arr, visualization=getattr(default.RVTVisualization, visualisation)), 1)
    else:
        _ecrire(c, out, arr, 6)


def calculer_ld(c: Couches, out: Path, *, angular_res: int, min_radius: int, max_radius: int, observer_h: float,
                ve: float, huit_bits: bool) -> None:
    """rvt_local_dom.py, par ``dominance_locale``."""
    _verifier("LD", angular_res=angular_res, min_radius=min_radius, max_radius=max_radius, observer_h=observer_h,
              ve_factor=ve)
    arr = dominance_locale(c.brut(), min_rad=int(min_radius), max_rad=int(max_radius), rad_inc=1,
                           angular_res=int(angular_res), observer_height=float(observer_h), ve_factor=float(ve),
                           no_data=c.no_data)
    _publier(c, out, arr, huit_bits, "LOCAL_DOMINANCE")
