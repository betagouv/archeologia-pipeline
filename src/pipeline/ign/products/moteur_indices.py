"""Moteur d'indices intégré : les indices de visualisation calculés dans le plugin avec la bibliothèque
rvt embarquée (``rvt_embarque/`` : copie figée de celle de rvt-qgis 1.0.1, voir son NOTICE), au lieu des
algorithmes Processing de l'extension rvt-qgis.

Règle : chaque produit appelle les MÊMES fonctions de rvt, avec les MÊMES arguments, que l'algorithme
Processing qu'il remplace (``processing_provider/rvt_*.py`` de rvt-qgis 1.0.1). Tout écart est nommé là où il
est fait et prouvé au pixel près (``tests/unit/test_moteur_indices.py``, bancs N2). Par dalle, ``Couches`` lit le
MNT une fois et partage une couche entre produits quand rvt l'aurait calculée à l'identique pour chacun : la clé
de partage est faite des paramètres que la fonction de rvt reçoit réellement, jamais du seul réglage de
l'interface.

Imports de numpy et de rvt différés : ce module s'importe sans eux (tests, préflight).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

#: « processing » : chaque produit qui a un algorithme Processing passe par l'extension rvt-qgis, comme avant
#: la 0.16.0 (diagnostic d'un écart ; l'extension est alors requise). CVAT et CRIM n'en ont pas.
VARIABLE_MOTEUR = "ARCHEO_INDICES_MOTEUR"

#: Produits calculés par le moteur intégré ; les autres passent encore par Processing (rvt-qgis).
PRODUITS_MOTEUR = frozenset({"CVAT", "CRIM", "LD", "HS", "SVF", "OPNS", "SLO", "SLRM"})

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


#: Au-delà, chaque produit recalcule ses couches : la mémoire d'avant sur les grands rasters (RF2).
# ponytail: plafond fixe = dalle LiDAR standard (1 km + 20 % de marge à 0,5 m) ; libérer chaque couche après son
# dernier consommateur si les grands rasters doivent aussi partager.
PARTAGE_MAX_PX = 2801 * 2801

#: Unités de pente de rvt:rvt_slope, dans l'ordre de son enum UNIT (l'interface envoie 0 ou 1).
UNITES_PENTE = ("degree", "radian", "percent")


class Couches:
    """Le MNT d'une dalle, lu une fois, et les couches de calcul partagées entre ses produits.

    Chaque couche est indexée par les paramètres que la fonction de rvt reçoit réellement : deux produits ne la
    partagent que si rvt l'aurait calculée à l'identique pour chacun. Toute couche est rendue en COPIE (les
    normalisations de rvt écrivent dans le tableau reçu).
    """

    def __init__(self, chemin: Path, brut, res_x: float, res_y: float, no_data) -> None:
        self.chemin = Path(chemin)
        self._brut = brut
        self.res_x, self.res_y, self.no_data = res_x, res_y, no_data
        self.taille = (brut.shape[-1], brut.shape[-2])
        self.partage = brut.size <= PARTAGE_MAX_PX
        self._memo: Dict[tuple, Any] = {}

    @classmethod
    def lire(cls, chemin: Path) -> "Couches":
        _, _, default, _ = _rvt()
        d = default.get_raster_arr(str(chemin))
        return cls(chemin, d["array"], d["resolution"][0], d["resolution"][1], d["no_data"])

    def brut(self):
        """Copie du MNT tel que lu : ce que reçoit un algorithme Processing, qui relit le fichier."""
        return self._brut.copy()

    def _une_fois(self, cle: tuple, calcul: Callable[[], Any]):
        if not self.partage:
            return calcul()
        if cle not in self._memo:
            self._memo[cle] = calcul()
        v = self._memo[cle]
        return {k: a.copy() for k, a in v.items()} if isinstance(v, dict) else v.copy()

    def ombrage(self, az: float, el: float, ve: float, rx: float, ry: float):
        _, _, _, vis = _rvt()
        return self._une_fois(("ombrage", az, el, ve, rx, ry), lambda: vis.hillshade(
            dem=self.brut(), resolution_x=rx, resolution_y=ry, sun_azimuth=az, sun_elevation=el,
            ve_factor=ve, no_data=self.no_data))

    def pente(self, unite: str, ve: float, rx: float, ry: float):
        _, _, _, vis = _rvt()
        return self._une_fois(("pente", unite, ve, rx, ry), lambda: vis.slope_aspect(
            dem=self.brut(), resolution_x=rx, resolution_y=ry, output_units=unite, ve_factor=ve,
            no_data=self.no_data)["slope"])

    def horizon(self, rayon: int, directions: int, bruit: int, ve: float, res: float) -> Dict[str, Any]:
        """SVF et openness positive d'un même balayage : rvt.vis.sky_view_factor les tire des mêmes pentes
        maximales, chaque sortie est celle d'un appel qui ne calcule qu'elle."""
        _, _, _, vis = _rvt()
        return self._une_fois(("horizon", rayon, directions, bruit, ve, res), lambda: vis.sky_view_factor(
            dem=self.brut(), resolution=res, compute_svf=True, compute_asvf=False, compute_opns=True,
            svf_n_dir=directions, svf_r_max=rayon, svf_noise=bruit, ve_factor=ve, no_data=self.no_data))

    def ouverture_negative(self, rayon: int, directions: int, bruit: int, ve: float, res: float, *,
                           nan_avant: bool):
        """Openness négative = openness de −MNT. ``nan_avant`` : le sans-donnée est-il déjà NaN quand rvt négative
        le MNT ? Non pour rvt:rvt_opns (−no_data n'est plus reconnu, le sans-donnée compte comme un relief :
        reproduit tel quel) ; oui dans CRIM et dans un mélange où une fonction « mutante » est passée avant."""
        _, _, _, vis = _rvt()

        def calcul():
            import numpy as np
            dem = self.brut()
            if nan_avant and self.no_data is not None:
                dem = dem.astype(np.float32)
                dem[dem == self.no_data] = np.nan
            return vis.sky_view_factor(
                dem=dem * -1, resolution=res, compute_svf=False, compute_asvf=False, compute_opns=True,
                svf_n_dir=directions, svf_r_max=rayon, svf_noise=bruit, ve_factor=ve,
                no_data=self.no_data)["opns"]
        return self._une_fois(("ouverture_negative", rayon, directions, bruit, ve, res, nan_avant), calcul)


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


def calculer_hs(c: Couches, out: Path, *, azimut: float, elevation: float, ve: float, huit_bits: bool) -> None:
    """rvt_hillshade.py."""
    _verifier("HS", sun_azimuth=azimut, sun_elevation=elevation, ve_factor=ve)
    _publier(c, out, c.ombrage(float(azimut), float(elevation), float(ve), c.res_x, c.res_y), huit_bits,
             "HILLSHADE")


def calculer_svf(c: Couches, out: Path, *, rayon: int, directions: int, bruit: int, ve: float,
                 huit_bits: bool) -> None:
    """rvt_svf.py."""
    _verifier("SVF", radius=rayon, num_directions=directions, noise_remove=bruit, ve_factor=ve)
    h = c.horizon(int(rayon), int(directions), int(bruit), float(ve), c.res_x)
    _publier(c, out, h["svf"], huit_bits, "SKY_VIEW_FACTOR")


def calculer_opns(c: Couches, out: Path, *, negative: bool, rayon: int, directions: int, bruit: int, ve: float,
                  huit_bits: bool) -> None:
    """rvt_opns.py : type négatif = openness de −MNT, sans-donnée non remplacé avant la négation (sans partage)."""
    _verifier("OPNS", radius=rayon, num_directions=directions, noise_remove=bruit, ve_factor=ve)
    if negative:
        arr = c.ouverture_negative(int(rayon), int(directions), int(bruit), float(ve), c.res_x, nan_avant=False)
    else:
        arr = c.horizon(int(rayon), int(directions), int(bruit), float(ve), c.res_x)["opns"]
    _publier(c, out, arr, huit_bits, "NEGATIVE_OPENNESS" if negative else "POSITIVE_OPENNESS")


def calculer_slo(c: Couches, out: Path, *, unite: int, ve: float, huit_bits: bool) -> None:
    """rvt_slope.py (``unite`` = indice de l'enum de l'algorithme, cf. UNITES_PENTE)."""
    _verifier("SLO", unit=unite, ve_factor=ve)
    _publier(c, out, c.pente(UNITES_PENTE[int(unite)], float(ve), c.res_x, c.res_y), huit_bits, "SLOPE")


def calculer_slrm(c: Couches, out: Path, *, rayon: int, ve: float, huit_bits: bool) -> None:
    """rvt_slrm.py."""
    _verifier("SLRM", radius=rayon, ve_factor=ve)
    _, _, _, vis = _rvt()
    arr = vis.slrm(dem=c.brut(), radius_cell=int(rayon), ve_factor=float(ve), no_data=c.no_data)
    _publier(c, out, arr, huit_bits, "SIMPLE_LOCAL_RELIEF_MODEL")
