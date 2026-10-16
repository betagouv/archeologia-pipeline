"""Moteur d'indices intégré : les indices de visualisation calculés dans le plugin avec la bibliothèque
rvt embarquée (``rvt_embarque/`` : copie figée de celle de rvt-qgis 1.0.1, voir son NOTICE), au lieu des
algorithmes Processing de l'extension rvt-qgis.

Règle : chaque produit appelle les MÊMES fonctions de rvt, avec les MÊMES arguments, que l'algorithme
Processing qu'il remplace (``processing_provider/rvt_*.py`` de rvt-qgis 1.0.1). Tout écart est nommé là où il
est fait et prouvé au pixel près (``tests/unit/test_moteur_indices.py``, bancs N2). Par dalle, ``Couches`` lit le
MNT une fois et partage une couche entre produits quand rvt l'aurait calculée à l'identique pour chacun : la clé
de partage est faite des paramètres que la fonction de rvt reçoit réellement, jamais du seul réglage de
l'interface.

Deux corrections voulues (0.16.0, changelog) : la luminosité du MSTP n'est plus arrondie à l'entier, et le M-HS
8 bits applique l'élévation et l'exagération réglées. indices.py note leur version à côté de l'intermédiaire
(VERSION_CALCUL) pour qu'un ancien soit recalculé une fois.

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
PRODUITS_MOTEUR = frozenset({"HS", "M_HS", "SVF", "OPNS", "SLO", "LD", "SLRM", "VAT", "MSTP", "CVAT", "PRISM",
                             "CRIM"})

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

#: Unités de pente de rvt:rvt_slope, dans l'ordre de son enum UNIT (l'interface envoie 0 ou 2).
UNITES_PENTE = ("degree", "radian", "percent")

#: Haut de l'étirement 8 bits d'une pente en pourcentage : tan(51°), soit l'étirement 0–51° de rvt (slp_bytscl, pensé
#: pour des degrés) exprimé en pourcentage. Sans lui, tout ce qui dépasse 51 % (≈ 27°) sortirait noir.
POURCENT_8BITS_MAX = 123.49


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

    def horizon(self, rayon: int, directions: int, bruit: int, ve: float, res: float, svf: bool = True,
                opns: bool = True) -> Dict[str, Any]:
        """SVF et openness positive d'un même balayage : rvt.vis.sky_view_factor les tire des mêmes pentes
        maximales, chaque sortie est celle d'un appel qui ne calcule qu'elle. Hors partage, seules les
        sorties demandées sont calculées (pas d'accumulateur de plus : la mémoire d'avant)."""
        _, _, _, vis = _rvt()
        svf_, opns_ = (True, True) if self.partage else (svf, opns)
        return self._une_fois(("horizon", rayon, directions, bruit, ve, res), lambda: vis.sky_view_factor(
            dem=self.brut(), resolution=res, compute_svf=svf_, compute_asvf=False, compute_opns=opns_,
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


def _publier(c: Couches, out: Path, arr, huit_bits: bool, visualisation: str, defaut=None) -> None:
    """Sortie d'un algorithme rvt:* : float_to_8bit puis Byte, ou le flottant brut en Float32. ``defaut`` : un
    DefaultValues réglé autrement que celui de rvt (étirement de la pente en pourcentage)."""
    _, _, default, _ = _rvt()
    if huit_bits:
        _ecrire(c, out, (defaut or default.DefaultValues()).float_to_8bit(
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
    h = c.horizon(int(rayon), int(directions), int(bruit), float(ve), c.res_x, svf=True, opns=False)
    _publier(c, out, h["svf"], huit_bits, "SKY_VIEW_FACTOR")


def calculer_opns(c: Couches, out: Path, *, negative: bool, rayon: int, directions: int, bruit: int, ve: float,
                  huit_bits: bool) -> None:
    """rvt_opns.py : type négatif = openness de −MNT, sans-donnée non remplacé avant la négation (sans partage)."""
    _verifier("OPNS", radius=rayon, num_directions=directions, noise_remove=bruit, ve_factor=ve)
    if negative:
        arr = c.ouverture_negative(int(rayon), int(directions), int(bruit), float(ve), c.res_x, nan_avant=False)
    else:
        arr = c.horizon(int(rayon), int(directions), int(bruit), float(ve), c.res_x, svf=False, opns=True)["opns"]
    _publier(c, out, arr, huit_bits, "NEGATIVE_OPENNESS" if negative else "POSITIVE_OPENNESS")


def calculer_slo(c: Couches, out: Path, *, unite: int, ve: float, huit_bits: bool) -> None:
    """rvt_slope.py (``unite`` = indice de l'enum de l'algorithme, cf. UNITES_PENTE)."""
    _verifier("SLO", unit=unite, ve_factor=ve)
    defaut = None
    if UNITES_PENTE[int(unite)] == "percent":
        _, _, default, _ = _rvt()
        defaut = default.DefaultValues()
        defaut.slp_bytscl = (defaut.slp_bytscl[0], defaut.slp_bytscl[1], POURCENT_8BITS_MAX)
    _publier(c, out, c.pente(UNITES_PENTE[int(unite)], float(ve), c.res_x, c.res_y), huit_bits, "SLOPE", defaut)


def calculer_slrm(c: Couches, out: Path, *, rayon: int, ve: float, huit_bits: bool) -> None:
    """rvt_slrm.py."""
    _verifier("SLRM", radius=rayon, ve_factor=ve)
    _, _, _, vis = _rvt()
    arr = vis.slrm(dem=c.brut(), radius_cell=int(rayon), ve_factor=float(ve), no_data=c.no_data)
    _publier(c, out, arr, huit_bits, "SIMPLE_LOCAL_RELIEF_MODEL")


def composer_mhs_8bits(c: Couches, *, elevation: float, ve: float):
    """Le M-HS 8 bits : trois ombrages (315°, 22,5°, 90°) étirés de 0 à 1 puis en octets, la branche « value » de
    DefaultValues.float_to_8bit(MULTI_HILLSHADE) — mais à l'élévation et à l'exagération réglées, que rvt-qgis
    remplaçait par 35° et 1 (décision 0.16.0 ; identique aux défauts)."""
    import numpy as np
    _, blend_func, default, vis = _rvt()
    bornes = default.DefaultValues().mhs_bytscl   # ("value", 0, 1)
    return np.array([vis.byte_scale(data=blend_func.normalize_lin(
        image=c.ombrage(float(az), float(elevation), float(ve), c.res_x, c.res_y),
        minimum=bornes[1], maximum=bornes[2]), no_data=np.nan, c_min=0, c_max=1) for az in (315, 22.5, 90)])


def calculer_mhs(c: Couches, out: Path, *, directions: int, elevation: float, ve: float, huit_bits: bool) -> None:
    """rvt_multi_hillshade.py. En flottant : N bandes, une par direction. En 8 bits : ``composer_mhs_8bits`` (le
    nombre de directions y est sans effet, comme dans rvt-qgis qui jetait son calcul)."""
    if not huit_bits:
        _verifier("M_HS", num_directions=directions, sun_elevation=elevation, ve_factor=ve)
        _, _, _, vis = _rvt()
        _ecrire(c, out, vis.multi_hillshade(
            dem=c.brut(), resolution_x=c.res_x, resolution_y=c.res_y, nr_directions=int(directions),
            sun_elevation=float(elevation), ve_factor=float(ve), no_data=c.no_data), 6)
        return
    _verifier("M_HS", sun_elevation=elevation, ve_factor=ve)
    _ecrire(c, out, composer_mhs_8bits(c, elevation=elevation, ve=ve), 1)


def calculer_mstp(c: Couches, out: Path, *, local: Tuple[int, int, int], meso: Tuple[int, int, int],
                  large: Tuple[int, int, int], luminosite: float, ve: float, huit_bits: bool) -> None:
    """rvt_mstp.py, mais la luminosité réglée telle quelle : rvt-qgis la passait par int() (1,2 → 1)."""
    _verifier("MSTP", lightness=luminosite, ve_factor=ve,
              **{f"{e}_scale_{k}": v for e, t in (("local", local), ("meso", meso), ("broad", large))
                 for k, v in zip(("min", "max", "step"), t)})
    _, _, _, vis = _rvt()
    arr = vis.mstp(dem=c.brut(), local_scale=tuple(local), meso_scale=tuple(meso), broad_scale=tuple(large),
                   lightness=float(luminosite), ve_factor=float(ve), no_data=c.no_data)
    _publier(c, out, arr, huit_bits, "MULTI_SCALE_TOPOGRAPHIC_POSITION")


#: Fonctions de rvt.vis qui écrivent NaN DANS le MNT reçu (``dem[dem == no_data] = np.nan`` avant toute copie) :
#: appelées par un mélange, elles laissent son MNT « NaN-isé » pour les couches calculées ensuite. « shadow » passe
#: par sky_illumination. Relevé sur rvt par tests/unit/test_moteur_indices.py:test_vis_mutantes_comme_rvt.
_VIS_MUTANTES = frozenset({"hillshade", "multiple directions hillshade", "simple local relief model",
                           "local dominance", "sky illumination", "shadow", "multi-scale relief model",
                           "multi-scale topographic position"})


def injecter_couches(c: Couches, comb, defaut) -> None:
    """Fournit à une combinaison rvt (après ``apply_terrain``) les couches qu'elle calculerait elle-même, avec les
    paramètres que ``BlenderCombination.render_all_images`` leur donnerait : ceux de ``defaut`` (que le terrain a
    réglés), la résolution X pour X et Y. Une couche d'un autre type reste vide : rvt la calcule, comme avant."""
    import numpy as np
    res = c.res_x
    ve = float(defaut.ve_factor)
    for i, couche in enumerate(comb.layers):
        nom = (couche.vis or "").lower()
        if nom == "hillshade":
            couche.image = c.ombrage(float(defaut.hs_sun_azi), float(defaut.hs_sun_el), ve, res, res)
        elif nom == "multiple directions hillshade":   # trois ombrages, exagération 1 (pas defaut.ve_factor)
            couche.image = np.array([c.ombrage(float(az), float(defaut.mhs_sun_el), 1.0, res, res)
                                     for az in (315, 22.5, 90)])
        elif nom == "slope gradient":
            couche.image = c.pente(defaut.slp_output_units, ve, res, res)
        elif nom in ("sky-view factor", "openness - positive"):
            svf = nom == "sky-view factor"
            h = c.horizon(int(defaut.svf_r_max), int(defaut.svf_n_dir), int(defaut.svf_noise), ve, res, svf=svf,
                          opns=not svf)
            couche.image = h["svf" if svf else "opns"]
        elif nom == "openness - negative":
            # render_all_images parcourt les couches de la dernière à la première : une fonction « mutante »
            # appelée avant celle-ci a déjà remplacé le sans-donnée par NaN dans le MNT de la combinaison.
            nan_avant = any((x.vis or "").lower() in _VIS_MUTANTES for x in comb.layers[i + 1:])
            couche.image = c.ouverture_negative(int(defaut.svf_r_max), int(defaut.svf_n_dir),
                                                int(defaut.svf_noise), ve, res, nan_avant=nan_avant)


def combinaison(indice: int, terrain: int):
    """(combinaison, DefaultValues) de rvt_blender.py : ``indice`` et ``terrain`` sont les indices de ses enums
    BLEND_COMBINATION et TERRAIN_TYPE (ordre des fichiers default_blender_combinations.json et
    default_terrains_settings.json). Copie neuve à chaque appel : rvt-qgis réutilisait un objet partagé entre les
    appels d'une même session QGIS."""
    import copy
    blend, _, default, _ = _rvt()
    combos = blend.BlenderCombinations()
    combos.read_from_file(str(chemin_reglages("default_blender_combinations.json")))
    comb = copy.deepcopy(combos.combinations[int(indice)])
    terrains = blend.TerrainsSettings()
    terrains.read_from_file(str(chemin_reglages("default_terrains_settings.json")))
    defaut = default.DefaultValues()
    terrains.terrains_settings[int(terrain)].apply_terrain(default=defaut, combination=comb)
    return comb, defaut


def rendre_melange(c: Couches, produit: str, indice: int, terrain: int, sortie_base: Path, huit_bits: bool) -> None:
    """rvt_blender.py (combinaisons ordinaires) : rvt écrit lui-même ``<sortie_base>_8bit.tif`` en 8 bits. Comme
    Processing, qui n'ajoute pas d'extension à OUTPUT (vérifié le 2026-10-15), le flottant va dans
    ``<sortie_base>`` sans extension, que indices.py ne trouve pas : VAT et PRISM flottants restent absents, comme
    avant (signalé à l'utilisateur)."""
    _verifier(produit, terrain_type=terrain, blend_combination=indice)
    comb, defaut = combinaison(indice, terrain)
    injecter_couches(c, comb, defaut)
    comb.add_dem_arr(dem_arr=c.brut(), dem_resolution=c.res_x)
    comb.add_dem_path(str(c.chemin))
    comb.render_all_images(default=defaut, save_visualizations=False, save_render_path=str(sortie_base),
                           save_float=not huit_bits, save_8bit=huit_bits, no_data=c.no_data)


def vat_cvat(c: Couches, terrain: str):
    """Une moitié de CVAT (cvat.py, branche CVAT de rvt_blender.py) : blender_VAT.json, terrain par son nom."""
    blend, _, default, _ = _rvt()
    defaut = default.DefaultValues()
    comb = blend.BlenderCombination()
    comb.read_from_file(str(chemin_reglages("blender_VAT.json")))
    terrains = blend.TerrainsSettings()
    terrains.read_from_file(str(chemin_reglages("default_terrains_settings.json")))
    terrains.select_terrain_settings_by_name(terrain).apply_terrain(default=defaut, combination=comb)
    injecter_couches(c, comb, defaut)
    comb.add_dem_arr(dem_arr=c.brut(), dem_resolution=c.res_x)
    return comb.render_all_images(default=defaut, no_data=c.no_data)


def crim(c: Couches, *, colormap: str, min_colormap_cut: float, max_colormap_cut: float):
    """rvt.blend.color_relief_image_map, mêmes couches et mêmes étirements, ses trois couches de calcul prises dans
    ``Couches`` (rvt : MNT « NaN-isé », openness ± à rayon 10 et paramètres par défaut, pente en radians)."""
    blend, _, default, _ = _rvt()
    d = default.DefaultValues()
    res = c.res_x
    ve = float(d.ve_factor)
    pos = c.horizon(10, int(d.svf_n_dir), int(d.svf_noise), ve, res, svf=False, opns=True)["opns"]
    neg = c.ouverture_negative(10, int(d.svf_n_dir), int(d.svf_noise), ve, res, nan_avant=True)
    pos_neg = pos - neg
    comb = blend.BlenderCombination()
    comb.create_layer(vis_method="Openness_Pos-Neg", normalization="value", minimum=-28, maximum=28,
                      blend_mode="overlay", opacity=50, image=pos_neg)
    comb.create_layer(vis_method="Openness_Pos-Neg", normalization="value", minimum=-28, maximum=28,
                      blend_mode="luminosity", opacity=50, image=pos_neg)   # le même tableau, comme rvt
    comb.create_layer(vis_method="slope gradient rad", normalization="value", minimum=0, maximum=0.8,
                      blend_mode="normal", opacity=100, colormap=colormap, min_colormap_cut=min_colormap_cut,
                      max_colormap_cut=max_colormap_cut, image=c.pente("radian", 1.0, res, res))
    return comb.render_all_images()
