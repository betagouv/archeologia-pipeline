"""Effet de chaque réglage de l'étape 2 sur l'image, de ``rvt_params`` (ce qu'écrit l'interface) jusqu'au tableau
écrit : la chaîne complète d'``indices.create_visualization_products`` et du moteur intégré, sur de petits terrains
synthétiques à réponse connue. Seuls la lecture et l'écriture GDAL sont remplacées (hors QGIS, pas d'osgeo).

Un réglage qui ne change rien à l'image, qui change le nom du fichier sans changer son contenu, ou dont l'effet
n'est pas celui annoncé à l'utilisateur, est un défaut : les défauts connus et non corrigés sont marqués
``xfail(strict=True)`` — le test passera au vert, et le signalera, le jour où la correction arrive.
"""
from __future__ import annotations

import ast
import importlib.util
import itertools
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_RACINE = Path(__file__).resolve().parents[2]
if str(_RACINE / "src") not in sys.path:
    sys.path.insert(0, str(_RACINE / "src"))

from pipeline.ign.products import indices  # noqa: E402
from pipeline.ign.products import moteur_indices as m  # noqa: E402


def _rvt_default():
    """rvt.default embarqué. Hors QGIS, osgeo manque : bouchon le temps de l'import (comme test_moteur_indices)."""
    bouchon = importlib.util.find_spec("osgeo") is None
    if bouchon:
        class _Bouchon(types.ModuleType):
            def __getattr__(self, nom):
                return type(nom, (), {})
        osgeo = types.ModuleType("osgeo")
        osgeo.gdal = _Bouchon("osgeo.gdal")
        sys.modules["osgeo"], sys.modules["osgeo.gdal"] = osgeo, osgeo.gdal
    try:
        from pipeline.ign.products.rvt_embarque import blend, default, vis  # noqa: F401
    finally:
        if bouchon:
            sys.modules.pop("osgeo", None)
            sys.modules.pop("osgeo.gdal", None)
    return default


rvt_default = _rvt_default()

RES = 0.5
ND = -9999.0
N = 120
PENTE = 20.0   # plan incliné, descend vers l'est (orientation 90°)


def _plan() -> np.ndarray:
    _, x = np.mgrid[0:N, 0:N].astype(np.float64)
    return (100 + math.tan(math.radians(PENTE)) * (N - x) * RES).astype(np.float32)


def _bosses() -> np.ndarray:
    rng = np.random.default_rng(3)
    y, x = np.mgrid[0:N, 0:N].astype(np.float64)
    z = 100 + 3 * np.exp(-((x - 35) ** 2 + (y - 35) ** 2) / 50) - 2 * np.exp(-((x - 85) ** 2 + (y - 40) ** 2) / 30)
    z -= 1.0 * (np.abs(y - 90) < 2)
    z += 1.0 * ((x > 60) & (y > 55) & (y < 75))
    return (z + rng.normal(0, 0.02, z.shape)).astype(np.float32)


def _defauts() -> dict:
    from config.config_manager import ConfigManager
    return ConfigManager(_RACINE).default_config()["rvt_params"]


_compteur = itertools.count()


@pytest.fixture
def calcul(monkeypatch, tmp_path):
    """calcul(produit, {section: {clé: valeur}}, terrain, dossier=None) -> (tableau ou None, nom du fichier, journal).

    ``dossier`` réutilisé = relance dans le même dossier de travail (cache des intermédiaires)."""
    ecrits = {}

    def ecrire_raster(src_raster_path, out_raster_path, out_raster_arr, no_data=None, e_type=6, **_):
        Path(out_raster_path).write_bytes(b"tif")
        nom = Path(out_raster_path).name
        nom = nom[:-4] if nom.endswith(".tmp") else nom   # voisin temporaire basculé ensuite (_ecrire, crim.py)
        ecrits[nom] = (np.array(out_raster_arr, dtype=np.float64), e_type)

    monkeypatch.setattr(rvt_default, "save_raster", ecrire_raster)
    monkeypatch.setattr(indices, "reclass_rvt_nodata", lambda *a, **k: None)

    def lancer(produit, reglages, terrain, dossier=None):
        ecrits.clear()
        d = dossier or tmp_path / f"cas{next(_compteur)}"
        d.mkdir(parents=True, exist_ok=True)
        mnt = d / "T_MNT.tif"
        if not mnt.exists():
            mnt.write_bytes(b"mnt")
        dem = terrain()
        monkeypatch.setattr(m.Couches, "lire", classmethod(lambda cls, p: cls(p, dem.copy(), RES, RES, ND)))
        params = _defauts()
        for sec, vals in reglages.items():
            params.setdefault(sec, {}).update(vals)
        journal = []
        res = indices.create_visualization_products(temp_dir=d, current_tile_name="T", products={produit: True},
                                                    rvt_params=params, log=journal.append)
        sortie = res.outputs.get(produit)
        if sortie is None:
            return None, None, journal
        # le fichier publié est soit écrit par le moteur…
        # sinon la sortie de rvt copiée sous le nom publié (mélanges, CVAT : <base>_8bit.tif, ou <base> en flottant)
        cle = sortie.name if sortie.name in ecrits else next(iter(ecrits), None)
        if cle is None:   # rien d'écrit : l'intermédiaire du lancement précédent a été resservi
            return None, sortie.name, journal
        return ecrits[cle][0], sortie.name, journal

    return lancer


def _coeur(a, b=0):
    a = a[b] if a.ndim == 3 else a
    return a[20:-20, 20:-20]


# ───────────────────────── attendus analytiques ─────────────────────────

def _ombrage(az, el, pente=PENTE, orient=90.0):
    zen, s = math.radians(90 - el), math.radians(pente)
    return max(0.0, math.cos(zen) * math.cos(s) + math.sin(zen) * math.sin(s) * math.cos(math.radians(az - orient)))


@pytest.mark.parametrize("az,el,ve", [(315, 35, 1), (90, 35, 1), (270, 35, 1), (0, 35, 1), (315, 60, 1),
                                      (315, 90, 1), (315, 35, 2)])
def test_ombrage_suit_azimut_elevation_et_exageration(calcul, az, el, ve):
    a, _, _ = calcul("HS", {"hs": {"sun_azimuth": az, "sun_elevation": el, "ve_factor": ve, "save_as_8bit": False}},
                     _plan)
    pente = math.degrees(math.atan(ve * math.tan(math.radians(PENTE))))
    assert np.nanmedian(_coeur(a)) == pytest.approx(_ombrage(az, el, pente), abs=1e-4)


def test_pente_en_degres_et_exageration(calcul):
    a, _, _ = calcul("SLO", {"slope": {"unit": 0, "save_as_8bit": False}}, _plan)
    assert np.nanmedian(_coeur(a)) == pytest.approx(PENTE, abs=1e-3)
    a, _, _ = calcul("SLO", {"slope": {"unit": 0, "ve_factor": 3, "save_as_8bit": False}}, _plan)
    assert np.nanmedian(_coeur(a)) == pytest.approx(math.degrees(math.atan(3 * math.tan(math.radians(PENTE)))),
                                                    abs=1e-3)


@pytest.mark.parametrize("code", [2, 1], ids=["pourcentage", "ancien_code_1"])
def test_pente_en_pourcentage(calcul, code):
    """2 = « percent » de rvt_slope ; 1 est ce qu'envoyait « Pourcentage » avant la correction (des radians) : relu
    comme un pourcentage, et rangé sous le même nom que 2, jamais sous l'ancien dossier _U1 (des radians)."""
    a, nom, _ = calcul("SLO", {"slope": {"unit": code, "save_as_8bit": False}}, _plan)
    assert np.nanmedian(_coeur(a)) == pytest.approx(100 * math.tan(math.radians(PENTE)), abs=1e-2)
    assert "_U2_" in nom


def test_pente_en_pourcentage_8_bits_etiree_comme_en_degres(calcul):
    """L'étirement 8 bits de rvt va de 0 à 51 (degrés) : en pourcentage, il va de 0 à tan 51° = 123,49 %."""
    a, _, _ = calcul("SLO", {"slope": {"unit": 2}}, _plan)
    pct = 100 * math.tan(math.radians(PENTE))
    assert np.nanmedian(_coeur(a)) == pytest.approx(255 * (1 - pct / 123.49), abs=1.0)


def test_luminosite_du_mstp_suit_sa_formule(calcul):
    """MSTP = 1 − e^(−L·|DEV|) par canal : la sortie à L se déduit de celle à 1, luminosité appliquée telle quelle."""
    base, _, _ = calcul("MSTP", {"mstp": {"lightness": 1.0, "save_as_8bit": False}}, _bosses)
    for lum in (0.5, 1.2, 2.0):
        a, _, _ = calcul("MSTP", {"mstp": {"lightness": lum, "save_as_8bit": False}}, _bosses)
        dev = -np.log(np.clip(1 - base, 1e-12, None))
        attendu = 1 - np.exp(-lum * dev)
        ok = np.isfinite(a) & np.isfinite(attendu) & (base < 0.999)
        assert np.abs(a[ok] - attendu[ok]).max() < 1e-3


def test_m_hs_flottant_une_bande_par_direction(calcul):
    for n in (4, 16, 32):
        a, _, _ = calcul("M_HS", {"mdh": {"num_directions": n, "save_as_8bit": False}}, _bosses)
        assert a.shape[0] == n


# ───────────────────────── chaque réglage change l'image ─────────────────────────
#: (produit, section, clé, valeur testée) : valeur ≠ défaut, dans les bornes, sur un terrain qui la révèle.
REGLAGES = [
    ("HS", "hs", "sun_azimuth", 90), ("HS", "hs", "sun_elevation", 60), ("HS", "hs", "ve_factor", 3),
    ("M_HS", "mdh", "sun_elevation", 60), ("M_HS", "mdh", "ve_factor", 3),
    ("SVF", "svf", "noise_remove", 2), ("SVF", "svf", "num_directions", 32), ("SVF", "svf", "radius", 20),
    ("SVF", "svf", "ve_factor", 3),
    ("OPNS", "opns", "opns_type", 1), ("OPNS", "opns", "noise_remove", 2), ("OPNS", "opns", "num_directions", 32),
    ("OPNS", "opns", "radius", 20), ("OPNS", "opns", "ve_factor", 3),
    ("SLO", "slope", "ve_factor", 3),
    ("LD", "ldo", "angular_res", 5), ("LD", "ldo", "min_radius", 5), ("LD", "ldo", "max_radius", 30),
    ("LD", "ldo", "observer_h", 5.0), ("LD", "ldo", "ve_factor", 3),
    ("SLRM", "slrm", "radius", 10), ("SLRM", "slrm", "ve_factor", 3),
    ("VAT", "vat", "terrain_type", 1), ("VAT", "vat", "terrain_type", 2),
    ("MSTP", "mstp", "local_scale_min", 5), ("MSTP", "mstp", "local_scale_max", 15),
    ("MSTP", "mstp", "local_scale_step", 4), ("MSTP", "mstp", "meso_scale_min", 31),
    ("MSTP", "mstp", "meso_scale_max", 61), ("MSTP", "mstp", "meso_scale_step", 6),
    ("MSTP", "mstp", "broad_scale_min", 70), ("MSTP", "mstp", "broad_scale_max", 300),
    ("MSTP", "mstp", "broad_scale_step", 120), ("MSTP", "mstp", "lightness", 1.2),
    ("PRISM", "prism", "terrain_type", 1), ("PRISM", "prism", "terrain_type", 2),
    ("CRIM", "crim", "colormap", "Greys_r"), ("CRIM", "crim", "min_colormap_cut", 0.3),
    ("CRIM", "crim", "max_colormap_cut", 0.7),
]


@pytest.mark.parametrize("produit,section,cle,valeur", REGLAGES, ids=[f"{p}.{c}={v}" for p, _, c, v in REGLAGES])
def test_le_reglage_change_l_image(calcul, produit, section, cle, valeur):
    ref, _, _ = calcul(produit, {}, _bosses)
    a, _, _ = calcul(produit, {section: {cle: valeur}}, _bosses)
    assert ref is not None and a is not None
    assert ref.shape != a.shape or not np.array_equal(np.nan_to_num(ref, nan=-1), np.nan_to_num(a, nan=-1))


@pytest.mark.parametrize("produit,section,cle,valeur", REGLAGES, ids=[f"{p}.{c}={v}" for p, _, c, v in REGLAGES])
def test_le_reglage_change_le_nom_du_fichier(calcul, produit, section, cle, valeur):
    """Sans quoi une relance dans le même dossier resservirait l'intermédiaire du réglage précédent."""
    _, nom_ref, _ = calcul(produit, {}, _bosses)
    _, nom, _ = calcul(produit, {section: {cle: valeur}}, _bosses)
    assert nom != nom_ref



# ───────────────────────── défauts mesurés ─────────────────────────

@pytest.mark.parametrize("produit,section", [("VAT", "vat"), ("PRISM", "prism"), ("CVAT", "cvat")])
def test_sortie_flottante_des_melanges(calcul, produit, section):
    """rvt écrit le flottant d'un mélange sous <base> sans extension : il doit être retrouvé et publié, en flottant."""
    a, nom, _ = calcul(produit, {section: {"save_as_8bit": False}}, _bosses)
    assert a is not None and nom.endswith("_F32.tif")
    assert not np.allclose(a[np.isfinite(a)], np.round(a[np.isfinite(a)]))   # des valeurs flottantes, pas des octets


@pytest.mark.parametrize("produit,section", [("SVF", "svf"), ("OPNS", "opns")])
@pytest.mark.xfail(strict=True, reason="DÉFAUT (rvt) : rayon min = max(arrondi(rayon × 10 %), 1) : au rayon par "
                                       "défaut de 10 px, la suppression du bruit « faible » vaut « aucune »")
def test_bruit_faible_au_rayon_par_defaut(calcul, produit, section):
    ref, _, _ = calcul(produit, {section: {"noise_remove": 0}}, _bosses)
    a, _, _ = calcul(produit, {section: {"noise_remove": 1}}, _bosses)
    assert not np.array_equal(np.nan_to_num(ref), np.nan_to_num(a))


def test_bruit_faible_agit_des_15_px(calcul):
    ref, _, _ = calcul("SVF", {"svf": {"noise_remove": 0, "radius": 20}}, _bosses)
    a, _, _ = calcul("SVF", {"svf": {"noise_remove": 1, "radius": 20}}, _bosses)
    assert not np.array_equal(np.nan_to_num(ref), np.nan_to_num(a))


@pytest.mark.xfail(strict=True, reason="DÉFAUT : rayon min du LD à 0 (permis par l'interface et les bornes) : "
                                       "division par une distance nulle, image saturée")
def test_ld_rayon_min_nul(calcul):
    a, _, _ = calcul("LD", {"ldo": {"min_radius": 0, "save_as_8bit": False}}, _bosses)
    assert np.isfinite(a).mean() > 0.9 and np.nanstd(a) > 0


def test_exageration_du_mstp_sans_effet_mesurable(calcul):
    """L'écart du MSTP est normalisé par l'écart type local : l'exagération verticale s'y annule (réglage à
    masquer, ou à documenter comme sans effet)."""
    ref, _, _ = calcul("MSTP", {"mstp": {"save_as_8bit": False}}, _bosses)
    a, _, _ = calcul("MSTP", {"mstp": {"ve_factor": 3, "save_as_8bit": False}}, _bosses)
    ok = np.isfinite(ref) & np.isfinite(a)
    d = np.abs(a[ok] - ref[ok])
    # mathématiquement nul ; seul reste le bruit d'arrondi du flottant (images intégrales de rvt : quelques pixels
    # plats de la grande échelle, où l'écart type local est presque nul, en gardent un écart visible)
    assert np.median(d) < 1e-4 and d.mean() < 5e-3


@pytest.mark.parametrize("produit,section", [("HS", "hs"), ("SLO", "slope"), ("MSTP", "mstp"), ("CRIM", "crim")])
def test_relance_apres_bascule_8_bits(calcul, tmp_path, produit, section):
    d = tmp_path / "meme_dossier"
    calcul(produit, {section: {"save_as_8bit": True}}, _bosses, dossier=d)
    a, _, journal = calcul(produit, {section: {"save_as_8bit": False}}, _bosses, dossier=d)
    assert not any("réutilisé" in ligne for ligne in journal), journal
    assert a is not None and not np.allclose(a, np.round(a))   # flottant, pas des octets


def test_relance_apres_changement_de_pas_du_mstp(calcul, tmp_path):
    d = tmp_path / "meme_dossier"
    calcul("MSTP", {}, _bosses, dossier=d)
    a, _, journal = calcul("MSTP", {"mstp": {"local_scale_step": 4}}, _bosses, dossier=d)
    assert not any("réutilisé" in ligne for ligne in journal)


# ───────────────────────── défauts et bornes d'un endroit à l'autre ─────────────────────────

def _reg_ui():
    """(section, clé) -> (défaut de _reg, bornes du widget) relus dans step_2_indices.py."""
    arbre = ast.parse((_RACINE / "src/ui/steps/step_2_indices.py").read_text(encoding="utf-8"))
    widgets, regs = {}, {}
    for n in ast.walk(arbre):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) \
                and n.value.func.attr in ("_mk_spin", "_mk_dspin"):
            widgets[ast.unparse(n.targets[0])] = [ast.literal_eval(a) for a in n.value.args]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_reg":
            sec = ast.literal_eval(n.args[0])
            try:
                defaut = ast.literal_eval(n.args[4])
            except ValueError:
                continue
            regs[(sec[-1], ast.literal_eval(n.args[1]))] = (defaut, widgets.get(ast.unparse(n.args[2])))
    return regs


SECTION_PRODUIT = {"hs": "HS", "mdh": "M_HS", "svf": "SVF", "opns": "OPNS", "slope": "SLO", "ldo": "LD",
                   "slrm": "SLRM", "mstp": "MSTP"}


def test_defauts_de_l_interface_egaux_a_ceux_de_la_configuration():
    rvt = _defauts()
    for (sec, cle), (defaut, _) in _reg_ui().items():
        if sec == "processing":
            continue
        assert rvt[sec][cle] == defaut, (sec, cle)


def test_replis_du_pipeline_egaux_aux_defauts_de_l_interface(calcul):
    _, nom_cle_absente, _ = calcul("MSTP", {"mstp": {}}, _bosses)
    params = _defauts()
    del params["mstp"]
    from pipeline.ign.products.rvt_naming import get_rvt_param_suffix
    assert get_rvt_param_suffix("MSTP", params) == get_rvt_param_suffix("MSTP", _defauts())


@pytest.mark.xfail(strict=True, reason="BORNES : l'interface permet des valeurs que le moteur refuse (dalle en "
                                       "échec au calcul) : M-HS directions, LD résolution/rayons/hauteur, SLRM rayon")
def test_bornes_de_l_interface_dans_celles_du_moteur():
    hors = []
    for (sec, cle), (_, widget) in _reg_ui().items():
        b = m.BORNES.get(SECTION_PRODUIT.get(sec, ""), {}).get(cle)
        if b and widget and (widget[0] < b[0] or widget[1] > b[1]):
            hors.append((sec, cle, widget[:2], b))
    assert not hors, hors


@pytest.mark.xfail(strict=True, reason="NOM TROMPEUR : le MNT publié s'appelle …_MNT_A_0M50_… quelle que soit la "
                                       "résolution réglée (la fonction ne la reçoit même pas)")
def test_nom_du_mnt_publie_ne_fige_pas_la_resolution():
    from pipeline.ign.products.rvt_naming import get_rvt_source_and_dest_filenames
    _, dest = get_rvt_source_and_dest_filenames("MNT", "T", "0816", "6333", {})
    assert "0M50" not in dest


@pytest.mark.parametrize("produit", ["HS", "M_HS", "SVF", "OPNS", "SLO", "LD", "SLRM", "VAT", "MSTP", "CVAT", "PRISM",
                                     "CRIM"])
def test_sortie_flottante_dans_le_nom_8_bits_inchange(produit):
    """Le 8 bits (défaut) garde son nom ; le flottant prend _F32 : les deux ne se resservent jamais l'un l'autre."""
    from pipeline.ign.products.rvt_naming import SECTION_DU_PRODUIT, get_rvt_folder_name
    defauts = _defauts()
    huit = get_rvt_folder_name(produit, defauts)
    defauts[SECTION_DU_PRODUIT[produit]]["save_as_8bit"] = False
    assert get_rvt_folder_name(produit, defauts) == huit + "_F32"
    assert "_F32" not in huit
