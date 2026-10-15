"""Moteur d'indices intégré (vague 2, 0.16.0) : identité au pixel avec rvt seul.

Chaque test compare, sur un petit MNT synthétique à sans-donnée (coin, bande, pixel isolé), le calcul du moteur
à l'appel de rvt que fait l'algorithme Processing de rvt-qgis 1.0.1 qu'il remplace : même type, même forme, mêmes
valeurs, NaN aux mêmes places.

Les tests SANS argument se lancent aussi hors pytest, avec le Python de QGIS 3.44.1 (numpy 1.26) et de QGIS 4.0.3
(numpy 2.x), dont les règles de promotion float32/float64 diffèrent :
  & C:\\OSGeo4W\\bin\\python-qgis.bat tests\\unit\\test_moteur_indices.py <fichier de sortie>
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

_RACINE = Path(__file__).resolve().parents[2]
if str(_RACINE / "src") not in sys.path:
    sys.path.insert(0, str(_RACINE / "src"))

from pipeline.ign.products import moteur_indices as m  # noqa: E402

ND = -9999.0
RES = 0.5


def _rvt():
    """Modules du rvt embarqué. Hors QGIS, osgeo manque : bouchon le temps de l'import (tile.py annote avec
    gdal.Dataset ; ces tests ne lisent ni n'écrivent de raster), retiré ensuite de sys.modules."""
    bouchon = importlib.util.find_spec("osgeo") is None
    if bouchon:
        class _Bouchon(types.ModuleType):
            def __getattr__(self, nom):
                return type(nom, (), {})
        osgeo = types.ModuleType("osgeo")
        osgeo.gdal = _Bouchon("osgeo.gdal")
        sys.modules["osgeo"], sys.modules["osgeo.gdal"] = osgeo, osgeo.gdal
    try:
        from pipeline.ign.products.rvt_embarque import blend, blend_func, default, vis
    finally:
        if bouchon:
            sys.modules.pop("osgeo", None)
            sys.modules.pop("osgeo.gdal", None)
    return blend, blend_func, default, vis


_rvt()   # chargé une fois ici : les méthodes de Couches l'importent ensuite sans GDAL, quel que soit l'ordre des tests


def _mnt(n: int = 120) -> np.ndarray:
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    d = (200 + 3 * np.sin(x / 7) + 2 * np.cos(y / 5) + rng.normal(0, 0.3, (n, n))).astype(np.float32)
    d[:12, :20] = ND      # coin sans donnée
    d[40:44, :] = ND      # bande
    d[60, 60] = ND        # pixel isolé
    return d


def _couches(dem: np.ndarray) -> m.Couches:
    return m.Couches(Path("mnt.tif"), dem.copy(), RES, RES, ND)


def _egal(a, b, quoi: str) -> None:
    a, b = np.asarray(a), np.asarray(b)
    assert (a.dtype, a.shape) == (b.dtype, b.shape), (quoi, a.dtype, b.dtype, a.shape, b.shape)
    if not np.array_equal(a, b, equal_nan=True):
        x, y = a.astype(np.float64), b.astype(np.float64)
        raise AssertionError(f"{quoi} : {int((~((x == y) | (np.isnan(x) & np.isnan(y)))).sum())} pixels différents")


#: (min_rad, max_rad, angular_res, observer_h, ve) : défauts, non défauts, bornes, min > max (rvt rend du NaN, le
#: moteur aussi), observateur entier.
JEUX_LD = [(10, 20, 15, 1.7, 1.0), (5, 30, 10, 3.5, 2.0), (0, 1, 4, 0.5, 1.0), (20, 20, 32, 20.0, -1.5),
           (12, 7, 15, 1.7, 1.0), (10, 20, 15, 2.0, 1.0)]


def test_dominance_locale_identique_a_rvt():
    _, _, _, vis = _rvt()
    dem = _mnt()
    for mn, mx, ang, obs, ve in JEUX_LD:
        with np.errstate(divide="ignore", invalid="ignore"):
            moi = m.dominance_locale(dem.copy(), min_rad=mn, max_rad=mx, angular_res=ang, observer_height=obs,
                                     ve_factor=ve, no_data=ND)
            rvt = vis.local_dominance(dem.copy(), min_rad=mn, max_rad=mx, angular_res=ang, observer_height=obs,
                                      ve_factor=ve, no_data=ND)
        _egal(moi, rvt, f"LD {(mn, mx, ang, obs, ve)}")


def test_hors_bornes_refuse_en_nommant_le_reglage():
    assert m.hors_bornes("LD", {"angular_res": 15, "max_radius": 20}) == ""
    assert m.hors_bornes("LD", {"angular_res": 45}) == "LD : angular_res=45 hors des bornes de RVT (4 à 32)"
    assert "max_radius=0" in m.hors_bornes("LD", {"max_radius": 0})
    assert m.hors_bornes("CVAT", {"x": 1}) == ""   # pas de borne : rien à refuser


def test_bornes_egales_a_celles_des_algorithmes_de_rvt_qgis():
    """BORNES recopie les minValue/maxValue des QgsProcessingParameterNumber de rvt-qgis 1.0.1 : relues dans
    l'extension voisine quand elle est installée (poste de développement)."""
    rvt_qgis = next((p for p in sorted(_RACINE.parent.glob("rvt*")) if (p / "processing_provider").is_dir()), None)
    if rvt_qgis is None:
        import pytest
        pytest.skip("extension rvt-qgis absente de ce poste")
    fichiers = {"HS": "rvt_hillshade.py", "M_HS": "rvt_multi_hillshade.py", "SVF": "rvt_svf.py",
                "OPNS": "rvt_opns.py", "SLO": "rvt_slope.py", "LD": "rvt_local_dom.py", "SLRM": "rvt_slrm.py",
                "MSTP": "rvt_mstp.py"}
    ecarts = {("LD", "max_radius"): (1, 100)}   # rvt rend une image vide à 0 : refusé ici (docstring de BORNES)
    for produit, nom in fichiers.items():
        lues = {}
        for n in ast.walk(ast.parse((rvt_qgis / "processing_provider" / nom).read_text(encoding="utf-8"))):
            appel = getattr(n, "func", None)
            if isinstance(n, ast.Call) and getattr(appel, "id", getattr(appel, "attr", None)) == "QgsProcessingParameterNumber":
                kw = {k.arg: k.value for k in n.keywords}
                lues[ast.literal_eval(kw["name"]).lower()] = (ast.literal_eval(kw["minValue"]),
                                                              ast.literal_eval(kw["maxValue"]))
        assert set(lues) <= set(m.BORNES[produit]), (produit, set(lues) - set(m.BORNES[produit]))
        for cle, bornes in lues.items():
            assert m.BORNES[produit][cle] == ecarts.get((produit, cle), bornes), (produit, cle, bornes)


def test_interrupteur_renvoie_ld_vers_processing(monkeypatch):
    monkeypatch.delenv(m.VARIABLE_MOTEUR, raising=False)
    assert m.par_le_moteur("LD")
    monkeypatch.setenv(m.VARIABLE_MOTEUR, "processing")
    assert not m.par_le_moteur("LD") and m.par_le_moteur("CVAT")


def test_svf_et_openness_d_un_seul_balayage_identiques_aux_appels_separes():
    """rvt.vis.sky_view_factor tire SVF et openness des mêmes pentes maximales : un appel qui calcule les deux
    rend, pour chacun, l'image d'un appel qui ne calcule que lui (rvt_svf.py, rvt_opns.py)."""
    _, _, _, vis = _rvt()
    dem = _mnt()
    for r, d, b, ve in [(10, 16, 0, 1.0), (20, 32, 2, 2.0), (50, 8, 3, 1.0)]:
        h = _couches(dem).horizon(r, d, b, ve, RES)
        for cle, svf, opns in (("svf", True, False), ("opns", False, True)):
            seul = vis.sky_view_factor(dem.copy(), resolution=RES, compute_svf=svf, compute_opns=opns, svf_n_dir=d,
                                       svf_r_max=r, svf_noise=b, ve_factor=ve, no_data=ND)[cle]
            _egal(h[cle], seul, f"{cle} {(r, d, b, ve)}")


def test_openness_negative_comme_rvt_opns():
    """rvt_opns.py (type négatif) passe −MNT tel quel : −no_data n'est plus reconnu, le sans-donnée compte comme un
    relief près des bords. Reproduit à l'identique (signalé à l'utilisateur, pas corrigé en silence)."""
    _, _, _, vis = _rvt()
    dem = _mnt()
    attendu = vis.sky_view_factor(dem.copy() * -1, resolution=RES, compute_svf=False, compute_opns=True,
                                  svf_n_dir=8, svf_r_max=15, svf_noise=1, ve_factor=1.0, no_data=ND)["opns"]
    _egal(_couches(dem).ouverture_negative(15, 8, 1, 1.0, RES, nan_avant=False), attendu, "OPNS négative")


def test_une_couche_partagee_est_rendue_en_copie():
    _, _, _, vis = _rvt()
    c = _couches(_mnt())
    a = c.ombrage(315.0, 35.0, 1.0, RES, RES)
    a[:] = 0   # un consommateur qui écrit dans sa couche (les normalisations de rvt le font)
    attendu = vis.hillshade(dem=_mnt(), resolution_x=RES, resolution_y=RES, sun_azimuth=315.0, sun_elevation=35.0,
                            ve_factor=1.0, no_data=ND)
    _egal(c.ombrage(315.0, 35.0, 1.0, RES, RES), attendu, "ombrage partagé")
    assert len(c._memo) == 1


def test_au_dela_du_plafond_rien_n_est_garde(monkeypatch):
    """Review Focus 4 : au-delà d'une dalle standard, chaque produit recalcule ses couches (mémoire d'avant)."""
    monkeypatch.setattr(m, "PARTAGE_MAX_PX", 100)
    c = _couches(_mnt())
    c.ombrage(315.0, 35.0, 1.0, RES, RES)
    assert not c.partage and c._memo == {} and c.taille == (120, 120)


def test_unite_de_pente_suit_l_enum_de_rvt_slope():
    """L'interface envoie 0 (« Degrés ») ou 1 (« Pourcentage ») ; rvt:rvt_slope lit l'indice dans
    ["degree", "radian", "percent"] : 1 donne des RADIANS. Reproduit tel quel (identité) et signalé à
    l'utilisateur : corriger changerait l'image, c'est à décider."""
    assert m.UNITES_PENTE == ("degree", "radian", "percent")


def test_sans_partage_le_balayage_ne_calcule_que_ce_qui_est_demande(monkeypatch):
    """Review round 1 : hors partage, SVF seul n'alloue pas l'accumulateur d'openness (rvt_svf.py : compute_opns=False)."""
    _, _, _, vis = _rvt()
    appels = []
    vrai = vis.sky_view_factor

    def espion(*a, **k):
        appels.append((k["compute_svf"], k["compute_opns"]))
        return vrai(*a, **k)
    monkeypatch.setattr(vis, "sky_view_factor", espion)
    dem = _mnt()
    monkeypatch.setattr(m, "PARTAGE_MAX_PX", 100)
    c = _couches(dem)
    h = c.horizon(10, 16, 0, 1.0, RES, svf=True, opns=False)
    _egal(h["svf"], vrai(dem.copy(), resolution=RES, compute_svf=True, compute_opns=False, svf_n_dir=16,
                         svf_r_max=10, svf_noise=0, ve_factor=1.0, no_data=ND)["svf"], "svf seul")
    h = c.horizon(10, 16, 0, 1.0, RES, svf=False, opns=True)
    _egal(h["opns"], vrai(dem.copy(), resolution=RES, compute_svf=False, compute_opns=True, svf_n_dir=16,
                          svf_r_max=10, svf_noise=0, ve_factor=1.0, no_data=ND)["opns"], "opns seul")
    assert appels[:2] == [(True, False), (False, True)]
    # avec partage : un seul balayage pour les deux sorties
    monkeypatch.setattr(m, "PARTAGE_MAX_PX", 2801 * 2801)
    appels.clear()
    c = _couches(dem)
    c.horizon(10, 16, 0, 1.0, RES, svf=True, opns=False)
    c.horizon(10, 16, 0, 1.0, RES, svf=False, opns=True)
    assert appels == [(True, True)] and len(c._memo) == 1


def test_m_hs_8_bits_aux_defauts_identique_a_rvt_qgis():
    """rvt_multi_hillshade.py en 8 bits : DefaultValues().float_to_8bit(MULTI_HILLSHADE) sur le MNT — trois ombrages
    (315°, 22,5°, 90°) à 35°, sans exagération. Le moteur, à 35° et VE 1, compose les mêmes trois ombrages."""
    _, _, default, _ = _rvt()
    dem = _mnt()
    attendu = default.DefaultValues().float_to_8bit(float_arr=dem.copy(), visualization=default.RVTVisualization.MULTI_HILLSHADE,
                                                     x_res=RES, y_res=RES, no_data=ND)
    _egal(m.composer_mhs_8bits(_couches(dem), elevation=35, ve=1), attendu, "M-HS 8 bits aux défauts")


def test_m_hs_8_bits_applique_l_elevation_reglee():
    """Changement voulu (0.16.0) : rvt-qgis ignorait l'élévation en 8 bits."""
    _, _, default, _ = _rvt()
    dem = _mnt()
    defaut = default.DefaultValues().float_to_8bit(float_arr=dem.copy(), visualization=default.RVTVisualization.MULTI_HILLSHADE,
                                                    x_res=RES, y_res=RES, no_data=ND)
    assert not np.array_equal(m.composer_mhs_8bits(_couches(dem), elevation=50, ve=1), defaut)


def test_m_hs_8_bits_partage_l_ombrage_du_hs():
    """Revue de la tâche 4 : aux mêmes réglages, le rouge du M-HS 8 bits (315°) est l'ombrage du HS par défaut."""
    c = _couches(_mnt())
    m.composer_mhs_8bits(c, elevation=35, ve=1)
    c.ombrage(315.0, 35.0, 1.0, RES, RES)   # clé de calculer_hs (azimut 315, élévation 35, VE 1)
    assert len(c._memo) == 3


if __name__ == "__main__":   # hors pytest : Pythons de QGIS (numpy 1.26 et 2.x)
    faits = []
    for nom, f in sorted(globals().items()):
        if nom.startswith("test_") and callable(f) and f.__code__.co_argcount == 0:
            try:
                f()
                faits.append(nom)
            except Exception as exc:  # noqa: BLE001 — le premier échec est écrit puis relevé
                Path(sys.argv[1]).write_text(f"ÉCHEC {nom} : {exc!r}\n", encoding="utf-8")
                raise
    Path(sys.argv[1]).write_text(f"OK {len(faits)} tests, numpy {np.__version__}, python {sys.version.split()[0]}\n"
                                 + "\n".join(faits) + "\n", encoding="utf-8")
