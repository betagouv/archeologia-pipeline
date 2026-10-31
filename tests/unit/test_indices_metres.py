"""Réglages spatiaux des indices en mètres (règle 2026-10-31) : conversion, migration, anneau des modèles."""
from __future__ import annotations

import copy

import pytest

from app.services import indices_metres as im
from config.config_manager import ConfigManager


def _defauts():
    return ConfigManager.__new__(ConfigManager).default_config()["rvt_params"]


@pytest.mark.parametrize("m,res,px", [
    (5, 0.5, 10), (10, 0.5, 20), (5, 1.0, 5), (10, 1.0, 10), (5, 0.25, 20), (10, 0.25, 40),
    (1.5, 0.5, 3), (101.5, 0.5, 203), (2.5, 1.0, 3),   # demi vers le haut, pas l'arrondi bancaire
    (0.3, 0.1, 3), (0.1, 1.0, 1), (0.0, 1.0, 1),       # bruit flottant absorbé ; jamais moins d'un pixel
])
def test_px_de_m(m, res, px):
    assert im.px_de_m(m, res) == px


def test_defauts_metres_redonnent_les_anciens_pixels_a_0_5m():
    """La France à 0,5 m reste au bit près : les défauts en m redonnent les défauts d'avant en px."""
    px = im.en_px(_defauts(), 0.5)
    assert (px["svf"]["radius"], px["opns"]["radius"], px["slrm"]["radius"]) == (10, 10, 20)
    assert (px["ldo"]["min_radius"], px["ldo"]["max_radius"]) == (10, 20)
    assert [px["mstp"][f"{e}_scale_{k}"] for e in ("local", "meso", "broad") for k in ("min", "max", "step")] == [
        3, 21, 2, 23, 203, 18, 100, 400, 60]
    assert px["resolution_m"] == 0.5


def test_defauts_sans_cle_pixel():
    """La configuration par défaut ne porte plus de rayon en pixels."""
    d = _defauts()
    for _p, section, cle_px, cle_m, defaut, _l in im.REGLAGES_SPATIAUX:
        assert cle_px not in d[section]
        assert d[section][cle_m] == defaut


@pytest.mark.parametrize("res,ld", [(1.0, (5, 10)), (0.25, (20, 40))])
def test_anneau_ld_physique_constant(res, ld):
    px = im.en_px(_defauts(), res)
    assert (px["ldo"]["min_radius"], px["ldo"]["max_radius"]) == ld


def test_en_px_ne_modifie_pas_l_entree_et_est_idempotent():
    d = _defauts()
    avant = copy.deepcopy(d)
    une = im.en_px(d, 1.0)
    assert d == avant
    assert im.en_px(une, 1.0) == une


def test_migration_px_vers_m_une_seule_fois():
    ancien = {"ldo": {"min_radius": 10, "max_radius": 30, "angular_res": 15}, "svf": {"radius": "12"},
              "mstp": {"broad_scale_max": 300}}
    msgs = im.migrer_px_vers_m(ancien)
    assert ancien["ldo"] == {"min_radius_m": 5.0, "max_radius_m": 15.0, "angular_res": 15}
    assert ancien["svf"] == {"radius_m": 6.0}
    assert ancien["mstp"] == {"broad_scale_max_m": 150.0}
    assert len(msgs) == 4 and "0,5 m/px supposé" in ancien["migration_px_vers_m"]
    assert "30 px" in msgs[2] and "15 m" in msgs[2]
    # Une seconde fois : rien ne bouge, rien n'est dit.
    deja = copy.deepcopy(ancien)
    assert im.migrer_px_vers_m(ancien) == [] and ancien == deja


def test_cle_metrique_gagne_sur_une_cle_pixel_residuelle():
    """Une clé pixel laissée par une ancienne version à côté de la clé métrique n'est jamais relue."""
    p = {"ldo": {"min_radius_m": 5.0, "min_radius": 99}}
    assert im.migrer_px_vers_m(p) == []
    assert p["ldo"] == {"min_radius_m": 5.0}


def test_en_px_migre_une_config_en_pixels():
    """Un appelant qui reçoit encore une config en pixels (headless, ancien fichier) n'est pas trompé."""
    px = im.en_px({"ldo": {"min_radius": 10, "max_radius": 20}}, 1.0)
    assert (px["ldo"]["min_radius"], px["ldo"]["max_radius"]) == (5, 10)


def test_format():
    assert im.format_m(10.5) == "10,5 m"
    assert im.format_m_px(5, 1.0) == "5 m (5 px à 1 m/px)"
    assert im.format_m_px(5, 0.5) == "5 m (10 px à 0,5 m/px)"


@pytest.mark.parametrize("params,tp,attendu", [
    ({"rmin_px": 10, "rmax_px": 20, "gsd_m": 0.5}, None, (5.0, 10.0)),
    ({"min_radius": 5, "max_radius": 10}, {"mnt": {"resolution": 1.0}}, (5.0, 10.0)),
    ({"rmin_m": 5, "rmax_m": 10, "gsd_m": 0.5, "rmin_px": 10, "rmax_px": 20}, None, (5.0, 10.0)),
    ({"min_radius": 5, "max_radius": 10}, None, None),   # pixels sans résolution : indécidable
])
def test_anneau_des_modeles(params, tp, attendu):
    assert im.anneau_m(params, im.gsd_de_fiche(params, tp)) == attendu


def test_config_manager_migre_un_fichier_en_pixels(tmp_path):
    """La valeur enregistrée en pixels survit à la fusion avec les défauts (qui portent déjà les clés m)."""
    import json
    (tmp_path / "last_ui_config.json").write_text(json.dumps(
        {"rvt_params": {"ldo": {"min_radius": 12, "max_radius": 30}}}), encoding="utf-8")
    cm = ConfigManager(tmp_path, settings_dir=tmp_path)
    cfg = cm.load_last_ui_config()
    assert cfg["rvt_params"]["ldo"]["min_radius_m"] == 6.0
    assert cfg["rvt_params"]["ldo"]["max_radius_m"] == 15.0
    assert "min_radius" not in cfg["rvt_params"]["ldo"]
    assert len(cm.migrations) == 2
    assert cfg["rvt_params"]["svf"]["radius_m"] == 5.0   # défaut intact


def _raster(chemin, pas):
    rasterio = pytest.importorskip("rasterio")
    import numpy as np
    from rasterio.transform import from_origin
    with rasterio.open(chemin, "w", driver="GTiff", width=20, height=20, count=1, dtype="float32",
                       crs="EPSG:25832", transform=from_origin(500000, 6000000, pas, pas)) as ds:
        ds.write(np.ones((1, 20, 20), dtype="float32"))


def test_rvt_params_du_run_a_la_resolution_des_mnt_fournis(tmp_path):
    """MNT allemand à 1 m : l'anneau 5–10 m devient 5–10 px (et non 10–20 px de l'IGN)."""
    from app.run_context import build_run_context, resolution_du_run, rvt_params_du_run
    from pipeline.ingest_plan import resolution_des_entrees
    _raster(tmp_path / "dgm1.tif", 1.0)
    assert resolution_des_entrees(tmp_path, ("tif",)) == 1.0
    assert resolution_des_entrees(tmp_path / "absent", ("tif",)) == 0.5
    ctx = build_run_context({"app": {"files": {"data_mode": "existing_mnt", "existing_mnt_dir": str(tmp_path),
                                               "output_dir": str(tmp_path / "out")}}, "rvt_params": _defauts()})
    assert resolution_du_run(ctx) == 1.0
    px = rvt_params_du_run(ctx)
    assert (px["ldo"]["min_radius"], px["ldo"]["max_radius"], px["resolution_m"]) == (5, 10, 1.0)
    ign = build_run_context({"processing": {"mnt_resolution": 0.5}, "rvt_params": _defauts()})
    assert rvt_params_du_run(ign)["ldo"]["max_radius"] == 20


def test_fusion_defauts_puis_migration_ne_perd_pas_la_valeur_en_pixels(tmp_path):
    """Audit 2026-10-31 (bloquant) : _apply_config de l'assistant fusionnait les défauts (déjà en mètres) AVANT la
    migration, qui voyait la clé _m et jetait la valeur enregistrée. La fusion passe par ConfigManager.fusionner."""
    cm = ConfigManager(tmp_path, settings_dir=tmp_path)
    charge = {"rvt_params": {"svf": {"radius": 14}, "ldo": {"min_radius": 6, "max_radius": 30}}}
    avant = copy.deepcopy(charge)
    cfg = cm.fusionner(charge)
    assert charge == avant                                    # le dict chargé n'est pas modifié
    assert cfg["rvt_params"]["svf"]["radius_m"] == 7.0
    assert (cfg["rvt_params"]["ldo"]["min_radius_m"], cfg["rvt_params"]["ldo"]["max_radius_m"]) == (3.0, 15.0)
    assert "radius" not in cfg["rvt_params"]["svf"]
    assert len(cm.migrations) == 3


def test_l_assistant_fusionne_par_config_manager():
    """Aucun autre point de fusion défauts + chargé que ConfigManager.fusionner dans src/ (lu en source)."""
    from pathlib import Path
    racine = Path(__file__).resolve().parents[2] / "src"
    fautifs = [str(p.relative_to(racine)) for p in racine.rglob("*.py")
               if "_deep_update(" in p.read_text(encoding="utf-8") and p.name != "config_manager.py"]
    assert fautifs == []


def test_resolution_des_entrees_ecarte_l_invraisemblable(tmp_path):
    """Audit 10 : un raster en degrés (pas 1e-5) n'a pas de résolution métrique : défaut, pas 1e-5 m."""
    from pipeline.ingest_plan import resolution_des_entrees
    _raster(tmp_path / "deg.tif", 1e-5)
    assert resolution_des_entrees(tmp_path, ("tif",)) == 0.5
