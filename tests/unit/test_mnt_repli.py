"""Repli du MNT (recette séance B, 2026-10-21).

Sur la dalle LHD_FXX_0727_6569, ``pdal:exportrastertin`` a planté (« le processus s'est arrêté de façon
inattendue », mémoire sous charge) et le MNT a été calculé par ``pdal:exportraster``, sans interpolation :
0,05 % de pixels vides là où les huit autres dalles en ont 0, sans rien dire à l'utilisateur, et une relance
resservait ce MNT du cache.

Tenu ici : une seconde tentative TIN avant le repli ; un repli marqué (résultat + note à côté du MNT) ; une
relance qui recalcule un MNT de repli au lieu de le réutiliser ; la note retirée dès qu'un calcul TIN réussit.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.ign.products import mnt


@pytest.fixture
def banc(tmp_path, monkeypatch):
    laz = tmp_path / "dalle_merged.laz"
    laz.write_bytes(b"laz")
    appels: list = []
    echecs = {"tin": 0}

    def faux_algo(algo, parametres, **_k):
        appels.append(algo)
        if algo == "pdal:exportrastertin" and echecs["tin"] > 0:
            echecs["tin"] -= 1
            raise RuntimeError("Le processus s'est arrêté de façon inattendue")
        Path(parametres["OUTPUT"]).write_bytes(b"tif")

    monkeypatch.setattr(mnt, "_run_under_watchdog", faux_algo)
    monkeypatch.setattr(mnt, "validate_las_or_laz_with_pdal", lambda p: (True, ""))
    monkeypatch.setattr(mnt, "_bornes_du_nuage", lambda p: None)
    monkeypatch.setattr(mnt, "assign_crs_if_missing", lambda p: None)

    def calculer():
        return mnt.create_terrain_model(
            input_laz_path=laz, temp_dir=tmp_path / "inter", current_tile_name="dalle",
            mnt_resolution=0.5, tile_overlap_percent=20, filter_expression="Classification = 2",
        )

    return calculer, appels, echecs


def test_un_plantage_isole_est_rattrape_par_une_seconde_tentative(banc):
    calculer, appels, echecs = banc
    echecs["tin"] = 1
    res = calculer()
    assert appels == ["pdal:exportrastertin", "pdal:exportrastertin"]
    assert res.repli is False
    assert not mnt.note_repli_path(res.mnt_path).exists()


def test_deux_plantages_basculent_sur_le_repli_et_le_disent(banc):
    calculer, appels, echecs = banc
    echecs["tin"] = 2
    res = calculer()
    assert appels == ["pdal:exportrastertin", "pdal:exportrastertin", "pdal:exportraster"]
    assert res.repli is True
    assert mnt.note_repli_path(res.mnt_path).exists()


def test_une_relance_recalcule_un_mnt_de_repli_puis_retire_la_note(banc):
    calculer, appels, echecs = banc
    echecs["tin"] = 2
    calculer()
    appels.clear()
    res = calculer()                      # le MNT de repli est plus récent que le LAZ : sans la note, réutilisé
    assert appels == ["pdal:exportrastertin"]
    assert res.repli is False
    assert not mnt.note_repli_path(res.mnt_path).exists()


def test_un_mnt_sain_reste_reutilise(banc):
    calculer, appels, _ = banc
    calculer()
    appels.clear()
    calculer()
    assert appels == []
