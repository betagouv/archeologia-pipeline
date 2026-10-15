"""create_visualization_products (0.16.0) : chaque produit par le moteur intégré, ou par Processing quand
ARCHEO_INDICES_MOTEUR=processing ; la ligne [PARAMS RVT/<P>] du journal dit lequel."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import pipeline.ign.products.indices as ind
from pipeline.ign.products import moteur_indices

TILE = "LHD_FXX_0623_6864"


@pytest.fixture
def banc(tmp_path, monkeypatch):
    """MNT factice ; Processing, lecture du MNT et calculs du moteur remplacés par des espions."""
    (tmp_path / f"{TILE}_MNT.tif").write_bytes(b"mnt")
    appels = {"processing": [], "moteur": []}
    monkeypatch.setattr(ind, "run_qgis_algorithm", lambda algo, params, **k: appels["processing"].append(algo))
    monkeypatch.setattr(moteur_indices.Couches, "lire",
                        classmethod(lambda cls, chemin: SimpleNamespace(chemin=chemin, partage=True)))
    monkeypatch.setattr(ind, "reclass_rvt_nodata", lambda *a: None)

    def brancher(*fonctions):
        for f in fonctions:
            def espion(c, out, _f=f, **k):
                appels["moteur"].append((_f, k))
                Path(out).write_bytes(b"x")
            monkeypatch.setattr(moteur_indices, f, espion)
    return tmp_path, appels, brancher


def _lancer(tmp, produits, rvt_params=None):
    logs = []
    ind.create_visualization_products(temp_dir=tmp, current_tile_name=TILE, products={p: True for p in produits},
                                      rvt_params=rvt_params or {}, log=logs.append)
    return logs


def test_ld_par_le_moteur(banc, monkeypatch):
    tmp, appels, brancher = banc
    brancher("calculer_ld")
    monkeypatch.delenv(moteur_indices.VARIABLE_MOTEUR, raising=False)
    logs = _lancer(tmp, ["LD"], {"ldo": {"angular_res": 10, "min_radius": 5, "max_radius": 30, "observer_h": 3.5,
                                         "ve_factor": 2}})
    assert appels["processing"] == []
    assert appels["moteur"] == [("calculer_ld", {"angular_res": 10, "min_radius": 5, "max_radius": 30,
                                                 "observer_h": 3.5, "ve": 2, "huit_bits": True})]
    assert any(ligne.startswith("[PARAMS RVT/LD]") and "moteur=interne" in ligne for ligne in logs)


def test_ld_force_par_processing(banc, monkeypatch):
    tmp, appels, brancher = banc
    brancher("calculer_ld")
    monkeypatch.setenv(moteur_indices.VARIABLE_MOTEUR, "processing")
    logs = _lancer(tmp, ["LD"])
    assert appels["processing"] == ["rvt:rvt_ld"] and appels["moteur"] == []
    assert any("moteur=processing" in ligne for ligne in logs)


def test_ld_hors_bornes_fait_echouer_la_dalle_comme_processing(banc, monkeypatch):
    tmp, appels, _ = banc
    monkeypatch.delenv(moteur_indices.VARIABLE_MOTEUR, raising=False)
    monkeypatch.setattr(moteur_indices, "dominance_locale", lambda *a, **k: pytest.fail("calcul lancé hors bornes"))
    with pytest.raises(ValueError, match="angular_res=45"):
        _lancer(tmp, ["LD"], {"ldo": {"angular_res": 45}})
