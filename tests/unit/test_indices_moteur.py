"""create_visualization_products (0.15.0) : chaque produit par le moteur intégré, ou par Processing quand
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


@pytest.mark.parametrize("produit,fonction,algo,attendu", [
    ("HS", "calculer_hs", "rvt:rvt_hillshade", {"azimut": 315, "elevation": 35, "ve": 1, "huit_bits": True}),
    ("SVF", "calculer_svf", "rvt:rvt_svf", {"rayon": 10, "directions": 16, "bruit": 0, "ve": 1, "huit_bits": True}),
    ("OPNS", "calculer_opns", "rvt:rvt_opns",
     {"negative": False, "rayon": 10, "directions": 16, "bruit": 0, "ve": 1, "huit_bits": True}),
    ("SLO", "calculer_slo", "rvt:rvt_slope", {"unite": 0, "ve": 1, "huit_bits": True}),
    ("SLRM", "calculer_slrm", "rvt:rvt_slrm", {"rayon": 20, "ve": 1, "huit_bits": True}),
])
def test_produit_seul_par_le_moteur_ou_force_par_processing(banc, monkeypatch, produit, fonction, algo, attendu):
    tmp, appels, brancher = banc
    brancher(fonction)
    monkeypatch.delenv(moteur_indices.VARIABLE_MOTEUR, raising=False)
    logs = _lancer(tmp, [produit])
    assert appels["moteur"] == [(fonction, attendu)] and appels["processing"] == []
    assert any(ligne.startswith(f"[PARAMS RVT/{produit}]") and "moteur=interne" in ligne for ligne in logs)
    for p in tmp.glob(f"{TILE}_*"):
        if not p.name.endswith("_MNT.tif"):
            p.unlink()
    monkeypatch.setenv(moteur_indices.VARIABLE_MOTEUR, "processing")
    _lancer(tmp, [produit])
    assert appels["processing"] == [algo]


@pytest.mark.parametrize("produit,indice,terrain,base", [("VAT", 0, 1, "_VAT_T1_B0_outputs"),
                                                         ("PRISM", 1, 2, "_PRISM_T2_outputs")])
def test_melange_par_le_moteur_ou_force_par_processing(banc, monkeypatch, produit, indice, terrain, base):
    tmp, appels, _ = banc
    rendus = []

    def rendre(c, prod, ind_, ter, sortie_base, huit_bits):
        rendus.append((prod, ind_, ter, sortie_base.name, huit_bits))
        Path(str(sortie_base) + "_8bit.tif").write_bytes(b"r")
    monkeypatch.setattr(moteur_indices, "rendre_melange", rendre)
    monkeypatch.delenv(moteur_indices.VARIABLE_MOTEUR, raising=False)
    section = {"VAT": "vat", "PRISM": "prism"}[produit]
    logs = _lancer(tmp, [produit], {section: {"terrain_type": terrain}})
    assert rendus == [(produit, indice, terrain, f"{TILE}{base}", True)] and appels["processing"] == []
    assert any(ligne.startswith(f"[PARAMS RVT/{produit}]") and "moteur=interne" in ligne for ligne in logs)
    assert (tmp / f"{TILE}{base[:-len('_outputs')]}.tif").read_bytes() == b"r"   # copié comme avant
    for p in tmp.glob(f"{TILE}_*"):
        if not p.name.endswith("_MNT.tif"):
            p.unlink()
    monkeypatch.setenv(moteur_indices.VARIABLE_MOTEUR, "processing")
    _lancer(tmp, [produit], {section: {"terrain_type": terrain}})
    assert appels["processing"] == ["rvt:rvt_blender"]


def test_cvat_et_crim_recoivent_les_couches_de_la_dalle(banc, monkeypatch):
    tmp, _, _ = banc
    recus = []
    monkeypatch.setattr(ind, "compute_cvat", lambda **k: recus.append(("CVAT", k["couches"]())))
    monkeypatch.setattr(ind, "compute_crim", lambda **k: recus.append(("CRIM", k["couches"]())))
    _lancer(tmp, ["CVAT", "CRIM"])
    assert [r[0] for r in recus] == ["CVAT", "CRIM"] and recus[0][1] is recus[1][1]   # le même MNT, lu une fois


def test_cvat_et_crim_sautes_si_le_mnt_ne_se_charge_pas(banc, monkeypatch):
    """Contrat du préflight (« CVAT et CRIM seuls ne bloquent pas ») : bibliothèque ou MNT en défaut → les deux
    indices sont sautés avec un message, la dalle continue — la lecture du MNT reste dans leur try."""
    tmp, _, _ = banc

    def lire(cls, chemin):
        raise ModuleNotFoundError("No module named 'scipy'")
    monkeypatch.setattr(moteur_indices.Couches, "lire", classmethod(lire))
    logs = _lancer(tmp, ["CVAT", "CRIM"])
    assert any(ligne.startswith("CVAT : échec") and "scipy" in ligne for ligne in logs)
    assert any(ligne.startswith("CRIM : échec") and "scipy" in ligne for ligne in logs)
