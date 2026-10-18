"""MSTP et M-HS (0.15.0) : leur image change à nom de fichier égal. Un intermédiaire calculé avant — sans note de
calcul, ou par Processing — est recalculé une fois ; à jour, il est réutilisé (Review Focus 2)."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import pipeline.ign.products.indices as ind
from pipeline.ign.products import moteur_indices
from pipeline.ign.products.rvt_naming import get_rvt_temp_filename

TILE = "LHD_FXX_0623_6864"


@pytest.fixture
def banc(tmp_path, monkeypatch):
    monkeypatch.delenv(moteur_indices.VARIABLE_MOTEUR, raising=False)
    mnt = tmp_path / f"{TILE}_MNT.tif"
    mnt.write_bytes(b"mnt")
    appels = []
    monkeypatch.setattr(moteur_indices.Couches, "lire", classmethod(lambda cls, chemin: SimpleNamespace(partage=True)))

    def espion(nom):
        def f(c, out, **k):
            appels.append((nom, k))
            Path(out).write_bytes(nom.encode())
        return f
    monkeypatch.setattr(moteur_indices, "calculer_mstp", espion("MSTP"))
    monkeypatch.setattr(moteur_indices, "calculer_mhs", espion("M_HS"))

    def processing(algo, params, **k):
        appels.append((algo, {}))
        Path(params["OUTPUT"]).write_bytes(b"p")
    monkeypatch.setattr(ind, "run_qgis_algorithm", processing)
    monkeypatch.setattr(ind, "reclass_rvt_nodata", lambda *a: None)
    return tmp_path, mnt, appels


def _intermediaire(tmp, mnt, produit, note=None):
    out = tmp / get_rvt_temp_filename(produit, TILE, {})
    out.write_bytes(b"ancien")
    t = mnt.stat().st_mtime + 10
    os.utime(out, (t, t))    # plus récent que son MNT : « à jour » au sens des dates
    if note is not None:
        Path(str(out) + ".calcul").write_text(note, encoding="utf-8")
    return out


def _lancer(tmp, produits):
    logs = []
    ind.create_visualization_products(temp_dir=tmp, current_tile_name=TILE, products={p: True for p in produits},
                                      rvt_params={}, log=logs.append)
    return logs


def test_un_mstp_d_une_version_precedente_est_recalcule(banc):
    tmp, mnt, appels = banc
    out = _intermediaire(tmp, mnt, "MSTP")
    _lancer(tmp, ["MSTP"])
    assert [a[0] for a in appels] == ["MSTP"]
    assert appels[0][1]["luminosite"] == 0.9   # le défaut (celui de la fenêtre de rvt-qgis), appliqué tel quel
    assert Path(str(out) + ".calcul").read_text(encoding="utf-8") == ind.VERSION_CALCUL["MSTP"]


def test_un_mstp_a_jour_est_reutilise(banc):
    tmp, mnt, appels = banc
    _intermediaire(tmp, mnt, "MSTP", note=ind.VERSION_CALCUL["MSTP"])
    logs = _lancer(tmp, ["MSTP"])
    assert appels == [] and any("MSTP réutilisé" in ligne for ligne in logs)


def test_m_hs_d_une_version_precedente_est_recalcule(banc):
    tmp, mnt, appels = banc
    _intermediaire(tmp, mnt, "M_HS", note="autre")
    _lancer(tmp, ["M_HS"])
    assert [a[0] for a in appels] == ["M_HS"]


def test_un_calcul_force_par_processing_ne_laisse_pas_de_note(banc, monkeypatch):
    tmp, mnt, appels = banc
    out = _intermediaire(tmp, mnt, "MSTP", note=ind.VERSION_CALCUL["MSTP"])
    os.utime(mnt, (out.stat().st_mtime + 10,) * 2)    # MNT plus récent : recalcul
    monkeypatch.setenv(moteur_indices.VARIABLE_MOTEUR, "processing")
    _lancer(tmp, ["MSTP"])
    assert [a[0] for a in appels] == ["rvt:rvt_mstp"]
    # Le secours ne recalcule plus à chaque relance : il laisse une note « processing », réutilisée tant qu'on y
    # reste, et le retour au moteur intégré recalcule une fois.
    assert Path(str(out) + ".calcul").read_text(encoding="utf-8") == ind.NOTE_PROCESSING
    os.utime(out, (mnt.stat().st_mtime + 10,) * 2)    # intermédiaire à jour de son MNT (daté dans le futur ci-dessus)
    logs = _lancer(tmp, ["MSTP"])
    assert [a[0] for a in appels] == ["rvt:rvt_mstp"] and any("MSTP réutilisé" in ligne for ligne in logs)
    monkeypatch.delenv(moteur_indices.VARIABLE_MOTEUR)
    _lancer(tmp, ["MSTP"])
    assert [a[0] for a in appels] == ["rvt:rvt_mstp", "MSTP"]
