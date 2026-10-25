"""« Revoir un résultat » (lot 9, 2026-10-24) : les traitements connus de ce poste, lus
dans le registre ``runs_connus.json`` puis dans ``livrable/traitement.json`` de chaque
dossier ; la configuration d'un traitement dans ``technique/journaux/metadata_<ts>.json`` ;
les chemins du livrable à recharger. Pur, sans QGIS."""
from __future__ import annotations

import json
from pathlib import Path

from app.services.fiabilite_observee import enregistrer_run
from app.services.resultats_connus import (
    Resultat,
    chemins_du_livrable,
    config_du_resultat,
    lister,
)


def _dossier(tmp_path: Path, nom: str, date: str, *, metadata: bool = True, trace: bool = True) -> Path:
    d = tmp_path / nom
    (d / "livrable" / "indices" / "SVF_R10" / "tif").mkdir(parents=True)
    (d / "livrable" / "indices" / "SVF_R10" / "tif" / "index_SVF.vrt").write_text("<VRT/>", encoding="utf-8")
    (d / "livrable" / "detections" / "parcellaire").mkdir(parents=True)
    (d / "livrable" / "detections" / "parcellaire" / "parcellaire.gpkg").write_bytes(b"")
    (d / "livrable" / "projet.qgs").write_text("<qgis/>", encoding="utf-8")
    (d / "livrable" / "rapport.html").write_text("<html/>", encoding="utf-8")
    if trace:
        (d / "livrable" / "traitement.json").write_text(json.dumps({
            "date": date, "active_products": ["MNT", "SVF"],
            "detections_entities": [{"entity": "parcellaire", "label": "Parcellaire", "slug": "parcellaire"}],
            "bilan_fiabilite": [{"label": "Parcellaire", "total": 103}],
            "rapport": "livrable/rapport.html",
        }), encoding="utf-8")
    if metadata:
        j = d / "technique" / "journaux"
        j.mkdir(parents=True)
        (j / "metadata_20261001_1200.json").write_text(json.dumps({"date": "ancien", "ui_config": {"app": {"files": {"data_mode": "local_laz"}}}}), encoding="utf-8")
        (j / "metadata_20261024_1200.json").write_text(json.dumps({
            "date": date,
            "ui_config": {"app": {"files": {"data_mode": "ign_laz", "output_dir": str(d)}},
                          "computer_vision": {"enabled": True, "runs": []}},
        }), encoding="utf-8")
    return d


def test_lister_trie_par_date(tmp_path):
    profil = tmp_path / "profil"
    ancien = _dossier(tmp_path, "ancien", "2026-10-20T10:00:00")
    recent = _dossier(tmp_path, "recent", "2026-10-24T09:30:00")
    enregistrer_run(profil, recent)
    enregistrer_run(profil, ancien)       # ordre du registre ≠ ordre des dates
    r = lister(profil)
    assert [x.dossier for x in r] == [recent, ancien]
    premier = r[0]
    assert premier.date == "2026-10-24T09:30:00" and premier.mode == "Téléchargement IGN"
    assert premier.images == ("MNT", "SVF") and premier.structures == ("Parcellaire",)
    assert premier.n_detections == 103
    assert premier.projet == recent / "livrable" / "projet.qgs" and premier.rapport == recent / "livrable" / "rapport.html"
    assert premier.metadata == recent / "technique" / "journaux" / "metadata_20261024_1200.json"


def test_dossier_disparu_ou_sans_trace_ignore(tmp_path):
    profil = tmp_path / "profil"
    bon = _dossier(tmp_path, "bon", "2026-10-24T09:30:00")
    sans_trace = _dossier(tmp_path, "sans_trace", "2026-10-25T09:30:00", trace=False)
    disparu = tmp_path / "disparu"
    disparu.mkdir()
    for d in (bon, sans_trace, disparu):
        enregistrer_run(profil, d)
    import shutil
    shutil.rmtree(disparu)
    assert [x.dossier for x in lister(profil)] == [bon]
    assert lister(tmp_path / "profil_vide") == []


def test_config_du_resultat_lit_ui_config(tmp_path):
    d = _dossier(tmp_path, "r", "2026-10-24T09:30:00")
    enregistrer_run(tmp_path / "profil", d)
    r = lister(tmp_path / "profil")[0]
    cfg = config_du_resultat(r)
    assert cfg["app"]["files"]["data_mode"] == "ign_laz" and cfg["computer_vision"]["enabled"] is True


def test_sans_metadata_renvoie_none(tmp_path):
    d = _dossier(tmp_path, "r", "2026-10-24T09:30:00", metadata=False)
    enregistrer_run(tmp_path / "profil", d)
    r = lister(tmp_path / "profil")[0]
    assert r.metadata is None and r.mode == "" and config_du_resultat(r) is None
    illisible = Resultat(d, "d", "", (), (), 0, d / "x.qgs", d / "x.html", d / "absent.json")
    assert config_du_resultat(illisible) is None


def test_chemins_du_livrable(tmp_path):
    d = _dossier(tmp_path, "r", "2026-10-24T09:30:00")
    (d / "livrable" / "indices" / "COUVERTURE").mkdir(parents=True)
    (d / "livrable" / "indices" / "COUVERTURE" / "zones_mal_couvertes.gpkg").write_bytes(b"")
    r = Resultat(d, "d", "", (), (), 0, d / "livrable" / "projet.qgs", d / "livrable" / "rapport.html", None)
    vrt, gpkg, qa = chemins_du_livrable(r)
    assert vrt == [str(d / "livrable" / "indices" / "SVF_R10" / "tif" / "index_SVF.vrt")]
    assert gpkg == [str(d / "livrable" / "detections" / "parcellaire" / "parcellaire.gpkg")]
    assert qa == [str(d / "livrable" / "indices" / "COUVERTURE" / "zones_mal_couvertes.gpkg")]


def test_trace_avec_app_nul_ne_vide_pas_la_liste(tmp_path):
    """Une trace dont ``ui_config.app`` vaut explicitement null ne doit pas faire tomber
    toute la liste : le point de départ est simplement inconnu."""
    d = _dossier(tmp_path, "r", "2026-10-24T09:30:00")
    (d / "technique" / "journaux" / "metadata_20261025_0000.json").write_text(
        json.dumps({"ui_config": {"app": None}}), encoding="utf-8")
    enregistrer_run(tmp_path / "profil", d)
    r = lister(tmp_path / "profil")
    assert len(r) == 1 and r[0].mode == ""
