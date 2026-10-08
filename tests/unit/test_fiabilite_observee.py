"""Fiabilité observée sur vos runs (2026-10-08) : registre des runs connus, lecture
sqlite des verdicts, agrégation par modèle / classe / niveau, phrases."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.services.fiabilite_observee import (
    MAX_RUNS,
    Compte,
    agreger,
    enregistrer_run,
    lire_verdicts_gpkg,
    observer,
    runs_connus,
)


def _gpkg(chemin: Path, couches: dict) -> Path:
    """Un faux GeoPackage : gpkg_contents + une table par couche, lignes
    (model_name, model_pred, fiabilite, validation, corr_pred)."""
    chemin.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(chemin)
    con.execute("CREATE TABLE gpkg_contents (table_name TEXT, data_type TEXT)")
    for nom, lignes in couches.items():
        con.execute("INSERT INTO gpkg_contents VALUES (?, 'features')", (nom,))
        con.execute(f'CREATE TABLE "{nom}" (fid INTEGER PRIMARY KEY, model_name TEXT, model_pred TEXT, '
                    'fiabilite TEXT, validation TEXT, corr_pred TEXT, confidence REAL)')
        con.executemany(f'INSERT INTO "{nom}" (model_name, model_pred, fiabilite, validation, corr_pred) VALUES (?,?,?,?,?)', lignes)
    con.commit()
    con.close()
    return chemin


def test_registre_ajoute_dedoublonne_et_plafonne(tmp_path):
    base = tmp_path / "profil"
    runs = [tmp_path / f"run{i}" for i in range(3)]
    for r in runs:
        r.mkdir()
    enregistrer_run(base, runs[0])
    enregistrer_run(base, runs[1])
    enregistrer_run(base, runs[0])                 # déjà là : remonte en fin
    assert runs_connus(base) == [runs[1], runs[0]]
    enregistrer_run(base, tmp_path / "disparu")   # dossier absent : filtré à la lecture
    assert runs_connus(base) == [runs[1], runs[0]]
    for i in range(MAX_RUNS + 5):
        d = tmp_path / f"beaucoup{i}"
        d.mkdir()
        enregistrer_run(base, d)
    assert len(json.loads((base / "runs_connus.json").read_text(encoding="utf-8"))) == MAX_RUNS
    (base / "runs_connus.json").write_text("{pas une liste", encoding="utf-8")
    assert runs_connus(base) == []
    enregistrer_run(base, runs[2])                 # registre illisible : repart de zéro
    assert runs_connus(base) == [runs[2]]


def test_lecture_et_agregation_des_verdicts(tmp_path):
    run = tmp_path / "run"
    _gpkg(run / "detections" / "parcellaire" / "parcellaire.gpkg", {"parcellaire": [
        ("lin_v3", "parcellaire", "Probable", "oui", ""),
        ("lin_v3", "parcellaire", "Probable", "oui", "parcellaire"),      # corrigé vers la même classe : vrai
        ("lin_v3", "parcellaire", "Probable", "oui", "chemin_creux"),     # corrigé ailleurs : faux pour parcellaire
        ("lin_v3", "parcellaire", "Probable", "non", ""),
        ("lin_v3", "parcellaire", "Probable", "peut-être", ""),
        ("lin_v3", "parcellaire", "Très probable", "oui", ""),
        ("lin_v3", "parcellaire", "Douteux", "", ""),                     # sans verdict : ignoré
        ("lin_v3", "parcellaire", "inconnu", "oui", ""),                  # niveau inconnu : ignoré
    ]})
    _gpkg(run / "detections" / "zone" / "zone.gpkg", {"zone_crateres": [
        ("crat", "zone_crateres", "", "oui", ""),                         # synthèse sans fiabilité : ignorée
    ]})
    _gpkg(run / "detections" / "legacy" / "legacy.gpkg", {"sans_colonnes": []})
    verdicts = list(lire_verdicts_gpkg(run / "detections" / "parcellaire" / "parcellaire.gpkg"))
    assert len(verdicts) == 7 and verdicts[0] == ("lin_v3", "parcellaire", "Probable", "oui", "")
    agg = agreger([run, tmp_path / "absent"])
    assert set(agg) == {("lin_v3", "parcellaire")}
    probable = agg[("lin_v3", "parcellaire")]["probable"]
    assert (probable.verifies, probable.vrais, probable.a_revoir) == (4, 2, 1)
    assert probable.part_vrais == 0.5
    assert agg[("lin_v3", "parcellaire")]["quasi_certain"].vrais == 1
    assert list(lire_verdicts_gpkg(tmp_path / "pas_un.gpkg")) == []


def test_observer_et_phrases(tmp_path):
    base = tmp_path / "profil"
    run = tmp_path / "run"
    lignes = [("m", "four", "Probable", "oui", "")] * 60 + [("m", "four", "Probable", "non", "")] * 25
    lignes += [("m", "four", "Douteux", "oui", "")] * 3 + [("m", "four", "Douteux", "non", "")] + [("m", "four", "Douteux", "peut-être", "")] * 2
    _gpkg(run / "detections" / "fours" / "fours.gpkg", {"four": lignes})
    assert observer(base, "m", "four").n_runs == 0           # pas de registre
    enregistrer_run(base, run)
    obs = observer(base, "m", "four")
    assert obs.n_runs == 1 and obs.total_verifies == 89 and obs.total_a_revoir == 2
    assert obs.par_categorie["probable"].phrase() == "71 % sur vos 85 vérifications"
    assert obs.par_categorie["douteux"].phrase() == "3 vraies sur 4 vérifiées, 2 à revoir"
    assert observer(base, "autre", "four").par_categorie == {}
    assert Compte().phrase() == "" and Compte(a_revoir=2).phrase() == "2 à revoir, aucune vérifiée"
    assert Compte(verifies=1, vrais=1).phrase() == "1 vraie sur 1 vérifiée"


def test_branchements_ui():
    """Garde-fou sans QGIS : le run est enregistré au lancement, la fiche reçoit
    l'observation, les couches live ont le même vocabulaire de verdict que le .qgs."""
    racine = Path(__file__).resolve().parents[2]
    vue = (racine / "src/ui/run_view.py").read_text(encoding="utf-8")
    assert "enregistrer_run(" in vue
    etape3 = (racine / "src/ui/steps/step_3_detection.py").read_text(encoding="utf-8")
    assert "observer(" in etape3 and "observes=observes" in etape3
    fiche = (racine / "src/ui/dialogs/class_info_dialog.py").read_text(encoding="utf-8")
    assert "chez vous" in fiche
    loader = (racine / "src/ui/layer_loader.py").read_text(encoding="utf-8")
    assert '{"peut-être": "peut-être"}' in loader
