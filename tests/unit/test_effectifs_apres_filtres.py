"""Effectifs du bilan (fiabilite.json) recomptés après les filtres d'aire.

La conversion écrit les effectifs par niveau au moment où elle écrit les détections ; le filtre d'aire du
run puis celui de chaque entité (fe19017) passent après. Sans recomptage, bilan et rapport annonçaient
44 parcellaires quand le GeoPackage n'en gardait que 31. Un GeoPackage vidé puis supprimé laissait son
sidecar : le bilan comptait une couche qui n'existe plus.
"""
import json
import sqlite3

from app.services.fiabilite import recompter_effectifs, write_sidecar


def _gpkg(path, couches):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE gpkg_contents (table_name TEXT, data_type TEXT)")
    for nom, niveaux in couches.items():
        con.execute("INSERT INTO gpkg_contents VALUES (?, 'features')", (nom,))
        con.execute(f'CREATE TABLE "{nom}" (fid INTEGER PRIMARY KEY, fiabilite TEXT)')
        con.executemany(f'INSERT INTO "{nom}" (fiabilite) VALUES (?)', [(n,) for n in niveaux])
    con.commit()
    con.close()


def _entree(effectifs):
    return {"classe": "parcellaire", "modele": "m", "provenance": "", "categories": [], "effectifs": effectifs}


def test_effectifs_recomptes_sur_le_geopackage(tmp_path):
    gp = tmp_path / "parcellaire" / "parcellaire.gpkg"
    gp.parent.mkdir()
    write_sidecar(gp, "Parcellaire", _entree({"probable": 30, "douteux": 14}))
    write_sidecar(gp, "Talus", _entree({"possible": 5}))
    _gpkg(gp, {"Parcellaire": ["probable"] * 25 + ["douteux"] * 6})
    recompter_effectifs(gp)
    data = json.loads((gp.parent / "fiabilite.json").read_text(encoding="utf-8"))
    assert data["Parcellaire"]["effectifs"] == {"probable": 25, "douteux": 6}
    assert data["Parcellaire"]["classe"] == "parcellaire"
    assert data["Talus"]["effectifs"] == {}   # couche absente du GeoPackage : plus rien à compter


def test_sidecar_d_un_geopackage_supprime_retire(tmp_path):
    gp = tmp_path / "parcellaire" / "parcellaire.gpkg"
    gp.parent.mkdir()
    write_sidecar(gp, "Parcellaire", _entree({"probable": 3}))
    recompter_effectifs(gp)
    assert not (gp.parent / "fiabilite.json").exists()


def test_recomptage_de_chaque_entite_du_run(tmp_path):
    from pipeline.modes.existing_rvt import _recompter_effectifs
    from pipeline.output_paths import detection_entity_dir

    gp = detection_entity_dir(tmp_path, "parcellaire") / "parcellaire.gpkg"
    gp.parent.mkdir(parents=True)
    write_sidecar(gp, "parcellaire", _entree({"probable": 44}))
    _gpkg(gp, {"parcellaire": ["probable"] * 31})
    vide = detection_entity_dir(tmp_path, "chemins") / "chemins.gpkg"
    vide.parent.mkdir(parents=True)
    write_sidecar(vide, "chemin", _entree({"possible": 2}))
    run = {"entities": [{"slug": "parcellaire", "classes": ["parcellaire"]},
                        {"slug": "chemins", "classes": ["chemin"]}]}
    _recompter_effectifs(tmp_path, run)
    data = json.loads((gp.parent / "fiabilite.json").read_text(encoding="utf-8"))
    assert data["parcellaire"]["effectifs"] == {"probable": 31}
    assert not (vide.parent / "fiabilite.json").exists()
