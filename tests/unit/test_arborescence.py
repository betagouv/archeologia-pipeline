"""Arborescence v3 du dossier de sortie (2026-10-08) : détection, plan, application de la migration."""
from __future__ import annotations

from pathlib import Path

from app.services.arborescence import LIVRABLE, appliquer, decrire, etat, non_reconnus, plan_migration
from pipeline import output_paths as op


def _fichier(p: Path, contenu: bytes = b"x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(contenu)
    return p


def _ancien_dossier(racine: Path) -> None:
    _fichier(racine / "indices" / "MNT" / "tif" / "a.tif")
    _fichier(racine / "indices" / "MNT" / "tif" / "index_MNT.vrt")
    _fichier(racine / "indices" / "LD_X" / "tif" / "a.tif")
    _fichier(racine / "indices" / "LD_X" / "png" / "a.png")
    _fichier(racine / "detections" / "parcellaire" / "parcellaire.gpkg")
    _fichier(racine / "detections" / "detections_validation.qgs")
    _fichier(racine / "detections" / "_technique" / "m" / "raw_detections" / "a.json")
    _fichier(racine / "intermediaires" / "halo" / "LD_X" / "a.tif")
    _fichier(racine / "sources" / "dalles" / "a.laz")
    _fichier(racine / "dalles_urls.txt")
    _fichier(racine / "pipeline_log_20260909_120000.txt")
    _fichier(racine / "pipeline_log_20261008_161911.txt")
    _fichier(racine / "metadata.json", b"{}")
    _fichier(racine / "rapport.html")
    _fichier(racine / "rapport_vignette.png")
    _fichier(racine / "MNT" / "tif" / "vieux.tif")          # toute première arborescence : inconnue
    _fichier(racine / "fichier_tri.txt")                    # posé par l'utilisateur


def test_etat_vide_ancienne_nouvelle(tmp_path):
    assert etat(tmp_path / "absent") == "vide" and etat(tmp_path) == "vide"
    _fichier(tmp_path / "notes.txt")
    assert etat(tmp_path) == "vide"                          # rien de connu
    _fichier(tmp_path / "pipeline_log_20261008_1.txt")
    assert etat(tmp_path) == "ancienne"
    (tmp_path / LIVRABLE).mkdir()
    assert etat(tmp_path) == "nouvelle"
    assert plan_migration(tmp_path / "absent") == [] and non_reconnus(tmp_path / "absent", []) == []


def test_plan_puis_application(tmp_path):
    _ancien_dossier(tmp_path)
    plan = plan_migration(tmp_path)
    rel = [(d.source.relative_to(tmp_path).as_posix(), d.destination.relative_to(tmp_path).as_posix()) for d in plan]
    assert rel == [
        ("indices/LD_X/png", "technique/png/LD_X"),
        ("indices", "livrable/indices"),
        ("detections/_technique/m", "technique/detection/m"),
        ("detections", "livrable/detections"),
        ("intermediaires", "technique/intermediaires"),
        ("sources", "technique/sources"),
        ("dalles_urls.txt", "technique/sources/dalles_urls.txt"),
        ("pipeline_log_20260909_120000.txt", "technique/journaux/pipeline_log_20260909_120000.txt"),
        ("pipeline_log_20261008_161911.txt", "technique/journaux/pipeline_log_20261008_161911.txt"),
        ("metadata.json", "technique/journaux/metadata_20261008_161911.json"),
        ("rapport.html", "livrable/rapport.html"),
        ("rapport_vignette.png", "livrable/rapport_vignette.png"),
    ]
    assert non_reconnus(tmp_path, plan) == ["MNT", "fichier_tri.txt"]
    texte = decrire(plan, tmp_path, non_reconnus(tmp_path, plan))
    assert "livrable/" in texte and "technique/" in texte and "12 déplacements" in texte
    assert "indices  →  livrable/indices" in texte and "MNT, fichier_tri.txt" in texte and "… et 2 à l'intérieur" in texte

    journal = []
    assert appliquer(plan, log=journal.append) == []
    assert (tmp_path / "livrable" / "indices" / "MNT" / "tif" / "index_MNT.vrt").is_file()
    assert (tmp_path / "technique" / "png" / "LD_X" / "a.png").is_file()
    assert not (tmp_path / "livrable" / "indices" / "LD_X" / "png").exists()
    assert (tmp_path / "livrable" / "detections" / "parcellaire" / "parcellaire.gpkg").is_file()
    assert (tmp_path / "livrable" / "detections" / "detections_validation.qgs").is_file()   # l'ancien projet suit, reste valide
    assert (tmp_path / "technique" / "detection" / "m" / "raw_detections" / "a.json").is_file()
    assert not (tmp_path / "livrable" / "detections" / "_technique").exists()               # vidé puis retiré
    assert (tmp_path / "technique" / "intermediaires" / "halo" / "LD_X" / "a.tif").is_file()
    assert (tmp_path / "technique" / "sources" / "dalles" / "a.laz").is_file()
    assert (tmp_path / "technique" / "sources" / "dalles_urls.txt").is_file()
    assert (tmp_path / "technique" / "journaux" / "metadata_20261008_161911.json").is_file()
    assert (tmp_path / "livrable" / "rapport.html").is_file()
    assert (tmp_path / "MNT" / "tif" / "vieux.tif").is_file() and (tmp_path / "fichier_tri.txt").is_file()
    assert len(journal) == 10 and journal[0].startswith("Réorganisation : indices → livrable/indices")
    # Après : plus rien à migrer, et les chemins du plugin tombent juste.
    assert etat(tmp_path) == "nouvelle" and plan_migration(tmp_path) == []
    assert op.indice_tif_dir(tmp_path, "MNT") == tmp_path / "livrable" / "indices" / "MNT" / "tif"
    assert op.indice_png_dir(tmp_path, "LD_X") == tmp_path / "technique" / "png" / "LD_X"
    assert op.detection_technique_raw_dir(tmp_path, "m") == tmp_path / "technique" / "detection" / "m" / "raw_detections"


def test_destination_occupee_laissee_en_place(tmp_path):
    _fichier(tmp_path / "indices" / "MNT" / "tif" / "a.tif")
    _fichier(tmp_path / "livrable" / "indices" / "MNT" / "tif" / "b.tif")     # déjà en v3, partiellement
    plan = plan_migration(tmp_path)
    assert plan == []
    assert non_reconnus(tmp_path, plan) == ["indices"]


def test_chemins_v3_du_module_output_paths(tmp_path):
    assert op.livrable_dir(tmp_path) == tmp_path / "livrable" and op.technique_dir(tmp_path) == tmp_path / "technique"
    assert op.projet_qgs_path(tmp_path) == tmp_path / "livrable" / "projet.qgs"
    assert op.traitement_json_path(tmp_path) == tmp_path / "livrable" / "traitement.json"
    assert op.journaux_dir(tmp_path) == tmp_path / "technique" / "journaux"
    assert op.dalles_urls_path(tmp_path) == tmp_path / "technique" / "sources" / "dalles_urls.txt"
    assert op.dalles_dir(tmp_path) == tmp_path / "technique" / "sources" / "dalles"
    assert op.intermediaires_dir(tmp_path) == tmp_path / "technique" / "intermediaires"
    assert op.detections_dir(tmp_path) == tmp_path / "livrable" / "detections"
    assert op.detection_entity_dir(tmp_path, "parcellaire") == tmp_path / "livrable" / "detections" / "parcellaire"
    assert op.VERSION_ARBORESCENCE == 3
