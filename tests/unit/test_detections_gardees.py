"""Fin de détection (recette séance B, 2026-10-21) : le journal annonçait « 2742 zones détectées », la
somme des détections brutes du moteur, avant dédoublonnage entre dalles, seuil et aire minimale ; les
couches en contenaient 310. L'annonce compte désormais ce que contiennent les GeoPackages des entités."""
from __future__ import annotations

import sqlite3

from pipeline.output_paths import detection_entity_dir, detections_gardees


def _gpkg(output_dir, slug, couches):
    d = detection_entity_dir(output_dir, slug)
    d.mkdir(parents=True)
    con = sqlite3.connect(d / f"{slug}.gpkg")
    for nom, n in couches.items():
        con.execute(f'create table "{nom}" (fid integer primary key)')
        con.executemany(f'insert into "{nom}" default values', [()] * n)
    con.commit()
    con.close()


def test_compte_les_couches_des_entites_du_run(tmp_path):
    _gpkg(tmp_path, "parcellaire", {"parcellaire": 127})
    _gpkg(tmp_path, "talus_et_fosses", {"talus_fosse": 137})
    runs = [{"entities": [
        {"slug": "parcellaire", "classes": ["parcellaire"]},
        {"slug": "talus_et_fosses", "classes": ["talus_fosse"]},
        {"slug": "chemins_creux", "classes": ["chemin_creux"]},     # rien gardé : pas de GeoPackage
    ]}]
    assert detections_gardees(tmp_path, runs) == 264


def test_zero_quand_tout_est_ecarte(tmp_path):
    assert detections_gardees(tmp_path, [{"entities": [{"slug": "fours", "classes": ["four"]}]}]) == 0


def test_inconnu_sans_entite(tmp_path):
    assert detections_gardees(tmp_path, [{"selected_model": "m"}]) is None


def test_le_narrateur_parle_de_detections_gardees():
    from unittest.mock import MagicMock
    from app.user_narrator import create_user_narrator
    r = MagicMock()
    create_user_narrator(r).cv_complete(310)
    assert "310 détections dans les couches" in r.user_info.call_args[0][0]
