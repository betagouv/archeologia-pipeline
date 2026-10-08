"""Carte des zones d'apprentissage (bandeau « Appris sur », 2026-10-08) : module pur,
fichier data/zones_corpus.json livré, et toute zone d'une fiche installée est située."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.carte_zones import (
    anneaux_du_pays,
    charger,
    normaliser,
    pays_presents,
    phrase_resume,
    projection,
    rayon,
    situer,
)

RACINE = Path(__file__).resolve().parents[2]
DATA = {
    "noms": {"Haut-Doubs (25)": "bfc/25_haut_doubs", "Sligo (Irlande)": "irlande/ie_sligo"},
    "zones": {"bfc/25_haut_doubs": {"pays": "france", "emprise": [6.0, 46.5, 6.4, 46.8]},
              "irlande/ie_sligo": {"pays": "irlande", "emprise": [-8.6, 54.0, -8.2, 54.2]}},
    "contours": {"france": [[[-4.0, 48.0], [7.0, 49.0], [7.5, 43.5], [-1.0, 43.4], [-4.0, 48.0]]]},
}


def z(nom, tuiles=10, objets=100):
    return SimpleNamespace(nom=nom, tuiles=tuiles, objets=objets)


def test_situer_nom_exact_normalise_et_inconnu():
    res = situer([z("Haut-Doubs (25)"), z("haut-doubs  (25)"), z("Inconnue (99)"), z("Sligo (Irlande)", 5, 7)], DATA)
    assert [(r.nom, r.pays) for r in res] == [("Haut-Doubs (25)", "france"), ("haut-doubs  (25)", "france"), ("Sligo (Irlande)", "irlande")]
    assert res[0].lon == pytest.approx(6.2) and res[0].lat == pytest.approx(46.65)
    assert normaliser("  Vosges  saônoises (70) ") == "vosges saonoises (70)"
    assert situer([z("Haut-Doubs (25)")], {}) == []
    assert pays_presents(res) == ["france", "irlande"]


def test_projection_rayon_et_resume():
    anneaux = anneaux_du_pays(DATA, "france")
    f, lu, hu = projection(anneaux, "france", 150, 150, marge=4)
    xs = [f(lon, lat)[0] for lon, lat in anneaux[0]]
    ys = [f(lon, lat)[1] for lon, lat in anneaux[0]]
    assert min(xs) >= 3.9 and max(xs) <= 146.1 and min(ys) >= 3.9 and max(ys) <= 146.1
    assert f(-4.0, 49.0)[1] < f(-4.0, 44.0)[1]            # le nord en haut
    assert anneaux_du_pays(DATA, "irlande") == []
    assert rayon(100, 100, 150) == pytest.approx(13.5) and rayon(0, 100, 150) == 3.0 and rayon(5, 0, 150) == 3.0
    assert phrase_resume([z("a", 1, 1035), z("b", 1, 971), z("c", 1, 180)]) == "3 zones · 2 186 objets annotés"
    assert phrase_resume([z("a", 1, 0)]) == "1 zone"


def test_fichier_livre_lisible():
    data = charger(RACINE)
    assert data, "data/zones_corpus.json absent ou illisible : relancer dev/fiches/zones_corpus.py"
    assert anneaux_du_pays(data, "france") and anneaux_du_pays(data, "irlande")
    for zid, e in data["zones"].items():
        lon0, lat0, lon1, lat1 = e["emprise"]
        assert lon0 < lon1 and lat0 < lat1, zid
        assert e["pays"] in ("france", "irlande"), zid
    assert set(data["noms"].values()) <= set(data["zones"]), "un nom pointe une zone sans emprise"


def test_toute_zone_des_fiches_installees_est_situee():
    """Un modèle installé dont la fiche cite une zone nouvelle : relancer
    ``C:/OSGeo4W/bin/python-qgis.bat dev/fiches/zones_corpus.py``."""
    yaml = pytest.importorskip("yaml")
    from app.services.class_fiche import build_all_fiches

    cards = sorted((RACINE / "data" / "models").glob("*/model_card.yaml"))
    if not cards:
        pytest.skip("aucun modèle installé")
    data = charger(RACINE)
    manquants = []
    for card_path in cards:
        for f in build_all_fiches(yaml.safe_load(card_path.read_text(encoding="utf-8")) or {}):
            if f.entrainement is None:
                continue
            situees = {s.nom for s in situer(f.entrainement.zones, data)}
            manquants += [f"{card_path.parent.name}/{f.nom} : {zz.nom}" for zz in f.entrainement.zones if zz.nom not in situees]
    assert manquants == [], f"zones non situées : {manquants}"


def test_la_fiche_pose_le_bandeau():
    src = (RACINE / "src/ui/dialogs/class_info_dialog.py").read_text(encoding="utf-8")
    assert "bandeau_appris_sur(" in src and "avec_zones=bandeau is None" in src
    widget = (RACINE / "src/ui/widgets/carte_zones.py").read_text(encoding="utf-8")
    assert "def surligner" in widget and "QToolTip.showText" in widget
    assert json.loads((RACINE / "data/zones_corpus.json").read_text(encoding="utf-8"))["source"]
