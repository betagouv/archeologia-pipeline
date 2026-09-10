"""Fiabilité des détections (propositions A + D, 2026-09-09) — module pur ``app.services.fiabilite``."""
from __future__ import annotations

import json

import pytest

from app.services.fiabilite import (
    CATEGORIES,
    CHAMP_LABEL,
    STYLE_SPEC,
    Categorie,
    categories_du_run,
    categories_effectives,
    categories_sidecar,
    categoriser,
    hint_etape3,
    labels_legende,
    maptip_html,
    parse_fiabilite,
    phrase_mesure,
    read_sidecar,
    run_block,
    texte_resume,
    write_sidecar,
)

DEPRESSIONS = {
    "fiabilite": {
        "par_classe": {
            "depression_circulaire_grande": [
                {"categorie": "douteux", "seuil": 0.29, "garanti": 0.0, "mesure": 0.14, "n": 447},
                {"categorie": "possible", "seuil": 0.35, "garanti": 0.35, "mesure": 0.47, "n": 272},
                {"categorie": "probable", "seuil": 0.45, "garanti": 0.60, "mesure": 0.72, "n": 335},
                {"categorie": "quasi_certain", "seuil": 0.65, "garanti": 0.85, "mesure": 0.95, "n": 1013},
            ],
        },
        "provenance": "metriques_eval.json (2026-09-08), coupures 2026-09-09",
    },
}


def _cats():
    return parse_fiabilite(DEPRESSIONS)[0]["depression_circulaire_grande"]


class TestParse:
    def test_bloc_complet(self):
        par_classe, prov = parse_fiabilite(DEPRESSIONS)
        cats = par_classe["depression_circulaire_grande"]
        assert [c.categorie for c in cats] == list(CATEGORIES)
        assert cats[0] == Categorie("douteux", 0.29, 0.0, 0.14, 447)
        assert cats[3].label == "Très probable" and cats[3].mesure == 0.95
        assert prov.startswith("metriques_eval.json")

    def test_absent_ou_invalide_tolere(self):
        assert parse_fiabilite({}) == ({}, "")
        assert parse_fiabilite(None) == ({}, "")
        assert parse_fiabilite({"fiabilite": "x"}) == ({}, "")
        # catégorie inconnue → classe ignorée, pas d'exception
        bloc = {"fiabilite": {"par_classe": {"a": [{"categorie": "sur", "seuil": 0.3}]}}}
        assert parse_fiabilite(bloc)[0] == {}
        # seuils non croissants → ignorée ; ordre des catégories non croissant → ignorée
        bloc = {"fiabilite": {"par_classe": {"a": [
            {"categorie": "douteux", "seuil": 0.5}, {"categorie": "possible", "seuil": 0.4}]}}}
        assert parse_fiabilite(bloc)[0] == {}
        bloc = {"fiabilite": {"par_classe": {"a": [
            {"categorie": "probable", "seuil": 0.3}, {"categorie": "possible", "seuil": 0.4}]}}}
        assert parse_fiabilite(bloc)[0] == {}
        # mesure null acceptée (effectif insuffisant), seuil hors [0,1] refusé
        bloc = {"fiabilite": {"par_classe": {"a": [
            {"categorie": "douteux", "seuil": 0.3, "mesure": None, "n": 12}]}}}
        assert parse_fiabilite(bloc)[0]["a"][0].mesure is None
        bloc = {"fiabilite": {"par_classe": {"a": [{"categorie": "douteux", "seuil": 1.5}]}}}
        assert parse_fiabilite(bloc)[0] == {}


class TestEffectives:
    def test_seuil_du_modele_inchange(self):
        eff = categories_effectives(_cats(), 0.29)
        assert [c.seuil for c in eff] == [0.29, 0.35, 0.45, 0.65]

    def test_seuil_releve_supprime_les_categories_sous_le_seuil(self):
        eff = categories_effectives(_cats(), 0.5)
        assert [c.categorie for c in eff] == ["probable", "quasi_certain"]
        assert eff[0].seuil == 0.5 and eff[0].mesure == 0.72  # démarre AU seuil effectif
        eff = categories_effectives(_cats(), 0.35)  # exactement sur une coupure
        assert [c.categorie for c in eff] == ["possible", "probable", "quasi_certain"]
        assert eff[0].seuil == 0.35

    def test_seuil_abaisse_etend_la_categorie_basse(self):
        eff = categories_effectives(_cats(), 0.2)
        assert eff[0].categorie == "douteux" and eff[0].seuil == 0.2 and eff[0].mesure == 0.14

    def test_seuil_tres_haut_garde_la_categorie_haute(self):
        eff = categories_effectives(_cats(), 0.99)  # la plus haute couvre [0,99 ; 1]
        assert [c.categorie for c in eff] == ["quasi_certain"] and eff[0].seuil == 0.99
        assert categories_effectives((), 0.3) == ()


class TestCategoriser:
    def test_bornes(self):
        cats = _cats()
        assert categoriser(0.28, cats) is None
        assert categoriser(0.29, cats).categorie == "douteux"
        assert categoriser(0.3499, cats).categorie == "douteux"
        assert categoriser(0.35, cats).categorie == "possible"
        assert categoriser(0.64, cats).categorie == "probable"
        assert categoriser(0.65, cats).categorie == "quasi_certain"
        assert categoriser(1.0, cats).categorie == "quasi_certain"
        assert categoriser(6.5, cats).categorie == "quasi_certain"  # confiance sur [0,10]
        assert categoriser(None, cats) is None
        assert categoriser("x", cats) is None
        assert categoriser(0.9, ()) is None


class TestTextes:
    def test_labels_legende(self):
        lab = labels_legende(_cats())
        assert lab["quasi_certain"] == "Très probable · ≥ 85 % de vrais"
        assert lab["probable"] == "Probable · ≥ 60 % de vrais"
        assert lab["douteux"] == "Douteux · < 35 % de vrais"
        seule = (Categorie("douteux", 0.3, 0.0, None, 5),)
        assert labels_legende(seule)["douteux"] == "Douteux · part de vrais non garantie"

    def test_phrase_mesure(self):
        cats = _cats()
        assert phrase_mesure(cats[3]) == "≥ 85 % de vrais objets sur le banc (mesuré : 95 % sur 1 013 détections)"
        assert phrase_mesure(cats[0]) == "part de vrais objets non garantie (mesuré : 14 % sur 447 détections)"
        faible = Categorie("probable", 0.5, 0.6, None, 25)
        assert phrase_mesure(faible) == "≥ 60 % de vrais objets sur le banc (effectif insuffisant : 25 détections)"

    def test_resume_couche(self):
        txt = texte_resume("Grandes dépressions (LD)", "depression_circulaire_grande", _cats(), "src")
        lignes = txt.splitlines()
        assert lignes[0].startswith("Fiabilité des détections « depression_circulaire_grande » — Grandes")
        assert lignes[1].startswith("• Très probable (score ≥ 0.65) : ≥ 85 % de vrais objets")
        assert lignes[4].startswith("• Douteux (score ≥ 0.29)")
        assert lignes[-1] == "Source : src"

    def test_maptip(self):
        html = maptip_html(_cats(), "Grandes dépressions (LD)")
        assert f'[% "{CHAMP_LABEL}" %]' in html and "round(\"confidence\", 2)" in html
        assert "WHEN \"fiabilite\" = 'Probable' THEN 'Sur le banc, 72 % des détections de cette tranche" in html
        assert "(garantie ≥ 60 %)" in html and "ELSE 'Fiabilité non renseignée.'" in html
        faible = (Categorie("probable", 0.5, 0.6, None, 25),)
        assert "effectif insuffisant sur le banc : 25" in maptip_html(faible, "m")
        # apostrophe dans le libellé/modèle : échappée dans l'expression
        assert "''" not in maptip_html(_cats(), "l'x")  # le nom du modèle n'est pas dans l'expression

    def test_hint_etape3(self):
        assert hint_etape3({"c": _cats()}) == (
            "Fiabilité affichée — douteux dès 0.29 · possible dès 0.35 · probable dès 0.45 · très probable dès 0.65")
        deux = hint_etape3({"a": _cats()[:2], "b": _cats()[2:]})
        assert deux.startswith("Fiabilité affichée — a : douteux dès 0.29 · possible dès 0.35 ; b : probable dès 0.45")
        assert hint_etape3({}) == "" and hint_etape3({"a": ()}) == ""


class TestRunBlockEtSidecar:
    def test_run_block_effectif_par_classe(self):
        par_classe = parse_fiabilite(DEPRESSIONS)[0]
        bloc = run_block(par_classe, {"depression_circulaire_grande": 0.5}, ["depression_circulaire_grande", "autre"],
                         modele="Grandes dépressions (LD)", provenance="p")
        assert bloc["modele"] == "Grandes dépressions (LD)" and bloc["provenance"] == "p"
        assert list(bloc["par_classe"]) == ["depression_circulaire_grande"]
        assert [c["categorie"] for c in bloc["par_classe"]["depression_circulaire_grande"]] == ["probable", "quasi_certain"]
        assert bloc["par_classe"]["depression_circulaire_grande"][0]["seuil"] == 0.5
        json.dumps(bloc)  # sérialisable dans config.json
        assert run_block(par_classe, {"x": 0.3}, ["x"], modele="m") is None
        assert run_block({}, {"a": 0.3}, ["a"], modele="m") is None
        cats = categories_du_run(bloc, "depression_circulaire_grande")
        assert [c.categorie for c in cats] == ["probable", "quasi_certain"]
        assert categories_du_run(bloc, "autre") == () and categories_du_run(None, "x") == ()

    def test_sidecar_lecture_ecriture(self, tmp_path):
        gpkg = tmp_path / "ent" / "ent.gpkg"
        entree = {"classe": "c", "modele": "m", "provenance": "p",
                  "categories": [c.to_dict() for c in _cats()]}
        p = write_sidecar(gpkg, "Couche A", entree)
        assert p == tmp_path / "ent" / "fiabilite.json" and p.is_file()
        write_sidecar(gpkg, "Couche B", {"classe": "d", "modele": "m", "categories": []})
        lu = read_sidecar(gpkg, "Couche A")
        assert lu["classe"] == "c" and len(lu["categories"]) == 4
        assert read_sidecar(gpkg, "Couche B")["classe"] == "d"  # fusion, pas d'écrasement
        assert read_sidecar(gpkg, "inconnue") is None
        assert read_sidecar(tmp_path / "nulle.gpkg", "x") is None
        cats = categories_sidecar(lu)
        assert cats[3].categorie == "quasi_certain" and cats[0].seuil == 0.29
        assert categories_sidecar(None) == ()
        p.write_text("{pas du json", encoding="utf-8")
        assert read_sidecar(gpkg, "Couche A") is None


def test_style_spec_degrade_de_la_couleur_de_classe():
    # Rendu retenu (2026-09-09) : contour seul, couleur de classe déclinée en luminosité
    # par catégorie (dégradé d'origine des tranches), jamais de remplissage.
    from pipeline.cv.color_palette import apply_confidence

    assert set(STYLE_SPEC) == set(CATEGORIES)
    assert all("fill_alpha" not in s and "dash" not in s for s in STYLE_SPEC.values())
    assert len({s["outline_width"] for s in STYLE_SPEC.values()}) == 1  # largeur constante
    reprs = [STYLE_SPEC[k]["repr"] for k in CATEGORIES]
    assert reprs == sorted(reprs) and all(0.0 <= r <= 1.0 for r in reprs)  # douteux -> très probable
    base = (200, 100, 40)
    nuances = [apply_confidence(base, STYLE_SPEC[k]["repr"]) for k in CATEGORIES]
    assert len(set(nuances)) == 4  # quatre paliers distincts du dégradé
    assert nuances[CATEGORIES.index("possible")] == base  # « possible » = couleur de base
    lum = [sum(n) for n in nuances]
    assert lum == sorted(lum, reverse=True)  # plus sûr = plus sombre


@pytest.mark.parametrize("cat", CATEGORIES)
def test_labels_fr(cat):
    assert Categorie(cat, 0.3, 0.0, None, 0).label[0].isupper()
