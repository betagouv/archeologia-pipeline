"""Catalogue de l'onglet « Visualisation » — module pur, testable hors QGIS."""
from __future__ import annotations

import json

import pytest

from src.app.services.visu_catalogue import (
    FAMILY_BASE,
    FAMILY_QUALITE,
    FAMILY_RVT,
    Catalogue,
    CatalogItem,
    Department,
    filter_departments,
    indice_info,
    load_catalogue,
)


def _dept(code="35", name="Ille-et-Vilaine", keys=("MNT", "SVF")) -> Department:
    return Department(
        code=code, name=name, updated="2026-07", resolution=0.5,
        items=[CatalogItem(key=k, source=f"/tmp/{k}.vrt") for k in keys],
    )


# ----------------------------------------------------------- vocabulaire

def test_libelle_metier_prime_sur_le_sigle():
    """Le titre d'une carte est métier ; le sigle reste le badge de vignette."""
    svf = indice_info("SVF")
    assert svf.metier == "Creux & dépressions"
    assert svf.sigle == "SVF"
    assert svf.name == "Sky-View Factor"       # nom technique, seconde ligne


def test_nom_technique_vient_de_indices_model():
    """Pas de second catalogue de produits : indices_model reste la source."""
    from src.app.services.indices_model import product

    for key in ("MNT", "LD", "CVAT", "DENSITE"):
        assert indice_info(key).name == product(key).full_name


def test_familles_des_produits():
    assert indice_info("MNT").family == FAMILY_BASE
    assert indice_info("SVF").family == FAMILY_RVT
    assert indice_info("DENSITE").family == FAMILY_QUALITE
    assert indice_info("COUVERTURE").family == FAMILY_QUALITE


def test_cle_inconnue_reste_affichable():
    """Un catalogue publié plus tard ne doit pas casser une version installée."""
    info = indice_info("FUTUR_INDICE")
    assert info.key == info.sigle == "FUTUR_INDICE"
    assert info.family == FAMILY_RVT


# ---------------------------------------------------------------- tri

def test_ordre_daffichage_socle_dabord():
    dept = _dept(keys=("SLO", "MNT", "SVF"))
    assert [i.key for i in dept.sorted_items()] == ["MNT", "SVF", "SLO"]


def test_filtre_par_famille():
    dept = _dept(keys=("MNT", "SVF", "DENSITE"))
    assert [i.key for i in dept.items_in_family(FAMILY_RVT)] == ["SVF"]
    assert [i.key for i in dept.items_in_family(FAMILY_BASE)] == ["MNT"]
    # « Tous » = pas de filtre
    assert len(dept.items_in_family(None)) == 3


# ------------------------------------------------------------- filtre rail

@pytest.mark.parametrize("query", ["cotes", "CÔTES", "Côtes", "22"])
def test_filtre_departement_insensible_casse_et_accents(query):
    depts = [_dept("22", "Côtes-d'Armor"), _dept("35", "Ille-et-Vilaine")]
    assert [d.code for d in filter_departments(depts, query)] == ["22"]


def test_filtre_vide_rend_tout():
    depts = [_dept("22"), _dept("35")]
    assert len(filter_departments(depts, "   ")) == 2


def test_filtre_sans_resultat():
    assert filter_departments([_dept()], "zzz") == []


# --------------------------------------------------------- couverture

def test_covered_ignore_les_departements_sans_indice():
    cat = Catalogue(departments=[_dept("35"), Department(code="75", name="Paris")])
    assert [d.code for d in cat.covered] == ["35"]
    assert cat.product_count == 2


# ------------------------------------------------------------ chargement

def test_load_catalogue(tmp_path):
    path = tmp_path / "catalogue.json"
    path.write_text(json.dumps({
        "updated": "2026-09",
        "resolution": 0.5,
        "departments": [{
            "code": "35", "name": "Ille-et-Vilaine", "updated": "2026-07",
            "items": [{
                "key": "LD", "source": "/data/index_LD.vrt",
                "thumbnail": "thumbs/LD.png", "size_go": 2.4,
                "extent": [357000, 6793000, 366000, 6803000], "streamed": False,
            }],
        }],
    }), encoding="utf-8")

    cat = load_catalogue(path)
    assert cat.updated == "2026-09"
    dept = cat.by_code("35")
    assert dept is not None and dept.resolution == 0.5     # hérité de la racine
    item = dept.items[0]
    assert item.source == "/data/index_LD.vrt"
    assert item.thumbnail == "thumbs/LD.png"   # laissé relatif : résolu par l'appelant
    assert item.extent == [357000, 6793000, 366000, 6803000]
    assert item.streamed is False
    assert item.info.metier == "Structures en relief"


def test_load_accepte_cog_url_du_contrat(tmp_path):
    """Le brief nomme le champ ``cog_url`` ; un catalogue distant peut l'utiliser."""
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"departments": [
        {"code": "33", "name": "Gironde",
         "items": [{"key": "SVF", "cog_url": "https://x/svf.tif"}]},
    ]}), encoding="utf-8")
    assert load_catalogue(path).by_code("33").items[0].source == "https://x/svf.tif"


def test_load_ignore_les_entrees_inexploitables(tmp_path):
    """Une entrée sans source ne doit pas faire tomber tout le mur."""
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"departments": [
        {"code": "35", "name": "I-et-V", "items": [
            {"key": "LD", "source": "/ok.vrt"},
            {"key": "SVF"},                       # pas de source
            {"source": "/orphelin.vrt"},          # pas de clé
            {"key": "MNT", "source": "/m.vrt", "extent": [1, 2]},   # emprise invalide
        ]},
        {"name": "sans code", "items": []},
    ]}), encoding="utf-8")
    cat = load_catalogue(path)
    assert len(cat.departments) == 1
    assert [i.key for i in cat.by_code("35").items] == ["LD", "MNT"]
    assert cat.by_code("35").items[1].extent is None


def test_load_catalogue_illisible(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_catalogue(path)
    with pytest.raises(FileNotFoundError):
        load_catalogue(tmp_path / "absent.json")


# ------------------------------------------------- catalogue livré

def test_catalogue_livre_est_coherent():
    """Le catalogue de démonstration doit rester chargeable et pointer ses vignettes."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    path = root / "data" / "demo_catalogue" / "catalogue.json"
    if not path.is_file():
        pytest.skip("catalogue de démonstration non généré")

    cat = load_catalogue(path)
    assert cat.covered, "aucun département consultable"
    for dept in cat.covered:
        assert dept.name and dept.code
        for item in dept.items:
            assert item.source
            if item.thumbnail:
                assert (path.parent / item.thumbnail).is_file(), \
                    f"vignette manquante : {item.thumbnail}"


def test_chaque_produit_du_pipeline_a_un_libelle_metier():
    """Un produit absent de ``_METIER`` s'affiche quand même — mais sans titre
    métier, sans famille, et hors de ``DISPLAY_ORDER`` : il part en fin de mur,
    incohérent avec les autres. Le repli étant silencieux (cf.
    ``test_cle_inconnue_reste_affichable``), rien ne signalait le trou.
    """
    from app.services.indices_model import all_products
    from app.services.visu_catalogue import _METIER

    attendus = {p.key for p in all_products()}
    assert len(attendus) >= 14, "anti-test-creux : catalogue vidé ?"
    manquants = sorted(attendus - set(_METIER))
    assert manquants == [], f"produits sans libellé métier : {manquants}"
