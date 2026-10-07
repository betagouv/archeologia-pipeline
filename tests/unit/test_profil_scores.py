"""Profil des scores d'une classe : lecture des bandes de l'évaluation et agrégation (pur)."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.app.services.fiabilite import Categorie
from src.app.services.profil_scores import (
    Bande,
    Profil,
    agreger,
    charger_bandes,
    profil_pour_classe,
)

CATS = (
    Categorie("douteux", 0.29, 0.0, 0.26, 100),
    Categorie("possible", 0.35, 0.35, 0.45, 100),
    Categorie("probable", 0.50, 0.60, 0.72, 100),
    Categorie("quasi_certain", 0.65, 0.85, 0.92, 100),
)


def _bandes(spec):
    return [{"lo": lo, "hi": round(lo + 0.01, 2), "tp": tp, "fp": fp} for lo, tp, fp in spec]


def _modele(tmp_path: Path, nom="m1", cle_modeles=None, fiabilite=None, bloc=None):
    d = tmp_path / nom
    (d / "entrainement" / "evaluation").mkdir(parents=True)
    card = {"thresholds": {"fiabilite": fiabilite if fiabilite is not None else {"provenance": "x"}}}
    (d / "model_card.yaml").write_text(yaml.safe_dump(card), encoding="utf-8")
    ev = {"modeles": {cle_modeles or nom: bloc or {}}}
    (d / "entrainement" / "evaluation" / "metriques_eval.json").write_text(json.dumps(ev), encoding="utf-8")
    return d


def test_bandes_globales_et_agregation(tmp_path):
    bloc = {"global": {"etude_seuil": {"bandes": _bandes([(0.05, 1, 10), (0.06, 2, 10), (0.10, 3, 5), (0.29, 4, 4), (0.30, 5, 1)])}}}
    d = _modele(tmp_path, bloc=bloc)
    p = profil_pour_classe(d, "cratere", CATS)
    assert p is not None and p.n_sous_seuil == 31      # bandes fines dont hi <= 0,29 : 11 + 12 + 8
    bandes = charger_bandes(d, "cratere")
    assert len(bandes) == 5
    agg = agreger(bandes)
    assert [(b.lo, b.hi, b.tp, b.fp) for b in agg] == [
        (0.05, 0.1, 3, 20), (0.1, 0.15, 3, 5), (0.25, 0.3, 4, 4), (0.3, 0.35, 5, 1),
    ]
    assert agg[0].part_vrais == 3 / 23 and agg[0].total == 23


def test_par_classe_prime_sur_global_et_cle_modele_differente(tmp_path):
    bloc = {
        "global": {"etude_seuil": {"bandes": _bandes([(0.5, 1, 1)])}},
        "par_classe": {"four": {"etude_seuil": {"bandes": _bandes([(0.5, 9, 1)])}}},
    }
    # L'id du dossier n'est pas la clé de ``modeles`` (poids renommés) : le seul modèle évalué est pris.
    d = _modele(tmp_path, nom="ponctuelles", cle_modeles="ponctuelles_ep34", bloc=bloc)
    assert charger_bandes(d, "four")[0].tp == 9
    assert charger_bandes(d, "charbonniere")[0].tp == 1       # repli global


def test_zones_sommees_comme_le_validateur(tmp_path):
    bloc = {
        "par_classe": {"parcellaire": {"etude_seuil": {"bandes": _bandes([(0.3, 100, 100)])}}},
        "par_zone_classe": {
            "A": {"parcellaire": {"bandes": _bandes([(0.3, 10, 2)])}},
            "B": {"parcellaire": {"bandes": _bandes([(0.3, 5, 1), (0.31, 1, 0)])}},
            "C": {"parcellaire": {"bandes": _bandes([(0.3, 1000, 1000)])}},   # hors zones déclarées
        },
    }
    fiab = {"provenance": "x", "zones": {"parcellaire": ["A", "B"]}}
    d = _modele(tmp_path, nom="lin", fiabilite=fiab, bloc=bloc)
    bandes = charger_bandes(d, "parcellaire", None, ["A", "B"])
    assert [(b.lo, b.tp, b.fp) for b in bandes] == [(0.3, 15, 3), (0.31, 1, 0)]
    p = profil_pour_classe(d, "parcellaire", CATS)
    assert p is not None and p.bandes[0].tp == 16        # zones lues dans le model_card
    assert p.n_sous_seuil == 0


def test_source_declaree_dans_le_model_card(tmp_path):
    d = _modele(tmp_path, nom="tr", fiabilite={"provenance": "x", "source": "entrainement/evaluation_couverture/metriques_eval.json"})
    autre = d / "entrainement" / "evaluation_couverture"
    autre.mkdir()
    (autre / "metriques_eval.json").write_text(json.dumps({"modeles": {"tr": {"global": {"etude_seuil": {"bandes": _bandes([(0.4, 7, 3)])}}}}}), encoding="utf-8")
    p = profil_pour_classe(d, "tranchees", CATS)
    assert p is not None and p.bandes == (Bande(0.4, 0.45, 7, 3),)


def test_profil_proprietes_et_cas_degrades(tmp_path):
    p = Profil("c", (Bande(0.25, 0.3, 1, 9), Bande(0.65, 0.7, 9, 1)), CATS)
    assert p.seuil == 0.29 and p.coupures == (0.29, 0.35, 0.5, 0.65) and p.total == 20
    assert p.categorie_de(0.2) is None
    assert p.categorie_de(0.29).categorie == "douteux"
    assert p.categorie_de(0.64).categorie == "probable"
    assert p.categorie_de(0.9).categorie == "quasi_certain"
    assert profil_pour_classe(tmp_path / "absent", "c", CATS) is None
    assert profil_pour_classe(tmp_path, "c", ()) is None
    d = _modele(tmp_path, nom="vide", bloc={"global": {}})
    assert profil_pour_classe(d, "c", CATS) is None
    assert agreger([]) == []


def test_les_deux_fiches_posent_la_figure():
    """Garde-fou sans QGIS : fiche de classe et fiche ⓘ du modèle appellent ``figure_profil``."""
    racine = Path(__file__).resolve().parents[2]
    for rel in ("src/ui/dialogs/class_info_dialog.py", "src/ui/dialogs/model_info_dialog.py"):
        assert "figure_profil(" in (racine / rel).read_text(encoding="utf-8"), rel


def test_agregation_coupee_aux_seuils():
    """La barre [0,25 ; 0,30[ ne chevauche plus le seuil 0,26 : elle est scindée."""
    fines = [Bande(round(0.25 + i * 0.01, 2), round(0.26 + i * 0.01, 2), i + 1, 10) for i in range(5)]
    agg = agreger(fines, 0.05, coupures=[0.26, 0.30])
    assert [(b.lo, b.hi) for b in agg] == [(0.25, 0.26), (0.26, 0.3)]
    assert (agg[0].tp, agg[0].fp) == (1, 10) and (agg[1].tp, agg[1].fp) == (2 + 3 + 4 + 5, 40)
    # sans coupure : une seule bande de 0,05
    assert [(b.lo, b.hi) for b in agreger(fines, 0.05)] == [(0.25, 0.3)]


def test_eclats_fusionnes_du_meme_cote_de_la_coupure():
    """0,20–0,26 d'un seul tenant sous le seuil 0,26 ; 0,29–0,35 d'un seul tenant au niveau douteux."""
    fines = [Bande(round(0.20 + i * 0.01, 2), round(0.21 + i * 0.01, 2), 1, 1) for i in range(15)]  # 0,20 → 0,35
    assert [(b.lo, b.hi) for b in agreger(fines, 0.05, coupures=[0.26, 0.30])] == [(0.2, 0.26), (0.26, 0.3), (0.3, 0.35)]
    assert [(b.lo, b.hi) for b in agreger(fines, 0.05, coupures=[0.29, 0.35])] == [(0.2, 0.25), (0.25, 0.29), (0.29, 0.35)]
    # les effectifs sont conservés
    assert sum(b.tp for b in agreger(fines, 0.05, coupures=[0.26, 0.30])) == 15
