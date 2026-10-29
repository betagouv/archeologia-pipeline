"""Validateur des modèles — bloc ``thresholds.fiabilite`` au critère « axe » et en unité
« polygone » (rapport linéaires du 2026-10-22) : la tolérance se déclare et se justifie, l'unité
déclarée est celle de l'évaluation, les bandes ``sous_seuil`` se re-dérivent des bandes de
l'évaluation comme les catégories."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "scripts"))

from validate_models_metadata import ValidationReport, _validate_fiabilite  # noqa: E402

CLASSES = {"parcellaire"}
SEUIL = 0.26
SOURCE = "entrainement/evaluation_axe/metriques_eval.json"


def _bandes_eval():
    """Bandes fines (0,01) de 0,15 à 0,29 : tp 1 / fp 3 chacune."""
    return [{"lo": round(0.15 + i / 100, 2), "hi": round(0.16 + i / 100, 2), "tp": 1, "fp": 3} for i in range(15)]


def _eval(model_dir: Path, *, critere="axe", tolerance=5.0, unite="polygone", bandes=None):
    d = model_dir / "entrainement" / "evaluation_axe"
    d.mkdir(parents=True, exist_ok=True)
    en_tete = {"critere": critere, "unite": unite}
    if tolerance is not None:
        en_tete["tolerance_m"] = tolerance
    (d / "metriques_eval.json").write_text(json.dumps({
        **en_tete,
        "modeles": {model_dir.name: {"par_classe": {
            "parcellaire": {"etude_seuil": {"bandes": bandes if bandes is not None else _bandes_eval()}}}}},
    }), encoding="utf-8")


def _fiab(**extra):
    bloc = {
        "source": SOURCE,
        "critere": "axe",
        "tolerance_m": 5.0,
        "tolerance_provenance": "3,5 m reproduit la couverture ; 5 m retenu (sensibilité publiée à 3,5 et 5)",
        "unite": "polygone",
        "par_classe": {"parcellaire": [
            # bandes 0,26–0,29 : 4 × (1 ; 3) → n 16, sous N_MIN_MESURE → mesure null
            {"categorie": "possible", "seuil": SEUIL, "garanti": 0.35, "mesure": None, "n": 16},
        ]},
        # fenêtres de 0,05 sous le seuil : [0,15 ; 0,20[ → 5 × (1 ; 3), [0,20 ; 0,25[ idem, [0,25 ; 0,26[ → 1 × (1 ; 3)
        "sous_seuil": {"parcellaire": [
            {"lo": 0.15, "tp": 5, "fp": 15}, {"lo": 0.20, "tp": 5, "fp": 15}, {"lo": 0.25, "tp": 1, "fp": 3},
        ]},
        "provenance": "p",
    }
    bloc.update(extra)
    for k in [k for k, v in bloc.items() if v is ...]:
        del bloc[k]
    return bloc


def _run(tmp_path, fiab):
    thresholds = {"confidence_default": SEUIL, "confidence_per_class": {"parcellaire": SEUIL}, "fiabilite": fiab}
    report = ValidationReport(model_dir=tmp_path)
    _validate_fiabilite(fiab, thresholds, CLASSES, SEUIL, thresholds["confidence_per_class"], {}, report,
                        model_dir=tmp_path)
    return report


def test_bloc_complet_conforme(tmp_path):
    _eval(tmp_path)
    r = _run(tmp_path, _fiab())
    assert r.errors == [] and r.warnings == []


def test_axe_exige_tolerance_et_justification(tmp_path):
    _eval(tmp_path)
    r = _run(tmp_path, _fiab(tolerance_m=...))
    assert any("tolerance_m" in e for e in r.errors)
    r = _run(tmp_path, _fiab(tolerance_m=3.5))
    assert any("tolerance_m" in e and "3.5" in e and "5.0" in e for e in r.errors)
    r = _run(tmp_path, _fiab(tolerance_provenance=...))
    assert any("tolerance_provenance" in e for e in r.errors)
    # l'évaluation seule dit « axe » (clé critere absente de la carte) : mêmes exigences
    r = _run(tmp_path, _fiab(critere=..., tolerance_m=..., tolerance_provenance=...))
    assert any("tolerance_m" in e for e in r.errors)


def test_critere_et_unite_declares_doivent_correspondre(tmp_path):
    _eval(tmp_path)
    r = _run(tmp_path, _fiab(critere="couverture"))
    assert any("critere" in w and "axe" in w for w in r.warnings)
    r = _run(tmp_path, _fiab(unite="fragment"))
    assert any("unite" in w and "polygone" in w for w in r.warnings)
    r = _run(tmp_path, _fiab(unite="objet"))
    assert any("unite" in e for e in r.errors)


def test_sous_seuil_rederive_des_bandes(tmp_path):
    _eval(tmp_path)
    faux = _fiab(sous_seuil={"parcellaire": [
        {"lo": 0.15, "tp": 9, "fp": 15}, {"lo": 0.20, "tp": 5, "fp": 15}, {"lo": 0.25, "tp": 1, "fp": 3}]})
    r = _run(tmp_path, faux)
    assert any("sous_seuil" in w and "0.15" in w and "9/15" in w and "5/15" in w for w in r.warnings)
    r = _run(tmp_path, _fiab(sous_seuil={"autre": [{"lo": 0.15, "tp": 1, "fp": 1}]}))
    assert any("sous_seuil" in e and "autre" in e for e in r.errors)
    r = _run(tmp_path, _fiab(sous_seuil={"parcellaire": [{"lo": 0.26, "tp": 1, "fp": 1}]}))
    assert any("sous_seuil" in e and "0.26" in e for e in r.errors)
    r = _run(tmp_path, _fiab(sous_seuil={"parcellaire": "x"}))
    assert any("sous_seuil" in e for e in r.errors)


def test_unite_polygone_sans_sous_seuil_avertit(tmp_path):
    _eval(tmp_path)
    r = _run(tmp_path, _fiab(sous_seuil=...))
    assert r.errors == [] and any("sous_seuil" in w for w in r.warnings)
    # unité fragment (contrat historique) : rien n'est exigé
    _eval(tmp_path, critere="couverture", tolerance=None, unite="fragment")
    r = _run(tmp_path, _fiab(critere=..., tolerance_m=..., tolerance_provenance=..., unite=..., sous_seuil=...))
    assert r.errors == [] and r.warnings == []


@pytest.mark.parametrize("cle", ["critere", "unite", "tolerance_m", "tolerance_provenance", "sous_seuil"])
def test_cles_nouvelles_optionnelles_hors_axe(tmp_path, cle):
    """Un modèle au critère objet (ponctuelles, dépressions) n'a aucune des clés nouvelles."""
    _eval(tmp_path, critere="iou", tolerance=None, unite="fragment")
    base = _fiab(critere=..., tolerance_m=..., tolerance_provenance=..., unite=..., sous_seuil=...)
    r = _run(tmp_path, base)
    assert cle not in base and r.errors == []
