"""Contrôle du bloc ``classes[].fiche`` par le validateur de métadonnées.

Le bloc est OPTIONNEL (aucun modèle installé ne l'a encore) : son absence est
un warning de suivi, jamais une erreur. En revanche, dès qu'il est présent il
doit être exploitable — une vignette qui pointe un fichier absent produit une
image cassée dans QGIS, donc c'est une erreur.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from validate_models_metadata import ValidationReport, _validate_fiche  # noqa: E402


CLASSES = {"depression_circulaire_grande"}


def _card(fiche, nom="depression_circulaire_grande"):
    return {"classes": [{"id": 0, "name": nom, "label_fr": "D", "fiche": fiche}]}


def _run(card, tmp_path):
    report = ValidationReport(model_dir=tmp_path)
    _validate_fiche(card, CLASSES, report, model_dir=tmp_path)
    return report


def _fiche_complete(tmp_path):
    (tmp_path / "vignettes").mkdir(exist_ok=True)
    (tmp_path / "vignettes" / "d0_brut.jpg").write_bytes(b"x")
    (tmp_path / "vignettes" / "d0_annote.jpg").write_bytes(b"x")
    return {
        "resume": "Cuvettes de 14 m et plus.",
        "usage": "Massifs forestiers, LD 0,5 m.",
        "hors_cible": ["Cuvettes de moins de 14 m"],
        "vignettes": [
            {"brut": "vignettes/d0_brut.jpg", "annote": "vignettes/d0_annote.jpg",
             "zone": "Chailluz (25)"},
        ],
        "entrainement": {
            "corpus": "depressions_grandes_648_v1",
            "annotation": "masques SAM 2.1 sur boîtes revues",
            "zones": [{"nom": "Chailluz (25)", "tuiles": 310, "objets": 902}],
            "splits": {"train": {"tuiles": 1285, "objets": 3577}},
        },
    }


class TestFicheValide:
    def test_fiche_complete_ne_produit_rien(self, tmp_path):
        r = _run(_card(_fiche_complete(tmp_path)), tmp_path)
        assert r.errors == []
        assert r.warnings == []

    def test_absence_de_bloc_est_un_warning_de_suivi(self, tmp_path):
        r = _run({"classes": [{"name": "depression_circulaire_grande"}]}, tmp_path)
        assert r.errors == []
        assert len(r.warnings) == 1
        assert "fiche" in r.warnings[0]
        for bloc in ("resume", "usage", "vignettes", "entrainement"):
            assert bloc in r.warnings[0]

    def test_fiche_partielle_nomme_ce_qui_manque(self, tmp_path):
        f = _fiche_complete(tmp_path)
        del f["usage"]
        r = _run(_card(f), tmp_path)
        assert r.errors == []
        assert "usage" in r.warnings[0]
        assert "resume" not in r.warnings[0]


class TestVignettes:
    def test_fichier_absent_est_une_erreur(self, tmp_path):
        f = _fiche_complete(tmp_path)
        f["vignettes"][0]["brut"] = "vignettes/introuvable.jpg"
        r = _run(_card(f), tmp_path)
        assert any("introuvable.jpg" in e for e in r.errors)

    def test_annote_absent_est_une_erreur(self, tmp_path):
        f = _fiche_complete(tmp_path)
        f["vignettes"][0]["annote"] = "vignettes/pas_la.jpg"
        r = _run(_card(f), tmp_path)
        assert any("pas_la.jpg" in e for e in r.errors)

    def test_annote_vide_est_tolere(self, tmp_path):
        f = _fiche_complete(tmp_path)
        f["vignettes"][0].pop("annote")
        r = _run(_card(f), tmp_path)
        assert r.errors == []

    def test_chemin_absolu_refuse(self, tmp_path):
        f = _fiche_complete(tmp_path)
        f["vignettes"][0]["brut"] = "C:/Users/moi/vignette.jpg"
        r = _run(_card(f), tmp_path)
        assert any("absolu" in e for e in r.errors)

    def test_vignette_sans_brut_refusee(self, tmp_path):
        f = _fiche_complete(tmp_path)
        f["vignettes"].append({"annote": "vignettes/d0_annote.jpg"})
        r = _run(_card(f), tmp_path)
        assert any("brut" in e for e in r.errors)

    def test_vignettes_non_liste_refusee(self, tmp_path):
        f = _fiche_complete(tmp_path)
        f["vignettes"] = "vignettes/d0_brut.jpg"
        r = _run(_card(f), tmp_path)
        assert any("liste" in e for e in r.errors)


class TestEntrainement:
    def test_zone_sans_nom_refusee(self, tmp_path):
        f = _fiche_complete(tmp_path)
        f["entrainement"]["zones"].append({"tuiles": 10, "objets": 20})
        r = _run(_card(f), tmp_path)
        assert any("nom" in e for e in r.errors)

    def test_effectif_non_numerique_refuse(self, tmp_path):
        f = _fiche_complete(tmp_path)
        f["entrainement"]["zones"][0]["objets"] = "beaucoup"
        r = _run(_card(f), tmp_path)
        assert any("objets" in e for e in r.errors)

    def test_split_inconnu_est_un_warning(self, tmp_path):
        f = _fiche_complete(tmp_path)
        f["entrainement"]["splits"]["holdout"] = {"tuiles": 5, "objets": 9}
        r = _run(_card(f), tmp_path)
        assert r.errors == []
        assert any("holdout" in w for w in r.warnings)

    def test_zones_sans_effectifs_est_un_warning(self, tmp_path):
        f = _fiche_complete(tmp_path)
        f["entrainement"]["zones"] = [{"nom": "Chailluz (25)"}]
        r = _run(_card(f), tmp_path)
        assert r.errors == []
        assert any("effectif" in w.lower() for w in r.warnings)


class TestRobustesse:
    def test_fiche_non_mapping_refusee(self, tmp_path):
        r = _run(_card("du texte"), tmp_path)
        assert any("mapping" in e for e in r.errors)

    def test_classe_hors_classes_txt_signalee(self, tmp_path):
        r = _run(_card(_fiche_complete(tmp_path), nom="inconnue"), tmp_path)
        assert any("inconnue" in e for e in r.errors)

    def test_card_sans_classes_ne_leve_pas(self, tmp_path):
        r = _run({}, tmp_path)
        assert r.errors == []


@pytest.mark.parametrize("bloc", ["resume", "usage"])
def test_texte_non_scalaire_refuse(tmp_path, bloc):
    f = _fiche_complete(tmp_path)
    f[bloc] = ["une", "liste"]
    r = _run(_card(f), tmp_path)
    assert any(bloc in e for e in r.errors)
