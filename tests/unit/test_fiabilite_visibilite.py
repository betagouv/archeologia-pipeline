"""Niveaux de fiabilité décochés à l'ouverture (2026-10-19) : « possible » et « douteux »
restent dans la légende mais ne se dessinent pas tant qu'on ne les coche pas ; rien n'est
filtré. Le même réglage sert au projet .qgs et au chargement direct (même fabrique)."""
from __future__ import annotations

from pathlib import Path

from app.services.fiabilite import CATEGORIES, NIVEAUX_DECOCHES_PAR_DEFAUT, visible_par_defaut

RACINE = Path(__file__).resolve().parents[2]


def test_seuls_les_niveaux_surs_sont_coches():
    assert NIVEAUX_DECOCHES_PAR_DEFAUT == ("possible", "douteux")
    assert {c: visible_par_defaut(c) for c in CATEGORIES} == {
        "douteux": False, "possible": False, "probable": True, "quasi_certain": True,
    }
    assert visible_par_defaut("inconnu")   # une catégorie imprévue reste visible


def test_la_fabrique_des_couches_applique_le_reglage():
    src = (RACINE / "src/ui/layer_loader.py").read_text(encoding="utf-8")
    bloc = src[src.index("def _apply_fiabilite_style"):src.index("def _apply_fiabilite_notices")]
    assert "setRenderState(visible_par_defaut(c.categorie))" in bloc
    # La voie conf_bin (modèles sans fiabilité) reste toute cochée.
    bloc_bins = src[src.index("def _apply_confidence_style"):src.index("def _fiabilite_symbol")]
    assert "setRenderState" not in bloc_bins


def test_le_manuel_et_la_recette_le_disent():
    aide = (RACINE / "aide/08-resultats.md").read_text(encoding="utf-8")
    assert "décochés" in aide and "très probable" in aide and "possible" in aide
    recette = (RACINE / "tests/TESTS_MANUELS_QGIS.md").read_text(encoding="utf-8")
    assert "## 57." in recette and "décochés" in recette
