"""``run_tests.py`` : les arguments passés à pytest (lot 1 du chantier d'outillage, 2026-10-20).

Chargé par chemin (le lanceur vit à la racine, hors du PYTHONPATH des tests)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_LANCEUR = Path(__file__).resolve().parents[2] / "run_tests.py"


@pytest.fixture(scope="module")
def lanceur():
    spec = importlib.util.spec_from_file_location("run_tests_sous_test", _LANCEUR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parallele_par_defaut_quand_xdist_est_la(lanceur):
    assert lanceur.arguments([], xdist=True) == ["tests/", "-v", "-n", "auto"]


def test_serie_sans_xdist_ou_sur_demande(lanceur):
    assert lanceur.arguments([], xdist=False) == ["tests/", "-v"]
    assert lanceur.arguments(["--serie"], xdist=True) == ["tests/", "-v"]


def test_un_n_explicite_n_est_pas_double(lanceur):
    assert lanceur.arguments(["-n", "2"], xdist=True) == ["-n", "2", "-v"]
    assert lanceur.arguments(["-n4"], xdist=True) == ["-n4", "-v"]


def test_raccourcis_et_verbosite(lanceur):
    assert lanceur.arguments(["unit"], xdist=False) == ["tests/unit", "-v"]
    assert lanceur.arguments(["integration", "-k", "x"], xdist=False) == ["tests/integration", "-k", "x", "-v"]
    # -q demandé : on n'ajoute pas un -v qui l'annulerait (le hook pre-push veut une sortie courte).
    assert lanceur.arguments(["-q"], xdist=False) == ["-q"]
