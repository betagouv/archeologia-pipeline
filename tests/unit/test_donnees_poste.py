"""``exiger_ou_sauter`` (tests/conftest.py) : un test qui n'a pas sa donnée du poste (binaire, modèles,
vignettes, grille) saute, sauf si ``ARCHEO_EXIGER_DONNEES=1`` — posé par le hook pre-push dans le dossier
principal, où ces données existent : le skip y devient un échec (lot 2 du chantier d'outillage)."""
from __future__ import annotations

import pytest


def test_sans_la_variable_le_test_saute(exiger_ou_sauter, monkeypatch):
    monkeypatch.delenv("ARCHEO_EXIGER_DONNEES", raising=False)
    with pytest.raises(pytest.skip.Exception, match="binaire absent"):
        exiger_ou_sauter("binaire absent")


def test_avec_la_variable_le_test_echoue(exiger_ou_sauter, monkeypatch):
    monkeypatch.setenv("ARCHEO_EXIGER_DONNEES", "1")
    with pytest.raises(pytest.fail.Exception, match="ARCHEO_EXIGER_DONNEES=1.*binaire absent"):
        exiger_ou_sauter("binaire absent")


def test_zero_ou_vide_valent_absence(exiger_ou_sauter, monkeypatch):
    for valeur in ("", "0"):
        monkeypatch.setenv("ARCHEO_EXIGER_DONNEES", valeur)
        with pytest.raises(pytest.skip.Exception):
            exiger_ou_sauter("x")
