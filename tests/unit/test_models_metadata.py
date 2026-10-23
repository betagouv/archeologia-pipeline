"""Vérifie que chaque modèle versionné dans ``data/models/`` respecte le contrat.

Délègue à :func:`scripts.validate_models_metadata.validate_model_dir`.

Skip gracieux si aucun modèle versionné n'est présent dans le repo (la liste
paramétrée est vide → pytest n'instancie aucun cas).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
MODELS_DIR = REPO_ROOT / "data" / "models"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from validate_models_metadata import find_model_dirs, validate_model_dir  # noqa: E402


def _model_dirs() -> list[Path]:
    return find_model_dirs(MODELS_DIR)


def test_des_modeles_sont_installes(exiger_ou_sauter):
    """Sans modèle, le parametrize ci-dessous est vide et tout saute en silence."""
    if not _model_dirs():
        exiger_ou_sauter("aucun modèle sous data/models (gitignoré : poste principal seulement)")


@pytest.mark.parametrize("model_dir", _model_dirs(), ids=lambda d: d.name)
def test_model_respects_contract(model_dir: Path) -> None:
    report = validate_model_dir(model_dir, strict=False)
    assert not report.errors, (
        f"Modèle {model_dir.name} viole le contrat :\n  - "
        + "\n  - ".join(report.errors)
    )
