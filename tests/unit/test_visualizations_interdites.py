"""Règle utilisateur 2026-09-15 : pas de visualisations de test dans data/models."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from validate_models_metadata import validate_model_dir  # noqa: E402


def test_visualizations_dans_le_modele_est_une_erreur(tmp_path: Path) -> None:
    (tmp_path / "entrainement" / "visualizations").mkdir(parents=True)
    (tmp_path / "entrainement" / "visualizations" / "pred_0001.png").write_bytes(b"x")
    report = validate_model_dir(tmp_path, strict=False)
    assert any("visualizations" in e for e in report.errors)


def test_sans_visualizations_rien_a_signaler(tmp_path: Path) -> None:
    (tmp_path / "entrainement" / "evaluation").mkdir(parents=True)
    report = validate_model_dir(tmp_path, strict=False)
    assert not any("visualizations" in e for e in report.errors)
