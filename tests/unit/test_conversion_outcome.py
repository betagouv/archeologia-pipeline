"""F1 (audit « rassemblement de polygones », 2026-06-12).

Le retour booléen de ``create_shapefile_from_detections`` n'était jamais
inspecté au point d'appel (``runner_shapefiles``) : toute panne de conversion
(retour ``False`` ou exception avalée) se traduisait par un « succès »
silencieux avec zéro détection. ``summarize_conversion_outcome`` classe le
résultat — échec / 0 détection / ok — pour le **remonter au narrateur**.
"""
from __future__ import annotations

import pytest

pytest.importorskip("shapely")  # pipeline.cv.__init__ tire shapely

from pipeline.cv.conversion_outcome import (  # noqa: E402
    ConversionOutcome,
    summarize_conversion_outcome,
)


def test_exception_is_failure_and_keeps_message():
    out = summarize_conversion_outcome(
        returned_ok=False, error="boom disk", n_gpkgs_written=0
    )
    assert isinstance(out, ConversionOutcome)
    assert out.status == "failed"
    assert out.is_failure
    assert "boom disk" in out.message


def test_returned_false_without_exception_is_failure():
    out = summarize_conversion_outcome(
        returned_ok=False, error=None, n_gpkgs_written=0
    )
    assert out.status == "failed"
    assert out.is_failure


def test_ok_but_nothing_written_is_empty_not_failure():
    # Distinction clé : « 0 détection » n'est PAS un échec (F1).
    out = summarize_conversion_outcome(
        returned_ok=True, error=None, n_gpkgs_written=0
    )
    assert out.status == "empty"
    assert not out.is_failure


def test_ok_with_layers_is_ok():
    out = summarize_conversion_outcome(
        returned_ok=True, error=None, n_gpkgs_written=3
    )
    assert out.status == "ok"
    assert not out.is_failure
    assert "3" in out.message
