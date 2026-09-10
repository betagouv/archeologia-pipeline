"""Cadence des lignes « SAHI: X/Y tuiles traitées » émises par l'inférence.

Bug d'affichage (2026-09-03) : la progression n'était loggée qu'une tuile
sur 10, donc jamais « 25/25 » — la ligne transiente du run restait figée
sur « analyse 20/25 » alors que le modèle suivant démarrait déjà.
La dernière tuile doit toujours être annoncée.
"""

from __future__ import annotations

import pytest

from pipeline.cv.computer_vision_onnx import _tile_progress_due


@pytest.mark.parametrize(
    "done, total, expected",
    [
        (10, 144, True),   # cadence régulière conservée
        (20, 25, True),
        (21, 25, False),   # pas de spam entre deux multiples de 10
        (25, 25, True),    # dernière tuile : toujours émise (le bug)
        (7, 7, True),      # image plus petite que la cadence
        (1, 1, True),
    ],
)
def test_tile_progress_due(done, total, expected):
    assert _tile_progress_due(done, total) is expected
