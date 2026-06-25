"""Tests du calcul de couverture (module pur numpy)."""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.coverage_math import (
    COVERAGE_NODATA,
    DISC_DIAMETER_CELLS,
    compute_coverage_percent,
    disc_offsets,
)


class TestDiscOffsets:
    def test_diametre_5_donne_13_cellules(self):
        # Le « /13 » de PCSAPS : disque euclidien de rayon 2 (dy²+dx² ≤ 4).
        assert len(disc_offsets(5)) == 13

    def test_contient_le_centre_et_les_croix(self):
        offs = set(disc_offsets(5))
        assert (0, 0) in offs
        assert (2, 0) in offs and (0, -2) in offs

    def test_exclut_les_coins(self):
        offs = set(disc_offsets(5))
        assert (2, 2) not in offs and (2, 1) not in offs


class TestComputeCoveragePercent:
    def test_tout_rempli_donne_100_partout(self):
        density = np.ones((8, 8), dtype=np.float32)
        out = compute_coverage_percent(density)
        assert out.dtype == np.uint8
        # Normalisation par le nombre réel de cellules : 100 % jusque dans les coins.
        assert (out == 100).all()

    def test_tout_vide_donne_0_partout(self):
        out = compute_coverage_percent(np.zeros((8, 8), dtype=np.float32))
        assert (out == 0).all()

    def test_point_isole_au_centre(self):
        density = np.zeros((9, 9), dtype=np.float32)
        density[4, 4] = 3.0
        out = compute_coverage_percent(density)
        # 1 cellule remplie / 13 → round(7.69) = 8.
        assert out[4, 4] == 8

    def test_normalisation_au_bord(self):
        # Raster 1x1 rempli : fenêtre tronquée à 1 cellule → 1/1 = 100 %.
        out = compute_coverage_percent(np.array([[5.0]], dtype=np.float32))
        assert out[0, 0] == 100

    def test_nodata_compte_comme_sans_points(self):
        density = np.full((5, 5), -9999.0, dtype=np.float32)
        out = compute_coverage_percent(density, density_nodata=-9999.0)
        assert (out == 0).all()

    def test_nodata_positif_exclu_explicitement(self):
        # Un nodata > 0 (ex. 65535) ne doit PAS compter comme « avec points ».
        density = np.full((5, 5), 65535.0, dtype=np.float32)
        out = compute_coverage_percent(density, density_nodata=65535.0)
        assert (out == 0).all()

    def test_nan_compte_comme_sans_points(self):
        density = np.full((5, 5), np.nan, dtype=np.float32)
        assert (compute_coverage_percent(density) == 0).all()

    def test_jamais_de_nodata_en_sortie(self):
        # (0,0) est toujours dans la fenêtre → total ≥ 1 partout.
        out = compute_coverage_percent(np.zeros((4, 7), dtype=np.float32))
        assert COVERAGE_NODATA not in out

    def test_rejette_les_tableaux_non_2d(self):
        with pytest.raises(ValueError):
            compute_coverage_percent(np.zeros(10, dtype=np.float32))

    def test_constantes(self):
        assert DISC_DIAMETER_CELLS == 5
        assert COVERAGE_NODATA == 255
