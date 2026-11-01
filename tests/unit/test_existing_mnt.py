from __future__ import annotations

from pathlib import Path

from pipeline.modes.existing_mnt import _classify_mnt_layout, _large_tile_name_for, _motif_petite_emprise


class TestClassifyMntLayout:
    def test_standard_when_one_kilometer_and_aligned(self):
        assert _classify_mnt_layout((700000, 6599000, 701000, 6600000)) == "standard"

    def test_small_when_smaller_than_ign_tile(self):
        assert _classify_mnt_layout((700000, 6599500, 700500, 6600000)) == "small"

    def test_small_when_one_kilometer_but_not_aligned(self):
        assert _classify_mnt_layout((700120, 6599120, 701120, 6600120)) == "small"

    def test_large_when_dimension_exceeds_tile_plus_tolerance(self):
        assert _classify_mnt_layout((700000, 6598000, 702000, 6600000)) == "large"


class TestLargeTileNameFor:
    def test_uses_north_west_kilometer_and_sanitized_source_stem(self):
        name = _large_tile_name_for(
            Path("site archéo_MNT.tif"),
            (700123.0, 6597000.0, 702456.0, 6600123.0),
        )

        assert name == "LHD_FXX_0700_6600_EXT_site_arch_o"


def test_motif_d_une_petite_emprise_dit_la_vraie_cause():
    """de_sh 2026-11-01 : « emprise < 1 km (ou non alignée IGN) » affiché pour un MNT de 1 011 m non aligné."""
    assert _motif_petite_emprise((455300.0, 6056000.0, 456311.0, 6057011.0)) ==         "emprise 1 011 × 1 011 m non alignée sur la grille de 1 km"
    assert _motif_petite_emprise((455300.0, 6056000.0, 455651.0, 6056351.0)) ==         "emprise 351 × 351 m, plus petite qu'une dalle de 1 km"

