from __future__ import annotations

from pathlib import Path

from pipeline.modes.existing_mnt import _bilan_marge, _classify_mnt_layout, _large_tile_name_for, _motif_petite_emprise


class TestClassifyMntLayout:
    def test_standard_when_one_kilometer_and_aligned(self):
        assert _classify_mnt_layout((700000, 6599000, 701000, 6600000)) == "standard"

    def test_small_when_smaller_than_ign_tile(self):
        assert _classify_mnt_layout((700000, 6599500, 700500, 6600000)) == "small"

    def test_small_when_one_kilometer_but_not_aligned(self):
        assert _classify_mnt_layout((700120, 6599120, 701120, 6600120)) == "small"

    def test_large_when_dimension_exceeds_tile_plus_tolerance(self):
        assert _classify_mnt_layout((700000, 6598000, 702000, 6600000)) == "large"

    def test_vraie_dalle_lidar_hd_ign_reste_standard(self):
        """Dalle LHD réelle (D:/veille_irlande/corpus_enclos_fr/dalles_ign) : 2 000 px de 0,5 m, coins à
        ½ pixel de la grille (217 999,75 / 6 866 000,25) → le découpage à la cellule ne perd aucun pixel."""
        assert _classify_mnt_layout((217999.75, 6865000.25, 218999.75, 6866000.25), pixel_m=0.5) == "standard"
        assert _classify_mnt_layout((410000.0, 140000.0, 411000.0, 141000.0), pixel_m=1.0) == "standard"

    def test_fenetre_proche_de_la_grille_n_est_pas_une_dalle(self):
        """Point 32 (gb_sct, mnt_canmore_41004.tif) : fenêtre de 1 010 m dont le coin tombe à 18-26 m de la
        grille, classée « standard » par la tolérance de 50 m puis rognée à la cellule km : 6,2 % du terrain
        perdu, bandes NoData. Décision : dalle seulement si le rognage ne perd ni n'ajoute aucun pixel."""
        assert _classify_mnt_layout((227018.0, 689026.0, 228028.0, 690036.0), pixel_m=0.5) == "small"
        assert _classify_mnt_layout((367045.0, 4961965.0, 368060.0, 4962980.0), pixel_m=5.0) == "small"
        # 1 000 m pile mais décalée d'un pixel, ou alignée mais d'un pixel trop large : pas une dalle non plus.
        assert _classify_mnt_layout((700000.5, 6599000.5, 701000.5, 6600000.5), pixel_m=0.5) == "small"
        assert _classify_mnt_layout((700000.0, 6599000.0, 701000.5, 6600000.0), pixel_m=0.5) == "small"


class TestLargeTileNameFor:
    def test_uses_north_west_kilometer_and_sanitized_source_stem(self):
        name = _large_tile_name_for(
            Path("site archéo_MNT.tif"),
            (700123.0, 6597000.0, 702456.0, 6600123.0),
        )

        assert name == "LHD_FXX_0700_6600_EXT_site_arch_o"


def test_bilan_de_marge_compte_tout_le_lot():
    """Point 33 : « 0 dalle(s) étendue(s) de 50 m depuis leurs voisines, 1 sans voisin » pour un lot de 34 MNT dont
    33 fenêtres hors grille : la ligne ne comptait que les dalles standard."""
    assert _bilan_marge(0, 1, 33) == ("Marge inter-dalles : 1 dalle de 1 km sans voisine ; "
                                      "33 MNT hors grille du km, traités seuls")
    assert _bilan_marge(0, 0, 1) == "Marge inter-dalles : aucune dalle de 1 km sur la grille ; 1 MNT hors grille du km, traité seul"
    assert _bilan_marge(4, 0, 0) == "Marge inter-dalles : 4 dalles étendues de 50 m depuis leurs voisines"
    assert _bilan_marge(1, 2, 0) == ("Marge inter-dalles : 1 dalle étendue de 50 m depuis ses voisines ; "
                                     "2 dalles de 1 km sans voisine")
    assert _bilan_marge(0, 0, 0) == ""


def test_motif_d_une_petite_emprise_dit_la_vraie_cause():
    """de_sh 2026-11-01 : « emprise < 1 km (ou non alignée IGN) » affiché pour un MNT de 1 011 m non aligné."""
    assert _motif_petite_emprise((455300.0, 6056000.0, 456311.0, 6057011.0)) ==         "emprise 1 011 × 1 011 m non alignée sur la grille de 1 km"
    assert _motif_petite_emprise((455300.0, 6056000.0, 455651.0, 6056351.0)) ==         "emprise 351 × 351 m, plus petite qu'une dalle de 1 km"

