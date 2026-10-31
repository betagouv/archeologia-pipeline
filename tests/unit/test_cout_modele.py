"""Coût structurel d'un modèle : fenêtres d'analyse par dalle (étape 3).

Le module ``app.services.cout_modele`` est PUR : il calcule, à partir du
découpage SAHI d'``args.yaml`` (``slice``, ``overlap``), le nombre de fenêtres
que le modèle analyse sur une dalle — un FAIT, jamais une durée. Les valeurs
sont épinglées sur ``pipeline.cv.sahi_lite.get_slice_bboxes`` (la fonction que
le binaire exécute) : la déduplication des fenêtres de bord donne 25 et non 36
à 672 px sur 2 800 px, une formule maison se tromperait.
"""
from __future__ import annotations

from app.services.cout_modele import (
    COTE_DALLE_MARGE_PX,
    COTE_DALLE_PX,
    fenetres_par_dalle,
    infobulle,
    libelle_menu,
    ligne_dialogue,
)


class TestFenetresParDalle:
    def test_dalle_standard_2000_px(self):
        assert COTE_DALLE_PX == 2000
        assert fenetres_par_dalle(252, 0.2) == 100
        assert fenetres_par_dalle(648, 0.2) == 16
        assert fenetres_par_dalle(672, 0.2) == 16

    def test_dalle_avec_marge_2800_px(self):
        assert COTE_DALLE_MARGE_PX == 2800
        assert fenetres_par_dalle(252, 0.2, COTE_DALLE_MARGE_PX) == 196
        assert fenetres_par_dalle(648, 0.2, COTE_DALLE_MARGE_PX) == 36
        # Fenêtre de bord ramenée en arrière puis dédupliquée : 25, pas 36.
        assert fenetres_par_dalle(672, 0.2, COTE_DALLE_MARGE_PX) == 25

    def test_identique_a_get_slice_bboxes(self):
        from pipeline.cv.sahi_lite import get_slice_bboxes

        for cote in (1000, 2000, 2200, 2800):
            for fen in (140, 252, 648, 672):
                attendu = len(get_slice_bboxes(cote, cote, fen, fen, 0.2, 0.2))
                assert fenetres_par_dalle(fen, 0.2, cote) == attendu

    def test_fenetre_inconnue_vaut_zero(self):
        assert fenetres_par_dalle(0, 0.2) == 0
        assert fenetres_par_dalle(-5, 0.2) == 0


class TestLibelles:
    def test_libelle_menu(self):
        assert libelle_menu("Modèle cratères LD v1", 252, 0.2) == (
            "Modèle cratères LD v1 — 100 fenêtres d'analyse par dalle"
        )

    def test_libelle_menu_sans_sahi_garde_le_nom_seul(self):
        assert libelle_menu("Modèle X", 0, 0.0) == "Modèle X"

    def test_ligne_dialogue_donne_les_deux_tailles(self):
        assert ligne_dialogue(252, 0.2) == (
            "252 px, recouvrement 20 % → 100 fenêtres par dalle de 1 km "
            "(2 000 px à 0,5 m), 196 avec la marge inter-dalles"
        )

    def test_infobulle_dit_le_fait_et_refuse_la_duree(self):
        texte = infobulle("Modèle cratères LD v1", 252, 0.2)
        assert texte.startswith("Modèle cratères LD v1 : ")
        assert "100 fenêtres de 252 px" in texte
        assert "196 avec la marge inter-dalles" in texte
        assert "deux fois plus de fenêtres = deux fois plus de calcul" in texte
        assert "Aucune durée n'est annoncée" in texte

    def test_infobulle_vide_sans_sahi(self):
        assert infobulle("Modèle X", 0, 0.0) == ""

    def test_recouvrement_non_numerique_vaut_zero(self):
        # args.yaml lu tel quel par le dialogue : une faute de saisie ne lève pas.
        assert fenetres_par_dalle(648, "vingt") == 0


class TestEchelle:
    """Audit 5 (2026-10-31) : la fenêtre comptée est celle que le run découpera, mise à l'échelle du modèle."""

    def test_enclos_en_france_4_et_9_fenetres(self):
        # Appris à 1 m, fenêtre 648 m de terrain = 1 296 px à 0,5 m : 4 fenêtres sur 2 000 px, 9 sur 2 800.
        assert libelle_menu("Enclos", 648, 0.2, gsd_m=1.0, resolution_m=0.5) == (
            "Enclos — 4 fenêtres d'analyse par dalle")
        assert ligne_dialogue(648, 0.2, gsd_m=1.0, resolution_m=0.5) == (
            "1 296 px (648 m de terrain), recouvrement 20 % → 4 fenêtres par dalle de 1 km "
            "(2 000 px à 0,5 m), 9 avec la marge inter-dalles")

    def test_modele_a_05_m_sur_mnt_a_1_m(self):
        # 324 px à 1 m ; dalle de 1 km = 1 000 px : 16 fenêtres, comme 648 px sur 2 000 px à 0,5 m.
        assert "16 fenêtres par dalle de 1 km (1 000 px à 1 m)" in ligne_dialogue(648, 0.2, gsd_m=0.5,
                                                                                 resolution_m=1.0)

    def test_resolution_inconnue_dite(self):
        t = ligne_dialogue(648, 0.2, gsd_m=1.0)
        assert "4 fenêtres" in t and "résolution des données pas encore connue" in t

    def test_sans_gsd_comme_avant(self):
        assert ligne_dialogue(252, 0.2, resolution_m=1.0) == ligne_dialogue(252, 0.2)
