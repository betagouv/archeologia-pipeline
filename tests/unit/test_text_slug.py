"""Tests pour ``app.text_slug.slugify``.

Helper pur (aucune dépendance QGIS) qui transforme un libellé d'entité FR en
nom de dossier sûr pour le système de fichiers : repli des accents (NFKD →
ASCII), runs de caractères non alphanumériques → ``_``, minuscules. Sert à
nommer les dossiers ``detections/<slug>/`` à partir du libellé présentable.
"""
from __future__ import annotations

from app.text_slug import slugify


class TestSlugify:
    def test_folds_french_accents(self):
        assert slugify("Dépressions circulaires") == "depressions_circulaires"
        assert slugify("Charbonnières") == "charbonnieres"
        assert slugify("Talus et fossés") == "talus_et_fosses"

    def test_apostrophes_and_spaces_become_underscores(self):
        assert slugify("Trous d'obus") == "trous_d_obus"
        assert slugify("Zones d'extraction de matériaux") == "zones_d_extraction_de_materiaux"

    def test_simple_label_lowercased(self):
        assert slugify("Parcellaire") == "parcellaire"
        assert slugify("Chemins creux") == "chemins_creux"

    def test_collapses_runs_and_strips_edges(self):
        assert slugify("  Talus // fossés  ") == "talus_fosses"
        assert slugify("A---B") == "a_b"

    def test_empty_or_punctuation_only_returns_empty(self):
        assert slugify("") == ""
        assert slugify("***") == ""
        assert slugify("   ") == ""
