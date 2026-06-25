"""Registre de couleurs de classes (spec 2026-06-12).

Mappe chaque nom de classe à un **rang stable** attribué à sa première
apparition (append-only), persisté en JSON dans le dossier de profil. Le rang
alimente ``base_color_for_rank`` → couleurs réparties et stables dans le temps.

Module pur (I/O fichier, pas de QGIS) → testable hors QGIS.
"""
from __future__ import annotations

import json
import math

import pytest

from pipeline.cv.class_color_registry import (
    ClassColorRegistry,
    color_for_class,
    rank_for_class,
    set_default_registry,
)


def _reg(tmp_path):
    return ClassColorRegistry(tmp_path / "class_color_registry.json")


@pytest.fixture
def shared_registry(tmp_path):
    """Override du singleton vers un fichier temporaire (évite d'écrire dans le
    profil/plugin pendant les tests, et isole l'état entre tests)."""
    set_default_registry(_reg(tmp_path))
    try:
        yield
    finally:
        set_default_registry(None)


class TestPointsDAccesPartages:
    """color_for_class / rank_for_class : source unique génération + affichage."""

    def test_meme_classe_meme_couleur_des_deux_points(self, shared_registry):
        # Génération (rank_for_class) et affichage (color_for_class) doivent
        # donner une couleur cohérente pour une même classe via le registre partagé.
        from pipeline.cv.color_palette import base_color_for_rank
        assert color_for_class("cratere") == base_color_for_rank(rank_for_class("cratere"))


class TestRangs:
    def test_rang_attribue_dans_l_ordre_de_decouverte(self, tmp_path):
        r = _reg(tmp_path)
        assert r.rank_for("cratere") == 0
        assert r.rank_for("tranchee") == 1
        assert r.rank_for("parcellaire") == 2

    def test_meme_classe_meme_rang(self, tmp_path):
        r = _reg(tmp_path)
        r.rank_for("cratere")
        r.rank_for("tranchee")
        assert r.rank_for("cratere") == 0  # inchangé

    def test_normalise_casse_et_espaces(self, tmp_path):
        r = _reg(tmp_path)
        assert r.rank_for("Cratere") == r.rank_for("  cratere ")


class TestStabilite:
    def test_ajouter_une_classe_ne_decale_pas_les_existantes(self, tmp_path):
        r = _reg(tmp_path)
        r.rank_for("cratere")
        r.rank_for("tranchee")
        couleur_cratere = r.color_for("cratere")
        r.rank_for("classe_ajoutee_plus_tard")
        assert r.color_for("cratere") == couleur_cratere

    def test_persistance_entre_sessions(self, tmp_path):
        r1 = _reg(tmp_path)
        r1.rank_for("cratere")
        r1.rank_for("tranchee")
        # Nouvelle instance lisant le même fichier : rangs conservés.
        r2 = _reg(tmp_path)
        assert r2.rank_for("tranchee") == 1
        assert r2.rank_for("cratere") == 0
        # Une nouvelle classe prend le rang suivant, sans réordonner.
        assert r2.rank_for("parcellaire") == 2


class TestCouleurs:
    def test_color_for_deterministe(self, tmp_path):
        r = _reg(tmp_path)
        assert r.color_for("cratere") == r.color_for("cratere")

    def test_classes_du_meme_lot_distinctes(self, tmp_path):
        r = _reg(tmp_path)
        classes = [
            "cratere", "tranchee", "parcellaire", "charbonniere", "zone_crateres",
            "tumulus", "enclos", "voie", "batiment", "fosse",
        ]
        cols = [r.color_for(c) for c in classes]
        assert len(set(cols)) == len(classes)
        mind = min(
            math.dist(a, b) for i, a in enumerate(cols) for b in cols[i + 1:]
        )
        assert mind > 40.0


class TestRobustesse:
    def test_fichier_absent_demarre_vide(self, tmp_path):
        r = _reg(tmp_path)
        assert r.rank_for("x") == 0

    def test_fichier_corrompu_ne_leve_pas(self, tmp_path):
        path = tmp_path / "class_color_registry.json"
        path.write_text("{ pas du json", encoding="utf-8")
        r = ClassColorRegistry(path)
        assert r.rank_for("x") == 0  # repart proprement

    def test_ecriture_atomique_sans_tmp_residuel(self, tmp_path):
        r = _reg(tmp_path)
        r.rank_for("cratere")
        leftovers = [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"]
        assert leftovers == []

    def test_format_persiste_lisible(self, tmp_path):
        r = _reg(tmp_path)
        r.rank_for("cratere")
        r.rank_for("tranchee")
        data = json.loads((tmp_path / "class_color_registry.json").read_text("utf-8"))
        # l'ordre encode le rang (cratere avant tranchee)
        classes = data["classes"] if isinstance(data, dict) else data
        assert classes.index("cratere") < classes.index("tranchee")
