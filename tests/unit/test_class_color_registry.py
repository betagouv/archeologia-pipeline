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


# ---------------------------------------------------------------- alias (N-01, 2026-10-25)

def test_alias_conserve_la_couleur_de_l_ancien_nom(tmp_path):
    """Une couche renommée par la structure (« Parcellaire ») garde la couleur que « parcellaire »
    avait : l'alias pointe sur l'ancien rang, le registre reste append-only."""
    reg = _reg(tmp_path)
    reg.rank_for("cratere")
    reg.rank_for("talus")
    assert reg.alias("Cratères", "cratere") is True
    assert reg.rank_for("Cratères") == reg.rank_for("cratere") == 0
    assert reg.color_for("Cratères") == reg.color_for("cratere")
    assert reg.alias("Cratères", "talus") is False               # déjà résolu : on ne bouge plus
    assert reg.alias("Inconnue", "jamais_vue") is False          # l'ancien nom est inconnu : rien
    assert reg.alias("talus", "cratere") is False                # un nom qui a son rang le garde
    assert reg.alias("Parcellaire", "parcellaire") is False      # même clé normalisée : rien à faire
    assert reg.rank_for("talus") == 1 and len(json.loads(reg.path.read_text(encoding="utf-8"))["classes"]) == 2
    # Persistance : relu depuis le JSON, l'alias tient ; les classes n'ont pas bougé.
    d = json.loads(reg.path.read_text(encoding="utf-8"))
    assert d["classes"] == ["cratere", "talus"] and d["alias"] == {"cratères": "cratere"}
    relu = _reg(tmp_path)
    assert relu.rank_for("Cratères") == 0 and relu.rank_for("talus") == 1
    # Depuis les runs : une couche nommée par sa structure prend la couleur de sa classe,
    # jamais une variante A/B (group_label), qui garde sa clé propre.
    from pipeline.cv.class_color_registry import aliaser_depuis_runs, set_default_registry
    set_default_registry(relu)
    try:
        runs = [{"entities": [{"slug": "talus", "layer_names": {"talus": "Talus et fossés"}},
                              {"slug": "cratere--m2", "group_label": "Cratères (comparaison)",
                               "layer_names": {"cratere": "Cratères — Modèle 2"}}]}, "pas un run"]
        assert aliaser_depuis_runs(runs) == 1
        assert relu.rank_for("Talus et fossés") == 1
        assert relu.rank_for("Cratères — Modèle 2") == 2        # rang propre, pas un alias
    finally:
        set_default_registry(None)
