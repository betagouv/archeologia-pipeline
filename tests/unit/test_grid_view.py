"""Tests des helpers purs de gestion de la couche quadrillage (hors QGIS).

Le rendu de la couche, l'arbre de couches et la barre de message vivent dans
``src/ui/map_tools/`` (non collecté par pytest). On isole ici les **décisions** —
réutilisation d'une couche existante, masquage à l'échelle — dans
``app.services.grid_view`` (pur) pour les couvrir sans QGIS.
"""
from __future__ import annotations

from app.services.grid_view import (
    FRANCE_METRO_2154_BBOX,
    decide_grid_reuse,
    grid_is_hidden,
)

_MIN_SCALE = 1_500_000.0


class TestDecideGridReuse:
    def test_valid_and_in_tree_is_reuse(self):
        assert decide_grid_reuse(is_valid=True, in_tree=True) == "reuse"

    def test_valid_but_orphan_is_readd(self):
        # Présente au registre mais absente de l'arbre → ne se dessine jamais :
        # c'est le cœur du bug « grille absente, réparée par redémarrage ».
        assert decide_grid_reuse(is_valid=True, in_tree=False) == "readd"

    def test_invalid_is_reload(self):
        assert decide_grid_reuse(is_valid=False, in_tree=True) == "reload"
        assert decide_grid_reuse(is_valid=False, in_tree=False) == "reload"


class TestGridIsHidden:
    def test_hidden_when_too_zoomed_out(self):
        # canvas_scale > min_scale = dénominateur plus grand = plus dézoomé.
        assert grid_is_hidden(True, canvas_scale=3_000_000.0, min_scale=_MIN_SCALE) is True

    def test_visible_when_zoomed_in_enough(self):
        assert grid_is_hidden(True, canvas_scale=200_000.0, min_scale=_MIN_SCALE) is False

    def test_at_threshold_is_visible(self):
        # Égalité = visible (le masquage est strictement « au-delà »).
        assert grid_is_hidden(True, canvas_scale=_MIN_SCALE, min_scale=_MIN_SCALE) is False

    def test_never_hidden_without_scale_visibility(self):
        assert grid_is_hidden(False, canvas_scale=9_000_000.0, min_scale=_MIN_SCALE) is False


class TestFranceMetroBbox:
    def test_bbox_is_ordered(self):
        xmin, ymin, xmax, ymax = FRANCE_METRO_2154_BBOX
        assert xmin < xmax and ymin < ymax

    def test_bbox_within_lambert93_metropolitan_domain(self):
        # Bornes plausibles de la métropole en L93 (m) : ~[0..1300k] x [6000k..7200k].
        xmin, ymin, xmax, ymax = FRANCE_METRO_2154_BBOX
        assert 0.0 <= xmin and xmax <= 1_300_000.0
        assert 6_000_000.0 <= ymin and ymax <= 7_200_000.0
