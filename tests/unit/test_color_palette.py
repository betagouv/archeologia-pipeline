"""Refonte couleurs (spec 2026-06-12) : attribution déterministe et stable
d'une couleur de base PAR CLASSE, dérivée du nom — remplace la palette fixe
de 12 couleurs + le mapping par index (source des collisions, ex. deux verts).

Module pur (hashlib + colorsys) → testable hors QGIS.
"""
from __future__ import annotations

import math

from pipeline.cv.color_palette import (
    base_color_for_class,
    base_color_for_rank,
    apply_confidence,
)


def _is_rgb(c):
    return (
        isinstance(c, tuple)
        and len(c) == 3
        and all(isinstance(v, int) and 0 <= v <= 255 for v in c)
    )


def _hue_deg(rgb):
    import colorsys
    r, g, b = (v / 255 for v in rgb)
    h, _l, _s = colorsys.rgb_to_hls(r, g, b)
    return h * 360.0


def _hue_dist(a, b):
    d = abs(_hue_deg(a) - _hue_deg(b)) % 360.0
    return min(d, 360.0 - d)


class TestBaseColorForRank:
    """Voie nominale : couleur par rang stable (alimenté par le registre)."""

    def test_retourne_un_rgb_valide(self):
        assert _is_rgb(base_color_for_rank(0))

    def test_deterministe(self):
        assert base_color_for_rank(7) == base_color_for_rank(7)

    def test_rangs_consecutifs_perceptiblement_distincts(self):
        # Nombre d'or sur rangs 0..N → distinction garantie. On vérifie la
        # distance euclidienne RGB (proxy ΔE) sur un jeu réaliste de 12 classes.
        cols = [base_color_for_rank(i) for i in range(12)]
        mind = min(
            math.dist(a, b)
            for i, a in enumerate(cols) for b in cols[i + 1:]
        )
        assert mind > 40.0

    def test_rang_negatif_borne(self):
        assert _is_rgb(base_color_for_rank(-5))


class TestBaseColorForClassFallback:
    """Repli sans registre : stable et déterministe, mais non garanti distinct."""

    def test_retourne_un_rgb_valide(self):
        assert _is_rgb(base_color_for_class("cratere"))

    def test_deterministe(self):
        assert base_color_for_class("cratere") == base_color_for_class("cratere")

    def test_insensible_casse_et_espaces(self):
        assert base_color_for_class("Cratere") == base_color_for_class("  cratere ")

    def test_nom_vide_ne_leve_pas(self):
        assert _is_rgb(base_color_for_class(""))


class TestApplyConfidence:
    BASE = (40, 160, 220)

    def test_confiance_moyenne_rend_la_couleur_de_base(self):
        assert apply_confidence(self.BASE, 0.5) == self.BASE

    def test_haute_confiance_plus_sombre_que_basse(self):
        haute = apply_confidence(self.BASE, 0.9)
        basse = apply_confidence(self.BASE, 0.1)
        assert sum(haute) < sum(self.BASE) < sum(basse)

    def test_monotonie_par_paliers(self):
        # Plus la confiance monte, plus c'est sombre (somme RGB décroissante).
        sommes = [sum(apply_confidence(self.BASE, c)) for c in (0.1, 0.3, 0.5, 0.7, 0.9)]
        assert sommes == sorted(sommes, reverse=True)

    def test_borne_les_valeurs(self):
        for conf in (-1.0, 0.0, 0.5, 1.0, 5.0):
            assert _is_rgb(apply_confidence(self.BASE, conf))
