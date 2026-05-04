"""Tests du noyau de fusion géométrique partagé (V2.3).

Avant V2.3, l'incantation ``buffer → unary_union → debuffer → extract``
était dupliquée mot pour mot dans ``postprocessing._merge_touching_same_class``
et ``computer_vision_onnx._merge_adjacent_polygons``. Une fix dans l'une
ne profitait pas à l'autre.

V2.3 a extrait :func:`pipeline.cv.postprocessing.buffer_union_debuffer`
comme noyau pur — testé ici. Les wrappers métier (filtrage par
compactness, confiance pondérée par aire, etc.) restent dans leurs
modules respectifs et sont hors scope de ce fichier.
"""
from __future__ import annotations

import pytest

shapely = pytest.importorskip("shapely")
from shapely.geometry import Polygon  # noqa: E402

from pipeline.cv.postprocessing import buffer_union_debuffer  # noqa: E402


# ----------------------------------------------------------------------
# Cas dégénérés
# ----------------------------------------------------------------------
class TestEdgeCases:
    def test_empty_input_returns_empty(self):
        assert buffer_union_debuffer([], 1.0) == []

    def test_single_polygon_returns_single(self):
        p = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        result = buffer_union_debuffer([p], 0.5)
        assert result is not None
        assert len(result) == 1


# ----------------------------------------------------------------------
# Comportement central : ce qui se touche fusionne, ce qui est éloigné non
# ----------------------------------------------------------------------
class TestCorePolicies:
    def test_disjoint_polygons_stay_separate(self):
        """Deux carrés très éloignés ne doivent pas fusionner."""
        a = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        b = Polygon([(100, 100), (110, 100), (110, 110), (100, 110)])
        result = buffer_union_debuffer([a, b], 1.0)
        assert result is not None
        assert len(result) == 2

    def test_touching_polygons_merge(self):
        """Deux carrés qui se touchent fusionnent en un."""
        a = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        b = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])
        result = buffer_union_debuffer([a, b], 1.0)
        assert result is not None
        assert len(result) == 1

    def test_overlapping_polygons_merge(self):
        """Deux carrés qui se chevauchent fusionnent."""
        a = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        b = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])
        result = buffer_union_debuffer([a, b], 0.5)
        assert result is not None
        assert len(result) == 1

    def test_close_but_not_touching_merge_with_large_buffer(self):
        """Le buffer permet de combler un petit gap (artefact pixellisation)."""
        a = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        b = Polygon([(11, 0), (20, 0), (20, 10), (11, 10)])  # gap de 1px
        # Avec un buffer de 1px on doit pouvoir combler.
        result = buffer_union_debuffer([a, b], 1.5)
        assert result is not None
        assert len(result) == 1

    def test_close_polygons_stay_separate_with_small_buffer(self):
        """Sans buffer suffisant, le gap reste — pas de fusion."""
        a = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        b = Polygon([(11, 0), (20, 0), (20, 10), (11, 10)])  # gap de 1px
        # Buffer 0.1 < 0.5 du gap → reste séparé.
        result = buffer_union_debuffer([a, b], 0.1)
        assert result is not None
        assert len(result) == 2


# ----------------------------------------------------------------------
# Préservation de la sémantique historique (join_style=2 = mitre)
# ----------------------------------------------------------------------
class TestSemantics:
    def test_default_join_style_is_mitre(self):
        """``join_style=2`` (mitre) est le défaut historique des deux call-sites."""
        import inspect

        sig = inspect.signature(buffer_union_debuffer)
        assert sig.parameters["join_style"].default == 2

    def test_buffer_size_is_respected(self):
        """Le ``buffer_px`` passé est bien le rayon utilisé."""
        # Avec un grand buffer, deux polygones distants peuvent être unis.
        a = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        b = Polygon([(20, 0), (25, 0), (25, 5), (20, 5)])  # gap de 15px
        # Petit buffer : pas de fusion.
        small = buffer_union_debuffer([a, b], 1.0)
        assert small is not None and len(small) == 2
        # Grand buffer : fusion.
        big = buffer_union_debuffer([a, b], 10.0)
        assert big is not None and len(big) == 1


# ----------------------------------------------------------------------
# Output type contract
# ----------------------------------------------------------------------
class TestOutputContract:
    def test_returns_list_of_polygons(self):
        a = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        result = buffer_union_debuffer([a], 1.0)
        assert result is not None
        for g in result:
            assert g.geom_type == "Polygon"

    def test_returns_none_on_internal_error(self):
        """Un input invalide qui fait planter shapely retourne None pour
        signaler à l'appelant qu'il doit fallback sur ses originaux."""
        # On ne peut pas facilement déclencher un crash interne ici sans
        # mock. On vérifie au moins que la fonction n'explose pas si on
        # lui donne une géométrie auto-intersectée — cas réel rencontré
        # avec des masques de segmentation bruyants.
        bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
        # Soit fusion réussie, soit None — pas d'exception.
        result = buffer_union_debuffer([bowtie], 0.5)
        assert result is None or isinstance(result, list)
