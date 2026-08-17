"""Option B (halo inter-dalles) : clip des détections au périmètre du run.

L'image d'inférence à halo déborde de la dalle. Vers une dalle voisine du run,
le halo est de la vraie donnée (doublons fusionnés en aval) ; vers l'EXTÉRIEUR
du périmètre commandé, c'est du fabriqué (NoData blanc, miroirs de noyaux RVT)
→ toute détection y est du bruit. ``clip_detections_to_valid_region`` restreint
les géométries à l'union des emprises des TIF rognés du run.
"""
from __future__ import annotations

import pytest

pytest.importorskip("shapely")

from shapely.geometry import Polygon

from pipeline.cv.postprocessing import clip_detections_to_valid_region


# Deux cellules 1 km adjacentes : x 0..2000, y 0..1000.
_CELLS = [(0.0, 0.0, 1000.0, 1000.0), (1000.0, 0.0, 2000.0, 1000.0)]


def _det(geom, **extra):
    return {"geometry": geom, "confidence": 0.8, **extra}


def _box(x0, y0, x1, y1):
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


class TestClipDetectionsToValidRegion:
    def test_detection_interieure_intacte(self):
        geom = _box(100, 100, 200, 200)
        data = {"c": [_det(geom)]}

        out = clip_detections_to_valid_region(data, _CELLS)

        assert len(out["c"]) == 1
        assert out["c"][0]["geometry"].equals(geom)

    def test_detection_a_cheval_sur_deux_cellules_intacte(self):
        # Frontière interne x=1000 : l'union des cellules couvre l'objet entier.
        geom = _box(900, 100, 1100, 200)
        data = {"c": [_det(geom)]}

        out = clip_detections_to_valid_region(data, _CELLS)

        assert out["c"][0]["geometry"].equals(geom)

    def test_detection_entierement_hors_perimetre_supprimee(self):
        # Dans le halo extérieur (au-delà de x=2000) : donnée fabriquée.
        data = {"c": [_det(_box(2050, 100, 2150, 200))]}

        out = clip_detections_to_valid_region(data, _CELLS)

        assert out["c"] == []

    def test_detection_debordante_rognee_au_perimetre(self):
        # Moitié dedans, moitié dans le halo extérieur → rognée à x=2000.
        data = {"c": [_det(_box(1900, 100, 2100, 200))]}

        out = clip_detections_to_valid_region(data, _CELLS)

        clipped = out["c"][0]["geometry"]
        assert clipped.bounds == (1900.0, 100.0, 2000.0, 200.0)
        # Les attributs sont préservés.
        assert out["c"][0]["confidence"] == 0.8

    def test_sans_region_les_donnees_sont_intactes(self):
        geom = _box(5000, 5000, 5100, 5100)
        data = {"c": [_det(geom)]}

        assert clip_detections_to_valid_region(data, None) is data
        assert clip_detections_to_valid_region(data, []) is data

    def test_geometrie_invalide_conservee(self):
        # Papillon auto-intersectant : en cas d'échec du clip, on préfère
        # conserver la détection (comportement conservateur).
        bowtie = Polygon([(0, 0), (100, 100), (100, 0), (0, 100)])
        data = {"c": [_det(bowtie)]}

        out = clip_detections_to_valid_region(data, _CELLS)

        assert len(out["c"]) == 1
