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


class TestOwnedByCell:
    """Règle du centroïde : chaque dalle ne rapporte que les objets dont le
    centre est dans SA cellule. Avec le halo, un objet à cheval est vu entier
    par les deux dalles (doublon) et un objet au bord du halo n'est vu qu'en
    fragment (coupe rectiligne à ± marge) : une seule dalle le possède, celle
    qui le voit entier."""

    def test_centre_dans_la_cellule(self):
        from pipeline.cv.postprocessing import owned_by_cell
        assert owned_by_cell(_box(900, 100, 1050, 200), _CELLS[0]) is True   # centre x=975
        assert owned_by_cell(_box(900, 100, 1050, 200), _CELLS[1]) is False

    def test_fragment_au_bord_du_halo_non_possede(self):
        from pipeline.cv.postprocessing import owned_by_cell
        # Vu par la dalle 0 dans son halo (x 1000..1050), centre à x=1030 → dalle 1.
        assert owned_by_cell(_box(1010, 100, 1050, 200), _CELLS[0]) is False

    def test_centre_sur_la_frontiere_appartient_a_une_seule_cellule(self):
        from pipeline.cv.postprocessing import owned_by_cell
        geom = _box(950, 100, 1050, 200)  # centre exactement x=1000
        owners = [owned_by_cell(geom, c) for c in _CELLS]
        assert owners == [False, True]  # intervalle semi-ouvert [xmin, xmax[

    def test_geometrie_vide_ou_invalide_conservee(self):
        from pipeline.cv.postprocessing import owned_by_cell
        assert owned_by_cell(None, _CELLS[0]) is True
        assert owned_by_cell(Polygon(), _CELLS[0]) is True
