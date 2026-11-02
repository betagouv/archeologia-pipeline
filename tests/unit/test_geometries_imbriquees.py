"""Détection perdue sans trace (crash test gb_eng, 2026-11-02) : un enclos « probable » (0,59, ~30 m²) dont la
réparation ``make_valid`` rendait une GeometryCollection contenant une MultiPolygon était jeté — seuls les Polygon
du premier niveau étaient regardés — et le journal partait du compte déjà filtré (« 111 brutes » puis « 110 -> 87 »)."""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("shapely")

import shapely.validation  # noqa: E402
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon  # noqa: E402

from pipeline.cv.postprocessing import _validate_class_dets, postprocess_geo_detections  # noqa: E402

NOEUD = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])      # papillon : invalide


def _carre(x0, cote=6.0):
    return Polygon([(x0, 0), (x0 + cote, 0), (x0 + cote, cote), (x0, cote)])


def _imbriquee(_g):
    """Ce que make_valid a rendu pour l'enclos perdu : GeometryCollection[MultiPolygon, LineString]."""
    return GeometryCollection([MultiPolygon([_carre(0), _carre(20, cote=1.0)]), LineString([(0, 0), (1, 1)])])


def test_collection_imbriquee_garde_la_detection(caplog):
    """Une détection reste une détection (enclos sans fusion) : le plus grand polygone, cherché en profondeur ;
    le fragment laissé est dit au journal."""
    det = {"geometry": NOEUD, "confidence": 0.59}
    with caplog.at_level(logging.INFO, logger="pipeline.cv.postprocessing"):
        valides = _validate_class_dets([det], _imbriquee, 0.0, "enclos")
    assert len(valides) == 1 and valides[0]["confidence"] == 0.59
    assert valides[0]["geometry"].geom_type == "Polygon" and valides[0]["geometry"].area == 36
    assert "1 fragment(s)" in caplog.text and "1.0 m²" in caplog.text


def test_geometrie_ecartee_journalisee_et_compte_depuis_le_brut(monkeypatch, caplog):
    monkeypatch.setattr(shapely.validation, "make_valid",
                        lambda g: _imbriquee(g) if g.equals(NOEUD) else LineString([(0, 0), (1, 1)]))
    dets = {"enclos": [{"geometry": NOEUD, "confidence": 0.59},
                       {"geometry": Polygon([(50, 0), (60, 10), (60, 0), (50, 10)]), "confidence": 0.4},
                       {"geometry": _carre(100), "confidence": 0.7}]}
    with caplog.at_level(logging.INFO, logger="pipeline.cv.postprocessing"):
        out = postprocess_geo_detections(dets, do_merge=False, overlap_strategy="relation")
    assert len(out["enclos"]) == 2                       # le papillon réparé + le carré
    texte = caplog.text
    assert "écartée" in texte and "enclos" in texte and "0.40" in texte    # la seule vraiment perdue, dite
    assert "validation des géométries 3 -> 2" in texte
