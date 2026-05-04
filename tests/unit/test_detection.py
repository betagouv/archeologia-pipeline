"""Tests bijectifs pour le type Detection.

Garantit que :class:`Detection.to_*_dict` reproduit bit-pour-bit les formats
historiques produits/consommés par le pipeline CV (in-memory et on-disk),
afin que la migration V2.1 n'introduise aucune régression sur les sorties.
"""
from __future__ import annotations

import pytest

from pipeline.cv.types import Detection


# ----------------------------------------------------------------------
# Round-trip format in-memory (computer_vision_onnx)
# ----------------------------------------------------------------------
class TestInternalDictRoundTrip:
    def test_bbox_only_round_trip(self):
        raw = {
            "class_id": 3,
            "confidence": 0.87,
            "bbox": [10.5, 20.0, 100.0, 200.5],
        }
        det = Detection.from_internal_dict(raw)
        assert det.class_id == 3
        assert det.confidence == pytest.approx(0.87)
        assert det.bbox == (10.5, 20.0, 100.0, 200.5)
        assert det.polygon is None
        assert det.polygon_holes is None
        assert det.area is None

        # Round-trip : le dict produit doit être identique à l'entrée.
        assert det.to_internal_dict() == raw

    def test_segmentation_simple_round_trip(self):
        raw = {
            "class_id": 1,
            "confidence": 1.0,
            "polygon": [0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.1, 0.4],
            "bbox": [10.0, 20.0, 30.0, 40.0],
            "area": 250.5,
        }
        det = Detection.from_internal_dict(raw)
        assert det.polygon == ((0.1, 0.2), (0.3, 0.2), (0.3, 0.4), (0.1, 0.4))
        assert det.polygon_holes is None
        assert det.area == pytest.approx(250.5)

        assert det.to_internal_dict() == raw

    def test_segmentation_with_holes_round_trip(self):
        raw = {
            "class_id": 2,
            "confidence": 0.91,
            "polygon": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            "bbox": [0.0, 0.0, 100.0, 100.0],
            "area": 10000.0,
            "polygon_holes": [
                [0.2, 0.2, 0.4, 0.2, 0.4, 0.4, 0.2, 0.4],
                [0.6, 0.6, 0.8, 0.6, 0.8, 0.8, 0.6, 0.8],
            ],
        }
        det = Detection.from_internal_dict(raw)
        assert det.polygon_holes is not None
        assert len(det.polygon_holes) == 2
        assert det.polygon_holes[0] == ((0.2, 0.2), (0.4, 0.2), (0.4, 0.4), (0.2, 0.4))

        assert det.to_internal_dict() == raw

    def test_to_internal_dict_omits_unset_optionals(self):
        det = Detection(
            class_id=0,
            confidence=0.5,
            bbox=(0.0, 0.0, 10.0, 10.0),
        )
        out = det.to_internal_dict()
        assert "polygon" not in out
        assert "polygon_holes" not in out
        assert "area" not in out
        assert out == {
            "class_id": 0,
            "confidence": 0.5,
            "bbox": [0.0, 0.0, 10.0, 10.0],
        }

    def test_invalid_bbox_raises(self):
        with pytest.raises(ValueError, match="bbox"):
            Detection.from_internal_dict(
                {"class_id": 0, "confidence": 0.5, "bbox": [1, 2, 3]}
            )

    def test_odd_polygon_length_raises(self):
        with pytest.raises(ValueError, match="pair"):
            Detection.from_internal_dict(
                {
                    "class_id": 0,
                    "confidence": 0.5,
                    "bbox": [0, 0, 1, 1],
                    "polygon": [0.1, 0.2, 0.3],
                }
            )


# ----------------------------------------------------------------------
# Round-trip format on-disk (cv_output -> conversion_shp)
# ----------------------------------------------------------------------
class TestDiskDictRoundTrip:
    def test_bbox_disk_round_trip(self):
        raw = {
            "class_id": 3,
            "confidence": 0.87,
            "bbox_absolute": {"minx": 10.5, "miny": 20.0, "maxx": 100.0, "maxy": 200.5},
        }
        det = Detection.from_disk_dict(raw)
        assert det.bbox == (10.5, 20.0, 100.0, 200.5)
        assert det.polygon is None
        assert det.to_disk_dict() == raw

    def test_segmentation_disk_round_trip(self):
        raw = {
            "class_id": 1,
            "confidence": 1.0,
            "polygon": [0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.1, 0.4],
        }
        det = Detection.from_disk_dict(raw)
        assert det.polygon == ((0.1, 0.2), (0.3, 0.2), (0.3, 0.4), (0.1, 0.4))
        # bbox dérivée du polygone (en coords normalisées) faute de mieux
        assert det.bbox == (0.1, 0.2, 0.3, 0.4)
        assert det.to_disk_dict() == raw

    def test_segmentation_with_holes_disk_round_trip(self):
        raw = {
            "class_id": 2,
            "confidence": 0.91,
            "polygon": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            "polygon_holes": [
                [0.2, 0.2, 0.4, 0.2, 0.4, 0.4, 0.2, 0.4],
            ],
        }
        det = Detection.from_disk_dict(raw)
        assert det.polygon_holes is not None
        assert len(det.polygon_holes) == 1
        assert det.to_disk_dict() == raw

    def test_disk_dict_bbox_mode_omits_polygon_keys(self):
        det = Detection(
            class_id=0,
            confidence=0.5,
            bbox=(0.0, 0.0, 10.0, 10.0),
        )
        out = det.to_disk_dict()
        assert "polygon" not in out
        assert "polygon_holes" not in out
        assert "bbox_absolute" in out

    def test_disk_dict_segmentation_omits_bbox_absolute(self):
        det = Detection(
            class_id=0,
            confidence=1.0,
            bbox=(0.0, 0.0, 1.0, 1.0),
            polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        )
        out = det.to_disk_dict()
        assert "bbox_absolute" not in out
        assert "polygon" in out

    def test_missing_bbox_and_polygon_raises(self):
        with pytest.raises(ValueError, match="ni 'bbox_absolute' ni 'polygon'"):
            Detection.from_disk_dict({"class_id": 0, "confidence": 0.5})


# ----------------------------------------------------------------------
# Immutabilité
# ----------------------------------------------------------------------
class TestImmutability:
    def test_detection_is_frozen(self):
        det = Detection(
            class_id=0,
            confidence=0.5,
            bbox=(0.0, 0.0, 10.0, 10.0),
        )
        with pytest.raises(Exception):
            det.class_id = 1  # type: ignore[misc]
