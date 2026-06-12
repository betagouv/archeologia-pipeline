"""TEST-03 (audit v1/v2, 1ʳᵉ tranche) : le décodage tensoriel ONNX n'avait
AUCUN test alors que le fallback in-process est le chemin d'inférence primaire
de tout install sans runner compilé. Tenseurs synthétiques connus →
seuillage, orientation (transpose), conversion de coordonnées, clamp.

Couvre ``_postprocess_yolo`` ; RF-DETR et SegFormer restent à couvrir
(suite du chantier TEST-03).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("shapely")  # pipeline.cv.__init__

from pipeline.cv.computer_vision_onnx import _postprocess_yolo

# Tuile : modèle 640×640, image 1280×1280 → échelle ×2.
IMG, MODEL = 1280, 640


def _row(xc, yc, w, h, scores):
    return [xc, yc, w, h, *scores]


def _decode(rows, threshold=0.25, img=IMG, model=MODEL):
    output = np.asarray(rows, dtype=np.float32)
    return _postprocess_yolo([output], img, img, model, model, threshold)


class TestSeuillageEtClasses:
    def test_detection_au_dessus_du_seuil_decodee(self):
        dets = _decode([_row(100, 200, 50, 80, [0.1, 0.9])])
        assert len(dets) == 1
        assert dets[0]["class_id"] == 1
        assert dets[0]["confidence"] == pytest.approx(0.9)

    def test_detection_sous_le_seuil_ignoree(self):
        dets = _decode([_row(100, 200, 50, 80, [0.1, 0.2])], threshold=0.25)
        assert dets == []

    def test_seuil_exact_conserve(self):
        # `confidence < threshold` exclut : l'égalité passe.
        dets = _decode([_row(100, 200, 50, 80, [0.25, 0.1])], threshold=0.25)
        assert len(dets) == 1
        assert dets[0]["class_id"] == 0


class TestCoordonnees:
    def test_conversion_centre_vers_coins_et_echelle(self):
        # centre (100, 200), taille (50, 80) en pixels modèle, échelle ×2.
        dets = _decode([_row(100, 200, 50, 80, [0.9, 0.1])])
        x1, y1, x2, y2 = dets[0]["bbox"]
        assert (x1, y1) == pytest.approx(((100 - 25) * 2, (200 - 40) * 2))
        assert (x2, y2) == pytest.approx(((100 + 25) * 2, (200 + 40) * 2))

    def test_clamp_aux_limites_image(self):
        # Boîte débordant du bord haut-gauche → clampée à 0.
        dets = _decode([_row(5, 5, 40, 40, [0.9, 0.1])])
        x1, y1, x2, y2 = dets[0]["bbox"]
        assert x1 == 0 and y1 == 0
        assert x2 == pytest.approx((5 + 20) * 2)
        assert y2 == pytest.approx((5 + 20) * 2)


class TestOrientationsDeSortie:
    def test_format_batch_num_canaux(self):
        # [1, num, 4+nc] : dimension batch retirée.
        output = np.asarray([[_row(100, 200, 50, 80, [0.9, 0.1])]], dtype=np.float32)
        dets = _postprocess_yolo([output], IMG, IMG, MODEL, MODEL, 0.25)
        assert len(dets) == 1

    def test_format_transpose_canaux_num(self):
        # [4+nc, num] (export ultralytics brut : des milliers d'anchors) :
        # doit être transposé et produire EXACTEMENT le même décodage que la
        # forme [num, 4+nc].
        rows = [
            _row(100, 200, 50, 80, [0.9, 0.1]),
            _row(300, 300, 20, 20, [0.1, 0.8]),
        ] + [_row(10, 10, 4, 4, [0.01, 0.01]) for _ in range(10)]  # 12 anchors
        reference = _decode(rows)
        assert len(reference) == 2  # garde : la référence décode bien
        transposed = np.asarray(rows, dtype=np.float32).T  # (6, 12)
        dets = _postprocess_yolo([transposed], IMG, IMG, MODEL, MODEL, 0.25)
        assert dets == reference

    def test_peu_de_detections_forme_canonique_non_transposee(self):
        # Régression (découverte par TEST-03) : un export avec NMS embarqué
        # renvoie PEU de lignes en forme canonique (ex. (3, 6)) — l'ancienne
        # heuristique la transposait à tort → 0 détection silencieusement.
        rows = [
            _row(100, 200, 50, 80, [0.9, 0.1]),
            _row(300, 300, 20, 20, [0.1, 0.8]),
            _row(400, 100, 30, 30, [0.7, 0.2]),
        ]  # (3, 6)
        dets = _decode(rows)
        assert len(dets) == 3

    def test_beaucoup_de_classes_forme_canonique(self):
        # ≥ 10 classes en forme [num, 4+nc] : l'heuristique de transpose
        # (shape[0] < shape[1] ET shape[0] < 10) ne doit PAS s'activer.
        # NB (limite documentée par l'audit) : un export TRANSPOSÉ à ≥ 10
        # classes ne serait pas détecté — couvert au moins dans ce sens-ci.
        nc = 12
        scores = [0.0] * nc
        scores[7] = 0.9
        rows = [_row(100, 200, 50, 80, scores) for _ in range(20)]  # (20, 16)
        dets = _decode(rows)
        assert len(dets) == 20
        assert all(d["class_id"] == 7 for d in dets)


class TestRobustesse:
    def test_sortie_vide(self):
        dets = _postprocess_yolo(
            [np.zeros((0, 6), dtype=np.float32)], IMG, IMG, MODEL, MODEL, 0.25
        )
        assert dets == []

    def test_scores_nuls_tous_filtres(self):
        dets = _decode([_row(100, 200, 50, 80, [0.0, 0.0])])
        assert dets == []
