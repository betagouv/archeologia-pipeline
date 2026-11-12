"""RF-DETR détection : les boîtes sont du cxcywh normalisé, même quand une requête vide déborde de la fenêtre.

Vécu le 2026-11-12 (Vosges, dalle LHD_FXX_0945_6766, fenêtre x 0–672, y 538–1210) : une requête de score 0,04 et de
cy = 1,0004 faisait lire les 300 boîtes de la fenêtre en pixels ; toutes passaient sous 10 px et étaient jetées —
26 charbonnières nettes sans détection, sans message. Mesuré sur 20 dalles : 3 fenêtres sur 320.
"""
import numpy as np
import pytest

pytest.importorskip("shapely")


def _sorties(boites, scores):
    """[pred_boxes [1, N, 4], pred_logits [1, N, classes]] comme l'export RF-DETR."""
    scores = np.asarray(scores, dtype=np.float32)
    return [np.asarray(boites, dtype=np.float32)[None], np.log(scores / (1 - scores))[None]]


CHARBONNIERE = [0.5, 0.5, 0.04, 0.04]          # 27 px de côté dans une fenêtre de 672 px


def test_une_requete_vide_qui_deborde_ne_fait_pas_perdre_la_fenetre():
    from pipeline.cv.computer_vision_onnx import _postprocess_rfdetr

    sorties = _sorties([CHARBONNIERE, [0.0216, 1.0004, 0.0262, 0.0035]], [[0.90, 0.01], [0.04, 0.01]])
    (det,) = _postprocess_rfdetr(sorties, 672, 672, 672, 672, 0.2957, 0)
    assert det["class_id"] == 0 and det["confidence"] == pytest.approx(0.90, abs=1e-3)
    assert det["bbox"] == pytest.approx([322.56, 322.56, 349.44, 349.44], abs=0.01)


def test_sans_debordement_rien_ne_change():
    from pipeline.cv.computer_vision_onnx import _postprocess_rfdetr

    (det,) = _postprocess_rfdetr(_sorties([CHARBONNIERE], [[0.90, 0.01]]), 672, 672, 672, 672, 0.2957, 0)
    assert det["bbox"] == pytest.approx([322.56, 322.56, 349.44, 349.44], abs=0.01)
