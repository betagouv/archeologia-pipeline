"""PARSE-12 (audit v2) : des paramètres SAHI aberrants dans args.yaml
(``overlap_ratio: 20`` — faute d'échelle pour 20 % —, ``slice ≤ 0``) rendaient
le pas d'avancement nul ou négatif dans ``get_slice_bboxes`` → boucle infinie
non annulable dans le thread worker (mémoire croissante, QGIS à tuer).

Deux niveaux de défense : bornes au parsing (model_config / model_profile,
avec warning) ET pas garanti > 0 dans get_slice_bboxes.
"""
from __future__ import annotations

import threading

import pytest

pytest.importorskip("shapely")
pytest.importorskip("numpy")

from pipeline.cv.sahi_lite import get_slice_bboxes
from pipeline.cv.model_config import load_sahi_config_from_model
from pipeline.cv.model_profile import _parse_sahi


def _run_with_timeout(fn, timeout=10):
    out: list = []
    done = threading.Event()

    def _call():
        try:
            out.append(fn())
        finally:
            done.set()

    threading.Thread(target=_call, daemon=True).start()
    assert done.wait(timeout), (
        "BOUCLE INFINIE : get_slice_bboxes ne progresse pas (PARSE-12)"
    )
    return out[0]


class TestGetSliceBboxesBornes:
    def test_overlap_egal_1_termine(self):
        boxes = _run_with_timeout(
            lambda: get_slice_bboxes(1000, 1000, 640, 640, 1.0, 1.0)
        )
        assert boxes

    def test_overlap_aberrant_20_termine(self):
        boxes = _run_with_timeout(
            lambda: get_slice_bboxes(1000, 1000, 640, 640, 20.0, 20.0)
        )
        assert boxes

    def test_slice_nul_termine(self):
        boxes = _run_with_timeout(lambda: get_slice_bboxes(100, 100, 0, 0))
        assert boxes

    def test_nominal_inchange(self):
        # Pas de régression : pas = 640 − 128 = 512 → 2 positions par axe.
        boxes = get_slice_bboxes(1000, 1000, 640, 640, 0.2, 0.2)
        assert len(boxes) == 4


class TestParseursClampent:
    def test_load_sahi_config_clampe(self, tmp_path):
        (tmp_path / "args.yaml").write_text(
            "sahi:\n  slice_height: 0\n  slice_width: 640\n  overlap_ratio: 20\n",
            encoding="utf-8",
        )
        cfg = load_sahi_config_from_model(tmp_path)
        assert cfg["slice_height"] >= 32
        assert 0.0 <= cfg["overlap_ratio"] <= 0.9

    def test_parse_sahi_clampe(self):
        cfg = _parse_sahi(
            {"sahi": {"slice_height": -5, "slice_width": 640, "overlap_ratio": 1.0}}
        )
        assert cfg.slice_height >= 32
        assert cfg.overlap_ratio <= 0.9

    def test_valeurs_nominales_inchangees(self, tmp_path):
        (tmp_path / "args.yaml").write_text(
            "sahi:\n  slice_height: 1024\n  slice_width: 1024\n  overlap_ratio: 0.25\n",
            encoding="utf-8",
        )
        cfg = load_sahi_config_from_model(tmp_path)
        assert cfg == {
            "slice_height": 1024, "slice_width": 1024, "overlap_ratio": 0.25,
        }
