"""Garde d'UNIFICATION des défauts de la chaîne CV (audit 2026-08-31).

Avant : trois défauts de confiance coexistaient (0.2 orchestrateur/config_manager,
0.3 runners, 0.5 signature onnx) et le binaire externe slicait à 750 quand tout le
reste disait 640 — le chemin d'entrée décidait du seuil de décodage. Le foyer est
``pipeline.cv.model_config.DEFAULT_*`` ; les modules hors du paquet pipeline
(orchestrateur, config_manager — pas de couplage app→pipeline) gardent des
littéraux que CE test verrouille sur les constantes.
"""
from __future__ import annotations

import re
from pathlib import Path

from pipeline.cv.model_config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    DEFAULT_SAHI_OVERLAP,
    DEFAULT_SAHI_SLICE,
)

PLUGIN_ROOT = Path(__file__).resolve().parents[2]


def test_valeurs_canoniques():
    assert DEFAULT_CONFIDENCE == 0.3
    assert DEFAULT_SAHI_SLICE == 640
    assert DEFAULT_SAHI_OVERLAP == 0.2
    assert DEFAULT_IOU == 0.5


def test_orchestrateur_aligne():
    from app.services.model_orchestrator import InstalledModel, _extract_thresholds

    m = InstalledModel(name="x", display_name="x", weights_path=None, target_rvt="LD",
                       status="beta", coverage={}, class_names=("a",))
    assert m.default_confidence == DEFAULT_CONFIDENCE
    assert m.default_iou == DEFAULT_IOU
    conf, pc, area, iou = _extract_thresholds({})
    assert conf == DEFAULT_CONFIDENCE and iou == DEFAULT_IOU and pc == {}


def test_model_profile_aligne():
    from pipeline.cv.model_profile import SahiConfig

    s = SahiConfig()
    assert s.slice_height == DEFAULT_SAHI_SLICE
    assert s.slice_width == DEFAULT_SAHI_SLICE
    assert s.overlap_ratio == DEFAULT_SAHI_OVERLAP


def test_config_manager_aligne_source():
    # config_manager ne doit pas importer pipeline (couche config) : contrôle
    # au niveau source — son littéral doit valoir DEFAULT_CONFIDENCE.
    src = (PLUGIN_ROOT / "src" / "config" / "config_manager.py").read_text(encoding="utf-8")
    m = re.search(r'"confidence_threshold":\s*([0-9.]+)', src)
    assert m, "confidence_threshold introuvable dans config_manager.py"
    assert float(m.group(1)) == DEFAULT_CONFIDENCE, (
        f"config_manager.py: confidence_threshold={m.group(1)} != DEFAULT_CONFIDENCE={DEFAULT_CONFIDENCE}"
    )


def test_cli_binaire_aligne_source():
    # Le CLI du binaire externe doit consommer les constantes (plus de 750 en dur).
    src = (PLUGIN_ROOT / "dev" / "runner_onnx" / "cv_runner_onnx_cli.py").read_text(encoding="utf-8")
    assert "DEFAULT_SAHI_SLICE" in src and "DEFAULT_CONFIDENCE" in src
    assert not re.search(r'slice_(height|width)",\s*750', src), "défaut SAHI 750 réintroduit"
