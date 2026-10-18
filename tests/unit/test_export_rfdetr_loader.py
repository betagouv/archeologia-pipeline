"""Chargeur RF-DETR de l'export ONNX (dev/runner_onnx/export_to_onnx.py).

Depuis rfdetr 1.8, un checkpoint « best » enregistre ``model_name`` et ``model_config`` et son ``args`` ne contient
plus ``resolution``. Le chargeur doit alors reconstruire la classe enregistrée avec sa résolution, au lieu de
déduire l'architecture de la largeur du backbone (384 pour RF-DETR Base, lue à tort comme hidden_dim de Large)
et de la taille des position_embeddings (518 px de pré-entraînement DINOv2).

torch et rfdetr sont remplacés par des modules factices : le test ne vérifie que le choix de la classe et de ses
paramètres.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

EXPORTER = Path(__file__).resolve().parents[2] / "dev" / "runner_onnx" / "export_to_onnx.py"


class _Shape(tuple):
    @property
    def shape(self):
        return self


def _fake_variant(name, calls):
    def init(self, **kwargs):
        calls.append((name, kwargs))
    return type(name, (), {"__init__": init})


@pytest.fixture
def loader(monkeypatch):
    calls = []
    detr = types.ModuleType("rfdetr.detr")
    for name in ("RFDETRBase", "RFDETRLarge", "RFDETRNano", "RFDETRSmall", "RFDETRMedium"):
        setattr(detr, name, _fake_variant(name, calls))
    rfdetr = types.ModuleType("rfdetr")
    rfdetr.detr = detr
    checkpoint = {}
    torch = types.ModuleType("torch")
    torch.device = lambda name: name
    torch.load = lambda *args, **kwargs: checkpoint
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "rfdetr", rfdetr)
    monkeypatch.setitem(sys.modules, "rfdetr.detr", detr)
    spec = importlib.util.spec_from_file_location("export_to_onnx_under_test", EXPORTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._load_rfdetr_model, checkpoint, calls


def _base_weights():
    # RF-DETR Base : backbone dinov2_windowed_small (largeur 384, patch 14), position_embeddings 37 x 37 + 1
    return {
        "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight": _Shape((384, 3, 14, 14)),
        "backbone.0.encoder.encoder.embeddings.position_embeddings": _Shape((1, 1370, 384)),
    }


def test_checkpoint_rfdetr_1_8_relu_avec_sa_classe_et_sa_resolution(loader):
    load, checkpoint, calls = loader
    checkpoint.update({
        "model": _base_weights(),
        "args": {"epochs": 20, "lr": 1e-4},  # configuration d'entraînement seule, sans resolution
        "model_name": "RFDETRBase",
        "model_config": {"encoder": "dinov2_windowed_small", "hidden_dim": 256, "patch_size": 14,
                         "resolution": 560, "positional_encoding_size": 37, "device": "cuda"},
    })
    load("best.pth")
    assert calls == [("RFDETRBase", {"pretrain_weights": "best.pth", "resolution": 560, "patch_size": 14,
                                     "positional_encoding_size": 37})]


def test_les_options_de_la_ligne_de_commande_restent_prioritaires(loader):
    load, checkpoint, calls = loader
    checkpoint.update({"model": _base_weights(), "args": {}, "model_name": "RFDETRBase",
                       "model_config": {"patch_size": 14, "resolution": 560, "positional_encoding_size": 37}})
    load("best.pth", patch_size_override=16, positional_encoding_size_override=35)
    assert calls[0][1]["patch_size"] == 16 and calls[0][1]["positional_encoding_size"] == 35


def test_ancien_checkpoint_garde_la_detection_existante(loader):
    load, checkpoint, calls = loader
    # checkpoint antérieur à rfdetr 1.8 : resolution dans args, ni model_name ni model_config
    checkpoint.update({"model": {"backbone.patch_embeddings.projection.weight": _Shape((384, 3, 16, 16))},
                       "args": {"resolution": 384}})
    load("best.pth")
    assert calls == [("RFDETRNano", {"pretrain_weights": "best.pth"})]
