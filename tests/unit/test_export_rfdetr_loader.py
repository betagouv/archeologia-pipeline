"""Chargeur RF-DETR de l'export ONNX (dev/runner_onnx/export_to_onnx.py).

Depuis rfdetr 1.8, un checkpoint « best » enregistre ``model_name`` et ``model_config`` et son ``args`` ne contient
plus ``resolution``. Le chargeur doit alors reconstruire la classe enregistrée avec sa résolution, au lieu de
déduire l'architecture de la largeur du backbone (384 pour RF-DETR Base, lue à tort comme hidden_dim de Large)
et de la taille des position_embeddings (518 px de pré-entraînement DINOv2).

torch et rfdetr sont remplacés par des modules factices : le test ne vérifie que le choix de la classe et de ses
paramètres.
"""
from __future__ import annotations

import contextlib
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


# Les champs de la configuration d'une variante (ModelConfig de rfdetr, extra="forbid" : tout autre champ lève)
_CHAMPS = ("encoder", "hidden_dim", "dec_layers", "patch_size", "num_windows", "resolution",
           "positional_encoding_size", "num_queries", "num_select", "num_classes", "segmentation_head",
           "group_detr", "pretrain_weights", "device")
_VARIANTES = ("RFDETRBase", "RFDETRLarge", "RFDETRNano", "RFDETRSmall", "RFDETRMedium",
              "RFDETRSegLarge", "RFDETRSegPreview")


def _fake_variant(name, calls):
    def init(self, **kwargs):
        calls.append((name, kwargs))
    config_class = type(f"{name}Config", (), {"model_fields": dict.fromkeys(_CHAMPS)})
    return type(name, (), {"__init__": init, "_model_config_class": config_class})


@pytest.fixture
def loader(monkeypatch):
    calls = []
    detr = types.ModuleType("rfdetr.detr")
    for name in _VARIANTES:
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
    # toute l'architecture enregistrée, sauf le périphérique d'entraînement
    assert calls == [("RFDETRBase", {"pretrain_weights": "best.pth", "encoder": "dinov2_windowed_small",
                                     "hidden_dim": 256, "patch_size": 14, "resolution": 560,
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


def _seg_weights():
    # RF-DETR Seg Large de nos modèles : backbone de largeur 384, patch 12, position_embeddings 54 x 54 + 1 (648 px)
    return {
        "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight": _Shape((384, 3, 12, 12)),
        "backbone.0.encoder.encoder.embeddings.position_embeddings": _Shape((1, 2917, 384)),
    }


# model_config de crateres_seg_ld_v1 (rfdetr 1.8.3), réduit aux champs qui comptent ici, plus deux qui ne sont pas
# des champs de la configuration et doivent être écartés
_CONFIG_SEG = {"encoder": "dinov2_windowed_small", "hidden_dim": 256, "dec_layers": 5, "patch_size": 12,
               "num_windows": 2, "resolution": 648, "positional_encoding_size": 54, "num_queries": 200,
               "num_select": 200, "num_classes": 1, "segmentation_head": True, "group_detr": 13,
               "pretrain_weights": "/content/base_weights.pth", "device": "cuda",
               "license": "Apache-2.0", "champ_inconnu": 1}
_ATTENDU_SEG = {"pretrain_weights": "best.pth", "encoder": "dinov2_windowed_small", "hidden_dim": 256,
                "dec_layers": 5, "patch_size": 12, "num_windows": 2, "resolution": 648,
                "positional_encoding_size": 54, "num_queries": 200, "num_select": 200, "num_classes": 1,
                "segmentation_head": True, "group_detr": 13}


def test_checkpoint_seg_reconstruit_toute_l_architecture_enregistree(loader):
    load, checkpoint, calls = loader
    checkpoint.update({"model": _seg_weights(), "args": {"segmentation_head": True},
                       "model_name": "RFDETRSegLarge", "model_config": dict(_CONFIG_SEG)})
    load("best.pth")
    assert calls == [("RFDETRSegLarge", _ATTENDU_SEG)]


def test_model_config_objet_lu_comme_un_dict(loader):
    load, checkpoint, calls = loader

    class _Config:  # objet pydantic : model_dump() rend le dict
        def model_dump(self):
            return dict(_CONFIG_SEG)

    checkpoint.update({"model": _seg_weights(), "args": {"segmentation_head": True},
                       "model_name": "RFDETRSegLarge", "model_config": _Config()})
    load("best.pth")
    assert calls == [("RFDETRSegLarge", _ATTENDU_SEG)]


def test_classe_absente_de_l_ancienne_liste_resolue(loader):
    load, checkpoint, calls = loader
    checkpoint.update({"model": _seg_weights(), "args": {}, "model_name": "RFDETRSegPreview",
                       "model_config": {"resolution": 432, "patch_size": 12, "positional_encoding_size": 36}})
    load("best.pth")
    assert calls[0][0] == "RFDETRSegPreview"


def test_model_name_sans_model_config_garde_la_detection_par_les_poids(loader):
    load, checkpoint, calls = loader
    # tranchees_seg_ld_v1 : model_name sans model_config ; les valeurs par défaut de RFDETRSegLarge (504 px)
    # seraient fausses, la résolution vient des position_embeddings
    checkpoint.update({"model": _seg_weights(), "args": {"segmentation_head": True},
                       "model_name": "RFDETRSegLarge"})
    load("best.pth")
    assert calls == [("RFDETRSegLarge", {"pretrain_weights": "best.pth", "resolution": 648, "patch_size": 12,
                                         "positional_encoding_size": 54})]


def test_export_sur_cpu_meme_si_rfdetr_annonce_cuda(monkeypatch, tmp_path):
    # rfdetr 1.8.3 sur un poste CUDA : config.device vaut "cuda", les poids restent sur le CPU ; l'entrée factice
    # créée sur config.device faisait échouer l'export (« Input type and weight type should be the same »)
    journal = []

    class _Arret(Exception):
        pass

    class _Reseau:
        device = "cuda"

        def cpu(self):
            self.device = "cpu"
            return self

        def parameters(self):
            return iter([types.SimpleNamespace(device=self.device)])

        def eval(self):
            return self

        def export(self):
            pass

        def __call__(self, x):
            journal.append(("forward", self.device, x))
            return {"pred_logits": 0, "pred_boxes": 0, "aux_outputs": 0, "enc_outputs": 0}

    reseau = _Reseau()
    torch = types.ModuleType("torch")
    torch.device = lambda name: name
    torch.no_grad = contextlib.nullcontext
    torch.randn = lambda *shape, device=None: f"entrée sur {device}"
    torch.onnx = types.SimpleNamespace(export=lambda *a, **k: (_ for _ in ()).throw(_Arret()))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "onnx", types.ModuleType("onnx"))
    onnxsim = types.ModuleType("onnxsim")
    onnxsim.simplify = None
    monkeypatch.setitem(sys.modules, "onnxsim", onnxsim)
    spec = importlib.util.spec_from_file_location("export_to_onnx_under_test", EXPORTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rf = types.SimpleNamespace(model=types.SimpleNamespace(model=reseau),
                               model_config=types.SimpleNamespace(resolution=560, device="cuda"))
    monkeypatch.setattr(module, "_load_rfdetr_model", lambda *a, **k: rf)
    assert module.export_rfdetr_to_onnx(tmp_path / "best.pth", tmp_path / "best.onnx") is False  # arrêté à l'export
    assert journal == [("forward", "cpu", "entrée sur cpu")]
