"""Tests de connectivité : un nouveau modèle ajouté est-il correctement "branché" ?

Pour chaque dossier modèle dans ``data/models/`` qui contient ``weights/best.onnx``,
on appelle les readers existants du pipeline (sans QGIS, sans onnxruntime) et on
vérifie que :

- ``ModelProfile.load`` retourne un objet exploitable.
- ``load_class_names_from_model`` retourne une liste non vide.
- ``len(class_colors) == len(class_names)`` si ``class_colors`` est défini.
- ``load_sahi_config_from_model`` retourne un dict valide.
- ``load_postprocess_config_from_model`` retourne un dict valide.
- Toute règle de clustering cible des classes présentes dans ``classes.txt``.
- ``clustering.output_class_name`` ne collisionne pas avec ``classes.txt``.
- Les modèles détectés comme RF-DETR ont ``class_offset == 1`` dans le sidecar.

Niveau "light" : pas d'inférence ONNX réelle. Si aucun modèle versionné n'est
présent, les tests sont skippés gracieusement (paramétrage vide).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.cv.class_utils import load_class_names_from_model
from pipeline.cv.model_config import (
    is_rfdetr_model,
    load_clustering_config_from_model,
    load_postprocess_config_from_model,
    load_sahi_config_from_model,
)
from pipeline.cv.model_profile import ModelProfile

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "data" / "models"


def _model_dirs() -> list[Path]:
    if not MODELS_DIR.is_dir():
        return []
    return [
        d for d in sorted(MODELS_DIR.iterdir())
        if d.is_dir() and (d / "weights" / "best.onnx").is_file()
    ]


@pytest.fixture(params=_model_dirs(), ids=lambda d: d.name)
def model_dir(request: pytest.FixtureRequest) -> Path:
    return request.param


def test_model_profile_loads(model_dir: Path) -> None:
    weights = model_dir / "weights" / "best.onnx"
    profile = ModelProfile.load(weights)
    assert profile is not None
    assert profile.class_names, "class_names ne doit pas être vide"


def test_class_names_from_classes_txt(model_dir: Path) -> None:
    names = load_class_names_from_model(model_dir)
    assert names, "Aucun nom de classe chargé via la cascade classes.txt"
    # Les doublons sont autorisés (sous-classes) ; on vérifie seulement la
    # non-vacuité et qu'aucun nom n'est vide.
    assert all(n for n in names), f"Nom vide détecté : {names}"


def test_class_colors_length_matches(model_dir: Path) -> None:
    weights = model_dir / "weights" / "best.onnx"
    profile = ModelProfile.load(weights)
    if profile.class_colors is None:
        pytest.skip("class_colors non défini")
    assert profile.class_names is not None
    assert len(profile.class_colors) == len(profile.class_names), (
        f"class_colors ({len(profile.class_colors)}) != "
        f"class_names ({len(profile.class_names)})"
    )


def test_sahi_config_loadable(model_dir: Path) -> None:
    cfg = load_sahi_config_from_model(model_dir)
    for key in ("slice_width", "slice_height", "overlap_ratio"):
        assert key in cfg, f"Clé '{key}' absente du SAHI config"
    assert cfg["slice_width"] > 0
    assert cfg["slice_height"] > 0
    assert 0.0 <= cfg["overlap_ratio"] < 1.0


def test_postprocess_config_loadable(model_dir: Path) -> None:
    cfg = load_postprocess_config_from_model(model_dir)
    assert "merge_adjacent" in cfg
    assert "remove_overlaps" in cfg
    assert isinstance(cfg["merge_adjacent"], bool)
    assert isinstance(cfg["remove_overlaps"], bool)


def test_clustering_targets_in_classes(model_dir: Path) -> None:
    rules = load_clustering_config_from_model(model_dir)
    if not rules:
        pytest.skip("Pas de clustering configuré")
    names = set(load_class_names_from_model(model_dir) or [])
    for i, rule in enumerate(rules):
        for target in rule.get("target_classes", []):
            assert target in names, (
                f"Règle {i} : target '{target}' absent de classes.txt {sorted(names)}"
            )


def test_clustering_output_name_not_in_classes(model_dir: Path) -> None:
    rules = load_clustering_config_from_model(model_dir)
    if not rules:
        pytest.skip("Pas de clustering configuré")
    names = set(load_class_names_from_model(model_dir) or [])
    for i, rule in enumerate(rules):
        out = rule.get("output_class_name")
        if out:
            assert out not in names, (
                f"Règle {i} : output_class_name '{out}' collisionne avec classes.txt"
            )


def test_rfdetr_class_offset_consistent(model_dir: Path) -> None:
    weights = model_dir / "weights" / "best.onnx"
    profile = ModelProfile.load(weights)
    if not is_rfdetr_model(model_dir):
        pytest.skip("Modèle non RF-DETR")
    offset = profile.metadata.get("class_offset")
    assert offset == 1, (
        f"RF-DETR attendu avec class_offset=1, trouvé {offset!r} "
        f"dans weights/best.json"
    )
