"""Tests pour ModelProfile : chargement complet à partir de fixtures sur disque.

Vérifie que le profil charge ``args.yaml``, le sidecar ``.json`` et les
fichiers de classes en une passe et expose les bonnes valeurs typées.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.cv.model_profile import (
    ClusteringRule,
    ModelProfile,
    PostprocessConfig,
    SahiConfig,
)


# ----------------------------------------------------------------------
# Fixtures helpers
# ----------------------------------------------------------------------
def _make_model(
    tmp_path: Path,
    *,
    args_yaml: str = "",
    sidecar_json: dict = None,
    classes_txt: str = None,
    classes_json: list = None,
) -> Path:
    """Construit une arborescence modèle dans tmp_path et retourne le chemin
    vers ``best.onnx`` (vide, juste pour exister)."""
    model_dir = tmp_path / "my_model"
    weights_dir = model_dir / "weights"
    weights_dir.mkdir(parents=True)
    weights_path = weights_dir / "best.onnx"
    weights_path.write_bytes(b"")  # fichier vide suffit pour exister

    if args_yaml:
        (model_dir / "args.yaml").write_text(args_yaml, encoding="utf-8")
    if sidecar_json is not None:
        (weights_dir / "best.json").write_text(json.dumps(sidecar_json), encoding="utf-8")
    if classes_txt is not None:
        (model_dir / "classes.txt").write_text(classes_txt, encoding="utf-8")
    if classes_json is not None:
        (model_dir / "classes.json").write_text(
            json.dumps(classes_json), encoding="utf-8"
        )

    return weights_path


# ----------------------------------------------------------------------
# Empty / minimal model
# ----------------------------------------------------------------------
class TestMinimalModel:
    def test_load_with_no_args_yaml(self, tmp_path):
        weights = _make_model(tmp_path)
        profile = ModelProfile.load(weights)
        assert profile.weights_path == weights
        assert profile.model_dir == weights.parent.parent
        assert profile.class_names is None
        assert profile.class_colors is None
        assert profile.sahi == SahiConfig()  # defaults
        assert profile.clustering == ()
        assert profile.postprocess == PostprocessConfig()
        assert profile.is_rfdetr is False
        assert profile.args_yaml == {}
        assert profile.metadata == {}

    def test_load_missing_weights_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ModelProfile.load(tmp_path / "nope.onnx")


# ----------------------------------------------------------------------
# args.yaml parsing
# ----------------------------------------------------------------------
class TestArgsYamlParsing:
    def test_sahi_loaded(self, tmp_path):
        weights = _make_model(
            tmp_path,
            args_yaml="""
sahi:
  slice_height: 1024
  slice_width: 768
  overlap_ratio: 0.3
""",
        )
        profile = ModelProfile.load(weights)
        assert profile.sahi == SahiConfig(slice_height=1024, slice_width=768, overlap_ratio=0.3)

    def test_clustering_single_dict(self, tmp_path):
        weights = _make_model(
            tmp_path,
            args_yaml="""
clustering:
  target_classes: [cratere_obus]
  min_confidence: 0.5
  min_cluster_size: 10
  eps_m: 25
  output_class_name: zone_crateres
""",
        )
        profile = ModelProfile.load(weights)
        assert len(profile.clustering) == 1
        rule = profile.clustering[0]
        assert rule.target_classes == ("cratere_obus",)
        assert rule.min_confidence == 0.5
        assert rule.min_confidence_extend == 0.5  # défaut = min_confidence
        assert rule.min_cluster_size == 10
        assert rule.eps_m == 25.0
        assert rule.output_class_name == "zone_crateres"
        assert rule.output_geometry == "convex_hull"  # défaut

    def test_clustering_with_hysteresis(self, tmp_path):
        weights = _make_model(
            tmp_path,
            args_yaml="""
clustering:
  - target_classes: [a, b]
    min_confidence: 0.5
    min_confidence_extend: 0.3
    output_class_name: cluster_ab
""",
        )
        profile = ModelProfile.load(weights)
        rule = profile.clustering[0]
        assert rule.min_confidence_extend == 0.3
        assert rule.target_classes == ("a", "b")

    def test_clustering_string_target_normalized_to_tuple(self, tmp_path):
        weights = _make_model(
            tmp_path,
            args_yaml="""
clustering:
  target_class: solo
""",
        )
        profile = ModelProfile.load(weights)
        assert len(profile.clustering) == 1
        assert profile.clustering[0].target_classes == ("solo",)

    def test_clustering_invalid_skipped(self, tmp_path):
        weights = _make_model(
            tmp_path,
            args_yaml="""
clustering:
  - target_classes: []
  - target_classes: [valid]
    output_class_name: x
""",
        )
        profile = ModelProfile.load(weights)
        # La 1re est ignorée (target vide), la 2e gardée
        assert len(profile.clustering) == 1
        assert profile.clustering[0].output_class_name == "x"

    def test_postprocess_partial(self, tmp_path):
        weights = _make_model(
            tmp_path,
            args_yaml="""
postprocess:
  merge_adjacent: false
""",
        )
        profile = ModelProfile.load(weights)
        assert profile.postprocess == PostprocessConfig(merge_adjacent=False, remove_overlaps=True)

    def test_class_colors(self, tmp_path):
        weights = _make_model(
            tmp_path,
            args_yaml="""
class_colors: [3, 1, 7]
""",
        )
        profile = ModelProfile.load(weights)
        assert profile.class_colors == (3, 1, 7)

    def test_rfdetr_detection(self, tmp_path):
        weights = _make_model(
            tmp_path,
            args_yaml="model: rf-detr-base\n",
        )
        profile = ModelProfile.load(weights)
        assert profile.is_rfdetr is True

    def test_yolo_default_not_rfdetr(self, tmp_path):
        weights = _make_model(tmp_path, args_yaml="model: yolov8n\n")
        profile = ModelProfile.load(weights)
        assert profile.is_rfdetr is False


# ----------------------------------------------------------------------
# Sidecar metadata
# ----------------------------------------------------------------------
class TestSidecarMetadata:
    def test_metadata_loaded(self, tmp_path):
        weights = _make_model(
            tmp_path,
            sidecar_json={
                "task": "semantic_segmentation",
                "model_type": "segformer",
                "confidence_threshold": 0.7,
                "bg_bias": 0.1,
            },
        )
        profile = ModelProfile.load(weights)
        assert profile.metadata["task"] == "semantic_segmentation"
        assert profile.task == "semantic_segmentation"
        assert profile.model_type == "segformer"

    def test_effective_confidence_uses_metadata_first(self, tmp_path):
        weights = _make_model(
            tmp_path,
            sidecar_json={"confidence_threshold": 0.7},
        )
        profile = ModelProfile.load(weights)
        assert profile.effective_confidence_threshold(run_default=0.3) == 0.7

    def test_effective_confidence_falls_back_to_run_default(self, tmp_path):
        weights = _make_model(tmp_path, sidecar_json={"task": "detect"})
        profile = ModelProfile.load(weights)
        assert profile.effective_confidence_threshold(run_default=0.45) == 0.45

    def test_effective_confidence_invalid_metadata_ignored(self, tmp_path):
        weights = _make_model(
            tmp_path,
            sidecar_json={"confidence_threshold": "not_a_number"},
        )
        profile = ModelProfile.load(weights)
        assert profile.effective_confidence_threshold(run_default=0.3) == 0.3


# ----------------------------------------------------------------------
# Class names cascade
# ----------------------------------------------------------------------
class TestClassNames:
    def test_classes_txt(self, tmp_path):
        weights = _make_model(tmp_path, classes_txt="alpha\nbeta\ngamma\n")
        profile = ModelProfile.load(weights)
        assert profile.class_names == ("alpha", "beta", "gamma")

    def test_classes_json_list(self, tmp_path):
        weights = _make_model(tmp_path, classes_json=["a", "b"])
        profile = ModelProfile.load(weights)
        assert profile.class_names == ("a", "b")

    def test_no_classes_returns_none(self, tmp_path):
        weights = _make_model(tmp_path)
        profile = ModelProfile.load(weights)
        assert profile.class_names is None
