"""PARSE-08 (audit v2) : format dict de classes.json.

Les clés d'un objet JSON sont toujours des *strings* ("0", "1", …) ; les
loaders doivent donc indexer avec ``str(i)`` sous peine de renvoyer
``classe_N`` pour toutes les classes — ce qui casse ensuite le filtre
``selected_classes`` (0 détection silencieusement).
"""
from __future__ import annotations

import json

import pytest

# L'import de pipeline.cv.* déclenche pipeline/cv/__init__ (shapely requis).
pytest.importorskip("shapely")

from pipeline.cv.class_utils import load_class_names_from_model
from pipeline.cv.model_profile import _load_class_names


CLASSES_DICT = {"0": "cratere", "1": "charbonniere", "2": "parcellaire"}


def _write_classes_json(tmp_path, payload):
    (tmp_path / "classes.json").write_text(json.dumps(payload), encoding="utf-8")
    return tmp_path


class TestClassesJsonDictFormat:
    def test_class_utils_lit_le_format_dict(self, tmp_path):
        model_dir = _write_classes_json(tmp_path, CLASSES_DICT)
        assert load_class_names_from_model(model_dir) == [
            "cratere", "charbonniere", "parcellaire",
        ]

    def test_model_profile_lit_le_format_dict(self, tmp_path):
        model_dir = _write_classes_json(tmp_path, CLASSES_DICT)
        assert _load_class_names(model_dir) == [
            "cratere", "charbonniere", "parcellaire",
        ]

    def test_trou_dans_les_indices_comble_par_placeholder(self, tmp_path):
        model_dir = _write_classes_json(tmp_path, {"0": "cratere", "2": "parcellaire"})
        assert load_class_names_from_model(model_dir) == [
            "cratere", "classe_1", "parcellaire",
        ]

    def test_format_liste_inchange(self, tmp_path):
        model_dir = _write_classes_json(tmp_path, ["cratere", "charbonniere"])
        assert load_class_names_from_model(model_dir) == ["cratere", "charbonniere"]
