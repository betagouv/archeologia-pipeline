"""Fraîcheur du binaire externe cv_runner_onnx.exe (T2 audit 2026-08-31).

L'exe PyInstaller fige un instantané de ``src/`` : le binaire livré du 16/06
avait raté 4 commits de correctifs CV pendant 2 mois et demi, sans qu'aucun
signal ne le dise. Ce test recompare le sha256 des sources opérantes
(``src/pipeline/cv/**.py`` + le CLI) au ``build_info.json`` posé à côté de
l'exe par ``dev/runner_onnx/build.py``.

Rouge = binaire périmé → ``python dev/runner_onnx/build.py`` (venv .venv_onnx,
cf. dev/runner_onnx/README.md).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[2]
BIN_DIR = PLUGIN_ROOT / "data" / "third_party" / "cv_runner_onnx" / "windows"


def _module_build():
    spec = importlib.util.spec_from_file_location(
        "_build_runner", PLUGIN_ROOT / "dev" / "runner_onnx" / "build.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_binaire_a_jour():
    exe = BIN_DIR / "cv_runner_onnx.exe"
    if not exe.exists():
        pytest.skip("binaire externe absent (poste sans runner compilé)")
    info_path = BIN_DIR / "build_info.json"
    assert info_path.exists(), (
        "binaire présent SANS build_info.json — impossible de dater ses sources. "
        "Recompiler : python dev/runner_onnx/build.py"
    )
    info = json.loads(info_path.read_text(encoding="utf-8"))
    attendu = _module_build().hash_cv_sources(PLUGIN_ROOT)
    assert info.get("sources_sha256") == attendu, (
        f"binaire PÉRIMÉ (build {info.get('date')}, commit {info.get('commit')}) : "
        "les sources src/pipeline/cv ont changé depuis la compilation. "
        "Recompiler : python dev/runner_onnx/build.py"
    )
