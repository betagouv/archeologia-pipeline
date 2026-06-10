"""Tests de la sélection d'exclusion du packaging (``dev/package_plugin.py``).

Garantit le correctif AUDIT PKG-01 : `.venv_dev` (et plus généralement tout
dossier caché / cache de tooling) ne doit JAMAIS finir dans le ZIP distribué,
tout en conservant les données requises au runtime (PKG-02 : `quadrillage_france`).

Le module est chargé par chemin (il vit sous ``dev/``, hors du PYTHONPATH des tests).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PKG_SCRIPT = _REPO_ROOT / "dev" / "package_plugin.py"


def _load_package_plugin():
    spec = importlib.util.spec_from_file_location("package_plugin_under_test", _PKG_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pkg():
    return _load_package_plugin()


def _mkdir(tmp_path: Path, name: str) -> Path:
    d = tmp_path / name
    d.mkdir()
    return d


class TestShouldExcludeDirs:
    @pytest.mark.parametrize(
        "name",
        [".venv_dev", ".venv", ".ruff_cache", ".superpowers", ".mypy_cache",
         ".pytest_cache", ".git", "dev", "tests", "node_modules"],
    )
    def test_excludes_dev_and_hidden_dirs(self, pkg, tmp_path, name):
        d = _mkdir(tmp_path, name)
        assert pkg.should_exclude(d, name) is True

    @pytest.mark.parametrize("name", ["src", "data", "quadrillage_france", "models", "third_party"])
    def test_keeps_runtime_dirs(self, pkg, tmp_path, name):
        # PKG-02 : quadrillage_france est REQUIS au runtime → ne pas exclure.
        d = _mkdir(tmp_path, name)
        assert pkg.should_exclude(d, name) is False


class TestShouldExcludeFiles:
    @pytest.mark.parametrize("name", ["config.json", "last_ui_config.json", "pytest.ini", ".gitignore"])
    def test_excludes_dev_files(self, pkg, tmp_path, name):
        f = tmp_path / name
        f.write_text("{}")
        assert pkg.should_exclude(f, name) is True

    @pytest.mark.parametrize("name", ["best.pt", "model.pth", "module.pyc"])
    def test_excludes_stripped_extensions(self, pkg, tmp_path, name):
        f = tmp_path / name
        f.write_text("x")
        assert pkg.should_exclude(f, name) is True

    @pytest.mark.parametrize("name", ["best.onnx", "main.py", "metadata.txt", "icon.png"])
    def test_keeps_runtime_files(self, pkg, tmp_path, name):
        f = tmp_path / name
        f.write_text("x")
        assert pkg.should_exclude(f, name) is False
