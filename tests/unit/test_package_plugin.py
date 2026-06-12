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
         ".pytest_cache", ".git", "dev", "tests", "node_modules",
         # PKG-04 (audit v2) : specs internes, scripts dev et artefacts de
         # build n'ont rien à faire chez l'utilisateur.
         "docs", "scripts", "dist"],
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
    @pytest.mark.parametrize("name", ["config.json", "last_ui_config.json", "class_color_registry.json", "pytest.ini", ".gitignore"])
    def test_excludes_dev_files(self, pkg, tmp_path, name):
        f = tmp_path / name
        f.write_text("{}")
        assert pkg.should_exclude(f, name) is True

    @pytest.mark.parametrize("name", ["best.pt", "model.pth", "module.pyc"])
    def test_excludes_stripped_extensions(self, pkg, tmp_path, name):
        f = tmp_path / name
        f.write_text("x")
        assert pkg.should_exclude(f, name) is True

    @pytest.mark.parametrize("name", ["best.onnx", "main.py", "metadata.txt", "icon.png", "README.md"])
    def test_keeps_runtime_files(self, pkg, tmp_path, name):
        f = tmp_path / name
        f.write_text("x")
        assert pkg.should_exclude(f, name) is False

    @pytest.mark.parametrize("name", ["AUDIT.md", "AUDIT_V1.md", "AUDIT_V2.md"])
    def test_excludes_audit_reports(self, pkg, tmp_path, name):
        # PKG-06 (audit v2) : un rapport d'audit interne (vulnérabilités non
        # corrigées, chemins perso) ne doit JAMAIS partir dans le ZIP distribué,
        # même recréé à la racine par erreur.
        f = tmp_path / name
        f.write_text("# audit")
        assert pkg.should_exclude(f, name) is True

    @pytest.mark.parametrize(
        "name",
        ["CLAUDE.md",                    # instructions assistant (PKG-04)
         "archive.tar.gz", "pkg.egg-info",  # PKG-05 : extensions composées —
         # path.suffix vaut ".gz"/".egg-info" ne matche jamais l'ancienne liste
         "module.pyc", "lib.so"],
    )
    def test_excludes_dev_files_and_compound_extensions(self, pkg, tmp_path, name):
        f = tmp_path / name
        f.write_text("x")
        assert pkg.should_exclude(f, name) is True


class TestZipSizeGuard:
    # PKG-05 (audit v2) : garde-fou — un ZIP anormalement gros (régression
    # type .venv embarqué) doit faire ÉCHOUER le build, pas livrer 750 Mo.
    def test_zip_trop_gros_leve(self, pkg, tmp_path):
        z = tmp_path / "main.zip"
        z.write_bytes(b"0" * 2048)
        with pytest.raises(RuntimeError):
            pkg.enforce_zip_size_guard(z, max_mb=0.001)

    def test_zip_normal_passe(self, pkg, tmp_path):
        z = tmp_path / "main.zip"
        z.write_bytes(b"0" * 2048)
        pkg.enforce_zip_size_guard(z, max_mb=1)
