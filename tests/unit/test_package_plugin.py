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
         "docs", "scripts", "dist",
         # État de dev / sorties / temp — ne doivent pas fuiter dans le ZIP utilisateur.
         "temp_zones", "results", "output", "output_test", "temp"],
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

    @pytest.mark.parametrize(
        "name",
        ["TA_diff_pkk_lidarhd_classe.shp", "TA_diff_pkk_lidarhd_classe.dbf",
         "TA_diff_pkk_lidarhd_classe.shx", "TA_diff_pkk_lidarhd_classe.prj",
         "TA_diff_pkk_lidarhd_classe.qix"],
    )
    def test_keeps_quadrillage_files(self, pkg, tmp_path, name):
        # PKG-02 : le quadrillage IGN (shapefile + son index spatial .qix) est
        # requis au runtime — l'index .qix (sélection des dalles sur carte) ne
        # doit pas être écarté du ZIP.
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

    @pytest.mark.parametrize("name", ["pipeline_log_20260618.txt", "pipeline_log_20260618_133000.txt"])
    def test_excludes_pipeline_logs(self, pkg, tmp_path, name):
        # Logs de run (.txt) : exclus par règle dédiée. L'extension .txt n'est PAS
        # exclue globalement (classes.txt / dalles_urls.txt sont légitimes).
        f = tmp_path / name
        f.write_text("log")
        assert pkg.should_exclude(f, name) is True

    @pytest.mark.parametrize("name", ["classes.txt", "dalles_urls.txt", "metadata.txt"])
    def test_keeps_legit_txt_files(self, pkg, tmp_path, name):
        # Garde-fou : la règle pipeline_log_*.txt ne doit pas emporter les .txt runtime.
        f = tmp_path / name
        f.write_text("x")
        assert pkg.should_exclude(f, name) is False


class TestZipFilename:
    """Le nom du ZIP suit la convention de **dépôt QGIS** ``<plugin_id>.<version>.zip``
    (ex. ``archeologia.0.7.0.zip``). Invariant : le préfixe avant le 1er point == le
    dossier racine du ZIP (``PLUGIN_NAME``), sinon l'install depuis un dépôt échoue
    (« répertoire mal nommé » — QGIS déduit le dossier du nom de fichier)."""

    @pytest.mark.parametrize(
        "plugin_id, version, expected",
        [
            ("archeologia", "0.7.0", "archeologia.0.7.0.zip"),
            ("archeologia", "1.2.3", "archeologia.1.2.3.zip"),
            ("archeologia", "2.0", "archeologia.2.0.zip"),
        ],
    )
    def test_build_name_is_id_dot_version(self, pkg, plugin_id, version, expected):
        assert pkg._build_zip_name(plugin_id, version) == expected

    def test_unreadable_version_omits_suffix(self, pkg):
        # Version illisible : repli ``<plugin_id>.zip`` (pas de « ? » invalide sous Windows).
        assert pkg._build_zip_name("archeologia", None) == "archeologia.zip"

    def test_prefix_before_first_dot_equals_root_dir(self, pkg):
        # Cœur du bug corrigé : QGIS coupe au 1er point pour déduire le dossier d'install,
        # qui DOIT == le dossier racine du ZIP (PLUGIN_NAME).
        name = pkg._build_zip_name(pkg.PLUGIN_NAME, "0.7.0")
        assert name.split(".")[0] == pkg.PLUGIN_NAME

    def test_zip_filename_from_real_metadata_matches_pattern(self, pkg):
        # Intégration : lit le vrai metadata.txt → archeologia.<x.y.z>.zip (robuste aux bumps).
        import re

        fn = pkg.zip_filename()
        assert fn.startswith(pkg.PLUGIN_NAME + ".")
        assert re.fullmatch(rf"{pkg.PLUGIN_NAME}\.\d+\.\d+\.\d+\.zip", fn)


class TestPluginsXml:
    """Génération du plugins.xml du dépôt (`--repo-url`) : file_name/download_url
    cohérents avec le ZIP produit, et XML bien formé."""

    def test_file_name_and_download_match_zip(self, pkg):
        xml = pkg._build_plugins_xml("http://host/qgis/")
        fn = pkg.zip_filename()
        assert f"<file_name>{fn}</file_name>" in xml
        assert f"<download_url>http://host/qgis/{fn}</download_url>" in xml

    def test_file_name_prefix_is_plugin_root(self, pkg):
        # Invariant QGIS : file_name commence par le dossier racine du ZIP.
        xml = pkg._build_plugins_xml("http://host/qgis")  # sans slash final
        assert f"<file_name>{pkg.PLUGIN_NAME}." in xml
        assert f"http://host/qgis/{pkg.PLUGIN_NAME}." in xml  # slash ajouté proprement

    def test_is_wellformed_xml(self, pkg):
        import xml.etree.ElementTree as ET

        ET.fromstring(pkg._build_plugins_xml("http://host/qgis"))  # ne doit pas lever


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


class TestEntrainementEvaluation:
    # 2026-09-29 : la fiche ⓘ d'un modèle affiche ses courbes d'évaluation, donc le
    # ZIP embarque entrainement/evaluation*/ (*.png + metriques_eval.json) et rien
    # d'autre de entrainement/ — surtout pas appariements.json (jusqu'à 46 Mo).
    _BASE = "data/models/m/entrainement"

    def _chemin(self, tmp_path, rel, dossier=False):
        p = tmp_path / rel
        if dossier:
            p.mkdir(parents=True, exist_ok=True)
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
        return p

    @pytest.mark.parametrize("rel", ["", "/evaluation", "/evaluation_couverture"])
    def test_descend_dans_les_dossiers_d_evaluation(self, pkg, tmp_path, rel):
        p = self._chemin(tmp_path, self._BASE + rel, dossier=True)
        assert pkg.should_exclude(p, self._BASE + rel) is False

    @pytest.mark.parametrize("rel", ["/comparaison_seg_v1", "/evaluation/sous_dossier"])
    def test_ne_descend_pas_ailleurs(self, pkg, tmp_path, rel):
        p = self._chemin(tmp_path, self._BASE + rel, dossier=True)
        assert pkg.should_exclude(p, self._BASE + rel) is True

    @pytest.mark.parametrize("rel", [
        "/evaluation/courbes_seuils_pr.png",
        "/evaluation/metriques_eval.json",
        "/evaluation_couverture/zones_et_masques.png",
    ])
    def test_embarque_courbes_et_resume(self, pkg, tmp_path, rel):
        p = self._chemin(tmp_path, self._BASE + rel)
        assert pkg.should_exclude(p, self._BASE + rel) is False

    @pytest.mark.parametrize("rel", [
        "/evaluation/appariements.json",
        "/evaluation/provenance_outils.txt",
        "/evaluation/desktop.ini",
        "/metrics.csv",
        "/hparams.yaml",
        "/events.out.tfevents.1789048309.x.0",
        "/courbe_hors_evaluation.png",
    ])
    def test_exclut_le_reste(self, pkg, tmp_path, rel):
        p = self._chemin(tmp_path, self._BASE + rel)
        assert pkg.should_exclude(p, self._BASE + rel) is True
