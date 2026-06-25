#!/usr/bin/env python3
"""
Script pour créer un ZIP du plugin prêt à distribuer.
Exclut tous les fichiers de développement (venv, pycache, git, tests, etc.)
"""

import configparser
import os
import zipfile
from pathlib import Path

# Dossier racine du plugin (ce script est dans dev/)
PLUGIN_ROOT = Path(__file__).resolve().parent.parent

# Nom du dossier racine DANS le ZIP (= dossier d'installation côté QGIS).
# NE PAS le versionner ni le renommer : c'est l'identité du plugin pour QGIS,
# elle doit rester stable d'une version à l'autre (sinon install dupliquée,
# réglages perdus). Le nom DU FICHIER ZIP, lui, reflète la version (cf. zip_filename).
PLUGIN_NAME = "archeologia"

# Patterns à exclure
EXCLUDE_DIRS = {
    ".git",
    ".githooks",
    ".pytest_cache",
    "__pycache__",
    ".claude",   # mémoire / plans de l'assistant (jamais distribués)
    "dev",       # Tout l'outillage développeur (requirements, runner_onnx, package_plugin)
    "tests",
    ".venv",
    "node_modules",
    "nouvelle_UI_a_supprimer_plus_tard",  # maquette de design (refonte V2)
    # AUDIT v2 PKG-04 : specs/doc internes, scripts dev, artefacts de build.
    "docs",
    "scripts",
    "dist",
    # État de dev / sorties / temp — jamais distribués chez l'utilisateur (cf. .gitignore ;
    # le plugin recrée ces dossiers au runtime s'il en a besoin, ex. data/temp_zones/).
    "temp_zones",
    "results",
    "output",
    "output_test",
    "temp",
}

EXCLUDE_FILES = {
    ".gitignore",
    ".gitkeep",
    ".talismanrc",
    "conftest.py",
    "config.json",
    "last_ui_config.json",  # état de session UI (jamais distribué)
    "class_color_registry.json",  # état runtime des couleurs (profil utilisateur)
    "pytest.ini",
    "run_tests.py",
    ".DS_Store",
    "Thumbs.db",
    "CLAUDE.md",  # instructions assistant (AUDIT v2 PKG-04)
}

# Taille maximale plausible du ZIP : un dépassement signale une régression
# d'exclusion (ex. virtualenv embarqué, cf. AUDIT v1 PKG-01 : 749 Mo) et doit
# faire ÉCHOUER le build plutôt que livrer l'artefact (AUDIT v2 PKG-05).
MAX_ZIP_SIZE_MB = 800

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".egg-info",
    ".egg",
    ".whl",
    ".tar.gz",
    ".log",
}

def should_exclude(path: Path, relative_path: str) -> bool:
    """Vérifie si un fichier/dossier doit être exclu."""
    name = path.name

    # Exclure les dossiers cachés (commençant par ".") ET les dossiers dev
    # explicitement listés. La règle "dossier caché" rend l'exclusion robuste :
    # aucun dossier en "." n'est requis au runtime QGIS, et elle attrape les
    # variantes que l'ancien match par nom EXACT laissait passer — .venv_dev
    # (≠ .venv), .ruff_cache, .superpowers, .mypy_cache… (cf. AUDIT PKG-01/03/05).
    if path.is_dir() and (name.startswith(".") or name in EXCLUDE_DIRS):
        return True

    # Exclure les fichiers spécifiques
    if path.is_file() and name in EXCLUDE_FILES:
        return True

    # Rapports d'audit internes (AUDIT*.md) : vulnérabilités non corrigées et
    # chemins personnels — jamais distribués, même recréés à la racine
    # (AUDIT v2 PKG-06 ; les rapports vivent normalement sous dev/).
    if path.is_file() and name.startswith("AUDIT") and path.suffix == ".md":
        return True

    # Logs de run du pipeline (pipeline_log_<horodatage>.txt) — état d'exécution local,
    # jamais distribué. On NE peut PAS exclure tous les .txt : classes.txt, dalles_urls.txt…
    # sont des fichiers runtime légitimes du plugin.
    if path.is_file() and name.startswith("pipeline_log_") and name.endswith(".txt"):
        return True

    # Exclure par extension — endswith et non path.suffix : les extensions
    # composées (.tar.gz, .egg-info) ne matchaient JAMAIS car suffix vaut
    # ".gz" / ".egg-info" sur le dernier point seulement (AUDIT v2 PKG-05).
    if path.is_file() and any(name.endswith(ext) for ext in EXCLUDE_EXTENSIONS):
        return True

    # Exclure les fichiers .pt et .pth (modèles PyTorch) - on garde seulement .onnx
    if path.is_file() and path.suffix in (".pt", ".pth"):
        return True

    return False


def enforce_zip_size_guard(zip_path: Path, max_mb: float = MAX_ZIP_SIZE_MB) -> None:
    """Échoue si le ZIP dépasse la taille plausible (régression d'exclusion)."""
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        raise RuntimeError(
            f"ZIP anormalement gros ({size_mb:.0f} Mo > {max_mb:.0f} Mo) : "
            "régression d'exclusion probable (venv/caches/données dev). "
            "Build refusé — vérifier should_exclude avant de distribuer."
        )


def _read_metadata_field(field: str) -> str | None:
    """Valeur de ``[general] <field>`` dans ``metadata.txt`` (source unique, =
    celle qu'inspecte QGIS). ``None`` si absente/illisible."""
    try:
        parser = configparser.ConfigParser()
        if not parser.read(PLUGIN_ROOT / "metadata.txt", encoding="utf-8"):
            return None
        value = parser.get("general", field, fallback="").strip()
        return value or None
    except (configparser.Error, OSError):
        return None


def _read_deploy_config() -> dict:
    """Réglages de déploiement **locaux** (non versionnés) = valeurs par défaut de la CLI.

    Lus depuis ``dev/docs/_local/deploy.config.json`` (gitignoré) s'il existe :
    ``{"repo_url": "...", "output_dir": "..."}``. Permet à ``python dev/package_plugin.py``
    (sans argument) de produire ZIP + plugins.xml au bon endroit, **sans jamais committer
    l'URL du dépôt privé** (le dépôt distant est public). Absent / illisible → ``{}``
    (comportement par défaut : pas de plugins.xml, ZIP dans le dossier parent)."""
    import json

    cfg_path = PLUGIN_ROOT / "dev" / "docs" / "_local" / "deploy.config.json"
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _build_zip_name(plugin_id: str, version: str | None) -> str:
    """Nom du ZIP au format **dépôt QGIS** ``<plugin_id>.<version>.zip`` (pur, testable).

    Ex. ``archeologia.0.7.0.zip`` ; repli ``<plugin_id>.zip`` si la version est absente.

    ⚠️ Invariant critique : à l'installation **depuis un dépôt**, QGIS déduit le dossier
    d'installation du **nom de fichier**, coupé au **premier point**
    (``archeologia.0.7.0.zip`` → dossier attendu ``archeologia``). Ce préfixe DOIT être
    identique au dossier racine du ZIP (= ``PLUGIN_NAME``), sinon l'install échoue avec
    « répertoire mal nommé ». Ne JAMAIS repasser au nom affiché ni à un suffixe ``_v`` :
    ``ArcheologIA_v0.7.0.zip`` donnerait le préfixe ``ArcheologIA_v0`` ≠ ``archeologia``."""
    return f"{plugin_id}.{version}.zip" if version else f"{plugin_id}.zip"


def zip_filename() -> str:
    """Nom du ZIP : ``archeologia.<version>.zip`` (ex. ``archeologia.0.7.0.zip``).

    Préfixe = ``PLUGIN_NAME`` (dossier racine du ZIP, identité QGIS) ; version lue dans
    ``[general] version`` de metadata.txt (suit les bumps). Voir `_build_zip_name` pour
    l'invariant de nommage QGIS."""
    return _build_zip_name(PLUGIN_NAME, _read_metadata_field("version"))


def _build_plugins_xml(base_url: str) -> str:
    """Contenu du ``plugins.xml`` (catalogue du dépôt QGIS), pur et testable, dérivé de
    metadata.txt.

    ``file_name`` = ``zip_filename()`` (``archeologia.<version>.zip``) et ``download_url``
    = ``base_url`` + ce nom — c'est ce que QGIS télécharge ; les deux DOIVENT rester
    cohérents avec le ZIP produit (cf. invariant de nommage dans `_build_zip_name`)."""
    from xml.sax.saxutils import escape, quoteattr

    fn = zip_filename()
    download_url = base_url.rstrip("/") + "/" + fn
    g = _read_metadata_field

    def el(tag: str, field: str) -> str:
        return f"    <{tag}>{escape(g(field) or '')}</{tag}>\n"

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<plugins>\n"
        f"  <pyqgis_plugin name={quoteattr(g('name') or PLUGIN_NAME)} "
        f"version={quoteattr(g('version') or '')}>\n"
        + el("description", "description")
        + el("about", "about")
        + el("qgis_minimum_version", "qgisMinimumVersion")
        + el("qgis_maximum_version", "qgisMaximumVersion")
        + f"    <file_name>{escape(fn)}</file_name>\n"
        + f"    <download_url>{escape(download_url)}</download_url>\n"
        + el("author_name", "author")
        + el("homepage", "homepage")
        + el("tracker", "tracker")
        + el("repository", "repository")
        + el("tags", "tags")
        + el("experimental", "experimental")
        + el("deprecated", "deprecated")
        + "  </pyqgis_plugin>\n"
        "</plugins>\n"
    )


def write_plugins_xml(base_url: str, output_dir: Path = None) -> Path:
    """Écrit ``plugins.xml`` à côté du ZIP, à partir de metadata.txt + `_build_plugins_xml`.

    ``base_url`` = URL de base où le ZIP sera déposé (ex. ``http://hote/qgis/``).
    Le fichier produit n'est PAS versionné (il contient l'URL du dépôt privé)."""
    if output_dir is None:
        output_dir = PLUGIN_ROOT.parent
    xml_path = output_dir / "plugins.xml"
    xml_path.write_text(_build_plugins_xml(base_url), encoding="utf-8")
    return xml_path


def create_plugin_zip(output_dir: Path = None) -> Path:
    """Crée le ZIP du plugin."""
    if output_dir is None:
        output_dir = PLUGIN_ROOT.parent

    zip_path = output_dir / zip_filename()
    
    print(f"Création du ZIP: {zip_path}")
    print(f"Source: {PLUGIN_ROOT}")
    print("-" * 60)
    
    files_added = 0
    files_excluded = 0
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(PLUGIN_ROOT):
            root_path = Path(root)
            relative_root = root_path.relative_to(PLUGIN_ROOT)
            
            # Filtrer les dossiers à exclure (modifie dirs in-place pour éviter la descente)
            dirs[:] = [d for d in dirs if not should_exclude(root_path / d, str(relative_root / d))]
            
            for file in files:
                file_path = root_path / file
                relative_path = relative_root / file
                
                if should_exclude(file_path, str(relative_path)):
                    files_excluded += 1
                    continue
                
                # Chemin dans le ZIP (avec le nom du plugin comme dossier racine)
                arcname = f"{PLUGIN_NAME}/{relative_path}"
                
                zf.write(file_path, arcname)
                files_added += 1
                print(f"  + {relative_path}")
    
    print("-" * 60)
    print(f"Fichiers ajoutés: {files_added}")
    print(f"Fichiers exclus: {files_excluded}")
    print(f"ZIP créé: {zip_path}")
    print(f"Taille: {zip_path.stat().st_size / (1024*1024):.1f} MB")

    enforce_zip_size_guard(zip_path)

    return zip_path


if __name__ == "__main__":
    import argparse

    _cfg = _read_deploy_config()  # valeurs par défaut locales (gitignorées)
    parser = argparse.ArgumentParser(
        description="Crée le ZIP du plugin (archeologia.<version>.zip) et, en option, le plugins.xml du dépôt."
    )
    parser.add_argument(
        "--repo-url",
        metavar="URL",
        default=_cfg.get("repo_url"),
        help="URL de base du dépôt (ex. http://hote/qgis/). Défaut : 'repo_url' de "
        "dev/docs/_local/deploy.config.json. Si présente, génère aussi plugins.xml.",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=_cfg.get("output_dir"),
        help="Dossier de sortie du ZIP + plugins.xml. Défaut : 'output_dir' du config "
        "local, sinon dossier parent du plugin. Chemin relatif = relatif au plugin.",
    )
    args = parser.parse_args()

    out_dir = None
    if args.output_dir:
        out_dir = Path(args.output_dir)
        if not out_dir.is_absolute():
            out_dir = PLUGIN_ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = create_plugin_zip(out_dir)
    if args.repo_url:
        xml_path = write_plugins_xml(args.repo_url, zip_path.parent)
        print(f"plugins.xml généré: {xml_path}")
        print(f"  file_name + download_url -> {zip_filename()}")
