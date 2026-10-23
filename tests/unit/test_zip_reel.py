"""Le ZIP réellement construit par ``dev/package_plugin.py`` (lot 4 du chantier d'outillage, 2026-10-20).

``test_package_plugin.py`` ne teste que ``should_exclude`` sur des noms inventés ; le packaging, lui, lit
le disque — et le ZIP 0.14.0 a embarqué ``data/local_catalogue/`` (clé d'accès) parce que le dossier était
là. Ici le ZIP est construit pour de vrai (dans ``tmp_path``, sans compression : même parcours, mêmes
exclusions, mêmes chemins d'archive) et son contenu comparé à ``git ls-files`` : tout ce qui est dans le
ZIP est versionné, sauf les données attendues et gitignorées (modèles, grille, binaire) ; tout ce qui est
versionné et non exclu est dans le ZIP.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PKG_SCRIPT = _REPO / "dev" / "package_plugin.py"

# Gitignorés mais livrés (CLAUDE.md § Packaging) : le ZIP peut les contenir sans qu'ils soient dans git.
_GITIGNORES_LIVRES = ("data/models/", "data/quadrillage_france/", "data/third_party/cv_runner_onnx/")


def _charger_package_plugin():
    spec = importlib.util.spec_from_file_location("package_plugin_zip_reel", _PKG_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fichiers_git() -> set[str]:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git introuvable : impossible de lister les fichiers versionnés")
    sortie = subprocess.run([git, "-C", str(_REPO), "ls-files", "-z"], capture_output=True, check=True).stdout
    return {f.decode("utf-8") for f in sortie.split(b"\0") if f}


def _exclu_par_le_packaging(pkg, rel: str) -> bool:
    """Un fichier versionné est exclu si lui ou l'un de ses dossiers l'est (os.walk ne descend pas)."""
    parties = Path(rel).parts
    for i in range(1, len(parties)):
        d = Path(*parties[:i])
        if pkg.should_exclude(_REPO / d, d.as_posix()):
            return True
    return pkg.should_exclude(_REPO / rel, rel)


@pytest.fixture(scope="module")
def zip_reel(tmp_path_factory):
    pkg = _charger_package_plugin()
    chemin = pkg.create_plugin_zip(tmp_path_factory.mktemp("zip"), compression=zipfile.ZIP_STORED)
    with zipfile.ZipFile(chemin) as z:
        noms = z.namelist()
    racine = pkg.PLUGIN_NAME + "/"
    assert noms and all(n.startswith(racine) for n in noms), "tout doit être sous le dossier racine archeologia/"
    return pkg, {n[len(racine):].replace("\\", "/") for n in noms}


def test_tout_ce_qui_part_est_versionne_ou_donnee_livree(zip_reel):
    """Incident 0.14.0 : un fichier présent sur le disque mais hors git (catalogue privé, sauvegarde,
    fichier de travail) ne doit pas partir chez l'utilisateur."""
    _pkg, contenu = zip_reel
    versionnes = _fichiers_git()
    intrus = sorted(f for f in contenu if f not in versionnes and not f.startswith(_GITIGNORES_LIVRES))
    assert intrus == [], f"dans le ZIP mais ni versionnés ni données livrées : {intrus}"


def test_tout_ce_qui_est_versionne_et_non_exclu_part(zip_reel):
    pkg, contenu = zip_reel
    attendus = {f for f in _fichiers_git() if not _exclu_par_le_packaging(pkg, f) and (_REPO / f).is_file()}
    manquants = sorted(attendus - contenu)
    assert manquants == [], f"versionnés, non exclus, absents du ZIP : {manquants}"


def test_rien_de_dev_ni_de_prive_ni_de_checkpoint(zip_reel):
    _pkg, contenu = zip_reel
    interdits = sorted(f for f in contenu
                       if f.startswith(("docs/", "dev/", "tests/", "data/local_catalogue/", ".github/", ".claude/"))
                       or f.endswith((".pt", ".pth")))
    assert interdits == []


def test_le_necessaire_au_runtime_part(zip_reel):
    _pkg, contenu = zip_reel
    rvt = "src/pipeline/ign/products/rvt_embarque/"
    for f in ("metadata.txt", "__init__.py", "main.py", "data/zones_corpus.json", "data/indices_fiches.json",
              "data/entities_catalog.json", rvt + "LICENSE", rvt + "NOTICE"):
        assert f in contenu, f"{f} absent du ZIP"
    assert any(f.startswith("aide/") and f.endswith(".md") for f in contenu), "manuel intégré (aide/*.md) absent"


def test_la_grille_et_le_binaire_partent(zip_reel, exiger_ou_sauter):
    """Gitignorés, présents sur le poste seulement (ARCHEO_EXIGER_DONNEES=1 les exige)."""
    _pkg, contenu = zip_reel
    grille = "data/quadrillage_france/TA_diff_pkk_lidarhd_classe"
    exe = "data/third_party/cv_runner_onnx/windows/cv_runner_onnx.exe"
    if not (_REPO / (grille + ".shp")).is_file() or not (_REPO / exe).is_file():
        exiger_ou_sauter("grille IGN ou binaire absents (poste principal seulement)")
    for f in (grille + ".shp", grille + ".qix", grille + ".dbf", exe):
        assert f in contenu, f"{f} absent du ZIP"
    assert any(f.startswith("data/models/") and f.endswith("/weights/best.onnx") for f in contenu), \
        "aucun modèle (data/models/*/weights/best.onnx) dans le ZIP"
