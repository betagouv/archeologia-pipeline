"""Lexique de l'interface (lot 1 de l'interface rationalisée, 2026-10-24) :
un seul mot par objet, des nombres à la française, les niveaux de fiabilité
au masculin singulier, les contrôles de l'environnement en français."""
import ast
from pathlib import Path

import pytest

from app.services.vocabulaire import (
    ENVIRONNEMENT,
    libelle_environnement,
    niveau,
    nombre,
    pluriel,
    resume_detail,
    titre_dossier_image,
    titre_image,
)

RACINE = Path(__file__).resolve().parents[2]


def test_titre_image_sigle_et_metier():
    assert titre_image("LD") == "LD · Structures en relief"
    assert titre_image("SVF") == "SVF · Creux & dépressions"
    assert titre_image("MNT") == "MNT · Relief nu"
    assert titre_image("M_HS") == "M-HS · Ombrage du relief"


def test_titre_image_inconnue_rend_la_cle():
    assert titre_image("XYZ") == "XYZ"


def test_nombre_virgule_et_milliers():
    assert nombre(0.25) == "0,25"
    assert nombre(0.245, 3) == "0,245"
    assert nombre(1234.5, 1) == "1\u202f234,5"   # espace fine insécable
    assert nombre(12, 0) == "12"
    assert nombre(952, 0) == "952"


def test_niveau_masculin_singulier_minuscule():
    assert [niveau(c) for c in ("quasi_certain", "probable", "possible", "douteux")] == [
        "très probable", "probable", "possible", "douteux",
    ]


def test_niveau_inconnu_rend_la_cle():
    assert niveau("autre") == "autre"


@pytest.mark.parametrize("n, attendu", [(0, "0 dalle"), (1, "1 dalle"), (12, "12 dalles")])
def test_pluriel(n, attendu):
    assert pluriel(n, "dalle") == attendu


def test_pluriel_irregulier():
    assert pluriel(2, "journal", "journaux") == "2 journaux"


def noms_preflight() -> set:
    """Les ``name=`` littéraux des ``CheckResult(...)`` de ``src/pipeline/preflight.py``."""
    arbre = ast.parse((RACINE / "src" / "pipeline" / "preflight.py").read_text(encoding="utf-8"))
    noms = set()
    for node in ast.walk(arbre):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "CheckResult":
            for kw in node.keywords:
                if kw.arg == "name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    noms.add(kw.value.value)
        # Les entrées (zone, liste de dalles, dossiers) passent par ``_check_input_path(cfg, clé, label)`` :
        # leur ``label`` devient le ``name`` du CheckResult (recette 65.5, défaut vu le 2026-10-26).
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_check_input_path":
            if len(node.args) >= 3 and isinstance(node.args[2], ast.Constant) and isinstance(node.args[2].value, str):
                noms.add(node.args[2].value)
    assert noms, "aucun CheckResult(name=...) littéral trouvé : le test ne vérifie rien"
    return noms


def test_environnement_couvre_tous_les_controles_du_preflight():
    manquants = noms_preflight() - set(ENVIRONNEMENT)
    assert not manquants, f"contrôles sans libellé français : {sorted(manquants)}"


def test_libelle_environnement_francais_et_repli():
    assert libelle_environnement("pdal (CLI)") == "Lecteur de nuages de points (pdal)"
    assert libelle_environnement("python:onnxruntime").startswith("Bibliothèque")
    assert libelle_environnement("python:inconnue") == "Bibliothèque Python inconnue"
    assert libelle_environnement("Dossier de sortie") == "Dossier de sortie"
    assert libelle_environnement("contrôle inconnu") == "contrôle inconnu"


def test_aucun_anglais_dans_les_libelles_environnement():
    for libelle in ENVIRONNEMENT.values():
        for mot in ("external", "not found", "skipped", "processing"):
            assert mot not in libelle.lower(), libelle


@pytest.mark.parametrize("details, attendu", [
    ("C:\\OSGeo4W\\bin\\pdal.exe", "trouvé"),
    ("/usr/bin/gdalwarp", "trouvé"),
    ("import ok", "trouvé"),
    ("import ok (version=1.0.1, file=C:\\x\\geopandas\\__init__.py)", "trouvé · version 1.0.1"),
    ("not found in PATH", "manquant"),
    ("not found (expected: C:\\x\\cv_runner_onnx.exe)", "manquant"),
    ("ModuleNotFoundError(\"No module named 'onnxruntime'\")", "manquant"),
    ("skipped (external runner present)", "non requis"),
    ("C:\\sortie · 120 Go libres", "trouvé · 120 Go libres"),
    ("1.17.0", "1.17.0"),
    ("EPSG:2154 · 16 dalle(s)", "EPSG:2154 · 16 dalle(s)"),
    ("", ""),
])
def test_resume_detail(details, attendu):
    """Le détail d'un contrôle se résume en un mot dans la colonne ; le texte complet
    (chemin, erreur) reste en infobulle."""
    assert resume_detail(details) == attendu


def test_titre_dossier_image():
    """N-03 : la mosaïque est nommée comme l'image de l'étape 2, les réglages du dossier à la suite."""
    assert titre_dossier_image("MNT") == "MNT · Relief nu"
    assert titre_dossier_image("SVF_R10_D16_V1_N0") == "SVF · Creux & dépressions · R10_D16_V1_N0"
    assert titre_dossier_image("M_HS_D16") == "M-HS · Ombrage du relief · D16"
    assert titre_dossier_image("RVT") == "RVT"
    assert titre_dossier_image("") == "index"


def test_lexique_visualisation_du_relief_et_produits():
    """Décision du 2026-10-25 : l'étape 2 s'appelle « Visualisation du relief », ce qu'elle
    calcule sont des « produits », dont les « indices de visualisation » (SVF, LD…)."""
    from app.services.vocabulaire import IMAGE, IMAGES, INDICE, INDICES, VISUALISATION
    assert (IMAGE, IMAGES) == ("produit", "produits")
    assert (INDICE, INDICES) == ("indice de visualisation", "indices de visualisation")
    assert VISUALISATION == "Visualisation du relief"
    src = (RACINE / "src" / "ui" / "wizard_dialog.py").read_text(encoding="utf-8")
    assert '{"label": "Visualisation du relief"' in src and '2: ("Visualisation du relief"' in src
    etape2 = (RACINE / "src" / "ui" / "steps" / "step_2_indices.py").read_text(encoding="utf-8")
    assert "'image')" not in etape2 and "du relief\"" not in etape2   # « MNT + 2 indices », pas « images du relief »
