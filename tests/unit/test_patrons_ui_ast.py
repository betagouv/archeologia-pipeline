"""Patrons uniques de l'interface rationalisée (2026-10-24), vérifiés par AST :
un seul bouton de fiche « ⓘ », un seul geste pour cocher, les réglages dans la
carte, les « ↺ » libellés par leur infobulle, le verrou du MNT sans toast."""
import ast
import re
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
UI = RACINE / "src" / "ui"
STEP2 = UI / "steps" / "step_2_indices.py"
STEP3 = UI / "steps" / "step_3_detection.py"
CARD = UI / "widgets" / "entity_card.py"
VIGNETTE = UI / "widgets" / "vignette.py"
DIALOG = UI / "dialogs" / "indice_info_dialog.py"


def _fonctions(source: str) -> dict:
    return {n.name: n for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _appels(node: ast.AST) -> set:
    noms = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            noms.add(f.id if isinstance(f, ast.Name) else getattr(f, "attr", ""))
    return noms


def test_un_seul_patron_de_fiche():
    for fichier in (STEP2, STEP3, CARD):
        src = fichier.read_text(encoding="utf-8")
        assert 'QPushButton("Fiche")' not in src, fichier.name
        assert "FicheButton(" not in src, fichier.name
    for fichier in (STEP2, CARD):
        assert "InfoButton" in fichier.read_text(encoding="utf-8"), fichier.name
    for fichier in list((UI / "steps").glob("*.py")) + list((UI / "widgets").glob("*.py")):
        assert "FicheButton" not in fichier.read_text(encoding="utf-8"), fichier


def test_mnt_verrou_sans_toast():
    fn = _fonctions(STEP2.read_text(encoding="utf-8"))["_on_product_clicked"]
    assert "show_toast" not in _appels(fn), "le verrou du MNT est une infobulle, plus un toast"


def test_etape_2_sans_vue_separee():
    src = STEP2.read_text(encoding="utf-8")
    for interdit in ("QStackedWidget", "QTabWidget", "QTabBar", "_build_advanced", "_make_param_tab"):
        assert interdit not in src, interdit
    assert "set_reglages(" in src, "les réglages se déplient dans la carte"


def test_reinit_tooltip_libellee():
    for fichier in (STEP2, CARD):
        src = fichier.read_text(encoding="utf-8")
        tooltips = re.findall(r'setToolTip\(\s*(?:f?"([^"]*)")', src)
        reinit = [t for t in tooltips if t.startswith("Revenir aux valeurs")]
        assert reinit, f"{fichier.name} : aucun « ↺ » libellé « Revenir aux valeurs … »"


def test_cartes_focusables_au_clavier():
    src = STEP2.read_text(encoding="utf-8")
    assert "Qt.FocusPolicy.StrongFocus" in src and "keyPressEvent" in _fonctions(src)


def test_lien_comparer_depuis_la_page():
    assert "Comparer les images ▸" in STEP2.read_text(encoding="utf-8")
    assert "__comparaison__" in DIALOG.read_text(encoding="utf-8")
