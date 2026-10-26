"""La frise de l'étape 1 reste le sélecteur du point de départ (choix de l'utilisateur,
2026-10-24) : les étapes qui précèdent le point de départ sont ignorées, chaque stade
cliquable prend le focus clavier et se choisit à la touche Espace. Vérifié par AST
(``src/ui`` n'est pas importable hors QGIS)."""
import ast
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
STAGE = RACINE / "src" / "ui" / "widgets" / "stage_button.py"
STEP1 = RACINE / "src" / "ui" / "steps" / "step_1_source.py"
RAIL = RACINE / "src" / "ui" / "widgets" / "stepper_rail.py"
WIZARD = RACINE / "src" / "ui" / "wizard_dialog.py"


def _methodes(source: str) -> set:
    arbre = ast.parse(source)
    return {n.name for n in ast.walk(arbre) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def test_les_stades_sautes_sont_barres():
    src = STAGE.read_text(encoding="utf-8")
    assert "setStrikeOut(" in src, "le libellé d'un stade sauté doit être barré (QFont.setStrikeOut)"
    assert '"sautée"' in src


def test_les_stades_cliquables_ont_le_focus_clavier():
    src = STAGE.read_text(encoding="utf-8")
    assert "Qt.FocusPolicy.StrongFocus" in src
    assert "keyPressEvent" in _methodes(src)
    assert "Qt.Key.Key_Space" in src


def test_le_badge_dit_point_de_depart():
    src = STEP1.read_text(encoding="utf-8")
    assert '"POINT DE DÉPART"' in src and '"ENTRÉE"' not in src


def test_le_rail_prend_ses_etats_du_module_pur():
    wizard = WIZARD.read_text(encoding="utf-8")
    assert "_classify_error_step" not in wizard, "plus de tri des erreurs par mot-clé dans le wizard"
    assert "classer_erreurs(" in wizard and "etat_rail(" in wizard
    rail = RAIL.read_text(encoding="utf-8")
    assert "set_states" in _methodes(rail)
    assert '"à corriger"' in rail
