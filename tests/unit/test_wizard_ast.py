"""Contrats de l'assistant vérifiés par AST (``src/ui`` n'est pas importable hors
QGIS) : un seul menu « Configuration ▾ », messages sous les champs de l'étape 1."""
import ast
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
WIZARD = RACINE / "src" / "ui" / "wizard_dialog.py"
STEP1 = RACINE / "src" / "ui" / "steps" / "step_1_source.py"


def _chaines(source: str) -> list:
    return [n.value for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def test_un_seul_bouton_configuration():
    chaines = _chaines(WIZARD.read_text(encoding="utf-8"))
    assert chaines.count("Configuration") == 1   # le bouton-menu ; Qt dessine la flèche
    for ancien in ("Charger une configuration", "Enregistrer la configuration", "Charger une config", "Enregistrer la config"):
        assert ancien not in chaines, ancien


def test_le_menu_porte_toutes_les_entrees():
    chaines = _chaines(WIZARD.read_text(encoding="utf-8"))
    for entree in ("Enregistrer sous…", "Gérer les configurations…", "Importer un fichier…",
                   "Tout remettre aux valeurs par défaut…", "Aucune configuration enregistrée"):
        assert entree in chaines, entree


def test_reinitialisation_globale_dans_le_menu_seulement():
    src = WIZARD.read_text(encoding="utf-8")
    assert src.count("_reset_to_defaults") == 2, "une définition et un seul branchement, depuis le menu"


def test_messages_sous_les_champs_de_l_etape_1():
    src = STEP1.read_text(encoding="utf-8")
    assert "message_champ(" in src
    assert src.count('setObjectName("FieldMessage")') == 2, "un message sous la source, un sous le dossier de sortie"
