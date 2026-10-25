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


# ---------------------------------------------------------------- lot 9 (2026-10-24)
RUN_VIEW = RACINE / "src" / "ui" / "run_view.py"
PAGE6 = RACINE / "src" / "ui" / "steps" / "step_6_resultats.py"


def test_six_etapes_dans_le_rail():
    src = WIZARD.read_text(encoding="utf-8")
    assert src.count('{"label": ') == 6 and '{"label": "Résultats"' in src
    assert "STEP_RESULTATS = 6" in src and "ResultatsPage(" in src
    assert "revoir_resultats.connect(" in src
    assert 'QPushButton("Revoir les résultats")' in RUN_VIEW.read_text(encoding="utf-8")
    assert "class ResultatsPage" in PAGE6.read_text(encoding="utf-8")
    from src.app.services.aide import CHAPITRE_PAR_ETAPE
    assert CHAPITRE_PAR_ETAPE[6] == "resultats"


def test_relancer_passe_par_apply_config():
    src = WIZARD.read_text(encoding="utf-8")
    corps = src[src.index("def _relancer_depuis_resultat("):src.index("def _on_mode_changed(")]
    assert "self._apply_config(" in corps and "self._goto_step(3)" in corps


def test_selection_sur_carte_sans_minimiser():
    """H-02 : choisir les dalles sur la carte ne réduit plus l'assistant ; il se range contre
    le bord de l'écran et reprend sa place à la fin."""
    src = STEP1.read_text(encoding="utf-8")
    assert "showMinimized" not in src
    assert "_geometrie_avant_carte" in src and "availableGeometry()" in src
