"""Sens des dépendances entre couches : ``app`` ne connaît jamais ``ui``.

``src/app/`` est la couche testable hors-QGIS, et plusieurs de ses modules sont
importés par le pipeline (``class_fiche``, ``fiabilite``, ``model_orchestrator``…).
``src/ui/`` importe Qt au chargement dans la quasi-totalité de ses modules, et
n'est PAS collecté par pytest (``conftest.collect_ignore_glob``) : un import
app → ui casserait donc l'exécution standalone et les tests, sans qu'aucun test
existant ne le voie — la casse n'apparaîtrait qu'au runtime dans QGIS.

Le cas réel : l'humanisation ``pretty_rvt_name`` / ``pretty_task`` vivait dans
``ui/dialogs/_model_info_data.py``. Elle a été descendue dans
``app/services/vocabulaire_modele.py`` quand la fiche de classe en a eu besoin,
et ``_model_info_data`` la ré-exporte. Ce test verrouille le sens.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
APP = SRC / "app"


def _modules_app() -> list[Path]:
    return sorted(p for p in APP.rglob("*.py") if "__pycache__" not in p.parts)


def _cibles_importees(source: str, chemin: Path) -> list[str]:
    """Modules visés par les imports, forme absolue ET relative résolue."""
    arbre = ast.parse(source, filename=str(chemin))
    cibles: list[str] = []
    # Profondeur du module dans ``src`` : app/services/x.py -> ['app', 'services']
    paquet = chemin.relative_to(SRC).parts[:-1]
    for noeud in ast.walk(arbre):
        if isinstance(noeud, ast.Import):
            cibles.extend(a.name for a in noeud.names)
        elif isinstance(noeud, ast.ImportFrom):
            if noeud.level:
                # ``from ...ui.x import y`` dans app/services/z.py :
                # niveau 1 = app.services, 2 = app, 3 = racine du plugin.
                base = list(paquet[: len(paquet) - (noeud.level - 1)])
                cibles.append(".".join(base + [noeud.module or ""]).strip("."))
            else:
                cibles.append(noeud.module or "")
    return cibles


@pytest.mark.parametrize("module", _modules_app(), ids=lambda p: str(p.relative_to(APP)))
def test_app_n_importe_pas_ui(module: Path) -> None:
    cibles = _cibles_importees(module.read_text(encoding="utf-8"), module)
    fautifs = [
        c for c in cibles
        if c == "ui" or c.startswith("ui.") or c.endswith(".ui") or ".ui." in c
    ]
    assert not fautifs, (
        f"{module.relative_to(SRC)} importe la couche ui ({fautifs}) — "
        "app doit rester importable sans QGIS. Descends le code partagé dans "
        "app/ et fais-le ré-exporter par ui/ si besoin."
    )


def test_le_module_partage_est_bien_dans_app() -> None:
    """Garde-fou du cas qui a motivé ce test."""
    assert (APP / "services" / "vocabulaire_modele.py").is_file()


def test_ui_reexporte_le_vocabulaire() -> None:
    """``_model_info_data`` doit continuer d'exposer les deux helpers : le
    dialog modèle et ses tests s'en servent par ce chemin."""
    source = (SRC / "ui" / "dialogs" / "_model_info_data.py").read_text(encoding="utf-8")
    assert "vocabulaire_modele import pretty_rvt_name, pretty_task" in source
