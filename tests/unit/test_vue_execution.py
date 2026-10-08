"""Vue d'exécution V2 (2026-10-08, validée par l'utilisateur) : la frise porte l'état,
le journal raconte, le projet QGIS s'ouvre au clic. Garde-fous sans QGIS (lecture du source)."""
from __future__ import annotations

import re
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]


def _vue() -> str:
    return (RACINE / "src/ui/run_view.py").read_text(encoding="utf-8")


def test_plus_de_carte_de_bandeau_ni_de_barre():
    vue = _vue()
    assert "QProgressBar" not in vue and "RunEndBanner" not in vue and "RunHeaderStep" not in vue
    assert "class _StepLine" in vue and "set_fill(" in vue          # la frise porte la progression mesurée
    assert '_fil_gauche.setObjectName("RunFil")' in vue and "_fil_droite" in vue


def test_le_projet_qgis_ne_s_ouvre_qu_au_clic():
    vue = _vue()
    on_load = vue[vue.index("def _on_load_layers("):vue.index("def _ouvrir_projet_qgis(")]
    assert "load_result_layers(" not in on_load and "_couches_a_charger = (" in on_load
    ouvrir = vue[vue.index("def _ouvrir_projet_qgis("):vue.index("def _write_validation_project(")]
    assert "load_result_layers(" in ouvrir and "_open_project_btn.setEnabled(False)" in ouvrir
    assert 'QPushButton("Ouvrir le projet QGIS")' in vue
    assert vue.count("load_result_layers(") == 1     # un seul appel, celui du clic
    # « Ouvrir le dossier » est dans le cadre de fin, à côté du projet (2026-10-08).
    assert "actions_fin.addWidget(self._open_dir_btn)" in vue
    assert "actions.addWidget(self._open_dir_btn)" not in vue


def test_le_journal_reste_accessible_apres_le_run():
    # Revenir d'une étape 1-3 sur l'étape 4 ne doit plus remplacer le journal
    # par le récap (constat utilisateur 2026-10-08) : le récap ne s'impose
    # qu'avant le premier run, et chaque sous-page a sa bascule.
    wizard = (RACINE / "src/ui/wizard_dialog.py").read_text(encoding="utf-8")
    assert "not self._launch_page.is_running() and not self._launch_page.has_run()" in wizard
    page = (RACINE / "src/ui/steps/step_4_launch.py").read_text(encoding="utf-8")
    assert "def has_run(" in page and "def show_run(" in page
    assert "_vers_journal_btn.clicked.connect(self.show_run)" in page
    assert "_vers_recap_btn.clicked.connect(self.show_recap)" in page


def test_un_seul_format_de_duree_et_la_synthese_dans_le_fil():
    vue = _vue()
    assert "return _format_duration(max(0.0, seconds))" in vue        # même format que le journal et le rapport
    assert not re.search(r'f"\{h\}:\{m:02d\}:\{s:02d\}"', vue)
    assert 'text = f"✓ Terminé en {elapsed}{suffix}"' in vue and "setPlaceholderText(" in vue


def test_le_manuel_et_la_recette_suivent():
    manuel = (RACINE / "aide/06-etape-4-lancer.md").read_text(encoding="utf-8")
    assert "Ouvrir le projet QGIS" in manuel and "barre de progression" not in manuel
    recette = (RACINE / "tests/TESTS_MANUELS_QGIS.md").read_text(encoding="utf-8")
    assert "## 44." in recette and "Ouvrir le projet QGIS" in recette
