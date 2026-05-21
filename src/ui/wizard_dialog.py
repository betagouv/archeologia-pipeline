"""WizardDialog — UI V2 (wizard 4 étapes).

QDialog + en-tête (titre dynamique par étape) + liseré de progression + rail
latéral (stepper) + QStackedWidget des 4 pages + barre d'actions. Point d'entrée
unique de l'UI (lancé par ``main.py``).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from qgis.PyQt.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..config.config_manager import ConfigManager
from .steps.step_1_source import SourcePage
from .steps.step_2_indices import IndicesPage
from .steps.step_3_detection import DetectionPage
from .steps.step_4_launch import LaunchPage
from .widgets.stepper_rail import StepperRail

try:
    from ..app.plugin_metadata import get_plugin_version
except Exception:  # pragma: no cover - défensif (hors QGIS)
    def get_plugin_version() -> str:
        return ""


def _classify_error_step(err: str) -> int:
    """Range une erreur de validation sur l'étape qui la corrige (pastille rail).

    Indices/produits → étape 2 ; tout le reste (mode, dossier de sortie, chemins
    d'entrée) → étape 1. La détection (étape 3) est facultative : aucune erreur
    bloquante n'en provient.
    """
    low = err.lower()
    if "indice" in low or "produit" in low:
        return 2
    return 1


class WizardDialog(QDialog):
    """Dialogue principal V2 en wizard 4 étapes."""

    RAIL_STEPS = [
        {"label": "Source", "sub": "—"},
        {"label": "Indices", "sub": "—"},
        {"label": "Détection IA", "sub": "—", "optional": True},
        {"label": "Lancer", "sub": "Vérification & run"},
    ]
    TITLES = {
        1: ("Nouveau traitement LiDAR", "Points d'entrée du pipeline"),
        2: ("Indices de visualisation", "Rasters dérivés du MNT"),
        3: ("Détection automatique", "Sélection par entités archéologiques"),
        4: ("Lancer le pipeline", "Vérification & récapitulatif"),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._n_steps = len(self.RAIL_STEPS)
        self._current_step = 1
        self._plugin_root = Path(__file__).resolve().parents[2]
        self._cm = ConfigManager(self._plugin_root)
        self._config = self._cm.load_last_ui_config()
        self._loading = False
        self._validation_errors: list = []

        version = get_plugin_version() or ""
        suffix = f" — v{version}" if version else ""
        self.setWindowTitle(f"Archéolog'IA{suffix}")
        self.resize(980, 660)

        self._apply_theme()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_title_bar())
        root.addWidget(self._build_progress_liseret())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self._rail = StepperRail(self.RAIL_STEPS)
        self._rail.step_clicked.connect(self._goto_step)
        self._stack = QStackedWidget()
        self._source_page = SourcePage()
        self._indices_page = IndicesPage()
        self._detection_page = DetectionPage(self._plugin_root)
        self._launch_page = LaunchPage(self._plugin_root, self._config)
        self._stack.addWidget(self._source_page)               # étape 1
        self._stack.addWidget(self._indices_page)              # étape 2
        self._stack.addWidget(self._detection_page)            # étape 3
        self._stack.addWidget(self._launch_page)               # étape 4
        body.addWidget(self._rail)
        body.addWidget(self._stack, 1)
        root.addLayout(body, 1)

        root.addWidget(self._build_action_bar())

        # Restaurer la config, brancher l'autosave + le sous-libellé du rail.
        self._source_page.load_from(self._config)
        self._indices_page.load_from(self._config)
        self._indices_page.set_mode(self._source_page.current_mode())
        self._detection_page.load_from(self._config)
        self._detection_page.set_active_rvts(self._indices_page.active_rvt_keys())
        self._source_page.changed.connect(self._autosave)
        self._indices_page.changed.connect(self._autosave)
        self._indices_page.changed.connect(self._refresh_rail_subs)
        self._indices_page.changed.connect(self._sync_detection_rvts)
        self._detection_page.changed.connect(self._autosave)
        self._detection_page.changed.connect(self._refresh_rail_subs)
        self._detection_page.activate_rvt.connect(self._indices_page.activate_product)
        self._source_page.mode_changed.connect(self._on_mode_changed)
        self._launch_page.run_started.connect(self._on_run_started)
        self._launch_page.run_finished.connect(self._on_run_finished)
        self._refresh_rail_subs()

        self._goto_step(1)
        self._refresh_validation()

    def _on_mode_changed(self, mode: str) -> None:
        """Le mode (étape 1) pilote la neutralisation des sections de l'étape 2."""
        self._indices_page.set_mode(mode)
        self._refresh_rail_subs()

    def _sync_detection_rvts(self) -> None:
        """Propage les indices RVT actifs (étape 2) à l'étape 3 (alertes RVT)."""
        self._detection_page.set_active_rvts(self._indices_page.active_rvt_keys())

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_title_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("WizardTitleBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 8, 16, 8)
        titles = QVBoxLayout()
        titles.setSpacing(1)
        self._title_label = QLabel("")
        self._title_label.setObjectName("WizardTitle")
        self._subtitle_label = QLabel("")
        self._subtitle_label.setObjectName("WizardSubtitle")
        titles.addWidget(self._title_label)
        titles.addWidget(self._subtitle_label)
        load_btn = QPushButton("Charger config")
        load_btn.setObjectName("GhostButton")
        load_btn.clicked.connect(self._load_config_file)
        save_btn = QPushButton("Enregistrer Config")
        save_btn.clicked.connect(self._save_config_file)
        layout.addLayout(titles)
        layout.addStretch(1)
        layout.addWidget(load_btn)
        layout.addWidget(save_btn)
        return bar

    def _build_progress_liseret(self) -> QWidget:
        self._progress = QProgressBar()
        self._progress.setObjectName("WizardProgress")
        self._progress.setRange(0, self._n_steps)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(4)
        return self._progress

    def _build_action_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("WizardActionBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 10, 16, 10)
        self._prev_btn = QPushButton("←  Précédent")
        self._prev_btn.clicked.connect(self._on_prev)
        self._next_btn = QPushButton("Suivant  →")
        self._next_btn.setObjectName("WizardPrimaryButton")
        self._next_btn.clicked.connect(self._on_next)
        layout.addWidget(self._prev_btn)
        layout.addStretch(1)
        layout.addWidget(self._next_btn)
        return bar

    def _apply_theme(self) -> None:
        qss_path = Path(__file__).parent / "theme" / "v2.qss"
        try:
            if qss_path.is_file():
                self.setStyleSheet(qss_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def _goto_step(self, step: int) -> None:
        step = max(1, min(self._n_steps, int(step)))
        self._current_step = step
        self._stack.setCurrentIndex(step - 1)
        self._rail.set_current(step)
        self._progress.setValue(step)
        title, subtitle = self.TITLES[step]
        self._title_label.setText(title)
        self._subtitle_label.setText(f"Étape {step} sur {self._n_steps} · {subtitle}")
        self._prev_btn.setEnabled(step > 1)
        is_last = step == self._n_steps
        self._next_btn.setText("▶  Lancer le pipeline" if is_last else "Suivant  →")
        if is_last:
            self._update_launch_recap()
            self._apply_validation()
        else:
            self._update_launch_button()

    def _on_prev(self) -> None:
        self._goto_step(self._current_step - 1)

    def _on_next(self) -> None:
        if self._current_step < self._n_steps:
            self._goto_step(self._current_step + 1)
        else:
            self._on_launch()

    def _on_launch(self) -> None:
        if self._launch_page.is_running():
            return
        self._update_launch_recap()  # collecte la config + remplit le récap
        self._launch_page.start_run(copy.deepcopy(self._config))

    def _update_launch_recap(self) -> None:
        self._collect_config()
        files = (self._config.get("app") or {}).get("files") or {}
        rows = [
            ("Source", self._source_page.summary()),
            ("Dossier de sortie", files.get("output_dir") or "—"),
            ("Indices", self._indices_page.summary()),
            ("Détection IA", self._detection_page.summary()),
        ]
        self._launch_page.update_recap(rows)

    def _on_run_started(self) -> None:
        self._set_nav_enabled(False)

    def _on_run_finished(self) -> None:
        self._set_nav_enabled(True)

    def _set_nav_enabled(self, on: bool) -> None:
        """Verrouille la navigation pendant un run (l'annulation reste possible
        via le bouton dédié du RunView)."""
        self._prev_btn.setEnabled(on and self._current_step > 1)
        self._rail.setEnabled(on)
        if on:
            self._update_launch_button()  # respecte la validation en étape 4
        else:
            self._next_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Validation bloquante (rail + bandeau étape 4 + bouton Lancer)
    # ------------------------------------------------------------------
    def _refresh_validation(self) -> None:
        self._collect_config()
        self._apply_validation()

    def _compute_errors(self) -> list:
        try:
            from ..app.run_context import build_run_context, validate_run_context
            ctx = build_run_context(self._config)
            errors, _warnings = validate_run_context(ctx)
            return list(errors)
        except Exception as e:  # noqa: BLE001
            return [f"Configuration illisible : {e}"]

    def _apply_validation(self) -> None:
        """Met à jour rail + bandeau + bouton à partir de ``self._config`` (déjà
        collecté). Les erreurs sont réparties par étape pour les pastilles."""
        errors = self._compute_errors()
        self._validation_errors = errors
        by_step: dict = {}
        for err in errors:
            by_step.setdefault(_classify_error_step(err), []).append(err)
        self._rail.set_errors(by_step)
        self._launch_page.set_validation(errors)
        self._update_launch_button()

    def _update_launch_button(self) -> None:
        if self._launch_page.is_running():
            self._next_btn.setEnabled(False)
            return
        if self._current_step == self._n_steps:
            ok = not self._validation_errors
            self._next_btn.setEnabled(ok)
            self._next_btn.setToolTip(
                "" if ok
                else "Corrigez avant de lancer :\n• " + "\n• ".join(self._validation_errors)
            )
        else:
            self._next_btn.setEnabled(True)
            self._next_btn.setToolTip("")

    def _refresh_rail_subs(self) -> None:
        self._rail.set_sub(1, self._source_page.summary())
        self._rail.set_sub(2, self._indices_page.summary())
        self._rail.set_sub(3, self._detection_page.summary())

    # ------------------------------------------------------------------
    # Persistance (autosave de last_ui_config.json)
    # ------------------------------------------------------------------
    def _collect_config(self) -> None:
        self._source_page.collect_into(self._config)
        self._indices_page.collect_into(self._config)
        self._detection_page.collect_into(self._config)

    def _autosave(self) -> None:
        if self._loading:
            return
        self._collect_config()
        try:
            self._cm.save_last_ui_config(self._config)
        except Exception:
            pass  # la persistance ne doit jamais bloquer l'UI
        self._apply_validation()  # config déjà collectée

    def closeEvent(self, event):  # noqa: N802 (signature Qt)
        try:
            self._collect_config()
            self._cm.save_last_ui_config(self._config)
        except Exception:
            pass
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Import / export d'un fichier de config (boutons de l'en-tête)
    # ------------------------------------------------------------------
    def _save_config_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer la configuration", "", "Configuration (*.json);;Tous (*.*)"
        )
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"
        self._collect_config()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Erreur", f"Impossible d'enregistrer : {e}")

    def _load_config_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Charger une configuration", "", "Configuration (*.json);;Tous (*.*)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Erreur", f"Impossible de charger : {e}")
            return
        if isinstance(loaded, dict) and isinstance(loaded.get("ui_config"), dict):
            loaded = loaded["ui_config"]  # compat ancien wrapper
        cfg = self._cm.default_config()
        self._cm._deep_update(cfg, loaded if isinstance(loaded, dict) else {})
        self._cm._migrate_cv_runs(cfg)
        self._config = cfg
        self._loading = True
        try:
            self._source_page.load_from(self._config)
            self._indices_page.load_from(self._config)
            self._indices_page.set_mode(self._source_page.current_mode())
            self._detection_page.load_from(self._config)
            self._detection_page.set_active_rvts(self._indices_page.active_rvt_keys())
        finally:
            self._loading = False
        self._refresh_rail_subs()
        self._autosave()
