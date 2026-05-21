"""WizardDialog — UI V2 (wizard 4 étapes).

QDialog + en-tête (titre dynamique par étape) + liseré de progression + rail
latéral (stepper) + QStackedWidget des 4 pages + barre d'actions. Isolé de
l'ancien ``MainDialog`` pour cohabiter pendant la migration.
"""
from __future__ import annotations

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
from .widgets.stepper_rail import StepperRail

try:
    from ..app.plugin_metadata import get_plugin_version
except Exception:  # pragma: no cover - défensif (hors QGIS)
    def get_plugin_version() -> str:
        return ""


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
        self._stack.addWidget(self._source_page)               # étape 1
        for n in (2, 3, 4):                                    # étapes 2-4 (placeholders)
            self._stack.addWidget(self._placeholder_page(n))
        body.addWidget(self._rail)
        body.addWidget(self._stack, 1)
        root.addLayout(body, 1)

        root.addWidget(self._build_action_bar())

        # Restaurer la config, brancher l'autosave + le sous-libellé du rail.
        self._source_page.load_from(self._config)
        self._source_page.changed.connect(self._autosave)
        self._source_page.mode_changed.connect(lambda _m: self._refresh_rail_subs())
        self._refresh_rail_subs()

        self._goto_step(1)

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

    def _placeholder_page(self, n: int) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(6)
        heading = QLabel(self.TITLES[n][0])
        heading.setObjectName("WizardPageHeading")
        placeholder = QLabel("(contenu à venir)")
        placeholder.setObjectName("WizardPagePlaceholder")
        layout.addWidget(heading)
        layout.addSpacing(8)
        layout.addWidget(placeholder)
        layout.addStretch(1)
        return page

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

    def _on_prev(self) -> None:
        self._goto_step(self._current_step - 1)

    def _on_next(self) -> None:
        if self._current_step < self._n_steps:
            self._goto_step(self._current_step + 1)
        else:
            self._on_launch()

    def _on_launch(self) -> None:
        QMessageBox.information(
            self, "À venir", "Le lancement du pipeline sera câblé au Jalon 6."
        )

    def _refresh_rail_subs(self) -> None:
        self._rail.set_sub(1, self._source_page.summary())

    # ------------------------------------------------------------------
    # Persistance (autosave de last_ui_config.json)
    # ------------------------------------------------------------------
    def _collect_config(self) -> None:
        self._source_page.collect_into(self._config)

    def _autosave(self) -> None:
        if self._loading:
            return
        self._collect_config()
        try:
            self._cm.save_last_ui_config(self._config)
        except Exception:
            pass  # la persistance ne doit jamais bloquer l'UI

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
        finally:
            self._loading = False
        self._refresh_rail_subs()
        self._autosave()
