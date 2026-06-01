"""WizardDialog — UI V2 (wizard 4 étapes).

QDialog + en-tête (titre dynamique par étape) + liseré de progression + rail
latéral (stepper) + QStackedWidget des 4 pages + barre d'actions. Point d'entrée
unique de l'UI (lancé par ``main.py``).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from qgis.PyQt.QtCore import Qt
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

from ..app.progress_stages import Stage
from ..app.services.indices_model import product, rvt_keys
from ..app.services.source_modes import mode_info
from ..config.config_manager import ConfigManager
from .steps.step_1_source import SourcePage
from .steps.step_2_indices import IndicesPage
from .steps.step_3_detection import DetectionPage
from .steps.step_4_launch import LaunchPage, RecapSection
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
        self._review_mode = False  # consultation lecture seule pendant un run
        self._plugin_root = Path(__file__).resolve().parents[2]
        self._cm = ConfigManager(self._plugin_root)
        self._config = self._cm.load_last_ui_config()
        self._loading = False
        self._validation_errors: list = []

        version = get_plugin_version() or ""
        suffix = f" — v{version}" if version else ""
        self.setWindowTitle(f"Archéolog'IA{suffix}")
        # Bouton « réduire » dans la barre de titre native (cf. main.py:run()
        # qui restaure via show() au reclic sur l'icône du plugin).
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint)
        self.resize(980, 660)

        self._apply_theme()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_title_bar())
        root.addWidget(self._build_progress_liseret())

        # Bandeau « lecture seule » affiché sur les étapes 1-3 pendant un run.
        self._review_banner = QLabel("🔒  Lecture seule — run en cours")
        self._review_banner.setObjectName("ReviewBanner")
        self._review_banner.setVisible(False)
        root.addWidget(self._review_banner)

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
        self._launch_page.workers_changed.connect(self._on_workers_changed)
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
        self._update_review_banner()

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
        proc = self._config.get("processing") or {}
        cv = self._config.get("computer_vision") or {}
        mode = files.get("data_mode") or "ign_laz"

        # Workers : la config alimente le spin des paramètres avancés ; le spin
        # devient ensuite la source de vérité (workers_changed réécrit la config).
        try:
            workers = int(proc.get("max_workers", 4) or 4)
        except (TypeError, ValueError):
            workers = 4
        self._launch_page.set_workers(workers)
        self._config.setdefault("processing", {})["max_workers"] = self._launch_page.workers_value()

        sections = [RecapSection("Mode", badges=[mode_info(mode).banner_label])]

        prods = self._indices_page.recap_products()
        res = self._indices_page.resolution()
        sections.append(RecapSection("Produits raster", badges=prods, value=f"{res:g} m/pixel"))

        if not cv.get("enabled"):
            sections.append(RecapSection("Détection IA", value="désactivée"))
        else:
            ents = self._detection_page.recap_entities()
            runs = self._detection_page.recap_runs()
            detail = (
                f"→ {len(runs)} run{'s' if len(runs) > 1 else ''} · " + " · ".join(runs)
                if runs else ""
            )
            val = "" if (ents or runs) else "aucune entité"
            sections.append(RecapSection("Détection IA", badges=ents, value=val, detail=detail))

        sections.append(RecapSection("Sortie", value=files.get("output_dir") or "—"))
        # (Plus de ligne « Performance » : les workers sont dans les paramètres avancés.)

        self._launch_page.update_recap(sections)
        self._launch_page.set_step_subtitles(self._step_subtitles())
        self._launch_page.refresh_preflight(self._config)
        if not self._launch_page.is_running():
            self._launch_page.show_recap()

    def _step_subtitles(self) -> dict:
        """Sous-libellés statiques de la timeline (depuis la config courante)."""
        proc = self._config.get("processing") or {}
        cv = self._config.get("computer_vision") or {}
        try:
            res = float(proc.get("mnt_resolution", 0.5) or 0.5)
        except (TypeError, ValueError):
            res = 0.5
        active = self._indices_page.active_rvt_keys()
        rvt_tags = [product(k).tag for k in rvt_keys() if k in active]
        # Pastille « Produits » = MNT + indices fusionnés : on combine la
        # résolution et les tags RVT en un seul sous-libellé.
        products_sub = " · ".join([f"{res:g} m/pixel", *rvt_tags])
        subs = {
            Stage.DOWNLOAD: "",  # téléchargement : comptage de dalles différé (préflight)
            Stage.PRODUCTS: products_sub,
            Stage.FINALIZE: "VRT · projet QGIS",
        }
        if cv.get("enabled"):
            n = self._detection_page.model_count()
            subs[Stage.DETECTION] = f"{n} modèle{'s' if n > 1 else ''}" if n else "—"
        else:
            subs[Stage.DETECTION] = "désactivée"
        return subs

    def _on_workers_changed(self, n: int) -> None:
        """Persiste le choix de workers (paramètres avancés de l'étape 4)."""
        self._config.setdefault("processing", {})["max_workers"] = int(n)
        try:
            self._cm.save_last_ui_config(self._config)
        except Exception:
            pass  # la persistance ne doit jamais bloquer l'UI

    def _on_run_started(self) -> None:
        self._enter_review_mode()

    def _on_run_finished(self) -> None:
        self._exit_review_mode()

    def _enter_review_mode(self) -> None:
        """Pendant un run : navigation conservée mais étapes 1-3 en lecture seule.

        L'utilisateur peut revenir consulter les paramètres lancés (rail +
        Précédent/Suivant restent actifs) sans pouvoir rien modifier ; seule la
        relance (bouton « Lancer » de l'étape 4) reste désactivée.
        """
        self._review_mode = True
        self._source_page.set_readonly(True)
        self._indices_page.set_readonly(True)
        self._detection_page.set_readonly(True)
        self._rail.setEnabled(True)
        self._prev_btn.setEnabled(self._current_step > 1)
        self._update_launch_button()
        self._update_review_banner()

    def _exit_review_mode(self) -> None:
        """Fin/annulation du run : restaure l'édition complète des étapes 1-3."""
        self._review_mode = False
        self._source_page.set_readonly(False)
        self._indices_page.set_readonly(False)
        self._detection_page.set_readonly(False)
        self._rail.setEnabled(True)
        self._prev_btn.setEnabled(self._current_step > 1)
        self._update_launch_button()
        self._update_review_banner()

    def _update_review_banner(self) -> None:
        """Affiche le bandeau « Lecture seule » uniquement sur les étapes 1-3
        pendant un run (masqué sur l'étape 4 = RunView, et hors run)."""
        show = self._review_mode and self._current_step < self._n_steps
        self._review_banner.setVisible(show)

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
        running = self._launch_page.is_running()
        if self._current_step == self._n_steps:
            # Étape 4 : pendant un run, jamais de relance possible.
            if running:
                self._next_btn.setEnabled(False)
                self._next_btn.setToolTip("Run en cours — relance impossible")
                return
            ok = not self._validation_errors
            self._next_btn.setEnabled(ok)
            self._next_btn.setToolTip(
                "" if ok
                else "Corrigez avant de lancer :\n• " + "\n• ".join(self._validation_errors)
            )
        else:
            # Étapes 1-3 : « Suivant » = navigation pure, autorisée même pendant
            # un run (permet de remonter jusqu'à l'étape 4 = suivi du run).
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
