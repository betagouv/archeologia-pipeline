"""WizardDialog — squelette de l'UI V2 (wizard 4 étapes).

Jalon 3 : structure de navigation seule (rail latéral + QStackedWidget de 4
pages vides + barre d'actions). Les contenus d'étapes et le câblage au pipeline
arrivent aux jalons suivants. Le dialogue est volontairement isolé de l'ancien
``MainDialog`` pour pouvoir cohabiter pendant la migration.
"""
from __future__ import annotations

from pathlib import Path

from qgis.PyQt.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .widgets.stepper_rail import StepperRail

try:
    from ..app.plugin_metadata import get_plugin_version
except Exception:  # pragma: no cover - défensif (hors QGIS)
    def get_plugin_version() -> str:
        return ""


class WizardDialog(QDialog):
    """Dialogue principal V2 en wizard 4 étapes (squelette)."""

    STEPS = [
        ("Source", "D'où partent vos données"),
        ("Indices RVT", "Produits à générer"),
        ("Détection IA", "Entités à détecter"),
        ("Lancer", "Vérification & exécution"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_step = 1

        version = get_plugin_version() or ""
        suffix = f" · {version}" if version else ""
        self.setWindowTitle(f"Archéolog'IA — Pipeline LiDAR (V2{suffix})")
        self.resize(960, 640)

        self._apply_theme()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_title_bar())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self._rail = StepperRail(self.STEPS)
        self._rail.step_clicked.connect(self._goto_step)
        self._stack = QStackedWidget()
        for i, (label, sub) in enumerate(self.STEPS, start=1):
            self._stack.addWidget(self._placeholder_page(i, label, sub))
        body.addWidget(self._rail)
        body.addWidget(self._stack, 1)
        root.addLayout(body, 1)

        root.addWidget(self._build_action_bar())

        self._goto_step(1)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_title_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("WizardTitleBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 10, 16, 10)
        title = QLabel("Archéolog'IA — Pipeline LiDAR")
        title.setObjectName("WizardTitle")
        self._step_label = QLabel("")
        self._step_label.setObjectName("WizardStepLabel")
        layout.addWidget(title)
        layout.addStretch(1)
        layout.addWidget(self._step_label)
        return bar

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

    def _placeholder_page(self, n: int, title: str, sub: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(6)
        heading = QLabel(f"Étape {n} — {title}")
        heading.setObjectName("WizardPageHeading")
        subtitle = QLabel(sub)
        subtitle.setObjectName("WizardPageSub")
        placeholder = QLabel("(contenu à venir)")
        placeholder.setObjectName("WizardPagePlaceholder")
        layout.addWidget(heading)
        layout.addWidget(subtitle)
        layout.addSpacing(12)
        layout.addWidget(placeholder)
        layout.addStretch(1)
        return page

    def _apply_theme(self) -> None:
        qss_path = Path(__file__).parent / "theme" / "v2.qss"
        try:
            if qss_path.is_file():
                self.setStyleSheet(qss_path.read_text(encoding="utf-8"))
        except Exception:
            pass  # le thème est cosmétique : ne jamais bloquer l'ouverture

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def _goto_step(self, step: int) -> None:
        step = max(1, min(len(self.STEPS), int(step)))
        self._current_step = step
        self._stack.setCurrentIndex(step - 1)
        self._rail.set_current(step)
        self._step_label.setText(f"Étape {step} / {len(self.STEPS)}")
        self._prev_btn.setEnabled(step > 1)
        is_last = step == len(self.STEPS)
        self._next_btn.setText("▶  Lancer" if is_last else "Suivant  →")

    def _on_prev(self) -> None:
        self._goto_step(self._current_step - 1)

    def _on_next(self) -> None:
        if self._current_step < len(self.STEPS):
            self._goto_step(self._current_step + 1)
        else:
            self._on_launch()

    def _on_launch(self) -> None:
        QMessageBox.information(
            self,
            "À venir",
            "Le lancement du pipeline sera câblé au Jalon 6.",
        )
