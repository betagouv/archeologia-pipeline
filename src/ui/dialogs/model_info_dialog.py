"""Dialog modal d'information sur un modèle CV.

Affiche le contenu du ``model_card.yaml`` (et le clustering du ``args.yaml``)
sous forme de sections pliables : ARCHITECTURE, INDICE RVT D'ENTRAÎNEMENT,
MNT D'ENTRAÎNEMENT, REGROUPEMENT (DBSCAN), NOTES & LIMITES. Un bouton
« Ouvrir le dossier » lance l'explorateur de fichiers sur ``model_dir``.

La logique de présentation (humanisation, alias non-canoniques, builders de
sections) est isolée dans :mod:`._model_info_data` (testable hors-QGIS) ; ce
module se contente de fabriquer les widgets Qt et de gérer les interactions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from qgis.PyQt.QtCore import Qt, QUrl, pyqtSignal
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...app.services.model_orchestrator import InstalledModel, load_model_card

from ._model_info_data import Section, build_sections


# ----------------------------------------------------------------------
# Lecture d'``args.yaml`` (best-effort, pour la section clustering)
# ----------------------------------------------------------------------
def _load_args_yaml(model_dir: Path) -> Optional[Dict[str, Any]]:
    """Charge ``<model_dir>/args.yaml`` si présent. Retourne ``None`` sinon.

    Échec silencieux (PyYAML manquant, YAML invalide…) : la section
    « REGROUPEMENT (DBSCAN) » sera simplement absente. Pas d'exception remontée.
    """
    f = model_dir / "args.yaml"
    if not f.is_file():
        return None
    try:
        import yaml  # import différé : pas de coût si la fonction n'est jamais appelée
    except ImportError:
        return None
    try:
        data = yaml.safe_load(f.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


# ----------------------------------------------------------------------
# Widgets internes (un par section, et leur header cliquable)
# ----------------------------------------------------------------------
class _SectionHeader(QFrame):
    """Header cliquable d'une section : chevron ▸/▾ + titre en majuscules."""

    clicked = pyqtSignal()

    def __init__(self, title: str, expanded: bool, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("ModelInfoSectionHeader")
        self.setCursor(Qt.PointingHandCursor)
        self._title = title
        self._expanded = expanded
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 6, 0, 6)
        lay.setSpacing(0)
        self._label = QLabel()
        self._label.setObjectName("ModelInfoSectionTitle")
        self._label.setTextFormat(Qt.RichText)
        lay.addWidget(self._label)
        lay.addStretch(1)
        self._refresh()

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self._refresh()

    def _refresh(self) -> None:
        chevron = "▾" if self._expanded else "▸"
        self._label.setText(
            f"<span style='color:#7a7a7a;'>{chevron}</span>&nbsp;&nbsp;"
            f"<b>{self._title}</b>"
        )

    def mousePressEvent(self, ev):  # noqa: N802 (signature Qt)
        if ev.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(ev)


class _SectionWidget(QFrame):
    """Section pliable : header + corps (lignes label/valeur)."""

    def __init__(self, section: Section, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("ModelInfoSection")
        self._expanded = not section.collapsed
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._header = _SectionHeader(section.title, self._expanded)
        self._header.clicked.connect(self._toggle)
        lay.addWidget(self._header)

        self._body = QFrame()
        self._body.setObjectName("ModelInfoSectionBody")
        body_lay = QVBoxLayout(self._body)
        body_lay.setContentsMargins(20, 4, 8, 10)
        body_lay.setSpacing(2)
        for row in section.rows:
            row_frame = QFrame()
            rl = QHBoxLayout(row_frame)
            rl.setContentsMargins(0, 1, 0, 1)
            rl.setSpacing(12)
            label = QLabel(row.label)
            label.setObjectName("ModelInfoRowLabel")
            label.setMinimumWidth(180)
            label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            value = QLabel(row.value)
            value.setObjectName(
                "ModelInfoRowValueMono" if row.mono else "ModelInfoRowValue"
            )
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            rl.addWidget(label)
            rl.addWidget(value, 1)
            body_lay.addWidget(row_frame)
        lay.addWidget(self._body)
        self._body.setVisible(self._expanded)

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self._header.set_expanded(self._expanded)
        self._body.setVisible(self._expanded)


# ----------------------------------------------------------------------
# Dialog principal
# ----------------------------------------------------------------------
class ModelInfoDialog(QDialog):
    """Fenêtre modale présentant les détails d'un :class:`InstalledModel`.

    Construite à partir du ``model_card.yaml`` (et du ``args.yaml`` si présent)
    via :func:`._model_info_data.build_sections`. Dégrade gracieusement si le
    YAML est absent ou illisible (en-tête tiré de ``InstalledModel``, sections
    omises).
    """

    def __init__(self, model: InstalledModel, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("ModelInfoDialog")
        self.setModal(True)
        self.setWindowTitle(f"Modèle — {model.display_name}")
        self.resize(620, 720)

        self._model = model

        # Lecture best-effort des YAML. Toute défaillance laisse ``card``/``args``
        # vides → l'en-tête se rabat sur les champs de InstalledModel et les
        # sections vides sont automatiquement omises par ``build_sections``.
        card: Dict[str, Any] = {}
        args: Optional[Dict[str, Any]] = None
        if model.model_dir is not None:
            loaded = load_model_card(model.model_dir)
            if isinstance(loaded, dict):
                card = loaded
            args = _load_args_yaml(model.model_dir)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header(card))

        # Zone scrollable pour les sections (modèles à beaucoup de classes /
        # limitations peuvent dépasser la hauteur initiale).
        scroll = QScrollArea()
        scroll.setObjectName("ModelInfoScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        content_lay = QVBoxLayout(content)
        content_lay.setContentsMargins(20, 4, 20, 12)
        content_lay.setSpacing(6)
        for section in build_sections(card, args):
            content_lay.addWidget(_SectionWidget(section))
        content_lay.addStretch(1)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        root.addWidget(self._build_footer())

    # ------------------------------------------------------------------
    def _build_header(self, card: Dict[str, Any]) -> QWidget:
        header = QFrame()
        header.setObjectName("ModelInfoHeader")
        lay = QVBoxLayout(header)
        lay.setContentsMargins(20, 16, 20, 14)
        lay.setSpacing(4)

        kicker = QLabel("MODÈLE ONNX")
        kicker.setObjectName("ModelInfoKicker")
        lay.addWidget(kicker)

        slug_text = str(card.get("id") or self._model.name)
        slug = QLabel(slug_text)
        slug.setObjectName("ModelInfoSlug")
        slug.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(slug)

        disp = str(card.get("display_name") or self._model.display_name or "").strip()
        if disp and disp != slug_text:
            sub = QLabel(disp)
            sub.setObjectName("ModelInfoDisplayName")
            sub.setWordWrap(True)
            lay.addWidget(sub)

        desc = card.get("description")
        if desc:
            d = QLabel(str(desc).strip())
            d.setObjectName("ModelInfoDescription")
            d.setWordWrap(True)
            lay.addWidget(d)

        meta_parts = []
        version = card.get("version")
        if version:
            meta_parts.append(f"Version {version}")
        status = str(card.get("status") or self._model.status or "").strip()
        if status:
            meta_parts.append(status)
        if meta_parts:
            meta = QLabel(" · ".join(meta_parts))
            meta.setObjectName("ModelInfoMeta")
            lay.addWidget(meta)

        return header

    def _build_footer(self) -> QWidget:
        footer = QFrame()
        footer.setObjectName("ModelInfoFooter")
        lay = QHBoxLayout(footer)
        lay.setContentsMargins(16, 10, 16, 12)
        lay.setSpacing(8)

        open_btn = QPushButton("📁  Ouvrir le dossier")
        open_btn.setObjectName("ModelInfoOpenDirButton")
        if self._model.model_dir is None or not self._model.model_dir.is_dir():
            open_btn.setEnabled(False)
            open_btn.setToolTip("Chemin du modèle indisponible sur le disque")
        else:
            open_btn.setToolTip(str(self._model.model_dir))
            open_btn.clicked.connect(self._open_dir)
        lay.addWidget(open_btn)

        lay.addStretch(1)

        close_btn = QPushButton("Fermer")
        close_btn.setObjectName("ModelInfoCloseButton")
        close_btn.setDefault(True)
        close_btn.clicked.connect(self.accept)
        lay.addWidget(close_btn)

        return footer

    def _open_dir(self) -> None:
        d = self._model.model_dir
        if d is not None and d.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))
