"""Fiche d'un modèle de détection.

Affiche le contenu du ``model_card.yaml`` (et le clustering du ``args.yaml``)
sous forme de sections pliables : ARCHITECTURE, INDICE RVT D'ENTRAÎNEMENT,
MNT D'ENTRAÎNEMENT, REGROUPEMENT (DBSCAN), MÉTRIQUES D'ÉVALUATION (résumé +
courbes de ``entrainement/evaluation*/``), NOTES & LIMITES. Un bouton
« Ouvrir le dossier » lance l'explorateur de fichiers sur ``model_dir``.

:func:`corps_fiche_modele` fabrique le corps que l'assistant pose dans son panneau
latéral (onglet « Modèle ») ; :class:`ModelInfoDialog` n'est qu'une enveloppe pour
le rendu hors écran.

La logique de présentation (humanisation, alias non-canoniques, builders de
sections) est isolée dans :mod:`._model_info_data` (testable hors-QGIS) ; ce
module se contente de fabriquer les widgets Qt et de gérer les interactions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from qgis.PyQt.QtCore import Qt, QUrl, pyqtSignal
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...app.services.model_orchestrator import InstalledModel, load_model_card

from ...app.services.fiabilite import parse_fiabilite
from ..icons import colored_pixmap
from ..widgets.profil_scores import couleur_de_classe, figure_profil, ligne_essai_seuil
from ..widgets.vignette import ARDOISE, pastille
from ..widgets.vignette import pixmap_ajuste
from ._model_info_data import Section, build_sections
from .class_info_dialog import FicheDialog

# Largeur des courbes : celle du corps de section dans le panneau de 420 px.
# ponytail: fixée à l'ouverture, ne suit pas un redimensionnement — le clic ouvre
# l'image en taille réelle.
_LARGEUR_COURBE = 340


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
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._title = title
        self._expanded = expanded
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 6, 0, 6)
        lay.setSpacing(0)
        self._label = QLabel()
        self._label.setObjectName("ModelInfoSectionTitle")
        self._label.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self._label)
        lay.addStretch(1)
        self._refresh()

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self._refresh()

    def _refresh(self) -> None:
        chevron = "▾" if self._expanded else "▸"
        self._label.setText(
            f"<span style='color:#5a5a5a;'>{chevron}</span>&nbsp;&nbsp;"
            f"<b>{self._title}</b>"
        )

    def mousePressEvent(self, ev):  # noqa: N802 (signature Qt)
        if ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(ev)


class _Courbe(QLabel):
    """Courbe d'évaluation à la largeur du dialog, nette à toute densité d'écran ;
    un clic l'ouvre en taille réelle dans la visionneuse du système (quatre
    graphes sur 540 px restent petits)."""

    def __init__(self, chemin: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._chemin = chemin
        self.setPixmap(pixmap_ajuste(chemin, _LARGEUR_COURBE, dpr=self.devicePixelRatioF()))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Cliquer pour ouvrir l'image en taille réelle")

    def mousePressEvent(self, ev):  # noqa: N802 (signature Qt)
        if ev.button() == Qt.MouseButton.LeftButton:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._chemin))
        super().mousePressEvent(ev)


class _SectionWidget(QFrame):
    """Section pliable : header + corps (lignes label/valeur)."""

    def __init__(self, section: Section, parent: Optional[QWidget] = None,
                 extras: Sequence[QWidget] = ()):
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
            label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            value = QLabel(row.value)
            value.setObjectName(
                "ModelInfoRowValueMono" if row.mono else "ModelInfoRowValue"
            )
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            rl.addWidget(label)
            rl.addWidget(value, 1)
            body_lay.addWidget(row_frame)
        for titre, chemin in section.images:
            t = QLabel(titre)
            t.setObjectName("ModelInfoRowLabel")
            t.setContentsMargins(0, 8, 0, 2)
            body_lay.addWidget(t)
            body_lay.addWidget(_Courbe(chemin))
        for w in extras:          # figures construites par le dialog (profil des scores)
            body_lay.addWidget(w)
        lay.addWidget(self._body)
        self._body.setVisible(self._expanded)

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self._header.set_expanded(self._expanded)
        self._body.setVisible(self._expanded)


# ----------------------------------------------------------------------
# Le corps de la fiche, et l'enveloppe hors écran
# ----------------------------------------------------------------------
def titre_fiche_modele(model: InstalledModel) -> str:
    """« Modèle · <nom> », sans doublon si le nom commence déjà par « Modèle »."""
    nom = model.display_name or model.name
    return nom if nom.lower().startswith("modèle") else f"Modèle · {nom}"


def _figures_fiabilite(section: Section, card: Dict[str, Any], model_dir: Optional[Path]) -> list:
    """Sous FIABILITÉ DES DÉTECTIONS : le profil des scores de chaque classe
    (bandes de l'évaluation livrée avec le modèle) — rien si elle ne l'est pas."""
    if model_dir is None or not section.title.startswith("FIABILITÉ"):
        return []
    par_classe, _provenance = parse_fiabilite((card or {}).get("thresholds"))
    out: list = []
    for classe, cats in par_classe.items():
        fig = figure_profil(model_dir, classe, cats)
        if fig is None:
            continue
        titre = QLabel(f"Profil des scores — {classe}")
        titre.setObjectName("ModelInfoRowLabel")
        titre.setContentsMargins(0, 10, 0, 2)
        # Essai de seuil (sans effet sur le traitement) : ligne, niveaux,
        # précision et rappel au banc suivent la case.
        out.extend([titre, fig, ligne_essai_seuil(fig)])
    return out


def _entete(model: InstalledModel, card: Dict[str, Any], dpr: float) -> QWidget:
    header = QFrame()
    header.setObjectName("ModelInfoHeader")
    lay = QVBoxLayout(header)
    lay.setContentsMargins(20, 16, 20, 14)
    lay.setSpacing(4)

    nature = QHBoxLayout()
    nature.setSpacing(6)
    icone = QLabel()
    icone.setPixmap(colored_pixmap("modele", ARDOISE, 12, dpr=dpr))
    kicker = QLabel("MODÈLE DE DÉTECTION · ONNX")
    kicker.setObjectName("ModelInfoKicker")
    nature.addWidget(icone)
    nature.addWidget(kicker)
    nature.addStretch(1)
    lay.addLayout(nature)

    slug_text = str(card.get("id") or model.name)
    slug = QLabel(slug_text)
    slug.setObjectName("ModelInfoSlug")
    slug.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    lay.addWidget(slug)

    disp = str(card.get("display_name") or model.display_name or "").strip()
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
    status = str(card.get("status") or model.status or "").strip()
    if status:
        meta_parts.append(status)
    if meta_parts:
        meta = QLabel(" · ".join(meta_parts))
        meta.setObjectName("ModelInfoMeta")
        lay.addWidget(meta)

    # « Détecte : » les classes du modèle, chacune avec la pastille de sa couleur de
    # couche — le lien visible avec les fiches de structure (option A, 2026-10-08).
    classes = [c for c in (card.get("classes") or []) if isinstance(c, dict) and c.get("name")]
    if classes:
        ligne = QHBoxLayout()
        ligne.setSpacing(6)
        libelle = QLabel("Détecte :")
        libelle.setObjectName("ModelInfoMeta")
        ligne.addWidget(libelle)
        for c in classes:
            rgb = couleur_de_classe(str(c["name"])) or (42, 120, 214)
            point = QLabel()
            point.setPixmap(pastille(rgb, 9, dpr))
            nom = QLabel(str(c.get("label_fr") or c["name"]))
            nom.setObjectName("ModelInfoDetecte")
            ligne.addWidget(point)
            ligne.addWidget(nom)
        ligne.addStretch(1)
        lay.addLayout(ligne)
    return header


def corps_fiche_modele(model: InstalledModel) -> QWidget:
    """Le corps de la fiche d'un modèle : liseré ardoise, en-tête, sections pliables
    (profil des scores sous FIABILITÉ), « Ouvrir le dossier ». Dégrade gracieusement
    si le YAML est absent ou illisible (en-tête tiré de ``InstalledModel``, sections
    omises)."""
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

    hote = QWidget()
    hote.setObjectName("ModelInfoCorps")
    root = QVBoxLayout(hote)
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)
    # Liseré ardoise : la fiche d'un modèle est neutre ; la couleur d'une classe
    # est réservée à sa fiche (option A, 2026-10-08).
    lisere = QFrame()
    lisere.setObjectName("ModelInfoLisere")
    lisere.setFixedHeight(4)
    root.addWidget(lisere)
    root.addWidget(_entete(model, card, hote.devicePixelRatioF()))

    content = QWidget()
    content_lay = QVBoxLayout(content)
    content_lay.setContentsMargins(20, 4, 20, 12)
    content_lay.setSpacing(6)
    for section in build_sections(card, args, model.model_dir):
        extras = _figures_fiabilite(section, card, model.model_dir)
        content_lay.addWidget(_SectionWidget(section, extras=extras))

    open_btn = QPushButton("📁  Ouvrir le dossier")
    open_btn.setObjectName("ModelInfoOpenDirButton")
    dossier = model.model_dir
    if dossier is None or not dossier.is_dir():
        open_btn.setEnabled(False)
        open_btn.setToolTip("Chemin du modèle indisponible sur le disque")
    else:
        open_btn.setToolTip(str(dossier))
        open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(dossier))))
    content_lay.addSpacing(8)
    content_lay.addWidget(open_btn, 0, Qt.AlignmentFlag.AlignLeft)
    content_lay.addStretch(1)
    root.addWidget(content)
    return hote


class ModelInfoDialog(FicheDialog):
    """La fiche d'un modèle en dialogue (rendu hors écran)."""

    def __init__(self, model: InstalledModel, parent: Optional[QWidget] = None):
        super().__init__(corps_fiche_modele(model), titre_fiche_modele(model), "ModelInfoDialog",
                         parent, taille=(620, 720))
