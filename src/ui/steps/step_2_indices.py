"""Étape 2 — Indices de visualisation.

Sélection des produits : MNT/Densité (modèle de base) + 6 indices RVT. Le MNT
est verrouillé tant qu'un indice RVT est coché (clic → toast explicatif). Selon
le mode de données, les sections déjà fournies sont masquées (existing_mnt /
existing_rvt). La logique pure vient de :mod:`app.services.indices_model`.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...app.services.indices_model import (
    all_products,
    count_selected,
    default_products,
    product,
    requires_mnt,
    rvt_keys,
    toggle,
)
from ..widgets.card import build_card
from ..widgets.toast import show_toast


class _IndexCard(QFrame):
    """Carte cliquable d'un indice RVT (tag + nom + description + coche)."""

    clicked = pyqtSignal(str)

    def __init__(self, key: str, tag: str, full_name: str, desc: str, parent=None):
        super().__init__(parent)
        self._key = key
        self.setObjectName("IndexCard")
        self.setProperty("checked", False)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)
        top = QHBoxLayout()
        self._tag = QLabel(tag)
        self._tag.setObjectName("IndexTag")
        self._check = QLabel("")
        self._check.setObjectName("IndexCheck")
        self._check.setFixedSize(15, 15)
        self._check.setAlignment(Qt.AlignCenter)
        top.addWidget(self._tag)
        top.addStretch(1)
        top.addWidget(self._check)
        self._name = QLabel(full_name)
        self._name.setObjectName("IndexName")
        desc_lbl = QLabel(desc)
        desc_lbl.setObjectName("IndexDesc")
        desc_lbl.setWordWrap(True)
        layout.addLayout(top)
        layout.addWidget(self._name)
        layout.addWidget(desc_lbl)

    def set_checked(self, on: bool) -> None:
        self.setProperty("checked", on)
        self._check.setText("✓" if on else "")
        self._check.setProperty("checked", on)
        # Re-polish la carte ET ses enfants stylés (tag/nom/coche) sinon les
        # sélecteurs descendants #IndexCard[checked] #IndexTag restent figés.
        for w in (self, self._check, self._tag, self._name):
            w.style().unpolish(w)
            w.style().polish(w)

    def mousePressEvent(self, event):  # noqa: N802 (signature Qt)
        self.clicked.emit(self._key)
        super().mousePressEvent(event)


class _Chip(QFrame):
    """Pastille MNT / Densité (QFrame + layout → dimensionnement fiable)."""

    clicked = pyqtSignal(str)

    def __init__(self, key: str, parent=None):
        super().__init__(parent)
        self._key = key
        self.setObjectName("Chip")
        self.setProperty("checked", False)
        self.setCursor(Qt.PointingHandCursor)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 5, 12, 5)
        layout.setSpacing(6)
        self._label = QLabel("")
        self._label.setObjectName("ChipLabel")
        layout.addWidget(self._label)

    def set_text(self, text: str) -> None:
        self._label.setText(text)

    def set_checked(self, on: bool) -> None:
        self.setProperty("checked", on)
        for w in (self, self._label):
            w.style().unpolish(w)
            w.style().polish(w)

    def mousePressEvent(self, event):  # noqa: N802 (signature Qt)
        self.clicked.emit(self._key)
        super().mousePressEvent(event)


class IndicesPage(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mode = "ign_laz"
        self._products = default_products()
        self._loading = False
        self._index_cards: dict = {}
        self._build()
        self._refresh()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        heading = QLabel("Choisir les indices à calculer")
        heading.setObjectName("WizardPageHeading")
        root.addWidget(heading)
        self._sub = QLabel("")
        self._sub.setObjectName("WizardPageSub")
        self._sub.setWordWrap(True)
        root.addWidget(self._sub)

        self._mode_banner = QLabel("")
        self._mode_banner.setObjectName("InfoBanner")
        self._mode_banner.setWordWrap(True)
        self._mode_banner.setVisible(False)
        root.addWidget(self._mode_banner)

        # ── ① Modèle de base ──
        self._base_card, bv = build_card("Modèle de base", "1")
        chips = QHBoxLayout()
        chips.setSpacing(8)
        self._mnt_chip = _Chip("MNT")
        self._mnt_chip.clicked.connect(self._on_product_clicked)
        self._dens_chip = _Chip("DENSITE")
        self._dens_chip.clicked.connect(self._on_product_clicked)
        chips.addWidget(self._mnt_chip)
        chips.addWidget(self._dens_chip)
        chips.addStretch(1)
        bv.addLayout(chips)
        self._mnt_hint = QLabel("MNT requis tant qu'un indice RVT est coché.")
        self._mnt_hint.setObjectName("MntHint")
        self._mnt_hint.setVisible(False)
        # Réserver la place du hint : le cocher d'un indice (qui force MNT et
        # affiche ce hint) ne doit pas faire grandir la carte « Modèle de base ».
        _hint_policy = self._mnt_hint.sizePolicy()
        _hint_policy.setRetainSizeWhenHidden(True)
        self._mnt_hint.setSizePolicy(_hint_policy)
        bv.addWidget(self._mnt_hint)
        res_row = QHBoxLayout()
        res_row.setSpacing(8)
        res_label = QLabel("Résolution MNT")
        res_label.setObjectName("FieldLabel")
        self._res_spin = QDoubleSpinBox()
        self._res_spin.setRange(0.1, 10.0)
        self._res_spin.setSingleStep(0.1)
        self._res_spin.setDecimals(2)
        self._res_spin.setValue(0.5)
        self._res_spin.setFixedWidth(80)
        self._res_spin.valueChanged.connect(self._on_changed)
        res_row.addWidget(res_label)
        res_row.addWidget(self._res_spin)
        res_row.addWidget(QLabel("m / pixel"))
        res_row.addStretch(1)
        bv.addLayout(res_row)
        root.addWidget(self._base_card)

        # ── ② Indices de visualisation ──
        self._rvt_card, rv = build_card("Indices de visualisation", "2")
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        for i, key in enumerate(rvt_keys()):
            p = product(key)
            card = _IndexCard(key, p.tag, p.full_name, p.description)
            card.clicked.connect(self._on_product_clicked)
            self._index_cards[key] = card
            grid.addWidget(card, i // 3, i % 3)
        rv.addLayout(grid)
        footer = QHBoxLayout()
        self._count_label = QLabel("")
        self._count_label.setObjectName("WizardPageSub")
        adv_btn = QPushButton("Réglages avancés…")
        adv_btn.setObjectName("GhostButton")
        adv_btn.clicked.connect(self._on_advanced)
        footer.addWidget(self._count_label)
        footer.addStretch(1)
        footer.addWidget(adv_btn)
        rv.addLayout(footer)
        root.addWidget(self._rvt_card)

        root.addStretch(1)

    # ------------------------------------------------------------------
    # Logique
    # ------------------------------------------------------------------
    def _on_product_clicked(self, key: str) -> None:
        new, toast = toggle(self._products, key)
        if toast:
            show_toast(self, toast)
        self._products = new
        self._refresh()
        if not self._loading:
            self.changed.emit()

    def _on_advanced(self) -> None:
        QMessageBox.information(
            self,
            "À venir",
            "Les réglages avancés par indice (onglets MNT / M-HS / SVF / …) "
            "seront branchés dans une prochaine itération.",
        )

    def _on_changed(self, *_args) -> None:
        if not self._loading:
            self.changed.emit()

    def _refresh(self) -> None:
        self._mnt_chip.set_checked(self._products.get("MNT", False))
        self._dens_chip.set_checked(self._products.get("DENSITE", False))
        self._dens_chip.set_text("Densité · points LiDAR / m²")
        locked = requires_mnt(self._products)
        self._mnt_chip.set_text(
            "MNT · altitude du sol" + ("   · REQUIS" if locked else "")
        )
        self._mnt_hint.setVisible(locked)
        for key, card in self._index_cards.items():
            card.set_checked(self._products.get(key, False))
        n = count_selected(self._products)
        self._count_label.setText(
            f"{n} produit{'s' if n > 1 else ''} sélectionné{'s' if n > 1 else ''}"
        )
        self._sub.setText(
            "Le MNT est la base des indices RVT. Cliquez un indice pour le sélectionner."
            if self._mode not in ("existing_mnt", "existing_rvt")
            else "Données déjà fournies en entrée — sélectionnez les indices RVT à calculer."
        )

    # ------------------------------------------------------------------
    # Mode (neutralisation des sections fournies)
    # ------------------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self._base_card.setVisible(mode not in ("existing_mnt", "existing_rvt"))
        self._rvt_card.setVisible(mode != "existing_rvt")
        if mode == "existing_mnt":
            self._mode_banner.setText(
                "ℹ  Mode MNT existant — sections MNT / Densité masquées (déjà disponibles)."
            )
            self._mode_banner.setVisible(True)
        elif mode == "existing_rvt":
            self._mode_banner.setText(
                "ℹ  Mode Indices RVT existants — cette étape est sans objet, passez à la détection."
            )
            self._mode_banner.setVisible(True)
        else:
            self._mode_banner.setVisible(False)
        self._refresh()

    # ------------------------------------------------------------------
    # Persistance
    # ------------------------------------------------------------------
    def active_rvt_keys(self) -> set:
        """Indices RVT actuellement cochés (pour l'étape 3)."""
        return {k for k in rvt_keys() if self._products.get(k)}

    def recap_products(self) -> list:
        """Tags des produits actifs, dans l'ordre du catalogue (récap étape 4)."""
        return [p.tag for p in all_products() if self._products.get(p.key)]

    def resolution(self) -> float:
        """Résolution MNT (m/pixel), pour le récap de l'étape 4."""
        return float(self._res_spin.value())

    def activate_product(self, key: str) -> None:
        """Active un produit (utilisé par « + Activer » de l'étape 3)."""
        if self._products.get(key):
            return
        self._products[key] = True
        if key in rvt_keys():
            self._products["MNT"] = True
        self._refresh()
        if not self._loading:
            self.changed.emit()

    def summary(self) -> str:
        n_rvt = sum(1 for k in rvt_keys() if self._products.get(k))
        base = "MNT" if self._products.get("MNT") else "—"
        if n_rvt:
            return f"{base} + {n_rvt} indice{'s' if n_rvt > 1 else ''} RVT"
        return base

    def load_from(self, config: dict) -> None:
        proc = config.get("processing") or {}
        prods = proc.get("products") or {}
        prev = self._loading
        self._loading = True
        try:
            self._products = {p.key: bool(prods.get(p.key, False)) for p in all_products()}
            try:
                self._res_spin.setValue(float(proc.get("mnt_resolution", 0.5)))
            except (TypeError, ValueError):
                self._res_spin.setValue(0.5)
            self._refresh()
        finally:
            self._loading = prev

    def collect_into(self, config: dict) -> None:
        proc = config.setdefault("processing", {})
        products = proc.setdefault("products", {})
        for p in all_products():
            products[p.key] = bool(self._products.get(p.key, False))
        proc["mnt_resolution"] = float(self._res_spin.value())
