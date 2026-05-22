"""Étape 2 — Indices de visualisation.

Sélection des produits : MNT/Densité (modèle de base) + 6 indices RVT. Le MNT
est verrouillé tant qu'un indice RVT est coché (clic → toast explicatif). Selon
le mode de données, les sections déjà fournies sont masquées (existing_mnt /
existing_rvt). La logique pure vient de :mod:`app.services.indices_model`.

Le bouton « Réglages avancés… » bascule (via un ``QStackedWidget`` interne) sur
une vue plein écran à onglets reproduisant tous les paramètres RVT de l'ancien
plugin (M-HS, SVF, Slope, LD, SLRM, VAT) + filtre PDAL / résolution densité /
marge de tuilage. Ces réglages sont persistés dans ``rvt_params`` et
``processing`` du config (consommés tels quels par le pipeline).
"""
from __future__ import annotations

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QTabWidget,
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
from ..widgets.no_wheel import NoWheelDoubleSpinBox, NoWheelSpinBox
from ..widgets.toast import show_toast

# Filtre PDAL par défaut (identique à ConfigManager.default_config).
DEFAULT_FILTER = (
    "Classification = 2 OR Classification = 6 OR Classification = 66 "
    "OR Classification = 67 OR Classification = 9"
)


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
        self._adv_fields: list = []  # descripteurs (section, key, widget, kind, default)
        self._build()
        self._refresh()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self._stack = QStackedWidget()
        self._overview_page = self._build_overview()
        self._advanced_page = self._build_advanced()
        self._stack.addWidget(self._overview_page)   # [0]
        self._stack.addWidget(self._advanced_page)    # [1]
        root.addWidget(self._stack)

    def _build_overview(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
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
        return page

    # ------------------------------------------------------------------
    # Vue avancée (onglets RVT — paramètres identiques à l'ancien plugin)
    # ------------------------------------------------------------------
    def _build_advanced(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        header = QHBoxLayout()
        back = QPushButton("←  Vue d'ensemble")
        back.setObjectName("GhostButton")
        back.clicked.connect(self._show_overview)
        title = QLabel("Réglages avancés des indices")
        title.setObjectName("WizardPageHeading")
        reset = QPushButton("Réinitialiser")
        reset.setObjectName("GhostButton")
        reset.clicked.connect(self._reset_advanced)
        header.addWidget(back)
        header.addSpacing(10)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(reset)
        root.addLayout(header)

        self._adv_tabs = QTabWidget()
        self._adv_tabs.setObjectName("AdvTabs")

        # — Onglet MNT : filtre PDAL + résolution densité —
        # (mnt_resolution reste sur la vue d'ensemble pour ne pas dupliquer.)
        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("Ex: Classification = 2 OR Classification = 6")
        self._filter_edit.textChanged.connect(self._on_changed)
        self._density_spin = self._mk_dspin(0.01, 100.0, 1.0)
        self._reg(("processing",), "filter_expression", self._filter_edit, "text", DEFAULT_FILTER)
        self._reg(("processing",), "density_resolution", self._density_spin, "float", 1.0)
        self._adv_tabs.addTab(self._make_tab([
            ("Filtre PDAL :", self._filter_edit),
            ("Résolution densité (m) :", self._density_spin),
        ]), "MNT")

        # — M-HS (mdh) —
        mdh_dirs = self._mk_spin(1, 360, 16)
        mdh_sun = self._mk_spin(0, 90, 35)
        mdh_ve = self._mk_spin(1, 100, 1)
        mdh_8 = self._mk_check()
        self._reg(("rvt_params", "mdh"), "num_directions", mdh_dirs, "int", 16)
        self._reg(("rvt_params", "mdh"), "sun_elevation", mdh_sun, "int", 35)
        self._reg(("rvt_params", "mdh"), "ve_factor", mdh_ve, "int", 1)
        self._reg(("rvt_params", "mdh"), "save_as_8bit", mdh_8, "bool", True)
        self._adv_tabs.addTab(self._make_tab([
            ("Nombre directions :", mdh_dirs),
            ("Élévation solaire (°) :", mdh_sun),
            ("Facteur VE :", mdh_ve),
            ("", mdh_8),
        ]), "M-HS")

        # — SVF (svf) —
        svf_noise = self._mk_spin(0, 9999, 0)
        svf_dirs = self._mk_spin(1, 360, 16)
        svf_radius = self._mk_spin(0, 100000, 10)
        svf_ve = self._mk_spin(1, 100, 1)
        svf_8 = self._mk_check()
        self._reg(("rvt_params", "svf"), "noise_remove", svf_noise, "int", 0)
        self._reg(("rvt_params", "svf"), "num_directions", svf_dirs, "int", 16)
        self._reg(("rvt_params", "svf"), "radius", svf_radius, "int", 10)
        self._reg(("rvt_params", "svf"), "ve_factor", svf_ve, "int", 1)
        self._reg(("rvt_params", "svf"), "save_as_8bit", svf_8, "bool", True)
        self._adv_tabs.addTab(self._make_tab([
            ("Suppression bruit :", svf_noise),
            ("Nombre directions :", svf_dirs),
            ("Rayon (px) :", svf_radius),
            ("Facteur VE :", svf_ve),
            ("", svf_8),
        ]), "SVF")

        # — Slope (slope) —
        slope_unit = QComboBox()
        slope_unit.addItem("Degrés", 0)
        slope_unit.addItem("Pourcentage", 1)
        slope_unit.currentIndexChanged.connect(self._on_changed)
        slope_ve = self._mk_spin(1, 100, 1)
        slope_8 = self._mk_check()
        self._reg(("rvt_params", "slope"), "unit", slope_unit, "combo", 0)
        self._reg(("rvt_params", "slope"), "ve_factor", slope_ve, "int", 1)
        self._reg(("rvt_params", "slope"), "save_as_8bit", slope_8, "bool", True)
        self._adv_tabs.addTab(self._make_tab([
            ("Unité :", slope_unit),
            ("Facteur VE :", slope_ve),
            ("", slope_8),
        ]), "Slope")

        # — LD (ldo) —
        ld_ang = self._mk_spin(1, 360, 15)
        ld_rmin = self._mk_spin(0, 100000, 10)
        ld_rmax = self._mk_spin(0, 100000, 20)
        ld_obs = self._mk_dspin(0.0, 10000.0, 1.7)
        ld_ve = self._mk_spin(1, 100, 1)
        ld_8 = self._mk_check()
        self._reg(("rvt_params", "ldo"), "angular_res", ld_ang, "int", 15)
        self._reg(("rvt_params", "ldo"), "min_radius", ld_rmin, "int", 10)
        self._reg(("rvt_params", "ldo"), "max_radius", ld_rmax, "int", 20)
        self._reg(("rvt_params", "ldo"), "observer_h", ld_obs, "float", 1.7)
        self._reg(("rvt_params", "ldo"), "ve_factor", ld_ve, "int", 1)
        self._reg(("rvt_params", "ldo"), "save_as_8bit", ld_8, "bool", True)
        self._adv_tabs.addTab(self._make_tab([
            ("Résolution angulaire (°) :", ld_ang),
            ("Rayon min (px) :", ld_rmin),
            ("Rayon max (px) :", ld_rmax),
            ("Hauteur observateur (m) :", ld_obs),
            ("Facteur VE :", ld_ve),
            ("", ld_8),
        ]), "LD")

        # — SLRM (slrm) —
        slrm_radius = self._mk_spin(1, 100000, 20)
        slrm_ve = self._mk_spin(1, 100, 1)
        slrm_8 = self._mk_check()
        self._reg(("rvt_params", "slrm"), "radius", slrm_radius, "int", 20)
        self._reg(("rvt_params", "slrm"), "ve_factor", slrm_ve, "int", 1)
        self._reg(("rvt_params", "slrm"), "save_as_8bit", slrm_8, "bool", True)
        self._adv_tabs.addTab(self._make_tab([
            ("Rayon (px) :", slrm_radius),
            ("Facteur VE :", slrm_ve),
            ("", slrm_8),
        ]), "SLRM")

        # — VAT (vat) —
        vat_terrain = QComboBox()
        vat_terrain.addItem("Général", 0)
        vat_terrain.addItem("Plat", 1)
        vat_terrain.addItem("Pentu", 2)
        vat_terrain.currentIndexChanged.connect(self._on_changed)
        vat_8 = self._mk_check()
        self._reg(("rvt_params", "vat"), "terrain_type", vat_terrain, "combo", 0)
        self._reg(("rvt_params", "vat"), "save_as_8bit", vat_8, "bool", True)
        self._adv_tabs.addTab(self._make_tab([
            ("Type de terrain :", vat_terrain),
            ("", vat_8),
        ]), "VAT")

        root.addWidget(self._adv_tabs)

        # — Tuilage (global à tous les indices) —
        ov_card, ovv = build_card("Tuilage des indices")
        self._overlap_spin = self._mk_spin(0, 100, 20)
        self._reg(("processing",), "tile_overlap", self._overlap_spin, "int", 20)
        ov_row = QHBoxLayout()
        ov_row.setSpacing(8)
        ov_lbl = QLabel("Marge tuiles")
        ov_lbl.setObjectName("FieldLabel")
        ov_row.addWidget(ov_lbl)
        ov_row.addWidget(self._overlap_spin)
        ov_row.addWidget(QLabel("%"))
        ov_row.addStretch(1)
        ov_hint = QLabel("Chevauchement entre tuiles — évite les artefacts en bordure.")
        ov_hint.setObjectName("MntHint")
        ovv.addLayout(ov_row)
        ovv.addWidget(ov_hint)
        root.addWidget(ov_card)

        root.addStretch(1)
        return page

    # — fabriques de widgets (numériques wheel-safe) —
    def _mk_spin(self, lo: int, hi: int, val: int) -> NoWheelSpinBox:
        s = NoWheelSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.valueChanged.connect(self._on_changed)
        return s

    def _mk_dspin(self, lo: float, hi: float, val: float) -> NoWheelDoubleSpinBox:
        s = NoWheelDoubleSpinBox()
        s.setDecimals(2)
        s.setRange(lo, hi)
        s.setSingleStep(0.1)
        s.setValue(val)
        s.valueChanged.connect(self._on_changed)
        return s

    def _mk_check(self) -> QCheckBox:
        c = QCheckBox("Sauver en 8bit")
        c.setChecked(True)
        c.toggled.connect(self._on_changed)
        return c

    def _reg(self, section: tuple, key: str, widget, kind: str, default) -> None:
        self._adv_fields.append((section, key, widget, kind, default))

    @staticmethod
    def _make_tab(rows) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(8)
        for label, widget in rows:
            form.addRow(label, widget)
        return tab

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
        self._stack.setCurrentWidget(self._advanced_page)

    def _show_overview(self) -> None:
        self._stack.setCurrentWidget(self._overview_page)

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
        # L'onglet MNT (filtre PDAL + densité) n'a de sens que pour les modes
        # qui calculent un MNT depuis un nuage de points.
        self._adv_tabs.setTabEnabled(0, mode in ("ign_laz", "local_laz"))
        # En existing_rvt l'étape est sans objet : revenir à la vue d'ensemble.
        if mode == "existing_rvt":
            self._show_overview()
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
            self._load_advanced(config)
            self._refresh()
        finally:
            self._loading = prev

    def collect_into(self, config: dict) -> None:
        proc = config.setdefault("processing", {})
        products = proc.setdefault("products", {})
        for p in all_products():
            products[p.key] = bool(self._products.get(p.key, False))
        proc["mnt_resolution"] = float(self._res_spin.value())
        self._collect_advanced(config)

    # ------------------------------------------------------------------
    # Réglages avancés — chargement / collecte / reset (data-driven)
    # ------------------------------------------------------------------
    def _load_advanced(self, config: dict) -> None:
        for section, key, widget, kind, default in self._adv_fields:
            container = config
            for part in section:
                container = container.get(part) if isinstance(container, dict) else None
                if container is None:
                    break
            raw = container.get(key, default) if isinstance(container, dict) else default
            self._apply_field(widget, kind, raw, default)

    def _collect_advanced(self, config: dict) -> None:
        for section, key, widget, kind, _default in self._adv_fields:
            container = config
            for part in section:
                container = container.setdefault(part, {})
            container[key] = self._read_field(widget, kind)

    def _reset_advanced(self) -> None:
        prev = self._loading
        self._loading = True
        try:
            for _section, _key, widget, kind, default in self._adv_fields:
                self._apply_field(widget, kind, default, default)
        finally:
            self._loading = prev
        if not self._loading:
            self.changed.emit()

    @staticmethod
    def _apply_field(widget, kind: str, raw, default) -> None:
        if kind == "int":
            try:
                widget.setValue(int(raw))
            except (TypeError, ValueError):
                widget.setValue(int(default))
        elif kind == "float":
            try:
                widget.setValue(float(raw))
            except (TypeError, ValueError):
                widget.setValue(float(default))
        elif kind == "bool":
            widget.setChecked(bool(raw))
        elif kind == "combo":
            idx = widget.findData(raw)
            if idx < 0:
                idx = widget.findData(default)
            widget.setCurrentIndex(max(0, idx))
        elif kind == "text":
            widget.setText("" if raw is None else str(raw))

    @staticmethod
    def _read_field(widget, kind: str):
        if kind == "int":
            return int(widget.value())
        if kind == "float":
            return float(widget.value())
        if kind == "bool":
            return bool(widget.isChecked())
        if kind == "combo":
            return widget.currentData()
        return widget.text()  # kind == "text"
