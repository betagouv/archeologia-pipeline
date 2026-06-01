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

from qgis.PyQt.QtCore import QRect, Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor, QPainter
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QTabBar,
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

# Descriptions longues affichées en tête de chaque onglet de paramètres détaillés.
_TAB_DESC = {
    "MNT": "Reconstruit le sol à partir du nuage de points LiDAR classé.",
    "HS": "Ombrage simple depuis une seule direction de lumière.",
    "M_HS": "Combine plusieurs angles d'éclairage simulés pour révéler le micro-relief.",
    "SVF": "Part de ciel visible en chaque point — révèle creux, fossés et dépressions.",
    "SLO": "Pente du terrain — met en évidence ruptures de pente et talus.",
    "LD": "Dominance locale — fait ressortir les structures en relief positif.",
    "SLRM": "Soustrait le relief général pour isoler les micro-reliefs.",
    "VAT": "Combinaison d'indices optimisée pour la prospection archéologique.",
}

# Aides communes à plusieurs indices.
_HELP_VE = "Exagération verticale (1 = aucune)."
_HELP_8BIT = "Fichier plus léger, suffisant pour l'affichage."


class _AdvTabBar(QTabBar):
    """Barre d'onglets des paramètres détaillés : grise les onglets d'indices
    désactivés et y dessine un badge « OFF » encadré.

    L'état désactivé est porté par ``tabData(i)`` (``True`` = indice désactivé).
    Les onglets sans données (``None``, ex. MNT) sont ignorés.
    """

    _PILL_W = 26
    _RESERVE = 44  # largeur réservée à droite pour le badge (anti-chevauchement)

    def tabSizeHint(self, index):  # noqa: N802 (signature Qt)
        size = super().tabSizeHint(index)
        if self.tabData(index):  # onglet désactivé → place pour le badge
            size.setWidth(size.width() + self._RESERVE)
        return size

    def paintEvent(self, event):  # noqa: N802 (signature Qt)
        super().paintEvent(event)  # onglets stylés par la QSS
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        font = painter.font()
        font.setPointSize(max(6, font.pointSize() - 2))
        font.setBold(True)
        painter.setFont(font)
        for i in range(self.count()):
            if not self.tabData(i):  # actif ou onglet non-indice → rien
                continue
            rect = self.tabRect(i)
            # Voile gris : l'onglet paraît désactivé.
            painter.fillRect(rect.adjusted(2, 2, -2, -2), QColor(236, 236, 236, 150))
            # Badge « OFF » encadré, calé à droite.
            pill = QRect(0, 0, self._PILL_W, 15)
            pill.moveCenter(rect.center())
            pill.moveRight(rect.right() - 8)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0xCF, 0xCF, 0xCF))
            painter.drawRoundedRect(pill, 3, 3)
            painter.setPen(QColor(0x5A, 0x5A, 0x5A))
            painter.drawText(pill, Qt.AlignCenter, "OFF")


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
        self._activate_checks: dict = {}  # clé RVT → QCheckBox « Indice activé »
        self._tab_index: dict = {}        # clé RVT → index d'onglet (badge OFF)
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
        self._adv_btn = QPushButton("Réglages avancés…")
        self._adv_btn.setObjectName("GhostButton")
        self._adv_btn.clicked.connect(self._on_advanced)
        footer.addWidget(self._count_label)
        footer.addStretch(1)
        footer.addWidget(self._adv_btn)
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
        self._adv_back_btn = QPushButton("←  Vue d'ensemble")
        self._adv_back_btn.setObjectName("GhostButton")
        self._adv_back_btn.clicked.connect(self._show_overview)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        title = QLabel("Paramètres détaillés")
        title.setObjectName("WizardPageHeading")
        subtitle = QLabel(
            "Tous les paramètres techniques par produit · valeurs RVT-py officielles"
        )
        subtitle.setObjectName("WizardPageSub")
        subtitle.setWordWrap(True)
        titles.addWidget(title)
        titles.addWidget(subtitle)
        self._reset_btn = QPushButton("↺  Réinitialiser")
        self._reset_btn.setObjectName("GhostButton")
        self._reset_btn.clicked.connect(self._reset_advanced)
        header.addWidget(self._adv_back_btn)
        header.addSpacing(10)
        header.addLayout(titles)
        header.addStretch(1)
        header.addWidget(self._reset_btn)
        root.addLayout(header)

        self._adv_tabs = QTabWidget()
        self._adv_tabs.setObjectName("AdvTabs")
        self._adv_tabs.setTabBar(_AdvTabBar())  # voile + badge « OFF » des indices off

        # — Onglet MNT : filtre PDAL + résolution densité —
        # (mnt_resolution reste sur la vue d'ensemble pour ne pas dupliquer.)
        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("Ex: Classification = 2 OR Classification = 6")
        self._filter_edit.textChanged.connect(self._on_changed)
        self._density_spin = self._mk_dspin(0.01, 100.0, 1.0)
        self._reg(("processing",), "filter_expression", self._filter_edit, "text", DEFAULT_FILTER)
        self._reg(("processing",), "density_resolution", self._density_spin, "float", 1.0)
        self._adv_tabs.addTab(self._make_param_tab("MNT", [
            ("Filtre PDAL", self._filter_edit,
             "Classes LiDAR conservées pour reconstruire le sol (codes ASPRS)."),
            ("Résolution densité (m)", self._density_spin,
             "Taille de cellule du raster de densité de points."),
        ]), "MNT")

        # — HS (hs) —
        hs_az = self._mk_spin(0, 360, 315)
        hs_sun = self._mk_spin(0, 90, 35)
        hs_ve = self._mk_spin(1, 100, 1)
        hs_8 = self._mk_check()
        self._reg(("rvt_params", "hs"), "sun_azimuth", hs_az, "int", 315)
        self._reg(("rvt_params", "hs"), "sun_elevation", hs_sun, "int", 35)
        self._reg(("rvt_params", "hs"), "ve_factor", hs_ve, "int", 1)
        self._reg(("rvt_params", "hs"), "save_as_8bit", hs_8, "bool", True)
        self._tab_index["HS"] = self._adv_tabs.addTab(self._make_param_tab("HS", [
            ("Azimut solaire (°)", hs_az,
             "Direction de la lumière (0 = N, 90 = E, 180 = S, 270 = O)."),
            ("Élévation solaire (°)", hs_sun,
             "Hauteur du soleil au-dessus de l'horizon."),
            ("Facteur VE", hs_ve, _HELP_VE),
            ("", hs_8, _HELP_8BIT),
        ]), "HS")

        # — M-HS (mdh) —
        mdh_dirs = self._mk_spin(1, 360, 16)
        mdh_sun = self._mk_spin(0, 90, 35)
        mdh_ve = self._mk_spin(1, 100, 1)
        mdh_8 = self._mk_check()
        self._reg(("rvt_params", "mdh"), "num_directions", mdh_dirs, "int", 16)
        self._reg(("rvt_params", "mdh"), "sun_elevation", mdh_sun, "int", 35)
        self._reg(("rvt_params", "mdh"), "ve_factor", mdh_ve, "int", 1)
        self._reg(("rvt_params", "mdh"), "save_as_8bit", mdh_8, "bool", True)
        self._tab_index["M_HS"] = self._adv_tabs.addTab(self._make_param_tab("M_HS", [
            ("Nombre de directions", mdh_dirs,
             "Angles d'éclairage simulés. 16 = bon compromis qualité/temps."),
            ("Élévation solaire (°)", mdh_sun,
             "Hauteur du soleil au-dessus de l'horizon."),
            ("Facteur VE", mdh_ve, _HELP_VE),
            ("", mdh_8, _HELP_8BIT),
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
        self._tab_index["SVF"] = self._adv_tabs.addTab(self._make_param_tab("SVF", [
            ("Suppression du bruit", svf_noise,
             "Niveau de lissage du bruit (0 = aucun)."),
            ("Nombre de directions", svf_dirs,
             "Directions d'horizon échantillonnées autour de chaque pixel."),
            ("Rayon (px)", svf_radius,
             "Distance de recherche de l'horizon, en pixels."),
            ("Facteur VE", svf_ve, _HELP_VE),
            ("", svf_8, _HELP_8BIT),
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
        self._tab_index["SLO"] = self._adv_tabs.addTab(self._make_param_tab("SLO", [
            ("Unité", slope_unit, "Pente exprimée en degrés ou en pourcentage."),
            ("Facteur VE", slope_ve, _HELP_VE),
            ("", slope_8, _HELP_8BIT),
        ]), "SLO")

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
        self._tab_index["LD"] = self._adv_tabs.addTab(self._make_param_tab("LD", [
            ("Résolution angulaire (°)", ld_ang,
             "Pas angulaire du balayage. Plus petit = plus précis mais plus lent."),
            ("Rayon min (px)", ld_rmin,
             "Distance minimale prise en compte autour du pixel."),
            ("Rayon max (px)", ld_rmax,
             "Distance maximale prise en compte autour du pixel."),
            ("Hauteur observateur (m)", ld_obs,
             "Hauteur de l'œil virtuel au-dessus du sol."),
            ("Facteur VE", ld_ve, _HELP_VE),
            ("", ld_8, _HELP_8BIT),
        ]), "LD")

        # — SLRM (slrm) —
        slrm_radius = self._mk_spin(1, 100000, 20)
        slrm_ve = self._mk_spin(1, 100, 1)
        slrm_8 = self._mk_check()
        self._reg(("rvt_params", "slrm"), "radius", slrm_radius, "int", 20)
        self._reg(("rvt_params", "slrm"), "ve_factor", slrm_ve, "int", 1)
        self._reg(("rvt_params", "slrm"), "save_as_8bit", slrm_8, "bool", True)
        self._tab_index["SLRM"] = self._adv_tabs.addTab(self._make_param_tab("SLRM", [
            ("Rayon (px)", slrm_radius,
             "Rayon du lissage : sépare le micro-relief du relief général."),
            ("Facteur VE", slrm_ve, _HELP_VE),
            ("", slrm_8, _HELP_8BIT),
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
        self._tab_index["VAT"] = self._adv_tabs.addTab(self._make_param_tab("VAT", [
            ("Type de terrain", vat_terrain,
             "Préréglage adapté au relief dominant de la zone."),
            ("", vat_8, _HELP_8BIT),
        ]), "VAT")

        root.addWidget(self._adv_tabs)

        # — Tuilage (global à tous les indices) —
        ov_card, ovv = build_card("Tuilage & overlap")
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
        ov_hint = QLabel(
            "Chevauchement entre tuiles lors du calcul RVT. Évite les artefacts "
            "aux bordures. S'applique à tous les indices."
        )
        ov_hint.setObjectName("MntHint")
        ov_hint.setWordWrap(True)
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
        c = QCheckBox("Sauver en 8 bits")
        c.setChecked(True)
        c.toggled.connect(self._on_changed)
        return c

    def _reg(self, section: tuple, key: str, widget, kind: str, default) -> None:
        self._adv_fields.append((section, key, widget, kind, default))

    def _make_param_tab(self, key: str, rows) -> QWidget:
        """Onglet de paramètres : en-tête (nom + description [+ « Indice activé »
        pour les indices RVT]) puis grille 2 colonnes de champs avec aide.

        ``rows`` = liste de ``(label, widget, help_text)``. Un ``QLineEdit``
        occupe les deux colonnes (champ texte large, ex. filtre PDAL).
        """
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(12)

        # — En-tête : nom complet + description + (case « Indice activé ») —
        head = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        title = QLabel(product(key).full_name)
        title.setObjectName("AdvIndexTitle")
        desc = QLabel(_TAB_DESC.get(key, product(key).description))
        desc.setObjectName("AdvIndexDesc")
        desc.setWordWrap(True)
        titles.addWidget(title)
        titles.addWidget(desc)
        head.addLayout(titles, 1)
        if key in rvt_keys():
            chk = QCheckBox("Indice activé")
            chk.setObjectName("ActivateCheck")
            chk.toggled.connect(lambda on, k=key: self._on_activate_toggled(k, on))
            self._activate_checks[key] = chk
            head.addWidget(chk, 0, Qt.AlignTop)
        outer.addLayout(head)

        # — Grille 2 colonnes de champs —
        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(12)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        r = c = 0
        for label, widget, help_text in rows:
            cell = self._param_cell(label, widget, help_text)
            if isinstance(widget, QLineEdit):
                if c != 0:
                    r += 1
                    c = 0
                grid.addWidget(cell, r, 0, 1, 2)
                r += 1
            else:
                grid.addWidget(cell, r, c)
                c += 1
                if c == 2:
                    c = 0
                    r += 1
        outer.addLayout(grid)
        outer.addStretch(1)
        return tab

    @staticmethod
    def _param_cell(label: str, widget, help_text: str) -> QWidget:
        """Cellule de paramètre : ligne [label · valeur] (ou case à cocher seule)
        surmontant un sous-texte d'aide."""
        cell = QVBoxLayout()
        cell.setContentsMargins(0, 0, 0, 0)
        cell.setSpacing(3)
        if isinstance(widget, QCheckBox):
            cell.addWidget(widget)
        elif isinstance(widget, QLineEdit):
            if label:
                lbl = QLabel(label)
                lbl.setObjectName("FieldLabel")
                cell.addWidget(lbl)
            cell.addWidget(widget)
        else:
            row = QHBoxLayout()
            row.setSpacing(8)
            lbl = QLabel(label)
            lbl.setObjectName("FieldLabel")
            if isinstance(widget, QComboBox):
                widget.setMinimumWidth(130)
            else:
                widget.setFixedWidth(96)
            row.addWidget(lbl)
            row.addStretch(1)
            row.addWidget(widget)
            cell.addLayout(row)
        if help_text:
            hint = QLabel(help_text)
            hint.setObjectName("FieldHint")
            hint.setWordWrap(True)
            cell.addWidget(hint)
        wrap = QWidget()
        wrap.setLayout(cell)
        return wrap

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

    def _on_activate_toggled(self, key: str, checked: bool) -> None:
        """Case « Indice activé » d'un onglet détaillé : même logique que les
        cartes de la vue d'ensemble (l'activation d'un indice RVT force MNT)."""
        if self._loading:
            return
        if checked == bool(self._products.get(key, False)):
            return  # déjà à l'état voulu (typiquement une resynchronisation)
        new, toast = toggle(self._products, key)
        if toast:
            show_toast(self, toast)
        self._products = new
        self._refresh()
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
        # Vue détaillée : cases « Indice activé » + badges « OFF » sur les onglets.
        for key, chk in self._activate_checks.items():
            on = self._products.get(key, False)
            chk.blockSignals(True)
            chk.setChecked(on)
            chk.blockSignals(False)
        tab_bar = self._adv_tabs.tabBar()
        for key, idx in self._tab_index.items():
            tab_bar.setTabData(idx, not self._products.get(key))  # True = désactivé
            tab_bar.setTabText(idx, product(key).tag)  # force le recalcul de largeur
        tab_bar.update()
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

    def set_readonly(self, ro: bool) -> None:
        """Verrouille la saisie pour consultation pendant un run (lecture seule).

        Désactive uniquement les widgets de *saisie* ; les widgets de
        navigation/dépliage (« Réglages avancés… », retour, barre d'onglets RVT,
        scroll) restent actifs pour permettre de tout consulter. N'agit que sur
        ``setEnabled``/``setReadOnly`` → aucun signal ``changed`` ni autosave.
        """
        self._mnt_chip.setEnabled(not ro)
        self._dens_chip.setEnabled(not ro)
        self._res_spin.setEnabled(not ro)
        self._reset_btn.setEnabled(not ro)
        for card in self._index_cards.values():
            card.setEnabled(not ro)
        for chk in self._activate_checks.values():
            chk.setEnabled(not ro)
        for _section, _key, widget, kind, _default in self._adv_fields:
            widget.setEnabled(not ro)
            if kind == "text":
                widget.setReadOnly(ro)

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
