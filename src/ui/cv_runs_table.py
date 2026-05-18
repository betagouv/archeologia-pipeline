"""Helpers de construction des lignes du tableau de runs CV.

Avant ce module, ``_add_det_run_row`` faisait 75 lignes dans
``MainDialog`` et mélangeait :

- la **construction** des cell-widgets Qt (combo modèles, combo RVT,
  spinbox aire, boutons info/×) ;
- le **branchement** des signaux Qt vers les callbacks du dialog ;
- des **dépendances de données** (liste des modèles dispos, liste des
  RVT actifs, valeurs préférées par modèle).

V4.3 isole la construction des cell-widgets dans des fonctions pures.
Le dialog reste responsable de :

- posséder le ``QTableWidget`` ;
- fournir les listes (``models``, ``rvt_keys``) ;
- réagir aux signaux (autosave, ``_refresh_model_classes``,
  ``_apply_preferred_*_for_row``).

Cette séparation rend la logique navigable sans extraire un wrapper
complet — l'API du tableau (insertion, suppression, itération) reste
celle de ``QTableWidget``, ce qui évite une couche d'abstraction
fragile pour des callbacks Qt difficiles à tester.

Note : cette extraction est explicitement *partielle*. Le doc
d'architecture a marqué V4.3 comme optionnel et coûteux ; on a fait
le minimum vraiment utile (locality des sub-widgets, pas de
sur-abstraction).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple


@dataclass(frozen=True)
class RunRowCallbacks:
    """Callbacks que le dialog injecte pour réagir aux changements de
    ligne. Chacun reçoit l'événement Qt, à charge du dialog de
    retrouver la ligne via ``self.sender()`` ou ``_find_row_for_sender()``.

    Tous typés ``Callable[[], None]`` pour rester agnostiques aux
    payloads Qt — le dialog ne s'intéresse qu'au "quelque chose a
    changé sur cette ligne".
    """

    on_any_changed: Callable[[], None]
    on_model_changed: Callable[[], None]
    on_refresh_classes: Callable[[], None]
    on_info_clicked: Callable[[], None]
    on_delete_clicked: Callable[[], None]


@dataclass
class RunRowData:
    """Données que le dialog passe pour configurer une nouvelle ligne."""

    available_models: List[Tuple[str, str]] = field(default_factory=list)
    rvt_keys: List[Tuple[str, str]] = field(default_factory=list)
    initial_model_name: str = ""
    initial_target_rvt: Optional[str] = None
    initial_min_area_m2: float = 0.0


def build_model_combo(
    combo_factory: Callable[[], Any],
    data: RunRowData,
    callbacks: RunRowCallbacks,
) -> Any:
    """Construit le combobox de modèles pour une ligne.

    ``combo_factory`` est passé pour que le dialog puisse fournir un
    ``NoWheelComboBox`` (héritage Qt local) sans que ce module ne dépende
    de ``qgis.PyQt``.
    """
    model_combo = combo_factory()
    for label, path in data.available_models:
        model_combo.addItem(label, path)
    if data.initial_model_name:
        for i in range(model_combo.count()):
            d = str(model_combo.itemData(i) or "")
            text = model_combo.itemText(i)
            if data.initial_model_name in (d, text) or text == data.initial_model_name:
                model_combo.setCurrentIndex(i)
                break
    model_combo.currentIndexChanged.connect(callbacks.on_any_changed)
    model_combo.currentIndexChanged.connect(callbacks.on_refresh_classes)
    model_combo.currentIndexChanged.connect(callbacks.on_model_changed)
    return model_combo


def build_rvt_combo(
    combo_factory: Callable[[], Any],
    data: RunRowData,
    callbacks: RunRowCallbacks,
) -> Any:
    """Construit le combobox du RVT cible pour une ligne."""
    rvt_combo = combo_factory()
    for key, label in data.rvt_keys:
        rvt_combo.addItem(label, key)
    if data.initial_target_rvt:
        idx = rvt_combo.findData(data.initial_target_rvt)
        if idx >= 0:
            rvt_combo.setCurrentIndex(idx)
    rvt_combo.currentIndexChanged.connect(callbacks.on_any_changed)
    return rvt_combo


def build_min_area_spin(
    spin_factory: Callable[[], Any],
    data: RunRowData,
    callbacks: RunRowCallbacks,
) -> Any:
    """Construit le spinbox de l'aire minimale pour une ligne."""
    area_spin = spin_factory()
    area_spin.setDecimals(0)
    area_spin.setRange(0.0, 100000.0)
    area_spin.setSingleStep(50.0)
    area_spin.setValue(data.initial_min_area_m2)
    area_spin.setSuffix(" m²")
    area_spin.setToolTip(
        "Aire minimale en m² (0 = pas de filtrage). "
        "Les détections plus petites seront supprimées."
    )
    area_spin.setMinimumWidth(90)
    area_spin.valueChanged.connect(callbacks.on_any_changed)
    return area_spin


def build_actions_widget(
    container_factory: Callable[[], Any],
    layout_factory: Callable[[Any], Any],
    button_factory: Callable[[str], Any],
    callbacks: RunRowCallbacks,
) -> Any:
    """Construit le widget des boutons d'action (info ℹ + supprimer ×).

    Tous les factories Qt sont passés en paramètres pour que ce module
    n'importe pas ``qgis.PyQt`` (compatibilité tests / portabilité).
    """
    actions_widget = container_factory()
    layout = layout_factory(actions_widget)
    layout.setContentsMargins(8, 0, 2, 0)
    layout.setSpacing(2)

    info_btn = button_factory("ℹ")
    info_btn.setFixedSize(24, 24)
    info_btn.setToolTip("Paramètres d'entraînement")
    info_btn.clicked.connect(callbacks.on_info_clicked)

    del_btn = button_factory("×")
    del_btn.setFixedSize(24, 24)
    del_btn.setToolTip("Supprimer ce modèle")
    del_btn.clicked.connect(callbacks.on_delete_clicked)

    layout.addWidget(info_btn)
    layout.addWidget(del_btn)
    return actions_widget
