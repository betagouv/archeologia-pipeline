"""Tests des helpers de construction des lignes du tableau de runs CV.

Les helpers utilisent ``Callable[[], Any]`` pour fabriquer les widgets
Qt — on injecte des mocks. Cela évite de dépendre de Qt en test.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ui.cv_runs_table import (
    RunRowCallbacks,
    RunRowData,
    build_actions_widget,
    build_min_area_spin,
    build_model_combo,
    build_rvt_combo,
)


# ----------------------------------------------------------------------
# Mocks
# ----------------------------------------------------------------------
class _ComboMock:
    def __init__(self):
        self._items = []
        self._idx = -1
        self.currentIndexChanged = MagicMock()
        # MagicMock pour ``connect`` permet de vérifier qu'on l'a branché.
        self.currentIndexChanged.connect = MagicMock()

    def addItem(self, label, data=None):
        self._items.append((label, data))

    def itemData(self, i):
        return self._items[i][1] if 0 <= i < len(self._items) else None

    def itemText(self, i):
        return self._items[i][0] if 0 <= i < len(self._items) else ""

    def setCurrentIndex(self, i):
        self._idx = i

    def currentIndex(self):
        return self._idx

    def count(self):
        return len(self._items)

    def findData(self, data):
        for i, (_label, d) in enumerate(self._items):
            if d == data:
                return i
        return -1


class _SpinMock:
    def __init__(self):
        self._v = 0.0
        self.valueChanged = MagicMock()
        self.valueChanged.connect = MagicMock()

    def setDecimals(self, n): self._dec = n
    def setRange(self, a, b): self._range = (a, b)
    def setSingleStep(self, s): self._step = s
    def setValue(self, v): self._v = v
    def value(self): return self._v
    def setSuffix(self, s): self._suffix = s
    def setToolTip(self, t): self._tt = t
    def setMinimumWidth(self, w): self._mw = w


class _ButtonMock:
    def __init__(self, _text):
        self._text = _text
        self.clicked = MagicMock()
        self.clicked.connect = MagicMock()

    def setFixedSize(self, w, h): self._size = (w, h)
    def setToolTip(self, t): self._tt = t


class _LayoutMock:
    def __init__(self, _container):
        self._widgets = []

    def setContentsMargins(self, *a): self._margins = a
    def setSpacing(self, s): self._spacing = s
    def addWidget(self, w): self._widgets.append(w)


def _make_callbacks() -> RunRowCallbacks:
    return RunRowCallbacks(
        on_any_changed=MagicMock(),
        on_model_changed=MagicMock(),
        on_refresh_classes=MagicMock(),
        on_info_clicked=MagicMock(),
        on_delete_clicked=MagicMock(),
    )


# ----------------------------------------------------------------------
# build_model_combo
# ----------------------------------------------------------------------
class TestBuildModelCombo:
    def test_populates_items_from_data(self):
        data = RunRowData(available_models=[("M1", "/path/m1.onnx"), ("M2", "/path/m2.onnx")])
        combo = build_model_combo(_ComboMock, data, _make_callbacks())
        assert combo.count() == 2
        assert combo.itemText(0) == "M1"
        assert combo.itemData(1) == "/path/m2.onnx"

    def test_selects_initial_model_by_label(self):
        data = RunRowData(
            available_models=[("M1", "/p1"), ("M2", "/p2")],
            initial_model_name="M2",
        )
        combo = build_model_combo(_ComboMock, data, _make_callbacks())
        assert combo.currentIndex() == 1

    def test_selects_initial_model_by_path(self):
        data = RunRowData(
            available_models=[("M1", "/p1"), ("M2", "/p2")],
            initial_model_name="/p2",
        )
        combo = build_model_combo(_ComboMock, data, _make_callbacks())
        assert combo.currentIndex() == 1

    def test_connects_three_callbacks(self):
        cbs = _make_callbacks()
        combo = build_model_combo(_ComboMock, RunRowData(), cbs)
        # 3 connexions : on_any_changed, on_refresh_classes, on_model_changed
        assert combo.currentIndexChanged.connect.call_count == 3


# ----------------------------------------------------------------------
# build_rvt_combo
# ----------------------------------------------------------------------
class TestBuildRvtCombo:
    def test_populates_with_label_text_and_key_data(self):
        # rvt_keys est (key, label) ; addItem reçoit (label, key) — la
        # data du combo est la clé (LD, SVF…), le label affiché est le
        # libellé humain.
        data = RunRowData(rvt_keys=[("LD", "Détection des dépressions"), ("SVF", "Sky View Factor")])
        combo = build_rvt_combo(_ComboMock, data, _make_callbacks())
        assert combo.count() == 2
        assert combo.itemText(0) == "Détection des dépressions"
        assert combo.itemData(0) == "LD"

    def test_selects_initial_target_rvt_by_data_key(self):
        data = RunRowData(
            rvt_keys=[("LD", "Det. dépressions"), ("SVF", "SVF")],
            initial_target_rvt="SVF",
        )
        combo = build_rvt_combo(_ComboMock, data, _make_callbacks())
        assert combo.currentIndex() == 1


# ----------------------------------------------------------------------
# build_min_area_spin
# ----------------------------------------------------------------------
class TestBuildMinAreaSpin:
    def test_sets_initial_value(self):
        data = RunRowData(initial_min_area_m2=120.0)
        spin = build_min_area_spin(_SpinMock, data, _make_callbacks())
        assert spin.value() == 120.0

    def test_default_zero_when_unset(self):
        spin = build_min_area_spin(_SpinMock, RunRowData(), _make_callbacks())
        assert spin.value() == 0.0

    def test_connects_value_changed(self):
        cbs = _make_callbacks()
        spin = build_min_area_spin(_SpinMock, RunRowData(), cbs)
        spin.valueChanged.connect.assert_called_once()


# ----------------------------------------------------------------------
# build_actions_widget
# ----------------------------------------------------------------------
class TestBuildActionsWidget:
    def test_creates_two_buttons_and_connects(self):
        cbs = _make_callbacks()
        captured: list = []

        def container_factory():
            obj = MagicMock()
            return obj

        def layout_factory(container):
            return _LayoutMock(container)

        def button_factory(text):
            btn = _ButtonMock(text)
            captured.append(btn)
            return btn

        build_actions_widget(container_factory, layout_factory, button_factory, cbs)
        assert len(captured) == 2  # info + delete
        # Le 1er bouton (info) connecte on_info_clicked, le 2nd (×) on_delete_clicked.
        captured[0].clicked.connect.assert_called_once_with(cbs.on_info_clicked)
        captured[1].clicked.connect.assert_called_once_with(cbs.on_delete_clicked)
