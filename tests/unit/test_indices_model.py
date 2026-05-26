"""Tests de la logique pure des indices (étape 2)."""
from __future__ import annotations

from app.services.indices_model import (
    all_products,
    base_keys,
    count_selected,
    default_products,
    product,
    requires_mnt,
    rvt_keys,
    toggle,
)


class TestCatalog:
    def test_rvt_keys_order(self):
        assert rvt_keys() == ["HS", "M_HS", "SVF", "SLO", "LD", "SLRM", "VAT"]

    def test_base_keys(self):
        assert base_keys() == ["MNT", "DENSITE"]

    def test_all_products_count(self):
        assert len(all_products()) == 9

    def test_product_lookup_has_metadata(self):
        p = product("M_HS")
        assert p.tag == "M-HS"
        assert p.full_name
        assert p.description
        assert p.is_rvt is True

    def test_default_products_none_selected(self):
        # Plus de sélection recommandée : aucun produit n'est pré-coché.
        d = default_products()
        assert set(d.keys()) == {p.key for p in all_products()}
        assert all(v is False for v in d.values())


class TestRequiresMnt:
    def test_true_when_any_rvt_on(self):
        assert requires_mnt({"LD": True}) is True

    def test_false_when_no_rvt(self):
        assert requires_mnt({"MNT": True, "DENSITE": True}) is False


class TestCountSelected:
    def test_counts_only_true(self):
        assert count_selected({"MNT": True, "LD": True, "SVF": False}) == 2


class TestToggle:
    def test_toggle_rvt_on_forces_mnt(self):
        new, toast = toggle({"MNT": False, "LD": False}, "LD")
        assert new["LD"] is True
        assert new["MNT"] is True
        assert toast is None

    def test_toggle_rvt_off_keeps_mnt(self):
        new, toast = toggle({"MNT": True, "LD": True, "SVF": True}, "LD")
        assert new["LD"] is False
        assert new["MNT"] is True  # SVF encore actif
        assert toast is None

    def test_toggle_mnt_off_without_rvt(self):
        new, toast = toggle({"MNT": True, "DENSITE": True}, "MNT")
        assert new["MNT"] is False
        assert toast is None

    def test_toggle_mnt_off_with_rvt_is_blocked_with_toast(self):
        original = {"MNT": True, "LD": True, "SVF": True}
        new, toast = toggle(original, "MNT")
        assert new["MNT"] is True  # verrou : non décochable
        assert toast is not None
        assert "M-HS" not in toast  # M-HS pas actif
        assert "LD" in toast and "SVF" in toast

    def test_toggle_densite_independent(self):
        new, toast = toggle({"MNT": True, "DENSITE": False}, "DENSITE")
        assert new["DENSITE"] is True
        assert toast is None
