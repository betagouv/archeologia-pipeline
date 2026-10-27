"""Tests de la logique pure des indices (étape 2)."""
from __future__ import annotations

from app.services.indices_model import (
    all_products,
    base_keys,
    count_selected,
    default_products,
    product,
    products_unavailable_in_mode,
    requires_mnt,
    rvt_keys,
    toggle,
)


class TestCatalog:
    def test_rvt_keys_order(self):
        assert rvt_keys() == ["HS", "M_HS", "SVF", "OPNS", "SLO", "LD", "SLRM", "VAT",
                              "MSTP", "CVAT", "PRISM", "CRIM"]

    def test_base_keys(self):
        assert base_keys() == ["MNT", "DENSITE", "COUVERTURE"]

    def test_all_products_count(self):
        assert len(all_products()) == 15

    def test_product_lookup_has_metadata(self):
        p = product("M_HS")
        assert p.tag == "M-HS"
        assert p.full_name
        assert p.description
        assert p.is_rvt is True

    def test_defauts_coches(self):
        # Trois images cochées d'avance (lot 3, 2026-10-24) : plus de pastille rouge à
        # la première ouverture, et un premier regard utile (relief nu, creux, reliefs).
        d = default_products()
        assert set(d.keys()) == {p.key for p in all_products()}
        assert {k for k, v in d.items() if v} == {"MNT", "SVF", "LD"}

    def test_sections_partitionnent_les_produits(self):
        from app.services.indices_model import other_rvt_keys, quality_keys, recommended_keys
        assert recommended_keys() == ["MNT", "SVF", "LD"]
        assert quality_keys() == ["DENSITE", "COUVERTURE"]
        autres = other_rvt_keys()
        assert "SVF" not in autres and "LD" not in autres and "MNT" not in autres
        tout = recommended_keys() + autres + quality_keys()
        assert sorted(tout) == sorted(p.key for p in all_products())
        assert len(tout) == len(set(tout))


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


class TestCouvertureEntry:
    def test_presente_au_catalogue_comme_produit_de_base(self):
        p = product("COUVERTURE")
        assert p.is_rvt is False
        assert p.tag == "Couverture"
        assert "COUVERTURE" in base_keys()

    def test_decochee_par_defaut(self):
        assert default_products()["COUVERTURE"] is False

    def test_toggle_ne_force_pas_mnt(self):
        rien = {k: False for k in default_products()}
        new, toast = toggle(rien, "COUVERTURE")
        assert new["COUVERTURE"] is True
        assert new["MNT"] is False
        assert toast is None


class TestProductsUnavailableInMode:
    """Produits dérivés du nuage de points : indisponibles dans les modes
    existants (entrée déjà interpolée). Sert à purger la sélection résiduelle
    au changement de mode (sinon COUVERTURE/DENSITE coché en mode LAZ persiste)."""

    def test_laz_modes_have_no_unavailable_products(self):
        assert products_unavailable_in_mode("ign_laz") == []
        assert products_unavailable_in_mode("local_laz") == []

    def test_existing_mnt_excludes_point_cloud_products(self):
        assert products_unavailable_in_mode("existing_mnt") == ["DENSITE", "COUVERTURE"]

    def test_existing_rvt_excludes_point_cloud_products(self):
        assert products_unavailable_in_mode("existing_rvt") == ["DENSITE", "COUVERTURE"]

    def test_mnt_input_stays_available_in_existing_mnt(self):
        # Le MNT est l'ENTRÉE du mode (copiable vers les résultats) — pas purgé.
        assert "MNT" not in products_unavailable_in_mode("existing_mnt")

    def test_rvt_products_stay_available_in_existing_mnt(self):
        # Les indices RVT se calculent depuis le MNT existant — pas purgés.
        assert len(rvt_keys()) >= 9, "anti-test-creux : plus aucun indice RVT à vérifier"
        for k in rvt_keys():
            assert k not in products_unavailable_in_mode("existing_mnt")

    def test_unknown_mode_is_safe_empty(self):
        assert products_unavailable_in_mode("") == []
        assert products_unavailable_in_mode("autre") == []
