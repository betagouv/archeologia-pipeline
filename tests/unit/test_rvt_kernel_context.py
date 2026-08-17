"""Contexte spatial requis par les noyaux RVT (app.services.rvt_kernel_context).

Chaque visualisation RVT lit un voisinage autour de chaque pixel. Si le raster
fourni au calcul ne s'étend pas d'au moins ce rayon, RVT fabrique le voisinage
manquant par symétrie (``np.pad(mode="symmetric")``) : la valeur produite est
inventée. Ces tests figent le diagnostic qui prévient l'utilisateur.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config.config_manager import ConfigManager

from app.services.rvt_kernel_context import (
    full_context_ratio,
    kernel_radius_px,
    mstp_scale_errors,
    raster_context_warnings,
    tile_margin_px,
    tiled_context_warnings,
)


class TestKernelRadiusPx:
    def test_reads_configured_radius(self):
        assert kernel_radius_px("MSTP", {"mstp": {"broad_scale_max": 800}}) == 800

    def test_falls_back_to_rvt_default_when_absent(self):
        assert kernel_radius_px("MSTP", {}) == 2023

    def test_local_dominance_section_is_ldo_not_ld(self):
        # Le code court du produit est « LD » mais la section rvt_params est « ldo ».
        assert kernel_radius_px("LD", {"ldo": {"max_radius": 55}}) == 55

    def test_none_for_products_without_a_tunable_kernel(self):
        # HS/SLO travaillent sur un 3x3 : aucun rayon à surveiller.
        assert kernel_radius_px("HS", {}) is None
        assert kernel_radius_px("SLO", {}) is None

    def test_ignores_unparsable_value(self):
        assert kernel_radius_px("SVF", {"svf": {"radius": "beaucoup"}}) == 10


class TestTileMarginPx:
    def test_twenty_percent_at_fifty_centimeters_gives_four_hundred_px(self):
        assert tile_margin_px(20, 0.5) == 400

    def test_zero_overlap_gives_no_context(self):
        assert tile_margin_px(0, 0.5) == 0

    def test_mirrors_the_999_meter_cap_of_prepare_merged_tiles(self):
        # preprocess.prepare_merged_tiles plafonne la marge à 999 m.
        assert tile_margin_px(100, 1.0) == 999

    def test_coarser_resolution_gives_fewer_pixels_for_the_same_margin(self):
        assert tile_margin_px(20, 1.0) == 200


class TestFullContextRatio:
    def test_fraction_of_pixels_with_complete_neighbourhood(self):
        # 6000 px de côté, rayon 2023 → (6000 - 2*2023)² / 6000²
        assert full_context_ratio(6000, 6000, 2023) == pytest.approx(0.1061, abs=1e-4)

    def test_zero_when_kernel_exceeds_half_the_raster(self):
        assert full_context_ratio(600, 1000, 2023) == 0.0

    def test_one_when_kernel_is_zero(self):
        assert full_context_ratio(600, 1000, 0) == 1.0

    def test_zero_on_degenerate_raster_instead_of_dividing_by_zero(self):
        assert full_context_ratio(0, 0, 10) == 0.0


class TestTiledContextWarnings:
    def _warn(self, rvt_params, *, overlap=20, resolution=0.5, products=None):
        return tiled_context_warnings(
            products if products is not None else {"MSTP": True},
            rvt_params,
            tile_overlap_percent=overlap,
            mnt_resolution=resolution,
        )

    def test_default_mstp_on_default_tiling_is_flagged(self):
        msgs = self._warn({})
        assert len(msgs) == 1
        assert "MSTP" in msgs[0]
        assert "2023" in msgs[0]
        assert "400" in msgs[0]

    def test_silent_once_the_kernel_fits_the_margin(self):
        assert self._warn({"mstp": {"broad_scale_max": 400}}) == []

    def test_kernel_exactly_equal_to_the_margin_is_accepted(self):
        assert self._warn({"mstp": {"broad_scale_max": 400}}) == []

    def test_unchecked_product_is_never_flagged(self):
        assert self._warn({}, products={"MSTP": False}) == []

    def test_check_is_generic_not_mstp_specific(self):
        msgs = self._warn({"svf": {"radius": 800}}, products={"SVF": True})
        assert len(msgs) == 1
        assert "SVF" in msgs[0]
        assert "800" in msgs[0]

    def test_names_the_field_to_edit_so_the_message_is_actionable(self):
        assert "Échelle large — rayon max" in self._warn({})[0]

    def test_one_message_per_offending_product(self):
        msgs = self._warn(
            {"svf": {"radius": 800}},
            products={"MSTP": True, "SVF": True, "HS": True},
        )
        assert len(msgs) == 2


class TestRasterContextWarnings:
    def test_small_mnt_reports_zero_percent_valid(self):
        msgs = raster_context_warnings(
            {"MSTP": True}, {}, width_px=600, height_px=1000
        )
        assert len(msgs) == 1
        assert "0 %" in msgs[0]
        assert "2023" in msgs[0]

    def test_large_mnt_reports_the_partial_coverage(self):
        msgs = raster_context_warnings(
            {"MSTP": True}, {}, width_px=6000, height_px=6000
        )
        assert len(msgs) == 1
        assert "11 %" in msgs[0]

    def test_silent_when_the_raster_is_wide_enough(self):
        assert raster_context_warnings(
            {"MSTP": True}, {"mstp": {"broad_scale_max": 200}},
            width_px=6000, height_px=6000,
        ) == []


class TestMstpScaleErrors:
    def test_rvt_defaults_are_valid(self):
        assert mstp_scale_errors({"MSTP": True}, {}) == []

    def test_rejects_a_span_smaller_than_its_step(self):
        # rvt.vis.mstp lève si (max - min) < step → la dalle entière planterait.
        msgs = mstp_scale_errors({"MSTP": True}, {"mstp": {"broad_scale_max": 400}})
        assert len(msgs) == 1
        assert "large" in msgs[0].lower()

    def test_rejects_inverted_bounds(self):
        msgs = mstp_scale_errors(
            {"MSTP": True}, {"mstp": {"meso_scale_min": 500, "meso_scale_max": 100}}
        )
        assert len(msgs) == 1
        assert "méso" in msgs[0].lower()

    def test_silent_when_mstp_is_not_requested(self):
        assert mstp_scale_errors({"MSTP": False}, {"mstp": {"broad_scale_max": 400}}) == []


class TestDefaultConfigIsSelfConsistent:
    """Les défauts livrés doivent tenir dans le tuilage livré.

    Verrouille la *relation* (noyau ≤ marge) et non un nombre : si quelqu'un
    change ``tile_overlap`` ou ``mnt_resolution`` par défaut sans revoir les
    échelles MSTP, la contradiction est attrapée ici et pas par l'utilisateur.
    """

    def _defaults(self):
        repo_root = Path(__file__).resolve().parents[2]
        return ConfigManager(repo_root).default_config()

    def test_mstp_kernel_fits_the_default_tile_margin(self):
        cfg = self._defaults()
        proc = cfg["processing"]
        assert tiled_context_warnings(
            {"MSTP": True},
            cfg["rvt_params"],
            tile_overlap_percent=proc["tile_overlap"],
            mnt_resolution=proc["mnt_resolution"],
        ) == []

    def test_every_kernel_product_fits_the_default_tile_margin(self):
        cfg = self._defaults()
        proc = cfg["processing"]
        all_on = {k: True for k in ("SVF", "LD", "SLRM", "MSTP")}
        assert tiled_context_warnings(
            all_on,
            cfg["rvt_params"],
            tile_overlap_percent=proc["tile_overlap"],
            mnt_resolution=proc["mnt_resolution"],
        ) == []

    def test_default_mstp_scales_are_accepted_by_rvt(self):
        assert mstp_scale_errors({"MSTP": True}, self._defaults()["rvt_params"]) == []
