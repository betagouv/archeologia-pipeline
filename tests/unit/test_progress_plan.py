from __future__ import annotations

import pytest

from app.runners.progress_plan import ProgressPlan, build_progress_plan, cv_pct

ALL_MODES = ["ign_laz", "local_laz", "existing_mnt", "existing_rvt"]


def _bands_in_order(plan: ProgressPlan):
    return [plan.download, plan.merge, plan.products, plan.cv, plan.finalize]


class TestProgressPlanAt:
    def test_endpoints(self):
        plan = ProgressPlan((0, 20), (20, 30), (30, 75), (75, 95), (95, 100))
        assert plan.at(plan.products, 0.0) == 30
        assert plan.at(plan.products, 1.0) == 75

    def test_midpoint_rounds(self):
        plan = build_progress_plan("ign_laz", cv_enabled=True)
        # products = (30, 75) → milieu = 52.5 → 52 (round half to even) ou 53
        mid = plan.at(plan.products, 0.5)
        assert 52 <= mid <= 53

    def test_fraction_is_clamped(self):
        plan = ProgressPlan((0, 0), (0, 0), (0, 95), (95, 95), (95, 100))
        assert plan.at(plan.products, -1.0) == 0
        assert plan.at(plan.products, 2.0) == 95

    def test_degenerate_band(self):
        plan = build_progress_plan("ign_laz", cv_enabled=False)
        # CV off → bande cv dégénérée (95, 95)
        assert plan.cv == (95, 95)
        assert plan.at(plan.cv, 0.0) == 95
        assert plan.at(plan.cv, 1.0) == 95


class TestBuildProgressPlan:
    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            build_progress_plan("nope", cv_enabled=True)

    @pytest.mark.parametrize("mode", ALL_MODES)
    @pytest.mark.parametrize("cv", [True, False])
    def test_bands_start_at_zero_end_at_100(self, mode, cv):
        plan = build_progress_plan(mode, cv_enabled=cv)
        bands = _bands_in_order(plan)
        assert bands[0][0] == 0, "première bande démarre à 0"
        assert bands[-1][1] == 100, "finalize finit à 100"

    @pytest.mark.parametrize("mode", ALL_MODES)
    @pytest.mark.parametrize("cv", [True, False])
    def test_bands_contiguous_and_monotone(self, mode, cv):
        plan = build_progress_plan(mode, cv_enabled=cv)
        bands = _bands_in_order(plan)
        for lo, hi in bands:
            assert lo <= hi, f"bande non croissante: ({lo},{hi})"
        for (a_lo, a_hi), (b_lo, b_hi) in zip(bands, bands[1:]):
            assert a_hi == b_lo, f"bandes non contiguës: {a_hi} != {b_lo}"

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_cv_band_degenerate_when_cv_off(self, mode):
        if mode == "existing_rvt":
            pytest.skip("existing_rvt a toujours une phase CV")
        plan = build_progress_plan(mode, cv_enabled=False)
        assert plan.cv[0] == plan.cv[1], "CV off → bande CV dégénérée"

    def test_ign_with_cv_known_bands(self):
        plan = build_progress_plan("ign_laz", cv_enabled=True)
        assert plan.download == (0, 20)
        assert plan.merge == (20, 30)
        assert plan.products == (30, 75)
        assert plan.cv == (75, 95)
        assert plan.finalize == (95, 100)

    def test_local_has_no_download_or_merge(self):
        plan = build_progress_plan("local_laz", cv_enabled=True)
        assert plan.download == (0, 0)
        assert plan.merge == (0, 0)
        assert plan.products[0] == 0

    def test_existing_rvt_has_prep_then_cv(self):
        plan = build_progress_plan("existing_rvt", cv_enabled=True)
        assert plan.products == (0, 10)  # prep TIF→PNG
        assert plan.cv == (10, 95)
        assert plan.finalize == (95, 100)


class TestCvPct:
    def test_single_run_endpoints(self):
        band = (10, 95)
        assert cv_pct(1, 1, 0, 10, band) == 10
        assert cv_pct(1, 1, 10, 10, band) == 95

    def test_within_band(self):
        band = (75, 95)
        v = cv_pct(1, 1, 5, 10, band)
        assert 75 <= v <= 95

    def test_monotone_across_images(self):
        band = (10, 95)
        seq = [cv_pct(1, 1, i, 10, band) for i in range(0, 11)]
        assert seq == sorted(seq)
        assert seq[0] == 10 and seq[-1] == 95

    def test_monotone_across_runs(self):
        band = (10, 95)
        # fin du run 1 <= début du run 2
        end_run1 = cv_pct(1, 2, 10, 10, band)
        start_run2 = cv_pct(2, 2, 0, 10, band)
        assert end_run1 <= start_run2

    def test_two_runs_split_band_evenly(self):
        band = (10, 90)
        # fin du run 1 ≈ milieu de la bande
        assert cv_pct(1, 2, 10, 10, band) == 50

    def test_bounded_when_total_zero(self):
        band = (10, 95)
        v = cv_pct(1, 1, 0, 0, band)
        assert 10 <= v <= 95

    def test_bounded_when_no_runs(self):
        band = (10, 95)
        v = cv_pct(1, 0, 1, 1, band)
        assert 10 <= v <= 95
