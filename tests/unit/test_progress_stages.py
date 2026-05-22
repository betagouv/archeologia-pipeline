from __future__ import annotations

import pytest

from app.progress_stages import (
    STAGE_LABELS,
    Stage,
    build_stage_sequence,
)

ALL_MODES = ["ign_laz", "local_laz", "existing_mnt", "existing_rvt"]


class TestStageLabels:
    def test_every_stage_id_has_a_label(self):
        ids = {Stage.DOWNLOAD, Stage.PRODUCTS, Stage.DETECTION, Stage.FINALIZE}
        assert ids <= set(STAGE_LABELS)
        assert all(STAGE_LABELS[i] for i in ids)


class TestBuildStageSequence:
    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            build_stage_sequence("nope", cv_enabled=True)

    @pytest.mark.parametrize("mode", ALL_MODES)
    @pytest.mark.parametrize("cv", [True, False])
    def test_always_ends_with_finalize(self, mode, cv):
        seq = build_stage_sequence(mode, cv_enabled=cv)
        assert seq[-1] == Stage.FINALIZE
        assert len(seq) >= 2

    @pytest.mark.parametrize("mode", ALL_MODES)
    @pytest.mark.parametrize("cv", [True, False])
    def test_all_ids_known(self, mode, cv):
        seq = build_stage_sequence(mode, cv_enabled=cv)
        assert all(s in STAGE_LABELS for s in seq)

    def test_ign_full_with_cv(self):
        assert build_stage_sequence("ign_laz", cv_enabled=True) == [
            Stage.DOWNLOAD,
            Stage.PRODUCTS,
            Stage.DETECTION,
            Stage.FINALIZE,
        ]

    def test_ign_without_cv_drops_detection(self):
        assert build_stage_sequence("ign_laz", cv_enabled=False) == [
            Stage.DOWNLOAD,
            Stage.PRODUCTS,
            Stage.FINALIZE,
        ]

    def test_local_has_no_download(self):
        seq = build_stage_sequence("local_laz", cv_enabled=True)
        assert Stage.DOWNLOAD not in seq
        assert seq == [Stage.PRODUCTS, Stage.DETECTION, Stage.FINALIZE]

    def test_existing_mnt_has_products_not_download(self):
        seq = build_stage_sequence("existing_mnt", cv_enabled=False)
        assert seq == [Stage.PRODUCTS, Stage.FINALIZE]

    def test_existing_rvt_is_detection_then_finalize(self):
        # existing_rvt est fondamentalement de la détection : pas de pastille
        # produits, et la détection reste affichée même si la prep TIF→PNG
        # occupe le début de la barre.
        assert build_stage_sequence("existing_rvt", cv_enabled=True) == [
            Stage.DETECTION,
            Stage.FINALIZE,
        ]
        assert build_stage_sequence("existing_rvt", cv_enabled=False) == [
            Stage.DETECTION,
            Stage.FINALIZE,
        ]

    @pytest.mark.parametrize("mode", ["ign_laz", "local_laz", "existing_mnt"])
    def test_detection_present_iff_cv(self, mode):
        assert Stage.DETECTION in build_stage_sequence(mode, cv_enabled=True)
        assert Stage.DETECTION not in build_stage_sequence(mode, cv_enabled=False)
