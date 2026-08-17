"""Tests pour la logique pure des modes de données (frise étape 1)."""
from __future__ import annotations

from pathlib import Path

from app.services.source_modes import (
    DATA_MODES,
    mode_info,
    normalize_vector_input,
    ordered_modes,
    path_is_valid,
    path_state,
    pipeline_stages,
)


class TestModeInfo:
    def test_ign_laz_is_file_source(self):
        info = mode_info("ign_laz")
        assert info.config_key == "input_file"
        assert info.is_file is True

    def test_local_laz_is_dir_source(self):
        info = mode_info("local_laz")
        assert info.config_key == "local_laz_dir"
        assert info.is_file is False

    def test_existing_mnt_dir(self):
        assert mode_info("existing_mnt").config_key == "existing_mnt_dir"

    def test_existing_rvt_dir(self):
        assert mode_info("existing_rvt").config_key == "existing_rvt_dir"

    def test_unknown_mode_falls_back_to_ign(self):
        assert mode_info("nope").mode == "ign_laz"

    def test_each_mode_has_labels_and_description(self):
        for mode in ordered_modes():
            info = mode_info(mode)
            assert info.source_label
            assert info.description
            assert info.banner_label
            assert info.icon
            assert info.entry_stage in (1, 2, 3, 4)

    def test_ordered_modes_matches_frise(self):
        assert ordered_modes() == ["ign_laz", "local_laz", "existing_mnt", "existing_rvt"]

    def test_data_modes_keyed_by_mode(self):
        assert set(DATA_MODES.keys()) == set(ordered_modes())

    def test_ign_has_vector_exts(self):
        assert ".shp" in mode_info("ign_laz").valid_exts
        assert ".dbf" in mode_info("ign_laz").valid_exts

    def test_dir_modes_have_no_exts(self):
        assert mode_info("local_laz").valid_exts == ()

    def test_entry_stage_mapping(self):
        assert mode_info("ign_laz").entry_stage == 1
        assert mode_info("local_laz").entry_stage == 2
        assert mode_info("existing_mnt").entry_stage == 3
        assert mode_info("existing_rvt").entry_stage == 4


class TestPipelineStages:
    def test_five_stages(self):
        assert len(pipeline_stages()) == 5

    def test_first_four_clickable_match_modes(self):
        stages = pipeline_stages()
        assert [s.mode for s in stages[:4]] == ordered_modes()

    def test_last_stage_is_detection_optional_non_clickable(self):
        last = pipeline_stages()[-1]
        assert last.mode is None
        assert last.optional is True
        assert last.id == 5

    def test_every_stage_has_icon_and_label(self):
        for s in pipeline_stages():
            assert s.icon
            assert s.label
            assert s.sub


class TestPathState:
    def test_empty_is_ok(self):
        assert path_state("", expect_dir=True) == "ok"

    def test_existing_dir_ok(self, tmp_path):
        assert path_state(str(tmp_path), expect_dir=True) == "ok"

    def test_missing_dir_allow_create_ok(self, tmp_path):
        assert path_state(str(tmp_path / "new"), expect_dir=True, allow_create=True) == "ok"

    def test_missing_dir_error(self, tmp_path):
        assert path_state(str(tmp_path / "new"), expect_dir=True) == "error"

    def test_missing_file_error(self, tmp_path):
        assert path_state(str(tmp_path / "x.shp"), expect_dir=False) == "error"

    def test_existing_file_ok_without_ext_constraint(self, tmp_path):
        f = tmp_path / "x.weird"
        f.write_text("x", encoding="utf-8")
        assert path_state(str(f), expect_dir=False) == "ok"

    def test_existing_file_unexpected_ext_warns(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("x", encoding="utf-8")
        assert path_state(str(f), expect_dir=False, valid_exts=(".shp", ".gpkg")) == "warn"

    def test_existing_file_expected_ext_ok(self, tmp_path):
        f = tmp_path / "x.shp"
        f.write_text("x", encoding="utf-8")
        assert path_state(str(f), expect_dir=False, valid_exts=(".shp", ".gpkg")) == "ok"


class TestNormalizeVectorInput:
    def test_dbf_with_sibling_shp_becomes_shp(self, tmp_path):
        (tmp_path / "zone.shp").write_text("x", encoding="utf-8")
        dbf = tmp_path / "zone.dbf"
        dbf.write_text("x", encoding="utf-8")
        assert normalize_vector_input(dbf) == tmp_path / "zone.shp"

    def test_dbf_without_sibling_shp_unchanged(self, tmp_path):
        dbf = tmp_path / "zone.dbf"
        dbf.write_text("x", encoding="utf-8")
        assert normalize_vector_input(dbf) == dbf

    def test_dbf_uppercase_suffix_normalized(self, tmp_path):
        (tmp_path / "zone.shp").write_text("x", encoding="utf-8")
        dbf = tmp_path / "zone.DBF"
        dbf.write_text("x", encoding="utf-8")
        assert normalize_vector_input(dbf) == tmp_path / "zone.shp"

    def test_non_dbf_untouched(self):
        assert normalize_vector_input(Path("/x/zone.shp")) == Path("/x/zone.shp")
        assert normalize_vector_input(Path("/x/dalles.txt")) == Path("/x/dalles.txt")


class TestPathIsValid:
    def test_empty_is_valid_neutral(self):
        assert path_is_valid("", expect_dir=True) is True
        assert path_is_valid("   ", expect_dir=False) is True

    def test_existing_dir_when_expecting_dir(self, tmp_path):
        assert path_is_valid(str(tmp_path), expect_dir=True) is True

    def test_existing_file_when_expecting_dir_is_invalid(self, tmp_path):
        f = tmp_path / "x.shp"
        f.write_text("x", encoding="utf-8")
        assert path_is_valid(str(f), expect_dir=True) is False

    def test_existing_file_when_expecting_file(self, tmp_path):
        f = tmp_path / "x.shp"
        f.write_text("x", encoding="utf-8")
        assert path_is_valid(str(f), expect_dir=False) is True

    def test_missing_dir_with_allow_create_is_valid(self, tmp_path):
        missing = tmp_path / "to_create"
        assert path_is_valid(str(missing), expect_dir=True, allow_create=True) is True

    def test_missing_dir_without_allow_create_is_invalid(self, tmp_path):
        missing = tmp_path / "nope"
        assert path_is_valid(str(missing), expect_dir=True) is False

    def test_missing_file_is_invalid(self, tmp_path):
        assert path_is_valid(str(tmp_path / "nope.shp"), expect_dir=False) is False
