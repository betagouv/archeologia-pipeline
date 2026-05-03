from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from config.config_manager import ConfigManager


@pytest.fixture
def cm(tmp_path: Path) -> ConfigManager:
    """ConfigManager with a temporary plugin root (no pre-existing config.json)."""
    return ConfigManager(tmp_path)


@pytest.fixture
def cm_with_file(tmp_path: Path) -> ConfigManager:
    """ConfigManager whose config.json already exists with custom values."""
    cfg = {
        "app": {"files": {"data_mode": "local_laz", "output_dir": "/custom"}},
        "processing": {"mnt_resolution": 1.0},
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return ConfigManager(tmp_path)


# ── default_config ──────────────────────────────────────────────

class TestDefaultConfig:
    def test_returns_dict(self, cm: ConfigManager):
        cfg = cm.default_config()
        assert isinstance(cfg, dict)

    def test_has_top_level_keys(self, cm: ConfigManager):
        cfg = cm.default_config()
        assert set(cfg.keys()) == {"app", "processing", "computer_vision", "rvt_params"}

    def test_default_data_mode_is_ign_laz(self, cm: ConfigManager):
        cfg = cm.default_config()
        assert cfg["app"]["files"]["data_mode"] == "ign_laz"

    def test_default_products_contains_slrm(self, cm: ConfigManager):
        cfg = cm.default_config()
        assert "SLRM" in cfg["processing"]["products"]

    def test_default_cv_disabled(self, cm: ConfigManager):
        cfg = cm.default_config()
        assert cfg["computer_vision"]["enabled"] is False

    def test_default_cv_runs_empty(self, cm: ConfigManager):
        cfg = cm.default_config()
        assert cfg["computer_vision"]["runs"] == []

    def test_default_rvt_params_has_all_sections(self, cm: ConfigManager):
        cfg = cm.default_config()
        expected = {"mdh", "svf", "slope", "ldo", "slrm", "vat"}
        assert set(cfg["rvt_params"].keys()) == expected


# ── save / load round-trip ──────────────────────────────────────

class TestSaveLoad:
    def test_save_creates_file(self, cm: ConfigManager):
        cm.save(cm.default_config())
        assert cm.path.exists()

    def test_load_creates_default_when_no_file(self, cm: ConfigManager):
        cfg = cm.load()
        assert cfg["app"]["files"]["data_mode"] == "ign_laz"
        assert cm.path.exists()  # file created as side-effect

    def test_round_trip_preserves_values(self, cm: ConfigManager):
        original = cm.default_config()
        original["app"]["files"]["output_dir"] = "/some/path"
        original["processing"]["mnt_resolution"] = 2.0
        cm.save(original)
        loaded = cm.load()
        assert loaded["app"]["files"]["output_dir"] == "/some/path"
        assert loaded["processing"]["mnt_resolution"] == 2.0

    def test_load_merges_with_defaults(self, cm_with_file: ConfigManager):
        cfg = cm_with_file.load()
        # Custom values preserved
        assert cfg["app"]["files"]["data_mode"] == "local_laz"
        assert cfg["processing"]["mnt_resolution"] == 1.0
        # Default values filled in for missing keys
        assert "computer_vision" in cfg
        assert "rvt_params" in cfg
        assert cfg["computer_vision"]["enabled"] is False

    def test_load_handles_corrupt_json(self, tmp_path: Path):
        (tmp_path / "config.json").write_text("{not valid json!!!", encoding="utf-8")
        cm = ConfigManager(tmp_path)
        cfg = cm.load()
        # Should fallback to defaults
        assert cfg["app"]["files"]["data_mode"] == "ign_laz"

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        nested = tmp_path / "sub" / "dir"
        cm = ConfigManager(nested)
        cm.save(cm.default_config())
        assert (nested / "config.json").exists()

    def test_custom_filename(self, tmp_path: Path):
        cm = ConfigManager(tmp_path, filename="custom.json")
        cm.save(cm.default_config())
        assert (tmp_path / "custom.json").exists()
        assert not (tmp_path / "config.json").exists()


# ── _deep_update ────────────────────────────────────────────────

class TestDeepUpdate:
    def test_simple_override(self, cm: ConfigManager):
        base = {"a": 1, "b": 2}
        cm._deep_update(base, {"a": 10})
        assert base == {"a": 10, "b": 2}

    def test_nested_merge(self, cm: ConfigManager):
        base = {"x": {"a": 1, "b": 2}}
        cm._deep_update(base, {"x": {"b": 3}})
        assert base == {"x": {"a": 1, "b": 3}}

    def test_adds_new_keys(self, cm: ConfigManager):
        base = {"a": 1}
        cm._deep_update(base, {"b": 2})
        assert base == {"a": 1, "b": 2}

    def test_replaces_non_dict_with_value(self, cm: ConfigManager):
        base = {"a": "string"}
        cm._deep_update(base, {"a": {"nested": True}})
        assert base == {"a": {"nested": True}}

    def test_replaces_dict_with_non_dict(self, cm: ConfigManager):
        base = {"a": {"nested": True}}
        cm._deep_update(base, {"a": 42})
        assert base == {"a": 42}

    def test_deeply_nested(self, cm: ConfigManager):
        base = {"l1": {"l2": {"l3": {"val": 0}}}}
        cm._deep_update(base, {"l1": {"l2": {"l3": {"val": 99}}}})
        assert base["l1"]["l2"]["l3"]["val"] == 99

    def test_empty_other(self, cm: ConfigManager):
        base = {"a": 1}
        cm._deep_update(base, {})
        assert base == {"a": 1}


# ── _migrate_cv_runs ────────────────────────────────────────────

class TestMigrateCvRuns:
    def test_migrates_old_format(self):
        cfg = {
            "computer_vision": {
                "selected_model": "my_model",
                "target_rvt": "SVF",
                "runs": [],
            }
        }
        ConfigManager._migrate_cv_runs(cfg)
        assert cfg["computer_vision"]["runs"] == [
            {"model": "my_model", "target_rvt": "SVF"}
        ]

    def test_does_not_overwrite_existing_runs(self):
        existing_runs = [{"model": "m1", "target_rvt": "LD"}]
        cfg = {
            "computer_vision": {
                "selected_model": "old_model",
                "target_rvt": "SVF",
                "runs": existing_runs,
            }
        }
        ConfigManager._migrate_cv_runs(cfg)
        assert cfg["computer_vision"]["runs"] is existing_runs

    def test_empty_model_gives_empty_runs(self):
        cfg = {
            "computer_vision": {
                "selected_model": "",
                "target_rvt": "LD",
                "runs": [],
            }
        }
        ConfigManager._migrate_cv_runs(cfg)
        assert cfg["computer_vision"]["runs"] == []

    def test_no_cv_section(self):
        cfg = {"app": {}}
        ConfigManager._migrate_cv_runs(cfg)  # Should not raise

    def test_non_dict_cv(self):
        cfg = {"computer_vision": "invalid"}
        ConfigManager._migrate_cv_runs(cfg)  # Should not raise

    def test_default_target_rvt_is_LD(self):
        cfg = {
            "computer_vision": {
                "selected_model": "model_x",
                "runs": [],
            }
        }
        ConfigManager._migrate_cv_runs(cfg)
        assert cfg["computer_vision"]["runs"][0]["target_rvt"] == "LD"

    def test_none_runs_triggers_migration(self):
        cfg = {
            "computer_vision": {
                "selected_model": "model_y",
                "target_rvt": "M_HS",
                "runs": None,
            }
        }
        ConfigManager._migrate_cv_runs(cfg)
        assert cfg["computer_vision"]["runs"] == [
            {"model": "model_y", "target_rvt": "M_HS"}
        ]
