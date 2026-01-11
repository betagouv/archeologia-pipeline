from __future__ import annotations

from pathlib import Path

import pytest

from app.services.cv_service import ComputerVisionService


class TestComputerVisionServiceProperties:
    def test_enabled_returns_true_when_enabled(self):
        config = {"enabled": True}
        service = ComputerVisionService(config, Path("/tmp"), lambda m: None)
        assert service.enabled is True

    def test_enabled_returns_false_when_disabled(self):
        config = {"enabled": False}
        service = ComputerVisionService(config, Path("/tmp"), lambda m: None)
        assert service.enabled is False

    def test_enabled_returns_false_when_missing(self):
        config = {}
        service = ComputerVisionService(config, Path("/tmp"), lambda m: None)
        assert service.enabled is False

    def test_enabled_returns_false_for_empty_config(self):
        service = ComputerVisionService({}, Path("/tmp"), lambda m: None)
        assert service.enabled is False

    def test_target_rvt_returns_configured_value(self):
        config = {"target_rvt": "VAT"}
        service = ComputerVisionService(config, Path("/tmp"), lambda m: None)
        assert service.target_rvt == "VAT"

    def test_target_rvt_defaults_to_ld(self):
        config = {}
        service = ComputerVisionService(config, Path("/tmp"), lambda m: None)
        assert service.target_rvt == "LD"

    def test_generate_shapefiles_returns_true_when_enabled(self):
        config = {"generate_shapefiles": True}
        service = ComputerVisionService(config, Path("/tmp"), lambda m: None)
        assert service.generate_shapefiles is True

    def test_generate_shapefiles_returns_false_when_disabled(self):
        config = {"generate_shapefiles": False}
        service = ComputerVisionService(config, Path("/tmp"), lambda m: None)
        assert service.generate_shapefiles is False

    def test_generate_shapefiles_defaults_to_false(self):
        config = {}
        service = ComputerVisionService(config, Path("/tmp"), lambda m: None)
        assert service.generate_shapefiles is False


class TestShouldProcessProduct:
    def test_returns_true_for_matching_product_when_enabled(self):
        config = {"enabled": True, "target_rvt": "LD"}
        service = ComputerVisionService(config, Path("/tmp"), lambda m: None)
        assert service.should_process_product("LD") is True

    def test_returns_true_case_insensitive(self):
        config = {"enabled": True, "target_rvt": "LD"}
        service = ComputerVisionService(config, Path("/tmp"), lambda m: None)
        assert service.should_process_product("ld") is True
        assert service.should_process_product("Ld") is True

    def test_returns_false_for_non_matching_product(self):
        config = {"enabled": True, "target_rvt": "LD"}
        service = ComputerVisionService(config, Path("/tmp"), lambda m: None)
        assert service.should_process_product("VAT") is False

    def test_returns_false_when_disabled(self):
        config = {"enabled": False, "target_rvt": "LD"}
        service = ComputerVisionService(config, Path("/tmp"), lambda m: None)
        assert service.should_process_product("LD") is False


class TestProcessSingleJpg:
    def test_does_nothing_when_disabled(self):
        config = {"enabled": False}
        service = ComputerVisionService(config, Path("/tmp/output"), lambda m: None)
        
        service.process_single_jpg(Path("/tmp/test.jpg"), Path("/tmp/rvt"))
        
        assert service._labels_dir is None
        assert service._shp_dir is None

    def test_updates_internal_state_when_enabled(self):
        config = {"enabled": True, "target_rvt": "LD"}
        logs = []
        service = ComputerVisionService(config, Path("/tmp/output"), logs.append)
        
        jpg_path = Path("/tmp/rvt/LD/jpg/test.jpg")
        rvt_base_dir = Path("/tmp/rvt/LD")
        tif_data = {"test": (0.5, -0.5, 100.0, 200.0)}
        
        service.process_single_jpg(jpg_path, rvt_base_dir, tif_data)
        
        assert service._labels_dir == jpg_path.parent
        assert service._shp_dir == rvt_base_dir / "shapefiles"
        assert "test" in service._tif_transform_data


class TestFinalize:
    def test_does_nothing_when_disabled(self):
        config = {"enabled": False, "generate_shapefiles": True}
        logs = []
        service = ComputerVisionService(config, Path("/tmp/output"), logs.append)
        service._labels_dir = Path("/tmp/labels")
        service._shp_dir = Path("/tmp/shp")
        
        service.finalize()

    def test_does_nothing_when_no_shapefiles_config(self):
        config = {"enabled": True, "generate_shapefiles": False}
        logs = []
        service = ComputerVisionService(config, Path("/tmp/output"), logs.append)
        
        service.finalize()

    def test_does_nothing_when_no_labels_dir(self):
        config = {"enabled": True, "generate_shapefiles": True}
        logs = []
        service = ComputerVisionService(config, Path("/tmp/output"), logs.append)
        service._labels_dir = None
        service._shp_dir = Path("/tmp/shp")
        
        service.finalize()

    def test_does_nothing_when_no_shp_dir(self):
        config = {"enabled": True, "generate_shapefiles": True}
        logs = []
        service = ComputerVisionService(config, Path("/tmp/output"), logs.append)
        service._labels_dir = Path("/tmp/labels")
        service._shp_dir = None
        
        service.finalize()


class TestReset:
    def test_clears_internal_state(self):
        config = {"enabled": True}
        service = ComputerVisionService(config, Path("/tmp/output"), lambda m: None)
        service._tif_transform_data = {"test": (1, 2, 3, 4)}
        service._labels_dir = Path("/tmp/labels")
        service._shp_dir = Path("/tmp/shp")
        
        service.reset()
        
        assert service._tif_transform_data == {}
        assert service._labels_dir is None
        assert service._shp_dir is None
