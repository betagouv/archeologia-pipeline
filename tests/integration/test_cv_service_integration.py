from __future__ import annotations

from pathlib import Path

import pytest

from app.services.cv_service import ComputerVisionService


class TestComputerVisionServiceIntegration:
    def test_service_initializes_with_real_config(self, sample_config_dict: dict, temp_output_dir: Path):
        cv_config = sample_config_dict.get("computer_vision", {})
        
        service = ComputerVisionService(
            cv_config=cv_config,
            output_dir=temp_output_dir,
            log=lambda m: None,
        )
        
        assert service.enabled is False
        assert service.target_rvt == "LD"

    def test_service_with_cv_enabled(self, sample_config_dict: dict, temp_output_dir: Path):
        cv_config = sample_config_dict.get("computer_vision", {}).copy()
        cv_config["enabled"] = True
        cv_config["selected_model"] = "test_model"
        
        service = ComputerVisionService(
            cv_config=cv_config,
            output_dir=temp_output_dir,
            log=lambda m: None,
        )
        
        assert service.enabled is True
        assert service.should_process_product("LD") is True
        assert service.should_process_product("VAT") is False

    def test_process_single_jpg_updates_state(self, temp_output_dir: Path, mock_jpg_with_jgw: Path):
        cv_config = {"enabled": True, "target_rvt": "LD"}
        logs = []
        
        service = ComputerVisionService(
            cv_config=cv_config,
            output_dir=temp_output_dir,
            log=logs.append,
        )
        
        rvt_base_dir = mock_jpg_with_jgw.parent.parent
        tif_data = {"test_image": (0.5, -0.5, 100000.0, 6500000.0)}
        
        service.process_single_jpg(
            jpg_path=mock_jpg_with_jgw,
            rvt_base_dir=rvt_base_dir,
            tif_transform_data=tif_data,
        )
        
        assert service._labels_dir == mock_jpg_with_jgw.parent
        assert service._shp_dir == rvt_base_dir / "shapefiles"
        assert "test_image" in service._tif_transform_data

    def test_reset_clears_state(self, temp_output_dir: Path):
        cv_config = {"enabled": True, "target_rvt": "LD"}
        
        service = ComputerVisionService(
            cv_config=cv_config,
            output_dir=temp_output_dir,
            log=lambda m: None,
        )
        
        service._labels_dir = Path("/tmp/labels")
        service._shp_dir = Path("/tmp/shp")
        service._tif_transform_data = {"test": (1, 2, 3, 4)}
        
        service.reset()
        
        assert service._labels_dir is None
        assert service._shp_dir is None
        assert service._tif_transform_data == {}
