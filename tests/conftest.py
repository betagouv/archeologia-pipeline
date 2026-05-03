from __future__ import annotations

import sys
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PLUGIN_ROOT / "src"

for p in list(sys.path):
    if "archeologia-pipeline-lidar-processing" in p and p != str(SRC_ROOT):
        sys.path.remove(p)

sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture
def sample_config() -> dict:
    return {
        "app": {
            "files": {
                "data_mode": "ign_laz",
                "output_dir": "/tmp/output",
                "input_file": "/tmp/dalles.txt",
                "local_laz_dir": "/tmp/laz",
                "existing_mnt_dir": "/tmp/mnt",
                "existing_rvt_dir": "/tmp/rvt",
            }
        },
        "processing": {
            "mnt_resolution": 0.5,
            "density_resolution": 1.0,
            "tile_overlap": 5,
            "filter_expression": "Classification = 2",
            "products": {
                "MNT": True,
                "DENSITE": False,
                "M_HS": True,
                "SVF": True,
                "SLO": True,
                "LD": True,
                "SLRM": False,
                "VAT": False,
            },
            "max_workers": 4,
            "output_formats": {
                "jpg": {"LD": True, "VAT": False}
            },
            "pyramids": {"enabled": True, "levels": [2, 4, 8]},
        },
        "computer_vision": {
            "enabled": True,
            "runs": [{"model": "test_model", "target_rvt": "LD"}],
            "selected_model": "test_model",
            "target_rvt": "LD",
            "confidence_threshold": 0.3,
            "iou_threshold": 0.5,
            "generate_annotated_images": False,
            "generate_shapefiles": True,
            "models_dir": "/tmp/models",
            "sahi": {
                "slice_height": 750,
                "slice_width": 750,
                "overlap_ratio": 0.2,
            },
        },
        "rvt_params": {
            "mdh": {"num_directions": 16, "sun_elevation": 35},
            "svf": {"num_directions": 16, "radius": 10},
        },
    }


@pytest.fixture
def minimal_config() -> dict:
    return {}


@pytest.fixture
def cv_disabled_config(sample_config: dict) -> dict:
    config = sample_config.copy()
    config["computer_vision"] = {"enabled": False}
    return config
