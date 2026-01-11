from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    temp_dir = Path(tempfile.mkdtemp(prefix="archeologia_test_"))
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_config_path() -> Path:
    return FIXTURES_DIR / "sample_config.json"


@pytest.fixture
def sample_config_dict(sample_config_path: Path) -> dict:
    with open(sample_config_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def config_with_output_dir(sample_config_dict: dict, temp_output_dir: Path) -> dict:
    config = sample_config_dict.copy()
    config["app"] = config.get("app", {}).copy()
    config["app"]["files"] = config["app"].get("files", {}).copy()
    config["app"]["files"]["output_dir"] = str(temp_output_dir)
    return config


@pytest.fixture
def mock_mnt_file(temp_output_dir: Path) -> Path:
    mnt_dir = temp_output_dir / "mnt"
    mnt_dir.mkdir(parents=True, exist_ok=True)
    mnt_file = mnt_dir / "test_mnt.tif"
    mnt_file.write_bytes(b"FAKE_TIF_DATA")
    return mnt_file


@pytest.fixture
def mock_rvt_dir(temp_output_dir: Path) -> Path:
    rvt_dir = temp_output_dir / "rvt" / "LD" / "tif"
    rvt_dir.mkdir(parents=True, exist_ok=True)
    rvt_file = rvt_dir / "test_rvt.tif"
    rvt_file.write_bytes(b"FAKE_RVT_TIF_DATA")
    return rvt_dir


@pytest.fixture
def mock_jpg_with_jgw(temp_output_dir: Path) -> Path:
    jpg_dir = temp_output_dir / "rvt" / "LD" / "jpg"
    jpg_dir.mkdir(parents=True, exist_ok=True)
    
    jpg_file = jpg_dir / "test_image.jpg"
    jpg_file.write_bytes(b"FAKE_JPG_DATA")
    
    jgw_file = jpg_dir / "test_image.jgw"
    jgw_content = "0.5\n0.0\n0.0\n-0.5\n100000.0\n6500000.0\n"
    jgw_file.write_text(jgw_content)
    
    return jpg_file
