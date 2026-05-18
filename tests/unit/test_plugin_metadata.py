"""Tests de :mod:`app.plugin_metadata`."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.plugin_metadata import get_plugin_version


@pytest.fixture(autouse=True)
def _clear_cache():
    get_plugin_version.cache_clear()
    yield
    get_plugin_version.cache_clear()


def _write_metadata(tmp_path: Path, body: str) -> Path:
    target = tmp_path / "metadata.txt"
    target.write_text(body, encoding="utf-8")
    return target


def test_returns_version_from_general_section(tmp_path: Path):
    target = _write_metadata(
        tmp_path,
        "[general]\nname=Plugin\nversion=1.2.3\n",
    )
    assert get_plugin_version(target) == "1.2.3"


def test_missing_file_returns_fallback(tmp_path: Path):
    assert get_plugin_version(tmp_path / "absent.txt") == "?"


def test_missing_general_section_returns_fallback(tmp_path: Path):
    target = _write_metadata(tmp_path, "[other]\nversion=9.9.9\n")
    assert get_plugin_version(target) == "?"
