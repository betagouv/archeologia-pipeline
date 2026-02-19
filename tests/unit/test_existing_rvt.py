from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from pipeline.modes.existing_rvt import _cleanup_orphans


class TestCleanupOrphans:
    def test_removes_empty_files(self, tmp_path):
        kept = tmp_path / "kept.tif"
        kept.write_bytes(b"DATA")
        empty = tmp_path / "orphan.tif"
        empty.write_bytes(b"")

        _cleanup_orphans(tmp_path, "*.tif", {kept.name})

        assert kept.exists()
        assert not empty.exists()

    def test_removes_numeric_stem_files(self, tmp_path):
        kept = tmp_path / "real_tile.tif"
        kept.write_bytes(b"DATA")
        numeric = tmp_path / "12345.tif"
        numeric.write_bytes(b"DATA")

        _cleanup_orphans(tmp_path, "*.tif", {kept.name})

        assert kept.exists()
        assert not numeric.exists()

    def test_keeps_files_in_kept_names(self, tmp_path):
        f1 = tmp_path / "001.tif"
        f1.write_bytes(b"DATA")

        _cleanup_orphans(tmp_path, "*.tif", {f1.name})

        assert f1.exists()

    def test_does_nothing_for_none_directory(self):
        _cleanup_orphans(None, "*.tif", set())

    def test_does_nothing_for_nonexistent_directory(self):
        _cleanup_orphans(Path("/nonexistent/dir"), "*.tif", set())

    def test_ignores_non_matching_glob(self, tmp_path):
        f = tmp_path / "12345.jpg"
        f.write_bytes(b"DATA")

        _cleanup_orphans(tmp_path, "*.tif", set())

        assert f.exists()

    def test_keeps_non_empty_non_numeric_files(self, tmp_path):
        f = tmp_path / "real_name.tif"
        f.write_bytes(b"DATA")

        _cleanup_orphans(tmp_path, "*.tif", set())

        assert f.exists()
