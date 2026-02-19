from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.runners.helpers import safe_float, log_section


class TestSafeFloat:
    def test_converts_valid_int(self):
        assert safe_float(42, 0.0) == 42.0

    def test_converts_valid_float(self):
        assert safe_float(3.14, 0.0) == 3.14

    def test_converts_valid_string(self):
        assert safe_float("2.5", 0.0) == 2.5

    def test_returns_default_for_none(self):
        assert safe_float(None, -1.0) == -1.0

    def test_returns_default_for_empty_string(self):
        assert safe_float("", 0.5) == 0.5

    def test_returns_default_for_non_numeric_string(self):
        assert safe_float("abc", 9.9) == 9.9

    def test_returns_default_for_dict(self):
        assert safe_float({}, 1.0) == 1.0

    def test_converts_bool_true(self):
        assert safe_float(True, 0.0) == 1.0

    def test_converts_bool_false(self):
        assert safe_float(False, 5.0) == 0.0

    def test_converts_negative(self):
        assert safe_float("-3.5", 0.0) == -3.5

    def test_converts_zero(self):
        assert safe_float(0, 99.0) == 0.0


class TestLogSection:
    def test_delegates_to_slog_when_present(self):
        slog = MagicMock()
        reporter = MagicMock()
        log_section("My Title", "download", slog=slog, reporter=reporter)
        slog.section.assert_called_once_with("My Title", "download")
        reporter.info.assert_not_called()

    def test_falls_back_to_reporter_when_no_slog(self):
        reporter = MagicMock()
        log_section("My Title", "process", slog=None, reporter=reporter)
        calls = [str(c) for c in reporter.info.call_args_list]
        assert any("My Title" in c for c in calls)

    def test_uses_default_icon_for_unknown_tag(self):
        reporter = MagicMock()
        log_section("Title", "unknown_tag", slog=None, reporter=reporter)
        calls = [str(c) for c in reporter.info.call_args_list]
        assert any("▶" in c for c in calls)
