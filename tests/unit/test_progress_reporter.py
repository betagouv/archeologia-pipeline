from __future__ import annotations

from app.progress_reporter import (
    NullProgressReporter,
    report_busy,
    report_stage_id,
)


class TestNullProgressReporter:
    def test_info_does_not_raise(self):
        reporter = NullProgressReporter()
        reporter.info("test message")

    def test_error_does_not_raise(self):
        reporter = NullProgressReporter()
        reporter.error("test error")

    def test_stage_does_not_raise(self):
        reporter = NullProgressReporter()
        reporter.stage("test stage")

    def test_progress_does_not_raise(self):
        reporter = NullProgressReporter()
        reporter.progress(50)

    def test_all_methods_accept_various_inputs(self):
        reporter = NullProgressReporter()
        reporter.info("")
        reporter.info("a" * 1000)
        reporter.error("")
        reporter.stage("")
        reporter.progress(0)
        reporter.progress(100)
        reporter.progress(-1)
        reporter.progress(999)

    def test_stage_id_and_busy_do_not_raise(self):
        reporter = NullProgressReporter()
        reporter.stage_id("products")
        reporter.busy(True)
        reporter.busy(False)


class _LegacyReporter:
    """Reporter sans stage_id/busy (cas legacy/test duck-typé)."""

    def __init__(self):
        self.calls = []


class _ModernReporter(_LegacyReporter):
    def stage_id(self, stage: str) -> None:
        self.calls.append(("stage_id", stage))

    def busy(self, active: bool) -> None:
        self.calls.append(("busy", active))


class TestDefensiveHelpers:
    def test_report_stage_id_noop_on_legacy(self):
        # Ne doit pas lever si la méthode n'existe pas.
        report_stage_id(_LegacyReporter(), "products")

    def test_report_busy_noop_on_legacy(self):
        report_busy(_LegacyReporter(), True)

    def test_report_stage_id_calls_modern(self):
        r = _ModernReporter()
        report_stage_id(r, "detection")
        assert r.calls == [("stage_id", "detection")]

    def test_report_busy_calls_modern_and_coerces_bool(self):
        r = _ModernReporter()
        report_busy(r, 1)
        assert r.calls == [("busy", True)]
