from __future__ import annotations

from app.progress_reporter import NullProgressReporter


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
