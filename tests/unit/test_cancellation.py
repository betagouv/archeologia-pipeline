from __future__ import annotations

import pytest

from pipeline.cancellation import PipelineCancelled, check_cancelled


class TestPipelineCancelled:
    def test_is_exception(self):
        assert issubclass(PipelineCancelled, Exception)

    def test_canonical_identity_with_pdal_validation(self):
        # pdal_validation réexporte la même classe (compat ascendante).
        from pipeline.ign.pdal_validation import PipelineCancelled as PC2
        assert PC2 is PipelineCancelled


class TestCheckCancelled:
    def test_raises_when_cancel_true(self):
        with pytest.raises(PipelineCancelled):
            check_cancelled(lambda: True)

    def test_noop_when_cancel_false(self):
        check_cancelled(lambda: False)  # ne doit rien lever

    def test_noop_when_cancel_none(self):
        check_cancelled(None)  # ne doit rien lever
