from __future__ import annotations

import pytest

from app.runners.registry import get_runner
from app.runners.ign_local_runner import IgnOrLocalRunner
from app.runners.existing_mnt_runner import ExistingMntRunner
from app.runners.existing_rvt_runner import ExistingRvtRunner


class TestGetRunner:
    def test_returns_ign_local_runner_for_ign_laz(self):
        runner = get_runner("ign_laz")
        assert isinstance(runner, IgnOrLocalRunner)

    def test_returns_ign_local_runner_for_local_laz(self):
        runner = get_runner("local_laz")
        assert isinstance(runner, IgnOrLocalRunner)

    def test_returns_existing_mnt_runner_for_existing_mnt(self):
        runner = get_runner("existing_mnt")
        assert isinstance(runner, ExistingMntRunner)

    def test_returns_existing_rvt_runner_for_existing_rvt(self):
        runner = get_runner("existing_rvt")
        assert isinstance(runner, ExistingRvtRunner)

    def test_raises_for_unknown_mode(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            get_runner("unknown_mode")

    def test_raises_for_empty_mode(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            get_runner("")

    def test_raises_for_none_mode(self):
        with pytest.raises((ValueError, TypeError)):
            get_runner(None)

    def test_mode_is_case_sensitive(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            get_runner("IGN_LAZ")
