from __future__ import annotations

from typing import Dict

from .base import ModeRunner
from .existing_mnt_runner import ExistingMntRunner
from .existing_rvt_runner import ExistingRvtRunner
from .ign_local_runner import IgnOrLocalRunner


_RUNNERS: Dict[str, ModeRunner] = {
    "ign_laz": IgnOrLocalRunner(),
    "local_laz": IgnOrLocalRunner(),
    "existing_mnt": ExistingMntRunner(),
    "existing_rvt": ExistingRvtRunner(),
}


def get_runner(mode: str) -> ModeRunner:
    m = str(mode or "").strip()
    if m in _RUNNERS:
        return _RUNNERS[m]
    raise ValueError(f"Unknown mode: {mode!r}")
