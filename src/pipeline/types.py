"""
Type aliases partagés par tous les modules du pipeline.
"""
from __future__ import annotations

from typing import Callable


LogFn = Callable[[str], None]
CancelCheckFn = Callable[[], bool]
CancelFn = Callable[[], bool]
ProgressFn = Callable[[int], None]
