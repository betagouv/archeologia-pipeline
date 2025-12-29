from __future__ import annotations

import subprocess
from pathlib import Path
from shutil import which
from typing import Optional, Tuple


def validate_las_or_laz_with_pdal(path: Path, timeout_s: int = 60) -> Tuple[bool, str]:
    """Validate a LAS/LAZ/COPC-LAZ using `pdal info --summary`.

    Returns (ok, message).

    Notes:
    - This checks real readability/decoding via PDAL, catching cases where the LAS header is present
      but the LAZ/COPC payload (EVLR/chunks/index) is truncated or corrupted.
    - If `pdal` is not available on PATH, this returns (False, "...").
    """

    pdal = which("pdal")
    if not pdal:
        return False, "pdal executable not found in PATH"

    cmd = [pdal, "info", str(path), "--summary"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except Exception as e:
        return False, f"pdal info failed to run: {e!r}"

    out = (r.stdout or "").strip()
    err = (r.stderr or "").strip()

    if r.returncode != 0:
        msg = err or out or f"pdal returned code {r.returncode}"
        return False, msg

    return True, "ok"


def require_valid_las_or_laz_with_pdal(path: Path, timeout_s: int = 60) -> None:
    ok, msg = validate_las_or_laz_with_pdal(path, timeout_s=timeout_s)
    if not ok:
        raise IOError(msg)
