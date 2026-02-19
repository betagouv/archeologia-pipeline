from __future__ import annotations

import os
import subprocess
from typing import Any, Dict


def subprocess_kwargs_no_window() -> Dict[str, Any]:
    """Retourne les kwargs subprocess pour masquer la fenêtre console sur Windows."""
    if os.name != "nt":
        return {}
    kwargs: Dict[str, Any] = {"creationflags": subprocess.CREATE_NO_WINDOW}
    try:
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0
        kwargs["startupinfo"] = si
    except Exception:
        pass
    return kwargs
