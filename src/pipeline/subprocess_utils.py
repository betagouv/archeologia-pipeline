from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .cancellation import PipelineCancelled


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


def run_subprocess_cancellable(
    cmd: List[str],
    *,
    cancel: Optional[Callable[[], bool]] = None,
    poll_interval_s: float = 0.2,
    timeout_s: Optional[float] = None,
    output_path: Optional[Path] = None,
) -> subprocess.CompletedProcess:
    """Comme ``subprocess.run(capture_output=True, text=True)`` mais sonde ``cancel``.

    Retourne le ``CompletedProcess`` en cas de succès. **Lève** :class:`PipelineCancelled`
    si une annulation est demandée (le process est alors terminé/tué) — idiome
    d'annulation unique du pipeline. Lève ``subprocess.TimeoutExpired`` si ``timeout_s``
    est dépassé.

    ``output_path`` (optionnel) : fichier de sortie écrit par la commande ; il est
    supprimé s'il existe lorsqu'on tue le process pour annulation, afin qu'un fichier
    partiel/tronqué ne soit pas pris pour « déjà fait » par les caches de reprise.
    """
    kwargs = subprocess_kwargs_no_window()
    creationflags = kwargs.pop("creationflags", 0)
    startupinfo = kwargs.pop("startupinfo", None)

    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creationflags,
        startupinfo=startupinfo,
    )
    start = time.time()

    try:
        while True:
            if cancel is not None and cancel():
                try:
                    p.terminate()
                except Exception:
                    pass
                try:
                    p.wait(timeout=2)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass
                if output_path is not None:
                    try:
                        Path(output_path).unlink(missing_ok=True)
                    except Exception:
                        pass
                raise PipelineCancelled()

            try:
                stdout, stderr = p.communicate(timeout=poll_interval_s)
                return subprocess.CompletedProcess(cmd, p.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                if timeout_s is not None and (time.time() - start) >= float(timeout_s):
                    try:
                        p.terminate()
                    except Exception:
                        pass
                    raise
    finally:
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass
