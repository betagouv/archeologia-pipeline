from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from shutil import which
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class PipelineCancelled(Exception):
    pass


# Cache thread-safe pour les fichiers déjà validés
# Clé: chemin absolu du fichier + mtime (pour détecter les modifications)
_validation_cache: Set[str] = set()
_validation_cache_lock = threading.Lock()


def _get_cache_key(path: Path) -> str:
    """Génère une clé de cache basée sur le chemin et la date de modification."""
    try:
        mtime = path.stat().st_mtime
        return f"{path.resolve()}:{mtime}"
    except Exception:
        return str(path.resolve())


def clear_validation_cache() -> None:
    """Vide le cache de validation (utile entre les runs)."""
    with _validation_cache_lock:
        _validation_cache.clear()


def is_validated(path: Path) -> bool:
    """Vérifie si un fichier a déjà été validé."""
    key = _get_cache_key(path)
    with _validation_cache_lock:
        return key in _validation_cache


def mark_validated(path: Path) -> None:
    """Marque un fichier comme validé."""
    key = _get_cache_key(path)
    with _validation_cache_lock:
        _validation_cache.add(key)


def _pdal_subprocess_kwargs() -> Dict[str, Any]:
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


def run_pdal_command(cmd: List[str], *, timeout_s: Optional[int] = None) -> subprocess.CompletedProcess:
    return run_pdal_command_cancellable(cmd, timeout_s=timeout_s)


def run_pdal_command_cancellable(
    cmd: List[str],
    *,
    timeout_s: Optional[int] = None,
    cancel: Optional[Callable[[], bool]] = None,
    poll_interval_s: float = 0.2,
) -> subprocess.CompletedProcess:
    kwargs = _pdal_subprocess_kwargs()
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
                raise PipelineCancelled("Annulation demandée")

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


def validate_las_or_laz_with_pdal(
    path: Path, timeout_s: int = 300, use_cache: bool = True
) -> Tuple[bool, str]:
    # Vérifier le cache d'abord
    if use_cache and is_validated(path):
        return True, "ok (cached)"

    pdal = which("pdal")
    if not pdal:
        return False, "pdal executable not found in PATH"

    # Utiliser --metadata pour une validation rapide (--all est trop lent sur gros fichiers)
    cmd = [pdal, "info", "--metadata", str(path)]
    try:
        r = run_pdal_command(cmd, timeout_s=timeout_s)
    except Exception as e:
        return False, f"pdal info failed to run: {e!r}"

    out = (r.stdout or "").strip()
    err = (r.stderr or "").strip()

    if r.returncode != 0:
        msg = err or out or f"pdal returned code {r.returncode}"
        return False, msg

    # Vérifier aussi les warnings/erreurs dans stderr (EVLR corrompus)
    if err and ("EVLR" in err or "End of file" in err or "Couldn't read" in err):
        return False, err

    # Marquer comme validé dans le cache
    if use_cache:
        mark_validated(path)

    return True, "ok"


def require_valid_las_or_laz_with_pdal(path: Path, timeout_s: int = 60) -> None:
    ok, msg = validate_las_or_laz_with_pdal(path, timeout_s=timeout_s)
    if not ok:
        raise IOError(msg)


def get_laz_bounds(path: Path, timeout_s: int = 60) -> Optional[Tuple[float, float, float, float]]:
    """
    Récupère les bounds (xmin, ymin, xmax, ymax) d'un fichier LAZ via PDAL.
    Retourne None si impossible de lire les bounds.
    """
    import json
    
    pdal = which("pdal")
    if not pdal:
        return None

    cmd = [pdal, "info", "--metadata", str(path)]
    try:
        r = run_pdal_command(cmd, timeout_s=timeout_s)
    except Exception:
        return None

    if r.returncode != 0:
        return None

    try:
        data = json.loads(r.stdout or "{}")
        metadata = data.get("metadata", {})
        
        minx = metadata.get("minx")
        miny = metadata.get("miny")
        maxx = metadata.get("maxx")
        maxy = metadata.get("maxy")
        
        if all(v is not None for v in [minx, miny, maxx, maxy]):
            return (float(minx), float(miny), float(maxx), float(maxy))
    except Exception:
        pass
    
    return None
