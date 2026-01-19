from __future__ import annotations

import os
import re
import shutil
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from qgis.PyQt.QtCore import QSettings
    HAS_QGIS = True
except ImportError:
    HAS_QGIS = False

from .coords_fallback import build_sorted_records_with_fallback


from .pdal_validation import validate_las_or_laz_with_pdal


def _extract_real_url(url: str) -> str:
    """Extract the real URL from various wrapper/protection services.
    
    Handles:
    - Proofpoint urldefense.com (v2, v3)
    - Google Safe Browsing redirects
    - Microsoft SafeLinks
    - Generic URL wrappers with embedded URLs
    - Direct URLs (returned as-is)
    """
    url = url.strip()
    
    # Proofpoint urldefense v3: https://urldefense.com/v3/__<real_url>__;...
    match = re.search(r"urldefense\.com/v3/__(.+?)(?:__;|$)", url)
    if match:
        return match.group(1)
    
    # Proofpoint urldefense v2: https://urldefense.proofpoint.com/v2/url?u=<encoded>&...
    match = re.search(r"urldefense\.proofpoint\.com/v2/url\?u=([^&]+)", url)
    if match:
        decoded = match.group(1).replace("-", "%").replace("_", "/")
        return urllib.parse.unquote(decoded)
    
    # Microsoft SafeLinks: https://...safelinks.protection.outlook.com/?url=<encoded>&...
    match = re.search(r"safelinks\.protection\.outlook\.com/?\?url=([^&]+)", url)
    if match:
        return urllib.parse.unquote(match.group(1))
    
    # Google redirect: https://www.google.com/url?q=<encoded>&...
    match = re.search(r"google\.com/url\?[^&]*q=([^&]+)", url)
    if match:
        return urllib.parse.unquote(match.group(1))
    
    # Generic: try to find an embedded https://data.geopf.fr or similar IGN URL
    match = re.search(r"(https?://data\.geopf\.fr/[^\s\"'<>]+\.(?:laz|las|copc\.laz))", url, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Fallback: look for any embedded https:// URL ending in .laz or .las
    match = re.search(r"(https?://[^\s\"'<>]+\.(?:laz|las|copc\.laz))", url, re.IGNORECASE)
    if match:
        return match.group(1)
    
    return url


LogFn = Callable[[str], None]
ProgressFn = Callable[[int], None]
StageFn = Callable[[str], None]
CancelFn = Callable[[], bool]


@dataclass(frozen=True)
class IgnDownloadResult:
    dalles_dir: Path
    sorted_list_file: Path
    total: int
    downloaded: int
    skipped_existing: int


def _is_valid_with_pdal(path: Path) -> Tuple[bool, str]:
    ok, msg = validate_las_or_laz_with_pdal(path)
    if ok:
        return True, ""
    return False, msg


def _default_log(_: str) -> None:
    return


def _default_progress(_: int) -> None:
    return


def _default_stage(_: str) -> None:
    return


def _default_cancel() -> bool:
    return False


def _get_qgis_proxy_settings() -> Optional[Dict[str, str]]:
    """Récupère les paramètres proxy depuis les settings QGIS.
    
    Retourne:
    - Dict avec proxy si configuré dans QGIS: {'http': 'http://host:port', 'https': 'http://host:port'}
    - None si pas de proxy QGIS (utiliser le proxy système)
    """
    if not HAS_QGIS:
        return None
    
    try:
        settings = QSettings()
        proxy_enabled = settings.value("proxy/proxyEnabled", False, type=bool)
        if not proxy_enabled:
            return None
        
        proxy_host = settings.value("proxy/proxyHost", "", type=str)
        proxy_port = settings.value("proxy/proxyPort", "", type=str)
        proxy_user = settings.value("proxy/proxyUser", "", type=str)
        proxy_password = settings.value("proxy/proxyPassword", "", type=str)
        
        if not proxy_host:
            return None
        
        # Construire l'URL du proxy
        if proxy_user and proxy_password:
            proxy_url = f"http://{proxy_user}:{proxy_password}@{proxy_host}"
        elif proxy_user:
            proxy_url = f"http://{proxy_user}@{proxy_host}"
        else:
            proxy_url = f"http://{proxy_host}"
        
        if proxy_port:
            proxy_url = f"{proxy_url}:{proxy_port}"
        
        return {"http": proxy_url, "https": proxy_url}
    except Exception:
        return None


def _is_proxy_reachable(proxy_url: str, timeout: float = 2.0) -> bool:
    """Vérifie si le proxy est accessible (résolution DNS + connexion TCP)."""
    import socket
    try:
        # Extraire host:port de l'URL proxy
        # Format: http://[user:pass@]host:port
        parsed = urllib.parse.urlparse(proxy_url)
        host = parsed.hostname
        port = parsed.port or 8080
        if not host:
            return False
        # Test de connexion TCP rapide
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        sock.close()
        return True
    except Exception:
        return False


def _get_proxy_config(log: LogFn = _default_log) -> Dict[str, str]:
    """Récupère la configuration proxy (QGIS ou système).
    
    Priorité: proxy QGIS (si accessible) > proxy système > pas de proxy
    """
    qgis_proxy = _get_qgis_proxy_settings()
    if qgis_proxy:
        proxy_url = qgis_proxy.get("http", "")
        # Masquer le mot de passe dans les logs
        if "@" in proxy_url:
            import re
            masked = re.sub(r"://([^:]+):([^@]+)@", r"://\1:****@", proxy_url)
        else:
            masked = proxy_url
        
        # Vérifier si le proxy est accessible
        if _is_proxy_reachable(proxy_url):
            log(f"⚙️ Proxy QGIS configuré et accessible: {masked}")
            return qgis_proxy
        else:
            log(f"⚙️ Proxy QGIS configuré mais non accessible: {masked} (connexion directe)")
    
    # Fallback: proxy système
    system_proxy = urllib.request.getproxies()
    if system_proxy:
        log(f"⚙️ Proxy système: {system_proxy}")
        return system_proxy
    
    log("⚙️ Connexion directe (aucun proxy)")
    return {}


def parse_ign_input_file(input_file: Path, sorted_output_file: Path, log: LogFn = _default_log) -> List[Tuple[str, str]]:
    log("Début du tri des fichiers")
    if not input_file.exists():
        raise FileNotFoundError(f"Fichier d'entrée non trouvé: {input_file}")

    # On ne force pas l'extraction des coords ici: elles peuvent être absentes.
    # Le tri final (fichier_tri.txt) sera fait après téléchargement, avec fallback PDAL.
    raw_items: List[Tuple[str, str]] = []

    with input_file.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if "," in line:
                filename, url = line.split(",", 1)
                filename = filename.strip()
                url = url.strip()
            else:
                url = line
                # Extract real URL from wrappers (urldefense, SafeLinks, etc.)
                url = _extract_real_url(url)
                try:
                    parsed = urllib.parse.urlparse(url)
                    filename = Path(parsed.path).name
                except Exception:
                    filename = ""

            if not filename:
                log(f"Ligne ignorée (nom de fichier introuvable): {line}")
                continue
            raw_items.append((filename, url))

    # Compat: on ne génère plus le fichier trié ici (le tri dépend du fallback post-download)
    return raw_items


def _is_local_url(url: str) -> Optional[Path]:
    try:
        p = url
        if url.startswith("file://"):
            p = url[7:]
        local = Path(p)
        if local.exists() and local.is_file():
            return local
    except Exception:
        return None
    return None


def download_one(
    url: str,
    filename: str,
    dalles_dir: Path,
    log: LogFn = _default_log,
    cancel: CancelFn = _default_cancel,
    timeout_s: int = 300,
    chunk_size: int = 8192,
    max_retries: int = 3,
    retry_delay_s: float = 5.0,
    proxies: Optional[Dict[str, str]] = None,
) -> Tuple[bool, bool]:
    """Returns (success, skipped_existing)."""

    if cancel():
        return False, False

    dalles_dir.mkdir(parents=True, exist_ok=True)
    dest = dalles_dir / filename

    if dest.exists():
        log(f"🔍 Validation PDAL de {filename}...")
        ok, msg = _is_valid_with_pdal(dest)
        if ok:
            log(f"✅ {filename} déjà téléchargé (validation OK)")
            return True, True
        log(f"⚠️ Fichier existant invalide via PDAL ({filename}) -> suppression et nouveau téléchargement")
        if msg:
            log(f"PDAL: {msg}")
        try:
            dest.unlink()
        except Exception:
            pass

    local = _is_local_url(url)
    if local is not None:
        log(f"📥 Copie du fichier local {local} vers {dest}...")
        shutil.copy2(str(local), str(dest))
        log(f"🔍 Validation PDAL de {filename}...")
        ok, msg = _is_valid_with_pdal(dest)
        if not ok:
            log(f"❌ Fichier copié mais invalide via PDAL: {filename}")
            if msg:
                log(f"PDAL: {msg}")
            try:
                dest.unlink()
            except Exception:
                pass
            return False, False
        return True, False

    log(f"📥 Téléchargement de {filename}...")

    if proxies is None:
        proxies = {}

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        if cancel():
            return False, False

        try:
            if HAS_REQUESTS:
                # Utiliser requests qui gère correctement les proxies HTTPS
                resp = requests.get(
                    url,
                    headers={"User-Agent": "QGIS-ArcheologiaPipeline/1.0"},
                    proxies=proxies if proxies else None,
                    timeout=timeout_s,
                    stream=True,
                )
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if cancel():
                            f.close()
                            if dest.exists():
                                dest.unlink()
                            return False, False
                        if chunk:
                            f.write(chunk)
            else:
                # Fallback urllib (peut ne pas fonctionner avec proxy HTTPS)
                req = urllib.request.Request(url, headers={"User-Agent": "QGIS-ArcheologiaPipeline/1.0"})
                proxy_handler = urllib.request.ProxyHandler(proxies if proxies else None)
                opener = urllib.request.build_opener(proxy_handler)
                with opener.open(req, timeout=timeout_s) as r:
                    with open(dest, "wb") as f:
                        while True:
                            if cancel():
                                f.close()
                                if dest.exists():
                                    dest.unlink()
                                return False, False
                            chunk = r.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)

            log(f"🔍 Validation PDAL de {filename}...")
            ok, msg = _is_valid_with_pdal(dest)
            if not ok:
                log(f"⚠️ Fichier invalide via PDAL après téléchargement ({filename}) -> suppression")
                if msg:
                    log(f"PDAL: {msg}")
                try:
                    dest.unlink()
                except Exception:
                    pass
                raise IOError(f"pdal-invalid: {msg}")

            return True, False
        except Exception as e:
            last_err = e
            log(f"Tentative {attempt}/{max_retries} échouée pour {filename}: {e}")
            if attempt < max_retries:
                log(f"Nouvelle tentative dans {int(retry_delay_s)} secondes...")
                time.sleep(retry_delay_s)

    if last_err is not None:
        log(f"Échec définitif du téléchargement pour {filename}: {last_err}")
    return False, False


@dataclass
class _DownloadTask:
    """Tâche de téléchargement pour un fichier."""
    index: int
    filename: str
    url: str


@dataclass
class _DownloadResult:
    """Résultat du téléchargement d'un fichier."""
    index: int
    filename: str
    success: bool
    skipped: bool
    error: Optional[str] = None


def download_ign_dalles(
    *,
    input_file: Path,
    output_dir: Path,
    log: LogFn = _default_log,
    progress: ProgressFn = _default_progress,
    stage: StageFn = _default_stage,
    cancel: CancelFn = _default_cancel,
    max_workers: Optional[int] = None,
) -> IgnDownloadResult:
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    dalles_dir = output_dir / "dalles"
    sorted_list = output_dir / "fichier_tri.txt"

    stage("Tri des fichiers")
    progress(0)
    file_list = parse_ign_input_file(input_file, sorted_list, log=log)

    total = len(file_list)
    if total == 0:
        raise ValueError("Aucun fichier à télécharger (liste vide après parsing)")

    stage("Téléchargement")

    # Détection du proxy une seule fois pour tous les téléchargements
    proxies = _get_proxy_config(log=log)

    # Configuration parallélisation
    if max_workers is None:
        max_workers = min(4, max(1, os.cpu_count() or 1))
    
    log(f"Téléchargement parallèle: {max_workers} worker(s) pour {total} fichier(s)")

    # Lock pour synchroniser les logs et le compteur
    log_lock = threading.Lock()
    completed_count = [0]
    downloaded = [0]
    skipped = [0]
    first_error: List[Optional[str]] = [None]

    def thread_safe_log(msg: str) -> None:
        with log_lock:
            log(msg)

    def update_progress_and_counts(result: _DownloadResult) -> None:
        with log_lock:
            completed_count[0] += 1
            if result.success:
                if result.skipped:
                    skipped[0] += 1
                else:
                    downloaded[0] += 1
            elif first_error[0] is None:
                first_error[0] = result.error
            pct = int(round(100.0 * completed_count[0] / max(1, total)))
            progress(pct)

    # Préparer les tâches
    tasks = [
        _DownloadTask(index=idx, filename=filename, url=url)
        for idx, (filename, url) in enumerate(file_list, start=1)
    ]

    # Exécution parallèle
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                _download_task_worker,
                task,
                dalles_dir,
                thread_safe_log,
                cancel,
                proxies,
            ): task
            for task in tasks
        }

        for future in as_completed(future_to_task):
            if cancel():
                executor.shutdown(wait=False, cancel_futures=True)
                break

            try:
                result = future.result()
                update_progress_and_counts(result)
            except Exception as e:
                task = future_to_task[future]
                thread_safe_log(f"Exception téléchargement {task.filename}: {e}")
                with log_lock:
                    if first_error[0] is None:
                        first_error[0] = str(e)

    # Vérifier s'il y a eu une erreur
    if first_error[0] is not None:
        raise RuntimeError(f"Échec du téléchargement: {first_error[0]}")

    # Tri final + fallback coords (Option B): si coords absentes, on infère via PDAL et on renomme le fichier.
    stage("Tri des fichiers (post-téléchargement)")

    records = build_sorted_records_with_fallback(file_list=file_list, dalles_dir=dalles_dir, cancel=cancel, log=log)
    if not records:
        raise ValueError("Impossible de déterminer les coordonnées des dalles (nom de fichier + fallback PDAL)")

    sorted_list.parent.mkdir(parents=True, exist_ok=True)
    with sorted_list.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(f"{rec.filename},{rec.url}\n")

    progress(100)

    return IgnDownloadResult(
        dalles_dir=dalles_dir,
        sorted_list_file=sorted_list,
        total=total,
        downloaded=downloaded[0],
        skipped_existing=skipped[0],
    )


def _download_task_worker(
    task: _DownloadTask,
    dalles_dir: Path,
    log: LogFn,
    cancel: CancelFn,
    proxies: Dict[str, str],
) -> _DownloadResult:
    """Worker pour télécharger un fichier. Thread-safe (HTTP requests)."""
    try:
        ok, was_skipped = download_one(
            url=task.url,
            filename=task.filename,
            dalles_dir=dalles_dir,
            log=log,
            cancel=cancel,
            proxies=proxies,
        )
        return _DownloadResult(
            index=task.index,
            filename=task.filename,
            success=ok,
            skipped=was_skipped,
            error=None if ok else f"Échec téléchargement {task.filename}",
        )
    except Exception as e:
        return _DownloadResult(
            index=task.index,
            filename=task.filename,
            success=False,
            skipped=False,
            error=str(e),
        )
