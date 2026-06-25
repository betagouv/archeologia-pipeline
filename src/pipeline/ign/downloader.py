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

from ..cancellation import PipelineCancelled
from .coords_fallback import build_sorted_records_with_fallback

try:  # fallback standalone (tests : src/ sur le path)
    from ...app.services.proxy_config import build_proxy_url, is_pac_like
except ImportError:  # pragma: no cover
    from app.services.proxy_config import build_proxy_url, is_pac_like


from .pdal_validation import validate_las_or_laz_with_pdal, validate_laz_deep


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


# Domaines autorisés pour le téléchargement de dalles (AUDIT NET-02) :
# diffusion IGN actuelle (Géoplateforme) + anciens buckets LiDAR HD.
ALLOWED_DOWNLOAD_HOST_SUFFIXES = (
    "data.geopf.fr",
    "geopf.fr",
    "ign.fr",
    "storage.sbg.cloud.ovh.net",
    "storage.gra.cloud.ovh.net",
)

# Cap de taille par dalle (AUDIT NET-01) : très au-dessus d'une dalle LiDAR HD
# réelle (50-400 Mo) — protège le disque d'un flux anormal (×4 workers), la
# validation PDAL n'intervenant qu'APRÈS écriture complète.
MAX_DOWNLOAD_SIZE_MB = 2048


def validate_download_url(url: str) -> Tuple[bool, str, str]:
    """Valide une URL de dalle : ``(ok, url_normalisée, raison_du_refus)``.

    ``_extract_real_url`` déballe des liens venant de MAILS
    (Proofpoint/SafeLinks) : sans liste blanche, une URL piégée déclenchait
    une requête sortante vers un hôte arbitraire (SSRF — AUDIT NET-02).
    https est imposé (une URL http vers un hôte autorisé est upgradée).
    """
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        return False, url, f"schéma non supporté ({parsed.scheme or 'aucun'})"
    host = (parsed.hostname or "").lower()
    if not any(
        host == suffix or host.endswith("." + suffix)
        for suffix in ALLOWED_DOWNLOAD_HOST_SUFFIXES
    ):
        return False, url, f"hôte non autorisé ({host or 'vide'})"
    if parsed.scheme == "http":
        url = urllib.parse.urlunparse(parsed._replace(scheme="https"))
    return True, url, ""


from ..types import LogFn, ProgressFn, CancelFn
StageFn = Callable[[str], None]


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


def _get_qgis_proxy_settings(log: LogFn = _default_log) -> Optional[Dict[str, str]]:
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

        # URL d'auto-configuration (PAC/WPAD) : ce n'est PAS un proxy direct
        # utilisable par requests (le chemin ``/xxx.pac`` est ignoré). On
        # l'ignore explicitement avec un message actionnable au lieu de
        # fabriquer une URL cassée — c'était la cause du bug
        # ``http://http://host/proxy.pac`` → ``Failed to resolve 'http'``.
        if is_pac_like(proxy_host):
            log(
                f"⚠️ Proxy QGIS ignoré : « {proxy_host} » est une URL "
                "d'auto-configuration (PAC), non prise en charge en "
                "téléchargement direct. Définissez un proxy hôte:port explicite "
                "(QGIS → Préférences → Réseau) ou les variables "
                "HTTP_PROXY / HTTPS_PROXY. Repli sur le proxy système…"
            )
            return None

        # ``build_proxy_url`` retire tout schéma déjà présent avant de préfixer
        # ``http://`` (sinon double ``http://`` sur un host déjà schémé).
        proxy_url = build_proxy_url(proxy_host, proxy_port, proxy_user, proxy_password)
        return {"http": proxy_url, "https": proxy_url}
    except Exception:
        return None


def _get_proxy_config(log: LogFn = _default_log) -> Dict[str, str]:
    """Récupère la configuration proxy (QGIS ou système).

    Priorité : proxy QGIS (Préférences → Réseau) > proxy système (variables
    d'environnement ``HTTP_PROXY`` / ``HTTPS_PROXY``…) > connexion directe.

    On NE teste PAS l'accessibilité du proxy avant de l'utiliser : un court
    test TCP (timeout 2 s) échouait à tort sur certains proxys d'entreprise
    (réponse lente, ou refus des connexions brutes), ce qui faisait
    silencieusement retomber le plugin en accès *direct* — lequel expire
    ensuite sur ``data.geopf.fr`` quand le réseau sortant est filtré. Si
    l'utilisateur a configuré un proxy dans QGIS, on l'utilise et on laisse
    ``requests`` gérer les échecs éventuels (avec les retries du download).
    """
    qgis_proxy = _get_qgis_proxy_settings(log=log)
    if qgis_proxy:
        proxy_url = qgis_proxy.get("http", "")
        # Masquer le mot de passe dans les logs
        if "@" in proxy_url:
            masked = re.sub(r"://([^:]+):([^@]+)@", r"://\1:****@", proxy_url)
        else:
            masked = proxy_url
        log(f"⚙️ Proxy QGIS: {masked}")
        return qgis_proxy

    # Fallback : proxy système (variables d'environnement HTTP(S)_PROXY, etc.)
    system_proxy = urllib.request.getproxies()
    if system_proxy:
        log(f"⚙️ Proxy système (variables d'environnement): {system_proxy}")
        return system_proxy

    log("⚙️ Connexion directe (aucun proxy configuré dans QGIS)")
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

            # Ignorer les lignes qui ne contiennent pas d'URL valide (ex: en-têtes CSV)
            if not url.startswith(("http://", "https://")):
                log(f"Ligne ignorée (pas une URL valide): {line}")
                continue

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
        log(f"🔍 Validation PDAL rapide de {filename}...")
        ok, msg = _is_valid_with_pdal(dest)
        if not ok:
            log(f"⚠️ Fichier existant invalide via PDAL ({filename}) -> suppression et nouveau téléchargement")
            if msg:
                log(f"PDAL: {msg}")
            try:
                dest.unlink()
            except Exception:
                pass
        else:
            # Validation profonde pour détecter les fichiers tronqués
            log(f"🔍 Validation PDAL profonde de {filename}...")
            ok_deep, msg_deep = validate_laz_deep(dest)
            if ok_deep:
                log(f"✅ {filename} déjà téléchargé (validation complète OK)")
                return True, True
            log(f"⚠️ Fichier existant tronqué/corrompu ({filename}) -> suppression et nouveau téléchargement")
            if msg_deep:
                log(f"PDAL: {msg_deep}")
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

    # Liste blanche d'hôtes + https imposé (AUDIT NET-02) — coupe AVANT toute
    # requête sortante une URL piégée déballée d'un mail.
    ok_url, url, refuse_reason = validate_download_url(_extract_real_url(url))
    if not ok_url:
        log(
            f"❌ URL refusée pour {filename}: {refuse_reason} — seuls les "
            f"domaines IGN connus sont autorisés ({', '.join(ALLOWED_DOWNLOAD_HOST_SUFFIXES[:2])}…)"
        )
        return False, False

    log(f"📥 Téléchargement de {filename}...")

    if proxies is None:
        proxies = {}

    max_bytes = int(MAX_DOWNLOAD_SIZE_MB * 1024 * 1024)

    def _declared_too_big(headers) -> bool:
        try:
            declared = int(headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            declared = 0
        if declared > max_bytes:
            log(
                f"❌ {filename}: taille annoncée {declared / 1e6:.0f} Mo > cap "
                f"de {MAX_DOWNLOAD_SIZE_MB} Mo — abandonné (AUDIT NET-01)"
            )
            return True
        return False

    def _over_cap(written: int, f) -> bool:
        if written <= max_bytes:
            return False
        f.close()
        if dest.exists():
            dest.unlink()
        log(
            f"❌ {filename}: dépasse le cap de {MAX_DOWNLOAD_SIZE_MB} Mo en "
            "cours de téléchargement — abandonné (anti-saturation disque)"
        )
        return True

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
                if _declared_too_big(resp.headers):
                    return False, False  # définitif : pas de retry
                written = 0
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if cancel():
                            f.close()
                            if dest.exists():
                                dest.unlink()
                            return False, False
                        if chunk:
                            written += len(chunk)
                            if _over_cap(written, f):
                                return False, False  # définitif : pas de retry
                            f.write(chunk)
            else:
                # Fallback urllib (peut ne pas fonctionner avec proxy HTTPS)
                req = urllib.request.Request(url, headers={"User-Agent": "QGIS-ArcheologiaPipeline/1.0"})
                proxy_handler = urllib.request.ProxyHandler(proxies if proxies else None)
                opener = urllib.request.build_opener(proxy_handler)
                with opener.open(req, timeout=timeout_s) as r:
                    if _declared_too_big(getattr(r, "headers", {})):
                        return False, False  # définitif : pas de retry
                    written = 0
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
                            written += len(chunk)
                            if _over_cap(written, f):
                                return False, False  # définitif : pas de retry
                            f.write(chunk)

            log(f"🔍 Validation PDAL rapide de {filename}...")
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

            # Validation profonde pour détecter les fichiers tronqués
            log(f"🔍 Validation PDAL profonde de {filename} (lecture complète)...")
            ok_deep, msg_deep = validate_laz_deep(dest)
            if not ok_deep:
                log(f"⚠️ Fichier tronqué/corrompu détecté ({filename}) -> suppression")
                if msg_deep:
                    log(f"PDAL: {msg_deep}")
                try:
                    dest.unlink()
                except Exception:
                    pass
                raise IOError(f"pdal-deep-invalid: {msg_deep}")

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
    on_tile_done: Optional[Callable[[int, int, str, bool], None]] = None,
) -> IgnDownloadResult:
    """Télécharge les dalles IGN listées dans ``input_file``.

    ``on_tile_done`` (optionnel) est invoqué après chaque dalle traitée
    avec ``(completed_index_1based, total, filename, success)`` — le caller s'en
    sert pour remonter une sous-progression à l'utilisateur (ligne
    réécrite dans le journal). Les téléchargements parallèles ne
    garantissent pas l'ordre des appels (``completed_index`` reflète
    l'ordre d'arrivée, pas l'ordre des tâches).
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    from ..output_paths import dalles_dir as _dalles_dir
    dalles_dir = _dalles_dir(output_dir)
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
    failed = [0]

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
            else:
                failed[0] += 1
            pct = int(round(100.0 * completed_count[0] / max(1, total)))
            progress(pct)
            current = completed_count[0]
        if on_tile_done is not None:
            try:
                on_tile_done(current, total, result.filename, result.success)
            except Exception:
                pass

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
                thread_safe_log(f"⚠️ Fichier ignoré (téléchargement échoué) {task.filename}: {e}")
                with log_lock:
                    failed[0] += 1
                    completed_count[0] += 1
                    pct = int(round(100.0 * completed_count[0] / max(1, total)))
                    progress(pct)
                    current = completed_count[0]
                if on_tile_done is not None:
                    try:
                        on_tile_done(current, total, task.filename, False)
                    except Exception:
                        pass

    # Log des fichiers échoués (mais on continue)
    if failed[0] > 0:
        log(f"⚠️ {failed[0]} fichier(s) ignoré(s) (téléchargement échoué ou invalide)")

    # Annulation pendant le téléchargement : sortie PROPRE via l'exception
    # canonique. Sans cela, le tri/fallback ci-dessous (court-circuité par
    # cancel) renvoie une liste vide → ValueError « Impossible de déterminer
    # les coordonnées des dalles », traceback trompeur pour l'utilisateur
    # qui vient juste de cliquer Annuler (AUDIT v2 ROB-17).
    if cancel():
        raise PipelineCancelled()

    # Aucune dalle aboutie (réseau coupé / proxy invalide) : on s'arrête ICI avec
    # un message actionnable, au lieu de laisser l'échec remonter en aval sous une
    # forme opaque — soit « Impossible de déterminer les coordonnées des dalles »
    # (tri sur liste vide), soit « fichier central introuvable » au stade fusion.
    if downloaded[0] == 0 and skipped[0] == 0:
        raise RuntimeError(
            f"Aucune dalle téléchargée ({failed[0]}/{total} en échec). "
            "Vérifiez votre connexion réseau et la configuration du proxy "
            "(QGIS → Préférences → Réseau)."
        )

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
