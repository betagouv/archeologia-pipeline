from __future__ import annotations

import json
import os
import re
import shutil
import socket
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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

# Fenêtre du chien de garde (audit 2026-10-19) : une connexion qui n'a pas livré un seul
# chunk pendant ce temps est coupée et reprise. Le délai de lecture de `requests` ne suffit pas :
# un serveur qui n'envoie plus qu'un filet d'octets (≈ 16 o/s, vécu sur data.geopf.fr) réarme
# ce délai à chaque octet alors que ``iter_content(8192)`` ne rend rien avant 8 Ko — fichier
# figé, connexion ouverte, aucune reprise pendant 8 minutes. 120 s : à 1,7–3,7 Mo/s un chunk
# arrive toutes les 2 à 5 ms et une dalle entière passe en 60–150 s ; ne pas recevoir 8 Ko le
# temps d'une dalle, c'est une connexion perdue — et 60 s couperait les proxys d'entreprise
# qui « rassurent » le client par un filet pendant qu'ils analysent le fichier.
BLOCAGE_S = 120.0

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

def _partial_telechargement(dest: Path) -> Path:
    """``<nom>.partial`` à côté de la cible : rien ne porte le nom final avant la
    fin complète du transfert (même volume → ``os.replace`` atomique). Sans
    extension ``.laz`` : ni PDAL ni un glob de dalles ne le prennent."""
    return dest.with_name(dest.name + ".partial")


# Qui tient chaque ``.partial`` : une annulation n'attend plus les téléchargements en cours (une lecture réseau
# bloquée ne se réveille pas sous Windows), et le fil abandonné finit plus tard. Si un nouveau lancement a repris
# la même dalle entre-temps, le fil abandonné ne doit pas supprimer le fichier de ce nouveau lancement.
_TENANTS_PARTIELS: Dict[str, int] = {}
_TENANTS_VERROU = threading.Lock()


def _prendre_partiel(partial: Path) -> int:
    """Le téléchargement courant devient le seul à pouvoir supprimer ``partial`` ; rend son jeton."""
    with _TENANTS_VERROU:
        jeton = _TENANTS_PARTIELS.get(str(partial), 0) + 1
        _TENANTS_PARTIELS[str(partial)] = jeton
        return jeton


def _liberer_partiel(partial: Path, jeton: int) -> None:
    """Supprime ``partial`` s'il appartient encore à ce téléchargement (jamais celui d'un lancement plus récent)."""
    with _TENANTS_VERROU:
        if _TENANTS_PARTIELS.get(str(partial)) != jeton:
            return
    try:
        partial.unlink(missing_ok=True)
    except OSError:
        pass


def note_verifiee_path(dest: Path) -> Path:
    return dest.with_name(dest.name + ".verifie.json")


def ecrire_note_verifiee(dest: Path, log: Optional[LogFn] = None) -> None:
    """Note « vérifiée » : taille et date du LAZ, posée quand il est prouvé entier
    (lecture complète réussie, ou taille égale à celle annoncée par le serveur)."""
    try:
        st = dest.stat()
        note_verifiee_path(dest).write_text(
            json.dumps({"taille": st.st_size, "mtime_ns": st.st_mtime_ns}), encoding="utf-8"
        )
    except OSError as exc:
        # accélérateur seulement : sans note, la prochaine relance relit le LAZ en entier — mais on le dit
        if log is not None:
            log(f"Note « vérifiée » de {dest.name} non écrite ({exc}) : la dalle sera relue en entier à la prochaine relance")


def note_verifiee_valide(dest: Path) -> bool:
    """Vrai si la note décrit encore le fichier (même taille, même date) : la
    lecture complète (14 à 36 s par dalle, mesuré) n'apprendrait rien."""
    try:
        note = json.loads(note_verifiee_path(dest).read_text(encoding="utf-8"))
        st = dest.stat()
    except (OSError, ValueError):
        return False
    return note == {"taille": st.st_size, "mtime_ns": st.st_mtime_ns}


def _taille_annoncee(headers) -> Optional[int]:
    """``Content-Length`` utilisable pour contrôler le fichier écrit ; ``None`` s'il
    manque ou si le corps est encodé (gzip… : la taille annoncée n'est pas celle
    écrite)."""
    try:
        if str(headers.get("Content-Encoding") or "identity").lower() != "identity":
            return None
        n = int(headers.get("Content-Length") or 0)
    except (TypeError, ValueError, AttributeError):
        return None
    return n if n > 0 else None


def _reprise_acceptee(resp, written: int, total: Optional[int]) -> bool:
    """206 dont le ``Content-Range`` reprend exactement à ``written`` sur ``total`` octets."""
    if total is None or int(getattr(resp, "status_code", 0) or 0) != 206:
        return False
    m = re.match(r"\s*bytes\s+(\d+)-(\d+)/(\d+)\s*$", str(resp.headers.get("Content-Range") or ""))
    return bool(m) and int(m.group(1)) == written and int(m.group(3)) == total


def _coupure(resp) -> Optional[Callable[[], None]]:
    """Comment couper la connexion d'une réponse ``requests`` en cours de lecture, depuis un
    autre fil : ``raw.shutdown()`` (urllib3 ≥ 2.3, fait pour cela), sinon ``shutdown`` du socket
    de la connexion, sinon du socket que tient le fichier de lecture (urllib3 2.2 de QGIS, ou
    réponse en ``Connection: close`` où la connexion a lâché son socket). Jamais
    ``resp.close()`` : il attend le verrou de la lecture en cours et bloque. ``None`` si rien
    n'est accessible (réponses doublées des tests) : pas de garde."""
    brut = getattr(resp, "raw", None)
    if brut is None:
        return None
    fn = getattr(brut, "shutdown", None)
    if callable(fn):
        return fn
    for sock in (getattr(getattr(brut, "connection", None), "sock", None),
                 getattr(getattr(getattr(getattr(brut, "_fp", None), "fp", None), "raw", None), "_sock", None)):
        if sock is not None and callable(getattr(sock, "shutdown", None)):
            return lambda s=sock: s.shutdown(socket.SHUT_RDWR)
    return None


def _chien_de_garde(couper: Callable[[], None], dernier: list, arret: threading.Event, cancel: CancelFn,
                    blocage_s: float, motif: Dict[str, str]) -> threading.Thread:
    """Fil démon qui coupe la connexion (:func:`_coupure`) si aucun chunk n'est arrivé depuis
    ``blocage_s`` ou si l'annulation est demandée — la lecture lève alors à l'octet suivant,
    en une seconde ; pendant un silence total, c'est le délai de lecture qui la réveille."""
    def course():
        while not arret.wait(0.5):
            if cancel():
                motif["cause"] = "annulation"
            elif time.monotonic() - dernier[0] > blocage_s:
                motif["cause"] = f"aucun octet reçu depuis {blocage_s:g} s"
            else:
                continue
            try:
                couper()
            except Exception:  # noqa: BLE001 — socket déjà fermé par l'autre bout
                pass
            return

    fil = threading.Thread(target=course, name="garde-telechargement", daemon=True)
    fil.start()
    return fil


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
    blocage_s: float = BLOCAGE_S,
) -> Tuple[bool, bool]:
    """Returns (success, skipped_existing).

    ``blocage_s`` : fenêtre du chien de garde (``BLOCAGE_S``) ; sert aussi de délai de
    lecture à ``requests`` (sous Windows, couper le socket ne réveille pas une lecture
    pendant un silence total : le délai s'en charge). Les tentatives suivantes reprennent
    à l'octet manquant (``Range``) quand le serveur l'accepte (206 + ``Content-Range``
    cohérent), sinon depuis zéro.
    """

    if cancel():
        return False, False

    dalles_dir.mkdir(parents=True, exist_ok=True)
    dest = dalles_dir / filename
    partial = _partial_telechargement(dest)
    jeton = _prendre_partiel(partial)
    if partial.exists():
        # Reste d'un téléchargement tué (QGIS fermé, plantage) : jamais repris tel quel.
        try:
            partial.unlink()
        except OSError:
            pass

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
        elif note_verifiee_valide(dest):
            # Lu en entier par un lancement précédent et inchangé depuis.
            log(f"✅ {filename} déjà présente (vérifiée)")
            return True, True
        else:
            # Validation profonde pour détecter les fichiers tronqués
            log(f"🔍 Validation PDAL profonde de {filename}...")
            ok_deep, msg_deep = validate_laz_deep(dest)
            if ok_deep:
                ecrire_note_verifiee(dest, log)
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
        # Comme le téléchargement : copie en .partial, validée, puis renommée — une copie interrompue
        # ne laisse jamais un fichier tronqué sous le nom de la dalle.
        partial = dest.with_name(dest.name + ".partial")
        log(f"📥 Copie du fichier local {local} vers {dest}...")
        shutil.copy2(str(local), str(partial))
        log(f"🔍 Validation PDAL de {filename}...")
        ok, msg = _is_valid_with_pdal(partial)
        if not ok:
            log(f"❌ Fichier copié mais invalide via PDAL: {filename}")
            if msg:
                log(f"PDAL: {msg}")
            try:
                partial.unlink()
            except Exception:
                pass
            return False, False
        os.replace(str(partial), str(dest))
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
        declared = _taille_annoncee(headers) or 0
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
        if partial.exists():
            partial.unlink()
        log(
            f"❌ {filename}: dépasse le cap de {MAX_DOWNLOAD_SIZE_MB} Mo en "
            "cours de téléchargement — abandonné (anti-saturation disque)"
        )
        return True

    last_err: Optional[Exception] = None
    total: Optional[int] = None        # taille annoncée au premier essai : référence des reprises
    etag: Optional[str] = None
    lecture_s = min(float(timeout_s), float(blocage_s))
    for attempt in range(1, max_retries + 1):
        if cancel():
            return False, False

        # Reprise à l'octet manquant : ce que l'essai précédent a écrit dans le .partial n'est pas
        # retéléchargé quand le serveur accepte Range ; la taille totale vient du premier essai.
        written = 0
        if total is not None and partial.exists():
            try:
                written = partial.stat().st_size
            except OSError:
                written = 0
            if written >= total:
                written = 0
        garde_motif: Dict[str, str] = {}
        try:
            # Écriture dans <nom>.partial (lot 3) : un flux coupé ou un QGIS tué
            # laissait un LAZ tronqué sous le nom final.
            if HAS_REQUESTS:
                # Utiliser requests qui gère correctement les proxies HTTPS
                entetes_req = {"User-Agent": "QGIS-ArcheologiaPipeline/1.0"}
                if written > 0:
                    entetes_req["Range"] = f"bytes={written}-"
                    if etag:
                        entetes_req["If-Range"] = etag
                    log(f"📥 Reprise de {filename} à l'octet {written} sur {total}")
                resp = requests.get(
                    url,
                    headers=entetes_req,
                    proxies=proxies if proxies else None,
                    timeout=lecture_s,
                    stream=True,
                )
                resp.raise_for_status()
                if _declared_too_big(resp.headers):
                    return False, False  # définitif : pas de retry
                if written > 0 and _reprise_acceptee(resp, written, total):
                    annoncee = total
                    mode = "ab"
                else:
                    if written > 0:
                        log(f"ℹ️ {filename} : reprise refusée par le serveur, téléchargement depuis zéro")
                        written = 0
                    annoncee = _taille_annoncee(resp.headers)
                    total = annoncee
                    etag = str(resp.headers.get("ETag") or "") or None
                    mode = "wb"
                couper = _coupure(resp)
                dernier = [time.monotonic()]
                arret = threading.Event()
                garde = (_chien_de_garde(couper, dernier, arret, cancel, blocage_s, garde_motif)
                         if couper is not None else None)
                try:
                    with open(partial, mode) as f:
                        for chunk in resp.iter_content(chunk_size=chunk_size):
                            dernier[0] = time.monotonic()
                            if cancel():
                                f.close()
                                _liberer_partiel(partial, jeton)
                                return False, False
                            if chunk:
                                written += len(chunk)
                                if _over_cap(written, f):
                                    return False, False  # définitif : pas de retry
                                f.write(chunk)
                finally:
                    arret.set()
                    if garde is not None:
                        garde.join(timeout=2.0)
            else:
                # Fallback urllib (peut ne pas fonctionner avec proxy HTTPS)
                req = urllib.request.Request(url, headers={"User-Agent": "QGIS-ArcheologiaPipeline/1.0"})
                proxy_handler = urllib.request.ProxyHandler(proxies if proxies else None)
                opener = urllib.request.build_opener(proxy_handler)
                with opener.open(req, timeout=lecture_s) as r:
                    entetes = getattr(r, "headers", {})
                    if _declared_too_big(entetes):
                        return False, False  # définitif : pas de retry
                    annoncee = _taille_annoncee(entetes)
                    written = 0
                    with open(partial, "wb") as f:
                        while True:
                            if cancel():
                                f.close()
                                _liberer_partiel(partial, jeton)
                                return False, False
                            chunk = r.read(chunk_size)
                            if not chunk:
                                break
                            written += len(chunk)
                            if _over_cap(written, f):
                                return False, False  # définitif : pas de retry
                            f.write(chunk)

            # Flux coupé net, sans erreur réseau : la taille annoncée le trahit.
            if annoncee is not None and written != annoncee:
                raise IOError(f"téléchargement incomplet : {written} octets reçus sur {annoncee} annoncés")
            os.replace(str(partial), str(dest))

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

            if annoncee is None:
                # Sans taille annoncée, seule une lecture complète prouve le fichier
                # entier — une fois : la note « vérifiée » la retient.
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

            ecrire_note_verifiee(dest, log)
            return True, False
        except Exception as e:
            if cancel() or garde_motif.get("cause") == "annulation":
                # Coupure voulue par l'utilisateur : rien à réessayer, rien à garder.
                _liberer_partiel(partial, jeton)
                return False, False
            last_err = e
            cause = garde_motif.get("cause")
            if cause:
                log(f"⚠️ {filename} : {cause}, connexion coupée")
            log(f"Tentative {attempt}/{max_retries} échouée pour {filename}: {cause or e}")
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
    on_tile_done: Optional[Callable[[int, int, str, bool, bool], None]] = None,
) -> IgnDownloadResult:
    """Télécharge les dalles IGN listées dans ``input_file``.

    ``on_tile_done`` (optionnel) est invoqué après chaque dalle traitée
    avec ``(completed_index_1based, total, filename, success, skipped)`` — le caller s'en
    sert pour remonter une sous-progression à l'utilisateur (ligne
    réécrite dans le journal). Les téléchargements parallèles ne
    garantissent pas l'ordre des appels (``completed_index`` reflète
    l'ordre d'arrivée, pas l'ordre des tâches).
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    from ..output_paths import dalles_dir as _dalles_dir, fichier_tri_path
    dalles_dir = _dalles_dir(output_dir)
    sorted_list = fichier_tri_path(output_dir)
    sorted_list.parent.mkdir(parents=True, exist_ok=True)

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
                on_tile_done(current, total, result.filename, result.success, result.skipped)
            except Exception:
                pass

    # Préparer les tâches
    tasks = [
        _DownloadTask(index=idx, filename=filename, url=url)
        for idx, (filename, url) in enumerate(file_list, start=1)
    ]

    # Exécution parallèle. L'annulation est examinée toutes les demi-secondes et n'attend pas les téléchargements
    # en cours : une lecture réseau bloquée (serveur muet) ne se réveille pas sous Windows avant des octets ou le
    # délai de lecture — 40 s d'attente en recette le 2026-10-21. Les fils abandonnés finissent seuls et ne
    # suppriment que leur propre ``.partial`` (_liberer_partiel).
    executor = ThreadPoolExecutor(max_workers=max_workers)
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
    en_cours = set(future_to_task)
    annule = False
    while en_cours and not annule:
        finis, en_cours = wait(en_cours, timeout=0.5, return_when=FIRST_COMPLETED)
        for future in finis:
            if cancel():
                annule = True
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
                        on_tile_done(current, total, task.filename, False, False)
                    except Exception:
                        pass

        if cancel():
            annule = True
    executor.shutdown(wait=not annule, cancel_futures=annule)

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
