from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import shutil
import time
import urllib.parse
import urllib.request

from .coords_fallback import build_sorted_records_with_fallback
from .pdal_validation import validate_las_or_laz_with_pdal


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
) -> Tuple[bool, bool]:
    """Returns (success, skipped_existing)."""

    if cancel():
        return False, False

    dalles_dir.mkdir(parents=True, exist_ok=True)
    dest = dalles_dir / filename

    if dest.exists():
        ok, msg = _is_valid_with_pdal(dest)
        if ok:
            log(f"✅ {filename} déjà téléchargé")
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

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        if cancel():
            return False, False

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "QGIS-ArcheologiaPipeline/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                with open(dest, "wb") as f:
                    while True:
                        if cancel():
                            try:
                                f.close()
                            finally:
                                if dest.exists():
                                    dest.unlink()
                            return False, False

                        chunk = r.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)

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


def download_ign_dalles(
    *,
    input_file: Path,
    output_dir: Path,
    log: LogFn = _default_log,
    progress: ProgressFn = _default_progress,
    stage: StageFn = _default_stage,
    cancel: CancelFn = _default_cancel,
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

    downloaded = 0
    skipped = 0

    for idx, (filename, url) in enumerate(file_list, start=1):
        if cancel():
            log("Annulation demandée")
            break

        pct = int(round(100.0 * (idx - 1) / max(1, total)))
        progress(pct)

        ok, was_skipped = download_one(url, filename, dalles_dir, log=log, cancel=cancel)
        if not ok:
            raise RuntimeError(f"Échec du téléchargement: {filename}")

        if was_skipped:
            skipped += 1
        else:
            downloaded += 1

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
        downloaded=downloaded,
        skipped_existing=skipped,
    )
