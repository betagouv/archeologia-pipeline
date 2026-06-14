"""Formatage du ``dalles_urls.txt`` issu de la sélection des dalles sur carte (pur).

L'outil de sélection (``src/ui/map_tools/tile_picker_tool.py``) laisse l'utilisateur
cliquer des dalles IGN sur le canevas QGIS ; la page d'entrée lit
``(nom_pkk, url_telech)`` des entités retenues et délègue ici la mise en forme.

Le fichier produit alimente **directement** le téléchargeur (``ign.downloader.
parse_ign_input_file`` : une ligne ``nom,url`` ou ``url`` par dalle ; lignes ``#``
ignorées ; ``IgnDownloadStrategy`` ne « résout » pas un ``.txt``). C'est le même
format que ``resolve_tiles_from_polygon`` — d'où l'absence de toute modification
du pipeline de téléchargement.

Module pur (aucun import QGIS) → collectable et testable hors QGIS.
"""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Iterable, Optional, Tuple
from urllib.parse import urlparse

# Bornes de taille observées d'une dalle LiDAR HD (cf. downloader.py:90-93).
_TILE_MIN_MB = 50
_TILE_MAX_MB = 400


def _filename_for(url: str, nom: str) -> str:
    """Nom de fichier sous lequel la dalle sera enregistrée par le téléchargeur.

    Il DOIT porter l'extension réelle (``.copc.laz``) : PDAL déduit le lecteur de
    l'extension, et un fichier sans extension est rejeté (« Cannot determine
    reader »). On prend donc le basename du chemin de l'URL — comme
    ``tile_resolver.resolve_tiles_from_polygon`` — et on ne retombe sur ``nom``
    (le ``nom_pkk``, sans extension) que si l'URL n'a pas de basename.
    """
    base = PurePosixPath(urlparse(url).path).name
    return base or nom


def format_dalles_urls(tiles: Iterable[Tuple[Optional[str], Optional[str]]]) -> str:
    """Rend le contenu d'un ``dalles_urls.txt`` à partir de ``(nom_pkk, url_telech)``.

    - ne garde que les dalles dont l'URL est une URL ``http(s)`` non vide ;
    - **déduplique par URL** en conservant l'ordre de première apparition ;
    - écrit ``filename,url`` où ``filename`` est le **basename de l'URL** (avec
      son extension ``.copc.laz``), cf. :func:`_filename_for` ;
    - préfixe un en-tête commentaire ``#`` indiquant le nombre de dalles (ignoré
      par le parseur et par le compteur de téléchargement).
    """
    lines: list[str] = []
    seen: set[str] = set()
    for nom, url in tiles:
        url = (url or "").strip()
        if not url.startswith(("http://", "https://")):
            continue
        if url in seen:
            continue
        seen.add(url)
        filename = _filename_for(url, (nom or "").strip())
        lines.append(f"{filename},{url}" if filename else url)

    header = f"# {len(lines)} dalle(s) sélectionnée(s) sur la carte"
    return "\n".join([header, *lines]) + "\n"


def estimate_download_size(n: int) -> str:
    """Fourchette de volume de téléchargement pour ``n`` dalles, en français.

    Ex. ``"≈ 50–400 Mo"`` (petits volumes) ou ``"≈ 0,5–4,0 Go"``. Chaîne vide
    si ``n <= 0``. Bornes par dalle : 50–400 Mo (estimation indicative).
    """
    if n <= 0:
        return ""
    lo, hi = n * _TILE_MIN_MB, n * _TILE_MAX_MB
    if hi < 1000:
        return f"≈ {lo}–{hi} Mo"
    lo_go = f"{lo / 1000:.1f}".replace(".", ",")
    hi_go = f"{hi / 1000:.1f}".replace(".", ",")
    return f"≈ {lo_go}–{hi_go} Go"
