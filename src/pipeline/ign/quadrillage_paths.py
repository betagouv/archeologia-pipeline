"""Résolveur de chemin du quadrillage IGN LiDAR HD (pur, sans QGIS/OGR).

Le quadrillage est la grille des dalles téléchargeables (un polygone par dalle,
attributs ``nom_pkk`` + ``url_telech``). C'est un shapefile lourd (~176 Mo,
~490 k entités) qui, **sans index spatial**, est pénible à manipuler de façon
interactive (chaque clic/rendu balaie toutes les entités). On l'accélère avec un
sidecar ``.qix`` (R-tree, ~2 Mo, cf. ``dev/build_quadrillage_index.py``) que
GDAL/OGR et QGIS utilisent automatiquement — sans changer le format livré.

Ce module fournit la **source de vérité unique** du chemin, partagée par
:mod:`pipeline.ign.tile_resolver` (intersection polygone) et l'outil UI de
sélection des dalles sur le canevas. Il renvoie le shapefile (notre artefact
livré). Un ``.gpkg`` équivalent reste accepté en option : s'il est présent, il
est préféré — la bascule est ainsi transparente si on régénère un jour la grille
dans ce format.

Le dépôt GitHub versionne la grille **compressée** (``data/quadrillage_france.zip``,
~23 Mo : le ``.dbf`` brut dépasse les 100 Mo par fichier qu'accepte GitHub) ; le ZIP
du plugin la livre décompressée. :func:`assurer_quadrillage` décompresse l'archive
au premier besoin, pour qu'une copie venue de GitHub télécharge aussi les dalles.

Aucun import QGIS/OGR ici : le module reste importable hors QGIS (et donc
collectable par pytest), et ``tile_resolver`` peut l'importer en intra-paquet
(``from .quadrillage_paths import ...``) sans dépendance croisée vers ``app``.
"""
from __future__ import annotations

import datetime as _dt
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Tuple

_QUADRILLAGE_DIR = Path("data") / "quadrillage_france"
_BASENAME = "TA_diff_pkk_lidarhd_classe"

QUADRILLAGE_GPKG_RELPATH = _QUADRILLAGE_DIR / f"{_BASENAME}.gpkg"
QUADRILLAGE_SHP_RELPATH = _QUADRILLAGE_DIR / f"{_BASENAME}.shp"
QUADRILLAGE_ZIP_RELPATH = _QUADRILLAGE_DIR.with_suffix(".zip")


def _en_tete_dbf(dbf: Path) -> Optional[bytes]:
    """Les 8 premiers octets du ``.dbf``, sur disque ou dans l'archive voisine (copie GitHub)."""
    if dbf.is_file():
        with open(dbf, "rb") as f:
            return f.read(8)
    archive = dbf.parent.with_suffix(".zip")
    if archive.is_file():
        with zipfile.ZipFile(archive) as z, z.open(dbf.name) as f:
            return f.read(8)
    return None


_MOIS = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
         "septembre", "octobre", "novembre", "décembre")


def quadrillage_info(chemin: Path) -> Optional[Tuple[_dt.date, Optional[int]]]:
    """``(date de la grille, nombre de dalles)`` lus dans le fichier, rien à maintenir.

    Un shapefile porte sa date de dernière écriture et son nombre d'enregistrements
    dans l'en-tête de son ``.dbf`` (octets 1-3 : année − 1900, mois, jour ; 4-7 :
    effectif, entier 32 bits petit-boutiste) — c'est la date de la régénération par
    ``dev/build_quadrillage_from_wfs.py``. Un GeoPackage n'a pas cet en-tête : date
    de modification du fichier, effectif inconnu. Fichier absent ou illisible →
    ``None`` (le bandeau n'affiche rien).
    """
    chemin = Path(chemin)
    try:
        tete = _en_tete_dbf(chemin.with_suffix(".dbf")) if chemin.suffix.lower() == ".shp" else None
        if tete is not None and len(tete) == 8:
            date = _dt.date(1900 + tete[1], tete[2], tete[3])
            return date, int.from_bytes(tete[4:8], "little")
        if chemin.is_file():
            return _dt.date.fromtimestamp(chemin.stat().st_mtime), None
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return None
    return None


def phrase_quadrillage(info: Optional[Tuple[_dt.date, Optional[int]]]) -> str:
    """« Grille IGN du 7 octobre 2026, 524 687 dalles. » — pour le bandeau de l'étape 1 :
    l'utilisateur sait si une dalle publiée depuis peut manquer. ``""`` sans info."""
    if info is None:
        return ""
    date, n = info
    jour = "1er" if date.day == 1 else str(date.day)
    texte = f"Grille IGN du {jour} {_MOIS[date.month - 1]} {date.year}"
    if n is not None:
        texte += ", " + f"{n:,}".replace(",", " ") + " dalles"
    return texte + "."


def resolve_quadrillage_path(plugin_root: Path) -> Path:
    """Chemin du quadrillage à utiliser, relatif à ``plugin_root``.

    Préfère le GeoPackage slim (léger + R-tree) s'il existe, sinon le shapefile
    legacy. Si aucun n'existe, renvoie le chemin ``.shp`` (la vérification
    d'existence et le message d'erreur restent à la charge de l'appelant).
    """
    gpkg = plugin_root / QUADRILLAGE_GPKG_RELPATH
    if gpkg.exists():
        return gpkg
    return plugin_root / QUADRILLAGE_SHP_RELPATH


def assurer_quadrillage(plugin_root: Path) -> Path:
    """:func:`resolve_quadrillage_path`, après avoir décompressé l'archive si la grille manque.

    Extraction dans un dossier voisin, puis déplacement fichier par fichier avec le
    ``.shp`` en dernier : une extraction interrompue ne laisse jamais une grille qui
    paraît complète, et la suivante recommence. Sans archive (ni grille), le chemin
    ``.shp`` est renvoyé tel quel : l'appelant signale son absence.
    """
    plugin_root = Path(plugin_root)
    chemin = resolve_quadrillage_path(plugin_root)
    archive = plugin_root / QUADRILLAGE_ZIP_RELPATH
    if chemin.exists() or not archive.is_file():
        return chemin
    dest = plugin_root / _QUADRILLAGE_DIR
    tmp = dest.with_name(dest.name + ".partiel")
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(archive) as z:
        noms = z.namelist()
        if any("/" in n or "\\" in n or n.startswith(".") for n in noms):
            raise ValueError(f"Archive de grille inattendue : {archive}")
        z.extractall(tmp)
    dest.mkdir(parents=True, exist_ok=True)
    for nom in sorted(noms, key=lambda n: n.lower().endswith(".shp")):  # le .shp en dernier
        os.replace(tmp / nom, dest / nom)
    shutil.rmtree(tmp, ignore_errors=True)
    return resolve_quadrillage_path(plugin_root)
