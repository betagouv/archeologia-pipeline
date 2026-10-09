#!/usr/bin/env python3
"""Régénère le quadrillage IGN LiDAR HD (``data/quadrillage_france/``) depuis le WFS.

L'ancien zip ``TA_diff_pkk_lidarhd_classe.zip`` de ``diffusion-lidarhd.ign.fr`` n'existe
plus (503 depuis la migration Géoplateforme, constaté le 2026-10-07). La grille de
référence est désormais la couche WFS ``IGNF_LIDAR-HD_METADONNEE:metadata`` de
``data.geopf.fr``, que cartes.gouv.fr liste comme « Métadonnée du LiDAR HD » sur la fiche
« Nuages de points LiDAR HD ». L'IGN y ajoute ~2 500 dalles par lot toutes les 1 à
3 semaines → **relancer ce script avant chaque release** (le shapefile part dans le ZIP,
cf. test PKG-02).

Le schéma livré est conservé à l'identique (``nom_pkk`` C80 + ``url_telech`` C153,
Polygon, Lambert-93) : rien ne change pour le plugin (``tile_resolver``, sélection sur
carte). Le WFS n'a pas de ``nom_pkk`` : il est reconstruit depuis le nom de fichier de
``url_npl``. Pièges du WFS absorbés ici :

- 5 000 entités max par requête → pagination ``STARTINDEX`` (≈ 525 k dalles, quelques
  minutes) ;
- ``url_npl`` parfois en ``http://`` → forcé en https ;
- ~160 dalles avec un tiret dans le nom de fichier (``LHD_FXX_0881-6546_…``) : l'URL
  telle quelle répond 404, la forme à underscore répond 200 → corrigée.

Pré-requis : ``ogr2ogr``/``ogrinfo`` sur le PATH (OSGeo4W), comme
``build_quadrillage_index.py`` (appelé à la fin pour le ``.qix``). L'ancien jeu est déplacé
dans ``dev/docs/_local/quadrillage_<date du dbf>/`` (gitignoré, hors ZIP).

Usage :
    python dev/build_quadrillage_from_wfs.py                 # remplace le quadrillage + .qix
    python dev/build_quadrillage_from_wfs.py --out D:/q.shp  # écrit ailleurs, ne touche à rien
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_quadrillage_index import build_index  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEST = _REPO_ROOT / "data" / "quadrillage_france" / "TA_diff_pkk_lidarhd_classe.shp"
_BACKUP_ROOT = _REPO_ROOT / "dev" / "docs" / "_local"
_SIDECARS = (".shp", ".shx", ".dbf", ".prj", ".qix", ".cpg")

WFS = "https://data.geopf.fr/wfs/ows"
LAYER = "IGNF_LIDAR-HD_METADONNEE:metadata"
PAGE = 5000  # plafond du serveur (COUNT plus grand = silencieusement tronqué à 5 000)

_DASH_TILE = re.compile(r"(LHD_[A-Z]{3}_\d{4})-(\d{4})")
_TILE_KEY = re.compile(r"LHD_[A-Z]{3}_\d{4}_\d{4}")


def tile_record(url_npl: str) -> tuple[str, str]:
    """``url_npl`` du WFS → ``(nom_pkk, url_telech)`` au schéma du shapefile livré."""
    url = _DASH_TILE.sub(r"\1_\2", url_npl.strip())
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]
    nom = url.rsplit("/", 1)[-1].split(".", 1)[0]
    return nom, url


def _fetch_page(start: int) -> dict:
    query = (
        f"{WFS}?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature&TYPENAMES={LAYER}"
        f"&COUNT={PAGE}&STARTINDEX={start}&SRSNAME=EPSG:2154"
        "&PROPERTYNAME=url_npl,geom&OUTPUTFORMAT=application/json"
    )
    req = urllib.request.Request(query, headers={"User-Agent": "archeologia-pipeline/dev"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.load(resp)
        except Exception as exc:  # noqa: BLE001 — réseau : on réessaie
            print(f"  nouvel essai à STARTINDEX={start} ({exc})")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"WFS injoignable (STARTINDEX={start})")


def dump_geojsonl(path: Path) -> set[str]:
    """Pagine tout le WFS vers un GeoJSONSeq ; renvoie les clés ``LHD_XXX_xxxx_yyyy``."""
    keys: set[str] = set()
    start = 0
    with path.open("w", encoding="utf-8") as out:
        while True:
            page = _fetch_page(start)
            feats = page.get("features", [])
            total = page.get("numberMatched", "?")
            for feat in feats:
                nom, url = tile_record(feat["properties"].get("url_npl") or "")
                if not url:
                    continue
                keys.add(nom[:17])
                record = {
                    "type": "Feature",
                    "properties": {"nom_pkk": nom, "url_telech": url},
                    "geometry": feat["geometry"],
                }
                out.write(json.dumps(record) + "\n")
            start += len(feats)
            print(f"  {start}/{total} dalles", end="\r", flush=True)
            if not feats or (isinstance(total, int) and start >= total):
                break
    print()
    return keys


def geojsonl_to_shapefile(src: Path, dest: Path) -> None:
    """Même schéma que le shapefile IGN d'origine (largeurs comprises) et Lambert-93."""
    subprocess.run(
        [
            "ogr2ogr", "-f", "ESRI Shapefile", str(dest), str(src),
            "-nln", dest.stem, "-nlt", "POLYGON", "-a_srs", "EPSG:2154",
            "-sql",
            "SELECT CAST(nom_pkk AS character(80)) AS nom_pkk, "
            f'CAST(url_telech AS character(153)) AS url_telech FROM "{src.stem}"',
        ],
        check=True,
    )


def _previous_keys(shp: Path, tmp: Path) -> set[str]:
    csv_path = tmp / "previous.csv"
    subprocess.run(
        ["ogr2ogr", "-f", "CSV", str(csv_path), str(shp), "-select", "url_telech"], check=True
    )
    with csv_path.open(encoding="utf-8", newline="") as f:
        return {m.group(0) for row in csv.DictReader(f)
                if (m := _TILE_KEY.search(row["url_telech"] or ""))}


def _dbf_stamp(dbf: Path) -> str:
    with dbf.open("rb") as f:
        _, yy, mm, dd = struct.unpack("<BBBB", f.read(4))
    return f"{1900 + yy}-{mm:02d}-{dd:02d}"


def ecrire_archive(shp: Path | None = None) -> Path:
    """``data/quadrillage_france.zip`` : la grille compressée que versionne le dépôt GitHub
    (le ``.dbf`` brut dépasse les 100 Mo par fichier de GitHub). Le plugin la décompresse
    au premier besoin (``quadrillage_paths.assurer_quadrillage``). À commiter après chaque
    reconstruction de la grille."""
    shp = Path(shp or _DEST)
    archive = shp.parent.with_suffix(".zip")
    tmp = archive.with_suffix(".zip.partiel")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for ext in _SIDECARS:
            if shp.with_suffix(ext).exists():
                z.write(shp.with_suffix(ext), shp.stem + ext)
    os.replace(tmp, archive)
    print(f"✅ archive pour le dépôt GitHub : {archive} ({archive.stat().st_size / 1e6:.1f} Mo) — à commiter")
    return archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, help="Shapefile de sortie (sinon remplace le quadrillage)")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
    if shutil.which("ogr2ogr") is None:
        print("❌ ogr2ogr introuvable sur le PATH (shell OSGeo4W / conda gdal).", file=sys.stderr)
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="quadrillage_wfs_"))
    try:
        print(f"Téléchargement de {LAYER} ({WFS})…")
        new_keys = dump_geojsonl(tmp / "tiles.geojsonl")
        out = args.out or tmp / _DEST.name
        geojsonl_to_shapefile(tmp / "tiles.geojsonl", out)
        print(f"✅ {len(new_keys)} dalles → {out}")
        if args.out:
            return 0

        if _DEST.exists():
            old_keys = _previous_keys(_DEST, tmp)
            backup = _BACKUP_ROOT / f"quadrillage_{_dbf_stamp(_DEST.with_suffix('.dbf'))}"
            backup.mkdir(parents=True, exist_ok=True)
            for ext in _SIDECARS:
                if _DEST.with_suffix(ext).exists():
                    shutil.move(str(_DEST.with_suffix(ext)), str(backup / (_DEST.stem + ext)))
            print(f"   ancien quadrillage ({len(old_keys)} dalles) déplacé dans {backup}")
            print(f"   +{len(new_keys - old_keys)} dalles ajoutées, "
                  f"-{len(old_keys - new_keys)} retirées")
        _DEST.parent.mkdir(parents=True, exist_ok=True)
        for ext in (".shp", ".shx", ".dbf", ".prj"):
            shutil.move(str(out.with_suffix(ext)), str(_DEST.with_suffix(ext)))
        build_index(_DEST, force=True)
        ecrire_archive(_DEST)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
