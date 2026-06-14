#!/usr/bin/env python3
"""Crée l'index spatial ``.qix`` du quadrillage IGN LiDAR HD.

Le shapefile livré par l'IGN (``TA_diff_pkk_lidarhd_classe.shp``, ~490 000 dalles,
~176 Mo) **n'a pas d'index spatial** : chaque identify/rendu interactif balaie
toutes les entités → l'expérience « cliquer une dalle sur la carte » est
inutilisable. Ce script génère le sidecar ``.qix`` (un R-tree, ~2 Mo) que GDAL/OGR
et QGIS utilisent automatiquement pour les requêtes spatiales (``setFilterRect``,
rendu fenêtré). Pas de changement de format ni de chemin : le shapefile reste la
source de vérité, l'index est juste posé à côté.

> Pourquoi pas un GeoPackage ? Mesuré : le ``.gpkg`` équivalent est *plus gros*
> (188 vs 179 Mo) et le filtrage des dalles sans URL n'écarte rien (couverture
> LiDAR HD complète). Le ``.qix`` apporte le même R-tree pour ~2 Mo, sans toucher
> au format livré par l'IGN.

Outil **one-shot** de maintenance (sous ``dev/`` → exclu du ZIP). À relancer
quand l'IGN livre une nouvelle version du shapefile. L'index est détecté au
runtime via :func:`pipeline.ign.quadrillage_paths.resolve_quadrillage_path`
(qui renvoie le shapefile).

Pré-requis : la CLI GDAL (``ogrinfo``) sur le PATH — déjà une dépendance du
plugin (cf. preflight : gdalwarp/gdal_translate).

Usage :
    python dev/build_quadrillage_index.py            # shapefile par défaut
    python dev/build_quadrillage_index.py --force    # recrée un .qix existant
    python dev/build_quadrillage_index.py --src X.shp
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_QUAD_DIR = _REPO_ROOT / "data" / "quadrillage_france"
_DEFAULT_SRC = _QUAD_DIR / "TA_diff_pkk_lidarhd_classe.shp"


def _size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024) if path.exists() else 0.0


def build_index(src: Path, *, force: bool = False) -> Path:
    if shutil.which("ogrinfo") is None:
        raise RuntimeError(
            "ogrinfo introuvable sur le PATH. Lancez ce script depuis un shell "
            "GDAL (OSGeo4W / conda gdal) ou installez GDAL."
        )
    if not src.exists():
        raise FileNotFoundError(f"Shapefile source introuvable : {src}")

    qix = src.with_suffix(".qix")
    if qix.exists():
        if not force:
            raise FileExistsError(f"{qix.name} existe déjà — utilisez --force pour recréer.")
        qix.unlink()

    layer = src.stem  # le nom de couche d'un shapefile = son nom de fichier
    print(f"Création de l'index spatial sur {src.name} (couche « {layer} »)…")
    # OGR crée le .qix via une requête SQL « CREATE SPATIAL INDEX ».
    subprocess.run(
        ["ogrinfo", str(src), "-sql", f"CREATE SPATIAL INDEX ON {layer}"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    if not qix.exists():
        raise RuntimeError("La commande a réussi mais aucun .qix n'a été produit.")

    print(f"✅ {qix.name} créé ({_size_mb(qix):.1f} Mo)")
    print("   Le shapefile reste la source ; QGIS/OGR utilisent le .qix automatiquement.")
    print("   Livrez le dossier data/quadrillage_france/ avec ce .qix.")
    return qix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", type=Path, default=_DEFAULT_SRC, help="Shapefile du quadrillage")
    parser.add_argument("--force", action="store_true", help="Recrée le .qix existant")
    args = parser.parse_args(argv)
    # Console Windows (cp1252) : les emoji des messages plantent sinon.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass
    try:
        build_index(args.src, force=args.force)
    except Exception as exc:  # noqa: BLE001 — outil CLI : message clair, code non nul
        print(f"❌ {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
