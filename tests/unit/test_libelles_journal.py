"""Libellés du journal fidèles à ce qui est fait (crash test de_sh du 2026-11-01)."""
from __future__ import annotations

from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"


def test_couches_geopackage_comptees_par_fichier():
    """« 3 couche(s) GeoPackage créée(s) dans chemins_creux.gpkg » : les 3 couches étaient dans 3 fichiers."""
    from pipeline.cv.conversion_shp import resume_couches_ecrites
    refs = ["D:/o/chemins_creux/chemins_creux.gpkg|layername=Chemins creux",
            "D:/o/parcellaire/parcellaire.gpkg|layername=Parcellaire",
            "D:/o/talus/talus.gpkg|layername=Talus"]
    assert resume_couches_ecrites(refs) == \
        "3 couches écrites dans 3 GeoPackage : chemins_creux.gpkg, parcellaire.gpkg, talus.gpkg"
    assert resume_couches_ecrites(refs[:1]) == "1 couche écrite dans 1 GeoPackage : chemins_creux.gpkg"


def test_libelles_faux_disparus():
    textes = {p: p.read_text(encoding="utf-8") for p in (
        SRC / "app/services/finalize_service.py", SRC / "pipeline/modes/existing_rvt.py",
        SRC / "pipeline/ign/products/results.py")}
    tout = "".join(textes.values())
    assert "shapefile(s) dans QGIS" not in tout          # ce sont des GeoPackage
    assert "(existing_rvt)" not in tout                   # aussi employé par les modes LiDAR et MNT existants
    assert "TIF rogné copié" not in tout                  # juste après « copie sans crop » : rien n'est rogné
