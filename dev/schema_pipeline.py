"""Rastérise le schéma de la chaîne de traitement du manuel.

    C:/OSGeo4W/bin/python-qgis.bat dev/schema_pipeline.py [--echelle 2]

Source : ``dev/docs/schema_pipeline.svg`` (exclu du ZIP). Sortie : ``aide/img/pipeline.png``,
en double résolution par défaut, comme les captures du manuel (le visionneur rastérise lui-même
à ``largeur × dpr``, un PNG 1× serait flou à 150 %). Le `.bat` de QGIS avale stdout : le résultat
est aussi écrit dans ``dev/docs/_local/rendu/schema.txt``.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
SOURCE = RACINE / "dev" / "docs" / "schema_pipeline.svg"
SORTIE = RACINE / "aide" / "img" / "pipeline.png"


def rasteriser(source: Path, sortie: Path, echelle: float) -> tuple[int, int, int]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if os.name == "nt":
        # Hors écran, la base de polices de Qt est vide tant que ce dossier n'est pas déclaré :
        # QPainter et QtSvg dessinent alors chaque glyphe en rectangle plein (les widgets, eux,
        # passent par un autre moteur et ne le montrent pas).
        os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))
    from qgis.core import QgsApplication
    from qgis.PyQt.QtCore import Qt
    from qgis.PyQt.QtGui import QImage, QPainter
    from qgis.PyQt.QtSvg import QSvgRenderer

    # QgsApplication plutôt que QGuiApplication : hors écran, c'est lui qui charge la base de
    # polices (avec QGuiApplication seul, chaque glyphe sort en rectangle plein).
    QgsApplication.setPrefixPath(os.environ.get("QGIS_PREFIX_PATH", ""), True)
    app = QgsApplication([], True)
    app.initQgis()
    rendu = QSvgRenderer(str(source))
    if not rendu.isValid():
        raise SystemExit(f"SVG invalide : {source}")
    taille = rendu.defaultSize()
    largeur, hauteur = round(taille.width() * echelle), round(taille.height() * echelle)
    image = QImage(largeur, hauteur, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.white)
    peintre = QPainter(image)
    peintre.setRenderHint(QPainter.RenderHint.Antialiasing)
    peintre.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    peintre.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    rendu.render(peintre)
    peintre.end()
    sortie.parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(sortie), "PNG"):
        raise SystemExit(f"écriture impossible : {sortie}")
    taille_fichier = sortie.stat().st_size
    app.exitQgis()  # ponytail: la faute de segmentation à la sortie est bénigne (cf. rendu_hors_ecran)
    return largeur, hauteur, taille_fichier


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--echelle", type=float, default=2.0, help="facteur de rastérisation (2 = double résolution)")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--sortie", type=Path, default=SORTIE)
    args = parser.parse_args()
    largeur, hauteur, octets = rasteriser(args.source, args.sortie, args.echelle)
    ligne = f"{args.sortie} : {largeur}×{hauteur} px, {octets / 1024:.0f} Ko"
    journal = RACINE / "dev" / "docs" / "_local" / "rendu" / "schema.txt"
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(ligne + "\n", encoding="utf-8")
    print(ligne)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
