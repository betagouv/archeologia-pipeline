"""Capture « la légende de fiabilité dans QGIS » du manuel, hors écran.

    C:/OSGeo4W/bin/python-qgis.bat dev/capture_legende.py --run <dossier de sortie v3> [--sortie aide/img/resultats-legende-fiabilite.jpg]

Reconstruit le projet de validation du run avec le code courant (``ui/qgs_writer.write_validation_project``)
dans un dossier temporaire — le dossier du run n'est jamais modifié — puis rend côte à côte le panneau
des couches (``QgsLayerTreeView``, légende dépliée, cases telles que le projet les écrit) et la carte
(``QgsMapRendererSequentialJob``) sur une fenêtre centrée sur les détections. Double résolution comme
les autres captures (``QT_SCALE_FACTOR=2``). Le `.bat` de QGIS avale stdout : le résultat est aussi
écrit dans ``dev/docs/_local/rendu/legende.txt``.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
SORTIE = RACINE / "aide" / "img" / "resultats-legende-fiabilite.jpg"
ECHELLE = 2
LARGEUR_PANNEAU, HAUTEUR = 470, 370          # px logiques, comme le panneau Couches de QGIS
LARGEUR_CARTE = 950
FENETRE_M = 1200                              # largeur de la fenêtre cartographique, en mètres


def _sources_detections(run: Path) -> list[str]:
    from osgeo import ogr

    sources = []
    for gpkg in sorted((run / "livrable" / "detections").glob("*/*.gpkg")):
        ds = ogr.Open(str(gpkg))
        for i in range(ds.GetLayerCount()):
            sources.append(f"{gpkg}|layername={ds.GetLayer(i).GetName()}")
    return sources


def _vrt_indice(run: Path, tmp: Path) -> str | None:
    """Le premier VRT d'indice du run ; sinon un VRT construit dans ``tmp`` depuis les TIF."""
    from osgeo import gdal

    for vrt in sorted((run / "livrable" / "indices").glob("*/tif/index_*.vrt")):
        if vrt.parent.parent.name != "MNT":
            return str(vrt)
    for dossier in sorted((run / "livrable" / "indices").glob("*/tif")):
        tifs = sorted(str(t) for t in dossier.glob("*.tif"))
        if tifs and dossier.parent.name != "MNT":
            cible = tmp / dossier.parent.name / "tif" / f"index_{dossier.parent.name}.vrt"
            cible.parent.mkdir(parents=True, exist_ok=True)
            gdal.BuildVRT(str(cible), tifs)
            return str(cible)
    return None


def capturer(run: Path, sortie: Path, log) -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["QT_SCALE_FACTOR"] = str(ECHELLE)
    if os.name == "nt":
        # Sans ce dossier, la base de polices hors écran est vide : les vues d'items (arbre des
        # couches) et QPainter n'écrivent aucun glyphe (cf. dev/schema_pipeline.py).
        os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))
    from qgis.core import (
        QgsApplication, QgsCoordinateReferenceSystem, QgsLayerTreeModel, QgsMapRendererSequentialJob,
        QgsMapSettings, QgsPointXY, QgsProject, QgsRectangle, QgsVectorLayer,
    )
    from qgis.gui import QgsLayerTreeView
    from qgis.PyQt.QtCore import QSize, Qt
    from qgis.PyQt.QtGui import QColor, QFont, QImage, QPainter
    from qgis.PyQt.QtWidgets import QLabel, QWidget

    QgsApplication.setPrefixPath(os.environ.get("QGIS_PREFIX_PATH", ""), True)
    app = QgsApplication([], True)
    app.initQgis()
    app.setFont(QFont("Segoe UI", 9))  # la police par défaut hors écran (« Helvetica ») n'existe pas
    sys.path.insert(0, str(RACINE))
    from src.ui.qgs_writer import write_validation_project

    logger = logging.getLogger("capture_legende")
    with tempfile.TemporaryDirectory(prefix="legende_") as tmp_s:
        tmp = Path(tmp_s)
        sources = _sources_detections(run)
        vrt = _vrt_indice(run, tmp)
        log(f"{len(sources)} couche(s) de détections, indice : {vrt}")
        qgs = write_validation_project(tmp / "projet.qgs", [vrt] if vrt else [], sources, logger=logger)
        if qgs is None:
            log("ECHEC : projet non écrit")
            return 1
        projet = QgsProject.instance()
        if not projet.read(str(qgs)):
            log("ECHEC : projet non relu")
            return 1
        couches = list(projet.mapLayers().values())
        log("couches :", [c.name() for c in couches])

        # Panneau des couches, légende dépliée, cases telles que le projet les porte.
        modele = QgsLayerTreeModel(projet.layerTreeRoot())
        modele.setFlag(QgsLayerTreeModel.Flag.ShowLegend, True)
        modele.setFlag(QgsLayerTreeModel.Flag.AllowLegendChangeState, True)
        modele.setFlag(QgsLayerTreeModel.Flag.AllowNodeChangeVisibility, True)
        # Géométrie posée à la main : hors écran, une disposition n'est pas toujours appliquée
        # avant grab(), et la vue restait à sa taille par défaut.
        hauteur_titre = 24
        panneau = QWidget()
        panneau.setStyleSheet("background: #f0f0f0; color: #000;")
        panneau.resize(LARGEUR_PANNEAU, HAUTEUR)
        titre = QLabel("Couches", panneau)
        titre.setStyleSheet("padding: 3px 6px; font-weight: 600; background: #e4e4e4;")
        titre.setGeometry(0, 0, LARGEUR_PANNEAU, hauteur_titre)
        vue = QgsLayerTreeView(panneau)
        vue.setModel(modele)
        vue.setStyleSheet("background: #ffffff;")
        vue.setGeometry(0, hauteur_titre, LARGEUR_PANNEAU, HAUTEUR - hauteur_titre)
        panneau.show()
        vue.expandAll()
        app.processEvents()
        app.processEvents()
        image_panneau = panneau.grab().toImage()
        image_panneau.setDevicePixelRatio(1.0)  # sinon drawImage la réduirait à sa taille logique

        # Carte : fenêtre centrée sur le barycentre des détections, indice en fond.
        xs, ys, n = 0.0, 0.0, 0
        vecteurs = [c for c in couches if isinstance(c, QgsVectorLayer)]
        for c in vecteurs:
            for f in c.getFeatures():
                p = f.geometry().centroid().asPoint()
                xs, ys, n = xs + p.x(), ys + p.y(), n + 1
        if n == 0:
            log("ECHEC : aucune détection")
            return 1
        centre = QgsPointXY(xs / n, ys / n)
        demi_l = FENETRE_M / 2
        demi_h = demi_l * HAUTEUR / LARGEUR_CARTE
        etendue = QgsRectangle(centre.x() - demi_l, centre.y() - demi_h, centre.x() + demi_l, centre.y() + demi_h)
        reglages = QgsMapSettings()
        reglages.setLayers(vecteurs + [c for c in couches if c not in vecteurs])
        reglages.setDestinationCrs(QgsCoordinateReferenceSystem("EPSG:2154"))
        reglages.setOutputSize(QSize(LARGEUR_CARTE * ECHELLE, HAUTEUR * ECHELLE))
        reglages.setOutputDpi(96 * ECHELLE)
        reglages.setExtent(etendue)
        reglages.setBackgroundColor(QColor("#ffffff"))
        travail = QgsMapRendererSequentialJob(reglages)
        travail.start()
        travail.waitForFinished()
        image_carte = travail.renderedImage()

        # Composition côte à côte.
        total = QImage(image_panneau.width() + image_carte.width(), max(image_panneau.height(), image_carte.height()),
                       QImage.Format.Format_ARGB32)
        total.fill(Qt.GlobalColor.white)
        peintre = QPainter(total)
        peintre.drawImage(0, 0, image_panneau)
        peintre.drawImage(image_panneau.width(), 0, image_carte)
        peintre.end()
        sortie.parent.mkdir(parents=True, exist_ok=True)
        if not total.save(str(sortie), "JPEG", 85):
            log("ECHEC : image non écrite")
            return 1
        log(f"{sortie} : {total.width()}×{total.height()} px, {sortie.stat().st_size / 1024:.0f} Ko · centre {centre.x():.0f}, {centre.y():.0f}")
    app.exitQgis()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="dossier de sortie d'un traitement (arborescence livrable/ + technique/)")
    parser.add_argument("--sortie", type=Path, default=SORTIE)
    args = parser.parse_args()
    journal = RACINE / "dev" / "docs" / "_local" / "rendu" / "legende.txt"
    journal.parent.mkdir(parents=True, exist_ok=True)
    lignes: list[str] = []

    def log(*morceaux):
        ligne = " ".join(str(m) for m in morceaux)
        lignes.append(ligne)
        print(ligne)
        journal.write_text("\n".join(lignes) + "\n", encoding="utf-8")

    return capturer(args.run.resolve(), args.sortie, log)


if __name__ == "__main__":
    raise SystemExit(main())
