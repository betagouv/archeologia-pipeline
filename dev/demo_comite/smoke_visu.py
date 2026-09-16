"""Rendu hors écran de l'onglet « Visualisation » — vérification avant démo.

`src/ui/` n'est pas couvert par pytest (pas de QGIS en autonome) : une régression
d'énuméré plat ou de QSS ne se verrait qu'au lancement dans QGIS. Ce script monte
l'onglet pour de vrai dans le Python de QGIS et en sort une image.

    cmd //c "C:\\Program Files\\QGIS 4.0.3\\bin\\python-qgis.bat" dev/demo_comite/smoke_visu.py

Sortie : ``visu_smoke.png`` (+ un second rendu, département filtré).
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[2]

# Le dossier du dépôt s'appelle « archeologia-pipeline » : un tiret n'est pas un
# identifiant Python. On enregistre donc le paquet sous un nom valide pour que
# les imports relatifs (``from ..app.services...``) se résolvent.
_pkg = types.ModuleType("archeo")
_pkg.__path__ = [str(ROOT)]
sys.modules["archeo"] = _pkg

from qgis.core import QgsApplication  # noqa: E402
from qgis.PyQt.QtCore import QSize  # noqa: E402
from qgis.PyQt.QtWidgets import QMainWindow  # noqa: E402


class _StubCanvas:
    """Canevas minimal : on veut surtout savoir sur quelle emprise on recadre."""

    def __init__(self):
        self.extents = []
        self.refreshes = 0

    def refresh(self):
        self.refreshes += 1

    def setExtent(self, rect):
        self.extents.append(rect)

    def mapSettings(self):
        from qgis.core import QgsMapSettings, QgsCoordinateReferenceSystem
        ms = QgsMapSettings()
        ms.setDestinationCrs(QgsCoordinateReferenceSystem("EPSG:2154"))
        return ms


class _StubIface:
    """Le strict minimum d'iface dont l'onglet se sert."""

    def __init__(self):
        self._win = QMainWindow()
        self._canvas = _StubCanvas()
        self.warnings = []

    def mainWindow(self):
        return self._win

    def mapCanvas(self):
        return self._canvas

    def messageBar(self):
        outer = self

        class _B:
            def pushWarning(self, title, msg):
                outer.warnings.append(msg)
                print(f"[messageBar] {title}: {msg}")
        return _B()


def main() -> int:
    QgsApplication.setPrefixPath(os.environ.get("QGIS_PREFIX_PATH", ""), True)
    app = QgsApplication([], True)
    app.initQgis()

    from archeo.src.ui.visualisation_tab import VisualisationTab

    # La plateforme « offscreen » ne voit aucune police système : sans ça tout
    # le texte sort en carrés et le rendu ne prouve plus rien sur les libellés.
    from qgis.PyQt.QtGui import QFont, QFontDatabase
    fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    for name in ("segoeui.ttf", "segoeuib.ttf", "consola.ttf"):
        if (fonts / name).is_file():
            QFontDatabase.addApplicationFont(str(fonts / name))
    app.setFont(QFont("Segoe UI", 9))

    qss = (ROOT / "src" / "ui" / "theme" / "v2.qss").read_text(encoding="utf-8")
    qss = qss.replace("@ICONS@", (ROOT / "src" / "ui" / "theme" / "icons").as_posix())
    app.setStyleSheet(qss)

    iface = _StubIface()
    tab = VisualisationTab(ROOT, iface=iface)
    tab.resize(QSize(1120, 660))
    tab.show()
    app.processEvents()

    dept = tab._dept
    print(f"département sélectionné : {dept.name if dept else '(aucun)'}")
    print(f"cartes construites      : {len(tab._cards)} -> {list(tab._cards)}")
    vides = [k for k, c in tab._cards.items() if c._thumb._pixmap.isNull()]
    print(f"vignettes manquantes    : {vides or 'aucune'}")

    tab.grab().save(str(ROOT / "visu_smoke.png"))

    # ---- le geste de la démo : ouvrir deux indices, en retirer un ----
    from qgis.core import QgsProject

    counts = []
    tab.layer_count_changed.connect(counts.append)

    for key in ("MNT", "LD"):
        tab._open_indice(key)
        app.processEvents()
    ouvertes = [layer.name() for layer in QgsProject.instance().mapLayers().values()]
    print(f"couches dans le projet  : {ouvertes}")
    print(f"états des cartes        : "
          f"{ {k: c._state for k, c in tab._cards.items() if c._state != 'idle'} }")
    print(f"emprise du recadrage    : "
          f"{[f'{r.xMinimum():.0f},{r.yMinimum():.0f} {r.xMaximum():.0f},{r.yMaximum():.0f}' for r in iface._canvas.extents]}")
    print(f"compteur d'onglet       : {counts}")
    print(f"avertissements          : {iface.warnings or 'aucun'}")
    tab.grab().save(str(ROOT / "visu_smoke_chargee.png"))

    tab._remove_indice("MNT")
    app.processEvents()
    print(f"après retrait du MNT    : {len(tab._layers)} couche(s) suivie(s), "
          f"carte MNT en '{tab._cards['MNT']._state}'")
    ok_interaction = (
        len(ouvertes) == 2 and not iface.warnings
        and tab._cards["LD"]._state == "loaded" and tab._cards["MNT"]._state == "idle")

    # second rendu : filtre du rail + famille, pour éprouver les deux interactions
    tab._filter.setText("morbihan")
    app.processEvents()
    tab._set_family("RVT")
    app.processEvents()
    print(f"après filtre            : {tab._dept.name if tab._dept else '—'}, "
          f"{len(tab._cards)} cartes")
    tab.grab().save(str(ROOT / "visu_smoke_filtre.png"))

    # Chemin d'erreur : catalogue absent. Le brief interdit une grille vide
    # sans explication, et c'est un cas qu'aucun autre test n'exerce.
    sans_cat = VisualisationTab(ROOT / "_dossier_inexistant", iface=_StubIface())
    sans_cat.resize(QSize(900, 500))
    sans_cat.show()                    # isVisible() est faux tant que rien n'est affiché
    app.processEvents()
    message = sans_cat._empty.text()
    explique = sans_cat._empty.isVisible() and "atalogue" in message
    print(f"catalogue absent        : {'explique' if explique else 'MUET'} "
          f"-> {message.splitlines()[0] if message else '(vide)'}")
    ok_interaction = ok_interaction and explique

    app.exitQgis()
    print("RESULTAT :", "OK" if (not vides and ok_interaction) else "ECHEC")
    return 0 if (not vides and ok_interaction) else 1


if __name__ == "__main__":
    sys.exit(main())
