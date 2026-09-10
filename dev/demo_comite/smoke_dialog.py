"""Rendu hors écran du dialogue COMPLET (les deux onglets).

Complète ``smoke_visu.py``, qui ne monte que l'onglet Visualisation isolé : ici
c'est le vrai ``WizardDialog``, donc on vérifie à la fois la barre d'onglets
(largeur des libellés) et que le wizard n'a rien perdu en devenant un onglet.

    & "C:\\Program Files\\QGIS 4.0.3\\bin\\python-qgis.bat" dev\\demo_comite\\smoke_dialog.py

Sortie : ``dialog_onglet1.png`` / ``dialog_onglet2.png`` + les largeurs mesurées.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[2]
_pkg = types.ModuleType("archeo")
_pkg.__path__ = [str(ROOT)]
sys.modules["archeo"] = _pkg

from qgis.core import QgsApplication  # noqa: E402
from qgis.PyQt.QtGui import QFont, QFontDatabase, QFontMetrics  # noqa: E402
from qgis.PyQt.QtWidgets import QMainWindow  # noqa: E402


class _StubIface:
    def __init__(self):
        self._win = QMainWindow()

    def mainWindow(self):
        return self._win

    def mapCanvas(self):
        class _C:
            def refresh(self):
                pass

            def setExtent(self, *_a):
                pass
        return _C()

    def messageBar(self):
        class _B:
            def pushWarning(self, title, msg):
                print(f"[messageBar] {title}: {msg}")
        return _B()


def main() -> int:
    QgsApplication.setPrefixPath(os.environ.get("QGIS_PREFIX_PATH", ""), True)
    app = QgsApplication([], True)
    app.initQgis()

    fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    for name in ("segoeui.ttf", "segoeuib.ttf", "consola.ttf"):
        if (fonts / name).is_file():
            QFontDatabase.addApplicationFont(str(fonts / name))
    app.setFont(QFont("Segoe UI", 9))

    qss = (ROOT / "src" / "ui" / "theme" / "v2.qss").read_text(encoding="utf-8")
    qss = qss.replace("@ICONS@", (ROOT / "src" / "ui" / "theme" / "icons").as_posix())
    app.setStyleSheet(qss)

    from archeo.src.ui.wizard_dialog import WizardDialog

    dlg = WizardDialog(iface=_StubIface())
    dlg.show()
    app.processEvents()

    bar = dlg._tabs.tabBar()
    ok = True
    for i in range(bar.count()):
        label = bar.tabText(i)
        largeur = bar.tabRect(i).width()
        besoin = QFontMetrics(bar.font()).horizontalAdvance(label)
        # Un onglet gras à la sélection s'élargit : on mesure aussi en gras.
        gras = QFont(bar.font())
        gras.setBold(True)
        besoin_gras = QFontMetrics(gras).horizontalAdvance(label)
        tient = largeur >= besoin_gras + 8
        ok = ok and tient
        print(f"onglet {i} « {label} » : {largeur} px pour {besoin} px de texte "
              f"({besoin_gras} en gras) -> {'OK' if tient else 'ROGNÉ'}")

    dlg.grab().save(str(ROOT / "dialog_onglet1.png"))
    dlg._tabs.setCurrentIndex(1)
    app.processEvents()
    dlg.grab().save(str(ROOT / "dialog_onglet2.png"))

    # le wizard doit rester entier : 4 pages, rail, barre d'actions
    print(f"pages du wizard         : {dlg._stack.count()}")
    print(f"étape courante          : {dlg._current_step}")
    print(f"onglet Visualisation    : {dlg._visu_tab._dept.name if dlg._visu_tab._dept else '—'}, "
          f"{len(dlg._visu_tab._cards)} cartes")

    ok = ok and dlg._stack.count() == 4
    app.exitQgis()
    print("RESULTAT :", "OK" if ok else "ECHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
