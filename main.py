import os
from qgis.PyQt.QtWidgets import QAction
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtCore import QCoreApplication

plugin_dir = os.path.dirname(__file__)

class ArcheologiaPipelinePlugin:
    def __init__(self, iface):
        self.iface = iface
        self.action = None
        self.action_v2 = None  # TEMPORAIRE (refonte V2) — retiré au Jalon 11
        self.dialog = None
        self.dialog_v2 = None

    def initGui(self):
        icon_path = os.path.join(plugin_dir, 'data', 'icon.png')
        self.action = QAction(QIcon(icon_path), self.tr("Archéolog'IA"), self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addPluginToMenu(self.tr("Archéolog'IA"), self.action)
        self.iface.addToolBarIcon(self.action)

        # TEMPORAIRE : point d'entrée du wizard V2 pour test en parallèle de
        # l'UI actuelle. Sera la seule entrée après bascule (Jalon 11).
        self.action_v2 = QAction(QIcon(icon_path), self.tr("Archéolog'IA (V2)"), self.iface.mainWindow())
        self.action_v2.triggered.connect(self.run_v2)
        self.iface.addPluginToMenu(self.tr("Archéolog'IA"), self.action_v2)
        self.iface.addToolBarIcon(self.action_v2)

    def unload(self):
        if self.action is not None:
            self.iface.removeToolBarIcon(self.action)
            self.iface.removePluginMenu(self.tr("Archéolog'IA"), self.action)
            self.action = None
        if self.action_v2 is not None:
            self.iface.removeToolBarIcon(self.action_v2)
            self.iface.removePluginMenu(self.tr("Archéolog'IA"), self.action_v2)
            self.action_v2 = None
        self.dialog = None
        self.dialog_v2 = None

    def run(self):
        from .src.ui.main_dialog import MainDialog

        if self.dialog is None:
            self.dialog = MainDialog(parent=self.iface.mainWindow())
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def run_v2(self):
        from .src.ui.wizard_dialog import WizardDialog

        if self.dialog_v2 is None:
            self.dialog_v2 = WizardDialog(parent=self.iface.mainWindow())
        self.dialog_v2.show()
        self.dialog_v2.raise_()
        self.dialog_v2.activateWindow()

    def tr(self, message):
        return QCoreApplication.translate('ArcheologiaPipelinePlugin', message)