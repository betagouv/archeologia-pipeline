import os
from qgis.PyQt.QtWidgets import QAction
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtCore import QCoreApplication

plugin_dir = os.path.dirname(__file__)

class ArcheologiaPipelinePlugin:
    def __init__(self, iface):
        self.iface = iface
        self.action = None
        self.dialog = None

    def initGui(self):
        icon_path = os.path.join(plugin_dir, 'icon.png')
        self.action = QAction(QIcon(icon_path), self.tr("Archeolog'IA pipeline"), self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addPluginToMenu(self.tr("Archeolog'IA pipeline"), self.action)
        self.iface.addToolBarIcon(self.action)

    def unload(self):
        if self.action is not None:
            self.iface.removeToolBarIcon(self.action)
            self.iface.removePluginMenu(self.tr("Archeolog'IA pipeline"), self.action)
            self.action = None
        self.dialog = None

    def run(self):
        from .src.ui.main_dialog import MainDialog

        if self.dialog is None:
            self.dialog = MainDialog(parent=self.iface.mainWindow())
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def tr(self, message):
        return QCoreApplication.translate('ArcheologiaPipelinePlugin', message)