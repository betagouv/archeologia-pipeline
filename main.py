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
        icon_path = os.path.join(plugin_dir, 'data', 'icon.png')
        self.action = QAction(QIcon(icon_path), self.tr("Archéolog'IA"), self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addPluginToMenu(self.tr("Archéolog'IA"), self.action)
        self.iface.addToolBarIcon(self.action)

    def unload(self):
        if self.action is not None:
            self.iface.removeToolBarIcon(self.action)
            self.iface.removePluginMenu(self.tr("Archéolog'IA"), self.action)
            self.action = None
        if self.dialog is not None:
            # Rechargement/désinstallation pendant un run : sans annulation,
            # le thread orphelin continuerait d'écrire avec l'ancien code
            # (AUDIT v2 THR-04). Pas de close() ici → pas de dialogue de
            # confirmation pendant l'unload.
            try:
                self.dialog.request_cancel_if_running()
            except Exception:
                pass
        self.dialog = None

    def run(self):
        from .src.ui.wizard_dialog import WizardDialog

        if self.dialog is None:
            self.dialog = WizardDialog(parent=self.iface.mainWindow())
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def tr(self, message):
        return QCoreApplication.translate('ArcheologiaPipelinePlugin', message)
