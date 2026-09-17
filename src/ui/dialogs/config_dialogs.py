"""Gestion des configurations enregistrées — widgets seulement.

Toute la logique (nommage, écriture, renommage, suppression) vit dans le module
pur ``app/services/config_store.py``, testé par ``tests/unit/test_config_store.py``.
Ici : une liste, trois boutons, et les messages d'erreur.
"""
from __future__ import annotations

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ...app.services.config_store import ConfigStore, InvalidConfigName


def demander_nom(parent, store: ConfigStore, propose: str = "") -> str | None:
    """Demande un nom de configuration, valide, et confirme un écrasement.

    Renvoie le nom retenu, ou ``None`` si l'utilisateur renonce.
    """
    while True:
        nom, ok = QInputDialog.getText(
            parent, "Enregistrer la configuration", "Nom de la configuration :",
            text=propose,
        )
        if not ok:
            return None
        try:
            from ...app.services.config_store import normalize_name
            nom = normalize_name(nom)
        except InvalidConfigName as e:
            QMessageBox.warning(parent, "Nom invalide", str(e))
            propose = nom
            continue
        if store.exists(nom):
            rep = QMessageBox.question(
                parent, "Écraser ?",
                f"« {nom} » existe déjà. Remplacer cette configuration ?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if rep != QMessageBox.StandardButton.Yes:
                propose = nom
                continue
        return nom


class ConfigsDialog(QDialog):
    """Liste des configurations enregistrées + renommer / supprimer."""

    def __init__(self, store: ConfigStore, parent=None):
        super().__init__(parent)
        self._store = store
        self.setWindowTitle("Gérer les configurations")
        self.resize(420, 340)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        chemin = QLabel(str(store.directory))
        chemin.setWordWrap(True)
        chemin.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        chemin.setToolTip("Dossier des configurations enregistrées.")
        root.addWidget(chemin)

        self._list = QListWidget()
        self._list.itemSelectionChanged.connect(self._sync_buttons)
        self._list.itemDoubleClicked.connect(lambda _: self._rename())
        root.addWidget(self._list, 1)

        actions = QHBoxLayout()
        self._rename_btn = QPushButton("Renommer…")
        self._rename_btn.clicked.connect(self._rename)
        self._delete_btn = QPushButton("Supprimer")
        self._delete_btn.clicked.connect(self._delete)
        open_btn = QPushButton("Ouvrir le dossier")
        open_btn.clicked.connect(self._open_folder)
        actions.addWidget(self._rename_btn)
        actions.addWidget(self._delete_btn)
        actions.addStretch(1)
        actions.addWidget(open_btn)
        root.addLayout(actions)

        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(self.reject)
        root.addWidget(box)

        self._refresh()

    # ------------------------------------------------------------------
    def _refresh(self) -> None:
        self._list.clear()
        self._list.addItems(self._store.list_names())
        self._sync_buttons()

    def _selected(self) -> str | None:
        item = self._list.currentItem()
        return item.text() if item is not None and item.isSelected() else None

    def _sync_buttons(self) -> None:
        actif = self._selected() is not None
        self._rename_btn.setEnabled(actif)
        self._delete_btn.setEnabled(actif)

    def _rename(self) -> None:
        ancien = self._selected()
        if ancien is None:
            return
        nouveau, ok = QInputDialog.getText(
            self, "Renommer", "Nouveau nom :", text=ancien
        )
        if not ok:
            return
        try:
            self._store.rename(ancien, nouveau)
        except (InvalidConfigName, FileExistsError, FileNotFoundError, OSError) as e:
            QMessageBox.warning(self, "Renommage impossible", str(e))
            return
        self._refresh()

    def _delete(self) -> None:
        nom = self._selected()
        if nom is None:
            return
        rep = QMessageBox.question(
            self, "Supprimer ?", f"Supprimer définitivement « {nom} » ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if rep != QMessageBox.StandardButton.Yes:
            return
        try:
            self._store.delete(nom)
        except (FileNotFoundError, OSError) as e:
            QMessageBox.warning(self, "Suppression impossible", str(e))
            return
        self._refresh()

    def _open_folder(self) -> None:
        from qgis.PyQt.QtCore import QUrl
        from qgis.PyQt.QtGui import QDesktopServices

        self._store.directory.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._store.directory)))
