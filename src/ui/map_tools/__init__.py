"""Outils-carte QGIS du plugin (premiers ``QgsMapTool`` du projet).

Importé uniquement en contexte QGIS (jamais collecté par pytest, cf. conftest qui
ignore ``src/ui/*``). Les modules ici peuvent donc importer ``qgis.gui`` /
``qgis.core`` au niveau module (un ``QgsMapTool`` a besoin de sa classe de base
dès la définition de classe).
"""
