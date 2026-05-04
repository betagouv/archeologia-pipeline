"""Contrat de reporting du pipeline vers l'UI.

Le reporter expose **deux canaux** distincts :

- **Canal technique** (``info`` / ``error``) : tous les logs détaillés
  émis par le pipeline (commandes PDAL/GDAL, paramètres RVT, lambdas
  internes…). Destinés au fichier de log pour le debug.
- **Canal narratif** (``user_info`` / ``user_warning`` / ``user_success``) :
  messages courts et clairs destinés à l'utilisateur final non expert.
  Ces messages sont émis avec un niveau de log ``USER_INFO`` supérieur
  à ``INFO``, ce qui permet à l'UI de les filtrer indépendamment du
  fichier de log.

Côté implémentation Qt, ``QtLogHandler`` filtre les enregistrements pour
ne propager à la fenêtre QGIS que ce qui est >= ``USER_INFO`` ; le
``FileHandler`` reçoit tout sans filtre.
"""
from __future__ import annotations

import logging
from typing import Protocol

# Niveau de log dédié aux messages narratifs adressés à l'utilisateur.
# Compris entre INFO (20) et WARNING (30) pour que la chaîne logging
# standard (root, archeologia_pipeline) le route correctement, et pour
# qu'un simple ``setLevel(USER_INFO)`` sur un handler suffise à filtrer.
USER_INFO = 25
logging.addLevelName(USER_INFO, "USER_INFO")


class ProgressReporter(Protocol):
    # Canal technique (file-only après filtrage UI)
    def info(self, msg: str) -> None: ...

    def error(self, msg: str) -> None: ...

    # Canal narratif (visible UI + fichier)
    def user_info(self, msg: str) -> None: ...

    def user_warning(self, msg: str) -> None: ...

    def user_success(self, msg: str) -> None: ...

    # Stage / progress / load_layers : pas de duplication, ils sont déjà
    # destinés à l'UI (barre de progression / label).
    def stage(self, msg: str) -> None: ...

    def progress(self, pct: int) -> None: ...

    def load_layers(self, vrt_paths: list, shapefile_paths: list, class_colors: list = None) -> None: ...


class NullProgressReporter:
    def info(self, msg: str) -> None:
        return

    def error(self, msg: str) -> None:
        return

    def user_info(self, msg: str) -> None:
        return

    def user_warning(self, msg: str) -> None:
        return

    def user_success(self, msg: str) -> None:
        return

    def stage(self, msg: str) -> None:
        return

    def progress(self, pct: int) -> None:
        return

    def load_layers(self, vrt_paths: list, shapefile_paths: list, class_colors: list = None) -> None:
        return
