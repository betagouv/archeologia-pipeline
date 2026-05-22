"""Contrat d'annulation unique du pipeline.

Toute opération longue qui observe une annulation utilisateur lève
:class:`PipelineCancelled`. Cette exception remonte depuis le subprocess /
la boucle la plus profonde jusqu'à un point de capture unique :

- chaque runner l'attrape autour de son traitement lourd pour enchaîner sur
  une **finalisation légère** (indexation + chargement des résultats partiels) ;
- ``PipelineController`` et le worker UI la capturent en backstop pour signaler
  une annulation **propre** (jamais une erreur).

Module volontairement pur (n'importe que :mod:`pipeline.types`) → testable
hors QGIS.
"""
from __future__ import annotations

from typing import Optional

from .types import CancelCheckFn


class PipelineCancelled(Exception):
    """Levée dès qu'une annulation utilisateur est observée."""


def check_cancelled(cancel: Optional[CancelCheckFn]) -> None:
    """Lève :class:`PipelineCancelled` si ``cancel`` est fourni et renvoie vrai.

    Point de contrôle bon marché à semer dans les boucles longues
    (slices SAHI, clustering, écriture par classe…) et entre sous-étapes.
    """
    if cancel is not None and cancel():
        raise PipelineCancelled()
