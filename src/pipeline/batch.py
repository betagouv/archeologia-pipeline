"""Itération résiliente sur un lot d'éléments.

Isole les erreurs **par élément** : une exception levée pour un élément
n'interrompt pas le traitement des autres (sauf annulation). C'est le socle
commun des correctifs de robustesse des boucles dalle/MNT/run-CV
(cf. AUDIT ROB-02/03/04), qui avant cela laissaient une seule erreur
avorter tout le lot.

Module volontairement **pur** (n'importe que :mod:`pipeline.cancellation`
et :mod:`pipeline.types`) → testable hors QGIS.
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Tuple, TypeVar

from .cancellation import PipelineCancelled
from .types import CancelCheckFn

T = TypeVar("T")


def process_items_isolated(
    items: Iterable[T],
    process: Callable[[int, T], None],
    *,
    cancel: Optional[CancelCheckFn] = None,
    on_failure: Optional[Callable[[int, T, Exception], None]] = None,
) -> Tuple[int, List[Tuple[int, T]]]:
    """Traite chaque élément de ``items`` via ``process(index, item)``.

    - ``index`` est **1-based** (1 pour le premier élément), pour des
      messages de progression lisibles (« dalle 3/10 »).
    - Avant chaque élément, si ``cancel`` est fourni et renvoie vrai, la
      boucle s'arrête proprement (court-circuit, **pas** une erreur).
    - ``PipelineCancelled`` levée par ``process`` est **propagée** telle
      quelle (l'annulation reste globale, jamais isolée).
    - Toute autre exception est **isolée** : ``on_failure(index, item, exc)``
      est appelé (si fourni) puis la boucle continue avec l'élément suivant.
      ``KeyboardInterrupt``/``SystemExit`` (``BaseException``) ne sont pas
      capturés.

    Returns:
        ``(nb_succes, echecs)`` où ``nb_succes`` est le nombre d'éléments
        traités sans exception et ``echecs`` la liste des ``(index, item)``
        ayant levé une exception (hors annulation), dans l'ordre.
    """
    succeeded = 0
    failures: List[Tuple[int, T]] = []
    for index, item in enumerate(items, start=1):
        if cancel is not None and cancel():
            break
        try:
            process(index, item)
        except PipelineCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 — isolation volontaire par élément
            failures.append((index, item))
            if on_failure is not None:
                on_failure(index, item, exc)
            continue
        succeeded += 1
    return succeeded, failures
