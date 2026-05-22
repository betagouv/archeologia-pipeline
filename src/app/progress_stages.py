"""Étapes sémantiques de la timeline, dérivées du mode.

La timeline du ``RunView`` était figée à 5 étapes (« Téléchargement, MNT,
Indices, Détection, Finalisation ») et avançait via un matching texte→bucket
fragile : selon le mode, des pastilles restaient grises (mortes) et le
matching se trompait (« Fusion » allumait « MNT »…).

Ici on définit un **jeu d'IDs sémantiques** émis explicitement par le pipeline
(``reporter.stage_id(...)``) et une fonction qui construit la *séquence
d'étapes applicables à un mode*. La CV n'apparaît que si elle est activée ;
le téléchargement n'apparaît qu'en mode IGN ; « MNT » et « Indices » sont
fusionnés en une seule pastille « Produits » (le calcul MNT+RVT est entrelacé
dalle par dalle, sans frontière réelle).

Module **pur-Python** (aucun import QGIS) → unit-testable hors QGIS.
"""
from __future__ import annotations

from typing import List


class Stage:
    """IDs d'étape (valeurs stables, émises par le pipeline)."""

    DOWNLOAD = "download"
    PRODUCTS = "products"  # MNT + indices RVT (+ fusion en IGN)
    DETECTION = "detection"
    FINALIZE = "finalize"


STAGE_LABELS = {
    Stage.DOWNLOAD: "Téléchargement",
    Stage.PRODUCTS: "Produits",
    Stage.DETECTION: "Détection",
    Stage.FINALIZE: "Finalisation",
}


def build_stage_sequence(mode: str, cv_enabled: bool) -> List[str]:
    """Séquence ordonnée des IDs d'étape applicables au mode.

    - ``ign_laz`` : Téléchargement → Produits (+ Détection si CV) → Finalisation
    - ``local_laz`` / ``existing_mnt`` : Produits (+ Détection si CV) → Finalisation
    - ``existing_rvt`` : Détection → Finalisation (la prep TIF→PNG reste sous
      Détection ; ce mode est fondamentalement de la détection)

    Lève :class:`ValueError` pour un mode inconnu.
    """
    if mode == "ign_laz":
        seq = [Stage.DOWNLOAD, Stage.PRODUCTS]
        if cv_enabled:
            seq.append(Stage.DETECTION)
        seq.append(Stage.FINALIZE)
        return seq

    if mode in ("local_laz", "existing_mnt"):
        seq = [Stage.PRODUCTS]
        if cv_enabled:
            seq.append(Stage.DETECTION)
        seq.append(Stage.FINALIZE)
        return seq

    if mode == "existing_rvt":
        return [Stage.DETECTION, Stage.FINALIZE]

    raise ValueError(f"Mode inconnu pour la séquence d'étapes : {mode!r}")
