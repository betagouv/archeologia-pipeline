"""Contours de la « Vérité terrain » d'une vignette dans la couleur de la classe — module PUR (numpy).

Les vignettes de fiche vont par paire : ``*_brut`` (le relief seul, en niveaux de gris)
et ``*_annote`` (le même cadre, contours de la vérité terrain dessinés en jaune). La
couleur d'une classe dépend du registre de couleurs du profil QGIS : elle ne peut pas
être cuite dans les fichiers livrés. On recolore donc à l'affichage (demande
utilisateur 2026-10-08) : le relief est gris (R = G = B), le tracé est la seule chose
colorée de l'image annotée ; sa **saturation** (max − min des canaux) donne, pixel par
pixel, la part de tracé — anticrénelage et flou JPEG compris — et l'on mélange le
relief brut avec la couleur de la classe dans cette proportion.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

#: Saturation sous laquelle un pixel est du relief (bruit de chrominance JPEG) ;
#: au-delà de SEUIL + PLEINE, c'est du tracé plein.
SEUIL = 14.0
PLEINE = 70.0      # un trait fin (saturation ~ 90 après JPEG) reste en couleur pleine


def part_de_trace(annote: np.ndarray) -> np.ndarray:
    """Part de tracé par pixel, de 0 (relief) à 1 (contour plein), depuis l'image
    annotée ``(h, w, 3)`` en RGB 0–255."""
    a = annote.astype(np.float32)
    sat = a.max(axis=2) - a.min(axis=2)
    return np.clip((sat - SEUIL) / PLEINE, 0.0, 1.0)


def recolorer(brut: np.ndarray, annote: np.ndarray, rgb: Sequence[int]) -> np.ndarray:
    """L'image annotée, contours dans la couleur ``rgb`` : ``brut`` × (1 − t) + ``rgb`` × t,
    ``t`` = part de tracé. Tailles différentes → l'image annotée telle quelle."""
    if brut.shape != annote.shape or brut.ndim != 3 or brut.shape[2] < 3:
        return annote
    t = part_de_trace(annote[..., :3])[..., None]
    couleur = np.asarray(rgb, dtype=np.float32)[:3]
    out = brut[..., :3].astype(np.float32) * (1.0 - t) + couleur * t
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
