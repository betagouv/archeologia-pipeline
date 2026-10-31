"""Coût structurel d'un modèle de détection : fenêtres d'analyse par dalle.

Ce que l'étape 3 peut dire du temps de calcul d'un modèle SANS rien inventer
(règle du projet : jamais d'estimation de durée dans l'UI) : le nombre de
fenêtres SAHI que le modèle analyse sur une dalle, calculé depuis le découpage
d'``args.yaml`` par la fonction même que le binaire exécute. C'est un FAIT ;
la durée réelle, elle, est chronométrée après coup (``UserNarrator.cv_run_done``).

Mesuré sur 75 journaux (2026-09-21) : le temps PAR FENÊTRE est quasi constant
d'un modèle à l'autre (≈ 0,8–1,2 s sur un Ryzen 5 sans GPU) et le temps par
image est linéaire au nombre de fenêtres — le compte de fenêtres ordonne donc
bien les modèles entre eux, sans qu'on affiche jamais un ratio ni des minutes.

Deux tailles d'image sont données partout, parce que l'ordre entre modèles à
648 et 672 px bascule entre elles (16 et 16 à 2 000 px, 36 et 25 à 2 800 px).

Module PUR (aucun import Qt ni QGIS) ; ``sahi_lite`` est importé en différé
(numpy seul, ~180 ms hors QGIS, sans shapely).
"""
from __future__ import annotations

from typing import Optional

# Dalle IGN de 1 km à 0,5 m/px, sans marge : la référence lisible par tous.
COTE_DALLE_PX = 2000
# La même dalle avec la marge inter-dalles par défaut (tile_overlap 20 %) des
# modes ign_laz/local_laz — c'est l'image réellement analysée dans ces modes.
COTE_DALLE_MARGE_PX = 2800

_ESPACE_FINE = " "  # séparateur de milliers français


def fenetres_par_dalle(slice_px: int, overlap: float, cote_px: int = COTE_DALLE_PX) -> int:
    """Nombre de fenêtres SAHI sur une image carrée de ``cote_px``.

    0 si la fenêtre est inconnue (``slice_px`` ≤ 0) : « rien à dire » plutôt
    qu'un chiffre par défaut. Déléguée à ``get_slice_bboxes`` (dédoublonnage des
    fenêtres de bord compris) : une réplique se tromperait de 44 % à 672/2 800.
    """
    try:
        slice_px = int(slice_px)
        cote_px = int(cote_px)
        overlap = float(overlap)
    except (TypeError, ValueError):
        return 0
    if slice_px <= 0 or cote_px <= 0 or not (0.0 <= overlap < 1.0):
        return 0
    try:
        from ...pipeline.cv.sahi_lite import get_slice_bboxes
    except ImportError:
        from pipeline.cv.sahi_lite import get_slice_bboxes  # type: ignore[no-redef]
    return len(get_slice_bboxes(cote_px, cote_px, slice_px, slice_px, overlap, overlap))


def _milliers(n: int) -> str:
    return f"{n:,}".replace(",", _ESPACE_FINE)


def _pct(overlap: float) -> str:
    return f"{int(round(float(overlap) * 100))} %"


#: Côté de la dalle IGN de 1 km et de la même dalle avec la marge inter-dalles par défaut (20 %), en mètres.
COTE_DALLE_M = 1000
COTE_DALLE_MARGE_M = 1400


def _echelle(slice_px: int, gsd_m: Optional[float], resolution_m: Optional[float]):
    """(fenêtre en px que le run découpera, côté de la dalle, côté avec marge, résolution, note) — règle du
    2026-10-31 : la fenêtre d'args.yaml est mise à l'échelle des données (model_config.sahi_a_l_echelle, la
    fonction même du run). Sans GSD d'entraînement connu, les références IGN à 0,5 m d'avant ; sans résolution
    des données connue, 0,5 m, et la note le dit."""
    if not gsd_m:
        return int(slice_px), COTE_DALLE_PX, COTE_DALLE_MARGE_PX, 0.5, ""
    note = "" if resolution_m else " — à 0,5 m/pixel, la résolution des données pas encore connue"
    res = float(resolution_m or 0.5)
    try:
        from ...pipeline.cv.model_config import sahi_a_l_echelle
    except ImportError:
        from pipeline.cv.model_config import sahi_a_l_echelle  # type: ignore[no-redef]
    fen = sahi_a_l_echelle({"slice_width": int(slice_px), "slice_height": int(slice_px)}, gsd_m, res)["slice_width"]
    return fen, int(round(COTE_DALLE_M / res)), int(round(COTE_DALLE_MARGE_M / res)), res, note


def _res(res: float) -> str:
    return f"{res:g}".replace(".", ",") + " m"


def libelle_menu(display_name: str, slice_px: int, overlap: float, *, gsd_m: Optional[float] = None,
                 resolution_m: Optional[float] = None) -> str:
    """Entrée du menu « Changer ▾ » : le nom, puis le compte s'il est connu (fenêtre mise à l'échelle des
    données quand le GSD d'entraînement du modèle est connu)."""
    fen, cote, _, _, _ = _echelle(slice_px, gsd_m, resolution_m)
    n = fenetres_par_dalle(fen, overlap, cote) if int(slice_px or 0) > 0 else 0
    if n <= 0:
        return display_name
    return f"{display_name} — {n} fenêtres d'analyse par dalle"


def ligne_dialogue(slice_px: int, overlap: float, *, gsd_m: Optional[float] = None,
                   resolution_m: Optional[float] = None) -> str:
    """Valeur de la Row « Fenêtres d'analyse » du dialogue du modèle."""
    if int(slice_px or 0) <= 0:
        return ""
    fen, cote, cote_marge, res, note = _echelle(slice_px, gsd_m, resolution_m)
    n = fenetres_par_dalle(fen, overlap, cote)
    if n <= 0:
        return ""
    n_marge = fenetres_par_dalle(fen, overlap, cote_marge)
    terrain = f" ({fen * res:g} m de terrain)".replace(".", ",") if gsd_m else ""
    return (
        f"{_milliers(fen)} px{terrain}, recouvrement {_pct(overlap)} → {n} fenêtres par dalle de 1 km "
        f"({_milliers(cote)} px à {_res(res)}), {n_marge} avec la marge inter-dalles{note}"
    )


def infobulle(display_name: str, slice_px: int, overlap: float, *, gsd_m: Optional[float] = None,
              resolution_m: Optional[float] = None) -> str:
    """Infobulle du nom de modèle (étape 3) : le fait, en phrase entière, et le
    refus explicite d'annoncer une durée. Vide si la fenêtre est inconnue."""
    if int(slice_px or 0) <= 0:
        return ""
    fen, cote, cote_marge, _, note = _echelle(slice_px, gsd_m, resolution_m)
    n = fenetres_par_dalle(fen, overlap, cote)
    if n <= 0:
        return ""
    n_marge = fenetres_par_dalle(fen, overlap, cote_marge)
    return (
        f"{display_name} : analyse une dalle de 1 km en {n} fenêtres de {_milliers(fen)} px "
        f"({n_marge} avec la marge inter-dalles de l'étape 2){note}. À recouvrement égal, "
        "deux fois plus de fenêtres = deux fois plus de calcul. "
        "Aucune durée n'est annoncée : celle de chaque analyse est dans le journal."
    )
