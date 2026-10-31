"""Réglages spatiaux des indices en MÈTRES — conversion vers les pixels du MNT réel (règle 2026-10-31).

**Pourquoi.** Le relief se juge en mètres : un anneau de dominance locale de 5 à 10 m reste le même objet
quelle que soit la résolution du MNT. Jusqu'à la 0.18 les rayons étaient réglés en pixels, calibrés sur le
LiDAR HD de l'IGN (0,5 m) : à 1 m (DGM1 allemand, accepté depuis le 2026-10-31) l'anneau faisait 10 à 20 m,
ce que les modèles n'ont jamais vu (ils ont tous appris sur l'anneau physique 5–10 m de
``training-models/tools/generer_ld.py``).

**Contrat.**

- La configuration (``rvt_params``) porte les rayons en mètres sous des clés ``<clé>_m`` (``radius_m``,
  ``min_radius_m``…) ; les anciennes clés en pixels n'y sont plus écrites.
- :func:`en_px` rend, pour la résolution RÉELLE du run, une copie où chaque clé pixel historique est posée
  depuis sa clé métrique (``radius = px_de_m(radius_m, résolution)``), plus ``resolution_m``. Tout le
  pipeline en aval (calcul, nom des dossiers, contrôles de noyau) lit cette copie, en pixels, comme avant :
  à 0,5 m les valeurs, les noms de dossiers et les images sont ceux d'avant, au bit près.
- Défauts en mètres = anciens défauts en pixels × 0,5 m.
- :func:`migrer_px_vers_m` convertit une configuration enregistrée en pixels (0,5 m/px supposé : c'est sur
  l'IGN que ces valeurs avaient été choisies), une seule fois (la clé métrique présente, rien ne bouge),
  et le dit (messages + ``migration_px_vers_m`` dans la configuration).

Pur : aucun import QGIS ni numpy, testé par ``tests/unit/test_indices_metres.py``.
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

#: Résolution sur laquelle les anciens réglages en pixels avaient été calibrés (LiDAR HD IGN).
RESOLUTION_REFERENCE_M = 0.5

#: (produit, section, clé pixel historique, clé métrique, défaut en m, libellé de l'étape 2).
REGLAGES_SPATIAUX: Tuple[Tuple[str, str, str, str, float, str], ...] = (
    ("SVF", "svf", "radius", "radius_m", 5.0, "Rayon"),
    ("OPNS", "opns", "radius", "radius_m", 5.0, "Rayon"),
    ("LD", "ldo", "min_radius", "min_radius_m", 5.0, "Rayon min"),
    ("LD", "ldo", "max_radius", "max_radius_m", 10.0, "Rayon max"),
    ("SLRM", "slrm", "radius", "radius_m", 10.0, "Rayon"),
    ("MSTP", "mstp", "local_scale_min", "local_scale_min_m", 1.5, "Échelle locale — rayon min"),
    ("MSTP", "mstp", "local_scale_max", "local_scale_max_m", 10.5, "Échelle locale — rayon max"),
    ("MSTP", "mstp", "local_scale_step", "local_scale_step_m", 1.0, "Échelle locale — pas"),
    ("MSTP", "mstp", "meso_scale_min", "meso_scale_min_m", 11.5, "Échelle méso — rayon min"),
    ("MSTP", "mstp", "meso_scale_max", "meso_scale_max_m", 101.5, "Échelle méso — rayon max"),
    ("MSTP", "mstp", "meso_scale_step", "meso_scale_step_m", 9.0, "Échelle méso — pas"),
    ("MSTP", "mstp", "broad_scale_min", "broad_scale_min_m", 50.0, "Échelle large — rayon min"),
    ("MSTP", "mstp", "broad_scale_max", "broad_scale_max_m", 200.0, "Échelle large — rayon max"),
    ("MSTP", "mstp", "broad_scale_step", "broad_scale_step_m", 30.0, "Échelle large — pas"),
)

#: Bornes des champs en mètres à l'étape 2 = anciennes bornes en pixels × 0,5 m (France inchangée). Au-delà
#: de la résolution de l'IGN, la conversion peut sortir des bornes de rvt-qgis : contrôlé par
#: ``rvt_kernel_context.bornes_errors`` à la résolution du run, jamais en silence.
BORNES_M: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("svf", "radius_m"): (5.0, 25.0),
    ("opns", "radius_m"): (5.0, 25.0),
    ("ldo", "min_radius_m"): (0.5, 50.0),
    ("ldo", "max_radius_m"): (0.5, 50.0),
    ("slrm", "radius_m"): (5.0, 25.0),
    **{("mstp", f"{e}_scale_{k}_m"): (0.5, 50000.0) for e in ("local", "meso", "broad") for k in ("min", "max", "step")},
}


def _nombre(valeur: Any) -> Optional[float]:
    try:
        return float(str(valeur).replace(",", "."))
    except (TypeError, ValueError):
        return None


def px_de_m(metres: float, resolution_m: float) -> int:
    """Pixels d'une distance en mètres à la résolution ``resolution_m`` : arrondi au plus proche, demi
    vers le haut (pas l'arrondi bancaire : 2,5 px → 3), jamais moins de 1 px. Exact à 0,5 m pour tout
    multiple de 0,5 m (les défauts). Le ``1e-9`` absorbe le bruit flottant (0,3 / 0,1 = 2,999…)."""
    res = float(resolution_m)
    if res <= 0:
        res = RESOLUTION_REFERENCE_M
    return max(1, int(math.floor(float(metres) / res + 0.5 + 1e-9)))


def valeur_m(rvt_params: Optional[Mapping], section: str, cle_m: str, defaut_m: float) -> float:
    """Valeur en mètres d'un réglage spatial (défaut si absente ou illisible)."""
    sec = (rvt_params or {}).get(section)
    v = _nombre(sec.get(cle_m)) if isinstance(sec, Mapping) and cle_m in sec else None
    return float(defaut_m) if v is None else v


def en_px(rvt_params: Optional[Mapping], resolution_m: float) -> Dict[str, Any]:
    """Copie de ``rvt_params`` où chaque réglage spatial porte sa valeur en PIXELS sous sa clé historique,
    à ``resolution_m`` (la résolution réelle du MNT du run) ; ``resolution_m`` y est noté. Les clés
    métriques restent (trace, rapport). Idempotent. Une configuration pas encore migrée (clés pixels sans
    clé métrique) est d'abord migrée (0,5 m/px supposé)."""
    out: Dict[str, Any] = copy.deepcopy(dict(rvt_params or {}))
    migrer_px_vers_m(out)
    res = float(resolution_m) if _nombre(resolution_m) and float(resolution_m) > 0 else RESOLUTION_REFERENCE_M
    for _p, section, cle_px, cle_m, defaut, _lib in REGLAGES_SPATIAUX:
        sec = out.get(section)
        if not isinstance(sec, dict):
            sec = out[section] = {}
        sec[cle_px] = px_de_m(valeur_m(out, section, cle_m, defaut), res)
    out["resolution_m"] = res
    return out


def resolution_de(rvt_params: Optional[Mapping]) -> float:
    """Résolution notée par :func:`en_px` (0,5 m sans note : réglages d'avant, en pixels de l'IGN)."""
    v = _nombre((rvt_params or {}).get("resolution_m"))
    return v if v and v > 0 else RESOLUTION_REFERENCE_M


def migrer_px_vers_m(rvt_params: Optional[Dict[str, Any]]) -> List[str]:
    """Convertit en mètres, en place, les réglages spatiaux encore en pixels (configuration enregistrée
    avant la règle du 2026-10-31) : ``<clé>_m = <clé> × 0,5 m`` (résolution de l'IGN, sur laquelle ils
    avaient été choisis), la clé pixel retirée. Une clé déjà en mètres n'est jamais reconvertie (la clé
    pixel éventuelle, écrite par une ancienne version, est alors retirée sans être lue). Rend un message
    par valeur convertie et note la conversion dans ``rvt_params["migration_px_vers_m"]``."""
    if not isinstance(rvt_params, dict):
        return []
    messages: List[str] = []
    for produit, section, cle_px, cle_m, _d, libelle in REGLAGES_SPATIAUX:
        sec = rvt_params.get(section)
        if not isinstance(sec, dict) or cle_px not in sec:
            continue
        brut = sec.pop(cle_px)
        if cle_m in sec:
            continue
        v = _nombre(brut)
        if v is None:
            continue
        sec[cle_m] = v * RESOLUTION_REFERENCE_M
        messages.append(f"{produit.replace('_', '-')} / « {libelle} » : {v:g} px enregistrés en pixels, "
                        f"convertis en {format_m(sec[cle_m])} (0,5 m/px supposé, résolution du LiDAR HD IGN).")
    if messages:
        rvt_params["migration_px_vers_m"] = ("Réglages en pixels convertis en mètres (0,5 m/px supposé) : "
                                             + " ; ".join(messages))
    return messages


def format_m(metres: float) -> str:
    """« 5 m », « 10,5 m », « 0,5 m »."""
    s = f"{float(metres):.2f}".rstrip("0").rstrip(".")
    return s.replace(".", ",") + " m"


def format_m_px(metres: float, resolution_m: float) -> str:
    """« 5 m (10 px à 0,5 m/px) » : la valeur réglée et ce que le calcul a reçu."""
    return (f"{format_m(metres)} ({px_de_m(metres, resolution_m)} px à "
            f"{format_m(resolution_m)[:-2]} m/px)")


# ---------------------------------------------------------------------------
# Anneau de dominance locale d'un modèle (fiches training_params.json / model_card preferred_rvt)
# ---------------------------------------------------------------------------

def gsd_de_fiche(params: Optional[Mapping], training_params: Optional[Mapping] = None) -> Optional[float]:
    """GSD d'entraînement (m/px) d'un modèle — SOURCE UNIQUE (fenêtre de détection, orchestrateur, validateur,
    harmonisation ; audit 2026-10-31) : ``gsd_m`` des paramètres d'indice donnés, sinon
    ``training_params.rvt.params.gsd_m``, sinon ``mnt.resolution`` (ou ``resolution``) de ``training_params.json``.
    ``None`` si inconnu. Le validateur refuse un ``gsd_m`` qui contredit ``mnt.resolution``."""
    tp = training_params or {}
    rvt = tp.get("rvt") if isinstance(tp.get("rvt"), Mapping) else {}
    tp_params = rvt.get("params") if isinstance(rvt.get("params"), Mapping) else {}
    mnt = tp.get("mnt") if isinstance(tp.get("mnt"), Mapping) else {}
    for v in ((params or {}).get("gsd_m"), tp_params.get("gsd_m"), mnt.get("resolution"), tp.get("resolution")):
        n = _nombre(v)
        if n and n > 0:
            return n
    return None


def gsd_du_modele(chemin) -> Optional[float]:
    """GSD d'entraînement d'un modèle installé, depuis son dossier ou ses poids (``<m>/weights/best.onnx``) :
    :func:`gsd_de_fiche` sur son ``training_params.json``. ``None`` sans fiche lisible."""
    import json
    from pathlib import Path
    p = Path(str(chemin))
    dossier = p.parent.parent if p.suffix and p.parent.name == "weights" else (p.parent if p.suffix else p)
    try:
        tp = json.loads((dossier / "training_params.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return gsd_de_fiche(None, tp if isinstance(tp, dict) else {})


def anneau_px_du_nom(texte: str) -> Optional[Tuple[int, int]]:
    """Anneau (pixels) d'un LD produit par le plugin, lu dans son nom de dossier ou de fichier
    (``LD_A15_Rmin10_Rmax20_H1p7_V1``) ; ``None`` sans ce motif."""
    import re
    m = re.search(r"LD_A\d+_Rmin(\d+)_Rmax(\d+)", str(texte))
    return (int(m.group(1)), int(m.group(2))) if m else None


def anneau_m(params: Optional[Mapping], gsd_m: Optional[float]) -> Optional[Tuple[float, float]]:
    """Anneau (rayon min, rayon max) EN MÈTRES des paramètres d'un LD, quelle que soit leur écriture :
    ``rmin_m``/``rmax_m`` (forme harmonisée du 2026-10-31), ``rmin_px``/``rmax_px`` ou
    ``min_radius``/``max_radius`` (pixels, convertis par ``gsd_m``). ``None`` si indécidable."""
    p = params or {}
    lo, hi = _nombre(p.get("rmin_m")), _nombre(p.get("rmax_m"))
    if lo is not None and hi is not None:
        return (lo, hi)
    for a, b in (("rmin_px", "rmax_px"), ("min_radius", "max_radius")):
        lo, hi = _nombre(p.get(a)), _nombre(p.get(b))
        if lo is not None and hi is not None and gsd_m:
            return (round(lo * gsd_m, 6), round(hi * gsd_m, 6))
    return None
