"""Plan de progression 0–100 % partagé par les 4 runners + 2 services.

Avant ce module, la pondération phase→pourcentage était dispersée et
incohérente : ``IgnDownloadStrategy`` portait des plages codées en dur,
``cv_post_service`` un ``base_progress`` « historique » (90/80) qui faisait
*reculer* la barre quand la CV démarrait, et ``existing_mnt`` / ``existing_rvt``
n'émettaient aucune progression pendant le gros du travail.

``ProgressPlan`` centralise la cadence par mode sous forme de **bandes
contiguës et monotones** : chaque phase occupe une plage ``[lo, hi]``, et la
phase suivante démarre exactement là où la précédente finit. Une phase mappe
sa fraction interne 0→1 dans sa bande via :meth:`ProgressPlan.at`. La barre ne
peut donc plus reculer entre phases.

Module **pur-Python** (aucun import QGIS/pipeline) → unit-testable hors QGIS
(``conftest.py`` exclut ``src/ui`` et ``src/pipeline`` de la collecte).

Les pourcentages sont une **pondération de cadence**, pas une estimation de
durée : ils règlent le *rythme visuel* de remplissage, pas un temps annoncé.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

Band = Tuple[int, int]


@dataclass(frozen=True)
class ProgressPlan:
    """Bandes de progression d'un run, dans l'ordre chronologique.

    Toutes les bandes sont contiguës (``download[1] == merge[0]`` …) et
    croissantes ; ``finalize[1] == 100``. Une bande peut être *dégénérée*
    (``lo == hi``) quand la phase n'existe pas pour le mode/la config
    (ex. ``download``/``merge`` en mode local, ``cv`` quand la CV est
    désactivée) : la phase suivante reprend alors sans trou ni recul.
    """

    download: Band
    merge: Band
    products: Band
    cv: Band
    finalize: Band

    def at(self, band: Band, frac: float) -> int:
        """Pourcentage à afficher pour une ``frac`` (0→1) dans ``band``.

        ``frac`` est bornée à ``[0, 1]`` : un appelant qui passe ``i/total``
        avec ``i > total`` (arrondi) reste dans la bande.
        """
        lo, hi = band
        f = max(0.0, min(1.0, float(frac)))
        return int(round(lo + (hi - lo) * f))


def build_progress_plan(mode: str, cv_enabled: bool) -> ProgressPlan:
    """Retourne le plan de bandes correspondant au mode + activation CV.

    Lève :class:`ValueError` pour un mode inconnu.
    """
    if mode == "ign_laz":
        if cv_enabled:
            return ProgressPlan((0, 20), (20, 30), (30, 75), (75, 95), (95, 100))
        return ProgressPlan((0, 20), (20, 30), (30, 95), (95, 95), (95, 100))

    if mode == "local_laz":
        if cv_enabled:
            return ProgressPlan((0, 0), (0, 0), (0, 75), (75, 95), (95, 100))
        return ProgressPlan((0, 0), (0, 0), (0, 95), (95, 95), (95, 100))

    if mode == "existing_mnt":
        if cv_enabled:
            return ProgressPlan((0, 0), (0, 0), (0, 70), (70, 95), (95, 100))
        return ProgressPlan((0, 0), (0, 0), (0, 95), (95, 95), (95, 100))

    if mode == "existing_rvt":
        # products = préparation TIF→PNG (rapide), cv = inférence (dominant).
        return ProgressPlan((0, 0), (0, 0), (0, 10), (10, 95), (95, 100))

    raise ValueError(f"Mode inconnu pour le plan de progression : {mode!r}")


def cv_pct(run_idx: int, n_runs: int, idx: int, total: int, band: Band) -> int:
    """Pourcentage pendant la CV multi-runs, interpolé dans ``band``.

    La bande CV est répartie équitablement entre les ``n_runs`` runs
    séquentiels ; à l'intérieur d'un run, on interpole sur les images
    ``idx``/``total``. Le résultat est monotone (croissant) au fil des images
    *et* des runs, et borné dans ``band``.

    Args:
        run_idx: index du run courant (1-indexé).
        n_runs: nombre total de runs CV.
        idx: index de l'image courante dans le run (1-indexé en pratique).
        total: nombre d'images du run.
        band: bande CV ``(lo, hi)`` du plan.
    """
    lo, hi = band
    n = max(1, int(n_runs))
    r = max(1, min(int(run_idx), n))
    run_lo = lo + (hi - lo) * (r - 1) / n
    run_hi = lo + (hi - lo) * r / n
    frac = max(0.0, min(1.0, int(idx) / max(1, int(total))))
    val = run_lo + (run_hi - run_lo) * frac
    return int(round(max(lo, min(hi, val))))
