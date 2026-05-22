"""Service partagé pour le lancement de la Computer Vision post-traitement.

Avant ce module, la boucle « pour chaque run CV configuré, résoudre le
RVT/TIF correspondant et lancer ``run_existing_rvt`` sur ce dossier »
était dupliquée presque mot pour mot entre :

- ``IgnOrLocalRunner._run_post_cv`` (ign_laz / local_laz)
- ``ExistingMntRunner.run`` (inline, mode existing_mnt)

Cela imposait de modifier la même logique à 2 endroits pour la moindre
évolution. ``ExistingRvtRunner`` n'est *pas* concerné : il consomme un
RVT externe fourni par l'utilisateur, pas un RVT généré par le
pipeline, et a donc une logique différente.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from ..progress_reporter import report_busy, report_stage_id
from ..progress_stages import Stage
from ..runners.progress_plan import cv_pct
from ..structured_logger import log_section
from ..user_narrator import create_user_narrator


def _model_display_name(selected_model: str) -> str:
    """Nom court d'un modèle à partir de la valeur ``selected_model``.

    Accepte aussi bien un nom de dossier (``mon_modele``) qu'un chemin
    complet vers ``best.onnx`` (``…/models/mon_modele/weights/best.onnx``).
    """
    p = Path(selected_model)
    if p.suffix.lower() == ".onnx" and p.parent.name == "weights":
        return p.parent.parent.name or selected_model
    return selected_model

if TYPE_CHECKING:
    from ..cancel_token import CancelToken
    from ..progress_reporter import ProgressReporter
    from ..run_context import RunContext
    from ..structured_logger import StructuredLogger


def run_cv_post_loop(
    *,
    ctx: "RunContext",
    output_structure: Dict[str, Any],
    rvt_params: Dict[str, Any],
    reporter: "ProgressReporter",
    cancel: "CancelToken",
    slog: Optional["StructuredLogger"],
    cv_band: Tuple[int, int] = (90, 95),
) -> None:
    """Lance la Computer Vision sur les RVT générés par le pipeline.

    Pour chaque ``cv_run`` configuré dans ``ctx.cv.runs`` :

    1. Résout ``indices/<RVT>/tif`` via :func:`resolve_rvt_tif_dir`.
    2. Construit (au premier run uniquement) le mapping global de
       couleurs partagé par tous les runs.
    3. Délègue à :func:`pipeline.modes.existing_rvt.run_existing_rvt`.

    Args:
        cv_band: Bande de progression ``(lo, hi)`` allouée à la CV par le
            :class:`~app.runners.progress_plan.ProgressPlan` du mode. La barre
            démarre à ``lo`` et progresse jusqu'à ``hi`` au fil des images
            (réparties équitablement entre les runs via :func:`cv_pct`) — plus
            de recul de la barre au démarrage de la CV.

    Cette fonction n'attrape pas les exceptions : l'appelant décide de
    sa politique (le pattern actuel est de logger via ``reporter.error``
    et de continuer vers la finalisation).
    """
    from ...pipeline.cv.class_utils import resolve_cv_runs
    from ...pipeline.modes.existing_rvt import run_existing_rvt
    from ...pipeline.output_paths import resolve_rvt_tif_dir

    from .finalize_service import _build_global_class_color_map

    cv_cfg = ctx.cv.raw
    cv_runs = resolve_cv_runs(cv_cfg)
    if not cv_runs:
        reporter.info("Computer Vision: aucun modèle configuré dans les runs")
        return

    global_color_map: Dict[str, int] = {}
    try:
        global_color_map = _build_global_class_color_map(cv_runs)
        reporter.info(f"Computer Vision: mapping couleurs global = {global_color_map}")
    except Exception as _e:  # noqa: BLE001 — on logge et continue
        reporter.info(f"Computer Vision: impossible de construire le mapping couleurs: {_e}")

    log_section("COMPUTER VISION", "cv", slog=slog, reporter=reporter)
    report_stage_id(reporter, Stage.DETECTION)
    reporter.stage("Computer Vision")
    reporter.progress(cv_band[0])

    narrator = create_user_narrator(reporter)
    narrator.cv_start(len(cv_runs))

    for run_idx, run_cfg in enumerate(cv_runs, start=1):
        if cancel.is_cancelled():
            break

        run_model = run_cfg.get("selected_model", "?")
        run_rvt = run_cfg.get("target_rvt", "LD")
        model_display = _model_display_name(run_model)
        reporter.info(
            f"Computer Vision: run {run_idx}/{len(cv_runs)} — "
            f"modèle={run_model}, RVT={run_rvt}"
        )
        narrator.cv_run_start(run_idx, len(cv_runs), model_display, run_rvt)

        generated_rvt_tif_dir = resolve_rvt_tif_dir(
            ctx.output_dir, run_rvt, output_structure, rvt_params
        )

        if not generated_rvt_tif_dir.exists() or not generated_rvt_tif_dir.is_dir():
            reporter.error(
                f"Computer Vision: dossier RVT/TIF non trouvé pour {run_rvt}: "
                f"{generated_rvt_tif_dir}"
            )
            continue

        # Callback bindé au modèle courant : remonte la sous-progression
        # (index/total images traitées) au narrator → ligne USER_INFO
        # discrète dans le journal, ET fait avancer la barre dans la bande CV
        # (répartie entre les runs). ``model_display``/``run_idx``/``n_runs``
        # sont figés par défaut-d'argument pour éviter le late-binding
        # classique en boucle Python.
        def _on_image_progress(
            idx, total, image_name,
            _model=model_display, _ri=run_idx, _n=len(cv_runs),
        ):
            narrator.cv_run_image_progress(_model, idx, total, image_name)
            reporter.progress(cv_pct(_ri, _n, idx, total, cv_band))

        run_existing_rvt(
            existing_rvt_dir=generated_rvt_tif_dir,
            output_dir=ctx.output_dir,
            cv_config=run_cfg,
            output_structure=output_structure,
            log=lambda m: reporter.info(m),
            cancel_check=cancel.is_cancelled,
            rvt_params=rvt_params,
            global_color_map=global_color_map,
            image_progress=_on_image_progress,
            on_busy=lambda active: report_busy(reporter, active),
        )
