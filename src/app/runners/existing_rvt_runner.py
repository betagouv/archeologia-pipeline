from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from ..cancel_token import CancelToken
from ..progress_reporter import ProgressReporter, report_busy, report_stage_id
from ..progress_stages import Stage
from ..run_context import RunContext
from ..services.cv_post_service import _model_display_name
from ..services.finalize_service import finalize_pipeline
from ..structured_logger import log_section
from ..user_narrator import create_user_narrator
from .progress_plan import build_progress_plan, cv_pct

try:  # cross-package import : OK en QGIS, fallback en tests standalone (src/ sur le path)
    from ...pipeline.cancellation import PipelineCancelled
except ImportError:  # pragma: no cover
    from pipeline.cancellation import PipelineCancelled

if TYPE_CHECKING:
    from ..structured_logger import StructuredLogger


class ExistingRvtRunner:
    def run(
        self,
        ctx: RunContext,
        reporter: ProgressReporter,
        cancel: CancelToken,
        slog: Optional["StructuredLogger"] = None,
    ) -> None:
        try:
            from ...pipeline.modes.existing_rvt import run_existing_rvt
        except ImportError:
            from pipeline.modes.existing_rvt import run_existing_rvt

        start_time = time.time()

        # Pré-conditions garanties par validate_run_context (V3.3).
        existing_rvt_dir = ctx.files.existing_rvt_dir
        assert existing_rvt_dir is not None
        assert ctx.output_dir is not None

        processing = ctx.processing
        cv_config = ctx.cv.raw
        target_rvt = str(cv_config.get("target_rvt", "LD"))

        # Collecter tous les RVT cibles uniques depuis les runs
        try:
            from ...pipeline.cv.class_utils import resolve_cv_runs
        except ImportError:
            from pipeline.cv.class_utils import resolve_cv_runs
        from ..services.finalize_service import _build_global_class_color_map
        cv_runs = resolve_cv_runs(cv_config)
        active_rvts = list(dict.fromkeys(
            r.get("target_rvt", target_rvt) for r in cv_runs
        )) or [target_rvt]
        run_configs = cv_runs or [dict(cv_config, enabled=False, target_rvt=target_rvt)]

        # Construire le mapping global couleurs AVANT les runs pour que chaque
        # modèle écrive les bonnes couleurs dans les shapefiles dès la génération
        global_color_map: dict = {}
        if cv_runs:
            try:
                global_color_map = _build_global_class_color_map(cv_runs)
                reporter.info(f"Computer Vision: mapping couleurs global = {global_color_map}")
            except Exception as _e:
                reporter.info(f"Computer Vision: impossible de construire le mapping couleurs: {_e}")

        # Bande de progression : prep TIF→PNG 0–10, inférence CV 10–95.
        plan = build_progress_plan(ctx.mode, ctx.cv.enabled)

        # Section: Traitement RVT existants
        log_section("TRAITEMENT RVT EXISTANTS", "cv", slog=slog, reporter=reporter)

        # Ce mode est fondamentalement de la détection : on allume directement
        # la pastille « Détection » de la timeline.
        report_stage_id(reporter, Stage.DETECTION)
        reporter.stage("Traitement RVT existants")
        reporter.progress(plan.products[0])

        total_images = 0
        if not cv_runs and cv_config.get("enabled", False):
            reporter.info("Computer Vision: aucun modèle configuré — inférence ignorée")

        narrator = create_user_narrator(reporter)
        if cv_runs:
            narrator.cv_start(len(cv_runs))
            reporter.stage("Computer Vision")
            # La barre entre dans la bande CV (10) ; elle progresse ensuite
            # image par image via cv_pct dans _on_image_progress.
            reporter.progress(plan.cv[0])

        # Annulation = arrêt rapide du travail lourd, puis finalisation légère
        # (VRT + projet QGIS + chargement des couches déjà produites).
        cancelled = False
        try:
            for run_idx, run_cfg in enumerate(run_configs, start=1):
                if cancel.is_cancelled():
                    break

                run_model = run_cfg.get("selected_model", "?")
                run_rvt = run_cfg.get("target_rvt", target_rvt)
                model_display = _model_display_name(run_model)
                if cv_runs:
                    reporter.info(f"Computer Vision: run {run_idx}/{len(cv_runs)} — modèle={run_model}, RVT={run_rvt}")
                    narrator.cv_run_start(run_idx, len(cv_runs), model_display, run_rvt)

                # ``_model``/``_ri``/``_n`` figés par défaut-d'argument (late-binding).
                def _on_image_progress(
                    idx, total, image_name,
                    _model=model_display, _ri=run_idx, _n=len(cv_runs),
                ):
                    narrator.cv_run_image_progress(_model, idx, total, image_name)
                    reporter.progress(cv_pct(_ri, _n, idx, total, plan.cv))

                res = run_existing_rvt(
                    existing_rvt_dir=existing_rvt_dir,
                    output_dir=ctx.output_dir,
                    cv_config=run_cfg,
                    output_structure=processing.output_structure,
                    log=lambda m: reporter.info(m),
                    cancel_check=cancel.is_cancelled,
                    rvt_params=ctx.rvt_params,
                    global_color_map=global_color_map,
                    indices_folder_name="RVT",
                    image_progress=_on_image_progress if cv_runs else None,
                    on_busy=lambda active: report_busy(reporter, active),
                )
                total_images = max(total_images, res.total_images)
        except PipelineCancelled:
            cancelled = True
            reporter.info("Annulation demandée — finalisation des résultats partiels…")

        # Finalisation commune LÉGÈRE — toujours exécutée (bornée → va au bout)
        finalize_pipeline(
            output_dir=ctx.output_dir,
            cv_cfg=cv_config,
            rvt_params=ctx.rvt_params,
            reporter=reporter,
            slog=slog,
            start_time=start_time,
            tiles_processed=total_images,
            active_products=active_rvts,
            extra_label="Images traitées",
            ui_config=ctx.ui_config,
        )

        if cancelled or cancel.is_cancelled():
            narrator.pipeline_cancelled()
