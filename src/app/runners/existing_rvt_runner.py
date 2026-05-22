from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from ..cancel_token import CancelToken
from ..progress_reporter import ProgressReporter
from ..run_context import RunContext
from ..services.cv_post_service import _model_display_name
from ..services.finalize_service import finalize_pipeline
from ..structured_logger import log_section
from ..user_narrator import create_user_narrator

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

        # Section: Traitement RVT existants
        log_section("TRAITEMENT RVT EXISTANTS", "cv", slog=slog, reporter=reporter)

        reporter.stage("Traitement RVT existants")
        reporter.progress(0)

        total_images = 0
        if not cv_runs and cv_config.get("enabled", False):
            reporter.info("Computer Vision: aucun modèle configuré — inférence ignorée")

        narrator = create_user_narrator(reporter)
        if cv_runs:
            narrator.cv_start(len(cv_runs))
            # Le vrai travail de ce mode est la détection : on avance la timeline
            # sur l'étape « Détection » (sinon elle reste bloquée sur « Indices »,
            # héritée de la section « Traitement RVT existants »). Cohérent avec
            # cv_post_service qui émet ce même stage pour les autres modes.
            reporter.stage("Computer Vision")

        for run_idx, run_cfg in enumerate(run_configs, start=1):
            if cancel.is_cancelled():
                break

            run_model = run_cfg.get("selected_model", "?")
            run_rvt = run_cfg.get("target_rvt", target_rvt)
            model_display = _model_display_name(run_model)
            if cv_runs:
                reporter.info(f"Computer Vision: run {run_idx}/{len(cv_runs)} — modèle={run_model}, RVT={run_rvt}")
                narrator.cv_run_start(run_idx, len(cv_runs), model_display, run_rvt)

            def _on_image_progress(idx, total, image_name, _model=model_display):
                narrator.cv_run_image_progress(_model, idx, total, image_name)

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
            )
            total_images = max(total_images, res.total_images)


        # Finalisation commune (VRT + shapefiles + load_layers)
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
