from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..cancel_token import CancelToken
from ..progress_reporter import ProgressReporter
from ..run_context import RunContext
from ..services.finalize_service import finalize_pipeline
from .helpers import log_section

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
        from ...pipeline.modes.existing_rvt import run_existing_rvt

        start_time = time.time()

        existing_rvt_dir_str = str((ctx.files_cfg.get("existing_rvt_dir") or "")).strip()
        if not existing_rvt_dir_str:
            reporter.error("Mode existing_rvt sélectionné mais aucun dossier RVT n'est configuré")
            return
        if ctx.output_dir is None:
            reporter.error("Aucun dossier de sortie n'est configuré")
            return

        processing_cfg = ctx.processing_cfg or {}
        output_structure = processing_cfg.get("output_structure", {})
        if not isinstance(output_structure, dict):
            output_structure = {}

        cv_config = ctx.cv_cfg or {}
        target_rvt = str(cv_config.get("target_rvt", "LD"))

        # Collecter tous les RVT cibles uniques depuis les runs
        from ...pipeline.cv.class_utils import resolve_cv_runs
        from ...app.services.finalize_service import _build_global_class_color_map
        cv_runs = resolve_cv_runs(cv_config)
        active_rvts = list(dict.fromkeys(
            r.get("target_rvt", target_rvt) for r in cv_runs
        )) or [target_rvt]

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
        for run_idx, run_cfg in enumerate(cv_runs, start=1):
            if cancel.is_cancelled():
                break

            run_model = run_cfg.get("selected_model", "?")
            run_rvt = run_cfg.get("target_rvt", target_rvt)
            reporter.info(f"Computer Vision: run {run_idx}/{len(cv_runs)} — modèle={run_model}, RVT={run_rvt}")

            res = run_existing_rvt(
                existing_rvt_dir=Path(existing_rvt_dir_str),
                output_dir=ctx.output_dir,
                cv_config=run_cfg,
                output_structure=output_structure,
                log=lambda m: reporter.info(m),
                cancel_check=cancel.is_cancelled,
                rvt_params=ctx.rvt_params or {},
                global_color_map=global_color_map,
            )
            total_images = max(total_images, res.total_images)

        # Finalisation commune (VRT + shapefiles + load_layers)
        finalize_pipeline(
            output_dir=ctx.output_dir,
            cv_cfg=cv_config,
            rvt_params=ctx.rvt_params or {},
            reporter=reporter,
            slog=slog,
            start_time=start_time,
            tiles_processed=total_images,
            active_products=active_rvts,
            extra_label="Images traitées",
        )
