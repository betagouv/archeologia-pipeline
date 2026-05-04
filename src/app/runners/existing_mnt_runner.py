from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from ..cancel_token import CancelToken
from ..progress_reporter import ProgressReporter
from ..run_context import RunContext
from ..services.finalize_service import finalize_pipeline
from ..structured_logger import log_section

if TYPE_CHECKING:
    from ..structured_logger import StructuredLogger


class ExistingMntRunner:
    def run(
        self,
        ctx: RunContext,
        reporter: ProgressReporter,
        cancel: CancelToken,
        slog: Optional["StructuredLogger"] = None,
    ) -> None:
        from ...pipeline.modes.existing_mnt import run_existing_mnt

        start_time = time.time()

        existing_mnt_dir = ctx.files.existing_mnt_dir
        if existing_mnt_dir is None:
            reporter.error("Mode existing_mnt sélectionné mais aucun dossier MNT n'est configuré")
            return
        if ctx.output_dir is None:
            reporter.error("Aucun dossier de sortie n'est configuré")
            return

        processing = ctx.processing
        products = processing.products
        rvt_params = ctx.rvt_params
        active_products = products.active()

        # Section: Traitement MNT
        log_section("TRAITEMENT DES MNT EXISTANTS", "mnt", slog=slog, reporter=reporter)

        reporter.stage("Traitement MNT existants")
        reporter.progress(0)

        res = run_existing_mnt(
            existing_mnt_dir=existing_mnt_dir,
            output_dir=ctx.output_dir,
            products=products.as_dict(),
            output_structure=processing.output_structure,
            output_formats=processing.output_formats,
            rvt_params=rvt_params,
            log=lambda m: reporter.info(m),
            cancel_check=cancel.is_cancelled,
        )

        reporter.info(f"✅ {res.total} MNT traités")

        if cancel.is_cancelled():
            reporter.info("Pipeline annulé après traitement MNT.")
            return

        # Lancer la CV si activée
        if ctx.cv.enabled:
            from ..services.cv_post_service import run_cv_post_loop
            try:
                run_cv_post_loop(
                    ctx=ctx,
                    output_structure=processing.output_structure,
                    rvt_params=rvt_params,
                    reporter=reporter,
                    cancel=cancel,
                    slog=slog,
                    base_progress=80,
                )
            except Exception as e:
                reporter.error(f"Erreur Computer Vision: {e}")

        # Finalisation commune (VRT + shapefiles + load_layers)
        finalize_pipeline(
            output_dir=ctx.output_dir,
            cv_cfg=ctx.cv.raw,
            rvt_params=rvt_params,
            reporter=reporter,
            slog=slog,
            start_time=start_time,
            tiles_processed=res.total,
            active_products=active_products,
            extra_label="MNT traités",
            ui_config=ctx.ui_config,
        )
