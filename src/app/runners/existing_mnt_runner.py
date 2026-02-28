from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..cancel_token import CancelToken
from ..progress_reporter import ProgressReporter
from ..run_context import RunContext
from ..services.finalize_service import finalize_pipeline
from .helpers import log_section, resolve_rvt_tif_dir

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

        existing_mnt_dir_str = str((ctx.files_cfg.get("existing_mnt_dir") or "")).strip()
        if not existing_mnt_dir_str:
            reporter.error("Mode existing_mnt sélectionné mais aucun dossier MNT n'est configuré")
            return
        if ctx.output_dir is None:
            reporter.error("Aucun dossier de sortie n'est configuré")
            return

        processing_cfg = ctx.processing_cfg or {}
        output_structure = processing_cfg.get("output_structure", {})
        if not isinstance(output_structure, dict):
            output_structure = {}

        output_formats = processing_cfg.get("output_formats", {})
        if not isinstance(output_formats, dict):
            output_formats = {}

        rvt_params = ctx.rvt_params or {}
        products = ctx.products_cfg or {}

        # Déterminer les produits actifs
        active_products = [k for k in ("MNT", "M_HS", "SVF", "SLO", "LD", "VAT") if products.get(k, False)]

        # Section: Traitement MNT
        log_section("TRAITEMENT DES MNT EXISTANTS", "mnt", slog=slog, reporter=reporter)

        reporter.stage("Traitement MNT existants")
        reporter.progress(0)

        res = run_existing_mnt(
            existing_mnt_dir=Path(existing_mnt_dir_str),
            output_dir=ctx.output_dir,
            products=products,
            output_structure=output_structure,
            output_formats=output_formats,
            pyramids_config=(processing_cfg.get("pyramids") or {}),
            rvt_params=rvt_params,
            log=lambda m: reporter.info(m),
            cancel_check=cancel.is_cancelled,
        )

        reporter.info(f"✅ {res.total} MNT traités")

        if cancel.is_cancelled():
            reporter.info("Pipeline annulé après traitement MNT.")
            return

        # Lancer la CV si activée
        cv_config = ctx.cv_cfg or {}
        cv_enabled = bool(cv_config.get("enabled", False))

        if cv_enabled:
            try:
                from ...pipeline.modes.existing_rvt import run_existing_rvt
                from ...pipeline.cv.class_utils import resolve_cv_runs
                from ...app.services.finalize_service import _build_global_class_color_map

                cv_runs = resolve_cv_runs(cv_config)
                if not cv_runs:
                    reporter.info("Computer Vision: aucun modèle configuré dans les runs")
                else:
                    global_color_map: dict = {}
                    try:
                        global_color_map = _build_global_class_color_map(cv_runs)
                        reporter.info(f"Computer Vision: mapping couleurs global = {global_color_map}")
                    except Exception as _e:
                        reporter.info(f"Computer Vision: impossible de construire le mapping couleurs: {_e}")

                    log_section("COMPUTER VISION", "cv", slog=slog, reporter=reporter)
                    reporter.stage("Computer Vision")
                    reporter.progress(80)

                    for run_idx, run_cfg in enumerate(cv_runs, start=1):
                        if cancel.is_cancelled():
                            break

                        run_model = run_cfg.get("selected_model", "?")
                        run_rvt = run_cfg.get("target_rvt", "LD")
                        reporter.info(f"Computer Vision: run {run_idx}/{len(cv_runs)} — modèle={run_model}, RVT={run_rvt}")

                        generated_rvt_tif_dir = resolve_rvt_tif_dir(ctx.output_dir, run_rvt, output_structure, rvt_params)

                        if not generated_rvt_tif_dir.exists() or not generated_rvt_tif_dir.is_dir():
                            reporter.error(f"Computer Vision: dossier RVT/TIF non trouvé pour {run_rvt}: {generated_rvt_tif_dir}")
                            continue

                        run_existing_rvt(
                            existing_rvt_dir=generated_rvt_tif_dir,
                            output_dir=ctx.output_dir,
                            cv_config=run_cfg,
                            output_structure=output_structure,
                            log=lambda m: reporter.info(m),
                            cancel_check=cancel.is_cancelled,
                            rvt_params=rvt_params,
                            global_color_map=global_color_map,
                        )
            except Exception as e:
                reporter.error(f"Erreur Computer Vision: {e}")

        # Finalisation commune (VRT + shapefiles + load_layers)
        finalize_pipeline(
            output_dir=ctx.output_dir,
            cv_cfg=ctx.cv_cfg or {},
            rvt_params=rvt_params,
            reporter=reporter,
            slog=slog,
            start_time=start_time,
            tiles_processed=res.total,
            active_products=active_products,
            extra_label="MNT traités",
        )
