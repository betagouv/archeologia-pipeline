from __future__ import annotations

from pathlib import Path

from ..cancel_token import CancelToken
from ..progress_reporter import ProgressReporter
from ..run_context import RunContext


class ExistingRvtRunner:
    def run(self, ctx: RunContext, reporter: ProgressReporter, cancel: CancelToken) -> None:
        from ...pipeline.modes.existing_rvt import run_existing_rvt

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

        reporter.stage("Computer Vision (existing RVT)")
        reporter.progress(0)

        res = run_existing_rvt(
            existing_rvt_dir=Path(existing_rvt_dir_str),
            output_dir=ctx.output_dir,
            cv_config=ctx.cv_cfg,
            output_structure=output_structure,
            log=lambda m: reporter.info(m),
        )

        reporter.stage("Terminé")
        reporter.progress(100)
        reporter.info(f"Mode existing_rvt terminé: {res.total_images} images")
