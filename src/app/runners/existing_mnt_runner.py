from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..cancel_token import CancelToken
from ..progress_reporter import ProgressReporter
from ..run_context import RunContext

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

        reporter.stage("Computer Vision (existing MNT)")
        reporter.progress(0)

        res = run_existing_mnt(
            existing_mnt_dir=Path(existing_mnt_dir_str),
            output_dir=ctx.output_dir,
            products=ctx.products_cfg,
            output_structure=output_structure,
            output_formats=output_formats,
            pyramids_config=(processing_cfg.get("pyramids") or {}),
            rvt_params=rvt_params,
            log=lambda m: reporter.info(m),
        )

        reporter.stage("Terminé")
        reporter.progress(100)
        reporter.info(f"Mode existing_mnt terminé: {res.total} MNT traités")
