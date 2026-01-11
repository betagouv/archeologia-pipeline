from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..cancel_token import CancelToken
from ..progress_reporter import ProgressReporter
from ..run_context import RunContext

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

        # Section: Computer Vision
        if slog:
            slog.section("COMPUTER VISION (RVT EXISTANTS)", "cv")
        else:
            reporter.info("")
            reporter.info("════════════════════════════════════════════════════════════")
            reporter.info("🤖 COMPUTER VISION (RVT EXISTANTS)")
            reporter.info("════════════════════════════════════════════════════════════")

        reporter.stage("Computer Vision (existing RVT)")
        reporter.progress(0)

        res = run_existing_rvt(
            existing_rvt_dir=Path(existing_rvt_dir_str),
            output_dir=ctx.output_dir,
            cv_config=cv_config,
            output_structure=output_structure,
            log=lambda m: reporter.info(m),
        )

        # Section finale
        elapsed = time.time() - start_time
        reporter.info("")
        reporter.info("════════════════════════════════════════════════════════════")
        reporter.info("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
        reporter.info("════════════════════════════════════════════════════════════")
        reporter.info(f"  ⏱️ Durée totale : {elapsed:.1f}s")
        reporter.info(f"  📄 Images traitées : {res.total_images}")
        reporter.info(f"  📦 RVT cible : {target_rvt}")
        reporter.info("════════════════════════════════════════════════════════════")
        reporter.info("")

        reporter.stage("Terminé")
        reporter.progress(100)
