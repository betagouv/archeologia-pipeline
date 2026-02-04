from __future__ import annotations

import time
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
        if slog:
            slog.section("TRAITEMENT DES MNT EXISTANTS", "mnt")
        else:
            reporter.info("")
            reporter.info("════════════════════════════════════════════════════════════")
            reporter.info("🔧 TRAITEMENT DES MNT EXISTANTS")
            reporter.info("════════════════════════════════════════════════════════════")

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
        )

        reporter.info(f"✅ {res.total} MNT traités")

        # Lancer la CV si activée
        cv_config = ctx.cv_cfg or {}
        cv_enabled = bool(cv_config.get("enabled", False))
        target_rvt = str(cv_config.get("target_rvt", "LD"))

        if cv_enabled:
            try:
                from ...pipeline.modes.existing_rvt import run_existing_rvt

                rvt_cfg = output_structure.get("RVT", {}) if isinstance(output_structure, dict) else {}
                base_dir_name = str(rvt_cfg.get("base_dir", "RVT"))
                type_dir_name = str(rvt_cfg.get(target_rvt, target_rvt))
                generated_rvt_tif_dir = (ctx.output_dir / "results") / base_dir_name / type_dir_name / "tif"

                if not generated_rvt_tif_dir.exists() or not generated_rvt_tif_dir.is_dir():
                    reporter.error(f"Computer Vision demandée mais aucun dossier RVT/TIF trouvé: {generated_rvt_tif_dir}")
                else:
                    # Section: Computer Vision
                    if slog:
                        slog.section("COMPUTER VISION", "cv")
                    else:
                        reporter.info("")
                        reporter.info("════════════════════════════════════════════════════════════")
                        reporter.info("🤖 COMPUTER VISION")
                        reporter.info("════════════════════════════════════════════════════════════")

                    reporter.stage("Computer Vision")
                    reporter.progress(80)
                    run_existing_rvt(
                        existing_rvt_dir=generated_rvt_tif_dir,
                        output_dir=ctx.output_dir,
                        cv_config=cv_config,
                        output_structure=output_structure,
                        log=lambda m: reporter.info(m),
                        cancel_check=cancel.is_cancelled,
                    )
            except Exception as e:
                reporter.error(f"Erreur Computer Vision: {e}")

        # Création des fichiers VRT pour indexer les dalles par produit
        from ...pipeline.ign.products.results import build_vrt_index
        reporter.info("Création des fichiers VRT d'indexation...")
        results_dir = ctx.output_dir / "results"
        if results_dir.exists():
            # VRT pour chaque dossier de produit TIF
            for tif_dir in results_dir.rglob("tif"):
                if tif_dir.is_dir() and list(tif_dir.glob("*.tif")):
                    build_vrt_index(tif_dir, pattern="*.tif", output_name="index.vrt", log=lambda m: reporter.info(m))
            # VRT pour chaque dossier JPG (images géoréférencées)
            for jpg_dir in results_dir.rglob("jpg"):
                if jpg_dir.is_dir() and list(jpg_dir.glob("*.jpg")):
                    build_vrt_index(jpg_dir, pattern="*.jpg", output_name="index.vrt", log=lambda m: reporter.info(m))
            # VRT pour annotated_images si présent
            annotated_dir = results_dir / "annotated_images"
            if annotated_dir.exists() and list(annotated_dir.glob("*.jpg")):
                build_vrt_index(annotated_dir, pattern="*.jpg", output_name="index.vrt", log=lambda m: reporter.info(m))

        # Section finale
        elapsed = time.time() - start_time
        reporter.info("")
        reporter.info("════════════════════════════════════════════════════════════")
        reporter.info("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
        reporter.info("════════════════════════════════════════════════════════════")
        reporter.info(f"  ⏱️ Durée totale : {elapsed:.1f}s")
        reporter.info(f"  📄 MNT traités : {res.total}")
        reporter.info(f"  📦 Produits : {', '.join(active_products) if active_products else 'aucun'}")
        reporter.info("════════════════════════════════════════════════════════════")
        reporter.info("")

        reporter.stage("Terminé")
        reporter.progress(100)
