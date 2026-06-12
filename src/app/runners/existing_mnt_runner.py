from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from ..cancel_token import CancelToken
from ..cancellable_feedback import create_cancellable_feedback
from ..progress_reporter import ProgressReporter, report_stage_id
from ..progress_stages import Stage
from ..run_context import RunContext
from ..services.finalize_service import finalize_pipeline
from ..structured_logger import log_section
from ..user_narrator import create_user_narrator
from .progress_plan import build_progress_plan

try:  # cross-package import : OK en QGIS, fallback en tests standalone (src/ sur le path)
    from ...pipeline.cancellation import PipelineCancelled
except ImportError:  # pragma: no cover
    from pipeline.cancellation import PipelineCancelled

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
        try:  # fallback standalone (tests : src/ sur le path)
            from ...pipeline.modes.existing_mnt import run_existing_mnt
        except ImportError:  # pragma: no cover
            from pipeline.modes.existing_mnt import run_existing_mnt

        start_time = time.time()

        # Pré-conditions garanties par validate_run_context (V3.3).
        existing_mnt_dir = ctx.files.existing_mnt_dir
        assert existing_mnt_dir is not None
        assert ctx.output_dir is not None

        processing = ctx.processing
        products = processing.products
        rvt_params = ctx.rvt_params
        active_products = products.active()

        # COUVERTURE a été neutralisée par build_run_context (pas de nuage de
        # points dans ce mode) : prévenir si la config brute la demandait.
        raw_products = ((ctx.ui_config.get("processing") or {}).get("products") or {})
        if raw_products.get("COUVERTURE"):
            reporter.user_warning(
                "Produit Couverture indisponible en mode MNT existant "
                "(nuage de points requis) — ignoré."
            )

        plan = build_progress_plan(ctx.mode, ctx.cv.enabled)
        narrator = create_user_narrator(reporter)

        # Section: Traitement MNT
        log_section("TRAITEMENT DES MNT EXISTANTS", "mnt", slog=slog, reporter=reporter)

        # Le calcul MNT→indices RVT est la phase dominante : on l'affiche sous
        # la pastille « Produits » et on fait avancer la barre par MNT traité.
        report_stage_id(reporter, Stage.PRODUCTS)
        reporter.stage("Traitement MNT existants")
        reporter.progress(plan.products[0])

        feedback = create_cancellable_feedback(cancel.is_cancelled)

        def _on_mnt_progress(idx: int, total: int, name: str) -> None:
            reporter.progress(plan.at(plan.products, idx / max(1, total)))
            narrator.mnt_progress(idx, total, name)

        # Annulation = arrêt rapide du travail lourd, puis finalisation légère
        # (VRT + projet QGIS + chargement des couches déjà produites).
        cancelled = False
        fatal = False
        tiles_processed = 0
        try:
            res = run_existing_mnt(
                existing_mnt_dir=existing_mnt_dir,
                output_dir=ctx.output_dir,
                products=products.as_dict(),
                output_structure=processing.output_structure,
                output_formats=processing.output_formats,
                rvt_params=rvt_params,
                log=lambda m: reporter.info(m),
                error_log=lambda m: reporter.error(m),
                cancel_check=cancel.is_cancelled,
                feedback=feedback,
                mnt_progress=_on_mnt_progress,
            )
            tiles_processed = res.total
            reporter.info(f"✅ {res.total} MNT traités")

            # Lancer la CV si activée
            if ctx.cv.enabled and not cancel.is_cancelled():
                from ..services.cv_post_service import run_cv_post_loop
                try:
                    run_cv_post_loop(
                        ctx=ctx,
                        output_structure=processing.output_structure,
                        rvt_params=rvt_params,
                        reporter=reporter,
                        cancel=cancel,
                        slog=slog,
                        cv_band=plan.cv,
                    )
                except PipelineCancelled:
                    raise
                except Exception as e:
                    reporter.error(f"Erreur Computer Vision: {e}")
        except PipelineCancelled:
            cancelled = True
            reporter.info("Annulation demandée — finalisation des résultats partiels…")
        except Exception:
            # L'issue est transmise à la finalisation (pas de faux « ✅ »,
            # AUDIT v2 ROB-14) puis l'erreur remonte au contrôleur.
            fatal = True
            raise
        finally:
            # Finalisation commune LÉGÈRE — TOUJOURS exécutée (y compris si une
            # erreur inattendue remonte) : indexe et charge les MNT déjà traités
            # avant que l'erreur éventuelle ne remonte au contrôleur (AUDIT ROB-01/03).
            if cancelled or cancel.is_cancelled():
                outcome = "cancelled"
            elif fatal:
                outcome = "failed"
            else:
                outcome = "success"
            finalize_pipeline(
                output_dir=ctx.output_dir,
                cv_cfg=ctx.cv.raw,
                rvt_params=rvt_params,
                reporter=reporter,
                slog=slog,
                start_time=start_time,
                tiles_processed=tiles_processed,
                active_products=active_products,
                extra_label="MNT traités",
                ui_config=ctx.ui_config,
                outcome=outcome,
            )

        if cancelled or cancel.is_cancelled():
            narrator.pipeline_cancelled()
