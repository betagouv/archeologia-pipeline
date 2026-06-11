from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..cancel_token import CancelToken
from ..cancellable_feedback import create_cancellable_feedback
from ..progress_reporter import ProgressReporter, report_stage_id
from ..progress_stages import Stage
from ..run_context import RunContext
from ..services.finalize_service import finalize_pipeline
from ..structured_logger import log_section
from ..user_narrator import create_user_narrator
from .input_strategy import select_input_strategy
from .progress_plan import build_progress_plan

try:  # cross-package import : OK en QGIS, fallback en tests standalone (src/ sur le path)
    from ...pipeline.cancellation import PipelineCancelled
    from ...pipeline.batch import process_items_isolated
except ImportError:  # pragma: no cover
    from pipeline.cancellation import PipelineCancelled
    from pipeline.batch import process_items_isolated

if TYPE_CHECKING:
    from ..structured_logger import StructuredLogger


class IgnOrLocalRunner:
    # ------------------------------------------------------------------ #
    #  Traitement d'une dalle individuelle                                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _process_tile(
        *,
        merged_path: Path,
        output_dir: Path,
        tile_overlap: float,
        mnt_resolution: float,
        density_resolution: float,
        filter_expression: str,
        products_cfg: Dict[str, Any],
        output_structure: Dict[str, Any],
        output_formats: Dict[str, Any],
        rvt_params: Dict[str, Any],
        reporter: ProgressReporter,
        cancel: CancelToken,
        feedback: Any,
        slog: Optional["StructuredLogger"],
        tile_index: int,
        total_tiles: int,
        active_products: list,
    ) -> None:
        from ...pipeline.ign.products.coverage import create_coverage_map
        from ...pipeline.ign.products.crop import crop_final_products
        from ...pipeline.ign.products.density import create_density_map
        from ...pipeline.ign.products.indices import create_visualization_products
        from ...pipeline.ign.products.mnt import create_terrain_model
        from ...pipeline.ign.products.results import copy_final_products_to_results

        from ...pipeline.output_paths import intermediaires_dir

        tile_name = merged_path.name.replace(".copc.laz", "").replace(".laz", "")
        temp_dir = intermediaires_dir(output_dir)

        reporter.stage(f"Traitement dalle {tile_index}/{total_tiles}")
        if slog:
            slog.tile_start(tile_index, total_tiles, tile_name)

        create_terrain_model(
            input_laz_path=merged_path,
            temp_dir=temp_dir,
            current_tile_name=tile_name,
            mnt_resolution=mnt_resolution,
            tile_overlap_percent=tile_overlap,
            filter_expression=str(filter_expression),
            log=lambda m: reporter.info(m),
            feedback=feedback,
        )

        if cancel.is_cancelled():
            return

        # La passe densité sert DENSITE (publié) ET COUVERTURE (dérivé).
        # Si DENSITE est décoché, son TIF reste en temp sans être publié
        # (les boucles crop/copy filtrent sur le dict produits).
        needs_density = bool(
            products_cfg.get("DENSITE", False) or products_cfg.get("COUVERTURE", False)
        )
        if needs_density:
            density_result = create_density_map(
                input_laz_path=merged_path,
                temp_dir=temp_dir,
                current_tile_name=tile_name,
                density_resolution=density_resolution,
                tile_overlap_percent=tile_overlap,
                filter_expression=str(filter_expression),
                log=lambda m: reporter.info(m),
                feedback=feedback,
            )

            if cancel.is_cancelled():
                return

            if products_cfg.get("COUVERTURE", False):
                create_coverage_map(
                    density_path=density_result.density_path,
                    temp_dir=temp_dir,
                    current_tile_name=tile_name,
                    log=lambda m: reporter.info(m),
                )

                if cancel.is_cancelled():
                    return

        create_visualization_products(
            temp_dir=temp_dir,
            current_tile_name=tile_name,
            products=products_cfg,
            rvt_params=rvt_params,
            log=lambda m: reporter.info(m),
            feedback=feedback,
        )

        if cancel.is_cancelled():
            return

        cropped = crop_final_products(
            temp_dir=temp_dir,
            current_tile_name=tile_name,
            products=products_cfg,
            rvt_params=rvt_params,
            log=lambda m: reporter.info(m),
            cancel_check=cancel.is_cancelled,
        )

        if cropped:
            copy_final_products_to_results(
                temp_dir=temp_dir,
                output_dir=output_dir,
                current_tile_name=tile_name,
                products=products_cfg,
                output_structure=output_structure,
                output_formats=output_formats,
                rvt_params=rvt_params,
                log=lambda m: reporter.info(m),
                cancel_check=cancel.is_cancelled,
            )

        if slog:
            slog.tile_end(tile_name, active_products)

    # ------------------------------------------------------------------ #
    #  Point d'entrée principal                                           #
    # ------------------------------------------------------------------ #
    def run(
        self,
        ctx: RunContext,
        reporter: ProgressReporter,
        cancel: CancelToken,
        slog: Optional["StructuredLogger"] = None,
    ) -> None:
        # Vider le cache de validation PDAL au début de chaque run
        from ...pipeline.ign.pdal_validation import clear_validation_cache
        clear_validation_cache()

        start_time = time.time()

        # Validation centralisée dans PipelineController.run() — ici on
        # peut supposer que ctx.output_dir et les chemins requis pour
        # le mode sont valides. L'assertion documente l'invariant et
        # rassure le type-checker (output_dir typé Optional[Path]).
        assert ctx.output_dir is not None

        processing = ctx.processing
        products = processing.products

        feedback = create_cancellable_feedback(cancel.is_cancelled)
        narrator = create_user_narrator(reporter)

        plan = build_progress_plan(ctx.mode, ctx.cv.enabled)
        strategy = select_input_strategy(ctx.mode, plan)

        result = strategy.acquire(
            ctx=ctx,
            reporter=reporter,
            cancel=cancel,
            slog=slog,
            processing=processing,
        )
        if result is None:
            return

        from ...pipeline.ign.preprocess import prepare_merged_tiles

        log_section("FUSION DES TUILES", "process", slog=slog, reporter=reporter)
        # Fusion + produits partagent la pastille « Produits » de la timeline
        # (le calcul MNT+RVT est entrelacé dalle par dalle, sans frontière).
        report_stage_id(reporter, Stage.PRODUCTS)
        reporter.stage("Fusion (voisins + merge)")
        reporter.progress(strategy.merge_progress_start())
        narrator.merging_start()

        def _on_tile_merged(i: int, n: int, tile_name: str) -> None:
            narrator.merging_tile_progress(i, n, tile_name)

        merged_result = prepare_merged_tiles(
            sorted_list_file=result.sorted_list_file,
            dalles_dir=result.dalles_dir,
            output_dir=ctx.output_dir,
            tile_overlap_percent=processing.tile_overlap,
            log=lambda m: reporter.info(m),
            cancel=lambda: cancel.is_cancelled(),
            stage=lambda s: reporter.stage(s),
            max_workers=processing.max_workers,
            on_tile_merged=_on_tile_merged,
        )

        merge_end = strategy.merge_progress_end()
        if merge_end is not None:
            reporter.progress(merge_end)

        active_products: list = []

        # Annulation = arrêt rapide du travail lourd, puis finalisation légère
        # (VRT + projet QGIS + chargement des couches déjà produites).
        cancelled = False
        try:
            if products.needs_tile_processing() and merged_result.merged_files:
                rvt_params = ctx.rvt_params
                active_products = products.active()

                log_section("TRAITEMENT DES DALLES", "process", slog=slog, reporter=reporter)
                reporter.stage("Traitement des dalles")
                reporter.progress(strategy.products_progress_start())

                total_mnt = len(merged_result.merged_files)
                narrator.products_phase_start(total_mnt, active_products)

                def _process_one_tile(i: int, merged_path: Path) -> None:
                    tile_label = merged_path.name.replace(".copc.laz", "").replace(".laz", "")
                    narrator.tile_progress(i, total_mnt, tile_label)
                    self._process_tile(
                        merged_path=merged_path,
                        output_dir=ctx.output_dir,
                        tile_overlap=processing.tile_overlap,
                        mnt_resolution=processing.mnt_resolution,
                        density_resolution=processing.density_resolution,
                        filter_expression=processing.filter_expression,
                        products_cfg=products.as_dict(),
                        output_structure=processing.output_structure,
                        output_formats=processing.output_formats,
                        rvt_params=rvt_params,
                        reporter=reporter,
                        cancel=cancel,
                        feedback=feedback,
                        slog=slog,
                        tile_index=i,
                        total_tiles=total_mnt,
                        active_products=active_products,
                    )
                    reporter.progress(strategy.products_progress_for_tile(i, total_mnt))

                def _on_tile_failure(i: int, merged_path: Path, exc: Exception) -> None:
                    label = merged_path.name.replace(".copc.laz", "").replace(".laz", "")
                    reporter.error(f"Dalle {label} en échec, ignorée : {exc}")

                # Isolation par dalle : une dalle illisible/échouée n'avorte plus
                # tout le lot — les autres dalles sont traitées, la finalisation a
                # toujours lieu (cf. AUDIT ROB-02).
                _ok, failed = process_items_isolated(
                    merged_result.merged_files,
                    _process_one_tile,
                    cancel=cancel.is_cancelled,
                    on_failure=_on_tile_failure,
                )
                if failed:
                    reporter.error(
                        f"⚠️ {len(failed)} dalle(s) sur {total_mnt} en échec "
                        "— voir le journal pour le détail."
                    )

                # Computer Vision globale (post-boucle)
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
        finally:
            # Finalisation commune LÉGÈRE — TOUJOURS exécutée (y compris si une
            # erreur inattendue remonte hors de la boucle isolée) : indexe et
            # charge les produits déjà générés avant que l'erreur éventuelle
            # ne remonte au contrôleur (cf. AUDIT ROB-01/02).
            finalize_pipeline(
                output_dir=ctx.output_dir,
                cv_cfg=ctx.cv.raw,
                rvt_params=ctx.rvt_params,
                reporter=reporter,
                slog=slog,
                start_time=start_time,
                tiles_processed=len(merged_result.merged_files) if merged_result else 0,
                active_products=active_products,
                extra_label="Dalles traitées",
                coverage_threshold_percent=processing.coverage_threshold_percent,
                ui_config=ctx.ui_config,
            )

        if cancelled or cancel.is_cancelled():
            narrator.pipeline_cancelled()
