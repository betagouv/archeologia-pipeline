from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..cancel_token import CancelToken
from ..cancellable_feedback import create_cancellable_feedback
from ..progress_reporter import ProgressReporter
from ..run_context import RunContext
from ..services.finalize_service import finalize_pipeline
from ..structured_logger import log_section
from ..user_narrator import create_user_narrator
from .input_strategy import select_input_strategy

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

        if products_cfg.get("DENSITE", False):
            create_density_map(
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

        if ctx.output_dir is None:
            reporter.error("Aucun dossier de sortie n'est configuré")
            return

        processing = ctx.processing
        products = processing.products

        feedback = create_cancellable_feedback(cancel.is_cancelled)
        narrator = create_user_narrator(reporter)

        strategy = select_input_strategy(ctx.mode)

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
        reporter.stage("Fusion (voisins + merge)")
        reporter.progress(strategy.merge_progress_start())
        narrator.merging_start()

        merged_result = prepare_merged_tiles(
            sorted_list_file=result.sorted_list_file,
            dalles_dir=result.dalles_dir,
            output_dir=ctx.output_dir,
            tile_overlap_percent=processing.tile_overlap,
            log=lambda m: reporter.info(m),
            cancel=lambda: cancel.is_cancelled(),
            stage=lambda s: reporter.stage(s),
            max_workers=processing.max_workers,
        )

        merge_end = strategy.merge_progress_end()
        if merge_end is not None:
            reporter.progress(merge_end)

        active_products: list = []

        if products.needs_mnt() and merged_result.merged_files:
            rvt_params = ctx.rvt_params
            active_products = products.active()

            log_section("TRAITEMENT DES DALLES", "process", slog=slog, reporter=reporter)
            reporter.stage("Traitement des dalles")
            reporter.progress(strategy.products_progress_start())

            total_mnt = len(merged_result.merged_files)
            narrator.products_phase_start(total_mnt, active_products)

            for i, merged_path in enumerate(merged_result.merged_files, start=1):
                if cancel.is_cancelled():
                    break

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
                        base_progress=90,
                    )
                except Exception as e:
                    reporter.error(f"Erreur Computer Vision: {e}")

        # Finalisation commune (VRT + shapefiles + load_layers)
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
            ui_config=ctx.ui_config,
        )
