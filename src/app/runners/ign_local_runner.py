from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..cancel_token import CancelToken
from ..cancellable_feedback import create_cancellable_feedback
from ..progress_reporter import ProgressReporter
from ..run_context import RunContext
from ..services.finalize_service import finalize_pipeline
from .helpers import log_section, safe_float, resolve_rvt_tif_dir

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
        pyramids_config: Dict[str, Any],
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

        tile_name = merged_path.name.replace(".copc.laz", "").replace(".laz", "")
        temp_dir = output_dir / "temp"

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
                pyramids_config=pyramids_config,
                log=lambda m: reporter.info(m),
            )

        if slog:
            slog.tile_end(tile_name, active_products)

    # ------------------------------------------------------------------ #
    #  Lancement de la Computer Vision globale (post-boucle)              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _run_post_cv(
        *,
        ctx: RunContext,
        output_structure: Dict[str, Any],
        rvt_params: Dict[str, Any],
        reporter: ProgressReporter,
        cancel: CancelToken,
        slog: Optional["StructuredLogger"],
    ) -> None:
        from ...pipeline.modes.existing_rvt import run_existing_rvt
        from ...pipeline.cv.class_utils import resolve_cv_runs
        from ...app.services.finalize_service import _build_global_class_color_map

        cv_cfg = ctx.cv_cfg or {}
        cv_runs = resolve_cv_runs(cv_cfg)
        if not cv_runs:
            reporter.info("Computer Vision: aucun modèle configuré dans les runs")
            return

        global_color_map: dict = {}
        try:
            global_color_map = _build_global_class_color_map(cv_runs)
            reporter.info(f"Computer Vision: mapping couleurs global = {global_color_map}")
        except Exception as _e:
            reporter.info(f"Computer Vision: impossible de construire le mapping couleurs: {_e}")

        log_section("COMPUTER VISION", "cv", slog=slog, reporter=reporter)
        reporter.stage("Computer Vision")
        reporter.progress(90)

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

        processing = ctx.processing_cfg or {}
        products = (processing.get("products") or {})
        if not isinstance(products, dict):
            products = {}

        need_mnt = bool(products.get("MNT", True)) or any(bool(products.get(k, False)) for k in ("M_HS", "SVF", "SLO", "LD", "VAT"))
        
        feedback = create_cancellable_feedback(cancel.is_cancelled)

        download_range = (0, 25)
        merge_range = (25, 35)
        products_range = (35, 95)

        if ctx.mode == "ign_laz":
            from ...pipeline.ign.downloader import download_ign_dalles

            input_file = str((ctx.files_cfg.get("input_file") or "")).strip()
            if not input_file:
                reporter.error("Mode IGN sélectionné mais aucun fichier de liste d'URLs n'est configuré")
                return
            input_path = Path(input_file)
            if not input_path.exists():
                reporter.error(f"Fichier dalles IGN introuvable: {input_path}")
                return
            log_section("TÉLÉCHARGEMENT DES DALLES IGN", "download", slog=slog, reporter=reporter)
            reporter.stage("Téléchargement des dalles")
            max_workers = safe_float(processing.get("max_workers", 4), 4)
            result = download_ign_dalles(
                input_file=input_path,
                output_dir=ctx.output_dir,
                log=lambda m: reporter.info(m),
                progress=lambda p: reporter.progress(
                    int(download_range[0] + (download_range[1] - download_range[0]) * (int(p) / 100.0))
                ),
                stage=lambda s: reporter.stage(str(s)),
                cancel=lambda: cancel.is_cancelled(),
                max_workers=max_workers,
            )
        else:
            from ...pipeline.modes.local_laz import run_local_laz

            local_dir_str = str((ctx.files_cfg.get("local_laz_dir") or "")).strip()
            if not local_dir_str:
                reporter.error("Mode local_laz sélectionné mais aucun dossier nuages locaux n'est configuré")
                return

            local_dir = Path(local_dir_str)
            log_section("INDEXATION DES NUAGES LOCAUX", "download", slog=slog, reporter=reporter)
            reporter.stage("Indexation des nuages locaux")
            reporter.progress(0)
            result = run_local_laz(
                local_laz_dir=local_dir,
                output_dir=ctx.output_dir,
                log=lambda m: reporter.info(m),
            )

        from ...pipeline.ign.preprocess import prepare_merged_tiles

        tile_overlap = safe_float(processing.get("tile_overlap", 5), 5.0)

        log_section("FUSION DES TUILES", "process", slog=slog, reporter=reporter)
        reporter.stage("Fusion (voisins + merge)")
        if ctx.mode == "ign_laz":
            reporter.progress(merge_range[0])
        else:
            reporter.progress(0)

        max_workers = processing.get("max_workers", 4)
        merged_result = prepare_merged_tiles(
            sorted_list_file=result.sorted_list_file,
            dalles_dir=result.dalles_dir,
            output_dir=ctx.output_dir,
            tile_overlap_percent=tile_overlap,
            log=lambda m: reporter.info(m),
            cancel=lambda: cancel.is_cancelled(),
            stage=lambda s: reporter.stage(s),
            max_workers=max_workers,
        )

        if ctx.mode == "ign_laz":
            reporter.progress(merge_range[1])

        active_products: list = []

        if need_mnt and merged_result.merged_files:
            mnt_resolution = safe_float(processing.get("mnt_resolution", 0.5), 0.5)

            filter_expression = processing.get(
                "filter_expression",
                "Classification = 2 OR Classification = 6 OR Classification = 66 OR Classification = 67 OR Classification = 9",
            )

            density_resolution = safe_float(processing.get("density_resolution", 1.0), 1.0)

            output_structure = processing.get("output_structure", {})
            if not isinstance(output_structure, dict):
                output_structure = {}
            output_formats = processing.get("output_formats", {})
            if not isinstance(output_formats, dict):
                output_formats = {}

            rvt_params = ctx.rvt_params or {}
            products_cfg = products if isinstance(products, dict) else {}
            active_products = [k for k, v in products_cfg.items() if v]

            log_section("TRAITEMENT DES DALLES", "process", slog=slog, reporter=reporter)
            reporter.stage("Traitement des dalles")
            if ctx.mode == "ign_laz":
                reporter.progress(products_range[0])
            else:
                reporter.progress(0)

            total_mnt = len(merged_result.merged_files)

            for i, merged_path in enumerate(merged_result.merged_files, start=1):
                if cancel.is_cancelled():
                    break

                self._process_tile(
                    merged_path=merged_path,
                    output_dir=ctx.output_dir,
                    tile_overlap=tile_overlap,
                    mnt_resolution=mnt_resolution,
                    density_resolution=density_resolution,
                    filter_expression=str(filter_expression),
                    products_cfg=products_cfg,
                    output_structure=output_structure,
                    output_formats=output_formats,
                    rvt_params=rvt_params,
                    pyramids_config=(processing.get("pyramids") or {}),
                    reporter=reporter,
                    cancel=cancel,
                    feedback=feedback,
                    slog=slog,
                    tile_index=i,
                    total_tiles=total_mnt,
                    active_products=active_products,
                )

                if ctx.mode == "ign_laz":
                    frac = i / max(1, total_mnt)
                    pct = int(round(products_range[0] + (products_range[1] - products_range[0]) * frac))
                else:
                    pct = int(round(100.0 * i / max(1, total_mnt)))
                reporter.progress(pct)

            # Computer Vision globale (post-boucle)
            cv_cfg = ctx.cv_cfg or {}
            cv_enabled = bool(cv_cfg.get("enabled", False))
            if cv_enabled and not cancel.is_cancelled():
                try:
                    self._run_post_cv(
                        ctx=ctx,
                        output_structure=output_structure,
                        rvt_params=rvt_params,
                        reporter=reporter,
                        cancel=cancel,
                        slog=slog,
                    )
                except Exception as e:
                    reporter.error(f"Erreur Computer Vision: {e}")

        # Finalisation commune (VRT + shapefiles + load_layers)
        finalize_pipeline(
            output_dir=ctx.output_dir,
            cv_cfg=ctx.cv_cfg or {},
            rvt_params=ctx.rvt_params or {},
            reporter=reporter,
            slog=slog,
            start_time=start_time,
            tiles_processed=len(merged_result.merged_files) if merged_result else 0,
            active_products=active_products,
            extra_label="Dalles traitées",
        )
