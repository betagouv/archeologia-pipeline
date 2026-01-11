from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..cancel_token import CancelToken
from ..cancellable_feedback import create_cancellable_feedback
from ..progress_reporter import ProgressReporter
from ..run_context import RunContext
from ..services.cv_service import ComputerVisionService

if TYPE_CHECKING:
    from ..structured_logger import StructuredLogger


class IgnOrLocalRunner:
    def run(
        self,
        ctx: RunContext,
        reporter: ProgressReporter,
        cancel: CancelToken,
        slog: Optional["StructuredLogger"] = None,
    ) -> None:
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
        finalize_range = (95, 100)

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
            if slog:
                slog.section("TÉLÉCHARGEMENT DES DALLES IGN", "download")
            else:
                reporter.info("")
                reporter.info("════════════════════════════════════════════════════════════")
                reporter.info("📥 TÉLÉCHARGEMENT DES DALLES IGN")
                reporter.info("════════════════════════════════════════════════════════════")
            result = download_ign_dalles(
                input_file=input_path,
                output_dir=ctx.output_dir,
                log=lambda m: reporter.info(m),
                progress=lambda p: reporter.progress(
                    int(download_range[0] + (download_range[1] - download_range[0]) * (int(p) / 100.0))
                ),
                stage=lambda s: reporter.stage(str(s)),
                cancel=lambda: cancel.is_cancelled(),
            )
        else:
            from ...pipeline.modes.local_laz import run_local_laz

            local_dir_str = str((ctx.files_cfg.get("local_laz_dir") or "")).strip()
            if not local_dir_str:
                reporter.error("Mode local_laz sélectionné mais aucun dossier nuages locaux n'est configuré")
                return

            local_dir = Path(local_dir_str)
            if slog:
                slog.section("INDEXATION DES NUAGES LOCAUX", "download")
            else:
                reporter.info("")
                reporter.info("════════════════════════════════════════════════════════════")
                reporter.info("📂 INDEXATION DES NUAGES LOCAUX")
                reporter.info("════════════════════════════════════════════════════════════")
                reporter.stage("Indexation des nuages locaux")
            reporter.progress(0)
            result = run_local_laz(
                local_laz_dir=local_dir,
                output_dir=ctx.output_dir,
                log=lambda m: reporter.info(m),
            )

        from ...pipeline.ign.preprocess import prepare_merged_tiles

        tile_overlap = processing.get("tile_overlap", 5)
        try:
            tile_overlap = float(tile_overlap)
        except Exception:
            tile_overlap = 5.0

        if slog:
            slog.section("FUSION DES TUILES", "process")
        else:
            reporter.info("")
            reporter.info("════════════════════════════════════════════════════════════")
            reporter.info("🔧 FUSION DES TUILES")
            reporter.info("════════════════════════════════════════════════════════════")
            reporter.stage("Fusion (voisins + merge)")
        if ctx.mode == "ign_laz":
            reporter.progress(merge_range[0])
        else:
            reporter.progress(0)

        merged_result = prepare_merged_tiles(
            sorted_list_file=result.sorted_list_file,
            dalles_dir=result.dalles_dir,
            output_dir=ctx.output_dir,
            tile_overlap_percent=tile_overlap,
            log=lambda m: reporter.info(m),
            cancel=lambda: cancel.is_cancelled(),
        )

        if ctx.mode == "ign_laz":
            reporter.progress(merge_range[1])

        if need_mnt and merged_result.merged_files:
            from ...pipeline.ign.products.crop import crop_final_products
            from ...pipeline.ign.products.density import create_density_map
            from ...pipeline.ign.products.indices import create_visualization_products
            from ...pipeline.ign.products.mnt import create_terrain_model
            from ...pipeline.ign.products.results import copy_final_products_to_results

            mnt_resolution = processing.get("mnt_resolution", 0.5)
            try:
                mnt_resolution = float(mnt_resolution)
            except Exception:
                mnt_resolution = 0.5

            filter_expression = processing.get(
                "filter_expression",
                "Classification = 2 OR Classification = 6 OR Classification = 66 OR Classification = 67 OR Classification = 9",
            )

            density_resolution = processing.get("density_resolution", 1.0)
            try:
                density_resolution = float(density_resolution)
            except Exception:
                density_resolution = 1.0

            output_structure = processing.get("output_structure", {})
            if not isinstance(output_structure, dict):
                output_structure = {}
            output_formats = processing.get("output_formats", {})
            if not isinstance(output_formats, dict):
                output_formats = {}

            rvt_params = ctx.rvt_params or {}

            cv_service = ComputerVisionService(
                cv_config=ctx.cv_cfg or {},
                output_dir=ctx.output_dir,
                log=lambda m: reporter.info(m),
            )

            if slog:
                slog.section("TRAITEMENT DES DALLES", "process")
            else:
                reporter.info("")
                reporter.info("════════════════════════════════════════════════════════════")
                reporter.info("🔧 TRAITEMENT DES DALLES")
                reporter.info("════════════════════════════════════════════════════════════")
                reporter.stage("Traitement des dalles")
            if ctx.mode == "ign_laz":
                reporter.progress(products_range[0])
            else:
                reporter.progress(0)

            total_mnt = len(merged_result.merged_files)
            active_products = [k for k, v in products.items() if v]
            
            for i, merged_path in enumerate(merged_result.merged_files, start=1):
                if cancel.is_cancelled():
                    if slog:
                        slog.warning("Annulation demandée par l'utilisateur")
                    else:
                        reporter.info("Annulation demandée")
                    break

                if ctx.mode == "ign_laz":
                    frac = (i - 1) / max(1, total_mnt)
                    pct = int(round(products_range[0] + (products_range[1] - products_range[0]) * frac))
                    reporter.progress(pct)
                else:
                    pct = int(round(100.0 * (i - 1) / max(1, total_mnt)))
                    reporter.progress(pct)

                tile_name = merged_path.name.replace(".copc.laz", "").replace(".laz", "")
                if slog:
                    slog.tile_start(i, total_mnt, tile_name)
                else:
                    reporter.stage(f"Traitement dalle {i}/{total_mnt}: {tile_name}")

                create_terrain_model(
                    input_laz_path=merged_path,
                    temp_dir=ctx.output_dir / "temp",
                    current_tile_name=tile_name,
                    mnt_resolution=mnt_resolution,
                    tile_overlap_percent=tile_overlap,
                    filter_expression=str(filter_expression),
                    log=lambda m: reporter.info(m),
                    feedback=feedback,
                )
                
                if cancel.is_cancelled():
                    if slog:
                        slog.warning("Annulation demandée par l'utilisateur")
                    else:
                        reporter.info("Annulation demandée")
                    break

                products_cfg = products if isinstance(products, dict) else {}
                if bool(products_cfg.get("DENSITE", False)):
                    create_density_map(
                        input_laz_path=merged_path,
                        temp_dir=ctx.output_dir / "temp",
                        current_tile_name=tile_name,
                        density_resolution=density_resolution,
                        tile_overlap_percent=tile_overlap,
                        filter_expression=str(filter_expression),
                        log=lambda m: reporter.info(m),
                        feedback=feedback,
                    )
                    
                    if cancel.is_cancelled():
                        if slog:
                            slog.warning("Annulation demandée par l'utilisateur")
                        else:
                            reporter.info("Annulation demandée")
                        break

                create_visualization_products(
                    temp_dir=ctx.output_dir / "temp",
                    current_tile_name=tile_name,
                    products=products_cfg,
                    rvt_params=rvt_params,
                    log=lambda m: reporter.info(m),
                    feedback=feedback,
                )
                
                if cancel.is_cancelled():
                    if slog:
                        slog.warning("Annulation demandée par l'utilisateur")
                    else:
                        reporter.info("Annulation demandée")
                    break

                cropped = crop_final_products(
                    temp_dir=ctx.output_dir / "temp",
                    current_tile_name=tile_name,
                    products=products_cfg,
                    rvt_params=rvt_params,
                    log=lambda m: reporter.info(m),
                )

                if cropped:
                    export_info = copy_final_products_to_results(
                        temp_dir=ctx.output_dir / "temp",
                        output_dir=ctx.output_dir,
                        current_tile_name=tile_name,
                        products=products_cfg,
                        output_structure=output_structure,
                        output_formats=output_formats,
                        rvt_params=rvt_params,
                        pyramids_config=(processing.get("pyramids") or {}),
                        log=lambda m: reporter.info(m),
                    )

                    if cv_service.enabled and cv_service.should_process_product(cv_service.target_rvt):
                        created_by_product = (export_info or {}).get("created_jpgs_by_product") or {}
                        created_jpgs = []
                        if isinstance(created_by_product, dict) and cv_service.target_rvt in created_by_product:
                            created_jpgs = created_by_product.get(cv_service.target_rvt) or []
                        if not created_jpgs:
                            created_jpgs = (export_info or {}).get("created_jpgs") or []

                        tif_transform_data = (export_info or {}).get("tif_transform_data") or {}

                        for jpg_path in created_jpgs:
                            if jpg_path is None:
                                continue
                            jpg_dir_path = Path(jpg_path).parent
                            rvt_base_dir = jpg_dir_path.parent
                            if str(rvt_base_dir.name).upper() != cv_service.target_rvt.upper():
                                continue
                            cv_service.process_single_jpg(
                                jpg_path=Path(jpg_path),
                                rvt_base_dir=rvt_base_dir,
                                tif_transform_data=tif_transform_data,
                            )

                if slog:
                    slog.tile_end(tile_name, active_products)
                    
                if ctx.mode == "ign_laz":
                    frac_done = i / max(1, total_mnt)
                    pct_done = int(round(products_range[0] + (products_range[1] - products_range[0]) * frac_done))
                    reporter.progress(pct_done)

            if cv_service.enabled and cv_service.generate_shapefiles:
                if ctx.mode == "ign_laz":
                    reporter.stage("Finalisation (shapefiles)")
                    reporter.progress(finalize_range[0])
                cv_service.finalize(temp_dir=ctx.output_dir / "temp")

            reporter.progress(100)

            if cv_service.enabled:
                try:
                    from ...pipeline.modes.existing_rvt import run_existing_rvt

                    rvt_cfg = output_structure.get("RVT", {}) if isinstance(output_structure, dict) else {}
                    base_dir_name = str(rvt_cfg.get("base_dir", "RVT"))
                    type_dir_name = str(rvt_cfg.get(cv_service.target_rvt, cv_service.target_rvt))
                    generated_rvt_tif_dir = (ctx.output_dir / "results") / base_dir_name / type_dir_name / "tif"

                    if not generated_rvt_tif_dir.exists() or not generated_rvt_tif_dir.is_dir():
                        reporter.error(f"Computer Vision demandée mais aucun dossier RVT/TIF trouvé: {generated_rvt_tif_dir}")
                    else:
                        if slog:
                            slog.section("COMPUTER VISION", "cv")
                        else:
                            reporter.info("")
                            reporter.info("════════════════════════════════════════════════════════════")
                            reporter.info("🤖 COMPUTER VISION")
                            reporter.info("════════════════════════════════════════════════════════════")
                        reporter.stage("Computer Vision")
                        reporter.progress(90)
                        run_existing_rvt(
                            existing_rvt_dir=generated_rvt_tif_dir,
                            output_dir=ctx.output_dir,
                            cv_config=ctx.cv_cfg or {},
                            output_structure=output_structure,
                            log=lambda m: reporter.info(m),
                        )
                        if cv_service.generate_shapefiles:
                            reporter.info("Computer Vision (existing MNT): shapefiles générés")
                except Exception as e:
                    reporter.error(f"Erreur Computer Vision (existing MNT): {e}")

        reporter.progress(100)
        
        # Calcul des statistiques finales
        elapsed = time.time() - start_time
        tiles_processed = len(merged_result.merged_files) if 'merged_result' in dir() and merged_result else 0
        products_list = active_products if 'active_products' in dir() else []

        if slog:
            slog.end_pipeline(
                success=True,
                tiles_processed=tiles_processed,
                tiles_total=tiles_processed,
                products=products_list,
            )
        else:
            # Section finale
            reporter.info("")
            reporter.info("════════════════════════════════════════════════════════════")
            reporter.info("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
            reporter.info("════════════════════════════════════════════════════════════")
            reporter.info(f"  ⏱️ Durée totale : {elapsed:.1f}s")
            reporter.info(f"  📄 Dalles traitées : {tiles_processed}")
            reporter.info(f"  📦 Produits : {', '.join(products_list) if products_list else 'aucun'}")
            reporter.info("════════════════════════════════════════════════════════════")
            reporter.info("")
            reporter.stage("Terminé")
