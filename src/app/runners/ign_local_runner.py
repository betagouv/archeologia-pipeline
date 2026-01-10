from __future__ import annotations

from pathlib import Path

from ..cancel_token import CancelToken
from ..progress_reporter import ProgressReporter
from ..run_context import RunContext


class IgnOrLocalRunner:
    def run(self, ctx: RunContext, reporter: ProgressReporter, cancel: CancelToken) -> None:
        if ctx.output_dir is None:
            reporter.error("Aucun dossier de sortie n'est configuré")
            return

        processing = ctx.processing_cfg or {}
        products = (processing.get("products") or {})
        if not isinstance(products, dict):
            products = {}

        need_mnt = bool(products.get("MNT", True)) or any(bool(products.get(k, False)) for k in ("M_HS", "SVF", "SLO", "LD", "VAT"))

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

            cv_config = ctx.cv_cfg or {}
            cv_enabled = bool(cv_config.get("enabled", False))
            cv_target_rvt = str(cv_config.get("target_rvt", "LD"))
            cv_generate_shapefiles = bool(cv_config.get("generate_shapefiles", False))
            cv_labels_dir = None
            cv_shp_dir = None
            cv_tif_transform_data = {}

            reporter.stage("Traitement des dalles")
            if ctx.mode == "ign_laz":
                reporter.progress(products_range[0])
            else:
                reporter.progress(0)

            total_mnt = len(merged_result.merged_files)
            for i, merged_path in enumerate(merged_result.merged_files, start=1):
                if cancel.is_cancelled():
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
                reporter.stage(f"Traitement dalle {i}/{total_mnt}: {tile_name}")

                create_terrain_model(
                    input_laz_path=merged_path,
                    temp_dir=ctx.output_dir / "temp",
                    current_tile_name=tile_name,
                    mnt_resolution=mnt_resolution,
                    tile_overlap_percent=tile_overlap,
                    filter_expression=str(filter_expression),
                    log=lambda m: reporter.info(m),
                )

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
                    )

                create_visualization_products(
                    temp_dir=ctx.output_dir / "temp",
                    current_tile_name=tile_name,
                    products=products_cfg,
                    rvt_params=rvt_params,
                    log=lambda m: reporter.info(m),
                )

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

                    if cv_enabled and bool(products_cfg.get(cv_target_rvt, False)):
                        created_by_product = (export_info or {}).get("created_jpgs_by_product") or {}
                        created_jpgs = []
                        if isinstance(created_by_product, dict) and cv_target_rvt in created_by_product:
                            created_jpgs = created_by_product.get(cv_target_rvt) or []
                        if not created_jpgs:
                            created_jpgs = (export_info or {}).get("created_jpgs") or []

                        tif_transform_data = (export_info or {}).get("tif_transform_data") or {}
                        if isinstance(tif_transform_data, dict):
                            cv_tif_transform_data.update(tif_transform_data)

                        for jpg_path in created_jpgs:
                            try:
                                if jpg_path is None:
                                    continue

                                from ...pipeline.cv.runner import run_cv_on_folder

                                jpg_dir_path = Path(jpg_path).parent
                                rvt_base_dir = jpg_dir_path.parent

                                if str(rvt_base_dir.name).upper() != str(cv_target_rvt).upper():
                                    continue

                                if cv_labels_dir is None:
                                    cv_labels_dir = jpg_dir_path
                                if cv_shp_dir is None:
                                    cv_shp_dir = rvt_base_dir / "shapefiles"

                                run_cv_on_folder(
                                    jpg_dir=jpg_dir_path,
                                    cv_config=cv_config,
                                    target_rvt=cv_target_rvt,
                                    rvt_base_dir=rvt_base_dir,
                                    tif_transform_data=cv_tif_transform_data,
                                    single_jpg=Path(jpg_path),
                                    run_shapefile_dedup=False,
                                    log=lambda m: reporter.info(m),
                                )
                            except Exception as e:
                                reporter.error(f"Erreur Computer Vision: {e}")

                if ctx.mode == "ign_laz":
                    frac_done = i / max(1, total_mnt)
                    pct_done = int(round(products_range[0] + (products_range[1] - products_range[0]) * frac_done))
                    reporter.progress(pct_done)

            if cv_enabled and cv_generate_shapefiles and cv_labels_dir is not None and cv_shp_dir is not None:
                if ctx.mode == "ign_laz":
                    reporter.stage("Finalisation (shapefiles)")
                    reporter.progress(finalize_range[0])
                try:
                    from ...pipeline.cv.runner import deduplicate_cv_shapefiles_final

                    deduplicate_cv_shapefiles_final(
                        labels_dir=cv_labels_dir,
                        shp_dir=cv_shp_dir,
                        target_rvt=cv_target_rvt,
                        cv_config=cv_config,
                        tif_transform_data=cv_tif_transform_data,
                        temp_dir=ctx.output_dir / "temp",
                        crs="EPSG:2154",
                        log=lambda m: reporter.info(m),
                    )
                except Exception as e:
                    reporter.error(f"Erreur déduplication shapefiles CV: {e}")

            reporter.progress(100)

            if cv_enabled:
                try:
                    from ...pipeline.modes.existing_rvt import run_existing_rvt

                    target_rvt = str((cv_config or {}).get("target_rvt", "LD"))
                    rvt_cfg = output_structure.get("RVT", {}) if isinstance(output_structure, dict) else {}
                    base_dir_name = str(rvt_cfg.get("base_dir", "RVT"))
                    type_dir_name = str(rvt_cfg.get(target_rvt, target_rvt))
                    generated_rvt_tif_dir = (ctx.output_dir / "results") / base_dir_name / type_dir_name / "tif"

                    if not generated_rvt_tif_dir.exists() or not generated_rvt_tif_dir.is_dir():
                        reporter.error(f"Computer Vision demandée mais aucun dossier RVT/TIF trouvé: {generated_rvt_tif_dir}")
                    else:
                        reporter.stage("Computer Vision (existing MNT)")
                        reporter.progress(90)
                        run_existing_rvt(
                            existing_rvt_dir=generated_rvt_tif_dir,
                            output_dir=ctx.output_dir,
                            cv_config=cv_config,
                            output_structure=output_structure,
                            log=lambda m: reporter.info(m),
                        )
                        if cv_generate_shapefiles:
                            reporter.info("Computer Vision (existing MNT): shapefiles générés")
                except Exception as e:
                    reporter.error(f"Erreur Computer Vision (existing MNT): {e}")

        reporter.stage("Terminé")
        reporter.progress(100)
        if ctx.mode == "ign_laz":
            reporter.info(
                f"Téléchargement IGN terminé: {result.downloaded} téléchargés, {result.skipped_existing} déjà présents (total {result.total}). Fichier trié: {result.sorted_list_file}"
            )
            reporter.info(
                f"Fusion IGN terminée: {len(merged_result.merged_files)} fichiers fusionnés. Dossier: {merged_result.merged_dir}"
            )
