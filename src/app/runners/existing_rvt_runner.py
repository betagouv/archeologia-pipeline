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
        
        # Charger les couleurs de classes depuis le modèle
        class_colors = None
        try:
            from ...pipeline.cv.class_utils import load_class_colors_from_model
            selected_model = str(cv_config.get("selected_model", "")).strip()
            models_dir = Path(cv_config.get("models_dir", "models"))
            weights_path = Path(selected_model)
            if not weights_path.exists() or not weights_path.is_file():
                weights_path = models_dir / selected_model / "weights" / "best.pt"
            if weights_path.exists():
                class_colors = load_class_colors_from_model(weights_path)
        except Exception:
            pass

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
            cancel_check=cancel.is_cancelled,
        )

        # Création des fichiers VRT pour indexer les dalles par produit
        from ...pipeline.ign.products.results import build_vrt_index
        from typing import List
        reporter.info("Création des fichiers VRT d'indexation...")
        
        vrt_paths: List[str] = []
        shapefile_paths: List[str] = []
        
        results_dir = ctx.output_dir / "results"
        if results_dir.exists():
            # VRT pour chaque dossier de produit TIF - collecter TOUS les VRT créés
            for tif_dir in results_dir.rglob("tif"):
                if tif_dir.is_dir() and list(tif_dir.glob("*.tif")):
                    build_vrt_index(tif_dir, pattern="*.tif", output_name="index.vrt", log=lambda m: reporter.info(m))
                    vrt_path = tif_dir / "index.vrt"
                    if vrt_path.exists():
                        vrt_paths.append(str(vrt_path))
            # VRT pour chaque dossier JPG (images géoréférencées)
            for jpg_dir in results_dir.rglob("jpg"):
                if jpg_dir.is_dir() and list(jpg_dir.glob("*.jpg")):
                    build_vrt_index(jpg_dir, pattern="*.jpg", output_name="index.vrt", log=lambda m: reporter.info(m))
            # VRT pour annotated_images si présent
            annotated_dir = results_dir / "annotated_images"
            if annotated_dir.exists() and list(annotated_dir.glob("*.jpg")):
                build_vrt_index(annotated_dir, pattern="*.jpg", output_name="index.vrt", log=lambda m: reporter.info(m))

            # Collecter les shapefiles de détection CV (uniquement du dossier shapefiles principal)
            shapefiles_dir = results_dir / "RVT" / target_rvt / "shapefiles"
            if shapefiles_dir.exists():
                for shp_file in shapefiles_dir.glob("*.shp"):
                    shapefile_paths.append(str(shp_file))

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

        # Charger les couches dans le projet QGIS courant
        if vrt_paths or shapefile_paths:
            reporter.stage("Chargement des couches")
            reporter.info(f"Chargement de {len(vrt_paths)} VRT et {len(shapefile_paths)} shapefile(s) dans QGIS...")
            try:
                reporter.load_layers(vrt_paths, shapefile_paths, class_colors)
            except Exception as e:
                reporter.info(f"Note: Chargement des couches non disponible ({e})")

        reporter.stage("Terminé")
        reporter.progress(100)
