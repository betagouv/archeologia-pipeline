from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from ..cancel_token import CancelToken
from ..progress_reporter import ProgressReporter, report_busy, report_stage_id
from ..progress_stages import Stage
from ..run_context import RunContext
from ..services.cv_post_service import _model_display_name
from ..services.finalize_service import finalize_pipeline
from ..structured_logger import log_section
from ..user_narrator import create_user_narrator
from .progress_plan import build_progress_plan, cv_pct

try:  # cross-package import : OK en QGIS, fallback en tests standalone (src/ sur le path)
    from ...pipeline.cancellation import PipelineCancelled
    from ...pipeline.batch import process_items_isolated
except ImportError:  # pragma: no cover
    from pipeline.cancellation import PipelineCancelled
    from pipeline.batch import process_items_isolated

if TYPE_CHECKING:
    from ..structured_logger import StructuredLogger


class ExistingRvtRunner:
    def run(
        self,
        ctx: RunContext,
        reporter: ProgressReporter,
        cancel: CancelToken,
        slog: Optional["StructuredLogger"] = None,
    ) -> Optional[bool]:
        try:
            from ...pipeline.modes.existing_rvt import run_existing_rvt
        except ImportError:
            from pipeline.modes.existing_rvt import run_existing_rvt

        start_time = time.time()

        # Pré-conditions garanties par validate_run_context (V3.3).
        existing_rvt_dir = ctx.files.existing_rvt_dir
        assert existing_rvt_dir is not None
        assert ctx.output_dir is not None

        processing = ctx.processing
        cv_config = ctx.cv.raw
        target_rvt = str(cv_config.get("target_rvt", "LD"))

        # Collecter tous les RVT cibles uniques depuis les runs
        try:
            from ...pipeline.cv.class_utils import resolve_cv_runs
        except ImportError:
            from pipeline.cv.class_utils import resolve_cv_runs
        cv_runs = resolve_cv_runs(cv_config)
        active_rvts = list(dict.fromkeys(
            r.get("target_rvt", target_rvt) for r in cv_runs
        )) or [target_rvt]
        run_configs = cv_runs or [dict(cv_config, enabled=False, target_rvt=target_rvt)]

        # La couleur dérive du nom de classe via le registre partagé (refonte
        # couleurs) ; plus de mapping global à pré-calculer. Paramètre conservé
        # (vide) dans la chaîne d'appel pour ne pas casser les signatures.
        global_color_map: dict = {}

        # Bande de progression : prep TIF→PNG 0–10, inférence CV 10–95.
        plan = build_progress_plan(ctx.mode, ctx.cv.enabled)

        # Section: Traitement RVT existants
        log_section("TRAITEMENT RVT EXISTANTS", "cv", slog=slog, reporter=reporter)

        # Ce mode est fondamentalement de la détection : on allume directement
        # la pastille « Détection » de la timeline.
        report_stage_id(reporter, Stage.DETECTION)
        reporter.stage("Traitement RVT existants")
        reporter.progress(plan.products[0])

        total_images = 0
        if not cv_runs and cv_config.get("enabled", False):
            reporter.info("Computer Vision: aucun modèle configuré — inférence ignorée")

        narrator = create_user_narrator(reporter)
        if cv_runs:
            narrator.cv_start(len(cv_runs))
            reporter.stage("Computer Vision")
            # La barre entre dans la bande CV (10) ; elle progresse ensuite
            # image par image via cv_pct dans _on_image_progress.
            reporter.progress(plan.cv[0])

        # total_detections par run (résumé du runner) — None exclus : si aucun
        # run n'a de résumé (fallback, vieux binaire), pas d'annonce de total.
        detection_counts: list = []

        def _process_run(run_idx: int, run_cfg: dict) -> None:
            nonlocal total_images
            run_model = run_cfg.get("selected_model", "?")
            run_rvt = run_cfg.get("target_rvt", target_rvt)
            model_display = _model_display_name(run_model)
            if cv_runs:
                reporter.info(f"Computer Vision: run {run_idx}/{len(cv_runs)} — modèle={run_model}, RVT={run_rvt}")
                narrator.cv_run_start(run_idx, len(cv_runs), model_display, run_rvt)

            # ``_model``/``_ri``/``_n`` figés par défaut-d'argument (late-binding).
            def _on_image_progress(
                idx, total, image_name,
                _model=model_display, _ri=run_idx, _n=len(cv_runs),
            ):
                narrator.cv_run_image_progress(_model, idx, total, image_name)
                reporter.progress(cv_pct(_ri, _n, idx, total, plan.cv))

            res = run_existing_rvt(
                existing_rvt_dir=existing_rvt_dir,
                output_dir=ctx.output_dir,
                cv_config=run_cfg,
                output_structure=processing.output_structure,
                log=lambda m: reporter.info(m),
                cancel_check=cancel.is_cancelled,
                rvt_params=ctx.rvt_params,
                global_color_map=global_color_map,
                indices_folder_name="RVT",
                image_progress=_on_image_progress if cv_runs else None,
                tile_progress=narrator.cv_run_tile_progress if cv_runs else None,
                on_busy=lambda active: report_busy(reporter, active),
            )
            total_images = max(total_images, res.total_images)
            if res.total_detections is not None:
                detection_counts.append(res.total_detections)

        def _on_run_failure(run_idx: int, run_cfg: dict, exc: Exception) -> None:
            model_display = _model_display_name(run_cfg.get("selected_model", "?"))
            reporter.error(
                f"Computer Vision: run {run_idx} (modèle={model_display}) en échec, "
                f"ignoré : {exc}"
            )

        # Annulation = arrêt rapide du travail lourd, puis finalisation légère
        # (VRT + projet QGIS + chargement des couches déjà produites).
        # Isolation par run : un TIF illisible / crash d'inférence n'empêche
        # plus les runs suivants ni la finalisation (AUDIT v2 ROB-13 — ce
        # runner avait été oublié par le correctif ROB-02/03/04).
        cancelled = False
        fatal = False
        final_ok: Optional[bool] = None
        try:
            _ok, failed = process_items_isolated(
                run_configs,
                _process_run,
                cancel=cancel.is_cancelled,
                on_failure=_on_run_failure,
            )
            if failed:
                reporter.error(
                    f"⚠️ Computer Vision: {len(failed)} run(s) sur "
                    f"{len(run_configs)} en échec — voir le journal."
                )
            if detection_counts:
                narrator.cv_complete(sum(detection_counts))
        except PipelineCancelled:
            cancelled = True
            reporter.info("Annulation demandée — finalisation des résultats partiels…")
        except Exception:
            # L'issue est transmise à la finalisation (pas de faux « ✅ »,
            # AUDIT v2 ROB-14) puis l'erreur remonte au contrôleur.
            fatal = True
            raise
        finally:
            # Finalisation commune LÉGÈRE — TOUJOURS exécutée, même si une
            # erreur inattendue remonte (les produits déjà calculés sont
            # indexés/chargés).
            if cancelled or cancel.is_cancelled():
                outcome = "cancelled"
            elif fatal:
                outcome = "failed"
            else:
                outcome = "success"
            final_ok = finalize_pipeline(
                output_dir=ctx.output_dir,
                cv_cfg=cv_config,
                rvt_params=ctx.rvt_params,
                reporter=reporter,
                slog=slog,
                start_time=start_time,
                tiles_processed=total_images,
                active_products=active_rvts,
                extra_label="Images traitées",
                ui_config=ctx.ui_config,
                outcome=outcome,
            )

        if cancelled or cancel.is_cancelled():
            narrator.pipeline_cancelled()
            return None
        return final_ok
