from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from .cancel_token import CancelToken
from .progress_reporter import ProgressReporter
from .run_context import RunContext, validate_run_context
from .structured_logger import StructuredLogger, create_structured_logger
from .user_narrator import create_user_narrator


def _files_as_dict(ctx: RunContext) -> dict:
    """Adapte ``ctx.files`` au contrat dict-only de ``preflight``.

    ``preflight`` est documenté comme ne dépendant que de la stdlib +
    quelques utilitaires CLI ; on ne le typage pas dans ce refactor pour
    éviter d'élargir le périmètre. À supprimer quand ``run_preflight``
    consommera ``FilesConfig`` directement.
    """
    f = ctx.files
    return {
        "data_mode": f.data_mode,
        "output_dir": str(f.output_dir) if f.output_dir else "",
        "input_file": str(f.input_file) if f.input_file else "",
        "local_laz_dir": str(f.local_laz_dir) if f.local_laz_dir else "",
        "existing_mnt_dir": str(f.existing_mnt_dir) if f.existing_mnt_dir else "",
        "existing_rvt_dir": str(f.existing_rvt_dir) if f.existing_rvt_dir else "",
    }


# Libellés FR des modes pour le narrateur (pas de jargon "ign_laz" en UI).
MODE_LABELS = {
    "ign_laz": "téléchargement IGN",
    "local_laz": "nuages LAZ locaux",
    "existing_mnt": "MNT existant",
    "existing_rvt": "RVT existant",
}


@contextmanager
def file_logging(output_dir: Optional[Path], reporter: ProgressReporter) -> Iterator[None]:
    file_handler = None
    root_logger = None
    root_prev_level = None
    # Le logger "archeologia_pipeline" a propagate=False (pour éviter les
    # doublons dans la console QGIS).  On doit lui attacher le FileHandler
    # directement, sinon tous les messages émis via reporter.info() / slog
    # sont perdus dans le fichier de log.
    app_logger = logging.getLogger("archeologia_pipeline")

    try:
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            log_path = output_dir / f"pipeline_log_{ts}.txt"
            file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
            # INFO pour avoir tous les logs techniques dans le fichier
            # (alors que l'UI filtre à USER_INFO=25 et n'en voit que les
            # narratifs).
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            root_logger = logging.getLogger()
            root_prev_level = root_logger.level
            if root_prev_level > logging.INFO:
                root_logger.setLevel(logging.INFO)
            root_logger.addHandler(file_handler)
            app_logger.addHandler(file_handler)
            reporter.info(f"Logs écrits dans: {log_path}")
        yield
    finally:
        try:
            if file_handler is not None:
                app_logger.removeHandler(file_handler)
                if root_logger is not None:
                    root_logger.removeHandler(file_handler)
                file_handler.close()
        except Exception:
            pass
        try:
            if root_logger is not None and root_prev_level is not None:
                root_logger.setLevel(root_prev_level)
        except Exception:
            pass


class PipelineController:
    def run(self, ctx: RunContext, reporter: ProgressReporter, cancel: CancelToken) -> None:
        # slog → écrit via reporter.info (level INFO) : visible UNIQUEMENT
        # dans le fichier de log (UI filtre à USER_INFO=25).
        slog = create_structured_logger(reporter.info)
        # narrator → écrit via reporter.user_info (level USER_INFO=25) :
        # visible UI **et** fichier.
        narrator = create_user_narrator(reporter)

        output_str = str(ctx.output_dir) if ctx.output_dir is not None else ""
        slog.start_pipeline(ctx.mode, output_str)
        narrator.pipeline_starting(MODE_LABELS.get(ctx.mode, ctx.mode))

        # Validation métier centralisée (mode + chemins requis).
        # Sépare les erreurs config-side ("dossier MNT non renseigné")
        # des erreurs preflight ("pdal n'est pas installé").
        # Les warnings sont tracés dans le fichier de log mais ne
        # bloquent pas l'exécution.
        ctx_errors, ctx_warnings = validate_run_context(ctx)
        for warn in ctx_warnings:
            slog.warning(warn)
        if ctx_errors:
            for err in ctx_errors:
                reporter.error(err)
            slog.end_pipeline(success=False)
            narrator.preflight_failed()
            return

        slog.section("VÉRIFICATION DES DÉPENDANCES", "info")

        from ..pipeline.preflight import run_preflight

        if not run_preflight(
            mode=str(ctx.mode),
            cv_config=ctx.cv.raw,
            products=ctx.processing.products.as_dict(),
            log=lambda m: reporter.info(m),
            files_config=_files_as_dict(ctx),
            output_dir=ctx.output_dir,
        ):
            slog.end_pipeline(success=False)
            narrator.preflight_failed()
            return

        narrator.preflight_ok()

        if cancel.is_cancelled():
            reporter.info("Annulation demandée avant le lancement du pipeline.")
            slog.end_pipeline(success=False)
            narrator.pipeline_cancelled()
            return

        from .runners.registry import get_runner

        runner = get_runner(ctx.mode)
        runner.run(ctx=ctx, reporter=reporter, cancel=cancel, slog=slog)
