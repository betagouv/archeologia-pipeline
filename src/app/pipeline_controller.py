from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from .cancel_token import CancelToken
from .progress_reporter import ProgressReporter
from .run_context import RunContext


@contextmanager
def file_logging(output_dir: Optional[Path], reporter: ProgressReporter) -> Iterator[None]:
    file_handler = None
    root_logger = None
    root_prev_level = None

    try:
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            log_path = output_dir / f"pipeline_log_{ts}.txt"
            file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            root_logger = logging.getLogger()
            root_prev_level = root_logger.level
            if root_prev_level > logging.INFO:
                root_logger.setLevel(logging.INFO)
            root_logger.addHandler(file_handler)
            reporter.info(f"Logs écrits dans: {log_path}")
        yield
    finally:
        try:
            if file_handler is not None:
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
        reporter.info(f"Mode: {ctx.mode}")
        reporter.info(f"Sortie: {str(ctx.output_dir) if ctx.output_dir is not None else ''}")

        from ..pipeline.preflight import run_preflight

        if not run_preflight(
            mode=str(ctx.mode),
            cv_config=ctx.cv_cfg,
            products=ctx.products_cfg,
            log=lambda m: reporter.info(m),
        ):
            return

        from .runners.registry import get_runner

        runner = get_runner(ctx.mode)
        runner.run(ctx=ctx, reporter=reporter, cancel=cancel)
