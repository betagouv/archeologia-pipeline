from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RunContext:
    mode: str
    output_dir: Optional[Path]
    files_cfg: Dict[str, Any]
    processing_cfg: Dict[str, Any]
    products_cfg: Dict[str, Any]
    rvt_params: Dict[str, Any]
    cv_cfg: Dict[str, Any]


def build_run_context(config: Dict[str, Any]) -> RunContext:
    app_cfg = (config.get("app") or {}) if isinstance(config, dict) else {}
    files_cfg = (app_cfg.get("files") or {}) if isinstance(app_cfg, dict) else {}

    mode = str(files_cfg.get("data_mode") or "").strip()
    output_dir_str = str(files_cfg.get("output_dir") or "").strip()
    output_dir = Path(output_dir_str) if output_dir_str else None

    processing_cfg = (config.get("processing") or {}) if isinstance(config, dict) else {}
    if not isinstance(processing_cfg, dict):
        processing_cfg = {}

    products_cfg = (processing_cfg.get("products") or {}) if isinstance(processing_cfg, dict) else {}
    if not isinstance(products_cfg, dict):
        products_cfg = {}

    cv_cfg = (config.get("computer_vision") or {}) if isinstance(config, dict) else {}
    if not isinstance(cv_cfg, dict):
        cv_cfg = {}

    rvt_params = (config.get("rvt_params") or {}) if isinstance(config, dict) else {}
    if not isinstance(rvt_params, dict):
        rvt_params = {}

    return RunContext(
        mode=mode,
        output_dir=output_dir,
        files_cfg=files_cfg if isinstance(files_cfg, dict) else {},
        processing_cfg=processing_cfg,
        products_cfg=products_cfg,
        rvt_params=rvt_params,
        cv_cfg=cv_cfg,
    )
