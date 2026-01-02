from __future__ import annotations

import importlib
import os
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


LogFn = Callable[[str], None]


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    details: str
    critical: bool


def _check_exe(name: str) -> Optional[str]:
    return shutil.which(name)


def _check_import(module_name: str) -> str:
    mod = importlib.import_module(module_name)
    ver = getattr(mod, "__version__", None)
    mod_file = getattr(mod, "__file__", None)
    parts: List[str] = ["import ok"]
    if ver is not None:
        parts.append(f"version={ver}")
    if mod_file:
        parts.append(f"file={mod_file}")
    return " ".join([parts[0], "(" + ", ".join(parts[1:]) + ")"]) if len(parts) > 1 else parts[0]


def _find_external_cv_runner() -> Optional[Path]:
    plugin_root = Path(__file__).resolve().parents[2]
    candidates: List[Path] = []
    if os.name == "nt":
        candidates.append(plugin_root / "third_party" / "cv_runner" / "windows" / "cv_runner.exe")
    else:
        candidates.append(plugin_root / "third_party" / "cv_runner" / "linux" / "cv_runner")

    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return p
        except Exception:
            continue
    return None


def run_preflight(
    *,
    mode: str,
    cv_config: Dict,
    products: Dict,
    log: LogFn,
) -> bool:
    results: List[CheckResult] = []

    cv_enabled = bool((cv_config or {}).get("enabled", False))
    need_pdal_cli = mode in ("ign_laz", "local_laz")
    need_gdalwarp = mode in ("ign_laz", "local_laz", "existing_mnt")
    need_gdal_translate = mode in ("ign_laz", "local_laz", "existing_mnt", "existing_rvt")

    need_processing = mode in ("ign_laz", "local_laz", "existing_mnt")

    if need_pdal_cli:
        p = _check_exe("pdal")
        results.append(
            CheckResult(
                name="pdal (CLI)",
                ok=bool(p),
                details=p or "not found in PATH",
                critical=True,
            )
        )

    if need_processing:
        try:
            _check_import("processing")
            results.append(CheckResult(name="QGIS processing", ok=True, details="import ok", critical=True))
        except Exception as e:
            results.append(CheckResult(name="QGIS processing", ok=False, details=repr(e), critical=True))

    if need_gdalwarp:
        p = _check_exe("gdalwarp")
        results.append(
            CheckResult(
                name="gdalwarp",
                ok=bool(p),
                details=p or "not found in PATH",
                critical=True,
            )
        )

    if need_gdal_translate:
        p = _check_exe("gdal_translate")
        results.append(
            CheckResult(
                name="gdal_translate",
                ok=bool(p),
                details=p or "not found in PATH",
                critical=(mode in ("existing_mnt", "existing_rvt")),
            )
        )

    need_rvt = bool(products.get("M_HS", False) or products.get("SVF", False) or products.get("SLO", False) or products.get("LD", False) or products.get("VAT", False))
    if need_rvt and mode in ("ign_laz", "local_laz", "existing_mnt"):
        try:
            _check_import("processing")
            results.append(CheckResult(name="RVT algos (via processing)", ok=True, details="expected available in QGIS", critical=False))
        except Exception as e:
            results.append(CheckResult(name="RVT algos (via processing)", ok=False, details=repr(e), critical=False))

    if cv_enabled:
        runner = _find_external_cv_runner()
        if runner is not None:
            results.append(CheckResult(name="cv_runner (external)", ok=True, details=str(runner), critical=True))
        else:
            expected = (
                "third_party/cv_runner/windows/cv_runner.exe" if os.name == "nt" else "third_party/cv_runner/linux/cv_runner"
            )
            results.append(CheckResult(name="cv_runner (external)", ok=False, details=f"not found (expected: {expected})", critical=False))

        deps_critical = runner is None
        for mod in ("ultralytics", "sahi", "PIL"):
            if runner is not None:
                results.append(
                    CheckResult(
                        name=f"python:{mod}",
                        ok=True,
                        details="skipped (external runner present)",
                        critical=False,
                    )
                )
                continue

            try:
                details = _check_import(mod)
                results.append(CheckResult(name=f"python:{mod}", ok=True, details=details, critical=deps_critical))
            except Exception as e:
                results.append(CheckResult(name=f"python:{mod}", ok=False, details=repr(e), critical=deps_critical))

        try:
            details = _check_import("geopandas")
            results.append(CheckResult(name="python:geopandas", ok=True, details=details, critical=False))
        except Exception as e:
            results.append(CheckResult(name="python:geopandas", ok=False, details=repr(e), critical=False))

    log("=== Préflight check: dépendances ===")
    any_critical_fail = False
    for r in results:
        status = "OK" if r.ok else "KO"
        crit = "CRITIQUE" if r.critical else "optionnel"
        log(f"[{status}] {r.name} ({crit}) -> {r.details}")
        if r.critical and not r.ok:
            any_critical_fail = True

    if any_critical_fail:
        log("❌ Préflight check: au moins une dépendance CRITIQUE est manquante. Arrêt.")
        return False

    log("✅ Préflight check: OK")
    return True
