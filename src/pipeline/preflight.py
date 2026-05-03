from __future__ import annotations

import importlib
import os
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

from .types import LogFn


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    details: str
    critical: bool


def _check_exe(name: str) -> Optional[str]:
    return shutil.which(name)


def _check_input_path(
    files_cfg: Dict,
    key: str,
    label: str,
    *,
    expect_dir: bool = True,
    extensions: Optional[List[str]] = None,
    results: List[CheckResult],
) -> None:
    """Vérifie qu'un chemin d'entrée existe et contient les fichiers attendus."""
    raw = str(files_cfg.get(key, "")).strip()
    if not raw:
        results.append(CheckResult(name=label, ok=False, details="non configuré", critical=True))
        return
    p = Path(raw)
    if expect_dir:
        if not p.exists() or not p.is_dir():
            results.append(CheckResult(name=label, ok=False, details=f"introuvable: {p}", critical=True))
            return
        if extensions:
            found: List[Path] = []
            for ext in extensions:
                found.extend(p.glob(f"*.{ext}"))
            if found:
                results.append(CheckResult(name=label, ok=True, details=f"{p} ({len(found)} fichiers)", critical=True))
            else:
                exts = "/".join(extensions).upper()
                results.append(CheckResult(name=label, ok=False, details=f"{p} (aucun fichier {exts} trouvé)", critical=True))
        else:
            results.append(CheckResult(name=label, ok=True, details=str(p), critical=True))
    else:
        if p.exists() and p.is_file():
            results.append(CheckResult(name=label, ok=True, details=str(p), critical=True))
        else:
            results.append(CheckResult(name=label, ok=False, details=f"introuvable: {p}", critical=True))


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
    """Trouve le runner ONNX externe (délègue à cv.runner)."""
    from .cv.runner import _find_external_cv_runner as _find
    return _find()


def run_preflight(
    *,
    mode: str,
    cv_config: Dict,
    products: Dict,
    log: LogFn,
    files_config: Optional[Dict] = None,
    output_dir: Optional[Path] = None,
) -> bool:
    results: List[CheckResult] = []
    files_cfg = files_config or {}

    cv_enabled = bool((cv_config or {}).get("enabled", False))
    need_pdal_cli = mode in ("ign_laz", "local_laz")
    need_gdalwarp = mode in ("ign_laz", "local_laz", "existing_mnt")
    need_gdal_translate = mode in ("ign_laz", "local_laz", "existing_mnt", "existing_rvt")
    need_gdaladdo = mode in ("ign_laz", "local_laz", "existing_mnt", "existing_rvt")

    need_processing = mode in ("ign_laz", "local_laz", "existing_mnt")
    _processing_ok: Optional[bool] = None  # cache pour éviter le double import

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
            _processing_ok = True
            results.append(CheckResult(name="QGIS processing", ok=True, details="import ok", critical=True))
        except Exception as e:
            _processing_ok = False
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

    if need_gdaladdo:
        p = _check_exe("gdaladdo")
        results.append(
            CheckResult(
                name="gdaladdo",
                ok=bool(p),
                details=p or "not found in PATH",
                critical=False,
            )
        )

    need_rvt = bool(products.get("M_HS", False) or products.get("SVF", False) or products.get("SLO", False) or products.get("LD", False) or products.get("SLRM", False) or products.get("VAT", False))
    if need_rvt and mode in ("ign_laz", "local_laz", "existing_mnt"):
        if _processing_ok is True:
            results.append(CheckResult(name="RVT algos (via processing)", ok=True, details="expected available in QGIS", critical=False))
        elif _processing_ok is False:
            results.append(CheckResult(name="RVT algos (via processing)", ok=False, details="processing import failed (see above)", critical=False))
        else:
            try:
                _check_import("processing")
                results.append(CheckResult(name="RVT algos (via processing)", ok=True, details="expected available in QGIS", critical=False))
            except Exception as e:
                results.append(CheckResult(name="RVT algos (via processing)", ok=False, details=repr(e), critical=False))

    if cv_enabled:
        runner = _find_external_cv_runner()
        if runner is not None:
            results.append(CheckResult(name="cv_runner_onnx (external)", ok=True, details=str(runner), critical=False))
        else:
            expected = (
                "data/third_party/cv_runner_onnx/windows/cv_runner_onnx.exe" if os.name == "nt" else "data/third_party/cv_runner_onnx/linux/cv_runner_onnx"
            )
            results.append(CheckResult(name="cv_runner_onnx (external)", ok=False, details=f"not found (expected: {expected})", critical=False))

        # Vérifier les dépendances Python pour le fallback (si pas de runner externe)
        deps_critical = runner is None
        for mod in ("onnxruntime", "PIL"):
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

    # Vérification des chemins selon le mode
    if output_dir is not None:
        if output_dir.exists():
            results.append(CheckResult(name="Dossier de sortie", ok=True, details=str(output_dir), critical=True))
        else:
            # On peut créer le dossier, donc ce n'est pas critique
            results.append(CheckResult(name="Dossier de sortie", ok=True, details=f"{output_dir} (sera créé)", critical=True))

    if mode == "ign_laz":
        raw_input = str(files_cfg.get("input_file", "")).strip()
        suffix = Path(raw_input).suffix.lower() if raw_input else ""
        if suffix in (".shp", ".geojson", ".json", ".gpkg"):
            _check_input_path(files_cfg, "input_file", "Zone d'étude (vecteur)", expect_dir=False, results=results)
        else:
            _check_input_path(files_cfg, "input_file", "Fichier liste URLs IGN", expect_dir=False, results=results)
    elif mode == "local_laz":
        _check_input_path(files_cfg, "local_laz_dir", "Dossier LAZ locaux", extensions=["laz", "las", "LAZ", "LAS"], results=results)
    elif mode == "existing_mnt":
        _check_input_path(files_cfg, "existing_mnt_dir", "Dossier MNT existants", extensions=["tif", "TIF", "tiff"], results=results)
    elif mode == "existing_rvt":
        _check_input_path(files_cfg, "existing_rvt_dir", "Dossier RVT existants", extensions=["tif", "TIF", "tiff"], results=results)

    log("=== Préflight check: dépendances et chemins ===")
    any_critical_fail = False
    for r in results:
        status = "✓" if r.ok else "✗"
        crit = "CRITIQUE" if r.critical else "optionnel"
        log(f"[{status}] {r.name} ({crit}) -> {r.details}")
        if r.critical and not r.ok:
            any_critical_fail = True

    if any_critical_fail:
        log("❌ Préflight check: au moins une vérification CRITIQUE a échoué. Arrêt.")
        return False

    log("✅ Préflight check: OK")
    return True
