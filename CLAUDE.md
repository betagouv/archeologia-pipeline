# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

QGIS plugin (Python) that runs a LiDAR processing pipeline → DTM / density / RVT indices, with an optional ONNX-based computer-vision detection step. README.md is the authoritative reference (~820 lines, French) — read the relevant section before doing anything non-trivial. This file captures only what isn't obvious from the file tree.

## Two execution contexts

Code in this repo runs in **one of two contexts**, and most surprises come from confusing them:

1. **Inside QGIS** (production): `__init__.py` → `main.py:ArcheologiaPipelinePlugin` is loaded by QGIS. `qgis.core`, `qgis.processing`, and `osgeo` are available. UI lives in `src/ui/main_dialog.py`. Pipeline modules under `src/pipeline/` import QGIS at module load time.
2. **Standalone** (tests / dev tooling): no QGIS available. `conftest.py` and `pytest.ini` deliberately exclude `src/ui/` and `src/pipeline/` from pytest collection (`norecursedirs`, `collect_ignore_glob`) because they would fail to import. Only modules under `src/app/` and pure helpers can be unit-tested directly. Don't add `from qgis.*` imports at module top level in code that needs to be testable — defer them inside functions, as `main.py:run()` already does.

## Common commands

```bash
# Tests — run from repo root, NOT pytest directly (run_tests.py sets sys.path)
python run_tests.py                  # all tests
python run_tests.py unit             # tests/unit only
python run_tests.py integration      # tests/integration only
python run_tests.py -k helpers       # filter by name (passes through to pytest)

ruff check src/                      # lint

# Dev tooling (all under dev/, excluded from packaged ZIP)
python dev/package_plugin.py                      # produces main.zip for QGIS install
python dev/runner_onnx/build.py [--gpu|--clean]   # compile cv_runner_onnx executable
python dev/runner_onnx/export_to_onnx.py ...      # convert .pt/.pth → .onnx (see README Tâche 2)

# Dependency install (split by task, see dev/requirements/)
pip install -r dev/requirements/test.txt          # pytest + ruff
pip install -r dev/requirements/export.txt        # ultralytics/torch/onnx (model export only)
pip install -r dev/requirements/build.txt         # pyinstaller/onnxruntime (runner build only)
pip install -r dev/requirements.txt               # all of the above
```

QGIS-side manual test checklist: `tests/TESTS_MANUELS_QGIS.txt`.

## Pipeline architecture (the parts that span multiple files)

Entry: `main.py` → `MainDialog` → worker thread → `PipelineController.run(ctx, reporter, cancel)` (`src/app/pipeline_controller.py`).

`PipelineController` does **three things only**:
1. `run_preflight(...)` — `src/pipeline/preflight.py` checks CLI tools (`pdal`, `gdalwarp`, `gdal_translate`, optional `gdaladdo`), QGIS Processing availability, RVT algos, and input paths. Returns False → pipeline aborts.
2. `get_runner(ctx.mode)` — `src/app/runners/registry.py` dispatches on `data_mode` to one of 4 runners.
3. `runner.run(...)` — runner does its mode-specific work, then calls the **shared** `finalize_pipeline(...)` from `src/app/services/finalize_service.py`.

The 4 modes (registered in `src/app/runners/registry.py`):

| `data_mode` | Runner | Purpose |
|---|---|---|
| `ign_laz` | `IgnOrLocalRunner` | Download IGN LiDAR HD tiles → MNT/RVT |
| `local_laz` | `IgnOrLocalRunner` | Same flow, local LAZ instead of download |
| `existing_mnt` | `ExistingMntRunner` | Skip LiDAR, compute RVT from existing DTM |
| `existing_rvt` | `ExistingRvtRunner` | Skip everything, just run CV on existing RVT TIFs |

All runners implement the `ModeRunner` Protocol (`src/app/runners/base.py`).

`RunContext` (frozen dataclass, `src/app/run_context.py`) is built from `config.json` by `build_run_context(config)`. The UI also writes `last_ui_config.json` for session persistence (gitignored).

### Computer vision: inference vs post-processing (critical distinction)

The CV pipeline is split across two execution boundaries:

- **Inference** is done by the external `cv_runner_onnx` subprocess (compiled binary in `data/third_party/cv_runner_onnx/<os>/`). It only emits raw JSON/TXT detections + optional annotated JPGs. If the runner is missing, `src/pipeline/cv/runner.py:_run_fallback_inference()` runs ONNX in-process via `computer_vision_onnx.py`.
- **Shapefile generation, polygon merging, overlap suppression, DBSCAN clustering, class filtering, and the consolidated `.qgs` project** all happen in the plugin's Python (under `src/pipeline/cv/`). The runner does **not** depend on `shapely`/`geopandas`/`fiona` — those are required only on the QGIS side.

Per-model behavior (clustering, SAHI slicing, image size, task type) lives in each model's `args.yaml` (in `data/models/<name>/`). `selected_classes` filtering: `None` = all classes, `[]` = short-circuit (no inference), `[x, y]` = explicit filter. The empty-list short-circuit is in `run_cv_on_folder()` and runs **before** any inference.

Multiple CV models can run in one pipeline (`computer_vision.runs` array in config). The consolidated output is always a single `detections/detections_validation.qgs` produced by `finalize_service`.

### Large rasters (`existing_mnt` / `existing_rvt` regime)

`_classify_mnt_layout` / `_classify_rvt_layout` (in `src/pipeline/modes/`) inspect raster bounds:
- **standard** (~1×1 km, IGN-aligned): legacy crop + IGN naming
- **small** (<1 km or unaligned): no crop, native extent preserved
- **large** (>1.05 km in any dim): no pre-tile-splitting at all — RVT computed on the full raster, SAHI handles 640×640 slicing in memory at inference. `Image.MAX_IMAGE_PIXELS = None` is set in `convert_tif_to_png.py` and `computer_vision_onnx.py`. Input MUST be EPSG:2154 (Lambert-93).

Don't reintroduce pre-tile-splitting for the large regime — it was removed deliberately to avoid NoData artifacts on sub-tile borders.

## Packaging conventions

`dev/package_plugin.py` produces `main.zip`. It excludes `dev/`, `tests/`, `.git/`, `.githooks/`, `__pycache__/`, virtualenvs, and named files (`config.json`, `pytest.ini`, `conftest.py`, `run_tests.py`, `.talismanrc`, `.gitignore`). It also strips `.pt`/`.pth` checkpoints (only `.onnx` ships). When adding new dev-only files, either put them under `dev/` or extend the exclusion lists in `package_plugin.py` — otherwise they end up in users' QGIS profile.

The `data/` directory is partially gitignored: `data/models/**`, `data/quadrillage_france/`, and the compiled CV runner binaries are NOT versioned (see `.gitignore`).

## Git hooks (Talisman)

A `pre-push` hook based on Talisman lives in `.githooks/` and must be enabled per-clone:

```bash
git config core.hooksPath .githooks
```

`.talismanrc` contains per-file checksums; if you legitimately modify a file Talisman flags, the checksum needs updating (recent commits `f3b5e05`, `a92bb51` are examples). Don't bypass with `--no-verify` without understanding the flag.
