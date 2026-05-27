# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

QGIS plugin (Python) that runs a LiDAR processing pipeline → DTM / density / RVT indices, with an optional ONNX-based computer-vision detection step. README.md is the authoritative reference (~820 lines, French) — read the relevant section before doing anything non-trivial. This file captures only what isn't obvious from the file tree.

## Two execution contexts

Code in this repo runs in **one of two contexts**, and most surprises come from confusing them:

1. **Inside QGIS** (production): `__init__.py` → `main.py:ArcheologiaPipelinePlugin` is loaded by QGIS. `qgis.core`, `qgis.processing`, and `osgeo` are available. UI is the 4-step wizard `src/ui/wizard_dialog.py` (pages in `src/ui/steps/`, run view in `src/ui/run_view.py`). Pipeline modules under `src/pipeline/` import QGIS at module load time.
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

Entry: `main.py` → `WizardDialog` (étape 4 → `LaunchPage`/`RunView`) → worker thread → `PipelineController.run(ctx, reporter, cancel)` (`src/app/pipeline_controller.py`).

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
- **Shapefile generation, polygon merging, overlap suppression, DBSCAN clustering, sub-threshold confidence filtering, and class filtering** all happen in the plugin's Python (under `src/pipeline/cv/`). The runner does **not** depend on `shapely`/`geopandas`/`fiona` — those are required only on the QGIS side. The consolidated `.qgs` project itself is written separately (see below), **not** under `src/pipeline/cv/`.

Per-model behavior (clustering, SAHI slicing, image size, task type) lives in each model's `args.yaml` (in `data/models/<name>/`). `selected_classes` filtering: `None` = all classes, `[]` = short-circuit (no inference), `[x, y]` = explicit filter. The empty-list short-circuit is in `run_cv_on_folder()` and runs **before** any inference.

Multiple CV models can run in one pipeline (`computer_vision.runs` array in config). The consolidated output is always a single `detections/detections_validation.qgs`, written by **`src/ui/qgs_writer.py:write_validation_project`** via the QGIS API (`QgsProject.write`) on the **main thread**, triggered from `run_view._on_load_layers` (same path as live layer loading). It is **not** written by `finalize_service` (worker thread, QGIS API not thread-safe) — `finalize_service` only emits the `load_layers` signal. Both the live load (`ui/layer_loader.py`) and the `.qgs` write share the layer factory `build_detection_vector_layer`.

**Confidence threshold = symbology = filtering (per-entity invariant).** The detection confidence threshold is set **per entity** (advanced settings → `computer_vision.runs[].confidence_threshold`), **not** the top-level `computer_vision.confidence_threshold` (a fallback only). Three things must use the *same* per-entity threshold or detections silently vanish: (1) `conf_bin` binning at conversion (`runner_shapefiles` → `create_shapefile_from_detections`); (2) **sub-threshold filtering** — `class_utils.filter_detections_below_confidence` drops `confidence < threshold` detections from the `.gpkg`, applied **after** clustering (so the DBSCAN `min_confidence_extend` hysteresis still absorbs sub-threshold points) and **exempting cluster output classes**; (3) the categorized `.qgs` symbology — `finalize_service.build_min_confidence_by_slug` maps each entity slug → its run threshold, fed through `run_view._on_load_layers` to `layer_loader`/`qgs_writer`, which resolve it **per layer** before `build_detection_vector_layer`. A QGIS categorized renderer matches by exact string with no "other values" bucket, so a legend bin (`[0.2:0.4[`) that doesn't match the data's `conf_bin` (`[0.3:0.4[`) renders **nothing**. Never reintroduce a single global threshold on the symbology side.

### Index folder naming (`indices/<PRODUCT><param-suffix>/`)

Index output folders are named `<PRODUCT>` + a parameter suffix derived from the RVT settings — e.g. `SVF_R10_D16_V1_N0`, `LD_A15_Rmin10_Rmax20_H1p7_V1`, `HS_Az315_E35_V1`. The single source of truth is `get_rvt_folder_name(product, rvt_params)` in `src/pipeline/ign/products/rvt_naming.py` (= product code + `get_rvt_param_suffix(...)`). This lets re-running into the same `output_dir` with different params produce separate folders instead of overwriting. MNT/DENSITE have no params → bare `MNT`/`DENSITE`; `existing_rvt` mode forces `indices/RVT/` (params unknown).

**Invariant — same `rvt_params` on both sides.** `get_rvt_param_suffix` applies *defaults* for missing keys (empty dict → the default suffix, **not** `""`). Creation (`results.py:copy_final_products_to_results`) and CV consumption (`output_paths.resolve_rvt_tif_dir` → fed by `cv_post_service`) must therefore receive the *same* `rvt_params`, or they resolve to *different* folders. `run_existing_rvt` mirrors this: when `indices_folder_name` is None it re-derives the suffixed name via `get_rvt_folder_name`. `output_paths.py` imports `rvt_naming` **deferred** (top-level import would pull `ign/products/__init__` → QGIS, breaking standalone/test imports).

### Entity orchestration (UI → `computer_vision.runs`)

The V2 UI (étape 3) doesn't pick models — the user checks **entities** (parcellaire, trous d'obus…). `src/app/services/model_orchestrator.py` resolves entities into `(model, target_rvt, selected_classes)` runs from two sources: `data/entities_catalog.json` (presentable vocabulary, versioned) + each installed model's `model_card.yaml` (coverage: a class covers entity `E` via its `entity:` alias, else by `name`). This is what **populates `computer_vision.runs`** — the array above is the underlying contract, auto-written by the UI. The orchestrator is **pure-Python and must never import `pipeline.cv`** (whose `__init__` pulls `shapely`); YAML reads are deferred.

**Derived targets.** A clustering output can also be surfaced as a first-class checkable entity (a *derived target*). A model declares it in `model_card.yaml` via a `derived_targets:` list mapping a clustering rule's `output_class` (from `args.yaml`) to a catalog `entity` (+ `include_source` to also output the individual source detections). `discover_installed_models` folds the derived entity into the model's `coverage` (classes = `output_class` [+ source classes]) and records it in `InstalledModel.derived_entities` — so `resolve_runs_from_entities` is unchanged and the clustering fires via the normal "output_class ∈ selected_classes" path. **Order invariant**: `_build_cluster_options` runs *before* `_merge_derived_targets`, so a derived entity never gets a redundant "Regrouper en clusters" toggle (the UI shows a "regroupement automatique" badge instead). This is how `zones_extraction_materiaux` ("Zones d'extraction de matériaux") is exposed on the crater models.

### Large rasters (`existing_mnt` / `existing_rvt` regime)

`_classify_mnt_layout` / `_classify_rvt_layout` (in `src/pipeline/modes/`) inspect raster bounds:
- **standard** (~1×1 km, IGN-aligned): legacy crop + IGN naming
- **small** (<1 km or unaligned): no crop, native extent preserved
- **large** (>1.05 km in any dim): no pre-tile-splitting at all — RVT computed on the full raster, SAHI handles 640×640 slicing in memory at inference. `Image.MAX_IMAGE_PIXELS = None` is set in `convert_tif_to_png.py` and `computer_vision_onnx.py`. Input MUST be EPSG:2154 (Lambert-93).

Don't reintroduce pre-tile-splitting for the large regime — it was removed deliberately to avoid NoData artifacts on sub-tile borders.

## Packaging conventions

`dev/package_plugin.py` produces `main.zip`. It excludes `dev/`, `tests/`, `.git/`, `.githooks/`, `__pycache__/`, virtualenvs, and named files (`config.json`, `pytest.ini`, `conftest.py`, `run_tests.py`, `.talismanrc`, `.gitignore`). It also strips `.pt`/`.pth` checkpoints (only `.onnx` ships). When adding new dev-only files, either put them under `dev/` or extend the exclusion lists in `package_plugin.py` — otherwise they end up in users' QGIS profile.

The `data/` directory is partially gitignored: `data/models/**`, `data/quadrillage_france/`, and the compiled CV runner binaries are NOT versioned (see `.gitignore`).

## Versionnage

Source de vérité : `metadata.txt` → `[general] version` (lu au runtime par `src/app/plugin_metadata.py`, affiché dans le titre du dialogue). Schéma : **SemVer**, mais on est en `0.x` → pas encore de stabilité publique garantie.

**Règles de bump (à proposer à l'utilisateur, jamais à appliquer silencieusement) :**

| Changement | Bump | Exemple |
|---|---|---|
| Correctif de bug, doc, lint, refactor interne sans impact utilisateur | **patch** `0.1.0 → 0.1.1` | `fix(ui): ...`, `fix(mnt): ...`, `chore: ...`, `docs: ...` |
| Nouvelle fonctionnalité, nouveau widget UI, nouveau mode, refactor visible (renommage UI, restructuration majeure) — rétro-compatible | **minor** `0.1.0 → 0.2.0` | `feat: ...`, refonte UI, nouvelle option de config |
| Breaking : format `config.json` incompatible, suppression d'un mode, API plugin cassée, ou premier release stable assumé | **major** `0.x.x → 1.0.0` | retrait de `existing_rvt`, changement de structure config |

**Quand proposer un bump :**

- À l'ouverture d'une PR vers `main`, ou quand l'utilisateur évoque « merge », « release », « publication », « PR », « tag », « livraison ».
- Quand l'utilisateur demande explicitement (« faut-il bumper ? »).
- **Ne jamais bumper sur un simple commit intra-branche** — la version marque une livraison utilisateur, pas un point dans l'historique git.

**Comment proposer :**

1. Lister les changements depuis la dernière version (`git log <last-tag>..HEAD --oneline` ou `git log main..HEAD --oneline` si pas de tag).
2. Classer le changement le plus impactant selon le tableau ci-dessus → c'est lui qui fixe le bump.
3. Proposer le nouveau numéro + une nouvelle entrée à ajouter **en tête** du champ `changelog=` de `metadata.txt` (le champ existe et est déjà rempli, vers la ligne 23).
4. Si l'utilisateur valide : éditer `metadata.txt`, puis suggérer `git tag v<version>` au moment du merge.

## Git hooks (Talisman)

A `pre-push` hook based on Talisman lives in `.githooks/` and must be enabled per-clone:

```bash
git config core.hooksPath .githooks
```

`.talismanrc` contains per-file checksums; if you legitimately modify a file Talisman flags, the checksum needs updating (recent commits `f3b5e05`, `a92bb51` are examples). Don't bypass with `--no-verify` without understanding the flag.
