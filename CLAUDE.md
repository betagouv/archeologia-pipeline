# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

QGIS plugin (Python) that runs a LiDAR processing pipeline → DTM / density / RVT indices, with an optional ONNX-based computer-vision detection step. README.md is the authoritative reference (~820 lines, French) — read the relevant section before doing anything non-trivial. This file captures only what isn't obvious from the file tree.

## Two execution contexts

Code in this repo runs in **one of two contexts**, and most surprises come from confusing them:

1. **Inside QGIS** (production): `__init__.py` → `main.py:ArcheologiaPipelinePlugin` is loaded by QGIS. `qgis.core`, `qgis.processing`, and `osgeo` are available. UI is the 4-step wizard `src/ui/wizard_dialog.py` (pages in `src/ui/steps/`, run view in `src/ui/run_view.py`). Pipeline modules under `src/pipeline/` import QGIS at module load time.
2. **Standalone** (tests / dev tooling): no QGIS available. `conftest.py` and `pytest.ini` deliberately exclude `src/ui/` and `src/pipeline/` from pytest collection (`norecursedirs`, `collect_ignore_glob`) because they would fail to import. Only modules under `src/app/` and pure helpers can be unit-tested directly. Don't add `from qgis.*` imports at module top level in code that needs to be testable — defer them inside functions, as `main.py:run()` already does.

## Qt5 / Qt6 compatibility (QGIS 3.34+ and 4.x)

The UI must run under **both Qt5 (QGIS 3.34–3.x) and Qt6 (QGIS 4.x)** from a single codebase (`metadata.txt`: `qgisMinimumVersion=3.34`, `qgisMaximumVersion=4.99`). QGIS 4.0 is the first Qt6 release; the original crash was flat enum access (`Qt.WindowMinimizeButtonHint`), which Qt6 removed. Two rules keep it dual-compatible:

- **Always scope enums** — the scoped form also works in PyQt5, so it's the *only* form to use: `Qt.AlignmentFlag.AlignCenter`, `Qt.WindowType.WindowMinimizeButtonHint`, `Qt.CursorShape.PointingHandCursor`, `Qt.PenStyle.NoPen`, `QFrame.Shape.HLine`, `QPainter.RenderHint.Antialiasing`, etc. (never `Qt.AlignCenter`). **This applies to every Qt class, not just the `Qt` namespace** — e.g. `QEvent.Type.Resize`, `QScrollArea.Shape.NoFrame`, `QAbstractSpinBox.ButtonSymbols.NoButtons`, `QTextCursor.MoveOperation.StartOfBlock`/`QTextCursor.MoveMode.KeepAnchor`. Same for QGIS class enums: `QgsWkbTypes.GeometryType.PolygonGeometry`, `QgsEditFormConfig.EditorLayout.TabLayout`, `QgsVectorFileWriter.WriterError.NoError`. Use `.exec()` not `.exec_()`. **Exception**: keep `.raise_()`/`.lower_()` with the trailing underscore (`raise` is a Python keyword, so PyQt6 retains it). Don't add `supportsQt6` to `metadata.txt` — removed in QGIS 4.
- Import Qt only via the `qgis.PyQt.*` shim (already the case everywhere), never `PyQt5`/`PyQt6` directly.

`src/ui/` isn't covered by pytest (standalone has no QGIS), so a flat-enum regression only surfaces at runtime in QGIS. **Verify by class, not by a fixed token list** (a closed list misses classes like `QEvent`/`QScrollArea` — that exact mistake shipped once). Sweep and confirm every hit is a scoped form (`QClass.EnumType.Value`) or a call (`(`):
- `rg "\bQ[A-Z]\w*\.[A-Z]\w+" src/` — all Qt widget/event classes
- `rg "\bQt\.[A-Z]\w+" src/` — the `Qt` namespace (separate: `Qt` is `Q`+lowercase)
- `rg "\bQgs\w*\.[A-Z]\w+|\bQgis\.[A-Z]\w+" src/` — QGIS classes

Any terminal `QClass.Value` (not followed by another `.Value`, not a `(...)` call) is a flat enum to scope.

## Common commands

```bash
# Tests — run from repo root, NOT pytest directly (run_tests.py sets sys.path)
python run_tests.py                  # all tests
python run_tests.py unit             # tests/unit only
python run_tests.py integration      # tests/integration only
python run_tests.py -k helpers       # filter by name (passes through to pytest)

ruff check src/                      # lint

# Dev tooling (all under dev/, excluded from packaged ZIP)
python dev/package_plugin.py                      # produces archeologia.<version>.zip for QGIS install
python dev/package_plugin.py --repo-url http://host/qgis/   # + generates matching plugins.xml
python dev/runner_onnx/build.py [--gpu|--clean]   # compile cv_runner_onnx executable
python dev/runner_onnx/export_to_onnx.py ...      # convert .pt/.pth → .onnx (see README Tâche 2)

# Dependency install (split by task, see dev/requirements/)
pip install -r dev/requirements/test.txt          # pytest + ruff
pip install -r dev/requirements/export.txt        # ultralytics/torch/onnx (model export only)
pip install -r dev/requirements/build.txt         # pyinstaller/onnxruntime (runner build only)
pip install -r dev/requirements.txt               # all of the above
```

QGIS-side manual test checklist: `tests/TESTS_MANUELS_QGIS.md`.

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

**`ign_laz` tile selection (UI → `dalles_urls.txt`).** Étape 1 offers three ways to designate IGN tiles: a polygon vector file (intersected by `tile_resolver.py`), a pre-made `.txt`, or **clicking tiles on the QGIS canvas** ("Sélectionner les dalles"). The map-pick path (`src/ui/map_tools/` — `tile_picker_tool.py` is the project's **first `QgsMapTool`**; `grid_layer.py` loads the grid layer) reads `nom_pkk`/`url_telech` from the selected grid features, formats them via the pure `app/services/tile_selection.py:format_dalles_urls`, and writes `data/temp_zones/dalles_selection.txt`. Because the downloader's `parse_ign_input_file` accepts `nom,url` lines and `IgnDownloadStrategy` treats a `.txt` as already-resolved (`is_vector` is False), this feeds the download with **zero pipeline changes**. The grid path (shapefile, or `.gpkg` if present) is resolved once by `pipeline/ign/quadrillage_paths.py:resolve_quadrillage_path` — the single source of truth shared by `tile_resolver` and the UI tool. The map tool's lifecycle (restore previous map tool, pop the message bar, remove the grid layer) is torn down via `SourcePage.cancel_dalles_selection_if_active()`, called from `WizardDialog.reject()`/`request_cancel_if_running()` and on readonly/mode change.

`RunContext` (frozen dataclass, `src/app/run_context.py`) is built from `config.json` by `build_run_context(config)`. The UI also writes `last_ui_config.json` for session persistence (gitignored).

### Computer vision: inference vs post-processing (critical distinction)

The CV pipeline is split across two execution boundaries:

- **Inference** is done by the external `cv_runner_onnx` subprocess (compiled binary in `data/third_party/cv_runner_onnx/<os>/`). It only emits raw JSON/TXT detections + optional annotated JPGs. If the runner is missing, `src/pipeline/cv/runner.py:_run_fallback_inference()` runs ONNX in-process via `computer_vision_onnx.py`.
- **Shapefile generation, polygon merging, overlap suppression, DBSCAN clustering, sub-threshold confidence filtering, and class filtering** all happen in the plugin's Python (under `src/pipeline/cv/`). The runner does **not** depend on `shapely`/`geopandas`/`fiona` — those are required only on the QGIS side. The consolidated `.qgs` project itself is written separately (see below), **not** under `src/pipeline/cv/`.

Per-model behavior (clustering, SAHI slicing, image size, task type) lives in each model's `args.yaml` (in `data/models/<name>/`). `selected_classes` filtering: `None` = all classes, `[]` = short-circuit (no inference), `[x, y]` = explicit filter. The empty-list short-circuit is in `run_cv_on_folder()` and runs **before** any inference.

Multiple CV models can run in one pipeline (`computer_vision.runs` array in config). The consolidated output is always a single `detections/detections_validation.qgs`, written by **`src/ui/qgs_writer.py:write_validation_project`** via the QGIS API (`QgsProject.write`) on the **main thread**, triggered from `run_view._on_load_layers` (same path as live layer loading). It is **not** written by `finalize_service` (worker thread, QGIS API not thread-safe) — `finalize_service` only emits the `load_layers` signal. Both the live load (`ui/layer_loader.py`) and the `.qgs` write share the layer factory `build_detection_vector_layer`.

**Index VRT lifecycle (re-run into the same `output_dir`).** Each `indices/<PRODUCT>/tif/` holds one mosaic VRT named `index_<PRODUCT>.vrt` (e.g. `index_MNT.vrt`, `index_CVAT.vrt`) — single source of truth `output_paths.index_vrt_filename(product)`, matching the QGIS layer name `index_<PRODUCT>` (so a file is identifiable when loaded manually). On a re-run into the same `output_dir`, the previous run's VRT layers are still loaded and QGIS would serialize its **stale in-memory VRT over the freshly regenerated file** (added tiles silently vanish). Guard: `ui/layer_loader.py:purge_output_dir_layers` removes the stale `index_<PRODUCT>.vrt` + detection GPKG layers **on the main thread at run launch** (`run_view.start_run`), *before* the worker regenerates them. The pure decision (which layers — by `indices/`/`detections/` subtree containment) lives in `src/app/services/layer_purge.py:select_layers_to_purge` (testable standalone); only the QGIS removal stays in `src/ui`. `_collect_vrt_paths_and_build` rebuilds under the distinctive name and deletes any legacy `index.vrt`.

**Confidence threshold = symbology = filtering (per-entity invariant).** The detection confidence threshold is set **per entity** (advanced settings → `computer_vision.runs[].confidence_threshold`), **not** the top-level `computer_vision.confidence_threshold` (a fallback only). Three things must use the *same* per-entity threshold or detections silently vanish: (1) `conf_bin` binning at conversion (`runner_shapefiles` → `create_shapefile_from_detections`); (2) **sub-threshold filtering** — `class_utils.filter_detections_below_confidence` drops `confidence < threshold` detections from the `.gpkg`, applied **after** clustering (so the DBSCAN `min_confidence_extend` hysteresis still absorbs sub-threshold points) and **exempting cluster output classes**; (3) the categorized `.qgs` symbology — `finalize_service.build_min_confidence_by_slug` maps each entity slug → its run threshold, fed through `run_view._on_load_layers` to `layer_loader`/`qgs_writer`, which resolve it **per layer** before `build_detection_vector_layer`. A QGIS categorized renderer matches by exact string with no "other values" bucket, so a legend bin (`[0.2:0.4[`) that doesn't match the data's `conf_bin` (`[0.3:0.4[`) renders **nothing**. Never reintroduce a single global threshold on the symbology side.

**Detection class colors** come from `src/pipeline/cv/class_color_registry.py` — stable **rank-based** colors (`color_palette.base_color_for_rank`, golden-ratio spread) persisted append-only in the QGIS profile as `class_color_registry.json`, the single source of truth used identically at generation and display. **Never assign colors by list index** (that caused two distinct entities both rendered green — a shipped bug).

### Index folder naming (`indices/<PRODUCT><param-suffix>/`)

Index output folders are named `<PRODUCT>` + a parameter suffix derived from the RVT settings — e.g. `SVF_R10_D16_V1_N0`, `LD_A15_Rmin10_Rmax20_H1p7_V1`, `HS_Az315_E35_V1`. The single source of truth is `get_rvt_folder_name(product, rvt_params)` in `src/pipeline/ign/products/rvt_naming.py` (= product code + `get_rvt_param_suffix(...)`). This lets re-running into the same `output_dir` with different params produce separate folders instead of overwriting. MNT/DENSITE/COUVERTURE have no params → bare `MNT`/`DENSITE`/`COUVERTURE`; CVAT has a fixed composition → bare `CVAT` (suffix `""`); `existing_rvt` mode forces `indices/RVT/` (params unknown).

**Visualization products are computed two ways.** Every index goes through `processing.run("rvt:...")` (provider from the third-party **rvt-qgis** plugin) — `HS`→`rvt_hillshade`, `M_HS`→`rvt_multi_hillshade`, `SVF`→`rvt_svf`, `SLO`→`rvt_slope`, `LD`→`rvt_ld` (file is `rvt_local_dom.py` but `name()`=`rvt_ld`), `SLRM`→`rvt_slrm`, `VAT`→`rvt_blender` (BLEND_COMBINATION=0), `MSTP`→`rvt_mstp` — **except `CVAT`**. rvt-qgis exposes CVAT only through its GUI dialog, not its Processing provider (the `rvt_blender` `BLEND_COMBINATION` enum lists only VAT/Prismatic/City; the CVAT branch in `rvt_blender.py` is unreachable). So `CVAT` is computed **in-process** in `src/pipeline/ign/products/cvat.py:compute_cvat` — it imports the bundled `rvt` package (located among sibling plugins via `_find_rvt_dir`, added to `sys.path`), reproduces rvt-qgis's CVAT recipe (blend VAT-general + VAT-flat at 50/100), and runs on the worker thread (numpy/gdal only, no Qt). If rvt-qgis is absent, CVAT is logged and skipped, not fatal. Don't try to route CVAT through `processing.run` — it won't resolve.

**Invariant — same `rvt_params` on both sides.** `get_rvt_param_suffix` applies *defaults* for missing keys (empty dict → the default suffix, **not** `""`). Creation (`results.py:copy_final_products_to_results`) and CV consumption (`output_paths.resolve_rvt_tif_dir` → fed by `cv_post_service`) must therefore receive the *same* `rvt_params`, or they resolve to *different* folders. `run_existing_rvt` mirrors this: when `indices_folder_name` is None it re-derives the suffixed name via `get_rvt_folder_name`. `output_paths.py` imports `rvt_naming` **deferred** (top-level import would pull `ign/products/__init__` → QGIS, breaking standalone/test imports).

### Entity orchestration (UI → `computer_vision.runs`)

The V2 UI (étape 3) doesn't pick models — the user checks **entities** (parcellaire, cratères…). `src/app/services/model_orchestrator.py` resolves entities into `(model, target_rvt, selected_classes)` runs from two sources: `data/entities_catalog.json` (presentable vocabulary, versioned) + each installed model's `model_card.yaml` (coverage: a class covers entity `E` via its `entity:` alias, else by `name`). This is what **populates `computer_vision.runs`** — the array above is the underlying contract, auto-written by the UI. The orchestrator is **pure-Python and must never import `pipeline.cv`** (whose `__init__` pulls `shapely`); YAML reads are deferred.

**Derived targets.** A clustering output can also be surfaced as a first-class checkable entity (a *derived target*). A model declares it in `model_card.yaml` via a `derived_targets:` list mapping a clustering rule's `output_class` (from `args.yaml`) to a catalog `entity` (+ `include_source` to also output the individual source detections). `discover_installed_models` folds the derived entity into the model's `coverage` (classes = `output_class` [+ source classes]) and records it in `InstalledModel.derived_entities` — so `resolve_runs_from_entities` is unchanged and the clustering fires via the normal "output_class ∈ selected_classes" path. **Order invariant**: `_build_cluster_options` runs *before* `_merge_derived_targets`, so a derived entity never gets a redundant "Regrouper en clusters" toggle (the UI shows a "regroupement automatique" badge instead). This is how `regroupement_crateres` ("Regroupement de cratères") is exposed on the crater models. ⚠️ The `entity:` value must exist in `data/entities_catalog.json` — an unknown id is silently ignored by the orchestrator.

### Large rasters (`existing_mnt` / `existing_rvt` regime)

`_classify_mnt_layout` / `_classify_rvt_layout` (in `src/pipeline/modes/`) inspect raster bounds:
- **standard** (~1×1 km, IGN-aligned): legacy crop + IGN naming
- **small** (<1 km or unaligned): no crop, native extent preserved
- **large** (>1.05 km in any dim): no pre-tile-splitting at all — RVT computed on the full raster, SAHI handles 640×640 slicing in memory at inference. `Image.MAX_IMAGE_PIXELS = None` is set in `convert_tif_to_png.py` and `computer_vision_onnx.py`. Input MUST be EPSG:2154 (Lambert-93).

Don't reintroduce pre-tile-splitting for the large regime — it was removed deliberately to avoid NoData artifacts on sub-tile borders.

## Packaging conventions

`dev/package_plugin.py` produces `archeologia.<version>.zip` (e.g. `archeologia.0.7.0.zip`) — `zip_filename()` = `<PLUGIN_NAME>.<version>` from `metadata.txt`'s `version`. **The prefix before the first dot MUST equal the ZIP's internal root folder (`archeologia`)**: on a *repository* install QGIS derives the install-dir name from the zip filename split on the first `.`, so a mismatch fails with "répertoire mal nommé" (the old `ArcheologIA_v<version>.zip` → expected dir `ArcheologIA_v0` ≠ `archeologia`; `Install from ZIP`, which reads `metadata.txt` instead, still worked — masking the bug). Pass `--repo-url <base>` to also emit a matching `plugins.xml` (`file_name`/`download_url` stay in sync with the zip; the repo URL is a CLI arg, **never committed** — it points at the private OVH repo). Defaults for `--repo-url`/`--output-dir` are read from the gitignored `dev/docs/_local/deploy.config.json` (template `dev/deploy.config.example.json`), so a bare `python dev/package_plugin.py` emits both the zip and `plugins.xml` into `_local/depot` **without hardcoding the private URL** in a committed file. The root folder `archeologia` is the QGIS plugin identity, never versioned. It excludes `dev/`, `tests/`, `.git/`, `.githooks/`, `__pycache__/`, virtualenvs, and named files (`config.json`, `pytest.ini`, `conftest.py`, `run_tests.py`, `.talismanrc`, `.gitignore`). It also strips `.pt`/`.pth` checkpoints (only `.onnx` ships). When adding new dev-only files, either put them under `dev/` or extend the exclusion lists in `package_plugin.py` — otherwise they end up in users' QGIS profile.

The `data/` directory is partially gitignored: `data/models/**`, `data/quadrillage_france/`, and the compiled CV runner binaries are NOT versioned (see `.gitignore`). The quadrillage ships **inside** the ZIP (test PKG-02) as the IGN shapefile **plus its `.qix` spatial index** — the index is what makes the on-canvas tile selection (and `tile_resolver`) responsive on ~490 k features; regenerate it with `python dev/build_quadrillage_index.py` whenever the IGN ships a new grid. (A GeoPackage was evaluated and rejected: measured *larger* than `.shp`+`.qix`, with no row-drop benefit since every tile has a URL.)

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
4. Si l'utilisateur valide : éditer **tous les fichiers qui portent la version en dur** — `metadata.txt` (`version=` **et** nouvelle entrée en tête de `changelog=`) **et** `README.md` (ligne ~6 `- Version : **X.Y.Z**`) — puis suggérer `git tag v<version>` au moment du merge. Le reste dérive de `metadata.txt` au runtime (`app/plugin_metadata.py`, le titre du dialogue, et `dev/package_plugin.py:zip_filename` qui nomme le ZIP `ArcheologIA_v<version>.zip`) → rien d'autre à éditer. ⚠️ Ne pas toucher aux références **historiques** (historique du `changelog=`, notes sous `dev/docs/` qui citent une version passée). Vérifier par `git grep <ancienne X.Y.Z>` qu'il ne reste pas d'occurrence active.

## Git hooks (Talisman)

A `pre-push` hook based on Talisman lives in `.githooks/` and must be enabled per-clone:

```bash
git config core.hooksPath .githooks
```

`.talismanrc` contains per-file checksums; if you legitimately modify a file Talisman flags, the checksum needs updating (recent commits `f3b5e05`, `a92bb51` are examples). Don't bypass with `--no-verify` without understanding the flag.
