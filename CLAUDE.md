# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

QGIS plugin (Python) that runs a LiDAR processing pipeline → DTM / density / RVT indices, with an optional ONNX-based computer-vision detection step. README.md is the authoritative reference (~820 lines, French) — read the relevant section before doing anything non-trivial. This file captures only what isn't obvious from the file tree.

## Trois dépôts — où travailler (règle 2026-09-28)

| Sujet | Dépôt |
|---|---|
| Plugin QGIS : UI, pipeline local, CV, onglet ② (lecture du catalogue) | **ici** (public) |
| Calcul distant + **publication Géoplateforme** : livraisons, pyramides, grille et format des tuiles, offres, clé, écriture du catalogue de l'onglet ② | `C:/projets/Archeologia/archeologia-ovh` (**privé**) : son `CLAUDE.md`, son runbook `.claude/commands/publish-gpf.md`, son `.venv`, ses `tests/` |
| Entraînement, évaluation, corpus des modèles | `C:/projets/Archeologia/training-models` (cf. « Installation d'un modèle » plus bas) |
| Carte des dépôts, skills communs aux dépôts | marketplace locale `C:/projets/Archeologia/claude-plugins` (plugin Claude Code `archeologia`, enregistré par `.claude/settings.json`) : `/archeologia:depots` à invoquer dès qu'une demande touche un autre dépôt |

Un défaut **vu dans le flux** (qualité des tuiles, grille, 401/403) se diagnostique et se corrige dans `archeologia-ovh`, jamais ici : le plugin ne fait qu'ouvrir les descripteurs `GDAL_WMS` que cet outil écrit dans `data/local_catalogue/` (gitignoré, porte la clé d'accès). Session ouverte ici pour un tel sujet : le dire, lire le `CLAUDE.md` et les `.claude/commands/` de l'autre dépôt **avant** d'agir, y faire les modifications et y lancer ses tests — sans jamais recopier ici son code, ses secrets ni ses documents (dépôt public).

## Workflows et outillage — par où commencer

| Besoin | Point d'entrée |
|---|---|
| Livrer une version | `/livraison` (`.claude/commands/livraison.md`, versionné) |
| Savoir ce qu'il reste à vérifier dans QGIS | `/recette` (`.claude/commands/recette.md`) |
| Une fonctionnalité visible change | skill `manuel-integre` : le chapitre du manuel dans le même commit |
| Installer un modèle, écrire une fiche de classe | `/archeologia:installer-modele-plugin` puis `/archeologia:fiche-classe-plugin` (skills partagés, § Installation d'un modèle) |
| Un sujet d'un autre dépôt, un fichier produit ici et lu ailleurs | `/archeologia:depots` |
| Vérifier une fenêtre sans QGIS de bureau | `C:/OSGeo4W/bin/python-qgis.bat dev/rendu_hors_ecran.py captures\|manuel\|profil\|journal` (§ Branches, worktrees) |
| Situer les zones d'une fiche nouvelle sur la carte « Appris sur » | `python-qgis.bat dev/fiches/zones_corpus.py` → `data/zones_corpus.json` (`test_carte_zones` rouge sinon) |
| Grille IGN à jour (avant chaque livraison) | `python dev/build_quadrillage_from_wfs.py` |
| Binaire CV périmé (`test_binaire_a_jour` rouge) | `python dev/runner_onnx/build.py` |
| Règles stables de données et de rendu | `docs/contrats.md` (fiabilité, fiches, OPNS, manuel, profil des scores, lots 0.15) |

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
C:/OSGeo4W/bin/python-qgis.bat dev/rendu_hors_ecran.py captures|manuel [--tous]|profil|journal   # UI rendue hors écran (cf. § Branches, worktrees)

# Dependency install (split by task, see dev/requirements/)
pip install -r dev/requirements/test.txt          # pytest + ruff
pip install -r dev/requirements/export.txt        # ultralytics/torch/onnx (model export only)
pip install -r dev/requirements/build.txt         # pyinstaller/onnxruntime (runner build only)
pip install -r dev/requirements.txt               # all of the above
```

QGIS-side manual test checklist: `tests/TESTS_MANUELS_QGIS.md` (`/recette` liste ce qui reste à jouer).
⚠ Le harnais Docker de `D:\tests_fonctionnement_plugin` (17 cas, 4 modes) date du 2026-08-05 et attend un
`metadata.json` à la racine du dossier de sortie : l'arborescence v3 (`livrable/` + `technique/`, 2026-10-08) le
rend obsolète tant qu'il n'est pas adapté — ne pas s'y fier sans l'avoir mis à jour.

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

**Vue d'exécution V2 (2026-10-08, validée par l'utilisateur : « la frise porte l'état, le journal raconte »).** `ui/run_view.py` n'a plus ni carte d'étape (`RunHeader`), ni bandeau de fin (`RunEndBanner`), ni `QProgressBar` : une ligne d'état au-dessus de la frise (`_fil_gauche` = résumé des choix via `_resume_config`, puis la synthèse de fin posée par `_show_end_banner` avec `kind` ok/cancel/err ; `_fil_droite` = « prêt à lancer » / « en cours · chrono total » / vide), la frise dont le fil qui suit la pastille active se remplit avec le compteur mesuré de `metric` (`_StepLine.set_ratio`, la seule progression affichée — rien d'estimé, `progress` global n'est plus montré), puis à la fin le cadre « Par où commencer » (`_bilan_box` : barres du bilan, « Ouvrir le projet QGIS », « Ouvrir le dossier », « Rapport », légende ; en bas il ne reste que « Log complet » et « Annuler ») et le journal (placeholder avant le lancement). **Plus de chargement automatique des couches** : `_on_load_layers` garde les arguments dans `_couches_a_charger` et écrit le `.qgs` (projet dédié) ; `load_result_layers` n'est appelé que par `_ouvrir_projet_qgis` (clic, une fois). Un seul format de durée (`_fmt_hms` délègue à `user_narrator._format_duration`). Garde-fous : `tests/unit/test_vue_execution.py`. Recette §44 ; §40.2 et §41.1 révisés.

**Arborescence v3 du dossier de sortie (2026-10-08).** Deux racines, une règle pour l'utilisateur : `livrable/` (`projet.qgs`, `rapport.html`, `traitement.json`, `indices/<PRODUIT>/tif/`, `detections/<slug>/`) se garde et se transmet ; `technique/` (`sources/`, `intermediaires/`, `png/<PRODUIT>/`, `detection/<modèle>/`, `journaux/`) se supprime. Source unique des chemins : `src/pipeline/output_paths.py` (`livrable_dir`, `technique_dir`, `projet_qgs_path`, `traitement_json_path`, `journaux_dir`, `dalles_urls_path`, `indice_png_dir` → `technique/png/<P>`, `detection_technique_dir` → `technique/detection/<m>`) ; `layer_purge` purge `livrable/` et les racines v2. La finalisation écrit la trace **complète** (avec `ui_config` et ses chemins de poste) dans `technique/journaux/metadata_<ts>.json` à côté du `pipeline_log_<ts>.txt` du même lancement, et `livrable/traitement.json` **sans** `ui_config`, chemins relatifs, `plugin_version` réelle et `arborescence: 3` ; `meta["rapport"]` vaut `livrable/rapport.html`. Un dossier v2 est **migré sur proposition, jamais en silence** : `app/services/arborescence.py` (pur : `etat`, `plan_migration` → `decrire` → `appliquer` par `os.replace`, `non_reconnus`), `run_view.start_run` pose le `QMessageBox`, purge les couches, applique, et refuse le lancement si un déplacement échoue ; ce qui n'est pas reconnu (`MNT/`, `RVT/` de la v1, fichiers de l'utilisateur, destination déjà prise) ne bouge jamais ; un ancien `detections_validation.qgs` suit son dossier `detections/` et reste valide (chemins relatifs). `validate_run_context` refuse un dossier de sortie **dans** le dossier source et signale l'inverse. `fiabilite_observee.agreger` lit les deux arborescences. Recette §43.

### Computer vision: inference vs post-processing (critical distinction)

The CV pipeline is split across two execution boundaries:

- **Inference** is done by the external `cv_runner_onnx` subprocess (compiled binary in `data/third_party/cv_runner_onnx/<os>/`). It only emits raw JSON/TXT detections + optional annotated JPGs. If the runner is missing, `src/pipeline/cv/runner.py:_run_fallback_inference()` runs ONNX in-process via `computer_vision_onnx.py`.
- **Shapefile generation, polygon merging, overlap suppression, DBSCAN clustering, sub-threshold confidence filtering, and class filtering** all happen in the plugin's Python (under `src/pipeline/cv/`). The runner does **not** depend on `shapely`/`geopandas`/`fiona` — those are required only on the QGIS side. The consolidated `.qgs` project itself is written separately (see below), **not** under `src/pipeline/cv/`.

Per-model behavior (clustering, SAHI slicing, image size, task type) lives in each model's `args.yaml` (in `data/models/<name>/`). `selected_classes` filtering: `None` = all classes, `[]` = short-circuit (no inference), `[x, y]` = explicit filter. The empty-list short-circuit is in `run_cv_on_folder()` and runs **before** any inference.

**Halo inter-dalles (ign_laz/local_laz).** Inference PNGs come from the **uncropped** RVT TIFs of `technique/intermediaires/` (tile + `tile_overlap` margin, resolved by `cv_post_service.resolve_uncropped_tif`), so objects straddling tile borders are seen whole; cross-tile duplicates merge downstream in geo space. Three things must stay referenced on the **same** source raster (never mix cropped/uncropped): the PNG content, the GEO-03 guard (`_png_consistent_with_tif`), and `tif_transform_data`/world files — the PNG keeps the *cropped* TIF's **name** (stable stems). Detections are clipped to the union of the run's cropped-TIF extents (`valid_region_bounds` → `clip_detections_to_valid_region`) to kill noise from the fabricated outer margin, and `raw_detections/` caches older than their PNG are auto-purged (`purge_stale_cached_detections`). `existing_mnt`/`existing_rvt` have no `technique/intermediaires/`: `run_existing_rvt` fabricates the halo itself from the run's neighbouring tiles (`modes/neighbor_halo.py:NeighborHalo`, GDAL VRT of the tiles → `technique/intermediaires/halo/<indice>/`, default 50 m = zero extra SAHI slice at 648/672 px; freshness = `.inputs.json` neighbour set + mtimes, same idea as the merged-LAZ sidecar) as a **fallback after** the explicit resolver. So every mode ends up with `valid_region_bounds` and the clip; a single tile or a "large" raster gets no halo. **Centroid rule** (`postprocessing.owned_by_cell`, applied while parsing labels in `conversion_shp` via `cell_bounds_by_stem` = cropped cell per PNG stem, built by `run_existing_rvt` for every halo source, real or fabricated): a tile only reports detections whose centroid lies in its own cell — kills cross-tile duplicates at the source and the fragments cut at the halo edge (the neighbour sees them whole). The `relation`/`keep_best` dedup stays for within-tile overlaps.

Multiple CV models can run in one pipeline (`computer_vision.runs` array in config). The consolidated output is always a single `livrable/projet.qgs`, written by **`src/ui/qgs_writer.py:write_validation_project`** via the QGIS API (`QgsProject.write`) on the **main thread**, triggered from `run_view._on_load_layers` (same path as live layer loading). It is **not** written by `finalize_service` (worker thread, QGIS API not thread-safe) — `finalize_service` only emits the `load_layers` signal. Both the live load (`ui/layer_loader.py`) and the `.qgs` write share the layer factory `build_detection_vector_layer`.

**Index VRT lifecycle (re-run into the same `output_dir`).** Each `livrable/indices/<PRODUCT>/tif/` holds one mosaic VRT named `index_<PRODUCT>.vrt` (e.g. `index_MNT.vrt`, `index_CVAT.vrt`) — single source of truth `output_paths.index_vrt_filename(product)`, matching the QGIS layer name `index_<PRODUCT>` (so a file is identifiable when loaded manually). On a re-run into the same `output_dir`, the previous run's VRT layers are still loaded and QGIS would serialize its **stale in-memory VRT over the freshly regenerated file** (added tiles silently vanish). Guard: `ui/layer_loader.py:purge_output_dir_layers` removes the stale `index_<PRODUCT>.vrt` + detection GPKG layers **on the main thread at run launch** (`run_view.start_run`), *before* the worker regenerates them. The pure decision (which layers — by `livrable/` subtree containment, plus the v2 `livrable/indices/`/`livrable/detections/` roots for layers still loaded after a migration) lives in `src/app/services/layer_purge.py:select_layers_to_purge` (testable standalone); only the QGIS removal stays in `src/ui`. `_collect_vrt_paths_and_build` rebuilds under the distinctive name and deletes any legacy `index.vrt`.

**Confidence threshold = symbology = filtering (per-entity invariant).** The detection confidence threshold is set **per entity** (advanced settings → `computer_vision.runs[].confidence_threshold`), **not** the top-level `computer_vision.confidence_threshold` (a fallback only). Three things must use the *same* per-entity threshold or detections silently vanish: (1) `conf_bin` binning at conversion (`runner_shapefiles` → `create_shapefile_from_detections`); (2) **sub-threshold filtering** — `class_utils.filter_detections_below_confidence` drops `confidence < threshold` detections from the `.gpkg`, applied **after** clustering (so the DBSCAN `min_confidence_extend` hysteresis still absorbs sub-threshold points) and **exempting cluster output classes**; (3) the categorized `.qgs` symbology — `finalize_service.build_min_confidence_by_slug` maps each entity slug → its run threshold, fed through `run_view._on_load_layers` to `layer_loader`/`qgs_writer`, which resolve it **per layer** before `build_detection_vector_layer`. A QGIS categorized renderer matches by exact string with no "other values" bucket, so a legend bin (`[0.2:0.4[`) that doesn't match the data's `conf_bin` (`[0.3:0.4[`) renders **nothing**. Never reintroduce a single global threshold on the symbology side.

**Default threshold of a model (`model_card.thresholds.confidence_default` / `confidence_per_class`) is chosen BELOW the F1-max, by studying the curve — never "= F1-max" (rule 2026-09-09).** F1 weighs a miss like a false positive; in prospection a miss is never recovered while a false positive is dismissed in seconds on the RVT. Each model's `entrainement/evaluation/metriques_eval.json` (produced by `tools/courbes_eval.py` in the training-models repo) carries, per class and globally, a `seuil_f1max` **and** an `etude_seuil` block (F2-max, F1 plateaus ≥ 98 %/95 %, R_max, FP per image, marginal precision of the added detections, table at 0.05 steps, `seuil_propose` = max(F2-max, bottom of the 95 % plateau) — a *starting point only*). The deployed value is picked **inside [bottom of the 95 % plateau ; F1-max]** by reading that table each time: narrow plateau or heavy FP/image (linéaires) → stay near F1-max; wide plateau and low FP/image (fours, dépressions, enclos IE) → go down to F2-max; rare/unusable class → don't lower; check `par_zone_classe` (a lower threshold returns recall to weak zones). The reasoning goes into `thresholds.seuils_provenance` (**required** whenever the eval carries `etude_seuil`); `scripts/validate_models_metadata.py` errors without it and warns when the value is outside the window or sits at the F1-max. Contract: `docs/model_contract.md`. An older eval without the block is completed offline by `completer_metriques_eval.py` (training-models). Models with no canonical eval (cratère, verdun, formes_lineaires, at a hand-set 0.20) are outside this rule until they get one.

**Contrats des fiches, de la fiabilité et de l'OPNS → `docs/contrats.md`** (déplacés le 2026-10-07, repris tels quels). Les règles à ne pas défaire, en une ligne chacune : la légende de fiabilité est **contour seul, sans remplissage**, couleur de la classe déclinée par `STYLE_SPEC` + `apply_confidence`, les mots douteux/possible/probable/très probable valent la même part de vrais objets pour tous les modèles et les classes linéaires sont calibrées sur le critère de couverture ; une fiche de classe vit dans `classes[].fiche` du `model_card.yaml` (chemins relatifs au dossier du modèle, tout optionnel mais exploitable dès que présent) et l'humanisation RVT/tâche reste dans `app/services/vocabulaire_modele.py`, jamais dans `ui/` ; une fiche de produit vit dans `data/indices_fiches.json` **sans** porter l'identité du produit, et ses deux tableaux comparatifs sont ceux de la documentation RVT, traduits, jamais réévalués ; toute citation anglophone est traduite et une composition nomme ses couches par sigle ; l'OPNS est **un** produit avec un réglage de type, bornes dures 10–50 px / 8–64 directions / bruit 0–3 lues par `rvt_naming.opns_settings` des deux côtés.

**Réinitialisation ciblée (2026-09-16).** Les deux boutons **globaux** sont supprimés : « ↺ Réinit. val. par défaut » (étape 2) remettait tous les produits d'un coup, « ↺ Réinit. val. défaut du modèle » (étape 3) effaçait les surcharges de toutes les entités. La portée est désormais **un produit** (un bouton par onglet de réglages avancés, plus un sur la carte Tuilage) ou **une entité** (un « ↺ » sur la ligne avancée de chaque `EntityCard`, signal `reset_requested`). Logique de portée dans le module **pur** `src/app/services/reglages_defaut.py`. ⚠ `step_2_indices._adv_fields` porte maintenant `(produit, section, key, widget, kind, default)` — **six** éléments : tout dépaquetage oublié lève à l'usage. ⚠ Le produit se déduit de la section pour les indices RVT (`SECTION_PRODUIT` : celle du LD est `ldo`, pas `ld`) mais **doit être nommé** pour `("processing",)`, que MNT, Densité, Couverture et le tuilage se partagent — sans quoi les trois onglets de base se réinitialiseraient ensemble. `tests/unit/test_reglages_defaut.py` teste la logique, `tests/unit/test_reset_defauts_ui.py` vérifie par AST que chaque produit du pipeline a au moins un champ (sinon son bouton ne ferait rien) et que les boutons globaux ne reviennent pas. Recette §34.

**Vignettes et densité d'écran (2026-09-16).** Toute vignette affichée passe par `ui/widgets/vignette.py:pixmap_ajuste(source, cote, dpr=widget.devicePixelRatioF(), cadrage=...)`, qui rastérise à `cote × dpr` pixels **physiques** et pose le ratio sur le pixmap — même discipline que `ui/icons.py:colored_pixmap`. ⚠ Un `QPixmap(chemin)` porte un ratio de 1 : sur un écran à 125 % ou 150 % il n'occupe que `cote / dpr` px logiques dans un cadre de `cote` px, et le fond clair du cadre apparaît en **liseré sur les côtés** (constat utilisateur, sur les vignettes de classes comme de produits) — en plus d'être flou. Les quatre points d'affichage passent par ce helper : `FicheThumb` (étape 2), `EntityCard` (étape 3) et les deux aperçus de fiche. Viser la taille du **contenu** et non celle du widget quand le QSS pose une bordure (`_VIGNETTE_MAX - 2`).

**Manuel intégré (`aide/`, 2026-10-07).** Un manuel livré dans le ZIP (`aide/<nn>-<cle>.md`, un chapitre par écran, Nouveautés rendu depuis `changelog=`), module pur `app/services/aide.py`, fenêtre `ui/dialogs/aide_dialog.py`, journal de l'étape 4 en `QTextBrowser` avec renvoi cliquable. **Règle : toute fonctionnalité visible modifie son chapitre dans le même commit** (skill `manuel-integre`) ; jamais d'estimation de durée, de version écrite à la main, de citation anglaise, ni d'image dans un élément de liste ; contrôle de chaque livraison par `dev/rendu_hors_ecran.py manuel --tous` et `journal`. Détail des mécanismes (ancres, recherche, historique, zoom, « quoi de neuf », images) : `docs/contrats.md` § Manuel intégré. Recette §38.

**Profil des scores (2026-10-07).** La figure derrière les quatre niveaux de fiabilité : module pur `app/services/profil_scores.py` (bandes de `metriques_eval.json`, même règle de provenance que le validateur, agrégées par 0,05 **en coupant aux seuils de la classe**), widget `ui/widgets/profil_scores.py`, posée dans la fiche de classe et la fiche ⓘ. Fiche de classe et fiche de modèle se distinguent (liseré, étiquette de nature) ; bandeau « Appris sur » (`data/zones_corpus.json` écrit par `dev/fiches/zones_corpus.py`, **à relancer après un modèle dont une fiche cite une zone nouvelle**) ; vérité terrain recolorée **à l'affichage** dans la couleur de la couche, jamais dans les fichiers. Règles complètes (échelle, éclats, libellés, couleurs, cartes) : `docs/contrats.md` § Profil des scores. Recettes §37.7 bis/ter/quater, §39.

**Pistes 0.15 livrées (2026-10-08, `feat/aide-integree`) : A1–A11, N1, N2, N3, N5 — un lot = un commit.** Les règles à ne pas défaire sont dans `docs/contrats.md` § Pistes 0.15 : seuil mobile du profil et bilan **par rapport au seuil du modèle** (jamais en part d'un total), point d'équilibre lu dans l'évaluation de référence, couleur = clé du registre = nom de couche, seuil appliqué en orange, une rangée par sorte de texte, « Tester un seuil » sans toucher au traitement ; N2 : `effectifs` dans le sidecar et bilan lu sur les seuls sidecars ; N5 : rapport de traitement pur qui **ne situe jamais la zone** ; N3 : verdicts fixes `oui` / `non` / `peut-être`, registre `runs_connus.json`, GeoPackages lus en sqlite3 ; A8 : fraîcheur de la grille lue dans le `.dbf`. ⚠ Toute modification sous `src/pipeline/cv` rend rouge `test_binaire_a_jour` : recompiler le binaire avant la livraison (runbook étape 5).

**Detection class colors** come from `src/pipeline/cv/class_color_registry.py` — stable **rank-based** colors (`color_palette.base_color_for_rank`, golden-ratio spread) persisted append-only in the QGIS profile as `class_color_registry.json`, the single source of truth used identically at generation and display. **Never assign colors by list index** (that caused two distinct entities both rendered green — a shipped bug).

**Coût de calcul d'un modèle (étape 3, 2026-09-21) — jamais d'estimation de durée dans l'UI.** Avant un run, l'étape 3 ne donne qu'un **fait** : le nombre de fenêtres SAHI que le modèle analyse par dalle (module **pur** `src/app/services/cout_modele.py` : `fenetres_par_dalle(slice, overlap, cote)` délègue à `pipeline.cv.sahi_lite.get_slice_bboxes` en import différé — la fonction même du binaire ; une réplique se tromperait de 44 % à 672 px sur 2 800 px à cause de la déduplication des fenêtres de bord). `InstalledModel.sahi_slice_px`/`sahi_overlap` sont lus d'`args.yaml` par l'orchestrateur (0 = inconnu → rien affiché, pas de défaut 640 dupliqué). Affiché **là où le choix se fait**, pas sur la carte (déjà saturée) : entrées du menu « Changer ▾ » (`libelle_menu`), infobulle du nom de modèle (`infobulle`, une phrase par modèle en A/B), Row « Fenêtres d'analyse » du dialogue ⓘ (`ligne_dialogue`) — **toujours aux deux tailles** 2 000 px (dalle 1 km sans marge) et 2 800 px (marge 20 % des modes LiDAR), parce que l'ordre 648 vs 672 px bascule entre elles (16/16 contre 36/25) ; jamais de ratio « ×6 » ni de pictogramme (le rapport des fenêtres n'est pas le rapport des temps). Après le run, la **durée réelle** de chaque modèle est mesurée côté plugin (`external_runner.run_external_cv_runner(stats=…)` : `images_inferees` = lignes `status=done`, cache exclu ; `secondes` ; nouveau callback `image_done` de `_parse_runner_stdout`), remontée par `run_cv_on_folder(stats=)` → `ExistingRvtResult.cv_stats` → `UserNarrator.cv_run_done` dans les deux boucles de runs (`cv_post_service`, `existing_rvt_runner`) : « ✓ « Modèle X » : 12 images analysées en 4min 07s (≈ 21s par image) », silencieuse si 0 image inférée. Depuis le 2026-10-08 ces durées alimentent aussi le rapport de traitement (`cv_stats` → `rapport.html`). Mesuré le 2026-09-21 sur 75 journaux : ≈ 0,8–1,2 s **par fenêtre** pour tous les modèles (Ryzen 5 sans GPU), temps par image linéaire au nombre de fenêtres. Un éventuel lot 2 (mesure locale en fourchette, registre profil à fenêtre glissante, jamais invalidé sur la date du binaire) et les options rejetées (banc + bloc `model_card`, clé par architecture) sont sur la page de propositions (mémoire `project-temps-par-modele-propositions`).

### Index folder naming (`livrable/indices/<PRODUCT><param-suffix>/`)

Index output folders are named `<PRODUCT>` + a parameter suffix derived from the RVT settings — e.g. `SVF_R10_D16_V1_N0`, `LD_A15_Rmin10_Rmax20_H1p7_V1`, `HS_Az315_E35_V1`. The single source of truth is `get_rvt_folder_name(product, rvt_params)` in `src/pipeline/ign/products/rvt_naming.py` (= product code + `get_rvt_param_suffix(...)`). This lets re-running into the same `output_dir` with different params produce separate folders instead of overwriting. MNT/DENSITE/COUVERTURE have no params → bare `MNT`/`DENSITE`/`COUVERTURE`; CVAT has a fixed composition → bare `CVAT` (suffix `""`); `existing_rvt` mode forces `livrable/indices/RVT/` (params unknown).

**Visualization products are computed two ways.** Every index goes through `processing.run("rvt:...")` (provider from the third-party **rvt-qgis** plugin) — `HS`→`rvt_hillshade`, `M_HS`→`rvt_multi_hillshade`, `SVF`→`rvt_svf`, `SLO`→`rvt_slope`, `LD`→`rvt_ld` (file is `rvt_local_dom.py` but `name()`=`rvt_ld`), `SLRM`→`rvt_slrm`, `VAT`→`rvt_blender` (BLEND_COMBINATION=0), `MSTP`→`rvt_mstp` — **except `CVAT`**. rvt-qgis exposes CVAT only through its GUI dialog, not its Processing provider (the `rvt_blender` `BLEND_COMBINATION` enum lists only VAT/Prismatic/City; the CVAT branch in `rvt_blender.py` is unreachable). So `CVAT` is computed **in-process** in `src/pipeline/ign/products/cvat.py:compute_cvat` — it imports the bundled `rvt` package (located among sibling plugins via `_find_rvt_dir`, added to `sys.path`), reproduces rvt-qgis's CVAT recipe (blend VAT-general + VAT-flat at 50/100), and runs on the worker thread (numpy/gdal only, no Qt). If rvt-qgis is absent, CVAT is logged and skipped, not fatal. Don't try to route CVAT through `processing.run` — it won't resolve.

**Invariant — same `rvt_params` on both sides.** `get_rvt_param_suffix` applies *defaults* for missing keys (empty dict → the default suffix, **not** `""`). Creation (`results.py:copy_final_products_to_results`) and CV consumption (`output_paths.resolve_rvt_tif_dir` → fed by `cv_post_service`) must therefore receive the *same* `rvt_params`, or they resolve to *different* folders. `run_existing_rvt` mirrors this: when `indices_folder_name` is None it re-derives the suffixed name via `get_rvt_folder_name`. `output_paths.py` imports `rvt_naming` **deferred** (top-level import would pull `ign/products/__init__` → QGIS, breaking standalone/test imports).

### Entity orchestration (UI → `computer_vision.runs`)

The V2 UI (étape 3) doesn't pick models — the user checks **entities** (parcellaire, cratères…). `src/app/services/model_orchestrator.py` resolves entities into `(model, target_rvt, selected_classes)` runs from two sources: `data/entities_catalog.json` (presentable vocabulary, versioned) + each installed model's `model_card.yaml` (coverage: a class covers entity `E` via its `entity:` alias, else by `name`). This is what **populates `computer_vision.runs`** — the array above is the underlying contract, auto-written by the UI. The orchestrator is **pure-Python and must never import `pipeline.cv`** (whose `__init__` pulls `shapely`); YAML reads are deferred.

**Multi-model comparison (A/B).** `entity_model_overrides` values are a str (legacy) **or a list** of model names; `effective_model_names` (plural, the only resolver — shared UI/pipeline) filters stale members and falls back to the default. An entity carried by ≥2 models lands in one run per model, and its `entities[]` block is **qualified per model** (`_entity_block(compared=True)`): slug `<base>--<model>`, label/layers suffixed `— <display_name>`, plus a shared `group_label` that `finalize_service.build_entity_grouping` turns into a common QGIS group. This keeps every downstream slug-keyed contract intact (one gpkg per variant, per-slug symbology thresholds, per-layer colors). Single-model entities are byte-identical to before — never qualify them. The `model_name` detection attribute comes from the **run's** `selected_model` (`runner_shapefiles._run_model_slug`), not top-level config.

**Derived targets.** A clustering output can also be surfaced as a first-class checkable entity (a *derived target*). A model declares it in `model_card.yaml` via a `derived_targets:` list mapping a clustering rule's `output_class` (from `args.yaml`) to a catalog `entity` (+ `include_source` to also output the individual source detections). `discover_installed_models` folds the derived entity into the model's `coverage` (classes = `output_class` [+ source classes]) and records it in `InstalledModel.derived_entities` — so `resolve_runs_from_entities` is unchanged and the clustering fires via the normal "output_class ∈ selected_classes" path. **Order invariant**: `_build_cluster_options` runs *before* `_merge_derived_targets`, so a derived entity never gets a redundant "Regrouper en clusters" toggle (the UI shows a "regroupement automatique" badge instead). This is how `regroupement_crateres` ("Regroupement de cratères") is exposed on the crater models. ⚠️ The `entity:` value must exist in `data/entities_catalog.json` — an unknown id is silently ignored by the orchestrator.

### Large rasters (`existing_mnt` / `existing_rvt` regime)

`_classify_mnt_layout` / `_classify_rvt_layout` (in `src/pipeline/modes/`) inspect raster bounds:
- **standard** (~1×1 km, IGN-aligned): legacy crop + IGN naming
- **small** (<1 km or unaligned): no crop, native extent preserved
- **large** (>1.05 km in any dim): no pre-tile-splitting at all — RVT computed on the full raster, SAHI handles 640×640 slicing in memory at inference. `Image.MAX_IMAGE_PIXELS = None` is set in `convert_tif_to_png.py` and `computer_vision_onnx.py`. Input MUST be EPSG:2154 (Lambert-93).

Don't reintroduce pre-tile-splitting for the large regime — it was removed deliberately to avoid NoData artifacts on sub-tile borders.

## Branches, worktrees et dossier du plugin (règle 2026-10-07)

Le dossier `plugins/archeologia-pipeline` du profil QGIS est **ce que QGIS charge** : la branche qui y est extraite est celle que l'utilisateur teste. Le travail parallèle (une autre conversation, une branche de fonctionnalité) se fait dans un worktree **hors** de `plugins/` : `git worktree add -b <branche> C:/projets/Archeologia/worktrees/<branche> refs/heads/dev` (écrire `refs/heads/dev`, `dev` seul est ambigu avec le dossier `dev/`). Un worktree posé dans `plugins/` serait chargé par QGIS comme une seconde extension. Trois pièges vécus :

- **Ne jamais commiter depuis un worktree détaché** (`git switch --detach`) : le commit existe mais la branche n'avance pas, et l'utilisateur qui a extrait la branche dans le dossier principal ne voit rien (2026-10-07 : images du manuel « invisibles »). Après avoir détaché un worktree pour libérer la branche, commiter depuis le dossier qui la porte, ou rattacher le worktree.
- **Le packaging se fait depuis le dossier principal** : la grille IGN, les modèles et le binaire CV sont gitignorés et n'existent que là. La branche livrée doit donc y être extraite au moment du `package_plugin.py`.
- `data/models/**` étant gitignoré, un worktree n'a pas de modèle : pour un rendu hors écran de l'étape 3 ou des fiches, poser des **jonctions** `data/models/<modèle>` vers le dossier principal (PowerShell `New-Item -ItemType Junction`), jamais une copie.

**Vérifier une fenêtre sans QGIS de bureau.** `dev/rendu_hors_ecran.py` (à lancer avec le Python de QGIS : `C:/OSGeo4W/bin/python-qgis.bat dev/rendu_hors_ecran.py <commande>`) monte un `QgsApplication` en `QT_QPA_PLATFORM=offscreen`, applique le thème QSS et rend l'assistant, le manuel ou les fiches en PNG. Ce qu'il faut savoir : l'écran virtuel fait 800×600, donc `resize()` après `show()` ; `QT_SCALE_FACTOR=2` **avant** la création de l'application donne des captures en double résolution ; le `.bat` avale stdout, écrire les contrôles dans un fichier ; la faute de segmentation à la sortie (`exitQgis`) est bénigne ; le préflight de l'étape 4 échoue hors QGIS (Processing absent), ne pas capturer cet écran ainsi. C'est ce qui a permis de vérifier le manuel (défilement, liens, largeur des images à 100 % et 150 %) et le profil des scores sans mobiliser l'utilisateur.

## Installation d'un modèle : skills partagés dans le plugin Claude Code `archeologia` (règle 2026-09-15, marketplace depuis le 2026-10-08)

Un modèle est entraîné dans `C:/projets/Archeologia/training-models` (corpus, notebook Colab,
`tools/courbes_eval.py`, `verif_poids_evalues.py`, `verif_parite_onnx.py`, `tableau_modeles.py`)
et installé ici (`data/models/<id>/`, contrat `docs/model_contract.md`, validateur
`scripts/validate_models_metadata.py`, fiches `dev/fiches/`, zones d'apprentissage
`dev/fiches/zones_corpus.py` → `data/zones_corpus.json`). La procédure est portée par deux skills
**partagés, qui n'existent qu'une fois** : dans la marketplace locale
`C:/projets/Archeologia/claude-plugins` (plugin `archeologia`, lu en place), enregistrée par le
`.claude/settings.json` de chaque dépôt (`extraKnownMarketplaces` + `enabledPlugins`) — ici le seul
fichier de `.claude/` suivi avec `commands/` et le skill `manuel-integre`, le reste est gitignoré.
`/archeologia:installer-modele-plugin` puis `/archeologia:fiche-classe-plugin` (chaîne `suivant:`
héritée de training-models), et `/archeologia:depots` pour la carte des dépôts : **les invoquer dès
que le sujet apparaît**, jamais re-dériver la procédure. Modifier un skill partagé = l'éditer et le
commiter là-bas (`/reload-plugins` pour la session courante) ; le script de synchronisation et les
tests de divergence ont été retirés le 2026-10-08. Le code n'est jamais dupliqué entre les dépôts.
Les tests de logique pure du plugin vivent ici, jamais dans training-models (les doublures
« informatives » qu'il gardait ont été rapatriées le 2026-09-15 : `tests/unit/test_reset_defauts_ui.py`,
`tests/unit/test_seuils_par_classe.py`), et `tests/unit/test_doc_fiches.py` garde
`dev/fiches/README.md` en phase avec les outils, comme `test_doc_cli.py` le fait là-bas.

## Packaging conventions

`dev/package_plugin.py` produces `archeologia.<version>.zip` (e.g. `archeologia.0.7.0.zip`) — `zip_filename()` = `<PLUGIN_NAME>.<version>` from `metadata.txt`'s `version`. **The prefix before the first dot MUST equal the ZIP's internal root folder (`archeologia`)**: on a *repository* install QGIS derives the install-dir name from the zip filename split on the first `.`, so a mismatch fails with "répertoire mal nommé" (the old `ArcheologIA_v<version>.zip` → expected dir `ArcheologIA_v0` ≠ `archeologia`; `Install from ZIP`, which reads `metadata.txt` instead, still worked — masking the bug). Pass `--repo-url <base>` to also emit a matching `plugins.xml` (`file_name`/`download_url` stay in sync with the zip; the repo URL is a CLI arg, **never committed** — it points at the private OVH repo). Defaults for `--repo-url`/`--output-dir` are read from the gitignored `dev/docs/_local/deploy.config.json` (template `dev/deploy.config.example.json`), so a bare `python dev/package_plugin.py` emits both the zip and `plugins.xml` into `_local/depot` **without hardcoding the private URL** in a committed file. The root folder `archeologia` is the QGIS plugin identity, never versioned. It excludes `dev/`, `tests/`, `.git/`, `.githooks/`, `__pycache__/`, virtualenvs, and named files (`config.json`, `pytest.ini`, `conftest.py`, `run_tests.py`, `.talismanrc`, `.gitignore`). It also strips `.pt`/`.pth` checkpoints (only `.onnx` ships). When adding new dev-only files, either put them under `dev/` or extend the exclusion lists in `package_plugin.py` — otherwise they end up in users' QGIS profile.

The `data/` directory is partially gitignored: `data/models/**`, `data/quadrillage_france/`, and the compiled CV runner binaries are NOT versioned (see `.gitignore`). The quadrillage ships **inside** the ZIP (test PKG-02) as the IGN shapefile **plus its `.qix` spatial index** — the index is what makes the on-canvas tile selection (and `tile_resolver`) responsive on ~525 k features. The IGN zip (`diffusion-lidarhd.ign.fr`) is dead (503 since the Géoplateforme migration): the grid is now rebuilt from the WFS layer `IGNF_LIDAR-HD_METADONNEE:metadata` by `python dev/build_quadrillage_from_wfs.py` (same schema `nom_pkk`/`url_telech`, pagination 5 000, http→https, dash-named tiles fixed, previous set moved to `dev/docs/_local/`, `.qix` rebuilt) — **run it before each release**, the IGN adds ~2 500 tiles every 1–3 weeks. (A GeoPackage was evaluated and rejected: measured *larger* than `.shp`+`.qix`, with no row-drop benefit since every tile has a URL.)

**Jamais de `visualizations/` dans `data/models/<id>/`** (règle utilisateur 2026-09-15) : prédictions de test et planches d'inférence d'un run : Drive `runs/training/<run>/visualizations/`, ou `D:\brouillons\<chantier>\` le temps d'un test. `scripts/validate_models_metadata.py` signale le dossier en ERREUR. Les 7 modèles installés ont été purgés le 2026-09-15, chaque fichier vérifié présent à l'identique sur le Drive avant suppression.

## Versionnage

Source de vérité : `metadata.txt` → `[general] version` (lu au runtime par `src/app/plugin_metadata.py`, affiché dans le titre du dialogue). Schéma : **SemVer**, mais on est en `0.x` → pas encore de stabilité publique garantie.

La procédure de livraison complète (grille, manuel, binaire, Talisman, PR, tag, packaging, dépôt) est le runbook **`/livraison`** (`.claude/commands/livraison.md`, versionné).

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
