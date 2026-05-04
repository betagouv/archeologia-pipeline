# Refonte architecturale — Archeolog'IA pipeline

## Contexte

Ce document propose une refonte progressive de l'architecture du plugin QGIS, motivée par trois objectifs explicitement priorisés :

1. **Testabilité accrue** — élargir la frontière des modules testables hors QGIS
2. **Navigabilité IA / lecture humaine** — réduire le bouncing entre fichiers, concentrer la logique liée
3. **Évolution facilitée du CV** — rendre simple l'ajout de modèles, types de tâches, post-processings

**Contrainte forte** : aucune régression fonctionnelle. Le pipeline doit produire exactement les mêmes résultats à chaque étape. Toutes les opportunités proposées sont des refactors comportementalement équivalents (bit-pour-bit côté outputs), pas des changements fonctionnels.

**Vocabulaire utilisé** :
- **Module** = unité avec interface + implémentation (fonction, classe, package).
- **Deep** = peu d'interface pour beaucoup de comportement (bonne leverage).
- **Shallow** = interface presque aussi complexe que l'implémentation (faible leverage).
- **Deletion test** = si on inline le module, la complexité se concentre (deep, à garder) ou se duplique (deep, à garder) ou disparaît (shallow, à inliner).
- **Seam** = endroit où l'on peut altérer un comportement sans éditer en place.
- **Locality** = un changement métier touche peu d'endroits.

---

## Synthèse des frictions trouvées

### CV (priorité utilisateur n°1)

| # | Friction | Localisation | Impact |
|---|----------|--------------|--------|
| C1 | `confidence_threshold` a 3 sources de vérité (cv_config / `.json` modèle / args.yaml) sans priorité claire | `runner.py:293`, `computer_vision_onnx.py:1231+`, `model_config.py:241` | Bugs silencieux selon la source qui l'emporte |
| C2 | `args.yaml` re-lu 3-4 fois par run (SAHI / clustering / postprocess) | `runner.py:140`, `conversion_shp.py:474`, `postprocessing.py`, `qgs_project.py` | Lectures répétées, pas de cache, source de vérité éclatée |
| C3 | Format `Detection` ambigu : `polygon` optionnel pour détection, obligatoire pour segmentation | seam `computer_vision_onnx.py` ↔ `conversion_shp.py` | 2 branches de code à maintenir, source de bugs |
| C4 | Validation de `selected_classes` dupliquée à 2 endroits avec sémantiques différentes | `runner.py:95-99` (court-circuit) + `conversion_shp.py:514` (filtre clustering) | Risque de générer du GPKG inutile |
| C5 | Logique de fusion de polygones dupliquée | `postprocessing.py` (`_merge_touching_same_class`) + `conversion_shp.py:939+` | Pas clair lequel s'exécute en priorité |
| C6 | `cv_output.py` (381 LOC) = pur I/O formatting, shallow | `cv_output.py` | Coût d'apprentissage > valeur ajoutée |
| C7 | Séparation `class_utils.py` / `model_config.py` artificielle, ré-exports croisés | `class_utils.py:22-28` ré-exporte `model_config` | Bouncing inutile, cycles d'imports flous |
| C8 | `runner.py` (538 LOC) mélange orchestration + cache + fallback ONNX | `runner.py` | Difficile à comprendre/modifier isolément |
| C9 | Parcours « détection → SHP final » oblige à lire 5-6 fichiers (~500 LOC) | tout `src/pipeline/cv/` | Friction de navigation forte |
| C10 | Frontière « pure-Python détections / QGIS-side géométrie » floue | `clustering.py`, `postprocessing.py`, `conversion_shp.py`, `qgs_project.py` | Empêche de tester la pipeline post-inférence sans QGIS installé |

### Runners et orchestration

| # | Friction | Localisation | Impact |
|---|----------|--------------|--------|
| R1 | Validation+extraction de config dupliquée entre les 3 runners (~20 lignes × 3) | `ign_local_runner.py`, `existing_mnt_runner.py`, `existing_rvt_runner.py` | ~60 lignes éliminables |
| R2 | Logique CV post-processing dupliquée entre `existing_mnt_runner` et `existing_rvt_runner` (la version `ign_local._run_post_cv` est correcte) | `existing_mnt_runner.py:75-125`, `existing_rvt_runner.py:46-89` | Doit être touchée en 2 endroits |
| R3 | `RunContext` est un transport de `Dict[str, Any]` non typés | `run_context.py` | `.get()` partout, aucun support IDE, validations dispersées |
| R4 | `IgnOrLocalRunner` traite 2 modes via 12 `if ctx.mode == "ign_laz"` | `ign_local_runner.py` | Complexité cyclomatique élevée |
| R5 | `helpers.py` mélange utilitaire vraiment partagé (`log_section`) et quasi-local (`safe_float`, `resolve_rvt_tif_dir`) | `runners/helpers.py` | Couplage faux-positif |
| R6 | `finalize_pipeline` est en revanche un excellent exemple de service partagé sans branche `if mode ==` | `services/finalize_service.py` | (à préserver tel quel) |

### Transverse et UI

| # | Friction | Localisation | Impact |
|---|----------|--------------|--------|
| T1 | `main_dialog.py` = **2745 lignes** mélangeant UI Qt, validation métier, transformation config, gestion CV | `src/ui/main_dialog.py` | Quasi-impossible à tester, navigation pénible |
| T2 | `build_run_context()` est une coquille vide (extrait des sous-dicts sans valider). Validation dupliquée en UI + preflight + runners | `app/run_context.py`, `pipeline/preflight.py`, `ui/main_dialog.py:956-990` | 3 endroits à toucher pour ajouter une règle |
| T3 | Aucune abstraction commune dans `pipeline/ign/products/` — 6 scripts qui appellent chacun `run_qgis_algorithm` avec leur propre logique de bounds, naming, params | `pipeline/ign/products/*.py` | Duplication `_as_int/_as_float` entre `indices.py` et `rvt_naming.py` |
| T4 | Modules pure-Python testables mais exclus globalement par `pytest.ini:norecursedirs = src/pipeline` | `coords.py`, `geo_utils.py`, `output_paths.py`, `pdal_validation.py`, `tile_resolver.py`, `coords_fallback.py` | Couverture de tests artificiellement basse |
| T5 | `tile_splitter.py` (301 LOC) orphelin — seule la constante `IGN_TILE_SIZE_M` est importée | `pipeline/ign/products/tile_splitter.py` | Code mort à supprimer |
| T6 | `subprocess_utils.py` shallow (21 LOC, 2 callers, 1 fonction triviale) | `pipeline/subprocess_utils.py` | Inlining envisageable |

---

## Opportunités de deepening (proposition)

Chaque opportunité applique le **deletion test** : on garde une nouvelle abstraction si elle concentre la complexité chez peu d'endroits ; on en supprime une existante si la complexité disparaît une fois inlinée.

Les opportunités sont ordonnées par **ratio impact/risque** pour servir directement de séquencement.

---

### Vague 1 — Préparer le terrain (faible risque, gain immédiat)

Ces refactors sont mécaniques, sans changement de comportement, et débloquent les vagues suivantes.

#### V1.1 — Élargir la frontière testable

**Problème** (T4) : `pytest.ini` exclut globalement `src/pipeline`, alors que ~6 modules y vivent sans aucun import QGIS au top-level.

**Solution** :
- Remplacer `norecursedirs = src/pipeline` par `collect_ignore_glob` ciblé sur les modules qui importent réellement QGIS.
- Concrètement, retirer de l'exclusion : `coords.py`, `geo_utils.py`, `output_paths.py`, `subprocess_utils.py`, `ign/coords_fallback.py`, `ign/pdal_validation.py`, `ign/tile_resolver.py`.
- Garder exclus : `ign/downloader.py` (importe QSettings au top-level), tout `ign/products/`, tout `modes/`, tout `cv/` (pour cette vague).

**Fichiers** : `pytest.ini`, `conftest.py`.

**Bénéfice testabilité** : ouvre la possibilité de tester ~1100 LOC de logique pure (algos coords, geotransform, validation PDAL).

**Bénéfice navigabilité** : nul direct, mais sert de base aux refactors CV.

**Risque** : zéro côté production. Risque de découvrir que certains de ces modules importent indirectement QGIS — auquel cas on reculer cas par cas.

#### V1.2 — Supprimer le code mort

**Problème** (T5) : `tile_splitter.py` (301 LOC) n'est plus appelé. Seule la constante `IGN_TILE_SIZE_M` est importée par `existing_mnt.py` et `existing_rvt.py`.

**Solution** :
- Déplacer `IGN_TILE_SIZE_M` dans un module léger `pipeline/constants.py` (ou inliner dans `output_paths.py` qui contient déjà des conventions IGN).
- Supprimer `tile_splitter.py`.
- Mettre à jour `dev/package_plugin.py` si besoin.

**Fichiers** : `pipeline/ign/products/tile_splitter.py` (suppression), `pipeline/modes/existing_mnt.py`, `pipeline/modes/existing_rvt.py`, nouveau `pipeline/constants.py` (ou édition `output_paths.py`).

**Bénéfice navigabilité** : -301 LOC à scanner, -1 fichier dans l'arborescence.

**Risque** : faible. Le CLAUDE.md mentionne qu'il est gardé pour « explicit opt-in cases » — vérifier dans la grep qu'aucun appel dynamique (par nom) n'existe avant suppression. Si un cas d'usage légitime apparaît, restaurer depuis git.

#### V1.3 — Inliner `subprocess_utils.py` ou le déplacer

**Problème** (T6) : 21 LOC, 1 fonction (`subprocess_kwargs_no_window`), 2 callers.

**Solution recommandée** : ne PAS inliner. Le déplacer sans le toucher dans `src/app/common/` ou le garder où il est. Justification : c'est une convention Windows transverse, le deletion test est ambigu (la complexité ne se duplique pas mais la dispersion d'un détail OS-specific est gênante). Garder centralisé.

**Décision** : laisser tel quel. Mentionné ici pour clore explicitement la question.

---

### Vague 2 — Le contrat CV (cœur de la priorité utilisateur)

Cette vague vise la racine de la friction CV : un seul objet `Detection` typé, une seule source de configuration par modèle, un seul chemin de fusion.

#### V2.1 — `Detection` dataclass + module pur

**Problèmes adressés** : C3 (format ambigu), C9 (bouncing), C10 (frontière pure/QGIS floue).

**Solution** :
- Créer `src/pipeline/cv/types.py` (ou enrichir `pipeline/types.py`) avec :
  ```python
  @dataclass(frozen=True)
  class Detection:
      bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (pixels)
      class_id: int
      confidence: float
      mask_polygon: Optional[list[tuple[float, float]]] = None  # contour pixels, segmentation seulement
  ```
- Module **zéro-dépendance lourde** (pas de shapely, pas de geopandas, pas de QGIS).
- Méthodes statiques `from_yolo_bbox()`, `from_segmentation_mask()`, `to_dict()`, `from_dict()` pour I/O JSON.
- Refactor de `cv_output.py` et `computer_vision_onnx.py` pour produire/consommer des `Detection`.
- Refactor de `conversion_shp.py` pour consommer des `Detection` au lieu de `dict`.

**Deletion test** : si on supprime `Detection` et qu'on retourne au dict, la duplication réapparaît dans 4 fichiers (`computer_vision_onnx`, `external_runner` côté JSON, `conversion_shp`, `cv_output`). → DEEP.

**Fichiers** : nouveau `src/pipeline/cv/types.py`, modifs `cv_output.py`, `conversion_shp.py`, `computer_vision_onnx.py`, `external_runner.py` (côté contrat JSON).

**Bénéfice testabilité** : on peut tester `Detection.from_yolo_bbox()` et `Detection.from_segmentation_mask()` avec des numpy synthétiques, sans aucun setup CV.

**Bénéfice navigabilité** : un développeur qui se demande « qu'est-ce qu'une détection » a un seul fichier à lire. Le seam JSON ↔ Python devient explicite.

**Risque** : moyen. Le contrat JSON sortant de `external_runner` (binaire compilé) ne change pas — seul le parsing côté plugin évolue. Tests de non-régression : comparer les SHP avant/après sur un dataset connu.

#### V2.2 — `ModelProfile` : une seule source de vérité par modèle

**Problèmes adressés** : C1 (3 sources pour confidence), C2 (args.yaml relu 3×), C7 (séparation class_utils/model_config artificielle).

**Solution** :
- Créer `src/pipeline/cv/model_profile.py` avec une classe `ModelProfile` :
  ```python
  @dataclass(frozen=True)
  class ModelProfile:
      name: str
      weights_path: Path
      class_names: list[str]
      class_colors: dict[int, tuple[int, int, int]]
      sahi: SahiConfig          # slice_height, slice_width, overlap, ...
      clustering: ClusteringConfig  # eps_m, min_samples, ...
      postprocess: PostprocessConfig  # merge, overlap_removal, ...
      confidence_threshold: float
      task_type: Literal["detection", "segmentation"]
      metadata: dict  # bg_bias, buffer_distance, etc., venant du .json modèle

      @classmethod
      def load(cls, model_dir: Path, run_overrides: dict | None = None) -> "ModelProfile":
          ...  # lit args.yaml + .json métadonnées + applique overrides du run
  ```
- Documenter et coder explicitement la priorité : `run_overrides > .json modèle > args.yaml > defaults`.
- Charger **une seule fois par run**, passer le `ModelProfile` partout.
- Fusionner `class_utils.py` + `model_config.py` dans ce nouveau module (ou en sous-modules clairs : `model_profile/loader.py`, `model_profile/types.py`).

**Deletion test** : si on supprime `ModelProfile` et qu'on revient à la lecture éparpillée, on retrouve 4 chargements YAML/JSON + 3 sources pour `confidence_threshold` + ré-exports cycliques. → DEEP.

**Fichiers** : nouveau `src/pipeline/cv/model_profile.py` (+ sous-modules), suppression de `class_utils.py` et `model_config.py` (le contenu y migre), modifs `runner.py`, `conversion_shp.py`, `computer_vision_onnx.py`, `postprocessing.py`, `qgs_project.py`.

**Bénéfice testabilité** : on peut tester `ModelProfile.load()` avec un répertoire fixture sans toucher au pipeline. La priorité des sources de config devient testable noir-sur-blanc.

**Bénéfice navigabilité** : le développeur cherche « comment configurer un modèle » → un seul fichier répond.

**Bénéfice évolution CV** : ajouter un nouveau type de modèle ou un nouveau paramètre = éditer un seul endroit (le ModelProfile + son loader). Aujourd'hui c'est 3-4 endroits.

**Risque** : moyen. Migration en 2 étapes recommandée : (a) créer `ModelProfile` qui *façade* les anciens loaders sans les supprimer ; (b) migrer les call sites un à un ; (c) supprimer les anciens. Tests de non-régression : run pipeline complet avec chaque modèle existant, diff des outputs.

#### V2.3 — Un seul module de fusion de polygones

**Problèmes adressés** : C5 (fusion dupliquée), C4 (validation `selected_classes` à 2 endroits).

**Solution** :
- Centraliser toute la logique de fusion/overlap-removal/clustering dans un module `src/pipeline/cv/geometry_pipeline.py` (à créer) ou enrichir `postprocessing.py` pour qu'il soit l'unique point d'entrée.
- Supprimer la fusion inline dans `conversion_shp.py:939+`.
- Définir une fonction unique : `apply_geometry_pipeline(detections: list[Detection], profile: ModelProfile, raster_meta: RasterMeta) -> list[GeometryFeature]`.
- Filtrer `selected_classes` une seule fois, en amont de tout (ou : centraliser le filtre dans `geometry_pipeline` et supprimer le court-circuit de `runner.py`, qui devient redondant).

**Deletion test** : sans ce module, on a 2 implémentations de fusion qui peuvent diverger silencieusement. → DEEP de centraliser.

**Fichiers** : `pipeline/cv/postprocessing.py` (renommé éventuellement `geometry_pipeline.py`), `pipeline/cv/conversion_shp.py` (perd ses fonctions de fusion), `pipeline/cv/runner.py` (perd le court-circuit `selected_classes=[]`, ou le garde mais le délègue).

**Bénéfice testabilité** : `apply_geometry_pipeline` prend des `Detection` et retourne des features → testable avec `pytest` et fixtures shapely synthétiques, sans QGIS. (Suppose shapely installé en environnement de test, ce qui est raisonnable.)

**Bénéfice navigabilité** : « comment une détection devient-elle un polygone fusionné » → un seul fichier répond.

**Risque** : moyen-élevé. La fusion est de la géométrie sensible — la moindre divergence peut produire des polygones différents (même si visuellement équivalents). Stratégie : refactor + tests de non-régression sur ≥ 2 datasets (1 détection bbox, 1 segmentation), comparer les GPKG produits avec un diff géométrique tolérant aux ré-ordonnancements.

#### V2.4 — Découper `runner.py` (538 LOC)

**Problème adressé** : C8.

**Solution** :
- `runner.py` (allégé) : orchestration pure (pour chaque image : appeler inférence → produire détections → déléguer à `geometry_pipeline` → écrire SHP).
- `runner_inference.py` : choix runner externe vs fallback ONNX, gestion subprocess, parsing JSON sortant.
- `runner_cache.py` : logique de short-circuit / cache / `selected_classes=[]`.

**Deletion test** : si on supprime ce découpage et qu'on garde un fichier monolithique, la complexité reste mais la lecture devient pénible. → SHALLOW *en termes de leverage technique* mais **DEEP en termes de locality / navigabilité**, qui est précisément la priorité utilisateur n°2. Garder le découpage.

**Fichiers** : `pipeline/cv/runner.py` (split), nouveaux `runner_inference.py`, `runner_cache.py`.

**Bénéfice navigabilité** : 3 fichiers de ~150-200 LOC à un sujet chacun, au lieu d'un fichier de 538 LOC à 3 sujets.

**Risque** : faible — pur déplacement de code.

---

### Vague 3 — Renforcer le contrat des runners

Cette vague typage le `RunContext` et factorise les patterns dupliqués entre les 3 runners.

#### V3.1 — Typer le `RunContext` avec des sous-dataclasses

**Problème adressé** : R3.

**Solution** :
- Remplacer les `Dict[str, Any]` par des dataclasses :
  ```python
  @dataclass(frozen=True)
  class ProductsConfig:
      mnt: bool
      densite: bool
      m_hs: bool
      svf: bool
      slo: bool
      ld: bool
      vat: bool

  @dataclass(frozen=True)
  class ProcessingConfig:
      max_workers: int
      tile_overlap: float
      filters: dict
      output_structure: OutputStructureConfig
      output_formats: list[str]

  @dataclass(frozen=True)
  class CvConfig:
      enabled: bool
      target_rvt: str
      runs: list[CvRun]
  ```
- `build_run_context` devient le **seul** lieu de validation et de typage des champs.
- Les runners consomment `ctx.products.svf` au lieu de `ctx.products_cfg.get("SVF", False)`.

**Deletion test** : retour aux dicts → re-explosion des `.get()` partout, perte de l'IDE support et des type checks. → DEEP.

**Fichiers** : `app/run_context.py` (gros refactor), `app/runners/*.py` (consommation), `app/services/finalize_service.py`.

**Bénéfice testabilité** : `build_run_context` peut être testé exhaustivement avec des configs valides/invalides.

**Bénéfice navigabilité** : autocomplétion dans l'IDE, schéma typé visible en 1 ouverture de fichier.

**Bénéfice évolution** : ajouter un produit RVT = ajouter un champ dans `ProductsConfig` + sa validation, le compilateur indique tous les call sites à mettre à jour.

**Risque** : moyen. Migration progressive possible : garder `products_cfg` en parallèle pendant la transition, déprécier en fin de chantier. Tests de non-régression : config snapshots existants doivent tous parser sans changement.

#### V3.2 — Extraire un `CvPostProcessingService`

**Problème adressé** : R2.

**Solution** :
- Extraire la logique CV post-loop (présente proprement dans `IgnOrLocalRunner._run_post_cv` mais dupliquée inline dans `existing_mnt_runner` lignes 75-125 et `existing_rvt_runner` lignes 46-89) dans un service partagé.
- Nouveau module `src/app/services/cv_post_service.py` exposant `run_cv_post(ctx, output_structure, rvt_params, reporter, cancel, slog)`.
- Les 3 runners l'appellent.

**Deletion test** : sans ce service, on doit toucher 3 fichiers pour modifier la boucle CV. → DEEP.

**Fichiers** : nouveau `app/services/cv_post_service.py`, modifs des 3 runners.

**Bénéfice testabilité** : un seul service à mocker pour tester l'orchestration des runners.

**Bénéfice navigabilité** : 1 endroit pour comprendre comment les runs CV sont enchaînés.

**Risque** : faible — extraction mécanique.

#### V3.3 — Factoriser l'extraction de config commune aux runners

**Problème adressé** : R1.

**Solution** :
- Une fois V3.1 fait, l'extraction est déjà typée. Reste à supprimer les vérifications répétées (`if dir_str: ...`, `if ctx.output_dir is None: ...`).
- Déplacer ces validations dans `build_run_context` (ou créer un `validate_for_mode(ctx)` appelé par `PipelineController.run()` juste après preflight).

**Fichiers** : `app/run_context.py`, `app/runners/*.py` (suppression code répété), éventuellement `app/pipeline_controller.py`.

**Bénéfice** : -60 LOC dupliquées, validation centralisée.

**Risque** : faible.

#### V3.4 — Ranger `helpers.py`

**Problème adressé** : R5.

**Solution** :
- `log_section()` → migrer dans `StructuredLogger` (vraie méthode plutôt qu'helper externe).
- `safe_float()` → garder en helpers, ou mieux : déplacer dans `pipeline/types.py` ou un futur `app/common/numeric.py` (utilitaire transverse).
- `resolve_rvt_tif_dir()` → déplacer dans `app/services/` (c'est un service, pas un helper).
- Supprimer `runners/helpers.py`.

**Fichiers** : `app/structured_logger.py`, `app/runners/helpers.py` (suppression), nouveaux ou enrichis modules cibles, modifs callers (3 runners).

**Bénéfice navigabilité** : disparition d'un dump module.

**Risque** : faible.

#### V3.5 — Décision sur `IgnOrLocalRunner`

**Problème adressé** : R4 (12 branchements `if ctx.mode == "ign_laz"`).

**Solution recommandée** : **ne pas séparer** en 2 classes pour l'instant. Le runner partage 80 % de sa logique entre les 2 modes — séparer dupliquerait la majorité du code. À la place :
- Extraire les 12 branchements dans une méthode `_get_input_strategy(ctx) -> InputStrategy` retournant un objet polymorphe (`IgnDownloadStrategy` ou `LocalLazStrategy`) qui encapsule la différence (où récupérer les LAZ, comment résoudre les dalles).
- Le runner devient inconscient du mode après cette ligne.

**Deletion test** : sans `InputStrategy`, on garde des if mode partout — au moindre 3e mode (ex : nouvelle source LiDAR), on touche 12 endroits. → DEEP de l'introduire.

**Fichiers** : `app/runners/ign_local_runner.py`, nouveau `app/runners/input_strategy.py` (ou inline dans le module).

**Bénéfice évolution** : ajouter une 3e source LiDAR = créer une 3e `InputStrategy`, plus de toucher au runner.

**Risque** : faible-moyen. Refactor mécanique mais à tester en intégration sur les 2 modes.

---

### Vague 4 — Décomposer l'UI

Vague la plus risquée car `main_dialog.py` est 100 % couplé Qt. À faire en dernier, après que les contrats backend soient stabilisés.

#### V4.1 — Extraire `ConfigWidgetAdapter`

**Problème adressé** : T1, T2.

**Solution** :
- Nouveau module `src/ui/config_widget_adapter.py`.
- Une classe `ConfigWidgetAdapter` qui prend en entrée les widgets Qt et expose deux méthodes : `to_config() -> dict` et `from_config(dict) -> None`.
- Le dialog devient un agent qui assemble des widgets ; toute la transformation widgets ↔ dict vit dans l'adapter.

**Deletion test** : sans cet adapter, `_collect_config_from_widgets` reste enfermé dans le dialog → impossible à tester sans Qt. → DEEP.

**Fichiers** : nouveau `ui/config_widget_adapter.py`, modifs `ui/main_dialog.py`.

**Bénéfice testabilité** : la conversion peut être testée avec des widgets Qt headless (`QCoreApplication` minimal) ou même mocked.

**Bénéfice navigabilité** : -100 LOC dans le dialog.

**Risque** : moyen. Tests manuels Qt via `tests/TESTS_MANUELS_QGIS.txt`.

#### V4.2 — Extraire `ValidationEngine`

**Problème adressé** : T2.

**Solution** :
- Une fois `build_run_context` typée et validante (V3.1 + V3.3), l'UI peut consommer ses validators directement plutôt que les redupliquer.
- Idéalement : `_validate_can_run()` devient un appel à `RunContext.try_build(config)` qui retourne soit le contexte, soit une liste d'erreurs.

**Fichiers** : `ui/main_dialog.py`, `app/run_context.py`.

**Bénéfice** : 1 source de vérité de validation au lieu de 3.

**Risque** : faible si V3.1 est fait proprement.

#### V4.3 — Extraire `CvRunsTableManager` (optionnel)

**Problème adressé** : T1.

**Solution** :
- 200+ LOC dans le dialog gèrent la table CV multi-runs.
- Extraire dans `src/ui/cv_runs_table.py` une classe widget dédiée.

**Risque** : moyen. À faire seulement si on touche encore beaucoup à cette zone, sinon laisser tel quel (refactor à coût élevé pour gain incrémental).

---

### Hors-vague — Décisions explicites de NE PAS faire

- **`pipeline/ign/products/` : ne pas tenter d'abstraction commune** (T3). Tentant car les 6 modules se ressemblent, mais le deletion test échoue : leurs paramètres sont fondamentalement différents (MNT prend un LAZ, indices prend un MNT, crop prend un raster + bbox). Une classe abstraite `Product` ferait fuiter ses params via `**kwargs` et empirerait la lisibilité. **Garder tel quel.** Action minimale : factoriser uniquement les helpers `_as_int`, `_as_float`, `_as_bool` dupliqués entre `indices.py` et `rvt_naming.py` dans `pipeline/types.py`.
- **`finalize_pipeline` : ne pas y toucher** (R6). C'est l'exemple à suivre. 0 branche `if mode ==`, paramétré par arguments, parfait deep module.
- **`PipelineController` : ne pas y toucher**. 89 LOC, orchestration pure, fait son travail.
- **Garder `external_runner` strict pure-Python** (CLAUDE.md). Toute évolution doit respecter cette frontière.

---

## Séquencement recommandé

| Vague | Risque | Effort estimé | Bloque quoi | À faire en parallèle ? |
|-------|--------|---------------|-------------|------------------------|
| V1.1 (frontière test) | Très faible | 0.5j | — | Oui |
| V1.2 (code mort) | Très faible | 0.5j | — | Oui |
| V2.1 (Detection dataclass) | Moyen | 1.5j | V2.3 | Non — fondation |
| V2.2 (ModelProfile) | Moyen | 2j | V2.3, V2.4 | Non — fondation |
| V2.3 (geometry pipeline) | Moyen-élevé | 2j | — | Après V2.1 |
| V2.4 (split runner.py) | Faible | 0.5j | — | Après V2.2 |
| V3.1 (typage RunContext) | Moyen | 1.5j | V3.2, V3.3, V4.2 | Non — fondation |
| V3.2 (CvPostService) | Faible | 0.5j | — | Après V3.1 |
| V3.3 (validation centralisée) | Faible | 0.5j | V4.2 | Après V3.1 |
| V3.4 (helpers.py rangé) | Faible | 0.3j | — | Indépendant |
| V3.5 (InputStrategy) | Faible | 1j | — | Indépendant |
| V4.1 (ConfigWidgetAdapter) | Moyen | 1.5j | V4.2 | Après V3.1 |
| V4.2 (ValidationEngine) | Faible | 0.5j | — | Après V4.1 + V3.3 |
| V4.3 (CvRunsTable) | Moyen | 1j | — | Optionnel |

**Total** : ~13-14j de travail si tout est fait. Chaque vague livre indépendamment de la valeur — possible de s'arrêter après la Vague 2 si la priorité CV est satisfaite.

**Ordre recommandé** : V1 → V2 (priorité CV utilisateur) → V3 → V4. À l'intérieur d'une vague, suivre l'ordre indiqué dans la table de séquencement (les fondations d'abord).

---

## Vérification de non-régression

La contrainte « zéro régression fonctionnelle » impose une stratégie de test à chaque vague.

### Tests automatiques
- `python run_tests.py` doit passer à 100 % avant et après chaque PR.
- À chaque vague, ajouter des tests unitaires pour les nouveaux modules (`Detection`, `ModelProfile`, `geometry_pipeline`, `RunContext` typé) — viser ≥ 80 % de couverture sur le code touché.
- `ruff check src/` propre.

### Tests d'intégration en environnement QGIS
Suivre `tests/TESTS_MANUELS_QGIS.txt` à chaque fin de vague. Pour chaque mode (`ign_laz`, `local_laz`, `existing_mnt`, `existing_rvt`) :
1. Lancer le pipeline complet sur un dataset de référence (à figer en début de chantier).
2. Comparer les outputs **bit-pour-bit** quand possible (TIFF, JSON metadata).
3. Comparer les GPKG/SHP avec un diff géométrique (`ogr2ogr` + script de comparaison) — tolérer les ré-ordonnancements de features mais pas les changements géométriques.

### Datasets de référence à préparer en amont
- 1 dalle IGN standard (mode `ign_laz` complet, 1 km × 1 km)
- 1 LAZ local (mode `local_laz`)
- 1 MNT existant petit format (mode `existing_mnt`, < 1 km)
- 1 RVT existant grand format (mode `existing_rvt`, > 1 km, déclenche le régime large)
- 1 modèle de détection bbox (YOLO)
- 1 modèle de segmentation (avec masks)

Stocker les outputs « avant refactor » comme baseline. Comparer après chaque vague.

### Critère d'acceptation par vague
Une vague est terminée quand :
- Tous les tests passent.
- Les 4 modes produisent des outputs identiques à la baseline.
- Le packaging `python dev/package_plugin.py` produit toujours un ZIP installable.
- Le plugin se charge sans erreur dans QGIS.

---

## Fichiers critiques (à connaître avant d'attaquer)

- `src/pipeline/cv/runner.py` — entrée CV, à splitter (V2.4)
- `src/pipeline/cv/computer_vision_onnx.py` — 1476 LOC, hôte de fonctions pures à extraire (V2.1)
- `src/pipeline/cv/conversion_shp.py` — consommateur principal des Detection (V2.1) et co-hôte de fusion (V2.3)
- `src/pipeline/cv/postprocessing.py` — futur `geometry_pipeline.py` (V2.3)
- `src/pipeline/cv/model_config.py` + `class_utils.py` — à fusionner en `model_profile.py` (V2.2)
- `src/pipeline/cv/external_runner.py` — contrat JSON, à respecter strictement
- `src/app/run_context.py` — à typer (V3.1)
- `src/app/runners/*.py` — consommateurs de RunContext typé (V3.1, V3.2, V3.3, V3.5)
- `src/app/services/finalize_service.py` — **ne pas toucher**
- `src/app/pipeline_controller.py` — **ne pas toucher**
- `src/ui/main_dialog.py` — décomposition différée à V4
- `pytest.ini`, `conftest.py` — V1.1
- `dev/package_plugin.py` — vérifier après chaque suppression de fichier

---

## Annexe : utilitaires partagés à créer pendant le chantier

Au fil du chantier, plusieurs petites concentrations naturelles vont apparaître. Les anticiper :
- `src/pipeline/types.py` enrichi : `Detection`, `RasterMeta`, helpers de coercion (`_as_int`, `_as_float`, `_as_bool` dédupliqués).
- `src/pipeline/cv/types.py` ou enrichissement `pipeline/types.py` : selon la taille finale.
- `src/pipeline/constants.py` : `IGN_TILE_SIZE_M` et autres constantes IGN.
- `src/app/common/` : numeric utils si nécessaire, sinon ignorer.

Ne créer ces modules **que quand 2+ sites les justifient** (la règle « one adapter = hypothetical seam, two adapters = real seam »).
