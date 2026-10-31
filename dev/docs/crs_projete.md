# MNT et indices étrangers : tout CRS projeté en mètres (2026-10-31)

Décision de l'utilisateur du 2026-10-31 : traiter des MNT et des indices étrangers (Allemagne
EPSG:25832, Royaume-Uni 27700, Irlande 2157, Pays-Bas 28992, Norvège 25833…) **sans les
reprojeter** en Lambert-93 (déformation d'échelle hors de France). Branche `feat/crs-projete`.

## Règle

| Entrée (`existing_mnt`, `existing_rvt`) | Préflight « CRS des rasters » | Sorties étiquetées en |
|---|---|---|
| CRS projeté en mètres (2154, 25832, 27700…) | ✓ `EPSG:25832 · 205 dalle(s)` | ce CRS |
| CRS géographique (degrés, ex. 4326) | ✗ bloquant | — |
| CRS projeté non métrique (pieds, ex. 2263) | ✗ bloquant | — |
| Mélange de CRS entre dalles | ✗ bloquant | — |
| Sans CRS, rien de déclaré | ⚠ `EPSG:2154 supposé` (inchangé) | EPSG:2154 |
| Sans CRS, `files.declared_crs` déclaré | ✓ | le CRS déclaré |

`ign_laz` / `local_laz` (LiDAR HD IGN) : inchangés, toujours EPSG:2154 (le runner LiDAR ne calcule
pas de CRS d'entrée ; tous les paramètres `crs` ont EPSG:2154 pour défaut).

## Source unique

`pipeline/ingest_plan.py:crs_des_entrees(dossier, extensions, declared_crs)` — CRS de la première
dalle exploitable (le préflight garantit qu'elles en ont toutes le même), sinon le déclaré, sinon
`CRS_PAR_DEFAUT` (EPSG:2154) ; `normaliser_crs` ramène un WKT géométriquement égal à Lambert-93 à
`EPSG:2154`, pour qu'une entrée française garde exactement l'étiquette d'avant. Calculé **une fois**
par `ExistingMntRunner` / `ExistingRvtRunner`, puis transmis :

- `run_existing_mnt(crs=)` → `_process_single_mnt_tile` : le MNT copié dans `technique/intermediaires/`
  reçoit le CRS s'il n'en a pas (assignation) ; indices, rognage et copies en héritent (GDAL/rvt
  conservent la projection de la source) ;
- `run_cv_post_loop(crs=)` / `run_existing_rvt(crs=)` : TIF publiés sans CRS étiquetés, et
  `cv_config["crs"]` → `runner.py` / `runner_inference.py` → `deduplicate_cv_shapefiles_final(crs=)`
  → GeoPackages de détections ; le dossier d'entrée de l'utilisateur n'est jamais modifié ;
- `finalize_pipeline(crs=)` : repli des mosaïques (`build_vrt_index(crs=)`), champ `crs` de
  `livrable/traitement.json` et de `technique/journaux/metadata_*.json`, ligne « Système de
  coordonnées des couches » du rapport ;
- projet `.qgs` : `run_view` lit `crs` dans la trace (`finalize_service.crs_du_traitement`) et le passe
  à `qgs_writer.write_validation_project(crs_authid=)` ; sans trace, CRS des couches, à défaut 2154.

## Réserves de l'audit du 2026-10-31 (traitées)

- **R1** : même CRS écrit en code EPSG et en WKT sans `AUTHORITY` (GDAL 3.11) → `identifier_epsg`, `meme_crs` ; plus de faux « mélange », étiquette toujours en code EPSG.
- **R2** : CRS du `.qgs` = celui des couches, la trace en repli (`crs_du_projet`) ; ⚠ « CRS du dossier de sortie » au préflight pour une relance dans un dossier d'un autre CRS (`alerte_relance_autre_crs`).
- **R3** : `existing_rvt._filtrer_aires_par_entite` passe `crs=` ; `layer_loader._ensure_layer_crs` se replie sur la trace du dossier de la couche (`crs_de_repli_pour`) ; `cv_runner_onnx_cli.py` garde EPSG:2154 sur son chemin de shapefiles, non emprunté (commentaire).
- **R4** : documentation (recette, contrats, docstring du préflight, manuel `livrable/indices/RVT/`), journal « Système de coordonnées du run » en indices existants.
- **R5** : le code EPSG reste dans le rapport (décision, cf. `docs/contrats.md`).

## Recensement des hypothèses Lambert-93 de `src/` (rg `2154|Lambert|L93|lambert`)

Traitées (modes à rasters fournis) :

| Lieu | Hypothèse | Traitement |
|---|---|---|
| `preflight._check_raster_crs` | `expected_crs="EPSG:2154"` : tout autre CRS refusé | plus d'`expected_crs` ; refus géographique, non métrique, mélange ; CRS affiché |
| `ingest_plan.plan_from_specs` | messages « Reprojetez en EPSG:2154 » ; pas de contrôle d'unité | contrôle `crs_unit_is_metre` (nouveau, `tilespec`) ; messages neutres ; `expected_crs` gardé (API) |
| `cv/runner.py` (×2), `cv/runner_inference.py` | `crs="EPSG:2154"` à la conversion des détections | `cv_config["crs"]`, défaut 2154 |
| `results.build_vrt_index` | `assign_crs_if_missing(src)` = 2154 | `crs=` du run |
| `modes/existing_mnt` | MNT sans CRS → indices sans CRS → mosaïque étiquetée 2154 | MNT copié étiqueté au CRS du run |
| `modes/existing_rvt` | TIF publiés sans CRS → 2154 à la mosaïque | étiquetés au CRS du run |
| `ui/qgs_writer.write_validation_project` | CRS projet `EPSG:2154` en dur (étendue initiale fausse hors 2154) | CRS de la trace, sinon des couches |
| `finalize_service`, `rapport_run` | aucune mention du CRS | `crs` dans la trace et le rapport |
| `app/services/source_modes` | « TIF en Lambert-93 » (frise et message du champ, étape 1) | « TIF projeté en mètres » |

Laissées telles quelles, et pourquoi :

| Lieu | Pourquoi |
|---|---|
| `ign/products/mnt.py`, `density.py` (`FILTER_EXTENT … [EPSG:2154]`), `results.copy_final_products_to_results` (`assign_crs_if_missing` sans CRS ; reçoit le CRS du run depuis le 2026-10-31) | LiDAR HD seulement (`ign_laz`/`local_laz`) ; en `existing_mnt` le MNT a déjà son CRS quand ces appels passent, ils sont sans effet |
| `ign/tile_resolver.py`, `ui/map_tools/grid_layer.py`, `tile_picker_tool.py`, `app/services/grid_view.py` | grille IGN des dalles LiDAR HD (France) |
| `coverage_polygons.write_low_coverage_gpkg(crs="EPSG:2154")` | produit Couverture : nuage de points requis, indisponible hors LiDAR |
| `ui/visualisation_tab.py` | catalogue des flux Géoplateforme (France), onglet masqué |
| `modes/existing_mnt._classify_mnt_layout`, `existing_rvt._classify_rvt_layout`, `neighbor_halo`, `_large_tile_name_for` | heuristiques au kilomètre **en unités du CRS** : valables dans tout CRS métrique (une dalle alignée au km en UTM est « standard ») |
| Noms des sorties `LHD_FXX_<x>_<y>_…_A_LAMB93_…` | **traité le 2026-10-31** (`rvt_naming.prefixe_dalle`/`jeton_crs`) : EPSG:2154 garde les noms d'avant à l'octet près ; un autre CRS publie `EPSG_<code>_<x>_<y>_…_A_…` (MNT `…_MNT_A_1M00_…`, sans `LAMB93`/`IGN69`, `CRS_PROJETE_` sans code EPSG). Deux jetons de préfixe : x/y restent `parts[2]`/`parts[3]` pour `coords.extract_xy_*`, `resolve_uncropped_tif`, le halo et la conversion. Une relance retire l'ancien nom Lambert-93 de la même dalle (TIF, annexes, PNG : `results._noms_remplaces`), sans quoi mosaïque et détection la compteraient deux fois |
| `cv/model_profile.py` (commentaire « Lambert-93 métrique ») | les aires minimales sont en m², vraies dans tout CRS métrique (garanti au préflight) |
| `rapport_run` (« coordonnée Lambert-93 au kilomètre » dans l'anonymisation) | le motif anonymise les noms de dalles quel que soit le CRS |

Binaire `cv_runner_onnx` : il ne lit ni n'écrit de CRS (inférence en pixels, `run_shapefile_dedup=False`,
le géoréférencement est fait côté plugin) ; `cv_config["crs"]` lui parvient et il l'ignore. Mais
`runner.py`/`runner_inference.py` sont dans le périmètre hashé de `src/pipeline/cv` : `test_binaire_a_jour`
sera rouge dans le dossier principal tant que le binaire n'est pas recompilé (`python dev/runner_onnx/build.py`,
étape 5 du runbook `/livraison`).

## Vérification

- `tests/unit/test_crs_projete.py` (rasters synthétiques 25832, 27700, 4326, 2263, sans CRS).
- Essai réel hors écran (QGIS 4.0.3 et 3.44, banc `run_e2e` de `D:/brouillons/optim_pipeline`) sur
  4 fenêtres MNT du Schleswig-Holstein (EPSG:25832, 1 m), `existing_mnt` (HS, LD, VAT + enclos_fr)
  puis `existing_rvt` sur les LD produits : 48 rasters, VRT et GeoPackage en EPSG:25832, `.qgs` en
  EPSG:25832, chaque emprise recoupe un monument de `de_sh_denkmalliste.gpkg` (EPSG:25832).
- Non-régression française (mêmes bancs `mnt_existant` et `ia_enclos_fr_seg_ld_v1 --smoke`, `dev` contre
  la branche) : TIF, VRT, PNG, world files identiques à l'octet ; GeoPackages identiques en contenu ;
  `traitement.json` identique au champ `crs` près ; `.qgs` : mêmes authid, même étendue.
