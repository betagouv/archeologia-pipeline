# Produit « Couverture » (QA des points sol) — Design

Date : 2026-06-11
Statut : validé par l'utilisateur (forme, activation, réglages, architecture, symbologie raster incluse).
Inspiration : script PCSAPS.sh (Yann Le Jeune, GPL) — bloc « Coverage » transposé nativement (pas de GRASS).

## Objectif

Signaler à l'utilisateur les zones où le MNT issu du TIN (`pdal:exportrastertin`) est essentiellement
de l'interpolation (couvert végétal dense, eau, occlusions) : là où très peu de points filtrés
(`Classification = 2/6/66/67/9`, le même filtre que le MNT) atteignent le sol, ni les indices RVT ni
les détections CV ne doivent être interprétés. Produit de contrôle qualité, pas de visualisation.

## Décisions validées

| Question | Décision |
|---|---|
| Forme du résultat | Raster continu 0–100 % **et** polygones « zones mal couvertes » |
| Activation | Carte « Couverture » à l'étape 2, **décochée par défaut**, modes `ign_laz`/`local_laz` uniquement |
| Réglages | Seuil réglable (défaut 30 %, spinbox 5–95) ; fenêtre fixe (disque de 5 cellules) |
| Architecture | Raster calculé **par dalle** (circuit produits existant) ; polygones extraits **une fois au finalize** sur la mosaïque |
| GRASS / script | Non — pur numpy/rasterio/shapely/geopandas, déjà présents |
| Symbologie raster | `index_COUVERTURE` reçoit une rampe pseudo-couleur dédiée (seul raster stylé du plugin) |

## Algorithme (module pur)

`src/pipeline/coverage_math.py` — numpy seul, aucune dépendance QGIS, testable standalone :

1. Masque binaire de présence : `densité > 0` (en excluant le nodata du raster densité).
2. Somme du masque sur un **disque de 5 cellules de diamètre** (13 cellules, décalages numpy,
   accumulateur `uint8` — le compte max est 13, le % tient dans 0–100 ; borne la RAM même sur un
   gros raster temporaire issu d'un LAZ fusionné).
3. Pourcentage **normalisé par le nombre réel de cellules de la fenêtre** à chaque position
   (corrige le biais de bord du `*100/13` fixe de PCSAPS).
4. Sortie `uint8` 0–100, nodata 255 (cellules sans information densité).

À la résolution densité par défaut (1 m), le disque de 5 cellules = voisinage de 5 m.

## Flux par dalle (worker)

- `ign_local_runner._process_tile` : la passe densité (`create_density_map`, idempotente) est lancée
  si `DENSITE` **ou** `COUVERTURE` est actif. Si `DENSITE` est décoché, son TIF reste en temp et
  n'est pas publié (les boucles crop/copy filtrent sur le dict produits).
- Le gate de la boucle dalles (`ign_local_runner.run`, aujourd'hui `products.needs_mnt()`) devient
  `needs_mnt() or DENSITE or COUVERTURE` — sinon une config « Couverture seule, sans indice RVT »
  sauterait silencieusement tout le traitement (quirk préexistant pour DENSITE seule, corrigé au
  passage).
- Nouveau wrapper `src/pipeline/ign/products/coverage.py` : lit le TIF densité (rasterio), applique
  `coverage_math`, écrit `<dalle>_couverture.tif` (Byte, nodata 255, même géoréférencement) en temp.
- Plomberie : `COUVERTURE` ajouté au nommage (`rvt_naming.py`, suffixe vide comme MNT/DENSITE,
  dossier `indices/COUVERTURE/`) et aux boucles produits. **Amélioration ciblée** : les trois listes
  en dur identiques (`crop.py:50`, `crop.py:148`, `results.py:214`) sont remplacées par une constante
  unique `PRODUCT_ORDER` exportée par `rvt_naming.py` (anti-bug « produit oublié dans une liste »,
  cf. avertissement `run_context.py:70-73`).
- Sortie : `indices/COUVERTURE/tif/*.tif` (crop gdalwarp ZSTD générique, pyramides), VRT construit
  automatiquement au finalize → couche raster `index_COUVERTURE`. Échec isolé par dalle.
- Fichier plus gros qu'une dalle 1 km (LAZ local fusionné) : comportement **identique à DENSITE**
  (bounds réels arrondis au km en temp, crop 1 km à la publication) — aucun cas particulier nouveau.

## Polygones au finalize (worker, opérations fichiers uniquement)

Dans `finalize_service`, après la construction des VRT, si `indices/COUVERTURE/tif/` existe :

- Lecture **par blocs** de la mosaïque (rasterio windows sur l'`index.vrt`) — la taille de la zone
  d'étude n'est jamais un problème de RAM.
- Masque `< seuil` (hors nodata) → `rasterio.features.shapes` → fusion des polygones contigus
  (shapely `unary_union`) pour effacer les coutures de blocs/dalles → filtre anti-bruit
  **surface < 25 m² éliminée** (équivalent du `v.clean rmarea` commenté dans PCSAPS).
- Attribut : `area_m2`. Écriture `indices/COUVERTURE/zones_mal_couvertes.gpkg`
  (geopandas, EPSG:2154, couche `zones_mal_couvertes`).
- Un échec de cette étape est loggé et **n'avorte pas** le finalize (audit ROB).
- Le seuil arrive par un nouveau paramètre explicite de `finalize_pipeline(...)`
  (`coverage_threshold_percent`), alimenté depuis `ctx.processing`.

## Config et RunContext

- `config.json` : `processing.products.COUVERTURE` (bool, défaut `false`) ;
  `processing.coverage_threshold_percent` (float, défaut `30.0`).
- `run_context.py` : champ `COUVERTURE: bool = False` dans `ProductsConfig` ; ajout à
  `_ALL_PRODUCTS` (pas à `_VISUALIZATION_PRODUCTS` : pas un indice RVT, ne déclenche pas
  `needs_mnt()`) ; champ `coverage_threshold_percent: float = 30.0` dans `ProcessingConfig`.
- **Normalisation** : `build_run_context` force `COUVERTURE=False` quand
  `data_mode ∉ {ign_laz, local_laz}` (un MNT livré est déjà interpolé — l'information « où étaient
  les points » n'existe plus ; vrai pour les trois layouts standard/small/large). Le runner logge
  l'indisponibilité si la config brute le demandait quand même.

## UI étape 2

- Carte « Couverture » à côté de « Densité », décochée par défaut, visible uniquement dans les
  modes LiDAR (même mécanique de visibilité par mode que le reste de l'étape).
- Réglages avancés : un seul contrôle, « Seuil zones mal couvertes (%) », spinbox 5–95, défaut 30.
- `collect_into` écrit les deux clés config ; `app/services/indices_model.py` (catalogue des
  produits) gagne l'entrée COUVERTURE ; `user_narrator.PRODUCT_LABELS` gagne « Couverture ».

## Chargement QGIS et `.qgs` de validation

- Signal `load_layers` étendu d'une 4ᵉ liste `qa_paths`
  (`log_bridge.py`, `qt_progress_reporter.py`, `run_view._on_load_layers`).
- **Couche vecteur** : factory partagée dans `layer_loader.py` — hachures rouges 45°/135°, contour
  rouge, intérieur transparent (même mécanique que le style cluster), nom
  « Zones mal couvertes (<30 %) » (seuil réel), placée **en haut** de l'arbre de couches.
- **Symbologie raster `index_COUVERTURE`** : factory partagée
  `apply_coverage_raster_symbology(layer, threshold)` — `QgsSingleBandPseudoColorRenderer` +
  `QgsColorRampShader` interpolé : 0 % → rouge opaque, seuil → orange semi-transparent,
  100 % → blanc entièrement transparent. Les zones bien couvertes s'effacent, les lacunes
  « brillent » par-dessus le MNT. Seul ce raster est stylé ; les autres restent au rendu par défaut.
- `qgs_writer.write_validation_project` ajoute la même couche vecteur (au-dessus des détections)
  et applique la même symbologie raster au VRT COUVERTURE, via les factories partagées
  (invariant existant : style identique entre chargement live et `.qgs`).
- Énums Qt/QGIS **toujours scopés** (`QgsColorRampShader.Type.Interpolated`, etc.) — compat Qt5/Qt6.

## Garde-fous

- Le produit n'est jamais une cible CV (l'orchestrateur ne cible que les dossiers RVT déclarés
  par les model cards — aucun impact).
- Pas de nouvel outil au preflight (numpy/rasterio/geopandas déjà requis côté QGIS).
- `existing_rvt` : non concerné (l'étape produits n'existe pas dans ce mode).

## Tests

- **Unitaires (standalone, sans QGIS)** : `coverage_math` (masque, % en disque, normalisation aux
  bords, nodata, types uint8) ; polygonisation sur rasters synthétiques (seuil, surface min,
  fusion de blocs) ; nommage COUVERTURE dans `rvt_naming` (+ `PRODUCT_ORDER` cohérent avec
  `_ALL_PRODUCTS`) ; `build_run_context` (nouvelles clés, normalisation par mode).
- **Manuels QGIS** (`tests/TESTS_MANUELS_QGIS.txt`) : run avec Couverture seule ; Couverture +
  Densité (la passe densité ne tourne qu'une fois) ; vérif couche hachurée, symbologie raster,
  `.qgs` de validation relu.

## Hors périmètre (assumé)

- Pas d'export PNG pour COUVERTURE (pas un support CV).
- Pas de couverture pour `existing_mnt`/`existing_rvt` (impossible sans nuage).
- Pas de symbologie pour les autres rasters (MNT, DENSITE, RVT) — sujet séparé.
- Levée de la troncature 1 km des gros LAZ locaux — chantier transverse hors feature.

## Versionnage

Feature rétro-compatible → bump **minor** à proposer au moment de la PR (convention CLAUDE.md).
