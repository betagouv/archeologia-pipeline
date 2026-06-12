# Refonte de l'attribution des couleurs des détections

**Date :** 2026-06-12 · **Statut :** design validé (brainstorming), à implémenter

## 1. Contexte et problème

Lors d'un run, deux entités (`tranchees_et_boyaux` et `parcellaire`) sont apparues
avec **la même couleur de base verte** dans QGIS, leurs tranches de confiance étant
seules à varier.

**Cause racine — collision d'index.** Le système actuel a trois étages :

1. **Palette fixe** ([`class_utils.py:127`](../../../src/pipeline/cv/class_utils.py)) :
   12 couleurs RGB, dont **3 verts perceptuellement proches** (index 1, 8, 10) et
   2 bleus proches (2, 11).
2. **Mapping `classe → index`** avec deux cascades **divergentes** :
   - à la génération du gpkg ([`conversion_shp.py:1189-1193`](../../../src/pipeline/cv/conversion_shp.py)) :
     `global_color_map[class_name]` → `class_colors[class_id]` → **`class_id` brut** ;
   - à l'affichage ([`layer_loader.py:413` `_resolve_color_idx`](../../../src/ui/layer_loader.py)) :
     `global_color_map[class_name]` → fuzzy substring → **`0` (rouge)** → lecture de `conf_color`.
3. **Application** : renderer catégorisé sur `conf_bin`, chaque tranche = la couleur de
   base déclinée en luminosité.

Deux classes de **modèles différents** partageant le même `class_id` (ex. 1) retombent
toutes deux sur l'index 1 = vert lime quand le mapping global n'est pas peuplé/transmis.
Aggravant : `class_colors` d'`args.yaml` est non gardé — `run_rf_detr_1` a `[2,2,2,0,7]`,
soit **3 classes forcées sur le même bleu**.

**Contrainte produit :** beaucoup de classes seront ajoutées avec le temps → toute palette
fixe est condamnée à terme.

## 2. Objectifs

- Couleur **distincte et stable par classe**, déterministe, **sans limite de nombre de classes**.
- **Une seule source de vérité**, cohérente entre l'affichage live, le `.qgs` et le `conf_color`
  du gpkg — y compris pour les **gpkg déjà générés** (pas de régénération).
- La **confiance reste portée par la luminosité** (5 niveaux, comportement inchangé).
- **Tout auto** : la couleur est toujours dérivée du nom de classe ; on supprime
  `class_colors` d'`args.yaml` (source des collisions manuelles).

## 3. Conception

> **Évolution validée en cours d'implémentation.** Un prototype « hash pur sans état »
> a été écrit puis écarté : sur un jeu réaliste de 10 classes, deux tombaient à une
> distance perceptuelle de 3/255 (quasi-collision) — aucune fonction *stateless* ne
> peut garantir à la fois stabilité et distinction. Après arbitrage utilisateur, on
> retient un **registre de rangs stable**.

### 3.1 Module pur `src/pipeline/cv/color_palette.py`

- `base_color_for_rank(rank: int) -> (r,g,b)` : teinte = `(rank × φ⁻¹) mod 1.0` (nombre
  d'or conjugué → rangs consécutifs maximalement écartés sur le cercle chromatique) ×
  saturation cyclique sur 3 paliers (sépare encore les rares rangs aux teintes voisines).
  Luminosité de base fixe (marge pour la confiance). Voie **nominale**.
- `base_color_for_class(name)` : repli **sans registre** (hash du nom → teinte), stable mais
  sans garantie de distinction — utilisé seulement hors contexte profil.
- `apply_confidence(base_rgb, confidence)` : déclinaison de luminosité par tranche
  (5 paliers, **identique** à l'ancien `get_color_for_confidence`), prend une couleur RGB
  au lieu d'un index.
- Pur (`hashlib`, `colorsys`) → testable hors QGIS.

### 3.2 Registre `src/pipeline/cv/class_color_registry.py`

- `ClassColorRegistry(path)` : mappe `class_name → rang` attribué à la **première
  apparition** (append-only), persisté en JSON dans le **profil QGIS**
  (`qgisSettingsDirPath()/archeologia/class_color_registry.json`, comme `last_ui_config.json`)
  → stable dans le temps (un rang ne bouge jamais) et survit aux mises à jour du plugin.
- Écriture atomique (tmp + `os.replace`), tolérant à l'absence/corruption.
- Points d'accès partagés : `color_for_class(name)` / `rank_for_class(name)` via un
  singleton (override possible pour les tests). **Source unique** : génération et affichage
  l'appellent → cohérence garantie (singleton en mémoire + fichier partagé).
- Pur (I/O fichier, pas d'API QGIS) → utilisable depuis le worker comme depuis l'UI.

### 3.3 Source de vérité unique & rétrocompatibilité

Génération **et** affichage dérivent la couleur du `class_name` via le registre :

- **Génération** (`conversion_shp`) : `conf_color` via `rank_for_class(class_name)` —
  cascade `global_color_map`/`class_colors`/`class_id` supprimée. `conf_bin` (la tranche)
  inchangé.
- **Affichage** (`layer_loader`, `qgs_writer`, `build_detection_vector_layer`) : couleur via
  `color_for_class(class_name)` du nom de couche — `_resolve_color_idx` supprimé. Les **gpkg
  existants** s'affichent donc correctement sans régénération.

Code retiré (mort) : `get_color_for_confidence`, `_resolve_color_idx`,
`_build_global_class_color_map`, `_load_class_colors`, `load_class_colors_from_model`,
`_lighten_color`/`_darken_color`. **Conservé** : `BASE_COLOR_PALETTE` + `get_class_color`,
qui servent encore à colorier les **images annotées** (canal distinct de la symbologie QGIS,
hors scope).

## 4. Limite assumée

Le registre garantit l'équidistribution pour des **rangs consécutifs**. Sur un *sous-ensemble*
de rangs non consécutifs (run n'affichant que certaines classes), deux teintes peuvent
rester proches dans de rares cas (paires liées aux dénominateurs de Fibonacci) — nettement
plus rare que le hash pur. Choix produit : **stabilité** privilégiée. Un garde-fou de
**dé-collision par run** reste une évolution future possible, hors scope.

## 5. Tests (TDD)

Module pur → testable hors QGIS :

- **Déterminisme** : même nom → même couleur, à travers deux appels/process.
- **Stabilité** : ajouter une classe ne modifie la couleur d'aucune classe existante.
- **Distinction** : sur un jeu de N noms synthétiques, les teintes sont réparties
  (distance min raisonnable sur le cercle).
- **Non-régression** : la déclinaison `conf_bin`/luminosité produit les mêmes tranches
  qu'avant pour une couleur de base donnée.

## 6. Hors scope

- Dé-collision par run (cf. §4).
- Re-stylage des couches déjà chargées dans des projets QGIS ouverts avant la refonte.
- Choix d'un canal alternatif pour la confiance (opacité/contour) — la luminosité est conservée.
