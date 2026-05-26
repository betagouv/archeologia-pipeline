# Cible dérivée « Zones d'extraction de matériaux » — design

**Date :** 2026-05-25
**Statut :** validé en brainstorming, prêt pour planification

## Problème

L'utilisateur détecte des **zones d'extraction de matériaux** en utilisant un
modèle de cratères circulaires (Verdun) avec l'option de clustering : le modèle
détecte des dépressions circulaires, le clustering DBSCAN les regroupe en
polygones (`zone_crateres`), que l'utilisateur interprète comme des zones
d'extraction.

Dans l'UI (étape 3), ce cas d'usage est invisible et incompréhensible :

1. **Découvrabilité** — la case « Regrouper en clusters » n'apparaît qu'après
   avoir coché l'entité « Trous d'obus », seulement si le modèle le supporte, et
   son libellé ne dit ni ce qu'on obtient ni à quoi ça sert.
2. **Vocabulaire** — le besoin « zones d'extraction de matériaux » n'est nommé
   nulle part ; l'utilisateur détourne « Trous d'obus » sans qu'aucun élément de
   l'UI ne le suggère.

Les deux frictions comptent à parts égales.

## Décision de design

- **Direction retenue : nommer par le cas d'usage.** Créer une cible
  cochable « Zones d'extraction de matériaux » que l'utilisateur sélectionne
  directement, clustering activé d'office. Résout les deux frictions :
  découvrable (case comme une autre) **et** nommée par le besoin.
- **Garde-fou honnêteté.** Le **libellé** nomme l'usage, mais la **description**
  reste honnête sur la méthode (regroupement de dépressions circulaires, modèle
  entraîné sur cratères). Un **badge** indique l'agrégation. L'utilisateur sait
  qu'il regroupe une *forme* détectée, pas une *fonction* certifiée.
- **Sortie : zones + dépressions individuelles** (polygones `zone_crateres` ET
  points `cratere_obus`), pour voir la densité et vérifier la détection.
- **Câblage dans `model_card.yaml`** (la couverture y reste centralisée, cf.
  CLAUDE.md). `args.yaml` n'est pas modifié.

## Concept central : la « cible dérivée »

Une **cible dérivée** = une sortie de clustering d'un modèle, présentée comme
une entité cochable à part entière. C'est le seul concept nouveau. Elle est
repliée dans la `coverage` existante de l'orchestrateur, ce qui rend la
résolution des runs inchangée.

## Changements détaillés

### 1. Catalogue d'entités — `data/entities_catalog.json`

Nouvelle entité :

```json
{
  "id": "zones_extraction_materiaux",
  "label": "Zones d'extraction de matériaux",
  "description": "Regroupe en zones les dépressions circulaires détectées (extraction de matériaux probable). Détection issue d'un modèle de cratères.",
  "display_order": 95
}
```

`display_order: 95` la place juste après « Dépressions circulaires » (90).

### 2. Model cards — section `derived_targets`

Ajoutée à **`cratere_circulaire_2`** et **`verdun_3_classes_1`** (les deux
produisent `zone_crateres`) :

```yaml
derived_targets:
  - output_class: zone_crateres          # défini dans args.yaml:clustering
    entity: zones_extraction_materiaux
    include_source: true                 # sortie = zones + dépressions individuelles
```

Le lien se fait par `output_class` ; les `target_classes` et paramètres DBSCAN
restent dans `args.yaml`, inchangés. Défaut de l'entité = `cratere_circulaire_2`
(mono-classe, le plus spécialisé via `_pick_default_model`) ; `verdun_3_classes_1`
(SVF) reste accessible via « Changer ▾ ».

### 3. Orchestrateur — `src/app/services/model_orchestrator.py`

1. **`_load_derived_targets(card)`** : lit la section `derived_targets` →
   `[(output_class, entity_id, include_source)]`. Tolérant (absent → `[]`).
2. **Résolution au chargement** (dans `discover_installed_models`) : pour chaque
   `derived_target`, joindre par `output_class` avec la règle de clustering déjà
   lue par `_load_args_clustering` pour obtenir les classes de la cible :
   - `include_source: true` → `(cratere_obus, zone_crateres)`
   - `include_source: false` → `(zone_crateres,)`
   - aucune règle `args.yaml` correspondante → ignoré + warning (anti-drift).
3. **`InstalledModel`** gagne `derived_entities: FrozenSet[str]`, et les entités
   dérivées sont **injectées dans `coverage`** avec leurs classes résolues.
   Conséquences automatiques, **sans modifier `resolve_runs_from_entities`** :
   - `build_entity_coverage` voit le modèle comme candidat de la nouvelle entité ;
   - le run inclut `zone_crateres` (+`cratere_obus`) dans `selected_classes` ;
   - `runner_shapefiles` voit `zone_crateres` ∈ `selected_classes` → clustering
     déclenché ;
   - « Trous d'obus » + « Zones d'extraction » sur le même modèle → un seul run
     (union des classes), pas de double inférence.
4. **Garde-fou** : `_build_cluster_options` **exclut** les `derived_entities`
   (sinon la cible afficherait une case « Regrouper en clusters » redondante).

Le pipeline `src/pipeline/cv/` n'est **pas touché** : tout passe par le mécanisme
existant « clustering actif si `output_class` ∈ `selected_classes` ».

### 4. UI étape 3 — `src/ui/steps/step_3_detection.py` + `src/ui/widgets/entity_card.py`

La nouvelle entité apparaît automatiquement. Ajustements pour les entités
**dérivées** uniquement :
- **pas de case « Regrouper en clusters »** (regroupement intrinsèque) — découle
  du garde-fou orchestrateur (point 4) ;
- **badge** `↳ regroupement automatique en zones` à la place ;
- `EntityCard` reçoit un flag `is_derived` pour choisir badge vs case.

Panneau « Runs IA programmés » et persistance config (`selected_entities`) :
**inchangés**.

## Tests

**Unitaires** — `tests/unit/test_model_orchestrator.py` (pur-Python, hors QGIS) :
- parsing `derived_targets` ; jointure `args.yaml` (`include_source` true/false →
  bonnes classes) ;
- `derived_target` sans règle clustering correspondante → ignoré + warning ;
- `build_entity_coverage` : bons modèles candidats, défaut = `cratere_circulaire_2` ;
- `resolve_runs_from_entities` : entité dérivée seule → run
  `{model, rvt: LD, selected_classes: [cratere_obus, zone_crateres]}` ;
- entité dérivée **exclue** de `cluster_options` ;
- fusion « Trous d'obus » + « Zones d'extraction » → un seul run.

**UI** (`src/ui/` exclu de pytest) → checklist manuelle QGIS
(`tests/TESTS_MANUELS_QGIS.txt`), +1 ligne.

## Hors périmètre (YAGNI) / limites connues

- Pas de réglage de clustering spécifique « extraction » : réutilisation des
  params DBSCAN du modèle de cratères (eps 40 m pour `cratere_circulaire_2`).
  Raffinement futur si la morphologie diverge trop des champs de cratères.
- Comportement existant « Trous d'obus » + case cluster : **inchangé**.
- Aucune modification du pipeline CV ni du format `config.json`.

## Récapitulatif des fichiers touchés

| Fichier | Changement |
|---|---|
| `data/entities_catalog.json` | +1 entité `zones_extraction_materiaux` |
| `data/models/cratere_circulaire_2/model_card.yaml` | +section `derived_targets` |
| `data/models/verdun_3_classes_1/model_card.yaml` | +section `derived_targets` |
| `src/app/services/model_orchestrator.py` | `_load_derived_targets`, résolution, `derived_entities`, garde-fou `_build_cluster_options` |
| `src/ui/steps/step_3_detection.py` | flag `is_derived` transmis aux cartes |
| `src/ui/widgets/entity_card.py` | badge vs case selon `is_derived` |
| `tests/unit/test_model_orchestrator.py` | tests cibles dérivées |
| `tests/TESTS_MANUELS_QGIS.txt` | +1 ligne de vérification UI |
