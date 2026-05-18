# Tests manuels — Plugin ArchéologIA pipeline

Ces tests doivent être exécutés **dans QGIS avec le plugin chargé**.
Les tests automatisés (pytest) couvrent la logique métier hors QGIS :

```bash
python run_tests.py
```

## Pré-requis

- QGIS 3.34+ installé
- Plugin activé dans QGIS
- PDAL, GDAL (`gdalwarp`, `gdal_translate`, `gdaladdo`) dans le `PATH`
- Runner ONNX compilé dans `data/third_party/cv_runner_onnx/<os>/`
- Au moins un modèle ONNX dans `data/models/` avec `classes.txt`

---

## Priorités après le refactor (branche `refactor/architecture-improvements`)

Cette grille indique, pour chaque vague livrée, **quelles sections de tests sont prioritaires** pour valider la non-régression.

| Vague livrée | Quoi a changé | Sections critiques à tester |
|---|---|---|
| **V1.1 / V1.2** | `tile_splitter.py` supprimé, `IGN_TILE_SIZE_M` déplacé | §10 (`existing_mnt`), §11 (`existing_rvt`) — tous les régimes (small/standard/large) |
| **V2.1** (Detection dataclass) | Format des détections en mémoire unifié, JSON disque inchangé | §12 (Détection IA) sur **modèle bbox** + **modèle segmentation** |
| **V2.2** (ModelProfile) | Lecture unifiée de `args.yaml` et sidecar `.json` | §12 et §13 (clustering) — vérifier que les paramètres SAHI / clustering / post-process sont bien lus |
| **V2.4** (split runner.py) | Découpage en 4 modules (orchestration / cache / inférence / shapefiles) | §12 — runner externe + fallback Python |
| **V3.2** (CvPostService) | Logique CV post-loop dédupliquée | §8 (`ign_laz` avec CV), §10 (`existing_mnt` avec CV) |
| **V3.4** (helpers éclaté) | `log_section`, `safe_float` déplacés | §8.1, §10.1, §11.1 — vérifier que les bandeaux de section s'affichent correctement |
| **V3.5** (InputStrategy) | Acquisition LAZ refactorée | §8 (`ign_laz`) **et** §9 (`local_laz`) — surtout la **progression** de la barre |

**Tests à faire en priorité (P0)** : §8, §9, §10, §11 (les 4 modes), §12 (CV), §13 (clustering si modèle configuré).

**Tests qui n'ont pas été touchés par le refactor** (UI, validation, sauvegarde config) — toujours utile de revérifier mais moins critique : §1, §2, §3, §4, §5, §6, §7, §14, §15, §16, §17, §18.

---

## 1. Chargement du plugin

- [ ] **1.1 Démarrage QGIS**
  - Ouvrir QGIS
  - Vérifier qu'aucune erreur n'apparaît dans la console Python
  - Vérifier que le plugin apparaît dans le menu Extensions
- [ ] **1.2 Ouverture du plugin**
  - Cliquer sur le menu/icône du plugin
  - Vérifier que `NewMainDialog` s'ouvre sans erreur
  - Vérifier le titre « Archéolog'IA »
  - Vérifier la présence : combo Mode (Simple/Expert), sections config, zone de logs, boutons Lancer / Annuler / Nettoyer logs
  - Vérifier le message « Pipeline prêt à être utilisé » dans les logs
- [ ] **1.3 Fermeture et réouverture**
  - Fermer puis rouvrir le plugin
  - Vérifier que la configuration précédente est conservée (`config.json`)

---

## 2. Mode Simple / Expert

- [ ] **2.1 Mode Simple (par défaut)**
  - Vérifier que les sections suivantes sont **masquées** : Paramètres MNT, Paramètres RVT détaillés, Performance
- [ ] **2.2 Basculer en mode Expert**
  - Sélectionner « Expert » dans le combo Mode
  - Vérifier que les sections MNT, RVT, Performance deviennent visibles
  - Vérifier que les colonnes « Indice cible » et « Aire min » sont visibles
  - Vérifier que les seuils détection expert sont visibles (si détection activée)
- [ ] **2.3 Bascule Expert → Simple → Expert**
  - Modifier des paramètres expert (ex. résolution MNT = 1.0)
  - Passer en Simple puis revenir en Expert
  - Vérifier que les valeurs modifiées sont conservées

---

## 3. Sources de données

- [ ] **3.1 Changement de mode de données**
  - Sélectionner chaque mode (Téléchargement IGN, Nuages locaux, MNT déjà calculés, Indices existants)
  - Vérifier que le label de l'étape 2 change selon le mode
  - Vérifier que la description (texte italique) change
  - Vérifier que le placeholder du champ source change
- [ ] **3.2 Restauration de la source par mode**
  - En mode « Téléchargement IGN », renseigner un chemin `.shp`
  - Basculer en « Nuages locaux », renseigner un dossier
  - Revenir en « Téléchargement IGN »
  - Vérifier que le chemin `.shp` est restauré automatiquement
- [ ] **3.3 Sélection du dossier de sortie**
  - Cliquer sur « Parcourir » à côté du dossier de sortie
  - Sélectionner un dossier valide → le chemin s'affiche
- [ ] **3.4 Validation visuelle des chemins**
  - Renseigner un chemin invalide → fond rose sur le champ
  - Renseigner un chemin valide → fond normal
  - Laisser un champ obligatoire vide → bouton Lancer désactivé avec tooltip expliquant ce qui manque
- [ ] **3.5 Visibilité des produits selon le mode**
  - En mode « Indices existants » : la section Produits est masquée
  - En mode « MNT déjà calculés » : MNT et DENSITÉ sont masqués
  - En mode « Téléchargement IGN » : tous les produits sont visibles

---

## 4. Produits de visualisation

- [ ] **4.1 Cocher / décocher chaque produit**
  - MNT, DENSITÉ, M-HS, SVF, SLO, LD, SLRM, VAT
  - Vérifier que le bouton Lancer est désactivé si aucun produit coché (hors mode « Indices existants »)

---

## 5. Détection par IA (multi-modèles)

- [ ] **5.1 Activer / désactiver la détection**
  - Cocher « Activer la détection par intelligence artificielle »
  - Vérifier que le contenu détection apparaît (table, classes)
  - Vérifier le hint « La détection ne sera pas exécutée » quand désactivé
  - En mode Expert : vérifier que la section seuils apparaît aussi
- [ ] **5.2 Ajouter un modèle**
  - Cliquer « + Ajouter un modèle »
  - Vérifier qu'une ligne apparaît avec le combo des modèles disponibles
  - Vérifier la présence des boutons ℹ et × sur chaque ligne
- [ ] **5.3 Multi-modèles**
  - Ajouter 2 modèles différents
  - Vérifier que les classes sont affichées groupées par modèle (en-tête en gras `── modèle_name ──`)
  - Sélectionner / désélectionner des classes individuelles
  - Vérifier « Tout sélectionner » / « Tout désélectionner »
- [ ] **5.4 Actualiser les modèles**
  - Cliquer « Actualiser »
  - Vérifier que les combos modèle sont mis à jour
- [ ] **5.5 Supprimer un modèle**
  - Cliquer × sur une ligne → la ligne disparaît
  - Vérifier que les classes sont mises à jour
- [ ] **5.6 Info modèle (bouton ℹ)**
  - Cliquer ℹ sur un modèle
  - Vérifier la boîte de dialogue avec les paramètres d'entraînement
  - Vérifier le bouton « Ouvrir le dossier »
- [ ] **5.7 Colonnes expert** (mode Expert uniquement)
  - Vérifier la colonne « Indice cible » (combo RVT)
  - Vérifier la colonne « Aire min (m²) » (spinbox)
  - Modifier les valeurs et vérifier la sauvegarde
- [ ] **5.8 Seuils expert** (mode Expert uniquement)
  - Modifier le seuil de confiance
  - Vérifier le label dynamique (« Très permissif » → « Très strict ») avec la couleur correspondante
  - Modifier le seuil IoU

---

## 6. Sauvegarde / chargement config

- [ ] **6.1 Autosave automatique**
  - Modifier n'importe quel paramètre (ex. cocher un produit)
  - Fermer puis rouvrir le plugin
  - Vérifier que le changement est conservé (`config.json` mis à jour)
- [ ] **6.2 Exporter configuration** (Sauvegarder config)
  - Cliquer « Sauvegarder config »
  - Choisir un emplacement et nom de fichier `.json`
  - Vérifier que le fichier est créé et contient les paramètres
- [ ] **6.3 Importer configuration** (Charger config)
  - Modifier les paramètres
  - Cliquer « Charger config » et sélectionner le fichier exporté
  - Vérifier que les paramètres sont restaurés
- [ ] **6.4 Reset Paramètres MNT** (mode Expert)
  - Modifier les résolutions et l'expression de filtrage
  - Cliquer « Remettre par défaut » dans la section MNT
  - Vérifier : résolution MNT = 0.5, densité = 1.0, filtre par défaut
- [ ] **6.5 Reset Paramètres RVT** (mode Expert)
  - Modifier des paramètres RVT
  - Cliquer « Remettre par défaut » dans la section RVT
  - Vérifier les valeurs par défaut (MDH : 16 directions, 35° élévation, etc.)
- [ ] **6.6 Reset Performance** (mode Expert)
  - Modifier max workers
  - Cliquer « Remettre par défaut » dans la section Performance
  - Vérifier : max workers = 4

---

## 7. Preflight (`run_preflight`)

- [ ] **7.1** Dépendances OK → preflight passe sans erreur
- [ ] **7.2** PDAL manquant → erreur affichée dans les logs (mode `ign_laz`)
- [ ] **7.3** Chemin d'entrée invalide → erreur « introuvable » dans les logs

---

## 8. Mode `ign_laz` ⭐ P0

> **Pré-requis** : Fichier polygone (shapefile) délimitant la zone d'étude.

- [ ] **8.1 Lancement avec fichier valide + MNT + LD**
  - Vérifier la boîte de confirmation (récapitulatif)
  - Vérifier la progression (StructuredLogger : bandeaux de section)
  - Vérifier les fichiers générés dans le dossier de sortie
  - **V3.5** : vérifier la barre de progression — phases 0-25 % téléchargement, 25-35 % fusion, 35-95 % produits
- [ ] **8.2 Annulation en cours** → arrêt propre (CancelToken)
- [ ] **8.3 Tous les produits activés** (y compris SLRM) → tous générés

---

## 9. Mode `local_laz` ⭐ P0

> **Pré-requis** : Dossier contenant des fichiers LAZ/LAS.

- [ ] **9.1** Dossier valide + MNT + LD → fichiers générés
  - **V3.5** : vérifier la progression — pas de phase téléchargement, échelle 0-100 % sur les produits
- [ ] **9.2** Dossier vide ou inexistant → erreur preflight

---

## 10. Mode `existing_mnt` ⭐ P0

> **Pré-requis** : Dossier contenant des MNT (GeoTIFF).
>
> **V1** : tester les 3 régimes — **standard** (~1×1 km IGN-aligné), **small** (<1 km ou non aligné), **large** (>1.05 km dans une dimension).

- [ ] **10.1 MNT valide + LD + SVF → produits RVT générés**
  - Régime standard
  - Régime small
  - Régime large
- [ ] **10.2 Avec détection activée** → inférence après les produits
  - **V3.2** : vérifier que la section « COMPUTER VISION » apparaît dans les logs
  - **V3.2** : vérifier la progression à 80 %

---

## 11. Mode `existing_rvt` ⭐ P0

> **Pré-requis** : Dossier contenant des RVT (GeoTIFF).
>
> **V1** : tester les 3 régimes (idem §10).

- [ ] **11.1** RVT valide → copie TIF, conversion JPG+JGW, nettoyage orphelins
- [ ] **11.2 Avec détection activée** → inférence ONNX sur les JPG, shapefiles
  - Régime large : SAHI doit slicer en mémoire (pas de pré-découpage)

---

## 12. Détection IA — Exécution ⭐ P0

> **Pré-requis** :
> - Runner ONNX : `data/third_party/cv_runner_onnx/<os>/cv_runner_onnx[.exe]`
> - Modèle ONNX dans `data/models/` avec `classes.txt`

- [ ] **12.1 Runner externe trouvé**
  - Vérifier dans les logs : « Runner externe trouvé »
  - Vérifier les labels (`.txt`) dans le dossier de sortie
  - Vérifier les images annotées (si option activée)
  - **V2.1** : ouvrir un `.json` produit, vérifier la structure inchangée (`bbox_absolute` ou `polygon`, `confidence`, `class_id`)
- [ ] **12.2 Runner externe absent → fallback Python ONNX**
  - Renommer temporairement le binaire
  - Vérifier dans les logs : « fallback interne ONNX »
  - Vérifier que l'inférence fonctionne quand même
  - **V2.4** : vérifier que la délégation `runner.py → runner_inference.py` fonctionne
- [ ] **12.3 Génération shapefiles**
  - Activer « Générer des shapefiles » (mode Expert)
  - Vérifier les `.shp` (ou `.gpkg`) dans le dossier de sortie
  - Vérifier le chargement automatique dans le projet QGIS
- [ ] **12.4 Filtrage de classes**
  - Désélectionner certaines classes dans la liste
  - Lancer le pipeline
  - Vérifier que seules les classes cochées sont détectées
- [ ] **12.5 Détection désactivée** → aucune détection effectuée
- [ ] **12.6 Détection activée sans classes** → avertissement
  - Activer la détection, désélectionner toutes les classes
  - Cliquer Lancer → message d'avertissement
- [ ] **12.7 Tester un modèle bbox** (détection)
  - **V2.1** : vérifier que les bbox sont correctement géoréférencées
- [ ] **12.8 Tester un modèle segmentation** (avec polygones)
  - **V2.1** : vérifier que les polygones sont fidèles aux masques
  - **V2.1** : si modèle RF-DETR Seg avec trous, vérifier que `polygon_holes` est bien dans le `.json` et que les trous apparaissent dans le shapefile

---

## 13. Clustering ⭐ P0 (si modèle configuré)

> **Pré-requis** : Modèle avec `clustering` configuré dans `args.yaml`.

- [ ] **13.1 Zones cluster générées**
  - Vérifier les shapefiles cluster (attribut `nb_detect`)
  - Vérifier que les zones cluster sont chargées dans QGIS
  - Vérifier le style hachures croisées sur les couches cluster
  - **V2.2** : vérifier que `min_confidence`, `eps_m`, `min_cluster_size` du `args.yaml` sont bien appliqués

---

## 14. Finalisation

- [ ] **14.1** VRT générés pour `tif/`, `jpg/`, `annotated_images/`
- [ ] **14.2 Shapefiles collectés et chargés dans QGIS**
  - Vérifier le style par confiance (couleurs par bin)
- [ ] **14.3 Zoom automatique sur l'étendue des résultats**
  - [ ] **14.3.a** Projet QGIS neuf, CRS par défaut (EPSG:4326) : à la fin du pipeline, le canvas doit se centrer sur les couches Lambert-93 chargées (transformation CRS appliquée). Log attendu : `Zoom sur l'étendue des résultats`.
  - [ ] **14.3.b** Projet QGIS configuré en EPSG:2154 : zoom direct, sans transformation, doit se centrer correctement.
  - [ ] **14.3.c** Re-run avec la **même** sortie (couches déjà présentes dans le projet) : les logs indiquent « Couche … déjà présente », et le zoom doit **quand même** se déclencher sur l'étendue cumulée.
- [ ] **14.4** Logs de fin de pipeline (`StructuredLogger.end_pipeline`)

---

## 15. Interface — logs et boutons

- [ ] **15.1 Zone de logs**
  - Vérifier le splitter redimensionnable (config en haut, logs en bas)
  - Vérifier le scroll automatique vers le bas
  - Vérifier le bouton « Nettoyer logs »
- [ ] **15.2 Barre de progression**
  - Vérifier que la barre progresse pendant l'exécution
  - Vérifier le label d'étape à côté de la barre
- [ ] **15.3 Bouton Lancer** (vert stylisé)
  - Grisé si config incomplète, vert si config valide
  - Tooltip explicatif quand désactivé
  - Boîte de confirmation avant lancement
- [ ] **15.4 Bouton Annuler**
  - Désactivé quand le pipeline ne tourne pas
  - Activé pendant l'exécution → annulation propre
- [ ] **15.5 Désactivation pendant l'exécution**
  - Vérifier que toute la zone config est grisée pendant le run
  - Vérifier que les boutons Charger / Sauvegarder sont désactivés

---

## 16. Labels dynamiques (mode Expert)

- [ ] **16.1 Label confiance**
  - 0.1 → « Très permissif » (rouge)
  - 0.3 → « Détection large » (orange)
  - 0.5 → « Équilibré » (vert)
  - 0.7 → « Sélectif » (bleu)
  - 0.9 → « Très strict » (violet)
- [ ] **16.2 Hint RAM workers**
  - 2 workers → « ≈ 4 Go RAM disponible nécessaire »
  - 4 workers → « ≈ 8 Go RAM disponible nécessaire »

---

## 17. Robustesse

- [ ] **17.1** Dossier de sortie en lecture seule → erreur appropriée
- [ ] **17.2** Fichier LAZ/MNT corrompu → erreur gérée proprement
- [ ] **17.3** Annulation pendant le téléchargement IGN → arrêt propre

---

## 18. Logs fichier

- [ ] **18.1** Fichier `pipeline_log_*.txt` créé dans le dossier de sortie
- [ ] **18.2** Contenu du fichier cohérent avec les logs affichés dans l'UI

---

## Résumé

- **Total tests** : 60+
- **Tests passés** : ☐ / 60
- **Tests échoués** : ☐ / 60

| Champ | Valeur |
|---|---|
| Date | __________________ |
| Testeur | __________________ |
| Version QGIS | __________________ |
| Version plugin | __________________ |
| Branche / commit | __________________ |
