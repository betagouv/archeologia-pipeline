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
- [ ] **1.2 Ouverture du plugin (assistant 4 étapes)**
  - Cliquer sur le menu/icône du plugin
  - Vérifier que l'assistant (`WizardDialog`) s'ouvre sans erreur
  - Vérifier le titre « Archéolog'IA » + numéro de version
  - Vérifier la présence : rail latéral 4 étapes, boutons Précédent / Suivant, en-tête avec « Charger une config » / « Enregistrer la config »
- [ ] **1.3 Fermeture et réouverture**
  - Fermer puis rouvrir le plugin
  - Vérifier que la configuration précédente est conservée (`last_ui_config.json`, dossier de profil QGIS)

---

## 2. Navigation de l'assistant

- [ ] **2.1 Rail latéral**
  - Vérifier que l'étape courante est mise en évidence dans le rail
  - Vérifier que les sous-libellés du rail reflètent les choix (mode, produits, entités)
- [ ] **2.2 Précédent / Suivant**
  - Parcourir les 4 étapes dans les deux sens
  - Vérifier que les saisies sont conservées en revenant en arrière
- [ ] **2.3 Erreurs bloquantes**
  - Laisser un champ obligatoire vide → le rail signale l'étape en erreur et le bouton Lancer (étape 4) est désactivé avec une explication

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
  - MNT, DENSITÉ, HS, M-HS, SVF, SLO, LD, SLRM, VAT
  - Vérifier que le bouton Lancer est désactivé si aucun produit coché (hors mode « Indices existants »)
  - Vérifier que HS est sélectionnable **seul** comme indice de visualisation

---

## 5. Détection par IA (étape 3 — sélection par entités)

- [ ] **5.1 Cocher une entité**
  - À l'étape 3, cocher une entité (carte) dans son panneau morphologique
  - Vérifier qu'un run apparaît dans « Runs IA programmés » (modèle, indice cible)
- [ ] **5.2 Info modèle**
  - Cliquer le bouton ⓘ d'un run programmé
  - Vérifier que le dialogue d'information du modèle s'ouvre (classes, indice, paramètres)
- [ ] **5.3 Multi-entités**
  - Cocher plusieurs entités couvertes par des modèles/indices différents
  - Vérifier que les runs sont regroupés par couple (modèle, indice cible)
- [ ] **5.4 Seuil par entité**
  - Ouvrir les réglages avancés d'une entité et modifier son seuil de confiance
  - Vérifier que le seuil modifié est rappelé à la réouverture
- [ ] **5.5 Décocher une entité**
  - Décocher l'entité → le run correspondant disparaît de « Runs IA programmés »
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
- [ ] **13.2 Cible dérivée « Regroupement de cratères »** (UI étape 3, sélection par entités)
  - À l'étape 3, cocher l'entité **« Regroupement de cratères »**
  - Vérifier qu'un badge **« ↳ regroupement automatique en zones »** s'affiche (et **non** une case « Regrouper en clusters »)
  - Panneau « Runs IA programmés » : **1 run** `Modèle Cratères d'obus` · `🔗 LD` · classes `cratere, zone_crateres`
  - Via « Changer ▾ », basculer sur `verdun_3_classes_1` → le run passe sur `🔗 SVF`
  - Lancer sur un RVT LD existant → GeoPackage avec polygones `zone_crateres` (zones) **ET** points/masques `cratere` (dépressions individuelles)
  - Cocher en plus « Cratères » (même modèle par défaut) → toujours **un seul run** (fusion, pas de double inférence)

---

## 14. Finalisation

- [ ] **14.1** VRT générés dans chaque `indices/<PRODUIT>/tif/`
  - [ ] **14.1.a Nom distinctif** : le fichier s'appelle `index_<PRODUIT>.vrt` (ex. `index_MNT.vrt`, `index_CVAT.vrt`, `index_SLO_U0_V1.vrt`) — **pas** le générique `index.vrt` — et correspond au nom de la couche dans QGIS (`index_<PRODUIT>`). Au chargement manuel d'un `.vrt` depuis l'explorateur, le nom de couche est donc immédiatement identifiable.
- [ ] **14.2 Shapefiles collectés et chargés dans QGIS**
  - Vérifier le style par confiance (couleurs par bin)
- [ ] **14.3 Zoom automatique sur l'étendue des résultats**
  - [ ] **14.3.a** Projet QGIS neuf, CRS par défaut (EPSG:4326) : à la fin du pipeline, le canvas doit se centrer sur les couches Lambert-93 chargées (transformation CRS appliquée). Log attendu : `Zoom sur l'étendue des résultats`.
  - [ ] **14.3.b** Projet QGIS configuré en EPSG:2154 : zoom direct, sans transformation, doit se centrer correctement.
  - [ ] **14.3.c** Re-run avec la **même** sortie (couches déjà présentes dans le projet) : les logs indiquent « Couche … déjà présente », et le zoom doit **quand même** se déclencher sur l'étendue cumulée.
- [ ] **14.4** Logs de fin de pipeline (`StructuredLogger.end_pipeline`)
- [ ] **14.5 CRS des sorties = EPSG:2154 (pas de « unnamed »)**
  - Après un run `local_laz`/`ign_laz` sur dalle(s) LiDAR HD : `gdalsrsinfo -o epsg <output>/indices/MNT/tif/index_MNT.vrt` → **EPSG:2154** (et non `EPSG:-1`)
  - Vérifier qu'aucun message **« Pas de transformation disponible entre unnamed et EPSG:2154 / Point outside of projection domain »** n'apparaît au chargement/zoom
  - Cas où le MNT sortait en CRS local : log attendu `CRS absent/local sur le MNT → affecté EPSG:2154` (côté pipeline) ou `CRS absent/local sur « … » → affecté EPSG:2154` (garde-fou chargement)
  - Les couches chargées s'affichent bien en Lambert-93 et le zoom se centre sur la zone (cf. §14.3)

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

## 19. Consultation lecture-seule pendant un run ⭐ P0

- [ ] **19.1** Lancer un run ; pendant l'exécution, cliquer **étape 1** dans le rail → page Source affichée, bandeau « 🔒 Lecture seule — run en cours » visible ; chemins, boutons « Parcourir… »/« Couche QGIS » et frise de mode **inactifs** mais valeurs lisibles
- [ ] **19.2** **Étape 2** → « Réglages avancés… » fonctionne, bascule entre les onglets RVT fonctionne ; tous les spinbox/checkbox/combos grisés-inactifs ; « ↺ Réinitialiser » inactif ; valeurs lancées affichées
- [ ] **19.3** **Étape 3** → interrupteur, cartes d'entités, combos « Changer ▾ », case « Regrouper », case « Générer images annotées » **inactifs** ; chips de filtre morphologique + scroll **utilisables** ; cocher « Réglages avancés » révèle les seuils par entité (en lecture seule)
- [ ] **19.4** Cliquer **étape 4** dans le rail (1 clic) → retour au **RunView en direct** (pas le récap), la progression continue ; bandeau lecture-seule disparu
- [ ] **19.5** Enchaîner « Suivant » de l'étape 1 jusqu'à l'étape 4 pendant le run → autorisé ; sur l'étape 4 le bouton « ▶ Lancer le pipeline » reste **désactivé** (pas de relance)
- [ ] **19.6** Fin / annulation du run → les 3 étapes redeviennent **éditables**, bandeau disparu, autosave + validation refonctionnent, l'étape 4 réaffiche le récap
- [ ] **19.7** Mode `existing_rvt` : étape 2 « sans objet » (sections masquées) reste masquée en lecture-seule ; détection désactivée → empty-state non activable pendant le run

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

---

## 20. Produit Couverture (QA points sol)

- [ ] **20.1 Étape 2 — carte et réglage**
  - Mode IGN/LAZ local : carte « Couverture · QA points sol » visible à côté de « Densité », décochée par défaut
  - Mode MNT existant : la carte « Modèle de base » (et donc Couverture) reste masquée
  - Réglages avancés > onglet MNT : « Seuil zones mal couvertes (%) » (défaut 30, bornes 5–95), persisté entre sessions
- [ ] **20.2 Run avec Couverture seule (sans Densité)**
  - La passe densité tourne quand même (sert de source au calcul)
  - `indices/COUVERTURE/tif/*.tif` créés, **pas** de dossier `indices/DENSITE/`
- [ ] **20.3 Run avec Couverture + Densité**
  - La passe densité ne tourne qu'**une** fois par dalle (vérifier le journal)
  - Les deux dossiers `indices/DENSITE/` et `indices/COUVERTURE/` existent
- [ ] **20.4 Polygones et couches QGIS**
  - `indices/COUVERTURE/zones_mal_couvertes.gpkg` créé s'il existe des zones sous le seuil
  - Couche « Zones mal couvertes (<30 %) » hachurée rouge **tout en haut** de l'arbre
  - Raster `index_COUVERTURE` en pseudo-couleur : 0 % rouge → seuil orange → 100 % transparent
- [ ] **20.5 Projet de validation**
  - Rouvrir `detections/detections_validation.qgs` : mêmes couches, mêmes styles (vecteur QA + raster COUVERTURE)
- [ ] **20.6 Mode MNT existant avec config résiduelle**
  - Éditer `last_ui_config.json` pour forcer `COUVERTURE: true` puis lancer en mode MNT existant
  - Warning « Produit Couverture indisponible… » dans le journal, pas d'erreur
- [ ] **20.7 Cohérence terrain**
  - Sur une zone partiellement boisée : les polygones correspondent aux masses boisées/plans d'eau (recouper visuellement avec `index_DENSITE`)

---

## 21. Sélection des dalles IGN sur la carte (mode `ign_laz`) ⭐ P0

> **Pré-requis** : `data/quadrillage_france/TA_diff_pkk_lidarhd_classe.shp` + `.qix`
> (générer le `.qix` via `python dev/build_quadrillage_index.py`). Avoir un fond de
> carte chargé dans QGIS et être zoomé sur une zone couverte par le LiDAR HD.

- [ ] **21.1 Visibilité du bouton** : « Sélectionner les dalles » visible **uniquement** en mode `ign_laz` (masqué dans les autres modes).
- [ ] **21.2 Activation** : clic → la couche « Quadrillage IGN LiDAR HD » apparaît (contour seul, intérieur transparent) dans un groupe « Quadrillage IGN » ; **seul le dialogue du plugin se minimise** (la fenêtre QGIS reste affichée, le canevas visible) ; une barre de message QGIS « Valider (0 dalle) / Tout effacer / Annuler » s'affiche.
- [ ] **21.3 Orientation + garde (U1)** : depuis une vue monde/ailleurs, le clic sur le bouton **zoome automatiquement à une échelle où la grille rend**, centré sur la zone visée (bornée à la France) ; tant que la vue est trop dézoomée (grille masquée), un clic/encadré **ne sélectionne rien** et affiche « Zoomez davantage… » ; après zoom, la sélection fonctionne. Dézoomé sur toute la France, la grille n'est pas dessinée (pas de figeage) ; en zoomant elle apparaît et reste fluide (index `.qix`).
- [ ] **21.4 Clic toggle + estimation (U2)** : cliquer une dalle → surlignée ; recliquer → désélectionnée ; le bouton affiche « Valider (N dalles ≈ X–Y Go/Mo) ».
- [ ] **21.5 Glisser-boîte (ajout)** : tracer un rectangle → toutes les dalles intersectées s'ajoutent ; rectangle **orange** ; compteur à jour.
- [ ] **21.5 bis Glisser-boîte + Ctrl (retrait)** : **Ctrl** enfoncé → rectangle **gris** ; encadrer des dalles sélectionnées → elles sont **retirées** ; compteur à jour.
- [ ] **21.5 ter Tout effacer** : le bouton « Tout effacer » remet la sélection à 0 (compteur « Valider (0 dalle) »).
- [ ] **21.6 CRS** : projet en EPSG:3857 (fond OSM/Google) → les clics tombent bien sur la dalle visée (transformation canevas→couche OK).
- [ ] **21.7 Échap / Annuler** : la sélection est abandonnée, la couche + le groupe retirés, la barre de message disparaît, l'outil-carte précédent est restauré, le champ source **inchangé**, le dialogue revient au premier plan.
- [ ] **21.8 Valider sans sélection** → message « Aucune dalle », on reste en mode sélection.
- [ ] **21.9 Valider 2–3 dalles** → `data/temp_zones/dalles_selection.txt` créé (lignes `nom_pkk,url_telech`, en-tête `#`) ; message vert « N dalle(s) enregistrée(s) » ; le champ source pointe ce fichier (bordure « ok ») ; couche retirée ; barre de message disparue ; dialogue ramené au premier plan.
- [ ] **21.9 bis Grosse sélection (U2)** : sélectionner > 50 dalles puis Valider → boîte de confirmation « Téléchargement volumineux… » ; « Non » garde la sélection ; « Oui » écrit le fichier.
- [ ] **21.10 Bout-en-bout** : renseigner le dossier de sortie puis lancer → le téléchargement démarre **directement** (pas d'étape « résolution des dalles » : `IgnDownloadStrategy` voit un `.txt`).
- [ ] **21.11 Réutilisation** : relancer la sélection sans fermer → pas de double-chargement de la couche.
- [ ] **21.12 Lecture seule pendant un run** : lancer un run, revenir à l'étape 1 → bouton désactivé ; si une sélection était active au lancement, elle est refermée proprement.
- [ ] **21.13 Fermeture / rechargement pendant la sélection** : fermer le dialogue (croix) ou recharger le plugin pendant une sélection active → pas de crash, pas d'outil-carte ni de couche orphelins.
- [ ] **21.14 Quadrillage absent** : renommer le `.shp` → clic sur le bouton affiche un message clair, l'outil ne s'active pas.
- [ ] **21.15 Persistance de la liste (reprise après interruption)** : lancer un run `ign_laz` à partir d'une **sélection sur carte**. Dès le début du téléchargement, `<output_dir>/dalles_urls.txt` est créé (lignes `nom,url`, en-tête `#`) et le journal indique « Liste des dalles enregistrée: dalles_urls.txt ». **Interrompre** le run après quelques dalles → le fichier est toujours présent et complet (toutes les dalles sélectionnées, pas seulement celles téléchargées).
- [ ] **21.16 Reprise** : relancer en mode `ign_laz` avec ce `dalles_urls.txt` comme source et le **même** `output_dir` → les dalles déjà présentes dans `sources/dalles/` sont **sautées** (« déjà téléchargé »), seules les manquantes sont récupérées.

> **Consolidation affichage (anti-« grille absente, réparée par redémarrage »)** — ⭐ P0

- [ ] **21.17 Rendu fiable à l'activation (régression bug principal)** : zoomé sur une ville (échelle ≤ 1:1 500 000), cliquer « Sélectionner les dalles » → le quadrillage orange **apparaît immédiatement** (pas besoin de déplacer/zoomer). Répéter activer → Annuler **5 fois de suite** → la grille s'affiche **à chaque fois** (plus d'intermittence). Le journal Python QGIS montre `Quadrillage : chargement frais …` ou `… → reuse/readd`.
- [ ] **21.18 Réutilisation avec nœud retiré de l'arbre** : pendant une sélection active, supprimer **manuellement** la couche « Quadrillage IGN LiDAR HD » depuis le panneau Couches, puis relancer la sélection → la grille **réapparaît** (ré-ajoutée au groupe, journal `… in_tree=False → readd`), pas de couche orpheline invisible.
- [ ] **21.19 Recadrage métropole (remplace l'ancien U1)** : depuis une vue trop dézoomée (monde / hors France), cliquer le bouton → la vue **se recadre sur la France métropolitaine** (Brest→Strasbourg, Lille→Perpignan). À cette échelle la grille reste masquée (le bandeau 21.20 invite à zoomer). **Si la vue est déjà zoomée** sur une zone (grille visible), cliquer le bouton **ne change pas la vue** (on ne perturbe pas une vue de travail).
- [ ] **21.20 Bandeau persistant + suivi d'échelle** : trop dézoomé → un bandeau **persistant** « Zoomez pour afficher les dalles — échelle 1:N (requise ≤ 1:1 500 000) » reste affiché et **le N se met à jour** en zoomant ; il **disparaît** dès que la grille devient visible ; il **réapparaît** en dézoomant à nouveau ; il est **absent** après Annuler/Valider (pas de signal `scaleChanged` fuité — vérifier qu'aucun bandeau ne ré-apparaît en zoomant après la fin de la sélection).
- [ ] **21.21 Échec d'enregistrement (robustesse ré-entrée)** : rendre `data/temp_zones` non inscriptible (ou la verrouiller), sélectionner des dalles puis Valider → message « Échec de l'enregistrement », la **sélection est conservée**, « Annuler » fonctionne, et **après**, « Sélectionner les dalles » est de nouveau **cliquable** (le bouton ne reste jamais grisé bloqué).
- [ ] **21.22 Auto-réparation d'état périmé** : changer l'outil-carte QGIS (p. ex. « Identifier les entités ») **pendant** une sélection active, puis recliquer « Sélectionner les dalles » → l'état périmé est nettoyé (journal `… état périmé … → nettoyage`) puis la sélection se ré-active proprement, **sans double barre de message**.

---

## 22. Re-run dans le même dossier de sortie (ajout de dalle, VRT régénéré) ⭐ P0

> **Contexte** : un re-run dans le **même** `output_dir` régénérait bien les `index_<PRODUIT>.vrt`
> sur disque, mais comme les couches du run précédent restaient chargées, QGIS
> réécrivait sa version périmée par-dessus → les dalles ajoutées étaient invisibles.
> Parade : purge des couches du dossier au **lancement** du run (thread principal,
> avant la régénération) + relecture défensive d'une couche réutilisée.

- [ ] **22.1 Premier run** : lancer un run `ign_laz`/`local_laz` sur ≥ 2 dalles contiguës dans un `output_dir` neuf → couches `index_*` chargées, mosaïque complète. **Ne pas fermer QGIS.**
- [ ] **22.2 Re-run avec dalle ajoutée** : relancer dans **le même** `output_dir` en ajoutant 1 dalle (distante de préférence). Au lancement, le journal indique « N couche(s) périmée(s) … retirée(s) avant régénération ». À la fin, les couches `index_*` affichent **toutes** les dalles (ancienne(s) + nouvelle).
- [ ] **22.3 Vérif disque** : dans `indices/<PRODUIT>/tif/index_<PRODUIT>.vrt`, le nombre de `<SourceFilename>` = nombre de TIF présents (toutes dalles incluses), **sans** balise `<OverviewList resampling="nearest">` ni bloc `STATISTICS_*` parasite (signature d'une réécriture QGIS périmée).
- [ ] **22.4 Persistance** : sauvegarder le projet QGIS puis le rouvrir → toujours toutes les dalles visibles.
- [ ] **22.5 Dossier différent (non-régression)** : relancer dans un `output_dir` **différent** → aucune couche du premier run n'est retirée ; fonds de carte, polygone d'emprise (étape 1) et couche quadrillage restent **intacts** (jamais purgés).
- [ ] **22.6 Détections** : si des détections existent, le re-run rafraîchit aussi les couches `detections/<entité>/*.gpkg` (pas de doublon, données à jour).

## 23. Brique enclos (entité dérivée « Enclos ») ⭐ P0

> **Contexte** : première brique de synthèse non-DBSCAN (`type: enclosure` dans
> `args.yaml:clustering` du modèle formes linéaires). Fermeture vectorielle des
> détections `parcellaire` (talus_fosse retiré après test Bretagne : ne créait
> que des faux positifs — dilatation T/2 + ré-extension des
> trous) puis scoring : `closure_ratio`, `isolement`, `rectangularite`,
> `compacite`, `elongation`, `forme`, `nb_sources`, `enclos_id`. Publication =
> seuils durs aire/élongation/fermeture uniquement, l'archéologue trie par
> attributs.

- [ ] **23.1 Entité cochable** : étape 3, groupe « zone » → carte « Enclos » avec badge « ↳ regroupement automatique en zones » (pas de case « Regrouper en clusters »). En réglages avancés, la boîte de paramètres montre : Pontage des interruptions (m), Surface min/max (m²), Fermeture min, Élongation max — et **pas** les paramètres DBSCAN (Distance max, Densité…).
- [ ] **23.2 Run sur zone de validation** : cocher « Enclos » seul et lancer sur une zone à enclos connus. Le journal montre `Synthèse: 1 règle(s) à traiter` puis `Enclosure [1/1]: … T=10m` et `… enclos 'enclos' publiés`.
- [ ] **23.3 Sorties QGIS** : groupe « Enclos » avec couche `Enclos` (polygones des cours intérieures) + couche `Linéaments sources` (fragments, avec `enclos_id` rempli pour les contributeurs). Couleur stable, distincte des autres entités.
- [ ] **23.4 Attributs** : table de la couche Enclos → colonnes `surface_m2`, `closure_ratio` (≥ 0,6), `isolement`, `rectangularite`, `compacite`, `elongation`, `forme` (quadrangulaire/curviligne/irregulier), `nb_sources`, `enclos_id`, `confidence` (moyenne des fragments). Filtrer `"isolement" < 0.2` doit écarter les mailles de parcellaire mitoyennes.
- [ ] **23.5 Petits enclos préservés** : un enclos < 500 m² (seuil global du modèle) reste présent dans le GPKG — l'exemption du filtre d'aire fonctionne (journal : « couche synthétique 'Enclos' épargnée »).
- [ ] **23.6 Itération sans réinférence** : relancer le même run en changeant seulement « Pontage des interruptions » (ex. 10 → 16 m) → le journal indique la réutilisation du cache (`détections en cache`), pas de nouvelle inférence, et le nombre d'enclos change de façon plausible (plus de trous pontés).
- [ ] **23.7 Enclos emboîtés** : sur un double enclos connu (ou fixture), les deux circuits sortent comme deux polygones distincts (pas de fusion/suppression).
- [ ] **23.8 Non-régression regroupement cratères** : un run « Regroupement de cratères » (DBSCAN) donne le même résultat qu'avant (mêmes zones, mêmes paramètres éditables, libellés désormais « Nb min de détections »).

## 24. Brique axe linéaire (entité dérivée « Axes linéaires ») ⭐ P0

> **Contexte** : troisième brique de synthèse (`type: alignment`). Détecte les
> bandes directionnelles à brins multiples — le signal « spaghettis alignés »
> d'une voie ancienne : plusieurs lignes parallèles de détections parcellaires
> co-orientées (fossés bordiers, agger, tronçons décalés) dans un couloir
> ≤ 40 m, continu sur ≥ 500 m. Zone de référence : la voie romaine repérée
> par le collègue archéologue.

- [ ] **24.1 Entité cochable** : étape 3, groupe « zone » → carte « Axes linéaires » avec badge « regroupement automatique ». En réglages avancés : Largeur max de la bande (m), Tolérance d'orientation (°), Longueur min (m), Interruption max (m), Couverture min, Nb min de fragments — et pas les paramètres enclos/DBSCAN.
- [ ] **24.2 Run sur la zone de référence** : cocher « Axes linéaires » et lancer sur la zone de la voie romaine connue. Journal : `Synthèse: … règle(s)` puis `Alignment [1/1]: … bande=40.0m` et `… axe(s) 'axe_lineaire' publiés`. **La voie connue doit sortir comme UN axe** (pas 3–4 axes parallèles concurrents).
- [ ] **24.3 Sorties QGIS** : groupe « Axes linéaires » avec couche corridor (rectangles orientés) + couche « Fragments sources » (`axe_id` rempli pour les contributeurs).
- [ ] **24.4 Attributs sur la voie connue** : `longueur_m` ≥ 500, `azimut_deg` cohérent avec le tracé, `nb_brins` ≥ 2 attendu (fossés + agger), `couverture` ≥ 0,25, `parallelisme` faible (≈ 0–1), `discordance_deg` élevée si la voie traverse le parcellaire local.
- [ ] **24.5 Anti-faux-positifs** : sur une zone à parcellaire coaxial dense, les corridors sortent avec `parallelisme` ≥ 2 et/ou `connecteurs_perp` élevés → filtrables par expression QGIS.
- [ ] **24.6 Itération sans réinférence** : relancer en passant Longueur min de 500 → 300 m ou la bande de 40 → 60 m → cache réutilisé, nombre d'axes change de façon plausible.
- [ ] **24.7 Cohabitation des briques** : cocher « Enclos » + « Axes linéaires » ensemble → un seul run du modèle, les deux groupes de couches sortent, aucun conflit (les fragments parcellaire portent enclos_id ET axe_id le cas échéant).
