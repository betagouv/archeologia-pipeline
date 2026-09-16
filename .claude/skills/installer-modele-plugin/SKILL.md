---
name: installer-modele-plugin
description: >
  Installer un modèle entraîné dans le plugin QGIS archeologia-pipeline :
  dossier data/models complet (contrat + métriques + courbes), export ONNX,
  porte de parité binarisée, sidecar, entité catalogue, validation. Utiliser
  quand l'utilisateur dit « installe le modèle dans le plugin », « ajoute X au
  plugin », « package et déploie », « exporte en ONNX ».
argument-hint: <id du modèle, ex. enclos_fr_seg_v2>
entrees:
  - "runs/training/<run>/package/ + evaluation/metriques_eval.json CONFORME (verif_courbes_eval)"
sorties:
  - "data/models/<id>/ complet (contrat + entrainement/evaluation/ + weights ONNX+sidecar)"
  - "entities_catalog.json à jour (commit) + dashboards régénérés"
suivant: [fiche-classe-plugin]
---

# Installation d'un modèle dans le plugin (checklist complète)

Interagir en français. Le plugin VIVANT est le checkout du profil QGIS
(`%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\archeologia-pipeline`,
branche git propre à vérifier) — `data/models/**` y est gitignoré (déploiement),
mais `data/entities_catalog.json` est SUIVI : le signaler pour le prochain commit.
Gabarit de référence : un modèle déjà conforme (ex. `cratere_circulaire_2` +
compléments 2026-08).

## Deux dépôts, une seule procédure
Ce skill est PARTAGÉ, octet pour octet, entre `training-models` et le plugin
`archeologia-pipeline` (`.claude/skills/` des deux) : modifier l'un = reporter sur
l'autre par `python <PLUGIN>/dev/sync_skills_training.py --check | --vers-training |
--depuis-training` (les deux suites de tests échouent tant qu'ils divergent). Racines,
quel que soit le dépôt courant :
- TRAINING = `C:\projets\Archeologia\training-models` : `tools\…`, `configs\`,
  `corpus\`, `tests\run_all.py` ;
- PLUGIN = `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\archeologia-pipeline` :
  `data/models/`, `dev/`, `scripts/`, `docs/model_contract.md`, `run_tests.py`.
Un chemin `tools\x.py` se lit sous TRAINING ; `dev/…`, `scripts/…`, `data/…` sous PLUGIN.

## Le dossier data/models/<id>/ DOIT contenir
- Contrat (À LA RACINE) : `args.yaml`, `classes.txt`, `config.json`,
  `model_card.yaml`, `training_params.json`, `evaluation_results.json` ;
- **Traçabilité entraînement (dans `entrainement/`)** : `metrics.csv`
  (+ historiques numérotés si reprises, avec NOTE-metriques.md disant quel CSV
  = le checkpoint déployé), `hparams.yaml`, tfevents — mais **JAMAIS
  `visualizations/`** (prédictions de test et planches d'inférence d'un run : Drive `runs/training/<run>/visualizations/`, ou `D:\brouillons\<chantier>\` le temps d'un test, règle utilisateur 2026-09-15 ; le
  validateur refuse le dossier), **`evaluation/`**
  (metriques_eval.json + planches + appariements.json du modèle DÉPLOYÉ,
  copiés depuis `runs/training/<run>/evaluation/` du Drive) et les
  superpositions `comparaison_<vs>/` (tools/courbes_eval.py ; convention
  2026-08-17 : toujours dans entrainement/). `evaluation_results.json` racine
  = legacy notebook, documentaire, ne JAMAIS l'écraser ni s'en servir comme
  source de seuils ;
- `weights/` : `best.pth`, `best.onnx`, `best.json` ;
- `vignettes/` : les cadres des fiches de classe (livrés dans le ZIP, à la
  différence d'`entrainement/`) — produits par `/fiche-classe-plugin`.

## Étapes
1. Copier `package/` du run Drive → `data/models/<id>/` + compléments ci-dessus.
1bis. **LES POIDS LIVRÉS SONT-ILS CEUX QU'ON A MESURÉS ?** (bloquant depuis 2026-09-14,
   après DEUX récidives : crateres_seg_ld_v1 et tranchees_seg_ld_v1 ont été installés sur
   des poids jamais évalués) — `tools/verif_poids_evalues.py data/models/<id> --strict` :
   **CONFORME** exigé. Le notebook mesure `checkpoint_best_ema.pth` et package
   `checkpoint_best_total.pth` : rien d'autre ne compare les deux. Les autres verdicts ne
   valent PAS un feu vert — `CONFORME_TAILLE` est une présomption (le checkpoint mesuré n'est
   plus sur le poste), `AUTO_REFERENCE` signifie que l'éval a mesuré le fichier livré lui-même
   (vrai par construction, aveugle à un remplacement ultérieur), `INDÉTERMINABLE` est
   exactement ce que rendent les deux bugs historiques quand le checkpoint est hors de portée.
   SUSPECT → remesurer les poids LIVRÉS par `courbes_eval` avant toute conclusion, et n'écrire
   dans le model_card que des chiffres issus de cette remesure. Penser à donner à
   `--modele <nom>=...` **l'id exact du modèle** : une clé différente dans `metriques_eval.json`
   casse le badge plugin du dashboard et crée une entrée fantôme au registre.
2. Cohérence des métadonnées : lancer `tools/verif_courbes_eval.py` sur
   `entrainement/evaluation/` — CONFORME AVANT de recopier les seuils ;
   `classes.txt` = ids d'ENTITÉ du catalogue ;
   `thresholds.confidence_default` + `confidence_per_class` = seuils CHOISIS
   (cellule 11bis, règle 2026-09-09) dans la fenêtre [bas du plateau F1 ≥ 95 % ;
   F1-max] du bloc `etude_seuil` de `entrainement/evaluation/metriques_eval.json`
   (jamais 0,3 par défaut, jamais « = F1-max » sans étude ; éval sans le bloc →
   `completer_metriques_eval.py` d'abord), champ `thresholds.seuils_provenance`
   renseigné (chemin + date + mesure + raison du choix) ; `thresholds.fiabilite`
   (catégories douteux/possible/probable/quasi_certain PAR CLASSE, cellule 11bis :
   `{categorie, seuil, garanti, mesure, n}` + provenance, 1re catégorie AU seuil de
   la classe, effectifs relus — sans ce bloc le plugin affiche les tranches de score
   historiques) ; vérifier
   clés `par_classe` de metriques_eval.json == classes.txt (piège du mapping
   croisé enclos ie/fr) ; version/description/known_limitations remplis ;
   bloc RVT/MNT = GSD et rayons LD du corpus d'entraînement.
3. Export ONNX : `dev/runner_onnx/.venv_onnx` du PROFIL, avec
   `PYTHONIOENCODING=utf-8` (sinon crash charmap en validation) :
   `export_to_onnx.py --model …best.pth --output …best.onnx --type rfdetr
   --imgsz <res> --opset 17`.
4. **Porte de parité** (BLOQUANTE depuis 2026-08-31) : le verdict de
   `validate_onnx_export` fait échouer l'export ; l'ancien faux positif atol
   bf16 des logits de masque est absorbé par la porte elle-même (sorties
   spatiales jugées sur la DÉCISION : masques binarisés identiques / argmax
   identique — le contrôle manuel « IoU 1,0 binarisé » est désormais intégré).
   Échec réel → ne pas installer.
4bis. **Parité de décision sur ≥ 100 tuiles de test** (obligatoire depuis 2026-09-08,
   la porte ne teste que 2 images) : `tools\verif_parite_onnx.py <best.pth> <best.onnx>
   <corpus>/test --resolution <res> --seuil <déployé>` (venv_onnx du plugin) →
   CONFORME ; résultat consigné dans `entrainement/NOTE-export-onnx.md`. Détection
   rfdetr : la porte juge la DÉCISION (permutation des requêtes = faux positif
   d'allclose, `_parite_decision_detection`).
5. Sidecar `best.json` : `class_offset` correct (rfdetr ≥1.8 → 0 ; vieux exports
   → 1 ; un offset faux SUPPRIME silencieusement des classes), `resolution`,
   `class_names` = classes.txt.
5bis. Registre `data_regions_v2\modeles.yaml` : entrée du nouveau modèle (statut,
   entités, zones, évaluation) ; le modèle remplacé passe en « retiré du plugin
   (<date>) » — jamais supprimé du Drive. Retrait d'un modèle → entités devenues
   orphelines retirées du catalogue (à recréer avec le modèle suivant), déplacer le
   dossier hors du plugin (`D:\entrainement_ponctuelles\_retire_plugin\`), pas rm.
   Piège : `validate_models_metadata.py data/models` traite le dossier parent comme
   un modèle (0/1) — valider modèle par modèle.
6. Entité(s) au catalogue `data/entities_catalog.json` (id snake_case, label,
   description, morphology, display_order) ; retirer/ne pas laisser d'entité
   orpheline sans modèle (elle s'affiche « Aucun modèle disponible » dans le
   wizard).
7. `scripts/validate_models_metadata.py data/models/<id>` → 1/1 OK exigé
   (validateur v2 : valeurs d'inference_choices, seuils adossés à
   metriques_eval.json, entités ⊆ catalogue, derived_targets ↔ clustering).
7bis. **Fiche de classe** (obligatoire depuis 2026-09-14, skill
   `/fiche-classe-plugin`) : chaque `classes[]` ET chaque `derived_targets[]` du
   model_card doit porter son bloc `fiche` (vignettes du corpus choisies par
   l'utilisateur, cadrage de l'icône, provenance et effectifs recomptés, usage au
   seuil déployé, hors-cible) relu adversairement avant injection. Une cible
   dérivée (règle `clustering` d'args.yaml) se CALIBRE d'abord sur un run de test
   du plugin contre des cibles dessinées, un GeoPackage par variante, avant d'être
   illustrée. Le validateur (étape 7) doit finir sans aucun WARNING de fiche ; le
   report Drive du model_card final (étape 9) inclut ces blocs et `vignettes/`.
8. Recharger le plugin dans QGIS (plugin reloader) ; rappeler le commit du
   catalogue à l'utilisateur.
9. **Régénérer le dashboard** : `tools/tableau_modeles.py` sur la racine
   model-training `--registre <data_regions_v2>\modeles.yaml` et
   `--plugin <data/models du plugin de référence, ou zip publié>` (règle CLAUDE.md
   « après tout dépôt d'évaluation » ; la fiche du modèle resitue le seuil déployé
   du model_card dans la fenêtre mesurée [bas du plateau F1 ≥ 95 % ; F1-max] : hors
   fenêtre = à justifier dans `seuils_provenance`, sinon signalé « non justifié » ;
   posé AU F1-max = « rappel non privilégié ») + resynchroniser le `package/` du run
   Drive avec le model_card final (la seule copie correcte ne doit pas être
   uniquement le data/models gitignoré du laptop).
10. **Clôture du brouillon de chantier** (CLAUDE.md § Brouillons de chantier) : le
    modèle est déposé et installé, le dossier `D:\brouillons\<chantier>` n'a plus de
    raison d'exister. Lancer `tools\cloturer_chantier.py D:\brouillons\<chantier>` ;
    déposer ou justifier chaque ligne SANS JUMEAU (`cloture-ignore.txt`), archiver ce
    fichier dans `docs/`, puis **proposer la suppression à l'utilisateur** (jamais
    supprimer de sa propre initiative). Exception : le
    `_retire_plugin\` d'un modèle retiré n'est pas un brouillon tant qu'il est la
    seule copie du modèle sortant — l'archiver sur le Drive avant clôture.

## Garde-fous
- Seuil de production ≠ F1-max (règle 2026-09-09) : choisi SOUS le F1-max par étude
  de la courbe (plateau, F2-max, FP/image, précision marginale, zones), justifié dans
  `seuils_provenance` ; cf. CLAUDE.md « Choix du seuil de production ».
- Résolution d'entraînement = résolution d'export = résolution d'inférence.
- Ancien mécanisme remplacé : COMMENTER (model_card/args), jamais supprimer.
- Tout chiffre affiché dans model_card provient d'une mesure tracée
  (`entrainement/evaluation/metriques_eval.json`) — pas de valeur héritée d'un
  autre modèle, pas de chiffre lu sur un PNG.
