---
name: fiche-classe-plugin
description: >
  Écrire la fiche de chaque classe détectable d'un modèle installé dans le plugin
  (vignettes du corpus, provenance d'entraînement, usage au seuil déployé,
  hors-cible), et calibrer puis documenter ses cibles dérivées (regroupements).
  Dernière étape du workflow modèle, après /installer-modele-plugin. Utiliser
  quand l'utilisateur dit « la fiche de X », « des images pour la classe »,
  « propose-moi des vignettes », « calibre le regroupement », ou quand le
  validateur du plugin signale « pas de bloc 'fiche' ».
argument-hint: <id du modèle installé, ex. crateres_seg_ld_v1 [classe]>
entrees:
  - "data/models/<id>/ installé et validé (/installer-modele-plugin) ; corpus COCO local C:\\projets\\Archeologia\\training-models\\corpus\\<corpus>\\"
  - "pour une cible dérivée : un run d'inférence du plugin sur une zone de test + les cibles dessinées par l'utilisateur"
sorties:
  - "model_card.yaml : bloc fiche (+ cadrage) sur chaque classes[] et derived_targets[] ; data/models/<id>/vignettes/"
  - "package/ du run Drive resynchronisé (reporter_sur_drive) ; sauvegardes D:\\brouillons\\vignettes_candidates\\<classe>\\ (candidats, page HTML, choix_<classe>.json, fiche, cadrages)"
suivant: []
---

# Fiche de classe dans le plugin (dernière étape du workflow modèle)

Interagir en français. Un modèle n'est pas livré tant que chacune de ses classes,
cibles dérivées comprises, n'a pas sa fiche : c'est ce que l'archéologue lit à
l'étape 3 avant de cocher une entité (illustration RVT, où et en quelle quantité
la classe a été apprise, rendement au seuil déployé, hors-cible). Le validateur
du plugin le rappelle par un WARNING `classes['x'] : pas de bloc 'fiche'` /
`derived_targets['x'] : pas de bloc 'fiche'` — le but est zéro warning de fiche.

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

Le contrat fait foi côté plugin : `docs/model_contract.md` § « Produire une fiche
de classe » (bloc `fiche` : `resume`, `reconnaitre`, `usage`, `hors_cible`,
`vignettes[]` + `cadrage`, `entrainement{corpus, annotation, zones[], splits}`).
Outils : `dev/fiches/` du plugin (`page_choix.py`, `appliquer_choix.py`,
`injecter_fiches.py`, `injecter_cadrages.py`, `comparer_drive.py`,
`reporter_sur_drive.py`, `bundles_drive.json`, README).
`data/models/**` est gitignoré : la copie qui survit à un ré-export est
`runs/training/<run>/package/model_card.yaml` du Drive, d'où le report final.

## Étapes (ordre imposé — le choix et la vérification ne se sautent pas)
1. **Candidats** : six cadres du corpus COCO (split train), pour chaque cadre le
   brut (RVT seul) et l'annoté (mêmes pixels + polygones de vérité terrain, trait
   jaune 1 px), en variant zones et états de conservation (dense / clairsemé /
   arasé sous labour / sous plantation / au contact du bâti). Source : les tuiles
   PNG + `_annotations.coco.json`, JAMAIS les planches `visualizations/` du run (sur le Drive, jamais dans le plugin)
   (montages 4 panneaux). Outil : `python dev/fiches/candidats_corpus.py <corpus>
   --categorie <cat COCO> --sortie D:rouillonsignettes_candidates\<classe>\<campagne>
   [--eviter-zone <zone de l'icône déjà installée>] [--gsd]` (fenêtre carrée à
   objets entiers, NoData et tuiles plates écartés, `candidats.json` + planche). Tuiles trop petites pour être lisibles (252 px) →
   mosaïque 2×2 de tuiles contiguës du même split, grille sans recouvrement.
   Écarter tout cadre touchant du NoData. Noter dans `candidats.json` les tuiles,
   l'emprise Lambert-93, le profil et le secteur. Relire les candidats par trois
   juges indépendants (archéologue / graphiste / sceptique : lisibilité du brut,
   de l'annoté, fenêtre lisible à 44 px, profil annoncé tenu) et remplacer les
   cadres écartés par les profils manquants avant de présenter.
2. **Choix humain, DANS LE NAVIGATEUR** (jamais une planche à commenter par
   numéros) : `python dev/fiches/page_choix.py <dossier> --classe <name ou
   output_class> --label "<label_fr>" --modele <id> --corpus "…" --objets "…"
   --sortie page.html` fabrique la page à partir de `candidats.json` et des
   `N_brut.jpg` / `N_annote.jpg` (base64, ~1,5 Mo pour six cadres). La publier
   avec l'outil Artifact (`capabilities: {"db": {}}`, favicon requis), donner le
   lien et S'ARRÊTER : l'utilisateur clique les cadres à retenir (l'ordre des
   clics fait l'ordre de la fiche, le premier est l'icône) et bascule brut /
   vérité terrain. Ne jamais choisir à sa place. Une fiche à UNE seule image (il
   en faut au moins deux) se complète par `--complement` : l'icône reste, pas
   d'étape de cadrage, puis `appliquer_choix.py --rang-depart 2`. Modèles : artefacts « Vignette
   des tranchées » (2026-09-10) et « Vignettes · Regroupement de cratères »
   (2026-09-15) dans la galerie du plugin.
3. **Cadrage, sur la même page** : section « 2 · Cadrer », fenêtre glissable et
   redimensionnable avec aperçu de l'icône à 44 px et ×2,5 — une vignette
   couvre 250 à 324 m, réduite telle quelle à 44 px elle est illisible. Chaque
   geste enregistre `choix/<classe>` = `{classe, modele, retenues, cadrages}`
   dans la base de l'artefact. Quand l'utilisateur dit avoir fini : relire par
   `read_db` (collection `choix`, doc `<classe>`), sauver le document en JSON
   dans `D:\brouillons\vignettes_candidates\<classe>\choix_<classe>.json`,
   puis `appliquer_choix.py <choix.json> <dossier>` (copie des cadres dans
   `vignettes/<classe>_0k_*.jpg`, JSON de cadrage de l'icône) et
   `injecter_cadrages.py <cadrages.json> <planche.png>`. REGARDER la planche :
   l'icône doit montrer la structure (disques noirs, semis), pas une route ni un
   remblai ; sans structure lisible à 44 px, revenir à la page et changer de cadre.
4. **Rédaction sur sources primaires, jamais de mémoire** :
   - effectifs : `corpus_manifest.yaml` + RECOMPTE des trois COCO (tuiles et
     masques par split, tuiles vides, médiane d'objets par tuile porteuse,
     tailles = côté de boîte × GSD, parts par tranche de taille) ;
   - rendement : `entrainement/evaluation/metriques_eval.json` — P/R au seuil
     DÉPLOYÉ re-dérivés des bandes tp/fp (somme des bandes ≥ seuil ; R = tp/n_gt),
     FP par tuile, vrais objets par tuile, parts de vrais par catégorie de
     fiabilité (mêmes coupures que `thresholds.fiabilite`), IoU médian nommé
     comme tel ;
   - provenance de l'annotation : `configs/vecteurs_<zone>.yaml`, journaux du
     chantier (`slice_*.log`, `revue_auto.log`, `build_zone_gpkg.log`),
     `CHANTIER.md` — outil de découpe = `slice_zone.py` (dataset), `build_corpus.py`
     n'assemble que le corpus ; composition réelle des couches sources (ex. d1-d5
     et un tiers de d6, 83 047 parties hors fichiers) ;
   - cohérence avec `model_card` (`known_limitations`, `inference_choices`,
     `recommended_use`) et `data/entities_catalog.json` ;
   - documents de la zone : `raw/docs/` de `data_regions_v2/<zone>/` sur le Drive
     (rapports, thèses, articles) — c'est là que se lit la NATURE de l'annotation
     (classes de la source, méthode, exhaustivité). Leçon cratère 2026-09-15 : la
     fiche disait « abris non annotés » alors que l'article de la source
     (de Matos-Machado et al., 2019, `interpretation_classes_ponctuelles.pdf`)
     montre que les classes D4-D6 fusionnées dans la couche sont des abris.
   Règles de contenu : `reconnaitre` = UNE phrase courte, sur le modèle « Sur le
   Local Dominance à 0,5 m, le cratère se lit comme une tache sombre, ronde ou
   légèrement ovale, le plus souvent de 2 à 5 m » — support (indice + résolution),
   aspect, taille courante en fourchette simple et arrondie, rien d'autre (décision
   utilisateur 2026-09-16 : le paragraphe de la fiche cratère était trop long ;
   contexte, répartition, coalescence, états de conservation et classes englobées
   vont dans `usage`, `hors_cible`, `entrainement.annotation` ou
   `known_limitations`, ou se coupent) ; `usage` = rendement franc au seuil de confiance déployé (le libellé
   de l'interface est « Seuil de confiance déployé », 2026-09-15), par zone, en disant
   quand la classe est faible, et si l'annotation de référence n'est pas exhaustive :
   précision et fiabilité mesurées sont alors des PLANCHERS (à dire aussi dans
   `entrainement.annotation` et `known_limitations`) ; `usage`, `entrainement.corpus` et
   `entrainement.annotation` = LISTES de puces courtes (une idée par puce) en langage d'archéologue — pas
   de hash de commit, de chemin de config ni de nom d'outil (la traçabilité vit dans
   le run et `manifests/`) ; `entrainement.annotation` CRÉDITE toute vérité terrain
   qui n'est pas la nôtre — organisme fournisseur + année, et la publication de la
   méthode quand elle existe (« couche de référence fournie par Sorbonne Université
   (2025), issue d'une détection semi-automatique publiée par de Matos-Machado et al.,
   2019 » ; « relevés LiDAR de l'ONF, Rambouillet 2024 ») : un crédit n'est pas du
   dispositif interne, il dit au lecteur ce qui fait autorité ; rien d'INVÉRIFIABLE
   dans un texte affiché — chaque énoncé, même qualitatif, se rattache à une source
   rejouable (COCO, banc, document de la zone, manifeste) ou à une vignette de la
   fiche ; le constat visuel de l'auteur sur un cadre écarté n'en est pas une (leçon
   2026-09-16 : « sous labour, la cuvette s'efface en tache floue dans les rayures de
   charrue » ne tient qu'aux candidats 5 et 6, absents de la fiche) — sourcer, montrer
   le cadre, ou couper ; `hors_cible` = classes voisines du même modèle +
   exclusions volontaires du corpus — JAMAIS une cible du catalogue (les fosses
   d'extraction sont une cible du regroupement, pas un hors-cible) ; le libellé
   de la 4e catégorie de fiabilité est « très probable » (celui du plugin) ;
   toponymes et distances seulement s'ils sont sourcés (grille de tuiles,
   `candidats.json`, manifeste de zone) ; mono-zone → le dire, sans inventer de
   mesure de généralisation ; sous la vignette l'interface n'affiche que le LIEU
   (`vignettes[].zone`, ex. « Alès, garrigues nord-est (30) », décision 2026-09-15) —
   `legende` reste une note interne de relecture, courte ; aucune référence au
   dispositif interne (zone de référence gelée, checkpoints, commits, manifestes)
   dans un texte affiché ; `hors_cible` se confronte à la COMPOSITION de la couche
   source (classes fusionnées à l'audit : les abris de Verdun sont DANS la classe
   cratère) ; à chaque leçon nouvelle, repasser sur TOUTES les fiches installées.
5. **Vérification adverse OBLIGATOIRE** : trois relecteurs indépendants (corpus,
   évaluation, contenu/légendes contre les images) dont le mandat est de prendre
   chaque chiffre en défaut, puis une synthèse qui tranche les contradictions ;
   appliquer les corrections, y compris celles qui déplacent une phrase de
   rubrique. Historique : 6 écarts bloquants sur 22 à la première campagne
   (statistiques inventées puis auto-attestées), 3 bloquants sur la fiche cratère
   (outil de découpe faux, cible du catalogue mise en hors-cible, secteur d'une
   vignette faux). Ne pas injecter avant ce passage.
6. **Injection et contrôle** : `injecter_fiches.py <fiches.json>` (liste de
   `{modele, classe, fiche}` ; `classe` = `name` d'une classe ou `output_class`
   d'une cible dérivée), puis cadrages, puis
   `scripts/validate_models_metadata.py` → 7/7 OK sans warning de fiche, et
   contrôle par `app.services.class_fiche.fiches_par_entite` (fiche complète,
   cible dérivée en tête de son entité). Sauvegarder `fiche_*.json`,
   `cadrages_*.json`, `model_card_avant/avec_fiche.yaml` dans
   `D:\brouillons\vignettes_candidates\<classe>\`.
7. **Report Drive** : ligne du modèle dans `dev/fiches/bundles_drive.json`
   (`<famille>/<modèle>/runs/training/<run>/package`), `reporter_sur_drive.py`
   (remplace les blocs fiche du Drive par ceux du plugin + vignettes), et copie À LA MAIN de toute autre modification
   du `model_card` / `args.yaml` (le report ne couvre que les fiches) ; puis
   `comparer_drive.py` = « identiques hors fiche ». Drive absent (streaming) →
   le dire, le noter en mémoire, refaire plus tard.

## Cibles dérivées (regroupements) — calibrer AVANT d'illustrer
Une cible dérivée (`derived_targets[]` du model_card ↔ règle `clustering` de
`args.yaml`) n'est pas apprise : sa règle se CALIBRE sur un run du plugin.
1. Lancer l'inférence du modèle sur une zone de test représentative de l'usage
   (ex. carrières du Gard pour le regroupement de cratères), faire dessiner par
   l'utilisateur les zones attendues (approximatives, non exhaustives par nature)
   et récupérer toute couche d'objets connus qui a servi à choisir les dalles.
2. Harnais figé : détections brutes géoréférencées (PNG normalisés × world file,
   règle du centroïde par dalle), cibles, objets connus ; `evaluer` = cible
   retrouvée si couverte ≥ 30 %, zone parasite si < 20 % dans une cible, et
   parasites ne touchant AUCUN objet connu comptés à part. Balayer les règles
   avec les fonctions mêmes du plugin (`src/pipeline/cv/clustering.py`), puis
   comparer les variantes par un GeoPackage chacune (statut cible / connue /
   parasite, `conf_p90`, `elong_med`) que l'utilisateur ouvre dans QGIS.
3. Leçons de l'audit 2026-09-14 (rapport dans `dev/docs/_local/` du plugin) :
   la densité de centroïdes est une mesure saturée (≈ 35 cratères/ha, chaque
   cible gagnée coûte 8-15 parasites), la confiance ne sépare rien au niveau du
   cratère mais agrégée par zone (90e centile, `min_conf_p90`) elle écarte les
   terrasses agricoles ; les cibles à moins de 35 objets ou hors gabarit du
   modèle sont hors de portée de tout regroupement — le dire dans la fiche.
   Vérifier les « parasites » sur le relief avant de les compter : plusieurs
   étaient des extractions non dessinées.
4. Écrire la règle dans `args.yaml` avec son commentaire de calibrage (jeu,
   cibles, résultat, manquées et pourquoi, falaises de réglage), le bloc
   `derived_targets` (`entity` ∈ catalogue, `label_fr`, `include_source`), puis
   la fiche `derived_targets[].fiche` (contrat 2026-09-14) : `reconnaitre` =
   le mécanisme (distances, effectifs, seuils, enveloppe), `usage` = chiffres du
   calibrage, `hors_cible` = groupes trop petits / peu sûrs / champs de bataille,
   `entrainement` = « pas d'apprentissage » + jeu et critère de calibrage ;
   vignettes = LD du test avec les zones produites (annoté : zones + cratères).
5. Toute modification de `src/pipeline/cv/**` invalide le binaire d'inférence
   (`test_runner_binary_fresh`) : recompiler par `dev/runner_onnx/build.py`.

## Garde-fous
- Aucun chiffre sans source primaire rejouable ; aucun toponyme sans coordonnée.
- Le choix des cadres est humain ; la relecture adverse n'est pas optionnelle.
- Ne jamais ré-injecter par-dessus une fiche : restaurer `model_card_avant_fiche.yaml`
  puis ré-injecter (l'injecteur ignore une classe déjà pourvue).
- Vignettes relatives au dossier du modèle, dans `vignettes/` (livrées dans le
  ZIP ; `entrainement/` ne l'est pas).
- Le model_card du plugin et celui du `package/` Drive doivent finir identiques.
