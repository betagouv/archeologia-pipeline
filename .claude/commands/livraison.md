---
description: Livrer une version du plugin — bump, grille IGN, manuel, binaire, Talisman, PR dev→main, tag, packaging depuis le dossier principal, dépôt OVH, retour dans les branches.
argument-hint: <version cible, ex. 0.14.0> [branche à fusionner dans dev, ex. feat/aide-integree]
---

# Runbook — livraison d'une version

Interagir en français. **Suis les étapes dans l'ordre** et **confirme avec l'utilisateur avant
chaque action sortante** (push, PR, merge, tag, dépôt). Rien de ce runbook n'écrit sur OVH :
le dépôt est déposé par l'utilisateur (FTP), le runbook s'arrête au ZIP et au `plugins.xml`.

## Entrées
- **Version** : `$1`. Bump proposé selon `CLAUDE.md` § Versionnage (minor dès qu'une fonctionnalité est
  visible). Jamais de bump sur un simple commit intra-branche.
- **Branche à intégrer** : `$2` (facultatif). Sinon, livrer `dev` tel quel.

## Où l'on travaille
- Le dossier `plugins/archeologia-pipeline` du profil QGIS est **ce que QGIS charge** et le seul à
  contenir la grille, les modèles et le binaire (gitignorés) : **le packaging se fait là**, avec la
  branche livrée extraite. Le git préparatoire peut se faire dans un worktree
  (`C:/projets/Archeologia/worktrees/<branche>`), jamais depuis un worktree **détaché** (la branche
  n'avancerait pas).
- Secrets : l'adresse du dépôt OVH et le dossier de sortie viennent de
  `dev/docs/_local/deploy.config.json` (gitignoré). Ne jamais écrire l'adresse dans un fichier suivi
  ni dans une image du manuel.

## Étapes

1. **État des branches.** `git fetch`, `git log --oneline origin/main..dev`, `git log --oneline dev..$2`.
   Lister ce qui part dans la version ; signaler ce qui doit rester caché (ex. l'onglet Visualisation :
   `VISUALISATION_TAB_ENABLED` reste False sur dev).
2. **Fusionner `$2` dans dev** avec commit de fusion (`git merge --no-ff`), jamais de squash. Résoudre,
   `python run_tests.py unit`, `ruff check src/`.
3. **Grille IGN à jour** : `python dev/build_quadrillage_from_wfs.py` depuis le dossier principal
   (quelques minutes ; l'IGN ajoute ~2 500 dalles toutes les 1 à 3 semaines). Noter le nombre de dalles
   pour le changelog.
4. **Manuel** : `python run_tests.py -k test_aide` vert ; chaque fonctionnalité de la version a son
   chapitre à jour (skill `manuel-integre`). Puis, **à chaque livraison**, les deux rendus hors écran
   (Python de QGIS, `C:/OSGeo4W/bin/python-qgis.bat`) :
   `dev/rendu_hors_ecran.py captures` refait les captures de l'assistant dans `aide/img/` (étapes 1 à 3
   et la vue d'exécution en cours de run ; à commiter si l'interface a changé, à vérifier d'un coup
   d'œil sinon), `dev/rendu_hors_ecran.py manuel --tous` rend chaque chapitre et doit finir sur
   `RESULTAT : OK` dans `dev/docs/_local/rendu/controles.txt` (barre horizontale à 0 px partout,
   historique, zoom), `dev/rendu_hors_ecran.py journal` (fin de run : cadre « Par où commencer »,
   renvoi cliquable du journal, bascule récap ↔ journal de l'étape 4) et
   `dev/rendu_hors_ecran.py profil --modele <id>` (fiche de classe et fiche de modèle : profil des
   scores, bandeau « Appris sur ») — un `ECHEC` est une image qui déborde ou un écran cassé, à corriger
   avant le bump. Le chapitre Nouveautés se remplit tout seul depuis le changelog.
5. **Binaire CV** : `python run_tests.py -k binaire_a_jour` doit être **vert**. Rouge →
   `python dev/runner_onnx/build.py` (venv dans `dev/runner_onnx/.venv_onnx`), puis smoke run de
   l'exe sur un PNG avec un modèle installé. Un test rouge au moment d'une livraison n'est jamais hors
   périmètre (v0.11.0 livrée avec un binaire périmé).
6. **Bump** : `metadata.txt` (`version=` + entrée **en tête** de `changelog=`, indentée comme les
   précédentes) et `README.md` ligne « Version ». Changelog **relu contre le diff**
   (`git diff v<précédente>..dev --stat`), en langage utilisateur, sans jargon de code, sans
   estimation de durée. `git grep '<ancienne version>'` : seules les références historiques restent.
7. **Talisman** : le hook pre-push lance ruff, le validateur des modèles et Talisman. Blocages connus :
   somme périmée d'un fichier listé → `talisman --checksum="<f1> <f2>"` et remplacer la ligne dans
   `.talismanrc` ; nouveau fichier flaggé → bloc imprimé par Talisman à copier ; faux positifs de langue
   (le mot « key », le verbe « passer » conjugué). Lire chaque motif avant d'ajouter une somme. Jamais
   `--no-verify`.
8. **Push dev** (`git push origin dev`) puis **PR** : `gh pr create --base main --head dev`. Attendre le
   contrôle GitGuardian : `UNSTABLE` juste après l'ouverture ne veut rien dire, réinterroger avant de
   conclure (`gh pr view <n> --json statusCheckRollup`).
9. **Merge** : `gh pr merge <n> --merge` (commit de fusion ; **jamais** `--squash`, dev doit rester
   ancêtre de main ; **jamais** `--delete-branch`).
10. **Tag annoté sur le commit de FUSION** : `git fetch`, `git tag -a v$1 <sha du merge> -m "v$1"`,
    `git push origin v$1`.
11. **Packager depuis le dossier principal** : y extraire `main` (`git switch main && git pull`), puis
    `python dev/package_plugin.py` (ZIP + `plugins.xml` dans `_local/depot`). Vérifier dans le ZIP :
    dossier racine `archeologia`, `version=` attendu, `aide/` avec ses images, `data/zones_corpus.json`
    et `data/indices_fiches.json` (le bandeau « Appris sur » et les fiches de produits en dépendent),
    **pas** de `docs/`, de `dev/`, de `tests/` ni de `local_catalogue/`, `build_info.json` du binaire au
    bon commit, grille `.shp` + `.qix`.
12. **Dépôt OVH** : l'utilisateur dépose le ZIP et `plugins.xml` (FTP, cf. `dev/docs/DEPLOIEMENT_DEPOT_OVH.md`),
    puis vérifie la pastille de mise à jour dans un QGIS. Le guide d'installation privé de
    `dev/docs/_local/` est à relire si l'installation a changé (menu Manuel, nouvelle dépendance).
13. **Retour dans les branches en cours** : fusionner `dev` (ou `main`) dans chaque branche de
    fonctionnalité vivante (ex. `feat/visu-flux-gpf`) pour qu'elle reparte de la version livrée ;
    remettre le dossier principal sur la branche de travail de l'utilisateur.
13bis. **Dépôt archeologia-ovh** : si la version touche `src/pipeline/` ou `src/app/` (chemins de
    sortie, finalisation, stratégie d'entrée…), le dire à l'utilisateur — ce dépôt privé vendorise une
    copie de `src/` et la resynchronise lui-même (puis ré-applique son patch v2). Rien à faire ici,
    jamais de copie depuis ce dépôt : le signaler, c'est tout (cf. CLAUDE.md § Trois dépôts).
14. **Mémoire** : noter la version livrée, ce qui reste (recettes non jouées, captures), et mettre à
    jour les mémoires de projet concernées.

## Sortie attendue
Un message à l'utilisateur avec : le numéro de version, le sha du merge et du tag, le chemin du ZIP et
sa taille, ce qui a été vérifié dans le ZIP, et la liste de ce qu'il lui reste à faire (dépôt FTP,
recettes QGIS).
