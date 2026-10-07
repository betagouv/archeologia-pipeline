---
name: manuel-integre
description: >
  Tenir à jour le manuel intégré du plugin (aide/*.md, livré dans le ZIP, ouvert par le bouton
  « Aide », F1 et le menu Manuel) : quel chapitre modifier quand une fonctionnalité visible change,
  comment écrire un lien interne, placer une image ou une capture, ajouter une rubrique de
  dépannage, et quels tests tiennent le contrat. Utiliser dès qu'un commit touche src/ui/ ou le
  pipeline de façon visible, ou quand l'utilisateur dit « documente », « ajoute au manuel »,
  « mets à jour l'aide », « une capture pour le manuel ».
---

# Manuel intégré — entretien

Interagir en français. Règle unique, la même que pour les fiches : **toute fonctionnalité visible
modifie son chapitre dans le même commit**. Sinon le manuel redevient la notice Word, six versions en
retard, qu'il a remplacée le 2026-10-07.

## Où ça vit
- Source : `aide/<nn>-<cle>.md`, un chapitre par écran ; `aide/img/` pour les images. Livré dans le ZIP
  (`docs/` reste la doc développeur, exclue du ZIP).
- Lecture : `src/app/services/aide.py` (pur : chapitres, sections, `slug`, liens, `rubrique_depannage`,
  `nouveautes_markdown`) ; fenêtre `src/ui/dialogs/aide_dialog.py`.
- Tests de contrat : `python run_tests.py -k test_aide` — chaque étape a son chapitre
  (`CHAPITRE_PAR_ETAPE`), liens internes et images résolvent, chaque produit (`tag`) est nommé dans le
  chapitre de l'étape 2 et chaque entité (`label`) dans celui de l'étape 3, aucune citation anglaise,
  aucune version écrite à la main, chaque rubrique de `rubrique_depannage` existe dans Dépannage.

## Quel chapitre pour quel changement
| Changement | Chapitre |
|---|---|
| Installation, dépendance, proxy, mise à jour | `01-installer.md` |
| Mode, source, sélection des dalles, dossier de sortie | `03-etape-1-source.md` |
| Produit, réglage avancé, tuilage, diagnostic de contexte | `04-etape-2-produits.md` (sans décrire le produit : c'est sa fiche) |
| Entité, modèle, seuil, regroupement, fiabilité, fiche de classe | `05-etape-3-detection.md` |
| Préflight, workers, vue d'exécution, journal, relance | `06-etape-4-lancer.md` |
| Dossier de sortie, couches QGIS, légende, attributs | `07-resultats.md` |
| Nouveau message ⚠/✗ du narrateur avec une cause connue | `08-depannage.md` + une ligne dans `_RUBRIQUES_DEPANNAGE` (`aide.py`) |
| Terme nouveau pour l'utilisateur | `09-glossaire.md` |
| Ce qui a changé dans la version | rien : le chapitre Nouveautés est rendu depuis `changelog=` de `metadata.txt` |

## Écrire
- Vocabulaire du narrateur (`src/app/user_narrator.py`) et des écrans, pas celui du code. Pas de nom de
  module, de fonction ni de clé de config dans le texte ; une commande ou un chemin de fichier va en `code`.
- Jamais d'estimation de durée ni de RAM. Jamais de numéro de version (le test échoue).
- Ne pas redécrire un produit ni une classe : renvoyer à sa fiche (« chaque carte a sa fiche »).
- Les titres `##` sont les entrées du sommaire : courts, un par question.
- Lien interne : `[texte](etape-2-produits.md#tuilage-et-marge)` — clé = nom de fichier **sans** préfixe
  numérique, ancre = `slug(titre)` (sans accents, minuscules, tirets). Lien externe : `https://…` seulement.
- Citations traduites en français ; un nom propre reste en VO.

## Images
- Fichier dans `aide/img/`, PNG pour une fenêtre, JPEG (qualité 85) pour le canevas ou une photo.
- Citer l'image **liée à elle-même** pour l'ouverture en taille réelle :
  `[![légende](img/x.png)](img/x.png)`, sur sa propre ligne, précédée et suivie d'une ligne vide.
- **Jamais à l'intérieur d'un élément de liste** : Qt la rend en ligne avec le texte (ligne géante, trous).
  Placer l'image après la liste.
- Captures de l'assistant : hors écran, en double résolution, chemins neutres dans les champs :
  `C:/OSGeo4W/bin/python-qgis.bat dev/rendu_hors_ecran.py captures` (écrit `aide/img/etape*.png`).
  Captures qui exigent QGIS de bureau (canevas, résultats, étape 4 au vert) : demandées à l'utilisateur,
  déposées dans son dossier de captures Windows avec le nom attendu, puis converties et placées.
- Jamais l'adresse du dépôt, un identifiant ou un chemin personnel dans une image livrée : masquer.
- La fenêtre ramène toute image à la largeur de lecture et la lisse à la densité d'écran : pas de
  redimensionnement à faire à la main.

## Vérifier
1. `python run_tests.py -k test_aide` et `ruff check src/`.
2. Rendu hors écran si une image ou un tableau a changé :
   `python-qgis.bat dev/rendu_hors_ecran.py manuel --chapitre <cle> [--ancre <slug>]` → PNG à regarder,
   barre horizontale à 0 px.
3. Ajouter ou mettre à jour le point de recette dans `tests/TESTS_MANUELS_QGIS.md` (§38 pour le manuel).
4. Rappeler à l'utilisateur de recharger le plugin : les chapitres sont relus à chaque ouverture du
   manuel, mais le code de la fenêtre ne l'est pas.

## Hors périmètre
Le guide d'installation **avec l'adresse et les identifiants du dépôt** n'est pas dans le manuel ni dans
le dépôt public : il vit dans `dev/docs/_local/` et se relit à chaque livraison (runbook `/livraison`).
