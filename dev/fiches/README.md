# Outillage des fiches

Deux familles de fiches, deux chaînes d'outils :

- **Fiches de classes** (étape 3) — ce que l'archéologue lit avant de cocher une
  *entité* à détecter : bloc `classes[].fiche` d'un `model_card.yaml`.
- **Fiches de produits** (étape 2) — ce qu'il lit avant de cocher un *produit* à
  calculer : `data/indices_fiches.json`. Voir la seconde moitié de ce fichier.

## Fiches de classes

Scripts d'appoint pour écrire le bloc `classes[].fiche` d'un `model_card.yaml` :
illustration, provenance des données d'entraînement, hors-cible, contexte
d'usage.

La procédure complète (dont l'ordre des étapes et les règles de contenu) est
dans [`docs/model_contract.md`](../../docs/model_contract.md), section
« Produire une fiche de classe ». Ici, seulement les outils.

Tous écrivent **textuellement** dans les `model_card.yaml` : un round-trip
PyYAML effacerait les commentaires qui documentent le choix des seuils et le
calibrage de la fiabilité. Chacun relit ensuite le fichier avec PyYAML pour
vérifier que ce qui a été écrit se relit à l'identique.

## `injecter_fiches.py`

```bash
python dev/fiches/injecter_fiches.py mes_fiches.json
```

`mes_fiches.json` = liste de `{modele, classe, fiche}`, où `fiche` est le bloc
complet (`resume`, `reconnaitre`, `usage`, `hors_cible`, `vignettes`,
`entrainement`). Une classe qui porte déjà un bloc est laissée intacte et
signalée. `classe` peut aussi être l'`output_class` d'une **cible dérivée**
(`derived_targets:` du `model_card`, ex. `zone_crateres`) : le bloc est alors
inséré sous cette entrée, même format.

## `candidats_corpus.py` — six candidats tirés du corpus

```bash
python dev/fiches/candidats_corpus.py C:/projets/Archeologia/training-models/corpus/<corpus>     --categorie <catégorie COCO> --sortie D:/brouillons/vignettes_candidates/<classe>/<campagne>     [--n 6] [--splits train] [--eviter-zone <zone de l'image déjà installée>] [--gsd 0.5]
```

Choisit les tuiles porteuses de la classe en variant les zones (celle passée en
`--eviter-zone` vient en dernier), cadre la plus petite fenêtre carrée qui garde le
plus d'objets entiers, écarte NoData et tuiles plates, écrit `N_brut.jpg`,
`N_annote.jpg` (512 px, polygones jaunes), `candidats.json` et une
`planche_candidats.png`. La planche se regarde avant de publier : l'algorithme
compte, il ne juge pas la lisibilité (une zone de stries annotées en peigne a été
écartée à la main le 2026-09-15).

## `page_choix.py` / `appliquer_choix.py` — le choix se fait dans le navigateur

```bash
python dev/fiches/page_choix.py D:/brouillons/vignettes_candidates/<classe>/     --classe <classe> --label "<label_fr>" --modele <modele>     --corpus "<d'où viennent les cadres>" --objets "<objets comptés>" --sortie page.html
```

Le dossier contient `N_brut.jpg`, `N_annote.jpg` et `candidats.json` (liste de
`{candidat, zone|secteur, profil|pourquoi, n_crateres|objets, emprise_m, tuiles|cible}`).
La page (images en base64, ~1,5 Mo pour six cadres) se publie avec l'outil
Artifact de Claude Code, capacité `db` déclarée : l'utilisateur clique les cadres
à retenir (l'ordre des clics fait l'ordre de la fiche, le premier est l'icône),
puis pose la fenêtre de l'icône sur chacun avec l'aperçu à 44 px. Chaque geste
enregistre le document `choix/<classe>` = `{classe, modele, retenues, cadrages}`
dans la base de l'artefact ; Claude le relit (`read_db`, collection `choix`) et
le sauve en JSON, puis :

```bash
python dev/fiches/appliquer_choix.py choix_<classe>.json D:/brouillons/vignettes_candidates/<classe>/
python dev/fiches/injecter_cadrages.py D:/brouillons/vignettes_candidates/<classe>/cadrages_<classe>.json planche.png
```

`appliquer_choix.py` copie les cadres retenus dans `vignettes/<classe>_0k_*.jpg`
et écrit le JSON de cadrage de l'icône ; les légendes des `vignettes[]` de la
fiche restent à rédiger. **Fiche qui a déjà son icône** (il en faut au moins deux
images) : `page_choix.py --complement` (icône conservée, pas d'étape de cadrage)
puis `appliquer_choix.py --rang-depart 2` (les cadres retenus deviennent 02, 03…,
sans cadrage). Historique : pages « Vignette des tranchées » (2026-09-10)
et « Vignettes · Regroupement de cratères » (2026-09-15).

## `ajouter_vignettes.py` — compléter une fiche déjà écrite

```bash
python dev/fiches/ajouter_vignettes.py ajouts.json
```

`ajouts.json` = liste de `{modele, classe, vignettes: [{brut, annote, zone, legende}]}`.
`injecter_fiches.py` refuse d'écraser une fiche existante ; ce script n'ajoute que des
entrées à sa liste `vignettes` (même chirurgie textuelle, commentaires préservés, une
vignette déjà déclarée est ignorée) puis relit le YAML et vérifie que chaque fichier
existe. C'est l'étape qui suit `appliquer_choix.py --rang-depart 2`.

## `injecter_cadrages.py`

```bash
python dev/fiches/injecter_cadrages.py mes_cadrages.json [planche.png]
```

`mes_cadrages.json` = liste de `{modele, vignette, cadrage: {x, y, cote}}`, en
fractions de l'image. Écrit la clé et rend une **planche de contrôle** : chaque
icône à sa taille réelle de 44 px, puis agrandie. Regarder la planche fait
partie du travail — un cadrage trop large redonne la bouillie grise qu'on
cherchait à éviter.

## `comparer_drive.py` / `reporter_sur_drive.py`

`data/models/**` est **gitignoré** : une fiche écrite dans le plugin n'est pas
versionnée, et le prochain ré-export du modèle l'effacerait. La copie qui fait
foi pour un ré-export est celle de l'archive,
`runs/training/<RUN_ID>/package/model_card.yaml`.

```bash
python dev/fiches/comparer_drive.py     # lecture seule : où les deux copies divergent
python dev/fiches/reporter_sur_drive.py # reporte fiches + vignettes vers l'archive
```

`bundles_drive.json` fait la correspondance `<modèle installé>` → chemin du
bundle dans l'archive, relatif à la racine `model-training`. Ajouter une ligne
par nouveau modèle.

⚠️ `reporter_sur_drive.py` ne synchronise que les blocs `fiche` et les
`vignettes/`. Toute **autre** correction faite au `model_card.yaml` du plugin
(un `known_limitations` rectifié, une clé `fiabilite.source` ajoutée…) reste à
reporter à la main — `comparer_drive.py` la signale, en distinguant bien
« seulement dans le plugin » de « seulement sur Drive ».

⚠️ L'archive est sur un disque Google Drive en streaming : le montage peut
disparaître en cours de route (vu le 2026-09-10). `comparer_drive.py` le dit
franchement (`Drive : ABSENT`) plutôt que de laisser croire à une divergence.
Relancer quand le disque est revenu — les deux scripts sont idempotents.

---

# Fiches de produits (étape 2)

Les douze produits de l'étape 2 — modèle d'altitude, densité, couverture et les
neuf indices de visualisation — ont chacun leur fiche : ce que montre l'image, à
quoi elle sert, ce qu'elle ne montre **pas**, comment elle est calculée, avec
quels réglages, d'après quelles sources.

Le texte vit dans `data/indices_fiches.json` (versionné, contrat et coercitions
dans `src/app/services/indice_fiche.py`), les images dans
`data/indices_vignettes/`. `tests/unit/test_indice_fiche.py` échoue si un
produit du pipeline n'a pas sa fiche complète, si un paramètre cité n'existe pas
dans la configuration, ou si une vignette déclarée est absente du disque.

**Ce qui distingue ces vignettes de celles des classes** : une classe se montre
sur des cadres variés, un produit se montre sur **le même terrain que les onze
autres**. C'est en comparant la même parcelle en Sky-View Factor et en Local
Dominance qu'on comprend lequel cocher. La chaîne choisit donc des *fenêtres de
terrain*, pas des cadres par produit.

## `candidats_indices.py` — tirer des fenêtres d'un run de référence

```bash
python dev/fiches/candidats_indices.py D:/pipeline_results/demo_comite     --sortie D:/brouillons/vignettes_indices [--n 6] [--cote-m 324] [--reference SLRM]
```

`<run>` est n'importe quel dossier de sortie du pipeline contenant `indices/`
avec les douze produits. L'outil note chaque fenêtre sur le micro-relief et la
densité de contours, en écartant le bâti (qui sature toujours plus fort qu'un
talus), puis retient une fenêtre par dalle : du bocage, plus **une fenêtre
bâtie** (pour illustrer honnêtement les toits que le filtre par défaut laisse
dans le modèle d'altitude) et **la dalle la moins couverte** (seule façon
d'illustrer Couverture autrement que par un aplat).

⚠️ La fenêtre est exprimée en **mètres**, pas en pixels : la densité et la
couverture sont au pas de densité (1 m) quand les indices sont au pas du modèle
d'altitude (0,5 m). En pixels, ces deux produits cadreraient deux fois plus de
terrain et les vignettes ne seraient plus comparables.

⚠️ Le run de référence (`demo_comite`, fenêtre `LHD_FXX_0392_6818` à
`xmin=392000, ymax=6817352`) date d'avant le produit **OPNS** : il n'a pas de
dossier `indices/OPNS_*`, et les deux vignettes d'openness ont donc été
recalculées hors chaîne le 2026-09-21 (paquet `rvt` de rvt-qgis sur le MNT du
run, même étirement 8 bits, contrôle : le SVF recalculé de la même façon
retrouve la vignette livrée à 0,995 de corrélation). **À la prochaine
régénération**, relancer d'abord `demo_comite` avec OPNS coché (un type par run,
les deux sont attendus) pour que la chaîne le couvre normalement.

Écrit `NN_<CLÉ>.jpg` (douze par fenêtre), `candidats.json` et
`planche_candidats.png`. **Regarder la planche avant de publier la page** :
l'algorithme note, il ne juge pas l'intérêt archéologique.

## `page_choix_indices.py` / `appliquer_choix_indices.py` — le choix au navigateur

```bash
python dev/fiches/page_choix_indices.py D:/brouillons/vignettes_indices     --sortie page_indices.html [--taille 384]
```

La page se publie avec l'outil Artifact, capacité `db` déclarée. Elle présente
les fenêtres candidates, une grille produits × fenêtres pour donner sa propre
fenêtre à un produit particulier, et le cadrage de l'icône de 44 px — un par
fenêtre retenue. Chaque geste enregistre `choix/indices` =
`{produits: {clé: n°}, cadrages: {n°: {x, y, cote}}}` ; Claude le relit
(`read_db`, collection `choix`, document `indices`) et le sauve en JSON, puis :

```bash
python dev/fiches/appliquer_choix_indices.py choix_indices.json     D:/brouillons/vignettes_indices [--legendes legendes.json] [--verifier]
```

`appliquer_choix_indices.py` copie chaque cadre retenu dans
`data/indices_vignettes/<CLÉ>.jpg` et déclare la vignette dans
`data/indices_fiches.json` avec son cadrage, sa provenance (dalle et emprise) et
sa licence. `--verifier` n'écrit rien et dit ce qui serait fait. Les **légendes**
sont le seul texte éditorial de la vignette : elles se rédigent à la main en
regardant l'image, et se passent par `--legendes` (`{clé: "phrase"}`) ; sans
elles la fiche reste valide, l'outil dit lesquelles manquent.

Pas de chirurgie textuelle ici, contrairement aux `model_card.yaml` :
`indices_fiches.json` est du JSON pur, sans commentaires à préserver.
