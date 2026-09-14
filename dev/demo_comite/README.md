# Démo comité d'investissement — onglet « Visualisation »

Onglet ajouté au dialogue du plugin, d'après `visu_handoff/VISUALISATION_IMPLEMENTATION.md`
(Claude Design) : parcourir le catalogue France entière des indices déjà produits
et les ouvrir dans QGIS d'un clic, sans relancer de pipeline.

## Le geste de la démo (3 minutes)

1. Ouvrir le plugin → l'onglet **Visualisation** (2ᵉ onglet).
   Il s'ouvre sur **Ille-et-Vilaine**, 12 indices, mur plein.
2. **« Le relief nu » → Afficher dans QGIS.** La carte se cale sur le bloc de
   6 × 5 km à l'est de Rennes. Un dégradé, presque rien d'autre : c'est le propos.
3. **« Structures en relief » (LD) → Afficher dans QGIS.** Le réseau de talwegs,
   les terrasses et le parcellaire surgissent. Même sol, autre lecture.
4. **« Voir dans QGIS → »** (en bas à droite) range la fenêtre et laisse la carte.
5. Rouvrir le plugin, filtrer « morbihan » ou cliquer un autre département :
   le mur montre la couverture, département par département.

Le contraste **MNT → LD** est l'argument. Ne pas le rater.

## Ce qui est vrai, ce qui est une maquette

| | |
|---|---|
| **Vrai** | Les **12 vignettes** sont des rendus réels de la dalle `0390_6818`, une image par indice, **même fenêtre de 800 × 448 m** — c'est ce qui permet de reconnaître un indice à son rendu. |
| **Vrai** | La dalle des vignettes est **dans** le bloc affiché : la vignette est un extrait de ce que « Afficher dans QGIS » va charger, à un autre zoom. |
| **Vrai** | « Afficher dans QGIS » charge une **vraie couche raster** depuis les mosaïques VRT calculées par le pipeline. |
| **Vrai** | 2 départements portent leurs propres données : **35** (jeu `demo_comite`) et **78** (run forêt de Saint-Germain). |
| **Maquette** | La couverture des 83 autres départements et les volumes affichés. Ils réutilisent les rasters de `demo_comite`. |
| **Maquette** | Il n'y a **pas** de diffusion en flux : les sources sont des fichiers locaux. L'écran le dit (« Aperçu local », « source locale ») plutôt que de laisser croire le contraire. |

Le mur montre les **12 produits** du pipeline, `DENSITE` et `COUVERTURE` compris.

## Regénérer

```bash
# vignettes : une image par indice, même dalle, depuis les vraies données
.venv_dev/Scripts/python.exe dev/demo_comite/build_thumbs.py
#   --run <dossier>    run source (défaut : demo_comite)
#   --tile 0390_6818   dalle à rendre ; vide = choix au score de texture
#   --planche x.png    planche-contact des 30 dalles, pour choisir à l'œil
#
# ⚠ Le score de texture vise les BOURGS : le bâti sature le LD et gagne à tous
#   les coups. Il donne un point de départ, jamais un verdict — regarder les
#   dalles et en imposer une avec --tile.

# catalogue : 101 départements, couverture déterministe, sources = VRT réels
.venv_dev/Scripts/python.exe dev/demo_comite/build_catalogue.py
#   --src D:/pipeline_results

# vérification hors écran (rend 3 PNG + teste l'ouverture de couches)
& "C:\Program Files\QGIS 4.0.3\bin\python-qgis.bat" dev\demo_comite\smoke_visu.py
```

Ces deux scripts sont le seul filet sur cet onglet : `src/ui/` n'est pas collecté
par pytest (pas de QGIS en autonome), donc une régression d'énuméré Qt6 ou de QSS
ne se verrait qu'au lancement. `smoke_visu.py` monte l'onglet, ouvre deux couches,
en retire une ; `smoke_dialog.py` monte le dialogue entier, mesure la largeur des
onglets et cherche les textes rognés.

La graisse de police conditionnée à un état QSS, elle, est verrouillée côté pytest
par `tests/unit/test_qss_graisse_etats.py` — un contrôle au rendu ne l'attrape pas
(`sizeHint()` ignore le gras venu du QSS).

## Si ça casse en salle

- **Mur vide / « Catalogue injoignable »** → `data/demo_catalogue/catalogue.json`
  absent : rejouer `build_catalogue.py`.
- **Vignettes grises** → `data/demo_catalogue/thumbs/*.png` absents : `build_thumbs.py`.
- **« Impossible d'ouvrir … »** dans la barre de messages QGIS → le disque `D:`
  n'est pas monté, ou les VRT ont bougé. Les chemins sont **absolus** dans le
  catalogue : c'est une démo, pas un livrable.
- **Rien ne s'affiche après le clic** → la couche est chargée mais le canevas est
  ailleurs : clic droit sur la couche → « Zoomer sur la couche ». ⚠ La mosaïque
  couvre tout le run, pas seulement la fenêtre visée : « Zoomer sur la couche »
  montrera donc des dalles éparpillées. Repasser par une carte du mur pour
  retrouver le bon cadrage.

## Le jeu de données

`D:\pipeline_results\demo_comite` a été **calculé pour cette démo** par
`run_pipeline.py` : 30 dalles **jointives** (6 × 5 km, Ille-et-Vilaine, à l'est de
Rennes) avec les **12 produits**. Emprise Lambert-93 : `[388000, 6815000, 394000,
6820000]`.

Il existe parce que les jeux précédents imposaient un compromis : les denses
n'avaient qu'un indice (Dreux 75 dalles / 1, Fénétrange 42 / 1), et le seul riche
— la Bretagne, 11 indices — est **dispersé** (1 % de remplissage sur 231 × 119 km ;
même son plus gros bloc contigu n'est rempli qu'à 37 %). S'y recadrer donnait un
écran de dalles éparpillées.

| département | source | fenêtre | dalles | indices |
|---|---|---|---|---|
| 35 Ille-et-Vilaine (vitrine) | `demo_comite` | 6 × 5 km, sans trou | 30 | **12** |
| les autres | `demo_comite` | idem | 30 | 3 à 12 |
| 78 Yvelines | Saint-Germain | 6 × 5 km, sans trou | 30 | 3 |

Refaire le jeu ailleurs :

```powershell
& "C:\Program Files\QGIS 4.0.3in\python-qgis.bat" dev\demo_comite
un_pipeline.py `
    --x0 388 --x1 393 --y0 6816 --y1 6820 --out D:/pipeline_results/demo_comite
#   --dry-run   n'écrit que la liste de dalles
#   --tiles N   se limite à N dalles (répétition à blanc)
```

Coût **mesuré** : 2 dalles / 12 produits = 7 min 46 et 1,1 Go ; les 30 dalles =
**2 h 59 et 27 Go**, dont 6 Go de LAZ téléchargés (le reste est du LAZ fusionné
intermédiaire, supprimable).

## Ce qui a été volontairement laissé de côté

Marqué `ponytail:` dans le code, avec la voie de sortie :

- **Vignettes synchrones** (`visualisation_tab._pixmap_for`) — fichiers locaux,
  donc instantanés. Un catalogue à vignettes distantes demandera un
  `QNetworkAccessManager` + cache disque (contrat §2.7).
- **Couche construite sur le thread principal** (`_open_indice`) — le brief
  suggérait un `QgsTask` ; un `QgsRasterLayer` n'est pas sûr à construire hors
  thread principal. Pour de vraies sources distantes, la voie propre est un
  `QgsTask` qui **valide l'URL**, la couche restant construite sur le thread UI.
- Pas de `.qml` par indice : le rendu par défaut de QGIS suffit sur ces rasters
  (uint8 0–255 pour les indices, étirement automatique pour le MNT float).
- Pas de rechargement réseau du catalogue : « Actualiser le catalogue » relit le
  fichier local.
