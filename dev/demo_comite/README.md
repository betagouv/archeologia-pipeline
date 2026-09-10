# Démo comité d'investissement — onglet « Visualisation »

Onglet ajouté au dialogue du plugin, d'après `visu_handoff/VISUALISATION_IMPLEMENTATION.md`
(Claude Design) : parcourir le catalogue France entière des indices déjà produits
et les ouvrir dans QGIS d'un clic, sans relancer de pipeline.

## Le geste de la démo (3 minutes)

1. Ouvrir le plugin → l'onglet **Visualisation** (2ᵉ onglet).
   Il s'ouvre sur **Ille-et-Vilaine**, 10 indices, mur plein.
2. **« Le relief nu » → Afficher dans QGIS.** La carte se cale sur une vallée
   encaissée de 3 × 3 km à l'est de Rennes. On ne voit presque rien : c'est le propos.
3. **« Structures en relief » (LD) → Afficher dans QGIS.** Le réseau de talwegs,
   les terrasses et le parcellaire surgissent. Même sol, autre lecture.
4. **« Voir dans QGIS → »** (en bas à droite) range la fenêtre et laisse la carte.
5. Rouvrir le plugin, filtrer « morbihan » ou cliquer un autre département :
   le mur montre la couverture, département par département.

Le contraste **MNT → LD** est l'argument. Ne pas le rater.

## Ce qui est vrai, ce qui est une maquette

| | |
|---|---|
| **Vrai** | Les 10 vignettes sont des rendus réels d'une dalle LiDAR HD (Bretagne, `0362_6800` — une étoile forestière), une image par indice, même emprise. |
| **Nuance** | La vignette et la couche chargée ne montrent pas le même endroit : la dalle de l'étoile n'est dans aucune fenêtre sans trou. La vignette est un échantillon de l'indice, pas un aperçu de la zone. |
| **Vrai** | « Afficher dans QGIS » charge une **vraie couche raster** depuis les mosaïques VRT calculées par le pipeline. |
| **Vrai** | 2 départements portent leurs propres données : **35** (run Bretagne) et **78** (run forêt de Saint-Germain). |
| **Maquette** | La couverture des 83 autres départements et les volumes affichés. Ils réutilisent les rasters bretons. |
| **Maquette** | Il n'y a **pas** de diffusion en flux : les sources sont des fichiers locaux. L'écran le dit (« Aperçu local », « source locale ») plutôt que de laisser croire le contraire. |

**Deux indices manquent** : `DENSITE` et `COUVERTURE` (famille Qualité) ne sont
calculés nulle part dans `D:\pipeline_results`, donc absents du catalogue — le mur
en montre 10, pas 12. Pour les avoir, il faut relancer le pipeline sur une dalle
avec ces deux produits cochés, puis rejouer `build_thumbs.py` et `build_catalogue.py`.

## Regénérer

```bash
# vignettes : une image par indice, même dalle, depuis les vraies données
.venv_dev/Scripts/python.exe dev/demo_comite/build_thumbs.py
#   --tile 0362_6800   dalle à rendre (défaut : l'étoile forestière retenue)
#   --pick             rechoisir au score de texture — vise les bourgs, à éviter

# catalogue : 101 départements, couverture déterministe, sources = VRT réels
.venv_dev/Scripts/python.exe dev/demo_comite/build_catalogue.py
#   --src D:/pipeline_results

# vérification hors écran (rend 3 PNG + teste l'ouverture de couches)
& "C:\Program Files\QGIS 4.0.3\bin\python-qgis.bat" dev\demo_comite\smoke_visu.py
```

`smoke_visu.py` est le seul filet sur cet onglet : `src/ui/` n'est pas collecté
par pytest (pas de QGIS en autonome), donc une régression d'énuméré Qt6 ou de QSS
ne se verrait qu'au lancement. Il monte l'onglet dans le Python de QGIS 4, ouvre
deux couches, en retire une, et sort `RESULTAT : OK`.

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

## Pourquoi les emprises sont si petites

Les 201 dalles bretonnes sont **dispersées** : 1 % de remplissage sur 231 × 119 km,
et même leur plus gros bloc contigu (33 dalles, 9 × 10 km) n'est rempli qu'à 37 %.
Se recadrer dessus donnait un écran de dalles éparpillées. Les emprises du
catalogue sont donc les plus grands **rectangles sans trou**, calculés sur les
dalles présentes dans *tous* les indices du run :

| département | fenêtre | dalles | indices |
|---|---|---|---|
| 35 Ille-et-Vilaine (vitrine) | vallée, 3 × 3 km | 9 | 10 |
| les autres | bocage, 5 × 2 km | 10 | 3 à 10 |
| 78 Yvelines | Saint-Germain, 6 × 5 km | 30 | 3 |

Le compromis est structurel : dans `D:\pipeline_results`, les jeux denses ont peu
d'indices (Dreux 75 dalles / 1 indice, Fénétrange 42 / 1) et le seul jeu riche
— la Bretagne, 11 indices — est dispersé. Un run dense **et** complet lèverait
la contrainte : ~30 dalles jointives avec les 12 produits cochés.

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
