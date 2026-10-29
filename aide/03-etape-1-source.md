# Étape 1 · Vos données

La frise du haut représente la chaîne de traitement (le [schéma complet](premier-traitement.md#ce-que-fait-le-plugin) est au chapitre Premier traitement). Cliquez sur ce que vous possédez déjà : c'est le **point de départ** du traitement, marqué d'un badge et rappelé dans un bandeau. Les étapes qui le précèdent sont ignorées et disent « sautée » : le traitement ne passe pas par là. Le champ de vos fichiers s'adapte. Il n'y a pas de liste déroulante ; au clavier, la touche Tab passe d'un point de départ à l'autre et Espace le choisit.

[![Étape 1 : la frise des points d'entrée, le bandeau du point de départ, la source et le dossier de sortie](img/etape1-source.png)](img/etape1-source.png)

## Les quatre points de départ

| Point de départ | Vous fournissez | Internet |
|---|---|---|
| Dalles IGN à télécharger | une zone d'étude, une liste de dalles, ou des dalles choisies sur la carte | oui |
| Nuages LiDAR | un dossier de fichiers `.laz` ou `.las` | non |
| MNT existant | un dossier de modèles de terrain `.tif`, `.tiff` ou `.asc` en Lambert-93 | non |
| Indices de visualisation existants | un dossier d'indices (SVF, LD…) `.tif` en Lambert-93 | non |

Les deux premiers points de départ exécutent toute la chaîne. Depuis un MNT existant, le traitement commence aux indices de visualisation. Depuis des indices de visualisation existants, il ne calcule aucun produit : avec une structure cochée, il les analyse ; sans détection, il se contente de les publier en mosaïque dans `livrable/images/RVT/`.

Dans tous les cas, indiquez un **dossier de sortie**, de préférence vide. Relancer dans le même dossier est possible : voir [Relancer dans le même dossier](etape-5-traitement.md#relancer-dans-le-meme-dossier).

Sous chaque champ, une ligne dit ce qui manque et le geste à faire : « Indiquez la zone d'étude… », « Ce dossier est introuvable : vérifiez le chemin », « Ce dossier est dans le dossier de vos données : choisissez un dossier de sortie à part ». Elle disparaît quand le champ est bon, et dit « 3 dalles choisies sur la carte » après une sélection sur la carte. Le rail de gauche marque « à corriger » l'étape qui porte encore un problème.

## Téléchargement IGN

Le plugin identifie les dalles LiDAR HD de l'IGN qui couvrent votre zone, les télécharge depuis la Géoplateforme, puis traite. Trois façons de désigner les dalles :

- **Choisir sur la carte** : la grille nationale s'affiche sur le canevas de QGIS. Elle n'apparaît qu'à une échelle assez fine : zoomez sur votre secteur jusqu'à ce que les dalles soient dessinées (la barre de messages indique l'échelle requise). Cliquez une dalle pour la prendre ou la retirer, encadrez à la souris pour en ajouter plusieurs, **Ctrl** + encadrer pour en retirer, **Échap** pour abandonner. **Valider** enregistre la liste, avec une estimation du volume à télécharger. Pendant le choix, l'assistant ne se réduit pas : il se range contre le bord droit de l'écran, à sa largeur minimale, pour laisser le canevas libre, et reprend sa place à la fin.
- **Zone d'étude** : un fichier vectoriel (`.shp`, `.gpkg`, `.geojson`), polygone ou points, dans n'importe quelle projection. Toutes les dalles qui touchent l'emprise sont retenues.
- **Liste de dalles** : un fichier `.txt`, une dalle par ligne au format `nom,url`, tel que l'écrit la sélection sur carte ou tel que le fournit le site de téléchargement de l'IGN.

[![La grille des dalles sur le canevas de QGIS, avec la barre de messages de la sélection](img/etape1-selection-dalles.jpg)](img/etape1-selection-dalles.jpg)

La grille des dalles est livrée avec le plugin et remise à jour à chaque version. Le bandeau du point de départ indique sa date et son nombre de dalles : les dalles publiées par l'IGN depuis cette date n'y figurent pas encore.

Chaque dalle couvre un kilomètre carré. Les dalles voisines sont fusionnées avant le calcul des produits, avec une marge, pour que les produits se raccordent sans couture : voir [Marge entre dalles](etape-2-produits.md#marge-entre-dalles).

## Nuages locaux

Vous avez déjà les fichiers `.laz` ou `.las`. Le plugin lit le dossier, reconnaît les dalles IGN à leur nom, fusionne les voisines et applique le même traitement que le point de départ Téléchargement IGN. Les fichiers doivent être en Lambert-93 ; ce n'est pas vérifié : un nuage dans une autre projection donne un MNT faux, sans message.

## MNT existant

Vous fournissez un dossier de modèles de terrain. Le plugin calcule les produits cochés à l'étape 2 puis, si elle est activée, la détection. Le relief nu et la qualité de la donnée ne sont pas proposés à l'étape 2 : le MNT existe déjà.

Les rasters doivent être en **Lambert-93 (EPSG:2154)**. Un raster dans une autre projection est refusé à la vérification de l'étape 4, avec un message qui le nomme : reprojetez-le avant.

Le plugin inspecte l'emprise de chaque raster :

- **dalle IGN** d'un kilomètre, alignée sur la grille : traitement habituel ;
- **petit raster**, moins d'un kilomètre ou non aligné : l'emprise est conservée telle quelle ;
- **grand raster**, plus d'un kilomètre dans une dimension : aucun pré-découpage, les produits sont calculés sur toute l'emprise et la détection découpe elle-même l'indice en fenêtres d'analyse.

Chaque raster est calculé seul, sans voisin : sur un lot de petites dalles, les indices à grand rayon peuvent montrer une couture entre deux dalles. La carte **Marge entre dalles** de l'étape 2 signale quand le rayon d'un indice dépasse ce que la dalle peut fournir.

## Indices de visualisation existants

Vous avez déjà les indices de visualisation (SVF, LD, M-HS…), par exemple produits par un traitement précédent ou par un autre outil. Le dossier est lu tel quel, les indices sont préparés pour les modèles et la détection s'exécute. Même exigence de projection : Lambert-93. Les résultats sont rangés sous `livrable/images/RVT/`, le plugin ne connaissant pas les réglages qui ont produit ces images.

Pour que les objets à cheval sur deux dalles soient vus entiers, le plugin fabrique une marge à partir des dalles voisines présentes dans le dossier. Une dalle isolée n'en a pas.

## Enregistrer et recharger une configuration

Dans l'en-tête de l'assistant, le menu **Configuration ▾** réunit tout ce qui touche à vos réglages : **Enregistrer sous…** garde l'ensemble de vos choix sous un nom ; les configurations enregistrées se chargent d'un clic ; **Gérer les configurations…** les renomme ou les supprime ; **Importer un fichier…** lit une configuration reçue d'un collègue ; **Tout remettre aux valeurs par défaut…** revient aux réglages d'origine après confirmation. Ces configurations vivent dans votre profil QGIS, pas dans le dossier du plugin : elles survivent aux mises à jour. Pendant un traitement, charger, importer et tout remettre sont grisés ; enregistrer reste possible.
