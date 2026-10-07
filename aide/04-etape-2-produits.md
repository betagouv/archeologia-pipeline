# Étape 2 · Produits

Cochez les produits à calculer pour chaque dalle. Chaque carte porte une vignette et un lien **Fiche** : ce que montre l'image, dans quelle optique s'en servir, ce qu'elle ne montre pas, comment elle est calculée, ses réglages et ses sources. Avant de choisir entre deux indices, ouvrez **Comparer les produits**, en tête de la liste des fiches : deux tableaux issus de la documentation de Relief Visualization Toolbox disent quel produit convient à quel type de forme.

## Les produits

| Famille | Sigle | Ce que c'est |
|---|---|---|
| Base | MNT | Modèle numérique de terrain : l'altitude du sol nu, à partir des points classés sol. Base de tous les indices |
| Qualité | Densité | Nombre de points LiDAR sol par cellule : où la donnée est dense ou clairsemée |
| Qualité | Couverture | Part des cellules réellement appuyées sur des points sol : signale les zones où le MNT n'est qu'une interpolation. Modes LiDAR seulement |
| Relief | HS | Ombrage simple, depuis une seule direction de lumière |
| Relief | M-HS | Ombrage multi-directionnel : plusieurs éclairages combinés, pour le micro-relief |
| Relief | SVF | Facteur de vue du ciel : part de ciel visible en chaque point. Révèle creux, fossés et dépressions |
| Relief | OPNS | Ouverture du relief. Un réglage choisit le type : positive pour les saillies, négative pour les creux |
| Relief | SLO | Pente du terrain : ruptures de pente et talus |
| Relief | LD | Dominance locale : fait ressortir les structures en relief |
| Relief | SLRM | Modèle de relief local : soustrait le relief général pour isoler le micro-relief |
| Composition | VAT | Visualisation archéologique : combinaison d'indices optimisée pour la prospection |
| Composition | CVAT | VAT combiné : fusion des variantes terrain général et terrain plat |
| Composition | MSTP | Position topographique multi-échelle : trois échelles composées en une image couleur |
| Composition | PRISM | Ouverture prismatique : ouvertures positive et négative sur un ombrage multi-directionnel, en couleurs |
| Composition | CRIM | Relief coloré : le micro-relief en gris, la pente en couleurs |

Les détails de calcul et de lecture sont dans la fiche de chaque produit, pas ici.

## Ce qu'il faut savoir avant de cocher

- **Le MNT est requis** tant qu'un indice de relief est coché : il en est la source. Pour le décocher, décochez d'abord les indices.
- **Un modèle de détection a besoin d'un indice précis.** À l'étape 3, chaque entité indique l'indice attendu ; s'il n'est pas coché ici, un bouton **+ Activer** l'ajoute.
- **Chaque produit coché est calculé pour chaque dalle.** Décocher les indices que vous ne regarderez pas raccourcit le traitement.
- Les compositions (VAT, CVAT, PRISM, CRIM) sont calculées à partir des mêmes ingrédients que les indices simples ; les cocher n'exige pas de cocher leurs ingrédients.

## Réglages avancés

Le bouton **Réglages avancés…** ouvre une vue à onglets : un onglet par produit coché, plus les onglets communs. Chaque onglet commence par une phrase qui dit ce que fait le produit ; sous chaque réglage, une ligne dit à quoi il sert. Un bouton **Réinitialiser** par onglet remet les valeurs par défaut de ce seul produit.

Les réglages qui comptent le plus :

- **Résolution du MNT** (mètre par pixel). Base de tous les indices ; une résolution plus grossière accélère beaucoup le calcul. Les rayons des indices sont exprimés en pixels : changer la résolution change donc leur portée sur le terrain, ce que chaque fiche rappelle.
- **Filtre des points** : les classes de points conservées pour reconstruire le sol (sol, bâti, eau…), au format de PDAL.
- **Résolution de la densité** et **seuil de couverture** : le pourcentage de cellules sans point sol au-delà duquel une zone est marquée mal couverte.
- Les réglages propres à chaque indice : direction et hauteur de lumière, rayons, nombre de directions, exagération verticale, sortie 8 bits. Certains ont des bornes fixées par l'algorithme, rappelées sous le champ ; hors de ces bornes, la dalle échouerait.

## Tuilage et marge

La carte **Tuilage** règle la **marge** ajoutée autour de chaque dalle lors du calcul des indices. Les dalles voisines sont fusionnées avec cette marge, puis chaque indice est recadré sur la dalle : les indices se raccordent sans couture et les objets en bordure sont vus entiers par la détection.

Sous les réglages, une ligne de diagnostic compare cette marge au rayon le plus grand demandé par vos indices :

- **✓ Contexte fourni aux noyaux** : la marge couvre le rayon, rien à faire.
- **⚠** suivi du nom d'un produit : son rayon dépasse la marge. Au-delà de la marge, l'algorithme reconstruit le voisinage par symétrie et les dalles ne se raccordent plus. Réduisez le rayon indiqué, ou augmentez la marge.

En MNT existant, chaque raster est calculé sans voisin : le diagnostic dit quelle part de l'emprise aura un voisinage complet. Sur un lot, le réglage vaut pour toutes les dalles ; ne réduisez un rayon que si la plupart des rasters ont cette taille.

## Mode MNT existant

Les sections MNT et Densité sont masquées : ces produits sont vos données d'entrée. La couverture, qui se calcule depuis les points, n'est pas disponible.
