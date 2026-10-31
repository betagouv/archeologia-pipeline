# Étape 2 · Visualisation du relief

Cochez les **produits** à calculer pour chaque dalle : tout ce qui rend le relief lisible. Trois sections : **Recommandé pour commencer** (le relief nu MNT, les creux SVF, les reliefs LD, cochés d'avance à la première ouverture), **Autres indices de visualisation** (les autres produits que la Relief Visualization Toolbox calcule depuis le MNT) et **Qualité de la donnée** (densité et couverture des points sol, pour les points de départ LiDAR). Un clic n'importe où sur une carte la coche ou la décoche ; au clavier, Tab passe de carte en carte et Espace coche. Le bouton **ⓘ** de chaque carte ouvre sa fiche dans une fenêtre à part, qui ne bloque rien (Fermer ou Échap) : ce que montre le produit, dans quelle optique s'en servir, ce qu'il ne montre pas, comment il est calculé, ses réglages et ses sources. Avant de choisir entre deux indices, le lien **Comparer les produits** en haut de la page ouvre deux tableaux issus de la documentation de Relief Visualization Toolbox, qui disent quel indice convient à quel type de forme.

[![Étape 2 : les produits recommandés cochés d'avance, les autres indices de visualisation et la qualité de la donnée, chaque carte avec sa vignette et son « ⓘ »](img/etape2-produits.png)](img/etape2-produits.png)

## Les produits

| Famille | Sigle | Ce que c'est |
|---|---|---|
| Base | MNT | Modèle numérique de terrain : l'altitude du sol nu, à partir des points classés sol. Base de tous les autres produits |
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

Les familles Indice et Composition sont les **indices de visualisation** ; les détails de calcul et de lecture sont dans la fiche de chaque produit, pas ici.

[![La fiche d'un produit dans sa fenêtre : ce qu'il montre, dans quelle optique l'utiliser ; la liste en tête passe aux autres produits](img/etape2-fiche-produit.png)](img/etape2-fiche-produit.png)

## Ce qu'il faut savoir avant de cocher

- **Le MNT est nécessaire** tant qu'un indice de visualisation est coché : il en est la source. Sa carte porte alors un cadenas et dit, en infobulle, quels indices le retiennent ; pour le décocher, décochez-les d'abord.
- **Un modèle de détection a besoin d'un indice précis.** À l'étape 3, chaque structure indique l'indice attendu ; s'il n'est pas coché ici, un lien **l'ajouter** le coche.
- **Chaque produit coché est calculé pour chaque dalle.** Décocher les produits que vous ne regarderez pas raccourcit le traitement.
- Les compositions (VAT, CVAT, PRISM, CRIM) sont calculées à partir des mêmes ingrédients que les indices simples ; les cocher n'exige pas de cocher leurs ingrédients.

## Réglages

Une carte cochée porte le lien **▸ Réglages (n)** : il déplie ses réglages dans la carte, sous la description, sans quitter la page ; la sélection reste visible et un clic dans les réglages règle sans décocher. Sous chaque réglage, une ligne dit à quoi il sert. Le bouton **↺** de la carte, « Revenir aux valeurs par défaut de … », ne remet que les valeurs de ce produit.

Les réglages qui comptent le plus :

- **Résolution du MNT** (mètre par pixel). Base de tous les autres produits ; une résolution plus grossière accélère beaucoup le calcul. Les rayons des indices se règlent **en mètres** et sont convertis en pixels à la résolution du MNT : un anneau de 5 à 10 m reste de 5 à 10 m au sol, soit 10 à 20 pixels à 0,5 m et 5 à 10 pixels à 1 m. En point de départ MNT existant, c'est la résolution des MNT fournis qui compte, et le journal la donne au début du traitement.
- **Filtre des points** : les classes de points conservées pour reconstruire le sol (sol, bâti, eau…). En point de départ LiDAR, la fusion d'une dalle avec ses voisines ne garde que ces points : le nuage fusionné de `technique/intermediaires/` ne contient pas la végétation, le MNT et la densité se calculent plusieurs fois plus vite, pour les mêmes produits. Un filtre écrit dans une syntaxe non reconnue, ou refusé par l'outil de fusion, laisse le nuage entier et le journal le dit (voir [Le filtre des points n'est pas reconnu](depannage.md#le-filtre-des-points-n-est-pas-reconnu)).
- **Résolution de la densité** et **seuil de couverture** : le pourcentage de cellules sans point sol au-delà duquel une zone est marquée mal couverte.
- Les réglages propres à chaque indice : direction et hauteur de lumière, rayons, nombre de directions, exagération verticale, sortie 8 bits. Tous ont des bornes (pour le facteur de vue du ciel et l'ouverture : rayon de 5 à 25 m, 8 à 64 directions, bruit de 0 à 3), et une seule règle : les champs ne permettent pas d'en sortir ; une configuration chargée qui en sort y est ramenée, et un message nomme le réglage, la valeur lue et la valeur retenue (même message pour un choix de liste que l'onglet ne propose pas, la palette du CRIM par exemple, remplacé par le défaut) ; ce qui échapperait encore aux champs est refusé à l'étape 4, avec une pastille sur l'étape 2. Le rayon minimum de la dominance locale (LD) ne peut pas dépasser son rayon maximum : relever l'un fait suivre l'autre. Tous les modèles de détection ont appris sur une dominance locale d'anneau 5 à 10 m, les valeurs par défaut.
- **M-HS** sauvé en 8 bits : l'image compose trois éclairages, dans le rouge, le vert et le bleu ; le nombre de directions ne concerne que le fichier flottant et n'est affiché que si **Sauver en 8 bits** est décoché ; le dossier du M-HS 8 bits ne le porte plus dans son nom (toujours `M_HS_D16_…`), seul celui du fichier flottant en dépend. L'élévation et l'exagération s'appliquent dans les deux cas.
- **Suppression du bruit** (SVF, OPNS) : le niveau 1, « faible », n'agit qu'à partir d'un rayon de 15 pixels (7,5 m à 0,5 m) ; au rayon par défaut de 5 m, il donne le même résultat que 0.
- **Une configuration enregistrée avant les rayons en mètres** porte des pixels : elle est convertie une fois en mètres, à raison de 0,5 m par pixel (la résolution sur laquelle ces valeurs avaient été choisies), et un message nomme chaque réglage converti.
- **OPNS négative** : autour d'une zone sans donnée (eau, bord de chantier, dalle étrangère), le vide n'est plus pris pour un relief ; une ouverture négative calculée par une version précédente est recalculée une fois à la relance, sous le même nom de dossier, comme le M-HS 8 bits (son élévation est maintenant appliquée) et le MSTP à luminosité réglée (sa luminosité est maintenant appliquée ; celui à la valeur par défaut, passée à 0,9, s'écrit dans un nouveau dossier).
- **Sauver en 8 bits** : coché, le produit est étiré en 256 niveaux, léger et prêt pour la détection ; décoché, il garde les valeurs mesurables (degrés, angles…) dans un dossier à part dont le nom finit par `_F32` ; pour les compositions (VAT, PRISM, CVAT), le flottant garde la composition en valeurs de 0 à 1, sans gain à l'affichage. Passer de l'un à l'autre et relancer dans le même dossier de sortie calcule bien l'autre version.
- **Pente (SLO)** en **Pourcentage** : 100 % vaut 45°. En 8 bits, le produit est étiré de 0 à 123 %, soit de 0 à 51° comme en degrés : le plat sort blanc, le pentu noir. Une pente « en pourcentage » calculée par une version précédente contenait en réalité des radians (image presque blanche) ; elle reste dans son dossier `SLO_U1_…`, à supprimer. Les nouvelles vont dans `SLO_U2_…`.
- **MSTP** : la luminosité s'applique telle que réglée ; elle vaut 0,9 par défaut. Il n'a pas d'exagération verticale : il compare chaque pixel à l'écart type de son voisinage, et l'exagération s'y annule ; celle qu'une ancienne configuration porterait encore est ignorée. Les trois pas d'échelle changent le résultat : dès qu'ils quittent leurs valeurs par défaut, ils s'ajoutent au nom du dossier (`…_P4-18-60`), et une relance avec d'autres pas recalcule le MSTP.

## Marge entre dalles

En bas de la page, la carte **Marge entre dalles** règle la **marge** ajoutée autour de chaque dalle lors du calcul des produits. Les dalles voisines sont fusionnées avec cette marge, puis chaque produit est recadré sur la dalle : les produits se raccordent sans couture et les objets en bordure sont vus entiers par la détection.

Sous les réglages, une ligne de diagnostic compare cette marge au rayon le plus grand demandé par vos indices :

- **✓ Contexte fourni aux noyaux** : la marge couvre le rayon, rien à faire.
- **⚠** suivi du nom d'un indice : son rayon dépasse la marge. Au-delà de la marge, l'algorithme reconstruit le voisinage par symétrie et les dalles ne se raccordent plus. Réduisez le rayon indiqué, ou augmentez la marge.

En MNT existant comme en Indices de visualisation existants, le champ de marge est masqué. En MNT existant, chaque raster est calculé sans voisin, la marge n'aurait aucun effet, et le diagnostic dit quelle part de l'emprise aura un voisinage complet. Sur un lot, le réglage vaut pour toutes les dalles ; ne réduisez un rayon que si la plupart des rasters ont cette taille.

## Point de départ MNT existant

Le relief nu et la qualité de la donnée ne sont pas proposés : le MNT est votre donnée d'entrée, et la densité comme la couverture se calculent depuis les points. Depuis des indices de visualisation existants, la page est sans objet et le dit.
