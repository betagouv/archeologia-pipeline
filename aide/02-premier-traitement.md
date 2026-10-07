# Premier traitement

Un parcours complet sur quelques dalles LiDAR HD de l'IGN, pour voir ce que produit le plugin avant de régler quoi que ce soit. Vous avez besoin d'une connexion Internet et d'un dossier vide sur un disque avec de la place.

## Ce que fait le plugin

À partir d'une zone et de données LiDAR, le plugin enchaîne automatiquement :

1. le **téléchargement** ou l'import des nuages de points ;
2. le calcul du **modèle numérique de terrain** (MNT) ;
3. le calcul des **indices de visualisation** du relief (ombrages, facteur de vue du ciel, dominance locale…) ;
4. la **détection automatique** d'entités archéologiques par des modèles d'IA, si vous l'activez ;
5. l'**assemblage** : mosaïques des rasters, GeoPackages de détections et un projet QGIS prêt à ouvrir.

Selon vos données, vous entrez dans cette chaîne à un endroit différent : c'est le **mode**, choisi à l'étape 1.

## L'assistant en quatre étapes

Le plugin s'ouvre sur un assistant : un rail à gauche indique l'étape courante, **Précédent** et **Suivant** en bas. Le rail signale par une pastille toute erreur qui empêcherait de lancer. Vos choix sont sauvegardés en continu : à la réouverture, vous retrouvez vos derniers réglages. Le bouton **?** en haut à droite, ou la touche **F1**, ouvre le chapitre de cette aide qui correspond à l'étape affichée.

## Étape 1 · Source

Cliquez **Téléchargement** sur la frise : c'est le mode IGN. Puis :

- **Sélectionner les dalles** : la grille des dalles LiDAR HD s'affiche sur le canevas de QGIS. Zoomez jusqu'à ce que les dalles apparaissent, cliquez-en trois ou quatre contiguës, puis **Valider**. Vous pouvez aussi donner un fichier de zone d'étude (polygone ou points) : toutes les dalles qu'il touche sont retenues.
- **Dossier de sortie** : un dossier vide.

Détail dans [Étape 1 · Source et mode](etape-1-source.md).

## Étape 2 · Produits

Cochez le **MNT**, puis deux indices qui se complètent bien pour un premier regard : **SVF** (facteur de vue du ciel, qui révèle les creux) et **LD** (dominance locale, qui fait ressortir les reliefs). Chaque carte a une vignette et un lien **Fiche** qui explique ce que montre l'image et ce qu'elle ne montre pas ; l'entrée **Comparer les produits** dit lequel choisir pour quel type de vestige.

Laissez les réglages par défaut. Détail dans [Étape 2 · Produits](etape-2-produits.md).

## Étape 3 · Détection

Activez la détection, puis cochez une entité, par exemple **Parcellaire** ou **Cratères**. Vous ne choisissez pas de modèle : le plugin retient le modèle entraîné pour cette entité et vérifie que l'indice dont il a besoin est bien coché à l'étape 2 (sinon, un bouton **+ Activer** l'ajoute). Le lien **Fiche** de chaque entité montre à quoi ressemble la structure sur le relief et où le modèle l'a apprise.

Détail dans [Étape 3 · Détection](etape-3-detection.md).

## Étape 4 · Lancer

Le bandeau du haut confirme que la configuration est valide ; le panneau **État du système** vérifie l'environnement. Cliquez **Lancer le pipeline**. L'écran bascule sur la vue d'exécution : une frise des phases, un chronomètre, une barre de progression et un journal. La durée dépend du nombre de dalles, des produits cochés et de votre machine ; le journal donne l'avancement réel, dalle par dalle.

Détail dans [Étape 4 · Lancer et suivre](etape-4-lancer.md).

## À la fin

Les couches sont chargées dans QGIS : les mosaïques des indices et, par entité, un GeoPackage de détections avec sa légende de fiabilité. Le projet `detections/detections_validation.qgs` dans le dossier de sortie regroupe tout, déjà stylé : c'est le point d'entrée pour valider les détections.

Ce qu'il y a dans le dossier de sortie et comment le lire : [Vos résultats dans QGIS](resultats.md).
