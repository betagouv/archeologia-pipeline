# Glossaire

| Terme | Définition |
|---|---|
| Carte graphique | Processeur graphique de l'ordinateur, que l'étape 4 peut faire servir à la détection par DirectML (Windows) après vérification contre le processeur |
| Cible dérivée | Résultat d'un regroupement présenté comme une entité cochable à part entière, par exemple le Regroupement de cratères |
| Couverture | Part des cellules du MNT réellement appuyées sur des points sol, par opposition aux cellules interpolées ; un indicateur de qualité de la donnée |
| Critère de l'axe | Règle d'évaluation des classes linéaires : une détection est vraie si son axe longe un tracé annoté à quelques mètres près (5 m), quelle que soit sa largeur ; la fiabilité des linéaires est mesurée ainsi, par polygone fusionné |
| Dalle | Un carré d'un kilomètre de côté de la grille LiDAR HD de l'IGN, unité de téléchargement et de calcul |
| Densité | Nombre de points LiDAR sol par cellule |
| Détection | Un polygone produit par un modèle, avec sa classe, son score et sa fiabilité |
| Entité | Catégorie archéologique proposée à l'étape 3 (parcellaire, cratères…), que le plugin résout en modèle et en indice |
| Fenêtre d'analyse | Portion d'image sur laquelle un modèle travaille ; une dalle est découpée en fenêtres qui se recouvrent |
| Fiabilité | Niveau affiché dans la légende, douteux à très probable, défini par la part de vrais objets mesurée à l'évaluation du modèle dans la bande de scores |
| Fiche | Page d'explication d'un produit, d'une classe détectable ou d'un modèle, ouverte depuis sa carte |
| GeoPackage | Format vectoriel des couches de détections, un fichier par entité |
| Indice de visualisation | Image dérivée du MNT qui rend lisible une propriété du relief : ombrage, pente, facteur de vue du ciel… |
| Lambert-93 | La projection de référence de la France métropolitaine (EPSG:2154), exigée pour les rasters fournis en entrée |
| LAZ, LAS | Formats de nuages de points LiDAR ; LAZ est la forme compressée |
| LiDAR HD | Programme de l'IGN de couverture de la France par télémétrie laser aéroportée à haute densité |
| Livrable, technique | Les deux racines du dossier de sortie : `livrable/` se garde et se transmet (projet, rapport, mosaïques, détections), `technique/` se supprime (sources, intermédiaires, images, journaux) |
| Marge | Bande ajoutée autour d'une dalle, prise sur ses voisines, pour que les indices se raccordent et que les objets en bordure soient vus entiers |
| MNT | Modèle numérique de terrain : raster de l'altitude du sol nu |
| Modèle | Réseau de neurones entraîné à reconnaître une ou plusieurs classes sur un indice de visualisation donné |
| Mode | Point d'entrée dans la chaîne de traitement : IGN, nuages locaux, MNT existant, indices existants |
| Morphologie | Famille de formes qui groupe les entités : circulaires, linéaires, zones |
| Mosaïque | Fichier `index_<PRODUIT>.vrt` qui assemble toutes les dalles d'un indice en une seule couche QGIS |
| PDAL | Bibliothèque de traitement des nuages de points, fournie avec QGIS par OSGeo4W |
| Préflight | Vérifications préalables de l'étape 4 : outils, moteurs d'indices et de détection, projection des rasters, dossiers |
| Produit | Ce qui se coche à l'étape 2 : le MNT, les cartes de qualité et les indices de visualisation |
| Regroupement | Agrégation de détections proches en zones, par densité |
| Relief Visualization Toolbox | Bibliothèque de calcul des indices de visualisation du relief, du ZRC SAZU et de l'Université de Ljubljana (UL FGG) ; le plugin en intègre une copie, sous licence Apache 2.0 |
| Run | Lancement d'un modèle sur un indice, pour une ou plusieurs entités |
| Score | Valeur entre 0 et 1 donnée par un modèle à une détection ; ce n'est pas une probabilité et son échelle change d'un modèle à l'autre |
| Seuil de confiance | Score en dessous duquel une détection est écartée, fixé par entité |
| Verdict | Votre jugement sur une détection, saisi dans QGIS : oui, non ou peut-être ; la fiche de la classe en tire la part de vrais objets « chez vous » |
| Worker | Un processus de calcul ; le nombre de workers est le nombre de dalles téléchargées, fusionnées et préparées pour la détection en parallèle |
