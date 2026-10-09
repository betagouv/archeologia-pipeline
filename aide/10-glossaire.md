# Glossaire

| Terme | Définition |
|---|---|
| Cible dérivée | Résultat d'un regroupement présenté comme une entité cochable à part entière, par exemple le Regroupement de cratères |
| COPC | Nuage de points optimisé pour le cloud : le format des fichiers LAZ publiés par l'IGN |
| Couverture | Part des cellules du MNT réellement appuyées sur des points sol, par opposition aux cellules interpolées ; un indicateur de qualité de la donnée |
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
| Marge | Bande ajoutée autour d'une dalle, prise sur ses voisines, pour que les indices se raccordent et que les objets en bordure soient vus entiers |
| MNT | Modèle numérique de terrain : raster de l'altitude du sol nu |
| Modèle | Réseau de neurones entraîné à reconnaître une ou plusieurs classes sur un indice de visualisation donné |
| Mode | Point d'entrée dans la chaîne de traitement : IGN, nuages locaux, MNT existant, indices existants |
| Morphologie | Famille de formes qui groupe les entités : circulaires, linéaires, zones |
| Mosaïque | Fichier `index_<PRODUIT>.vrt` qui assemble toutes les dalles d'un indice en une seule couche QGIS |
| PDAL | Bibliothèque de traitement des nuages de points, fournie avec QGIS par OSGeo4W |
| Préflight | Vérifications préalables de l'étape 4 : outils, algorithmes, dossiers |
| Produit | Ce qui se coche à l'étape 2 : le MNT, les cartes de qualité et les indices de visualisation |
| Regroupement | Agrégation de détections proches en zones, par densité |
| Relief Visualization Toolbox | Plugin QGIS qui calcule les indices de visualisation du relief, dépendance du plugin |
| Run | Lancement d'un modèle sur un indice, pour une ou plusieurs entités |
| Score | Valeur entre 0 et 1 donnée par un modèle à une détection ; ce n'est pas une probabilité et son échelle change d'un modèle à l'autre |
| Seuil de confiance | Score en dessous duquel une détection est écartée, fixé par entité |
| Worker | Un processus de calcul ; le nombre de workers est le nombre de dalles traitées en parallèle |
