# Glossaire

| Terme | Définition |
|---|---|
| Analyse | Le passage d'un modèle sur une image du relief, pour une ou plusieurs structures ; la carte « Ce qui sera analysé » de l'étape 3 les liste |
| Carte graphique | Processeur graphique de l'ordinateur, que l'étape 4 peut faire servir à la détection par DirectML (Windows) après vérification contre le processeur |
| Cible dérivée | Résultat d'un regroupement présenté comme une structure cochable à part entière, par exemple le Regroupement de cratères |
| Couverture | Part des cellules du MNT réellement appuyées sur des points sol, par opposition aux cellules interpolées ; un indicateur de qualité de la donnée |
| Critère de l'axe | Règle d'évaluation des classes linéaires : une détection est vraie si son axe longe un tracé annoté à quelques mètres près (5 m), quelle que soit sa largeur ; la fiabilité des linéaires est mesurée ainsi, par polygone fusionné |
| Dalle | Un carré d'un kilomètre de côté de la grille LiDAR HD de l'IGN, unité de téléchargement et de calcul |
| Dalles en parallèle | Nombre de dalles téléchargées, fusionnées et préparées pour la détection en même temps, réglé à l'étape 4 |
| Densité | Nombre de points LiDAR sol par cellule |
| Détection | Un polygone produit par un modèle, avec sa classe, son score et sa fiabilité |
| Fenêtre d'analyse | Portion d'image sur laquelle un modèle travaille ; une dalle est découpée en fenêtres qui se recouvrent |
| Fiabilité | Niveau affiché dans la légende, douteux à très probable, défini par la part de vrais objets mesurée à l'évaluation du modèle dans la bande de scores |
| Fiche | Page d'explication d'une image du relief, d'une structure détectable ou d'un modèle, ouverte depuis sa carte |
| GeoPackage | Format vectoriel des couches de détections, un fichier par structure |
| Image du relief | Ce qui se coche à l'étape 2 : une image dérivée du MNT qui rend lisible une propriété du relief (ombrage, pente, facteur de vue du ciel…), nommée par son sigle et ce qu'elle montre (« LD · Structures en relief ») ; le relief nu (MNT) et les cartes de qualité de la donnée en font partie |
| Lambert-93 | La projection de référence de la France métropolitaine (EPSG:2154), exigée pour les rasters fournis en entrée |
| LAZ, LAS | Formats de nuages de points LiDAR ; LAZ est la forme compressée |
| LiDAR HD | Programme de l'IGN de couverture de la France par télémétrie laser aéroportée à haute densité |
| Livrable, technique | Les deux racines du dossier de sortie : `livrable/` se garde et se transmet (projet, rapport, mosaïques, détections), `technique/` se supprime (sources, intermédiaires, images, journaux) |
| Marge | Bande ajoutée autour d'une dalle, prise sur ses voisines, pour que les images se raccordent et que les objets en bordure soient vus entiers |
| MNT | Modèle numérique de terrain : raster de l'altitude du sol nu ; « Relief nu » à l'étape 2 |
| Modèle | Réseau de neurones entraîné à reconnaître une ou plusieurs classes sur une image du relief donnée |
| Morphologie | Famille de formes qui groupe les structures : circulaires, linéaires, zones |
| Mosaïque | Fichier `index_<PRODUIT>.vrt` qui assemble toutes les dalles d'une image du relief en une seule couche QGIS |
| PDAL | Bibliothèque de traitement des nuages de points, fournie avec QGIS par OSGeo4W |
| Point de départ | Ce que vous possédez déjà, choisi sur la frise de l'étape 1 : des dalles IGN à télécharger, des nuages LiDAR, un MNT existant ou des images du relief existantes ; le traitement part de là et les étapes précédentes sont barrées |
| Regroupement | Agrégation de détections proches en zones, par densité |
| Relief Visualization Toolbox | Bibliothèque de calcul des images du relief, du ZRC SAZU et de l'Université de Ljubljana (UL FGG) ; le plugin en intègre une copie, sous licence Apache 2.0 |
| Score | Valeur entre 0 et 1 donnée par un modèle à une détection ; ce n'est pas une probabilité et son échelle change d'un modèle à l'autre |
| Seuil de confiance | Score en dessous duquel une détection est écartée, fixé par structure |
| Structure | Ce qui se coche à l'étape 3 : une catégorie archéologique (parcellaire, cratères…) que le plugin résout en modèle et en image du relief ; le modèle l'apprend sous le nom d'une classe |
| Traitement | Un lancement de l'assistant, du point de départ à la fin : téléchargement ou lecture des données, MNT, images du relief, structures, assemblage ; il a son dossier de sortie, son journal et son rapport |
| Vérifications préalables | Les contrôles de l'étape 4, dans le panneau Environnement : outils, moteurs de calcul et de détection, projection des rasters, dossiers |
| Verdict | Votre jugement sur une détection, saisi dans QGIS : oui, non ou peut-être ; la fiche de la classe en tire la part de vrais objets « chez vous » |
