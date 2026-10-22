# Étape 4 · Vérifications

Avant de lancer, cette étape vérifie que tout est prêt et reprend vos choix. **Lancer le pipeline**, en bas à droite, démarre le traitement et ouvre l'[étape 5](etape-5-traitement.md), où il se suit.

## Configuration et récapitulatif

Le bandeau du haut dit si la configuration est valide. S'il manque quelque chose, il nomme les points à corriger et le rail de gauche pose une pastille sur l'étape concernée. Tant qu'un point reste à corriger, **Lancer le pipeline** est grisé et son infobulle les liste. Un réglage d'indice hors des bornes de l'algorithme (une configuration modifiée à la main, par exemple) en fait partie, avec le nom du champ, et la pastille se pose sur l'étape 2 : lancé tel quel, il ferait échouer chaque dalle après le téléchargement et le MNT.

Le **récapitulatif** reprend vos choix : mode et source, produits, entités et modèles, dossier de sortie.

## État du système

Le panneau vérifie en tâche de fond ce dont le traitement a besoin :

- les outils en ligne de commande, nommés comme à l'écran : « pdal (CLI) » pour les nuages de points, « gdalwarp », « gdal_translate » et « gdaladdo » pour les rasters ;
- « QGIS processing » et « Moteur d'indices intégré » (la bibliothèque Relief Visualization Toolbox livrée avec le plugin) ;
- si la détection est activée, le moteur de détection « cv_runner_onnx (external) » et ses bibliothèques « python:onnxruntime », « python:PIL », « python:geopandas » ;
- « Dossier de sortie » (avec les Go libres du disque), l'entrée du mode (« Fichier liste URLs IGN », dossier des nuages, des MNT ou des indices) et « CRS des rasters » en MNT ou Indices existants.

Chaque contrôle affiche son état et son détail. Un élément manquant passe le bandeau en rouge et dit quoi installer ; le bouton reste actif, mais le traitement s'arrête aussitôt sur ses vérifications préalables : voir [Dépannage](depannage.md#les-verifications-prealables-ont-echoue).

[![Étape 4 : configuration valide, état du système au vert, récapitulatif du run](img/etape4-verifications.png)](img/etape4-verifications.png)

## Paramètres avancés

**Workers parallèles** : le nombre de dalles téléchargées, fusionnées avec leurs voisines et préparées pour la détection en même temps. Le MNT et les indices, eux, sont calculés une dalle après l'autre. Plus de workers accélère ces étapes sur une machine à plusieurs cœurs mais consomme plus de mémoire : si une dalle échoue dans PDAL, voir [Erreur PDAL ou mémoire](depannage.md#erreur-pdal-ou-memoire).

**Détection sur la carte graphique** (Windows, décochée par défaut) : la détection est de loin l'étape la plus longue d'un traitement, et une carte graphique l'exécute plusieurs fois plus vite que le processeur. Cochée, l'option fait vérifier la carte par le moteur de détection au premier usage de chaque modèle sur votre poste : la première image est analysée sur le processeur, puis rejouée sur chaque carte de la machine. Une carte n'est gardée que si elle est nettement plus rapide et donne les mêmes détections que le processeur, à une bascule près pour les objets proches du seuil (à moins de 0,10 au-dessus). Le journal dit ce qui a été décidé, avec les durées mesurées ; la décision est mémorisée pour ce poste, ce modèle et cette version du moteur, et un refus est réessayé après sept jours. Si la carte échoue en cours de traitement, la détection continue sur le processeur et le journal le dit : aucune image n'est perdue. Le rapport et `traitement.json` notent l'appareil employé. Une carte intégrée au processeur est souvent plus lente que lui et sera écartée ; une session de bureau à distance aussi. Voir [Dépannage](depannage.md#la-carte-graphique-n-est-pas-utilisee).

## Pendant un traitement

L'étape reste consultable pendant un traitement, en lecture seule comme les étapes 1 à 3 : la pastille de l'en-tête le signale, et les workers du traitement en cours ne se changent plus. Le bouton du bas devient **Suivant** et ramène au suivi, à l'étape 5.
