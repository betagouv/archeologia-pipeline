# Étape 4 · Vérifications

Avant de lancer, cette étape vérifie que tout est prêt et reprend vos choix. **Lancer le pipeline**, en bas à droite, démarre le traitement et ouvre l'[étape 5](etape-5-traitement.md), où il se suit.

## Configuration et récapitulatif

Le bandeau du haut dit si la configuration est valide. S'il manque quelque chose, il nomme les points à corriger et le rail de gauche pose une pastille sur l'étape concernée. Tant qu'un point reste à corriger, **Lancer le pipeline** est grisé et son infobulle les liste.

Le **récapitulatif** reprend vos choix : mode et source, produits, entités et modèles, dossier de sortie.

## État du système

Le panneau vérifie en tâche de fond ce dont le traitement a besoin :

- les outils en ligne de commande : PDAL pour les nuages de points, GDAL pour les rasters ;
- QGIS Processing et les algorithmes de Relief Visualization Toolbox ;
- le moteur de détection et ses bibliothèques, si la détection est activée ;
- les dossiers d'entrée et de sortie, et la projection des rasters en MNT ou Indices existants.

Chaque contrôle affiche son état et son détail. Un élément manquant bloque le lancement et dit quoi installer : voir [Dépannage](depannage.md#les-verifications-prealables-ont-echoue).

[![Étape 4 : configuration valide, état du système au vert, récapitulatif du run](img/etape4-lancer.png)](img/etape4-lancer.png)

## Paramètres avancés

**Workers parallèles** : le nombre de dalles traitées en même temps. Plus de workers accélère le traitement sur une machine à plusieurs cœurs mais consomme plus de mémoire. Sur un poste à 16 Go, restez à deux ou trois.

## Pendant un traitement

L'étape reste consultable pendant un traitement, en lecture seule comme les étapes 1 à 3 : la pastille de l'en-tête le signale, et les workers du traitement en cours ne se changent plus. Le bouton du bas devient **Suivant** et ramène au suivi, à l'étape 5.
