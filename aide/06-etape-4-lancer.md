# Étape 4 · Lancer et suivre

## Avant de lancer

Le bandeau du haut dit si la configuration est valide. S'il manque quelque chose, il nomme les points à corriger et le rail de gauche pose une pastille sur l'étape concernée.

Le **récapitulatif** reprend vos choix : mode et source, produits, entités et modèles, dossier de sortie.

### État du système

Le panneau vérifie en tâche de fond ce dont le traitement a besoin :

- les outils en ligne de commande : PDAL pour les nuages de points, GDAL pour les rasters ;
- QGIS Processing et les algorithmes de Relief Visualization Toolbox ;
- le moteur de détection et ses bibliothèques, si la détection est activée ;
- les dossiers d'entrée et de sortie, et la projection des rasters en MNT ou Indices existants.

Chaque contrôle affiche son état et son détail. Un élément manquant bloque le lancement et dit quoi installer : voir [Dépannage](depannage.md#les-verifications-prealables-ont-echoue).

[![Étape 4 : configuration valide, état du système au vert, récapitulatif du run](img/etape4-lancer.png)](img/etape4-lancer.png)

### Paramètres avancés

**Workers parallèles** : le nombre de dalles traitées en même temps. Plus de workers accélère le traitement sur une machine à plusieurs cœurs mais consomme plus de mémoire. Sur un poste à 16 Go, restez à deux ou trois.

## Pendant le traitement

**Lancer le pipeline** bascule l'écran sur la vue d'exécution :

- une **frise** des phases, selon le mode : Téléchargement, Produits, Détection, Finalisation, avec un chronomètre par phase ;
- une **barre de progression**, qui passe en mouvement continu quand une phase n'a pas de fin prévisible, par exemple le calcul d'un grand raster ;
- le **journal** : un message par événement, dans le vocabulaire de cette aide. Les pastilles **⚠** et **✗** comptent les avertissements et les erreurs et filtrent l'affichage ; **Copier** et **Effacer** agissent sur le texte affiché.

[![La vue d'exécution pendant un téléchargement : frise des phases, progression et journal](img/run-journal.png)](img/run-journal.png)

Pendant le traitement, les étapes 1 à 3 restent consultables mais en lecture seule, signalé par une pastille dans l'en-tête. **Annuler** arrête proprement à la fin de la dalle en cours.

### Lire le journal

- **▶ Démarrage**, puis les vérifications préalables.
- **Téléchargement** : les dalles identifiées, puis chaque dalle téléchargée.
- **Fusion des dalles avec leurs voisines** : la marge est constituée.
- **Calcul des produits** : une ligne par dalle, puis par modèle de terrain.
- **Détection** : une ligne par modèle, puis par image analysée ; à la fin de chaque modèle, le nombre d'images et la durée mesurée.
- **Assemblage** : mosaïques, GeoPackages, projet QGIS, puis le nombre de couches ajoutées.

Un **⚠** signale quelque chose qui n'arrête pas le traitement mais mérite d'être lu : un raster trop petit pour le rayon d'un indice, une dalle écartée. Un **✗** est une erreur ; si le traitement s'interrompt, le message en donne la cause et le journal complet du dossier de sortie le détail. Quand ce manuel a une rubrique pour le message, la ligne se termine par « voir Manuel › Dépannage › … » : un clic l'ouvre directement.

## À la fin

Le bandeau de fin donne la durée totale et le nombre d'avertissements. Les couches sont chargées dans QGIS, le projet de validation est écrit. **Ouvrir le dossier** ouvre le dossier de sortie, **Log complet** le journal détaillé.

Ce que contient le dossier et comment lire les couches : [Vos résultats dans QGIS](resultats.md).

## Relancer dans le même dossier

Vous pouvez relancer un traitement dans un dossier de sortie déjà utilisé, par exemple pour ajouter un indice ou une entité :

- un produit déjà calculé avec les mêmes réglages n'est pas recalculé ;
- un indice relancé avec d'autres réglages va dans un dossier distinct, dont le nom porte les réglages ;
- les mosaïques sont régénérées et les couches périmées retirées du projet QGIS au lancement, pour que les nouvelles dalles apparaissent.

Les sorties intermédiaires ne sont jamais supprimées par le plugin : vous pouvez les effacer vous-même une fois satisfait, voir [Faire de la place](resultats.md#faire-de-la-place).
