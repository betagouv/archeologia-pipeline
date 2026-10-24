# Dépannage

Les situations rencontrées, par ordre d'apparition dans un traitement. Quand le journal de l'étape 5 affiche un message, cherchez-le ici.

## Le dépôt ne s'affiche pas connecté

- Vérifiez le nom d'utilisateur et le mot de passe communiqués, en particulier les espaces en début ou en fin après un copier-coller.
- Vérifiez que la configuration d'authentification est bien de type **Basic authentication** et qu'elle est sélectionnée pour ce dépôt ; rouvrez le dépôt et resélectionnez-la au besoin.
- Sur un réseau d'entreprise, l'accès au dépôt passe aussi par le proxy : configurez-le d'abord, voir [le proxy](installer.md#reseau-d-entreprise-le-proxy).
- QGIS vous demande un **mot de passe principal** : c'est celui du coffre local de QGIS, pas celui du dépôt. Si vous l'avez oublié, le gestionnaire d'authentification de QGIS permet d'effacer le coffre et de recommencer.

## Le plugin n'apparaît pas dans la liste

- L'adresse du dépôt est exacte et le dépôt est connecté.
- L'onglet **Tout** ou **Non installées** est sélectionné, pas **Installées**.

## Alerte antivirus à l'installation

Certains antivirus signalent ou mettent en quarantaine `cv_runner_onnx.exe`, le moteur de détection, dans `data\third_party\cv_runner_onnx\windows`. C'est un faux positif : ce programme est un composant du plugin, compilé par l'équipe, qui ne fait que lire des images et écrire des détections. Restaurez le fichier depuis la quarantaine et ajoutez une exclusion pour le dossier du plugin. Sans lui, la détection est impossible : voir [La détection est impossible](#la-detection-est-impossible).

## Le plugin ne se lance pas

- Si le gestionnaire d'extensions affiche une erreur au chargement, notez le message complet et transmettez-le avec la version de QGIS.

## Les vérifications préalables ont échoué

Le panneau **Environnement** de l'étape 4 nomme ce qui manque :

- **PDAL** ou **GDAL** introuvables : installez QGIS par OSGeo4W, qui les fournit, ou ajoutez-les au chemin du système.
- **Moteur d'images intégré** indisponible : une bibliothèque livrée avec QGIS (numpy, scipy ou matplotlib) manque ou ne se charge pas ; le détail la nomme. Réinstallez QGIS par OSGeo4W.
- **Algorithmes RVT** absents : n'apparaît que si l'ancien calcul par l'extension a été forcé pour un diagnostic (variable d'environnement `ARCHEO_INDICES_MOTEUR=processing`) ; installez alors Relief Visualization Toolbox, puis relancez QGIS. Ce mode de secours garde les comportements de l'extension : pente « Pourcentage » étirée de 0 à 51 % en 8 bits, nombre de directions du M-HS limité à 64 même en 8 bits ; ses images sont réutilisées d'une relance à l'autre tant qu'on y reste, et recalculées une fois au retour du calcul intégré.
- « cv_runner_onnx (external) » en ✗, puis « python:onnxruntime » signalé : le moteur de détection manque, voir l'alerte antivirus ci-dessus.
- « CRS des rasters » refusé : un raster d'entrée n'est pas en Lambert-93 (EPSG:2154). Reprojetez-le avant.
- **Dossier** introuvable ou vide : le chemin, ou l'extension des fichiers attendue par le point de départ.

## Impossible de lancer, corrigez

Le bandeau de l'étape 4 liste les points bloquants et le rail pose une pastille sur l'étape concernée : une source manquante, un dossier de sortie absent, aucune image cochée.

## Le téléchargement échoue

- **Délai dépassé vers data.geopf.fr** : votre réseau impose un proxy que QGIS ne connaît pas. Voir [le proxy](installer.md#reseau-d-entreprise-le-proxy). Si le proxy exige une authentification Windows automatique, demandez une exception réseau.
- Une dalle refusée ou absente du serveur est signalée dans le journal et le traitement continue avec les autres.
- **Une dalle qui n'avance plus** : il arrive que le serveur de l'IGN garde une connexion ouverte sans plus rien envoyer, ou presque. Le plugin surveille chaque téléchargement : après deux minutes sans un seul bloc reçu, il coupe la connexion (« aucun octet reçu depuis 120 s, connexion coupée »), puis reprend là où il s'était arrêté, sans retélécharger ce qui est déjà sur le disque (« Reprise de … à l'octet … »), jusqu'à trois tentatives. Annuler pendant un tel blocage rend la main en une seconde ou deux. Si la reprise est refusée par le serveur, le téléchargement repart de zéro ; si les trois tentatives échouent, la dalle est abandonnée et le traitement continue avec les autres.

## Zoomez davantage pour afficher les dalles

La grille des dalles ne se dessine qu'à une échelle assez fine, pour rester fluide. Zoomez sur votre secteur : la barre de messages indique l'échelle requise.

## Méthode de calcul mise à jour

Le journal dit « ⚠️ Méthode de calcul mise à jour » ou « ⚠️ Paramètres de traitement modifiés » au début du traitement. Ce n'est pas une erreur : le dossier de sortie a été calculé par une version précédente du plugin, ou les réglages de fusion ou de MNT ont changé depuis, et la fusion, le MNT et les images de chaque dalle sont refaits une fois, puis réutilisés aux lancements suivants. Rien n'est supprimé, les détections sont refaites sur les nouvelles images. Voir [Un MNT reproductible](resultats.md#un-mnt-reproductible).

## Le noyau atteint N pixels

Le rayon d'une image dépasse la marge entre dalles, ou la taille d'un raster en MNT existant. Au-delà, l'algorithme reconstruit le voisinage par symétrie : les dalles ne se raccordent plus, ou l'image n'est pas fiable sur une partie de l'emprise. Réduisez le rayon nommé, ou augmentez la marge à l'étape 2. Sur un lot de rasters, ne réduisez le rayon que si la plupart ont cette taille : le réglage vaut pour tous. Voir [Marge entre dalles](etape-2-produits.md#marge-entre-dalles).

## Une couture entre deux dalles

Sur une image à grand rayon, une ligne droite visible à la frontière de deux dalles vient d'une marge insuffisante : même cause et même remède que ci-dessus. En MNT existant, chaque raster est calculé sans voisin et la couture est possible : fournissez une mosaïque plutôt que des dalles découpées si vous le pouvez.

## Une dalle est abandonnée

Si le calcul d'un modèle de terrain ne rend pas la main après deux heures, la dalle est abandonnée avec un message qui nomme la cause, pour ne pas bloquer le lot. Un fichier LiDAR laissé incomplet par un traitement interrompu n'est jamais pris pour un fichier valide : il est retéléchargé ou recalculé.

## Erreur PDAL ou mémoire

Le journal dit « N dalle(s) non fusionnée(s) », « Dalle … en échec, ignorée », « N dalle(s) sur M en échec », « MNT … en échec, ignoré » ou « MNT de la dalle … calculé sans interpolation », et le journal complet donne l'erreur de PDAL. Une erreur de PDAL pendant la fusion d'une dalle avec ses voisines, ou le calcul du MNT, en particulier avec un code de violation d'accès, vient presque toujours d'un manque de mémoire : plusieurs dalles traitées en parallèle sur un poste qui ne peut pas les tenir. Réduisez les **dalles en parallèle** à 1 ou 2 à l'étape 4 et relancez dans le même dossier : les dalles déjà calculées sont conservées.

Quand le calcul du MNT d'une dalle plante deux fois de suite, le plugin le calcule sans interpolation entre les points sol : la dalle peut garder quelques pixels vides là où le sol n'a pas été touché par le laser, et ses images en héritent. Le journal le dit par « MNT de la dalle … calculé sans interpolation ». Relancez dans le même dossier, de préférence avec moins de programmes ouverts : ce MNT est recalculé, les autres sont réutilisés.

## Le filtre des points n'est pas reconnu

Le journal dit « Filtre des points non reconnu … : fusion sans filtre » ou « Filtre des points refusé par PDAL … : fusion sans filtre ». Dans le premier cas, le filtre de l'étape 2 est écrit dans une syntaxe que le plugin ne sait pas traduire pour ne garder que ces points dès la fusion des dalles : comparaisons d'une dimension à un nombre (`Classification = 2`, `Z >= 10`), reliées par `AND`, `OR`, `NOT` et des parenthèses. Dans le second, la syntaxe est bonne mais l'outil de fusion la refuse, le plus souvent pour une dimension mal orthographiée ; la fin du message cite sa raison. Le traitement n'est pas en cause : le nuage fusionné garde tous les points, et le MNT comme la densité appliquent le filtre eux-mêmes, comme dans les versions précédentes, en plus lent. Ramenez le filtre à cette syntaxe, ou au défaut par le bouton ↺ de l'onglet, pour retrouver la fusion filtrée.

## La carte graphique n'est pas utilisée

L'option **Détection sur la carte graphique** de l'étape 4 n'impose rien : au premier usage de chaque modèle sur votre poste, le moteur compare la carte au processeur sur la première image et ne la garde que si elle est plus rapide et donne les mêmes détections. Le journal dit pourquoi elle a été écartée :

- « pas plus rapide que le processeur » : c'est le cas d'une carte intégrée au processeur, d'une session de bureau à distance ou d'une carte déjà occupée par un autre programme. Rien à corriger ; la détection se fait sur le processeur, comme sans l'option.
- « résultats différents du processeur » : le pilote de la carte calcule autrement. Mettez-le à jour, puis relancez ; la décision est réessayée d'elle-même après sept jours, ou tout de suite si le pilote, le modèle ou le moteur de détection ont changé.
- « session impossible » ou « DirectML indisponible » : la carte ou son pilote ne prend pas en charge DirectX 12, ou la mémoire de la carte ne suffit pas pour ce modèle.
- « erreur à l'image N … ; suite de la détection sur le processeur » : la carte a échoué en cours de traitement ; l'image en cause et les suivantes ont été analysées sur le processeur, rien n'est perdu, et la carte est écartée pour sept jours. Si cela se répète, mettez le pilote à jour.
- « indisponible aujourd'hui » : la carte retenue sur ce poste n'a pas pu être ouverte cette fois (occupée, pilote en cours de mise à jour, session distante) ; rien n'est mémorisé, le prochain traitement la réessaie.
- « parité non vérifiable ici » : la première image n'avait aucune détection, la carte a été gardée pour ce traitement parce qu'elle est plus rapide, et sera comparée de nouveau au traitement suivant.
- « non vérifiée » : aucune image n'a abouti sur le processeur dans ce traitement, la sonde n'a pas pu avoir lieu.
- « le moteur de détection livré ne la prend pas en charge » : l'extension installée n'a pas le moteur compilé pour la carte graphique ; réinstallez-la depuis le dépôt.

Pour forcer une nouvelle vérification avant les sept jours, supprimez le fichier `archeologia/carte_graphique.json` du profil QGIS (menu **Préférences › Profils utilisateurs › Ouvrir le dossier du profil actif**). Hors Windows, l'option est grisée.

## La détection est impossible

Le moteur de détection est un programme livré avec l'extension. S'il manque, l'état du système de l'étape 4 le signale en rouge, et un traitement lancé quand même s'arrête aussitôt sur ses vérifications. S'il s'arrête en cours de traitement, par exemple mis en quarantaine par un antivirus à sa première exécution, la fin du traitement le dit en rouge, avec la cause, et le rapport aussi. Les images du relief, eux, sont calculés ; seules manquent les détections du modèle en cause. Quand deux modèles sont comparés sur une même structure, l'autre détecte normalement et la ligne de fin nomme celui qui a échoué.

- Un antivirus l'a mis en quarantaine : voir [Alerte antivirus à l'installation](#alerte-antivirus-a-l-installation).
- Le fichier est endommagé, ou une mise à jour a été interrompue : réinstallez l'extension depuis le dépôt.

Relancez ensuite dans le même dossier : les images déjà calculés sont conservés, seule la détection est refaite.

## Rien à analyser, ou aucune détection

- Au moins une structure est cochée et couverte par un modèle : la carte **Ce qui sera analysé** le dit.
- L'image requise par le modèle est cochée à l'étape 2 : le sigle sur la carte de structure est orange sinon.
- Le seuil de confiance de la structure n'est pas trop haut : baissez-le pour diagnostiquer.
- Une structure dont toutes les détections sont écartées par le seuil ou l'aire minimale ne produit pas de couche : ce n'est pas une erreur.

## Les couches ne se chargent pas dans QGIS

Les couches ne se chargent plus d'elles-mêmes : à la fin du traitement, cliquez **Ouvrir le projet QGIS** dans le cadre « Par où commencer » ; le bouton n'agit qu'une fois. S'il est grisé, le traitement n'a pas abouti ou le bouton a déjà servi : tout est dans le dossier de sortie, ouvrez `livrable/projet.qgs`, ou ajoutez les mosaïques `index_<PRODUIT>.vrt` et les GeoPackages à la main.

## Le traitement s'est interrompu

Le dernier message du journal donne la cause. **Journal complet (fichier)** ouvre le journal détaillé du dossier de sortie, dans lequel les erreurs sont préfixées. Relancer dans le même dossier reprend ce qui est déjà calculé.

## Le traitement est très long

La durée dépend du nombre de dalles, des images cochées, des modèles et de votre machine. Leviers, dans l'ordre : décocher les images que vous ne regarderez pas ; baisser la résolution du MNT ; augmenter les dalles en parallèle si la mémoire le permet ; traiter une grande zone en plusieurs lots. Le nombre de fenêtres d'analyse affiché à l'étape 3 permet de comparer le coût de deux modèles avant de lancer.

## Signaler un problème

Joignez le journal complet du dossier de sortie et la trace `metadata_<date>.json` du même lancement, tous deux dans `technique/journaux/` (voir [Les traces d'un traitement](resultats.md#les-traces-d-un-traitement)), la version du plugin (dans le titre de la fenêtre) et celle de QGIS. Les anomalies se signalent sur le suivi public du projet : https://github.com/betagouv/archeologia-traitement/issues
