# Dépannage

Les situations rencontrées, par ordre d'apparition dans un traitement. Quand le journal de l'étape 4 affiche un message, cherchez-le ici.

## Le dépôt ne s'affiche pas connecté

- Vérifiez le nom d'utilisateur et le mot de passe communiqués, en particulier les espaces en début ou en fin après un copier-coller.
- Vérifiez que la configuration d'authentification est bien de type **Basic authentication** et qu'elle est sélectionnée pour ce dépôt ; rouvrez le dépôt et resélectionnez-la au besoin.
- Sur un réseau d'entreprise, l'accès au dépôt passe aussi par le proxy : configurez-le d'abord, voir [le proxy](installer.md#reseau-d-entreprise-le-proxy).
- QGIS vous demande un **mot de passe principal** : c'est celui du coffre local de QGIS, pas celui du dépôt. Si vous l'avez oublié, le gestionnaire d'authentification de QGIS permet d'effacer le coffre et de recommencer.

## Le plugin n'apparaît pas dans la liste

- L'adresse du dépôt est exacte et le dépôt est connecté.
- L'onglet **Tout** ou **Non installées** est sélectionné, pas **Installées**.

## Alerte antivirus à l'installation

Certains antivirus signalent ou mettent en quarantaine `cv_runner_onnx.exe`, le moteur de détection, dans `data\third_party\cv_runner_onnx\windows`. C'est un faux positif : ce programme est un composant du plugin, compilé par l'équipe, qui ne fait que lire des images et écrire des détections. Restaurez le fichier depuis la quarantaine et ajoutez une exclusion pour le dossier du plugin ; à défaut, le plugin détecte quand même, par un moteur de secours plus lent, et le signale dans le journal.

## Le plugin ne se lance pas

- Vérifiez que **Relief Visualization Toolbox** est installée et activée dans le gestionnaire d'extensions.
- Si le gestionnaire d'extensions affiche une erreur au chargement, notez le message complet et transmettez-le avec la version de QGIS.

## Les vérifications préalables ont échoué

Le panneau **État du système** de l'étape 4 nomme ce qui manque :

- **PDAL** ou **GDAL** introuvables : installez QGIS par OSGeo4W, qui les fournit, ou ajoutez-les au chemin du système.
- **Algorithmes RVT** absents : installez ou activez Relief Visualization Toolbox, puis relancez QGIS.
- **Moteur de détection** absent : voir l'alerte antivirus ci-dessus.
- **Projection** refusée : un raster d'entrée n'est pas en Lambert-93 (EPSG:2154). Reprojetez-le avant.
- **Dossier** introuvable ou vide : le chemin, ou l'extension des fichiers attendue par le mode.

## Impossible de lancer, corrigez

Le bandeau de l'étape 4 liste les points bloquants et le rail pose une pastille sur l'étape concernée : une source manquante, un dossier de sortie absent, aucun produit coché.

## Le téléchargement échoue

- **Délai dépassé vers data.geopf.fr** : votre réseau impose un proxy que QGIS ne connaît pas. Voir [le proxy](installer.md#reseau-d-entreprise-le-proxy). Si le proxy exige une authentification Windows automatique, demandez une exception réseau.
- Une dalle refusée ou absente du serveur est signalée dans le journal et le traitement continue avec les autres.

## Zoomez davantage pour afficher les dalles

La grille des dalles ne se dessine qu'à une échelle assez fine, pour rester fluide. Zoomez sur votre secteur : la barre de messages indique l'échelle requise.

## Le noyau atteint N pixels

Le rayon d'un indice dépasse la marge entre dalles, ou la taille d'un raster en MNT existant. Au-delà, l'algorithme reconstruit le voisinage par symétrie : les dalles ne se raccordent plus, ou l'indice n'est pas fiable sur une partie de l'emprise. Réduisez le rayon nommé, ou augmentez la marge à l'étape 2. Sur un lot de rasters, ne réduisez le rayon que si la plupart ont cette taille : le réglage vaut pour tous. Voir [Tuilage et marge](etape-2-produits.md#tuilage-et-marge).

## Une couture entre deux dalles

Sur un indice à grand rayon, une ligne droite visible à la frontière de deux dalles vient d'une marge insuffisante : même cause et même remède que ci-dessus. En MNT existant, chaque raster est calculé sans voisin et la couture est possible : fournissez une mosaïque plutôt que des dalles découpées si vous le pouvez.

## Une dalle est abandonnée

Si le calcul d'un modèle de terrain ne rend pas la main après deux heures, la dalle est abandonnée avec un message qui nomme la cause, pour ne pas bloquer le lot. Un fichier LiDAR laissé incomplet par un traitement interrompu n'est jamais pris pour un fichier valide : il est retéléchargé ou recalculé.

## Erreur PDAL ou mémoire

Une erreur de PDAL pendant le découpage ou la fusion des dalles, en particulier avec un code de violation d'accès, vient presque toujours d'un manque de mémoire : plusieurs dalles traitées en parallèle sur un poste qui ne peut pas les tenir. Réduisez les **workers** à 1 ou 2 à l'étape 4 et relancez dans le même dossier : les dalles déjà calculées sont conservées.

## Aucun run, ou aucune détection

- Au moins une entité est cochée et couverte par un modèle : la carte **Runs IA programmés** le dit.
- L'indice requis par le modèle est coché à l'étape 2 : le sigle sur la carte d'entité est orange sinon.
- Le seuil de confiance de l'entité n'est pas trop haut : baissez-le pour diagnostiquer.
- Une entité dont toutes les détections sont écartées par le seuil ou l'aire minimale ne produit pas de couche : ce n'est pas une erreur.

## Les couches ne se chargent pas dans QGIS

Tout est dans le dossier de sortie : ouvrez `detections/detections_validation.qgs`, ou ajoutez les mosaïques `index_<PRODUIT>.vrt` et les GeoPackages à la main.

## Le traitement s'est interrompu

Le dernier message du journal donne la cause. **Log complet** ouvre le journal détaillé du dossier de sortie, dans lequel les erreurs sont préfixées. Relancer dans le même dossier reprend ce qui est déjà calculé.

## Le traitement est très long

La durée dépend du nombre de dalles, des produits cochés, des modèles et de votre machine. Leviers, dans l'ordre : décocher les indices que vous ne regarderez pas ; baisser la résolution du MNT ; augmenter les workers si la mémoire le permet ; traiter une grande zone en plusieurs lots. Le nombre de fenêtres d'analyse affiché à l'étape 3 permet de comparer le coût de deux modèles avant de lancer.

## Signaler un problème

Joignez le journal complet du dossier de sortie, le fichier `metadata.json`, la version du plugin (dans le titre de la fenêtre) et celle de QGIS. Les anomalies se signalent sur le suivi public du projet : https://github.com/betagouv/archeologia-pipeline/issues
