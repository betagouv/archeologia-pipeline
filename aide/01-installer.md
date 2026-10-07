# Installer et mettre à jour

Le plugin se distribue par un dépôt d'extensions QGIS privé. Vous l'ajoutez une seule fois ; QGIS vous propose ensuite chaque nouvelle version. Ce chapitre est aussi disponible séparément, avec l'adresse du dépôt, dans le message qui vous a été envoyé : on ne lit pas l'aide intégrée avant d'avoir installé.

## Ce qu'il faut

| Logiciel | Version | Remarque |
|---|---|---|
| QGIS | 3.34 ou plus récent, QGIS 4 compris | Installation OSGeo4W recommandée : elle apporte PDAL, utilisé pour lire les nuages de points |
| Relief Visualization Toolbox | plugin QGIS officiel | Calcule les indices de visualisation ; QGIS propose de l'installer en même temps que le plugin |
| Accès au dépôt | adresse et identifiants | Communiqués par l'équipe, jamais écrits dans cette aide |

Une bonne connexion est utile : le plugin est volumineux, car il embarque les modèles d'IA et la grille des dalles LiDAR HD de l'IGN.

## Ajouter le dépôt

1. Dans QGIS : **Extensions → Installer/Gérer les extensions…**, onglet **Paramètres**.
2. Section **Dépôts de plugins** → **Ajouter…**. Nom : `Archéolog'IA`, URL : l'adresse communiquée.
3. Zone **Authentification** : bouton **+** (nouvelle configuration), méthode **Basic authentication**, puis le nom d'utilisateur et le mot de passe communiqués. **Enregistrer**.
   - À la première utilisation, QGIS demande de créer un **mot de passe principal**. C'est un coffre local à votre ordinateur qui protège les identifiants enregistrés ; choisissez-en un et conservez-le, il n'a aucun rapport avec celui du dépôt. En QGIS 4, le coffre s'ouvre tout seul à la session suivante ; en QGIS 3, il peut vous le redemander.
4. **OK** : le dépôt doit afficher **connecté**.

Si le dépôt refuse les identifiants ou reste muet, voir [Dépannage](depannage.md#le-depot-ne-s-affiche-pas-connecte).

## Installer le plugin

1. Onglet **Tout** (ou **Non installées**), recherche `Arché`.
2. **Archéolog'IA** → **Installer le plugin**. Le téléchargement est long ; la barre peut sembler figée, c'est normal.
3. Si QGIS propose d'installer la dépendance **Relief Visualization Toolbox**, acceptez. Sinon : onglet **Tout**, recherche `Relief`, installez-la.

Le plugin apparaît dans le menu **Extensions → Archéolog'IA** et dans la barre d'outils. Le même menu contient **Manuel**, qui ouvre cette aide.

### Repli : installation depuis un fichier ZIP

Si le dépôt n'est pas utilisable sur votre poste :

1. Téléchargez `archeologia.<version>.zip` depuis l'adresse du dépôt, dans un navigateur, avec les mêmes identifiants.
2. **Extensions → Installer/Gérer les extensions… → Installer depuis un ZIP**, choisissez le fichier, **Installer l'extension**.

Une installation par ZIP ne reçoit pas les mises à jour automatiques : il faut recommencer à chaque version.

Le dossier des extensions, si vous devez y aller à la main : **Préférences → Profils utilisateurs → Ouvrir le dossier du profil actif**, puis `python\plugins`. C'est le bon dossier quelle que soit la version de QGIS.

## Réseau d'entreprise : le proxy

Si votre poste accède à Internet par un proxy (ministère, collectivité…), QGIS doit le connaître. Sans cela, l'ajout du dépôt, l'installation et le téléchargement des dalles IGN échouent, avec un message du type « délai dépassé vers data.geopf.fr ».

1. **Préférences → Options…**, onglet **Réseau**.
2. Cochez **Utiliser un proxy pour l'accès Internet**.
3. Type **HttpProxy**, hôte et port fournis par votre service informatique (les mêmes que dans votre navigateur), identifiants si le proxy en demande.
4. **OK**.

Le plugin lit cette configuration pour télécharger les dalles. À défaut, il essaie aussi les variables d'environnement `HTTP_PROXY` et `HTTPS_PROXY`.

> Limite : l'authentification automatique Windows (NTLM ou Kerberos) n'est pas prise en charge. Si votre proxy l'exige, demandez à votre service informatique une exception réseau vers `data.geopf.fr` ou un proxy acceptant l'authentification basique.

## Vérifier l'installation

Rien à faire de particulier : à l'étape 4 de l'assistant, le panneau **État du système** vérifie en tâche de fond les outils (PDAL, GDAL), QGIS Processing et les algorithmes de Relief Visualization Toolbox, le moteur de détection, et vos dossiers d'entrée et de sortie. Un élément manquant est nommé et bloque le lancement. Voir [Étape 4 · Lancer et suivre](etape-4-lancer.md#etat-du-systeme).

## Mettre à jour

Avec le dépôt, rien à réinstaller : quand une version est publiée, le gestionnaire d'extensions affiche une pastille. **Extensions → Installer/Gérer les extensions… → Mises à jour → Tout mettre à jour**. Le chapitre [Nouveautés](nouveautes.md) de cette aide dit ce qui a changé.

Vos configurations enregistrées et vos réglages ne sont pas dans le dossier du plugin : ils sont conservés dans votre profil QGIS et survivent aux mises à jour.
