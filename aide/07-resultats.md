# Vos résultats dans QGIS

## Le dossier de sortie

Deux dossiers, une règle : `livrable/` se garde et se transmet, `technique/` se supprime.

```
dossier_de_sortie/
├── livrable/                          ce que vous gardez et transmettez
│   ├── projet.qgs                     le projet QGIS, tout regroupé et stylé : le point d'entrée
│   ├── rapport.html                   le rapport du traitement, à ouvrir dans le navigateur
│   ├── traitement.json                résumé du dernier traitement, sans chemin de votre poste
│   ├── indices/
│   │   ├── MNT/tif/                   le modèle de terrain, dalle par dalle
│   │   │   └── index_MNT.vrt          la mosaïque, chargée dans QGIS
│   │   └── SVF_R10_D16_V1_N0/tif/     un indice, avec ses réglages dans le nom
│   │       └── index_SVF.vrt
│   └── detections/
│       ├── parcellaire/parcellaire.gpkg   un GeoPackage par entité
│       └── parcellaire/fiabilite.json     les coupures de fiabilité de cette couche
└── technique/                         ce que vous pouvez supprimer
    ├── sources/dalles/                dalles LiDAR téléchargées, re-téléchargeables
    ├── intermediaires/                fichiers de calcul par dalle
    ├── png/                           images préparées pour la détection
    ├── detection/<modèle>/            sorties brutes et images annotées
    └── journaux/                      un journal et une trace complète par lancement
```

Copier `livrable/` suffit à transmettre un traitement : le projet QGIS y retrouve ses couches par des chemins relatifs. Supprimer `technique/` libère la place ; un traitement relancé re-télécharge ou recalcule ce qui manque.

### Le nom d'un dossier d'indice

Le dossier d'un indice porte ses réglages : `SVF_R10_D16_V1_N0` est un facteur de vue du ciel à rayon 10 pixels, 16 directions, exagération 1, sans suppression de bruit ; `LD_A15_Rmin10_Rmax20_H1p7_V1` une dominance locale avec ses rayons et sa hauteur d'observation. Relancer avec d'autres réglages crée un autre dossier au lieu d'écraser le premier. Le MNT, la densité et la couverture n'ont pas de réglage : leur dossier porte le nom seul. En mode Indices existants, tout va sous `livrable/indices/RVT/`.

Dans chaque dossier `tif/`, la mosaïque `index_<PRODUIT>.vrt` assemble toutes les dalles : c'est elle que QGIS charge, sous le même nom. Chargée à la main, elle reste reconnaissable.

## Le projet de validation

`livrable/projet.qgs` est écrit à la fin de chaque traitement et constitue le point d'entrée pour valider les détections. Il contient les mosaïques des indices et, par entité, un groupe avec sa couche de détections stylée. Deux modèles comparés sur une même entité donnent deux couches dans un groupe marqué « comparaison ».

Les mêmes couches sont chargées directement dans votre projet QGIS courant à la fin du traitement.

## La légende de fiabilité

Chaque couche de détections est catégorisée en quatre niveaux : **douteux**, **possible**, **probable**, **très probable**. Un niveau est défini par la part de vrais objets mesurée à l'évaluation du modèle dans la bande de scores correspondante : au moins 35 % pour possible, 60 % pour probable, 85 % pour très probable. Le mot veut donc dire la même chose pour tous les modèles ; seuls les scores de coupure changent.

Le dessin : contour seul, sans remplissage, pour laisser lire la structure détectée dessous ; une couleur par classe, déclinée du plus foncé (très probable) au plus clair (douteux). L'infobulle de la couche et son résumé, dans les propriétés, donnent la part mesurée pour chaque niveau et le nombre de détections sur lequel elle a été mesurée.

[![Dans QGIS : une couche par entité, quatre niveaux de fiabilité, contours sur l'indice](img/resultats-legende-fiabilite.jpg)](img/resultats-legende-fiabilite.jpg)

Pour voir d'où viennent les coupures d'une classe, ouvrez sa fiche à l'étape 3 : le **profil des scores** range les détections de l'évaluation par score, vraies en couleur et fausses en gris, et trace les coupures. Sous le seuil, presque tout est faux ; au-dessus de la dernière coupure, presque tout est vrai.

Une couche issue d'un modèle sans fiabilité mesurée est catégorisée par tranches de score.

## Les attributs d'une détection

| Attribut | Contenu |
|---|---|
| `model_pred` | la classe prédite par le modèle |
| `confidence` | le score du modèle, entre 0 et 1 |
| `fiabilite` | le niveau de fiabilité affiché dans la légende |
| `fiabilite_pct` | la part de vrais objets mesurée pour ce niveau, en pour cent |
| `conf_bin` | la tranche de score, pour les modèles sans fiabilité mesurée |
| `model_name` | le modèle qui a produit la détection |
| `cluster_id` | le regroupement auquel la détection appartient, le cas échéant |
| `corr_pred`, `validation` | deux champs vides, à votre usage pour corriger la classe et noter votre verdict |

Les zones issues d'un regroupement portent en plus le nombre de détections qu'elles contiennent, leur surface en mètres carrés et leur densité par hectare.

## Valider les détections

Ouvrez le projet de validation, parcourez une couche de détections avec l'indice sur lequel le modèle a travaillé en dessous, et renseignez `validation` au fil de l'eau avec l'un des trois verdicts du formulaire : **oui** (vrai objet), **non** (fausse détection), **peut-être** (à revoir). Si la classe réelle est une autre, notez-la dans `corr_pred`. Ces verdicts servent ensuite : la fiche d'une classe (étape 3) affiche, à côté de la part de vrais objets mesurée au banc, la part observée **chez vous**, niveau par niveau, sur les traitements lancés depuis ce poste. Les détections **douteux** valent le coup d'œil mais rarement plus : en prospection, une fausse détection s'écarte en quelques secondes, une structure manquée ne se rattrape pas, c'est pourquoi le seuil par défaut laisse passer cette catégorie.

Pour une dalle, les détections de toutes les entités se superposent sans doublon aux bords : chaque dalle ne rapporte que les objets dont le centre est chez elle.

## Les traces d'un traitement

`livrable/traitement.json` résume le dernier traitement : version du plugin, date, nombre de dalles, produits et leurs réglages, modèles lancés, entités produites avec le chemin de leur GeoPackage, le bilan de fiabilité par entité. Il ne porte aucun chemin de votre poste et se transmet avec le livrable.

`technique/journaux/` garde, pour chaque lancement, le journal détaillé `pipeline_log_<date>.txt` et la trace complète `metadata_<date>.json`, avec la configuration entière de l'assistant, chemins de votre poste compris. C'est ce qu'il faut joindre à une demande d'aide.

## Faire de la place

Une fois le traitement validé, supprimez `technique/` : rien du livrable n'en dépend. Si vous voulez garder les nuages de points téléchargés, déplacez d'abord `technique/sources/`. Un traitement relancé dans le même dossier re-télécharge ou recalcule ce qui manque.

Sur Windows, un dossier de sortie très profond peut dépasser la limite de 260 caractères d'un chemin : choisissez un dossier de sortie court.

## Un dossier d'une version précédente

Un dossier de sortie écrit avant cette organisation garde ses anciens dossiers à la racine : `indices/`, `detections/`, `intermediaires/`, `sources/`, les journaux et `metadata.json`. Au lancement d'un traitement dans ce dossier, le plugin propose de le réorganiser et attend votre accord : il montre ce qui sera déplacé, déplace sans copier, et retire du projet QGIS les couches chargées depuis les anciens emplacements, rechargées en fin de traitement. Ce qu'il ne reconnaît pas, par exemple un dossier que vous avez posé vous-même, ne bouge pas et vous est indiqué. Refusez, et le traitement n'est pas lancé : choisissez alors un autre dossier de sortie.
