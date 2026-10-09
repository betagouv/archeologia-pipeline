# Étape 3 · Détection

La détection est facultative : tant qu'elle est désactivée, le traitement s'arrête après les indices. Activez-la, puis cochez les **entités** à chercher. Vous ne choisissez pas de modèle : pour chaque entité, le plugin retient le modèle entraîné pour elle et l'indice de visualisation sur lequel il a appris.

## Les entités

Les entités sont groupées par morphologie. Les pastilles de filtre, au-dessus, ne changent que l'affichage, pas votre sélection.

| Morphologie | Entités |
|---|---|
| Circulaires et ponctuelles | Cratères · Charbonnières · Fours · Dépressions circulaires grandes · Enclos · Enclos circulaires |
| Linéaires | Tranchées et boyaux · Chemins creux · Parcellaire · Talus et fossés |
| Zones | Regroupement de cratères |

La liste dépend des modèles installés : une entité n'apparaît que si un modèle la couvre.

[![Étape 3 : les entités par morphologie, avec vignette, fiche et indice requis](img/etape3-detection.png)](img/etape3-detection.png)

## La carte d'une entité

- **Cocher une entité** : cliquez n'importe où sur sa carte. Sur une carte cochée, la partie réglages ne coche ni ne décoche : un clic sur la ligne Confiance, la figure ou les paramètres de regroupement sert seulement à régler.
- **Vignette** et lien **Fiche** : le lien ouvre la fiche, qui se lit dans l'ordre des questions qu'on se pose avant de cocher : à quoi ressemble la structure sur le relief, dans quelle optique l'utiliser et ce que le modèle ne détecte pas, dans quel corpus et sur quel terrain il l'a apprise, sa fiabilité mesurée avec le **profil des scores** (les détections de l'évaluation rangées par score, les vraies en couleur et les fausses en gris, avec les coupures des quatre niveaux), ses limites, puis le contexte technique. Depuis la fiche, un lien ouvre la fiche du modèle. En « Vérité terrain », les contours des objets annotés sont tracés dans la couleur de la couche de la classe, celle que vous retrouvez dans QGIS. Dans « Ce que le modèle a appris », une petite carte montre les zones où la structure a été apprise et mesurée, chacune en disque de la couleur de sa couche, d'autant plus grand qu'elle compte d'objets annotés ; à côté, la liste des zones avec leurs effectifs. Survoler une zone de la liste ou un disque de la carte met les deux en valeur : la ligne se teinte et le disque s'entoure. Les deux se distinguent d'un coup d'œil : la fiche d'une structure porte un liseré et l'étiquette « Structure détectable » dans la couleur de sa couche, la fiche d'un modèle un liseré ardoise, l'étiquette « Modèle de détection » et la liste des structures qu'il détecte, chacune avec sa pastille de couleur.
- **Indice requis** : le sigle de l'indice sur lequel le modèle travaille. S'il n'est pas coché à l'étape 2, le sigle passe en orange et un bouton **+ Activer** l'ajoute.
- **Modèle** : « seul disponible », ou un menu **Changer ▾** quand plusieurs modèles couvrent l'entité. Chaque entrée du menu indique le nombre de **fenêtres d'analyse** que le modèle découpe dans une dalle : c'est un fait sur le volume de calcul, pas une estimation de durée. Le bouton **ⓘ** ouvre la fiche du modèle : architecture, seuils, métriques d'évaluation et courbes.
- **Regrouper en zones** : pour certaines entités, une case regroupe les détections proches en zones. Le **Regroupement de cratères** est une entité à part entière, avec le badge **regroupement automatique** : la cocher produit les zones et les cratères qui les composent.

[![La fiche d'une classe : l'étiquette « Structure détectable », la vignette du corpus avec la bascule relief seul / vérité terrain, puis ce qu'il faut savoir avant de cocher](img/etape3-fiche-classe.png)](img/etape3-fiche-classe.png)

## Comparer deux modèles

Quand deux modèles couvrent une entité, cochez-les tous les deux dans le menu **Changer ▾**. L'entité est détectée deux fois, une fois par modèle ; les sorties portent le nom du modèle et QGIS les charge dans un même groupe marqué « comparaison », pour les superposer.

## Seuils et réglages par entité

La case **Réglages avancés (seuils par entité)** déplie, sur chaque carte :

- **Seuil de confiance** : les détections dont le score est sous ce seuil sont écartées. Le défaut vient du modèle, fixé à son évaluation ; il est choisi un peu en dessous du point d'équilibre entre précision et rappel, parce qu'en prospection une structure manquée ne se rattrape pas alors qu'une fausse détection s'écarte en quelques secondes sur le relief. Baisser le seuil augmente le rappel et les fausses détections ; le monter fait l'inverse.
- **Aire minimale** des détections conservées, en mètres carrés.
- Pour les regroupements : distance maximale entre deux détections, nombre minimal de détections, confiance minimale pour participer au regroupement, aire minimale d'une zone, marge autour de la zone. Les enclos et les axes linéaires ont leurs propres réglages (fermeture, élongation, longueur minimale…), expliqués en infobulle.
- Le bouton **↺** d'une carte remet cette seule entité aux valeurs du modèle.

Quand le modèle est livré avec son évaluation, un **profil des scores** en miniature apparaît sous la case : les barres grises sont les fausses détections du banc d'évaluation, les barres colorées les vraies, et la ligne du seuil suit la valeur que vous saisissez. Sous l'axe, une barre découpée en segments montre la plage de scores de chaque niveau, dans sa teinte de la légende, avec son nom dessous. La phrase sous la figure dit ce que votre réglage change par rapport au seuil du modèle : détections correctes et fausses détections gagnées ou perdues sur le banc. Ces chiffres sont ceux du banc, sur votre terrain ils varient ; ils donnent le sens et l'ordre de grandeur d'un réglage, pas une promesse.

> Le seuil est appliqué après le regroupement : des détections sous le seuil peuvent encore contribuer à une zone, puis disparaître de la couche des détections individuelles.

## Fiabilité affichée

Un score de modèle n'est pas une probabilité, et son échelle change d'un modèle à l'autre. Dans QGIS, la légende d'une couche de détections n'affiche donc pas le score mais une **fiabilité** en quatre niveaux, mesurée à l'évaluation du modèle : **douteux**, **possible**, **probable**, **très probable**. Le mot veut dire la même chose pour tous les modèles ; seuls les scores de coupure changent, par classe. L'infobulle de la couche et son résumé dans le panneau des couches donnent la part mesurée de vrais objets pour chaque niveau. Voir [Vos résultats dans QGIS](resultats.md#la-legende-de-fiabilite).

La **fiche** d'une classe montre le profil complet : les quatre niveaux sous l'axe avec leur part mesurée et leur effectif, une ligne pointillée au point d'équilibre entre précision et rappel (le seuil du modèle est choisi en dessous), le bilan du seuil réglé, et, quand l'évaluation couvre plusieurs zones, un petit profil par zone : une classe peut être sûre dans une forêt et plus faible dans une autre. En haut à droite de la figure, la précision et le rappel mesurés au banc pour le seuil courant. La case **Tester un seuil** de la fiche déplace la ligne, recalcule les niveaux, la précision et le rappel, sans changer le seuil du traitement : c'est un essai, le seuil appliqué reste celui de la carte. Sur les petits profils par zone, le rappel est recalculé à partir des objets annotés de la zone ; il n'existe pas pour les classes linéaires, évaluées au critère de couverture, où les détections correctes sont des fragments et non des objets. Survolez une barre pour ses effectifs ; un clic droit sur la figure l'enregistre ou la copie, pour un rapport ou une présentation. En comparaison de deux modèles, la figure prend la couleur de la couche de ce modèle, comme la légende. Quand vous avez saisi des verdicts dans QGIS (voir [Valider les détections](resultats.md#valider-les-detections)), chaque niveau de la fiche ajoute « chez vous : 64 % sur vos 85 vérifications » : le banc est un plancher annoncé, le terrain dit ce qu'il vaut ici.

## Objets à cheval sur deux dalles

Les images analysées débordent de la dalle, grâce à la marge de l'étape 2 ou, en mode Indices existants, grâce aux dalles voisines du dossier. Un objet coupé par le bord est vu entier par l'une des deux dalles, et chaque dalle ne rapporte que les détections dont le centre est chez elle : pas de doublon et pas de fragment au bord.

## Récapitulatif

La carte **Runs IA programmés** liste ce qui sera lancé : un modèle, ses classes, l'indice sur lequel il travaille. Un modèle travaille toujours sur un seul indice. La case **Générer images annotées** produit en plus des images avec les détections dessinées, rangées dans la zone technique du dossier de sortie.

À la fin du traitement, le journal donne pour chaque modèle le nombre d'images analysées et la durée mesurée, pour comparer deux modèles sur un cas réel.
