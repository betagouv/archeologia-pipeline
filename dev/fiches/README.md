# Outillage des fiches de classe

Scripts d'appoint pour écrire le bloc `classes[].fiche` d'un `model_card.yaml` —
ce que l'archéologue lit à l'étape 3 avant de cocher une entité : illustration,
provenance des données d'entraînement, hors-cible, contexte d'usage.

La procédure complète (dont l'ordre des étapes et les règles de contenu) est
dans [`docs/model_contract.md`](../../docs/model_contract.md), section
« Produire une fiche de classe ». Ici, seulement les outils.

Tous écrivent **textuellement** dans les `model_card.yaml` : un round-trip
PyYAML effacerait les commentaires qui documentent le choix des seuils et le
calibrage de la fiabilité. Chacun relit ensuite le fichier avec PyYAML pour
vérifier que ce qui a été écrit se relit à l'identique.

## `injecter_fiches.py`

```bash
python dev/fiches/injecter_fiches.py mes_fiches.json
```

`mes_fiches.json` = liste de `{modele, classe, fiche}`, où `fiche` est le bloc
complet (`resume`, `reconnaitre`, `usage`, `hors_cible`, `vignettes`,
`entrainement`). Une classe qui porte déjà un bloc est laissée intacte et
signalée.

## `injecter_cadrages.py`

```bash
python dev/fiches/injecter_cadrages.py mes_cadrages.json [planche.png]
```

`mes_cadrages.json` = liste de `{modele, vignette, cadrage: {x, y, cote}}`, en
fractions de l'image. Écrit la clé et rend une **planche de contrôle** : chaque
icône à sa taille réelle de 44 px, puis agrandie. Regarder la planche fait
partie du travail — un cadrage trop large redonne la bouillie grise qu'on
cherchait à éviter.

## `comparer_drive.py` / `reporter_sur_drive.py`

`data/models/**` est **gitignoré** : une fiche écrite dans le plugin n'est pas
versionnée, et le prochain ré-export du modèle l'effacerait. La copie qui fait
foi pour un ré-export est celle de l'archive,
`runs/training/<RUN_ID>/package/model_card.yaml`.

```bash
python dev/fiches/comparer_drive.py     # lecture seule : où les deux copies divergent
python dev/fiches/reporter_sur_drive.py # reporte fiches + vignettes vers l'archive
```

`bundles_drive.json` fait la correspondance `<modèle installé>` → chemin du
bundle dans l'archive, relatif à la racine `model-training`. Ajouter une ligne
par nouveau modèle.

⚠️ L'archive est sur un disque Google Drive en streaming : le montage peut
disparaître en cours de route (vu le 2026-09-10). `comparer_drive.py` le dit
franchement (`Drive : ABSENT`) plutôt que de laisser croire à une divergence.
Relancer quand le disque est revenu — les deux scripts sont idempotents.
