---
description: Lister ce qu'il reste à vérifier dans QGIS — les paragraphes de tests/TESTS_MANUELS_QGIS.md ajoutés ou modifiés depuis le dernier tag (ou une référence donnée), avec le rappel des recettes notées « reste à jouer » en mémoire.
argument-hint: [référence git de départ, ex. v0.13.0 ou dev ; défaut : le dernier tag]
---

# Recette QGIS — ce qu'il reste à jouer

Interagir en français. L'utilisateur reprend ses vérifications dans QGIS et demande « qu'est-ce que je
dois checker ? » : répondre par une liste à cocher, rien d'autre.

## Étapes
1. **Point de départ** : `$1` s'il est donné, sinon le dernier tag (`git describe --tags --abbrev=0`).
2. **Paragraphes touchés** : `git diff <départ>..HEAD -- tests/TESTS_MANUELS_QGIS.md`. Relever chaque
   ligne `- [ ] **<n>.<m> …**` ajoutée ou modifiée et le titre `## <n>.` de sa section. Un paragraphe
   modifié compte comme à rejouer (sa consigne a changé).
3. **Commits sans recette** : `git log <départ>..HEAD --oneline -- src/ui src/app src/pipeline aide`
   ; tout commit visible (`feat`, `fix` d'UI) dont aucun paragraphe ne porte la date ou le sujet est
   signalé « sans point de recette » — c'est un oubli à corriger dans le même lot, pas à ignorer.
4. **Mémoire** : relire les mémoires de projet qui disent « reste recette §… » et les ajouter si leurs
   paragraphes ne sont pas déjà dans la liste.
5. **Prérequis** : rappeler de recharger l'extension (plugin reloader) et, si un commit touche
   `src/pipeline/cv`, que `python run_tests.py -k binaire_a_jour` doit être vert avant une recette de
   détection.

## Sortie attendue
Une liste par section, dans l'ordre des numéros, une ligne par paragraphe : numéro, titre court, et
entre parenthèses le commit ou la date qui l'a introduit. En tête, le point de départ retenu et le
nombre de paragraphes. En queue, les commits « sans point de recette » s'il y en a. Pas de recette
jouée à la place de l'utilisateur : la liste s'arrête là.
