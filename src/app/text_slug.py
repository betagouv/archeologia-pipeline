"""Slugification de libellés en noms sûrs pour le système de fichiers.

Helper pur (aucune dépendance QGIS / pipeline) : sert à dériver le nom de
dossier ``detections/<slug>/`` depuis le libellé FR présentable d'une entité
(ex. « Dépressions circulaires » → ``depressions_circulaires``). On replie les
accents en ASCII (NFKD), on remplace tout run de caractères non alphanumériques
par ``_`` et on passe en minuscules. Une chaîne vide ou uniquement ponctuée
renvoie ``""`` (au site d'appel de prévoir un repli, ex. l'``id`` d'entité).
"""
from __future__ import annotations

import re
import unicodedata

_NON_ALNUM = re.compile(r"[^0-9A-Za-z]+")


def slugify(label: str) -> str:
    """Transforme ``label`` en slug ASCII minuscule séparé par ``_``."""
    folded = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode("ascii")
    return _NON_ALNUM.sub("_", folded).strip("_").lower()
