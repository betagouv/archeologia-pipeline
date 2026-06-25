"""Classification du résultat de la conversion en GeoPackage (F1).

``create_shapefile_from_detections`` renvoie un booléen et peut lever une
exception ; historiquement, ni l'un ni l'autre n'était inspecté au point
d'appel — une panne de conversion produisait un « succès » silencieux à zéro
détection. Ce module est **pure-Python** (aucune dépendance lourde) et fournit
une classification testable du résultat, à remonter au narrateur.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ConversionOutcome:
    """Résultat classé de la conversion.

    status :
        - ``"failed"``  : exception levée, ou retour ``False`` (panne réelle).
        - ``"empty"``   : conversion ok mais aucun GeoPackage écrit
          (0 détection légitime — distinct d'un échec).
        - ``"ok"``      : conversion ok, au moins un GeoPackage écrit.
    """

    status: str
    message: str

    @property
    def is_failure(self) -> bool:
        return self.status == "failed"


def summarize_conversion_outcome(
    *,
    returned_ok: bool,
    error: Optional[str] = None,
    n_gpkgs_written: int = 0,
) -> ConversionOutcome:
    """Classe le résultat de ``create_shapefile_from_detections``.

    Args:
        returned_ok: valeur de retour de la fonction de conversion.
        error: message d'exception si une exception a été levée, sinon ``None``.
        n_gpkgs_written: nombre de GeoPackages attendus réellement présents
            sur disque après l'appel.
    """
    if error is not None:
        return ConversionOutcome(
            "failed",
            f"❌ Échec de la conversion en GeoPackage (exception) : {error}",
        )
    if not returned_ok:
        return ConversionOutcome(
            "failed",
            "❌ Échec de la conversion en GeoPackage (aucune couche écrite).",
        )
    if n_gpkgs_written <= 0:
        return ConversionOutcome(
            "empty",
            "⚠️ Conversion terminée sans aucune détection écrite (0 GeoPackage).",
        )
    return ConversionOutcome(
        "ok",
        f"✅ Conversion en GeoPackage réussie ({n_gpkgs_written} fichier(s)).",
    )
