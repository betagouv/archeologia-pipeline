"""Types partagés du pipeline CV.

Ce module définit le type :class:`Detection`, source unique de vérité pour
représenter une détection produite par un modèle CV (bbox seule ou polygone
de segmentation).

Le module est volontairement **pure-Python** : aucune dépendance à shapely,
geopandas, numpy ou QGIS. Il peut être importé hors environnement QGIS et
testé unitairement.

Deux formats de dict coexistent dans la base de code (raison historique) ;
ce module fournit la (dé)sérialisation pour les deux :

- **format in-memory** (clé ``bbox`` = liste ``[x1, y1, x2, y2]``) : produit
  par les post-traitements de :mod:`computer_vision_onnx`.
- **format disk** (clé ``bbox_absolute`` = dict ``{minx, miny, maxx, maxy}``) :
  écrit dans les ``.json`` par :mod:`cv_output` et consommé par
  :mod:`conversion_shp`.

Les méthodes ``from_*_dict`` / ``to_*_dict`` sont conçues pour être
bijectives sur les formats produits aujourd'hui par le pipeline (round-trip
identique). Voir tests dans ``tests/unit/test_detection.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


BBox = Tuple[float, float, float, float]
"""Bounding box en pixels image : ``(x1, y1, x2, y2)`` avec x1<=x2, y1<=y2."""

Polygon = Tuple[Tuple[float, float], ...]
"""Polygone en coordonnées **normalisées** ``[0, 1]`` (x/img_width, y/img_height)."""

PolygonHoles = Tuple[Polygon, ...]
"""Trous d'un polygone, mêmes conventions que :data:`Polygon`."""


@dataclass(frozen=True)
class Detection:
    """Une détection produite par un modèle CV.

    Sémantique :

    - Mode **détection bbox** : seul ``bbox`` est significatif. ``polygon``,
      ``polygon_holes``, ``area`` valent ``None``.
    - Mode **segmentation** : ``polygon`` (et éventuellement
      ``polygon_holes``, ``area``) est rempli. ``bbox`` est la bounding box
      du polygone (calculée par le post-processing du modèle).

    Champs :
        class_id: identifiant de classe (0-indexé).
        confidence: score de confiance ``[0, 1]``. Pour la segmentation
            sémantique, fixée à ``1.0`` par convention historique.
        bbox: bbox en pixels image, ``(x1, y1, x2, y2)``.
        polygon: contour normalisé ``[0, 1]`` ; ``None`` en mode bbox.
        polygon_holes: trous du polygone, normalisés ``[0, 1]`` ; ``None``
            si pas applicable. Une liste vide ``()`` est autorisée et
            sémantiquement équivalente à ``None`` côté consommateur, mais
            est préservée pour la fidélité au format d'entrée.
        area: aire du polygone en pixels² ; ``None`` en mode bbox.
    """

    class_id: int
    confidence: float
    bbox: BBox
    polygon: Optional[Polygon] = None
    polygon_holes: Optional[PolygonHoles] = None
    area: Optional[float] = None

    # ------------------------------------------------------------------
    # Format in-memory (bbox = liste plate, polygon = liste plate normalisée)
    # ------------------------------------------------------------------
    @classmethod
    def from_internal_dict(cls, raw: Dict[str, Any]) -> "Detection":
        """Parse un dict au format in-memory produit par computer_vision_onnx.

        Format attendu :

        - mode bbox : ``{"class_id", "confidence", "bbox": [x1, y1, x2, y2]}``
        - segmentation : ``{"class_id", "confidence", "bbox", "polygon":
          [x1_n, y1_n, x2_n, y2_n, ...], "area", "polygon_holes"?:
          [[x1_n, y1_n, ...], ...]}``
        """
        bbox = _parse_bbox_list(raw["bbox"])
        polygon = _parse_flat_polygon(raw.get("polygon"))
        polygon_holes = _parse_polygon_holes(raw.get("polygon_holes"))
        area = raw.get("area")
        return cls(
            class_id=int(raw["class_id"]),
            confidence=float(raw["confidence"]),
            bbox=bbox,
            polygon=polygon,
            polygon_holes=polygon_holes,
            area=float(area) if area is not None else None,
        )

    def to_internal_dict(self) -> Dict[str, Any]:
        """Sérialise au format in-memory.

        Le dict produit reproduit bit-pour-bit le format actuel : la clé
        ``polygon_holes`` n'est ajoutée que si le champ est non-``None``,
        idem pour ``polygon`` et ``area``.
        """
        out: Dict[str, Any] = {
            "class_id": self.class_id,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
        }
        if self.polygon is not None:
            out["polygon"] = _polygon_to_flat(self.polygon)
        if self.polygon_holes is not None:
            out["polygon_holes"] = [_polygon_to_flat(h) for h in self.polygon_holes]
        if self.area is not None:
            out["area"] = self.area
        return out

    # ------------------------------------------------------------------
    # Format disk (bbox_absolute = dict avec minx/miny/maxx/maxy)
    # ------------------------------------------------------------------
    @classmethod
    def from_disk_dict(cls, raw: Dict[str, Any]) -> "Detection":
        """Parse un dict au format JSON sur disque (écrit par cv_output).

        Format attendu :

        - mode bbox : ``{"class_id", "confidence",
          "bbox_absolute": {"minx", "miny", "maxx", "maxy"}}``
        - segmentation : ``{"class_id", "confidence",
          "polygon": [...], "polygon_holes"?: [[...]]}``

        En segmentation, ``bbox`` n'est pas présente sur disque : on la
        reconstruit depuis le polygone (pixels image), avec ``img_width``
        et ``img_height`` à fournir via :meth:`from_disk_dict_with_dims`
        si la bbox absolue est nécessaire. Sans dimensions, la bbox est
        approximée en coordonnées normalisées (``min``/``max`` du polygone).
        """
        class_id = int(raw["class_id"])
        confidence = float(raw["confidence"])
        polygon = _parse_flat_polygon(raw.get("polygon"))
        polygon_holes = _parse_polygon_holes(raw.get("polygon_holes"))
        area = raw.get("area")

        if "bbox_absolute" in raw:
            bbox = _parse_bbox_absolute(raw["bbox_absolute"])
        elif polygon is not None:
            # Segmentation : pas de bbox sur disque, on la dérive du polygone
            # en coordonnées normalisées (l'appelant connaît les dims image).
            bbox = _bbox_of_polygon(polygon)
        else:
            raise ValueError(
                "Detection.from_disk_dict: ni 'bbox_absolute' ni 'polygon' "
                f"dans le dict (clés : {sorted(raw.keys())})"
            )

        return cls(
            class_id=class_id,
            confidence=confidence,
            bbox=bbox,
            polygon=polygon,
            polygon_holes=polygon_holes,
            area=float(area) if area is not None else None,
        )

    def to_disk_dict(self) -> Dict[str, Any]:
        """Sérialise au format JSON sur disque.

        Le dict produit reproduit le format actuel écrit par
        :func:`cv_output.save_detections_to_files` :

        - mode bbox : ``{"class_id", "confidence", "bbox_absolute": {...}}``
        - segmentation : ``{"class_id", "confidence", "polygon": [...],
          "polygon_holes"?: [[...]]}`` (sans ``bbox_absolute``).
        """
        out: Dict[str, Any] = {
            "class_id": self.class_id,
            "confidence": self.confidence,
        }
        if self.polygon is not None:
            out["polygon"] = _polygon_to_flat(self.polygon)
            if self.polygon_holes is not None:
                out["polygon_holes"] = [
                    _polygon_to_flat(h) for h in self.polygon_holes
                ]
        else:
            x1, y1, x2, y2 = self.bbox
            out["bbox_absolute"] = {
                "minx": x1,
                "miny": y1,
                "maxx": x2,
                "maxy": y2,
            }
        return out


# ----------------------------------------------------------------------
# Helpers privés
# ----------------------------------------------------------------------
def _parse_bbox_list(raw: Any) -> BBox:
    if not (isinstance(raw, (list, tuple)) and len(raw) == 4):
        raise ValueError(f"bbox attendue [x1,y1,x2,y2], reçu {raw!r}")
    x1, y1, x2, y2 = raw
    return (float(x1), float(y1), float(x2), float(y2))


def _parse_bbox_absolute(raw: Any) -> BBox:
    if not isinstance(raw, dict):
        raise ValueError(f"bbox_absolute doit être un dict, reçu {type(raw).__name__}")
    return (
        float(raw["minx"]),
        float(raw["miny"]),
        float(raw["maxx"]),
        float(raw["maxy"]),
    )


def _parse_flat_polygon(raw: Any) -> Optional[Polygon]:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"polygon doit être une liste plate, reçu {type(raw).__name__}")
    if len(raw) % 2 != 0:
        raise ValueError(
            f"polygon doit avoir un nombre pair de valeurs (x,y), reçu {len(raw)}"
        )
    return tuple((float(raw[i]), float(raw[i + 1])) for i in range(0, len(raw), 2))


def _parse_polygon_holes(raw: Any) -> Optional[PolygonHoles]:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            f"polygon_holes doit être une liste de listes, reçu {type(raw).__name__}"
        )
    holes: List[Polygon] = []
    for h in raw:
        parsed = _parse_flat_polygon(h)
        if parsed is None:
            continue
        holes.append(parsed)
    return tuple(holes)


def _polygon_to_flat(poly: Polygon) -> List[float]:
    out: List[float] = []
    for x, y in poly:
        out.append(x)
        out.append(y)
    return out


def _bbox_of_polygon(poly: Polygon) -> BBox:
    if not poly:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))
