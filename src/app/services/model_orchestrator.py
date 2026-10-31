"""Orchestrateur de modèles CV : entités → modèles → runs.

Dans la V2, l'utilisateur ne choisit plus des *modèles* puis filtre leurs
*classes* ; il coche des **entités** à détecter (parcellaire, cratères…)
et cet orchestrateur résout automatiquement quels modèles lancer, sur quel
indice RVT, avec quelles classes.

Sources de vérité :

- ``data/entities_catalog.json`` : vocabulaire d'entités présentable
  (id, libellé, description, ordre d'affichage). Stable, versionné.
- ``model_card.yaml`` de chaque modèle installé : ``preferred_rvt.type``
  (l'indice RVT du modèle) + ``classes`` (les classes détectées, chacune
  pouvant déclarer une ``entity:`` de catalogue si son nom diffère de l'id
  d'entité). Le catalogue donne la présentation, les modèles déclarent leur
  couverture (découverte automatique « drop-in »).

Couverture : un modèle couvre l'entité ``E`` si l'une de ses classes a
``entity == E`` (alias explicite) ou ``name == E`` (défaut implicite).

Résolution des runs : on regroupe les entités sélectionnées par couple
``(modèle, target_rvt)`` ; chaque couple = un run. Les ``selected_classes``
d'un run valent :

- ``None`` si les entités choisies couvrent **toutes** les classes du modèle
  (aucun filtre → les détections brutes ET le clustering passent — cf. le
  filtre de ``runner_shapefiles`` qui ne s'applique que si ``selected_classes``
  n'est pas ``None``) ;
- sinon la **liste explicite** triée des classes (sous-ensemble) — le
  clustering dont l'``output_class_name`` n'est pas sélectionné est alors
  désactivé, comportement hérité voulu.

Le module est **pur-Python** : il ne doit jamais importer ``pipeline.cv``
(dont l'``__init__`` tire ``shapely``). La lecture YAML est différée.
"""
from __future__ import annotations

import json
import logging
import re

from .fiabilite import parse_fiabilite, parse_sous_seuil, run_block
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from ..text_slug import slugify

logger = logging.getLogger(__name__)


# Vocabulaire morphologique (axe d'organisation de l'étape 3). ``autre`` = repli
# pour une entité sans morphologie déclarée (catalogue v1) ou de valeur inconnue.
VALID_MORPHOLOGIES = ("circulaire", "lineaire", "zone")
_MORPHOLOGY_FALLBACK = "autre"
# Sections présentables : (clé, libellé, glyphe), dans l'ordre d'affichage.
MORPHOLOGY_SECTIONS = (
    ("circulaire", "Ponctuelles", "●"),
    ("lineaire", "Linéaires", "╱"),
    ("zone", "Zones / surfaces", "▦"),
    (_MORPHOLOGY_FALLBACK, "Autres", "•"),
)


# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class EntityDef:
    """Une entité du catalogue (vocabulaire présentable, stable)."""
    id: str
    label: str
    description: str = ""
    display_order: int = 1_000_000
    morphology: str = _MORPHOLOGY_FALLBACK   # circulaire | lineaire | zone | autre


@dataclass(frozen=True)
class InstalledModel:
    """Un modèle installé, lu depuis son ``model_card.yaml``."""
    name: str                              # slug = nom du dossier
    display_name: str
    weights_path: Optional[Path]
    target_rvt: str                        # preferred_rvt.type (MAJ), défaut "LD"
    status: str                            # ex. "production"
    coverage: Dict[str, Tuple[str, ...]]   # entity_id -> classes (de CE modèle)
    class_names: Tuple[str, ...]           # toutes les classes uniques du modèle
    # entity_id -> sorties de clustering disponibles (args.yaml:clustering),
    # proposées comme option « regrouper en zones » sur la carte d'entité.
    cluster_options: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    # Paramètres DBSCAN par défaut lus depuis args.yaml:clustering, indexés par
    # output_class_name (ex. {"zone_crateres": {"eps_m": 40.0, …}}). Sert à
    # pré-remplir les champs éditables côté UI sans importer ``pipeline.cv``.
    cluster_defaults: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # entity_ids dont la couverture provient d'une *cible dérivée* (model_card:
    # derived_targets) : une sortie de clustering présentée comme une entité à
    # part entière. Le regroupement y est intrinsèque (pas de case cluster).
    derived_entities: FrozenSet[str] = frozenset()
    # Pour chaque entité dérivée : ses classes SOURCES (les détections
    # individuelles à dupliquer/renommer dans le dossier de la dérivée) et le
    # libellé de couche source optionnel (model_card derived_targets.source_label).
    # Absent → repli ``<classe>_source`` côté routage.
    derived_source_classes: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    derived_source_labels: Dict[str, str] = field(default_factory=dict)
    # entity_id → libellé de couche du cluster (model_card derived_targets.output_label).
    # Absent → la couche cluster garde son nom de classe brut.
    derived_output_labels: Dict[str, str] = field(default_factory=dict)
    # entité de base → entité dérivée qui l'INCLUT (2026-09-15) : toutes ses classes
    # sont des classes sources d'une cible dérivée à ``include_source``. Cocher la
    # dérivée produit déjà sa couche (décision C du routage : une seule couche, dans
    # le groupe) → l'étape 3 la coche d'office et son seuil se règle sur la dérivée.
    implied_entities: Dict[str, str] = field(default_factory=dict)
    # Seuils par défaut (model_card:thresholds) — injectés par run, surchargeables
    # par entité côté UI (confiance + aire min). IoU jamais exposé dans l'UI.
    # 0.3 = défaut UNIFIÉ de la chaîne CV (= pipeline.cv.model_config.DEFAULT_CONFIDENCE,
    # littéral ici pour ne pas coupler app→pipeline ; gardé par test_defauts_cv_unifies).
    default_confidence: float = 0.3
    default_min_area: float = 0.0
    default_iou: float = 0.5
    # preferred_rvt.params canonisés (tuple trié) — cf. _extract_rvt_params.
    rvt_params: Tuple[Tuple[str, Any], ...] = ()
    # Seuils de confiance PAR CLASSE (model_card:thresholds.confidence_per_class,
    # {nom de classe: seuil}). Mesure au banc : les optima par classe s'étalent de
    # 0,10 à 0,30 sur lineaires_seg_v2_1, un seuil unique sacrifie les classes
    # rares. Une classe absente du dict retombe sur ``default_confidence``.
    default_confidence_per_class: Dict[str, float] = field(default_factory=dict)
    # Fiabilité affichée (model_card:thresholds.fiabilite, 2026-09-09) : catégories
    # douteux/possible/probable/quasi_certain PAR CLASSE, définies par la part de
    # vrais objets mesurée au banc (cf. app.services.fiabilite). Vide = tranches
    # conf_bin historiques.
    fiabilite_per_class: Dict[str, Tuple[Any, ...]] = field(default_factory=dict)
    fiabilite_provenance: str = ""
    # Bandes ``thresholds.fiabilite.sous_seuil`` (2026-10-22) : part de vrais des polygones
    # formés entre eux sous le seuil du modèle, pour mesurer « douteux » quand l'utilisateur
    # abaisse le seuil (``fiabilite.parse_sous_seuil``). Vide = non mesuré.
    fiabilite_sous_seuil: Dict[str, Tuple[Dict[str, Any], ...]] = field(default_factory=dict)
    # Dossier du modèle sur disque (``data/models/<name>/``). Utile côté UI pour
    # ouvrir le dossier dans l'explorateur ou (re)lire ``model_card.yaml`` /
    # ``args.yaml`` à la demande sans relancer ``discover_installed_models``.
    model_dir: Optional[Path] = None
    # Découpage SAHI d'``args.yaml`` (2026-09-21) : fenêtre (px) et recouvrement,
    # pour le coût structurel affiché à l'étape 3 (``app.services.cout_modele`` :
    # fenêtres par dalle, un fait — jamais une durée). 0 = inconnu → rien n'est
    # affiché ; aucun défaut 640 dupliqué ici.
    sahi_slice_px: int = 0
    sahi_overlap: float = 0.0
    # Échelle d'entraînement (2026-10-31) : résolution des images (m/px, training_params.json) et anneau du LD
    # en mètres, quelle que soit l'écriture des paramètres (indices_metres.anneau_m). None = inconnu.
    gsd_m: Optional[float] = None
    anneau_m: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class EntityCoverage:
    """Quels modèles couvrent une entité, et lequel est le défaut."""
    entity: EntityDef
    candidate_models: Tuple[str, ...]
    default_model: Optional[str]


# ----------------------------------------------------------------------
# Catalogue
# ----------------------------------------------------------------------
def load_entities_catalog(catalog_path: Any) -> List[EntityDef]:
    """Charge le catalogue d'entités, trié par ``display_order`` puis ``id``.

    Tolérant : fichier absent/illisible → ``[]`` ; entrée sans ``id`` ou
    ``label`` ignorée ; ``id`` dupliqué → premier gagné.
    """
    path = Path(catalog_path)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.warning("Catalogue d'entités illisible (%s): %s", path, e)
        return []

    raw = data.get("entities") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []

    result: List[EntityDef] = []
    seen: set = set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        eid = str(entry.get("id") or "").strip()
        label = str(entry.get("label") or "").strip()
        if not eid or not label or eid in seen:
            continue
        seen.add(eid)
        try:
            order = int(entry.get("display_order", 1_000_000))
        except (TypeError, ValueError):
            order = 1_000_000
        morpho = str(entry.get("morphology") or "").strip().lower()
        if morpho not in VALID_MORPHOLOGIES:
            morpho = _MORPHOLOGY_FALLBACK
        result.append(
            EntityDef(
                id=eid,
                label=label,
                description=str(entry.get("description") or ""),
                display_order=order,
                morphology=morpho,
            )
        )
    result.sort(key=lambda e: (e.display_order, e.id))
    return result


def group_entities_by_morphology(
    catalog: Sequence[EntityDef],
) -> List[Tuple[str, str, str, List[EntityDef]]]:
    """Regroupe le catalogue par morphologie pour l'affichage de l'étape 3.

    Renvoie ``[(clé, libellé_section, glyphe, [entités triées])]`` pour les
    sections **non vides** seulement, dans l'ordre canonique de
    ``MORPHOLOGY_SECTIONS``. Les entités de chaque section sont triées par
    ``display_order`` puis ``id``. Pur (testable sans QGIS) : la source unique
    du vocabulaire/ordre morphologique vit ici, pas dans l'UI.
    """
    by_key: Dict[str, List[EntityDef]] = {}
    for e in catalog:
        by_key.setdefault(e.morphology, []).append(e)
    out: List[Tuple[str, str, str, List[EntityDef]]] = []
    for key, label, glyph in MORPHOLOGY_SECTIONS:
        bucket = by_key.get(key)
        if bucket:
            out.append(
                (key, label, glyph, sorted(bucket, key=lambda e: (e.display_order, e.id)))
            )
    return out


# ----------------------------------------------------------------------
# Découverte des modèles installés (lecture model_card.yaml)
# ----------------------------------------------------------------------
def discover_installed_models(models_dir: Any) -> List[InstalledModel]:
    """Scanne ``models_dir`` et lit le ``model_card.yaml`` de chaque modèle.

    Un sous-dossier sans ``model_card.yaml`` lisible (ou sans classe
    exploitable) est ignoré avec un warning.
    """
    root = Path(models_dir)
    models: List[InstalledModel] = []
    if not root.is_dir():
        return models

    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        card = _load_model_card(sub)
        if card is None:
            logger.warning("model_card.yaml absent/illisible pour '%s', modèle ignoré", sub.name)
            continue
        coverage, class_names = _extract_coverage(card)
        if not class_names:
            logger.warning("Modèle '%s' sans classe exploitable, ignoré", sub.name)
            continue
        conf, conf_pc, area, iou = _extract_thresholds(card)
        fiab_pc, fiab_prov = parse_fiabilite(card.get("thresholds"))
        fiab_sous = parse_sous_seuil(card.get("thresholds"))
        clustering_rules = _load_args_clustering(sub)
        # cluster_options construites AVANT le merge des cibles dérivées : sinon
        # une cible déjà agrégée se verrait proposer une case « cluster » redondante.
        cluster_options = _build_cluster_options(coverage, clustering_rules)
        derived_targets = _load_derived_targets(card)
        derived_meta = _merge_derived_targets(coverage, derived_targets, clustering_rules)
        # Une sortie de clustering déjà exposée comme entité dérivée (ex.
        # zone_crateres → « Regroupement de cratères ») ne doit pas aussi proposer
        # une case « regrouper en clusters » sur l'entité source (redondant).
        cluster_options = _strip_derived_outputs(cluster_options, derived_targets)
        sahi_slice_px, sahi_overlap = _load_args_sahi(sub)
        gsd_m, anneau = _echelle_entrainement(card, sub)
        models.append(
            InstalledModel(
                name=sub.name,
                display_name=str(card.get("display_name") or sub.name),
                weights_path=_find_weights(sub),
                target_rvt=_extract_target_rvt(card),
                rvt_params=_extract_rvt_params(card),
                status=str(card.get("status") or "").strip(),
                coverage=coverage,
                class_names=class_names,
                cluster_options=cluster_options,
                cluster_defaults=_load_cluster_defaults(sub),
                derived_entities=frozenset(derived_meta.keys()),
                derived_source_classes={k: v[0] for k, v in derived_meta.items()},
                derived_source_labels={k: v[1] for k, v in derived_meta.items() if v[1]},
                derived_output_labels={k: v[2] for k, v in derived_meta.items() if v[2]},
                implied_entities=_implied_entities(
                    coverage, {k: v[0] for k, v in derived_meta.items()}
                ),
                default_confidence=conf,
                default_confidence_per_class=conf_pc,
                fiabilite_per_class=fiab_pc,
                fiabilite_provenance=fiab_prov,
                fiabilite_sous_seuil=fiab_sous,
                default_min_area=area,
                default_iou=iou,
                model_dir=sub,
                sahi_slice_px=sahi_slice_px,
                sahi_overlap=sahi_overlap,
                gsd_m=gsd_m,
                anneau_m=anneau,
            )
        )
    return models


def _echelle_entrainement(card: Dict[str, Any], model_dir: Path) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    """(GSD d'entraînement en m/px, anneau du LD en mètres) d'un modèle, depuis ``preferred_rvt.params`` du
    model_card et ``training_params.json`` (``mnt.resolution``, ``rvt.params``). Les fiches écrivent l'anneau
    de trois façons (``min_radius``/``max_radius`` px, ``rmin_px``/``rmax_px`` + ``gsd_m``, ``rmin_m``/``rmax_m``) :
    ``indices_metres.anneau_m`` les ramène toutes en mètres. Jamais d'exception."""
    from .indices_metres import anneau_m, gsd_de_fiche
    try:
        tp = json.loads((model_dir / "training_params.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        tp = {}
    pref = card.get("preferred_rvt") if isinstance(card.get("preferred_rvt"), dict) else {}
    params = pref.get("params") if isinstance(pref.get("params"), dict) else {}
    if not params and isinstance((tp.get("rvt") or {}).get("params"), dict):
        params = tp["rvt"]["params"]
    gsd = gsd_de_fiche(params, tp)
    return gsd, anneau_m(params, gsd)


def avertissements_echelle(models: Sequence["InstalledModel"], resolution_m: float,
                           rvt_params_px: Optional[Dict[str, Any]], *, indices_calcules: bool = True,
                           anneau_fourni_px: Optional[Tuple[int, int]] = None) -> List[str]:
    """Messages pour l'archéologue quand l'échelle d'un modèle ne correspond pas à ses données (règle
    2026-10-31), une fois par modèle et par run :

    - résolution des données ≠ résolution d'entraînement : les fenêtres de détection sont mises à l'échelle
      (le modèle voit le terrain à sa taille habituelle), mais il n'a jamais vu de données à cette résolution ;
    - LD calculé avec un autre anneau (en mètres) que celui de l'entraînement : détections dégradées ;
    - indices fournis (``indices_calcules`` faux) : anneau lu dans leur nom (``anneau_fourni_px``,
      ``LD_A15_Rmin10_Rmax20_…``) et comparé comme un LD calculé, sinon réglages inconnus, à vérifier.

    L'anneau comparé est celui que le calcul a REÇU, en pixels × résolution (audit 2026-10-31 : à 2 m/px, 5–10 m
    deviennent 3–5 px, soit 6–10 m), à une demi-résolution près : l'arrondi au pixel est inévitable, et le LD
    d'entraînement à cette résolution l'aurait subi aussi.
    """
    from .indices_metres import format_m
    msgs: List[str] = []
    for m in models:
        nom = f"« {m.display_name or m.name} »"
        if m.gsd_m and resolution_m and abs(m.gsd_m - resolution_m) > 1e-6:
            fen = ""
            if m.sahi_slice_px:
                px = max(32, int(m.sahi_slice_px * m.gsd_m / resolution_m + 0.5))
                fen = (f" : chaque fenêtre d'analyse fait {px} px au lieu de {m.sahi_slice_px}, soit "
                       f"{format_m(px * resolution_m)} de terrain comme à l'entraînement")
            msgs.append(f"Modèle {nom} : vos données sont à {format_m(resolution_m)[:-2]} m/pixel, il a appris à "
                        f"{format_m(m.gsd_m)[:-2]} m/pixel{fen}. Les structures gardent leur taille au sol, mais "
                        "le modèle n'a jamais vu cette résolution : relisez ses détections avec plus de prudence.")
        if m.target_rvt != "LD" or m.anneau_m is None:
            continue
        attendu = f"{format_m(m.anneau_m[0])[:-2]} à {format_m(m.anneau_m[1])}"
        if not indices_calcules and anneau_fourni_px is None:
            msgs.append(f"Modèle {nom} : il a appris sur une dominance locale (LD) d'anneau {attendu}. Les "
                        "indices fournis ont été calculés hors du plugin, leurs réglages ne sont pas connus : "
                        "vérifiez qu'ils ont cet anneau, sinon ses détections seront dégradées.")
            continue
        ld = (rvt_params_px or {}).get("ldo") or {}
        px = anneau_fourni_px if not indices_calcules else (ld.get("min_radius", 10), ld.get("max_radius", 20))
        try:
            lo, hi = float(px[0]) * resolution_m, float(px[1]) * resolution_m
        except (TypeError, ValueError, IndexError):
            continue
        tolerance = resolution_m / 2 + 1e-6
        if abs(lo - m.anneau_m[0]) > tolerance or abs(hi - m.anneau_m[1]) > tolerance:
            source = ("les indices fournis (d'après leur nom)" if not indices_calcules
                      else "le LD de ce traitement (étape 2, image LD)")
            msgs.append(f"Modèle {nom} : il a appris sur un LD d'anneau {attendu} ; {source} a un anneau de "
                        f"{format_m(lo)[:-2]} à {format_m(hi)} ({int(px[0])} à {int(px[1])} px à "
                        f"{format_m(resolution_m)[:-2]} m/pixel) : ses détections seront dégradées. "
                        "Remettez l'anneau à la valeur d'entraînement.")
    return list(dict.fromkeys(msgs))



def _load_model_card(model_dir: Path) -> Optional[Dict[str, Any]]:
    card_file = model_dir / "model_card.yaml"
    if not card_file.is_file():
        return None
    try:
        import yaml  # import différé : pas de coût si non utilisé
    except ImportError:
        logger.warning("PyYAML indisponible, model_card.yaml ignoré")
        return None
    try:
        data = yaml.safe_load(card_file.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.warning("Erreur lecture %s: %s", card_file, e)
        return None
    return data if isinstance(data, dict) else None


# Alias public : la fonction est utile à l'UI (dialog d'info modèle) pour
# (re)lire le ``model_card.yaml`` à l'ouverture du dialog. Pas de duplication.
load_model_card = _load_model_card


def _find_weights(model_dir: Path) -> Optional[Path]:
    for rel in ("weights/best.onnx", "weights/best.pt"):
        candidate = model_dir / rel
        if candidate.is_file():
            return candidate
    return None


def _extract_target_rvt(card: Dict[str, Any]) -> str:
    pref = card.get("preferred_rvt")
    if isinstance(pref, dict):
        rvt = str(pref.get("type") or "").strip().upper()
        if rvt:
            return rvt
    return "LD"


def _extract_rvt_params(card: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """``preferred_rvt.params`` canonisés (tuple trié) — sert à détecter deux
    modèles au même TYPE de RVT mais aux paramètres divergents (ex. enclos LD
    Rmin5/Rmax10 vs cratères LD Rmin10/Rmax20) : le regroupement des runs par
    type seul leur ferait partager UN raster aux paramètres de l'un des deux
    (audit 2026-08-31, §RVT)."""
    pref = card.get("preferred_rvt")
    if isinstance(pref, dict) and isinstance(pref.get("params"), dict):
        return tuple(sorted((str(k), v) for k, v in pref["params"].items()))
    return ()


def _extract_thresholds(card: Dict[str, Any]) -> Tuple[float, Dict[str, float], float, float]:
    """``(confidence_default, confidence_per_class, min_area_m2, iou)`` depuis
    ``model_card:thresholds``.

    Défauts : confiance 0.3 (défaut UNIFIÉ de la chaîne CV, cf.
    pipeline.cv.model_config.DEFAULT_CONFIDENCE), aire min 0, IoU 0.5. L'IoU peut
    être déclaré sous ``iou`` ou ``iou_threshold`` (jamais exposé dans l'UI,
    seulement le pipeline). ``confidence_per_class`` est optionnel :
    ``{nom de classe: seuil}``. Une entrée non castable est ignorée, pas fatale
    (model_card édité à la main).
    """
    conf, area, iou = 0.3, 0.0, 0.5
    conf_pc: Dict[str, float] = {}
    th = card.get("thresholds")
    if isinstance(th, dict):
        try:
            conf = float(th.get("confidence_default", conf))
        except (TypeError, ValueError):
            pass
        pc = th.get("confidence_per_class")
        if isinstance(pc, dict):
            for nom, val in pc.items():
                try:
                    conf_pc[str(nom)] = float(val)
                except (TypeError, ValueError):
                    logger.warning(
                        "thresholds.confidence_per_class[%r]=%r non numérique, ignoré",
                        nom, val)
        try:
            area = float(th.get("min_area_m2", area))
        except (TypeError, ValueError):
            pass
        for key in ("iou", "iou_threshold"):
            if key in th:
                try:
                    iou = float(th[key])
                    break
                except (TypeError, ValueError):
                    pass
    return conf, conf_pc, area, iou


def _extract_coverage(
    card: Dict[str, Any],
) -> Tuple[Dict[str, Tuple[str, ...]], Tuple[str, ...]]:
    """Construit ``{entity_id: (classes…)}`` et la liste des classes uniques.

    ``entity_id`` = ``class.entity`` si présent, sinon ``class.name``. Les
    noms de classes répétés (sous-types fusionnés par nom, ex. charbonnière
    A/B/C) sont dédupliqués.
    """
    classes = card.get("classes")
    if not isinstance(classes, list):
        return {}, ()

    coverage: Dict[str, List[str]] = {}
    all_names: List[str] = []
    for cls in classes:
        if not isinstance(cls, dict):
            continue
        name = str(cls.get("name") or "").strip()
        if not name:
            continue
        if name not in all_names:
            all_names.append(name)
        entity_id = str(cls.get("entity") or "").strip() or name
        bucket = coverage.setdefault(entity_id, [])
        if name not in bucket:
            bucket.append(name)

    return {k: tuple(v) for k, v in coverage.items()}, tuple(all_names)


def _load_args_sahi(model_dir: Path) -> Tuple[int, float]:
    """Lit ``args.yaml:sahi`` → ``(slice_px, overlap_ratio)`` ; ``(0, 0.0)`` si
    absent ou illisible (0 = inconnu, rien n'est affiché). Lecture YAML directe,
    tolérante, comme ``_load_cluster_defaults`` — jamais d'exception."""
    args_file = model_dir / "args.yaml"
    if not args_file.is_file():
        return 0, 0.0
    try:
        import yaml  # import différé
        data = yaml.safe_load(args_file.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.warning("Erreur lecture %s: %s", args_file, e)
        return 0, 0.0
    sahi = data.get("sahi") if isinstance(data, dict) else None
    if not isinstance(sahi, dict):
        return 0, 0.0
    try:
        slice_px = int(sahi.get("slice_width", sahi.get("slice_height", 0)) or 0)
        overlap = float(sahi.get("overlap_ratio", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0, 0.0
    return (slice_px, overlap) if slice_px > 0 else (0, 0.0)


def _load_args_clustering(model_dir: Path) -> List[Tuple[FrozenSet[str], str]]:
    """Lit ``args.yaml:clustering`` → ``[(target_classes, output_class_name)]``.

    Seuls les champs ``target_classes`` et ``output_class_name`` nous
    intéressent (savoir qu'une option de cluster existe et son nom de sortie) ;
    les paramètres DBSCAN restent gérés par le pipeline. Lecture seule.
    """
    args_file = model_dir / "args.yaml"
    if not args_file.is_file():
        return []
    try:
        import yaml  # import différé
    except ImportError:
        return []
    try:
        data = yaml.safe_load(args_file.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.warning("Erreur lecture %s: %s", args_file, e)
        return []
    if not isinstance(data, dict):
        return []

    raw = data.get("clustering")
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    rules: List[Tuple[FrozenSet[str], str]] = []
    for cfg in raw:
        if not isinstance(cfg, dict):
            continue
        targets = cfg.get("target_classes", cfg.get("target_class"))
        if isinstance(targets, str):
            targets = [targets]
        if not isinstance(targets, list) or not targets:
            continue
        output = str(cfg.get("output_class_name") or "").strip()
        if not output:
            continue
        rules.append((frozenset(str(t).strip() for t in targets), output))
    return rules


_CLUSTER_PARAM_INT = ("min_cluster_size", "min_samples", "min_sources")
# Paramètres exposables : DBSCAN + briques enclosure et alignment. Seuls ceux
# présents dans la règle args.yaml du modèle sont retenus, donc les clés d'un
# type n'apparaissent jamais sur une règle d'un autre type.
_CLUSTER_PARAM_FLOAT = (
    "eps_m", "min_confidence", "min_confidence_extend", "min_area_m2", "buffer_m",
    "gap_tolerance_m", "max_area_m2", "min_closure", "max_elongation",
    "min_ancrage", "max_isolement", "min_rectangularite",
    "band_width_m", "angle_tolerance_deg", "min_length_m", "max_gap_m",
    "min_coverage",
)


def _load_cluster_defaults(model_dir: Path) -> Dict[str, Dict[str, float]]:
    """Lit ``args.yaml:clustering`` → ``{output_class_name: {param: défaut}}``.

    Seuls les paramètres exposables/éditables dans l'UI sont retenus
    (``eps_m``, ``min_cluster_size``, ``min_samples``, ``min_confidence``,
    ``min_area_m2``, ``buffer_m``). Lecture YAML directe — l'orchestrateur ne
    doit pas importer ``pipeline.cv``/``ModelProfile``. Tolérant : absent → {}.
    """
    args_file = model_dir / "args.yaml"
    if not args_file.is_file():
        return {}
    try:
        import yaml  # import différé
    except ImportError:
        return {}
    try:
        data = yaml.safe_load(args_file.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.warning("Erreur lecture %s: %s", args_file, e)
        return {}
    if not isinstance(data, dict):
        return {}
    raw = data.get("clustering")
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for cfg in raw:
        if not isinstance(cfg, dict):
            continue
        output = str(cfg.get("output_class_name") or "").strip()
        if not output:
            continue
        params: Dict[str, float] = {}
        for key in _CLUSTER_PARAM_FLOAT:
            if key in cfg:
                try:
                    params[key] = float(cfg[key])
                except (TypeError, ValueError):
                    pass
        for key in _CLUSTER_PARAM_INT:
            if key in cfg:
                try:
                    params[key] = int(cfg[key])
                except (TypeError, ValueError):
                    pass
        out[output] = params
    return out


def _build_cluster_options(
    coverage: Dict[str, Tuple[str, ...]],
    rules: Sequence[Tuple[FrozenSet[str], str]],
) -> Dict[str, Tuple[str, ...]]:
    """``{entity_id: (cluster outputs…)}`` — une règle de cluster est rattachée
    à une entité si ses ``target_classes`` ⊆ aux classes de cette entité."""
    options: Dict[str, List[str]] = {}
    for entity_id, classes in coverage.items():
        class_set = set(classes)
        for targets, output in rules:
            if targets <= class_set:
                bucket = options.setdefault(entity_id, [])
                if output not in bucket:
                    bucket.append(output)
    return {k: tuple(v) for k, v in options.items()}


def _strip_derived_outputs(
    cluster_options: Dict[str, Tuple[str, ...]],
    derived_targets: Sequence[Tuple[str, str, bool, Optional[str], Optional[str]]],
) -> Dict[str, Tuple[str, ...]]:
    """Retire des ``cluster_options`` les sorties déjà exposées comme entité
    dérivée : une seule voie de regroupement (l'entité dérivée), pas de case
    « regrouper en clusters » redondante sur l'entité source. Une entité dont
    toutes les options disparaissent est retirée."""
    derived_outputs = {output for output, *_ in derived_targets}
    if not derived_outputs:
        return cluster_options
    pruned = {
        eid: tuple(o for o in outs if o not in derived_outputs)
        for eid, outs in cluster_options.items()
    }
    return {eid: outs for eid, outs in pruned.items() if outs}


def _load_derived_targets(
    card: Dict[str, Any],
) -> List[Tuple[str, str, bool, Optional[str], Optional[str]]]:
    """Lit ``model_card:derived_targets`` → ``[(output_class, entity_id, include_source, source_label, output_label)]``.

    Une *cible dérivée* présente une sortie de clustering (``output_class``,
    définie dans ``args.yaml``) comme une entité de catalogue à part entière.
    ``include_source`` (défaut ``True``) : inclure aussi les classes sources du
    clustering (les détections individuelles) dans la couverture de l'entité.
    ``output_label`` (optionnel) : nom de couche à donner au cluster dans le
    GeoPackage de l'entité (sinon nom de classe inchangé). ``source_label``
    (optionnel) : nom de couche des sources (les distingue de l'entité de base
    homonyme) ; absent → repli ``<classe>_source`` côté routage. Tolérant :
    section absente/malformée → ``[]``.
    """
    raw = card.get("derived_targets")
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    result: List[Tuple[str, str, bool, Optional[str], Optional[str]]] = []
    for cfg in raw:
        if not isinstance(cfg, dict):
            continue
        output = str(cfg.get("output_class") or "").strip()
        entity = str(cfg.get("entity") or "").strip()
        if not output or not entity:
            continue
        source_label = str(cfg.get("source_label") or "").strip() or None
        output_label = str(cfg.get("output_label") or "").strip() or None
        result.append((output, entity, bool(cfg.get("include_source", True)), source_label, output_label))
    return result


def _merge_derived_targets(
    coverage: Dict[str, Tuple[str, ...]],
    derived: Sequence[Tuple[str, str, bool, Optional[str], Optional[str]]],
    clustering_rules: Sequence[Tuple[FrozenSet[str], str]],
) -> Dict[str, Tuple[Tuple[str, ...], Optional[str], Optional[str]]]:
    """Replie les cibles dérivées dans ``coverage`` (mutation) ; renvoie leur méta.

    Chaque cible est rattachée à la règle de clustering dont l'``output_class_name``
    correspond, pour récupérer ses ``target_classes`` (classes sources). La
    couverture de l'entité vaut alors ``output_class`` (+ classes sources si
    ``include_source``), triée. Aucune règle correspondante → ignoré + warning
    (anti-dérive : la cible déclarée n'est plus produite par le modèle).

    Renvoie ``{entity_id: (source_classes, source_label, output_label)}`` : les
    classes sources (vides si ``include_source`` faux) + les libellés de couche
    optionnels, pour piloter le renommage des couches (routage entité-centré).
    """
    outputs = {output: targets for targets, output in clustering_rules}
    meta: Dict[str, Tuple[Tuple[str, ...], Optional[str], Optional[str]]] = {}
    for output_class, entity_id, include_source, source_label, output_label in derived:
        targets = outputs.get(output_class)
        if targets is None:
            logger.warning(
                "Cible dérivée '%s' → sortie de clustering '%s' introuvable, ignorée",
                entity_id, output_class,
            )
            continue
        classes = {output_class}
        source_classes: Tuple[str, ...] = ()
        if include_source:
            classes.update(targets)
            source_classes = tuple(sorted(targets))
        coverage[entity_id] = tuple(sorted(classes))
        meta[entity_id] = (source_classes, source_label, output_label)
    return meta


def _implied_entities(
    coverage: Dict[str, Tuple[str, ...]],
    derived_source_classes: Dict[str, Tuple[str, ...]],
) -> Dict[str, str]:
    """``{entité de base: entité dérivée}`` — les entités INCLUSES par une dérivée.

    Une entité de base dont toutes les classes sont des classes SOURCES d'une
    cible dérivée (``include_source``) n'a rien à produire en plus quand la
    dérivée est cochée : le routage n'écrit sa couche qu'une fois, dans le groupe
    de la dérivée (décision C). Cocher « Regroupement de cratères » inclut donc
    « Cratères ». ``include_source: false`` (sources vides) n'inclut rien.
    """
    out: Dict[str, str] = {}
    for derived, sources in derived_source_classes.items():
        src = set(sources)
        for eid, classes in coverage.items():
            if eid in derived_source_classes or not classes:
                continue
            if set(classes) <= src:
                out[eid] = derived
    return out


# ----------------------------------------------------------------------
# Couverture entité → modèles
# ----------------------------------------------------------------------
def build_entity_coverage(
    catalog: Sequence[EntityDef], installed: Sequence[InstalledModel]
) -> List[EntityCoverage]:
    """Pour chaque entité du catalogue, liste les modèles candidats + le défaut."""
    result: List[EntityCoverage] = []
    for entity in catalog:
        candidates = [m for m in installed if entity.id in m.coverage]
        result.append(
            EntityCoverage(
                entity=entity,
                candidate_models=tuple(sorted(m.name for m in candidates)),
                default_model=_pick_default_model(candidates),
            )
        )
    return result


def _pick_default_model(candidates: Sequence[InstalledModel]) -> Optional[str]:
    """Défaut = ``production`` d'abord, puis — à statut égal — le modèle porteur de
    seuils par classe MESURÉS (audit 2026-08-31 : sans ce critère, un modèle beta
    dont les seuils F1-max ont été mesurés perdait systématiquement contre un
    production jamais calibré), puis le plus spécialisé (moins de classes),
    départage alphabétique."""
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda m: (
            0 if m.status == "production" else 1,
            0 if m.default_confidence_per_class else 1,
            len(m.class_names),
            m.name,
        ),
    )
    return ranked[0].name


def _compute_layer_names(
    model: InstalledModel, eid: str, ent_classes: Sequence[str], label: str = ""
) -> Dict[str, str]:
    """Renommage des couches d'une entité ``classe → nom_de_couche``.

    Entité non dérivée (N-01, 2026-10-25) : la couche porte le **libellé de la
    structure** (« Cratères ») ; plusieurs classes → « <Structure> · <classe> »,
    sans collision ; sans ``label`` (appel historique), vide : la couche garde le
    nom de sa classe. Pour une entité dérivée :

    - classe **cluster** (sortie) : ``output_label`` si configuré, sinon le nom
      de classe est conservé (pas de renommage par défaut) ;
    - classe **source** : ``source_label`` si configuré, sinon repli
      ``<classe>_source`` (pour la distinguer de l'entité de base homonyme).

    Plusieurs clusters/sources + un seul libellé → libellé suffixé par la classe,
    pour éviter une collision de noms de couche dans le même GeoPackage.
    """
    if eid not in model.derived_entities:
        if not label:
            return {}
        return {c: (label if len(ent_classes) == 1 else f"{label} · {c}") for c in ent_classes}
    src = set(model.derived_source_classes.get(eid, ()))
    out_label = model.derived_output_labels.get(eid)
    src_label = model.derived_source_labels.get(eid)
    out_classes = [c for c in ent_classes if c not in src]
    src_classes = [c for c in ent_classes if c in src]
    names: Dict[str, str] = {}
    for oc in out_classes:
        if out_label:
            names[oc] = out_label if len(out_classes) == 1 else f"{out_label}_{oc}"
    for sc in src_classes:
        if src_label:
            names[sc] = src_label if len(src_classes) == 1 else f"{src_label}_{sc}"
        else:
            names[sc] = f"{sc}_source"
    return names


def layer_name_for_class(
    model: InstalledModel, eid: str, classe: str, compared: bool = False, label: str = ""
) -> str:
    """Nom de la couche de détections de ``classe`` pour l'entité ``eid`` — la
    **clé du registre de couleurs** (``class_color_registry``, indexé par nom de
    couche). Même règle que ``_entity_block`` : nom de classe (ou libellé d'une
    dérivée), suffixé « — Modèle » en comparaison A/B. Sert à l'UI pour donner
    à la figure du profil la couleur de la couche qu'elle décrit (A9, 2026-10-08).
    ``label`` = libellé de la structure, celui que ``_entity_block`` donne à la couche (N-01)."""
    noms = _compute_layer_names(model, eid, sorted(model.coverage.get(eid, ())), label=label)
    nom = noms.get(classe, classe)
    return f"{nom} — {model.display_name}" if compared else nom


# ----------------------------------------------------------------------
# Résolution des runs
# ----------------------------------------------------------------------
def _model_slug_qualifier(model_name: str) -> str:
    """Qualificatif de slug pour une variante comparée.

    Même classe de caractères que la re-sanitisation de
    ``output_paths.build_entity_class_targets`` (``[a-z0-9_-]``) pour que le
    slug qualifié traverse le routage sans altération, en PRÉSERVANT les
    tirets : « formes-v2 » et « formes_v2 » restent des qualificatifs
    distincts (``slugify`` les aurait confondus en ``formes_v2``).
    """
    folded = unicodedata.normalize("NFKD", model_name).encode("ascii", "ignore").decode("ascii")
    out = re.sub(r"[^a-z0-9_-]+", "_", folded.lower()).strip("_")
    return out or "modele"


def _entity_block(
    eid: str,
    model: InstalledModel,
    classes_sorted: List[str],
    label_by_id: Dict[str, str],
    *,
    compared: bool,
) -> Dict[str, Any]:
    """Bloc ``entities[i]`` d'un run pour une entité.

    ``compared=False`` (une entité = un modèle, historique) : slug/label/couches
    inchangés. ``compared=True`` (entité portée par ≥ 2 modèles, comparaison
    A/B) : les sorties sont **qualifiées par modèle** pour que chaque variante
    ait son dossier (``detections/<slug>--<modèle>/``), ses noms de couche et —
    via le registre indexé par nom de couche — sa couleur propres ; le
    ``group_label`` commun regroupe les variantes sous un même groupe QGIS
    (cf. ``finalize_service.build_entity_grouping``).
    """
    base_label = label_by_id.get(eid, eid)
    base_slug = slugify(base_label) or eid
    layer_names = _compute_layer_names(model, eid, classes_sorted, label=base_label)
    block: Dict[str, Any] = {
        "id": eid,
        "label": base_label,
        "slug": base_slug,
        "classes": classes_sorted,
        "is_derived": eid in model.derived_entities,
        "layer_names": layer_names,
    }
    if compared:
        block["slug"] = f"{base_slug}--{_model_slug_qualifier(model.name)}"
        block["label"] = f"{base_label} — {model.display_name}"
        block["group_label"] = f"{base_label} (comparaison)"
        block["layer_names"] = {
            c: f"{layer_names.get(c, c)} — {model.display_name}"
            for c in classes_sorted
        }
    return block


def effective_model_names(
    ec: EntityCoverage, overrides: Optional[Dict[str, Any]]
) -> List[str]:
    """Modèles effectifs d'une entité (1..n) : la surcharge UI filtrée aux
    candidats encore valides (installés ET couvrant l'entité), sinon le défaut.

    La surcharge (``computer_vision.entity_model_overrides``) accepte une
    **chaîne** (historique, un seul modèle) ou une **liste** de noms — plusieurs
    modèles cochés = mode comparaison A/B (un run par modèle en aval). Un membre
    périmé — model_card modifié entre deux sessions, modèle désinstallé — est
    filtré avec un ``logger.warning`` ; liste vide après filtrage → retour au
    modèle par défaut (l'entité ne disparaît jamais silencieusement, garde du
    commit 21d3113). Partagé par ``resolve_runs_from_entities`` et l'affichage
    des cartes (étape 3) pour que l'UI et le pipeline restent cohérents.
    """
    raw = (overrides or {}).get(ec.entity.id)
    if raw is None:
        names: List[str] = []
    elif isinstance(raw, str):
        names = [raw]
    elif isinstance(raw, (list, tuple, set)):
        names = [str(n) for n in raw]
    else:
        # Scalaire inattendu (int/bool… d'une config éditée à la main) : traité
        # comme une surcharge périmée → warning + repli défaut, jamais de crash
        # (même tolérance que la garde 21d3113 / AUDIT PARSE-04).
        names = [str(raw)]
    valid, seen = [], set()
    for name in names:
        if name in ec.candidate_models:
            if name not in seen:
                valid.append(name)
                seen.add(name)
        else:
            logger.warning(
                "Surcharge périmée pour '%s' : modèle '%s' invalide, ignoré "
                "(défaut : '%s')",
                ec.entity.id, name, ec.default_model,
            )
    if valid:
        return valid
    return [ec.default_model] if ec.default_model else []


def avec_extension(params: Dict[str, float], defauts: Dict[str, float]) -> Dict[str, float]:
    """Surcharges d'une règle de regroupement, seuil d'extension décalé avec « Confiance min ».

    Une règle à hystérésis (``min_confidence_extend`` < ``min_confidence`` dans args.yaml : les détections entre les
    deux rejoignent une zone sans pouvoir l'ouvrir) gardait son extension quand l'utilisateur relevait « Confiance
    min » : à 0,60 sur les cratères (0,40 / 0,29), les détections dès 0,29 s'agrégeaient encore. L'extension suit
    maintenant le réglage, à l'écart calibré du modèle. Non exposée à l'écran ; une surcharge explicite l'emporte.
    """
    sortie = dict(params or {})
    d_min, d_ext = defauts.get("min_confidence"), defauts.get("min_confidence_extend")
    if "min_confidence" in sortie and "min_confidence_extend" not in sortie and d_min is not None             and d_ext is not None:
        sortie["min_confidence_extend"] = round(max(0.0, float(d_ext) + float(sortie["min_confidence"]) - float(d_min)), 4)
    return sortie


def resolve_runs_from_entities(
    selected_entity_ids: Sequence[str],
    overrides: Optional[Dict[str, Any]],
    installed_models: Sequence[InstalledModel],
    catalog: Sequence[EntityDef],
    cluster_enabled: Optional[Set[str]] = None,
    entity_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
    entity_cluster_params: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[Dict[str, Any]]:
    """Résout les entités sélectionnées en liste de ``runs`` (schéma pipeline).

    Chaque run = ``{"model": slug, "target_rvt": rvt, "selected_classes": [..]}``.
    Regroupe par ``(modèle, target_rvt)``. ``selected_classes`` est **toujours
    une liste explicite triée** : les classes brutes des entités du groupe, plus
    — pour les entités présentes dans ``cluster_enabled`` — les sorties de
    clustering du modèle (ex. ``zone_crateres``). Ainsi le clustering ne se
    déclenche que si l'utilisateur l'a coché (cf. filtre de ``runner_shapefiles``
    sur ``output_class_name``). Entité hors catalogue / sans modèle → ignorée.

    Une entité dont la surcharge liste **plusieurs** modèles (comparaison A/B)
    tombe dans un run par modèle, avec des sorties qualifiées par modèle
    (cf. :func:`_entity_block`).
    """
    overrides = overrides or {}
    cluster_enabled = cluster_enabled or set()
    entity_thresholds = entity_thresholds or {}
    entity_cluster_params = entity_cluster_params or {}
    coverage_by_id = {ec.entity.id: ec for ec in build_entity_coverage(catalog, installed_models)}
    models_by_name = {m.name: m for m in installed_models}
    label_by_id = {e.id: e.label for e in catalog}

    # (modèle, rvt) -> {classes: set, entities: [ids]} pour pouvoir agréger les
    # seuils surchargés par entité au niveau du run. Une entité peut porter
    # PLUSIEURS modèles effectifs (comparaison A/B) → elle tombe alors dans un
    # groupe par modèle ; models_per_eid trace les modèles ayant réellement
    # retenu l'entité (pour qualifier ses sorties en aval).
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    models_per_eid: Dict[str, List[str]] = {}
    for eid in selected_entity_ids:
        ec = coverage_by_id.get(eid)
        if ec is None:
            logger.warning("Entité hors catalogue ignorée: %s", eid)
            continue
        model_names = effective_model_names(ec, overrides)
        if not model_names:
            logger.warning("Entité '%s' sans modèle disponible, ignorée", eid)
            continue
        for model_name in model_names:
            model = models_by_name.get(model_name)
            if model is None:
                logger.warning("Modèle '%s' (pour '%s') introuvable, ignoré", model_name, eid)
                continue
            classes = model.coverage.get(eid)
            if not classes:
                logger.warning("Modèle '%s' ne couvre pas l'entité '%s', ignoré", model_name, eid)
                continue
            ec_classes = set(classes)
            if eid in cluster_enabled:
                ec_classes.update(model.cluster_options.get(eid, ()))
            group = groups.setdefault(
                (model.name, model.target_rvt),
                {"classes": set(), "entities": [], "entity_classes": {}},
            )
            group["classes"].update(ec_classes)
            group["entities"].append(eid)
            group["entity_classes"][eid] = ec_classes
            models_per_eid.setdefault(eid, []).append(model.name)

    # Garde (audit 2026-08-31, §RVT ; refaite le 2026-10-31) : les runs d'un même TYPE d'indice partagent UN
    # raster, calculé avec les réglages de l'étape 2 — jamais avec ceux d'un modèle. On compare donc ce qui
    # compte physiquement, l'anneau du LD EN MÈTRES (les fiches l'écrivent en pixels à 0,5 m ou à 1 m, en clés
    # différentes : la comparaison des clés brutes signalait à tort enclos 5/10 px à 1 m contre cratères 10/20 px
    # à 0,5 m, le même anneau 5–10 m). L'écart avec le LD réellement calculé est dit au run
    # (avertissements_echelle, à la résolution des données).
    _anneaux: Dict[str, Dict[Tuple[float, float], List[str]]] = {}
    for model_name, rvt in groups:
        m = models_by_name[model_name]
        if rvt == "LD" and m.anneau_m is not None:
            _anneaux.setdefault(rvt, {}).setdefault(m.anneau_m, []).append(model_name)
    for rvt, variantes in _anneaux.items():
        if len(variantes) > 1:
            logger.warning(
                "LD partagé par des modèles appris sur des anneaux DIFFÉRENTS (%s) : un seul LD est calculé, "
                "avec l'anneau réglé à l'étape 2 — au moins un modèle le verra autrement qu'à l'entraînement",
                " ; ".join(f"{', '.join(sorted(n))} : {a[0]:g}–{a[1]:g} m" for a, n in sorted(variantes.items())),
            )

    runs: List[Dict[str, Any]] = []
    for key in sorted(groups):
        model_name, rvt = key
        model = models_by_name[model_name]
        group = groups[key]
        # Seuils de confiance PAR CLASSE : défauts du model_card, puis surcharge
        # par entité (UI) appliquée aux classes de CETTE entité seulement.
        # L'ancien comportement — min() de toutes les surcharges imposé au run
        # entier — donnait le seuil le plus bas à TOUTES les classes du run : une
        # entité non surchargée héritait du seuil d'une autre. Le scalaire
        # ``confidence_threshold`` reste émis comme PLANCHER de décodage (min des
        # seuils applicables) pour les consommateurs qui ne connaissent pas le
        # dict (log, garde-fou clustering) ; le filtre fin par classe est fait au
        # décodage ONNX via ``confidence_per_class``.
        conf_par_classe: Dict[str, float] = {
            c: v for c, v in model.default_confidence_per_class.items()
            if c in group["classes"]
        }
        # Entité INCLUSE par une cible dérivée de ce run (ex. Cratères sous
        # Regroupement de cratères) : son seuil se règle sur la carte de la
        # dérivée — ses surcharges propres sont ignorées, sinon « dernier gagne ».
        incluses = {b for b, d in model.implied_entities.items() if d in group["entities"]}
        reglees = [e for e in group["entities"] if e not in incluses]
        for e in reglees:
            ov_e = entity_thresholds.get(e, {})
            if "confidence_threshold" in ov_e:
                for c in group["entity_classes"][e]:
                    conf_par_classe[c] = float(ov_e["confidence_threshold"])
        plancher_conf = min(
            [conf_par_classe.get(c, model.default_confidence) for c in group["classes"]]
            or [model.default_confidence]
        )
        # Aire minimale PAR ENTITÉ (même correction que la confiance) : celle de l'entité si elle l'a réglée, sinon
        # le défaut du modèle. Le run garde la plus petite (filtre commun du run, runner_shapefiles), et chaque
        # entité dont l'aire est plus grande voit son GeoPackage filtré à la sienne après le run
        # (output_paths.aires_par_entite). L'ancien min() des seules surcharges imposait l'aire d'une entité aux
        # autres entités du modèle (« Chemins creux » perdait des détections réglées sur « Parcellaire »).
        aire_par_entite = {
            e: float(entity_thresholds.get(e, {}).get("min_area_m2", model.default_min_area))
            for e in group["entities"]
        }
        for e in incluses:   # entité incluse par une dérivée : réglée sur la carte de la dérivée
            derivee = model.implied_entities[e]
            aire_par_entite[e] = aire_par_entite.get(derivee, float(model.default_min_area))
        # Surcharges de paramètres de clustering, mappées par output_class_name
        # (ce que le pipeline applique). Une entité (dérivée, ou de base avec
        # « regrouper » coché) porte ses params ; on les rattache à la/les
        # sortie(s) de clustering présentes dans ses classes.
        clustering_overrides: Dict[str, Dict[str, float]] = {}
        for e in group["entities"]:
            params = entity_cluster_params.get(e)
            if not params:
                continue
            for oc in group["entity_classes"][e]:
                if oc in model.cluster_defaults:
                    clustering_overrides[oc] = avec_extension(params, model.cluster_defaults[oc])
        # Fiabilité affichée : catégories EFFECTIVES (au seuil de chaque classe dans
        # ce run, surcharge UI comprise) — consommées à la conversion (champs
        # fiabilite/fiabilite_pct + sidecar fiabilite.json) puis par la symbologie.
        fiab_bloc = run_block(
            model.fiabilite_per_class,
            {c: conf_par_classe.get(c, model.default_confidence) for c in group["classes"]},
            sorted(group["classes"]),
            modele=model.display_name,
            provenance=model.fiabilite_provenance,
            sous_seuil=model.fiabilite_sous_seuil,
        )
        runs.append(
            {
                "model": model_name,
                "target_rvt": rvt,
                "selected_classes": sorted(group["classes"]),
                "clustering_overrides": clustering_overrides,
                **({"fiabilite": fiab_bloc} if fiab_bloc else {}),
                "entities": [
                    {**_entity_block(
                        eid, model,
                        sorted(group["entity_classes"][eid]),
                        label_by_id,
                        compared=len(models_per_eid.get(eid, [])) > 1,
                    ), "min_area_m2": aire_par_entite[eid],
                        # sorties de regroupement : leur règle porte sa propre aire, jamais rabotées
                        "classes_synthese": sorted(c for c in group["entity_classes"][eid]
                                                   if c in model.cluster_defaults)}
                    for eid in sorted(group["entities"])
                ],
                "confidence_threshold": float(plancher_conf),
                "confidence_per_class": {c: float(v) for c, v in sorted(conf_par_classe.items())},
                # Seuil du MODÈLE par classe (2026-10-22) : la fusion intra-classe se fait en
                # deux passes de part et d'autre (postprocess_geo_detections) quand le seuil
                # du run est abaissé — les ajoutées fusionnent entre elles, jamais avec les bonnes.
                "seuil_fusion_par_classe": {
                    c: float(model.default_confidence_per_class.get(c, model.default_confidence))
                    for c in sorted(group["classes"])},
                "iou_threshold": float(model.default_iou),
                "min_area_m2": float(min(aire_par_entite.values(), default=model.default_min_area)),
            }
        )
    return runs
