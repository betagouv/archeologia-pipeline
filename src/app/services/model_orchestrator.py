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
    # Seuils par défaut (model_card:thresholds) — injectés par run, surchargeables
    # par entité côté UI (confiance + aire min). IoU jamais exposé dans l'UI.
    # 0.3 = défaut UNIFIÉ de la chaîne CV (= pipeline.cv.model_config.DEFAULT_CONFIDENCE,
    # littéral ici pour ne pas coupler app→pipeline ; gardé par test_defauts_cv_unifies).
    default_confidence: float = 0.3
    default_min_area: float = 0.0
    default_iou: float = 0.5
    # Seuils de confiance PAR CLASSE (model_card:thresholds.confidence_per_class,
    # {nom de classe: seuil}). Mesure au banc : les optima par classe s'étalent de
    # 0,10 à 0,30 sur lineaires_seg_v2_1, un seuil unique sacrifie les classes
    # rares. Une classe absente du dict retombe sur ``default_confidence``.
    default_confidence_per_class: Dict[str, float] = field(default_factory=dict)
    # Dossier du modèle sur disque (``data/models/<name>/``). Utile côté UI pour
    # ouvrir le dossier dans l'explorateur ou (re)lire ``model_card.yaml`` /
    # ``args.yaml`` à la demande sans relancer ``discover_installed_models``.
    model_dir: Optional[Path] = None


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
        models.append(
            InstalledModel(
                name=sub.name,
                display_name=str(card.get("display_name") or sub.name),
                weights_path=_find_weights(sub),
                target_rvt=_extract_target_rvt(card),
                status=str(card.get("status") or "").strip(),
                coverage=coverage,
                class_names=class_names,
                cluster_options=cluster_options,
                cluster_defaults=_load_cluster_defaults(sub),
                derived_entities=frozenset(derived_meta.keys()),
                derived_source_classes={k: v[0] for k, v in derived_meta.items()},
                derived_source_labels={k: v[1] for k, v in derived_meta.items() if v[1]},
                derived_output_labels={k: v[2] for k, v in derived_meta.items() if v[2]},
                default_confidence=conf,
                default_confidence_per_class=conf_pc,
                default_min_area=area,
                default_iou=iou,
                model_dir=sub,
            )
        )
    return models


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
    "eps_m", "min_confidence", "min_area_m2", "buffer_m",
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
    """Défaut = modèle ``production`` le plus spécialisé (moins de classes),
    départage alphabétique."""
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda m: (0 if m.status == "production" else 1, len(m.class_names), m.name),
    )
    return ranked[0].name


def _compute_layer_names(
    model: InstalledModel, eid: str, ent_classes: Sequence[str]
) -> Dict[str, str]:
    """Renommage des couches d'une entité ``classe → nom_de_couche``.

    Vide pour une entité non dérivée (chaque couche garde le nom de sa classe).
    Pour une entité dérivée :

    - classe **cluster** (sortie) : ``output_label`` si configuré, sinon le nom
      de classe est conservé (pas de renommage par défaut) ;
    - classe **source** : ``source_label`` si configuré, sinon repli
      ``<classe>_source`` (pour la distinguer de l'entité de base homonyme).

    Plusieurs clusters/sources + un seul libellé → libellé suffixé par la classe,
    pour éviter une collision de noms de couche dans le même GeoPackage.
    """
    if eid not in model.derived_entities:
        return {}
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


# ----------------------------------------------------------------------
# Résolution des runs
# ----------------------------------------------------------------------
def effective_model_name(
    ec: EntityCoverage, overrides: Optional[Dict[str, str]]
) -> Optional[str]:
    """Modèle effectif d'une entité : la surcharge UI si elle est encore valide
    (modèle installé ET couvrant l'entité), sinon le modèle par défaut.

    Une surcharge périmée — model_card modifié entre deux sessions, modèle
    désinstallé — persiste dans ``last_ui_config.json`` ; sans cette garde elle
    faisait disparaître l'entité du run silencieusement (simple logger.warning).
    Partagé par ``resolve_runs_from_entities`` et l'affichage des cartes (étape 3)
    pour que l'UI et le pipeline restent cohérents.
    """
    name = (overrides or {}).get(ec.entity.id)
    if name in ec.candidate_models:
        return name
    if name:
        logger.warning(
            "Surcharge périmée pour '%s' : modèle '%s' invalide, retour au défaut '%s'",
            ec.entity.id, name, ec.default_model,
        )
    return ec.default_model


def resolve_runs_from_entities(
    selected_entity_ids: Sequence[str],
    overrides: Optional[Dict[str, str]],
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
    """
    overrides = overrides or {}
    cluster_enabled = cluster_enabled or set()
    entity_thresholds = entity_thresholds or {}
    entity_cluster_params = entity_cluster_params or {}
    coverage_by_id = {ec.entity.id: ec for ec in build_entity_coverage(catalog, installed_models)}
    models_by_name = {m.name: m for m in installed_models}
    label_by_id = {e.id: e.label for e in catalog}

    # (modèle, rvt) -> {classes: set, entities: [ids]} pour pouvoir agréger les
    # seuils surchargés par entité au niveau du run.
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for eid in selected_entity_ids:
        ec = coverage_by_id.get(eid)
        if ec is None:
            logger.warning("Entité hors catalogue ignorée: %s", eid)
            continue
        model_name = effective_model_name(ec, overrides)
        if not model_name:
            logger.warning("Entité '%s' sans modèle disponible, ignorée", eid)
            continue
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
        for e in group["entities"]:
            ov_e = entity_thresholds.get(e, {})
            if "confidence_threshold" in ov_e:
                for c in group["entity_classes"][e]:
                    conf_par_classe[c] = float(ov_e["confidence_threshold"])
        plancher_conf = min(
            [conf_par_classe.get(c, model.default_confidence) for c in group["classes"]]
            or [model.default_confidence]
        )
        area_over = [
            entity_thresholds[e]["min_area_m2"]
            for e in group["entities"]
            if e in entity_thresholds and "min_area_m2" in entity_thresholds[e]
        ]
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
                    clustering_overrides[oc] = dict(params)
        runs.append(
            {
                "model": model_name,
                "target_rvt": rvt,
                "selected_classes": sorted(group["classes"]),
                "clustering_overrides": clustering_overrides,
                "entities": [
                    {
                        "id": eid,
                        "label": label_by_id.get(eid, eid),
                        "slug": slugify(label_by_id.get(eid, eid)) or eid,
                        "classes": sorted(group["entity_classes"][eid]),
                        "is_derived": eid in model.derived_entities,
                        "layer_names": _compute_layer_names(
                            model, eid, sorted(group["entity_classes"][eid])
                        ),
                    }
                    for eid in sorted(group["entities"])
                ],
                "confidence_threshold": float(plancher_conf),
                "confidence_per_class": {c: float(v) for c, v in sorted(conf_par_classe.items())},
                "iou_threshold": float(model.default_iou),
                "min_area_m2": float(min(area_over) if area_over else model.default_min_area),
            }
        )
    return runs
