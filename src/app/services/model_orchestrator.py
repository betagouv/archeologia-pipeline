"""Orchestrateur de modèles CV : entités → modèles → runs.

Dans la V2, l'utilisateur ne choisit plus des *modèles* puis filtre leurs
*classes* ; il coche des **entités** à détecter (parcellaire, trous d'obus…)
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

logger = logging.getLogger(__name__)


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
        result.append(
            EntityDef(
                id=eid,
                label=label,
                description=str(entry.get("description") or ""),
                display_order=order,
            )
        )
    result.sort(key=lambda e: (e.display_order, e.id))
    return result


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
        models.append(
            InstalledModel(
                name=sub.name,
                display_name=str(card.get("display_name") or sub.name),
                weights_path=_find_weights(sub),
                target_rvt=_extract_target_rvt(card),
                status=str(card.get("status") or "").strip(),
                coverage=coverage,
                class_names=class_names,
                cluster_options=_build_cluster_options(coverage, _load_args_clustering(sub)),
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


# ----------------------------------------------------------------------
# Résolution des runs
# ----------------------------------------------------------------------
def resolve_runs_from_entities(
    selected_entity_ids: Sequence[str],
    overrides: Optional[Dict[str, str]],
    installed_models: Sequence[InstalledModel],
    catalog: Sequence[EntityDef],
    cluster_enabled: Optional[Set[str]] = None,
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
    coverage_by_id = {ec.entity.id: ec for ec in build_entity_coverage(catalog, installed_models)}
    models_by_name = {m.name: m for m in installed_models}

    groups: Dict[Tuple[str, str], set] = {}
    for eid in selected_entity_ids:
        ec = coverage_by_id.get(eid)
        if ec is None:
            logger.warning("Entité hors catalogue ignorée: %s", eid)
            continue
        model_name = overrides.get(eid) or ec.default_model
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
        group = groups.setdefault((model.name, model.target_rvt), set())
        group.update(classes)
        if eid in cluster_enabled:
            group.update(model.cluster_options.get(eid, ()))

    runs: List[Dict[str, Any]] = []
    for key in sorted(groups):
        model_name, rvt = key
        runs.append(
            {"model": model_name, "target_rvt": rvt, "selected_classes": sorted(groups[key])}
        )
    return runs
