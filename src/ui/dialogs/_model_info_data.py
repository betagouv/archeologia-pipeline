"""Logique pure du dialog d'info modèle — testable hors-QGIS.

Pas d'import Qt, pas d'I/O. Le builder prend un ``model_card.yaml`` déjà
parsé (et optionnellement un ``args.yaml`` pour les règles de clustering) et
produit une liste de :class:`Section` à afficher. Cela isole tout ce qui peut
être testé sans QGIS et garde le widget Qt (``model_info_dialog.py``) minimal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple


# ----------------------------------------------------------------------
# Dataclasses de présentation
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Row:
    """Une ligne ``label : value`` dans une section."""
    label: str
    value: str
    mono: bool = False  # rendre la valeur en police monospace (filter_expression…)


@dataclass(frozen=True)
class Section:
    """Une section pliable du dialog (titre + lignes)."""
    title: str
    rows: Tuple[Row, ...]
    collapsed: bool = False  # ouvert par défaut


# ----------------------------------------------------------------------
# Humanisation (codes courts → libellés longs FR)
# ----------------------------------------------------------------------
_RVT_LONG_NAMES: Dict[str, str] = {
    "LD": "Local Dominance (LD)",
    "SVF": "Sky View Factor (SVF)",
    "M_HS": "Hillshade multi-directionnel (M-HS)",
    "HS": "Hillshade simple (HS)",
    "SLO": "Pente (SLO)",
    "SLRM": "Simple Local Relief Model (SLRM)",
    "VAT": "Visualisation Archéologique Totale (VAT)",
    "MSTP": "Multi-Scale Topographic Position (MSTP)",
    "CVAT": "Combined VAT (CVAT)",
}

_TASK_LABELS: Dict[str, str] = {
    "object_detection": "Détection d'objets",
    "instance_segmentation": "Segmentation d'instances",
    "semantic_segmentation": "Segmentation sémantique",
}


def pretty_rvt_name(code: str) -> str:
    """Code court (``LD``, ``SVF``, …) → libellé long FR. Repli : code brut."""
    raw = code or ""
    return _RVT_LONG_NAMES.get(raw.upper(), raw)


def pretty_task(code: str) -> str:
    """Code de tâche (``object_detection``…) → libellé FR. Repli : valeur brute."""
    raw = code or ""
    return _TASK_LABELS.get(raw, raw)


# ----------------------------------------------------------------------
# Libellés des paramètres RVT (canoniques + alias non-canoniques connus)
# ----------------------------------------------------------------------
# Pour chaque type RVT, mapping ``clé_yaml → libellé FR``. Inclut des alias
# non-canoniques rencontrés en pratique (ex. ``verdun_3_classes_1`` utilise
# ``svf_n_dir`` / ``svf_r_max`` / ``svf_noise`` au lieu des clés canoniques
# ``num_directions`` / ``radius`` / ``noise_remove``) pour offrir une UI
# uniforme côté utilisateur. Une clé inconnue d'un type est affichée brute par
# le builder, sans planter.
RVT_PARAM_LABELS: Dict[str, Dict[str, str]] = {
    "HS": {
        "sun_azimuth": "Azimut solaire",
        "sun_elevation": "Élévation solaire",
        "ve_factor": "Facteur VE",
        "save_as_8bit": "Export 8 bits",
    },
    "M_HS": {
        "num_directions": "Nombre de directions",
        "sun_elevation": "Élévation solaire",
        "ve_factor": "Facteur VE",
        "save_as_8bit": "Export 8 bits",
    },
    "SVF": {
        # Canoniques (config_manager / indices.py).
        "num_directions": "Nombre de directions",
        "radius": "Rayon",
        "noise_remove": "Suppression du bruit",
        "ve_factor": "Facteur VE",
        "save_as_8bit": "Export 8 bits",
        # Alias non-canoniques rencontrés (verdun_3_classes_1/model_card.yaml).
        "svf_n_dir": "Nombre de directions",
        "svf_r_max": "Rayon",
        "svf_noise": "Suppression du bruit",
    },
    "SLO": {
        "unit": "Unité (0 = °, 1 = %)",
        "ve_factor": "Facteur VE",
        "save_as_8bit": "Export 8 bits",
    },
    "LD": {
        "angular_res": "Résolution angulaire",
        "min_radius": "Rayon min",
        "max_radius": "Rayon max",
        "observer_h": "Hauteur observateur",
        "ve_factor": "Facteur VE",
        "save_as_8bit": "Export 8 bits",
    },
    "SLRM": {
        "radius": "Rayon de lissage",
        "ve_factor": "Facteur VE",
        "save_as_8bit": "Export 8 bits",
    },
    "VAT": {
        "terrain_type": "Type de terrain (0 = général, 1 = plat)",
        "blend_combination": "Combinaison de fusion",
        "save_as_8bit": "Export 8 bits",
    },
    "MSTP": {
        "local_scale_min": "Échelle locale — min",
        "local_scale_max": "Échelle locale — max",
        "local_scale_step": "Échelle locale — pas",
        "meso_scale_min": "Échelle méso — min",
        "meso_scale_max": "Échelle méso — max",
        "meso_scale_step": "Échelle méso — pas",
        "broad_scale_min": "Échelle large — min",
        "broad_scale_max": "Échelle large — max",
        "broad_scale_step": "Échelle large — pas",
        "lightness": "Luminosité",
        "ve_factor": "Facteur VE",
        "save_as_8bit": "Export 8 bits",
    },
    "CVAT": {
        "save_as_8bit": "Export 8 bits",
    },
}


# Libellés FR des paramètres DBSCAN (``args.yaml:clustering[i]``).
_CLUSTER_LABELS: Dict[str, str] = {
    "target_classes": "Classes source",
    "output_class_name": "Classe de sortie",
    "eps_m": "Distance maximale (m)",
    "min_cluster_size": "Taille minimale du groupe",
    "min_samples": "Voisins minimum (DBSCAN)",
    "min_confidence": "Confiance minimale",
    "buffer_m": "Marge (m)",
    "min_area_m2": "Surface minimale (m²)",
    "output_geometry": "Géométrie de sortie",
    "confidence_weight": "Pondération par confiance",
}


# ----------------------------------------------------------------------
# Helpers internes
# ----------------------------------------------------------------------
def _fmt_value(v: Any) -> str:
    """Sérialise une valeur YAML pour l'affichage (pas de ``True``/``False`` bruts)."""
    if isinstance(v, bool):
        return "Oui" if v else "Non"
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return str(v)


def _unique_class_names(classes: Any) -> List[str]:
    """Liste des noms de classes uniques (ordre d'apparition).

    Plusieurs modèles déclarent volontairement des doublons (ex.
    ``run_rf_detr_1`` : 3 sous-types ``charbonniere`` fusionnés en sortie).
    """
    seen: List[str] = []
    if not isinstance(classes, list):
        return seen
    for c in classes:
        if not isinstance(c, Mapping):
            continue
        name = str(c.get("name") or "").strip()
        if name and name not in seen:
            seen.append(name)
    return seen


# ----------------------------------------------------------------------
# Builders de sections
# ----------------------------------------------------------------------
def _build_architecture(card: Mapping[str, Any]) -> Optional[Section]:
    rows: List[Row] = []
    arch = card.get("architecture")
    variant = card.get("variant")
    if arch:
        model_value = str(arch) + (f" ({variant})" if variant else "")
        rows.append(Row("Modèle", model_value))
    task = card.get("task")
    if task:
        rows.append(Row("Tâche", pretty_task(str(task))))
    res_inf = card.get("resolution_inference")
    res_train = card.get("resolution_train")
    if res_inf is not None:
        if res_train is not None and res_train != res_inf:
            rows.append(Row("Taille d'image", f"{res_inf} px (entr. {res_train})"))
        else:
            rows.append(Row("Taille d'image", f"{res_inf} px"))
    names = _unique_class_names(card.get("classes"))
    if names:
        # Compte dans le label (« Classes (n) »), liste dans la valeur — cohérent
        # avec la maquette utilisateur.
        rows.append(Row(f"Classes ({len(names)})", ", ".join(names)))
    if not rows:
        return None
    return Section(title="ARCHITECTURE", rows=tuple(rows))


def _build_rvt(card: Mapping[str, Any]) -> Optional[Section]:
    pref = card.get("preferred_rvt")
    if not isinstance(pref, Mapping):
        return None
    rvt_type = str(pref.get("type") or "").upper().strip()
    if not rvt_type:
        return None
    params = pref.get("params")
    rows: List[Row] = []
    if isinstance(params, Mapping):
        labels_map = RVT_PARAM_LABELS.get(rvt_type, {})
        for key, value in params.items():
            label = labels_map.get(str(key), str(key))
            rows.append(Row(label, _fmt_value(value)))
    title = f"INDICE RVT D'ENTRAÎNEMENT — {pretty_rvt_name(rvt_type)}"
    return Section(title=title, rows=tuple(rows))


def _build_mnt(card: Mapping[str, Any]) -> Optional[Section]:
    mnt = card.get("mnt")
    if not isinstance(mnt, Mapping):
        return None
    rows: List[Row] = []
    res = mnt.get("resolution")
    if res is not None:
        rows.append(Row("Résolution", f"{res} m/px"))
    filt = mnt.get("filter_expression")
    if filt:
        rows.append(Row("Filtre LiDAR", str(filt), mono=True))
    if not rows:
        return None
    return Section(title="MNT D'ENTRAÎNEMENT", rows=tuple(rows))


def _build_clustering(args: Optional[Mapping[str, Any]]) -> Optional[Section]:
    if not isinstance(args, Mapping):
        return None
    rules = args.get("clustering") or []
    if not isinstance(rules, list) or not rules:
        return None
    rows: List[Row] = []
    for i, rule in enumerate(rules):
        if not isinstance(rule, Mapping):
            continue
        if len(rules) > 1 and i > 0:
            rows.append(Row("—", "—"))  # séparateur visuel
        for key, value in rule.items():
            label = _CLUSTER_LABELS.get(str(key), str(key))
            rows.append(Row(label, _fmt_value(value)))
    if not rows:
        return None
    return Section(title="REGROUPEMENT (DBSCAN)", rows=tuple(rows), collapsed=True)


def _build_notes(card: Mapping[str, Any]) -> Optional[Section]:
    rows: List[Row] = []
    rec = card.get("recommended_use")
    if rec:
        rows.append(Row("Usage recommandé", str(rec)))
    lims = card.get("known_limitations")
    if isinstance(lims, list) and lims:
        for i, lim in enumerate(lims, 1):
            rows.append(Row(f"Limite #{i}", str(lim)))
    choices = card.get("inference_choices")
    if isinstance(choices, list) and choices:
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            field_ = choice.get("field")
            value = choice.get("value")
            reason = str(choice.get("reason") or "").strip()
            label = f"Choix d'inférence : {field_}" if field_ else "Choix d'inférence"
            text = f"{value}" if not reason else f"{value} — {reason}"
            rows.append(Row(label, text))
    if not rows:
        return None
    return Section(title="NOTES & LIMITES", rows=tuple(rows), collapsed=True)


# ----------------------------------------------------------------------
# API publique
# ----------------------------------------------------------------------
def build_sections(
    card: Mapping[str, Any],
    args: Optional[Mapping[str, Any]] = None,
) -> List[Section]:
    """Construit la liste ordonnée des sections à afficher dans le dialog.

    Sections principales (ouvertes par défaut, omises si vides) :
      1. ARCHITECTURE (architecture, tâche, taille d'image, classes)
      2. INDICE RVT D'ENTRAÎNEMENT — *<nom long>*
      3. MNT D'ENTRAÎNEMENT

    Sections secondaires (fermées par défaut, conditionnelles) :
      4. REGROUPEMENT (DBSCAN)  — si ``args["clustering"]`` n'est pas vide
      5. NOTES & LIMITES         — si ``recommended_use`` / ``known_limitations``
         / ``inference_choices`` sont présents
    """
    if not isinstance(card, Mapping):
        return []
    sections: List[Section] = []
    for builder in (_build_architecture, _build_rvt, _build_mnt):
        s = builder(card)
        if s is not None:
            sections.append(s)
    cluster = _build_clustering(args)
    if cluster is not None:
        sections.append(cluster)
    notes = _build_notes(card)
    if notes is not None:
        sections.append(notes)
    return sections
