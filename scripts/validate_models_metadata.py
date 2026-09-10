"""Validateur du contrat de standardisation des modèles CV.

Vérifie que chaque dossier modèle dans ``data/models/<nom>/`` respecte le contrat
décrit dans ``docs/model_contract.md`` :

- Fichiers obligatoires présents.
- Cohérences inter-fichiers (classes, num_classes, task, resolution_train).
- ``classes.txt`` snake_case ASCII, doublons autorisés, pas de ligne vide.
- ``args.yaml.class_colors`` longueur cohérente.
- ``clustering.target_classes`` ⊆ classes.txt, ``output_class_name`` ∉ classes.txt.
- ``weights/best.json.source`` relatif (pas de chemin absolu local).
- Divergences ``imgsz`` / SAHI documentées dans ``model_card.inference_choices``.

Vérifications v2 (audit 2026-08-31) :

- ``inference_choices[].value`` == la valeur RÉELLE dans args.yaml (le cas
  cratere_circulaire_2 déclarait sahi 350 pour un args.yaml à 140 — l'UI relayait
  le mensonge en vert).
- ``thresholds`` bornés ; ``confidence_per_class`` ⊆ classes.txt.
- ``confidence_default`` / ``confidence_per_class`` adossés à
  ``entrainement/evaluation/metriques_eval.json`` — règle 2026-09-09 : quand l'éval
  porte le bloc ``etude_seuil``, ``seuils_provenance`` est REQUIS (le choix est une
  étude, pas une formule) et un seuil hors de la fenêtre [bas du plateau F1 ≥ 95 % ;
  F1-max], ou posé AU F1-max, est signalé (WARN) ; éval antérieure sans le bloc :
  |Δ| ≤ 0,05 avec le seuil_f1max mesuré, sinon ``seuils_provenance`` obligatoire.
- entités (classes[].entity/name, derived_targets[].entity) ⊆ entities_catalog.json
  (une entité hors catalogue = modèle installé mais INVISIBLE dans l'UI).
- ``weights/best.json.source`` pointe un fichier existant.
- ``derived_targets[].output_class`` a sa règle ``clustering.output_class_name``.
- ``best.json.confidence_threshold`` divergent du model_card = ERR (il ÉCRASE le
  seuil UI côté binaire en segmentation).

Utilisation:
    python scripts/validate_models_metadata.py
    python scripts/validate_models_metadata.py data/models/cratere_circulaire_2
    python scripts/validate_models_metadata.py --strict
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = REPO_ROOT / "data" / "models"
if str(REPO_ROOT / "src") not in sys.path:  # app.services.fiabilite (contrat des catégories)
    sys.path.insert(0, str(REPO_ROOT / "src"))

VALID_TASKS = {"object_detection", "instance_segmentation", "semantic_segmentation"}
CLASS_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
ABS_PATH_RE = re.compile(r"^([A-Za-z]:[\\/]|/Users/|/home/|/root/)")


@dataclass
class ValidationReport:
    model_dir: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    infos: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def status(self) -> str:
        if self.errors:
            return "ERR"
        if self.warnings:
            return "WARN"
        return "OK"


def find_model_dirs(models_root: Path) -> list[Path]:
    """Retourne les dossiers candidats sous ``models_root`` (présence de ``weights/``)."""
    if not models_root.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(models_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "weights").is_dir():
            out.append(child)
    return out


def _bandes_fiabilite(model_dir: Path, fiab_raw: dict, metriques_par_classe: dict,
                      report: "ValidationReport") -> tuple[dict, dict]:
    """``(bloc par_classe, bloc par_zone_classe)`` de l'évaluation qui a calibré la
    fiabilité : ``fiabilite.source`` (chemin relatif, ex. ``entrainement/
    evaluation_couverture/metriques_eval.json`` pour le critère couverture des
    linéaires), sinon l'évaluation canonique déjà lue."""
    source = str(fiab_raw.get("source") or "").strip()
    if not source:
        return metriques_par_classe, {}
    path = model_dir / source
    if not path.is_file():
        report.errors.append(f"thresholds.fiabilite.source introuvable : {source}")
        return {}, {}
    try:
        m = json.loads(path.read_text(encoding="utf-8-sig"))
        modeles = m.get("modeles") or {}
        bloc = modeles.get(model_dir.name) or (next(iter(modeles.values())) if modeles else {})
        return (bloc.get("par_classe") or {}), (bloc.get("par_zone_classe") or {})
    except (ValueError, AttributeError) as exc:
        report.warnings.append(f"thresholds.fiabilite.source illisible : {exc}")
        return {}, {}


def _validate_fiabilite(
    fiab_raw: Any,
    mc_thresholds: dict,
    class_set: set,
    conf_default: Any,
    pc: Any,
    metriques_par_classe: dict,
    report: "ValidationReport",
    model_dir: Path | None = None,
) -> None:
    """``thresholds.fiabilite`` (2026-09-09) : classes ⊆ classes.txt, listes valides
    (``app.services.fiabilite.parse_fiabilite``), première catégorie AU seuil de la
    classe, provenance présente ; mesures/effectifs re-dérivés des ``bandes`` de
    l'évaluation source (``fiabilite.source``, sinon la canonique), restreintes aux
    ``fiabilite.zones`` (zones à annotation exhaustive) quand elles sont déclarées
    (écart → WARN)."""
    try:
        from app.services.fiabilite import CATEGORIES, N_MIN_MESURE, parse_fiabilite
    except ImportError:
        report.warnings.append("app.services.fiabilite introuvable : thresholds.fiabilite non vérifié")
        return
    if not isinstance(fiab_raw, dict) or not isinstance(fiab_raw.get("par_classe"), dict):
        report.errors.append(
            "thresholds.fiabilite doit être un mapping {par_classe: {classe: [catégories]}, provenance}"
        )
        return
    par_classe, provenance = parse_fiabilite(mc_thresholds)
    if not provenance:
        report.warnings.append(
            "thresholds.fiabilite.provenance manquante (mesure + date + relecture des coupures)"
        )
    if model_dir is not None:
        metriques_par_classe, par_zone_classe = _bandes_fiabilite(
            model_dir, fiab_raw, metriques_par_classe, report)
    else:
        par_zone_classe = {}
    zones_raw = fiab_raw.get("zones")  # liste (toutes classes) OU {classe: liste}
    zones_par_classe: dict = {}
    if isinstance(zones_raw, list) and all(isinstance(z, str) for z in zones_raw):
        zones_par_classe = {c: list(zones_raw) for c in fiab_raw["par_classe"]}
    elif isinstance(zones_raw, dict) and all(
        isinstance(v, list) and all(isinstance(z, str) for z in v) for v in zones_raw.values()
    ):
        zones_par_classe = {str(c): list(v) for c, v in zones_raw.items()}
    elif zones_raw is not None:
        report.errors.append(
            "thresholds.fiabilite.zones : liste d'ids de zone, ou mapping {classe: [zones]}"
        )
    zones = bool(zones_par_classe)
    if zones and not par_zone_classe:
        report.warnings.append(
            "thresholds.fiabilite.zones déclarées mais l'évaluation source n'a pas de "
            "par_zone_classe.bandes (completer_metriques_eval.py) — mesures non re-dérivées"
        )
    for classe, _liste in fiab_raw["par_classe"].items():
        if class_set and classe not in class_set:
            report.errors.append(
                f"thresholds.fiabilite.par_classe : '{classe}' absent de classes.txt "
                "(catégories silencieusement ignorées au runtime)"
            )
            continue
        if classe not in par_classe:
            report.errors.append(
                f"thresholds.fiabilite.par_classe['{classe}'] invalide : catégories ∈ "
                f"{list(CATEGORIES)} dans cet ordre, seuils strictement croissants dans [0,1], "
                "garanti/mesure ∈ [0,1] (mesure null autorisée), n entier"
            )
            continue
        cats = par_classe[classe]
        seuil_classe = pc.get(classe) if isinstance(pc, dict) and classe in pc else conf_default
        if isinstance(seuil_classe, (int, float)) and abs(cats[0].seuil - float(seuil_classe)) > 1e-6:
            report.errors.append(
                f"thresholds.fiabilite.par_classe['{classe}'] : première catégorie à "
                f"{cats[0].seuil} ≠ seuil de la classe {seuil_classe} — la catégorie basse "
                "commence AU seuil (invariant seuil = symbologie = filtrage)"
            )
        zones_classe = zones_par_classe.get(classe) if zones else None
        if zones_classe and par_zone_classe:
            # somme des tables de calibrage des zones déclarées (annotation exhaustive)
            manquantes = [z for z in zones_classe if z not in par_zone_classe]
            if manquantes:
                report.errors.append(
                    f"thresholds.fiabilite.zones['{classe}'] : zone(s) absente(s) de "
                    f"l'évaluation source : {manquantes}"
                )
                continue
            bandes = None
            for z in zones_classe:
                bz = ((par_zone_classe.get(z) or {}).get(classe) or {}).get("bandes")
                if not isinstance(bz, list):
                    bandes = None
                    break
                if bandes is None:
                    bandes = [dict(b) for b in bz]
                else:
                    for acc, b in zip(bandes, bz):
                        acc["tp"] += int(b.get("tp", 0))
                        acc["fp"] += int(b.get("fp", 0))
        else:
            bandes = ((metriques_par_classe.get(classe) or {}).get("etude_seuil") or {}).get("bandes")
        if not (isinstance(bandes, list) and bandes):
            continue
        for i, c in enumerate(cats):
            fin = cats[i + 1].seuil if i + 1 < len(cats) else 1.01
            dans = [b for b in bandes if c.seuil - 1e-9 <= float(b.get("lo", 0)) < fin - 1e-9]
            tp = sum(int(b.get("tp", 0)) for b in dans)
            fp = sum(int(b.get("fp", 0)) for b in dans)
            n = tp + fp
            mesure = round(tp / n, 3) if n >= N_MIN_MESURE else None
            if n != c.n or (mesure is None) != (c.mesure is None) or (
                mesure is not None and abs(mesure - float(c.mesure)) > 0.0051
            ):
                report.warnings.append(
                    f"thresholds.fiabilite.par_classe['{classe}'][{c.categorie}] : mesure/n "
                    f"{c.mesure}/{c.n} ≠ re-dérivés des bandes de metriques_eval.json "
                    f"({mesure}/{n}) — recalculer (courbes_eval.fiabilite_par_classe)"
                )


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON invalide : {path.name} ({exc})") from exc


def _read_yaml(path: Path) -> dict[str, Any] | None:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return None
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML invalide : {path.name} ({exc})") from exc


def _read_classes_txt(path: Path) -> list[str] | None:
    if not path.is_file():
        return None
    raw = path.read_text(encoding="utf-8-sig")
    return raw.splitlines()


def _check_required_files(model_dir: Path, report: ValidationReport) -> None:
    required = [
        "model_card.yaml",
        "args.yaml",
        "classes.txt",
        "training_params.json",
        "config.json",
        "weights/best.onnx",
        "weights/best.json",
    ]
    for rel in required:
        if not (model_dir / rel).is_file():
            report.errors.append(f"Fichier obligatoire absent : {rel}")
    if not (model_dir / "evaluation_results.json").is_file():
        report.warnings.append("evaluation_results.json absent (recommandé)")


def _check_classes_txt(lines: list[str], report: ValidationReport) -> list[str]:
    """Valide ``classes.txt`` et retourne la liste nettoyée (sans lignes vides finales)."""
    if not lines:
        report.errors.append("classes.txt : fichier vide")
        return []
    cleaned: list[str] = []
    for i, raw in enumerate(lines):
        if raw == "" and i == len(lines) - 1:
            continue
        if raw == "":
            report.errors.append(f"classes.txt ligne {i+1} : ligne vide non autorisée")
            continue
        if not CLASS_NAME_RE.match(raw):
            report.errors.append(
                f"classes.txt ligne {i+1} : '{raw}' ne match pas snake_case ASCII (^[a-z][a-z0-9_]*$)"
            )
        cleaned.append(raw)
    return cleaned


def _check_task_value(task: Any, where: str, report: ValidationReport) -> None:
    if task not in VALID_TASKS:
        report.errors.append(
            f"{where} : task='{task}' invalide (attendu : {sorted(VALID_TASKS)})"
        )


def _check_no_absolute_path(value: Any, where: str, report: ValidationReport) -> None:
    if not isinstance(value, str):
        return
    if ABS_PATH_RE.match(value):
        report.errors.append(f"{where} : chemin absolu local détecté ({value!r})")


def _equal_lists(a: list[Any], b: list[Any]) -> bool:
    return list(a) == list(b)


def validate_model_dir(model_dir: Path, strict: bool = False) -> ValidationReport:
    """Valide un dossier modèle et retourne le rapport."""
    report = ValidationReport(model_dir=model_dir)
    _check_required_files(model_dir, report)

    # Charger ce qu'on peut, même si certains fichiers manquent (on reportera errors).
    try:
        args_yaml = _read_yaml(model_dir / "args.yaml") or {}
    except ValueError as exc:
        report.errors.append(str(exc))
        args_yaml = {}
    try:
        model_card = _read_yaml(model_dir / "model_card.yaml") or {}
    except ValueError as exc:
        report.errors.append(str(exc))
        model_card = {}
    try:
        training_params = _read_json(model_dir / "training_params.json") or {}
    except ValueError as exc:
        report.errors.append(str(exc))
        training_params = {}
    try:
        config = _read_json(model_dir / "config.json") or {}
    except ValueError as exc:
        report.errors.append(str(exc))
        config = {}
    try:
        best_json = _read_json(model_dir / "weights" / "best.json") or {}
    except ValueError as exc:
        report.errors.append(str(exc))
        best_json = {}

    raw_lines = _read_classes_txt(model_dir / "classes.txt")
    classes_txt = _check_classes_txt(raw_lines, report) if raw_lines is not None else []

    # weights/classes.txt (toléré, doit être byte-identique si présent)
    weights_classes = model_dir / "weights" / "classes.txt"
    if weights_classes.is_file():
        root_classes = model_dir / "classes.txt"
        if root_classes.is_file():
            if weights_classes.read_bytes() != root_classes.read_bytes():
                msg = "weights/classes.txt diffère de classes.txt racine (byte-identique requis)"
                if strict:
                    report.errors.append(msg)
                else:
                    report.warnings.append(msg)

    # ----- args.yaml -----
    args_task = args_yaml.get("task")
    args_imgsz = args_yaml.get("imgsz")
    args_sahi = args_yaml.get("sahi") or {}
    args_class_colors = args_yaml.get("class_colors")
    args_clustering = args_yaml.get("clustering") or []

    if args_task is not None:
        _check_task_value(args_task, "args.yaml.task", report)

    if args_class_colors is None:
        report.errors.append("args.yaml.class_colors absent (obligatoire dans le contrat)")
    elif not isinstance(args_class_colors, list):
        report.errors.append("args.yaml.class_colors doit être une liste")
    elif classes_txt and len(args_class_colors) != len(classes_txt):
        report.errors.append(
            f"args.yaml.class_colors longueur {len(args_class_colors)} != "
            f"classes.txt ({len(classes_txt)} lignes)"
        )

    class_set = set(classes_txt)
    for i, rule in enumerate(args_clustering):
        if not isinstance(rule, dict):
            report.errors.append(f"args.yaml.clustering[{i}] doit être un mapping")
            continue
        targets = rule.get("target_classes") or []
        for t in targets:
            if t not in class_set:
                report.errors.append(
                    f"args.yaml.clustering[{i}].target_classes : '{t}' absent de classes.txt"
                )
        out_name = rule.get("output_class_name")
        if out_name and out_name in class_set:
            report.errors.append(
                f"args.yaml.clustering[{i}].output_class_name : "
                f"'{out_name}' collision avec classes.txt"
            )

    # ----- weights/best.json -----
    bj_task = best_json.get("task")
    bj_class_names = best_json.get("class_names")
    bj_num_classes = best_json.get("num_classes")
    bj_resolution = best_json.get("resolution")
    bj_source = best_json.get("source")
    if bj_task is not None:
        _check_task_value(bj_task, "weights/best.json.task", report)
    if bj_source is not None:
        _check_no_absolute_path(bj_source, "weights/best.json.source", report)

    # ----- config.json -----
    cfg_task = config.get("task")
    cfg_model = config.get("model") or {}
    cfg_class_names = cfg_model.get("class_names")
    cfg_num_classes = cfg_model.get("num_classes")
    cfg_resolution = cfg_model.get("resolution")
    if cfg_task is not None:
        _check_task_value(cfg_task, "config.json.task", report)

    # ----- training_params.json -----
    tp_model = training_params.get("model") or {}
    tp_task = tp_model.get("task")
    tp_imgsz = tp_model.get("imgsz")
    if tp_task is not None:
        _check_task_value(tp_task, "training_params.json.model.task", report)

    # ----- model_card.yaml -----
    mc_task = model_card.get("task")
    mc_classes = model_card.get("classes") or []
    mc_class_names = [c.get("name") for c in mc_classes if isinstance(c, dict)]
    mc_res_train = model_card.get("resolution_train")
    mc_res_inf = model_card.get("resolution_inference")
    mc_inference_choices = model_card.get("inference_choices") or []
    if mc_task is not None:
        _check_task_value(mc_task, "model_card.task", report)

    # ----- Cohérence noms de classes (ordre + valeurs) -----
    expected = classes_txt
    if expected:
        for label, candidate in (
            ("config.json.model.class_names", cfg_class_names),
            ("weights/best.json.class_names", bj_class_names),
            ("model_card.classes[].name", mc_class_names if mc_classes else None),
        ):
            if candidate is None:
                report.errors.append(f"{label} absent")
                continue
            if not isinstance(candidate, list):
                report.errors.append(f"{label} doit être une liste")
                continue
            if not _equal_lists(expected, candidate):
                report.errors.append(
                    f"{label} != classes.txt : {candidate} vs {expected}"
                )

    # ----- Cohérence num_classes -----
    if expected:
        n = len(expected)
        for label, candidate in (
            ("weights/best.json.num_classes", bj_num_classes),
            ("config.json.model.num_classes", cfg_num_classes),
        ):
            if candidate is None:
                report.errors.append(f"{label} absent")
            elif candidate != n:
                report.errors.append(f"{label}={candidate} != len(classes.txt)={n}")

    # ----- Cohérence task -----
    task_values = {
        "args.yaml.task": args_task,
        "weights/best.json.task": bj_task,
        "config.json.task": cfg_task,
        "training_params.json.model.task": tp_task,
        "model_card.task": mc_task,
    }
    non_null_tasks = {k: v for k, v in task_values.items() if v is not None}
    distinct = set(non_null_tasks.values())
    if len(distinct) > 1:
        report.errors.append(
            "Divergence task entre fichiers : "
            + ", ".join(f"{k}='{v}'" for k, v in non_null_tasks.items())
        )

    # ----- Cohérence resolution_train -----
    res_values = {
        "model_card.resolution_train": mc_res_train,
        "weights/best.json.resolution": bj_resolution,
        "config.json.model.resolution": cfg_resolution,
        "training_params.json.model.imgsz": tp_imgsz,
    }
    non_null_res = {k: v for k, v in res_values.items() if v is not None}
    distinct_res = set(non_null_res.values())
    if len(distinct_res) > 1:
        report.errors.append(
            "Divergence résolution d'entraînement : "
            + ", ".join(f"{k}={v}" for k, v in non_null_res.items())
        )

    # ----- model_card.resolution_inference doit matcher args.yaml.imgsz -----
    if mc_res_inf is not None and args_imgsz is not None and mc_res_inf != args_imgsz:
        report.errors.append(
            f"model_card.resolution_inference={mc_res_inf} != args.yaml.imgsz={args_imgsz}"
        )

    # ----- Divergences imgsz / SAHI : info + vérif documentation -----
    documented_fields = {ic.get("field") for ic in mc_inference_choices if isinstance(ic, dict)}

    if args_imgsz is not None and bj_resolution is not None and args_imgsz != bj_resolution:
        msg = (
            f"args.yaml.imgsz ({args_imgsz}) != weights/best.json.resolution ({bj_resolution})"
        )
        if "imgsz" in documented_fields:
            report.infos.append(msg + " — documenté dans model_card.inference_choices")
        else:
            report.warnings.append(msg + " — non documenté dans model_card.inference_choices")

    sahi_w = args_sahi.get("slice_width") if isinstance(args_sahi, dict) else None
    sahi_h = args_sahi.get("slice_height") if isinstance(args_sahi, dict) else None
    if args_imgsz is not None:
        for axis, value in (("sahi.slice_width", sahi_w), ("sahi.slice_height", sahi_h)):
            if value is not None and value != args_imgsz:
                msg = f"args.yaml.{axis} ({value}) != args.yaml.imgsz ({args_imgsz})"
                if axis in documented_fields or "sahi.slice_width" in documented_fields:
                    report.infos.append(msg + " — documenté dans model_card.inference_choices")
                else:
                    report.warnings.append(msg + " — non documenté dans model_card.inference_choices")

    # ================= Vérifications v2 (audit 2026-08-31) =================

    # ----- inference_choices : les VALEURS doivent correspondre à args.yaml -----
    def _resoudre(chemin: str) -> Any:
        obj: Any = args_yaml
        for part in str(chemin).split("."):
            if not isinstance(obj, dict) or part not in obj:
                return None
            obj = obj[part]
        return obj

    for i, ic in enumerate(mc_inference_choices):
        if not isinstance(ic, dict) or "field" not in ic or "value" not in ic:
            continue
        reel = _resoudre(ic["field"])
        if reel is not None and reel != ic["value"]:
            report.errors.append(
                f"model_card.inference_choices[{i}] : {ic['field']}={ic['value']!r} "
                f"documenté mais args.yaml porte {reel!r} — le model_card MENT "
                "(et l'UI relaie la valeur documentée)"
            )

    # ----- thresholds : bornes + per_class ⊆ classes -----
    mc_thresholds = model_card.get("thresholds") or {}
    conf_default = mc_thresholds.get("confidence_default")
    if conf_default is not None and not (
        isinstance(conf_default, (int, float)) and 0 < float(conf_default) <= 1
    ):
        report.errors.append(
            f"thresholds.confidence_default={conf_default!r} hors (0, 1]"
        )
    min_area = mc_thresholds.get("min_area_m2")
    if min_area is not None and (
        not isinstance(min_area, (int, float)) or float(min_area) < 0
    ):
        report.errors.append(f"thresholds.min_area_m2={min_area!r} invalide (>= 0 requis)")
    pc = mc_thresholds.get("confidence_per_class")
    if pc is not None:
        if not isinstance(pc, dict):
            report.errors.append("thresholds.confidence_per_class doit être un mapping")
        else:
            for cname, val in pc.items():
                if class_set and cname not in class_set:
                    report.errors.append(
                        f"thresholds.confidence_per_class : '{cname}' absent de "
                        "classes.txt (seuil silencieusement ignoré au runtime)"
                    )
                if not (isinstance(val, (int, float)) and 0 < float(val) <= 1):
                    report.errors.append(
                        f"thresholds.confidence_per_class['{cname}']={val!r} hors (0, 1]"
                    )

    # ----- seuils adossés à la mesure canonique -----
    # Règle 2026-09-09 : le seuil de production se choisit SOUS le F1-max, par étude de la
    # courbe (bloc etude_seuil de metriques_eval.json), dans la fenêtre
    # [bas du plateau F1 >= 95 % ; F1-max] ; le choix est justifié dans seuils_provenance.
    # Éval sans etude_seuil (antérieure) : tolérance historique |delta| <= 0,05 au F1-max.
    metriques_path = model_dir / "entrainement" / "evaluation" / "metriques_eval.json"
    metriques_par_classe: dict = {}  # bloc par_classe du modèle évalué (pour la fiabilité)
    if metriques_path.is_file():
        try:
            metriques = json.loads(metriques_path.read_text(encoding="utf-8-sig"))
            provenance = str(mc_thresholds.get("seuils_provenance") or "").strip()
            blocs = [
                (nom, m.get("global") or {}, m.get("par_classe") or {})
                for nom, m in (metriques.get("modeles") or {}).items()
                if isinstance((m.get("global") or {}).get("seuil_f1max"), (int, float))
            ]
            # le modèle évalué = celui dont le nom est l'id du dossier, sinon le premier publié
            bloc = next((b for b in blocs if b[0] == model_dir.name), blocs[0] if blocs else None)
            if bloc is not None:
                _, g, par_classe = bloc
                metriques_par_classe = par_classe
                a_verifier = [("confidence_default", conf_default, g)]
                if isinstance(pc, dict):
                    a_verifier += [
                        (f"confidence_per_class['{c}']", v, par_classe.get(c) or {})
                        for c, v in pc.items()
                    ]
                for champ, val, b in a_verifier:
                    f1max = b.get("seuil_f1max")
                    if not (isinstance(val, (int, float)) and isinstance(f1max, (int, float))):
                        continue
                    etude = b.get("etude_seuil")
                    if not isinstance(etude, dict):
                        ecart = abs(float(val) - float(f1max))
                        if ecart > 0.05 and not provenance:
                            report.errors.append(
                                f"thresholds.{champ}={val} s'écarte de {ecart:.3f} du "
                                "seuil_f1max mesuré (éval sans etude_seuil) sans "
                                "thresholds.seuils_provenance pour le justifier"
                            )
                        continue
                    lo = float(etude["plateau_f1_95"][0])
                    if not provenance:
                        report.errors.append(
                            f"thresholds.{champ}={val} : l'éval porte une étude de seuil "
                            "(etude_seuil) mais thresholds.seuils_provenance ne documente "
                            "pas le choix (plateau, F2-max, FP/image) — requis"
                        )
                    if float(val) < lo - 1e-6 or float(val) > float(f1max) + 1e-6:
                        report.warnings.append(
                            f"thresholds.{champ}={val} hors de la fenêtre mesurée "
                            f"[{lo:.3f} ; {float(f1max):.3f}] (bas du plateau F1 >= 95 % ; "
                            f"F1-max ; proposé {etude.get('seuil_propose')}) — vérifier la "
                            "justification dans seuils_provenance"
                        )
                    elif abs(float(val) - float(f1max)) < 1e-6:
                        report.warnings.append(
                            f"thresholds.{champ}={val} = F1-max, haut de la fenêtre "
                            f"[{lo:.3f} ; {float(f1max):.3f}] : le rappel n'est pas "
                            f"privilégié (proposé {etude.get('seuil_propose')}) — assumer "
                            "dans seuils_provenance"
                        )
        except (ValueError, AttributeError, KeyError, TypeError) as exc:
            report.warnings.append(f"metriques_eval.json illisible : {exc}")
    else:
        report.warnings.append(
            "seuils non adossés à une mesure : entrainement/evaluation/"
            "metriques_eval.json absent (cf. tools/courbes_eval.py du repo training-models)"
        )

    # ----- fiabilité affichée (thresholds.fiabilite, 2026-09-09) -----
    # Catégories douteux/possible/probable/quasi_certain PAR CLASSE, définies par la
    # part de vrais objets mesurée au banc. Contrat : classes ⊆ classes.txt, catégories
    # ordonnées à seuils croissants (app.services.fiabilite.parse_fiabilite), première
    # catégorie AU seuil de la classe (invariant seuil = symbologie = filtrage) ;
    # mesures re-dérivées des bandes de metriques_eval.json quand elles existent.
    fiab_raw = mc_thresholds.get("fiabilite")
    if fiab_raw is not None:
        _validate_fiabilite(fiab_raw, mc_thresholds, class_set, conf_default, pc,
                            metriques_par_classe, report, model_dir=model_dir)

    # ----- entités ⊆ catalogue (hors catalogue = modèle INVISIBLE dans l'UI) -----
    catalog_path = REPO_ROOT / "data" / "entities_catalog.json"
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8-sig"))
        catalog_ids = {
            e.get("id") for e in catalog.get("entities", []) if isinstance(e, dict)
        }
    except (OSError, ValueError):
        catalog_ids = set()
        report.warnings.append("entities_catalog.json illisible — contrôle catalogue sauté")
    mc_derived = model_card.get("derived_targets") or []
    if catalog_ids:
        for c in mc_classes:
            if not isinstance(c, dict):
                continue
            entite = c.get("entity") or c.get("name")
            if entite and entite not in catalog_ids:
                report.errors.append(
                    f"model_card.classes : entité '{entite}' absente du catalogue "
                    "entities_catalog.json — l'entité n'apparaîtra PAS dans l'UI"
                )
        for i, dt in enumerate(mc_derived):
            if isinstance(dt, dict) and dt.get("entity") and dt["entity"] not in catalog_ids:
                report.errors.append(
                    f"model_card.derived_targets[{i}] : entité '{dt['entity']}' "
                    "absente du catalogue entities_catalog.json"
                )

    # ----- derived_targets ↔ clustering -----
    cluster_outputs = {
        str(r.get("output_class_name") or "").strip()
        for r in args_clustering
        if isinstance(r, dict)
    }
    for i, dt in enumerate(mc_derived):
        if not isinstance(dt, dict):
            continue
        out = str(dt.get("output_class") or "").strip()
        if out and out not in cluster_outputs:
            report.errors.append(
                f"model_card.derived_targets[{i}] : output_class '{out}' sans règle "
                "args.yaml.clustering correspondante — le plugin l'ignorera en silence"
            )

    # ----- best.json.source pointe un fichier existant -----
    if isinstance(bj_source, str) and bj_source and not ABS_PATH_RE.match(bj_source):
        if not (model_dir / bj_source).is_file():
            report.warnings.append(
                f"weights/best.json.source='{bj_source}' : fichier ABSENT "
                "(modèle non réexportable/réévaluable)"
            )

    # ----- best.json.confidence_threshold écrase le seuil UI (binaire, seg) -----
    bj_conf = best_json.get("confidence_threshold")
    if (
        isinstance(bj_conf, (int, float))
        and isinstance(conf_default, (int, float))
        and abs(float(bj_conf) - float(conf_default)) > 1e-9
    ):
        report.errors.append(
            f"weights/best.json.confidence_threshold={bj_conf} != "
            f"model_card.thresholds.confidence_default={conf_default} — côté binaire "
            "en segmentation, le sidecar ÉCRASE silencieusement le seuil de l'UI"
        )

    # ----- architecture croisée (indicatif) -----
    mc_arch = str(model_card.get("architecture") or "")
    bj_model_type = str(best_json.get("model_type") or "")
    if mc_arch and bj_model_type:
        attendu = {"rfdetr": "RF-DETR", "yolo": "YOLO", "segformer": "SegFormer", "smp": "SMP"}
        prefixe = attendu.get(bj_model_type.lower())
        if prefixe and not mc_arch.upper().startswith(prefixe.upper()):
            report.warnings.append(
                f"model_card.architecture='{mc_arch}' vs weights/best.json.model_type="
                f"'{bj_model_type}' : familles divergentes"
            )

    # ----- Mode strict : warnings → errors -----
    if strict and report.warnings:
        report.errors.extend(report.warnings)
        report.warnings = []

    return report


def format_report(report: ValidationReport) -> str:
    lines: list[str] = []
    status = report.status()
    lines.append(f"[{status:4}] {report.model_dir}")
    for err in report.errors:
        lines.append(f"  ERR  {err}")
    for warn in report.warnings:
        lines.append(f"  WARN {warn}")
    for info in report.infos:
        lines.append(f"  INFO {info}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        help="Dossiers de modèles à valider. Par défaut : tous sous data/models/.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Traiter les warnings comme des erreurs.",
    )
    args = parser.parse_args(argv)

    if args.paths:
        targets = [Path(p).resolve() for p in args.paths]
    else:
        targets = find_model_dirs(DEFAULT_MODELS_DIR)

    if not targets:
        print(f"Aucun modèle trouvé sous {DEFAULT_MODELS_DIR}", file=sys.stderr)
        return 0

    failures = 0
    for target in targets:
        if not target.is_dir():
            print(f"[SKIP] {target} : n'est pas un dossier", file=sys.stderr)
            continue
        report = validate_model_dir(target, strict=args.strict)
        print(format_report(report))
        if not report.ok:
            failures += 1

    print()
    print(f"Résultat : {len(targets) - failures}/{len(targets)} modèle(s) OK")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
