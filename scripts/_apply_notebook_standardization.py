"""Patch ponctuel du notebook ``data/models/rfdetr_unified_pipeline.ipynb``.

Ajoute :

1. À la fin de la cellule 2 (config globale) : les nouveaux dicts ``INFERENCE``,
   ``MNT``, ``RVT``, ``UI``, ``MODEL_CARD_META`` qui pilotent le packaging
   standardisé.
2. À la fin de la cellule 13 (analyse dataset) : la canonicalisation
   ``CANONICAL_CLASS_NAMES`` (snake_case ASCII, doublons préservés) et
   ``CLASS_LABELS_FR`` (forme originale du COCO).
3. À la toute fin du notebook : une nouvelle section markdown
   "## 16. Packaging" + une cellule code qui produit
   ``runs/training/<RUN_NAME>/package/`` conforme à ``docs/model_contract.md``.

Idempotent : détecte un marqueur ``# === STANDARDIZATION:`` pour ne pas
ré-appliquer le patch deux fois.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NB_PATH = REPO_ROOT / "data" / "models" / "rfdetr_unified_pipeline.ipynb"

CELL_2_PATCH = '''
# === STANDARDIZATION: dicts contractuels (cf. docs/model_contract.md) ===
# Ces structures complémentent les variables ci-dessus pour piloter la
# cellule finale "Packaging" qui produit args.yaml / model_card.yaml /
# training_params.json / classes.txt / config.json conformes au contrat.

INFERENCE = {
    "imgsz": RESOLUTION,                      # peut différer du training si voulu
    "sahi": {
        "slice_width": RESOLUTION,
        "slice_height": RESOLUTION,
        "overlap_ratio": 0.2,
    },
    "postprocess": {
        "merge_adjacent": True,
        "remove_overlaps": True,
    },
    # Liste de règles DBSCAN spatial post-détection. Exemple commenté :
    # "clustering": [
    #     {
    #         "target_classes": ["cratere_obus"],
    #         "min_confidence": 0.4,
    #         "min_confidence_extend": 0.3,
    #         "min_cluster_size": 10,
    #         "min_samples": 5,
    #         "eps_m": 60,
    #         "output_class_name": "zone_crateres",
    #         "output_geometry": "convex_hull",
    #         "buffer_m": 10,
    #         "min_area_m2": 500,
    #         "confidence_weight": 0.0,
    #     },
    # ],
    "clustering": [],
    # Documente les divergences imgsz vs resolution / sahi vs imgsz si
    # l'utilisateur en introduit (cf. docs/model_contract.md §1.b).
    "inference_choices": [],
}

MNT = {
    "resolution": 0.5,
    "filter_expression": (
        "Classification = 2 OR Classification = 6 OR Classification = 66 "
        "OR Classification = 67 OR Classification = 9"
    ),
}

# RVT par défaut LD à 15° / 10-20 m. À adapter si le dataset cible un autre RVT.
RVT = {
    "type": "LD",
    "params": {
        "angular_res": 15,
        "min_radius": 10,
        "max_radius": 20,
        "observer_h": 1.7,
        "ve_factor": 1,
        "save_as_8bit": True,
    },
}

UI = {
    "min_area_m2": 0,
    "confidence_default": CONFIDENCE_THRESHOLD,
}

MODEL_CARD_META = {
    "id": TARGET_NAME,
    "display_name": TARGET_NAME.replace("_", " ").title(),
    "version": "",                        # ex. "2026-05" ; renseigner au packaging
    "status": "beta",                     # production | beta | deprecated | broken
    "description": "",                    # description libre, sera écrite dans model_card.yaml
    "recommended_use": "",
    "known_limitations": [],
}

# Validation early-fail
assert TASK in {"object_detection", "instance_segmentation"}, f"TASK invalide : {TASK!r}"
assert INFERENCE["imgsz"] > 0, "INFERENCE['imgsz'] doit être > 0"
'''

CELL_13_PATCH = '''

# === STANDARDIZATION: canonicalisation snake_case ASCII ===
# CLASS_NAMES garde les libellés originaux du COCO (peuvent être accentués,
# en majuscule, avec espaces). CANONICAL_CLASS_NAMES est la forme runtime
# utilisée par le pipeline (classes.txt, weights/best.json, etc.).
# Les DOUBLONS sont préservés (sous-classes RF-DETR fusionnées post-inférence
# par nom — voir docs/model_contract.md).
import unicodedata as _unicodedata
import re as _re


def _canonical_class_name(name):
    s = _unicodedata.normalize("NFKD", str(name)).encode("ASCII", "ignore").decode()
    s = _re.sub(r"[^a-z0-9_]+", "_", s.lower()).strip("_")
    if not s or not s[0].isalpha():
        raise ValueError(f"Nom de classe non canonicalisable : {name!r}")
    return s


CANONICAL_CLASS_NAMES = [_canonical_class_name(n) for n in CLASS_NAMES]
CLASS_LABELS_FR = list(CLASS_NAMES)

_dups = [n for n in set(CANONICAL_CLASS_NAMES) if CANONICAL_CLASS_NAMES.count(n) > 1]
print(f"📋 Noms canoniques : {CANONICAL_CLASS_NAMES}")
if _dups:
    print(f"   Doublons détectés (= sous-classes RF-DETR fusionnées post-inférence) : {_dups}")
'''

PACKAGING_MARKDOWN = '''## 16. Packaging — produit `package/` conforme au contrat

Génère le sous-dossier `runs/training/<RUN_NAME>/package/` contenant les
fichiers texte standardisés (`model_card.yaml`, `args.yaml`, `classes.txt`,
`training_params.json`, `config.json`, `evaluation_results.json`) plus une
copie du meilleur checkpoint (`weights/best.pth`).

Pour finaliser l'installation dans le plugin :

1. Copier `package/` → `data/models/<RUN_NAME>/`.
2. `python dev/runner_onnx/export_to_onnx.py --model data/models/<RUN_NAME>/weights/best.pth --output data/models/<RUN_NAME>/weights/best.onnx`
3. `python scripts/validate_models_metadata.py data/models/<RUN_NAME>`
'''

PACKAGING_CODE = '''# === STANDARDIZATION: packaging final ===
# Produit `runs/training/<RUN_NAME>/package/` conforme à docs/model_contract.md.
import json as _json
import shutil as _shutil
import yaml as _yaml
from pathlib import Path as _Path


def _class_color_indices(n_classes):
    """Palette runner_onnx (12 couleurs) cyclique par index de classe."""
    return [i % 12 for i in range(n_classes)]


def _build_args_yaml(inference, model_str, task_str, class_colors):
    args = {
        "model": model_str,
        "task": task_str,
        "imgsz": inference["imgsz"],
        "sahi": {
            "slice_width": inference["sahi"]["slice_width"],
            "slice_height": inference["sahi"]["slice_height"],
            "overlap_ratio": inference["sahi"]["overlap_ratio"],
        },
        "postprocess": {
            "merge_adjacent": inference["postprocess"]["merge_adjacent"],
            "remove_overlaps": inference["postprocess"]["remove_overlaps"],
        },
        "class_colors": list(class_colors),
    }
    if inference.get("clustering"):
        args["clustering"] = inference["clustering"]
    return args


def _build_model_card(meta, run, model_str, variant, classes_canon, labels_fr, color_indices,
                     inference, mnt, rvt, ui, training_resolution):
    classes_block = []
    for idx, (canon, fr, color) in enumerate(zip(classes_canon, labels_fr, color_indices)):
        classes_block.append({
            "id": idx,
            "name": canon,
            "label_fr": str(fr),
            "color_index": color,
            "description": "",
        })

    inference_choices = list(inference.get("inference_choices") or [])
    # Auto-détecte les divergences si non documentées
    documented = {ic.get("field") for ic in inference_choices if isinstance(ic, dict)}
    if inference["imgsz"] != training_resolution and "imgsz" not in documented:
        inference_choices.append({
            "field": "imgsz",
            "value": inference["imgsz"],
            "reason": f"imgsz inférence ({inference['imgsz']}) != résolution training ({training_resolution}) — à compléter.",
        })
    sahi_w = inference["sahi"]["slice_width"]
    if sahi_w != inference["imgsz"] and "sahi.slice_width" not in documented:
        inference_choices.append({
            "field": "sahi.slice_width",
            "value": sahi_w,
            "reason": f"SAHI slice_width ({sahi_w}) != imgsz ({inference['imgsz']}) — à compléter.",
        })
    sahi_h = inference["sahi"]["slice_height"]
    if sahi_h != inference["imgsz"] and "sahi.slice_height" not in documented:
        inference_choices.append({
            "field": "sahi.slice_height",
            "value": sahi_h,
            "reason": f"SAHI slice_height ({sahi_h}) != imgsz ({inference['imgsz']}) — à compléter.",
        })

    return {
        "id": meta["id"],
        "display_name": meta["display_name"] or meta["id"],
        "version": meta.get("version") or "",
        "status": meta.get("status") or "beta",
        "description": meta.get("description") or "",
        "task": run["task"],
        "architecture": model_str,
        "variant": variant,
        "resolution_train": training_resolution,
        "resolution_inference": inference["imgsz"],
        "preferred_rvt": {
            "type": rvt["type"],
            "params": dict(rvt["params"]),
        },
        "mnt": {
            "resolution": mnt["resolution"],
            "filter_expression": mnt["filter_expression"],
        },
        "classes": classes_block,
        "thresholds": {
            "confidence_default": ui["confidence_default"],
            "min_area_m2": ui["min_area_m2"],
        },
        "inference_choices": inference_choices,
        "recommended_use": meta.get("recommended_use") or "",
        "known_limitations": list(meta.get("known_limitations") or []),
    }


def _build_training_params(run, mnt, rvt, ui, training_resolution, variant):
    return {
        "description": "Paramètres utilisés pour générer les images d'entraînement de ce modèle",
        "model": {
            "architecture": run.get("architecture", "RF-DETR"),
            "variant": variant,
            "task": run["task"],
            "imgsz": training_resolution,
        },
        "mnt": {
            "resolution": mnt["resolution"],
            "filter_expression": mnt["filter_expression"],
        },
        "rvt": {
            "type": rvt["type"],
            "params": dict(rvt["params"]),
        },
        "detection": {"min_area_m2": ui["min_area_m2"]},
    }


def _build_config_json(run, training_resolution, classes_canon, training_dict, dataset_dict):
    return {
        "task": run["task"],
        "model": {
            "architecture": run.get("architecture", "RF-DETR"),
            "variant": run["variant"],
            "resolution": training_resolution,
            "num_classes": len(classes_canon),
            "class_names": list(classes_canon),
        },
        "training": dict(training_dict),
        "dataset": dict(dataset_dict),
        "inference": {"confidence_threshold": run.get("confidence_threshold", 0.3)},
    }


# ---- Construction des structures à partir des variables du notebook ----
_model_str_map = {
    ("object_detection", "base"): "RF-DETR",
    ("object_detection", "large"): "RF-DETR-Large",
    ("instance_segmentation", "nano"): "RF-DETR-Seg-Nano",
    ("instance_segmentation", "small"): "RF-DETR-Seg-Small",
    ("instance_segmentation", "medium"): "RF-DETR-Seg-Medium",
    ("instance_segmentation", "large"): "RF-DETR-Seg-Large",
}
_model_str = _model_str_map.get((TASK, MODEL_VARIANT), f"RF-DETR-{MODEL_VARIANT}")

_run = {
    "id": TARGET_NAME,
    "task": TASK,
    "variant": MODEL_VARIANT,
    "architecture": _model_str.split("-")[0] + "-DETR" if "DETR" in _model_str else "RF-DETR",
    "confidence_threshold": CONFIDENCE_THRESHOLD,
}

_class_colors = _class_color_indices(len(CANONICAL_CLASS_NAMES))

_args_yaml = _build_args_yaml(INFERENCE, _model_str, TASK, _class_colors)

_training_dict = {
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "grad_accum_steps": GRAD_ACCUM_STEPS,
    "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
    "learning_rate": LEARNING_RATE,
    "lr_encoder": LR_ENCODER,
    "weight_decay": WEIGHT_DECAY,
    "lr_scheduler": LR_SCHEDULER,
    "warmup_epochs": WARMUP_EPOCHS,
    "lr_drop": LR_DROP,
    "freeze_at": FREEZE_AT,
    "use_amp": USE_AMP,
    "use_ema": USE_EMA,
    "gradient_checkpointing": GRADIENT_CHECKPOINTING,
    "early_stopping": EARLY_STOPPING,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
    "early_stopping_use_ema": EARLY_STOPPING_USE_EMA,
    "checkpoint_interval": CHECKPOINT_INTERVAL,
    "seed": SEED,
}

_dataset_dict = {
    "roboflow_workspace": ROBOFLOW_WORKSPACE,
    "roboflow_project": ROBOFLOW_PROJECT,
    "roboflow_version": ROBOFLOW_VERSION,
    "roboflow_url": ROBOFLOW_URL,
    "ignore_prefixes": list(IGNORE_PREFIXES),
    "ignore_classes": list(IGNORE_CLASSES),
}

_training_resolution = RESOLUTION

_model_card = _build_model_card(
    MODEL_CARD_META, _run, _model_str, MODEL_VARIANT,
    CANONICAL_CLASS_NAMES, CLASS_LABELS_FR, _class_colors,
    INFERENCE, MNT, RVT, UI, _training_resolution,
)

_training_params = _build_training_params(_run, MNT, RVT, UI, _training_resolution, MODEL_VARIANT)
_config_json = _build_config_json(_run, _training_resolution, CANONICAL_CLASS_NAMES, _training_dict, _dataset_dict)

# ---- Écriture du package ----
_package_dir = _Path(TARGET_TRAINING_PATH) / "package"
(_package_dir / "weights").mkdir(parents=True, exist_ok=True)

with open(_package_dir / "model_card.yaml", "w", encoding="utf-8") as f:
    _yaml.safe_dump(_model_card, f, sort_keys=False, allow_unicode=True)

with open(_package_dir / "args.yaml", "w", encoding="utf-8") as f:
    _yaml.safe_dump(_args_yaml, f, sort_keys=False, allow_unicode=True)

(_package_dir / "classes.txt").write_text(
    "\\n".join(CANONICAL_CLASS_NAMES) + "\\n", encoding="utf-8"
)

with open(_package_dir / "training_params.json", "w", encoding="utf-8") as f:
    _json.dump(_training_params, f, indent=2, ensure_ascii=False)
    f.write("\\n")

with open(_package_dir / "config.json", "w", encoding="utf-8") as f:
    _json.dump(_config_json, f, indent=2, ensure_ascii=False)
    f.write("\\n")

# Copier evaluation_results.json si déjà produit
_eval_src = _Path(TARGET_TRAINING_PATH) / "evaluation_results.json"
if _eval_src.exists():
    _shutil.copy2(_eval_src, _package_dir / "evaluation_results.json")
else:
    print("⚠️  evaluation_results.json absent — sera à régénérer avant publication.")

# Copier le meilleur checkpoint
_ckpt_candidates = [
    _Path(TARGET_TRAINING_PATH) / "checkpoints" / "checkpoint_best_ema.pth",
    _Path(TARGET_TRAINING_PATH) / "output" / "checkpoint_best_ema.pth",
    _Path(TARGET_TRAINING_PATH) / "checkpoints" / "checkpoint_best_total.pth",
    _Path(TARGET_TRAINING_PATH) / "output" / "checkpoint_best_total.pth",
]
_ckpt_found = next((c for c in _ckpt_candidates if c.exists()), None)
if _ckpt_found is not None:
    _shutil.copy2(_ckpt_found, _package_dir / "weights" / "best.pth")
    print(f"✅ Checkpoint copié : {_ckpt_found.name} → {_package_dir / 'weights' / 'best.pth'}")
else:
    print("⚠️  Aucun checkpoint best.pth trouvé — à copier manuellement.")

print()
print(f"📦 Package prêt : {_package_dir}")
print("   Pour installer dans le plugin :")
print(f"     1. Copier {_package_dir.name}/ → data/models/{TARGET_NAME}/")
print(f"     2. python dev/runner_onnx/export_to_onnx.py "
      f"--model data/models/{TARGET_NAME}/weights/best.pth "
      f"--output data/models/{TARGET_NAME}/weights/best.onnx")
print(f"     3. python scripts/validate_models_metadata.py data/models/{TARGET_NAME}")
'''


def main() -> int:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    cells = nb["cells"]

    # Détection idempotence
    raw_dump = json.dumps(cells)
    if "# === STANDARDIZATION:" in raw_dump:
        print("Notebook déjà patché (marqueur STANDARDIZATION trouvé). Aucune modification.")
        return 0

    # Patch cellule 2 (config globale) : append à la fin du source
    src_2 = "".join(cells[2]["source"])
    cells[2]["source"] = (src_2.rstrip() + "\n" + CELL_2_PATCH).splitlines(keepends=True)

    # Patch cellule 13 (analyse dataset)
    src_13 = "".join(cells[13]["source"])
    cells[13]["source"] = (src_13.rstrip() + "\n" + CELL_13_PATCH).splitlines(keepends=True)

    # Nouvelles cellules à la fin
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": PACKAGING_MARKDOWN.splitlines(keepends=True),
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": PACKAGING_CODE.splitlines(keepends=True),
    })

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Notebook patché : {NB_PATH}")
    print(f"  Cellule 2 augmentée (+{len(CELL_2_PATCH.splitlines())} lignes)")
    print(f"  Cellule 13 augmentée (+{len(CELL_13_PATCH.splitlines())} lignes)")
    print("  +2 cellules en fin de notebook (markdown + code packaging)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
