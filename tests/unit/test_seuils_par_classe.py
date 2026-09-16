"""Seuils de confiance PAR CLASSE : contrats de `resolve_runs_from_entities` (pur Python).

Rapatrié de training-models/tests/test_seuils_par_classe_plugin.py le 2026-09-15
(doublure qui vivait dans l'autre dépôt ; la vérité est ici). Trois contrats :

  1. les défauts `thresholds.confidence_per_class` du model_card arrivent dans le
     run, filtrés aux classes du run ;
  2. une surcharge d'entité (UI) ne touche QUE les classes de cette entité — c'est
     la correction du vieux `min()` qui imposait le seuil le plus bas à tout le run ;
  3. le scalaire `confidence_threshold` émis reste le PLANCHER effectif (min des
     seuils applicables), et sans per-class ni surcharge il vaut le défaut du
     modèle (rétro-compatibilité stricte).
"""
from app.services.model_orchestrator import (
    EntityDef,
    InstalledModel,
    _extract_thresholds,
    resolve_runs_from_entities,
)


def modele(**kw):
    base = dict(
        name="lineaires_seg_v2_1",
        display_name="Linéaires",
        weights_path=None,
        target_rvt="LD",
        status="beta",
        coverage={
            "parcellaire": ("parcellaire",),
            "talus_fosse": ("talus_fosse",),
            "chemin_creux": ("chemin_creux",),
        },
        class_names=("parcellaire", "talus_fosse", "chemin_creux"),
        default_confidence=0.25,
    )
    base.update(kw)
    return InstalledModel(**base)


CATALOG = [
    EntityDef(id="parcellaire", label="Parcellaire"),
    EntityDef(id="talus_fosse", label="Talus/fossé"),
    EntityDef(id="chemin_creux", label="Chemin creux"),
]


def runs_pour(m, entites, seuils=None):
    return resolve_runs_from_entities(
        selected_entity_ids=entites, overrides=None, installed_models=[m],
        catalog=CATALOG, entity_thresholds=seuils)


def test_defauts_model_card():
    m = modele(default_confidence_per_class={"chemin_creux": 0.15, "talus_fosse": 0.30})
    (run,) = runs_pour(m, ["parcellaire", "talus_fosse", "chemin_creux"])
    assert run["confidence_per_class"] == {"chemin_creux": 0.15, "talus_fosse": 0.30}, run
    # plancher = min(0.15, 0.30, défaut 0.25 pour parcellaire)
    assert abs(run["confidence_threshold"] - 0.15) < 1e-9, run


def test_defauts_filtres_aux_classes_du_run():
    m = modele(default_confidence_per_class={"chemin_creux": 0.15})
    (run,) = runs_pour(m, ["parcellaire"])          # chemin_creux PAS dans le run
    assert run["confidence_per_class"] == {}, run
    assert abs(run["confidence_threshold"] - 0.25) < 1e-9, run


def test_surcharge_entite_ne_touche_que_ses_classes():
    m = modele()
    (run,) = runs_pour(m, ["parcellaire", "chemin_creux"],
                       seuils={"chemin_creux": {"confidence_threshold": 0.4}})
    # AVANT : min() global -> 0.4 s'appliquait aussi au parcellaire. APRÈS :
    assert run["confidence_per_class"] == {"chemin_creux": 0.4}, run
    # le plancher reste le défaut : parcellaire décodé à 0.25, pas 0.4
    assert abs(run["confidence_threshold"] - 0.25) < 1e-9, run


def test_surcharge_prime_sur_le_defaut_de_classe():
    m = modele(default_confidence_per_class={"chemin_creux": 0.15})
    (run,) = runs_pour(m, ["chemin_creux"],
                       seuils={"chemin_creux": {"confidence_threshold": 0.35}})
    assert run["confidence_per_class"] == {"chemin_creux": 0.35}, run
    assert abs(run["confidence_threshold"] - 0.35) < 1e-9, run


def test_retrocompat_sans_per_class():
    m = modele()
    (run,) = runs_pour(m, ["parcellaire", "chemin_creux"])
    assert run["confidence_per_class"] == {}, run
    assert abs(run["confidence_threshold"] - 0.25) < 1e-9, run


def test_extract_thresholds_tolerant():
    conf, pc, area, iou = _extract_thresholds({"thresholds": {
        "confidence_default": 0.25,
        "confidence_per_class": {"chemin_creux": 0.15, "cassee": "abc"},
        "min_area_m2": 200}})
    assert abs(conf - 0.25) < 1e-9 and abs(area - 200) < 1e-9
    assert pc == {"chemin_creux": 0.15}, pc      # l'entrée pourrie est ignorée, pas fatale


def test_extract_thresholds_absent():
    conf, pc, area, iou = _extract_thresholds({})
    # 0.3 = défaut UNIFIÉ de la chaîne CV (audit 2026-08-31, ex-0.2)
    assert pc == {} and abs(conf - 0.3) < 1e-9
