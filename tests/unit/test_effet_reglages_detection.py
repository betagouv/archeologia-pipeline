"""Effet des réglages d'entité de l'étape 3 : ce que l'orchestrateur écrit dans ``computer_vision.runs`` et ce que
le regroupement en fait. Un réglage d'entité ne doit agir que sur cette entité, et faire ce que dit son infobulle.

Défauts connus et non corrigés : ``xfail(strict=True)``.
"""
from __future__ import annotations

import pytest

from app.services.model_orchestrator import EntityDef, InstalledModel, resolve_runs_from_entities

CATALOG = [EntityDef(id="parcellaire", label="Parcellaire"), EntityDef(id="talus_fosse", label="Talus/fossé"),
           EntityDef(id="chemin_creux", label="Chemin creux")]


def _modele(**kw):
    base = dict(name="lineaires", display_name="Linéaires", weights_path=None, target_rvt="LD", status="beta",
                coverage={"parcellaire": ("parcellaire",), "talus_fosse": ("talus_fosse",),
                          "chemin_creux": ("chemin_creux",)},
                class_names=("parcellaire", "talus_fosse", "chemin_creux"), default_confidence=0.25)
    base.update(kw)
    return InstalledModel(**base)


def _run(seuils):
    (run,) = resolve_runs_from_entities(
        selected_entity_ids=["parcellaire", "talus_fosse", "chemin_creux"], overrides=None,
        installed_models=[_modele()], catalog=CATALOG, entity_thresholds=seuils)
    return run


def test_seuil_de_confiance_d_une_entite_ne_touche_qu_elle():
    run = _run({"parcellaire": {"confidence_threshold": 0.6}})
    assert run["confidence_per_class"]["parcellaire"] == 0.6
    assert "talus_fosse" not in run["confidence_per_class"] or run["confidence_per_class"]["talus_fosse"] != 0.6


def test_aire_min_d_une_entite_arrive_dans_le_run():
    assert _run({"parcellaire": {"min_area_m2": 500.0}})["min_area_m2"] == 500.0


@pytest.mark.xfail(strict=True, reason="DÉFAUT : l'aire minimale est portée par le RUN (min des surcharges), puis "
                                       "appliquée à tous ses GeoPackages (runner_shapefiles) : régler l'aire de "
                                       "« parcellaire » efface aussi les petits « chemin creux » et « talus/fossé »")
def test_aire_min_d_une_entite_ne_touche_qu_elle():
    run = _run({"parcellaire": {"min_area_m2": 500.0}})
    par_classe = run.get("min_area_per_class") or {}
    assert par_classe.get("parcellaire") == 500.0 and run["min_area_m2"] == 0.0


# ───────────────────────── regroupement (DBSCAN à hystérésis) ─────────────────────────

shapely = pytest.importorskip("shapely")
pytest.importorskip("scipy")
from shapely.geometry import box  # noqa: E402

from pipeline.cv.clustering import run_clustering  # noqa: E402
from pipeline.cv.clustering_bounds import sanitize_clustering_overrides  # noqa: E402


def _regle_du_modele():
    """La règle des cratères telle que args.yaml la déclare : cœur 0,40, extension 0,29."""
    return {"type": "dbscan", "target_classes": ["cratere"], "min_confidence": 0.40, "min_confidence_extend": 0.29,
            "min_cluster_size": 3, "min_samples": 2, "eps_m": 30.0, "output_class_name": "zone_crateres",
            "output_geometry": "convex_hull", "buffer_m": 5.0, "min_area_m2": 0.0}


def _cratères(confs):
    return {"cratere": [{"geometry": box(i * 5.0, 0.0, i * 5.0 + 2.0, 2.0), "confidence": c,
                         "model_pred": "cratere", "model_name": "m"} for i, c in enumerate(confs)]}


def test_confiance_min_du_regroupement_abaissee_agit():
    regle = _regle_du_modele()
    regle.update(sanitize_clustering_overrides({"min_confidence": 0.30}, rule_type="dbscan"))
    zones, _ = run_clustering(_cratères([0.31, 0.32, 0.33, 0.34]), [regle])
    assert zones["zone_crateres"][0]["nb_detect"] == 4


@pytest.mark.xfail(strict=True, reason="DÉFAUT : « Confiance min » (infobulle : ignore les détections sous ce seuil "
                                       "pour le regroupement) ne relève que le seuil cœur ; min_confidence_extend du "
                                       "modèle (0,29) reste, et les détections de 0,29 au nouveau seuil s'agrègent")
def test_confiance_min_du_regroupement_relevee_ignore_les_detections_dessous():
    regle = _regle_du_modele()
    regle.update(sanitize_clustering_overrides({"min_confidence": 0.60}, rule_type="dbscan"))
    zones, _ = run_clustering(_cratères([0.9, 0.9, 0.9, 0.5, 0.5, 0.5]), [regle])
    assert zones["zone_crateres"][0]["nb_detect"] == 3
