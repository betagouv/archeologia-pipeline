"""Tests pour l'orchestrateur de modèles (entités → modèles → runs).

L'orchestrateur est PUR (src/app/services) : il lit le ``model_card.yaml`` de
chaque modèle installé (``preferred_rvt`` + ``classes``) et résout, à partir
des entités sélectionnées par l'utilisateur, la liste des ``runs`` CV au schéma
attendu par le pipeline. Il ne doit jamais importer ``pipeline.cv`` (→ shapely).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from app.services.model_orchestrator import (
    EntityDef,
    build_entity_coverage,
    discover_installed_models,
    group_entities_by_morphology,
    load_entities_catalog,
    resolve_runs_from_entities,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _write_model(
    models_dir: Path,
    name: str,
    model_card_yaml: str | None,
    *,
    args_yaml: str | None = None,
    weights: bool = True,
) -> Path:
    """Crée un dossier modèle avec model_card.yaml (+ optionnel args.yaml, poids)."""
    d = models_dir / name
    (d / "weights").mkdir(parents=True)
    if weights:
        (d / "weights" / "best.onnx").write_bytes(b"")
    if model_card_yaml is not None:
        (d / "model_card.yaml").write_text(model_card_yaml, encoding="utf-8")
    if args_yaml is not None:
        (d / "args.yaml").write_text(args_yaml, encoding="utf-8")
    return d


# args.yaml minimal avec une règle de clustering (cratère -> zone_crateres)
CRATERE_ARGS = """
clustering:
  - target_classes: ["cratere"]
    output_class_name: "zone_crateres"
    min_confidence: 0.4
"""


# model_cards synthétiques (proches des vrais)
CRATERE = """
display_name: "Cratères circulaires"
status: production
preferred_rvt:
  type: LD
classes:
  - {id: 0, name: cratere, label_fr: "Cratère d'obus"}
"""

VERDUN = """
display_name: "Verdun multi-classes"
status: production
preferred_rvt:
  type: SVF
classes:
  - {name: abri}
  - {name: cratere}
  - {name: tranchees_et_boyaux}
"""

FORMES = """
display_name: "Formes linéaires"
status: production
preferred_rvt:
  type: LD
classes:
  - {name: chemin_creux}
  - {name: parcellaire}
  - {name: talus_fosse}
"""

CHARBON = """
display_name: "Charbonnières / fours"
status: production
preferred_rvt:
  type: LD
classes:
  - {name: charbonniere}
  - {name: charbonniere}
  - {name: charbonniere}
  - {name: circular_depression}
  - {name: four}
"""

ALIAS = """
display_name: "Cratères (renommé)"
status: production
preferred_rvt:
  type: LD
classes:
  - {name: cratere_circulaire, entity: cratere}
"""

THRESH = """
display_name: "Modèle à seuils"
status: production
preferred_rvt:
  type: LD
classes:
  - {name: parcellaire}
thresholds:
  confidence_default: 0.45
  min_area_m2: 200
  iou: 0.6
"""


# Cratère + cible dérivée : la sortie de clustering 'zone_crateres' est présentée
# comme l'entité 'regroupement_crateres' (zones + dépressions individuelles).
CRATERE_DERIVED = """
display_name: "Cratères circulaires"
status: production
preferred_rvt:
  type: LD
classes:
  - {id: 0, name: cratere, label_fr: "Cratère d'obus"}
derived_targets:
  - output_class: zone_crateres
    entity: regroupement_crateres
    include_source: true
"""

# Cratère + cible dérivée AVEC libellés de couche configurés (cluster + source).
CRATERE_DERIVED_LABELED = """
display_name: "Cratères circulaires"
status: production
preferred_rvt:
  type: LD
classes:
  - {id: 0, name: cratere, label_fr: "Cratère d'obus"}
derived_targets:
  - output_class: zone_crateres
    entity: regroupement_crateres
    include_source: true
    output_label: zones_extraction
    source_label: crateres_constitutifs
"""

# Idem mais zones SEULEMENT (sans les dépressions individuelles).
CRATERE_DERIVED_ZONES_ONLY = """
display_name: "Cratères circulaires"
status: production
preferred_rvt:
  type: LD
classes:
  - {id: 0, name: cratere}
derived_targets:
  - output_class: zone_crateres
    entity: regroupement_crateres
    include_source: false
"""

# Cible dérivée pointant vers une sortie de clustering inexistante (aucune règle
# args.yaml correspondante) → doit être ignorée sans casser.
CRATERE_DERIVED_DANGLING = """
display_name: "Cratères circulaires"
status: production
preferred_rvt:
  type: LD
classes:
  - {id: 0, name: cratere}
derived_targets:
  - output_class: zone_inexistante
    entity: regroupement_crateres
    include_source: true
"""

# Verdun (SVF) avec la même cible dérivée → second modèle candidat.
VERDUN_DERIVED = VERDUN + """derived_targets:
  - output_class: zone_crateres
    entity: regroupement_crateres
    include_source: true
"""


def _summ(runs):
    """Réduit les runs à (model, target_rvt, selected_classes) — compare la
    structure sans dépendre des seuils injectés (testés à part)."""
    return [(r["model"], r["target_rvt"], r["selected_classes"]) for r in runs]


def _catalog() -> list:
    return [
        EntityDef(id="cratere", label="Cratères", display_order=10),
        EntityDef(id="abri", label="Abris", display_order=20),
        EntityDef(id="tranchees_et_boyaux", label="Tranchées", display_order=30),
        EntityDef(id="chemin_creux", label="Chemins creux", display_order=40),
        EntityDef(id="parcellaire", label="Parcellaire", display_order=50),
        EntityDef(id="talus_fosse", label="Talus/fossés", display_order=60),
        EntityDef(id="charbonniere", label="Charbonnières", display_order=70),
        EntityDef(id="regroupement_crateres", label="Regroupement de cratères", display_order=95),
    ]


# ----------------------------------------------------------------------
# load_entities_catalog
# ----------------------------------------------------------------------
class TestLoadEntitiesCatalog:
    def test_loads_and_sorts_by_display_order(self, tmp_path):
        p = tmp_path / "cat.json"
        p.write_text(json.dumps({
            "schema_version": 1,
            "entities": [
                {"id": "b", "label": "B", "display_order": 20},
                {"id": "a", "label": "A", "display_order": 10, "description": "desc A"},
            ],
        }), encoding="utf-8")
        cat = load_entities_catalog(p)
        assert [e.id for e in cat] == ["a", "b"]
        assert cat[0].label == "A"
        assert cat[0].description == "desc A"

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_entities_catalog(tmp_path / "nope.json") == []

    def test_entry_without_id_or_label_skipped(self, tmp_path):
        p = tmp_path / "cat.json"
        p.write_text(json.dumps({"entities": [
            {"id": "ok", "label": "OK"},
            {"id": "no_label"},
            {"label": "no id"},
        ]}), encoding="utf-8")
        cat = load_entities_catalog(p)
        assert [e.id for e in cat] == ["ok"]

    def test_duplicate_id_first_wins(self, tmp_path):
        p = tmp_path / "cat.json"
        p.write_text(json.dumps({"entities": [
            {"id": "x", "label": "First"},
            {"id": "x", "label": "Second"},
        ]}), encoding="utf-8")
        cat = load_entities_catalog(p)
        assert len(cat) == 1
        assert cat[0].label == "First"


# ----------------------------------------------------------------------
# discover_installed_models
# ----------------------------------------------------------------------
class TestDiscoverInstalledModels:
    def test_reads_model_card_fields(self, tmp_path):
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE)
        models = discover_installed_models(tmp_path)
        assert len(models) == 1
        m = models[0]
        assert m.name == "cratere_circulaire_2"
        assert m.display_name == "Cratères circulaires"
        assert m.target_rvt == "LD"
        assert m.status == "production"
        assert m.class_names == ("cratere",)
        assert m.coverage == {"cratere": ("cratere",)}
        assert m.weights_path is not None and m.weights_path.name == "best.onnx"

    def test_target_rvt_default_when_absent(self, tmp_path):
        _write_model(tmp_path, "m", "display_name: X\nclasses:\n  - {name: foo}\n")
        models = discover_installed_models(tmp_path)
        assert models[0].target_rvt == "LD"

    def test_target_rvt_uppercased(self, tmp_path):
        _write_model(tmp_path, "m", "preferred_rvt:\n  type: svf\nclasses:\n  - {name: foo}\n")
        assert discover_installed_models(tmp_path)[0].target_rvt == "SVF"

    def test_repeated_class_names_deduped(self, tmp_path):
        _write_model(tmp_path, "run_rf_detr_1", CHARBON)
        m = discover_installed_models(tmp_path)[0]
        assert m.class_names == ("charbonniere", "circular_depression", "four")
        assert m.coverage["charbonniere"] == ("charbonniere",)

    def test_entity_alias_maps_class_to_catalog_entity(self, tmp_path):
        _write_model(tmp_path, "cratere_circulaire_2", ALIAS)
        m = discover_installed_models(tmp_path)[0]
        # la classe 'cratere_circulaire' couvre l'entité 'cratere'
        assert m.coverage == {"cratere": ("cratere_circulaire",)}
        assert m.class_names == ("cratere_circulaire",)

    def test_missing_model_card_skipped(self, tmp_path):
        _write_model(tmp_path, "no_card", None)
        assert discover_installed_models(tmp_path) == []

    def test_non_directory_entries_ignored(self, tmp_path):
        (tmp_path / "stray.txt").write_text("x", encoding="utf-8")
        _write_model(tmp_path, "m", CRATERE)
        assert len(discover_installed_models(tmp_path)) == 1

    def test_models_dir_absent_returns_empty(self, tmp_path):
        assert discover_installed_models(tmp_path / "nope") == []

    def test_cluster_options_read_from_args_yaml(self, tmp_path):
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE, args_yaml=CRATERE_ARGS)
        m = discover_installed_models(tmp_path)[0]
        assert m.cluster_options == {"cratere": ("zone_crateres",)}

    def test_no_clustering_means_empty_cluster_options(self, tmp_path):
        _write_model(tmp_path, "formes", FORMES)  # pas d'args.yaml
        m = discover_installed_models(tmp_path)[0]
        assert m.cluster_options == {}


# ----------------------------------------------------------------------
# build_entity_coverage
# ----------------------------------------------------------------------
class TestBuildEntityCoverage:
    def test_candidates_and_specialized_default(self, tmp_path):
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE)  # 1 classe
        _write_model(tmp_path, "verdun_3_classes_1", VERDUN)     # 3 classes
        installed = discover_installed_models(tmp_path)
        cov = {ec.entity.id: ec for ec in build_entity_coverage(_catalog(), installed)}
        crat = cov["cratere"]
        assert set(crat.candidate_models) == {"cratere_circulaire_2", "verdun_3_classes_1"}
        # le plus spécialisé (moins de classes) est le défaut
        assert crat.default_model == "cratere_circulaire_2"

    def test_entity_without_model_has_no_default(self, tmp_path):
        _write_model(tmp_path, "formes", FORMES)
        installed = discover_installed_models(tmp_path)
        cov = {ec.entity.id: ec for ec in build_entity_coverage(_catalog(), installed)}
        # 'charbonniere' n'est couverte par aucun modèle installé
        assert cov["charbonniere"].candidate_models == ()
        assert cov["charbonniere"].default_model is None

    def test_production_preferred_over_other_status(self, tmp_path):
        _write_model(tmp_path, "exp_model", CRATERE.replace("status: production", "status: experimental"))
        _write_model(tmp_path, "prod_model", VERDUN)  # production, 3 classes
        installed = discover_installed_models(tmp_path)
        cov = {ec.entity.id: ec for ec in build_entity_coverage(_catalog(), installed)}
        # production prime sur le nombre de classes
        assert cov["cratere"].default_model == "prod_model"


# ----------------------------------------------------------------------
# resolve_runs_from_entities
# ----------------------------------------------------------------------
class TestResolveRuns:
    def test_single_entity_yields_explicit_class(self, tmp_path):
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE)
        installed = discover_installed_models(tmp_path)
        runs = resolve_runs_from_entities(["cratere"], {}, installed, _catalog())
        assert _summ(runs) == [("cratere_circulaire_2", "LD", ["cratere"])]

    def test_subset_selection_yields_explicit_classes(self, tmp_path):
        _write_model(tmp_path, "verdun_3_classes_1", VERDUN)
        installed = discover_installed_models(tmp_path)
        runs = resolve_runs_from_entities(["cratere"], {}, installed, _catalog())
        assert _summ(runs) == [("verdun_3_classes_1", "SVF", ["cratere"])]

    def test_two_entities_same_model_grouped_into_one_run(self, tmp_path):
        _write_model(tmp_path, "formes", FORMES)
        installed = discover_installed_models(tmp_path)
        runs = resolve_runs_from_entities(["parcellaire", "talus_fosse"], {}, installed, _catalog())
        assert _summ(runs) == [("formes", "LD", ["parcellaire", "talus_fosse"])]

    def test_all_classes_of_multiclass_model_explicit(self, tmp_path):
        _write_model(tmp_path, "formes", FORMES)
        installed = discover_installed_models(tmp_path)
        runs = resolve_runs_from_entities(
            ["chemin_creux", "parcellaire", "talus_fosse"], {}, installed, _catalog()
        )
        assert _summ(runs) == [("formes", "LD", ["chemin_creux", "parcellaire", "talus_fosse"])]

    def test_different_models_yield_separate_sorted_runs(self, tmp_path):
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE)  # LD
        _write_model(tmp_path, "formes", FORMES)                 # LD
        installed = discover_installed_models(tmp_path)
        runs = resolve_runs_from_entities(["parcellaire", "cratere"], {}, installed, _catalog())
        assert _summ(runs) == [
            ("cratere_circulaire_2", "LD", ["cratere"]),
            ("formes", "LD", ["parcellaire"]),
        ]

    def test_override_redirects_to_other_model(self, tmp_path):
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE)  # défaut cratère
        _write_model(tmp_path, "verdun_3_classes_1", VERDUN)
        installed = discover_installed_models(tmp_path)
        runs = resolve_runs_from_entities(
            ["cratere"], {"cratere": "verdun_3_classes_1"}, installed, _catalog()
        )
        assert _summ(runs) == [("verdun_3_classes_1", "SVF", ["cratere"])]

    def test_stale_override_falls_back_to_default_model(self, tmp_path):
        # Surcharge périmée (model_card modifié entre deux sessions) : le modèle
        # surchargé est installé mais ne couvre plus l'entité → retour au défaut,
        # l'entité ne doit PAS disparaître silencieusement du run.
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE)  # ne couvre pas parcellaire
        _write_model(tmp_path, "formes", FORMES)                 # défaut parcellaire
        installed = discover_installed_models(tmp_path)
        runs = resolve_runs_from_entities(
            ["parcellaire"], {"parcellaire": "cratere_circulaire_2"}, installed, _catalog()
        )
        assert _summ(runs) == [("formes", "LD", ["parcellaire"])]

    def test_stale_override_to_uninstalled_model_falls_back(self, tmp_path):
        # Surcharge vers un modèle désinstallé → même auto-réparation.
        _write_model(tmp_path, "formes", FORMES)
        installed = discover_installed_models(tmp_path)
        runs = resolve_runs_from_entities(
            ["parcellaire"], {"parcellaire": "modele_disparu"}, installed, _catalog()
        )
        assert _summ(runs) == [("formes", "LD", ["parcellaire"])]

    def test_entity_without_model_skipped(self, tmp_path):
        _write_model(tmp_path, "formes", FORMES)
        installed = discover_installed_models(tmp_path)
        # 'charbonniere' non couverte -> ignorée, 'parcellaire' produit un run
        runs = resolve_runs_from_entities(["charbonniere", "parcellaire"], {}, installed, _catalog())
        assert _summ(runs) == [("formes", "LD", ["parcellaire"])]

    def test_no_selection_yields_no_runs(self, tmp_path):
        _write_model(tmp_path, "formes", FORMES)
        installed = discover_installed_models(tmp_path)
        assert resolve_runs_from_entities([], {}, installed, _catalog()) == []

    def test_cluster_enabled_adds_cluster_output(self, tmp_path):
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE, args_yaml=CRATERE_ARGS)
        installed = discover_installed_models(tmp_path)
        runs = resolve_runs_from_entities(
            ["cratere"], {}, installed, _catalog(), cluster_enabled={"cratere"}
        )
        assert _summ(runs) == [("cratere_circulaire_2", "LD", ["cratere", "zone_crateres"])]

    def test_cluster_disabled_excludes_cluster_output(self, tmp_path):
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE, args_yaml=CRATERE_ARGS)
        installed = discover_installed_models(tmp_path)
        runs = resolve_runs_from_entities(["cratere"], {}, installed, _catalog())
        assert _summ(runs) == [("cratere_circulaire_2", "LD", ["cratere"])]


# ----------------------------------------------------------------------
# Cibles dérivées : une sortie de clustering présentée comme une entité
# ----------------------------------------------------------------------
class TestDerivedTargets:
    def test_include_source_covers_source_and_output_classes(self, tmp_path):
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE_DERIVED, args_yaml=CRATERE_ARGS)
        m = discover_installed_models(tmp_path)[0]
        assert m.coverage["regroupement_crateres"] == ("cratere", "zone_crateres")
        assert m.derived_entities == {"regroupement_crateres"}

    def test_zones_only_covers_output_class_alone(self, tmp_path):
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE_DERIVED_ZONES_ONLY, args_yaml=CRATERE_ARGS)
        m = discover_installed_models(tmp_path)[0]
        assert m.coverage["regroupement_crateres"] == ("zone_crateres",)

    def test_derived_output_removed_from_source_cluster_options(self, tmp_path):
        # Une sortie de clustering exposée comme entité dérivée ne doit PAS aussi
        # proposer une case « regrouper en clusters » sur l'entité source : la
        # seule voie de regroupement est l'entité dérivée « Regroupement de cratères ».
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE_DERIVED, args_yaml=CRATERE_ARGS)
        m = discover_installed_models(tmp_path)[0]
        assert m.cluster_options == {}
        assert "regroupement_crateres" not in m.cluster_options

    def test_dangling_derived_target_ignored(self, tmp_path):
        # output_class sans règle de clustering correspondante → ignoré, pas d'exception.
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE_DERIVED_DANGLING, args_yaml=CRATERE_ARGS)
        m = discover_installed_models(tmp_path)[0]
        assert "regroupement_crateres" not in m.coverage
        assert m.derived_entities == frozenset()

    def test_coverage_lists_both_models_default_specialized(self, tmp_path):
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE_DERIVED, args_yaml=CRATERE_ARGS)
        _write_model(tmp_path, "verdun_3_classes_1", VERDUN_DERIVED, args_yaml=CRATERE_ARGS)
        installed = discover_installed_models(tmp_path)
        cov = {ec.entity.id: ec for ec in build_entity_coverage(_catalog(), installed)}
        ze = cov["regroupement_crateres"]
        assert set(ze.candidate_models) == {"cratere_circulaire_2", "verdun_3_classes_1"}
        assert ze.default_model == "cratere_circulaire_2"

    def test_resolve_run_for_derived_entity(self, tmp_path):
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE_DERIVED, args_yaml=CRATERE_ARGS)
        installed = discover_installed_models(tmp_path)
        runs = resolve_runs_from_entities(["regroupement_crateres"], {}, installed, _catalog())
        assert _summ(runs) == [("cratere_circulaire_2", "LD", ["cratere", "zone_crateres"])]

    def test_crater_and_extraction_merge_into_one_run(self, tmp_path):
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE_DERIVED, args_yaml=CRATERE_ARGS)
        installed = discover_installed_models(tmp_path)
        runs = resolve_runs_from_entities(
            ["cratere", "regroupement_crateres"], {}, installed, _catalog()
        )
        assert _summ(runs) == [("cratere_circulaire_2", "LD", ["cratere", "zone_crateres"])]


class TestThresholds:
    def test_discover_reads_default_thresholds(self, tmp_path):
        _write_model(tmp_path, "m", THRESH)
        m = discover_installed_models(tmp_path)[0]
        assert m.default_confidence == 0.45
        assert m.default_min_area == 200.0
        assert m.default_iou == 0.6

    def test_discover_threshold_defaults_when_absent(self, tmp_path):
        _write_model(tmp_path, "formes", FORMES)
        m = discover_installed_models(tmp_path)[0]
        assert m.default_confidence == 0.3  # défaut UNIFIÉ 2026-08-31 (model_card sans seuil)
        assert m.default_min_area == 0.0
        assert m.default_iou == 0.5

    def test_run_carries_model_default_thresholds(self, tmp_path):
        _write_model(tmp_path, "m", THRESH)
        installed = discover_installed_models(tmp_path)
        run = resolve_runs_from_entities(["parcellaire"], {}, installed, _catalog())[0]
        assert run["confidence_threshold"] == 0.45
        assert run["iou_threshold"] == 0.6
        assert run["min_area_m2"] == 200.0

    def test_entity_override_takes_precedence_except_iou(self, tmp_path):
        _write_model(tmp_path, "m", THRESH)
        installed = discover_installed_models(tmp_path)
        run = resolve_runs_from_entities(
            ["parcellaire"], {}, installed, _catalog(),
            entity_thresholds={"parcellaire": {"confidence_threshold": 0.7, "min_area_m2": 50}},
        )[0]
        assert run["confidence_threshold"] == 0.7
        assert run["min_area_m2"] == 50.0
        assert run["iou_threshold"] == 0.6  # IoU jamais surchargé par l'UI


# ----------------------------------------------------------------------
# Découpage par entité dans chaque run (clé "entities")
# ----------------------------------------------------------------------
# Le run porte, en plus de selected_classes (plat, rétro-compat), un détail
# par entité {id, label, slug, classes, is_derived} permettant à l'étape de
# sortie d'organiser detections/<slug>/ par entité (vocabulaire utilisateur).
class TestRunEntities:
    def test_single_entity_breakdown(self, tmp_path):
        _write_model(tmp_path, "formes", FORMES)
        installed = discover_installed_models(tmp_path)
        run = resolve_runs_from_entities(["parcellaire"], {}, installed, _catalog())[0]
        assert run["entities"] == [
            {"id": "parcellaire", "label": "Parcellaire", "slug": "parcellaire",
             "classes": ["parcellaire"], "is_derived": False, "layer_names": {}},
        ]

    def test_n_entities_one_model_listed_separately_sorted(self, tmp_path):
        _write_model(tmp_path, "formes", FORMES)
        installed = discover_installed_models(tmp_path)
        run = resolve_runs_from_entities(
            ["talus_fosse", "chemin_creux", "parcellaire"], {}, installed, _catalog()
        )[0]
        assert [e["id"] for e in run["entities"]] == ["chemin_creux", "parcellaire", "talus_fosse"]
        assert all(e["is_derived"] is False for e in run["entities"])
        assert {e["id"]: e["classes"] for e in run["entities"]} == {
            "chemin_creux": ["chemin_creux"],
            "parcellaire": ["parcellaire"],
            "talus_fosse": ["talus_fosse"],
        }
        # slug dérive du libellé FR (≠ id) : "Chemins creux" -> chemins_creux
        slugs = {e["id"]: e["slug"] for e in run["entities"]}
        assert slugs["chemin_creux"] == "chemins_creux"
        assert slugs["talus_fosse"] == "talus_fosses"

    def test_derived_entity_marked_and_carries_output_and_source_classes(self, tmp_path):
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE_DERIVED, args_yaml=CRATERE_ARGS)
        installed = discover_installed_models(tmp_path)
        run = resolve_runs_from_entities(["regroupement_crateres"], {}, installed, _catalog())[0]
        ent = run["entities"]
        assert len(ent) == 1
        assert ent[0]["id"] == "regroupement_crateres"
        assert ent[0]["is_derived"] is True
        assert ent[0]["classes"] == ["cratere", "zone_crateres"]
        assert ent[0]["slug"] == "regroupement_de_crateres"

    def test_cluster_enabled_class_attributed_to_entity(self, tmp_path):
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE, args_yaml=CRATERE_ARGS)
        installed = discover_installed_models(tmp_path)
        run = resolve_runs_from_entities(
            ["cratere"], {}, installed, _catalog(), cluster_enabled={"cratere"}
        )[0]
        ent = run["entities"]
        assert len(ent) == 1
        assert ent[0]["id"] == "cratere"
        assert ent[0]["is_derived"] is False
        # la sortie de clustering activée est rattachée à l'entité
        assert ent[0]["classes"] == ["cratere", "zone_crateres"]

    def test_derived_entity_layer_names_with_configured_labels(self, tmp_path):
        # output_label renomme la couche cluster, source_label la couche source
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE_DERIVED_LABELED, args_yaml=CRATERE_ARGS)
        installed = discover_installed_models(tmp_path)
        run = resolve_runs_from_entities(["regroupement_crateres"], {}, installed, _catalog())[0]
        ent = run["entities"][0]
        assert ent["id"] == "regroupement_crateres"
        assert ent["layer_names"] == {
            "zone_crateres": "zones_extraction",        # cluster (output_label)
            "cratere": "crateres_constitutifs",    # source (source_label)
        }

    def test_derived_entity_layer_names_defaults(self, tmp_path):
        # sans libellés : cluster gardé tel quel, source suffixée _source
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE_DERIVED, args_yaml=CRATERE_ARGS)
        installed = discover_installed_models(tmp_path)
        run = resolve_runs_from_entities(["regroupement_crateres"], {}, installed, _catalog())[0]
        ent = run["entities"][0]
        assert ent["layer_names"] == {"cratere": "cratere_source"}

    def test_non_derived_entity_no_layer_names(self, tmp_path):
        _write_model(tmp_path, "formes", FORMES)
        installed = discover_installed_models(tmp_path)
        run = resolve_runs_from_entities(["parcellaire"], {}, installed, _catalog())[0]
        ent = run["entities"][0]
        assert ent["layer_names"] == {}


# ----------------------------------------------------------------------
# Morphologie : champ d'entité + regroupement présentable
# ----------------------------------------------------------------------
class TestMorphology:
    def test_morphology_loaded_from_catalog(self, tmp_path):
        p = tmp_path / "cat.json"
        p.write_text(json.dumps({"schema_version": 2, "entities": [
            {"id": "a", "label": "A", "morphology": "circulaire"},
            {"id": "b", "label": "B", "morphology": "lineaire"},
        ]}), encoding="utf-8")
        cat = load_entities_catalog(p)
        assert {e.id: e.morphology for e in cat} == {"a": "circulaire", "b": "lineaire"}

    def test_missing_morphology_defaults_to_autre(self, tmp_path):
        p = tmp_path / "cat.json"
        p.write_text(json.dumps({"entities": [{"id": "a", "label": "A"}]}), encoding="utf-8")
        assert load_entities_catalog(p)[0].morphology == "autre"

    def test_invalid_morphology_falls_back_to_autre(self, tmp_path):
        p = tmp_path / "cat.json"
        p.write_text(json.dumps({"entities": [
            {"id": "a", "label": "A", "morphology": "banane"},
        ]}), encoding="utf-8")
        assert load_entities_catalog(p)[0].morphology == "autre"

    def test_group_by_morphology_canonical_order_nonempty_only(self):
        cat = [
            EntityDef(id="z", label="Z", display_order=95, morphology="zone"),
            EntityDef(id="c1", label="C1", display_order=10, morphology="circulaire"),
            EntityDef(id="l1", label="L1", display_order=20, morphology="lineaire"),
            EntityDef(id="c2", label="C2", display_order=30, morphology="circulaire"),
        ]
        groups = group_entities_by_morphology(cat)
        # ordre canonique circulaire, lineaire, zone ; 'autre' absente (vide)
        assert [key for key, _l, _g, _e in groups] == ["circulaire", "lineaire", "zone"]
        circ = next(ents for key, _l, _g, ents in groups if key == "circulaire")
        # triées par display_order
        assert [e.id for e in circ] == ["c1", "c2"]

    def test_group_skips_empty_sections(self):
        cat = [EntityDef(id="l", label="L", morphology="lineaire")]
        groups = group_entities_by_morphology(cat)
        assert [key for key, *_ in groups] == ["lineaire"]

    def test_real_catalog_morphology_mapping(self):
        catalog_path = Path(__file__).resolve().parents[2] / "data" / "entities_catalog.json"
        cat = load_entities_catalog(catalog_path)
        ids = {e.id for e in cat}
        assert "cratere" in ids and "regroupement_crateres" in ids
        assert "cratere_obus" not in ids and "zones_extraction_materiaux" not in ids
        by_morph: dict = {}
        for e in cat:
            by_morph.setdefault(e.morphology, []).append(e.id)
        assert len(by_morph.get("circulaire", [])) == 7
        assert len(by_morph.get("lineaire", [])) == 6
        assert by_morph.get("zone", []) == ["regroupement_crateres", "axe_lineaire"]


# args.yaml avec les paramètres DBSCAN complets (défauts exposables dans l'UI)
CRATERE_ARGS_FULL = """
clustering:
  - target_classes: ["cratere"]
    output_class_name: "zone_crateres"
    min_confidence: 0.4
    min_cluster_size: 40
    min_samples: 5
    eps_m: 40
    min_area_m2: 1000
    buffer_m: 10
"""


class TestClusterDefaults:
    def test_cluster_defaults_read_from_args(self, tmp_path):
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE, args_yaml=CRATERE_ARGS_FULL)
        m = discover_installed_models(tmp_path)[0]
        assert m.cluster_defaults == {
            "zone_crateres": {
                "eps_m": 40.0,
                "min_cluster_size": 40,
                "min_samples": 5,
                "min_confidence": 0.4,
                "min_area_m2": 1000.0,
                "buffer_m": 10.0,
            }
        }

    def test_no_clustering_means_empty_cluster_defaults(self, tmp_path):
        _write_model(tmp_path, "formes", FORMES)
        assert discover_installed_models(tmp_path)[0].cluster_defaults == {}


class TestClusterParamOverrides:
    def test_derived_entity_cluster_params_injected_as_overrides(self, tmp_path):
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE_DERIVED, args_yaml=CRATERE_ARGS_FULL)
        installed = discover_installed_models(tmp_path)
        run = resolve_runs_from_entities(
            ["regroupement_crateres"], {}, installed, _catalog(),
            entity_cluster_params={"regroupement_crateres": {"eps_m": 99.0, "min_cluster_size": 12}},
        )[0]
        # mappé vers l'output_class du clustering de cette entité dérivée
        assert run["clustering_overrides"] == {"zone_crateres": {"eps_m": 99.0, "min_cluster_size": 12}}

    def test_no_cluster_params_means_empty_overrides(self, tmp_path):
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE_DERIVED, args_yaml=CRATERE_ARGS_FULL)
        installed = discover_installed_models(tmp_path)
        run = resolve_runs_from_entities(["regroupement_crateres"], {}, installed, _catalog())[0]
        assert run.get("clustering_overrides", {}) == {}


# ----------------------------------------------------------------------
# Isolation des imports : l'orchestrateur ne doit pas tirer shapely / pipeline.cv
# ----------------------------------------------------------------------
class TestImportIsolation:
    def test_import_does_not_pull_shapely_or_pipeline_cv(self):
        src = str(Path(__file__).resolve().parents[2] / "src")
        code = (
            "import sys\n"
            "import app.services.model_orchestrator\n"
            "assert 'shapely' not in sys.modules, 'shapely importé !'\n"
            "bad = [m for m in sys.modules if m == 'pipeline.cv' or m.startswith('pipeline.cv.')]\n"
            "assert not bad, f'pipeline.cv importé : {bad}'\n"
            "print('OK')\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env
        )
        assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
        assert "OK" in result.stdout


# ----------------------------------------------------------------------
# Brique enclosure : entité dérivée « enclos » + défauts exposables UI
# ----------------------------------------------------------------------
FORMES_ENCLOS_ARGS = """
clustering:
  - type: enclosure
    target_classes: ["parcellaire", "talus_fosse"]
    output_class_name: "enclos"
    gap_tolerance_m: 10
    min_area_m2: 50
    max_area_m2: 60000
    min_closure: 0.6
    max_elongation: 3
    max_isolement: 0.3
    min_rectangularite: 0.5
"""

FORMES_ENCLOS = FORMES + """derived_targets:
  - output_class: enclos
    entity: enclos
    include_source: true
    output_label: Enclos
    source_label: Linéaments sources
"""


class TestEnclosureEntity:
    def _installed(self, tmp_path):
        _write_model(tmp_path, "formes", FORMES_ENCLOS, args_yaml=FORMES_ENCLOS_ARGS)
        return discover_installed_models(tmp_path)

    def _cat(self):
        return _catalog() + [
            EntityDef(id="enclos", label="Enclos", display_order=97, morphology="zone")
        ]

    def test_enclosure_defaults_exposed_for_ui(self, tmp_path):
        m = self._installed(tmp_path)[0]
        assert m.cluster_defaults["enclos"] == {
            "gap_tolerance_m": 10.0,
            "min_area_m2": 50.0,
            "max_area_m2": 60000.0,
            "min_closure": 0.6,
            "max_elongation": 3.0,
            "max_isolement": 0.3,
            "min_rectangularite": 0.5,
        }

    def test_enclos_entity_is_derived_with_sources(self, tmp_path):
        m = self._installed(tmp_path)[0]
        assert "enclos" in m.derived_entities
        assert m.coverage["enclos"] == ("enclos", "parcellaire", "talus_fosse")

    def test_resolve_run_selects_output_and_overrides(self, tmp_path):
        installed = self._installed(tmp_path)
        runs = resolve_runs_from_entities(
            ["enclos"], {}, installed, self._cat(),
            entity_cluster_params={"enclos": {"gap_tolerance_m": 12.0}},
        )
        assert len(runs) == 1
        run = runs[0]
        assert "enclos" in run["selected_classes"]
        assert "parcellaire" in run["selected_classes"]
        assert run["clustering_overrides"] == {"enclos": {"gap_tolerance_m": 12.0}}
        ent = next(e for e in run["entities"] if e["id"] == "enclos")
        assert ent["is_derived"] is True


# ----------------------------------------------------------------------
# Brique alignment : entité dérivée « axe_lineaire » + défauts exposables UI
# ----------------------------------------------------------------------
FORMES_AXE_ARGS = """
clustering:
  - type: enclosure
    target_classes: ["parcellaire", "talus_fosse"]
    output_class_name: "enclos"
    gap_tolerance_m: 10
    min_area_m2: 50
  - type: alignment
    target_classes: ["parcellaire"]
    output_class_name: "axe_lineaire"
    band_width_m: 40
    angle_tolerance_deg: 20
    min_length_m: 500
    max_gap_m: 200
    min_coverage: 0.25
    min_sources: 5
"""

FORMES_AXE = FORMES + """derived_targets:
  - output_class: axe_lineaire
    entity: axe_lineaire
    include_source: true
    output_label: "Axes linéaires"
    source_label: "Fragments sources"
"""


class TestAlignmentEntity:
    def _installed(self, tmp_path):
        _write_model(tmp_path, "formes", FORMES_AXE, args_yaml=FORMES_AXE_ARGS)
        return discover_installed_models(tmp_path)

    def _cat(self):
        return _catalog() + [
            EntityDef(id="axe_lineaire", label="Axes linéaires",
                      display_order=98, morphology="zone")
        ]

    def test_alignment_defaults_exposed_for_ui(self, tmp_path):
        m = self._installed(tmp_path)[0]
        assert m.cluster_defaults["axe_lineaire"] == {
            "band_width_m": 40.0,
            "angle_tolerance_deg": 20.0,
            "min_length_m": 500.0,
            "max_gap_m": 200.0,
            "min_coverage": 0.25,
            "min_sources": 5,
        }

    def test_axe_entity_derived_with_sources(self, tmp_path):
        m = self._installed(tmp_path)[0]
        assert "axe_lineaire" in m.derived_entities
        assert m.coverage["axe_lineaire"] == ("axe_lineaire", "parcellaire")

    def test_resolve_run_with_overrides(self, tmp_path):
        installed = self._installed(tmp_path)
        runs = resolve_runs_from_entities(
            ["axe_lineaire"], {}, installed, self._cat(),
            entity_cluster_params={"axe_lineaire": {"band_width_m": 60.0,
                                                    "min_sources": 8}},
        )
        assert len(runs) == 1
        run = runs[0]
        assert "axe_lineaire" in run["selected_classes"]
        assert run["clustering_overrides"] == {
            "axe_lineaire": {"band_width_m": 60.0, "min_sources": 8}}
