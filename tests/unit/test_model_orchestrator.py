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
  - target_classes: ["cratere_obus"]
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
  - {id: 0, name: cratere_obus, label_fr: "Cratère d'obus"}
"""

VERDUN = """
display_name: "Verdun multi-classes"
status: production
preferred_rvt:
  type: SVF
classes:
  - {name: abri}
  - {name: cratere_obus}
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
  - {name: cratere_circulaire, entity: cratere_obus}
"""


def _catalog() -> list:
    return [
        EntityDef(id="cratere_obus", label="Trous d'obus", display_order=10),
        EntityDef(id="abri", label="Abris", display_order=20),
        EntityDef(id="tranchees_et_boyaux", label="Tranchées", display_order=30),
        EntityDef(id="chemin_creux", label="Chemins creux", display_order=40),
        EntityDef(id="parcellaire", label="Parcellaire", display_order=50),
        EntityDef(id="talus_fosse", label="Talus/fossés", display_order=60),
        EntityDef(id="charbonniere", label="Charbonnières", display_order=70),
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
        assert m.class_names == ("cratere_obus",)
        assert m.coverage == {"cratere_obus": ("cratere_obus",)}
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
        # la classe 'cratere_circulaire' couvre l'entité 'cratere_obus'
        assert m.coverage == {"cratere_obus": ("cratere_circulaire",)}
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
        assert m.cluster_options == {"cratere_obus": ("zone_crateres",)}

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
        crat = cov["cratere_obus"]
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
        assert cov["cratere_obus"].default_model == "prod_model"


# ----------------------------------------------------------------------
# resolve_runs_from_entities
# ----------------------------------------------------------------------
class TestResolveRuns:
    def test_single_entity_yields_explicit_class(self, tmp_path):
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE)
        installed = discover_installed_models(tmp_path)
        runs = resolve_runs_from_entities(["cratere_obus"], {}, installed, _catalog())
        # selected_classes est toujours explicite (cluster désactivé par défaut)
        assert runs == [
            {"model": "cratere_circulaire_2", "target_rvt": "LD", "selected_classes": ["cratere_obus"]}
        ]

    def test_subset_selection_yields_explicit_classes(self, tmp_path):
        _write_model(tmp_path, "verdun_3_classes_1", VERDUN)
        installed = discover_installed_models(tmp_path)
        runs = resolve_runs_from_entities(["cratere_obus"], {}, installed, _catalog())
        # verdun a 3 classes : sélection partielle -> liste explicite
        assert runs == [
            {"model": "verdun_3_classes_1", "target_rvt": "SVF", "selected_classes": ["cratere_obus"]}
        ]

    def test_two_entities_same_model_grouped_into_one_run(self, tmp_path):
        _write_model(tmp_path, "formes", FORMES)
        installed = discover_installed_models(tmp_path)
        runs = resolve_runs_from_entities(["parcellaire", "talus_fosse"], {}, installed, _catalog())
        assert runs == [
            {"model": "formes", "target_rvt": "LD",
             "selected_classes": ["parcellaire", "talus_fosse"]}
        ]

    def test_all_classes_of_multiclass_model_explicit(self, tmp_path):
        _write_model(tmp_path, "formes", FORMES)
        installed = discover_installed_models(tmp_path)
        runs = resolve_runs_from_entities(
            ["chemin_creux", "parcellaire", "talus_fosse"], {}, installed, _catalog()
        )
        assert runs == [{
            "model": "formes", "target_rvt": "LD",
            "selected_classes": ["chemin_creux", "parcellaire", "talus_fosse"],
        }]

    def test_different_models_yield_separate_sorted_runs(self, tmp_path):
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE)  # LD
        _write_model(tmp_path, "formes", FORMES)                 # LD
        installed = discover_installed_models(tmp_path)
        runs = resolve_runs_from_entities(["parcellaire", "cratere_obus"], {}, installed, _catalog())
        assert runs == [
            {"model": "cratere_circulaire_2", "target_rvt": "LD", "selected_classes": ["cratere_obus"]},
            {"model": "formes", "target_rvt": "LD", "selected_classes": ["parcellaire"]},
        ]

    def test_override_redirects_to_other_model(self, tmp_path):
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE)  # défaut cratère
        _write_model(tmp_path, "verdun_3_classes_1", VERDUN)
        installed = discover_installed_models(tmp_path)
        runs = resolve_runs_from_entities(
            ["cratere_obus"], {"cratere_obus": "verdun_3_classes_1"}, installed, _catalog()
        )
        assert runs == [
            {"model": "verdun_3_classes_1", "target_rvt": "SVF", "selected_classes": ["cratere_obus"]}
        ]

    def test_entity_without_model_skipped(self, tmp_path):
        _write_model(tmp_path, "formes", FORMES)
        installed = discover_installed_models(tmp_path)
        # 'charbonniere' non couverte -> ignorée, 'parcellaire' produit un run
        runs = resolve_runs_from_entities(["charbonniere", "parcellaire"], {}, installed, _catalog())
        assert runs == [
            {"model": "formes", "target_rvt": "LD", "selected_classes": ["parcellaire"]}
        ]

    def test_no_selection_yields_no_runs(self, tmp_path):
        _write_model(tmp_path, "formes", FORMES)
        installed = discover_installed_models(tmp_path)
        assert resolve_runs_from_entities([], {}, installed, _catalog()) == []

    def test_cluster_enabled_adds_cluster_output(self, tmp_path):
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE, args_yaml=CRATERE_ARGS)
        installed = discover_installed_models(tmp_path)
        runs = resolve_runs_from_entities(
            ["cratere_obus"], {}, installed, _catalog(), cluster_enabled={"cratere_obus"}
        )
        assert runs == [{
            "model": "cratere_circulaire_2", "target_rvt": "LD",
            "selected_classes": ["cratere_obus", "zone_crateres"],
        }]

    def test_cluster_disabled_excludes_cluster_output(self, tmp_path):
        _write_model(tmp_path, "cratere_circulaire_2", CRATERE, args_yaml=CRATERE_ARGS)
        installed = discover_installed_models(tmp_path)
        runs = resolve_runs_from_entities(["cratere_obus"], {}, installed, _catalog())
        assert runs == [{
            "model": "cratere_circulaire_2", "target_rvt": "LD",
            "selected_classes": ["cratere_obus"],
        }]


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
