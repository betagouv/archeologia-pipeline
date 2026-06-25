"""PARSE-02/03/04/05/07 (audit v1, persistants) : validation tolérante des
paramètres venus des YAML modèles et des configs UI/partagées.

- PARSE-02 : eps_m/min_samples… castés sans bornes → DBSCAN silencieusement
  vide (eps ≤ 0) ou crash scipy.
- PARSE-03 : clustering_overrides fusionnés par cc.update(ov) sans validation
  (clés inconnues, types, plages).
- PARSE-04 : float() non gardé sur les seuils d'un run brut → une valeur vide
  casse TOUTE la phase CV (et désormais la finalisation, en finally).
- PARSE-05 : slug d'entité (repli id) inséré dans un Path sans re-slugification.
- PARSE-07 : un seul float() invalide jetait TOUTES les règles de clustering.
"""
from __future__ import annotations

import pytest

pytest.importorskip("shapely")

from pipeline.cv.clustering_bounds import (
    sanitize_clustering_overrides,
    sanitize_clustering_rule,
)
from pipeline.cv.model_config import (
    load_clustering_config_from_model,
    resolve_cv_runs,
)
from pipeline.cv.model_profile import _parse_clustering
from pipeline.output_paths import build_entity_class_targets


class TestSanitizeRule:
    def test_eps_negatif_clampe_strictement_positif(self):
        rule = sanitize_clustering_rule({"eps_m": -10, "min_samples": 0})
        assert rule["eps_m"] > 0
        assert rule["min_samples"] >= 1

    def test_confiance_clampee_dans_0_1(self):
        rule = sanitize_clustering_rule({"min_confidence": 7})
        assert 0.0 <= rule["min_confidence"] <= 1.0


class TestSanitizeOverrides:
    def test_cles_inconnues_ignorees_et_valeurs_castees(self):
        ov = sanitize_clustering_overrides(
            {"eps_m": "25", "min_samples": "4", "__proto__": "x", "target_classes": ["hack"]}
        )
        assert ov == {"eps_m": 25.0, "min_samples": 4}

    def test_valeur_non_castable_ignoree(self):
        ov = sanitize_clustering_overrides({"eps_m": "abc", "buffer_m": 5})
        assert "eps_m" not in ov
        assert ov["buffer_m"] == 5.0

    def test_geometrie_hors_liste_blanche_ignoree(self):
        assert sanitize_clustering_overrides({"output_geometry": "exotic"}) == {}
        assert sanitize_clustering_overrides({"output_geometry": "concave_hull"}) == {
            "output_geometry": "concave_hull"
        }


class TestParse07RegleIsolee:
    def test_une_regle_invalide_ne_jette_pas_les_autres(self, tmp_path):
        (tmp_path / "args.yaml").write_text(
            "clustering:\n"
            "  - target_classes: [cratere]\n"
            "    min_confidence: abc\n"          # règle 1 : invalide
            "  - target_classes: [charbonniere]\n"
            "    eps_m: 30\n",                    # règle 2 : valide
            encoding="utf-8",
        )
        configs = load_clustering_config_from_model(tmp_path)
        assert configs is not None and len(configs) == 1
        assert configs[0]["target_classes"] == ["charbonniere"]

    def test_eps_negatif_clampe_au_chargement(self, tmp_path):
        (tmp_path / "args.yaml").write_text(
            "clustering:\n  - target_classes: [cratere]\n    eps_m: -10\n",
            encoding="utf-8",
        )
        configs = load_clustering_config_from_model(tmp_path)
        assert configs is not None and configs[0]["eps_m"] > 0

    def test_model_profile_clampe_aussi(self):
        rules = _parse_clustering(
            {"clustering": [{"target_classes": ["c"], "min_samples": 0, "eps_m": -5}]}
        )
        assert rules and rules[0].min_samples >= 1 and rules[0].eps_m > 0


class TestParse04SeuilsDeRun:
    def test_seuil_non_castable_ne_casse_pas_la_phase_cv(self):
        cv_config = {
            "enabled": True,
            "runs": [
                {"model": "modele_a", "confidence_threshold": ""},
                {"model": "modele_b", "iou_threshold": None, "min_area_m2": "x"},
            ],
        }
        runs = resolve_cv_runs(cv_config)  # ne doit PAS lever
        assert len(runs) == 2


class TestParse05SlugDeRepli:
    def test_id_de_repli_re_slugifie_avant_le_path(self, tmp_path):
        targets = build_entity_class_targets(
            tmp_path, [{"id": "Zone / éphémère:1", "classes": ["c"]}]
        )
        (gpkg, _layer) = targets["c"][0]
        # Aucun séparateur/char interdit Windows dans le segment d'entité.
        assert "/" not in gpkg.replace(str(tmp_path), "")
        for ch in ':*?"<>|':
            assert ch not in gpkg.replace(str(tmp_path), "")
