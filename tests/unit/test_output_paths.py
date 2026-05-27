"""Tests pour les résolutions de chemins entité-centrées de ``pipeline.output_paths``.

``detections/`` est réorganisé par ENTITÉ (vocabulaire utilisateur, ex.
``detections/parcellaire/``) ; l'échafaudage technique (raw/annotated) descend
sous ``detections/_technique/<model_slug>/``. Le module reste importable en
standalone (import ``rvt_naming`` différé) → testable hors QGIS.
"""
from __future__ import annotations

from pathlib import Path

from pipeline.output_paths import (
    build_entity_class_targets,
    detection_entity_dir,
    detection_technique_annotated_dir,
    detection_technique_dir,
    detection_technique_raw_dir,
    detections_dir,
)

OUT = Path("/out")


def _gpkg(slug: str) -> str:
    return str(OUT / "detections" / slug / f"{slug}.gpkg")


class TestEntityPaths:
    def test_entity_dir_under_detections(self):
        assert detection_entity_dir(OUT, "parcellaire") == OUT / "detections" / "parcellaire"

    def test_technique_dir_isolated_under_detections(self):
        d = detection_technique_dir(OUT, "cratere_circulaire_2")
        assert d == OUT / "detections" / "_technique" / "cratere_circulaire_2"

    def test_technique_raw_and_annotated_nested(self):
        m = "formes_lineaires_x"
        assert detection_technique_raw_dir(OUT, m) == (
            OUT / "detections" / "_technique" / m / "raw_detections"
        )
        assert detection_technique_annotated_dir(OUT, m) == (
            OUT / "detections" / "_technique" / m / "annotated_images"
        )

    def test_technique_lives_inside_detections_root(self):
        # _technique est un sous-dossier de detections/ (pas un sibling)
        assert detections_dir(OUT) in detection_technique_dir(OUT, "m").parents


def _ent(eid, slug, classes, *, layer_names=None):
    return {"id": eid, "slug": slug, "classes": list(classes),
            "layer_names": dict(layer_names or {})}


class TestEntityClassTargets:
    """Routage classe → [(GeoPackage, nom_de_couche)], piloté par ``layer_names``.

    Cas critique (décision C) : « Cratères » (base) et « Regroupement de cratères »
    (dérivée : cratere + zone_crateres) résolvent au même run. La classe source
    cratere n'est PAS dupliquée : elle n'apparaît qu'une fois, comme constituant
    renommé du groupe (la couche « Cratères » à plat est supprimée).
    """

    def test_shared_source_deduped_into_derived_group(self):
        entities = [
            _ent("cratere", "crateres", ["cratere"]),
            _ent("regroupement_crateres", "regroupement_de_crateres",
                 ["cratere", "zone_crateres"],
                 layer_names={"cratere": "Cratères",
                              "zone_crateres": "Regroupements"}),
        ]
        t = build_entity_class_targets(OUT, entities)
        # cratere n'apparaît QUE dans le groupe (couche renommée), pas à plat
        assert t["cratere"] == [
            (_gpkg("regroupement_de_crateres"), "Cratères"),
        ]
        assert t["zone_crateres"] == [(_gpkg("regroupement_de_crateres"), "Regroupements")]

    def test_dedup_independent_of_entity_order(self):
        # même si la dérivée est listée d'abord, la dédup garde la couche du groupe
        entities = [
            _ent("regroupement_crateres", "regroupement_de_crateres",
                 ["cratere", "zone_crateres"],
                 layer_names={"cratere": "Cratères"}),
            _ent("cratere", "crateres", ["cratere"]),
        ]
        t = build_entity_class_targets(OUT, entities)
        assert t["cratere"] == [(_gpkg("regroupement_de_crateres"), "Cratères")]

    def test_base_entity_alone_keeps_canonical(self):
        entities = [_ent("cratere", "crateres", ["cratere"])]
        t = build_entity_class_targets(OUT, entities)
        assert t["cratere"] == [(_gpkg("crateres"), "cratere")]

    def test_derived_alone_keeps_source_layer(self):
        # entité dérivée seule : la source garde son libellé (pas de canonique à plat)
        entities = [
            _ent("regroupement_crateres", "regroupement_de_crateres",
                 ["cratere", "zone_crateres"],
                 layer_names={"cratere": "cratere_source"}),
        ]
        t = build_entity_class_targets(OUT, entities)
        assert t["cratere"] == [(_gpkg("regroupement_de_crateres"), "cratere_source")]
        # cluster non renommé (absent de layer_names) → garde son nom de classe
        assert t["zone_crateres"] == [(_gpkg("regroupement_de_crateres"), "zone_crateres")]

    def test_independent_entities_each_own_layer(self):
        entities = [
            _ent("parcellaire", "parcellaire", ["parcellaire"]),
            _ent("chemin_creux", "chemins_creux", ["chemin_creux"]),
        ]
        t = build_entity_class_targets(OUT, entities)
        assert t["parcellaire"] == [(_gpkg("parcellaire"), "parcellaire")]
        assert t["chemin_creux"] == [(_gpkg("chemins_creux"), "chemin_creux")]

    def test_no_entities_returns_empty(self):
        assert build_entity_class_targets(OUT, []) == {}
        assert build_entity_class_targets(OUT, None) == {}
