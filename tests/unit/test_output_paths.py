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

    Cas critique : « Trous d'obus » (cratere_obus) et « Zones d'extraction »
    (dérivée : cratere_obus + zone_crateres) résolvent au même run. La classe
    source cratere_obus est DUPLIQUÉE : couche 'cratere_obus' dans le dossier
    Trous d'obus, et couche renommée (layer_names) dans le dossier Zones.
    """

    def test_shared_source_duplicated_with_configured_labels(self):
        entities = [
            _ent("cratere_obus", "trous_d_obus", ["cratere_obus"]),
            _ent("zones_extraction_materiaux", "zones_d_extraction_de_materiaux",
                 ["cratere_obus", "zone_crateres"],
                 layer_names={"cratere_obus": "crateres_constitutifs",
                              "zone_crateres": "zones_extraction"}),
        ]
        t = build_entity_class_targets(OUT, entities)
        # cratere_obus écrit dans les DEUX, couche canonique (Trous d'obus) en 1er
        assert t["cratere_obus"] == [
            (_gpkg("trous_d_obus"), "cratere_obus"),
            (_gpkg("zones_d_extraction_de_materiaux"), "crateres_constitutifs"),
        ]
        # le cluster est renommé via layer_names
        assert t["zone_crateres"] == [(_gpkg("zones_d_extraction_de_materiaux"), "zones_extraction")]

    def test_canonical_target_is_first_regardless_of_order(self):
        # même si la dérivée est listée d'abord, la couche canonique reste primaire
        entities = [
            _ent("zones_extraction_materiaux", "zones_d_extraction_de_materiaux",
                 ["cratere_obus", "zone_crateres"],
                 layer_names={"cratere_obus": "crateres_constitutifs"}),
            _ent("cratere_obus", "trous_d_obus", ["cratere_obus"]),
        ]
        t = build_entity_class_targets(OUT, entities)
        assert t["cratere_obus"][0] == (_gpkg("trous_d_obus"), "cratere_obus")

    def test_no_rename_when_layer_names_empty(self):
        # entité dérivée seule, layer_names ne renomme que la source (défaut _source)
        entities = [
            _ent("zones_extraction_materiaux", "zones_d_extraction_de_materiaux",
                 ["cratere_obus", "zone_crateres"],
                 layer_names={"cratere_obus": "cratere_obus_source"}),
        ]
        t = build_entity_class_targets(OUT, entities)
        assert t["cratere_obus"] == [(_gpkg("zones_d_extraction_de_materiaux"), "cratere_obus_source")]
        # cluster non renommé (absent de layer_names) → garde son nom de classe
        assert t["zone_crateres"] == [(_gpkg("zones_d_extraction_de_materiaux"), "zone_crateres")]

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
