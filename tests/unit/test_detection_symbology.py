"""Prédicat ``is_synthesis_layer`` : quelle symbologie pour une couche de détection ?

Bug corrigé (§23 QGIS, sortie Bretagne) : la couche « Enclos » (60 entités)
était invisible — le choix du style zone était câblé sur le champ ``nb_detect``
(spécifique DBSCAN), et les sorties enclosure/alignment (``nb_sources``)
tombaient dans le rendu catégorisé ``conf_bin`` qui ne matche rien (pas de
tranches sur les sorties de synthèse).
"""
from __future__ import annotations

from app.services.detection_symbology import is_synthesis_layer


def test_cluster_layer_detected():
    assert is_synthesis_layer(["fid", "model_pred", "nb_detect", "area_m2"])


def test_enclosure_and_alignment_layers_use_confidence_categories():
    # Depuis la confiance composite, les enclos/axes portent un conf_bin
    # cohérent → rendu catégorisé par confiance comme les détections
    # (« même granularité que les parcellaires »). Seules les zones DBSCAN
    # (nb_detect) gardent le style hachuré.
    assert not is_synthesis_layer(["model_pred", "nb_sources", "enclos_id", "conf_bin"])
    assert not is_synthesis_layer(["model_pred", "nb_sources", "axe_id", "conf_bin"])


def test_source_layer_with_tag_columns_not_synthesis():
    # Les couches SOURCES portent cluster_id/enclos_id/axe_id (traçabilité
    # membre→synthèse) mais PAS nb_detect → symbologie confiance.
    assert not is_synthesis_layer(
        ["model_pred", "confidence", "conf_bin", "cluster_id", "enclos_id", "axe_id"]
    )


def test_plain_detection_layer_not_synthesis():
    assert not is_synthesis_layer(["model_pred", "confidence", "conf_bin"])
    assert not is_synthesis_layer([])
