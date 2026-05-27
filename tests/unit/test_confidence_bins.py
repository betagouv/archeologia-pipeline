"""Tests purs pour les tranches de confiance (``conf_bin``).

Le bug « dégradé non appliqué » vient d'un décalage entre le ``min_confidence``
ayant binné ``conf_bin`` (au moment de la conversion) et celui ayant servi à
bâtir le renderer catégorisé du ``.qgs`` : si la 1ʳᵉ tranche diffère (ex. data
``[0.3:0.4[`` vs renderer ``[0.2:0.4[``), les détections de la tranche basse ne
matchent aucune catégorie → non colorées. La correction dérive le
``min_confidence`` du renderer du **minimum réel** des ``conf_bin`` présents.
"""
from __future__ import annotations

from pipeline.cv.class_utils import (
    compute_confidence_bins,
    conf_bin_lower_bound,
    filter_detections_below_confidence,
)


class TestConfBinLowerBound:
    def test_normal_open_bin(self):
        assert conf_bin_lower_bound("[0.3:0.4[") == 0.3

    def test_last_closed_bin(self):
        assert conf_bin_lower_bound("[0.8:1]") == 0.8

    def test_zero_lower(self):
        assert conf_bin_lower_bound("[0:0.2[") == 0.0

    def test_invalid_returns_none(self):
        assert conf_bin_lower_bound("garbage") is None
        assert conf_bin_lower_bound("") is None
        assert conf_bin_lower_bound(None) is None


class TestBinsMatchDataLabels:
    def test_min_from_data_reproduces_data_labels(self):
        # données binnées à 0.3 → ces libellés sont présents dans le gpkg
        data_labels = ["[0.3:0.4[", "[0.4:0.6[", "[0.6:0.8[", "[0.8:1]"]
        m = min(conf_bin_lower_bound(lbl) for lbl in data_labels)
        bins = compute_confidence_bins(m)
        # le renderer reconstruit avec ce min reproduit EXACTEMENT les libellés data
        assert [b["label"] for b in bins] == data_labels

    def test_partial_data_still_starts_at_min(self):
        # même si la tranche haute manque, on part du min réel (0.3)
        data_labels = ["[0.3:0.4[", "[0.4:0.6[", "[0.6:0.8["]
        m = min(conf_bin_lower_bound(lbl) for lbl in data_labels)
        labels = [b["label"] for b in compute_confidence_bins(m)]
        assert labels[0] == "[0.3:0.4["


def _det(conf):
    """Détection minimale (les autres attributs sont sans importance ici)."""
    return {"confidence": conf, "model_pred": "x"}


class TestFilterBelowConfidence:
    """Filtrage final : le .gpkg ne doit contenir que des détections ≥ seuil du run.

    Appelé APRÈS le clustering — les classes de clusters (déjà ≥ seuil cluster, et
    issues d'une hystérésis qui absorbe volontairement des points sous-seuil) sont
    épargnées.
    """

    def test_drops_below_keeps_at_or_above(self):
        data = {"parcellaire": [_det(0.05), _det(0.29), _det(0.3), _det(0.55)]}
        out = filter_detections_below_confidence(data, 0.3)
        confs = [d["confidence"] for d in out["parcellaire"]]
        assert confs == [0.3, 0.55]

    def test_exempt_cluster_class_untouched(self):
        # une classe cluster est conservée intégralement, même si sous le seuil
        data = {
            "crateres": [_det(0.1), _det(0.4)],
            "regroupement_crateres": [_det(0.1), _det(0.25)],
        }
        out = filter_detections_below_confidence(
            data, 0.3, exempt_classes={"regroupement_crateres"}
        )
        assert [d["confidence"] for d in out["crateres"]] == [0.4]
        assert [d["confidence"] for d in out["regroupement_crateres"]] == [0.1, 0.25]

    def test_min_confidence_zero_is_noop(self):
        data = {"a": [_det(0.0), _det(0.1), _det(0.9)]}
        out = filter_detections_below_confidence(data, 0.0)
        assert [d["confidence"] for d in out["a"]] == [0.0, 0.1, 0.9]

    def test_negative_or_none_threshold_is_noop(self):
        data = {"a": [_det(0.1)]}
        assert filter_detections_below_confidence(data, -1)["a"] == data["a"]
        assert filter_detections_below_confidence(data, None)["a"] == data["a"]

    def test_confidence_none_is_kept(self):
        # une détection sans confiance n'est jamais filtrée (cas défensif)
        data = {"a": [_det(None), _det(0.1), _det(0.5)]}
        out = filter_detections_below_confidence(data, 0.3)
        confs = [d["confidence"] for d in out["a"]]
        assert None in confs and 0.5 in confs and 0.1 not in confs

    def test_all_below_yields_empty_list(self):
        # scénario chemins_creux : max 0.22 < seuil 0.3 → couche vide
        data = {"chemins_creux": [_det(0.05), _det(0.18), _det(0.22)]}
        out = filter_detections_below_confidence(data, 0.3)
        assert out["chemins_creux"] == []

    def test_does_not_mutate_input(self):
        data = {"a": [_det(0.1), _det(0.5)]}
        filter_detections_below_confidence(data, 0.3)
        assert len(data["a"]) == 2  # entrée intacte
