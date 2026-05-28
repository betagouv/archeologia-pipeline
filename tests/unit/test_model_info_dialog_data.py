"""Tests pour la logique pure du dialog d'info modèle.

Le module ``ui.dialogs._model_info_data`` est PUR : pas d'import Qt, pas d'I/O.
Il prend un ``model_card.yaml`` déjà parsé (et optionnellement un ``args.yaml``)
et produit une liste de ``Section`` à afficher. Cela isole tout ce qui peut être
testé hors-QGIS et garde le widget Qt minimal.
"""
from __future__ import annotations

from ui.dialogs._model_info_data import (
    RVT_PARAM_LABELS,
    Row,
    Section,
    build_sections,
    pretty_rvt_name,
    pretty_task,
)


# ----------------------------------------------------------------------
# Helpers d'humanisation
# ----------------------------------------------------------------------
class TestPrettyRvtName:
    def test_known_codes(self):
        assert pretty_rvt_name("LD") == "Local Dominance (LD)"
        assert pretty_rvt_name("SVF") == "Sky View Factor (SVF)"
        assert pretty_rvt_name("M_HS") == "Hillshade multi-directionnel (M-HS)"
        assert pretty_rvt_name("HS") == "Hillshade simple (HS)"
        assert pretty_rvt_name("SLO") == "Pente (SLO)"
        assert pretty_rvt_name("SLRM") == "Simple Local Relief Model (SLRM)"
        assert pretty_rvt_name("VAT") == "Visualisation Archéologique Totale (VAT)"

    def test_unknown_code_falls_back_to_raw(self):
        # Repli : on retourne le code brut sans planter.
        assert pretty_rvt_name("FOO") == "FOO"
        assert pretty_rvt_name("") == ""


class TestPrettyTask:
    def test_known_codes(self):
        assert pretty_task("object_detection") == "Détection d'objets"
        assert pretty_task("instance_segmentation") == "Segmentation d'instances"
        assert pretty_task("semantic_segmentation") == "Segmentation sémantique"

    def test_unknown_task_falls_back_to_raw(self):
        assert pretty_task("classification") == "classification"
        assert pretty_task("") == ""


# ----------------------------------------------------------------------
# Tables d'alias des paramètres RVT
# ----------------------------------------------------------------------
class TestRvtParamLabels:
    def test_svf_canonical_keys(self):
        labels = RVT_PARAM_LABELS["SVF"]
        assert labels["num_directions"] == "Nombre de directions"
        assert labels["radius"] == "Rayon"
        assert labels["noise_remove"] == "Suppression du bruit"

    def test_svf_non_canonical_aliases(self):
        # Verdun_3_classes_1 utilise des clés non-canoniques dans son
        # preferred_rvt.params. On les mappe sur les MÊMES libellés FR que
        # les clés canoniques pour offrir une UI uniforme côté utilisateur.
        labels = RVT_PARAM_LABELS["SVF"]
        assert labels["svf_n_dir"] == "Nombre de directions"
        assert labels["svf_r_max"] == "Rayon"
        assert labels["svf_noise"] == "Suppression du bruit"

    def test_ld_canonical_keys(self):
        labels = RVT_PARAM_LABELS["LD"]
        assert labels["angular_res"] == "Résolution angulaire"
        assert labels["min_radius"] == "Rayon min"
        assert labels["max_radius"] == "Rayon max"
        assert labels["observer_h"] == "Hauteur observateur"
        assert labels["ve_factor"] == "Facteur VE"
        assert labels["save_as_8bit"] == "Export 8 bits"

    def test_hs_canonical_keys(self):
        labels = RVT_PARAM_LABELS["HS"]
        assert labels["sun_azimuth"] == "Azimut solaire"
        assert labels["sun_elevation"] == "Élévation solaire"
        assert labels["ve_factor"] == "Facteur VE"


# ----------------------------------------------------------------------
# build_sections
# ----------------------------------------------------------------------
class TestBuildSections:
    def test_full_card_yields_architecture_rvt_mnt_sections(self):
        card = {
            "id": "test_model",
            "display_name": "Test Model",
            "task": "instance_segmentation",
            "architecture": "RF-DETR-Seg-Large",
            "variant": "large",
            "resolution_train": 504,
            "resolution_inference": 504,
            "preferred_rvt": {
                "type": "LD",
                "params": {
                    "angular_res": 15,
                    "min_radius": 10,
                    "max_radius": 20,
                    "observer_h": 1.7,
                    "ve_factor": 1,
                    "save_as_8bit": True,
                },
            },
            "mnt": {
                "resolution": 0.5,
                "filter_expression": "Classification = 2 OR Classification = 6",
            },
            "classes": [
                {"id": 0, "name": "cratere", "label_fr": "Cratère"},
            ],
        }
        sections = build_sections(card, args=None)
        titles = [s.title for s in sections]
        assert "ARCHITECTURE" in titles
        assert any(t.startswith("INDICE RVT D'ENTRAÎNEMENT") and "(LD)" in t for t in titles)
        assert "MNT D'ENTRAÎNEMENT" in titles

    def test_architecture_section_humanizes_task_and_counts_unique_classes(self):
        card = {
            "task": "object_detection",
            "architecture": "RF-DETR",
            "resolution_inference": 704,
            "classes": [
                {"name": "charbonniere"},
                {"name": "charbonniere"},  # doublon → dédupliqué (le modèle fusionne)
                {"name": "four"},
            ],
        }
        sections = build_sections(card, args=None)
        arch = next(s for s in sections if s.title == "ARCHITECTURE")
        labels_lower = [r.label.lower() for r in arch.rows]
        values = [r.value for r in arch.rows]
        assert "modèle" in labels_lower
        assert "tâche" in labels_lower
        assert "Détection d'objets" in values
        # Maquette : compte des classes UNIQUES dans le label (« Classes (2) »),
        # liste des noms dans la valeur.
        classes_row = next(r for r in arch.rows if "lasse" in r.label)
        assert "2" in classes_row.label
        assert "charbonniere" in classes_row.value
        assert "four" in classes_row.value

    def test_architecture_includes_variant_when_present(self):
        # On choisit ``base`` parce que ``RF-DETR`` ne contient pas ``base``
        # — l'apparition du variant dans la valeur ne peut venir que du suffixe.
        card = {"architecture": "RF-DETR", "variant": "base"}
        sections = build_sections(card, args=None)
        arch = next(s for s in sections if s.title == "ARCHITECTURE")
        model_row = next(r for r in arch.rows if r.label.lower() == "modèle")
        assert "base" in model_row.value.lower()

    def test_rvt_section_uses_canonical_labels(self):
        card = {
            "preferred_rvt": {
                "type": "SVF",
                "params": {"num_directions": 16, "radius": 10, "noise_remove": 0},
            },
        }
        sections = build_sections(card, args=None)
        rvt = next(s for s in sections if s.title.startswith("INDICE RVT"))
        labels = [r.label for r in rvt.rows]
        assert "Nombre de directions" in labels
        assert "Rayon" in labels
        assert "Suppression du bruit" in labels

    def test_rvt_section_resolves_non_canonical_svf_aliases(self):
        # verdun_3_classes_1 : clés svf_n_dir / svf_r_max / svf_noise → mêmes
        # libellés FR que les clés canoniques. Aucune clé brute ne doit fuir à l'écran.
        card = {
            "preferred_rvt": {
                "type": "SVF",
                "params": {
                    "svf_n_dir": 16,
                    "svf_r_max": 10,
                    "svf_noise": 0,
                    "save_as_8bit": True,
                },
            },
        }
        sections = build_sections(card, args=None)
        rvt = next(s for s in sections if s.title.startswith("INDICE RVT"))
        labels = [r.label for r in rvt.rows]
        assert "svf_n_dir" not in labels
        assert "svf_r_max" not in labels
        assert "svf_noise" not in labels
        assert "Nombre de directions" in labels
        assert "Rayon" in labels
        assert "Suppression du bruit" in labels

    def test_rvt_section_unknown_key_displayed_raw(self):
        # Clé inconnue → on garde la clé brute dans le label (pas de plantage).
        card = {
            "preferred_rvt": {
                "type": "LD",
                "params": {"angular_res": 15, "magic_param": 42},
            },
        }
        sections = build_sections(card, args=None)
        rvt = next(s for s in sections if s.title.startswith("INDICE RVT"))
        labels = [r.label for r in rvt.rows]
        assert "magic_param" in labels

    def test_no_rvt_section_when_preferred_rvt_missing(self):
        card = {"architecture": "X", "task": "object_detection"}
        sections = build_sections(card, args=None)
        rvt_titles = [s.title for s in sections if s.title.startswith("INDICE RVT")]
        assert rvt_titles == []

    def test_mnt_section_has_resolution_and_filter(self):
        card = {
            "mnt": {
                "resolution": 0.5,
                "filter_expression": "Classification = 2 OR Classification = 6",
            },
        }
        sections = build_sections(card, args=None)
        mnt = next(s for s in sections if s.title == "MNT D'ENTRAÎNEMENT")
        labels = [r.label for r in mnt.rows]
        values = [r.value for r in mnt.rows]
        assert any("solution" in l.lower() for l in labels)
        assert any("0.5" in v for v in values)
        assert any("iltre" in l for l in labels)
        # Le filtre doit être rendu en monospace (sera utilisé par le widget Qt).
        filter_row = next(r for r in mnt.rows if "iltre" in r.label)
        assert filter_row.mono is True

    def test_no_mnt_section_when_mnt_missing(self):
        card = {"architecture": "X"}
        sections = build_sections(card, args=None)
        titles = [s.title for s in sections]
        assert "MNT D'ENTRAÎNEMENT" not in titles

    def test_minimal_card_does_not_crash(self):
        sections = build_sections({}, args=None)
        assert isinstance(sections, list)

    def test_clustering_section_when_args_has_clustering(self):
        card: dict = {}
        args = {
            "clustering": [
                {
                    "target_classes": ["cratere"],
                    "output_class_name": "zone_crateres",
                    "eps_m": 40,
                    "min_cluster_size": 40,
                    "min_samples": 5,
                    "min_confidence": 0.4,
                    "buffer_m": 10,
                    "min_area_m2": 1000,
                }
            ]
        }
        sections = build_sections(card, args=args)
        cluster_sections = [s for s in sections if "REGROUPEMENT" in s.title]
        assert cluster_sections, "Section regroupement attendue quand args.yaml a une règle clustering"
        cluster = cluster_sections[0]
        # Section fermée par défaut (la maquette laisse REGROUPEMENT optionnel).
        assert cluster.collapsed is True
        values = [r.value for r in cluster.rows]
        # Tous les paramètres clés doivent apparaître quelque part dans les valeurs.
        joined = " ".join(values)
        for needed in ("cratere", "zone_crateres", "40", "5", "0.4", "10", "1000"):
            assert needed in joined

    def test_no_clustering_section_when_args_missing_or_empty(self):
        sections = build_sections({}, args=None)
        assert [s for s in sections if "REGROUPEMENT" in s.title] == []

        sections = build_sections({}, args={"clustering": []})
        assert [s for s in sections if "REGROUPEMENT" in s.title] == []

    def test_notes_section_when_recommended_use_or_limitations_present(self):
        card = {
            "recommended_use": "Sur emprises forestières.",
            "known_limitations": ["limitation A", "limitation B"],
        }
        sections = build_sections(card, args=None)
        notes = [s for s in sections if "NOTES" in s.title]
        assert notes, "Section NOTES attendue quand recommended_use ou known_limitations existent"
        # Fermée par défaut (secondaire).
        assert notes[0].collapsed is True
        values = " ".join(r.value for r in notes[0].rows)
        assert "forestières" in values
        assert "limitation A" in values

    def test_no_notes_section_when_nothing_to_show(self):
        sections = build_sections({"architecture": "X"}, args=None)
        assert [s for s in sections if "NOTES" in s.title] == []


# ----------------------------------------------------------------------
# Sanity check : dataclasses immuables
# ----------------------------------------------------------------------
class TestDataclasses:
    def test_section_is_frozen(self):
        s = Section(title="X", rows=())
        try:
            s.title = "Y"  # type: ignore[misc]
        except Exception:
            pass
        else:
            raise AssertionError("Section devrait être frozen")

    def test_row_has_default_mono_false(self):
        r = Row(label="A", value="B")
        assert r.mono is False
