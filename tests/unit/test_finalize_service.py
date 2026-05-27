from __future__ import annotations

from pathlib import Path

from app.services import finalize_service


class TestBuildEntityGrouping:
    """slug→libellé + ensemble des slugs d'entités dérivées (regroupement partagé)."""

    def test_derived_entity_in_derived_slugs(self):
        runs = [{
            "model": "cratere_circulaire_2",
            "entities": [
                {"id": "cratere", "slug": "crateres", "label": "Cratères", "is_derived": False},
                {"id": "regroupement_crateres", "slug": "regroupement_de_crateres",
                 "label": "Regroupement de cratères", "is_derived": True},
            ],
        }]
        labels, derived = finalize_service.build_entity_grouping(runs)
        assert labels == {
            "crateres": "Cratères",
            "regroupement_de_crateres": "Regroupement de cratères",
        }
        assert derived == {"regroupement_de_crateres"}

    def test_no_derived_means_empty_set(self):
        runs = [{"entities": [{"slug": "parcellaire", "label": "Parcellaire"}]}]
        labels, derived = finalize_service.build_entity_grouping(runs)
        assert labels == {"parcellaire": "Parcellaire"}
        assert derived == set()

    def test_empty_or_malformed_runs(self):
        assert finalize_service.build_entity_grouping(None) == ({}, set())
        assert finalize_service.build_entity_grouping([]) == ({}, set())
        assert finalize_service.build_entity_grouping([{"model": "m"}]) == ({}, set())
        assert finalize_service.build_entity_grouping(["bad", {"entities": ["x"]}]) == ({}, set())

    def test_label_falls_back_to_slug(self):
        runs = [{"entities": [{"slug": "fours"}]}]
        labels, _ = finalize_service.build_entity_grouping(runs)
        assert labels == {"fours": "fours"}


class TestCollectVrtPathsAndBuild:
    def test_builds_and_returns_vrt_for_tif_dirs_only(self, tmp_path: Path, monkeypatch):
        """Seuls les dossiers tif/ sont indexés ; png/ et annotated_images/ sont ignorés."""
        idx_dir = tmp_path / "indices"
        det_dir = tmp_path / "detections"
        tif_dir = idx_dir / "LD" / "tif"
        png_dir = idx_dir / "LD" / "png"
        annotated_dir = det_dir / "model" / "annotated_images"
        for d in (tif_dir, png_dir, annotated_dir):
            d.mkdir(parents=True)
        (tif_dir / "tile.tif").write_bytes(b"tif")
        (png_dir / "tile.png").write_bytes(b"png")
        (annotated_dir / "tile.png").write_bytes(b"png")

        built: list[Path] = []

        def fake_build_vrt_index(folder, *, pattern="*.tif", output_name="index.vrt", log=lambda _m: None):
            built.append(folder)
            (folder / output_name).write_text(pattern, encoding="utf-8")
            return True

        monkeypatch.setattr(
            "pipeline.ign.products.results.build_vrt_index",
            fake_build_vrt_index,
        )

        paths = finalize_service._collect_vrt_paths_and_build(idx_dir, det_dir, lambda _m: None)

        assert built == [tif_dir]
        assert paths == [str(tif_dir / "index.vrt")]
        assert not (png_dir / "index.vrt").exists()
        assert not (annotated_dir / "index.vrt").exists()

    def test_always_rebuilds_existing_vrt(self, tmp_path: Path, monkeypatch):
        """Un index.vrt déjà présent est systématiquement régénéré (pas de skip-if-exists)."""
        idx_dir = tmp_path / "indices"
        det_dir = tmp_path / "detections"
        tif_dir = idx_dir / "LD" / "tif"
        tif_dir.mkdir(parents=True)
        (tif_dir / "tile.tif").write_bytes(b"tif")
        (tif_dir / "index.vrt").write_text("stale", encoding="utf-8")  # VRT obsolète déjà présent

        built: list[Path] = []

        def fake_build_vrt_index(folder, *, pattern="*.tif", output_name="index.vrt", log=lambda _m: None):
            built.append(folder)
            (folder / output_name).write_text("fresh", encoding="utf-8")
            return True

        monkeypatch.setattr(
            "pipeline.ign.products.results.build_vrt_index",
            fake_build_vrt_index,
        )

        paths = finalize_service._collect_vrt_paths_and_build(idx_dir, det_dir, lambda _m: None)

        assert built == [tif_dir]  # rebuild déclenché malgré le VRT existant
        assert paths == [str(tif_dir / "index.vrt")]
        assert (tif_dir / "index.vrt").read_text(encoding="utf-8") == "fresh"

    def test_skips_tif_dir_without_tif_files(self, tmp_path: Path, monkeypatch):
        """Un dossier tif/ sans .tif ne génère ni ne retourne de VRT, même avec un index.vrt résiduel."""
        idx_dir = tmp_path / "indices"
        det_dir = tmp_path / "detections"
        tif_dir = idx_dir / "LD" / "tif"
        tif_dir.mkdir(parents=True)
        (tif_dir / "index.vrt").write_text("stale", encoding="utf-8")  # résiduel, sans .tif

        def fail_build(*_a, **_k):
            raise AssertionError("build_vrt_index ne doit pas être appelé sans .tif")

        monkeypatch.setattr("pipeline.ign.products.results.build_vrt_index", fail_build)

        paths = finalize_service._collect_vrt_paths_and_build(idx_dir, det_dir, lambda _m: None)

        assert paths == []


class TestBuildMinConfidenceBySlug:
    """Le seuil de confiance par couche du .qgs consolidé doit refléter le
    seuil du run qui a produit la couche (= valeur utilisée pour binner conf_bin),
    pas un unique seuil global. Sinon les catégories de légende ne matchent pas
    les conf_bin et la tranche basse devient invisible.
    """

    def test_maps_each_entity_slug_to_its_run_threshold(self):
        runs = [
            {"confidence_threshold": 0.3, "entities": [{"slug": "parcellaire", "label": "Parcellaire"}]},
            {"confidence_threshold": 0.5, "entities": [{"slug": "talus_et_fosses"}]},
        ]
        result = finalize_service.build_min_confidence_by_slug(runs)
        assert result == {"parcellaire": 0.3, "talus_et_fosses": 0.5}

    def test_same_slug_from_two_runs_keeps_minimum(self):
        runs = [
            {"confidence_threshold": 0.5, "entities": [{"slug": "x"}]},
            {"confidence_threshold": 0.3, "entities": [{"slug": "x"}]},
        ]
        result = finalize_service.build_min_confidence_by_slug(runs)
        assert result == {"x": 0.3}

    def test_run_without_entities_uses_model_slug(self):
        runs = [{"confidence_threshold": 0.3, "selected_model": "whatever"}]
        result = finalize_service.build_min_confidence_by_slug(
            runs, model_slug_fn=lambda _run: "mymodel"
        )
        assert result == {"mymodel": 0.3}

    def test_missing_or_empty_threshold_defaults_to_zero(self):
        runs = [
            {"entities": [{"slug": "a"}]},  # pas de confidence_threshold
            {"confidence_threshold": None, "entities": [{"slug": "b"}]},
        ]
        result = finalize_service.build_min_confidence_by_slug(runs)
        assert result == {"a": 0.0, "b": 0.0}

    def test_ignores_empty_input_and_malformed_runs(self):
        assert finalize_service.build_min_confidence_by_slug([]) == {}
        assert finalize_service.build_min_confidence_by_slug(None) == {}
        # entité sans slug ignorée, run non-dict ignoré
        runs = ["bad", {"confidence_threshold": 0.4, "entities": [{"label": "no slug"}]}]
        assert finalize_service.build_min_confidence_by_slug(runs) == {}

    def test_output_78_regression_per_entity_threshold_not_global(self):
        # Régression du bug d'affichage : sur output_78, le seuil GLOBAL
        # (computer_vision.confidence_threshold = 0.2) doit être ignoré au profit
        # du seuil PAR RUN (0.3, posé par entité dans l'UI). La map ne dépend que
        # des runs — la symbologie doit la consommer, pas le global 0.2.
        runs = [
            {"confidence_threshold": 0.3, "entities": [{"slug": "regroupement_de_crateres"}]},
            {"confidence_threshold": 0.3, "entities": [
                {"slug": "chemins_creux"}, {"slug": "parcellaire"}, {"slug": "talus_et_fosses"},
            ]},
            {"confidence_threshold": 0.3, "entities": [
                {"slug": "charbonnieres"}, {"slug": "depressions_circulaires"}, {"slug": "fours"},
            ]},
        ]
        result = finalize_service.build_min_confidence_by_slug(runs)
        assert set(result.values()) == {0.3}  # aucun 0.2
        assert result["regroupement_de_crateres"] == 0.3
        assert result["parcellaire"] == 0.3


class TestCollectDetectionLayers:
    def test_collects_geopackage_layers_from_shapefiles_directory(self, tmp_path: Path, monkeypatch):
        gpkg = tmp_path / "detections" / "model" / "shapefiles" / "detections.gpkg"
        gpkg.parent.mkdir(parents=True)
        gpkg.write_bytes(b"gpkg")
        monkeypatch.setattr(finalize_service, "_list_gpkg_layers", lambda _p: ["cratere", "zone"])

        paths = finalize_service._collect_shapefiles(tmp_path / "detections")

        assert paths == [
            f"{gpkg}|layername=cratere",
            f"{gpkg}|layername=zone",
        ]
