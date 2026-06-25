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
        # Nom distinctif = nom de couche QGIS : index_<PRODUIT>.vrt (ici produit « LD »).
        assert paths == [str(tif_dir / "index_LD.vrt")]
        assert not list(png_dir.glob("*.vrt"))
        assert not list(annotated_dir.glob("*.vrt"))

    def test_always_rebuilds_existing_vrt(self, tmp_path: Path, monkeypatch):
        """Un VRT distinctif déjà présent est systématiquement régénéré (pas de skip-if-exists)."""
        idx_dir = tmp_path / "indices"
        det_dir = tmp_path / "detections"
        tif_dir = idx_dir / "LD" / "tif"
        tif_dir.mkdir(parents=True)
        (tif_dir / "tile.tif").write_bytes(b"tif")
        (tif_dir / "index_LD.vrt").write_text("stale", encoding="utf-8")  # VRT obsolète déjà présent

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
        assert paths == [str(tif_dir / "index_LD.vrt")]
        assert (tif_dir / "index_LD.vrt").read_text(encoding="utf-8") == "fresh"

    def test_removes_legacy_plain_index_vrt(self, tmp_path: Path, monkeypatch):
        """Un ``index.vrt`` hérité (runs antérieurs au nommage distinctif) est supprimé
        après régénération sous le nouveau nom — pas de doublon périmé sur disque."""
        idx_dir = tmp_path / "indices"
        det_dir = tmp_path / "detections"
        tif_dir = idx_dir / "LD" / "tif"
        tif_dir.mkdir(parents=True)
        (tif_dir / "tile.tif").write_bytes(b"tif")
        (tif_dir / "index.vrt").write_text("legacy", encoding="utf-8")  # ancien nom générique

        def fake_build_vrt_index(folder, *, pattern="*.tif", output_name="index.vrt", log=lambda _m: None):
            (folder / output_name).write_text("fresh", encoding="utf-8")
            return True

        monkeypatch.setattr(
            "pipeline.ign.products.results.build_vrt_index",
            fake_build_vrt_index,
        )

        paths = finalize_service._collect_vrt_paths_and_build(idx_dir, det_dir, lambda _m: None)

        assert paths == [str(tif_dir / "index_LD.vrt")]
        assert not (tif_dir / "index.vrt").exists()  # legacy nettoyé

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

    def test_skips_empty_geopackage_layers(self, tmp_path: Path, monkeypatch):
        # Une couche vide (0 entité, p.ex. vidée par le filtre d'aire) ne doit
        # être ni collectée ni chargée → pas d'« avertissement CRS » au final.
        gpkg = tmp_path / "detections" / "talus" / "talus.gpkg"
        gpkg.parent.mkdir(parents=True)
        gpkg.write_bytes(b"gpkg")
        monkeypatch.setattr(finalize_service, "_list_gpkg_layers", lambda _p: ["talus_fosse", "cratere"])
        counts = {"talus_fosse": 0, "cratere": 5}
        monkeypatch.setattr(
            finalize_service, "_gpkg_layer_feature_count", lambda _p, layer: counts[layer], raising=False
        )

        paths = finalize_service._collect_shapefiles(tmp_path / "detections")

        assert paths == [f"{gpkg}|layername=cratere"]

    def test_keeps_layer_when_feature_count_unknown(self, tmp_path: Path, monkeypatch):
        # Comptage indéterminable (-1) → on conserve la couche (prudence).
        gpkg = tmp_path / "detections" / "m" / "m.gpkg"
        gpkg.parent.mkdir(parents=True)
        gpkg.write_bytes(b"gpkg")
        monkeypatch.setattr(finalize_service, "_list_gpkg_layers", lambda _p: ["x"])
        monkeypatch.setattr(
            finalize_service, "_gpkg_layer_feature_count", lambda _p, layer: -1, raising=False
        )

        paths = finalize_service._collect_shapefiles(tmp_path / "detections")

        assert paths == [f"{gpkg}|layername=x"]


class TestBuildCoveragePolygons:
    @staticmethod
    def _make_coverage_indices(tmp_path: Path) -> Path:
        import numpy as np
        import pytest

        rasterio = pytest.importorskip("rasterio")
        from rasterio.transform import from_origin

        tif_dir = tmp_path / "indices" / "COUVERTURE" / "tif"
        tif_dir.mkdir(parents=True)
        arr = np.full((40, 40), 80, dtype=np.uint8)
        arr[5:15, 5:15] = 0  # 100 m² sous le seuil
        with rasterio.open(
            tif_dir / "LHD_FXX_0624_6864_couverture_A_LAMB93.tif", "w",
            driver="GTiff", height=40, width=40, count=1, dtype="uint8",
            nodata=255, transform=from_origin(0.0, 40.0, 1.0, 1.0), crs="EPSG:2154",
        ) as ds:
            ds.write(arr, 1)
        return tmp_path / "indices"

    def test_dossier_absent_renvoie_none(self, tmp_path):
        out = finalize_service._build_coverage_polygons(
            tmp_path / "indices", 30.0, lambda m: None
        )
        assert out is None

    def test_genere_le_gpkg(self, tmp_path):
        import pytest

        pytest.importorskip("shapely")
        pytest.importorskip("geopandas")
        idx_dir = self._make_coverage_indices(tmp_path)
        out = finalize_service._build_coverage_polygons(idx_dir, 30.0, lambda m: None)
        assert out is not None
        assert Path(out) == idx_dir / "COUVERTURE" / "zones_mal_couvertes.gpkg"
        assert Path(out).exists()

    def test_erreur_isolee_renvoie_none(self, tmp_path):
        # Un TIF corrompu ne doit JAMAIS faire échouer la finalisation (audit ROB).
        tif_dir = tmp_path / "indices" / "COUVERTURE" / "tif"
        tif_dir.mkdir(parents=True)
        (tif_dir / "corrompu.tif").write_bytes(b"pas un tif")
        logs = []
        out = finalize_service._build_coverage_polygons(
            tmp_path / "indices", 30.0, logs.append
        )
        assert out is None
        assert any("non g" in m for m in logs)

    def test_fallback_sans_vrt_couvre_toutes_les_dalles(self, tmp_path):
        # Sans index.vrt (échec gdalbuildvrt), le repli doit traiter TOUTES les
        # dalles, pas seulement la première (QA silencieusement incomplète sinon).
        import numpy as np
        import pytest

        rasterio = pytest.importorskip("rasterio")
        pytest.importorskip("shapely")
        gpd = pytest.importorskip("geopandas")
        from rasterio.transform import from_origin

        tif_dir = tmp_path / "indices" / "COUVERTURE" / "tif"
        tif_dir.mkdir(parents=True)
        for i, x0 in enumerate((0.0, 1000.0)):
            arr = np.full((40, 40), 80, dtype=np.uint8)
            arr[5:15, 5:15] = 0
            with rasterio.open(
                tif_dir / f"dalle_{i}_couverture.tif", "w",
                driver="GTiff", height=40, width=40, count=1, dtype="uint8",
                nodata=255, transform=from_origin(x0, 40.0, 1.0, 1.0), crs="EPSG:2154",
            ) as ds:
                ds.write(arr, 1)

        logs = []
        out = finalize_service._build_coverage_polygons(
            tmp_path / "indices", 30.0, logs.append
        )
        assert out is not None
        gdf = gpd.read_file(out, layer="zones_mal_couvertes")
        assert len(gdf) == 2  # une zone par dalle, pas seulement la première
        assert any("dalle par dalle" in m for m in logs)

    def test_collecte_sans_vrt_couvre_toutes_les_dalles(self, tmp_path):
        # Variante standalone (sans geopandas) : la COLLECTE du repli sans VRT
        # doit produire les zones de toutes les dalles, pas seulement la première.
        import numpy as np
        import pytest

        rasterio = pytest.importorskip("rasterio")
        pytest.importorskip("shapely")
        from rasterio.transform import from_origin

        tif_dir = tmp_path / "indices" / "COUVERTURE" / "tif"
        tif_dir.mkdir(parents=True)
        for i, x0 in enumerate((0.0, 1000.0)):
            arr = np.full((40, 40), 80, dtype=np.uint8)
            arr[5:15, 5:15] = 0
            with rasterio.open(
                tif_dir / f"dalle_{i}_couverture.tif", "w",
                driver="GTiff", height=40, width=40, count=1, dtype="uint8",
                nodata=255, transform=from_origin(x0, 40.0, 1.0, 1.0), crs="EPSG:2154",
            ) as ds:
                ds.write(arr, 1)

        logs = []
        polygons = finalize_service._collect_low_coverage_polygons(
            tif_dir, 30.0, logs.append
        )
        assert len(polygons) == 2
        assert any("dalle par dalle" in m for m in logs)
