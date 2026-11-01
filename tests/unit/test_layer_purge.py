"""Purge des couches périmées avant un re-run dans le même ``output_dir``.

Au re-run, les couches ``index.vrt`` du run précédent restent chargées dans QGIS :
QGIS continue d'afficher le VRT périmé et réécrit même sa version en mémoire
par-dessus le VRT régénéré. La parade est de retirer, au lancement du run, les
couches dont la source pointe dans ``<output_dir>/indices`` ou ``.../detections``.

Helper PUR (aucun import QGIS) : décide *quelles* couches retirer à partir d'un
``{id: source}`` ; le retrait QGIS lui-même (``removeMapLayers``) reste côté UI.
"""
from __future__ import annotations

from pathlib import Path

from app.services.layer_purge import select_layers_to_purge

_OUT = r"D:\pipeline_results\output_bretagne"


def test_selects_indices_vrt_under_output_dir():
    sources = {"L1": r"D:\pipeline_results\output_bretagne\indices\CVAT\tif\index.vrt"}
    assert select_layers_to_purge(sources, _OUT) == ["L1"]


def test_selects_detections_gpkg_stripping_layername_decoration():
    src = r"D:\pipeline_results\output_bretagne\detections\parcellaire\parcellaire.gpkg|layername=parcellaire"
    assert select_layers_to_purge({"L1": src}, _OUT) == ["L1"]


def test_ignores_layer_outside_output_dir():
    sources = {"L1": r"D:\autre_projet\indices\MNT\tif\index.vrt"}
    assert select_layers_to_purge(sources, _OUT) == []


def test_sibling_prefix_is_not_a_match():
    # output_dir = ...\output_bretagne ; une couche sous ...\output_bretagne_old
    # ne doit PAS être prise (préfixe de chaîne ≠ préfixe de composants).
    src = r"D:\pipeline_results\output_bretagne_old\indices\MNT\tif\index.vrt"
    assert select_layers_to_purge({"L1": src}, _OUT) == []


def test_ignores_sources_and_intermediaires_subtrees():
    sources = {
        "L1": r"D:\pipeline_results\output_bretagne\sources\dalles\LHD.copc.laz",
        "L2": r"D:\pipeline_results\output_bretagne\intermediaires\foo.tif",
        "L3": r"D:\pipeline_results\output_bretagne\metadata.json",
    }
    assert select_layers_to_purge(sources, _OUT) == []


def test_windows_path_normalization_case_and_separators():
    # output_dir en casse mixte + antislash ; source en minuscules + slash avant.
    out = r"D:\Pipeline_Results\Output_Bretagne"
    src = "d:/pipeline_results/output_bretagne/indices/MNT/tif/index.vrt"
    assert select_layers_to_purge({"L1": src}, out) == ["L1"]


def test_different_output_dir_selects_nothing():
    src = r"D:\pipeline_results\autre_sortie\indices\MNT\tif\index.vrt"
    assert select_layers_to_purge({"L1": src}, _OUT) == []


def test_non_filesystem_and_empty_sources_are_ignored():
    sources = {
        "L1": "",
        "L2": "Point?crs=EPSG:2154&field=id:integer",  # couche mémoire
        "L3": "dbname='gis' host=localhost port=5432 table=\"public\".\"t\"",  # postgres
    }
    assert select_layers_to_purge(sources, _OUT) == []


def test_mixed_layers_returns_exactly_the_two_subtree_ids():
    sources = {
        "vrt": r"D:\pipeline_results\output_bretagne\indices\CVAT\tif\index.vrt",
        "det": r"D:\pipeline_results\output_bretagne\detections\crateres\crateres.gpkg|layername=crateres",
        "src": r"D:\pipeline_results\output_bretagne\sources\dalles\x.laz",
        "ext": r"D:\autre\indices\MNT\tif\index.vrt",
        "grid": "C:/grille/quadrillage.shp",
    }
    assert set(select_layers_to_purge(sources, _OUT)) == {"vrt", "det"}


def test_posix_paths_under_output_dir_are_selected():
    out = "/home/u/out/zone"
    src = "/home/u/out/zone/indices/MNT/tif/index.vrt"
    assert select_layers_to_purge({"L1": src}, out) == ["L1"]


def test_empty_output_dir_selects_nothing():
    sources = {"L1": r"D:\pipeline_results\output_bretagne\indices\CVAT\tif\index.vrt"}
    assert select_layers_to_purge(sources, "") == []


def test_la_purge_retire_les_groupes_qu_elle_a_vides():
    """Constat de recette 68.4 (2026-10-26) : après le lancement d'un re-run, « Structures » et
    « Regroupement de cratères » restaient vides dans l'arbre de QGIS. La purge retire les groupes
    du plugin qu'elle a vidés — jamais un groupe de l'utilisateur, jamais un groupe encore peuplé."""
    src = (Path(__file__).resolve().parents[2] / "src" / "ui" / "layer_loader.py").read_text(encoding="utf-8")
    # La règle (groupe de l'analyse, Structures, Indices, ancien Structures de la racine) est
    # testée sur un arbre simulé dans test_nom_analyse.py ; ici, son branchement.
    helper = src[src.index("def _retirer_groupes_vides("):src.index("def purge_output_dir_layers(")]
    assert "retirer_groupes_vides(project.layerTreeRoot(), QgsLayerTree.isGroup)" in helper
    purge = src[src.index("def purge_output_dir_layers("):src.index("def _deferred_zoom(")]
    assert "_retirer_groupes_vides(project)" in purge


def test_purge_par_chemin_pas_par_nom(tmp_path):
    """Les couches sont purgées par leur source sous livrable/, jamais par leur nom : renommer
    une couche (« Parcellaire », « LD · Structures en relief ») ne change rien à la purge."""
    out = tmp_path / "sortie"
    sources = {
        "Parcellaire": str(out / "livrable" / "detections" / "parcellaire" / "parcellaire.gpkg") + "|layername=Parcellaire",
        "LD · Structures en relief": str(out / "livrable" / "indices" / "LD_A15" / "tif" / "index_LD_A15.vrt"),
        "index_LD": str(tmp_path / "ailleurs" / "index_LD.vrt"),
    }
    assert sorted(select_layers_to_purge(sources, out)) == ["LD · Structures en relief", "Parcellaire"]
