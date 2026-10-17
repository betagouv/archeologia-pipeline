"""Lot 1 (accélération 2026-10) : fusion d'une dalle et de ses voisines en UN
`pdal pipeline --stream`, tout lu par readers.las (dalle isolée comprise) : le
LAZ fusionné — donc le MNT — est le même d'un run à l'autre. L'ancienne fusion
(rognages puis `pdal merge`, centrale COPC lue sur plusieurs fils) reste derrière
ARCHEO_FUSION_LEGACY=1 pendant une version."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from pipeline.cancellation import PipelineCancelled
from pipeline.ign import preprocess as pp
from pipeline.ign.pdal_validation import lire_version_pdal

TILE = "LHD_FXX_0500_6500"
B1 = {"xmin": "0499800", "xmax": "0500000", "ymin": "6499000", "ymax": "6500000"}
B2 = {"xmin": "0501000", "xmax": "0501200", "ymin": "6499000", "ymax": "6500000"}


def _laz(p: Path) -> Path:
    p.write_bytes(b"LASF")
    return p


@pytest.fixture
def pdal(monkeypatch):
    """PDAL factice : lit le pipeline, échoue si une source a des points illisibles."""
    etat = {"pipelines": [], "commandes": [], "illisibles_entete": set(), "illisibles_points": set(),
            "plante": False, "sans_flux": False, "lectures": []}

    def valider(path, *a, **k):
        etat["lectures"].append(Path(path).name)
        return (Path(path).name not in etat["illisibles_entete"], "")

    def valider_profond(path, *a, **k):
        etat["lectures"].append(Path(path).name)
        return (Path(path).name not in etat["illisibles_points"], "")

    def executer(cmd, cancel=None, **k):
        etat["commandes"].append(list(cmd))
        doc = json.loads(Path(cmd[-1]).read_text(encoding="utf-8"))
        etat["pipelines"].append(doc)
        if etat["sans_flux"] and "--stream" in cmd:
            return subprocess.CompletedProcess(
                cmd, 1, "", "PDAL: Attempting to use stream mode with a stage that doesn't support streaming.")
        lus = {Path(e["filename"]).name for e in doc["pipeline"] if e["type"] == "readers.las"}
        if lus & etat["illisibles_points"]:
            return subprocess.CompletedProcess(cmd, 1, "", "readers.las: Couldn't read point data")
        Path(doc["pipeline"][-1]["filename"]).write_bytes(b"LASF-fusion")
        if etat["plante"]:
            return subprocess.CompletedProcess(cmd, 3221225477, "", "access violation")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(pp, "validate_las_or_laz_with_pdal", valider)
    monkeypatch.setattr(pp, "validate_laz_deep", valider_profond)
    monkeypatch.setattr(pp, "run_pdal_command_cancellable", executer)
    monkeypatch.setattr(pp, "_pdal_exe", lambda: "pdal")
    return etat


def _sources(doc):
    return [Path(e["filename"]).name for e in doc["pipeline"] if e["type"] == "readers.las"]


class TestPipelineFusion:
    def test_centrale_puis_bandes_toutes_en_readers_las(self, tmp_path):
        c, v1, v2 = tmp_path / "c.copc.laz", tmp_path / "v1.copc.laz", tmp_path / "v2.laz"
        etapes = pp.pipeline_fusion(c, [(v1, B1), (v2, B2)], tmp_path / "s.laz")["pipeline"]
        assert [e["type"] for e in etapes] == [
            "readers.las", "readers.las", "filters.crop", "readers.las", "filters.crop", "writers.las"]
        assert etapes[0]["filename"] == str(c)
        assert etapes[2]["bounds"] == "([0499800,0500000],[6499000,6500000])"
        assert etapes[2]["inputs"] == [etapes[1]["tag"]]
        assert etapes[-1]["inputs"] == [etapes[0]["tag"], etapes[2]["tag"], etapes[4]["tag"]]
        assert etapes[-1]["compression"] == "laszip" and etapes[-1]["filename"] == str(tmp_path / "s.laz")

    def test_dalle_isolee_lue_aussi_par_readers_las(self, tmp_path):
        etapes = pp.pipeline_fusion(tmp_path / "c.copc.laz", [], tmp_path / "s.laz")["pipeline"]
        assert [e["type"] for e in etapes] == ["readers.las", "writers.las"]


class TestMergeTilesFlux:
    def _fusion(self, tmp_path, voisines):
        c = _laz(tmp_path / f"{TILE}.copc.laz")
        out = tmp_path / f"{TILE}_merged.laz"
        ok = pp.merge_tiles_flux(central_path=c, voisines=voisines, output_path=out)
        return ok, out

    def test_un_seul_pipeline_en_flux(self, tmp_path, pdal):
        ok, out = self._fusion(tmp_path, [(_laz(tmp_path / "v1.copc.laz"), B1)])
        assert ok is True
        assert [c[1:3] for c in pdal["commandes"]] == [["pipeline", "--stream"]]
        assert out.read_bytes() == b"LASF-fusion" and not pp._partial_path(out).exists()
        assert json.loads(pp.merged_inputs_sidecar(out).read_text(encoding="utf-8")) == {
            "fusion": pp.FUSION_FLUX, "voisins": ["v1.copc.laz"], "filtre": None}

    def test_dalle_isolee(self, tmp_path, pdal):
        ok, out = self._fusion(tmp_path, [])
        assert ok is True
        assert [e["type"] for e in pdal["pipelines"][0]["pipeline"]] == ["readers.las", "writers.las"]
        assert json.loads(pp.merged_inputs_sidecar(out).read_text(encoding="utf-8"))["voisins"] == []

    def test_voisine_a_l_entete_illisible_ignoree(self, tmp_path, pdal):
        pdal["illisibles_entete"] = {"v2.laz"}
        ok, _out = self._fusion(tmp_path, [(_laz(tmp_path / "v1.copc.laz"), B1), (_laz(tmp_path / "v2.laz"), B2)])
        assert ok is True
        assert _sources(pdal["pipelines"][0]) == [f"{TILE}.copc.laz", "v1.copc.laz"]

    def test_voisine_aux_points_illisibles_ecartee_puis_refusion(self, tmp_path, pdal):
        pdal["illisibles_points"] = {"v2.laz"}
        ok, out = self._fusion(tmp_path, [(_laz(tmp_path / "v1.copc.laz"), B1), (_laz(tmp_path / "v2.laz"), B2)])
        assert ok is True
        assert len(pdal["commandes"]) == 2
        assert _sources(pdal["pipelines"][1]) == [f"{TILE}.copc.laz", "v1.copc.laz"]
        # Le sidecar garde les candidates : la voisine illisible n'est pas retentée à chaque run.
        assert json.loads(pp.merged_inputs_sidecar(out).read_text(encoding="utf-8"))["voisins"] == [
            "v1.copc.laz", "v2.laz"]

    def test_meme_jeu_reutilise_le_fusionne(self, tmp_path, pdal):
        v1 = _laz(tmp_path / "v1.copc.laz")
        self._fusion(tmp_path, [(v1, B1)])
        ok, _out = self._fusion(tmp_path, [(v1, B1)])
        assert ok is True and len(pdal["commandes"]) == 1

    def test_fusionne_d_une_version_precedente_refait(self, tmp_path, pdal):
        out = _laz(tmp_path / f"{TILE}_merged.laz")
        pp.merged_inputs_sidecar(out).write_text(json.dumps([f"{TILE}_neighbor_4.laz"]), encoding="utf-8")
        ok, _out = self._fusion(tmp_path, [(_laz(tmp_path / "v1.copc.laz"), B1)])
        assert ok is True and len(pdal["commandes"]) == 1
        assert json.loads(pp.merged_inputs_sidecar(out).read_text(encoding="utf-8"))["fusion"] == pp.FUSION_FLUX

    def test_fusionne_sans_sidecar_refait(self, tmp_path, pdal):
        _laz(tmp_path / f"{TILE}_merged.laz")
        ok, _out = self._fusion(tmp_path, [])
        assert ok is True and len(pdal["commandes"]) == 1

    def test_voisine_ecartee_inchangee_ni_relue_ni_refusionnee(self, tmp_path, pdal):
        pdal["illisibles_points"] = {"v2.laz"}
        voisines = [(_laz(tmp_path / "v1.copc.laz"), B1), (_laz(tmp_path / "v2.laz"), B2)]
        self._fusion(tmp_path, voisines)
        pdal["lectures"].clear()
        ok, _out = self._fusion(tmp_path, voisines)
        assert ok is True and len(pdal["commandes"]) == 2
        assert "v2.laz" not in pdal["lectures"]

    def test_voisine_ecartee_reparee_refusionnee_avec_sa_bande(self, tmp_path, pdal):
        pdal["illisibles_points"] = {"v2.laz"}
        v1, v2 = _laz(tmp_path / "v1.copc.laz"), _laz(tmp_path / "v2.laz")
        self._fusion(tmp_path, [(v1, B1), (v2, B2)])
        v2.write_bytes(b"LASF-reparee")
        pdal["illisibles_points"] = set()
        ok, out = self._fusion(tmp_path, [(v1, B1), (v2, B2)])
        assert ok is True and len(pdal["commandes"]) == 3
        assert _sources(pdal["pipelines"][2]) == [f"{TILE}.copc.laz", "v1.copc.laz", "v2.laz"]
        assert "ecartees" not in json.loads(pp.merged_inputs_sidecar(out).read_text(encoding="utf-8"))

    def test_pdal_qui_plante_ne_laisse_pas_de_laz_final(self, tmp_path, pdal):
        pdal["plante"] = True
        ok, out = self._fusion(tmp_path, [(_laz(tmp_path / "v1.copc.laz"), B1)])
        assert ok is False
        assert not out.exists() and not pp._partial_path(out).exists()
        assert not pp.merged_inputs_sidecar(out).exists()

    def test_annulation_ne_laisse_ni_fusionne_ni_partial(self, tmp_path, pdal, monkeypatch):
        def annule(cmd, cancel=None, **k):
            doc = json.loads(Path(cmd[-1]).read_text(encoding="utf-8"))
            Path(doc["pipeline"][-1]["filename"]).write_bytes(b"LASF-moitie")
            raise PipelineCancelled("Annulation demandée")

        monkeypatch.setattr(pp, "run_pdal_command_cancellable", annule)
        c = _laz(tmp_path / f"{TILE}.copc.laz")
        out = tmp_path / f"{TILE}_merged.laz"
        with pytest.raises(PipelineCancelled):
            pp.merge_tiles_flux(central_path=c, voisines=[(_laz(tmp_path / "v1.copc.laz"), B1)], output_path=out)
        assert not out.exists() and not pp._partial_path(out).exists()
        assert not pp.merged_inputs_sidecar(out).exists()

    def test_pdal_sans_flux_retombe_sur_la_meme_fusion_en_memoire(self, tmp_path, pdal):
        pdal["sans_flux"] = True
        ok, _out = self._fusion(tmp_path, [(_laz(tmp_path / "v1.copc.laz"), B1)])
        assert ok is True
        assert "--stream" in pdal["commandes"][0] and "--stream" not in pdal["commandes"][1]
        assert pdal["pipelines"][0] == pdal["pipelines"][1]


def _bloc(tmp_path, garder):
    """Dalle centrale 0500_6500 et les voisines dont la place est dans ``garder``."""
    dalles = tmp_path / "dalles"
    dalles.mkdir()
    central = _laz(dalles / f"{TILE}_PTS_C_LAMB93_IGN69.copc.laz")
    index = {}
    for vx, vy, place in pp.calculate_neighbor_coordinates("0500", "6500"):
        if place in garder:
            cle = f"{pp.format_coordinate(vx)}_{pp.format_coordinate(vy)}"
            nom = f"LHD_FXX_{cle}_PTS_C_LAMB93_IGN69.copc.laz"
            _laz(dalles / nom)
            index[cle] = (nom, f"https://data.geopf.fr/{nom}")
    task = pp._PreprocessTask(
        index=1, total=1, filename=central.name, url="https://data.geopf.fr/c", tile_name=TILE,
        x="0500", y="6500", central_path=central, temp_dir=tmp_path / "t", merged_dir=tmp_path / "t",
        margin_m=200, dalles_dir=dalles,
    )
    attendu = [
        (dalles / index[f"{pp.format_coordinate(vx)}_{pp.format_coordinate(vy)}"][0],
         pp.calculate_crop_bounds(vx, vy, place, 200))
        for vx, vy, place in pp.calculate_neighbor_coordinates("0500", "6500") if place in garder
    ]
    return task, index, attendu


class TestChoixDesVoisines:
    @pytest.mark.parametrize("garder", [
        {1, 2, 3, 4, 5, 6, 7, 8},   # intérieure
        {4, 5, 6, 7, 8},            # bord : 5 voisines
        {4, 6, 7},                  # coin : 3 voisines
        set(),                      # isolée
    ])
    def test_memes_voisines_et_memes_bornes_que_l_ancienne_fusion(self, tmp_path, monkeypatch, garder):
        task, index, attendu = _bloc(tmp_path, garder)
        recu: dict = {}

        def flux(**kw):
            recu.update(kw)
            return True

        monkeypatch.delenv("ARCHEO_FUSION_LEGACY", raising=False)
        monkeypatch.setattr(pp, "merge_tiles_flux", flux)
        assert pp._process_single_tile_preprocess(task, index, lambda m: None, lambda: False).success
        assert recu["voisines"] == attendu

        rognages: list = []

        def rogner(**kw):
            rognages.append((kw["input_path"], kw["bounds"]))
            return True

        monkeypatch.setenv("ARCHEO_FUSION_LEGACY", "1")
        monkeypatch.setattr(pp, "crop_neighbor_tile", rogner)
        monkeypatch.setattr(pp, "merge_tiles", lambda **kw: True)
        assert pp._process_single_tile_preprocess(task, index, lambda m: None, lambda: False).success
        assert rognages == attendu

    def test_interrupteur(self, monkeypatch):
        monkeypatch.delenv("ARCHEO_FUSION_LEGACY", raising=False)
        assert pp.fusion_mode() == pp.FUSION_FLUX
        monkeypatch.setenv("ARCHEO_FUSION_LEGACY", "1")
        assert pp.fusion_mode() == pp.FUSION_ANCIENNE


def test_version_pdal_lue_dans_la_banniere():
    banniere = "-" * 80 + "\npdal 2.10.0 (git-version: eb9126)\n" + "-" * 80 + "\n"
    assert lire_version_pdal(banniere) == "2.10.0 (git-version: eb9126)"
    assert lire_version_pdal("") == ""


def test_versions_lidar_dans_la_trace(tmp_path):
    pytest.importorskip("shapely")  # resolve_cv_runs importé par finalize_pipeline
    import time

    from app.progress_reporter import NullProgressReporter
    from app.services.finalize_service import finalize_pipeline

    finalize_pipeline(output_dir=tmp_path, cv_cfg={}, rvt_params={}, reporter=NullProgressReporter(),
                      slog=None, start_time=time.time(), pdal_version="2.10.0 (git-version: eb9126)",
                      fusion=pp.FUSION_FLUX)
    trace = json.loads((tmp_path / "livrable" / "traitement.json").read_text(encoding="utf-8"))
    assert trace["pdal_version"] == "2.10.0 (git-version: eb9126)" and trace["fusion_laz"] == "flux-2"
