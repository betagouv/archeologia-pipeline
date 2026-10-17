"""Nuage fusionné pré-filtré (2026-10-17) : la fusion en flux ne garde que les classes du filtre de l'étape 2,
les seules que le MNT et la densité utilisent. Mesuré sur la dalle 0816_6333 : 97 → 22 M points, 543 → 126 Mo,
MNT 170 → 42 s et 10,7 → 6,3 Go, MNT et densité identiques au pixel. Les bornes du nuage AVANT filtrage sont
relevées en flux (filters.stats) et écrites dans le sidecar : MNT et densité y calent leur grille, pas sur
l'en-tête du nuage filtré, qui peut être plus étroit."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from pipeline.ign import preprocess as pp
from pipeline.ign.products import mnt as mnt_mod

B1 = {"xmin": "0499800", "xmax": "0500000", "ymin": "6499000", "ymax": "6500000"}
QGIS = "Classification = 2 OR Classification = 6 OR Classification = 66 OR Classification = 67 OR Classification = 9"
PDAL = "Classification == 2 || Classification == 6 || Classification == 66 || Classification == 67 || Classification == 9"


class TestExpressionPdal:
    def test_defaut_du_plugin(self):
        assert pp.expression_pdal(QGIS) == PDAL

    def test_operateurs_et_parentheses(self):
        assert pp.expression_pdal("(Classification = 2 OR Classification <> 7) AND Z >= 10") == \
            "(Classification == 2 || Classification != 7) && Z >= 10"

    @pytest.mark.parametrize("inconnue", ["Classification IN (2, 6)", "Classification = 'sol'", "", "   ", "Z = 2 ; x"])
    def test_expression_non_reconnue(self, inconnue):
        assert pp.expression_pdal(inconnue) is None


@pytest.fixture
def pdal(monkeypatch):
    """PDAL factice : écrit le fusionné et, si demandé, les métadonnées de filters.stats (bornes des X, Y)."""
    etat = {"commandes": [], "pipelines": []}

    def executer(cmd, cancel=None, **k):
        etat["commandes"].append(list(cmd))
        doc = json.loads(Path(cmd[cmd.index("pipeline") + 2 if "--stream" in cmd else cmd.index("pipeline") + 1])
                         .read_text(encoding="utf-8"))
        etat["pipelines"].append(doc)
        Path(doc["pipeline"][-1]["filename"]).write_bytes(b"LASF-fusion")
        if "--metadata" in cmd:
            Path(cmd[cmd.index("--metadata") + 1]).write_text(json.dumps({"stages": {"filters.stats": {"statistic": [
                {"name": "X", "minimum": 815799.76, "maximum": 817200.31},
                {"name": "Y", "minimum": 6331800.0, "maximum": 6333199.5}]}}}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(pp, "validate_las_or_laz_with_pdal", lambda p, *a, **k: (True, ""))
    monkeypatch.setattr(pp, "run_pdal_command_cancellable", executer)
    monkeypatch.setattr(pp, "_pdal_exe", lambda: "pdal")
    return etat


class TestPipelineFiltre:
    def test_filtre_apres_les_bandes_et_bornes_relevees_avant(self, tmp_path):
        etapes = pp.pipeline_fusion(tmp_path / "c.laz", [(tmp_path / "v.laz", B1)], tmp_path / "s.laz", filtre=PDAL)["pipeline"]
        assert [e["type"] for e in etapes] == [
            "readers.las", "readers.las", "filters.crop", "filters.stats", "filters.expression", "writers.las"]
        stats, expr, ecriture = etapes[3], etapes[4], etapes[5]
        assert stats["inputs"] == [etapes[0]["tag"], etapes[2]["tag"]] and stats["dimensions"] == "X,Y"
        assert expr["expression"] == PDAL and expr["inputs"] == [stats["tag"]]
        assert ecriture["inputs"] == [expr["tag"]]

    def test_sans_filtre_pipeline_inchange(self, tmp_path):
        etapes = pp.pipeline_fusion(tmp_path / "c.laz", [(tmp_path / "v.laz", B1)], tmp_path / "s.laz")["pipeline"]
        assert [e["type"] for e in etapes] == ["readers.las", "readers.las", "filters.crop", "writers.las"]


class TestFusionFiltree:
    def _fusion(self, tmp_path, pdal, filtre, journal=None):
        out = tmp_path / "merged.laz"
        ok = pp.merge_tiles_flux(central_path=tmp_path / "c.laz", voisines=[(tmp_path / "v.laz", B1)], output_path=out,
                                 filtre=filtre, log=(journal.append if journal is not None else lambda m: None))
        assert ok
        return json.loads(pp.merged_inputs_sidecar(out).read_text(encoding="utf-8"))

    def test_sidecar_porte_le_filtre_et_les_bornes_du_nuage_brut(self, tmp_path, pdal):
        sidecar = self._fusion(tmp_path, pdal, PDAL)
        assert "--metadata" in pdal["commandes"][0]
        assert sidecar["filtre"] == PDAL
        assert sidecar["bornes"] == [815799.76, 817200.31, 6331800.0, 6333199.5]
        assert sidecar["fusion"] == pp.FUSION_FLUX == "flux-2"

    def test_sans_filtre_ni_bornes(self, tmp_path, pdal):
        sidecar = self._fusion(tmp_path, pdal, None)
        assert "--metadata" not in pdal["commandes"][0]
        assert sidecar["filtre"] is None and "bornes" not in sidecar

    def test_filtre_change_refusionne(self, tmp_path, pdal):
        self._fusion(tmp_path, pdal, PDAL)
        self._fusion(tmp_path, pdal, "Classification == 2")
        assert len(pdal["commandes"]) == 2

    def test_meme_filtre_reutilise(self, tmp_path, pdal):
        self._fusion(tmp_path, pdal, PDAL)
        self._fusion(tmp_path, pdal, PDAL)
        assert len(pdal["commandes"]) == 1


class TestTacheDeFusion:
    def test_le_filtre_de_l_etape_2_arrive_traduit_a_la_fusion(self, tmp_path, monkeypatch):
        recu: dict = {}
        monkeypatch.delenv("ARCHEO_FUSION_LEGACY", raising=False)
        monkeypatch.setattr(pp, "merge_tiles_flux", lambda **kw: recu.update(kw) or True)
        task = pp._PreprocessTask(index=1, total=1, filename="c.laz", url="u", tile_name="LHD_FXX_0500_6500", x="0500",
                                  y="6500", central_path=tmp_path / "c.laz", temp_dir=tmp_path, merged_dir=tmp_path,
                                  margin_m=200, dalles_dir=tmp_path, filtre=QGIS)
        assert pp._process_single_tile_preprocess(task, {}, lambda m: None, lambda: False).success
        assert recu["filtre"] == PDAL

    def test_filtre_non_reconnu_fusion_sans_filtre_et_le_journal_le_dit(self, tmp_path, monkeypatch):
        recu: dict = {}
        journal: list = []
        monkeypatch.delenv("ARCHEO_FUSION_LEGACY", raising=False)
        monkeypatch.setattr(pp, "merge_tiles_flux", lambda **kw: recu.update(kw) or True)
        task = pp._PreprocessTask(index=1, total=1, filename="c.laz", url="u", tile_name="LHD_FXX_0500_6500", x="0500",
                                  y="6500", central_path=tmp_path / "c.laz", temp_dir=tmp_path, merged_dir=tmp_path,
                                  margin_m=200, dalles_dir=tmp_path, filtre="Classification IN (2, 6)")
        assert pp._process_single_tile_preprocess(task, {}, journal.append, lambda: False).success
        assert recu["filtre"] is None
        assert any("non reconnu" in m and "sans filtre" in m for m in journal)


def test_le_journal_renvoie_au_depannage():
    from app.services.aide import rubrique_depannage
    assert rubrique_depannage("⚠️ Filtre des points non reconnu (Classification IN (2)) : fusion sans filtre, "
                              "le MNT et la densité filtrent eux-mêmes (plus lent)") == "Le filtre des points n'est pas reconnu"


class TestBornesDuNuage:
    def test_le_sidecar_prime_sur_l_entete(self, tmp_path, monkeypatch):
        laz = tmp_path / "LHD_merged.laz"
        laz.write_bytes(b"LASF")
        pp.merged_inputs_sidecar(laz).write_text(json.dumps({"fusion": "flux-2", "voisins": [], "filtre": PDAL,
                                                            "bornes": [1.0, 2.0, 3.0, 4.0]}), encoding="utf-8")
        monkeypatch.setattr(mnt_mod.shutil, "which", lambda n: None)   # aucun pdal : l'en-tête ne peut pas être lue
        assert mnt_mod._bornes_du_nuage(laz) == (1.0, 2.0, 3.0, 4.0)

    def test_sans_bornes_dans_le_sidecar_l_entete(self, tmp_path, monkeypatch):
        laz = tmp_path / "LHD_merged.laz"
        laz.write_bytes(b"LASF")
        pp.merged_inputs_sidecar(laz).write_text(json.dumps({"fusion": "flux-1", "voisins": []}), encoding="utf-8")
        monkeypatch.setattr(mnt_mod.shutil, "which", lambda n: None)
        assert mnt_mod._bornes_du_nuage(laz) is None
