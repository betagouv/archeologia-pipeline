"""Mineurs laissés par les revues du chantier d'accélération, embarqués dans la 0.15.0 (décision 2026-10-18) :
annulation pendant la relecture profonde des voisines ; copie d'un LAZ local en .partial ; note « vérifiée »
dont l'échec d'écriture est journalisé ; mode de secours Processing qui ne recalcule plus MSTP, M-HS et OPNS
à chaque relance ; moteur d'indices sondé une fois ; interrupteur de fusion tolérant ; nom du M-HS 8 bits sans
les directions ; bornes SVF/OPNS refusées avant le lancement comme LD/SLRM ; secours Processing réservé ;
clé « declared_crs » transmise au préflight ; clés de configuration sans effet retirées."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from app.services.rvt_kernel_context import bornes_errors, message_ecretage
from pipeline.ign import downloader as dl
from pipeline.ign import preprocess as pp
from pipeline.ign.products import moteur_indices
from pipeline.ign.products.rvt_naming import get_rvt_param_suffix

B1 = {"xmin": "0499800", "xmax": "0500000", "ymin": "6499000", "ymax": "6500000"}


# ── fusion : annulation pendant la relecture profonde des voisines ───────────────────────────────────────────
def test_l_annulation_arrete_la_relecture_profonde_des_voisines(tmp_path, monkeypatch):
    profondes: list = []
    etat = {"annule": False}

    def executer(cmd, cancel=None, **k):
        return subprocess.CompletedProcess(cmd, 1, "", "PDAL: readers.las: Unable to open stream")

    def profonde(path, *a, **k):
        profondes.append(Path(path).name)
        etat["annule"] = True            # l'utilisateur annule pendant la première relecture
        return (True, "")

    monkeypatch.setattr(pp, "validate_las_or_laz_with_pdal", lambda p, *a, **k: (True, ""))
    monkeypatch.setattr(pp, "validate_laz_deep", profonde)
    monkeypatch.setattr(pp, "run_pdal_command_cancellable", executer)
    monkeypatch.setattr(pp, "_pdal_exe", lambda: "pdal")
    ok = pp.merge_tiles_flux(central_path=tmp_path / "c.laz", voisines=[(tmp_path / "v1.laz", B1), (tmp_path / "v2.laz", B1)],
                             output_path=tmp_path / "m.laz", cancel=lambda: etat["annule"])
    assert ok is False and profondes == ["v1.laz"]


# ── téléchargement : copie locale en .partial, note « vérifiée » journalisée ─────────────────────────────────
@pytest.fixture
def copie_locale(tmp_path, monkeypatch):
    src = tmp_path / "src" / "LHD_FXX_0500_6500.copc.laz"
    src.parent.mkdir()
    src.write_bytes(b"LASF" * 10)
    monkeypatch.setattr(dl, "_is_local_url", lambda url: src)
    dalles = tmp_path / "dalles"
    dalles.mkdir()
    return src, dalles


def test_une_copie_locale_invalide_ne_laisse_ni_dalle_ni_partial(copie_locale, monkeypatch):
    src, dalles = copie_locale
    monkeypatch.setattr(dl, "_is_valid_with_pdal", lambda p: (False, "corrompu"))
    ok, _ = dl.download_one("file:///x", src.name, dalles, log=lambda m: None, cancel=lambda: False)
    assert ok is False and sorted(p.name for p in dalles.iterdir()) == []


def test_une_copie_locale_valide_passe_par_un_partial(copie_locale, monkeypatch):
    src, dalles = copie_locale
    vus: list = []
    monkeypatch.setattr(dl, "_is_valid_with_pdal", lambda p: vus.append(Path(p).name) or (True, ""))
    ok, _ = dl.download_one("file:///x", src.name, dalles, log=lambda m: None, cancel=lambda: False)
    assert ok is True and (dalles / src.name).read_bytes() == src.read_bytes()
    assert vus == [src.name + ".partial"] and not list(dalles.glob("*.partial"))


def test_l_echec_d_ecriture_de_la_note_verifiee_est_journalise(tmp_path):
    dest = tmp_path / "LHD_FXX_0500_6500.copc.laz"
    dest.write_bytes(b"LASF")
    dl.note_verifiee_path(dest).mkdir()          # la note ne peut pas s'écrire : un dossier porte son nom
    journal: list = []
    dl.ecrire_note_verifiee(dest, log=journal.append)
    assert journal and "note" in journal[0].lower() and dest.name in journal[0]


# ── indices : moteur sondé une fois ; interrupteur de fusion tolérant ────────────────────────────────────────
def test_le_moteur_d_indices_n_est_sonde_qu_une_fois():
    moteur_indices.moteur_disponible.cache_clear()
    moteur_indices.moteur_disponible()
    moteur_indices.moteur_disponible()
    assert moteur_indices.moteur_disponible.cache_info().hits == 1


@pytest.mark.parametrize("valeur,attendu", [("1", pp.FUSION_ANCIENNE), ("true", pp.FUSION_ANCIENNE), ("OUI", pp.FUSION_ANCIENNE),
                                            ("yes", pp.FUSION_ANCIENNE), ("on", pp.FUSION_ANCIENNE),
                                            ("0", pp.FUSION_FLUX), ("non", pp.FUSION_FLUX), ("", pp.FUSION_FLUX)])
def test_l_interrupteur_de_fusion_accepte_oui_vrai_et_1(monkeypatch, valeur, attendu):
    monkeypatch.setenv("ARCHEO_FUSION_LEGACY", valeur)
    assert pp.fusion_mode() == attendu


# ── nom du M-HS : en 8 bits les directions ne comptent pas, le nom ne les porte plus ────────────────────────
def test_le_m_hs_8_bits_ne_porte_pas_ses_directions_dans_son_nom():
    assert get_rvt_param_suffix("M_HS", {"mdh": {"num_directions": 8}}) == "_D16_E35_V1"
    assert get_rvt_param_suffix("M_HS", {"mdh": {"num_directions": 8, "save_as_8bit": True}}) == "_D16_E35_V1"
    assert get_rvt_param_suffix("M_HS", {"mdh": {"num_directions": 8, "save_as_8bit": False}}) == "_D8_E35_V1_F32"


# ── bornes : SVF et OPNS refusés avant le lancement comme LD et SLRM ; secours Processing réservé ───────────
@pytest.mark.parametrize("produit,section,cle,valeur,ok,libelle", [
    ("SVF", "svf", "radius", 60, 20, "Rayon"), ("SVF", "svf", "num_directions", 100, 32, "Nombre de directions"),
    ("OPNS", "opns", "radius", 5, 20, "Rayon"), ("OPNS", "opns", "noise_remove", 5, 2, "Suppression du bruit"),
])
def test_svf_et_opns_hors_bornes_refuses_avant_le_lancement(produit, section, cle, valeur, ok, libelle):
    erreurs = bornes_errors({produit: True}, {section: {cle: valeur}})
    assert len(erreurs) == 1 and produit in erreurs[0] and libelle in erreurs[0]
    assert bornes_errors({produit: True}, {section: {cle: ok}}) == []
    assert libelle in message_ecretage(produit, cle, valeur, 10, 50, False)


def test_en_secours_processing_les_directions_du_m_hs_8_bits_sont_bornees():
    params = {"mdh": {"num_directions": 100, "save_as_8bit": True}}
    assert bornes_errors({"M_HS": True}, params) == []
    assert len(bornes_errors({"M_HS": True}, params, secours_processing=True)) == 1


# ── configuration : clés sans effet retirées, « declared_crs » transmis au préflight ────────────────────────
def test_les_cles_sans_effet_ne_sont_plus_dans_les_defauts():
    from config.config_manager import ConfigManager
    cv = ConfigManager(Path(__file__).resolve().parents[2]).default_config()["computer_vision"]
    assert "scan_all" not in cv and "generate_shapefiles" not in cv
    exemple = json.loads((Path(__file__).resolve().parents[2] / "config.example.json").read_text(encoding="utf-8"))
    assert "scan_all" not in exemple["computer_vision"] and "generate_shapefiles" not in exemple["computer_vision"]
    charge = {"computer_vision": {"scan_all": True, "generate_shapefiles": False, "iou_threshold": 0.5}}
    ConfigManager._strip_deprecated_keys(charge)     # une configuration d'une version précédente est nettoyée
    assert charge == {"computer_vision": {"iou_threshold": 0.5}}


def test_le_crs_declare_est_transmis_au_controle_des_rasters(tmp_path, monkeypatch):
    pytest.importorskip("osgeo")
    from pipeline import ingest_plan, preflight
    recu: dict = {}

    def plan(paths, declared_crs=None, expected_crs=None):
        recu.update(declared_crs=declared_crs)
        raise ingest_plan.IngestValidationError("stop")
    monkeypatch.setattr(ingest_plan, "plan_raster_inputs", plan)
    (tmp_path / "a.asc").write_text("x", encoding="utf-8")
    preflight._check_raster_crs(tmp_path, ("asc",), [], declared_crs="EPSG:2154")
    assert recu["declared_crs"] == "EPSG:2154"
