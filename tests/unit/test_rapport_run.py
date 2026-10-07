"""Rapport de traitement (2026-10-08) : extraction, rendu HTML pur, écriture par la finalisation."""
from __future__ import annotations

import json
import time
from pathlib import Path

from app.progress_reporter import NullProgressReporter
from app.services import finalize_service
from app.services.rapport_run import (
    NOM_RAPPORT,
    Avertissement,
    DonneesRapport,
    choisir_vrt_vignette,
    construire_html,
    dalles_depuis_vrt,
    extraire_avertissements,
    format_duree,
    nom_dalle,
    vignette_depuis_vrt,
)


class _Reporter(NullProgressReporter):
    def __init__(self):
        self.users: list = []

    def user_info(self, msg: str) -> None:
        self.users.append(msg)

    def user_success(self, msg: str) -> None:
        self.users.append(msg)


def test_extraire_avertissements_du_journal_de_fichier():
    log = (
        "2026-10-08 10:00:01,123 - INFO - Lancement\n"
        "2026-10-08 10:00:02,000 - WARNING - LD : le noyau atteint 40 px mais la marge n'en fait que 20\n"
        "2026-10-08 10:00:03,000 - WARNING - LD : le noyau atteint 40 px mais la marge n'en fait que 20\n"
        "2026-10-08 10:00:04,000 - ERROR - Connection to data.geopf.fr timed out\n"
        "ligne sans format\n"
    )
    a = extraire_avertissements(log)
    assert [(x.niveau, x.rubrique) for x in a] == [
        ("warn", "Le noyau atteint N pixels"), ("err", "Le téléchargement échoue"),
    ]
    assert a[0].message.startswith("LD : le noyau")
    assert extraire_avertissements("", 5) == []
    assert len(extraire_avertissements("\n".join(f"x - WARNING - m{i}" for i in range(9)), 3)) == 3


def test_dalles_vignette_et_formats(tmp_path):
    assert nom_dalle("LHD_FXX_0873_6506_MNT_A_0M50_LAMB93_IGN69") == "0873-6506"
    assert nom_dalle("mon_raster") == "mon_raster"
    tif = tmp_path / "indices" / "SVF_R10" / "tif"
    tif.mkdir(parents=True)
    for s in ("LHD_FXX_0873_6506_MNT_A_0M50_LAMB93_IGN69", "LHD_FXX_0874_6506_MNT_A_0M50_LAMB93_IGN69"):
        (tif / f"{s}.tif").write_bytes(b"")
    vrt = str(tif / "index_SVF.vrt")
    assert dalles_depuis_vrt([vrt]) == ["0873-6506", "0874-6506"]
    assert dalles_depuis_vrt([]) == []
    mnt = str(tmp_path / "indices" / "MNT" / "tif" / "index_MNT.vrt")
    assert choisir_vrt_vignette([mnt, vrt]) == vrt                 # SVF plus parlant que MNT
    assert choisir_vrt_vignette([mnt]) == mnt and choisir_vrt_vignette([]) is None
    assert vignette_depuis_vrt(str(tmp_path / "absent.vrt"), tmp_path / "v.png") is None   # sans GDAL / absent : None
    assert format_duree(7) == "7s" and format_duree(247) == "4min 07s" and format_duree(3725) == "1h 02min"


def test_construire_html_echappe_et_couvre_les_sections():
    d = DonneesRapport(
        version="0.14.0", date="8 octobre 2026 à 10:42", mode="Téléchargement IGN", output_dir="C:/x/<sortie>",
        issue="success", duree_s=3725, tiles_processed=9, tiles_total=9, dalles=("0873-6506",),
        produits=(("SVF", "facteur de vue du ciel (SVF)"),), rvt_params={"svf": {"radius": 10}},
        cv_runs=({"modele": "Modèle <A>", "target_rvt": "LD", "entites": ["Parcellaire"], "images": 12, "secondes": 247},
                 {"modele": "B", "target_rvt": "LD", "entites": [], "images": 0, "secondes": 0}),
        bilan=({"label": "Parcellaire", "total": 103, "effectifs": {"quasi_certain": 12, "probable": 30, "possible": 41, "douteux": 20}},),
        avertissements=(Avertissement("warn", "noyau <40> px", "Le noyau atteint N pixels"),),
        vignette="rapport_vignette.png",
    )
    h = construire_html(d)
    assert "&lt;sortie&gt;" in h and "Modèle &lt;A&gt;" in h and "<sortie>" not in h and "noyau &lt;40&gt;" in h
    for morceau in ("Zone traitée", "Produits et réglages", "Détection automatique", "Bilan de fiabilité",
                    "Avertissements du journal", "12 images en 4min 07s", "aucune image analysée",
                    "12 très probables, 30 probables, 41 possibles, 20 douteuses", "1h 02min",
                    'src="rapport_vignette.png"', "radius</th>" if False else "radius = 10", "Le noyau atteint N pixels"):
        assert morceau in h, morceau
    assert "None" not in h
    sans = construire_html(DonneesRapport(version="?", date="d", mode="m", output_dir="o", issue="cancelled"))
    assert "annulé" in sans and "Pas de vignette" in sans and "Aucun avertissement" in sans
    assert "Pas de détection automatique" in sans and "Aucune détection avec fiabilité" in sans


def test_finalize_ecrit_le_rapport(tmp_path, monkeypatch):
    monkeypatch.setattr(finalize_service, "_collect_vrt_paths_and_build", lambda *a, **k: [])
    monkeypatch.setattr(finalize_service, "_build_coverage_polygons", lambda *a, **k: None)
    (tmp_path / "pipeline_log_20261008_104200.txt").write_text(
        "2026-10-08 10:42:00,000 - WARNING - Dalle 3 abandonnée après 2 tentatives\n", encoding="utf-8")
    cv_cfg = {"enabled": True, "runs": [{"selected_model": "M", "target_rvt": "LD",
                                        "entities": [{"id": "parcellaire", "slug": "parcellaire", "label": "Parcellaire"}]}]}
    r = _Reporter()
    ok = finalize_service.finalize_pipeline(
        output_dir=tmp_path, cv_cfg=cv_cfg, rvt_params={"ld": {"min_radius": 10}}, reporter=r, slog=None,
        start_time=time.time() - 65, tiles_processed=2, tiles_total=3, active_products=["LD"],
        ui_config={"app": {"files": {"data_mode": "local_laz"}}}, outcome="success",
        cv_stats=[{"modele": "Modèle test", "model": "M", "target_rvt": "LD", "images": 4, "secondes": 9.0}],
    )
    assert ok is True
    rapport = tmp_path / NOM_RAPPORT
    assert rapport.is_file()
    h = rapport.read_text(encoding="utf-8")
    assert "Nuages locaux" in h and "2 sur 3" in h and "min_radius = 10" in h
    assert "Modèle test" in h and "4 images en 9s" in h and "Parcellaire" in h
    assert "Dalle 3 abandonnée" in h and "Une dalle est abandonnée" in h
    assert any(NOM_RAPPORT in m for m in r.users)
    meta = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert meta["rapport"] == NOM_RAPPORT


def test_la_vue_ouvre_le_rapport():
    racine = Path(__file__).resolve().parents[2]
    vue = (racine / "src/ui/run_view.py").read_text(encoding="utf-8")
    assert "NOM_RAPPORT" in vue and "_open_report" in vue
