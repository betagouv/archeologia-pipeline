"""Rapport de traitement (2026-10-08) : extraction, rendu HTML pur, écriture par la finalisation.
Depuis le 2026-10-08 (demande utilisateur), rien ne situe la zone : ni dalle, ni chemin."""
from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

from app.progress_reporter import NullProgressReporter
from app.services import finalize_service
from app.services.rapport_run import (
    NOM_RAPPORT,
    Avertissement,
    DonneesRapport,
    anonymiser,
    choisir_vrt_vignette,
    construire_html,
    decrire_reglages,
    extraire_avertissements,
    format_duree,
    surface_km2_depuis_vrt,
    vignette_depuis_vrt,
)


class _Reporter(NullProgressReporter):
    def __init__(self):
        self.users: list = []

    def user_info(self, msg: str) -> None:
        self.users.append(msg)

    def user_success(self, msg: str) -> None:
        self.users.append(msg)


def test_anonymiser_chemins_et_dalles():
    assert anonymiser(r"❌ Erreur rasterio pour D:\pipeline_results\fenetrange\indices\LD\tif\LHD_FXX_0988_6872_LD_A_LAMB93.tif: boom") == \
        "❌ Erreur rasterio pour LHD_FXX_…_LD_A_LAMB93.tif: boom"
    assert anonymiser("Dalle 0986-6859 abandonnée après 2 tentatives") == "Dalle … abandonnée après 2 tentatives"
    assert anonymiser("Lecture de /home/val/zone/x.laz impossible") == "Lecture de x.laz impossible"
    assert anonymiser(r"\\serveur\partage\dalle.tif") == "dalle.tif"
    assert anonymiser("1/4 des dalles, seuil 0.29, 1 000 px") == "1/4 des dalles, seuil 0.29, 1 000 px"


def test_extraire_avertissements_anonymise_et_regroupe():
    log = (
        "2026-10-08 10:00:01,123 - INFO - Lancement\n"
        "2026-10-08 10:00:02,000 - WARNING - LD : le noyau atteint 40 px mais la marge n'en fait que 20\n"
        "2026-10-08 10:00:03,000 - WARNING - LD : le noyau atteint 40 px mais la marge n'en fait que 20\n"
        "2026-10-08 10:00:04,000 - ERROR - Connection to data.geopf.fr timed out\n"
        "2026-10-08 10:00:05,000 - WARNING - ❌ Erreur rasterio pour D:\\x\\LHD_FXX_0988_6872_LD.tif: a\n"
        "2026-10-08 10:00:06,000 - WARNING - ❌ Erreur rasterio pour D:\\x\\LHD_FXX_0987_6865_LD.tif: a\n"
        "ligne sans format\n"
    )
    a = extraire_avertissements(log)
    assert [(x.niveau, x.occurrences, x.rubrique) for x in a] == [
        ("warn", 2, "Le noyau atteint N pixels"), ("err", 1, "Le téléchargement échoue"), ("warn", 2, ""),
    ]
    assert a[0].message.startswith("LD : le noyau")
    assert a[2].message == "❌ Erreur rasterio pour LHD_FXX_…_LD.tif: a"     # deux dalles → un seul message
    assert "0988" not in "".join(x.message for x in a)
    assert extraire_avertissements("", 5) == []
    assert len(extraire_avertissements("\n".join(f"x - WARNING - m{i}" for i in range(9)), 3)) == 3


def test_surface_mesuree_sur_les_rasters(tmp_path, monkeypatch):
    tif = tmp_path / "indices" / "SVF_R10" / "tif"
    tif.mkdir(parents=True)
    for s in ("a", "b", "c"):
        (tif / f"{s}.tif").write_bytes(b"")
    vrt = str(tif / "index_SVF.vrt")

    class _Ds:
        RasterXSize, RasterYSize = 2000, 2000

        def GetGeoTransform(self):  # noqa: N802 (API GDAL)
            return (0.0, 0.5, 0.0, 0.0, 0.0, -0.5)

    gdal = types.SimpleNamespace(UseExceptions=lambda: None, Open=lambda p: _Ds())
    osgeo = types.ModuleType("osgeo")
    osgeo.gdal = gdal
    monkeypatch.setitem(sys.modules, "osgeo", osgeo)
    monkeypatch.setitem(sys.modules, "osgeo.gdal", gdal)
    assert surface_km2_depuis_vrt([vrt]) == 3.0                    # trois dalles de 1 km²
    assert surface_km2_depuis_vrt([]) is None
    assert surface_km2_depuis_vrt([str(tmp_path / "vide.vrt")]) is None


def test_vignette_et_formats(tmp_path):
    vrt = str(tmp_path / "indices" / "SVF_R10" / "tif" / "index_SVF.vrt")
    mnt = str(tmp_path / "indices" / "MNT" / "tif" / "index_MNT.vrt")
    assert choisir_vrt_vignette([mnt, vrt]) == vrt                 # SVF plus parlant que MNT
    assert choisir_vrt_vignette([mnt]) == mnt and choisir_vrt_vignette([]) is None
    assert vignette_depuis_vrt(str(tmp_path / "absent.vrt"), tmp_path / "v.png") is None   # sans GDAL / absent : None
    assert format_duree(7) == "7s" and format_duree(247) == "4min 07s" and format_duree(3725) == "1h 02min"


def test_decrire_reglages_en_mots():
    assert decrire_reglages("svf", {"radius": 10, "num_directions": 16, "noise_remove": 0, "ve_factor": 1, "save_as_8bit": True}) == \
        "rayon 10 px, 16 directions, sans suppression du bruit"
    assert decrire_reglages("ldo", {"angular_res": 15, "min_radius": 10, "max_radius": 20, "observer_h": 1.7, "ve_factor": 2}) == \
        "rayon de 10 à 20 px, résolution angulaire 15°, hauteur d'observateur 1,7 m, exagération verticale ×2"
    assert decrire_reglages("opns", {"opns_type": 1, "radius": 12, "num_directions": 8, "noise_remove": 2}) == \
        "ouverture négative, les creux, rayon 12 px, 8 directions, suppression du bruit 2"
    assert decrire_reglages("vat", {"terrain_type": 2}) == "terrain pentu"
    assert decrire_reglages("slope", {"unit": 1}) == "en pourcentage"
    assert decrire_reglages("hs", {"sun_azimuth": 315, "sun_elevation": 35}) == "azimut solaire 315°, élévation solaire 35°"
    assert decrire_reglages("crim", {"colormap": "Greys_r", "min_colormap_cut": 0.0, "max_colormap_cut": 0.9}) == \
        "palette gris inversé, coupes de la palette de 0 à 0,9"
    assert decrire_reglages("cvat", None) == "composition fixe" and decrire_reglages("hs", None) == "réglages par défaut"
    mstp = decrire_reglages("mstp", {"local_scale_min": 3, "local_scale_max": 21, "local_scale_step": 2,
                                     "meso_scale_min": 23, "meso_scale_max": 203, "meso_scale_step": 18,
                                     "broad_scale_min": 100, "broad_scale_max": 400, "broad_scale_step": 60,
                                     "lightness": 1.2, "ve_factor": 1, "save_as_8bit": True})
    assert mstp.startswith("échelle locale de 3 à 21 px par pas de 2, échelle méso") and "luminosité 1,2" in mstp
    assert "save_as_8bit" not in mstp and "exagération" not in mstp


def test_construire_html_ne_situe_pas_et_couvre_les_sections():
    d = DonneesRapport(
        version="0.14.0", date="8 octobre 2026 à 10:42", mode="Téléchargement IGN", data_mode="ign_laz",
        issue="success", duree_s=3725, tiles_processed=9, tiles_total=9, surface_km2=9.0,
        produits=(("SVF", "facteur de vue du ciel (SVF)"), ("MNT", "modèle de terrain (MNT)")),
        rvt_params={"svf": {"radius": 10, "num_directions": 16, "noise_remove": 0}},
        cv_runs=({"modele": "Modèle <A>", "target_rvt": "LD", "entites": ["Parcellaire"], "seuil": 0.26,
                  "images": 12, "secondes": 247},
                 {"modele": "B", "target_rvt": "LD", "entites": [], "seuil": None, "images": 0, "secondes": 0}),
        bilan=({"label": "Parcellaire", "total": 103,
                "effectifs": {"quasi_certain": 12, "probable": 30, "possible": 41, "douteux": 20}},),
        etapes=(("Téléchargement", "9 dalles", 12.0), ("Finalisation", "mosaïques, bilan, rapport", 1.4)),
        avertissements=(Avertissement("warn", "noyau <40> px", "Le noyau atteint N pixels", 3),),
        vignette="rapport_vignette.png",
    )
    h = construire_html(d)
    assert "Modèle &lt;A&gt;" in h and "noyau &lt;40&gt;" in h and "<A>" not in h
    for morceau in (
        "hypothèses", "Zone traitée", "9 km² couverts, en 9 dalles.", "ni nom de dalle ni coordonnée",
        "<h2>Produits</h2>", "rayon 10 px, 16 directions, sans suppression du bruit", "sans réglage",
        "Détection automatique", "<th>Seuil</th>", "0,26", ", sur LD", "12 images en 4min 07s (≈ 21s par image)", "reprise du run précédent",
        "Bilan de fiabilité", "Par km²", "11,4", "12 très probables, 30 probables, 41 possibles, 20 douteuses",
        "très probable : au moins 85 % de vrais objets", "douteux : moins de 35 %",
        "Durées par étape", "Téléchargement", "9 dalles", "1h 02min", "(×3)", "Le noyau atteint N pixels",
        "<h2>Sources</h2>", "Licence Ouverte 2.0", "Relief Visualization Toolbox", "Modèles de détection : Modèle &lt;A&gt;, B",
        'src="rapport_vignette.png"',
    ):
        assert morceau in h, morceau
    assert "None" not in h and "Dossier" not in h
    # Sans détection, en indices existants : réglages inconnus, pas de RVT dans les sources, garde courte.
    sans = construire_html(DonneesRapport(
        version="?", date="d", mode="Indices RVT existants", data_mode="existing_rvt", issue="cancelled",
        tiles_processed=1, produits=(("LD", "détection des dépressions locales"),),
    ))
    assert "annulé" in sans and "Pas de vignette" in sans and "Aucun avertissement" in sans and "1 dalle traitée." in sans
    assert "Pas de détection automatique" in sans and "Aucune détection avec fiabilité" in sans
    assert "leurs réglages ne sont pas connus" in sans and "Relief Visualization" not in sans
    assert "Durées par étape" not in sans and "ne localise pas la zone" in sans
    assert "indices de visualisation fournis" in sans and "Par km²" not in sans


def test_finalize_ecrit_le_rapport_sans_rien_situer(tmp_path, monkeypatch):
    monkeypatch.setattr(finalize_service, "_collect_vrt_paths_and_build", lambda *a, **k: [])
    monkeypatch.setattr(finalize_service, "_build_coverage_polygons", lambda *a, **k: None)
    raster = tmp_path / "indices" / "LHD_FXX_0988_6872_LD.tif"
    (tmp_path / "pipeline_log_20261008_104200.txt").write_text(
        "2026-10-08 10:42:00,000 - WARNING - Dalle 3 abandonnée après 2 tentatives\n"
        f"2026-10-08 10:42:01,000 - WARNING - Erreur rasterio pour {raster}: x\n", encoding="utf-8")
    cv_cfg = {"enabled": True, "runs": [{
        "selected_model": "M", "target_rvt": "LD", "confidence_threshold": 0.29,
        "fiabilite": {"modele": "Modèle test (LD)"},
        "entities": [{"id": "parcellaire", "slug": "parcellaire", "label": "Parcellaire"}],
    }]}
    r = _Reporter()
    ok = finalize_service.finalize_pipeline(
        output_dir=tmp_path, cv_cfg=cv_cfg,
        rvt_params={"ldo": {"min_radius": 10, "max_radius": 20, "angular_res": 15, "observer_h": 1.7}},
        reporter=r, slog=None, start_time=time.time() - 65, tiles_processed=2, tiles_total=3, active_products=["LD"],
        ui_config={"app": {"files": {"data_mode": "local_laz"}}}, outcome="success",
        cv_stats=[{"modele": "M", "model": "M", "target_rvt": "LD", "images": 4, "secondes": 9.0}],
    )
    assert ok is True
    h = (tmp_path / NOM_RAPPORT).read_text(encoding="utf-8")
    assert "Nuages locaux" in h and "2 dalles traitées, sur 3 prévues." in h
    assert "rayon de 10 à 20 px, résolution angulaire 15°, hauteur d'observateur 1,7 m" in h
    assert "Modèle test (LD), sur LD" in h and "0,29" in h and "4 images en 9s" in h and "Parcellaire" in h
    assert "Dalle 3 abandonnée" in h and "Une dalle est abandonnée" in h
    assert str(tmp_path) not in h and "0988" not in h and "LHD_FXX_…_LD.tif" in h
    assert "Durées par étape" in h and "Finalisation" in h          # marque posée par report_stage_id
    assert "nuages de points LiDAR fournis" in h and "Relief Visualization Toolbox" in h
    assert any(NOM_RAPPORT in m for m in r.users)
    meta = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert meta["rapport"] == NOM_RAPPORT


def test_la_vue_ouvre_le_rapport():
    racine = Path(__file__).resolve().parents[2]
    vue = (racine / "src/ui/run_view.py").read_text(encoding="utf-8")
    assert "NOM_RAPPORT" in vue and "_open_report" in vue
