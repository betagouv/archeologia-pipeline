"""Bilan de fiabilité de fin de run (2026-10-08) : lecture pure des sidecars, phrase,
metadata.json et narration dans la finalisation."""
from __future__ import annotations

import json
import time
from pathlib import Path

from app.progress_reporter import NullProgressReporter
from app.services import finalize_service
from app.services.bilan_fiabilite import (
    NIVEAUX_DU_PLUS_SUR,
    LigneBilan,
    collecter,
    effectifs_par_categorie,
)
from app.services.fiabilite import Categorie

CATS = [
    {"categorie": "douteux", "seuil": 0.29, "garanti": 0.0, "mesure": 0.26, "n": 100},
    {"categorie": "probable", "seuil": 0.5, "garanti": 0.6, "mesure": 0.72, "n": 100},
    {"categorie": "quasi_certain", "seuil": 0.65, "garanti": 0.85, "mesure": 0.92, "n": 100},
]


def _sidecar(det_dir: Path, slug: str, entrees: dict) -> None:
    d = det_dir / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "fiabilite.json").write_text(json.dumps(entrees, ensure_ascii=False), encoding="utf-8")


class _Reporter(NullProgressReporter):
    def __init__(self):
        self.users: list = []

    def user_info(self, msg: str) -> None:
        self.users.append(msg)

    def user_success(self, msg: str) -> None:
        self.users.append(msg)


def test_ordre_phrase_et_dict():
    cats = tuple(Categorie(c["categorie"], c["seuil"], c["garanti"], c["mesure"], c["n"]) for c in CATS)
    ligne = LigneBilan("parcellaire", "Parcellaire", "parcellaire", "parcellaire", "m", cats,
                       {"douteux": 20, "quasi_certain": 1, "probable": 30})
    assert NIVEAUX_DU_PLUS_SUR == ("quasi_certain", "probable", "possible", "douteux")
    assert ligne.par_niveau() == [("quasi_certain", 1), ("probable", 30), ("douteux", 20)]
    assert ligne.phrase() == "Parcellaire : 51 détections — 1 très probable, 30 probables, 20 douteuses"
    assert ligne.to_dict()["effectifs"] == {"quasi_certain": 1, "probable": 30, "douteux": 20}
    vide = LigneBilan("x", "Fours", "four", "four", "m", cats, {})
    assert vide.phrase() == "Fours : aucune détection" and vide.total == 0


def test_effectifs_par_categorie_depuis_les_libelles():
    assert effectifs_par_categorie({"effectifs": {"Très probable": 12, "probable": "3", "inconnu": 9}}) == {
        "quasi_certain": 12, "probable": 3,
    }
    assert effectifs_par_categorie({"categories": CATS}) is None      # sidecar d'un run ancien
    assert effectifs_par_categorie(None) is None


def test_collecter_dans_l_ordre_des_runs_puis_le_reste(tmp_path):
    det = tmp_path / "detections"
    _sidecar(det, "crateres", {"cratere": {"classe": "cratere", "modele": "M1", "categories": CATS,
                                           "effectifs": {"Très probable": 240, "Douteux": 60}}})
    _sidecar(det, "parcellaire", {"parcellaire": {"classe": "parcellaire", "modele": "M2", "categories": CATS,
                                                  "effectifs": {"Probable": 30}}})
    _sidecar(det, "ancien", {"four": {"classe": "four", "categories": CATS}})          # sans effectifs : ignoré
    _sidecar(det, "deux", {"a": {"classe": "a", "categories": CATS, "effectifs": {"Probable": 1}},
                           "b": {"classe": "b", "categories": CATS, "effectifs": {"Douteux": 2}}})
    runs = [{"entities": [{"slug": "parcellaire", "label": "Parcellaire"}, {"slug": "crateres", "label": "Cratères"}]}]
    lignes = collecter(det, runs)
    assert [(ligne.slug, ligne.label, ligne.total) for ligne in lignes] == [
        ("parcellaire", "Parcellaire", 30), ("crateres", "Cratères", 300),
        ("deux", "deux · a", 1), ("deux", "deux · b", 2),
    ]
    assert lignes[1].categories[0].categorie == "douteux" and lignes[1].modele == "M1"
    assert lignes[1].couche == "cratere"
    assert collecter(tmp_path / "absent", runs) == [] and collecter(det, None)[0].label == "crateres"


def test_finalize_ecrit_le_bilan_et_le_narre(tmp_path, monkeypatch):
    monkeypatch.setattr(finalize_service, "_collect_vrt_paths_and_build", lambda *a, **k: [])
    monkeypatch.setattr(finalize_service, "_build_coverage_polygons", lambda *a, **k: None)
    det = tmp_path / "livrable" / "detections"
    _sidecar(det, "parcellaire", {"parcellaire": {"classe": "parcellaire", "modele": "M", "categories": CATS,
                                                  "effectifs": {"Très probable": 2, "Douteux": 5}}})
    cv_cfg = {"enabled": True, "runs": [{"selected_model": "M", "target_rvt": "LD",
                                        "entities": [{"id": "parcellaire", "slug": "parcellaire", "label": "Parcellaire"}]}]}
    r = _Reporter()
    ok = finalize_service.finalize_pipeline(
        output_dir=tmp_path, cv_cfg=cv_cfg, rvt_params={}, reporter=r, slog=None,
        start_time=time.time(), tiles_processed=1, tiles_total=1, active_products=["LD"],
        ui_config={}, outcome="success",
    )
    assert ok is True
    meta = json.loads((tmp_path / "livrable" / "traitement.json").read_text(encoding="utf-8"))
    assert meta["bilan_fiabilite"] == [{
        "slug": "parcellaire", "label": "Parcellaire", "couche": "parcellaire", "classe": "parcellaire",
        "modele": "M", "total": 7, "effectifs": {"quasi_certain": 2, "douteux": 5},
    }]
    i = next(k for k, m in enumerate(r.users) if "Bilan de fiabilité" in m)
    assert r.users[i + 1] == "   • Parcellaire : 7 détections — 2 très probables, 5 douteuses"
    assert any("Traitement terminé" in m for m in r.users[i + 2:])      # le ✅ reste en dernier
    # sans CV : clé vide, pas de bloc narré
    r2 = _Reporter()
    out2 = tmp_path / "sans_cv"
    out2.mkdir()
    finalize_service.finalize_pipeline(
        output_dir=out2, cv_cfg={}, rvt_params={}, reporter=r2, slog=None,
        start_time=time.time(), tiles_processed=1, tiles_total=1, active_products=["MNT"],
        ui_config={}, outcome="success",
    )
    assert json.loads((out2 / "livrable" / "traitement.json").read_text(encoding="utf-8"))["bilan_fiabilite"] == []
    assert not any("Bilan de fiabilité" in m for m in r2.users)


def test_la_vue_et_le_widget_sont_branches():
    racine = Path(__file__).resolve().parents[2]
    vue = (racine / "src/ui/run_view.py").read_text(encoding="utf-8")
    assert "collecter_bilan(" in vue and "BilanFiabiliteWidget(" in vue and "_afficher_bilan" in vue
    widget = (racine / "src/ui/widgets/bilan_fiabilite.py").read_text(encoding="utf-8")
    assert "teinte_niveau" in widget and "couleur_de_classe" in widget   # mêmes teintes que la légende
