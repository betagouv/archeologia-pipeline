"""Durées : une seule source et un même arrondi partout (crash tests du 2026-11-01). Le journal disait « Durée
totale : 24min 12s » quand l'écran et le rapport disaient 24min 05s (deux chronomètres), un modèle « 1min 25s »
à l'écran et « 1min 26s » au rapport (troncature contre arrondi), « ≈ 3s » contre « ≈ 2,6 s » par image."""
from __future__ import annotations

from unittest.mock import MagicMock

from app.services.rapport_run import DonneesRapport, construire_html, format_duree
from app.structured_logger import StructuredLogger
from app.user_narrator import _format_duration, create_user_narrator, format_par_image


def test_arrondi_et_format_partages():
    assert _format_duration(85.6) == "1min 26s" == format_duree(85.6)        # arrondi, pas troncature
    assert format_duree(120) == _format_duration(120)
    assert format_par_image(85.6, 33) == "2,6 s" and format_par_image(99.0, 33) == "3 s"
    assert format_par_image(247.0, 12) == "21s"


def test_ecran_et_rapport_disent_la_meme_chose_par_modele():
    reporter = MagicMock()
    create_user_narrator(reporter).cv_run_done("Modèle", 33, 85.6)
    ecran = reporter.user_info.call_args[0][0]
    assert ecran.endswith("33 images analysées en 1min 26s (≈ 2,6 s par image)")
    h = construire_html(DonneesRapport(version="x", date="d", mode="m", cv_runs=(
        {"modele": "Modèle", "target_rvt": "LD", "entites": [], "seuil": 0.3, "images": 33, "secondes": 85.6},)))
    assert "33 images en 1min 26s (≈ 2,6 s par image)" in h


def test_duree_totale_du_journal_sur_le_chronometre_du_run():
    logs: list = []
    slog = StructuredLogger(logs.append)
    slog._start_time = 0.0                       # chronomètre propre, démarré plus tôt
    slog.end_pipeline(success=True, duree_s=1445.0)
    assert any(ligne.strip() == "⏱️ Durée totale : 24min 05s" for ligne in logs)


def test_finalisation_une_seule_mesure(tmp_path, monkeypatch):
    """Journal, écran et rapport reçoivent la MÊME durée, mesurée une fois."""
    import ast
    from pathlib import Path
    src = Path(__file__).resolve().parents[2] / "src" / "app" / "services" / "finalize_service.py"
    arbre = ast.parse(src.read_text(encoding="utf-8"))
    fin = next(f for f in ast.walk(arbre) if isinstance(f, ast.FunctionDef) and f.name == "finalize_pipeline")
    mesures = [n for n in ast.walk(fin) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "elapsed" for t in n.targets)]
    assert len(mesures) == 1
    for appel in ("end_pipeline", "pipeline_complete", "pipeline_failed"):
        for n in ast.walk(fin):
            if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == appel:
                assert {k.arg for k in n.keywords} >= {"duree_s"}, appel


def test_rapport_nomme_la_trace_et_le_journal_du_lancement():
    """Le rapport citait « metadata.json », fichier qui n'existe pas (arborescence v3 : metadata_<ts>.json)."""
    h = construire_html(DonneesRapport(version="x", date="d", mode="m", horodatage="20261101_012122"))
    assert "metadata_20261101_012122.json" in h and "technique/journaux/" in h and "pipeline_log_20261101_012122.txt" in h
    assert "metadata.json" not in h
    sans = construire_html(DonneesRapport(version="x", date="d", mode="m"))
    assert "metadata_*.json" in sans and "metadata.json" not in sans
