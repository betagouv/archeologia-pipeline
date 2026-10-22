"""ROB-17 (audit v2) : annuler pendant le téléchargement IGN doit se terminer
par l'exception canonique PipelineCancelled (→ « ⏹ Traitement annulé » en UI),
pas par un ValueError « Impossible de déterminer les coordonnées des dalles »
(le tri post-download court-circuité par cancel renvoie une liste vide).
"""
from __future__ import annotations

import pytest

from pipeline.cancellation import PipelineCancelled
import pipeline.ign.downloader as dl


def test_annulation_pendant_download_leve_pipeline_cancelled(tmp_path, monkeypatch):
    input_file = tmp_path / "dalles.txt"
    input_file.write_text(
        "https://exemple.invalid/dalle_0500_6500.laz\n", encoding="utf-8"
    )

    # Pas de réseau ni de QGIS dans ce test : parsing, proxy et worker stubés.
    monkeypatch.setattr(
        dl,
        "parse_ign_input_file",
        lambda *a, **k: [
            ("dalle_0500_6500.laz", "https://exemple.invalid/dalle_0500_6500.laz")
        ],
    )
    monkeypatch.setattr(dl, "_get_proxy_config", lambda **k: {})
    monkeypatch.setattr(
        dl,
        "_download_task_worker",
        lambda task, *a, **k: dl._DownloadResult(
            index=task.index,
            filename=task.filename,
            success=False,
            skipped=False,
            error="annulé",
        ),
    )

    with pytest.raises(PipelineCancelled):
        dl.download_ign_dalles(
            input_file=input_file,
            output_dir=tmp_path / "out",
            cancel=lambda: True,
        )


def test_annuler_n_attend_pas_un_telechargement_bloque(tmp_path, monkeypatch):
    """Recette séance B (2026-10-21) : annulé pendant que le serveur de l'IGN ne répondait plus, le
    traitement a mis 40 s à rendre la main — la boucle n'examinait l'annulation qu'à la fin d'une dalle,
    puis attendait les téléchargements en cours, bloqués dans une lecture qu'aucune coupure ne réveille
    sous Windows. L'annulation doit rendre la main sans attendre le fil bloqué."""
    import threading
    import time

    input_file = tmp_path / "dalles.txt"
    input_file.write_text("https://exemple.invalid/dalle_0500_6500.laz\n", encoding="utf-8")
    monkeypatch.setattr(
        dl, "parse_ign_input_file",
        lambda *a, **k: [("dalle_0500_6500.laz", "https://exemple.invalid/dalle_0500_6500.laz")],
    )
    monkeypatch.setattr(dl, "_get_proxy_config", lambda **k: {})
    debloquer = threading.Event()

    def bloque(task, *a, **k):          # une lecture réseau qui ne se réveille pas
        debloquer.wait(30)
        return dl._DownloadResult(index=task.index, filename=task.filename, success=False, skipped=False)

    monkeypatch.setattr(dl, "_download_task_worker", bloque)
    debut = time.monotonic()
    try:
        with pytest.raises(PipelineCancelled):
            dl.download_ign_dalles(
                input_file=input_file, output_dir=tmp_path / "out",
                cancel=lambda: time.monotonic() - debut > 0.3,
            )
        assert time.monotonic() - debut < 3.0
    finally:
        debloquer.set()


def test_un_telechargement_abandonne_ne_touche_pas_au_fichier_d_un_nouveau_lancement(tmp_path):
    """Le fil abandonné par une annulation se réveille plus tard (données ou délai de lecture) : si un
    nouveau lancement a repris la même dalle entre-temps, il ne doit pas supprimer son ``.partial``."""
    partiel = tmp_path / "dalle.laz.partial"
    ancien = dl._prendre_partiel(partiel)
    nouveau = dl._prendre_partiel(partiel)
    partiel.write_bytes(b"en cours")
    dl._liberer_partiel(partiel, ancien)
    assert partiel.exists(), "le fil abandonné a supprimé le fichier du nouveau lancement"
    dl._liberer_partiel(partiel, nouveau)
    assert not partiel.exists()
