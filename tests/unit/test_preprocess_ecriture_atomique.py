"""Écriture atomique des LAZ intermédiaires (incident 2026-09-19).

Trois runs consécutifs ont calé 8 à 10 h sur la dalle LHD_FXX_0821_6327 :
son ``_merged.laz`` avait sa taille complète mais 161 Mio de zéros en queue.
Origine : un crash PDAL (0xC0000374) avait fait tomber ``merge_tiles`` dans
la branche « aucun voisin valide » (``shutil.copy2`` vers le chemin FINAL) et
l'annulation 4 s plus tard a tué le processus en pleine copie. L'en-tête
restant intact, ``validate_las_or_laz_with_pdal`` (= ``pdal info --metadata``)
le déclarait valide à chaque run suivant, et PDAL partait en boucle infinie
dès qu'il lisait la zone de zéros.

Règle : tant que l'écriture n'a pas réussi, RIEN ne doit porter le nom final.
Une écriture tuée laisse au pire un ``.partial`` que personne ne réutilise.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from pipeline.ign import preprocess


def _laz(path: Path, content: bytes = b"LASF-ok") -> Path:
    path.write_bytes(content)
    return path


@pytest.fixture
def pdal_dit_oui(monkeypatch):
    """Toute validation PDAL répond OK (on teste l'écriture, pas PDAL)."""
    monkeypatch.setattr(
        preprocess, "validate_las_or_laz_with_pdal", lambda *a, **k: (True, "ok")
    )


def test_fusion_pdal_echouee_ne_laisse_pas_de_laz_final(tmp_path, pdal_dit_oui, monkeypatch):
    central = _laz(tmp_path / "central.laz")
    voisin = _laz(tmp_path / "voisin_1.laz")
    sortie = tmp_path / "dalle_merged.laz"

    def pdal_ecrit_puis_plante(cmd, **kwargs):
        # pdal merge écrit son fichier de sortie, puis le processus meurt.
        Path(cmd[-1]).write_bytes(b"LASF" + b"\x00" * 64)
        return subprocess.CompletedProcess(cmd, 3221225794, "", "heap corruption")

    monkeypatch.setattr(preprocess, "run_pdal_command_cancellable", pdal_ecrit_puis_plante)

    assert preprocess.merge_tiles(
        central_path=central, neighbor_paths=[voisin], output_path=sortie
    ) is False
    assert not sortie.exists(), "un LAZ fusionné corrompu porte le nom final"
    assert not preprocess.merged_inputs_sidecar(sortie).exists()


def test_copie_interrompue_ne_laisse_pas_de_laz_final(tmp_path, pdal_dit_oui, monkeypatch):
    # Branche « aucun voisin valide » : c'est celle de l'incident.
    central = _laz(tmp_path / "central.laz")
    sortie = tmp_path / "dalle_merged.laz"

    def copie_tuee_en_vol(src, dst, *a, **k):
        Path(dst).write_bytes(b"LASF" + b"\x00" * 64)
        raise OSError("processus tué pendant la copie")

    monkeypatch.setattr(preprocess.shutil, "copy2", copie_tuee_en_vol)

    with pytest.raises(OSError):
        preprocess.merge_tiles(central_path=central, neighbor_paths=[], output_path=sortie)
    assert not sortie.exists(), "une copie interrompue porte le nom final"
    assert not preprocess.merged_inputs_sidecar(sortie).exists()


def test_rognage_pdal_echoue_ne_laisse_pas_de_laz_final(tmp_path, pdal_dit_oui, monkeypatch):
    entree = _laz(tmp_path / "voisine.laz")
    sortie = tmp_path / "voisine_crop.laz"

    def pdal_ecrit_puis_plante(cmd, **kwargs):
        # `pdal pipeline <json>` : la destination réelle est dans le JSON.
        etapes = json.loads(Path(cmd[-1]).read_text(encoding="utf-8"))["pipeline"]
        dest = next(e["filename"] for e in etapes if e["type"] == "writers.las")
        Path(dest).write_bytes(b"LASF" + b"\x00" * 64)
        return subprocess.CompletedProcess(cmd, 3221225794, "", "heap corruption")

    monkeypatch.setattr(preprocess, "run_pdal_command_cancellable", pdal_ecrit_puis_plante)
    monkeypatch.setattr(preprocess, "_pdal_exe", lambda: "pdal")

    assert preprocess.crop_neighbor_tile(
        input_path=entree,
        output_path=sortie,
        bounds={"xmin": "0", "xmax": "1", "ymin": "0", "ymax": "1"},
    ) is False
    assert not sortie.exists(), "un LAZ rogné corrompu porte le nom final"


def test_le_fichier_partiel_garde_une_extension_que_pdal_sait_lire(tmp_path, monkeypatch):
    """Régression 2026-09-22 : PDAL choisit son lecteur (et son écrivain) sur
    l'extension. Nommé ``X.laz.partial``, le rognage était validé sur un nom
    que ``pdal info`` refuse (« Cannot determine reader »), donc jeté : plus
    aucun voisin fusionné, MNT sans marge, couture visible entre dalles."""
    entree = _laz(tmp_path / "voisine.laz")
    sortie = tmp_path / "voisine_crop.laz"
    valides: list[Path] = []

    def pdal_ecrit(cmd, **kwargs):
        etapes = json.loads(Path(cmd[-1]).read_text(encoding="utf-8"))["pipeline"]
        dest = next(e["filename"] for e in etapes if e["type"] == "writers.las")
        Path(dest).write_bytes(b"LASF-ok")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def pdal_info(path, *a, **k):
        valides.append(Path(path))
        return (True, "ok")

    monkeypatch.setattr(preprocess, "run_pdal_command_cancellable", pdal_ecrit)
    monkeypatch.setattr(preprocess, "validate_las_or_laz_with_pdal", pdal_info)
    monkeypatch.setattr(preprocess, "_pdal_exe", lambda: "pdal")

    assert preprocess.crop_neighbor_tile(
        input_path=entree,
        output_path=sortie,
        bounds={"xmin": "0", "xmax": "1", "ymin": "0", "ymax": "1"},
    ) is True
    assert valides, "le rognage doit être validé avant de prendre son nom final"
    assert all(p.suffix == ".laz" for p in valides), [p.name for p in valides]
    assert sortie.exists()
