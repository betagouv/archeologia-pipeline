"""Halo inter-dalles fabriqué depuis les dalles voisines (modes sans intermediaires/).

En ``existing_rvt`` / ``existing_mnt`` chaque dalle 1 km était inférée seule :
un objet à cheval sur une frontière sortait coupé au bord (ou pas du tout).
``NeighborHalo`` découpe dalle + marge dans la mosaïque des dalles fournies,
avec une fraîcheur par jeu de voisins (sidecar) + mtime, comme le LAZ fusionné.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import pipeline.modes.neighbor_halo as nh
from pipeline.modes.neighbor_halo import DEFAULT_HALO_MARGIN_M, NeighborHalo, halo_inputs, halo_window


def _cell(x_km: int, y_km: int):
    """Emprise (xmin, ymin, xmax, ymax) de la dalle IGN dont ``y_km`` est le bord nord."""
    return (x_km * 1000.0, (y_km - 1) * 1000.0, (x_km + 1) * 1000.0, y_km * 1000.0)


def _tiles(tmp_path, cells):
    tiles = {}
    for x, y in cells:
        p = tmp_path / f"LHD_FXX_{x:04d}_{y:04d}_LD_A_LAMB93.tif"
        p.write_bytes(b"tif")
        tiles[p] = _cell(x, y)
    return tiles


class TestHaloInputs:
    def test_fenetre_etendue(self):
        assert halo_window(_cell(872, 6904), 50) == (871950.0, 6902950.0, 873050.0, 6904050.0)

    def test_voisins_intersectant_la_fenetre(self, tmp_path):
        tiles = _tiles(tmp_path, [(872, 6904), (873, 6904), (874, 6904)])
        a, b, c = sorted(tiles)
        assert halo_inputs(b, tiles, 50) == [a, b, c]
        assert halo_inputs(a, tiles, 50) == [a, b]  # c est à 1 km : hors fenêtre

    def test_voisin_diagonal_inclus(self, tmp_path):
        tiles = _tiles(tmp_path, [(872, 6904), (873, 6904), (872, 6905), (873, 6905)])
        sw = next(p for p in tiles if "0872_6904" in p.name)
        assert len(halo_inputs(sw, tiles, 50)) == 4

    def test_dalle_isolee(self, tmp_path):
        tiles = _tiles(tmp_path, [(872, 6904), (880, 6904)])
        a = next(p for p in tiles if "0872" in p.name)
        assert halo_inputs(a, tiles, 50) == [a]


@pytest.fixture
def halo_env(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    tiles = _tiles(tmp_path / "src", [(872, 6904), (873, 6904)])
    calls = []

    def fake_extract(inputs, window, dst):
        calls.append((list(inputs), window, Path(dst)))
        Path(dst).write_bytes(b"halo")

    monkeypatch.setattr(nh, "_gdal_extract", fake_extract)
    logs = []
    halo = NeighborHalo(tiles, tmp_path / "halo", 50, log=logs.append)
    return {"tiles": tiles, "halo": halo, "calls": calls, "logs": logs, "tmp": tmp_path}


class TestNeighborHaloResolve:
    def test_fabrique_dalle_plus_marge_depuis_les_voisins(self, halo_env):
        a, b = sorted(halo_env["tiles"])
        out = halo_env["halo"].resolve(a)

        assert out == halo_env["tmp"] / "halo" / a.name and out.exists()
        inputs, window, dst = halo_env["calls"][0]
        assert inputs == [a, b]
        assert window == halo_window(_cell(872, 6904), 50)
        sidecar = json.loads(out.with_suffix(".inputs.json").read_text(encoding="utf-8"))
        assert sidecar == [a.name, b.name]

    def test_reutilise_le_halo_a_jour(self, halo_env):
        a, _b = sorted(halo_env["tiles"])
        first = halo_env["halo"].resolve(a)
        again = halo_env["halo"].resolve(a)

        assert again == first
        assert len(halo_env["calls"]) == 1
        assert halo_env["halo"].built == 1 and halo_env["halo"].reused == 1

    def test_dalle_sans_voisin_pas_de_halo(self, tmp_path, monkeypatch):
        tiles = _tiles(tmp_path, [(872, 6904)])
        monkeypatch.setattr(nh, "_gdal_extract", lambda *a: pytest.fail("extraction inattendue"))
        halo = NeighborHalo(tiles, tmp_path / "halo", 50)

        assert halo.resolve(next(iter(tiles))) is None
        assert halo.skipped == 1

    def test_jeu_de_voisins_modifie_refabrique(self, halo_env):
        # Extension de zone {A,B} → {A,B,C} : la marge nord de A était fabriquée
        # (aplat), elle doit être re-découpée avec la vraie donnée de C.
        a, _b = sorted(halo_env["tiles"])
        halo_env["halo"].resolve(a)
        tiles = dict(halo_env["tiles"])
        tiles.update(_tiles(halo_env["tmp"] / "src", [(872, 6905)]))
        halo2 = NeighborHalo(tiles, halo_env["tmp"] / "halo", 50)

        halo2.resolve(a)

        assert len(halo_env["calls"]) == 2
        assert len(halo_env["calls"][1][0]) == 3

    def test_entree_plus_recente_refabrique(self, halo_env):
        a, b = sorted(halo_env["tiles"])
        out = halo_env["halo"].resolve(a)
        t = out.stat().st_mtime
        os.utime(b, (t + 10, t + 10))  # voisin remplacé après la fabrication

        halo_env["halo"].resolve(a)

        assert len(halo_env["calls"]) == 2

    def test_echec_extraction_retombe_sur_none(self, halo_env, monkeypatch):
        def boom(*_a):
            raise RuntimeError("gdal KO")

        monkeypatch.setattr(nh, "_gdal_extract", boom)
        a, _b = sorted(halo_env["tiles"])

        assert halo_env["halo"].resolve(a) is None
        assert any("gdal KO" in m for m in halo_env["logs"])

    def test_dalle_inconnue(self, halo_env):
        assert halo_env["halo"].resolve(Path("ailleurs.tif")) is None


class TestDefaultMargin:
    def test_marge_gratuite_pour_la_grille_sahi(self):
        # 2000 px + 2 × marge (0,5 m/px) doit rester sous 648 + 3 × 518 = 2202 px :
        # la grille SAHI 4×4 des modèles 648/672 px n'ajoute aucune tuile.
        assert 2000 + 2 * DEFAULT_HALO_MARGIN_M / 0.5 <= 2202
