"""Reconstruction ``(nom_pkk, url_telech)`` depuis ``url_npl`` du WFS IGN (pur, sans réseau)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "dev" / "build_quadrillage_from_wfs.py"
_spec = importlib.util.spec_from_file_location("build_quadrillage_from_wfs", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

_LOT = "https://data.geopf.fr/telechargement/download/LiDARHD-NUALID/NUALHD_1-0__LAZ_LAMB93_OK_2025-05-22/"


def test_nom_pkk_est_le_nom_de_fichier_sans_extension():
    nom, url = _mod.tile_record(_LOT + "LHD_FXX_0836_6835_PTS_LAMB93_IGN69.copc.laz")
    assert nom == "LHD_FXX_0836_6835_PTS_LAMB93_IGN69"
    assert url == _LOT + "LHD_FXX_0836_6835_PTS_LAMB93_IGN69.copc.laz"


def test_http_force_en_https():
    _, url = _mod.tile_record("http://data.geopf.fr/x/LHD_FXX_0371_6389_PTS_LAMB93_IGN69.copc.laz")
    assert url.startswith("https://data.geopf.fr/")


def test_tiret_du_wfs_corrige_en_underscore():
    # Le WFS publie ~160 noms « LHD_FXX_0881-6546_… » : cette URL répond 404, la forme
    # à underscore répond 200 (vérifié le 2026-10-07). Le lot (avec ses tirets de date)
    # ne doit pas être touché.
    nom, url = _mod.tile_record(_LOT + "LHD_FXX_0881-6546_PTS_LAMB93_IGN69.copc.laz")
    assert nom == "LHD_FXX_0881_6546_PTS_LAMB93_IGN69"
    assert url == _LOT + "LHD_FXX_0881_6546_PTS_LAMB93_IGN69.copc.laz"


def test_dom_conserve_son_nom():
    nom, _ = _mod.tile_record(
        "https://d/NUALHD_1-0__LAZ_RGR92UTM40S_REU_2025-06-18/LHD_REU_0357_7687_PTS_RGR92UTM40S_REUN89.copc.laz"
    )
    assert nom == "LHD_REU_0357_7687_PTS_RGR92UTM40S_REUN89"
