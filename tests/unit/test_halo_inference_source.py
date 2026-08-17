"""Option B (halo inter-dalles) : résolution du TIF non rogné d'intermediaires/.

L'inférence CV doit tourner sur l'image AVEC marge (donnée voisine réelle)
plutôt que sur le TIF rogné 1 km, pour ne plus couper les détections aux
frontières de dalles. ``resolve_uncropped_tif`` associe un TIF rogné de
``indices/<X>/tif/`` à son TIF source non rogné dans ``intermediaires/`` :
même produit + mêmes paramètres RVT (queue de nom), et emprise couvrant la
cellule 1 km du TIF rogné.
"""
from __future__ import annotations

from pathlib import Path

from app.services.cv_post_service import resolve_uncropped_tif


# Queue de nom source pour LD aux paramètres par défaut (rvt_naming).
_LD_TAIL = "_LD_A15_Rmin10_Rmax20_H1p7_V1.tif"

# Cellule 872/6904 : x 872000..873000, y 6903000..6904000 (y = bord nord).
_CROPPED = "LHD_FXX_0872_6904_LD_A15_Rmin10_Rmax20_H1p7_V1_A_LAMB93.tif"

# Emprise non rognée correspondante avec 200 m de marge par côté.
_UNCROPPED_BOUNDS = (871800.0, 6902800.0, 873200.0, 6904200.0)


def _mk(temp_dir: Path, name: str) -> Path:
    p = temp_dir / name
    p.write_bytes(b"x")
    return p


class TestResolveUncroppedTif:
    def test_resout_le_tif_non_rogne_de_la_bonne_cellule(self, tmp_path):
        good = _mk(tmp_path, f"LHD_FXX_0872_6904_PTS_C_LAMB93_IGN69{_LD_TAIL}")
        neighbor = _mk(tmp_path, f"LHD_FXX_0873_6904_PTS_C_LAMB93_IGN69{_LD_TAIL}")
        bounds = {
            good.name: _UNCROPPED_BOUNDS,
            neighbor.name: (872800.0, 6902800.0, 874200.0, 6904200.0),
        }

        resolved = resolve_uncropped_tif(
            Path(_CROPPED), tmp_path, "LD", {},
            bounds_fn=lambda p: bounds.get(Path(p).name),
        )

        assert resolved == good

    def test_suffixe_de_parametres_different_ignore(self, tmp_path):
        # Même dalle mais autres paramètres RVT -> pas le bon dossier de run.
        _mk(tmp_path, "LHD_FXX_0872_6904_PTS_C_LAMB93_IGN69_LD_A20_Rmin5_Rmax15_H1p7_V1.tif")

        resolved = resolve_uncropped_tif(
            Path(_CROPPED), tmp_path, "LD", {},
            bounds_fn=lambda p: _UNCROPPED_BOUNDS,
        )

        assert resolved is None

    def test_candidat_ne_couvrant_pas_la_cellule_ignore(self, tmp_path):
        # Bon suffixe mais emprise d'une autre cellule (dalle voisine).
        _mk(tmp_path, f"LHD_FXX_0873_6904_PTS_C_LAMB93_IGN69{_LD_TAIL}")

        resolved = resolve_uncropped_tif(
            Path(_CROPPED), tmp_path, "LD", {},
            bounds_fn=lambda p: (872800.0, 6902800.0, 874200.0, 6904200.0),
        )

        assert resolved is None

    def test_emprise_illisible_ignore(self, tmp_path):
        _mk(tmp_path, f"LHD_FXX_0872_6904_PTS_C_LAMB93_IGN69{_LD_TAIL}")

        resolved = resolve_uncropped_tif(
            Path(_CROPPED), tmp_path, "LD", {},
            bounds_fn=lambda p: None,
        )

        assert resolved is None

    def test_nom_rogne_sans_coordonnees(self, tmp_path):
        _mk(tmp_path, f"LHD_FXX_0872_6904_PTS_C_LAMB93_IGN69{_LD_TAIL}")

        resolved = resolve_uncropped_tif(
            Path("mon_rvt.tif"), tmp_path, "LD", {},
            bounds_fn=lambda p: _UNCROPPED_BOUNDS,
        )

        assert resolved is None

    def test_temp_dir_inexistant(self, tmp_path):
        resolved = resolve_uncropped_tif(
            Path(_CROPPED), tmp_path / "absent", "LD", {},
            bounds_fn=lambda p: _UNCROPPED_BOUNDS,
        )

        assert resolved is None

    def test_temp_dir_none(self):
        resolved = resolve_uncropped_tif(
            Path(_CROPPED), None, "LD", {},
            bounds_fn=lambda p: _UNCROPPED_BOUNDS,
        )

        assert resolved is None
