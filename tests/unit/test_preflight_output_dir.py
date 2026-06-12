"""CFG-05 (audit v2) : le préflight affichait ✓ « (sera créé) » pour un
dossier de sortie sur un volume disparu (disque externe débranché, partage
réseau démonté — chemin conservé par last_ui_config), puis le lancement
échouait en WinError brut. Le préflight doit tester la créabilité réelle.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from pipeline.preflight import check_output_dir_creatable, collect_preflight_results


def _missing_drive() -> str:
    for letter in "ZYXWVUTSRQPONMLKJIHGFE":
        if not Path(f"{letter}:\\").exists():
            return f"{letter}:"
    pytest.skip("aucune lettre de lecteur libre sur cette machine")


class TestCheckOutputDirCreatable:
    def test_sous_dossier_inexistant_d_un_parent_existant_est_creable(self, tmp_path):
        ok, _ = check_output_dir_creatable(tmp_path / "nouveau" / "sous_dossier")
        assert ok is True

    @pytest.mark.skipif(os.name != "nt", reason="lecteurs Windows")
    def test_volume_disparu_est_refuse(self):
        drive = _missing_drive()
        ok, details = check_output_dir_creatable(Path(f"{drive}/archeo/output"))
        assert ok is False
        assert "inaccessible" in details


class TestPreflightOutputDir:
    @pytest.mark.skipif(os.name != "nt", reason="lecteurs Windows")
    def test_panneau_marque_le_volume_disparu_en_echec(self):
        drive = _missing_drive()
        results = collect_preflight_results(
            mode="existing_mnt",
            cv_config={},
            products={},
            files_config={"existing_mnt_dir": ""},
            output_dir=Path(f"{drive}/archeo/output"),
        )
        out = [r for r in results if r.name == "Dossier de sortie"]
        assert out and out[0].ok is False
        assert out[0].critical is True

    def test_dossier_creable_reste_ok(self, tmp_path):
        results = collect_preflight_results(
            mode="existing_mnt",
            cv_config={},
            products={},
            files_config={"existing_mnt_dir": ""},
            output_dir=tmp_path / "out",
        )
        out = [r for r in results if r.name == "Dossier de sortie"]
        assert out and out[0].ok is True
        assert "sera créé" in out[0].details
