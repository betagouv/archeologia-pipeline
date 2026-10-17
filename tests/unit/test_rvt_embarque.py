"""Bibliothèque rvt embarquée sous un nom privé (vague 2, lot 4, 0.15.0).

Le plugin calcule ses indices avec une copie figée de la bibliothèque rvt de rvt-qgis 1.0.1
(``src/pipeline/ign/products/rvt_embarque``), imports internes rendus relatifs : elle ne lit ni n'écrit
``sys.modules["rvt"]``, que l'extension rvt-qgis (ou un autre plugin) occupe peut-être avec SA version
(Review Focus 3). Copie figée : ses empreintes sont fixées ici. La mettre à jour = refaire l'embarquement
(NOTICE), changer ces empreintes, puis prouver l'identité au banc N2.
"""
from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[2]
SRC = RACINE / "src"
EMBARQUE = SRC / "pipeline" / "ign" / "products" / "rvt_embarque"

#: SHA-256 des fichiers embarqués, fins de ligne ramenées à LF (git les rend selon core.autocrlf).
EMPREINTES = {
    "__init__.py": "99caa03d87eb9c3b798fef3b9f35959a5be256449ce7dc952d3ae08003cc6894",
    "blend.py": "10c57d18fc277eb06f0f4f274681ae8cbca7265ce4243c04e3af47e96a2b489b",
    "blend_func.py": "098451e7923ea955cd5a0ba6a2dfe347be20d06dc0291ebdb4a67599f62ee5be",
    "default.py": "a232912337287fdd2f308d408be2c335ee9da2a8c2e4b37e355a013f7fba8aa0",
    "tile.py": "eee386e989a0318372ce4112b91946dfa489261566d46ee8952dcf0dab8ebef5",
    "vis.py": "d6ab82d6afc9c9fa691e4525374e530586bbd8249ea4f9f8a3cf57780ffb85d6",
    "settings/blender_VAT.json": "6eb0259d2a6e243100a3a0d64f08f3f39515487a5cbabd7560af080bb9e90dfd",
    "settings/default_blender_combinations.json": "86399b11f757ef85bfd9a4190330797553cdce697de2bcbe65a291b22cdab366",
    "settings/default_terrains_settings.json": "51aedc90860164584fe06e0af714c5f75dfcf945c4e7e01f80d249c6fda3301e",
}


def test_copie_figee():
    lues = {p.relative_to(EMBARQUE).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in EMBARQUE.rglob("*") if p.suffix in (".py", ".json") and "__pycache__" not in p.parts}
    assert lues == EMPREINTES


def test_licence_notice_et_entetes():
    assert "Apache License" in (EMBARQUE / "LICENSE").read_text(encoding="utf-8")
    notice = (EMBARQUE / "NOTICE").read_text(encoding="utf-8")
    for mot in ("ZRC SAZU", "Ljubljana", "Apache", "rvt-qgis", "1.0.1", "import rvt"):
        assert mot in notice, mot
    for nom in ("blend.py", "default.py", "tile.py"):
        assert (EMBARQUE / nom).read_text(encoding="utf-8").startswith("# Modifié pour Archéolog'IA"), nom


def _imports_globaux_de_rvt(chemin: Path):
    for n in ast.walk(ast.parse(chemin.read_bytes())):
        if isinstance(n, ast.Import):
            noms = [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.level == 0:
            noms = [n.module or ""]
        else:
            continue
        for nom in noms:
            if nom == "rvt" or nom.startswith("rvt."):
                yield f"{chemin.relative_to(RACINE).as_posix()}:{n.lineno}"


def test_aucun_import_global_de_rvt_dans_src():
    sources = [p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts]
    imports = [f for p in sources for f in _imports_globaux_de_rvt(p)]
    textes = [p.relative_to(RACINE).as_posix() for p in sources
              if any(m in p.read_text(encoding="utf-8", errors="replace") for m in ('modules["rvt', "modules['rvt"))]
    assert imports == [] and textes == []


_CODE = r'''
import importlib.util, sys, types
src, autre, ordre = sys.argv[1:4]
sys.path[:0] = [src, autre]
if importlib.util.find_spec("osgeo") is None:   # Python sans GDAL : bouchon le temps de l'import
    class _B(types.ModuleType):
        def __getattr__(self, nom):
            return type(nom, (), {})
    sys.modules["osgeo"] = types.ModuleType("osgeo")
    sys.modules["osgeo"].gdal = sys.modules["osgeo.gdal"] = _B("osgeo.gdal")
if ordre == "autre_d_abord":
    import rvt.vis
from pipeline.ign.products.rvt_embarque import blend, default, vis
import rvt
import rvt.vis as autre_vis
assert rvt.ORIGINE == "autre" and autre_vis.ORIGINE == "autre", "le rvt de l'autre extension a été remplacé"
assert blend.rvt is sys.modules["pipeline.ign.products.rvt_embarque"], blend.rvt
assert blend.rvt.vis is vis and default.rvt.vis is vis and not hasattr(vis, "ORIGINE")
print("ok")
'''


@pytest.mark.parametrize("ordre", ["autre_d_abord", "embarque_d_abord"])
def test_coexiste_avec_le_rvt_d_une_autre_extension(tmp_path, ordre):
    autre = tmp_path / "rvt-qgis" / "rvt"
    autre.mkdir(parents=True)
    for nom in ("__init__.py", "vis.py"):
        (autre / nom).write_text("ORIGINE = 'autre'\n", encoding="utf-8")
    r = subprocess.run([sys.executable, "-c", _CODE, str(SRC), str(autre.parent), ordre],
                       capture_output=True, text=True, timeout=180)
    assert r.stdout.strip().endswith("ok"), r.stdout + r.stderr


def test_reglages_embarques_lus_par_cvat_et_crim():
    from pipeline.ign.products.moteur_indices import chemin_reglages
    for nom in ("blender_VAT.json", "default_blender_combinations.json", "default_terrains_settings.json"):
        assert chemin_reglages(nom).is_file(), nom


def test_interrupteur(monkeypatch):
    from pipeline.ign.products import moteur_indices as m
    monkeypatch.delenv(m.VARIABLE_MOTEUR, raising=False)
    assert m.par_le_moteur("CVAT") and m.par_le_moteur("CRIM")
    monkeypatch.setenv(m.VARIABLE_MOTEUR, "processing")
    assert m.par_le_moteur("CVAT") and m.par_le_moteur("CRIM")   # aucun algorithme Processing pour eux
