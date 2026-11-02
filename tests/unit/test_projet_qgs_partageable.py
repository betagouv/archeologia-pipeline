"""``livrable/projet.qgs`` se transmet (relance de_sh, 2026-11-02) : il porte le titre de l'analyse et plus le nom de
l'utilisateur Windows — QGIS 3.44 et 4.0.3 écrivent ``saveUser``/``saveUserFull`` sur la racine et l'auteur des
métadonnées du projet, même mis à vide avant l'écriture (vérifié sous les deux) : retirés après l'écriture."""
from __future__ import annotations

from pathlib import Path

from app.services.finalize_service import anonymiser_projet_qgs

QGS = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis saveDateTime="2026-11-02T12:01:12" projectname="de_sh — 02/11/2026 12:01" saveUser="valen" saveUserFull="Valentin VIEUX" version="3.44.1-Solothurn">
  <title>de_sh — 02/11/2026 12:01</title>
  <maplayer><resourceMetadata><author>IGN</author></resourceMetadata></maplayer>
  <projectMetadata>
    <title>de_sh — 02/11/2026 12:01</title>
    <author>Valentin VIEUX</author>
    <creation>2026-11-02T12:01:11</creation>
  </projectMetadata>
</qgis>
"""


def test_nom_utilisateur_retire_du_qgs(tmp_path):
    qgs = tmp_path / "projet.qgs"
    qgs.write_bytes(QGS.replace("\n", "\r\n").encode("utf-8"))
    anonymiser_projet_qgs(qgs)
    texte = qgs.read_bytes().decode("utf-8")
    assert "valen" not in texte and "VIEUX" not in texte
    assert '<qgis saveDateTime="2026-11-02T12:01:12" projectname="de_sh — 02/11/2026 12:01" ' \
           'version="3.44.1-Solothurn">' in texte
    assert "<author></author>" in texte and "<author>IGN</author>" in texte    # seul l'auteur du projet
    assert texte.count("\r\n") == QGS.count("\n")                             # rien d'autre ne bouge
    anonymiser_projet_qgs(qgs)
    assert qgs.read_bytes().decode("utf-8") == texte                           # idempotent


def test_qgs_writer_titre_et_anonymise():
    code = (Path(__file__).resolve().parents[2] / "src" / "ui" / "qgs_writer.py").read_text(encoding="utf-8")
    assert "proj.setTitle(titre_analyse)" in code
    assert code.index("proj.write()") < code.index("anonymiser_projet_qgs(qgs_path)")
