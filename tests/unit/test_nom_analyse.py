"""Nom de l'analyse (0.19.0) : champ de l'étape 1, groupe parent des couches dans QGIS,
titre du rapport, champ ``nom_analyse`` de la trace.

L'arbre des couches est simulé (même interface que ``QgsLayerTreeGroup`` pour ce que le plugin
en appelle) : ``app.services.nom_analyse`` place les couches, ``ui/layer_loader`` et
``ui/qgs_writer`` ne font que lui passer la racine de leur projet — même arbre par construction.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from app.services import nom_analyse as na
from app.services.rapport_run import DonneesRapport, construire_html

SORTIE = r"C:\Users\x\_crash_tests\be_vl\indices"


# ── nom proposé, suivi du dossier ───────────────────────────────────
def test_nom_par_defaut_parent_et_dossier():
    assert na.nom_par_defaut(SORTIE) == "be_vl · indices"
    assert na.nom_par_defaut(SORTIE + "\\") == "be_vl · indices"
    assert na.nom_par_defaut("/home/u/sorties/zone_a/") == "sorties · zone_a"
    assert na.nom_par_defaut(r"D:\sortie") == "sortie"      # le lecteur n'est pas un dossier parent
    assert na.nom_par_defaut("") == ""


def test_nom_suit_le_dossier_tant_qu_il_n_est_pas_modifie():
    assert na.est_automatique("", SORTIE)
    assert na.est_automatique("be_vl · indices", SORTIE)
    assert na.est_automatique("  be_vl · indices ", SORTIE)
    assert not na.est_automatique("Prospection Vlaanderen", SORTIE)
    # Saisi à la main, puis le dossier change : le nom reste celui de l'utilisateur.
    assert not na.est_automatique("Prospection Vlaanderen", r"C:\autre\dossier")


def test_nom_de_config_et_ancienne_config():
    cfg = {"app": {"files": {"output_dir": SORTIE, "nom_analyse": "Ma prospection"}}}
    assert na.nom_de_config(cfg) == "Ma prospection"
    ancienne = {"app": {"files": {"output_dir": SORTIE}}}          # avant 0.19.0 : pas de champ
    assert na.nom_de_config(ancienne) == "be_vl · indices"
    vide = {"app": {"files": {"output_dir": SORTIE, "nom_analyse": "   "}}}
    assert na.nom_de_config(vide) == "be_vl · indices"
    assert na.nom_de_config(None) == ""


def test_config_par_defaut_porte_le_champ_et_garde_la_valeur(tmp_path):
    from config.config_manager import ConfigManager

    cm = ConfigManager(tmp_path)
    assert cm.default_config()["app"]["files"]["nom_analyse"] == ""
    garde = cm.fusionner({"app": {"files": {"output_dir": SORTIE, "nom_analyse": "Ma prospection"}}})
    assert garde["app"]["files"]["nom_analyse"] == "Ma prospection"
    ancienne = cm.fusionner({"app": {"files": {"output_dir": SORTIE}}})
    assert na.nom_de_config(ancienne) == "be_vl · indices"


def test_titre_du_groupe_avec_date_et_heure():
    assert na.titre_groupe("be_vl · indices", "2026-11-01T14:05:09") == "be_vl · indices — 01/11/2026 14:05"
    assert na.titre_groupe("be_vl · indices", "") == "be_vl · indices"
    assert na.titre_groupe("", "2026-11-01T14:05:09") == "Analyse — 01/11/2026 14:05"


def test_titre_lu_dans_la_trace_du_livrable(tmp_path):
    out = tmp_path / "be_vl" / "indices"
    (out / "livrable").mkdir(parents=True)
    trace = out / "livrable" / "traitement.json"
    trace.write_text(json.dumps({"date": "2026-11-01T14:05:09", "nom_analyse": "Flandre"}), encoding="utf-8")
    assert na.titre_du_livrable(out) == "Flandre — 01/11/2026 14:05"
    trace.write_text(json.dumps({"date": "2026-11-01T14:05:09"}), encoding="utf-8")   # trace antérieure
    assert na.titre_du_livrable(out) == "be_vl · indices — 01/11/2026 14:05"
    assert na.titre_du_livrable(tmp_path / "absent") == f"{tmp_path.name} · absent"   # sans trace : sans date


# ── arbre des couches simulé ────────────────────────────────────────
class _Couche:
    def __init__(self, nom):
        self.nom = nom


class _Noeud:
    def __init__(self, nom, couche=None):
        self._nom, self.couche, self._enfants, self._props = nom, couche, [], {}

    def name(self):
        return self._nom

    def setName(self, n):
        self._nom = n

    def children(self):
        return list(self._enfants)

    def customProperty(self, cle, defaut=None):
        return self._props.get(cle, defaut)

    def setCustomProperty(self, cle, valeur):
        self._props[cle] = valeur

    def findGroup(self, nom):
        for e in self._enfants:
            if e.couche is None:
                if e.name() == nom:
                    return e
                trouve = e.findGroup(nom)
                if trouve is not None:
                    return trouve
        return None

    def insertGroup(self, i, nom):
        g = _Noeud(nom)
        self._enfants.insert(i, g)
        return g

    def addGroup(self, nom):
        return self.insertGroup(len(self._enfants), nom)

    def addLayer(self, couche):
        self._enfants.append(_Noeud(couche.nom, couche))

    def insertLayer(self, i, couche):
        self._enfants.insert(i, _Noeud(couche.nom, couche))

    def removeChildNode(self, n):
        self._enfants.remove(n)

    def arbre(self):
        return [(e.name(), e.arbre()) if e.couche is None else e.name() for e in self._enfants]


def _est_groupe(n):
    return n.couche is None


def _charger(root, titre, *, mnt=True, derive=False):
    """Ce que font load_result_layers et write_validation_project, dans le même ordre."""
    parent = na.groupe_analyse(root, titre)
    if mnt:
        na.groupe_indices(parent).addLayer(_Couche("MNT · Relief nu"))
    na.groupe_structures(parent).addLayer(_Couche("Cratères"))
    if derive:
        na.groupe_structures(parent, "Regroupement de cratères").addLayer(_Couche("Zones"))
    parent.insertLayer(0, _Couche("Zones mal couvertes (<30 %)"))
    return parent


def test_arbre_groupe_parent_structures_indices_qa():
    root = _Noeud("")
    root.addLayer(_Couche("OpenStreetMap"))                 # couche de l'utilisateur
    _charger(root, "be_vl · indices — 01/11/2026 14:05", derive=True)
    assert root.arbre() == [
        ("be_vl · indices — 01/11/2026 14:05", [
            "Zones mal couvertes (<30 %)",
            ("Structures", [("Regroupement de cratères", ["Zones"]), "Cratères"]),
            ("Indices", ["MNT · Relief nu"]),
        ]),
        "OpenStreetMap",
    ]


def test_deux_analyses_deux_groupes_la_derniere_en_haut():
    root = _Noeud("")
    _charger(root, "A — 01/11/2026 14:05")
    _charger(root, "B — 01/11/2026 15:30")
    assert [n for n, _ in root.arbre()] == ["B — 01/11/2026 15:30", "A — 01/11/2026 14:05"]


def test_recharger_la_meme_analyse_reutilise_son_groupe_meme_renomme():
    root = _Noeud("")
    g = na.groupe_analyse(root, "A — 01/11/2026 14:05")
    assert na.groupe_analyse(root, "A — 01/11/2026 14:05") is g
    g.setName("Mon groupe")                                  # renommé par l'utilisateur
    assert na.groupe_analyse(root, "A — 01/11/2026 14:05") is g
    assert len(root.children()) == 1
    assert na.groupe_indices(g) is na.groupe_indices(g)
    assert na.groupe_structures(g, "R") is na.groupe_structures(g, "R")


def test_sans_titre_l_arbre_reste_a_plat():
    root = _Noeud("")
    assert na.groupe_analyse(root, "") is root
    assert na.groupe_indices(root) is root      # rasters à la racine, comme avant 0.19.0


def test_purge_retire_les_groupes_du_plugin_vides_jamais_ceux_de_l_utilisateur():
    root = _Noeud("")
    vieux = _charger(root, "A — 01/11/2026 14:05", derive=True)
    root.addGroup("Mon groupe vide")                         # groupe de l'utilisateur, vide
    root.addGroup("A — 01/11/2026 14:05")                    # même nom, mais pas du plugin
    garde = _charger(root, "B — 01/11/2026 15:30")
    # La purge (par chemin) a retiré toutes les couches de A : ne restent que ses groupes.
    for n in vieux.children():
        if not _est_groupe(n):
            vieux.removeChildNode(n)
    for g in (na.groupe_indices(vieux), na.groupe_structures(vieux, "Regroupement de cratères")):
        for n in g.children():
            g.removeChildNode(n)
    s = na.groupe_structures(vieux)
    for n in s.children():
        if not _est_groupe(n):
            s.removeChildNode(n)
    # Un groupe ajouté par l'utilisateur DANS le groupe d'analyse reste.
    perso = vieux.addGroup("Mes notes")
    na.retirer_groupes_vides(root, _est_groupe)
    noms = [n.name() for n in root.children()]
    assert noms == ["B — 01/11/2026 15:30", "A — 01/11/2026 14:05", "Mon groupe vide", "A — 01/11/2026 14:05"]
    assert vieux.children() == [perso]                      # Structures et Indices vides retirés
    assert garde.findGroup("Structures") is not None
    vieux.removeChildNode(perso)
    na.retirer_groupes_vides(root, _est_groupe)
    assert [n.name() for n in root.children()] == ["B — 01/11/2026 15:30", "Mon groupe vide", "A — 01/11/2026 14:05"]


def test_purge_retire_aussi_l_ancien_structures_a_la_racine():
    root = _Noeud("")
    s = root.insertGroup(0, "Structures")                    # chargé avant 0.19.0
    s.addGroup("Regroupement de cratères")
    na.retirer_groupes_vides(root, _est_groupe)
    assert root.children() == []


def test_chargement_et_qgs_passent_par_le_meme_placement():
    src = Path(__file__).resolve().parents[2] / "src" / "ui"
    for nom in ("layer_loader.py", "qgs_writer.py"):
        code = (src / nom).read_text(encoding="utf-8")
        assert "groupe_analyse(" in code and "groupe_structures(" in code and "groupe_indices(" in code, nom
    loader = (src / "layer_loader.py").read_text(encoding="utf-8")
    assert "retirer_groupes_vides(project.layerTreeRoot(), QgsLayerTree.isGroup)" in loader
    vue = (src / "run_view.py").read_text(encoding="utf-8")
    assert vue.count("titre_du_livrable(") >= 2               # chargement en direct et .qgs
    assert "titre_du_livrable(" in (src / "steps" / "step_6_resultats.py").read_text(encoding="utf-8")


# ── rapport et trace ────────────────────────────────────────────────
def test_titre_du_rapport():
    d = DonneesRapport(version="0.19.0", date="01/11/2026 à 14:05", mode="MNT existant",
                       nom_analyse="be_vl · indices")
    h = construire_html(d)
    assert "<h1>be_vl · indices</h1>" in h and "<title>be_vl · indices — 01/11/2026 à 14:05</title>" in h
    assert "Rapport de traitement" in h
    sans = construire_html(DonneesRapport(version="0.19.0", date="d", mode="m"))
    assert "<h1>Rapport de traitement</h1>" in sans


def test_traitement_json_et_rapport_portent_le_nom(tmp_path, monkeypatch):
    from app.services import finalize_service

    monkeypatch.setattr(finalize_service, "_collect_vrt_paths_and_build", lambda *a, **k: [])
    monkeypatch.setattr(finalize_service, "_build_coverage_polygons", lambda *a, **k: None)

    class _R:
        def __getattr__(self, _n):
            return lambda *a, **k: None

    out = tmp_path / "be_vl" / "indices"
    out.mkdir(parents=True)
    finalize_service.finalize_pipeline(output_dir=out, cv_cfg={}, rvt_params={}, reporter=_R(),
                                       start_time=time.time(), tiles_processed=1, active_products=["MNT"],
                                       ui_config={"app": {"files": {"output_dir": str(out)}}})
    trace = json.loads((out / "livrable" / "traitement.json").read_text(encoding="utf-8"))
    assert trace["nom_analyse"] == "be_vl · indices"          # config sans le champ : nom proposé
    assert "<h1>be_vl · indices</h1>" in (out / "livrable" / "rapport.html").read_text(encoding="utf-8")
    meta = next((out / "technique" / "journaux").glob("metadata_*.json"))
    assert json.loads(meta.read_text(encoding="utf-8"))["nom_analyse"] == "be_vl · indices"
    finalize_service.finalize_pipeline(output_dir=out, cv_cfg={}, rvt_params={}, reporter=_R(),
                                       start_time=time.time(), tiles_processed=1, active_products=["MNT"],
                                       ui_config={"app": {"files": {"output_dir": str(out),
                                                                    "nom_analyse": "Flandre"}}})
    trace = json.loads((out / "livrable" / "traitement.json").read_text(encoding="utf-8"))
    assert trace["nom_analyse"] == "Flandre"
