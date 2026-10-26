"""Erreurs de configuration rangées sur l'étape et le champ qui les corrigent (lot 2 de
l'interface rationalisée) : plus de tri par mot-clé qui envoyait les erreurs MSTP sur
l'étape 1. Chaque message de ``validate_run_context`` doit avoir sa place."""
import ast
from pathlib import Path

from app.services.rvt_kernel_context import bornes_errors, mstp_scale_errors
from app.services.validation_ui import champ_de_l_erreur, classer_erreurs, etat_rail

RACINE = Path(__file__).resolve().parents[2]


def messages_litteraux_de_validate_run_context():
    arbre = ast.parse((RACINE / "src" / "app" / "run_context.py").read_text(encoding="utf-8"))
    for node in ast.walk(arbre):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "append" and getattr(node.func.value, "id", "") == "errors":
            arg = node.args[0]
            if isinstance(arg, ast.Constant):
                yield arg.value
            elif isinstance(arg, ast.JoinedStr):
                yield "".join(v.value for v in arg.values if isinstance(v, ast.Constant)) + " x"


def test_chaque_message_de_validate_run_context_est_classe():
    msgs = list(messages_litteraux_de_validate_run_context())
    assert len(msgs) >= 13
    non_classes = [m for m in msgs if 0 in classer_erreurs([m])]
    assert not non_classes, f"messages sans étape : {non_classes}"


def test_erreurs_de_bornes_et_mstp_vont_a_l_etape_2():
    msgs = bornes_errors({"LD": True, "SLRM": True}, {"ldo": {"min_radius": 30, "max_radius": 20}, "slrm": {"radius": 80}})
    msgs += mstp_scale_errors({"MSTP": True}, {"mstp": {"local_scale_min": 30, "local_scale_max": 21}})
    assert len(msgs) == 3
    assert classer_erreurs(msgs) == {2: msgs}
    assert all(champ_de_l_erreur(m) == "images" for m in msgs)


def test_messages_de_bornes_parlent_d_images():
    msgs = bornes_errors({"SLRM": True}, {"slrm": {"radius": 80}})
    assert msgs[0].startswith("Image SLRM / «"), msgs


def test_dossier_de_sortie_et_source_vont_a_l_etape_1():
    out = classer_erreurs([
        "Indiquez le dossier de sortie",
        "Le dossier de sortie est dans le dossier source (C:/x) : choisissez un dossier de sortie à part",
        "Indiquez la zone d'étude ou choisissez des dalles sur la carte",
        "Ce dossier est introuvable : C:/mnt",
        "Point de départ inconnu : 'x'",
    ])
    assert set(out) == {1}
    assert champ_de_l_erreur("Indiquez le dossier de sortie") == "sortie"
    assert champ_de_l_erreur("Ce dossier est introuvable : C:/mnt") == "source"


def test_images_vont_a_l_etape_2_et_inconnu_a_zero():
    out = classer_erreurs(["Cochez au moins un produit à calculer à l'étape 2", "Une phrase que personne n'attend"])
    assert out == {2: ["Cochez au moins un produit à calculer à l'étape 2"], 0: ["Une phrase que personne n'attend"]}
    assert champ_de_l_erreur("Une phrase que personne n'attend") == ""


def test_etat_rail():
    erreurs = {2: ["Cochez au moins un produit à calculer à l'étape 2"]}
    assert [etat_rail(n, 1, erreurs) for n in range(1, 6)] == ["active", "a_corriger", "a_venir", "a_venir", "a_venir"]
    assert [etat_rail(n, 4, erreurs) for n in range(1, 6)] == ["ok", "a_corriger", "ok", "active", "a_venir"]
    assert [etat_rail(n, 3, {}) for n in range(1, 6)] == ["ok", "ok", "active", "a_venir", "a_venir"]
