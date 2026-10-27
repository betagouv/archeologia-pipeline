"""Carte d'entité (étape 3) — paramètres du regroupement émis à l'écran.

Le code UI exige QGIS : on évalue par AST l'expression réelle qui fixe
``_active_cluster_keys`` dans ``EntityRow.update_state``, avec les défauts que
l'orchestrateur peut produire. Toute clé active doit avoir un champ, sinon
``_on_cluster_params_changed`` lève ``KeyError`` au premier changement et
n'émet plus rien (régression de 8d334dc : ``min_confidence_extend``, lu
d'args.yaml pour décaler l'extension, sans champ à l'écran).
"""
import ast
import os
from types import SimpleNamespace

from src.app.services import model_orchestrator as mo

CARD = os.path.join(os.path.dirname(__file__), "..", "..", "src", "ui", "widgets", "entity_row.py")


def _arbre():
    with open(CARD, encoding="utf-8") as f:
        return ast.parse(f.read())


def _cles_des_champs(arbre):
    for n in arbre.body:
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "_CLUSTER_PARAM_SPECS":
            return {spec[0] for spec in ast.literal_eval(n.value)}
    raise AssertionError("_CLUSTER_PARAM_SPECS introuvable")


def _expression_cles_actives(arbre):
    for n in ast.walk(arbre):
        if isinstance(n, ast.FunctionDef) and n.name == "update_state":
            for a in ast.walk(n):
                if isinstance(a, ast.Assign) and ast.unparse(a.targets[0]) == "self._active_cluster_keys":
                    return ast.unparse(a.value)
    raise AssertionError("affectation de _active_cluster_keys introuvable dans update_state")


def test_seules_les_cles_qui_ont_un_champ_sont_emises():
    arbre = _arbre()
    champs = _cles_des_champs(arbre)
    # Tout ce que l'orchestrateur peut lire d'args.yaml, dont min_confidence_extend.
    defaults = {k: 0.3 for k in (*mo._CLUSTER_PARAM_FLOAT, *mo._CLUSTER_PARAM_INT)}
    assert "min_confidence_extend" in defaults
    self = SimpleNamespace(_cluster_spins=dict.fromkeys(champs))
    actives = eval(_expression_cles_actives(arbre), {"defaults": defaults, "self": self})  # noqa: S307
    assert set(actives) <= champs
    assert {"eps_m", "min_confidence", "min_cluster_size"} <= set(actives)
