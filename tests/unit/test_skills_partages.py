"""Les skills d'installation d'un modèle sont identiques ici et dans training-models.

Sauté quand training-models n'est pas sur le poste (CI, autre machine).
Divergence = quelqu'un a modifié un skill dans un seul dépôt :
``python dev/sync_skills_training.py --vers-training`` ou ``--depuis-training``.
"""
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _module():
    spec = importlib.util.spec_from_file_location("sync_skills_training", os.path.join(ROOT, "dev", "sync_skills_training.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_skills_partages_identiques():
    sync = _module()
    if not os.path.isdir(sync.TRAINING):
        pytest.skip(f"training-models absent : {sync.TRAINING}")
    etats = sync.comparer(sync.TRAINING)
    assert etats, "aucun skill partagé trouvé"
    divergents = [(s, n, e) for s, n, e in etats if e != "identique"]
    assert not divergents, f"skills partagés divergents (dev/sync_skills_training.py) : {divergents}"
