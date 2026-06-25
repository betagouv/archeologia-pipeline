"""CFG-04 (audit v2) : config.example.json (modèle du format, créé par
SEC-01) avait dérivé du schéma réel — clés V2 absentes (selected_entities,
entity_*…), produit HS absent, run d'exemple SANS confidence_threshold
(= exactement la config legacy qui déclenche la divergence de seuil ARCH-01).

Ce test verrouille : l'exemple expose EXACTEMENT le jeu de clés de
``ConfigManager.default_config()`` (le schéma vivant), pour empêcher toute
nouvelle dérive dans un sens comme dans l'autre.
"""
from __future__ import annotations

import json
from pathlib import Path

from config.config_manager import ConfigManager

_REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = _REPO_ROOT / "config.example.json"


def _key_paths(d: dict, prefix: str = "") -> set:
    paths = set()
    for k, v in d.items():
        if k == "_comment":
            continue
        p = f"{prefix}.{k}" if prefix else k
        paths.add(p)
        if isinstance(v, dict):
            paths |= _key_paths(v, p)
    return paths


def _example() -> dict:
    return json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))


def test_example_expose_exactement_le_schema_par_defaut():
    example_keys = _key_paths(_example())
    default_keys = _key_paths(ConfigManager(_REPO_ROOT).default_config())
    manquantes = default_keys - example_keys
    en_trop = example_keys - default_keys
    assert not manquantes and not en_trop, (
        f"config.example.json a dérivé du schéma : manquantes={sorted(manquantes)}, "
        f"en trop={sorted(en_trop)}"
    )


def test_le_run_d_exemple_porte_un_seuil_par_run():
    # ARCH-01 : sans confidence_threshold PAR RUN, binning (repli 0.3) et
    # symbologie (repli 0.0) divergent → tranche basse invisible.
    runs = _example()["computer_vision"]["runs"]
    assert runs, "l'exemple doit montrer au moins un run"
    assert all("confidence_threshold" in r for r in runs)


def test_le_commentaire_ne_renvoie_plus_vers_config_json():
    # CFG-01 : config.json n'est PLUS lu par le plugin (le wizard charge
    # last_ui_config.json) — le template ne doit plus enseigner de le créer.
    comment = _example().get("_comment", "")
    assert "Copier en 'config.json'" not in comment
    assert "Charger" in comment  # pointe vers le bouton « Charger une config »
