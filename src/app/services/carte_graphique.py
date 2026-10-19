"""Carte graphique pour la détection (option de l'étape 4, 0.15.0) : côté plugin.

Le moteur (binaire) fait la sonde (``pipeline.cv.appareil``) ; ici on décide seulement s'il faut
sonder, ou appliquer une décision déjà prise sur ce poste pour ces cartes, ces pilotes, ce
binaire et ce modèle : le registre ``carte_graphique.json`` du profil QGIS (dossier
``archeologia``, comme les couleurs de classes). Une carte retenue le reste tant que la clé ne
change pas ; un refus (plus lente, résultats différents, pilote) est réessayé après sept jours.
Les cartes du poste viennent de DXGI (``pipeline.cv.appareil.adaptateurs_dxgi``, Windows) et ne
servent ici qu'à la clé : l'ordre DXGI dépend du processus (QGIS voit NVIDIA puis AMD là où le
binaire voit AMD puis NVIDIA, vécu le 18/10), donc le moteur énumère lui-même pour les
``device_id`` et vérifie qu'une carte mémorisée est encore à son index. Pur : pas de Qt.
"""
from __future__ import annotations

import datetime as dt
import functools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

NOM_REGISTRE = "carte_graphique.json"
VALIDITE_REFUS_JOURS = 7
CARTE = "carte_graphique"
PROCESSEUR = "processeur"


# ---------------------------------------------------------------------------
# Cartes du poste (DXGI)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def adaptateurs() -> List[Dict[str, Any]]:
    """Cartes du poste vues depuis QGIS (``pipeline.cv.appareil.adaptateurs_dxgi``) : servent à la
    **clé** (noms et pilotes, triés) et à dire « aucune carte ». Leur ordre n'est pas celui du
    moteur (DXGI ordonne par processus) : le moteur énumère lui-même pour les ``device_id``."""
    if sys.platform != "win32":
        return []
    try:
        try:
            from ...pipeline.cv.appareil import adaptateurs_dxgi
        except ImportError:  # pragma: no cover
            from pipeline.cv.appareil import adaptateurs_dxgi
        return adaptateurs_dxgi()
    except Exception:  # noqa: BLE001 — poste sans DXGI (session distante, pilote absent)
        return []




# ---------------------------------------------------------------------------
# Registre du profil
# ---------------------------------------------------------------------------
def registre_path() -> Path:
    """Dans le profil QGIS (``archeologia/``), à côté du registre des couleurs ; repli à la
    racine du plugin hors QGIS."""
    try:
        from qgis.core import QgsApplication
        base = Path(QgsApplication.qgisSettingsDirPath()) / "archeologia"
    except Exception:  # noqa: BLE001
        base = Path(__file__).resolve().parents[3]
    return base / NOM_REGISTRE


def empreinte_modele(poids: Optional[Path]) -> str:
    """Nom, taille et date du ``.onnx`` : change avec le modèle, sans le relire (100 Mo)."""
    if poids is None:
        return "?"
    p = Path(poids)
    try:
        st = p.stat()
    except OSError:
        return p.name
    return f"{p.name}:{st.st_size}:{int(st.st_mtime)}"


def cle(adaptateurs_: Sequence[Dict[str, Any]], binaire: Dict[str, Any], empreinte: str) -> str:
    cartes = sorted(f"{a.get('nom')}|{a.get('pilote')}" for a in adaptateurs_)
    ort = binaire.get("onnxruntime") if isinstance(binaire.get("onnxruntime"), dict) else {}
    return " ; ".join(["+".join(cartes), str(binaire.get("sources_sha256") or "?"),
                       str(ort.get("version") or "?"), empreinte])


def lire(path: Path) -> Dict[str, Any]:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def memoriser(path: Path, cle_: str, decision: Dict[str, Any], maintenant: dt.datetime) -> None:
    path = Path(path)
    reg = lire(path)
    reg[cle_] = {**decision, "date": maintenant.isoformat(timespec="seconds")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")


def decision_valide(registre: Dict[str, Any], cle_: str, maintenant: dt.datetime) -> Optional[Dict[str, Any]]:
    d = registre.get(cle_)
    if not isinstance(d, dict) or d.get("appareil") not in (CARTE, PROCESSEUR):
        return None
    if d["appareil"] == CARTE:
        return d
    try:
        date = dt.datetime.fromisoformat(str(d.get("date")))
    except (TypeError, ValueError):
        return None
    return d if maintenant - date < dt.timedelta(days=VALIDITE_REFUS_JOURS) else None


# ---------------------------------------------------------------------------
# Bloc transmis au moteur et trace du journal
# ---------------------------------------------------------------------------
def _date_fr(iso: Any) -> str:
    try:
        return dt.datetime.fromisoformat(str(iso)).strftime("%d/%m/%Y")
    except (TypeError, ValueError):
        return "?"


def binaire_directml(binaire: Dict[str, Any]) -> bool:
    ort = binaire.get("onnxruntime") if isinstance(binaire.get("onnxruntime"), dict) else {}
    return "directml" in str(ort.get("paquet") or "").lower()


def bloc_run(demande: bool, adaptateurs_: Sequence[Dict[str, Any]], decision: Optional[Dict[str, Any]],
             binaire: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """``cv_config["carte_graphique"]`` du run : ``None`` option décochée ; ``demande`` faux avec
    un ``message`` à narrer quand rien n'est à sonder ; sinon la demande de sonde, ou le
    ``device_id`` et le nom d'une carte déjà retenue (vérifiés par le moteur)."""
    if not demande:
        return None
    if not adaptateurs_:
        return {"demande": False, "message": "carte graphique demandée, mais aucune carte graphique "
                                             "utilisable sur ce poste (Windows seulement) ; détection sur le processeur"}
    if not binaire_directml(binaire):
        return {"demande": False, "message": "carte graphique demandée, mais le moteur de détection livré ne la "
                                             "prend pas en charge ; détection sur le processeur"}
    if decision and decision.get("appareil") == PROCESSEUR:
        return {"demande": False, "message": f"carte graphique écartée sur ce poste le {_date_fr(decision.get('date'))} "
                                             f"({decision.get('motif', '?')}) ; détection sur le processeur, "
                                             f"nouvel essai après {VALIDITE_REFUS_JOURS} jours"}
    bloc: Dict[str, Any] = {"demande": True}
    if decision and decision.get("appareil") == CARTE:
        # Le device_id est celui du moteur, qui vérifie lui-même que la carte à cet index porte
        # encore ce nom (sinon il re-sonde) : l'ordre vu d'ici n'est pas le sien.
        try:
            bloc["device_id"] = int(decision.get("device_id", 0))
        except (TypeError, ValueError):
            return bloc
        bloc["carte"] = str(decision.get("carte") or "?")
    return bloc


def preparer(run_cfg: Dict[str, Any], demande: bool, *, adaptateurs: Sequence[Dict[str, Any]],
             binaire: Dict[str, Any], empreinte_modele: str, registre: Path,
             maintenant: Optional[dt.datetime] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Copie de ``run_cfg`` avec son bloc ``carte_graphique``, et le contexte ``{cle, decision,
    message}`` : ``message`` est à narrer tout de suite (rien à sonder), ``cle`` sert à
    ``enregistrer`` après le run."""
    ads = list(adaptateurs)
    maintenant = maintenant or dt.datetime.now()
    if not demande:
        return dict(run_cfg), {"cle": None, "decision": None, "message": None}
    cle_ = cle(ads, binaire, empreinte_modele)
    decision = decision_valide(lire(registre), cle_, maintenant)
    bloc = bloc_run(True, ads, decision, binaire)
    cfg = dict(run_cfg)
    if bloc is not None:
        cfg["carte_graphique"] = bloc
    return cfg, {"cle": cle_, "decision": decision, "message": (bloc or {}).get("message")}


def enregistrer(stats: Dict[str, Any], cle_: Optional[str], registre: Path,
                maintenant: Optional[dt.datetime] = None) -> Optional[Dict[str, Any]]:
    """Mémorise la décision remontée par le moteur (``stats["appareil"]``). Jamais : une
    décision déjà mémorisée, ni ce que le moteur marque ``memorisable: False`` (carte
    indisponible un jour, parité non vérifiable, sonde impossible). Un repli en cours de run
    devient un refus de sept jours : une carte qui plante à l'image k ne doit pas être
    re-sondée, retenue et repliée à chaque traitement."""
    d = (stats or {}).get("appareil")
    if not cle_ or not isinstance(d, dict) or d.get("memorise") or d.get("memorisable") is False:
        return None
    if d.get("appareil") not in (CARTE, PROCESSEUR):
        return None
    if d.get("repli"):
        decision = {"appareil": PROCESSEUR, "repli": True,
                    "motif": f"erreur de la carte en cours de traitement, à l'image {d.get('image', '?')} "
                             f"({d.get('motif', '?')})"}
    else:
        decision = {k: v for k, v in d.items() if k != "message"}
    memoriser(registre, cle_, decision, maintenant or dt.datetime.now())
    return decision


# ---------------------------------------------------------------------------
# Entrée depuis les runners (imports différés : pipeline.cv tire shapely)
# ---------------------------------------------------------------------------
def infos_binaire() -> Dict[str, Any]:
    """``build_info.json`` posé à côté de l'exe par ``dev/runner_onnx/build.py`` ; ``{}`` sans exe."""
    try:
        try:
            from ...pipeline.cv.external_runner import find_external_cv_runner
        except ImportError:  # pragma: no cover
            from pipeline.cv.external_runner import find_external_cv_runner
        exe = find_external_cv_runner()
        if exe is None:
            return {}
        return json.loads((exe.parent / "build_info.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def preparer_run(run_cfg: Dict[str, Any], demande: bool) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``preparer`` avec les cartes du poste, le binaire livré et l'empreinte du modèle du run."""
    if not demande:
        return dict(run_cfg), {"cle": None, "decision": None, "message": None}
    try:
        try:
            from ...pipeline.cv.model_config import resolve_model_weights_path
        except ImportError:  # pragma: no cover
            from pipeline.cv.model_config import resolve_model_weights_path
        poids = resolve_model_weights_path(run_cfg)
    except Exception:  # noqa: BLE001
        poids = None
    return preparer(run_cfg, True, adaptateurs=adaptateurs(), binaire=infos_binaire(),
                    empreinte_modele=empreinte_modele(poids), registre=registre_path())
