"""Carte graphique pour la détection (option de l'étape 4, 0.15.0) : côté plugin.

Le moteur (binaire) fait la sonde (``pipeline.cv.appareil``) ; ici on décide seulement s'il faut
sonder, ou appliquer une décision déjà prise sur ce poste pour ces cartes, ces pilotes, ce
binaire et ce modèle : le registre ``carte_graphique.json`` du profil QGIS (dossier
``archeologia``, comme les couleurs de classes). Une carte retenue le reste tant que la clé ne
change pas ; un refus (plus lente, résultats différents, pilote) est réessayé après sept jours.
Les cartes du poste viennent de DXGI (ctypes, Windows seulement) : l'ordre d'énumération est
celui des ``device_id`` de DirectML, l'adaptateur logiciel (WARP) est écarté. Pur : pas de Qt.
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
DXGI_ADAPTER_FLAG_SOFTWARE = 2


# ---------------------------------------------------------------------------
# Cartes du poste (DXGI)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def adaptateurs() -> List[Dict[str, Any]]:
    """``[{"id", "nom", "pilote", "vram_go"}, …]`` dans l'ordre DXGI = ``device_id`` DirectML ;
    ``[]`` hors Windows ou si DXGI ne répond pas (le journal dira « aucune carte »)."""
    if sys.platform != "win32":
        return []
    try:
        return _adaptateurs_dxgi()
    except Exception:  # noqa: BLE001 — poste sans DXGI (session distante, pilote absent)
        return []


def _adaptateurs_dxgi() -> List[Dict[str, Any]]:
    import ctypes
    import ctypes.wintypes as w

    class GUID(ctypes.Structure):
        _fields_ = [("a", w.DWORD), ("b", w.WORD), ("c", w.WORD), ("d", ctypes.c_ubyte * 8)]

    class LUID(ctypes.Structure):
        _fields_ = [("LowPart", w.DWORD), ("HighPart", w.LONG)]

    class DESC1(ctypes.Structure):
        _fields_ = [("Description", w.WCHAR * 128), ("VendorId", w.UINT), ("DeviceId", w.UINT),
                    ("SubSysId", w.UINT), ("Revision", w.UINT), ("DedicatedVideoMemory", ctypes.c_size_t),
                    ("DedicatedSystemMemory", ctypes.c_size_t), ("SharedSystemMemory", ctypes.c_size_t),
                    ("AdapterLuid", LUID), ("Flags", w.UINT)]

    def methode(obj, index, restype, *argtypes):
        vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.c_void_p))[0]
        fn = ctypes.cast(vtbl, ctypes.POINTER(ctypes.c_void_p))[index]
        return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(fn)

    # IID_IDXGIFactory1 = 770aae78-f26f-4dba-a829-253c83d1b387
    iid = GUID(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
    factory = ctypes.c_void_p()
    if ctypes.windll.dxgi.CreateDXGIFactory1(ctypes.byref(iid), ctypes.byref(factory)) != 0:
        return []
    iid_device = GUID(0x54EC77FA, 0x1377, 0x44E6, (ctypes.c_ubyte * 8)(0x8C, 0x32, 0x88, 0xFD, 0x5F, 0x44, 0xC8, 0x4C))
    out: List[Dict[str, Any]] = []
    try:
        enum_adapters1 = methode(factory, 12, ctypes.c_long, w.UINT, ctypes.POINTER(ctypes.c_void_p))
        i = 0
        while True:
            adapter = ctypes.c_void_p()
            if enum_adapters1(factory, i, ctypes.byref(adapter)) != 0:
                break
            try:
                desc = DESC1()
                methode(adapter, 10, ctypes.c_long, ctypes.POINTER(DESC1))(adapter, ctypes.byref(desc))
                if not desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE:
                    umd = ctypes.c_longlong(0)   # version du pilote en mode utilisateur (IDXGIAdapter::CheckInterfaceSupport)
                    methode(adapter, 9, ctypes.c_long, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_longlong))(
                        adapter, ctypes.byref(iid_device), ctypes.byref(umd))
                    v = umd.value
                    pilote = ".".join(str((v >> s) & 0xFFFF) for s in (48, 32, 16, 0)) if v else "?"
                    out.append({"id": i, "nom": desc.Description.strip() or f"carte {i}", "pilote": pilote,
                                "vram_go": round(desc.DedicatedVideoMemory / 2 ** 30, 1)})
            finally:
                methode(adapter, 2, w.ULONG)(adapter)
            i += 1
    finally:
        methode(factory, 2, w.ULONG)(factory)
    return out


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
    un ``message`` à narrer quand rien n'est à sonder ; sinon les adaptateurs à sonder, ou le
    ``device_id`` d'une carte déjà retenue."""
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
    bloc: Dict[str, Any] = {"demande": True, "adaptateurs": [dict(a) for a in adaptateurs_]}
    if decision and decision.get("appareil") == CARTE:
        bloc["device_id"] = int(decision.get("device_id", 0))
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
    """Mémorise la décision de la sonde remontée par le moteur (``stats["appareil"]``) ; rien pour
    un repli en cours de run (transitoire) ni pour une décision déjà mémorisée."""
    d = (stats or {}).get("appareil")
    if not cle_ or not isinstance(d, dict) or d.get("repli") or d.get("memorise"):
        return None
    if d.get("appareil") not in (CARTE, PROCESSEUR):
        return None
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
