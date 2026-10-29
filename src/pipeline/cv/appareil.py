"""Appareil d'inférence du moteur de détection : processeur, ou carte graphique par DirectML.

Option de l'étape 4, décochée par défaut (0.15.0, vague 3 ; Windows seulement). Au premier
usage d'un modèle sur un poste, le binaire **sonde** : la première image est inférée sur le
processeur (résultat gardé), puis rejouée sur chaque carte candidate. Une carte est retenue si
ORT ouvre vraiment une session DirectML dessus (sinon il retente en silence sur le processeur :
``get_providers`` fait foi), si ses détections recoupent celles du processeur (mêmes objets à
IoU ≥ 0,5 hors bascules au seuil, écart de confiance borné) et si elle est nettement plus rapide.
La parité fine est prouvée au banc (13 546/13 546 objets finaux, écart de confiance ≤ 0,009), pas
par cette sonde, sans puissance statistique : elle écarte un pilote cassé, une carte lente (iGPU,
WARP, bureau à distance) ou un modèle refusé. À la première erreur en cours de traitement, repli
sur une session processeur, annoncé. Aucun Qt, aucun QGIS : le binaire embarque ce module.
"""
from __future__ import annotations

import json
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

PROCESSEUR = "processeur"
CARTE = "carte_graphique"
FOURNISSEUR_DML = "DmlExecutionProvider"
FOURNISSEUR_CPU = "CPUExecutionProvider"
PREFIXE_LIGNE = "appareil="

#: Deux détections sont le même objet à partir de ce recouvrement (même classe).
IOU_APPARIEMENT = 0.5
#: Un objet absent d'un côté dont la confiance est à moins de cela au-dessus de son seuil est une
#: bascule au seuil, attendue entre deux arithmétiques, pas un écart. Même borne que l'écart
#: toléré sur une paire (le banc mesure jusqu'à 0,071 sur les linéaires ; à 0,05 le modèle
#: linéaire était écarté pour une seule bascule, vécu le 18/10).
TOLERANCE_SEUIL = 0.10
#: Écart de confiance maximal toléré sur un objet apparié (le banc mesure 0,071 au pire).
ECART_CONF_MAX = 0.10
#: La carte doit faire la première image en moins de cette part du temps processeur.
RATIO_VITESSE_MAX = 0.8

Detection = Dict[str, Any]
SeuilDe = Callable[[Detection], float]


def providers_dml(device_id: int) -> list:
    return [(FOURNISSEUR_DML, {"device_id": int(device_id)}), FOURNISSEUR_CPU]


def session_sur_carte(session: Any) -> bool:
    """Vrai si la session tourne bien sur DirectML : ORT, quand l'ouverture échoue, la recrée
    sur le processeur sans lever (il l'imprime seulement)."""
    try:
        return FOURNISSEUR_DML in list(session.get_providers())
    except Exception:  # noqa: BLE001 — pas une session ORT
        return False


# ---------------------------------------------------------------------------
# Cartes du poste (DXGI) — énumérées DANS le processus qui ouvre DirectML
# ---------------------------------------------------------------------------
DXGI_ADAPTER_FLAG_SOFTWARE = 2
VENDOR_MICROSOFT = 0x1414   # WARP / Basic Render Driver : pas une carte (CI GitHub, 2026-10-27)


def adaptateurs_dxgi() -> List[Dict[str, Any]]:
    """``[{"id", "nom", "pilote", "vram_go"}, …]`` dans l'ordre DXGI **du processus appelant**,
    qui est l'ordre des ``device_id`` DirectML de ce même processus ; l'adaptateur logiciel
    (WARP) est écarté. L'ordre change d'un processus à l'autre (QGIS : NVIDIA puis AMD ; le
    binaire : AMD puis NVIDIA, vécu le 18/10) : le moteur énumère toujours lui-même.
    ``[]`` hors Windows ou sans DXGI."""
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
                if methode(adapter, 10, ctypes.c_long, ctypes.POINTER(DESC1))(adapter, ctypes.byref(desc)) != 0:
                    continue
                if not desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE and desc.VendorId != VENDOR_MICROSOFT:
                    umd = ctypes.c_longlong(0)   # version du pilote en mode utilisateur (IDXGIAdapter::CheckInterfaceSupport)
                    if methode(adapter, 9, ctypes.c_long, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_longlong))(
                            adapter, ctypes.byref(iid_device), ctypes.byref(umd)) != 0:
                        umd.value = 0
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


def carte_au_bon_index(adaptateurs: Sequence[Dict[str, Any]], device_id: Any, carte: str) -> bool:
    """Une carte mémorisée ne s'applique que si l'adaptateur à son ``device_id`` porte encore
    son nom (écran basculé, carte externe, autre ordre) ; sinon on re-sonde."""
    try:
        did = int(device_id)
    except (TypeError, ValueError):
        return False
    return 0 <= did < len(adaptateurs) and str(adaptateurs[did].get("nom")) == str(carte)


# ---------------------------------------------------------------------------
# Comparaison des détections processeur / carte
# ---------------------------------------------------------------------------
def _bbox(d: Detection) -> Optional[Tuple[float, float, float, float]]:
    b = d.get("bbox_absolute")
    if isinstance(b, dict):
        try:
            return float(b["minx"]), float(b["miny"]), float(b["maxx"]), float(b["maxy"])
        except (KeyError, TypeError, ValueError):
            return None
    poly = d.get("polygon")
    if poly:
        if isinstance(poly[0], (list, tuple)):
            xs, ys = [float(p[0]) for p in poly], [float(p[1]) for p in poly]
        else:
            xs, ys = [float(v) for v in poly[0::2]], [float(v) for v in poly[1::2]]
        if xs and ys:
            return min(xs), min(ys), max(xs), max(ys)
    return None


def iou(a: Optional[Tuple[float, ...]], b: Optional[Tuple[float, ...]]) -> float:
    if a is None or b is None:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def comparer_detections(cpu: Sequence[Detection], gpu: Sequence[Detection], seuil_de: SeuilDe) -> Dict[str, Any]:
    """Apparie les détections (même classe, IoU ≥ 0,5, gloutonnement par meilleur IoU) et juge :
    conforme si aucun objet franc (confiance ≥ seuil + tolérance) n'est absent d'un côté et si
    l'écart de confiance max des paires reste borné. ``verifiable`` est faux sans aucune détection
    des deux côtés : la sonde ne peut alors juger que la vitesse."""
    restants = list(range(len(gpu)))
    paires: List[Tuple[Detection, Detection]] = []
    orphelins_cpu: List[Detection] = []
    for d in cpu:
        bd, meilleur, i_best = _bbox(d), 0.0, None
        for i in restants:
            g = gpu[i]
            if g.get("class_id") != d.get("class_id"):
                continue
            v = iou(bd, _bbox(g))
            if v > meilleur:
                meilleur, i_best = v, i
        if i_best is not None and meilleur >= IOU_APPARIEMENT:
            restants.remove(i_best)
            paires.append((d, gpu[i_best]))
        else:
            orphelins_cpu.append(d)
    orphelins_gpu = [gpu[i] for i in restants]

    def franche(d: Detection) -> bool:
        return float(d.get("confidence", 0.0)) >= float(seuil_de(d)) + TOLERANCE_SEUIL

    manques = [d for d in orphelins_cpu if franche(d)]
    en_trop = [d for d in orphelins_gpu if franche(d)]
    ecarts = sorted(abs(float(a.get("confidence", 0.0)) - float(b.get("confidence", 0.0))) for a, b in paires)
    res = {
        "apparies": len(paires),
        "bascules": len(orphelins_cpu) + len(orphelins_gpu) - len(manques) - len(en_trop),
        "non_retrouvees": len(manques),
        "en_trop": len(en_trop),
        "ecart_conf_max": round(ecarts[-1], 4) if ecarts else 0.0,
        "ecart_conf_median": round(ecarts[len(ecarts) // 2], 4) if ecarts else 0.0,
        "verifiable": bool(paires or manques or en_trop),
    }
    res["conforme"] = not manques and not en_trop and res["ecart_conf_max"] <= ECART_CONF_MAX
    return res


# ---------------------------------------------------------------------------
# Sonde : une carte par une carte, sur la première image
# ---------------------------------------------------------------------------
def _fmt_s(s: float) -> str:
    return f"{s:.1f} s".replace(".", ",")


def _fmt_conf(x: float) -> str:
    return f"{float(x):.3f}".replace(".", ",")


def sonder(
    adaptateurs: Sequence[Dict[str, Any]],
    secondes_processeur: float,
    dets_processeur: Sequence[Detection],
    seuil_de: SeuilDe,
    ouvrir: Callable[[int], Any],
    inferer: Callable[[Any], Tuple[float, Sequence[Detection]]],
    log: Callable[[str], None],
) -> Tuple[Dict[str, Any], Any]:
    """Essaie chaque adaptateur (``{"id", "nom"}``) : ``ouvrir(device_id)`` rend la session
    (tuple ``_load_onnx_model``), ``None`` si ORT est retombé sur le processeur, ou lève ;
    ``inferer(session)`` rejoue la première image et rend ``(secondes, détections)``. Renvoie la
    décision (sérialisable, avec ``message`` pour le journal) et la session retenue, ou ``None``."""
    essais: List[Dict[str, Any]] = []
    sessions: Dict[int, Any] = {}
    for ad in adaptateurs:
        did = int(ad.get("id", 0))
        nom = str(ad.get("nom") or f"carte {did}")
        essai: Dict[str, Any] = {"device_id": did, "nom": nom}
        try:
            session = ouvrir(did)
        except Exception as e:  # noqa: BLE001 — pilote, mémoire, modèle refusé
            essai["motif"] = f"session impossible ({court(e)})"
            essais.append(essai)
            continue
        if session is None:
            essai["motif"] = "DirectML indisponible sur cette carte (le moteur est retombé sur le processeur)"
            essais.append(essai)
            continue
        try:
            secondes, dets = inferer(session)
        except Exception as e:  # noqa: BLE001
            essai["motif"] = f"erreur d'inférence ({court(e)})"
            essais.append(essai)
            continue
        cmp = comparer_detections(dets_processeur, dets, seuil_de)
        essai.update(cmp, secondes=round(float(secondes), 2), secondes_processeur=round(float(secondes_processeur), 2))
        if not cmp["conforme"]:
            essai["motif"] = (f"résultats différents du processeur ({cmp['non_retrouvees']} non retrouvée(s), "
                              f"{cmp['en_trop']} en trop, écart de confiance max {_fmt_conf(cmp['ecart_conf_max'])})")
        elif secondes > secondes_processeur * RATIO_VITESSE_MAX:
            essai["motif"] = (f"pas plus rapide que le processeur ({_fmt_s(secondes)} contre "
                              f"{_fmt_s(secondes_processeur)} sur la première image)")
        else:
            sessions[did] = session
        essais.append(essai)
        log(f"sonde carte graphique : {nom} : {essai.get('motif') or 'conforme et plus rapide'}")
    retenus = [e for e in essais if "motif" not in e]
    if retenus:
        best = min(retenus, key=lambda e: e["secondes"])
        # Sans détection sur la première image, la parité n'a pas été vue : la carte sert pour
        # ce traitement (plus rapide, pas de plantage) mais n'est pas mémorisée — re-sondée au suivant.
        decision = {"appareil": CARTE, "device_id": best["device_id"], "carte": best["nom"],
                    "memorisable": bool(best.get("verifiable")),
                    "sonde": {k: v for k, v in best.items() if k not in ("device_id", "nom")},
                    "ecartees": [f"{e['nom']} : {e['motif']}" for e in essais if "motif" in e]}
        decision["message"] = message(decision)
        return decision, sessions[best["device_id"]]
    motif = " ; ".join(f"{e['nom']} : {e['motif']}" for e in essais) or "aucune carte graphique"
    decision = {"appareil": PROCESSEUR, "motif": motif}
    decision["message"] = message(decision)
    return decision, None


def court(e: BaseException) -> str:
    """Une erreur en une ligne courte, pour le journal."""
    return " ".join(str(e).split())[:160]


_court = court


# ---------------------------------------------------------------------------
# Messages du journal (français, une phrase) et décisions sans sonde
# ---------------------------------------------------------------------------
def message(decision: Dict[str, Any]) -> str:
    if decision.get("appareil") == CARTE:
        s = decision.get("sonde") or {}
        base = f"détection sur la carte graphique « {decision.get('carte', '?')} »"
        if decision.get("memorise"):
            return base + " (vérifiée sur ce poste)"
        if not s:
            return base
        vitesse = f"première image en {_fmt_s(s.get('secondes', 0.0))} contre {_fmt_s(s.get('secondes_processeur', 0.0))} sur le processeur"
        if s.get("verifiable"):
            parite = (f"{s.get('apparies', 0)} détection(s) retrouvée(s), écart de confiance max "
                      f"{_fmt_conf(s.get('ecart_conf_max', 0.0))}")
        else:
            parite = ("aucune détection sur la première image, parité non vérifiable ici "
                      "(vérification refaite au prochain traitement)")
        return f"{base} : {vitesse}, {parite}"
    if decision.get("repli"):
        return (f"carte graphique : erreur à l'image {decision.get('image', '?')} ({decision.get('motif', '?')}) ; "
                "suite de la détection sur le processeur, carte écartée pour sept jours")
    if decision.get("non_sondee"):
        return f"carte graphique non vérifiée : {decision.get('motif', '?')} ; détection sur le processeur"
    return f"carte graphique écartée : {decision.get('motif', '?')} ; détection sur le processeur"


def decision_repli(motif: str, image: int) -> Dict[str, Any]:
    d = {"appareil": PROCESSEUR, "repli": True, "motif": motif, "image": int(image)}
    d["message"] = message(d)
    return d


def decision_memorisee(device_id: int, carte: str) -> Dict[str, Any]:
    d = {"appareil": CARTE, "device_id": int(device_id), "carte": carte, "memorise": True}
    d["message"] = message(d)
    return d


def decision_indisponible(carte: str, motif: str) -> Dict[str, Any]:
    """Carte retenue sur ce poste mais impossible à ouvrir aujourd'hui : transitoire, jamais mémorisée."""
    d = {"appareil": PROCESSEUR, "memorisable": False,
         "motif": f"« {carte} » indisponible aujourd'hui ({motif})"}
    d["message"] = message(d)
    return d


def decision_non_sondee(motif: str) -> Dict[str, Any]:
    """Option demandée mais aucune sonde possible dans ce traitement : dit, jamais mémorisé."""
    d = {"appareil": PROCESSEUR, "non_sondee": True, "memorisable": False, "motif": motif}
    d["message"] = message(d)
    return d


def ligne(decision: Dict[str, Any]) -> str:
    """Ligne de stdout du binaire, relue par le plugin (``external_runner``)."""
    return PREFIXE_LIGNE + json.dumps(decision, ensure_ascii=False)


def lire_ligne(line: str) -> Optional[Dict[str, Any]]:
    if not line.startswith(PREFIXE_LIGNE):
        return None
    try:
        d = json.loads(line[len(PREFIXE_LIGNE):])
    except ValueError:
        return None
    return d if isinstance(d, dict) else None


# ---------------------------------------------------------------------------
# Session avec repli processeur
# ---------------------------------------------------------------------------
class SessionAvecRepli:
    """Délègue à la session DirectML ; à sa première exception de ``run``, ouvre la session
    processeur, le dit par ``on_repli(motif)`` et rejoue l'appel. Une seconde erreur remonte."""

    def __init__(self, session: Any, ouvrir_processeur: Callable[[], tuple], on_repli: Callable[[str], None]):
        self._s = session
        self._ouvrir = ouvrir_processeur
        self._on_repli = on_repli
        self.replie = False

    def run(self, *args, **kwargs):
        try:
            return self._s.run(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 — n'importe quelle erreur du pilote
            if self.replie:
                raise
            self.replie = True
            self._s = self._ouvrir()[0]
            self._on_repli(f"{type(e).__name__}: {_court(e)}")
            return self._s.run(*args, **kwargs)

    def get_inputs(self):
        return self._s.get_inputs()

    def get_outputs(self):
        return self._s.get_outputs()

    def get_providers(self):
        return self._s.get_providers()


def avec_repli(session_tuple: tuple, ouvrir_processeur: Callable[[], tuple], on_repli: Callable[[str], None]) -> tuple:
    """Même tuple que ``_load_onnx_model`` (session, input_name, input_shape, model_meta), la
    session enveloppée."""
    session, *reste = session_tuple
    return (SessionAvecRepli(session, ouvrir_processeur, on_repli), *reste)
