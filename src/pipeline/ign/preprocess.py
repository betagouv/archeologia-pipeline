from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ..coords import extract_xy_from_filename, infer_xy_from_file
from .downloader import download_one, _get_proxy_config
from .pdal_validation import (
    run_pdal_command_cancellable,
    validate_las_or_laz_with_pdal,
    validate_laz_deep,
)
from ..types import LogFn, CancelFn


@dataclass(frozen=True)
class IgnPreprocessResult:
    merged_dir: Path
    temp_dir: Path
    merged_files: List[Path]
    # Dalles non fusionnées (« nom: erreur ») — la fusion n'est plus
    # tout-ou-rien (AUDIT v2 ROB-12) : l'appelant rapporte ces échecs et
    # continue avec ``merged_files``.
    failed: List[str] = field(default_factory=list)


def _default_log(_: str) -> None:
    return


def _default_cancel() -> bool:
    return False


#: Méthode de fusion des voisines, inscrite dans la signature du cache des
#: intermédiaires (cache_guard) et dans traitement.json : la changer refait
#: fusions et MNT d'un dossier écrit autrement, au lieu de les resservir.
FUSION_ANCIENNE = "pdal-merge"
#: flux-2 (2026-10-17) : le fusionné ne garde que les classes du filtre de l'étape 2 (les seules que MNT et
#: densité lisent) — 4× moins de points et de temps de MNT en forêt, images identiques ; ses bornes avant
#: filtrage vont au sidecar. flux-1 (non livré) écrivait tout le nuage.
FUSION_FLUX = "flux-2"


def fusion_mode() -> str:
    """Méthode de fusion de ce run : en flux (lot 1), sauf ``ARCHEO_FUSION_LEGACY=1``
    — l'ancienne fusion, gardée une version pour diagnostiquer un MNT chez un
    utilisateur."""
    # « 1 », « true », « oui », « yes », « on », sans la casse : un diagnostic ne doit pas échouer sur la forme
    legacy = os.environ.get("ARCHEO_FUSION_LEGACY", "").strip().lower() in ("1", "true", "oui", "yes", "on")
    return FUSION_ANCIENNE if legacy else FUSION_FLUX


def calculate_neighbor_coordinates(x: str, y: str) -> List[Tuple[int, int, int]]:
    min_x = int(f"1{x}") - 10000
    min_y = int(f"1{y}") - 10000

    neighbors: List[Tuple[int, int, int]] = []
    place_dalle = 0

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            place_dalle += 1
            voisin_x = min_x + dx
            voisin_y = min_y + dy
            neighbors.append((voisin_x, voisin_y, place_dalle))

    return neighbors


def format_coordinate(coord: int) -> str:
    return f"{coord:04d}"


def calculate_crop_bounds(voisin_x: int, voisin_y: int, place_dalle: int, margin_m: int) -> Dict[str, str]:
    bounds: Dict[str, str] = {}

    if place_dalle == 1:
        xnum = int(f"1{voisin_x:04d}") - 10000
        bounds["xmin"] = f"{xnum:04d}{(1000 - margin_m):03d}"
        xnum2 = int(f"1{voisin_x:04d}") - 10000 + 1
        bounds["xmax"] = f"{xnum2:04d}000"
        ynum = int(f"1{voisin_y:04d}") - 10000 - 1
        bounds["ymin"] = f"{ynum:04d}{(1000 - margin_m):03d}"
        bounds["ymax"] = f"{voisin_y:04d}000"

    elif place_dalle == 2:
        xnum = int(f"1{voisin_x:04d}") - 10000
        bounds["xmin"] = f"{xnum:04d}{(1000 - margin_m):03d}"
        xnum2 = int(f"1{voisin_x:04d}") - 10000 + 1
        bounds["xmax"] = f"{xnum2:04d}000"
        ynum = int(f"1{voisin_y:04d}") - 10000 - 1
        bounds["ymin"] = f"{ynum:04d}000"
        bounds["ymax"] = f"{voisin_y:04d}000"

    elif place_dalle == 3:
        xnum = int(f"1{voisin_x:04d}") - 10000
        bounds["xmin"] = f"{xnum:04d}{(1000 - margin_m):03d}"
        xnum2 = int(f"1{voisin_x:04d}") - 10000 + 1
        bounds["xmax"] = f"{xnum2:04d}000"
        ynum = int(f"1{voisin_y:04d}") - 10000 - 1
        bounds["ymin"] = f"{ynum:04d}000"
        ynum2 = int(f"1{voisin_y:04d}") - 10000 - 1
        bounds["ymax"] = f"{ynum2:04d}{margin_m:03d}"

    elif place_dalle == 4:
        bounds["xmin"] = f"{voisin_x:04d}000"
        xnum = int(f"1{voisin_x:04d}") - 10000 + 1
        bounds["xmax"] = f"{xnum:04d}000"
        ynum = int(f"1{voisin_y:04d}") - 10000 - 1
        bounds["ymin"] = f"{ynum:04d}{(1000 - margin_m):03d}"
        bounds["ymax"] = f"{voisin_y:04d}000"

    elif place_dalle == 5:
        bounds["xmin"] = f"{voisin_x:04d}000"
        xnum = int(f"1{voisin_x:04d}") - 10000 + 1
        bounds["xmax"] = f"{xnum:04d}000"
        ynum = int(f"1{voisin_y:04d}") - 10000 - 1
        bounds["ymin"] = f"{ynum:04d}000"
        ynum2 = int(f"1{voisin_y:04d}") - 10000 - 1
        bounds["ymax"] = f"{ynum2:04d}{margin_m:03d}"

    elif place_dalle == 6:
        xnum = int(f"1{voisin_x:04d}") - 10000
        bounds["xmin"] = f"{xnum:04d}000"
        bounds["xmax"] = f"{xnum:04d}{margin_m:03d}"
        ynum = int(f"1{voisin_y:04d}") - 10000 - 1
        bounds["ymin"] = f"{ynum:04d}{(1000 - margin_m):03d}"
        bounds["ymax"] = f"{voisin_y:04d}000"

    elif place_dalle == 7:
        xnum = int(f"1{voisin_x:04d}") - 10000
        bounds["xmin"] = f"{xnum:04d}000"
        bounds["xmax"] = f"{xnum:04d}{margin_m:03d}"
        ynum = int(f"1{voisin_y:04d}") - 10000 - 1
        bounds["ymin"] = f"{ynum:04d}000"
        bounds["ymax"] = f"{voisin_y:04d}000"

    else:
        xnum = int(f"1{voisin_x:04d}") - 10000
        bounds["xmin"] = f"{xnum:04d}000"
        bounds["xmax"] = f"{xnum:04d}{margin_m:03d}"
        ynum = int(f"1{voisin_y:04d}") - 10000 - 1
        bounds["ymin"] = f"{ynum:04d}000"
        ynum2 = int(f"1{voisin_y:04d}") - 10000 - 1
        bounds["ymax"] = f"{ynum2:04d}{margin_m:03d}"

    return bounds


def _pdal_exe() -> str:
    p = shutil.which("pdal")
    if not p:
        raise FileNotFoundError("pdal executable not found in PATH")
    return p


def _partial_path(output_path: Path) -> Path:
    """Chemin d'écriture, à côté de la cible (même volume → rename atomique).

    Tant que l'écriture n'a pas réussi, rien ne porte le nom final : un run
    tué en pleine écriture ne laisse qu'un ``.partial`` inerte au lieu d'un
    LAZ corrompu que la validation d'en-tête déclarerait valide à vie
    (incident 2026-09-19, dalle LHD_FXX_0821_6327 : 161 Mio de zéros en
    queue, MNT bloqué 10 h sur une boucle infinie de PDAL).

    L'extension ``.laz`` reste en dernier : PDAL choisit lecteur et écrivain
    sur l'extension, et ``X.laz.partial`` lui est illisible (« Cannot
    determine reader ») — régression 2026-09-22, tout voisin rogné rejeté,
    MNT sans marge, couture visible entre dalles.
    """
    return output_path.with_name(output_path.stem + ".partial" + output_path.suffix)


def _discard_partial(partial: Path) -> None:
    try:
        partial.unlink()
    except OSError:
        pass


def crop_neighbor_tile(
    *,
    input_path: Path,
    output_path: Path,
    bounds: Dict[str, str],
    log: LogFn = _default_log,
    cancel: CancelFn = _default_cancel,
) -> bool:
    if cancel():
        return False

    if output_path.exists():
        ok, _ = validate_las_or_laz_with_pdal(output_path)
        if ok:
            return True
        try:
            output_path.unlink()
        except Exception:
            pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = _partial_path(output_path)

    pipeline_config = {
        "pipeline": [
            {"type": "readers.las", "filename": str(input_path)},
            {
                "type": "filters.crop",
                "bounds": f"([{bounds['xmin']},{bounds['xmax']}],[{bounds['ymin']},{bounds['ymax']}])",
            },
            {"type": "writers.las", "filename": str(partial), "compression": "laszip"},
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(pipeline_config, tf, indent=2)
        pipeline_file = tf.name

    try:
        cmd = [_pdal_exe(), "pipeline", pipeline_file]
        result = run_pdal_command_cancellable(cmd, cancel=cancel)
        if result.returncode != 0:
            log(f"Erreur PDAL crop (code {result.returncode})")
            if result.returncode == 3221225477:
                log("PDAL a crashé (0xC0000005). Conseil: relancez le pipeline avec moins de workers (ex: max_workers=2).")
            if result.stderr:
                log(result.stderr.strip())
            return False

        ok, msg = validate_las_or_laz_with_pdal(partial)
        if not ok:
            log(f"Fichier voisin rogné invalide via PDAL: {output_path.name}")
            if msg:
                log(f"PDAL: {msg}")
            return False

        os.replace(str(partial), str(output_path))
        return True
    finally:
        try:
            Path(pipeline_file).unlink()
        except Exception:
            pass
        _discard_partial(partial)


def merged_inputs_sidecar(output_path: Path) -> Path:
    """Sidecar du jeu de voisins fusionnés : ``<tile>_merged.inputs.json``."""
    return output_path.with_suffix(".inputs.json")


def merged_inputs_match(output_path: Path, neighbor_paths: List[Path]):
    """Compare le jeu de voisins courant à celui enregistré à la fusion.

    ``None`` = sidecar absent (fusionné legacy, provenance inconnue) ;
    ``True``/``False`` sinon (sidecar illisible → ``False``, on refusionne).
    """
    sidecar = merged_inputs_sidecar(output_path)
    if not sidecar.exists():
        return None
    try:
        recorded = json.loads(sidecar.read_text(encoding="utf-8"))
        recorded = sorted(str(n) for n in recorded)
    except Exception:
        return False
    return recorded == sorted(p.name for p in neighbor_paths)


def write_merged_inputs_sidecar(output_path: Path, neighbor_paths: List[Path]) -> None:
    merged_inputs_sidecar(output_path).write_text(
        json.dumps(sorted(p.name for p in neighbor_paths)), encoding="utf-8"
    )


def est_copc(path: Path) -> bool:
    """Vrai si le LAZ porte le VLR ``copc`` (premier VLR, juste après l'en-tête)."""
    try:
        with open(path, "rb") as f:
            return b"copc" in f.read(1024)
    except OSError:
        return False


def merge_tiles(
    *,
    central_path: Path,
    neighbor_paths: List[Path],
    output_path: Path,
    log: LogFn = _default_log,
    cancel: CancelFn = _default_cancel,
) -> bool:
    if cancel():
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        ok, _ = validate_las_or_laz_with_pdal(output_path)
        if ok and est_copc(output_path):
            # Copie brute d'une dalle isolée par une version ≤ 0.14.0 : illisible
            # par QGIS (cf. plus bas) → réécrite.
            ok = False
        if ok:
            # Le cache n'est valable que pour le MÊME jeu de voisins : au
            # re-run « extension de zone » ({A} puis {A,B}), la marge de A
            # vers B était restée FABRIQUÉE et ses détections de marge
            # ressortaient dans la cellule de B (clip élargi). Un jeu
            # différent → re-fusion, dont le mtime frais ré-arme toute la
            # chaîne de fraîcheur aval (MNT → RVT → PNG → cache CV).
            match = merged_inputs_match(output_path, neighbor_paths)
            if match is None:
                # Fusionné legacy (pas de sidecar) : adoption du jeu courant
                # comme référence, sans re-fusion forcée (cf. cache_guard).
                write_merged_inputs_sidecar(output_path, neighbor_paths)
                log(f"LAZ fusionné réutilisé (cache intermédiaire) : {output_path.name}")
                return True
            if match:
                log(f"LAZ fusionné réutilisé (cache intermédiaire) : {output_path.name}")
                return True
            log(f"Jeu de voisins modifié → re-fusion : {output_path.name}")
        try:
            output_path.unlink()
        except Exception:
            pass

    ok_c, msg_c = validate_las_or_laz_with_pdal(central_path)
    if not ok_c:
        log(f"Fichier central invalide via PDAL: {central_path.name}")
        if msg_c:
            log(f"PDAL: {msg_c}")
        return False

    valid_files: List[Path] = [central_path]
    for p in neighbor_paths:
        if cancel():
            return False
        ok, _ = validate_las_or_laz_with_pdal(p)
        if ok:
            valid_files.append(p)

    partial = _partial_path(output_path)
    try:
        # Sans voisin, la dalle est RÉÉCRITE par PDAL, jamais copiée : la dalle
        # IGN est un COPC dont le fournisseur pdal de QGIS refuse les octets
        # supplémentaires (« Extra byte specification exceeds point length »),
        # et pdal:exportrastertin échouait sur toute dalle isolée (recette 0.14.0).
        if len(valid_files) <= 1:
            cmd = [_pdal_exe(), "translate", str(central_path), str(partial)]
        else:
            cmd = [_pdal_exe(), "merge"] + [str(p) for p in valid_files] + [str(partial)]
        result = run_pdal_command_cancellable(cmd, cancel=cancel)
        if result.returncode != 0:
            log(f"Erreur PDAL {cmd[1]} (code {result.returncode})")
            log("💡 Conseil: réduisez max_workers dans config.json (ex: max_workers=1 ou 2) pour éviter les crashs mémoire.")
            if result.returncode in (3221225477, 3221226505):
                # 0xC0000005 = ACCESS_VIOLATION, 0xC0000409 = STACK_BUFFER_OVERRUN
                log("PDAL a crashé (erreur mémoire).")
            if result.stderr:
                log(result.stderr.strip())
            return False

        ok_out, msg_out = validate_las_or_laz_with_pdal(partial)
        if not ok_out:
            log(f"Fichier fusionné invalide via PDAL: {output_path.name}")
            if msg_out:
                log(f"PDAL: {msg_out}")
            return False

        os.replace(str(partial), str(output_path))
    finally:
        _discard_partial(partial)

    write_merged_inputs_sidecar(output_path, neighbor_paths)
    return True


def expression_pdal(expression_qgis: str) -> Optional[str]:
    """Le filtre des points de l'étape 2 (syntaxe de QGIS : ``Classification = 2 OR …``) en syntaxe PDAL
    (``== … ||``), telle que QGIS la passe à pdal_wrench. ``None`` si l'expression sort de la grammaire
    reconnue (comparaisons de dimensions à des nombres, AND/OR/NOT, parenthèses) : la fusion se fait alors
    sans filtre et MNT/densité filtrent eux-mêmes, comme avant."""
    import re
    jetons = re.compile(r"\s*(?:(?P<ou>OR\b)|(?P<et>AND\b)|(?P<non>NOT\b)|(?P<cmp><>|!=|<=|>=|==|=|<|>)"
                        r"|(?P<nb>-?\d+(?:\.\d+)?)|(?P<id>[A-Za-z_][A-Za-z0-9_]*)|(?P<par>[()]))", re.IGNORECASE)
    traduction = {"ou": "||", "et": "&&", "non": "!", "=": "==", "<>": "!="}
    sortie: List[str] = []
    pos, comparaisons = 0, 0
    while pos < len(expression_qgis):
        m = jetons.match(expression_qgis, pos)
        if m is None:
            return None if expression_qgis[pos:].strip() else _assembler(sortie, comparaisons)
        pos = m.end()
        genre = m.lastgroup
        valeur = m.group(genre)
        if genre == "cmp":
            comparaisons += 1
            sortie.append(traduction.get(valeur, valeur))
        else:
            sortie.append(traduction.get(genre, valeur))
    return _assembler(sortie, comparaisons)


def _assembler(jetons: List[str], comparaisons: int) -> Optional[str]:
    if not comparaisons:
        return None
    return " ".join(jetons).replace("( ", "(").replace(" )", ")")


def pipeline_fusion(central: Path, voisines: List[Tuple[Path, Dict[str, str]]], sortie: Path,
                    filtre: Optional[str] = None) -> Dict:
    """Pipeline PDAL de :func:`merge_tiles_flux` : la centrale, puis chaque voisine
    rognée à sa bande (bornes de :func:`calculate_crop_bounds`), toutes lues par
    ``readers.las`` — ordre du fichier — et écrites dans cet ordre par un seul
    ``writers.las``. Avec ``filtre`` (syntaxe PDAL) : les bornes X, Y du nuage
    complet sont relevées (``filters.stats``) puis seuls les points du filtre sont
    écrits, dans le même ordre."""
    etapes: List[Dict] = [{"type": "readers.las", "filename": str(central), "tag": "centrale"}]
    entrees = ["centrale"]
    for i, (source, b) in enumerate(voisines, start=1):
        etapes.append({"type": "readers.las", "filename": str(source), "tag": f"voisine{i}"})
        etapes.append({
            "type": "filters.crop",
            "bounds": f"([{b['xmin']},{b['xmax']}],[{b['ymin']},{b['ymax']}])",
            "inputs": [f"voisine{i}"],
            "tag": f"bande{i}",
        })
        entrees.append(f"bande{i}")
    if filtre:
        etapes.append({"type": "filters.stats", "dimensions": "X,Y", "inputs": entrees, "tag": "bornes"})
        etapes.append({"type": "filters.expression", "expression": filtre, "inputs": ["bornes"], "tag": "filtre"})
        entrees = ["filtre"]
    etapes.append({"type": "writers.las", "filename": str(sortie), "compression": "laszip", "inputs": entrees})
    return {"pipeline": etapes}


def _bornes_des_metadonnees(chemin: Path) -> Optional[List[float]]:
    """``[minx, maxx, miny, maxy]`` relevés par ``filters.stats`` dans les métadonnées du pipeline, ou None."""
    try:
        stats = json.loads(chemin.read_text(encoding="utf-8"))["stages"]["filters.stats"]["statistic"]
        par_dim = {s["name"]: s for s in stats}
        return [float(par_dim["X"]["minimum"]), float(par_dim["X"]["maximum"]),
                float(par_dim["Y"]["minimum"]), float(par_dim["Y"]["maximum"])]
    except (OSError, ValueError, KeyError, TypeError):
        return None


class _FiltreRefuse(RuntimeError):
    """PDAL refuse l'expression du filtre : la grammaire de :func:`expression_pdal` l'avait acceptée
    (faute de frappe dans un nom de dimension, par exemple). Ce n'est pas une voisine illisible."""


def _executer_fusion(
    central: Path, voisines: List[Tuple[Path, Dict[str, str]]], sortie: Path, *, log: LogFn, cancel: CancelFn,
    filtre: Optional[str] = None,
) -> Tuple[bool, Optional[List[float]]]:
    """Lance `pdal pipeline --stream` sur :func:`pipeline_fusion` : (réussi, bornes du nuage avant filtrage
    — None sans filtre ou si PDAL ne les a pas rendues). Lève :class:`_FiltreRefuse` si PDAL refuse
    ``filtre`` (son message est préfixé « filters.expression: »)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(pipeline_fusion(central, voisines, sortie, filtre), tf, indent=1)
        fichier = tf.name
    meta = Path(fichier).with_suffix(".meta.json")
    options = ["--metadata", str(meta)] if filtre else []
    try:
        cmd = [_pdal_exe(), "pipeline", "--stream", fichier, *options]
        result = run_pdal_command_cancellable(cmd, cancel=cancel)
        erreur = (result.stderr or "").lower()
        if result.returncode != 0 and ("stream mode" in erreur or "streamable" in erreur):
            # PDAL plus ancien que ceux mesurés (2.9 de QGIS 3.44, 2.10 de QGIS 4.0)
            # sans flux pour ce pipeline : même lecture ordonnée, en mémoire.
            log("PDAL : fusion en flux indisponible, fusion en mémoire")
            result = run_pdal_command_cancellable([cmd[0], "pipeline", fichier, *options], cancel=cancel)
        bornes = _bornes_des_metadonnees(meta) if filtre else None
    finally:
        for p in (Path(fichier), meta):
            try:
                p.unlink()
            except OSError:
                pass
    if result.returncode == 0:
        if filtre and bornes is None:
            log("⚠️ Bornes du nuage non relevées à la fusion : grille calée sur l'en-tête du fusionné filtré")
        return True, bornes
    if filtre and "filters.expression" in (result.stderr or ""):
        raise _FiltreRefuse((result.stderr or "").strip().splitlines()[0])
    log(f"Erreur PDAL pipeline (code {result.returncode})")
    if result.returncode in (3221225477, 3221226505):
        # 0xC0000005 = ACCESS_VIOLATION, 0xC0000409 = STACK_BUFFER_OVERRUN
        log("PDAL a crashé (erreur mémoire).")
    if result.stderr:
        log(result.stderr.strip())
    return False, None


def _empreinte(path: Path) -> Optional[List[int]]:
    """``[taille, mtime_ns]`` : dit qu'une voisine écartée a changé sans la relire."""
    try:
        st = path.stat()
    except OSError:
        return None
    return [st.st_size, st.st_mtime_ns]


def _sidecar_flux(voisins: List[str], ecartees: Dict[str, Optional[List[int]]], filtre: Optional[str],
                  bornes: Optional[List[float]] = None) -> Dict:
    """Contenu du sidecar de :func:`merge_tiles_flux` (``ecartees`` et ``bornes`` seulement s'il y en a).
    ``bornes`` = X, Y du nuage AVANT filtrage : c'est là que MNT et densité calent leur grille."""
    contenu: Dict = {"fusion": FUSION_FLUX, "voisins": voisins, "filtre": filtre}
    if ecartees:
        contenu["ecartees"] = ecartees
    if bornes:
        contenu["bornes"] = bornes
    return contenu


def merge_tiles_flux(
    *,
    central_path: Path,
    voisines: List[Tuple[Path, Dict[str, str]]],
    output_path: Path,
    log: LogFn = _default_log,
    cancel: CancelFn = _default_cancel,
    filtre: Optional[str] = None,
) -> bool:
    """Fusionne la dalle centrale et la bande de marge de chaque voisine en UN
    `pdal pipeline --stream` (lot 1, 2026-10), en ne gardant que les points de
    ``filtre`` (syntaxe PDAL, :func:`expression_pdal`) s'il est donné.

    Tout est lu par ``readers.las``, dans l'ordre du fichier : le LAZ fusionné est
    identique d'un run à l'autre (même sha256 que ``pdal merge --driver
    readers.las``), là où ``pdal merge`` lisait la centrale COPC sur plusieurs fils
    dans un ordre variable — d'où des MNT différents entre deux runs identiques. En
    flux, la mémoire reste celle d'un tampon de points, pas du nuage entier. Même
    écriture atomique (``.partial`` validé puis ``os.replace``) que
    :func:`merge_tiles`, qui reste la fusion de ``ARCHEO_FUSION_LEGACY=1``.

    ``voisines`` : ``(LAZ source, bornes de sa bande)`` dans l'ordre de
    :func:`calculate_neighbor_coordinates`. Le sidecar enregistre la méthode, les
    voisines candidates et, s'il y en a, les voisines écartées (illisibles) avec leur
    ``[taille, mtime_ns]`` : un fusionné d'une autre méthode (sidecar « liste » des
    versions ≤ 0.14.1, ou absent) est refait, jamais adopté ; une voisine écartée
    n'est pas relue tant qu'elle n'a pas changé sur disque, et refusionnée dès
    qu'elle a été réparée ou remplacée.
    """
    if cancel():
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    voisins = sorted(p.name for p, _ in voisines)
    sidecar = merged_inputs_sidecar(output_path)

    if output_path.exists():
        ok, _ = validate_las_or_laz_with_pdal(output_path)
        try:
            enregistre = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            enregistre = None
        avant = enregistre.get("ecartees") if isinstance(enregistre, dict) else None
        par_nom = {p.name: p for p, _ in voisines}
        actuel = _sidecar_flux(voisins, {
            nom: _empreinte(par_nom[nom]) for nom in avant if nom in par_nom
        } if isinstance(avant, dict) else {}, filtre)
        sans_bornes = {k: v for k, v in enregistre.items() if k != "bornes"} if isinstance(enregistre, dict) else None
        if ok and sans_bornes == actuel:
            log(f"LAZ fusionné réutilisé (cache intermédiaire) : {output_path.name}")
            return True
        log(f"Fusion d'une version précédente ou voisines modifiées → re-fusion : {output_path.name}")
        try:
            output_path.unlink()
        except OSError:
            pass

    ok_c, msg_c = validate_las_or_laz_with_pdal(central_path)
    if not ok_c:
        log(f"Fichier central invalide via PDAL: {central_path.name}")
        if msg_c:
            log(f"PDAL: {msg_c}")
        return False

    valides: List[Tuple[Path, Dict[str, str]]] = []
    ecartees: Dict[str, Optional[List[int]]] = {}
    for source, bornes in voisines:
        if cancel():
            return False
        if validate_las_or_laz_with_pdal(source)[0]:
            valides.append((source, bornes))
        else:
            log(f"Voisine illisible via PDAL, ignorée : {source.name}")
            ecartees[source.name] = _empreinte(source)

    partial = _partial_path(output_path)
    try:
        try:
            ok, bornes = _executer_fusion(central_path, valides, partial, log=log, cancel=cancel, filtre=filtre)
        except _FiltreRefuse as e:
            log(f"⚠️ Filtre des points refusé par PDAL ({filtre}) : fusion sans filtre, "
                f"le MNT et la densité filtrent eux-mêmes (plus lent) — {e}")
            filtre = None
            ok, bornes = _executer_fusion(central_path, valides, partial, log=log, cancel=cancel)
        if not ok and valides and not cancel():
            # Une voisine à l'en-tête sain mais aux points illisibles (LAZ tronqué)
            # fait échouer tout le pipeline : on l'isole par une lecture complète et
            # on refusionne sans elle — comme l'ancienne fusion, où seul son
            # rognage échouait.
            saines: List[Tuple[Path, Dict[str, str]]] = []
            for s, b in valides:
                if cancel():        # une relecture complète dure 14 à 36 s par voisine : l'annulation n'attend pas la fin
                    return False
                if validate_laz_deep(s)[0]:
                    saines.append((s, b))
                else:
                    ecartees[s.name] = _empreinte(s)
            if len(saines) < len(valides):
                log(f"{len(valides) - len(saines)} voisine(s) illisible(s) écartée(s) → nouvelle fusion")
                ok, bornes = _executer_fusion(central_path, saines, partial, log=log, cancel=cancel, filtre=filtre)
        if not ok:
            return False
        ok_out, msg_out = validate_las_or_laz_with_pdal(partial)
        if not ok_out:
            log(f"Fichier fusionné invalide via PDAL: {output_path.name}")
            if msg_out:
                log(f"PDAL: {msg_out}")
            return False
        os.replace(str(partial), str(output_path))
    finally:
        _discard_partial(partial)

    sidecar.write_text(json.dumps(_sidecar_flux(voisins, ecartees, filtre, bornes)), encoding="utf-8")
    return True


def _fusion_ancienne(
    task: "_PreprocessTask",
    voisines: List[Tuple[Path, Dict[str, str], int]],
    merged_path: Path,
    log: LogFn,
    cancel: CancelFn,
) -> bool:
    """Fusion d'avant le lot 1 (bandes rognées écrites, puis `pdal merge`), gardée
    une version derrière ``ARCHEO_FUSION_LEGACY=1``."""
    rognees: List[Path] = []
    for source, bornes, place in voisines:
        if cancel():
            return False
        sortie = task.temp_dir / f"{task.tile_name}_neighbor_{place}.laz"
        if crop_neighbor_tile(input_path=source, output_path=sortie, bounds=bornes, log=log, cancel=cancel):
            rognees.append(sortie)
    return merge_tiles(
        central_path=task.central_path, neighbor_paths=rognees, output_path=merged_path, log=log, cancel=cancel,
    )


@dataclass
class _PreprocessTask:
    """Tâche de prétraitement pour une dalle."""
    index: int
    total: int
    filename: str
    url: str
    tile_name: str
    x: str
    y: str
    central_path: Path
    temp_dir: Path
    merged_dir: Path
    margin_m: int
    dalles_dir: Path
    # Proxy (QGIS ou env) résolu UNE fois par prepare_merged_tiles : les
    # re-téléchargements de voisins doivent passer par le même proxy que
    # l'acquisition (AUDIT v2 NET-06 — réseau d'entreprise).
    proxies: Dict[str, str] = field(default_factory=dict)
    # Filtre des points de l'étape 2, déjà en syntaxe PDAL (traduit UNE fois par prepare_merged_tiles) :
    # le fusionné ne garde que ces classes. None = nuage entier.
    filtre: Optional[str] = None


@dataclass
class _PreprocessResult:
    """Résultat du prétraitement d'une dalle."""
    index: int
    tile_name: str
    merged_path: Optional[Path]
    success: bool
    error: Optional[str] = None


def _process_single_tile_preprocess(
    task: _PreprocessTask,
    file_index: Dict[str, Tuple[str, str]],
    log: LogFn,
    cancel: CancelFn,
) -> _PreprocessResult:
    """
    Traite le prétraitement d'une seule dalle (voisines + fusion).
    Utilise uniquement PDAL (subprocess externe) → thread-safe.
    """
    def _annule() -> _PreprocessResult:
        return _PreprocessResult(
            index=task.index, tile_name=task.tile_name, merged_path=None,
            success=False, error="Annulation demandée",
        )

    def tlog(m: str) -> None:
        log(f"[{task.tile_name}] {m}")

    try:
        if cancel():
            return _annule()

        log(f"[Dalle {task.index}/{task.total}] Début prétraitement: {task.tile_name}")

        # Voisines présentes (téléchargées au besoin), dans l'ordre de
        # calculate_neighbor_coordinates, avec la bande de marge à garder.
        voisines: List[Tuple[Path, Dict[str, str], int]] = []
        for voisin_x, voisin_y, place_dalle in calculate_neighbor_coordinates(task.x, task.y):
            if cancel():
                return _annule()

            # Recherche rapide via l'index dict
            coord_key = f"{format_coordinate(voisin_x)}_{format_coordinate(voisin_y)}"
            neigh = file_index.get(coord_key)
            if not neigh:
                continue

            neighbor_file, neighbor_url = neigh
            neighbor_input = task.dalles_dir / neighbor_file
            if not neighbor_input.exists():
                ok, _was_skipped = download_one(
                    neighbor_url,
                    neighbor_file,
                    task.dalles_dir,
                    log=tlog,
                    cancel=cancel,
                    proxies=task.proxies,
                )
                if not ok or not neighbor_input.exists():
                    continue

            voisines.append((
                neighbor_input,
                calculate_crop_bounds(voisin_x, voisin_y, place_dalle, task.margin_m),
                place_dalle,
            ))

        merged_path = task.merged_dir / f"{task.tile_name}_merged.laz"
        if fusion_mode() == FUSION_ANCIENNE:
            ok = _fusion_ancienne(task, voisines, merged_path, tlog, cancel)
        else:
            ok = merge_tiles_flux(
                central_path=task.central_path,
                voisines=[(source, bornes) for source, bornes, _place in voisines],
                output_path=merged_path,
                log=tlog,
                cancel=cancel,
                filtre=task.filtre,
            )
        if not ok:
            if cancel():
                return _annule()
            return _PreprocessResult(
                index=task.index,
                tile_name=task.tile_name,
                merged_path=None,
                success=False,
                error=f"Échec merge pour {task.tile_name}",
            )

        log(f"[Dalle {task.index}/{task.total}] Prétraitement terminé: {task.tile_name}")

        return _PreprocessResult(
            index=task.index,
            tile_name=task.tile_name,
            merged_path=merged_path,
            success=True,
        )

    except Exception as e:
        return _PreprocessResult(
            index=task.index,
            tile_name=task.tile_name,
            merged_path=None,
            success=False,
            error=str(e),
        )


def _build_file_index(sorted_list_file: Path) -> Dict[str, Tuple[str, str]]:
    """
    Construit un index dict {coord_key: (filename, url)} pour recherche O(1).
    coord_key = "XXXX_YYYY" (ex: "0948_6633")
    """
    index: Dict[str, Tuple[str, str]] = {}
    try:
        with sorted_list_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",", 1)
                if len(parts) != 2:
                    continue
                filename, url = parts[0].strip(), parts[1].strip()
                # Extraire les coordonnées du nom de fichier
                # Format attendu: LHD_FXX_XXXX_YYYY_...
                name_parts = filename.replace(".copc.laz", "").replace(".laz", "").split("_")
                if len(name_parts) >= 4:
                    coord_key = f"{name_parts[2]}_{name_parts[3]}"
                    index[coord_key] = (filename, url)
    except Exception:
        pass
    return index


StageFn = Callable[[str], None]


def _default_stage(_: str) -> None:
    return


def prepare_merged_tiles(
    *,
    sorted_list_file: Path,
    dalles_dir: Path,
    output_dir: Path,
    tile_overlap_percent: float,
    log: LogFn = _default_log,
    cancel: CancelFn = _default_cancel,
    max_workers: Optional[int] = None,
    stage: StageFn = _default_stage,
    on_tile_merged: Optional[Callable[[int, int, str], None]] = None,
    filter_expression: str = "",
) -> IgnPreprocessResult:
    """Fusionne chaque dalle avec ses voisines (préfixage des bords).

    ``on_tile_merged`` (optionnel) est invoqué après chaque dalle
    finalisée avec ``(completed_index_1based, total, tile_name)``. Le
    caller (runner) s'en sert pour remonter la sous-progression à
    l'utilisateur (ligne réécrite dans le journal). L'exécution étant
    parallèle, ``completed_index`` reflète l'ordre d'arrivée, pas
    l'ordre original des dalles.
    """
    from ..output_paths import intermediaires_dir
    temp_dir = intermediaires_dir(output_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = temp_dir

    margin_m = max(0, min(999, int(round(1000.0 * float(tile_overlap_percent) / 100.0))))

    # Le filtre de l'étape 2 traduit UNE fois pour tout le lot : un filtre non reconnu
    # s'annonce une fois, pas une fois par dalle.
    filtre = expression_pdal(filter_expression) if filter_expression else None
    if filter_expression and filtre is None:
        log(f"⚠️ Filtre des points non reconnu ({filter_expression}) : fusion sans filtre, "
            "le MNT et la densité filtrent eux-mêmes (plus lent)")

    # Construire l'index pour recherche rapide des voisins
    file_index = _build_file_index(sorted_list_file)

    with sorted_list_file.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    total = len(lines)
    failed: List[str] = []

    # Proxy résolu UNE fois (même source que l'acquisition : QGIS puis env) —
    # transporté par chaque tâche jusqu'aux download_one des voisins
    # (AUDIT v2 NET-06).
    try:
        proxies = _get_proxy_config(log=log)
    except Exception:  # noqa: BLE001 — best-effort, jamais bloquant
        proxies = {}

    # Préparer les tâches
    tasks: List[_PreprocessTask] = []
    for idx, line in enumerate(lines, start=1):
        parts = line.split(",", 1)
        if len(parts) != 2:
            continue

        filename, url = parts[0].strip(), parts[1].strip()
        base_tile_name = filename.replace(".copc.laz", "").replace(".laz", "")

        central_path = dalles_dir / filename
        if not central_path.exists():
            # Dalle listée dans fichier_tri.txt mais absente du disque
            # (download échoué, nom parsable — ROB-08 v1) : on continue avec
            # les autres dalles au lieu d'avorter tout le lot par un
            # FileNotFoundError fatal (AUDIT v2 ROB-12).
            log(f"⚠️ Dalle absente du disque, ignorée pour la fusion : {filename}")
            failed.append(f"{filename}: fichier central introuvable")
            continue

        x, y = _extract_coordinates(filename, dalles_dir=dalles_dir, log=log, cancel=cancel)
        tile_name = base_tile_name
        if len(base_tile_name.split("_")) < 4:
            tile_name = f"LHD_FXX_{x}_{y}"

        tasks.append(_PreprocessTask(
            index=idx,
            total=total,
            filename=filename,
            url=url,
            tile_name=tile_name,
            x=x,
            y=y,
            central_path=central_path,
            temp_dir=temp_dir,
            merged_dir=merged_dir,
            margin_m=margin_m,
            dalles_dir=dalles_dir,
            proxies=proxies,
            filtre=filtre,
        ))

    # Configuration parallélisation
    # PDAL est un subprocess externe → thread-safe
    if max_workers is None:
        max_workers = min(4, max(1, os.cpu_count() or 1))
    
    log(f"Prétraitement parallèle: {max_workers} worker(s) pour {total} dalle(s)")

    # Lock pour synchroniser les logs et le compteur
    log_lock = threading.Lock()
    completed_count = [0]
    
    def thread_safe_log(msg: str) -> None:
        with log_lock:
            log(msg)

    def update_stage_count(tile_name: str = "") -> None:
        with log_lock:
            completed_count[0] += 1
            current = completed_count[0]
            stage(f"Fusion dalle {current}/{total}")
        if on_tile_merged is not None:
            try:
                on_tile_merged(current, total, tile_name)
            except Exception:
                pass

    # Exécution parallèle
    merged_files: List[Path] = []
    results_by_index: Dict[int, _PreprocessResult] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                _process_single_tile_preprocess,
                task,
                file_index,
                thread_safe_log,
                cancel,
            ): task
            for task in tasks
        }

        for future in as_completed(future_to_task):
            if cancel():
                executor.shutdown(wait=False, cancel_futures=True)
                break

            task = future_to_task[future]
            try:
                result = future.result()
                results_by_index[result.index] = result
                update_stage_count(result.tile_name)

                if not result.success:
                    thread_safe_log(f"Échec prétraitement dalle {result.tile_name}: {result.error}")
            except Exception as e:
                thread_safe_log(f"Exception prétraitement dalle {task.tile_name}: {e}")
                failed.append(f"{task.tile_name}: {e}")

    # Reconstituer la liste dans l'ordre original. Isolation par dalle
    # (AUDIT v2 ROB-12) : un merge en échec n'est plus re-levé en
    # RuntimeError fatal — on continue avec les dalles fusionnées et on
    # rapporte les échecs à l'appelant via ``failed``.
    for idx in sorted(results_by_index.keys()):
        result = results_by_index[idx]
        if result.success and result.merged_path is not None:
            merged_files.append(result.merged_path)
        elif result.error and "Annulation" not in (result.error or ""):
            failed.append(f"{result.tile_name}: {result.error}")

    # Échec TOTAL (aucune dalle fusionnée alors qu'il y avait des échecs et
    # pas d'annulation) : fatal — rien à traiter en aval.
    if failed and not merged_files and not cancel():
        raise RuntimeError(
            f"Fusion impossible : {len(failed)} dalle(s) en échec, aucune "
            f"fusionnée. Première erreur : {failed[0]}"
        )

    return IgnPreprocessResult(
        merged_dir=merged_dir, temp_dir=temp_dir,
        merged_files=merged_files, failed=failed,
    )


def _extract_coordinates(
    filename: str,
    *,
    dalles_dir: Path,
    log: LogFn = _default_log,
    cancel: CancelFn = _default_cancel,
) -> Tuple[str, str]:
    inferred = infer_xy_from_file(dalles_dir / filename, cancel=cancel)
    if inferred is not None:
        x_str = f"{int(inferred.x_km):04d}"
        y_str = f"{int(inferred.y_km):04d}"
        log(f"Coordonnées inférées via metadata pour {filename}: x={x_str}, y={y_str}")
        return x_str, y_str

    xy = extract_xy_from_filename(filename)
    if xy is not None:
        return f"{int(xy.x_km):04d}", f"{int(xy.y_km):04d}"

    raise ValueError(f"Impossible d'extraire / inférer les coordonnées de: {filename}")
