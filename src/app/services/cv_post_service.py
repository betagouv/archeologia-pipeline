"""Service partagé pour le lancement de la Computer Vision post-traitement.

Avant ce module, la boucle « pour chaque run CV configuré, résoudre le
RVT/TIF correspondant et lancer ``run_existing_rvt`` sur ce dossier »
était dupliquée presque mot pour mot entre :

- ``IgnOrLocalRunner._run_post_cv`` (ign_laz / local_laz)
- ``ExistingMntRunner.run`` (inline, mode existing_mnt)

Cela imposait de modifier la même logique à 2 endroits pour la moindre
évolution. ``ExistingRvtRunner`` n'est *pas* concerné : il consomme un
RVT externe fourni par l'utilisateur, pas un RVT généré par le
pipeline, et a donc une logique différente.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from ..progress_reporter import report_busy, report_stage_id
from ..progress_stages import Stage
from ..runners.progress_plan import cv_pct
from ..structured_logger import log_section
from ..user_narrator import create_user_narrator
from .carte_graphique import enregistrer as enregistrer_appareil, preparer_run as preparer_appareil, registre_path


def _model_display_name(selected_model: str) -> str:
    """Nom court d'un modèle à partir de la valeur ``selected_model``.

    Accepte aussi bien un nom de dossier (``mon_modele``) qu'un chemin
    complet vers ``best.onnx`` (``…/models/mon_modele/weights/best.onnx``).
    Le ``display_name`` du ``model_card.yaml`` quand il existe (« Cratères
    d'obus (LD) »), sinon le nom du dossier : le journal de l'assistant nommait les
    modèles par leur identifiant (recette 0.14.0).
    """
    from .model_orchestrator import load_model_card

    p = Path(selected_model)
    if p.suffix.lower() == ".onnx" and p.parent.name == "weights":
        dossier = p.parent.parent
    elif p.is_absolute():
        dossier = p
    else:
        dossier = Path(__file__).resolve().parents[3] / "data" / "models" / selected_model
    nom = dossier.name or selected_model
    card = load_model_card(dossier) if dossier.is_dir() else None
    return str((card or {}).get("display_name") or nom)


def noter_echec_detection(
    stats: Optional[list], run_cfg: Dict[str, Any], model_display: str, exc: Exception
) -> None:
    """Trace dans ``stats`` un run sans détection possible (moteur absent ou en
    échec, sans repli) : la finalisation en tire ``cv_engine = "none"``, une fin de
    run non ✓ et le motif dans traitement.json et rapport.html."""
    try:  # fallback standalone (tests : src/ sur le path)
        from ...pipeline.cancellation import DetectionImpossible
    except ImportError:  # pragma: no cover
        from pipeline.cancellation import DetectionImpossible
    from .rapport_run import anonymiser

    if stats is None or not isinstance(exc, DetectionImpossible):
        return
    stats.append({
        "modele": model_display, "model": run_cfg.get("selected_model", "?"),
        "target_rvt": run_cfg.get("target_rvt", ""),
        "entites": [e.get("label") or e.get("slug") or "" for e in (run_cfg.get("entities") or [])
                    if isinstance(e, dict)],
        "images": 0, "secondes": 0.0, "moteur": "none", "motif": anonymiser(str(exc)),
    })

if TYPE_CHECKING:
    from ..cancel_token import CancelToken
    from ..progress_reporter import ProgressReporter
    from ..run_context import RunContext
    from ..structured_logger import StructuredLogger


#: Tolérance (m) sur l'inclusion de la cellule 1 km dans l'emprise candidate.
_CELL_TOLERANCE_M = 1.0


def resolve_uncropped_tif(
    cropped_tif: Path,
    temp_dir: Optional[Path],
    target_rvt: str,
    rvt_params: Optional[Dict[str, Any]] = None,
    *,
    bounds_fn: Optional[Callable[[Path], Optional[Tuple[float, float, float, float]]]] = None,
) -> Optional[Path]:
    """TIF source non rogné (avec marge) correspondant à un TIF rogné 1 km.

    Option B (halo inter-dalles) : le RVT est calculé sur l'emprise étendue
    (dalle + ``tile_overlap``) puis rogné à la cellule 1 km exacte. Le TIF
    étendu survit dans ``intermediaires/`` — l'inférence CV peut donc tourner
    dessus pour voir les objets à cheval sur les frontières de dalles en
    entier. Appariement : même produit + mêmes paramètres RVT (queue de nom
    de :func:`get_rvt_temp_filename`) et emprise couvrant la cellule 1 km
    déduite du nom rogné (``LHD_FXX_{x}_{y}_…``, ``y`` = bord nord).

    Renvoie ``None`` faute de correspondance sûre — l'appelant retombe alors
    sur le TIF rogné (comportement historique, détections coupées).
    """
    if temp_dir is None or not Path(temp_dir).is_dir():
        return None
    try:  # fallback standalone (tests : src/ sur le path)
        from ...pipeline.coords import extract_xy_from_filename, get_raster_bounds
        from ...pipeline.ign.products.rvt_naming import get_rvt_temp_filename
    except ImportError:  # pragma: no cover
        from pipeline.coords import extract_xy_from_filename, get_raster_bounds
        from pipeline.ign.products.rvt_naming import get_rvt_temp_filename

    xy = extract_xy_from_filename(Path(cropped_tif).name)
    if xy is None:
        return None
    cx0, cx1 = xy.x_km * 1000.0, (xy.x_km + 1) * 1000.0
    cy0, cy1 = (xy.y_km - 1) * 1000.0, xy.y_km * 1000.0

    read_bounds = bounds_fn or get_raster_bounds
    tail = get_rvt_temp_filename(target_rvt, "", rvt_params or {})
    tol = _CELL_TOLERANCE_M
    for cand in sorted(Path(temp_dir).glob(f"*{tail}")):
        b = read_bounds(cand)
        if b is None:
            continue
        if b[0] <= cx0 + tol and b[1] <= cy0 + tol and b[2] >= cx1 - tol and b[3] >= cy1 - tol:
            return cand
    return None


def annoncer_echelle(cv_cfg: Dict[str, Any], cv_runs: List[Dict[str, Any]], reporter: "ProgressReporter",
                     rvt_params_px: Optional[Dict[str, Any]], *, indices_calcules: bool = True,
                     anneau_fourni_px: Optional[Tuple[int, int]] = None) -> None:
    """Avertissements d'échelle du run (``model_orchestrator.avertissements_echelle``), un par cause, modèles listés :
    résolution des données ≠ résolution d'entraînement, anneau du LD différent, indices fournis aux réglages
    inconnus. Best-effort : un modèle illisible ne bloque jamais la détection."""
    from .indices_metres import resolution_de
    from .model_orchestrator import avertissements_echelle, discover_installed_models
    try:
        racine = Path(str((cv_cfg or {}).get("models_dir") or "data/models"))
        if not racine.is_absolute():
            racine = Path(__file__).resolve().parents[3] / racine
        noms = {Path(str(r.get("selected_model") or r.get("model") or "")).name for r in cv_runs}
        modeles = [m for m in discover_installed_models(racine) if m.name in noms]
        for msg in avertissements_echelle(modeles, resolution_de(rvt_params_px), rvt_params_px,
                                          indices_calcules=indices_calcules, anneau_fourni_px=anneau_fourni_px):
            reporter.user_warning(msg)
    except Exception as e:  # noqa: BLE001
        reporter.info(f"Avertissements d'échelle ignorés ({e})")


def avertissements_remplissage(stats: List[Dict[str, Any]]) -> List[str]:
    """Avertissement VISIBLE (un message pour la cause, modèles listés, comme les avertissements d'échelle) quand
    des images occupent moins de la moitié de la fenêtre d'analyse d'un modèle : le reste est complété par du vide,
    un cas que le modèle n'a pas vu à l'entraînement. ``stats`` : entrées ``{modele, remplissage}`` des runs
    (``remplissage`` = ``runner.remplissage_des_fenetres``) ; le détail par image reste au journal technique."""
    from .indices_metres import format_m

    def _cote(r: Dict[str, Any]) -> str:
        return f"{format_m(r['fenetre_m'])} de côté" if r.get("fenetre_m") else f"{r['fenetre_px']} px de côté"

    lus = [(str(s.get("modele") or s.get("model") or "?"), s["remplissage"]) for s in stats or []
           if isinstance(s, dict) and isinstance(s.get("remplissage"), dict)]
    if not lus:
        return []
    grand = max(lus, key=lambda t: (t[1].get("fenetre_m") or 0, t[1]["fenetre_px"]))[1]
    conseil = (f"Fournissez des MNT d'au moins {format_m(grand['fenetre_m'])} de côté." if grand.get("fenetre_m")
               else f"Fournissez des rasters d'au moins {grand['fenetre_px']} px de côté.")
    if len(lus) == 1:
        nom, r = lus[0]
        return [f"{r['n']} images sur {r['total']} sont plus petites que la fenêtre d'analyse du modèle « {nom} » "
                f"({_cote(r)}) : l'image n'en occupe en moyenne que {round(100 * r['part_moyenne'])} %, le reste est "
                "complété par du vide, ce que le modèle n'a pas vu à l'entraînement : relisez ses détections avec "
                f"prudence. {conseil}"]
    detail = " ; ".join(f"« {nom} » ({_cote(r)}) : {r['n']} images sur {r['total']}, "
                        f"{round(100 * r['part_moyenne'])} % en moyenne" for nom, r in lus)
    return [f"Des images sont plus petites que la fenêtre d'analyse de {len(lus)} modèles — {detail}. L'image "
            "n'occupe qu'une partie de la fenêtre, le reste est complété par du vide, ce que ces modèles n'ont pas vu "
            f"à l'entraînement : relisez leurs détections avec prudence. {conseil}"]


def run_cv_post_loop(
    *,
    ctx: "RunContext",
    output_structure: Dict[str, Any],
    rvt_params: Dict[str, Any],
    reporter: "ProgressReporter",
    cancel: "CancelToken",
    slog: Optional["StructuredLogger"],
    cv_band: Tuple[int, int] = (90, 95),
    halo_source_dir: Optional[Path] = None,
    stats_out: Optional[list] = None,
    crs: Optional[str] = None,
) -> None:
    """Lance la Computer Vision sur les RVT générés par le pipeline.

    Pour chaque ``cv_run`` configuré dans ``ctx.cv.runs`` :

    1. Résout ``indices/<RVT>/tif`` via :func:`resolve_rvt_tif_dir`.
    2. Construit (au premier run uniquement) le mapping global de
       couleurs partagé par tous les runs.
    3. Délègue à :func:`pipeline.modes.existing_rvt.run_existing_rvt`.

    Args:
        cv_band: Bande de progression ``(lo, hi)`` allouée à la CV par le
            :class:`~app.runners.progress_plan.ProgressPlan` du mode. La barre
            démarre à ``lo`` et progresse jusqu'à ``hi`` au fil des images
            (réparties équitablement entre les runs via :func:`cv_pct`) — plus
            de recul de la barre au démarrage de la CV.
        halo_source_dir: Dossier des TIF non rognés (``intermediaires/``).
            Fourni par les modes à marge inter-dalles (ign_laz/local_laz) :
            l'inférence tourne alors sur l'image AVEC halo via
            :func:`resolve_uncropped_tif` (option B — détections non coupées
            aux frontières de dalles). ``None`` (existing_mnt) = comportement
            historique sur TIF rognés.
        crs: CRS du run (``ingest_plan.crs_des_entrees``) des modes à rasters fournis,
            transmis à ``run_existing_rvt`` ; ``None`` (modes LiDAR) = EPSG:2154.

    Cette fonction n'attrape pas les exceptions : l'appelant décide de
    sa politique (le pattern actuel est de logger via ``reporter.error``
    et de continuer vers la finalisation).
    """
    try:  # fallback standalone (tests : src/ sur le path)
        from ...pipeline.batch import process_items_isolated
        from ...pipeline.cv.class_utils import resolve_cv_runs
        from ...pipeline.modes.existing_rvt import run_existing_rvt
        from ...pipeline.output_paths import detections_gardees, resolve_rvt_tif_dir
    except ImportError:  # pragma: no cover
        from pipeline.batch import process_items_isolated
        from pipeline.cv.class_utils import resolve_cv_runs
        from pipeline.modes.existing_rvt import run_existing_rvt
        from pipeline.output_paths import detections_gardees, resolve_rvt_tif_dir

    cv_cfg = ctx.cv.raw
    cv_runs = resolve_cv_runs(cv_cfg)
    if not cv_runs:
        reporter.info("Computer Vision: aucun modèle configuré dans les runs")
        return

    # La couleur dérive du nom de classe via le registre partagé (refonte
    # couleurs) ; plus de mapping global à pré-calculer. Paramètre conservé
    # (vide) dans la chaîne d'appel pour ne pas casser les signatures.
    global_color_map: Dict[str, int] = {}

    log_section("COMPUTER VISION", "cv", slog=slog, reporter=reporter)
    report_stage_id(reporter, Stage.DETECTION)
    reporter.stage("Computer Vision")
    reporter.progress(cv_band[0])

    narrator = create_user_narrator(reporter)
    narrator.cv_start(len(cv_runs))
    annoncer_echelle(cv_cfg, cv_runs, reporter, rvt_params)

    # total_detections par run (résumé du runner) — None exclus : si aucun
    # run n'a de résumé (fallback, vieux binaire), pas d'annonce de total.
    detection_counts: list = []
    remplissages: list = []   # fenêtres surtout vides, par modèle → un avertissement visible après les runs

    def _process_run(run_idx: int, run_cfg: Dict[str, Any]) -> None:
        run_model = run_cfg.get("selected_model", "?")
        run_rvt = run_cfg.get("target_rvt", "LD")
        model_display = _model_display_name(run_model)
        reporter.info(
            f"Computer Vision: run {run_idx}/{len(cv_runs)} — "
            f"modèle={run_model}, RVT={run_rvt}"
        )
        narrator.cv_run_start(run_idx, len(cv_runs), model_display, run_rvt)
        # Carte graphique (étape 4, décochée par défaut) : bloc posé dans la config du
        # run (sonde à faire, ou carte déjà retenue sur ce poste) ; rien à sonder → dit.
        run_cfg, cg_ctx = preparer_appareil(run_cfg, bool((ctx.cv.raw or {}).get("carte_graphique")))
        if cg_ctx["message"]:
            narrator.cv_appareil(model_display, {"appareil": "processeur", "message": cg_ctx["message"]})

        generated_rvt_tif_dir = resolve_rvt_tif_dir(
            ctx.output_dir, run_rvt, output_structure, rvt_params
        )

        if not generated_rvt_tif_dir.exists() or not generated_rvt_tif_dir.is_dir():
            reporter.error(
                f"Computer Vision: dossier RVT/TIF non trouvé pour {run_rvt}: "
                f"{generated_rvt_tif_dir}"
            )
            return

        # Callback bindé au modèle courant : remonte la sous-progression
        # (index/total images traitées) au narrator → ligne USER_INFO
        # discrète dans le journal, ET fait avancer la barre dans la bande CV
        # (répartie entre les runs). ``model_display``/``run_idx``/``n_runs``
        # sont figés par défaut-d'argument pour éviter le late-binding
        # classique en boucle Python.
        def _on_image_progress(
            idx, total, image_name,
            _model=model_display, _ri=run_idx, _n=len(cv_runs),
        ):
            narrator.cv_run_image_progress(_model, idx, total, image_name)
            reporter.progress(cv_pct(_ri, _n, idx, total, cv_band))

        # Option B (halo inter-dalles) : chaque TIF rogné est ré-associé à son
        # TIF non rogné (marge = donnée voisine réelle) pour l'inférence.
        inference_tif_resolver = None
        if halo_source_dir is not None:
            def inference_tif_resolver(
                cropped, _rvt=run_rvt, _src=halo_source_dir,
            ):
                return resolve_uncropped_tif(cropped, _src, _rvt, rvt_params)

        res = run_existing_rvt(
            existing_rvt_dir=generated_rvt_tif_dir,
            output_dir=ctx.output_dir,
            cv_config=run_cfg,
            output_structure=output_structure,
            log=lambda m: reporter.info(m),
            cancel_check=cancel.is_cancelled,
            rvt_params=rvt_params,
            global_color_map=global_color_map,
            image_progress=_on_image_progress,
            tile_progress=narrator.cv_run_tile_progress,
            on_busy=lambda active: report_busy(reporter, active),
            inference_tif_resolver=inference_tif_resolver,
            max_workers=ctx.processing.max_workers,
            on_appareil=lambda d, _m=model_display: narrator.cv_appareil(_m, d),
            crs=crs,
        )
        if res.total_detections is not None:
            detection_counts.append(res.total_detections)
        st = getattr(res, "cv_stats", None) or {}
        if cg_ctx["cle"]:   # décision de la sonde mémorisée pour ce poste, ce modèle, ce binaire
            enregistrer_appareil(st, cg_ctx["cle"], registre_path())
        appareil = st.get("appareil") if isinstance(st.get("appareil"), dict) else {}
        if st.get("images_inferees"):
            narrator.cv_run_done(
                model_display, int(st["images_inferees"]), float(st.get("secondes", 0.0)),
                appareil=appareil.get("appareil"),
            )
        remplissages.append({"modele": model_display, "remplissage": st.get("remplissage")})
        if stats_out is not None:   # durée mesurée par modèle, pour le rapport de run
            stats_out.append({
                "modele": model_display, "model": run_model, "target_rvt": run_rvt,
                "entites": [e.get("label") or e.get("slug") or "" for e in (run_cfg.get("entities") or [])
                            if isinstance(e, dict)],
                "images": int(st.get("images_inferees") or 0), "secondes": float(st.get("secondes") or 0.0),
                "moteur": st.get("moteur"), "appareil": appareil.get("appareil"), "carte": appareil.get("carte"),
                "repli": bool(st.get("appareil_repli")),
            })

    def _on_run_failure(run_idx: int, run_cfg: Dict[str, Any], exc: Exception) -> None:
        model_display = _model_display_name(run_cfg.get("selected_model", "?"))
        reporter.error(
            f"Computer Vision: run {run_idx} (modèle={model_display}) en échec, "
            f"ignoré : {exc}"
        )
        noter_echec_detection(stats_out, run_cfg, model_display, exc)

    # Isolation par run : l'échec d'inférence/conversion d'un modèle n'empêche
    # plus les runs suivants de produire leurs détections (cf. AUDIT ROB-04).
    # PipelineCancelled reste propagée (annulation globale gérée par l'appelant).
    _ok, failed = process_items_isolated(
        cv_runs,
        _process_run,
        cancel=cancel.is_cancelled,
        on_failure=_on_run_failure,
    )
    if failed:
        reporter.error(
            f"⚠️ Computer Vision: {len(failed)} run(s) sur {len(cv_runs)} en échec "
            "— voir le journal."
        )
    for msg in avertissements_remplissage(remplissages):
        reporter.user_warning(msg)

    # Compte des couches (après dédoublonnage, seuil, aire minimale), pas les détections brutes du moteur.
    gardees = detections_gardees(ctx.output_dir, cv_runs)
    if gardees is not None:
        narrator.cv_complete(gardees)
    elif detection_counts:
        narrator.cv_complete(sum(detection_counts))
