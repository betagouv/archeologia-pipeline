#!/usr/bin/env python3
"""Rendu hors écran de l'interface — captures et contrôles sans QGIS de bureau.

À lancer avec le **Python de QGIS** (le plugin importe ``qgis.PyQt``) :

    C:/OSGeo4W/bin/python-qgis.bat dev/rendu_hors_ecran.py captures            # aide/img/etape{1,2,3,4}-*.png (2×)
    C:/OSGeo4W/bin/python-qgis.bat dev/rendu_hors_ecran.py manuel [--chapitre etape-2-produits --ancre tuilage-et-marge] [--recherche mot] [--tous]
    C:/OSGeo4W/bin/python-qgis.bat dev/rendu_hors_ecran.py journal                             # vue d'exécution : renvoi ⚠ cliquable
    C:/OSGeo4W/bin/python-qgis.bat dev/rendu_hors_ecran.py profil [--modele crateres_seg_ld_v1]

Monte un ``QgsApplication`` en ``QT_QPA_PLATFORM=offscreen`` avec le thème QSS du plugin, puis :

- ``captures`` : l'assistant aux étapes 1 à 4, la carte d'une entité en réglages avancés, la fiche d'une
  classe et la vue d'exécution (en cours, puis à la fin avec « Par où commencer »), avec des
  chemins neutres dans les champs et quelques produits cochés, en **double résolution**
  (``QT_SCALE_FACTOR=2``, posé avant la création de l'application) → ``aide/img/``. L'étape 4 est
  au vert : Processing et le fournisseur de rvt-qgis sont chargés hors écran, les contrôles du
  préflight sont les vrais, seul le dossier personnel est remplacé dans leurs détails.
- ``manuel`` : la fenêtre du manuel sur un chapitre (et une ancre), PNG dans ``--sortie`` + contrôle
  que la barre horizontale est à 0 px (une image trop large la ferait apparaître) ; ``--recherche``
  vérifie la recherche dans tout le manuel ; ``--tous`` rend chaque chapitre, puis l'historique et le
  zoom — c'est le contrôle de non-régression du runbook ``/livraison``.
- ``journal`` : la vue d'exécution hors run, avec une ligne ⚠ dont le renvoi au manuel est un lien ; puis les
  étapes 4 et 5 de l'assistant : bouton principal, « Suivant » vers le suivi pendant un run, journal gardé.
- ``profil`` : la fiche de classe et la fiche ⓘ d'un modèle, avec la figure du profil des scores,
  puis l'étape 3 en réglages avancés (mini-profil et bilan sur les cartes des entités du modèle).

Ce qu'il faut savoir (appris le 2026-10-07) : l'écran virtuel fait 800×600, d'où ``resize()`` après
``show()`` ; le ``.bat`` avale stdout, les contrôles sont écrits dans ``<sortie>/controles.txt`` ;
la faute de segmentation à la sortie (``exitQgis``) est bénigne ; un worktree n'a pas de modèle
(``data/models`` gitignoré) : lancer depuis le dossier du plugin, ou poser des jonctions.
"""
from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]


def _app(echelle: float):
    """QgsApplication hors écran + police + thème QSS. ``echelle`` : facteur avant création."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if echelle and echelle != 1:
        os.environ["QT_SCALE_FACTOR"] = str(echelle)
    # Le plugin s'importe comme un paquet (imports relatifs ``...app``) : alias sur sa racine.
    paquet = types.ModuleType("archeo")
    paquet.__path__ = [str(RACINE)]
    sys.modules["archeo"] = paquet

    from qgis.core import QgsApplication
    from qgis.PyQt.QtGui import QFont, QFontDatabase

    QgsApplication.setPrefixPath(os.environ.get("QGIS_PREFIX_PATH", ""), True)
    app = QgsApplication([], True)
    app.initQgis()
    polices = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    for nom in ("segoeui.ttf", "segoeuib.ttf", "consola.ttf", "seguisym.ttf"):
        if (polices / nom).is_file():
            QFontDatabase.addApplicationFont(str(polices / nom))
    app.setFont(QFont("Segoe UI", 9))
    qss = (RACINE / "src" / "ui" / "theme" / "v2.qss").read_text(encoding="utf-8")
    app.setStyleSheet(qss.replace("@ICONS@", (RACINE / "src" / "ui" / "theme" / "icons").as_posix()))
    return app


class _Journal:
    """Les contrôles vont dans un fichier : le .bat de QGIS n'affiche pas stdout."""

    def __init__(self, sortie: Path):
        sortie.mkdir(parents=True, exist_ok=True)
        self._f = open(sortie / "controles.txt", "w", encoding="utf-8")

    def __call__(self, *morceaux) -> None:
        self._f.write(" ".join(str(m) for m in morceaux) + "\n")
        self._f.flush()


# ----------------------------------------------------------------------
def _journal_en_cours(app, img: Path, log) -> None:
    """aide/img/run-journal.png : la vue d'exécution V2 pendant une détection (états posés à la
    main, sans worker ni préflight — la frise porte l'état, le journal raconte)."""
    import time

    from archeo.src.ui.run_view import RunView, _fmt_hms

    cfg = {"app": {"files": {"data_mode": "ign_laz", "output_dir": "C:/Temp/archeologia_demo/sortie"}},
           "processing": {"products": {"MNT": True, "SVF": True, "LD": True}},
           "computer_vision": {"enabled": True, "runs": [{"selected_model": "crateres_seg_ld_v1", "target_rvt": "LD"}]}}
    v = RunView(cfg)
    v.show()
    v.resize(880, 520)
    v._apply_stage_sequence()
    v._running = True
    v._run_started_at = time.monotonic() - 1575
    v._activity_dot.setVisible(True)
    v._cancel_btn.setEnabled(True)
    v._open_log_btn.setEnabled(True)
    for ligne in ("▶ Démarrage du traitement — Téléchargement IGN", "📍 Zone : 12 dalles à télécharger",
                  "      ↳ Dalle 12/12 : 174 Mo", "✓ 12 dalles téléchargées", "✓ MNT et indices : 12 dalles, 3 produits",
                  "🔍 « Modèle cratères LD v1 » sur LD : 43 images", "      ↳ Image 31/43 : préparation"):
        v._append_log("INFO", ligne)
    v._on_stage_id("download")
    v._on_metric(12, 12, "dalles")
    v._on_stage_id("products")
    v._step_elapsed[0] = 61
    v._timeline.set_timing(0, _fmt_hms(61))
    v._on_metric(12, 12, "dalles")
    v._on_stage_id("detection")
    v._step_elapsed[1] = 1202
    v._timeline.set_timing(1, _fmt_hms(1202))
    v._active_started = time.monotonic() - 312
    v._on_metric(31, 43, "images")
    v._tick_active()
    v._update_header()
    app.processEvents()
    v.grab().save(str(img / "run-journal.png"))
    log("run-journal :", v.width(), "x", v.height(), "| fil :", v._fil_gauche.text(), "|", v._fil_droite.text())


def _journal_fin(app, img: Path, log) -> bool:
    """aide/img/run-fin.png : la vue d'exécution à la fin d'un run avec détection — synthèse dans la
    ligne du haut, frise faite avec ses durées, cadre « Par où commencer ». Run simulé : états et
    bilan posés à la main, classes déjà présentes dans le registre de couleurs du profil."""
    import time

    from archeo.src.app.services.bilan_fiabilite import LigneBilan
    from archeo.src.app.services.fiabilite import Categorie
    from archeo.src.ui.run_view import RunView, _fmt_hms

    cfg = {"app": {"files": {"data_mode": "ign_laz", "output_dir": "C:/Temp/archeologia_demo/sortie"}},
           "processing": {"products": {"MNT": True, "SVF": True, "LD": True}},
           "computer_vision": {"enabled": True, "runs": [
               {"selected_model": "crateres_seg_ld_v1", "target_rvt": "LD"},
               {"selected_model": "lineaires_seg_v3_1", "target_rvt": "LD"},
               {"selected_model": "ponctuelles_2cl_det_ld_v1", "target_rvt": "LD"}]}}
    v = RunView(cfg)
    v.show()
    v.resize(880, 580)                               # tout tient sans défilement
    v._apply_stage_sequence()
    v._running = True
    durees = (61, 1202, 590, 82)                     # 32min 15s au total
    v._run_started_at = time.monotonic() - sum(durees)
    for ligne in ("▶ Démarrage du traitement — Téléchargement IGN", "✓ 12 dalles téléchargées",
                  "✓ MNT et indices : 12 dalles, 3 produits",
                  "✓ « Modèle cratères LD v1 » : 43 images analysées en 3min 12s (≈ 4s par image)",
                  "✓ « Structures linéaires 3 classes (LD) » : 43 images analysées en 3min 40s (≈ 5s par image)",
                  "✓ « Modèle ponctuelles LD v1 » : 43 images analysées en 2min 58s (≈ 4s par image)",
                  "📂 Projet QGIS écrit : livrable/projet.qgs"):
        v._append_log("INFO", ligne)
    for i, (sid, metrique) in enumerate((("download", (12, 12, "dalles")), ("products", (12, 12, "dalles")),
                                          ("detection", (43, 43, "images")), ("finalize", None))):
        v._on_stage_id(sid)
        if i:
            v._step_elapsed[i - 1] = durees[i - 1]
            v._timeline.set_timing(i - 1, _fmt_hms(durees[i - 1]))
        if metrique:
            v._on_metric(*metrique)
    v._active_started = time.monotonic() - durees[-1]

    def niveaux(coupures, parts):
        noms = ("douteux", "possible", "probable", "quasi_certain")
        return tuple(Categorie(n, c, g, g, 100) for n, c, g in zip(noms, coupures, parts))

    v._bilan = [
        LigneBilan("crateres", "Cratères", "cratere", "cratere", "crateres_seg_ld_v1",
                   niveaux((0.29, 0.35, 0.5, 0.65), (0.0, 0.35, 0.6, 0.85)),
                   {"quasi_certain": 412, "probable": 158, "possible": 61, "douteux": 97}),
        LigneBilan("parcellaire", "Parcellaire", "parcellaire", "parcellaire", "lineaires_seg_v3_1",
                   niveaux((0.26, 0.3, 0.45, 0.6), (0.0, 0.35, 0.6, 0.85)),
                   {"quasi_certain": 18, "probable": 46, "possible": 39, "douteux": 22}),
        LigneBilan("talus_fosse", "Talus et fossés", "talus_fosse", "talus_fosse", "lineaires_seg_v3_1",
                   niveaux((0.25, 0.3, 0.45, 0.7), (0.0, 0.35, 0.6, 0.85)),
                   {"quasi_certain": 9, "probable": 31, "possible": 27, "douteux": 14}),
        LigneBilan("charbonnieres", "Charbonnières", "charbonniere", "charbonniere", "ponctuelles_2cl_det_ld_v1",
                   niveaux((0.2, 0.3, 0.45, 0.6), (0.0, 0.35, 0.6, 0.85)),
                   {"quasi_certain": 6, "probable": 4, "possible": 3, "douteux": 5}),
    ]
    v._couches_a_charger = ((), (), (), ())          # le projet s'ouvrirait au clic : bouton actif
    v._on_run_enabled(True)
    v._open_report_btn.setEnabled(True)              # rapport.html absent du dossier de démonstration
    for _ in range(5):                               # le cadre apparu relance la mise en page de la zone défilante
        app.processEvents()
    v.grab().save(str(img / "run-fin.png"))
    from qgis.PyQt.QtWidgets import QScrollArea
    defile = v.findChild(QScrollArea).verticalScrollBar().maximum()
    log("run-fin :", v.width(), "x", v.height(), "| fil :", v._fil_gauche.text(), "| cadre visible :",
        v._bilan_box.isVisible(), "| défilement", defile, "px (attendu 0)")
    return v._bilan_box.isVisible() and not defile


def _carte_avancee(app, w, img: Path, log, entite: str = "parcellaire") -> bool:
    """aide/img/etape3-carte-avancee.png : la carte d'une entité en réglages avancés — confiance,
    aire minimale, mini-profil des scores avec le seuil en orange."""
    from qgis.PyQt.QtWidgets import QScrollArea

    page = w._detection_page
    page._on_entity_toggled(entite, True)
    page._adv_check.setChecked(True)
    app.processEvents()
    carte = page._cards.get(entite)
    if carte is None:
        log("carte avancée : entité absente :", entite)
        return False
    zone = page.findChild(QScrollArea)
    if zone is not None:
        zone.ensureWidgetVisible(carte, 0, 20)
    app.processEvents()
    carte.grab().save(str(img / "etape3-carte-avancee.png"))
    page._adv_check.setChecked(False)
    app.processEvents()
    log("carte avancée :", entite, carte.width(), "x", carte.height())
    return True


def _processing_hors_ecran(log) -> bool:
    """Processing et le fournisseur rvt-qgis dans l'application hors écran, comme dans QGIS :
    sans eux le préflight de l'étape 4 est rouge (« QGIS processing », « Algorithmes RVT »)."""
    import importlib.util

    try:
        prefixe = Path(os.environ.get("QGIS_PREFIX_PATH", r"C:\OSGeo4W\apps\qgis"))
        sys.path.insert(0, str(prefixe / "python" / "plugins"))
        from processing.core.Processing import Processing

        Processing.initialize()
        rvt = RACINE.parent / "rvt-qgis"            # extension sœur du profil QGIS
        sys.path.insert(0, str(rvt))                # ses algorithmes importent ``rvt.*``
        spec = importlib.util.spec_from_file_location(
            "rvt_qgis", rvt / "__init__.py", submodule_search_locations=[str(rvt)])
        sys.modules["rvt_qgis"] = importlib.util.module_from_spec(spec)
        from qgis.core import QgsApplication
        from rvt_qgis.processing_provider.provider import Provider

        QgsApplication.processingRegistry().addProvider(Provider())
        return True
    except Exception as e:  # noqa: BLE001
        log("Processing / rvt-qgis non chargés hors écran :", repr(e))
        return False


def _fiche_classe(app, img: Path, log, modele: str = "crateres_seg_ld_v1") -> bool:
    """aide/img/etape3-fiche-classe.png : la fiche d'une classe en haut, vérité terrain affichée."""
    from archeo.src.app.services.class_fiche import build_all_fiches
    from archeo.src.app.services.model_orchestrator import discover_installed_models, load_model_card
    from archeo.src.ui.dialogs.class_info_dialog import ClassInfoDialog

    modeles = {m.name: m for m in discover_installed_models(RACINE / "data" / "models")}
    if modele not in modeles:
        log("fiche de classe : modèle absent :", modele)
        return False
    m = modeles[modele]
    card = load_model_card(m.model_dir)
    card = card[0] if isinstance(card, tuple) else card
    fiches = build_all_fiches(card)
    dlg = ClassInfoDialog(fiches, model_dirs={f.modele_id: m.model_dir for f in fiches}, titre=modele)
    dlg.show()
    dlg.resize(900, 640)
    app.processEvents()
    dlg.grab().save(str(img / "etape3-fiche-classe.png"))
    log("fiche de classe :", modele, dlg.width(), "x", dlg.height())
    return True


def _etape4(app, w, img: Path, log) -> bool:
    """aide/img/etape4-verifications.png : bandeau, état du système au vert, récapitulatif."""
    import time

    import archeo.src.pipeline.preflight as pf
    from dataclasses import replace

    if not _processing_hors_ecran(log):
        return False
    brut = pf.collect_preflight_results
    maison = str(Path.home())

    def neutre(**kw):  # le chemin du moteur de détection passe par le profil de l'utilisateur
        return [replace(r, details=r.details.replace(maison, r"C:\Users\utilisateur")) for r in brut(**kw)]

    pf.collect_preflight_results = neutre
    try:
        w._goto_step(4)
        limite = time.monotonic() + 120
        while w._launch_page._preflight_status == "pending" and time.monotonic() < limite:
            app.processEvents()
            time.sleep(0.05)
    finally:
        pf.collect_preflight_results = brut
    app.processEvents()
    w.grab().save(str(img / "etape4-verifications.png"))
    statut = w._launch_page._preflight_status
    log("étape 4 : préflight =", statut, "| bouton :", w._next_btn.text(), "| actif :", w._next_btn.isEnabled())
    return statut == "ok" and w._next_btn.isEnabled()


def captures(app, sortie: Path, log) -> int:
    """Étapes 1 à 4 de l'assistant, la fiche d'une classe et la vue d'exécution en cours →
    aide/img/, chemins neutres, SVF + LD cochés, détection activée."""
    import archeo.src.ui.wizard_dialog as wizard
    from archeo.src.ui.wizard_dialog import WizardDialog

    # ponytail: l'onglet Visualisation n'a pas de chapitre et reste masqué sur dev et main ; les
    # captures montrent l'assistant livré, quelle que soit la branche. Retirer le jour où il est livré.
    wizard.VISUALISATION_TAB_ENABLED = False

    img = RACINE / "aide" / "img"
    img.mkdir(exist_ok=True)
    _journal_en_cours(app, img, log)
    fin_ok = _journal_fin(app, img, log)
    w = WizardDialog()
    w.show()
    w.resize(980, 760)
    app.processEvents()

    def _fichiers(cfg):
        if isinstance(cfg, dict):
            if isinstance(cfg.get("files"), dict):
                return cfg["files"]
            for v in cfg.values():
                r = _fichiers(v)
                if r is not None:
                    return r
        return None

    fichiers = _fichiers(w._config)
    if fichiers is not None:
        fichiers["input_file"] = "C:/Temp/archeologia_demo/dalles.txt"
        fichiers["output_dir"] = "C:/Temp/archeologia_demo/sortie"
        w._source_page.load_from(w._config)
        w._refresh_validation()
    w._goto_step(1)
    app.processEvents()
    w.grab().save(str(img / "etape1-source.png"))
    for cle in ("SVF", "LD"):
        w._indices_page.activate_product(cle)
    w._goto_step(2)
    app.processEvents()
    w.grab().save(str(img / "etape2-produits.png"))
    w._detection_page._enable_check.setChecked(True)
    w._goto_step(3)
    app.processEvents()
    n = len(getattr(w._detection_page, "_cards", {}) or [])
    log("entités affichées à l'étape 3 :", n, "(0 = aucun modèle sous data/models)")
    w.grab().save(str(img / "etape3-detection.png"))
    carte_ok = _carte_avancee(app, w, img, log)
    fiche_ok = _fiche_classe(app, img, log)
    etape4_ok = _etape4(app, w, img, log)
    log("captures écrites dans", img, "— taille de l'assistant :", w.width(), "x", w.height())
    return 0 if n and fin_ok and carte_ok and fiche_ok and etape4_ok else 1


def manuel(app, sortie: Path, log, chapitre: str, ancre: str, recherche: str = "", tous: bool = False) -> int:
    from archeo.src.ui.dialogs.aide_dialog import ouvrir_aide

    dlg = ouvrir_aide(RACINE / "aide", metadata_path=RACINE / "metadata.txt")
    app.processEvents()
    log("chapitres :", [c.cle for c in dlg._chapitres])
    if tous:
        # Contrôle de non-régression (runbook /livraison) : chaque chapitre rendu,
        # barre horizontale à 0 px, PNG par chapitre ; puis historique et zoom.
        echecs = []
        for c in dlg._chapitres:
            dlg.ouvrir(c.cle)
            app.processEvents()
            b = dlg._texte.horizontalScrollBar().maximum()
            log(f"  {c.cle:<22} barre horizontale {b} px | hauteur {dlg._texte.verticalScrollBar().maximum()} px")
            dlg.grab().save(str(sortie / f"manuel_{c.cle}.png"))
            if b:
                echecs.append(c.cle)
        n = len(dlg._chapitres)
        dlg._precedent()
        app.processEvents()
        log("historique :", len(dlg._historique), "entrées ; après Précédent :", dlg._cle,
            "(attendu :", dlg._chapitres[n - 2].cle + ")")
        ok_hist = dlg._cle == dlg._chapitres[n - 2].cle
        dlg._suivant()
        app.processEvents()
        ok_hist = ok_hist and dlg._cle == dlg._chapitres[n - 1].cle
        dlg.ouvrir(dlg._chapitres[1].cle)
        for _ in range(3):
            dlg._zoomer(1)
        app.processEvents()
        bz = dlg._texte.horizontalScrollBar().maximum()
        log(f"zoom {round(dlg._zoom * 100)} % : barre horizontale {bz} px")
        dlg.grab().save(str(sortie / "manuel_zoom_130.png"))
        dlg._zoomer(0)
        app.processEvents()
        log("chapitres en échec (barre horizontale) :", echecs or "aucun", "| historique :", "OK" if ok_hist else "ECHEC")
        return 0 if not echecs and ok_hist and bz == 0 else 1
    if chapitre:
        dlg.ouvrir(chapitre, ancre)
        app.processEvents()
    if recherche:
        dlg._recherche.setText(recherche)
        dlg._chercher()
        app.processEvents()
        log(f"recherche « {recherche} » : mode résultats = {dlg._mode_resultats}, "
            f"{dlg._sommaire.topLevelItemCount()} chapitre(s) dans le sommaire, chapitre ouvert = {dlg._cle}, "
            f"{len(dlg._texte.extraSelections())} occurrence(s) surlignée(s)")
        dlg._chercher()
        app.processEvents()
        log(f"  Entrée à nouveau : {len(dlg._texte.extraSelections())} surlignée(s), défilement "
            f"{dlg._texte.verticalScrollBar().value()} px")
    barre = dlg._texte.horizontalScrollBar().maximum()
    log(f"chapitre {dlg._cle} / ancre {ancre or '-'} : barre horizontale {barre} px",
        "(doit être 0)", "| défilement", dlg._texte.verticalScrollBar().value(), "px")
    chemin = sortie / f"manuel_{dlg._cle}{('_' + ancre) if ancre else ''}{'_recherche' if recherche else ''}.png"
    dlg.grab().save(str(chemin))
    log("rendu :", chemin)
    return 0 if barre == 0 else 1


def profil(app, sortie: Path, log, modele: str) -> int:
    from qgis.PyQt.QtWidgets import QScrollArea

    from archeo.src.app.services.class_fiche import build_all_fiches
    from archeo.src.app.services.model_orchestrator import discover_installed_models, load_model_card
    from archeo.src.ui.dialogs.class_info_dialog import ClassInfoDialog
    from archeo.src.ui.dialogs.model_info_dialog import ModelInfoDialog
    from archeo.src.ui.widgets.profil_scores import ProfilScoresWidget

    modeles = {m.name: m for m in discover_installed_models(RACINE / "data" / "models")}
    if modele not in modeles:
        log("modèle inconnu :", modele, "— installés :", sorted(modeles))
        return 1
    m = modeles[modele]
    card = load_model_card(m.model_dir)
    card = card[0] if isinstance(card, tuple) else card
    fiches = build_all_fiches(card)
    dlg = ClassInfoDialog(fiches, model_dirs={f.modele_id: m.model_dir for f in fiches}, titre=modele)
    dlg.resize(900, 640)
    dlg.show()
    app.processEvents()
    figures = dlg.findChildren(ProfilScoresWidget)
    debord = dlg.findChild(QScrollArea).horizontalScrollBar().maximum()
    log(modele, "fiche de classe : figures =", len(figures), "| barre horizontale", debord, "px (attendu 0)")
    if figures:
        zone = dlg.findChild(QScrollArea)
        if zone is not None:
            zone.ensureWidgetVisible(figures[0], 0, 60)
        app.processEvents()
        figures[0].grab().save(str(sortie / f"profil_widget_{modele}.png"))
    dlg.grab().save(str(sortie / f"profil_fiche_{modele}.png"))
    md = ModelInfoDialog(m)
    md.resize(620, 820)
    md.show()
    app.processEvents()
    figures_modele = md.findChildren(ProfilScoresWidget)
    log(modele, "fiche ⓘ : figures =", len(figures_modele))
    if figures_modele:
        zone = md.findChild(QScrollArea)
        if zone is not None:
            zone.ensureWidgetVisible(figures_modele[0], 0, 80)
        app.processEvents()
    md.grab().save(str(sortie / f"profil_modele_{modele}.png"))
    # Étape 3 en réglages avancés : le mini-profil et son bilan sur la carte de
    # chaque entité du modèle (aide au choix du seuil).
    from archeo.src.ui.wizard_dialog import WizardDialog

    w = WizardDialog()
    w.show()
    w.resize(980, 900)
    page = w._detection_page
    page._enable_check.setChecked(True)
    for eid in m.coverage:
        page._on_entity_toggled(eid, True)
    page._adv_check.setChecked(True)
    w._goto_step(3)
    app.processEvents()
    minis = [f for f in page.findChildren(ProfilScoresWidget) if f.objectName() == "ProfilScoresMini" and f.isVisible()]
    log(modele, "étape 3 avancée : mini-profils visibles =", len(minis))
    if minis:
        zone = page.findChild(QScrollArea)
        if zone is not None:
            zone.ensureWidgetVisible(minis[0], 0, 120)
        app.processEvents()
        log("  bilan :", minis[0].phrase_bilan())
    w.grab().save(str(sortie / f"carte_{modele}.png"))
    return 0 if figures and not debord else 1


def journal(app, sortie: Path, log) -> int:
    """La vue d'exécution hors run : lignes info / ⚠ / ✗, le renvoi ⚠ est un lien."""
    from archeo.src.ui.run_view import RunView

    v = RunView({})
    v.show()
    v.resize(900, 520)
    v._append_log("INFO", "▶ Démarrage du traitement")
    v._append_log("WARNING", "LD : le noyau atteint 40 px mais la marge n'en fait que 20 — bords dégradés")
    v._append_log("ERROR", "Les vérifications préalables ont échoué : pdal introuvable")
    v._update_transient_log("img", "INFO", "      ↳ Image 3/12 : dalle_0881_6546.png")
    app.processEvents()
    html_journal = v._journal.toHtml()
    n_liens = html_journal.count('href="manuel:')
    log("journal : lignes =", v._journal.document().blockCount(), "| liens vers le manuel =", n_liens, "(attendu 2)")
    log("  texte copié :", v._journal.toPlainText().replace("\n", " ¶ ")[:300])
    # Bilan de fiabilité de fin de run, sur des lignes fabriquées.
    from archeo.src.app.services.bilan_fiabilite import LigneBilan
    from archeo.src.app.services.fiabilite import Categorie

    cats = (Categorie("douteux", 0.29, 0.0, 0.26, 100), Categorie("possible", 0.35, 0.35, 0.45, 100),
            Categorie("probable", 0.5, 0.6, 0.72, 100), Categorie("quasi_certain", 0.65, 0.85, 0.92, 100))
    v._bilan = [
        LigneBilan("parcellaire", "Parcellaire", "parcellaire", "parcellaire", "m", cats,
                   {"quasi_certain": 12, "probable": 30, "possible": 41, "douteux": 20}),
        LigneBilan("crateres", "Cratères", "cratere", "cratere", "m", cats,
                   {"quasi_certain": 240, "probable": 90, "possible": 35, "douteux": 60}),
        LigneBilan("charbonnieres", "Charbonnières", "charbonniere", "charbonniere", "m", cats,
                   {"probable": 3, "possible": 1}),
    ]
    v._run_started_at = __import__("time").monotonic() - 125
    v._on_run_enabled(True)          # V2 : synthèse dans le fil + cadre « Par où commencer »
    app.processEvents()
    log("bilan : cadre visible =", v._bilan_box.isVisible(), "| fil :", v._fil_gauche.text(), "| phrases :", [ligne.phrase() for ligne in v._bilan])
    v.grab().save(str(sortie / "journal.png"))
    # Étapes 4 « Vérifications » et 5 « Traitement » (scindées le 2026-10-09) : bouton
    # principal, « Suivant » vers le suivi pendant un run, lecture seule des workers, et
    # le journal d'un run passé qui survit à la navigation. Sans worker : run simulé.
    from archeo.src.ui.wizard_dialog import WizardDialog

    w = WizardDialog()
    w.show()
    w.resize(1120, 700)
    w._goto_step(4)
    app.processEvents()
    etape4 = (w._stack.currentIndex(), w._next_btn.text())
    w._run_view._running = True
    w._enter_review_mode()
    w._goto_step(4)
    app.processEvents()
    pendant = (w._next_btn.text(), w._review_pill.isVisible(), w._launch_page._workers_body.isEnabled())
    w._on_next()
    app.processEvents()
    suivi = (w._current_step, w._next_btn.text(), w._next_btn.isEnabled())
    w.grab().save(str(sortie / "etape5_pendant_run.png"))
    w._run_view._running = False
    w._run_view._run_started_at = 0.0
    w._run_view._append_log("INFO", "✓ Terminé")
    w._exit_review_mode()
    w._goto_step(2)
    w._goto_step(5)
    app.processEvents()
    garde = "Terminé" in w._run_view._journal.toPlainText()
    etapes_ok = (etape4 == (3, "Lancer le pipeline") and pendant == ("Suivant  →", True, False)
                 and suivi == (5, "Lancer le pipeline", False) and garde)
    log("étape 4 (page, bouton) =", etape4, "| pendant un run (bouton, pastille, workers actifs) =", pendant,
        "| Suivant → (étape, bouton, actif) =", suivi, "| journal gardé après navigation =", garde,
        "| attendu (3, 'Lancer le pipeline'), ('Suivant  →', True, False), (5, 'Lancer le pipeline', False), True")
    return 0 if n_liens == 2 and v._bilan_box.isVisible() and etapes_ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("commande", choices=("captures", "manuel", "profil", "journal"))
    parser.add_argument("--sortie", type=Path, default=RACINE / "dev" / "docs" / "_local" / "rendu",
                        help="dossier des PNG de contrôle (gitignoré par défaut)")
    parser.add_argument("--chapitre", default="", help="manuel : clé du chapitre (ex. etape-2-produits)")
    parser.add_argument("--ancre", default="", help="manuel : slug du titre (ex. tuilage-et-marge)")
    parser.add_argument("--recherche", default="", help="manuel : mot à chercher et surligner")
    parser.add_argument("--tous", action="store_true",
                        help="manuel : rendre chaque chapitre (barre horizontale à 0), puis historique et zoom")
    parser.add_argument("--modele", default="crateres_seg_ld_v1", help="profil : dossier du modèle")
    parser.add_argument("--echelle", type=float, default=None,
                        help="facteur d'échelle Qt (défaut : 2 pour captures, 1 sinon)")
    args = parser.parse_args(argv)
    echelle = args.echelle if args.echelle is not None else (2.0 if args.commande == "captures" else 1.0)
    log = _Journal(args.sortie)
    app = _app(echelle)
    try:
        if args.commande == "captures":
            code = captures(app, args.sortie, log)
        elif args.commande == "manuel":
            code = manuel(app, args.sortie, log, args.chapitre, args.ancre, args.recherche, args.tous)
        elif args.commande == "journal":
            code = journal(app, args.sortie, log)
        else:
            code = profil(app, args.sortie, log, args.modele)
    except Exception as exc:  # noqa: BLE001 — le .bat n'affiche rien : tout va dans le journal
        log("ERREUR :", repr(exc))
        code = 2
    log("RESULTAT :", "OK" if code == 0 else f"ECHEC ({code})")
    app.exitQgis()
    return code


if __name__ == "__main__":
    sys.exit(main())
