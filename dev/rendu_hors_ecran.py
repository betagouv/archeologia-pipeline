#!/usr/bin/env python3
"""Rendu hors écran de l'interface — captures et contrôles sans QGIS de bureau.

À lancer avec le **Python de QGIS** (le plugin importe ``qgis.PyQt``) :

    C:/OSGeo4W/bin/python-qgis.bat dev/rendu_hors_ecran.py captures            # aide/img/etape{1,2,3}-*.png (2×)
    C:/OSGeo4W/bin/python-qgis.bat dev/rendu_hors_ecran.py manuel [--chapitre etape-2-produits --ancre tuilage-et-marge]
    C:/OSGeo4W/bin/python-qgis.bat dev/rendu_hors_ecran.py profil [--modele crateres_seg_ld_v1]

Monte un ``QgsApplication`` en ``QT_QPA_PLATFORM=offscreen`` avec le thème QSS du plugin, puis :

- ``captures`` : l'assistant aux étapes 1 à 3, avec des chemins neutres dans les champs et quelques
  produits cochés, en **double résolution** (``QT_SCALE_FACTOR=2``, posé avant la création de
  l'application) → ``aide/img/``. L'étape 4 n'est pas capturée : le préflight échoue hors QGIS.
- ``manuel`` : la fenêtre du manuel sur un chapitre (et une ancre), PNG dans ``--sortie`` + contrôle
  que la barre horizontale est à 0 px (une image trop large la ferait apparaître).
- ``profil`` : la fiche de classe et la fiche ⓘ d'un modèle, avec la figure du profil des scores.

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
def captures(app, sortie: Path, log) -> int:
    """Étapes 1 à 3 de l'assistant → aide/img/, chemins neutres, SVF + LD cochés, détection activée."""
    from archeo.src.ui.wizard_dialog import WizardDialog

    img = RACINE / "aide" / "img"
    img.mkdir(exist_ok=True)
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
    log("captures écrites dans", img, "— taille de l'assistant :", w.width(), "x", w.height())
    return 0 if n else 1


def manuel(app, sortie: Path, log, chapitre: str, ancre: str) -> int:
    from archeo.src.ui.dialogs.aide_dialog import ouvrir_aide

    dlg = ouvrir_aide(RACINE / "aide", metadata_path=RACINE / "metadata.txt")
    app.processEvents()
    log("chapitres :", [c.cle for c in dlg._chapitres])
    if chapitre:
        dlg.ouvrir(chapitre, ancre)
        app.processEvents()
    barre = dlg._texte.horizontalScrollBar().maximum()
    log(f"chapitre {dlg._cle} / ancre {ancre or '-'} : barre horizontale {barre} px",
        "(doit être 0)", "| défilement", dlg._texte.verticalScrollBar().value(), "px")
    chemin = sortie / f"manuel_{dlg._cle}{('_' + ancre) if ancre else ''}.png"
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
    log(modele, "fiche de classe : figures =", len(figures))
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
    return 0 if figures else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("commande", choices=("captures", "manuel", "profil"))
    parser.add_argument("--sortie", type=Path, default=RACINE / "dev" / "docs" / "_local" / "rendu",
                        help="dossier des PNG de contrôle (gitignoré par défaut)")
    parser.add_argument("--chapitre", default="", help="manuel : clé du chapitre (ex. etape-2-produits)")
    parser.add_argument("--ancre", default="", help="manuel : slug du titre (ex. tuilage-et-marge)")
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
            code = manuel(app, args.sortie, log, args.chapitre, args.ancre)
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
