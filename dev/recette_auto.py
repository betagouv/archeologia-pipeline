# -*- coding: utf-8 -*-
"""Recette automatique hors écran — ce qui se lit à l'écran sans vrai traitement (2026-10-26).

Joue, sous ``QT_QPA_PLATFORM=offscreen`` (mêmes mécanismes que ``rendu_hors_ecran.py``), les points
de ``tests/TESTS_MANUELS_QGIS.md`` § 61 à 70 qui n'exigent ni traitement réel, ni canevas, ni outil
absent : libellés, frise, messages, rail, menus, cartes, lignes, fiches, préflight, récapitulatif,
vue de fin simulée, étape 6, manuel. Écrit ``<sortie>/verdicts.json`` (un verdict par point :
``ok`` vrai/faux/null = non joué, ``preuve``) et ``<sortie>/controles.txt``, plus quelques captures.

    C:/OSGeo4W/bin/python-qgis.bat dev/recette_auto.py --sortie D:/brouillons/recette_0.18.0/auto_q3
    "C:/Program Files/QGIS 4.0.3/bin/python-qgis.bat" dev/recette_auto.py --sortie .../auto_q4

À lancer sous les deux QGIS : un point n'est conforme que s'il l'est sous les deux.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rendu_hors_ecran import RACINE, _app, _processing_hors_ecran  # noqa: E402

VERDICTS: dict = {}
LOG = None


def log(*parts):
    ligne = " ".join(str(p) for p in parts)
    LOG.write(ligne + "\n")
    LOG.flush()


def verdict(num: str, ok, preuve: str) -> None:
    VERDICTS[num] = {"ok": ok, "preuve": preuve}
    log(f"[{'OK' if ok else ('--' if ok is None else 'KO')}] {num} : {preuve}")


def point(num: str):
    """Décorateur : une exception = verdict faux avec la trace, jamais un arrêt."""
    def deco(fn):
        def run(*a, **k):
            try:
                return fn(*a, **k)
            except Exception as e:  # noqa: BLE001
                verdict(num, False, f"exception {type(e).__name__} : {e} | {traceback.format_exc().splitlines()[-3]}")
        return run
    return deco


def pump(app, n=6):
    for _ in range(n):
        app.processEvents()


def labels(widget):
    from qgis.PyQt.QtWidgets import QLabel
    return [lb.text() for lb in widget.findChildren(QLabel)]


def boutons(widget):
    from qgis.PyQt.QtWidgets import QPushButton
    return {b.text(): b for b in widget.findChildren(QPushButton)}


def attendre_preflight(app, page, limite=150):
    t0 = time.monotonic()
    while page._preflight_status == "pending" and time.monotonic() - t0 < limite:
        app.processEvents()
        time.sleep(0.05)
    pump(app, 10)
    return page._preflight_status


def mot_le_plus_large(label) -> int:
    fm = label.fontMetrics()
    return max((fm.horizontalAdvance(m) for m in label.text().split()), default=0)


# ----------------------------------------------------------------------------------------- § 61
@point("61.1")
def p61_1(app, w):
    attendu = ["Vos données", "Visualisation du relief", "Structures à chercher", "Prêt à lancer", "Traitement", "Résultats"]
    rail = [it._label.text() for it in w._rail._items]
    debords = [it._label.text() for it in w._rail._items if mot_le_plus_large(it._label) > it._label.width()]
    titres = {n: w.TITLES[n][0] for n in range(1, 7)}
    w._goto_step(2)
    pump(app)
    entete = [t for t in labels(w._indices_page) if t == "Quels produits voulez-vous ?"]
    sous = w._rail._items[1]._sub.text()
    ok = rail == attendu and not debords and titres[2] == "Visualisation du relief" and bool(entete) and sous.startswith("MNT + ")
    verdict("61.1", ok, f"rail {rail} ; débords {debords} ; titre 2 « {titres[2]} » ; en-tête {bool(entete)} ; sous-libellé « {sous} » ; rail {w._rail.width()} px")


@point("61.2")
def p61_2(app, w):
    w._goto_step(1)
    pump(app)
    btns = w._source_page._stage_buttons
    vus = [(b._label_text, b._sub_text) for b in btns.values()]
    attendu = [("Dalles IGN à télécharger", "connexion Internet"), ("Nuages LiDAR", "fichiers LAZ ou LAS"),
               ("MNT existant", "TIF en Lambert-93"), ("Indices de visualisation existants", "TIF en Lambert-93"),
               ("Structures à chercher", "facultatif")]
    ok = vus == attendu and w._source_page._dalles_btn.text() == "Choisir sur la carte"
    verdict("61.2", ok, f"frise {vus} ; bouton « {w._source_page._dalles_btn.text()} »")


@point("61.3")
def p61_3(app, w):
    w._goto_step(4)
    pump(app)
    lancer = w._next_btn.text()
    w._goto_step(5)
    pump(app)
    bas = {w._run_view._open_log_btn.text(), w._run_view._cancel_btn.text()}
    w._rebuild_config_menu()
    menu = [a.text() for a in w._config_menu.actions() if a.text()]
    aide = w._aide_btn.text()
    src_main = (RACINE / "main.py").read_text(encoding="utf-8")
    menu_qgis = 'self.tr("Aide")' in src_main
    ok = (lancer == "Lancer le traitement" and bas == {"Journal complet (fichier)", "Annuler"}
          and menu[0] == "Enregistrer sous…" and "Gérer les configurations…" in menu and "Importer un fichier…" in menu
          and "Tout remettre aux valeurs par défaut…" in menu and aide == "Aide" and menu_qgis)
    verdict("61.3", ok, f"étape 4 « {lancer} » ; étape 5 {sorted(bas)} ; menu {menu} ; bouton « {aide} » ; menu QGIS « Aide » {menu_qgis}")


@point("61.4")
def p61_4(app, w):
    src = "".join((RACINE / "src" / "ui" / p).read_text(encoding="utf-8") for p in ("wizard_dialog.py", "run_view.py", "steps/step_1_source.py"))
    voulus = ["Tout remettre par défaut", "Garder mes réglages", "Réorganiser et lancer", "Revenir à la sélection"]
    manquants = [v for v in voulus if v not in src]
    standards = re.findall(r"StandardButton\.(Yes|No|Cancel|Ok)\b", src)
    ok = not manquants and not standards
    verdict("61.4", ok, f"boutons nommés présents sauf {manquants} ; boutons standards Oui/Non/Annuler : {standards or 'aucun'} ; « Télécharger N dalles » : {'Télécharger ' in src}")


@point("61.6")
def p61_6(app, w, cfg_base):
    cfg = json.loads(json.dumps(cfg_base))
    cfg["app"]["files"]["input_file"] = ""
    cfg["app"]["files"]["output_dir"] = ""
    cfg["processing"]["products"] = {k: False for k in cfg["processing"]["products"]}
    w._apply_config(cfg)
    w._goto_step(4)
    pump(app, 10)
    erreurs = list(w._validation_errors)
    bandeau = w._launch_page._banner.text()
    rail = {i + 1: (it._a_corriger, it._sub.text()) for i, it in enumerate(w._rail._items)}
    attendus = ["Indiquez le dossier de sortie", "Indiquez la zone d'étude ou choisissez des dalles sur la carte",
                "Cochez au moins un produit à calculer à l'étape 2"]
    ok = all(any(a in e for e in erreurs) for a in attendus) and all(a in bandeau for a in attendus) \
        and rail[1][0] and rail[2][0] and rail[1][1] == "à corriger" and rail[2][1] == "à corriger" \
        and "n'est configuré" not in bandeau and "Lancement impossible" in bandeau
    verdict("61.6", ok, f"erreurs {erreurs} ; rail {rail[1]} {rail[2]} ; bandeau : {re.sub('<[^>]+>', ' ', bandeau)[:160]}")
    w._apply_config(cfg_base)
    pump(app)


@point("61.7")
def p61_7(app, w):
    from archeo.src.app.services.aide import CHAPITRE_PAR_ETAPE, charger_chapitres
    from archeo.src.ui.dialogs.aide_dialog import ouvrir_aide
    chapitres = charger_chapitres(RACINE / "aide", RACINE / "metadata.txt")
    titres = [c.titre for c in chapitres]
    glossaire = next(c for c in chapitres if c.cle == "glossaire").markdown
    entrees = [lg.split("|")[1].strip() for lg in glossaire.splitlines() if lg.startswith("| ") and not lg.startswith("| Terme")]
    voulus = ["Produit", "Indice de visualisation", "Structure", "Point de départ", "Traitement", "Dalles en parallèle"]
    dlg = ouvrir_aide(RACINE / "aide", parent=w, cle=CHAPITRE_PAR_ETAPE[2])
    pump(app)
    titre_fenetre = dlg.windowTitle()
    cle = dlg._cle
    dlg.close()
    ok = (all(t in titres for t in ("Étape 1 · Vos données", "Étape 2 · Visualisation du relief", "Étape 3 · Structures à chercher", "Étape 4 · Prêt à lancer"))
          and all(v in entrees for v in voulus) and "Image du relief" not in entrees
          and titre_fenetre.startswith("Aide — Archéolog'IA") and cle == "etape-2-produits")
    verdict("61.7", ok, f"fenêtre « {titre_fenetre} » sur {cle} ; chapitres {titres[:6]} ; glossaire {[v for v in voulus if v in entrees]} ; « Image du relief » absent : {'Image du relief' not in entrees}")


# ----------------------------------------------------------------------------------------- § 62
@point("62.1")
def p62_1(app, w, captures):
    from qgis.PyQt.QtCore import QPoint, Qt
    from qgis.PyQt.QtTest import QTest
    page = w._source_page
    w._goto_step(1)
    pump(app)
    btn = page._stage_buttons[3]
    QTest.mouseClick(btn, Qt.MouseButton.LeftButton)
    pump(app)
    roles = {i: b.property("role") for i, b in page._stage_buttons.items()}
    subs = {i: b._sub.text() for i, b in page._stage_buttons.items()}
    puces = {i: b._name.text()[:1] for i, b in page._stage_buttons.items()}
    badge = page._entry_badge
    pos_badge = badge.mapTo(page._frise_card, QPoint(0, 0)).x()
    pos_btn = btn.mapTo(page._frise_card, QPoint(0, 0)).x()
    dans = pos_btn - 10 <= pos_badge <= pos_btn + btn.width()
    w.grab().save(str(captures / "62-1_mnt_existant.png"))
    ok = (page.current_mode() == "existing_mnt" and roles[1] == roles[2] == "skipped" and roles[3] == "entry"
          and subs[1] == subs[2] == "sautée" and puces[3] == "◉" and puces[1] == "○" and badge.isVisible() and dans)
    verdict("62.1", ok, f"mode {page.current_mode()} ; rôles {roles} ; sous-libellés {subs} ; puces {puces} ; badge visible {badge.isVisible()} sur MNT {dans} (Tab/Espace : à jouer au clavier)")


@point("62.2")
def p62_2(app, w, cfg_base, recette):
    page = w._source_page
    w._apply_config(cfg_base)
    pump(app)
    page._source_edit.setText("")
    page._source_edit.editingFinished.emit()
    pump(app)
    m_vide = page._source_msg.text()
    page._source_edit.setText("D:/brouillons/recette_0.18.0/entrees/absent.txt")
    page._source_edit.editingFinished.emit()
    pump(app)
    m_absent, k_absent = page._source_msg.text(), page._source_msg.property("kind")
    cfg = json.loads(json.dumps(cfg_base))      # nuages locaux : le dossier de sortie DANS le dossier des données
    cfg["app"]["files"]["data_mode"] = "local_laz"
    cfg["app"]["files"]["local_laz_dir"] = str(recette / "entrees")
    cfg["app"]["files"]["output_dir"] = str(recette / "entrees" / "sortie_test")
    w._apply_config(cfg)
    pump(app, 8)
    m_dedans, k_dedans = page._output_msg.text(), page._output_msg.property("kind")
    w._apply_config(cfg_base)
    pump(app)
    ok = (m_vide.startswith("Indiquez la zone d'étude") and "choisissez les dalles sur la carte" in m_vide
          and m_absent.startswith("Ce fichier est introuvable") and k_absent == "error"
          and m_dedans.startswith("Ce dossier est dans le dossier de vos données") and k_dedans == "error")
    verdict("62.2", ok, f"vide « {m_vide[:70]}… » ; absent « {m_absent[:50]} » ({k_absent}) ; dedans « {m_dedans[:60]} » ({k_dedans}) ; « N dalles choisies » : au canevas")


@point("62.3")
def p62_3(app, w, cfg_base):
    cfg = json.loads(json.dumps(cfg_base))
    cfg["processing"]["products"] = {k: False for k in cfg["processing"]["products"]}
    w._apply_config(cfg)
    w._goto_step(1)
    pump(app, 8)
    it2 = w._rail._items[1]
    sans_image = (it2._a_corriger, it2._sub.text(), it2.toolTip())
    cfg2 = json.loads(json.dumps(cfg_base))
    cfg2["processing"]["products"]["MSTP"] = True
    cfg2.setdefault("rvt_params", {}).setdefault("mstp", {})["local_scale_min"] = 50   # > max 21 (section rvt_params.mstp)
    w._apply_config(cfg2)
    w._goto_step(1)
    pump(app, 8)
    mstp = (w._rail._items[1]._a_corriger, w._rail._items[0]._a_corriger, w._rail._items[1].toolTip()[:80])
    w._apply_config(cfg_base)
    w._goto_step(1)
    w._goto_step(2)
    w._goto_step(3)
    pump(app, 8)
    coches = [it._idx.text() for it in w._rail._items[:2]]
    ok = sans_image[0] and sans_image[1] == "à corriger" and "produit" in sans_image[2] and mstp[0] and not mstp[1] and coches == ["✓", "✓"]
    verdict("62.3", ok, f"sans produit : {sans_image[:2]} ; MSTP incohérent : étape 2 {mstp[0]}, étape 1 {mstp[1]} « {mstp[2]} » ; étapes parcourues {coches}")


@point("62.4")
def p62_4(app, w):
    w._rebuild_config_menu()
    menu = [a for a in w._config_menu.actions() if a.text()]
    noms = [a.text() for a in menu]
    w._enter_review_mode()
    w._rebuild_config_menu()
    etats = {a.text(): (a.isEnabled(), a.toolTip()) for a in w._config_menu.actions() if a.text()}
    w._exit_review_mode() if hasattr(w, "_exit_review_mode") else None
    grises = [t for t, (en, _tip) in etats.items() if not en]
    ok = (noms[0] == "Enregistrer sous…" and "Gérer les configurations…" in noms and "Importer un fichier…" in noms
          and "Tout remettre aux valeurs par défaut…" in noms and etats["Enregistrer sous…"][0]
          and not etats["Importer un fichier…"][0] and not etats["Tout remettre aux valeurs par défaut…"][0]
          and all("Indisponible pendant un traitement" in etats[t][1] for t in ("Importer un fichier…", "Tout remettre aux valeurs par défaut…")))
    verdict("62.4", ok, f"entrées {noms} ; en traitement, grisées : {grises} ; infobulle « {etats['Importer un fichier…'][1]} »")


def p_manuel(num, cle, mots):
    from archeo.src.app.services.aide import charger_chapitres
    chapitres = charger_chapitres(RACINE / "aide", RACINE / "metadata.txt")
    md = next(c for c in chapitres if c.cle == cle).markdown
    manquants = [m for m in mots if m.lower() not in md.lower()]
    verdict(num, not manquants, f"chapitre {cle} : {len(mots) - len(manquants)}/{len(mots)} notions trouvées ; manquantes {manquants}")


# ----------------------------------------------------------------------------------------- § 63
@point("63.1")
def p63_1(app, w, cfg_base):
    from archeo.src.app.services.indices_model import default_products
    defauts = {k for k, v in default_products().items() if v}
    cfg = json.loads(json.dumps(cfg_base))
    cfg["processing"]["products"] = {k: False for k in cfg["processing"]["products"]}
    w._apply_config(cfg)
    pump(app)
    rien = {k for k, c in w._indices_page._index_cards.items() if c.property("checked")}
    w._apply_config(cfg_base)
    pump(app)
    ok = defauts == {"MNT", "SVF", "LD"} and not rien
    verdict("63.1", ok, f"défauts d'un profil neuf {sorted(defauts)} ; configuration sans produit : cochés {sorted(rien) or 'aucun'}")


@point("63.2")
def p63_2(app, w):
    page = w._indices_page
    w._goto_step(2)
    pump(app)
    tous = labels(page)
    titres = [t for t in ("Recommandé pour commencer", "Autres indices de visualisation", "Qualité de la donnée") if t in tous]
    rvt = [k for k, c in page._index_cards.items() if c.parent() is not None and page._rvt_card.isAncestorOf(c)]
    page.set_mode("existing_mnt")
    pump(app)
    qual_cachee = not any(page._index_cards[k].isVisibleTo(page) for k in ("DENSITE", "COUVERTURE") if k in page._index_cards)
    page.set_mode("existing_rvt")
    pump(app)
    banniere = page._mode_banner.text()
    page.set_mode("ign_laz")
    pump(app)
    ok = len(titres) == 3 and len(rvt) == 10 and qual_cachee and "sans objet" in banniere and "indices de visualisation" in banniere
    verdict("63.2", ok, f"sections {titres} ; autres indices : {len(rvt)} cartes (12 indices moins SVF et LD recommandés) ; qualité masquée en MNT existant {qual_cachee} ; bandeau existing_rvt « {banniere[:90]} »")


@point("63.3")
def p63_3(app, w):
    from qgis.PyQt.QtCore import Qt
    from qgis.PyQt.QtTest import QTest
    page = w._indices_page
    w._goto_step(2)
    pump(app)
    carte = page._index_cards["HS"]
    avant = bool(carte.property("checked"))
    # la vignette est transparente aux clics : le clic atteint la carte à cet endroit
    QTest.mouseClick(carte, Qt.MouseButton.LeftButton, pos=carte._thumb.geometry().center())
    pump(app)
    apres = bool(carte.property("checked"))
    fiches_avant = len([d for d in w._fiches.values() if d.isVisible()])
    carte._info_btn.click()
    pump(app, 8)
    fiche = w._fiches.get("Produit")
    ouverte = fiche is not None and fiche.isVisible()
    if fiche is not None:
        fiche.close()
    QTest.mouseClick(carte, Qt.MouseButton.LeftButton, pos=carte._thumb.geometry().center())      # remettre
    pump(app)
    QTest.keyClick(carte, Qt.Key.Key_Space)
    pump(app)
    espace = bool(carte.property("checked"))
    QTest.keyClick(carte, Qt.Key.Key_Space)
    pump(app)
    ok = apres != avant and ouverte and fiches_avant == 0 and espace != avant
    verdict("63.3", ok, f"vignette : {avant}→{apres} (coche) ; ⓘ ouvre la fenêtre Produit {ouverte} ; Espace bascule {avant}→{espace}")


@point("63.4")
def p63_4(app, w):
    from qgis.PyQt.QtCore import Qt
    from qgis.PyQt.QtTest import QTest
    page = w._indices_page
    page.activate_product("SVF")
    pump(app)
    mnt = page._index_cards["MNT"]
    QTest.mouseClick(mnt, Qt.MouseButton.LeftButton)
    pump(app)
    ok = bool(mnt.property("checked")) and bool(mnt.property("locked")) and "Nécessaire aux produits cochés" in mnt.toolTip() and "SVF" in mnt.toolTip()
    verdict("63.4", ok, f"MNT coché {mnt.property('checked')} verrou {mnt.property('locked')} ; infobulle « {mnt.toolTip()[:70]} » ; coche « {mnt._check.text()} »")


@point("63.5")
def p63_5(app, w):
    page = w._indices_page
    ld = page._index_cards["LD"]
    page.activate_product("LD")
    pump(app)
    texte = ld._toggle.text()
    ld._toggle.click()
    pump(app)
    depliee = ld._reglages.isVisible() and ld._toggle.text().startswith("▾")
    reinit = page._reset_btns["LD"].toolTip() if "LD" in page._reset_btns else "?"
    # LD : le rayon min pousse le rayon max
    spins = {}
    for (prod, section, key, widget, kind, default) in page._adv_fields:
        if prod == "ldo" or (isinstance(section, tuple) and "ld" in "".join(section).lower()):
            spins[key] = widget
    rmin, rmax = spins.get("min_rad"), spins.get("max_rad")
    pousse = None
    if rmin is not None and rmax is not None:
        rmax.setValue(20)
        rmin.setValue(25)
        pump(app)
        pousse = rmax.value() >= 25
        rmin.setValue(10)
        rmax.setValue(20)
    page._on_product_clicked("LD")                                  # décocher → replie
    pump(app)
    repliee = not ld._reglages.isVisible()
    page.activate_product("LD")
    pump(app)
    ok = texte.startswith("▸ Réglages (") and depliee and "Revenir aux valeurs par défaut de" in reinit and repliee and pousse is not False
    verdict("63.5", ok, f"lien « {texte} » ; déplié {depliee} ; ↺ « {reinit[:50]} » ; replié en décochant {repliee} ; rayon min pousse max : {pousse}")


@point("63.6")
def p63_6(app, w):
    page = w._indices_page
    w._goto_step(2)
    pump(app)
    diag = page._ctx_hint.text()
    visible = page._marge_card.isVisibleTo(page)
    page.set_mode("existing_rvt")
    pump(app)
    cachee = not page._marge_card.isVisibleTo(page)
    page.set_mode("existing_mnt")
    pump(app)
    champ_cache = not page._ov_marge.isVisibleTo(page) and page._marge_card.isVisibleTo(page)
    page.set_mode("ign_laz")
    pump(app)
    ok = visible and (diag.startswith("✓") or diag.startswith("⚠")) and cachee and champ_cache
    verdict("63.6", ok, f"carte visible {visible} ; diagnostic « {diag[:60]} » ; masquée en indices existants {cachee} ; champ masqué en MNT existant {champ_cache}")


@point("63.7")
def p63_7(app, w):
    page = w._indices_page
    page._comparer_btn.click()
    pump(app, 8)
    from qgis.PyQt.QtWidgets import QComboBox, QTableWidget
    fiche = w._fiches.get("Produit")
    ouverte = fiche is not None and fiche.isVisible()
    titres = labels(fiche) if fiche else []
    combo = fiche.findChild(QComboBox) if fiche else None
    comparatif = (any("Caractéristiques" in t or "Quel produit pour quel" in t for t in titres)
                  or (fiche is not None and fiche.findChild(QTableWidget) is not None)
                  or (combo is not None and "Comparer" in combo.currentText()))
    if fiche is not None:
        fiche.close()
    ok = page._comparer_btn.text() == "Comparer les produits ▸" and ouverte and comparatif
    verdict("63.7", ok, f"lien « {page._comparer_btn.text()} » ; fenêtre ouverte {ouverte} sur les tableaux {comparatif}")


# ----------------------------------------------------------------------------------------- § 64 et § 69
def ligne(w, eid):
    return w._detection_page._rows[eid]


@point("64.1")
def p64_1(app, w):
    page = w._detection_page
    page._enable_check.setChecked(True)
    w._goto_step(3)
    pump(app, 8)
    n = len(page._rows)
    tous = labels(page)
    morphos = [t for t in tous if t.strip().endswith(("Ponctuelles", "Linéaires", "Zones", "Surfaces", "Zonales"))]
    filtres = [t for t in tous if t.startswith("Filtre")]
    elidees = [eid for eid, r in page._rows.items() if r._desc.text() != r._desc._texte]
    infobulles = all(r._desc.toolTip() == r._desc._texte for r in page._rows.values())
    hauteur_ok = page._content_scroll.verticalScrollBar().maximum() if page._content_scroll else -1
    ok = n == 11 and len(morphos) >= 2 and not filtres and infobulles
    verdict("64.1", ok, f"{n} lignes ; morphologies {morphos} ; filtres {filtres or 'aucun'} ; descriptions élidées {elidees or 'aucune'} (infobulle = texte complet {infobulles}) ; défilement restant {hauteur_ok} px à {w.height()} px")


@point("64.2")
def p64_2(app, w):
    from qgis.PyQt.QtCore import QPoint, Qt
    from qgis.PyQt.QtTest import QTest
    page = w._detection_page
    r = ligne(w, "four")
    avant = r._selected
    QTest.mouseClick(r, Qt.MouseButton.LeftButton, pos=QPoint(r.width() // 2, 14))
    pump(app)
    clic = r._selected
    r.setFocus()
    QTest.keyClick(r, Qt.Key.Key_Space)
    pump(app)
    espace = r._selected
    actif = page._enable_check.isChecked()
    QTest.mouseClick(page._toggle_frame, Qt.MouseButton.LeftButton, pos=QPoint(page._toggle_frame.width() // 2, page._toggle_frame.height() // 2))
    pump(app)
    bascule = page._enable_check.isChecked() != actif
    page._enable_check.setChecked(True)
    pump(app)
    ok = clic != avant and espace == avant and bascule
    verdict("64.2", ok, f"clic ligne {avant}→{clic} ; Espace →{espace} ; clic sur le bandeau bascule l'interrupteur {bascule}")


@point("64.3")
def p64_3(app, w):
    page = w._detection_page
    page._on_entity_toggled("cratere", True)
    pump(app)
    largeur = w.width()
    ligne(w, "cratere")._fiche_btn.click()
    pump(app, 8)
    f_struct = w._fiches.get("Structure")
    ok1 = f_struct is not None and f_struct.isVisible() and w.width() == largeur
    titre_s = f_struct.windowTitle() if f_struct else ""
    w._goto_step(2)
    pump(app)
    w._indices_page._index_cards["SVF"]._info_btn.click()
    pump(app, 8)
    f_prod = w._fiches.get("Produit")
    ok2 = f_prod is not None and f_prod.isVisible()
    w._goto_step(3)
    pump(app)
    # lien « Fiche du modèle » dans la fiche de la structure → fenêtre Modèle
    lien = [b for t, b in boutons(f_struct).items() if "Fiche du modèle" in t] if f_struct else []
    if lien:
        lien[0].click()
        pump(app, 8)
    f_mod = w._fiches.get("Modèle")
    ok3 = f_mod is not None and f_mod.isVisible()
    # ouvrir une autre structure remplace la fiche
    ligne(w, "four")._fiche_btn.click()
    pump(app, 8)
    f2 = w._fiches.get("Structure")
    remplace = f2 is not None and f2 is not f_struct and f2.isVisible()
    for d in list(w._fiches.values()):
        d.close()
    ok = ok1 and ok2 and ok3 and remplace
    verdict("64.3", ok, f"fenêtre Structure « {titre_s} » {ok1} (largeur inchangée {w.width() == largeur}) ; fenêtre Produit {ok2} ; lien Fiche du modèle → fenêtre Modèle {ok3} ; autre structure remplace {remplace}")


@point("64.4")
def p64_4(app, w):
    from qgis.PyQt.QtWidgets import QComboBox, QScrollArea
    ligne(w, "regroupement_crateres")._fiche_btn.click()
    pump(app, 8)
    f = w._fiches.get("Structure")
    zone = f.findChild(QScrollArea)
    barre_h = zone.horizontalScrollBar().maximum()
    combo = f.findChild(QComboBox)
    liste = combo is not None and combo.count() >= 2
    taille = (f.width(), f.height())
    f.close()
    # fiche du modèle : « Ouvrir le dossier » en bas
    ligne(w, "cratere")._thumb.clicked.emit()
    pump(app, 8)
    fm = w._fiches.get("Modèle")
    ouvrir = any("Ouvrir le dossier" in t for t in boutons(fm)) if fm else False
    barre_m = fm.findChild(QScrollArea).horizontalScrollBar().maximum() if fm else -1
    if fm:
        fm.close()
    ok = barre_h == 0 and liste and ouvrir and barre_m == 0
    verdict("64.4", ok, f"fenêtre {taille} ; barre horizontale {barre_h} px ; liste des classes en tête {liste} ; fiche du modèle : « Ouvrir le dossier » {ouvrir}, barre {barre_m} px")


@point("64.5")
def p64_5(app, w):
    page = w._detection_page
    page._on_entity_toggled("chemin_creux", True)
    pump(app)
    r = ligne(w, "chemin_creux")
    if not r.est_deplie():
        r._basculer_reglages()
    pump(app)
    dec = r._conf_spin.decimals()
    tip = r._conf_spin.toolTip().split("\n")[0]
    reinit = r._reinit_btn.toolTip()
    r.deplier_profil(True)
    pump(app)
    profil = r._profil_section.isVisible()
    ok = dec == 2 and tip.startswith("Seuil appliqué : ") and "seuil du modèle" in tip and reinit.startswith("Revenir au seuil du modèle (") and r._area_spin.isVisible()
    verdict("64.5", ok, f"décimales {dec} ; infobulle « {tip} » ; ↺ « {reinit} » ; profil des scores déplié {profil}")


@point("64.6")
def p64_6(app, w):
    page = w._detection_page
    page._on_entity_toggled("parcellaire", True)
    pump(app)
    r = ligne(w, "parcellaire")
    if not r.est_deplie():
        r._basculer_reglages()
    pump(app)
    r._conf_spin.setValue(0.4)
    pump(app)
    cles = sorted(page._entity_thresholds.get("parcellaire", {}).keys())
    cand = list(r._candidates)
    suivi = None
    if len(cand) > 1:
        autre = [c for c in cand if c not in r._current_models][0]
        page._on_models_changed("parcellaire", [autre])
        pump(app)
        suivi = (ligne(w, "parcellaire")._area_spin.value(), page._models[autre].default_min_area)
        page._on_models_changed("parcellaire", [cand[0]])
        pump(app)
    cfg = {}
    page.collect_into(cfg)
    persiste = cfg.get("computer_vision", {}).get("entity_thresholds", {}).get("parcellaire")
    r._reinit_btn.click()
    pump(app)
    ok = cles == ["confidence_threshold"] and persiste == {"confidence_threshold": 0.4} and (suivi is None or abs(suivi[0] - suivi[1]) < 1e-6)
    verdict("64.6", ok, f"après réglage de la confiance : clés {cles} ; aire min suit le modèle changé {suivi} ; config écrite {persiste}")


@point("64.7")
def p64_7(app, w):
    page = w._detection_page
    r = ligne(w, "parcellaire")
    if not r.est_deplie():
        r._basculer_reglages()
    r._conf_spin.setValue(0.33)
    pump(app)
    page._on_entity_toggled("parcellaire", False)
    w._collect_config()                                      # enregistrer la configuration…
    w._apply_config(json.loads(json.dumps(w._config)))      # …et la recharger
    pump(app)
    page._on_entity_toggled("parcellaire", True)
    pump(app)
    r = ligne(w, "parcellaire")
    seuil = page._entity_thresholds.get("parcellaire", {}).get("confidence_threshold")
    w._goto_step(4)
    pump(app, 8)
    recap = labels(w._launch_page)
    w._goto_step(3)
    pump(app)
    r._reinit_btn.click()
    pump(app)
    ok = seuil == 0.33 and not any("Fours" in t for t in recap)
    verdict("64.7", ok, f"seuil retrouvé après décocher/recharger/recocher : {seuil} ; récapitulatif sans « Fours » (décochée) : {not any('Fours' in t for t in recap)}")


@point("64.8")
def p64_8(app, w):
    page = w._detection_page
    page._on_entity_toggled("cratere", False)
    page._on_entity_toggled("regroupement_crateres", True)
    pump(app)
    rc, rg = ligne(w, "cratere"), ligne(w, "regroupement_crateres")
    badge = rc._derived_badge.text()
    ok = rc._selected and "inclus dans" in badge and not rc._adv_row.isEnabled() and rg._derived_badge.isVisible()
    verdict("64.8", ok, f"Regroupement coché → Cratères coché {rc._selected}, badge « {badge[:60]} », seuils grisés {not rc._adv_row.isEnabled()} ; badge regroupement visible {rg._derived_badge.isVisible()}")
    page._on_entity_toggled("regroupement_crateres", False)
    pump(app)


@point("64.9")
def p64_9(app, w):
    page = w._detection_page
    page._on_entity_toggled("cratere", True)
    pump(app)
    phrases = labels(page._runs_host)
    analyses = [p for p in phrases if "lit l'indice" in p]
    compteur = page._runs_count.text() if page._runs_count else ""
    from archeo.src.ui.widgets.info_button import InfoButton
    n_i = len(page._runs_host.findChildren(InfoButton))
    ok = analyses and all("<b>" in p for p in analyses) and n_i == len(analyses) and "analyse" in compteur
    verdict("64.9", ok, f"{len(analyses)} phrases, ex. « {re.sub('<[^>]+>', '', analyses[0]) if analyses else '—'} » ; ⓘ {n_i} ; compteur « {compteur} »")


@point("64.10")
def p64_10(app, w):
    page = w._detection_page
    page.set_readonly(True)
    pump(app)
    grises = (not page._enable_check.isEnabled(), not ligne(w, "cratere")._conf_spin.isEnabled())
    i_actif = ligne(w, "cratere")._fiche_btn.isEnabled()
    ligne(w, "cratere")._fiche_btn.click()
    pump(app, 8)
    f = w._fiches.get("Structure")
    ouverte = f is not None and f.isVisible()
    if f:
        f.close()
    page.set_readonly(False)
    pump(app)
    ok = all(grises) and i_actif and ouverte
    verdict("64.10", ok, f"lecture seule : interrupteur grisé {grises[0]}, seuils grisés {grises[1]}, ⓘ actif {i_actif}, fiche ouverte {ouverte}")


@point("69.1")
def p69_1(app, w):
    from archeo.src.ui.widgets.info_button import InfoButton
    page = w._detection_page
    page._on_entity_toggled("cratere", True)
    page._on_entity_toggled("parcellaire", True)
    pump(app)
    rc, rp = ligne(w, "cratere"), ligne(w, "parcellaire")
    n_i = {eid: len(r.findChildren(InfoButton)) for eid, r in page._rows.items()}
    multi = len(rp._candidates) > 1
    ok = (not hasattr(rc, "_model_name") and all(n == 1 for n in n_i.values())
          and rp._change_btn.isVisible() == multi and rp._change_btn.text() == "Modèle ▾"
          and (not multi or "fenêtre" in rp._change_btn.toolTip() or rp._change_btn.toolTip())
          and not rc._change_btn.isVisible() or len(rc._candidates) > 1)
    verdict("69.1", ok, f"un ⓘ par ligne {set(n_i.values())} ; Parcellaire : {len(rp._candidates)} modèles, « {rp._change_btn.text()} » visible {rp._change_btn.isVisible()}, infobulle « {rp._change_btn.toolTip()[:70]} » ; Cratères : {len(rc._candidates)} modèle, bouton visible {rc._change_btn.isVisible()}")


@point("69.2")
def p69_2(app, w):
    page = w._detection_page
    phrases = [re.sub("<[^>]+>", "", p) for p in labels(page._runs_host) if "lit l'indice" in p]
    ligne(w, "cratere")._fiche_btn.click()
    pump(app, 8)
    f = w._fiches.get("Structure")
    lien = [b for t, b in boutons(f).items() if "Fiche du modèle" in t] if f else []
    if f:
        f.close()
    ok = bool(phrases) and bool(lien)
    verdict("69.2", ok, f"phrases « {phrases[0] if phrases else '—'} » ; lien « Fiche du modèle » dans la fiche {bool(lien)}")


@point("69.3")
def p69_3(app, w):
    page = w._detection_page
    rp = ligne(w, "parcellaire")
    cand = list(rp._candidates)
    if len(cand) < 2:
        verdict("69.3", None, "un seul modèle couvre Parcellaire sur ce poste : comparaison non jouable")
        return
    page._on_models_changed("parcellaire", cand[:2])
    pump(app)
    phrases = [re.sub("<[^>]+>", "", p) for p in labels(page._runs_host) if "lit l'indice" in p]
    deux = sum("Parcellaire" in p for p in phrases)
    aspect = (ligne(w, "parcellaire")._change_btn.isVisible(), ligne(w, "parcellaire")._change_btn.text())
    page._on_models_changed("parcellaire", cand[:1])
    pump(app)
    verdict("69.3", deux >= 2 and aspect == (True, "Modèle ▾"), f"deux modèles cochés → {deux} analyses mentionnent Parcellaire ; ligne {aspect}")


@point("69.4")
def p69_4(app, w):
    from qgis.PyQt.QtCore import Qt
    from qgis.PyQt.QtTest import QTest
    page = w._detection_page
    r = ligne(w, "four")
    page._on_entity_toggled("four", True)
    pump(app)
    visible = r._reglages_toggle.isVisible()
    texte = r._reglages_toggle.text()
    r._reglages_toggle.click()
    pump(app)
    deplie = r._adv_row.isVisible() and r._reglages_toggle.text() == "▾ Réglages"
    r._reglages_toggle.setFocus()
    QTest.keyClick(r._reglages_toggle, Qt.Key.Key_Space)
    pump(app)
    replie_espace = not r._adv_row.isVisible()
    r._reglages_toggle.click()
    pump(app)
    page._on_entity_toggled("four", False)
    pump(app)
    replie_decoche = not r.est_deplie()
    case = any("Seuils et réglages par structure" in t for t in labels(page))
    ok = visible and texte == "▸ Réglages" and deplie and replie_espace and replie_decoche and not case
    verdict("69.4", ok, f"lien « {texte} » visible {visible} ; déplié {deplie} ; Espace replie {replie_espace} ; décocher replie {replie_decoche} ; plus de case globale {not case}")


@point("69.5")
def p69_5(app, w):
    page = w._detection_page
    w.resize(980, 760)
    pump(app, 8)
    elidees = [eid for eid, r in page._rows.items() if r._desc.text() != r._desc._texte]
    longueurs = max(len(r._desc._texte) for r in page._rows.values())
    verdict("69.5", not elidees, f"à {w.width()} px : descriptions élidées {elidees or 'aucune'} ; la plus longue {longueurs} caractères")


@point("69.6")
def p69_6(app, w):
    from qgis.PyQt.QtCore import Qt
    from qgis.PyQt.QtTest import QTest
    page = w._detection_page
    page._on_entity_toggled("charbonniere", False)
    pump(app)
    r = ligne(w, "charbonniere")
    avant = r._selected
    for d in list(w._fiches.values()):
        d.close()
    QTest.mouseClick(r._thumb, Qt.MouseButton.LeftButton)
    pump(app, 8)
    f = w._fiches.get("Structure")       # révisé 2026-10-26 : la vignette ouvre la fiche de la structure, comme le « i »
    ouverte = f is not None and f.isVisible()
    titre = f.windowTitle() if f else ""
    if f:
        f.close()
    ok = ouverte and "Charbonni" in titre and r._selected == avant and r._thumb.cursor().shape() == Qt.CursorShape.PointingHandCursor and r._thumb.toolTip() == "Fiche de la structure"
    verdict("69.6", ok, f"vignette (ligne non cochée) → fenêtre Structure « {titre} » {ouverte} ; ligne inchangée {r._selected == avant} ; curseur main {r._thumb.cursor().shape() == Qt.CursorShape.PointingHandCursor} ; infobulle « {r._thumb.toolTip()} »")


@point("69.7")
def p69_7(app, w):
    from archeo.src.ui.widgets.info_button import InfoButton
    tous = w.findChildren(InfoButton)
    autres = sorted({b.text() for b in tous if b.text() != "i"})
    tailles = {(b.width(), b.height()) for b in tous}
    verdict("69.7", bool(tous) and not autres, f"{len(tous)} boutons de fiche, texte « i » partout (autres : {autres or 'aucun'}) ; tailles {tailles}")


# ----------------------------------------------------------------------------------------- § 65
@point("65.1")
def p65_1(app, w, cfg_base, captures):
    cfg = json.loads(json.dumps(cfg_base))
    cfg["app"]["files"]["output_dir"] = ""
    w._apply_config(cfg)
    w._goto_step(4)
    pump(app, 10)
    bandeau = re.sub("<[^>]+>", " ", w._launch_page._banner.text())
    un = (w._next_hint.text(), w._next_btn.isEnabled())
    cfg["processing"]["products"] = {k: False for k in cfg["processing"]["products"]}
    w._apply_config(cfg)
    w._goto_step(4)
    pump(app, 10)
    deux = w._next_hint.text()
    w.grab().save(str(captures / "65-1_a_corriger.png"))
    w._apply_config(cfg_base)
    pump(app)
    ok = ("Lancement impossible" in bandeau and "Étape 1 : Indiquez le dossier de sortie" in bandeau
          and un == ("Lancement impossible : 1 point à corriger (étape 1)", False)
          and deux == "Lancement impossible : 2 points à corriger (étapes 1, 2)")
    verdict("65.1", ok, f"bandeau « {bandeau.strip()[:90]} » ; indication 1 : {un} ; indication 2 : « {deux} »")


@point("65.2")
def p65_2(app, w, captures):
    w._goto_step(1)
    w._goto_step(4)
    w._launch_page.refresh_preflight(w._config)          # lu AVANT tout événement : l'état « en cours » est fugace ici
    pendant = re.sub("<[^>]+>", "", w._launch_page._banner.text())
    statut = attendre_preflight(app, w._launch_page)
    apres = re.sub("<[^>]+>", "", w._launch_page._banner.text())
    pastille = w._launch_page._preflight_pill.text()
    w.grab().save(str(captures / "65-2_pret.png"))
    ok = pendant == "Vérification de l'environnement…" and statut == "ok" and apres == "Prêt à lancer." and re.search(r"●\s+(\d+)/\1$", pastille) is not None
    verdict("65.2", ok, f"pendant « {pendant} » ; statut {statut} ; bandeau « {apres} » ; pastille « {pastille} »")


@point("65.3")
def p65_3(app, w, captures):
    pdal = shutil.which("pdal")
    if not pdal:
        verdict("65.3", None, "pdal introuvable dans le PATH de ce poste : non joué")
        return
    chemin = os.environ["PATH"]
    dossier = str(Path(pdal).parent)
    os.environ["PATH"] = os.pathsep.join(p for p in chemin.split(os.pathsep) if os.path.normcase(p.rstrip("\\/")) != os.path.normcase(dossier))
    try:
        w._launch_page._reverifier_btn.click()
        statut = attendre_preflight(app, w._launch_page)
        bandeau = re.sub("<[^>]+>", " ", w._launch_page._banner.text())
        lien = "manuel:depannage#les-verifications-prealables-ont-echoue" in w._launch_page._banner.text()
        actif = w._next_btn.isEnabled()
        w.grab().save(str(captures / "65-3_outil_manquant.png"))
    finally:
        os.environ["PATH"] = chemin
        w._launch_page._reverifier_btn.click()
        attendre_preflight(app, w._launch_page)
    ok = statut == "err" and "Un outil manque : Lecteur de nuages de points (pdal)" in bandeau and lien and actif
    verdict("65.3", ok, f"sans pdal : statut {statut} ; bandeau « {bandeau.strip()[:110]} » ; lien Dépannage {lien} ; bouton actif {actif}")


@point("65.4")
def p65_4(app, w, cfg_base, recette, captures):
    from osgeo import gdal
    dossier = recette / "entrees" / "mnt_sans_crs"
    dossier.mkdir(parents=True, exist_ok=True)
    tif = dossier / "sans_crs.tif"
    if not tif.is_file():
        ds = gdal.GetDriverByName("GTiff").Create(str(tif), 64, 64, 1, gdal.GDT_Float32)
        ds.SetGeoTransform((700000, 0.5, 0, 6700000, 0, -0.5))
        ds.GetRasterBand(1).Fill(100.0)
        ds = None
    cfg = json.loads(json.dumps(cfg_base))
    cfg["app"]["files"]["data_mode"] = "existing_mnt"
    cfg["app"]["files"]["existing_mnt_dir"] = str(dossier)
    w._apply_config(cfg)
    w._goto_step(1)
    w._goto_step(4)
    statut = attendre_preflight(app, w._launch_page)
    bandeau = re.sub("<[^>]+>", "", w._launch_page._banner.text())
    lignes_crs = [t for t in labels(w._launch_page) if "coordonn" in t.lower()]
    tips_crs = [lb.toolTip() for lb in w._launch_page.findChildren(type(w._launch_page._banner)) if "coordonn" in lb.text().lower()]
    w.grab().save(str(captures / "65-4_avertissement.png"))
    w._apply_config(cfg_base)
    w._goto_step(1)
    w._goto_step(4)
    attendre_preflight(app, w._launch_page)
    ok = statut == "warn" and bandeau == "Prêt à lancer, avec un avertissement : Système de coordonnées des rasters."
    verdict("65.4", ok, f"MNT sans CRS : statut {statut} ; bandeau « {bandeau} » ; ligne CRS {lignes_crs} infobulle {[t[:80] for t in tips_crs]}")


@point("65.5")
def p65_5(app, w):
    from qgis.PyQt.QtWidgets import QLabel
    from archeo.src.app.services.vocabulaire import ENVIRONNEMENT
    grille = w._launch_page._preflight_grid
    lignes = []
    for r in range(grille.rowCount()):
        cases = [grille.itemAtPosition(r, c) for c in range(grille.columnCount())]
        textes = [i.widget().text() if i and isinstance(i.widget(), QLabel) else "" for i in cases]
        tips = [i.widget().toolTip() if i and isinstance(i.widget(), QLabel) else "" for i in cases]
        if any(textes):
            lignes.append((textes, tips))
    noms = [t[1] for t, _ in lignes if len(t) > 1]
    francais = [n for n in noms if n in ENVIRONNEMENT.values() or n.startswith("Bibliothèque Python ")]
    pas_francais = [n for n in noms if n not in francais]
    colonnes = [t[2] for t, _ in lignes if len(t) > 2]
    sobres = [c for c in colonnes if not re.search(r"[A-Za-z]:[\\/]|Traceback|Error", c)]
    chemins_en_bulle = sum(1 for _t, tips in lignes if len(tips) > 2 and tips[2] and tips[2] != _t[2])
    ok = lignes and not pas_francais and len(sobres) == len(colonnes)
    verdict("65.5", ok, f"{len(lignes)} contrôles ; libellés français {len(francais)}/{len(noms)} (autres : {pas_francais}) ; colonnes sans chemin {len(sobres)}/{len(colonnes)} ex. {colonnes[:4]} ; détails en infobulle sur {chemins_en_bulle} lignes")


@point("65.6")
def p65_6(app, w):
    page = w._launch_page
    page._reverifier_btn.click()                              # lu AVANT tout événement : vérifications très rapides ici
    pendant = (page._preflight_pill.text(), page._reverifier_btn.isEnabled())
    statut = attendre_preflight(app, page)
    ok = "…" in pendant[0] and not pendant[1] and statut == "ok" and page._reverifier_btn.isEnabled()
    verdict("65.6", ok, f"pendant : pastille « {pendant[0]} », bouton actif {pendant[1]} ; après : {statut}, pastille « {page._preflight_pill.text()} », bouton actif {page._reverifier_btn.isEnabled()}")


@point("65.7")
def p65_7(app, w, cfg_base, recette):
    from archeo.src.app.services.sorties_prevues import sorties_prevues
    page = w._launch_page
    w._goto_step(4)
    pump(app, 8)
    sections = [s for s in labels(page) if s in ("Données", "Visualisation du relief", "Structures", "Dossier de sortie", "Ce qui sera produit")]
    liens = [b for t, b in boutons(page).items() if t == "modifier"]
    cibles = []
    orig = page.modifier_demande
    def noter(n):
        cibles.append(n)
    orig.connect(noter)
    for b in page.findChildren(type(liens[0])) if liens else []:
        if b.text() == "modifier":
            b.click()
    pump(app)
    orig.disconnect(noter)
    w._goto_step(4)
    pump(app)
    sorties = sorties_prevues(w._config)
    attendus = ["livrable/projet.qgs", "livrable/rapport.html", "livrable/traitement.json"]
    indices = [s for s in sorties if "/indices/" in s]
    cfg = json.loads(json.dumps(cfg_base))
    cfg["app"]["files"]["data_mode"] = "existing_rvt"
    cfg["app"]["files"]["existing_rvt_dir"] = str(recette / "entrees" / "entrees_ld2")
    sans = [s for s in sorties_prevues(cfg) if "/indices/" in s]
    ok = (set(sections) == {"Données", "Visualisation du relief", "Structures", "Dossier de sortie", "Ce qui sera produit"}
          and sorted(set(cibles)) == [1, 2, 3] and all(any(a in s for s in sorties) for a in attendus)
          and any("SVF_" in s and "/tif/" in s for s in indices) and not sans)
    verdict("65.7", ok, f"sections {sorted(set(sections))} ; « modifier » → étapes {sorted(set(cibles))} ; sorties {sorties} ; indices existants → {sans or 'aucun indices/'}")


@point("65.8")
def p65_8(app, w):
    page = w._launch_page
    page._adv.set_expanded(True)                 # « Réglages du poste » est un accordéon
    page.set_decision_carte({"carte": "NVIDIA GeForce RTX 3060", "date": "2026-10-20T10:00:00", "retenue": True, "device_id": 0})
    pump(app)
    avec = (page._carte_memo.text(), page._carte_memo.isVisible(), page._oublier_btn.isVisible())
    page.set_decision_carte(None)
    pump(app)
    sans = (page._carte_memo.isVisible(), page._oublier_btn.isVisible())
    ok = "Décision mémorisée pour ce poste" in avec[0] and avec[1] and avec[2] and not sans[0] and not sans[1]
    verdict("65.8", ok, f"avec décision : « {avec[0][:80]} », visible {avec[1]}, Oublier {avec[2]} ; sans : ligne {sans[0]}, bouton {sans[1]} (clic Oublier : à jouer, il efface le fichier du profil)")


@point("65.9")
def p65_9(app, w):
    page = w._launch_page
    page.set_readonly(True)
    pump(app)
    etat = (page._workers_spin.isEnabled(), page._carte_check.isEnabled(), page._reverifier_btn.isEnabled(),
            all(b.isEnabled() for t, b in boutons(page).items() if t == "modifier"))
    page.set_readonly(False)
    pump(app)
    ok = etat == (False, False, True, True)
    verdict("65.9", ok, f"lecture seule : dalles en parallèle actif {etat[0]}, carte graphique actif {etat[1]}, Revérifier actif {etat[2]}, « modifier » actifs {etat[3]}")


# ----------------------------------------------------------------------------------------- § 66 (simulée)
def vue_fin(app):
    """La vue d'exécution à la fin d'un traitement simulé (même recette que rendu_hors_ecran)."""
    from archeo.src.app.services.bilan_fiabilite import LigneBilan
    from archeo.src.app.services.fiabilite import Categorie
    from archeo.src.ui.run_view import RunView, _fmt_hms
    cfg = {"app": {"files": {"data_mode": "ign_laz", "output_dir": "C:/Temp/archeologia_demo/sortie"}},
           "processing": {"products": {"MNT": True, "SVF": True, "LD": True}},
           "computer_vision": {"enabled": True, "runs": [{"selected_model": "crateres_seg_ld_v1", "target_rvt": "LD"}]}}
    v = RunView(cfg)
    v.show()
    v.resize(880, 620)
    v._apply_stage_sequence()
    v._running = True
    durees = (61, 1202, 590, 82)
    v._run_started_at = time.monotonic() - sum(durees)
    for i, (sid, metrique) in enumerate((("download", (12, 12, "dalles")), ("products", (12, 12, "dalles")),
                                          ("detection", (43, 43, "images")), ("finalize", None))):
        v._on_stage_id(sid)
        if i:
            v._step_elapsed[i - 1] = durees[i - 1]
            v._timeline.set_timing(i - 1, _fmt_hms(durees[i - 1]))
        if metrique:
            v._on_metric(*metrique)
    v._active_started = time.monotonic() - durees[-1]
    noms = ("douteux", "possible", "probable", "quasi_certain")
    cats = tuple(Categorie(n, c, g, g, 100) for n, c, g in zip(noms, (0.29, 0.35, 0.5, 0.65), (0.0, 0.35, 0.6, 0.85)))
    v._bilan = [LigneBilan("crateres", "Cratères", "cratere", "cratere", "crateres_seg_ld_v1", cats,
                           {"quasi_certain": 120, "probable": 80, "possible": 40, "douteux": 12})]
    v._couches_a_charger = ((), (), (), ())
    v._on_run_enabled(True)
    pump(app, 8)
    return v


@point("66.1")
def p66_1(app, v, captures):
    b = boutons(v)
    un = ("Ajouter au projet QGIS courant" in b, "Ouvrir livrable/projet.qgs" not in b, "Ouvrir le dossier" in b, "Rapport" in b)
    actif = v._open_project_btn.isEnabled()
    v.grab().save(str(captures / "66-1_fin_simulee.png"))
    verdict("66.1", all(un) and actif and v._bilan_box.isVisible(), f"cadre « Par où commencer » visible {v._bilan_box.isVisible()} ; boutons (Ajouter présent, Ouvrir livrable absent, dossier, rapport) {un} ; « Ajouter » actif {actif} (chargement réel et grisage : au traitement réel)")


@point("66.2")
def p66_2(app, w):
    w._goto_step(2)
    pump(app)
    avant = w._relancer_btn.isVisible()
    w._run_view._run_started_at = time.monotonic() - 10      # un traitement a eu lieu
    w._run_view._running = False
    w._goto_step(2)
    pump(app)
    a2 = w._relancer_btn.isVisible()
    w._goto_step(4)
    pump(app)
    a4 = w._relancer_btn.isVisible()
    w._goto_step(5)
    pump(app)
    a5 = w._relancer_btn.isVisible()
    w._run_view._running = True
    w._goto_step(3)
    pump(app)
    pendant = w._relancer_btn.isVisible()
    w._run_view._running = False
    w._run_view._run_started_at = None
    w._goto_step(3)
    pump(app)
    ok = not avant and a2 and not a4 and not a5 and not pendant and w._relancer_btn.text() == "Relancer maintenant"
    verdict("66.2", ok, f"« {w._relancer_btn.text()} » : avant tout traitement {avant} ; après, étape 2 {a2}, étape 4 {a4}, étape 5 {a5}, pendant un traitement {pendant} (le clic : au traitement réel)")


@point("66.4")
def p66_4(app, v):
    from archeo.src.app.services.bilan_fiabilite import texte_niveaux_effectifs
    from archeo.src.ui.widgets import bilan_fiabilite as bf
    texte = texte_niveaux_effectifs(v._bilan[0].effectifs)
    widget = v._bilan_box.findChild(bf.BilanFiabiliteWidget)
    rangee = getattr(bf, "_RANGEE", None)
    ok = texte == "très probable 120 · probable 80 · possible 40 · douteux 12" and rangee == 34 and widget is not None
    verdict("66.4", ok, f"texte sous la barre « {texte} » ; rangée {rangee} px ; widget présent {widget is not None}")


@point("66.5")
def p66_5(app):
    from archeo.src.app.services import couleurs_couches
    src = (RACINE / "src" / "app" / "services" / "rapport_run.py").read_text(encoding="utf-8")
    ok = "Par niveau" in src and "texte_niveaux_effectifs" in src and hasattr(couleurs_couches, "couleurs_du_bilan") and "background:" in src
    verdict("66.5", ok, "rapport : colonne « Par niveau » et barres colorées par le registre (tests unitaires test_rapport_run, test_couleurs_couches) ; rendu visuel à ouvrir après le traitement réel")


@point("66.6")
def p66_6(app):
    src = (RACINE / "src" / "app" / "user_narrator.py").read_text(encoding="utf-8")
    ok = "Voir le journal {journal}" in src and 'journal: str = ""' in src
    verdict("66.6", ok, "pipeline_failed cite « Voir le journal technique/journaux/pipeline_log_<date>.txt » (chaîne du narrateur ; à voir sur un vrai échec)")


# ----------------------------------------------------------------------------------------- § 67
@point("67.1")
def p67_1(app, w):
    from archeo.src.app.services.aide import CHAPITRE_PAR_ETAPE, charger_chapitres
    w._goto_step(6)
    pump(app, 6)
    six = len(w._rail._items) == 6 and w._rail._items[5]._label.text() == "Résultats"
    sous = w._rail._items[5]._sub.text()
    suivant = (w._next_btn.isEnabled(), w._next_btn.toolTip())
    chap = next(c for c in charger_chapitres(RACINE / "aide", RACINE / "metadata.txt") if c.cle == CHAPITRE_PAR_ETAPE.get(6))
    ok = six and ("traitement" in sous) and suivant == (False, "Dernier écran") and chap.titre == "Vos résultats dans QGIS"
    verdict("67.1", ok, f"rail 6 étapes {six} ; sous-libellé « {sous} » ; Suivant {suivant} ; F1 → « {chap.titre} »")


@point("67.2")
def p67_2(app, w, recette):
    from archeo.src.app.services.resultats_connus import Resultat
    page = w._resultats_page
    page.set_resultats([])
    pump(app)
    vide = (page._vide.isVisible(), page._vide.text())
    base = recette / "sorties"
    r1 = Resultat(base / "ancien", "2026-10-20T10:00:00", "Nuages LiDAR locaux", ("MNT", "LD"), ("Cratères",), 4120,
                  base / "ancien/livrable/projet.qgs", base / "ancien/livrable/rapport.html", None)
    meta = base / "recent" / "technique" / "journaux"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "metadata_20261024_1200.json").write_text(json.dumps({"ui_config": {"app": {"files": {"data_mode": "ign_laz", "output_dir": str(base / "recent")}}, "computer_vision": {"enabled": True, "selected_entities": ["cratere"]}}}), encoding="utf-8")
    r2 = Resultat(base / "recent", "2026-10-24T09:30:00", "Téléchargement IGN", ("MNT", "SVF", "LD"), ("Cratères", "Parcellaire"), 952,
                  base / "recent/livrable/projet.qgs", base / "recent/livrable/rapport.html", meta / "metadata_20261024_1200.json")
    page.set_resultats([r1, r2])
    pump(app)
    textes = labels(page)
    premiere = next((t for t in textes if "24/10/2026" in t or "2026-10-24" in t), "")
    ok = vide == (True, "Aucun traitement connu sur ce poste.") and len(page._lignes) == 2 and any("Téléchargement IGN" in t for t in textes) and any("952" in t for t in textes)
    verdict("67.2", ok, f"sans traitement : {vide} ; deux résultats affichés {len(page._lignes)} ; point de départ, images, structures, détections dans les textes ; en tête « {premiere[:40]} » (tri du registre réel : au traitement)")
    return r2


@point("67.3")
def p67_3(app, w, r2):
    page = w._resultats_page
    l_recent = page._lignes[-1] if page._lignes else None
    b = boutons(l_recent) if l_recent else {}
    etats = {t: bt.isEnabled() for t, bt in b.items()}
    b["Relancer avec un seuil changé"].click()
    pump(app, 8)
    etape = w._current_step
    structures = [e for e, on in w._detection_page._selected.items() if on]
    ok = "Ouvrir livrable/projet.qgs" not in etats and etats.get("Rapport") is False and etats.get("Relancer avec un seuil changé") is True and etape == 3 and "cratere" in structures
    verdict("67.3", ok, f"boutons (projet et rapport absents sur disque) {etats} ; Relancer → étape {etape}, structures {structures} (Ajouter / Ouvrir / Rapport réels : au traitement)")


@point("67.4")
def p67_4(app, w):
    w._goto_step(5)
    pump(app)
    w._run_view.revoir_resultats.emit()
    pump(app)
    verdict("67.4", w._current_step == 6, f"« Revoir les résultats » → étape {w._current_step} (traitement en tête : au traitement réel)")


# ----------------------------------------------------------------------------------------- § 68, § 70
@point("68.6")
def p68_6(app, w, recette):
    import archeo.src.ui.visualisation_tab as vt
    from qgis.PyQt.QtWidgets import QApplication
    d1, d2 = vt.DEFAULT_CATALOGUE, vt.LOCAL_CATALOGUE
    vt.DEFAULT_CATALOGUE, vt.LOCAL_CATALOGUE = Path("data/aucun/c.json"), Path("data/aucun2/c.json")
    try:
        tab = vt.VisualisationTab(RACINE)
        tab.show()
        pump(app)
        texte = tab._empty.text()
        actif = tab._refresh_btn.isEnabled()
        tab._sur_lien("manuel:consulter-les-flux")
        pump(app, 6)
        dlgs = [x for x in QApplication.topLevelWidgets() if type(x).__name__ == "AideDialog" and x.isVisible()]
        cle = dlgs[0]._cle if dlgs else ""
        for x in dlgs:
            x.close()
        tab.close()
    finally:
        vt.DEFAULT_CATALOGUE, vt.LOCAL_CATALOGUE = d1, d2
    ok = texte.startswith("Aucun catalogue n'est installé sur ce poste") and "Aide › Consulter les flux" in texte and actif and cle == "consulter-les-flux" and "injoignable" not in texte
    verdict("68.6", ok, f"message « {re.sub('<[^>]+>', '', texte)[:90]} » ; Actualiser actif {actif} ; lien → chapitre {cle}")


@point("70.1")
def p70_1(app, w):
    from archeo.src.app.services.vocabulaire import ENVIRONNEMENT
    from archeo.src.app.services.source_modes import mode_info
    from archeo.src.app import progress_stages
    page2, page3, page4 = w._indices_page, w._detection_page, w._launch_page
    w._goto_step(2)
    pump(app)
    compteur = page2._count_label.text()
    w._goto_step(3)
    pump(app)
    page3._on_entity_toggled("cratere", True)
    pump(app)
    tip = ligne(w, "cratere")._rvt_tag.toolTip()
    phrases = [re.sub("<[^>]+>", "", p) for p in labels(page3._runs_host) if "lit l'indice" in p]
    w._goto_step(4)
    pump(app, 8)
    recap = [t for t in labels(page4) if t == "Visualisation du relief"]
    etape5 = [t for t in labels(w._run_view._timeline) if t == "Visualisation du relief"]
    narr = (RACINE / "src" / "app" / "user_narrator.py").read_text(encoding="utf-8")
    flux = w._visu_tab._wall_meta.text() if w._visu_tab else ""
    sous_titre = [t for t in labels(w._visu_tab) if t.startswith("Produits déjà publiés")] if w._visu_tab else []
    stage = [v for v in vars(progress_stages).values() if isinstance(v, dict) and "Visualisation du relief" in v.values()]
    checks = {
        "rail": w._rail._items[1]._label.text() == "Visualisation du relief",
        "titre": w.TITLES[2][0] == "Visualisation du relief",
        "compteur": re.match(r"\d+ produits? cochés?$", compteur) is not None,
        "comparer": page2._comparer_btn.text() == "Comparer les produits ▸",
        "point de départ": mode_info("existing_rvt").banner_label == "Indices de visualisation existants",
        "infobulle sigle": tip.startswith("Indice de visualisation lu par le modèle"),
        "lit l'indice": bool(phrases),
        "récap": bool(recap),
        "environnement": ENVIRONNEMENT["Moteur d'indices intégré"] == "Calcul des indices de visualisation (moteur intégré)",
        "phase étape 5": bool(etape5) or bool(stage),
        "journal": "Calcul des indices de visualisation sur" in narr and '"produits : "' in narr,
        "flux": "produits ·" in flux and bool(sous_titre),
    }
    rates = [k for k, v in checks.items() if not v]
    verdict("70.1", not rates, f"{len(checks) - len(rates)}/{len(checks)} libellés conformes ; ratés {rates or 'aucun'} ; compteur « {compteur} » ; flux « {flux[:40]} »")


# ----------------------------------------------------------------------------------------- pilote
def main(argv=None) -> int:
    global LOG
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sortie", type=Path, required=True)
    parser.add_argument("--recette", type=Path, default=Path("D:/brouillons/recette_0.18.0"))
    args = parser.parse_args(argv)
    args.sortie.mkdir(parents=True, exist_ok=True)
    captures = args.sortie / "captures"
    captures.mkdir(exist_ok=True)
    LOG = open(args.sortie / "controles.txt", "w", encoding="utf-8")
    app = _app(1.0)
    from qgis.core import Qgis
    log("QGIS", Qgis.QGIS_VERSION)
    processing_ok = _processing_hors_ecran(log)
    from archeo.src.ui.wizard_dialog import WizardDialog

    w = WizardDialog()
    w.show()
    w.resize(980, 760)
    pump(app, 8)
    cfg_base = json.loads(json.dumps(w._config))
    cfg_base["app"]["files"].update({"data_mode": "ign_laz", "input_file": str(args.recette / "entrees" / "dalle_A.txt"),
                                     "output_dir": str(args.recette / "sorties" / "auto")})
    cfg_base["processing"]["products"] = {k: (k in ("MNT", "SVF", "LD")) for k in cfg_base["processing"]["products"]}
    cfg_base["computer_vision"]["enabled"] = True
    cfg_base["computer_vision"]["selected_entities"] = ["cratere", "parcellaire"]
    cfg_base["computer_vision"]["entity_thresholds"] = {}
    w._apply_config(cfg_base)
    pump(app, 8)

    for etape in (lambda: p61_1(app, w), lambda: p61_2(app, w), lambda: p61_3(app, w), lambda: p61_4(app, w),
                  lambda: p61_6(app, w, cfg_base), lambda: p61_7(app, w)):
        etape()
    verdict("61.5", None, "journal et rapport : au premier traitement réel (§ 66)")
    for etape in (lambda: p62_1(app, w, captures), lambda: p62_2(app, w, cfg_base, args.recette),
                  lambda: p62_3(app, w, cfg_base), lambda: p62_4(app, w)):
        etape()
    p_manuel("62.5", "etape-1-source", ["point de départ", "Choisir sur la carte", "Configuration", "introuvable"])
    for etape in (lambda: p63_1(app, w, cfg_base), lambda: p63_2(app, w), lambda: p63_3(app, w), lambda: p63_4(app, w),
                  lambda: p63_5(app, w), lambda: p63_6(app, w), lambda: p63_7(app, w)):
        etape()
    p_manuel("63.8", "etape-2-produits", ["Recommandé pour commencer", "ⓘ", "▸ Réglages", "Marge entre dalles", "Comparer les produits"])
    w._apply_config(cfg_base)
    pump(app)
    for fn in (p64_1, p64_2, p64_3, p64_4, p64_5, p64_6, p64_7, p64_8, p64_9, p64_10):
        fn(app, w)
    p_manuel("64.11", "etape-3-detection", ["fenêtre à part", "deux décimales", "conservés", "Ce qui sera analysé"])
    for fn in (p69_1, p69_2, p69_3, p69_4, p69_5, p69_6, p69_7):
        fn(app, w)
    w._apply_config(cfg_base)
    pump(app)
    if processing_ok:
        for etape in (lambda: p65_1(app, w, cfg_base, captures), lambda: p65_2(app, w, captures), lambda: p65_3(app, w, captures),
                      lambda: p65_4(app, w, cfg_base, args.recette, captures), lambda: p65_5(app, w), lambda: p65_6(app, w),
                      lambda: p65_7(app, w, cfg_base, args.recette)):
            etape()
    else:
        for n in ("65.1", "65.2", "65.3", "65.4", "65.5", "65.6", "65.7"):
            verdict(n, None, "Processing indisponible hors écran")
    p65_8(app, w)
    p65_9(app, w)
    p_manuel("65.10", "etape-4-verifications", ["Lancement impossible", "Environnement", "Revérifier", "Réglages du poste", "Oublier", "Ce qui sera produit", "modifier"])
    v = vue_fin(app)
    p66_1(app, v, captures)
    p66_2(app, w)
    p66_4(app, v)
    p66_5(app)
    p66_6(app)
    verdict("66.3", None, "journal de fichier : au traitement réel (marge à 0)")
    p_manuel("66.7", "etape-5-traitement", ["Ajouter au projet QGIS courant", "Relancer maintenant", "très probable 120", "pipeline_log"])
    v.close()
    p67_1(app, w)
    r2 = p67_2(app, w, args.recette)
    if r2 is not None:
        p67_3(app, w, r2)
    p67_4(app, w)
    verdict("67.5", None, "canevas QGIS : à jouer en séance")
    p_manuel("67.6", "resultats", ["Revoir un résultat"])
    from archeo.src.app.services.visu_catalogue import VISUALISATION_TAB_ENABLED
    if VISUALISATION_TAB_ENABLED:
        p68_6(app, w, args.recette)
        p_manuel("68.7", "consulter-les-flux", ["catalogue", "Afficher dans QGIS", "Tout retirer", "Voir dans QGIS"])
    else:
        for n in ("68.6", "68.7"):
            verdict(n, None, "onglet « Consulter les flux » masqué (VISUALISATION_TAB_ENABLED faux depuis la 0.18.1)")
    for n in ("68.1", "68.2", "68.3", "68.4", "68.5"):
        verdict(n, None, "traitement réel ou catalogue : à jouer en séance")
    p70_1(app, w)
    p_manuel("70.2", "etape-2-produits", ["Visualisation du relief", "Les produits", "indices de visualisation"])
    p_manuel("70.2b", "glossaire", ["| Produit |", "| Indice de visualisation |", "livrable"])
    verdict("21.2", None, "canevas QGIS : à jouer en séance")

    (args.sortie / "verdicts.json").write_text(json.dumps(VERDICTS, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = sum(1 for v in VERDICTS.values() if v["ok"] is True)
    ko = sum(1 for v in VERDICTS.values() if v["ok"] is False)
    nj = sum(1 for v in VERDICTS.values() if v["ok"] is None)
    log(f"RESULTAT : {ok} conformes, {ko} défauts, {nj} non joués")
    LOG.close()
    return 0


if __name__ == "__main__":
    code = main()
    os._exit(code)   # exitQgis fait une faute de segmentation bénigne hors écran
