"""Page « choisir puis cadrer » les vignettes des fiches de produits, dans le navigateur.

Pendant, côté produits, de ``page_choix.py``. Le geste n'est pas le même : une
classe se choisit un cadre parmi des candidats, un produit se choisit **quelle
fenêtre de terrain** l'illustre le mieux. La page présente donc une grille
produits × fenêtres — une case par couple — plus le cadrage de l'icône de 44 px,
posé une fois par fenêtre retenue.

Le choix par défaut est une seule fenêtre pour les douze produits : c'est ce qui
rend les vignettes comparables, et la comparaison est la raison d'être de ces
fiches. On ne s'en écarte que pour les produits dont le propos tient à un
terrain particulier — la couverture veut une zone lacunaire, la densité un
couvert forestier, le modèle d'altitude un bâti.

Usage ::

    python dev/fiches/page_choix_indices.py D:/brouillons/vignettes_indices \\
        --sortie page_indices.html

Le dossier est celui produit par ``candidats_indices.py`` (``NN_<CLÉ>.jpg`` et
``candidats.json``). Publier ensuite le HTML avec l'outil Artifact, capacité
``db`` déclarée ; la page enregistre ``choix/indices`` =
``{produits: {clé: n°}, cadrages: {n°: {x, y, cote}}}``, que Claude relit
(``read_db``) pour installer les vignettes (``appliquer_choix_indices.py``).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys

_RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_RACINE, "src"))

from app.services.visu_catalogue import indice_info  # noqa: E402

#: Côté des images embarquées dans la page (compromis poids / lisibilité).
TAILLE_PAGE = 384

GABARIT = r"""<title>Vignettes des produits</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap">
<style>
:root{
  --ink:#14181b; --ink-2:#3d464c; --muted:#5f6a71;
  --paper:#e6e9eb; --surface:#ffffff; --surface-2:#f3f5f6;
  --line:#c8ced2; --line-soft:#dfe4e7;
  --accent:#2b79c2; --accent-deep:#1d5a96; --accent-soft:#e2edf7;
  --ok:#2f7d4f; --ok-soft:#e4f1e9;
  --ocre:#8a5e18; --ocre-bg:#f6efdc; --ocre-line:#b07a20;
  --shadow:0 1px 2px rgba(15,25,35,.06), 0 6px 18px rgba(15,25,35,.07);
  --serif:"Spectral",Georgia,serif;
  --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ink:#e7ebee; --ink-2:#b9c3ca; --muted:#8d99a1;
    --paper:#101316; --surface:#191e22; --surface-2:#21272c;
    --line:#39424a; --line-soft:#2a3238;
    --accent:#5fa6e2; --accent-deep:#8fc3ef; --accent-soft:#1a2a38;
    --ok:#63b587; --ok-soft:#17281f;
    --ocre:#e0b463; --ocre-bg:#31281a; --ocre-line:#8a6b2c;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 6px 18px rgba(0,0,0,.34);
  }
}
:root[data-theme="dark"]{
  --ink:#e7ebee; --ink-2:#b9c3ca; --muted:#8d99a1;
  --paper:#101316; --surface:#191e22; --surface-2:#21272c;
  --line:#39424a; --line-soft:#2a3238;
  --accent:#5fa6e2; --accent-deep:#8fc3ef; --accent-soft:#1a2a38;
  --ok:#63b587; --ok-soft:#17281f;
  --ocre:#e0b463; --ocre-bg:#31281a; --ocre-line:#8a6b2c;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 6px 18px rgba(0,0,0,.34);
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);font-family:var(--sans);font-size:14px;line-height:1.55;
  margin:0;padding-block:0 70px;padding-left:20px;padding-right:20px}
.wrap{max-width:1240px;margin:0 auto}
.bar{position:sticky;top:env(safe-area-inset-top,0px);z-index:20;margin:0 -20px;padding:11px 20px;
  background:color-mix(in srgb,var(--paper) 88%,transparent);backdrop-filter:blur(8px);
  border-bottom:1px solid var(--line);display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.bar h1{font-family:var(--serif);font-weight:600;font-size:17px;margin:0}
.etat{font-size:12px;color:var(--muted);margin-left:auto;font-family:var(--mono)}
.etat.ok{color:var(--ok)}
h2{font-family:var(--serif);font-weight:600;font-size:19px;margin:30px 0 4px}
p.aide{color:var(--ink-2);margin:0 0 14px;max-width:72ch}
.avis{background:var(--ocre-bg);border:1px solid var(--ocre-line);color:var(--ocre);
  padding:9px 13px;border-radius:7px;margin:14px 0;font-size:13px}
/* ---- fenêtres candidates ---- */
.fen{background:var(--surface);border:1px solid var(--line);border-radius:9px;padding:11px 13px;
  margin-bottom:11px;box-shadow:var(--shadow);cursor:pointer}
.fen[aria-pressed="true"]{border-color:var(--accent);background:var(--accent-soft)}
.fen .tete{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:8px}
.fen .num{font-family:var(--mono);font-weight:500;color:var(--accent-deep)}
.fen .pourquoi{color:var(--ink-2);font-size:13px}
.fen .meta{font-family:var(--mono);font-size:11px;color:var(--muted);margin-left:auto}
.fen .compte{font-family:var(--mono);font-size:11px;padding:1px 8px;border-radius:999px;
  border:1px solid var(--line);color:var(--muted);white-space:nowrap}
.fen .compte[data-actif="oui"]{background:var(--accent);border-color:var(--accent-deep);color:#fff}
.fen .compte[data-actif="partiel"]{background:var(--accent-soft);border-color:var(--accent);
  color:var(--accent-deep)}
.fen[aria-pressed="true"] .bande img{border-color:var(--accent)}
.bande{display:flex;gap:3px;overflow-x:auto;padding-bottom:3px}
.bande figure{margin:0;flex:0 0 auto;width:78px}
.bande img{display:block;width:78px;height:78px;object-fit:cover;border-radius:3px;border:1px solid var(--line-soft)}
.bande figcaption{font-family:var(--mono);font-size:9px;color:var(--muted);text-align:center;margin-top:2px}
/* ---- grille produits × fenêtres ---- */
.tbl{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:9px;
  box-shadow:var(--shadow)}
table{border-collapse:collapse;width:100%}
th,td{padding:5px 7px;border-bottom:1px solid var(--line-soft);text-align:left;vertical-align:middle}
thead th{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
  background:var(--surface)}
td.prod{white-space:nowrap}
td.prod b{font-family:var(--mono);color:var(--accent-deep)}
td.prod span{color:var(--muted);font-size:12px;margin-left:7px}
.cell{border:2px solid transparent;border-radius:4px;padding:0;background:none;cursor:pointer;
  display:block;line-height:0;width:62px;flex:0 0 62px}
.cell img{width:58px;height:58px;object-fit:cover;border-radius:3px;display:block}
.cell[aria-pressed="true"]{border-color:var(--accent);box-shadow:0 0 0 2px var(--accent-soft)}
.cell:focus-visible{outline:2px solid var(--accent-deep);outline-offset:2px}
/* ---- cadrage ---- */
.cadres{display:flex;flex-wrap:wrap;gap:22px}
.cadreur{background:var(--surface);border:1px solid var(--line);border-radius:9px;padding:13px;
  box-shadow:var(--shadow)}
.scene{position:relative;width:384px;max-width:100%;cursor:crosshair;touch-action:none;
  user-select:none;overflow:hidden;border-radius:4px;background:var(--surface-2);
  border:1px solid var(--line)}
.scene img{display:block;width:100%}
.scene:focus-visible{outline:2px solid var(--accent-deep);outline-offset:2px}
.fenetre{position:absolute;border:2px solid #ffd34d;background:rgba(255,211,77,.07);
  box-shadow:0 0 0 9999px rgba(10,16,22,.5);cursor:grab}
.fenetre:active{cursor:grabbing}
.fenetre::after{content:"";position:absolute;right:-7px;bottom:-7px;width:14px;height:14px;
  background:#ffd34d;border:2px solid rgba(20,24,27,.7);border-radius:2px;cursor:nwse-resize}
.apercu{display:flex;align-items:center;gap:12px;margin-top:11px;flex-wrap:wrap}
.apercu canvas{border:1px solid var(--line);border-radius:3px}
.apercu .vraie{width:44px;height:44px}
.apercu .zoom{width:112px;height:112px}
.apercu .leg{font-family:var(--mono);font-size:9px;color:var(--muted);text-align:center;
  display:block;margin-top:2px}
.apercu .txt{font-size:12px;color:var(--muted)}
.reglages{display:flex;align-items:center;gap:10px;margin-top:9px;flex-wrap:wrap}
input[type=range]{width:220px}
button.plain{font:inherit;background:var(--surface-2);border:1px solid var(--line);border-radius:6px;
  padding:4px 11px;cursor:pointer;color:var(--ink)}
button.plain:hover{border-color:var(--accent)}
.pied{margin-top:28px;color:var(--muted);font-size:12.5px;max-width:78ch}
@media (max-width:720px){.scene{width:100%}}
</style>

<div class="wrap">
  <div class="bar">
    <h1>Vignettes des produits</h1>
    <span class="etat" id="etat">—</span>
  </div>

  <p class="aide" style="margin-top:16px">__LEDE__</p>
  <div class="avis" id="avis" hidden><span id="avis-txt"></span></div>

  <h2>1 · La fenêtre de référence</h2>
  <p class="aide">Clique la fenêtre qui illustrera les douze produits. C'est le
  même terrain partout : c'est ce qui permet de comparer un indice à l'autre,
  et c'est toute la raison d'être de ces fiches.</p>
  <div id="fenetres"></div>

  <h2>2 · Les exceptions</h2>
  <p class="aide">Trois produits ne parlent pas du relief et gagnent souvent une
  autre fenêtre : la couverture veut une zone lacunaire, la densité un couvert
  forestier ou un plan d'eau, le modèle d'altitude un bâti dense. Change la case
  d'une ligne pour lui donner sa propre fenêtre.</p>
  <div class="tbl">
    <table>
      <thead><tr><th>Produit</th><th id="entetes-fen"></th></tr></thead>
      <tbody id="grille"></tbody>
    </table>
  </div>

  <h2>3 · Le cadrage de l'icône</h2>
  <p class="aide">Chaque carte de l'étape 2 porte une icône de 44 px. La
  vignette entière couvre __EMPRISE__ m de terrain : réduite telle quelle, elle
  devient une bouillie grise. Pose la fenêtre de l'icône sur la zone qui porte
  la structure — un cadrage par fenêtre retenue. Saisis le carré pour le
  déplacer, tire sa poignée en bas à droite pour le redimensionner, ou clique
  ailleurs dans l'image pour l'y poser. Au clavier : flèches pour déplacer,
  <code>+</code> et <code>-</code> pour la taille, <kbd>Maj</kbd> pour aller
  plus vite. Les deux aperçus montrent le même cadre : à sa taille réelle de
  44 px, et agrandi pour le juger.</p>
  <div class="cadres" id="cadreurs"></div>

  <p class="pied" id="pied"></p>
</div>

<script>
const DATA = __DATA__;
let db = null;
const choix = {produits: {}, cadrages: {}};
const timers = {};

// --- utilitaires -------------------------------------------------------
const $ = (s, r) => (r || document).querySelector(s);
const el = (tag, cls, txt) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (txt != null) n.textContent = txt;
  return n;
};
const img64 = (src) => { const i = new Image(); i.src = src; return i; };

// --- état par défaut : tout sur la première fenêtre ---------------------
function defauts() {
  const premier = DATA.candidats[0] ? DATA.candidats[0].candidat : 0;
  DATA.ordre.forEach(k => { choix.produits[k] = premier; });
  choix.cadrages[premier] = {x: 0.25, y: 0.25, cote: 0.5};
}

// --- 1. fenêtres candidates -------------------------------------------
function rendreFenetres() {
  const hote = $("#fenetres");
  hote.textContent = "";
  DATA.candidats.forEach(c => {
    const b = el("div", "fen");
    b.setAttribute("role", "button");
    b.setAttribute("tabindex", "0");
    b.dataset.n = c.candidat;
    const tete = el("div", "tete");
    tete.append(el("span", "num", "Fenêtre " + c.candidat));
    tete.append(el("span", "pourquoi", c.pourquoi));
    const compte = el("span", "compte");
    compte.dataset.n = c.candidat;
    tete.append(compte);
    tete.append(el("span", "meta", c.dalle + " · " + c.emprise_m + " m"));
    b.append(tete);
    const bande = el("div", "bande");
    DATA.ordre.forEach(k => {
      const f = el("figure");
      const im = img64(DATA.images[c.candidat][k]);
      im.alt = k;
      f.append(im, el("figcaption", null, DATA.sigles[k]));
      bande.append(f);
    });
    b.append(bande);
    const prendre = () => { toutSur(+b.dataset.n); };
    b.addEventListener("click", prendre);
    b.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); prendre(); }
    });
    hote.append(b);
  });
}

function toutSur(n) {
  DATA.ordre.forEach(k => { choix.produits[k] = n; });
  rafraichir();
  planifier();
}

// --- 2. grille produits × fenêtres -------------------------------------
function rendreGrille() {
  // En-tête aligné sur les cases : même conteneur flex, mêmes largeurs.
  const th = $("#entetes-fen");
  th.textContent = "";
  th.style.display = "flex";
  th.style.gap = "6px";
  DATA.candidats.forEach(c => {
    const n = el("span", null, "n° " + c.candidat);
    n.style.cssText = "width:62px;text-align:center";
    n.title = c.pourquoi;
    th.append(n);
  });
  const corps = $("#grille");
  corps.textContent = "";
  DATA.ordre.forEach(k => {
    const tr = el("tr");
    const td = el("td", "prod");
    td.append(el("b", null, DATA.sigles[k]));
    td.append(el("span", null, DATA.titres[k]));
    tr.append(td);
    const cases = el("td");
    cases.style.display = "flex";
    cases.style.gap = "6px";
    DATA.candidats.forEach(c => {
      const b = el("button", "cell");
      b.type = "button";
      b.dataset.produit = k;
      b.dataset.n = c.candidat;
      b.title = "Fenêtre " + c.candidat + " — " + c.pourquoi;
      b.setAttribute("aria-label", DATA.sigles[k] + " · fenêtre " + c.candidat);
      b.append(img64(DATA.images[c.candidat][k]));
      b.addEventListener("click", () => {
        choix.produits[k] = c.candidat;
        rafraichir();
        planifier();
      });
      cases.append(b);
    });
    tr.append(cases);
    corps.append(tr);
  });
}

// --- 3. cadrage --------------------------------------------------------
function fenetresUtilisees() {
  return [...new Set(Object.values(choix.produits))].sort((a, b) => a - b);
}

let cadreursMontes = "";   // signature des fenêtres actuellement affichées

function rendreCadreurs() {
  const utiles = fenetresUtilisees();
  const signature = utiles.join(",");
  const hote = $("#cadreurs");
  if (signature === cadreursMontes) {
    // Mêmes fenêtres : seules les listes de produits ont pu bouger. Reconstruire
    // tout remettrait chaque cadre à zéro et effacerait l'aperçu à chaque clic.
    utiles.forEach(n => {
      const sous = hote.querySelector('[data-sous="' + n + '"]');
      if (sous) sous.textContent = DATA.ordre
        .filter(k => choix.produits[k] === n).map(k => DATA.sigles[k]).join(" · ");
    });
    return;
  }
  cadreursMontes = signature;
  hote.textContent = "";
  utiles.forEach(n => {
    if (!choix.cadrages[n]) choix.cadrages[n] = {x: 0.25, y: 0.25, cote: 0.5};
    const c = DATA.candidats.find(x => x.candidat === n);
    const produits = DATA.ordre.filter(k => choix.produits[k] === n);
    const carte = el("div", "cadreur");
    carte.append(el("div", "num", "Fenêtre " + n));
    const sous = el("div", "txt");
    sous.dataset.sous = n;
    sous.style.cssText = "font-size:12px;color:var(--muted);margin-bottom:9px";
    sous.textContent = produits.map(k => DATA.sigles[k]).join(" · ");
    carte.append(sous);

    // La scène montre l'indice le plus lisible du lot ; le cadrage vaut pour tous.
    const ref = produits.includes("CVAT") ? "CVAT" : produits[0];
    const scene = el("div", "scene");
    scene.tabIndex = 0;
    scene.setAttribute("role", "application");
    scene.setAttribute("aria-label", "Cadrage de l'icône, fenêtre " + n);
    const im = img64(DATA.images[n][ref]);
    im.alt = "Fenêtre " + n + " en " + DATA.sigles[ref];
    const cadre = el("div", "fenetre");
    scene.append(im, cadre);
    carte.append(scene);

    // Deux aperçus : la taille réelle de l'icône, et un agrandissement pour
    // juger le cadre. Les canevas sont dimensionnés à la densité de l'écran,
    // sinon l'aperçu est flou là où la carte, elle, sera nette.
    const dpr = Math.max(1, Math.min(3, window.devicePixelRatio || 1));
    const bas = el("div", "apercu");
    const faire = (cls, cote) => {
      const boite = el("div");
      const c = el("canvas", cls);
      c.width = cote * dpr; c.height = cote * dpr;
      boite.append(c);
      boite.append(el("span", "leg", cote === 44 ? "44 px réels" : "agrandi"));
      bas.append(boite);
      return c;
    };
    const cvVraie = faire("vraie", 44);
    const cvZoom = faire("zoom", 112);
    const txt = el("span", "txt");
    bas.append(txt);
    carte.append(bas);

    const reglages = el("div", "reglages");
    const curseur = el("input");
    curseur.type = "range"; curseur.min = "10"; curseur.max = "100"; curseur.step = "1";
    curseur.setAttribute("aria-label", "Taille de la fenêtre de l'icône");
    const reset = el("button", "plain", "Centrer");
    reglages.append(curseur, reset);
    carte.append(reglages);
    hote.append(carte);

    const maj = () => {
      const g = choix.cadrages[n];
      // Le carré se pose en POURCENTAGES : clientWidth vaut 0 tant que la carte
      // n'est pas mise en page, et le cadre restait alors invisible.
      cadre.style.left = (g.x * 100) + "%";
      cadre.style.top = (g.y * 100) + "%";
      cadre.style.width = (g.cote * 100) + "%";
      cadre.style.height = (g.cote * 100) + "%";
      curseur.value = Math.round(g.cote * 100);
      txt.textContent = Math.round(g.cote * (c ? c.emprise_m : 0)) + " m de côté";
      [[cvVraie, 44], [cvZoom, 112]].forEach(([cv, cote]) => {
        const ctx = cv.getContext("2d");
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, cv.width, cv.height);
        if (!(im.complete && im.naturalWidth)) return;
        const s = im.naturalWidth;
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";
        ctx.drawImage(im, g.x * s, g.y * s, g.cote * s, g.cote * s,
                      0, 0, cote * dpr, cote * dpr);
      });
    };
    const MIN_COTE = 0.10;
    const borne = g => {
      const cote = Math.min(Math.max(MIN_COTE, g.cote), 1);
      return {
        cote,
        x: Math.min(Math.max(0, g.x), 1 - cote),
        y: Math.min(Math.max(0, g.y), 1 - cote),
      };
    };
    // Redimensionner CONSERVE LE CENTRE. Ne borner que x et y, comme on le
    // faisait, laisse le coin haut-gauche fixe : on serre sur une structure
    // centrée et elle sort du cadre par le bas à droite.
    const redimensionner = cote => {
      const g = choix.cadrages[n];
      const c2 = Math.min(Math.max(MIN_COTE, cote), 1);
      choix.cadrages[n] = borne({
        cote: c2, x: g.x + (g.cote - c2) / 2, y: g.y + (g.cote - c2) / 2,
      });
      maj(); planifier();
    };
    const deplacer = (dx, dy) => {
      const g = choix.cadrages[n];
      choix.cadrages[n] = borne({cote: g.cote, x: g.x + dx, y: g.y + dy});
      maj(); planifier();
    };

    // — glisser le cadre, ou tirer sa poignée au coin bas-droit —
    let mode = null, depart = null, aGlisse = false;
    cadre.addEventListener("pointerdown", ev => {
      const r = cadre.getBoundingClientRect();
      mode = (ev.clientX > r.right - 16 && ev.clientY > r.bottom - 16) ? "taille" : "deplace";
      depart = {mx: ev.clientX, my: ev.clientY, ...choix.cadrages[n],
                boite: scene.getBoundingClientRect()};
      aGlisse = false;
      cadre.setPointerCapture(ev.pointerId);
      ev.preventDefault();
    });
    cadre.addEventListener("pointermove", ev => {
      if (!mode || !depart.boite.width) return;
      aGlisse = true;
      const dx = (ev.clientX - depart.mx) / depart.boite.width;
      const dy = (ev.clientY - depart.my) / depart.boite.height;
      choix.cadrages[n] = mode === "deplace"
        ? borne({cote: depart.cote, x: depart.x + dx, y: depart.y + dy})
        : borne({cote: depart.cote + Math.max(dx, dy), x: depart.x, y: depart.y});
      maj();
    });
    const finGlisse = ev => {
      if (!mode) return;
      mode = null;
      planifier();
      try { cadre.releasePointerCapture(ev.pointerId); } catch (e) { /* déjà relâché */ }
    };
    cadre.addEventListener("pointerup", finGlisse);
    cadre.addEventListener("pointercancel", finGlisse);

    // — cliquer ailleurs dans l'image pose le cadre là —
    scene.addEventListener("click", ev => {
      // Un glisser se termine par un click sur la scène : sans cette garde, le
      // cadre sautait au point de relâchement juste après avoir été glissé.
      if (aGlisse) { aGlisse = false; return; }
      const r = scene.getBoundingClientRect();
      if (!r.width || !r.height) return;
      const g = choix.cadrages[n];
      deplacer(
        (ev.clientX - r.left) / r.width - g.cote / 2 - g.x,
        (ev.clientY - r.top) / r.height - g.cote / 2 - g.y
      );
    });

    // — clavier : le cadrage doit être réglable sans souris —
    scene.addEventListener("keydown", ev => {
      const pas = ev.shiftKey ? 0.10 : 0.01;
      const gestes = {
        ArrowLeft: () => deplacer(-pas, 0), ArrowRight: () => deplacer(pas, 0),
        ArrowUp: () => deplacer(0, -pas), ArrowDown: () => deplacer(0, pas),
        "+": () => redimensionner(choix.cadrages[n].cote + pas),
        "=": () => redimensionner(choix.cadrages[n].cote + pas),
        "-": () => redimensionner(choix.cadrages[n].cote - pas),
      };
      if (gestes[ev.key]) { gestes[ev.key](); ev.preventDefault(); }
    });

    // Pas d'écouteur sur la molette : elle confisquait le défilement de la page
    // dès que le curseur passait sur l'image, avançait par paliers fixes quel
    // que soit le geste, et un défilement horizontal (deltaY nul) zoomait.
    // Le curseur ci-dessous et la poignée du cadre font le même travail.
    curseur.addEventListener("input", () => redimensionner(+curseur.value / 100));
    reset.addEventListener("click", () => {
      const g = choix.cadrages[n];
      choix.cadrages[n] = borne({cote: g.cote, x: (1 - g.cote) / 2, y: (1 - g.cote) / 2});
      maj(); planifier();
    });
    // L'image est une data URI : selon le navigateur elle peut être décodée
    // après ce tour de boucle. Sans ce redessin, l'aperçu reste vide.
    im.addEventListener("load", maj);
    if (im.decode) im.decode().then(maj).catch(() => {});
    maj();
  });
}

// --- rafraîchissement global ------------------------------------------
// La fenêtre mise en avant est celle qui sert au PLUS de produits : sans cela,
// donner sa propre fenêtre à un seul produit éteignait toute la section 1 et
// plus rien ne paraissait choisi.
function dominante() {
  const n = {};
  DATA.ordre.forEach(k => { n[choix.produits[k]] = (n[choix.produits[k]] || 0) + 1; });
  let best = null, max = -1;
  Object.keys(n).forEach(k => { if (n[k] > max) { max = n[k]; best = +k; } });
  return {fenetre: best, combien: max, total: DATA.ordre.length};
}

function rafraichir() {
  const dom = dominante();
  document.querySelectorAll(".fen").forEach(b => {
    const n = +b.dataset.n;
    const combien = DATA.ordre.filter(k => choix.produits[k] === n).length;
    b.setAttribute("aria-pressed", n === dom.fenetre ? "true" : "false");
    const c = b.querySelector(".compte");
    if (c) {
      c.textContent = combien + " / " + DATA.ordre.length + " produits";
      c.dataset.actif = combien === DATA.ordre.length ? "oui"
                      : combien > 0 ? "partiel" : "non";
    }
  });
  document.querySelectorAll(".cell").forEach(b => {
    b.setAttribute("aria-pressed",
      choix.produits[b.dataset.produit] === +b.dataset.n ? "true" : "false");
  });
  rendreCadreurs();
}

// --- enregistrement ----------------------------------------------------
function planifier() { clearTimeout(timers.t); timers.t = setTimeout(enregistrer, 350); }

async function enregistrer() {
  const utiles = fenetresUtilisees();
  const doc = {
    produits: {...choix.produits},
    cadrages: Object.fromEntries(utiles.map(n => [n, choix.cadrages[n]])),
  };
  if (!db) { etat("hors ligne", false); return; }
  try {
    await db.doc("choix/indices").set(doc);
    etat("enregistré", true);
  } catch (e) {
    etat("échec de l'enregistrement", false);
  }
}

function etat(t, ok) {
  const e = $("#etat");
  e.textContent = t;
  e.className = "etat" + (ok ? " ok" : "");
}

function avis(t) {
  $("#avis-txt").textContent = t;
  $("#avis").hidden = false;
}

// --- démarrage ---------------------------------------------------------
defauts();
rendreFenetres();
rendreGrille();
rafraichir();
$("#pied").textContent = DATA.pied;

(async () => {
  db = await (window.claude?.use?.("db") ?? Promise.resolve(null));
  if (!db) {
    avis("Enregistrement partagé indisponible : tes choix restent dans ce navigateur.");
    return;
  }
  etat("prêt", false);
  db.doc("choix/indices").onSnapshot(
    d => { if (d && d.produits) { etat("enregistré", true); } },
    () => {}
  );
})();
</script>
"""


def _b64(chemin: str, taille: int) -> str:
    """JPEG redimensionné, en data URI."""
    from PIL import Image

    im = Image.open(chemin).convert("RGB")
    if im.width > taille:
        im = im.resize((taille, taille), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def construire(dossier: str, taille: int = TAILLE_PAGE) -> str:
    meta = json.loads(
        io.open(os.path.join(dossier, "candidats.json"), encoding="utf-8").read()
    )
    ordre = meta["ordre"]
    images = {}
    for c in meta["candidats"]:
        images[c["candidat"]] = {
            k: _b64(os.path.join(dossier, nom), taille)
            for k, nom in c["images"].items()
        }
    data = {
        "ordre": ordre,
        "sigles": {k: indice_info(k).sigle for k in ordre},
        "titres": {k: indice_info(k).metier for k in ordre},
        "candidats": [
            {k: v for k, v in c.items() if k != "images"} for c in meta["candidats"]
        ],
        "images": images,
        "pied": (
            f"{len(meta['candidats'])} fenêtres de {meta['emprise_m']} m tirées de "
            f"{os.path.basename(meta['run'])}. Les vignettes retenues seront copiées "
            "dans data/indices_vignettes/ et déclarées dans data/indices_fiches.json."
        ),
    }
    return (
        GABARIT
        .replace("__EMPRISE__", str(meta["emprise_m"]))
        .replace(
            "__LEDE__",
            "Les douze produits de l'étape 2 ont leur fiche, mais aucune illustration. "
            "Choisis la fenêtre de terrain qui les montre le mieux, donne au besoin sa "
            "propre fenêtre à la couverture, à la densité et au modèle d'altitude, puis "
            "pose la fenêtre de l'icône de 44 px.",
        )
        .replace("__DATA__", json.dumps(data, ensure_ascii=False))
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dossier", help="dossier produit par candidats_indices.py")
    ap.add_argument("--taille", type=int, default=TAILLE_PAGE,
                    help=f"côté des images embarquées (défaut {TAILLE_PAGE})")
    ap.add_argument("--sortie", default="page_choix_indices.html")
    a = ap.parse_args()
    page = construire(a.dossier, a.taille)
    with open(a.sortie, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"page : {a.sortie} ({os.path.getsize(a.sortie) / 1e6:.1f} Mo) — publier avec "
          "Artifact, capabilities {db: {}} ; relire le choix : read_db collection "
          "'choix', doc 'indices'")


if __name__ == "__main__":
    main()
