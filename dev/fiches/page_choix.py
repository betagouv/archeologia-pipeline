"""Page « choisir puis cadrer » les vignettes d'une classe, à publier dans le navigateur.

Gabarit de la campagne des fiches (2026-09-10, page « Vignette des tranchées »),
rendu réutilisable le 2026-09-15 : l'utilisateur clique les cadres retenus (l'ordre
des clics fait l'ordre de la fiche, le premier est l'icône), puis pose sur chacun
la fenêtre de l'icône 44 px avec aperçu à taille réelle. La page enregistre le
choix dans la base partagée de l'artefact (capacité ``db``, document
``choix/<classe>`` = ``{classe, modele, retenues, cadrages}``), que Claude relit
ensuite (``read_db``) pour installer les vignettes (``appliquer_choix.py``).

Usage ::

    python dev/fiches/page_choix.py <dossier_candidats> --classe zone_crateres \\
        --label "Regroupement de cratères" --modele crateres_seg_ld_v1 \\
        --corpus "run detection_carrieres (Gard)" --objets "cratères regroupés" \\
        --sortie page.html

``<dossier_candidats>`` contient ``N_brut.jpg``, ``N_annote.jpg`` (optionnel) et
``candidats.json`` (liste de ``{candidat, ...}``) ; les champs ``zone``/``secteur``,
``profil``/``pourquoi``, ``n_crateres``/``objets``, ``emprise_m`` et ``tuiles`` ou
``cible`` sont repris s'ils existent. Publier ensuite le HTML avec l'outil
Artifact (``capabilities: {"db": {}}``).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys

GABARIT = r"""<title>__TITRE__</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{
  --ink:#14181b; --ink-2:#3d464c; --muted:#5f6a71;
  --paper:#e6e9eb; --surface:#ffffff; --surface-2:#f3f5f6;
  --line:#c8ced2; --line-soft:#dfe4e7;
  --accent:#2b79c2; --accent-deep:#1d5a96; --accent-soft:#e2edf7;
  --ok:#2f7d4f; --ok-soft:#e4f1e9;
  --ocre:#8a5e18; --ocre-bg:#f6efdc; --ocre-line:#b07a20;
  --voile:rgba(12,20,28,.58);
  --shadow:0 1px 2px rgba(15,25,35,.06), 0 6px 18px rgba(15,25,35,.07);
  --shadow-on:0 2px 4px rgba(43,121,194,.18), 0 10px 26px rgba(43,121,194,.20);
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
    --voile:rgba(4,8,12,.68);
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 6px 18px rgba(0,0,0,.34);
    --shadow-on:0 2px 4px rgba(95,166,226,.25), 0 10px 26px rgba(0,0,0,.4);
  }
}
:root[data-theme="dark"]{
  --ink:#e7ebee; --ink-2:#b9c3ca; --muted:#8d99a1;
  --paper:#101316; --surface:#191e22; --surface-2:#21272c;
  --line:#39424a; --line-soft:#2a3238;
  --accent:#5fa6e2; --accent-deep:#8fc3ef; --accent-soft:#1a2a38;
  --ok:#63b587; --ok-soft:#17281f;
  --ocre:#e0b463; --ocre-bg:#31281a; --ocre-line:#8a6b2c;
  --voile:rgba(4,8,12,.68);
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 6px 18px rgba(0,0,0,.34);
  --shadow-on:0 2px 4px rgba(95,166,226,.25), 0 10px 26px rgba(0,0,0,.4);
}
*{box-sizing:border-box}
body{background:var(--paper); color:var(--ink); font-family:var(--sans); font-size:14px; line-height:1.55; margin:0;
  padding-block:0 70px; padding-left:20px; padding-right:20px}
.wrap{max-width:1180px; margin:0 auto}
.bar{position:sticky; top:0; z-index:20; margin:0 -20px; padding:11px 20px;
  background:color-mix(in srgb, var(--paper) 88%, transparent); backdrop-filter:blur(8px); border-bottom:1px solid var(--line);
  display:flex; align-items:center; gap:14px; flex-wrap:wrap}
.bar h1{font-family:var(--serif); font-weight:600; font-size:17px; margin:0; white-space:nowrap}
.bar .tag{font-family:var(--mono); font-size:10.5px; padding:2px 8px; border-radius:3px; border:1px solid var(--accent); color:var(--accent-deep); background:var(--accent-soft)}
.etat{font-family:var(--mono); font-size:11.5px; color:var(--muted); margin-left:auto}
.etat b{color:var(--ok)}
.intro{padding-block:22px 4px; max-width:74ch}
.intro p{margin:0 0 9px; color:var(--ink-2)}
.intro .lede{font-size:16px; color:var(--ink)}
.avis{display:flex; gap:9px; align-items:flex-start; font-size:12.5px; background:var(--ocre-bg); border:1px solid var(--ocre-line); color:var(--ocre); border-radius:5px; padding:9px 12px; margin-top:4px}
h2.sect{font-family:var(--serif); font-weight:600; font-size:20px; margin:26px 0 4px}
p.sect{margin:0 0 14px; color:var(--muted); font-size:12.5px}
.outils{display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:12px}
.bascule{display:flex; gap:0}
.bascule button, .btn{font:inherit; font-size:11.5px; padding:3px 12px; cursor:pointer; border-radius:3px; border:1px solid var(--line); background:var(--surface); color:var(--ink-2)}
.bascule button:first-child{border-radius:3px 0 0 3px}
.bascule button:last-child{border-radius:0 3px 3px 0; border-left:0}
.bascule button[aria-pressed="true"]{background:var(--accent); border-color:var(--accent-deep); color:#fff}
.btn:hover, .bascule button:hover{border-color:var(--accent)}
.grille{display:grid; gap:14px; grid-template-columns:repeat(auto-fill, minmax(230px,1fr))}
.carte{display:flex; flex-direction:column; text-align:left; font:inherit; cursor:pointer; background:var(--surface); color:var(--ink); padding:0; overflow:hidden;
  border:2px solid var(--line-soft); border-radius:8px; box-shadow:var(--shadow); transition:border-color .12s, box-shadow .12s, transform .12s}
.carte:hover{border-color:var(--accent)}
.carte[aria-pressed="true"]{border-color:var(--accent); box-shadow:var(--shadow-on)}
@media (prefers-reduced-motion:no-preference){ .carte[aria-pressed="true"]{transform:translateY(-2px)} }
.vue{position:relative; aspect-ratio:1; background:var(--surface-2)}
.vue img{width:100%; height:100%; object-fit:cover; display:block}
.vue img.off{display:none}
.marque{position:absolute; top:8px; right:8px; min-width:24px; height:24px; padding:0 6px; border-radius:12px; display:grid; place-items:center; font-size:12px; font-weight:700;
  background:var(--surface); color:var(--muted); border:2px solid var(--line)}
.carte[aria-pressed="true"] .marque{background:var(--accent); border-color:var(--accent-deep); color:#fff}
.rang{position:absolute; top:8px; left:8px; font-family:var(--mono); font-size:10px; padding:1px 6px; border-radius:3px; background:rgba(10,16,22,.72); color:#fff}
.pied{padding:9px 11px 11px; display:flex; flex-direction:column; gap:5px}
.pied .lieu{font-size:12px; font-weight:600}
.pied .chiffres{font-family:var(--mono); font-size:10.5px; color:var(--muted)}
.pied .pourquoi{font-size:11.5px; color:var(--ink-2); line-height:1.45}
.cadres{display:flex; flex-direction:column; gap:16px}
.cadreur{background:var(--surface); border:1px solid var(--line-soft); border-radius:8px; box-shadow:var(--shadow); padding:16px;
  display:grid; grid-template-columns:minmax(260px,360px) 1fr; gap:20px; align-items:start}
.cadreur h3{margin:0 0 8px; font-size:13.5px; font-weight:600}
.cadreur h3 span{font-family:var(--mono); font-size:10.5px; color:var(--muted); font-weight:400}
.scene{position:relative; user-select:none; touch-action:none; width:100%; aspect-ratio:1; border:1px solid var(--line); border-radius:4px; overflow:hidden; background:var(--surface-2)}
.scene img{position:absolute; inset:0; width:100%; height:100%; object-fit:cover; display:block}
.scene img.off{visibility:hidden}
.voile{position:absolute; inset:0; background:var(--voile); pointer-events:none}
.fen{position:absolute; border:2px solid #ffd34d; cursor:grab; background:rgba(255,211,77,.07); box-shadow:0 2px 10px rgba(0,0,0,.45)}
.fen:active{cursor:grabbing}
.fen::after{content:""; position:absolute; right:-7px; bottom:-7px; width:14px; height:14px; background:#ffd34d; border:2px solid rgba(20,24,27,.7); border-radius:2px; cursor:nwse-resize}
.trou{position:absolute; overflow:hidden; pointer-events:none}
.trou img{position:absolute; object-fit:cover; max-width:none}
.coords{font-family:var(--mono); font-size:10.5px; color:var(--muted); font-variant-numeric:tabular-nums}
.apercus{display:flex; flex-direction:column; gap:14px}
.rangee{display:flex; gap:16px; align-items:flex-end; flex-wrap:wrap}
.vue2{display:flex; flex-direction:column; gap:5px; align-items:center}
.boite{border:1px solid var(--line); border-radius:3px; overflow:hidden; background:var(--surface-2)}
.lab{font-family:var(--mono); font-size:9.5px; color:var(--muted)}
.carte-demo{display:grid; grid-template-columns:44px 1fr; gap:9px; align-items:start; border:1px solid var(--line); border-radius:6px; padding:8px 10px; max-width:320px}
.carte-demo .titre{font-weight:600; font-size:13px}
.carte-demo .desc{font-size:10.5px; color:var(--muted)}
.aide{font-size:11.5px; color:var(--muted); margin:0}
kbd{font-family:var(--mono); font-size:10px; padding:0 4px; border-radius:3px; border:1px solid var(--line); background:var(--surface-2); color:var(--ink-2)}
.vide{border:1px dashed var(--line); border-radius:8px; padding:22px; text-align:center; color:var(--muted); font-size:12.5px}
footer{margin-top:30px; padding-top:16px; border-top:1px solid var(--line); font-size:12.5px; color:var(--muted); max-width:74ch}
:focus-visible{outline:2px solid var(--accent); outline-offset:2px}
@media (max-width:820px){ .cadreur{grid-template-columns:1fr} }
</style>

<div class="wrap">
  <div class="bar">
    <h1>__LABEL__</h1>
    <span class="tag">__MODELE__</span>
    <span class="etat" id="etat">—</span>
  </div>

  <div class="intro">
    <p class="lede">__LEDE__</p>
    <p>__INTRO__ __SUITE__</p>
    <div class="avis" id="avis" hidden><span aria-hidden="true">▲</span><span id="avis-txt"></span></div>
  </div>

  <h2 class="sect">1 · Choisir</h2>
  <p class="sect">__CONSIGNE__</p>
  <div class="outils">
    <div class="bascule" id="vue-globale">
      <button type="button" data-vue="brut" aria-pressed="true">Relief seul</button>
      <button type="button" data-vue="annote" aria-pressed="false">Vérité terrain</button>
    </div>
  </div>
  <div class="grille" id="grille"></div>

  <h2 class="sect" id="titre-cadrer">2 · Cadrer</h2>
  <p class="sect" id="aide-cadrer">Glisse la fenêtre, tire le coin jaune pour la redimensionner.
    Au clavier : <kbd>←↑→↓</kbd> déplacent, <kbd>+</kbd> <kbd>−</kbd> redimensionnent
    (<kbd>Maj</kbd> = pas de 10).</p>
  <div class="cadres" id="cadres"></div>

  <footer id="pied"></footer>
</div>

<script>
const DATA = __DATA__;

const DEFAUT = {x: 0.28, y: 0.28, cote: 0.44};
const MIN_COTE = 0.12;

let db = null;
const retenues = [];               // n, dans l'ordre des clics
const cadrages = {};               // n -> {x, y, cote}
const timers = {};
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const norm = c => {
  const cote = clamp(c.cote, MIN_COTE, 1);
  return {cote, x: clamp(c.x, 0, 1 - cote), y: clamp(c.y, 0, 1 - cote)};
};
const cand = n => DATA.candidats.find(c => c.n === n);

// ---------- choix ----------
function carte(c) {
  const b = document.createElement("button");
  b.type = "button"; b.className = "carte";
  b.setAttribute("aria-pressed", "false"); b.dataset.n = String(c.n);
  b.innerHTML = `
    <div class="vue">
      <img data-vue="brut" src="data:image/jpeg;base64,${c.brut}" alt="Relief LiDAR — ${c.dalle}">
      ${c.annote ? `<img data-vue="annote" class="off" src="data:image/jpeg;base64,${c.annote}" alt="Vérité terrain">` : ""}
      <span class="rang">${c.n}</span><span class="marque" aria-hidden="true">✓</span>
    </div>
    <div class="pied">
      <span class="lieu"></span><span class="chiffres"></span><span class="pourquoi"></span>
    </div>`;
  b.querySelector(".lieu").textContent = c.zone;
  b.querySelector(".chiffres").textContent = `${c.objets} ${DATA.objets_label} · ${c.dalle}`;
  b.querySelector(".pourquoi").textContent = c.pourquoi;
  b.addEventListener("click", () => basculer(c.n, b));
  return b;
}

function basculer(n, bouton) {
  const i = retenues.indexOf(n);
  if (i >= 0) { retenues.splice(i, 1); delete cadrages[n]; }
  else { retenues.push(n); cadrages[n] = {...DEFAUT}; }
  bouton.setAttribute("aria-pressed", String(retenues.includes(n)));
  majMarques(); rendreCadres(); rafraichir(); planifier();
}

function majMarques() {
  document.querySelectorAll(".carte").forEach(b => {
    const n = Number(b.dataset.n);
    const rang = retenues.indexOf(n);
    b.querySelector(".marque").textContent = rang < 0 ? "✓" : (rang === 0 && !DATA.complement ? "1 · icône" : String(rang + 1));
  });
}

// ---------- cadrage ----------
function rendreCadres() {
  const hote = document.getElementById("cadres");
  if (DATA.complement) { hote.hidden = true; return; }
  hote.innerHTML = "";
  if (!retenues.length) {
    hote.innerHTML = `<div class="vide">Retiens d'abord un cadre ci-dessus.</div>`;
    return;
  }
  retenues.forEach((n, rang) => hote.appendChild(cadreur(cand(n), rang)));
}

function cadreur(c, rang) {
  const sec = document.createElement("section");
  sec.className = "cadreur";
  const gauche = document.createElement("div");
  gauche.innerHTML = `
    <h3>Cadre ${c.n} <span>${rang === 0 ? "— icône de la carte" : "— feuilleté dans la fiche"}</span></h3>
    <div class="scene" tabindex="0" role="application" aria-label="Cadrage du cadre ${c.n}">
      <img data-vue="brut" src="data:image/jpeg;base64,${c.brut}" alt="Relief LiDAR">
      ${c.annote ? `<img data-vue="annote" class="off" src="data:image/jpeg;base64,${c.annote}" alt="Vérité terrain">` : ""}
      <div class="voile"></div>
      <div class="trou"><img src="data:image/jpeg;base64,${c.brut}" alt=""></div>
      <div class="fen"></div>
    </div>
    <div class="outils" style="margin:9px 0 0">
      <div class="bascule">
        <button type="button" data-vue="brut" aria-pressed="true">Relief</button>
        ${c.annote ? `<button type="button" data-vue="annote" aria-pressed="false">Vérité terrain</button>` : ""}
      </div>
      <button type="button" class="btn" data-act="centre">Recentrer</button>
      <span class="coords"></span>
    </div>`;
  const droite = document.createElement("div");
  droite.className = "apercus";
  droite.innerHTML = `
    <div class="rangee">
      <div class="vue2"><div class="boite" style="width:44px;height:44px"><canvas width="44" height="44"></canvas></div><span class="lab">44 px — taille réelle</span></div>
      <div class="vue2"><div class="boite" style="width:110px;height:110px"><canvas width="110" height="110"></canvas></div><span class="lab">agrandi ×2,5</span></div>
    </div>
    <div class="carte-demo">
      <div class="boite" style="width:44px;height:44px"><canvas width="44" height="44"></canvas></div>
      <div><div class="titre">${DATA.label}</div><div class="desc">aperçu de la carte d'entité, étape 3</div></div>
    </div>
    <p class="aide">${c.pourquoi}</p>`;
  sec.appendChild(gauche); sec.appendChild(droite);
  queueMicrotask(() => cabler(c, gauche, droite));
  return sec;
}

function cabler(c, gauche, droite) {
  const scene = gauche.querySelector(".scene");
  const fen = scene.querySelector(".fen");
  const trou = scene.querySelector(".trou");
  const trouImg = trou.querySelector("img");
  const coords = gauche.querySelector(".coords");
  const canvas = [...droite.querySelectorAll("canvas")];
  const sources = {};
  let vueCourante = "brut";

  ["brut", "annote"].forEach(v => {
    if (!c[v]) return;
    const im = new Image();
    im.onload = () => { sources[v] = im; peindre(); };
    im.src = "data:image/jpeg;base64," + c[v];
  });

  function peindre() {
    const cd = cadrages[c.n]; if (!cd) return;
    const pc = n => (n * 100).toFixed(1) + "%";
    fen.style.left = pc(cd.x); fen.style.top = pc(cd.y);
    fen.style.width = pc(cd.cote); fen.style.height = pc(cd.cote);
    trou.style.left = pc(cd.x); trou.style.top = pc(cd.y);
    trou.style.width = pc(cd.cote); trou.style.height = pc(cd.cote);
    trouImg.style.width = (100 / cd.cote) + "%";
    trouImg.style.height = (100 / cd.cote) + "%";
    trouImg.style.left = (-cd.x / cd.cote * 100) + "%";
    trouImg.style.top = (-cd.y / cd.cote * 100) + "%";
    const src = sources[vueCourante] || sources.brut;
    if (src) trouImg.src = src.src;
    coords.textContent = `x ${cd.x.toFixed(2)} · y ${cd.y.toFixed(2)} · côté ${cd.cote.toFixed(2)} (${Math.round(cd.cote * (c.emprise_m || 324))} m)`;
    if (src) {
      const s = src.naturalWidth;
      canvas.forEach(cv => {
        const ctx = cv.getContext("2d");
        ctx.imageSmoothingQuality = "high";
        ctx.clearRect(0, 0, cv.width, cv.height);
        ctx.drawImage(src, cd.x * s, cd.y * s, cd.cote * s, cd.cote * s, 0, 0, cv.width, cv.height);
      });
    }
  }

  function bouge(dx, dy) { cadrages[c.n] = norm({...cadrages[c.n], x: cadrages[c.n].x + dx, y: cadrages[c.n].y + dy}); peindre(); planifier(); }
  function zoome(d) {
    const cd = cadrages[c.n];
    const cote = clamp(cd.cote + d, MIN_COTE, 1);
    cadrages[c.n] = norm({cote, x: cd.x + (cd.cote - cote) / 2, y: cd.y + (cd.cote - cote) / 2});
    peindre(); planifier();
  }

  let mode = null, depart = null;
  fen.addEventListener("pointerdown", ev => {
    const r = fen.getBoundingClientRect();
    mode = (ev.clientX > r.right - 16 && ev.clientY > r.bottom - 16) ? "taille" : "deplace";
    depart = {mx: ev.clientX, my: ev.clientY, ...cadrages[c.n], boite: scene.getBoundingClientRect()};
    fen.setPointerCapture(ev.pointerId); ev.preventDefault();
  });
  fen.addEventListener("pointermove", ev => {
    if (!mode) return;
    const dx = (ev.clientX - depart.mx) / depart.boite.width;
    const dy = (ev.clientY - depart.my) / depart.boite.height;
    cadrages[c.n] = mode === "deplace"
      ? norm({...depart, x: depart.x + dx, y: depart.y + dy})
      : norm({...depart, cote: depart.cote + Math.max(dx, dy)});
    peindre();
  });
  fen.addEventListener("pointerup", ev => { if (mode) { mode = null; planifier(); fen.releasePointerCapture(ev.pointerId); } });

  scene.addEventListener("keydown", ev => {
    const pas = ev.shiftKey ? 0.10 : 0.01;
    const a = {ArrowLeft: () => bouge(-pas, 0), ArrowRight: () => bouge(pas, 0),
               ArrowUp: () => bouge(0, -pas), ArrowDown: () => bouge(0, pas),
               "+": () => zoome(pas), "=": () => zoome(pas), "-": () => zoome(-pas)};
    if (a[ev.key]) { a[ev.key](); ev.preventDefault(); }
  });

  gauche.querySelectorAll(".bascule button").forEach(b => {
    b.addEventListener("click", () => {
      vueCourante = b.dataset.vue;
      gauche.querySelectorAll(".bascule button").forEach(o =>
        o.setAttribute("aria-pressed", String(o.dataset.vue === vueCourante)));
      scene.querySelectorAll("img[data-vue]").forEach(im =>
        im.classList.toggle("off", im.dataset.vue !== vueCourante));
      peindre();
    });
  });
  gauche.querySelector('[data-act="centre"]').addEventListener("click", () => {
    cadrages[c.n] = {...DEFAUT}; peindre(); planifier();
  });
  peindre();
}

// ---------- persistance ----------
function etatCourant() {
  return {
    classe: DATA.classe, modele: DATA.modele,
    retenues: [...retenues],
    cadrages: Object.fromEntries(retenues.map(n => [n, {
      x: +cadrages[n].x.toFixed(3), y: +cadrages[n].y.toFixed(3), cote: +cadrages[n].cote.toFixed(3),
    }])),
  };
}

function local(doc) {
  try {
    if (doc) localStorage.setItem("choix-" + DATA.classe, JSON.stringify(doc));
    else return JSON.parse(localStorage.getItem("choix-" + DATA.classe) || "null");
  } catch (_) { /* stockage bloqué */ }
  return null;
}

function planifier() { clearTimeout(timers.t); timers.t = setTimeout(enregistrer, 350); }

async function enregistrer() {
  const doc = etatCourant();
  local(doc);
  if (!db) return;
  try { await db.doc("choix/" + DATA.classe).set(doc); }
  catch (e) { avis("Choix gardé dans ce navigateur seulement (" + (e && e.code ? e.code : "erreur") + ")."); }
}

function rafraichir() {
  const el = document.getElementById("etat");
  el.innerHTML = retenues.length
    ? `<b>${retenues.length} cadre${retenues.length > 1 ? "s" : ""} retenu${retenues.length > 1 ? "s" : ""}</b>`
    : "aucun cadre retenu";
}

function avis(t) {
  document.getElementById("avis-txt").textContent = t;
  document.getElementById("avis").hidden = false;
}

function appliquer(doc) {
  if (!doc || !Array.isArray(doc.retenues)) return;
  retenues.length = 0;
  doc.retenues.forEach(n => { if (cand(n)) { retenues.push(n); cadrages[n] = norm((doc.cadrages || {})[n] || DEFAUT); } });
  document.querySelectorAll(".carte").forEach(b =>
    b.setAttribute("aria-pressed", String(retenues.includes(Number(b.dataset.n)))));
  majMarques(); rendreCadres(); rafraichir();
}

// ---------- démarrage ----------
const grille = document.getElementById("grille");
DATA.candidats.forEach(c => grille.appendChild(carte(c)));
document.querySelectorAll("#vue-globale button").forEach(b => {
  b.addEventListener("click", () => {
    const annote = b.dataset.vue === "annote";
    document.querySelectorAll("#vue-globale button").forEach(o =>
      o.setAttribute("aria-pressed", String((o.dataset.vue === "annote") === annote)));
    grille.querySelectorAll("img[data-vue]").forEach(im =>
      im.classList.toggle("off", (im.dataset.vue === "annote") !== annote));
  });
});
if (DATA.complement) {
  document.getElementById("titre-cadrer").hidden = true;
  document.getElementById("aide-cadrer").hidden = true;
}
appliquer(local());
rendreCadres(); rafraichir();
document.getElementById("pied").textContent = DATA.pied;

(async () => {
  db = await (window.claude?.use?.("db") ?? Promise.resolve(null));
  if (!db) { avis("Enregistrement partagé indisponible : tes choix restent dans ce navigateur."); return; }
  db.doc("choix/" + DATA.classe).onSnapshot(
    snap => { if (snap.exists) { appliquer(snap.data()); local(snap.data()); } },
    () => avis("Lecture interrompue — recharge la page si besoin."));
})();
</script>
"""


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _premier(d: dict, *cles, defaut=""):
    for k in cles:
        if d.get(k) not in (None, ""):
            return d[k]
    return defaut


def construire(dossier: str, classe: str, label: str, modele: str, corpus: str, objets_label: str,
               complement: bool = False) -> str:
    cands = json.load(open(os.path.join(dossier, "candidats.json"), encoding="utf-8"))
    liste = []
    for c in cands:
        n = int(_premier(c, "candidat", "n"))
        brut = os.path.join(dossier, f"{n}_brut.jpg")
        if not os.path.isfile(brut):
            print(f"  ! {brut} absent, candidat {n} ignoré", file=sys.stderr)
            continue
        annote = os.path.join(dossier, f"{n}_annote.jpg")
        tuiles = c.get("tuiles") or []
        dalle = _premier(c, "dalle", defaut=(
            f"tuile {tuiles[0]}" if tuiles else (f"cible {c['cible']}" if c.get("cible") else "")))
        emprise = c.get("emprise_m")
        if emprise:
            dalle = (dalle + " · " if dalle else "") + f"{int(round(emprise))} m de côté"
        liste.append({
            "n": n,
            "brut": _b64(brut),
            "annote": _b64(annote) if os.path.isfile(annote) else "",
            "zone": _premier(c, "zone", "secteur"),
            "dalle": dalle,
            "objets": _premier(c, "objets", "n_crateres", "n", defaut=""),
            "pourquoi": _premier(c, "pourquoi", "profil"),
            "emprise_m": emprise or 324,
        })
    data = {
        "classe": classe, "label": label, "modele": modele, "objets_label": objets_label, "complement": complement,
        "pied": (f"{len(liste)} cadres tirés de {corpus}. Les retenus seront copiés dans "
                 f"data/models/{modele}/vignettes/ et déclarés dans la fiche de {classe} du model_card.yaml."),
        "candidats": liste,
    }
    page = (GABARIT
            .replace("__TITRE__", f"Vignettes · {label}")
            .replace("__LABEL__", label)
            .replace("__MODELE__", modele)
            .replace("__LEDE__", (f"La fiche « {label} » du modèle {modele} n'a qu'une image, son icône. "
                                  "Choisis un ou deux cadres de plus, qui se feuilletteront après elle dans la fiche.")
                     if complement else
                     (f"Le modèle {modele} est installé. Il manque l'illustration de « {label} » : "
                      "choisis deux ou trois cadres, puis pose la fenêtre qui servira d'icône sur la carte d'entité."))
            .replace("__INTRO__", f"Les {len(liste)} candidats viennent de {corpus}, avec la vérité terrain dessinée.")
            .replace("__SUITE__", "L'icône actuelle est conservée ; l'ordre des clics fait l'ordre des images suivantes."
                     if complement else "Le premier cadre retenu devient l'icône ; les suivants se feuillettent dans la fiche.")
            .replace("__CONSIGNE__", "Clique un cadre pour le retenir. Un seul suffit, deux si le second montre autre chose."
                     if complement else "Clique un cadre pour le retenir. L'ordre des clics fixe l'ordre dans la fiche.")
            .replace("__DATA__", json.dumps(data, ensure_ascii=False)))
    return page


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dossier", help="dossier des candidats (N_brut.jpg, N_annote.jpg, candidats.json)")
    ap.add_argument("--classe", required=True, help="name de la classe ou output_class de la cible dérivée")
    ap.add_argument("--label", required=True, help="libellé affiché (label_fr)")
    ap.add_argument("--modele", required=True, help="id du modèle installé")
    ap.add_argument("--corpus", default="du corpus d'entraînement", help="d'où viennent les cadres (texte)")
    ap.add_argument("--objets", default="objets annotés", help="libellé des objets comptés par cadre")
    ap.add_argument("--complement", action="store_true",
                    help="l'icône existe déjà : les cadres retenus s'ajoutent à la suite, sans cadrage")
    ap.add_argument("--sortie", default="page_choix.html")
    a = ap.parse_args()
    page = construire(a.dossier, a.classe, a.label, a.modele, a.corpus, a.objets, a.complement)
    with open(a.sortie, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"page : {a.sortie} ({os.path.getsize(a.sortie) / 1e6:.1f} Mo) — publier avec Artifact, capabilities {{db: {{}}}} ; "
          f"relire le choix : read_db collection 'choix', doc '{a.classe}'")


if __name__ == "__main__":
    main()
