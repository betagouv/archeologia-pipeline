"""Téléchargement bloqué (2026-10-19, D:/pipeline_results/test) : le serveur de l'IGN n'envoyait
plus qu'un filet d'octets (≈ 16 o/s) — chaque octet réarmait le délai de lecture de 300 s et
``iter_content(8192)`` ne rendait rien : fichier figé, connexion ouverte, aucune reprise pendant
8 minutes. Audit Fable du 19/10 : un chien de garde sur la croissance du fichier qui coupe le
socket (``shutdown``, jamais ``resp.close()`` qui bloque), un délai de lecture égal à sa fenêtre,
et la reprise à l'octet manquant par ``Range`` (le serveur rend 206)."""
from __future__ import annotations

import socket
import threading
import time

import pytest

import pipeline.ign.downloader as dl

NOM = "LHD_FXX_0500_6500_PTS_C_LAMB93_IGN69.copc.laz"
URL = f"https://data.geopf.fr/telechargement/download/LiDARHD/x/{NOM}"


class _Sock:
    """Socket factice : ``shutdown`` pose l'événement que le filet attend."""

    def __init__(self):
        self.coupe = threading.Event()
        self.appels = 0

    def shutdown(self, _how):
        self.appels += 1
        self.coupe.set()


class _Reponse:
    """Réponse ``requests`` doublée. ``morceaux`` : octets rendus ; ``"filet"`` : plus aucun
    chunk complet n'arrive — le générateur attend la coupure du socket puis lève, comme
    ``iter_content`` après ``sock.shutdown`` (ConnectionAbortedError 10053)."""

    def __init__(self, morceaux, headers, status_code=200, avec_socket=True):
        self._morceaux, self.headers, self.status_code = morceaux, headers, status_code
        self.sock = _Sock() if avec_socket else None
        self.raw = type("Raw", (), {})()
        self.raw.connection = type("Conn", (), {"sock": self.sock})() if avec_socket else None

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=8192):
        for m in self._morceaux:
            if m == "filet":
                if not self.sock.coupe.wait(5.0):
                    raise AssertionError("le chien de garde n'a pas coupé le socket")
                raise ConnectionError("ConnectionAbortedError(10053)")
            yield m


@pytest.fixture
def reseau(monkeypatch):
    etat = {"reponses": [], "appels": []}

    class _Requests:
        @staticmethod
        def get(url, headers=None, **_k):
            etat["appels"].append(dict(headers or {}))
            return etat["reponses"].pop(0)

    monkeypatch.setattr(dl, "HAS_REQUESTS", True)
    monkeypatch.setattr(dl, "requests", _Requests, raising=False)
    monkeypatch.setattr(dl, "_is_valid_with_pdal", lambda _p: (True, ""))
    monkeypatch.setattr(dl, "validate_laz_deep", lambda _p, *a, **k: (True, "ok"))
    return etat


def _telecharger(tmp_path, logs=None, cancel=None, retries=3, blocage_s=0.2):
    return dl.download_one(URL, NOM, tmp_path, log=(logs.append if logs is not None else (lambda _m: None)),
                           cancel=cancel or (lambda: False), max_retries=retries, retry_delay_s=0,
                           blocage_s=blocage_s)


def test_filet_coupe_puis_reprise_a_l_octet_manquant(tmp_path, reseau):
    logs: list = []
    premiere = _Reponse([b"a" * 10, "filet"], {"Content-Length": "20", "ETag": '"e1"'})
    reseau["reponses"] = [premiere,
                          _Reponse([b"b" * 10], {"Content-Length": "10", "Content-Range": "bytes 10-19/20"}, 206)]
    assert _telecharger(tmp_path, logs) == (True, False)
    assert premiere.sock.appels == 1
    assert (tmp_path / NOM).read_bytes() == b"a" * 10 + b"b" * 10
    assert reseau["appels"][1]["Range"] == "bytes=10-" and reseau["appels"][1]["If-Range"] == '"e1"'
    assert any("aucun octet reçu depuis" in m for m in logs)
    assert any("Reprise" in m and "octet 10" in m for m in logs)


def test_serveur_sans_reprise_repart_de_zero(tmp_path, reseau):
    reseau["reponses"] = [_Reponse([b"a" * 10, "filet"], {"Content-Length": "20"}),
                          _Reponse([b"c" * 20], {"Content-Length": "20"}, 200)]   # 200 : Range ignoré
    assert _telecharger(tmp_path) == (True, False)
    assert (tmp_path / NOM).read_bytes() == b"c" * 20


def test_content_range_incoherent_repart_de_zero(tmp_path, reseau):
    reseau["reponses"] = [_Reponse([b"a" * 10, "filet"], {"Content-Length": "20"}),
                          _Reponse([b"d" * 20], {"Content-Length": "20", "Content-Range": "bytes 10-29/30"}, 206)]
    assert _telecharger(tmp_path) == (True, False)
    assert (tmp_path / NOM).read_bytes() == b"d" * 20


def test_annulation_pendant_un_blocage_rend_la_main_vite(tmp_path, reseau):
    drapeau = threading.Event()
    reseau["reponses"] = [_Reponse([b"a" * 10, "filet"], {"Content-Length": "20"})]
    threading.Timer(0.3, drapeau.set).start()
    t0 = time.monotonic()
    assert _telecharger(tmp_path, cancel=drapeau.is_set, blocage_s=30) == (False, False)
    assert time.monotonic() - t0 < 3
    assert not (tmp_path / (NOM + ".partial")).exists() and not (tmp_path / NOM).exists()


def test_sans_socket_pas_de_garde_et_comportement_inchange(tmp_path, reseau):
    reseau["reponses"] = [_Reponse([b"x" * 20], {"Content-Length": "20"}, avec_socket=False)]
    assert _telecharger(tmp_path) == (True, False)
    assert (tmp_path / NOM).read_bytes() == b"x" * 20


def test_fichier_complet_invalide_repart_de_zero_sans_range(tmp_path, reseau, monkeypatch):
    verdicts = iter([(False, "corrompu"), (True, "")])
    monkeypatch.setattr(dl, "_is_valid_with_pdal", lambda _p: next(verdicts))
    reseau["reponses"] = [_Reponse([b"a" * 20], {"Content-Length": "20"}),
                          _Reponse([b"b" * 20], {"Content-Length": "20"})]
    assert _telecharger(tmp_path) == (True, False)
    assert "Range" not in reseau["appels"][1]
    assert (tmp_path / NOM).read_bytes() == b"b" * 20


def test_echec_definitif_apres_les_tentatives(tmp_path, reseau):
    reseau["reponses"] = [_Reponse([b"a" * 10, "filet"], {"Content-Length": "20"}),
                          _Reponse(["filet"], {"Content-Length": "10", "Content-Range": "bytes 10-19/20"}, 206)]
    logs: list = []
    assert _telecharger(tmp_path, logs, retries=2) == (False, False)
    assert any("Échec définitif" in m for m in logs)


def test_le_journal_renvoie_au_depannage():
    from app.services.aide import rubrique_depannage
    assert rubrique_depannage("⚠️ LHD_… : aucun octet reçu depuis 120 s, connexion coupée") == "Le téléchargement échoue"


# ----------------------------------------------------------------------
# Boucle locale : la vraie sémantique socket (shutdown + délai de lecture)
# ----------------------------------------------------------------------
class _ServeurFilet(threading.Thread):
    """Serveur HTTP minimal : envoie ``rapide`` octets d'un coup, puis un octet par seconde
    (filet) jusqu'à ce que le client coupe ; sur une requête ``Range``, sert le reste d'un coup."""

    def __init__(self, contenu: bytes, rapide: int):
        super().__init__(daemon=True)
        self.contenu, self.rapide = contenu, rapide
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(2)
        self.port = self.srv.getsockname()[1]
        self.requetes: list = []

    def run(self):
        for _ in range(3):
            conn, _ = self.srv.accept()
            requete = b""
            while b"\r\n\r\n" not in requete:
                requete += conn.recv(4096)
            self.requetes.append(requete.decode("latin-1"))
            debut = 0
            for ligne in requete.split(b"\r\n"):
                if ligne.lower().startswith(b"range: bytes="):
                    debut = int(ligne.split(b"=")[1].split(b"-")[0])
            if debut:
                corps = self.contenu[debut:]
                conn.sendall(f"HTTP/1.1 206 Partial Content\r\nContent-Length: {len(corps)}\r\n"
                             f"Content-Range: bytes {debut}-{len(self.contenu) - 1}/{len(self.contenu)}\r\n"
                             "Connection: close\r\n\r\n".encode() + corps)
                conn.close()
                return
            conn.sendall(f"HTTP/1.1 200 OK\r\nContent-Length: {len(self.contenu)}\r\nConnection: close\r\n\r\n".encode()
                         + self.contenu[:self.rapide])
            try:
                for i in range(self.rapide, len(self.contenu)):
                    time.sleep(1.0)
                    conn.sendall(self.contenu[i:i + 1])
            except OSError:
                pass
            conn.close()


@pytest.mark.skipif(not dl.HAS_REQUESTS, reason="requests absent")
def test_boucle_locale_filet_coupe_et_repris(tmp_path, monkeypatch):
    contenu = bytes(range(256)) * 400   # 102 400 octets
    serveur = _ServeurFilet(contenu, rapide=40_000)
    serveur.start()
    monkeypatch.setattr(dl, "validate_download_url", lambda u: (True, u, ""))
    monkeypatch.setattr(dl, "_is_valid_with_pdal", lambda _p: (True, ""))
    monkeypatch.setattr(dl, "validate_laz_deep", lambda _p, *a, **k: (True, "ok"))
    logs: list = []
    t0 = time.monotonic()
    ok = dl.download_one(f"http://127.0.0.1:{serveur.port}/{NOM}", NOM, tmp_path, log=logs.append,
                         timeout_s=2, max_retries=3, retry_delay_s=0, blocage_s=2)
    duree = time.monotonic() - t0
    assert ok == (True, False), logs
    assert (tmp_path / NOM).read_bytes() == contenu
    assert any("aucun octet reçu depuis" in m for m in logs)
    assert len(serveur.requetes) == 2 and "Range: bytes=" in serveur.requetes[1]
    assert duree < 15, duree
