"""Lot 3 (accélération 2026-10) : une dalle déjà lue en entier n'est plus relue
(note « vérifiée » : taille + date), et un téléchargement s'écrit dans
<nom>.partial — un flux coupé ou un QGIS tué ne laisse jamais de LAZ tronqué
sous le nom final. Sans taille annoncée, lecture complète une fois."""
from __future__ import annotations

import os

import pytest

import pipeline.ign.downloader as dl

NOM = "LHD_FXX_0500_6500_PTS_C_LAMB93_IGN69.copc.laz"
URL = f"https://data.geopf.fr/telechargement/download/LiDARHD/x/{NOM}"


class _Reponse:
    def __init__(self, morceaux, headers, coupe_avant=None):
        self._morceaux, self.headers, self._coupe = morceaux, headers, coupe_avant

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=8192):
        for i, m in enumerate(self._morceaux):
            if self._coupe is not None and i == self._coupe:
                raise ConnectionError("connexion coupée")
            yield m


@pytest.fixture
def reseau(monkeypatch):
    etat = {"reponse": None, "profondes": 0}

    class _Requests:
        @staticmethod
        def get(*_a, **_k):
            return etat["reponse"]

    def profonde(_p, *a, **k):
        etat["profondes"] += 1
        return True, "ok"

    monkeypatch.setattr(dl, "HAS_REQUESTS", True)
    monkeypatch.setattr(dl, "requests", _Requests, raising=False)
    monkeypatch.setattr(dl, "_is_valid_with_pdal", lambda _p: (True, ""))
    monkeypatch.setattr(dl, "validate_laz_deep", profonde)
    return etat


def _telecharger(tmp_path, logs=None):
    return dl.download_one(URL, NOM, tmp_path, log=(logs.append if logs is not None else (lambda _m: None)),
                           max_retries=1, retry_delay_s=0)


def test_flux_coupe_a_mi_chemin_ne_laisse_qu_un_partial(tmp_path, reseau):
    reseau["reponse"] = _Reponse([b"a" * 10, b"b" * 10], {"Content-Length": "20"}, coupe_avant=1)
    assert _telecharger(tmp_path) == (False, False)
    assert not (tmp_path / NOM).exists()
    assert (tmp_path / (NOM + ".partial")).exists()


def test_flux_court_sans_erreur_refuse(tmp_path, reseau):
    reseau["reponse"] = _Reponse([b"x" * 20], {"Content-Length": "30"})
    assert _telecharger(tmp_path) == (False, False)
    assert not (tmp_path / NOM).exists()


def test_partial_orphelin_supprime(tmp_path, reseau):
    (tmp_path / (NOM + ".partial")).write_bytes(b"reste d'un run tue")
    reseau["reponse"] = _Reponse([b"x" * 20], {"Content-Length": "20"})
    assert _telecharger(tmp_path) == (True, False)
    assert not (tmp_path / (NOM + ".partial")).exists()
    assert (tmp_path / NOM).read_bytes() == b"x" * 20


def test_taille_annoncee_conforme_pas_de_lecture_complete(tmp_path, reseau):
    reseau["reponse"] = _Reponse([b"x" * 20], {"Content-Length": "20"})
    _telecharger(tmp_path)
    assert reseau["profondes"] == 0
    assert dl.note_verifiee_valide(tmp_path / NOM)


def test_sans_taille_annoncee_lecture_complete_une_fois(tmp_path, reseau):
    reseau["reponse"] = _Reponse([b"x" * 20], {})
    assert _telecharger(tmp_path) == (True, False)
    assert reseau["profondes"] == 1
    assert _telecharger(tmp_path) == (True, True)   # relance : la note suffit
    assert reseau["profondes"] == 1


def test_corps_encode_traite_comme_sans_taille(tmp_path, reseau):
    # gzip : la taille annoncée n'est pas celle écrite → pas de contrôle de taille, lecture complète.
    reseau["reponse"] = _Reponse([b"x" * 20], {"Content-Length": "5", "Content-Encoding": "gzip"})
    assert _telecharger(tmp_path) == (True, False)
    assert reseau["profondes"] == 1


def test_relance_saute_la_validation_profonde_si_la_note_correspond(tmp_path, reseau):
    dest = tmp_path / NOM
    dest.write_bytes(b"x" * 20)
    dl.ecrire_note_verifiee(dest)
    logs: list = []
    assert _telecharger(tmp_path, logs) == (True, True)
    assert reseau["profondes"] == 0
    assert any("déjà présente (vérifiée)" in m for m in logs)


def test_sans_note_lecture_complete_puis_note(tmp_path, reseau):
    dest = tmp_path / NOM
    dest.write_bytes(b"x" * 20)
    assert _telecharger(tmp_path) == (True, True)
    assert reseau["profondes"] == 1 and dl.note_verifiee_valide(dest)


def test_taille_ou_date_changees_invalident_la_note(tmp_path):
    dest = tmp_path / NOM
    dest.write_bytes(b"x" * 20)
    dl.ecrire_note_verifiee(dest)
    with open(dest, "ab") as f:
        f.write(b"y")
    assert not dl.note_verifiee_valide(dest)
    dl.ecrire_note_verifiee(dest)
    st = dest.stat()
    os.utime(dest, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
    assert not dl.note_verifiee_valide(dest)


def test_ecriture_de_la_note_impossible_n_echoue_pas(tmp_path, reseau, monkeypatch):
    # La note n'est qu'un accélérateur : disque plein ou droit refusé ne doit ni
    # faire échouer ni refaire télécharger une dalle valide.
    def refuse(*_a, **_k):
        raise OSError("disque plein")

    monkeypatch.setattr(dl.Path, "write_text", refuse)
    reseau["reponse"] = _Reponse([b"x" * 20], {"Content-Length": "20"})
    assert _telecharger(tmp_path) == (True, False)
    assert (tmp_path / NOM).read_bytes() == b"x" * 20
    assert not dl.note_verifiee_valide(tmp_path / NOM)
