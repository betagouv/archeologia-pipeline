"""NET-01/NET-02 (audit v1, persistants) — sécurité du téléchargeur IGN.

NET-02 : ``_extract_real_url`` déballe des liens venus de MAILS
(Proofpoint/SafeLinks) et acceptait N'IMPORTE QUEL hôte → une URL piégée
déclenchait une requête sortante arbitraire (SSRF sur réseau d'entreprise),
http en clair toléré. → liste blanche de domaines IGN + https imposé.

NET-01 : téléchargement sans borne de taille (ni Content-Length ni cap en
streaming), validation PDAL seulement après écriture complète → saturation
disque possible (×4 workers). → cap déclaré ET cap streaming, sans retry.
"""
from __future__ import annotations

import pytest

import pipeline.ign.downloader as dl


class TestValidateDownloadUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "https://data.geopf.fr/telechargement/dalle.copc.laz",
            "https://wxs.ign.fr/lidar/dalle.laz",
            "https://storage.sbg.cloud.ovh.net/v1/AUTH_x/lidar/dalle.laz",
        ],
    )
    def test_hotes_ign_acceptes(self, url):
        ok, normalized, _ = dl.validate_download_url(url)
        assert ok is True
        assert normalized == url

    @pytest.mark.parametrize(
        "url",
        [
            "https://evil.com/x.laz",
            "https://fakeign.fr/x.laz",          # suffixe partiel ≠ sous-domaine
            "https://ign.fr.evil.com/x.laz",     # préfixe usurpé
            "ftp://data.geopf.fr/x.laz",
        ],
    )
    def test_hotes_inconnus_refuses(self, url):
        ok, _, why = dl.validate_download_url(url)
        assert ok is False
        assert why

    def test_http_clair_upgrade_en_https(self):
        ok, normalized, _ = dl.validate_download_url("http://data.geopf.fr/d.laz")
        assert ok is True
        assert normalized.startswith("https://")

    def test_chaine_ssrf_depuis_un_mail_refusee(self):
        # URL piégée enveloppée Proofpoint : le déballage donne evil.com →
        # la validation doit couper AVANT toute requête.
        wrapped = "https://urldefense.com/v3/__https://evil.com/payload.laz__;!!x"
        ok, _, _ = dl.validate_download_url(dl._extract_real_url(wrapped))
        assert ok is False


class _FakeUrlResponse:
    """Réponse urllib factice : flux « infini » (le cap doit couper)."""

    headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, n):
        return b"0" * n


class _FakeOpener:
    def __init__(self, response):
        self._response = response

    def open(self, req, timeout=None):
        return self._response


class TestDownloadOneGuards:
    """Branche urllib forcée (HAS_REQUESTS=False) : déterministe sur tous les
    postes, requests étant absent du venv de test."""

    def test_url_hors_liste_blanche_refusee_sans_requete(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dl, "HAS_REQUESTS", False)

        def _no_network(*a, **k):
            raise AssertionError("requête émise vers un hôte non autorisé")

        monkeypatch.setattr(dl.urllib.request, "build_opener", _no_network)
        logs: list[str] = []
        ok, skipped = dl.download_one(
            "https://evil.com/x.laz", "x.laz", tmp_path,
            log=logs.append, cancel=lambda: False,
        )
        assert (ok, skipped) == (False, False)
        assert not (tmp_path / "x.laz").exists()
        assert any("refusée" in m for m in logs)

    def test_content_length_au_dela_du_cap_refuse(self, tmp_path, monkeypatch):
        resp = _FakeUrlResponse()
        resp.headers = {"Content-Length": str(10**12)}  # 1 To annoncé
        monkeypatch.setattr(dl, "HAS_REQUESTS", False)
        monkeypatch.setattr(
            dl.urllib.request, "build_opener", lambda *a, **k: _FakeOpener(resp)
        )
        logs: list[str] = []
        ok, _ = dl.download_one(
            "https://data.geopf.fr/d.laz", "d.laz", tmp_path,
            log=logs.append, cancel=lambda: False, max_retries=1,
        )
        assert ok is False
        assert not (tmp_path / "d.laz").exists()

    def test_cap_streaming_interrompt_et_nettoie(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dl, "HAS_REQUESTS", False)
        monkeypatch.setattr(
            dl.urllib.request,
            "build_opener",
            lambda *a, **k: _FakeOpener(_FakeUrlResponse()),
        )
        monkeypatch.setattr(dl, "MAX_DOWNLOAD_SIZE_MB", 2)  # cap à 2 Mo
        logs: list[str] = []
        ok, _ = dl.download_one(
            "https://data.geopf.fr/d.laz", "d.laz", tmp_path,
            log=logs.append, cancel=lambda: False, max_retries=3,
        )
        assert ok is False
        assert not (tmp_path / "d.laz").exists()  # partiel supprimé
        # Cap = définitif : PAS de retry (sinon 3 × 2 Go gaspillés).
        assert sum("cap" in m.lower() for m in logs) == 1
