"""Helpers purs de configuration proxy (sans QGIS) — testables en standalone.

Régression du bug du double ``http://`` : un proxy QGIS renseigné avec une URL
PAC (``http://host/xxx.pac``) était re-préfixé par ``http://`` →
``http://http://host/xxx.pac`` → ``requests`` tentait de résoudre un hôte nommé
« http » (``NameResolutionError: Failed to resolve 'http'``) → tous les
téléchargements échouaient.

``build_proxy_url`` retire tout schéma déjà présent avant de préfixer ``http://``.
``is_pac_like`` détecte les URL d'auto-configuration (PAC), qui ne sont PAS des
proxys directs utilisables par ``requests``.
"""
from __future__ import annotations

from app.services.proxy_config import build_proxy_url, is_pac_like


class TestBuildProxyUrl:
    def test_plain_host(self):
        assert build_proxy_url("proxy.corp") == "http://proxy.corp"

    def test_host_and_port(self):
        assert build_proxy_url("proxy.corp", "3128") == "http://proxy.corp:3128"

    def test_host_user_password_port(self):
        assert (
            build_proxy_url("proxy.corp", "3128", "alice", "s3cret")
            == "http://alice:s3cret@proxy.corp:3128"
        )

    def test_host_user_only(self):
        assert build_proxy_url("proxy.corp", user="alice") == "http://alice@proxy.corp"

    def test_existing_http_scheme_is_not_doubled(self):
        # Le coeur du bug : un schéma déjà présent ne doit pas être re-préfixé.
        out = build_proxy_url("http://proxy.corp", "3128")
        assert out == "http://proxy.corp:3128"
        assert "http://http://" not in out

    def test_existing_https_scheme_is_normalised_to_http(self):
        # Le proxy QGIS est un proxy HTTP ; on normalise vers http://.
        assert build_proxy_url("https://proxy.corp") == "http://proxy.corp"

    def test_bug_case_pac_url_with_port_never_doubles_scheme(self):
        # Reproduction exacte du journal : host=http://topaze/proxy.pac, port=8000.
        out = build_proxy_url("http://topaze/proxy.pac", "8000")
        assert "http://http://" not in out

    def test_whitespace_is_stripped(self):
        assert build_proxy_url("  proxy.corp  ", "3128") == "http://proxy.corp:3128"

    def test_trailing_slash_is_stripped(self):
        # Barre oblique finale (typo/copier-coller) : ne doit pas casser l'URL.
        assert build_proxy_url("proxy.corp:3128/") == "http://proxy.corp:3128"
        assert build_proxy_url("http://proxy.corp/", "3128") == "http://proxy.corp:3128"


class TestIsPacLike:
    def test_bare_host_is_not_pac(self):
        assert is_pac_like("proxy.corp") is False

    def test_host_port_is_not_pac(self):
        assert is_pac_like("proxy.corp:3128") is False

    def test_scheme_without_path_is_not_pac(self):
        # Un proxy direct avec schéma reste utilisable (pas un PAC).
        assert is_pac_like("https://proxy.corp") is False

    def test_pac_url_is_detected(self):
        # Le cas du bug.
        assert is_pac_like("http://topaze/proxy.pac") is True

    def test_pac_host_without_scheme_is_detected(self):
        assert is_pac_like("topaze/proxy.pac") is True

    def test_wpad_path_is_detected(self):
        # Toute URL avec chemin n'est pas un proxy direct host:port.
        assert is_pac_like("http://wpad.example/wpad.dat") is True

    def test_empty_host_is_not_pac(self):
        assert is_pac_like("") is False

    def test_trailing_slash_only_is_not_pac(self):
        # Une barre finale seule n'est pas une auto-config : proxy direct valide.
        assert is_pac_like("proxy.corp:3128/") is False
        assert is_pac_like("http://proxy.corp:3128/") is False
