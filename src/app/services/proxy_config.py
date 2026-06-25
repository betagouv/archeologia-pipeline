"""Helpers purs (sans QGIS) pour normaliser la configuration proxy.

Extrait de ``pipeline.ign.downloader`` pour rester testable en standalone :
``src/pipeline`` n'est pas collecté par pytest (il importe QGIS), alors que
``src/app`` l'est. La logique du bug (double ``http://``, détection PAC) vit
donc ici et le downloader s'y branche.
"""
from __future__ import annotations


def _strip_scheme(host: str) -> str:
    """Retire un schéma de tête (``http://``, ``https://``, ``socks5://``…)."""
    host = host.strip()
    if "://" in host:
        host = host.split("://", 1)[1]
    return host


def build_proxy_url(
    host: str, port: str = "", user: str = "", password: str = ""
) -> str:
    """Construit une URL proxy ``http://[user[:password]@]host[:port]``.

    Retire tout schéma déjà présent dans ``host`` AVANT de préfixer ``http://`` :
    sans cela, un host déjà schémé (ex. URL PAC ``http://h/x.pac``) produisait
    ``http://http://…`` et ``requests`` échouait sur ``Failed to resolve 'http'``.
    Une barre oblique finale (``hôte:port/`` — typo/copier-coller) est aussi
    retirée pour ne pas produire une URL de proxy malformée.
    """
    host = _strip_scheme(host).rstrip("/")
    if user and password:
        proxy_url = f"http://{user}:{password}@{host}"
    elif user:
        proxy_url = f"http://{user}@{host}"
    else:
        proxy_url = f"http://{host}"
    if port:
        proxy_url = f"{proxy_url}:{port}"
    return proxy_url


def is_pac_like(host: str) -> bool:
    """Vrai si ``host`` désigne une auto-configuration (PAC), pas un proxy direct.

    Un proxy direct est un simple ``hôte[:port]``. Une URL PAC/WPAD comporte un
    chemin (``/proxy.pac``, ``/wpad.dat``) — non utilisable tel quel par
    ``requests`` (le chemin est ignoré). On détecte donc tout ``.pac`` ou tout
    chemin non vide une fois le schéma retiré. Une simple barre oblique finale
    (``hôte:port/``) n'est PAS un PAC : sinon on écarterait un proxy valide.
    """
    rest = _strip_scheme(host)
    if not rest:
        return False
    if ".pac" in rest.lower():
        return True
    _head, sep, tail = rest.partition("/")
    return bool(sep) and tail != ""
