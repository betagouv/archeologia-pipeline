#!/usr/bin/env python3
"""Génère un fichier `htpasswd.txt` pour protéger le dépôt QGIS hébergé sur OVH.

Pourquoi : Apache (auth Basic) ne stocke jamais le mot de passe en clair, il compare
un *hachage*. Ce script produit la ligne `identifiant:hash` et l'écrit dans
`htpasswd.txt` ; à uploader dans `www/qgis/` puis renommer en `.htpasswd` sur le serveur
(à côté du `.htaccess`).

Usage :
    python dev/make_htpasswd.py [dossier_de_sortie]

`dossier_de_sortie` par défaut = répertoire courant. Le mot de passe est saisi masqué
(getpass), jamais affiché ni journalisé. Le fichier est écrit en ASCII + fins de ligne
LF (pas de BOM ni de CRLF, qui corrompraient le hachage ou feraient échouer Apache).

Format produit : `{SHA}` (SHA-1 Base64), accepté par Apache pour l'auth Basic. Suffisant
ici : le `.htpasswd` n'est pas servi publiquement (Apache bloque les fichiers `.ht*`) et
protège un identifiant partagé à faible enjeu. Si l'outil Apache `htpasswd` est dispo,
`htpasswd -nbB <user> <pass>` (bcrypt) est un cran au-dessus et tout aussi accepté.
"""

import base64
import getpass
import hashlib
import sys
from pathlib import Path


def make_line(user: str, password: str) -> str:
    """Retourne la ligne `.htpasswd` au format `user:{SHA}base64(sha1(password))`."""
    digest = hashlib.sha1(password.encode("utf-8")).digest()
    return f"{user}:{{SHA}}" + base64.b64encode(digest).decode("ascii")


if __name__ == "__main__":
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()

    user = input("Identifiant (celui à saisir dans QGIS) : ").strip()
    if not user:
        raise SystemExit("Identifiant vide — abandon.")
    password = getpass.getpass("Mot de passe : ")
    if password != getpass.getpass("Confirme le mot de passe : "):
        raise SystemExit("Les mots de passe ne correspondent pas — abandon.")

    line = make_line(user, password)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "htpasswd.txt"
    out_file.write_bytes((line + "\n").encode("ascii"))

    print(f"\nFichier cree : {out_file.resolve()}")
    print("-> Upload-le dans www/qgis/, puis renomme-le en .htpasswd sur le serveur.")
