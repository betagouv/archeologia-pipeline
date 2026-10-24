"""Étape 1 : le message sous le champ dit ce qui manque et le geste (lot 2 de
l'interface rationalisée), au lieu d'une bordure rouge muette."""
from app.services.source_modes import message_champ


def test_champ_vide_dit_le_geste_par_point_de_depart():
    assert message_champ("ign_laz", "", "ok") == (
        "Indiquez la zone d'étude (.shp, .gpkg, .geojson), une liste de dalles (.txt), "
        "ou choisissez les dalles sur la carte."
    )
    assert message_champ("local_laz", "", "ok") == "Indiquez le dossier des fichiers LAZ ou LAS."
    assert message_champ("existing_mnt", "", "ok") == "Indiquez le dossier des MNT (TIF en Lambert-93)."
    assert message_champ("existing_rvt", "", "ok") == "Indiquez le dossier des images du relief (TIF en Lambert-93)."


def test_fichier_ou_dossier_introuvable():
    assert message_champ("ign_laz", "C:/nulle/part.gpkg", "error") == (
        "Ce fichier est introuvable : vérifiez le chemin, ou choisissez les dalles sur la carte."
    )
    assert message_champ("existing_mnt", "C:/nulle/part", "error") == "Ce dossier est introuvable : vérifiez le chemin."


def test_extension_inattendue():
    assert message_champ("ign_laz", "C:/x/zone.kml", "warn") == (
        "Extension inattendue : attendu .shp, .gpkg, .geojson ou une liste .txt."
    )


def test_dalles_choisies_sur_la_carte(tmp_path):
    liste = tmp_path / "dalles_selection.txt"
    liste.write_text("a,https://x/a\nb,https://x/b\n\nc,https://x/c\n", encoding="utf-8")
    assert message_champ("ign_laz", str(liste), "ok") == "3 dalles choisies sur la carte"
    liste.write_text("a,https://x/a\n", encoding="utf-8")
    assert message_champ("ign_laz", str(liste), "ok") == "1 dalle choisie sur la carte"


def test_chemin_valide_ne_dit_rien():
    assert message_champ("local_laz", "C:/data/laz", "ok") == ""


def test_dossier_de_sortie():
    assert message_champ("ign_laz", "", "ok", sortie=True) == (
        "Indiquez le dossier qui recevra livrable/ et technique/ ; il peut ne pas exister encore."
    )
    assert message_champ("ign_laz", "C:/x/fichier.txt", "error", sortie=True) == (
        "Ce chemin n'est pas un dossier : indiquez un dossier, existant ou à créer."
    )
    assert message_champ("ign_laz", "C:/sortie", "ok", sortie=True) == ""


def test_dossier_de_sortie_dans_la_source(tmp_path):
    source = tmp_path / "laz"
    source.mkdir()
    dedans = source / "sortie"
    assert message_champ("local_laz", str(dedans), "ok", sortie=True, source=str(source)) == (
        "Ce dossier est dans le dossier de vos données : choisissez un dossier de sortie à part."
    )
    assert message_champ("local_laz", str(tmp_path / "ailleurs"), "ok", sortie=True, source=str(source)) == ""
