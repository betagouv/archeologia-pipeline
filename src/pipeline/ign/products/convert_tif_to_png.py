import logging
from pathlib import Path

from PIL import Image

# Les rasters RVT/MNT "existing_*" peuvent dépasser 100 Mpx (datasets LiDAR
# locaux multi-km²). Par défaut PIL lève `DecompressionBombError` au-delà de
# ~178 Mpx pour se protéger d'images malicieuses. On est sur des TIF locaux
# maîtrisés, donc on désactive la limite pour laisser passer les grandes
# emprises, et SAHI fera ensuite le slicing à l'inférence.
Image.MAX_IMAGE_PIXELS = None


def convert_tif_to_png(
    input_path,
    output_path,
    create_world_file=True,
    reference_tif_path=None,
):
    try:
        with Image.open(input_path) as img:
            if img.mode == "P":
                img = img.convert("RGBA")
            if img.mode not in ("RGB", "RGBA", "L", "LA"):
                img = img.convert("RGB")

            img.save(output_path, "PNG", optimize=True)

        if create_world_file:
            ref_tif = reference_tif_path if reference_tif_path else input_path
            success_world = create_world_file_from_tif(ref_tif, output_path)
            if not success_world:
                logging.warning(f"Impossible de créer le fichier world pour {output_path}")

        return True

    except Exception as e:
        logging.error(f"Erreur lors de la conversion de {input_path}: {e}")
        return False


def create_world_file_from_tif(input_tif_path, output_png_path):
    """Crée un fichier world (.pgw) pour output_png_path à partir du géoréférencement de input_tif_path."""
    from ...geo_utils import create_world_file_from_tif as _create
    return _create(Path(input_tif_path), Path(output_png_path))
