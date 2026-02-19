import logging
from pathlib import Path

from PIL import Image


def convert_tif_to_jpg(
    input_path,
    output_path,
    quality=95,
    create_world_file=True,
    reference_tif_path=None,
):
    try:
        with Image.open(input_path) as img:
            if img.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            img.save(output_path, "JPEG", quality=quality, optimize=True)

        if create_world_file:
            ref_tif = reference_tif_path if reference_tif_path else input_path
            success_world = create_world_file_from_tif(ref_tif, output_path)
            if not success_world:
                logging.warning(f"Impossible de créer le fichier world pour {output_path}")

        return True

    except Exception as e:
        logging.error(f"Erreur lors de la conversion de {input_path}: {e}")
        return False


def create_world_file_from_tif(input_tif_path, output_jpg_path):
    """Crée un fichier world pour output_jpg_path à partir du géoréférencement de input_tif_path."""
    from ...geo_utils import create_world_file_from_tif as _create
    return _create(Path(input_tif_path), Path(output_jpg_path))
