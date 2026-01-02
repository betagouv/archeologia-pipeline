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
    try:
        try:
            import rasterio

            with rasterio.open(input_tif_path) as dataset:
                transform = dataset.transform

                pixel_width = transform.a
                row_rotation = transform.b
                x_origin = transform.c
                col_rotation = transform.d
                pixel_height = transform.e
                y_origin = transform.f

                logging.info("Géoréférencement extrait avec rasterio")

        except Exception:
            try:
                from osgeo import gdal

                dataset = gdal.Open(input_tif_path)
                if not dataset:
                    logging.warning(f"Impossible d'ouvrir le fichier TIF: {input_tif_path}")
                    return False

                geotransform = dataset.GetGeoTransform()
                if not geotransform or geotransform == (0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
                    logging.warning(f"Pas d'informations de géoréférencement dans: {input_tif_path}")
                    dataset = None
                    return False

                x_origin, pixel_width, row_rotation, y_origin, col_rotation, pixel_height = geotransform
                dataset = None

                logging.info("Géoréférencement extrait avec GDAL (bindings)")

            except Exception:
                import json
                import shutil
                import subprocess

                gdalinfo = shutil.which("gdalinfo") or shutil.which(r"C:\\OSGeo4W\\bin\\gdalinfo.exe")
                if not gdalinfo:
                    logging.warning("Ni rasterio ni GDAL disponibles - impossible de créer le fichier world")
                    return False

                try:
                    proc = subprocess.run(
                        [gdalinfo, "-json", input_tif_path],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    info = json.loads(proc.stdout)
                    gt = info.get("geoTransform") or info.get("geoTransform", None)
                    if not gt or len(gt) < 6:
                        logging.warning(
                            f"gdalinfo n'a pas renvoyé de geoTransform utilisable pour: {input_tif_path}"
                        )
                        return False
                    x_origin, pixel_width, row_rotation, y_origin, col_rotation, pixel_height = gt[:6]
                    logging.info("Géoréférencement extrait avec gdalinfo (CLI)")
                except Exception as cli_err:
                    logging.warning(f"Échec gdalinfo -json pour {input_tif_path}: {cli_err}")
                    return False

        jgw_path = Path(output_jpg_path).with_suffix(".jgw")

        with open(jgw_path, "w") as f:
            f.write(f"{pixel_width:.10f}\n")
            f.write(f"{row_rotation:.10f}\n")
            f.write(f"{col_rotation:.10f}\n")
            f.write(f"{pixel_height:.10f}\n")
            f.write(f"{x_origin:.10f}\n")
            f.write(f"{y_origin:.10f}\n")

        logging.info(f"Fichier world créé: {jgw_path}")
        return True

    except Exception as e:
        logging.error(f"Erreur lors de la création du fichier world: {e}")
        return False
