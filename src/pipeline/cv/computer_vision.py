"""
Module de computer vision pour la détection d'objets sur les images RVT
Utilise YOLO pour la détection et génère des shapefiles des résultats
"""

import logging
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import tempfile

from .class_utils import get_num_classes_from_model, detect_indexing_offset
from .cv_output import (
    get_detection_output_path,
    save_empty_outputs,
    save_detections_to_files,
    save_annotated_image,
)

logger = logging.getLogger(__name__)


def _get_ultralytics_task(model_path: str) -> Optional[str]:
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        task = getattr(model, "task", None)
        if isinstance(task, str) and task:
            return task
    except Exception as e:
        logger.debug(f"Impossible de déterminer la tâche Ultralytics pour {model_path}: {e}")
    return None


# Alias pour compatibilité avec le code existant
_save_empty_outputs = save_empty_outputs


def run_inference(image_path: str, model_path: str, args_path: str, output_path: str,
                 confidence_threshold: float = 0.5, slice_height: int = 640,
                 slice_width: int = 640, overlap_ratio: float = 0.2,
                 generate_annotated_images: bool = False, annotated_output_dir: str = None,
                 iou_threshold: float = 0.5, jpg_folder_path: str = None,
                 max_det: int = 1000) -> bool:
    """
    Exécute l'inférence YOLO sur une image

    Args:
        image_path: Chemin vers l'image d'entrée
        model_path: Chemin vers le modèle YOLO
        args_path: Chemin vers le fichier d'arguments
        output_path: Chemin de sortie pour l'image avec détections
        confidence_threshold: Seuil de confiance pour les détections
        slice_height: Hauteur des tuiles pour SAHI
        slice_width: Largeur des tuiles pour SAHI
        overlap_ratio: Ratio de chevauchement pour les tuiles
        generate_annotated_images: Générer des images annotées
        annotated_output_dir: Répertoire de sortie pour les images annotées
        iou_threshold: Seuil IoU pour la suppression des doublons
        jpg_folder_path: Chemin vers le dossier jpg pour sauvegarder les fichiers .txt

    Returns:
        True si l'inférence a réussi, False sinon
    """
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image non trouvée: {image_path}")
            return False

        if not os.path.exists(model_path):
            logger.error(f"Modèle non trouvé: {model_path}")
            return False

        task = _get_ultralytics_task(model_path)
        if task:
            logger.info(f"Tâche du modèle détectée (Ultralytics): {task}")
        else:
            task = "detect"
            logger.info("Tâche du modèle non déterminée, fallback: detect")

        if task == "segment":
            return _run_segmentation_inference(
                image_path=image_path,
                model_path=model_path,
                output_path=output_path,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                generate_annotated_images=generate_annotated_images,
                jpg_folder_path=jpg_folder_path,
                max_det=max_det,
            )

        try:
            try:
                from sahi.auto_model import AutoDetectionModel
            except Exception:
                from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
        except Exception as e:
            logger.warning(f"Import SAHI impossible ({e}). Fallback Ultralytics direct.")
            return _run_ultralytics_bbox_inference(
                image_path=image_path,
                model_path=model_path,
                output_path=output_path,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                generate_annotated_images=generate_annotated_images,
                annotated_output_dir=annotated_output_dir,
                jpg_folder_path=jpg_folder_path,
                max_det=max_det,
            )

        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device="cpu"
        )

        try:
            if hasattr(detection_model.model, "overrides") and isinstance(detection_model.model.overrides, dict):
                detection_model.model.overrides["max_det"] = max_det
            if hasattr(detection_model.model, "predictor") and hasattr(detection_model.model.predictor, "args"):
                try:
                    detection_model.model.predictor.args["max_det"] = max_det
                except Exception:
                    pass
            logger.info(f"Paramètre max_det réglé à {max_det}")
        except Exception as e:
            logger.debug(f"Impossible de régler max_det: {e}")

        result = get_sliced_prediction(
            image=image_path,
            detection_model=detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio
        )

        has_detections = len(result.object_prediction_list) > 0

        if has_detections:
            result.export_visuals(export_dir=str(Path(output_path).parent))

            exported_path = Path(output_path).parent / "prediction_visual.png"
            output_path_obj = Path(output_path)

            if exported_path.exists():
                if output_path_obj.exists():
                    output_path_obj.unlink()
                    logger.info(f"Fichier existant supprimé: {output_path}")

                exported_path.rename(output_path)
            save_detections_to_yolo_format(result, image_path, output_path, jpg_folder_path)
        else:
            logger.info(f"Aucune détection trouvée dans {Path(image_path).name}, image non sauvegardée")
            _save_empty_outputs(image_path=image_path, output_path=output_path, jpg_folder_path=jpg_folder_path)

        return has_detections
    except Exception as e:
        logger.error(f"Erreur lors de l'inférence YOLO: {e}")
        return False


def _run_ultralytics_bbox_inference(
    *,
    image_path: str,
    model_path: str,
    output_path: str,
    confidence_threshold: float,
    iou_threshold: float,
    generate_annotated_images: bool,
    annotated_output_dir: Optional[str],
    jpg_folder_path: Optional[str],
    max_det: int,
) -> bool:
    try:
        from ultralytics import YOLO
        from PIL import Image

        with Image.open(image_path) as img:
            img_width, img_height = img.size

        model = YOLO(model_path)
        results = model.predict(
            source=image_path,
            conf=confidence_threshold,
            iou=iou_threshold,
            max_det=max_det,
            verbose=False,
        )
        if not results:
            _save_empty_outputs(image_path=image_path, output_path=output_path, jpg_folder_path=jpg_folder_path)
            return False

        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            _save_empty_outputs(image_path=image_path, output_path=output_path, jpg_folder_path=jpg_folder_path)
            return False

        xyxy = boxes.xyxy.cpu().numpy().tolist() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
        cls = boxes.cls.cpu().numpy().tolist() if getattr(boxes, "cls", None) is not None and hasattr(boxes.cls, "cpu") else []
        confs = boxes.conf.cpu().numpy().tolist() if getattr(boxes, "conf", None) is not None and hasattr(boxes.conf, "cpu") else []

        image_name = Path(image_path).stem
        txt_path = Path(jpg_folder_path) / f"{image_name}.txt" if jpg_folder_path else Path(output_path).with_suffix('.txt')
        txt_path.parent.mkdir(parents=True, exist_ok=True)

        detections_data = []
        with open(txt_path, "w") as f:
            for i, bb in enumerate(xyxy or []):
                if bb is None or len(bb) != 4:
                    continue
                x1, y1, x2, y2 = [float(v) for v in bb]
                x_center_rel = ((x1 + x2) / 2.0) / float(img_width)
                y_center_rel = ((y1 + y2) / 2.0) / float(img_height)
                w_rel = (x2 - x1) / float(img_width)
                h_rel = (y2 - y1) / float(img_height)
                class_id = int(cls[i]) if i < len(cls) else 0
                f.write(f"{class_id} {x_center_rel:.6f} {y_center_rel:.6f} {w_rel:.6f} {h_rel:.6f}\n")
                detections_data.append(
                    {
                        "class_id": class_id,
                        "confidence": float(confs[i]) if i < len(confs) else None,
                        "bbox_absolute": {"minx": x1, "miny": y1, "maxx": x2, "maxy": y2},
                    }
                )

        json_path = txt_path.with_suffix('.json')
        json_path.write_text(
            json.dumps(
                {
                    "image_path": image_path,
                    "image_dimensions": {"width": img_width, "height": img_height},
                    "detections": detections_data,
                    "task": "detect",
                },
                indent=2,
            )
        )

        if generate_annotated_images:
            try:
                im = r0.plot()
                from PIL import Image as PILImage
                out_img = PILImage.fromarray(im[..., ::-1]) if hasattr(im, 'shape') else None
                if out_img is not None:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    out_img.save(output_path)
            except Exception as e:
                logger.warning(f"Impossible de sauvegarder l'image annotée (detect): {e}")

        return len(detections_data) > 0
    except Exception as e:
        logger.error(f"Erreur fallback Ultralytics bbox: {e}")
        return False


def _run_segmentation_inference(
    image_path: str,
    model_path: str,
    output_path: str,
    confidence_threshold: float,
    iou_threshold: float,
    generate_annotated_images: bool,
    jpg_folder_path: Optional[str],
    max_det: int,
) -> bool:
    try:
        from ultralytics import YOLO
        from PIL import Image

        model = YOLO(model_path)

        with Image.open(image_path) as img:
            img_width, img_height = img.size

        results = model.predict(
            source=image_path,
            conf=confidence_threshold,
            iou=iou_threshold,
            max_det=max_det,
            verbose=False,
        )

        if not results:
            _save_empty_outputs(image_path=image_path, output_path=output_path, jpg_folder_path=jpg_folder_path)
            return False

        r0 = results[0]
        masks = getattr(r0, "masks", None)
        if masks is None or getattr(masks, "xy", None) is None:
            logger.warning("Le modèle est marqué segment mais aucun masque n'a été produit")
            _save_empty_outputs(image_path=image_path, output_path=output_path, jpg_folder_path=jpg_folder_path)
            return False

        polygons_xy = list(masks.xy)
        if not polygons_xy:
            _save_empty_outputs(image_path=image_path, output_path=output_path, jpg_folder_path=jpg_folder_path)
            return False

        classes = []
        confidences = []
        try:
            if getattr(r0, "boxes", None) is not None:
                b = r0.boxes
                if getattr(b, "cls", None) is not None:
                    classes = [int(x) for x in b.cls.cpu().numpy().tolist()]
                if getattr(b, "conf", None) is not None:
                    confidences = [float(x) for x in b.conf.cpu().numpy().tolist()]
        except Exception:
            classes = []
            confidences = []

        while len(classes) < len(polygons_xy):
            classes.append(0)
        while len(confidences) < len(polygons_xy):
            confidences.append(None)

        image_name = Path(image_path).stem
        if jpg_folder_path:
            txt_path = Path(jpg_folder_path) / f"{image_name}.txt"
        else:
            txt_path = Path(output_path).with_suffix('.txt')

        detections_data = []
        with open(txt_path, 'w') as f:
            for idx, (cls_id, poly) in enumerate(zip(classes, polygons_xy)):
                if poly is None or len(poly) < 3:
                    continue

                coords = []
                for x, y in poly:
                    x_rel = float(x) / float(img_width)
                    y_rel = float(y) / float(img_height)
                    coords.extend([x_rel, y_rel])

                f.write(f"{int(cls_id)} " + " ".join(f"{v:.6f}" for v in coords) + "\n")

                detections_data.append({
                    "class_id": int(cls_id),
                    "confidence": confidences[idx],
                    "polygon": coords,
                })

        json_path = txt_path.with_suffix('.json')
        json_path.write_text(json.dumps({
            "image_path": image_path,
            "image_dimensions": {"width": img_width, "height": img_height},
            "detections": detections_data,
            "task": "segment",
        }, indent=2))

        logger.info(f"Segmentations sauvegardées au format YOLO-seg: {txt_path}")
        logger.info(f"Données complètes sauvegardées en JSON: {json_path}")

        if generate_annotated_images:
            try:
                im = r0.plot()
                from PIL import Image as PILImage
                out_img = PILImage.fromarray(im[..., ::-1]) if hasattr(im, 'shape') else None
                if out_img is not None:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    out_img.save(output_path)
            except Exception as e:
                logger.warning(f"Impossible de sauvegarder l'image annotée (segmentation): {e}")

        return len(detections_data) > 0

    except Exception as e:
        logger.error(f"Erreur lors de l'inférence segmentation: {e}")
        return False


def get_class_names_from_model(model_path: str) -> Optional[List[str]]:
    """
    Récupère les noms des classes depuis un modèle YOLO

    Args:
        model_path: Chemin vers le modèle YOLO

    Returns:
        Liste des noms de classes ou None si échec
    """
    try:
        from ultralytics import YOLO

        model = YOLO(model_path)
        names = getattr(model, "names", None)
        if names is None:
            return None

        # ultralytics expose parfois `names` en dict {id: name} ou en list.
        if isinstance(names, dict):
            try:
                # Garantir l'ordre 0..n (sinon on peut avoir un mismatch)
                return [names[k] for k in sorted(names.keys())]
            except Exception:
                # Dernier recours: ordre d'itération
                return list(names.values())

        if isinstance(names, (list, tuple)):
            return [str(x) for x in names]

        return None

    except Exception as e:
        logger.error(f"Erreur lors de la récupération des noms de classes: {e}")
        return None


def save_detections_to_yolo_format(result, image_path: str, output_path: str, jpg_folder_path: str = None):
    """
    Sauvegarde les détections SAHI au format YOLO (.txt) pour la création de shapefiles

    Args:
        result: Résultat de prédiction SAHI
        image_path: Chemin vers l'image source
        output_path: Chemin vers l'image avec détections
        jpg_folder_path: Chemin vers le dossier jpg pour sauvegarder les fichiers .txt
    """
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            img_width, img_height = img.size

        if jpg_folder_path:
            image_name = Path(image_path).stem
            txt_path = Path(jpg_folder_path) / f"{image_name}.txt"
        else:
            output_path_obj = Path(output_path)
            txt_path = output_path_obj.with_suffix('.txt')

        with open(txt_path, 'w') as f:
            for detection in result.object_prediction_list:
                bbox = detection.bbox

                x_center_rel = (bbox.minx + bbox.maxx) / 2 / img_width
                y_center_rel = (bbox.miny + bbox.maxy) / 2 / img_height
                width_rel = (bbox.maxx - bbox.minx) / img_width
                height_rel = (bbox.maxy - bbox.miny) / img_height

                f.write(f"{detection.category.id} {x_center_rel:.6f} {y_center_rel:.6f} {width_rel:.6f} {height_rel:.6f}\n")

        logger.info(f"Détections sauvegardées au format YOLO: {txt_path}")

        json_path = txt_path.with_suffix('.json')
        detections_data = []

        for detection in result.object_prediction_list:
            bbox = detection.bbox

            x_center_rel = (bbox.minx + bbox.maxx) / 2 / img_width
            y_center_rel = (bbox.miny + bbox.maxy) / 2 / img_height
            width_rel = (bbox.maxx - bbox.minx) / img_width
            height_rel = (bbox.maxy - bbox.miny) / img_height

            detections_data.append({
                "class_id": detection.category.id,
                "x_center": x_center_rel,
                "y_center": y_center_rel,
                "width": width_rel,
                "height": height_rel,
                "confidence": detection.score.value,
                "bbox_absolute": {
                    "minx": bbox.minx,
                    "miny": bbox.miny,
                    "maxx": bbox.maxx,
                    "maxy": bbox.maxy
                }
            })

        with open(json_path, 'w') as json_file:
            json.dump({
                "image_path": image_path,
                "image_dimensions": {"width": img_width, "height": img_height},
                "detections": detections_data
            }, json_file, indent=2)

        logger.info(f"Données complètes sauvegardées en JSON: {json_path}")

    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des détections YOLO: {e}")


def create_shapefile_from_detections(detection_results: Dict, output_shapefile: str,
                                   class_names: Optional[List[str]] = None) -> bool:
    """
    Crée un shapefile à partir des résultats de détection

    Args:
        detection_results: Résultats de détection SAHI
        output_shapefile: Chemin de sortie pour le shapefile
        class_names: Noms des classes (optionnel)

    Returns:
        True si le shapefile a été créé avec succès, False sinon
    """
    try:
        try:
            import geopandas as gpd
            from shapely.geometry import Polygon
            import pandas as pd
        except ImportError:
            logger.warning("geopandas ou shapely non disponibles. Shapefile non créé.")
            return False

        detections = []
        if hasattr(detection_results, 'object_prediction_list'):
            for detection in detection_results.object_prediction_list:
                bbox = detection.bbox

                polygon = Polygon([
                    (bbox.minx, bbox.miny),
                    (bbox.maxx, bbox.miny),
                    (bbox.maxx, bbox.maxy),
                    (bbox.minx, bbox.maxy)
                ])

                class_name = "unknown"
                if class_names and detection.category.id < len(class_names):
                    class_name = class_names[detection.category.id]

                detections.append({
                    'geometry': polygon,
                    'class_id': detection.category.id,
                    'class_name': class_name,
                    'confidence': detection.score.value
                })

        if not detections:
            logger.info("Aucune détection trouvée pour créer le shapefile")
            return False

        gdf = gpd.GeoDataFrame(detections)

        try:
            logger.info(f"🔍 DIAGNOSTIC: Tentative de sauvegarde shapefile...")
            logger.info(f"   GeoDataFrame shape: {gdf.shape}")
            logger.info(f"   Colonnes: {list(gdf.columns)}")
            logger.info(f"   CRS: {getattr(gdf, 'crs', 'Non défini')}")
            logger.info(f"   Chemin de sortie: {output_shapefile}")

            try:
                import pandas._libs.window.aggregations
                logger.info("   ✅ pandas._libs.window.aggregations disponible")
            except Exception as mod_e:
                logger.error(f"   ❌ pandas._libs.window.aggregations: {mod_e}")

            gdf.to_file(output_shapefile)
            logger.info(f"✅ Shapefile créé avec succès: {output_shapefile}")

        except Exception as save_e:
            logger.error(f"❌ ERREUR SAUVEGARDE SHAPEFILE: {save_e}")
            logger.error(f"   Type d'erreur: {type(save_e).__name__}")

            import traceback
            tb_lines = traceback.format_exc().split('\n')
            for i, line in enumerate(tb_lines):
                if line.strip():
                    logger.error(f"   TB[{i}]: {line}")

            try:
                logger.info("🔄 Tentative de sauvegarde alternative...")
                gdf_no_crs = gdf.copy()
                gdf_no_crs.crs = None
                gdf_no_crs.to_file(output_shapefile)
                logger.info("✅ Sauvegarde alternative réussie (sans CRS)")
            except Exception as alt_e:
                logger.error(f"❌ Sauvegarde alternative échouée: {alt_e}")
                raise save_e

        return True

    except Exception as e:
        logger.error(f"Erreur lors de la création du shapefile: {e}")
        return False


def setup_computer_vision_environment():
    """
    Configure l'environnement pour la computer vision
    Vérifie les dépendances et affiche des avertissements si nécessaire
    """
    missing_deps = []

    try:
        import sahi
    except ImportError:
        missing_deps.append("sahi")

    try:
        import ultralytics
    except ImportError:
        missing_deps.append("ultralytics")

    try:
        import geopandas
    except ImportError:
        missing_deps.append("geopandas")

    try:
        import shapely
    except ImportError:
        missing_deps.append("shapely")

    if missing_deps:
        logger.warning(f"Dépendances manquantes pour la computer vision: {', '.join(missing_deps)}")
        logger.warning("Installez-les avec: pip install sahi ultralytics geopandas shapely")
        return False

    logger.info("Environnement computer vision configuré avec succès")
    return True
