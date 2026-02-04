"""
Module commun pour la génération des fichiers de sortie CV (.txt, .json).
Utilisé par les runners YOLO et RF-DETR pour éviter la duplication de code.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def get_detection_output_path(jpg_path: str, target_rvt: str, annotated_output_dir: str = None) -> str:
    """
    Génère le chemin de sortie pour les détections dans le dossier annotated_images.

    Args:
        jpg_path: Chemin vers l'image JPG
        target_rvt: Type de produit RVT (LDO, SVF, etc.)
        annotated_output_dir: Répertoire de sortie pour les images annotées

    Returns:
        Chemin vers le fichier de détection de sortie dans annotated_images
    """
    jpg_file = Path(jpg_path)
    detection_name = jpg_file.stem + "_detections.jpg"

    if annotated_output_dir:
        detection_path = Path(annotated_output_dir) / detection_name
    else:
        detection_path = jpg_file.parent / detection_name

    return str(detection_path)


def save_empty_outputs(image_path: str, output_path: str, jpg_folder_path: Optional[str] = None) -> None:
    """
    Sauvegarde des fichiers vides quand aucune détection n'est trouvée.
    
    Args:
        image_path: Chemin vers l'image source
        output_path: Chemin de sortie pour l'image annotée
        jpg_folder_path: Chemin vers le dossier jpg pour sauvegarder les fichiers
    """
    try:
        image_name = Path(image_path).stem
        if jpg_folder_path:
            labels_txt = Path(jpg_folder_path) / f"{image_name}.txt"
            labels_json = Path(jpg_folder_path) / f"{image_name}.json"
        else:
            output_path_obj = Path(output_path)
            labels_txt = output_path_obj.with_suffix('.txt')
            labels_json = output_path_obj.with_suffix('.json')

        if not labels_txt.exists():
            labels_txt.write_text("")
            logger.info(f"Fichier labels vide créé (aucune détection): {labels_txt}")

        try:
            from PIL import Image
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception:
            img_width, img_height = None, None

        payload = {
            "image_path": image_path,
            "image_dimensions": {"width": img_width, "height": img_height},
            "detections": []
        }
        labels_json.write_text(json.dumps(payload, indent=2))
        logger.info(f"JSON vide créé (aucune détection): {labels_json}")
    except Exception as write_e:
        logger.warning(f"Impossible d'écrire les sorties vides pour {image_path}: {write_e}")


def save_detections_to_files(
    *,
    image_path: str,
    output_path: str,
    detections: List[Dict],
    img_width: int,
    img_height: int,
    jpg_folder_path: Optional[str] = None,
    task: str = "detect",
    model_type: str = "yolo",
) -> Tuple[Path, Path]:
    """
    Sauvegarde les détections au format YOLO (.txt) et JSON.
    
    Args:
        image_path: Chemin vers l'image source
        output_path: Chemin de sortie pour l'image annotée
        detections: Liste des détections avec format:
            - Pour bbox: {"class_id": int, "confidence": float, "bbox": [x1, y1, x2, y2]}
            - Pour polygon: {"class_id": int, "confidence": float, "polygon": [...]}
        img_width: Largeur de l'image
        img_height: Hauteur de l'image
        jpg_folder_path: Chemin vers le dossier jpg pour sauvegarder les fichiers
        task: Type de tâche ("detect" ou "segment")
        model_type: Type de modèle ("yolo" ou "rfdetr")
    
    Returns:
        Tuple (txt_path, json_path)
    """
    image_name = Path(image_path).stem
    if jpg_folder_path:
        txt_path = Path(jpg_folder_path) / f"{image_name}.txt"
    else:
        txt_path = Path(output_path).with_suffix('.txt')
    
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    
    detections_data = []
    
    with open(txt_path, "w") as f:
        for det in detections:
            class_id = det.get("class_id", 0)
            confidence = det.get("confidence")
            
            if "polygon" in det:
                # Format segmentation: class_id x1 y1 x2 y2 ...
                polygon = det["polygon"]
                coords = []
                for i in range(0, len(polygon), 2):
                    x_rel = polygon[i]
                    y_rel = polygon[i + 1]
                    coords.extend([x_rel, y_rel])
                f.write(f"{class_id} " + " ".join(f"{v:.6f}" for v in coords) + "\n")
                detections_data.append({
                    "class_id": class_id,
                    "confidence": confidence,
                    "polygon": polygon,
                })
            elif "bbox" in det:
                # Format détection: class_id x_center y_center width height
                x1, y1, x2, y2 = det["bbox"]
                x_center_rel = ((x1 + x2) / 2.0) / float(img_width)
                y_center_rel = ((y1 + y2) / 2.0) / float(img_height)
                w_rel = (x2 - x1) / float(img_width)
                h_rel = (y2 - y1) / float(img_height)
                f.write(f"{class_id} {x_center_rel:.6f} {y_center_rel:.6f} {w_rel:.6f} {h_rel:.6f}\n")
                detections_data.append({
                    "class_id": class_id,
                    "confidence": confidence,
                    "bbox_absolute": {"minx": x1, "miny": y1, "maxx": x2, "maxy": y2},
                })
            elif "bbox_absolute" in det:
                # Format déjà en bbox_absolute
                bbox = det["bbox_absolute"]
                x1, y1, x2, y2 = bbox["minx"], bbox["miny"], bbox["maxx"], bbox["maxy"]
                x_center_rel = ((x1 + x2) / 2.0) / float(img_width)
                y_center_rel = ((y1 + y2) / 2.0) / float(img_height)
                w_rel = (x2 - x1) / float(img_width)
                h_rel = (y2 - y1) / float(img_height)
                f.write(f"{class_id} {x_center_rel:.6f} {y_center_rel:.6f} {w_rel:.6f} {h_rel:.6f}\n")
                detections_data.append({
                    "class_id": class_id,
                    "confidence": confidence,
                    "bbox_absolute": bbox,
                })
    
    logger.info(f"Détections sauvegardées au format YOLO: {txt_path}")
    
    # Sauvegarder le JSON
    json_path = txt_path.with_suffix('.json')
    json_payload = {
        "image_path": image_path,
        "image_dimensions": {"width": img_width, "height": img_height},
        "detections": detections_data,
        "task": task,
    }
    if model_type != "yolo":
        json_payload["model_type"] = model_type
    
    json_path.write_text(json.dumps(json_payload, indent=2))
    logger.info(f"Données complètes sauvegardées en JSON: {json_path}")
    
    return txt_path, json_path


# Import des couleurs depuis class_utils (source unique de vérité)
from .class_utils import BASE_COLOR_PALETTE, get_class_color, get_color_for_confidence

# Alias pour compatibilité
CLASS_COLORS = BASE_COLOR_PALETTE


def save_annotated_image(
    pil_image,
    detections: List[Dict],
    output_path: str,
    class_names: List[str] = None,
    class_colors: List[int] = None,
    width: int = 2,
    jpeg_quality: int = 95,
) -> None:
    """
    Sauvegarde une image annotée avec les détections.
    
    Args:
        pil_image: Image PIL source
        detections: Liste des détections avec "bbox" [x1, y1, x2, y2]
        output_path: Chemin de sortie
        class_names: Liste des noms de classes (optionnel)
        class_colors: Liste des indices de couleurs par classe (optionnel, depuis args.yaml)
        width: Épaisseur des lignes
        jpeg_quality: Qualité JPEG (1-100)
    """
    try:
        from PIL import ImageDraw, ImageFont
        
        img_copy = pil_image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # Police compacte
        try:
            font = ImageFont.truetype("arial.ttf", 10)
        except Exception:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
            except Exception:
                font = ImageFont.load_default()
        
        # Log pour debug des couleurs (print pour être visible dans le runner compilé et console Python QGIS)
        print(f"[CV_OUTPUT] save_annotated_image: class_colors={class_colors} (type={type(class_colors).__name__ if class_colors else 'None'})", flush=True)
        print(f"[CV_OUTPUT] save_annotated_image: class_names={class_names}", flush=True)
        print(f"[CV_OUTPUT] save_annotated_image: detections count={len(detections)}", flush=True)
        logger.info(f"save_annotated_image: class_colors={class_colors} (type={type(class_colors).__name__ if class_colors else 'None'})")
        
        for det in detections:
            if "bbox" in det:
                x1, y1, x2, y2 = det["bbox"]
            elif "bbox_absolute" in det:
                bbox = det["bbox_absolute"]
                x1, y1, x2, y2 = bbox["minx"], bbox["miny"], bbox["maxx"], bbox["maxy"]
            else:
                continue
            
            class_id = det.get("class_id", 0)
            confidence = det.get("confidence", 0)
            
            # Couleur selon la classe (avec support des couleurs personnalisées)
            color = get_class_color(class_id, class_colors)
            logger.debug(f"Detection class_id={class_id}, color_index={class_colors[class_id] if class_colors and class_id < len(class_colors) else class_id}, color={color}")
            
            # Dessiner le rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            
            # Label compact: juste le % de confiance
            label = f"{int(confidence * 100)}%"
            
            # Calculer la taille du texte
            try:
                bbox_text = draw.textbbox((0, 0), label, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
            except Exception:
                text_width, text_height = len(label) * 6, 10
            
            # Position du label: coin supérieur gauche INTÉRIEUR de la bbox
            padding = 1
            label_x = x1 + padding
            label_y = y1 + padding
            
            # Petit fond coloré pour le label
            draw.rectangle(
                [label_x, label_y, label_x + text_width + padding * 2, label_y + text_height + padding],
                fill=color
            )
            
            # Texte blanc
            draw.text((label_x + padding, label_y), label, fill=(255, 255, 255), font=font)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder avec haute qualité
        if output_path.lower().endswith(('.jpg', '.jpeg')):
            img_copy.save(output_path, quality=jpeg_quality, optimize=True)
        else:
            img_copy.save(output_path)
        logger.info(f"Image annotée sauvegardée: {output_path}")
    except Exception as e:
        logger.warning(f"Impossible de sauvegarder l'image annotée: {e}")
        import traceback
        traceback.print_exc()


def save_legend_file(
    output_dir: str,
    class_names: List[str],
    class_colors: List[int] = None,
) -> None:
    """
    Crée un fichier légende (image + texte) indiquant quelle classe correspond à quelle couleur.
    
    Args:
        output_dir: Dossier de sortie (annotated_images)
        class_names: Liste des noms de classes
        class_colors: Liste des indices de couleurs par classe (optionnel, depuis args.yaml)
    """
    if not class_names:
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Créer le fichier texte
    txt_path = output_path / "legend.txt"
    lines = ["LÉGENDE DES CLASSES", "=" * 30, ""]
    for i, name in enumerate(class_names):
        color = get_class_color(i, class_colors)
        lines.append(f"{i}: {name} - RGB{color}")
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Légende texte sauvegardée: {txt_path}")
    
    # Créer l'image légende
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Dimensions
        row_height = 30
        padding = 10
        color_box_size = 20
        img_width = 400
        img_height = padding * 2 + row_height * len(class_names) + 30  # +30 pour le titre
        
        img = Image.new("RGB", (img_width, img_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Police
        try:
            font = ImageFont.truetype("arial.ttf", 14)
            font_title = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
                font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except Exception:
                font = ImageFont.load_default()
                font_title = font
        
        # Titre
        draw.text((padding, padding), "Légende des classes", fill=(0, 0, 0), font=font_title)
        
        # Lignes de légende
        y = padding + 30
        for i, name in enumerate(class_names):
            color = get_class_color(i, class_colors)
            
            # Carré de couleur
            draw.rectangle(
                [padding, y, padding + color_box_size, y + color_box_size],
                fill=color,
                outline=(0, 0, 0)
            )
            
            # Nom de la classe
            draw.text((padding + color_box_size + 10, y + 2), name, fill=(0, 0, 0), font=font)
            
            y += row_height
        
        img_path = output_path / "legend.png"
        img.save(img_path)
        logger.info(f"Légende image sauvegardée: {img_path}")
    except Exception as e:
        logger.warning(f"Impossible de créer l'image légende: {e}")
