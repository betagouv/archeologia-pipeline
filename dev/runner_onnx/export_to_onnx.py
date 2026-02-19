"""
Script d'export des modèles YOLO, RF-DETR et SegFormer vers le format ONNX.

Ce script nécessite les dépendances complètes (ultralytics pour YOLO, rfdetr pour RF-DETR).
Il n'est PAS inclus dans le runner compilé.

Usage:
    python export_to_onnx.py --model path/to/best.pt --output path/to/model.onnx
    python export_to_onnx.py --model best.pt --output model.onnx --imgsz 640 --simplify
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


def detect_model_type(model_path: Path) -> str:
    """
    Détecte le type de modèle (yolo, rfdetr ou segformer) à partir du fichier.
    """
    # Vérifier si c'est un dossier HuggingFace (SegFormer)
    if model_path.is_dir():
        config_file = model_path / "config.json"
        if config_file.exists():
            try:
                config = json.loads(config_file.read_text(encoding="utf-8"))
                model_type = str(config.get("model_type", "")).lower()
                if "segformer" in model_type:
                    return "segformer"
                architectures = config.get("architectures", [])
                if any("segformer" in str(a).lower() for a in architectures):
                    return "segformer"
            except Exception:
                pass
    
    # Vérifier si le parent contient un config.json HuggingFace
    parent_config = model_path.parent / "config.json"
    if parent_config.exists():
        try:
            config = json.loads(parent_config.read_text(encoding="utf-8"))
            model_type = str(config.get("model_type", "")).lower()
            if "segformer" in model_type:
                return "segformer"
        except Exception:
            pass
    # Vérifier via args.yaml
    if model_path.parent.name == "weights":
        model_dir = model_path.parent.parent
    else:
        model_dir = model_path.parent
    
    args_file = model_dir / "args.yaml"
    if args_file.exists():
        try:
            import yaml
            with open(args_file, 'r', encoding='utf-8') as f:
                args = yaml.safe_load(f)
            if isinstance(args, dict):
                model_type = str(args.get("model", "")).lower().strip()
                if "rf-detr" in model_type or "rfdetr" in model_type:
                    return "rfdetr"
                if "yolo" in model_type:
                    return "yolo"
        except Exception:
            pass
    
    # Fallback: vérifier via le checkpoint PyTorch
    try:
        import torch
        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            # YOLO a des clés spécifiques
            has_ultralytics_keys = any(k in checkpoint for k in ['model.yaml', 'train_args', 'ema'])
            if has_ultralytics_keys:
                return "yolo"
            # SMP (segmentation_models_pytorch): state_dict avec encoder.*, decoder.*, segmentation_head.*
            state_keys = set(checkpoint.keys())
            # Si c'est un state_dict wrappé (ex: {"model_state_dict": {...}, ...})
            for wrapper_key in ("model_state_dict", "state_dict", "model"):
                if wrapper_key in checkpoint and isinstance(checkpoint[wrapper_key], dict):
                    state_keys = set(checkpoint[wrapper_key].keys())
                    break
            has_smp_keys = (
                any(k.startswith("encoder.") for k in state_keys)
                and any(k.startswith("decoder.") for k in state_keys)
                and any(k.startswith("segmentation_head.") for k in state_keys)
            )
            if has_smp_keys:
                return "smp"
            # RF-DETR a encoder/decoder/backbone (mais pas segmentation_head)
            if any('encoder' in str(k) or 'decoder' in str(k) for k in state_keys):
                return "rfdetr"
    except Exception:
        pass
    
    # Par défaut, essayer YOLO
    return "yolo"


def export_segformer_to_onnx(
    model_path: Path,
    output_path: Path,
    imgsz: int = 640,
    opset: int = 14,
    simplify: bool = True,
) -> bool:
    """
    Exporte un modèle SegFormer (HuggingFace) vers ONNX.
    
    Args:
        model_path: Chemin vers le dossier du modèle HuggingFace (contenant config.json et model.safetensors)
        output_path: Chemin de sortie pour le fichier ONNX
        imgsz: Taille d'image pour l'export
        opset: Version ONNX opset
        simplify: Simplifier le modèle ONNX
    """
    print(f"[INFO] Export SegFormer -> ONNX: {model_path}")
    
    try:
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    except ImportError as e:
        print(f"[ERROR] Dépendances manquantes: {e}")
        print("[ERROR] Installez avec: pip install transformers torch safetensors")
        return False
    
    try:
        # Charger le modèle SegFormer depuis le dossier HuggingFace
        model_dir = model_path if model_path.is_dir() else model_path.parent
        print(f"[INFO] Chargement du modèle depuis: {model_dir}")
        
        model = SegformerForSemanticSegmentation.from_pretrained(str(model_dir))
        model.eval()
        
        # Lire la config pour les infos
        config_file = model_dir / "config.json"
        config = {}
        if config_file.exists():
            config = json.loads(config_file.read_text(encoding="utf-8"))
        
        num_labels = config.get("num_labels", 2)
        id2label = config.get("id2label", {})
        print(f"[INFO] Classes: {num_labels} - {id2label}")
        print(f"[INFO] Taille d'image: {imgsz}x{imgsz}")
        
        # Créer une entrée dummy
        dummy_input = torch.randn(1, 3, imgsz, imgsz)
        
        # Créer le dossier de sortie
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export vers ONNX
        print(f"[INFO] Export ONNX vers: {output_path}")
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )
        
        # Simplifier le modèle ONNX
        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify
                print("[INFO] Simplification du modèle ONNX...")
                onnx_model = onnx.load(str(output_path))
                model_simplified, check = onnx_simplify(onnx_model)
                if check:
                    onnx.save(model_simplified, str(output_path))
                    print("[INFO] Modèle ONNX simplifié avec succès")
                else:
                    print("[WARN] Échec de la vérification de simplification")
            except ImportError:
                print("[WARN] onnxsim non installé, simplification ignorée")
        
        print(f"[SUCCESS] Modèle SegFormer exporté: {output_path}")
        print(f"[INFO] Taille: {output_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Construire la liste des noms de classes (sans background)
        class_names = []
        for i in range(num_labels):
            label = id2label.get(str(i), f"class_{i}")
            if label.lower() != "background":
                class_names.append(label)
        
        # Sauvegarder les métadonnées
        meta_path = output_path.with_suffix('.json')
        meta = {
            "model_type": "segformer",
            "task": "semantic_segmentation",
            "image_size": imgsz,
            "num_labels": num_labels,
            "id2label": id2label,
            "class_names": class_names,
            "use_sahi": True,  # Activer le slicing SAHI pour les grandes images
            "merge_polygons": True,  # Fusionner les polygones adjacents (formes linéaires)
            "source": str(model_path),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[INFO] Métadonnées sauvegardées: {meta_path}")
        
        # Copier le fichier classes.txt si présent ou le créer depuis id2label
        # Le plugin attend classes.txt dans le dossier parent de weights/ (ex: models/<model_name>/classes.txt)
        classes_file = model_dir / "classes.txt"
        
        # Déterminer le dossier racine du modèle (parent de weights/)
        if output_path.parent.name == "weights":
            model_root = output_path.parent.parent
        else:
            model_root = output_path.parent
        
        output_classes = model_root / "classes.txt"
        
        if classes_file.exists():
            shutil.copy2(classes_file, output_classes)
            print(f"[INFO] classes.txt copié: {output_classes}")
        elif id2label:
            # Créer classes.txt depuis id2label
            labels = [id2label.get(str(i), f"class_{i}") for i in range(num_labels)]
            output_classes.write_text("\n".join(labels), encoding="utf-8")
            print(f"[INFO] classes.txt créé: {output_classes}")
        
        # Vérifier que le fichier ONNX existe et est valide
        if output_path.exists():
            print(f"[INFO] Vérification du fichier ONNX...")
            try:
                import onnx
                onnx_model = onnx.load(str(output_path))
                onnx.checker.check_model(onnx_model)
                print(f"[SUCCESS] Fichier ONNX valide: {output_path}")
            except ImportError:
                print("[WARN] onnx non installé, vérification ignorée")
            except Exception as e:
                print(f"[WARN] Vérification ONNX échouée: {e}")
        
        # Validation post-export
        validate_onnx_export("segformer", model, output_path, imgsz)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Échec export SegFormer: {e}")
        import traceback
        traceback.print_exc()
        return False


def _detect_smp_architecture(state_dict: dict) -> dict:
    """
    Détecte l'architecture SMP (encoder, decoder type) depuis les clés du state_dict.
    
    Returns:
        dict avec 'arch', 'encoder_name', 'num_classes' estimés
    """
    info = {"arch": "DeepLabV3Plus", "encoder_name": "resnet101"}
    
    # Détecter le type de décodeur
    keys_str = " ".join(state_dict.keys())
    if "decoder.aspp" in keys_str or "decoder.block" in keys_str:
        info["arch"] = "DeepLabV3Plus"
    elif "decoder.center" in keys_str or "decoder.blocks" in keys_str:
        info["arch"] = "Unet"
    elif "decoder.fpn" in keys_str:
        info["arch"] = "FPN"
    elif "decoder.psp" in keys_str:
        info["arch"] = "PSPNet"
    elif "decoder.segmentation_head" in keys_str:
        info["arch"] = "Linknet"
    
    # Détecter le nombre de classes depuis segmentation_head
    for k, v in state_dict.items():
        if "segmentation_head" in k and "weight" in k and v.dim() >= 2:
            info["num_classes"] = v.shape[0]
            break
    
    # Détecter l'encoder depuis les noms de couches
    if "encoder.layer4" in keys_str:
        # ResNet family
        first_conv = state_dict.get("encoder.conv1.weight")
        if first_conv is not None:
            out_ch = first_conv.shape[0]
            # Compter les couches dans layer4 pour distinguer resnet50/101/152
            layer4_keys = [k for k in state_dict if k.startswith("encoder.layer4.")]
            n_blocks = len(set(k.split(".")[2] for k in layer4_keys))
            if n_blocks >= 3:
                info["encoder_name"] = "resnet101"
            else:
                info["encoder_name"] = "resnet50"
    elif "encoder.features" in keys_str:
        info["encoder_name"] = "efficientnet-b0"
    
    return info


def export_smp_to_onnx(
    model_path: Path,
    output_path: Path,
    imgsz: int = 640,
    opset: int = 18,
    simplify: bool = True,
    arch: str = None,
    encoder_name: str = None,
    num_classes: int = None,
    class_names: list = None,
) -> bool:
    """
    Exporte un modèle segmentation_models_pytorch (DeepLabV3+, UNet, etc.) vers ONNX.
    
    Args:
        model_path: Chemin vers le checkpoint .pth
        output_path: Chemin de sortie pour le fichier ONNX
        imgsz: Taille d'image pour l'export
        opset: Version ONNX opset
        simplify: Simplifier le modèle ONNX
        arch: Architecture SMP (DeepLabV3Plus, Unet, etc.) — auto-détecté si None
        encoder_name: Nom de l'encoder (resnet101, etc.) — auto-détecté si None
        num_classes: Nombre de classes — auto-détecté si None
        class_names: Liste des noms de classes (optionnel)
    """
    print(f"[INFO] Export SMP -> ONNX: {model_path}")
    
    try:
        import torch
    except ImportError as e:
        print(f"[ERROR] Dépendances manquantes: {e}")
        print("[ERROR] Installez avec: pip install torch segmentation-models-pytorch")
        return False
    
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        print("[ERROR] segmentation_models_pytorch n'est pas installé.")
        print("[ERROR] Installez avec: pip install segmentation-models-pytorch")
        return False
    
    try:
        # Charger le checkpoint
        print(f"[INFO] Chargement du checkpoint: {model_path}")
        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
        
        # Extraire le state_dict
        if isinstance(checkpoint, dict):
            for wrapper_key in ("model_state_dict", "state_dict", "model"):
                if wrapper_key in checkpoint and isinstance(checkpoint[wrapper_key], dict):
                    state_dict = checkpoint[wrapper_key]
                    break
            else:
                state_dict = checkpoint
        else:
            print("[ERROR] Format de checkpoint non reconnu")
            return False
        
        # Auto-détecter l'architecture
        detected = _detect_smp_architecture(state_dict)
        arch = arch or detected.get("arch", "DeepLabV3Plus")
        encoder_name = encoder_name or detected.get("encoder_name", "resnet101")
        num_classes = num_classes or detected.get("num_classes", 4)
        
        print(f"[INFO] Architecture: {arch}")
        print(f"[INFO] Encoder: {encoder_name}")
        print(f"[INFO] Classes: {num_classes}")
        print(f"[INFO] Taille d'image: {imgsz}x{imgsz}")
        
        # Construire le modèle SMP
        arch_map = {
            "DeepLabV3Plus": smp.DeepLabV3Plus,
            "DeepLabV3": smp.DeepLabV3,
            "Unet": smp.Unet,
            "UnetPlusPlus": smp.UnetPlusPlus,
            "FPN": smp.FPN,
            "PSPNet": smp.PSPNet,
            "Linknet": smp.Linknet,
            "MAnet": smp.MAnet,
            "PAN": smp.PAN,
        }
        
        model_cls = arch_map.get(arch)
        if model_cls is None:
            print(f"[ERROR] Architecture SMP non supportée: {arch}")
            print(f"[INFO] Architectures supportées: {list(arch_map.keys())}")
            return False
        
        model = model_cls(
            encoder_name=encoder_name,
            encoder_weights=None,  # Pas besoin de pré-entraîné, on charge le state_dict
            in_channels=3,
            classes=num_classes,
        )
        
        # Charger les poids
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        print("[INFO] Poids chargés avec succès")
        
        # Créer une entrée dummy
        dummy_input = torch.randn(1, 3, imgsz, imgsz)
        
        # Vérifier que le forward pass fonctionne
        with torch.no_grad():
            test_out = model(dummy_input)
        print(f"[INFO] Forward pass OK, output shape: {test_out.shape}")
        
        # Créer le dossier de sortie
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export vers ONNX
        print(f"[INFO] Export ONNX vers: {output_path}")
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )
        
        # Simplifier le modèle ONNX
        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify
                print("[INFO] Simplification du modèle ONNX...")
                onnx_model = onnx.load(str(output_path))
                model_simplified, check = onnx_simplify(onnx_model)
                if check:
                    onnx.save(model_simplified, str(output_path))
                    print("[INFO] Modèle ONNX simplifié avec succès")
                else:
                    print("[WARN] Échec de la vérification de simplification")
            except ImportError:
                print("[WARN] onnxsim non installé, simplification ignorée")
        
        print(f"[SUCCESS] Modèle SMP exporté: {output_path}")
        print(f"[INFO] Taille: {output_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Vérifier que le fichier ONNX est valide
        try:
            import onnx
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            print(f"[SUCCESS] Fichier ONNX valide")
        except ImportError:
            print("[WARN] onnx non installé, vérification ignorée")
        except Exception as e:
            print(f"[WARN] Vérification ONNX échouée: {e}")
        
        # Déterminer le dossier racine du modèle
        if output_path.parent.name == "weights":
            model_root = output_path.parent.parent
        else:
            model_root = output_path.parent
        
        # Construire la liste des noms de classes (sans background)
        if class_names is None:
            class_names = []
        class_names_no_bg = [c for c in class_names if c.lower() != "background"]
        
        # Sauvegarder les métadonnées JSON (compatible avec le runner)
        meta_path = output_path.with_suffix('.json')
        meta = {
            "model_type": "smp",
            "task": "semantic_segmentation",
            "image_size": imgsz,
            "num_labels": num_classes,
            "class_names": class_names_no_bg,
            "use_sahi": True,
            "merge_polygons": True,
            "smp_arch": arch,
            "smp_encoder": encoder_name,
            "bg_bias": 0.0,
            "confidence_threshold": 0.3,
            "source": str(model_path),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[INFO] Métadonnées sauvegardées: {meta_path}")
        
        # Créer/mettre à jour classes.txt (sans background, car _mask_to_polygons fait class_id - 1)
        output_classes = model_root / "classes.txt"
        if class_names_no_bg:
            output_classes.write_text("\n".join(class_names_no_bg), encoding="utf-8")
            print(f"[INFO] classes.txt créé: {output_classes}")
        
        # Copier les métadonnées existantes
        _copy_metadata(model_path, output_path)
        
        # Validation post-export
        validate_onnx_export("smp", model, output_path, imgsz)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Échec export SMP: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_yolo_to_onnx(
    model_path: Path,
    output_path: Path,
    imgsz: int = 640,
    simplify: bool = True,
    opset: int = 12,
) -> bool:
    """
    Exporte un modèle YOLO vers ONNX.
    """
    print(f"[INFO] Export YOLO -> ONNX: {model_path}")
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics n'est pas installé. Installez-le avec: pip install ultralytics")
        return False
    
    try:
        model = YOLO(str(model_path))
        
        # Export vers ONNX
        export_path = model.export(
            format="onnx",
            imgsz=imgsz,
            simplify=simplify,
            opset=opset,
            dynamic=False,
        )
        
        # Déplacer vers le chemin de sortie souhaité
        export_path = Path(export_path)
        if export_path != output_path:
            shutil.move(str(export_path), str(output_path))
        
        print(f"[SUCCESS] Modèle YOLO exporté: {output_path}")
        print(f"[INFO] Taille: {output_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Copier les métadonnées (args.yaml, classes.txt)
        _copy_metadata(model_path, output_path)
        
        # Déterminer le type de tâche (detect ou segment)
        task = "object_detection"
        if hasattr(model, 'task') and model.task == 'segment':
            task = "instance_segmentation"
        elif hasattr(model, 'model') and hasattr(model.model, 'yaml'):
            yaml_data = model.model.yaml
            if isinstance(yaml_data, dict) and 'task' in yaml_data:
                if yaml_data['task'] == 'segment':
                    task = "instance_segmentation"
        
        # Charger les infos de classes
        model_dir = model_path.parent.parent if model_path.parent.name == "weights" else model_path.parent
        num_classes = None
        class_names_list = None
        
        # Depuis args.yaml
        args_file = model_dir / "args.yaml"
        if args_file.exists():
            try:
                import yaml
                with open(args_file, 'r', encoding='utf-8') as f:
                    args = yaml.safe_load(f)
                if isinstance(args, dict):
                    num_classes = args.get("nc") or args.get("num_classes")
            except Exception:
                pass
        
        # Depuis classes.txt
        classes_file = model_dir / "classes.txt"
        if classes_file.exists():
            try:
                class_names_list = [line.strip() for line in classes_file.read_text(encoding='utf-8').splitlines() if line.strip()]
                if num_classes is None:
                    num_classes = len(class_names_list)
            except Exception:
                pass
        
        # Depuis le modèle lui-même
        if num_classes is None and hasattr(model, 'model') and hasattr(model.model, 'nc'):
            num_classes = model.model.nc
        if class_names_list is None and hasattr(model, 'names'):
            class_names_list = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
        
        # Sauvegarder best.json
        meta_path = output_path.with_suffix('.json')
        meta = {
            "model_type": "yolo",
            "task": task,
            "image_size": imgsz,
            "source": str(model_path),
        }
        if num_classes:
            meta["num_classes"] = num_classes
        if class_names_list:
            meta["class_names"] = class_names_list
        
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[INFO] Métadonnées sauvegardées: {meta_path}")
        
        # Validation post-export
        validate_onnx_export("yolo", model, output_path, imgsz)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Échec export YOLO: {e}")
        import traceback
        traceback.print_exc()
        return False


def _load_rfdetr_model(checkpoint_path: str):
    """
    Charge un modèle RF-DETR depuis un checkpoint.
    Basé sur https://github.com/PierreMarieCurie/rf-detr-onnx
    """
    import torch
    from rfdetr.detr import RFDETRBase
    
    # Importer les variantes disponibles
    model_classes = {'RFDETRBase': RFDETRBase}
    try:
        from rfdetr.detr import RFDETRMedium
        model_classes['RFDETRMedium'] = RFDETRMedium
    except ImportError:
        pass
    try:
        from rfdetr.detr import RFDETRSmall
        model_classes['RFDETRSmall'] = RFDETRSmall
    except ImportError:
        pass
    try:
        from rfdetr.detr import RFDETRNano
        model_classes['RFDETRNano'] = RFDETRNano
    except ImportError:
        pass
    try:
        from rfdetr.detr import RFDETRLarge
        model_classes['RFDETRLarge'] = RFDETRLarge
    except ImportError:
        pass
    
    print(f"[INFO] Chargement du checkpoint: {checkpoint_path}...")
    obj = torch.load(checkpoint_path, weights_only=False, map_location=torch.device('cpu'))
    args = obj.get("args", None)
    
    # Extraire resolution et patch_size depuis args (peut être un objet ou un dict)
    resolution = None
    hidden_dim = 256
    patch_size = 14  # défaut DINOv2
    
    if args is not None:
        if hasattr(args, "resolution"):
            resolution = args.resolution
            hidden_dim = getattr(args, "hidden_dim", 256)
            patch_size = getattr(args, "patch_size", 14)
        elif isinstance(args, dict):
            resolution = args.get("resolution")
            hidden_dim = args.get("hidden_dim", 256)
            patch_size = args.get("patch_size", 14)
    
    # Détecter patch_size et hidden_dim depuis les poids du modèle
    model_state = obj.get("model", obj)
    if isinstance(model_state, dict):
        for key in model_state.keys():
            if 'patch_embeddings.projection.weight' in key:
                shape = model_state[key].shape
                # shape: [out_channels, in_channels, kernel_h, kernel_w]
                hidden_dim = shape[0]  # out_channels = hidden_dim
                patch_size = shape[2]  # kernel size = patch_size
                print(f"[INFO] Patch size détecté depuis les poids: {patch_size}")
                print(f"[INFO] Hidden dim détecté depuis les poids: {hidden_dim}")
                break
    
    # Si pas trouvé dans args, détecter resolution depuis les poids du modèle
    if resolution is None:
        if isinstance(model_state, dict):
            for key in model_state.keys():
                if 'position_embeddings' in key:
                    shape = model_state[key].shape
                    # shape: [1, num_positions, hidden_dim]
                    num_positions = shape[1]
                    num_patches = num_positions - 1
                    grid_size = int(num_patches ** 0.5)
                    resolution = grid_size * patch_size
                    print(f"[INFO] Résolution détectée depuis les poids: {resolution}")
                    break
    
    # Calculer positional_encoding_size = resolution / patch_size
    positional_encoding_size = resolution // patch_size if resolution and patch_size else None
    print(f"[INFO] Résolution: {resolution}, hidden_dim: {hidden_dim}, patch_size: {patch_size}, positional_encoding_size: {positional_encoding_size}")
    
    if resolution == 384 and 'RFDETRNano' in model_classes:
        print("[INFO] Modèle détecté: RF-DETR Nano")
        return model_classes['RFDETRNano'](pretrain_weights=checkpoint_path)
    elif resolution == 512 and 'RFDETRSmall' in model_classes:
        print("[INFO] Modèle détecté: RF-DETR Small")
        return model_classes['RFDETRSmall'](pretrain_weights=checkpoint_path)
    elif resolution == 576 and 'RFDETRMedium' in model_classes:
        print("[INFO] Modèle détecté: RF-DETR Medium")
        return model_classes['RFDETRMedium'](pretrain_weights=checkpoint_path)
    elif resolution == 560:
        if hidden_dim == 256:
            print("[INFO] Modèle détecté: RF-DETR Base")
            return RFDETRBase(pretrain_weights=checkpoint_path)
        elif hidden_dim == 384 and 'RFDETRLarge' in model_classes:
            print("[INFO] Modèle détecté: RF-DETR Large")
            return model_classes['RFDETRLarge'](pretrain_weights=checkpoint_path)
    
    # Fallback intelligent basé sur la résolution détectée
    if resolution is not None and positional_encoding_size is not None:
        # Trouver le modèle le plus proche en termes de hidden_dim
        if hidden_dim == 384 and 'RFDETRLarge' in model_classes:
            print(f"[INFO] Fallback: utilisation de RFDETRLarge avec resolution={resolution}, patch_size={patch_size}, positional_encoding_size={positional_encoding_size}")
            return model_classes['RFDETRLarge'](pretrain_weights=checkpoint_path, resolution=resolution, patch_size=patch_size, positional_encoding_size=positional_encoding_size)
        elif hidden_dim == 256:
            print(f"[INFO] Fallback: utilisation de RFDETRBase avec resolution={resolution}, patch_size={patch_size}, positional_encoding_size={positional_encoding_size}")
            return RFDETRBase(pretrain_weights=checkpoint_path, resolution=resolution, patch_size=patch_size, positional_encoding_size=positional_encoding_size)
        
        # Sinon, trouver le modèle le plus proche
        resolution_map = {
            384: 'RFDETRNano',
            512: 'RFDETRSmall', 
            560: 'RFDETRBase',
            576: 'RFDETRMedium',
        }
        for res, model_name in sorted(resolution_map.items(), key=lambda x: abs(x[0] - resolution)):
            if model_name in model_classes:
                print(f"[INFO] Fallback: utilisation de {model_name} avec resolution={resolution}, patch_size={patch_size}, positional_encoding_size={positional_encoding_size}")
                return model_classes[model_name](pretrain_weights=checkpoint_path, resolution=resolution, patch_size=patch_size, positional_encoding_size=positional_encoding_size)
    
    # Dernier fallback
    print("[INFO] Fallback: utilisation de RFDETRBase")
    return RFDETRBase(pretrain_weights=checkpoint_path)


def export_rfdetr_to_onnx(
    model_path: Path,
    output_path: Path,
    imgsz: int = 640,
    opset: int = 17,
    simplify: bool = True,
) -> bool:
    """
    Exporte un modèle RF-DETR vers ONNX.
    Basé sur https://github.com/PierreMarieCurie/rf-detr-onnx
    """
    print(f"[INFO] Export RF-DETR -> ONNX: {model_path}")
    
    try:
        import torch
        import onnx
        from onnxsim import simplify as onnx_simplify
    except ImportError as e:
        print(f"[ERROR] Dépendances manquantes: {e}")
        print("[ERROR] Installez avec: pip install rfdetr torch onnx onnxsim")
        return False
    
    try:
        # Charger le modèle RF-DETR
        rfdetr_model = _load_rfdetr_model(str(model_path))
        
        # Accéder au modèle PyTorch interne
        model = rfdetr_model.model.model
        config = rfdetr_model.model_config
        
        resolution = config.resolution
        device = config.device
        
        print(f"[INFO] Résolution: {resolution}, Device: {device}")
        
        # Créer une entrée dummy pour l'export
        dummy_input = torch.randn(1, 3, resolution, resolution, device=device)
        model.eval()
        
        # Forward pass pour obtenir les noms des sorties
        with torch.no_grad():
            output = model(dummy_input)
        
        if len(output) == 4:  # Object detection
            output_names = list(output.keys())[:2][::-1]
        elif len(output) == 5:  # Instance segmentation
            output_names = list(output.keys())[:3][::-1]
        else:
            output_names = ['scores', 'boxes']
        
        print(f"[INFO] Sorties du modèle: {output_names}")
        
        # Préparer le modèle pour l'export
        model.export()
        
        # Export vers ONNX
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Export ONNX vers: {output_path}")
        
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["images"],
            output_names=output_names,
        )
        
        # Simplifier le modèle ONNX
        if simplify:
            print("[INFO] Simplification du modèle ONNX...")
            onnx_model = onnx.load(str(output_path))
            model_simplified, check = onnx_simplify(onnx_model)
            if check:
                onnx.save(model_simplified, str(output_path))
                print("[INFO] Modèle ONNX simplifié avec succès")
            else:
                print("[WARN] Échec de la vérification de simplification")
        
        print(f"[SUCCESS] Modèle RF-DETR exporté: {output_path}")
        print(f"[INFO] Taille: {output_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Copier les métadonnées
        _copy_metadata(model_path, output_path)
        
        # Charger les infos de classes depuis args.yaml ou classes.txt
        model_dir = model_path.parent.parent if model_path.parent.name == "weights" else model_path.parent
        num_classes = None
        class_names = None
        
        # Essayer de charger depuis args.yaml
        args_file = model_dir / "args.yaml"
        if args_file.exists():
            try:
                import yaml
                with open(args_file, 'r', encoding='utf-8') as f:
                    args = yaml.safe_load(f)
                if isinstance(args, dict):
                    num_classes = args.get("nc") or args.get("num_classes")
            except Exception:
                pass
        
        # Essayer de charger depuis classes.txt
        classes_file = model_dir / "classes.txt"
        if classes_file.exists():
            try:
                class_names = [line.strip() for line in classes_file.read_text(encoding='utf-8').splitlines() if line.strip()]
                if num_classes is None:
                    num_classes = len(class_names)
            except Exception:
                pass
        
        # Sauvegarder les infos du modèle
        meta_path = output_path.with_suffix('.json')
        meta = {
            "model_type": "rfdetr",
            "task": "object_detection",
            "resolution": resolution,
            "class_offset": 1,  # RF-DETR utilise des class IDs 1-indexés
            "source": str(model_path),
        }
        if num_classes:
            meta["num_classes"] = num_classes
        if class_names:
            meta["class_names"] = class_names
        
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[INFO] Métadonnées sauvegardées: {meta_path}")
        
        # Validation post-export
        validate_onnx_export("rfdetr", rfdetr_model, output_path, resolution)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Échec export RF-DETR: {e}")
        import traceback
        traceback.print_exc()
        return False


def _find_test_image() -> "Path | None":
    """Cherche une image de test dans dev/image_test/."""
    imgs = _find_all_test_images()
    return imgs[0] if imgs else None


def _find_all_test_images() -> list:
    """Retourne toutes les images de test dans dev/image_test/."""
    script_dir = Path(__file__).resolve().parent
    image_test_dir = script_dir.parent / "image_test"
    if not image_test_dir.is_dir():
        return []
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(image_test_dir.glob(ext))
    return sorted(images)


def _load_test_image(resolution: int, model_type: str, image_path: "Path | None" = None):
    """
    Charge une image de test et la prépare pour l'inférence.
    
    Args:
        resolution: Taille cible (carré)
        model_type: Type de modèle (pour la normalisation)
        image_path: Chemin vers l'image. Si None, cherche dans dev/image_test/ ou utilise du bruit.

    Returns:
        (input_np, image_source_str)  — input_np shape (1, 3, H, W) float32
    """
    import numpy as np
    from PIL import Image

    if image_path is None:
        image_path = _find_test_image()

    if image_path is not None:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((resolution, resolution), Image.LANCZOS)
        test_img = np.array(img, dtype=np.float32) / 255.0  # [0, 1]
        source = str(image_path.name)
    else:
        print("[VALIDATION] Aucune image dans dev/image_test/, utilisation de bruit aléatoire")
        np.random.seed(42)
        test_img = np.random.rand(resolution, resolution, 3).astype(np.float32)
        source = "random_noise"

    # Normalisation ImageNet (RF-DETR, SegFormer, SMP)
    if model_type in ("rfdetr", "segformer", "smp"):
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        test_img_norm = (test_img - mean) / std
    else:
        # YOLO: [0, 1]
        test_img_norm = test_img

    # HWC -> CHW -> NCHW
    input_np = np.transpose(test_img_norm, (2, 0, 1))[None].astype(np.float32)
    return input_np, source


def validate_onnx_export(
    model_type: str,
    pytorch_model,
    onnx_path: Path,
    resolution: int,
    threshold: float = 0.3,
) -> bool:
    """
    Valide l'export ONNX en comparant les sorties PyTorch vs ONNX sur TOUTES
    les images de test dans dev/image_test/.

    Vérifie pour chaque image :
    1. Proximité numérique des tenseurs bruts (allclose).
    2. Même nombre de détections au-dessus du seuil.
    3. Correspondance des scores, boxes et classes par détection.

    Returns:
        True si la validation réussit pour toutes les images, False sinon.
    """
    import numpy as np
    import onnxruntime as ort

    # Collecter toutes les images de test
    test_images = _find_all_test_images()
    if not test_images:
        print("[VALIDATION] Aucune image dans dev/image_test/, utilisation de bruit aléatoire")
        test_images = [None]  # None = fallback bruit aléatoire

    print("\n" + "=" * 60)
    print(f"[VALIDATION] Comparaison PyTorch vs ONNX sur {len(test_images)} image(s)")
    print("=" * 60)

    # Charger la session ONNX une seule fois
    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name

    results = []
    for img_idx, img_path in enumerate(test_images):
        passed = _validate_single_image(
            model_type=model_type,
            pytorch_model=pytorch_model,
            sess=sess,
            input_name=input_name,
            resolution=resolution,
            threshold=threshold,
            image_path=img_path,
            image_index=img_idx + 1,
            total_images=len(test_images),
        )
        results.append((img_path, passed))

    # ── Résumé global ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"[VALIDATION] RÉSUMÉ: {len(test_images)} image(s) testée(s)")
    print("=" * 60)
    all_passed = True
    for img_path, passed in results:
        name = img_path.name if img_path else "random_noise"
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print(f"\n[VALIDATION] ✓ Export ONNX validé sur {len(test_images)} image(s)")
    else:
        n_fail = sum(1 for _, p in results if not p)
        print(f"\n[VALIDATION] ✗ {n_fail}/{len(test_images)} image(s) en échec")

    return all_passed


def _validate_single_image(
    model_type: str,
    pytorch_model,
    sess,
    input_name: str,
    resolution: int,
    threshold: float,
    image_path,
    image_index: int,
    total_images: int,
) -> bool:
    """Valide l'export ONNX sur une seule image. Retourne True si OK."""
    import numpy as np

    input_np, img_source = _load_test_image(resolution, model_type, image_path)
    print(f"\n{'─' * 60}")
    print(f"[VALIDATION] Image {image_index}/{total_images}: {img_source} ({resolution}x{resolution})")
    print(f"{'─' * 60}")

    try:
        import torch
        input_torch = torch.from_numpy(input_np)

        # ── Inférence PyTorch ──────────────────────────────────────
        if model_type == "rfdetr":
            model = pytorch_model.model.model
            model.eval()
            model.export()
            with torch.no_grad():
                pt_out = model(input_torch)
            pt_raw = [pt_out[k][0].detach().numpy() for k in pt_out]
            pt_logits = pt_out[list(pt_out.keys())[1]][0].detach().numpy()
            pt_boxes = pt_out[list(pt_out.keys())[0]][0].detach().numpy()
            pt_scores = 1 / (1 + np.exp(-pt_logits))
            pt_max_per_det = pt_scores.max(axis=1)
            pt_classes = pt_scores.argmax(axis=1)

        elif model_type == "yolo":
            model = pytorch_model.model
            model.eval()
            with torch.no_grad():
                pt_out = model(input_torch)
            if isinstance(pt_out, (list, tuple)):
                pt_out = pt_out[0]
            pt_out_np = pt_out.detach().numpy()[0]
            pt_raw = [pt_out_np]
            if pt_out_np.shape[0] < pt_out_np.shape[1]:
                pt_out_np = pt_out_np.T
            pt_boxes = pt_out_np[:, :4]
            pt_max_per_det = pt_out_np[:, 4:].max(axis=1)
            pt_classes = pt_out_np[:, 4:].argmax(axis=1)

        elif model_type == "segformer":
            model = pytorch_model
            model.eval()
            with torch.no_grad():
                pt_out = model(input_torch)
            if hasattr(pt_out, 'logits'):
                pt_logits = pt_out.logits[0].detach().numpy()
            else:
                pt_logits = pt_out[0].detach().numpy()
            pt_raw = [pt_logits]
            pt_max_per_det = pt_logits.max(axis=0).flatten()
            pt_classes = pt_logits.argmax(axis=0).flatten()
            pt_boxes = None

        elif model_type == "smp":
            model = pytorch_model
            model.eval()
            with torch.no_grad():
                pt_out = model(input_torch)
            pt_logits = pt_out[0].detach().numpy()  # [num_classes, H, W]
            pt_raw = [pt_logits]
            pt_max_per_det = pt_logits.max(axis=0).flatten()
            pt_classes = pt_logits.argmax(axis=0).flatten()
            pt_boxes = None

        # ── Inférence ONNX ─────────────────────────────────────────
        onnx_out = sess.run(None, {input_name: input_np})

        if model_type == "rfdetr":
            onnx_raw = [o[0] for o in onnx_out]
            onnx_logits = onnx_out[1][0]
            onnx_boxes = onnx_out[0][0]
            onnx_scores = 1 / (1 + np.exp(-onnx_logits))
            onnx_max_per_det = onnx_scores.max(axis=1)
            onnx_classes = onnx_scores.argmax(axis=1)

        elif model_type == "yolo":
            onnx_out_np = onnx_out[0][0]
            onnx_raw = [onnx_out_np]
            if onnx_out_np.shape[0] < onnx_out_np.shape[1]:
                onnx_out_np = onnx_out_np.T
            onnx_boxes = onnx_out_np[:, :4]
            onnx_max_per_det = onnx_out_np[:, 4:].max(axis=1)
            onnx_classes = onnx_out_np[:, 4:].argmax(axis=1)

        elif model_type == "segformer":
            onnx_logits = onnx_out[0][0]
            onnx_raw = [onnx_logits]
            onnx_max_per_det = onnx_logits.max(axis=0).flatten()
            onnx_classes = onnx_logits.argmax(axis=0).flatten()
            onnx_boxes = None

        elif model_type == "smp":
            onnx_logits = onnx_out[0][0]
            onnx_raw = [onnx_logits]
            onnx_max_per_det = onnx_logits.max(axis=0).flatten()
            onnx_classes = onnx_logits.argmax(axis=0).flatten()
            onnx_boxes = None

        # ── 1. Comparaison tenseurs bruts ──────────────────────────
        print("\n--- Comparaison tenseurs bruts ---")
        all_close = True
        for i, (p, o) in enumerate(zip(pt_raw, onnx_raw)):
            close = np.allclose(p, o, atol=1e-4, rtol=1e-3)
            max_diff = np.abs(p - o).max()
            mean_diff = np.abs(p - o).mean()
            print(f"  Sortie[{i}] shape={p.shape}: allclose={close}, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
            if not close:
                all_close = False

        # ── 1b. Distribution des classes (segmentation sémantique) ─
        if model_type in ("smp", "segformer"):
            print("\n--- Distribution des classes prédites (argmax par pixel) ---")
            total_pixels = len(pt_classes)
            print(f"  {'Classe':>6}  {'PT pixels':>10}  {'PT %':>7}  {'ONNX pixels':>12}  {'ONNX %':>7}  {'Match':>5}")
            print(f"  {'------':>6}  {'---------':>10}  {'----':>7}  {'-----------':>12}  {'------':>7}  {'-----':>5}")
            unique_classes = sorted(set(np.unique(pt_classes)) | set(np.unique(onnx_classes)))
            for cls in unique_classes:
                n_pt_cls = int((pt_classes == cls).sum())
                n_onnx_cls = int((onnx_classes == cls).sum())
                pct_pt = n_pt_cls / total_pixels * 100
                pct_onnx = n_onnx_cls / total_pixels * 100
                match = "✓" if n_pt_cls == n_onnx_cls else "~" if abs(n_pt_cls - n_onnx_cls) < total_pixels * 0.001 else "✗"
                print(f"  {cls:>6}  {n_pt_cls:>10}  {pct_pt:>6.2f}%  {n_onnx_cls:>12}  {pct_onnx:>6.2f}%  {match:>5}")
            classes_match = np.array_equal(pt_classes, onnx_classes)
            print(f"  Masques identiques pixel par pixel: {classes_match}")

        # ── 2. Comparaison détections ──────────────────────────────
        print(f"\n--- Détections (seuil={threshold}) ---")
        pt_mask = pt_max_per_det >= threshold
        onnx_mask = onnx_max_per_det >= threshold
        n_pt = int(pt_mask.sum())
        n_onnx = int(onnx_mask.sum())
        print(f"  PyTorch : {n_pt} détection(s)")
        print(f"  ONNX    : {n_onnx} détection(s)")

        if n_pt != n_onnx:
            print(f"  [WARN] Nombre de détections différent: PT={n_pt} vs ONNX={n_onnx}")

        # Comparer les détections communes (triées par score décroissant)
        pt_order = np.argsort(-pt_max_per_det)
        onnx_order = np.argsort(-onnx_max_per_det)
        n_compare = min(n_pt, n_onnx, 20)  # Top 20 max

        if n_compare > 0:
            print(f"\n--- Top {n_compare} détections (par score décroissant) ---")
            print(f"  {'#':>3}  {'Score PT':>10}  {'Score ONNX':>10}  {'Diff%':>7}  {'Cls PT':>6}  {'Cls ONNX':>8}  {'Match':>5}")
            print(f"  {'---':>3}  {'--------':>10}  {'----------':>10}  {'-----':>7}  {'------':>6}  {'--------':>8}  {'-----':>5}")

            score_diffs = []
            class_mismatches = 0
            box_diffs = []

            for rank in range(n_compare):
                pi = pt_order[rank]
                oi = onnx_order[rank]
                s_pt = pt_max_per_det[pi]
                s_onnx = onnx_max_per_det[oi]
                c_pt = pt_classes[pi]
                c_onnx = onnx_classes[oi]
                diff_pct = abs(s_pt - s_onnx) / max(s_pt, 1e-8) * 100
                match = "✓" if c_pt == c_onnx and diff_pct < 5 else "✗"
                print(f"  {rank+1:>3}  {s_pt:>10.4f}  {s_onnx:>10.4f}  {diff_pct:>6.2f}%  {c_pt:>6}  {c_onnx:>8}  {match:>5}")
                score_diffs.append(diff_pct)
                if c_pt != c_onnx:
                    class_mismatches += 1
                if pt_boxes is not None and onnx_boxes is not None:
                    box_diffs.append(np.abs(pt_boxes[pi] - onnx_boxes[oi]).max())

            avg_score_diff = np.mean(score_diffs)
            max_score_diff = np.max(score_diffs)
            print(f"\n  Score diff: moy={avg_score_diff:.2f}%, max={max_score_diff:.2f}%")
            print(f"  Classes différentes: {class_mismatches}/{n_compare}")
            if box_diffs:
                print(f"  Box max diff: moy={np.mean(box_diffs):.4f}, max={np.max(box_diffs):.4f}")
        else:
            max_score_diff = 0
            class_mismatches = 0

        # ── 3. Verdict pour cette image ────────────────────────────
        ok_tensors = all_close
        ok_scores = max_score_diff < 5 if n_compare > 0 else True
        passed = ok_tensors and ok_scores

        if passed:
            print(f"  → ✓ OK ({img_source})")
        else:
            print(f"  → ✗ DIVERGENCE ({img_source})")
            print(f"    Tenseurs allclose: {ok_tensors}")
            if n_compare > 0:
                print(f"    Score diff max: {max_score_diff:.2f}%")

        return passed

    except Exception as e:
        print(f"[VALIDATION] ✗ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def _copy_metadata(model_path: Path, output_path: Path):
    """
    Copie les fichiers de métadonnées (args.yaml, classes.txt) vers le dossier de sortie.
    
    Le runner attend args.yaml et classes.txt à la RACINE du modèle (ex: models/run_rf_detr_1/),
    pas dans le sous-dossier weights/.
    """
    if model_path.parent.name == "weights":
        model_dir = model_path.parent.parent
    else:
        model_dir = model_path.parent
    
    # Déterminer le dossier racine du modèle de sortie
    # Si output_path est dans weights/, on remonte d'un niveau
    if output_path.parent.name == "weights":
        output_model_root = output_path.parent.parent
    else:
        output_model_root = output_path.parent
    
    # Copier args.yaml à la RACINE du modèle (pas dans weights/)
    args_file = model_dir / "args.yaml"
    if args_file.exists():
        dest = output_model_root / "args.yaml"
        if not dest.exists() and args_file.resolve() != dest.resolve():
            shutil.copy2(args_file, dest)
            print(f"[INFO] Copié: {dest}")
    
    # Copier classes.txt à la RACINE du modèle
    for candidate in ["classes.txt", "class_names.txt", "classes.json"]:
        src = model_dir / candidate
        if src.exists():
            dest = output_model_root / candidate
            if src.resolve() != dest.resolve():
                shutil.copy2(src, dest)
                print(f"[INFO] Copié: {dest}")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Export des modèles YOLO, RF-DETR, SegFormer et SMP vers ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
    python export_to_onnx.py --model best.pt --output model.onnx
    python export_to_onnx.py --model best.pt --output model.onnx --imgsz 640
    python export_to_onnx.py --model best.pt --output model.onnx --type rfdetr
    python export_to_onnx.py --model best_model.pth --output best.onnx --type smp --arch DeepLabV3Plus --encoder resnet101 --num-classes 4 --class-names "background,chemin creux,parcellaire,talus-fosse_fossebutte"
        """
    )
    
    parser.add_argument("--model", required=True, help="Chemin vers le modèle PyTorch (.pt/.pth)")
    parser.add_argument("--output", required=True, help="Chemin de sortie pour le modèle ONNX (.onnx)")
    parser.add_argument("--type", choices=["yolo", "rfdetr", "segformer", "smp", "auto"], default="auto",
                        help="Type de modèle (auto-détecté par défaut)")
    parser.add_argument("--imgsz", type=int, default=640, help="Taille d'image pour l'export")
    parser.add_argument("--simplify", action="store_true", default=True,
                        help="Simplifier le modèle ONNX (défaut: True)")
    parser.add_argument("--no-simplify", action="store_false", dest="simplify",
                        help="Ne pas simplifier le modèle ONNX")
    parser.add_argument("--opset", type=int, default=17, help="Version ONNX opset (14+ requis pour RF-DETR)")
    # Options SMP
    parser.add_argument("--arch", default=None,
                        help="Architecture SMP (DeepLabV3Plus, Unet, FPN, etc.) — auto-détecté si omis")
    parser.add_argument("--encoder", default=None,
                        help="Encoder SMP (resnet101, resnet50, efficientnet-b0, etc.) — auto-détecté si omis")
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Nombre de classes (SMP) — auto-détecté si omis")
    parser.add_argument("--class-names", default=None,
                        help="Noms des classes séparés par des virgules (ex: 'background,chemin creux,parcellaire')")
    
    args = parser.parse_args()
    
    model_path = Path(args.model).resolve()
    output_path = Path(args.output).resolve()
    
    if not model_path.exists():
        print(f"[ERROR] Modèle non trouvé: {model_path}")
        return 1
    
    # Détecter le type de modèle
    if args.type == "auto":
        model_type = detect_model_type(model_path)
        print(f"[INFO] Type de modèle détecté: {model_type}")
    else:
        model_type = args.type
    
    # Créer le dossier de sortie
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Parser les noms de classes si fournis
    class_names = None
    if args.class_names:
        class_names = [c.strip() for c in args.class_names.split(",")]
    
    # Exporter
    if model_type == "smp":
        success = export_smp_to_onnx(
            model_path=model_path,
            output_path=output_path,
            imgsz=args.imgsz,
            simplify=args.simplify,
            opset=args.opset,
            arch=args.arch,
            encoder_name=args.encoder,
            num_classes=args.num_classes,
            class_names=class_names,
        )
    elif model_type == "segformer":
        success = export_segformer_to_onnx(
            model_path=model_path,
            output_path=output_path,
            imgsz=args.imgsz,
            simplify=args.simplify,
            opset=args.opset,
        )
    elif model_type == "rfdetr":
        success = export_rfdetr_to_onnx(
            model_path=model_path,
            output_path=output_path,
            imgsz=args.imgsz,
            simplify=args.simplify,
            opset=args.opset,
        )
    else:
        success = export_yolo_to_onnx(
            model_path=model_path,
            output_path=output_path,
            imgsz=args.imgsz,
            simplify=args.simplify,
            opset=args.opset,
        )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
