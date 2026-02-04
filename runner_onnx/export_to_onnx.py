"""
Script d'export des modèles YOLO et RF-DETR vers le format ONNX.

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
    Détecte le type de modèle (yolo ou rfdetr) à partir du fichier.
    """
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
            # RF-DETR a encoder/decoder/backbone
            if any('encoder' in str(k) or 'decoder' in str(k) for k in checkpoint.keys()):
                return "rfdetr"
    except Exception:
        pass
    
    # Par défaut, essayer YOLO
    return "yolo"


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
    obj = torch.load(checkpoint_path, weights_only=False)
    args = obj.get("args", None)
    
    # Extraire resolution depuis args (peut être un objet ou un dict)
    resolution = None
    hidden_dim = 256
    
    if args is not None:
        if hasattr(args, "resolution"):
            resolution = args.resolution
            hidden_dim = getattr(args, "hidden_dim", 256)
        elif isinstance(args, dict):
            resolution = args.get("resolution")
            hidden_dim = args.get("hidden_dim", 256)
    
    # Si pas trouvé dans args, détecter depuis les poids du modèle
    if resolution is None:
        model_state = obj.get("model", obj)
        if isinstance(model_state, dict):
            for key in model_state.keys():
                if 'position_embeddings' in key:
                    shape = model_state[key].shape
                    # shape: [1, num_positions, hidden_dim]
                    num_positions = shape[1]
                    num_patches = num_positions - 1
                    # Essayer patch_size 16 d'abord
                    grid_size = int(num_patches ** 0.5)
                    resolution = grid_size * 16
                    print(f"[INFO] Résolution détectée depuis les poids: {resolution}")
                    break
    
    print(f"[INFO] Résolution: {resolution}, hidden_dim: {hidden_dim}")
    
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
    if resolution is not None:
        # Trouver le modèle le plus proche
        resolution_map = {
            384: 'RFDETRNano',
            512: 'RFDETRSmall', 
            560: 'RFDETRBase',
            576: 'RFDETRMedium',
        }
        for res, model_name in sorted(resolution_map.items(), key=lambda x: abs(x[0] - resolution)):
            if model_name in model_classes:
                print(f"[INFO] Fallback: utilisation de {model_name} (résolution proche)")
                return model_classes[model_name](pretrain_weights=checkpoint_path)
    
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
        
        # Sauvegarder les infos du modèle
        meta_path = output_path.with_suffix('.json')
        meta = {
            "model_type": "rfdetr",
            "resolution": resolution,
            "source": str(model_path),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"[INFO] Métadonnées sauvegardées: {meta_path}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Échec export RF-DETR: {e}")
        import traceback
        traceback.print_exc()
        return False


def _copy_metadata(model_path: Path, output_path: Path):
    """
    Copie les fichiers de métadonnées (args.yaml, classes.txt) vers le dossier de sortie.
    """
    if model_path.parent.name == "weights":
        model_dir = model_path.parent.parent
    else:
        model_dir = model_path.parent
    
    output_dir = output_path.parent
    
    # Copier args.yaml
    args_file = model_dir / "args.yaml"
    if args_file.exists():
        dest = output_dir / "args.yaml"
        shutil.copy2(args_file, dest)
        print(f"[INFO] Copié: {dest}")
    
    # Copier classes.txt ou équivalent
    for candidate in ["classes.txt", "class_names.txt", "classes.json"]:
        src = model_dir / candidate
        if src.exists():
            dest = output_dir / candidate
            shutil.copy2(src, dest)
            print(f"[INFO] Copié: {dest}")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Export des modèles YOLO et RF-DETR vers ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
    python export_to_onnx.py --model best.pt --output model.onnx
    python export_to_onnx.py --model best.pt --output model.onnx --imgsz 640
    python export_to_onnx.py --model best.pt --output model.onnx --type rfdetr
        """
    )
    
    parser.add_argument("--model", required=True, help="Chemin vers le modèle PyTorch (.pt)")
    parser.add_argument("--output", required=True, help="Chemin de sortie pour le modèle ONNX (.onnx)")
    parser.add_argument("--type", choices=["yolo", "rfdetr", "auto"], default="auto",
                        help="Type de modèle (auto-détecté par défaut)")
    parser.add_argument("--imgsz", type=int, default=640, help="Taille d'image pour l'export")
    parser.add_argument("--simplify", action="store_true", default=True,
                        help="Simplifier le modèle ONNX (défaut: True)")
    parser.add_argument("--no-simplify", action="store_false", dest="simplify",
                        help="Ne pas simplifier le modèle ONNX")
    parser.add_argument("--opset", type=int, default=17, help="Version ONNX opset (14+ requis pour RF-DETR)")
    
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
    
    # Exporter
    if model_type == "rfdetr":
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
