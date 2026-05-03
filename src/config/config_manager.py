import json
from pathlib import Path
from typing import Any, Dict


class ConfigManager:
    def __init__(self, plugin_root: Path, filename: str = "config.json"):
        self.plugin_root = plugin_root
        self.path = plugin_root / filename
        self.last_ui_path = plugin_root / "last_ui_config.json"

    def default_config(self) -> Dict[str, Any]:
        return {
            "app": {
                "files": {
                    "output_dir": "",
                    "data_mode": "ign_laz",
                    "input_file": "",
                    "local_laz_dir": "",
                    "existing_mnt_dir": "",
                    "existing_rvt_dir": "",
                }
            },
            "processing": {
                "mnt_resolution": 0.5,
                "density_resolution": 1.0,
                "tile_overlap": 20,
                "filter_expression": "Classification = 2 OR Classification = 6 OR Classification = 66 OR Classification = 67 OR Classification = 9",
                "max_workers": 4,
                "products": {
                    "MNT": True,
                    "DENSITE": False,
                    "M_HS": False,
                    "SVF": False,
                    "SLO": False,
                    "LD": False,
                    "SLRM": False,
                    "VAT": False,
                },
                "output_formats": {
                    "jpg": {
                        "M_HS": False,
                        "SVF": False,
                        "SLO": False,
                        "LD": False,
                        "VAT": False,
                    }
                },
            },
            "computer_vision": {
                "enabled": False,
                "runs": [],
                "selected_model": "",
                "target_rvt": "LD",
                "confidence_threshold": 0.3,
                "iou_threshold": 0.5,
                "generate_annotated_images": False,
                "generate_shapefiles": False,
                "models_dir": "data/models",
                "export_runner_config": False,
                "scan_all": True,
            },
            "rvt_params": {
                "mdh": {
                    "num_directions": 16,
                    "sun_elevation": 35,
                    "ve_factor": 1,
                    "save_as_8bit": True,
                },
                "svf": {
                    "noise_remove": 0,
                    "num_directions": 16,
                    "radius": 10,
                    "ve_factor": 1,
                    "save_as_8bit": True,
                },
                "slope": {
                    "unit": 0,
                    "ve_factor": 1,
                    "save_as_8bit": True,
                },
                "ldo": {
                    "angular_res": 15,
                    "min_radius": 10,
                    "max_radius": 20,
                    "observer_h": 1.7,
                    "ve_factor": 1,
                    "save_as_8bit": True,
                },
                "slrm": {
                    "radius": 20,
                    "ve_factor": 1,
                    "save_as_8bit": True,
                },
                "vat": {
                    "terrain_type": 0,
                    "save_as_8bit": True,
                },
            },
        }

    def load(self) -> Dict[str, Any]:
        """Charge la config par défaut (config.json) fusionnée avec le fichier sur disque."""
        if not self.path.exists():
            cfg = self.default_config()
            self.save(cfg)
            return cfg

        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

        cfg = self.default_config()
        self._deep_update(cfg, data)
        self._migrate_cv_runs(cfg)
        return cfg

    def save(self, config: Dict[str, Any]) -> None:
        """Sauvegarde config.json (valeurs par défaut)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def load_last_ui_config(self) -> Dict[str, Any]:
        """Charge la dernière config UI (last_ui_config.json).

        Retourne les défauts si le fichier n'existe pas.
        """
        if not self.last_ui_path.exists():
            return self.default_config()

        try:
            with self.last_ui_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return self.default_config()

        cfg = self.default_config()
        self._deep_update(cfg, data)
        self._migrate_cv_runs(cfg)
        return cfg

    def save_last_ui_config(self, config: Dict[str, Any]) -> None:
        """Sauvegarde la dernière config UI (last_ui_config.json)."""
        self._strip_deprecated_keys(config)
        self.last_ui_path.parent.mkdir(parents=True, exist_ok=True)
        with self.last_ui_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _migrate_cv_runs(cfg: Dict[str, Any]) -> None:
        """Migre l'ancien format mono-modèle vers le nouveau format 'runs'."""
        cv = cfg.get("computer_vision")
        if not isinstance(cv, dict):
            return
        runs = cv.get("runs")
        if isinstance(runs, list) and runs:
            return  # Déjà migré
        # Migration: selected_model + target_rvt -> runs[0]
        model = str(cv.get("selected_model") or "").strip()
        rvt = str(cv.get("target_rvt") or "LD").strip()
        if model:
            cv["runs"] = [{"model": model, "target_rvt": rvt}]
        else:
            cv["runs"] = []

    @staticmethod
    def _strip_deprecated_keys(cfg: Dict[str, Any]) -> None:
        """Supprime les clés dépréciées / legacy de la configuration."""
        _legacy_root = {
            "mode", "data_mode", "source_path", "output_dir", "products",
            "detection_enabled", "mnt_resolution", "density_resolution",
            "tile_overlap", "max_workers", "filter_expression",
            "det_confidence", "det_iou", "det_generate_annotated", "det_generate_shp",
        }
        for k in _legacy_root:
            cfg.pop(k, None)
        cv = cfg.get("computer_vision")
        if isinstance(cv, dict):
            cv.pop("sahi", None)
            cv.pop("selected_classes", None)

    def _deep_update(self, base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in other.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                self._deep_update(base[k], v)
            else:
                base[k] = v
        return base
