import json
from pathlib import Path
from typing import Any, Dict


class ConfigManager:
    def __init__(self, plugin_root: Path, filename: str = "config.json"):
        self.plugin_root = plugin_root
        self.path = plugin_root / filename

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
                "pyramids": {
                    "enabled": True,
                    "levels": [2, 4, 8, 16, 32, 64],
                },
                "products": {
                    "MNT": True,
                    "DENSITE": False,
                    "M_HS": False,
                    "SVF": False,
                    "SLO": False,
                    "LD": False,
                    "VAT": False,
                },
            },
            "computer_vision": {
                "enabled": False,
                "selected_model": "",
                "target_rvt": "LD",
                "confidence_threshold": 0.3,
                "iou_threshold": 0.5,
                "generate_annotated_images": False,
                "generate_shapefiles": False,
                "models_dir": "models",
                "sahi": {
                    "slice_height": 640,
                    "slice_width": 640,
                    "overlap_ratio": 0.2,
                },
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
                "vat": {
                    "terrain_type": 0,
                    "save_as_8bit": True,
                },
            },
        }

    def load(self) -> Dict[str, Any]:
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
        return cfg

    def save(self, config: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def _deep_update(self, base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in other.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                self._deep_update(base[k], v)
            else:
                base[k] = v
        return base
