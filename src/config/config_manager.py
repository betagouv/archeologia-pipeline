import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigManager:
    def __init__(
        self,
        plugin_root: Path,
        filename: str = "config.json",
        settings_dir: Optional[Path] = None,
    ):
        self.plugin_root = plugin_root
        self.path = plugin_root / filename
        # AUDIT v2 CFG-02 : les réglages utilisateur vivent HORS du dossier du
        # plugin (remplacé à chaque mise à jour par ZIP → tout était perdu)
        # quand l'hôte fournit un dossier de profil (côté QGIS :
        # QgsApplication.qgisSettingsDirPath()/archeologia). L'ancien
        # emplacement est migré automatiquement au premier lancement.
        base = Path(settings_dir) if settings_dir is not None else plugin_root
        self.last_ui_path = base / "last_ui_config.json"
        if settings_dir is not None:
            self._migrate_legacy_last_ui()

    def _migrate_legacy_last_ui(self) -> None:
        legacy = self.plugin_root / "last_ui_config.json"
        try:
            if legacy.exists() and not self.last_ui_path.exists():
                self.last_ui_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(legacy), str(self.last_ui_path))
        except Exception:
            pass  # best-effort : au pire, on repart des défauts

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
                "max_workers": 3,
                "products": {
                    "MNT": False,
                    "DENSITE": False,
                    "HS": False,
                    "M_HS": False,
                    "SVF": False,
                    "SLO": False,
                    "LD": False,
                    "SLRM": False,
                    "VAT": False,
                    "MSTP": False,
                    "CVAT": False,
                },
                "output_formats": {
                    "jpg": {
                        "HS": False,
                        "M_HS": False,
                        "SVF": False,
                        "SLO": False,
                        "LD": False,
                        "VAT": False,
                        "MSTP": False,
                        "CVAT": False,
                    }
                },
            },
            "computer_vision": {
                "enabled": False,
                "runs": [],
                "selected_entities": [],
                "entity_model_overrides": {},
                "entity_cluster_enabled": [],
                "entity_thresholds": {},
                "entity_cluster_params": {},
                "selected_model": "",
                "target_rvt": "LD",
                "confidence_threshold": 0.3,  # = pipeline.cv.model_config.DEFAULT_CONFIDENCE (unifié 2026-08-31)
                "iou_threshold": 0.5,
                "generate_annotated_images": False,
                "generate_shapefiles": False,
                "models_dir": "data/models",
                "export_runner_config": False,
                "scan_all": True,
            },
            "rvt_params": {
                "hs": {
                    "sun_azimuth": 315,
                    "sun_elevation": 35,
                    "ve_factor": 1,
                    "save_as_8bit": True,
                },
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
                "mstp": {
                    "local_scale_min": 3,
                    "local_scale_max": 21,
                    "local_scale_step": 2,
                    "meso_scale_min": 23,
                    "meso_scale_max": 203,
                    "meso_scale_step": 18,
                    # Échelle large calibrée sur le tuilage du plugin, PAS sur le
                    # défaut RVT (223/2023/180). Un rayon de 2023 px demande
                    # 2023 px de contexte de chaque côté ; la marge par défaut
                    # (tile_overlap 20 % à 0,5 m) n'en fournit que 400 → RVT
                    # fabriquerait le voisinage manquant par symétrie et les
                    # dalles ne se raccorderaient plus. Cf.
                    # app.services.rvt_kernel_context, qui vérifie la relation
                    # à chaque run et signale les autres géométries.
                    "broad_scale_min": 100,
                    "broad_scale_max": 400,
                    "broad_scale_step": 60,
                    "lightness": 1.2,
                    "ve_factor": 1,
                    "save_as_8bit": True,
                },
                "cvat": {
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
        self._migrate_entity_ids(cfg)
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
        self._migrate_entity_ids(cfg)
        self._migrate_cv_runs(cfg)
        return cfg

    def save_last_ui_config(self, config: Dict[str, Any]) -> None:
        """Sauvegarde la dernière config UI (last_ui_config.json).

        Écriture ATOMIQUE (tmp + os.replace) : un crash pendant l'autosave
        n'efface plus toute la configuration (AUDIT v2 UIX-07/CFG-06).
        """
        self._strip_deprecated_keys(config)
        self.last_ui_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.last_ui_path.with_name(self.last_ui_path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.last_ui_path)

    # Renommages d'identifiants d'entités (cf. refonte morphologique étape 3) :
    # une sélection sauvegardée avec un ancien id deviendrait un fantôme (droppée
    # silencieusement + case sans carte). On remappe à la lecture.
    _ENTITY_ID_RENAMES = {
        "cratere_obus": "cratere",
        "zones_extraction_materiaux": "regroupement_crateres",
    }

    @classmethod
    def _migrate_entity_ids(cls, cfg: Dict[str, Any]) -> None:
        """Remappe les anciens ids d'entités/classes dans la section CV.

        Couvre ``selected_entities`` / ``entity_cluster_enabled`` (listes),
        ``entity_model_overrides`` / ``entity_thresholds`` / ``entity_cluster_params``
        (dicts indexés par entité), et ``runs[].selected_classes`` +
        ``runs[].entities[].{id,classes}``. ``zone_crateres`` (sortie de
        clustering) n'est pas concerné. Tolérant aux structures malformées.
        """
        cv = cfg.get("computer_vision")
        if not isinstance(cv, dict):
            return
        rename = cls._ENTITY_ID_RENAMES

        def _remap(value: str) -> str:
            return rename.get(value, value)

        for key in ("selected_entities", "entity_cluster_enabled"):
            v = cv.get(key)
            if isinstance(v, list):
                cv[key] = [_remap(str(x)) for x in v]
        for key in ("entity_model_overrides", "entity_thresholds", "entity_cluster_params"):
            v = cv.get(key)
            if isinstance(v, dict):
                cv[key] = {_remap(str(k)): val for k, val in v.items()}
        runs = cv.get("runs")
        if isinstance(runs, list):
            for run in runs:
                if not isinstance(run, dict):
                    continue
                sc = run.get("selected_classes")
                if isinstance(sc, list):
                    run["selected_classes"] = [_remap(str(c)) for c in sc]
                for ent in run.get("entities") or []:
                    if not isinstance(ent, dict):
                        continue
                    if "id" in ent:
                        ent["id"] = _remap(str(ent["id"]))
                    if isinstance(ent.get("classes"), list):
                        ent["classes"] = [_remap(str(c)) for c in ent["classes"]]

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
        # V2 : le toggle Simple/Expert est supprimé.
        ui = cfg.get("ui")
        if isinstance(ui, dict):
            ui.pop("display_mode", None)

    def _deep_update(self, base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in other.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                self._deep_update(base[k], v)
            else:
                base[k] = v
        return base
