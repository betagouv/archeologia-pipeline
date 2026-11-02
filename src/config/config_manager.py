import copy
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
        # Bibliothèque de configurations nommées (menu « Charger une config »).
        # Même dossier que la session pour la même raison : une mise à jour du
        # plugin ne doit pas emporter les configs de l'utilisateur.
        self.configs_dir = base / "configs"
        #: Messages de la dernière conversion pixels → mètres (vide si rien à convertir).
        self.migrations: list = []
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
                    "nom_analyse": "",   # vide = proposé d'après le dossier de sortie (app.services.nom_analyse)
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
                "coverage_threshold_percent": 30,
                "filter_expression": "Classification = 2 OR Classification = 6 OR Classification = 66 OR Classification = 67 OR Classification = 9",
                "max_workers": 3,
                "products": {
                    "MNT": True,
                    "DENSITE": False,
                    "COUVERTURE": False,
                    "HS": False,
                    "M_HS": False,
                    "SVF": True,
                    "OPNS": False,
                    "SLO": False,
                    "LD": True,
                    "SLRM": False,
                    "VAT": False,
                    "MSTP": False,
                    "CVAT": False,
                    "PRISM": False,
                    "CRIM": False,
                },
                "output_formats": {
                    "jpg": {
                        "HS": False,
                        "M_HS": False,
                        "SVF": False,
                        "OPNS": False,
                        "SLO": False,
                        "LD": False,
                        "VAT": False,
                        "MSTP": False,
                        "CVAT": False,
                        "PRISM": False,
                        "CRIM": False,
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
                "target_rvt": "LD",
                "confidence_threshold": 0.3,  # = pipeline.cv.model_config.DEFAULT_CONFIDENCE (unifié 2026-08-31)
                "iou_threshold": 0.5,
                "generate_annotated_images": False,
                "models_dir": "data/models",
                "export_runner_config": False,
                # Détection sur la carte graphique (DirectML, Windows) : option de
                # l'étape 4, décochée par défaut (0.15.0) — sonde par poste, repli visible.
                "carte_graphique": True,   # cochée par défaut depuis le 2026-10-26 (décision de l'utilisateur)
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
                # Rayons et échelles en MÈTRES (clés ``*_m``, règle 2026-10-31) : convertis en pixels à la
                # résolution réelle du MNT par app.services.indices_metres.en_px. Défauts = anciens défauts
                # en pixels × 0,5 m (calibrés sur l'IGN) : la France à 0,5 m reste au bit près.
                "svf": {
                    "noise_remove": 0,
                    "num_directions": 16,
                    "radius_m": 5.0,
                    "ve_factor": 1,
                    "save_as_8bit": True,
                },
                "opns": {
                    # Directions 8-64 (bornes de rvt:rvt_opns, rvt_naming.opns_settings) ; rayon en mètres.
                    "opns_type": 0,
                    "noise_remove": 0,
                    "num_directions": 16,
                    "radius_m": 5.0,
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
                    "min_radius_m": 5.0,
                    "max_radius_m": 10.0,
                    "observer_h": 1.7,
                    "ve_factor": 1,
                    "save_as_8bit": True,
                },
                "slrm": {
                    "radius_m": 10.0,
                    "ve_factor": 1,
                    "save_as_8bit": True,
                },
                "vat": {
                    "terrain_type": 0,
                    "save_as_8bit": True,
                },
                "prism": {
                    "terrain_type": 0,
                    "save_as_8bit": True,
                },
                "crim": {
                    # Colormap matplotlib appliquée à la PENTE. "OrRd" est le
                    # défaut de rvt.blend.color_relief_image_map : fond sable,
                    # pentes en rouge. C'est le rendu qui donne à CRIM sa
                    # signature, changer la colormap change toute l'image.
                    "colormap": "OrRd",
                    "min_colormap_cut": 0.0,
                    "max_colormap_cut": 1.0,
                    "save_as_8bit": True,
                },
                "mstp": {
                    "local_scale_min_m": 1.5,
                    "local_scale_max_m": 10.5,
                    "local_scale_step_m": 1.0,
                    "meso_scale_min_m": 11.5,
                    "meso_scale_max_m": 101.5,
                    "meso_scale_step_m": 9.0,
                    # Échelle large calibrée sur le tuilage du plugin, PAS sur le
                    # défaut RVT (223/2023/180 px). Un rayon de 2023 px demande
                    # 2023 px de contexte de chaque côté ; la marge par défaut
                    # (tile_overlap 20 % à 0,5 m) n'en fournit que 400 → RVT
                    # fabriquerait le voisinage manquant par symétrie et les
                    # dalles ne se raccorderaient plus. Cf.
                    # app.services.rvt_kernel_context, qui vérifie la relation
                    # à chaque run et signale les autres géométries.
                    "broad_scale_min_m": 50.0,
                    "broad_scale_max_m": 200.0,
                    "broad_scale_step_m": 30.0,
                    "lightness": 0.9,
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

        return self.fusionner(data)

    def fusionner(self, charge: Any) -> Dict[str, Any]:
        """Défauts + configuration chargée, normalisée : point de passage UNIQUE de toute fusion (session,
        config.json, configuration nommée, import, relance depuis un résultat). Les rayons encore en pixels
        sont convertis en mètres AVANT la fusion (les défauts portent déjà les clés en mètres, qui
        masqueraient la valeur enregistrée — audit 2026-10-31) ; les messages restent dans
        ``self.migrations``. ``charge`` n'est pas modifié."""
        data = copy.deepcopy(charge) if isinstance(charge, dict) else {}
        self._migrate_rvt_metres(data)
        cfg = self.default_config()
        self._deep_update(cfg, data)
        self._migrate_entity_ids(cfg)
        self._migrate_cv_runs(cfg)
        # Clé mono-modèle d'avant runs[], migrée ci-dessus puis retirée : plus rien ne l'écrit, mais une vieille
        # session la portait (…/run_rf_detr_1, modèle retiré) jusque dans la trace des runs (2026-10-31).
        if isinstance(cfg.get("computer_vision"), dict):
            cfg["computer_vision"].pop("selected_model", None)
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

        return self.fusionner(data)

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

    def _migrate_rvt_metres(self, cfg: Dict[str, Any]) -> None:
        """Rayons des indices enregistrés en pixels → mètres (0,5 m/px supposé), une seule fois
        (app.services.indices_metres.migrer_px_vers_m). Les messages restent dans ``self.migrations``
        pour que l'interface les dise."""
        try:
            from ..app.services.indices_metres import migrer_px_vers_m
        except ImportError:  # pragma: no cover - tests (src/ sur le chemin)
            from app.services.indices_metres import migrer_px_vers_m
        self.migrations = migrer_px_vers_m(cfg.get("rvt_params")) if isinstance(cfg, dict) else []

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
            cv.pop("selected_model", None)   # mono-modèle d'avant runs[] (cf. fusionner)
            # 0.15.0 : sans effet depuis l'assistant (le run analyse toujours toutes les images et convertit
            # lui-même ses détections) — constat 20 de l'audit des réglages.
            cv.pop("scan_all", None)
            cv.pop("generate_shapefiles", None)
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
