from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

LogFn = Callable[[str], None]
CancelCheckFn = Callable[[], bool]


class ComputerVisionService:
    def __init__(
        self,
        cv_config: Dict[str, Any],
        output_dir: Path,
        log: LogFn = lambda _: None,
        cancel_check: Optional[CancelCheckFn] = None,
    ):
        self._config = cv_config or {}
        self._output_dir = output_dir
        self._log = log
        self._cancel_check = cancel_check
        self._tif_transform_data: Dict[str, Tuple[float, float, float, float]] = {}
        self._labels_dir: Optional[Path] = None
        self._shp_dir: Optional[Path] = None

    @property
    def enabled(self) -> bool:
        return bool(self._config.get("enabled", False))

    @property
    def target_rvt(self) -> str:
        return str(self._config.get("target_rvt", "LD"))

    @property
    def generate_shapefiles(self) -> bool:
        return bool(self._config.get("generate_shapefiles", False))

    def should_process_product(self, product_name: str) -> bool:
        if not self.enabled:
            return False
        return product_name.upper() == self.target_rvt.upper()

    def process_single_jpg(
        self,
        jpg_path: Path,
        rvt_base_dir: Path,
        tif_transform_data: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    ) -> None:
        if not self.enabled:
            return

        if tif_transform_data:
            self._tif_transform_data.update(tif_transform_data)

        jpg_dir = jpg_path.parent
        if self._labels_dir is None:
            self._labels_dir = jpg_dir
        if self._shp_dir is None:
            self._shp_dir = rvt_base_dir / "shapefiles"

        try:
            from ...pipeline.cv.runner import run_cv_on_folder

            run_cv_on_folder(
                jpg_dir=jpg_dir,
                cv_config=self._config,
                target_rvt=self.target_rvt,
                rvt_base_dir=rvt_base_dir,
                tif_transform_data=self._tif_transform_data,
                single_jpg=jpg_path,
                run_shapefile_dedup=False,
                log=self._log,
                cancel_check=self._cancel_check,
            )
        except Exception as e:
            self._log(f"Erreur Computer Vision: {e}")

    def process_folder(
        self,
        jpg_dir: Path,
        rvt_base_dir: Path,
        tif_transform_data: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    ) -> None:
        if not self.enabled:
            return

        if tif_transform_data:
            self._tif_transform_data.update(tif_transform_data)

        if self._labels_dir is None:
            self._labels_dir = jpg_dir
        if self._shp_dir is None:
            self._shp_dir = rvt_base_dir / "shapefiles"

        try:
            from ...pipeline.cv.runner import run_cv_on_folder

            run_cv_on_folder(
                jpg_dir=jpg_dir,
                cv_config=self._config,
                target_rvt=self.target_rvt,
                rvt_base_dir=rvt_base_dir,
                tif_transform_data=self._tif_transform_data,
                single_jpg=None,
                run_shapefile_dedup=False,
                log=self._log,
                cancel_check=self._cancel_check,
            )
        except Exception as e:
            self._log(f"Erreur Computer Vision: {e}")

    def finalize(self, temp_dir: Optional[Path] = None, crs: str = "EPSG:2154") -> None:
        if not self.enabled:
            return
        if not self.generate_shapefiles:
            return
        if self._labels_dir is None or self._shp_dir is None:
            return

        try:
            from ...pipeline.cv.runner import deduplicate_cv_shapefiles_final

            deduplicate_cv_shapefiles_final(
                labels_dir=self._labels_dir,
                shp_dir=self._shp_dir,
                target_rvt=self.target_rvt,
                cv_config=self._config,
                tif_transform_data=self._tif_transform_data,
                temp_dir=temp_dir or (self._output_dir / "temp"),
                crs=crs,
                log=self._log,
            )
        except Exception as e:
            self._log(f"Erreur déduplication shapefiles CV: {e}")

    def reset(self) -> None:
        self._tif_transform_data.clear()
        self._labels_dir = None
        self._shp_dir = None
