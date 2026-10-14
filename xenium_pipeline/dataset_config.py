"""Dataset registry used by wrappers, notebooks, and cross-dataset reviews."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("XENIUM_ROOT", CODE_ROOT)).expanduser().resolve()
MANIFEST_PATH = PROJECT_ROOT / "dataset_manifest.json"
if not MANIFEST_PATH.is_file():
    MANIFEST_PATH = CODE_ROOT / "dataset_manifest.json"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    display_name: str
    root: str
    output_dir: str
    pipeline: str
    run_script: str
    entry_point: str | None
    description: str
    sample_discovery: str
    input_layout: str
    he_alignment: str
    cell_labels: str
    selective_archive_extract: bool
    require_alignment: bool | None
    require_cell_groups: bool
    missing_cell_label: str
    cell_groups_label_source: str
    cells_output_suffix: str
    completion_marker_suffix: str
    skip_completed_before_validation: bool
    tile_size_px: int
    min_cells_per_tile: int
    min_cell_area_fraction: float
    preview_count: int
    pyramid_level: int
    annotation_alpha: int
    transform_dir: str | None = None
    notes: tuple[str, ...] = ()
    sample_image_metadata: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "DatasetConfig":
        values = dict(payload)
        values["notes"] = tuple(values.get("notes", ()))
        values["sample_image_metadata"] = tuple(values.get("sample_image_metadata", ()))
        allowed = {field.name for field in fields(cls)}
        values = {key: value for key, value in values.items() if key in allowed}
        return cls(**values)

    def root_path(self, project_root: str | Path | None = None) -> Path:
        return Path(project_root or PROJECT_ROOT) / self.root

    def script_path(self) -> Path:
        """Resolve checked-in helpers independently of the server data registry."""
        if self.pipeline == "standard_xenium":
            return CODE_ROOT / "xenium_pipeline" / "standard_xenium.py"
        return CODE_ROOT / "xenium_pipeline" / "bilous_tiles.py"

    def output_path(self, project_root: str | Path | None = None) -> Path:
        return Path(project_root or PROJECT_ROOT) / self.output_dir

    def transform_path(self, project_root: str | Path | None = None) -> Path | None:
        if self.transform_dir is None:
            return None
        return Path(project_root or PROJECT_ROOT) / self.transform_dir

    def standard_default_options(self) -> dict[str, Any]:
        """Return CLI/run defaults shared by standard 10x-style datasets."""
        return {
            "tile_size": self.tile_size_px,
            "min_cells_per_tile": self.min_cells_per_tile,
            "min_cell_area_fraction": self.min_cell_area_fraction,
            "preview_count": self.preview_count,
            "pyramid_level": self.pyramid_level,
            "annotation_alpha": self.annotation_alpha,
        }

    def standard_behavior_kwargs(self) -> dict[str, Any]:
        """Return dataset-specific behavior switches for standard_xenium."""
        return {
            "selective_archive_extract": self.selective_archive_extract,
            "require_alignment": self.require_alignment,
            "require_cell_groups": self.require_cell_groups,
            "missing_cell_label": self.missing_cell_label,
            "cell_groups_label_source": self.cell_groups_label_source,
            "cells_output_suffix": self.cells_output_suffix,
            "completion_marker_suffix": self.completion_marker_suffix,
            "skip_completed_before_validation": self.skip_completed_before_validation,
        }

    def standard_run_kwargs(self) -> dict[str, Any]:
        return {
            **self.standard_default_options(),
            **self.standard_behavior_kwargs(),
        }

    def validation_kwargs(self) -> dict[str, Any]:
        return {
            "selective_archive_extract": self.selective_archive_extract,
            "require_alignment": self.require_alignment,
            "require_cell_groups": self.require_cell_groups,
        }

    def tile_filter_output_entry(self) -> Path:
        return Path(self.output_dir)


@lru_cache(maxsize=None)
def _load_dataset_manifest_cached(manifest_path: str) -> dict[str, Any]:
    path = Path(manifest_path)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset_manifest(path: str | Path | None = None) -> dict[str, Any]:
    manifest_path = Path(path or MANIFEST_PATH).resolve()
    return _load_dataset_manifest_cached(str(manifest_path))


def dataset_configs(path: str | Path | None = None) -> dict[str, DatasetConfig]:
    manifest = load_dataset_manifest(path)
    return {
        config.name: config
        for config in (
            DatasetConfig.from_mapping(item)
            for item in manifest.get("datasets", [])
        )
    }


def get_dataset_config(name: str) -> DatasetConfig:
    configs = dataset_configs()
    try:
        return configs[name]
    except KeyError as error:
        available = ", ".join(sorted(configs))
        raise KeyError(f"Unknown dataset {name!r}. Available datasets: {available}") from error


def dataset_output_dirs(*, pipelines: set[str] | None = None) -> dict[str, Path]:
    configs = dataset_configs()
    return {
        name: config.tile_filter_output_entry()
        for name, config in configs.items()
        if pipelines is None or config.pipeline in pipelines
    }
