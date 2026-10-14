#!/usr/bin/env python3
"""Prepare Bilous 2026 H&E alignment helpers and legacy tile wrappers.

Tile creation should use ``xenium_pipeline.standard_xenium`` with the
``tile_input_manifest.csv`` written during H&E preparation.  This module remains
as a Bilous-specific prepare-stage helper for mapping validation, manual
alignment QC, pre-crop tile previews, and manifest generation.

The pipeline starts from the manually verified H&E-to-sample mapping and the
selected manual landmark transform for each sample.  Xenium centroids
(and, when available, cell boundaries) are transformed from micrometers into
the local full-resolution H&E candidate coordinate system.  Only complete H&E
tiles containing at least ``min_cells_per_tile`` cells are written.

Outputs are kept under ``tile_outputs/<sample_folder>`` by default so that raw
GEO downloads and alignment-review artifacts remain untouched.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
import gc
import gzip
import json
import os
import math
from pathlib import Path
import sys
import traceback
from typing import Any, Iterable

CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("XENIUM_ROOT", CODE_ROOT)).expanduser().resolve()
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
from xenium_pipeline.tile_utils import xenium_explorer_imagealignment_matrix
import pandas as pd
from PIL import Image, ImageDraw
import tifffile
from tqdm.auto import tqdm
import zarr

from xenium_pipeline.dataset_config import get_dataset_config
from xenium_pipeline import standard_xenium
from xenium_pipeline.tile_utils import (
    apply_affine,
    create_color_map,
    ensure_rgb,
    extract_rgb_tile,
    get_hw,
    resolve_zarr_array,
)
from xenium_pipeline.bilous_image_alignment import (
    extracted_he_path,
    load_he_candidate,
    load_region_lookup,
    source_path,
)


DATASET_CONFIG = get_dataset_config("Bilous_2026")
DEFAULT_OUTPUT_SUBDIR = Path(DATASET_CONFIG.output_dir).name


@dataclass(frozen=True)
class PipelineConfig:
    tile_size_px: int = 1024
    min_cells_per_tile: int = 1
    preview_count: int = 12
    annotation_alpha: int = 210
    overwrite: bool = False
    make_qc_plots: bool = True
    recreate_qc_plots: bool = False
    label_column: str | None = None


@dataclass(frozen=True)
class SampleInput:
    root: Path
    sample_folder: str
    sample_label: str
    gsm: str
    sample_directory: Path
    mapping: dict[str, Any]
    transform_path: Path
    he_path: Path
    candidate_width: int
    candidate_height: int
    source_origin_x: int
    source_origin_y: int
    xenium_um_to_he_local_px: np.ndarray


@dataclass
class BoundaryData:
    polygons_by_cell_id: dict[str, np.ndarray]


TILE_FILTER_QUADRANTS = (
    "low_count_low_area",
    "high_count_low_area",
    "low_count_high_area",
    "high_count_high_area",
)

TILE_FILTER_QUADRANT_COLORS = {
    "low_count_low_area": "#8C8C8C",
    "high_count_low_area": "#F28E2B",
    "low_count_high_area": "#4E79A7",
    "high_count_high_area": "#59A14F",
}


def parse_args() -> argparse.Namespace:
    default_root = DATASET_CONFIG.root_path(PROJECT_ROOT)
    parser = argparse.ArgumentParser(
        description=DATASET_CONFIG.description
    )
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--samples", nargs="+", help="Optional sample folder names")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tile-size", type=int, default=DATASET_CONFIG.tile_size_px)
    parser.add_argument(
        "--min-cells-per-tile",
        type=int,
        default=DATASET_CONFIG.min_cells_per_tile,
    )
    parser.add_argument("--preview-count", type=int, default=DATASET_CONFIG.preview_count)
    parser.add_argument(
        "--annotation-alpha",
        type=int,
        default=DATASET_CONFIG.annotation_alpha,
    )
    parser.add_argument("--label-column")
    parser.add_argument(
        "--transform-dir",
        type=Path,
        help=(
            "Optional directory containing selected transform JSON files. "
            "Defaults to manual_alignment_from_mapping."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reprocess-completed", action="store_true")
    parser.add_argument("--skip-qc-plots", action="store_true")
    parser.add_argument(
        "--recreate-qc-plots",
        action="store_true",
        help=(
            "Regenerate QC plots from existing tile manifests and PNGs without "
            "cropping or rewriting tile PNGs."
        ),
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def read_mapping_table(root: Path) -> pd.DataFrame:
    path = root / "mapping_previews" / "he_sample_mapping.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Mapping table not found: {path}")
    mapping = pd.read_csv(path, dtype=str).fillna("")
    required = {
        "sample_folder",
        "sample_label",
        "gsm",
        "source_file",
        "candidate_type",
        "candidate_index",
    }
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Mapping table is missing columns: {sorted(missing)}")
    if mapping["sample_folder"].duplicated().any():
        duplicates = mapping.loc[
            mapping["sample_folder"].duplicated(keep=False), "sample_folder"
        ].tolist()
        raise ValueError(f"Duplicate mapping rows: {duplicates}")
    return mapping


def selected_transform_path(
    root: Path,
    sample_folder: str,
    transform_dir: Path | None = None,
) -> Path:
    """Find the transform JSON for a sample.

    By default this uses the manually selected landmark transforms.
    Passing ``transform_dir`` lets the same tile pipeline consume matrices
    exported from Xenium Explorer or QuPath/Warpy after they have been converted
    into the small JSON format expected here.
    """
    directory = (
        Path(transform_dir).expanduser().resolve()
        if transform_dir
        else root / "manual_alignment_from_mapping"
    )
    candidates = [
        directory / f"{sample_folder}_manual_alignment.json",
        directory / f"{sample_folder}_selected_transform.json",
        directory / f"{sample_folder}_external_alignment_transform.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(directory.glob(f"{sample_folder}*transform*.json"))
    return matches[0] if matches else candidates[0]


def locate_sample_directory(root: Path, sample_folder: str, gsm: str) -> Path:
    for candidate in (root / sample_folder, root / gsm):
        if candidate.is_dir():
            return candidate
    patterns = (f"{gsm}_*_cells.parquet", f"{gsm}_*_cells.parquet.gz")
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(root.glob(f"*/{pattern}"))
    parents = sorted({path.parent for path in matches})
    if len(parents) == 1:
        return parents[0]
    if not parents:
        raise FileNotFoundError(
            f"No sample directory or {gsm} cells parquet was found for {sample_folder}"
        )
    raise ValueError(f"Multiple possible sample directories for {sample_folder}: {parents}")


def _read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read ordinary parquet or a GEO file wrapped in an outer gzip stream."""
    if path.name.endswith(".parquet.gz"):
        with gzip.open(path, "rb") as handle:
            return pd.read_parquet(BytesIO(handle.read()), columns=columns)
    return pd.read_parquet(path, columns=columns)


def find_sample_file(sample: SampleInput, suffix: str) -> Path | None:
    uncompressed = sorted(sample.sample_directory.glob(f"{sample.gsm}_*_{suffix}"))
    compressed = sorted(sample.sample_directory.glob(f"{sample.gsm}_*_{suffix}.gz"))
    return next(iter(uncompressed + compressed), None)


def _find_sample_file_in_directory(
    sample_directory: Path,
    gsm: str,
    suffix: str,
) -> Path | None:
    uncompressed = sorted(sample_directory.glob(f"{gsm}_*_{suffix}"))
    compressed = sorted(sample_directory.glob(f"{gsm}_*_{suffix}.gz"))
    return next(iter(uncompressed + compressed), None)


def _resolve_selected_matrix(payload: dict[str, Any]) -> np.ndarray:
    selected = payload.get("selected_transform", payload)
    key = "matrix_xenium_um_to_he_candidate_local_full_pixel"
    if key not in selected:
        raise KeyError(f"Selected transform does not contain {key!r}")
    matrix = np.asarray(selected[key], dtype=float)
    if matrix.shape != (3, 3) or not np.allclose(matrix[2], [0, 0, 1]):
        raise ValueError(f"Invalid Xenium-to-H&E matrix: shape={matrix.shape}")
    return matrix


def build_sample_input(
    root: Path,
    row: pd.Series,
    region_lookup: dict[tuple[str, int], dict[str, int]],
    *,
    preview_dimension: int = 2000,
    transform_dir: Path | None = None,
) -> SampleInput:
    sample_folder = str(row["sample_folder"])
    gsm = str(row["gsm"])
    transform_path = selected_transform_path(root, sample_folder, transform_dir)
    if not transform_path.is_file():
        raise FileNotFoundError(f"Selected transform not found: {transform_path}")
    payload = json.loads(transform_path.read_text(encoding="utf-8"))
    matrix = _resolve_selected_matrix(payload)
    mapping = {name: str(value) for name, value in row.to_dict().items()}
    he = load_he_candidate(root, mapping, region_lookup, preview_dimension)
    he_path = Path(he["source"])
    if not he_path.is_file():
        raise FileNotFoundError(f"Extracted full-resolution H&E not found: {he_path}")
    return SampleInput(
        root=root,
        sample_folder=sample_folder,
        sample_label=str(row["sample_label"]),
        gsm=gsm,
        sample_directory=locate_sample_directory(root, sample_folder, gsm),
        mapping=mapping,
        transform_path=transform_path,
        he_path=he_path,
        candidate_width=int(he["full_width"]),
        candidate_height=int(he["full_height"]),
        source_origin_x=int(he["origin_x_in_source"]),
        source_origin_y=int(he["origin_y_in_source"]),
        xenium_um_to_he_local_px=matrix,
    )


def discover_inputs(
    root: str | Path,
    sample_names: Iterable[str] | None = None,
    *,
    transform_dir: str | Path | None = None,
) -> tuple[list[SampleInput], list[dict[str, Any]]]:
    root = Path(root).expanduser().resolve()
    mapping = read_mapping_table(root)
    requested = set(sample_names or [])
    if requested:
        known = set(mapping["sample_folder"])
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown sample names: {sorted(unknown)}")
        mapping = mapping.loc[mapping["sample_folder"].isin(requested)]
    region_path = root / "mapping_previews" / "ndpi_candidate_regions.csv"
    region_lookup = load_region_lookup(region_path) if region_path.is_file() else {}
    inputs: list[SampleInput] = []
    validation: list[dict[str, Any]] = []
    for _, row in mapping.iterrows():
        name = str(row["sample_folder"])
        try:
            sample = build_sample_input(
                root,
                row,
                region_lookup,
                transform_dir=Path(transform_dir) if transform_dir else None,
            )
            cells = find_sample_file(sample, "cells.parquet")
            if cells is None:
                raise FileNotFoundError(f"No cells parquet found in {sample.sample_directory}")
            inputs.append(sample)
            validation.append(
                {
                    "sample": name,
                    "status": "valid",
                    "gsm": sample.gsm,
                    "candidate_type": sample.mapping["candidate_type"],
                    "candidate_index": int(sample.mapping["candidate_index"]),
                    "candidate_width_px": sample.candidate_width,
                    "candidate_height_px": sample.candidate_height,
                    "he_path": str(sample.he_path),
                    "cells_path": str(cells),
                    "transform_path": str(sample.transform_path),
                    "error": "",
                }
            )
        except Exception as error:
            validation.append(
                {
                    "sample": name,
                    "status": "invalid",
                    "gsm": str(row["gsm"]),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return inputs, validation


def validate_all_inputs(
    root: str | Path,
    sample_names: Iterable[str] | None = None,
    *,
    transform_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Return a notebook-friendly validation table without writing any tiles.

    This is intentionally path-based and lightweight.  Tile creation still calls
    ``discover_inputs`` so it can build full ``SampleInput`` objects, but the
    notebook validation cell should not open large H&E images just to report
    whether inputs are ready.
    """
    root = Path(root).expanduser().resolve()
    transform_root = Path(transform_dir).expanduser().resolve() if transform_dir else None
    mapping = read_mapping_table(root)
    requested = set(sample_names or [])
    if requested:
        known = set(mapping["sample_folder"])
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown sample names: {sorted(unknown)}")
        mapping = mapping.loc[mapping["sample_folder"].isin(requested)]
    region_path = root / "mapping_previews" / "ndpi_candidate_regions.csv"
    region_lookup = load_region_lookup(region_path) if region_path.is_file() else {}

    records: list[dict[str, Any]] = []
    for _, row in mapping.iterrows():
        name = str(row["sample_folder"])
        gsm = str(row["gsm"])
        try:
            mapping_record = {field: str(value) for field, value in row.to_dict().items()}
            transform_path = selected_transform_path(root, name, transform_root)
            if not transform_path.is_file():
                raise FileNotFoundError(f"Selected transform not found: {transform_path}")
            payload = json.loads(transform_path.read_text(encoding="utf-8"))
            _resolve_selected_matrix(payload)

            he_path = extracted_he_path(root, mapping_record)
            if he_path is None:
                he_path = source_path(root, mapping_record["source_file"])
            if not he_path.is_file():
                raise FileNotFoundError(f"Mapped H&E source not found: {he_path}")

            sample_directory = locate_sample_directory(root, name, gsm)
            cells = _find_sample_file_in_directory(sample_directory, gsm, "cells.parquet")
            if cells is None:
                raise FileNotFoundError(f"No cells parquet found in {sample_directory}")

            candidate_width = ""
            candidate_height = ""
            if mapping_record["candidate_type"] != "scene":
                region_key = (mapping_record["source_file"], int(mapping_record["candidate_index"]))
                if region_key in region_lookup:
                    box = region_lookup[region_key]
                    candidate_width = box["fullres_x1"] - box["fullres_x0"]
                    candidate_height = box["fullres_y1"] - box["fullres_y0"]

            records.append(
                {
                    "sample": name,
                    "status": "valid",
                    "gsm": gsm,
                    "candidate_type": mapping_record["candidate_type"],
                    "candidate_index": int(mapping_record["candidate_index"]),
                    "candidate_width_px": candidate_width,
                    "candidate_height_px": candidate_height,
                    "he_path": str(he_path),
                    "cells_path": str(cells),
                    "transform_path": str(transform_path),
                    "error": "",
                }
            )
        except Exception as error:
            records.append(
                {
                    "sample": name,
                    "status": "invalid",
                    "gsm": gsm,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return pd.DataFrame(records)


def validate_mapping_inputs(
    root: str | Path,
    sample_names: Iterable[str] | None = None,
    *,
    transform_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Validate mapped H&E and Xenium cell inputs before manual alignment.

    This intentionally does not require a selected transform JSON. Use
    ``validate_all_inputs`` once manual alignment files have been created.
    """
    root = Path(root).expanduser().resolve()
    mapping = read_mapping_table(root)
    requested = set(sample_names or [])
    if requested:
        known = set(mapping["sample_folder"])
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown sample names: {sorted(unknown)}")
        mapping = mapping.loc[mapping["sample_folder"].isin(requested)]
    records: list[dict[str, Any]] = []
    for _, row in mapping.iterrows():
        name = str(row["sample_folder"])
        gsm = str(row["gsm"])
        try:
            mapping_record = {field: str(value) for field, value in row.to_dict().items()}
            he_path = extracted_he_path(root, mapping_record)
            if he_path is None:
                he_path = source_path(root, mapping_record["source_file"])
            if not he_path.is_file():
                raise FileNotFoundError(f"Mapped H&E source not found: {he_path}")
            sample_directory = locate_sample_directory(root, name, gsm)
            cells = _find_sample_file_in_directory(
                sample_directory,
                gsm,
                "cells.parquet",
            )
            if cells is None:
                raise FileNotFoundError(f"No cells parquet found in {sample_directory}")
            records.append(
                {
                    "sample": name,
                    "status": "valid",
                    "gsm": gsm,
                    "candidate_type": mapping_record["candidate_type"],
                    "candidate_index": int(mapping_record["candidate_index"]),
                    "candidate_width_px": "",
                    "candidate_height_px": "",
                    "he_path": str(he_path),
                    "cells_path": str(cells),
                    "expected_transform_path": str(
                        selected_transform_path(
                            root,
                            name,
                            Path(transform_dir) if transform_dir else None,
                        )
                    ),
                    "error": "",
                }
            )
        except Exception as error:
            records.append(
                {
                    "sample": name,
                    "status": "invalid",
                    "gsm": gsm,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return pd.DataFrame(records)


def _pixel_size_from_transform(transform_path: Path) -> float | None:
    payload = json.loads(transform_path.read_text(encoding="utf-8"))
    selected = payload.get("selected_transform", payload)
    values = selected.get("xenium_pixel_size_um_xy") or payload.get("xenium_pixel_size_um_xy")
    if values:
        x_value = float(values[0])
        y_value = float(values[1]) if len(values) > 1 else x_value
        if not np.isclose(x_value, y_value):
            raise ValueError(f"Non-square Xenium pixel size in {transform_path}: {values}")
        return x_value
    return None


def write_tile_input_manifest(
    root: str | Path,
    sample_names: Iterable[str] | None = None,
    *,
    output_dir: str | Path | None = None,
    transform_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    """Write resolved Bilous tile inputs for the shared standard pipeline."""
    root = Path(root).expanduser().resolve()
    output_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else DATASET_CONFIG.output_path(PROJECT_ROOT)
    )
    transform_root = Path(transform_dir).expanduser().resolve() if transform_dir else None
    mapping = read_mapping_table(root)
    requested = set(sample_names or [])
    if requested:
        known = set(mapping["sample_folder"])
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown sample names: {sorted(unknown)}")
        mapping = mapping.loc[mapping["sample_folder"].isin(requested)]
    records: list[dict[str, object]] = []
    for _, row in mapping.iterrows():
        sample_folder = str(row["sample_folder"])
        gsm = str(row["gsm"])
        mapping_record = {field: str(value) for field, value in row.to_dict().items()}
        he_path = extracted_he_path(root, mapping_record)
        if he_path is None:
            raise FileNotFoundError(
                "Resolved full-resolution H&E OME-TIFF not found for "
                f"{sample_folder}. Run H&E extraction before writing the tile manifest."
            )
        sample_directory = locate_sample_directory(root, sample_folder, gsm)
        cells = _find_sample_file_in_directory(sample_directory, gsm, "cells.parquet")
        if cells is None:
            raise FileNotFoundError(f"No cells parquet found in {sample_directory}")
        transform_path = selected_transform_path(root, sample_folder, transform_root)
        if not transform_path.is_file():
            raise FileNotFoundError(f"Selected transform not found: {transform_path}")
        alignment_csv = transform_path.parent / f"{sample_folder}_he_imagealignment.csv"
        if not alignment_csv.is_file():
            region_path = root / "mapping_previews" / "ndpi_candidate_regions.csv"
            sample = build_sample_input(
                root,
                row,
                load_region_lookup(region_path) if region_path.is_file() else {},
                transform_dir=transform_root,
            )
            alignment_csv = ensure_transform_imagealignment_csv(sample) or alignment_csv
        cell_boundaries = _find_sample_file_in_directory(
            sample_directory,
            gsm,
            "cell_boundaries.parquet",
        )
        nucleus_boundaries = _find_sample_file_in_directory(
            sample_directory,
            gsm,
            "nucleus_boundaries.parquet",
        )
        pixel_size_um = _pixel_size_from_transform(transform_path)
        records.append(
            {
                "sample": sample_folder,
                "output_dir": str(output_root / sample_folder),
                "sample_dir": str(sample_directory),
                "he_image": str(he_path),
                "cells": str(cells),
                "outs_dir": "",
                "outs_zip": "",
                "alignment_csv": str(alignment_csv),
                "manual_alignment_json": str(transform_path),
                "pixel_size_um": pixel_size_um if pixel_size_um is not None else "",
                "cell_groups": "",
                "cell_boundaries": str(cell_boundaries or ""),
                "nucleus_boundaries": str(nucleus_boundaries or ""),
            }
        )
    manifest = pd.DataFrame(records)
    destination = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path
        else output_root / "tile_input_manifest.csv"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(destination, index=False)
    return manifest


def load_cells(sample: SampleInput, label_column: str | None) -> tuple[pd.DataFrame, str]:
    path = find_sample_file(sample, "cells.parquet")
    if path is None:
        raise FileNotFoundError(f"No cells parquet found for {sample.sample_folder}")
    cells = _read_parquet(path)
    cells = cells.rename(
        columns={
            "cell_centroid_x": "x_centroid",
            "cell_centroid_y": "y_centroid",
            "nucleus_centroid_x": "nucleus_x_centroid",
            "nucleus_centroid_y": "nucleus_y_centroid",
            "barcode": "cell_id",
        }
    )
    required = {"x_centroid", "y_centroid"}
    missing = required - set(cells.columns)
    if missing:
        raise ValueError(f"Cells parquet is missing columns: {sorted(missing)}")
    if "cell_id" not in cells:
        cells.insert(0, "cell_id", np.arange(len(cells)).astype(str))
    cells["cell_id"] = cells["cell_id"].astype(str)
    cells.insert(0, "cell_index", np.arange(len(cells), dtype=np.int64))

    label_source = "none"
    labels: pd.Series | None = None
    if label_column:
        if label_column not in cells:
            raise KeyError(
                f"Requested label column {label_column!r} is absent. "
                f"Available columns: {list(cells.columns)}"
            )
        labels = cells[label_column]
        label_source = f"cells.parquet:{label_column}"
    else:
        custom_candidates = sorted(
            sample.sample_directory.glob(f"{sample.gsm}_*cell_groups.csv")
        ) + sorted(sample.sample_directory.glob("*cell_annotations.csv"))
        if custom_candidates:
            custom = pd.read_csv(custom_candidates[0])
            id_column = next(
                (name for name in ("cell_id", "barcode") if name in custom), None
            )
            group_column = next(
                (
                    name
                    for name in ("cell_type", "group", "annotation", "cluster")
                    if name in custom
                ),
                None,
            )
            if id_column and group_column:
                custom[id_column] = custom[id_column].astype(str)
                lookup = custom.drop_duplicates(id_column).set_index(id_column)[group_column]
                labels = cells["cell_id"].map(lookup)
                label_source = f"{custom_candidates[0].name}:{group_column}"
        if labels is None:
            automatic = next(
                (
                    name
                    for name in ("cell_type", "annotation", "group", "cluster")
                    if name in cells
                ),
                None,
            )
            if automatic:
                labels = cells[automatic]
                label_source = f"cells.parquet:{automatic}"
    if labels is None:
        labels = pd.Series("Unassigned", index=cells.index)
    cells["cell_type"] = labels.fillna("Unassigned").astype(str)

    xy_he = apply_affine(
        cells[["x_centroid", "y_centroid"]].to_numpy(),
        sample.xenium_um_to_he_local_px,
    )
    cells["he_x_px"] = xy_he[:, 0]
    cells["he_y_px"] = xy_he[:, 1]
    cells["inside_he_candidate"] = (
        cells["he_x_px"].between(0, sample.candidate_width, inclusive="left")
        & cells["he_y_px"].between(0, sample.candidate_height, inclusive="left")
    )
    if {"nucleus_x_centroid", "nucleus_y_centroid"}.issubset(cells.columns):
        nucleus_xy = cells[["nucleus_x_centroid", "nucleus_y_centroid"]].to_numpy(dtype=float)
        nucleus_he = np.full((len(cells), 2), np.nan, dtype=float)
        valid_nuclei = np.isfinite(nucleus_xy).all(axis=1)
        if valid_nuclei.any():
            nucleus_he[valid_nuclei] = apply_affine(
                nucleus_xy[valid_nuclei],
                sample.xenium_um_to_he_local_px,
            )
        cells["nucleus_he_x_px"] = nucleus_he[:, 0]
        cells["nucleus_he_y_px"] = nucleus_he[:, 1]
        cells["nucleus_inside_he_candidate"] = (
            cells["nucleus_he_x_px"].between(0, sample.candidate_width, inclusive="left")
            & cells["nucleus_he_y_px"].between(0, sample.candidate_height, inclusive="left")
        )
    return cells, label_source


def load_boundaries(sample: SampleInput, boundary_kind: str = "cell") -> BoundaryData | None:
    if boundary_kind not in {"cell", "nucleus"}:
        raise ValueError("boundary_kind must be 'cell' or 'nucleus'")
    path = find_sample_file(sample, f"{boundary_kind}_boundaries.parquet")
    if path is None:
        return None
    boundaries = _read_parquet(path)
    boundaries = boundaries.rename(
        columns={
            "barcode": "cell_id",
            "x": "vertex_x",
            "y": "vertex_y",
            "x_location": "vertex_x",
            "y_location": "vertex_y",
        }
    )
    required = {"cell_id", "vertex_x", "vertex_y"}
    if not required.issubset(boundaries.columns):
        print(
            f"[{sample.sample_folder}] {boundary_kind} boundary columns not recognized: "
            f"{list(boundaries.columns)}; using centroids only"
        )
        return None
    boundaries["cell_id"] = boundaries["cell_id"].astype(str)
    transformed = apply_affine(
        boundaries[["vertex_x", "vertex_y"]].to_numpy(),
        sample.xenium_um_to_he_local_px,
    )
    boundaries["he_x_px"] = transformed[:, 0]
    boundaries["he_y_px"] = transformed[:, 1]
    polygons = {
        str(cell_id): group[["he_x_px", "he_y_px"]].to_numpy()
        for cell_id, group in boundaries.groupby("cell_id", sort=False)
        if len(group) >= 3
    }
    return BoundaryData(polygons)


def marker_he_coordinates(cells: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
    """Return nucleus H&E coordinates when available, otherwise cell centroids."""
    cell_x = pd.to_numeric(cells["he_x_px"], errors="coerce")
    cell_y = pd.to_numeric(cells["he_y_px"], errors="coerce")
    if {"nucleus_he_x_px", "nucleus_he_y_px"}.issubset(cells.columns):
        nucleus_x = pd.to_numeric(cells["nucleus_he_x_px"], errors="coerce")
        nucleus_y = pd.to_numeric(cells["nucleus_he_y_px"], errors="coerce")
        valid = nucleus_x.notna() & nucleus_y.notna()
        if valid.any():
            return cell_x.where(~valid, nucleus_x), cell_y.where(~valid, nucleus_y), "Nucleus center"
    return cell_x, cell_y, "Cell centroid"


def marker_he_coordinates_from_record(record: object) -> tuple[float, float]:
    nucleus_x = float(getattr(record, "nucleus_he_x_px", np.nan))
    nucleus_y = float(getattr(record, "nucleus_he_y_px", np.nan))
    if np.isfinite(nucleus_x) and np.isfinite(nucleus_y):
        return nucleus_x, nucleus_y
    return float(getattr(record, "he_x_px")), float(getattr(record, "he_y_px"))


def create_tile_labels(
    sample: SampleInput,
    cells: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_columns = sample.candidate_width // config.tile_size_px
    n_rows = sample.candidate_height // config.tile_size_px
    if n_columns <= 0 or n_rows <= 0:
        raise ValueError("H&E candidate is smaller than one complete tile")
    grid_rows, grid_cols = np.indices((n_rows, n_columns))
    tile_metrics = pd.DataFrame(
        {
            "tile_row": grid_rows.ravel(),
            "tile_col": grid_cols.ravel(),
        }
    )
    tile_metrics.insert(0, "sample", sample.sample_folder)
    tile_metrics["x0_px"] = tile_metrics["tile_col"] * config.tile_size_px
    tile_metrics["y0_px"] = tile_metrics["tile_row"] * config.tile_size_px
    tile_metrics["x1_px"] = tile_metrics["x0_px"] + config.tile_size_px
    tile_metrics["y1_px"] = tile_metrics["y0_px"] + config.tile_size_px
    tile_metrics["source_x0_px"] = tile_metrics["x0_px"] + sample.source_origin_x
    tile_metrics["source_y0_px"] = tile_metrics["y0_px"] + sample.source_origin_y
    tile_metrics["source_x1_px"] = tile_metrics["x1_px"] + sample.source_origin_x
    tile_metrics["source_y1_px"] = tile_metrics["y1_px"] + sample.source_origin_y

    tile_cells = cells.loc[cells["inside_he_candidate"]].copy()
    tile_cells["tile_col"] = (tile_cells["he_x_px"] // config.tile_size_px).astype(int)
    tile_cells["tile_row"] = (tile_cells["he_y_px"] // config.tile_size_px).astype(int)
    tile_cells = tile_cells.loc[
        (tile_cells["tile_col"] < n_columns) & (tile_cells["tile_row"] < n_rows)
    ].copy()
    cell_counts = tile_cells.groupby(["tile_row", "tile_col"], observed=True).size()
    cell_counts = cell_counts.rename("n_cells").reset_index()
    tile_metrics = tile_metrics.merge(cell_counts, on=["tile_row", "tile_col"], how="left")
    tile_metrics["n_cells"] = tile_metrics["n_cells"].fillna(0).astype(int)
    tile_metrics["has_cells"] = tile_metrics["n_cells"] > 0
    tile_metrics["cell_area_um2"] = pd.NA
    tile_metrics["tile_area_um2"] = pd.NA
    tile_metrics["cell_area_fraction"] = pd.NA
    tile_metrics["meets_min_cells"] = tile_metrics["n_cells"] >= config.min_cells_per_tile
    tile_metrics["meets_min_cell_area_fraction"] = True
    tile_metrics["passes_tile_filter"] = tile_metrics["has_cells"] & tile_metrics["meets_min_cells"]
    low_count = ~tile_metrics["meets_min_cells"]
    low_area = ~tile_metrics["meets_min_cell_area_fraction"]
    tile_metrics["tile_filter_quadrant"] = np.select(
        [
            low_count & low_area,
            ~low_count & low_area,
            low_count & ~low_area,
            ~low_count & ~low_area,
        ],
        list(TILE_FILTER_QUADRANTS),
        default="unclassified",
    )

    counts = (
        tile_cells.groupby(["tile_row", "tile_col", "cell_type"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    if counts.empty:
        raise ValueError("No complete H&E tiles contain aligned Xenium cells")
    counts.columns = [f"count__{value}" for value in counts.columns]
    labels = (
        tile_metrics.loc[tile_metrics["n_cells"] > 0]
        .merge(counts.reset_index(), on=["tile_row", "tile_col"], how="left")
        .copy()
    )
    count_columns = list(counts.columns)
    labels[count_columns] = labels[count_columns].fillna(0).astype(int)
    for column in counts.columns:
        label = column.removeprefix("count__")
        labels[f"proportion__{label}"] = labels[column] / labels["n_cells"]
    return tile_cells, labels, tile_metrics


def save_tile_filter_compat_outputs(
    output: Path,
    sample_folder: str,
    tile_metrics: pd.DataFrame,
) -> None:
    """Write standard-style tile-filter audit files for Bilous outputs.

    Bilous currently filters tiles by cell count only.  The standard output
    schema includes cell-area fields, so those columns are present but empty.
    """
    tile_metrics.to_csv(output / f"{sample_folder}_tile_filter_metrics.csv", index=False)
    quantile_rows: list[dict[str, Any]] = []
    for column in ["n_cells", "cell_area_fraction"]:
        values = pd.to_numeric(tile_metrics[column], errors="coerce").dropna()
        if values.empty:
            continue
        for quantile in [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]:
            quantile_rows.append(
                {
                    "metric": column,
                    "quantile": quantile,
                    "value": float(values.quantile(quantile)),
                }
            )
    pd.DataFrame(quantile_rows).to_csv(
        output / f"{sample_folder}_tile_filter_quantiles.csv",
        index=False,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].hist(tile_metrics["n_cells"], bins=50, color="#4E79A7")
    axes[0].set_title("Cells per complete tile")
    axes[0].set_xlabel("n_cells")
    axes[0].set_ylabel("tiles")
    counts = (
        tile_metrics["tile_filter_quadrant"]
        .value_counts()
        .reindex(TILE_FILTER_QUADRANTS, fill_value=0)
    )
    colors = [TILE_FILTER_QUADRANT_COLORS[value] for value in counts.index]
    axes[1].bar(counts.index, counts.values, color=colors)
    axes[1].set_title("Tile filter quadrants")
    axes[1].tick_params(axis="x", rotation=45)
    fig.savefig(output / f"{sample_folder}_tile_filter_qc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def crop_tiles(
    sample: SampleInput,
    output: Path,
    labels: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_tiles = labels.loc[labels["n_cells"] > 0].copy()
    if all_tiles.empty:
        raise ValueError("No complete H&E tiles contain aligned Xenium cells")
    tile_directory = output / "he_tiles"
    tile_directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    with tifffile.TiffFile(sample.he_path) as tif:
        level = tif.series[0].levels[0]
        store = level.aszarr()
        try:
            array = resolve_zarr_array(zarr.open(store, mode="r"))
            print(
                f"[{sample.sample_folder}] extracting {len(all_tiles):,} nonempty tiles "
                f"from full-resolution shape {tuple(array.shape)}"
            )
            for row in tqdm(
                all_tiles.itertuples(index=False),
                total=len(all_tiles),
                desc=f"{sample.sample_folder}: tiles",
                unit="tile",
                dynamic_ncols=True,
            ):
                filename = (
                    f"{sample.sample_folder}_row-{int(row.tile_row):05d}_"
                    f"col-{int(row.tile_col):05d}.png"
                )
                path = tile_directory / filename
                if config.overwrite or not path.exists():
                    tile = extract_rgb_tile(
                        array,
                        int(row.source_x0_px),
                        int(row.source_y0_px),
                        int(row.source_x1_px),
                        int(row.source_y1_px),
                    )
                    expected = (config.tile_size_px, config.tile_size_px, 3)
                    if tile.shape != expected:
                        raise ValueError(
                            f"Tile {filename} has shape {tile.shape}; expected {expected}"
                        )
                    Image.fromarray(tile).save(path, format="PNG", compress_level=3)
                paths.append(path.relative_to(output).as_posix())
        finally:
            close = getattr(store, "close", None)
            if close:
                close()
    all_tiles["tile_path"] = paths
    if "passes_tile_filter" in all_tiles.columns:
        selected = all_tiles.loc[all_tiles["passes_tile_filter"]].copy()
    else:
        selected = all_tiles.loc[all_tiles["n_cells"] >= config.min_cells_per_tile].copy()
    all_tiles.to_csv(output / f"{sample.sample_folder}_all_tile_labels_with_paths.csv", index=False)
    selected.to_csv(output / f"{sample.sample_folder}_tile_labels_with_paths.csv", index=False)
    if selected.empty:
        raise ValueError(
            f"No tiles satisfy min_cells_per_tile={config.min_cells_per_tile}; "
            f"{len(all_tiles):,} nonempty tile PNG(s) were still written"
        )
    return all_tiles, selected


def candidate_thumbnail(sample: SampleInput, max_dimension: int = 2000) -> np.ndarray:
    mapping = {name: str(value) for name, value in sample.mapping.items()}
    root = sample.root
    region_path = root / "mapping_previews" / "ndpi_candidate_regions.csv"
    region_lookup = load_region_lookup(region_path) if region_path.is_file() else {}
    return ensure_rgb(load_he_candidate(root, mapping, region_lookup, max_dimension)["image"])


def draw_cells_on_axis(
    axis: Any,
    cells: pd.DataFrame,
    cell_boundaries: BoundaryData | None,
    nucleus_boundaries: BoundaryData | None = None,
    *,
    x0: float = 0,
    y0: float = 0,
    scale_x: float = 1,
    scale_y: float = 1,
) -> None:
    """Draw cell boundaries, nucleus boundaries, and nucleus-center markers."""
    def draw_boundary_layer(
        boundaries: BoundaryData | None,
        *,
        color: str,
        linewidth: float,
        alpha: float,
    ) -> None:
        if boundaries is None:
            return
        for cell in cells.itertuples(index=False):
            points = boundaries.polygons_by_cell_id.get(str(cell.cell_id))
            if points is None:
                continue
            local = (points - np.array([x0, y0])) * np.array([scale_x, scale_y])
            axis.plot(
                local[:, 0],
                local[:, 1],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
            )
    draw_boundary_layer(cell_boundaries, color="deepskyblue", linewidth=0.6, alpha=0.8)
    draw_boundary_layer(nucleus_boundaries, color="magenta", linewidth=0.7, alpha=0.9)
    marker_x, marker_y, _ = marker_he_coordinates(cells)
    axis.scatter(
        (marker_x - x0) * scale_x,
        (marker_y - y0) * scale_y,
        s=10,
        marker="+",
        color="yellow",
        linewidths=0.8,
    )


def save_alignment_plot(
    sample: SampleInput,
    output: Path,
    cells: pd.DataFrame,
    thumbnail: np.ndarray,
) -> None:
    inside = cells.loc[cells["inside_he_candidate"]]
    shown = inside.sample(n=min(30_000, len(inside)), random_state=1)
    marker_x, marker_y, marker_label = marker_he_coordinates(shown)
    if {"nucleus_x_centroid", "nucleus_y_centroid"}.issubset(shown.columns):
        xenium_x = shown["nucleus_x_centroid"].where(
            shown["nucleus_x_centroid"].notna(),
            shown["x_centroid"],
        )
        xenium_y = shown["nucleus_y_centroid"].where(
            shown["nucleus_y_centroid"].notna(),
            shown["y_centroid"],
        )
    else:
        xenium_x = shown["x_centroid"]
        xenium_y = shown["y_centroid"]
    sx = thumbnail.shape[1] / sample.candidate_width
    sy = thumbnail.shape[0] / sample.candidate_height
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), constrained_layout=True)
    axes[0].scatter(
        xenium_x,
        xenium_y,
        s=0.2,
        alpha=0.5,
        color="deepskyblue",
    )
    axes[0].invert_yaxis()
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("X (um)")
    axes[0].set_ylabel("Y (um)")
    axes[0].set_title(f"Xenium {marker_label.lower()}s")
    axes[1].imshow(thumbnail)
    axes[1].set_title("Post-Xenium H&E")
    axes[1].axis("off")
    axes[2].imshow(thumbnail)
    axes[2].scatter(
        marker_x * sx,
        marker_y * sy,
        s=0.2,
        alpha=0.4,
        color="deepskyblue",
    )
    axes[2].set_xlim(0, thumbnail.shape[1])
    axes[2].set_ylim(thumbnail.shape[0], 0)
    axes[2].set_title(f"Aligned Xenium {marker_label.lower()}s over H&E")
    axes[2].axis("off")
    fig.savefig(
        output / f"{sample.sample_folder}_xenium_he_alignment.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def save_label_plot(
    sample: SampleInput,
    output: Path,
    cells: pd.DataFrame,
    thumbnail: np.ndarray,
    colors: dict[str, tuple[int, int, int]],
) -> None:
    inside = cells.loc[cells["inside_he_candidate"]]
    shown = inside.sample(n=min(100_000, len(inside)), random_state=1)
    marker_x, marker_y, _ = marker_he_coordinates(shown)
    sx = thumbnail.shape[1] / sample.candidate_width
    sy = thumbnail.shape[0] / sample.candidate_height
    counts = inside["cell_type"].value_counts().sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    axes[0].imshow(thumbnail)
    axes[0].scatter(
        marker_x * sx,
        marker_y * sy,
        c=[np.array(colors[str(value)]) / 255 for value in shown["cell_type"]],
        s=0.25,
        alpha=0.55,
    )
    axes[0].set_xlim(0, thumbnail.shape[1])
    axes[0].set_ylim(thumbnail.shape[0], 0)
    axes[0].set_title("Cell labels over H&E")
    axes[0].axis("off")
    axes[1].barh(
        counts.index,
        counts.values,
        color=[np.array(colors[str(value)]) / 255 for value in counts.index],
    )
    axes[1].set_xlabel("Cells inside H&E candidate")
    axes[1].set_title("Cell label counts")
    fig.savefig(
        output / f"{sample.sample_folder}_cell_clusters.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def save_tile_preview(
    sample: SampleInput,
    output: Path,
    tile_cells: pd.DataFrame,
    manifest: pd.DataFrame,
    cell_boundaries: BoundaryData | None,
    nucleus_boundaries: BoundaryData | None,
    config: PipelineConfig,
) -> None:
    preview = manifest.sample(n=min(config.preview_count, len(manifest)), random_state=1)
    columns = 4
    rows = int(math.ceil(len(preview) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows), squeeze=False)
    flat = axes.ravel()
    for axis, row in zip(flat, preview.itertuples(index=False)):
        image = load_tile_image_for_qc(sample, output, row, config)
        axis.imshow(image)
        selected = tile_cells.loc[
            (tile_cells["tile_row"] == row.tile_row)
            & (tile_cells["tile_col"] == row.tile_col)
        ]
        draw_cells_on_axis(
            axis,
            selected,
            cell_boundaries,
            nucleus_boundaries,
            x0=float(row.x0_px),
            y0=float(row.y0_px),
        )
        axis.set_xlim(0, config.tile_size_px)
        axis.set_ylim(config.tile_size_px, 0)
        axis.set_title(f"row={row.tile_row}, col={row.tile_col}\ncells={len(selected)}")
        axis.axis("off")
    for axis in flat[len(preview):]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(
        output / f"{sample.sample_folder}_tile_preview_with_cells.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        output / f"{sample.sample_folder}_tile_filter_quadrant_examples.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def save_precrop_alignment_tile_preview(
    sample: SampleInput,
    output: Path,
    tile_cells: pd.DataFrame,
    labels: pd.DataFrame,
    cell_boundaries: BoundaryData | None,
    nucleus_boundaries: BoundaryData | None,
    config: PipelineConfig,
    *,
    n_examples: int = 9,
) -> Path:
    """Save a 3x3 in-memory tile preview before the expensive crop pass."""
    preview = (
        labels.loc[labels["n_cells"] > 0]
        .sort_values("n_cells", ascending=False)
        .head(max(n_examples * 4, n_examples))
        .sample(n=min(n_examples, len(labels.loc[labels["n_cells"] > 0])), random_state=1)
        .sort_values(["tile_row", "tile_col"])
    )
    if preview.empty:
        raise ValueError("No nonempty complete tiles are available for preview")
    columns = 3
    rows = int(math.ceil(len(preview) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, 4 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    flat = axes.ravel()
    with tifffile.TiffFile(sample.he_path) as tif:
        store = tif.series[0].levels[0].aszarr()
        try:
            image_array = resolve_zarr_array(zarr.open(store, mode="r"))
            for axis, row in zip(flat, preview.itertuples(index=False)):
                tile = extract_rgb_tile(
                    image_array,
                    int(row.source_x0_px),
                    int(row.source_y0_px),
                    int(row.source_x1_px),
                    int(row.source_y1_px),
                )
                axis.imshow(tile)
                selected = tile_cells.loc[
                    (tile_cells["tile_row"] == row.tile_row)
                    & (tile_cells["tile_col"] == row.tile_col)
                ]
                draw_cells_on_axis(
                    axis,
                    selected,
                    cell_boundaries,
                    nucleus_boundaries,
                    x0=float(row.x0_px),
                    y0=float(row.y0_px),
                )
                axis.set_xlim(0, config.tile_size_px)
                axis.set_ylim(config.tile_size_px, 0)
                axis.set_title(
                    f"row={int(row.tile_row)}, col={int(row.tile_col)}\n"
                    f"cells={len(selected)}"
                )
                axis.axis("off")
        finally:
            close = getattr(store, "close", None)
            if close:
                close()
    for axis in flat[len(preview):]:
        axis.axis("off")
    fig.suptitle(
        f"{sample.sample_folder}: pre-cropping alignment tile check\n"
        "yellow +=nucleus center when available; cyan=cell boundary; magenta=nucleus boundary",
        fontsize=13,
    )
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{sample.sample_folder}_precrop_alignment_tile_examples.png"
    fig.savefig(path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_orientation_plot(
    sample: SampleInput,
    output: Path,
    tile_cells: pd.DataFrame,
    manifest: pd.DataFrame,
    cell_boundaries: BoundaryData | None,
    nucleus_boundaries: BoundaryData | None,
    thumbnail: np.ndarray,
    config: PipelineConfig,
) -> None:
    selected = manifest.sort_values("n_cells", ascending=False).iloc[0]
    tile = np.asarray(load_tile_image_for_qc(sample, output, selected, config))
    cells = tile_cells.loc[
        (tile_cells["tile_row"] == selected["tile_row"])
        & (tile_cells["tile_col"] == selected["tile_col"])
    ]
    x0, y0 = int(selected["x0_px"]), int(selected["y0_px"])
    sx = thumbnail.shape[1] / sample.candidate_width
    sy = thumbnail.shape[0] / sample.candidate_height
    fig, axes = plt.subplots(1, 3, figsize=(14, 7), constrained_layout=True)
    axes[0].imshow(thumbnail)
    axes[0].add_patch(
        Rectangle(
            (x0 * sx, y0 * sy),
            config.tile_size_px * sx,
            config.tile_size_px * sy,
            edgecolor="yellow",
            facecolor="none",
            linewidth=2,
        )
    )
    axes[0].set_title(
        f"Tile location\nrow={selected['tile_row']}, col={selected['tile_col']}"
    )
    axes[0].axis("off")
    axes[1].imshow(tile)
    labels = [
        (4, 12, "TOP-LEFT", "left"),
        (config.tile_size_px - 4, 12, "TOP-RIGHT", "right"),
        (4, config.tile_size_px - 8, "BOTTOM-LEFT", "left"),
        (config.tile_size_px - 4, config.tile_size_px - 8, "BOTTOM-RIGHT", "right"),
    ]
    for x, y, label, alignment in labels:
        axes[1].text(
            x,
            y,
            label,
            color="yellow",
            fontsize=8,
            horizontalalignment=alignment,
        )
    axes[1].set_title("Saved full-resolution tile")
    axes[1].axis("off")
    axes[2].imshow(tile)
    draw_cells_on_axis(
        axes[2],
        cells,
        cell_boundaries,
        nucleus_boundaries,
        x0=x0,
        y0=y0,
    )
    axes[2].set_xlim(0, config.tile_size_px)
    axes[2].set_ylim(config.tile_size_px, 0)
    axes[2].set_title(f"Xenium cells and nuclei over H&E\n{len(cells)} cells")
    axes[2].axis("off")
    _, _, marker_label = marker_he_coordinates(cells)
    axes[2].legend(
        handles=[
            Line2D([0], [0], color="deepskyblue", linewidth=1.5, label="Cell boundary"),
            Line2D([0], [0], color="magenta", linewidth=1.5, label="Nucleus boundary"),
            Line2D([0], [0], color="yellow", marker="+", linestyle="none", label=marker_label),
        ],
        loc="lower right",
        fontsize=7,
        framealpha=0.8,
    )
    fig.savefig(
        output / f"{sample.sample_folder}_tile_orientation_check.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def save_reconstruction_plot(
    sample: SampleInput,
    output: Path,
    tile_cells: pd.DataFrame,
    manifest: pd.DataFrame,
    boundaries: BoundaryData | None,
    thumbnail: np.ndarray,
    colors: dict[str, tuple[int, int, int]],
    config: PipelineConfig,
) -> None:
    height, width = thumbnail.shape[:2]
    sx = width / sample.candidate_width
    sy = height / sample.candidate_height
    reconstructed = np.full((height, width, 3), 255, dtype=np.uint8)
    coverage = np.zeros((height, width), dtype=bool)
    for row in tqdm(
        manifest.itertuples(index=False),
        total=len(manifest),
        desc=f"{sample.sample_folder}: reconstruction",
        unit="tile",
        dynamic_ncols=True,
    ):
        with Image.open(output / row.tile_path) as image:
            tile = image.convert("RGB")
        x0, y0 = round(row.x0_px * sx), round(row.y0_px * sy)
        x1, y1 = round(row.x1_px * sx), round(row.y1_px * sy)
        if x1 <= x0 or y1 <= y0:
            continue
        reconstructed[y0:y1, x0:x1] = np.asarray(
            tile.resize((x1 - x0, y1 - y0), Image.Resampling.BOX)
        )
        coverage[y0:y1, x0:x1] = True
    Image.fromarray(reconstructed).save(output / f"{sample.sample_folder}_reconstructed_from_tiles.png")

    keys = manifest[["tile_row", "tile_col"]].drop_duplicates()
    cells = (
        tile_cells.merge(keys, on=["tile_row", "tile_col"], how="inner")
        .drop_duplicates("cell_id")
    )
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    for cell in tqdm(
        cells.itertuples(index=False),
        total=len(cells),
        desc=f"{sample.sample_folder}: annotated cells",
        unit="cell",
        dynamic_ncols=True,
    ):
        cell_type = str(cell.cell_type) if pd.notna(cell.cell_type) else "Unassigned"
        color = colors.get(cell_type, (140, 140, 140))
        polygon = None if boundaries is None else boundaries.polygons_by_cell_id.get(str(cell.cell_id))
        if polygon is not None:
            points = [(round(x * sx), round(y * sy)) for x, y in polygon]
            draw.polygon(points, fill=color + (config.annotation_alpha,), outline=color + (255,))
        marker_x, marker_y = marker_he_coordinates_from_record(cell)
        cx, cy = round(marker_x * sx), round(marker_y * sy)
        draw.ellipse((cx - 1, cy - 1, cx + 1, cy + 1), fill=color + (255,))
    layer_array = np.asarray(layer).copy()
    layer_array[~coverage, 3] = 0
    annotated = Image.alpha_composite(
        Image.fromarray(reconstructed).convert("RGBA"), Image.fromarray(layer_array)
    )
    annotated.save(output / f"{sample.sample_folder}_reconstructed_tiles_with_cells.png")

    fig, axes = plt.subplots(1, 3, figsize=(17, 10), constrained_layout=True)
    axes[0].imshow(thumbnail)
    axes[0].set_title("Original mapped H&E")
    axes[1].imshow(reconstructed)
    axes[1].set_title("Reconstructed from saved tiles")
    axes[2].imshow(annotated)
    axes[2].set_title("Reconstructed tiles with\nXenium cell labels")
    for axis in axes:
        axis.axis("off")
    present = sorted(cells["cell_type"].astype(str).unique())
    if len(present) <= 30:
        axes[2].legend(
            handles=[
                Patch(
                    facecolor=np.array(colors.get(label, (140, 140, 140))) / 255,
                    edgecolor=np.array(colors.get(label, (140, 140, 140))) / 255,
                    label=label,
                    alpha=max(config.annotation_alpha / 255, 0.75),
                )
                for label in present
            ],
            title="Cell annotation",
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=7,
            title_fontsize=8,
            frameon=True,
        )
    fig.savefig(
        output / f"{sample.sample_folder}_annotated_tile_reconstruction.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def completion_path(output: Path, sample_folder: str) -> Path:
    return output / f"{sample_folder}_pipeline_complete.json"


def existing_manifest(output: Path, sample_folder: str) -> pd.DataFrame:
    path = output / f"{sample_folder}_tile_labels_with_paths.csv"
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def existing_tile_manifests_for_qc(output: Path, sample: SampleInput) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_folder = sample.sample_folder
    selected_path = output / f"{sample_folder}_tile_labels_with_paths.csv"
    if not selected_path.is_file():
        raise FileNotFoundError(
            f"Cannot recreate tile QC plots without {selected_path}"
        )
    all_path = output / f"{sample_folder}_all_tile_labels_with_paths.csv"
    selected = pd.read_csv(selected_path)
    all_tiles = pd.read_csv(all_path) if all_path.is_file() else selected.copy()
    for label, manifest in {"selected": selected, "all": all_tiles}.items():
        if "tile_path" not in manifest.columns:
            raise ValueError(f"Existing {label} tile manifest has no tile_path column")
        missing = [
            value
            for value in manifest["tile_path"].head(1000)
            if not (output / str(value)).is_file()
        ]
        if missing and sample.he_path is None:
            raise FileNotFoundError(
                "Existing tile manifest references missing tile PNG(s), and no H&E "
                f"image is available for in-memory QC crops; for example: {missing[0]}"
            )
    return all_tiles, selected


def manifest_tile_files_available(output: Path, manifest: pd.DataFrame) -> bool:
    if "tile_path" not in manifest.columns:
        return False
    return bool(
        manifest["tile_path"].map(lambda value: (output / str(value)).is_file()).all()
    )


def load_tile_image_for_qc(
    sample: SampleInput,
    output: Path,
    row: object,
    config: PipelineConfig,
) -> Image.Image:
    tile_path = output / str(getattr(row, "tile_path"))
    if tile_path.is_file():
        with Image.open(tile_path) as image:
            return image.convert("RGB")
    with tifffile.TiffFile(sample.he_path) as tif:
        store = tif.series[0].levels[0].aszarr()
        try:
            image_array = resolve_zarr_array(zarr.open(store, mode="r"))
            tile = extract_rgb_tile(
                image_array,
                int(getattr(row, "source_x0_px")),
                int(getattr(row, "source_y0_px")),
                int(getattr(row, "source_x1_px")),
                int(getattr(row, "source_y1_px")),
            )
        finally:
            close = getattr(store, "close", None)
            if close:
                close()
    expected = (config.tile_size_px, config.tile_size_px, 3)
    if tile.shape != expected:
        raise ValueError(f"Unexpected in-memory QC tile shape {tile.shape}; expected {expected}")
    return Image.fromarray(tile).convert("RGB")


def run_sample(
    sample: SampleInput,
    output_root: Path,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = output_root / sample.sample_folder
    output.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Processing {sample.sample_folder} ===", flush=True)
    cells, label_source = load_cells(sample, config.label_column)
    inside = int(cells["inside_he_candidate"].sum())
    print(
        f"[{sample.sample_folder}] cells={len(cells):,}, "
        f"inside mapped H&E={inside:,} ({inside / max(len(cells), 1):.1%}), "
        f"labels={label_source}"
    )
    tile_cells, labels, tile_metrics = create_tile_labels(sample, cells, config)
    if not config.recreate_qc_plots:
        labels.to_csv(output / f"{sample.sample_folder}_tile_labels.csv", index=False)
        save_tile_filter_compat_outputs(output, sample.sample_folder, tile_metrics)
        aligned_columns = [
            "cell_index", "cell_id", "x_centroid", "y_centroid", "cell_type",
            "he_x_px", "he_y_px", "inside_he_candidate",
        ]
        aligned_columns.extend(
            column
            for column in [
                "nucleus_x_centroid",
                "nucleus_y_centroid",
                "nucleus_he_x_px",
                "nucleus_he_y_px",
                "nucleus_inside_he_candidate",
            ]
            if column in cells.columns
        )
        cells[aligned_columns].to_csv(
            output / f"{sample.sample_folder}_cells_aligned_to_he.csv",
            index=False,
        )
        cells[aligned_columns].to_csv(
            output / f"{sample.sample_folder}_cells_with_annotations.csv",
            index=False,
        )
    if config.recreate_qc_plots:
        all_manifest, manifest = existing_tile_manifests_for_qc(output, sample)
        if config.make_qc_plots:
            cell_boundaries = load_boundaries(sample, "cell")
            nucleus_boundaries = load_boundaries(sample, "nucleus")
            thumbnail = candidate_thumbnail(sample)
            colors = create_color_map(cells.loc[cells["inside_he_candidate"], "cell_type"])
            save_alignment_plot(sample, output, cells, thumbnail)
            present_labels = set(cells.loc[cells["inside_he_candidate"], "cell_type"])
            if present_labels != {"Unassigned"}:
                save_label_plot(sample, output, cells, thumbnail, colors)
            save_tile_preview(
                sample,
                output,
                tile_cells,
                manifest,
                cell_boundaries,
                nucleus_boundaries,
                config,
            )
            save_orientation_plot(
                sample,
                output,
                tile_cells,
                manifest,
                cell_boundaries,
                nucleus_boundaries,
                thumbnail,
                config,
            )
            if manifest_tile_files_available(output, manifest):
                save_reconstruction_plot(sample, output, tile_cells, manifest, cell_boundaries, thumbnail, colors, config)
            else:
                print(
                    f"[{sample.sample_folder}] skipped reconstruction QC refresh because "
                    "tile PNGs are not present"
                )
        record = {
            "sample": sample.sample_folder,
            "status": "qc_recreated",
            "gsm": sample.gsm,
            "he_source": str(sample.he_path),
            "transform": str(sample.transform_path),
            "label_source": label_source,
            "n_cells": len(cells),
            "n_cells_inside_he": inside,
            "inside_fraction": inside / max(len(cells), 1),
            "n_labeled_tiles": len(labels),
            "n_written_tiles": len(all_manifest),
            "n_retained_tiles": len(manifest),
            "tile_size_px": config.tile_size_px,
        }
        combined = manifest.copy()
        combined["tile_path"] = combined["tile_path"].map(
            lambda value: (Path(sample.sample_folder) / str(value)).as_posix()
        )
        return combined, {**record, "error": ""}
    all_manifest, manifest = crop_tiles(sample, output, labels, config)
    cell_boundaries = load_boundaries(sample, "cell") if config.make_qc_plots else None
    nucleus_boundaries = load_boundaries(sample, "nucleus") if config.make_qc_plots else None
    if config.make_qc_plots:
        thumbnail = candidate_thumbnail(sample)
        colors = create_color_map(cells.loc[cells["inside_he_candidate"], "cell_type"])
        save_alignment_plot(sample, output, cells, thumbnail)
        present_labels = set(cells.loc[cells["inside_he_candidate"], "cell_type"])
        if present_labels != {"Unassigned"}:
            save_label_plot(sample, output, cells, thumbnail, colors)
        save_tile_preview(
            sample,
            output,
            tile_cells,
            manifest,
            cell_boundaries,
            nucleus_boundaries,
            config,
        )
        save_orientation_plot(
            sample,
            output,
            tile_cells,
            manifest,
            cell_boundaries,
            nucleus_boundaries,
            thumbnail,
            config,
        )
        save_reconstruction_plot(sample, output, tile_cells, manifest, cell_boundaries, thumbnail, colors, config)
    record = {
        "sample": sample.sample_folder,
        "status": "success",
        "gsm": sample.gsm,
        "he_source": str(sample.he_path),
        "transform": str(sample.transform_path),
        "label_source": label_source,
        "n_cells": len(cells),
        "n_cells_inside_he": inside,
        "inside_fraction": inside / max(len(cells), 1),
        "n_labeled_tiles": len(labels),
        "n_written_tiles": len(all_manifest),
        "n_retained_tiles": len(manifest),
        "tile_size_px": config.tile_size_px,
    }
    completion_path(output, sample.sample_folder).write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    combined = manifest.copy()
    combined["tile_path"] = combined["tile_path"].map(
        lambda value: (Path(sample.sample_folder) / value).as_posix()
    )
    return combined, {**record, "error": ""}




def write_xenium_explorer_imagealignment_csv(
    path: str | Path,
    xenium_to_he: np.ndarray,
) -> np.ndarray:
    """Write a Xenium Explorer-compatible 3x3 imagealignment CSV."""
    imagealignment = xenium_explorer_imagealignment_matrix(xenium_to_he)
    np.savetxt(Path(path), imagealignment, delimiter=",", fmt="%.15g")
    return imagealignment


def ensure_transform_imagealignment_csv(sample: SampleInput) -> Path | None:
    """Export a selected transform JSON as an Explorer-style CSV when possible."""
    payload = json.loads(sample.transform_path.read_text(encoding="utf-8"))
    selected = payload.get("selected_transform", payload)
    matrix = selected.get("matrix_xenium_full_pixel_to_he_candidate_local_full_pixel")
    if matrix is None:
        return None
    imagealignment_path = sample.transform_path.parent / (
        f"{sample.sample_folder}_he_imagealignment.csv"
    )
    he_to_xenium = write_xenium_explorer_imagealignment_csv(
        imagealignment_path,
        np.asarray(matrix, dtype=float),
    )
    payload.setdefault(
        "xenium_explorer_matrix_convention",
        (
            "3x3 row-major affine matrix mapping H&E candidate "
            "full-resolution pixels to Xenium full-resolution pixels; "
            "final row is [0, 0, 1]."
        ),
    )
    payload["matrix_he_candidate_local_full_pixel_to_xenium_full_pixel"] = (
        he_to_xenium.tolist()
    )
    payload["xenium_explorer_imagealignment_csv"] = str(imagealignment_path)
    sample.transform_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return imagealignment_path


def run_alignment_qc(
    root: str | Path,
    sample_names: list[str] | None = None,
    *,
    output_dir: str | Path | None = None,
    transform_dir: str | Path | None = None,
    label_column: str | None = None,
    fail_fast: bool = False,
) -> pd.DataFrame:
    """Create only the Xenium/H&E alignment QC plot for each mapped sample."""
    root = Path(root).expanduser().resolve()
    output_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else root / DEFAULT_OUTPUT_SUBDIR
    )
    output_root.mkdir(parents=True, exist_ok=True)
    samples, validation = discover_inputs(root, sample_names, transform_dir=transform_dir)
    records: list[dict[str, Any]] = []
    invalid_names = {row["sample"] for row in validation if row["status"] == "invalid"}
    for row in validation:
        if row["sample"] in invalid_names:
            records.append(
                {
                    "sample": row["sample"],
                    "status": "invalid",
                    "gsm": row.get("gsm", ""),
                    "he_source": "",
                    "transform": row.get("transform_path", ""),
                    "imagealignment_csv": "",
                    "alignment_plot": "",
                    "n_cells": pd.NA,
                    "n_cells_inside_he": pd.NA,
                    "inside_fraction": pd.NA,
                    "error": row.get("error", "input validation failed"),
                }
            )
    for sample in tqdm(samples, desc="Bilous alignment QC", unit="sample", dynamic_ncols=True):
        output = output_root / sample.sample_folder
        output.mkdir(parents=True, exist_ok=True)
        try:
            imagealignment_path = ensure_transform_imagealignment_csv(sample)
            cells, label_source = load_cells(sample, label_column)
            inside = int(cells["inside_he_candidate"].sum())
            if inside <= 0:
                raise ValueError("No cell centroids fall inside the mapped H&E candidate")
            thumbnail = candidate_thumbnail(sample)
            save_alignment_plot(sample, output, cells, thumbnail)
            records.append(
                {
                    "sample": sample.sample_folder,
                    "status": "success",
                    "gsm": sample.gsm,
                    "he_source": str(sample.he_path),
                    "transform": str(sample.transform_path),
                    "imagealignment_csv": (
                        str(imagealignment_path) if imagealignment_path else ""
                    ),
                    "alignment_plot": str(output / f"{sample.sample_folder}_xenium_he_alignment.png"),
                    "label_source": label_source,
                    "n_cells": len(cells),
                    "n_cells_inside_he": inside,
                    "inside_fraction": inside / max(len(cells), 1),
                    "error": "",
                }
            )
        except Exception as error:
            print(f"[{sample.sample_folder}] alignment QC FAILED: {error}", file=sys.stderr)
            traceback.print_exc()
            records.append(
                {
                    "sample": sample.sample_folder,
                    "status": "failed",
                    "gsm": sample.gsm,
                    "he_source": str(sample.he_path),
                    "transform": str(sample.transform_path),
                    "imagealignment_csv": "",
                    "alignment_plot": "",
                    "n_cells": pd.NA,
                    "n_cells_inside_he": pd.NA,
                    "inside_fraction": pd.NA,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            if fail_fast:
                break
        finally:
            plt.close("all")
            gc.collect()
    summary = pd.DataFrame(records)
    summary.to_csv(output_root / "alignment_qc_summary.csv", index=False)
    return summary


def run_precrop_tile_alignment_qc(
    root: str | Path,
    sample_names: list[str] | None = None,
    *,
    output_dir: str | Path | None = None,
    transform_dir: str | Path | None = None,
    tile_size: int = 1024,
    label_column: str | None = None,
    n_examples: int = 9,
    fail_fast: bool = False,
) -> pd.DataFrame:
    """Create 9 example H&E tile crops with aligned nuclei and boundaries.

    This is intentionally lighter than ``run_all_samples``: it computes aligned
    cells and tile labels, then extracts only a small preview panel in memory.
    It does not write tile PNGs or tile manifests for downstream analysis.
    """
    root = Path(root).expanduser().resolve()
    output_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else root / DEFAULT_OUTPUT_SUBDIR
    )
    output_root.mkdir(parents=True, exist_ok=True)
    samples, validation = discover_inputs(root, sample_names, transform_dir=transform_dir)
    config = PipelineConfig(
        tile_size_px=int(tile_size),
        min_cells_per_tile=1,
        preview_count=int(n_examples),
        label_column=label_column,
        make_qc_plots=True,
    )
    records: list[dict[str, Any]] = []
    invalid_names = {row["sample"] for row in validation if row["status"] == "invalid"}
    for row in validation:
        if row["sample"] in invalid_names:
            records.append(
                {
                    "sample": row["sample"],
                    "status": "invalid",
                    "preview_plot": "",
                    "n_cells": pd.NA,
                    "n_cells_inside_he": pd.NA,
                    "n_preview_tiles": pd.NA,
                    "error": row.get("error", "input validation failed"),
                }
            )
    for sample in tqdm(samples, desc="Bilous pre-crop tile QC", unit="sample", dynamic_ncols=True):
        output = output_root / sample.sample_folder
        output.mkdir(parents=True, exist_ok=True)
        try:
            cells, label_source = load_cells(sample, label_column)
            inside = int(cells["inside_he_candidate"].sum())
            if inside <= 0:
                raise ValueError("No cell centroids fall inside the mapped H&E candidate")
            tile_cells, labels, _tile_metrics = create_tile_labels(sample, cells, config)
            cell_boundaries = load_boundaries(sample, "cell")
            nucleus_boundaries = load_boundaries(sample, "nucleus")
            preview_path = save_precrop_alignment_tile_preview(
                sample,
                output,
                tile_cells,
                labels,
                cell_boundaries,
                nucleus_boundaries,
                config,
                n_examples=n_examples,
            )
            records.append(
                {
                    "sample": sample.sample_folder,
                    "status": "success",
                    "gsm": sample.gsm,
                    "he_source": str(sample.he_path),
                    "transform": str(sample.transform_path),
                    "preview_plot": str(preview_path),
                    "label_source": label_source,
                    "n_cells": len(cells),
                    "n_cells_inside_he": inside,
                    "inside_fraction": inside / max(len(cells), 1),
                    "n_labeled_tiles": len(labels),
                    "n_preview_tiles": min(int(n_examples), int((labels["n_cells"] > 0).sum())),
                    "tile_size_px": config.tile_size_px,
                    "error": "",
                }
            )
        except Exception as error:
            print(f"[{sample.sample_folder}] pre-crop tile QC FAILED: {error}", file=sys.stderr)
            traceback.print_exc()
            records.append(
                {
                    "sample": sample.sample_folder,
                    "status": "failed",
                    "gsm": sample.gsm,
                    "he_source": str(sample.he_path),
                    "transform": str(sample.transform_path),
                    "preview_plot": "",
                    "n_cells": pd.NA,
                    "n_cells_inside_he": pd.NA,
                    "n_preview_tiles": pd.NA,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            if fail_fast:
                break
        finally:
            plt.close("all")
            gc.collect()
    summary = pd.DataFrame(records)
    summary_path = output_root / "precrop_tile_alignment_qc_summary.csv"
    if sample_names and len(sample_names) == 1:
        summary_path = output_root / str(sample_names[0]) / (
            f"{sample_names[0]}_precrop_tile_alignment_qc_summary.csv"
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    return summary


def _cohort_axis_labels(labels: Iterable[str]) -> list[str]:
    labels = [str(label) for label in labels]
    if labels and all("_GSM" in label for label in labels):
        return [label.split("_GSM", 1)[0] for label in labels]
    return labels


def _plot_cohort_bars(axis: plt.Axes, labels: list[str], values: pd.Series, title: str) -> None:
    positions = np.arange(len(labels))
    axis.bar(positions, np.asarray(values, dtype=float))
    axis.set_title(title)
    axis.set_xlim(-0.5, len(labels) - 0.5)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=8)


def save_dataset_summary(output_root: Path, summary: pd.DataFrame) -> None:
    successful = summary.loc[summary["status"].isin(["success", "skipped"])].copy()
    required = {"sample", "n_cells", "n_cells_inside_he", "n_written_tiles"}
    if successful.empty or not required.issubset(successful.columns):
        return
    for column in ["n_cells", "n_cells_inside_he", "n_written_tiles"]:
        successful[column] = pd.to_numeric(successful[column], errors="coerce")
    successful = successful.dropna(subset=list(required))
    if successful.empty:
        return
    successful["inside_fraction"] = (
        successful["n_cells_inside_he"] / successful["n_cells"]
    )
    labels = _cohort_axis_labels(successful["sample"])
    fig_width = max(17, len(labels) * 0.8)
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, 6), constrained_layout=True)
    _plot_cohort_bars(axes[0], labels, successful["n_cells"], "Detected cells")
    _plot_cohort_bars(axes[1], labels, successful["inside_fraction"] * 100, "Cells inside H&E (%)")
    _plot_cohort_bars(axes[2], labels, successful["n_written_tiles"], "Saved H&E tiles")
    fig.savefig(output_root / "dataset_qc_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_all_samples(
    root: str | Path,
    sample_names: list[str] | None = None,
    *,
    input_manifest: str | Path | None = None,
    output_dir: str | Path | None = None,
    transform_dir: str | Path | None = None,
    tile_size: int = 1024,
    min_cells_per_tile: int = 1,
    preview_count: int = 12,
    annotation_alpha: int = 210,
    label_column: str | None = None,
    overwrite: bool = False,
    skip_processed: bool = True,
    make_qc_plots: bool = True,
    recreate_qc_plots: bool = False,
    validate_only: bool = False,
    fail_fast: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and process every mapped sample; returns summary and tile tables."""
    if tile_size <= 0 or min_cells_per_tile <= 0:
        raise ValueError("tile_size and min_cells_per_tile must be positive")
    if not 0 <= annotation_alpha <= 255:
        raise ValueError("annotation_alpha must be between 0 and 255")
    root = Path(root).expanduser().resolve()
    output_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else root / DEFAULT_OUTPUT_SUBDIR
    )
    if input_manifest is not None:
        return standard_xenium.run_all_samples(
            root=output_root,
            sample_names=sample_names,
            input_manifest=input_manifest,
            tile_size=tile_size,
            min_cells_per_tile=min_cells_per_tile,
            min_cell_area_fraction=0.0,
            preview_count=preview_count,
            annotation_alpha=annotation_alpha,
            label_mode="auto" if label_column is None else "auto",
            overwrite=overwrite,
            skip_processed=skip_processed,
            make_qc_plots=make_qc_plots,
            validate_only=validate_only,
            recreate_qc_plots=recreate_qc_plots,
            fail_fast=fail_fast,
            require_alignment=True,
            missing_cell_label="Unassigned",
        )
    output_root.mkdir(parents=True, exist_ok=True)
    samples, validation = discover_inputs(root, sample_names, transform_dir=transform_dir)
    validation_table = pd.DataFrame(validation)
    validation_table.to_csv(output_root / "input_validation_summary.csv", index=False)
    for row in validation:
        print(f"[{row['status']}] {row['sample']}" + (f": {row['error']}" if row.get("error") else ""))
    if validate_only:
        return validation_table, pd.DataFrame()

    config = PipelineConfig(
        tile_size_px=tile_size,
        min_cells_per_tile=min_cells_per_tile,
        preview_count=preview_count,
        annotation_alpha=annotation_alpha,
        overwrite=overwrite,
        make_qc_plots=make_qc_plots,
        recreate_qc_plots=recreate_qc_plots,
        label_column=label_column,
    )
    manifests: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    invalid_names = {row["sample"] for row in validation if row["status"] == "invalid"}
    for row in validation:
        if row["sample"] in invalid_names:
            summaries.append({"sample": row["sample"], "status": "invalid", "error": row["error"]})
    for sample in tqdm(samples, desc="Bilous samples", unit="sample", dynamic_ncols=True):
        output = output_root / sample.sample_folder
        completed = completion_path(output, sample.sample_folder)
        prior_manifest = existing_manifest(output, sample.sample_folder)
        if (
            skip_processed
            and not overwrite
            and not recreate_qc_plots
            and completed.is_file()
            and not prior_manifest.empty
        ):
            previous = json.loads(completed.read_text(encoding="utf-8"))
            summaries.append({**previous, "status": "skipped", "error": "completion record exists"})
            if not prior_manifest.empty:
                prior_manifest["tile_path"] = prior_manifest["tile_path"].map(
                    lambda value: (Path(sample.sample_folder) / value).as_posix()
                )
                manifests.append(prior_manifest)
            print(f"[{sample.sample_folder}] skipped: completion record exists")
            continue
        try:
            manifest, summary = run_sample(sample, output_root, config)
            manifests.append(manifest)
            summaries.append(summary)
        except Exception as error:
            print(f"[{sample.sample_folder}] FAILED: {error}", file=sys.stderr)
            traceback.print_exc()
            summaries.append(
                {
                    "sample": sample.sample_folder,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            if fail_fast:
                break
        finally:
            plt.close("all")
            gc.collect()
    summary = pd.DataFrame(summaries)
    if not recreate_qc_plots:
        summary.to_csv(output_root / "pipeline_summary.csv", index=False)
    all_tiles = pd.DataFrame()
    if manifests:
        all_tiles = pd.concat(manifests, ignore_index=True, sort=False)
        label_columns = [
            column
            for column in all_tiles.columns
            if column.startswith("count__") or column.startswith("proportion__")
        ]
        all_tiles[label_columns] = all_tiles[label_columns].fillna(0)
        if not recreate_qc_plots:
            all_tiles.to_csv(output_root / "all_samples_tile_labels.csv", index=False)
    if make_qc_plots and not recreate_qc_plots:
        save_dataset_summary(output_root, summary)
    return summary, all_tiles


def main() -> int:
    args = parse_args()
    summary, _ = run_all_samples(
        args.root,
        args.samples,
        output_dir=args.output_dir,
        transform_dir=args.transform_dir,
        tile_size=args.tile_size,
        min_cells_per_tile=args.min_cells_per_tile,
        preview_count=args.preview_count,
        annotation_alpha=args.annotation_alpha,
        label_column=args.label_column,
        overwrite=args.overwrite,
        skip_processed=not args.reprocess_completed,
        make_qc_plots=not args.skip_qc_plots,
        recreate_qc_plots=args.recreate_qc_plots,
        validate_only=args.validate_only,
        fail_fast=args.fail_fast,
    )
    return int(summary["status"].isin(["invalid", "failed"]).any())


if __name__ == "__main__":
    raise SystemExit(main())
