#!/usr/bin/env python3
"""Shared Xenium-to-H&E tiling pipeline for standard 10x-style datasets.

This module covers datasets whose sample folders contain an H&E OME-TIFF,
optional ``*_he_imagealignment.csv`` affine matrix, Xenium output bundle or
extracted ``outs`` folder, and optional ``<sample>_cell_groups.csv`` labels.

Dataset directories keep small wrappers so existing notebooks can continue to
``import run_all_samples`` locally while the duplicated implementation lives in
one place.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import os
import shutil
import sys
import traceback
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import zarr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from zarr.storage import ZipStore

try:
    from .tile_utils import (
        apply_affine,
        create_color_map,
        ensure_rgb,
        extract_rgb_tile,
        get_hw,
        resolve_zarr_array,
        xenium_explorer_imagealignment_matrix,
    )
    from .image_metadata_manifest import ome_physical_pixel_size
except ImportError:  # pragma: no cover - supports direct script execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from xenium_pipeline.tile_utils import (  # type: ignore[no-redef]
        apply_affine,
        create_color_map,
        ensure_rgb,
        extract_rgb_tile,
        get_hw,
        resolve_zarr_array,
        xenium_explorer_imagealignment_matrix,
    )
    from xenium_pipeline.image_metadata_manifest import ome_physical_pixel_size


@dataclass
class SamplePaths:
    name: str
    base: Path
    outs: Path
    outs_zip: Path | None
    he_image: Path | None
    alignment: Path | None
    manual_alignment: Path
    cell_groups: Path | None
    cell_table: Path | None = None
    pixel_size_um: float | None = None
    cell_boundaries: Path | None = None
    nucleus_boundaries: Path | None = None

    @property
    def xe_outs(self) -> Path:
        """Compatibility alias used by the Janesick 2025 notebook."""
        return self.outs

    @property
    def xe_outs_zip(self) -> Path:
        """Compatibility alias; returns the expected path when no ZIP exists."""
        return self.outs_zip or self.base / f"{self.name}_xe_outs.zip"


@dataclass(frozen=True)
class PipelineConfig:
    tile_size_px: int = 256
    min_cells_per_tile: int = 1
    min_cell_area_fraction: float = 0.0
    preview_count: int = 12
    pyramid_level: int = 3
    annotation_alpha: int = 160
    overwrite: bool = False
    prune_stale_tiles: bool = False
    make_qc_plots: bool = True
    tile_qc_only: bool = False
    recreate_qc_plots: bool = False
    label_mode: str = "auto"
    selective_archive_extract: bool = False
    require_alignment: bool | None = None
    require_cell_groups: bool = False
    missing_cell_label: str = "Unassigned"
    cell_groups_label_source: str = "custom_cell_groups"
    cells_output_suffix: str = "_cells_with_annotations.csv"
    completion_marker_suffix: str = "_pipeline_complete.json"


@dataclass
class BoundaryData:
    num_vertices: np.ndarray
    vertices: np.ndarray
    polygon_row_for_cell: np.ndarray


@dataclass(frozen=True)
class BoundarySource:
    source: str
    path: Path | None = None
    polygon_set: str = ""
    detail: str = ""


@dataclass(frozen=True)
class ImageMetadata:
    thumbnail: np.ndarray
    full_height: int
    full_width: int
    thumbnail_height: int
    thumbnail_width: int
    thumbnail_scale_x: float
    thumbnail_scale_y: float
    physical_size_x_um: float | None
    physical_size_y_um: float | None


TILE_FILTER_QUADRANTS = (
    "low_count_low_area",
    "high_count_low_area",
    "low_count_high_area",
    "high_count_high_area",
)

TILE_FILTER_QUADRANT_LABELS = {
    "low_count_low_area": "low count / low area",
    "high_count_low_area": "high count / low area",
    "low_count_high_area": "low count / high area",
    "high_count_high_area": "high count / high area",
}

TILE_FILTER_QUADRANT_COLORS = {
    "low_count_low_area": "#8C8C8C",
    "high_count_low_area": "#F28E2B",
    "low_count_high_area": "#4E79A7",
    "high_count_high_area": "#59A14F",
}


def _filename_value(value: int | float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def tile_filter_cutoff_slug(
    min_cells_per_tile: int,
    min_cell_area_fraction: float,
    min_nucleus_area_fraction: float = 0.0,
    min_he_tissue_fraction: float = 0.0,
) -> str:
    """Return a readable, filename-safe label for tile-filter cutoffs."""
    slug = (
        f"mincells-{int(min_cells_per_tile)}_"
        f"minarea-{_filename_value(float(min_cell_area_fraction))}"
    )
    if float(min_nucleus_area_fraction) > 0:
        slug += f"_minnucarea-{_filename_value(float(min_nucleus_area_fraction))}"
    if float(min_he_tissue_fraction) > 0:
        slug += f"_mintissue-{_filename_value(float(min_he_tissue_fraction))}"
    return slug


def apply_tile_filter_cutoffs(
    tile_table: pd.DataFrame,
    min_cells_per_tile: int,
    min_cell_area_fraction: float,
    min_nucleus_area_fraction: float = 0.0,
    min_he_tissue_fraction: float = 0.0,
    *,
    nucleus_area_fraction_column: str = "nucleus_area_fraction",
    require_nucleus_area_fraction: bool = False,
    he_tissue_fraction_column: str = "he_tissue_fraction",
    require_he_tissue_fraction: bool = False,
) -> pd.DataFrame:
    """Recompute tile-filter columns for arbitrary cutoff values."""
    if tile_table.empty:
        return tile_table.copy()
    filtered = tile_table.copy()
    n_cells = pd.to_numeric(filtered["n_cells"], errors="coerce").fillna(0)
    area_fraction = pd.to_numeric(
        filtered.get("cell_area_fraction", pd.Series(np.nan, index=filtered.index)),
        errors="coerce",
    )
    nucleus_area_fraction = pd.to_numeric(
        filtered.get(nucleus_area_fraction_column, pd.Series(np.nan, index=filtered.index)),
        errors="coerce",
    )
    he_tissue_fraction = pd.to_numeric(
        filtered.get(he_tissue_fraction_column, pd.Series(np.nan, index=filtered.index)),
        errors="coerce",
    )
    has_nucleus_area_fraction = bool(nucleus_area_fraction.notna().any())
    has_he_tissue_fraction = bool(he_tissue_fraction.notna().any())
    filtered["n_cells"] = n_cells.astype(int)
    filtered["cell_area_fraction"] = area_fraction
    filtered["nucleus_area_fraction_for_filter"] = nucleus_area_fraction
    filtered["he_tissue_fraction_for_filter"] = he_tissue_fraction
    filtered["has_cells"] = n_cells > 0
    filtered["has_nucleus_area_fraction"] = has_nucleus_area_fraction
    filtered["has_he_tissue_fraction"] = has_he_tissue_fraction
    filtered["meets_min_cells"] = n_cells >= int(min_cells_per_tile)
    if float(min_cell_area_fraction) > 0:
        filtered["meets_min_cell_area_fraction"] = (
            area_fraction >= float(min_cell_area_fraction)
        )
    else:
        filtered["meets_min_cell_area_fraction"] = True
    nucleus_cutoff_applies = (
        float(min_nucleus_area_fraction) > 0
        and (has_nucleus_area_fraction or require_nucleus_area_fraction)
    )
    filtered["nucleus_area_fraction_cutoff_applied"] = nucleus_cutoff_applies
    if nucleus_cutoff_applies:
        filtered["meets_min_nucleus_area_fraction"] = (
            nucleus_area_fraction.fillna(0.0) >= float(min_nucleus_area_fraction)
        )
    else:
        filtered["meets_min_nucleus_area_fraction"] = True
    tissue_cutoff_applies = (
        float(min_he_tissue_fraction) > 0
        and (has_he_tissue_fraction or require_he_tissue_fraction)
    )
    filtered["he_tissue_fraction_cutoff_applied"] = tissue_cutoff_applies
    if tissue_cutoff_applies:
        filtered["meets_min_he_tissue_fraction"] = (
            he_tissue_fraction.fillna(0.0) >= float(min_he_tissue_fraction)
        )
    else:
        filtered["meets_min_he_tissue_fraction"] = True
    filtered["passes_tile_filter"] = (
        filtered["has_cells"]
        & filtered["meets_min_cells"]
        & filtered["meets_min_cell_area_fraction"]
        & filtered["meets_min_nucleus_area_fraction"]
        & filtered["meets_min_he_tissue_fraction"]
    )
    low_count = ~filtered["meets_min_cells"]
    low_area = ~filtered["meets_min_cell_area_fraction"]
    filtered["tile_filter_quadrant"] = np.select(
        [
            low_count & low_area,
            ~low_count & low_area,
            low_count & ~low_area,
            ~low_count & ~low_area,
        ],
        list(TILE_FILTER_QUADRANTS),
        default="unclassified",
    )
    return filtered


def parse_args(
    *,
    default_root: Path | None = None,
    description: str = "Create aligned H&E tiles and XOA-cluster labels.",
    defaults: dict[str, Any] | None = None,
) -> argparse.Namespace:
    default_root = Path(default_root or Path.cwd()).resolve()
    defaults = defaults or {}
    parser = argparse.ArgumentParser(
        description=description
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=f"Directory containing sample folders (default: {default_root}).",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        help="Optional sample folder names; all discoverable samples are used by default.",
    )
    parser.add_argument("--tile-size", type=int, default=int(defaults.get("tile_size", 256)))
    parser.add_argument(
        "--min-cells-per-tile",
        type=int,
        default=int(defaults.get("min_cells_per_tile", 1)),
    )
    parser.add_argument(
        "--min-cell-area-fraction",
        type=float,
        default=float(defaults.get("min_cell_area_fraction", 0.0)),
        help=(
            "Minimum fraction of tile area occupied by centroid-assigned cells. "
            "Use --tile-qc-only first to inspect the distribution."
        ),
    )
    parser.add_argument("--preview-count", type=int, default=int(defaults.get("preview_count", 12)))
    parser.add_argument("--pyramid-level", type=int, default=int(defaults.get("pyramid_level", 3)))
    parser.add_argument(
        "--annotation-alpha",
        type=int,
        default=int(defaults.get("annotation_alpha", 160)),
    )
    parser.add_argument(
        "--label-mode",
        choices=("auto", "graphclust", "none"),
        default="auto",
        help=(
            "auto: custom cell groups, then graph clusters; graphclust: ignore custom "
            "groups; none: label every cell Unassigned."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--prune-stale-tiles",
        action="store_true",
        help=(
            "After successfully writing the current retained tiles, delete "
            "pipeline-generated tile PNGs for this sample that are absent from "
            "the current manifest."
        ),
    )
    parser.add_argument(
        "--reprocess-completed",
        action="store_true",
        help="Process samples even when their completion record exists.",
    )
    parser.add_argument("--skip-qc-plots", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Extract archives if needed and validate inputs without making tiles.",
    )
    parser.add_argument(
        "--tile-qc-only",
        action="store_true",
        help=(
            "Align cells, write tile-filter metrics and plots, then stop before "
            "cropping H&E tile PNGs."
        ),
    )
    parser.add_argument(
        "--recreate-qc-plots",
        action="store_true",
        help=(
            "Regenerate QC plots from existing tile manifests and PNGs without "
            "cropping or rewriting tile PNGs."
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def first_existing(paths: Iterable[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def locate_outs(sample_dir: Path) -> Path:
    """Return a directory that contains the XOA files, if one is available."""
    name = sample_dir.name
    candidates = [
        sample_dir / f"{name}_outs",
        sample_dir / f"{name}_xe_outs",
        sample_dir / "outs",
        sample_dir / "xe_outs",
    ]
    candidates.extend(sorted(sample_dir.glob("*_outs")))
    candidates.extend(sorted(sample_dir.glob("*_xe_outs")))
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_dir():
            continue
        if (candidate / "experiment.xenium").is_file():
            return candidate
        nested = sorted(candidate.rglob("experiment.xenium"))
        if nested:
            return nested[0].parent
    return sample_dir / f"{name}_outs"


def build_sample_paths(sample_dir: Path) -> SamplePaths:
    name = sample_dir.name
    image_candidates = [
        sample_dir / f"{name}_he_image.ome.tif",
        sample_dir / f"{name}_he_unaligned_image.ome.tif",
    ]
    image_candidates.extend(sorted(sample_dir.glob("*he*image*.ome.tif")))
    alignment_candidates = [sample_dir / f"{name}_he_imagealignment.csv"]
    alignment_candidates.extend(sorted(sample_dir.glob("*he*imagealignment.csv")))
    zip_candidates = [
        sample_dir / f"{name}_outs.zip",
        sample_dir / f"{name}_xe_outs.zip",
        sample_dir / "outs.zip",
        sample_dir / "xe_outs.zip",
    ]
    zip_candidates.extend(sorted(sample_dir.glob("*_outs.zip")))
    zip_candidates.extend(sorted(sample_dir.glob("*_xe_outs.zip")))
    group_candidates = [sample_dir / f"{name}_cell_groups.csv"]
    group_candidates.extend(sorted(sample_dir.glob("*_cell_groups.csv")))
    return SamplePaths(
        name=name,
        base=sample_dir,
        outs=locate_outs(sample_dir),
        outs_zip=first_existing(dict.fromkeys(zip_candidates)),
        he_image=first_existing(dict.fromkeys(image_candidates)),
        alignment=first_existing(dict.fromkeys(alignment_candidates)),
        manual_alignment=sample_dir / f"{name}_manual_alignment.json",
        cell_groups=first_existing(dict.fromkeys(group_candidates)),
    )


def _optional_manifest_path(
    row: pd.Series,
    column: str,
    manifest_dir: Path,
) -> Path | None:
    value = str(row.get(column, "") or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else manifest_dir / path


def discover_samples_from_manifest(
    manifest_path: str | Path,
    requested: list[str] | None = None,
) -> list[SamplePaths]:
    """Resolve samples from an explicit tile-input manifest CSV.

    The manifest lets dataset-specific preparation code do path discovery once,
    while this shared tiling pipeline consumes ordinary resolved files.
    """
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest_dir = manifest_path.parent
    table = pd.read_csv(manifest_path, dtype=str).fillna("")
    required = {"sample", "he_image", "cells"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{manifest_path} is missing columns: {sorted(missing)}")
    if table["sample"].duplicated().any():
        duplicates = table.loc[table["sample"].duplicated(keep=False), "sample"].tolist()
        raise ValueError(f"Duplicate sample rows in {manifest_path}: {duplicates}")
    if requested:
        requested_set = set(requested)
        known = set(table["sample"])
        unknown = sorted(requested_set - known)
        if unknown:
            raise ValueError(f"Requested samples are absent from {manifest_path}: {unknown}")
        table = table.loc[table["sample"].isin(requested_set)]

    samples: list[SamplePaths] = []
    for _, row in table.iterrows():
        name = str(row["sample"])
        base = _optional_manifest_path(row, "output_dir", manifest_dir)
        if base is None:
            base = _optional_manifest_path(row, "sample_dir", manifest_dir)
        if base is None:
            base = manifest_dir / name
        pixel_size_text = str(row.get("pixel_size_um", "") or "").strip()
        pixel_size = float(pixel_size_text) if pixel_size_text else None
        samples.append(
            SamplePaths(
                name=name,
                base=base,
                outs=_optional_manifest_path(row, "outs_dir", manifest_dir) or base / "outs",
                outs_zip=_optional_manifest_path(row, "outs_zip", manifest_dir),
                he_image=_optional_manifest_path(row, "he_image", manifest_dir),
                alignment=_optional_manifest_path(row, "alignment_csv", manifest_dir)
                or _optional_manifest_path(row, "alignment", manifest_dir),
                manual_alignment=_optional_manifest_path(
                    row, "manual_alignment_json", manifest_dir
                )
                or base / f"{name}_manual_alignment.json",
                cell_groups=_optional_manifest_path(row, "cell_groups", manifest_dir),
                cell_table=_optional_manifest_path(row, "cells", manifest_dir),
                pixel_size_um=pixel_size,
                cell_boundaries=_optional_manifest_path(row, "cell_boundaries", manifest_dir),
                nucleus_boundaries=_optional_manifest_path(row, "nucleus_boundaries", manifest_dir),
            )
        )
    if not samples:
        raise FileNotFoundError(f"No samples selected from {manifest_path}")
    return samples


def write_tile_input_manifest(
    root: str | Path,
    sample_names: list[str] | None = None,
    *,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Write a resolved input manifest for the standard folder layout."""
    root = Path(root).expanduser().resolve()
    samples = discover_samples(root, sample_names)
    rows: list[dict[str, object]] = []
    for sample in samples:
        pixel_size = read_experiment_pixel_size(sample)
        cell_boundary_table = find_boundary_table(sample, "cell")
        nucleus_boundary_table = find_boundary_table(sample, "nucleus")
        cell_boundary_source = describe_boundary_source(
            sample,
            "cell",
            exported_table=cell_boundary_table,
        )
        nucleus_boundary_source = describe_boundary_source(
            sample,
            "nucleus",
            exported_table=nucleus_boundary_table,
        )
        rows.append(
            {
                "sample": sample.name,
                "output_dir": str(sample.base),
                "sample_dir": str(sample.base),
                "he_image": str(sample.he_image or ""),
                "cells": str(find_cell_table(sample) or ""),
                "outs_dir": str(sample.outs),
                "outs_zip": str(sample.outs_zip or ""),
                "alignment_csv": str(sample.alignment or ""),
                "manual_alignment_json": (
                    str(sample.manual_alignment)
                    if sample.manual_alignment.is_file()
                    else ""
                ),
                "pixel_size_um": pixel_size if pixel_size is not None else "",
                "cell_groups": str(sample.cell_groups or ""),
                "cell_boundaries": str(cell_boundary_table or ""),
                "nucleus_boundaries": str(nucleus_boundary_table or ""),
                "cell_boundary_source": cell_boundary_source.source,
                "cell_boundary_source_path": str(cell_boundary_source.path or ""),
                "cell_boundary_polygon_set": cell_boundary_source.polygon_set,
                "cell_boundary_detail": cell_boundary_source.detail,
                "nucleus_boundary_source": nucleus_boundary_source.source,
                "nucleus_boundary_source_path": str(nucleus_boundary_source.path or ""),
                "nucleus_boundary_polygon_set": nucleus_boundary_source.polygon_set,
                "nucleus_boundary_detail": nucleus_boundary_source.detail,
            }
        )
    manifest = pd.DataFrame(rows)
    destination = Path(output_path).expanduser().resolve() if output_path else root / "tile_input_manifest.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(destination, index=False)
    return manifest


def discover_samples(root: Path, requested: list[str] | None) -> list[SamplePaths]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Sample root does not exist: {root}")
    if requested:
        directories = [root / name for name in requested]
        missing = [path for path in directories if not path.is_dir()]
        if missing:
            raise FileNotFoundError(f"Missing sample directories: {missing}")
    else:
        directories = []
        for path in sorted(root.iterdir()):
            if not path.is_dir() or path.name.startswith("."):
                continue
            sample = build_sample_paths(path)
            output_markers = [
                path / f"{path.name}_tile_labels.csv",
                path / f"{path.name}_tile_labels_with_paths.csv",
                path / f"{path.name}_annotated_tile_reconstruction.png",
                path / f"{path.name}_pipeline_complete.json",
            ]
            if (
                sample.outs.is_dir()
                or sample.outs_zip
                or sample.he_image
                or any(marker.is_file() for marker in output_markers)
            ):
                directories.append(path)
    samples = [build_sample_paths(path) for path in directories]
    if not samples:
        raise FileNotFoundError(f"No Xenium sample folders found under {root}")
    return samples


def safe_extract(archive_path: Path, destination: Path, sample_name: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        for member in members:
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination_resolved):
                raise ValueError(f"Unsafe path in {archive_path}: {member.filename}")
        for member in tqdm(
            members,
            desc=f"{sample_name}: extracting",
            unit="file",
            dynamic_ncols=True,
        ):
            archive.extract(member, destination)


REQUIRED_ARCHIVE_BASENAMES = {
    "experiment.xenium",
    "cells.zarr.zip",
    "cells.parquet",
    "cells.csv.gz",
    "cells.csv",
    "analysis.zarr.zip",
}


def archive_member_is_required(filename: str) -> bool:
    """Return whether an outer XOA ZIP member is needed by this pipeline."""
    path = Path(filename)
    basename = path.name.lower()
    if basename in REQUIRED_ARCHIVE_BASENAMES:
        return True
    normalized = filename.replace("\\", "/").lower()
    return basename == "clusters.csv" and "graphclust" in normalized


def inspect_archives(
    root: str | Path,
    sample_names: list[str] | None = None,
) -> pd.DataFrame:
    """Inspect large outer archives without extracting their contents."""
    rows: list[dict[str, object]] = []
    for paths in discover_samples(Path(root), sample_names):
        row: dict[str, object] = {
            "sample": paths.name,
            "archive": str(paths.outs_zip or ""),
            "archive_size_gb": pd.NA,
            "experiment": False,
            "cells_zarr": False,
            "cells_table": False,
            "analysis_zarr": False,
            "graphclust_csv": False,
            "error": "",
        }
        if paths.outs_zip is None:
            row["error"] = "No *_outs.zip found; an extracted output may already exist"
            rows.append(row)
            continue
        try:
            row["archive_size_gb"] = paths.outs_zip.stat().st_size / 1_000_000_000
            with zipfile.ZipFile(paths.outs_zip) as archive:
                names = [member.filename.replace("\\", "/").lower() for member in archive.infolist()]
            basenames = {Path(name).name for name in names}
            row["experiment"] = "experiment.xenium" in basenames
            row["cells_zarr"] = "cells.zarr.zip" in basenames
            row["cells_table"] = bool({"cells.parquet", "cells.csv.gz", "cells.csv"} & basenames)
            row["analysis_zarr"] = "analysis.zarr.zip" in basenames
            row["graphclust_csv"] = any(
                Path(name).name == "clusters.csv" and "graphclust" in name
                for name in names
            )
        except (OSError, zipfile.BadZipFile) as error:
            row["error"] = str(error)
        rows.append(row)
    return pd.DataFrame(rows)


def safe_extract_required(
    archive_path: Path,
    destination: Path,
    sample_name: str,
) -> None:
    """Safely extract only cell, experiment, and clustering data."""
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            member
            for member in archive.infolist()
            if not member.is_dir() and archive_member_is_required(member.filename)
        ]
        if not members:
            raise FileNotFoundError(
                f"No supported Xenium cell files were found inside {archive_path}"
            )
        for member in members:
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination_resolved):
                raise ValueError(f"Unsafe path in {archive_path}: {member.filename}")
        selected_bytes = sum(member.file_size for member in members)
        print(
            f"[{sample_name}] selected {len(members)} archive member(s), "
            f"{selected_bytes / 1_000_000_000:.2f} GB uncompressed"
        )
        for member in tqdm(
            members,
            desc=f"{sample_name}: extracting required files",
            unit="file",
            dynamic_ncols=True,
        ):
            archive.extract(member, destination)


def prepare_outs(paths: SamplePaths, *, selective_archive_extract: bool = False) -> None:
    """Extract the XOA archive once, then locate its actual output root."""
    if paths.cell_table is not None and paths.cell_table.is_file():
        return
    if (paths.outs / "experiment.xenium").is_file():
        if not selective_archive_extract:
            return
        has_cells = any(
            (paths.outs / filename).is_file()
            for filename in ("cells.zarr.zip", "cells.parquet", "cells.csv.gz", "cells.csv")
        )
        if has_cells:
            return
    if paths.outs_zip is None or not paths.outs_zip.is_file():
        return
    target = paths.base / f"{paths.name}_outs"
    if selective_archive_extract:
        print(f"[{paths.name}] extracting required data from {paths.outs_zip.name} -> {target}")
        safe_extract_required(paths.outs_zip, target, paths.name)
    else:
        print(f"[{paths.name}] extracting {paths.outs_zip.name} -> {target}")
        safe_extract(paths.outs_zip, target, paths.name)
    paths.outs = locate_outs(paths.base)


def find_outs_file(paths: SamplePaths, filename: str) -> Path | None:
    direct = paths.outs / filename
    if direct.is_file():
        return direct
    matches = sorted(paths.outs.rglob(filename)) if paths.outs.is_dir() else []
    return matches[0] if matches else None


def find_cell_table(paths: SamplePaths) -> Path | None:
    if paths.cell_table is not None and paths.cell_table.is_file():
        return paths.cell_table
    for filename in ("cells.parquet", "cells.csv.gz", "cells.csv"):
        found = find_outs_file(paths, filename)
        if found:
            return found
    return None


def find_boundary_table(paths: SamplePaths, boundary_kind: str) -> Path | None:
    if boundary_kind not in {"cell", "nucleus"}:
        raise ValueError("boundary_kind must be 'cell' or 'nucleus'")
    direct = paths.cell_boundaries if boundary_kind == "cell" else paths.nucleus_boundaries
    if direct is not None and direct.is_file():
        return direct
    for filename in (
        f"{boundary_kind}_boundaries.parquet.gz",
        f"{boundary_kind}_boundaries.parquet",
        f"{boundary_kind}_boundaries.csv.gz",
        f"{boundary_kind}_boundaries.csv",
    ):
        found = find_outs_file(paths, filename)
        if found:
            return found
    return None


def _has_current_polygon_set(polygons: object) -> bool:
    return all(name in polygons for name in ("cell_index", "num_vertices", "vertices"))


def _decode_polygon_set_names(polygon_set_names: Iterable[object]) -> list[str]:
    return [
        value.decode() if isinstance(value, bytes) else str(value)
        for value in polygon_set_names
    ]


def xoa1_cell_polygon_set_index(
    number_polygon_sets: int,
    polygon_set_names: Iterable[object] = (),
) -> int:
    names = _decode_polygon_set_names(polygon_set_names)
    cell_sets = [
        index
        for index, name in enumerate(names)
        if "cell" in name.lower() and "nucleus" not in name.lower()
    ]
    if cell_sets:
        return cell_sets[0]
    return 1 if number_polygon_sets >= 2 else 0


def describe_boundary_source(
    paths: SamplePaths,
    boundary_kind: str,
    *,
    exported_table: Path | None = None,
) -> BoundarySource:
    """Describe where a boundary kind can be loaded from without loading vertices."""
    if boundary_kind not in {"cell", "nucleus"}:
        raise ValueError("boundary_kind must be 'cell' or 'nucleus'")
    exported_table = exported_table or find_boundary_table(paths, boundary_kind)
    if exported_table is not None:
        return BoundarySource("exported_table", exported_table)

    cells_zarr_path = find_outs_file(paths, "cells.zarr.zip")
    if cells_zarr_path is None:
        return BoundarySource("missing", detail="cells.zarr.zip absent")

    try:
        with ZipStore(str(cells_zarr_path), mode="r") as store:
            root = zarr.open_group(store=store, mode="r")
            if "polygon_sets" in root:
                polygon_set = "1" if boundary_kind == "cell" else "0"
                polygon_sets = root["polygon_sets"]
                if polygon_set in polygon_sets and _has_current_polygon_set(polygon_sets[polygon_set]):
                    return BoundarySource(
                        "embedded_cells_zarr",
                        cells_zarr_path,
                        polygon_set=polygon_set,
                    )
                return BoundarySource(
                    "missing",
                    cells_zarr_path,
                    detail=f"polygon_sets/{polygon_set} absent or incomplete",
                )

            if "polygon_num_vertices" in root and "polygon_vertices" in root:
                num_vertices = root["polygon_num_vertices"]
                vertices = root["polygon_vertices"]
                num_vertices_ndim = len(num_vertices.shape)
                vertices_ndim = len(vertices.shape)
                if num_vertices_ndim == 1 and vertices_ndim == 2:
                    if boundary_kind == "cell":
                        return BoundarySource(
                            "embedded_cells_zarr",
                            cells_zarr_path,
                            polygon_set="xoa1_single_set",
                        )
                    return BoundarySource(
                        "missing",
                        cells_zarr_path,
                        detail="XOA 1.x Zarr has one polygon set",
                    )
                if num_vertices_ndim == 2 and vertices_ndim == 3:
                    polygon_set_index = (
                        0
                        if boundary_kind == "nucleus"
                        else xoa1_cell_polygon_set_index(
                            int(num_vertices.shape[0]),
                            root.attrs.get("polygon_set_names", ()),
                        )
                    )
                    if 0 <= polygon_set_index < int(num_vertices.shape[0]):
                        return BoundarySource(
                            "embedded_cells_zarr",
                            cells_zarr_path,
                            polygon_set=f"xoa1_axis0_{polygon_set_index}",
                        )
                return BoundarySource(
                    "missing",
                    cells_zarr_path,
                    detail=(
                        "unsupported XOA 1.x polygon shapes: "
                        f"num_vertices={num_vertices.shape}, vertices={vertices.shape}"
                    ),
                )
            return BoundarySource(
                "missing",
                cells_zarr_path,
                detail="no supported polygon arrays",
            )
    except Exception as error:
        return BoundarySource("unreadable", cells_zarr_path, detail=str(error))


def read_experiment_pixel_size(paths: SamplePaths) -> float | None:
    experiment_path = find_outs_file(paths, "experiment.xenium")
    if experiment_path is None:
        return None
    with experiment_path.open() as handle:
        experiment = json.load(handle)
    return float(experiment["pixel_size"])


def validate_sample(
    paths: SamplePaths,
    *,
    selective_archive_extract: bool = False,
    require_alignment: bool | None = None,
    require_cell_groups: bool = False,
    allow_manual_alignment: bool = True,
) -> dict[str, str]:
    prepare_outs(paths, selective_archive_extract=selective_archive_extract)
    refreshed = build_sample_paths(paths.base)
    paths.he_image = paths.he_image or refreshed.he_image
    paths.alignment = paths.alignment or refreshed.alignment
    paths.cell_groups = paths.cell_groups or refreshed.cell_groups
    experiment = find_outs_file(paths, "experiment.xenium")
    cells_zarr = find_outs_file(paths, "cells.zarr.zip")
    cell_table = find_cell_table(paths)
    missing: list[str] = []
    if paths.he_image is None or not paths.he_image.is_file():
        missing.append("H&E OME-TIFF (*he*image*.ome.tif)")
    if experiment is None and paths.pixel_size_um is None:
        missing.append("experiment.xenium")
    if cells_zarr is None and cell_table is None:
        missing.append("cells.zarr.zip or cells.parquet/cells.csv.gz")
    alignment_required = (
        paths.he_image is not None
        and "unaligned" in paths.he_image.name.lower()
        if require_alignment is None
        else require_alignment
    )
    has_manual_alignment = allow_manual_alignment and paths.manual_alignment.is_file()
    has_affine_alignment = paths.alignment is not None and paths.alignment.is_file()
    if alignment_required and not has_affine_alignment and not has_manual_alignment:
        missing.append(
            "H&E alignment matrix"
            + (" or selected manual alignment JSON" if allow_manual_alignment else "")
        )
    if require_cell_groups and (paths.cell_groups is None or not paths.cell_groups.is_file()):
        missing.append("*_cell_groups.csv")
    if missing:
        raise FileNotFoundError(
            f"Missing inputs for {paths.name}:\n" + "\n".join(f"  - {x}" for x in missing)
        )
    return {
        "he_image": str(paths.he_image),
        "alignment": str(
            paths.alignment
            or (
                paths.manual_alignment
                if has_manual_alignment
                else "pre-aligned image"
            )
        ),
        "cells": str(cells_zarr or cell_table),
        "cell_groups": str(paths.cell_groups or ""),
    }


def xenium_int_cell_id_to_str(cell_id_array: np.ndarray) -> list[str]:
    """Decode cell IDs from both XOA 1.x and XOA 2.x Zarr layouts.

    Current XOA versions store ``(prefix, dataset_suffix)`` pairs.  The public
    XOA 1.0.2 breast bundles store only the prefix as a one-dimensional uint32
    array; their omitted dataset suffix is the documented default value, 1.
    A transposed two-row representation is accepted as well.
    """
    cell_ids = np.asarray(cell_id_array)
    if cell_ids.ndim == 1:
        prefixes = cell_ids
        suffixes = np.ones(len(cell_ids), dtype=np.uint32)
    elif cell_ids.ndim == 2 and cell_ids.shape[1] == 2:
        prefixes = cell_ids[:, 0]
        suffixes = cell_ids[:, 1]
    elif cell_ids.ndim == 2 and cell_ids.shape[0] == 2:
        prefixes = cell_ids[0]
        suffixes = cell_ids[1]
    else:
        raise ValueError(
            "Unsupported cells.zarr cell_id shape "
            f"{cell_ids.shape}; expected (n,), (n, 2), or (2, n)"
        )

    output: list[str] = []
    for prefix, suffix in zip(prefixes, suffixes):
        hexadecimal = f"{int(prefix):08x}"
        letters = "".join(chr(ord("a") + int(character, 16)) for character in hexadecimal)
        output.append(f"{letters}-{int(suffix)}")
    return output


def load_cells(paths: SamplePaths) -> tuple[pd.DataFrame, bool]:
    cells_zarr_path = None if paths.cell_table is not None else find_outs_file(paths, "cells.zarr.zip")
    if cells_zarr_path:
        with ZipStore(str(cells_zarr_path), mode="r") as store:
            root = zarr.open_group(store=store, mode="r")
            cell_ids = xenium_int_cell_id_to_str(root["cell_id"][:])
            summary = root["cell_summary"][:]
            columns = list(root["cell_summary"].attrs.get("column_names", []))
        if len(columns) != summary.shape[1]:
            fallback = [
                "cell_centroid_x",
                "cell_centroid_y",
                "cell_area",
                "nucleus_centroid_x",
                "nucleus_centroid_y",
                "nucleus_area",
                "z_level",
                "nucleus_count",
            ]
            columns = fallback[: summary.shape[1]]
        cells = pd.DataFrame(summary, columns=columns)
        cells.insert(0, "cell_id", cell_ids)
        zarr_order = True
    else:
        table_path = find_cell_table(paths)
        if table_path is None:
            raise FileNotFoundError("No cell table was found")
        if table_path.name.endswith(".parquet.gz"):
            with gzip.open(table_path, "rb") as handle:
                cells = pd.read_parquet(BytesIO(handle.read()))
        elif table_path.suffix == ".parquet":
            cells = pd.read_parquet(table_path)
        else:
            cells = pd.read_csv(table_path)
        cells = cells.rename(
            columns={
                "x_centroid": "cell_centroid_x",
                "y_centroid": "cell_centroid_y",
                "barcode": "cell_id",
            }
        )
        zarr_order = False
    required = {"cell_id", "cell_centroid_x", "cell_centroid_y"}
    missing = required - set(cells.columns)
    if missing:
        raise ValueError(f"Cell data is missing columns: {sorted(missing)}")
    if cells["cell_id"].duplicated().any():
        raise ValueError("Duplicate Xenium cell IDs detected")
    cells.insert(0, "cell_index", np.arange(len(cells), dtype=np.int64))
    return cells, zarr_order


def custom_cell_labels(
    paths: SamplePaths,
    cells: pd.DataFrame,
    *,
    missing_label: str = "Unassigned",
) -> pd.Series | None:
    if paths.cell_groups is None:
        return None
    groups = pd.read_csv(paths.cell_groups)
    required = {"cell_id", "group"}
    if not required.issubset(groups.columns):
        raise ValueError(f"{paths.cell_groups} must contain cell_id and group columns")
    if groups["cell_id"].duplicated().any():
        raise ValueError(f"Duplicate IDs in {paths.cell_groups}")
    mapping = groups.set_index("cell_id")["group"]
    return cells["cell_id"].map(mapping).fillna(missing_label).astype(str)


def cluster_labels_from_csv(
    paths: SamplePaths,
    cells: pd.DataFrame,
    *,
    missing_label: str = "Unassigned",
) -> pd.Series | None:
    if not paths.outs.is_dir():
        return None
    candidates = sorted(paths.outs.rglob("*graphclust*/clusters.csv"))
    if not candidates:
        return None
    clusters = pd.read_csv(candidates[0])
    lower = {str(column).lower(): column for column in clusters.columns}
    id_column = lower.get("barcode") or lower.get("cell_id")
    cluster_column = lower.get("cluster")
    if id_column is None or cluster_column is None:
        return None
    mapping = clusters.set_index(id_column)[cluster_column]
    labels = cells["cell_id"].map(mapping)
    return labels.map(lambda value: f"Cluster {int(value)}" if pd.notna(value) else missing_label)


def cluster_labels_from_zarr(
    paths: SamplePaths,
    number_cells: int,
    *,
    missing_label: str = "Unassigned",
) -> pd.Series | None:
    analysis_path = find_outs_file(paths, "analysis.zarr.zip")
    if analysis_path is None:
        return None
    with ZipStore(str(analysis_path), mode="r") as store:
        root = zarr.open_group(store=store, mode="r")
        if "cell_groups" not in root:
            return None
        groups = root["cell_groups"]
        grouping_names = [str(name) for name in groups.attrs.get("grouping_names", [])]
        if not grouping_names:
            return None
        graph_indices = [index for index, name in enumerate(grouping_names) if "graphclust" in name]
        grouping_index = graph_indices[0] if graph_indices else 0
        key = str(grouping_index)
        if key not in groups:
            return None
        indices = np.asarray(groups[key]["indices"][:], dtype=np.int64)
        indptr = np.asarray(groups[key]["indptr"][:], dtype=np.int64)
        all_group_names = groups.attrs.get("group_names", [])
        names = list(all_group_names[grouping_index]) if len(all_group_names) > grouping_index else []

    labels = np.full(number_cells, missing_label, dtype=object)
    for cluster_index, start in enumerate(indptr):
        end = int(indptr[cluster_index + 1]) if cluster_index + 1 < len(indptr) else len(indices)
        name = str(names[cluster_index]) if cluster_index < len(names) else f"Cluster {cluster_index + 1}"
        cell_indices = indices[int(start) : end]
        valid = cell_indices[(cell_indices >= 0) & (cell_indices < number_cells)]
        labels[valid] = name
    return pd.Series(labels)


def add_cell_labels(
    paths: SamplePaths,
    cells: pd.DataFrame,
    zarr_order: bool,
    label_mode: str,
    *,
    missing_label: str = "Unassigned",
    cell_groups_label_source: str = "custom_cell_groups",
) -> tuple[pd.DataFrame, str]:
    labels: pd.Series | None = None
    source = "none"
    if label_mode == "auto":
        labels = custom_cell_labels(paths, cells, missing_label=missing_label)
        if labels is not None:
            source = cell_groups_label_source
    if labels is None and label_mode != "none":
        if zarr_order:
            labels = cluster_labels_from_zarr(paths, len(cells), missing_label=missing_label)
            if labels is not None:
                source = "xoa_graphclust_zarr"
        if labels is None:
            labels = cluster_labels_from_csv(paths, cells, missing_label=missing_label)
            if labels is not None:
                source = "xoa_graphclust_csv"
    if labels is None:
        automatic = next(
            (
                name
                for name in ("cell_type", "annotation", "group", "cluster")
                if name in cells.columns
            ),
            None,
        )
        if automatic is not None:
            labels = cells[automatic]
            source = f"cells_table:{automatic}"
    if labels is None:
        labels = pd.Series(np.full(len(cells), missing_label, dtype=object))
    cells = cells.copy()
    cells["cell_type"] = labels.to_numpy()
    cells["label_source"] = source
    print(f"[{paths.name}] cells={len(cells):,}, label_source={source}")
    return cells, source




def load_image_metadata(paths: SamplePaths) -> ImageMetadata:
    if paths.he_image is None:
        raise FileNotFoundError("H&E image was not resolved")
    with tifffile.TiffFile(paths.he_image) as tif:
        series = tif.series[0]
        full_shape = tuple(series.levels[0].shape)
        thumbnail = ensure_rgb(series.levels[-1].asarray())
        physical_x, physical_y = ome_physical_pixel_size(tif.ome_metadata)
    full_height, full_width = get_hw(full_shape)
    thumbnail_height, thumbnail_width = thumbnail.shape[:2]
    return ImageMetadata(
        thumbnail=thumbnail,
        full_height=full_height,
        full_width=full_width,
        thumbnail_height=thumbnail_height,
        thumbnail_width=thumbnail_width,
        thumbnail_scale_x=thumbnail_width / full_width,
        thumbnail_scale_y=thumbnail_height / full_height,
        physical_size_x_um=physical_x,
        physical_size_y_um=physical_y,
    )


def experiment_pixel_size(paths: SamplePaths) -> float:
    if paths.pixel_size_um is not None:
        return float(paths.pixel_size_um)
    pixel_size = read_experiment_pixel_size(paths)
    if pixel_size is None:
        raise FileNotFoundError("experiment.xenium was not found")
    return pixel_size


def align_xy_to_he(
    xy_um: np.ndarray,
    paths: SamplePaths,
    metadata: ImageMetadata,
    xenium_pixel_size_um: float,
) -> tuple[np.ndarray, str, np.ndarray | None]:
    if paths.alignment is not None:
        he_to_xenium = np.loadtxt(paths.alignment, delimiter=",")
        if he_to_xenium.shape != (3, 3) or not np.allclose(he_to_xenium[2], [0, 0, 1]):
            raise ValueError(f"Invalid 3x3 affine matrix: {paths.alignment}")
        xenium_to_he = np.linalg.inv(he_to_xenium)
        xy_xenium_px = np.asarray(xy_um, dtype=float) / xenium_pixel_size_um
        return apply_affine(xy_xenium_px, xenium_to_he), "affine_matrix", xenium_to_he

    if paths.manual_alignment.is_file():
        with paths.manual_alignment.open(encoding="utf-8") as handle:
            manual = json.load(handle)
        xenium_to_he = np.asarray(
            manual["matrix_xenium_full_pixel_to_he_full_pixel"],
            dtype=float,
        )
        if xenium_to_he.shape != (3, 3):
            raise ValueError(f"Invalid manual 3x3 transform: {paths.manual_alignment}")
        xy_xenium_px = np.asarray(xy_um, dtype=float) / xenium_pixel_size_um
        mode = "manual_landmarks"
        return apply_affine(xy_xenium_px, xenium_to_he), mode, xenium_to_he

    pixel_x = metadata.physical_size_x_um or xenium_pixel_size_um
    pixel_y = metadata.physical_size_y_um or xenium_pixel_size_um
    xy = np.asarray(xy_um, dtype=float) / np.array([pixel_x, pixel_y])
    return xy, "prealigned_he", None


def align_cells(
    paths: SamplePaths,
    cells: pd.DataFrame,
    metadata: ImageMetadata,
) -> tuple[pd.DataFrame, float, str, np.ndarray | None]:
    pixel_size = experiment_pixel_size(paths)
    xy_um = cells[["cell_centroid_x", "cell_centroid_y"]].to_numpy()
    xy_he, mode, xenium_to_he = align_xy_to_he(xy_um, paths, metadata, pixel_size)
    cells = cells.copy()
    cells["he_x_px"] = xy_he[:, 0]
    cells["he_y_px"] = xy_he[:, 1]
    cells["inside_he_image"] = (
        cells["he_x_px"].between(0, metadata.full_width, inclusive="left")
        & cells["he_y_px"].between(0, metadata.full_height, inclusive="left")
    )
    if {"nucleus_centroid_x", "nucleus_centroid_y"}.issubset(cells.columns):
        nucleus_xy = cells[["nucleus_centroid_x", "nucleus_centroid_y"]].to_numpy(dtype=float)
        nucleus_he = np.full((len(cells), 2), np.nan, dtype=float)
        valid_nuclei = np.isfinite(nucleus_xy).all(axis=1)
        if valid_nuclei.any():
            nucleus_he[valid_nuclei] = boundary_to_he(
                nucleus_xy[valid_nuclei],
                paths,
                metadata,
                pixel_size,
                xenium_to_he,
            )
        cells["nucleus_he_x_px"] = nucleus_he[:, 0]
        cells["nucleus_he_y_px"] = nucleus_he[:, 1]
        cells["nucleus_inside_he_image"] = (
            cells["nucleus_he_x_px"].between(0, metadata.full_width, inclusive="left")
            & cells["nucleus_he_y_px"].between(0, metadata.full_height, inclusive="left")
        )
    inside = float(cells["inside_he_image"].mean())
    print(f"[{paths.name}] alignment={mode}, cells inside H&E={inside:.2%}")
    if not cells["inside_he_image"].any():
        raise ValueError("No cell centroids fall inside the aligned H&E image")
    return cells, pixel_size, mode, xenium_to_he


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


def boundary_to_he(
    xy_um: np.ndarray,
    paths: SamplePaths,
    metadata: ImageMetadata,
    xenium_pixel_size_um: float,
    xenium_to_he: np.ndarray | None,
) -> np.ndarray:
    if xenium_to_he is not None:
        return apply_affine(np.asarray(xy_um) / xenium_pixel_size_um, xenium_to_he)
    pixel_x = metadata.physical_size_x_um or xenium_pixel_size_um
    pixel_y = metadata.physical_size_y_um or xenium_pixel_size_um
    return np.asarray(xy_um) / np.array([pixel_x, pixel_y])


def select_xoa1_cell_boundaries(
    num_vertices: np.ndarray,
    vertices: np.ndarray,
    polygon_set_names: Iterable[object] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Select cell polygons from the XOA 1.x boundary arrays.

    Early Xenium bundles store arrays as ``(polygon_set, cell)`` and
    ``(polygon_set, cell, flattened_xy)``.  Polygon set 0 is the nucleus and
    set 1 is the cell.  Some XOA 1.x bundles contain only one polygon set,
    represented directly as ``(cell,)`` and ``(cell, flattened_xy)``.
    """
    num_vertices = np.asarray(num_vertices)
    vertices = np.asarray(vertices)
    if num_vertices.ndim == 1 and vertices.ndim == 2:
        if len(num_vertices) != len(vertices):
            raise ValueError(
                "XOA 1.x boundary arrays disagree on the number of cells: "
                f"{num_vertices.shape} versus {vertices.shape}"
            )
        return num_vertices, vertices

    if num_vertices.ndim != 2 or vertices.ndim != 3:
        raise ValueError(
            "Unsupported XOA 1.x boundary shapes: "
            f"num_vertices={num_vertices.shape}, vertices={vertices.shape}"
        )
    if num_vertices.shape != vertices.shape[:2]:
        raise ValueError(
            "XOA 1.x boundary arrays disagree on polygon-set/cell dimensions: "
            f"{num_vertices.shape} versus {vertices.shape}"
        )

    cell_set_index = xoa1_cell_polygon_set_index(
        int(num_vertices.shape[0]),
        polygon_set_names,
    )

    if cell_set_index >= num_vertices.shape[0]:
        raise ValueError(
            f"Cell polygon set {cell_set_index} is absent from {num_vertices.shape[0]} sets"
        )
    return num_vertices[cell_set_index], vertices[cell_set_index]


def load_boundaries(
    paths: SamplePaths,
    cells_or_number_cells: pd.DataFrame | int,
    *,
    boundary_kind: str = "cell",
) -> BoundaryData | None:
    if boundary_kind not in {"cell", "nucleus"}:
        raise ValueError("boundary_kind must be 'cell' or 'nucleus'")
    if isinstance(cells_or_number_cells, pd.DataFrame):
        cells = cells_or_number_cells
        if "cell_id" not in cells.columns:
            raise ValueError("Boundary loading by exported CSV requires a cell_id column")
        if "cell_index" in cells.columns and cells["cell_index"].notna().any():
            cell_indices = pd.to_numeric(cells["cell_index"], errors="raise").astype(np.int64)
            number_cells = int(cell_indices.max()) + 1 if len(cell_indices) else 0
        else:
            cell_indices = pd.Series(np.arange(len(cells), dtype=np.int64), index=cells.index)
            number_cells = len(cells)
        cell_index_by_id = dict(zip(cells["cell_id"].astype(str), cell_indices.astype(np.int64)))

        direct_boundary = paths.cell_boundaries if boundary_kind == "cell" else paths.nucleus_boundaries
        boundary_table_path = (
            direct_boundary
            if direct_boundary is not None and direct_boundary.is_file()
            else find_boundary_table(paths, boundary_kind)
        )
        if boundary_table_path is not None:
            if boundary_table_path.name.endswith(".parquet.gz"):
                with gzip.open(boundary_table_path, "rb") as handle:
                    boundary_table = pd.read_parquet(BytesIO(handle.read()))
            elif boundary_table_path.suffix == ".parquet":
                boundary_table = pd.read_parquet(boundary_table_path)
            else:
                boundary_table = pd.read_csv(boundary_table_path)
            required = {"cell_id", "vertex_x", "vertex_y"}
            missing = required - set(boundary_table.columns)
            if missing:
                raise ValueError(
                    f"{boundary_table_path} is missing boundary columns: {sorted(missing)}"
                )
            if "vertex_index" in boundary_table.columns:
                boundary_table = boundary_table.sort_values(
                    ["cell_id", "vertex_index"],
                    kind="mergesort",
                )
            boundary_table["cell_id"] = boundary_table["cell_id"].astype(str)
            grouped = boundary_table.groupby("cell_id", sort=False, observed=True)
            counts = grouped.size().astype(np.int64)
            max_vertices = int(counts.max()) if len(counts) else 0
            if max_vertices < 3:
                print(f"[{paths.name}] exported {boundary_kind} boundaries are empty")
                return None
            vertices = np.zeros((len(counts), 2 * max_vertices), dtype=np.float32)
            num_vertices = counts.to_numpy(dtype=np.int64)
            row_for_cell = np.full(number_cells, -1, dtype=np.int64)
            for row_index, (cell_id, group) in enumerate(grouped):
                xy = group[["vertex_x", "vertex_y"]].to_numpy(dtype=np.float32)
                vertices[row_index, : 2 * len(xy)] = xy.reshape(-1)
                cell_index = cell_index_by_id.get(str(cell_id))
                if cell_index is not None and 0 <= cell_index < number_cells:
                    row_for_cell[int(cell_index)] = row_index
            mapped = int((row_for_cell >= 0).sum())
            if mapped:
                print(
                    f"[{paths.name}] loaded {mapped:,} exported {boundary_kind}-boundary polygons "
                    f"from {boundary_table_path.name}"
                )
                return BoundaryData(num_vertices, vertices, row_for_cell)
            print(
                f"[{paths.name}] exported {boundary_kind} boundaries did not match loaded cell IDs; "
                "falling back to cells.zarr.zip"
            )
    else:
        number_cells = int(cells_or_number_cells)

    cells_zarr_path = find_outs_file(paths, "cells.zarr.zip")
    if cells_zarr_path is None:
        print(f"[{paths.name}] cells.zarr.zip absent; boundary overlays will use centroids only")
        return None
    with ZipStore(str(cells_zarr_path), mode="r") as store:
        root = zarr.open_group(store=store, mode="r")
        if "polygon_sets" in root:
            polygon_set = "1" if boundary_kind == "cell" else "0"
            if polygon_set not in root["polygon_sets"]:
                print(
                    f"[{paths.name}] {boundary_kind} boundary polygon set {polygon_set} is absent; "
                    "boundary overlays will use centroids only"
                )
                return None
            polygons = root["polygon_sets"][polygon_set]
            missing_arrays = [
                name
                for name in ("cell_index", "num_vertices", "vertices")
                if name not in polygons
            ]
            if missing_arrays:
                print(
                    f"[{paths.name}] {boundary_kind} boundary polygon set {polygon_set} "
                    f"is missing arrays {missing_arrays}; boundary overlays will use centroids only"
                )
                return None
            indices = polygons["cell_index"][:]
            num_vertices = polygons["num_vertices"][:]
            vertices = polygons["vertices"][:]
        elif "polygon_num_vertices" in root and "polygon_vertices" in root:
            # XOA 1.x layout. Usually axis 0 contains nucleus then cell.
            try:
                if boundary_kind == "nucleus":
                    all_num_vertices = root["polygon_num_vertices"][:]
                    all_vertices = root["polygon_vertices"][:]
                    num_vertices, vertices = all_num_vertices[0], all_vertices[0]
                else:
                    num_vertices, vertices = select_xoa1_cell_boundaries(
                        root["polygon_num_vertices"][:],
                        root["polygon_vertices"][:],
                        root.attrs.get("polygon_set_names", ()),
                    )
            except ValueError as error:
                print(
                    f"[{paths.name}] {error}; boundary overlays will use centroids only"
                )
                return None
            except IndexError:
                print(
                    f"[{paths.name}] {boundary_kind} boundary set is absent; "
                    "boundary overlays will use centroids only"
                )
                return None
            indices = np.arange(len(num_vertices), dtype=np.int64)
        else:
            print(
                f"[{paths.name}] no supported {boundary_kind}-boundary polygons; "
                "boundary overlays will use centroids only"
            )
            return None
    row_for_cell = np.full(number_cells, -1, dtype=np.int64)
    valid = (indices >= 0) & (indices < number_cells)
    row_for_cell[indices[valid]] = np.arange(len(indices), dtype=np.int64)[valid]
    return BoundaryData(num_vertices, vertices, row_for_cell)


def boundary_in_he_pixels(
    cell_index: int,
    boundaries: BoundaryData | None,
    paths: SamplePaths,
    metadata: ImageMetadata,
    pixel_size: float,
    xenium_to_he: np.ndarray | None,
) -> np.ndarray | None:
    if boundaries is None or not 0 <= cell_index < len(boundaries.polygon_row_for_cell):
        return None
    row = int(boundaries.polygon_row_for_cell[cell_index])
    if row < 0:
        return None
    count = int(boundaries.num_vertices[row])
    if count < 3:
        return None
    xy_um = boundaries.vertices[row, : 2 * count].reshape(count, 2)
    return boundary_to_he(xy_um, paths, metadata, pixel_size, xenium_to_he)


def save_alignment_plot(
    paths: SamplePaths,
    cells: pd.DataFrame,
    metadata: ImageMetadata,
    alignment_mode: str,
) -> None:
    inside = cells.loc[cells["inside_he_image"]]
    sample = inside.sample(n=min(30_000, len(inside)), random_state=1)
    sample_marker_x, sample_marker_y, marker_label = marker_he_coordinates(sample)
    if {"nucleus_centroid_x", "nucleus_centroid_y"}.issubset(sample.columns):
        xenium_x = sample["nucleus_centroid_x"].where(
            sample["nucleus_centroid_x"].notna(),
            sample["cell_centroid_x"],
        )
        xenium_y = sample["nucleus_centroid_y"].where(
            sample["nucleus_centroid_y"].notna(),
            sample["cell_centroid_y"],
        )
    else:
        xenium_x = sample["cell_centroid_x"]
        xenium_y = sample["cell_centroid_y"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), constrained_layout=True)
    axes[0].scatter(xenium_x, xenium_y, s=0.2, alpha=0.5)
    axes[0].invert_yaxis()
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("X (um)")
    axes[0].set_ylabel("Y (um)")
    axes[0].set_title(f"Xenium {marker_label.lower()}s")
    axes[1].imshow(metadata.thumbnail)
    axes[1].set_title("H&E")
    axes[1].axis("off")
    axes[2].imshow(metadata.thumbnail)
    axes[2].scatter(
        sample_marker_x * metadata.thumbnail_scale_x,
        sample_marker_y * metadata.thumbnail_scale_y,
        s=0.2,
        alpha=0.4,
        color="deepskyblue",
    )
    axes[2].set_xlim(0, metadata.thumbnail_width)
    axes[2].set_ylim(metadata.thumbnail_height, 0)
    axes[2].set_title(f"{marker_label}s over H&E ({alignment_mode})")
    axes[2].axis("off")
    fig.savefig(paths.base / f"{paths.name}_xenium_he_alignment.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_cluster_plots(paths: SamplePaths, cells: pd.DataFrame, metadata: ImageMetadata) -> None:
    inside = cells.loc[cells["inside_he_image"]]
    sample = inside.sample(n=min(100_000, len(inside)), random_state=1)
    sample_marker_x, sample_marker_y, _ = marker_he_coordinates(sample)
    colors = create_color_map(inside["cell_type"])
    point_colors = [np.array(colors[str(value)]) / 255 for value in sample["cell_type"]]
    counts = inside["cell_type"].value_counts().sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    axes[0].imshow(metadata.thumbnail)
    axes[0].scatter(
        sample_marker_x * metadata.thumbnail_scale_x,
        sample_marker_y * metadata.thumbnail_scale_y,
        c=point_colors,
        s=0.25,
        alpha=0.55,
    )
    axes[0].set_xlim(0, metadata.thumbnail_width)
    axes[0].set_ylim(metadata.thumbnail_height, 0)
    axes[0].set_title("Cell labels over H&E")
    axes[0].axis("off")
    axes[1].barh(counts.index.astype(str), counts.values, color=[np.array(colors[x]) / 255 for x in counts.index])
    axes[1].set_xlabel("Cells inside H&E")
    axes[1].set_title("Cell label counts")
    fig.savefig(paths.base / f"{paths.name}_cell_clusters.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def he_tile_area_um2(
    metadata: ImageMetadata,
    tile_size_px: int,
    xenium_pixel_size_um: float,
    xenium_to_he: np.ndarray | None,
) -> float:
    """Return the physical area represented by one square H&E tile."""
    if xenium_to_he is not None:
        he_to_xenium = np.linalg.inv(xenium_to_he)
        pixel_area_scale = abs(float(np.linalg.det(he_to_xenium[:2, :2])))
        if not np.isfinite(pixel_area_scale) or pixel_area_scale <= 0:
            raise ValueError("Could not estimate tile area from the alignment transform")
        he_pixel_area_um2 = pixel_area_scale * xenium_pixel_size_um**2
    else:
        pixel_x = metadata.physical_size_x_um or xenium_pixel_size_um
        pixel_y = metadata.physical_size_y_um or xenium_pixel_size_um
        he_pixel_area_um2 = pixel_x * pixel_y
    return float(tile_size_px * tile_size_px * he_pixel_area_um2)


def create_tile_labels(
    paths: SamplePaths,
    cells: pd.DataFrame,
    metadata: ImageMetadata,
    config: PipelineConfig,
    pixel_size: float,
    xenium_to_he: np.ndarray | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_columns = metadata.full_width // config.tile_size_px
    n_rows = metadata.full_height // config.tile_size_px
    if n_columns <= 0 or n_rows <= 0:
        raise ValueError("H&E image is smaller than one complete tile")

    grid_rows, grid_cols = np.indices((n_rows, n_columns))
    tile_metrics = pd.DataFrame(
        {
            "tile_row": grid_rows.ravel(),
            "tile_col": grid_cols.ravel(),
        }
    )
    tile_metrics.insert(0, "sample", paths.name)
    tile_metrics["x0_px"] = tile_metrics["tile_col"] * config.tile_size_px
    tile_metrics["y0_px"] = tile_metrics["tile_row"] * config.tile_size_px
    tile_metrics["x1_px"] = tile_metrics["x0_px"] + config.tile_size_px
    tile_metrics["y1_px"] = tile_metrics["y0_px"] + config.tile_size_px

    tile_cells = cells.loc[cells["inside_he_image"]].copy()
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

    tile_area = he_tile_area_um2(metadata, config.tile_size_px, pixel_size, xenium_to_he)
    tile_metrics["tile_area_um2"] = tile_area
    if "cell_area" in tile_cells.columns:
        tile_cells["cell_area"] = pd.to_numeric(tile_cells["cell_area"], errors="coerce").fillna(0)
        cell_area = (
            tile_cells.groupby(["tile_row", "tile_col"], observed=True)["cell_area"]
            .sum()
            .rename("cell_area_um2")
            .reset_index()
        )
        tile_metrics = tile_metrics.merge(cell_area, on=["tile_row", "tile_col"], how="left")
        tile_metrics["cell_area_um2"] = tile_metrics["cell_area_um2"].fillna(0.0)
        tile_metrics["cell_area_fraction"] = tile_metrics["cell_area_um2"] / tile_area
    else:
        if config.min_cell_area_fraction > 0:
            raise ValueError(
                "Cell data has no cell_area column, so min_cell_area_fraction cannot be applied"
            )
        tile_metrics["cell_area_um2"] = pd.NA
        tile_metrics["cell_area_fraction"] = pd.NA

    tile_metrics["meets_min_cells"] = tile_metrics["n_cells"] >= config.min_cells_per_tile
    if config.min_cell_area_fraction > 0:
        tile_metrics["meets_min_cell_area_fraction"] = (
            tile_metrics["cell_area_fraction"] >= config.min_cell_area_fraction
        )
    else:
        tile_metrics["meets_min_cell_area_fraction"] = True
    tile_metrics["passes_tile_filter"] = (
        tile_metrics["has_cells"]
        & tile_metrics["meets_min_cells"]
        & tile_metrics["meets_min_cell_area_fraction"]
    )
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
    tile_metrics.to_csv(paths.base / f"{paths.name}_tile_filter_metrics.csv", index=False)

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
    proportion_columns: list[str] = []
    for column in counts.columns:
        value = column.removeprefix("count__")
        proportion_column = f"proportion__{value}"
        labels[proportion_column] = labels[column] / labels["n_cells"]
        proportion_columns.append(proportion_column)
    coordinate_columns = ["x0_px", "y0_px", "x1_px", "y1_px"]
    metric_columns = [
        "has_cells",
        "cell_area_um2",
        "tile_area_um2",
        "cell_area_fraction",
        "meets_min_cells",
        "meets_min_cell_area_fraction",
        "passes_tile_filter",
        "tile_filter_quadrant",
    ]
    labels = labels[
        ["sample", "tile_row", "tile_col"]
        + count_columns
        + ["n_cells"]
        + proportion_columns
        + metric_columns
        + coordinate_columns
    ]
    labels.to_csv(paths.base / f"{paths.name}_tile_labels.csv", index=False)
    passing = int(tile_metrics["passes_tile_filter"].sum())
    print(
        f"[{paths.name}] tile metrics: {len(tile_metrics):,} complete tiles, "
        f"{len(labels):,} with cells, {passing:,} pass current filter"
    )
    return tile_cells, labels, tile_metrics


def save_tile_filter_qc_plot(
    paths: SamplePaths,
    tile_metrics: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    if tile_metrics.empty:
        return
    n_cells = pd.to_numeric(tile_metrics["n_cells"], errors="coerce").fillna(0)
    area_fraction = pd.to_numeric(tile_metrics["cell_area_fraction"], errors="coerce")
    finite_area = area_fraction[np.isfinite(area_fraction)]
    total = len(tile_metrics)
    passing = int(tile_metrics["passes_tile_filter"].sum())

    quantiles = [0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    summary = pd.DataFrame(
        {
            "quantile": quantiles,
            "n_cells": n_cells.quantile(quantiles).to_numpy(),
            "cell_area_fraction": (
                finite_area.quantile(quantiles).to_numpy()
                if len(finite_area)
                else np.full(len(quantiles), np.nan)
            ),
        }
    )
    summary.to_csv(paths.base / f"{paths.name}_tile_filter_quantiles.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    max_cells = int(n_cells.max()) if len(n_cells) else 0
    if max_cells <= 80:
        cell_bins = np.arange(-0.5, max_cells + 1.5, 1)
    else:
        cell_bins = 80
    axes[0, 0].hist(n_cells, bins=cell_bins, color="#4C78A8", edgecolor="white")
    axes[0, 0].axvline(config.min_cells_per_tile, color="#D62728", linewidth=2)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Cells per complete tile")
    axes[0, 0].set_xlabel("n_cells")
    axes[0, 0].set_ylabel("Tiles (log scale)")

    if len(finite_area):
        axes[0, 1].hist(finite_area, bins=80, color="#59A14F", edgecolor="white")
        axes[0, 1].axvline(config.min_cell_area_fraction, color="#D62728", linewidth=2)
        axes[0, 1].set_yscale("log")
    else:
        axes[0, 1].text(0.5, 0.5, "No cell_area values", ha="center", va="center")
    axes[0, 1].set_title("Cell area fraction per complete tile")
    axes[0, 1].set_xlabel("cell_area_fraction")
    axes[0, 1].set_ylabel("Tiles (log scale)")

    scatter = (
        tile_metrics.sample(n=50_000, random_state=1)
        if len(tile_metrics) > 50_000
        else tile_metrics
    )
    scatter_cells = pd.to_numeric(scatter["n_cells"], errors="coerce").fillna(0)
    scatter_area = pd.to_numeric(scatter["cell_area_fraction"], errors="coerce")
    colors = scatter["tile_filter_quadrant"].map(TILE_FILTER_QUADRANT_COLORS).fillna("#B8B8B8")
    axes[1, 0].scatter(scatter_cells, scatter_area, s=4, c=colors, alpha=0.45, linewidths=0)
    axes[1, 0].axvline(config.min_cells_per_tile, color="#D62728", linewidth=1.5)
    axes[1, 0].axhline(config.min_cell_area_fraction, color="#D62728", linewidth=1.5)
    axes[1, 0].set_title("Joint tile-filter metrics")
    axes[1, 0].set_xlabel("n_cells")
    axes[1, 0].set_ylabel("cell_area_fraction")
    axes[1, 0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=TILE_FILTER_QUADRANT_COLORS[key],
                markersize=5,
                label=TILE_FILTER_QUADRANT_LABELS[key],
            )
            for key in TILE_FILTER_QUADRANTS
        ],
        loc="best",
        fontsize=7,
        frameon=False,
    )

    axes[1, 1].axis("off")
    lines = [
        f"Complete tiles: {total:,}",
        f"Tiles with cells: {int((n_cells > 0).sum()):,}",
        f"Pass current filter: {passing:,} ({passing / total:.1%})",
        "",
        f"min_cells_per_tile = {config.min_cells_per_tile}",
        f"min_cell_area_fraction = {config.min_cell_area_fraction:g}",
        "",
        "Median / 90th / 95th percentile",
        (
            "n_cells: "
            f"{n_cells.quantile(0.50):.1f} / "
            f"{n_cells.quantile(0.90):.1f} / "
            f"{n_cells.quantile(0.95):.1f}"
        ),
    ]
    if len(finite_area):
        lines.append(
            "cell_area_fraction: "
            f"{finite_area.quantile(0.50):.4f} / "
            f"{finite_area.quantile(0.90):.4f} / "
            f"{finite_area.quantile(0.95):.4f}"
        )
    axes[1, 1].text(0, 1, "\n".join(lines), va="top", fontsize=11)

    fig.suptitle(f"{paths.name}: tile filter QC", fontsize=14)
    fig.savefig(paths.base / f"{paths.name}_tile_filter_qc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def crop_tiles(
    paths: SamplePaths,
    tile_labels: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if paths.he_image is None:
        raise FileNotFoundError("H&E image was not resolved")
    tile_directory = paths.base / "he_tiles"
    tile_directory.mkdir(parents=True, exist_ok=True)
    all_tiles = tile_labels.loc[tile_labels["n_cells"] > 0].copy()
    if all_tiles.empty:
        raise ValueError("No complete H&E tiles contain aligned Xenium cells")
    if "passes_tile_filter" in tile_labels.columns:
        selected_mask = all_tiles["passes_tile_filter"]
    else:
        selected_mask = all_tiles["n_cells"] >= config.min_cells_per_tile
        if config.min_cell_area_fraction > 0:
            if "cell_area_fraction" not in all_tiles.columns:
                raise ValueError(
                    "Tile labels have no cell_area_fraction column, so "
                    "min_cell_area_fraction cannot be applied"
                )
            selected_mask &= all_tiles["cell_area_fraction"] >= config.min_cell_area_fraction
    relative_paths: list[str] = []
    expected_outputs: set[Path] = set()
    n_pruned_tiles = 0
    with tifffile.TiffFile(paths.he_image) as tif:
        store = tif.series[0].levels[0].aszarr()
        try:
            image_array = resolve_zarr_array(zarr.open(store, mode="r"))
            print(
                f"[{paths.name}] extracting {len(all_tiles):,} nonempty tiles "
                f"from {tuple(image_array.shape)}"
            )
            for row in tqdm(
                all_tiles.itertuples(index=False),
                total=len(all_tiles),
                desc=f"{paths.name}: tiles",
                unit="tile",
                dynamic_ncols=True,
            ):
                filename = f"{paths.name}_row-{int(row.tile_row):05d}_col-{int(row.tile_col):05d}.png"
                output = tile_directory / filename
                expected_outputs.add(output)
                if config.overwrite or not output.exists():
                    tile = extract_rgb_tile(
                        image_array,
                        int(row.x0_px),
                        int(row.y0_px),
                        int(row.x1_px),
                        int(row.y1_px),
                    )
                    expected = (config.tile_size_px, config.tile_size_px, 3)
                    if tile.shape != expected:
                        raise ValueError(f"Unexpected tile shape {tile.shape}; expected {expected}")
                    if tile.dtype != np.uint8:
                        raise TypeError(f"Expected uint8 H&E data, received {tile.dtype}")
                    Image.fromarray(tile).save(output, format="PNG", compress_level=3)
                relative_paths.append(output.relative_to(paths.base).as_posix())
        finally:
            close = getattr(store, "close", None)
            if close:
                close()
    if config.prune_stale_tiles:
        stale_tiles = sorted(
            path
            for path in tile_directory.glob(f"{paths.name}_row-*_col-*.png")
            if path not in expected_outputs
        )
        for path in stale_tiles:
            path.unlink()
        n_pruned_tiles = len(stale_tiles)
        if stale_tiles:
            print(f"[{paths.name}] pruned {n_pruned_tiles:,} stale tile PNG(s)")
    all_tiles["tile_path"] = relative_paths
    selected = all_tiles.loc[selected_mask].copy()
    all_tiles.attrs["n_pruned_tiles"] = n_pruned_tiles
    selected.attrs["n_pruned_tiles"] = n_pruned_tiles
    selected.attrs["n_written_tile_pngs"] = len(all_tiles)
    all_tiles.to_csv(paths.base / f"{paths.name}_all_tile_labels_with_paths.csv", index=False)
    selected.to_csv(paths.base / f"{paths.name}_tile_labels_with_paths.csv", index=False)
    if selected.empty:
        raise ValueError(
            "No tiles satisfy "
            f"min_cells_per_tile={config.min_cells_per_tile} and "
            f"min_cell_area_fraction={config.min_cell_area_fraction:g}; "
            f"{len(all_tiles):,} nonempty tile PNG(s) were still written"
        )
    return all_tiles, selected


def representative_quadrant_examples(
    manifest: pd.DataFrame,
    quadrant: str,
    *,
    count: int = 3,
) -> pd.DataFrame:
    """Return deterministic, central examples from a tile-filter quadrant."""
    if "tile_filter_quadrant" not in manifest.columns:
        return pd.DataFrame()
    candidates = manifest.loc[manifest["tile_filter_quadrant"] == quadrant].copy()
    if candidates.empty:
        return candidates
    n_cells = pd.to_numeric(candidates["n_cells"], errors="coerce").fillna(0)
    area_fraction = pd.to_numeric(
        candidates["cell_area_fraction"],
        errors="coerce",
    ).fillna(0.0)
    candidates["_example_distance"] = (
        (n_cells.rank(pct=True) - 0.5).abs()
        + (area_fraction.rank(pct=True) - 0.5).abs()
    )
    return (
        candidates.sort_values(
            ["_example_distance", "n_cells", "cell_area_fraction", "tile_row", "tile_col"],
            ascending=[True, False, False, True, True],
        )
        .head(count)
        .drop(columns="_example_distance")
    )


def save_tile_filter_quadrant_examples(
    paths: SamplePaths,
    all_manifest: pd.DataFrame,
    config: PipelineConfig,
    *,
    tile_cells: pd.DataFrame | None = None,
    cell_boundaries: BoundaryData | None = None,
    nucleus_boundaries: BoundaryData | None = None,
    metadata: ImageMetadata | None = None,
    pixel_size: float | None = None,
    xenium_to_he: np.ndarray | None = None,
    examples_per_quadrant: int = 3,
    max_scatter_points: int = 50_000,
    output_path: Path | None = None,
    area_fraction_label: str = "cell_area_fraction",
    min_nucleus_area_fraction: float = 0.0,
    min_he_tissue_fraction: float = 0.0,
    require_nucleus_area_fraction: bool = False,
    require_he_tissue_fraction: bool = False,
) -> Path | None:
    """Save a scatterplot plus example H&E crops from each threshold quadrant."""
    if all_manifest.empty or "tile_path" not in all_manifest.columns:
        return None
    all_manifest = apply_tile_filter_cutoffs(
        all_manifest,
        config.min_cells_per_tile,
        config.min_cell_area_fraction,
        min_nucleus_area_fraction,
        min_he_tissue_fraction,
        require_nucleus_area_fraction=require_nucleus_area_fraction,
        require_he_tissue_fraction=require_he_tissue_fraction,
    )
    can_overlay_markers = (
        tile_cells is not None
        and {"tile_row", "tile_col", "he_x_px", "he_y_px"}.issubset(tile_cells.columns)
    )
    can_overlay_boundaries = (
        can_overlay_markers
        and metadata is not None
        and pixel_size is not None
        and "cell_index" in tile_cells.columns
    )

    scatter = (
        all_manifest.sample(n=max_scatter_points, random_state=1)
        if len(all_manifest) > max_scatter_points
        else all_manifest
    )
    scatter_cells = pd.to_numeric(scatter["n_cells"], errors="coerce").fillna(0)
    scatter_area = pd.to_numeric(scatter["cell_area_fraction"], errors="coerce").fillna(0.0)
    scatter_colors = (
        scatter["tile_filter_quadrant"]
        .map(TILE_FILTER_QUADRANT_COLORS)
        .fillna("#B8B8B8")
    )

    fig = plt.figure(figsize=(15, 18), constrained_layout=True)
    grid = fig.add_gridspec(
        nrows=1 + len(TILE_FILTER_QUADRANTS),
        ncols=examples_per_quadrant,
        height_ratios=[1.8, *([1] * len(TILE_FILTER_QUADRANTS))],
    )
    scatter_axis = fig.add_subplot(grid[0, :])
    scatter_axis.scatter(
        scatter_cells,
        scatter_area,
        s=5,
        c=scatter_colors,
        alpha=0.5,
        linewidths=0,
    )
    scatter_axis.axvline(config.min_cells_per_tile, color="#D62728", linewidth=1.6)
    scatter_axis.axhline(config.min_cell_area_fraction, color="#D62728", linewidth=1.6)
    scatter_axis.set_xlabel("n_cells")
    scatter_axis.set_ylabel(area_fraction_label)
    scatter_axis.set_title("Nonempty tiles by count/area threshold quadrant")
    scatter_axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=TILE_FILTER_QUADRANT_COLORS[key],
                markersize=6,
                label=(
                    f"{TILE_FILTER_QUADRANT_LABELS[key]} "
                    f"({int((all_manifest['tile_filter_quadrant'] == key).sum()):,})"
                ),
            )
            for key in TILE_FILTER_QUADRANTS
        ],
        loc="best",
        fontsize=8,
        frameon=False,
    )

    for row_index, quadrant in enumerate(TILE_FILTER_QUADRANTS, start=1):
        examples = representative_quadrant_examples(
            all_manifest,
            quadrant,
            count=examples_per_quadrant,
        )
        row_axes = [fig.add_subplot(grid[row_index, column]) for column in range(examples_per_quadrant)]
        for axis_index, axis in enumerate(row_axes):
            if axis_index >= len(examples):
                axis.text(
                    0.5,
                    0.5,
                    "no tile",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#666666",
                )
                axis.set_facecolor("#F3F3F3")
                axis.set_xticks([])
                axis.set_yticks([])
                if axis_index == 0:
                    axis.set_ylabel(TILE_FILTER_QUADRANT_LABELS[quadrant], fontsize=10)
                continue
            tile = examples.iloc[axis_index]
            image = load_tile_image_for_qc(paths, tile, config)
            axis.imshow(image)
            if can_overlay_markers:
                cells = tile_cells.loc[
                    (tile_cells["tile_row"] == tile.tile_row)
                    & (tile_cells["tile_col"] == tile.tile_col)
                ]
                if can_overlay_boundaries:
                    draw_boundaries_on_axis(
                        axis,
                        cells,
                        int(tile.x0_px),
                        int(tile.y0_px),
                        cell_boundaries,
                        paths,
                        metadata,
                        float(pixel_size),
                        xenium_to_he,
                        nucleus_boundaries=nucleus_boundaries,
                    )
                else:
                    marker_x, marker_y, _ = marker_he_coordinates(cells)
                    axis.scatter(
                        marker_x - int(tile.x0_px),
                        marker_y - int(tile.y0_px),
                        s=10,
                        marker="+",
                        color="yellow",
                    )
                axis.set_xlim(0, config.tile_size_px)
                axis.set_ylim(config.tile_size_px, 0)
            area_value = pd.to_numeric(
                pd.Series([tile.cell_area_fraction]),
                errors="coerce",
            ).iloc[0]
            area_text = f"{float(area_value):.3f}" if pd.notna(area_value) else "NA"
            tissue_value = pd.to_numeric(
                pd.Series([getattr(tile, "he_tissue_fraction", np.nan)]),
                errors="coerce",
            ).iloc[0]
            tissue_text = f"{float(tissue_value):.3f}" if pd.notna(tissue_value) else "NA"
            title = (
                f"row={int(tile.tile_row)}, col={int(tile.tile_col)}\n"
                f"cells={int(tile.n_cells)}, area={area_text}, tissue={tissue_text}"
            )
            axis.set_title(title, fontsize=9)
            axis.axis("off")
            if axis_index == 0:
                axis.set_ylabel(TILE_FILTER_QUADRANT_LABELS[quadrant], fontsize=10)
    fig.suptitle(
        (
            f"{paths.name}: 3 example nonempty tiles per tile-filter quadrant "
            f"(min cells={config.min_cells_per_tile}, "
            f"min {area_fraction_label}={config.min_cell_area_fraction:g}"
            + (
                f", min nucleus_area_fraction={float(min_nucleus_area_fraction):g}"
                if float(min_nucleus_area_fraction) > 0
                else ""
            )
            + (
                f", min he_tissue_fraction={float(min_he_tissue_fraction):g}"
                if float(min_he_tissue_fraction) > 0
                else ""
            )
            + ")"
        ),
        fontsize=14,
    )
    output_path = output_path or paths.base / f"{paths.name}_tile_filter_quadrant_examples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    return output_path


def existing_cells_table(paths: SamplePaths) -> Path | None:
    candidates = [
        paths.base / f"{paths.name}_cells_with_annotations.csv",
        paths.base / f"{paths.name}_cells_with_cluster_labels.csv",
        paths.base / f"{paths.name}_cells_aligned_to_he.csv",
    ]
    return next((path for path in candidates if path.is_file()), None)


def tile_cells_from_existing_output(
    paths: SamplePaths,
    config: PipelineConfig,
    metadata: ImageMetadata,
) -> pd.DataFrame | None:
    cells_path = existing_cells_table(paths)
    if cells_path is None:
        return None
    cells = pd.read_csv(cells_path)
    required = {"he_x_px", "he_y_px"}
    if not required.issubset(cells.columns):
        return None
    if "inside_he_image" in cells.columns:
        cells = cells.loc[cells["inside_he_image"]].copy()
    elif "inside_he_candidate" in cells.columns:
        cells = cells.loc[cells["inside_he_candidate"]].copy()
    n_columns = metadata.full_width // config.tile_size_px
    n_rows = metadata.full_height // config.tile_size_px
    cells["tile_col"] = (cells["he_x_px"] // config.tile_size_px).astype(int)
    cells["tile_row"] = (cells["he_y_px"] // config.tile_size_px).astype(int)
    return cells.loc[
        (cells["tile_col"] >= 0)
        & (cells["tile_col"] < n_columns)
        & (cells["tile_row"] >= 0)
        & (cells["tile_row"] < n_rows)
    ].copy()


def load_tile_filter_overlay_context(
    paths: SamplePaths,
    config: PipelineConfig,
) -> dict[str, object]:
    """Load saved cell and boundary data used to draw cells on tile examples."""
    metadata = load_image_metadata(paths)
    tile_cells = tile_cells_from_existing_output(paths, config, metadata)
    cell_boundaries = None
    nucleus_boundaries = None
    pixel_size: float | None = None
    xenium_to_he: np.ndarray | None = None
    if tile_cells is not None and not tile_cells.empty and "cell_index" in tile_cells.columns:
        try:
            pixel_size = experiment_pixel_size(paths)
            if {"cell_centroid_x", "cell_centroid_y"}.issubset(tile_cells.columns):
                _, _, xenium_to_he = align_xy_to_he(
                    tile_cells[["cell_centroid_x", "cell_centroid_y"]]
                    .head(1)
                    .to_numpy(),
                    paths,
                    metadata,
                    pixel_size,
                )
            cell_boundaries = load_boundaries(paths, tile_cells, boundary_kind="cell")
        except Exception as error:
            print(f"[{paths.name}] cell-boundary overlay unavailable: {error}")
            cell_boundaries = None
        try:
            nucleus_boundaries = load_boundaries(paths, tile_cells, boundary_kind="nucleus")
        except Exception as error:
            print(f"[{paths.name}] nucleus-boundary overlay unavailable: {error}")
            nucleus_boundaries = None
    return {
        "tile_cells": tile_cells,
        "cell_boundaries": cell_boundaries,
        "nucleus_boundaries": nucleus_boundaries,
        "metadata": metadata,
        "pixel_size": pixel_size,
        "xenium_to_he": xenium_to_he,
    }


def save_existing_tile_filter_quadrant_examples(
    paths: SamplePaths,
    config: PipelineConfig,
    *,
    output_path: Path | None = None,
) -> Path:
    """Rebuild quadrant examples from saved manifests and cell tables."""
    all_manifest_path = paths.base / f"{paths.name}_all_tile_labels_with_paths.csv"
    if not all_manifest_path.is_file():
        raise FileNotFoundError(all_manifest_path)
    all_manifest = pd.read_csv(all_manifest_path)
    overlay = load_tile_filter_overlay_context(paths, config)
    plot_path = save_tile_filter_quadrant_examples(
        paths,
        all_manifest,
        config,
        tile_cells=overlay["tile_cells"],
        cell_boundaries=overlay["cell_boundaries"],
        nucleus_boundaries=overlay["nucleus_boundaries"],
        metadata=overlay["metadata"],
        pixel_size=overlay["pixel_size"],
        xenium_to_he=overlay["xenium_to_he"],
        output_path=output_path,
    )
    return plot_path or (output_path or paths.base / f"{paths.name}_tile_filter_quadrant_examples.png")


def tile_source_path(paths: SamplePaths, tile_path: object) -> Path:
    tile_path = Path(str(tile_path))
    return tile_path if tile_path.is_absolute() else paths.base / tile_path


def add_tile_ids(
    paths: SamplePaths,
    tiles: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Add stable tile IDs and cutoff metadata before writing a selection table."""
    table = tiles.copy()
    tile_ids = [
        f"{paths.name}_row-{int(row.tile_row):05d}_col-{int(row.tile_col):05d}"
        for row in table.itertuples(index=False)
    ]
    table.insert(0, "tile_id", tile_ids)
    table.insert(1, "min_cells_per_tile", int(config.min_cells_per_tile))
    table.insert(2, "min_cell_area_fraction", float(config.min_cell_area_fraction))
    if "tile_path" in table.columns:
        source_paths = [str(tile_source_path(paths, value)) for value in table["tile_path"]]
        table.insert(table.columns.get_loc("tile_path") + 1, "source_tile_path", source_paths)
    return table


def materialize_tile_filter_tiles(
    paths: SamplePaths,
    selected_tiles: pd.DataFrame,
    config: PipelineConfig,
    destination: Path,
    *,
    mode: str = "copy",
) -> pd.DataFrame:
    """Copy, symlink, or move selected tile PNGs into a cutoff-specific folder."""
    if mode not in {"copy", "symlink", "move"}:
        raise ValueError("mode must be one of: copy, symlink, move")
    destination.mkdir(parents=True, exist_ok=True)
    columns = [
        "tile_id",
        "mode",
        "status",
        "source_tile_path",
        "materialized_tile_path",
    ]
    records: list[dict[str, object]] = []
    selected_with_ids = add_tile_ids(paths, selected_tiles, config)
    for tile in selected_with_ids.itertuples(index=False):
        source = tile_source_path(paths, tile.tile_path)
        target = destination / source.name
        status = mode
        if not source.is_file():
            status = "missing_source"
        else:
            if target.exists() or target.is_symlink():
                target.unlink()
            if mode == "copy":
                shutil.copy2(source, target)
            elif mode == "symlink":
                target.symlink_to(source.resolve())
            else:
                shutil.move(str(source), str(target))
        records.append(
            {
                "tile_id": tile.tile_id,
                "mode": mode,
                "status": status,
                "source_tile_path": str(source),
                "materialized_tile_path": str(target),
            }
        )
    return pd.DataFrame(records, columns=columns)


def save_tile_filter_cutoff_outputs(
    paths: SamplePaths,
    config: PipelineConfig,
    *,
    output_dir: str | Path | None = None,
    materialize_tiles: bool = False,
    materialize_mode: str = "copy",
    materialized_folder: str | Path | None = None,
    examples_per_quadrant: int = 3,
    max_scatter_points: int = 50_000,
) -> dict[str, object]:
    """Save cutoff-specific quadrant examples and passing tile IDs."""
    all_manifest_path = paths.base / f"{paths.name}_all_tile_labels_with_paths.csv"
    if not all_manifest_path.is_file():
        raise FileNotFoundError(all_manifest_path)
    output_dir = Path(output_dir) if output_dir is not None else paths.base / "tile_filter_cutoff_tests"
    output_dir.mkdir(parents=True, exist_ok=True)

    slug = tile_filter_cutoff_slug(
        config.min_cells_per_tile,
        config.min_cell_area_fraction,
    )
    all_manifest = apply_tile_filter_cutoffs(
        pd.read_csv(all_manifest_path),
        config.min_cells_per_tile,
        config.min_cell_area_fraction,
    )
    sort_columns = [column for column in ["tile_row", "tile_col"] if column in all_manifest.columns]
    if sort_columns:
        all_manifest = all_manifest.sort_values(sort_columns).reset_index(drop=True)
    passing_tiles = all_manifest.loc[all_manifest["passes_tile_filter"]].copy()

    passing_tiles_path = output_dir / f"{paths.name}_{slug}_passing_tile_ids.csv"
    add_tile_ids(paths, passing_tiles, config).to_csv(passing_tiles_path, index=False)

    metrics_path = paths.base / f"{paths.name}_tile_filter_metrics.csv"
    if metrics_path.is_file():
        summary_source = apply_tile_filter_cutoffs(
            pd.read_csv(metrics_path),
            config.min_cells_per_tile,
            config.min_cell_area_fraction,
        )
    else:
        summary_source = all_manifest
    summary, _ = tile_filter_selection_summary(
        summary_source,
        config.min_cells_per_tile,
        config.min_cell_area_fraction,
    )
    summary.insert(0, "sample", paths.name)
    summary.insert(1, "cutoff_slug", slug)
    summary["nonempty_tiles_written"] = len(all_manifest)
    summary["passing_nonempty_tiles_written"] = len(passing_tiles)
    summary["passing_tile_ids_path"] = str(passing_tiles_path)

    overlay = load_tile_filter_overlay_context(paths, config)
    quadrant_examples_path = output_dir / f"{paths.name}_{slug}_tile_filter_quadrant_examples.png"
    plot_path = save_tile_filter_quadrant_examples(
        paths,
        all_manifest,
        config,
        tile_cells=overlay["tile_cells"],
        cell_boundaries=overlay["cell_boundaries"],
        nucleus_boundaries=overlay["nucleus_boundaries"],
        metadata=overlay["metadata"],
        pixel_size=overlay["pixel_size"],
        xenium_to_he=overlay["xenium_to_he"],
        examples_per_quadrant=examples_per_quadrant,
        max_scatter_points=max_scatter_points,
        output_path=quadrant_examples_path,
    )
    summary["quadrant_examples_path"] = str(plot_path) if plot_path is not None else ""

    materialized_tile_dir = ""
    materialized_tile_manifest = ""
    n_materialized = 0
    n_missing_tile_files = 0
    if materialize_tiles:
        if materialized_folder is None:
            materialized_path = output_dir / f"{paths.name}_{slug}_tiles"
        else:
            materialized_path = Path(materialized_folder)
            if not materialized_path.is_absolute():
                materialized_path = output_dir / materialized_path
        materialized = materialize_tile_filter_tiles(
            paths,
            passing_tiles,
            config,
            materialized_path,
            mode=materialize_mode,
        )
        materialized_tile_manifest_path = output_dir / f"{paths.name}_{slug}_materialized_tiles.csv"
        materialized.to_csv(materialized_tile_manifest_path, index=False)
        materialized_tile_dir = str(materialized_path)
        materialized_tile_manifest = str(materialized_tile_manifest_path)
        n_materialized = int((materialized["status"] == materialize_mode).sum())
        n_missing_tile_files = int((materialized["status"] == "missing_source").sum())

    summary["materialized_tile_dir"] = materialized_tile_dir
    summary["materialized_tile_manifest"] = materialized_tile_manifest
    summary["n_materialized_tiles"] = n_materialized
    summary["n_missing_tile_files"] = n_missing_tile_files
    summary_path = output_dir / f"{paths.name}_{slug}_tile_filter_summary.csv"
    summary.to_csv(summary_path, index=False)

    result = summary.iloc[0].to_dict()
    result.update(
        {
            "summary_path": str(summary_path),
            "passing_tile_ids_path": str(passing_tiles_path),
            "quadrant_examples_path": str(plot_path) if plot_path is not None else "",
        }
    )
    return result


def draw_boundaries_on_axis(
    axis: object,
    cells: pd.DataFrame,
    x0: int,
    y0: int,
    cell_boundaries: BoundaryData | None,
    paths: SamplePaths,
    metadata: ImageMetadata,
    pixel_size: float,
    xenium_to_he: np.ndarray | None,
    *,
    nucleus_boundaries: BoundaryData | None = None,
) -> None:
    def draw_layer(
        boundaries: BoundaryData | None,
        *,
        color: str,
        linewidth: float,
        alpha: float,
    ) -> None:
        for cell in cells.itertuples(index=False):
            boundary = boundary_in_he_pixels(
                int(cell.cell_index), boundaries, paths, metadata, pixel_size, xenium_to_he
            )
            if boundary is not None:
                local = boundary - np.array([x0, y0])
                axis.plot(
                    local[:, 0],
                    local[:, 1],
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                )

    draw_layer(cell_boundaries, color="blue", linewidth=0.6, alpha=0.8)
    draw_layer(nucleus_boundaries, color="magenta", linewidth=0.55, alpha=0.85)
    marker_x, marker_y, _ = marker_he_coordinates(cells)
    axis.scatter(marker_x - x0, marker_y - y0, s=10, marker="+", color="yellow")


def save_tile_preview(
    paths: SamplePaths,
    tile_cells: pd.DataFrame,
    manifest: pd.DataFrame,
    cell_boundaries: BoundaryData | None,
    nucleus_boundaries: BoundaryData | None,
    metadata: ImageMetadata,
    pixel_size: float,
    xenium_to_he: np.ndarray | None,
    config: PipelineConfig,
) -> None:
    preview = manifest.sample(n=min(config.preview_count, len(manifest)), random_state=1)
    columns = 4
    rows = int(np.ceil(len(preview) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12, rows * 3), squeeze=False)
    flat = axes.ravel()
    for axis, row in zip(flat, preview.itertuples(index=False)):
        image = load_tile_image_for_qc(paths, row, config)
        axis.imshow(image)
        cells = tile_cells.loc[
            (tile_cells["tile_row"] == row.tile_row) & (tile_cells["tile_col"] == row.tile_col)
        ]
        draw_boundaries_on_axis(
            axis, cells, int(row.x0_px), int(row.y0_px), cell_boundaries,
            paths, metadata, pixel_size, xenium_to_he,
            nucleus_boundaries=nucleus_boundaries,
        )
        axis.set_xlim(0, config.tile_size_px)
        axis.set_ylim(config.tile_size_px, 0)
        axis.set_title(f"row={row.tile_row}, col={row.tile_col}\ncells={len(cells)}")
        axis.axis("off")
    for axis in flat[len(preview) :]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(paths.base / f"{paths.name}_tile_preview_with_cells.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_orientation_plot(
    paths: SamplePaths,
    tile_cells: pd.DataFrame,
    manifest: pd.DataFrame,
    cell_boundaries: BoundaryData | None,
    nucleus_boundaries: BoundaryData | None,
    metadata: ImageMetadata,
    pixel_size: float,
    xenium_to_he: np.ndarray | None,
    config: PipelineConfig,
) -> None:
    selected = manifest.sort_values("n_cells", ascending=False).iloc[0]
    x0, y0 = int(selected.x0_px), int(selected.y0_px)
    tile = np.asarray(load_tile_image_for_qc(paths, selected, config))
    cells = tile_cells.loc[
        (tile_cells["tile_row"] == selected.tile_row) & (tile_cells["tile_col"] == selected.tile_col)
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 7), constrained_layout=True)
    axes[0].imshow(metadata.thumbnail)
    axes[0].add_patch(
        Rectangle(
            (x0 * metadata.thumbnail_scale_x, y0 * metadata.thumbnail_scale_y),
            config.tile_size_px * metadata.thumbnail_scale_x,
            config.tile_size_px * metadata.thumbnail_scale_y,
            edgecolor="yellow",
            facecolor="none",
            linewidth=2,
        )
    )
    axes[0].set_title(f"Tile location\nrow={selected.tile_row}, col={selected.tile_col}")
    axes[0].axis("off")
    axes[1].imshow(tile)
    axes[1].text(4, 12, "TOP-LEFT", color="yellow", fontsize=8)
    axes[1].text(config.tile_size_px - 4, 12, "TOP-RIGHT", color="yellow", fontsize=8, ha="right")
    axes[1].text(4, config.tile_size_px - 8, "BOTTOM-LEFT", color="yellow", fontsize=8)
    axes[1].text(config.tile_size_px - 4, config.tile_size_px - 8, "BOTTOM-RIGHT", color="yellow", fontsize=8, ha="right")
    axes[1].set_title("Saved full-resolution tile")
    axes[1].axis("off")
    axes[2].imshow(tile)
    draw_boundaries_on_axis(
        axes[2], cells, x0, y0, cell_boundaries,
        paths, metadata, pixel_size, xenium_to_he,
        nucleus_boundaries=nucleus_boundaries,
    )
    axes[2].set_xlim(0, config.tile_size_px)
    axes[2].set_ylim(config.tile_size_px, 0)
    axes[2].set_title(f"Xenium cells and nuclei over H&E\n{len(cells)} cells")
    axes[2].axis("off")
    _, _, marker_label = marker_he_coordinates(cells)
    legend_handles: list[Line2D] = []
    if cell_boundaries is not None:
        legend_handles.append(Line2D([0], [0], color="blue", label="Cell boundary"))
    if nucleus_boundaries is not None:
        legend_handles.append(Line2D([0], [0], color="magenta", label="Nucleus boundary"))
    legend_handles.append(
        Line2D([0], [0], color="yellow", marker="+", linestyle="none", label=marker_label)
    )
    axes[2].legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=7,
    )
    fig.savefig(paths.base / f"{paths.name}_tile_orientation_check.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_reconstruction_plot(
    paths: SamplePaths,
    tile_cells: pd.DataFrame,
    manifest: pd.DataFrame,
    boundaries: BoundaryData | None,
    metadata: ImageMetadata,
    pixel_size: float,
    xenium_to_he: np.ndarray | None,
    config: PipelineConfig,
) -> None:
    if paths.he_image is None:
        raise FileNotFoundError("H&E image was not resolved")
    with tifffile.TiffFile(paths.he_image) as tif:
        level = min(config.pyramid_level, len(tif.series[0].levels) - 1)
        reference = ensure_rgb(tif.series[0].levels[level].asarray())
    height, width = reference.shape[:2]
    scale_x = width / metadata.full_width
    scale_y = height / metadata.full_height
    reconstructed = np.full(reference.shape, 255, dtype=np.uint8)
    coverage = np.zeros((height, width), dtype=bool)
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc=f"{paths.name}: reconstruction"):
        with Image.open(paths.base / row.tile_path) as image:
            tile = image.convert("RGB")
        x0, y0 = round(row.x0_px * scale_x), round(row.y0_px * scale_y)
        x1, y1 = round(row.x1_px * scale_x), round(row.y1_px * scale_y)
        if x1 <= x0 or y1 <= y0:
            continue
        reconstructed[y0:y1, x0:x1] = np.asarray(tile.resize((x1 - x0, y1 - y0), Image.Resampling.BOX))
        coverage[y0:y1, x0:x1] = True
    Image.fromarray(reconstructed).save(paths.base / f"{paths.name}_reconstructed_from_tiles.png")

    keys = manifest[["tile_row", "tile_col"]].drop_duplicates()
    annotation_cells = tile_cells.merge(keys, on=["tile_row", "tile_col"], how="inner").drop_duplicates("cell_id")
    colors = create_color_map(annotation_cells["cell_type"])
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    for cell in tqdm(annotation_cells.itertuples(index=False), total=len(annotation_cells), desc=f"{paths.name}: cell overlay"):
        color = colors.get(str(cell.cell_type), (140, 140, 140))
        boundary = boundary_in_he_pixels(
            int(cell.cell_index), boundaries, paths, metadata, pixel_size, xenium_to_he
        )
        if boundary is not None:
            points = [
                (int(round(x * scale_x)), int(round(y * scale_y)))
                for x, y in boundary
            ]
            draw.polygon(points, fill=color + (config.annotation_alpha,), outline=color + (255,))
        marker_x, marker_y = marker_he_coordinates_from_record(cell)
        cx, cy = int(round(marker_x * scale_x)), int(round(marker_y * scale_y))
        draw.ellipse((cx - 1, cy - 1, cx + 1, cy + 1), fill=color + (255,))
    layer_array = np.asarray(layer).copy()
    layer_array[~coverage, 3] = 0
    annotated = Image.alpha_composite(Image.fromarray(reconstructed).convert("RGBA"), Image.fromarray(layer_array))
    annotated.save(paths.base / f"{paths.name}_reconstructed_tiles_with_cells.png")

    fig, axes = plt.subplots(1, 3, figsize=(17, 10), constrained_layout=True)
    axes[0].imshow(reference)
    axes[0].set_title(f"Original H&E pyramid level {level}")
    axes[1].imshow(reconstructed)
    axes[1].set_title("Reconstructed from saved tiles")
    axes[2].imshow(annotated)
    axes[2].set_title("Tiles with cell-label overlay")
    for axis in axes:
        axis.axis("off")
    present = sorted(annotation_cells["cell_type"].astype(str).unique())
    axes[2].legend(
        handles=[Patch(facecolor=np.array(colors[label]) / 255, label=label) for label in present],
        title="Cell label",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=7,
    )
    fig.savefig(paths.base / f"{paths.name}_annotated_tile_reconstruction.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def completion_marker(paths: SamplePaths, suffix: str = "_pipeline_complete.json") -> Path:
    return paths.base / f"{paths.name}{suffix}"


def completion_record(paths: SamplePaths) -> Path:
    return completion_marker(paths)


def existing_manifest(paths: SamplePaths) -> pd.DataFrame:
    path = paths.base / f"{paths.name}_tile_labels_with_paths.csv"
    if not path.is_file():
        return pd.DataFrame()
    manifest = pd.read_csv(path)
    manifest["sample"] = paths.name
    if "tile_path" in manifest:
        prefix = f"{paths.name}/"
        manifest["tile_path"] = manifest["tile_path"].astype(str).map(
            lambda value: value if value.startswith(prefix) else prefix + value
        )
    return manifest


def normalize_local_tile_paths(manifest: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    manifest = manifest.copy()
    if "tile_path" in manifest.columns:
        prefix = f"{sample_name}/"
        manifest["tile_path"] = manifest["tile_path"].astype(str).map(
            lambda value: value.removeprefix(prefix)
        )
    return manifest


def existing_tile_manifests_for_qc(paths: SamplePaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_path = paths.base / f"{paths.name}_tile_labels_with_paths.csv"
    if not selected_path.is_file():
        raise FileNotFoundError(
            f"Cannot recreate tile QC plots without {selected_path}"
        )
    all_path = paths.base / f"{paths.name}_all_tile_labels_with_paths.csv"
    selected = normalize_local_tile_paths(pd.read_csv(selected_path), paths.name)
    all_tiles = (
        normalize_local_tile_paths(pd.read_csv(all_path), paths.name)
        if all_path.is_file()
        else selected.copy()
    )
    for label, manifest in {"selected": selected, "all": all_tiles}.items():
        if "tile_path" not in manifest.columns:
            raise ValueError(f"Existing {label} tile manifest has no tile_path column")
        missing = [
            value
            for value in manifest["tile_path"].head(1000)
            if not (paths.base / str(value)).is_file()
        ]
        if missing and paths.he_image is None:
            raise FileNotFoundError(
                "Existing tile manifest references missing tile PNG(s), and no H&E "
                f"image is available for in-memory QC crops; for example: {missing[0]}"
            )
    return all_tiles, selected


def manifest_tile_files_available(paths: SamplePaths, manifest: pd.DataFrame) -> bool:
    if "tile_path" not in manifest.columns:
        return False
    return bool(
        manifest["tile_path"].map(lambda value: (paths.base / str(value)).is_file()).all()
    )


def load_tile_image_for_qc(
    paths: SamplePaths,
    row: object,
    config: PipelineConfig,
) -> Image.Image:
    tile_path = paths.base / str(getattr(row, "tile_path"))
    if tile_path.is_file():
        with Image.open(tile_path) as image:
            return image.convert("RGB")
    if paths.he_image is None:
        raise FileNotFoundError(f"Missing tile PNG and H&E image for {tile_path}")
    with tifffile.TiffFile(paths.he_image) as tif:
        store = tif.series[0].levels[0].aszarr()
        try:
            image_array = resolve_zarr_array(zarr.open(store, mode="r"))
            tile = extract_rgb_tile(
                image_array,
                int(getattr(row, "x0_px")),
                int(getattr(row, "y0_px")),
                int(getattr(row, "x1_px")),
                int(getattr(row, "y1_px")),
            )
        finally:
            close = getattr(store, "close", None)
            if close:
                close()
    expected = (config.tile_size_px, config.tile_size_px, 3)
    if tile.shape != expected:
        raise ValueError(f"Unexpected in-memory QC tile shape {tile.shape}; expected {expected}")
    return Image.fromarray(tile).convert("RGB")


def existing_tile_count(paths: SamplePaths) -> int | object:
    labels_path = paths.base / f"{paths.name}_tile_labels.csv"
    if not labels_path.is_file():
        return pd.NA
    return len(pd.read_csv(labels_path))


def previous_completion_summary(
    paths: SamplePaths,
    *,
    completion_marker_suffix: str = "_pipeline_complete.json",
) -> dict[str, object]:
    marker = completion_marker(paths, completion_marker_suffix)
    if marker.suffix == ".json":
        return json.loads(marker.read_text())
    manifest = existing_manifest(paths)
    all_manifest_path = paths.base / f"{paths.name}_all_tile_labels_with_paths.csv"
    n_written_tiles = (
        len(pd.read_csv(all_manifest_path))
        if all_manifest_path.is_file()
        else (len(manifest) if not manifest.empty else pd.NA)
    )
    return {
        "sample": paths.name,
        "alignment_mode": pd.NA,
        "label_source": pd.NA,
        "n_cells": pd.NA,
        "n_cells_inside_he": pd.NA,
        "n_labeled_tiles": existing_tile_count(paths),
        "n_filter_pass_tiles": len(manifest) if not manifest.empty else pd.NA,
        "n_written_tiles": n_written_tiles,
        "n_retained_tiles": len(manifest) if not manifest.empty else pd.NA,
    }


def run_sample(paths: SamplePaths, config: PipelineConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    print(f"\n=== Processing {paths.name} ===", flush=True)
    paths.base.mkdir(parents=True, exist_ok=True)
    validate_sample(
        paths,
        selective_archive_extract=config.selective_archive_extract,
        require_alignment=config.require_alignment,
        require_cell_groups=config.require_cell_groups,
    )
    cells, zarr_order = load_cells(paths)
    cells, label_source = add_cell_labels(
        paths,
        cells,
        zarr_order,
        config.label_mode,
        missing_label=config.missing_cell_label,
        cell_groups_label_source=config.cell_groups_label_source,
    )
    metadata = load_image_metadata(paths)
    cells, pixel_size, alignment_mode, xenium_to_he = align_cells(paths, cells, metadata)
    if config.make_qc_plots:
        save_alignment_plot(paths, cells, metadata, alignment_mode)
        save_cluster_plots(paths, cells, metadata)
    tile_cells, labels, tile_metrics = create_tile_labels(
        paths,
        cells,
        metadata,
        config,
        pixel_size,
        xenium_to_he,
    )
    if config.make_qc_plots:
        save_tile_filter_qc_plot(paths, tile_metrics, config)
    if not config.recreate_qc_plots:
        cells.to_csv(paths.base / f"{paths.name}{config.cells_output_suffix}", index=False)
    if config.tile_qc_only:
        record = {
            "sample": paths.name,
            "status": "tile_qc_only",
            "alignment_mode": alignment_mode,
            "label_source": label_source,
            "n_cells": len(cells),
            "n_cells_inside_he": int(cells["inside_he_image"].sum()),
            "n_complete_tiles": len(tile_metrics),
            "n_labeled_tiles": len(labels),
            "n_filter_pass_tiles": int(tile_metrics["passes_tile_filter"].sum()),
            "n_written_tiles": 0,
            "n_retained_tiles": 0,
            "n_pruned_tiles": 0,
        }
        return pd.DataFrame(), {**record, "error": ""}
    if config.recreate_qc_plots:
        all_manifest, manifest = existing_tile_manifests_for_qc(paths)
        if config.make_qc_plots:
            cell_boundaries = load_boundaries(paths, cells, boundary_kind="cell")
            nucleus_boundaries = load_boundaries(paths, cells, boundary_kind="nucleus")
            save_tile_filter_quadrant_examples(
                paths,
                all_manifest,
                config,
                tile_cells=tile_cells,
                cell_boundaries=cell_boundaries,
                nucleus_boundaries=nucleus_boundaries,
                metadata=metadata,
                pixel_size=pixel_size,
                xenium_to_he=xenium_to_he,
            )
            save_tile_preview(
                paths,
                tile_cells,
                manifest,
                cell_boundaries,
                nucleus_boundaries,
                metadata,
                pixel_size,
                xenium_to_he,
                config,
            )
            save_orientation_plot(
                paths,
                tile_cells,
                manifest,
                cell_boundaries,
                nucleus_boundaries,
                metadata,
                pixel_size,
                xenium_to_he,
                config,
            )
            if manifest_tile_files_available(paths, manifest):
                save_reconstruction_plot(paths, tile_cells, manifest, cell_boundaries, metadata, pixel_size, xenium_to_he, config)
            else:
                print(
                    f"[{paths.name}] skipped reconstruction QC refresh because "
                    "tile PNGs are not present"
                )
        record = {
            "sample": paths.name,
            "status": "qc_recreated",
            "alignment_mode": alignment_mode,
            "label_source": label_source,
            "n_cells": len(cells),
            "n_cells_inside_he": int(cells["inside_he_image"].sum()),
            "n_complete_tiles": len(tile_metrics),
            "n_labeled_tiles": len(labels),
            "n_filter_pass_tiles": int(tile_metrics["passes_tile_filter"].sum()),
            "n_written_tiles": len(all_manifest),
            "n_retained_tiles": len(manifest),
            "n_pruned_tiles": 0,
        }
        combined = manifest.copy()
        combined["tile_path"] = combined["tile_path"].map(
            lambda value: (Path(paths.name) / str(value)).as_posix()
        )
        return combined, {**record, "error": ""}
    all_manifest, manifest = crop_tiles(paths, labels, config)
    n_pruned_tiles = int(manifest.attrs.get("n_pruned_tiles", 0))
    if config.make_qc_plots:
        cell_boundaries = load_boundaries(paths, cells, boundary_kind="cell")
        nucleus_boundaries = load_boundaries(paths, cells, boundary_kind="nucleus")
        save_tile_filter_quadrant_examples(
            paths,
            all_manifest,
            config,
            tile_cells=tile_cells,
            cell_boundaries=cell_boundaries,
            nucleus_boundaries=nucleus_boundaries,
            metadata=metadata,
            pixel_size=pixel_size,
            xenium_to_he=xenium_to_he,
        )
        save_tile_preview(paths, tile_cells, manifest, cell_boundaries, nucleus_boundaries, metadata, pixel_size, xenium_to_he, config)
        save_orientation_plot(paths, tile_cells, manifest, cell_boundaries, nucleus_boundaries, metadata, pixel_size, xenium_to_he, config)
        save_reconstruction_plot(paths, tile_cells, manifest, cell_boundaries, metadata, pixel_size, xenium_to_he, config)
    record = {
        "sample": paths.name,
        "status": "success",
        "alignment_mode": alignment_mode,
        "label_source": label_source,
        "n_cells": len(cells),
        "n_cells_inside_he": int(cells["inside_he_image"].sum()),
        "n_complete_tiles": len(tile_metrics),
        "n_labeled_tiles": len(labels),
        "n_filter_pass_tiles": int(tile_metrics["passes_tile_filter"].sum()),
        "n_written_tiles": len(all_manifest),
        "n_retained_tiles": len(manifest),
        "n_pruned_tiles": n_pruned_tiles,
    }
    marker = completion_marker(paths, config.completion_marker_suffix)
    if marker.suffix == ".json":
        marker.write_text(json.dumps(record, indent=2) + "\n")
    combined = manifest.copy()
    combined["tile_path"] = combined["tile_path"].map(lambda value: (Path(paths.name) / value).as_posix())
    return combined, {**record, "error": ""}




def write_xenium_explorer_imagealignment_csv(
    path: str | Path,
    xenium_to_he: np.ndarray,
) -> np.ndarray:
    """Write a Xenium Explorer-compatible 3x3 imagealignment CSV."""
    imagealignment = xenium_explorer_imagealignment_matrix(xenium_to_he)
    np.savetxt(Path(path), imagealignment, delimiter=",", fmt="%.15g")
    return imagealignment


XENIUM_EXPLORER_ALIGNMENT_HELP_URL = (
    "https://www.10xgenomics.com/support/software/xenium-explorer/latest/"
    "tutorials/xe-image-alignment"
)


def _validate_he_to_xenium_matrix(matrix: np.ndarray, source: Path) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"{source} does not contain a 3x3 matrix: {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{source} contains non-finite matrix values")
    if not np.allclose(matrix[2], [0, 0, 1], atol=1e-9):
        raise ValueError(f"{source} final row is not [0, 0, 1]: {matrix[2]}")
    return matrix


def _read_alignment_zip_member(
    zip_handle: zipfile.ZipFile,
    basename: str,
) -> tuple[str, bytes] | None:
    matches = [
        name
        for name in zip_handle.namelist()
        if Path(name).name.lower() == basename.lower()
        and not name.endswith("/")
    ]
    if not matches:
        return None
    name = sorted(matches)[0]
    return name, zip_handle.read(name)


def _candidate_alignment_zips(sample: SamplePaths) -> list[Path]:
    exact_names = [
        f"{sample.name}_he_image_alignment_files.zip",
        f"{sample.name}_he_imagealignment_files.zip",
        f"{sample.name}_alignment_files.zip",
    ]
    candidates = [sample.base / name for name in exact_names]
    candidates.extend(sorted(sample.base.glob("*alignment_files.zip")))
    return [
        path
        for path in dict.fromkeys(candidates)
        if path.is_file()
    ]


def samples_missing_imagealignment(
    root: str | Path,
    sample_names: list[str] | None = None,
) -> list[str]:
    """Return selected samples that have H&E images but no active alignment CSV."""
    root = Path(root).expanduser().resolve()
    return [
        sample.name
        for sample in discover_samples(root, sample_names)
        if sample.he_image is not None
        and (sample.alignment is None or not sample.alignment.is_file())
    ]


def import_xenium_explorer_alignment_zips(
    root: str | Path,
    sample_names: list[str] | None = None,
    *,
    overwrite: bool = True,
) -> pd.DataFrame:
    """Import Xenium Explorer alignment zip files placed in sample folders.

    Xenium Explorer's "Download Image Alignment File" action can export a
    zipped folder containing ``matrix.csv`` and ``keypoints.csv``.  For each
    selected sample, this helper reads ``matrix.csv`` and writes the active
    ``<sample>_he_imagealignment.csv`` that the tiling pipeline consumes.  When
    present, ``keypoints.csv`` is copied beside the matrix for provenance.
    """
    root = Path(root).expanduser().resolve()
    records: list[dict[str, object]] = []
    for sample in discover_samples(root, sample_names):
        output_csv = sample.base / f"{sample.name}_he_imagealignment.csv"
        keypoints_csv = sample.base / f"{sample.name}_he_imagealignment_keypoints.csv"
        zips = _candidate_alignment_zips(sample)
        if not zips:
            if output_csv.is_file():
                records.append(
                    {
                        "sample": sample.name,
                        "status": "existing_csv_no_zip",
                        "zip": "",
                        "alignment_csv": str(output_csv),
                        "keypoints_csv": (
                            str(keypoints_csv) if keypoints_csv.is_file() else ""
                        ),
                        "error": "",
                    }
                )
                continue
            records.append(
                {
                    "sample": sample.name,
                    "status": "missing_zip",
                    "zip": "",
                    "alignment_csv": str(output_csv),
                    "keypoints_csv": "",
                    "error": (
                        "Place the Xenium Explorer alignment zip in the sample "
                        "folder and rerun this cell."
                    ),
                }
            )
            continue
        if output_csv.is_file() and not overwrite:
            records.append(
                {
                    "sample": sample.name,
                    "status": "skipped_existing_csv",
                    "zip": str(zips[0]),
                    "alignment_csv": str(output_csv),
                    "keypoints_csv": str(keypoints_csv) if keypoints_csv.is_file() else "",
                    "error": "",
                }
            )
            continue
        zip_path = zips[0]
        try:
            with zipfile.ZipFile(zip_path) as zip_handle:
                matrix_member = _read_alignment_zip_member(zip_handle, "matrix.csv")
                if matrix_member is None:
                    raise FileNotFoundError("matrix.csv was not found in the zip")
                matrix_name, matrix_bytes = matrix_member
                matrix = np.loadtxt(
                    matrix_bytes.decode("utf-8-sig").splitlines(),
                    delimiter=",",
                )
                matrix = _validate_he_to_xenium_matrix(matrix, zip_path / matrix_name)
                np.savetxt(output_csv, matrix, delimiter=",", fmt="%.15g")

                keypoints_member = _read_alignment_zip_member(zip_handle, "keypoints.csv")
                keypoints_path = ""
                if keypoints_member is not None:
                    _, keypoints_bytes = keypoints_member
                    keypoints_csv.write_bytes(keypoints_bytes)
                    keypoints_path = str(keypoints_csv)
            records.append(
                {
                    "sample": sample.name,
                    "status": "imported",
                    "zip": str(zip_path),
                    "alignment_csv": str(output_csv),
                    "keypoints_csv": keypoints_path,
                    "error": "",
                }
            )
        except Exception as error:
            records.append(
                {
                    "sample": sample.name,
                    "status": "failed",
                    "zip": str(zip_path),
                    "alignment_csv": str(output_csv),
                    "keypoints_csv": "",
                    "error": str(error),
                }
            )
    summary = pd.DataFrame(records)
    summary.to_csv(root / "xenium_explorer_alignment_import_summary.csv", index=False)
    return summary


def ensure_manual_imagealignment_csv(paths: SamplePaths) -> Path | None:
    """Export a manual JSON transform as an Explorer-style CSV when needed."""
    if paths.alignment is not None or not paths.manual_alignment.is_file():
        return paths.alignment
    with paths.manual_alignment.open(encoding="utf-8") as handle:
        manual = json.load(handle)
    matrix = np.asarray(
        manual["matrix_xenium_full_pixel_to_he_full_pixel"],
        dtype=float,
    )
    imagealignment_path = paths.base / f"{paths.name}_he_imagealignment.csv"
    he_to_xenium = write_xenium_explorer_imagealignment_csv(
        imagealignment_path,
        matrix,
    )
    manual.setdefault(
        "xenium_explorer_matrix_convention",
        (
            "3x3 row-major affine matrix mapping H&E full-resolution image "
            "pixels to Xenium full-resolution pixels; final row is [0, 0, 1]."
        ),
    )
    manual["matrix_he_full_pixel_to_xenium_full_pixel"] = he_to_xenium.tolist()
    manual["xenium_explorer_imagealignment_csv"] = str(imagealignment_path)
    paths.manual_alignment.write_text(json.dumps(manual, indent=2) + "\n")
    return imagealignment_path


def run_alignment_qc(
    root: str | Path,
    sample_names: list[str] | None = None,
    *,
    selective_archive_extract: bool = False,
    require_alignment: bool | None = None,
    require_cell_groups: bool = False,
    allow_manual_alignment: bool = True,
    fail_fast: bool = False,
) -> pd.DataFrame:
    """Create only the Xenium/H&E alignment QC plot for each sample."""
    root = Path(root).expanduser().resolve()
    samples = discover_samples(root, sample_names)
    print("Alignment QC samples:\n" + "\n".join(f"  - {sample.name}" for sample in samples))
    records: list[dict[str, object]] = []
    for sample in samples:
        try:
            validate_sample(
                sample,
                selective_archive_extract=selective_archive_extract,
                require_alignment=require_alignment,
                require_cell_groups=require_cell_groups,
                allow_manual_alignment=allow_manual_alignment,
            )
            alignment_file = (
                ensure_manual_imagealignment_csv(sample)
                if allow_manual_alignment
                else sample.alignment
            )
            cells, _ = load_cells(sample)
            metadata = load_image_metadata(sample)
            cells, _, alignment_mode, _ = align_cells(sample, cells, metadata)
            save_alignment_plot(sample, cells, metadata, alignment_mode)
            records.append(
                {
                    "sample": sample.name,
                    "status": "success",
                    "alignment_mode": alignment_mode,
                    "alignment_file": str(alignment_file) if alignment_file else "",
                    "manual_alignment_json": (
                        str(sample.manual_alignment)
                        if sample.manual_alignment.is_file()
                        else ""
                    ),
                    "alignment_plot": str(
                        sample.base / f"{sample.name}_xenium_he_alignment.png"
                    ),
                    "n_cells": len(cells),
                    "n_cells_inside_he": int(cells["inside_he_image"].sum()),
                    "inside_he_fraction": float(cells["inside_he_image"].mean()),
                    "error": "",
                }
            )
        except Exception as error:
            print(f"[{sample.name}] alignment QC FAILED: {error}", file=sys.stderr)
            traceback.print_exc()
            records.append(
                {
                    "sample": sample.name,
                    "status": "failed",
                    "alignment_mode": "",
                    "alignment_file": str(sample.alignment) if sample.alignment else "",
                    "manual_alignment_json": (
                        str(sample.manual_alignment)
                        if sample.manual_alignment.is_file()
                        else ""
                    ),
                    "alignment_plot": "",
                    "n_cells": pd.NA,
                    "n_cells_inside_he": pd.NA,
                    "inside_he_fraction": pd.NA,
                    "error": str(error),
                }
            )
            if fail_fast:
                break
        finally:
            plt.close("all")
            gc.collect()
    summary = pd.DataFrame(records)
    summary.to_csv(root / "alignment_qc_summary.csv", index=False)
    return summary


def save_combined_tile_filter_metrics(root: Path, samples: list[SamplePaths]) -> pd.DataFrame:
    metric_tables: list[pd.DataFrame] = []
    for sample in samples:
        path = sample.base / f"{sample.name}_tile_filter_metrics.csv"
        if path.is_file():
            metric_tables.append(pd.read_csv(path))
    if not metric_tables:
        return pd.DataFrame()
    combined = pd.concat(metric_tables, ignore_index=True, sort=False)
    combined.to_csv(root / "all_samples_tile_filter_metrics.csv", index=False)

    quantiles = [0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    rows: list[pd.DataFrame] = []
    groups = [("ALL", combined)]
    groups.extend((str(name), group) for name, group in combined.groupby("sample", sort=True))
    for name, group in groups:
        n_cells = pd.to_numeric(group["n_cells"], errors="coerce").fillna(0)
        area_fraction = pd.to_numeric(group["cell_area_fraction"], errors="coerce")
        finite_area = area_fraction[np.isfinite(area_fraction)]
        rows.append(
            pd.DataFrame(
                {
                    "sample": name,
                    "quantile": quantiles,
                    "n_cells": n_cells.quantile(quantiles).to_numpy(),
                    "cell_area_fraction": (
                        finite_area.quantile(quantiles).to_numpy()
                        if len(finite_area)
                        else np.full(len(quantiles), np.nan)
                    ),
                }
            )
        )
    pd.concat(rows, ignore_index=True).to_csv(
        root / "all_samples_tile_filter_quantiles.csv",
        index=False,
    )
    return combined


def load_tile_filter_metrics(
    root: str | Path,
    sample_names: list[str] | None = None,
) -> pd.DataFrame:
    """Load per-sample tile-filter metrics for notebook cutoff exploration."""
    root = Path(root).expanduser().resolve()
    if sample_names:
        paths = [
            root / sample / f"{sample}_tile_filter_metrics.csv"
            for sample in sample_names
        ]
    else:
        paths = sorted(root.glob("*/*_tile_filter_metrics.csv"))
    tables: list[pd.DataFrame] = []
    for path in paths:
        if not path.is_file():
            continue
        table = pd.read_csv(path)
        if "sample" not in table.columns:
            table.insert(0, "sample", path.parent.name)
        tables.append(table)
    if not tables:
        raise FileNotFoundError(
            f"No *_tile_filter_metrics.csv files found under {root}. "
            "Run the tile-filter QC notebook cell first."
        )
    metrics = pd.concat(tables, ignore_index=True, sort=False)
    metrics["sample"] = metrics["sample"].astype(str)
    metrics["n_cells"] = pd.to_numeric(metrics["n_cells"], errors="coerce").fillna(0)
    metrics["cell_area_fraction"] = pd.to_numeric(
        metrics["cell_area_fraction"],
        errors="coerce",
    )
    if "cell_area_um2" in metrics.columns:
        metrics["cell_area_um2"] = pd.to_numeric(
            metrics["cell_area_um2"],
            errors="coerce",
        ).fillna(0.0)
    return metrics


def tile_filter_selection_summary(
    tile_metrics: pd.DataFrame,
    min_cells_per_tile: int,
    min_cell_area_fraction: float,
    min_nucleus_area_fraction: float = 0.0,
    min_he_tissue_fraction: float = 0.0,
    *,
    nucleus_area_fraction_column: str = "nucleus_area_fraction",
    require_nucleus_area_fraction: bool = False,
    he_tissue_fraction_column: str = "he_tissue_fraction",
    require_he_tissue_fraction: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Summarize tiles and cells retained by a prospective tile filter."""
    if tile_metrics.empty:
        raise ValueError("tile_metrics is empty")
    n_cells = pd.to_numeric(tile_metrics["n_cells"], errors="coerce").fillna(0)
    area_fraction = pd.to_numeric(tile_metrics["cell_area_fraction"], errors="coerce")
    nucleus_area_fraction = pd.to_numeric(
        tile_metrics.get(
            nucleus_area_fraction_column,
            pd.Series(np.nan, index=tile_metrics.index),
        ),
        errors="coerce",
    )
    he_tissue_fraction = pd.to_numeric(
        tile_metrics.get(
            he_tissue_fraction_column,
            pd.Series(np.nan, index=tile_metrics.index),
        ),
        errors="coerce",
    )
    has_nucleus_area_fraction = bool(nucleus_area_fraction.notna().any())
    has_he_tissue_fraction = bool(he_tissue_fraction.notna().any())
    nucleus_cutoff_applies = (
        float(min_nucleus_area_fraction) > 0
        and (has_nucleus_area_fraction or require_nucleus_area_fraction)
    )
    tissue_cutoff_applies = (
        float(min_he_tissue_fraction) > 0
        and (has_he_tissue_fraction or require_he_tissue_fraction)
    )
    passes = (n_cells > 0) & (n_cells >= int(min_cells_per_tile))
    if min_cell_area_fraction > 0:
        passes &= area_fraction >= float(min_cell_area_fraction)
    if nucleus_cutoff_applies:
        passes &= nucleus_area_fraction.fillna(0.0) >= float(min_nucleus_area_fraction)
    if tissue_cutoff_applies:
        passes &= he_tissue_fraction.fillna(0.0) >= float(min_he_tissue_fraction)

    total_tiles = len(tile_metrics)
    tiles_with_cells = int((n_cells > 0).sum())
    passing_tiles = int(passes.sum())
    total_cells = int(n_cells.sum())
    retained_cells = int(n_cells[passes].sum())
    total_cell_area = (
        float(tile_metrics["cell_area_um2"].sum())
        if "cell_area_um2" in tile_metrics.columns
        else np.nan
    )
    retained_cell_area = (
        float(tile_metrics.loc[passes, "cell_area_um2"].sum())
        if "cell_area_um2" in tile_metrics.columns
        else np.nan
    )
    total_nucleus_area = (
        float(tile_metrics["nucleus_area_um2"].sum())
        if "nucleus_area_um2" in tile_metrics.columns
        else np.nan
    )
    retained_nucleus_area = (
        float(tile_metrics.loc[passes, "nucleus_area_um2"].sum())
        if "nucleus_area_um2" in tile_metrics.columns
        else np.nan
    )
    summary = pd.DataFrame(
        [
            {
                "min_cells_per_tile": int(min_cells_per_tile),
                "min_cell_area_fraction": float(min_cell_area_fraction),
                "min_nucleus_area_fraction": float(min_nucleus_area_fraction),
                "min_he_tissue_fraction": float(min_he_tissue_fraction),
                "nucleus_area_fraction_column": nucleus_area_fraction_column,
                "he_tissue_fraction_column": he_tissue_fraction_column,
                "has_nucleus_area_fraction": has_nucleus_area_fraction,
                "has_he_tissue_fraction": has_he_tissue_fraction,
                "nucleus_area_fraction_cutoff_applied": nucleus_cutoff_applies,
                "he_tissue_fraction_cutoff_applied": tissue_cutoff_applies,
                "complete_tiles": total_tiles,
                "tiles_with_cells": tiles_with_cells,
                "retained_tiles": passing_tiles,
                "retained_tile_fraction": passing_tiles / total_tiles,
                "cells_in_complete_tiles": total_cells,
                "cells_in_retained_tiles": retained_cells,
                "retained_cell_fraction": (
                    retained_cells / total_cells
                    if total_cells
                    else np.nan
                ),
                "cell_area_um2_in_complete_tiles": total_cell_area,
                "cell_area_um2_in_retained_tiles": retained_cell_area,
                "retained_cell_area_fraction": (
                    retained_cell_area / total_cell_area
                    if total_cell_area and np.isfinite(total_cell_area)
                    else np.nan
                ),
                "nucleus_area_um2_in_complete_tiles": total_nucleus_area,
                "nucleus_area_um2_in_retained_tiles": retained_nucleus_area,
                "retained_nucleus_area_fraction": (
                    retained_nucleus_area / total_nucleus_area
                    if total_nucleus_area and np.isfinite(total_nucleus_area)
                    else np.nan
                ),
            }
        ]
    )
    return summary, passes


def tile_filter_quantile_summary(tile_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return decision-friendly quantiles for tile-filter metrics."""
    quantiles = [0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    n_cells = pd.to_numeric(tile_metrics["n_cells"], errors="coerce").fillna(0)
    area_fraction = pd.to_numeric(tile_metrics["cell_area_fraction"], errors="coerce")
    finite_area = area_fraction[np.isfinite(area_fraction)]
    return pd.DataFrame(
        {
            "quantile": quantiles,
            "n_cells": n_cells.quantile(quantiles).to_numpy(),
            "cell_area_fraction": (
                finite_area.quantile(quantiles).to_numpy()
                if len(finite_area)
                else np.full(len(quantiles), np.nan)
            ),
        }
    )


def plot_tile_filter_selection(
    tile_metrics: pd.DataFrame,
    min_cells_per_tile: int,
    min_cell_area_fraction: float,
    min_nucleus_area_fraction: float = 0.0,
    min_he_tissue_fraction: float = 0.0,
    *,
    title: str = "Tile-filter threshold explorer",
    max_scatter_points: int = 50_000,
    area_fraction_label: str = "cell_area_fraction",
    nucleus_area_fraction_column: str = "nucleus_area_fraction",
    nucleus_area_fraction_label: str | None = None,
    require_nucleus_area_fraction: bool = False,
    he_tissue_fraction_column: str = "he_tissue_fraction",
    require_he_tissue_fraction: bool = False,
) -> plt.Figure:
    """Create an interactive-friendly static view for a prospective filter."""
    nucleus_area_fraction_label = (
        nucleus_area_fraction_label or nucleus_area_fraction_column
    )
    tile_metrics = apply_tile_filter_cutoffs(
        tile_metrics,
        min_cells_per_tile,
        min_cell_area_fraction,
        min_nucleus_area_fraction,
        min_he_tissue_fraction,
        nucleus_area_fraction_column=nucleus_area_fraction_column,
        require_nucleus_area_fraction=require_nucleus_area_fraction,
        he_tissue_fraction_column=he_tissue_fraction_column,
        require_he_tissue_fraction=require_he_tissue_fraction,
    )
    summary, _passes = tile_filter_selection_summary(
        tile_metrics,
        min_cells_per_tile,
        min_cell_area_fraction,
        min_nucleus_area_fraction,
        min_he_tissue_fraction,
        nucleus_area_fraction_column=nucleus_area_fraction_column,
        require_nucleus_area_fraction=require_nucleus_area_fraction,
        he_tissue_fraction_column=he_tissue_fraction_column,
        require_he_tissue_fraction=require_he_tissue_fraction,
    )
    n_cells = pd.to_numeric(tile_metrics["n_cells"], errors="coerce").fillna(0)
    area_fraction = pd.to_numeric(tile_metrics["cell_area_fraction"], errors="coerce")
    finite_area = area_fraction[np.isfinite(area_fraction)]
    nucleus_area_fraction = pd.to_numeric(
        tile_metrics.get(
            "nucleus_area_fraction_for_filter",
            tile_metrics.get(
                nucleus_area_fraction_column,
                pd.Series(np.nan, index=tile_metrics.index),
            ),
        ),
        errors="coerce",
    )
    finite_nucleus_area = nucleus_area_fraction[np.isfinite(nucleus_area_fraction)]
    he_tissue_fraction = pd.to_numeric(
        tile_metrics.get(
            "he_tissue_fraction_for_filter",
            tile_metrics.get(
                he_tissue_fraction_column,
                pd.Series(np.nan, index=tile_metrics.index),
            ),
        ),
        errors="coerce",
    )
    finite_he_tissue = he_tissue_fraction[np.isfinite(he_tissue_fraction)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    max_cells = int(n_cells.max()) if len(n_cells) else 0
    cell_bins = np.arange(-0.5, max_cells + 1.5, 1) if max_cells <= 80 else 80
    axes[0, 0].hist(n_cells, bins=cell_bins, color="#4C78A8", edgecolor="white")
    axes[0, 0].axvline(min_cells_per_tile, color="#D62728", linewidth=2)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Cells per complete tile")
    axes[0, 0].set_xlabel("n_cells")
    axes[0, 0].set_ylabel("Tiles (log scale)")

    if len(finite_area):
        axes[0, 1].hist(finite_area, bins=80, color="#59A14F", edgecolor="white")
        axes[0, 1].axvline(min_cell_area_fraction, color="#D62728", linewidth=2)
        axes[0, 1].set_yscale("log")
    else:
        axes[0, 1].text(0.5, 0.5, f"No {area_fraction_label} values", ha="center", va="center")
    axes[0, 1].set_title("Area fraction per complete tile")
    axes[0, 1].set_xlabel(area_fraction_label)
    axes[0, 1].set_ylabel("Tiles (log scale)")

    if len(finite_nucleus_area):
        axes[0, 2].hist(
            finite_nucleus_area,
            bins=80,
            color="#B07AA1",
            edgecolor="white",
        )
        if float(min_nucleus_area_fraction) > 0:
            axes[0, 2].axvline(
                min_nucleus_area_fraction,
                color="#D62728",
                linewidth=2,
            )
        axes[0, 2].set_yscale("log")
    else:
        axes[0, 2].text(
            0.5,
            0.5,
            f"No {nucleus_area_fraction_label} values",
            ha="center",
            va="center",
        )
    axes[0, 2].set_title("Nucleus area fraction per complete tile")
    axes[0, 2].set_xlabel(nucleus_area_fraction_label)
    axes[0, 2].set_ylabel("Tiles (log scale)")

    scatter = (
        tile_metrics.sample(n=max_scatter_points, random_state=1)
        if len(tile_metrics) > max_scatter_points
        else tile_metrics
    )
    scatter_cells = pd.to_numeric(scatter["n_cells"], errors="coerce").fillna(0)
    scatter_area = pd.to_numeric(scatter["cell_area_fraction"], errors="coerce")
    colors = scatter["tile_filter_quadrant"].map(TILE_FILTER_QUADRANT_COLORS).fillna("#B8B8B8")
    axes[1, 0].scatter(scatter_cells, scatter_area, s=4, c=colors, alpha=0.45, linewidths=0)
    axes[1, 0].axvline(min_cells_per_tile, color="#D62728", linewidth=1.5)
    axes[1, 0].axhline(min_cell_area_fraction, color="#D62728", linewidth=1.5)
    axes[1, 0].set_title("Joint tile-filter metrics")
    axes[1, 0].set_xlabel("n_cells")
    axes[1, 0].set_ylabel(area_fraction_label)
    axes[1, 0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=TILE_FILTER_QUADRANT_COLORS[key],
                markersize=5,
                label=TILE_FILTER_QUADRANT_LABELS[key],
            )
            for key in TILE_FILTER_QUADRANTS
        ],
        loc="best",
        fontsize=7,
        frameon=False,
    )

    scatter_nucleus = pd.to_numeric(
        scatter.get(
            "nucleus_area_fraction_for_filter",
            scatter.get(
                nucleus_area_fraction_column,
                pd.Series(np.nan, index=scatter.index),
            ),
        ),
        errors="coerce",
    )
    finite_joint = scatter_area.notna() & scatter_nucleus.notna()
    if finite_joint.any():
        axes[1, 1].scatter(
            scatter_area.loc[finite_joint],
            scatter_nucleus.loc[finite_joint],
            s=4,
            c=colors.loc[finite_joint],
            alpha=0.45,
            linewidths=0,
        )
        axes[1, 1].axvline(min_cell_area_fraction, color="#D62728", linewidth=1.5)
        if float(min_nucleus_area_fraction) > 0:
            axes[1, 1].axhline(
                min_nucleus_area_fraction,
                color="#D62728",
                linewidth=1.5,
            )
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "No paired cell/nucleus area values",
            ha="center",
            va="center",
        )
    axes[1, 1].set_title("Cell vs nucleus area fraction")
    axes[1, 1].set_xlabel(area_fraction_label)
    axes[1, 1].set_ylabel(nucleus_area_fraction_label)

    axes[1, 2].axis("off")
    row = summary.iloc[0]
    def percent_text(value: float) -> str:
        return f"{value:.1%}" if pd.notna(value) and np.isfinite(value) else "n/a"

    text_lines = [
        f"Retained tiles: {int(row.retained_tiles):,} / {int(row.complete_tiles):,} "
        f"({row.retained_tile_fraction:.1%})",
        f"Cells in retained tiles: {int(row.cells_in_retained_tiles):,} / "
        f"{int(row.cells_in_complete_tiles):,} ({row.retained_cell_fraction:.1%})",
        f"Cell area retained: {percent_text(row.retained_cell_area_fraction)}",
        f"Nucleus area retained: {percent_text(row.retained_nucleus_area_fraction)}",
        "",
        f"min_cells_per_tile = {int(min_cells_per_tile)}",
        f"min_{area_fraction_label} = {float(min_cell_area_fraction):g}",
        f"min_{nucleus_area_fraction_label} = {float(min_nucleus_area_fraction):g}",
        f"min_{he_tissue_fraction_column} = {float(min_he_tissue_fraction):g}",
        f"nucleus cutoff applied = {bool(row.nucleus_area_fraction_cutoff_applied)}",
        f"H&E tissue cutoff applied = {bool(row.he_tissue_fraction_cutoff_applied)}",
        "",
        "Median / 90th / 95th percentile",
        (
            "n_cells: "
            f"{n_cells.quantile(0.50):.1f} / "
            f"{n_cells.quantile(0.90):.1f} / "
            f"{n_cells.quantile(0.95):.1f}"
        ),
    ]
    if len(finite_area):
        text_lines.append(
            f"{area_fraction_label}: "
            f"{finite_area.quantile(0.50):.4f} / "
            f"{finite_area.quantile(0.90):.4f} / "
            f"{finite_area.quantile(0.95):.4f}"
        )
    if len(finite_nucleus_area):
        text_lines.append(
            f"{nucleus_area_fraction_label}: "
            f"{finite_nucleus_area.quantile(0.50):.4f} / "
            f"{finite_nucleus_area.quantile(0.90):.4f} / "
            f"{finite_nucleus_area.quantile(0.95):.4f}"
        )
    if len(finite_he_tissue):
        text_lines.append(
            f"{he_tissue_fraction_column}: "
            f"{finite_he_tissue.quantile(0.50):.4f} / "
            f"{finite_he_tissue.quantile(0.90):.4f} / "
            f"{finite_he_tissue.quantile(0.95):.4f}"
        )
    axes[1, 2].text(0, 1, "\n".join(text_lines), va="top", fontsize=11)
    fig.suptitle(title, fontsize=14)
    return fig


def display_tile_filter_explorer(
    root: str | Path,
    sample_names: list[str] | None = None,
    *,
    default_min_cells: int = 1,
    default_min_cell_area_fraction: float = 0.0,
    max_scatter_points: int = 50_000,
) -> object:
    """Display an ipywidgets threshold explorer for tile-filter metrics."""
    metrics = load_tile_filter_metrics(root, sample_names)
    try:
        import io
        import ipywidgets as widgets
        from IPython.display import Image as DisplayImage
        from IPython.display import clear_output, display
    except ImportError as error:
        print(
            "Interactive controls require ipywidgets in the active notebook kernel. "
            "Install ipywidgets, or use plot_tile_filter_selection(...) directly."
        )
        raise error

    samples = ["ALL"] + sorted(metrics["sample"].dropna().astype(str).unique())
    max_cells = max(int(metrics["n_cells"].max()), int(default_min_cells), 1)
    finite_area = metrics["cell_area_fraction"][np.isfinite(metrics["cell_area_fraction"])]
    max_area = float(max(finite_area.max(), default_min_cell_area_fraction, 0.01)) if len(finite_area) else 1.0
    area_step = 0.001 if max_area <= 1 else 0.01

    sample_dropdown = widgets.Dropdown(
        options=samples,
        value="ALL",
        description="Sample",
        layout=widgets.Layout(width="420px"),
    )
    min_cells = widgets.BoundedIntText(
        value=int(default_min_cells),
        min=0,
        max=max_cells,
        step=1,
        description="Min cells",
    )
    min_area = widgets.BoundedFloatText(
        value=float(default_min_cell_area_fraction),
        min=0.0,
        max=max_area,
        step=area_step,
        description="Min area",
    )
    output = widgets.Output()

    def selected_metrics() -> pd.DataFrame:
        if sample_dropdown.value == "ALL":
            return metrics
        return metrics.loc[metrics["sample"] == sample_dropdown.value]

    def redraw(*_: object) -> None:
        current = selected_metrics()
        title = f"{sample_dropdown.value}: tile-filter threshold explorer"
        with output:
            clear_output(wait=True)
            summary, _ = tile_filter_selection_summary(
                current,
                min_cells.value,
                min_area.value,
            )
            quantiles = tile_filter_quantile_summary(current)
            display(summary)
            display(quantiles)
            fig = plot_tile_filter_selection(
                current,
                min_cells.value,
                min_area.value,
                title=title,
                max_scatter_points=max_scatter_points,
            )
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
            display(DisplayImage(data=buffer.getvalue()))
            plt.close(fig)

    for control in (sample_dropdown, min_cells, min_area):
        control.observe(redraw, names="value")

    controls = widgets.VBox(
        [
            widgets.HBox([sample_dropdown, min_cells, min_area]),
            output,
        ]
    )
    display(controls)
    redraw()
    return {
        "widget": controls,
        "metrics": metrics,
        "sample": sample_dropdown,
        "min_cells": min_cells,
        "min_cell_area_fraction": min_area,
    }


def _count_csv_rows(path: Path) -> int:
    return len(pd.read_csv(path, usecols=[0]))


def _existing_cell_summary(root: Path, sample: str) -> dict[str, int]:
    cells_path = root / sample / f"{sample}_cells_with_annotations.csv"
    if not cells_path.is_file():
        return {}
    columns = pd.read_csv(cells_path, nrows=0).columns
    if "inside_he_image" not in columns:
        return {"n_cells": _count_csv_rows(cells_path)}
    inside = pd.read_csv(cells_path, usecols=["inside_he_image"])["inside_he_image"]
    if inside.dtype == bool:
        n_inside = int(inside.sum())
    else:
        n_inside = int(inside.astype(str).str.lower().isin({"true", "1", "yes"}).sum())
    return {"n_cells": len(inside), "n_cells_inside_he": n_inside}


def _existing_written_tile_count(root: Path, sample: str) -> int | None:
    sample_dir = root / sample
    for suffix in (
        "_all_tile_labels_with_paths.csv",
        "_tile_labels_with_paths.csv",
        "_tile_labels.csv",
    ):
        path = sample_dir / f"{sample}{suffix}"
        if path.is_file():
            return _count_csv_rows(path)
    return None


def _compact_axis_labels(labels: Iterable[str]) -> list[str]:
    labels = [str(label) for label in labels]
    if len(labels) < 2:
        return labels
    prefix = os.path.commonprefix(labels)
    underscore_boundaries = [index + 1 for index, char in enumerate(prefix) if char == "_"]
    for boundary in reversed(underscore_boundaries):
        if boundary < 8:
            continue
        compact = [
            label[boundary:] if label.startswith(prefix[:boundary]) else label
            for label in labels
        ]
        if all(len(label) >= 3 for label in compact):
            return compact
    return labels


def _plot_sample_bars(axis: plt.Axes, labels: list[str], values: pd.Series, title: str) -> None:
    positions = np.arange(len(labels))
    axis.bar(positions, np.asarray(values, dtype=float))
    axis.set_title(title)
    axis.set_xlim(-0.5, len(labels) - 0.5)
    axis.set_xticks(positions)
    max_label_length = max((len(label) for label in labels), default=0)
    if max_label_length <= 12:
        axis.set_xticklabels(labels, fontsize=8)
    elif max_label_length <= 25:
        axis.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    else:
        axis.set_xticklabels(labels, rotation=90, ha="center", fontsize=8)


def save_dataset_summary_plot(root: Path, summary: pd.DataFrame) -> None:
    completed = summary.loc[summary["status"].isin(["success", "skipped"])].copy()
    metric_columns = ["n_cells", "n_cells_inside_he", "n_written_tiles"]
    required_columns = ["sample", *metric_columns]
    if completed.empty or "sample" not in completed:
        return
    for column in metric_columns:
        if column not in completed:
            completed[column] = pd.NA
    completed[metric_columns] = completed[metric_columns].apply(pd.to_numeric, errors="coerce")

    for index, row in completed.iterrows():
        sample = str(row["sample"])
        if pd.isna(row["n_cells"]) or pd.isna(row["n_cells_inside_he"]):
            for column, value in _existing_cell_summary(root, sample).items():
                completed.at[index, column] = value
        if pd.isna(row["n_written_tiles"]):
            n_written_tiles = _existing_written_tile_count(root, sample)
            if n_written_tiles is not None:
                completed.at[index, "n_written_tiles"] = n_written_tiles

    completed = completed.dropna(subset=required_columns)
    completed = completed.loc[completed["n_cells"] > 0]
    if completed.empty:
        return
    completed["inside_fraction"] = completed["n_cells_inside_he"] / completed["n_cells"]
    labels = _compact_axis_labels(completed["sample"])
    fig_width = max(17, len(labels) * 1.2)
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, 6), constrained_layout=True)
    _plot_sample_bars(axes[0], labels, completed["n_cells"], "Detected cells")
    _plot_sample_bars(axes[1], labels, completed["inside_fraction"] * 100, "Cells inside H&E (%)")
    _plot_sample_bars(axes[2], labels, completed["n_written_tiles"], "Saved nonempty H&E tiles")
    fig.savefig(root / "dataset_qc_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_all_samples(
    root: str | Path,
    sample_names: list[str] | None = None,
    *,
    input_manifest: str | Path | None = None,
    tile_size: int = 256,
    min_cells_per_tile: int = 1,
    min_cell_area_fraction: float = 0.0,
    preview_count: int = 12,
    pyramid_level: int = 3,
    annotation_alpha: int = 160,
    label_mode: str = "auto",
    overwrite: bool = False,
    prune_stale_tiles: bool = False,
    skip_processed: bool = True,
    make_qc_plots: bool = True,
    validate_only: bool = False,
    tile_qc_only: bool = False,
    recreate_qc_plots: bool = False,
    fail_fast: bool = False,
    selective_archive_extract: bool = False,
    require_alignment: bool | None = None,
    require_cell_groups: bool = False,
    missing_cell_label: str = "Unassigned",
    cell_groups_label_source: str = "custom_cell_groups",
    cells_output_suffix: str = "_cells_with_annotations.csv",
    completion_marker_suffix: str = "_pipeline_complete.json",
    skip_completed_before_validation: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if tile_size <= 0 or min_cells_per_tile <= 0:
        raise ValueError("tile_size and min_cells_per_tile must be positive")
    if min_cell_area_fraction < 0:
        raise ValueError("min_cell_area_fraction must be non-negative")
    if not 0 <= annotation_alpha <= 255:
        raise ValueError("annotation_alpha must be between 0 and 255")
    if label_mode not in {"auto", "graphclust", "none"}:
        raise ValueError("label_mode must be auto, graphclust, or none")
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    samples = (
        discover_samples_from_manifest(input_manifest, sample_names)
        if input_manifest is not None
        else discover_samples(root, sample_names)
    )
    print("Samples:\n" + "\n".join(f"  - {sample.name}" for sample in samples))
    validation: list[dict[str, object]] = []
    for sample in samples:
        try:
            marker = completion_marker(sample, completion_marker_suffix)
            if (
                skip_completed_before_validation
                and skip_processed
                and not overwrite
                and not tile_qc_only
                and not recreate_qc_plots
                and marker.is_file()
            ):
                validation.append(
                    {
                        "sample": sample.name,
                        "status": "valid",
                        "error": "",
                        "skipped_existing": True,
                    }
                )
                print(f"[valid] {sample.name}: existing completion marker")
                continue
            details = validate_sample(
                sample,
                selective_archive_extract=selective_archive_extract,
                require_alignment=require_alignment,
                require_cell_groups=require_cell_groups,
            )
            validation.append({"sample": sample.name, "status": "valid", "error": "", **details})
            print(f"[valid] {sample.name}")
        except Exception as error:
            validation.append({"sample": sample.name, "status": "invalid", "error": str(error)})
            print(f"[invalid] {sample.name}: {error}", file=sys.stderr)
    if validate_only:
        table = pd.DataFrame(validation)
        table.to_csv(root / "validation_summary.csv", index=False)
        return table, pd.DataFrame()

    config = PipelineConfig(
        tile_size_px=tile_size,
        min_cells_per_tile=min_cells_per_tile,
        min_cell_area_fraction=min_cell_area_fraction,
        preview_count=preview_count,
        pyramid_level=pyramid_level,
        annotation_alpha=annotation_alpha,
        overwrite=overwrite,
        prune_stale_tiles=prune_stale_tiles,
        make_qc_plots=make_qc_plots,
        tile_qc_only=tile_qc_only,
        recreate_qc_plots=recreate_qc_plots,
        label_mode=label_mode,
        selective_archive_extract=selective_archive_extract,
        require_alignment=require_alignment,
        require_cell_groups=require_cell_groups,
        missing_cell_label=missing_cell_label,
        cell_groups_label_source=cell_groups_label_source,
        cells_output_suffix=cells_output_suffix,
        completion_marker_suffix=completion_marker_suffix,
    )
    manifests: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    invalid_names = {row["sample"] for row in validation if row["status"] == "invalid"}
    for sample in samples:
        if sample.name in invalid_names:
            summaries.append({"sample": sample.name, "status": "invalid", "error": "input validation failed"})
            continue
        marker = completion_marker(sample, completion_marker_suffix)
        if (
            skip_processed
            and not overwrite
            and not tile_qc_only
            and not recreate_qc_plots
            and marker.is_file()
        ):
            manifest = existing_manifest(sample)
            if not manifest.empty:
                manifests.append(manifest)
            previous = previous_completion_summary(
                sample,
                completion_marker_suffix=completion_marker_suffix,
            )
            summaries.append({**previous, "status": "skipped", "error": f"{marker.name} exists"})
            print(f"[{sample.name}] skipped: {marker.name} exists")
            continue
        try:
            manifest, summary = run_sample(sample, config)
            manifests.append(manifest)
            summaries.append(summary)
        except Exception as error:
            print(f"[{sample.name}] FAILED: {error}", file=sys.stderr)
            traceback.print_exc()
            summaries.append({"sample": sample.name, "status": "failed", "error": str(error)})
            if fail_fast:
                break
        finally:
            plt.close("all")
            gc.collect()
    summary = pd.DataFrame(summaries)
    if not recreate_qc_plots:
        summary.to_csv(root / "pipeline_summary.csv", index=False)
    if not recreate_qc_plots:
        save_combined_tile_filter_metrics(root, samples)
    all_tiles = pd.DataFrame()
    if manifests:
        all_tiles = pd.concat(manifests, ignore_index=True, sort=False)
        label_columns = [
            column for column in all_tiles.columns
            if column.startswith("count__") or column.startswith("proportion__")
        ]
        all_tiles[label_columns] = all_tiles[label_columns].fillna(0)
        if not recreate_qc_plots:
            all_tiles.to_csv(root / "all_samples_tile_labels.csv", index=False)
    if make_qc_plots and not recreate_qc_plots:
        save_dataset_summary_plot(root, summary)
    return summary, all_tiles


def cli_main(
    *,
    default_root: Path | None = None,
    description: str = "Create aligned H&E tiles and XOA-cluster labels.",
    default_options: dict[str, Any] | None = None,
    selective_archive_extract: bool = False,
    require_alignment: bool | None = None,
    require_cell_groups: bool = False,
    missing_cell_label: str = "Unassigned",
    cell_groups_label_source: str = "custom_cell_groups",
    cells_output_suffix: str = "_cells_with_annotations.csv",
    completion_marker_suffix: str = "_pipeline_complete.json",
    skip_completed_before_validation: bool = False,
) -> int:
    args = parse_args(
        default_root=default_root,
        description=description,
        defaults=default_options,
    )
    summary, _ = run_all_samples(
        root=args.root,
        sample_names=args.samples,
        tile_size=args.tile_size,
        min_cells_per_tile=args.min_cells_per_tile,
        min_cell_area_fraction=args.min_cell_area_fraction,
        preview_count=args.preview_count,
        pyramid_level=args.pyramid_level,
        annotation_alpha=args.annotation_alpha,
        label_mode=args.label_mode,
        overwrite=args.overwrite,
        prune_stale_tiles=args.prune_stale_tiles,
        skip_processed=not args.reprocess_completed,
        make_qc_plots=not args.skip_qc_plots,
        validate_only=args.validate_only,
        tile_qc_only=args.tile_qc_only,
        recreate_qc_plots=args.recreate_qc_plots,
        fail_fast=args.fail_fast,
        selective_archive_extract=selective_archive_extract,
        require_alignment=require_alignment,
        require_cell_groups=require_cell_groups,
        missing_cell_label=missing_cell_label,
        cell_groups_label_source=cell_groups_label_source,
        cells_output_suffix=cells_output_suffix,
        completion_marker_suffix=completion_marker_suffix,
        skip_completed_before_validation=skip_completed_before_validation,
    )
    failed = summary["status"].isin(["failed", "invalid"]).any()
    return 1 if failed else 0


def main() -> int:
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
