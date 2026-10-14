"""Cross-dataset tile-filter review helpers.

Dataset-specific notebooks create the nonempty tile inventory.  This module
loads those inventories from every dataset folder and applies prospective
tile-filter thresholds without rerunning the tile extraction pipelines.
"""

from __future__ import annotations

import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]


from .dataset_config import dataset_output_dirs
from . import he_tissue, standard_xenium


DEFAULT_DATASET_OUTPUTS = dataset_output_dirs()

AREA_OCCUPANCY_METHOD_VERSION = "boundary_union_he_pixels_v1"
HE_TISSUE_MASK_METHOD_VERSION = he_tissue.MASK_METHOD_VERSION
HE_TISSUE_MIN_MASK_PIXELS_PER_TILE = 16
HE_TISSUE_DARK_NEUTRAL_VALUE_MAX = he_tissue.DARK_NEUTRAL_VALUE_MAX
HE_TISSUE_DARK_NEUTRAL_CHROMA_MAX = he_tissue.DARK_NEUTRAL_CHROMA_MAX
HE_TISSUE_BRIGHT_NEUTRAL_VALUE_MIN = he_tissue.BRIGHT_NEUTRAL_VALUE_MIN
HE_TISSUE_BRIGHT_NEUTRAL_CHROMA_MAX = he_tissue.BRIGHT_NEUTRAL_CHROMA_MAX

SKIP_SAMPLE_DIRS = {
    "__pycache__",
    "he_tiles",
    "images",
    "logs",
    "mapped_side_by_side",
    "mapping_previews",
    "manual_alignment_from_mapping",
    "manual_alignment_reviews",
    "tile_outputs",
    "full_resolution_he",
    "selected",
}


@dataclass(frozen=True)
class TileFilterSample:
    dataset: str
    sample: str
    dataset_output_dir: Path
    sample_dir: Path
    manifest_path: Path
    manifest_scope: str
    metrics_path: Path | None

    @property
    def name(self) -> str:
        return self.sample

    @property
    def base(self) -> Path:
        return self.sample_dir


def safe_slug(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "value"


def default_dataset_outputs(root: str | Path) -> dict[str, Path]:
    root = Path(root).expanduser().resolve()
    return {
        dataset: root / relative
        for dataset, relative in DEFAULT_DATASET_OUTPUTS.items()
    }


def _is_candidate_sample_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and not path.is_symlink()
        and not path.name.startswith(".")
        and path.name not in SKIP_SAMPLE_DIRS
    )


def _manifest_for_sample(sample_dir: Path) -> tuple[Path | None, str]:
    sample = sample_dir.name
    all_nonempty = sample_dir / f"{sample}_all_tile_labels_with_paths.csv"
    if all_nonempty.is_file():
        return all_nonempty, "all_nonempty"
    path_manifest = sample_dir / f"{sample}_tile_labels_with_paths.csv"
    if path_manifest.is_file():
        return path_manifest, "path_manifest"
    return None, "missing"


def _combined_manifest_for_dataset(output_dir: Path) -> Path | None:
    for filename in (
        "all_samples_tile_labels_with_paths.csv",
        "all_samples_tile_labels.csv",
    ):
        path = output_dir / filename
        if path.is_file():
            return path
    return None


def _sample_names_from_manifest(path: Path) -> list[str]:
    try:
        samples = pd.read_csv(path, usecols=["sample"])["sample"]
    except Exception:
        return []
    return sorted(
        sample
        for sample in samples.dropna().astype(str).unique()
        if sample and sample.lower() != "nan"
    )


def _project_local_path(path_value: object) -> Path | None:
    text = str(path_value or "").strip()
    if not text or text.lower() == "nan":
        return None
    path = Path(text)
    if path.is_file():
        return path
    parts = path.parts
    if "Xenium" in parts:
        suffix = parts[parts.index("Xenium") + 1 :]
        candidate = PROJECT_ROOT.joinpath(*suffix)
        if candidate.is_file():
            return candidate
    if not path.is_absolute():
        candidate = PROJECT_ROOT / path
        if candidate.is_file():
            return candidate
    return path if path.exists() else None


def _estimate_tile_area_from_input_manifest(
    sample: TileFilterSample,
    table: pd.DataFrame,
) -> float | None:
    input_manifest = sample.dataset_output_dir / "tile_input_manifest.csv"
    if not input_manifest.is_file():
        return None
    try:
        manifest = pd.read_csv(input_manifest)
    except Exception:
        return None
    if "sample" not in manifest.columns:
        return None
    rows = manifest.loc[manifest["sample"].astype(str).eq(sample.sample)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    try:
        pixel_size_um = float(row.get("pixel_size_um"))
    except (TypeError, ValueError):
        return None

    he_to_xenium: np.ndarray | None = None
    alignment_path = _project_local_path(
        row.get("alignment_csv", row.get("alignment", ""))
    )
    if alignment_path is not None and alignment_path.is_file():
        try:
            he_to_xenium = np.loadtxt(alignment_path, delimiter=",")
        except Exception:
            he_to_xenium = None

    if he_to_xenium is None:
        manual_path = _project_local_path(row.get("manual_alignment_json", ""))
        if manual_path is not None and manual_path.is_file():
            try:
                import json

                with manual_path.open(encoding="utf-8") as handle:
                    manual = json.load(handle)
                xenium_to_he = np.asarray(
                    manual["matrix_xenium_full_pixel_to_he_full_pixel"],
                    dtype=float,
                )
                he_to_xenium = np.linalg.inv(xenium_to_he)
            except Exception:
                he_to_xenium = None

    if he_to_xenium is None or he_to_xenium.shape != (3, 3):
        return None
    det = abs(float(np.linalg.det(he_to_xenium[:2, :2])))
    if not np.isfinite(det) or det <= 0:
        return None
    if table.empty:
        tile_size_px = 256
    else:
        widths = pd.to_numeric(
            table.get("tile_width_px", pd.Series(np.nan, index=table.index)),
            errors="coerce",
        )
        heights = pd.to_numeric(
            table.get("tile_height_px", pd.Series(np.nan, index=table.index)),
            errors="coerce",
        )
        sizes = pd.concat([widths, heights], ignore_index=True).dropna()
        tile_size_px = int(round(sizes.median())) if len(sizes) else 256
    if not tile_size_px or tile_size_px <= 0:
        return None
    return float(tile_size_px * tile_size_px * det * pixel_size_um**2)


def discover_tile_filter_samples(
    root: str | Path,
    *,
    dataset_outputs: dict[str, str | Path] | None = None,
) -> list[TileFilterSample]:
    """Discover sample tile manifests across every dataset output folder."""
    root = Path(root).expanduser().resolve()
    outputs = (
        {name: Path(path) for name, path in dataset_outputs.items()}
        if dataset_outputs is not None
        else default_dataset_outputs(root)
    )
    samples: list[TileFilterSample] = []
    for dataset, output_dir in outputs.items():
        output_dir = Path(output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = root / output_dir
        if not output_dir.is_dir():
            continue
        discovered_for_dataset: set[str] = set()
        for sample_dir in sorted(output_dir.iterdir(), key=lambda path: path.name):
            if not _is_candidate_sample_dir(sample_dir):
                continue
            manifest_path, scope = _manifest_for_sample(sample_dir)
            if manifest_path is None:
                continue
            sample = sample_dir.name
            metrics_path = sample_dir / f"{sample}_tile_filter_metrics.csv"
            samples.append(
                TileFilterSample(
                    dataset=dataset,
                    sample=sample,
                    dataset_output_dir=output_dir,
                    sample_dir=sample_dir,
                    manifest_path=manifest_path,
                    manifest_scope=scope,
                    metrics_path=metrics_path if metrics_path.is_file() else None,
                )
            )
            discovered_for_dataset.add(sample)
        combined_manifest = _combined_manifest_for_dataset(output_dir)
        if combined_manifest is None:
            continue
        combined_metrics = output_dir / "all_samples_tile_filter_metrics.csv"
        for sample in _sample_names_from_manifest(combined_manifest):
            if sample in discovered_for_dataset:
                continue
            samples.append(
                TileFilterSample(
                    dataset=dataset,
                    sample=sample,
                    dataset_output_dir=output_dir,
                    sample_dir=output_dir,
                    manifest_path=combined_manifest,
                    manifest_scope="combined_all_samples",
                    metrics_path=combined_metrics if combined_metrics.is_file() else None,
                )
            )
    return samples


def add_physical_tile_size_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Add micron-scale tile-size columns when tile area is available."""
    if table.empty:
        return table.copy()
    out = table.copy()
    if "tile_area_um2" not in out.columns:
        return out

    tile_area = pd.to_numeric(out["tile_area_um2"], errors="coerce")
    tile_area = tile_area.where(tile_area > 0)
    out["tile_area_um2"] = tile_area

    width_px = pd.to_numeric(
        out.get("tile_width_px", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    height_px = pd.to_numeric(
        out.get("tile_height_px", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    aspect = (width_px / height_px).where((width_px > 0) & (height_px > 0))

    out["tile_side_um"] = np.sqrt(tile_area)
    out["tile_width_um"] = np.sqrt(tile_area * aspect)
    out["tile_height_um"] = np.sqrt(tile_area / aspect)
    missing_dimensions = out["tile_width_um"].isna() | out["tile_height_um"].isna()
    out.loc[missing_dimensions, "tile_width_um"] = out.loc[missing_dimensions, "tile_side_um"]
    out.loc[missing_dimensions, "tile_height_um"] = out.loc[missing_dimensions, "tile_side_um"]

    out["effective_pixel_size_x_um"] = out["tile_width_um"] / width_px.where(width_px > 0)
    out["effective_pixel_size_y_um"] = out["tile_height_um"] / height_px.where(height_px > 0)
    out["effective_pixel_size_um"] = np.sqrt(
        out["effective_pixel_size_x_um"] * out["effective_pixel_size_y_um"]
    )
    return out


def load_sample_manifest(sample: TileFilterSample) -> pd.DataFrame:
    """Load one sample's nonempty tile manifest with standardized columns."""
    table = pd.read_csv(sample.manifest_path)
    if sample.manifest_scope == "combined_all_samples" and "sample" in table.columns:
        table = table.loc[table["sample"].astype(str).eq(sample.sample)].copy()
    if "dataset" in table.columns:
        table["dataset"] = table["dataset"].fillna(sample.dataset).astype(str)
    else:
        table.insert(0, "dataset", sample.dataset)
    if "sample" not in table.columns:
        table.insert(1, "sample", sample.sample)
    else:
        table["sample"] = table["sample"].fillna(sample.sample).astype(str)
    table["manifest_scope"] = sample.manifest_scope
    table["manifest_path"] = str(sample.manifest_path)
    table["sample_dir"] = str(sample.sample_dir)
    table["has_all_nonempty_manifest"] = sample.manifest_scope in {
        "all_nonempty",
        "combined_all_samples",
    }
    if "n_cells" not in table.columns:
        raise ValueError(f"{sample.manifest_path} has no n_cells column")
    table["n_cells"] = pd.to_numeric(table["n_cells"], errors="coerce").fillna(0).astype(int)
    if "cell_area_fraction" in table.columns:
        table["cell_area_fraction"] = pd.to_numeric(
            table["cell_area_fraction"],
            errors="coerce",
        )
    else:
        table["cell_area_fraction"] = np.nan
    if "cell_area_um2" in table.columns:
        table["cell_area_um2"] = pd.to_numeric(
            table["cell_area_um2"],
            errors="coerce",
        )
    if "nucleus_area_fraction" in table.columns:
        table["nucleus_area_fraction"] = pd.to_numeric(
            table["nucleus_area_fraction"],
            errors="coerce",
        )
    else:
        table["nucleus_area_fraction"] = np.nan
    if "nucleus_area_um2" in table.columns:
        table["nucleus_area_um2"] = pd.to_numeric(
            table["nucleus_area_um2"],
            errors="coerce",
        )
    if "he_tissue_fraction" in table.columns:
        table["he_tissue_fraction"] = pd.to_numeric(
            table["he_tissue_fraction"],
            errors="coerce",
        )
    else:
        table["he_tissue_fraction"] = np.nan
    if "tile_path" in table.columns:
        table["tile_path"] = table["tile_path"].astype(str)
    else:
        table["tile_path"] = ""
    if {"x0_px", "x1_px"}.issubset(table.columns):
        table["tile_width_px"] = pd.to_numeric(table["x1_px"], errors="coerce") - pd.to_numeric(
            table["x0_px"],
            errors="coerce",
        )
    if {"y0_px", "y1_px"}.issubset(table.columns):
        table["tile_height_px"] = pd.to_numeric(table["y1_px"], errors="coerce") - pd.to_numeric(
            table["y0_px"],
            errors="coerce",
        )
    tile_area = pd.to_numeric(
        table.get("tile_area_um2", pd.Series(np.nan, index=table.index)),
        errors="coerce",
    )
    if not tile_area.gt(0).any():
        estimated_tile_area = _estimate_tile_area_from_input_manifest(sample, table)
        if estimated_tile_area is not None:
            table["tile_area_um2"] = estimated_tile_area
    return add_physical_tile_size_columns(table)


def load_combined_tile_filter_table(samples: Iterable[TileFilterSample]) -> pd.DataFrame:
    tables = [load_sample_manifest(sample) for sample in samples]
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True, sort=False)


def inventory_table(samples: Iterable[TileFilterSample]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample in samples:
        table = load_sample_manifest(sample)
        area = pd.to_numeric(table["cell_area_fraction"], errors="coerce")
        nucleus_area = pd.to_numeric(table["nucleus_area_fraction"], errors="coerce")
        he_tissue = pd.to_numeric(table["he_tissue_fraction"], errors="coerce")
        tile_width = (
            pd.to_numeric(table.get("tile_width_px"), errors="coerce").dropna().median()
            if "tile_width_px" in table
            else np.nan
        )
        tile_height = (
            pd.to_numeric(table.get("tile_height_px"), errors="coerce").dropna().median()
            if "tile_height_px" in table
            else np.nan
        )
        tile_area = pd.to_numeric(
            table.get("tile_area_um2", pd.Series(np.nan, index=table.index)),
            errors="coerce",
        )
        tile_width_um = pd.to_numeric(
            table.get("tile_width_um", pd.Series(np.nan, index=table.index)),
            errors="coerce",
        )
        tile_height_um = pd.to_numeric(
            table.get("tile_height_um", pd.Series(np.nan, index=table.index)),
            errors="coerce",
        )
        effective_pixel_size = pd.to_numeric(
            table.get("effective_pixel_size_um", pd.Series(np.nan, index=table.index)),
            errors="coerce",
        )
        has_physical_tile_size = bool(tile_area.notna().any())
        has_all_nonempty_manifest = sample.manifest_scope in {
            "all_nonempty",
            "combined_all_samples",
        }
        rows.append(
            {
                "dataset": sample.dataset,
                "sample": sample.sample,
                "manifest_scope": sample.manifest_scope,
                "has_all_nonempty_manifest": has_all_nonempty_manifest,
                "has_cell_area_fraction": bool(area.notna().any()),
                "has_nucleus_area_fraction": bool(nucleus_area.notna().any()),
                "has_he_tissue_fraction": bool(he_tissue.notna().any()),
                "has_physical_tile_size_um": has_physical_tile_size,
                "n_nonempty_tiles": len(table),
                "median_cells_per_tile": table["n_cells"].median(),
                "max_cells_per_tile": table["n_cells"].max(),
                "median_cell_area_fraction": area.dropna().median() if area.notna().any() else np.nan,
                "median_nucleus_area_fraction": (
                    nucleus_area.dropna().median()
                    if nucleus_area.notna().any()
                    else np.nan
                ),
                "median_he_tissue_fraction": (
                    he_tissue.dropna().median()
                    if he_tissue.notna().any()
                    else np.nan
                ),
                "tile_width_px": tile_width,
                "tile_height_px": tile_height,
                "median_tile_area_um2": (
                    tile_area.dropna().median()
                    if tile_area.notna().any()
                    else np.nan
                ),
                "median_tile_width_um": (
                    tile_width_um.dropna().median()
                    if tile_width_um.notna().any()
                    else np.nan
                ),
                "median_tile_height_um": (
                    tile_height_um.dropna().median()
                    if tile_height_um.notna().any()
                    else np.nan
                ),
                "median_effective_pixel_size_um": (
                    effective_pixel_size.dropna().median()
                    if effective_pixel_size.notna().any()
                    else np.nan
                ),
                "manifest_path": str(sample.manifest_path),
            }
        )
    return pd.DataFrame(rows)


def otsu_gray_threshold(image: np.ndarray) -> int:
    """Return an Otsu threshold for an RGB or grayscale uint8 image."""
    array = np.asarray(image)
    if array.ndim == 3:
        rgb = array[..., :3].astype(float)
        array = np.rint(
            0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        )
    return he_tissue.otsu_gray_threshold(array)


def he_color_tissue_mask(
    image: np.ndarray,
    *,
    dark_neutral_value_max: int = HE_TISSUE_DARK_NEUTRAL_VALUE_MAX,
    dark_neutral_chroma_max: int = HE_TISSUE_DARK_NEUTRAL_CHROMA_MAX,
    bright_neutral_value_min: int = HE_TISSUE_BRIGHT_NEUTRAL_VALUE_MIN,
    bright_neutral_chroma_max: int = HE_TISSUE_BRIGHT_NEUTRAL_CHROMA_MAX,
) -> tuple[np.ndarray, int, int]:
    """Return an H&E tissue mask with adaptive darkness and chroma thresholds."""
    rgb = standard_xenium.ensure_rgb(np.asarray(image))
    return he_tissue.color_tissue_mask_rgb(
        rgb,
        dark_neutral_value_max=dark_neutral_value_max,
        dark_neutral_chroma_max=dark_neutral_chroma_max,
        bright_neutral_value_min=bright_neutral_value_min,
        bright_neutral_chroma_max=bright_neutral_chroma_max,
    )


def _select_he_mask_level(
    levels: Iterable[object],
    *,
    full_height: int,
    full_width: int,
    tile_width_px: float,
    tile_height_px: float,
    min_mask_pixels_per_tile: int,
) -> tuple[int, object, float, float, float]:
    """Select the coarsest pyramid level that preserves within-tile detail."""
    candidates: list[tuple[int, object, int, float, float, float]] = []
    for index, level in enumerate(levels):
        level_height, level_width = standard_xenium.get_hw(tuple(level.shape))
        scale_x = level_width / float(full_width)
        scale_y = level_height / float(full_height)
        pixels_per_tile = min(
            float(tile_width_px) * scale_x,
            float(tile_height_px) * scale_y,
        )
        candidates.append(
            (
                index,
                level,
                level_height * level_width,
                scale_x,
                scale_y,
                pixels_per_tile,
            )
        )
    if not candidates:
        raise ValueError("Source H&E image has no readable pyramid levels")
    eligible = [
        candidate
        for candidate in candidates
        if candidate[5] >= int(min_mask_pixels_per_tile)
    ]
    chosen = (
        min(eligible, key=lambda candidate: candidate[2])
        if eligible
        else max(candidates, key=lambda candidate: candidate[2])
    )
    index, level, _area, scale_x, scale_y, pixels_per_tile = chosen
    return index, level, scale_x, scale_y, pixels_per_tile


def load_resolution_aware_he_tissue_mask(
    paths: standard_xenium.SamplePaths,
    *,
    tile_width_px: float,
    tile_height_px: float,
    min_mask_pixels_per_tile: int = HE_TISSUE_MIN_MASK_PIXELS_PER_TILE,
) -> tuple[np.ndarray, dict[str, object]]:
    """Load a suitable H&E pyramid level and derive a color-aware tissue mask."""
    if paths.he_image is None:
        raise FileNotFoundError("H&E image was not resolved")
    with tifffile.TiffFile(paths.he_image) as tif:
        series = tif.series[0]
        levels = list(series.levels) if getattr(series, "levels", None) else [series]
        full_height, full_width = standard_xenium.get_hw(tuple(levels[0].shape))
        level_index, level, scale_x, scale_y, pixels_per_tile = _select_he_mask_level(
            levels,
            full_height=full_height,
            full_width=full_width,
            tile_width_px=tile_width_px,
            tile_height_px=tile_height_px,
            min_mask_pixels_per_tile=min_mask_pixels_per_tile,
        )
        image = standard_xenium.ensure_rgb(level.asarray())
    mask, darkness_threshold, chroma_threshold = he_color_tissue_mask(image)
    metadata = {
        "method_version": HE_TISSUE_MASK_METHOD_VERSION,
        "level_index": int(level_index),
        "level_height": int(mask.shape[0]),
        "level_width": int(mask.shape[1]),
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "pixels_per_tile": float(pixels_per_tile),
        "min_mask_pixels_per_tile": int(min_mask_pixels_per_tile),
        # Preserve the original key as a compatibility alias for darkness.
        "otsu_threshold": int(darkness_threshold),
        "darkness_otsu_threshold": int(darkness_threshold),
        "chroma_otsu_threshold": int(chroma_threshold),
        "dark_neutral_value_max": HE_TISSUE_DARK_NEUTRAL_VALUE_MAX,
        "dark_neutral_chroma_max": HE_TISSUE_DARK_NEUTRAL_CHROMA_MAX,
        "bright_neutral_value_min": HE_TISSUE_BRIGHT_NEUTRAL_VALUE_MIN,
        "bright_neutral_chroma_max": HE_TISSUE_BRIGHT_NEUTRAL_CHROMA_MAX,
    }
    return mask, metadata


def he_tissue_fraction_manifest_path(
    sample: TileFilterSample,
    output_root: str | Path,
) -> Path:
    root = Path(output_root).expanduser().resolve()
    return (
        root
        / safe_slug(sample.dataset)
        / safe_slug(sample.sample)
        / f"{sample.sample}_all_tile_labels_with_paths.csv"
    )


def _tile_fractions_from_thumbnail_mask(
    manifest: pd.DataFrame,
    mask: np.ndarray,
    metadata: standard_xenium.ImageMetadata,
) -> np.ndarray:
    required = {"x0_px", "y0_px", "x1_px", "y1_px"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Tile manifest is missing image bounds: {sorted(missing)}")
    height, width = mask.shape
    x0 = np.floor(
        pd.to_numeric(manifest["x0_px"], errors="coerce").to_numpy(float)
        * metadata.thumbnail_scale_x
    )
    y0 = np.floor(
        pd.to_numeric(manifest["y0_px"], errors="coerce").to_numpy(float)
        * metadata.thumbnail_scale_y
    )
    x1 = np.ceil(
        pd.to_numeric(manifest["x1_px"], errors="coerce").to_numpy(float)
        * metadata.thumbnail_scale_x
    )
    y1 = np.ceil(
        pd.to_numeric(manifest["y1_px"], errors="coerce").to_numpy(float)
        * metadata.thumbnail_scale_y
    )
    finite = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(x1) & np.isfinite(y1)
    x0 = np.clip(np.nan_to_num(x0, nan=0).astype(int), 0, width)
    y0 = np.clip(np.nan_to_num(y0, nan=0).astype(int), 0, height)
    x1 = np.clip(np.nan_to_num(x1, nan=0).astype(int), 0, width)
    y1 = np.clip(np.nan_to_num(y1, nan=0).astype(int), 0, height)
    finite &= (x1 > x0) & (y1 > y0)

    integral = np.pad(mask.astype(np.uint64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    tissue_pixels = (
        integral[y1, x1]
        - integral[y0, x1]
        - integral[y1, x0]
        + integral[y0, x0]
    ).astype(float)
    tile_pixels = ((x1 - x0) * (y1 - y0)).astype(float)
    fractions = np.full(len(manifest), np.nan, dtype=float)
    fractions[finite] = tissue_pixels[finite] / tile_pixels[finite]
    return fractions


def calculate_he_tissue_fraction_for_sample(
    sample: TileFilterSample,
    *,
    output_root: str | Path,
    overwrite: bool = False,
    min_mask_pixels_per_tile: int = HE_TISSUE_MIN_MASK_PIXELS_PER_TILE,
) -> tuple[dict[str, object], TileFilterSample]:
    """Write a manifest copy containing color-derived H&E tissue fractions."""
    output_manifest = he_tissue_fraction_manifest_path(sample, output_root)
    source_manifest = str(sample.manifest_path.expanduser().resolve())
    summary: dict[str, object] = {
        "dataset": sample.dataset,
        "sample": sample.sample,
        "status": "ok",
        "input_manifest_path": source_manifest,
        "output_manifest_path": str(output_manifest),
        "n_tiles": 0,
        "n_tiles_with_he_tissue_fraction": 0,
        "he_tissue_otsu_threshold": np.nan,
        "he_tissue_darkness_otsu_threshold": np.nan,
        "he_tissue_chroma_otsu_threshold": np.nan,
        "he_tissue_mask_method": HE_TISSUE_MASK_METHOD_VERSION,
        "he_tissue_mask_level": pd.NA,
        "he_tissue_mask_pixels_per_tile": np.nan,
        "median_he_tissue_fraction": np.nan,
        "error": "",
    }
    try:
        source = load_sample_manifest(sample)
        summary["n_tiles"] = len(source)
        if output_manifest.is_file() and not overwrite:
            cached = pd.read_csv(output_manifest)
            cached_source = cached.get(
                "he_tissue_source_manifest_path",
                pd.Series("", index=cached.index),
            ).astype(str)
            identity_columns = [
                column
                for column in (
                    "tile_row",
                    "tile_col",
                    "x0_px",
                    "y0_px",
                    "x1_px",
                    "y1_px",
                )
                if column in source.columns and column in cached.columns
            ]
            same_tile_identity = bool(identity_columns)
            for column in identity_columns:
                source_values = pd.to_numeric(
                    source[column], errors="coerce"
                ).to_numpy(float)
                cached_values = pd.to_numeric(
                    cached[column], errors="coerce"
                ).to_numpy(float)
                if not np.allclose(source_values, cached_values, equal_nan=True):
                    same_tile_identity = False
                    break
            valid_cache = (
                len(cached) == len(source)
                and "he_tissue_fraction" in cached.columns
                and bool(cached_source.eq(source_manifest).all())
                and same_tile_identity
                and "he_tissue_mask_method" in cached.columns
                and bool(
                    cached["he_tissue_mask_method"]
                    .astype(str)
                    .eq(HE_TISSUE_MASK_METHOD_VERSION)
                    .all()
                )
                and "he_tissue_min_mask_pixels_per_tile" in cached.columns
                and bool(
                    pd.to_numeric(
                        cached["he_tissue_min_mask_pixels_per_tile"],
                        errors="coerce",
                    )
                    .eq(int(min_mask_pixels_per_tile))
                    .all()
                )
            )
            if valid_cache:
                tissue_columns = [
                    column
                    for column in cached.columns
                    if column == "he_tissue_fraction" or column.startswith("he_tissue_")
                ]
                refreshed = source.copy()
                for column in tissue_columns:
                    refreshed[column] = cached[column].to_numpy()
                refreshed.to_csv(output_manifest, index=False)
                values = pd.to_numeric(
                    refreshed["he_tissue_fraction"], errors="coerce"
                )
                thresholds = pd.to_numeric(
                    refreshed.get(
                        "he_tissue_otsu_threshold",
                        pd.Series(np.nan, index=refreshed.index),
                    ),
                    errors="coerce",
                ).dropna()
                summary["status"] = "cached"
                summary["n_tiles_with_he_tissue_fraction"] = int(values.notna().sum())
                summary["he_tissue_otsu_threshold"] = (
                    float(thresholds.iloc[0]) if len(thresholds) else np.nan
                )
                darkness_thresholds = pd.to_numeric(
                    refreshed.get(
                        "he_tissue_darkness_otsu_threshold",
                        refreshed.get(
                            "he_tissue_otsu_threshold",
                            pd.Series(np.nan, index=refreshed.index),
                        ),
                    ),
                    errors="coerce",
                ).dropna()
                chroma_thresholds = pd.to_numeric(
                    refreshed.get(
                        "he_tissue_chroma_otsu_threshold",
                        pd.Series(np.nan, index=refreshed.index),
                    ),
                    errors="coerce",
                ).dropna()
                summary["he_tissue_darkness_otsu_threshold"] = (
                    float(darkness_thresholds.iloc[0])
                    if len(darkness_thresholds)
                    else np.nan
                )
                summary["he_tissue_chroma_otsu_threshold"] = (
                    float(chroma_thresholds.iloc[0])
                    if len(chroma_thresholds)
                    else np.nan
                )
                levels = pd.to_numeric(
                    refreshed.get(
                        "he_tissue_mask_level",
                        pd.Series(np.nan, index=refreshed.index),
                    ),
                    errors="coerce",
                ).dropna()
                pixels_per_tile = pd.to_numeric(
                    refreshed.get(
                        "he_tissue_mask_pixels_per_tile",
                        pd.Series(np.nan, index=refreshed.index),
                    ),
                    errors="coerce",
                ).dropna()
                summary["he_tissue_mask_level"] = (
                    int(levels.iloc[0]) if len(levels) else pd.NA
                )
                summary["he_tissue_mask_pixels_per_tile"] = (
                    float(pixels_per_tile.iloc[0])
                    if len(pixels_per_tile)
                    else np.nan
                )
                summary["median_he_tissue_fraction"] = (
                    float(values.median()) if values.notna().any() else np.nan
                )
                return summary, replace(sample, manifest_path=output_manifest, metrics_path=None)

        paths = _sample_paths_for_area_recalculation(sample)
        metadata = standard_xenium.load_image_metadata(paths)
        tile_width_px = pd.to_numeric(
            source["x1_px"], errors="coerce"
        ) - pd.to_numeric(source["x0_px"], errors="coerce")
        tile_height_px = pd.to_numeric(
            source["y1_px"], errors="coerce"
        ) - pd.to_numeric(source["y0_px"], errors="coerce")
        tissue_mask, mask_metadata = load_resolution_aware_he_tissue_mask(
            paths,
            tile_width_px=float(tile_width_px.dropna().median()),
            tile_height_px=float(tile_height_px.dropna().median()),
            min_mask_pixels_per_tile=min_mask_pixels_per_tile,
        )
        mask_image_metadata = replace(
            metadata,
            thumbnail=tissue_mask,
            thumbnail_height=int(tissue_mask.shape[0]),
            thumbnail_width=int(tissue_mask.shape[1]),
            thumbnail_scale_x=float(mask_metadata["scale_x"]),
            thumbnail_scale_y=float(mask_metadata["scale_y"]),
        )
        fractions = _tile_fractions_from_thumbnail_mask(
            source,
            tissue_mask,
            mask_image_metadata,
        )
        source["he_tissue_fraction"] = fractions
        source["he_tissue_otsu_threshold"] = mask_metadata["otsu_threshold"]
        source["he_tissue_darkness_otsu_threshold"] = mask_metadata[
            "darkness_otsu_threshold"
        ]
        source["he_tissue_chroma_otsu_threshold"] = mask_metadata[
            "chroma_otsu_threshold"
        ]
        source["he_tissue_mask_method"] = mask_metadata["method_version"]
        source["he_tissue_mask_level"] = mask_metadata["level_index"]
        source["he_tissue_mask_height"] = mask_metadata["level_height"]
        source["he_tissue_mask_width"] = mask_metadata["level_width"]
        source["he_tissue_mask_scale_x"] = mask_metadata["scale_x"]
        source["he_tissue_mask_scale_y"] = mask_metadata["scale_y"]
        source["he_tissue_mask_pixels_per_tile"] = mask_metadata["pixels_per_tile"]
        source["he_tissue_min_mask_pixels_per_tile"] = mask_metadata[
            "min_mask_pixels_per_tile"
        ]
        source["he_tissue_dark_neutral_value_max"] = mask_metadata[
            "dark_neutral_value_max"
        ]
        source["he_tissue_dark_neutral_chroma_max"] = mask_metadata[
            "dark_neutral_chroma_max"
        ]
        source["he_tissue_bright_neutral_value_min"] = mask_metadata[
            "bright_neutral_value_min"
        ]
        source["he_tissue_bright_neutral_chroma_max"] = mask_metadata[
            "bright_neutral_chroma_max"
        ]
        source["he_tissue_source_image"] = str(paths.he_image or "")
        source["he_tissue_source_manifest_path"] = source_manifest
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
        source.to_csv(output_manifest, index=False)

        values = pd.to_numeric(source["he_tissue_fraction"], errors="coerce")
        summary["n_tiles_with_he_tissue_fraction"] = int(values.notna().sum())
        summary["he_tissue_otsu_threshold"] = mask_metadata["otsu_threshold"]
        summary["he_tissue_darkness_otsu_threshold"] = mask_metadata[
            "darkness_otsu_threshold"
        ]
        summary["he_tissue_chroma_otsu_threshold"] = mask_metadata[
            "chroma_otsu_threshold"
        ]
        summary["he_tissue_mask_level"] = mask_metadata["level_index"]
        summary["he_tissue_mask_pixels_per_tile"] = mask_metadata["pixels_per_tile"]
        summary["median_he_tissue_fraction"] = (
            float(values.median()) if values.notna().any() else np.nan
        )
        return summary, replace(sample, manifest_path=output_manifest, metrics_path=None)
    except Exception as error:
        summary["status"] = "error"
        summary["error"] = str(error)
        summary["output_manifest_path"] = ""
        return summary, sample


def calculate_he_tissue_fraction_for_samples(
    samples: Iterable[TileFilterSample],
    *,
    output_root: str | Path,
    overwrite: bool = False,
    min_mask_pixels_per_tile: int = HE_TISSUE_MIN_MASK_PIXELS_PER_TILE,
) -> tuple[pd.DataFrame, list[TileFilterSample]]:
    summaries: list[dict[str, object]] = []
    updated: list[TileFilterSample] = []
    for sample in samples:
        print(f"[{sample.dataset}/{sample.sample}] calculating H&E tissue occupancy")
        summary, active_sample = calculate_he_tissue_fraction_for_sample(
            sample,
            output_root=output_root,
            overwrite=overwrite,
            min_mask_pixels_per_tile=min_mask_pixels_per_tile,
        )
        summaries.append(summary)
        updated.append(active_sample)
    return pd.DataFrame(summaries), updated


def tile_source_path(sample: TileFilterSample, tile_path: object) -> Path:
    path = Path(str(tile_path))
    return path if path.is_absolute() else sample.sample_dir / path


def add_cross_dataset_tile_ids(
    sample: TileFilterSample,
    tiles: pd.DataFrame,
    config: standard_xenium.PipelineConfig,
) -> pd.DataFrame:
    table = tiles.copy()
    tile_ids = [
        (
            f"{safe_slug(sample.dataset)}__{sample.sample}_"
            f"row-{int(row.tile_row):05d}_col-{int(row.tile_col):05d}"
        )
        for row in table.itertuples(index=False)
    ]
    if "tile_id" in table.columns:
        table["tile_id"] = tile_ids
    else:
        table.insert(0, "tile_id", tile_ids)
    if "dataset" in table.columns:
        table["dataset"] = sample.dataset
    else:
        table.insert(1, "dataset", sample.dataset)
    if "min_cells_per_tile" in table.columns:
        table["min_cells_per_tile"] = int(config.min_cells_per_tile)
    else:
        table.insert(2, "min_cells_per_tile", int(config.min_cells_per_tile))
    if "min_cell_area_fraction" in table.columns:
        table["min_cell_area_fraction"] = float(config.min_cell_area_fraction)
    else:
        table.insert(3, "min_cell_area_fraction", float(config.min_cell_area_fraction))
    if "tile_path" in table.columns:
        source_paths = [str(tile_source_path(sample, value)) for value in table["tile_path"]]
        if "source_tile_path" in table.columns:
            table["source_tile_path"] = source_paths
        else:
            table.insert(table.columns.get_loc("tile_path") + 1, "source_tile_path", source_paths)
    return table


def materialize_selected_tiles(
    sample: TileFilterSample,
    selected_tiles: pd.DataFrame,
    config: standard_xenium.PipelineConfig,
    destination: Path,
    *,
    mode: str = "copy",
) -> pd.DataFrame:
    if mode not in {"copy", "symlink", "move"}:
        raise ValueError("mode must be one of: copy, symlink, move")
    destination.mkdir(parents=True, exist_ok=True)
    columns = [
        "tile_id",
        "dataset",
        "sample",
        "mode",
        "status",
        "source_tile_path",
        "materialized_tile_path",
    ]
    records: list[dict[str, object]] = []
    selected_with_ids = add_cross_dataset_tile_ids(sample, selected_tiles, config)
    for tile in selected_with_ids.itertuples(index=False):
        source = tile_source_path(sample, tile.tile_path)
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
                "dataset": sample.dataset,
                "sample": sample.sample,
                "mode": mode,
                "status": status,
                "source_tile_path": str(source),
                "materialized_tile_path": str(target),
            }
        )
    return pd.DataFrame(records, columns=columns)


def _example_context(
    sample: TileFilterSample,
    config: standard_xenium.PipelineConfig,
    *,
    include_overlays: bool = True,
) -> dict[str, object]:
    try:
        paths = _sample_paths_for_area_recalculation(sample)
    except Exception as error:
        print(f"[{sample.dataset}/{sample.sample}] example context unavailable: {error}")
        return {}
    if not include_overlays:
        return {"paths": paths}
    try:
        overlay = standard_xenium.load_tile_filter_overlay_context(paths, config)
    except Exception as error:
        print(f"[{sample.dataset}/{sample.sample}] cell overlay unavailable: {error}")
        return {"paths": paths}
    overlay["paths"] = paths
    return overlay


def sample_output_stem(
    sample: TileFilterSample,
    config: standard_xenium.PipelineConfig,
    min_nucleus_area_fraction: float = 0.0,
    min_he_tissue_fraction: float = 0.0,
) -> str:
    cutoff_slug = standard_xenium.tile_filter_cutoff_slug(
        config.min_cells_per_tile,
        config.min_cell_area_fraction,
        min_nucleus_area_fraction,
        min_he_tissue_fraction,
    )
    return f"{safe_slug(sample.dataset)}__{safe_slug(sample.sample)}__{cutoff_slug}"


def infer_tile_size_px(sample: TileFilterSample, default: int = 256) -> int:
    table = load_sample_manifest(sample)
    values: list[float] = []
    if {"x0_px", "x1_px"}.issubset(table.columns):
        widths = pd.to_numeric(table["x1_px"], errors="coerce") - pd.to_numeric(
            table["x0_px"],
            errors="coerce",
        )
        values.extend(widths.dropna().tolist())
    if {"y0_px", "y1_px"}.issubset(table.columns):
        heights = pd.to_numeric(table["y1_px"], errors="coerce") - pd.to_numeric(
            table["y0_px"],
            errors="coerce",
        )
        values.extend(heights.dropna().tolist())
    positive = [value for value in values if value > 0]
    if not positive:
        return int(default)
    return int(round(float(pd.Series(positive).median())))


def _sample_paths_for_area_recalculation(sample: TileFilterSample) -> standard_xenium.SamplePaths:
    input_manifest = sample.dataset_output_dir / "tile_input_manifest.csv"
    if input_manifest.is_file():
        try:
            manifest_samples = standard_xenium.discover_samples_from_manifest(
                input_manifest,
                requested=[sample.sample],
            )
            if manifest_samples:
                return manifest_samples[0]
        except Exception as error:
            print(
                f"[{sample.dataset}/{sample.sample}] could not use tile_input_manifest.csv "
                f"for area recalculation: {error}"
            )
    return standard_xenium.build_sample_paths(sample.sample_dir)


def _assign_tiles_to_cells(
    cells: pd.DataFrame,
    metadata: standard_xenium.ImageMetadata,
    tile_size_px: int,
) -> pd.DataFrame:
    if not {"he_x_px", "he_y_px"}.issubset(cells.columns):
        raise ValueError("Cells must have he_x_px and he_y_px before tile assignment")
    table = cells.copy()
    if "inside_he_image" in table.columns:
        table = table.loc[table["inside_he_image"]].copy()
    n_columns = metadata.full_width // int(tile_size_px)
    n_rows = metadata.full_height // int(tile_size_px)
    table["tile_col"] = (pd.to_numeric(table["he_x_px"], errors="coerce") // int(tile_size_px)).astype("Int64")
    table["tile_row"] = (pd.to_numeric(table["he_y_px"], errors="coerce") // int(tile_size_px)).astype("Int64")
    table = table.loc[
        table["tile_col"].notna()
        & table["tile_row"].notna()
        & (table["tile_col"] >= 0)
        & (table["tile_col"] < n_columns)
        & (table["tile_row"] >= 0)
        & (table["tile_row"] < n_rows)
    ].copy()
    table["tile_col"] = table["tile_col"].astype(int)
    table["tile_row"] = table["tile_row"].astype(int)
    return table


def _load_cells_for_area_recalculation(
    paths: standard_xenium.SamplePaths,
    config: standard_xenium.PipelineConfig,
    metadata: standard_xenium.ImageMetadata,
) -> tuple[pd.DataFrame, float, np.ndarray | None]:
    cells = standard_xenium.tile_cells_from_existing_output(paths, config, metadata)
    pixel_size = standard_xenium.experiment_pixel_size(paths)
    _, _, xenium_to_he = standard_xenium.align_xy_to_he(
        np.array([[0.0, 0.0]], dtype=float),
        paths,
        metadata,
        pixel_size,
    )
    if cells is None or cells.empty or "cell_index" not in cells.columns:
        raw_cells, _ = standard_xenium.load_cells(paths)
        aligned_cells, pixel_size, _mode, xenium_to_he = standard_xenium.align_cells(
            paths,
            raw_cells,
            metadata,
        )
        cells = _assign_tiles_to_cells(aligned_cells, metadata, config.tile_size_px)
    else:
        cells = _assign_tiles_to_cells(cells, metadata, config.tile_size_px)
    cells["cell_index"] = pd.to_numeric(cells["cell_index"], errors="raise").astype(int)
    return cells, pixel_size, xenium_to_he


def _boundary_occupancy_by_tile(
    paths: standard_xenium.SamplePaths,
    tile_cells: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    metadata: standard_xenium.ImageMetadata,
    pixel_size: float,
    xenium_to_he: np.ndarray | None,
    tile_size_px: int,
    boundary_kind: str,
    block_tiles: int = 16,
) -> tuple[pd.DataFrame, int]:
    """Rasterize boundary unions in H&E pixels and return tile occupancy."""
    boundaries = standard_xenium.load_boundaries(
        paths,
        tile_cells,
        boundary_kind=boundary_kind,
    )
    if boundaries is None:
        raise ValueError(f"No {boundary_kind} boundary polygons were available")
    required = {"tile_row", "tile_col"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Tile manifest is missing occupancy keys: {sorted(missing)}")

    tile_size_px = int(tile_size_px)
    block_tiles = int(block_tiles)
    if tile_size_px <= 0 or block_tiles <= 0:
        raise ValueError("tile_size_px and block_tiles must be positive")
    n_columns = metadata.full_width // tile_size_px
    n_rows = metadata.full_height // tile_size_px

    active_tiles: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    manifest_tiles = manifest[["tile_row", "tile_col"]].copy()
    manifest_tiles["tile_row"] = pd.to_numeric(
        manifest_tiles["tile_row"], errors="raise"
    ).astype(int)
    manifest_tiles["tile_col"] = pd.to_numeric(
        manifest_tiles["tile_col"], errors="raise"
    ).astype(int)
    if manifest_tiles.duplicated().any():
        raise ValueError("Tile manifest contains duplicate tile_row/tile_col keys")
    outside_complete_grid = (
        (manifest_tiles["tile_row"] < 0)
        | (manifest_tiles["tile_row"] >= n_rows)
        | (manifest_tiles["tile_col"] < 0)
        | (manifest_tiles["tile_col"] >= n_columns)
    )
    if outside_complete_grid.any():
        raise ValueError(
            f"Tile manifest contains {int(outside_complete_grid.sum()):,} tile(s) "
            "outside the complete H&E tile grid"
        )
    for row in manifest_tiles.itertuples(index=False):
        tile_row = int(row.tile_row)
        tile_col = int(row.tile_col)
        active_tiles[(tile_row // block_tiles, tile_col // block_tiles)].append(
            (tile_row, tile_col)
        )
    active_blocks = set(active_tiles)
    block_candidates: dict[tuple[int, int], list[int]] = defaultdict(list)
    cell_indices = (
        pd.to_numeric(tile_cells["cell_index"], errors="raise")
        .astype(int)
        .drop_duplicates()
        .to_numpy()
    )
    n_cells_with_boundary = 0
    for cell_index in cell_indices:
        polygon = standard_xenium.boundary_in_he_pixels(
            int(cell_index),
            boundaries,
            paths,
            metadata,
            pixel_size,
            xenium_to_he,
        )
        if polygon is None or len(polygon) < 3 or not np.isfinite(polygon).all():
            continue
        n_cells_with_boundary += 1
        min_x = max(0.0, float(np.min(polygon[:, 0])))
        min_y = max(0.0, float(np.min(polygon[:, 1])))
        max_x = min(float(n_columns * tile_size_px), float(np.max(polygon[:, 0])))
        max_y = min(float(n_rows * tile_size_px), float(np.max(polygon[:, 1])))
        if max_x <= min_x or max_y <= min_y:
            continue
        first_block_col = int(np.floor(min_x / (block_tiles * tile_size_px)))
        first_block_row = int(np.floor(min_y / (block_tiles * tile_size_px)))
        last_block_col = int(np.ceil(max_x / (block_tiles * tile_size_px)) - 1)
        last_block_row = int(np.ceil(max_y / (block_tiles * tile_size_px)) - 1)
        for block_row in range(first_block_row, last_block_row + 1):
            for block_col in range(first_block_col, last_block_col + 1):
                block_key = (block_row, block_col)
                if block_key in active_blocks:
                    block_candidates[block_key].append(int(cell_index))

    print(
        f"[{paths.name}] rasterizing {boundary_kind} occupancy for "
        f"{len(manifest):,} tile(s) from {n_cells_with_boundary:,} boundary polygon(s)"
    )

    fractions: dict[tuple[int, int], float] = {
        tile_key: 0.0
        for tiles in active_tiles.values()
        for tile_key in tiles
    }
    for (block_row, block_col), tiles in active_tiles.items():
        first_tile_row = block_row * block_tiles
        first_tile_col = block_col * block_tiles
        block_tile_rows = min(block_tiles, n_rows - first_tile_row)
        block_tile_cols = min(block_tiles, n_columns - first_tile_col)
        block_height = block_tile_rows * tile_size_px
        block_width = block_tile_cols * tile_size_px
        origin_x = first_tile_col * tile_size_px
        origin_y = first_tile_row * tile_size_px
        mask_image = Image.new("1", (block_width, block_height), 0)
        draw = ImageDraw.Draw(mask_image)
        for cell_index in block_candidates.get((block_row, block_col), []):
            polygon = standard_xenium.boundary_in_he_pixels(
                cell_index,
                boundaries,
                paths,
                metadata,
                pixel_size,
                xenium_to_he,
            )
            if polygon is None or len(polygon) < 3 or not np.isfinite(polygon).all():
                continue
            points = [
                (float(x) - origin_x, float(y) - origin_y)
                for x, y in polygon
            ]
            draw.polygon(points, fill=1)
        block_mask = np.asarray(mask_image, dtype=bool)
        for tile_row, tile_col in tiles:
            local_row = tile_row - first_tile_row
            local_col = tile_col - first_tile_col
            y0 = local_row * tile_size_px
            x0 = local_col * tile_size_px
            tile_mask = block_mask[
                y0 : y0 + tile_size_px,
                x0 : x0 + tile_size_px,
            ]
            fractions[(tile_row, tile_col)] = float(tile_mask.mean())

    fraction_column = f"recalculated_{boundary_kind}_area_fraction"
    occupancy = pd.DataFrame(
        [
            {
                "tile_row": tile_row,
                "tile_col": tile_col,
                fraction_column: fraction,
            }
            for (tile_row, tile_col), fraction in fractions.items()
        ]
    )
    return occupancy, n_cells_with_boundary


def recalculate_boundary_area_metrics_for_sample(
    sample: TileFilterSample,
    *,
    output_root: str | Path,
    area_kinds: Iterable[str] = ("cell",),
    tile_size_px: int | None = None,
) -> tuple[dict[str, object], TileFilterSample]:
    """Write a manifest copy with boundary-derived cell and/or nucleus areas."""
    if isinstance(area_kinds, str):
        area_kinds = ("cell", "nucleus") if area_kinds == "both" else (area_kinds,)
    area_kinds = tuple(dict.fromkeys(str(kind) for kind in area_kinds))
    unknown = sorted(set(area_kinds) - {"cell", "nucleus"})
    if unknown:
        raise ValueError(f"area_kinds must contain only 'cell' and/or 'nucleus': {unknown}")

    output_root = Path(output_root).expanduser().resolve()
    manifest = load_sample_manifest(sample)
    original_manifest = pd.read_csv(sample.manifest_path)
    if "cell_area_um2" in original_manifest.columns and "original_cell_area_um2" not in manifest.columns:
        manifest["original_cell_area_um2"] = pd.to_numeric(
            original_manifest["cell_area_um2"],
            errors="coerce",
        )
    if "cell_area_fraction" in original_manifest.columns and "original_cell_area_fraction" not in manifest.columns:
        manifest["original_cell_area_fraction"] = pd.to_numeric(
            original_manifest["cell_area_fraction"],
            errors="coerce",
        )
    config = standard_xenium.PipelineConfig(
        tile_size_px=int(tile_size_px or infer_tile_size_px(sample)),
        min_cells_per_tile=1,
        min_cell_area_fraction=0.0,
    )
    output_dir = output_root / safe_slug(sample.dataset) / safe_slug(sample.sample)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = output_dir / f"{sample.sample}_all_tile_labels_with_paths.csv"

    summary: dict[str, object] = {
        "dataset": sample.dataset,
        "sample": sample.sample,
        "status": "ok",
        "area_kinds": ",".join(area_kinds),
        "input_manifest_path": str(sample.manifest_path),
        "output_manifest_path": str(output_manifest),
        "n_tiles": len(manifest),
        "n_cells_with_cell_boundary": pd.NA,
        "n_cells_with_nucleus_boundary": pd.NA,
        "error": "",
    }

    try:
        paths = _sample_paths_for_area_recalculation(sample)
        metadata = standard_xenium.load_image_metadata(paths)
        tile_cells, pixel_size, xenium_to_he = _load_cells_for_area_recalculation(
            paths,
            config,
            metadata,
        )
        tile_area_um2 = standard_xenium.he_tile_area_um2(
            metadata,
            config.tile_size_px,
            pixel_size,
            xenium_to_he,
        )
        summary["tile_area_um2"] = tile_area_um2

        any_success = False
        failed_kinds: list[str] = []
        for kind in area_kinds:
            try:
                occupancy_by_tile, n_cells_with_boundary = _boundary_occupancy_by_tile(
                    paths,
                    tile_cells,
                    manifest,
                    metadata=metadata,
                    pixel_size=pixel_size,
                    xenium_to_he=xenium_to_he,
                    tile_size_px=config.tile_size_px,
                    boundary_kind=kind,
                )
            except Exception as error:
                failed_kinds.append(f"{kind}: {error}")
                continue

            any_success = True
            area_column = f"{kind}_area_um2"
            fraction_column = f"{kind}_area_fraction"
            recalculated_area_column = f"recalculated_{kind}_area_um2"
            recalculated_fraction_column = f"recalculated_{kind}_area_fraction"
            manifest = manifest.merge(
                occupancy_by_tile,
                on=["tile_row", "tile_col"],
                how="left",
            )
            manifest[recalculated_fraction_column] = pd.to_numeric(
                manifest[recalculated_fraction_column],
                errors="coerce",
            ).fillna(0.0)
            invalid_fraction = (
                (manifest[recalculated_fraction_column] < 0.0)
                | (manifest[recalculated_fraction_column] > 1.0)
            )
            if invalid_fraction.any():
                raise ValueError(
                    f"Rasterized {kind} occupancy produced values outside [0, 1]"
                )
            manifest[recalculated_area_column] = (
                manifest[recalculated_fraction_column] * tile_area_um2
            )
            manifest[f"recalculated_{kind}_area_method"] = (
                AREA_OCCUPANCY_METHOD_VERSION
            )
            if kind == "nucleus":
                manifest[area_column] = manifest[recalculated_area_column]
                manifest[fraction_column] = manifest[recalculated_fraction_column]
            elif area_column not in manifest.columns:
                manifest[area_column] = manifest[recalculated_area_column]
                manifest[fraction_column] = manifest[recalculated_fraction_column]
            summary[f"n_cells_with_{kind}_boundary"] = n_cells_with_boundary
            summary[f"recalculated_{kind}_area_method"] = (
                AREA_OCCUPANCY_METHOD_VERSION
            )
            summary[f"total_{kind}_area_um2"] = float(
                manifest[recalculated_area_column].sum()
            )
            summary[f"median_{kind}_area_fraction"] = float(
                manifest[recalculated_fraction_column].median()
            )
            summary[f"max_{kind}_area_fraction"] = float(
                manifest[recalculated_fraction_column].max()
            )

        if not any_success:
            raise ValueError("; ".join(failed_kinds) or "No area kind was calculated")

        if failed_kinds:
            summary["status"] = "partial"
            summary["error"] = "; ".join(failed_kinds)

        manifest.to_csv(output_manifest, index=False)
        return summary, replace(
            sample,
            manifest_path=output_manifest,
            manifest_scope=sample.manifest_scope,
            metrics_path=None,
        )
    except Exception as error:
        summary["status"] = "error"
        summary["error"] = str(error)
        summary["output_manifest_path"] = ""
        return summary, sample


def recalculate_boundary_area_metrics_for_samples(
    samples: Iterable[TileFilterSample],
    *,
    output_root: str | Path,
    area_kinds: Iterable[str] = ("cell",),
    tile_size_px: int | None = None,
) -> tuple[pd.DataFrame, list[TileFilterSample]]:
    summaries: list[dict[str, object]] = []
    recalculated_samples: list[TileFilterSample] = []
    for sample in samples:
        print(f"[{sample.dataset}/{sample.sample}] recalculating boundary occupancy")
        summary, updated_sample = recalculate_boundary_area_metrics_for_sample(
            sample,
            output_root=output_root,
            area_kinds=area_kinds,
            tile_size_px=tile_size_px,
        )
        summaries.append(summary)
        recalculated_samples.append(updated_sample)
    return pd.DataFrame(summaries), recalculated_samples


def tile_filter_sample_key(sample: TileFilterSample) -> str:
    return f"{sample.dataset}/{sample.sample}"


def recalculated_area_manifest_path(
    sample: TileFilterSample,
    recalculated_root: str | Path,
) -> Path:
    root = Path(recalculated_root).expanduser().resolve()
    return (
        root
        / safe_slug(sample.dataset)
        / safe_slug(sample.sample)
        / f"{sample.sample}_all_tile_labels_with_paths.csv"
    )


def select_active_tile_filter_samples(
    samples: Iterable[TileFilterSample],
    *,
    manifest_source: str = "original",
    recalculated_root: str | Path | None = None,
    source_by_dataset: dict[str, str] | None = None,
    source_by_sample_key: dict[str, str] | None = None,
    strict: bool = True,
) -> tuple[list[TileFilterSample], pd.DataFrame]:
    """Choose original or saved recalculated-area manifests for filtering."""
    valid_sources = {"original", "recalculated", "recalculated_if_available"}
    if manifest_source not in valid_sources:
        raise ValueError(f"manifest_source must be one of {sorted(valid_sources)}")
    source_by_dataset = source_by_dataset or {}
    source_by_sample_key = source_by_sample_key or {}
    recalculated_root = Path(recalculated_root).expanduser().resolve() if recalculated_root else None

    active_samples: list[TileFilterSample] = []
    rows: list[dict[str, object]] = []
    missing_required: list[str] = []
    for sample in samples:
        key = tile_filter_sample_key(sample)
        requested_source = source_by_sample_key.get(
            key,
            source_by_dataset.get(sample.dataset, manifest_source),
        )
        if requested_source not in valid_sources:
            raise ValueError(
                f"Invalid manifest source {requested_source!r} for {key}; "
                f"use one of {sorted(valid_sources)}"
            )

        active_sample = sample
        loaded_source = "original"
        status = "ok"
        recalculated_path = (
            recalculated_area_manifest_path(sample, recalculated_root)
            if recalculated_root is not None
            else None
        )
        if requested_source in {"recalculated", "recalculated_if_available"}:
            if recalculated_path is not None and recalculated_path.is_file():
                active_sample = replace(
                    sample,
                    manifest_path=recalculated_path,
                    metrics_path=None,
                )
                loaded_source = "recalculated"
            elif requested_source == "recalculated_if_available" or not strict:
                status = "missing_recalculated_using_original"
            else:
                status = "missing_recalculated"
                missing_required.append(key)

        active_samples.append(active_sample)
        rows.append(
            {
                "dataset": sample.dataset,
                "sample": sample.sample,
                "sample_key": key,
                "requested_manifest_source": requested_source,
                "loaded_manifest_source": loaded_source,
                "status": status,
                "original_manifest_path": str(sample.manifest_path),
                "recalculated_manifest_path": str(recalculated_path or ""),
                "active_manifest_path": str(active_sample.manifest_path),
            }
        )

    summary = pd.DataFrame(rows)
    if missing_required:
        raise FileNotFoundError(
            "Missing recalculated area manifest(s) for: "
            + ", ".join(missing_required)
            + ". Use manifest_source='recalculated_if_available', set strict=False, "
            + "or recompute boundary occupancy for those samples."
        )
    return active_samples, summary


def resolve_area_fraction_column(
    sample: TileFilterSample,
    *,
    default: str = "cell_area_fraction",
    by_dataset: dict[str, str] | None = None,
    by_sample_key: dict[str, str] | None = None,
) -> str:
    by_dataset = by_dataset or {}
    by_sample_key = by_sample_key or {}
    key = tile_filter_sample_key(sample)
    if key in by_sample_key:
        return str(by_sample_key[key])
    if sample.dataset in by_dataset:
        return str(by_dataset[sample.dataset])
    return str(default)


def area_um2_column_for_fraction(area_fraction_column: str) -> str | None:
    if area_fraction_column.endswith("_area_fraction"):
        return f"{area_fraction_column.removesuffix('_area_fraction')}_area_um2"
    if area_fraction_column.endswith("_fraction"):
        return f"{area_fraction_column.removesuffix('_fraction')}_um2"
    return None


def manifest_with_selected_area_fraction(
    manifest: pd.DataFrame,
    area_fraction_column: str,
) -> pd.DataFrame:
    """Return a manifest copy whose filter-compatible area column is selected."""
    table = manifest.copy()
    if area_fraction_column not in table.columns:
        raise ValueError(
            f"Area fraction column {area_fraction_column!r} is absent. "
            f"Available fraction columns: {sorted(column for column in table.columns if 'fraction' in column)}"
        )
    if area_fraction_column.startswith("recalculated_") and area_fraction_column.endswith(
        "_area_fraction"
    ):
        area_kind = area_fraction_column.removeprefix("recalculated_").removesuffix(
            "_area_fraction"
        )
        method_column = f"recalculated_{area_kind}_area_method"
        if method_column not in table.columns or not bool(
            table[method_column]
            .astype(str)
            .eq(AREA_OCCUPANCY_METHOD_VERSION)
            .all()
        ):
            raise ValueError(
                f"{area_fraction_column} was not produced by "
                f"{AREA_OCCUPANCY_METHOD_VERSION!r}. Recompute boundary-area "
                "recalculation before filtering."
            )
    if area_fraction_column != "cell_area_fraction":
        table["source_cell_area_fraction"] = table.get("cell_area_fraction", np.nan)
        table["cell_area_fraction"] = pd.to_numeric(
            table[area_fraction_column],
            errors="coerce",
        )
        area_um2_column = area_um2_column_for_fraction(area_fraction_column)
        if area_um2_column is not None and area_um2_column in table.columns:
            table["source_cell_area_um2"] = table.get("cell_area_um2", np.nan)
            table["cell_area_um2"] = pd.to_numeric(
                table[area_um2_column],
                errors="coerce",
            )
    table["filter_area_fraction"] = pd.to_numeric(
        table["cell_area_fraction"],
        errors="coerce",
    )
    table["area_fraction_filter_column"] = area_fraction_column
    return table


def combined_table_with_selected_area_fraction(
    table: pd.DataFrame,
    *,
    default: str = "cell_area_fraction",
    by_dataset: dict[str, str] | None = None,
    by_sample_key: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Apply per-dataset/per-sample area metric choices to a combined table."""
    by_dataset = by_dataset or {}
    by_sample_key = by_sample_key or {}
    if table.empty:
        return table.copy()
    prepared: list[pd.DataFrame] = []
    keyed = table.copy()
    keyed["sample_key"] = keyed["dataset"].astype(str) + "/" + keyed["sample"].astype(str)
    for sample_key_value, group in keyed.groupby("sample_key", sort=False):
        dataset = str(group["dataset"].iloc[0])
        area_column = by_sample_key.get(
            str(sample_key_value),
            by_dataset.get(dataset, default),
        )
        prepared.append(manifest_with_selected_area_fraction(group, str(area_column)))
    return pd.concat(prepared, ignore_index=True, sort=False)


def estimate_lower_mad_cutoff(
    values: pd.Series,
    *,
    metric_kind: str,
    n_mads: float = 3.0,
    min_finite: int = 20,
) -> dict[str, object]:
    """Estimate a one-sided lower cutoff on a transformed metric scale."""
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)].astype(float)
    result: dict[str, object] = {
        "status": "ok",
        "n_finite": int(len(finite)),
        "n_out_of_range": 0,
        "transformed_median": np.nan,
        "transformed_mad": np.nan,
        "cutoff": np.nan,
    }
    if len(finite) < int(min_finite):
        result["status"] = "insufficient_values"
        return result

    if metric_kind == "count":
        out_of_range = finite < 0
        result["n_out_of_range"] = int(out_of_range.sum())
        finite = finite.loc[~out_of_range]
        transformed = np.log1p(finite.to_numpy())
    elif metric_kind == "fraction":
        tolerance = 1e-9
        out_of_range = (finite < -tolerance) | (finite > 1.0 + tolerance)
        result["n_out_of_range"] = int(out_of_range.sum())
        if out_of_range.any():
            result["status"] = "invalid_out_of_range_values"
            return result
        finite = finite.clip(lower=0.0, upper=1.0)
        epsilon = 0.5 / (len(finite) + 1.0)
        clipped = finite.clip(lower=epsilon, upper=1.0 - epsilon).to_numpy()
        transformed = np.log(clipped / (1.0 - clipped))
    else:
        raise ValueError("metric_kind must be 'count' or 'fraction'")

    transformed = transformed[np.isfinite(transformed)]
    if len(transformed) < int(min_finite):
        result["status"] = "insufficient_values"
        result["n_finite"] = int(len(transformed))
        return result
    median = float(np.median(transformed))
    mad = float(np.median(np.abs(transformed - median)))
    result["transformed_median"] = median
    result["transformed_mad"] = mad
    if not np.isfinite(mad) or mad <= np.finfo(float).eps:
        result["status"] = "zero_or_invalid_mad"
        return result

    lower = median - float(n_mads) * mad
    if metric_kind == "count":
        cutoff = max(1, int(np.ceil(np.expm1(lower))))
    else:
        lower = float(np.clip(lower, -700, 700))
        cutoff = float(1.0 / (1.0 + np.exp(-lower)))
        cutoff = float(np.clip(cutoff, 0.0, 1.0))
    result["cutoff"] = cutoff
    return result


def estimate_data_adaptive_cutoffs(
    tile_table: pd.DataFrame,
    *,
    n_mads: float = 3.0,
    min_finite: int = 20,
    manual_fallbacks: dict[str, float] | None = None,
    apply_nucleus_area_fraction: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate sample cutoffs with dataset and manual fallback levels."""
    if tile_table.empty:
        return pd.DataFrame(), pd.DataFrame()
    required = {"dataset", "sample", "n_cells", "filter_area_fraction"}
    missing = required - set(tile_table.columns)
    if missing:
        raise ValueError(f"tile_table is missing adaptive-filter columns: {sorted(missing)}")
    table = tile_table.copy()
    table["sample_key"] = table["dataset"].astype(str) + "/" + table["sample"].astype(str)
    defaults = {
        "n_cells": 1.0,
        "filter_area_fraction": 0.0,
        "nucleus_area_fraction": 0.0,
        "he_tissue_fraction": 0.0,
    }
    defaults.update(manual_fallbacks or {})
    metric_specs = {
        "n_cells": ("count", "min_cells_per_tile"),
        "filter_area_fraction": ("fraction", "min_cell_area_fraction"),
        "nucleus_area_fraction": ("fraction", "min_nucleus_area_fraction"),
        "he_tissue_fraction": ("fraction", "min_he_tissue_fraction"),
    }
    for metric in metric_specs:
        if metric not in table.columns:
            table[metric] = np.nan
    invalid_fraction_rows: list[str] = []
    for metric in (
        "filter_area_fraction",
        "nucleus_area_fraction",
        "he_tissue_fraction",
    ):
        values = pd.to_numeric(table[metric], errors="coerce")
        invalid = values.notna() & ((values < -1e-9) | (values > 1.0 + 1e-9))
        if invalid.any():
            counts = (
                table.loc[invalid]
                .groupby("sample_key", sort=True)
                .size()
                .sort_values(ascending=False)
            )
            examples = ", ".join(
                f"{sample_key}: {count:,}"
                for sample_key, count in counts.head(5).items()
            )
            invalid_fraction_rows.append(
                f"{metric} has {int(invalid.sum()):,} invalid value(s) ({examples})"
            )
    if invalid_fraction_rows:
        raise ValueError(
            "Adaptive filtering requires fraction metrics in [0, 1]. "
            + "; ".join(invalid_fraction_rows)
        )

    dataset_estimates: dict[tuple[str, str], dict[str, object]] = {}
    for dataset, group in table.groupby("dataset", sort=True):
        for metric, (kind, _output_column) in metric_specs.items():
            dataset_estimates[(str(dataset), metric)] = estimate_lower_mad_cutoff(
                group[metric],
                metric_kind=kind,
                n_mads=n_mads,
                min_finite=min_finite,
            )

    recommendations: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for sample_key, group in table.groupby("sample_key", sort=True):
        dataset = str(group["dataset"].iloc[0])
        sample = str(group["sample"].iloc[0])
        recommendation: dict[str, object] = {
            "dataset": dataset,
            "sample": sample,
            "sample_key": str(sample_key),
            "n_mads": float(n_mads),
            "min_finite": int(min_finite),
            "area_fraction_filter_column": str(
                group.get(
                    "area_fraction_filter_column",
                    pd.Series("filter_area_fraction", index=group.index),
                ).iloc[0]
            ),
        }
        for metric, (kind, output_column) in metric_specs.items():
            sample_estimate = estimate_lower_mad_cutoff(
                group[metric],
                metric_kind=kind,
                n_mads=n_mads,
                min_finite=min_finite,
            )
            dataset_estimate = dataset_estimates[(dataset, metric)]
            fallback = float(defaults[metric])
            if str(sample_estimate["status"]).startswith("ok"):
                raw_cutoff = float(sample_estimate["cutoff"])
                source = "sample"
            elif (
                int(sample_estimate["n_finite"]) > 0
                and str(dataset_estimate["status"]).startswith("ok")
            ):
                raw_cutoff = float(dataset_estimate["cutoff"])
                source = "dataset"
            else:
                raw_cutoff = fallback
                source = "manual_fallback"
            chosen = max(raw_cutoff, fallback)
            if metric == "n_cells":
                chosen = int(np.ceil(chosen))
            if metric == "nucleus_area_fraction" and not apply_nucleus_area_fraction:
                applied = 0.0
            else:
                applied = chosen
            recommendation[output_column] = applied
            recommendation[f"recommended_{output_column}"] = chosen
            recommendation[f"{output_column}_source"] = source
            recommendation[f"{output_column}_manual_floor_applied"] = bool(chosen > raw_cutoff)
            diagnostics.append(
                {
                    "dataset": dataset,
                    "sample": sample,
                    "sample_key": str(sample_key),
                    "metric": metric,
                    "output_column": output_column,
                    "sample_status": sample_estimate["status"],
                    "sample_n_finite": sample_estimate["n_finite"],
                    "sample_n_out_of_range": sample_estimate["n_out_of_range"],
                    "sample_transformed_median": sample_estimate["transformed_median"],
                    "sample_transformed_mad": sample_estimate["transformed_mad"],
                    "sample_cutoff": sample_estimate["cutoff"],
                    "dataset_status": dataset_estimate["status"],
                    "dataset_cutoff": dataset_estimate["cutoff"],
                    "manual_fallback": fallback,
                    "chosen_cutoff": chosen,
                    "applied_cutoff": applied,
                    "cutoff_source": source,
                }
            )

        evaluation_group = group.copy()
        evaluation_group["cell_area_fraction"] = pd.to_numeric(
            evaluation_group["filter_area_fraction"],
            errors="coerce",
        )
        filtered = standard_xenium.apply_tile_filter_cutoffs(
            evaluation_group,
            int(recommendation["min_cells_per_tile"]),
            float(recommendation["min_cell_area_fraction"]),
            float(recommendation["min_nucleus_area_fraction"]),
            float(recommendation["min_he_tissue_fraction"]),
        )
        recommendation["n_nonempty_tiles"] = len(filtered)
        recommendation["n_passing_tiles"] = int(filtered["passes_tile_filter"].sum())
        recommendation["retained_tile_fraction"] = float(
            filtered["passes_tile_filter"].mean()
        )
        recommendations.append(recommendation)
    return pd.DataFrame(recommendations), pd.DataFrame(diagnostics)


def save_cutoff_outputs(
    sample: TileFilterSample,
    config: standard_xenium.PipelineConfig,
    *,
    output_root: str | Path,
    area_fraction_column: str = "cell_area_fraction",
    materialize_tiles: bool = False,
    materialize_mode: str = "copy",
    materialized_root: str | Path | None = None,
    examples_per_quadrant: int = 3,
    max_scatter_points: int = 50_000,
    save_quadrant_examples: bool = True,
    draw_example_overlays: bool = True,
    min_nucleus_area_fraction: float = 0.0,
    min_he_tissue_fraction: float = 0.0,
    require_nucleus_area_fraction: bool = False,
    require_he_tissue_fraction: bool = False,
) -> dict[str, object]:
    """Save plot, passing tile IDs, and summary for one sample/cutoff combination."""
    output_root = Path(output_root).expanduser().resolve()
    sample_output_dir = output_root / safe_slug(sample.dataset)
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    manifest = manifest_with_selected_area_fraction(
        load_sample_manifest(sample),
        area_fraction_column,
    )
    manifest = standard_xenium.apply_tile_filter_cutoffs(
        manifest,
        config.min_cells_per_tile,
        config.min_cell_area_fraction,
        min_nucleus_area_fraction,
        min_he_tissue_fraction,
        require_nucleus_area_fraction=require_nucleus_area_fraction,
        require_he_tissue_fraction=require_he_tissue_fraction,
    )
    manifest["min_nucleus_area_fraction"] = float(min_nucleus_area_fraction)
    manifest["min_he_tissue_fraction"] = float(min_he_tissue_fraction)
    if {"tile_row", "tile_col"}.issubset(manifest.columns):
        manifest = manifest.sort_values(["tile_row", "tile_col"]).reset_index(drop=True)
    selected = manifest.loc[manifest["passes_tile_filter"]].copy()

    stem = sample_output_stem(
        sample,
        config,
        min_nucleus_area_fraction,
        min_he_tissue_fraction,
    )
    if area_fraction_column != "cell_area_fraction":
        stem = f"{stem}__area-{safe_slug(area_fraction_column)}"
    passing_path = sample_output_dir / f"{stem}__passing_tile_ids.csv"
    add_cross_dataset_tile_ids(sample, selected, config).to_csv(passing_path, index=False)

    summary, _ = standard_xenium.tile_filter_selection_summary(
        manifest,
        config.min_cells_per_tile,
        config.min_cell_area_fraction,
        min_nucleus_area_fraction,
        min_he_tissue_fraction,
        require_nucleus_area_fraction=require_nucleus_area_fraction,
        require_he_tissue_fraction=require_he_tissue_fraction,
    )
    summary.insert(0, "dataset", sample.dataset)
    summary.insert(1, "sample", sample.sample)
    summary.insert(2, "cutoff_slug", standard_xenium.tile_filter_cutoff_slug(
        config.min_cells_per_tile,
        config.min_cell_area_fraction,
        min_nucleus_area_fraction,
        min_he_tissue_fraction,
    ))
    summary["manifest_scope"] = sample.manifest_scope
    summary["has_all_nonempty_manifest"] = sample.manifest_scope in {
        "all_nonempty",
        "combined_all_samples",
    }
    summary["has_cell_area_fraction"] = bool(manifest["cell_area_fraction"].notna().any())
    summary["has_nucleus_area_fraction"] = bool(
        pd.to_numeric(
            manifest.get("nucleus_area_fraction", pd.Series(np.nan, index=manifest.index)),
            errors="coerce",
        ).notna().any()
    )
    summary["has_he_tissue_fraction"] = bool(
        pd.to_numeric(
            manifest.get("he_tissue_fraction", pd.Series(np.nan, index=manifest.index)),
            errors="coerce",
        ).notna().any()
    )
    summary["area_fraction_filter_column"] = area_fraction_column
    summary["n_nonempty_tiles_in_manifest"] = len(manifest)
    summary["n_passing_nonempty_tiles"] = len(selected)
    for column in [
        "tile_area_um2",
        "tile_width_um",
        "tile_height_um",
        "tile_side_um",
        "effective_pixel_size_um",
    ]:
        if column in manifest.columns:
            values = pd.to_numeric(manifest[column], errors="coerce").dropna()
            summary[f"median_{column}"] = values.median() if len(values) else np.nan
    summary["passing_tile_ids_path"] = str(passing_path)
    summary["manifest_path"] = str(sample.manifest_path)

    plot = None
    plot_error = ""
    if save_quadrant_examples and examples_per_quadrant > 0:
        plot_path = sample_output_dir / f"{stem}__tile_filter_quadrant_examples.png"
        overlay = _example_context(
            sample,
            config,
            include_overlays=draw_example_overlays,
        )
        example_paths = overlay.get("paths")
        if example_paths is None:
            plot_error = "example context unavailable"
        else:
            try:
                plot = standard_xenium.save_tile_filter_quadrant_examples(
                    example_paths,
                    manifest,
                    config,
                    tile_cells=overlay.get("tile_cells"),
                    cell_boundaries=overlay.get("cell_boundaries"),
                    nucleus_boundaries=overlay.get("nucleus_boundaries"),
                    metadata=overlay.get("metadata"),
                    pixel_size=overlay.get("pixel_size"),
                    xenium_to_he=overlay.get("xenium_to_he"),
                    examples_per_quadrant=examples_per_quadrant,
                    max_scatter_points=max_scatter_points,
                    output_path=plot_path,
                    area_fraction_label=area_fraction_column,
                    min_nucleus_area_fraction=min_nucleus_area_fraction,
                    min_he_tissue_fraction=min_he_tissue_fraction,
                    require_nucleus_area_fraction=require_nucleus_area_fraction,
                    require_he_tissue_fraction=require_he_tissue_fraction,
                )
            except Exception as error:
                plot_error = str(error)
                print(f"[{sample.dataset}/{sample.sample}] quadrant examples unavailable: {error}")
    summary["quadrant_examples_path"] = str(plot) if plot is not None else ""
    summary["quadrant_examples_error"] = plot_error

    materialized_tile_dir = ""
    materialized_tile_manifest = ""
    n_materialized = 0
    n_missing_tile_files = 0
    if materialize_tiles:
        cutoff_slug = standard_xenium.tile_filter_cutoff_slug(
            config.min_cells_per_tile,
            config.min_cell_area_fraction,
            min_nucleus_area_fraction,
            min_he_tissue_fraction,
        )
        if area_fraction_column != "cell_area_fraction":
            cutoff_slug = f"{cutoff_slug}__area-{safe_slug(area_fraction_column)}"
        root = Path(materialized_root).expanduser().resolve() if materialized_root else output_root / "selected_tiles"
        destination = root / cutoff_slug / safe_slug(sample.dataset) / safe_slug(sample.sample)
        materialized = materialize_selected_tiles(
            sample,
            selected,
            config,
            destination,
            mode=materialize_mode,
        )
        materialized_path = sample_output_dir / f"{stem}__materialized_tiles.csv"
        materialized.to_csv(materialized_path, index=False)
        materialized_tile_dir = str(destination)
        materialized_tile_manifest = str(materialized_path)
        n_materialized = int((materialized["status"] == materialize_mode).sum())
        n_missing_tile_files = int((materialized["status"] == "missing_source").sum())
    summary["materialized_tile_dir"] = materialized_tile_dir
    summary["materialized_tile_manifest"] = materialized_tile_manifest
    summary["n_materialized_tiles"] = n_materialized
    summary["n_missing_tile_files"] = n_missing_tile_files

    summary_path = sample_output_dir / f"{stem}__tile_filter_summary.csv"
    summary.to_csv(summary_path, index=False)

    result = summary.iloc[0].to_dict()
    result.update(
        {
            "summary_path": str(summary_path),
            "passing_tile_ids_path": str(passing_path),
            "quadrant_examples_path": str(plot) if plot is not None else "",
        }
    )
    return result


def same_cutoff(left: dict[str, object], right: dict[str, object] | None) -> bool:
    if right is None:
        return False
    return (
        int(left["min_cells_per_tile"]) == int(right["min_cells_per_tile"])
        and abs(
            float(left["min_cell_area_fraction"])
            - float(right["min_cell_area_fraction"])
        )
        <= 1e-12
        and abs(
            float(left.get("min_nucleus_area_fraction", 0.0))
            - float(right.get("min_nucleus_area_fraction", 0.0))
        )
        <= 1e-12
        and abs(
            float(left.get("min_he_tissue_fraction", 0.0))
            - float(right.get("min_he_tissue_fraction", 0.0))
        )
        <= 1e-12
    )


def save_cutoff_grid_outputs(
    samples: Iterable[TileFilterSample],
    cutoffs: Iterable[dict[str, object]],
    *,
    tile_size_px: int | None = None,
    output_root: str | Path,
    materialize_cutoff: dict[str, object] | None = None,
    materialize_mode: str = "copy",
    materialized_root: str | Path | None = None,
    examples_per_quadrant: int = 3,
    max_scatter_points: int = 50_000,
    area_fraction_column: str = "cell_area_fraction",
    area_fraction_column_by_dataset: dict[str, str] | None = None,
    area_fraction_column_by_sample_key: dict[str, str] | None = None,
    save_quadrant_examples: bool = True,
    draw_example_overlays: bool = True,
    require_nucleus_area_fraction: bool = False,
    require_he_tissue_fraction: bool = False,
) -> pd.DataFrame:
    results: list[dict[str, object]] = []
    samples = list(samples)
    for cutoff in cutoffs:
        materialize = same_cutoff(cutoff, materialize_cutoff)
        for sample in samples:
            resolved_area_fraction_column = resolve_area_fraction_column(
                sample,
                default=area_fraction_column,
                by_dataset=area_fraction_column_by_dataset,
                by_sample_key=area_fraction_column_by_sample_key,
            )
            config = standard_xenium.PipelineConfig(
                tile_size_px=int(tile_size_px or infer_tile_size_px(sample)),
                min_cells_per_tile=int(cutoff["min_cells_per_tile"]),
                min_cell_area_fraction=float(cutoff["min_cell_area_fraction"]),
            )
            min_nucleus_area_fraction = float(
                cutoff.get("min_nucleus_area_fraction", 0.0)
            )
            min_he_tissue_fraction = float(
                cutoff.get("min_he_tissue_fraction", 0.0)
            )
            results.append(
                save_cutoff_outputs(
                    sample,
                    config,
                    output_root=output_root,
                    area_fraction_column=resolved_area_fraction_column,
                    materialize_tiles=materialize,
                    materialize_mode=materialize_mode,
                    materialized_root=materialized_root,
                    examples_per_quadrant=examples_per_quadrant,
                    max_scatter_points=max_scatter_points,
                    save_quadrant_examples=save_quadrant_examples,
                    draw_example_overlays=draw_example_overlays,
                    min_nucleus_area_fraction=min_nucleus_area_fraction,
                    min_he_tissue_fraction=min_he_tissue_fraction,
                    require_nucleus_area_fraction=require_nucleus_area_fraction,
                    require_he_tissue_fraction=require_he_tissue_fraction,
                )
            )
    summary = pd.DataFrame(results)
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "all_cutoff_test_summary.csv", index=False)
    return summary


def save_adaptive_cutoff_outputs(
    samples: Iterable[TileFilterSample],
    recommendations: pd.DataFrame,
    *,
    output_root: str | Path,
    tile_size_px: int | None = None,
    area_fraction_column: str = "cell_area_fraction",
    area_fraction_column_by_dataset: dict[str, str] | None = None,
    area_fraction_column_by_sample_key: dict[str, str] | None = None,
    materialize_tiles: bool = False,
    materialize_mode: str = "copy",
    materialized_root: str | Path | None = None,
    examples_per_quadrant: int = 3,
    max_scatter_points: int = 50_000,
    save_quadrant_examples: bool = True,
    draw_example_overlays: bool = True,
    require_nucleus_area_fraction: bool = False,
    require_he_tissue_fraction: bool = False,
) -> pd.DataFrame:
    """Apply one recommended cutoff combination per sample and save outputs."""
    if recommendations.empty:
        raise ValueError("recommendations is empty")
    required = {
        "sample_key",
        "min_cells_per_tile",
        "min_cell_area_fraction",
        "min_nucleus_area_fraction",
        "min_he_tissue_fraction",
    }
    missing = required - set(recommendations.columns)
    if missing:
        raise ValueError(f"recommendations is missing columns: {sorted(missing)}")
    indexed = recommendations.copy()
    if indexed["sample_key"].duplicated().any():
        duplicates = indexed.loc[
            indexed["sample_key"].duplicated(keep=False), "sample_key"
        ].astype(str).tolist()
        raise ValueError(f"Duplicate adaptive cutoff rows: {duplicates}")
    indexed = indexed.set_index("sample_key", drop=False)
    results: list[dict[str, object]] = []
    for sample in samples:
        key = tile_filter_sample_key(sample)
        if key not in indexed.index:
            raise KeyError(f"No adaptive cutoff recommendation for {key}")
        cutoff = indexed.loc[key]
        resolved_area_fraction_column = resolve_area_fraction_column(
            sample,
            default=area_fraction_column,
            by_dataset=area_fraction_column_by_dataset,
            by_sample_key=area_fraction_column_by_sample_key,
        )
        config = standard_xenium.PipelineConfig(
            tile_size_px=int(tile_size_px or infer_tile_size_px(sample)),
            min_cells_per_tile=int(cutoff["min_cells_per_tile"]),
            min_cell_area_fraction=float(cutoff["min_cell_area_fraction"]),
        )
        result = save_cutoff_outputs(
            sample,
            config,
            output_root=output_root,
            area_fraction_column=resolved_area_fraction_column,
            materialize_tiles=materialize_tiles,
            materialize_mode=materialize_mode,
            materialized_root=materialized_root,
            examples_per_quadrant=examples_per_quadrant,
            max_scatter_points=max_scatter_points,
            save_quadrant_examples=save_quadrant_examples,
            draw_example_overlays=draw_example_overlays,
            min_nucleus_area_fraction=float(cutoff["min_nucleus_area_fraction"]),
            min_he_tissue_fraction=float(cutoff["min_he_tissue_fraction"]),
            require_nucleus_area_fraction=require_nucleus_area_fraction,
            require_he_tissue_fraction=require_he_tissue_fraction,
        )
        result["cutoff_mode"] = "data_adaptive"
        for column in recommendations.columns:
            if column.endswith("_source") or column.startswith("recommended_"):
                result[column] = cutoff[column]
        results.append(result)
    summary = pd.DataFrame(results)
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "all_adaptive_cutoff_summary.csv", index=False)
    return summary


def display_tile_filter_explorer(
    tile_table: pd.DataFrame,
    *,
    default_min_cells: int = 1,
    default_min_cell_area_fraction: float = 0.0,
    default_min_nucleus_area_fraction: float = 0.0,
    default_min_he_tissue_fraction: float = 0.0,
    nucleus_area_fraction_column: str = "nucleus_area_fraction",
    require_nucleus_area_fraction: bool = False,
    require_he_tissue_fraction: bool = False,
) -> dict[str, object]:
    """Display an ipywidgets explorer for the combined cross-dataset table."""
    try:
        import io
        import ipywidgets as widgets
        from IPython.display import Image as DisplayImage
        from IPython.display import clear_output, display
    except ImportError as error:
        raise ImportError("ipywidgets is required for the interactive explorer") from error

    if tile_table.empty:
        raise ValueError("tile_table is empty")
    table = tile_table.copy()
    table["key"] = table["dataset"].astype(str) + "/" + table["sample"].astype(str)
    dataset_options = [("All datasets", "__ALL__")]
    dataset_options.extend(
        (dataset, f"dataset::{dataset}")
        for dataset in sorted(table["dataset"].dropna().astype(str).unique())
    )
    sample_options = [
        (key, f"sample::{key}")
        for key in sorted(table["key"].dropna().astype(str).unique())
    ]
    selector = widgets.Dropdown(
        options=dataset_options + sample_options,
        value="__ALL__",
        description="Scope:",
        layout=widgets.Layout(width="520px"),
    )
    min_cells = widgets.BoundedIntText(
        value=int(default_min_cells),
        min=1,
        max=max(1, int(pd.to_numeric(table["n_cells"], errors="coerce").max())),
        step=1,
        description="Min cells:",
    )
    min_area = widgets.FloatText(
        value=float(default_min_cell_area_fraction),
        description="Min area:",
    )
    min_nucleus_area = widgets.FloatText(
        value=float(default_min_nucleus_area_fraction),
        description="Min nucleus:",
    )
    min_he_tissue = widgets.FloatText(
        value=float(default_min_he_tissue_fraction),
        description="Min tissue:",
    )
    output = widgets.Output()

    def subset_for_value(value: str) -> pd.DataFrame:
        if value == "__ALL__":
            return table
        if value.startswith("dataset::"):
            return table.loc[table["dataset"].astype(str).eq(value.removeprefix("dataset::"))]
        return table.loc[table["key"].eq(value.removeprefix("sample::"))]

    def selector_label() -> str:
        for label, value in selector.options:
            if value == selector.value:
                return str(label)
        return str(selector.value)

    def resolved_nucleus_area_column(subset: pd.DataFrame) -> str:
        if (
            nucleus_area_fraction_column in subset.columns
            and pd.to_numeric(
                subset[nucleus_area_fraction_column],
                errors="coerce",
            ).notna().any()
        ):
            return nucleus_area_fraction_column
        for candidate in ("nucleus_area_fraction", "recalculated_nucleus_area_fraction"):
            if (
                candidate in subset.columns
                and pd.to_numeric(subset[candidate], errors="coerce").notna().any()
            ):
                return candidate
        return nucleus_area_fraction_column

    def update(_change=None) -> None:
        with output:
            clear_output(wait=True)
            subset = subset_for_value(selector.value)
            if "area_fraction_filter_column" in subset.columns:
                area_labels = subset["area_fraction_filter_column"].dropna().astype(str).unique()
                area_label = area_labels[0] if len(area_labels) == 1 else "selected_area_fraction"
            else:
                area_label = "cell_area_fraction"
            nucleus_column = resolved_nucleus_area_column(subset)
            title = f"Tile-filter threshold explorer: {selector_label()}"
            fig = standard_xenium.plot_tile_filter_selection(
                subset,
                int(min_cells.value),
                float(min_area.value),
                float(min_nucleus_area.value),
                float(min_he_tissue.value),
                title=title,
                area_fraction_label=area_label,
                nucleus_area_fraction_column=nucleus_column,
                nucleus_area_fraction_label=nucleus_column,
                require_nucleus_area_fraction=require_nucleus_area_fraction,
                require_he_tissue_fraction=require_he_tissue_fraction,
            )
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
            display(DisplayImage(data=buffer.getvalue()))
            standard_xenium.plt.close(fig)

    for widget in (selector, min_cells, min_area, min_nucleus_area, min_he_tissue):
        widget.observe(update, names="value")
    display(
        widgets.VBox(
            [
                widgets.HBox([selector, min_cells, min_area, min_nucleus_area, min_he_tissue]),
                output,
            ]
        )
    )
    update()
    return {
        "selector": selector,
        "min_cells": min_cells,
        "min_cell_area_fraction": min_area,
        "min_nucleus_area_fraction": min_nucleus_area,
        "min_he_tissue_fraction": min_he_tissue,
        "output": output,
    }
