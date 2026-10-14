#!/usr/bin/env python3
"""Create H&E-versus-Xenium contact sheets for sample mapping.

For CZI sources, the top row contains one thumbnail per CZI scene. For NDPI
sources, the top row contains automatically detected tissue-region candidates
from a low-resolution pyramid level. The bottom row contains cell-centroid
silhouettes for the Xenium samples named in the source filename.

The figures support visual mapping only. A candidate must still be confirmed by
registering the H&E image to the Xenium morphology image and inspecting an
overlay of cell centroids or boundaries.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import csv
import gzip
from io import BytesIO
import math
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy import ndimage
import tifffile
from tqdm.auto import tqdm

from xenium_pipeline.bilous_extract_batch import find_breast_he_files
from xenium_pipeline.bilous_extract import (
    ensure_uncompressed,
    normalize_rgb_tile,
    require_common_dependencies,
    source_kind,
    strip_known_suffixes,
    zarr_array_from_root,
)


SAMPLE_DIRECTORY_PATTERN = re.compile(
    r"^(?P<label>B\d+(?:_\d+)?)_(?P<gsm>GSM\d+)$"
)


@dataclass(frozen=True)
class Sample:
    directory: Path
    label: str
    gsm: str


def discover_samples(root: Path) -> list[Sample]:
    samples = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        match = SAMPLE_DIRECTORY_PATTERN.match(directory.name)
        if match:
            samples.append(
                Sample(
                    directory=directory,
                    label=match.group("label"),
                    gsm=match.group("gsm"),
                )
            )
    return samples


def sample_filename_token(label: str) -> str:
    """Convert B10 or B1_3 into its unambiguous GEO filename token."""
    parts = label.split("_")
    patient = parts[0]
    if len(parts) == 1:
        return f"_{patient}_{patient}"
    return f"_{patient}_{patient}_{parts[1]}"


def samples_for_source(source: Path, samples: list[Sample]) -> list[Sample]:
    return [
        sample
        for sample in samples
        if sample_filename_token(sample.label) in source.name
    ]


def read_cells(sample: Sample) -> pd.DataFrame:
    candidates = sorted(sample.directory.glob(f"{sample.gsm}_*_cells.parquet"))
    if candidates:
        return pd.read_parquet(
            candidates[0], columns=["x_centroid", "y_centroid"]
        )

    compressed = sorted(
        sample.directory.glob(f"{sample.gsm}_*_cells.parquet.gz")
    )
    if not compressed:
        raise FileNotFoundError(
            f"No cells parquet file found for {sample.directory.name}"
        )
    with gzip.open(compressed[0], "rb") as handle:
        payload = BytesIO(handle.read())
    return pd.read_parquet(
        payload, columns=["x_centroid", "y_centroid"]
    )


def plot_centroids(axis: Any, sample: Sample) -> None:
    cells = read_cells(sample)
    axis.scatter(
        cells["x_centroid"],
        cells["y_centroid"],
        s=0.25,
        alpha=0.65,
        color="black",
        linewidths=0,
        rasterized=True,
    )
    axis.invert_yaxis()
    axis.set_aspect("equal")
    axis.set_title(
        f"{sample.label} / {sample.gsm}\n{len(cells):,} cell centroids"
    )
    axis.axis("off")


def czi_scene_previews(
    source: Path,
    max_dimension: int,
) -> tuple[list[dict[str, Any]], Path]:
    try:
        from aicspylibczi import CziFile
    except ImportError as exc:
        raise ImportError("Install aicspylibczi before reading CZI files") from exc

    _, _, tqdm_class = require_common_dependencies()
    uncompressed, _ = ensure_uncompressed(source, tqdm_class)
    czi = CziFile(uncompressed)
    boxes = czi.get_all_mosaic_scene_bounding_boxes()
    candidates = []
    for scene_index in sorted(boxes):
        box = boxes[scene_index]
        scale = min(1.0, max_dimension / max(box.w, box.h))
        image = czi.read_mosaic(
            C=0,
            scale_factor=scale,
            region=(box.x, box.y, box.w, box.h),
            background_color=(1.0, 1.0, 1.0),
        )
        image = normalize_rgb_tile(image, str(czi.pixel_type), np)
        candidates.append(
            {
                "index": scene_index,
                "image": image,
                "title": (
                    f"scene_{scene_index:02d}\n"
                    f"{box.w:,} × {box.h:,} px; aspect {box.w / box.h:.3f}"
                ),
                "bbox": (box.x, box.y, box.w, box.h),
            }
        )
    return candidates, uncompressed


def spatial_area(series: Any) -> int:
    if "Y" not in series.axes or "X" not in series.axes:
        return 0
    return int(
        series.shape[series.axes.index("Y")]
        * series.shape[series.axes.index("X")]
    )


def ndarray_to_yxs(image: Any, axes: str) -> np.ndarray:
    array = np.asarray(image)
    if len(axes) != array.ndim or "Y" not in axes or "X" not in axes:
        raise ValueError(f"Cannot match image shape {array.shape} to axes {axes!r}")

    selection = []
    retained_axes = []
    for axis, size in zip(axes, array.shape):
        if axis in {"Y", "X"}:
            selection.append(slice(None))
            retained_axes.append(axis)
        elif axis in {"S", "C", "A"} and size in {3, 4}:
            selection.append(slice(None))
            retained_axes.append(axis)
        else:
            selection.append(0)
    array = array[tuple(selection)]
    y_axis = retained_axes.index("Y")
    x_axis = retained_axes.index("X")
    sample_axes = [
        retained_axes.index(axis)
        for axis in ("S", "C", "A")
        if axis in retained_axes
    ]
    if sample_axes:
        array = np.moveaxis(array, (y_axis, x_axis, sample_axes[0]), (0, 1, 2))
    else:
        array = np.moveaxis(array, (y_axis, x_axis), (0, 1))
    return normalize_rgb_tile(array, "rgb", np)


def read_zarr_thumbnail(level: Any, max_dimension: int) -> np.ndarray:
    import zarr

    store = level.aszarr()
    try:
        array = zarr_array_from_root(zarr.open(store, mode="r"))
        axes = level.axes
        y_axis = axes.index("Y")
        x_axis = axes.index("X")
        stride = max(
            1,
            int(
                math.ceil(
                    max(array.shape[y_axis], array.shape[x_axis])
                    / max_dimension
                )
            ),
        )
        selection = []
        retained_axes = []
        for axis, size in zip(axes, array.shape):
            if axis in {"Y", "X"}:
                selection.append(slice(None, None, stride))
                retained_axes.append(axis)
            elif axis in {"S", "C", "A"} and size in {3, 4}:
                selection.append(slice(None))
                retained_axes.append(axis)
            else:
                selection.append(0)
        image = np.asarray(array[tuple(selection)])
        return ndarray_to_yxs(image, "".join(retained_axes))
    finally:
        if hasattr(store, "close"):
            store.close()


def ndpi_thumbnail(
    source: Path,
    max_dimension: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    with tifffile.TiffFile(source) as tif:
        spatial_series = [series for series in tif.series if spatial_area(series)]
        if not spatial_series:
            raise ValueError(f"No spatial series found in {source}")
        base = max(spatial_series, key=spatial_area)
        base_height = int(base.shape[base.axes.index("Y")])
        base_width = int(base.shape[base.axes.index("X")])
        levels = list(getattr(base, "levels", [base]))

        small_enough = [
            level
            for level in levels
            if max(
                level.shape[level.axes.index("Y")],
                level.shape[level.axes.index("X")],
            )
            <= max_dimension
        ]
        if small_enough:
            chosen = max(small_enough, key=spatial_area)
            image = ndarray_to_yxs(chosen.asarray(), chosen.axes)
        else:
            chosen = min(levels, key=spatial_area)
            chosen_maximum = max(
                chosen.shape[chosen.axes.index("Y")],
                chosen.shape[chosen.axes.index("X")],
            )
            if chosen_maximum <= max_dimension * 2:
                image = ndarray_to_yxs(chosen.asarray(), chosen.axes)
                stride = max(1, math.ceil(chosen_maximum / max_dimension))
                image = image[::stride, ::stride]
            else:
                image = read_zarr_thumbnail(chosen, max_dimension)
    return image, (base_width, base_height)


def detect_tissue_regions(
    image: np.ndarray,
    expected_regions: int,
) -> list[tuple[int, int, int, int]]:
    """Return approximate tissue boxes in thumbnail coordinates."""
    rgb = image[..., :3].astype(np.float32) / 255.0
    maximum = rgb.max(axis=-1)
    minimum = rgb.min(axis=-1)
    saturation = (maximum - minimum) / np.maximum(maximum, 1e-6)
    brightness = rgb.mean(axis=-1)

    tissue = (
        ((saturation > 0.055) & (brightness < 0.985))
        | ((brightness < 0.88) & (brightness > 0.12))
    )
    tissue = ndimage.binary_opening(tissue, iterations=2)
    tissue = ndimage.binary_closing(tissue, iterations=6)
    tissue = ndimage.binary_fill_holes(tissue)
    tissue = ndimage.binary_dilation(tissue, iterations=8)

    labels, count = ndimage.label(tissue)
    objects = ndimage.find_objects(labels)
    components = []
    for component_index, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        area = int(np.count_nonzero(labels[slices] == component_index))
        y_slice, x_slice = slices
        components.append(
            (area, x_slice.start, y_slice.start, x_slice.stop, y_slice.stop)
        )
    components.sort(reverse=True)
    selected = components[:expected_regions]
    boxes = []
    padding = max(5, int(round(max(image.shape[:2]) * 0.01)))
    for _, x0, y0, x1, y1 in selected:
        boxes.append(
            (
                max(0, x0 - padding),
                max(0, y0 - padding),
                min(image.shape[1], x1 + padding),
                min(image.shape[0], y1 + padding),
            )
        )
    # Stable visual ordering: top-to-bottom, then left-to-right.
    return sorted(boxes, key=lambda box: (box[1], box[0]))


def save_ndpi_overview(
    source: Path,
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    output_dir: Path,
) -> Path:
    fig, axis = plt.subplots(figsize=(16, 10), constrained_layout=True)
    axis.imshow(image, interpolation="none", resample=False)
    colors = plt.get_cmap("tab10")
    for index, (x0, y0, x1, y1) in enumerate(boxes):
        axis.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor="none",
                edgecolor=colors(index),
                linewidth=3,
            )
        )
        axis.text(
            x0,
            max(0, y0 - 5),
            f"region_{index:02d}",
            color=colors(index),
            fontsize=12,
            weight="bold",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    axis.set_title(f"NDPI tissue-region candidates\n{source.name}")
    axis.axis("off")
    output = output_dir / f"{strip_known_suffixes(source)}_regions_overview.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def save_contact_sheet(
    source: Path,
    candidates: list[dict[str, Any]],
    samples: list[Sample],
    output_dir: Path,
) -> Path:
    columns = max(len(candidates), len(samples))
    fig, axes = plt.subplots(
        2,
        columns,
        figsize=(5 * columns, 10),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, candidate in zip(axes[0], candidates):
        axis.imshow(candidate["image"], interpolation="none", resample=False)
        axis.set_title(candidate["title"])
        axis.axis("off")
    for axis in axes[0, len(candidates) :]:
        axis.axis("off")

    for axis, sample in zip(axes[1], samples):
        plot_centroids(axis, sample)
    for axis in axes[1, len(samples) :]:
        axis.axis("off")

    fig.suptitle(
        "Match the H&E candidates (top) to Xenium centroid silhouettes (bottom)\n"
        + source.name,
        fontsize=14,
    )
    output = output_dir / f"{strip_known_suffixes(source)}_mapping_candidates.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def create_mapping_previews(
    root: Path | str,
    output_dir: Path | str | None = None,
    max_dimension: int = 2200,
) -> tuple[Path, Path]:
    root = Path(root).expanduser().resolve()
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else root / "mapping_previews"
    )
    destination.mkdir(parents=True, exist_ok=True)

    sources = find_breast_he_files(root)
    samples = discover_samples(root)
    if not sources:
        raise FileNotFoundError(f"No breast H&E source images found under {root}")
    if not samples:
        raise FileNotFoundError(f"No sample directories found under {root}")

    mapping_rows = []
    region_rows = []
    failures = []
    for source in tqdm(sources, desc="Mapping contact sheets", unit="slide"):
        grouped_samples = samples_for_source(source, samples)
        if not grouped_samples:
            failures.append((source.name, "No matching sample directories"))
            continue
        try:
            if source_kind(source) == "czi":
                candidates, _ = czi_scene_previews(source, max_dimension)
                candidate_type = "scene"
            else:
                preview, full_shape = ndpi_thumbnail(source, max_dimension)
                boxes = detect_tissue_regions(preview, len(grouped_samples))
                save_ndpi_overview(source, preview, boxes, destination)
                candidates = []
                scale_x = full_shape[0] / preview.shape[1]
                scale_y = full_shape[1] / preview.shape[0]
                for index, (x0, y0, x1, y1) in enumerate(boxes):
                    candidates.append(
                        {
                            "index": index,
                            "image": preview[y0:y1, x0:x1],
                            "title": (
                                f"region_{index:02d}\n"
                                f"full-res box ≈ ({round(x0 * scale_x):,}, "
                                f"{round(y0 * scale_y):,})–"
                                f"({round(x1 * scale_x):,}, {round(y1 * scale_y):,})"
                            ),
                            "bbox": (x0, y0, x1, y1),
                        }
                    )
                    region_rows.append(
                        {
                            "source_file": source.name,
                            "candidate_index": index,
                            "preview_x0": x0,
                            "preview_y0": y0,
                            "preview_x1": x1,
                            "preview_y1": y1,
                            "fullres_x0": round(x0 * scale_x),
                            "fullres_y0": round(y0 * scale_y),
                            "fullres_x1": round(x1 * scale_x),
                            "fullres_y1": round(y1 * scale_y),
                        }
                    )
                candidate_type = "region"

            contact_sheet = save_contact_sheet(
                source, candidates, grouped_samples, destination
            )
            print(f"Saved: {contact_sheet}")
            for sample in grouped_samples:
                mapping_rows.append(
                    {
                        "sample_folder": sample.directory.name,
                        "sample_label": sample.label,
                        "gsm": sample.gsm,
                        "source_file": source.name,
                        "source_type": source_kind(source).upper(),
                        "candidate_type": candidate_type,
                        "candidate_index": "",
                        "rotation_degrees": "",
                        "flip": "",
                        "verified": "no",
                        "notes": "",
                    }
                )
        except Exception as exc:
            failures.append((source.name, f"{type(exc).__name__}: {exc}"))
            print(f"FAILED: {source.name}: {exc}", file=sys.stderr)

    mapping_path = destination / "he_sample_mapping.csv"
    mapping_fields = [
        "sample_folder",
        "sample_label",
        "gsm",
        "source_file",
        "source_type",
        "candidate_type",
        "candidate_index",
        "rotation_degrees",
        "flip",
        "verified",
        "notes",
    ]
    with mapping_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=mapping_fields)
        writer.writeheader()
        writer.writerows(mapping_rows)

    regions_path = destination / "ndpi_candidate_regions.csv"
    region_fields = [
        "source_file",
        "candidate_index",
        "preview_x0",
        "preview_y0",
        "preview_x1",
        "preview_y1",
        "fullres_x0",
        "fullres_y0",
        "fullres_x1",
        "fullres_y1",
    ]
    with regions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=region_fields)
        writer.writeheader()
        writer.writerows(region_rows)

    if failures:
        print("\nFailures:", file=sys.stderr)
        for source_name, message in failures:
            print(f" - {source_name}: {message}", file=sys.stderr)
    print(f"Mapping template: {mapping_path}")
    print(f"NDPI region boxes: {regions_path}")
    return mapping_path, regions_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create H&E scene/region versus Xenium centroid contact sheets."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-dimension", type=int, default=2200)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        create_mapping_previews(
            root=args.root,
            output_dir=args.output_dir,
            max_dimension=args.max_dimension,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
