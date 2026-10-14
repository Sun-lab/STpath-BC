#!/usr/bin/env python3
"""Create H&E-to-Xenium affine registration candidates and overlay figures.

The script uses the candidate scene/region assignments in he_sample_mapping.csv.
It compares the H&E tissue silhouette with the Xenium cell-centroid footprint,
searches rotations and reflections, initializes from their tissue bounding
boxes, and refines the global affine transform against the complete masks.

This is a verification aid, not a final cell-accurate registration. Review the
saved transformed-image overlays before setting ``verified=yes``. Local tissue
deformation may require a later non-rigid registration step.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, optimize
from skimage.transform import AffineTransform, resize, warp
import tifffile
from tqdm.auto import tqdm

from xenium_pipeline.bilous_mapping import (
    Sample,
    czi_scene_previews,
    discover_samples,
    ndpi_thumbnail,
    read_cells,
)
from xenium_pipeline.bilous_extract import (
    ensure_uncompressed,
    require_common_dependencies,
    strip_known_suffixes,
    zarr_array_from_root,
)


_CZI_PREVIEW_CACHE: dict[tuple[str, int], list[dict[str, Any]]] = {}
_NDPI_PREVIEW_CACHE: dict[tuple[str, int], tuple[np.ndarray, tuple[int, int]]] = {}


def source_path(root: Path, source_name: str) -> Path:
    candidate = root / "images" / source_name
    if candidate.exists():
        return candidate
    if not source_name.lower().endswith(".gz"):
        compressed = candidate.with_name(candidate.name + ".gz")
        if compressed.exists():
            return compressed
    raise FileNotFoundError(candidate)


def extracted_he_path(
    root: Path,
    mapping_row: dict[str, str],
) -> Path | None:
    """Return the mapped native-resolution OME-TIFF when extraction exists."""
    output_dir = root / "images" / "full_resolution_he"
    stem = strip_known_suffixes(Path(mapping_row["source_file"]))
    if mapping_row["candidate_type"] == "scene":
        index = int(mapping_row["candidate_index"])
        candidate = output_dir / f"{stem}_scene_{index:02d}.ome.tif"
    else:
        candidate = output_dir / f"{stem}.ome.tif"
    return candidate if candidate.exists() else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_ome_pixel_sizes(ome_xml: str | None) -> tuple[float | None, float | None]:
    if not ome_xml:
        return None, None
    root = ET.fromstring(ome_xml)
    namespace = {"ome": root.tag.split("}")[0].strip("{")}
    pixels = root.find(".//ome:Pixels", namespace)
    if pixels is None:
        return None, None
    x = pixels.attrib.get("PhysicalSizeX")
    y = pixels.attrib.get("PhysicalSizeY")
    return (float(x) if x else None, float(y) if y else None)


def morphology_preview(
    sample: Sample,
    max_dimension: int,
) -> dict[str, Any]:
    """Load a downsampled max projection in Xenium morphology coordinates."""
    morphology_candidates = sorted(
        sample.directory.glob(f"{sample.gsm}_*_morphology_focus.ome.tif")
    ) + sorted(
        sample.directory.glob(f"{sample.gsm}_*_morphology_focus.ome.tif.gz")
    )
    if not morphology_candidates:
        morphology_candidates = sorted(
            sample.directory.glob(f"{sample.gsm}_*_morphology.ome.tif")
        ) + sorted(
            sample.directory.glob(f"{sample.gsm}_*_morphology.ome.tif.gz")
        )
    if not morphology_candidates:
        raise FileNotFoundError(
            f"No Xenium morphology image found for {sample.directory.name}"
        )

    _, _, tqdm_class = require_common_dependencies()
    path, _ = ensure_uncompressed(morphology_candidates[0], tqdm_class)
    with tifffile.TiffFile(path) as tif:
        spatial_series = [
            series
            for series in tif.series
            if "Y" in series.axes and "X" in series.axes
        ]
        if not spatial_series:
            raise ValueError(f"No spatial image series found in {path}")
        series = max(
            spatial_series,
            key=lambda item: (
                item.shape[item.axes.index("Y")]
                * item.shape[item.axes.index("X")]
            ),
        )
        ome_xml = tif.ome_metadata
        store = series.aszarr()
        try:
            import zarr

            array = zarr_array_from_root(zarr.open(store, mode="r"))
            axes = series.axes
            if len(axes) != array.ndim:
                raise ValueError(
                    f"Morphology axes {axes!r} do not match shape {array.shape}"
                )
            y_axis = axes.index("Y")
            x_axis = axes.index("X")
            full_height = int(array.shape[y_axis])
            full_width = int(array.shape[x_axis])
            stride = max(
                1,
                int(math.ceil(max(full_width, full_height) / max_dimension)),
            )
            selection = [slice(None)] * array.ndim
            selection[y_axis] = slice(None, None, stride)
            selection[x_axis] = slice(None, None, stride)
            image = np.asarray(array[tuple(selection)])
            image = np.moveaxis(image, (y_axis, x_axis), (-2, -1))
            image = image.reshape(-1, image.shape[-2], image.shape[-1]).max(axis=0)
        finally:
            if hasattr(store, "close"):
                store.close()

    finite = image[np.isfinite(image)]
    low, high = np.percentile(finite, [1.0, 99.8])
    display = np.clip(
        (image.astype(np.float32) - low) / max(high - low, 1e-12),
        0,
        1,
    )
    pixel_size_x, pixel_size_y = read_ome_pixel_sizes(ome_xml)
    if pixel_size_x is None or pixel_size_y is None:
        raise ValueError(f"Missing physical pixel size in {path}")
    return {
        "image": display,
        "full_width": full_width,
        "full_height": full_height,
        "scale_x": display.shape[1] / full_width,
        "scale_y": display.shape[0] / full_height,
        "pixel_size_x_um": pixel_size_x,
        "pixel_size_y_um": pixel_size_y,
        "path": path,
    }


def load_region_lookup(path: Path) -> dict[tuple[str, int], dict[str, int]]:
    lookup = {}
    for row in read_csv(path):
        key = (row["source_file"], int(row["candidate_index"]))
        lookup[key] = {
            name: int(row[name])
            for name in ("fullres_x0", "fullres_y0", "fullres_x1", "fullres_y1")
        }
    return lookup


def load_he_candidate(
    root: Path,
    mapping_row: dict[str, str],
    region_lookup: dict[tuple[str, int], dict[str, int]],
    max_dimension: int,
) -> dict[str, Any]:
    extracted_source = extracted_he_path(root, mapping_row)
    if extracted_source is not None:
        archive_source = root / "images" / mapping_row["source_file"]
        source = extracted_source
    else:
        archive_source = source_path(root, mapping_row["source_file"])
        source = archive_source
    index = int(mapping_row["candidate_index"])
    if extracted_source is not None:
        cache_key = (str(extracted_source.resolve()), max_dimension)
        if cache_key not in _NDPI_PREVIEW_CACHE:
            _NDPI_PREVIEW_CACHE[cache_key] = ndpi_thumbnail(
                extracted_source, max_dimension
            )
        full_preview, full_shape = _NDPI_PREVIEW_CACHE[cache_key]
        if mapping_row["candidate_type"] == "scene":
            image = full_preview
            full_width, full_height = full_shape
            origin_x = 0
            origin_y = 0
        else:
            key = (mapping_row["source_file"], index)
            if key not in region_lookup:
                raise KeyError(f"No NDPI region coordinates for {key}")
            box = region_lookup[key]
            scale_x = full_preview.shape[1] / full_shape[0]
            scale_y = full_preview.shape[0] / full_shape[1]
            px0 = max(0, int(math.floor(box["fullres_x0"] * scale_x)))
            py0 = max(0, int(math.floor(box["fullres_y0"] * scale_y)))
            px1 = min(
                full_preview.shape[1],
                int(math.ceil(box["fullres_x1"] * scale_x)),
            )
            py1 = min(
                full_preview.shape[0],
                int(math.ceil(box["fullres_y1"] * scale_y)),
            )
            image = full_preview[py0:py1, px0:px1]
            origin_x = box["fullres_x0"]
            origin_y = box["fullres_y0"]
            full_width = box["fullres_x1"] - box["fullres_x0"]
            full_height = box["fullres_y1"] - box["fullres_y0"]
    elif mapping_row["candidate_type"] == "scene":
        cache_key = (str(source.resolve()), max_dimension)
        if cache_key not in _CZI_PREVIEW_CACHE:
            _CZI_PREVIEW_CACHE[cache_key], _ = czi_scene_previews(
                source, max_dimension
            )
        candidates = _CZI_PREVIEW_CACHE[cache_key]
        matches = [candidate for candidate in candidates if candidate["index"] == index]
        if not matches:
            raise KeyError(f"Scene {index} not found in {source.name}")
        selected = matches[0]
        _, _, full_width, full_height = selected["bbox"]
        image = selected["image"]
        origin_x = 0
        origin_y = 0
    else:
        key = (mapping_row["source_file"], index)
        if key not in region_lookup:
            raise KeyError(f"No NDPI region coordinates for {key}")
        box = region_lookup[key]
        cache_key = (str(source.resolve()), max_dimension)
        if cache_key not in _NDPI_PREVIEW_CACHE:
            _NDPI_PREVIEW_CACHE[cache_key] = ndpi_thumbnail(
                source, max_dimension
            )
        full_preview, full_shape = _NDPI_PREVIEW_CACHE[cache_key]
        scale_x = full_preview.shape[1] / full_shape[0]
        scale_y = full_preview.shape[0] / full_shape[1]
        px0 = max(0, int(math.floor(box["fullres_x0"] * scale_x)))
        py0 = max(0, int(math.floor(box["fullres_y0"] * scale_y)))
        px1 = min(
            full_preview.shape[1],
            int(math.ceil(box["fullres_x1"] * scale_x)),
        )
        py1 = min(
            full_preview.shape[0],
            int(math.ceil(box["fullres_y1"] * scale_y)),
        )
        image = full_preview[py0:py1, px0:px1]
        origin_x = box["fullres_x0"]
        origin_y = box["fullres_y0"]
        full_width = box["fullres_x1"] - box["fullres_x0"]
        full_height = box["fullres_y1"] - box["fullres_y0"]

    return {
        "image": image,
        "full_width": int(full_width),
        "full_height": int(full_height),
        "origin_x_in_source": int(origin_x),
        "origin_y_in_source": int(origin_y),
        "scale_x": image.shape[1] / full_width,
        "scale_y": image.shape[0] / full_height,
        "source": source,
        "archive_source": archive_source,
        "used_extracted_ome_tiff": extracted_source is not None,
        "candidate_index": index,
        "candidate_type": mapping_row["candidate_type"],
    }


def remove_small_components(mask: np.ndarray, minimum_fraction: float) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if not count:
        return mask
    sizes = np.bincount(labels.ravel())
    threshold = max(10, int(mask.size * minimum_fraction))
    keep = sizes >= threshold
    keep[0] = False
    return keep[labels]


def he_tissue_mask(image: np.ndarray) -> np.ndarray:
    rgb = image[..., :3].astype(np.float32) / 255.0
    maximum = rgb.max(axis=-1)
    minimum = rgb.min(axis=-1)
    saturation = (maximum - minimum) / np.maximum(maximum, 1e-6)
    brightness = rgb.mean(axis=-1)
    mask = (
        ((saturation > 0.05) & (brightness < 0.985))
        | ((brightness < 0.88) & (brightness > 0.10))
    )
    mask = ndimage.binary_opening(mask, iterations=2)
    mask = ndimage.binary_closing(mask, iterations=5)
    mask = ndimage.binary_fill_holes(mask)
    mask = remove_small_components(mask, 0.001)
    return ndimage.binary_dilation(mask, iterations=2)


def centroid_mask(
    cells: Any,
    morphology: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_full = cells["x_centroid"].to_numpy() / morphology["pixel_size_x_um"]
    y_full = cells["y_centroid"].to_numpy() / morphology["pixel_size_y_um"]
    x_preview = x_full * morphology["scale_x"]
    y_preview = y_full * morphology["scale_y"]
    valid = (
        (x_preview >= 0)
        & (x_preview < morphology["image"].shape[1])
        & (y_preview >= 0)
        & (y_preview < morphology["image"].shape[0])
    )
    x_indices = np.floor(x_preview[valid]).astype(int)
    y_indices = np.floor(y_preview[valid]).astype(int)
    density = np.zeros(morphology["image"].shape, dtype=np.float32)
    np.add.at(density, (y_indices, x_indices), 1)
    density = ndimage.gaussian_filter(density, sigma=3.0)
    positive = density[density > 0]
    threshold = max(float(np.percentile(positive, 10)), density.max() * 0.015)
    mask = density >= threshold
    mask = ndimage.binary_closing(mask, iterations=4)
    mask = ndimage.binary_fill_holes(mask)
    mask = remove_small_components(mask, 0.0005)
    mask = ndimage.binary_dilation(mask, iterations=3)
    return mask, x_full[valid], y_full[valid]


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError("Empty registration mask")
    return int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)


def orientation_matrix(
    width: int,
    height: int,
    rotations_ccw: int,
    flip_horizontal: bool,
) -> tuple[np.ndarray, int, int]:
    k = rotations_ccw % 4
    if k == 0:
        matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
        oriented_width, oriented_height = width, height
    elif k == 1:
        matrix = np.array([[0, 1, 0], [-1, 0, width], [0, 0, 1]], float)
        oriented_width, oriented_height = height, width
    elif k == 2:
        matrix = np.array([[-1, 0, width], [0, -1, height], [0, 0, 1]], float)
        oriented_width, oriented_height = width, height
    else:
        matrix = np.array([[0, -1, height], [1, 0, 0], [0, 0, 1]], float)
        oriented_width, oriented_height = height, width

    if flip_horizontal:
        flip = np.array(
            [[-1, 0, oriented_width], [0, 1, 0], [0, 0, 1]],
            float,
        )
        matrix = flip @ matrix
    return matrix, oriented_width, oriented_height


def oriented_array(
    array: np.ndarray,
    rotations_ccw: int,
    flip_horizontal: bool,
) -> np.ndarray:
    result = np.rot90(array, rotations_ccw)
    if flip_horizontal:
        result = np.fliplr(result)
    return result


def dice_score(first: np.ndarray, second: np.ndarray) -> float:
    denominator = int(first.sum()) + int(second.sum())
    if denominator == 0:
        return 0.0
    return 2.0 * int(np.logical_and(first, second).sum()) / denominator


def transformed_mask_dice(
    he_mask: np.ndarray,
    source_xenium_mask: np.ndarray,
    warped_xenium_mask: np.ndarray,
    xenium_to_he_matrix: np.ndarray,
) -> tuple[float, float]:
    """Return overlap and coverage while penalizing tissue outside the canvas.

    A normal Dice score only sees the part of a warped image that remains in
    the output array. That can reward a transform that pushes difficult tissue
    outside the H&E canvas. The affine determinant estimates the complete
    transformed Xenium-mask area, including the portion outside the canvas.
    """
    expected_xenium_area = float(source_xenium_mask.sum()) * abs(
        float(np.linalg.det(xenium_to_he_matrix[:2, :2]))
    )
    observed_xenium_area = float(warped_xenium_mask.sum())
    complete_xenium_area = max(expected_xenium_area, observed_xenium_area)
    intersection = float(
        (
            he_mask.astype(np.float32)
            * warped_xenium_mask.astype(np.float32)
        ).sum()
    )
    denominator = float(he_mask.sum()) + complete_xenium_area
    score = 2.0 * intersection / max(denominator, 1e-12)
    coverage = observed_xenium_area / max(complete_xenium_area, 1e-12)
    return score, min(1.0, coverage)


def estimate_coarse_transform(
    he_mask: np.ndarray,
    xenium_mask: np.ndarray,
    allowed_rotations: tuple[int, ...] | None = None,
    allowed_flips: tuple[bool, ...] | None = None,
) -> dict[str, Any]:
    xenium_box = mask_bbox(xenium_mask)
    xx0, xy0, xx1, xy1 = xenium_box
    xenium_crop = xenium_mask[xy0:xy1, xx0:xx1]
    score_shape = (512, 512)
    xenium_normalized = resize(
        xenium_crop.astype(float),
        score_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ) > 0.5

    best = None
    rotations_to_try = allowed_rotations or (0, 1, 2, 3)
    flips_to_try = allowed_flips or (False, True)
    for rotations in rotations_to_try:
        for flip in flips_to_try:
            oriented = oriented_array(he_mask, rotations, flip)
            hx0, hy0, hx1, hy1 = mask_bbox(oriented)
            crop = oriented[hy0:hy1, hx0:hx1]
            normalized = resize(
                crop.astype(float),
                score_shape,
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            ) > 0.5
            score = dice_score(normalized, xenium_normalized)
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "rotations_ccw": rotations,
                    "flip_horizontal": flip,
                    "he_oriented_bbox": (hx0, hy0, hx1, hy1),
                    "xenium_bbox": xenium_box,
                }

    orientation, _, _ = orientation_matrix(
        he_mask.shape[1],
        he_mask.shape[0],
        best["rotations_ccw"],
        best["flip_horizontal"],
    )
    hx0, hy0, hx1, hy1 = best["he_oriented_bbox"]
    scale_x = (xx1 - xx0) / max(hx1 - hx0, 1)
    scale_y = (xy1 - xy0) / max(hy1 - hy0, 1)
    oriented_to_xenium = np.array(
        [
            [scale_x, 0, xx0 - scale_x * hx0],
            [0, scale_y, xy0 - scale_y * hy0],
            [0, 0, 1],
        ],
        float,
    )
    he_preview_to_xenium_preview = oriented_to_xenium @ orientation
    best["he_preview_to_xenium_preview"] = he_preview_to_xenium_preview
    best["xenium_preview_to_he_preview"] = np.linalg.inv(
        he_preview_to_xenium_preview
    )
    return best


def diagonal_matrix(x_scale: float, y_scale: float) -> np.ndarray:
    return np.array(
        [[x_scale, 0, 0], [0, y_scale, 0], [0, 0, 1]],
        float,
    )


def transform_points(matrix: np.ndarray, x: np.ndarray, y: np.ndarray):
    points = np.vstack([x, y, np.ones_like(x)])
    transformed = matrix @ points
    return transformed[0], transformed[1]


def parse_orientation_overrides(
    mapping_row: dict[str, str],
) -> tuple[tuple[int, ...] | None, tuple[bool, ...] | None]:
    """Read optional manual orientation constraints from the mapping table."""
    rotation_text = mapping_row.get("rotation_degrees", "").strip()
    if rotation_text:
        degrees = int(float(rotation_text)) % 360
        if degrees % 90:
            raise ValueError(
                "rotation_degrees must be one of 0, 90, 180, or 270"
            )
        rotations = (degrees // 90,)
    else:
        rotations = None

    flip_text = mapping_row.get("flip", "").strip().lower()
    if not flip_text:
        flips = None
    elif flip_text in {"1", "true", "yes", "y", "horizontal"}:
        flips = (True,)
    elif flip_text in {"0", "false", "no", "n", "none"}:
        flips = (False,)
    else:
        raise ValueError(
            "flip must be blank or one of yes/no, true/false, or 1/0"
        )
    return rotations, flips


def warp_xenium_to_he(
    image: np.ndarray,
    xenium_preview_to_he_preview: np.ndarray,
    output_shape: tuple[int, int],
    *,
    order: int,
) -> np.ndarray:
    """Warp a Xenium-preview image into the H&E-preview pixel canvas."""
    output_he_to_input_xenium = AffineTransform(
        matrix=np.linalg.inv(xenium_preview_to_he_preview)
    )
    return warp(
        image,
        inverse_map=output_he_to_input_xenium,
        output_shape=output_shape,
        order=order,
        preserve_range=True,
        mode="constant",
        cval=0,
    )


def _resize_mask_for_registration(
    mask: np.ndarray,
    max_dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    scale = min(1.0, max_dimension / max(mask.shape))
    output_shape = (
        max(2, int(round(mask.shape[0] * scale))),
        max(2, int(round(mask.shape[1] * scale))),
    )
    resized = resize(
        mask.astype(np.float32),
        output_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ) > 0.5
    preview_to_registration = diagonal_matrix(
        resized.shape[1] / mask.shape[1],
        resized.shape[0] / mask.shape[0],
    )
    return resized, preview_to_registration


def _affine_correction(
    parameters: np.ndarray,
    center_x: float,
    center_y: float,
) -> np.ndarray:
    """Build a small affine correction around the H&E tissue center."""
    tx, ty, angle, log_scale_x, log_scale_y, shear = parameters
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    rotation = np.array(
        [
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1],
        ],
        dtype=float,
    )
    scale_and_shear = np.array(
        [
            [math.exp(log_scale_x), math.tan(shear), 0],
            [0, math.exp(log_scale_y), 0],
            [0, 0, 1],
        ],
        dtype=float,
    )
    to_origin = np.array(
        [[1, 0, -center_x], [0, 1, -center_y], [0, 0, 1]],
        dtype=float,
    )
    from_origin = np.array(
        [
            [1, 0, center_x + tx],
            [0, 1, center_y + ty],
            [0, 0, 1],
        ],
        dtype=float,
    )
    return from_origin @ rotation @ scale_and_shear @ to_origin


def refine_affine_transform(
    he_mask: np.ndarray,
    xenium_mask: np.ndarray,
    initial_xenium_to_he: np.ndarray,
    max_dimension: int = 384,
) -> dict[str, Any]:
    """Refine the coarse transform by maximizing full-mask overlap.

    The optimization runs on small masks for speed, but the returned matrix is
    expressed in the original preview pixel coordinates.
    """
    he_small, he_preview_to_small = _resize_mask_for_registration(
        he_mask, max_dimension
    )
    xenium_small, xenium_preview_to_small = _resize_mask_for_registration(
        xenium_mask, max_dimension
    )
    initial_small = (
        he_preview_to_small
        @ initial_xenium_to_he
        @ np.linalg.inv(xenium_preview_to_small)
    )

    hy, hx = np.nonzero(he_small)
    center_x = float(np.median(hx))
    center_y = float(np.median(hy))
    he_float = he_small.astype(np.float32)

    def matrix_for(parameters: np.ndarray) -> np.ndarray:
        return (
            _affine_correction(parameters, center_x, center_y)
            @ initial_small
        )

    def objective(parameters: np.ndarray) -> float:
        candidate_matrix = matrix_for(parameters)
        warped_mask = warp_xenium_to_he(
            xenium_small.astype(np.float32),
            candidate_matrix,
            he_small.shape,
            order=1,
        )
        score, _ = transformed_mask_dice(
            he_float,
            xenium_small.astype(np.float32),
            warped_mask,
            candidate_matrix,
        )
        return 1.0 - score

    initial_parameters = np.zeros(6, dtype=float)
    initial_loss = objective(initial_parameters)
    translation_x = max(4.0, he_small.shape[1] * 0.15)
    translation_y = max(4.0, he_small.shape[0] * 0.15)
    bounds = [
        (-translation_x, translation_x),
        (-translation_y, translation_y),
        (-math.radians(15), math.radians(15)),
        (-math.log(1.25), math.log(1.25)),
        (-math.log(1.25), math.log(1.25)),
        (-math.radians(12), math.radians(12)),
    ]
    result = optimize.minimize(
        objective,
        initial_parameters,
        method="Powell",
        bounds=bounds,
        options={"maxiter": 70, "xtol": 2e-3, "ftol": 2e-4},
    )
    refined_parameters = result.x if result.fun < initial_loss else initial_parameters
    refined_small = matrix_for(refined_parameters)
    refined_preview = (
        np.linalg.inv(he_preview_to_small)
        @ refined_small
        @ xenium_preview_to_small
    )
    return {
        "xenium_preview_to_he_preview": refined_preview,
        "initial_soft_dice": 1.0 - initial_loss,
        "refined_soft_dice": 1.0 - min(float(result.fun), initial_loss),
        "parameters": refined_parameters.tolist(),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }


def verify_sample(
    root: Path,
    sample: Sample,
    mapping_row: dict[str, str],
    region_lookup: dict[tuple[str, int], dict[str, int]],
    output_dir: Path,
    max_dimension: int,
    random_seed: int,
    refine_alignment: bool,
    registration_max_dimension: int,
) -> dict[str, Any]:
    cells = read_cells(sample)
    morphology = morphology_preview(sample, max_dimension)
    he = load_he_candidate(
        root, mapping_row, region_lookup, max_dimension=max_dimension
    )
    he_mask = he_tissue_mask(he["image"])
    x_mask, cell_x_full, cell_y_full = centroid_mask(cells, morphology)
    allowed_rotations, allowed_flips = parse_orientation_overrides(mapping_row)
    coarse = estimate_coarse_transform(
        he_mask,
        x_mask,
        allowed_rotations=allowed_rotations,
        allowed_flips=allowed_flips,
    )
    initial_xenium_to_he = coarse["xenium_preview_to_he_preview"]
    if refine_alignment:
        refinement = refine_affine_transform(
            he_mask,
            x_mask,
            initial_xenium_to_he,
            max_dimension=registration_max_dimension,
        )
        xenium_preview_to_he_preview = refinement[
            "xenium_preview_to_he_preview"
        ]
    else:
        xenium_preview_to_he_preview = initial_xenium_to_he
        initial_warp = warp_xenium_to_he(
            x_mask.astype(np.float32),
            initial_xenium_to_he,
            he_mask.shape,
            order=0,
        ) > 0.5
        initial_score, _ = transformed_mask_dice(
            he_mask,
            x_mask,
            initial_warp,
            initial_xenium_to_he,
        )
        refinement = {
            "xenium_preview_to_he_preview": initial_xenium_to_he,
            "initial_soft_dice": initial_score,
            "refined_soft_dice": initial_score,
            "parameters": [0.0] * 6,
            "optimizer_success": True,
            "optimizer_message": "Affine refinement disabled",
        }
    he_preview_to_xenium_preview = np.linalg.inv(
        xenium_preview_to_he_preview
    )

    xenium_full_to_preview = diagonal_matrix(
        morphology["scale_x"], morphology["scale_y"]
    )
    he_preview_to_full = diagonal_matrix(
        1.0 / he["scale_x"], 1.0 / he["scale_y"]
    )
    xenium_full_to_he_local_full = (
        he_preview_to_full
        @ xenium_preview_to_he_preview
        @ xenium_full_to_preview
    )
    xenium_um_to_pixel = diagonal_matrix(
        1.0 / morphology["pixel_size_x_um"],
        1.0 / morphology["pixel_size_y_um"],
    )
    xenium_um_to_he_local_full = (
        xenium_full_to_he_local_full @ xenium_um_to_pixel
    )

    he_x_full, he_y_full = transform_points(
        xenium_full_to_he_local_full, cell_x_full, cell_y_full
    )
    he_x_preview = he_x_full * he["scale_x"]
    he_y_preview = he_y_full * he["scale_y"]
    inside = (
        (he_x_preview >= 0)
        & (he_x_preview < he["image"].shape[1])
        & (he_y_preview >= 0)
        & (he_y_preview < he["image"].shape[0])
    )

    rng = np.random.default_rng(random_seed)
    valid_indices = np.flatnonzero(inside)
    shown = rng.choice(
        valid_indices,
        size=min(40_000, len(valid_indices)),
        replace=False,
    )
    shown_xenium = rng.choice(
        len(cell_x_full),
        size=min(40_000, len(cell_x_full)),
        replace=False,
    )

    xenium_on_he = warp_xenium_to_he(
        x_mask.astype(float),
        xenium_preview_to_he_preview,
        he_mask.shape,
        order=0,
    ) > 0.5
    morphology_on_he = warp_xenium_to_he(
        morphology["image"].astype(np.float32),
        xenium_preview_to_he_preview,
        he_mask.shape,
        order=1,
    )
    final_dice, xenium_mask_fraction_inside = transformed_mask_dice(
        he_mask,
        x_mask,
        xenium_on_he,
        xenium_preview_to_he_preview,
    )

    cyan_morphology = np.zeros((*he_mask.shape, 4), dtype=np.float32)
    cyan_morphology[..., 1] = 1.0
    cyan_morphology[..., 2] = 1.0
    tissue_values = morphology_on_he[xenium_on_he]
    if tissue_values.size:
        low, high = np.percentile(tissue_values, [2.0, 99.5])
        morphology_scaled = np.clip(
            (morphology_on_he - low) / max(high - low, 1e-12),
            0,
            1,
        )
    else:
        morphology_scaled = morphology_on_he
    cyan_morphology[..., 3] = np.where(
        xenium_on_he,
        0.10 + 0.42 * np.sqrt(morphology_scaled),
        0.0,
    )

    mask_overlay = np.ones((*he_mask.shape, 3), dtype=np.float32)
    mask_overlay[he_mask, 0] = 1.0
    mask_overlay[he_mask, 1:] *= 0.35
    mask_overlay[xenium_on_he, 1] = 1.0
    mask_overlay[xenium_on_he, 2] = 1.0
    mask_overlay[xenium_on_he, 0] *= 0.25

    fig, axes_grid = plt.subplots(
        2,
        3,
        figsize=(18, 12),
        constrained_layout=True,
    )
    axes = axes_grid.ravel()
    axes[0].imshow(morphology["image"], cmap="gray", interpolation="none")
    axes[0].scatter(
        cell_x_full[shown_xenium] * morphology["scale_x"],
        cell_y_full[shown_xenium] * morphology["scale_y"],
        s=0.3,
        color="deepskyblue",
        alpha=0.45,
        linewidths=0,
        rasterized=True,
    )
    axes[0].set_title("Xenium morphology + centroids")
    axes[0].axis("off")

    axes[1].imshow(he["image"], interpolation="none", resample=False)
    axes[1].set_title(
        f"H&E {mapping_row['candidate_type']}_{he['candidate_index']:02d}"
    )
    axes[1].axis("off")

    axes[2].imshow(
        morphology_on_he,
        cmap="gray",
        interpolation="none",
        vmin=0,
        vmax=1,
    )
    axes[2].scatter(
        he_x_preview[shown],
        he_y_preview[shown],
        s=0.35,
        color="deepskyblue",
        alpha=0.5,
        linewidths=0,
        rasterized=True,
    )
    axes[2].set_xlim(0, he["image"].shape[1])
    axes[2].set_ylim(he["image"].shape[0], 0)
    axes[2].set_title(
        "Transformed Xenium morphology + centroids\n"
        "shown in the H&E pixel canvas"
    )
    axes[2].axis("off")

    axes[3].imshow(he["image"], interpolation="none", resample=False)
    axes[3].scatter(
        he_x_preview[shown],
        he_y_preview[shown],
        s=0.35,
        color="deepskyblue",
        alpha=0.55,
        linewidths=0,
        rasterized=True,
    )
    axes[3].set_xlim(0, he["image"].shape[1])
    axes[3].set_ylim(he["image"].shape[0], 0)
    axes[3].set_title(
        "H&E + transformed Xenium centroids\n"
        f"{inside.mean():.1%} inside H&E canvas"
    )
    axes[3].axis("off")

    axes[4].imshow(he["image"], interpolation="none", resample=False)
    axes[4].imshow(cyan_morphology, interpolation="none")
    axes[4].scatter(
        he_x_preview[shown],
        he_y_preview[shown],
        s=0.25,
        color="yellow",
        alpha=0.45,
        linewidths=0,
        rasterized=True,
    )
    axes[4].set_xlim(0, he["image"].shape[1])
    axes[4].set_ylim(he["image"].shape[0], 0)
    axes[4].set_title(
        "Actual transformed-image overlay\n"
        "H&E + cyan Xenium morphology + yellow centroids"
    )
    axes[4].axis("off")

    axes[5].imshow(mask_overlay, interpolation="none")
    axes[5].set_title(
        "Refined silhouette overlap\n"
        f"Dice={final_dice:.3f}; mask inside={xenium_mask_fraction_inside:.1%}\n"
        "red=H&E, cyan=Xenium"
    )
    axes[5].axis("off")
    fig.suptitle(
        f"{sample.directory.name}: candidate alignment for visual review",
        fontsize=15,
    )

    overlay_path = output_dir / f"{sample.directory.name}_alignment_review.png"
    fig.savefig(overlay_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    transform_path = output_dir / f"{sample.directory.name}_alignment_candidate.json"
    transform_record = {
        "sample_folder": sample.directory.name,
        "sample_label": sample.label,
        "gsm": sample.gsm,
        "source_file": mapping_row["source_file"],
        "candidate_type": mapping_row["candidate_type"],
        "candidate_index": he["candidate_index"],
        "status": "needs_visual_review",
        "registration_method": (
            "coarse_bbox_plus_affine_mask_refinement"
            if refine_alignment
            else "coarse_bbox_only"
        ),
        "coarse_normalized_silhouette_dice": coarse["score"],
        "initial_full_canvas_soft_dice": refinement["initial_soft_dice"],
        "silhouette_dice": final_dice,
        "xenium_mask_fraction_inside_he_canvas": xenium_mask_fraction_inside,
        "affine_refinement_parameters": refinement["parameters"],
        "optimizer_success": refinement["optimizer_success"],
        "optimizer_message": refinement["optimizer_message"],
        "rotation_degrees_ccw_applied_to_he": coarse["rotations_ccw"] * 90,
        "flip_he_horizontally_after_rotation": coarse["flip_horizontal"],
        "cell_fraction_inside_he_canvas": float(inside.mean()),
        "xenium_morphology_full_shape_yx": [
            morphology["full_height"],
            morphology["full_width"],
        ],
        "xenium_pixel_size_um_xy": [
            morphology["pixel_size_x_um"],
            morphology["pixel_size_y_um"],
        ],
        "he_candidate_full_shape_yx": [he["full_height"], he["full_width"]],
        "he_candidate_origin_xy_in_source": [
            he["origin_x_in_source"],
            he["origin_y_in_source"],
        ],
        "matrix_xenium_preview_pixel_to_he_preview_pixel": (
            xenium_preview_to_he_preview.tolist()
        ),
        "matrix_he_preview_pixel_to_xenium_preview_pixel": (
            he_preview_to_xenium_preview.tolist()
        ),
        "matrix_xenium_full_pixel_to_he_candidate_local_full_pixel": (
            xenium_full_to_he_local_full.tolist()
        ),
        "matrix_xenium_um_to_he_candidate_local_full_pixel": (
            xenium_um_to_he_local_full.tolist()
        ),
        "overlay_figure": str(overlay_path),
        "warning": (
            "Global affine registration candidate only. Do not use for final "
            "cell-level labels until the transformed morphology overlay is "
            "reviewed and local/non-rigid alignment is assessed."
        ),
    }
    with transform_path.open("w", encoding="utf-8") as handle:
        json.dump(transform_record, handle, indent=2)
        handle.write("\n")

    return {
        "sample_folder": sample.directory.name,
        "sample_label": sample.label,
        "gsm": sample.gsm,
        "candidate_type": mapping_row["candidate_type"],
        "candidate_index": he["candidate_index"],
        "rotation_degrees_ccw": coarse["rotations_ccw"] * 90,
        "flip_horizontal": coarse["flip_horizontal"],
        "silhouette_dice": final_dice,
        "xenium_mask_fraction_inside_he_canvas": xenium_mask_fraction_inside,
        "cell_fraction_inside_he_canvas": float(inside.mean()),
        "status": "needs_visual_review",
        "overlay_figure": str(overlay_path),
        "transform_json": str(transform_path),
    }


def verify_all_alignments(
    root: Path | str,
    mapping_csv: Path | str | None = None,
    ndpi_regions_csv: Path | str | None = None,
    output_dir: Path | str | None = None,
    max_dimension: int = 1600,
    random_seed: int = 1,
    refine_alignment: bool = True,
    registration_max_dimension: int = 384,
) -> Path:
    root = Path(root).expanduser().resolve()
    mapping_path = (
        Path(mapping_csv).expanduser().resolve()
        if mapping_csv is not None
        else root / "mapping_previews" / "he_sample_mapping.csv"
    )
    regions_path = (
        Path(ndpi_regions_csv).expanduser().resolve()
        if ndpi_regions_csv is not None
        else root / "mapping_previews" / "ndpi_candidate_regions.csv"
    )
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else root / "alignment_reviews"
    )
    destination.mkdir(parents=True, exist_ok=True)

    samples = {sample.directory.name: sample for sample in discover_samples(root)}
    mapping_rows = read_csv(mapping_path)
    region_lookup = load_region_lookup(regions_path)
    selected_rows = [row for row in mapping_rows if row["candidate_index"].strip()]
    if not selected_rows:
        raise ValueError(f"No candidate_index values found in {mapping_path}")

    summary = []
    failures = []
    for row in tqdm(selected_rows, desc="Alignment reviews", unit="sample"):
        sample = samples.get(row["sample_folder"])
        if sample is None:
            failures.append((row["sample_folder"], "Sample directory not found"))
            continue
        try:
            record = verify_sample(
                root=root,
                sample=sample,
                mapping_row=row,
                region_lookup=region_lookup,
                output_dir=destination,
                max_dimension=max_dimension,
                random_seed=random_seed,
                refine_alignment=refine_alignment,
                registration_max_dimension=registration_max_dimension,
            )
            summary.append(record)
            print(f"Saved: {record['overlay_figure']}")
        except Exception as exc:
            failures.append((sample.directory.name, f"{type(exc).__name__}: {exc}"))
            print(
                f"FAILED: {sample.directory.name}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    summary_path = destination / "alignment_review_summary.csv"
    fields = [
        "sample_folder",
        "sample_label",
        "gsm",
        "candidate_type",
        "candidate_index",
        "rotation_degrees_ccw",
        "flip_horizontal",
        "silhouette_dice",
        "xenium_mask_fraction_inside_he_canvas",
        "cell_fraction_inside_he_canvas",
        "status",
        "overlay_figure",
        "transform_json",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    if failures:
        failures_path = destination / "alignment_review_failures.csv"
        with failures_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_folder", "error"])
            writer.writerows(failures)
        print(f"Failures: {failures_path}", file=sys.stderr)
    print(f"Alignment summary: {summary_path}")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create refined affine H&E/Xenium registration reviews."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--mapping-csv", type=Path, default=None)
    parser.add_argument("--ndpi-regions-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-dimension", type=int, default=1600)
    parser.add_argument(
        "--registration-max-dimension",
        type=int,
        default=384,
        help="Largest mask dimension used by affine refinement (default: 384)",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Disable affine mask refinement and keep the coarse bounding-box fit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        verify_all_alignments(
            root=args.root,
            mapping_csv=args.mapping_csv,
            ndpi_regions_csv=args.ndpi_regions_csv,
            output_dir=args.output_dir,
            max_dimension=args.max_dimension,
            refine_alignment=not args.no_refine,
            registration_max_dimension=args.registration_max_dimension,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
