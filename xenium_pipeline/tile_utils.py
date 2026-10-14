"""Small shared helpers for affine transforms, image arrays, and tile colors."""

from __future__ import annotations

import math
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


def apply_affine(xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 3 × 3 homogeneous affine transform to an ``(n, 2)`` array."""
    xy = np.asarray(xy, dtype=float)
    homogeneous = np.column_stack([xy, np.ones(len(xy), dtype=float)])
    transformed = homogeneous @ np.asarray(matrix, dtype=float).T
    return transformed[:, :2] / transformed[:, 2, None]


def get_hw(shape: tuple[int, ...]) -> tuple[int, int]:
    """Return height and width for common grayscale/RGB image array layouts."""
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    if len(shape) == 3 and shape[-1] in (3, 4):
        return int(shape[0]), int(shape[1])
    if len(shape) == 3 and shape[0] in (3, 4):
        return int(shape[1]), int(shape[2])
    raise ValueError(f"Unexpected image shape: {shape}")


def ensure_rgb(array: np.ndarray) -> np.ndarray:
    """Convert common grayscale/RGB/RGBA arrays into uint8 RGB."""
    array = np.asarray(array)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    elif array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"Expected RGB/RGBA image, received shape {array.shape}")

    array = array[..., :3]
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.integer):
        maximum = np.iinfo(array.dtype).max
        return np.round(array.astype(np.float32) * (255.0 / maximum)).astype(np.uint8)

    finite = array[np.isfinite(array)]
    if not finite.size:
        return np.zeros(array.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [0.1, 99.9])
    scaled = np.clip((array - low) / max(high - low, 1e-12), 0, 1)
    return np.round(scaled * 255).astype(np.uint8)


def resolve_zarr_array(root: Any) -> Any:
    """Find the largest image-like array inside a TIFF-backed Zarr store."""
    if hasattr(root, "shape"):
        return root
    arrays: list[Any] = []

    def visit(node: Any) -> None:
        if hasattr(node, "shape"):
            arrays.append(node)
        elif hasattr(node, "keys"):
            for key in node.keys():
                visit(node[key])

    if hasattr(root, "keys") and "0" in root:
        visit(root["0"])
    if not arrays:
        visit(root)
    if not arrays:
        raise KeyError("No image array found in the TIFF-backed Zarr store")
    return max(arrays, key=lambda array: math.prod(get_hw(tuple(array.shape))))


def extract_rgb_tile(array: Any, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Read a rectangular tile from a grayscale/RGB/RGBA array as uint8 RGB."""
    shape = tuple(array.shape)
    if len(shape) == 2:
        tile = np.asarray(array[y0:y1, x0:x1])
    elif len(shape) == 3 and shape[-1] in (3, 4):
        tile = np.asarray(array[y0:y1, x0:x1, :])
    elif len(shape) == 3 and shape[0] in (3, 4):
        tile = np.asarray(array[:, y0:y1, x0:x1])
    else:
        raise ValueError(f"Unexpected image array shape: {shape}")
    return ensure_rgb(tile)


def create_color_map(labels: Iterable[str]) -> dict[str, tuple[int, int, int]]:
    """Create a stable tab20 color map and reserve gray for missing labels."""
    values = sorted(set(map(str, labels)))
    cmap = plt.get_cmap("tab20", max(len(values), 1))
    colors = {
        label: tuple(int(channel * 255) for channel in cmap(index)[:3])
        for index, label in enumerate(values)
    }
    for missing_label in ("Unassigned", "Unannotated"):
        if missing_label in colors:
            colors[missing_label] = (140, 140, 140)
    return colors


def xenium_explorer_imagealignment_matrix(xenium_to_he: np.ndarray) -> np.ndarray:
    """Return the Explorer-style H&E-pixel to Xenium-pixel matrix."""
    matrix = np.asarray(xenium_to_he, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 affine matrix, got {matrix.shape}")
    he_to_xenium = np.linalg.inv(matrix)
    if abs(float(he_to_xenium[2, 2])) < 1e-12:
        raise ValueError("Inverted affine matrix has invalid homogeneous scale")
    he_to_xenium = he_to_xenium / he_to_xenium[2, 2]
    if not np.allclose(he_to_xenium[2], [0, 0, 1], atol=1e-9):
        raise ValueError("Inverted transform is not a 3x3 affine matrix")
    he_to_xenium[2] = [0.0, 0.0, 1.0]
    return he_to_xenium
