#!/usr/bin/env python3
"""Interactively refine a Xenium-to-H&E transform with paired landmarks.

This module is intended for use from a Jupyter notebook. By default it starts
directly from the mapped native Xenium and H&E previews, asks the user to click
corresponding internal anatomical landmarks, estimates an affine transform,
and saves the transform plus a visual review figure. It can optionally refine
an existing alignment-candidate JSON.
"""

from __future__ import annotations

import base64
import csv
import gzip
from io import BytesIO
import json
import math
from pathlib import Path
import sys
from typing import Any
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw
from skimage.transform import AffineTransform
from tqdm.auto import tqdm

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xenium_pipeline.tile_utils import xenium_explorer_imagealignment_matrix

from xenium_pipeline.bilous_mapping import discover_samples, read_cells
from xenium_pipeline.bilous_image_alignment import (
    centroid_mask,
    diagonal_matrix,
    load_he_candidate,
    load_region_lookup,
    morphology_preview,
    transform_points,
    warp_xenium_to_he,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _find_one(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    matches = [row for row in rows if row.get(key) == value]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one row with {key}={value!r}; found {len(matches)}"
        )
    return matches[0]


def _normalise_display(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if not finite.size:
        return np.zeros_like(image, dtype=np.float32)
    low, high = np.percentile(finite, [1.0, 99.7])
    return np.clip(
        (image.astype(np.float32) - low) / max(high - low, 1e-12),
        0,
        1,
    )




def write_xenium_explorer_imagealignment_csv(
    path: str | Path,
    xenium_to_he: np.ndarray,
) -> np.ndarray:
    """Write a Xenium Explorer-compatible 3x3 imagealignment CSV."""
    imagealignment = xenium_explorer_imagealignment_matrix(xenium_to_he)
    np.savetxt(Path(path), imagealignment, delimiter=",", fmt="%.15g")
    return imagealignment


def collect_paired_landmarks(
    xenium_image: np.ndarray,
    he_image: np.ndarray,
    n_points: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect alternating Xenium/H&E clicks from a side-by-side figure."""
    if n_points < 3:
        raise ValueError("At least three landmark pairs are required")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 8),
        constrained_layout=True,
    )
    axes[0].imshow(xenium_image, cmap="gray", interpolation="none")
    axes[0].set_title("Xenium morphology")
    axes[1].imshow(he_image, interpolation="none", resample=False)
    axes[1].set_title("H&E")
    axes[0].set_xlim(0, xenium_image.shape[1])
    axes[0].set_ylim(xenium_image.shape[0], 0)
    axes[1].set_xlim(0, he_image.shape[1])
    axes[1].set_ylim(he_image.shape[0], 0)
    axes[0].axis("off")
    axes[1].axis("off")

    fig.suptitle(
        "Alternate clicks: Xenium landmark 1, matching H&E landmark 1, "
        "then pair 2, etc.\n"
        "Use internal structures spread across the tissue; middle-click "
        "removes the previous click.",
        fontsize=13,
    )
    plt.show(block=False)
    print(
        f"Please make {2 * n_points} alternating clicks: "
        "left Xenium, right H&E."
    )
    clicks = np.asarray(
        fig.ginput(2 * n_points, timeout=0, show_clicks=True),
        dtype=float,
    )
    plt.close(fig)
    if clicks.shape != (2 * n_points, 2):
        raise RuntimeError(
            f"Expected {2 * n_points} clicks, received {len(clicks)}"
        )
    return clicks[0::2], clicks[1::2]


def _cyan_rgba(image: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
    scaled = _normalise_display(image)
    rgba = np.zeros((*image.shape, 4), dtype=np.float32)
    rgba[..., 1] = 1.0
    rgba[..., 2] = 1.0
    rgba[..., 3] = np.where(
        tissue_mask,
        0.08 + 0.40 * np.sqrt(scaled),
        0.0,
    )
    return rgba


def _png_data_url(image: np.ndarray) -> str:
    """Encode a preview array as an embedded PNG for the HTML picker."""
    if image.ndim == 2:
        payload = np.round(_normalise_display(image) * 255).astype(np.uint8)
        pil_image = Image.fromarray(payload, mode="L")
    else:
        payload = image[..., :3]
        if payload.dtype != np.uint8:
            finite = payload[np.isfinite(payload)]
            if finite.size and finite.max() <= 1.0:
                payload = payload * 255
            payload = np.clip(payload, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(payload, mode="RGB")
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG", optimize=False)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _read_parquet_maybe_gzip(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.name.endswith(".parquet.gz"):
        with gzip.open(path, "rb") as handle:
            return pd.read_parquet(BytesIO(handle.read()), columns=columns)
    return pd.read_parquet(path, columns=columns)


def _find_sample_file(sample: Any, suffix: str) -> Path | None:
    uncompressed = sorted(sample.directory.glob(f"{sample.gsm}_*_{suffix}"))
    compressed = sorted(sample.directory.glob(f"{sample.gsm}_*_{suffix}.gz"))
    return next(iter(uncompressed + compressed), None)


def _read_cells_for_overlay(sample: Any) -> pd.DataFrame:
    path = _find_sample_file(sample, "cells.parquet")
    if path is None:
        raise FileNotFoundError(f"No cells parquet file found for {sample.directory.name}")
    cells = _read_parquet_maybe_gzip(path)
    cells = cells.rename(
        columns={
            "barcode": "cell_id",
            "cell_centroid_x": "x_centroid",
            "cell_centroid_y": "y_centroid",
            "nucleus_centroid_x": "nucleus_x_centroid",
            "nucleus_centroid_y": "nucleus_y_centroid",
        }
    )
    if "cell_id" not in cells:
        cells.insert(0, "cell_id", np.arange(len(cells)).astype(str))
    cells["cell_id"] = cells["cell_id"].astype(str)
    return cells


def _read_boundaries_for_overlay(
    sample: Any,
    boundary_kind: str,
) -> dict[str, np.ndarray]:
    path = _find_sample_file(sample, f"{boundary_kind}_boundaries.parquet")
    if path is None:
        return {}
    boundaries = _read_parquet_maybe_gzip(path)
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
            f"[{sample.directory.name}] {boundary_kind} boundary columns not recognized: "
            f"{list(boundaries.columns)}; skipping {boundary_kind} boundaries",
            flush=True,
        )
        return {}
    boundaries["cell_id"] = boundaries["cell_id"].astype(str)
    order_column = next(
        (
            name
            for name in (
                "vertex_index",
                "vertex_id",
                "boundary_index",
                "point_index",
                "index",
            )
            if name in boundaries.columns
        ),
        None,
    )
    if order_column is not None:
        boundaries = boundaries.sort_values(["cell_id", order_column])
    return {
        str(cell_id): group[["vertex_x", "vertex_y"]].to_numpy(dtype=float)
        for cell_id, group in boundaries.groupby("cell_id", sort=False)
        if len(group) >= 3
    }


def _xy_um_to_preview_pixels(
    xy_um: np.ndarray,
    morphology: dict[str, Any],
) -> np.ndarray:
    return np.column_stack(
        [
            xy_um[:, 0] / morphology["pixel_size_x_um"] * morphology["scale_x"],
            xy_um[:, 1] / morphology["pixel_size_y_um"] * morphology["scale_y"],
        ]
    )


def _draw_boundary_overlay(
    overlay: Image.Image,
    polygons_by_cell_id: dict[str, np.ndarray],
    selected_cell_ids: set[str],
    morphology: dict[str, Any],
    *,
    color: tuple[int, int, int, int],
    width: int,
) -> None:
    draw = ImageDraw.Draw(overlay, "RGBA")
    for cell_id in selected_cell_ids:
        polygon = polygons_by_cell_id.get(cell_id)
        if polygon is None:
            continue
        preview = _xy_um_to_preview_pixels(polygon, morphology)
        finite = np.isfinite(preview).all(axis=1)
        preview = preview[finite]
        if len(preview) < 3:
            continue
        points = [(float(x), float(y)) for x, y in preview]
        draw.line(points + [points[0]], fill=color, width=width)


def _morphology_with_cell_overlays(
    sample: Any,
    morphology: dict[str, Any],
    maximum_cells: int = 120_000,
    maximum_boundary_cells: int = 4_000,
) -> np.ndarray:
    """Overlay native nucleus centers and cell/nucleus boundaries on morphology."""
    gray = _normalise_display(morphology["image"])
    rgb = np.repeat((gray * 0.68)[..., None], 3, axis=2)
    cells = _read_cells_for_overlay(sample)
    marker_x_column = (
        "nucleus_x_centroid"
        if {"nucleus_x_centroid", "nucleus_y_centroid"}.issubset(cells.columns)
        else "x_centroid"
    )
    marker_y_column = "nucleus_y_centroid" if marker_x_column == "nucleus_x_centroid" else "y_centroid"
    marker_xy = cells[[marker_x_column, marker_y_column]].to_numpy(dtype=float)
    fallback_xy = cells[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
    missing_marker = ~np.isfinite(marker_xy).all(axis=1)
    marker_xy[missing_marker] = fallback_xy[missing_marker]
    preview_xy = _xy_um_to_preview_pixels(marker_xy, morphology)
    x = preview_xy[:, 0]
    y = preview_xy[:, 1]
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0)
        & (x < rgb.shape[1])
        & (y >= 0)
        & (y < rgb.shape[0])
    )
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) > maximum_cells:
        keep = np.linspace(0, len(valid_indices) - 1, maximum_cells).astype(int)
        valid_indices = valid_indices[keep]
    canvas = Image.fromarray(np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8), mode="RGB")
    boundary_overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    boundary_indices = valid_indices
    if len(boundary_indices) > maximum_boundary_cells:
        keep = np.linspace(
            0,
            len(boundary_indices) - 1,
            maximum_boundary_cells,
        ).astype(int)
        boundary_indices = boundary_indices[keep]
    boundary_cell_ids = set(cells.iloc[boundary_indices]["cell_id"].astype(str))
    _draw_boundary_overlay(
        boundary_overlay,
        _read_boundaries_for_overlay(sample, "cell"),
        boundary_cell_ids,
        morphology,
        color=(0, 210, 255, 90),
        width=1,
    )
    _draw_boundary_overlay(
        boundary_overlay,
        _read_boundaries_for_overlay(sample, "nucleus"),
        boundary_cell_ids,
        morphology,
        color=(255, 0, 255, 105),
        width=1,
    )
    canvas = Image.alpha_composite(canvas.convert("RGBA"), boundary_overlay).convert("RGB")
    rgb8 = np.asarray(canvas).copy()
    x_marker = np.floor(x[valid_indices]).astype(int)
    y_marker = np.floor(y[valid_indices]).astype(int)
    for offset_y, offset_x in ((0, 0), (0, 1), (1, 0), (-1, 0), (0, -1)):
        xx = np.clip(x_marker + offset_x, 0, rgb8.shape[1] - 1)
        yy = np.clip(y_marker + offset_y, 0, rgb8.shape[0] - 1)
        rgb8[yy, xx, 0] = 255
        rgb8[yy, xx, 1] = 230
        rgb8[yy, xx, 2] = 0
    return rgb8


def show_landmark_selector(
    root: Path | str,
    sample_folder: str,
    *,
    initial_review_dir: Path | str | None = None,
    mapping_csv: Path | str | None = None,
    ndpi_regions_csv: Path | str | None = None,
    n_points: int = 8,
    max_dimension: int = 1000,
    use_morphology_mip: bool = True,
) -> Any:
    """Display a pure-HTML paired-landmark picker in JupyterLab.

    The picker requires no Matplotlib widget extension. Copy its JSON output
    and pass it to ``manual_landmark_align(..., landmarks=that_json)``.
    """
    from IPython.display import HTML, display

    print(f"Preparing landmark selector for {sample_folder}...", flush=True)
    root = Path(root).expanduser().resolve()
    initial_dir = (
        Path(initial_review_dir).expanduser().resolve()
        if initial_review_dir is not None
        else None
    )
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
    samples = {sample.directory.name: sample for sample in discover_samples(root)}
    if sample_folder not in samples:
        raise KeyError(f"Sample folder not found: {sample_folder}")
    sample = samples[sample_folder]
    mapping_row = _find_one(
        _read_csv(mapping_path), "sample_folder", sample_folder
    )
    print("Loading Xenium morphology preview...", flush=True)
    morphology = morphology_preview(sample, max_dimension)
    print("Loading H&E preview...", flush=True)
    he = load_he_candidate(
        root,
        mapping_row,
        load_region_lookup(regions_path),
        max_dimension=max_dimension,
    )
    print(f"Using H&E image: {he['source']}", flush=True)
    if initial_dir is None:
        print("Adding native Xenium nuclei and boundaries...", flush=True)
        source_image = _morphology_with_cell_overlays(sample, morphology)
        source_coordinate_space = "native_xenium_preview"
        source_title = "Native Xenium morphology + cell/nucleus overlays"
    else:
        initial_path = initial_dir / f"{sample_folder}_alignment_candidate.json"
        with initial_path.open(encoding="utf-8") as handle:
            initial_record = json.load(handle)
        initial_xenium_to_he = np.asarray(
            initial_record["matrix_xenium_preview_pixel_to_he_preview_pixel"],
            dtype=float,
        )
        source_image = warp_xenium_to_he(
            morphology["image"].astype(np.float32),
            initial_xenium_to_he,
            he["image"].shape[:2],
            order=1,
        )
        source_coordinate_space = "initially_transformed_xenium_preview"
        source_title = "Initially transformed Xenium morphology"

    print("Encoding the two browser images...", flush=True)
    left_url = _png_data_url(source_image)
    right_url = _png_data_url(he["image"])
    element_id = "bilous_landmarks_" + uuid.uuid4().hex
    source_label = source_title
    config = {
        "elementId": element_id,
        "sampleName": sample_folder,
        "nPairs": int(n_points),
        "maxDimension": int(max_dimension),
        "leftUrl": left_url,
        "rightUrl": right_url,
        "sourceCoordinateSpace": source_coordinate_space,
        "sourceLabel": source_label,
        "sourceReferenceKind": "native_xenium_morphology_with_cell_nucleus_overlays",
    }
    html = r"""
<div id="__ELEMENT_ID__" style="font-family:sans-serif;max-width:1700px">
  <h3>__SAMPLE_NAME__: paired landmark selector</h3>
  <p>Click each internal landmark on the left Xenium reference, then the same
  structure on the right H&amp;E. Use the mouse wheel to zoom, drag to pan, and
  rotate the H&amp;E view when it helps identify corresponding anatomy.</p>
  <p><b>Xenium overlay:</b> yellow +=nucleus center when available,
  faint cyan=cell boundary, faint magenta=nucleus boundary.</p>
  <div style="display:flex;gap:12px;align-items:flex-start">
    <div style="width:50%">
      <b id="__ELEMENT_ID___left_label"></b><br>
      <canvas id="__ELEMENT_ID___left" style="width:100%;border:1px solid #888;cursor:crosshair;background:#111"></canvas>
    </div>
    <div style="width:50%">
      <b>H&amp;E</b><br>
      <canvas id="__ELEMENT_ID___right" style="width:100%;border:1px solid #888;cursor:crosshair;background:#111"></canvas>
    </div>
  </div>
  <p id="__ELEMENT_ID___status" style="font-weight:bold">Loading images...</p>
  <div style="display:flex;flex-wrap:wrap;gap:6px;margin:8px 0">
    <button id="__ELEMENT_ID___undo">Undo</button>
    <button id="__ELEMENT_ID___reset">Reset landmarks</button>
    <button id="__ELEMENT_ID___reset_view">Reset zoom</button>
    <button id="__ELEMENT_ID___linked_motion" title="Toggle dependent movement between H&E and Xenium views">🔗 Link views</button>
    <button id="__ELEMENT_ID___rot_left">Rotate H&amp;E left</button>
    <button id="__ELEMENT_ID___rot_right">Rotate H&amp;E right</button>
    <button id="__ELEMENT_ID___flip_h">Flip H&amp;E horizontal</button>
    <button id="__ELEMENT_ID___copy">Copy JSON</button>
  </div>
  <textarea id="__ELEMENT_ID___json" rows="9" style="width:100%;margin-top:8px" readonly></textarea>
</div>
<script>
(() => {
  const config = __CONFIG_JSON__;
  const id = config.elementId;
  const leftCanvas = document.getElementById(id + '_left');
  const rightCanvas = document.getElementById(id + '_right');
  const leftCtx = leftCanvas.getContext('2d');
  const rightCtx = rightCanvas.getContext('2d');
  const status = document.getElementById(id + '_status');
  const output = document.getElementById(id + '_json');
  const leftImage = new Image();
  const rightImage = new Image();
  const source = [];
  const target = [];
  let expectLeft = true;
  let loaded = 0;
  let linkedMovementEnabled = true;
  const orientation = {rotation: 0, flipH: false, flipV: false};
  const leftView = {canvas: leftCanvas, ctx: leftCtx, image: leftImage, side: 'left',
                    scale: 1, offsetX: 0, offsetY: 0, dragging: false,
                    moved: false, lastX: 0, lastY: 0};
  const rightView = {canvas: rightCanvas, ctx: rightCtx, image: rightImage, side: 'right',
                     scale: 1, offsetX: 0, offsetY: 0, dragging: false,
                     moved: false, lastX: 0, lastY: 0};
  document.getElementById(id + '_left_label').textContent = config.sourceLabel;

  function orientedSize() {
    const w = rightImage.naturalWidth;
    const h = rightImage.naturalHeight;
    return orientation.rotation % 2 === 0 ? {w, h} : {w: h, h: w};
  }

  function imageToDisplay(point) {
    const w = rightImage.naturalWidth;
    const h = rightImage.naturalHeight;
    const size = orientedSize();
    let x = point[0];
    let y = point[1];
    if (orientation.rotation === 1) {
      [x, y] = [y, w - x];
    } else if (orientation.rotation === 2) {
      [x, y] = [w - x, h - y];
    } else if (orientation.rotation === 3) {
      [x, y] = [h - y, x];
    }
    if (orientation.flipH) x = size.w - x;
    if (orientation.flipV) y = size.h - y;
    return [x, y];
  }

  function displayToImage(point) {
    const w = rightImage.naturalWidth;
    const h = rightImage.naturalHeight;
    const size = orientedSize();
    let x = point[0];
    let y = point[1];
    if (orientation.flipH) x = size.w - x;
    if (orientation.flipV) y = size.h - y;
    if (orientation.rotation === 1) {
      [x, y] = [w - y, x];
    } else if (orientation.rotation === 2) {
      [x, y] = [w - x, h - y];
    } else if (orientation.rotation === 3) {
      [x, y] = [y, h - x];
    }
    return [x, y];
  }

  function applyImageTransform(ctx) {
    const w = rightImage.naturalWidth;
    const h = rightImage.naturalHeight;
    const size = orientedSize();
    if (orientation.flipH) {
      ctx.translate(size.w, 0);
      ctx.scale(-1, 1);
    }
    if (orientation.flipV) {
      ctx.translate(0, size.h);
      ctx.scale(1, -1);
    }
    if (orientation.rotation === 1) {
      ctx.translate(0, w);
      ctx.rotate(-Math.PI / 2);
    } else if (orientation.rotation === 2) {
      ctx.translate(w, h);
      ctx.rotate(Math.PI);
    } else if (orientation.rotation === 3) {
      ctx.translate(h, 0);
      ctx.rotate(Math.PI / 2);
    }
  }

  function markerRadius(canvas) {
    return Math.max(6, Math.min(canvas.width, canvas.height) / 150);
  }

  function drawMarker(ctx, canvas, point, index) {
    const r = markerRadius(canvas);
    ctx.save();
    ctx.lineWidth = Math.max(2, r / 3);
    ctx.strokeStyle = '#ffffff';
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(point[0], point[1], r + 1.5, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.strokeStyle = '#ff1e1e';
    ctx.fillStyle = '#ff1e1e';
    ctx.beginPath();
    ctx.arc(point[0], point[1], r, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.font = Math.max(14, r * 2.2) + 'px sans-serif';
    ctx.lineWidth = 4;
    ctx.strokeStyle = '#ffffff';
    ctx.strokeText(String(index + 1), point[0] + r + 3, point[1] - r - 3);
    ctx.fillText(String(index + 1), point[0] + r + 3, point[1] - r - 3);
    ctx.restore();
  }

  function resetTransform(ctx, canvas) {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  function clampView(view) {
    const size = view.side === 'right'
      ? orientedSize()
      : {w: view.image.naturalWidth, h: view.image.naturalHeight};
    const minX = Math.min(0, view.canvas.width - size.w * view.scale);
    const minY = Math.min(0, view.canvas.height - size.h * view.scale);
    view.offsetX = Math.min(0, Math.max(minX, view.offsetX));
    view.offsetY = Math.min(0, Math.max(minY, view.offsetY));
  }

  function drawLeft() {
    resetTransform(leftCtx, leftCanvas);
    leftCtx.save();
    leftCtx.setTransform(leftView.scale, 0, 0, leftView.scale,
                         leftView.offsetX, leftView.offsetY);
    leftCtx.drawImage(leftImage, 0, 0);
    source.forEach((point, index) => drawMarker(leftCtx, leftCanvas, point, index));
    leftCtx.restore();
  }

  function drawRight() {
    resetTransform(rightCtx, rightCanvas);
    rightCtx.save();
    rightCtx.setTransform(rightView.scale, 0, 0, rightView.scale,
                          rightView.offsetX, rightView.offsetY);
    rightCtx.save();
    applyImageTransform(rightCtx);
    rightCtx.drawImage(rightImage, 0, 0);
    rightCtx.restore();
    target.forEach((point, index) => {
      drawMarker(rightCtx, rightCanvas, imageToDisplay(point), index);
    });
    rightCtx.restore();
  }

  function redraw() {
    if (loaded < 2) return;
    clampView(leftView);
    clampView(rightView);
    drawLeft();
    drawRight();
  }

  function update() {
    redraw();
    output.value = JSON.stringify({
      sample_folder: config.sampleName,
      sample_name: config.sampleName,
      max_dimension: config.maxDimension,
      source_coordinate_space: config.sourceCoordinateSpace,
      source_reference_kind: config.sourceReferenceKind,
      source_reference_label: config.sourceLabel,
      target_coordinate_space: 'he_preview_original',
      source_points: source,
      target_points: target,
      target_points_display_xy: target.map(imageToDisplay),
      he_display_orientation: {
        rotation_degrees_ccw: orientation.rotation * 90,
        flip_horizontal: orientation.flipH,
        flip_vertical: orientation.flipV
      },
      linked_movement_enabled: linkedMovementEnabled
    }, null, 2);
    if (source.length >= config.nPairs && target.length >= config.nPairs) {
      status.textContent = 'Complete - copy the JSON.';
    } else if (expectLeft) {
      status.textContent = `Pair ${source.length + 1}/${config.nPairs}: click LEFT Xenium reference.`;
    } else {
      status.textContent = `Pair ${source.length}/${config.nPairs}: click matching RIGHT H&E.`;
    }
  }

  function configureLeft() {
    leftCanvas.width = leftImage.naturalWidth;
    leftCanvas.height = leftImage.naturalHeight;
    loaded += 1;
    update();
  }

  function configureRight({resetView = true} = {}) {
    const size = orientedSize();
    rightCanvas.width = size.w;
    rightCanvas.height = size.h;
    if (resetView) {
      rightView.scale = 1;
      rightView.offsetX = 0;
      rightView.offsetY = 0;
    }
    update();
  }

  function canvasCoords(event, canvas) {
    const rect = canvas.getBoundingClientRect();
    return [
      (event.clientX - rect.left) * canvas.width / rect.width,
      (event.clientY - rect.top) * canvas.height / rect.height
    ];
  }

  function viewPoint(event, view) {
    const p = canvasCoords(event, view.canvas);
    const displayPoint = [
      (p[0] - view.offsetX) / view.scale,
      (p[1] - view.offsetY) / view.scale
    ];
    return view.side === 'right' ? displayToImage(displayPoint) : displayPoint;
  }

  function addPoint(view, event) {
    if (view.side === 'left') {
      if (!expectLeft || source.length >= config.nPairs) return;
      source.push(viewPoint(event, view));
      expectLeft = false;
    } else {
      if (expectLeft || target.length >= config.nPairs) return;
      target.push(viewPoint(event, view));
      expectLeft = true;
    }
    update();
  }

  function hasRoughAlignment() {
    return linkedMovementEnabled && source.length >= 3 && target.length >= 3;
  }

  function pairedView(view) {
    return view.side === 'right' ? leftView : rightView;
  }

  function zoomAt(view, canvasPoint, factor) {
    const before = [(canvasPoint[0] - view.offsetX) / view.scale,
                    (canvasPoint[1] - view.offsetY) / view.scale];
    const nextScale = Math.max(1, Math.min(32, view.scale * factor));
    view.scale = nextScale;
    view.offsetX = canvasPoint[0] - before[0] * view.scale;
    view.offsetY = canvasPoint[1] - before[1] * view.scale;
  }

  function zoomBoth(view, canvasPoint, factor) {
    zoomAt(view, canvasPoint, factor);
    if (linkedMovementEnabled && (view.side === 'right' || hasRoughAlignment())) {
      const other = pairedView(view);
      zoomAt(
        other,
        [other.canvas.width / 2, other.canvas.height / 2],
        factor
      );
    }
  }

  function panView(view, dx, dy) {
    view.offsetX += dx;
    view.offsetY += dy;
  }

  function panLinkedViews(view, dx, dy) {
    panView(view, dx, dy);
    if (hasRoughAlignment()) {
      panView(pairedView(view), dx, dy);
    }
  }

  function installInteractions(view) {
    view.canvas.addEventListener('wheel', event => {
      event.preventDefault();
      const p = canvasCoords(event, view.canvas);
      const factor = event.deltaY < 0 ? 1.25 : 0.8;
      zoomBoth(view, p, factor);
      redraw();
    }, {passive: false});
    view.canvas.addEventListener('mousedown', event => {
      view.dragging = true;
      view.moved = false;
      const p = canvasCoords(event, view.canvas);
      view.lastX = p[0];
      view.lastY = p[1];
    });
    window.addEventListener('mousemove', event => {
      if (!view.dragging) return;
      const p = canvasCoords(event, view.canvas);
      const dx = p[0] - view.lastX;
      const dy = p[1] - view.lastY;
      if (Math.hypot(dx, dy) > 2) view.moved = true;
      if (view.scale > 1) {
        panLinkedViews(view, dx, dy);
        view.lastX = p[0];
        view.lastY = p[1];
        redraw();
      }
    });
    window.addEventListener('mouseup', event => {
      if (!view.dragging) return;
      view.dragging = false;
      if (!view.moved) addPoint(view, event);
    });
  }

  document.getElementById(id + '_undo').onclick = () => {
    if (!expectLeft && source.length > target.length) {
      source.pop();
      expectLeft = true;
    } else if (target.length) {
      source.pop();
      target.pop();
      expectLeft = true;
    }
    update();
  };
  document.getElementById(id + '_reset').onclick = () => {
    source.length = 0;
    target.length = 0;
    expectLeft = true;
    update();
  };
  document.getElementById(id + '_reset_view').onclick = () => {
    for (const view of [leftView, rightView]) {
      view.scale = 1;
      view.offsetX = 0;
      view.offsetY = 0;
    }
    update();
  };
  function updateLinkedMotionButton() {
    const button = document.getElementById(id + '_linked_motion');
    button.textContent = linkedMovementEnabled ? '🔗 Link views' : '○ Link views';
    button.style.fontWeight = linkedMovementEnabled ? 'bold' : 'normal';
  }
  document.getElementById(id + '_linked_motion').onclick = () => {
    linkedMovementEnabled = !linkedMovementEnabled;
    updateLinkedMotionButton();
    update();
  };
  document.getElementById(id + '_rot_left').onclick = () => {
    orientation.rotation = (orientation.rotation + 1) % 4;
    configureRight();
  };
  document.getElementById(id + '_rot_right').onclick = () => {
    orientation.rotation = (orientation.rotation + 3) % 4;
    configureRight();
  };
  document.getElementById(id + '_flip_h').onclick = () => {
    orientation.flipH = !orientation.flipH;
    configureRight({resetView: false});
  };
  document.getElementById(id + '_copy').onclick = async () => {
    output.select();
    try {
      await navigator.clipboard.writeText(output.value);
    } catch (error) {
      document.execCommand('copy');
    }
    status.textContent = 'JSON copied.';
  };

  installInteractions(leftView);
  installInteractions(rightView);
  updateLinkedMotionButton();
  leftImage.onload = configureLeft;
  rightImage.onload = () => {
    loaded += 1;
    configureRight();
  };
  leftImage.src = config.leftUrl;
  rightImage.src = config.rightUrl;
})();
</script>
"""
    html = (
        html.replace("__ELEMENT_ID__", element_id)
        .replace("__SAMPLE_NAME__", sample_folder)
        .replace("__CONFIG_JSON__", json.dumps(config))
    )
    widget = HTML(html)
    display(widget)
    print("Landmark selector displayed.", flush=True)
    return widget


def mapped_sample_folders(
    root: Path | str,
    mapping_csv: Path | str | None = None,
) -> list[str]:
    """Return mapped sample folders in mapping-table order."""
    root = Path(root).expanduser().resolve()
    mapping_path = (
        Path(mapping_csv).expanduser().resolve()
        if mapping_csv is not None
        else root / "mapping_previews" / "he_sample_mapping.csv"
    )
    return [
        row["sample_folder"]
        for row in _read_csv(mapping_path)
        if row.get("candidate_index", "").strip()
    ]


def show_all_landmark_selector(
    root: Path | str,
    *,
    mapping_csv: Path | str | None = None,
    ndpi_regions_csv: Path | str | None = None,
    sample_folders: list[str] | tuple[str, ...] | None = None,
    n_points: int = 8,
    max_dimension: int = 1400,
) -> Any:
    """Collect landmarks for every mapped sample in one HTML session.

    The browser keeps all clicks locally. After the final sample, copy one
    combined JSON record and save each landmark fit with
    ``save_landmarks_and_show_next`` or ``manual_landmark_align``.
    """
    from IPython.display import HTML, display

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
    rows = [
        row
        for row in _read_csv(mapping_path)
        if row.get("candidate_index", "").strip()
    ]
    if sample_folders is not None:
        requested = set(sample_folders)
        rows = [row for row in rows if row["sample_folder"] in requested]
        missing = requested - {row["sample_folder"] for row in rows}
        if missing:
            raise KeyError(f"Mapped samples not found: {sorted(missing)}")
    if not rows:
        raise ValueError("No mapped samples selected")

    samples = {sample.directory.name: sample for sample in discover_samples(root)}
    regions = load_region_lookup(regions_path)
    browser_samples = []
    for row in tqdm(rows, desc="Preparing landmark images", unit="sample"):
        sample_folder = row["sample_folder"]
        if sample_folder not in samples:
            raise KeyError(f"Sample folder not found: {sample_folder}")
        sample = samples[sample_folder]
        morphology = morphology_preview(sample, max_dimension)
        he = load_he_candidate(
            root,
            row,
            regions,
            max_dimension=max_dimension,
        )
        xenium_display = _morphology_with_cell_overlays(sample, morphology)
        browser_samples.append(
            {
                "sample_folder": sample_folder,
                "left_url": _png_data_url(xenium_display),
                "right_url": _png_data_url(he["image"]),
            }
        )

    element_id = "landmark_session_" + uuid.uuid4().hex
    samples_json = json.dumps(browser_samples)
    html = f"""
<div id="{element_id}" style="font-family:sans-serif;max-width:1500px">
  <h3>Mapped-sample landmark collection</h3>
  <p id="{element_id}_sample" style="font-size:16px;font-weight:bold"></p>
  <p>For each pair, click <b>left native Xenium</b> and then the matching
  <b>right H&amp;E</b>. Complete {n_points} pairs before advancing.</p>
  <div style="display:flex;gap:12px;align-items:flex-start">
    <div style="width:50%"><b>Native Xenium morphology + cell/nucleus overlays</b><br>
      <canvas id="{element_id}_left" style="width:100%;border:1px solid #888;cursor:crosshair"></canvas>
    </div>
    <div style="width:50%"><b>Mapped H&amp;E</b><br>
      <canvas id="{element_id}_right" style="width:100%;border:1px solid #888;cursor:crosshair"></canvas>
    </div>
  </div>
  <p id="{element_id}_status" style="font-weight:bold">Loading…</p>
  <button id="{element_id}_previous">Previous sample</button>
  <button id="{element_id}_undo">Undo last pair</button>
  <button id="{element_id}_reset">Reset current</button>
  <button id="{element_id}_next">Next sample</button>
  <button id="{element_id}_copy">Copy all JSON</button>
  <textarea id="{element_id}_json" rows="8" style="width:100%;margin-top:8px" readonly></textarea>
</div>
<script>
(() => {{
  const imageRecords = {samples_json};
  const nPairs = {int(n_points)};
  const previewMaxDimension = {int(max_dimension)};
  const records = imageRecords.map(item => ({{
    sample_folder: item.sample_folder,
    source_coordinate_space: 'native_xenium_preview',
    max_dimension: previewMaxDimension,
    source_points: [],
    target_points: []
  }}));
  let current = 0;
  let expectLeft = true;
  let loaded = 0;
  const leftCanvas = document.getElementById('{element_id}_left');
  const rightCanvas = document.getElementById('{element_id}_right');
  const leftCtx = leftCanvas.getContext('2d');
  const rightCtx = rightCanvas.getContext('2d');
  const leftImage = new Image();
  const rightImage = new Image();
  const status = document.getElementById('{element_id}_status');
  const sampleLabel = document.getElementById('{element_id}_sample');
  const output = document.getElementById('{element_id}_json');

  function redraw(canvas, ctx, image, points) {{
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    ctx.font = Math.max(13, canvas.width / 90) + 'px sans-serif';
    ctx.lineWidth = Math.max(2, canvas.width / 700);
    points.forEach((p, i) => {{
      ctx.beginPath();
      ctx.arc(p[0], p[1], Math.max(5, canvas.width / 180), 0, 2*Math.PI);
      ctx.strokeStyle = '#ff1f1f'; ctx.stroke();
      ctx.fillStyle = '#ff1f1f';
      ctx.fillText(String(i + 1), p[0] + 7, p[1] - 7);
    }});
  }}

  function combinedJSON() {{
    return JSON.stringify({{
      max_dimension: previewMaxDimension,
      landmark_sets: records
    }}, null, 2);
  }}

  function update() {{
    const record = records[current];
    redraw(leftCanvas, leftCtx, leftImage, record.source_points);
    redraw(rightCanvas, rightCtx, rightImage, record.target_points);
    sampleLabel.textContent = `Sample ${{current + 1}}/${{records.length}}: ${{record.sample_folder}}`;
    output.value = combinedJSON();
    const complete = record.source_points.length === nPairs && record.target_points.length === nPairs;
    document.getElementById('{element_id}_next').disabled = !complete || current === records.length - 1;
    document.getElementById('{element_id}_previous').disabled = current === 0;
    if (complete && current === records.length - 1) {{
      status.textContent = 'All samples complete. Click Copy all JSON.';
    }} else if (complete) {{
      status.textContent = 'Current sample complete. Click Next sample.';
    }} else if (expectLeft) {{
      status.textContent = `Pair ${{record.source_points.length + 1}}/${{nPairs}}: click LEFT Xenium.`;
    }} else {{
      status.textContent = `Pair ${{record.source_points.length}}/${{nPairs}}: click matching RIGHT H&E.`;
    }}
  }}

  function configure(canvas, image) {{
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    loaded += 1;
    if (loaded === 2) update();
  }}

  function loadCurrent() {{
    loaded = 0;
    expectLeft = records[current].source_points.length === records[current].target_points.length;
    leftImage.onload = () => configure(leftCanvas, leftImage);
    rightImage.onload = () => configure(rightCanvas, rightImage);
    leftImage.src = imageRecords[current].left_url;
    rightImage.src = imageRecords[current].right_url;
    status.textContent = 'Loading sample images…';
  }}

  function coordinates(event, canvas) {{
    const rect = canvas.getBoundingClientRect();
    return [
      (event.clientX - rect.left) * canvas.width / rect.width,
      (event.clientY - rect.top) * canvas.height / rect.height
    ];
  }}
  leftCanvas.onclick = event => {{
    const record = records[current];
    if (!expectLeft || record.source_points.length >= nPairs) return;
    record.source_points.push(coordinates(event, leftCanvas));
    expectLeft = false; update();
  }};
  rightCanvas.onclick = event => {{
    const record = records[current];
    if (expectLeft || record.target_points.length >= nPairs) return;
    record.target_points.push(coordinates(event, rightCanvas));
    expectLeft = true; update();
  }};
  document.getElementById('{element_id}_previous').onclick = () => {{
    if (current > 0) {{ current -= 1; loadCurrent(); }}
  }};
  document.getElementById('{element_id}_next').onclick = () => {{
    if (current < records.length - 1) {{ current += 1; loadCurrent(); }}
  }};
  document.getElementById('{element_id}_undo').onclick = () => {{
    const record = records[current];
    if (!expectLeft && record.source_points.length > record.target_points.length) record.source_points.pop();
    else {{ record.source_points.pop(); record.target_points.pop(); }}
    expectLeft = true; update();
  }};
  document.getElementById('{element_id}_reset').onclick = () => {{
    records[current].source_points.length = 0;
    records[current].target_points.length = 0;
    expectLeft = true; update();
  }};
  document.getElementById('{element_id}_copy').onclick = async () => {{
    output.value = combinedJSON(); output.select();
    try {{ await navigator.clipboard.writeText(output.value); }}
    catch (error) {{ document.execCommand('copy'); }}
    status.textContent = 'Combined landmark JSON copied.';
  }};
  loadCurrent();
}})();
</script>
"""
    widget = HTML(html)
    display(widget)
    print(
        f"Landmark session displayed for {len(browser_samples)} samples. "
        "Copy the combined JSON after the final sample.",
        flush=True,
    )
    return widget


def next_landmark_sample(
    root: Path | str,
    *,
    mapping_csv: Path | str | None = None,
    output_dir: Path | str | None = None,
    skip_review_dir: Path | str | None = None,
) -> str | None:
    """Find the next mapped sample without a saved manual transformation."""
    root = Path(root).expanduser().resolve()
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else root / "manual_alignment_from_mapping"
    )
    skip_dir = (
        Path(skip_review_dir).expanduser().resolve()
        if skip_review_dir is not None
        else None
    )
    for sample_folder in mapped_sample_folders(root, mapping_csv):
        completed = destination / f"{sample_folder}_manual_alignment.json"
        accepted = (
            skip_dir is not None
            and (
                (skip_dir / f"{sample_folder}_alignment_candidate.json").exists()
                or (skip_dir / f"{sample_folder}_manual_alignment.json").exists()
            )
        )
        if not completed.exists() and not accepted:
            return sample_folder
    return None


def samples_missing_alignment(
    root: Path | str,
    *,
    mapping_csv: Path | str | None = None,
    output_dir: Path | str | None = None,
    sample_names: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Return mapped samples that still need a Xenium Explorer alignment CSV."""
    root = Path(root).expanduser().resolve()
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else root / "manual_alignment_from_mapping"
    )
    folders = mapped_sample_folders(root, mapping_csv)
    if sample_names is not None:
        requested = set(sample_names)
        folders = [name for name in folders if name in requested]
    return [
        sample_folder
        for sample_folder in folders
        if not (destination / f"{sample_folder}_he_imagealignment.csv").is_file()
    ]


def show_next_landmark_selector(
    root: Path | str,
    *,
    mapping_csv: Path | str | None = None,
    ndpi_regions_csv: Path | str | None = None,
    output_dir: Path | str | None = None,
    skip_review_dir: Path | str | None = None,
    n_points: int = 8,
    max_dimension: int = 1000,
    use_morphology_mip: bool = True,
) -> str | None:
    """Display the next incomplete sample without requiring its ID."""
    sample_folder = next_landmark_sample(
        root,
        mapping_csv=mapping_csv,
        output_dir=output_dir,
        skip_review_dir=skip_review_dir,
    )
    if sample_folder is None:
        print("All mapped samples in this landmark session are complete.")
        return None
    folders = mapped_sample_folders(root, mapping_csv)
    print(
        f"Next sample: {sample_folder} "
        f"({folders.index(sample_folder) + 1} of {len(folders)} in mapping order)"
    )
    show_landmark_selector(
        root,
        sample_folder,
        mapping_csv=mapping_csv,
        ndpi_regions_csv=ndpi_regions_csv,
        n_points=n_points,
        max_dimension=max_dimension,
        use_morphology_mip=use_morphology_mip,
    )
    return sample_folder


def manual_landmark_align(
    root: Path | str,
    sample_folder: str,
    *,
    initial_review_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    mapping_csv: Path | str | None = None,
    ndpi_regions_csv: Path | str | None = None,
    n_points: int = 8,
    max_dimension: int = 1000,
    random_seed: int = 1,
    landmarks: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate and save a landmark-based affine transform.

    With ``initial_review_dir=None`` (the default), landmark coordinates map
    the native Xenium preview directly to the mapped H&E preview. Supplying an
    initial review directory instead estimates a correction to that transform.
    """
    root = Path(root).expanduser().resolve()
    initial_dir = (
        Path(initial_review_dir).expanduser().resolve()
        if initial_review_dir is not None
        else None
    )
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else root / "alignment_reviews_manual"
    )
    destination.mkdir(parents=True, exist_ok=True)
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

    samples = {sample.directory.name: sample for sample in discover_samples(root)}
    if sample_folder not in samples:
        raise KeyError(f"Sample folder not found: {sample_folder}")
    sample = samples[sample_folder]
    mapping_row = _find_one(
        _read_csv(mapping_path), "sample_folder", sample_folder
    )
    morphology = morphology_preview(sample, max_dimension)
    he = load_he_candidate(
        root,
        mapping_row,
        load_region_lookup(regions_path),
        max_dimension=max_dimension,
    )
    cells = read_cells(sample)
    xenium_mask, cell_x_full, cell_y_full = centroid_mask(cells, morphology)
    if initial_dir is None:
        initial_path = None
        initial_record = {
            "sample_folder": sample.directory.name,
            "sample_label": sample.label,
            "gsm": sample.gsm,
            "source_file": mapping_row["source_file"],
            "candidate_type": mapping_row["candidate_type"],
            "candidate_index": int(mapping_row["candidate_index"]),
            "he_candidate_full_shape_yx": [
                he["full_height"],
                he["full_width"],
            ],
            "he_candidate_origin_xy_in_source": [
                he["origin_x_in_source"],
                he["origin_y_in_source"],
            ],
            "xenium_morphology_full_shape_yx": [
                morphology["full_height"],
                morphology["full_width"],
            ],
            "xenium_pixel_size_um_xy": [
                morphology["pixel_size_x_um"],
                morphology["pixel_size_y_um"],
            ],
        }
        initial_xenium_to_he = None
        landmark_source_image = _morphology_with_cell_overlays(sample, morphology)
        expected_coordinate_space = "native_xenium_preview"
    else:
        initial_path = initial_dir / f"{sample_folder}_alignment_candidate.json"
        if not initial_path.exists():
            raise FileNotFoundError(initial_path)
        with initial_path.open(encoding="utf-8") as handle:
            initial_record = json.load(handle)
        initial_xenium_to_he = np.asarray(
            initial_record["matrix_xenium_preview_pixel_to_he_preview_pixel"],
            dtype=float,
        )
        landmark_source_image = warp_xenium_to_he(
            morphology["image"].astype(np.float32),
            initial_xenium_to_he,
            he["image"].shape[:2],
            order=1,
        )
        expected_coordinate_space = "initially_transformed_xenium_preview"

    if landmarks is None:
        source_points, target_points = collect_paired_landmarks(
            landmark_source_image,
            he["image"],
            n_points=n_points,
        )
    else:
        landmark_record = (
            json.loads(landmarks) if isinstance(landmarks, str) else landmarks
        )
        recorded_space = landmark_record.get("source_coordinate_space")
        if recorded_space and recorded_space != expected_coordinate_space:
            raise ValueError(
                "Landmark JSON coordinate space does not match this run: "
                f"expected {expected_coordinate_space!r}, got {recorded_space!r}"
            )
        recorded_dimension = landmark_record.get("max_dimension")
        if recorded_dimension is not None and int(recorded_dimension) != max_dimension:
            raise ValueError(
                "Landmark JSON was created with max_dimension="
                f"{recorded_dimension}, but this run uses {max_dimension}"
            )
        source_points = np.asarray(
            landmark_record["source_points"], dtype=float
        )
        target_points = np.asarray(
            landmark_record["target_points"], dtype=float
        )
        if source_points.shape != target_points.shape:
            raise ValueError("Source and target landmark arrays must match")
        if source_points.ndim != 2 or source_points.shape[1] != 2:
            raise ValueError("Landmarks must be N x 2 coordinate arrays")
        if len(source_points) < 3:
            raise ValueError("At least three landmark pairs are required")
        n_points = int(len(source_points))
    correction = AffineTransform()
    if not correction.estimate(source_points, target_points):
        raise RuntimeError("Could not estimate an affine landmark correction")
    correction_matrix = correction.params
    final_xenium_to_he = (
        correction_matrix
        if initial_xenium_to_he is None
        else correction_matrix @ initial_xenium_to_he
    )

    predicted_targets = correction(source_points)
    residuals = np.linalg.norm(predicted_targets - target_points, axis=1)
    landmark_rmse = float(math.sqrt(np.mean(residuals**2)))

    transformed_final = warp_xenium_to_he(
        morphology["image"].astype(np.float32),
        final_xenium_to_he,
        he["image"].shape[:2],
        order=1,
    )
    transformed_mask = warp_xenium_to_he(
        xenium_mask.astype(np.float32),
        final_xenium_to_he,
        he["image"].shape[:2],
        order=0,
    ) > 0.5
    cyan = _cyan_rgba(transformed_final, transformed_mask)

    xenium_x_preview = cell_x_full * morphology["scale_x"]
    xenium_y_preview = cell_y_full * morphology["scale_y"]
    he_x_preview, he_y_preview = transform_points(
        final_xenium_to_he,
        xenium_x_preview,
        xenium_y_preview,
    )
    inside = (
        (he_x_preview >= 0)
        & (he_x_preview < he["image"].shape[1])
        & (he_y_preview >= 0)
        & (he_y_preview < he["image"].shape[0])
    )
    rng = np.random.default_rng(random_seed)
    valid = np.flatnonzero(inside)
    shown = rng.choice(valid, size=min(40_000, len(valid)), replace=False)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 7),
        constrained_layout=True,
    )
    axes[0].imshow(he["image"], interpolation="none", resample=False)
    axes[0].set_title("H&E")
    axes[1].imshow(transformed_final, cmap="gray", interpolation="none")
    axes[1].scatter(
        target_points[:, 0],
        target_points[:, 1],
        s=36,
        facecolors="none",
        edgecolors="red",
        linewidths=1.2,
    )
    axes[1].set_title(
        f"Landmark-corrected Xenium\nlandmark RMSE={landmark_rmse:.1f} preview px"
    )
    axes[2].imshow(he["image"], interpolation="none", resample=False)
    axes[2].imshow(cyan, interpolation="none")
    axes[2].scatter(
        he_x_preview[shown],
        he_y_preview[shown],
        s=0.25,
        color="red",
        alpha=0.45,
        linewidths=0,
        rasterized=True,
    )
    axes[2].set_title(
        "Manual landmark overlay\n"
        f"{inside.mean():.1%} of cell centroids inside H&E canvas"
    )
    for axis in axes:
        axis.set_xlim(0, he["image"].shape[1])
        axis.set_ylim(he["image"].shape[0], 0)
        axis.axis("off")
    fig.suptitle(f"{sample_folder}: manual affine landmark review", fontsize=15)

    figure_path = destination / f"{sample_folder}_manual_alignment_review.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

    xenium_full_to_preview = diagonal_matrix(
        morphology["scale_x"], morphology["scale_y"]
    )
    he_preview_to_full = diagonal_matrix(
        1.0 / he["scale_x"], 1.0 / he["scale_y"]
    )
    xenium_full_to_he_full = (
        he_preview_to_full
        @ final_xenium_to_he
        @ xenium_full_to_preview
    )
    xenium_um_to_he_full = xenium_full_to_he_full @ diagonal_matrix(
        1.0 / morphology["pixel_size_x_um"],
        1.0 / morphology["pixel_size_y_um"],
    )
    imagealignment_path = destination / f"{sample_folder}_he_imagealignment.csv"
    he_to_xenium_full = write_xenium_explorer_imagealignment_csv(
        imagealignment_path,
        xenium_full_to_he_full,
    )

    output_record = dict(initial_record)
    output_record.update(
        {
            "status": "manual_landmark_alignment_needs_visual_review",
            "registration_method": (
                "manual_native_xenium_to_mapped_he_affine"
                if initial_path is None
                else "manual_affine_correction_of_initial_transform"
            ),
            "initial_transform_json": (
                str(initial_path) if initial_path is not None else None
            ),
            "manual_source_coordinate_space": expected_coordinate_space,
            "preview_max_dimension": int(max_dimension),
            "loaded_he_path": str(he["source"]),
            "used_extracted_ome_tiff": he.get(
                "used_extracted_ome_tiff",
                Path(he["source"]).parent.name == "full_resolution_he",
            ),
            "manual_landmark_count": int(n_points),
            "manual_landmark_rmse_preview_pixels": landmark_rmse,
            "manual_reflection_applied": bool(
                np.linalg.det(final_xenium_to_he[:2, :2]) < 0
            ),
            "manual_source_points_preview_xy": source_points.tolist(),
            "manual_target_points_he_preview_xy": target_points.tolist(),
            "manual_fitted_affine_matrix": correction_matrix.tolist(),
            "matrix_xenium_preview_pixel_to_he_preview_pixel": (
                final_xenium_to_he.tolist()
            ),
            "matrix_he_preview_pixel_to_xenium_preview_pixel": (
                np.linalg.inv(final_xenium_to_he).tolist()
            ),
            "matrix_xenium_full_pixel_to_he_candidate_local_full_pixel": (
                xenium_full_to_he_full.tolist()
            ),
            "matrix_xenium_um_to_he_candidate_local_full_pixel": (
                xenium_um_to_he_full.tolist()
            ),
            "matrix_he_candidate_local_full_pixel_to_xenium_full_pixel": (
                he_to_xenium_full.tolist()
            ),
            "xenium_explorer_imagealignment_csv": str(imagealignment_path),
            "xenium_explorer_matrix_convention": (
                "3x3 row-major affine matrix mapping H&E candidate "
                "full-resolution pixels to Xenium full-resolution pixels; "
                "final row is [0, 0, 1]."
            ),
            "cell_fraction_inside_he_canvas": float(inside.mean()),
            "overlay_figure": str(figure_path),
            "warning": (
                "Manually landmarked global affine candidate. Review the saved "
                "overlay; samples with independently displaced tissue pieces "
                "may still require piecewise or non-rigid registration."
            ),
        }
    )
    transform_path = destination / f"{sample_folder}_manual_alignment.json"
    with transform_path.open("w", encoding="utf-8") as handle:
        json.dump(output_record, handle, indent=2)
        handle.write("\n")

    print(f"Saved review: {figure_path}")
    print(f"Saved transform: {transform_path}")
    print(f"Xenium Explorer-style matrix: {imagealignment_path}")
    return output_record


def save_landmarks_and_show_next(
    root: Path | str,
    landmarks: str | dict[str, Any],
    *,
    mapping_csv: Path | str | None = None,
    ndpi_regions_csv: Path | str | None = None,
    output_dir: Path | str | None = None,
    skip_review_dir: Path | str | None = None,
    n_points: int = 8,
    max_dimension: int = 2000,
    use_morphology_mip: bool = True,
    random_seed: int = 1,
) -> dict[str, Any]:
    """Save copied landmark JSON and automatically display the next sample."""
    landmark_record = (
        json.loads(landmarks) if isinstance(landmarks, str) else landmarks
    )
    sample_folder = landmark_record.get("sample_folder")
    if not sample_folder:
        raise ValueError(
            "Landmark JSON has no sample_folder. Regenerate it with the "
            "updated show_landmark_selector function."
        )
    result = manual_landmark_align(
        root,
        sample_folder,
        output_dir=output_dir,
        mapping_csv=mapping_csv,
        ndpi_regions_csv=ndpi_regions_csv,
        max_dimension=max_dimension,
        random_seed=random_seed,
        landmarks=landmark_record,
    )
    next_sample = show_next_landmark_selector(
        root,
        mapping_csv=mapping_csv,
        ndpi_regions_csv=ndpi_regions_csv,
        output_dir=output_dir,
        skip_review_dir=skip_review_dir,
        n_points=n_points,
        max_dimension=max_dimension,
        use_morphology_mip=use_morphology_mip,
    )
    return {
        "saved_sample": sample_folder,
        "next_sample": next_sample,
        "transform": result,
    }
