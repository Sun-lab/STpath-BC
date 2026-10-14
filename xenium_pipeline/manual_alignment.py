#!/usr/bin/env python3
"""Shared manual landmark alignment for standard Xenium sample folders.

The workflow mirrors Xenium Explorer's keypoint alignment: click paired
landmarks on a Xenium reference preview and the H&E image, fit an
arbitrary-angle similarity transform with optional reflection, then export the
selected transform as a 3x3 ``*_imagealignment.csv``.
The selected JSON is consumed by ``xenium_pipeline.standard_xenium``.
"""

from __future__ import annotations

import base64
from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any
import uuid

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tifffile

try:
    from . import standard_xenium as pipeline
    from .tile_utils import xenium_explorer_imagealignment_matrix
except ImportError:  # pragma: no cover - supports direct script execution.
    import standard_xenium as pipeline  # type: ignore[no-redef]
    from xenium_pipeline.tile_utils import xenium_explorer_imagealignment_matrix


def diagonal_matrix(x_scale: float, y_scale: float) -> np.ndarray:
    return np.array(
        [[x_scale, 0, 0], [0, y_scale, 0], [0, 0, 1]],
        dtype=float,
    )


def transform_xy(matrix: np.ndarray, xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    homogeneous = np.column_stack([xy, np.ones(len(xy))])
    transformed = homogeneous @ np.asarray(matrix, dtype=float).T
    return transformed[:, :2] / transformed[:, 2, None]




def write_xenium_explorer_imagealignment_csv(
    path: str | Path,
    xenium_to_he: np.ndarray,
) -> np.ndarray:
    """Write a Xenium Explorer-compatible 3x3 imagealignment CSV."""
    imagealignment = xenium_explorer_imagealignment_matrix(xenium_to_he)
    np.savetxt(Path(path), imagealignment, delimiter=",", fmt="%.15g")
    return imagealignment


def _fit_similarity_candidate(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    reflection: bool,
) -> dict[str, Any]:
    """Fit one 2D uniform-scale transform, with or without reflection."""
    source_points = np.asarray(source_points, dtype=float)
    target_points = np.asarray(target_points, dtype=float)
    if source_points.shape != target_points.shape or source_points.ndim != 2:
        raise ValueError("Source and target landmarks must be matching N x 2 arrays")
    if source_points.shape[1] != 2:
        raise ValueError("Landmarks must be N x 2 arrays")
    if len(source_points) < 2:
        raise ValueError("At least two landmark pairs are required")

    rows: list[list[float]] = []
    values: list[float] = []
    for (x, y), (u, v) in zip(source_points, target_points):
        if reflection:
            rows.append([x, y, 1.0, 0.0])
            values.append(u)
            rows.append([y, -x, 0.0, 1.0])
            values.append(v)
        else:
            rows.append([x, -y, 1.0, 0.0])
            values.append(u)
            rows.append([y, x, 0.0, 1.0])
            values.append(v)

    params, *_ = np.linalg.lstsq(
        np.asarray(rows, dtype=float),
        np.asarray(values, dtype=float),
        rcond=None,
    )
    a, b, tx, ty = (float(value) for value in params)
    if reflection:
        linear = np.array([[a, b], [b, -a]], dtype=float)
    else:
        linear = np.array([[a, -b], [b, a]], dtype=float)
    scale = float(math.hypot(a, b))
    if scale <= 1e-12:
        raise ValueError("Landmarks do not span a usable area")

    matrix = np.array(
        [
            [linear[0, 0], linear[0, 1], tx],
            [linear[1, 0], linear[1, 1], ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    predicted = transform_xy(matrix, source_points)
    residuals = np.linalg.norm(predicted - target_points, axis=1)
    return {
        "valid": True,
        "reflection": bool(reflection),
        "uniform_scale_preview_pixels": scale,
        "rotation_degrees_ccw": float(math.degrees(math.atan2(b, a))),
        "translation_he_preview_xy": [tx, ty],
        "landmark_rmse_preview_pixels": float(math.sqrt(np.mean(residuals**2))),
        "landmark_residuals_preview_pixels": residuals.tolist(),
        "matrix_xenium_preview_pixel_to_he_preview_pixel": matrix,
        "predicted_target_points_he_preview_xy": predicted,
    }


def fit_xenium_like_similarity(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    allow_reflection: bool = True,
) -> dict[str, Any]:
    """Fit the lowest-RMSE Xenium-like keypoint transform."""
    candidates = [
        _fit_similarity_candidate(
            source_points,
            target_points,
            reflection=False,
        )
    ]
    if allow_reflection:
        candidates.append(
            _fit_similarity_candidate(
                source_points,
                target_points,
                reflection=True,
            )
        )
    return min(candidates, key=lambda item: item["landmark_rmse_preview_pixels"])


def samples_missing_alignment(
    root: str | Path,
    sample_names: list[str] | None = None,
) -> list[str]:
    """Return samples that still lack an ``*_he_imagealignment.csv`` file."""
    samples = pipeline.discover_samples(Path(root), requested=sample_names)
    return [
        sample.name
        for sample in samples
        if sample.he_image is not None
        and (sample.alignment is None or not sample.alignment.is_file())
    ]


def _load_image_preview(path: Path, max_dimension: int) -> dict[str, Any]:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        full_height, full_width = pipeline.get_hw(tuple(series.levels[0].shape))
        level_index = len(series.levels) - 1
        for index, level in enumerate(series.levels):
            height, width = pipeline.get_hw(tuple(level.shape))
            if max(height, width) <= max_dimension:
                level_index = index
                break
        image = pipeline.ensure_rgb(series.levels[level_index].asarray())
    height, width = image.shape[:2]
    return {
        "image": image,
        "full_height": full_height,
        "full_width": full_width,
        "scale_x": width / full_width,
        "scale_y": height / full_height,
        "level_index": level_index,
    }


def _load_he_preview(path: Path, max_dimension: int) -> dict[str, Any]:
    return _load_image_preview(path, max_dimension)


def _find_morphology_mip(sample: Any) -> Path | None:
    candidates = [
        sample.outs / "morphology_mip.ome.tif",
        sample.base / "morphology_mip.ome.tif",
    ]
    candidates.extend(sample.base.glob("*_outs/morphology_mip.ome.tif"))
    candidates.extend(sample.base.glob("*_xe_outs/morphology_mip.ome.tif"))
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            return candidate
    return None


def _centroid_intensity(
    xy_full_pixels: np.ndarray,
    width: int,
    height: int,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    finite = np.isfinite(xy_full_pixels).all(axis=1)
    xy = np.asarray(xy_full_pixels[finite], dtype=float)
    if not len(xy):
        return np.zeros((height, width), dtype=np.float32)
    x = np.floor(xy[:, 0] * scale_x).astype(int)
    y = np.floor(xy[:, 1] * scale_y).astype(int)
    keep = (0 <= x) & (x < width) & (0 <= y) & (y < height)
    density = np.zeros((height, width), dtype=np.uint32)
    np.add.at(density, (y[keep], x[keep]), 1)
    intensity = np.log1p(density.astype(np.float32))
    positive = intensity[intensity > 0]
    if positive.size:
        high = max(float(np.percentile(positive, 99.5)), 1e-12)
        intensity = np.clip(intensity / high, 0, 1)
    return intensity


def _overlay_centroids_on_image(
    image: np.ndarray,
    xy_full_pixels: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    intensity = _centroid_intensity(
        xy_full_pixels,
        width,
        height,
        scale_x,
        scale_y,
    )
    base = np.asarray(image, dtype=np.float32) / 255.0
    dimmed = base * (1.0 - 0.35 * intensity[..., None])
    cyan = np.zeros_like(dimmed)
    cyan[..., 1] = 0.95
    cyan[..., 2] = 1.0
    blended = dimmed * (1.0 - 0.75 * intensity[..., None]) + cyan * (
        0.75 * intensity[..., None]
    )
    return np.round(np.clip(blended, 0, 1) * 255).astype(np.uint8)


def _reference_preview(
    sample: Any,
    xy_full_pixels: np.ndarray,
    max_dimension: int,
) -> dict[str, Any]:
    morphology_mip = _find_morphology_mip(sample)
    if morphology_mip is not None:
        preview = _load_image_preview(morphology_mip, max_dimension)
        preview["image"] = _overlay_centroids_on_image(
            preview["image"],
            xy_full_pixels,
            preview["scale_x"],
            preview["scale_y"],
        )
        preview.update(
            {
                "xy_full_pixels": xy_full_pixels,
                "source_kind": "morphology_mip_with_centroids",
                "source_label": "Xenium morphology MIP + centroids",
                "morphology_mip": str(morphology_mip),
            }
        )
        return preview

    preview = _centroid_preview(xy_full_pixels, max_dimension)
    preview.update(
        {
            "source_kind": "native_xenium_centroid_density",
            "source_label": "Native Xenium centroid density",
            "morphology_mip": "",
        }
    )
    return preview


def _centroid_preview(
    xy_full_pixels: np.ndarray,
    max_dimension: int,
) -> dict[str, Any]:
    finite = np.isfinite(xy_full_pixels).all(axis=1)
    xy = xy_full_pixels[finite]
    if not len(xy):
        raise ValueError("No finite Xenium centroids")
    full_width = max(1, int(math.ceil(xy[:, 0].max())) + 1)
    full_height = max(1, int(math.ceil(xy[:, 1].max())) + 1)
    scale = min(1.0, max_dimension / max(full_width, full_height))
    width = max(1, int(math.ceil(full_width * scale)))
    height = max(1, int(math.ceil(full_height * scale)))
    x = np.clip(np.floor(xy[:, 0] * scale).astype(int), 0, width - 1)
    y = np.clip(np.floor(xy[:, 1] * scale).astype(int), 0, height - 1)
    density = np.zeros((height, width), dtype=np.uint32)
    np.add.at(density, (y, x), 1)
    intensity = np.log1p(density.astype(np.float32))
    positive = intensity[intensity > 0]
    if positive.size:
        high = max(float(np.percentile(positive, 99.5)), 1e-12)
        intensity = np.clip(intensity / high, 0, 1)
    rgb = np.ones((height, width, 3), dtype=np.float32)
    rgb[..., 0] -= intensity
    rgb[..., 1] -= intensity * 0.20
    rgb = np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return {
        "image": rgb,
        "full_height": full_height,
        "full_width": full_width,
        "scale_x": width / full_width,
        "scale_y": height / full_height,
        "xy_full_pixels": xy_full_pixels,
    }


def load_alignment_inputs(
    root: str | Path,
    sample_name: str,
    max_dimension: int = 1600,
    source_kind: str | None = None,
    use_morphology_mip: bool = True,
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    sample = pipeline.build_sample_paths(root / sample_name)
    if not sample.base.is_dir():
        raise FileNotFoundError(sample.base)
    pipeline.validate_sample(sample, require_alignment=False)
    if sample.he_image is None:
        raise FileNotFoundError("H&E OME-TIFF was not resolved")
    cells, _ = pipeline.load_cells(sample)
    pixel_size_um = pipeline.experiment_pixel_size(sample)
    xy_um = cells[["cell_centroid_x", "cell_centroid_y"]].to_numpy(float)
    xy_full_pixels = xy_um / pixel_size_um
    if source_kind == "native_xenium_centroid_density" or not use_morphology_mip:
        source = _centroid_preview(xy_full_pixels, max_dimension)
        source.update(
            {
                "source_kind": "native_xenium_centroid_density",
                "source_label": "Native Xenium centroid density",
                "morphology_mip": "",
            }
        )
    else:
        source = _reference_preview(sample, xy_full_pixels, max_dimension)
        if (
            source_kind == "morphology_mip_with_centroids"
            and source["source_kind"] != source_kind
        ):
            raise FileNotFoundError("morphology_mip.ome.tif was not found")
    he = _load_he_preview(sample.he_image, max_dimension)
    return {
        "root": root,
        "sample": sample,
        "cells": cells,
        "pixel_size_um": pixel_size_um,
        "source": source,
        "he": he,
        "max_dimension": max_dimension,
    }


def _png_data_url(image: np.ndarray) -> str:
    buffer = BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def show_landmark_selector(
    root: str | Path,
    sample_name: str,
    *,
    n_points: int = 8,
    max_dimension: int = 1000,
    use_morphology_mip: bool = True,
) -> Any:
    """Show a no-extension HTML landmark picker inside JupyterLab."""
    if n_points < 3:
        raise ValueError("At least three landmark pairs are required")
    from IPython.display import HTML, display

    print(f"[{sample_name}] Preparing landmark selector...", flush=True)
    print(
        f"[{sample_name}] Loading previews at max_dimension={max_dimension}",
        flush=True,
    )
    inputs = load_alignment_inputs(
        root,
        sample_name,
        max_dimension,
        use_morphology_mip=use_morphology_mip,
    )
    print(
        f"[{sample_name}] Reference: {inputs['source']['source_label']}",
        flush=True,
    )
    print(f"[{sample_name}] Encoding browser images...", flush=True)
    element_id = "janesick_landmarks_" + uuid.uuid4().hex
    config = {
        "elementId": element_id,
        "sampleName": sample_name,
        "nPairs": int(n_points),
        "maxDimension": int(max_dimension),
        "leftUrl": _png_data_url(inputs["source"]["image"]),
        "rightUrl": _png_data_url(inputs["he"]["image"]),
        "sourceKind": inputs["source"]["source_kind"],
        "sourceLabel": inputs["source"]["source_label"],
        "morphologyMip": inputs["source"].get("morphology_mip", ""),
    }
    html = r"""
<div id="__ELEMENT_ID__" style="font-family:sans-serif;max-width:1700px">
  <h3>__SAMPLE_NAME__: paired landmark selector</h3>
  <p>Click each internal landmark on the left Xenium reference, then the same
  structure on the right H&amp;E. Use the mouse wheel to zoom, drag to pan, and
  rotate the H&amp;E view when it helps identify corresponding anatomy.</p>
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
      sample_name: config.sampleName,
      max_dimension: config.maxDimension,
      source_coordinate_space: 'xenium_reference_preview',
      source_reference_kind: config.sourceKind,
      source_reference_label: config.sourceLabel,
      morphology_mip: config.morphologyMip,
      target_coordinate_space: 'he_preview_original',
      source_points: source,
      target_points: target,
      target_points_display_xy: target.map(imageToDisplay),
      he_display_orientation: {
        rotation_degrees_ccw: orientation.rotation * 90,
        flip_horizontal: orientation.flipH,
        flip_vertical: orientation.flipV
      }
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

  function installInteractions(view) {
    view.canvas.addEventListener('wheel', event => {
      event.preventDefault();
      const p = canvasCoords(event, view.canvas);
      const before = [(p[0] - view.offsetX) / view.scale,
                      (p[1] - view.offsetY) / view.scale];
      const factor = event.deltaY < 0 ? 1.25 : 0.8;
      view.scale = Math.max(1, Math.min(32, view.scale * factor));
      view.offsetX = p[0] - before[0] * view.scale;
      view.offsetY = p[1] - before[1] * view.scale;
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
        view.offsetX += dx;
        view.offsetY += dy;
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
        .replace("__SAMPLE_NAME__", sample_name)
        .replace("__CONFIG_JSON__", json.dumps(config))
    )
    widget = HTML(html)
    display(widget)
    print(f"[{sample_name}] Landmark selector displayed.", flush=True)
    return widget


def review_landmarks(
    root: str | Path,
    landmarks: str | dict[str, Any],
    *,
    output_dir: str | Path | None = None,
    maximum_centroids: int = 60_000,
) -> dict[str, Any]:
    """Fit a Xenium-like arbitrary-angle similarity transform and review it."""
    record = json.loads(landmarks) if isinstance(landmarks, str) else landmarks
    sample_name = record.get("sample_name")
    if not sample_name:
        raise ValueError("Landmark JSON has no sample_name")
    if record.get("source_coordinate_space") not in {
        "xenium_reference_preview",
        "native_xenium_centroid_preview",
    }:
        raise ValueError("Landmarks were collected in an unsupported coordinate space")
    max_dimension = int(record.get("max_dimension", 1600))
    source_kind = record.get("source_reference_kind")
    if record.get("source_coordinate_space") == "native_xenium_centroid_preview":
        source_kind = "native_xenium_centroid_density"
    inputs = load_alignment_inputs(
        root,
        sample_name,
        max_dimension,
        source_kind=source_kind,
    )
    source_points = np.asarray(record["source_points"], dtype=float)
    target_points = np.asarray(record["target_points"], dtype=float)
    if source_points.shape != target_points.shape or source_points.ndim != 2:
        raise ValueError("Source and target landmarks must be matching N x 2 arrays")
    if len(source_points) < 3:
        raise ValueError("At least three landmark pairs are required")

    source_image = inputs["source"]["image"]
    he_image = inputs["he"]["image"]
    selected = fit_xenium_like_similarity(
        source_points,
        target_points,
        allow_reflection=True,
    )

    xy_preview = inputs["source"]["xy_full_pixels"] * np.array(
        [inputs["source"]["scale_x"], inputs["source"]["scale_y"]]
    )
    if len(xy_preview) > maximum_centroids:
        indices = np.random.default_rng(1).choice(
            len(xy_preview), maximum_centroids, replace=False
        )
        xy_preview = xy_preview[indices]

    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else inputs["sample"].base / "manual_alignment_reviews"
    )
    destination.mkdir(parents=True, exist_ok=True)
    figure_path = destination / f"{sample_name}_manual_alignment_review.png"
    preview_matrix = selected["matrix_xenium_preview_pixel_to_he_preview_pixel"]
    transformed = transform_xy(preview_matrix, xy_preview)
    predicted = selected["predicted_target_points_he_preview_xy"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 9), constrained_layout=True)
    axes[0].imshow(source_image)
    axes[0].scatter(
        source_points[:, 0],
        source_points[:, 1],
        s=50,
        facecolors="none",
        edgecolors="red",
        linewidths=1.5,
    )
    for index, point in enumerate(source_points, start=1):
        axes[0].text(point[0] + 6, point[1] - 6, str(index), color="red", fontsize=9)
    axes[0].set_title(inputs["source"]["source_label"])
    axes[0].set_xlim(0, source_image.shape[1])
    axes[0].set_ylim(source_image.shape[0], 0)
    axes[0].axis("off")

    axes[1].imshow(he_image)
    axes[1].scatter(
        transformed[:, 0],
        transformed[:, 1],
        s=0.2,
        color="deepskyblue",
        alpha=0.45,
        linewidths=0,
        rasterized=True,
    )
    axes[1].scatter(
        target_points[:, 0],
        target_points[:, 1],
        s=55,
        facecolors="none",
        edgecolors="red",
        linewidths=1.5,
    )
    axes[1].scatter(
        predicted[:, 0],
        predicted[:, 1],
        s=36,
        marker="x",
        color="black",
        linewidths=1.2,
    )
    for index, point in enumerate(target_points, start=1):
        axes[1].text(point[0] + 6, point[1] - 6, str(index), color="red", fontsize=9)
    axes[1].set_title(
        "H&E with transformed Xenium centroids\n"
        f"rotation={selected['rotation_degrees_ccw']:.2f} deg, "
        f"reflection={selected['reflection']}, "
        f"RMSE={selected['landmark_rmse_preview_pixels']:.1f} px"
    )
    axes[1].set_xlim(0, he_image.shape[1])
    axes[1].set_ylim(he_image.shape[0], 0)
    axes[1].axis("off")

    fig.suptitle(
        f"{sample_name}: manual landmark alignment review\n"
        "red=clicked landmarks; black x=predicted H&E landmark",
        fontsize=15,
    )
    fig.savefig(figure_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    source_full_to_preview = diagonal_matrix(
        inputs["source"]["scale_x"], inputs["source"]["scale_y"]
    )
    he_preview_to_full = diagonal_matrix(
        1 / inputs["he"]["scale_x"], 1 / inputs["he"]["scale_y"]
    )
    full_matrix = he_preview_to_full @ preview_matrix @ source_full_to_preview
    serializable_transform = {
        key: value
        for key, value in selected.items()
        if key not in {
            "matrix_xenium_preview_pixel_to_he_preview_pixel",
            "predicted_target_points_he_preview_xy",
        }
    }
    serializable_transform.update(
        {
            "predicted_target_points_he_preview_xy": predicted.tolist(),
            "matrix_xenium_preview_pixel_to_he_preview_pixel": preview_matrix.tolist(),
            "matrix_xenium_full_pixel_to_he_full_pixel": full_matrix.tolist(),
            "matrix_xenium_um_to_he_full_pixel": (
                full_matrix
                @ diagonal_matrix(
                    1 / inputs["pixel_size_um"],
                    1 / inputs["pixel_size_um"],
                )
            ).tolist(),
        }
    )

    output = {
        "sample_name": sample_name,
        "sample_dir": str(inputs["sample"].base),
        "status": "reviewed_needs_final_selection",
        "registration_method": (
            "paired_landmarks_xenium_like_similarity_uniform_scale_"
            "rotation_translation_optional_reflection"
        ),
        "source_reference_kind": inputs["source"]["source_kind"],
        "source_reference_label": inputs["source"]["source_label"],
        "morphology_mip": inputs["source"].get("morphology_mip", ""),
        "source_points_xenium_reference_preview_xy": source_points.tolist(),
        "target_points_he_preview_xy": target_points.tolist(),
        "preview_max_dimension": max_dimension,
        "xenium_pixel_size_um": inputs["pixel_size_um"],
        "he_image": str(inputs["sample"].he_image),
        "selected_rotation_degrees_ccw": selected["rotation_degrees_ccw"],
        "selected_reflection": selected["reflection"],
        "selected_uniform_scale_preview_pixels": selected[
            "uniform_scale_preview_pixels"
        ],
        "selected_landmark_rmse_preview_pixels": selected[
            "landmark_rmse_preview_pixels"
        ],
        "selected_transform": serializable_transform,
        "review_figure": str(figure_path),
        "warning": "Select by anatomical overlay, not RMSE alone.",
    }
    candidate_path = destination / f"{sample_name}_manual_alignment_candidates.json"
    candidate_path.write_text(json.dumps(output, indent=2) + "\n")
    output["candidate_json"] = str(candidate_path)
    print(
        "Fitted transform: "
        f"rotation={output['selected_rotation_degrees_ccw']:.2f} deg, "
        f"reflection={output['selected_reflection']}, "
        f"RMSE={output['selected_landmark_rmse_preview_pixels']:.1f} px"
    )
    print(f"Review figure: {figure_path}")
    print(f"Candidate JSON: {candidate_path}")
    return output


def select_manual_alignment(
    candidate_json: str | Path,
) -> Path:
    """Save one reviewed candidate where the tiling pipeline will find it."""
    candidate_path = Path(candidate_json).expanduser().resolve()
    record = json.loads(candidate_path.read_text())
    if "selected_transform" not in record:
        raise ValueError("Candidate JSON does not contain a selectable transform")
    selected = record["selected_transform"]
    output = {
        key: value
        for key, value in record.items()
        if key != "selected_transform"
    }
    output.update(
        {
            "status": "selected_needs_final_tile_qc",
            "selected_rotation_degrees_ccw": selected["rotation_degrees_ccw"],
            "selected_reflection": selected["reflection"],
            "selected_uniform_scale_preview_pixels": selected[
                "uniform_scale_preview_pixels"
            ],
            "selected_landmark_rmse_preview_pixels": selected[
                "landmark_rmse_preview_pixels"
            ],
            "matrix_xenium_full_pixel_to_he_full_pixel": selected[
                "matrix_xenium_full_pixel_to_he_full_pixel"
            ],
            "matrix_xenium_um_to_he_full_pixel": selected[
                "matrix_xenium_um_to_he_full_pixel"
            ],
            "source_candidate_json": str(candidate_path),
        }
    )
    sample_dir = Path(record["sample_dir"]).expanduser().resolve()
    if not sample_dir.is_dir():
        raise FileNotFoundError(sample_dir)
    xenium_to_he = np.asarray(
        selected["matrix_xenium_full_pixel_to_he_full_pixel"],
        dtype=float,
    )
    imagealignment_path = sample_dir / (
        f"{record['sample_name']}_he_imagealignment.csv"
    )
    he_to_xenium = write_xenium_explorer_imagealignment_csv(
        imagealignment_path,
        xenium_to_he,
    )
    output.update(
        {
            "matrix_he_full_pixel_to_xenium_full_pixel": (
                he_to_xenium.tolist()
            ),
            "xenium_explorer_imagealignment_csv": str(imagealignment_path),
            "xenium_explorer_matrix_convention": (
                "3x3 row-major affine matrix mapping H&E full-resolution "
                "image pixels to Xenium full-resolution pixels; final row "
                "is [0, 0, 1]."
            ),
        }
    )
    output_path = sample_dir / f"{record['sample_name']}_manual_alignment.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Selected manual alignment: {output_path}")
    print(f"Xenium Explorer-style matrix: {imagealignment_path}")
    return output_path
