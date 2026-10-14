"""Compare a few TIFF crops with matching and highest-resolution IDC images."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw

from .he_tissue import color_tissue_mask_rgb
from .tcga_brca_header_audit import DICOMWEB, dicom_spacing
from .tcga_brca_images import image_array, inspect_slide, rgb_region, write_csv


def image_layout(record: dict, slide_id: str) -> dict:
    """Limit the pilot to complete, single-plane RGB DICOM tile matrices."""
    c, metadata = record["candidate"], record["dicom_metadata"]
    row = dict(slide_id=slide_id, width_px=c["TotalPixelMatrixColumns"],
               height_px=c["TotalPixelMatrixRows"])
    row.update({f"matched_{key}": c[key] for key in
                ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID")})
    if not all(re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", row[f"matched_{key}"])
               for key in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID")):
        raise ValueError("Invalid DICOM UID")
    mx, my = dicom_spacing(metadata, row)
    d = metadata[0]
    value = lambda tag: d[tag]["Value"][0]
    if (value("00209311") != "TILED_FULL" or value("00480303") != 1 or
            len(d["00480105"]["Value"]) != 1 or "00209161" in d):
        raise ValueError("Pilot requires one optical path/plane and no DICOM concatenation")
    if (value("00280002"), value("00280100"), value("00280103"), value("00280004")) != (3, 8, 0, "RGB"):
        raise ValueError("Pilot requires unsigned 8-bit RGB DICOM images")
    tw, th = int(value("00280011")), int(value("00280010"))
    if not 0 < tw <= 1024 or not 0 < th <= 1024:
        raise ValueError("Unsupported DICOM tile size")
    cols = math.ceil(row["width_px"] / tw)
    frames = int(value("00280008"))
    if frames != cols * math.ceil(row["height_px"] / th):
        raise ValueError("Frame count differs from complete tile matrix")
    row.update(mpp_x=mx, mpp_y=my, tile_width=tw, tile_height=th,
               frame_columns=cols, n_frames=frames, metadata=d)
    row["url"] = (f"{DICOMWEB}/studies/{c['StudyInstanceUID']}/series/"
                  f"{c['SeriesInstanceUID']}/instances/{c['SOPInstanceUID']}")
    return row


def validate_pair(matched: dict, highest: dict) -> None:
    """Require a shared image coordinate system before comparing locations."""
    for key in ("slide_id", "matched_StudyInstanceUID", "matched_SeriesInstanceUID"):
        if matched[key] != highest[key]:
            raise ValueError(f"Source pair differs in {key}")
    # The origin is the center of the first pixel, not the image's outer edge.
    for tag in ("00480008", "00480102", "00200052"):
        if tag not in matched["metadata"] or matched["metadata"][tag] != highest["metadata"].get(tag):
            raise ValueError(f"Source coordinate systems differ or are missing: {tag}")
    if highest["mpp_x"] > matched["mpp_x"] or highest["mpp_y"] > matched["mpp_y"]:
        raise ValueError("Highest-resolution candidate is coarser than the matched level")


def matched_bounds(bounds: tuple, source: dict, target: dict) -> tuple:
    """Map pixel-edge coordinates with the DICOM pixel-center origin convention."""
    x0, y0, x1, y1 = bounds
    rx, ry = source["mpp_x"] / target["mpp_x"], source["mpp_y"] / target["mpp_y"]
    mapped = tuple(round((v - 0.5) * ratio + 0.5)
                   for v, ratio in zip((x0, y0, x1, y1), (rx, ry, rx, ry)))
    if not (0 <= mapped[0] < mapped[2] <= target["width_px"] and
            0 <= mapped[1] < mapped[3] <= target["height_px"]):
        raise ValueError("Mapped crop crosses the source image boundary")
    return mapped


def read_frame(layout: dict, frame_number: int, cache: Path) -> np.ndarray:
    """Fetch one lossless PNG rendering; cache it without downloading a WSI."""
    if not 1 <= frame_number <= layout["n_frames"]:
        raise ValueError("DICOM frame number outside image")
    folder = cache / layout["matched_SOPInstanceUID"]
    path = folder / f"{frame_number}.png"
    if path.exists():
        raw = path.read_bytes()
    else:
        request = Request(f"{layout['url']}/frames/{frame_number}/rendered",
                          headers={"Accept": "image/png"})
        with urlopen(request, timeout=30) as response:
            if response.headers.get_content_type() != "image/png":
                raise ValueError("DICOMweb did not return a PNG image")
            raw = response.read(4_000_001)
        if len(raw) > 4_000_000:
            raise ValueError("Frame response exceeds pilot limit")
    with Image.open(io.BytesIO(raw)) as image:
        if (image.format != "PNG" or image.mode != "RGB" or
                image.size != (layout["tile_width"], layout["tile_height"])):
            raise ValueError("Rendered frame format/dimensions differ from metadata")
        result = np.array(image)
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return result


def source_crop(layout: dict, bounds: tuple, cache: Path) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    if not (0 <= x0 < x1 <= layout["width_px"] and 0 <= y0 < y1 <= layout["height_px"]):
        raise ValueError("Source crop is outside image")
    tw, th = layout["tile_width"], layout["tile_height"]
    columns = range(x0 // tw, (x1 - 1) // tw + 1)
    rows = range(y0 // th, (y1 - 1) // th + 1)
    if len(columns) * len(rows) > 16:
        raise ValueError("Crop exceeds 16 source frames; reduce physical field of view")
    crop = np.empty((y1-y0, x1-x0, 3), dtype=np.uint8)
    for r in rows:
        for c in columns:
            tile = read_frame(layout, r * layout["frame_columns"] + c + 1, cache)
            left, top, right, bottom = max(x0, c*tw), max(y0, r*th), min(x1, (c+1)*tw), min(y1, (r+1)*th)
            crop[top-y0:bottom-y0, left-x0:right-x0] = tile[top-r*th:bottom-r*th, left-c*tw:right-c*tw]
    return crop


def tissue_centers(thumbnail: np.ndarray, width: int, height: int, crop_width: int, crop_height: int) -> list:
    mask, _, _ = color_tissue_mask_rgb(thumbnail)
    yy, xx = np.nonzero(mask)
    cx, cy = (xx + 0.5)*width/mask.shape[1], (yy + 0.5)*height/mask.shape[0]
    inside = ((cx > crop_width) & (cx < width-crop_width) &
              (cy > crop_height) & (cy < height-crop_height))
    points = list(zip(cx[inside], cy[inside]))
    if len(points) < 3:
        raise ValueError("Fewer than three interior tissue sampling locations")
    return [points[round(q*(len(points)-1))] for q in (0.2, 0.5, 0.8)]


def run_comparison(input_dir: Path, manifest_path: Path, output_dir: Path,
                   *, field_um: float = 70.0) -> dict:
    """Three-slide pilot only; save original crops plus galleries for review."""
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    manifest_path = Path(manifest_path).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"TIFF directory not accessible: {input_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("Write outputs outside the source TIFF directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Use a new empty output directory")
    if not math.isfinite(field_um) or not 10 <= field_um <= 150:
        raise ValueError("Pilot field of view must be between 10 and 150 microns")
    manifest = json.loads(manifest_path.read_text())
    slides = manifest["slides"]
    if len(slides) != 3 or len({s["slide_id"] for s in slides}) != 3:
        raise ValueError("This pilot requires exactly three distinct slides")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / "public_frame_cache"
    rows, slide_results = [], []
    for item in slides:
        slide_id = item["slide_id"]
        result = dict(slide_id=slide_id, status="error", error="")
        try:
            name = item["filename"]
            if Path(name).name != name or Path(name).stem != slide_id:
                raise ValueError("Invalid TIFF filename")
            path = input_dir / name
            header = inspect_slide(path)
            recorded = item["audit_row"]
            if (header["width_px"], header["height_px"], header["file_size_bytes"], header["source_mtime_ns"]) != tuple(
                    int(recorded[k]) for k in ("width_px", "height_px", "file_size_bytes", "source_mtime_ns")):
                raise ValueError("TIFF changed since the supplied audit; repeat the header audit")
            if header["orientation"] != 1:
                raise ValueError("TIFF orientation needs review")
            matched = image_layout(item["matched"], slide_id)
            highest = image_layout(item["highest"], slide_id)
            validate_pair(matched, highest)
            if (header["width_px"], header["height_px"]) != (matched["width_px"], matched["height_px"]):
                raise ValueError("TIFF and DICOM image dimensions differ")
            folder = output_dir / slide_id
            folder.mkdir()
            cw, ch = round(field_um/matched["mpp_x"]), round(field_um/matched["mpp_y"])
            with image_array(path, header["series_index"], 0) as (array, axes, reader):
                step = max(1, math.ceil(max(header["width_px"], header["height_px"])/1024))
                thumb = rgb_region(array, axes, (0, 0, header["width_px"], header["height_px"]), step)
                centers = tissue_centers(thumb, header["width_px"], header["height_px"], cw, ch)
                overview = Image.fromarray(thumb)
                draw = ImageDraw.Draw(overview)
                gallery = Image.new("RGB", (960, 3*360), "white")
                gd = ImageDraw.Draw(gallery)
                for anchor, (cx, cy) in enumerate(centers, 1):
                    x0, y0 = round(cx-cw/2), round(cy-ch/2)
                    bounds = (x0, y0, x0+cw, y0+ch)
                    native_bounds = matched_bounds(bounds, matched, highest)
                    local = rgb_region(array, axes, bounds)
                    public = source_crop(matched, bounds, cache)
                    native = source_crop(highest, native_bounds, cache)
                    difference = np.abs(local.astype(np.int16)-public.astype(np.int16))
                    crops = [("stored_tiff", local), ("public_matched_level", public), ("public_highest_level", native)]
                    for col, (label, crop) in enumerate(crops):
                        image = Image.fromarray(crop)
                        image.save(folder / f"anchor_{anchor:02d}__{label}.png")
                        # Display-only resizing: raw crops above remain at their native pixel size.
                        gallery.paste(image.resize((304, 304), Image.Resampling.BICUBIC), (col*320+8, (anchor-1)*360+48))
                        gd.text((col*320+8, (anchor-1)*360+6), f"{anchor}: {label}", fill="black")
                        gd.text((col*320+8, (anchor-1)*360+25), f"{crop.shape[1]} x {crop.shape[0]} pixels", fill="black")
                    px, py = cx/header["width_px"]*thumb.shape[1], cy/header["height_px"]*thumb.shape[0]
                    draw.ellipse((px-5, py-5, px+5, py+5), outline="yellow", width=2)
                    draw.text((px+7, py), str(anchor), fill="yellow")
                    rows.append(dict(slide_id=slide_id, anchor=anchor, requested_field_um=field_um,
                                     tiff_bounds=bounds, highest_bounds=native_bounds,
                                     tiff_mpp_x=matched["mpp_x"], tiff_mpp_y=matched["mpp_y"],
                                     highest_mpp_x=highest["mpp_x"], highest_mpp_y=highest["mpp_y"],
                                     tiff_field_width_um=cw*matched["mpp_x"], tiff_field_height_um=ch*matched["mpp_y"],
                                     highest_field_width_um=(native_bounds[2]-native_bounds[0])*highest["mpp_x"],
                                     highest_field_height_um=(native_bounds[3]-native_bounds[1])*highest["mpp_y"],
                                     tiff_vs_public_mae=float(difference.mean()),
                                     tiff_vs_public_max_difference=int(difference.max()),
                                     tiff_vs_public_exact_fraction=float((difference == 0).all(axis=2).mean()),
                                     crop_tissue_fraction=float(color_tissue_mask_rgb(local)[0].mean()),
                                     matched_SOPInstanceUID=matched["matched_SOPInstanceUID"],
                                     highest_SOPInstanceUID=highest["matched_SOPInstanceUID"]))
                    print(f"{slide_id}: location {anchor}/3", flush=True)
                gallery.save(folder / "comparison.png")
                overview.save(folder / "overview.png")
            result.update(status="complete", gallery=str(folder/"comparison.png"), reader=reader)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            print(f"{slide_id}: {result['error']}", flush=True)
        slide_results.append(result)
        write_csv(output_dir / "crop_comparison.csv", rows)
        (output_dir / "slide_results.json").write_text(json.dumps(slide_results, indent=2)+"\n")
    summary = dict(status="complete" if all(s["status"] == "complete" for s in slide_results) else "partial",
                   output_dir=str(output_dir), field_um=field_um, n_slides=len(slides), n_locations=len(rows),
                   manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                   slides=slide_results,
                   interpretation="Public tiles are PNG renderings. Inspect structure and alignment; color rendering can differ from TIFF conversion. Pixel differences are descriptive, with no automatic pass threshold. Exact matches establish agreement only at sampled locations. TIFF scale remains inferred. No encoder transforms or model predictions were run.")
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(f"Comparison {summary['status']}: {output_dir}", flush=True)
    return summary
