#!/usr/bin/env python3

import os
import csv
import math
from datetime import datetime

import numpy as np
import pydicom
from PIL import Image


# input and output path
base_dir = "/fh/working/sun_w/zsui/st2image_data/TCGA_BRCA_HE"
output_dir = "/fh/working/sun_w/zsui/st2image_data/TCGA_BRCA_HE_converted"
os.makedirs(output_dir, exist_ok=True)

log_path = os.path.join(output_dir, "processing_log.csv")
LOG_FIELDS = [
    "source_path", "slide_id", "status", "converted",
    "non_pathological", "frozen", "abnormal_tile_numbers",
    "frames", "modality", "series_description",
    "out_path", "timestamp", "error_message"
]

def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def init_log(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()

def read_processed_index(path):
    """Return set of source_path values that have been logged (any terminal state)."""
    processed = set()
    if not os.path.exists(path):
        return processed
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sp = row.get("source_path")
            if sp:
                processed.add(sp)
    return processed

def append_log(row):
    clean = {k: row.get(k, "") for k in LOG_FIELDS}
    with open(log_path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(clean)

def collect_dicoms(root_dir):
    out = []
    for r, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(".dcm"):
                out.append(os.path.join(r, fn))
    return out

def main():
    init_log(log_path)

    # Build todo list by subtracting already-processed files
    all_files = collect_dicoms(base_dir)
    already_processed = read_processed_index(log_path)
    todo_files = [f for f in all_files if f not in already_processed]

    print(f"Found {len(all_files)} DICOM(s); {len(todo_files)} need processing.")

    for fpath in todo_files:
        try:
            # Minimal read for screening and naming
            ds_meta = pydicom.dcmread(fpath, stop_before_pixels=True)
            modality = str(ds_meta.get("Modality", "")).upper()
            desc     = str(ds_meta.get("SeriesDescription", "")).upper()
            frames   = int(ds_meta.get("NumberOfFrames", 0))

            # Slide ID (from (0040,0512) per your original logic)
            raw_id = ds_meta.get((0x0040, 0x0512), "unknownslide")
            slide_id = raw_id.value if hasattr(raw_id, "value") else str(raw_id)
            slide_id = slide_id.replace(" ", "_") if slide_id else "unknownslide"

            out_name = f"{slide_id}.tiff"
            out_path = os.path.join(output_dir, out_name)

            # If TIFF already exists but not logged, mark converted and skip
            if os.path.exists(out_path):
                append_log({
                    "source_path": fpath,
                    "slide_id": slide_id,
                    "status": "converted",
                    "converted": 1,
                    "non_pathological": 0,
                    "frozen": 0,
                    "abnormal_tile_numbers": 0,
                    "frames": frames,
                    "modality": modality,
                    "series_description": desc,
                    "out_path": out_path,
                    "timestamp": now_iso(),
                    "error_message": ""
                })
                print(f"⏭️ Exists → log as converted & skip: {out_name}")
                continue

            # --- Filtering rules with logging ---
            if modality != "SM":
                # shutil.move(fpath, os.path.join(q_non_path, os.path.basename(fpath)))
                os.remove(fpath)
                append_log({
                    "source_path": fpath,
                    "slide_id": slide_id,
                    "status": "non_pathological",
                    "converted": 0,
                    "non_pathological": 1,
                    "frozen": 0,
                    "abnormal_tile_numbers": 0,
                    "frames": frames,
                    "modality": modality,
                    "series_description": desc,
                    "out_path": "",
                    "timestamp": now_iso(),
                    "error_message": ""
                })
                print(f"❌ Deleted non-SM (non-pathological): {os.path.basename(fpath)}")
                continue

            if "FROZEN" in desc:
                # shutil.move(fpath, os.path.join(q_frozen, os.path.basename(fpath)))
                os.remove(fpath)
                append_log({
                    "source_path": fpath,
                    "slide_id": slide_id,
                    "status": "frozen",
                    "converted": 0,
                    "non_pathological": 0,
                    "frozen": 1,
                    "abnormal_tile_numbers": 0,
                    "frames": frames,
                    "modality": modality,
                    "series_description": desc,
                    "out_path": "",
                    "timestamp": now_iso(),
                    "error_message": ""
                })
                print(f"❌ Deleted Frozen: {os.path.basename(fpath)}")
                continue

            if frames < 2000 or frames > 60000:
                # shutil.move(fpath, os.path.join(q_abn, os.path.basename(fpath)))
                os.remove(fpath)
                append_log({
                    "source_path": fpath,
                    "slide_id": slide_id,
                    "status": "abnormal_tile_numbers",
                    "converted": 0,
                    "non_pathological": 0,
                    "frozen": 0,
                    "abnormal_tile_numbers": 1,
                    "frames": frames,
                    "modality": modality,
                    "series_description": desc,
                    "out_path": "",
                    "timestamp": now_iso(),
                    "error_message": ""
                })
                print(f"❌ Deleted abnormal tile number ({frames}): {os.path.basename(fpath)}")
                continue

            # --- Full read & stitching ---
            ds = pydicom.dcmread(fpath)
            tile_h, tile_w = ds.Rows, ds.Columns
            total_w = int(ds.TotalPixelMatrixColumns)
            total_h = int(ds.TotalPixelMatrixRows)

            cols = math.ceil(total_w / tile_w)
            rows = math.ceil(total_h / tile_h)
            print(f"🧩 Stitching {os.path.basename(fpath)} → {cols}×{rows} tiles")

            tiles = ds.pixel_array  # expected shape: (rows*cols, tile_h, tile_w, 3) for RGB
            stitched = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

            for idx, tile in enumerate(tiles):
                r = idx // cols
                c = idx % cols
                stitched[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w, :] = tile

            stitched = stitched[:total_h, :total_w, :]

            # Save as standard TIFF (not BigTIFF)
            Image.fromarray(stitched).save(out_path)
            print(f"✅ Saved: {out_name}")

            # Log success
            append_log({
                "source_path": fpath,
                "slide_id": slide_id,
                "status": "converted",
                "converted": 1,
                "non_pathological": 0,
                "frozen": 0,
                "abnormal_tile_numbers": 0,
                "frames": frames,
                "modality": modality,
                "series_description": desc,
                "out_path": out_path,
                "timestamp": now_iso(),
                "error_message": ""
            })

        except Exception as e:
            # Best-effort metadata for logging on failure
            try:
                # Try to reuse ds_meta; if absent, fill blanks
                slide_id_val = slide_id if 'slide_id' in locals() else ""
                frames_val   = frames   if 'frames' in locals() else ""
                modality_val = modality if 'modality' in locals() else ""
                desc_val     = desc     if 'desc' in locals() else ""
            except Exception:
                slide_id_val = frames_val = modality_val = desc_val = ""

            append_log({
                "source_path": fpath,
                "slide_id": slide_id_val,
                "status": "error",
                "converted": 0,
                "non_pathological": 0,
                "frozen": 0,
                "abnormal_tile_numbers": 0,
                "frames": frames_val,
                "modality": modality_val,
                "series_description": desc_val,
                "out_path": "",
                "timestamp": now_iso(),
                "error_message": str(e)
            })
            print(f"⚠️ Failed {os.path.basename(fpath)}: {e}")

if __name__ == "__main__":
    main()