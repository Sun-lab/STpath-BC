"""CPU-only eligible-tile inventory and balanced assignments; no job submission."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from . import he_tissue, tcga_brca_images as pilot
from .tcga_brca_cohort import check_source_stat, select_inputs


TILE_COLUMNS = ["tile_index", "tile_id", "slide_id", "patient_id", "candidate", "nominal_size_um",
                "tile_row", "tile_col", "x0_px", "y0_px", "x1_px", "y1_px",
                "tile_width_um", "tile_height_um", "tissue_fraction_thumbnail_estimate"]

# The batch refactor changes scheduling/status writes only. Retain the frozen
# config hash for inventories made by the original notebook implementation.
PRE_BATCH_CODE_SHA256 = "9d7fea4d0b194d0fc1621e338236f5aa487ad7a8cdf44c546ef589af4e472058"


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def freeze_config(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Settings/source identity changed; choose a new output folder: {path.parent}")
    else:
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)


def checked_record(output, slide, config_sha256):
    record = json.loads((output / "slides" / f"{slide}.json").read_text())
    if record["status"] != "complete" or record["config_sha256"] != config_sha256 or record["slide_id"] != slide:
        raise ValueError(f"Invalid tile inventory completion record: {slide}")
    check_source_stat(record["metadata"])
    path = output / "slides" / f"{slide}.csv.gz"
    if file_hash(path) != record["tiles_sha256"]:
        raise ValueError(f"Tile coordinate file changed: {path}")
    return record


def count_valid_tiles(input_dir, audit_csv, slide_ids, output_dir, *, crop_size_um=75, min_tissue=0.60,
                      task_index=None, prepare_only=False):
    """Save each selected slide's full eligible grid, resuming completed inventories.

    Eligibility matches cohort preparation: the same thumbnail, tissue mask,
    rounded physical grid and partial-edge exclusion. No crop PNGs are created.
    A task_index processes just that slide and writes only slide-specific files.
    prepare_only freezes metadata without reading TIFF pixels. The collector is
    the sole writer of global summaries for batch runs.
    """
    if not math.isfinite(crop_size_um) or crop_size_um <= 0:
        raise ValueError("Crop size must be positive and finite")
    if not math.isfinite(min_tissue) or not 0 < min_tissue <= 1:
        raise ValueError("Tissue cutoff must be in (0, 1]")
    if not slide_ids:
        raise ValueError("Select slides explicitly for the tile inventory")
    source, audit_path, output = (Path(p).resolve() for p in (input_dir, audit_csv, output_dir))
    if output == source or source in output.parents:
        raise ValueError("Keep tile inventories outside the input TIFF directory")
    paths = select_inputs(source, slide_ids)
    with audit_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    audit = {r["slide_id"]: r for r in rows}
    if len(audit) != len(rows) or any(p.stem not in audit for p in paths):
        raise ValueError("Audit must contain exactly one record for every selected slide")
    config = dict(version=1, input_dir=str(source), audit_csv=str(audit_path), audit_sha256=file_hash(audit_path),
                  slides=[p.stem for p in paths], crop_size_um=float(crop_size_um), min_tissue=float(min_tissue),
                  mask_method=he_tissue.MASK_METHOD_VERSION, partial_edges="exclude", order="slide_id_then_row_major",
                  code_sha256={p.name: file_hash(p) for p in (Path(__file__), Path(pilot.__file__), Path(he_tissue.__file__))})
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "inventory_config.json"
    if not config_path.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty inventory folder: {output}")
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved.get("code_sha256", {}).get(Path(__file__).name) == PRE_BATCH_CODE_SHA256:
            config["code_sha256"][Path(__file__).name] = PRE_BATCH_CODE_SHA256
    freeze_config(config_path, config)
    config_hash = file_hash(config_path)
    (output / "slides").mkdir(exist_ok=True)
    if prepare_only:
        return config
    if task_index is not None:
        if isinstance(task_index, bool) or not isinstance(task_index, int) or not 0 <= task_index < len(paths):
            raise ValueError("Task index is outside the frozen slide list")
        paths = [paths[task_index]]
    else:
        (output / "count_summary.json").unlink(missing_ok=True)
    counts = []
    for number, path in enumerate(paths, 1):
        slide, started = path.stem, time.monotonic()
        row = dict(slide_id=slide, status="failed", n_valid_tiles=0, n_full_tiles=0,
                   n_partial_tiles=0, reused=False, seconds=0.0, error="")
        print(f"[{number}/{len(paths)}] Counting {slide}", flush=True)
        try:
            marker = output / "slides" / f"{slide}.json"
            if marker.exists():
                record = checked_record(output, slide, config_hash)
                row["reused"] = True
            else:
                metadata = pilot.inspect_slide(path)
                pilot.apply_audited_spacing(metadata, audit[slide])
                if metadata["orientation"] != 1:
                    raise ValueError("Non-default TIFF orientation requires explicit handling")
                thumb = pilot.read_tissue_thumbnail(path, metadata)
                mask, _, _ = he_tissue.color_tissue_mask_rgb(thumb)
                xs, ys, fractions, full, eligible = pilot.eligible_physical_grid(metadata, mask, crop_size_um, min_tissue)
                destination = output / "slides" / f"{slide}.csv.gz"
                temporary = destination.with_suffix(".tmp")
                candidate = f"physical_um_{crop_size_um:g}"
                with gzip.open(temporary, "wt", newline="", encoding="utf-8", compresslevel=1) as stream:
                    writer = csv.writer(stream)
                    writer.writerow(TILE_COLUMNS)
                    for index, flat in enumerate(eligible):
                        r, c = (int(v) for v in np.unravel_index(flat, fractions.shape))
                        x0, x1, y0, y1 = int(xs[c]), int(xs[c+1]), int(ys[r]), int(ys[r+1])
                        writer.writerow([index, f"{slide}__{candidate}__r{r+1}_c{c+1}", slide, metadata["patient_id"],
                                         candidate, crop_size_um, r+1, c+1, x0, y0, x1, y1,
                                         (x1-x0)*metadata["tiling_mpp_x"], (y1-y0)*metadata["tiling_mpp_y"],
                                         float(fractions[r, c])])
                check_source_stat(metadata)
                if file_hash(audit_path) != config["audit_sha256"]:
                    raise ValueError("Audit changed during counting; choose a new inventory folder")
                temporary.replace(destination)
                record = dict(status="complete", slide_id=slide, config_sha256=config_hash, metadata=metadata,
                              n_valid_tiles=len(eligible), n_full_tiles=int(full.sum()),
                              n_partial_tiles=int(full.size-full.sum()), tiles_sha256=file_hash(destination))
                write_json(marker, record)
            row.update({k: record[k] for k in ("status", "n_valid_tiles", "n_full_tiles", "n_partial_tiles")})
            print(f"  {row['n_valid_tiles']:,} valid full tiles" + (" (reused)" if row["reused"] else ""), flush=True)
        except (OSError, ValueError, KeyError, RuntimeError, ImportError) as error:
            row["error"] = f"{type(error).__name__}: {error}"
            print(f"  FAILED: {row['error']}", flush=True)
        row["seconds"] = round(time.monotonic()-started, 3)
        counts.append(row)
        write_json(output / "slides" / f"{slide}.status.json", row)
        if task_index is None:
            pilot.write_csv(output / "slide_counts.csv", counts)
    failed = sum(r["status"] != "complete" for r in counts)
    summary = dict(status="incomplete" if failed else "complete", n_slides=len(paths), n_failed_slides=failed,
                   n_zero_tile_slides=sum(r["status"] == "complete" and r["n_valid_tiles"] == 0 for r in counts),
                   n_valid_tiles=sum(r["n_valid_tiles"] for r in counts), config_sha256=config_hash,
                   crop_size_um=float(crop_size_um), min_tissue=float(min_tissue),
                   interpretation="Full grid tiles passing the thumbnail-estimated tissue threshold; no subsampling, crops, encoders or predictions.")
    if task_index is None:
        write_json(output / "count_summary.json", summary)
    return summary


def collect_counts(output_dir):
    """Check per-slide inventories without decoding TIFFs; publish cohort status."""
    output = Path(output_dir).resolve()
    config_path = output / "inventory_config.json"
    config = json.loads(config_path.read_text())
    config_hash = file_hash(config_path)
    if file_hash(config["audit_csv"]) != config["audit_sha256"]:
        raise ValueError("Audit changed since counting; create a new inventory")
    counts = []
    for slide in config["slides"]:
        row = dict(slide_id=slide, status="missing", n_valid_tiles=0, n_full_tiles=0,
                   n_partial_tiles=0, error="")
        try:
            if (output / "slides" / f"{slide}.json").exists():
                record = checked_record(output, slide, config_hash)
                row.update({k: record[k] for k in ("status", "n_valid_tiles", "n_full_tiles", "n_partial_tiles")})
            else:
                status = output / "slides" / f"{slide}.status.json"
                if status.exists():
                    previous = json.loads(status.read_text())
                    row.update(status="failed", error=previous.get("error") or "Completion marker missing")
        except (OSError, ValueError, KeyError) as error:
            row.update(status="invalid", error=f"{type(error).__name__}: {error}")
        counts.append(row)
    summary = dict(status="complete" if all(r["status"] == "complete" for r in counts) else "incomplete",
                   n_slides=len(counts), n_failed_slides=sum(r["status"] in {"failed", "invalid"} for r in counts),
                   n_missing_slides=sum(r["status"] == "missing" for r in counts),
                   n_zero_tile_slides=sum(r["status"] == "complete" and r["n_valid_tiles"] == 0 for r in counts),
                   n_valid_tiles=sum(r["n_valid_tiles"] for r in counts), config_sha256=config_hash,
                   crop_size_um=config["crop_size_um"], min_tissue=config["min_tissue"],
                   interpretation="Collected completed slide inventories; missing, failed or changed outputs block grouping.")
    pilot.write_csv(output / "slide_counts.csv", counts)
    write_json(output / "count_summary.json", summary)
    return summary


def assign_groups(output_dir, n_groups=20):
    """Partition the complete inventory into disjoint, contiguous tile-index ranges.

    A range [tile_start, tile_stop) addresses data rows in its slide's CSV.gz,
    excluding the header. Slides may span groups. Every eligible tile is assigned
    once; group sizes differ by at most one. Range records avoid copying millions
    of coordinate rows into a second set of files.
    """
    if isinstance(n_groups, bool) or not isinstance(n_groups, int) or n_groups < 1:
        raise ValueError("n_groups must be a positive integer")
    output = Path(output_dir).resolve()
    config_path = output / "inventory_config.json"
    config = json.loads(config_path.read_text())
    config_hash = file_hash(config_path)
    count_summary = json.loads((output / "count_summary.json").read_text())
    if count_summary["status"] != "complete" or count_summary["config_sha256"] != config_hash:
        raise ValueError("Finish all slide counts before grouping; inspect slide_counts.csv and rerun counting")
    if file_hash(config["audit_csv"]) != config["audit_sha256"]:
        raise ValueError("Audit changed since counting; create a new inventory")
    records = [checked_record(output, slide, config_hash) for slide in config["slides"]]
    total = sum(r["n_valid_tiles"] for r in records)
    if total != count_summary["n_valid_tiles"]:
        raise ValueError("Tile counts disagree with the completed inventory")
    if total < n_groups:
        raise ValueError(f"Only {total} eligible tiles; choose at most {total} groups")
    identity = dict(config_sha256=config_hash, n_groups=n_groups,
                    slide_tiles_sha256={r["slide_id"]: r["tiles_sha256"] for r in records})
    group_dir = output / f"groups_{n_groups}"
    group_dir.mkdir(exist_ok=True)
    freeze_config(group_dir / "group_config.json", identity)
    (group_dir / "group_plan.json").unlink(missing_ok=True)
    base, remainder = divmod(total, n_groups)
    boundaries = np.cumsum([0] + [base + (i < remainder) for i in range(n_groups)]).tolist()
    assignments, offset, group_index = [], 0, 0
    for record in records:
        stop = offset + record["n_valid_tiles"]
        cursor = offset
        while cursor < stop:
            end = min(stop, boundaries[group_index+1])
            assignments.append(dict(group_index=group_index, slide_id=record["slide_id"],
                                    tile_start=cursor-offset, tile_stop=end-offset, n_tiles=end-cursor,
                                    tile_manifest=str(output / "slides" / f"{record['slide_id']}.csv.gz"),
                                    tiles_sha256=record["tiles_sha256"]))
            cursor = end
            if cursor == boundaries[group_index+1]:
                group_index += 1
        offset = stop
    groups = [dict(group_index=i, n_tiles=boundaries[i+1]-boundaries[i],
                   n_slides=sum(r["group_index"] == i for r in assignments)) for i in range(n_groups)]
    assert sum(r["n_tiles"] for r in assignments) == total
    pilot.write_csv(group_dir / "group_assignments.csv", assignments)
    pilot.write_csv(group_dir / "group_summary.csv", groups)
    summary = dict(status="ready_for_prediction", crop_size_um=config["crop_size_um"], n_groups=n_groups,
                   n_slides=len(records), n_zero_tile_slides=sum(r["n_valid_tiles"] == 0 for r in records),
                   n_valid_tiles=total, min_group_tiles=base, max_group_tiles=base+bool(remainder),
                   n_split_slides=sum(sum(a["slide_id"] == r["slide_id"] for a in assignments) > 1 for r in records),
                   group_dir=str(group_dir), config_sha256=config_hash,
                   assignment_sha256=file_hash(group_dir / "group_assignments.csv"),
                   summary_sha256=file_hash(group_dir / "group_summary.csv"),
                   interpretation="Assignments only. No GPU jobs submitted. Existing per-slide runner does not consume these ranges. Tile ranges are zero-based, stop-exclusive data-row indices.")
    write_json(group_dir / "group_plan.json", summary)
    return summary
