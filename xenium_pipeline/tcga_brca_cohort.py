"""Predict all eligible tissue tiles locally or through a Slurm slide array."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .hf_auth import ensure_hf_token, login_once
from . import slurm


DEFAULT_ROOT = str(Path(__file__).resolve().parents[1])
DEFAULT_INPUT = os.environ.get("STPATH_TCGA_TIFF_ROOT", str(Path(os.environ.get("XENIUM_ROOT", Path(__file__).resolve().parents[1])) / "data/TCGA_BRCA_HE_filtered_tiff"))
AUDIT_RELATIVE = "tcga_brca_header_audit/20260923T195233_983546Z/tiff_resolution_audit.csv"
COMPLETED_75UM_RELATIVE = "tcga_brca_prediction/all_75um_v1/slides"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_source_stat(record):
    """Check the identity recorded when the upgrade's pixel validation completed."""
    path = Path(record["source_path"])
    info = path.stat()
    if (info.st_size, info.st_mtime_ns) != (int(record["file_size_bytes"]), int(record["source_mtime_ns"])):
        raise ValueError(f"TIFF changed since its recorded upgrade: {path}; use a new audit and output folder")


def upgraded_slide_selection(input_dir, upgrade_dir, audit_csv=None):
    """Select validated, published upgrades, retaining a reason for every audit slide.

    Read only journal/CSV metadata and file stats here. Full pixel verification and
    TIFF SHA256 were performed by the upgrader; inference also checks TIFF headers.
    """
    input_dir, upgrade_dir = Path(input_dir).resolve(), Path(upgrade_dir).resolve()
    audit_csv = Path(audit_csv or upgrade_dir / "tiff_resolution_audit_updated.csv")
    state_dir = upgrade_dir / "upgrade_state"
    if not state_dir.is_dir():
        raise FileNotFoundError(f"Upgrade journals missing: {state_dir}")
    with audit_csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [row["slide_id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Updated audit is empty or contains duplicate slide IDs")
    extra = {p.stem for p in state_dir.glob("*.json")} - set(ids)
    if extra:
        raise ValueError(f"Upgrade journals absent from updated audit: {sorted(extra)}")
    selection, sources = [], {}
    for row in sorted(rows, key=lambda r: r["slide_id"]):
        sid = row["slide_id"]
        journal = state_dir / f"{sid}.json"
        record = json.loads(journal.read_text()) if journal.is_file() else {}
        status = record.get("status", "no_upgrade_record")
        selected = status == "complete"
        selection.append({"slide_id": sid, "upgrade_status": status, "selected": selected})
        if not selected:
            if row.get("spacing_status") == "dicom_xy_verified_conversion_verified":
                raise ValueError(f"Verified upgraded audit has no completed upgrade journal: {sid}")
            continue
        saved = record["new_audit"]
        source = input_dir / row["filename"]
        if (record["slide_id"] != sid or source.stem != sid or source.parent != input_dir or
                Path(record["source_path"]).resolve() != source.resolve() or
                Path(saved["source_path"]).resolve() != source.resolve()):
            raise ValueError(f"Upgrade source identity differs from audit/input directory: {sid}")
        integer_keys = ("width_px", "height_px", "file_size_bytes", "source_mtime_ns", "orientation", "series_index")
        if any(int(row[k]) != int(saved[k]) for k in integer_keys):
            raise ValueError(f"Updated audit differs from completed upgrade: {sid}")
        if (row["spacing_status"] != "dicom_xy_verified_conversion_verified" or
                row["match_status"] != "matched_dimensions" or int(row["n_matching_instances"]) != 1 or
                row["matched_SOPInstanceUID"] != saved["matched_SOPInstanceUID"]):
            raise ValueError(f"Updated audit lacks verified upgrade spacing: {sid}")
        for axis in ("x", "y"):
            key = f"inferred_tiff_mpp_{axis}"
            new, old = float(row[key]), float(record["original_audit"][key])
            if not (0 < new < old and math.isfinite(old) and new == float(saved[key])):
                raise ValueError(f"Completed upgrade has inconsistent or unimproved {axis} spacing: {sid}")
        checksum = record["tiff_sha256"]
        if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError(f"Invalid upgraded TIFF checksum: {sid}")
        sources[sid] = {"source_path": str(source.resolve()), "tiff_sha256": checksum,
                        "file_size_bytes": int(saved["file_size_bytes"]),
                        "source_mtime_ns": int(saved["source_mtime_ns"])}
        check_source_stat(sources[sid])
    return selection, sources


def completed_75um_summary(project_root, slide_id):
    """Return the original 75 um completion marker for this exact slide, if complete."""
    path = Path(project_root) / COMPLETED_75UM_RELATIVE / slide_id / "prediction_summary.json"
    try:
        summary = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read the 75 um prediction summary: {path}") from error
    if not isinstance(summary, dict):
        raise ValueError(f"Invalid 75 um prediction summary: {path}")
    return path if summary.get("status") == "complete" else None


def completed_for_run(config, slide_id):
    """Skip this crop's completed output, optionally accepting the original 75 um run."""
    if config.get("upgraded_sources"):
        check_source_stat(config["upgraded_sources"][slide_id])
    if not config.get("include_completed_75um", False):
        path = completed_75um_summary(config["project_root"], slide_id)
        if path is not None:
            return path, "completed_75um"
    path = Path(config["output_dir"]) / "slides" / slide_id / "prediction_summary.json"
    if path.is_file():
        summary = json.loads(path.read_text())
        if not isinstance(summary, dict):
            raise ValueError(f"Invalid prediction summary: {path}")
        if summary.get("status") == "complete":
            if summary.get("sizes_um") != [config["crop_size_um"]]:
                raise ValueError(f"Completed prediction crop differs from this run: {path}")
            if config.get("upgraded_sources"):
                preparation = json.loads((path.parent / "tile_preparation.json").read_text())
                metadata = preparation.get("source_metadata", [])
                source = config["upgraded_sources"][slide_id]
                if (preparation.get("audit_csv_sha256") != config["audit_sha256"] or
                        len(metadata) != 1 or metadata[0].get("slide_id") != slide_id or
                        any(metadata[0].get(k) != source[k] for k in ("source_path", "file_size_bytes", "source_mtime_ns"))):
                    raise ValueError(f"Completed prediction uses a different source TIFF/audit: {path}")
            return path, "completed_requested_crop"
    return None, None


def select_inputs(input_dir, names=None):
    """Select exact TIFF basenames or stems; retain every slide for each patient."""
    files = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in (".tif", ".tiff"))
    if len({p.stem for p in files}) != len(files):
        raise ValueError("Duplicate TIFF slide stems in input directory")
    lookup = {key: path for path in files for key in (path.name, path.stem)}
    if names is None:
        selected = files
    else:
        missing = [name for name in names if name not in lookup]
        if missing:
            raise ValueError(f"Slides not found; use exact slide IDs or TIFF filenames: {missing}")
        selected = sorted(lookup[name] for name in names)
        if len(set(selected)) != len(selected):
            raise ValueError("Duplicate slides selected")
    if not selected:
        raise ValueError("No TIFF slides selected")
    return selected


def prepare_run(args):
    root, source, output = (p.expanduser().resolve() for p in (args.project_root, args.input_dir, args.output_dir))
    if output == source or source in output.parents:
        raise ValueError("Keep prediction outputs outside the source TIFF directory")
    if not math.isfinite(args.crop_size_um) or args.crop_size_um <= 0:
        raise ValueError("Crop size must be positive and finite")
    if not math.isfinite(args.min_tissue) or not 0 < args.min_tissue <= 1:
        raise ValueError("Tissue cutoff must be in (0, 1]")
    if args.batch_size < 1 or getattr(args, 'max_concurrent', 1) < 1:
        raise ValueError("Batch size and maximum concurrent jobs must be positive")
    names = args.slides
    if args.slides_file:
        names = [line.strip() for line in args.slides_file.read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
    upgraded_sources, upgrade_selection = {}, []
    upgrade_dir = (args.upgrade_dir or source.parent / "TCGA_BRCA_high_resolution").expanduser().resolve()
    audit = (args.audit_csv or (upgrade_dir / "tiff_resolution_audit_updated.csv" if args.upgraded_only
                              else root / AUDIT_RELATIVE)).expanduser().resolve()
    if args.upgraded_only:
        upgrade_selection, upgraded_sources = upgraded_slide_selection(source, upgrade_dir, audit)
        names = sorted(upgraded_sources)
        if not names:
            raise ValueError("No completed image upgrades found; no prediction jobs selected")
    selected = select_inputs(source, names)
    if args.exclude_slides:
        excluded = set(select_inputs(source, args.exclude_slides))
        selected = [path for path in selected if path not in excluded]
        if not selected:
            raise ValueError("No TIFF slides remain after exclusions")
    with audit.open(newline="") as stream:
        audit_rows = list(csv.DictReader(stream))
    audit_ids = [row["slide_id"] for row in audit_rows]
    if len(audit_ids) != len(set(audit_ids)):
        raise ValueError("Audit contains duplicate slide IDs")
    missing = sorted({p.stem for p in selected} - set(audit_ids))
    if missing:
        raise ValueError(f"Slides missing from the resolution audit: {missing}")
    config = {
        "project_root": str(root), "input_dir": str(source), "output_dir": str(output),
        "crop_size_um": args.crop_size_um, "min_tissue": args.min_tissue,
        "batch_size": args.batch_size, "tile_selection": "all_eligible",
        "audit_csv": str(audit), "audit_sha256": file_hash(audit),
        "slides": [p.stem for p in selected],
        "training_feature_root": str(args.training_feature_root.expanduser().resolve()) if args.training_feature_root else None,
        "model_dirs_json": str(args.model_dirs_json.expanduser().resolve()) if args.model_dirs_json else None,
        "model_dirs_sha256": file_hash(args.model_dirs_json) if args.model_dirs_json else None,
    }
    # Omit the default to keep existing frozen configs and their hashes unchanged.
    if args.include_completed_75um or args.upgraded_only:
        config["include_completed_75um"] = True
    if args.upgraded_only:
        config["upgrade_dir"] = str(upgrade_dir)
        config["upgraded_sources"] = {p.stem: upgraded_sources[p.stem] for p in selected}
    config_path = output / "cohort_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("This output folder has different slides or settings; choose a new output folder")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output folder is not empty and has no cohort configuration")
        output.mkdir(parents=True, exist_ok=True)
        with config_path.open("x") as stream:
            json.dump(config, stream, indent=2, allow_nan=False)
            stream.write("\n")
    with (output / "slide_manifest.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task_index", "slide_id", "tiff_path", "output_dir"])
        for index, path in enumerate(selected):
            writer.writerow([index, path.stem, path, output / "slides" / path.stem])
    (output / "log").mkdir(exist_ok=True)
    if args.upgraded_only:
        with (output / "upgrade_selection.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["slide_id", "upgrade_status", "selected"])
            writer.writeheader()
            writer.writerows(dict(row, selected=row["slide_id"] in config["slides"]) for row in upgrade_selection)
        print(f"Selected {len(selected)} completed TIFF upgrades from {len(upgrade_selection)} audited slides", flush=True)
    return config_path, config


def submit(args):
    if not args.dry_run:
        ensure_hf_token()
    config_path, config = prepare_run(args)
    output = Path(config["output_dir"])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    eligibility = output / "log" / f"eligibility_{timestamp}.csv"
    indices, rows = [], []
    for index, slide in enumerate(config["slides"]):
        completed, reason = completed_for_run(config, slide)
        if completed is None:
            indices.append(index)
        rows.append([index, slide, f"skip_{reason}" if completed else "run", str(completed or "")])
    with eligibility.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task_index", "slide_id", "action", "source_summary"])
        writer.writerows(rows)
    skipped = len(rows) - len(indices)
    print(f"Skipping {skipped} slides with accepted completed predictions; record: {eligibility}", flush=True)
    if not indices:
        print("All requested slides already have accepted completed predictions; no jobs submitted", flush=True)
        return
    # Preserve the frozen task numbers even when more 75 um slides finish between submissions.
    array = f"0-{len(config['slides']) - 1}" if not skipped else ",".join(map(str, indices))
    command = slurm.command(args) + [
        f"--array={array}%{args.max_concurrent}",
        *([f"--job-name={args.job_name}"] if args.job_name else []),
        f"--output={output / 'log/slide_%A_%a.out'}", f"--error={output / 'log/slide_%A_%a.err'}",
        str(slurm.LAUNCHER), '-m', 'xenium_pipeline.tcga_brca_cohort',
        "run", "--config", str(config_path), "--config-sha256", file_hash(config_path),
        '--device', 'cuda' if args.gpus else 'cpu',
    ]
    print(f"{len(indices)} slide jobs; {args.gpus} GPUs and {args.cpus} CPUs each; at most {args.max_concurrent} running concurrently", flush=True)
    print(f"Crop side: {config['crop_size_um']:g} µm; all eligible full tiles; tissue threshold: {config['min_tissue']:.0%}", flush=True)
    print(f"Slide/task mapping: {output / 'slide_manifest.csv'}", flush=True)
    print(f"Logs: {output / 'log'}", flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        print("Preview prepared; no jobs submitted", flush=True)
        return
    if config.get("upgraded_sources"):
        # Validate the compressed TIFF reader in the inference environment before
        # scheduling many GPU jobs. This small region read needs no GPU.
        from .tcga_brca_images import image_array, inspect_slide, rgb_region
        source = Path(config["upgraded_sources"][config["slides"][indices[0]]]["source_path"])
        metadata = inspect_slide(source)
        with image_array(source, metadata["series_index"], 0) as (pixels, axes, reader):
            rgb_region(pixels, axes, (0, 0, min(256, metadata["width_px"]), min(256, metadata["height_px"])))
        print(f"Upgraded TIFF region reader passed: {reader}", flush=True)
    submitted = slurm.submit(command)
    (output / "log" / f"submission_{timestamp}.json").write_text(json.dumps(
        {"slurm_submission": submitted, "command": command, "max_concurrent": args.max_concurrent,
         "submitted_task_indices": indices, "eligibility_report": str(eligibility)}, indent=2) + "\n")
    print(f"Submitted array: {submitted}", flush=True)


def run_local(args):
    if args.threads < 1:
        raise ValueError('CPU threads must be positive')
    config_path, config = prepare_run(args)
    if args.dry_run:
        print(f"Local preview: {len(config['slides'])} slides; device={args.device}; no inference started")
        return
    digest = file_hash(config_path)
    for index in range(len(config['slides'])):
        run_slide(argparse.Namespace(config=config_path, config_sha256=digest,
            task_index=index, device=args.device, threads=args.threads))


def run_slide(args):
    if file_hash(args.config) != args.config_sha256:
        raise ValueError("Cohort configuration changed after submission")
    config = json.loads(args.config.read_text())
    index = args.task_index
    if index is None:
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if not 0 <= index < len(config["slides"]):
        raise ValueError(f"Array task index outside slide manifest: {index}")
    slide = config["slides"][index]
    completed, reason = completed_for_run(config, slide)
    if completed is not None:
        log = Path(config["output_dir"]) / "log"
        log.mkdir(parents=True, exist_ok=True)
        array_id = str(int(os.environ["SLURM_ARRAY_JOB_ID"])) if "SLURM_ARRAY_JOB_ID" in os.environ else "manual"
        (log / f"skip_{array_id}_{index}.json").write_text(json.dumps({
            "status": "skipped", "reason": reason, "slide_id": slide,
            "task_index": index, "requested_crop_size_um": config["crop_size_um"],
            "source_summary": str(completed), "checked_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2) + "\n")
        print(f"SKIP {slide}: {reason} at {completed}", flush=True)
        return
    if file_hash(config["audit_csv"]) != config["audit_sha256"]:
        raise ValueError("Resolution audit changed after submission")
    if config["model_dirs_json"] and file_hash(config["model_dirs_json"]) != config["model_dirs_sha256"]:
        raise ValueError("Model-directory selection changed after submission")
    output = Path(config["output_dir"]) / "slides" / slide
    command = [sys.executable, "-u", "-m", "xenium_pipeline.tcga_brca_prediction",
               "--project-root", config["project_root"], "--input-dir", config["input_dir"],
               "--audit-csv", config["audit_csv"], "--slides", slide, "--all-tiles",
               "--sizes-um", str(config["crop_size_um"]), "--min-tissue", str(config["min_tissue"]),
               "--batch-size", str(config["batch_size"]), "--output-dir", str(output),
               "--device", args.device]
    if args.threads is not None:
        command.extend(['--threads', str(args.threads)])
    for field, flag in (("training_feature_root", "--training-feature-root"), ("model_dirs_json", "--model-dirs-json")):
        if config[field]:
            command.extend([flag, config[field]])
    print(f"Array task {index}: {slide}; {config['crop_size_um']:g} µm; all eligible full tiles", flush=True)
    print(f"Slide output: {output}", flush=True)
    environment = {**os.environ, "XENIUM_ROOT": config["project_root"],
                   "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    subprocess.run(command, check=True, env=environment)
    print(f"Slide job complete: {slide}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("login", help="Save a Hugging Face login once using hidden token entry")
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--project-root", type=Path, default=Path(os.environ.get("XENIUM_ROOT", DEFAULT_ROOT)))
    shared.add_argument("--input-dir", type=Path, default=Path(DEFAULT_INPUT))
    shared.add_argument("--output-dir", type=Path, required=True)
    shared.add_argument("--crop-size-um", type=float, default=75)
    selection = shared.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all-slides", action="store_true")
    selection.add_argument("--slides", nargs="+", help="Exact slide IDs or TIFF filenames")
    selection.add_argument("--slides-file", type=Path, help="One exact slide ID or filename per line")
    selection.add_argument("--upgraded-only", action="store_true",
                           help="Select completed TIFF upgrades; always recompute from upgraded pixels in a separate output folder")
    shared.add_argument("--upgrade-dir", type=Path,
                            help="Upgrade output containing upgrade_state and tiff_resolution_audit_updated.csv; defaults beside input-dir")
    shared.add_argument("--exclude-slides", nargs="+", default=[],
                            help="Exclude these exact slide IDs or TIFF filenames from the selected slides")
    shared.add_argument("--include-completed-75um", action="store_true",
                            help="Require predictions at this run's crop size even if 75 um predictions exist")
    shared.add_argument("--min-tissue", type=float, default=.60)
    shared.add_argument("--audit-csv", type=Path)
    shared.add_argument("--training-feature-root", type=Path)
    shared.add_argument("--model-dirs-json", type=Path)
    shared.add_argument("--dry-run", action="store_true", help="Save the manifest without starting prediction")
    submission = commands.add_parser("submit", parents=[shared], help="Submit a GPU slide array to Slurm")
    slurm.add_arguments(submission, cpus=8, memory='64G', time_limit='7-00:00:00', gpus=1)
    submission.add_argument("--job-name", help="Optional Slurm job name for this cohort")
    submission.add_argument("--max-concurrent", type=int, default=4)
    submission.add_argument("--batch-size", type=int, default=8)
    local = commands.add_parser("local", parents=[shared], help="Predict selected slides sequentially without Slurm")
    local.add_argument("--device", choices=('cpu', 'cuda'), default='cpu')
    local.add_argument("--threads", type=int, default=1)
    local.add_argument("--batch-size", type=int, default=1)
    worker = commands.add_parser("run", help="Execute the slide assigned by SLURM_ARRAY_TASK_ID")
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--config-sha256", required=True)
    worker.add_argument("--task-index", type=int, help="Slide index; defaults to SLURM_ARRAY_TASK_ID")
    worker.add_argument("--device", choices=('cpu', 'cuda'), default='cuda')
    worker.add_argument("--threads", type=int)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "login":
        login_once()
        return 0
    {'submit': submit, 'local': run_local, 'run': run_slide}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
