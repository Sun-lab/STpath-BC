"""Submit/resume one CPU counting job per upgraded slide, then collect and group."""
from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import tcga_brca_tile_groups as inventory
from .tcga_brca_cohort import DEFAULT_ROOT, DEFAULT_INPUT, upgraded_slide_selection
from . import slurm


def load_config(args):
    path = args.output_dir.resolve() / "inventory_config.json"
    if args.config_sha256 and inventory.file_hash(path) != args.config_sha256:
        raise ValueError("Frozen inventory configuration changed after submission")
    return json.loads(path.read_text())


def run_slide(args):
    config = load_config(args)
    task = args.task_index
    if task is None:
        task = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if not 0 <= task < len(config["slides"]):
        raise ValueError("Array task index is outside the frozen slide manifest")
    print(f"CPU counting task {task}: {config['slides'][task]}; Python: {os.sys.executable}", flush=True)
    summary = inventory.count_valid_tiles(config["input_dir"], config["audit_csv"], config["slides"], args.output_dir,
                                          crop_size_um=config["crop_size_um"], min_tissue=config["min_tissue"],
                                          task_index=task)
    if summary["status"] != "complete":
        raise RuntimeError(f"Slide counting failed; inspect slides/{config['slides'][task]}.status.json")


def finalize(args):
    load_config(args)
    summary = inventory.collect_counts(args.output_dir)
    print(json.dumps(summary, indent=2), flush=True)
    if summary["status"] != "complete":
        raise RuntimeError("Some slide inventories are missing/failed. Inspect slide_counts.csv and resubmit counting; no groups created.")
    plan = inventory.assign_groups(args.output_dir, n_groups=args.groups)
    print(json.dumps(plan, indent=2), flush=True)
    print("CPU counting and group assignment complete. No GPU prediction jobs submitted.", flush=True)


def check_active_submissions(output):
    previous_ids = set()
    for path in (output / "log").glob("submission_*.json"):
        saved = json.loads(path.read_text())
        previous_ids.update(str(saved[key]) for key in ("array_job_id", "finalize_job_id") if saved.get(key))
    if previous_ids:
        result = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%F"],
                                check=True, text=True, capture_output=True)
        active = previous_ids & set(result.stdout.split())
        if active:
            raise RuntimeError(f"Counting/finalization jobs still active: {', '.join(sorted(active))}. Wait before resubmitting this folder.")


def submit(args):
    if args.max_concurrent < 1 or args.groups < 1:
        raise ValueError("Concurrency and number of groups must be positive")
    source, output = (p.resolve() for p in (args.input_dir, args.output_dir))
    launcher = slurm.LAUNCHER
    if not launcher.is_file():
        raise FileNotFoundError(f"Missing job launcher: {launcher}")
    check_active_submissions(output)
    upgrade = (args.upgrade_dir or source.parent / "TCGA_BRCA_high_resolution").resolve()
    audit = (args.audit_csv or upgrade / "tiff_resolution_audit_updated.csv").resolve()
    _, sources = upgraded_slide_selection(source, upgrade, audit)
    config = inventory.count_valid_tiles(source, audit, sorted(sources), output,
                                         crop_size_um=args.crop_size_um, min_tissue=args.min_tissue, prepare_only=True)
    config_hash = inventory.file_hash(output / "inventory_config.json")
    inventory.collect_counts(output)
    with (output / "slide_counts.csv").open(newline="") as stream:
        counts = {row["slide_id"]: row for row in csv.DictReader(stream)}
    indices = [i for i, slide in enumerate(config["slides"]) if counts[slide]["status"] != "complete"]
    log_dir = output / "log"
    log_dir.mkdir(exist_ok=True)
    with (output / "count_task_manifest.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task_index", "slide_id", "action"])
        writer.writerows([i, slide, "run" if i in indices else "reuse_complete"] for i, slide in enumerate(config["slides"]))
    common = slurm.command(args)
    worker = [str(launcher), '-m', 'xenium_pipeline.tcga_brca_count_batch']
    payload = ["--output-dir", str(output), "--config-sha256", config_hash]
    finish = common + ["--job-name=brca-count-group", f"--output={log_dir}/group_%j.out",
                       f"--error={log_dir}/group_%j.err", *worker, "finalize", *payload,
                       "--groups", str(args.groups)]
    print(f"{len(indices)} slides need counting; {len(config['slides'])-len(indices)} completed inventories reused.", flush=True)
    print(f"One counting job per slide, {args.cpus} CPUs / {args.memory}, at most {args.max_concurrent} running; no token required.", flush=True)
    print(f"{config['crop_size_um']:g} µm; tissue cutoff {config['min_tissue']:.0%}; logs: {log_dir}", flush=True)
    record_path = log_dir / f"submission_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')}.json"
    record = dict(output_dir=str(output), config_sha256=config_hash, task_indices=indices, groups=args.groups,
                  array_job_id=None, finalize_job_id=None)
    if indices:
        array = ",".join(map(str, indices)) + f"%{args.max_concurrent}"
        command = common + ["--job-name=brca-count", f"--array={array}", f"--output={log_dir}/count_%A_%a.out",
                            f"--error={log_dir}/count_%A_%a.err", *worker, "run", *payload]
        print(shlex.join(command), flush=True)
        if args.dry_run:
            finish.insert(len(common), "--dependency=afterany:COUNT_ARRAY_JOB_ID")
        else:
            record["array_job_id"] = slurm.submit(command)
            inventory.write_json(record_path, record)  # Keep the array ID even if collector submission fails.
            print(f"Submitted counting array: {record['array_job_id']}", flush=True)
            finish.insert(len(common), f"--dependency=afterany:{record['array_job_id']}")
    print(shlex.join(finish), flush=True)
    if args.dry_run:
        print("Dry run: metadata prepared; no jobs submitted and no TIFF pixels decoded.")
        return record
    record["finalize_job_id"] = slurm.submit(finish)
    inventory.write_json(record_path, record)
    print(f"Submitted collection/grouping job: {record['finalize_job_id']}", flush=True)
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    submission = commands.add_parser("submit")
    submission.add_argument("--project-root", type=Path, default=Path(os.environ.get('XENIUM_ROOT', DEFAULT_ROOT)))
    submission.add_argument("--input-dir", type=Path, default=Path(DEFAULT_INPUT))
    submission.add_argument("--upgrade-dir", type=Path)
    submission.add_argument("--audit-csv", type=Path)
    submission.add_argument("--crop-size-um", type=float, default=75)
    submission.add_argument("--min-tissue", type=float, default=.60)
    submission.add_argument("--max-concurrent", type=int, default=20)
    submission.add_argument("--groups", type=int, default=20)
    slurm.add_arguments(submission, cpus=1, memory='8G', time_limit='1-00:00:00')
    submission.add_argument("--dry-run", action="store_true")
    for sub in (submission, commands.add_parser("run"), commands.add_parser("finalize")):
        sub.add_argument("--output-dir", type=Path, required=True)
        if sub is not submission:
            sub.add_argument("--config-sha256")
    commands.choices["run"].add_argument("--task-index", type=int)
    commands.choices["finalize"].add_argument("--groups", type=int, default=20)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    {"submit": submit, "run": run_slide, "finalize": finalize}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
