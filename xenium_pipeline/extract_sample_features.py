"""Prepare a sample task table or extract one sample with the encoder extraction recipe."""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

RECIPE_VERSION = "stpath_brca_training_compatible_v1"
TASK_COLUMNS = ("dataset", "sample", "materialized_tile_manifest", "n_final_selected_tiles")


def prepare_tasks(selection_manifest: Path, task_table: Path) -> int:
    """Freeze sample order without opening tiles or importing the model stack."""
    with selection_manifest.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(TASK_COLUMNS).issubset(reader.fieldnames or []):
            raise ValueError(f"Selection manifest requires {TASK_COLUMNS}")
        rows = [{key: row[key] for key in TASK_COLUMNS} for row in reader]
    if not rows:
        raise ValueError("Selection manifest contains no samples")
    identities = set()
    destinations = set()
    for row in rows:
        if any(not row[key].strip() for key in TASK_COLUMNS):
            raise ValueError("Sample rows require nonempty task fields")
        if int(row["n_final_selected_tiles"]) < 1:
            raise ValueError(f"Sample has no selected tiles: {row['sample']}")
        identity = (row["dataset"], row["sample"])
        destination = tuple(re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") for name in identity)
        if identity in identities or destination in destinations or not all(destination):
            raise ValueError(f"Duplicate or colliding sample destination: {identity}")
        identities.add(identity)
        destinations.add(destination)
    task_table.parent.mkdir(parents=True, exist_ok=True)
    with task_table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TASK_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def task_identity(task_table: Path, sample_index: int) -> tuple[str, str]:
    with task_table.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not 0 <= sample_index < len(rows):
        raise ValueError(f"sample-index {sample_index} is outside 0-{len(rows) - 1}")
    row = rows[sample_index]
    return row["dataset"], row["sample"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path,
                        default=Path(os.environ.get("XENIUM_ROOT", str(Path(__file__).resolve().parents[1]))))
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--task-table", type=Path, required=True)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--prepare-tasks", action="store_true")
    scope.add_argument("--sample-index", type=int)
    parser.add_argument("--model", default="ResNet50", choices=[
        "ResNet50", "UNI2h", "Virchow", "Virchow2", "ProvGigapath", "Conch",
    ])
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    root = args.project_root.expanduser().resolve()
    task_table = args.task_table.expanduser().resolve()
    if args.prepare_tasks:
        selection = args.selection_manifest or root / "tile_filter_cutoff_tests/final_automatic_tile_selection_manifest.csv"
        if not selection.is_absolute():
            selection = root / selection
        print(prepare_tasks(selection, task_table))
        return 0
    dataset, sample = task_identity(task_table, args.sample_index)
    from .feature_extraction import load_selected_tile_index, run_extraction, validate_outputs
    from .stpath_xgboost import log
    output = args.output_root or root / "foundation_features" / RECIPE_VERSION
    if not output.is_absolute():
        output = root / output
    log(f"Sample task {args.sample_index}: {dataset}/{sample}; encoder={args.model}")
    tiles = load_selected_tile_index(root, task_table, sample_keys=[f"{dataset}/{sample}"])
    run_extraction(tiles, [args.model], output, args.overwrite,
                   device=args.device, batch_size=args.batch_size,
                   num_workers=args.num_workers, recipe_version=RECIPE_VERSION)
    validation = validate_outputs(tiles, [args.model], output)
    if not bool(validation["valid"].all()):
        raise RuntimeError(f"Output validation failed: {validation.to_dict('records')}")
    log(f"Sample task {args.sample_index} complete: {dataset}/{sample}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
