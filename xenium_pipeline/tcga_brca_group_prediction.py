"""Resumable GPU inference on balanced tile ranges, followed by per-slide merging."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import gc
import getpass
from importlib.metadata import version
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from PIL import Image

from . import tcga_brca_tile_groups as inventory
from . import tcga_brca_images as pilot
from . import tcga_brca_prediction as prediction
from . import stpath_xgboost as sx
from .hf_auth import ensure_hf_token
from .tcga_brca_cohort import DEFAULT_ROOT, check_source_stat
from . import slurm


def read_json(path):
    return json.loads(Path(path).read_text())


@contextmanager
def exclusive(path):
    """Release the lock automatically even when a worker is killed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is writing this run: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def code_identity():
    base = Path(__file__).parent
    return {name: inventory.file_hash(base / name) for name in (
        Path(__file__).name, "stpath_foundation.py", "stpath_xgboost.py",
        "tcga_brca_prediction.py", "tcga_brca_images.py")}


def environment_identity():
    return {name: version(name) for name in (
        "torch", "torchvision", "timm", "conch", "huggingface-hub", "xgboost", "numpy", "Pillow", "tifffile", "zarr")}


def load_plan(group_dir):
    """Verify checksums and exact interval coverage without decoding TIFF pixels."""
    group_dir = Path(group_dir).resolve()
    root = group_dir.parent
    plan = read_json(group_dir / "group_plan.json")
    config = read_json(root / "inventory_config.json")
    config_hash = inventory.file_hash(root / "inventory_config.json")
    if (plan["status"] != "ready_for_prediction" or plan["config_sha256"] != config_hash or
            plan["crop_size_um"] != config["crop_size_um"]):
        raise ValueError("Group plan and inventory configuration disagree")
    for name, key in (("group_assignments.csv", "assignment_sha256"), ("group_summary.csv", "summary_sha256")):
        if inventory.file_hash(group_dir / name) != plan[key]:
            raise ValueError(f"Group plan checksum mismatch: {name}")
    if inventory.file_hash(config["audit_csv"]) != config["audit_sha256"]:
        raise ValueError("Resolution audit changed after counting")
    parts = pd.read_csv(group_dir / "group_assignments.csv").to_dict("records")
    records = {sid: inventory.checked_record(root, sid, config_hash) for sid in config["slides"]}
    if not records or any(r["n_valid_tiles"] < 1 for r in records.values()):
        raise ValueError("Each selected slide must have at least one valid tile")
    if set(p["slide_id"] for p in parts) != set(records):
        raise ValueError("Group assignments do not cover the selected slides")
    totals = [0] * plan["n_groups"]
    for sid, record in records.items():
        cursor = 0
        for part in sorted((p for p in parts if p["slide_id"] == sid), key=lambda p: p["tile_start"]):
            g, start, stop, count = (part[k] for k in ("group_index", "tile_start", "tile_stop", "n_tiles"))
            if (any(type(v) is not int for v in (g, start, stop, count)) or
                    not 0 <= g < len(totals) or start != cursor or stop <= start or count != stop-start):
                raise ValueError(f"Overlapping, missing or invalid tile ranges: {sid}")
            if (Path(part["tile_manifest"]).resolve() != root / "slides" / f"{sid}.csv.gz" or
                    part["tiles_sha256"] != record["tiles_sha256"]):
                raise ValueError(f"Tile manifest identity mismatch: {sid}")
            totals[g] += count
            cursor = stop
        if cursor != record["n_valid_tiles"]:
            raise ValueError(f"Incomplete tile range coverage: {sid}")
    summary = pd.read_csv(group_dir / "group_summary.csv").sort_values("group_index")
    if (summary.group_index.tolist() != list(range(len(totals))) or summary.n_tiles.tolist() != totals or
            summary.n_slides.tolist() != [len({p["slide_id"] for p in parts if p["group_index"] == i}) for i in range(len(totals))] or
            sum(totals) != plan["n_valid_tiles"] or len(records) != plan["n_slides"] or
            max(totals)-min(totals) > 1 or min(totals) < 1):
        raise ValueError("Group totals disagree with the inventory")
    return config, plan, parts, records


def check_cached_revision(name, training):
    """The unchanged training loader uses main; require its offline cache to match."""
    from huggingface_hub import constants
    from .stpath_foundation import MODEL_SOURCES
    expected = training[name]["recorded_revision"]
    if not expected:
        raise ValueError(f"No recorded training revision for {name}; resolve provenance before prediction")
    cache = Path(constants.HF_HUB_CACHE) / ("models--" + MODEL_SOURCES[name].replace("/", "--"))
    ref = cache / "refs/main"
    if not ref.is_file() or ref.read_text().strip() != expected:
        raise ValueError(f"{name}: cached main must match training revision {expected}; cache: {cache}")
    snapshot = cache / "snapshots" / expected
    weights = [p for p in snapshot.glob("*") if p.suffix in {".bin", ".safetensors"} and p.is_file() and p.stat().st_size]
    if not weights or (name != "Conch" and not (snapshot / "config.json").is_file()):
        raise FileNotFoundError(f"Incomplete cached {name} weights/configuration: {snapshot}")
    return expected


def prepare(args):
    root, group_dir, output = (p.resolve() for p in (args.project_root, args.group_dir, args.output_dir))
    if args.batch_size < 1 or args.chunk_size < 1 or args.max_concurrent < 1:
        raise ValueError("Batch size, chunk size and concurrency must be positive")
    source, plan, _, _ = load_plan(group_dir)
    if output == Path(source["input_dir"]) or Path(source["input_dir"]) in output.parents:
        raise ValueError("Keep predictions outside the source TIFF directory")
    model_dirs = (read_json(args.model_dirs_json) if args.model_dirs_json else
                  {target: str(root / "xgboost_training/runs" / run / sx.safe_name(target))
                   for target, run in prediction.DEFAULT_MODEL_RUNS.items()})
    model_dirs = {name: str(Path(path).resolve()) for name, path in model_dirs.items()}
    exports = prediction.check_model_exports(model_dirs, nthread=1)
    feature_root = args.training_feature_root or root / "foundation_features" / prediction.RECIPE
    training = prediction.training_encoder_revisions(feature_root)
    for name in sx.MODEL_NAMES:
        check_cached_revision(name, training)
    config = dict(version=1, project_root=str(root), output_dir=str(output), group_dir=str(group_dir),
                  group_plan_sha256=inventory.file_hash(group_dir / "group_plan.json"),
                  inventory_sha256=plan["config_sha256"], n_groups=plan["n_groups"],
                  slides=source["slides"], crop_size_um=source["crop_size_um"], min_tissue=source["min_tissue"],
                  audit_sha256=source["audit_sha256"], tile_selection="all_eligible",
                  model_dirs=model_dirs, model_exports=exports, training_revisions=training,
                  batch_size=args.batch_size, chunk_size=args.chunk_size,
                  code_sha256=code_identity(), versions=environment_identity())
    inventory.freeze_config(output / "cohort_config.json", config)
    return config, inventory.file_hash(output / "cohort_config.json")


def load_run(args):
    path = args.output_dir.resolve() / "cohort_config.json"
    digest = inventory.file_hash(path)
    if args.config_sha256 and digest != args.config_sha256:
        raise ValueError("Prediction configuration changed after submission")
    config = read_json(path)
    if config["code_sha256"] != code_identity() or config["versions"] != environment_identity():
        raise ValueError("Prediction code/environment changed; restore the submitted versions before resuming")
    group_dir = Path(config["group_dir"])
    if inventory.file_hash(group_dir / "group_plan.json") != config["group_plan_sha256"]:
        raise ValueError("Group plan changed after submission")
    source, plan, parts, records = load_plan(group_dir)
    if plan["config_sha256"] != config["inventory_sha256"] or config["slides"] != source["slides"]:
        raise ValueError("Inventory changed after submission")
    for export in config["model_exports"]:
        for relative, expected in export["file_sha256"].items():
            if inventory.file_hash(Path(export["target_dir"]) / relative) != expected:
                raise ValueError(f"Final model export changed: {export['cell_type']}")
    return config, digest, parts, records


def pieces(parts, chunk_size):
    for part in parts:
        for start in range(part["tile_start"], part["tile_stop"], chunk_size):
            yield {**part, "tile_start": start, "tile_stop": min(start+chunk_size, part["tile_stop"])}


def piece_dir(output, part):
    return Path(output) / "groups" / f"group_{part['group_index']:02d}" / part["slide_id"] / f"{part['tile_start']}_{part['tile_stop']}"


def piece_identity(digest, part, name):
    return dict(config_sha256=digest, slide_id=part["slide_id"], group_index=part["group_index"],
                tile_start=part["tile_start"], tile_stop=part["tile_stop"], name=name)


def cached_array(folder, name, identity, tile_ids=None, width=None):
    marker = folder / f"{name}.json"
    if not marker.exists():
        return None
    meta = read_json(marker)
    path = folder / f"{name}.npz"
    if meta.get("identity") != identity or meta["array_sha256"] != inventory.file_hash(path):
        raise ValueError(f"Changed checkpoint: {path}")
    with np.load(path, allow_pickle=False) as payload:
        values, ids = payload["values"], payload["tile_ids"].tolist()
    if (values.ndim != 2 or values.dtype != np.float32 or not np.isfinite(values).all() or
            len(ids) != identity["tile_stop"]-identity["tile_start"] or len(set(ids)) != len(ids) or
            values.shape[0] != len(ids) or (width is not None and values.shape[1] != width) or
            (tile_ids is not None and ids != tile_ids)):
        raise ValueError(f"Invalid checkpoint values/tile order: {path}")
    return values


def save_array(folder, name, identity, ids, values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or len(values) != len(ids) or not np.isfinite(values).all():
        raise ValueError(f"Invalid {name} values")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.npz"
    temporary = folder / f"{name}.tmp.npz"
    np.savez(temporary, values=values, tile_ids=np.asarray(ids, dtype=str))
    temporary.replace(path)
    inventory.write_json(folder / f"{name}.json", dict(identity=identity, array_sha256=inventory.file_hash(path)))


def read_tiles(part, record):
    table = pd.read_csv(part["tile_manifest"])
    metadata = record["metadata"]
    if (len(table) != record["n_valid_tiles"] or table.tile_index.tolist() != list(range(len(table))) or
            table.tile_id.duplicated().any() or not table.slide_id.eq(part["slide_id"]).all() or
            not table.patient_id.eq(part["slide_id"][:12]).all()):
        raise ValueError(f"Invalid tile inventory: {part['slide_id']}")
    for axis, limit in (("x", metadata["width_px"]), ("y", metadata["height_px"])):
        start, end = table[f"{axis}0_px"], table[f"{axis}1_px"]
        if not (start.ge(0) & end.gt(start) & end.le(limit) & start.mod(1).eq(0) & end.mod(1).eq(0)).all():
            raise ValueError(f"Invalid tile coordinates: {part['slide_id']}")
    return table


def cuda_setup(threads):
    import torch
    torch.set_num_threads(threads)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this allocation; a working GPU is required")
    # Exercise the allocation before reading many TIFFs or loading large models.
    (torch.ones(1, device="cuda") + 1).cpu()
    print(f"GPU: {torch.cuda.get_device_name()}; threads: {threads}", flush=True)


def build_encoder(name, config):
    from . import stpath_foundation as foundation
    check_cached_revision(name, config["training_revisions"])
    model, transform = foundation.build_model(name, "cuda")
    check_cached_revision(name, config["training_revisions"])
    return model, transform


def release_gpu():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def encode_chunk(name, model, transform, pixels, axes, table, batch_size):
    import torch
    from .stpath_foundation import encode_batch
    values = []
    with torch.inference_mode():
        for offset in range(0, len(table), batch_size):
            batch = table.iloc[offset:offset+batch_size]
            images = []
            for row in batch.itertuples():
                bounds = (int(row.x0_px), int(row.y0_px), int(row.x1_px), int(row.y1_px))
                rgb = pilot.rgb_region(pixels, axes, bounds)
                if rgb.dtype != np.uint8 or rgb.shape != (bounds[3]-bounds[1], bounds[2]-bounds[0], 3):
                    raise ValueError(f"Unexpected native RGB tile: {row.tile_id}")
                images.append(transform(Image.fromarray(rgb)))
            embedding = encode_batch(name, model, torch.stack(images).to("cuda"))
            if embedding.ndim != 2 or embedding.shape != (len(batch), sx.EXPECTED_DIMS[name]):
                raise ValueError(f"Unexpected feature shape for {name}: {tuple(embedding.shape)}")
            values.append(embedding.detach().float().cpu().numpy())
    return np.concatenate(values)


def extract_group(config, digest, parts, records):
    errors = []
    for name in sx.MODEL_NAMES:
        model = transform = None
        processed = 0
        try:
            for part in parts:
                record = records[part["slide_id"]]
                table = read_tiles(part, record)
                pending = []
                for chunk in pieces([part], config["chunk_size"]):
                    rows = table.iloc[chunk["tile_start"]:chunk["tile_stop"]]
                    found = cached_array(piece_dir(config["output_dir"], chunk), name,
                                         piece_identity(digest, chunk, name), rows.tile_id.tolist(), sx.EXPECTED_DIMS[name])
                    if found is None:
                        pending.append(chunk)
                    else:
                        processed += len(rows)
                if not pending:
                    print(f"Reusing {name} {part['slide_id']} range {part['tile_start']}:{part['tile_stop']}", flush=True)
                    continue
                if model is None:
                    try:
                        model, transform = build_encoder(name, config)
                    except Exception as exc:
                        errors.append(dict(encoder=name, error=f"{type(exc).__name__}: {exc}"))
                        print(f"ERROR loading {name}: {exc}; continuing with next encoder", file=sys.stderr, flush=True)
                        break
                metadata = record["metadata"]
                check_source_stat(metadata)
                with pilot.image_array(Path(metadata["source_path"]), metadata["series_index"], 0) as (pixels, axes, _):
                    for chunk in pending:
                        rows = table.iloc[chunk["tile_start"]:chunk["tile_stop"]]
                        started = time.monotonic()
                        values = encode_chunk(name, model, transform, pixels, axes, rows, config["batch_size"])
                        check_source_stat(metadata)
                        save_array(piece_dir(config["output_dir"], chunk), name, piece_identity(digest, chunk, name),
                                   rows.tile_id.tolist(), values)
                        processed += len(rows)
                        print(f"{name} group {part['group_index']}: {processed:,} tiles saved; "
                              f"{part['slide_id']} {chunk['tile_start']}:{chunk['tile_stop']} "
                              f"({time.monotonic()-started:.1f}s)", flush=True)
                        del values
        finally:
            del model, transform
            release_gpu()
    if errors:
        folder = Path(config["output_dir"]) / "groups" / f"group_{parts[0]['group_index']:02d}"
        inventory.write_json(folder / "encoder_load_errors.json", errors)
        raise RuntimeError("Some encoders failed to load; successful chunks were saved. All five are required for Combined predictions.")


def load_boosters(config, threads):
    xgb = sx._load_xgboost()
    models = {}
    for target, folder in config["model_dirs"].items():
        metadata = read_json(Path(folder) / "model_metadata/Combined.json")
        booster = xgb.Booster()
        booster.load_model(str(Path(folder) / "models/Combined.ubj"))
        booster.set_param({"nthread": threads})
        models[target] = (booster, metadata["selected_feature_indices"])
    return models


def predict_group(config, digest, parts, records, threads):
    models = None
    for part in parts:
        table = read_tiles(part, records[part["slide_id"]])
        for chunk in pieces([part], config["chunk_size"]):
            rows = table.iloc[chunk["tile_start"]:chunk["tile_stop"]]
            folder = piece_dir(config["output_dir"], chunk)
            identity = piece_identity(digest, chunk, "predictions")
            if cached_array(folder, "predictions", identity, rows.tile_id.tolist(), len(config["model_dirs"])) is not None:
                continue
            matrices = {name: cached_array(folder, name, piece_identity(digest, chunk, name), rows.tile_id.tolist(), sx.EXPECTED_DIMS[name])
                        for name in sx.MODEL_NAMES}
            if any(v is None for v in matrices.values()):
                raise RuntimeError(f"Missing encoder chunks: {folder}")
            if models is None:
                models = load_boosters(config, threads)
            predicted = []
            for booster, selections in models.values():
                features = sx.concatenate_selected_features(matrices, selections, np.arange(len(rows)))
                predicted.append(sx.predict_booster(booster, features, nthread=threads))
            save_array(folder, "predictions", identity, rows.tile_id.tolist(), np.column_stack(predicted))
            print(f"Predictions saved: {part['slide_id']} {chunk['tile_start']}:{chunk['tile_stop']}", flush=True)


def run_group(args):
    config, digest, parts, records = load_run(args)
    index = args.group_index if args.group_index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    parts = [p for p in parts if p["group_index"] == index]
    if not parts:
        raise ValueError(f"Unknown/empty group: {index}")
    folder = Path(config["output_dir"]) / "groups" / f"group_{index:02d}"
    with exclusive(folder / ".worker.lock"):
        status = dict(status="running", config_sha256=digest, group_index=index)
        inventory.write_json(folder / "status.json", status)
        try:
            threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
            ensure_hf_token()
            cuda_setup(threads)
            extract_group(config, digest, parts, records)
            predict_group(config, digest, parts, records, threads)
        except Exception as exc:
            inventory.write_json(folder / "status.json", {**status, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            raise
        inventory.write_json(folder / "status.json", {**status, "status": "complete"})
        print(f"Group {index} complete", flush=True)


def merge_slide(config, digest, parts, record):
    table = read_tiles(parts[0], record)
    blocks = []
    for chunk in pieces(sorted(parts, key=lambda p: p["tile_start"]), config["chunk_size"]):
        rows = table.iloc[chunk["tile_start"]:chunk["tile_stop"]]
        values = cached_array(piece_dir(config["output_dir"], chunk), "predictions",
                              piece_identity(digest, chunk, "predictions"), rows.tile_id.tolist(), len(config["model_dirs"]))
        if values is None:
            return False
        blocks.append(values)
    values = np.concatenate(blocks)
    if len(values) != len(table):
        raise ValueError(f"Incomplete merged prediction: {record['slide_id']}")
    folder = Path(config["output_dir"]) / "slides" / record["slide_id"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "prediction_summary.json").unlink(missing_ok=True)
    # Completion is published last. Intermediate writes are never accepted alone.
    table.to_csv(folder / "sampled_tiles.csv", index=False)
    manifest_hash = inventory.file_hash(folder / "sampled_tiles.csv")
    summaries, frames = [], []
    for index, target in enumerate(config["model_dirs"]):
        raw = values[:, index]
        frame = table.copy()
        frame["cell_type"], frame["predicted_raw"] = target, raw
        frame["outside_0_1"] = (raw < 0) | (raw > 1)
        frames.append(frame)
        summaries.append(dict(slide_id=record["slide_id"], patient_id=record["slide_id"][:12],
                              candidate=table.candidate.iloc[0], cell_type=target, n_sampled_tiles=len(raw),
                              sample_mean=float(np.mean(raw, dtype=np.float64)), sample_median=float(np.median(raw)),
                              p05=float(np.quantile(raw, .05)), p95=float(np.quantile(raw, .95)),
                              min=float(raw.min()), max=float(raw.max()), fraction_outside_0_1=float(frame.outside_0_1.mean())))
    pd.concat(frames, ignore_index=True).to_csv(folder / "tile_predictions_raw.csv.gz", index=False)
    pd.DataFrame(summaries).to_csv(folder / "sampled_slide_summary.csv", index=False)
    pd.DataFrame([dict(slide_id=record["slide_id"], candidate=table.candidate.iloc[0], n_sampled_tiles=len(table),
                       n_eligible_full_tiles_estimate=len(table))]).to_csv(folder / "sampling_summary.csv", index=False)
    preparation = dict(status="complete", sizes_um=[config["crop_size_um"]], recipe=prediction.RECIPE,
                       max_tiles_per_slide_size=None, sampling_mode="all_eligible_grouped",
                       audit_csv_sha256=config["audit_sha256"], min_tissue_fraction=config["min_tissue"],
                       manifest_sha256=manifest_hash, source_metadata=[record["metadata"]])
    inventory.write_json(folder / "tile_preparation.json", preparation)
    provenance = {name: dict(recipe=prediction.RECIPE, recipe_code_sha256=config["code_sha256"]["stpath_foundation.py"],
                            resolved_hf_revision=config["training_revisions"][name]["recorded_revision"],
                            versions=config["versions"]) for name in sx.MODEL_NAMES}
    files = ("sampled_tiles.csv", "tile_predictions_raw.csv.gz", "sampled_slide_summary.csv", "sampling_summary.csv", "tile_preparation.json")
    check_source_stat(record["metadata"])
    inventory.write_json(folder / "prediction_summary.json", dict(status="complete", n_slides=1, n_crops=len(table),
        sizes_um=[config["crop_size_um"]], cell_types=list(config["model_dirs"]), model_exports=config["model_exports"],
        calibration=False, clipping=False, across_cell_type_normalization=False, tile_selection="all_eligible",
        manifest_sha256=manifest_hash, feature_provenance=provenance, grouped_config_sha256=digest,
        output_sha256={name: inventory.file_hash(folder / name) for name in files},
        interpretation="All eligible full tiles, merged exactly once across groups. Slide means are tile averages, not cell-count-weighted fractions."))
    return True


def merge(args):
    config, digest, parts, records = load_run(args)
    output = Path(config["output_dir"])
    with exclusive(output / ".merge.lock"):
        statuses = []
        for sid, record in records.items():
            try:
                marker = output / "slides" / sid / "prediction_summary.json"
                if marker.is_file():
                    saved = read_json(marker)
                    if saved.get("grouped_config_sha256") != digest or any(
                            inventory.file_hash(marker.parent / name) != value for name, value in saved["output_sha256"].items()):
                        raise ValueError(f"Changed completed slide output: {sid}")
                    complete = saved["status"] == "complete"
                else:
                    complete = merge_slide(config, digest, [p for p in parts if p["slide_id"] == sid], record)
                statuses.append(dict(slide_id=sid, status="complete" if complete else "incomplete", error=""))
                print(f"{sid}: {statuses[-1]['status']}", flush=True)
            except (OSError, ValueError, KeyError) as exc:
                statuses.append(dict(slide_id=sid, status="invalid", error=str(exc)))
        pd.DataFrame(statuses).to_csv(output / "slide_completion.csv", index=False)
        n_complete = sum(row["status"] == "complete" for row in statuses)
        inventory.write_json(output / "merge_summary.json", dict(status="complete" if n_complete == len(records) else "incomplete",
                             config_sha256=digest, n_slides=len(records), n_slides_complete=n_complete,
                             n_slides_missing_predictions=len(records)-n_complete))
        if n_complete != len(records):
            raise RuntimeError(f"{len(records)-n_complete} slides incomplete/invalid; inspect slide_completion.csv and group logs")


def check_active(output):
    jobs = set()
    for path in (output / "log").glob("submission_*.json"):
        saved = read_json(path)
        jobs.update(str(saved[k]) for k in ("array_job_id", "merge_job_id") if saved.get(k))
    if jobs:
        queue = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%F"], check=True, text=True, capture_output=True)
        active = jobs & set(queue.stdout.split())
        if active:
            raise RuntimeError(f"Prediction/merge jobs still active: {sorted(active)}; wait before resubmitting")


def submit(args):
    output = args.output_dir.resolve()
    if args.gpus < 1:
        raise ValueError('Grouped prediction requires at least one GPU; use cohort local for CPU inference')
    launcher = slurm.LAUNCHER
    if not launcher.is_file():
        raise FileNotFoundError(f"Missing job launcher: {launcher}")
    with exclusive(output / ".submit.lock"):
        check_active(output)
        ensure_hf_token()
        config, digest = prepare(args)
        print("Tile ranges, source/audit identity, final models and cached training revisions validated.", flush=True)
        _, _, parts, _ = load_plan(config["group_dir"])
        indices = []
        for index in range(config["n_groups"]):
            done = True
            for chunk in pieces([p for p in parts if p["group_index"] == index], config["chunk_size"]):
                if cached_array(piece_dir(output, chunk), "predictions", piece_identity(digest, chunk, "predictions"),
                                width=len(config["model_dirs"])) is None:
                    done = False
                    break
            if not done:
                indices.append(index)
        log = output / "log"
        log.mkdir(exist_ok=True)
        common = slurm.command(args)
        merge_options = slurm.command(args, cpus=args.merge_cpus, memory=args.merge_memory,
            time_limit=args.merge_time, gpus=0)
        worker = [str(launcher), '-m', 'xenium_pipeline.tcga_brca_group_prediction']
        payload = ["--output-dir", str(output), "--config-sha256", digest]
        finish = merge_options + ["--job-name=brca-group-merge",
                           f"--output={log}/merge_%j.out", f"--error={log}/merge_%j.err", *worker, "merge", *payload]
        record = dict(array_job_id=None, merge_job_id=None, config_sha256=digest, group_indices=indices)
        path = log / f"submission_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')}.json"
        print(f"{len(indices)}/{config['n_groups']} groups need prediction; {args.gpus} GPUs / {args.cpus} CPUs / {args.memory} / {args.time_limit} per group", flush=True)
        print(f"Concurrency: {args.max_concurrent}; crop: {config['crop_size_um']:g} µm; logs: {log}", flush=True)
        if indices:
            command = common + ["--job-name=brca-group",
                       f"--array={','.join(map(str, indices))}%{args.max_concurrent}",
                       f"--output={log}/group_%A_%a.out", f"--error={log}/group_%A_%a.err", *worker, "run", *payload]
            print(shlex.join(command), flush=True)
            if args.dry_run:
                dependency = "GROUP_ARRAY_JOB_ID"
            else:
                dependency = record["array_job_id"] = slurm.submit(command)
                inventory.write_json(path, record)
                print(f"Submitted GPU array: {dependency}", flush=True)
            finish.insert(len(merge_options), f"--dependency=afterany:{dependency}")
        print(shlex.join(finish), flush=True)
        if args.dry_run:
            print("Dry run complete; no jobs submitted.", flush=True)
            return record
        record["merge_job_id"] = slurm.submit(finish)
        inventory.write_json(path, record)
        print(f"Submitted CPU merge job: {record['merge_job_id']}", flush=True)
        return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    submission = commands.add_parser("submit")
    submission.add_argument("--project-root", type=Path, default=Path(os.environ.get('XENIUM_ROOT', DEFAULT_ROOT)))
    submission.add_argument("--group-dir", type=Path, required=True)
    submission.add_argument("--training-feature-root", type=Path)
    submission.add_argument("--model-dirs-json", type=Path)
    submission.add_argument("--max-concurrent", type=int, default=20)
    slurm.add_arguments(submission, cpus=8, memory='64G', time_limit='7-00:00:00', gpus=1)
    submission.add_argument('--merge-cpus', type=int, default=4)
    submission.add_argument('--merge-mem', dest='merge_memory', default='32G')
    submission.add_argument('--merge-time', default='1-00:00:00')
    submission.add_argument("--batch-size", type=int, default=8)
    submission.add_argument("--chunk-size", type=int, default=4096)
    submission.add_argument("--dry-run", action="store_true")
    for sub in (submission, commands.add_parser("run"), commands.add_parser("merge")):
        sub.add_argument("--output-dir", type=Path, required=True)
        if sub is not submission:
            sub.add_argument("--config-sha256")
    commands.choices["run"].add_argument("--group-index", type=int)
    return parser.parse_args(argv)


def main(argv=None):
    # This must precede imports of huggingface_hub, timm, CONCH, or torch.
    os.environ["HF_HUB_OFFLINE"] = "1"
    args = parse_args(argv)
    {"submit": submit, "run": run_group, "merge": merge}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
