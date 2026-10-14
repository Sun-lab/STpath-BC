"""Shared notebook and batch feature extraction, preserving canonical tile order."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import gc
import os
import re
import time
import pandas as pd
from .stpath_xgboost import resolve_project_path, tile_id_checksum, log

VALID_MATERIALIZATION_STATUSES = {"copy", "symlink", "move"}
TILE_COORDINATE_PATTERN = re.compile(r"_row-(\d+)_col-(\d+)$")


def load_selected_tile_index(project_root, selection_manifest, sample_keys=None, tile_limit_per_sample=None):
    project_root = Path(project_root).expanduser().resolve()
    selection_manifest = resolve_project_path(selection_manifest, project_root)
    if not selection_manifest.is_file():
        raise FileNotFoundError(f"Selection manifest not found: {selection_manifest}")

    summary = pd.read_csv(selection_manifest)
    required = {"dataset", "sample", "materialized_tile_manifest", "n_final_selected_tiles"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Selection manifest is missing columns: {sorted(missing)}")

    if summary.duplicated(["dataset", "sample"]).any():
        raise ValueError("Selection manifest contains duplicate dataset/sample rows")
    summary["sample_key"] = summary["dataset"].astype(str) + "/" + summary["sample"].astype(str)
    if sample_keys is not None:
        requested = set(map(str, sample_keys))
        unknown = requested.difference(summary["sample_key"])
        if unknown:
            raise ValueError(f"Unknown SAMPLE_KEYS: {sorted(unknown)}")
        summary = summary.loc[summary["sample_key"].isin(requested)].copy()

    sample_tables = []
    sample_count = len(summary)
    expected_counts = summary["n_final_selected_tiles"].astype(int)
    if tile_limit_per_sample is not None:
        expected_counts = expected_counts.clip(upper=int(tile_limit_per_sample))
    target_tiles = int(expected_counts.sum())
    indexed_tiles = 0
    index_started = time.perf_counter()
    log(f"Building canonical tile index: {sample_count:,} samples, {target_tiles:,} tiles")

    for sample_number, row in enumerate(summary.itertuples(index=False), start=1):
        sample_started = time.perf_counter()
        expected = int(row.n_final_selected_tiles)
        manifest_path = resolve_project_path(row.materialized_tile_manifest, project_root)
        log(
            f"[{sample_number}/{sample_count}] Reading {row.sample_key}: "
            f"expected {expected:,} retained rows from {manifest_path}"
        )
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Materialized-tile manifest not found for {row.sample_key}: {manifest_path}"
            )
        read_started = time.perf_counter()
        table = pd.read_csv(manifest_path, dtype={"tile_id": "string"})
        log(
            f"[{sample_number}/{sample_count}] CSV loaded: {len(table):,} rows "
            f"in {time.perf_counter() - read_started:.1f}s"
        )
        needed = {"tile_id", "dataset", "sample", "status", "materialized_tile_path"}
        missing = needed.difference(table.columns)
        if missing:
            raise ValueError(f"{manifest_path} is missing columns: {sorted(missing)}")

        table = table.loc[
            table["status"].astype(str).isin(VALID_MATERIALIZATION_STATUSES)
        ].copy()
        if len(table) != expected:
            raise ValueError(
                f"{row.sample_key}: expected {expected:,} materialized tiles, found {len(table):,}"
            )

        if not table["dataset"].astype(str).eq(str(row.dataset)).all() or not table["sample"].astype(str).eq(str(row.sample)).all():
            raise ValueError(f"Manifest contains another sample: {row.sample_key}")
        coordinates = table["tile_id"].str.extract(TILE_COORDINATE_PATTERN)
        if coordinates.isna().any(axis=None):
            bad = table.loc[coordinates.isna().any(axis=1), "tile_id"].head().tolist()
            raise ValueError(f"Could not parse tile row/column from IDs: {bad}")
        table["tile_row"] = coordinates[0].astype(int)
        table["tile_col"] = coordinates[1].astype(int)
        table["sample_key"] = row.sample_key
        table["materialized_tile_path"] = table["materialized_tile_path"].map(
            lambda value: str(resolve_project_path(value, project_root))
        )
        if tile_limit_per_sample is not None:
            table = table.head(int(tile_limit_per_sample)).copy()
        sample_tables.append(
            table[[
                "tile_id", "dataset", "sample", "sample_key",
                "tile_row", "tile_col", "materialized_tile_path",
            ]]
        )
        indexed_tiles += len(table)
        elapsed = time.perf_counter() - index_started
        remaining = max(target_tiles - indexed_tiles, 0)
        eta_seconds = remaining / (indexed_tiles / elapsed) if indexed_tiles else 0
        log(
            f"[{sample_number}/{sample_count}] Indexed {row.sample_key}: {len(table):,} tiles "
            f"in {time.perf_counter() - sample_started:.1f}s | "
            f"cumulative {indexed_tiles:,}/{target_tiles:,} | "
            f"elapsed {timedelta(seconds=int(elapsed))} | "
            f"ETA {timedelta(seconds=int(eta_seconds))}"
        )

    if not sample_tables:
        raise ValueError("No tiles matched the requested scope.")
    log(f"Combining {sample_count:,} sample indexes...")
    tiles = pd.concat(sample_tables, ignore_index=True)
    log(f"Combined index: {len(tiles):,} rows; checking duplicate tile IDs...")
    duplicates = tiles["tile_id"].duplicated(keep=False)
    if duplicates.any():
        raise ValueError(f"Duplicate tile IDs found: {tiles.loc[duplicates, 'tile_id'].head().tolist()}")
    log(
        f"Canonical tile index ready: {len(tiles):,} tiles across "
        f"{tiles['sample_key'].nunique():,} samples in "
        f"{timedelta(seconds=int(time.perf_counter() - index_started))}"
    )
    return tiles



def safe_path_component(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")


def output_path(output_root, model_name, dataset, sample):
    return (
        Path(output_root)
        / model_name
        / safe_path_component(dataset)
        / f"{safe_path_component(sample)}.pt"
    )


def load_shard(path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_shard(path, expected_table, expected_dim, *, model_name=None, recipe_version=None):
    import torch
    try:
        payload = load_shard(path)
        if model_name is not None and payload.get("model_name") != model_name:
            return False, "model identity mismatch"
        if recipe_version is not None and payload.get("recipe_version") != recipe_version:
            return False, "extraction recipe mismatch"
        embeddings = payload["embeddings"]
        expected_ids = expected_table["tile_id"].astype(str).tolist()
        observed_ids = list(map(str, payload["tile_ids"]))
        if not isinstance(embeddings, torch.Tensor):
            return False, "embeddings is not a tensor"
        if tuple(embeddings.shape) != (len(expected_ids), int(expected_dim)):
            return False, f"shape {tuple(embeddings.shape)} != {(len(expected_ids), int(expected_dim))}"
        if observed_ids != expected_ids:
            return False, "ordered tile IDs do not match the manifest"
        if payload.get("tile_id_checksum") != tile_id_checksum(expected_ids):
            return False, "tile-ID checksum does not match"
        if embeddings.dtype != torch.float32:
            return False, f"dtype is {embeddings.dtype}, expected torch.float32"
        if not bool(torch.isfinite(embeddings).all()):
            return False, "embeddings contain NaN or Inf"
        return True, "ok"
    except Exception as exc:
        return False, f"could not read shard: {exc}"


def run_extraction(tile_table, model_names, output_root, overwrite=False, *,
                   device="cpu", batch_size=32, num_workers=8,
                   recipe_version="stpath_brca_training_compatible_v1"):
    import torch
    import torchvision
    import timm
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from .stpath_foundation import (
        MODEL_NAMES, EXPECTED_DIMS, MODEL_SOURCES, TRANSFORM_DESCRIPTIONS,
        POOLING_DESCRIPTIONS, ManifestTileDataset, build_model, encode_batch,
        resolve_hf_revision,
    )
    device = torch.device(device)
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    unknown = set(model_names).difference(MODEL_NAMES)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    if tile_table.empty:
        raise ValueError("The tile table is empty.")

    grouped = list(tile_table.groupby(["dataset", "sample"], sort=False))
    total_tiles = len(tile_table) * len(model_names)
    completed_tiles = 0
    computed_tiles = 0
    run_start = time.monotonic()
    log(
        f"Extraction started: {len(model_names)} models, {len(grouped)} samples, "
        f"{total_tiles:,} model-tile evaluations"
    )

    for model_number, model_name in enumerate(model_names, start=1):
        log(f"MODEL {model_number}/{len(model_names)} START: {model_name}")
        model = None
        try:
            for sample_number, ((dataset, sample), sample_table) in enumerate(grouped, start=1):
                sample_table = sample_table.reset_index(drop=True)
                n_tiles = len(sample_table)
                destination = output_path(output_root, model_name, dataset, sample)
                destination.parent.mkdir(parents=True, exist_ok=True)

                if destination.is_file() and not overwrite:
                    valid, reason = validate_shard(
                        destination, sample_table, EXPECTED_DIMS[model_name],
                        model_name=model_name, recipe_version=recipe_version,
                    )
                    if valid:
                        completed_tiles += n_tiles
                        log(
                            f"SKIP {model_name} sample {sample_number}/{len(grouped)} "
                            f"{dataset}/{sample}: validated {n_tiles:,}-tile shard"
                        )
                        continue
                    log(f"RECOMPUTE {destination}: invalid existing shard ({reason})")

                if model is None:
                    model, transform = build_model(model_name, device)
                    resolved_revision = resolve_hf_revision(model_name)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                sample_start = time.monotonic()
                log(
                    f"START {model_name} sample {sample_number}/{len(grouped)}: "
                    f"{dataset}/{sample} ({n_tiles:,} tiles)"
                )

                dataset_object = ManifestTileDataset(sample_table, transform)
                loader = DataLoader(
                    dataset_object,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=device.type == "cuda",
                    persistent_workers=num_workers > 0,
                )
                embedding_chunks = []
                observed_tile_ids = []
                with torch.inference_mode():
                    progress = tqdm(
                        loader,
                        desc=f"{model_name} | {dataset}/{sample}",
                        unit="batch",
                        dynamic_ncols=True,
                        mininterval=2.0,
                        leave=True,
                    )
                    for images, batch_tile_ids in progress:
                        images = images.to(device, non_blocking=device.type == "cuda")
                        embeddings = encode_batch(model_name, model, images)
                        if embeddings.ndim != 2 or embeddings.shape[1] != EXPECTED_DIMS[model_name]:
                            raise ValueError(
                                f"{model_name} returned shape {tuple(embeddings.shape)}; "
                                f"expected [batch, {EXPECTED_DIMS[model_name]}]"
                            )
                        embedding_chunks.append(embeddings.detach().float().cpu())
                        observed_tile_ids.extend(map(str, batch_tile_ids))

                embeddings = torch.cat(embedding_chunks, dim=0)
                expected_tile_ids = sample_table["tile_id"].astype(str).tolist()
                if observed_tile_ids != expected_tile_ids:
                    raise ValueError(f"Tile order changed while processing {dataset}/{sample}")
                if not bool(torch.isfinite(embeddings).all()):
                    raise ValueError(f"Non-finite embeddings produced for {dataset}/{sample}")

                payload = {
                    "embeddings": embeddings,
                    "tile_ids": expected_tile_ids,
                    "tile_rows": torch.as_tensor(sample_table["tile_row"].to_numpy(), dtype=torch.int32),
                    "tile_cols": torch.as_tensor(sample_table["tile_col"].to_numpy(), dtype=torch.int32),
                    "source_paths": sample_table["materialized_tile_path"].astype(str).tolist(),
                    "tile_id_checksum": tile_id_checksum(expected_tile_ids),
                    "dataset": str(dataset),
                    "sample": str(sample),
                    "model_name": model_name,
                    "model_source": MODEL_SOURCES[model_name],
                    "resolved_hf_revision": resolved_revision,
                    "recipe_version": recipe_version,
                    "transform": TRANSFORM_DESCRIPTIONS[model_name],
                    "pooling": POOLING_DESCRIPTIONS[model_name],
                    "dtype": "float32",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "versions": {
                        "torch": torch.__version__,
                        "torchvision": torchvision.__version__,
                        "timm": timm.__version__,
                        "pandas": pd.__version__,
                    },
                }
                temporary = destination.with_suffix(".pt.tmp")
                torch.save(payload, temporary)
                os.replace(temporary, destination)

                elapsed = time.monotonic() - sample_start
                completed_tiles += n_tiles
                computed_tiles += n_tiles
                overall_elapsed = time.monotonic() - run_start
                rate = computed_tiles / overall_elapsed if computed_tiles else 0.0
                remaining = total_tiles - completed_tiles
                eta = timedelta(seconds=remaining / rate) if rate > 0 else "unknown"
                peak_text = ""
                if device.type == "cuda":
                    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
                    peak_text = f"; peak GPU memory {peak_gib:.1f} GiB"
                log(
                    f"DONE {model_name} {dataset}/{sample}: {n_tiles:,} tiles in "
                    f"{timedelta(seconds=int(elapsed))} ({n_tiles / elapsed:.1f} tiles/s){peak_text}; "
                    f"overall {completed_tiles:,}/{total_tiles:,}; ETA {eta}; saved {destination}"
                )

                del loader, dataset_object, embedding_chunks, embeddings, payload
                gc.collect()
        finally:
            if model is not None:
                del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()
        log(f"MODEL {model_number}/{len(model_names)} COMPLETE: {model_name}")

    log(f"Extraction complete in {timedelta(seconds=int(time.monotonic() - run_start))}")


def validate_outputs(tile_table, model_names, output_root):
    from tqdm.auto import tqdm
    from .stpath_foundation import EXPECTED_DIMS
    records = []
    grouped = list(tile_table.groupby(["dataset", "sample"], sort=False))
    for model_name in model_names:
        for (dataset, sample), sample_table in tqdm(
            grouped, desc=f"Validate {model_name}", unit="sample"
        ):
            path = output_path(output_root, model_name, dataset, sample)
            if path.is_file():
                valid, detail = validate_shard(path, sample_table, EXPECTED_DIMS[model_name],
                                               model_name=model_name,
                                               recipe_version="stpath_brca_training_compatible_v1")
            else:
                valid, detail = False, "missing"
            records.append({
                "model": model_name,
                "dataset": dataset,
                "sample": sample,
                "n_tiles": len(sample_table),
                "valid": valid,
                "detail": detail,
                "path": str(path),
            })
    result = pd.DataFrame(records)
    log(f"Validation complete: {int(result['valid'].sum())}/{len(result)} shards valid")
    return result