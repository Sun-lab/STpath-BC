"""Train STpath-style XGBoost models for Xenium tile cell-type proportions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


MODEL_NAMES = ("UNI2h", "Virchow", "Virchow2", "ProvGigapath", "Conch")
BASELINE_MODEL_NAME = "ResNet50"
ENCODER_NAMES = (*MODEL_NAMES, BASELINE_MODEL_NAME)
ALL_MODEL_NAMES = (*MODEL_NAMES, "Combined")
SUPPORTED_MODEL_NAMES = (*ALL_MODEL_NAMES, BASELINE_MODEL_NAME)
BOOST_PROGRESS_INTERVAL = 100
EXPECTED_DIMS = {
    "UNI2h": 1536,
    "Virchow": 2560,
    "Virchow2": 2560,
    "ProvGigapath": 1536,
    "Conch": 512,
    "ResNet50": 2048,
}
VALID_MATERIALIZATION_STATUSES = {"copy", "symlink", "move"}
MAPPING_COLUMNS = (
    "dataset",
    "sample",
    "individual_id",
    "individual_id_source",
    "review_status",
    "notes",
)
ANNOTATION_COLUMNS = ("dataset", "sample", "cell_id", "tile_id", "cell_type")
TARGET_REVIEW_COLUMNS = ("cell_type", "include", "review_status", "notes")


@dataclass(frozen=True)
class TrainingConfig:
    num_boost_round: int = 800
    seed: int = 42
    feature_fraction: float = 0.30
    n_quantiles: int = 100
    inner_calibration_folds: int = 5
    min_positive_tiles: int = 100
    min_positive_individuals: int = 3
    calibrate: bool = True
    nthread: int | None = None
    runtime_nthread: int | None = None
    combined_runtime_nthread: int | None = None
    parallel_model_jobs: int = 1
    include_resnet50: bool = False
    resnet50_only: bool = False

    def __post_init__(self) -> None:
        if self.num_boost_round <= 0:
            raise ValueError("num_boost_round must be positive")
        if not 0 < self.feature_fraction <= 1:
            raise ValueError("feature_fraction must be in (0, 1]")
        if self.n_quantiles < 2:
            raise ValueError("n_quantiles must be at least 2")
        if self.inner_calibration_folds < 2:
            raise ValueError("inner_calibration_folds must be at least 2")
        if self.min_positive_tiles < 1 or self.min_positive_individuals < 1:
            raise ValueError("target eligibility thresholds must be positive")
        if self.nthread is not None and self.nthread < 1:
            raise ValueError("nthread must be positive when provided")
        if self.runtime_nthread is not None and self.runtime_nthread < 1:
            raise ValueError("runtime_nthread must be positive when provided")
        if (
            self.combined_runtime_nthread is not None
            and self.combined_runtime_nthread < 1
        ):
            raise ValueError(
                "combined_runtime_nthread must be positive when provided"
            )
        if not 1 <= self.parallel_model_jobs <= len(self.encoder_names):
            raise ValueError(f"parallel_model_jobs must be between 1 and {len(self.encoder_names)}")
        if self.parallel_model_jobs > 1 and self.effective_nthread is None:
            raise ValueError("Parallel model training requires an explicit nthread")

    @property
    def encoder_names(self) -> tuple[str, ...]:
        if self.resnet50_only:
            return (BASELINE_MODEL_NAME,)
        return ENCODER_NAMES if self.include_resnet50 else MODEL_NAMES

    @property
    def model_names(self) -> tuple[str, ...]:
        return self.encoder_names if self.resnet50_only else (*self.encoder_names, "Combined")

    @property
    def effective_nthread(self) -> int | None:
        return self.runtime_nthread if self.runtime_nthread is not None else self.nthread

    @property
    def effective_combined_nthread(self) -> int | None:
        return (
            self.combined_runtime_nthread
            if self.combined_runtime_nthread is not None
            else self.effective_nthread
        )


def log(message: str) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{stamp}] {message}", flush=True)


def safe_name(value: object) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return result or "unnamed"


def tile_id_checksum(tile_ids: Iterable[object]) -> str:
    text = "\n".join(map(str, tile_ids))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def configuration_fingerprint(config: TrainingConfig) -> str:
    scientific_config = asdict(config)
    if not config.include_resnet50:
        scientific_config.pop("include_resnet50")
    if not config.resnet50_only:
        scientific_config.pop("resnet50_only")
    scientific_config.pop("parallel_model_jobs", None)
    scientific_config.pop("runtime_nthread", None)
    scientific_config.pop("combined_runtime_nthread", None)
    payload = json.dumps(scientific_config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def available_cpu_count() -> int | None:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count()


def validate_cpu_budget(config: TrainingConfig) -> None:
    available = available_cpu_count()
    foundation_nthread = config.effective_nthread
    combined_nthread = config.effective_combined_nthread
    if available is None:
        return
    foundation_requested = (
        config.parallel_model_jobs * foundation_nthread
        if foundation_nthread is not None
        else 0
    )
    combined_requested = combined_nthread or 0
    requested = max(foundation_requested, combined_requested)
    if requested > available:
        raise ValueError(
            f"XGBoost configuration requests {requested} CPU threads at peak "
            f"(foundation stage: {config.parallel_model_jobs} jobs x "
            f"{foundation_nthread} threads; Combined stage: "
            f"{combined_nthread} threads), "
            f"but only {available} CPUs are available"
        )


def _require_columns(table: pd.DataFrame, columns: Iterable[str], source: object) -> None:
    missing = sorted(set(columns).difference(table.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def _as_bool(series: pd.Series, source: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.casefold()
    allowed = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    unknown = sorted(set(normalized).difference(allowed))
    if unknown:
        raise ValueError(f"{source} contains invalid boolean values: {unknown}")
    return normalized.map(allowed).astype(bool)


def _atomic_json(payload: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _atomic_csv(table: pd.DataFrame, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    compression = "gzip" if destination.suffix == ".gz" else None
    table.to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, destination)


def _atomic_npz(path: str | Path, **arrays: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, destination)


def _atomic_copy(source: str | Path, destination: str | Path) -> None:
    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_name(destination_path.name + ".tmp")
    shutil.copyfile(source_path, temporary)
    os.replace(temporary, destination_path)


def _array_checksum(values: Sequence[Any] | np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def training_data_fingerprint(
    tile_targets: pd.DataFrame,
    cell_type: str,
    config: TrainingConfig,
) -> str:
    target_column = f"proportion__{cell_type}"
    _require_columns(tile_targets, ("tile_id", "individual_id", target_column), "tile targets")
    target = pd.to_numeric(tile_targets[target_column], errors="raise").to_numpy(dtype=np.float64)
    payload = {
        "cell_type": cell_type,
        "training_config_fingerprint": configuration_fingerprint(config),
        "tile_id_checksum": tile_id_checksum(tile_targets["tile_id"].astype(str)),
        "individual_id_checksum": tile_id_checksum(tile_targets["individual_id"].astype(str)),
        "target_checksum": _array_checksum(target),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_checkpoint_compatible_targets(
    target_path: str | Path,
    cell_type: str,
    checkpoint_root: str | Path,
    config: TrainingConfig,
) -> tuple[pd.DataFrame, str]:
    """Load targets using the CSV float parser that matches existing checkpoints."""
    path = Path(target_path)
    round_trip = pd.read_csv(
        path, dtype={"tile_id": "string"}, float_precision="round_trip"
    )
    root = Path(checkpoint_root)
    metadata_paths = sorted(
        [
            *root.glob("outer_fold_*/complete.json"),
            *root.glob("outer_fold_*/skipped.json"),
            *root.glob("outer_fold_*/**/model_metadata/*.json"),
        ]
    )
    if not metadata_paths:
        return round_trip, "round_trip"

    saved_signatures = {
        json.loads(metadata_path.read_text(encoding="utf-8")).get("checkpoint_signature")
        for metadata_path in metadata_paths
    }
    if None in saved_signatures or len(saved_signatures) != 1:
        raise ValueError(
            "Existing checkpoints do not share one checkpoint signature; use a new RUN_ID "
            "or regenerate the inconsistent folds"
        )
    saved_signature = next(iter(saved_signatures))

    candidates = [
        ("round_trip", round_trip),
        ("standard", pd.read_csv(path, dtype={"tile_id": "string"})),
    ]
    for parse_mode, targets in candidates:
        if training_data_fingerprint(targets, cell_type, config) == saved_signature:
            return targets, parse_mode

    raise ValueError(
        f"{path} does not reproduce the saved checkpoint signature with either "
        "round-trip or standard CSV float parsing; the target table or scientific "
        "training configuration changed"
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def resolve_project_path(value: object, project_root: str | Path) -> Path:
    root = Path(project_root).expanduser().resolve()
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        return root / path
    indices = [index for index, part in enumerate(path.parts) if part == root.name]
    if indices:
        return root.joinpath(*path.parts[indices[-1] + 1 :])
    return path


def load_retained_tiles(
    project_root: str | Path,
    selection_manifest: str | Path,
) -> pd.DataFrame:
    """Load retained per-tile manifests in their authoritative order."""
    root = Path(project_root).expanduser().resolve()
    selection_path = resolve_project_path(selection_manifest, root)
    selection = pd.read_csv(selection_path)
    _require_columns(
        selection,
        (
            "dataset",
            "sample",
            "materialized_tile_manifest",
            "passing_tile_ids_path",
            "n_final_selected_tiles",
        ),
        selection_path,
    )
    if selection.duplicated(["dataset", "sample"]).any():
        raise ValueError(f"{selection_path} contains duplicate dataset/sample rows")

    parts: list[pd.DataFrame] = []
    for sample_order, row in enumerate(selection.itertuples(index=False)):
        manifest_path = resolve_project_path(row.materialized_tile_manifest, root)
        table = pd.read_csv(manifest_path, dtype={"tile_id": "string"})
        _require_columns(
            table,
            ("dataset", "sample", "tile_id", "status"),
            manifest_path,
        )
        table = table.loc[
            table["status"].astype(str).isin(VALID_MATERIALIZATION_STATUSES)
        ].copy()
        expected = int(row.n_final_selected_tiles)
        if len(table) != expected:
            raise ValueError(
                f"{row.dataset}/{row.sample}: expected {expected:,} retained tiles, "
                f"found {len(table):,} in {manifest_path}"
            )
        if not table["dataset"].astype(str).eq(str(row.dataset)).all():
            raise ValueError(f"{manifest_path} contains an unexpected dataset")
        if not table["sample"].astype(str).eq(str(row.sample)).all():
            raise ValueError(f"{manifest_path} contains an unexpected sample")
        passing_path = resolve_project_path(row.passing_tile_ids_path, root)
        passing = pd.read_csv(passing_path, dtype={"tile_id": "string"})
        _require_columns(passing, ("tile_id", "n_cells"), passing_path)
        if passing["tile_id"].isna().any() or passing["tile_id"].duplicated().any():
            raise ValueError(f"{passing_path} contains missing or duplicate tile IDs")
        counts = passing.set_index("tile_id")["n_cells"]
        table["n_cells"] = table["tile_id"].map(counts)
        if table["n_cells"].isna().any():
            examples = table.loc[table["n_cells"].isna(), "tile_id"].head().tolist()
            raise ValueError(f"{passing_path} lacks cell counts for retained tiles: {examples}")
        table["n_cells"] = pd.to_numeric(table["n_cells"], errors="raise").astype(int)
        if (table["n_cells"] <= 0).any():
            raise ValueError(f"{manifest_path} contains a retained tile with no cells")
        table["sample_order"] = sample_order
        table["tile_order_within_sample"] = np.arange(len(table), dtype=np.int64)
        parts.append(table)

    if not parts:
        raise ValueError(f"No retained samples were found in {selection_path}")
    result = pd.concat(parts, ignore_index=True, sort=False)
    if result["tile_id"].isna().any() or result["tile_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Retained tile IDs must be nonempty")
    if result["tile_id"].duplicated().any():
        examples = result.loc[result["tile_id"].duplicated(False), "tile_id"].head().tolist()
        raise ValueError(f"Retained tile IDs are not globally unique: {examples}")
    return result.reset_index(drop=True)


def validate_individual_mapping(
    retained_tiles: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    require_approved: bool = True,
) -> pd.DataFrame:
    """Validate one explicit split group per retained sample and attach it to tiles."""
    _require_columns(retained_tiles, ("dataset", "sample", "tile_id"), "retained tiles")
    _require_columns(mapping, MAPPING_COLUMNS, "sample-to-individual mapping")
    mapping = mapping.loc[:, MAPPING_COLUMNS].copy()
    for column in MAPPING_COLUMNS:
        mapping[column] = mapping[column].fillna("").astype(str).str.strip()
    if mapping.duplicated(["dataset", "sample"]).any():
        raise ValueError("Sample-to-individual mapping contains duplicate dataset/sample rows")

    expected = set(map(tuple, retained_tiles[["dataset", "sample"]].astype(str).drop_duplicates().to_numpy()))
    observed = set(map(tuple, mapping[["dataset", "sample"]].to_numpy()))
    missing = sorted(expected.difference(observed))
    extra = sorted(observed.difference(expected))
    if missing or extra:
        raise ValueError(f"Sample-to-individual mapping key mismatch; missing={missing}, extra={extra}")
    if mapping["individual_id"].eq("").any():
        bad = mapping.loc[mapping["individual_id"].eq(""), ["dataset", "sample"]].values.tolist()
        raise ValueError(f"Missing individual_id values: {bad}")
    allowed_statuses = {"pending", "approved"}
    invalid_status = sorted(set(mapping["review_status"]).difference(allowed_statuses))
    if invalid_status:
        raise ValueError(f"Invalid individual mapping review_status values: {invalid_status}")
    if require_approved and not mapping["review_status"].eq("approved").all():
        pending = mapping.loc[~mapping["review_status"].eq("approved"), ["dataset", "sample"]]
        raise ValueError(f"Individual mappings require approval: {pending.values.tolist()}")

    return retained_tiles.merge(
        mapping[["dataset", "sample", "individual_id"]],
        on=["dataset", "sample"],
        how="left",
        validate="many_to_one",
        sort=False,
    )


def annotation_path(annotation_root: str | Path, dataset: object, sample: object) -> Path:
    return Path(annotation_root) / safe_name(dataset) / safe_name(sample) / "cell_annotations.csv.gz"


def discover_annotation_labels(
    retained_tiles: pd.DataFrame,
    annotation_root: str | Path,
) -> tuple[list[str], pd.DataFrame]:
    """Read only annotation labels and return their vocabulary plus a file audit."""
    labels: set[str] = set()
    audit: list[dict[str, Any]] = []
    samples = retained_tiles[["dataset", "sample"]].drop_duplicates()
    for row in samples.itertuples(index=False):
        path = annotation_path(annotation_root, row.dataset, row.sample)
        record: dict[str, Any] = {
            "dataset": row.dataset,
            "sample": row.sample,
            "annotation_path": str(path),
            "exists": path.is_file(),
            "n_cells": 0,
            "status": "missing",
            "error": "",
        }
        if path.is_file():
            try:
                table = pd.read_csv(path, usecols=["cell_type"], dtype={"cell_type": "string"})
                values = table["cell_type"].fillna("").astype(str).str.strip()
                if values.eq("").any():
                    raise ValueError("cell_type contains missing or empty values")
                labels.update(values.unique())
                record.update(n_cells=len(table), status="ready")
            except Exception as exc:
                record.update(status="invalid", error=str(exc))
        audit.append(record)
    return sorted(labels, key=str.casefold), pd.DataFrame(audit)


def target_review_template(cell_types: Iterable[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_type": sorted({str(value).strip() for value in cell_types if str(value).strip()}, key=str.casefold),
            "include": False,
            "review_status": "pending",
            "notes": "",
        }
    )


def add_combined_target(
    tile_targets: pd.DataFrame,
    cell_type: str,
    component_cell_types: Sequence[str],
) -> pd.DataFrame:
    """Add one tile target by summing disjoint component cell counts."""
    name = str(cell_type).strip()
    components = tuple(str(value).strip() for value in component_cell_types)
    if not name or len(components) < 2 or any(not value for value in components):
        raise ValueError("Combined targets require a name and at least two component cell types")
    if len(set(components)) != len(components) or name in components:
        raise ValueError("Combined target components must be unique and exclude the target name")
    count_columns = [f"count__{value}" for value in components]
    _require_columns(tile_targets, ("n_cells_total", *count_columns), "tile targets")

    result = tile_targets.copy()
    component_counts = result[count_columns].apply(pd.to_numeric, errors="raise")
    totals = pd.to_numeric(result["n_cells_total"], errors="raise")
    values = component_counts.to_numpy(dtype=float)
    total_values = totals.to_numpy(dtype=float)
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or not np.isfinite(total_values).all()
        or (total_values <= 0).any()
    ):
        raise ValueError("Combined target counts require finite nonnegative counts and positive totals")
    combined_counts = component_counts.sum(axis=1)
    if (combined_counts > totals).any():
        raise ValueError("Combined target counts exceed total annotated cells")
    result[f"count__{name}"] = combined_counts.astype(int)
    result[f"proportion__{name}"] = combined_counts / totals
    return result


def validate_target_review(
    observed_cell_types: Iterable[object],
    review: pd.DataFrame,
    *,
    require_approved: bool = True,
) -> pd.DataFrame:
    _require_columns(review, TARGET_REVIEW_COLUMNS, "target-cell-type review")
    result = review.loc[:, TARGET_REVIEW_COLUMNS].copy()
    result["cell_type"] = result["cell_type"].fillna("").astype(str).str.strip()
    result["review_status"] = result["review_status"].fillna("").astype(str).str.strip().str.casefold()
    result["notes"] = result["notes"].fillna("").astype(str)
    result["include"] = _as_bool(result["include"], "target-cell-type review include")
    if result["cell_type"].eq("").any() or result["cell_type"].duplicated().any():
        raise ValueError("Target-cell-type review requires unique, nonempty cell_type values")
    expected = {str(value).strip() for value in observed_cell_types}
    observed = set(result["cell_type"])
    if expected != observed:
        raise ValueError(
            "Target-cell-type vocabulary mismatch; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    invalid_status = sorted(set(result["review_status"]).difference({"pending", "approved"}))
    if invalid_status:
        raise ValueError(f"Invalid target review_status values: {invalid_status}")
    if require_approved and not result["review_status"].eq("approved").all():
        pending = result.loc[~result["review_status"].eq("approved"), "cell_type"].tolist()
        raise ValueError(f"Target cell types require approval: {pending}")
    return result


def build_tile_targets(
    retained_tiles: pd.DataFrame,
    annotation_root: str | Path,
    target_review: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate per-cell labels into complete retained-tile targets."""
    _require_columns(
        retained_tiles,
        ("dataset", "sample", "tile_id", "n_cells", "individual_id"),
        "retained tiles",
    )
    _require_columns(target_review, TARGET_REVIEW_COLUMNS, "target-cell-type review")
    include_mask = _as_bool(target_review["include"], "target-cell-type review include")
    included = target_review.loc[include_mask, "cell_type"].astype(str).tolist()
    if not included:
        raise ValueError("No cell types are approved for training")

    target_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    excluded_rows: list[pd.DataFrame] = []
    grouped = retained_tiles.groupby(["dataset", "sample"], sort=False, observed=True)
    for (dataset, sample), sample_tiles in grouped:
        path = annotation_path(annotation_root, dataset, sample)
        header = pd.read_csv(path, nrows=0)
        _require_columns(header, ANNOTATION_COLUMNS, path)
        cells = pd.read_csv(path, usecols=list(ANNOTATION_COLUMNS), dtype="string")
        for column in ANNOTATION_COLUMNS:
            cells[column] = cells[column].fillna("").astype(str).str.strip()
        if cells[list(ANNOTATION_COLUMNS)].eq("").any(axis=None):
            raise ValueError(f"{path} contains missing required values")
        if not cells["dataset"].eq(str(dataset)).all() or not cells["sample"].eq(str(sample)).all():
            raise ValueError(f"{path} contains rows for another dataset or sample")
        if cells["cell_id"].duplicated().any():
            examples = cells.loc[cells["cell_id"].duplicated(False), "cell_id"].head().tolist()
            raise ValueError(f"{path} contains duplicate cell IDs: {examples}")

        retained_ids = set(sample_tiles["tile_id"].astype(str))
        retained_mask = cells["tile_id"].isin(retained_ids)
        outside = cells.loc[~retained_mask, ["dataset", "sample", "cell_id", "tile_id", "cell_type"]]
        if not outside.empty:
            excluded_rows.append(outside)
        cells = cells.loc[retained_mask].copy()

        observed_counts = cells.groupby("tile_id", observed=True).size().rename("n_annotation_cells")
        expected_counts = sample_tiles.set_index("tile_id")["n_cells"].astype(int)
        coverage = pd.concat([expected_counts, observed_counts], axis=1).fillna(0)
        coverage["n_annotation_cells"] = coverage["n_annotation_cells"].astype(int)
        coverage["cell_count_difference"] = (
            coverage["n_annotation_cells"] - coverage["n_cells"]
        )
        mismatched = coverage.loc[coverage["n_cells"] != coverage["n_annotation_cells"]]
        expected_total = int(coverage["n_cells"].sum())
        observed_total = int(coverage["n_annotation_cells"].sum())
        if expected_total != observed_total:
            examples = mismatched.head().reset_index().to_dict("records")
            raise ValueError(
                f"{dataset}/{sample} has incomplete sample-level annotation coverage: "
                f"expected {expected_total:,} retained cells, found {observed_total:,}; "
                f"per-tile examples: {examples}"
            )

        counts = pd.crosstab(cells["tile_id"], cells["cell_type"])
        output = sample_tiles.copy().set_index("tile_id", drop=False)
        output["n_cells_total"] = observed_counts.reindex(output.index).fillna(0).astype(int)
        included_count_columns: list[str] = []
        for cell_type in included:
            column = f"count__{cell_type}"
            output[column] = counts.get(cell_type, pd.Series(0, index=counts.index)).reindex(
                output.index, fill_value=0
            ).astype(int)
            output[f"proportion__{cell_type}"] = output[column] / output["n_cells_total"]
            included_count_columns.append(column)
        output["n_cells_excluded"] = output["n_cells_total"] - output[included_count_columns].sum(axis=1)
        if (output["n_cells_excluded"] < 0).any():
            raise ValueError(f"{dataset}/{sample} included cell counts exceed total cells")
        output["proportion_excluded"] = output["n_cells_excluded"] / output["n_cells_total"]
        target_parts.append(output.reset_index(drop=True))
        audit_rows.append(
            {
                "dataset": dataset,
                "sample": sample,
                "annotation_path": str(path),
                "n_annotation_cells": len(cells) + len(outside),
                "n_cells_in_retained_tiles": len(cells),
                "n_cells_outside_retained_tiles": len(outside),
                "n_retained_tiles": len(output),
                "n_expected_retained_cells": expected_total,
                "n_tiles_with_count_mismatch": len(mismatched),
                "sum_absolute_tile_count_difference": int(
                    coverage["cell_count_difference"].abs().sum()
                ),
                "max_absolute_tile_count_difference": int(
                    coverage["cell_count_difference"].abs().max()
                ),
                "coverage_complete": True,
            }
        )

    result = pd.concat(target_parts, ignore_index=True, sort=False)
    proportion_columns = [column for column in result if column.startswith("proportion__")]
    values = result[proportion_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Tile target proportions must be finite and in [0, 1]")
    outside_result = (
        pd.concat(excluded_rows, ignore_index=True)
        if excluded_rows
        else pd.DataFrame(columns=ANNOTATION_COLUMNS)
    )
    return result, pd.DataFrame(audit_rows), outside_result


def summarize_target_eligibility(
    tile_targets: pd.DataFrame,
    cell_types: Sequence[str],
    config: TrainingConfig,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for cell_type in cell_types:
        column = f"proportion__{cell_type}"
        if column not in tile_targets:
            raise ValueError(f"Tile targets are missing {column}")
        positive = pd.to_numeric(tile_targets[column], errors="raise") > 0
        n_positive_tiles = int(positive.sum())
        n_positive_individuals = int(tile_targets.loc[positive, "individual_id"].nunique())
        reasons = []
        if n_positive_tiles < config.min_positive_tiles:
            reasons.append(f"positive tiles {n_positive_tiles} < {config.min_positive_tiles}")
        if n_positive_individuals < config.min_positive_individuals:
            reasons.append(
                f"positive individuals {n_positive_individuals} < {config.min_positive_individuals}"
            )
        records.append(
            {
                "cell_type": cell_type,
                "n_tiles": len(tile_targets),
                "n_positive_tiles": n_positive_tiles,
                "n_positive_individuals": n_positive_individuals,
                "eligible": not reasons,
                "exclusion_reason": "; ".join(reasons),
            }
        )
    return pd.DataFrame(records)


def _load_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required to load notebook-26 foundation feature shards."
        ) from exc
    return torch


def _load_xgboost() -> Any:
    try:
        import xgboost as xgb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "XGBoost is required for model training. Install it in the server environment."
        ) from exc
    return xgb


def feature_shard_path(feature_root: str | Path, model_name: str, dataset: object, sample: object) -> Path:
    return Path(feature_root) / model_name / safe_name(dataset) / f"{safe_name(sample)}.pt"


def load_feature_matrices(
    tile_targets: pd.DataFrame,
    feature_root: str | Path,
    *,
    expected_recipe_version: str = "stpath_brca_training_compatible_v1",
    model_names: Sequence[str] = MODEL_NAMES,
    show_progress: bool = False,
) -> dict[str, np.ndarray]:
    """Load five aligned feature matrices in tile-target row order."""
    torch = _load_torch()
    matrices: dict[str, np.ndarray] = {}
    grouped = list(tile_targets.groupby(["dataset", "sample"], sort=False, observed=True))
    total_shards = len(model_names) * len(grouped)
    completed_shards = 0

    def report_progress(label: str) -> None:
        if not show_progress:
            return
        width = 24
        filled = round(width * completed_shards / total_shards)
        bar = "#" * filled + "-" * (width - filled)
        percent = round(100 * completed_shards / total_shards)
        print(
            f"\rLoading foundation features [{bar}] "
            f"{completed_shards}/{total_shards} ({percent:3d}%) {label[:80]:<80}",
            end="",
            flush=True,
        )

    for model_name in model_names:
        if model_name not in EXPECTED_DIMS:
            raise ValueError(f"Unknown foundation model: {model_name}")
        chunks: list[np.ndarray] = []
        for (dataset, sample), sample_tiles in grouped:
            path = feature_shard_path(feature_root, model_name, dataset, sample)
            report_progress(f"{model_name}: {dataset}/{sample}")
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
            required = {"embeddings", "tile_ids", "tile_id_checksum", "model_name", "recipe_version"}
            missing = sorted(required.difference(payload))
            if missing:
                raise ValueError(f"{path} is missing payload fields: {missing}")
            expected_ids = sample_tiles["tile_id"].astype(str).tolist()
            observed_ids = list(map(str, payload["tile_ids"]))
            if observed_ids != expected_ids:
                raise ValueError(f"{path} tile order does not match tile targets")
            if payload["tile_id_checksum"] != tile_id_checksum(expected_ids):
                raise ValueError(f"{path} tile checksum does not match")
            if str(payload["model_name"]) != model_name:
                raise ValueError(f"{path} contains model_name={payload['model_name']!r}")
            if str(payload["recipe_version"]) != expected_recipe_version:
                raise ValueError(f"{path} uses recipe_version={payload['recipe_version']!r}")
            embeddings = payload["embeddings"]
            if not torch.is_tensor(embeddings):
                raise ValueError(f"{path} embeddings are not a tensor")
            expected_shape = (len(expected_ids), EXPECTED_DIMS[model_name])
            if tuple(embeddings.shape) != expected_shape:
                raise ValueError(f"{path} shape {tuple(embeddings.shape)} != {expected_shape}")
            if embeddings.dtype != torch.float32:
                raise ValueError(f"{path} dtype {embeddings.dtype} != torch.float32")
            array = embeddings.detach().cpu().numpy()
            if not np.isfinite(array).all():
                raise ValueError(f"{path} contains nonfinite embeddings")
            chunks.append(array)
            completed_shards += 1
        report_progress(f"{model_name}: concatenating")
        matrices[model_name] = np.concatenate(chunks, axis=0)
    if show_progress:
        report_progress("complete")
        print()
    row_counts = {name: matrix.shape[0] for name, matrix in matrices.items()}
    if set(row_counts.values()) != {len(tile_targets)}:
        raise ValueError(f"Feature matrix row-count mismatch: {row_counts}")
    return matrices


def xgboost_parameters(config: TrainingConfig) -> dict[str, Any]:
    params: dict[str, Any] = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "eta": 0.01,
        "gamma": 0,
        "min_child_weight": 5,
        "colsample_bytree": 0.05,
        "subsample": 0.5,
        "alpha": 0.1,
        "lambda": 0.01,
        "max_depth": 12,
        "seed": config.seed,
    }
    if config.effective_nthread is not None:
        params["nthread"] = int(config.effective_nthread)
    return params


def _boost_progress_callback(
    xgb: Any,
    label: str,
    total_rounds: int,
    *,
    interval: int = BOOST_PROGRESS_INTERVAL,
) -> Any:
    class ProgressCallback(xgb.callback.TrainingCallback):
        def after_iteration(
            self, model: Any, epoch: int, evals_log: Mapping[str, Any]
        ) -> bool:
            completed = epoch + 1
            if completed == 1 or completed % interval == 0 or completed == total_rounds:
                percent = 100.0 * completed / total_rounds
                log(
                    f"{label}: boosting round {completed}/{total_rounds} "
                    f"({percent:.1f}%)"
                )
            return False

    return ProgressCallback()


def train_booster(
    features: np.ndarray,
    target: np.ndarray,
    config: TrainingConfig,
    *,
    progress_label: str | None = None,
) -> Any:
    xgb = _load_xgboost()
    nthread = config.effective_nthread
    matrix_kwargs = {"nthread": int(nthread)} if nthread is not None else {}
    if progress_label is not None:
        log(f"{progress_label}: preparing XGBoost matrix")
    matrix = xgb.DMatrix(
        np.asarray(features, dtype=np.float32),
        label=np.asarray(target, dtype=float),
        **matrix_kwargs,
    )
    callbacks = None
    if progress_label is not None:
        log(f"{progress_label}: starting {config.num_boost_round} boosting rounds")
        callbacks = [
            _boost_progress_callback(
                xgb,
                progress_label,
                config.num_boost_round,
            )
        ]
    return xgb.train(
        xgboost_parameters(config),
        matrix,
        num_boost_round=config.num_boost_round,
        verbose_eval=False,
        callbacks=callbacks,
    )


def predict_booster(
    booster: Any,
    features: np.ndarray,
    *,
    nthread: int | None = None,
) -> np.ndarray:
    xgb = _load_xgboost()
    matrix_kwargs = {"nthread": int(nthread)} if nthread is not None else {}
    matrix = xgb.DMatrix(np.asarray(features, dtype=np.float32), **matrix_kwargs)
    return np.asarray(booster.predict(matrix), dtype=float)


def gain_vector(booster: Any, n_features: int) -> np.ndarray:
    result = np.zeros(int(n_features), dtype=float)
    for name, value in booster.get_score(importance_type="gain").items():
        if not str(name).startswith("f"):
            raise ValueError(f"Unexpected XGBoost feature name: {name}")
        index = int(str(name)[1:])
        if not 0 <= index < len(result):
            raise ValueError(f"Feature importance index {index} is outside width {len(result)}")
        result[index] = float(value)
    return result


def select_top_features(gains: Sequence[float], fraction: float = 0.30) -> np.ndarray:
    gains_array = np.asarray(gains, dtype=float)
    if gains_array.ndim != 1 or not np.isfinite(gains_array).all():
        raise ValueError("Feature gains must be a finite one-dimensional array")
    count = max(1, round(len(gains_array) * float(fraction)))
    indices = np.arange(len(gains_array), dtype=int)
    order = np.lexsort((indices, -gains_array))
    return order[:count]


def concatenate_selected_features(
    feature_matrices: Mapping[str, np.ndarray],
    selections: Mapping[str, Sequence[int]],
    rows: Sequence[int] | np.ndarray,
) -> np.ndarray:
    row_indices = np.asarray(rows, dtype=int)
    chunks = [
        np.asarray(feature_matrices[name])[np.ix_(row_indices, np.asarray(selections[name], dtype=int))]
        for name in MODEL_NAMES
    ]
    return np.concatenate(chunks, axis=1)


def validate_feature_matrices(
    feature_matrices: Mapping[str, np.ndarray],
    n_rows: int,
    *,
    expected_dims: Mapping[str, int] | None = None,
    model_names: Sequence[str] | None = None,
) -> dict[str, int]:
    """Validate aligned pathology matrices and the optional ResNet50 baseline."""
    if model_names is None:
        model_names = ENCODER_NAMES if BASELINE_MODEL_NAME in feature_matrices else MODEL_NAMES
    if set(model_names).difference(ENCODER_NAMES):
        raise ValueError("Unknown encoder names")
    missing = sorted(set(model_names).difference(feature_matrices))
    extra = sorted(set(feature_matrices).difference(model_names))
    if missing or extra:
        raise ValueError(f"Foundation feature key mismatch; missing={missing}, extra={extra}")
    widths: dict[str, int] = {}
    for model_name in model_names:
        matrix = np.asarray(feature_matrices[model_name])
        if matrix.ndim != 2 or matrix.shape[0] != n_rows:
            raise ValueError(
                f"{model_name} feature shape {matrix.shape} must have {n_rows} rows and two dimensions"
            )
        if expected_dims is not None and matrix.shape[1] != int(expected_dims[model_name]):
            raise ValueError(
                f"{model_name} feature width {matrix.shape[1]} != {int(expected_dims[model_name])}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"{model_name} features contain nonfinite values")
        widths[model_name] = int(matrix.shape[1])
    return widths


def leave_one_group_out(groups: Sequence[object]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    values = np.asarray(list(map(str, groups)), dtype=object)
    unique = sorted(set(values), key=str.casefold)
    if len(unique) < 2:
        raise ValueError("LOIO evaluation requires at least two individuals")
    return [
        (group, np.flatnonzero(values != group), np.flatnonzero(values == group))
        for group in unique
    ]


def balanced_group_folds(groups: Sequence[object], n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Assign whole groups to deterministic folds while balancing row counts."""
    values = np.asarray(list(map(str, groups)), dtype=object)
    unique, counts = np.unique(values, return_counts=True)
    folds = min(int(n_splits), len(unique))
    if folds < 2:
        raise ValueError("Grouped cross-fitting requires at least two individuals")
    ranked = sorted(zip(unique.tolist(), counts.tolist()), key=lambda item: (-item[1], item[0].casefold()))
    fold_groups: list[list[str]] = [[] for _ in range(folds)]
    fold_sizes = [0] * folds
    for group, count in ranked:
        fold = min(range(folds), key=lambda index: (fold_sizes[index], index))
        fold_groups[fold].append(group)
        fold_sizes[fold] += int(count)
    result = []
    for held_groups in fold_groups:
        test = np.flatnonzero(np.isin(values, held_groups))
        train = np.flatnonzero(~np.isin(values, held_groups))
        result.append((train, test))
    return result


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(truth) & np.isfinite(prediction)
    truth = truth[finite]
    prediction = prediction[finite]
    if not len(truth):
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "pearson": np.nan, "spearman": np.nan, "correlation_status": "no finite pairs"}
    mae = float(np.mean(np.abs(truth - prediction)))
    rmse = float(np.sqrt(np.mean((truth - prediction) ** 2)))
    if len(truth) < 2:
        pearson = spearman = np.nan
        status = "fewer than two finite pairs"
    elif np.ptp(truth) == 0 or np.ptp(prediction) == 0:
        pearson = spearman = np.nan
        status = "constant vector"
    else:
        pearson = float(np.corrcoef(truth, prediction)[0, 1])
        truth_rank = pd.Series(truth).rank(method="average").to_numpy()
        prediction_rank = pd.Series(prediction).rank(method="average").to_numpy()
        spearman = float(np.corrcoef(truth_rank, prediction_rank)[0, 1])
        status = "ok"
    return {
        "n": len(truth),
        "mae": mae,
        "rmse": rmse,
        "pearson": pearson,
        "spearman": spearman,
        "correlation_status": status,
    }


def fit_quantile_calibrator(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    n_quantiles: int = 100,
) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(truth) & np.isfinite(prediction)
    truth = truth[finite]
    prediction = prediction[finite]
    if not len(truth):
        raise ValueError("Calibration requires at least one finite prediction/target pair")
    if np.ptp(prediction) == 0:
        return {
            "method": "constant",
            "constant": float(np.median(truth)),
            "n_pairs": int(len(truth)),
            "n_quantiles": int(n_quantiles),
        }
    levels = np.linspace(0.0, 1.0, int(n_quantiles))
    x = np.quantile(prediction, levels)
    y = np.quantile(truth, levels)
    unique_x, inverse = np.unique(x, return_inverse=True)
    y_sums = np.zeros(len(unique_x), dtype=float)
    y_counts = np.zeros(len(unique_x), dtype=int)
    np.add.at(y_sums, inverse, y)
    np.add.at(y_counts, inverse, 1)
    unique_y = y_sums / y_counts
    if len(unique_x) == 1:
        return {
            "method": "constant",
            "constant": float(unique_y[0]),
            "n_pairs": int(len(truth)),
            "n_quantiles": int(n_quantiles),
        }
    return {
        "method": "quantile_mapping",
        "prediction_knots": unique_x.tolist(),
        "target_knots": unique_y.tolist(),
        "n_pairs": int(len(truth)),
        "n_quantiles": int(n_quantiles),
    }


def apply_quantile_calibrator(calibrator: Mapping[str, Any], predictions: Sequence[float]) -> np.ndarray:
    values = np.asarray(predictions, dtype=float)
    if calibrator["method"] == "constant":
        calibrated = np.full(values.shape, float(calibrator["constant"]), dtype=float)
    elif calibrator["method"] == "quantile_mapping":
        calibrated = np.interp(
            values,
            np.asarray(calibrator["prediction_knots"], dtype=float),
            np.asarray(calibrator["target_knots"], dtype=float),
        )
    else:
        raise ValueError(f"Unknown calibration method: {calibrator['method']}")
    return np.clip(calibrated, 0.0, 1.0)


def _fit_or_load_checkpointed_model(
    model_name: str,
    train_features: np.ndarray,
    train_target: np.ndarray,
    test_features: np.ndarray,
    config: TrainingConfig,
    *,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    checkpoint_dir: Path | None,
    checkpoint_signature: str | None,
    overwrite: bool,
    select_features: bool,
    context: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    expected_metadata = {
        "checkpoint_version": 1,
        "checkpoint_signature": checkpoint_signature,
        "model_type": model_name,
        "training_config_fingerprint": configuration_fingerprint(config),
        "input_dimension": int(train_features.shape[1]),
        "n_train_rows": int(len(train_indices)),
        "n_test_rows": int(len(test_indices)),
        "train_indices_checksum": _array_checksum(np.asarray(train_indices, dtype=np.int64)),
        "test_indices_checksum": _array_checksum(np.asarray(test_indices, dtype=np.int64)),
    }
    model_path = result_path = metadata_path = None
    if checkpoint_dir is not None:
        if checkpoint_signature is None:
            raise ValueError("checkpoint_signature is required when checkpoint_dir is set")
        model_path = checkpoint_dir / "models" / f"{model_name}.ubj"
        result_path = checkpoint_dir / "results" / f"{model_name}.npz"
        metadata_path = checkpoint_dir / "model_metadata" / f"{model_name}.json"
        if metadata_path.is_file() and not overwrite:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            mismatched = [
                key for key, value in expected_metadata.items() if metadata.get(key) != value
            ]
            if mismatched:
                raise ValueError(
                    f"Incompatible checkpoint {metadata_path}; mismatched fields={mismatched}. "
                    "Use a new RUN_ID or set OVERWRITE=True."
                )
            if not model_path.is_file() or model_path.stat().st_size == 0:
                raise ValueError(f"Committed checkpoint is missing its model: {model_path}")
            if not result_path.is_file() or result_path.stat().st_size == 0:
                raise ValueError(f"Committed checkpoint is missing its results: {result_path}")
            with np.load(result_path, allow_pickle=False) as saved:
                prediction = np.asarray(saved["prediction"], dtype=float)
                gains = np.asarray(saved["gains"], dtype=float) if select_features else None
                selections = (
                    np.asarray(saved["selections"], dtype=int) if select_features else None
                )
            if prediction.shape != (len(test_indices),) or not np.isfinite(prediction).all():
                raise ValueError(f"Checkpoint predictions are invalid: {result_path}")
            if select_features:
                if gains is None or gains.shape != (train_features.shape[1],):
                    raise ValueError(f"Checkpoint gains are invalid: {result_path}")
                if (
                    selections is None
                    or selections.ndim != 1
                    or not len(selections)
                    or (selections < 0).any()
                    or (selections >= train_features.shape[1]).any()
                ):
                    raise ValueError(f"Checkpoint selections are invalid: {result_path}")
            log(f"{context}: loading checkpoint for {model_name}")
            return prediction, gains, selections

    log(f"{context}: fitting {model_name}")
    progress_label = f"{context}: {model_name}" if context else model_name
    booster = train_booster(
        train_features,
        train_target,
        config,
        progress_label=progress_label,
    )
    prediction = predict_booster(
        booster, test_features, nthread=config.effective_nthread
    )
    gains = gain_vector(booster, train_features.shape[1]) if select_features else None
    selections = (
        select_top_features(gains, config.feature_fraction) if gains is not None else None
    )
    if checkpoint_dir is not None:
        assert model_path is not None and result_path is not None and metadata_path is not None
        _save_booster(booster, model_path)
        arrays = {"prediction": prediction}
        if gains is not None and selections is not None:
            arrays.update({"gains": gains, "selections": selections})
        _atomic_npz(result_path, **arrays)
        _atomic_json(
            {
                **expected_metadata,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            metadata_path,
        )
    return prediction, gains, selections


def _fit_six_models(
    feature_matrices: Mapping[str, np.ndarray],
    target: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    config: TrainingConfig,
    *,
    context: str = "",
    checkpoint_dir: str | Path | None = None,
    checkpoint_signature: str | None = None,
    overwrite: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    predictions: dict[str, np.ndarray] = {}
    gains: dict[str, np.ndarray] = {}
    selections: dict[str, np.ndarray] = {}
    destination = Path(checkpoint_dir) if checkpoint_dir is not None else None

    def fit_foundation_model(
        model_name: str,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        matrix = feature_matrices[model_name]
        return _fit_or_load_checkpointed_model(
            model_name,
            matrix[train_indices],
            target[train_indices],
            matrix[test_indices],
            config,
            train_indices=train_indices,
            test_indices=test_indices,
            checkpoint_dir=destination,
            checkpoint_signature=checkpoint_signature,
            overwrite=overwrite,
            select_features=True,
            context=context,
        )

    if config.parallel_model_jobs == 1:
        fitted = map(fit_foundation_model, config.encoder_names)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=config.parallel_model_jobs)
        fitted = executor.map(fit_foundation_model, config.encoder_names)
    try:
        model_results = dict(zip(config.encoder_names, fitted))
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    for model_name in config.encoder_names:
        prediction, model_gains, model_selections = model_results[model_name]
        assert model_gains is not None and model_selections is not None
        predictions[model_name] = prediction
        gains[model_name] = model_gains
        selections[model_name] = model_selections
    if config.resnet50_only:
        return predictions, gains, selections
    combined_train = concatenate_selected_features(feature_matrices, selections, train_indices)
    combined_test = concatenate_selected_features(feature_matrices, selections, test_indices)
    combined_config = replace(
        config, runtime_nthread=config.effective_combined_nthread
    )
    predictions["Combined"], _combined_gains, _combined_selections = (
        _fit_or_load_checkpointed_model(
            "Combined",
            combined_train,
            target[train_indices],
            combined_test,
            combined_config,
            train_indices=train_indices,
            test_indices=test_indices,
            checkpoint_dir=destination,
            checkpoint_signature=checkpoint_signature,
            overwrite=overwrite,
            select_features=False,
            context=context,
        )
    )
    return predictions, gains, selections


def nested_calibration_predictions(
    feature_matrices: Mapping[str, np.ndarray],
    target: np.ndarray,
    groups: np.ndarray,
    outer_train_indices: np.ndarray,
    config: TrainingConfig,
    *,
    checkpoint_dir: str | Path | None = None,
    checkpoint_signature: str | None = None,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    """Generate inner OOF predictions with all selection confined to inner training data."""
    local_groups = groups[outer_train_indices]
    local_folds = balanced_group_folds(local_groups, config.inner_calibration_folds)
    result = {name: np.full(len(outer_train_indices), np.nan, dtype=float) for name in config.model_names}
    destination = Path(checkpoint_dir) if checkpoint_dir is not None else None
    for fold_number, (local_train, local_test) in enumerate(local_folds, start=1):
        global_train = outer_train_indices[local_train]
        global_test = outer_train_indices[local_test]
        predictions, _gains, _selections = _fit_six_models(
            feature_matrices,
            target,
            global_train,
            global_test,
            config,
            context=f"inner calibration fold {fold_number}/{len(local_folds)}",
            checkpoint_dir=(
                destination / f"inner_fold_{fold_number:03d}" if destination is not None else None
            ),
            checkpoint_signature=checkpoint_signature,
            overwrite=overwrite,
        )
        for model_name, values in predictions.items():
            result[model_name][local_test] = values
    for model_name, values in result.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Inner calibration predictions are incomplete for {model_name}")
    return result


def _load_completed_fold(
    checkpoint_dir: Path | None,
    expected_metadata: Mapping[str, Any],
    n_test_rows: int,
    *,
    overwrite: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]] | None:
    if checkpoint_dir is None or overwrite:
        return None
    completion_path = checkpoint_dir / "complete.json"
    if not completion_path.is_file():
        return None
    metadata = json.loads(completion_path.read_text(encoding="utf-8"))
    mismatched = [
        key for key, value in expected_metadata.items() if metadata.get(key) != value
    ]
    if mismatched:
        raise ValueError(
            f"Incompatible completed fold {completion_path}; mismatched fields={mismatched}. "
            "Use a new RUN_ID or set OVERWRITE=True."
        )
    result_path = checkpoint_dir / "fold_result.npz"
    if not result_path.is_file() or result_path.stat().st_size == 0:
        raise ValueError(f"Completed fold is missing its result: {result_path}")
    raw: dict[str, np.ndarray] = {}
    calibrated: dict[str, np.ndarray] = {}
    with np.load(result_path, allow_pickle=False) as saved:
        for model_name in expected_metadata.get("model_types", ALL_MODEL_NAMES):
            raw[model_name] = np.asarray(saved[f"raw__{model_name}"], dtype=float)
            calibrated[model_name] = np.asarray(
                saved[f"calibrated__{model_name}"], dtype=float
            )
    if any(
        values.shape != (n_test_rows,) or not np.isfinite(values).all()
        for values in (*raw.values(), *calibrated.values())
    ):
        raise ValueError(f"Completed fold contains invalid predictions: {result_path}")
    log(f"loading completed outer-fold result from {result_path}")
    return raw, calibrated


def _save_completed_fold(
    checkpoint_dir: Path | None,
    metadata: Mapping[str, Any],
    raw: Mapping[str, np.ndarray],
    calibrated: Mapping[str, np.ndarray],
) -> None:
    if checkpoint_dir is None:
        return
    arrays = {
        **{f"raw__{name}": values for name, values in raw.items()},
        **{f"calibrated__{name}": values for name, values in calibrated.items()},
    }
    _atomic_npz(checkpoint_dir / "fold_result.npz", **arrays)
    _atomic_json(
        {**metadata, "completed_at_utc": datetime.now(timezone.utc).isoformat()},
        checkpoint_dir / "complete.json",
    )


def _save_skipped_fold(
    checkpoint_dir: Path | None,
    metadata: Mapping[str, Any],
    target: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    if checkpoint_dir is None:
        return
    held_out_target = target[test_indices]
    _atomic_json(
        {
            **metadata,
            "status": "skipped_all_zero",
            "reason": "all held-out tile target proportions are zero",
            "n_positive_test_tiles": int((held_out_target > 0).sum()),
            "observed_min": float(held_out_target.min()),
            "observed_max": float(held_out_target.max()),
            "skipped_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        checkpoint_dir / "skipped.json",
    )


def _metric_rows_for_fold(
    cell_type: str,
    fold_number: int,
    held_out: str,
    target: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    raw_predictions: Mapping[str, np.ndarray],
    calibrated_predictions: Mapping[str, np.ndarray],
    *,
    calibrate: bool,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows: list[dict[str, Any]] = []
    identity = {
        "cell_type": cell_type,
        "outer_fold": fold_number,
        "held_out_individual": held_out,
    }
    test_summary = {
        "n_test_tiles": len(test_indices),
        "n_positive_test_tiles": int((target[test_indices] > 0).sum()),
        "observed_min": float(target[test_indices].min()),
        "observed_max": float(target[test_indices].max()),
    }
    for model_name in raw_predictions:
        prediction_scales = [("raw", raw_predictions[model_name])]
        if calibrate:
            prediction_scales.append(
                ("calibrated", calibrated_predictions[model_name])
            )
        for scale, values in prediction_scales:
            rows.append(
                {
                    **identity,
                    "model_type": model_name,
                    "prediction_scale": scale,
                    **test_summary,
                    **regression_metrics(target[test_indices], values),
                }
            )
    baseline = np.full(len(test_indices), float(target[train_indices].mean()))
    rows.append(
        {
            **identity,
            "model_type": "TrainingMeanBaseline",
            "prediction_scale": "raw",
            **test_summary,
            **regression_metrics(target[test_indices], baseline),
        }
    )
    return rows, baseline


def evaluate_cell_type(
    feature_matrices: Mapping[str, np.ndarray],
    tile_targets: pd.DataFrame,
    cell_type: str,
    config: TrainingConfig,
    *,
    checkpoint_dir: str | Path | None = None,
    overwrite: bool = False,
    skip_all_zero_test_folds: bool = False,
    outer_folds: Iterable[int] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run leakage-free outer LOIO evaluation for all or selected folds."""
    target_column = f"proportion__{cell_type}"
    _require_columns(tile_targets, ("tile_id", "dataset", "sample", "individual_id", target_column), "tile targets")
    target = pd.to_numeric(tile_targets[target_column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(target).all() or (target < 0).any() or (target > 1).any():
        raise ValueError(f"{target_column} must contain finite values in [0, 1]")
    groups = tile_targets["individual_id"].astype(str).to_numpy()
    n_rows = len(tile_targets)
    validate_feature_matrices(feature_matrices, n_rows, model_names=config.encoder_names)
    checkpoint_signature = training_data_fingerprint(tile_targets, cell_type, config)
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None

    raw = {name: np.full(n_rows, np.nan, dtype=float) for name in config.model_names}
    calibrated = {name: np.full(n_rows, np.nan, dtype=float) for name in config.model_names}
    baseline_raw = np.full(n_rows, np.nan, dtype=float)
    outer_fold_numbers = np.zeros(n_rows, dtype=int)
    metric_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    fold_status_rows: list[dict[str, Any]] = []
    folds = leave_one_group_out(groups)
    requested_folds = None
    if outer_folds is not None:
        requested_folds = {int(value) for value in outer_folds}
        invalid = sorted(
            value for value in requested_folds if value < 1 or value > len(folds)
        )
        if not requested_folds or invalid:
            raise ValueError(
                f"outer_folds must contain fold numbers from 1 to {len(folds)}; "
                f"invalid={invalid}"
            )
    for fold_number, (held_out, train_indices, test_indices) in enumerate(folds, start=1):
        if requested_folds is not None and fold_number not in requested_folds:
            continue
        log(f"{cell_type}: outer fold {fold_number}/{len(folds)}; held out {held_out}")
        outer_checkpoint = (
            checkpoint_root
            / f"outer_fold_{fold_number:03d}__heldout_{safe_name(held_out)}"
            if checkpoint_root is not None
            else None
        )
        fold_metadata = {
            "checkpoint_version": 1,
            **({"model_types": list(config.model_names)} if config.include_resnet50 or config.resnet50_only else {}),
            "checkpoint_signature": checkpoint_signature,
            "outer_fold": fold_number,
            "held_out_individual": held_out,
            "test_indices_checksum": _array_checksum(
                np.asarray(test_indices, dtype=np.int64)
            ),
            "n_test_rows": int(len(test_indices)),
        }
        completion_exists = (
            outer_checkpoint is not None
            and (outer_checkpoint / "complete.json").is_file()
        )
        if (
            skip_all_zero_test_folds
            and not completion_exists
            and np.all(target[test_indices] == 0)
        ):
            _save_skipped_fold(
                outer_checkpoint, fold_metadata, target, test_indices
            )
            fold_status_rows.append(
                {
                    "outer_fold": fold_number,
                    "held_out_individual": held_out,
                    "status": "skipped_all_zero",
                    "n_test_rows": len(test_indices),
                    "n_positive_test_tiles": 0,
                    "observed_min": 0.0,
                    "observed_max": 0.0,
                    "checkpoint_dir": str(outer_checkpoint or ""),
                }
            )
            log(
                f"{cell_type} outer fold {fold_number}/{len(folds)}: "
                "skipped because all held-out target proportions are zero"
            )
            continue
        predictions, gains, selections = _fit_six_models(
            feature_matrices,
            target,
            train_indices,
            test_indices,
            config,
            context=f"{cell_type} outer fold {fold_number}/{len(folds)}",
            checkpoint_dir=(outer_checkpoint / "outer" if outer_checkpoint is not None else None),
            checkpoint_signature=checkpoint_signature,
            overwrite=overwrite,
        )
        completed_fold = _load_completed_fold(
            outer_checkpoint, fold_metadata, len(test_indices), overwrite=overwrite
        )
        if completed_fold is not None:
            predictions, calibrated_predictions = completed_fold
        else:
            inner_predictions = (
                nested_calibration_predictions(
                    feature_matrices,
                    target,
                    groups,
                    train_indices,
                    config,
                    checkpoint_dir=outer_checkpoint,
                    checkpoint_signature=checkpoint_signature,
                    overwrite=overwrite,
                )
                if config.calibrate
                else None
            )
            calibrated_predictions: dict[str, np.ndarray] = {}
            for model_name in config.model_names:
                if inner_predictions is None:
                    calibrated_predictions[model_name] = np.clip(
                        predictions[model_name], 0.0, 1.0
                    )
                else:
                    calibrator = fit_quantile_calibrator(
                        target[train_indices],
                        inner_predictions[model_name],
                        n_quantiles=config.n_quantiles,
                    )
                    calibrated_predictions[model_name] = apply_quantile_calibrator(
                        calibrator, predictions[model_name]
                    )
            _save_completed_fold(
                outer_checkpoint,
                fold_metadata,
                predictions,
                calibrated_predictions,
            )
        for model_name in config.model_names:
            raw[model_name][test_indices] = predictions[model_name]
            calibrated_values = calibrated_predictions[model_name]
            calibrated[model_name][test_indices] = calibrated_values
        fold_metric_rows, baseline = _metric_rows_for_fold(
            cell_type,
            fold_number,
            held_out,
            target,
            train_indices,
            test_indices,
            predictions,
            calibrated_predictions,
            calibrate=config.calibrate,
        )
        metric_rows.extend(fold_metric_rows)
        baseline_raw[test_indices] = baseline
        outer_fold_numbers[test_indices] = fold_number
        fold_status_rows.append(
            {
                "outer_fold": fold_number,
                "held_out_individual": held_out,
                "status": "complete",
                "n_test_rows": len(test_indices),
                "n_positive_test_tiles": int((target[test_indices] > 0).sum()),
                "observed_min": float(target[test_indices].min()),
                "observed_max": float(target[test_indices].max()),
                "checkpoint_dir": str(outer_checkpoint or ""),
            }
        )
        for encoder in config.encoder_names:
            for rank, feature_index in enumerate(selections[encoder], start=1):
                selection_rows.append(
                    {
                        "cell_type": cell_type,
                        "selection_scope": "outer_fold",
                        "outer_fold": fold_number,
                        "held_out_individual": held_out,
                        "encoder": encoder,
                        "rank": rank,
                        "feature_index": int(feature_index),
                        "feature_name": f"f{int(feature_index)}",
                        "gain": float(gains[encoder][feature_index]),
                    }
                )

    evaluated_rows = outer_fold_numbers > 0
    for model_name in config.model_names:
        if not np.isfinite(raw[model_name][evaluated_rows]).all():
            raise ValueError(f"Outer predictions are incomplete for {model_name}")
        if config.calibrate and not np.isfinite(
            calibrated[model_name][evaluated_rows]
        ).all():
            raise ValueError(f"Calibrated outer predictions are incomplete for {model_name}")
    identity = tile_targets.loc[
        evaluated_rows, ["tile_id", "dataset", "sample", "individual_id"]
    ].reset_index(drop=True)
    prediction_table = identity.copy()
    prediction_table["cell_type"] = cell_type
    prediction_table["outer_fold"] = outer_fold_numbers[evaluated_rows]
    prediction_table["observed_proportion"] = target[evaluated_rows]
    for model_name in config.model_names:
        prediction_table[f"predicted_raw__{model_name}"] = raw[model_name][evaluated_rows]
        if config.calibrate:
            prediction_table[f"predicted_calibrated__{model_name}"] = calibrated[
                model_name
            ][evaluated_rows]
    prediction_table["predicted_raw__TrainingMeanBaseline"] = baseline_raw[
        evaluated_rows
    ]
    fold_status = pd.DataFrame(fold_status_rows)
    fold_status["n_completed_folds"] = int(fold_status["status"].eq("complete").sum())
    fold_status["n_skipped_folds"] = int(
        fold_status["status"].eq("skipped_all_zero").sum()
    )
    fold_status["n_expected_folds"] = len(folds)
    return {
        "predictions": prediction_table,
        "fold_metrics": pd.DataFrame(metric_rows),
        "selected_features": pd.DataFrame(selection_rows),
        "fold_status": fold_status,
    }


def skip_outer_fold_if_all_zero(
    tile_targets: pd.DataFrame,
    cell_type: str,
    outer_fold: int,
    checkpoint_root: str | Path,
    config: TrainingConfig,
) -> bool:
    """Record an all-zero held-out fold without loading feature matrices."""
    target_column = f"proportion__{cell_type}"
    _require_columns(
        tile_targets,
        ("tile_id", "individual_id", target_column),
        "tile targets",
    )
    target = pd.to_numeric(tile_targets[target_column], errors="raise").to_numpy(
        dtype=float
    )
    if not np.isfinite(target).all() or (target < 0).any() or (target > 1).any():
        raise ValueError(f"{target_column} must contain finite values in [0, 1]")
    groups = tile_targets["individual_id"].astype(str).to_numpy()
    folds = leave_one_group_out(groups)
    fold_number = int(outer_fold)
    if fold_number < 1 or fold_number > len(folds):
        raise ValueError(
            f"outer_fold must be between 1 and {len(folds)}; got {fold_number}"
        )
    held_out, _train_indices, test_indices = folds[fold_number - 1]
    if not np.all(target[test_indices] == 0):
        return False
    fold_dir = (
        Path(checkpoint_root)
        / f"outer_fold_{fold_number:03d}__heldout_{safe_name(held_out)}"
    )
    if (fold_dir / "complete.json").is_file():
        return False
    metadata = {
        "checkpoint_version": 1,
        "checkpoint_signature": training_data_fingerprint(
            tile_targets, cell_type, config
        ),
        "outer_fold": fold_number,
        "held_out_individual": held_out,
        "test_indices_checksum": _array_checksum(
            np.asarray(test_indices, dtype=np.int64)
        ),
        "n_test_rows": int(len(test_indices)),
    }
    _save_skipped_fold(fold_dir, metadata, target, test_indices)
    log(
        f"{cell_type} outer fold {fold_number}/{len(folds)}: "
        "skipped before feature loading because all held-out targets are zero"
    )
    return True


def collect_completed_fold_metrics(
    tile_targets: pd.DataFrame,
    cell_type: str,
    checkpoint_root: str | Path,
    config: TrainingConfig,
    *,
    outer_folds: Iterable[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read committed outer folds and calculate an in-progress metric snapshot."""
    target_column = f"proportion__{cell_type}"
    _require_columns(
        tile_targets,
        ("tile_id", "dataset", "sample", "individual_id", target_column),
        "tile targets",
    )
    target = pd.to_numeric(tile_targets[target_column], errors="raise").to_numpy(
        dtype=float
    )
    if not np.isfinite(target).all() or (target < 0).any() or (target > 1).any():
        raise ValueError(f"{target_column} must contain finite values in [0, 1]")

    groups = tile_targets["individual_id"].astype(str).to_numpy()
    folds = leave_one_group_out(groups)
    signature = training_data_fingerprint(tile_targets, cell_type, config)
    root = Path(checkpoint_root)
    metric_rows: list[dict[str, Any]] = []
    prediction_tables: list[pd.DataFrame] = []
    status_rows: list[dict[str, Any]] = []

    requested_folds = None
    if outer_folds is not None:
        requested_folds = {int(value) for value in outer_folds}
        invalid = sorted(
            value for value in requested_folds if value < 1 or value > len(folds)
        )
        if not requested_folds or invalid:
            raise ValueError(
                f"outer_folds must contain fold numbers from 1 to {len(folds)}; "
                f"invalid={invalid}"
            )

    for fold_number, (held_out, train_indices, test_indices) in enumerate(
        folds, start=1
    ):
        if requested_folds is not None and fold_number not in requested_folds:
            continue
        fold_dir = root / f"outer_fold_{fold_number:03d}__heldout_{safe_name(held_out)}"
        completion_path = fold_dir / "complete.json"
        skipped_path = fold_dir / "skipped.json"
        expected_metadata = {
            "checkpoint_version": 1,
            **({"model_types": list(config.model_names)} if config.include_resnet50 or config.resnet50_only else {}),
            "checkpoint_signature": signature,
            "outer_fold": fold_number,
            "held_out_individual": held_out,
            "test_indices_checksum": _array_checksum(
                np.asarray(test_indices, dtype=np.int64)
            ),
            "n_test_rows": int(len(test_indices)),
        }
        status_row = {
            "outer_fold": fold_number,
            "held_out_individual": held_out,
            "n_test_rows": len(test_indices),
            "checkpoint_dir": str(fold_dir),
            "status": "incomplete",
            "completed_at_utc": "",
            "skipped_at_utc": "",
            "skip_reason": "",
            "error": "",
        }
        if not completion_path.is_file():
            if skipped_path.is_file():
                try:
                    skipped_metadata = json.loads(
                        skipped_path.read_text(encoding="utf-8")
                    )
                    expected_skip_metadata = {
                        **expected_metadata,
                        "status": "skipped_all_zero",
                        "n_positive_test_tiles": 0,
                        "observed_min": 0.0,
                        "observed_max": 0.0,
                    }
                    mismatched = [
                        key
                        for key, value in expected_skip_metadata.items()
                        if skipped_metadata.get(key) != value
                    ]
                    if mismatched:
                        raise ValueError(
                            f"Incompatible skipped fold {skipped_path}; "
                            f"mismatched fields={mismatched}"
                        )
                    status_row["status"] = "skipped_all_zero"
                    status_row["skipped_at_utc"] = skipped_metadata.get(
                        "skipped_at_utc", ""
                    )
                    status_row["skip_reason"] = skipped_metadata.get("reason", "")
                except Exception as exc:
                    status_row["status"] = "invalid"
                    status_row["error"] = str(exc)
            status_rows.append(status_row)
            continue
        try:
            completed = _load_completed_fold(
                fold_dir, expected_metadata, len(test_indices), overwrite=False
            )
            if completed is None:
                status_rows.append(status_row)
                continue
            raw_predictions, calibrated_predictions = completed
            completion_metadata = json.loads(
                completion_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            status_row["status"] = "invalid"
            status_row["error"] = str(exc)
            status_rows.append(status_row)
            continue

        fold_metric_rows, baseline = _metric_rows_for_fold(
            cell_type,
            fold_number,
            held_out,
            target,
            train_indices,
            test_indices,
            raw_predictions,
            calibrated_predictions,
            calibrate=config.calibrate,
        )
        metric_rows.extend(fold_metric_rows)

        predictions = tile_targets.iloc[test_indices][
            ["tile_id", "dataset", "sample", "individual_id"]
        ].copy()
        predictions.insert(0, "_row_index", test_indices)
        predictions["cell_type"] = cell_type
        predictions["outer_fold"] = fold_number
        predictions["observed_proportion"] = target[test_indices]
        for model_name in config.model_names:
            predictions[f"predicted_raw__{model_name}"] = raw_predictions[model_name]
            if config.calibrate:
                predictions[f"predicted_calibrated__{model_name}"] = (
                    calibrated_predictions[model_name]
                )
        predictions["predicted_raw__TrainingMeanBaseline"] = baseline
        prediction_tables.append(predictions)

        status_row["status"] = "complete"
        status_row["completed_at_utc"] = completion_metadata.get(
            "completed_at_utc", ""
        )
        status_rows.append(status_row)

    metric_columns = [
        "cell_type",
        "outer_fold",
        "held_out_individual",
        "model_type",
        "prediction_scale",
        "n_test_tiles",
        "n_positive_test_tiles",
        "observed_min",
        "observed_max",
        "n",
        "mae",
        "rmse",
        "pearson",
        "spearman",
        "correlation_status",
    ]
    metrics = pd.DataFrame(metric_rows, columns=metric_columns)
    if prediction_tables:
        predictions = (
            pd.concat(prediction_tables, ignore_index=True)
            .sort_values("_row_index", kind="stable")
            .drop(columns="_row_index")
            .reset_index(drop=True)
        )
    else:
        prediction_columns = [
            "tile_id",
            "dataset",
            "sample",
            "individual_id",
            "cell_type",
            "outer_fold",
            "observed_proportion",
            *[f"predicted_raw__{name}" for name in config.model_names],
        ]
        if config.calibrate:
            prediction_columns.extend(
                f"predicted_calibrated__{name}" for name in config.model_names
            )
        prediction_columns.append("predicted_raw__TrainingMeanBaseline")
        predictions = pd.DataFrame(columns=prediction_columns)

    status = pd.DataFrame(status_rows)
    status["n_completed_folds"] = int(status["status"].eq("complete").sum())
    status["n_skipped_folds"] = int(
        status["status"].eq("skipped_all_zero").sum()
    )
    status["n_expected_folds"] = len(folds)
    return metrics, predictions, status


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["mae", "rmse", "pearson", "spearman"]
    grouped = fold_metrics.groupby(["cell_type", "model_type", "prediction_scale"], observed=True)
    rows = []
    for keys, table in grouped:
        row = dict(zip(("cell_type", "model_type", "prediction_scale"), keys))
        row["n_outer_folds"] = table["held_out_individual"].nunique()
        for metric in metric_columns:
            values = pd.to_numeric(table[metric], errors="coerce")
            row[f"mean_{metric}"] = float(values.mean())
            row[f"std_{metric}"] = float(values.std())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_evaluation_metrics(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Return primary individual-unweighted and secondary pooled-tile summaries."""
    individual = summarize_fold_metrics(fold_metrics)
    individual.insert(3, "summary_scope", "individual_unweighted")
    pooled_rows: list[dict[str, Any]] = []
    truth = predictions["observed_proportion"].to_numpy(dtype=float)
    cell_type = str(predictions["cell_type"].iloc[0])
    for column in predictions.columns:
        match = re.fullmatch(r"predicted_(raw|calibrated)__(.+)", column)
        if match is None:
            continue
        scale, model_type = match.groups()
        pooled_rows.append(
            {
                "cell_type": cell_type,
                "model_type": model_type,
                "prediction_scale": scale,
                "summary_scope": "pooled_tiles",
                "n_outer_folds": predictions["individual_id"].nunique(),
                **{
                    f"mean_{name}": value
                    for name, value in regression_metrics(truth, predictions[column]).items()
                    if name in {"mae", "rmse", "pearson", "spearman"}
                },
                "std_mae": np.nan,
                "std_rmse": np.nan,
                "std_pearson": np.nan,
                "std_spearman": np.nan,
            }
        )
    return pd.concat([individual, pd.DataFrame(pooled_rows)], ignore_index=True, sort=False)


def _save_booster(booster: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
    booster.save_model(str(temporary))
    os.replace(temporary, destination)


def predict_saved_models(
    model_output_dir: str | Path,
    feature_matrices: Mapping[str, np.ndarray],
    *,
    extraction_recipe: str = "stpath_brca_training_compatible_v1",
    calibrate: bool = True,
    model_names: Sequence[str] = ALL_MODEL_NAMES,
    nthread: int | None = None,
) -> pd.DataFrame:
    """Predict with selected exported models after validating their feature contract."""
    if not model_names or len(set(model_names)) != len(model_names) or set(model_names).difference(SUPPORTED_MODEL_NAMES):
        raise ValueError("model_names must be a nonempty, unique selection of saved model types")
    if nthread is not None and nthread < 1:
        raise ValueError("nthread must be positive")
    destination = Path(model_output_dir)
    if not feature_matrices:
        raise ValueError("Foundation feature matrices are empty")
    first_matrix = np.asarray(next(iter(feature_matrices.values())))
    if first_matrix.ndim != 2:
        raise ValueError("Foundation feature matrices must be two-dimensional")
    widths = validate_feature_matrices(
        feature_matrices, first_matrix.shape[0], model_names=tuple(feature_matrices)
    )
    xgb = _load_xgboost()
    result: dict[str, np.ndarray] = {}

    for model_name in model_names:
        metadata_path = destination / "model_metadata" / f"{model_name}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("model_type") != model_name:
            raise ValueError(f"{metadata_path} has an unexpected model_type")
        if metadata.get("extraction_recipe") != extraction_recipe:
            raise ValueError(
                f"{metadata_path} requires extraction recipe {metadata.get('extraction_recipe')!r}, "
                f"not {extraction_recipe!r}"
            )
        recorded_widths = {
            str(name): int(width)
            for name, width in metadata.get("foundation_feature_dimensions", {}).items()
        }
        required_widths = {name: widths[name] for name in recorded_widths if name in widths}
        if recorded_widths != required_widths:
            raise ValueError(
                f"{metadata_path} foundation feature widths {recorded_widths} != {widths}"
            )

        if model_name in ENCODER_NAMES:
            model_features = np.asarray(feature_matrices[model_name])
        else:
            recorded_order = metadata.get("encoder_order")
            if recorded_order != list(MODEL_NAMES):
                raise ValueError(f"{metadata_path} has an unexpected encoder order")
            raw_selections = metadata.get("selected_feature_indices")
            if not isinstance(raw_selections, dict):
                raise ValueError(f"{metadata_path} is missing combined feature selections")
            selections = {
                name: np.asarray(raw_selections.get(name, []), dtype=int) for name in MODEL_NAMES
            }
            for name, indices in selections.items():
                if not len(indices) or (indices < 0).any() or (indices >= widths[name]).any():
                    raise ValueError(f"{metadata_path} has invalid {name} feature indices")
            model_features = concatenate_selected_features(
                feature_matrices,
                selections,
                np.arange(first_matrix.shape[0], dtype=int),
            )

        expected_input_dimension = int(metadata.get("input_dimension", -1))
        if model_features.shape[1] != expected_input_dimension:
            raise ValueError(
                f"{metadata_path} expects {expected_input_dimension} inputs, "
                f"but constructed {model_features.shape[1]}"
            )
        booster = xgb.Booster()
        model_path = destination / "models" / f"{model_name}.ubj"
        booster.load_model(str(model_path))
        if nthread is not None:
            booster.set_param({"nthread": nthread})
        if int(booster.num_features()) != expected_input_dimension:
            raise ValueError(f"{model_path} width does not match its metadata")
        raw_prediction = predict_booster(booster, model_features, **({"nthread": nthread} if nthread is not None else {}))
        result[f"predicted_raw__{model_name}"] = raw_prediction
        if calibrate:
            calibrator_path = destination / "calibration" / f"{model_name}.json"
            calibrator = json.loads(calibrator_path.read_text(encoding="utf-8"))
            if calibrator.get("cell_type") != metadata.get("cell_type"):
                raise ValueError(f"{calibrator_path} target does not match model metadata")
            result[f"predicted_calibrated__{model_name}"] = apply_quantile_calibrator(
                calibrator, raw_prediction
            )
    return pd.DataFrame(result)


def train_final_models(
    feature_matrices: Mapping[str, np.ndarray],
    tile_targets: pd.DataFrame,
    cell_type: str,
    evaluation: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
    config: TrainingConfig,
    *,
    extraction_recipe: str = "stpath_brca_training_compatible_v1",
    overwrite: bool = False,
) -> pd.DataFrame:
    """Fit and save configured full-cohort models and calibration tables."""
    destination = Path(output_dir)
    models_dir = destination / "models"
    metadata_dir = destination / "model_metadata"
    calibration_dir = destination / "calibration"
    target = tile_targets[f"proportion__{cell_type}"].to_numpy(dtype=float)
    all_rows = np.arange(len(tile_targets), dtype=int)
    selection_rows: list[dict[str, Any]] = []
    feature_widths = validate_feature_matrices(feature_matrices, len(tile_targets), model_names=config.encoder_names)
    checkpoint_signature = training_data_fingerprint(tile_targets, cell_type, config)
    final_checkpoint = destination / "checkpoints" / "final"
    _final_predictions, gains, selections = _fit_six_models(
        feature_matrices,
        target,
        all_rows,
        all_rows,
        config,
        context=f"{cell_type} final full-cohort model",
        checkpoint_dir=final_checkpoint,
        checkpoint_signature=checkpoint_signature,
        overwrite=overwrite,
    )
    for model_name in config.encoder_names:
        for rank, index in enumerate(selections[model_name], start=1):
            selection_rows.append(
                {
                    "cell_type": cell_type,
                    "selection_scope": "full_cohort",
                    "outer_fold": pd.NA,
                    "held_out_individual": "",
                    "encoder": model_name,
                    "rank": rank,
                    "feature_index": int(index),
                    "feature_name": f"f{int(index)}",
                    "gain": float(gains[model_name][index]),
                }
            )
    combined_features = (
        None if config.resnet50_only
        else concatenate_selected_features(feature_matrices, selections, all_rows)
    )

    predictions = evaluation["predictions"]
    checksum = tile_id_checksum(tile_targets["tile_id"].astype(str))
    try:
        xgb_version = _load_xgboost().__version__
    except AttributeError:
        xgb_version = "unknown"
    for model_name in config.model_names:
        model_config = (
            replace(config, runtime_nthread=config.effective_combined_nthread)
            if model_name == "Combined"
            else config
        )
        model_path = models_dir / f"{model_name}.ubj"
        _atomic_copy(final_checkpoint / "models" / f"{model_name}.ubj", model_path)
        if config.calibrate:
            calibrator = fit_quantile_calibrator(
                target,
                predictions[f"predicted_raw__{model_name}"].to_numpy(dtype=float),
                n_quantiles=config.n_quantiles,
            )
            calibrator.update(
                {
                    "cell_type": cell_type,
                    "model_type": model_name,
                    "source": "complete raw LOIO predictions",
                }
            )
            _atomic_json(calibrator, calibration_dir / f"{model_name}.json")
        metadata = {
            "cell_type": cell_type,
            "model_type": model_name,
            "input_dimension": (
                feature_widths[model_name]
                if model_name in feature_widths
                else int(combined_features.shape[1])
            ),
            "foundation_feature_dimensions": (
                {name: feature_widths[name] for name in MODEL_NAMES}
                if model_name != BASELINE_MODEL_NAME
                else {BASELINE_MODEL_NAME: feature_widths[BASELINE_MODEL_NAME]}
            ),
            "encoder_order": list(MODEL_NAMES) if model_name == "Combined" else [model_name],
            "selected_feature_indices": (
                {name: selections[name].tolist() for name in MODEL_NAMES}
                if model_name == "Combined"
                else None
            ),
            "xgboost_parameters": xgboost_parameters(model_config),
            "num_boost_round": config.num_boost_round,
            "training_config_fingerprint": configuration_fingerprint(config),
            "training_tile_id_checksum": checksum,
            "n_training_tiles": len(tile_targets),
            "individual_ids": sorted(tile_targets["individual_id"].astype(str).unique()),
            "extraction_recipe": extraction_recipe,
            "versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "xgboost": xgb_version,
            },
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(metadata, metadata_dir / f"{model_name}.json")
    return pd.DataFrame(selection_rows)


def run_training(
    feature_matrices: Mapping[str, np.ndarray],
    tile_targets: pd.DataFrame,
    target_eligibility: pd.DataFrame,
    output_root: str | Path,
    config: TrainingConfig,
    *,
    overwrite: bool = False,
    skip_all_zero_test_folds: bool = False,
    finalize_with_skipped_folds: bool = False,
    force_finalize: bool = False,
) -> pd.DataFrame:
    """Evaluate and fit eligible targets with restartable fold boundaries."""
    validate_cpu_budget(config)
    _require_columns(
        target_eligibility,
        ("cell_type", "eligible", "exclusion_reason"),
        "target eligibility",
    )
    target_table = target_eligibility.copy()
    target_table["cell_type"] = target_table["cell_type"].fillna("").astype(str).str.strip()
    target_table["eligible"] = _as_bool(target_table["eligible"], "target eligibility")
    target_table["exclusion_reason"] = (
        target_table["exclusion_reason"].fillna("").astype(str)
    )
    if target_table["cell_type"].eq("").any() or target_table["cell_type"].duplicated().any():
        raise ValueError("Target eligibility requires unique, nonempty cell_type values")
    missing_targets = [
        cell_type
        for cell_type in target_table.loc[target_table["eligible"], "cell_type"]
        if f"proportion__{cell_type}" not in tile_targets
    ]
    if missing_targets:
        raise ValueError(f"Tile targets are missing eligible cell types: {missing_targets}")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    validate_feature_matrices(feature_matrices, len(tile_targets), model_names=config.encoder_names)
    safe_targets = target_table["cell_type"].map(safe_name)
    if safe_targets.duplicated().any():
        collisions = target_table.loc[safe_targets.duplicated(False), "cell_type"].tolist()
        raise ValueError(f"Cell-type names collide after path sanitization: {collisions}")
    run_checksum = tile_id_checksum(tile_targets["tile_id"].astype(str))
    run_fingerprint = configuration_fingerprint(config)
    _atomic_json(
        {
            "training_config": asdict(config),
            "training_config_fingerprint": run_fingerprint,
            "tile_id_checksum": run_checksum,
            "n_tiles": len(tile_targets),
            "model_types": list(config.model_names),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        root / "run_metadata.json",
    )
    summaries: list[dict[str, Any]] = []
    for row in target_table.itertuples(index=False):
        cell_type = str(row.cell_type)
        if not row.eligible:
            summaries.append({
                "cell_type": cell_type,
                "status": "ineligible",
                "error": row.exclusion_reason,
                "n_skipped_folds": 0,
            })
            continue
        target_dir = root / safe_name(cell_type)
        target_fingerprint = training_data_fingerprint(tile_targets, cell_type, config)
        completion_path = target_dir / "complete.json"
        if completion_path.is_file() and not overwrite and not force_finalize:
            try:
                completed = json.loads(completion_path.read_text(encoding="utf-8"))
                required_files = [
                    target_dir / "oof_predictions.csv.gz",
                    target_dir / "fold_metrics.csv",
                    target_dir / "metric_summary.csv",
                    target_dir / "selected_features.csv",
                    *[target_dir / "models" / f"{name}.ubj" for name in config.model_names],
                    *[target_dir / "model_metadata" / f"{name}.json" for name in config.model_names],
                ]
                if config.calibrate:
                    required_files.extend(
                        target_dir / "calibration" / f"{name}.json"
                        for name in config.model_names
                    )
                valid_completion = (
                    completed.get("training_config_fingerprint") == run_fingerprint
                    and completed.get("tile_id_checksum") == run_checksum
                    and completed.get("training_data_fingerprint") == target_fingerprint
                    and completed.get("n_tiles") == len(tile_targets)
                    and all(path.is_file() and path.stat().st_size > 0 for path in required_files)
                )
                if valid_completion:
                    summaries.append({
                        "cell_type": cell_type,
                        "status": "skipped_existing",
                        "error": "",
                        "n_skipped_folds": int(
                            completed.get("n_skipped_folds", 0)
                        ),
                    })
                    continue
            except Exception:
                pass
        try:
            evaluation = evaluate_cell_type(
                feature_matrices,
                tile_targets,
                cell_type,
                config,
                checkpoint_dir=target_dir / "checkpoints",
                overwrite=overwrite,
                skip_all_zero_test_folds=skip_all_zero_test_folds,
            )
            _atomic_csv(evaluation["predictions"], target_dir / "oof_predictions.csv.gz")
            _atomic_csv(evaluation["fold_metrics"], target_dir / "fold_metrics.csv")
            metric_summary = (
                summarize_evaluation_metrics(
                    evaluation["predictions"], evaluation["fold_metrics"]
                )
                if not evaluation["predictions"].empty
                and not evaluation["fold_metrics"].empty
                else pd.DataFrame()
            )
            _atomic_csv(metric_summary, target_dir / "metric_summary.csv")
            n_skipped_folds = int(
                evaluation["fold_status"]["status"].eq("skipped_all_zero").sum()
            )
            if n_skipped_folds and not finalize_with_skipped_folds:
                _atomic_csv(
                    evaluation["selected_features"],
                    target_dir / "selected_features.csv",
                )
                summaries.append({
                    "cell_type": cell_type,
                    "status": "partial_skipped_folds",
                    "error": "",
                    "n_skipped_folds": n_skipped_folds,
                })
                continue
            final_selections = train_final_models(
                feature_matrices,
                tile_targets,
                cell_type,
                evaluation,
                target_dir,
                config,
                overwrite=overwrite,
            )
            selections = pd.concat(
                [evaluation["selected_features"], final_selections], ignore_index=True, sort=False
            )
            _atomic_csv(selections, target_dir / "selected_features.csv")
            _atomic_json(
                {
                    "cell_type": cell_type,
                    "status": "complete",
                    "training_config_fingerprint": run_fingerprint,
                    "tile_id_checksum": run_checksum,
                    "training_data_fingerprint": target_fingerprint,
                    "n_tiles": len(tile_targets),
                    "n_skipped_folds": n_skipped_folds,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                completion_path,
            )
            summaries.append({
                "cell_type": cell_type,
                "status": (
                    "success_with_skipped_folds"
                    if n_skipped_folds
                    else "success"
                ),
                "error": "",
                "n_skipped_folds": n_skipped_folds,
            })
        except Exception as exc:
            error_text = traceback.format_exc()
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "failure.txt").write_text(error_text, encoding="utf-8")
            summaries.append({
                "cell_type": cell_type,
                "status": "failed",
                "error": str(exc),
                "n_skipped_folds": 0,
            })
    summary = pd.DataFrame(summaries)
    _atomic_csv(summary, root / "run_summary.csv")
    return summary
