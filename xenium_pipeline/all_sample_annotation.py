"""Operational cross-dataset marker-score cell annotation.

This module packages the workflow from ``annotate_all_sample_cells.ipynb`` so
it can be run from scripts, notebooks, and command-line jobs without copying
notebook cells.  It consumes the gene-panel review tables under
``gene_panel_review/`` and writes marker-score labels under
``cell_annotation/``.
"""

from __future__ import annotations

import argparse
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import expression_annotation
from .gene_panel_utils import natural_key


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORE_DISTRIBUTION_QUANTILES = (
    0.00,
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    1.00,
)


@dataclass(frozen=True)
class AllSampleAnnotationConfig:
    """Configuration for cross-dataset marker-score annotation."""

    normalization_target: float = 10_000.0
    min_score: float = 0.25
    min_margin: float = 0.10
    min_detected_markers_in_cell: int = 2
    uncertain_label: str = "Uncertain"
    use_direct_author_marker_groups: bool = True
    use_panel_implied_when_direct_missing: bool = True
    include_panel_implied_extra_groups: bool = False
    allow_single_marker_groups: bool = False
    write_full_score_matrix: bool = False
    max_score_plot_cells_per_sample: int | None = 5_000
    max_group_score_plot_cells_per_sample: int | None = 2_000
    score_distribution_quantiles: tuple[float, ...] = DEFAULT_SCORE_DISTRIBUTION_QUANTILES
    write_outputs: bool = True
    overwrite: bool = False
    skip_completed: bool = False


def safe_name(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_") or "sample"


def path_for_record(path: Path, root: str | Path = PROJECT_ROOT) -> str:
    root = Path(root).expanduser().resolve()
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)


def resolve_existing_path(path_text: object, root: str | Path = PROJECT_ROOT) -> Path | None:
    if path_text is None or pd.isna(path_text):
        return None
    text = str(path_text).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path(root).expanduser().resolve() / path
    return path if path.exists() else None


def parse_path_list(value: object, root: str | Path = PROJECT_ROOT) -> list[Path]:
    if value is None or pd.isna(value):
        return []
    paths: list[Path] = []
    for item in str(value).split(";"):
        path = resolve_existing_path(item.strip(), root)
        if path is not None and path not in paths:
            paths.append(path)
    return paths


def read_required_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing required review table: {path}")
    return pd.read_csv(path)


def split_gene_list(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return sorted({part.strip().upper() for part in re.split(r"[;,]", text) if part.strip()})


def marker_row_is_usable(
    n_markers: object,
    status: object | None = None,
    *,
    allow_single_marker_groups: bool,
) -> bool:
    try:
        n = int(float(n_markers))
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        return False
    if not allow_single_marker_groups and n < 2:
        return False
    if status is not None and str(status) == "no_markers_in_assay":
        return False
    return True


def load_panel_review_inputs(
    review_dir: str | Path,
    *,
    dataset_names: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    review_dir = Path(review_dir).expanduser().resolve()
    sample_panel_summary = read_required_csv(review_dir / "sample_panel_summary.csv")
    marker_group_coverage = read_required_csv(review_dir / "marker_group_coverage.csv")
    panel_implied_marker_group_coverage = read_required_csv(
        review_dir / "panel_implied_marker_group_coverage.csv"
    )
    if dataset_names is not None:
        keep = set(map(str, dataset_names))
        sample_panel_summary = sample_panel_summary.loc[
            sample_panel_summary["dataset"].astype(str).isin(keep)
        ].copy()
        marker_group_coverage = marker_group_coverage.loc[
            marker_group_coverage["dataset"].astype(str).isin(keep)
        ].copy()
        panel_implied_marker_group_coverage = panel_implied_marker_group_coverage.loc[
            panel_implied_marker_group_coverage["dataset"].astype(str).isin(keep)
        ].copy()
    return sample_panel_summary, marker_group_coverage, panel_implied_marker_group_coverage


def select_samples_to_annotate(
    sample_panel_summary: pd.DataFrame,
    *,
    sample_names: Iterable[str] | None = None,
    sample_limit: int | None = None,
) -> pd.DataFrame:
    samples = sample_panel_summary.loc[
        sample_panel_summary["status"].astype(str).eq("ok")
    ].copy()
    if sample_names is not None:
        keep = set(map(str, sample_names))
        samples = samples.loc[samples["sample"].astype(str).isin(keep)].copy()
    samples = samples.sort_values(
        ["dataset", "sample"],
        key=lambda column: column.map(natural_key),
        kind="stable",
    ).reset_index(drop=True)
    if sample_limit is not None:
        samples = samples.head(int(sample_limit)).copy()
    return samples


def direct_marker_sets_for_sample(
    marker_group_coverage: pd.DataFrame,
    dataset: str,
    sample: str,
    config: AllSampleAnnotationConfig,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    rows = marker_group_coverage.loc[
        marker_group_coverage["dataset"].astype(str).eq(dataset)
        & marker_group_coverage["sample"].astype(str).eq(sample)
    ].copy()
    if rows.empty or not config.use_direct_author_marker_groups:
        return {}, rows
    marker_sets: dict[str, list[str]] = {}
    rows["n_markers_in_assay"] = (
        pd.to_numeric(rows["n_markers_in_assay"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    for row in rows.sort_values(["group"]).itertuples(index=False):
        status = getattr(row, "marker_group_review_status", None)
        if not marker_row_is_usable(
            row.n_markers_in_assay,
            status,
            allow_single_marker_groups=config.allow_single_marker_groups,
        ):
            continue
        genes = split_gene_list(row.markers_found)
        if genes:
            marker_sets[str(row.group)] = genes
    return marker_sets, rows


def panel_implied_marker_sets_for_sample(
    panel_implied_marker_group_coverage: pd.DataFrame,
    dataset: str,
    sample: str,
    config: AllSampleAnnotationConfig,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    rows = panel_implied_marker_group_coverage.loc[
        panel_implied_marker_group_coverage["dataset"].astype(str).eq(dataset)
        & panel_implied_marker_group_coverage["sample"].astype(str).eq(sample)
    ].copy()
    if rows.empty:
        return {}, rows
    marker_sets: dict[str, list[str]] = {}
    rows["n_matching_marker_genes"] = (
        pd.to_numeric(rows["n_matching_marker_genes"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    for row in rows.sort_values(["group"]).itertuples(index=False):
        status = getattr(row, "panel_implied_marker_coverage_status", None)
        if not marker_row_is_usable(
            row.n_matching_marker_genes,
            status,
            allow_single_marker_groups=config.allow_single_marker_groups,
        ):
            continue
        genes = split_gene_list(row.matching_marker_genes)
        if genes:
            marker_sets[str(row.group)] = genes
    return marker_sets, rows


def build_marker_sets_for_sample(
    marker_group_coverage: pd.DataFrame,
    panel_implied_marker_group_coverage: pd.DataFrame,
    dataset: str,
    sample: str,
    config: AllSampleAnnotationConfig,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    direct_sets, _direct_rows = direct_marker_sets_for_sample(
        marker_group_coverage,
        dataset,
        sample,
        config,
    )
    implied_sets, _implied_rows = panel_implied_marker_sets_for_sample(
        panel_implied_marker_group_coverage,
        dataset,
        sample,
        config,
    )
    marker_sets: dict[str, list[str]] = {}
    records: list[dict[str, object]] = []

    if direct_sets:
        marker_sets.update(direct_sets)
        for group, genes in direct_sets.items():
            records.append(
                {
                    "dataset": dataset,
                    "sample": sample,
                    "group": group,
                    "marker_source_priority": "direct_author_marker_group",
                    "n_marker_genes_used": len(genes),
                    "marker_genes_used": "; ".join(genes),
                }
            )

    should_use_implied = (
        (not direct_sets and config.use_panel_implied_when_direct_missing)
        or config.include_panel_implied_extra_groups
    )
    if should_use_implied:
        for group, genes in implied_sets.items():
            if group in marker_sets:
                continue
            marker_sets[group] = genes
            records.append(
                {
                    "dataset": dataset,
                    "sample": sample,
                    "group": group,
                    "marker_source_priority": "panel_implied_marker_group",
                    "n_marker_genes_used": len(genes),
                    "marker_genes_used": "; ".join(genes),
                }
            )
    return marker_sets, pd.DataFrame(records)


def build_marker_dictionary_tables(
    samples_to_annotate: pd.DataFrame,
    marker_group_coverage: pd.DataFrame,
    panel_implied_marker_group_coverage: pd.DataFrame,
    config: AllSampleAnnotationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    marker_source_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for row in samples_to_annotate.itertuples(index=False):
        marker_sets, source_table = build_marker_sets_for_sample(
            marker_group_coverage,
            panel_implied_marker_group_coverage,
            str(row.dataset),
            str(row.sample),
            config,
        )
        if not source_table.empty:
            marker_source_tables.append(source_table)
        source_counts = (
            source_table["marker_source_priority"].value_counts().to_dict()
            if not source_table.empty
            else {}
        )
        summary_rows.append(
            {
                "dataset": row.dataset,
                "sample": row.sample,
                "n_marker_groups_used": len(marker_sets),
                "n_direct_author_marker_groups_used": source_counts.get(
                    "direct_author_marker_group",
                    0,
                ),
                "n_panel_implied_marker_groups_used": source_counts.get(
                    "panel_implied_marker_group",
                    0,
                ),
                "min_marker_genes_per_group": min(
                    (len(genes) for genes in marker_sets.values()),
                    default=0,
                ),
                "max_marker_genes_per_group": max(
                    (len(genes) for genes in marker_sets.values()),
                    default=0,
                ),
            }
        )

    marker_dictionary_summary = pd.DataFrame(
        summary_rows,
        columns=[
            "dataset",
            "sample",
            "n_marker_groups_used",
            "n_direct_author_marker_groups_used",
            "n_panel_implied_marker_groups_used",
            "min_marker_genes_per_group",
            "max_marker_genes_per_group",
        ],
    )
    marker_source_table = (
        pd.concat(marker_source_tables, ignore_index=True)
        if marker_source_tables
        else pd.DataFrame()
    )
    summary_cols = [
        column
        for column in [
            "dataset",
            "sample",
            "sample_dir",
            "panel_resolution",
            "marker_group_resolution",
            "n_assay_genes",
            "n_marker_group_files",
            "n_marker_groups",
            "n_feature_matrix_files",
        ]
        if column in samples_to_annotate.columns
    ]
    annotation_input_readiness = samples_to_annotate[summary_cols].copy()
    annotation_input_readiness = annotation_input_readiness.merge(
        marker_dictionary_summary,
        on=["dataset", "sample"],
        how="left",
    )
    count_columns = [
        "n_feature_matrix_files",
        "n_marker_groups_used",
        "n_direct_author_marker_groups_used",
        "n_panel_implied_marker_groups_used",
        "min_marker_genes_per_group",
        "max_marker_genes_per_group",
    ]
    for column in count_columns:
        if column in annotation_input_readiness.columns:
            annotation_input_readiness[column] = (
                pd.to_numeric(annotation_input_readiness[column], errors="coerce")
                .fillna(0)
                .astype(int)
            )
    annotation_input_readiness["marker_dictionary_source"] = np.select(
        [
            annotation_input_readiness["n_direct_author_marker_groups_used"].gt(0)
            & annotation_input_readiness["n_panel_implied_marker_groups_used"].gt(0),
            annotation_input_readiness["n_direct_author_marker_groups_used"].gt(0),
            annotation_input_readiness["n_panel_implied_marker_groups_used"].gt(0),
        ],
        [
            "direct_author_plus_panel_implied",
            "direct_author_marker_groups",
            "panel_implied_marker_groups",
        ],
        default="none",
    )
    annotation_input_readiness["annotation_input_status"] = np.select(
        [
            annotation_input_readiness.get("n_feature_matrix_files", pd.Series(1, index=annotation_input_readiness.index)).eq(0),
            annotation_input_readiness["n_marker_groups_used"].eq(0),
        ],
        ["missing_feature_matrix", "missing_marker_dictionary"],
        default="ready_to_score",
    )

    if marker_source_table.empty:
        marker_group_inventory = pd.DataFrame(
            columns=[
                "dataset",
                "group",
                "marker_source_priority",
                "n_samples_using_group",
                "min_marker_genes_used",
                "median_marker_genes_used",
                "max_marker_genes_used",
                "example_marker_genes_used",
            ]
        )
    else:
        marker_group_inventory = (
            marker_source_table.groupby(
                ["dataset", "group", "marker_source_priority"],
                dropna=False,
            )
            .agg(
                n_samples_using_group=("sample", "nunique"),
                min_marker_genes_used=("n_marker_genes_used", "min"),
                median_marker_genes_used=("n_marker_genes_used", "median"),
                max_marker_genes_used=("n_marker_genes_used", "max"),
                example_marker_genes_used=("marker_genes_used", _compact_gene_examples),
            )
            .reset_index()
            .sort_values(["dataset", "marker_source_priority", "group"], kind="stable")
        )
    return (
        annotation_input_readiness,
        marker_dictionary_summary,
        marker_source_table,
        marker_group_inventory,
    )


def _compact_gene_examples(values: pd.Series, max_genes: int = 30) -> str:
    genes: list[str] = []
    for value in values.dropna().astype(str):
        genes.extend(split_gene_list(value))
    unique_genes = sorted(set(genes))
    suffix = "" if len(unique_genes) <= max_genes else f"; ... (+{len(unique_genes) - max_genes} more)"
    return "; ".join(unique_genes[:max_genes]) + suffix


def select_feature_matrix(row: pd.Series, root: str | Path = PROJECT_ROOT) -> Path | None:
    candidates = parse_path_list(row.get("feature_matrix_files", ""), root)
    if not candidates:
        sample_dir = resolve_existing_path(row.get("sample_dir", ""), root)
        if sample_dir is not None:
            candidates = sorted(sample_dir.rglob("*cell_feature_matrix.h5"))
            candidates.extend(sorted(sample_dir.rglob("cell_feature_matrix.zarr.zip")))
    if not candidates:
        return None
    h5_candidates = [path for path in candidates if path.name.endswith(".h5")]
    if h5_candidates:
        return h5_candidates[0]
    zarr_candidates = [path for path in candidates if path.name.endswith(".zarr.zip")]
    if zarr_candidates:
        return zarr_candidates[0]
    return candidates[0]


def ensure_can_write(paths: list[Path], *, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "Output file(s) already exist and overwrite is False: " + formatted
        )


def call_status_from_assignment(row: pd.Series, uncertain_label: str) -> str:
    if row["group"] != uncertain_label:
        return "assigned"
    if not bool(row.get("passed_score", False)):
        return "uncertain_low_score"
    if not bool(row.get("passed_margin", False)):
        return "uncertain_low_margin"
    if not bool(row.get("passed_marker_count", False)):
        return "uncertain_too_few_detected_markers"
    return "uncertain_other"


def score_and_assign_sample(
    dataset: str,
    sample: str,
    matrix_path: Path,
    marker_sets: dict[str, list[str]],
    config: AllSampleAnnotationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if matrix_path.name.endswith(".zarr.zip"):
        cell_ids = expression_annotation.cell_ids_from_cells_zarr(matrix_path.parent)
        if cell_ids is None:
            cells = expression_annotation.read_optional_cells_csv(
                expression_annotation.find_cells_csv(matrix_path.parent)
            )
            if cells is None:
                raise FileNotFoundError(
                    "Zarr matrix requires cells.zarr.zip or a cell CSV with cell_id in matrix order."
                )
            cell_ids = cells["cell_id"].astype(str).tolist()
        scores, detected, coverage = expression_annotation.score_zarr_marker_sets(
            matrix_path,
            cell_ids,
            marker_sets,
            config.normalization_target,
        )
    elif matrix_path.name.endswith(".h5"):
        scores, detected, coverage = expression_annotation.score_h5_marker_sets(
            matrix_path,
            marker_sets,
            config.normalization_target,
        )
    else:
        raise ValueError(f"Unsupported feature-matrix file: {matrix_path}")

    assignments = expression_annotation.assign_cells_from_marker_scores(
        scores,
        detected,
        min_score=config.min_score,
        min_margin=config.min_margin,
        min_markers=config.min_detected_markers_in_cell,
        uncertain_label=config.uncertain_label,
    )
    assignments.insert(0, "dataset", dataset)
    assignments.insert(1, "sample", sample)
    assignments["call_status"] = assignments.apply(
        lambda row: call_status_from_assignment(row, config.uncertain_label),
        axis=1,
    )

    group_summary = expression_annotation.summarize_annotations(assignments)
    group_summary.insert(0, "dataset", dataset)
    group_summary.insert(1, "sample", sample)

    marker_score_summary = coverage.copy()
    marker_score_summary.insert(0, "dataset", dataset)
    marker_score_summary.insert(1, "sample", sample)
    marker_score_summary["n_cells_scored"] = len(assignments)
    marker_score_summary["n_cells_with_any_marker_detected"] = [
        int((detected[group] > 0).sum()) for group in marker_score_summary["group"]
    ]
    marker_score_summary["fraction_cells_with_any_marker_detected"] = (
        marker_score_summary["n_cells_with_any_marker_detected"] / len(assignments)
        if len(assignments)
        else np.nan
    )
    marker_score_summary["median_score"] = [
        float(scores[group].median()) for group in marker_score_summary["group"]
    ]
    marker_score_summary["max_score"] = [
        float(scores[group].max()) for group in marker_score_summary["group"]
    ]
    marker_score_summary["n_cells_top_marker_group"] = (
        assignments["top_marker_group"]
        .value_counts()
        .reindex(marker_score_summary["group"])
        .fillna(0)
        .astype(int)
        .to_numpy()
    )
    marker_score_summary["n_cells_assigned_to_group"] = (
        assignments.loc[assignments["group"] != config.uncertain_label, "group"]
        .value_counts()
        .reindex(marker_score_summary["group"])
        .fillna(0)
        .astype(int)
        .to_numpy()
    )
    marker_score_summary["fraction_cells_assigned_to_group"] = (
        marker_score_summary["n_cells_assigned_to_group"] / len(assignments)
        if len(assignments)
        else np.nan
    )

    full_scores = scores.add_prefix("score__").copy()
    full_scores.insert(0, "cell_id", scores.index.astype(str))
    full_scores.insert(0, "sample", sample)
    full_scores.insert(0, "dataset", dataset)
    return assignments, group_summary, marker_score_summary, full_scores


def quantile_column_name(quantile: float) -> str:
    return f"q{int(round(quantile * 100)):02d}"


def numeric_distribution_row(
    *,
    dataset: str,
    sample: str,
    metric: str,
    values: pd.Series,
    config: AllSampleAnnotationConfig,
    level: str,
    stratum: str = "all_cells",
    group: str = "",
) -> dict[str, object]:
    series = pd.to_numeric(values, errors="coerce").dropna()
    row: dict[str, object] = {
        "dataset": dataset,
        "sample": sample,
        "level": level,
        "metric": metric,
        "stratum": stratum,
        "group": group,
        "n_values": int(series.shape[0]),
        "mean": float(series.mean()) if not series.empty else np.nan,
        "sd": float(series.std(ddof=1)) if series.shape[0] > 1 else np.nan,
        "min": float(series.min()) if not series.empty else np.nan,
        "max": float(series.max()) if not series.empty else np.nan,
    }
    if series.empty:
        for quantile in config.score_distribution_quantiles:
            row[quantile_column_name(quantile)] = np.nan
        return row
    quantiles = series.quantile(config.score_distribution_quantiles)
    for quantile, value in quantiles.items():
        row[quantile_column_name(float(quantile))] = float(value)
    return row


def summarize_assignment_score_distributions(
    dataset: str,
    sample: str,
    assignments: pd.DataFrame,
    config: AllSampleAnnotationConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric in ["top_score", "runner_up_score", "score_margin", "n_top_markers_detected"]:
        rows.append(
            numeric_distribution_row(
                dataset=dataset,
                sample=sample,
                metric=metric,
                values=assignments[metric],
                config=config,
                level="cell_assignment",
            )
        )
        for call_status, table in assignments.groupby("call_status", dropna=False):
            rows.append(
                numeric_distribution_row(
                    dataset=dataset,
                    sample=sample,
                    metric=metric,
                    values=table[metric],
                    config=config,
                    level="cell_assignment",
                    stratum=f"call_status:{call_status}",
                )
            )
    return rows


def summarize_marker_group_score_distributions(
    dataset: str,
    sample: str,
    score_matrix: pd.DataFrame,
    config: AllSampleAnnotationConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column in score_matrix.columns:
        if not column.startswith("score__"):
            continue
        group = column.removeprefix("score__")
        rows.append(
            numeric_distribution_row(
                dataset=dataset,
                sample=sample,
                metric="marker_score",
                values=score_matrix[column],
                config=config,
                level="marker_group_score",
                group=group,
            )
        )
    return rows


def sample_scores_for_plot(
    assignments: pd.DataFrame,
    config: AllSampleAnnotationConfig,
) -> pd.DataFrame:
    columns = [
        "dataset",
        "sample",
        "cell_id",
        "group",
        "top_marker_group",
        "call_status",
        "top_score",
        "runner_up_score",
        "score_margin",
        "n_top_markers_detected",
    ]
    table = assignments[columns].copy()
    max_cells = config.max_score_plot_cells_per_sample
    if max_cells is not None and len(table) > int(max_cells):
        table = table.sample(n=int(max_cells), random_state=1729)
    table["plot_sample_weight"] = len(assignments) / len(table) if len(table) else np.nan
    return table


def sample_all_group_scores_for_plot(
    score_matrix: pd.DataFrame,
    config: AllSampleAnnotationConfig,
) -> pd.DataFrame:
    score_columns = [column for column in score_matrix.columns if column.startswith("score__")]
    if not score_columns:
        return pd.DataFrame(
            columns=["dataset", "sample", "cell_id", "group", "marker_score", "plot_sample_weight"]
        )
    table = score_matrix[["dataset", "sample", "cell_id", *score_columns]].copy()
    original_n_cells = len(table)
    max_cells = config.max_group_score_plot_cells_per_sample
    if max_cells is not None and len(table) > int(max_cells):
        table = table.sample(n=int(max_cells), random_state=1729)
    long_table = table.melt(
        id_vars=["dataset", "sample", "cell_id"],
        value_vars=score_columns,
        var_name="group",
        value_name="marker_score",
    )
    long_table["group"] = long_table["group"].str.removeprefix("score__")
    long_table["marker_score"] = pd.to_numeric(long_table["marker_score"], errors="coerce")
    long_table["plot_sample_weight"] = original_n_cells / len(table) if len(table) else np.nan
    return long_table


def _load_existing_sample_outputs(
    sample_output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(sample_output_dir / "cell_marker_labels.csv.gz")
    group_summary = pd.read_csv(sample_output_dir / "cell_annotation_group_summary.csv")
    marker_score_summary = pd.read_csv(sample_output_dir / "marker_score_summary.csv")
    return labels, group_summary, marker_score_summary


def _write_annotation_checkpoint(output_dir: Path, rows: list[dict[str, object]], write_outputs: bool) -> None:
    if write_outputs and rows:
        pd.DataFrame(rows).to_csv(output_dir / "annotation_run_summary.csv", index=False)


def run_all_sample_annotation(
    root: str | Path = PROJECT_ROOT,
    *,
    review_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    dataset_names: Iterable[str] | None = None,
    sample_names: Iterable[str] | None = None,
    sample_limit: int | None = None,
    config: AllSampleAnnotationConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Run marker-score annotation across selected samples."""
    root = Path(root).expanduser().resolve()
    review_dir = Path(review_dir or root / "gene_panel_review").expanduser().resolve()
    output_dir = Path(output_dir or root / "cell_annotation").expanduser().resolve()
    config = config or AllSampleAnnotationConfig()
    if config.write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)

    (
        sample_panel_summary,
        marker_group_coverage,
        panel_implied_marker_group_coverage,
    ) = load_panel_review_inputs(review_dir, dataset_names=dataset_names)
    samples_to_annotate = select_samples_to_annotate(
        sample_panel_summary,
        sample_names=sample_names,
        sample_limit=sample_limit,
    )
    (
        annotation_input_readiness,
        marker_dictionary_summary,
        marker_source_table,
        marker_group_inventory,
    ) = build_marker_dictionary_tables(
        samples_to_annotate,
        marker_group_coverage,
        panel_implied_marker_group_coverage,
        config,
    )
    if config.write_outputs:
        samples_to_annotate.to_csv(output_dir / "sample_input_table.csv", index=False)
        annotation_input_readiness.to_csv(output_dir / "annotation_input_readiness.csv", index=False)
        marker_dictionary_summary.to_csv(output_dir / "marker_dictionary_summary.csv", index=False)
        marker_source_table.to_csv(output_dir / "marker_dictionary_details.csv", index=False)
        marker_group_inventory.to_csv(output_dir / "marker_group_inventory.csv", index=False)

    run_rows: list[dict[str, object]] = []
    all_group_summaries: list[pd.DataFrame] = []
    all_marker_score_summaries: list[pd.DataFrame] = []
    all_cell_score_distribution_rows: list[dict[str, object]] = []
    all_marker_group_score_distribution_rows: list[dict[str, object]] = []
    all_score_plot_samples: list[pd.DataFrame] = []
    all_marker_group_score_plot_samples: list[pd.DataFrame] = []

    for row in samples_to_annotate.itertuples(index=False):
        dataset = str(row.dataset)
        sample = str(row.sample)
        print(f"Annotating {dataset} / {sample} ...", flush=True)
        sample_output_dir = output_dir / safe_name(dataset) / safe_name(sample)
        if config.write_outputs:
            sample_output_dir.mkdir(parents=True, exist_ok=True)

        marker_sets, source_table = build_marker_sets_for_sample(
            marker_group_coverage,
            panel_implied_marker_group_coverage,
            dataset,
            sample,
            config,
        )
        matrix_path = select_feature_matrix(pd.Series(row._asdict()), root)
        labels_path = sample_output_dir / "cell_marker_labels.csv.gz"
        group_summary_path = sample_output_dir / "cell_annotation_group_summary.csv"
        marker_summary_path = sample_output_dir / "marker_score_summary.csv"
        dictionary_path = sample_output_dir / "marker_dictionary_used.csv"
        score_matrix_path = sample_output_dir / "cell_marker_score_matrix.csv.gz"

        row_record: dict[str, object] = {
            "dataset": dataset,
            "sample": sample,
            "status": "not_run",
            "matrix_path": path_for_record(matrix_path, root) if matrix_path else "",
            "n_marker_groups_used": len(marker_sets),
            "n_cells": 0,
            "n_assigned_cells": 0,
            "assigned_fraction": np.nan,
            "n_uncertain_cells": 0,
            "uncertain_fraction": np.nan,
            "cell_marker_labels_path": "",
            "cell_annotation_group_summary_path": "",
            "marker_score_summary_path": "",
            "marker_dictionary_path": "",
            "score_matrix_path": "",
            "error": "",
        }
        try:
            if config.skip_completed and labels_path.is_file() and group_summary_path.is_file() and marker_summary_path.is_file():
                labels, group_summary, marker_score_summary = _load_existing_sample_outputs(sample_output_dir)
                all_group_summaries.append(group_summary)
                all_marker_score_summaries.append(marker_score_summary)
                n_cells = len(labels)
                n_assigned = int(labels["group"].ne(config.uncertain_label).sum()) if "group" in labels else 0
                row_record.update(
                    {
                        "status": "skipped_existing",
                        "n_cells": n_cells,
                        "n_assigned_cells": n_assigned,
                        "assigned_fraction": n_assigned / n_cells if n_cells else np.nan,
                        "n_uncertain_cells": n_cells - n_assigned,
                        "uncertain_fraction": (n_cells - n_assigned) / n_cells if n_cells else np.nan,
                        "cell_marker_labels_path": path_for_record(labels_path, root),
                        "cell_annotation_group_summary_path": path_for_record(group_summary_path, root),
                        "marker_score_summary_path": path_for_record(marker_summary_path, root),
                        "marker_dictionary_path": path_for_record(dictionary_path, root) if dictionary_path.is_file() else "",
                        "score_matrix_path": path_for_record(score_matrix_path, root) if score_matrix_path.is_file() else "",
                    }
                )
                print("  skipped existing outputs", flush=True)
                run_rows.append(row_record)
                _write_annotation_checkpoint(output_dir, run_rows, config.write_outputs)
                continue

            if not marker_sets:
                raise ValueError("No usable marker groups after marker-coverage filtering.")
            if matrix_path is None:
                raise FileNotFoundError("No existing cell_feature_matrix.h5 or cell_feature_matrix.zarr.zip found.")

            assignments, group_summary, marker_score_summary, score_matrix = score_and_assign_sample(
                dataset,
                sample,
                matrix_path,
                marker_sets,
                config,
            )
            output_paths = [labels_path, group_summary_path, marker_summary_path, dictionary_path]
            if config.write_full_score_matrix:
                output_paths.append(score_matrix_path)
            if config.write_outputs:
                ensure_can_write(output_paths, overwrite=config.overwrite)
                assignments.to_csv(labels_path, index=False, compression="gzip")
                group_summary.to_csv(group_summary_path, index=False)
                marker_score_summary.to_csv(marker_summary_path, index=False)
                source_table.to_csv(dictionary_path, index=False)
                if config.write_full_score_matrix:
                    score_matrix.to_csv(score_matrix_path, index=False, compression="gzip")

            all_group_summaries.append(group_summary)
            all_marker_score_summaries.append(marker_score_summary)
            all_cell_score_distribution_rows.extend(
                summarize_assignment_score_distributions(dataset, sample, assignments, config)
            )
            all_marker_group_score_distribution_rows.extend(
                summarize_marker_group_score_distributions(dataset, sample, score_matrix, config)
            )
            all_score_plot_samples.append(sample_scores_for_plot(assignments, config))
            all_marker_group_score_plot_samples.append(sample_all_group_scores_for_plot(score_matrix, config))

            n_cells = len(assignments)
            n_assigned = int(assignments["group"].ne(config.uncertain_label).sum())
            n_uncertain = n_cells - n_assigned
            row_record.update(
                {
                    "status": "success",
                    "n_cells": n_cells,
                    "n_assigned_cells": n_assigned,
                    "assigned_fraction": n_assigned / n_cells if n_cells else np.nan,
                    "n_uncertain_cells": n_uncertain,
                    "uncertain_fraction": n_uncertain / n_cells if n_cells else np.nan,
                    "cell_marker_labels_path": path_for_record(labels_path, root),
                    "cell_annotation_group_summary_path": path_for_record(group_summary_path, root),
                    "marker_score_summary_path": path_for_record(marker_summary_path, root),
                    "marker_dictionary_path": path_for_record(dictionary_path, root),
                    "score_matrix_path": path_for_record(score_matrix_path, root) if config.write_full_score_matrix else "",
                }
            )
            print(f"  success: {n_assigned:,} / {n_cells:,} cells assigned", flush=True)
        except Exception as error:
            row_record.update(
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            if config.write_outputs:
                (sample_output_dir / "annotation_error.txt").write_text(
                    traceback.format_exc(),
                    encoding="utf-8",
                )
            print(f"  failed: {row_record['error']}", flush=True)

        run_rows.append(row_record)
        _write_annotation_checkpoint(output_dir, run_rows, config.write_outputs)

    annotation_run_summary = pd.DataFrame(run_rows)
    combined_group_summary = (
        pd.concat(all_group_summaries, ignore_index=True)
        if all_group_summaries
        else pd.DataFrame()
    )
    combined_marker_score_summary = (
        pd.concat(all_marker_score_summaries, ignore_index=True)
        if all_marker_score_summaries
        else pd.DataFrame()
    )
    cell_score_distribution_summary = pd.DataFrame(all_cell_score_distribution_rows)
    marker_group_score_distribution = pd.DataFrame(all_marker_group_score_distribution_rows)
    cell_score_plot_sample = (
        pd.concat(all_score_plot_samples, ignore_index=True)
        if all_score_plot_samples
        else pd.DataFrame()
    )
    marker_group_score_plot_sample = (
        pd.concat(all_marker_group_score_plot_samples, ignore_index=True)
        if all_marker_group_score_plot_samples
        else pd.DataFrame()
    )

    if config.write_outputs:
        annotation_run_summary.to_csv(output_dir / "annotation_run_summary.csv", index=False)
        combined_group_summary.to_csv(output_dir / "combined_cell_annotation_group_summary.csv", index=False)
        combined_marker_score_summary.to_csv(output_dir / "combined_marker_score_summary.csv", index=False)
        cell_score_distribution_summary.to_csv(output_dir / "cell_score_distribution_summary.csv", index=False)
        marker_group_score_distribution.to_csv(output_dir / "marker_group_score_distribution.csv", index=False)
        cell_score_plot_sample.to_csv(output_dir / "cell_score_plot_sample.csv", index=False)
        marker_group_score_plot_sample.to_csv(output_dir / "marker_group_score_plot_sample.csv", index=False)

    return {
        "samples_to_annotate": samples_to_annotate,
        "annotation_input_readiness": annotation_input_readiness,
        "marker_dictionary_summary": marker_dictionary_summary,
        "marker_source_table": marker_source_table,
        "marker_group_inventory": marker_group_inventory,
        "annotation_run_summary": annotation_run_summary,
        "combined_group_summary": combined_group_summary,
        "combined_marker_score_summary": combined_marker_score_summary,
        "cell_score_distribution_summary": cell_score_distribution_summary,
        "marker_group_score_distribution": marker_group_score_distribution,
        "cell_score_plot_sample": cell_score_plot_sample,
        "marker_group_score_plot_sample": marker_group_score_plot_sample,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate all selected Xenium samples with marker-score labels."
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--review-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--samples", nargs="+")
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--normalization-target", type=float, default=10_000.0)
    parser.add_argument("--min-score", type=float, default=0.25)
    parser.add_argument("--min-margin", type=float, default=0.10)
    parser.add_argument("--min-detected-markers-in-cell", type=int, default=2)
    parser.add_argument("--uncertain-label", default="Uncertain")
    parser.add_argument("--no-direct-author-marker-groups", action="store_true")
    parser.add_argument("--no-panel-implied-when-direct-missing", action="store_true")
    parser.add_argument("--include-panel-implied-extra-groups", action="store_true")
    parser.add_argument("--allow-single-marker-groups", action="store_true")
    parser.add_argument("--write-full-score-matrix", action="store_true")
    parser.add_argument("--max-score-plot-cells-per-sample", type=int, default=5_000)
    parser.add_argument("--max-group-score-plot-cells-per-sample", type=int, default=2_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--no-write-outputs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AllSampleAnnotationConfig(
        normalization_target=args.normalization_target,
        min_score=args.min_score,
        min_margin=args.min_margin,
        min_detected_markers_in_cell=args.min_detected_markers_in_cell,
        uncertain_label=args.uncertain_label,
        use_direct_author_marker_groups=not args.no_direct_author_marker_groups,
        use_panel_implied_when_direct_missing=not args.no_panel_implied_when_direct_missing,
        include_panel_implied_extra_groups=args.include_panel_implied_extra_groups,
        allow_single_marker_groups=args.allow_single_marker_groups,
        write_full_score_matrix=args.write_full_score_matrix,
        max_score_plot_cells_per_sample=args.max_score_plot_cells_per_sample,
        max_group_score_plot_cells_per_sample=args.max_group_score_plot_cells_per_sample,
        write_outputs=not args.no_write_outputs,
        overwrite=args.overwrite,
        skip_completed=args.skip_completed,
    )
    results = run_all_sample_annotation(
        args.root,
        review_dir=args.review_dir,
        output_dir=args.output_dir,
        dataset_names=args.datasets,
        sample_names=args.samples,
        sample_limit=args.sample_limit,
        config=config,
    )
    summary = results["annotation_run_summary"]
    if not summary.empty:
        print(summary[["dataset", "sample", "status", "n_cells", "n_assigned_cells", "error"]])
    return int(summary["status"].eq("failed").any()) if not summary.empty else 0


if __name__ == "__main__":
    raise SystemExit(main())
