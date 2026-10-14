"""Cell-label mapping, composition, and agreement utilities."""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


BIOLOGICAL_LAYERS: dict[str, dict[str, str]] = {
    "author": {
        "refined": "author_refined_label",
        "broad": "author_broad_label",
    },
    "marker_score": {
        "refined": "marker_score_refined_label",
        "broad": "marker_score_broad_label",
    },
    "expression": {
        "refined": "expression_refined_label",
        "broad": "expression_broad_label",
    },
}

LAYER_ORDER = ("author", "marker_score", "expression", "xoa")
IMMUNE_BROAD_LABELS = {
    "T cells",
    "B cells",
    "NK cells",
    "Myeloid / other immune",
}

BROAD_LABEL_ORDER = (
    "Tumor cells",
    "Normal epithelial cells",
    "T cells",
    "B cells",
    "Myeloid / other immune",
    "Stromal cells",
    "Smooth muscle cells",
    "NK cells",
    "Other / uncertain",
    "Other / unmapped",
)


def natural_key(value: object) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    )


def clean_author_refined_label(value: object) -> str:
    """Remove only a leading numeric ``<cluster>_`` author prefix."""
    text = "Missing" if pd.isna(value) else str(value)
    return re.sub(r"^\d+_", "", text)


def normalize_label(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    text = text.replace("\\_", "_").replace("*", "")
    text = text.replace("&", " and ")
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^A-Za-z0-9+ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def collapse_label(
    refined_label: object,
    *,
    keep_b_cells_separate: bool = True,
    keep_nk_cells_separate: bool = True,
) -> tuple[str, str, str]:
    """Return ``(broad_label, mapping_rule, provenance)`` for one label."""
    normalized = normalize_label(refined_label)
    if not normalized or normalized in {"missing", "nan", "none"}:
        return "Other / uncertain", "missing_label", "PI response"

    uncertain_patterns = (
        r"\buncertain\b",
        r"\bunlabeled\b",
        r"\bunannotated\b",
        r"\bunassigned\b",
        r"\bnecrotic\b",
        r"\bstromal and t cell hybrid\b",
        r"\bt cell and tumor hybrid\b",
        r"\bactivated lymphoid cells?\b",
    )
    if any(re.search(pattern, normalized) for pattern in uncertain_patterns):
        return "Other / uncertain", "explicit_uncertain_or_hybrid", "Xenium_celltypes.pptx slide 2"

    if re.search(r"\bnk cells?\b", normalized):
        if keep_nk_cells_separate:
            return "NK cells", "nk_separate", "PI response"
        return "Myeloid / other immune", "nk_merged", "PI response"

    if re.search(r"\b(b cells?|b lymphocytes?)\b", normalized):
        if keep_b_cells_separate:
            return "B cells", "b_cell_separate", "PI response"
        return "Myeloid / other immune", "b_cell_merged", "PI response"

    t_cell_patterns = (
        r"\bt cells?\b",
        r"\bt lymphocytes?\b",
        r"\bcd4\+? t cells?\b",
        r"\bcd8\+? t cells?\b",
        r"\beffector t lymphocytes?\b",
    )
    if any(re.search(pattern, normalized) for pattern in t_cell_patterns):
        return "T cells", "t_cell_lineage", "Xenium_celltypes.pptx slides 1-2"

    myeloid_patterns = (
        r"\bmacrophages?\b",
        r"\bmyeloid\b",
        r"\bmonocytes?\b",
        r"\bdendritic\b",
        r"\bdcs?\b",
        r"\birf7\+? dcs?\b",
        r"\blamp3\+? dcs?\b",
        r"\bmast cells?\b",
        r"\bplasma cells?\b",
        r"\bneutrophils?\b",
    )
    if any(re.search(pattern, normalized) for pattern in myeloid_patterns):
        return "Myeloid / other immune", "non_t_immune", "Xenium_celltypes.pptx slides 1-2 and PI response"

    if re.search(r"\bsmooth muscle cells?\b", normalized):
        return "Smooth muscle cells", "smooth_muscle_separate", "PI response"

    stromal_patterns = (
        r"\badipocytes?\b",
        r"\bendothelial\b",
        r"\bfibroblasts?\b",
        r"\bstromal\b",
        r"\bcafs?\b",
        r"\bperivascular\b",
    )
    if any(re.search(pattern, normalized) for pattern in stromal_patterns):
        return "Stromal cells", "stromal_or_vascular", "Xenium_celltypes.pptx slides 1-2"

    normal_epithelial_patterns = (
        r"\bnormal epithelial\b",
        r"\bbreast glandular cells?\b",
        r"\bbreast myoepithelial cells?\b",
        r"\bmyoepithelial\b",
        r"\bmyoepi\b",
        r"\bductal epithelial\b",
        r"\balveolar epithelial\b",
        r"\bepithelial and myoepithelial\b",
        r"^epithelial cells?$",
    )
    if any(re.search(pattern, normalized) for pattern in normal_epithelial_patterns):
        return "Normal epithelial cells", "normal_epithelial_lineage", "Xenium_celltypes.pptx slides 1-2"

    tumor_patterns = (
        r"\bbreast cancer\b",
        r"\btumor\b",
        r"\btumour\b",
        r"\bdcis\b",
        r"\binvasive\b",
        r"\bepithelial and tumor\b",
    )
    if any(re.search(pattern, normalized) for pattern in tumor_patterns):
        return "Tumor cells", "tumor_lineage", "Xenium_celltypes.pptx slides 1-2"

    return "Other / unmapped", "unmapped_refined_label", "not mapped"


def validate_source_rows(
    table: pd.DataFrame,
    label_columns: Iterable[str],
    *,
    source_name: str,
) -> pd.DataFrame:
    """Reject conflicting duplicate cell IDs, then return one row per cell."""
    if table.empty:
        return table.copy()
    if "cell_id" not in table.columns:
        raise KeyError(f"{source_name} has no cell_id column")
    label_columns = [column for column in label_columns if column in table.columns]
    checked = table[["cell_id", *label_columns]].copy()
    checked["cell_id"] = checked["cell_id"].astype(str)
    duplicate_mask = checked["cell_id"].duplicated(keep=False)
    if not duplicate_mask.any():
        return checked.reset_index(drop=True)

    conflicting: list[str] = []
    if label_columns:
        duplicate_rows = checked.loc[
            duplicate_mask, ["cell_id", *label_columns]
        ].copy()
        duplicate_rows[label_columns] = duplicate_rows[label_columns].astype("string")
        label_cardinality = duplicate_rows.groupby("cell_id", sort=False)[
            label_columns
        ].nunique(dropna=True)
        conflicting = (
            label_cardinality.index[label_cardinality.gt(1).any(axis=1)]
            .astype(str)
            .tolist()
        )
    if conflicting:
        preview = ", ".join(conflicting[:10])
        raise ValueError(
            f"{source_name} assigns conflicting labels to {len(conflicting):,} "
            f"duplicate cell IDs; examples: {preview}"
        )
    return checked.drop_duplicates("cell_id", keep="first").reset_index(drop=True)


def add_broad_labels(
    evidence: pd.DataFrame,
    *,
    keep_b_cells_separate: bool = True,
    keep_nk_cells_separate: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add broad biological labels and return the observed collapse map."""
    result = evidence.copy()
    mapping_rows: list[dict[str, Any]] = []
    for layer, columns in BIOLOGICAL_LAYERS.items():
        refined_column = columns["refined"]
        broad_column = columns["broad"]
        status_column = f"{layer}_broad_mapping_status"
        if refined_column not in result.columns:
            result[broad_column] = pd.NA
            result[status_column] = "source_unavailable"
            continue
        observed = sorted(
            result[refined_column].dropna().astype(str).unique().tolist(),
            key=natural_key,
        )
        layer_lookup: dict[str, str] = {}
        status_lookup: dict[str, str] = {}
        for refined_label in observed:
            broad_label, rule, provenance = collapse_label(
                refined_label,
                keep_b_cells_separate=keep_b_cells_separate,
                keep_nk_cells_separate=keep_nk_cells_separate,
            )
            layer_lookup[refined_label] = broad_label
            status = "unmapped" if broad_label == "Other / unmapped" else "mapped"
            status_lookup[refined_label] = status
            mapping_rows.append(
                {
                    "layer": layer,
                    "refined_label": refined_label,
                    "normalized_label": normalize_label(refined_label),
                    "broad_label": broad_label,
                    "mapping_rule": rule,
                    "provenance": provenance,
                    "mapping_status": status,
                    "keep_b_cells_separate": bool(keep_b_cells_separate),
                    "keep_nk_cells_separate": bool(keep_nk_cells_separate),
                }
            )
        raw = result[refined_column]
        result[broad_column] = raw.map(
            lambda value: layer_lookup.get(str(value), pd.NA) if pd.notna(value) else pd.NA
        )
        result[status_column] = raw.map(
            lambda value: status_lookup.get(str(value), "missing_label")
            if pd.notna(value)
            else "missing_label"
        )
    mapping = pd.DataFrame(mapping_rows)
    if not mapping.empty:
        mapping = mapping.sort_values(
            ["layer", "refined_label"],
            key=lambda column: column.map(natural_key),
            kind="stable",
        ).reset_index(drop=True)
    return result, mapping


def _representation_specs(evidence: pd.DataFrame, granularity: str) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for layer, columns in BIOLOGICAL_LAYERS.items():
        column = columns[granularity]
        if column in evidence.columns and evidence[column].notna().any():
            specs[layer] = {"column": column, "is_cluster": False}
        elif (
            layer == "expression"
            and granularity == "refined"
            and "expression_cluster_id" in evidence.columns
            and evidence["expression_cluster_id"].notna().any()
        ):
            specs[layer] = {"column": "expression_cluster_id", "is_cluster": True}
        else:
            specs[layer] = {"column": column, "is_cluster": False}
    specs["xoa"] = {"column": "xoa_cluster_id", "is_cluster": True}
    return specs


def build_composition_table(evidence: pd.DataFrame) -> pd.DataFrame:
    """Create one dense long-form composition table for every representation."""
    output_columns = [
        "record_type",
        "scope",
        "dataset",
        "sample",
        "layer",
        "granularity",
        "label",
        "n_cells",
        "total_cells",
        "immune_cells",
        "mapped_cells",
        "mapping_coverage",
        "fraction_of_all_cells",
        "fraction_of_immune_cells",
        "n_samples",
        "n_samples_nonzero",
        "min_sample_fraction_of_all_cells",
        "median_sample_fraction_of_all_cells",
        "max_sample_fraction_of_all_cells",
        "min_sample_fraction_of_immune_cells",
        "median_sample_fraction_of_immune_cells",
        "max_sample_fraction_of_immune_cells",
    ]
    if evidence.empty:
        return pd.DataFrame(columns=output_columns)

    representation_rows: list[tuple[str, str, str, bool]] = []
    for layer, columns in BIOLOGICAL_LAYERS.items():
        for granularity, column in columns.items():
            if column in evidence.columns and evidence[column].notna().any():
                representation_rows.append((layer, granularity, column, False))
    for layer, column in (
        ("expression", "expression_cluster_id"),
        ("xoa", "xoa_cluster_id"),
    ):
        if column in evidence.columns and evidence[column].notna().any():
            representation_rows.append((layer, "cluster", column, True))

    sample_outputs: list[pd.DataFrame] = []
    dataset_outputs: list[pd.DataFrame] = []
    for layer, granularity, column, is_cluster in representation_rows:
        selected_columns = ["dataset", "sample", column]
        mapping_status_column = f"{layer}_broad_mapping_status"
        if mapping_status_column in evidence.columns:
            selected_columns.append(mapping_status_column)
        work = evidence[selected_columns].loc[evidence[column].notna()].copy()
        if work.empty:
            continue
        work = work.rename(columns={column: "label"})
        work["label"] = work["label"].astype(str)
        if is_cluster:
            work["immune_broad_label"] = pd.NA
        elif granularity == "broad":
            work["immune_broad_label"] = work["label"]
        else:
            work["immune_broad_label"] = work["label"].map(
                lambda value: collapse_label(value)[0]
            )
        work["is_immune"] = work["immune_broad_label"].isin(IMMUNE_BROAD_LABELS)
        work["is_mapped"] = (
            ~work[mapping_status_column].astype(str).eq("unmapped")
            if mapping_status_column in work.columns
            else np.nan
        )

        keys = ["dataset", "sample"]
        totals = (
            work.groupby(keys, dropna=False)
            .agg(
                total_cells=("label", "size"),
                immune_cells=("is_immune", "sum"),
            )
            .reset_index()
        )
        if mapping_status_column in work.columns:
            mapped = (
                work.groupby(keys, dropna=False)["is_mapped"]
                .sum()
                .rename("mapped_cells")
                .reset_index()
            )
            totals = totals.merge(mapped, on=keys, how="left")
        else:
            totals["mapped_cells"] = np.nan

        counts = (
            work.groupby([*keys, "label"], dropna=False)
            .size()
            .rename("n_cells")
            .reset_index()
        )
        dense_parts: list[pd.DataFrame] = []
        for dataset, dataset_totals in totals.groupby("dataset", sort=False):
            labels = sorted(
                work.loc[work["dataset"].astype(str).eq(str(dataset)), "label"]
                .astype(str)
                .unique()
                .tolist(),
                key=natural_key,
            )
            grid = dataset_totals[keys].merge(pd.DataFrame({"label": labels}), how="cross")
            dense_parts.append(grid)
        dense = pd.concat(dense_parts, ignore_index=True)
        dense = dense.merge(counts, on=[*keys, "label"], how="left").merge(
            totals, on=keys, how="left"
        )
        dense["n_cells"] = dense["n_cells"].fillna(0).astype(int)
        dense["fraction_of_all_cells"] = (
            dense["n_cells"] / dense["total_cells"].replace(0, np.nan)
        )
        dense["fraction_of_immune_cells"] = np.where(
            dense["label"].isin(IMMUNE_BROAD_LABELS) & dense["immune_cells"].gt(0),
            dense["n_cells"] / dense["immune_cells"].replace(0, np.nan),
            np.nan,
        )
        if granularity == "refined" and not is_cluster:
            refined_is_immune = dense["label"].map(
                lambda value: collapse_label(value)[0] in IMMUNE_BROAD_LABELS
            )
            dense["fraction_of_immune_cells"] = np.where(
                refined_is_immune & dense["immune_cells"].gt(0),
                dense["n_cells"] / dense["immune_cells"].replace(0, np.nan),
                np.nan,
            )
        dense["mapping_coverage"] = (
            dense["mapped_cells"] / dense["total_cells"].replace(0, np.nan)
        )
        dense.insert(0, "record_type", "composition")
        dense.insert(1, "scope", "sample")
        dense["layer"] = layer
        dense["granularity"] = granularity
        dense["n_samples"] = 1
        dense["n_samples_nonzero"] = dense["n_cells"].gt(0).astype(int)
        for prefix in ("min", "median", "max"):
            dense[f"{prefix}_sample_fraction_of_all_cells"] = dense["fraction_of_all_cells"]
            dense[f"{prefix}_sample_fraction_of_immune_cells"] = dense["fraction_of_immune_cells"]
        sample_outputs.append(dense[output_columns])

        dataset_summary = (
            dense.groupby(["dataset", "label"], dropna=False)
            .agg(
                n_cells=("n_cells", "sum"),
                total_cells=("total_cells", "sum"),
                immune_cells=("immune_cells", "sum"),
                mapped_cells=("mapped_cells", lambda values: values.sum(min_count=1)),
                n_samples=("sample", "nunique"),
                n_samples_nonzero=("n_cells", lambda values: int((values > 0).sum())),
                min_sample_fraction_of_all_cells=("fraction_of_all_cells", "min"),
                median_sample_fraction_of_all_cells=("fraction_of_all_cells", "median"),
                max_sample_fraction_of_all_cells=("fraction_of_all_cells", "max"),
                min_sample_fraction_of_immune_cells=("fraction_of_immune_cells", "min"),
                median_sample_fraction_of_immune_cells=("fraction_of_immune_cells", "median"),
                max_sample_fraction_of_immune_cells=("fraction_of_immune_cells", "max"),
            )
            .reset_index()
        )
        dataset_summary["fraction_of_all_cells"] = (
            dataset_summary["n_cells"]
            / dataset_summary["total_cells"].replace(0, np.nan)
        )
        dataset_summary["fraction_of_immune_cells"] = np.where(
            dataset_summary["label"].isin(IMMUNE_BROAD_LABELS)
            & dataset_summary["immune_cells"].gt(0),
            dataset_summary["n_cells"]
            / dataset_summary["immune_cells"].replace(0, np.nan),
            np.nan,
        )
        if granularity == "refined" and not is_cluster:
            refined_is_immune = dataset_summary["label"].map(
                lambda value: collapse_label(value)[0] in IMMUNE_BROAD_LABELS
            )
            dataset_summary["fraction_of_immune_cells"] = np.where(
                refined_is_immune & dataset_summary["immune_cells"].gt(0),
                dataset_summary["n_cells"]
                / dataset_summary["immune_cells"].replace(0, np.nan),
                np.nan,
            )
        dataset_summary["mapping_coverage"] = (
            dataset_summary["mapped_cells"]
            / dataset_summary["total_cells"].replace(0, np.nan)
        )
        dataset_summary.insert(0, "record_type", "composition")
        dataset_summary.insert(1, "scope", "dataset_pooled")
        dataset_summary.insert(3, "sample", "")
        dataset_summary["layer"] = layer
        dataset_summary["granularity"] = granularity
        dataset_outputs.append(dataset_summary[output_columns])

    if not sample_outputs and not dataset_outputs:
        return pd.DataFrame(columns=output_columns)
    output = pd.concat([*sample_outputs, *dataset_outputs], ignore_index=True)
    return output.sort_values(
        ["dataset", "scope", "sample", "layer", "granularity", "label"],
        key=lambda column: column.map(natural_key),
        kind="stable",
    ).reset_index(drop=True)


def build_pi_b_nk_check(
    composition: pd.DataFrame,
    *,
    evidence_layers: Iterable[str] = ("marker_score", "author"),
) -> pd.DataFrame:
    """Summarize B-cell and NK-cell abundance by biological evidence layer."""
    output_columns = [
        "evidence_layer",
        "dataset",
        "sample",
        "scope",
        "label",
        "n_cells",
        "total_cells",
        "immune_cells",
        "fraction_of_all_cells",
        "fraction_of_immune_cells",
        "n_samples",
        "n_samples_nonzero",
        "min_sample_fraction_of_all_cells",
        "median_sample_fraction_of_all_cells",
        "max_sample_fraction_of_all_cells",
        "min_sample_fraction_of_immune_cells",
        "median_sample_fraction_of_immune_cells",
        "max_sample_fraction_of_immune_cells",
    ]
    required = {
        "dataset",
        "sample",
        "scope",
        "layer",
        "granularity",
        "label",
        "n_cells",
        "total_cells",
        "immune_cells",
    }
    if composition.empty or not required.issubset(composition.columns):
        return pd.DataFrame(columns=output_columns)

    requested_layers = tuple(dict.fromkeys(map(str, evidence_layers)))
    biological = composition.loc[
        composition["scope"].astype(str).eq("sample")
        & composition["layer"].astype(str).isin(requested_layers)
        & composition["granularity"].astype(str).eq("refined")
    ].copy()
    if biological.empty:
        return pd.DataFrame(columns=output_columns)
    biological = biological.rename(columns={"layer": "evidence_layer"})

    sample_denominators = biological[
        ["evidence_layer", "dataset", "sample", "total_cells", "immune_cells"]
    ].drop_duplicates(["evidence_layer", "dataset", "sample"])
    biological["pi_label"] = biological["label"].map(
        lambda value: collapse_label(value)[0]
    )
    observed = (
        biological.loc[biological["pi_label"].isin({"B cells", "NK cells"})]
        .groupby(
            ["evidence_layer", "dataset", "sample", "pi_label"],
            dropna=False,
        )["n_cells"]
        .sum()
        .rename("n_cells")
        .reset_index()
        .rename(columns={"pi_label": "label"})
    )

    target_labels = pd.DataFrame({"label": ["B cells", "NK cells"]})
    sample_rows = sample_denominators.merge(target_labels, how="cross").merge(
        observed,
        on=["evidence_layer", "dataset", "sample", "label"],
        how="left",
    )
    sample_rows["n_cells"] = sample_rows["n_cells"].fillna(0).astype(int)
    sample_rows["fraction_of_all_cells"] = (
        sample_rows["n_cells"] / sample_rows["total_cells"].replace(0, np.nan)
    )
    sample_rows["fraction_of_immune_cells"] = (
        sample_rows["n_cells"] / sample_rows["immune_cells"].replace(0, np.nan)
    )
    sample_rows["scope"] = "sample"
    sample_rows["n_samples"] = 1
    sample_rows["n_samples_nonzero"] = sample_rows["n_cells"].gt(0).astype(int)
    for prefix in ("min", "median", "max"):
        sample_rows[f"{prefix}_sample_fraction_of_all_cells"] = sample_rows[
            "fraction_of_all_cells"
        ]
        sample_rows[f"{prefix}_sample_fraction_of_immune_cells"] = sample_rows[
            "fraction_of_immune_cells"
        ]

    pooled_rows = (
        sample_rows.groupby(["evidence_layer", "dataset", "label"], dropna=False)
        .agg(
            n_cells=("n_cells", "sum"),
            total_cells=("total_cells", "sum"),
            immune_cells=("immune_cells", "sum"),
            n_samples=("sample", "nunique"),
            n_samples_nonzero=("n_cells", lambda values: int((values > 0).sum())),
            min_sample_fraction_of_all_cells=("fraction_of_all_cells", "min"),
            median_sample_fraction_of_all_cells=("fraction_of_all_cells", "median"),
            max_sample_fraction_of_all_cells=("fraction_of_all_cells", "max"),
            min_sample_fraction_of_immune_cells=("fraction_of_immune_cells", "min"),
            median_sample_fraction_of_immune_cells=("fraction_of_immune_cells", "median"),
            max_sample_fraction_of_immune_cells=("fraction_of_immune_cells", "max"),
        )
        .reset_index()
    )
    pooled_rows["fraction_of_all_cells"] = (
        pooled_rows["n_cells"] / pooled_rows["total_cells"].replace(0, np.nan)
    )
    pooled_rows["fraction_of_immune_cells"] = (
        pooled_rows["n_cells"] / pooled_rows["immune_cells"].replace(0, np.nan)
    )
    pooled_rows["sample"] = ""
    pooled_rows["scope"] = "dataset_pooled"

    output = pd.concat(
        [sample_rows[output_columns], pooled_rows[output_columns]],
        ignore_index=True,
    )
    return output.sort_values(
        ["evidence_layer", "dataset", "scope", "sample", "label"],
        key=lambda column: column.map(natural_key),
        kind="stable",
    ).reset_index(drop=True)


def _comb2(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sum(values * (values - 1.0) / 2.0))


def clustering_metrics_from_contingency(counts: pd.DataFrame) -> dict[str, float]:
    matrix = counts.to_numpy(dtype=float)
    n = float(matrix.sum())
    if n <= 0:
        return {
            "adjusted_rand_index": np.nan,
            "normalized_mutual_info": np.nan,
            "homogeneity": np.nan,
            "completeness": np.nan,
            "v_measure": np.nan,
            "purity_a_to_b": np.nan,
            "purity_b_to_a": np.nan,
        }
    row = matrix.sum(axis=1)
    column = matrix.sum(axis=0)
    expected = np.outer(row, column) / n
    positive = matrix > 0
    mutual_info = float(
        np.sum((matrix[positive] / n) * np.log((matrix[positive] * n) / expected[positive]))
    )

    def entropy(values: np.ndarray) -> float:
        probabilities = values[values > 0] / n
        return float(-np.sum(probabilities * np.log(probabilities)))

    entropy_a = entropy(row)
    entropy_b = entropy(column)
    homogeneity = mutual_info / entropy_a if entropy_a > 0 else 1.0
    completeness = mutual_info / entropy_b if entropy_b > 0 else 1.0
    v_measure = (
        2.0 * homogeneity * completeness / (homogeneity + completeness)
        if homogeneity + completeness > 0
        else 0.0
    )
    mean_entropy = (entropy_a + entropy_b) / 2.0
    normalized_mutual_info = mutual_info / mean_entropy if mean_entropy > 0 else 1.0

    index = _comb2(matrix.ravel())
    row_pairs = _comb2(row)
    column_pairs = _comb2(column)
    total_pairs = n * (n - 1.0) / 2.0
    expected_index = row_pairs * column_pairs / total_pairs if total_pairs > 0 else 0.0
    maximum_index = (row_pairs + column_pairs) / 2.0
    denominator = maximum_index - expected_index
    adjusted_rand = (index - expected_index) / denominator if denominator else 1.0

    return {
        "adjusted_rand_index": float(adjusted_rand),
        "normalized_mutual_info": float(normalized_mutual_info),
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
        "purity_a_to_b": float(matrix.max(axis=1).sum() / n),
        "purity_b_to_a": float(matrix.max(axis=0).sum() / n),
    }


COMPARISON_COLUMNS = [
    "record_type",
    "scope",
    "dataset",
    "sample",
    "granularity",
    "layer_a",
    "layer_b",
    "representation_a",
    "representation_b",
    "a_is_cluster",
    "b_is_cluster",
    "status",
    "error",
    "label_a",
    "label_b",
    "n_cells",
    "n_layer_a_cells",
    "n_layer_b_cells",
    "n_matched_cells",
    "matched_fraction_of_layer_a",
    "matched_fraction_of_layer_b",
    "n_label_a_cells",
    "n_label_b_cells",
    "fraction_of_label_a",
    "fraction_of_label_b",
    "a_to_b_rank",
    "b_to_a_rank",
    "is_best_match_for_a",
    "is_best_match_for_b",
    "exact_agreement_cells",
    "exact_agreement_fraction",
    "adjusted_rand_index",
    "normalized_mutual_info",
    "homogeneity",
    "completeness",
    "v_measure",
    "purity_a_to_b",
    "purity_b_to_a",
]


def _comparison_for_pair(
    table: pd.DataFrame,
    *,
    scope: str,
    dataset: str,
    sample: str,
    granularity: str,
    layer_a: str,
    layer_b: str,
    spec_a: dict[str, Any],
    spec_b: dict[str, Any],
) -> list[dict[str, Any]]:
    column_a = str(spec_a["column"])
    column_b = str(spec_b["column"])
    a_is_cluster = bool(spec_a["is_cluster"])
    b_is_cluster = bool(spec_b["is_cluster"])
    base = {
        "scope": scope,
        "dataset": dataset,
        "sample": sample,
        "granularity": granularity,
        "layer_a": layer_a,
        "layer_b": layer_b,
        "representation_a": column_a,
        "representation_b": column_b,
        "a_is_cluster": a_is_cluster,
        "b_is_cluster": b_is_cluster,
        "error": "",
    }
    if column_a not in table.columns or column_b not in table.columns:
        return [{**base, "record_type": "summary", "status": "representation_unavailable"}]
    n_a = int(table[column_a].notna().sum())
    n_b = int(table[column_b].notna().sum())
    matched = table.loc[table[column_a].notna() & table[column_b].notna(), ["sample", column_a, column_b]].copy()
    if matched.empty:
        return [
            {
                **base,
                "record_type": "summary",
                "status": "no_matched_cells",
                "n_layer_a_cells": n_a,
                "n_layer_b_cells": n_b,
                "n_matched_cells": 0,
            }
        ]
    matched[column_a] = matched[column_a].astype(str)
    matched[column_b] = matched[column_b].astype(str)
    if scope == "dataset_pooled":
        if a_is_cluster:
            matched[column_a] = matched["sample"].astype(str) + "::" + matched[column_a]
        if b_is_cluster:
            matched[column_b] = matched["sample"].astype(str) + "::" + matched[column_b]
    counts = pd.crosstab(matched[column_a], matched[column_b], dropna=False)
    metrics = clustering_metrics_from_contingency(counts)
    exact_is_meaningful = not a_is_cluster and not b_is_cluster
    exact_cells = int((matched[column_a] == matched[column_b]).sum()) if exact_is_meaningful else np.nan
    summary = {
        **base,
        "record_type": "summary",
        "status": "compared",
        "n_layer_a_cells": n_a,
        "n_layer_b_cells": n_b,
        "n_matched_cells": len(matched),
        "matched_fraction_of_layer_a": len(matched) / n_a if n_a else np.nan,
        "matched_fraction_of_layer_b": len(matched) / n_b if n_b else np.nan,
        "exact_agreement_cells": exact_cells,
        "exact_agreement_fraction": exact_cells / len(matched) if exact_is_meaningful else np.nan,
        **metrics,
    }

    pairs = (
        matched.groupby([column_a, column_b], dropna=False)
        .size()
        .rename("n_cells")
        .reset_index()
        .rename(columns={column_a: "label_a", column_b: "label_b"})
    )
    a_totals = pairs.groupby("label_a")["n_cells"].sum().rename("n_label_a_cells")
    b_totals = pairs.groupby("label_b")["n_cells"].sum().rename("n_label_b_cells")
    pairs = pairs.merge(a_totals, on="label_a").merge(b_totals, on="label_b")
    pairs["fraction_of_label_a"] = pairs["n_cells"] / pairs["n_label_a_cells"]
    pairs["fraction_of_label_b"] = pairs["n_cells"] / pairs["n_label_b_cells"]
    pairs["a_to_b_rank"] = pairs.groupby("label_a")["n_cells"].rank(method="min", ascending=False).astype(int)
    pairs["b_to_a_rank"] = pairs.groupby("label_b")["n_cells"].rank(method="min", ascending=False).astype(int)
    pairs["is_best_match_for_a"] = pairs["a_to_b_rank"].eq(1)
    pairs["is_best_match_for_b"] = pairs["b_to_a_rank"].eq(1)
    pair_rows = [
        {
            **base,
            "record_type": "label_pair",
            "status": "compared",
            **row,
        }
        for row in pairs.to_dict("records")
    ]
    return [summary, *pair_rows]


def build_comparison_table(evidence: pd.DataFrame) -> pd.DataFrame:
    """Compare every available evidence-layer pair at refined and broad levels."""
    if evidence.empty:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)
    rows: list[dict[str, Any]] = []
    dataset_values = sorted(evidence["dataset"].dropna().astype(str).unique(), key=natural_key)
    for granularity in ("refined", "broad"):
        specs = _representation_specs(evidence, granularity)
        for dataset in dataset_values:
            dataset_table = evidence.loc[evidence["dataset"].astype(str).eq(dataset)].copy()
            sample_values = sorted(dataset_table["sample"].dropna().astype(str).unique(), key=natural_key)
            scopes = [("dataset_pooled", "", dataset_table)]
            scopes.extend(
                (
                    "sample",
                    sample,
                    dataset_table.loc[dataset_table["sample"].astype(str).eq(sample)].copy(),
                )
                for sample in sample_values
            )
            for scope, sample, scope_table in scopes:
                for layer_a, layer_b in itertools.combinations(LAYER_ORDER, 2):
                    rows.extend(
                        _comparison_for_pair(
                            scope_table,
                            scope=scope,
                            dataset=dataset,
                            sample=sample,
                            granularity=granularity,
                            layer_a=layer_a,
                            layer_b=layer_b,
                            spec_a=specs[layer_a],
                            spec_b=specs[layer_b],
                        )
                    )
    output = pd.DataFrame(rows)
    for column in COMPARISON_COLUMNS:
        if column not in output.columns:
            output[column] = np.nan
    output = output[COMPARISON_COLUMNS]
    return output.sort_values(
        ["dataset", "scope", "sample", "granularity", "layer_a", "layer_b", "record_type", "n_cells"],
        ascending=[True, True, True, True, True, True, True, False],
        key=lambda column: column.map(natural_key)
        if column.name not in {"n_cells"}
        else column,
        kind="stable",
    ).reset_index(drop=True)
