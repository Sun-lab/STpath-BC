"""Expression-based annotation of study-level reclusters."""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

import numpy as np
import pandas as pd

from . import automatic_cluster_annotation, cluster_review, signature_scores
from .automatic_cluster_annotation import AutomaticAnnotationConfig
from .cell_clustering import cluster_key_for_resolution, path_for_record, safe_name
from .cell_identity_evidence import BROAD_LABEL_ORDER, collapse_label, natural_key
from . import uncertain_reclustering


STUDY_SAMPLE_LABEL = "study_combined"
REVIEW_BROAD_LABELS = tuple(
    label for label in BROAD_LABEL_ORDER if label not in {"Other / uncertain", "Other / unmapped"}
) + ("Uncertain",)
REVIEW_DECISION_KEYS = ["dataset", "cluster_key", "cluster"]
REVIEW_DECISION_COLUMNS = REVIEW_DECISION_KEYS + [
    "review_class", "selected_label", "decision_status", "notes", "updated_at_utc",
]


def load_recluster_review_decisions(path: str | Path) -> pd.DataFrame:
    """Read reviewer-entered decisions, preserving cluster IDs and literal text."""
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame(columns=REVIEW_DECISION_COLUMNS)
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(REVIEW_DECISION_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"Decision CSV is missing columns: {', '.join(sorted(missing))}")
    if table.duplicated(REVIEW_DECISION_KEYS).any():
        raise ValueError("Decision CSV contains duplicate study/resolution/cluster rows.")
    return table


def save_recluster_review_decision(path: str | Path, record: dict[str, object]) -> pd.DataFrame:
    """Update one decision and atomically replace the CSV after a successful write."""
    path = Path(path)
    row = {
        column: "" if pd.isna(record.get(column, "")) else str(record.get(column, ""))
        for column in REVIEW_DECISION_COLUMNS
    }
    for column in REVIEW_DECISION_KEYS + ["selected_label", "decision_status"]:
        row[column] = row[column].strip()
    if any(not row[column] for column in REVIEW_DECISION_KEYS):
        raise ValueError("Select a study, resolution, and cluster before saving.")
    if row["decision_status"] not in {"needs_review", "tentative", "accepted"}:
        raise ValueError("Decision status must be needs_review, tentative, or accepted.")
    if row["selected_label"] not in REVIEW_BROAD_LABELS:
        raise ValueError("Select a PI broad cell type or Uncertain from the label dropdown.")
    row["updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    table = load_recluster_review_decisions(path)
    match = pd.Series(True, index=table.index)
    for column in REVIEW_DECISION_KEYS:
        match &= table[column].eq(row[column])
    # Retain any extra user-added columns on the row being updated.
    existing = table.loc[match].iloc[0].to_dict() if match.any() else {}
    table = pd.concat(
        [table.loc[~match], pd.DataFrame([{**existing, **row}])], ignore_index=True
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
            table.to_csv(handle, index=False)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return table


REQUIRED_GENE_REVIEW_FILES = (
    "marker_gene_table.csv",
    "gene_group_annotation_conflicts.csv",
    "panel_implied_marker_groups.csv",
)
UNRESOLVED_CANDIDATE_COLUMNS = (
    "dataset", "sample", "clustering_source", "cluster_key", "cluster",
    "n_cluster_cells", "candidate_label", "candidate_rank",
    "candidate_evidence_tier", "n_methods_nominating",
    "n_qualifying_methods_nominating", "nominating_methods",
    "qualifying_methods", "marker_score_candidate_fraction",
    "marker_score_fraction_gap", "assigned_score_fraction",
    "signature_median_score", "signature_score_gap", "signature_coverage",
    "signature_positive_cell_fraction", "gene_hint_support_fraction",
    "gene_hint_support_gap", "gene_hint_n_supporting_genes",
    "gene_hint_supporting_genes", "automatic_label_reason",
    "sample_specificity_category", "dominant_sample",
    "dominant_sample_enrichment",
    "jensen_shannon_divergence_from_study_baseline",
)


def recluster_key(resolution: float) -> str:
    return cluster_key_for_resolution(float(resolution)).replace("leiden_", "recluster_")


def load_recluster_marker_score_review(
    assignment_path: str | Path, score_dir: str | Path, study: str, cluster_key: str, cluster: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read full score matrices for every cell in one cluster; never subsample."""
    ids = ["dataset", "sample", "cell_id"]
    assignments = pd.read_csv(assignment_path, usecols=ids + [cluster_key], dtype=str, keep_default_na=False)
    assignments = assignments.loc[assignments["dataset"].eq(study) & assignments[cluster_key].eq(str(cluster))]
    if assignments[ids].eq("").any().any() or assignments.duplicated(ids).any():
        raise ValueError("Recluster assignments contain empty or duplicate cell keys")
    if assignments.empty:
        raise ValueError(f"No recluster assignments for {study}/{cluster_key}/{cluster}")
    totals = assignments.groupby([cluster_key, "sample"]).size().rename("n_cluster_cells").reset_index()
    paths = {sample: Path(score_dir) / safe_name(study) / safe_name(sample) / "cell_marker_score_matrix.csv.gz"
             for sample in assignments["sample"].unique()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Full score matrices are required; generate them with the score command and "
                                "WRITE_FULL_SCORE_MATRIX=True. Missing: " + "; ".join(missing))
    parts = []
    for sample, path in paths.items():
        expected = assignments.loc[assignments["sample"].eq(sample)]
        header = pd.read_csv(path, nrows=0).columns
        groups = [column for column in header if column.startswith("score__")]
        if not set(ids).issubset(header) or not groups:
            raise ValueError(f"{path} is not a full cell-by-marker-group score matrix")
        uncertain_reclustering._progress(f"{study}/{cluster_key}/cluster {cluster}: reading all {len(expected):,} cells from {sample}")
        sample_parts = []
        for chunk in pd.read_csv(path, usecols=ids + groups, dtype={key: str for key in ids},
                                 keep_default_na=False, chunksize=50_000):
            chunk = chunk.loc[chunk["cell_id"].isin(expected["cell_id"])]
            if chunk.empty:
                continue
            if not (chunk["dataset"].eq(study) & chunk["sample"].eq(sample)).all():
                raise ValueError(f"Incorrect dataset/sample identifiers in {path}")
            sample_parts.append(chunk)
        found = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame(columns=ids + groups)
        if found.duplicated(ids).any():
            raise ValueError(f"Duplicate cell score rows in {path}")
        absent = set(expected["cell_id"]) - set(found["cell_id"])
        if absent:
            raise ValueError(f"{path} lacks {len(absent):,} cells from the selected cluster. "
                             "No partial distribution will be shown; regenerate the full score matrix.")
        long = found.melt(id_vars=ids, value_vars=groups, var_name="group", value_name="marker_score")
        long["group"] = long["group"].str.removeprefix("score__")
        parts.append(long)
    scores = pd.concat(parts, ignore_index=True)
    scores[cluster_key] = str(cluster)
    scores["marker_score"] = pd.to_numeric(scores["marker_score"], errors="coerce")
    scores.loc[~np.isfinite(scores["marker_score"]), "marker_score"] = np.nan
    for column in ids + ["group", cluster_key]:
        scores[column] = scores[column].astype("category")
    return scores, totals


def plot_recluster_marker_scores(
    scores: pd.DataFrame, totals: pd.DataFrame, cluster_key: str, cluster: str,
) -> tuple[object | None, pd.DataFrame]:
    """Show observed all-group distributions and sample medians, without imputation."""
    import matplotlib.pyplot as plt

    selected = scores.loc[scores[cluster_key].eq(str(cluster))].copy()
    coverage = totals.loc[totals[cluster_key].eq(str(cluster)), ["sample", "n_cluster_cells"]].copy()
    finite = selected.loc[selected["marker_score"].notna()]
    observed = selected.groupby("sample", observed=True)["cell_id"].nunique()
    coverage["n_cells_with_saved_scores"] = coverage["sample"].map(observed).fillna(0).astype(int)
    coverage["fraction_with_saved_scores"] = coverage["n_cells_with_saved_scores"] / coverage["n_cluster_cells"]
    samples = sorted(coverage["sample"], key=natural_key)
    sample_keys = {sample: f"S{index + 1:02d}" for index, sample in enumerate(samples)}
    coverage.insert(0, "sample_key", coverage["sample"].map(sample_keys))
    if finite.empty:
        return None, coverage
    groups = finite.groupby("group", observed=True)["marker_score"].median().sort_values(ascending=False).index.tolist()
    groups += sorted(set(selected["group"]) - set(groups), key=natural_key)
    figure, (distribution, heatmap) = plt.subplots(1, 2, figsize=(max(16, 9 + len(samples) * 0.5), max(5, len(groups) * 0.32 + 3)))
    for position, group in enumerate(groups):
        values = finite.loc[finite["group"].eq(group), "marker_score"].to_numpy()
        if len(values):
            distribution.boxplot([values], positions=[position], vert=False, widths=0.65,
                                 showfliers=False, patch_artist=True,
                                 boxprops={"facecolor": "#69B4AD"}, medianprops={"color": "#202020"})
    group_counts = finite.groupby("group", observed=True).size()
    distribution.set_yticks(range(len(groups)))
    distribution.set_yticklabels([f'{group} (n={group_counts.get(group, 0):,})' for group in groups])
    distribution.set_ylim(len(groups) - 0.5, -0.5)
    distribution.set_xlabel("Marker score (all cluster cells; unweighted)")
    distribution.set_title("All marker groups: selected cluster")
    distribution.grid(axis="x", alpha=0.2)
    medians = finite.pivot_table(index="group", columns="sample", values="marker_score", aggfunc="median", observed=True)
    medians = medians.reindex(index=groups, columns=samples)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#DCDCDC")
    heat = heatmap.imshow(np.ma.masked_invalid(medians.to_numpy()), aspect="auto", cmap=cmap)
    heatmap.set_yticks(range(len(groups)))
    heatmap.set_yticklabels(groups)
    sample_counts = coverage.set_index("sample")
    heatmap.set_xticks(range(len(samples)))
    heatmap.set_xticklabels([
        sample_keys[sample] + f'\nn={sample_counts.loc[sample, "n_cells_with_saved_scores"]:,}' for sample in samples
    ], rotation=45, ha="right", fontsize=8)
    heatmap.set_title("Median score within each sample; gray = missing")
    figure.colorbar(heat, ax=heatmap, label="Median marker score", shrink=0.65)
    figure.suptitle(f"Cluster {cluster} | full notebook-15 scores; no cell subsampling", fontsize=12)
    figure.tight_layout()
    return figure, coverage


def propagate_recluster_labels(
    cells: pd.DataFrame, decisions: pd.DataFrame, cluster_key: str,
) -> pd.DataFrame:
    """Propagate consensus and accepted broad decisions without inventing subtypes."""
    ids = ["dataset", "sample", "cell_id"]
    required = ids + [cluster_key, "automatic_cluster_label", "automatic_label_confidence"]
    missing = set(required) - set(cells)
    if missing:
        raise ValueError(f"Per-cell recluster input is missing {sorted(missing)}")
    result = cells.copy().reset_index(drop=True)
    for column in required:
        result[column] = result[column].fillna("").astype(str)
    if result[ids].eq("").any().any() or result.duplicated(ids).any():
        raise ValueError("Per-cell recluster input has empty or duplicate cell keys")
    present = ~result[cluster_key].str.strip().str.lower().isin({"", "nan", "none", "missing", "unassigned", "<na>"})
    certain = present & automatic_cluster_annotation.cell_label_masks(result["automatic_cluster_label"])[1]
    certain &= result["automatic_label_confidence"].str.strip().str.lower().isin({"moderate", "high"})
    result["recluster_refined_label"] = result["automatic_cluster_label"].where(certain, "Uncertain")
    mapping = {label: collapse_label(label)[0] for label in result["recluster_refined_label"].unique()}
    mapping["Uncertain"] = "Uncertain"
    result["recluster_broad_label"] = result["recluster_refined_label"].map(mapping)
    result["recluster_status"] = np.where(certain, "certain_automatic_consensus", "unresolved_consensus")
    result["manual_decision_status"] = ""
    result["manual_selected_broad_label"] = ""
    result["manual_decision_updated_at_utc"] = ""
    if not decisions.empty:
        d = decisions.loc[
            decisions["cluster_key"].eq(cluster_key)
            & decisions["dataset"].isin(result["dataset"].unique())
        ].copy()
        if d.duplicated(REVIEW_DECISION_KEYS).any():
            raise ValueError("Duplicate manual decisions for a study/resolution/cluster")
        d["cluster"] = d["cluster"].astype(str)
        d = d.rename(columns={"cluster": cluster_key}).set_index(["dataset", cluster_key])
        keys = pd.MultiIndex.from_frame(result[["dataset", cluster_key]])
        for source, target in [("decision_status", "manual_decision_status"),
                               ("selected_label", "manual_selected_broad_label"),
                               ("updated_at_utc", "manual_decision_updated_at_utc")]:
            if source in d:
                result[target] = d[source].reindex(keys).fillna("").to_numpy()
        use = present & result["manual_decision_status"].eq("accepted")
        manual = result["manual_selected_broad_label"]
        if not manual[use].isin(REVIEW_BROAD_LABELS).all():
            raise ValueError("Accepted manual decision is not a PI broad label or Uncertain")
        compatible = certain & result["recluster_broad_label"].eq(manual) & manual.ne("Uncertain")
        result.loc[use & ~compatible, "recluster_refined_label"] = "Uncertain"
        result.loc[use, "recluster_broad_label"] = manual[use]
        result.loc[use, "recluster_status"] = "accepted_manual_broad_only"
        result.loc[use & compatible, "recluster_status"] = "accepted_manual_with_compatible_refined"
        result.loc[use & manual.eq("Uncertain"), "recluster_status"] = "accepted_manual_uncertain"
    return result


def export_reviewed_recluster_cells(
    output_dir: str | Path, decisions_path: str | Path, partitions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refresh reviewed cell exports from saved inputs; never rerun expression work."""
    output_dir = Path(output_dir)
    decisions_path = Path(decisions_path)
    decisions = load_recluster_review_decisions(decisions_path)
    selected = partitions[["dataset", "cluster_key"]].drop_duplicates()
    if selected.empty:
        raise ValueError("No study/resolution partitions selected")
    paths = [output_dir / safe_name(str(row.dataset)) / str(row.cluster_key) / "cell_recluster_annotations.csv.gz"
             for row in selected.itertuples(index=False)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing automatic per-cell files: " + "; ".join(missing))
    rows, counts_parts = [], []
    for row, path in zip(selected.itertuples(index=False), paths):
        uncertain_reclustering._progress(f"{row.dataset}/{row.cluster_key}: propagating saved cluster labels to cells")
        cells = pd.read_csv(path, dtype={key: str for key in ["dataset", "sample", "cell_id", row.cluster_key]}, keep_default_na=False)
        if cells.empty or set(cells["dataset"]) != {str(row.dataset)}:
            raise ValueError(f"Empty or incorrect dataset in {path}")
        d = decisions.loc[decisions["dataset"].eq(row.dataset) & decisions["cluster_key"].eq(row.cluster_key)]
        unknown = sorted(set(d.loc[d["decision_status"].eq("accepted"), "cluster"]) - set(cells[row.cluster_key]))
        if unknown:
            raise ValueError(f"{row.dataset}/{row.cluster_key}: accepted decisions reference absent clusters: {unknown}")
        reviewed = propagate_recluster_labels(cells, decisions, str(row.cluster_key))
        target = path.with_name("cell_recluster_annotations_reviewed.csv.gz")
        temporary = None
        try:
            with NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
            reviewed.to_csv(temporary, index=False, compression="gzip")
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        counts = reviewed.groupby(["dataset", "sample", "recluster_status"], dropna=False).size().rename("n_cells").reset_index()
        counts["cluster_key"] = str(row.cluster_key)
        counts["fraction_of_sample"] = counts["n_cells"] / counts.groupby(["dataset", "sample"])["n_cells"].transform("sum")
        counts_parts.append(counts)
        rows.append(dict(dataset=row.dataset, cluster_key=row.cluster_key, n_cells=len(reviewed),
                         n_accepted_manual_clusters=d["decision_status"].eq("accepted").sum(),
                         decisions_path=str(decisions_path) if decisions_path.is_file() else "",
                         reviewed_cell_path=str(target), status="success"))
    index = pd.DataFrame(rows)
    counts = pd.concat(counts_parts, ignore_index=True)
    index.to_csv(output_dir / "reviewed_cell_annotation_index.csv", index=False)
    counts.to_csv(output_dir / "reviewed_cell_label_status_counts.csv", index=False)
    return index, counts


def build_annotation_inventory(
    root: str | Path,
    *,
    reclustering_dir: str | Path | None = None,
    gene_panel_review_dir: str | Path | None = None,
    selected_resolution_by_study: dict[str, float] | None = None,
    default_resolution: float = 1.3,
    selected_studies: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit assignment, expression-matrix, and notebook-14 inputs by study."""
    root = Path(root).expanduser().resolve()
    reclustering_dir = Path(reclustering_dir or root / "uncertain_cell_reclustering")
    gene_panel_review_dir = Path(gene_panel_review_dir or root / "gene_panel_review")
    sample_inventory = uncertain_reclustering.build_reclustering_inventory(root)
    studies = sorted(sample_inventory["dataset"].astype(str).unique(), key=natural_key)
    if selected_studies is not None:
        selected = set(map(str, selected_studies))
        studies = [study for study in studies if study in selected]
    resolution_map = {str(key): float(value) for key, value in (selected_resolution_by_study or {}).items()}
    review_missing = [name for name in REQUIRED_GENE_REVIEW_FILES if not (gene_panel_review_dir / name).is_file()]
    rows: list[dict[str, object]] = []
    for study in studies:
        resolution = resolution_map.get(study, float(default_resolution))
        cluster_key = recluster_key(resolution)
        assignment_path = reclustering_dir / safe_name(study) / "study_recluster_assignments.csv.gz"
        header = uncertain_reclustering._csv_header(assignment_path)
        required = {
            "dataset", "sample", "cell_id", "assignment_method",
            "marker_score_label", cluster_key,
        }
        missing_assignment_columns = sorted(required - set(header))
        n_target_cells = 0
        n_direct_cells = 0
        participating_samples: list[str] = []
        if assignment_path.is_file() and not missing_assignment_columns:
            assignment_index = pd.read_csv(
                assignment_path,
                usecols=["sample", "assignment_method"],
                dtype={"sample": "string"},
            )
            n_target_cells = len(assignment_index)
            n_direct_cells = int(assignment_index["assignment_method"].isin({"full_leiden", "sketch_leiden"}).sum())
            participating_samples = sorted(assignment_index["sample"].astype(str).unique(), key=natural_key)
        study_samples = sample_inventory.loc[sample_inventory["dataset"].astype(str).eq(study)].copy()
        matrix_ready = set(
            study_samples.loc[study_samples["has_feature_matrix"].astype(bool), "sample"].astype(str)
        )
        missing_matrix_samples = sorted(set(participating_samples) - matrix_ready, key=natural_key)
        ready = (
            assignment_path.is_file()
            and not missing_assignment_columns
            and n_direct_cells > 0
            and not missing_matrix_samples
            and not review_missing
        )
        rows.append(
            {
                "dataset": study,
                "selected_resolution": resolution,
                "cluster_key": cluster_key,
                "assignment_path": path_for_record(assignment_path, root),
                "has_recluster_assignments": assignment_path.is_file(),
                "missing_assignment_columns": "; ".join(missing_assignment_columns),
                "n_target_cells": n_target_cells,
                "n_direct_expression_cells": n_direct_cells,
                "n_participating_samples": len(participating_samples),
                "participating_samples": "; ".join(participating_samples),
                "missing_matrix_samples": "; ".join(missing_matrix_samples),
                "missing_gene_review_files": "; ".join(review_missing),
                "ready_for_annotation": ready,
            }
        )
    return pd.DataFrame(rows), sample_inventory


def _study_hint_sources(
    study: str,
    hint_sources: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Pool the study's sample-specific notebook-14 relationships."""
    result = {key: table.copy() for key, table in hint_sources.items()}
    for key in ("marker_gene_table", "panel_implied_groups"):
        table = result.get(key, pd.DataFrame())
        if not table.empty and "dataset" in table.columns and "sample" in table.columns:
            pooled = table.loc[table["dataset"].astype(str).eq(str(study))].copy()
            pooled["sample"] = STUDY_SAMPLE_LABEL
            result[key] = pd.concat([table, pooled], ignore_index=True)
    return result


def _load_direct_expression(
    study: str,
    assignments: pd.DataFrame,
    sample_inventory: pd.DataFrame,
    root: Path,
):
    import anndata as ad

    direct_cells = assignments.loc[
        assignments["assignment_method"].isin({"full_leiden", "sketch_leiden"})
    ].copy()
    if direct_cells.empty:
        raise ValueError(f"{study}: no directly clustered cells are available")
    participating = set(direct_cells["sample"].astype(str))
    inventory = sample_inventory.loc[
        sample_inventory["dataset"].astype(str).eq(study)
        & sample_inventory["sample"].astype(str).isin(participating)
    ].copy()
    common_genes, gene_inventory = uncertain_reclustering.common_gene_inventory(inventory, root)
    if len(common_genes) < 2:
        raise ValueError(f"{study}: fewer than two common measured genes")
    parts = []
    records = inventory.sort_values("sample", key=lambda values: values.map(natural_key)).to_dict("records")
    for index, row in enumerate(records, start=1):
        sample = str(row["sample"])
        sample_cells = direct_cells.loc[direct_cells["sample"].astype(str).eq(sample)]
        if sample_cells.empty:
            continue
        uncertain_reclustering._progress(
            f"{study}: annotation matrix [{index}/{len(records)}] {sample} ({len(sample_cells):,} cells)"
        )
        matrix_path = uncertain_reclustering._resolve_path(row["matrix_path"], root)
        if matrix_path is None:
            raise FileNotFoundError(row["matrix_path"])
        parts.append(
            uncertain_reclustering._load_target_adata(
                matrix_path,
                sample_cells,
                common_genes,
                study,
                sample,
            )
        )
    return ad.concat(parts, join="inner", merge="same", index_unique=None), direct_cells, gene_inventory


def _global_cell_ids(table: pd.DataFrame) -> pd.Series:
    return table["sample"].astype(str) + "::" + table["cell_id"].astype(str)


def build_unresolved_cluster_candidates(
    automatic_labels: pd.DataFrame,
    *,
    uncertain_label: str = "Uncertain",
) -> pd.DataFrame:
    """Rank method-nominated labels for clusters that lack accepted consensus."""
    if automatic_labels.empty:
        return pd.DataFrame(columns=UNRESOLVED_CANDIDATE_COLUMNS)

    def clean(value: object) -> str:
        if value is None or pd.isna(value):
            return ""
        text = str(value).strip()
        return "" if text.casefold() in {"", "nan", "none", uncertain_label.casefold()} else text

    def truth(value: object) -> bool:
        return str(value).strip().casefold() in {"true", "1", "yes"}

    confidence = automatic_labels.get(
        "automatic_label_confidence", pd.Series("", index=automatic_labels.index)
    ).astype(str).str.casefold()
    unresolved = automatic_labels.loc[~confidence.isin({"moderate", "high"})].copy()
    rows: list[dict[str, object]] = []
    method_specs = (
        ("marker_score_vote", "score_vote_candidate_1", "score_vote_qualified"),
        ("ucell_signature", "signature_candidate_1", "signature_qualified"),
        ("de_gene_hint", "gene_hint_candidate_1", "gene_hint_qualified"),
    )
    for row in unresolved.to_dict("records"):
        nominations: dict[str, list[tuple[str, bool]]] = {}
        for method, label_column, qualified_column in method_specs:
            label = clean(row.get(label_column, ""))
            if label:
                nominations.setdefault(label, []).append(
                    (method, truth(row.get(qualified_column, False)))
                )
        if not nominations:
            nominations = {"": []}
        cluster_rows: list[dict[str, object]] = []
        for candidate, methods in nominations.items():
            method_names = [method for method, _qualified in methods]
            qualifying = [method for method, qualified in methods if qualified]
            if not candidate:
                tier = "no_candidate"
            elif len(methods) >= 2 and qualifying:
                tier = "near_consensus"
            elif len(qualifying) == 1:
                tier = "single_qualified_method"
            elif len(methods) >= 2:
                tier = "repeated_unqualified_signal"
            else:
                tier = "single_unqualified_signal"
            marker_nominates = "marker_score_vote" in method_names
            signature_nominates = "ucell_signature" in method_names
            gene_hint_nominates = "de_gene_hint" in method_names
            cluster_rows.append(
                {
                    "dataset": row.get("dataset", ""),
                    "sample": row.get("sample", ""),
                    "clustering_source": row.get("clustering_source", ""),
                    "cluster_key": row.get("cluster_key", ""),
                    "cluster": str(row.get("cluster", "")),
                    "n_cluster_cells": row.get("n_cluster_cells", np.nan),
                    "candidate_label": candidate,
                    "candidate_evidence_tier": tier,
                    "n_methods_nominating": len(methods),
                    "n_qualifying_methods_nominating": len(qualifying),
                    "nominating_methods": "; ".join(method_names),
                    "qualifying_methods": "; ".join(qualifying),
                    "marker_score_candidate_fraction": row.get("score_vote_candidate_1_fraction", np.nan) if marker_nominates else np.nan,
                    "marker_score_fraction_gap": row.get("score_vote_fraction_gap", np.nan) if marker_nominates else np.nan,
                    "assigned_score_fraction": row.get("assigned_score_fraction", np.nan) if marker_nominates else np.nan,
                    "signature_median_score": row.get("candidate_1_median_score", np.nan) if signature_nominates else np.nan,
                    "signature_score_gap": row.get("candidate_score_gap", np.nan) if signature_nominates else np.nan,
                    "signature_coverage": row.get("candidate_1_signature_coverage", np.nan) if signature_nominates else np.nan,
                    "signature_positive_cell_fraction": row.get("fraction_cells_with_positive_signature", np.nan) if signature_nominates else np.nan,
                    "gene_hint_support_fraction": row.get("candidate_1_support_fraction", np.nan) if gene_hint_nominates else np.nan,
                    "gene_hint_support_gap": row.get("candidate_support_gap", np.nan) if gene_hint_nominates else np.nan,
                    "gene_hint_n_supporting_genes": row.get("candidate_1_n_supporting_genes", np.nan) if gene_hint_nominates else np.nan,
                    "gene_hint_supporting_genes": row.get("candidate_1_supporting_genes", "") if gene_hint_nominates else "",
                    "automatic_label_reason": row.get("automatic_label_reason", ""),
                    "sample_specificity_category": row.get("sample_specificity_category", ""),
                    "dominant_sample": row.get("dominant_sample", ""),
                    "dominant_sample_enrichment": row.get("dominant_sample_enrichment", np.nan),
                    "jensen_shannon_divergence_from_study_baseline": row.get("jensen_shannon_divergence_from_study_baseline", np.nan),
                }
            )
        tier_order = {
            "near_consensus": 0,
            "single_qualified_method": 1,
            "repeated_unqualified_signal": 2,
            "single_unqualified_signal": 3,
            "no_candidate": 4,
        }
        cluster_rows.sort(
            key=lambda candidate: (
                tier_order[candidate["candidate_evidence_tier"]],
                -candidate["n_qualifying_methods_nominating"],
                -candidate["n_methods_nominating"],
                natural_key(candidate["candidate_label"]),
            )
        )
        for rank, candidate in enumerate(cluster_rows, start=1):
            candidate["candidate_rank"] = rank
            rows.append(candidate)
    return pd.DataFrame(rows).reindex(columns=UNRESOLVED_CANDIDATE_COLUMNS)


def route_unresolved_cluster_review(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Route review using marker nominations; sample sharing never votes on identity."""
    keys = ["dataset", "cluster_key", "cluster"]
    summary_columns = keys + [
        "n_cluster_cells", "review_class", "provisional_review_label", "review_reason",
        "n_epithelial_methods", "n_other_methods", "n_unmapped_methods",
        "n_qualified_epithelial_methods", "n_qualified_other_methods",
        "epithelial_candidates", "other_candidates", "unmapped_candidates",
        "sample_specificity_category", "dominant_sample", "dominant_sample_enrichment",
        "jensen_shannon_divergence_from_study_baseline",
    ]
    enriched = candidates.copy()
    enriched["candidate_review_lineage"] = pd.Series(index=enriched.index, dtype="string")
    if enriched.empty:
        return enriched, pd.DataFrame(columns=summary_columns)

    def lineage(label: object) -> str:
        broad, _rule, _provenance = collapse_label(label)
        if broad in {"Tumor cells", "Normal epithelial cells"}:
            return "epithelial"
        if broad in {"Other / uncertain", "Other / unmapped"}:
            return "unmapped"
        return "other"

    def methods(value: object) -> set[str]:
        return set() if pd.isna(value) else {part.strip() for part in str(value).split(";") if part.strip()}

    enriched["candidate_review_lineage"] = enriched["candidate_label"].map(lineage)
    rows = []
    for identity, group in enriched.groupby(keys, sort=False, dropna=False):
        nominees = {name: set() for name in ("epithelial", "other", "unmapped")}
        qualified = {name: set() for name in nominees}
        labels = {name: [] for name in nominees}
        for candidate in group.to_dict("records"):
            name = candidate["candidate_review_lineage"]
            nominees[name].update(methods(candidate.get("nominating_methods", "")))
            qualified[name].update(methods(candidate.get("qualifying_methods", "")))
            value = candidate.get("candidate_label")
            if pd.notna(value) and str(value).strip():
                labels[name].append(str(value))
        # Broad-lineage agreement is for triage, not a new refined-label consensus.
        supported = [
            name for name in ("epithelial", "other")
            if len(nominees[name]) >= 2 and qualified[name]
            and not any(qualified[opponent] for opponent in nominees if opponent != name)
        ]
        if supported == ["epithelial"]:
            review_class = "likely_epithelial"
            provisional_label = "Epithelial, tumor status unresolved"
            reason = "At least two methods nominate epithelial identities; at least one qualifies and no opposing lineage qualifies."
        elif supported == ["other"]:
            review_class = "likely_other"
            provisional_label = "Non-epithelial, refined identity unresolved"
            reason = "At least two methods nominate non-epithelial identities; at least one qualifies and no opposing lineage qualifies."
        else:
            review_class = "mixed_or_insufficient"
            provisional_label = "Uncertain"
            reason = "Lineage nominations conflict or lack sufficient qualifying support; review all candidates."
        first = group.iloc[0]
        row = dict(zip(keys, identity))
        row.update({
            "n_cluster_cells": first.get("n_cluster_cells", np.nan),
            "review_class": review_class,
            "provisional_review_label": provisional_label,
            "review_reason": reason,
            "n_epithelial_methods": len(nominees["epithelial"]),
            "n_other_methods": len(nominees["other"]),
            "n_unmapped_methods": len(nominees["unmapped"]),
            "n_qualified_epithelial_methods": len(qualified["epithelial"]),
            "n_qualified_other_methods": len(qualified["other"]),
            "epithelial_candidates": "; ".join(sorted(set(labels["epithelial"]), key=natural_key)),
            "other_candidates": "; ".join(sorted(set(labels["other"]), key=natural_key)),
            "unmapped_candidates": "; ".join(sorted(set(labels["unmapped"]), key=natural_key)),
        })
        for column in summary_columns[-4:]:
            row[column] = first.get(column, np.nan)
        rows.append(row)
    summary = pd.DataFrame(rows, columns=summary_columns)
    return enriched, summary


def analyze_study_reclusters(
    study: str,
    cluster_key: str,
    assignment_path: str | Path,
    sample_inventory: pd.DataFrame,
    *,
    root: str | Path,
    output_dir: str | Path,
    gene_panel_review_dir: str | Path,
    annotation_config: AutomaticAnnotationConfig = AutomaticAnnotationConfig(),
    normalization_target: float = 100_000.0,
    lfc_pseudocount: float = 0.1,
    min_fraction_expressing: float = 0.05,
    min_fraction_difference: float = 0.05,
    min_log2_fold_change: float = 0.25,
    top_n_markers: int = 25,
    run_wilcoxon: bool = True,
    max_cells_for_wilcoxon: int | None = 100_000,
    min_signature_genes: int = 3,
    ucell_max_rank: int | None = None,
    ucell_chunk_size: int = 10_000,
    random_state: int = 1729,
    overwrite_existing: bool = False,
    skip_completed: bool = True,
) -> dict[str, object]:
    """Discover markers and assign a two-of-three label to one study partition."""
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir)
    gene_panel_review_dir = Path(gene_panel_review_dir)
    assignment_path = Path(assignment_path)
    if not assignment_path.is_absolute():
        assignment_path = root / assignment_path
    study_dir = output_dir / safe_name(study) / safe_name(cluster_key)
    automatic_path = study_dir / "automatic_recluster_labels.csv"
    cell_path = study_dir / "cell_recluster_annotations.csv.gz"
    candidate_path = study_dir / "unresolved_cluster_candidates.csv"
    if skip_completed and not overwrite_existing and automatic_path.is_file() and cell_path.is_file():
        automatic_columns = set(pd.read_csv(automatic_path, nrows=0).columns)
        cell_columns = set(pd.read_csv(cell_path, nrows=0).columns)
        reusable_columns = {
            "cluster", "automatic_cluster_label", "automatic_label_confidence",
        }.issubset(automatic_columns) and {
            "automatic_cluster_label", "automatic_label_confidence",
        }.issubset(cell_columns)
        if reusable_columns:
            automatic_existing = pd.read_csv(
                automatic_path,
                dtype={"cluster": "string"},
                low_memory=False,
            )
            current_sources = pd.Series(dtype="string")
            if "post_reclustering_label_source" in cell_columns:
                current_sources = pd.read_csv(
                    cell_path,
                    usecols=["post_reclustering_label_source"],
                )["post_reclustering_label_source"].astype("string")
            current_schema = (
                candidate_path.is_file()
                and not current_sources.empty
                and not current_sources.eq("marker_score_fallback").any()
            )
            if current_schema:
                candidate_existing = pd.read_csv(
                    candidate_path,
                    dtype={"cluster": "string"},
                    low_memory=False,
                )
                certain_existing = current_sources.eq("recluster_consensus")
                status = "skipped_existing"
                uncertain_reclustering._progress(
                    f"{study}/{cluster_key}: completed annotation reused"
                )
            else:
                uncertain_reclustering._progress(
                    f"{study}/{cluster_key}: upgrading saved labels without recomputing expression evidence"
                )
                candidate_existing = build_unresolved_cluster_candidates(
                    automatic_existing,
                    uncertain_label=annotation_config.uncertain_label,
                )
                cells_existing = pd.read_csv(
                    cell_path,
                    dtype={"cell_id": "string", cluster_key: "string"},
                    low_memory=False,
                )
                _available, biological = automatic_cluster_annotation.cell_label_masks(
                    cells_existing["automatic_cluster_label"]
                )
                certain_existing = (
                    biological
                    & cells_existing["automatic_label_confidence"]
                    .astype(str)
                    .str.casefold()
                    .isin({"moderate", "high"})
                )
                cells_existing["post_reclustering_label"] = annotation_config.uncertain_label
                cells_existing["post_reclustering_label_source"] = "unresolved"
                cells_existing.loc[certain_existing, "post_reclustering_label"] = (
                    cells_existing.loc[certain_existing, "automatic_cluster_label"]
                )
                cells_existing.loc[
                    certain_existing, "post_reclustering_label_source"
                ] = "recluster_consensus"
                cells_existing["recluster_consensus_is_certain"] = certain_existing
                candidate_existing.to_csv(candidate_path, index=False)
                cells_existing.to_csv(cell_path, index=False, compression="gzip")
                current_sources = cells_existing["post_reclustering_label_source"].astype("string")
                status = "updated_existing"
                uncertain_reclustering._progress(
                    f"{study}/{cluster_key}: per-cell hierarchy and unresolved candidates updated"
                )
            return {
                "dataset": study,
                "cluster_key": cluster_key,
                "status": status,
                "n_target_cells": len(current_sources),
                "n_clusters": automatic_existing["cluster"].nunique(),
                "n_certain_clusters": int(
                    automatic_existing["automatic_label_confidence"]
                    .astype(str)
                    .str.casefold()
                    .isin({"moderate", "high"})
                    .sum()
                ),
                "n_cells_with_recluster_consensus": int(certain_existing.sum()),
                "n_clusters_without_certain_recluster_label": int(
                    candidate_existing[["cluster_key", "cluster"]].drop_duplicates().shape[0]
                ),
                "n_cells_without_certain_recluster_label": int((~certain_existing).sum()),
                "automatic_labels_path": path_for_record(automatic_path, root),
                "cell_annotations_path": path_for_record(cell_path, root),
                "unresolved_candidates_path": path_for_record(candidate_path, root),
            }

    uncertain_reclustering._progress(f"{study}/{cluster_key}: loading recluster assignments")
    assignments = pd.read_csv(assignment_path, dtype={"cell_id": "string", cluster_key: "string"}, low_memory=False)
    for column in ("dataset", "sample", "cell_id", cluster_key):
        assignments[column] = assignments[column].astype(str)
    if assignments.duplicated(["sample", "cell_id"]).any():
        raise ValueError(f"{study}: duplicate sample/cell_id keys in recluster assignments")
    direct, direct_cells, gene_inventory = _load_direct_expression(
        study, assignments, sample_inventory, root
    )
    direct_lookup = direct_cells.assign(_global_id=_global_cell_ids(direct_cells)).set_index("_global_id")
    labels = direct_lookup[cluster_key].reindex(direct.obs_names.astype(str))
    if labels.isna().any():
        raise ValueError(f"{study}: direct expression cells do not align with {cluster_key}")

    uncertain_reclustering._progress(f"{study}/{cluster_key}: calculating cluster-vs-rest expression effects")
    marker_table = cluster_review.calculate_cluster_expression_marker_metrics(
        direct,
        labels,
        dataset=study,
        sample=STUDY_SAMPLE_LABEL,
        cluster_key=cluster_key,
        normalization_target=normalization_target,
        lfc_pseudocount=lfc_pseudocount,
        min_fraction_expressing=min_fraction_expressing,
        min_fraction_difference=min_fraction_difference,
        min_log2_fold_change=min_log2_fold_change,
    )
    if run_wilcoxon:
        uncertain_reclustering._progress(f"{study}/{cluster_key}: running Wilcoxon marker tests")
        wilcoxon = cluster_review.run_scanpy_wilcoxon_markers(
            direct,
            labels,
            cluster_key,
            normalization_target=normalization_target,
            max_cells=max_cells_for_wilcoxon,
            random_state=random_state,
        )
        if not wilcoxon.empty:
            marker_table = marker_table.merge(wilcoxon, on=["cluster", "var_name"], how="left")
    marker_table = cluster_review.add_wilcoxon_support_indicator(marker_table)
    top_markers = cluster_review.top_cluster_expression_markers(marker_table, top_n=top_n_markers)
    marker_summary = cluster_review.summarize_cluster_expression_markers(top_markers)

    hint_sources = cluster_review.load_gene_marker_hint_sources(
        root, gene_panel_review_dir=gene_panel_review_dir
    )
    study_hints = _study_hint_sources(study, hint_sources)
    annotated_markers = cluster_review.annotate_cluster_top_markers_with_gene_hints(
        top_markers, root=root, hint_sources=study_hints
    )
    gene_hint_groups = cluster_review.summarize_cluster_marker_gene_group_candidates(annotated_markers)
    gene_hint_top_two = cluster_review.top_two_gene_hint_candidates(
        annotated_markers,
        gene_hint_groups,
        clustering_source="study_recluster",
    )

    uncertain_reclustering._progress(f"{study}/{cluster_key}: summarizing notebook-15 marker-score votes")
    vote_assignments = assignments[["sample", "cell_id", cluster_key]].copy()
    vote_assignments["cell_id"] = _global_cell_ids(vote_assignments)
    vote_labels = assignments[["sample", "cell_id", "marker_score_label"]].copy()
    vote_labels["cell_id"] = _global_cell_ids(vote_labels)
    vote_labels = vote_labels.rename(columns={"marker_score_label": "group"})
    if "marker_score_call_status" in assignments.columns:
        vote_labels["call_status"] = assignments["marker_score_call_status"].astype(str)
    vote_summary, vote_composition = automatic_cluster_annotation.summarize_marker_score_labels_by_cluster(
        vote_assignments,
        vote_labels,
        dataset=study,
        sample=STUDY_SAMPLE_LABEL,
        clustering_source="study_recluster",
        cluster_key=cluster_key,
        uncertain_label=annotation_config.uncertain_label,
    )

    uncertain_reclustering._progress(f"{study}/{cluster_key}: calculating refined and broad UCell signatures")
    signature_sets = signature_scores.build_sample_signature_sets(
        study,
        STUDY_SAMPLE_LABEL,
        direct.var_names.astype(str),
        hint_sources=study_hints,
        min_signature_genes=min_signature_genes,
    )
    signature_coverage = signature_sets["coverage"]
    signatures_by_granularity = signature_sets["signatures"]
    signature_cluster_summary = pd.DataFrame()
    signature_top_two = pd.DataFrame(
        columns=[
            "dataset", "sample", "clustering_source", "cluster_key", "cluster",
            "method", "granularity",
        ]
    )
    combined_signatures = {
        f"{granularity}::{cell_type}": genes
        for granularity in ("refined", "broad")
        for cell_type, genes in signatures_by_granularity[granularity].items()
    }
    if combined_signatures:
        combined_scores, _signature_metadata = signature_scores.compute_panel_aware_ucell_scores(
            direct,
            combined_signatures,
            max_rank=ucell_max_rank,
            chunk_size=ucell_chunk_size,
        )
        signature_assignments = pd.DataFrame(
            {"cell_id": direct.obs_names.astype(str), cluster_key: labels.to_numpy()}
        )
        signature_parts = []
        for granularity in ("refined", "broad"):
            prefix = f"{granularity}::"
            columns = [column for column in combined_scores if column.startswith(prefix)]
            if not columns:
                continue
            scores = combined_scores[columns].rename(columns=lambda value: value[len(prefix):])
            coverage_subset = signature_coverage.loc[
                signature_coverage["granularity"].eq(granularity)
                & signature_coverage["eligible_for_primary_analysis"].astype(bool)
            ]
            signature_parts.append(
                signature_scores.summarize_cell_type_scores_by_cluster(
                    scores,
                    signature_assignments,
                    dataset=study,
                    sample=STUDY_SAMPLE_LABEL,
                    clustering_source="study_recluster",
                    cluster_key=cluster_key,
                    method="ucell",
                    granularity=granularity,
                    coverage=coverage_subset,
                    rank_statistic="median",
                )
            )
        signature_cluster_summary = pd.concat(signature_parts, ignore_index=True)
        signature_top_two = signature_scores.build_signature_top_two(
            signature_cluster_summary
        )

    if gene_hint_top_two.empty:
        gene_hint_top_two = pd.DataFrame(
            columns=["dataset", "sample", "clustering_source", "cluster_key", "cluster"]
        )
    uncertain_reclustering._progress(f"{study}/{cluster_key}: applying two-of-three consensus")
    automatic_labels = automatic_cluster_annotation.assign_automatic_cluster_labels(
        vote_summary,
        signature_top_two,
        gene_hint_top_two,
        config=annotation_config,
        signature_method="ucell",
        signature_granularity="refined",
    )
    gene_detail_columns = [
        "dataset", "sample", "clustering_source", "cluster_key", "cluster",
        "candidate_1_supporting_genes",
    ]
    if "candidate_1_supporting_genes" in gene_hint_top_two.columns:
        automatic_labels = automatic_labels.merge(
            gene_hint_top_two[gene_detail_columns],
            on=["dataset", "sample", "clustering_source", "cluster_key", "cluster"],
            how="left",
        )
    specificity, sample_composition = uncertain_reclustering.summarize_cluster_sample_specificity(
        assignments, cluster_key
    )
    automatic_labels = automatic_labels.merge(
        specificity.drop(columns=["n_cluster_cells"], errors="ignore"),
        on=["dataset", "cluster_key", "cluster"],
        how="left",
    )
    unresolved_candidates = build_unresolved_cluster_candidates(
        automatic_labels,
        uncertain_label=annotation_config.uncertain_label,
    )

    mapping = automatic_labels[[
        "cluster", "automatic_cluster_label", "automatic_label_confidence",
        "automatic_label_reason", "n_qualifying_evidence_methods",
        "n_supporting_evidence_methods",
    ]].rename(columns={"cluster": cluster_key})
    cells = assignments.merge(mapping, on=cluster_key, how="left", validate="many_to_one")
    _available, biological = automatic_cluster_annotation.cell_label_masks(cells["automatic_cluster_label"])
    cluster_certain = biological & cells["automatic_label_confidence"].astype(str).str.casefold().isin({"moderate", "high"})
    cells["post_reclustering_label"] = annotation_config.uncertain_label
    cells["post_reclustering_label_source"] = "unresolved"
    cells.loc[cluster_certain, "post_reclustering_label"] = cells.loc[cluster_certain, "automatic_cluster_label"]
    cells.loc[cluster_certain, "post_reclustering_label_source"] = "recluster_consensus"
    cells["recluster_consensus_is_certain"] = cluster_certain

    study_dir.mkdir(parents=True, exist_ok=True)
    marker_table.to_csv(study_dir / "expression_markers_all_genes.csv.gz", index=False, compression="gzip")
    top_markers.to_csv(study_dir / "expression_markers_top.csv", index=False)
    marker_summary.to_csv(study_dir / "expression_marker_summary.csv", index=False)
    annotated_markers.to_csv(study_dir / "expression_markers_top_with_gene_hints.csv", index=False)
    gene_hint_groups.to_csv(study_dir / "gene_hint_group_candidates.csv", index=False)
    gene_hint_top_two.to_csv(study_dir / "gene_hint_top_two.csv", index=False)
    vote_summary.to_csv(study_dir / "marker_score_vote_summary.csv", index=False)
    vote_composition.to_csv(study_dir / "marker_score_vote_composition.csv", index=False)
    signature_coverage.to_csv(study_dir / "signature_coverage.csv", index=False)
    signature_cluster_summary.to_csv(study_dir / "signature_cluster_summary.csv.gz", index=False, compression="gzip")
    signature_top_two.to_csv(study_dir / "signature_top_two.csv", index=False)
    specificity.to_csv(study_dir / "cluster_sample_specificity.csv", index=False)
    sample_composition.to_csv(study_dir / "cluster_sample_composition.csv", index=False)
    gene_inventory.to_csv(study_dir / "common_gene_inventory.csv", index=False)
    automatic_labels.to_csv(automatic_path, index=False)
    unresolved_candidates.to_csv(candidate_path, index=False)
    cells.to_csv(cell_path, index=False, compression="gzip")
    uncertain_reclustering._progress(f"{study}/{cluster_key}: annotation outputs written")
    return {
        "dataset": study,
        "cluster_key": cluster_key,
        "status": "success",
        "n_target_cells": len(assignments),
        "n_direct_expression_cells": len(direct_cells),
        "n_clusters": assignments[cluster_key].nunique(),
        "n_certain_clusters": int(automatic_labels["automatic_label_confidence"].isin({"moderate", "high"}).sum()),
        "n_cells_with_recluster_consensus": int(cluster_certain.sum()),
        "n_clusters_without_certain_recluster_label": int(unresolved_candidates[["cluster_key", "cluster"]].drop_duplicates().shape[0]),
        "n_cells_without_certain_recluster_label": int((~cluster_certain).sum()),
        "automatic_labels_path": path_for_record(automatic_path, root),
        "cell_annotations_path": path_for_record(cell_path, root),
        "unresolved_candidates_path": path_for_record(candidate_path, root),
    }


def run_annotation_inventory(
    annotation_inventory: pd.DataFrame,
    sample_inventory: pd.DataFrame,
    *,
    root: str | Path,
    output_dir: str | Path,
    gene_panel_review_dir: str | Path,
    **analysis_parameters: object,
) -> pd.DataFrame:
    """Run ready studies independently and checkpoint after every study."""
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir)
    rows: list[dict[str, object]] = []
    selected = annotation_inventory.loc[annotation_inventory["ready_for_annotation"].astype(bool)]
    for index, row in enumerate(selected.itertuples(index=False), start=1):
        uncertain_reclustering._progress(f"[{index}/{len(selected)}] Annotating {row.dataset}/{row.cluster_key}")
        try:
            result = analyze_study_reclusters(
                str(row.dataset),
                str(row.cluster_key),
                str(row.assignment_path),
                sample_inventory,
                root=root,
                output_dir=output_dir,
                gene_panel_review_dir=gene_panel_review_dir,
                **analysis_parameters,
            )
        except Exception as error:
            result = {
                "dataset": str(row.dataset),
                "cluster_key": str(row.cluster_key),
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(limit=8),
            }
        rows.append(result)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_dir / "annotation_run_summary.csv", index=False)
    return pd.DataFrame(rows)
