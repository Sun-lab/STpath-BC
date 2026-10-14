"""Operational per-sample Xenium expression clustering.

This module packages the workflow from ``cluster_all_sample_cells.ipynb`` into
importable functions and a command-line entrypoint.  It clusters each selected
sample independently from its Xenium expression matrix and writes outputs under
``cell_clustering/``.
"""

from __future__ import annotations

import argparse
import gc
import re
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd

from . import dataset_config, expression_annotation
from .gene_panel_utils import is_gene_expression, natural_key


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ClusteringConfig:
    normalization_target: float = 10_000.0
    min_counts_per_cell: float = 10.0
    min_genes_per_cell: int = 5
    min_cells_per_gene: int = 10
    use_highly_variable_genes: bool = False
    n_highly_variable_genes: int = 2_000
    n_pcs: int = 20
    n_neighbors: int = 20
    leiden_resolutions: tuple[float, ...] = (1.3,)
    default_resolution: float = 1.3
    random_state: int = 1729
    max_cells_per_sample_for_clustering: int | None = None
    run_umap: bool = True
    max_umap_plot_cells: int | None = 50_000
    umap_point_size: float = 1.0
    run_cluster_markers: bool = False
    n_marker_genes_per_cluster: int = 20
    max_cells_for_marker_test: int | None = 100_000
    write_outputs: bool = True
    overwrite: bool = False
    skip_completed: bool = True


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


def dependency_status(module_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for module_name in module_names:
        try:
            module = __import__(module_name)
            rows.append(
                {
                    "module": module_name,
                    "available": True,
                    "version": getattr(module, "__version__", ""),
                    "error": "",
                }
            )
        except Exception as error:
            rows.append(
                {
                    "module": module_name,
                    "available": False,
                    "version": "",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return pd.DataFrame(rows)


def dataset_search_roots(dataset: object, root: str | Path = PROJECT_ROOT) -> list[Path]:
    root = Path(root).expanduser().resolve()
    roots: list[Path] = []
    dataset_name = str(dataset)
    try:
        config = dataset_config.dataset_configs()[dataset_name]
        for candidate in (config.output_path(root), config.root_path(root)):
            if candidate.is_dir() and candidate not in roots:
                roots.append(candidate)
    except Exception:
        pass
    fallback = root / dataset_name
    if fallback.is_dir() and fallback not in roots:
        roots.append(fallback)
    return roots


def fallback_feature_matrix_search(row: pd.Series, root: str | Path = PROJECT_ROOT) -> list[Path]:
    sample_text = str(row.get("sample", "")).strip()
    tokens = [sample_text] if sample_text else []
    tokens.extend(re.findall(r"GSM\d+", sample_text))
    tokens = list(dict.fromkeys(token for token in tokens if token))
    if not tokens:
        return []
    matches: list[Path] = []
    for search_root in dataset_search_roots(row.get("dataset", ""), root):
        for pattern in ("*cell_feature_matrix.h5", "cell_feature_matrix.zarr.zip"):
            for matrix_path in search_root.rglob(pattern):
                matrix_text = str(matrix_path)
                if any(token in matrix_text for token in tokens) and matrix_path not in matches:
                    matches.append(matrix_path)
    root = Path(root).expanduser().resolve()
    return sorted(matches, key=lambda value: natural_key(path_for_record(value, root)))


def select_feature_matrix(row: pd.Series, root: str | Path = PROJECT_ROOT) -> Path | None:
    candidates = parse_path_list(row.get("feature_matrix_files", ""), root)
    if not candidates:
        sample_dir = resolve_existing_path(row.get("sample_dir", ""), root)
        if sample_dir is not None:
            candidates = sorted(sample_dir.rglob("*cell_feature_matrix.h5"))
            candidates.extend(sorted(sample_dir.rglob("cell_feature_matrix.zarr.zip")))
    if not candidates:
        candidates = fallback_feature_matrix_search(row, root)
    if not candidates:
        return None
    h5_candidates = [path for path in candidates if path.name.endswith(".h5")]
    if h5_candidates:
        return h5_candidates[0]
    zarr_candidates = [path for path in candidates if path.name.endswith(".zarr.zip")]
    if zarr_candidates:
        return zarr_candidates[0]
    return candidates[0]


def discover_samples_from_manifest(root: str | Path = PROJECT_ROOT) -> pd.DataFrame:
    root = Path(root).expanduser().resolve()
    rows: list[dict[str, object]] = []
    for dataset, config in dataset_config.dataset_configs().items():
        search_root = config.output_path(root)
        if not search_root.is_dir():
            search_root = config.root_path(root)
        if not search_root.is_dir():
            continue
        for sample_dir in sorted(search_root.iterdir(), key=lambda value: natural_key(value.name)):
            if not sample_dir.is_dir() or sample_dir.name.startswith("."):
                continue
            matrix = expression_annotation.find_feature_matrix(sample_dir)
            if matrix is None:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "sample": sample_dir.name,
                    "sample_dir": path_for_record(sample_dir, root),
                    "status": "ok",
                    "feature_matrix_files": path_for_record(matrix, root),
                }
            )
    return pd.DataFrame(rows)


def build_clustering_input_table(
    root: str | Path = PROJECT_ROOT,
    *,
    review_dir: str | Path | None = None,
    dataset_names: Iterable[str] | None = None,
    sample_names: Iterable[str] | None = None,
    sample_limit: int | None = None,
    sample_order: str = "smallest_matrix_first",
) -> pd.DataFrame:
    root = Path(root).expanduser().resolve()
    review_dir = Path(review_dir or root / "gene_panel_review").expanduser().resolve()
    sample_summary_path = review_dir / "sample_panel_summary.csv"
    if sample_summary_path.is_file():
        sample_registry = pd.read_csv(sample_summary_path)
        sample_registry = sample_registry.loc[
            sample_registry["status"].astype(str).eq("ok")
        ].copy()
    else:
        sample_registry = discover_samples_from_manifest(root)

    if dataset_names is not None:
        keep = set(map(str, dataset_names))
        sample_registry = sample_registry.loc[
            sample_registry["dataset"].astype(str).isin(keep)
        ].copy()
    if sample_names is not None:
        keep = set(map(str, sample_names))
        sample_registry = sample_registry.loc[
            sample_registry["sample"].astype(str).isin(keep)
        ].copy()

    if sample_registry.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "sample",
                "sample_dir",
                "has_matrix",
                "matrix_type",
                "matrix_size_mb",
                "matrix_path",
            ]
        )

    sample_registry = sample_registry.sort_values(
        ["dataset", "sample"],
        key=lambda column: column.map(natural_key),
        kind="stable",
    ).reset_index(drop=True)
    matrix_paths = [select_feature_matrix(row, root) for _, row in sample_registry.iterrows()]
    sample_registry["matrix_path"] = [
        path_for_record(path, root) if path is not None else "" for path in matrix_paths
    ]
    sample_registry["has_matrix"] = sample_registry["matrix_path"].astype(str).ne("")
    sample_registry["matrix_type"] = np.where(
        sample_registry["matrix_path"].astype(str).str.endswith(".zarr.zip"),
        "zarr_zip",
        np.where(sample_registry["matrix_path"].astype(str).str.endswith(".h5"), "h5", ""),
    )
    sample_registry["matrix_size_mb"] = [
        round(path.stat().st_size / 1_000_000, 3) if path is not None else np.nan
        for path in matrix_paths
    ]

    if sample_order == "smallest_matrix_first":
        sample_registry = sample_registry.sort_values(
            ["has_matrix", "matrix_size_mb"],
            ascending=[False, True],
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)
    elif sample_order == "dataset_sample":
        sample_registry = sample_registry.sort_values(
            ["dataset", "sample"],
            key=lambda column: column.map(natural_key),
            kind="stable",
        ).reset_index(drop=True)
    else:
        raise ValueError("sample_order must be 'smallest_matrix_first' or 'dataset_sample'")

    if sample_limit is not None:
        sample_registry = sample_registry.head(int(sample_limit)).copy()

    columns = [
        "dataset",
        "sample",
        "sample_dir",
        "has_matrix",
        "matrix_type",
        "matrix_size_mb",
        "matrix_path",
    ]
    return sample_registry[[column for column in columns if column in sample_registry.columns]].copy()


def log_step(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def decode_text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def make_unique_names(names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique: list[str] = []
    for name in names:
        text = str(name)
        count = counts.get(text, 0)
        if count == 0:
            unique.append(text)
        else:
            unique.append(f"{text}_{count + 1}")
        counts[text] = count + 1
    return unique


def require_scanpy_stack() -> tuple[object, object]:
    try:
        import anndata as ad
        import scanpy as sc
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Clustering requires scanpy and anndata. Install them before running all-sample clustering."
        ) from error
    return sc, ad


def zarr_feature_metadata_from_attrs(matrix: object, root: object) -> tuple[list[str], list[str]]:
    attrs = dict(root.attrs)
    attrs.update(dict(matrix.attrs))
    names_raw = attrs.get("feature_keys") or attrs.get("feature_names")
    if names_raw is None:
        raise KeyError("cell_feature_matrix.zarr.zip has no feature_keys or feature_names metadata")
    feature_names = [decode_text(value) for value in names_raw]
    types_raw = attrs.get("feature_types")
    if types_raw is None or len(types_raw) != len(feature_names):
        feature_types = ["Gene Expression"] * len(feature_names)
    else:
        feature_types = [decode_text(value) for value in types_raw]
    return feature_names, feature_types


def load_h5_gene_expression_adata(matrix_path: Path):
    _sc, ad = require_scanpy_stack()
    try:
        import h5py
        from scipy import sparse
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("Reading cell_feature_matrix.h5 requires h5py and scipy.") from error

    with h5py.File(matrix_path, "r") as handle:
        group = handle["matrix"]
        data = group["data"][:]
        indices = group["indices"][:]
        indptr = group["indptr"][:]
        shape = tuple(int(value) for value in group["shape"][:])
        features = group["features"]
        feature_key = "name" if "name" in features else "id"
        feature_names = [decode_text(value) for value in features[feature_key][:]]
        if "feature_type" in features:
            feature_types = [decode_text(value) for value in features["feature_type"][:]]
        else:
            feature_types = ["Gene Expression"] * len(feature_names)
        cell_ids = [decode_text(value) for value in group["barcodes"][:]]

    gene_by_cell = sparse.csc_matrix((data, indices, indptr), shape=shape)
    cell_by_gene = gene_by_cell.T.tocsr()
    gene_columns = [
        index
        for index, (name, feature_type) in enumerate(zip(feature_names, feature_types))
        if is_gene_expression(feature_type, name)
    ]
    if not gene_columns:
        raise ValueError(f"No gene-expression features found in {matrix_path}")

    selected_gene_names = [feature_names[index] for index in gene_columns]
    selected_feature_types = [feature_types[index] for index in gene_columns]
    adata = ad.AnnData(X=cell_by_gene[:, gene_columns].tocsr())
    adata.obs_names = pd.Index(cell_ids).astype(str)
    adata.var_names = make_unique_names(selected_gene_names)
    adata.var["gene_symbol"] = selected_gene_names
    adata.var["feature_type"] = selected_feature_types
    adata.var["matrix_feature_index"] = gene_columns
    adata.uns["matrix_path"] = str(matrix_path)
    adata.uns["matrix_type"] = "h5"
    return adata


def load_zarr_gene_expression_adata(matrix_path: Path):
    _sc, ad = require_scanpy_stack()
    try:
        import zarr
        from scipy import sparse
        from zarr.storage import ZipStore
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Reading cell_feature_matrix.zarr.zip requires zarr and scipy."
        ) from error

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

    with ZipStore(str(matrix_path), mode="r") as store:
        root = zarr.open_group(store=store, mode="r")
        matrix = root["cell_features"] if "cell_features" in root else root
        data = np.asarray(matrix["data"][:])
        indices = np.asarray(matrix["indices"][:])
        indptr = np.asarray(matrix["indptr"][:])
        feature_names, feature_types = zarr_feature_metadata_from_attrs(matrix, root)
        number_features = len(feature_names)
        number_cells = int(matrix.attrs.get("number_cells", root.attrs.get("number_cells", len(cell_ids))))
        if number_cells != len(cell_ids):
            raise ValueError(
                f"Cell-id table has {len(cell_ids):,} cells but matrix reports {number_cells:,}; row order cannot be assumed."
            )

    if len(indptr) == number_features + 1:
        feature_by_cell = sparse.csr_matrix(
            (data, indices, indptr),
            shape=(number_features, number_cells),
        )
        cell_by_gene = feature_by_cell.T.tocsr()
        sparse_orientation = "feature_by_cell_csr"
    elif len(indptr) == number_cells + 1:
        cell_by_gene = sparse.csr_matrix(
            (data, indices, indptr),
            shape=(number_cells, number_features),
        ).tocsr()
        sparse_orientation = "cell_by_feature_csr"
    else:
        raise ValueError(
            "Unsupported Zarr sparse orientation: "
            f"indptr has length {len(indptr):,}, but expected number_features + 1 "
            f"({number_features + 1:,}) or number_cells + 1 ({number_cells + 1:,})."
        )

    gene_columns = [
        index
        for index, (name, feature_type) in enumerate(zip(feature_names, feature_types))
        if is_gene_expression(feature_type, name)
    ]
    if not gene_columns:
        raise ValueError(f"No gene-expression features found in {matrix_path}")

    selected_gene_names = [feature_names[index] for index in gene_columns]
    selected_feature_types = [feature_types[index] for index in gene_columns]
    adata = ad.AnnData(X=cell_by_gene[:, gene_columns].tocsr())
    adata.obs_names = pd.Index(cell_ids).astype(str)
    adata.var_names = make_unique_names(selected_gene_names)
    adata.var["gene_symbol"] = selected_gene_names
    adata.var["feature_type"] = selected_feature_types
    adata.var["matrix_feature_index"] = gene_columns
    adata.uns["matrix_path"] = str(matrix_path)
    adata.uns["matrix_type"] = "zarr_zip"
    adata.uns["zarr_sparse_orientation"] = sparse_orientation
    return adata


def load_feature_matrix_as_adata(matrix_path: Path):
    if matrix_path.name.endswith(".h5"):
        return load_h5_gene_expression_adata(matrix_path)
    if matrix_path.name.endswith(".zarr.zip"):
        return load_zarr_gene_expression_adata(matrix_path)
    raise ValueError(f"Unsupported feature matrix: {matrix_path}")


def cluster_key_for_resolution(resolution: float) -> str:
    text = f"{float(resolution):g}".replace(".", "_")
    return f"leiden_r{text}"


def calculate_qc_metrics(adata) -> None:
    from scipy import sparse

    counts = adata.X
    adata.obs["total_counts"] = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    if sparse.issparse(counts):
        adata.obs["n_genes_by_counts"] = np.diff(counts.tocsr().indptr).astype(int)
    else:
        adata.obs["n_genes_by_counts"] = (counts > 0).sum(axis=1).astype(int)


def filter_adata_for_clustering(adata, config: ClusteringConfig):
    from scipy import sparse

    calculate_qc_metrics(adata)
    cell_mask = (
        (adata.obs["total_counts"].to_numpy(dtype=float) >= config.min_counts_per_cell)
        & (adata.obs["n_genes_by_counts"].to_numpy(dtype=float) >= config.min_genes_per_cell)
    )
    if sparse.issparse(adata.X):
        gene_detected = np.asarray((adata.X > 0).sum(axis=0)).ravel()
    else:
        gene_detected = (adata.X > 0).sum(axis=0)
    gene_mask = gene_detected >= config.min_cells_per_gene
    filtered = adata[cell_mask, gene_mask].copy()
    filtered.uns["n_cells_before_filter"] = int(adata.n_obs)
    filtered.uns["n_genes_before_filter"] = int(adata.n_vars)
    filtered.uns["n_cells_after_filter"] = int(filtered.n_obs)
    filtered.uns["n_genes_after_filter"] = int(filtered.n_vars)
    return filtered


def optionally_subsample_cells(adata, dataset: str, sample: str, config: ClusteringConfig):
    if config.max_cells_per_sample_for_clustering is None:
        adata.obs["used_for_clustering"] = True
        return adata
    max_cells = int(config.max_cells_per_sample_for_clustering)
    if adata.n_obs <= max_cells:
        adata.obs["used_for_clustering"] = True
        return adata
    sc, _ad = require_scanpy_stack()
    sampled = sc.pp.subsample(adata, n_obs=max_cells, random_state=config.random_state, copy=True)
    sampled.obs["used_for_clustering"] = True
    log_step(f"{dataset} / {sample}: subsampled from {adata.n_obs:,} to {sampled.n_obs:,} cells for clustering.")
    return sampled


def preprocess_and_cluster(adata, dataset: str, sample: str, config: ClusteringConfig):
    sc, _ad = require_scanpy_stack()
    log_step(f"{dataset} / {sample}: filtering cells and genes")
    adata = filter_adata_for_clustering(adata, config)
    log_step(f"{dataset} / {sample}: retained {adata.n_obs:,} cells and {adata.n_vars:,} genes after filtering")
    if adata.n_obs < max(3, config.n_neighbors + 1):
        raise ValueError(f"Too few cells after filtering for clustering: {adata.n_obs:,}")
    if adata.n_vars < 2:
        raise ValueError(f"Too few genes after filtering for clustering: {adata.n_vars:,}")

    adata = optionally_subsample_cells(adata, dataset, sample, config)
    calculate_qc_metrics(adata)

    log_step(f"{dataset} / {sample}: normalizing to {config.normalization_target:g} counts per cell and applying log1p")
    sc.pp.normalize_total(adata, target_sum=config.normalization_target)
    sc.pp.log1p(adata)

    if config.use_highly_variable_genes and adata.n_vars > config.n_highly_variable_genes:
        log_step(f"{dataset} / {sample}: selecting highly variable genes")
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=min(config.n_highly_variable_genes, adata.n_vars),
            flavor="seurat",
        )
        n_hvg = int(adata.var["highly_variable"].sum())
        if n_hvg >= 2:
            adata = adata[:, adata.var["highly_variable"]].copy()
            log_step(f"{dataset} / {sample}: using {n_hvg:,} highly variable genes for PCA")

    n_pcs = min(int(config.n_pcs), adata.n_obs - 1, adata.n_vars - 1)
    if n_pcs < 2:
        raise ValueError(f"Cannot compute at least 2 PCs: n_cells={adata.n_obs:,}, n_genes={adata.n_vars:,}")
    log_step(f"{dataset} / {sample}: computing PCA with {n_pcs:,} PCs")
    sc.pp.pca(adata, n_comps=n_pcs, svd_solver="arpack", random_state=config.random_state)
    n_neighbors = min(int(config.n_neighbors), adata.n_obs - 1)
    log_step(f"{dataset} / {sample}: building kNN graph with n_neighbors={n_neighbors:,}")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=config.random_state)
    if config.run_umap:
        log_step(f"{dataset} / {sample}: computing UMAP")
        sc.tl.umap(adata, random_state=config.random_state)
    else:
        log_step(f"{dataset} / {sample}: skipping UMAP")

    for resolution in config.leiden_resolutions:
        key = cluster_key_for_resolution(float(resolution))
        log_step(f"{dataset} / {sample}: Leiden clustering at resolution {float(resolution):g}")
        sc.tl.leiden(
            adata,
            resolution=float(resolution),
            key_added=key,
            random_state=config.random_state,
        )
    adata.uns["n_pcs_used"] = int(n_pcs)
    return adata


def cluster_assignments_table(adata, dataset: str, sample: str, config: ClusteringConfig) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "dataset": dataset,
            "sample": sample,
            "cell_id": adata.obs_names.astype(str),
            "total_counts": pd.to_numeric(adata.obs.get("total_counts", np.nan), errors="coerce"),
            "n_genes_by_counts": pd.to_numeric(adata.obs.get("n_genes_by_counts", np.nan), errors="coerce"),
        }
    )
    if "X_umap" in adata.obsm:
        table["umap_1"] = adata.obsm["X_umap"][:, 0]
        table["umap_2"] = adata.obsm["X_umap"][:, 1]
    for resolution in config.leiden_resolutions:
        key = cluster_key_for_resolution(float(resolution))
        if key in adata.obs:
            table[key] = adata.obs[key].astype(str).to_numpy()
    return table


def cluster_summary_table(clusters: pd.DataFrame, config: ClusteringConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for resolution in config.leiden_resolutions:
        key = cluster_key_for_resolution(float(resolution))
        if key not in clusters:
            continue
        counts = clusters[key].astype(str).value_counts()
        for cluster, n_cells in counts.sort_index(key=lambda index: index.map(natural_key)).items():
            rows.append(
                {
                    "dataset": clusters["dataset"].iloc[0],
                    "sample": clusters["sample"].iloc[0],
                    "resolution": float(resolution),
                    "cluster_key": key,
                    "cluster": str(cluster),
                    "n_cells": int(n_cells),
                    "fraction": float(n_cells / len(clusters)) if len(clusters) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def pca_variance_table(adata, dataset: str, sample: str) -> pd.DataFrame:
    variance_ratio = np.asarray(adata.uns.get("pca", {}).get("variance_ratio", []), dtype=float)
    if variance_ratio.size == 0:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "dataset": dataset,
            "sample": sample,
            "pc": np.arange(1, len(variance_ratio) + 1),
            "variance_ratio": variance_ratio,
            "cumulative_variance_ratio": np.cumsum(variance_ratio),
        }
    )


def rank_cluster_marker_genes(
    adata,
    dataset: str,
    sample: str,
    cluster_key: str,
    config: ClusteringConfig,
) -> pd.DataFrame:
    if not config.run_cluster_markers:
        return pd.DataFrame()
    sc, _ad = require_scanpy_stack()
    marker_adata = adata
    if config.max_cells_for_marker_test is not None and marker_adata.n_obs > int(config.max_cells_for_marker_test):
        marker_adata = sc.pp.subsample(
            marker_adata,
            n_obs=int(config.max_cells_for_marker_test),
            random_state=config.random_state,
            copy=True,
        )
    sc.tl.rank_genes_groups(
        marker_adata,
        groupby=cluster_key,
        method="wilcoxon",
        n_genes=int(config.n_marker_genes_per_cluster),
    )
    try:
        table = sc.get.rank_genes_groups_df(marker_adata, group=None)
    except Exception:
        return pd.DataFrame()
    table.insert(0, "dataset", dataset)
    table.insert(1, "sample", sample)
    table.insert(2, "cluster_key", cluster_key)
    return table


def plot_cluster_umap(
    clusters: pd.DataFrame,
    cluster_key: str,
    output_path: Path,
    config: ClusteringConfig,
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    required = {"umap_1", "umap_2", cluster_key}
    if clusters.empty or not required.issubset(clusters.columns):
        return None
    plot_table = clusters.copy()
    if config.max_umap_plot_cells is not None and len(plot_table) > int(config.max_umap_plot_cells):
        plot_table = plot_table.sample(n=int(config.max_umap_plot_cells), random_state=config.random_state)
    labels = sorted(plot_table[cluster_key].astype(str).unique(), key=natural_key)
    label_to_code = {label: index for index, label in enumerate(labels)}
    colors = plot_table[cluster_key].astype(str).map(label_to_code).to_numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        plot_table["umap_1"],
        plot_table["umap_2"],
        c=colors,
        s=float(config.umap_point_size),
        cmap="tab20",
        linewidths=0,
        alpha=0.8,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"{clusters['dataset'].iloc[0]} / {clusters['sample'].iloc[0]}\n{cluster_key}")
    if len(labels) <= 30:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=scatter.cmap(scatter.norm(label_to_code[label])),
                markersize=5,
                label=label,
            )
            for label in labels
        ]
        ax.legend(handles=handles, title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def ensure_can_write(paths: list[Path], *, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "Output file(s) already exist and overwrite is False: " + formatted
        )


def _write_run_summary_checkpoint(output_dir: Path, rows: list[dict[str, object]], write_outputs: bool) -> None:
    if write_outputs and rows:
        pd.DataFrame(rows).to_csv(output_dir / "clustering_run_summary.csv", index=False)


def run_cell_clustering(
    root: str | Path = PROJECT_ROOT,
    *,
    review_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    dataset_names: Iterable[str] | None = None,
    sample_names: Iterable[str] | None = None,
    sample_limit: int | None = None,
    sample_order: str = "smallest_matrix_first",
    config: ClusteringConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Run per-sample expression clustering for selected samples."""
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir or root / "cell_clustering").expanduser().resolve()
    figure_dir = output_dir / "figures"
    config = config or ClusteringConfig()
    if config.write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)

    clustering_input_table = build_clustering_input_table(
        root,
        review_dir=review_dir,
        dataset_names=dataset_names,
        sample_names=sample_names,
        sample_limit=sample_limit,
        sample_order=sample_order,
    )
    if config.write_outputs:
        clustering_input_table.to_csv(output_dir / "clustering_input_table.csv", index=False)

    run_rows: list[dict[str, object]] = []
    all_cluster_summaries: list[pd.DataFrame] = []
    all_pca_variance_tables: list[pd.DataFrame] = []
    all_marker_gene_tables: list[pd.DataFrame] = []
    umap_figure_rows: list[dict[str, object]] = []

    default_cluster_key = cluster_key_for_resolution(float(config.default_resolution))

    for row_index, row in enumerate(clustering_input_table.itertuples(index=False), start=1):
        dataset = str(row.dataset)
        sample = str(row.sample)
        matrix_path = resolve_existing_path(row.matrix_path, root)
        sample_output_dir = output_dir / safe_name(dataset) / safe_name(sample)
        clusters_path = sample_output_dir / "cell_clusters.csv.gz"
        cluster_summary_path = sample_output_dir / "cluster_summary.csv"
        pca_variance_path = sample_output_dir / "pca_variance.csv"
        marker_genes_path = sample_output_dir / "cluster_marker_genes.csv"
        umap_path = figure_dir / f"umap__{safe_name(dataset)}__{safe_name(sample)}__{default_cluster_key}.png"
        if config.write_outputs and matrix_path is not None:
            sample_output_dir.mkdir(parents=True, exist_ok=True)

        sample_start = perf_counter()
        log_step(f"[{row_index:,}/{len(clustering_input_table):,}] Clustering {dataset} / {sample}")
        record: dict[str, object] = {
            "dataset": dataset,
            "sample": sample,
            "status": "not_run",
            "matrix_path": path_for_record(matrix_path, root) if matrix_path else "",
            "matrix_type": getattr(row, "matrix_type", ""),
            "matrix_size_mb": getattr(row, "matrix_size_mb", np.nan),
            "n_cells_loaded": 0,
            "n_genes_loaded": 0,
            "n_cells_clustered": 0,
            "n_genes_clustered": 0,
            "n_pcs_used": np.nan,
            "default_cluster_key": default_cluster_key,
            "n_default_clusters": np.nan,
            "cell_clusters_path": "",
            "cluster_summary_path": "",
            "pca_variance_path": "",
            "cluster_marker_genes_path": "",
            "umap_figure_path": "",
            "elapsed_seconds": np.nan,
            "error": "",
        }

        if matrix_path is None:
            record.update({"status": "missing_matrix", "error": "No selected feature matrix was found."})
            record["elapsed_seconds"] = round(perf_counter() - sample_start, 2)
            run_rows.append(record)
            _write_run_summary_checkpoint(output_dir, run_rows, config.write_outputs)
            continue

        if config.skip_completed and clusters_path.is_file() and cluster_summary_path.is_file():
            log_step(f"{dataset} / {sample}: existing clustering output found; skipping")
            try:
                existing_summary = pd.read_csv(cluster_summary_path)
                all_cluster_summaries.append(existing_summary)
                default_rows = existing_summary.loc[
                    existing_summary["cluster_key"].astype(str).eq(default_cluster_key)
                ]
                if not default_rows.empty:
                    record["n_cells_clustered"] = int(default_rows["n_cells"].sum())
                    record["n_default_clusters"] = int(default_rows["cluster"].nunique())
                if pca_variance_path.is_file():
                    all_pca_variance_tables.append(pd.read_csv(pca_variance_path))
                if marker_genes_path.is_file():
                    marker_genes = pd.read_csv(marker_genes_path)
                    if not marker_genes.empty:
                        all_marker_gene_tables.append(marker_genes)
                if umap_path.is_file():
                    umap_figure_rows.append(
                        {
                            "dataset": dataset,
                            "sample": sample,
                            "cluster_key": default_cluster_key,
                            "figure_path": path_for_record(umap_path, root),
                            "status": "existing",
                        }
                    )
            except Exception as error:
                record["error"] = f"Could not read existing summary: {type(error).__name__}: {error}"
            record.update(
                {
                    "status": "skipped_existing",
                    "cell_clusters_path": path_for_record(clusters_path, root),
                    "cluster_summary_path": path_for_record(cluster_summary_path, root),
                    "pca_variance_path": path_for_record(pca_variance_path, root) if pca_variance_path.is_file() else "",
                    "cluster_marker_genes_path": path_for_record(marker_genes_path, root) if marker_genes_path.is_file() else "",
                    "umap_figure_path": path_for_record(umap_path, root) if umap_path.is_file() else "",
                    "elapsed_seconds": round(perf_counter() - sample_start, 2),
                }
            )
            run_rows.append(record)
            _write_run_summary_checkpoint(output_dir, run_rows, config.write_outputs)
            continue

        try:
            log_step(f"{dataset} / {sample}: loading feature matrix {matrix_path.name}")
            adata = load_feature_matrix_as_adata(matrix_path)
            record["n_cells_loaded"] = int(adata.n_obs)
            record["n_genes_loaded"] = int(adata.n_vars)
            log_step(f"{dataset} / {sample}: loaded {adata.n_obs:,} cells x {adata.n_vars:,} genes")

            clustered = preprocess_and_cluster(adata, dataset, sample, config)
            log_step(f"{dataset} / {sample}: building output tables")
            clusters = cluster_assignments_table(clustered, dataset, sample, config)
            cluster_summary = cluster_summary_table(clusters, config)
            pca_variance = pca_variance_table(clustered, dataset, sample)
            marker_genes = rank_cluster_marker_genes(clustered, dataset, sample, default_cluster_key, config)

            output_paths = [clusters_path, cluster_summary_path, pca_variance_path]
            if config.run_umap:
                output_paths.append(umap_path)
            if config.run_cluster_markers:
                output_paths.append(marker_genes_path)
            if config.write_outputs:
                ensure_can_write(output_paths, overwrite=config.overwrite)
                log_step(f"{dataset} / {sample}: writing per-sample outputs")
                clusters.to_csv(clusters_path, index=False, compression="gzip")
                cluster_summary.to_csv(cluster_summary_path, index=False)
                pca_variance.to_csv(pca_variance_path, index=False)
                if config.run_cluster_markers:
                    marker_genes.to_csv(marker_genes_path, index=False)

            saved_umap_path = (
                plot_cluster_umap(clusters, default_cluster_key, umap_path, config)
                if config.write_outputs and config.run_umap
                else None
            )

            all_cluster_summaries.append(cluster_summary)
            all_pca_variance_tables.append(pca_variance)
            if config.run_cluster_markers and not marker_genes.empty:
                all_marker_gene_tables.append(marker_genes)
            if saved_umap_path is not None:
                umap_figure_rows.append(
                    {
                        "dataset": dataset,
                        "sample": sample,
                        "cluster_key": default_cluster_key,
                        "figure_path": path_for_record(saved_umap_path, root),
                        "status": "created",
                    }
                )

            record.update(
                {
                    "status": "success",
                    "n_cells_clustered": int(len(clusters)),
                    "n_genes_clustered": int(clustered.n_vars),
                    "n_pcs_used": int(clustered.uns.get("n_pcs_used", np.nan)),
                    "n_default_clusters": int(clusters[default_cluster_key].nunique()) if default_cluster_key in clusters else np.nan,
                    "cell_clusters_path": path_for_record(clusters_path, root),
                    "cluster_summary_path": path_for_record(cluster_summary_path, root),
                    "pca_variance_path": path_for_record(pca_variance_path, root),
                    "cluster_marker_genes_path": path_for_record(marker_genes_path, root) if config.run_cluster_markers else "",
                    "umap_figure_path": path_for_record(saved_umap_path, root) if saved_umap_path is not None else "",
                    "elapsed_seconds": round(perf_counter() - sample_start, 2),
                }
            )
            log_step(f"{dataset} / {sample}: finished in {record['elapsed_seconds']:.2f} seconds")
        except Exception as error:
            record.update(
                {
                    "status": "failed",
                    "elapsed_seconds": round(perf_counter() - sample_start, 2),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            traceback.print_exc(limit=2)
        finally:
            for name in ("adata", "clustered"):
                if name in locals():
                    del locals()[name]
            gc.collect()

        run_rows.append(record)
        _write_run_summary_checkpoint(output_dir, run_rows, config.write_outputs)

    clustering_run_summary = pd.DataFrame(run_rows)
    combined_cluster_summary = (
        pd.concat(all_cluster_summaries, ignore_index=True)
        if all_cluster_summaries
        else pd.DataFrame()
    )
    combined_pca_variance = (
        pd.concat(all_pca_variance_tables, ignore_index=True)
        if all_pca_variance_tables
        else pd.DataFrame()
    )
    combined_cluster_marker_genes = (
        pd.concat(all_marker_gene_tables, ignore_index=True)
        if all_marker_gene_tables
        else pd.DataFrame()
    )
    umap_figure_index = pd.DataFrame(umap_figure_rows)

    if config.write_outputs:
        clustering_run_summary.to_csv(output_dir / "clustering_run_summary.csv", index=False)
        combined_cluster_summary.to_csv(output_dir / "combined_cluster_summary.csv", index=False)
        combined_pca_variance.to_csv(output_dir / "combined_pca_variance.csv", index=False)
        combined_cluster_marker_genes.to_csv(output_dir / "combined_cluster_marker_genes.csv", index=False)
        umap_figure_index.to_csv(output_dir / "umap_figure_index.csv", index=False)

    return {
        "clustering_input_table": clustering_input_table,
        "clustering_run_summary": clustering_run_summary,
        "combined_cluster_summary": combined_cluster_summary,
        "combined_pca_variance": combined_pca_variance,
        "combined_cluster_marker_genes": combined_cluster_marker_genes,
        "umap_figure_index": umap_figure_index,
    }


def _parse_resolutions(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in re.split(r"[, ]+", value.strip()) if part)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster selected Xenium samples from expression matrices."
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--review-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--samples", nargs="+")
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument(
        "--sample-order",
        choices=("smallest_matrix_first", "dataset_sample"),
        default="smallest_matrix_first",
    )
    parser.add_argument("--normalization-target", type=float, default=10_000.0)
    parser.add_argument("--min-counts-per-cell", type=float, default=10.0)
    parser.add_argument("--min-genes-per-cell", type=int, default=5)
    parser.add_argument("--min-cells-per-gene", type=int, default=10)
    parser.add_argument("--use-highly-variable-genes", action="store_true")
    parser.add_argument("--n-highly-variable-genes", type=int, default=2_000)
    parser.add_argument("--n-pcs", type=int, default=20)
    parser.add_argument("--n-neighbors", type=int, default=20)
    parser.add_argument("--leiden-resolutions", default="1.3")
    parser.add_argument("--default-resolution", type=float, default=1.3)
    parser.add_argument("--random-state", type=int, default=1729)
    parser.add_argument("--max-cells-per-sample-for-clustering", type=int, default=None)
    parser.add_argument("--full-sample-clustering", action="store_true")
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument("--max-umap-plot-cells", type=int, default=50_000)
    parser.add_argument("--umap-point-size", type=float, default=1.0)
    parser.add_argument("--run-cluster-markers", action="store_true")
    parser.add_argument("--n-marker-genes-per-cluster", type=int, default=20)
    parser.add_argument("--max-cells-for-marker-test", type=int, default=100_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-skip-completed", action="store_true")
    parser.add_argument("--no-write-outputs", action="store_true")
    parser.add_argument("--check-dependencies", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check_dependencies:
        status = dependency_status(["scanpy", "anndata", "scipy", "h5py", "matplotlib", "zarr", "leidenalg", "igraph"])
        print(status)
        if status.loc[status["module"].isin(["scanpy", "anndata", "scipy", "h5py", "matplotlib"]) & ~status["available"]].any().any():
            return 1
    config = ClusteringConfig(
        normalization_target=args.normalization_target,
        min_counts_per_cell=args.min_counts_per_cell,
        min_genes_per_cell=args.min_genes_per_cell,
        min_cells_per_gene=args.min_cells_per_gene,
        use_highly_variable_genes=args.use_highly_variable_genes,
        n_highly_variable_genes=args.n_highly_variable_genes,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        leiden_resolutions=_parse_resolutions(args.leiden_resolutions),
        default_resolution=args.default_resolution,
        random_state=args.random_state,
        max_cells_per_sample_for_clustering=(
            None if args.full_sample_clustering else args.max_cells_per_sample_for_clustering
        ),
        run_umap=not args.skip_umap,
        max_umap_plot_cells=args.max_umap_plot_cells,
        umap_point_size=args.umap_point_size,
        run_cluster_markers=args.run_cluster_markers,
        n_marker_genes_per_cluster=args.n_marker_genes_per_cluster,
        max_cells_for_marker_test=args.max_cells_for_marker_test,
        write_outputs=not args.no_write_outputs,
        overwrite=args.overwrite,
        skip_completed=not args.no_skip_completed,
    )
    results = run_cell_clustering(
        args.root,
        review_dir=args.review_dir,
        output_dir=args.output_dir,
        dataset_names=args.datasets,
        sample_names=args.samples,
        sample_limit=args.sample_limit,
        sample_order=args.sample_order,
        config=config,
    )
    summary = results["clustering_run_summary"]
    if not summary.empty:
        print(summary[["dataset", "sample", "status", "n_cells_clustered", "n_default_clusters", "error"]])
    return int(summary["status"].eq("failed").any()) if not summary.empty else 0


if __name__ == "__main__":
    raise SystemExit(main())
