"""Panel-aware UCell evidence for expression clusters."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
from . import cluster_review
from .cell_clustering import load_feature_matrix_as_adata
from .cell_identity_evidence import collapse_label
from .gene_panel_utils import natural_key
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _split_values(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [part.strip() for part in str(value).replace('|', ';').split(';') if part.strip()]

def _sample_key(dataset: object, sample: object) -> str:
    return f'{dataset} | {sample}'

def _normalize_sample_selection(values: Iterable[str | tuple[str, str]] | None) -> set[str] | None:
    if values is None:
        return None
    selected: set[str] = set()
    for value in values:
        if isinstance(value, tuple) and len(value) == 2:
            selected.add(_sample_key(value[0], value[1]))
        else:
            selected.add(str(value).strip())
    return selected

def signature_evidence_paths(output_dir: str | Path) -> dict[str, Path]:
    folder = Path(output_dir) / 'signature_scores'
    return {'run_summary': folder / 'signature_scores_run_summary.csv', 'coverage': folder / 'signature_scores_signature_coverage.csv', 'cluster_summary': folder / 'combined_cluster_signature_scores_summary.csv', 'top_two': folder / 'combined_cluster_signature_scores_top_two.csv'}

def _known_marker_genes(hint_sources: dict[str, pd.DataFrame]) -> list[str]:
    genes: set[str] = set()
    for table_name in ('marker_gene_table', 'annotation_conflicts'):
        table = hint_sources.get(table_name, pd.DataFrame())
        if not table.empty and 'gene' in table.columns:
            genes.update(table['gene'].dropna().astype(str).str.upper().str.strip().tolist())
    return sorted((gene for gene in genes if gene), key=natural_key)

def _signatures_from_hints(hints: pd.DataFrame, *, broad: bool) -> dict[str, set[str]]:
    signatures: dict[str, set[str]] = {}
    for row in hints.itertuples(index=False):
        groups = _split_values(getattr(row, 'gene_marker_groups', ''))
        if len(groups) > 1:
            raise ValueError(f"Gene {getattr(row, 'gene', '')} has multiple refined assignments: " + '; '.join(groups))
        if not groups:
            continue
        label = groups[0]
        if broad:
            label = collapse_label(label)[0]
            if label.startswith('Other /'):
                continue
        gene = str(getattr(row, 'gene', '')).upper().strip()
        if gene:
            signatures.setdefault(label, set()).add(gene)
    return signatures

def build_sample_signature_sets(dataset: str, sample: str, measured_genes: Iterable[object], *, hint_sources: dict[str, pd.DataFrame], min_signature_genes: int=3) -> dict[str, object]:
    """Build panel-aware refined and broad positive-marker signatures."""
    measured = {str(gene).upper().strip() for gene in measured_genes if str(gene).strip()}
    known_genes = _known_marker_genes(hint_sources)
    reference_hints = cluster_review._make_gene_marker_hint_table_for_sample(str(dataset), str(sample), pd.Series(known_genes), hint_sources)
    reference = {'refined': _signatures_from_hints(reference_hints, broad=False), 'broad': _signatures_from_hints(reference_hints, broad=True)}
    eligible: dict[str, dict[str, list[str]]] = {'refined': {}, 'broad': {}}
    coverage_rows: list[dict[str, object]] = []
    for granularity in ('refined', 'broad'):
        for cell_type, reference_genes in reference[granularity].items():
            measured_signature = sorted(reference_genes & measured, key=natural_key)
            n_reference = len(reference_genes)
            n_measured = len(measured_signature)
            is_eligible = n_measured >= int(min_signature_genes)
            if is_eligible:
                eligible[granularity][cell_type] = measured_signature
            coverage_rows.append({'dataset': str(dataset), 'sample': str(sample), 'granularity': granularity, 'cell_type': cell_type, 'n_reference_signature_genes': int(n_reference), 'n_measured_signature_genes': int(n_measured), 'signature_coverage': n_measured / n_reference if n_reference else np.nan, 'eligible_for_primary_analysis': bool(is_eligible), 'measured_signature_genes': '; '.join(measured_signature), 'reference_signature_genes': '; '.join(sorted(reference_genes, key=natural_key)), 'minimum_signature_genes': int(min_signature_genes)})
    return {'signatures': eligible, 'coverage': pd.DataFrame(coverage_rows), 'reference_signatures': reference}

def _gene_symbol_index(adata: Any) -> tuple[list[str], dict[str, int]]:
    symbols = adata.var['gene_symbol'].astype(str).tolist() if 'gene_symbol' in adata.var.columns else adata.var_names.astype(str).tolist()
    lookup: dict[str, int] = {}
    for index, symbol in enumerate(symbols):
        lookup.setdefault(str(symbol).upper().strip(), index)
    return (symbols, lookup)

def compute_panel_aware_ucell_scores(adata: Any, signatures: dict[str, list[str]], *, max_rank: int | None=None, chunk_size: int=5000) -> tuple[pd.DataFrame, dict[str, object]]:
    """Calculate current UCell scores from raw counts in bounded chunks."""
    try:
        from scipy import sparse
        from scipy.stats import rankdata
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError('UCell scoring requires scipy.') from error
    if not signatures:
        raise ValueError('No adequately covered signatures are available')
    matrix = adata.X.tocsr() if sparse.issparse(adata.X) else np.asarray(adata.X)
    _symbols, lookup = _gene_symbol_index(adata)
    indices: dict[str, np.ndarray] = {}
    for name, genes in signatures.items():
        selected = sorted({lookup[gene.upper()] for gene in genes if gene.upper() in lookup})
        if selected:
            indices[name] = np.asarray(selected, dtype=int)
    if not indices:
        raise ValueError('None of the requested signature genes are present in the matrix')
    if sparse.issparse(matrix):
        detected_per_cell = np.diff(matrix.indptr)
    else:
        detected_per_cell = np.count_nonzero(matrix, axis=1)
    median_detected = int(max(1, round(float(np.median(detected_per_cell)))))
    largest_signature = max((len(value) for value in indices.values()))
    automatic_max_rank = min(int(adata.n_vars), max(2, median_detected, largest_signature))
    if max_rank is None:
        effective_max_rank = automatic_max_rank
    else:
        effective_max_rank = min(int(adata.n_vars), int(max_rank))
        if effective_max_rank < largest_signature:
            raise ValueError(f'max_rank={effective_max_rank} is smaller than the largest measured signature ({largest_signature} genes)')
    if effective_max_rank < 2:
        raise ValueError('UCell max_rank must be at least 2')
    arrays = {name: np.empty(int(adata.n_obs), dtype=np.float32) for name in indices}
    step = max(1, int(chunk_size))
    for start in range(0, int(adata.n_obs), step):
        stop = min(start + step, int(adata.n_obs))
        block = matrix[start:stop]
        dense = block.toarray() if sparse.issparse(block) else np.asarray(block)
        ranks = rankdata(-dense, axis=1, method='average')
        ranks = np.minimum(ranks, effective_max_rank)
        for name, gene_indices in indices.items():
            number_genes = len(gene_indices)
            minimum_rank_sum = number_genes * (number_genes + 1) / 2
            maximum_u = number_genes * effective_max_rank - minimum_rank_sum
            if maximum_u <= 0:
                raise ValueError(f'UCell normalization is undefined for {name}: {number_genes} genes and max_rank={effective_max_rank}')
            u_statistic = ranks[:, gene_indices].sum(axis=1) - minimum_rank_sum
            arrays[name][start:stop] = np.clip(1.0 - u_statistic / maximum_u, 0.0, 1.0).astype(np.float32)
    scores = pd.DataFrame(arrays, index=pd.Index(adata.obs_names.astype(str)))
    metadata = {'n_cells_scored': int(adata.n_obs), 'n_measured_genes': int(adata.n_vars), 'n_signatures_scored': int(len(scores.columns)), 'median_detected_genes_per_cell': int(median_detected), 'largest_measured_signature': int(largest_signature), 'max_rank': int(effective_max_rank), 'chunk_size': int(step), 'implementation': 'native_current_ucell_formula'}
    return (scores, metadata)

def _load_partition_assignments(source_row: pd.Series, matrix_cell_ids: pd.Index, *, root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    if str(source_row.get('clustering_source', '')) == 'xoa':
        assignments, _path, diagnostics = cluster_review.load_cluster_source_assignments_for_matrix(source_row, matrix_cell_ids, root=root)
    else:
        assignments, _path = cluster_review.load_cluster_source_assignments(source_row, root=root)
        diagnostics = {'cell_id_match_method': 'exact'}
    cluster_key = str(source_row['cluster_key'])
    return (cluster_review.informative_cluster_assignments(assignments, cluster_key), diagnostics)

def summarize_cell_type_scores_by_cluster(scores: pd.DataFrame, assignments: pd.DataFrame, *, dataset: str, sample: str, clustering_source: str, cluster_key: str, method: str, granularity: str, coverage: pd.DataFrame, rank_statistic: str) -> pd.DataFrame:
    """Summarize a cell-by-type score matrix within one cluster partition."""
    if rank_statistic not in {'mean', 'median'}:
        raise ValueError('rank_statistic must be mean or median')
    assignment_table = assignments[['cell_id', cluster_key]].copy()
    assignment_table['cell_id'] = assignment_table['cell_id'].astype(str)
    joined = assignment_table.set_index('cell_id').join(scores, how='inner')
    if joined.empty:
        raise ValueError('No cluster-assignment IDs matched the score matrix')
    type_columns = list(scores.columns)
    row_maximum = joined[type_columns].max(axis=1)
    positive_maximum = row_maximum.gt(0)
    top_membership = joined[type_columns].eq(row_maximum, axis=0).mul(positive_maximum, axis=0)
    top_counts = top_membership.groupby(joined[cluster_key], dropna=False).sum()
    positive_counts = positive_maximum.groupby(joined[cluster_key], dropna=False).sum()
    cluster_sizes = joined.groupby(cluster_key, dropna=False).size()
    coverage_lookup = coverage.set_index('cell_type').to_dict('index') if not coverage.empty else {}
    rows: list[dict[str, object]] = []
    for cell_type in type_columns:
        grouped = joined.groupby(cluster_key, dropna=False)[cell_type]
        statistics = grouped.agg(['size', 'mean', 'median'])
        statistics['q1'] = grouped.quantile(0.25)
        statistics['q3'] = grouped.quantile(0.75)
        for cluster, statistic in statistics.iterrows():
            top_number = int(top_counts.loc[cluster, cell_type]) if cluster in top_counts.index and cell_type in top_counts.columns else 0
            coverage_row = coverage_lookup.get(cell_type, {})
            n_positive = int(positive_counts.loc[cluster]) if cluster in positive_counts.index else 0
            rows.append({'dataset': dataset, 'sample': sample, 'clustering_source': clustering_source, 'cluster_key': cluster_key, 'cluster': str(cluster), 'method': method, 'granularity': granularity, 'cell_type': cell_type, 'n_cluster_cells': int(statistic['size']), 'mean_score': float(statistic['mean']), 'median_score': float(statistic['median']), 'q1_score': float(statistic['q1']), 'q3_score': float(statistic['q3']), 'top_cell_fraction': top_number / int(cluster_sizes.loc[cluster]), 'n_cells_with_positive_signature': n_positive, 'fraction_cells_with_positive_signature': n_positive / int(cluster_sizes.loc[cluster]), 'n_assignment_cells': int(len(assignments)), 'n_matched_score_cells': int(len(joined)), 'score_cell_match_fraction': len(joined) / len(assignments) if len(assignments) else np.nan, 'n_reference_signature_genes': coverage_row.get('n_reference_signature_genes', np.nan), 'n_measured_signature_genes': coverage_row.get('n_measured_signature_genes', np.nan), 'signature_coverage': coverage_row.get('signature_coverage', np.nan), 'measured_signature_genes': coverage_row.get('measured_signature_genes', ''), 'rank_statistic': rank_statistic})
    result = pd.DataFrame(rows)
    score_column = f'{rank_statistic}_score'
    result['cluster_signature_rank'] = result.groupby(['dataset', 'sample', 'clustering_source', 'cluster_key', 'cluster', 'method', 'granularity'], dropna=False)[score_column].rank(method='first', ascending=False).astype(int)
    group_columns = ['dataset', 'sample', 'clustering_source', 'cluster_key', 'cluster', 'method', 'granularity']
    gap_lookup: dict[tuple[object, ...], float] = {}
    for key, table in result.groupby(group_columns, dropna=False):
        ordered = table.sort_values('cluster_signature_rank')[score_column].tolist()
        gap_lookup[key] = float(ordered[0] - ordered[1]) if len(ordered) > 1 else np.nan
    result['top_two_score_gap'] = [gap_lookup[tuple((row[column] for column in group_columns))] for _, row in result.iterrows()]
    return result.sort_values([*group_columns, 'cluster_signature_rank'], key=lambda column: column.map(natural_key) if column.name in {'dataset', 'sample', 'cluster', 'cell_type'} else column, kind='stable').reset_index(drop=True)

def build_signature_top_two(cluster_summary: pd.DataFrame) -> pd.DataFrame:
    if cluster_summary.empty:
        return pd.DataFrame()
    keys = ['dataset', 'sample', 'clustering_source', 'cluster_key', 'cluster', 'method', 'granularity']
    rows: list[dict[str, object]] = []
    for key, table in cluster_summary.groupby(keys, dropna=False):
        rank_statistic = str(table.iloc[0].get('rank_statistic', 'median'))
        score_column = f'{rank_statistic}_score'
        if score_column not in table.columns:
            raise KeyError(f'Cannot rank signature evidence: {score_column} is absent')
        ranked = table.copy()
        ranked[score_column] = pd.to_numeric(ranked[score_column], errors='coerce')
        ranked = ranked.loc[ranked[score_column].notna()].copy()
        ranked = ranked.sort_values([score_column, 'cell_type'], ascending=[False, True], key=lambda column: column.map(natural_key) if column.name == 'cell_type' else column, kind='stable')
        first = ranked.iloc[0] if not ranked.empty else table.iloc[0]
        top_score = float(ranked.iloc[0][score_column]) if not ranked.empty else np.nan
        top_tied = ranked.loc[np.isclose(ranked[score_column], top_score)] if np.isfinite(top_score) else ranked.iloc[0:0]
        top_tied_signatures = sorted(top_tied['cell_type'].astype(str).tolist(), key=natural_key)
        if not np.isfinite(top_score) or top_score <= 0:
            result_status = 'no_positive_rank_signal'
            top_tied_signatures = []
            first_candidate = None
            second_candidate = None
        elif len(top_tied) > 1:
            result_status = 'tied_positive_top_signal'
            first_candidate = None
            second_candidate = None
        else:
            result_status = 'positive_unique_top_signal'
            first_candidate = ranked.iloc[0]
            remaining = ranked.iloc[1:].copy()
            second_candidate = remaining.iloc[0] if not remaining.empty else None
            if second_candidate is not None and float(second_candidate[score_column]) <= 0:
                second_candidate = None

        def candidate_value(candidate: pd.Series | None, column: str, default: object) -> object:
            return candidate.get(column, default) if candidate is not None else default
        row = dict(zip(keys, key))
        row.update({'signature_result_status': result_status, 'top_rank_score': top_score, 'n_top_tied_signatures': len(top_tied_signatures), 'top_tied_signatures': '; '.join(top_tied_signatures), 'candidate_1': candidate_value(first_candidate, 'cell_type', ''), 'candidate_1_mean_score': candidate_value(first_candidate, 'mean_score', np.nan), 'candidate_1_median_score': candidate_value(first_candidate, 'median_score', np.nan), 'candidate_1_top_cell_fraction': candidate_value(first_candidate, 'top_cell_fraction', np.nan), 'candidate_1_signature_coverage': candidate_value(first_candidate, 'signature_coverage', np.nan), 'candidate_2': candidate_value(second_candidate, 'cell_type', ''), 'candidate_2_mean_score': candidate_value(second_candidate, 'mean_score', np.nan), 'candidate_2_median_score': candidate_value(second_candidate, 'median_score', np.nan), 'candidate_2_top_cell_fraction': candidate_value(second_candidate, 'top_cell_fraction', np.nan), 'candidate_2_signature_coverage': candidate_value(second_candidate, 'signature_coverage', np.nan), 'candidate_score_gap': first.get('top_two_score_gap', np.nan) if result_status == 'positive_unique_top_signal' else np.nan, 'n_cluster_cells': first['n_cluster_cells'], 'n_cells_with_positive_signature': first.get('n_cells_with_positive_signature', np.nan), 'fraction_cells_with_positive_signature': first.get('fraction_cells_with_positive_signature', np.nan), 'rank_statistic': rank_statistic})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['method', 'granularity', 'clustering_source', 'dataset', 'sample', 'cluster_key', 'cluster'], key=lambda column: column.map(natural_key) if column.name in {'dataset', 'sample', 'cluster', 'candidate_1'} else column, kind='stable').reset_index(drop=True)

def run_signature_evidence(root: str | Path=PROJECT_ROOT, *, output_dir: str | Path | None=None, annotation_dir: str | Path | None=None, gene_panel_review_dir: str | Path | None=None, our_cluster_keys: Iterable[str]=('leiden_r1_3',), clustering_sources: Iterable[str]=('our_expression',), sample_keys: Iterable[str | tuple[str, str]] | None=None, run_ucell: bool=True, min_signature_genes: int=3, ucell_max_rank: int | None=None, ucell_chunk_size: int=5000, random_state: int=1729, min_cell_id_match_fraction: float=0.8, write_outputs: bool=True) -> dict[str, object]:
    """Score signatures with UCell and summarize expression clusters."""
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir or root / 'cell_clustering').expanduser().resolve()
    gene_panel_review_dir = Path(gene_panel_review_dir or root / 'gene_panel_review').expanduser().resolve()
    paths = signature_evidence_paths(output_dir)
    inventory = cluster_review.build_cluster_source_inventory(root, output_dir=output_dir, annotation_dir=annotation_dir, gene_panel_review_dir=gene_panel_review_dir)
    if inventory.empty:
        return {'run_summary': pd.DataFrame(), 'signature_coverage': pd.DataFrame(), 'cluster_summary': pd.DataFrame(), 'top_two': pd.DataFrame(), 'paths': paths}
    sources = set(map(str, clustering_sources))
    leiden_keys = set(map(str, our_cluster_keys))
    selected = inventory.loc[inventory['clustering_source'].astype(str).isin(sources) & inventory['has_cluster_output'].astype(bool) & inventory['has_feature_matrix'].astype(bool)].copy()
    selected = selected.loc[~selected['clustering_source'].astype(str).eq('our_expression') | selected['cluster_key'].astype(str).isin(leiden_keys)].copy()
    requested_samples = _normalize_sample_selection(sample_keys)
    if requested_samples is not None:
        selected = selected.loc[selected.apply(lambda row: _sample_key(row['dataset'], row['sample']) in requested_samples, axis=1)].copy()
    hint_sources = cluster_review.load_gene_marker_hint_sources(root, gene_panel_review_dir=gene_panel_review_dir)
    coverage_tables: list[pd.DataFrame] = []
    cluster_tables: list[pd.DataFrame] = []
    run_rows: list[dict[str, object]] = []
    for (dataset, sample), sample_inventory in selected.groupby(['dataset', 'sample'], sort=False):
        dataset = str(dataset)
        sample = str(sample)
        sample_key = _sample_key(dataset, sample)
        base_record = {'dataset': dataset, 'sample': sample, 'sample_key': sample_key}
        try:
            matrix_path = cluster_review.resolve_existing_path(sample_inventory.iloc[0].get('matrix_path', ''), root)
            if matrix_path is None:
                raise FileNotFoundError('Feature matrix is missing')
            adata = load_feature_matrix_as_adata(matrix_path)
            matrix_ids = pd.Index(adata.obs_names.astype(str))
            partition_records: list[tuple[pd.Series, pd.DataFrame, dict[str, object]]] = []
            union_ids: set[str] = set()
            for _, source_row in sample_inventory.iterrows():
                assignments, diagnostics = _load_partition_assignments(source_row, matrix_ids, root=root)
                matched = assignments['cell_id'].astype(str).isin(matrix_ids)
                match_fraction = float(matched.mean()) if len(assignments) else 0.0
                if match_fraction < float(min_cell_id_match_fraction):
                    run_rows.append({**base_record, 'method': 'partition_alignment', 'clustering_source': source_row['clustering_source'], 'cluster_key': source_row['cluster_key'], 'status': 'failed', 'error': f'Only {match_fraction:.1%} of assignment IDs matched the matrix'})
                    continue
                assignments = assignments.loc[matched].copy()
                union_ids.update(assignments['cell_id'].astype(str))
                partition_records.append((source_row, assignments, diagnostics))
            if not partition_records:
                raise ValueError('No cluster partitions passed cell-ID matching')
            cell_mask = matrix_ids.isin(union_ids)
            analysis_adata = adata[cell_mask].copy()
            gene_symbols = analysis_adata.var['gene_symbol'].astype(str).tolist() if 'gene_symbol' in analysis_adata.var.columns else analysis_adata.var_names.astype(str).tolist()
            signature_result = build_sample_signature_sets(dataset, sample, gene_symbols, hint_sources=hint_sources, min_signature_genes=min_signature_genes)
            coverage = signature_result['coverage']
            coverage_tables.append(coverage)
            signatures = signature_result['signatures']
        except Exception as error:
            run_rows.append({**base_record, 'method': 'sample_setup', 'clustering_source': '', 'cluster_key': '', 'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
            continue
        if run_ucell:
            combined_signatures = {f'{granularity}::{cell_type}': genes for granularity in ('refined', 'broad') for cell_type, genes in signatures[granularity].items()}
            try:
                combined_scores, metadata = compute_panel_aware_ucell_scores(analysis_adata, combined_signatures, max_rank=ucell_max_rank, chunk_size=ucell_chunk_size)
                for source_row, assignments, diagnostics in partition_records:
                    for granularity in ('refined', 'broad'):
                        prefix = f'{granularity}::'
                        columns = [column for column in combined_scores if column.startswith(prefix)]
                        if not columns:
                            continue
                        score_table = combined_scores[columns].rename(columns=lambda value: value[len(prefix):])
                        coverage_subset = coverage.loc[coverage['granularity'].eq(granularity) & coverage['eligible_for_primary_analysis'].astype(bool)].copy()
                        summary = summarize_cell_type_scores_by_cluster(score_table, assignments, dataset=dataset, sample=sample, clustering_source=str(source_row['clustering_source']), cluster_key=str(source_row['cluster_key']), method='ucell', granularity=granularity, coverage=coverage_subset, rank_statistic='median')
                        cluster_tables.append(summary)
                    run_rows.append({**base_record, 'method': 'ucell', 'clustering_source': source_row['clustering_source'], 'cluster_key': source_row['cluster_key'], 'status': 'success', 'error': '', **metadata, 'cell_id_match_method': diagnostics.get('cell_id_match_method', '')})
            except Exception as error:
                run_rows.append({**base_record, 'method': 'ucell', 'clustering_source': '', 'cluster_key': '', 'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
    run_summary = pd.DataFrame(run_rows)
    signature_coverage = pd.concat(coverage_tables, ignore_index=True, sort=False) if coverage_tables else pd.DataFrame()
    cluster_summary = pd.concat(cluster_tables, ignore_index=True, sort=False) if cluster_tables else pd.DataFrame()
    top_two = build_signature_top_two(cluster_summary)
    if write_outputs:
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        run_summary.to_csv(paths['run_summary'], index=False)
        signature_coverage.to_csv(paths['coverage'], index=False)
        cluster_summary.to_csv(paths['cluster_summary'], index=False)
        top_two.to_csv(paths['top_two'], index=False)
    return {'run_summary': run_summary, 'signature_coverage': signature_coverage, 'cluster_summary': cluster_summary, 'top_two': top_two, 'paths': paths, 'cluster_source_inventory': selected}

def plot_signature_heatmap(cluster_summary: pd.DataFrame, *, dataset: str, sample: str, clustering_source: str, cluster_key: str, method: str='ucell', granularity: str='broad', maximum_cell_types: int=20):
    """Plot a cluster-by-cell-type heatmap from the compact long summary."""
    import matplotlib.pyplot as plt
    selected = cluster_summary.loc[cluster_summary['dataset'].astype(str).eq(str(dataset)) & cluster_summary['sample'].astype(str).eq(str(sample)) & cluster_summary['clustering_source'].astype(str).eq(str(clustering_source)) & cluster_summary['cluster_key'].astype(str).eq(str(cluster_key)) & cluster_summary['method'].astype(str).eq(str(method)) & cluster_summary['granularity'].astype(str).eq(str(granularity))].copy()
    if selected.empty:
        raise ValueError('No signature-sensitivity rows match the requested plot')
    score_column = 'median_score'
    leading_types = selected.groupby('cell_type')[score_column].max().sort_values(ascending=False).head(int(maximum_cell_types)).index
    selected = selected.loc[selected['cell_type'].isin(leading_types)].copy()
    pivot = selected.pivot(index='cluster', columns='cell_type', values=score_column)
    pivot = pivot.reindex(sorted(pivot.index, key=natural_key))
    pivot = pivot.reindex(columns=list(leading_types))
    width = max(8.0, 0.45 * len(pivot.columns) + 3.0)
    height = max(4.5, 0.3 * len(pivot.index) + 2.5)
    figure, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(pivot.to_numpy(dtype=float), aspect='auto', vmin=0, vmax=1, cmap='viridis')
    axis.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=60, ha='right')
    axis.set_yticks(range(len(pivot.index)), labels=pivot.index)
    axis.set_xlabel(f'{granularity.capitalize()} cell type')
    axis.set_ylabel('Cluster')
    axis.set_title(f'{method.upper()} {granularity} sensitivity | {dataset} / {sample}\n{clustering_source} / {cluster_key}')
    figure.colorbar(image, ax=axis, label=score_column.replace('_', ' '))
    figure.tight_layout()
    return figure
