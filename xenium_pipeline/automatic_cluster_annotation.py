"""Reference-free automatic cluster annotation for the Xenium workflow.

The functions in this module combine three related but distinct summaries:

* the distribution of notebook-15 marker-score labels within each cluster;
* notebook-18 cluster-level UCell signature rankings; and
* notebook-18 differential-expression marker-gene hints.

No author cell labels or external single-cell reference atlas is used to create
the automatic labels. Author labels can therefore remain a later comparison
layer rather than leaking into the annotation procedure.
"""
from __future__ import annotations
from .storage import data_path
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from .cluster_review import MISSING_CLUSTER_ASSIGNMENT_LABELS, match_cluster_source_cell_ids
KEY_COLUMNS = ['dataset', 'sample', 'clustering_source', 'cluster_key', 'cluster']
XOA_LABEL_SOURCE_TOKEN = 'xoa_graphclust'
UNAVAILABLE_CELL_LABELS = {'', 'Not available', 'Missing XOA assignment', 'Not clustered (QC)', 'XOA unassigned'}
NONBIOLOGICAL_CELL_LABELS = {'Uncertain', 'Custom', 'Other / uncertain', 'Other / unmapped'}

@dataclass(frozen=True)
class AutomaticAnnotationConfig:
    """Thresholds used to qualify evidence and form a consensus label."""
    uncertain_label: str = 'Uncertain'
    min_cluster_cells: int = 20
    min_assigned_score_fraction: float = 0.2
    min_top_score_label_fraction: float = 0.5
    min_score_label_fraction_gap: float = 0.15
    min_signature_coverage: float = 0.5
    min_signature_score_gap: float = 0.02
    min_signature_positive_cell_fraction: float = 0.1
    min_consensus_methods: int = 2

def _stringify(series: pd.Series) -> pd.Series:
    return series.fillna('').astype(str).str.strip()

def _valid_cluster_assignment(series: pd.Series) -> pd.Series:
    return ~_stringify(series).str.lower().isin(MISSING_CLUSTER_ASSIGNMENT_LABELS)

def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())

def _table_columns(path: Path) -> list[str]:
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []

def _sample_from_filename(filename: str, samples: Iterable[str]) -> str:
    matches = [sample for sample in samples if filename.startswith(f'{sample}_')]
    return max(matches, key=len) if matches else ''

def select_sample_leiden_resolutions(samples: pd.DataFrame, default_resolution: float=1.3, resolution_by_sample: dict[tuple[str, str], float] | None=None) -> pd.DataFrame:
    """Select one explicitly identified Leiden partition for each sample."""
    from .cell_clustering import cluster_key_for_resolution
    selected = samples[['dataset', 'sample']].copy().astype(str)
    if selected.empty or selected.duplicated().any():
        raise ValueError('Sample selection must be nonempty and unique')
    overrides = resolution_by_sample or {}
    unknown = set(overrides) - set(selected.itertuples(index=False, name=None))
    if unknown:
        raise ValueError(f'Unknown dataset/sample resolution overrides: {unknown}')
    values = [float(overrides.get(pair, default_resolution)) for pair in selected.itertuples(index=False, name=None)]
    if any((not np.isfinite(value) or value <= 0 for value in values)):
        raise ValueError('Every selected resolution must be finite and positive')
    selected['selected_resolution'] = values
    selected['cluster_key'] = [cluster_key_for_resolution(value) for value in values]
    return selected

def select_leiden_evidence(table: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    """Keep only our selected sample partitions; never substitute XOA/other resolutions."""
    keys = ['dataset', 'sample', 'cluster_key']
    own = table.loc[table['clustering_source'].eq('our_expression')].copy()
    for key in keys:
        own[key] = _stringify(own[key])
    return own.merge(selection[keys], on=keys, how='inner', validate='many_to_one')

def propagate_selected_leiden_labels(marker_labels: pd.DataFrame, assignments: pd.DataFrame, labels: pd.DataFrame, *, dataset: str, sample: str, cluster_key: str) -> pd.DataFrame:
    """Export actual-key and resolution-independent initial labels, with no XOA columns."""
    from .cell_identity_evidence import collapse_label
    if marker_labels['cell_id'].duplicated().any() or assignments['cell_id'].duplicated().any():
        raise ValueError('Duplicate cell IDs in marker labels or Leiden assignments')
    expected = set(_stringify(assignments.loc[_valid_cluster_assignment(_stringify(assignments[cluster_key])), cluster_key]))
    mapping = labels.loc[labels['dataset'].eq(dataset) & labels['sample'].eq(sample) & labels['clustering_source'].eq('our_expression') & labels['cluster_key'].eq(cluster_key)]
    if mapping['cluster'].duplicated().any() or expected - set(_stringify(mapping['cluster'])):
        raise ValueError('Selected cluster labels are duplicated or missing; regenerate selected consensus labels')
    if set(_stringify(assignments['cell_id'])) - set(_stringify(marker_labels['cell_id'])):
        raise ValueError('Some clustered cells are absent from notebook-15 labels')
    cells = propagate_labels_to_sample_cells(marker_labels, assignments, labels, dataset=dataset, sample=sample, leiden_keys=(cluster_key,))
    cells = cells.drop(columns=[column for column in cells if column.startswith('xoa_')])
    cells['initial_cluster_key'] = cluster_key
    cells['initial_cluster_id'] = cells[cluster_key]
    cells['initial_cluster_automatic_label'] = cells[f'{cluster_key}_automatic_label']
    cells['initial_cluster_automatic_confidence'] = cells[f'{cluster_key}_automatic_confidence']
    certain = _valid_cluster_assignment(cells['initial_cluster_id']) & cell_label_masks(cells['initial_cluster_automatic_label'])[1]
    certain &= _stringify(cells['initial_cluster_automatic_confidence']).str.lower().isin({'moderate', 'high'})
    cells['initial_cluster_is_certain'] = certain
    cells['initial_cluster_refined_label'] = cells['initial_cluster_automatic_label'].where(certain, 'Uncertain')
    broad = {label: collapse_label(label)[0] for label in cells['initial_cluster_refined_label'].unique()}
    broad['Uncertain'] = 'Uncertain'
    cells['initial_cluster_broad_label'] = cells['initial_cluster_refined_label'].map(broad)
    return cells

def audit_required_inputs(root: str | Path, *, annotation_dir: str | Path | None=None, clustering_dir: str | Path | None=None, gene_panel_review_dir: str | Path | None=None, leiden_keys: tuple[str, ...]=('leiden_r1_3',), default_leiden_resolution: float | None=None, leiden_resolution_by_sample: dict[tuple[str, str], float] | None=None) -> dict[str, pd.DataFrame]:
    """Check required global and per-sample outputs without loading raw matrices."""
    root = Path(root).expanduser().resolve()
    annotation_dir = Path(annotation_dir or root / 'cell_annotation').resolve()
    clustering_dir = Path(clustering_dir or root / 'cell_clustering').resolve()
    gene_panel_review_dir = Path(gene_panel_review_dir or root / 'gene_panel_review').resolve()
    global_paths = {'sample_panel_summary': gene_panel_review_dir / 'sample_panel_summary.csv', 'marker_gene_table': gene_panel_review_dir / 'marker_gene_table.csv', 'panel_implied_marker_groups': gene_panel_review_dir / 'panel_implied_marker_groups.csv', 'annotation_run_summary': annotation_dir / 'annotation_run_summary.csv', 'clustering_input_table': clustering_dir / 'clustering_input_table.csv', 'clustering_run_summary': clustering_dir / 'clustering_run_summary.csv', 'marker_discovery_summary': clustering_dir / 'cluster_source_expression_marker_discovery_summary.csv', 'expression_marker_summary': clustering_dir / 'combined_cluster_source_expression_marker_summary.csv', 'expression_top_markers': clustering_dir / 'combined_cluster_source_expression_top_markers.csv', 'gene_hint_candidates': clustering_dir / 'combined_cluster_source_top_two_gene_hint_candidates.csv', 'signature_summary': data_path(root, 'signature_summary'), 'signature_top_two': data_path(root, 'signature_top_two')}
    file_rows = [{'input_name': name, 'path': _safe_relative(path, root), 'exists': path.is_file(), 'nonempty': path.is_file() and path.stat().st_size > 0, 'status': 'ready' if path.is_file() and path.stat().st_size > 0 else 'missing_or_empty'} for name, path in global_paths.items()]
    file_checks = pd.DataFrame(file_rows)
    annotation_summary_path = global_paths['annotation_run_summary']
    clustering_input_path = global_paths['clustering_input_table']
    if not annotation_summary_path.is_file() or not clustering_input_path.is_file():
        return {'file_checks': file_checks, 'sample_checks': pd.DataFrame(), 'sample_table': pd.DataFrame()}
    annotation_summary = pd.read_csv(annotation_summary_path)
    clustering_input = pd.read_csv(clustering_input_path)
    sample_table = annotation_summary[['dataset', 'sample']].drop_duplicates().merge(clustering_input[['dataset', 'sample']].drop_duplicates(), on=['dataset', 'sample'], how='outer', indicator=True)
    sample_table['in_both_workflows'] = sample_table.pop('_merge').eq('both')
    if default_leiden_resolution is not None:
        selection = select_sample_leiden_resolutions(sample_table, default_leiden_resolution, leiden_resolution_by_sample)
        sample_table = sample_table.merge(selection, on=['dataset', 'sample'], validate='one_to_one')
    annotation_required = {'dataset', 'sample', 'cell_id', 'group', 'top_marker_group', 'top_score', 'runner_up', 'runner_up_score', 'score_margin', 'call_status'}
    sample_rows: list[dict[str, object]] = []
    for row in sample_table.itertuples(index=False):
        dataset, sample = (str(row.dataset), str(row.sample))
        selected_keys = (row.cluster_key,) if default_leiden_resolution is not None else leiden_keys
        cluster_required = {'dataset', 'sample', 'cell_id', *selected_keys}
        annotation_path = annotation_dir / dataset / sample / 'cell_marker_labels.csv.gz'
        cluster_path = clustering_dir / dataset / sample / 'cell_clusters.csv.gz'
        annotation_columns = set(_table_columns(annotation_path))
        cluster_columns = set(_table_columns(cluster_path))
        missing_annotation_columns = sorted(annotation_required - annotation_columns)
        missing_cluster_columns = sorted(cluster_required - cluster_columns)
        ready = bool(row.in_both_workflows) and annotation_path.is_file() and cluster_path.is_file() and (not missing_annotation_columns) and (not missing_cluster_columns)
        sample_rows.append({'dataset': dataset, 'sample': sample, 'annotation_path': _safe_relative(annotation_path, root), 'cluster_path': _safe_relative(cluster_path, root), 'annotation_file_exists': annotation_path.is_file(), 'cluster_file_exists': cluster_path.is_file(), 'missing_annotation_columns': '; '.join(missing_annotation_columns), 'missing_cluster_columns': '; '.join(missing_cluster_columns), 'status': 'ready' if ready else 'not_ready'})
    sample_checks = pd.DataFrame(sample_rows)
    return {'file_checks': file_checks, 'sample_checks': sample_checks, 'sample_table': sample_table.sort_values(['dataset', 'sample'], kind='stable').reset_index(drop=True)}

def build_xoa_source_inventory(root: str | Path, sample_table: pd.DataFrame, *, xoa_dir: str | Path | None=None, marker_discovery_summary_path: str | Path | None=None, label_source_token: str=XOA_LABEL_SOURCE_TOKEN) -> pd.DataFrame:
    """Inventory candidate files and select only explicit XOA graph assignments."""
    root = Path(root).expanduser().resolve()
    xoa_dir = Path(xoa_dir or root / 'XOA').resolve()
    marker_discovery_summary_path = Path(marker_discovery_summary_path or root / 'cell_clustering' / 'cluster_source_expression_marker_discovery_summary.csv').resolve()
    sample_lookup = {str(row.sample): str(row.dataset) for row in sample_table.itertuples(index=False)}
    candidates: dict[Path, tuple[str, str, str]] = {}
    if xoa_dir.is_dir():
        for path in sorted(xoa_dir.glob('*.csv')):
            sample = _sample_from_filename(path.name, sample_lookup)
            dataset = sample_lookup.get(sample, '')
            candidates[path.resolve()] = (dataset, sample, 'xoa_subfolder')
    if marker_discovery_summary_path.is_file():
        discovery = pd.read_csv(marker_discovery_summary_path)
        if 'clustering_source' in discovery.columns:
            discovery = discovery.loc[discovery['clustering_source'].astype(str).eq('xoa')]
        for row in discovery.itertuples(index=False):
            value = str(getattr(row, 'cluster_source_path', '')).strip()
            if not value:
                continue
            source_path = Path(value)
            if not source_path.is_absolute():
                source_path = root / source_path
            if source_path.is_file():
                candidates.setdefault(source_path.resolve(), (str(row.dataset), str(row.sample), 'notebook_06_source'))
    rows: list[dict[str, object]] = []
    for path, (dataset, sample, discovered_from) in candidates.items():
        columns = _table_columns(path)
        required = {'cell_id', 'cell_type', 'label_source'}
        missing_columns = sorted(required - set(columns))
        label_sources: list[str] = []
        if not missing_columns:
            label_sources = sorted(pd.read_csv(path, usecols=['label_source'], dtype='string')['label_source'].dropna().str.strip().str.lower().unique().tolist())
        is_genuine_xoa = bool(label_sources) and all((str(label_source_token).lower() in source for source in label_sources))
        if not dataset or not sample:
            status = 'unrecognized_sample'
        elif missing_columns:
            status = 'missing_required_columns'
        elif is_genuine_xoa:
            status = 'ready_xoa'
        else:
            status = 'excluded_non_xoa'
        rows.append({'dataset': dataset, 'sample': sample, 'path': _safe_relative(path, root), 'filename': path.name, 'discovered_from': discovered_from, 'label_sources': '; '.join(label_sources), 'missing_required_columns': '; '.join(missing_columns), 'is_genuine_xoa': is_genuine_xoa, 'status': status, 'selected': False})
    inventory = pd.DataFrame(rows)
    if inventory.empty:
        return inventory
    ready = inventory.loc[inventory['status'].eq('ready_xoa')].copy()
    for (_dataset, _sample), group in ready.groupby(['dataset', 'sample'], sort=False):
        preferred = group.sort_values('discovered_from', key=lambda value: value.map({'xoa_subfolder': 0, 'notebook_06_source': 1}), kind='stable').index[0]
        inventory.loc[preferred, 'selected'] = True
    return inventory.sort_values(['dataset', 'sample', 'status', 'filename'], kind='stable').reset_index(drop=True)

def load_xoa_assignments(path: str | Path, target_cell_ids: pd.Index | pd.Series | list[str], *, label_source_token: str=XOA_LABEL_SOURCE_TOKEN) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load a genuine XOA table and align its cell IDs to a target cell table."""
    path = Path(path)
    source = pd.read_csv(path, usecols=['cell_id', 'cell_type', 'label_source'], dtype={'cell_id': 'string', 'cell_type': 'string', 'label_source': 'string'})
    sources = sorted(source['label_source'].dropna().str.strip().str.lower().unique())
    token = str(label_source_token).lower()
    if not sources or any((token not in value for value in sources)):
        raise ValueError(f'{path.name} is not an XOA graph-cluster table; label_source={sources}')
    assignments, diagnostics = match_cluster_source_cell_ids(source, cell_id_columns=['cell_id'], cluster_column='cell_type', cluster_key='xoa_cluster_id', matrix_cell_ids=pd.Index(pd.Series(target_cell_ids).astype(str)))
    return (assignments, diagnostics)

def partition_pair_summary(matched: pd.DataFrame, left_key: str, right_key: str) -> pd.DataFrame:
    """Count every observed pair and calculate row, column, and sample fractions."""
    pairs = matched.groupby([left_key, right_key], dropna=False).size().rename('n_cells').reset_index()
    if pairs.empty:
        return pairs
    left_totals = pairs.groupby(left_key)['n_cells'].transform('sum')
    right_totals = pairs.groupby(right_key)['n_cells'].transform('sum')
    pairs['fraction_of_left_cluster'] = pairs['n_cells'] / left_totals
    pairs['fraction_of_right_cluster'] = pairs['n_cells'] / right_totals
    pairs['fraction_of_matched_cells'] = pairs['n_cells'] / pairs['n_cells'].sum()
    return pairs.sort_values([left_key, 'n_cells', right_key], ascending=[True, False, True], kind='stable').reset_index(drop=True)

def partition_similarity_from_pairs(pairs: pd.DataFrame, left_key: str, right_key: str) -> dict[str, float | int]:
    """Calculate ARI, NMI, and directional purity from a pair-count table."""
    if pairs.empty:
        return {'n_matched_cells': 0, 'n_left_clusters': 0, 'n_right_clusters': 0, 'adjusted_rand_index': np.nan, 'normalized_mutual_information': np.nan, 'left_to_right_purity': np.nan, 'right_to_left_purity': np.nan}
    table = pairs.pivot_table(index=left_key, columns=right_key, values='n_cells', aggfunc='sum', fill_value=0).to_numpy(dtype=float)
    n = float(table.sum())
    row_totals = table.sum(axis=1)
    column_totals = table.sum(axis=0)
    choose2 = lambda values: np.sum(values * (values - 1.0) / 2.0)
    sum_cells = float(choose2(table))
    sum_rows = float(choose2(row_totals))
    sum_columns = float(choose2(column_totals))
    total_pairs = n * (n - 1.0) / 2.0
    expected = sum_rows * sum_columns / total_pairs if total_pairs else 0.0
    maximum = 0.5 * (sum_rows + sum_columns)
    ari = (sum_cells - expected) / (maximum - expected) if maximum != expected else 1.0
    probabilities = table / n
    row_probabilities = row_totals / n
    column_probabilities = column_totals / n
    expected_probabilities = np.outer(row_probabilities, column_probabilities)
    positive = probabilities > 0
    mutual_information = float(np.sum(probabilities[positive] * np.log(probabilities[positive] / expected_probabilities[positive])))
    left_entropy = float(-np.sum(row_probabilities[row_probabilities > 0] * np.log(row_probabilities[row_probabilities > 0])))
    right_entropy = float(-np.sum(column_probabilities[column_probabilities > 0] * np.log(column_probabilities[column_probabilities > 0])))
    denominator = np.sqrt(left_entropy * right_entropy)
    nmi = mutual_information / denominator if denominator else 1.0
    return {'n_matched_cells': int(n), 'n_left_clusters': int(table.shape[0]), 'n_right_clusters': int(table.shape[1]), 'adjusted_rand_index': float(ari), 'normalized_mutual_information': float(nmi), 'left_to_right_purity': float(table.max(axis=1).sum() / n), 'right_to_left_purity': float(table.max(axis=0).sum() / n)}

def compare_xoa_with_leiden(root: str | Path, xoa_inventory: pd.DataFrame, *, clustering_dir: str | Path | None=None, leiden_keys: tuple[str, ...]=('leiden_r1_3',), xoa_label_source_token: str=XOA_LABEL_SOURCE_TOKEN) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare genuine XOA partitions with Leiden using shared cells only."""
    root = Path(root).expanduser().resolve()
    clustering_dir = Path(clustering_dir or root / 'cell_clustering').resolve()
    metric_rows: list[dict[str, object]] = []
    pair_tables: list[pd.DataFrame] = []
    selected = xoa_inventory.loc[xoa_inventory['selected'].astype(bool)]
    for source in selected.itertuples(index=False):
        cluster_path = clustering_dir / str(source.dataset) / str(source.sample) / 'cell_clusters.csv.gz'
        leiden = pd.read_csv(cluster_path, usecols=['cell_id', *leiden_keys], dtype={'cell_id': 'string'})
        leiden['cell_id'] = _stringify(leiden['cell_id'])
        xoa_path = Path(str(source.path))
        if not xoa_path.is_absolute():
            xoa_path = root / xoa_path
        xoa, diagnostics = load_xoa_assignments(xoa_path, leiden['cell_id'], label_source_token=xoa_label_source_token)
        xoa = xoa.loc[_valid_cluster_assignment(xoa['xoa_cluster_id'])].copy()
        for leiden_key in leiden_keys:
            matched = xoa.merge(leiden[['cell_id', leiden_key]], on='cell_id', how='inner')
            matched['xoa_cluster_id'] = _stringify(matched['xoa_cluster_id'])
            matched[leiden_key] = _stringify(matched[leiden_key])
            pairs = partition_pair_summary(matched, 'xoa_cluster_id', leiden_key)
            metrics = partition_similarity_from_pairs(pairs, 'xoa_cluster_id', leiden_key)
            metric_rows.append({'dataset': str(source.dataset), 'sample': str(source.sample), 'leiden_cluster_key': leiden_key, **metrics, 'xoa_cell_id_match_method': diagnostics['cell_id_match_method']})
            pairs.insert(0, 'leiden_cluster_key', leiden_key)
            pairs.insert(0, 'sample', str(source.sample))
            pairs.insert(0, 'dataset', str(source.dataset))
            pair_tables.append(pairs)
    return (pd.DataFrame(metric_rows), pd.concat(pair_tables, ignore_index=True) if pair_tables else pd.DataFrame())

def summarize_marker_score_labels_by_cluster(assignments: pd.DataFrame, labels: pd.DataFrame, *, dataset: str, sample: str, clustering_source: str, cluster_key: str, uncertain_label: str='Uncertain') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize notebook-15 hard calls within every cluster."""
    assignment_table = assignments[['cell_id', cluster_key]].copy()
    assignment_table['cell_id'] = _stringify(assignment_table['cell_id'])
    assignment_table[cluster_key] = _stringify(assignment_table[cluster_key])
    label_columns = ['cell_id', 'group']
    if 'call_status' in labels.columns:
        label_columns.append('call_status')
    label_table = labels[label_columns].copy()
    label_table['cell_id'] = _stringify(label_table['cell_id'])
    matched = assignment_table.merge(label_table, on='cell_id', how='left')
    matched['group'] = _stringify(matched['group'])
    assigned = matched['group'].ne('') & matched['group'].ne(uncertain_label)
    if 'call_status' in matched.columns:
        assigned &= matched['call_status'].fillna('').astype(str).eq('assigned')
    matched['is_assigned_score_label'] = assigned
    cluster_sizes = matched.groupby(cluster_key, dropna=False).agg(n_cluster_cells=('cell_id', 'size'), n_matched_score_cells=('group', lambda value: int(value.ne('').sum())), n_assigned_score_cells=('is_assigned_score_label', 'sum')).reset_index().rename(columns={cluster_key: 'cluster'})
    cluster_sizes['assigned_score_fraction'] = cluster_sizes['n_assigned_score_cells'] / cluster_sizes['n_cluster_cells']
    composition = matched.loc[matched['is_assigned_score_label']].groupby([cluster_key, 'group'], dropna=False).size().rename('n_cells').reset_index().rename(columns={cluster_key: 'cluster', 'group': 'marker_score_label'})
    if not composition.empty:
        composition = composition.merge(cluster_sizes[['cluster', 'n_cluster_cells', 'n_assigned_score_cells']], on='cluster', how='left')
        composition['fraction_of_cluster'] = composition['n_cells'] / composition['n_cluster_cells']
        composition['fraction_of_assigned_score_cells'] = composition['n_cells'] / composition['n_assigned_score_cells']
        composition = composition.sort_values(['cluster', 'n_cells', 'marker_score_label'], ascending=[True, False, True], kind='stable')
        composition['score_label_rank'] = composition.groupby('cluster').cumcount() + 1
    candidate_rows: list[dict[str, object]] = []
    composition_groups = {str(cluster): group.reset_index(drop=True) for cluster, group in composition.groupby('cluster', sort=False)} if not composition.empty else {}
    for row in cluster_sizes.itertuples(index=False):
        ranked = composition_groups.get(str(row.cluster), pd.DataFrame())
        first = ranked.iloc[0] if len(ranked) else None
        second = ranked.iloc[1] if len(ranked) > 1 else None
        first_fraction = float(first['fraction_of_assigned_score_cells']) if first is not None else 0.0
        second_fraction = float(second['fraction_of_assigned_score_cells']) if second is not None else 0.0
        candidate_rows.append({'cluster': str(row.cluster), 'n_cluster_cells': int(row.n_cluster_cells), 'n_matched_score_cells': int(row.n_matched_score_cells), 'n_assigned_score_cells': int(row.n_assigned_score_cells), 'assigned_score_fraction': float(row.assigned_score_fraction), 'score_vote_candidate_1': str(first['marker_score_label']) if first is not None else '', 'score_vote_candidate_1_fraction': first_fraction, 'score_vote_candidate_2': str(second['marker_score_label']) if second is not None else '', 'score_vote_candidate_2_fraction': second_fraction, 'score_vote_fraction_gap': first_fraction - second_fraction})
    summary = pd.DataFrame(candidate_rows)
    for table in (summary, composition):
        table.insert(0, 'cluster_key', cluster_key)
        table.insert(0, 'clustering_source', clustering_source)
        table.insert(0, 'sample', sample)
        table.insert(0, 'dataset', dataset)
    return (summary, composition.reset_index(drop=True))

def build_cluster_score_vote_tables(root: str | Path, sample_table: pd.DataFrame, xoa_inventory: pd.DataFrame | None=None, *, annotation_dir: str | Path | None=None, clustering_dir: str | Path | None=None, leiden_keys: tuple[str, ...]=('leiden_r1_3',), uncertain_label: str='Uncertain', xoa_label_source_token: str=XOA_LABEL_SOURCE_TOKEN) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build cluster-level marker-score vote summaries for Leiden and XOA."""
    root = Path(root).expanduser().resolve()
    annotation_dir = Path(annotation_dir or root / 'cell_annotation').resolve()
    clustering_dir = Path(clustering_dir or root / 'cell_clustering').resolve()
    selected_xoa = {} if xoa_inventory is None else {(str(row.dataset), str(row.sample)): row for row in xoa_inventory.loc[xoa_inventory['selected'].astype(bool)].itertuples(index=False)}
    summaries: list[pd.DataFrame] = []
    compositions: list[pd.DataFrame] = []
    for row in sample_table.itertuples(index=False):
        dataset, sample = (str(row.dataset), str(row.sample))
        selected_keys = (str(row.cluster_key),) if hasattr(row, 'cluster_key') else leiden_keys
        labels_path = annotation_dir / dataset / sample / 'cell_marker_labels.csv.gz'
        clusters_path = clustering_dir / dataset / sample / 'cell_clusters.csv.gz'
        labels = pd.read_csv(labels_path, usecols=['cell_id', 'group', 'call_status'], dtype={'cell_id': 'string', 'group': 'string', 'call_status': 'string'})
        clusters = pd.read_csv(clusters_path, usecols=['cell_id', *selected_keys], dtype={column: 'string' for column in ['cell_id', *selected_keys]})
        for leiden_key in selected_keys:
            summary, composition = summarize_marker_score_labels_by_cluster(clusters[['cell_id', leiden_key]], labels, dataset=dataset, sample=sample, clustering_source='our_expression', cluster_key=leiden_key, uncertain_label=uncertain_label)
            summaries.append(summary)
            compositions.append(composition)
        xoa_row = selected_xoa.get((dataset, sample))
        if xoa_row is not None:
            xoa_path = Path(str(xoa_row.path))
            if not xoa_path.is_absolute():
                xoa_path = root / xoa_path
            xoa, _diagnostics = load_xoa_assignments(xoa_path, labels['cell_id'], label_source_token=xoa_label_source_token)
            xoa = xoa.loc[_valid_cluster_assignment(xoa['xoa_cluster_id'])].copy()
            summary, composition = summarize_marker_score_labels_by_cluster(xoa, labels, dataset=dataset, sample=sample, clustering_source='xoa', cluster_key='xoa_cluster_id', uncertain_label=uncertain_label)
            summaries.append(summary)
            compositions.append(composition)
    return (pd.concat(summaries, ignore_index=True), pd.concat(compositions, ignore_index=True))

def assign_automatic_cluster_labels(score_vote_summary: pd.DataFrame, signature_top_two: pd.DataFrame, gene_hint_candidates: pd.DataFrame, *, config: AutomaticAnnotationConfig=AutomaticAnnotationConfig(), signature_method: str='ucell', signature_granularity: str='refined') -> pd.DataFrame:
    """Qualify three evidence candidates and assign a two-of-three consensus."""
    scores = score_vote_summary.copy()
    for column in KEY_COLUMNS:
        scores[column] = _stringify(scores[column])
    signatures = signature_top_two.copy()
    signatures = signatures.loc[signatures['method'].astype(str).eq(signature_method) & signatures['granularity'].astype(str).eq(signature_granularity)].copy()
    for column in KEY_COLUMNS:
        signatures[column] = _stringify(signatures[column])
    signature_columns = KEY_COLUMNS + ['signature_result_status', 'candidate_1', 'candidate_2', 'candidate_1_median_score', 'candidate_2_median_score', 'candidate_1_top_cell_fraction', 'candidate_1_signature_coverage', 'candidate_score_gap', 'fraction_cells_with_positive_signature']
    signatures = signatures[[column for column in signature_columns if column in signatures]]
    signatures = signatures.rename(columns={'candidate_1': 'signature_candidate_1', 'candidate_2': 'signature_candidate_2'})
    hints = gene_hint_candidates.copy()
    for column in KEY_COLUMNS:
        hints[column] = _stringify(hints[column])
    hint_columns = KEY_COLUMNS + ['automatic_candidate_1', 'automatic_candidate_2', 'candidate_1_support_fraction', 'candidate_2_support_fraction', 'candidate_support_gap', 'candidate_1_n_supporting_genes', 'heuristic_recommendation_code', 'heuristic_recommendation_reason']
    hints = hints[[column for column in hint_columns if column in hints]].rename(columns={'automatic_candidate_1': 'gene_hint_candidate_1', 'automatic_candidate_2': 'gene_hint_candidate_2'})
    combined = scores.merge(signatures, on=KEY_COLUMNS, how='left').merge(hints, on=KEY_COLUMNS, how='left')
    numeric_columns = ['n_cluster_cells', 'assigned_score_fraction', 'score_vote_candidate_1_fraction', 'score_vote_fraction_gap', 'candidate_1_signature_coverage', 'candidate_score_gap', 'fraction_cells_with_positive_signature']
    for column in numeric_columns:
        combined[column] = pd.to_numeric(combined.get(column), errors='coerce')
    combined['score_vote_qualified'] = combined['n_cluster_cells'].ge(config.min_cluster_cells) & combined['assigned_score_fraction'].ge(config.min_assigned_score_fraction) & combined['score_vote_candidate_1_fraction'].ge(config.min_top_score_label_fraction) & combined['score_vote_fraction_gap'].ge(config.min_score_label_fraction_gap) & _stringify(combined['score_vote_candidate_1']).ne('')
    combined['score_vote_label'] = _stringify(combined['score_vote_candidate_1']).where(combined['score_vote_qualified'], '')
    combined['signature_qualified'] = _stringify(combined.get('signature_result_status', pd.Series(index=combined.index))).eq('positive_unique_top_signal') & combined['candidate_1_signature_coverage'].ge(config.min_signature_coverage) & combined['candidate_score_gap'].ge(config.min_signature_score_gap) & combined['fraction_cells_with_positive_signature'].ge(config.min_signature_positive_cell_fraction) & _stringify(combined.get('signature_candidate_1', pd.Series(index=combined.index))).ne('')
    combined['signature_label'] = _stringify(combined.get('signature_candidate_1', pd.Series(index=combined.index))).where(combined['signature_qualified'], '')
    combined['gene_hint_qualified'] = _stringify(combined.get('heuristic_recommendation_code', pd.Series(index=combined.index))).eq('candidate_1') & _stringify(combined.get('gene_hint_candidate_1', pd.Series(index=combined.index))).ne('')
    combined['gene_hint_label'] = _stringify(combined.get('gene_hint_candidate_1', pd.Series(index=combined.index))).where(combined['gene_hint_qualified'], '')
    label_rows: list[dict[str, object]] = []
    methods = (('marker_score_vote', 'score_vote_label'), ('ucell_signature', 'signature_label'), ('de_gene_hint', 'gene_hint_label'))
    for row in combined.itertuples(index=False):
        votes = [(method, str(getattr(row, column)).strip()) for method, column in methods if str(getattr(row, column)).strip()]
        counts: dict[str, int] = {}
        for _method, label in votes:
            counts[label] = counts.get(label, 0) + 1
        winner = max(counts, key=lambda label: (counts[label], label)) if counts else ''
        support = counts.get(winner, 0)
        accepted = bool(winner) and support >= config.min_consensus_methods
        supporting_methods = [method for method, label in votes if label == winner]
        disagreeing_methods = [method for method, label in votes if label != winner]
        if accepted and support == 3:
            confidence = 'high'
        elif accepted:
            confidence = 'moderate'
        else:
            confidence = 'uncertain'
        if accepted:
            reason = f'{support} qualifying evidence methods agree'
        elif not votes:
            reason = 'No evidence method passed its qualification thresholds'
        else:
            reason = 'Fewer than two qualifying evidence methods agree'
        label_rows.append({'automatic_cluster_label': winner if accepted else config.uncertain_label, 'automatic_label_confidence': confidence, 'n_qualifying_evidence_methods': len(votes), 'n_supporting_evidence_methods': support if accepted else 0, 'supporting_evidence_methods': '; '.join(supporting_methods) if accepted else '', 'disagreeing_evidence_methods': '; '.join(disagreeing_methods), 'automatic_label_reason': reason})
    return pd.concat([combined.reset_index(drop=True), pd.DataFrame(label_rows)], axis=1)

def propagate_labels_to_sample_cells(marker_labels: pd.DataFrame, leiden_assignments: pd.DataFrame, automatic_cluster_labels: pd.DataFrame, *, dataset: str, sample: str, leiden_keys: tuple[str, ...]=('leiden_r1_3',), xoa_assignments: pd.DataFrame | None=None) -> pd.DataFrame:
    """Create one cell table containing the direct and cluster-derived labels."""
    direct_columns = ['cell_id', 'group', 'top_marker_group', 'top_score', 'runner_up', 'runner_up_score', 'score_margin', 'call_status']
    cells = marker_labels[[column for column in direct_columns if column in marker_labels]].copy()
    cells['cell_id'] = _stringify(cells['cell_id'])
    cells = cells.rename(columns={'group': 'marker_score_label', 'top_marker_group': 'marker_score_top_group', 'call_status': 'marker_score_call_status'})
    leiden = leiden_assignments[['cell_id', *leiden_keys]].copy()
    leiden['cell_id'] = _stringify(leiden['cell_id'])
    for leiden_key in leiden_keys:
        leiden[leiden_key] = _stringify(leiden[leiden_key])
    cells = cells.merge(leiden, on='cell_id', how='left')
    labels = automatic_cluster_labels.copy()
    for column in KEY_COLUMNS:
        labels[column] = _stringify(labels[column])
    for leiden_key in leiden_keys:
        mapping = labels.loc[labels['dataset'].eq(dataset) & labels['sample'].eq(sample) & labels['clustering_source'].eq('our_expression') & labels['cluster_key'].eq(leiden_key), ['cluster', 'automatic_cluster_label', 'automatic_label_confidence']].rename(columns={'cluster': leiden_key, 'automatic_cluster_label': f'{leiden_key}_automatic_label', 'automatic_label_confidence': f'{leiden_key}_automatic_confidence'})
        cells[leiden_key] = _stringify(cells[leiden_key])
        cells = cells.merge(mapping, on=leiden_key, how='left')
        missing_cluster = cells[leiden_key].eq('')
        cells.loc[missing_cluster, f'{leiden_key}_automatic_label'] = 'Not clustered (QC)'
        cells.loc[missing_cluster, f'{leiden_key}_automatic_confidence'] = 'not_available'
    if xoa_assignments is not None:
        xoa = xoa_assignments[['cell_id', 'xoa_cluster_id']].copy()
        xoa['cell_id'] = _stringify(xoa['cell_id'])
        xoa['xoa_cluster_id'] = _stringify(xoa['xoa_cluster_id'])
        cells = cells.merge(xoa, on='cell_id', how='left')
        mapping = labels.loc[labels['dataset'].eq(dataset) & labels['sample'].eq(sample) & labels['clustering_source'].eq('xoa') & labels['cluster_key'].eq('xoa_cluster_id'), ['cluster', 'automatic_cluster_label', 'automatic_label_confidence']].rename(columns={'cluster': 'xoa_cluster_id', 'automatic_cluster_label': 'xoa_automatic_label', 'automatic_label_confidence': 'xoa_automatic_confidence'})
        cells['xoa_cluster_id'] = _stringify(cells['xoa_cluster_id'])
        cells = cells.merge(mapping, on='xoa_cluster_id', how='left')
        missing_xoa = cells['xoa_cluster_id'].eq('')
        unassigned_xoa = ~missing_xoa & ~_valid_cluster_assignment(cells['xoa_cluster_id'])
        cells.loc[missing_xoa, 'xoa_automatic_label'] = 'Missing XOA assignment'
        cells.loc[missing_xoa, 'xoa_automatic_confidence'] = 'not_available'
        cells.loc[unassigned_xoa, 'xoa_automatic_label'] = 'XOA unassigned'
        cells.loc[unassigned_xoa, 'xoa_automatic_confidence'] = 'not_available'
    else:
        cells['xoa_cluster_id'] = ''
        cells['xoa_automatic_label'] = 'Not available'
        cells['xoa_automatic_confidence'] = 'not_available'
    cells.insert(0, 'sample', sample)
    cells.insert(0, 'dataset', dataset)
    return cells

def summarize_cell_annotation_sets(cells: pd.DataFrame, *, leiden_keys: tuple[str, ...]=('leiden_r1_3',)) -> pd.DataFrame:
    """Count cells in every label for each annotation set."""
    label_columns = ['marker_score_label'] + [f'{key}_automatic_label' for key in leiden_keys] + ['xoa_automatic_label']
    rows: list[pd.DataFrame] = []
    for column in label_columns:
        if column not in cells.columns:
            continue
        counts = cells.groupby(['dataset', 'sample', column], dropna=False).size().rename('n_cells').reset_index().rename(columns={column: 'cell_type'})
        counts.insert(2, 'annotation_set', column)
        counts['fraction_of_sample'] = counts['n_cells'] / counts.groupby(['dataset', 'sample', 'annotation_set'])['n_cells'].transform('sum')
        rows.append(counts)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def cell_label_masks(labels: pd.Series, *, uncertain_label: str='Uncertain') -> tuple[pd.Series, pd.Series]:
    """Return masks for available and biological cell labels."""
    clean = _stringify(labels)
    available = ~clean.isin(UNAVAILABLE_CELL_LABELS)
    nonbiological = NONBIOLOGICAL_CELL_LABELS | {uncertain_label}
    biological = available & ~clean.isin(nonbiological)
    return (available, biological)

def cohen_kappa(left: pd.Series, right: pd.Series) -> float:
    """Calculate unweighted Cohen's kappa for two same-vocabulary label series."""
    left = _stringify(left)
    right = _stringify(right)
    if len(left) == 0:
        return np.nan
    observed = float(left.eq(right).mean())
    labels = sorted(set(left) | set(right))
    expected = sum((float(left.eq(label).mean()) * float(right.eq(label).mean()) for label in labels))
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0

def cohen_kappa_from_confusion(counts: pd.DataFrame) -> float:
    """Calculate Cohen's kappa from a confusion matrix with named axes."""
    labels = sorted(set(map(str, counts.index)) | set(map(str, counts.columns)))
    aligned = counts.copy()
    aligned.index = aligned.index.astype(str)
    aligned.columns = aligned.columns.astype(str)
    aligned = aligned.reindex(index=labels, columns=labels, fill_value=0)
    n = float(aligned.to_numpy().sum())
    if n == 0:
        return np.nan
    observed = float(np.trace(aligned.to_numpy()) / n)
    expected = float(np.sum(aligned.sum(axis=1).to_numpy(dtype=float) * aligned.sum(axis=0).to_numpy(dtype=float)) / (n * n))
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0

def compare_cell_label_columns(cells: pd.DataFrame, left_column: str, right_column: str, *, comparison: str, dataset: str, sample: str, uncertain_label: str='Uncertain') -> dict[str, object]:
    """Summarize availability and biological agreement for two annotation sets."""
    left = _stringify(cells[left_column])
    right = _stringify(cells[right_column])
    left_available, left_biological = cell_label_masks(left, uncertain_label=uncertain_label)
    right_available, right_biological = cell_label_masks(right, uncertain_label=uncertain_label)
    comparable = left_available & right_available
    jointly_biological = left_biological & right_biological
    exact = left.eq(right) & jointly_biological
    n_cells = len(cells)
    n_comparable = int(comparable.sum())
    n_joint = int(jointly_biological.sum())
    return {'dataset': dataset, 'sample': sample, 'comparison': comparison, 'left_annotation': left_column, 'right_annotation': right_column, 'n_sample_cells': n_cells, 'n_left_available_cells': int(left_available.sum()), 'n_right_available_cells': int(right_available.sum()), 'n_comparable_cells': n_comparable, 'comparable_fraction': n_comparable / n_cells if n_cells else np.nan, 'n_left_biological_cells': int(left_biological.sum()), 'n_right_biological_cells': int(right_biological.sum()), 'n_jointly_biological_cells': n_joint, 'jointly_biological_fraction': n_joint / n_cells if n_cells else np.nan, 'n_exact_biological_agreement_cells': int(exact.sum()), 'exact_biological_agreement_fraction': float(exact.sum() / n_joint) if n_joint else np.nan, 'cohen_kappa': cohen_kappa(left.loc[jointly_biological], right.loc[jointly_biological]), 'n_left_biological_right_nonbiological': int((left_biological & ~right_biological).sum()), 'n_left_nonbiological_right_biological': int((~left_biological & right_biological).sum())}

def biological_confusion_counts(cells: pd.DataFrame, left_column: str, right_column: str, *, uncertain_label: str='Uncertain') -> pd.DataFrame:
    """Return a biological-label-only confusion matrix."""
    left = _stringify(cells[left_column])
    right = _stringify(cells[right_column])
    _left_available, left_biological = cell_label_masks(left, uncertain_label=uncertain_label)
    _right_available, right_biological = cell_label_masks(right, uncertain_label=uncertain_label)
    keep = left_biological & right_biological
    return pd.crosstab(left.loc[keep], right.loc[keep], dropna=False)

def top_biological_disagreements(cells: pd.DataFrame, left_column: str, right_column: str, *, top_n: int=10, uncertain_label: str='Uncertain') -> pd.DataFrame:
    """Return only the most frequent unequal biological-label pairs."""
    left = _stringify(cells[left_column])
    right = _stringify(cells[right_column])
    _left_available, left_biological = cell_label_masks(left, uncertain_label=uncertain_label)
    _right_available, right_biological = cell_label_masks(right, uncertain_label=uncertain_label)
    keep = left_biological & right_biological & left.ne(right)
    if not keep.any():
        return pd.DataFrame(columns=['left_label', 'right_label', 'n_cells'])
    return pd.DataFrame({'left_label': left.loc[keep], 'right_label': right.loc[keep]}).groupby(['left_label', 'right_label'], dropna=False).size().rename('n_cells').reset_index().sort_values('n_cells', ascending=False, kind='stable').head(top_n).reset_index(drop=True)

def diagnose_cluster_evidence_failures(labels: pd.DataFrame, *, config: AutomaticAnnotationConfig=AutomaticAnnotationConfig()) -> pd.DataFrame:
    """Describe why each cluster evidence method did or did not qualify."""

    def text(row: pd.Series, column: str) -> str:
        value = row.get(column, '')
        return '' if pd.isna(value) else str(value).strip()

    def number(row: pd.Series, column: str) -> float:
        return float(pd.to_numeric(row.get(column), errors='coerce'))
    rows: list[dict[str, object]] = []
    for _, row in labels.iterrows():
        score_failures = []
        if not number(row, 'n_cluster_cells') >= config.min_cluster_cells:
            score_failures.append('cluster_too_small')
        if not number(row, 'assigned_score_fraction') >= config.min_assigned_score_fraction:
            score_failures.append('low_assigned_score_fraction')
        if not number(row, 'score_vote_candidate_1_fraction') >= config.min_top_score_label_fraction:
            score_failures.append('low_top_vote_fraction')
        if not number(row, 'score_vote_fraction_gap') >= config.min_score_label_fraction_gap:
            score_failures.append('small_vote_fraction_gap')
        if not text(row, 'score_vote_candidate_1'):
            score_failures.append('missing_score_vote_candidate')
        signature_failures = []
        if text(row, 'signature_result_status') != 'positive_unique_top_signal':
            signature_failures.append('top_signature_not_unique_positive')
        if not number(row, 'candidate_1_signature_coverage') >= config.min_signature_coverage:
            signature_failures.append('low_signature_coverage')
        if not number(row, 'candidate_score_gap') >= config.min_signature_score_gap:
            signature_failures.append('small_signature_score_gap')
        if not number(row, 'fraction_cells_with_positive_signature') >= config.min_signature_positive_cell_fraction:
            signature_failures.append('low_positive_signature_fraction')
        if not text(row, 'signature_candidate_1'):
            signature_failures.append('missing_signature_candidate')
        gene_hint_failures = []
        if text(row, 'heuristic_recommendation_code') != 'candidate_1':
            gene_hint_failures.append('gene_hint_not_single_candidate')
        if not text(row, 'gene_hint_candidate_1'):
            gene_hint_failures.append('missing_gene_hint_candidate')
        qualified = []
        for method, flag, label_column in (('marker_score_vote', bool(row.get('score_vote_qualified', False)), 'score_vote_label'), ('ucell_signature', bool(row.get('signature_qualified', False)), 'signature_label'), ('de_gene_hint', bool(row.get('gene_hint_qualified', False)), 'gene_hint_label')):
            label = text(row, label_column)
            if flag and label:
                qualified.append((method, label))
        distinct_labels = {label for _, label in qualified}
        reason = text(row, 'automatic_label_reason')
        if reason.startswith('No evidence method passed'):
            failure_mode = 'no_evidence_method_qualified'
        elif reason.startswith('Fewer than two qualifying evidence methods agree'):
            failure_mode = 'insufficient_evidence_consensus'
        else:
            failure_mode = 'clear_consensus'
        rows.append({'score_vote_failure_reasons': '; '.join(score_failures), 'signature_failure_reasons': '; '.join(signature_failures), 'gene_hint_failure_reasons': '; '.join(gene_hint_failures), 'qualified_method_pattern': '; '.join((method for method, _ in qualified)), 'qualified_label_pattern': '; '.join((f'{method}={label}' for method, label in qualified)), 'n_distinct_qualified_labels': len(distinct_labels), 'consensus_failure_mode': failure_mode})
    return pd.concat([labels.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
