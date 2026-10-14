"""Audit and study-level reclustering for cells without a certain label.

Use the annotation command for batch processing and the manual recluster
notebook for accepted review decisions. Scientific dependencies load only
inside reclustering functions.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from . import dataset_config
from .automatic_cluster_annotation import cell_label_masks
from .cell_clustering import build_clustering_input_table, cluster_key_for_resolution, load_feature_matrix_as_adata, path_for_record, safe_name
from .cell_identity_evidence import clean_author_refined_label, natural_key
from .gene_panel_utils import is_gene_expression
HIERARCHY_LAYERS = ('certain_leiden_label', 'author_label_rescue', 'reclustering_target', 'qc_excluded')
DEFAULT_AUTHOR_UNUSABLE_LABELS = ('Unlabeled', 'Missing', 'Unknown', 'Unassigned', 'Uncertain')
AUTHOR_LABEL_SOURCE_TOKENS = ('author', 'provided', 'published', 'manual', 'reviewed')
CELL_ID_COLUMN_CANDIDATES = ('cell_id', 'barcode', 'cell')
LABEL_COLUMN_CANDIDATES = ('author_label', 'author_cell_type', 'cell_type', 'group', 'annotation', 'label', 'cell_label', 'cluster', 'type')
AUTOMATIC_REQUIRED_COLUMNS = ('dataset', 'sample', 'cell_id', 'leiden_r1_3', 'leiden_r1_3_automatic_label', 'leiden_r1_3_automatic_confidence')
AUTOMATIC_OPTIONAL_COLUMNS = ('marker_score_label', 'marker_score_top_group', 'top_score', 'runner_up', 'runner_up_score', 'score_margin', 'marker_score_call_status', 'xoa_cluster_id', 'xoa_automatic_label', 'xoa_automatic_confidence')

@dataclass(frozen=True)
class ReclusteringConfig:
    mode: str = 'balanced_sketch'
    normalization_target: float = 10000.0
    n_pcs: int = 20
    n_neighbors: int = 20
    leiden_resolutions: tuple[float, ...] = (0.7,)
    random_state: int = 1729
    run_umap: bool = True
    max_full_cells_per_study: int = 300000
    max_sketch_cells_per_study: int = 250000
    min_sketch_cells_per_stratum: int = 100
    max_sketch_cells_per_stratum: int = 10000
    projection_neighbors: int = 15
    projection_chunk_size: int = 50000
    min_projection_vote_fraction: float = 0.5
    min_projection_vote_margin: float = 0.1
    max_umap_plot_cells: int = 100000
    write_model_cache: bool = True
    overwrite_existing: bool = False
    skip_completed: bool = True

def _progress(message: str, *, started_at: float | None=None) -> None:
    """Print one timestamped progress line, optionally with elapsed time."""
    elapsed = '' if started_at is None else f' | elapsed {time.perf_counter() - started_at:,.1f}s'
    print(f'[{datetime.now():%H:%M:%S}] {message}{elapsed}', flush=True)

def normalize_status(value: object) -> str:
    """Normalize a status token while preserving biological label punctuation."""
    text = '' if pd.isna(value) else str(value).strip().casefold()
    return re.sub('\\s+', ' ', text)

def parameter_fingerprint(config: ReclusteringConfig) -> str:
    payload = asdict(config)
    for operational_key in ('max_full_cells_per_study', 'max_umap_plot_cells', 'projection_chunk_size', 'write_model_cache', 'overwrite_existing', 'skip_completed'):
        payload.pop(operational_key, None)
    payload['leiden_resolutions'] = list(config.leiden_resolutions)
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]

def _resolve_path(value: object, root: Path) -> Path | None:
    if value is None or pd.isna(value) or (not str(value).strip()):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path if path.exists() else None

def _csv_header(path: Path | None) -> list[str]:
    if path is None or not path.is_file():
        return []
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []

def _infer_column(columns: Iterable[str], candidates: Iterable[str]) -> str:
    lookup = {str(column).casefold(): str(column) for column in columns}
    return next((lookup[str(candidate).casefold()] for candidate in candidates if str(candidate).casefold() in lookup), '')

def _preview_label_sources(path: Path) -> set[str]:
    if 'label_source' not in _csv_header(path):
        return set()
    try:
        preview = pd.read_csv(path, nrows=500, usecols=['label_source'])
    except Exception:
        return set()
    return set(preview['label_source'].dropna().astype(str).str.casefold().unique())

def _dataset_uses_provided_groups(dataset: str) -> bool:
    config = dataset_config.dataset_configs().get(dataset)
    if config is None:
        return False
    source = str(config.cell_groups_label_source).casefold()
    return any((token in source for token in AUTHOR_LABEL_SOURCE_TOKENS))

def discover_author_source(root: str | Path, dataset: str, sample: str, sample_dir: str | Path | None, *, consolidated_path: str | Path | None=None, parquet_label_columns: dict[str, str] | None=None) -> dict[str, str]:
    """Find an author-label source using the author-source priority."""
    root = Path(root).expanduser().resolve()
    sample_path = _resolve_path(sample_dir, root)
    parquet_label_columns = parquet_label_columns or {}
    candidates: list[tuple[Path, str, str, str]] = []
    if sample_path is not None and sample_path.is_dir():
        for filename in (f'{sample}_author_cell_groups.csv', f'{sample}_author_annotations.csv', f'{sample}_published_cell_groups.csv', f'{sample}_provided_cell_groups.csv'):
            path = sample_path / filename
            if path.is_file():
                candidates.append((path, 'explicit_author_file', '', ''))
        cell_groups = sample_path / f'{sample}_cell_groups.csv'
        if cell_groups.is_file() and _dataset_uses_provided_groups(dataset):
            candidates.append((cell_groups, 'provided_cell_groups', '', ''))
        annotated = sample_path / f'{sample}_cells_with_annotations.csv'
        if annotated.is_file():
            sources = _preview_label_sources(annotated)
            is_author = not sources or any((any((token in source for token in AUTHOR_LABEL_SOURCE_TOKENS)) for source in sources))
            if is_author:
                candidates.append((annotated, 'cells_with_annotations', '', ''))
        parquet_column = parquet_label_columns.get(dataset, '')
        if parquet_column:
            for path in sorted(sample_path.glob('*cells*.parquet*'), key=lambda item: natural_key(item.name)):
                candidates.append((path, 'parquet_manual_column', 'cell_id', parquet_column))
    first_invalid: tuple[Path, str] | None = None
    for path, source_type, cell_column, label_column in candidates:
        if source_type != 'parquet_manual_column':
            columns = _csv_header(path)
            cell_column = _infer_column(columns, CELL_ID_COLUMN_CANDIDATES)
            label_column = _infer_column(columns, LABEL_COLUMN_CANDIDATES)
        if cell_column and label_column:
            return {'author_path': path_for_record(path, root), 'author_source_type': source_type, 'author_cell_id_column': cell_column, 'author_label_column': label_column, 'author_source_status': 'ready'}
        if first_invalid is None:
            first_invalid = (path, source_type)
    consolidated = _resolve_path(consolidated_path, root)
    if consolidated is not None:
        columns = _csv_header(consolidated)
        label_column = _infer_column(columns, ('author_refined_label', 'author_raw_label', 'author_label'))
        required = {'dataset', 'sample', 'cell_id'}
        status = 'ready' if required.issubset(columns) and label_column else 'invalid_columns'
        return {'author_path': path_for_record(consolidated, root), 'author_source_type': 'consolidated_evidence', 'author_cell_id_column': 'cell_id', 'author_label_column': label_column, 'author_source_status': status}
    return {'author_path': '', 'author_source_type': first_invalid[1] if first_invalid else '', 'author_cell_id_column': '', 'author_label_column': '', 'author_source_status': 'invalid_columns' if first_invalid else 'not_found'}

def build_reclustering_inventory(root: str | Path, *, consolidated_author_path: str | Path | None=None, selected_studies: Iterable[str] | None=None, parquet_label_columns: dict[str, str] | None=None) -> pd.DataFrame:
    """Build one input-audit row per sample."""
    root = Path(root).expanduser().resolve()
    registry = build_clustering_input_table(root, sample_order='dataset_sample')
    if selected_studies is not None:
        selected = {str(value) for value in selected_studies}
        registry = registry.loc[registry['dataset'].astype(str).isin(selected)].copy()
    rows: list[dict[str, object]] = []
    for row in registry.itertuples(index=False):
        dataset, sample = (str(row.dataset), str(row.sample))
        auto_path = root / 'automatic_cluster_annotation' / safe_name(dataset) / safe_name(sample) / 'cell_automatic_annotations.csv.gz'
        author = discover_author_source(root, dataset, sample, getattr(row, 'sample_dir', ''), consolidated_path=consolidated_author_path, parquet_label_columns=parquet_label_columns)
        matrix_path = _resolve_path(getattr(row, 'matrix_path', ''), root)
        automatic_columns = _csv_header(auto_path)
        missing_auto_columns = sorted(set(AUTOMATIC_REQUIRED_COLUMNS) - set(automatic_columns))
        has_automatic = auto_path.is_file() and (not missing_auto_columns)
        has_author = author['author_source_status'] == 'ready'
        has_matrix = matrix_path is not None
        rows.append({'dataset': dataset, 'sample': sample, 'sample_dir': str(getattr(row, 'sample_dir', '')), 'automatic_annotation_path': path_for_record(auto_path, root), 'has_automatic_annotations': has_automatic, 'missing_automatic_columns': '; '.join(missing_auto_columns), 'matrix_path': path_for_record(matrix_path, root) if matrix_path else '', 'has_feature_matrix': has_matrix, **author, 'has_author_labels': has_author, 'ready_for_hierarchy': has_automatic, 'ready_for_reclustering': has_automatic and has_matrix})
    return pd.DataFrame(rows)

def load_automatic_cell_annotations(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    columns = _csv_header(path)
    missing = sorted(set(AUTOMATIC_REQUIRED_COLUMNS) - set(columns))
    if missing:
        raise KeyError(f"{path} is missing required columns: {', '.join(missing)}")
    usecols = [column for column in (*AUTOMATIC_REQUIRED_COLUMNS, *AUTOMATIC_OPTIONAL_COLUMNS) if column in columns]
    table = pd.read_csv(path, usecols=usecols, dtype={'cell_id': 'string'})
    table['cell_id'] = table['cell_id'].astype(str)
    if table['cell_id'].duplicated().any():
        raise ValueError(f'{path} contains duplicate cell_id values')
    return table

def load_author_labels(source: pd.Series | dict[str, object], root: str | Path, *, consolidated_table: pd.DataFrame | None=None) -> pd.DataFrame:
    """Load one clean author label per cell for one sample."""
    root = Path(root).expanduser().resolve()
    source = dict(source)
    if source.get('author_source_status') != 'ready':
        return pd.DataFrame(columns=['cell_id', 'author_raw_label', 'author_label'])
    dataset, sample = (str(source['dataset']), str(source['sample']))
    source_type = str(source['author_source_type'])
    cell_column = str(source['author_cell_id_column'])
    label_column = str(source['author_label_column'])
    if source_type == 'consolidated_evidence' and consolidated_table is not None:
        table = consolidated_table.loc[consolidated_table['dataset'].astype(str).eq(dataset) & consolidated_table['sample'].astype(str).eq(sample), ['cell_id', label_column]].copy()
    else:
        path = _resolve_path(source.get('author_path'), root)
        if path is None:
            raise FileNotFoundError(source.get('author_path', ''))
        if '.parquet' in ''.join(path.suffixes).casefold():
            table = pd.read_parquet(path, columns=[cell_column, label_column])
        else:
            table = pd.read_csv(path, usecols=[cell_column, label_column], dtype={cell_column: 'string'}, keep_default_na=False)
        table = table.rename(columns={cell_column: 'cell_id'})
    table = table.rename(columns={label_column: 'author_raw_label'})
    table['cell_id'] = table['cell_id'].astype(str)
    table['author_label'] = table['author_raw_label'].map(clean_author_refined_label)
    duplicates = table.loc[table['cell_id'].duplicated(keep=False)]
    if not duplicates.empty:
        conflicts = duplicates.groupby('cell_id')['author_label'].nunique(dropna=False).gt(1)
        if conflicts.any():
            raise ValueError(f'{dataset}/{sample} author source has conflicting duplicate cell IDs')
        table = table.drop_duplicates('cell_id', keep='first')
    return table[['cell_id', 'author_raw_label', 'author_label']].reset_index(drop=True)

def author_label_usable_mask(labels: pd.Series, unusable_labels: Iterable[str]=DEFAULT_AUTHOR_UNUSABLE_LABELS) -> pd.Series:
    unusable = {normalize_status(value) for value in unusable_labels} | {'', 'nan', 'none', '<na>'}
    return ~labels.map(normalize_status).isin(unusable)

def build_cell_label_hierarchy(automatic: pd.DataFrame, author_labels: pd.DataFrame | None=None, *, unusable_author_labels: Iterable[str]=DEFAULT_AUTHOR_UNUSABLE_LABELS) -> pd.DataFrame:
    """Assign every automatic-annotation row to exactly one hierarchy layer."""
    missing = sorted(set(AUTOMATIC_REQUIRED_COLUMNS) - set(automatic.columns))
    if missing:
        raise KeyError(f"Automatic annotations are missing: {', '.join(missing)}")
    cells = automatic.copy()
    cells['cell_id'] = cells['cell_id'].astype(str)
    if author_labels is None or author_labels.empty:
        cells['author_raw_label'] = pd.NA
        cells['author_label'] = pd.NA
    else:
        author = author_labels.copy()
        author['cell_id'] = author['cell_id'].astype(str)
        cells = cells.merge(author, on='cell_id', how='left', validate='one_to_one')
    cluster_text = cells['leiden_r1_3'].fillna('').astype(str).str.strip()
    has_cluster = ~cluster_text.str.casefold().isin({'', 'nan', 'none', 'unassigned', 'missing'})
    _available, biological = cell_label_masks(cells['leiden_r1_3_automatic_label'])
    confidence = cells['leiden_r1_3_automatic_confidence'].map(normalize_status)
    certain = has_cluster & biological & confidence.isin({'moderate', 'high'})
    author_usable = author_label_usable_mask(cells['author_label'], unusable_author_labels)
    cells['has_original_leiden_r1_3_cluster'] = has_cluster
    cells['leiden_r1_3_label_is_biological'] = biological
    cells['leiden_r1_3_confidence_normalized'] = confidence
    cells['leiden_r1_3_confidence_is_moderate_or_high'] = confidence.isin({'moderate', 'high'})
    cells['leiden_r1_3_label_is_certain'] = certain
    cells['author_label_is_usable'] = author_usable
    cells['hierarchy_layer'] = 'reclustering_target'
    cells.loc[author_usable, 'hierarchy_layer'] = 'author_label_rescue'
    cells.loc[certain, 'hierarchy_layer'] = 'certain_leiden_label'
    cells.loc[~has_cluster, 'hierarchy_layer'] = 'qc_excluded'
    cells['hierarchy_selected_label'] = 'Needs study-level reclustering'
    cells.loc[cells['hierarchy_layer'].eq('certain_leiden_label'), 'hierarchy_selected_label'] = cells['leiden_r1_3_automatic_label']
    cells.loc[cells['hierarchy_layer'].eq('author_label_rescue'), 'hierarchy_selected_label'] = cells['author_label']
    cells.loc[cells['hierarchy_layer'].eq('qc_excluded'), 'hierarchy_selected_label'] = 'Not clustered (QC)'
    reason_map = {'certain_leiden_label': 'biological r1.3 cluster label with moderate/high confidence', 'author_label_rescue': 'uncertain r1.3 cluster label with usable author label', 'reclustering_target': 'uncertain r1.3 cluster label without usable author label', 'qc_excluded': 'no original r1.3 cluster assignment'}
    cells['hierarchy_reason'] = cells['hierarchy_layer'].map(reason_map)
    cells['reclustering_eligible'] = cells['hierarchy_layer'].eq('reclustering_target')
    cells['reclustering_exclusion_reason'] = cells['hierarchy_layer'].map({'certain_leiden_label': 'already has certain Leiden label', 'author_label_rescue': 'rescued by usable author label', 'reclustering_target': '', 'qc_excluded': 'failed or absent original clustering QC'})
    observed = set(cells['hierarchy_layer'].unique())
    if not observed.issubset(HIERARCHY_LAYERS) or cells['hierarchy_layer'].isna().any():
        raise AssertionError('Hierarchy assignment is not exhaustive')
    return cells

def summarize_hierarchy(hierarchy: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Return one wide count row per requested sample or study grouping."""
    grouped = hierarchy.groupby(group_columns, dropna=False, sort=False)
    totals = grouped.size().rename('n_total_cells')
    layer_counts = hierarchy.groupby([*group_columns, 'hierarchy_layer'], dropna=False).size().unstack(fill_value=0).reindex(columns=HIERARCHY_LAYERS, fill_value=0)
    layer_counts.columns = [f'n_{column}' for column in layer_counts.columns]
    result = totals.to_frame().join(layer_counts).reset_index()
    result['n_clustering_eligible_cells'] = result['n_total_cells'] - result['n_qc_excluded']
    for layer in HIERARCHY_LAYERS:
        count_column = f'n_{layer}'
        result[f'fraction_{layer}_of_all_cells'] = result[count_column] / result['n_total_cells'].replace(0, np.nan)
        result[f'fraction_{layer}_of_clustering_eligible_cells'] = result[count_column] / result['n_clustering_eligible_cells'].replace(0, np.nan)
    result['fraction_qc_excluded_of_clustering_eligible_cells'] = np.nan
    count_columns = [f'n_{layer}' for layer in HIERARCHY_LAYERS]
    if not result[count_columns].sum(axis=1).eq(result['n_total_cells']).all():
        raise AssertionError('Hierarchy counts do not sum to total cells')
    return result

def _load_consolidated_author_table(inventory: pd.DataFrame, root: Path) -> pd.DataFrame | None:
    rows = inventory.loc[inventory['author_source_type'].eq('consolidated_evidence')]
    if rows.empty:
        return None
    path = _resolve_path(rows.iloc[0]['author_path'], root)
    if path is None:
        return None
    label_columns = rows['author_label_column'].dropna().astype(str).unique().tolist()
    if len(label_columns) != 1:
        raise ValueError('Consolidated author evidence must use one author-label column')
    needed = set(zip(rows['dataset'].astype(str), rows['sample'].astype(str)))
    needed_index = pd.MultiIndex.from_tuples(sorted(needed), names=['dataset', 'sample'])
    parts = []
    for chunk in pd.read_csv(path, usecols=['dataset', 'sample', 'cell_id', label_columns[0]], dtype={'cell_id': 'string'}, keep_default_na=False, chunksize=500000):
        chunk_keys = pd.MultiIndex.from_arrays([chunk['dataset'].astype(str), chunk['sample'].astype(str)], names=['dataset', 'sample'])
        keep = pd.Series(chunk_keys.isin(needed_index), index=chunk.index)
        if keep.any():
            parts.append(chunk.loc[keep].copy())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['dataset', 'sample', 'cell_id', label_columns[0]])

def run_hierarchy_audit(inventory: pd.DataFrame, root: str | Path, output_dir: str | Path, *, unusable_author_labels: Iterable[str]=DEFAULT_AUTHOR_UNUSABLE_LABELS, write_outputs: bool=True, overwrite_existing: bool=True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build per-cell hierarchy files and return sample/study counts plus audit."""
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    consolidated = _load_consolidated_author_table(inventory, root)
    count_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for row in inventory.to_dict('records'):
        dataset, sample = (str(row['dataset']), str(row['sample']))
        audit = dict(row)
        audit.update({'hierarchy_status': 'not_run', 'n_cells': 0, 'error': ''})
        if not bool(row.get('ready_for_hierarchy', False)):
            audit['hierarchy_status'] = 'missing_automatic_annotations'
            audit_rows.append(audit)
            continue
        try:
            automatic_path = _resolve_path(row['automatic_annotation_path'], root)
            if automatic_path is None:
                raise FileNotFoundError(row['automatic_annotation_path'])
            automatic = load_automatic_cell_annotations(automatic_path)
            observed_datasets = set(automatic['dataset'].astype(str).unique())
            observed_samples = set(automatic['sample'].astype(str).unique())
            if observed_datasets != {dataset} or observed_samples != {sample}:
                raise ValueError(f'Automatic table identity mismatch: datasets={sorted(observed_datasets)}, samples={sorted(observed_samples)}')
            author = load_author_labels(row, root, consolidated_table=consolidated)
            hierarchy = build_cell_label_hierarchy(automatic, author, unusable_author_labels=unusable_author_labels)
            counts = summarize_hierarchy(hierarchy, ['dataset', 'sample'])
            count_parts.append(counts)
            output_path = output_dir / safe_name(dataset) / safe_name(sample) / 'cell_label_hierarchy.csv.gz'
            if write_outputs:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if output_path.exists() and (not overwrite_existing):
                    raise FileExistsError(output_path)
                hierarchy.to_csv(output_path, index=False, compression='gzip')
            audit.update({'hierarchy_status': 'success', 'n_cells': len(hierarchy), 'n_has_original_leiden_r1_3_cluster': int(hierarchy['has_original_leiden_r1_3_cluster'].sum()), 'n_biological_r1_3_automatic_label': int(hierarchy['leiden_r1_3_label_is_biological'].sum()), 'n_moderate_or_high_r1_3_confidence': int(hierarchy['leiden_r1_3_confidence_is_moderate_or_high'].sum()), 'n_certain_leiden_label': int(hierarchy['leiden_r1_3_label_is_certain'].sum()), 'r1_3_confidence_values': '; '.join(sorted(hierarchy['leiden_r1_3_confidence_normalized'].dropna().astype(str).unique())), 'n_author_labels_matched': int(hierarchy['author_label'].notna().sum()), 'hierarchy_output_path': path_for_record(output_path, root)})
        except Exception as error:
            audit['hierarchy_status'] = 'failed'
            audit['error'] = f'{type(error).__name__}: {error}'
        audit_rows.append(audit)
    sample_counts = pd.concat(count_parts, ignore_index=True) if count_parts else pd.DataFrame()
    study_counts = pd.DataFrame()
    if not sample_counts.empty:
        count_columns = ['n_total_cells', 'n_clustering_eligible_cells', *[f'n_{layer}' for layer in HIERARCHY_LAYERS]]
        study_counts = sample_counts.groupby('dataset', as_index=False)[count_columns].sum()
        for layer in HIERARCHY_LAYERS:
            study_counts[f'fraction_{layer}_of_all_cells'] = study_counts[f'n_{layer}'] / study_counts['n_total_cells'].replace(0, np.nan)
            study_counts[f'fraction_{layer}_of_clustering_eligible_cells'] = study_counts[f'n_{layer}'] / study_counts['n_clustering_eligible_cells'].replace(0, np.nan)
        study_counts['fraction_qc_excluded_of_clustering_eligible_cells'] = np.nan
    audit_table = pd.DataFrame(audit_rows)
    if write_outputs:
        sample_counts.to_csv(output_dir / 'hierarchy_counts_by_sample.csv', index=False)
        study_counts.to_csv(output_dir / 'hierarchy_counts_by_study.csv', index=False)
        audit_table.to_csv(output_dir / 'hierarchy_input_audit.csv', index=False)
    return (sample_counts, study_counts, audit_table)

def allocate_balanced_counts(capacities: Iterable[int], total: int) -> np.ndarray:
    """Allocate an integer budget as evenly as possible without exceeding capacities."""
    capacity = np.asarray(list(capacities), dtype=int)
    allocation = np.zeros_like(capacity)
    remaining = min(max(int(total), 0), int(capacity.sum()))
    while remaining > 0:
        active = np.flatnonzero(allocation < capacity)
        if active.size == 0:
            break
        share = max(1, remaining // active.size)
        additions = np.minimum(capacity[active] - allocation[active], share)
        addition_total = int(additions.sum())
        if addition_total > remaining:
            additions[:] = 0
            additions[:remaining] = 1
            addition_total = remaining
        allocation[active] += additions
        remaining -= addition_total
    return allocation

def select_balanced_sketch(targets: pd.DataFrame, *, max_cells: int, min_cells_per_stratum: int, max_cells_per_stratum: int, random_state: int) -> pd.DataFrame:
    """Sample cells evenly by sample, then by original uncertain cluster."""
    required = {'sample', 'cell_id', 'leiden_r1_3'}
    if not required.issubset(targets.columns):
        raise KeyError(f"Targets are missing: {', '.join(sorted(required - set(targets.columns)))}")
    if targets.empty:
        return targets.copy()
    work = targets.copy()
    work['sample'] = work['sample'].astype(str)
    work['cell_id'] = work['cell_id'].astype(str)
    work['leiden_r1_3'] = work['leiden_r1_3'].astype(str)
    sample_sizes = work.groupby('sample', sort=False).size().sort_index(key=lambda index: index.map(natural_key))
    sample_allocations = allocate_balanced_counts(sample_sizes.to_numpy(), int(max_cells))
    selected_parts: list[pd.DataFrame] = []
    for sample, sample_budget in zip(sample_sizes.index, sample_allocations):
        sample_rows = work.loc[work['sample'].eq(sample)]
        strata = sample_rows.groupby('leiden_r1_3', sort=False).size().sort_index(key=lambda index: index.map(natural_key))
        maximums = np.minimum(strata.to_numpy(dtype=int), int(max_cells_per_stratum))
        minimums = np.minimum(maximums, int(min_cells_per_stratum))
        if minimums.sum() <= sample_budget:
            allocations = minimums + allocate_balanced_counts(maximums - minimums, int(sample_budget - minimums.sum()))
        else:
            allocations = allocate_balanced_counts(minimums, int(sample_budget))
        for cluster, n_select in zip(strata.index, allocations):
            if n_select <= 0:
                continue
            stratum = sample_rows.loc[sample_rows['leiden_r1_3'].eq(cluster)]
            token = f'{random_state}|{sample}|{cluster}'.encode()
            seed = int.from_bytes(hashlib.sha256(token).digest()[:4], 'little')
            chosen = stratum.sample(n=int(n_select), random_state=seed) if len(stratum) > n_select else stratum
            selected_parts.append(chosen)
    result = pd.concat(selected_parts, ignore_index=True) if selected_parts else work.iloc[0:0].copy()
    return result.sort_values(['sample', 'leiden_r1_3', 'cell_id'], key=lambda column: column.map(natural_key), kind='stable').reset_index(drop=True)

def matrix_gene_features(matrix_path: str | Path) -> list[str]:
    """Read gene names without materializing the sparse count matrix."""
    matrix_path = Path(matrix_path)
    if matrix_path.name.endswith('.h5'):
        import h5py
        with h5py.File(matrix_path, 'r') as handle:
            features = handle['matrix/features']
            key = 'name' if 'name' in features else 'id'
            names = [value.decode() if isinstance(value, bytes) else str(value) for value in features[key][:]]
            types = [value.decode() if isinstance(value, bytes) else str(value) for value in features['feature_type'][:]] if 'feature_type' in features else ['Gene Expression'] * len(names)
    elif matrix_path.name.endswith('.zarr.zip'):
        from zarr.storage import ZipStore
        import zarr
        from .cell_clustering import zarr_feature_metadata_from_attrs
        with ZipStore(str(matrix_path), mode='r') as store:
            root = zarr.open_group(store=store, mode='r')
            matrix = root['cell_features'] if 'cell_features' in root else root
            names, types = zarr_feature_metadata_from_attrs(matrix, root)
    else:
        raise ValueError(f'Unsupported feature matrix: {matrix_path}')
    return list(dict.fromkeys((name for name, feature_type in zip(names, types) if is_gene_expression(feature_type, name))))

def common_gene_inventory(study_inventory: pd.DataFrame, root: str | Path) -> tuple[list[str], pd.DataFrame]:
    root = Path(root).expanduser().resolve()
    gene_sets: list[set[str]] = []
    rows: list[dict[str, object]] = []
    for row in study_inventory.to_dict('records'):
        path = _resolve_path(row.get('matrix_path'), root)
        if path is None:
            raise FileNotFoundError(row.get('matrix_path', ''))
        genes = matrix_gene_features(path)
        gene_sets.append(set(genes))
        rows.extend(({'dataset': row['dataset'], 'sample': row['sample'], 'gene': gene, 'measured': True} for gene in genes))
    common = sorted(set.intersection(*gene_sets) if gene_sets else set(), key=natural_key)
    inventory = pd.DataFrame(rows)
    if not inventory.empty:
        common_set = set(common)
        inventory['used_for_study_reclustering'] = inventory['gene'].isin(common_set)
    return (common, inventory)

def _load_target_adata(matrix_path: Path, targets: pd.DataFrame, common_genes: list[str], dataset: str, sample: str):
    adata = load_feature_matrix_as_adata(matrix_path)
    matrix_ids = pd.Index(adata.obs_names.astype(str))
    target_ids = targets['cell_id'].astype(str)
    positions = matrix_ids.get_indexer(target_ids)
    if (positions < 0).any():
        examples = ', '.join(target_ids.iloc[np.flatnonzero(positions < 0)[:5]].tolist())
        raise ValueError(f'{dataset}/{sample}: {int((positions < 0).sum()):,} target cell IDs are absent from the feature matrix; examples: {examples}')
    gene_symbols = pd.Index(adata.var.get('gene_symbol', adata.var_names).astype(str))
    first_position = pd.Series(np.arange(len(gene_symbols)), index=gene_symbols).groupby(level=0).first()
    gene_positions = first_position.reindex(common_genes).fillna(-1).to_numpy(dtype=int)
    if (gene_positions < 0).any():
        raise ValueError(f'{dataset}/{sample}: common-gene alignment failed')
    selected = adata[positions, gene_positions].copy()
    selected.obs['dataset'] = dataset
    selected.obs['sample'] = sample
    selected.obs['cell_id'] = target_ids.to_numpy()
    selected.obs_names = pd.Index([f'{sample}::{cell_id}' for cell_id in target_ids])
    selected.var_names = pd.Index(common_genes)
    return selected

def _normalize_log1p(adata, target_sum: float) -> None:
    import scanpy as sc
    sc.pp.normalize_total(adata, target_sum=float(target_sum))
    sc.pp.log1p(adata)

def _plot_umap(assignments: pd.DataFrame, cluster_key: str, path: Path, max_cells: int, random_state: int) -> Path | None:
    if not {'umap_1', 'umap_2', cluster_key}.issubset(assignments.columns):
        return None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot = assignments.dropna(subset=['umap_1', 'umap_2'])
    if plot.empty:
        return None
    if len(plot) > max_cells:
        plot = plot.sample(max_cells, random_state=random_state)
    labels = sorted(plot[cluster_key].astype(str).unique(), key=natural_key)
    codes = plot[cluster_key].astype(str).map({label: index for index, label in enumerate(labels)})
    figure, axis = plt.subplots(figsize=(7.5, 6.2))
    axis.scatter(plot['umap_1'], plot['umap_2'], c=codes, cmap='tab20', s=1, linewidths=0, alpha=0.75)
    suffix = '' if len(plot) == len(assignments) else ' | directly clustered sketch cells'
    axis.set(title=f"{assignments['dataset'].iloc[0]} | {cluster_key}{suffix}", xlabel='UMAP 1', ylabel='UMAP 2')
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(figure)
    return path

def _cluster_summaries(assignments: pd.DataFrame, resolutions: Iterable[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_parts: list[pd.DataFrame] = []
    composition_parts: list[pd.DataFrame] = []
    for resolution in resolutions:
        key = cluster_key_for_resolution(float(resolution)).replace('leiden_', 'recluster_')
        if key not in assignments:
            continue
        counts = assignments.groupby(key).size().rename('n_cells').reset_index().rename(columns={key: 'cluster'})
        counts.insert(0, 'cluster_key', key)
        counts.insert(0, 'resolution', float(resolution))
        counts.insert(0, 'dataset', assignments['dataset'].iloc[0])
        counts['fraction_of_study_targets'] = counts['n_cells'] / len(assignments)
        summary_parts.append(counts)
        composition = assignments.groupby([key, 'sample']).size().rename('n_cells').reset_index().rename(columns={key: 'cluster'})
        composition.insert(0, 'cluster_key', key)
        composition.insert(0, 'resolution', float(resolution))
        composition.insert(0, 'dataset', assignments['dataset'].iloc[0])
        composition['fraction_of_cluster'] = composition['n_cells'] / composition.groupby('cluster')['n_cells'].transform('sum')
        composition['fraction_of_sample_targets'] = composition['n_cells'] / composition.groupby('sample')['n_cells'].transform('sum')
        diagnostics = composition.groupby('cluster', as_index=False).agg(n_contributing_samples=('sample', 'nunique'), dominant_sample_fraction=('fraction_of_cluster', 'max'))
        entropy = composition.groupby('cluster')['fraction_of_cluster'].apply(lambda values: float(-(values * np.log(values)).sum() / np.log(len(values))) if len(values) > 1 else 0.0).rename('normalized_sample_composition_entropy').reset_index()
        counts = counts.merge(diagnostics, on='cluster', how='left').merge(entropy, on='cluster', how='left')
        summary_parts[-1] = counts
        composition_parts.append(composition)
    return (pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame(), pd.concat(composition_parts, ignore_index=True) if composition_parts else pd.DataFrame())

def _write_assignment_derived_outputs(assignments: pd.DataFrame, study_dir: Path, root: Path, config: ReclusteringConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Regenerate summaries and figures from saved per-cell assignments."""
    study = str(assignments['dataset'].iloc[0])
    resolution_keys = [(float(resolution), cluster_key_for_resolution(resolution).replace('leiden_', 'recluster_')) for resolution in config.leiden_resolutions]
    cluster_summary, sample_composition = _cluster_summaries(assignments, config.leiden_resolutions)
    cluster_summary.to_csv(study_dir / 'cluster_summary.csv', index=False)
    sample_composition.to_csv(study_dir / 'cluster_sample_composition.csv', index=False)
    for evidence_column, filename in (('leiden_r1_3', 'cluster_original_leiden_composition.csv'), ('marker_score_label', 'cluster_marker_score_composition.csv'), ('xoa_cluster_id', 'cluster_xoa_composition.csv')):
        parts = []
        if evidence_column in assignments:
            for resolution, key in resolution_keys:
                observed = assignments.loc[assignments[evidence_column].notna()]
                if observed.empty:
                    continue
                part = observed.groupby([key, evidence_column]).size().rename('n_cells').reset_index().rename(columns={key: 'cluster', evidence_column: 'evidence_label'})
                part.insert(0, 'cluster_key', key)
                part.insert(0, 'resolution', resolution)
                part.insert(0, 'dataset', study)
                part['fraction_of_cluster'] = part['n_cells'] / part.groupby('cluster')['n_cells'].transform('sum')
                parts.append(part)
        table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['dataset', 'resolution', 'cluster_key', 'cluster', 'evidence_label', 'n_cells', 'fraction_of_cluster'])
        table.to_csv(study_dir / filename, index=False)
    figure_rows: list[dict[str, object]] = []
    if config.run_umap and {'umap_1', 'umap_2'}.issubset(assignments.columns):
        for resolution, key in resolution_keys:
            figure_path = study_dir / 'figures' / f'umap__{key}.png'
            written = _plot_umap(assignments, key, figure_path, config.max_umap_plot_cells, config.random_state)
            if written:
                figure_rows.append({'dataset': study, 'resolution': resolution, 'cluster_key': key, 'figure_path': path_for_record(written, root)})
    figure_index = pd.DataFrame(figure_rows, columns=['dataset', 'resolution', 'cluster_key', 'figure_path'])
    figure_index.to_csv(study_dir / 'umap_figure_index.csv', index=False)
    return (cluster_summary, sample_composition, figure_index)

def _distance_weighted_labels(neighbor_indices: np.ndarray, neighbor_distances: np.ndarray, sketch_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(sketch_labels).astype(str)
    unique = np.array(sorted(set(labels), key=natural_key), dtype=object)
    codes = pd.Categorical(labels, categories=unique).codes
    neighbor_codes = codes[neighbor_indices]
    weights = 1.0 / np.maximum(neighbor_distances, 1e-08)
    scores = np.stack([np.sum(weights * (neighbor_codes == code), axis=1) for code in range(len(unique))], axis=1)
    order = np.argsort(scores, axis=1)
    top = order[:, -1]
    top_score = scores[np.arange(len(scores)), top]
    second_score = scores[np.arange(len(scores)), order[:, -2]] if len(unique) > 1 else np.zeros(len(scores))
    totals = scores.sum(axis=1)
    return (unique[top].astype(str), top_score / totals, (top_score - second_score) / totals)

def cell_key_checksum(table: pd.DataFrame) -> tuple[int, int]:
    """Return an order-independent checksum without constructing a large Python set."""
    keys = table[['sample', 'cell_id']].astype(str)
    hashes = pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype=np.uint64)
    return (int(hashes.sum(dtype=np.uint64)), int(np.bitwise_xor.reduce(hashes, initial=np.uint64(0))))

def summarize_cluster_sample_specificity(assignments: pd.DataFrame, cluster_key: str, *, min_cells_for_presence: int=20, broad_min_sample_fraction: float=0.5, broad_max_js_divergence: float=0.1, dominated_min_cluster_fraction: float=0.8, dominated_min_enrichment: float=2.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare each cluster's sample composition with the study baseline."""
    required = {'dataset', 'sample', cluster_key}
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise KeyError(f"Assignments are missing: {', '.join(missing)}")
    if min_cells_for_presence < 1:
        raise ValueError('min_cells_for_presence must be positive')
    table = assignments[['dataset', 'sample', cluster_key]].copy().rename(columns={cluster_key: 'cluster'})
    table = table.dropna(subset=['cluster'])
    if table.empty:
        return (pd.DataFrame(), pd.DataFrame())
    table['dataset'] = table['dataset'].astype(str)
    table['sample'] = table['sample'].astype(str)
    table['cluster'] = table['cluster'].astype(str)
    datasets = table['dataset'].unique()
    if len(datasets) != 1:
        raise ValueError('Sample-specificity analysis expects one study at a time')
    samples = sorted(table['sample'].unique(), key=natural_key)
    clusters = sorted(table['cluster'].unique(), key=natural_key)
    counts = table.groupby(['cluster', 'sample']).size().reindex(pd.MultiIndex.from_product([clusters, samples], names=['cluster', 'sample']), fill_value=0).rename('n_cells').reset_index()
    sample_totals = table.groupby('sample').size().rename('n_sample_target_cells')
    cluster_totals = table.groupby('cluster').size().rename('n_cluster_cells')
    counts = counts.join(sample_totals, on='sample').join(cluster_totals, on='cluster')
    counts['fraction_of_cluster'] = counts['n_cells'] / counts['n_cluster_cells']
    counts['fraction_of_sample_targets'] = counts['n_cells'] / counts['n_sample_target_cells']
    counts['baseline_sample_fraction'] = counts['n_sample_target_cells'] / len(table)
    counts['sample_enrichment'] = counts['fraction_of_cluster'] / counts['baseline_sample_fraction']
    counts['log2_sample_enrichment'] = np.log2(counts['sample_enrichment'].replace(0, np.nan))
    counts['meets_presence_threshold'] = counts['n_cells'].ge(int(min_cells_for_presence))
    counts.insert(0, 'cluster_key', cluster_key)
    counts.insert(0, 'dataset', datasets[0])
    summary_rows: list[dict[str, object]] = []
    baseline = counts.drop_duplicates('sample').set_index('sample')['baseline_sample_fraction'].reindex(samples).to_numpy(float)
    for cluster, cluster_table in counts.groupby('cluster', sort=False):
        cluster_table = cluster_table.set_index('sample').reindex(samples)
        proportions = cluster_table['fraction_of_cluster'].to_numpy(float)
        midpoint = 0.5 * (proportions + baseline)

        def kl_divergence(left: np.ndarray, right: np.ndarray) -> float:
            positive = left > 0
            return float(np.sum(left[positive] * np.log2(left[positive] / right[positive])))
        js_divergence = 0.5 * kl_divergence(proportions, midpoint) + 0.5 * kl_divergence(baseline, midpoint)
        positive = proportions > 0
        entropy = float(-np.sum(proportions[positive] * np.log(proportions[positive])))
        normalized_entropy = entropy / math.log(len(samples)) if len(samples) > 1 else np.nan
        dominant_sample = str(cluster_table['n_cells'].idxmax())
        dominant = cluster_table.loc[dominant_sample]
        present = cluster_table['meets_presence_threshold'].astype(bool)
        n_present = int(present.sum())
        fraction_present = n_present / len(samples)
        if len(samples) == 1:
            category = 'not_assessable'
        elif n_present <= 1:
            category = 'single_sample'
        elif float(dominant['fraction_of_cluster']) >= dominated_min_cluster_fraction and float(dominant['sample_enrichment']) >= dominated_min_enrichment:
            category = 'sample_dominated'
        elif fraction_present >= broad_min_sample_fraction and js_divergence <= broad_max_js_divergence:
            category = 'broadly_shared'
        else:
            category = 'shared_but_sample_enriched'
        summary_rows.append({'dataset': datasets[0], 'cluster_key': cluster_key, 'cluster': str(cluster), 'n_cluster_cells': int(cluster_table['n_cells'].sum()), 'n_study_samples': len(samples), 'n_contributing_samples': int(cluster_table['n_cells'].gt(0).sum()), 'n_samples_meeting_presence_threshold': n_present, 'fraction_study_samples_meeting_presence_threshold': fraction_present, 'dominant_sample': dominant_sample, 'dominant_sample_fraction': float(dominant['fraction_of_cluster']), 'dominant_sample_enrichment': float(dominant['sample_enrichment']), 'jensen_shannon_divergence_from_study_baseline': js_divergence, 'sample_composition_entropy': entropy, 'normalized_sample_composition_entropy': normalized_entropy, 'effective_n_samples': float(math.exp(entropy)), 'sample_specificity_category': category, 'min_cells_for_sample_presence': int(min_cells_for_presence)})
    summary = pd.DataFrame(summary_rows).sort_values('cluster', key=lambda values: values.map(natural_key)).reset_index(drop=True)
    return (summary, counts.reset_index(drop=True))

def run_study_reclustering(study: str, study_inventory: pd.DataFrame, hierarchy_dir: str | Path, root: str | Path, output_dir: str | Path, config: ReclusteringConfig) -> dict[str, object]:
    if config.mode != 'balanced_sketch':
        raise ValueError('Reclustering uses balanced_sketch.')
    'Recluster one study and write assignments plus compact diagnostics.'
    import anndata as ad
    import scanpy as sc
    study_started = time.perf_counter()
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir)
    hierarchy_dir = Path(hierarchy_dir)
    study_dir = output_dir / safe_name(study)
    assignment_path = study_dir / 'study_recluster_assignments.csv.gz'
    fingerprint = parameter_fingerprint(config)
    target_parts: list[pd.DataFrame] = []
    target_columns = {'dataset', 'sample', 'cell_id', 'hierarchy_layer', 'leiden_r1_3', 'leiden_r1_3_automatic_label', 'leiden_r1_3_automatic_confidence', 'author_label', 'marker_score_label', 'marker_score_top_group', 'top_score', 'runner_up', 'runner_up_score', 'score_margin', 'marker_score_call_status', 'xoa_cluster_id', 'xoa_automatic_label'}
    inventory_records = study_inventory.to_dict('records')
    _progress(f'{study}: reading reclustering targets from {len(inventory_records)} sample(s)')
    for sample_index, row in enumerate(inventory_records, start=1):
        _progress(f"{study}: target table [{sample_index}/{len(inventory_records)}] {row['sample']}")
        path = hierarchy_dir / safe_name(study) / safe_name(row['sample']) / 'cell_label_hierarchy.csv.gz'
        if not path.is_file():
            raise FileNotFoundError(path)
        hierarchy_header = list(pd.read_csv(path, nrows=0).columns)
        hierarchy = pd.read_csv(path, usecols=[column for column in hierarchy_header if column in target_columns], dtype={'cell_id': 'string'})
        target_parts.append(hierarchy.loc[hierarchy['hierarchy_layer'].eq('reclustering_target')].copy())
    targets = pd.concat(target_parts, ignore_index=True) if target_parts else pd.DataFrame()
    if targets.empty:
        _progress(f'{study}: no reclustering targets', started_at=study_started)
        return {'dataset': study, 'status': 'no_targets', 'parameter_fingerprint': fingerprint, 'n_target_cells': 0}
    participating_samples = set(targets['sample'].astype(str))
    study_inventory = study_inventory.loc[study_inventory['sample'].astype(str).isin(participating_samples)].copy()
    unavailable = study_inventory.loc[~study_inventory['ready_for_reclustering'].fillna(False), 'sample'].astype(str).tolist()
    if unavailable:
        raise FileNotFoundError(f"Target-bearing samples are not ready for reclustering: {', '.join(unavailable)}")
    _progress(f'{study}: {len(targets):,} target cells across {len(participating_samples)} participating sample(s)')
    if assignment_path.is_file() and config.skip_completed and (not config.overwrite_existing):
        _progress(f'{study}: validating existing assignment cache')
        expected_cluster_columns = [cluster_key_for_resolution(resolution).replace('leiden_', 'recluster_') for resolution in config.leiden_resolutions]
        header = list(pd.read_csv(assignment_path, nrows=0).columns)
        required_columns = {'dataset', 'sample', 'cell_id', 'assignment_method', 'parameter_fingerprint', *expected_cluster_columns}
        required_columns.update((f'{key}_{suffix}' for key in expected_cluster_columns for suffix in ('vote_fraction', 'vote_margin', 'projection_ambiguous')))
        if config.run_umap:
            required_columns.update({'umap_1', 'umap_2'})
        useful_columns = required_columns | {'leiden_r1_3', 'marker_score_label', 'xoa_cluster_id', 'umap_coordinate_source'}
        existing = pd.read_csv(assignment_path, usecols=[column for column in header if column in useful_columns], dtype={'cell_id': 'string'})
        valid_cache = required_columns.issubset(existing.columns) and len(existing) == len(targets) and existing['parameter_fingerprint'].astype(str).eq(fingerprint).all() and (not existing.duplicated(['dataset', 'sample', 'cell_id']).any()) and (cell_key_checksum(existing) == cell_key_checksum(targets))
        if valid_cache:
            _progress(f'{study}: valid assignment cache found; rebuilding compact outputs')
            study_dir.mkdir(parents=True, exist_ok=True)
            _write_assignment_derived_outputs(existing, study_dir, root, config)
            _progress(f'{study}: existing assignments reused', started_at=study_started)
            return {'dataset': study, 'status': 'skipped_existing', 'parameter_fingerprint': fingerprint, 'n_target_cells': len(targets), 'assignment_path': path_for_record(assignment_path, root)}
    if min(config.n_pcs, config.n_neighbors, config.projection_neighbors, config.projection_chunk_size) < 1:
        raise ValueError('PCA, neighbor, and projection sizes must be positive')
    if config.mode == 'balanced_sketch' and config.max_sketch_cells_per_study < 2:
        raise ValueError('balanced_sketch requires at least two sketch cells')
    _progress(f'{study}: finding genes measured in every participating sample')
    common_genes, gene_inventory = common_gene_inventory(study_inventory, root)
    if len(common_genes) < 2:
        raise ValueError(f'{study} has fewer than two common measured genes')
    _progress(f'{study}: {len(common_genes):,} common gene-expression features')
    direct_targets = select_balanced_sketch(targets, max_cells=config.max_sketch_cells_per_study, min_cells_per_stratum=config.min_sketch_cells_per_stratum, max_cells_per_stratum=config.max_sketch_cells_per_stratum, random_state=config.random_state)
    _progress(f'{study}: {config.mode} will directly cluster {len(direct_targets):,} of {len(targets):,} target cells')
    direct_parts = []
    direct_records = study_inventory.to_dict('records')
    for sample_index, row in enumerate(direct_records, start=1):
        sample_targets = direct_targets.loc[direct_targets['sample'].astype(str).eq(str(row['sample']))]
        if sample_targets.empty:
            continue
        _progress(f"{study}: loading direct cells [{sample_index}/{len(direct_records)}] {row['sample']} ({len(sample_targets):,} cells)")
        matrix_path = _resolve_path(row['matrix_path'], root)
        if matrix_path is None:
            raise FileNotFoundError(row['matrix_path'])
        direct_parts.append(_load_target_adata(matrix_path, sample_targets, common_genes, study, str(row['sample'])))
    _progress(f'{study}: combining {len(direct_parts)} direct expression matrix part(s)')
    direct = ad.concat(direct_parts, join='inner', merge='same', index_unique=None)
    _progress(f'{study}: normalizing and log-transforming {direct.n_obs:,} cells x {direct.n_vars:,} genes')
    _normalize_log1p(direct, config.normalization_target)
    n_pcs = min(config.n_pcs, direct.n_obs - 1, direct.n_vars - 1)
    if n_pcs < 2:
        raise ValueError(f'{study} cannot compute at least two PCs')
    pca_mean = np.asarray(direct.X.mean(axis=0)).ravel().astype(np.float32)
    _progress(f'{study}: computing PCA ({n_pcs} components)')
    sc.pp.pca(direct, n_comps=n_pcs, svd_solver='arpack', random_state=config.random_state)
    direct.uns['pca_projection_mean'] = pca_mean
    n_neighbors = min(config.n_neighbors, direct.n_obs - 1)
    _progress(f'{study}: building neighbor graph (n_neighbors={n_neighbors})')
    sc.pp.neighbors(direct, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=config.random_state)
    if config.run_umap:
        _progress(f'{study}: computing UMAP')
        sc.tl.umap(direct, random_state=config.random_state)
    resolution_keys: list[tuple[float, str]] = []
    for resolution in config.leiden_resolutions:
        key = cluster_key_for_resolution(resolution).replace('leiden_', 'recluster_')
        resolution_started = time.perf_counter()
        _progress(f'{study}: Leiden resolution {float(resolution):g} started')
        sc.tl.leiden(direct, resolution=float(resolution), key_added=key, random_state=config.random_state)
        _progress(f'{study}: Leiden resolution {float(resolution):g} finished ({direct.obs[key].nunique()} clusters)', started_at=resolution_started)
        resolution_keys.append((float(resolution), key))
    direct_assignments = pd.DataFrame({'dataset': study, 'sample': direct.obs['sample'].astype(str).to_numpy(), 'cell_id': direct.obs['cell_id'].astype(str).to_numpy(), 'assignment_method': 'sketch_leiden'})
    if 'X_umap' in direct.obsm:
        direct_assignments['umap_1'] = direct.obsm['X_umap'][:, 0]
        direct_assignments['umap_2'] = direct.obsm['X_umap'][:, 1]
        direct_assignments['umap_coordinate_source'] = 'direct_graph_umap'
    for _resolution, key in resolution_keys:
        direct_assignments[key] = direct.obs[key].astype(str).to_numpy()
        direct_assignments[f'{key}_vote_fraction'] = 1.0
        direct_assignments[f'{key}_vote_margin'] = 1.0
        direct_assignments[f'{key}_projection_ambiguous'] = False
    try:
        from pynndescent import NNDescent
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError('balanced_sketch projection requires pynndescent') from error
    projection_neighbors = min(config.projection_neighbors, direct.n_obs)
    index_neighbors = min(max(projection_neighbors, n_neighbors), direct.n_obs - 1)
    _progress(f'{study}: building projection index from {direct.n_obs:,} sketch cells')
    index = NNDescent(np.asarray(direct.obsm['X_pca'], dtype=np.float32), n_neighbors=index_neighbors, random_state=config.random_state)
    direct_key_set = set(zip(direct_assignments['sample'], direct_assignments['cell_id']))
    projection_parts: list[pd.DataFrame] = []
    loadings = np.asarray(direct.varm['PCs'], dtype=np.float32)
    pca_offset = pca_mean @ loadings
    projection_records = study_inventory.to_dict('records')
    for sample_index, row in enumerate(projection_records, start=1):
        sample = str(row['sample'])
        sample_targets = targets.loc[targets['sample'].astype(str).eq(sample)]
        direct_sample_ids = {cell_id for direct_sample, cell_id in direct_key_set if direct_sample == sample}
        remaining = sample_targets.loc[~sample_targets['cell_id'].astype(str).isin(direct_sample_ids)]
        if remaining.empty:
            continue
        sample_started = time.perf_counter()
        _progress(f'{study}: projection [{sample_index}/{len(projection_records)}] {sample} started ({len(remaining):,} cells)')
        matrix_path = _resolve_path(row['matrix_path'], root)
        if matrix_path is None:
            raise FileNotFoundError(row['matrix_path'])
        sample_adata = _load_target_adata(matrix_path, remaining, common_genes, study, sample)
        _normalize_log1p(sample_adata, config.normalization_target)
        for start in range(0, sample_adata.n_obs, config.projection_chunk_size):
            stop = min(start + config.projection_chunk_size, sample_adata.n_obs)
            values = sample_adata.X[start:stop]
            projected = values @ loadings - pca_offset
            projected = np.asarray(projected, dtype=np.float32)
            neighbor_indices, neighbor_distances = index.query(projected, k=projection_neighbors)
            chunk = pd.DataFrame({'dataset': study, 'sample': sample, 'cell_id': sample_adata.obs['cell_id'].iloc[start:stop].astype(str).to_numpy(), 'assignment_method': 'nearest_neighbor_projection', 'umap_coordinate_source': 'not_computed_for_projection'})
            for _resolution, key in resolution_keys:
                labels, vote_fraction, vote_margin = _distance_weighted_labels(neighbor_indices, neighbor_distances, direct.obs[key].astype(str).to_numpy())
                chunk[key] = labels
                chunk[f'{key}_vote_fraction'] = vote_fraction
                chunk[f'{key}_vote_margin'] = vote_margin
                chunk[f'{key}_projection_ambiguous'] = (vote_fraction < config.min_projection_vote_fraction) | (vote_margin < config.min_projection_vote_margin)
            projection_parts.append(chunk)
            _progress(f'{study}: {sample} projected {stop:,}/{sample_adata.n_obs:,} cells ({100 * stop / sample_adata.n_obs:.1f}%)')
        _progress(f'{study}: projection for {sample} finished', started_at=sample_started)
    assignments = pd.concat([direct_assignments, *projection_parts], ignore_index=True)
    evidence_columns = ['dataset', 'sample', 'cell_id', 'leiden_r1_3', 'leiden_r1_3_automatic_label', 'leiden_r1_3_automatic_confidence', 'author_label', 'marker_score_label', 'marker_score_top_group', 'top_score', 'runner_up', 'runner_up_score', 'score_margin', 'marker_score_call_status', 'xoa_cluster_id', 'xoa_automatic_label']
    assignments = assignments.merge(targets[[column for column in evidence_columns if column in targets.columns]], on=['dataset', 'sample', 'cell_id'], how='left', validate='one_to_one')
    assignments['parameter_fingerprint'] = fingerprint
    if len(assignments) != len(targets):
        raise AssertionError(f'{study}: assignment rows do not equal target rows')
    if assignments.duplicated(['dataset', 'sample', 'cell_id']).any():
        raise AssertionError(f'{study}: duplicate cell assignments')
    variance = np.asarray(direct.uns.get('pca', {}).get('variance_ratio', []), dtype=float)
    pca_variance = pd.DataFrame({'dataset': study, 'pc': np.arange(1, len(variance) + 1), 'variance_ratio': variance, 'cumulative_variance_ratio': np.cumsum(variance)})
    study_dir.mkdir(parents=True, exist_ok=True)
    _progress(f'{study}: writing {len(assignments):,} per-cell assignments and summary tables')
    assignments.to_csv(assignment_path, index=False, compression='gzip')
    _write_assignment_derived_outputs(assignments, study_dir, root, config)
    pca_variance.to_csv(study_dir / 'pca_variance.csv', index=False)
    gene_inventory.to_csv(study_dir / 'common_gene_inventory.csv', index=False)
    direct_targets[['dataset', 'sample', 'cell_id', 'leiden_r1_3']].to_csv(study_dir / 'sketch_membership.csv.gz', index=False, compression='gzip')
    model_path = study_dir / 'study_recluster_model.h5ad'
    if config.write_model_cache:
        _progress(f'{study}: writing model cache')
        cache = ad.AnnData(obs=direct.obs.copy())
        cache.obsm['X_pca'] = np.asarray(direct.obsm['X_pca'])
        if 'X_umap' in direct.obsm:
            cache.obsm['X_umap'] = np.asarray(direct.obsm['X_umap'])
        for key in ('distances', 'connectivities'):
            if key in direct.obsp:
                cache.obsp[key] = direct.obsp[key]
        cache.uns['neighbors'] = direct.uns.get('neighbors', {})
        cache.uns['pca'] = direct.uns.get('pca', {})
        cache.uns['parameter_fingerprint'] = fingerprint
        cache.write_h5ad(model_path, compression='gzip')
    _progress(f'{study}: reclustering finished successfully', started_at=study_started)
    return {'dataset': study, 'status': 'success', 'mode': config.mode, 'n_target_cells': len(targets), 'n_direct_clustered_cells': len(direct_assignments), 'n_projected_cells': int(len(assignments) - len(direct_assignments)), 'n_common_genes': len(common_genes), 'n_pcs_used': n_pcs, 'n_neighbors_used': n_neighbors, 'projection_neighbors_used': projection_neighbors, 'parameter_fingerprint': fingerprint, 'assignment_path': path_for_record(assignment_path, root), 'model_cache_path': path_for_record(model_path, root) if config.write_model_cache else ''}

def run_selected_studies(inventory: pd.DataFrame, hierarchy_dir: str | Path, root: str | Path, output_dir: str | Path, config: ReclusteringConfig) -> pd.DataFrame:
    """Run studies independently so one failure does not stop the rest."""
    rows: list[dict[str, object]] = []
    studies = sorted(inventory['dataset'].astype(str).unique(), key=natural_key)
    run_started = time.perf_counter()
    _progress(f'Reclustering run started: {len(studies)} study/studies, mode={config.mode}')
    for study_index, study in enumerate(studies, start=1):
        study_started = time.perf_counter()
        _progress(f'[{study_index}/{len(studies)}] Starting {study}')
        study_inventory = inventory.loc[inventory['dataset'].astype(str).eq(study)].copy()
        try:
            result = run_study_reclustering(study, study_inventory, hierarchy_dir, root, output_dir, config)
        except Exception as error:
            result = {'dataset': study, 'status': 'failed', 'mode': config.mode, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc(limit=8), 'parameter_fingerprint': parameter_fingerprint(config)}
            _progress(f'[{study_index}/{len(studies)}] {study} failed: {type(error).__name__}: {error}', started_at=study_started)
        else:
            _progress(f"[{study_index}/{len(studies)}] {study} status={result['status']}", started_at=study_started)
        rows.append(result)
        summary = pd.DataFrame(rows)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        summary.to_csv(Path(output_dir) / 'reclustering_run_summary.csv', index=False)
    figure_parts = []
    for study in sorted(inventory['dataset'].astype(str).unique(), key=natural_key):
        path = Path(output_dir) / safe_name(study) / 'umap_figure_index.csv'
        if path.is_file():
            figure_parts.append(pd.read_csv(path))
    pd.concat(figure_parts, ignore_index=True).to_csv(Path(output_dir) / 'umap_figure_index.csv', index=False) if figure_parts else pd.DataFrame(columns=['dataset', 'resolution', 'cluster_key', 'figure_path']).to_csv(Path(output_dir) / 'umap_figure_index.csv', index=False)
    _progress('Reclustering run finished', started_at=run_started)
    return pd.DataFrame(rows)
