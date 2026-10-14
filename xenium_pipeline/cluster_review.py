"""Review helpers for per-sample Xenium cell clustering outputs."""
from __future__ import annotations
import argparse
import json
import warnings
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from . import expression_annotation
from .cell_clustering import cluster_key_for_resolution, load_feature_matrix_as_adata
from .gene_panel_utils import natural_key
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANUAL_CLUSTER_ANNOTATION_COLUMNS = ['dataset', 'sample', 'clustering_source', 'cluster_key', 'cluster', 'manual_annotation', 'refined_annotation', 'broad_annotation', 'annotation_source', 'automatic_candidate_1', 'automatic_candidate_2', 'automatic_refined_annotation', 'review_selection', 'candidate_1_supporting_genes', 'candidate_2_supporting_genes', 'candidate_1_support_fraction', 'candidate_2_support_fraction', 'candidate_1_support_weight', 'candidate_2_support_weight', 'candidate_support_gap', 'candidate_support_ratio', 'top_two_support_fraction', 'heuristic_recommendation', 'confidence', 'notes', 'suggested_gene_marker_group', 'suggested_gene_hint_confidence', 'suggested_gene_hint_supporting_genes', 'suggested_marker_score_label', 'suggested_fraction_of_cluster', 'n_cluster_cells', 'n_matched_cells', 'updated_at']
CLUSTERING_SOURCE_LABELS = {'our_expression': 'Our Leiden clustering', 'xoa': 'XOA graph clustering'}
BROAD_CLUSTER_ANNOTATION_OPTIONS = ['', 'Tumor cell', 'Normal epithelia', 'T cell', 'B cell', 'NK cell', 'Myeloid / other immune', 'Stromal', 'Smooth muscle', 'Other / uncertain']
CELL_ID_COLUMN_CANDIDATES = ('cell_id', 'barcode', 'cell')
XOA_CLUSTER_COLUMN_CANDIDATES = ('cell_type', 'group', 'cluster', 'annotation', 'label', 'cell_label', 'type')
XOA_LABEL_SOURCE_TOKENS = ('xoa_graphclust',)
MISSING_CLUSTER_ASSIGNMENT_LABELS = {'', 'missing', 'unassigned', 'unannotated', 'nan', 'none'}

@dataclass(frozen=True)
class ClusterReviewConfig:
    default_cluster_key: str = 'leiden_r1_3'
    write_outputs: bool = True
    compare_to_marker_score_labels: bool = True
    review_top_labels_per_cluster: int = 8

def safe_name(value: object) -> str:
    text = str(value).strip()
    text = re.sub('[^A-Za-z0-9._-]+', '_', text)
    return text.strip('_') or 'sample'

def path_for_record(path: Path, root: str | Path=PROJECT_ROOT) -> str:
    root = Path(root).expanduser().resolve()
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path)

def path_for_display(path: Path, root: str | Path=PROJECT_ROOT) -> str:
    return path_for_record(path, root)

def resolve_existing_path(path_text: object, root: str | Path=PROJECT_ROOT) -> Path | None:
    if path_text is None or pd.isna(path_text):
        return None
    text = str(path_text).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path(root).expanduser().resolve() / path
    return path if path.exists() else None

def load_output_table(output_dir: str | Path, filename: str) -> pd.DataFrame:
    path = Path(output_dir).expanduser().resolve() / filename
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)

def split_path_list(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(';') if part.strip()]

def cluster_key_resolution(cluster_key: object) -> float | None:
    match = re.fullmatch('leiden_r(\\d+(?:_\\d+)?)', str(cluster_key).strip())
    if not match:
        return None
    try:
        return float(match.group(1).replace('_', '.'))
    except ValueError:
        return None

def select_cluster_key(cluster_keys: list[str], preferred_cluster_key: str | None='leiden_r1_3') -> str:
    if not cluster_keys:
        raise KeyError('No Leiden cluster columns are available')
    cluster_keys = sorted([str(key) for key in cluster_keys], key=natural_key)
    if preferred_cluster_key and preferred_cluster_key in cluster_keys:
        return str(preferred_cluster_key)
    preferred_resolution = cluster_key_resolution(preferred_cluster_key) if preferred_cluster_key else None
    if preferred_resolution is not None:
        keyed_resolutions = [(key, cluster_key_resolution(key)) for key in cluster_keys]
        numeric_keys = [(key, resolution) for key, resolution in keyed_resolutions if resolution is not None]
        if numeric_keys:
            return min(numeric_keys, key=lambda item: (abs(float(item[1]) - preferred_resolution), -float(item[1]), natural_key(item[0])))[0]
    return cluster_keys[0]

def normalize_cluster_key_selection(cluster_keys: str | list[str] | tuple[str, ...] | set[str] | None, fallback_cluster_key: str | None=None) -> tuple[str, ...]:
    if isinstance(cluster_keys, str):
        values = [cluster_keys]
    elif cluster_keys is None:
        values = [fallback_cluster_key] if fallback_cluster_key else []
    else:
        values = list(cluster_keys)
    normalized: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in normalized:
            normalized.append(text)
    if not normalized:
        raise ValueError('At least one Leiden cluster key must be selected')
    return tuple(normalized)

def build_cluster_count_overview(cluster_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cluster_summary.empty:
        return (pd.DataFrame(), pd.DataFrame())
    counts = cluster_summary.groupby(['dataset', 'sample', 'resolution'], dropna=False)['cluster'].nunique().rename('n_clusters').reset_index()
    counts['sample_key'] = counts['dataset'].astype(str) + ' | ' + counts['sample'].astype(str)
    pivot = counts.pivot(index='sample_key', columns='resolution', values='n_clusters')
    pivot = pivot.sort_index(key=lambda index: index.map(natural_key))
    pivot = pivot.reindex(sorted(pivot.columns, key=float), axis=1)
    return (counts, pivot)

def plot_cluster_count_heatmap(cluster_count_pivot: pd.DataFrame, output_path: str | Path) -> Path | None:
    if cluster_count_pivot.empty:
        return None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None
    output_path = Path(output_path)
    values = cluster_count_pivot.to_numpy(dtype=float)
    height = max(6, 0.28 * len(cluster_count_pivot) + 1.5)
    width = max(8, 0.9 * len(cluster_count_pivot.columns) + 6)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = ax.imshow(values, aspect='auto', interpolation='nearest', cmap='viridis')
    ax.set_xticks(np.arange(len(cluster_count_pivot.columns)))
    ax.set_xticklabels([f'{float(value):g}' for value in cluster_count_pivot.columns])
    ax.set_yticks(np.arange(len(cluster_count_pivot.index)))
    ax.set_yticklabels(cluster_count_pivot.index, fontsize=7)
    ax.set_xlabel('Leiden resolution')
    ax.set_ylabel('Sample')
    ax.set_title('Cluster count by sample and resolution')
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            if np.isfinite(value):
                ax.text(col_index, row_index, f'{int(value)}', ha='center', va='center', fontsize=7, color='white')
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label='n clusters')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path

def plot_default_cluster_size_fraction_histogram(cluster_summary: pd.DataFrame, default_cluster_key: str, output_path: str | Path) -> Path | None:
    if cluster_summary.empty or 'cluster_key' not in cluster_summary.columns:
        return None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None
    default_rows = cluster_summary.loc[cluster_summary['cluster_key'].astype(str).eq(str(default_cluster_key))].copy()
    if default_rows.empty:
        return None
    fractions = pd.to_numeric(default_rows['fraction'], errors='coerce').dropna()
    if fractions.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(fractions, bins=40, color='#4E79A7', edgecolor='white')
    ax.set_xlabel('Cluster fraction within sample')
    ax.set_ylabel('Clusters')
    ax.set_title(f'Default-resolution cluster size distribution ({default_cluster_key})')
    ax.grid(axis='y', alpha=0.25)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output_path

def write_cluster_count_overview(root: str | Path=PROJECT_ROOT, *, output_dir: str | Path | None=None, default_cluster_key: str='leiden_r1_3', write_outputs: bool=True) -> dict[str, pd.DataFrame | Path | None]:
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir or root / 'cell_clustering').expanduser().resolve()
    figure_dir = output_dir / 'figures'
    cluster_summary = load_output_table(output_dir, 'combined_cluster_summary.csv')
    counts, pivot = build_cluster_count_overview(cluster_summary)
    heatmap_path = None
    histogram_path = None
    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        figure_dir.mkdir(parents=True, exist_ok=True)
        counts.to_csv(output_dir / 'cluster_count_by_sample_resolution.csv', index=False)
        pivot.reset_index().to_csv(output_dir / 'cluster_count_by_resolution_matrix.csv', index=False)
        heatmap_path = plot_cluster_count_heatmap(pivot, figure_dir / 'cluster_count_by_resolution_heatmap.png')
        histogram_cluster_key = default_cluster_key
        if 'cluster_key' in cluster_summary.columns and (not cluster_summary.empty):
            try:
                histogram_cluster_key = select_cluster_key(sorted(cluster_summary['cluster_key'].dropna().astype(str).unique().tolist(), key=natural_key), default_cluster_key)
            except KeyError:
                histogram_cluster_key = default_cluster_key
        histogram_path = plot_default_cluster_size_fraction_histogram(cluster_summary, histogram_cluster_key, figure_dir / 'default_cluster_size_fraction_histogram.png')
    return {'combined_cluster_summary': cluster_summary, 'cluster_count_by_sample_resolution': counts, 'cluster_count_pivot': pivot, 'cluster_count_heatmap_path': heatmap_path, 'default_cluster_size_fraction_histogram_path': histogram_path}

def infer_umap_record_from_path(path: Path, root: str | Path=PROJECT_ROOT) -> dict[str, object]:
    stem = path.stem
    parts = stem.split('__', 3)
    if len(parts) == 4 and parts[0] == 'umap':
        dataset, sample, cluster_key = parts[1:]
    else:
        dataset, sample, cluster_key = ('', '', stem)
    return {'dataset': dataset, 'sample': sample, 'cluster_key': cluster_key, 'figure_path': path_for_record(path, root)}

def load_umap_figure_index(output_dir: str | Path, *, root: str | Path=PROJECT_ROOT, scan_figures: bool=True) -> pd.DataFrame:
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    records: list[dict[str, object]] = []
    index_path = output_dir / 'umap_figure_index.csv'
    if index_path.is_file():
        table = pd.read_csv(index_path).fillna('')
        for _, row in table.iterrows():
            record = row.to_dict()
            record.setdefault('dataset', '')
            record.setdefault('sample', '')
            record.setdefault('cluster_key', '')
            record.setdefault('figure_path', '')
            record['source'] = 'umap_figure_index.csv'
            records.append(record)
    summary = load_output_table(output_dir, 'clustering_run_summary.csv')
    if not summary.empty:
        for _, row in summary.iterrows():
            path_values = split_path_list(row.get('umap_figure_paths', ''))
            if not path_values:
                path_values = split_path_list(row.get('umap_figure_path', ''))
            if not path_values:
                continue
            key_values = split_path_list(row.get('umap_cluster_key', ''))
            for index, path_text in enumerate(path_values):
                figure_path = Path(path_text)
                inferred = infer_umap_record_from_path(figure_path, root)
                cluster_key = key_values[index] if index < len(key_values) else str(inferred.get('cluster_key', ''))
                records.append({'dataset': row.get('dataset', inferred.get('dataset', '')), 'sample': row.get('sample', inferred.get('sample', '')), 'cluster_key': cluster_key, 'n_cells_plotted': row.get('n_cells_clustered', ''), 'n_clusters': '', 'figure_path': path_text, 'source': 'clustering_run_summary.csv'})
    if scan_figures:
        figure_dir = output_dir / 'figures'
        for path in figure_dir.glob('umap__*.png'):
            record = infer_umap_record_from_path(path, root)
            record['source'] = 'figures folder'
            records.append(record)
    if not records:
        return pd.DataFrame(columns=['dataset', 'sample', 'cluster_key', 'figure_path', 'source'])
    table = pd.DataFrame(records).fillna('')
    for column in ['dataset', 'sample', 'cluster_key', 'figure_path', 'source']:
        if column not in table.columns:
            table[column] = ''
    resolved_path_keys: list[str] = []
    for value in table['figure_path']:
        resolved = resolve_existing_path(value, root)
        resolved_path_keys.append(str(resolved.resolve()) if resolved is not None else str(value))
    table['_resolved_figure_path'] = resolved_path_keys
    source_priority = {'umap_figure_index.csv': 0, 'clustering_run_summary.csv': 1, 'figures folder': 2}
    table['_source_priority'] = table['source'].map(source_priority).fillna(99)
    table = table.sort_values(['dataset', 'sample', 'cluster_key', '_source_priority', 'source'], key=lambda column: column.map(natural_key) if column.name != '_source_priority' else column).drop_duplicates('_resolved_figure_path', keep='first').drop(columns=['_resolved_figure_path', '_source_priority']).reset_index(drop=True)
    return table

def umap_cluster_key_options(figure_index: pd.DataFrame) -> list[str]:
    if figure_index.empty or 'cluster_key' not in figure_index.columns:
        return []
    return sorted(figure_index['cluster_key'].dropna().astype(str).loc[lambda values: values.ne('')].unique().tolist(), key=natural_key)

def build_figure_grid_html(figure_index: pd.DataFrame, *, n_columns: int=4, root: str | Path=PROJECT_ROOT) -> str:
    from html import escape
    if figure_index.empty or 'figure_path' not in figure_index.columns:
        return '<p>No UMAP figures are available.</p>'
    ready = figure_index.loc[figure_index['figure_path'].astype(str).ne('')].copy()
    if ready.empty:
        return '<p>No UMAP figures were created.</p>'
    root = Path(root).expanduser().resolve()
    cards: list[str] = []
    for row in ready.itertuples(index=False):
        path = resolve_existing_path(row.figure_path, root) or Path(str(row.figure_path))
        src = path_for_display(path, root)
        caption = f'{escape(str(row.dataset))}<br>{escape(str(row.sample))}<br>{escape(str(row.cluster_key))}'
        cards.append(f'<figure style="margin:0; padding:0; break-inside:avoid;"><a href="{escape(src)}" target="_blank" title="Open saved PNG"><img src="{escape(src)}" style="width:100%; height:auto; border:1px solid #d9d9d9; background:white;"></a><figcaption style="font-size:11px; line-height:1.25; margin-top:4px; color:#333;">{caption}<br><a href="{escape(src)}" target="_blank">open saved PNG</a></figcaption></figure>')
    return f'<div style="display:grid; grid-template-columns:repeat({int(n_columns)}, minmax(0, 1fr)); gap:18px; align-items:start; width:100%;">' + ''.join(cards) + '</div>'

def display_umap_figure_grid_by_resolution(figure_index: pd.DataFrame, *, n_columns: int=4, root: str | Path=PROJECT_ROOT, default_cluster_key: str | None=None) -> None:
    try:
        import ipywidgets as widgets
        from IPython.display import HTML, clear_output, display
    except ModuleNotFoundError:
        print('ipywidgets is not available; showing all UMAP figures.')
        try:
            from IPython.display import HTML, display
            display(HTML(build_figure_grid_html(figure_index, n_columns=n_columns, root=root)))
        except ModuleNotFoundError:
            print(build_figure_grid_html(figure_index, n_columns=n_columns, root=root))
        return
    ready = figure_index.copy()
    if ready.empty:
        display(HTML(build_figure_grid_html(ready, n_columns=n_columns, root=root)))
        return
    keys = umap_cluster_key_options(ready)
    if not keys:
        display(HTML(build_figure_grid_html(ready, n_columns=n_columns, root=root)))
        return
    selected_key = select_cluster_key(keys, default_cluster_key) if default_cluster_key else keys[0]
    options = [(key, key) for key in keys] + [('All resolutions', '__all__')]
    dropdown = widgets.Dropdown(options=options, value=selected_key, description='Resolution:', layout=widgets.Layout(width='320px'))
    output = widgets.Output()

    def render(*_args: object) -> None:
        value = str(dropdown.value)
        selected = ready if value == '__all__' else ready.loc[ready['cluster_key'].astype(str).eq(value)].copy()
        with output:
            clear_output(wait=True)
            label = 'all resolutions' if value == '__all__' else value
            print(f'Showing {len(selected):,} saved UMAP figure(s) for {label}.')
            display(HTML(build_figure_grid_html(selected, n_columns=n_columns, root=root)))
    dropdown.observe(render, names='value')
    display(dropdown, output)
    render()

def expected_marker_label_path(annotation_dir: str | Path, dataset: str, sample: str) -> Path:
    return Path(annotation_dir) / safe_name(dataset) / safe_name(sample) / 'cell_marker_labels.csv.gz'

def resolved_marker_label_path(root: str | Path, annotation_dir: str | Path, dataset: str, sample: str) -> Path | None:
    root = Path(root).expanduser().resolve()
    annotation_dir = Path(annotation_dir).expanduser().resolve()
    summary_path = annotation_dir / 'annotation_run_summary.csv'
    if summary_path.is_file():
        summary = pd.read_csv(summary_path)
        rows = summary.loc[summary['dataset'].astype(str).eq(str(dataset)) & summary['sample'].astype(str).eq(str(sample))]
        for value in rows.get('cell_marker_labels_path', pd.Series(dtype=object)).dropna():
            path = resolve_existing_path(value, root)
            if path is not None:
                return path
    path = expected_marker_label_path(annotation_dir, dataset, sample)
    return path if path.is_file() else None

def load_marker_score_labels(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    table = pd.read_csv(path)
    if 'cell_id' not in table.columns or 'group' not in table.columns:
        raise KeyError(f'{path} must contain cell_id and group columns')
    labels = table[['cell_id', 'group']].copy()
    labels['cell_id'] = labels['cell_id'].astype(str)
    labels['group'] = labels['group'].fillna('Missing').astype(str)
    return labels.drop_duplicates('cell_id')

def available_cluster_keys(clusters: pd.DataFrame) -> list[str]:
    keys = [column for column in clusters.columns if str(column).startswith('leiden_r')]
    return sorted(keys, key=natural_key)

def cluster_size_summary(clusters: pd.DataFrame, cluster_key: str) -> pd.DataFrame:
    if cluster_key not in clusters.columns:
        raise KeyError(f'{cluster_key} is not in cluster table')
    counts = clusters[cluster_key].fillna('Missing').astype(str).value_counts(dropna=False).rename_axis('cluster').rename('n_cluster_cells').reset_index()
    counts = counts.sort_values('cluster', key=lambda column: column.map(natural_key)).reset_index(drop=True)
    for column, value in {'top_marker_score_label': '', 'top_label_cells': 0, 'top_label_fraction_of_cluster': np.nan, 'second_marker_score_label': '', 'second_label_cells': 0, 'second_label_fraction_of_cluster': np.nan, 'n_marker_score_labels_present': 0}.items():
        counts[column] = value
    return counts

def cluster_label_pair_summary(clusters: pd.DataFrame, labels: pd.DataFrame, cluster_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_labels = clusters[['cell_id', cluster_key]].copy()
    cluster_labels['cell_id'] = cluster_labels['cell_id'].astype(str)
    cluster_labels[cluster_key] = cluster_labels[cluster_key].fillna('Missing').astype(str)
    marker_labels = labels[['cell_id', 'group']].copy()
    marker_labels['cell_id'] = marker_labels['cell_id'].astype(str)
    marker_labels['group'] = marker_labels['group'].fillna('Missing').astype(str)
    matched = cluster_labels.merge(marker_labels, on='cell_id', how='inner')
    if matched.empty:
        return (matched, pd.DataFrame())
    pairs = matched.groupby([cluster_key, 'group'], dropna=False).size().rename('n_cells').reset_index().rename(columns={cluster_key: 'cluster', 'group': 'marker_score_label'}).sort_values('n_cells', ascending=False)
    cluster_totals = matched.groupby(cluster_key, dropna=False).size().rename('n_cluster_cells').reset_index().rename(columns={cluster_key: 'cluster'})
    label_totals = matched.groupby('group', dropna=False).size().rename('n_label_cells').reset_index().rename(columns={'group': 'marker_score_label'})
    pairs = pairs.merge(cluster_totals, on='cluster', how='left')
    pairs = pairs.merge(label_totals, on='marker_score_label', how='left')
    pairs['fraction_of_cluster'] = pairs['n_cells'] / pairs['n_cluster_cells'].replace(0, np.nan)
    pairs['fraction_of_label'] = pairs['n_cells'] / pairs['n_label_cells'].replace(0, np.nan)
    return (matched, pairs)

def cross_tables_for_cluster_key(clusters: pd.DataFrame, labels: pd.DataFrame, cluster_key: str) -> dict[str, pd.DataFrame]:
    matched, pairs = cluster_label_pair_summary(clusters, labels, cluster_key)
    if matched.empty:
        empty = pd.DataFrame()
        return {'matched': matched, 'pairs': pairs, 'counts': empty, 'row_fraction': empty, 'column_fraction': empty, 'cluster_summary': cluster_size_summary(clusters, cluster_key)}
    counts = pd.crosstab(matched[cluster_key].astype(str), matched['group'].astype(str), dropna=False)
    counts = counts.reindex(sorted(counts.index, key=natural_key))
    counts = counts.reindex(sorted(counts.columns, key=natural_key), axis=1)
    counts.index.name = 'cluster'
    counts.columns.name = 'marker_score_label'
    row_fraction = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    column_fraction = counts.div(counts.sum(axis=0).replace(0, np.nan), axis=1)
    rows: list[dict[str, object]] = []
    for cluster, count_row in counts.iterrows():
        nonzero = count_row[count_row > 0].sort_values(ascending=False)
        top_label = str(nonzero.index[0]) if len(nonzero) else ''
        second_label = str(nonzero.index[1]) if len(nonzero) > 1 else ''
        n_cluster_cells = int(count_row.sum())
        top_n = int(nonzero.iloc[0]) if len(nonzero) else 0
        second_n = int(nonzero.iloc[1]) if len(nonzero) > 1 else 0
        rows.append({'cluster': str(cluster), 'n_cluster_cells': n_cluster_cells, 'top_marker_score_label': top_label, 'top_label_cells': top_n, 'top_label_fraction_of_cluster': top_n / n_cluster_cells if n_cluster_cells else np.nan, 'second_marker_score_label': second_label, 'second_label_cells': second_n, 'second_label_fraction_of_cluster': second_n / n_cluster_cells if n_cluster_cells else np.nan, 'n_marker_score_labels_present': int((count_row > 0).sum())})
    cluster_summary = pd.DataFrame(rows).sort_values('cluster', key=lambda column: column.map(natural_key)).reset_index(drop=True)
    return {'matched': matched, 'pairs': pairs, 'counts': counts, 'row_fraction': row_fraction, 'column_fraction': column_fraction, 'cluster_summary': cluster_summary}

def resolve_cluster_path_from_row(row: pd.Series, root: str | Path, output_dir: str | Path) -> Path | None:
    for column in ('cell_clusters_path', 'cluster_path'):
        if column in row and pd.notna(row[column]):
            path = resolve_existing_path(row[column], root)
            if path is not None:
                return path
    expected = Path(output_dir) / safe_name(row.get('dataset', '')) / safe_name(row.get('sample', '')) / 'cell_clusters.csv.gz'
    return expected if expected.is_file() else None

def resolve_feature_matrix_path_from_row(row: pd.Series, root: str | Path, output_dir: str | Path) -> Path | None:
    for column in ('matrix_path', 'feature_matrix_path'):
        if column in row and pd.notna(row[column]):
            path = resolve_existing_path(row[column], root)
            if path is not None:
                return path
    input_path = Path(output_dir) / 'clustering_input_table.csv'
    if input_path.is_file():
        try:
            input_table = pd.read_csv(input_path)
        except Exception:
            input_table = pd.DataFrame()
        if not input_table.empty and {'dataset', 'sample', 'matrix_path'}.issubset(input_table.columns):
            rows = input_table.loc[input_table['dataset'].astype(str).eq(str(row.get('dataset', ''))) & input_table['sample'].astype(str).eq(str(row.get('sample', '')))]
            for value in rows['matrix_path'].dropna():
                path = resolve_existing_path(value, root)
                if path is not None:
                    return path
    return None

def build_cluster_review_inventory(root: str | Path=PROJECT_ROOT, *, output_dir: str | Path | None=None, annotation_dir: str | Path | None=None) -> pd.DataFrame:
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir or root / 'cell_clustering').expanduser().resolve()
    annotation_dir = Path(annotation_dir or root / 'cell_annotation').expanduser().resolve()
    summary = load_output_table(output_dir, 'clustering_run_summary.csv')
    if summary.empty:
        return pd.DataFrame()
    ready_statuses = {'success', 'skipped_existing', 'repaired_missing_outputs'}
    rows: list[dict[str, object]] = []
    for _, row in summary.loc[summary['status'].astype(str).isin(ready_statuses)].iterrows():
        dataset = str(row['dataset'])
        sample = str(row['sample'])
        cluster_path = resolve_cluster_path_from_row(row, root, output_dir)
        matrix_path = resolve_feature_matrix_path_from_row(row, root, output_dir)
        label_path = resolved_marker_label_path(root, annotation_dir, dataset, sample)
        has_cluster_output = cluster_path is not None
        has_feature_matrix = matrix_path is not None
        has_marker_score_labels = label_path is not None
        rows.append({'dataset': dataset, 'sample': sample, 'sample_key': f'{dataset} | {sample}', 'status': row.get('status', ''), 'cluster_path': path_for_record(cluster_path, root) if cluster_path is not None else '', 'matrix_path': path_for_record(matrix_path, root) if matrix_path is not None else '', 'marker_label_path': path_for_record(label_path, root) if label_path is not None else '', 'has_cluster_output': has_cluster_output, 'has_feature_matrix': has_feature_matrix, 'has_marker_score_labels': has_marker_score_labels, 'ready_for_review': has_cluster_output and has_marker_score_labels, 'ready_for_annotation': has_cluster_output, 'ready_for_marker_gene_review': has_cluster_output and has_feature_matrix})
    rows = sorted(rows, key=lambda item: (not bool(item['ready_for_review']), natural_key(item['dataset']), natural_key(item['sample'])))
    return pd.DataFrame(rows)

def table_header(path: Path | None) -> list[str]:
    if path is None or not path.is_file():
        return []
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []

def infer_table_column(columns: list[str], candidates: tuple[str, ...]) -> str:
    lookup = {str(column).lower(): str(column) for column in columns}
    return next((lookup[str(candidate).lower()] for candidate in candidates if str(candidate).lower() in lookup), '')

def preview_label_sources(path: Path) -> set[str]:
    if 'label_source' not in table_header(path):
        return set()
    try:
        preview = pd.read_csv(path, nrows=500, usecols=['label_source'])
    except Exception:
        return set()
    return set(preview['label_source'].dropna().astype(str).str.lower().unique())

def discover_xoa_cluster_source(sample: str, sample_dir: Path | None) -> dict[str, object]:
    candidates: list[tuple[Path, str, str]] = []
    if sample_dir is not None and sample_dir.is_dir():
        dedicated = sample_dir / f'{sample}_cells_with_cluster_labels.csv'
        if dedicated.is_file():
            candidates.append((dedicated, 'cells_with_cluster_labels', 'dedicated_xoa_file'))
        annotated = sample_dir / f'{sample}_cells_with_annotations.csv'
        if annotated.is_file():
            sources = preview_label_sources(annotated)
            if any((any((token in source for token in XOA_LABEL_SOURCE_TOKENS)) for source in sources)):
                candidates.append((annotated, 'cells_with_annotations', '|'.join(sorted(sources))))
    if not candidates:
        return {'path': '', 'source_type': '', 'source_reason': 'not_found', 'cell_id_column': '', 'cluster_column': ''}
    path, source_type, source_reason = candidates[0]
    columns = table_header(path)
    return {'path': str(path), 'source_type': source_type, 'source_reason': source_reason, 'cell_id_column': infer_table_column(columns, CELL_ID_COLUMN_CANDIDATES), 'cluster_column': infer_table_column(columns, XOA_CLUSTER_COLUMN_CANDIDATES)}

def build_cluster_source_inventory(root: str | Path=PROJECT_ROOT, *, output_dir: str | Path | None=None, annotation_dir: str | Path | None=None, gene_panel_review_dir: str | Path | None=None) -> pd.DataFrame:
    """Inventory our Leiden partitions and XOA graph-cluster partitions.

    The returned rows point to the original cluster-assignment tables. XOA
    labels are not copied or interpreted as biological labels here.
    """
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir or root / 'cell_clustering').expanduser().resolve()
    annotation_dir = Path(annotation_dir or root / 'cell_annotation').expanduser().resolve()
    gene_panel_review_dir = Path(gene_panel_review_dir or root / 'gene_panel_review').expanduser().resolve()
    leiden_inventory = build_cluster_review_inventory(root, output_dir=output_dir, annotation_dir=annotation_dir)
    rows: list[dict[str, object]] = []
    sample_records: dict[tuple[str, str], dict[str, object]] = {}
    for review_row in leiden_inventory.itertuples(index=False):
        dataset = str(review_row.dataset)
        sample = str(review_row.sample)
        sample_key = (dataset, sample)
        sample_records.setdefault(sample_key, {'dataset': dataset, 'sample': sample, 'sample_dir': '', 'matrix_path': review_row.matrix_path})
        cluster_path = resolve_existing_path(review_row.cluster_path, root)
        if cluster_path is None:
            continue
        try:
            clusters = pd.read_csv(cluster_path, nrows=5)
        except Exception:
            continue
        for cluster_key in available_cluster_keys(clusters):
            rows.append({'dataset': dataset, 'sample': sample, 'sample_key': f'{dataset} | {sample}', 'clustering_source': 'our_expression', 'clustering_source_label': CLUSTERING_SOURCE_LABELS['our_expression'], 'cluster_key': cluster_key, 'cluster_path': path_for_record(cluster_path, root), 'matrix_path': str(review_row.matrix_path), 'cell_id_column': 'cell_id', 'cluster_column': cluster_key, 'source_type': 'cell_clusters', 'source_reason': 'notebook_05_leiden', 'has_cluster_output': True, 'has_feature_matrix': bool(review_row.has_feature_matrix), 'ready_for_marker_gene_review': bool(review_row.ready_for_marker_gene_review)})
    clustering_input_path = output_dir / 'clustering_input_table.csv'
    if clustering_input_path.is_file():
        try:
            clustering_inputs = pd.read_csv(clustering_input_path)
        except Exception:
            clustering_inputs = pd.DataFrame()
        if {'dataset', 'sample'}.issubset(clustering_inputs.columns):
            for input_row in clustering_inputs.itertuples(index=False):
                dataset = str(input_row.dataset)
                sample = str(input_row.sample)
                record = sample_records.setdefault((dataset, sample), {'dataset': dataset, 'sample': sample, 'sample_dir': '', 'matrix_path': ''})
                matrix_value = getattr(input_row, 'matrix_path', '')
                if str(matrix_value).strip():
                    record['matrix_path'] = matrix_value
    sample_summary_path = gene_panel_review_dir / 'sample_panel_summary.csv'
    if sample_summary_path.is_file():
        try:
            sample_summary = pd.read_csv(sample_summary_path)
        except Exception:
            sample_summary = pd.DataFrame()
        if {'dataset', 'sample'}.issubset(sample_summary.columns):
            if 'status' in sample_summary.columns:
                sample_summary = sample_summary.loc[sample_summary['status'].astype(str).eq('ok')].copy()
            for sample_row in sample_summary.itertuples(index=False):
                dataset = str(sample_row.dataset)
                sample = str(sample_row.sample)
                record = sample_records.setdefault((dataset, sample), {'dataset': dataset, 'sample': sample, 'sample_dir': '', 'matrix_path': ''})
                record['sample_dir'] = getattr(sample_row, 'sample_dir', '')
    for record in sample_records.values():
        dataset = str(record['dataset'])
        sample = str(record['sample'])
        sample_dir = resolve_existing_path(record.get('sample_dir', ''), root)
        xoa = discover_xoa_cluster_source(sample, sample_dir)
        cluster_path = resolve_existing_path(xoa['path'], root)
        matrix_path = resolve_existing_path(record.get('matrix_path', ''), root)
        has_cluster_output = cluster_path is not None and bool(str(xoa['cell_id_column']).strip()) and bool(str(xoa['cluster_column']).strip())
        rows.append({'dataset': dataset, 'sample': sample, 'sample_key': f'{dataset} | {sample}', 'clustering_source': 'xoa', 'clustering_source_label': CLUSTERING_SOURCE_LABELS['xoa'], 'cluster_key': 'xoa_cluster_id', 'cluster_path': path_for_record(cluster_path, root) if cluster_path is not None else '', 'matrix_path': path_for_record(matrix_path, root) if matrix_path is not None else '', 'cell_id_column': str(xoa['cell_id_column']), 'cluster_column': str(xoa['cluster_column']), 'source_type': str(xoa['source_type']), 'source_reason': str(xoa['source_reason']), 'has_cluster_output': has_cluster_output, 'has_feature_matrix': matrix_path is not None, 'ready_for_marker_gene_review': has_cluster_output and matrix_path is not None})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(['clustering_source', 'dataset', 'sample', 'cluster_key'], key=lambda column: column.map(natural_key), kind='stable').reset_index(drop=True)

def load_cluster_source_assignments(source_row: pd.Series, *, root: str | Path=PROJECT_ROOT) -> tuple[pd.DataFrame, Path]:
    root = Path(root).expanduser().resolve()
    cluster_path = resolve_existing_path(source_row.get('cluster_path', ''), root)
    if cluster_path is None:
        raise FileNotFoundError('Cluster-assignment table is missing')
    cell_id_column = str(source_row.get('cell_id_column', ''))
    cluster_column = str(source_row.get('cluster_column', ''))
    cluster_key = str(source_row.get('cluster_key', ''))
    if not cell_id_column or not cluster_column or (not cluster_key):
        raise KeyError('Cluster source is missing a cell-ID column, cluster column, or cluster key')
    assignments = pd.read_csv(cluster_path, usecols=[cell_id_column, cluster_column])
    assignments = assignments.rename(columns={cell_id_column: 'cell_id', cluster_column: cluster_key})
    assignments['cell_id'] = assignments['cell_id'].astype(str)
    assignments[cluster_key] = assignments[cluster_key].fillna('Missing').astype(str)
    conflicting = assignments.groupby('cell_id', dropna=False)[cluster_key].nunique(dropna=False).gt(1)
    if conflicting.any():
        raise ValueError(f'{int(conflicting.sum()):,} cell IDs have conflicting {cluster_key} assignments')
    return (assignments.drop_duplicates('cell_id', keep='first'), cluster_path)

def normalize_cluster_source_cell_ids(values: pd.Series, method: str) -> pd.Series:
    """Normalize cell IDs using a small set of collision-checked strategies."""
    normalized = values.fillna('').astype(str).str.strip()
    if method == 'trimmed':
        return normalized
    normalized = normalized.str.lower()
    if method == 'casefold':
        return normalized
    if method not in {'xenium_barcode', 'xenium_integer_prefix'}:
        raise ValueError(f'Unsupported cell-ID matching method: {method}')
    normalized = normalized.str.replace('^b[\'\\"](.*)[\'\\"]$', '\\1', regex=True)
    if method == 'xenium_integer_prefix':
        integer_parts = normalized.str.extract('^([a-p]{8})[-_](\\d+)(?:\\.0)?$', expand=True)
        valid_integer_barcode = integer_parts[0].notna() & integer_parts[1].eq('1')
        if valid_integer_barcode.any():
            valid_index = integer_parts.index[valid_integer_barcode]
            prefixes = integer_parts.loc[valid_index, 0].astype(str).tolist()
            encoded = ''.join(prefixes).encode('ascii')
            digits = np.frombuffer(encoded, dtype=np.uint8).reshape(-1, 8).astype(np.uint64) - np.uint64(ord('a'))
            hexadecimal_weights = np.array([16 ** power for power in range(7, -1, -1)], dtype=np.uint64)
            integer_ids = digits @ hexadecimal_weights
            normalized.loc[valid_index] = pd.Series(integer_ids.astype(str), index=valid_index)
        return normalized
    barcode_parts = normalized.str.extract('^([a-p]{8})(?:[-_](\\d+)(?:\\.0)?)?$', expand=True)
    valid_barcode = barcode_parts[0].notna()
    if valid_barcode.any():
        suffix = barcode_parts.loc[valid_barcode, 1].fillna('1')
        normalized.loc[valid_barcode] = barcode_parts.loc[valid_barcode, 0] + '-' + suffix
    return normalized

def load_feature_matrix_cell_ids(matrix_path: str | Path) -> pd.Index:
    """Load only cell IDs from a Xenium H5 or Zarr feature-matrix source."""
    matrix_path = Path(matrix_path).expanduser().resolve()
    if matrix_path.name.endswith('.h5'):
        try:
            import h5py
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError('Reading H5 cell IDs requires h5py') from error
        with h5py.File(matrix_path, 'r') as handle:
            barcodes = handle['matrix']['barcodes'][:]
        return pd.Index([value.decode('utf-8') if isinstance(value, bytes) else str(value) for value in barcodes], dtype='object')
    if matrix_path.name.endswith('.zarr.zip'):
        cell_ids = expression_annotation.cell_ids_from_cells_zarr(matrix_path.parent)
        if cell_ids is None:
            cells = expression_annotation.read_optional_cells_csv(expression_annotation.find_cells_csv(matrix_path.parent))
            if cells is None:
                raise FileNotFoundError('Zarr matrix requires cells.zarr.zip or a cell CSV with cell_id')
            cell_ids = cells['cell_id'].astype(str).tolist()
        return pd.Index(list(map(str, cell_ids)), dtype='object')
    raise ValueError(f'Unsupported feature matrix for cell-ID preview: {matrix_path}')

def cluster_source_candidate_id_columns(source_row: pd.Series, cluster_path: Path) -> list[str]:
    """Return configured and conventional ID columns found in a source table."""
    columns = table_header(cluster_path)
    lookup = {str(column).lower(): str(column) for column in columns}
    configured = str(source_row.get('cell_id_column', '')).strip()
    candidate_columns: list[str] = []
    configured_actual = lookup.get(configured.lower(), '') if configured else ''
    if configured_actual:
        candidate_columns.append(configured_actual)
    for candidate in CELL_ID_COLUMN_CANDIDATES:
        actual = lookup.get(str(candidate).lower(), '')
        if actual and actual not in candidate_columns:
            candidate_columns.append(actual)
    return candidate_columns

def cluster_source_cell_id_overlap_scores(source_table: pd.DataFrame, *, cell_id_columns: list[str] | tuple[str, ...], matrix_cell_ids: pd.Index | list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Score raw and safely normalized ID overlap without selecting assignments."""
    matrix_raw = pd.Series(pd.Index(matrix_cell_ids).astype(str), dtype='string')
    methods = ('trimmed', 'casefold', 'xenium_barcode', 'xenium_integer_prefix')
    rows: list[dict[str, object]] = []
    for method_index, method in enumerate(methods):
        matrix_normalized = normalize_cluster_source_cell_ids(matrix_raw, method)
        matrix_map = pd.DataFrame({'raw': matrix_raw.astype(str), 'normalized': matrix_normalized.astype(str)}).drop_duplicates()
        matrix_map = matrix_map.loc[matrix_map['normalized'].ne('')].copy()
        matrix_collision = bool(matrix_map.groupby('normalized')['raw'].nunique().gt(1).any())
        matrix_ids = pd.Index(matrix_map['normalized'].drop_duplicates())
        for column_index, cell_id_column in enumerate(cell_id_columns):
            if str(cell_id_column) not in source_table.columns:
                continue
            source_raw = source_table[str(cell_id_column)].fillna('').astype(str).str.strip()
            source_normalized = normalize_cluster_source_cell_ids(source_raw, method)
            source_map = pd.DataFrame({'raw': source_raw, 'normalized': source_normalized}).drop_duplicates()
            source_map = source_map.loc[source_map['normalized'].ne('')].copy()
            source_collision = bool(source_map.groupby('normalized')['raw'].nunique().gt(1).any())
            source_ids = pd.Index(source_map['normalized'].drop_duplicates())
            valid = not source_collision and (not matrix_collision)
            n_matched = int(source_ids.isin(matrix_ids).sum()) if valid else 0
            n_source = int(len(source_ids))
            rows.append({'cell_id_column': str(cell_id_column), 'method': method, 'method_order': method_index, 'column_order': column_index, 'source_dtype': str(source_table[str(cell_id_column)].dtype), 'n_source_cell_ids': n_source, 'n_matrix_cell_ids': int(len(matrix_ids)), 'n_matched_cell_ids': n_matched, 'match_fraction': float(n_matched / n_source) if n_source else 0.0, 'source_collision': source_collision, 'matrix_collision': matrix_collision, 'valid_for_matching': valid})
    return pd.DataFrame(rows)

def cell_id_preview_table(values: pd.Series | pd.Index | list[str] | tuple[str, ...], *, origin: str, cell_id_column: str, preview_n: int) -> pd.DataFrame:
    """Build a small raw/normalized preview table for notebook display."""
    raw = pd.Series(values, dtype='string').head(max(1, int(preview_n)))
    return pd.DataFrame({'origin': origin, 'cell_id_column': str(cell_id_column), 'position': np.arange(1, len(raw) + 1), 'raw_id': raw.astype(str).to_numpy(), 'trimmed_id': normalize_cluster_source_cell_ids(raw, 'trimmed').to_numpy(), 'casefold_id': normalize_cluster_source_cell_ids(raw, 'casefold').to_numpy(), 'xenium_barcode_id': normalize_cluster_source_cell_ids(raw, 'xenium_barcode').to_numpy(), 'xenium_integer_prefix_id': normalize_cluster_source_cell_ids(raw, 'xenium_integer_prefix').to_numpy()})

def build_xoa_cell_id_preflight(root: str | Path=PROJECT_ROOT, *, output_dir: str | Path | None=None, annotation_dir: str | Path | None=None, gene_panel_review_dir: str | Path | None=None, preview_n: int=10, min_cell_id_match_fraction: float=0.8) -> dict[str, pd.DataFrame]:
    """Preview and score XOA-versus-expression cell IDs without loading counts."""
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir or root / 'cell_clustering').expanduser().resolve()
    inventory = build_cluster_source_inventory(root, output_dir=output_dir, annotation_dir=annotation_dir, gene_panel_review_dir=gene_panel_review_dir)
    if inventory.empty:
        empty = pd.DataFrame()
        return {'summary': empty, 'overlap_scores': empty, 'id_previews': empty}
    selected = inventory.loc[inventory['clustering_source'].astype(str).eq('xoa') & inventory['has_cluster_output'].astype(bool) & inventory['has_feature_matrix'].astype(bool)].copy()
    summaries: list[dict[str, object]] = []
    all_scores: list[pd.DataFrame] = []
    all_previews: list[pd.DataFrame] = []
    for row in selected.sort_values(['dataset', 'sample'], key=lambda column: column.map(natural_key), kind='stable').itertuples(index=False):
        source_row = pd.Series(row._asdict())
        dataset = str(row.dataset)
        sample = str(row.sample)
        cluster_path = resolve_existing_path(source_row.get('cluster_path', ''), root)
        matrix_path = resolve_existing_path(source_row.get('matrix_path', ''), root)
        record: dict[str, object] = {'dataset': dataset, 'sample': sample, 'status': 'failed', 'cluster_source_path': path_for_record(cluster_path, root) if cluster_path is not None else '', 'matrix_path': path_for_record(matrix_path, root) if matrix_path is not None else '', 'candidate_id_columns': '', 'best_cell_id_column': '', 'best_match_method': '', 'n_source_cell_ids': np.nan, 'n_matrix_cell_ids': np.nan, 'n_matched_cell_ids': np.nan, 'match_fraction': np.nan, 'has_collision': False, 'error': ''}
        try:
            if cluster_path is None or matrix_path is None:
                raise FileNotFoundError('XOA source table or expression matrix is missing')
            candidate_columns = cluster_source_candidate_id_columns(source_row, cluster_path)
            if not candidate_columns:
                raise KeyError('XOA source has no cell_id, barcode, or cell column')
            source_ids = pd.read_csv(cluster_path, usecols=candidate_columns)
            matrix_ids = load_feature_matrix_cell_ids(matrix_path)
            scores = cluster_source_cell_id_overlap_scores(source_ids, cell_id_columns=candidate_columns, matrix_cell_ids=matrix_ids)
            scores.insert(0, 'sample', sample)
            scores.insert(0, 'dataset', dataset)
            all_scores.append(scores)
            for candidate_column in candidate_columns:
                preview = cell_id_preview_table(source_ids[candidate_column], origin='XOA', cell_id_column=candidate_column, preview_n=preview_n)
                preview.insert(0, 'sample', sample)
                preview.insert(0, 'dataset', dataset)
                all_previews.append(preview)
            matrix_preview = cell_id_preview_table(matrix_ids, origin='expression_matrix', cell_id_column='matrix_cell_id', preview_n=preview_n)
            matrix_preview.insert(0, 'sample', sample)
            matrix_preview.insert(0, 'dataset', dataset)
            all_previews.append(matrix_preview)
            valid_scores = scores.loc[scores['valid_for_matching'].astype(bool)].copy()
            if valid_scores.empty:
                best = scores.sort_values(['method_order', 'column_order'], kind='stable').iloc[0]
            else:
                best = valid_scores.sort_values(['n_matched_cell_ids', 'match_fraction', 'method_order', 'column_order'], ascending=[False, False, True, True], kind='stable').iloc[0]
            best_fraction = float(best['match_fraction'])
            record.update({'status': 'compatible' if bool(best['valid_for_matching']) and best_fraction >= float(min_cell_id_match_fraction) else 'incompatible', 'candidate_id_columns': '; '.join(candidate_columns), 'best_cell_id_column': str(best['cell_id_column']), 'best_match_method': str(best['method']), 'n_source_cell_ids': int(best['n_source_cell_ids']), 'n_matrix_cell_ids': int(best['n_matrix_cell_ids']), 'n_matched_cell_ids': int(best['n_matched_cell_ids']), 'match_fraction': best_fraction, 'has_collision': bool(best['source_collision'] or best['matrix_collision'])})
        except Exception as error:
            record['error'] = f'{type(error).__name__}: {error}'
        summaries.append(record)
    return {'summary': pd.DataFrame(summaries), 'overlap_scores': pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame(), 'id_previews': pd.concat(all_previews, ignore_index=True) if all_previews else pd.DataFrame()}

def match_cluster_source_cell_ids(source_table: pd.DataFrame, *, cell_id_columns: list[str] | tuple[str, ...], cluster_column: str, cluster_key: str, matrix_cell_ids: pd.Index | list[str] | tuple[str, ...]) -> tuple[pd.DataFrame, dict[str, object]]:
    """Choose the safest source ID column and align it to matrix cell IDs.

    Candidate columns and normalization methods are scored by unique overlap.
    Any method that collapses distinct source or matrix IDs is rejected.
    """
    available_columns = [str(column) for column in cell_id_columns if str(column) in source_table.columns]
    if not available_columns:
        raise KeyError('The cluster source has no candidate cell-ID columns')
    if cluster_column not in source_table.columns:
        raise KeyError(f'The cluster source has no {cluster_column} column')
    matrix_raw = pd.Series(pd.Index(matrix_cell_ids).astype(str), dtype='string')
    methods = ('trimmed', 'casefold', 'xenium_barcode', 'xenium_integer_prefix')
    target_cache: dict[str, tuple[pd.Series, pd.Series, bool]] = {}
    score_rows: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for method_index, method in enumerate(methods):
        matrix_normalized = normalize_cluster_source_cell_ids(matrix_raw, method)
        target_map = pd.DataFrame({'raw': matrix_raw.astype(str), 'normalized': matrix_normalized.astype(str)}).drop_duplicates()
        target_map = target_map.loc[target_map['normalized'].ne('')].copy()
        target_collision = bool(target_map.groupby('normalized')['raw'].nunique().gt(1).any())
        target_lookup = target_map.drop_duplicates('normalized', keep='first').set_index('normalized')['raw']
        target_cache[method] = (matrix_normalized, target_lookup, target_collision)
        for column_index, cell_id_column in enumerate(available_columns):
            source_raw = source_table[cell_id_column].fillna('').astype(str).str.strip()
            source_normalized = normalize_cluster_source_cell_ids(source_raw, method)
            source_map = pd.DataFrame({'raw': source_raw, 'normalized': source_normalized}).drop_duplicates()
            source_map = source_map.loc[source_map['normalized'].ne('')].copy()
            source_collision = bool(source_map.groupby('normalized')['raw'].nunique().gt(1).any())
            valid = not source_collision and (not target_collision)
            source_ids = pd.Index(source_map['normalized'].drop_duplicates())
            target_ids = pd.Index(target_lookup.index)
            n_matched = int(source_ids.isin(target_ids).sum()) if valid else 0
            n_source = int(len(source_ids))
            match_fraction = float(n_matched / n_source) if n_source else 0.0
            score_rows.append({'cell_id_column': cell_id_column, 'method': method, 'n_source_cell_ids': n_source, 'n_matched_cell_ids': n_matched, 'match_fraction': match_fraction, 'source_collision': source_collision, 'matrix_collision': target_collision})
            if valid:
                candidates.append({'cell_id_column': cell_id_column, 'method': method, 'method_index': method_index, 'column_index': column_index, 'n_source': n_source, 'n_matched': n_matched, 'match_fraction': match_fraction, 'source_raw': source_raw, 'source_normalized': source_normalized})
    ranked = sorted(candidates, key=lambda item: (-int(item['n_matched']), -float(item['match_fraction']), int(item['method_index']), int(item['column_index'])))
    best = ranked[0] if ranked else None
    score_table = pd.DataFrame(score_rows)
    overlap_summary = '; '.join((f'{row.cell_id_column}/{row.method}={int(row.n_matched_cell_ids):,}/{int(row.n_source_cell_ids):,}' + (' [collision]' if row.source_collision or row.matrix_collision else '') for row in score_table.itertuples(index=False)))
    if best is None or int(best['n_matched']) == 0:
        source_examples = {column: source_table[column].dropna().astype(str).head(3).tolist() for column in available_columns}
        raise ValueError(f'No XOA cell IDs matched the expression matrix. Candidate overlaps: {overlap_summary}. XOA examples: {source_examples}. Matrix examples: {matrix_raw.head(3).tolist()}')
    chosen_method = str(best['method'])
    chosen_column = str(best['cell_id_column'])
    source_raw = best['source_raw']
    source_normalized = best['source_normalized']
    if not isinstance(source_raw, pd.Series) or not isinstance(source_normalized, pd.Series):
        raise TypeError('Internal cell-ID matching state is invalid')
    _matrix_normalized, target_lookup, _target_collision = target_cache[chosen_method]
    mapped = source_normalized.astype(str).map(target_lookup)
    assignments = source_table[[chosen_column, cluster_column]].copy()
    assignments = assignments.rename(columns={cluster_column: cluster_key})
    assignments['cell_id'] = mapped.where(mapped.notna(), source_raw).astype(str)
    assignments[cluster_key] = assignments[cluster_key].fillna('Missing').astype(str)
    assignments = assignments.loc[assignments['cell_id'].str.strip().ne('')].copy()
    conflicting = assignments.groupby('cell_id', dropna=False)[cluster_key].nunique(dropna=False).gt(1)
    if conflicting.any():
        raise ValueError(f'{int(conflicting.sum()):,} normalized cell IDs have conflicting {cluster_key} assignments')
    assignments = assignments[['cell_id', cluster_key]].drop_duplicates('cell_id', keep='first')
    matched_mask = assignments['cell_id'].isin(pd.Index(matrix_raw.astype(str)))
    unmatched_examples = assignments.loc[~matched_mask, 'cell_id'].head(5).tolist()
    diagnostics: dict[str, object] = {'cell_id_column_used': chosen_column, 'cell_id_match_method': chosen_method, 'n_candidate_id_columns': int(len(available_columns)), 'candidate_id_columns': '; '.join(available_columns), 'candidate_overlap_summary': overlap_summary, 'unmatched_cell_id_examples': '; '.join(map(str, unmatched_examples))}
    return (assignments, diagnostics)

def load_cluster_source_assignments_for_matrix(source_row: pd.Series, matrix_cell_ids: pd.Index | list[str] | tuple[str, ...], *, root: str | Path=PROJECT_ROOT) -> tuple[pd.DataFrame, Path, dict[str, object]]:
    """Load XOA assignments and select the ID column that matches the matrix."""
    root = Path(root).expanduser().resolve()
    cluster_path = resolve_existing_path(source_row.get('cluster_path', ''), root)
    if cluster_path is None:
        raise FileNotFoundError('Cluster-assignment table is missing')
    cluster_column = str(source_row.get('cluster_column', ''))
    cluster_key = str(source_row.get('cluster_key', ''))
    if not cluster_column or not cluster_key:
        raise KeyError('Cluster source is missing a cluster column or cluster key')
    candidate_columns = cluster_source_candidate_id_columns(source_row, cluster_path)
    if not candidate_columns:
        raise KeyError('Cluster source has no cell_id, barcode, or cell column')
    source_table = pd.read_csv(cluster_path, usecols=list(dict.fromkeys([*candidate_columns, cluster_column])))
    assignments, diagnostics = match_cluster_source_cell_ids(source_table, cell_id_columns=candidate_columns, cluster_column=cluster_column, cluster_key=cluster_key, matrix_cell_ids=matrix_cell_ids)
    return (assignments, cluster_path, diagnostics)

def informative_cluster_assignments(assignments: pd.DataFrame, cluster_key: str) -> pd.DataFrame:
    if cluster_key not in assignments.columns:
        raise KeyError(f'Cluster assignments have no {cluster_key} column')
    normalized = assignments[cluster_key].astype(str).str.strip().str.lower()
    return assignments.loc[~normalized.isin(MISSING_CLUSTER_ASSIGNMENT_LABELS)].copy()

def cluster_review_table_paths(output_dir: str | Path, dataset: str, sample: str, cluster_key: str) -> dict[str, Path]:
    sample_output_dir = Path(output_dir) / safe_name(dataset) / safe_name(sample)
    prefix = f'cluster_marker_score_review__{cluster_key}'
    return {'cluster_summary': sample_output_dir / f'{prefix}_cluster_summary.csv', 'counts': sample_output_dir / f'{prefix}_counts.csv', 'row_fraction': sample_output_dir / f'{prefix}_row_fraction.csv', 'column_fraction': sample_output_dir / f'{prefix}_column_fraction.csv', 'pairs': sample_output_dir / f'{prefix}_pairs.csv'}

def save_cluster_review_tables(output_dir: str | Path, dataset: str, sample: str, cluster_key: str, tables: dict[str, pd.DataFrame]) -> dict[str, Path]:
    paths = cluster_review_table_paths(output_dir, dataset, sample, cluster_key)
    for name, output_path in paths.items():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table = tables.get(name, pd.DataFrame())
        if not isinstance(table, pd.DataFrame):
            continue
        if name in {'counts', 'row_fraction', 'column_fraction'}:
            table.reset_index().to_csv(output_path, index=False)
        else:
            table.to_csv(output_path, index=False)
    return paths

def load_review_sample(root: str | Path, inventory: pd.DataFrame, dataset: str, sample: str, *, require_marker_score_labels: bool=True, load_marker_score_labels_table: bool=True) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path | None]:
    root = Path(root).expanduser().resolve()
    rows = inventory.loc[inventory['dataset'].astype(str).eq(str(dataset)) & inventory['sample'].astype(str).eq(str(sample))]
    if rows.empty:
        raise KeyError(f'No review inventory row for {dataset} / {sample}')
    row = rows.iloc[0]
    cluster_path = resolve_existing_path(row['cluster_path'], root)
    label_path = resolve_existing_path(row['marker_label_path'], root)
    if cluster_path is None:
        raise FileNotFoundError(f'Cluster output missing for {dataset} / {sample}')
    if label_path is None:
        if require_marker_score_labels:
            raise FileNotFoundError(f'Marker-score labels missing for {dataset} / {sample}')
        labels = pd.DataFrame(columns=['cell_id', 'group'])
    elif load_marker_score_labels_table:
        labels = load_marker_score_labels(label_path)
    else:
        labels = pd.DataFrame(columns=['cell_id', 'group'])
    clusters = pd.read_csv(cluster_path)
    if 'cell_id' in clusters.columns:
        clusters['cell_id'] = clusters['cell_id'].astype(str)
    return (clusters, labels, cluster_path, label_path)

def review_cluster_marker_cross_table(dataset: str, sample: str, cluster_key: str, *, root: str | Path=PROJECT_ROOT, output_dir: str | Path | None=None, annotation_dir: str | Path | None=None, inventory: pd.DataFrame | None=None, write_outputs: bool=True, require_marker_score_labels: bool=False) -> dict[str, pd.DataFrame | dict[str, Path]]:
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir or root / 'cell_clustering').expanduser().resolve()
    inventory = inventory if inventory is not None else build_cluster_review_inventory(root, output_dir=output_dir, annotation_dir=annotation_dir)
    clusters, labels, cluster_path, label_path = load_review_sample(root, inventory, dataset, sample, require_marker_score_labels=require_marker_score_labels)
    if cluster_key not in clusters.columns:
        raise KeyError(f'{cluster_key} is not in {cluster_path}')
    tables = cross_tables_for_cluster_key(clusters, labels, cluster_key)
    if write_outputs:
        paths = save_cluster_review_tables(output_dir, dataset, sample, cluster_key, tables)
    else:
        paths = cluster_review_table_paths(output_dir, dataset, sample, cluster_key)
    print(f'{dataset} / {sample}')
    print(f'Cluster table: {path_for_display(cluster_path, root)}')
    if label_path is None:
        print('Marker-score labels: not available')
    else:
        print(f'Marker-score labels: {path_for_display(label_path, root)}')
    print(f"Matched cells: {len(tables['matched']):,}")
    cluster_summary = tables.get('cluster_summary', pd.DataFrame())
    n_clusters = len(cluster_summary) if isinstance(cluster_summary, pd.DataFrame) else 0
    print(f'Clusters at {cluster_key}: {n_clusters:,}')
    print(f"Marker-score labels in matched cells: {(tables['counts'].shape[1] if not tables['counts'].empty else 0):,}")
    output: dict[str, pd.DataFrame | dict[str, Path]] = dict(tables)
    output['saved_paths'] = paths
    return output

def cluster_expression_marker_table_paths(output_dir: str | Path, dataset: str, sample: str, cluster_key: str, clustering_source: str='our_expression') -> dict[str, Path]:
    sample_output_dir = Path(output_dir) / safe_name(dataset) / safe_name(sample)
    source_prefix = '' if clustering_source == 'our_expression' else f'{safe_name(clustering_source)}__'
    prefix = f'cluster_expression_markers__{source_prefix}{safe_name(cluster_key)}'
    return {'all_genes': sample_output_dir / f'{prefix}_all_genes.csv.gz', 'top_genes': sample_output_dir / f'{prefix}_top_genes.csv', 'summary': sample_output_dir / f'{prefix}_summary.csv'}
MARKER_CACHE_REQUIRED_TOP_COLUMNS = {'cluster', 'gene', 'passes_basic_marker_filter', 'marker_rank_score', 'log2_fold_change', 'fraction_expressing_cluster', 'fraction_expressing_rest', 'fraction_expressing_difference', 'rank_for_annotation'}
MARKER_CACHE_REQUIRED_SUMMARY_COLUMNS = {'cluster', 'n_eligible_marker_genes_retained', 'has_eligible_marker_genes'}

def marker_method_parameters(*, normalization_target: float, lfc_pseudocount: float, min_fraction_expressing: float, min_fraction_difference: float, min_log2_fold_change: float, top_n: int, run_wilcoxon: bool, max_cells_for_wilcoxon: int | None, random_state: int) -> dict[str, object]:
    return {'normalization_target': float(normalization_target), 'lfc_pseudocount': float(lfc_pseudocount), 'min_fraction_expressing': float(min_fraction_expressing), 'min_fraction_difference': float(min_fraction_difference), 'min_log2_fold_change': float(min_log2_fold_change), 'top_n': int(top_n), 'run_wilcoxon': bool(run_wilcoxon), 'max_cells_for_wilcoxon': int(max_cells_for_wilcoxon) if max_cells_for_wilcoxon is not None else None, 'random_state': int(random_state)}

def marker_method_signature(**parameters: object) -> str:
    return json.dumps(parameters, sort_keys=True, separators=(',', ':'))

def validate_cached_cluster_expression_marker_tables(source_row: pd.Series, paths: dict[str, Path], *, root: str | Path=PROJECT_ROOT, top_n: int, expected_method_signature: str) -> tuple[bool, str]:
    """Validate a cached partition without loading its expression matrix."""
    missing_files = [name for name, path in paths.items() if not path.is_file()]
    if missing_files:
        return (False, 'missing_files:' + ','.join(sorted(missing_files)))
    try:
        top_markers = pd.read_csv(paths['top_genes'], dtype={'cluster': str})
        summary = pd.read_csv(paths['summary'], dtype={'cluster': str})
    except Exception as error:
        return (False, f'unreadable_cache:{type(error).__name__}:{error}')
    missing_top_columns = sorted(MARKER_CACHE_REQUIRED_TOP_COLUMNS - set(top_markers.columns))
    if missing_top_columns:
        return (False, 'missing_top_columns:' + ','.join(missing_top_columns))
    missing_summary_columns = sorted(MARKER_CACHE_REQUIRED_SUMMARY_COLUMNS - set(summary.columns))
    if missing_summary_columns:
        return (False, 'missing_summary_columns:' + ','.join(missing_summary_columns))
    if top_markers.empty or summary.empty:
        return (False, 'empty_top_or_summary_table')
    cluster_key = str(source_row.get('cluster_key', ''))
    try:
        assignments, _cluster_path = load_cluster_source_assignments(source_row, root=root)
        assignments = informative_cluster_assignments(assignments, cluster_key)
    except Exception as error:
        return (False, f'assignment_validation_failed:{type(error).__name__}:{error}')
    expected_clusters = set(assignments[cluster_key].astype(str))
    top_clusters = set(top_markers['cluster'].astype(str))
    summary_clusters = set(summary['cluster'].astype(str))
    if top_clusters != expected_clusters:
        missing = sorted(expected_clusters - top_clusters, key=natural_key)
        extra = sorted(top_clusters - expected_clusters, key=natural_key)
        return (False, f'top_cluster_set_mismatch:missing={missing};extra={extra}')
    if summary_clusters != expected_clusters:
        missing = sorted(expected_clusters - summary_clusters, key=natural_key)
        extra = sorted(summary_clusters - expected_clusters, key=natural_key)
        return (False, f'summary_cluster_set_mismatch:missing={missing};extra={extra}')
    retained_counts = top_markers.groupby('cluster', dropna=False).size()
    if retained_counts.gt(int(top_n)).any():
        return (False, f'retained_top_genes_exceed_configured_top_n:{int(top_n)}')
    ranks = pd.to_numeric(top_markers['rank_for_annotation'], errors='coerce')
    if ranks.isna().any() or ranks.gt(int(top_n)).any():
        return (False, f'invalid_rank_for_annotation_for_top_n:{int(top_n)}')
    signatures = summary['marker_method_signature'].dropna().astype(str).unique().tolist() if 'marker_method_signature' in summary.columns else []
    if len(signatures) > 1:
        return (False, 'multiple_marker_method_signatures')
    if signatures and signatures[0] != expected_method_signature:
        return (False, 'marker_method_signature_mismatch')
    return (True, 'valid_current_signature' if signatures else 'valid_cached_schema')

def load_cluster_expression_top_markers(output_dir: str | Path, dataset: str, sample: str, cluster_key: str, clustering_source: str='our_expression') -> pd.DataFrame:
    path = cluster_expression_marker_table_paths(output_dir, dataset, sample, cluster_key, clustering_source)['top_genes']
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)

def load_cluster_expression_all_markers(output_dir: str | Path, dataset: str, sample: str, cluster_key: str, clustering_source: str='our_expression') -> pd.DataFrame:
    path = cluster_expression_marker_table_paths(output_dir, dataset, sample, cluster_key, clustering_source)['all_genes']
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)

def cluster_marker_gene_hint_table_paths(output_dir: str | Path, dataset: str, sample: str, cluster_key: str) -> dict[str, Path]:
    sample_output_dir = Path(output_dir) / safe_name(dataset) / safe_name(sample)
    prefix = f'cluster_marker_gene_hints__{cluster_key}'
    return {'annotated_top_genes': sample_output_dir / f'{prefix}_annotated_top_genes.csv', 'group_candidates': sample_output_dir / f'{prefix}_group_candidates.csv', 'label_suggestions': sample_output_dir / f'{prefix}_label_suggestions.csv'}

def load_gene_marker_hint_sources(root: str | Path=PROJECT_ROOT, *, gene_panel_review_dir: str | Path | None=None) -> dict[str, pd.DataFrame]:
    """Load notebook-14 gene-panel marker relationship tables."""
    root = Path(root).expanduser().resolve()
    review_dir = Path(gene_panel_review_dir or root / 'gene_panel_review').expanduser()
    if not review_dir.is_absolute():
        review_dir = root / review_dir
    tables = {}
    for key, filename in {'marker_gene_table': 'marker_gene_table.csv', 'gene_group_by_dataset': 'gene_group_annotation_by_dataset.csv', 'annotation_conflicts': 'gene_group_annotation_conflicts.csv', 'panel_implied_groups': 'panel_implied_marker_groups.csv'}.items():
        path = review_dir / filename
        tables[key] = pd.read_csv(path) if path.is_file() else pd.DataFrame()
    return tables

def _unique_join(values: list[object]) -> str:
    parts: list[str] = []
    for value in values:
        for part in split_path_list(value):
            if part not in parts:
                parts.append(part)
    return '; '.join(parts)

def _make_gene_marker_hint_table_for_sample(dataset: str, sample: str, genes: pd.Series, hint_sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build one row per requested gene with notebook-14 marker-group hints."""
    gene_order = pd.Series(genes.dropna().astype(str).unique(), name='gene').to_frame()
    if gene_order.empty:
        return pd.DataFrame(columns=['gene'])
    marker_gene_table = hint_sources.get('marker_gene_table', pd.DataFrame()).copy()
    conflict_table = hint_sources.get('annotation_conflicts', pd.DataFrame()).copy()
    panel_implied = hint_sources.get('panel_implied_groups', pd.DataFrame()).copy()
    direct = pd.DataFrame(columns=['gene'])
    if not marker_gene_table.empty and {'dataset', 'sample', 'gene', 'group'}.issubset(marker_gene_table.columns):
        direct_source = marker_gene_table.loc[marker_gene_table['dataset'].astype(str).eq(str(dataset)) & marker_gene_table['sample'].astype(str).eq(str(sample)) & marker_gene_table['gene'].astype(str).isin(gene_order['gene'])].copy()
        if 'in_assay' in direct_source.columns:
            direct_source = direct_source.loc[direct_source['in_assay'].astype(str).str.lower().isin({'true', '1', 'yes'})]
        if not direct_source.empty:
            direct = direct_source.groupby('gene', dropna=False).agg(direct_author_marker_groups=('group', lambda values: _unique_join(list(values))), direct_author_marker_group_files=('marker_group_file', lambda values: _unique_join(list(values))) if 'marker_group_file' in direct_source.columns else ('group', lambda values: '')).reset_index()
    implied = pd.DataFrame(columns=['gene'])
    if not panel_implied.empty and {'dataset', 'sample', 'group', 'matching_marker_genes'}.issubset(panel_implied.columns):
        implied_rows: list[dict[str, object]] = []
        sample_implied = panel_implied.loc[panel_implied['dataset'].astype(str).eq(str(dataset)) & panel_implied['sample'].astype(str).eq(str(sample))].copy()
        for row in sample_implied.itertuples(index=False):
            row_dict = row._asdict()
            for gene in split_path_list(row_dict.get('matching_marker_genes', '')):
                if gene in set(gene_order['gene']):
                    implied_rows.append({'gene': gene, 'panel_implied_marker_group': str(row_dict.get('group', '')), 'panel_implied_evidence_type': str(row_dict.get('evidence_type', '')), 'panel_implied_reference_datasets': str(row_dict.get('reference_datasets', ''))})
        if implied_rows:
            implied_source = pd.DataFrame(implied_rows)
            implied = implied_source.groupby('gene', dropna=False).agg(panel_implied_marker_groups=('panel_implied_marker_group', lambda values: _unique_join(list(values))), panel_implied_evidence_types=('panel_implied_evidence_type', lambda values: _unique_join(list(values))), panel_implied_reference_datasets=('panel_implied_reference_datasets', lambda values: _unique_join(list(values)))).reset_index()
    conflict = pd.DataFrame(columns=['gene'])
    if not conflict_table.empty and 'gene' in conflict_table.columns:
        conflict_columns = [column for column in ['gene', 'n_datasets_annotated', 'n_unique_dataset_group_assignments', 'has_conflicting_dataset_annotations', 'has_multiple_groups_within_dataset', 'has_dataset_annotation_presence_difference', 'has_any_annotation_difference', 'datasets_annotated', 'groups_by_dataset', 'unique_group_assignments'] if column in conflict_table.columns]
        conflict = conflict_table.loc[conflict_table['gene'].astype(str).isin(gene_order['gene']), conflict_columns].copy()
        conflict['gene'] = conflict['gene'].astype(str)
    hints = gene_order.merge(direct, on='gene', how='left').merge(implied, on='gene', how='left').merge(conflict, on='gene', how='left')
    for column in ['direct_author_marker_groups', 'direct_author_marker_group_files', 'panel_implied_marker_groups', 'panel_implied_evidence_types', 'panel_implied_reference_datasets', 'groups_by_dataset', 'unique_group_assignments', 'datasets_annotated']:
        if column not in hints.columns:
            hints[column] = ''
        hints[column] = hints[column].fillna('').astype(str)
    for column in ['has_conflicting_dataset_annotations', 'has_multiple_groups_within_dataset', 'has_dataset_annotation_presence_difference', 'has_any_annotation_difference']:
        if column not in hints.columns:
            hints[column] = False
        hints[column] = hints[column].map(lambda value: str(value).strip().lower() in {'true', '1', 'yes'})
    preferred_groups: list[str] = []
    evidence_types: list[str] = []
    for row in hints.itertuples(index=False):
        direct_groups = str(getattr(row, 'direct_author_marker_groups', '') or '')
        implied_groups = str(getattr(row, 'panel_implied_marker_groups', '') or '')
        cross_dataset_groups = str(getattr(row, 'unique_group_assignments', '') or '')
        if direct_groups:
            preferred_groups.append(direct_groups)
            evidence_types.append('direct_author_marker_group')
        elif implied_groups:
            preferred_groups.append(implied_groups)
            evidence_types.append('panel_implied_marker_group')
        elif cross_dataset_groups:
            preferred_groups.append(cross_dataset_groups)
            evidence_types.append('cross_dataset_marker_group')
        else:
            preferred_groups.append('')
            evidence_types.append('unannotated_gene')
    hints['gene_marker_groups'] = preferred_groups
    hints['gene_marker_evidence_type'] = evidence_types
    hints['n_gene_marker_groups'] = hints['gene_marker_groups'].map(lambda value: len(split_path_list(value)))
    hints['has_gene_marker_hint'] = hints['n_gene_marker_groups'].gt(0)
    return hints

def _marker_hint_weight(table: pd.DataFrame) -> pd.Series:
    if 'marker_rank_score' in table.columns:
        score = pd.to_numeric(table['marker_rank_score'], errors='coerce').fillna(0).clip(lower=0)
        if score.gt(0).any():
            return score
    fallback = pd.Series(0, index=table.index, dtype=float)
    log2_fc = pd.to_numeric(table['log2_fold_change'], errors='coerce').fillna(0).clip(lower=0) if 'log2_fold_change' in table.columns else fallback
    fraction_difference = pd.to_numeric(table['fraction_expressing_difference'], errors='coerce').fillna(0).clip(lower=0) if 'fraction_expressing_difference' in table.columns else fallback
    weight = log2_fc * fraction_difference
    if weight.gt(0).any():
        return weight
    rank = pd.to_numeric(table['rank_for_annotation'], errors='coerce').fillna(1).clip(lower=1) if 'rank_for_annotation' in table.columns else pd.Series(1, index=table.index, dtype=float)
    return 1 / rank

def annotate_cluster_top_markers_with_gene_hints(top_markers: pd.DataFrame, *, root: str | Path=PROJECT_ROOT, gene_panel_review_dir: str | Path | None=None, hint_sources: dict[str, pd.DataFrame] | None=None) -> pd.DataFrame:
    if top_markers.empty or 'gene' not in top_markers.columns:
        return top_markers.copy()
    sources = hint_sources if hint_sources is not None else load_gene_marker_hint_sources(root, gene_panel_review_dir=gene_panel_review_dir)
    output_tables: list[pd.DataFrame] = []
    grouping_columns = [column for column in ['dataset', 'sample'] if column in top_markers.columns]
    if len(grouping_columns) < 2:
        table = top_markers.copy()
        table['dataset'] = table.get('dataset', '')
        table['sample'] = table.get('sample', '')
        grouping_columns = ['dataset', 'sample']
    else:
        table = top_markers.copy()
    for (dataset, sample), marker_table in table.groupby(grouping_columns, dropna=False):
        hints = _make_gene_marker_hint_table_for_sample(str(dataset), str(sample), marker_table['gene'], sources)
        output_tables.append(marker_table.merge(hints, on='gene', how='left'))
    annotated = pd.concat(output_tables, ignore_index=True) if output_tables else table
    annotated['marker_hint_weight'] = _marker_hint_weight(annotated)
    return annotated

def summarize_cluster_marker_gene_group_candidates(annotated_top_markers: pd.DataFrame) -> pd.DataFrame:
    if annotated_top_markers.empty or 'gene_marker_groups' not in annotated_top_markers.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    base_columns = ['dataset', 'sample', 'cluster_key', 'cluster']
    for key, table in annotated_top_markers.groupby(base_columns, dropna=False):
        dataset, sample, cluster_key, cluster = key
        if 'passes_basic_marker_filter' in table.columns:
            table = table.loc[table['passes_basic_marker_filter'].fillna(False).astype(bool)].copy()
        annotated_weight_total = float(table.loc[table['gene_marker_groups'].fillna('').astype(str).str.strip().ne(''), 'marker_hint_weight'].sum())
        group_rows: list[dict[str, object]] = []
        for row in table.itertuples(index=False):
            groups = split_path_list(getattr(row, 'gene_marker_groups', ''))
            if not groups:
                continue
            weight = float(getattr(row, 'marker_hint_weight', 0) or 0)
            split_weight = weight / max(len(groups), 1)
            for group in groups:
                group_rows.append({'marker_group': group, 'gene': str(getattr(row, 'gene', '')), 'weight': split_weight, 'marker_rank_score': getattr(row, 'marker_rank_score', np.nan), 'log2_fold_change': getattr(row, 'log2_fold_change', np.nan), 'fraction_expressing_cluster': getattr(row, 'fraction_expressing_cluster', np.nan), 'fraction_expressing_difference': getattr(row, 'fraction_expressing_difference', np.nan), 'gene_marker_evidence_type': getattr(row, 'gene_marker_evidence_type', ''), 'has_any_annotation_difference': bool(getattr(row, 'has_any_annotation_difference', False))})
        if not group_rows:
            rows.append({'dataset': dataset, 'sample': sample, 'cluster_key': cluster_key, 'cluster': cluster, 'marker_group': 'Unannotated marker genes', 'n_supporting_genes': 0, 'supporting_genes': '', 'support_weight': 0.0, 'total_annotated_marker_weight': 0.0, 'fraction_of_annotated_marker_weight': np.nan, 'mean_marker_rank_score': np.nan, 'mean_log2_fold_change': np.nan, 'mean_fraction_expressing_cluster': np.nan, 'mean_fraction_expressing_difference': np.nan, 'strongest_gene': '', 'evidence_types': '', 'n_direct_author_supporting_genes': 0, 'n_panel_implied_supporting_genes': 0, 'n_cross_dataset_supporting_genes': 0, 'has_conflicting_or_partial_gene_annotations': False})
            continue
        group_table = pd.DataFrame(group_rows)
        for group, group_marker_table in group_table.groupby('marker_group', dropna=False):
            strongest = group_marker_table.sort_values('weight', ascending=False).iloc[0]
            rows.append({'dataset': dataset, 'sample': sample, 'cluster_key': cluster_key, 'cluster': cluster, 'marker_group': group, 'n_supporting_genes': int(group_marker_table['gene'].nunique()), 'supporting_genes': '; '.join(group_marker_table['gene'].drop_duplicates().astype(str).head(12).tolist()), 'support_weight': float(group_marker_table['weight'].sum()), 'total_annotated_marker_weight': annotated_weight_total, 'fraction_of_annotated_marker_weight': float(group_marker_table['weight'].sum()) / annotated_weight_total if annotated_weight_total > 0 else np.nan, 'mean_marker_rank_score': pd.to_numeric(group_marker_table['marker_rank_score'], errors='coerce').mean(), 'mean_log2_fold_change': pd.to_numeric(group_marker_table['log2_fold_change'], errors='coerce').mean(), 'mean_fraction_expressing_cluster': pd.to_numeric(group_marker_table['fraction_expressing_cluster'], errors='coerce').mean(), 'mean_fraction_expressing_difference': pd.to_numeric(group_marker_table['fraction_expressing_difference'], errors='coerce').mean(), 'strongest_gene': strongest['gene'], 'evidence_types': '; '.join(group_marker_table['gene_marker_evidence_type'].drop_duplicates().astype(str).tolist()), 'n_direct_author_supporting_genes': int(group_marker_table.loc[group_marker_table['gene_marker_evidence_type'].astype(str).eq('direct_author_marker_group'), 'gene'].nunique()), 'n_panel_implied_supporting_genes': int(group_marker_table.loc[group_marker_table['gene_marker_evidence_type'].astype(str).eq('panel_implied_marker_group'), 'gene'].nunique()), 'n_cross_dataset_supporting_genes': int(group_marker_table.loc[group_marker_table['gene_marker_evidence_type'].astype(str).eq('cross_dataset_marker_group'), 'gene'].nunique()), 'has_conflicting_or_partial_gene_annotations': bool(group_marker_table['has_any_annotation_difference'].any())})
    return pd.DataFrame(rows).sort_values(['dataset', 'sample', 'cluster_key', 'cluster', 'support_weight', 'marker_group'], ascending=[True, True, True, True, False, True], key=lambda column: column.map(natural_key) if column.name in {'dataset', 'sample', 'cluster', 'marker_group'} else column).reset_index(drop=True)

def suggest_cluster_labels_from_gene_hints(annotated_top_markers: pd.DataFrame, group_candidates: pd.DataFrame) -> pd.DataFrame:
    if annotated_top_markers.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    base_columns = ['dataset', 'sample', 'cluster_key', 'cluster']
    for key, marker_table in annotated_top_markers.groupby(base_columns, dropna=False):
        dataset, sample, cluster_key, cluster = key
        eligible_marker_table = marker_table.loc[marker_table['passes_basic_marker_filter'].fillna(False).astype(bool)].copy() if 'passes_basic_marker_filter' in marker_table.columns else marker_table.copy()
        if group_candidates.empty:
            candidate_table = pd.DataFrame()
        else:
            candidate_table = group_candidates.loc[group_candidates['dataset'].astype(str).eq(str(dataset)) & group_candidates['sample'].astype(str).eq(str(sample)) & group_candidates['cluster_key'].astype(str).eq(str(cluster_key)) & group_candidates['cluster'].astype(str).eq(str(cluster)) & group_candidates['marker_group'].astype(str).ne('Unannotated marker genes')].copy()
        candidate_table = candidate_table.sort_values(['support_weight', 'n_supporting_genes', 'marker_group'], ascending=[False, False, True], key=lambda column: column.map(natural_key) if column.name == 'marker_group' else column)
        annotated_gene_count = int(eligible_marker_table['has_gene_marker_hint'].fillna(False).astype(bool).sum())
        unannotated_gene_count = int(len(eligible_marker_table) - annotated_gene_count)
        top = candidate_table.iloc[0] if not candidate_table.empty else pd.Series(dtype=object)
        second = candidate_table.iloc[1] if len(candidate_table) > 1 else pd.Series(dtype=object)
        top_fraction = float(top.get('fraction_of_annotated_marker_weight', np.nan)) if not top.empty else np.nan
        top_gene_count = int(top.get('n_supporting_genes', 0) or 0) if not top.empty else 0
        if eligible_marker_table.empty:
            confidence = 'low'
            note = 'No top marker genes passed all expression-effect eligibility thresholds.'
        elif pd.isna(top_fraction) or top_gene_count == 0:
            confidence = 'low'
            note = "No notebook-14 gene-marker hints matched this cluster's eligible top marker genes."
        elif top_fraction >= 0.55 and top_gene_count >= 2:
            confidence = 'high'
            note = 'Dominant marker group among annotated top cluster genes.'
        elif top_fraction >= 0.35:
            confidence = 'medium'
            note = 'One marker group is enriched, but review other groups and marker strength before final labeling.'
        else:
            confidence = 'low'
            note = 'Marker-group support is diffuse; consider Mixed or Uncertain unless expression evidence is clear.'
        rows.append({'dataset': dataset, 'sample': sample, 'cluster_key': cluster_key, 'cluster': cluster, 'suggested_marker_group': top.get('marker_group', ''), 'suggestion_confidence': confidence, 'suggestion_note': note, 'top_group_fraction_of_annotated_marker_weight': top_fraction, 'top_group_n_supporting_genes': top_gene_count, 'top_group_supporting_genes': top.get('supporting_genes', ''), 'second_marker_group': second.get('marker_group', ''), 'second_group_fraction_of_annotated_marker_weight': second.get('fraction_of_annotated_marker_weight', np.nan), 'n_candidate_marker_groups': int(len(candidate_table)), 'n_eligible_top_genes': int(len(eligible_marker_table)), 'n_annotated_top_genes': annotated_gene_count, 'n_unannotated_top_genes': unannotated_gene_count})
    return pd.DataFrame(rows).sort_values(['dataset', 'sample', 'cluster_key', 'cluster'], key=lambda column: column.map(natural_key)).reset_index(drop=True)

def top_two_candidate_decision_criteria(candidate_1: str, candidate_2: str, candidate_1_support_fraction: object, candidate_2_support_fraction: object, candidate_1_n_supporting_genes: object, candidate_2_n_supporting_genes: object) -> dict[str, object]:
    """Summarize transparent review heuristics; support fractions are not probabilities."""
    fraction_1 = pd.to_numeric(pd.Series([candidate_1_support_fraction]), errors='coerce').iloc[0]
    fraction_2 = pd.to_numeric(pd.Series([candidate_2_support_fraction]), errors='coerce').iloc[0]
    fraction_1 = float(fraction_1) if pd.notna(fraction_1) else 0.0
    fraction_2 = float(fraction_2) if pd.notna(fraction_2) else 0.0
    genes_1_value = pd.to_numeric(pd.Series([candidate_1_n_supporting_genes]), errors='coerce').iloc[0]
    genes_2_value = pd.to_numeric(pd.Series([candidate_2_n_supporting_genes]), errors='coerce').iloc[0]
    genes_1 = int(genes_1_value) if pd.notna(genes_1_value) else 0
    genes_2 = int(genes_2_value) if pd.notna(genes_2_value) else 0
    support_gap = fraction_1 - fraction_2
    top_two_support = fraction_1 + fraction_2
    support_ratio = fraction_1 / fraction_2 if fraction_2 > 0 else np.nan
    if candidate_1 and candidate_2 and (fraction_1 >= 0.2) and (fraction_2 >= 0.2) and (top_two_support >= 0.7) and (support_gap < 0.15) and (genes_1 >= 2) and (genes_2 >= 2):
        recommendation = 'Both'
        recommendation_code = 'both'
        reason = 'Both candidates have at least 20% support, their combined support is at least 70%, their gap is below 15 percentage points, and each is supported by at least two genes.'
    elif candidate_1 and fraction_1 >= 0.55 and (support_gap >= 0.15) and (genes_1 >= 3):
        recommendation = 'Candidate 1'
        recommendation_code = 'candidate_1'
        reason = 'Candidate 1 has at least 55% support, leads by at least 15 percentage points, and is supported by at least three genes.'
    elif not candidate_1 or fraction_1 < 0.35 or genes_1 < 2:
        recommendation = 'Other / uncertain'
        recommendation_code = 'other'
        reason = 'No leading candidate has both at least 35% support and at least two supporting genes.'
    else:
        recommendation = 'Manual review'
        recommendation_code = 'manual_review'
        reason = 'The marker evidence is intermediate and does not satisfy the conservative single-candidate, both-candidate, or uncertain guide.'
    return {'candidate_support_gap': support_gap, 'candidate_support_ratio': support_ratio, 'top_two_support_fraction': top_two_support, 'heuristic_recommendation': recommendation, 'heuristic_recommendation_code': recommendation_code, 'heuristic_recommendation_reason': reason}

def top_two_gene_hint_candidates(annotated_top_markers: pd.DataFrame, group_candidates: pd.DataFrame, *, clustering_source: str) -> pd.DataFrame:
    """Reduce ranked gene-hint groups to two review candidates per cluster."""
    if annotated_top_markers.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    key_columns = ['dataset', 'sample', 'cluster_key', 'cluster']
    for key, marker_table in annotated_top_markers.groupby(key_columns, dropna=False):
        dataset, sample, cluster_key, cluster = key
        eligible_marker_table = marker_table.loc[marker_table['passes_basic_marker_filter'].fillna(False).astype(bool)].copy() if 'passes_basic_marker_filter' in marker_table.columns else marker_table.copy()
        if group_candidates.empty:
            candidates = pd.DataFrame()
        else:
            candidates = group_candidates.loc[group_candidates['dataset'].astype(str).eq(str(dataset)) & group_candidates['sample'].astype(str).eq(str(sample)) & group_candidates['cluster_key'].astype(str).eq(str(cluster_key)) & group_candidates['cluster'].astype(str).eq(str(cluster)) & group_candidates['marker_group'].astype(str).ne('Unannotated marker genes')].copy()
        candidates = candidates.sort_values(['support_weight', 'n_supporting_genes', 'marker_group'], ascending=[False, False, True], key=lambda column: column.map(natural_key) if column.name == 'marker_group' else column, kind='stable').head(2)
        first = candidates.iloc[0] if len(candidates) >= 1 else pd.Series(dtype=object)
        second = candidates.iloc[1] if len(candidates) >= 2 else pd.Series(dtype=object)
        candidate_1 = str(first.get('marker_group', '') or '')
        candidate_2 = str(second.get('marker_group', '') or '')
        decision_criteria = top_two_candidate_decision_criteria(candidate_1, candidate_2, first.get('fraction_of_annotated_marker_weight', np.nan), second.get('fraction_of_annotated_marker_weight', np.nan), first.get('n_supporting_genes', 0), second.get('n_supporting_genes', 0))
        if candidate_1 and candidate_2:
            automatic_refined = f'{candidate_1} + {candidate_2}'
            automatic_selection = 'both'
        elif candidate_1:
            automatic_refined = candidate_1
            automatic_selection = 'candidate_1'
        else:
            automatic_refined = ''
            automatic_selection = 'other'
        rows.append({'dataset': dataset, 'sample': sample, 'clustering_source': str(clustering_source), 'cluster_key': cluster_key, 'cluster': cluster, 'automatic_candidate_1': candidate_1, 'automatic_candidate_2': candidate_2, 'automatic_refined_annotation': automatic_refined, 'automatic_review_selection': automatic_selection, 'candidate_1_supporting_genes': first.get('supporting_genes', ''), 'candidate_2_supporting_genes': second.get('supporting_genes', ''), 'candidate_1_support_fraction': first.get('fraction_of_annotated_marker_weight', np.nan), 'candidate_2_support_fraction': second.get('fraction_of_annotated_marker_weight', np.nan), 'candidate_1_support_weight': first.get('support_weight', np.nan), 'candidate_2_support_weight': second.get('support_weight', np.nan), 'total_annotated_marker_weight': first.get('total_annotated_marker_weight', second.get('total_annotated_marker_weight', np.nan)), 'candidate_1_n_supporting_genes': first.get('n_supporting_genes', 0), 'candidate_2_n_supporting_genes': second.get('n_supporting_genes', 0), 'candidate_1_strongest_gene': first.get('strongest_gene', ''), 'candidate_2_strongest_gene': second.get('strongest_gene', ''), 'candidate_1_mean_marker_rank_score': first.get('mean_marker_rank_score', np.nan), 'candidate_2_mean_marker_rank_score': second.get('mean_marker_rank_score', np.nan), 'candidate_1_mean_log2_fold_change': first.get('mean_log2_fold_change', np.nan), 'candidate_2_mean_log2_fold_change': second.get('mean_log2_fold_change', np.nan), 'candidate_1_mean_fraction_expressing_cluster': first.get('mean_fraction_expressing_cluster', np.nan), 'candidate_2_mean_fraction_expressing_cluster': second.get('mean_fraction_expressing_cluster', np.nan), 'candidate_1_mean_fraction_expressing_difference': first.get('mean_fraction_expressing_difference', np.nan), 'candidate_2_mean_fraction_expressing_difference': second.get('mean_fraction_expressing_difference', np.nan), 'candidate_1_n_direct_author_genes': first.get('n_direct_author_supporting_genes', 0), 'candidate_2_n_direct_author_genes': second.get('n_direct_author_supporting_genes', 0), 'candidate_1_n_panel_implied_genes': first.get('n_panel_implied_supporting_genes', 0), 'candidate_2_n_panel_implied_genes': second.get('n_panel_implied_supporting_genes', 0), 'candidate_1_n_cross_dataset_genes': first.get('n_cross_dataset_supporting_genes', 0), 'candidate_2_n_cross_dataset_genes': second.get('n_cross_dataset_supporting_genes', 0), 'candidate_1_evidence_types': first.get('evidence_types', ''), 'candidate_2_evidence_types': second.get('evidence_types', ''), **decision_criteria, 'n_candidate_marker_groups': int(len(group_candidates.loc[group_candidates['dataset'].astype(str).eq(str(dataset)) & group_candidates['sample'].astype(str).eq(str(sample)) & group_candidates['cluster_key'].astype(str).eq(str(cluster_key)) & group_candidates['cluster'].astype(str).eq(str(cluster)) & group_candidates['marker_group'].astype(str).ne('Unannotated marker genes')])) if not group_candidates.empty else 0, 'n_top_marker_genes_reviewed': int(len(marker_table)), 'n_eligible_top_marker_genes': int(len(eligible_marker_table)), 'n_top_marker_genes_with_hints': int(marker_table.get('has_gene_marker_hint', pd.Series(False, index=marker_table.index)).fillna(False).astype(bool).sum()), 'n_eligible_top_marker_genes_with_hints': int(eligible_marker_table.get('has_gene_marker_hint', pd.Series(False, index=eligible_marker_table.index)).fillna(False).astype(bool).sum())})
    return pd.DataFrame(rows).sort_values(['clustering_source', 'dataset', 'sample', 'cluster_key', 'cluster'], key=lambda column: column.map(natural_key), kind='stable').reset_index(drop=True)

def build_cluster_marker_gene_hint_tables(root: str | Path=PROJECT_ROOT, *, output_dir: str | Path | None=None, annotation_dir: str | Path | None=None, gene_panel_review_dir: str | Path | None=None, cluster_key: str='leiden_r1_3', dataset_names: list[str] | tuple[str, ...] | set[str] | None=None, sample_names: list[str] | tuple[str, ...] | set[str] | None=None, write_outputs: bool=True) -> dict[str, pd.DataFrame]:
    """Attach notebook-14 gene-marker hints to saved top marker genes."""
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir or root / 'cell_clustering').expanduser().resolve()
    annotation_dir = Path(annotation_dir or root / 'cell_annotation').expanduser().resolve()
    inventory = build_cluster_expression_marker_inventory(root, output_dir=output_dir, annotation_dir=annotation_dir)
    if inventory.empty:
        empty = pd.DataFrame()
        return {'gene_hint_summary': empty, 'combined_annotated_top_genes': empty, 'combined_group_candidates': empty, 'combined_label_suggestions': empty, 'marker_inventory': inventory}
    ready = inventory.loc[inventory['cluster_key'].astype(str).eq(str(cluster_key)) & inventory['has_saved_top_gene_table'].astype(bool)].copy()
    if dataset_names is not None:
        ready = ready.loc[ready['dataset'].astype(str).isin(set(map(str, dataset_names)))].copy()
    if sample_names is not None:
        ready = ready.loc[ready['sample'].astype(str).isin(set(map(str, sample_names)))].copy()
    hint_sources = load_gene_marker_hint_sources(root, gene_panel_review_dir=gene_panel_review_dir)
    rows: list[dict[str, object]] = []
    annotated_tables: list[pd.DataFrame] = []
    candidate_tables: list[pd.DataFrame] = []
    suggestion_tables: list[pd.DataFrame] = []
    for row in ready.sort_values(['dataset', 'sample'], key=lambda column: column.map(natural_key)).itertuples(index=False):
        dataset = str(row.dataset)
        sample = str(row.sample)
        paths = cluster_marker_gene_hint_table_paths(output_dir, dataset, sample, str(cluster_key))
        record: dict[str, object] = {'dataset': dataset, 'sample': sample, 'cluster_key': str(cluster_key), 'status': 'not_run', 'annotated_top_genes_path': path_for_record(paths['annotated_top_genes'], root), 'group_candidates_path': path_for_record(paths['group_candidates'], root), 'label_suggestions_path': path_for_record(paths['label_suggestions'], root), 'n_top_marker_rows': 0, 'n_marker_rows_with_gene_hints': 0, 'n_group_candidate_rows': 0, 'n_label_suggestion_rows': 0, 'error': ''}
        try:
            top_markers = load_cluster_expression_top_markers(output_dir, dataset, sample, str(cluster_key))
            annotated = annotate_cluster_top_markers_with_gene_hints(top_markers, root=root, gene_panel_review_dir=gene_panel_review_dir, hint_sources=hint_sources)
            candidates = summarize_cluster_marker_gene_group_candidates(annotated)
            suggestions = suggest_cluster_labels_from_gene_hints(annotated, candidates)
            if write_outputs:
                for path in paths.values():
                    path.parent.mkdir(parents=True, exist_ok=True)
                annotated.to_csv(paths['annotated_top_genes'], index=False)
                candidates.to_csv(paths['group_candidates'], index=False)
                suggestions.to_csv(paths['label_suggestions'], index=False)
            annotated_tables.append(annotated)
            candidate_tables.append(candidates)
            suggestion_tables.append(suggestions)
            record.update({'status': 'success', 'n_top_marker_rows': int(len(annotated)), 'n_marker_rows_with_gene_hints': int(annotated.get('has_gene_marker_hint', pd.Series(dtype=bool)).fillna(False).astype(bool).sum()), 'n_group_candidate_rows': int(len(candidates)), 'n_label_suggestion_rows': int(len(suggestions))})
        except Exception as error:
            record.update({'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
        rows.append(record)
    summary = pd.DataFrame(rows)
    combined_annotated = pd.concat(annotated_tables, ignore_index=True) if annotated_tables else pd.DataFrame()
    combined_candidates = pd.concat(candidate_tables, ignore_index=True) if candidate_tables else pd.DataFrame()
    combined_suggestions = pd.concat(suggestion_tables, ignore_index=True) if suggestion_tables else pd.DataFrame()
    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = safe_name(cluster_key)
        summary.to_csv(output_dir / f'cluster_marker_gene_hint_summary__{suffix}.csv', index=False)
        if not combined_annotated.empty:
            combined_annotated.to_csv(output_dir / f'combined_cluster_marker_gene_hints__{suffix}.csv', index=False)
        if not combined_candidates.empty:
            combined_candidates.to_csv(output_dir / f'combined_cluster_marker_gene_group_candidates__{suffix}.csv', index=False)
        if not combined_suggestions.empty:
            combined_suggestions.to_csv(output_dir / f'combined_cluster_marker_gene_label_suggestions__{suffix}.csv', index=False)
    return {'gene_hint_summary': summary, 'combined_annotated_top_genes': combined_annotated, 'combined_group_candidates': combined_candidates, 'combined_label_suggestions': combined_suggestions, 'marker_inventory': inventory}

def align_adata_to_cluster_table(adata: Any, clusters: pd.DataFrame, cluster_key: str) -> tuple[Any, pd.Series, pd.DataFrame]:
    if 'cell_id' not in clusters.columns:
        raise KeyError('Cluster table must contain cell_id')
    if cluster_key not in clusters.columns:
        raise KeyError(f'Cluster table has no {cluster_key} column')
    cluster_table = clusters[['cell_id', cluster_key]].dropna(subset=['cell_id']).copy()
    cluster_table['cell_id'] = cluster_table['cell_id'].astype(str)
    cluster_table[cluster_key] = cluster_table[cluster_key].fillna('Missing').astype(str)
    cluster_table = cluster_table.drop_duplicates('cell_id', keep='first')
    matrix_cells = pd.Index(adata.obs_names.astype(str))
    available = cluster_table['cell_id'].isin(matrix_cells)
    matched = cluster_table.loc[available].copy()
    if matched.empty:
        raise ValueError('No clustered cell IDs were found in the expression matrix')
    adata_subset = adata[matched['cell_id'].tolist(), :].copy()
    labels = matched.set_index('cell_id')[cluster_key].reindex(adata_subset.obs_names.astype(str)).astype(str)
    return (adata_subset, labels, matched)

def calculate_cluster_expression_marker_metrics(adata: Any, labels: pd.Series, *, dataset: str, sample: str, cluster_key: str, normalization_target: float=100000.0, lfc_pseudocount: float=0.1, min_fraction_expressing: float=0.05, min_fraction_difference: float=0.05, min_log2_fold_change: float=0.25) -> pd.DataFrame:
    try:
        from scipy import sparse
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError('Cluster marker metrics require scipy') from error
    raw = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    n_cells = raw.shape[0]
    totals = np.asarray(raw.sum(axis=1)).ravel().astype(float)
    scale = np.divide(float(normalization_target), totals, out=np.zeros_like(totals, dtype=float), where=totals > 0)
    normalized = raw.multiply(scale[:, None]).tocsr()
    log_normalized = normalized.copy()
    log_normalized.data = np.log1p(log_normalized.data)
    detected = raw.copy()
    detected.data = np.ones_like(detected.data, dtype=np.uint8)
    total_normalized = np.asarray(normalized.sum(axis=0)).ravel().astype(float)
    total_log = np.asarray(log_normalized.sum(axis=0)).ravel().astype(float)
    total_detected = np.asarray(detected.sum(axis=0)).ravel().astype(float)
    var_names = pd.Index(adata.var_names.astype(str))
    if 'gene_symbol' in adata.var.columns:
        genes = adata.var['gene_symbol'].astype(str).to_numpy()
    else:
        genes = var_names.astype(str).to_numpy()
    label_values = pd.Series(labels.to_numpy(dtype=str), index=adata.obs_names.astype(str), name=cluster_key)
    clusters = sorted(label_values.dropna().astype(str).unique().tolist(), key=natural_key)
    rows: list[pd.DataFrame] = []
    for cluster in clusters:
        mask = label_values.astype(str).eq(str(cluster)).to_numpy()
        n_cluster = int(mask.sum())
        n_rest = int(n_cells - n_cluster)
        if n_cluster == 0:
            continue
        cluster_normalized_sum = np.asarray(normalized[mask, :].sum(axis=0)).ravel().astype(float)
        cluster_log_sum = np.asarray(log_normalized[mask, :].sum(axis=0)).ravel().astype(float)
        cluster_detected_sum = np.asarray(detected[mask, :].sum(axis=0)).ravel().astype(float)
        rest_normalized_sum = total_normalized - cluster_normalized_sum
        rest_log_sum = total_log - cluster_log_sum
        rest_detected_sum = total_detected - cluster_detected_sum
        mean_cluster = cluster_normalized_sum / max(n_cluster, 1)
        mean_rest = rest_normalized_sum / max(n_rest, 1) if n_rest else np.zeros_like(mean_cluster)
        mean_log_cluster = cluster_log_sum / max(n_cluster, 1)
        mean_log_rest = rest_log_sum / max(n_rest, 1) if n_rest else np.zeros_like(mean_log_cluster)
        fraction_cluster = cluster_detected_sum / max(n_cluster, 1)
        fraction_rest = rest_detected_sum / max(n_rest, 1) if n_rest else np.zeros_like(fraction_cluster)
        log2_fc = np.log2((mean_cluster + float(lfc_pseudocount)) / (mean_rest + float(lfc_pseudocount)))
        fraction_difference = fraction_cluster - fraction_rest
        marker_rank_score = np.maximum(log2_fc, 0) * np.maximum(fraction_difference, 0)
        rows.append(pd.DataFrame({'dataset': dataset, 'sample': sample, 'cluster_key': cluster_key, 'cluster': str(cluster), 'var_name': var_names.to_numpy(dtype=str), 'gene': genes, 'n_cluster_cells': n_cluster, 'n_rest_cells': n_rest, 'mean_normalized_expression_cluster': mean_cluster, 'mean_normalized_expression_rest': mean_rest, 'mean_log1p_expression_cluster': mean_log_cluster, 'mean_log1p_expression_rest': mean_log_rest, 'fraction_expressing_cluster': fraction_cluster, 'fraction_expressing_rest': fraction_rest, 'fraction_expressing_difference': fraction_difference, 'log2_fold_change': log2_fc, 'marker_rank_score': marker_rank_score}))
    if not rows:
        return pd.DataFrame()
    table = pd.concat(rows, ignore_index=True)
    table['passes_min_fraction_expressing'] = table['fraction_expressing_cluster'].ge(float(min_fraction_expressing))
    table['passes_min_fraction_difference'] = table['fraction_expressing_difference'].ge(float(min_fraction_difference))
    table['passes_min_log2_fold_change'] = table['log2_fold_change'].ge(float(min_log2_fold_change))
    table['passes_basic_marker_filter'] = table['passes_min_fraction_expressing'] & table['passes_min_fraction_difference'] & table['passes_min_log2_fold_change']
    table = table.sort_values(['dataset', 'sample', 'cluster_key', 'cluster', 'passes_basic_marker_filter', 'marker_rank_score', 'log2_fold_change', 'fraction_expressing_difference', 'fraction_expressing_cluster', 'gene'], ascending=[True, True, True, True, False, False, False, False, False, True], key=lambda column: column.map(natural_key) if column.name in {'dataset', 'sample', 'cluster', 'gene'} else column).reset_index(drop=True)
    table['rank_by_marker_score'] = table.groupby('cluster', dropna=False).cumcount() + 1
    return table

def run_scanpy_wilcoxon_markers(adata: Any, labels: pd.Series, cluster_key: str, *, normalization_target: float=100000.0, max_cells: int | None=100000, random_state: int=1729) -> pd.DataFrame:
    try:
        import scanpy as sc
    except ModuleNotFoundError:
        return pd.DataFrame()
    marker_adata = adata.copy()
    marker_adata.obs[cluster_key] = pd.Categorical(labels.reindex(marker_adata.obs_names.astype(str)).astype(str))
    if max_cells is not None and marker_adata.n_obs > int(max_cells):
        marker_adata = sc.pp.subsample(marker_adata, n_obs=int(max_cells), random_state=int(random_state), copy=True)
    sc.pp.normalize_total(marker_adata, target_sum=float(normalization_target))
    sc.pp.log1p(marker_adata)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sc.tl.rank_genes_groups(marker_adata, groupby=cluster_key, method='wilcoxon', n_genes=marker_adata.n_vars)
    try:
        table = sc.get.rank_genes_groups_df(marker_adata, group=None)
    except Exception:
        return pd.DataFrame()
    if table.empty:
        return pd.DataFrame()
    table = table.rename(columns={'group': 'cluster', 'names': 'var_name', 'scores': 'wilcoxon_score', 'logfoldchanges': 'scanpy_logfoldchange', 'pvals': 'wilcoxon_pval', 'pvals_adj': 'wilcoxon_pval_adj'})
    table['cluster'] = table['cluster'].astype(str)
    table['var_name'] = table['var_name'].astype(str)
    table['wilcoxon_n_cells_used'] = int(marker_adata.n_obs)
    return table[[column for column in ['cluster', 'var_name', 'wilcoxon_score', 'scanpy_logfoldchange', 'wilcoxon_pval', 'wilcoxon_pval_adj', 'wilcoxon_n_cells_used'] if column in table.columns]].copy()

def add_wilcoxon_support_indicator(marker_table: pd.DataFrame, *, alpha: float=0.05) -> pd.DataFrame:
    """Record Wilcoxon FDR support without making it a marker-eligibility rule."""
    table = marker_table.copy()
    if 'wilcoxon_pval_adj' not in table.columns:
        table['has_wilcoxon_fdr_support'] = pd.Series(pd.NA, index=table.index, dtype='boolean')
        return table
    adjusted = pd.to_numeric(table['wilcoxon_pval_adj'], errors='coerce')
    table['has_wilcoxon_fdr_support'] = adjusted.lt(float(alpha)).where(adjusted.notna(), pd.NA).astype('boolean')
    return table

def top_cluster_expression_markers(marker_table: pd.DataFrame, *, top_n: int=25) -> pd.DataFrame:
    if marker_table.empty:
        return pd.DataFrame()
    sort_columns = ['cluster', 'passes_basic_marker_filter', 'marker_rank_score', 'log2_fold_change', 'fraction_expressing_difference', 'fraction_expressing_cluster', 'wilcoxon_pval_adj', 'gene']
    available_columns = [column for column in sort_columns if column in marker_table.columns]
    ascending = []
    for column in available_columns:
        if column in {'cluster', 'gene'}:
            ascending.append(True)
        elif column == 'wilcoxon_pval_adj':
            ascending.append(True)
        else:
            ascending.append(False)
    table = marker_table.sort_values(available_columns, ascending=ascending, key=lambda column: column.map(natural_key) if column.name in {'cluster', 'gene'} else column).copy()
    table['rank_for_annotation'] = table.groupby('cluster', dropna=False).cumcount() + 1
    return table.loc[table['rank_for_annotation'].le(int(top_n))].reset_index(drop=True)

def summarize_cluster_expression_markers(top_markers: pd.DataFrame) -> pd.DataFrame:
    if top_markers.empty:
        return pd.DataFrame()
    rows = []
    for (dataset, sample, cluster_key, cluster), table in top_markers.groupby(['dataset', 'sample', 'cluster_key', 'cluster'], dropna=False):
        best = table.iloc[0]
        eligible = table['passes_basic_marker_filter'].fillna(False).astype(bool) if 'passes_basic_marker_filter' in table.columns else pd.Series(True, index=table.index)
        rows.append({'dataset': dataset, 'sample': sample, 'cluster_key': cluster_key, 'cluster': cluster, 'n_cluster_cells': best.get('n_cluster_cells', np.nan), 'top_gene': best.get('gene', ''), 'top_gene_log2_fold_change': best.get('log2_fold_change', np.nan), 'top_gene_fraction_expressing_cluster': best.get('fraction_expressing_cluster', np.nan), 'top_gene_fraction_expressing_rest': best.get('fraction_expressing_rest', np.nan), 'top_gene_wilcoxon_pval_adj': best.get('wilcoxon_pval_adj', np.nan), 'top_marker_genes': ', '.join(table['gene'].astype(str).head(10).tolist()), 'n_eligible_marker_genes_retained': int(eligible.sum()), 'has_eligible_marker_genes': bool(eligible.any())})
    return pd.DataFrame(rows).sort_values(['dataset', 'sample', 'cluster_key', 'cluster'], key=lambda column: column.map(natural_key)).reset_index(drop=True)

def compute_cluster_expression_markers(dataset: str, sample: str, cluster_key: str, *, root: str | Path=PROJECT_ROOT, output_dir: str | Path | None=None, annotation_dir: str | Path | None=None, inventory: pd.DataFrame | None=None, normalization_target: float=100000.0, lfc_pseudocount: float=0.1, min_fraction_expressing: float=0.05, min_fraction_difference: float=0.05, min_log2_fold_change: float=0.25, top_n: int=25, run_wilcoxon: bool=True, max_cells_for_wilcoxon: int | None=100000, random_state: int=1729, write_outputs: bool=True) -> dict[str, pd.DataFrame | dict[str, Path] | Path | None]:
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir or root / 'cell_clustering').expanduser().resolve()
    inventory = inventory if inventory is not None else build_cluster_review_inventory(root, output_dir=output_dir, annotation_dir=annotation_dir)
    rows = inventory.loc[inventory['dataset'].astype(str).eq(str(dataset)) & inventory['sample'].astype(str).eq(str(sample))]
    if rows.empty:
        raise KeyError(f'No review inventory row for {dataset} / {sample}')
    row = rows.iloc[0]
    cluster_path = resolve_existing_path(row.get('cluster_path', ''), root)
    matrix_path = resolve_existing_path(row.get('matrix_path', ''), root)
    if cluster_path is None:
        raise FileNotFoundError(f'Cluster output missing for {dataset} / {sample}')
    if matrix_path is None:
        raise FileNotFoundError(f'Feature matrix missing for {dataset} / {sample}')
    clusters = pd.read_csv(cluster_path)
    clusters['cell_id'] = clusters['cell_id'].astype(str)
    adata = load_feature_matrix_as_adata(matrix_path)
    adata_subset, labels, matched_clusters = align_adata_to_cluster_table(adata, clusters, cluster_key)
    marker_table = calculate_cluster_expression_marker_metrics(adata_subset, labels, dataset=dataset, sample=sample, cluster_key=cluster_key, normalization_target=normalization_target, lfc_pseudocount=lfc_pseudocount, min_fraction_expressing=min_fraction_expressing, min_fraction_difference=min_fraction_difference, min_log2_fold_change=min_log2_fold_change)
    wilcoxon_table = pd.DataFrame()
    if run_wilcoxon:
        wilcoxon_table = run_scanpy_wilcoxon_markers(adata_subset, labels, cluster_key, normalization_target=normalization_target, max_cells=max_cells_for_wilcoxon, random_state=random_state)
        if not wilcoxon_table.empty:
            marker_table = marker_table.merge(wilcoxon_table, on=['cluster', 'var_name'], how='left')
    marker_table = add_wilcoxon_support_indicator(marker_table)
    top_markers = top_cluster_expression_markers(marker_table, top_n=top_n)
    summary = summarize_cluster_expression_markers(top_markers)
    paths = cluster_expression_marker_table_paths(output_dir, dataset, sample, cluster_key)
    if write_outputs:
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        marker_table.to_csv(paths['all_genes'], index=False, compression='gzip')
        top_markers.to_csv(paths['top_genes'], index=False)
        summary.to_csv(paths['summary'], index=False)
    print(f'{dataset} / {sample} / {cluster_key}')
    print(f'Cluster table: {path_for_display(cluster_path, root)}')
    print(f'Feature matrix: {path_for_display(matrix_path, root)}')
    print(f'Cells with both cluster labels and expression matrix rows: {len(matched_clusters):,}')
    print(f'Genes tested: {adata_subset.n_vars:,}')
    print(f'Clusters: {labels.nunique():,}')
    if run_wilcoxon and wilcoxon_table.empty:
        print('Wilcoxon p-values were not added. Check whether scanpy is available and the sample has enough cells.')
    return {'all_genes': marker_table, 'top_genes': top_markers, 'summary': summary, 'saved_paths': paths, 'cluster_path': cluster_path, 'matrix_path': matrix_path}

def run_cluster_expression_marker_discovery(root: str | Path=PROJECT_ROOT, *, output_dir: str | Path | None=None, annotation_dir: str | Path | None=None, cluster_key: str='leiden_r1_3', dataset_names: list[str] | tuple[str, ...] | set[str] | None=None, sample_names: list[str] | tuple[str, ...] | set[str] | None=None, skip_existing: bool=True, normalization_target: float=100000.0, lfc_pseudocount: float=0.1, min_fraction_expressing: float=0.05, min_fraction_difference: float=0.05, min_log2_fold_change: float=0.25, top_n: int=25, run_wilcoxon: bool=True, max_cells_for_wilcoxon: int | None=100000, random_state: int=1729, write_outputs: bool=True) -> dict[str, pd.DataFrame]:
    """Discover cluster-vs-rest expression marker genes for all ready samples at one cluster key."""
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir or root / 'cell_clustering').expanduser().resolve()
    annotation_dir = Path(annotation_dir or root / 'cell_annotation').expanduser().resolve()
    review_inventory = build_cluster_review_inventory(root, output_dir=output_dir, annotation_dir=annotation_dir)
    marker_inventory = build_cluster_expression_marker_inventory(root, output_dir=output_dir, annotation_dir=annotation_dir)
    if review_inventory.empty:
        empty = pd.DataFrame()
        return {'marker_discovery_summary': empty, 'combined_top_markers': empty, 'combined_marker_summary': empty, 'marker_inventory': marker_inventory}
    ready_samples = review_inventory.loc[review_inventory.get('ready_for_marker_gene_review', False).astype(bool) if 'ready_for_marker_gene_review' in review_inventory.columns else pd.Series(False, index=review_inventory.index)].copy()
    if dataset_names is not None:
        keep = set(map(str, dataset_names))
        ready_samples = ready_samples.loc[ready_samples['dataset'].astype(str).isin(keep)].copy()
    if sample_names is not None:
        keep = set(map(str, sample_names))
        ready_samples = ready_samples.loc[ready_samples['sample'].astype(str).isin(keep)].copy()
    rows: list[dict[str, object]] = []
    top_tables: list[pd.DataFrame] = []
    marker_summaries: list[pd.DataFrame] = []
    marker_lookup = marker_inventory.set_index(['dataset', 'sample', 'cluster_key']) if not marker_inventory.empty else pd.DataFrame()
    for sample_row in ready_samples.sort_values(['dataset', 'sample'], key=lambda column: column.map(natural_key)).itertuples(index=False):
        dataset = str(sample_row.dataset)
        sample = str(sample_row.sample)
        marker_index = (dataset, sample, str(cluster_key))
        paths = cluster_expression_marker_table_paths(output_dir, dataset, sample, str(cluster_key))
        record: dict[str, object] = {'dataset': dataset, 'sample': sample, 'cluster_key': str(cluster_key), 'status': 'not_run', 'skipped_existing': False, 'top_genes_path': path_for_record(paths['top_genes'], root), 'all_genes_path': path_for_record(paths['all_genes'], root), 'summary_path': path_for_record(paths['summary'], root), 'n_top_marker_rows': 0, 'n_marker_summary_rows': 0, 'n_clusters_with_markers': 0, 'error': ''}
        if marker_lookup.empty or marker_index not in marker_lookup.index:
            record['status'] = 'missing_cluster_key'
            record['error'] = f'{cluster_key} is not available in cell_clusters.csv.gz'
            rows.append(record)
            continue
        has_existing_outputs = all((path.is_file() for path in paths.values()))
        if skip_existing and has_existing_outputs:
            top_markers = load_cluster_expression_top_markers(output_dir, dataset, sample, str(cluster_key))
            summary = pd.read_csv(paths['summary']) if paths['summary'].is_file() else summarize_cluster_expression_markers(top_markers)
            if not top_markers.empty:
                top_tables.append(top_markers)
            if not summary.empty:
                marker_summaries.append(summary)
            record.update({'status': 'skipped_existing', 'skipped_existing': True, 'n_top_marker_rows': int(len(top_markers)), 'n_marker_summary_rows': int(len(summary)), 'n_clusters_with_markers': int(top_markers['cluster'].nunique()) if 'cluster' in top_markers else 0})
            rows.append(record)
            continue
        try:
            result = compute_cluster_expression_markers(dataset, sample, str(cluster_key), root=root, output_dir=output_dir, annotation_dir=annotation_dir, inventory=review_inventory, normalization_target=normalization_target, lfc_pseudocount=lfc_pseudocount, min_fraction_expressing=min_fraction_expressing, min_fraction_difference=min_fraction_difference, min_log2_fold_change=min_log2_fold_change, top_n=top_n, run_wilcoxon=run_wilcoxon, max_cells_for_wilcoxon=max_cells_for_wilcoxon, random_state=random_state, write_outputs=write_outputs)
            top_markers = result['top_genes'] if isinstance(result.get('top_genes'), pd.DataFrame) else pd.DataFrame()
            summary = result['summary'] if isinstance(result.get('summary'), pd.DataFrame) else pd.DataFrame()
            if not top_markers.empty:
                top_tables.append(top_markers)
            if not summary.empty:
                marker_summaries.append(summary)
            record.update({'status': 'success', 'n_top_marker_rows': int(len(top_markers)), 'n_marker_summary_rows': int(len(summary)), 'n_clusters_with_markers': int(top_markers['cluster'].nunique()) if 'cluster' in top_markers else 0})
        except Exception as error:
            record['status'] = 'failed'
            record['error'] = f'{type(error).__name__}: {error}'
        rows.append(record)
    run_summary = pd.DataFrame(rows)
    combined_top = pd.concat(top_tables, ignore_index=True) if top_tables else pd.DataFrame()
    combined_summary = pd.concat(marker_summaries, ignore_index=True) if marker_summaries else pd.DataFrame()
    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = safe_name(cluster_key)
        run_summary.to_csv(output_dir / f'cluster_expression_marker_discovery_summary__{suffix}.csv', index=False)
        if not combined_top.empty:
            combined_top.to_csv(output_dir / f'combined_cluster_expression_top_markers__{suffix}.csv', index=False)
        if not combined_summary.empty:
            combined_summary.to_csv(output_dir / f'combined_cluster_expression_marker_summary__{suffix}.csv', index=False)
    return {'marker_discovery_summary': run_summary, 'combined_top_markers': combined_top, 'combined_marker_summary': combined_summary, 'marker_inventory': marker_inventory}

def compute_cluster_source_expression_markers(dataset: str, sample: str, clustering_source: str, cluster_key: str, *, root: str | Path=PROJECT_ROOT, output_dir: str | Path | None=None, annotation_dir: str | Path | None=None, gene_panel_review_dir: str | Path | None=None, inventory: pd.DataFrame | None=None, normalization_target: float=100000.0, lfc_pseudocount: float=0.1, min_fraction_expressing: float=0.05, min_fraction_difference: float=0.05, min_log2_fold_change: float=0.25, top_n: int=25, run_wilcoxon: bool=True, max_cells_for_wilcoxon: int | None=100000, random_state: int=1729, min_cell_id_match_fraction: float=0.8, write_outputs: bool=True) -> dict[str, object]:
    """Compute cluster-vs-rest markers for one Leiden or XOA partition."""
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir or root / 'cell_clustering').expanduser().resolve()
    method_parameters = marker_method_parameters(normalization_target=normalization_target, lfc_pseudocount=lfc_pseudocount, min_fraction_expressing=min_fraction_expressing, min_fraction_difference=min_fraction_difference, min_log2_fold_change=min_log2_fold_change, top_n=top_n, run_wilcoxon=run_wilcoxon, max_cells_for_wilcoxon=max_cells_for_wilcoxon, random_state=random_state)
    method_signature = marker_method_signature(**method_parameters)
    inventory = inventory if inventory is not None else build_cluster_source_inventory(root, output_dir=output_dir, annotation_dir=annotation_dir, gene_panel_review_dir=gene_panel_review_dir)
    selected = inventory.loc[inventory['dataset'].astype(str).eq(str(dataset)) & inventory['sample'].astype(str).eq(str(sample)) & inventory['clustering_source'].astype(str).eq(str(clustering_source)) & inventory['cluster_key'].astype(str).eq(str(cluster_key))]
    if selected.empty:
        raise KeyError(f'No cluster source found for {dataset} / {sample} / {clustering_source} / {cluster_key}')
    source_row = selected.iloc[0]
    matrix_path = resolve_existing_path(source_row.get('matrix_path', ''), root)
    if matrix_path is None:
        raise FileNotFoundError(f'Feature matrix missing for {dataset} / {sample}')
    adata = load_feature_matrix_as_adata(matrix_path)
    cell_id_diagnostics: dict[str, object] = {'cell_id_column_used': str(source_row.get('cell_id_column', '')), 'cell_id_match_method': 'exact', 'n_candidate_id_columns': 1, 'candidate_id_columns': str(source_row.get('cell_id_column', '')), 'candidate_overlap_summary': '', 'unmatched_cell_id_examples': ''}
    if str(clustering_source) == 'xoa':
        all_assignments, cluster_path, cell_id_diagnostics = load_cluster_source_assignments_for_matrix(source_row, pd.Index(adata.obs_names.astype(str)), root=root)
    else:
        all_assignments, cluster_path = load_cluster_source_assignments(source_row, root=root)
    assignments = informative_cluster_assignments(all_assignments, str(cluster_key))
    if assignments.empty:
        raise ValueError('The cluster source contains no assigned cluster IDs')
    adata_subset, labels, matched_assignments = align_adata_to_cluster_table(adata, assignments, str(cluster_key))
    match_fraction = len(matched_assignments) / len(assignments) if len(assignments) else 0.0
    if match_fraction < float(min_cell_id_match_fraction):
        raise ValueError(f"Only {match_fraction:.1%} of cluster-assignment cell IDs matched the expression matrix; minimum required is {float(min_cell_id_match_fraction):.1%}. ID column={cell_id_diagnostics['cell_id_column_used']}; method={cell_id_diagnostics['cell_id_match_method']}; candidate overlaps={cell_id_diagnostics['candidate_overlap_summary']}; unmatched examples={cell_id_diagnostics['unmatched_cell_id_examples']}")
    marker_table = calculate_cluster_expression_marker_metrics(adata_subset, labels, dataset=str(dataset), sample=str(sample), cluster_key=str(cluster_key), normalization_target=normalization_target, lfc_pseudocount=lfc_pseudocount, min_fraction_expressing=min_fraction_expressing, min_fraction_difference=min_fraction_difference, min_log2_fold_change=min_log2_fold_change)
    marker_table.insert(2, 'clustering_source', str(clustering_source))
    wilcoxon_table = pd.DataFrame()
    if run_wilcoxon:
        wilcoxon_table = run_scanpy_wilcoxon_markers(adata_subset, labels, str(cluster_key), normalization_target=normalization_target, max_cells=max_cells_for_wilcoxon, random_state=random_state)
        if not wilcoxon_table.empty:
            marker_table = marker_table.merge(wilcoxon_table, on=['cluster', 'var_name'], how='left')
    marker_table = add_wilcoxon_support_indicator(marker_table)
    top_markers = top_cluster_expression_markers(marker_table, top_n=top_n)
    summary = summarize_cluster_expression_markers(top_markers)
    if not summary.empty:
        summary.insert(2, 'clustering_source', str(clustering_source))
        summary['n_source_cells'] = int(len(assignments))
        summary['n_expression_matched_cells'] = int(len(matched_assignments))
        summary['cell_id_match_fraction'] = float(match_fraction)
        for column, value in cell_id_diagnostics.items():
            summary[column] = value
        for parameter, value in method_parameters.items():
            summary[f'marker_{parameter}'] = value
        summary['marker_method_signature'] = method_signature
    paths = cluster_expression_marker_table_paths(output_dir, str(dataset), str(sample), str(cluster_key), str(clustering_source))
    if write_outputs:
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        marker_table.to_csv(paths['all_genes'], index=False, compression='gzip')
        top_markers.to_csv(paths['top_genes'], index=False)
        summary.to_csv(paths['summary'], index=False)
    print(f'{dataset} / {sample} / {CLUSTERING_SOURCE_LABELS.get(clustering_source, clustering_source)}')
    print(f'Cluster table: {path_for_display(cluster_path, root)}')
    print(f'Feature matrix: {path_for_display(matrix_path, root)}')
    print(f'Cell-ID match: {len(matched_assignments):,} / {len(assignments):,} ({match_fraction:.1%})')
    print(f"Cell-ID column/method: {cell_id_diagnostics['cell_id_column_used']} / {cell_id_diagnostics['cell_id_match_method']}")
    print(f'Clusters: {labels.nunique():,}; genes tested: {adata_subset.n_vars:,}')
    if run_wilcoxon and wilcoxon_table.empty:
        print('Wilcoxon p-values were not added. Check Scanpy availability and cluster sizes.')
    return {'all_genes': marker_table, 'top_genes': top_markers, 'summary': summary, 'saved_paths': paths, 'cluster_path': cluster_path, 'matrix_path': matrix_path, 'n_source_cells': int(len(assignments)), 'n_expression_matched_cells': int(len(matched_assignments)), 'cell_id_match_fraction': float(match_fraction), 'marker_method_signature': method_signature, **cell_id_diagnostics}

def run_cluster_source_expression_marker_discovery(root: str | Path=PROJECT_ROOT, *, output_dir: str | Path | None=None, annotation_dir: str | Path | None=None, gene_panel_review_dir: str | Path | None=None, our_cluster_key: str | None='leiden_r1_3', our_cluster_keys: list[str] | tuple[str, ...] | set[str] | None=None, clustering_sources: tuple[str, ...]=('our_expression',), skip_existing: bool=True, normalization_target: float=100000.0, lfc_pseudocount: float=0.1, min_fraction_expressing: float=0.05, min_fraction_difference: float=0.05, min_log2_fold_change: float=0.25, top_n: int=25, run_wilcoxon: bool=True, max_cells_for_wilcoxon: int | None=100000, random_state: int=1729, min_cell_id_match_fraction: float=0.8, write_outputs: bool=True) -> dict[str, pd.DataFrame]:
    """Run the same marker discovery for selected Leiden partitions and XOA once."""
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir or root / 'cell_clustering').expanduser().resolve()
    selected_our_cluster_keys = normalize_cluster_key_selection(our_cluster_keys, fallback_cluster_key=our_cluster_key)
    method_parameters = marker_method_parameters(normalization_target=normalization_target, lfc_pseudocount=lfc_pseudocount, min_fraction_expressing=min_fraction_expressing, min_fraction_difference=min_fraction_difference, min_log2_fold_change=min_log2_fold_change, top_n=top_n, run_wilcoxon=run_wilcoxon, max_cells_for_wilcoxon=max_cells_for_wilcoxon, random_state=random_state)
    expected_method_signature = marker_method_signature(**method_parameters)
    inventory = build_cluster_source_inventory(root, output_dir=output_dir, annotation_dir=annotation_dir, gene_panel_review_dir=gene_panel_review_dir)
    if inventory.empty:
        empty = pd.DataFrame()
        return {'marker_discovery_summary': empty, 'combined_top_markers': empty, 'combined_marker_summary': empty, 'cluster_source_inventory': empty}
    selected = inventory.loc[inventory['clustering_source'].astype(str).isin(set(map(str, clustering_sources))) & inventory['ready_for_marker_gene_review'].astype(bool)].copy()
    selected = selected.loc[~selected['clustering_source'].astype(str).eq('our_expression') | selected['cluster_key'].astype(str).isin(set(selected_our_cluster_keys))].copy()
    records: list[dict[str, object]] = []
    top_tables: dict[tuple[str, str, str, str], pd.DataFrame] = {}
    summary_tables: dict[tuple[str, str, str, str], pd.DataFrame] = {}
    for row in selected.sort_values(['clustering_source', 'dataset', 'sample', 'cluster_key'], key=lambda column: column.map(natural_key), kind='stable').itertuples(index=False):
        dataset = str(row.dataset)
        sample = str(row.sample)
        clustering_source = str(row.clustering_source)
        cluster_key = str(row.cluster_key)
        partition_key = (dataset, sample, clustering_source, cluster_key)
        paths = cluster_expression_marker_table_paths(output_dir, dataset, sample, cluster_key, clustering_source)
        record: dict[str, object] = {'dataset': dataset, 'sample': sample, 'clustering_source': clustering_source, 'cluster_key': cluster_key, 'status': 'not_run', 'skipped_existing': False, 'cache_validation_status': 'not_checked', 'cache_validation_reason': '', 'marker_method_signature': expected_method_signature, 'n_source_cells': np.nan, 'n_expression_matched_cells': np.nan, 'cell_id_match_fraction': np.nan, 'n_clusters_with_markers': 0, 'cluster_source_path': str(getattr(row, 'cluster_path', '')), 'matrix_path': str(getattr(row, 'matrix_path', '')), 'configured_cell_id_column': str(getattr(row, 'cell_id_column', '')), 'cell_id_column_used': '', 'cell_id_match_method': '', 'n_candidate_id_columns': np.nan, 'candidate_id_columns': '', 'candidate_overlap_summary': '', 'unmatched_cell_id_examples': '', 'top_genes_path': path_for_record(paths['top_genes'], root), 'all_genes_path': path_for_record(paths['all_genes'], root), 'summary_path': path_for_record(paths['summary'], root), 'error': ''}
        try:
            reuse_cache = False
            if skip_existing:
                cache_valid, cache_reason = validate_cached_cluster_expression_marker_tables(pd.Series(row._asdict()), paths, root=root, top_n=top_n, expected_method_signature=expected_method_signature)
                reuse_cache = cache_valid
                record['cache_validation_status'] = 'valid_reused' if cache_valid else 'invalid_recomputed'
                record['cache_validation_reason'] = cache_reason
            else:
                record['cache_validation_status'] = 'disabled_recomputed'
                record['cache_validation_reason'] = 'skip_existing_is_false'
            if reuse_cache:
                top_markers = pd.read_csv(paths['top_genes'], dtype={'cluster': str})
                summary = pd.read_csv(paths['summary'], dtype={'cluster': str})
                if 'clustering_source' not in top_markers.columns:
                    top_markers.insert(2, 'clustering_source', clustering_source)
                if 'clustering_source' not in summary.columns:
                    summary.insert(2, 'clustering_source', clustering_source)
                record['status'] = 'skipped_existing'
                record['skipped_existing'] = True
            else:
                result = compute_cluster_source_expression_markers(dataset, sample, clustering_source, cluster_key, root=root, output_dir=output_dir, annotation_dir=annotation_dir, gene_panel_review_dir=gene_panel_review_dir, inventory=inventory, normalization_target=normalization_target, lfc_pseudocount=lfc_pseudocount, min_fraction_expressing=min_fraction_expressing, min_fraction_difference=min_fraction_difference, min_log2_fold_change=min_log2_fold_change, top_n=top_n, run_wilcoxon=run_wilcoxon, max_cells_for_wilcoxon=max_cells_for_wilcoxon, random_state=random_state, min_cell_id_match_fraction=min_cell_id_match_fraction, write_outputs=write_outputs)
                top_markers = result['top_genes']
                summary = result['summary']
                record.update({'status': 'success', 'n_source_cells': result['n_source_cells'], 'n_expression_matched_cells': result['n_expression_matched_cells'], 'cell_id_match_fraction': result['cell_id_match_fraction'], 'marker_method_signature': result.get('marker_method_signature', expected_method_signature), 'cell_id_column_used': result.get('cell_id_column_used', ''), 'cell_id_match_method': result.get('cell_id_match_method', ''), 'n_candidate_id_columns': result.get('n_candidate_id_columns', np.nan), 'candidate_id_columns': result.get('candidate_id_columns', ''), 'candidate_overlap_summary': result.get('candidate_overlap_summary', ''), 'unmatched_cell_id_examples': result.get('unmatched_cell_id_examples', '')})
            if not top_markers.empty:
                top_tables[partition_key] = top_markers
                record['n_clusters_with_markers'] = int(top_markers['cluster'].nunique())
            if not summary.empty:
                summary_tables[partition_key] = summary
                for column in ('n_source_cells', 'n_expression_matched_cells', 'cell_id_match_fraction', 'cell_id_column_used', 'cell_id_match_method', 'n_candidate_id_columns', 'candidate_id_columns', 'candidate_overlap_summary', 'unmatched_cell_id_examples'):
                    if column not in summary.columns:
                        continue
                    current = record[column]
                    missing = pd.isna(current) if not isinstance(current, str) else not current
                    if missing:
                        record[column] = summary.iloc[0][column]
        except Exception as error:
            record['status'] = 'failed'
            record['error'] = f'{type(error).__name__}: {error}'
        records.append(record)
    if write_outputs:
        comprehensive = inventory.loc[inventory['clustering_source'].astype(str).isin(set(CLUSTERING_SOURCE_LABELS)) & inventory['has_cluster_output'].astype(bool)].copy()
        comprehensive = comprehensive.loc[~comprehensive['clustering_source'].astype(str).eq('our_expression') | comprehensive['cluster_key'].astype(str).isin(set(selected_our_cluster_keys))].copy()
        recorded_keys = {(str(record['dataset']), str(record['sample']), str(record['clustering_source']), str(record['cluster_key'])) for record in records}
        for row in comprehensive.itertuples(index=False):
            dataset = str(row.dataset)
            sample = str(row.sample)
            clustering_source = str(row.clustering_source)
            cluster_key = str(row.cluster_key)
            partition_key = (dataset, sample, clustering_source, cluster_key)
            paths = cluster_expression_marker_table_paths(output_dir, dataset, sample, cluster_key, clustering_source)
            if partition_key not in top_tables or partition_key not in summary_tables:
                cache_valid, cache_reason = validate_cached_cluster_expression_marker_tables(pd.Series(row._asdict()), paths, root=root, top_n=top_n, expected_method_signature=expected_method_signature)
                if not cache_valid:
                    continue
                saved_top = pd.read_csv(paths['top_genes'], dtype={'cluster': str})
                saved_summary = pd.read_csv(paths['summary'], dtype={'cluster': str})
                if 'clustering_source' not in saved_top.columns:
                    saved_top.insert(2, 'clustering_source', clustering_source)
                if 'clustering_source' not in saved_summary.columns:
                    saved_summary.insert(2, 'clustering_source', clustering_source)
                top_tables[partition_key] = saved_top
                summary_tables[partition_key] = saved_summary
            else:
                cache_reason = 'already_loaded_from_current_run'
            if partition_key not in recorded_keys and partition_key in top_tables and (partition_key in summary_tables):
                saved_top = top_tables[partition_key]
                saved_summary = summary_tables[partition_key]
                first_summary = saved_summary.iloc[0] if not saved_summary.empty else pd.Series()
                records.append({'dataset': dataset, 'sample': sample, 'clustering_source': clustering_source, 'cluster_key': cluster_key, 'status': 'available_saved', 'skipped_existing': True, 'cache_validation_status': 'valid_reused', 'cache_validation_reason': cache_reason, 'marker_method_signature': first_summary.get('marker_method_signature', expected_method_signature), 'n_source_cells': first_summary.get('n_source_cells', np.nan), 'n_expression_matched_cells': first_summary.get('n_expression_matched_cells', np.nan), 'cell_id_match_fraction': first_summary.get('cell_id_match_fraction', np.nan), 'cluster_source_path': str(getattr(row, 'cluster_path', '')), 'matrix_path': str(getattr(row, 'matrix_path', '')), 'configured_cell_id_column': str(getattr(row, 'cell_id_column', '')), 'cell_id_column_used': first_summary.get('cell_id_column_used', ''), 'cell_id_match_method': first_summary.get('cell_id_match_method', ''), 'n_candidate_id_columns': first_summary.get('n_candidate_id_columns', np.nan), 'candidate_id_columns': first_summary.get('candidate_id_columns', ''), 'candidate_overlap_summary': first_summary.get('candidate_overlap_summary', ''), 'unmatched_cell_id_examples': first_summary.get('unmatched_cell_id_examples', ''), 'n_clusters_with_markers': int(saved_top['cluster'].nunique()) if not saved_top.empty and 'cluster' in saved_top.columns else 0, 'top_genes_path': path_for_record(paths['top_genes'], root), 'all_genes_path': path_for_record(paths['all_genes'], root), 'summary_path': path_for_record(paths['summary'], root), 'error': ''})
                recorded_keys.add(partition_key)
    run_summary = pd.DataFrame(records)
    combined_top = pd.concat(list(top_tables.values()), ignore_index=True) if top_tables else pd.DataFrame()
    combined_summary = pd.concat(list(summary_tables.values()), ignore_index=True) if summary_tables else pd.DataFrame()
    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        run_summary.to_csv(output_dir / 'cluster_source_expression_marker_discovery_summary.csv', index=False)
        if not combined_top.empty:
            combined_top.to_csv(output_dir / 'combined_cluster_source_expression_top_markers.csv', index=False)
        if not combined_summary.empty:
            combined_summary.to_csv(output_dir / 'combined_cluster_source_expression_marker_summary.csv', index=False)
    return {'marker_discovery_summary': run_summary, 'combined_top_markers': combined_top, 'combined_marker_summary': combined_summary, 'cluster_source_inventory': inventory}

def cluster_source_top_two_candidate_path(output_dir: str | Path, our_cluster_key: str | None=None) -> Path:
    if our_cluster_key:
        return Path(output_dir) / f'combined_cluster_source_top_two_gene_hint_candidates__{safe_name(our_cluster_key)}.csv'
    return Path(output_dir) / 'combined_cluster_source_top_two_gene_hint_candidates.csv'

def load_cluster_source_top_two_gene_hint_candidates(output_dir: str | Path, our_cluster_key: str | None=None) -> pd.DataFrame:
    path = cluster_source_top_two_candidate_path(output_dir)
    if not path.is_file() and our_cluster_key:
        path = cluster_source_top_two_candidate_path(output_dir, our_cluster_key)
    return pd.read_csv(path, dtype={'cluster': str}) if path.is_file() else pd.DataFrame()

def load_cluster_signature_scores_top_two(output_dir: str | Path) -> pd.DataFrame:
    path = Path(output_dir) / 'signature_scores' / 'combined_cluster_signature_scores_top_two.csv'
    return pd.read_csv(path, dtype={'cluster': str}) if path.is_file() else pd.DataFrame()

def build_cluster_source_top_two_gene_hint_candidates(root: str | Path=PROJECT_ROOT, *, output_dir: str | Path | None=None, annotation_dir: str | Path | None=None, gene_panel_review_dir: str | Path | None=None, our_cluster_key: str | None='leiden_r1_3', our_cluster_keys: list[str] | tuple[str, ...] | set[str] | None=None, clustering_sources: tuple[str, ...]=('our_expression',), write_outputs: bool=True) -> dict[str, object]:
    """Build one top-two automatic candidate table for Leiden and XOA clusters."""
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir or root / 'cell_clustering').expanduser().resolve()
    selected_our_cluster_keys = normalize_cluster_key_selection(our_cluster_keys, fallback_cluster_key=our_cluster_key)
    gene_panel_review_dir = Path(gene_panel_review_dir or root / 'gene_panel_review').expanduser().resolve()
    inventory = build_cluster_source_inventory(root, output_dir=output_dir, annotation_dir=annotation_dir, gene_panel_review_dir=gene_panel_review_dir)
    output_path = cluster_source_top_two_candidate_path(output_dir)
    if inventory.empty:
        return {'top_two_candidates': pd.DataFrame(), 'build_summary': pd.DataFrame(), 'output_path': output_path}
    selected = inventory.loc[inventory['clustering_source'].astype(str).isin(set(map(str, clustering_sources))) & inventory['has_cluster_output'].astype(bool)].copy()
    selected = selected.loc[~selected['clustering_source'].astype(str).eq('our_expression') | selected['cluster_key'].astype(str).isin(set(selected_our_cluster_keys))].copy()
    hint_sources = load_gene_marker_hint_sources(root, gene_panel_review_dir=gene_panel_review_dir)
    candidate_tables: list[pd.DataFrame] = []
    status_rows: list[dict[str, object]] = []
    for row in selected.sort_values(['clustering_source', 'dataset', 'sample', 'cluster_key'], key=lambda column: column.map(natural_key), kind='stable').itertuples(index=False):
        dataset = str(row.dataset)
        sample = str(row.sample)
        clustering_source = str(row.clustering_source)
        cluster_key = str(row.cluster_key)
        record: dict[str, object] = {'dataset': dataset, 'sample': sample, 'clustering_source': clustering_source, 'cluster_key': cluster_key, 'status': 'not_run', 'n_clusters_with_candidates': 0, 'error': ''}
        try:
            top_markers = load_cluster_expression_top_markers(output_dir, dataset, sample, cluster_key, clustering_source)
            if top_markers.empty:
                record['status'] = 'missing_top_markers'
                status_rows.append(record)
                continue
            annotated = annotate_cluster_top_markers_with_gene_hints(top_markers, root=root, gene_panel_review_dir=gene_panel_review_dir, hint_sources=hint_sources)
            group_candidates = summarize_cluster_marker_gene_group_candidates(annotated)
            top_two = top_two_gene_hint_candidates(annotated, group_candidates, clustering_source=clustering_source)
            if not top_two.empty:
                candidate_tables.append(top_two)
            record['status'] = 'success'
            record['n_clusters_with_candidates'] = int(len(top_two))
        except Exception as error:
            record['status'] = 'failed'
            record['error'] = f'{type(error).__name__}: {error}'
        status_rows.append(record)
    combined = pd.concat(candidate_tables, ignore_index=True, sort=False) if candidate_tables else pd.DataFrame()
    if not combined.empty:
        combined = combined.sort_values(['clustering_source', 'dataset', 'sample', 'cluster_key', 'cluster'], key=lambda column: column.map(natural_key), kind='stable').reset_index(drop=True)
    if write_outputs and (not combined.empty):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)
        if len(selected_our_cluster_keys) == 1:
            combined.to_csv(cluster_source_top_two_candidate_path(output_dir, selected_our_cluster_keys[0]), index=False)
    return {'top_two_candidates': combined, 'build_summary': pd.DataFrame(status_rows), 'output_path': output_path}

def build_cluster_expression_marker_inventory(root: str | Path=PROJECT_ROOT, *, output_dir: str | Path | None=None, annotation_dir: str | Path | None=None) -> pd.DataFrame:
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir or root / 'cell_clustering').expanduser().resolve()
    inventory = build_cluster_review_inventory(root, output_dir=output_dir, annotation_dir=annotation_dir)
    if inventory.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for row in inventory.itertuples(index=False):
        cluster_path = resolve_existing_path(row.cluster_path, root)
        if cluster_path is None:
            continue
        try:
            clusters = pd.read_csv(cluster_path, nrows=5)
        except Exception:
            continue
        for cluster_key in available_cluster_keys(clusters):
            paths = cluster_expression_marker_table_paths(output_dir, str(row.dataset), str(row.sample), cluster_key)
            rows.append({'dataset': str(row.dataset), 'sample': str(row.sample), 'cluster_key': cluster_key, 'ready_for_marker_gene_review': bool(getattr(row, 'ready_for_marker_gene_review', False)), 'has_saved_all_gene_table': paths['all_genes'].is_file(), 'has_saved_top_gene_table': paths['top_genes'].is_file(), 'has_saved_summary': paths['summary'].is_file(), 'top_genes_path': path_for_record(paths['top_genes'], root), 'all_genes_path': path_for_record(paths['all_genes'], root), 'summary_path': path_for_record(paths['summary'], root)})
    return pd.DataFrame(rows).sort_values(['dataset', 'sample', 'cluster_key'], key=lambda column: column.map(natural_key)).reset_index(drop=True)

def compare_clusters_to_marker_score_labels(root: str | Path=PROJECT_ROOT, *, output_dir: str | Path | None=None, annotation_dir: str | Path | None=None, default_cluster_key: str='leiden_r1_3', write_outputs: bool=True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir or root / 'cell_clustering').expanduser().resolve()
    inventory = build_cluster_review_inventory(root, output_dir=output_dir, annotation_dir=annotation_dir)
    cluster_annotation_rows: list[dict[str, object]] = []
    all_cluster_annotation_pairs: list[pd.DataFrame] = []
    if not inventory.empty:
        for row in inventory.loc[inventory['ready_for_review']].itertuples(index=False):
            dataset = str(row.dataset)
            sample = str(row.sample)
            cluster_path = resolve_existing_path(row.cluster_path, root)
            label_path = resolve_existing_path(row.marker_label_path, root)
            record: dict[str, object] = {'dataset': dataset, 'sample': sample, 'status': 'not_run', 'cluster_key': default_cluster_key, 'cluster_path': path_for_record(cluster_path, root) if cluster_path else '', 'marker_label_path': path_for_record(label_path, root) if label_path else '', 'n_clustered_cells': np.nan, 'n_marker_labeled_cells': np.nan, 'n_matched_cells': np.nan, 'matched_fraction_of_clustered_cells': np.nan, 'matched_fraction_of_marker_labeled_cells': np.nan, 'output_path': '', 'error': ''}
            sample_output_dir = output_dir / safe_name(dataset) / safe_name(sample)
            try:
                if cluster_path is None or label_path is None:
                    raise FileNotFoundError('Cluster table or marker-score label table is missing')
                clusters = pd.read_csv(cluster_path)
                labels = load_marker_score_labels(label_path)
                cluster_key = select_cluster_key(available_cluster_keys(clusters), default_cluster_key)
                record['cluster_key'] = cluster_key
                matched, pairs = cluster_label_pair_summary(clusters, labels, cluster_key)
                pairs.insert(0, 'sample', sample)
                pairs.insert(0, 'dataset', dataset)
                output_path = sample_output_dir / f'cluster_marker_score_label_pairs__{cluster_key}.csv'
                if write_outputs:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    pairs.to_csv(output_path, index=False)
                all_cluster_annotation_pairs.append(pairs)
                record.update({'status': 'compared', 'n_clustered_cells': len(clusters), 'n_marker_labeled_cells': len(labels), 'n_matched_cells': len(matched), 'matched_fraction_of_clustered_cells': len(matched) / len(clusters) if len(clusters) else np.nan, 'matched_fraction_of_marker_labeled_cells': len(matched) / len(labels) if len(labels) else np.nan, 'output_path': path_for_record(output_path, root)})
            except Exception as error:
                record.update({'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
            cluster_annotation_rows.append(record)
    comparison_summary = pd.DataFrame(cluster_annotation_rows)
    combined_pairs = pd.concat(all_cluster_annotation_pairs, ignore_index=True) if all_cluster_annotation_pairs else pd.DataFrame()
    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        inventory.to_csv(output_dir / 'cluster_review_inventory.csv', index=False)
        comparison_summary.to_csv(output_dir / 'cluster_marker_label_comparison_summary.csv', index=False)
        combined_pairs.to_csv(output_dir / 'combined_cluster_marker_label_pairs.csv', index=False)
    return (inventory, comparison_summary, combined_pairs)

def load_manual_cluster_annotations(manual_annotation_path: str | Path) -> pd.DataFrame:
    manual_annotation_path = Path(manual_annotation_path)
    if manual_annotation_path.is_file():
        table = pd.read_csv(manual_annotation_path, dtype=str).fillna('')
    else:
        table = pd.DataFrame(columns=MANUAL_CLUSTER_ANNOTATION_COLUMNS)
    for column in MANUAL_CLUSTER_ANNOTATION_COLUMNS:
        if column not in table.columns:
            table[column] = ''
    table['clustering_source'] = table['clustering_source'].replace('', 'our_expression')
    missing_refined = table['refined_annotation'].astype(str).str.strip().eq('')
    table.loc[missing_refined, 'refined_annotation'] = table.loc[missing_refined, 'manual_annotation']
    missing_manual = table['manual_annotation'].astype(str).str.strip().eq('')
    table.loc[missing_manual, 'manual_annotation'] = table.loc[missing_manual, 'refined_annotation']
    table['annotation_source'] = table['annotation_source'].replace('', 'expression_cluster_markers')
    return table[MANUAL_CLUSTER_ANNOTATION_COLUMNS].copy()

def upsert_manual_cluster_annotation(record: dict[str, object], manual_annotation_path: str | Path) -> pd.DataFrame:
    manual_annotation_path = Path(manual_annotation_path)
    table = load_manual_cluster_annotations(manual_annotation_path)
    key_columns = ['dataset', 'sample', 'clustering_source', 'cluster_key', 'cluster']
    record_text = {column: str(record.get(column, '')) for column in MANUAL_CLUSTER_ANNOTATION_COLUMNS}
    mask = pd.Series(False, index=table.index)
    if not table.empty:
        mask = pd.Series(True, index=table.index)
        for column in key_columns:
            mask &= table[column].astype(str).eq(record_text[column])
    table = table.loc[~mask].copy()
    table = pd.concat([table, pd.DataFrame([record_text])], ignore_index=True)
    table = table.sort_values(['clustering_source', 'dataset', 'sample', 'cluster_key', 'cluster'], key=lambda column: column.map(natural_key)).reset_index(drop=True)
    manual_annotation_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(manual_annotation_path, index=False)
    return table

def manual_annotation_for_cluster(manual_annotation_path: str | Path, dataset: str, sample: str, cluster_key: str, cluster: str, clustering_source: str='our_expression') -> pd.Series | None:
    table = load_manual_cluster_annotations(manual_annotation_path)
    rows = table.loc[table['dataset'].astype(str).eq(str(dataset)) & table['sample'].astype(str).eq(str(sample)) & table['clustering_source'].astype(str).eq(str(clustering_source)) & table['cluster_key'].astype(str).eq(str(cluster_key)) & table['cluster'].astype(str).eq(str(cluster))]
    return rows.iloc[-1] if not rows.empty else None

def export_manual_cluster_cell_labels(dataset: str, sample: str, cluster_key: str, *, root: str | Path=PROJECT_ROOT, output_dir: str | Path | None=None, annotation_dir: str | Path | None=None, manual_annotation_path: str | Path | None=None) -> Path:
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir or root / 'cell_clustering').expanduser().resolve()
    manual_annotation_path = Path(manual_annotation_path or output_dir / 'manual_cluster_annotations.csv')
    inventory = build_cluster_review_inventory(root, output_dir=output_dir, annotation_dir=annotation_dir)
    rows = inventory.loc[inventory['dataset'].astype(str).eq(str(dataset)) & inventory['sample'].astype(str).eq(str(sample))]
    if rows.empty:
        raise KeyError(f'No clustering output found for {dataset} / {sample}')
    cluster_path = resolve_existing_path(rows.iloc[0]['cluster_path'], root)
    if cluster_path is None:
        raise FileNotFoundError(f'Cluster output missing for {dataset} / {sample}')
    clusters = pd.read_csv(cluster_path)
    if cluster_key not in clusters.columns:
        raise KeyError(f'{cluster_key} is not in cluster table')
    manual = load_manual_cluster_annotations(manual_annotation_path)
    manual = manual.loc[manual['dataset'].astype(str).eq(str(dataset)) & manual['sample'].astype(str).eq(str(sample)) & manual['clustering_source'].astype(str).eq('our_expression') & manual['cluster_key'].astype(str).eq(str(cluster_key))].copy()
    if manual.empty:
        raise ValueError(f'No manual annotations recorded for {dataset} / {sample} / {cluster_key}')
    mapping_columns = ['cluster', 'manual_annotation', 'confidence', 'notes', 'suggested_gene_marker_group', 'suggested_gene_hint_confidence', 'suggested_gene_hint_supporting_genes']
    mapping = manual[[column for column in mapping_columns if column in manual.columns]].copy()
    mapping['cluster'] = mapping['cluster'].astype(str)
    output = clusters[['dataset', 'sample', 'cell_id', cluster_key]].copy()
    output = output.rename(columns={cluster_key: 'cluster'})
    output['cluster'] = output['cluster'].astype(str)
    output = output.merge(mapping, on='cluster', how='left')
    output = output.rename(columns={'manual_annotation': 'manual_cluster_annotation'})
    output_path = output_dir / safe_name(dataset) / safe_name(sample) / f'cell_manual_cluster_annotations__{cluster_key}.csv.gz'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, compression='gzip')
    return output_path

def export_manual_cluster_source_cell_labels(dataset: str, sample: str, clustering_source: str, cluster_key: str, *, root: str | Path=PROJECT_ROOT, output_dir: str | Path | None=None, annotation_dir: str | Path | None=None, gene_panel_review_dir: str | Path | None=None, manual_annotation_path: str | Path | None=None, inventory: pd.DataFrame | None=None) -> Path:
    """Join reviewed refined/broad annotations back to original cluster cells."""
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir or root / 'cell_clustering').expanduser().resolve()
    manual_annotation_path = Path(manual_annotation_path or output_dir / 'manual_cluster_annotations.csv')
    inventory = inventory if inventory is not None else build_cluster_source_inventory(root, output_dir=output_dir, annotation_dir=annotation_dir, gene_panel_review_dir=gene_panel_review_dir)
    rows = inventory.loc[inventory['dataset'].astype(str).eq(str(dataset)) & inventory['sample'].astype(str).eq(str(sample)) & inventory['clustering_source'].astype(str).eq(str(clustering_source)) & inventory['cluster_key'].astype(str).eq(str(cluster_key))]
    if rows.empty:
        raise KeyError(f'No cluster source found for {dataset} / {sample} / {clustering_source} / {cluster_key}')
        assignments, _cluster_path = load_cluster_source_assignments(rows.iloc[0], root=root)
    manual = load_manual_cluster_annotations(manual_annotation_path)
    manual = manual.loc[manual['dataset'].astype(str).eq(str(dataset)) & manual['sample'].astype(str).eq(str(sample)) & manual['clustering_source'].astype(str).eq(str(clustering_source)) & manual['cluster_key'].astype(str).eq(str(cluster_key))].copy()
    if manual.empty:
        raise ValueError(f'No manual annotations recorded for {dataset} / {sample} / {clustering_source} / {cluster_key}')
    mapping_columns = ['cluster', 'refined_annotation', 'broad_annotation', 'confidence', 'notes', 'annotation_source', 'automatic_candidate_1', 'automatic_candidate_2', 'automatic_refined_annotation', 'review_selection', 'candidate_1_supporting_genes', 'candidate_2_supporting_genes', 'candidate_1_support_fraction', 'candidate_2_support_fraction', 'suggested_gene_marker_group', 'suggested_gene_hint_confidence', 'suggested_gene_hint_supporting_genes']
    mapping = manual[mapping_columns].copy()
    mapping['cluster'] = mapping['cluster'].astype(str)
    output = assignments.rename(columns={str(cluster_key): 'cluster'}).copy()
    output['cluster'] = output['cluster'].astype(str)
    output = output.merge(mapping, on='cluster', how='left')
    output.insert(0, 'sample', str(sample))
    output.insert(0, 'dataset', str(dataset))
    output.insert(2, 'clustering_source', str(clustering_source))
    output.insert(3, 'cluster_key', str(cluster_key))
    output.insert(5, str(cluster_key), output['cluster'])
    output['manual_cluster_annotation'] = output['refined_annotation']
    source_suffix = '' if clustering_source == 'our_expression' else f'{safe_name(clustering_source)}__'
    output_path = output_dir / safe_name(dataset) / safe_name(sample) / f'cell_manual_cluster_annotations__{source_suffix}{safe_name(cluster_key)}.csv.gz'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, compression='gzip')
    return output_path

def build_manual_cluster_annotation_progress(output_dir: str | Path, manual_annotation_path: str | Path, *, write_outputs: bool=True) -> pd.DataFrame:
    output_dir = Path(output_dir).expanduser().resolve()
    cluster_summary = load_output_table(output_dir, 'combined_cluster_summary.csv')
    cluster_columns = ['dataset', 'sample', 'cluster_key', 'cluster']
    required_columns = ['dataset', 'sample', 'clustering_source', 'cluster_key', 'cluster']
    if cluster_summary.empty or not set(cluster_columns).issubset(cluster_summary.columns):
        return pd.DataFrame()
    clusters = cluster_summary[cluster_columns].drop_duplicates().copy()
    clusters.insert(2, 'clustering_source', 'our_expression')
    for column in required_columns:
        clusters[column] = clusters[column].astype(str)
    manual = load_manual_cluster_annotations(manual_annotation_path)
    manual = manual.loc[manual['manual_annotation'].astype(str).str.strip().ne('')].copy()
    for column in required_columns:
        manual[column] = manual[column].astype(str)
    annotated = manual[required_columns + ['confidence']].drop_duplicates(required_columns)
    joined = clusters.merge(annotated.assign(is_annotated=True), on=required_columns, how='left')
    joined['is_annotated'] = joined['is_annotated'].eq(True)
    progress = joined.groupby(['dataset', 'sample', 'clustering_source', 'cluster_key'], dropna=False).agg(n_clusters=('cluster', 'nunique'), n_annotated_clusters=('is_annotated', 'sum')).reset_index()
    progress['n_unannotated_clusters'] = progress['n_clusters'] - progress['n_annotated_clusters']
    progress['fraction_annotated'] = progress['n_annotated_clusters'] / progress['n_clusters'].replace(0, np.nan)
    progress['status'] = np.where(progress['n_clusters'].eq(0), 'no_clusters', np.where(progress['n_unannotated_clusters'].eq(0), 'complete', 'in_progress'))
    progress = progress.sort_values(['status', 'clustering_source', 'dataset', 'sample', 'cluster_key'], key=lambda column: column.map(natural_key)).reset_index(drop=True)
    if write_outputs:
        output_path = output_dir / 'manual_cluster_annotation_progress.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        progress.to_csv(output_path, index=False)
    return progress

def build_cluster_source_annotation_progress(root: str | Path=PROJECT_ROOT, *, output_dir: str | Path | None=None, annotation_dir: str | Path | None=None, gene_panel_review_dir: str | Path | None=None, manual_annotation_path: str | Path | None=None, our_cluster_key: str | None='leiden_r1_3', our_cluster_keys: list[str] | tuple[str, ...] | set[str] | None=None, write_outputs: bool=True) -> pd.DataFrame:
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir or root / 'cell_clustering').expanduser().resolve()
    selected_our_cluster_keys = normalize_cluster_key_selection(our_cluster_keys, fallback_cluster_key=our_cluster_key)
    manual_annotation_path = Path(manual_annotation_path or output_dir / 'manual_cluster_annotations.csv')
    inventory = build_cluster_source_inventory(root, output_dir=output_dir, annotation_dir=annotation_dir, gene_panel_review_dir=gene_panel_review_dir)
    if inventory.empty:
        return pd.DataFrame()
    selected = inventory.loc[inventory['has_cluster_output'].astype(bool)].copy()
    selected = selected.loc[~selected['clustering_source'].astype(str).eq('our_expression') | selected['cluster_key'].astype(str).isin(set(selected_our_cluster_keys))].copy()
    cluster_rows: list[pd.DataFrame] = []
    for row in selected.itertuples(index=False):
        source_row = pd.Series(row._asdict())
        try:
            assignments, _path = load_cluster_source_assignments(source_row, root=root)
            assignments = informative_cluster_assignments(assignments, str(row.cluster_key))
        except Exception:
            continue
        clusters = assignments[[str(row.cluster_key)]].drop_duplicates().copy()
        clusters = clusters.rename(columns={str(row.cluster_key): 'cluster'})
        clusters.insert(0, 'cluster_key', str(row.cluster_key))
        clusters.insert(0, 'clustering_source', str(row.clustering_source))
        clusters.insert(0, 'sample', str(row.sample))
        clusters.insert(0, 'dataset', str(row.dataset))
        cluster_rows.append(clusters)
    if not cluster_rows:
        return pd.DataFrame()
    cluster_table = pd.concat(cluster_rows, ignore_index=True).drop_duplicates()
    key_columns = ['dataset', 'sample', 'clustering_source', 'cluster_key', 'cluster']
    for column in key_columns:
        cluster_table[column] = cluster_table[column].astype(str)
    manual = load_manual_cluster_annotations(manual_annotation_path)
    manual = manual.loc[manual['refined_annotation'].astype(str).str.strip().ne('') & manual['broad_annotation'].astype(str).str.strip().ne('')].copy()
    annotated = manual[key_columns + ['confidence']].drop_duplicates(key_columns)
    joined = cluster_table.merge(annotated.assign(is_annotated=True), on=key_columns, how='left')
    joined['is_annotated'] = joined['is_annotated'].eq(True)
    progress = joined.groupby(['dataset', 'sample', 'clustering_source', 'cluster_key'], dropna=False).agg(n_clusters=('cluster', 'nunique'), n_annotated_clusters=('is_annotated', 'sum')).reset_index()
    progress['n_unannotated_clusters'] = progress['n_clusters'] - progress['n_annotated_clusters']
    progress['fraction_annotated'] = progress['n_annotated_clusters'] / progress['n_clusters'].replace(0, np.nan)
    progress['status'] = np.where(progress['n_unannotated_clusters'].eq(0), 'complete', 'in_progress')
    progress = progress.sort_values(['status', 'clustering_source', 'dataset', 'sample', 'cluster_key'], key=lambda column: column.map(natural_key), kind='stable').reset_index(drop=True)
    if write_outputs:
        output_path = output_dir / 'cluster_source_annotation_progress.csv'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        progress.to_csv(output_path, index=False)
    return progress

def run_cluster_review(root: str | Path=PROJECT_ROOT, *, output_dir: str | Path | None=None, annotation_dir: str | Path | None=None, config: ClusterReviewConfig | None=None) -> dict[str, pd.DataFrame | Path | None]:
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir or root / 'cell_clustering').expanduser().resolve()
    annotation_dir = Path(annotation_dir or root / 'cell_annotation').expanduser().resolve()
    config = config or ClusterReviewConfig()
    overview = write_cluster_count_overview(root, output_dir=output_dir, default_cluster_key=config.default_cluster_key, write_outputs=config.write_outputs)
    umap_figure_index = load_umap_figure_index(output_dir)
    if config.compare_to_marker_score_labels:
        inventory, comparison_summary, combined_pairs = compare_clusters_to_marker_score_labels(root, output_dir=output_dir, annotation_dir=annotation_dir, default_cluster_key=config.default_cluster_key, write_outputs=config.write_outputs)
    else:
        inventory = build_cluster_review_inventory(root, output_dir=output_dir, annotation_dir=annotation_dir)
        comparison_summary = pd.DataFrame()
        combined_pairs = pd.DataFrame()
    return {**overview, 'umap_figure_index': umap_figure_index, 'cluster_review_inventory': inventory, 'cluster_marker_label_comparison_summary': comparison_summary, 'combined_cluster_marker_label_pairs': combined_pairs}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build review tables for cell clustering outputs.')
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--annotation-dir', type=Path)
    parser.add_argument('--default-cluster-key')
    parser.add_argument('--default-resolution', type=float, default=0.8)
    parser.add_argument('--skip-marker-comparison', action='store_true')
    parser.add_argument('--no-write-outputs', action='store_true')
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    default_cluster_key = args.default_cluster_key or cluster_key_for_resolution(args.default_resolution)
    results = run_cluster_review(args.root, output_dir=args.output_dir, annotation_dir=args.annotation_dir, config=ClusterReviewConfig(default_cluster_key=default_cluster_key, write_outputs=not args.no_write_outputs, compare_to_marker_score_labels=not args.skip_marker_comparison))
    counts = results['cluster_count_by_sample_resolution']
    inventory = results['cluster_review_inventory']
    comparison = results['cluster_marker_label_comparison_summary']
    print(f'Cluster count rows: {(len(counts) if isinstance(counts, pd.DataFrame) else 0):,}')
    print(f'Review inventory rows: {(len(inventory) if isinstance(inventory, pd.DataFrame) else 0):,}')
    print(f'Marker-label comparison rows: {(len(comparison) if isinstance(comparison, pd.DataFrame) else 0):,}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
