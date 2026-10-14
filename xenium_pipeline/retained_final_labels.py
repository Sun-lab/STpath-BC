"""Final-label composition in the materialized tiles used for encoder extraction."""

from pathlib import Path

import numpy as np
import pandas as pd

from . import cluster_review, tile_filter_review


BROAD_COLORS = {
    "Tumor Cells": "#C64A50",
    "Normal Epithelial Cells": "#4B8DBD",
    "T Cells": "#459B69",
    "B Cells": "#D5AC30",
    "pan-APCs": "#9368AC",
    "Stromal Cells": "#49A8A0",
    "Smooth Muscle Cells": "#DB8848",
    "Uncertain": "#B4B4B4",
}


def resolve_project_path(value, root):
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        return root / path
    indices = [i for i, part in enumerate(path.parts) if part == root.name]
    return root.joinpath(*path.parts[indices[-1] + 1:]) if indices else path


def retained_membership(coordinates, tiles, tile_size_px):
    """Assign valid H&E centroids to the original nonoverlapping tile grid."""
    if not np.isfinite(tile_size_px) or tile_size_px <= 0:
        raise ValueError("A positive tile size must be available from the tile manifest.")
    if coordinates['cell_id'].isna().any() or coordinates['cell_id'].astype(str).str.strip().eq('').any():
        raise ValueError("Aligned coordinates contain missing cell IDs.")
    if coordinates['cell_id'].duplicated().any():
        raise ValueError("Aligned coordinates contain duplicate cell IDs.")
    if tiles.duplicated(['tile_row', 'tile_col']).any():
        raise ValueError("Retained tiles contain duplicate grid coordinates.")
    x = pd.to_numeric(coordinates['he_x_px'], errors='coerce')
    y = pd.to_numeric(coordinates['he_y_px'], errors='coerce')
    valid = np.isfinite(x) & np.isfinite(y) & x.ge(0) & y.ge(0)
    inside = next((c for c in ('inside_he_image', 'inside_he_candidate') if c in coordinates), None)
    if inside:
        valid &= coordinates[inside].fillna('').astype(str).str.lower().str.strip().isin({'true', '1', 'yes'})
    cells = coordinates.loc[valid, ['cell_id']].copy()
    cells['tile_col'] = np.floor(x[valid] / tile_size_px).astype(int)
    cells['tile_row'] = np.floor(y[valid] / tile_size_px).astype(int)
    tile_columns = ['tile_row', 'tile_col'] + (['tile_id'] if 'tile_id' in tiles else [])
    return cells.merge(tiles[tile_columns], on=['tile_row', 'tile_col'],
                       how='inner', validate='many_to_one')


def match_retained_labels(labels, membership):
    """Match retained cells to final labels while preserving their tile assignment."""
    if labels['cell_id'].str.strip().eq('').any() or labels['cell_id'].duplicated().any():
        raise ValueError("Final labels contain blank or duplicate cell IDs.")
    if membership.empty:
        columns = list(labels.columns) + (['tile_id'] if 'tile_id' in membership else [])
        return pd.DataFrame(columns=columns), {
            'cell_id_match_method': 'no_retained_cells', 'n_retained_cells_with_labels': 0,
            'n_retained_cells_without_labels': 0,
        }
    assignment_column = 'tile_id' if 'tile_id' in membership else '_retained_tile'
    source = membership[['cell_id']].copy()
    source[assignment_column] = (
        membership['tile_id'].astype(str) if assignment_column == 'tile_id' else 'retained'
    )
    matched, diagnostics = cluster_review.match_cluster_source_cell_ids(
        source, cell_id_columns=['cell_id'], cluster_column=assignment_column,
        cluster_key=assignment_column,
        matrix_cell_ids=pd.Index(labels['cell_id']),
    )
    if len(matched) != len(membership):
        raise ValueError("Cell-ID matching changed the number of retained cells.")
    selected = labels.merge(matched, on='cell_id', validate='one_to_one')
    return selected, {
        'cell_id_match_method': diagnostics['cell_id_match_method'],
        'n_retained_cells_with_labels': len(selected),
        'n_retained_cells_without_labels': len(membership) - len(selected),
    }


def count_retained_labels(labels, membership):
    """Match IDs without guessing by row order; retain Uncertain in counts."""
    selected, audit = match_retained_labels(labels, membership)
    broad = selected['final_broad_label'].str.strip().replace('', 'Uncertain')
    unexpected = set(broad) - set(BROAD_COLORS)
    if unexpected:
        raise ValueError(f"Unexpected broad labels: {sorted(unexpected)}")
    return broad.value_counts().reindex(BROAD_COLORS, fill_value=0), audit


def write_cell_annotations(selected, dataset, sample, annotation_dir):
    """Write the complete retained-cell table used for model training."""
    output = selected[['dataset', 'sample', 'cell_id', 'tile_id', 'final_broad_label']].copy()
    output['cell_type'] = output.pop('final_broad_label').str.strip().replace('', 'Uncertain')
    path = Path(annotation_dir) / str(dataset) / str(sample) / 'cell_annotations.csv.gz'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    output.to_csv(temporary, index=False, compression='gzip')
    temporary.replace(path)
    return path


def summarize_retained_final_labels(root, label_dir, selection_manifest, annotation_dir=None, broad_colors=None):
    """Return distribution and per-sample audit; failed samples are never plotted."""
    root, label_dir = Path(root).resolve(), Path(label_dir)
    broad_colors = BROAD_COLORS if broad_colors is None else broad_colors
    selection = pd.read_csv(selection_manifest, dtype={'dataset': str, 'sample': str})
    required = {'dataset', 'sample', 'materialized_tile_manifest', 'passing_tile_ids_path',
                'n_final_selected_tiles'}
    if required - set(selection):
        raise ValueError(f"Selection manifest missing columns: {sorted(required - set(selection))}")
    if selection.empty or selection.duplicated(['dataset', 'sample']).any():
        raise ValueError("Selection manifest must have one row per dataset/sample.")
    sample_lookup = {(s.dataset, s.sample): s for s in tile_filter_review.discover_tile_filter_samples(root)}
    distributions, audits = [], []
    for i, row in enumerate(selection.itertuples(index=False), 1):
        print(f'[{i}/{len(selection)}] Retained-tile labels: {row.dataset} / {row.sample}', flush=True)
        audit = {'dataset': row.dataset, 'sample': row.sample, 'status': 'ok', 'error': ''}
        try:
            expected = float(row.n_final_selected_tiles)
            if not np.isfinite(expected) or expected < 0 or expected != int(expected):
                raise ValueError('n_final_selected_tiles must be a nonnegative integer.')
            manifest_path = resolve_project_path(row.materialized_tile_manifest, root)
            materialized = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
            materialized = materialized.loc[materialized['status'].isin({'copy', 'symlink', 'move'})]
            if len(materialized) != int(expected):
                raise ValueError(f'Expected {int(expected)} materialized tiles, found {len(materialized)}.')
            if not (materialized['dataset'].eq(row.dataset) & materialized['sample'].eq(row.sample)).all():
                raise ValueError('Materialized tiles belong to another dataset/sample.')
            coordinates = materialized['tile_id'].str.extract(r'_row-(\d+)_col-(\d+)$')
            if coordinates.isna().any(axis=None):
                raise ValueError('Could not parse materialized tile IDs.')
            coordinates.columns = ['tile_row', 'tile_col']
            grid = materialized[['tile_id']].reset_index(drop=True).copy()
            grid[['tile_row', 'tile_col']] = coordinates.astype(int).reset_index(drop=True)
            if grid.duplicated(['tile_row', 'tile_col']).any():
                raise ValueError('Duplicate materialized tile coordinates.')
            audit['n_retained_tiles'] = len(grid)
            if grid.empty:
                audit.update(status='no_retained_tiles', n_retained_coordinate_cells=0,
                             n_retained_cells_with_labels=0, n_retained_cells_without_labels=0)
                audits.append(audit)
                continue
            sample = sample_lookup[(row.dataset, row.sample)]
            coordinate_path = next((sample.sample_dir / f'{row.sample}_{suffix}.csv'
                                    for suffix in ('cells_with_annotations', 'cells_with_cluster_labels', 'cells_aligned_to_he')
                                    if (sample.sample_dir / f'{row.sample}_{suffix}.csv').is_file()), None)
            if coordinate_path is None:
                raise FileNotFoundError(f'Missing H&E-aligned cell table under {sample.sample_dir}')
            passing = pd.read_csv(resolve_project_path(row.passing_tile_ids_path, root))
            selected_tiles = grid.merge(passing[['tile_row', 'tile_col', 'n_cells']],
                                        on=['tile_row', 'tile_col'], how='left', validate='one_to_one')
            expected_cells = pd.to_numeric(selected_tiles['n_cells'], errors='raise')
            if expected_cells.isna().any() or (expected_cells < 0).any() or (expected_cells % 1 != 0).any():
                raise ValueError('Retained tiles lack valid cell counts in the passing-tile table.')
            tile_size = tile_filter_review.infer_tile_size_px(sample, default=0)
            coordinates = pd.read_csv(coordinate_path, dtype={'cell_id': str}, keep_default_na=False,
                                      usecols=lambda c: c in {'cell_id', 'he_x_px', 'he_y_px', 'inside_he_image', 'inside_he_candidate'})
            membership = retained_membership(coordinates, grid, tile_size)
            audit.update(tile_size_px=tile_size, n_retained_coordinate_cells=len(membership),
                         n_expected_tile_cells=int(expected_cells.sum()))
            label_path = label_dir / row.dataset / row.sample / 'final_cell_labels.csv.gz'
            labels = pd.read_csv(label_path, dtype=str, keep_default_na=False,
                                 usecols=['dataset', 'sample', 'cell_id', 'final_broad_label'])
            if not (labels['dataset'].eq(row.dataset) & labels['sample'].eq(row.sample)).all():
                raise ValueError('Final-label rows belong to another dataset/sample.')
            audit['n_final_label_cells'] = len(labels)
            selected, match_audit = match_retained_labels(labels, membership)
            audit.update(match_audit)
            if audit['n_expected_tile_cells'] != len(membership):
                raise ValueError('Centroid membership count differs from retained-tile cell counts.')
            if audit['n_retained_cells_without_labels']:
                raise ValueError('Some retained cells have no matching final-label row.')
            broad = selected['final_broad_label'].str.strip().replace('', 'Uncertain')
            unexpected = set(broad) - set(broad_colors)
            if unexpected:
                raise ValueError(f"Unexpected broad labels: {sorted(unexpected)}")
            counts = broad.value_counts().reindex(broad_colors, fill_value=0)
            if annotation_dir is not None:
                annotation_path = write_cell_annotations(
                    selected, row.dataset, row.sample, annotation_dir,
                )
                audit['annotation_path'] = str(annotation_path)
                audit['n_exported_annotation_cells'] = len(membership)
            total = int(counts.sum())
            for broad, count in counts.items():
                distributions.append(dict(dataset=row.dataset, sample=row.sample, broad_cell_type=broad,
                                          n_cells=int(count), n_retained_sample_cells=total,
                                          fraction_of_retained_sample=count / total if total else np.nan))
        except (OSError, ValueError, KeyError) as error:
            audit.update(status='failed', error=str(error))
        audits.append(audit)
    return pd.DataFrame(distributions), pd.DataFrame(audits)


def plot_retained_distribution(distribution, output_dir, broad_colors=None):
    """Save one count/percentage figure per study, using all retained cells."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, StrMethodFormatter
    from textwrap import fill
    from .cell_identity_evidence import natural_key

    output_dir = Path(output_dir)
    broad_colors = BROAD_COLORS if broad_colors is None else broad_colors
    unexpected = set(distribution.broad_cell_type) - set(broad_colors)
    if unexpected:
        raise ValueError(f'Plot classes are absent from the selected mapping: {sorted(unexpected)}')
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for dataset, rows in distribution.groupby('dataset', sort=False):
        counts = rows.pivot(index='sample', columns='broad_cell_type', values='n_cells')
        counts = counts.reindex(index=sorted(counts.index, key=natural_key), columns=list(broad_colors))
        counts = counts.loc[counts.sum(axis=1).gt(0)]
        if counts.empty:
            continue
        fractions = counts.div(counts.sum(axis=1), axis=0) * 100
        y = np.arange(len(counts))
        figure, axes = plt.subplots(1, 2, sharey=True, figsize=(16, max(4.2, .62 * len(counts) + 2.4)))
        for axis, values, title in zip(axes, (counts, fractions), ('Cell counts', 'Retained-cell composition')):
            left = np.zeros(len(counts))
            for broad, color in broad_colors.items():
                axis.barh(y, values[broad], left=left, color=color, height=.7, label=broad)
                left += values[broad].to_numpy()
            axis.set_title(title)
            axis.set_axisbelow(True)
            axis.grid(axis='x', alpha=.2)
            axis.spines[['top', 'right']].set_visible(False)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels([fill(s, 46) for s in counts.index], fontsize=9)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Number of cells in retained tiles')
        axes[0].xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        axes[1].set_xlim(0, 100)
        axes[1].set_xlabel('Percentage of retained sample cells')
        axes[1].xaxis.set_major_formatter(PercentFormatter(xmax=100))
        figure.suptitle(f'{dataset}: final broad cell types in retained tiles', fontsize=13)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=9)
        figure.tight_layout(rect=(0, .10, 1, .95))
        path = output_dir / f'{dataset}_retained_final_broad_distribution.png'
        figure.savefig(path, dpi=180, bbox_inches='tight')
        plt.close(figure)
        paths.append(path)
    return paths
