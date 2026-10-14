"""Breast Xenium S9 exports from the annotations and tiles used for training."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .epithelial_morphology import match_rows
from .final_cell_labels import PPTX_BROAD_COLORS
BROAD_ORDER = list(PPTX_BROAD_COLORS)
TARGET_COLORS = {'Tumor Cells': '#C64A50', 'Normal Epithelial Cells': '#4B8DBD', 'T Cells': '#459B69', 'pan-APC + B': '#9368AC', 'Stromal Cells': '#49A8A0', 'Other/uncertain': '#B4B4B4'}
HE_GRID_STYLE = dict(color='#404040', linewidth=0.6, alpha=0.65)
MARKER_GENES = ['EPCAM', 'KRT8', 'KRT5', 'PIGR', 'ERBB2', 'ESR1', 'MKI67', 'CD3E', 'TRAC', 'CD8A', 'NKG7', 'MS4A1', 'CD79A', 'LYZ', 'CD68', 'FCER1A', 'TPSAB1', 'LUM', 'PDGFRA', 'PECAM1', 'VWF', 'ADIPOQ', 'ACTA2', 'MYH11']

def training_group(broad):
    """Five reported targets plus the denominator-only remainder."""
    broad = broad.fillna('').astype(str).str.strip().replace('', 'Uncertain')
    unknown = set(broad) - set(BROAD_ORDER) - {'B Cells', 'Smooth Muscle Cells', 'NK Cells', 'Unannotated', 'Unassigned'}
    if unknown:
        raise ValueError(f'Unmapped broad labels: {sorted(unknown)}')
    return broad.replace({'B Cells': 'pan-APC + B', 'pan-APCs': 'pan-APC + B', 'NK Cells': 'T Cells', 'Smooth Muscle Cells': 'Other/uncertain', 'Uncertain': 'Other/uncertain', 'Unannotated': 'Other/uncertain', 'Unassigned': 'Other/uncertain'})

def load_training_tiles(reference_path, cohort_path):
    """Use the saved training snapshot, with exact Table S2 cohort/count checks."""
    tiles = pd.read_csv(reference_path, dtype={'dataset': str, 'sample': str, 'tile_id': str}, usecols=lambda c: c in {'dataset', 'sample', 'tile_id', 'n_cells_total'} or c.startswith('count__'))
    required = {'dataset', 'sample', 'tile_id', 'n_cells_total'}
    if required - set(tiles):
        raise ValueError(f'Training snapshot lacks {sorted(required - set(tiles))}')
    if tiles.empty or tiles.tile_id.isna().any() or tiles.tile_id.duplicated().any():
        raise ValueError('Training snapshot must contain unique nonempty tile IDs.')
    if tiles.tile_id.str.strip().eq('').any():
        raise ValueError('Training snapshot has blank tile IDs.')
    counts = pd.to_numeric(tiles.n_cells_total, errors='raise')
    if not np.isfinite(counts).all() or (counts <= 0).any() or (counts % 1 != 0).any():
        raise ValueError('Training denominators must be positive integer cell counts.')
    cohort = pd.read_csv(cohort_path, dtype={'dataset': str, 'sample': str})
    if cohort.duplicated(['dataset', 'sample']).any():
        raise ValueError('Table S2 has duplicate samples.')
    observed = tiles.groupby(['dataset', 'sample']).size().rename('observed_tiles')
    expected = cohort.set_index(['dataset', 'sample']).retained_tiles
    check = pd.concat([expected, observed], axis=1)
    if check.isna().any(axis=None) or not check.retained_tiles.eq(check.observed_tiles).all():
        raise ValueError(f'Training snapshot does not match Table S2:\n{check}')
    return (tiles, cohort)

def load_training_references(reference_paths, cohort_path):
    """Combine saved target snapshots only when tile identities/denominators agree."""
    combined = None
    for path in reference_paths:
        table, cohort = load_training_tiles(path, cohort_path)
        if not any((c.startswith('count__') for c in table)):
            raise ValueError(f'Saved target snapshot has no count columns: {path}')
        table = table.set_index('tile_id').sort_index()
        if combined is None:
            combined = table
            continue
        if not table.index.equals(combined.index):
            raise ValueError(f'Target snapshot tile IDs differ: {path}')
        for column in table:
            if column in combined:
                if not table[column].equals(combined[column]):
                    raise ValueError(f'Saved snapshots disagree on {column}: {path}')
            else:
                combined[column] = table[column]
    if combined is None:
        raise ValueError('Provide saved training target snapshots.')
    return (combined.reset_index(), cohort)


def load_training_cells(sources, target_sources, tiles, dataset, sample, *, display_source):
    """Validate each model's own source; keep the figure's annotation source explicit."""
    if display_source not in sources or display_source not in set(target_sources.values()):
        raise ValueError('The figure needs an explicit annotation source with saved model targets.')
    missing = set(target_sources.values()) - set(sources)
    if missing:
        raise ValueError(f'Missing configured model annotation sources: {sorted(missing)}')
    sample_tiles = tiles.loc[tiles.dataset.eq(dataset) & tiles['sample'].eq(sample)]
    required = {f'count__{target}' for target in target_sources}
    if required - set(tiles):
        raise ValueError(f'Missing saved target count columns: {sorted(required - set(tiles))}')
    loaded, audits = ({}, {})
    identity_columns = ['dataset', 'sample', 'tile_id', 'n_cells_total']
    for source_name in dict.fromkeys(target_sources.values()):
        targets = [target for target, source in target_sources.items() if source == source_name]
        count_columns = [f'count__{target}' for target in targets]
        if 'pan-APC + B' in targets:
            count_columns += [c for c in ('count__B Cells', 'count__pan-APCs') if c in tiles]
        label_dir, annotation_dir = sources[source_name]
        try:
            loaded[source_name], audits[source_name] = load_sample_cells(label_dir, annotation_dir, sample_tiles[identity_columns + count_columns], dataset, sample)
        except (ValueError, FileNotFoundError) as error:
            raise ValueError(f"{source_name} annotations failed validation for their own model targets ({', '.join(targets)}) at {label_dir}: {error}") from error
        audits[source_name].update(label_dir=str(label_dir), annotation_dir=str(annotation_dir))
    display = loaded[display_source]
    for source_name, cells in loaded.items():
        matched, _ = match_rows(cells, display.cell_id)
        check = display[['cell_id', 'tile_id']].merge(matched[['cell_id', 'tile_id']], on='cell_id', how='left', validate='one_to_one', suffixes=('_display', '_model'))
        if len(cells) != len(display) or not check.tile_id_display.eq(check.tile_id_model).all():
            raise ValueError(f'Retained cell IDs/tile membership differ between {display_source} and {source_name}.')
    expected = sample_tiles.set_index('tile_id')
    displayed_counts = pd.crosstab(display.tile_id, display.cell_type).reindex(expected.index, fill_value=0)
    rows = []
    for target, source_name in target_sources.items():
        components = ['B Cells', 'pan-APCs'] if target == 'pan-APC + B' else [target]
        actual = sum((displayed_counts.get(c, pd.Series(0, index=expected.index)) for c in components))
        difference = actual - expected[f'count__{target}']
        rows.append(dict(dataset=dataset, sample=sample, model_target=target, status='ok', model_annotation_source=source_name, model_label_dir=audits[source_name]['label_dir'], model_annotation_path=audits[source_name]['training_annotation_path'], display_annotation_source=display_source, display_label_dir=audits[display_source]['label_dir'], display_matches_saved_target=bool(difference.eq(0).all()), n_display_target_mismatch_tiles=int(difference.ne(0).sum()), sum_absolute_display_count_difference=int(difference.abs().sum()), validation_scope='Annotation source checked against model targets; display differences recorded'))
    audit = dict(audits[display_source], display_annotation_source=display_source, validation_scope='Cell IDs/denominators; each model validated against its own annotation source', n_models_validated=len(rows), n_models_differing_from_display=sum((not r['display_matches_saved_target'] for r in rows)))
    return (display, audit, pd.DataFrame(rows))

def validate_training_cells(cells, labels, tiles):
    """Require final labels to reproduce the actual training annotation source."""
    for table in (cells, labels):
        if table.cell_id.isna().any() or table.cell_id.astype(str).str.strip().eq('').any() or table.cell_id.duplicated().any():
            raise ValueError('Blank or duplicate cell IDs in annotations/final labels.')
    if cells.cell_type.fillna('').astype(str).str.strip().eq('').any():
        raise ValueError('Training cells need explicit biological or uncertain labels.')
    if set(labels.final_broad_label) - set(PPTX_BROAD_COLORS):
        raise ValueError('Final labels are not the eight source classes in xgboost_LOIO_results.pptx.')
    if not cells.tile_id.isin(tiles.tile_id).all():
        raise ValueError('Training annotations contain cells outside the saved retained tiles.')
    expected = tiles.set_index('tile_id').n_cells_total.astype(int)
    observed = cells.groupby('tile_id').size().reindex(expected.index, fill_value=0)
    if not observed.eq(expected).all():
        raise ValueError('Per-tile cell counts differ from the saved training denominators.')
    aligned, method = match_rows(labels, cells.cell_id)
    if len(aligned) != len(cells):
        raise ValueError('Some training cells have no matching final annotation.')
    result = cells.merge(aligned[['cell_id', 'final_refined_label', 'final_broad_label']], on='cell_id', validate='one_to_one')
    if not result.cell_type.astype(str).str.strip().eq(result.final_broad_label.astype(str).str.strip()).all():
        raise ValueError('Final labels differ from the training annotations. Provide the matching annotation source.')
    result['plot_group'] = training_group(result.cell_type)
    verify_target_counts(result, tiles)
    return (result, method)

def verify_target_counts(cells, tiles):
    """Check every available saved count target, including the original denominator."""
    expected = tiles.set_index('tile_id').n_cells_total.astype(int)
    if set(cells.tile_id) != set(expected.index):
        raise ValueError('Annotation and saved target tile sets differ.')
    observed = cells.groupby('tile_id').size().reindex(expected.index, fill_value=0)
    if not observed.eq(expected).all():
        raise ValueError('Per-tile cell counts differ from the saved training denominators.')
    broad_counts = pd.crosstab(cells.tile_id, cells.cell_type).reindex(expected.index, fill_value=0)
    for column in tiles:
        if not column.startswith('count__'):
            continue
        label = column.removeprefix('count__')
        components = ['pan-APCs', 'B Cells'] if label == 'pan-APC + B' else [label]
        actual = sum((broad_counts.get(c, pd.Series(0, index=expected.index)) for c in components))
        recorded = pd.to_numeric(tiles.set_index('tile_id')[column], errors='raise')
        if not actual.eq(recorded).all():
            raise ValueError(f'Annotation counts differ from the saved training target {label}.')

def load_sample_cells(label_dir, annotation_dir, tiles, dataset, sample):
    sample_tiles = tiles.loc[tiles.dataset.eq(dataset) & tiles['sample'].eq(sample)].copy()
    if sample_tiles.empty:
        raise ValueError(f'{dataset}/{sample} is outside the manuscript cohort.')
    label_path = Path(label_dir) / dataset / sample / 'final_cell_labels.csv.gz'
    annotation_path = Path(annotation_dir) / dataset / sample / 'cell_annotations.csv.gz'
    labels = pd.read_csv(label_path, dtype=str, keep_default_na=False, usecols=['dataset', 'sample', 'cell_id', 'final_refined_label', 'final_broad_label'])
    cells = pd.read_csv(annotation_path, dtype=str, keep_default_na=False, usecols=['dataset', 'sample', 'cell_id', 'tile_id', 'cell_type'])
    for table in (cells, labels):
        if not (table.dataset.eq(dataset) & table['sample'].eq(sample)).all():
            raise ValueError(f'Annotation identity mismatch for {dataset}/{sample}.')
    selected, method = validate_training_cells(cells, labels, sample_tiles)
    return (selected, dict(dataset=dataset, sample=sample, n_cells=len(selected), n_tiles=len(sample_tiles), cell_id_match_method=method, final_label_path=str(label_path), training_annotation_path=str(annotation_path)))

def load_annotation_figure_inputs(root, selection_manifest, cohort_path, audit_path):
    """Check the retained grid and successful Annotation audit, without model targets."""
    from .stpath_xgboost import load_retained_tiles
    tiles = load_retained_tiles(root, selection_manifest)[['dataset', 'sample', 'tile_id']]
    cohort = pd.read_csv(cohort_path, dtype={'dataset': str, 'sample': str})
    audit = pd.read_csv(audit_path, dtype={'dataset': str, 'sample': str})
    if cohort.duplicated(['dataset', 'sample']).any() or audit.duplicated(['dataset', 'sample']).any():
        raise ValueError('Duplicate samples in Table S2 or the annotation audit.')
    expected = cohort.set_index(['dataset', 'sample']).retained_tiles
    observed = tiles.groupby(['dataset', 'sample']).size().rename('observed_tiles')
    check = pd.concat([expected, observed], axis=1)
    if check.isna().any(axis=None) or not check.retained_tiles.eq(check.observed_tiles).all():
        raise ValueError('Retained tile selection does not match Table S2.')
    if audit.status.eq('failed').any():
        raise ValueError('The annotation audit has failed samples; resolve its membership audit first.')
    audit = audit.set_index(['dataset', 'sample']).reindex(expected.index)
    if not audit.status.eq('ok').all() or not audit.n_retained_tiles.eq(expected).all():
        raise ValueError('Annotation audit does not cover the Table S2 retained tiles.')
    if not audit.n_retained_cells_without_labels.eq(0).all():
        raise ValueError('Annotation audit contains unmatched retained cells.')
    return (tiles, cohort, audit.reset_index())

def load_annotation_sample_cells(label_dir, annotation_dir, tiles, membership_audit, dataset, sample):
    """Validate current exported labels; compute denominators from those same cells."""
    path = Path(annotation_dir) / dataset / sample / 'cell_annotations.csv.gz'
    cells = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=['tile_id'])
    sample_tiles = tiles.loc[tiles.dataset.eq(dataset) & tiles['sample'].eq(sample)].copy()
    counts = cells.groupby('tile_id').size()
    if set(counts.index) != set(sample_tiles.tile_id):
        raise ValueError('Audited annotation and retained grid tile sets differ.')
    row = membership_audit.loc[membership_audit.dataset.eq(dataset) & membership_audit['sample'].eq(sample)]
    if len(row) != 1:
        raise ValueError('Need one Annotation audit row for this sample.')
    for column in ('n_expected_tile_cells', 'n_retained_coordinate_cells', 'n_retained_cells_with_labels', 'n_exported_annotation_cells'):
        if row.iloc[0][column] != len(cells):
            raise ValueError(f'Audited annotation cell count differs from audit field {column}.')
    sample_tiles['n_cells_total'] = sample_tiles.tile_id.map(counts)
    return load_sample_cells(label_dir, annotation_dir, sample_tiles, dataset, sample)

def composition_table(cells):
    counts = cells.cell_type.value_counts().reindex(PPTX_BROAD_COLORS, fill_value=0)
    total = len(cells)
    if not total:
        raise ValueError('A sample has no retained cells.')
    return pd.DataFrame({'target': counts.index, 'n_cells': counts.values, 'n_retained_sample_cells': total, 'fraction': counts.values / total})

def tile_proportions(cells):
    counts = pd.crosstab(cells.tile_id, cells.plot_group).reindex(columns=TARGET_COLORS, fill_value=0)
    fractions = counts.div(counts.sum(axis=1), axis=0)
    fractions['n_cells_total'] = counts.sum(axis=1)
    return fractions.reset_index()

def marker_statistics(adata, cells, genes=MARKER_GENES):
    """Exact retained-cell matching; all assayed genes set the library-size denominator."""
    from scipy import sparse
    source = pd.DataFrame({'cell_id': adata.obs_names.astype(str), '_matrix_row': np.arange(adata.n_obs)})
    matched, method = match_rows(source, cells.cell_id)
    if len(matched) != len(cells):
        raise ValueError('Expression matrix does not cover every retained training cell.')
    matched = cells[['cell_id', 'final_broad_label']].merge(matched, on='cell_id', validate='one_to_one')
    matrix = sparse.csr_matrix(adata.X[matched._matrix_row.to_numpy(int)], dtype=float)
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any() or (not np.allclose(matrix.data, np.round(matrix.data))):
        raise ValueError('Marker QC requires raw nonnegative integer counts, not a normalized checkpoint.')
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    normal = matrix.multiply(np.divide(10000, totals, out=np.zeros_like(totals), where=totals > 0)[:, None]).tocsr()
    normal.data = np.log1p(normal.data)
    symbols = pd.Index(adata.var['gene_symbol'].astype(str) if 'gene_symbol' in adata.var else adata.var_names.astype(str))
    rows = []
    for gene in genes:
        indices = np.flatnonzero(symbols == gene)
        if len(indices) > 1:
            raise ValueError(f'Duplicate matrix gene symbol: {gene}')
        for broad in BROAD_ORDER:
            mask = matched.final_broad_label.eq(broad).to_numpy()
            n = int(mask.sum())
            if not n:
                continue
            available = len(indices) == 1
            index = int(indices[0]) if available else None
            rows.append(dict(broad_cell_type=broad, gene=gene, in_assay=available, n_cells=n, mean_log1p_10k=float(normal[mask, index].mean()) if available else np.nan, fraction_expressing=float((matrix[mask, index] > 0).mean()) if available else np.nan, cell_id_match_method=method))
    result = pd.DataFrame(rows)
    if result.empty or not result.in_assay.any():
        raise ValueError('No selected marker genes are assayed in this sample.')
    extrema = result.groupby('gene').mean_log1p_10k.agg(['min', 'max'])
    result['scaled_mean'] = result.apply(lambda r: (r.mean_log1p_10k - extrema.loc[r.gene, 'min']) / (extrema.loc[r.gene, 'max'] - extrema.loc[r.gene, 'min']) if r.in_assay and extrema.loc[r.gene, 'max'] > extrema.loc[r.gene, 'min'] else 0.0 if r.in_assay else np.nan, axis=1)
    return result

def save_figure(figure, output_dir, stem):
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if stem in ('Figure_S9A', 'Figure_S9B'):
        from matplotlib.text import Text
        for text in figure.findobj(Text):
            text.set_fontweight('bold')
    paths = []
    for suffix in ('png', 'pdf'):
        path = output_dir / f'{stem}.{suffix}'
        figure.savefig(path, dpi=300, bbox_inches='tight')
        paths.append(path)
    plt.close(figure)
    return paths

def plot_marker_evidence(statistics, output_dir, genes=MARKER_GENES):
    import matplotlib.pyplot as plt
    groups = list(statistics.groupby(['dataset', 'sample'], sort=False))
    fig, axes = plt.subplots(len(groups), 1, figsize=(13, 2.9 * len(groups) + 1), squeeze=False)
    for ax, ((dataset, sample), rows) in zip(axes.flat, groups):
        labels = [b for b in BROAD_ORDER if b in set(rows.broad_cell_type)]
        x = rows.gene.map({g: i for i, g in enumerate(genes)})
        y = rows.broad_cell_type.map({b: i for i, b in enumerate(labels)})
        present = rows.in_assay.astype(bool)
        points = ax.scatter(x[present], y[present], s=180 * rows.loc[present, 'fraction_expressing'], c=rows.loc[present, 'scaled_mean'], cmap='Blues', vmin=0, vmax=1)
        ax.scatter(x[~present], y[~present], s=14, marker='x', color='#AAAAAA')
        ax.set(xticks=range(len(genes)), yticks=range(len(labels)), yticklabels=labels, xlim=(-0.6, len(genes) - 0.4), ylim=(len(labels) - 0.4, -0.6), title=f'{dataset} / {sample}')
        ax.set_xticklabels(genes, rotation=60, ha='right', fontsize=8)
        ax.tick_params(axis='y', labelsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        fig.colorbar(points, ax=ax, fraction=0.02, pad=0.01, label='Scaled mean expression')
    for fraction in (0.1, 0.5, 1):
        axes[-1, 0].scatter([], [], s=180 * fraction, color='#397BAC', label=f'{fraction:.0%}')
    axes[-1, 0].legend(title='Cells expressing', loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=3, frameon=False)
    fig.suptitle('Marker evidence for retained-cell annotations', x=0.03, ha='left', fontweight='bold')
    fig.text(0.03, 0.005, 'Cross: gene not assayed. Expression: log1p(counts per 10,000), scaled within each gene and sample.', fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97), h_pad=2)
    return save_figure(fig, output_dir, 'Annotation_marker_QC')

def figure_label_mapping(smooth_muscle_as_uncertain=False):
    """Pool B cells with pan-APCs for display, retaining source annotations."""
    return dict({'B Cells': 'pan-APCs'}, **{'Smooth Muscle Cells': 'Uncertain'} if smooth_muscle_as_uncertain else {})

def figure_colors(smooth_muscle_as_uncertain=False):
    return {k: v for k, v in PPTX_BROAD_COLORS.items() if k not in figure_label_mapping(smooth_muscle_as_uncertain)}

def figure_composition(composition, *, smooth_muscle_as_uncertain=False):
    """Group counts for display only; preserve the audited biological composition."""
    result = composition.copy()
    result['target'] = result.target.replace(figure_label_mapping(smooth_muscle_as_uncertain))
    return result.groupby(['dataset', 'sample', 'target'], sort=False, as_index=False).agg(n_cells=('n_cells', 'sum'), n_retained_sample_cells=('n_retained_sample_cells', 'first'), fraction=('fraction', 'sum'))

def plot_composition(composition, cohort, output_dir, *, smooth_muscle_as_uncertain=False):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    order = pd.MultiIndex.from_frame(cohort[['dataset', 'sample']])
    colors = figure_colors(smooth_muscle_as_uncertain)
    composition = figure_composition(composition, smooth_muscle_as_uncertain=smooth_muscle_as_uncertain)
    wide = composition.pivot(index=['dataset', 'sample'], columns='target', values='fraction').reindex(index=order, columns=colors)
    if wide.isna().any(axis=None) or not np.allclose(wide.sum(axis=1), 1):
        raise ValueError('Panel A has missing sample/categories or fractions do not sum to one.')
    fig, ax = plt.subplots(figsize=(16, max(8, 0.36 * len(wide) + 2)))
    left = np.zeros(len(wide))
    for target, color in colors.items():
        ax.barh(np.arange(len(wide)), wide[target], left=left, color=color, height=0.8, label='pan-APC' if target == 'pan-APCs' else target)
        left += wide[target].to_numpy()
    ax.set(yticks=range(len(wide)), xlim=(0, 1), xlabel='Fraction of all cells assigned to retained training tiles')
    ax.set_yticklabels([f'{d} / {s}' for d, s in wide.index], fontsize=11, fontweight='bold')
    for i in range(1, len(order)):
        if order[i][0] != order[i - 1][0]:
            ax.axhline(i - 0.5, color='white', linewidth=3)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('Breast Xenium retained-sample annotation composition', loc='left', fontweight='bold', fontsize=16)
    if smooth_muscle_as_uncertain:
        ax.set_title('Breast Xenium retained-sample annotation composition\nSmooth muscle grouped with Uncertain for display', loc='left', fontweight='bold', fontsize=15)
    fig.legend(*ax.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.5, 0.015), ncol=len(colors), frameon=False, prop={'size': 12, 'weight': 'bold'})
    ax.xaxis.label.set_size(14)
    ax.tick_params(axis='x', labelsize=12)
    from matplotlib.text import Text
    for text in fig.findobj(Text):
        text.set_fontweight('bold')
    fig.tight_layout(rect=(0, 0.7 / fig.get_figheight(), 1, 1))
    return save_figure(fig, output_dir, 'Figure_S9A')

def attach_coordinates(cells, coordinates, tile_size_px):
    """Keep the saved H&E coordinate frame and verify the training tile assignment."""
    matched, method = match_rows(coordinates, cells.cell_id)
    if len(matched) != len(cells):
        raise ValueError('Some training cells lack saved H&E coordinates.')
    result = cells.merge(matched[['cell_id', 'he_x_px', 'he_y_px']], on='cell_id', validate='one_to_one')
    xy = result[['he_x_px', 'he_y_px']].apply(pd.to_numeric, errors='raise')
    if tile_size_px <= 0 or not np.isfinite(xy).all(axis=None) or (xy < 0).any(axis=None):
        raise ValueError('Invalid tile size or H&E coordinates.')
    grid = result.tile_id.str.extract('_row-(\\d+)_col-(\\d+)$').astype(int)
    if not (np.floor(xy.he_y_px / tile_size_px).astype(int).eq(grid[0]) & np.floor(xy.he_x_px / tile_size_px).astype(int).eq(grid[1])).all():
        raise ValueError('Saved coordinates disagree with training tile assignments; check image frame/version.')
    result[['he_x_px', 'he_y_px']] = xy
    return (result, method)

def spatial_roi(cells, tile_size_px, roi=None, side_tiles=4):
    if roi is None:
        grid = cells.tile_id.drop_duplicates().str.extract('_row-(\\d+)_col-(\\d+)$').astype(int)
        center = grid.iloc[np.argmin(np.sum((grid - grid.median()) ** 2, axis=1))]
        y0, x0 = np.maximum(0, center.to_numpy() - side_tiles // 2) * tile_size_px
        roi = (int(x0), int(y0), int(x0 + side_tiles * tile_size_px), int(y0 + side_tiles * tile_size_px))
    if len(roi) != 4 or not np.isfinite(roi).all() or any((int(v) != v for v in roi)):
        raise ValueError('ROI must be four integer H&E pixel coordinates: x0,y0,x1,y1.')
    x0, y0, x1, y1 = map(int, roi)
    if min(x0, y0) < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError('ROI must be positive and ordered.')
    return (x0, y0, x1, y1)

def select_tumor_samples(composition, cohort, n_samples=3):
    """Prioritize higher pooled tumor fractions, with distinct Table S2 split groups."""
    tumor = composition.loc[composition.target.eq('Tumor Cells'), ['dataset', 'sample', 'fraction']]
    ranked = cohort.merge(tumor, on=['dataset', 'sample'], how='left', validate='one_to_one')
    if ranked.fraction.isna().any() or not ranked.fraction.between(0, 1).all():
        raise ValueError('Panel A must cover the cohort with valid tumor fractions.')
    if ranked.evaluation_group.isna().any():
        raise ValueError('Table S2 requires evaluation groups for example selection.')
    return ranked.sort_values(['fraction', 'dataset', 'sample'], ascending=[False, True, True]).drop_duplicates('evaluation_group').head(n_samples).rename(columns={'fraction': 'sample_tumor_fraction'})

def region_compositions(cells, tile_size_px, side_tiles=10, image_size_px=None):
    """Cell-weighted compositions of tile-aligned windows, including uncertain cells."""
    if side_tiles < 1 or int(side_tiles) != side_tiles or tile_size_px < 1 or (int(tile_size_px) != tile_size_px):
        raise ValueError('Window and tile sizes must be positive integers.')
    side_tiles, tile_size_px = (int(side_tiles), int(tile_size_px))
    if cells.empty or set(cells.plot_group) - set(TARGET_COLORS):
        raise ValueError('Need retained cells with valid plotting groups.')
    counts = pd.crosstab(cells.tile_id, cells.plot_group).reindex(columns=TARGET_COLORS, fill_value=0)
    grid = counts.index.to_series().str.extract('_row-(\\d+)_col-(\\d+)$')
    if grid.isna().any(axis=None):
        raise ValueError('Could not parse retained tile grid IDs.')
    grid = grid.astype(int)
    index = pd.MultiIndex.from_arrays([grid[0], grid[1]])
    if index.duplicated().any():
        raise ValueError('Select regions from one sample with unique grid coordinates.')
    counts.index = index
    totals = np.zeros((len(counts), len(TARGET_COLORS)), dtype=np.int64)
    coverage = np.zeros(len(counts), dtype=np.int64)
    for dy in range(side_tiles):
        for dx in range(side_tiles):
            neighbors = pd.MultiIndex.from_arrays([grid[0] + dy, grid[1] + dx])
            values = counts.reindex(neighbors)
            totals += values.fillna(0).to_numpy(dtype=np.int64)
            coverage += values.iloc[:, 0].notna().to_numpy(dtype=np.int64)
    result = pd.DataFrame({'x0': grid[1].to_numpy() * tile_size_px, 'y0': grid[0].to_numpy() * tile_size_px, 'n_cells': totals.sum(axis=1), 'n_retained_tiles': coverage})
    result['x1'] = result.x0 + side_tiles * tile_size_px
    result['y1'] = result.y0 + side_tiles * tile_size_px
    result['retained_tile_fraction'] = coverage / side_tiles ** 2
    for i, target in enumerate(TARGET_COLORS):
        result[f'count__{target}'] = totals[:, i]
        result[f'fraction__{target}'] = totals[:, i] / result.n_cells
    if image_size_px is not None:
        width, height = image_size_px
        result = result.loc[result.x1.le(width) & result.y1.le(height)]
    return result

def mixed_region_candidates(cells, tile_size_px, *, side_tiles=10, min_retained_fraction=0.75, min_cells=100, featured_range=(0.25, 0.6), other_min_fraction=0.1, min_other_types=2, preferred_other_fraction=0.05, image_size_px=None):
    """Best qualifying window per featured type, plus a threshold audit for this sample."""
    targets = list(TARGET_COLORS)[:-1]
    if not (0 < min_retained_fraction <= 1 and min_cells >= 1 and (0 <= featured_range[0] <= featured_range[1] <= 1) and (0 <= other_min_fraction <= 1) and (0 <= preferred_other_fraction <= 1) and (1 <= min_other_types <= len(targets) - 1)):
        raise ValueError('Invalid mixed-composition region thresholds.')
    windows = region_compositions(cells, tile_size_px, side_tiles, image_size_px)
    covered = windows.loc[windows.n_cells.ge(min_cells) & windows.retained_tile_fraction.ge(min_retained_fraction)]
    best, audits = ([], [])
    for target in targets:
        featured = covered[f'fraction__{target}']
        others = covered[[f'fraction__{other}' for other in targets if other != target]]
        featured_ok = featured.between(*featured_range)
        mixed_ok = others.ge(other_min_fraction).sum(axis=1).ge(min_other_types)
        eligible = covered.loc[featured_ok & mixed_ok].copy()
        audits.append(dict(featured_target=target, n_windows=len(windows), n_coverage_cell_pass=len(covered), n_featured_range_pass=int(featured_ok.sum()), n_mixed_pass=len(eligible)))
        if eligible.empty:
            continue
        eligible['featured_target'] = target
        eligible['featured_fraction'] = featured.loc[eligible.index]
        eligible['min_other_fraction'] = others.min(axis=1).loc[eligible.index]
        eligible['all_other_types_ge_preferred'] = eligible.min_other_fraction.ge(preferred_other_fraction)
        eligible['quality_score'] = 10 * eligible.all_other_types_ge_preferred.astype(int) + eligible.featured_fraction + eligible.min_other_fraction
        eligible = eligible.sort_values(['quality_score', 'n_cells', 'y0', 'x0'], ascending=[False, False, True, True])
        best.append(eligible.iloc[[0]])
    columns = list(windows) + ['featured_target', 'featured_fraction', 'min_other_fraction', 'all_other_types_ge_preferred', 'quality_score']
    return (pd.concat(best, ignore_index=True) if best else pd.DataFrame(columns=columns), pd.DataFrame(audits))

def revalidate_saved_regions(cells, saved_regions, tile_size_px, rules):
    """Recompute only provided crop locations; discard all old composition metrics."""
    grid = cells.tile_id.str.extract('_row-(\\d+)_col-(\\d+)$')
    if grid.isna().any(axis=None):
        raise ValueError('Cannot parse retained cell tile IDs.')
    grid = grid.astype(int)
    x, y = (grid[1] * tile_size_px, grid[0] * tile_size_px)
    eligible, audit = ([], [])
    for example in saved_regions.itertuples(index=False):
        roi = spatial_roi(cells, tile_size_px, (example.x0, example.y0, example.x1, example.y1))
        if (roi[2] - roi[0], roi[3] - roi[1]) != (rules['side_tiles'] * tile_size_px,) * 2:
            raise ValueError('Saved crop dimensions differ from the requested tile window.')
        if any((v % tile_size_px for v in roi)):
            raise ValueError('Saved crop is not aligned to the training tile grid.')
        target = 'pan-APC + B' if example.featured_target == 'pan-APCs' else example.featured_target
        if target not in list(TARGET_COLORS)[:-1]:
            raise ValueError(f'Unknown featured target: {target}')
        region = cells.loc[x.ge(roi[0]) & x.lt(roi[2]) & y.ge(roi[1]) & y.lt(roi[3])]
        counts = region.plot_group.value_counts().reindex(TARGET_COLORS, fill_value=0)
        fractions = counts / len(region) if len(region) else counts.astype(float)
        others = fractions.drop([target, 'Other/uncertain'])
        coverage = region.tile_id.nunique() / rules['side_tiles'] ** 2
        checks = dict(cell_count_pass=len(region) >= rules['min_cells'], coverage_pass=coverage >= rules['min_retained_fraction'], featured_range_pass=rules['featured_range'][0] <= fractions[target] <= rules['featured_range'][1], other_types_pass=others.ge(rules['other_min_fraction']).sum() >= rules['min_other_types'])
        ok = all(checks.values())
        row = dict(dataset=example.dataset, sample=example.sample, featured_target=target, x0=roi[0], y0=roi[1], x1=roi[2], y1=roi[3], n_cells=len(region), n_retained_tiles=region.tile_id.nunique(), retained_tile_fraction=coverage, featured_fraction=float(fractions[target]), min_other_fraction=float(others.min()), all_other_types_ge_preferred=bool(others.ge(rules['preferred_other_fraction']).all()))
        row['quality_score'] = 10 * row['all_other_types_ge_preferred'] + row['featured_fraction'] + row['min_other_fraction']
        row.update({f'count__{t}': int(counts[t]) for t in TARGET_COLORS})
        row.update({f'fraction__{t}': float(fractions[t]) for t in TARGET_COLORS})
        audit.append(dict(row, **checks, n_other_types_ge_min=int(others.ge(rules['other_min_fraction']).sum()), failed_rules=', '.join((name.removesuffix('_pass') for name, passed in checks.items() if not passed)), qualifies=bool(ok), validation='Saved coordinates; recomputed updated-label counts'))
        if ok:
            eligible.append(row)
    return (pd.DataFrame(eligible, columns=list(row) if audit else []), pd.DataFrame(audit))

def select_revalidated_saved_examples(saved, candidates, failure_audit=None):
    """Keep qualifying original crops; replace failures only within their dataset/target pair."""
    if saved.empty or saved.dataset.duplicated().any() or saved.featured_target.duplicated().any():
        raise ValueError('Saved examples need distinct datasets and featured targets.')
    chosen, audit = ([], [])
    for old in saved.itertuples(index=False):
        pair = candidates.loc[candidates.dataset.eq(old.dataset) & candidates.featured_target.eq(old.featured_target)] if not candidates.empty else candidates
        if pair.empty:
            details = ''
            if failure_audit is not None:
                failed = failure_audit.loc[failure_audit.dataset.eq(old.dataset) & failure_audit.featured_target.eq(old.featured_target)]
                columns = [c for c in ['sample', 'featured_fraction', 'n_cells', 'retained_tile_fraction', 'n_other_types_ge_min', 'failed_rules'] if c in failed]
                details = '\n' + failed[columns].to_string(index=False)
            raise ValueError(f'No rechecked cached {old.featured_target} crop qualifies in {old.dataset}. See Figure_S9B_location_revalidation_audit.csv for actual fractions and failed_rules. The thresholds were preserved; no full region search or annotation audit was run.' + details)
        original = pair.loc[pair['sample'].eq(old.sample)]
        for coordinate in ['x0', 'y0', 'x1', 'y1']:
            original = original.loc[original[coordinate].eq(getattr(old, coordinate))]
        kept = not original.empty
        selected = (original if kept else pair.sort_values(['quality_score', 'n_cells', 'sample', 'y0', 'x0'], ascending=[False, False, True, True, True])).iloc[0]
        chosen.append(selected.to_dict())
        audit.append(dict(dataset=old.dataset, featured_target=old.featured_target, previous_sample=old.sample, selected_sample=selected['sample'], previous_roi=[getattr(old, c) for c in ['x0', 'y0', 'x1', 'y1']], selected_roi=[int(selected[c]) for c in ['x0', 'y0', 'x1', 'y1']], kept_original_crop=kept, status='original qualifying crop kept' if kept else 'failed crop replaced with qualifying cached location'))
    return (pd.DataFrame(chosen), pd.DataFrame(audit))

def evaluate_updated_region_counts(regions, rules):
    """Evaluate every featured target using already recomputed counts for each saved ROI."""
    keys = ['dataset', 'sample', 'x0', 'y0', 'x1', 'y1']
    count_columns = [f'count__{target}' for target in TARGET_COLORS]
    required = keys + ['n_cells', 'n_retained_tiles'] + count_columns
    if set(required) - set(regions):
        raise ValueError(f'Updated ROI table lacks {sorted(set(required) - set(regions))}')
    if regions.groupby(keys, dropna=False)[['n_cells', 'n_retained_tiles'] + count_columns].nunique().gt(1).any(axis=None):
        raise ValueError('Updated counts disagree for the same cached region.')
    base = regions[required].drop_duplicates(keys).copy()
    counts = base[count_columns].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(counts).all(axis=None) or counts.lt(0).any(axis=None) or counts.mod(1).ne(0).any(axis=None) or (not counts.sum(axis=1).eq(base.n_cells).all()):
        raise ValueError('Updated region counts must be nonnegative integers summing to n_cells.')
    fractions = counts.div(base.n_cells.replace(0, np.nan), axis=0).fillna(0)
    for target in TARGET_COLORS:
        base[f'fraction__{target}'] = fractions[f'count__{target}']
    base['retained_tile_fraction'] = base.n_retained_tiles / rules['side_tiles'] ** 2
    evaluations = []
    targets = list(TARGET_COLORS)[:-1]
    for target in targets:
        result = base.copy()
        featured = result[f'fraction__{target}']
        others = result[[f'fraction__{other}' for other in targets if other != target]]
        result['featured_target'] = target
        result['featured_fraction'] = featured
        result['min_other_fraction'] = others.min(axis=1)
        result['all_other_types_ge_preferred'] = result.min_other_fraction.ge(rules['preferred_other_fraction'])
        result['quality_score'] = 10 * result.all_other_types_ge_preferred.astype(int) + featured + result.min_other_fraction
        result['cell_count_pass'] = result.n_cells.ge(rules['min_cells'])
        result['coverage_pass'] = result.retained_tile_fraction.ge(rules['min_retained_fraction'])
        result['featured_range_pass'] = featured.between(*rules['featured_range'])
        result['n_other_types_ge_min'] = others.ge(rules['other_min_fraction']).sum(axis=1)
        result['other_types_pass'] = result.n_other_types_ge_min.ge(rules['min_other_types'])
        checks = ['cell_count_pass', 'coverage_pass', 'featured_range_pass', 'other_types_pass']
        result['qualifies'] = result[checks].all(axis=1)
        result['failed_rules'] = result[checks].apply(lambda row: ', '.join((name.removesuffix('_pass') for name, passed in row.items() if not passed)), axis=1)
        result['validation'] = 'All targets evaluated from updated counts at saved coordinates'
        evaluations.append(result)
    audit = pd.concat(evaluations, ignore_index=True)
    return (audit.loc[audit.qualifies].copy(), audit)

def assign_revalidated_cached_examples(candidates, saved, datasets, targets):
    """Choose a feasible dataset/target pairing, preserving original qualifying crops first."""
    if candidates.empty:
        raise ValueError('No saved region qualifies under the updated annotation rules.')
    keys = ['dataset', 'sample', 'featured_target', 'x0', 'y0', 'x1', 'y1']
    originals = set(saved[keys].itertuples(index=False, name=None))
    ranked = candidates.copy()
    kept = ranked[keys].apply(lambda row: tuple(row) in originals, axis=1)
    bonus = len(targets) * (ranked.quality_score.max() - ranked.quality_score.min() + 1) + 1
    ranked['_reference_quality_score'] = ranked.quality_score
    ranked['quality_score'] = ranked.quality_score + bonus * kept.astype(int)
    chosen = assign_dataset_examples(ranked, datasets, targets)
    chosen['assignment_score'] = chosen.quality_score
    chosen['quality_score'] = chosen.pop('_reference_quality_score')
    audit = []
    for row in chosen.itertuples(index=False):
        previous = saved.loc[saved.featured_target.eq(row.featured_target)].iloc[0]
        match = row.dataset == previous.dataset and row.sample == previous['sample'] and all((getattr(row, c) == previous[c] for c in ['x0', 'y0', 'x1', 'y1']))
        audit.append(dict(featured_target=row.featured_target, previous_dataset=previous.dataset, selected_dataset=row.dataset, previous_sample=previous['sample'], selected_sample=row.sample, previous_roi=[int(previous[c]) for c in ['x0', 'y0', 'x1', 'y1']], selected_roi=[int(getattr(row, c)) for c in ['x0', 'y0', 'x1', 'y1']], kept_original_crop=match, status='original qualifying crop kept' if match else 'cached dataset/target reassignment'))
    return (chosen, pd.DataFrame(audit))

def load_saved_composition_search(search_dir, rules, cohort):
    """Reuse the three search exports, with rule/cohort/geometry/composition checks."""
    search_dir = Path(search_dir)
    names = ['Figure_S9B_candidates.csv', 'Figure_S9B_candidate_audit.csv', 'Figure_S9B_selection_rules.json']
    missing = [name for name in names if not (search_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f'Saved S9B search lacks {missing}. Set S9_B_SEARCH_DIR to the saved folder. Full search runs only with S9_B_REUSE_SEARCH=False.')
    saved_rules = json.loads((search_dir / names[2]).read_text())
    if saved_rules != json.loads(json.dumps(rules)):
        raise ValueError(f'Saved S9B search rules differ from current settings: saved={saved_rules}, current={rules}. A grid-only redraw can reuse matching search files. No new search was started.')
    candidates = pd.read_csv(search_dir / names[0], dtype={'dataset': str, 'sample': str, 'featured_target': str})
    audit = pd.read_csv(search_dir / names[1], dtype={'dataset': str, 'sample': str, 'featured_target': str})
    targets = list(TARGET_COLORS)[:-1]
    keys = ['dataset', 'sample', 'featured_target']
    expected = {(r.dataset, r.sample, target) for r in cohort.itertuples(index=False) for target in targets}
    if audit.duplicated(keys).any() or set(audit[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError('Saved search audit does not cover the current Table S2 samples and targets.')
    stage_columns = ['n_windows', 'n_coverage_cell_pass', 'n_featured_range_pass', 'n_mixed_pass']
    stages = audit[stage_columns].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(stages).all(axis=None) or (stages < 0).any(axis=None) or (stages % 1 != 0).any(axis=None):
        raise ValueError('Invalid saved search audit counts.')
    if not (stages.n_windows.ge(stages.n_coverage_cell_pass) & stages.n_coverage_cell_pass.ge(stages.n_featured_range_pass) & stages.n_featured_range_pass.ge(stages.n_mixed_pass)).all():
        raise ValueError('Saved search audit has inconsistent threshold counts.')
    available = set(candidates[keys].itertuples(index=False, name=None))
    qualifying = set(audit.loc[stages.n_mixed_pass.gt(0), keys].itertuples(index=False, name=None))
    if candidates.duplicated(keys).any() or available != qualifying:
        raise ValueError('Saved candidates do not match the qualifying sample/target audit rows.')
    if candidates.empty:
        return (candidates, audit)
    geometry = candidates.merge(cohort[['dataset', 'sample', 'tile_width_px', 'tile_height_px']], on=['dataset', 'sample'], validate='many_to_one')
    coords = geometry[['x0', 'y0', 'x1', 'y1']].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(coords).all(axis=None) or (coords < 0).any(axis=None) or (coords % 1 != 0).any(axis=None):
        raise ValueError('Saved search has invalid ROI coordinates.')
    if not ((coords.x1 - coords.x0).eq(rules['side_tiles'] * geometry.tile_width_px) & (coords.y1 - coords.y0).eq(rules['side_tiles'] * geometry.tile_height_px) & (coords.x0 % geometry.tile_width_px).eq(0) & (coords.y0 % geometry.tile_height_px).eq(0)).all():
        raise ValueError('Saved candidate region sizes/origins disagree with the current tile geometry.')
    fractions = candidates[[f'fraction__{target}' for target in TARGET_COLORS]].apply(pd.to_numeric, errors='raise')
    counts = candidates[[f'count__{target}' for target in TARGET_COLORS]].apply(pd.to_numeric, errors='raise')
    totals = pd.to_numeric(candidates.n_cells, errors='raise')
    coverage = pd.to_numeric(candidates.n_retained_tiles, errors='raise')
    if not (np.isfinite(fractions).all(axis=None) and fractions.ge(0).all(axis=None) and fractions.le(1).all(axis=None) and np.allclose(fractions.sum(axis=1), 1) and np.isfinite(counts).all(axis=None) and counts.ge(0).all(axis=None) and (counts % 1).eq(0).all(axis=None) and counts.sum(axis=1).eq(totals).all() and totals.ge(rules['min_cells']).all() and np.allclose(fractions, counts.to_numpy() / totals.to_numpy()[:, None]) and np.isfinite(coverage).all() and (coverage % 1).eq(0).all() and coverage.le(rules['side_tiles'] ** 2).all() and (coverage / rules['side_tiles'] ** 2).ge(rules['min_retained_fraction']).all() and np.allclose(candidates.retained_tile_fraction, coverage / rules['side_tiles'] ** 2)):
        raise ValueError('Saved candidate fractions/counts/coverage are inconsistent.')
    for i, row in candidates.iterrows():
        featured = fractions.loc[i, f'fraction__{row.featured_target}']
        others = fractions.loc[i, [f'fraction__{target}' for target in targets if target != row.featured_target]]
        low, high = rules['featured_range']
        if not (low <= featured <= high and others.ge(rules['other_min_fraction']).sum() >= rules['min_other_types']):
            raise ValueError('A saved candidate fails the current composition rules.')
        preferred = bool(others.min() >= rules['preferred_other_fraction'])
        score = 10 * preferred + featured + others.min()
        if not np.isclose(row.quality_score, score) or not np.isclose(row.featured_fraction, featured):
            raise ValueError('Saved candidate ranking differs from its reference composition.')
        candidates.loc[i, 'all_other_types_ge_preferred'] = preferred
        candidates.loc[i, 'min_other_fraction'] = others.min()
    return (candidates, audit)

def assign_dataset_examples(candidates, datasets, targets):
    """Find the best complete one-dataset/one-featured-type assignment; no relaxation."""
    from itertools import permutations
    datasets, targets = (list(datasets), list(targets))
    if not datasets or len(datasets) != len(targets) or len(set(datasets)) != len(datasets) or (len(set(targets)) != len(targets)):
        raise ValueError('Need equally sized distinct dataset and target lists.')
    best_pairs = candidates.sort_values(['quality_score', 'sample', 'y0', 'x0'], ascending=[False, True, True, True]).drop_duplicates(['dataset', 'featured_target'])
    lookup = {(r['dataset'], r['featured_target']): r for _, r in best_pairs.iterrows()}
    chosen, best_score = (None, -np.inf)
    for order in permutations(datasets):
        pairs = list(zip(order, targets))
        if not all((pair in lookup for pair in pairs)):
            continue
        score = sum((float(lookup[pair].quality_score) for pair in pairs))
        if score > best_score:
            chosen, best_score = ([lookup[pair] for pair in pairs], score)
    if chosen is None:
        raise ValueError('No complete dataset/cell-type pairing meets the composition rules. Inspect Figure_S9B_candidate_audit.csv; thresholds were not relaxed.')
    return pd.DataFrame(chosen).reset_index(drop=True)

def prefer_featured_samples(candidates, selected, preferences):
    """Replace only requested featured examples with qualifying cached rows from the same dataset."""
    result, audit = (selected.copy(), [])
    if result.featured_target.duplicated().any() or result.dataset.duplicated().any():
        raise ValueError('Selected examples must have distinct datasets and featured targets.')
    for target, (dataset, sample) in preferences.items():
        current = result.loc[result.featured_target.eq(target)]
        if len(current) != 1 or current.dataset.iloc[0] != dataset:
            raise ValueError('A preferred sample must belong to the existing featured example dataset.')
        available = candidates.loc[candidates.featured_target.eq(target) & candidates.dataset.eq(dataset) & candidates['sample'].eq(sample)]
        accepted = not available.empty
        if accepted:
            replacement = available.sort_values(['quality_score', 'y0', 'x0'], ascending=[False, True, True]).iloc[[0]]
            position = list(result.featured_target).index(target)
            result = pd.concat([result.iloc[:position], replacement, result.iloc[position + 1:]], ignore_index=True)
        audit.append(dict(featured_target=target, requested_dataset=dataset, requested_sample=sample, qualifying_saved_candidate=accepted, selected_sample=result.loc[result.featured_target.eq(target), 'sample'].iloc[0], status='preferred sample selected' if accepted else 'no qualifying saved candidate; previous example retained'))
    return (result, pd.DataFrame(audit))

def select_tumor_rois(cells, tile_size_px, *, side_tiles=4, n_regions=2, min_retained_fraction=0.75, min_cells=100, exclude_rois=()):
    """Rank disjoint windows by pooled tumor cell fraction among retained cells."""
    if side_tiles < 1 or int(side_tiles) != side_tiles or tile_size_px <= 0:
        raise ValueError('Window and tile sizes must be positive integers.')
    if n_regions < 1 or int(n_regions) != n_regions or (not 0 < min_retained_fraction <= 1) or (min_cells < 1):
        raise ValueError('Invalid number of regions, coverage threshold or cell-count threshold.')
    ranked = region_compositions(cells, tile_size_px, side_tiles)
    ranked['n_tumor_cells'] = ranked['count__Tumor Cells']
    ranked['tumor_fraction'] = ranked['fraction__Tumor Cells']
    ranked = ranked.loc[ranked.n_cells.ge(min_cells) & ranked.retained_tile_fraction.ge(min_retained_fraction)]
    ranked = ranked.sort_values(['tumor_fraction', 'n_cells', 'y0', 'x0'], ascending=[False, False, True, True])
    chosen, occupied = ([], list(exclude_rois))
    for row in ranked.itertuples(index=False):
        roi = (row.x0, row.y0, row.x1, row.y1)
        if any((roi[0] < other[2] and roi[2] > other[0] and (roi[1] < other[3]) and (roi[3] > other[1]) for other in occupied)):
            continue
        chosen.append(row._asdict())
        occupied.append(roi)
        if len(chosen) == n_regions:
            break
    return pd.DataFrame(chosen, columns=ranked.columns)

def load_all_annotated_region(label_dir, dataset, sample, coordinates, roi, retained_region):
    """Match all final labels inside the H&E ROI, independently of retained tile membership."""
    from .epithelial_morphology import validate_ids
    label_path = Path(label_dir) / dataset / sample / 'final_cell_labels.csv.gz'
    labels = pd.read_csv(label_path, dtype=str, keep_default_na=False, usecols=['dataset', 'sample', 'cell_id', 'final_refined_label', 'final_broad_label'])
    validate_ids(labels)
    validate_ids(coordinates)
    if not (labels.dataset.eq(dataset) & labels['sample'].eq(sample)).all():
        raise ValueError('Full final labels belong to another sample.')
    if set(labels.final_broad_label) - set(PPTX_BROAD_COLORS):
        raise ValueError('Full final labels must use the same PPTX-aligned annotation source.')
    xy = coordinates[['he_x_px', 'he_y_px']].apply(pd.to_numeric, errors='coerce')
    inside = np.isfinite(xy).all(axis=1) & xy.he_x_px.ge(roi[0]) & xy.he_x_px.lt(roi[2]) & xy.he_y_px.ge(roi[1]) & xy.he_y_px.lt(roi[3])
    located = coordinates.loc[inside, ['cell_id']].copy()
    located[['he_x_px', 'he_y_px']] = xy.loc[inside]
    matched, method = match_rows(labels, located.cell_id)
    annotated = located.merge(matched, on='cell_id', validate='one_to_one')
    if annotated.empty:
        raise ValueError('The H&E region has no cells with final annotations.')
    annotated['cell_type'] = annotated.final_broad_label
    annotated['plot_group'] = training_group(annotated.final_broad_label)
    annotated['_display_cell_id'] = annotated.cell_id
    retained, retained_method = match_rows(annotated, retained_region.cell_id)
    if len(retained) != len(retained_region):
        raise ValueError('Some retained region cells are absent from the full final annotations.')
    check_columns = ['cell_id', 'plot_group'] + (['cell_type'] if 'cell_type' in retained_region else [])
    check = retained[check_columns].merge(retained_region[check_columns], on='cell_id', validate='one_to_one', suffixes=('_full', '_retained'))
    if any((not check[f'{c}_full'].eq(check[f'{c}_retained']).all() for c in check_columns[1:])):
        raise ValueError('Full annotations differ from the retained reference labels.')
    annotated['is_retained_training_cell'] = annotated.cell_id.isin(retained['_display_cell_id'])
    annotated = annotated.drop(columns='_display_cell_id')
    metadata = dict(final_label_path=str(label_path), n_coordinate_region_cells=len(located), n_all_annotated_region_cells=len(annotated), n_coordinate_region_cells_without_final_labels=len(located) - len(annotated), n_annotated_region_cells_outside_retained_tiles=int((~annotated.is_retained_training_cell).sum()), label_coordinate_match_method=method, retained_id_match_method=retained_method)
    return (annotated, metadata)

def load_region_polygons(root, sample, aligned, region, *, cell_table_path=None, boundary_path=None):
    """Load actual Xenium cell boundaries and verify the saved micron-to-H&E affine."""
    from dataclasses import replace
    from . import standard_xenium, tile_filter_review, epithelial_morphology
    from .retained_final_labels import resolve_project_path
    from .tile_utils import apply_affine
    paths = tile_filter_review._sample_paths_for_area_recalculation(sample)
    overrides = {key: resolve_project_path(value, Path(root)) for key, value in [('cell_table', cell_table_path), ('cell_boundaries', boundary_path)] if value is not None}
    if overrides:
        paths = replace(paths, **overrides)
    boundary_table = standard_xenium.find_boundary_table(paths, 'cell')
    zarr_path = standard_xenium.find_outs_file(paths, 'cells.zarr.zip')
    if boundary_table is None and zarr_path is None:
        raise FileNotFoundError('No actual cell boundaries: provide cell_boundaries CSV/parquet or cells.zarr.zip.')
    if boundary_table is None:
        paths = replace(paths, cell_table=None)
    source, native_order = standard_xenium.load_cells(paths)
    source['cell_id'] = source.cell_id.astype(str)
    source = source.rename(columns={'cell_centroid_x': 'x_centroid', 'cell_centroid_y': 'y_centroid'})
    affine, residual = epithelial_morphology.infer_he_affine(source, aligned, tolerance=1.0)
    source['_source_cell_id'] = source.cell_id
    selected, method = match_rows(source, region.cell_id)
    if len(selected) != len(region):
        raise ValueError('Region cells do not all match the source segmentation cells.')
    source_to_plot = dict(zip(selected._source_cell_id, selected.cell_id))
    raw_polygons = {str(cell_id): [] for cell_id in region.cell_id}
    if boundary_table is not None:
        table = epithelial_morphology.read_table(boundary_table)
        if not {'cell_id', 'vertex_x', 'vertex_y'}.issubset(table):
            raise ValueError('Cell-boundary table needs cell_id, vertex_x and vertex_y.')
        boundary_ids = table[['cell_id']].drop_duplicates().assign(_boundary_id=lambda x: x.cell_id)
        mapped, _ = match_rows(boundary_ids, selected._source_cell_id)
        boundary_to_plot = dict(zip(mapped['_boundary_id'], mapped.cell_id.map(source_to_plot)))
        table = table.loc[table.cell_id.isin(boundary_to_plot)]
        for raw_id, group in table.groupby('cell_id', sort=False):
            rings = group.groupby('label_id', sort=True, dropna=False) if 'label_id' in group else [(None, group)]
            for _, ring in rings:
                if 'vertex_index' in ring:
                    ring = ring.sort_values('vertex_index', kind='stable')
                raw_polygons[boundary_to_plot[raw_id]].append(ring[['vertex_x', 'vertex_y']].apply(pd.to_numeric, errors='coerce').to_numpy(float))
        polygon_source = str(boundary_table)
    else:
        if not native_order:
            raise ValueError('Native polygon indices require the original cells.zarr cell order.')
        boundaries = standard_xenium.load_boundaries(paths, source, boundary_kind='cell')
        if boundaries is None:
            raise ValueError('cells.zarr.zip has no supported cell-boundary polygon set.')
        for cell in selected.itertuples(index=False):
            index = int(cell.cell_index)
            row = int(boundaries.polygon_row_for_cell[index])
            if row < 0:
                continue
            count = int(boundaries.num_vertices[row])
            raw_polygons[cell.cell_id].append(boundaries.vertices[row, :2 * count].reshape(count, 2))
        polygon_source = str(zarr_path)
    polygons, audit = ({}, [])
    for cell in region.itertuples(index=False):
        rings = raw_polygons[cell.cell_id]
        status = 'ok' if rings else 'missing_cell_boundary'
        transformed = []
        for xy in rings:
            valid = len(xy) >= 3 and np.isfinite(xy).all() and (len(np.unique(xy, axis=0)) >= 3)
            if valid:
                centered = xy - xy[0]
                area = abs(np.sum(centered[:, 0] * np.roll(centered[:, 1], -1) - centered[:, 1] * np.roll(centered[:, 0], -1))) / 2
                valid = area > 0
            if not valid:
                status, transformed = ('invalid_cell_boundary', [])
                break
            transformed.append(apply_affine(xy, affine))
        if transformed:
            polygons[cell.cell_id] = transformed
        audit.append(dict(cell_id=cell.cell_id, plot_group=cell.plot_group, polygon_status=status, n_polygons=len(transformed), n_vertices=sum((len(xy) for xy in transformed))))
    metadata = dict(boundary_source=polygon_source, boundary_kind='cell', source_units='microns', plot_units='full-resolution H&E pixels', affine_um_to_he_pixels=affine.tolist(), affine_max_error_px=residual, cell_id_match_method=method, n_region_cells=len(region), n_cells_with_polygons=len(polygons), n_cells_without_polygons=len(region) - len(polygons))
    return (polygons, pd.DataFrame(audit), metadata)

def plot_spatial(crop, cells, fractions, roi, tile_size_px, output_dir, title, pixel_size_um=None, stem='Figure_S9B', polygons=None, annotation_cells=None, he_grid_style=None, scale_bar_color='black', scale_bar_um=None, annotation_source=None, smooth_muscle_as_uncertain=False):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Rectangle, Patch
    x0, y0, x1, y1 = roi
    inside = cells.he_x_px.ge(x0) & cells.he_x_px.lt(x1) & cells.he_y_px.ge(y0) & cells.he_y_px.lt(y1)
    region = cells.loc[inside].copy()
    if region.empty:
        raise ValueError('Chosen ROI contains no retained training cells.')
    full_annotation_panel = annotation_cells is not None
    annotation_cells = cells if annotation_cells is None else annotation_cells
    annotation_inside = annotation_cells.he_x_px.ge(x0) & annotation_cells.he_x_px.lt(x1) & annotation_cells.he_y_px.ge(y0) & annotation_cells.he_y_px.lt(y1)
    annotation_region = annotation_cells.loc[annotation_inside].copy()
    if not polygons or not any((cell_id in polygons for cell_id in annotation_region.cell_id)):
        raise ValueError('The annotation panel requires actual cell polygons; no centroid fallback.')
    maps = fractions.copy()
    grid = maps.tile_id.str.extract('_row-(\\d+)_col-(\\d+)$').astype(int)
    maps['x0_px'], maps['y0_px'] = (grid[1] * tile_size_px, grid[0] * tile_size_px)
    maps = maps.loc[(maps.x0_px < x1) & (maps.x0_px + tile_size_px > x0) & (maps.y0_px < y1) & (maps.y0_px + tile_size_px > y0)]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = list(axes.flat)
    fig.subplots_adjust(bottom=0.16, top=0.87, left=0.03, right=0.86, wspace=0.25, hspace=0.25)
    for ax in axes:
        ax.set(xlim=(x0, x1), ylim=(y1, y0), aspect='equal')
        ax.set_axis_off()
    axes[0].imshow(crop, extent=(x0, x1, y1, y0))
    x_edges = np.arange(np.ceil(x0 / tile_size_px) * tile_size_px, x1 + 1, tile_size_px)
    y_edges = np.arange(np.ceil(y0 / tile_size_px) * tile_size_px, y1 + 1, tile_size_px)
    grid_style = dict(HE_GRID_STYLE, **he_grid_style or {})
    axes[0].vlines(x_edges, y0, y1, **grid_style)
    axes[0].hlines(y_edges, x0, x1, **grid_style)
    axes[0].set_title('Registered H&E')
    axes[1].set_facecolor('white')
    axes[1].set_title('All annotated cell polygons' if full_annotation_panel else 'Annotated cell polygons')
    polygon_vertices, polygon_colors = ([], [])
    source_labels = annotation_region.get('cell_type', annotation_region.get('final_broad_label', annotation_region.plot_group))
    if set(source_labels) - set(PPTX_BROAD_COLORS):
        raise ValueError('Polygon colors require the eight source broad labels.')
    colors = figure_colors(smooth_muscle_as_uncertain)
    source_labels = source_labels.replace(figure_label_mapping(smooth_muscle_as_uncertain))
    annotation_region['figure_broad_label'] = source_labels
    for group, color in colors.items():
        subset = annotation_region.loc[source_labels.eq(group)]
        for cell_id in subset.cell_id:
            for xy in polygons.get(cell_id, []):
                polygon_vertices.append(xy)
                polygon_colors.append(color)
    axes[1].add_collection(PolyCollection(polygon_vertices, facecolors=polygon_colors, edgecolors='white', linewidths=0.15, rasterized=True))
    for ax, target in zip(list(axes)[2:], TARGET_COLORS):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor='#EEEEEE', edgecolor='none'))
        for _, row in maps.iterrows():
            ax.add_patch(Rectangle((row.x0_px, row.y0_px), tile_size_px, tile_size_px, facecolor=plt.get_cmap('magma')(row[target]), linewidth=0.2, edgecolor='white'))
        ax.set_title('pan-APC' if target == 'pan-APC + B' else 'Uncertain' if smooth_muscle_as_uncertain and target == 'Other/uncertain' else target)
    if scale_bar_um is not None and pixel_size_um is None:
        raise ValueError('An explicit micrometer scale bar requires verified H&E pixel spacing.')
    if pixel_size_um is not None:
        if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
            raise ValueError('A scale bar requires a positive verified H&E pixel size.')
        bar_um = scale_bar_um if scale_bar_um is not None else 10 ** np.floor(np.log10((x1 - x0) * pixel_size_um / 4))
        if not np.isfinite(bar_um) or bar_um <= 0 or bar_um / pixel_size_um > 0.88 * (x1 - x0):
            raise ValueError('Requested physical scale bar is invalid or too long for this region.')
        bx, by = (x0 + 0.06 * (x1 - x0), y1 - 0.08 * (y1 - y0))
        axes[0].plot([bx, bx + bar_um / pixel_size_um], [by, by], color=scale_bar_color, linewidth=3)
        axes[0].text(bx, by - 0.03 * (y1 - y0), f'{bar_um:g} µm', fontsize=9, color=scale_bar_color)
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap='magma'), cax=fig.add_axes([0.89, 0.25, 0.014, 0.48]), ticks=[0, 0.25, 0.5, 0.75, 1], label='Cell-count fraction (all cells per tile)')
    fig.legend([Patch(facecolor=c) for c in colors.values()], ['pan-APC' if group == 'pan-APCs' else group for group in colors], loc='lower center', ncol=4, frameon=False, fontsize=9)
    fig.suptitle(title, x=0.03, ha='left', fontweight='bold')
    missing = sum((cell_id not in polygons for cell_id in annotation_region.cell_id))
    population = 'all annotated cells in region' if full_annotation_panel else 'supplied annotated cells'
    note = f'Polygons: {population}. Maps: retained training tiles; gray areas excluded.\nMaps pool pan-APCs + B; the remainder includes smooth muscle and uncertain cells.'
    if annotation_source is not None:
        note = f'Polygons and reference fractions: {annotation_source}; polygon scope: {population}.\nMaps: retained tiles; gray areas excluded. pan-APCs + B pooled; remainder: smooth muscle + uncertain.'
    if smooth_muscle_as_uncertain:
        note = note.replace('the remainder includes smooth muscle and uncertain cells.', 'smooth muscle grouped with Uncertain for display.').replace('remainder: smooth muscle + uncertain.', 'smooth muscle grouped with Uncertain for display.')
    if missing:
        note += f'\nPolygons drawn: {len(annotation_region) - missing:,}/{len(annotation_region):,} cells.'
    fig.text(0.03, 0.115, note, fontsize=9)
    paths = save_figure(fig, output_dir, stem)
    region.to_csv(Path(output_dir) / f'{stem}_cells.csv.gz', index=False)
    annotation_region.to_csv(Path(output_dir) / f'{stem}_annotated_cells.csv.gz', index=False)
    maps.to_csv(Path(output_dir) / f'{stem}_tile_proportions.csv', index=False)
    return paths

def sample_he_image_path(root, sample, image_path=None):
    from . import tile_filter_review
    from .retained_final_labels import resolve_project_path
    candidate = image_path or tile_filter_review._sample_paths_for_area_recalculation(sample).he_image
    path = resolve_project_path(candidate, Path(root)) if candidate else None
    if path is None or not path.is_file():
        raise FileNotFoundError('No matching full-resolution H&E image; set S9_B_IMAGE_PATHS for this sample.')
    return path

def he_image_size(root, sample, image_path=None):
    """Read original image dimensions from the TIFF header, without decoding a slide."""
    import tifffile
    from .tile_utils import get_hw
    with tifffile.TiffFile(sample_he_image_path(root, sample, image_path)) as tif:
        height, width = get_hw(tif.series[0].levels[0].shape)
    return (width, height)

def export_spatial_example(root, cells, sample, tile_size_px, output_dir, *, roi=None, pixel_size_um=None, image_path=None, coordinate_path=None, side_tiles=10, stem='Figure_S9B', selection_method=None, featured_target='Tumor Cells', composition_rules=None, cell_table_path=None, boundary_path=None, label_dir=None, he_grid_style=None, scale_bar_color='black', scale_bar_um=None, annotation_source=None, smooth_muscle_as_uncertain=False):
    """Read a full-resolution WSI region without loading the whole slide."""
    from contextlib import ExitStack
    import tifffile
    import zarr
    from .tile_utils import extract_rgb_tile, get_hw, resolve_zarr_array
    automatic_roi = roi is None
    image_path = sample_he_image_path(root, sample, image_path)
    if coordinate_path is None:
        coordinate_path = next((sample.sample_dir / f'{sample.sample}_{suffix}.csv' for suffix in ('cells_with_annotations', 'cells_with_cluster_labels', 'cells_aligned_to_he') if (sample.sample_dir / f'{sample.sample}_{suffix}.csv').is_file()), None)
    if coordinate_path is None:
        raise FileNotFoundError('No saved H&E-aligned cell coordinates; set S9_B_COORDINATE_PATHS for this sample.')
    coordinates = pd.read_csv(coordinate_path, dtype={'cell_id': str}, keep_default_na=False, usecols=['cell_id', 'he_x_px', 'he_y_px'])
    matched, method = attach_coordinates(cells, coordinates, tile_size_px)
    roi = spatial_roi(matched, tile_size_px, roi, side_tiles=side_tiles)
    with ExitStack() as stack:
        tif = stack.enter_context(tifffile.TiffFile(image_path))
        store = tif.series[0].levels[0].aszarr()
        stack.callback(store.close)
        image = resolve_zarr_array(zarr.open(store, mode='r'))
        height, width = get_hw(image.shape)
        x0, y0, x1, y1 = roi
        roi = (x0, y0, min(x1, width), min(y1, height))
        if roi[2] <= x0 or roi[3] <= y0:
            raise ValueError('Chosen ROI lies outside the H&E image.')
        crop = extract_rgb_tile(image, *roi)
    inside = matched.he_x_px.ge(roi[0]) & matched.he_x_px.lt(roi[2]) & matched.he_y_px.ge(roi[1]) & matched.he_y_px.lt(roi[3])
    region = matched.loc[inside]
    if region.empty or featured_target not in list(TARGET_COLORS)[:-1]:
        raise ValueError('Need retained region cells and a valid featured target.')
    fractions = region.plot_group.value_counts().reindex(TARGET_COLORS, fill_value=0) / len(region)
    if composition_rules is not None:
        others = fractions.drop([featured_target, 'Other/uncertain'])
        low, high = composition_rules['featured_range']
        required_side = composition_rules['side_tiles'] * tile_size_px
        if not (roi[2] - roi[0] == required_side and roi[3] - roi[1] == required_side and (region.tile_id.nunique() / composition_rules['side_tiles'] ** 2 >= composition_rules['min_retained_fraction']) and (len(region) >= composition_rules['min_cells']) and (low <= fractions[featured_target] <= high) and (others.ge(composition_rules['other_min_fraction']).sum() >= composition_rules['min_other_types'])):
            raise ValueError('Actual H&E crop composition fails the selected rules; inspect image bounds/ROI.')
    annotation_region, annotation_metadata = load_all_annotated_region(label_dir or Path(root) / 'final_cell_labels', sample.dataset, sample.sample, coordinates, roi, region)
    polygons, polygon_audit, polygon_metadata = load_region_polygons(root, sample, coordinates, annotation_region, cell_table_path=cell_table_path, boundary_path=boundary_path)
    polygon_audit.to_csv(Path(output_dir, f'{stem}_polygon_audit.csv'), index=False)
    polygon_records = []
    for cell_id, rings in polygons.items():
        for ring_index, xy in enumerate(rings):
            polygon_records.extend((dict(cell_id=cell_id, polygon_index=ring_index, vertex_index=i, he_x_px=float(x), he_y_px=float(y)) for i, (x, y) in enumerate(xy)))
    pd.DataFrame(polygon_records, columns=['cell_id', 'polygon_index', 'vertex_index', 'he_x_px', 'he_y_px']).to_csv(Path(output_dir, f'{stem}_cell_polygons.csv.gz'), index=False)
    figures = plot_spatial(crop, matched, tile_proportions(cells), roi, tile_size_px, output_dir, f"{sample.dataset} / {sample.sample} | Reference {('pan-APC' if featured_target == 'pan-APC + B' else featured_target)}: {fractions[featured_target]:.1%}", pixel_size_um, stem=stem, polygons=polygons, annotation_cells=annotation_region, he_grid_style=he_grid_style, scale_bar_color=scale_bar_color, scale_bar_um=scale_bar_um, annotation_source=annotation_source, smooth_muscle_as_uncertain=smooth_muscle_as_uncertain)
    metadata = dict(dataset=sample.dataset, sample=sample.sample, he_image_path=str(image_path), coordinate_path=str(coordinate_path), cell_id_match_method=method, roi_he_pixels=list(roi), tile_size_px=int(tile_size_px), effective_he_pixel_size_um=pixel_size_um, roi_size_um=[(roi[2] - roi[0]) * pixel_size_um, (roi[3] - roi[1]) * pixel_size_um] if pixel_size_um else None, n_retained_region_cells=len(region), tumor_fraction=float(fractions['Tumor Cells']), featured_target=featured_target, featured_fraction=float(fractions[featured_target]), reference_annotation_source=annotation_source, figure_display_mapping=figure_label_mapping(smooth_muscle_as_uncertain), model_training_validation='See separate Figure_S9_model_target_version_audit.csv', reference_fractions=fractions.to_dict(), composition_rules=composition_rules, annotation_panel='all final-annotated cells in ROI, independent of retained tiles', annotation_metadata=annotation_metadata, polygon_metadata=polygon_metadata, annotation_region_fractions=annotation_region.plot_group.value_counts().reindex(TARGET_COLORS, fill_value=0).div(len(annotation_region)).to_dict(), annotation_broad_fractions=annotation_region.cell_type.value_counts().reindex(PPTX_BROAD_COLORS, fill_value=0).div(len(annotation_region)).to_dict(), annotation_figure_broad_fractions=annotation_region.cell_type.replace(figure_label_mapping(smooth_muscle_as_uncertain)).value_counts().reindex(list(figure_colors(smooth_muscle_as_uncertain)), fill_value=0).div(len(annotation_region)).to_dict(), he_tile_grid=dict(dict(HE_GRID_STYLE, **he_grid_style or {}), spacing_px=int(tile_size_px), origin_he_pixels=[0, 0]), he_scale_bar=dict(color=scale_bar_color, length_um=float(scale_bar_um if scale_bar_um is not None else 10 ** np.floor(np.log10((roi[2] - roi[0]) * pixel_size_um / 4))) if pixel_size_um else None), roi_selection=selection_method or ('nearest retained tile to median grid position' if automatic_roi else 'explicit'))
    Path(output_dir, f'{stem}_metadata.json').write_text(json.dumps(metadata, indent=2))
    return figures
