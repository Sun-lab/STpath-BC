"""Validate matched LOIO model comparisons before plotting completed runs."""
import numpy as np
import pandas as pd
from .stpath_xgboost import MODEL_NAMES

# Display order from manuscript Figure 7A; training encoder order is unchanged.
COMPARISON_MODELS = ('ResNet50', 'Conch', 'ProvGigapath', 'UNI2h', 'Virchow', 'Virchow2', 'Combined')
CELL_TYPE_LABELS = {
    'Tumor Cells': 'Tumor', 'Stromal Cells': 'Stromal',
    'Normal Epithelial Cells': 'Normal Epithelial', 'T Cells': 'T',
    'pan-APC + B': 'pan-APC',
}


def matched_comparison_metrics(metrics, cell_types, models=COMPARISON_MODELS):
    """Require the requested models on identical folds; preserve undefined correlations."""
    if 'Combined' not in models or len(set(models)) != len(models):
        raise ValueError('Comparison models must be unique and include Combined as the fold reference')
    required = {'cell_type', 'model_type', 'prediction_scale', 'outer_fold',
                'held_out_individual', 'n_test_tiles', 'n_positive_test_tiles',
                'observed_min', 'observed_max', 'mae', 'pearson'}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f'Comparison metrics are missing columns: {sorted(missing)}')
    selected = metrics.loc[metrics.cell_type.isin(cell_types)
                           & metrics.model_type.isin(models)
                           & metrics.prediction_scale.eq('raw')].copy()
    keys = ['cell_type', 'outer_fold', 'held_out_individual', 'model_type']
    if selected.duplicated(keys).any():
        raise ValueError('Duplicate raw model/fold metrics; configure one source per model')
    for cell_type in cell_types:
        target = selected.loc[selected.cell_type.eq(cell_type)]
        reference = target.loc[target.model_type.eq('Combined')]
        fold_keys = ['outer_fold', 'held_out_individual']
        expected = set(reference[fold_keys].itertuples(index=False, name=None))
        if not expected:
            raise ValueError(f'{cell_type}: Combined has no completed fold metrics')
        summary_columns = ['n_test_tiles', 'n_positive_test_tiles', 'observed_min', 'observed_max']
        reference = reference.set_index(fold_keys)[summary_columns].sort_index()
        for model in models:
            model_data = target.loc[target.model_type.eq(model)]
            observed = set(model_data[fold_keys].itertuples(index=False, name=None))
            if observed != expected:
                raise ValueError(f'{cell_type}/{model}: held-out folds do not match Combined; '
                                 f'missing={sorted(expected - observed)}, extra={sorted(observed - expected)}')
            summary = model_data.set_index(fold_keys)[summary_columns].sort_index()
            if not np.allclose(summary.to_numpy(dtype=float), reference.to_numpy(dtype=float),
                               rtol=0, atol=1e-12, equal_nan=False):
                raise ValueError(f'{cell_type}/{model}: held-out target summaries differ')
    return selected.reset_index(drop=True)




def plot_model_comparison(metrics, output_dir, output_stem="Figure_6A", *, models=COMPARISON_MODELS):
    """Plot the selected horizontal LOIO comparison, preserving undefined correlations."""
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FormatStrFormatter
    MONITOR_OUTPUT_DIR = Path(output_dir)
    MONITOR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_order = list(models)
    cell_type_order = list(CELL_TYPE_LABELS)
    cell_type_labels = [CELL_TYPE_LABELS[name] for name in cell_type_order]
    model_colors = {'ResNet50': '#FAF0E3', 'Conch': '#C43E96', 'ProvGigapath': '#DEDBEE', 'UNI2h': '#06948E', 'Virchow': '#F3CDCC', 'Virchow2': '#F0CF7F', 'Combined': '#FF6B6B', 'Combined (CRC selection)': '#527F58'}
    model_display_names = {'ProvGigapath': 'Prov-GigaPath', 'UNI2h': 'UNI2-h'}
    figure_data = matched_comparison_metrics(metrics, cell_type_order, models=model_order)
    keys = ['cell_type', 'held_out_individual', 'model_type']
    if figure_data.duplicated(keys).any():
        raise ValueError('Duplicate raw fold metrics found for the figure')
    centers = np.arange(len(cell_type_order), dtype=float)
    offsets = np.linspace(-0.32, 0.32, len(model_order))
    box_width = min(0.1, 0.6 / max(1, len(model_order)))
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 4.2), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.94, bottom=0.32, wspace=0.25)
    for ax, (metric, ylabel) in zip(axes.ravel(), [('mae', 'Mean Absolute Error (MAE)'), ('pearson', 'Pearson Correlation')]):
        for cell_index, cell_type in enumerate(cell_type_order):
            for model_index, model_name in enumerate(model_order):
                values = pd.to_numeric(figure_data.loc[figure_data['cell_type'].eq(cell_type) & figure_data['model_type'].eq(model_name), metric], errors='coerce').to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if not len(values):
                    continue
                position = centers[cell_index] + offsets[model_index]
                box = ax.boxplot(values, positions=[position], widths=box_width, patch_artist=True, showfliers=False, manage_ticks=False, medianprops={'color': 'black', 'linewidth': 1.1}, whiskerprops={'color': 'black', 'linewidth': 0.9}, capprops={'color': 'black', 'linewidth': 0.9}, boxprops={'edgecolor': 'black', 'linewidth': 0.9})
                box['boxes'][0].set_facecolor(model_colors[model_name])
                jitter = rng.uniform(-box_width * 0.22, box_width * 0.22, len(values))
                ax.scatter(np.full(len(values), position) + jitter, values, s=9, alpha=0.45, color=model_colors[model_name], edgecolor='black', linewidth=0.25, zorder=3)
        finite_cell_types = set(figure_data.loc[pd.to_numeric(figure_data[metric], errors='coerce').notna(), 'cell_type'])
        for cell_index, cell_type in enumerate(cell_type_order):
            if cell_type not in finite_cell_types:
                ax.text(centers[cell_index], 0.04, 'no valid\nfolds', transform=ax.get_xaxis_transform(), ha='center', va='bottom', color='#777777', fontsize=9)
        ax.set_xticks(centers, cell_type_labels, rotation=25, ha='right', fontsize=10, fontweight='bold')
        ax.tick_params(axis='y', labelsize=9, length=3, width=1.5)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=11)
        ax.set_axisbelow(True)
        ax.grid(axis='y', linestyle='--', color='#BDBDBD', linewidth=0.7, alpha=0.75)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.set_xlim(-0.55, len(cell_type_order) - 0.45)
        ax.set_xlabel('')
        if metric == 'mae':
            ax.set_ylim(0.0, 0.4)
            ax.set_yticks(np.arange(0.0, 0.401, 0.05))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        else:
            ax.set_ylim(-0.5, 1.0)
            ax.set_yticks(np.arange(-0.4, 1.001, 0.2))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        for tick_label in ax.get_yticklabels():
            tick_label.set_fontweight('bold')
    undefined_pearson = int(pd.to_numeric(figure_data['pearson'], errors='coerce').isna().sum())
    all_zero_fold_keys = ['cell_type', 'outer_fold', 'held_out_individual']
    all_zero_folds = figure_data.loc[figure_data['n_positive_test_tiles'].eq(0) & figure_data['observed_min'].eq(0) & figure_data['observed_max'].eq(0), all_zero_fold_keys].drop_duplicates()
    all_zero_counts = all_zero_folds.groupby('cell_type', sort=False).size().reindex(cell_type_order, fill_value=0)
    all_zero_breakdown = '; '.join((f'{cell_type}: {int(count)}' for cell_type, count in all_zero_counts.items() if count))
    note_parts = []
    outside_axes = int((figure_data['mae'].lt(0) | figure_data['mae'].gt(0.4)).sum())
    outside_axes += int((figure_data['pearson'].lt(-0.5) | figure_data['pearson'].gt(1.0)).sum())
    if outside_axes:
        note_parts.append(f'{outside_axes} metric values fall outside the fixed manuscript axes')
    if len(all_zero_folds):
        note_parts.append(f'{len(all_zero_folds)} all-zero held-out folds ({all_zero_breakdown})')
    if undefined_pearson:
        note_parts.append(f'Pearson correlation undefined for {undefined_pearson} model entries')
    legend_handles = [Patch(facecolor=model_colors[name], edgecolor='black', label=model_display_names.get(name, name)) for name in model_order]
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.52, 0.015), ncol=len(model_order), frameon=True, facecolor='white', edgecolor='#BDBDBD', prop={'size': 10, 'weight': 'bold'}, borderpad=0.45, labelspacing=0.35, handlelength=1.7)
    caption_path = MONITOR_OUTPUT_DIR / f'{output_stem}_caption.txt'
    caption_path.write_text('LOIO performance across the five pathology encoders, ResNet50, and Combined. ' + 'Left: MAE. Right: Pearson correlation. ' + 'Each point represents a held-out individual; boxes show interquartile ranges and median lines. ' + ('; '.join(note_parts) + '.' if note_parts else '') + '\n', encoding='utf-8')
    png_path = MONITOR_OUTPUT_DIR / f'{output_stem}.png'
    pdf_path = MONITOR_OUTPUT_DIR / f'{output_stem}.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f'Saved: {png_path}')
    print(f'Saved: {pdf_path}')
    return (fig, axes, legend_handles, figure_data)
