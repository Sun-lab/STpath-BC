from __future__ import annotations
from pathlib import Path
import os
CODE_ROOT = Path(__file__).resolve().parents[1]

def target_settings(root, target):
    from xenium_pipeline import stpath_xgboost as sx
    from xenium_pipeline.storage import load_layout, data_path, curated_table
    settings = load_layout(root)['targets']
    if target not in settings:
        raise ValueError(f'Unknown cell type: {target}')
    spec = settings[target]
    root = Path(root).expanduser().resolve()
    source = spec['annotation_source']
    env = 'XENIUM_EPITHELIAL_ANNOTATION_ROOT' if source == 'epithelial' else 'XENIUM_ANNOTATION_ROOT'
    annotation_root = Path(os.environ.get(env, data_path(root, source + '_annotations'))).expanduser().resolve()
    training_root = root / 'xgboost_training'
    review = curated_table(root, f'target_cell_types_{target}.csv')
    run_id = os.environ.get(f'XGBOOST_{target.upper()}_RUN_ID', spec['run_id'])
    return dict(cell_type=spec['cell_type'], stem=target, run_id=run_id, target_table=training_root / f"tile_targets_{spec['stem']}.csv.gz", annotation_root=annotation_root, feature_root=data_path(root, 'feature_root'), training_root=training_root, mapping_path=curated_table(root, 'sample_individual_mapping.csv'), review_path=review, output_root=training_root / 'runs' / run_id, groups={name: tuple(values) for name, values in spec['groups'].items()}, prevalence=spec['prevalence'], config=sx.TrainingConfig(calibrate=False, nthread=5, runtime_nthread=5, combined_runtime_nthread=15, parallel_model_jobs=3), skip_all_zero=target != 'normal_epithelial_cells')

def prepare_targets(root, target):
    from xenium_pipeline import stpath_xgboost as sx
    import numpy as np
    import pandas as pd
    from xenium_pipeline.workflow_output import display
    settings = target_settings(root, target)
    PROJECT_ROOT = Path(root).expanduser().resolve()
    FULL_CONFIG = settings['config']
    SELECTION_MANIFEST = PROJECT_ROOT / 'tile_filter_cutoff_tests/final_automatic_tile_selection_manifest.csv'
    ANNOTATION_ROOT = settings['annotation_root']
    FEATURE_ROOT = settings['feature_root']
    TRAINING_ROOT = settings['training_root']
    MAPPING_PATH = settings['mapping_path']
    TARGET_REVIEW_PATH = settings['review_path']
    TILE_TARGETS_PATH = settings['target_table']
    TARGET_STEM = settings['stem']
    TARGET_GROUPS = settings['groups']
    RUN_OUTPUT_ROOT = settings['output_root']
    CELL_TYPE = settings['cell_type']
    TARGET_SETTINGS = {'prevalence': settings['prevalence']}
    retained_tiles = None
    sample_table = None
    input_audit = None
    retained_tiles = sx.load_retained_tiles(PROJECT_ROOT, SELECTION_MANIFEST)
    sample_table = retained_tiles[['dataset', 'sample']].drop_duplicates().reset_index(drop=True)
    audit_rows = []
    for row in sample_table.itertuples(index=False):
        annotation = sx.annotation_path(ANNOTATION_ROOT, row.dataset, row.sample)
        audit_rows.append({'input_type': 'cell_annotations', 'dataset': row.dataset, 'sample': row.sample, 'model_type': '', 'path': str(annotation), 'status': 'ready' if annotation.is_file() else 'missing', 'details': ''})
        for model_name in FULL_CONFIG.encoder_names:
            shard = sx.feature_shard_path(FEATURE_ROOT, model_name, row.dataset, row.sample)
            audit_rows.append({'input_type': 'foundation_features', 'dataset': row.dataset, 'sample': row.sample, 'model_type': model_name, 'path': str(shard), 'status': 'ready' if shard.is_file() else 'missing', 'details': ''})
    input_audit = pd.DataFrame(audit_rows)
    sx._atomic_csv(input_audit, TRAINING_ROOT / f'input_audit_{TARGET_STEM}.csv')
    print(f'Retained tiles: {len(retained_tiles):,} across {len(sample_table):,} samples')
    display(input_audit.groupby(['input_type', 'status'], dropna=False).size().rename('n_files').reset_index())
    mapping = None
    tiles_with_individuals = None
    mapping = pd.read_csv(MAPPING_PATH, dtype='string').fillna('')
    tiles_with_individuals = sx.validate_individual_mapping(retained_tiles, mapping, require_approved=False)
    mapping_status = mapping.groupby('review_status').size().rename('n_samples').reset_index()
    display(mapping_status)
    display(mapping.loc[mapping['review_status'].ne('approved')])
    print(f"Proposed split groups: {mapping['individual_id'].nunique()}")
    target_review = pd.read_csv(TARGET_REVIEW_PATH, dtype='string').fillna('')
    observed_cell_types = []
    annotation_audit = None
    observed_cell_types, annotation_audit = sx.discover_annotation_labels(retained_tiles, ANNOTATION_ROOT)
    display(annotation_audit.groupby('status').size().rename('n_samples').reset_index())
    if observed_cell_types and target_review.empty:
        target_review = sx.target_review_template(observed_cell_types)
        sx._atomic_csv(target_review, TARGET_REVIEW_PATH)
        print(f'Wrote {len(target_review)} pending labels to {TARGET_REVIEW_PATH}')
    elif observed_cell_types:
        target_review = sx.validate_target_review(observed_cell_types, target_review, require_approved=False)
    elif not target_review.empty:
        print('Annotation files are not complete; existing target review was not changed.')
    display(target_review)
    tile_targets = None
    target_eligibility = None
    approved_targets = None
    coverage_audit = None
    outside_cells = None
    tiles_with_individuals = sx.validate_individual_mapping(retained_tiles, mapping)
    approved_targets = sx.validate_target_review(observed_cell_types, target_review)
    tile_targets, coverage_audit, outside_cells = sx.build_tile_targets(tiles_with_individuals, ANNOTATION_ROOT, approved_targets)
    if tile_targets is not None:
        reviewed_components = approved_targets.loc[approved_targets['include'], 'cell_type'].tolist()
        if TARGET_GROUPS:
            configured_components = list(dict.fromkeys((component for components in TARGET_GROUPS.values() for component in components)))
            if set(reviewed_components) != set(configured_components):
                raise ValueError(f'Included review rows {reviewed_components} must exactly match grouped components {configured_components}')
            for combined_name, components in TARGET_GROUPS.items():
                tile_targets = sx.add_combined_target(tile_targets, combined_name, components)
            included_cell_types = list(TARGET_GROUPS)
        else:
            included_cell_types = reviewed_components
        target_eligibility = sx.summarize_target_eligibility(tile_targets, included_cell_types, FULL_CONFIG)
        sx._atomic_csv(tile_targets, TILE_TARGETS_PATH)
        sx._atomic_csv(coverage_audit, TRAINING_ROOT / f'annotation_coverage_audit_{TARGET_STEM}.csv')
        sx._atomic_csv(outside_cells, TRAINING_ROOT / f'cells_outside_retained_tiles_{TARGET_STEM}.csv.gz')
        sx._atomic_csv(target_eligibility, TRAINING_ROOT / f'target_eligibility_{TARGET_STEM}.csv')
        sx._atomic_csv(target_eligibility.loc[~target_eligibility['eligible']], TRAINING_ROOT / f'excluded_targets_{TARGET_STEM}.csv')
        print(f'Wrote {len(tile_targets):,} tile targets to {TILE_TARGETS_PATH}')
        display(target_eligibility)
    if tile_targets is not None and TARGET_SETTINGS['prevalence']:
        target_column = f'proportion__{CELL_TYPE}'
        individual_prevalence = tile_targets.groupby('individual_id', sort=False, observed=True)[target_column].agg(n_tiles='size', n_positive_tiles=lambda values: int(values.gt(0).sum()), observed_min='min', observed_max='max', mean_proportion='mean').reset_index()
        individual_prevalence['positive_tile_fraction'] = individual_prevalence['n_positive_tiles'] / individual_prevalence['n_tiles']
        individual_prevalence['observed_range'] = individual_prevalence['observed_max'] - individual_prevalence['observed_min']
        individual_prevalence['all_zero'] = individual_prevalence['n_positive_tiles'].eq(0)
        sx._atomic_csv(individual_prevalence, TRAINING_ROOT / f"{TARGET_SETTINGS['prevalence']}_individual_prevalence.csv")
        print(f"Positive individuals: {(~individual_prevalence['all_zero']).sum():,} / {len(individual_prevalence):,}; all-zero individuals: {individual_prevalence['all_zero'].sum():,}")
        display(individual_prevalence.sort_values(['all_zero', 'positive_tile_fraction', 'observed_max'], ascending=[True, False, False]))
    return TILE_TARGETS_PATH

def compare_models(root):
    from dataclasses import fields
    import json
    import os
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from xenium_pipeline.workflow_output import display
    from xenium_pipeline import stpath_xgboost as sx
    PROJECT_ROOT = Path(root).expanduser().resolve()
    TRAINING_ROOT = PROJECT_ROOT / 'xgboost_training'
    MONITOR_OUTPUT_DIR = TRAINING_ROOT / 'fold_metric_monitor'
    from xenium_pipeline.storage import load_layout
    TARGET_RUNS = [dict(cell_type=spec['cell_type'], run_id=spec['run_id'],
        resnet50_run_id=spec['baseline_run_id'], tile_targets=TRAINING_ROOT / f"tile_targets_{spec['stem']}.csv.gz")
        for spec in load_layout(root)['targets'].values()]
    COMPARISON_RUNS = []
    for spec in TARGET_RUNS:
        COMPARISON_RUNS.append({**spec, 'models': list(sx.ALL_MODEL_NAMES)})
        COMPARISON_RUNS.append({**spec, 'run_id': spec['resnet50_run_id'], 'models': ['ResNet50']})
    display(pd.DataFrame(COMPARISON_RUNS))
    CRC_COMBINED_METRIC_FILES = {spec['cell_type']: None for spec in TARGET_RUNS}
    crc_candidates = sorted((TRAINING_ROOT / 'crc_fixed_selection_runs').glob('*/*/fold_metrics.csv'))
    for cell_type in CRC_COMBINED_METRIC_FILES:
        matches = [path for path in crc_candidates if path.parent.name == sx.safe_name(cell_type)]
        if len(matches) == 1 and CRC_COMBINED_METRIC_FILES[cell_type] is None:
            CRC_COMBINED_METRIC_FILES[cell_type] = matches[0]
    if crc_candidates:
        display(pd.DataFrame({'available_crc_metrics': [str(path) for path in crc_candidates]}))
    display(pd.DataFrame([{'cell_type': name, 'selected_crc_metrics': str(path) if path else 'set path above'} for name, path in CRC_COMBINED_METRIC_FILES.items()]))
    metric_tables = []
    summary_tables = []
    status_tables = []
    known_config_fields = {field.name for field in fields(sx.TrainingConfig)}
    for target_spec in COMPARISON_RUNS:
        cell_type = target_spec['cell_type']
        run_id = target_spec['run_id']
        tile_targets_path = Path(target_spec['tile_targets']).expanduser().resolve()
        run_root = TRAINING_ROOT / 'runs' / run_id
        run_metadata_path = run_root / 'run_metadata.json'
        checkpoint_root = run_root / sx.safe_name(cell_type) / 'checkpoints'
        unavailable = {'cell_type': cell_type, 'run_id': run_id, 'outer_fold': pd.NA, 'held_out_individual': '', 'n_test_rows': pd.NA, 'checkpoint_dir': str(checkpoint_root), 'completed_at_utc': '', 'target_parse_mode': '', 'n_completed_folds': 0, 'n_expected_folds': 0}
        if not tile_targets_path.is_file():
            status_tables.append(pd.DataFrame([{**unavailable, 'status': 'target_table_missing', 'error': str(tile_targets_path)}]))
            continue
        if not run_metadata_path.is_file():
            status_tables.append(pd.DataFrame([{**unavailable, 'status': 'run_not_started', 'error': str(run_metadata_path)}]))
            continue
        try:
            run_metadata = json.loads(run_metadata_path.read_text(encoding='utf-8'))
            config_payload = run_metadata.get('training_config', {})
            unknown_fields = sorted(set(config_payload).difference(known_config_fields))
            if unknown_fields:
                raise ValueError(f'Unknown training fields: {unknown_fields}')
            training_config = sx.TrainingConfig(**config_payload)
            tile_targets, target_parse_mode = sx.load_checkpoint_compatible_targets(tile_targets_path, cell_type, checkpoint_root, training_config)
            current_checksum = sx.tile_id_checksum(tile_targets['tile_id'].astype(str))
            if run_metadata.get('tile_id_checksum') != current_checksum:
                raise ValueError("Tile IDs do not match this run's metadata")
            target_column = f'proportion__{cell_type}'
            if target_column not in tile_targets:
                raise ValueError(f'Missing target column: {target_column}')
            fold_metrics, partial_predictions, fold_status = sx.collect_completed_fold_metrics(tile_targets, cell_type, checkpoint_root, training_config)
            fold_status.insert(0, 'run_id', run_id)
            fold_status.insert(0, 'cell_type', cell_type)
            fold_status.insert(2, 'target_parse_mode', target_parse_mode)
            status_tables.append(fold_status)
            fold_metrics = fold_metrics.loc[fold_metrics['model_type'].isin(target_spec['models'])].copy()
            if not fold_metrics.empty:
                fold_metrics.insert(1, 'run_id', run_id)
                metric_tables.append(fold_metrics)
                metric_summary = sx.summarize_evaluation_metrics(partial_predictions, fold_metrics)
                metric_summary.insert(1, 'run_id', run_id)
                summary_tables.append(metric_summary)
        except Exception as exc:
            status_tables.append(pd.DataFrame([{**unavailable, 'status': 'invalid', 'error': str(exc)}]))
    all_status = pd.concat(status_tables, ignore_index=True, sort=False)
    all_fold_metrics = pd.concat(metric_tables, ignore_index=True, sort=False) if metric_tables else pd.DataFrame()
    all_metric_summary = pd.concat(summary_tables, ignore_index=True, sort=False) if summary_tables else pd.DataFrame()
    MONITOR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    status_path = MONITOR_OUTPUT_DIR / 'all_cell_types_completed_fold_status.csv'
    metrics_path = MONITOR_OUTPUT_DIR / 'all_cell_types_fold_metrics_with_resnet50.csv'
    summary_path = MONITOR_OUTPUT_DIR / 'all_cell_types_metric_summary_with_resnet50.csv'
    sx._atomic_csv(all_status, status_path)
    sx._atomic_csv(all_fold_metrics, metrics_path)
    sx._atomic_csv(all_metric_summary, summary_path)
    progress = all_status.groupby(['cell_type', 'run_id'], sort=False, dropna=False).agg(completed_folds=('status', lambda values: int(values.eq('complete').sum())), skipped_folds=('status', lambda values: int(values.eq('skipped_all_zero').sum())), expected_folds=('n_expected_folds', 'max'), invalid_folds=('status', lambda values: int(values.eq('invalid').sum())), target_parse_modes=('target_parse_mode', lambda values: ', '.join(sorted(set(filter(None, map(str, values)))))), statuses=('status', lambda values: ', '.join(sorted(set(map(str, values)))))).reset_index()
    display(progress)
    if not all_fold_metrics.empty:
        display(all_fold_metrics.sort_values(['cell_type', 'outer_fold', 'prediction_scale', 'model_type']))
        display(all_metric_summary.sort_values(['cell_type', 'summary_scope', 'prediction_scale', 'mean_rmse']))
    if all_status['status'].eq('invalid').any():
        print('WARNING: one or more runs or committed folds failed validation.')
        display(all_status.loc[all_status['status'].eq('invalid')])
    print(f'Saved: {status_path}')
    print(f'Saved: {metrics_path}')
    print(f'Saved: {summary_path}')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.ticker import FormatStrFormatter
    from xenium_pipeline.model_comparison import COMPARISON_MODELS, CELL_TYPE_LABELS, matched_comparison_metrics
    model_order = list(COMPARISON_MODELS)
    model_colors = {'ResNet50': '#FAF0E3', 'Conch': '#C43E96', 'ProvGigapath': '#DEDBEE', 'UNI2h': '#06948E', 'Virchow': '#F3CDCC', 'Virchow2': '#F0CF7F', 'Combined': '#FF6B6B', 'Combined (CRC selection)': '#527F58'}
    selected_targets = {spec['cell_type'] for spec in TARGET_RUNS}
    cell_type_order = [name for name in CELL_TYPE_LABELS if name in selected_targets]
    cell_type_labels = [CELL_TYPE_LABELS[name] for name in cell_type_order]

    def plot_comparison(metrics, model_order, output_stem):
        from xenium_pipeline.model_comparison import plot_model_comparison
        return plot_model_comparison(metrics, MONITOR_OUTPUT_DIR, output_stem, models=model_order)
    if all_fold_metrics.empty:
        print('No completed outer folds are available for the figure yet.')
    else:
        fig, axes, legend_handles, figure_data = plot_comparison(all_fold_metrics, model_order, 'Figure_6A')
    correlation_columns = ['cell_type', 'run_id', 'outer_fold', 'held_out_individual', 'model_type', 'prediction_scale', 'n_test_tiles', 'n_positive_test_tiles', 'observed_min', 'observed_max', 'pearson', 'spearman', 'correlation_status']
    if all_fold_metrics.empty:
        individual_correlations = pd.DataFrame(columns=correlation_columns)
    else:
        individual_correlations = figure_data.loc[figure_data['prediction_scale'].eq('raw') & figure_data['model_type'].isin(model_order), correlation_columns].copy()
    individual_path = MONITOR_OUTPUT_DIR / 'individual_model_correlations_with_resnet50.csv'
    sx._atomic_csv(individual_correlations, individual_path)
    print(f'Saved: {individual_path}')
    if individual_correlations.empty:
        print('No completed outer-fold correlations are available yet.')
    else:
        duplicate_keys = ['cell_type', 'held_out_individual', 'model_type']
        if individual_correlations.duplicated(duplicate_keys).any():
            raise ValueError('Duplicate per-individual raw correlations found')
        individual_order = individual_correlations[['outer_fold', 'held_out_individual']].sort_values(['outer_fold', 'held_out_individual'], kind='stable').drop_duplicates('held_out_individual')['held_out_individual'].tolist()
        heatmap_cmap = plt.get_cmap('RdYlBu').copy()
        heatmap_cmap.set_bad('#FFFFFF')
        undefined_color = '#BDBDBD'
        unavailable_color = '#F7F7F7'
        for model_name in model_order:
            model_data = individual_correlations.loc[individual_correlations['model_type'].eq(model_name)]
            figure_height = max(8.0, 0.43 * len(individual_order))
            fig = plt.figure(figsize=(18, figure_height))
            grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045], left=0.18, right=0.94, top=0.9, bottom=0.12, wspace=0.12)
            pearson_ax = fig.add_subplot(grid[0, 0])
            spearman_ax = fig.add_subplot(grid[0, 1], sharey=pearson_ax)
            colorbar_ax = fig.add_subplot(grid[0, 2])
            axes = [pearson_ax, spearman_ax]
            image = None
            for ax, (metric, title) in zip(axes, [('pearson', 'Pearson correlation'), ('spearman', 'Spearman correlation')]):
                matrix = model_data.pivot(index='held_out_individual', columns='cell_type', values=metric).reindex(index=individual_order, columns=cell_type_order).apply(pd.to_numeric, errors='coerce')
                availability = model_data.assign(_present=True).pivot(index='held_out_individual', columns='cell_type', values='_present').reindex(index=individual_order, columns=cell_type_order, fill_value=False).fillna(False).astype(bool)
                values = matrix.to_numpy(dtype=float)
                present_values = availability.to_numpy(dtype=bool)
                image = ax.imshow(np.ma.masked_invalid(values), cmap=heatmap_cmap, vmin=-1, vmax=1, aspect='auto', interpolation='none')
                for row_index in range(values.shape[0]):
                    for column_index in range(values.shape[1]):
                        value = values[row_index, column_index]
                        present = present_values[row_index, column_index]
                        if np.isfinite(value):
                            label = f'{value:.2f}'
                        else:
                            color = undefined_color if present else unavailable_color
                            ax.add_patch(Rectangle((column_index - 0.5, row_index - 0.5), 1, 1, facecolor=color, edgecolor='white', linewidth=0.8))
                            label = 'U' if present else 'N/A'
                        ax.text(column_index, row_index, label, ha='center', va='center', fontsize=8, color='black')
                ax.set_title(title, fontweight='bold')
                ax.set_xticks(np.arange(len(cell_type_order)), cell_type_labels)
                ax.set_xlabel('Cell type')
                ax.tick_params(axis='x', rotation=0)
            pearson_ax.set_yticks(np.arange(len(individual_order)), individual_order)
            pearson_ax.set_ylabel('Held-out individual')
            spearman_ax.tick_params(axis='y', left=False, labelleft=False)
            fig.colorbar(image, cax=colorbar_ax, label='Correlation')
            state_handles = [Patch(facecolor=undefined_color, edgecolor='#888888', label='U = undefined'), Patch(facecolor=unavailable_color, edgecolor='#888888', label='N/A = unavailable')]
            fig.legend(handles=state_handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.035), frameon=False)
            fig.suptitle(f'Held-out individual correlations — {model_name}', fontweight='bold', y=0.97)
            fig.text(0.5, 0.01, 'Raw LOIO correlations from completed folds', ha='center', fontsize=9)
            model_png_path = MONITOR_OUTPUT_DIR / f'individual_correlations__{sx.safe_name(model_name)}__with_resnet50.png'
            model_pdf_path = MONITOR_OUTPUT_DIR / f'individual_correlations__{sx.safe_name(model_name)}__with_resnet50.pdf'
            fig.savefig(model_png_path, dpi=300, bbox_inches='tight', facecolor='white')
            fig.savefig(model_pdf_path, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f'Saved: {model_png_path}')
            print(f'Saved: {model_pdf_path}')
    distribution_colors = {'T Cells': '#4E79A7', 'Stromal Cells': '#59A14F', 'pan-APC + B': '#B07AA1', 'Tumor Cells': '#E15759', 'Normal Epithelial Cells': '#76B7B2'}
    for target_spec in TARGET_RUNS:
        cell_type = target_spec['cell_type']
        tile_targets_path = Path(target_spec['tile_targets']).expanduser().resolve()
        target_column = f'proportion__{cell_type}'
        if not tile_targets_path.is_file():
            print(f'Skipped {cell_type}: target table is unavailable at {tile_targets_path}')
            continue
        distribution_data = pd.read_csv(tile_targets_path, usecols=['dataset', 'sample', 'sample_order', target_column], float_precision='round_trip')
        distribution_data[target_column] = pd.to_numeric(distribution_data[target_column], errors='coerce')
        if distribution_data[target_column].isna().any():
            raise ValueError(f'{tile_targets_path} contains missing or non-numeric target values')
        if not distribution_data[target_column].between(0, 1).all():
            raise ValueError(f'{tile_targets_path} contains target values outside [0, 1]')
        sample_table = distribution_data[['dataset', 'sample', 'sample_order']].drop_duplicates().sort_values(['sample_order', 'dataset', 'sample'], kind='stable').reset_index(drop=True)
        sample_labels = (sample_table['dataset'].astype(str) + ' / ' + sample_table['sample'].astype(str)).tolist()
        figure_height = max(8.0, 0.36 * len(sample_table))
        fig, ax = plt.subplots(figsize=(14, figure_height))
        for position, sample_row in enumerate(sample_table.itertuples(index=False)):
            values = distribution_data.loc[distribution_data['dataset'].eq(sample_row.dataset) & distribution_data['sample'].eq(sample_row.sample), target_column]
            zero_percentage = float(values.eq(0).mean() * 100)
            numeric_values = values.to_numpy(dtype=float)
            if np.ptp(numeric_values) > 0:
                violin = ax.violinplot([numeric_values], positions=[position], vert=False, widths=0.72, points=100, showmeans=False, showmedians=False, showextrema=False)
                for body in violin['bodies']:
                    body.set_facecolor(distribution_colors[cell_type])
                    body.set_edgecolor('black')
                    body.set_linewidth(0.7)
                    body.set_alpha(0.8)
            else:
                ax.scatter(numeric_values[0], position, marker='D', s=32, color=distribution_colors[cell_type], edgecolor='black', zorder=3)
            q25, median, q75 = np.quantile(numeric_values, [0.25, 0.5, 0.75])
            ax.hlines(position, q25, q75, color='black', linewidth=2.4, zorder=4)
            ax.scatter(median, position, s=22, color='white', edgecolor='black', zorder=5)
            ax.text(1.015, position, f'{zero_percentage:.1f}%', transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=8, clip_on=False)
        ax.set_yticks(np.arange(len(sample_table)), sample_labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel('Tile proportion', fontweight='bold')
        ax.set_ylabel('Dataset / sample', fontweight='bold')
        ax.set_title(f'Tile-proportion distributions — {cell_type}', fontweight='bold')
        ax.grid(axis='x', linestyle='--', linewidth=0.8, alpha=0.4)
        ax.text(1.015, 1.01, 'Zero tiles', transform=ax.transAxes, ha='left', va='bottom', fontsize=9, fontweight='bold')
        fig.subplots_adjust(left=0.34, right=0.88, top=0.94, bottom=0.08)
        distribution_png_path = MONITOR_OUTPUT_DIR / f'tile_proportion_distribution__{sx.safe_name(cell_type)}.png'
        distribution_pdf_path = MONITOR_OUTPUT_DIR / f'tile_proportion_distribution__{sx.safe_name(cell_type)}.pdf'
        fig.savefig(distribution_png_path, dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(distribution_pdf_path, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'Saved: {distribution_png_path}')
        print(f'Saved: {distribution_pdf_path}')
    crc_comparison_models = [*COMPARISON_MODELS, 'Combined (CRC selection)']
    missing_paths = [name for name, path in CRC_COMBINED_METRIC_FILES.items() if path is None]
    if missing_paths:
        print('Set CRC_COMBINED_METRIC_FILES for: ' + ', '.join(missing_paths))
    else:
        crc_tables = []
        for cell_type in cell_type_order:
            path = Path(CRC_COMBINED_METRIC_FILES[cell_type]).expanduser().resolve()
            table = pd.read_csv(path, float_precision='round_trip')
            selected = table.loc[table.model_type.eq('Combined') & table.prediction_scale.eq('raw')].copy()
            if selected.empty or set(selected.cell_type) != {cell_type}:
                raise ValueError(f'Missing or mismatched raw Combined target in {path}')
            selected['model_type'] = 'Combined (CRC selection)'
            selected['run_id'] = path.parent.parent.name
            selected['source_metrics_path'] = str(path)
            crc_tables.append(selected)
        crc_metrics = pd.concat([all_fold_metrics, *crc_tables], ignore_index=True, sort=False)
        crc_matched = matched_comparison_metrics(crc_metrics, cell_type_order, models=crc_comparison_models)
        crc_matched.to_csv(MONITOR_OUTPUT_DIR / 'all_cell_types_fold_metrics_with_crc_selection.csv', index=False)
        crc_fig, crc_axes, crc_handles, crc_figure_data = plot_comparison(crc_matched, crc_comparison_models, 'saishi_style_all_cell_types_with_resnet50_crc_selection')
        caption_path = MONITOR_OUTPUT_DIR / 'saishi_style_all_cell_types_with_resnet50_crc_selection_caption.txt'
        caption_path.write_text(caption_path.read_text() + 'Combined (CRC selection) uses fixed features selected from mean single-encoder LOIO gains; feature selection is not fully nested.\n')
    return MONITOR_OUTPUT_DIR

def extract_features(root, *, device='auto', models=None, batch_size=8, workers=0):
    from xenium_pipeline import feature_extraction as features
    root = Path(root).expanduser().resolve()
    if device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    tiles = features.load_selected_tile_index(root, root / 'tile_filter_cutoff_tests/final_automatic_tile_selection_manifest.csv')
    encoders = models or ['UNI2h', 'Virchow', 'Virchow2', 'ProvGigapath', 'Conch', 'ResNet50']
    from xenium_pipeline.storage import data_path
    output = data_path(root, 'feature_root')
    summary = features.run_extraction(tiles, encoders, output, overwrite=False, device=device, batch_size=batch_size, num_workers=workers, recipe_version='stpath_brca_training_compatible_v1')
    validation = features.validate_outputs(tiles, encoders, output)
    if validation.empty or not validation['valid'].all():
        raise RuntimeError(f'Encoder extraction is incomplete; see {output}')
    return validation

def train_target(root, target, extra_args=()):
    from xenium_pipeline.train_stpath_xgboost import main
    spec = target_settings(root, target)
    if '--resnet50-only' in extra_args:
        from xenium_pipeline.storage import load_layout
        binding = load_layout(root)['targets'][target]
        reference = spec['output_root']
        spec = dict(spec, run_id=binding['baseline_run_id'])
        extra_args = ['--reference-run-dir', str(reference), *extra_args]
    from xenium_pipeline.stpath_xgboost import available_cpu_count
    threads = str(min(5, available_cpu_count() or 1))
    arguments = ['--project-root', str(Path(root).resolve()), '--cell-type', spec['cell_type'], '--target-table', str(spec['target_table']), '--run-id', spec['run_id'], '--parallel-model-jobs', '1', '--checkpoint-nthread', '5', '--nthread', threads, '--combined-nthread', threads]
    if not spec['skip_all_zero']:
        arguments.append('--include-all-zero-test-folds')
    return main([*arguments, *extra_args])
