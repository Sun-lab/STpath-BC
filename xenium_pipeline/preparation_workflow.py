from __future__ import annotations
from pathlib import Path
import os
CODE_ROOT = Path(__file__).resolve().parents[1]

def select_tiles(root):
    from pathlib import Path
    import sys, importlib
    import pandas as pd
    from xenium_pipeline.workflow_output import HTML, Image as DisplayImage, Markdown, display
    ROOT = Path(root).expanduser().resolve()
    from xenium_pipeline.dataset_config import dataset_configs, dataset_output_dirs
    import xenium_pipeline.standard_xenium as shared_pipeline
    import xenium_pipeline.tile_filter_review as tile_filter_review
    pd.set_option('display.max_columns', 60)
    pd.set_option('display.max_colwidth', 140)
    DATASET_CONFIGS = dataset_configs()
    DATASET_OUTPUTS = tile_filter_review.default_dataset_outputs(ROOT)
    TILE_FILTER_OUTPUT_DIR = ROOT / 'tile_filter_cutoff_tests'
    samples = tile_filter_review.discover_tile_filter_samples(ROOT, dataset_outputs=DATASET_OUTPUTS)
    if not samples:
        raise FileNotFoundError('No registered tile manifests found.')
    TILE_FILTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory = tile_filter_review.inventory_table(samples)
    inventory['sample_key'] = inventory['dataset'].astype(str) + '/' + inventory['sample'].astype(str)
    inventory.to_csv(TILE_FILTER_OUTPUT_DIR / 'tile_filter_inventory.csv', index=False)
    samples_for_filtering = samples
    filter_inventory = inventory.copy()
    RUN_AREA_RECALCULATION = True
    AREA_RECALCULATION_KINDS = ['cell', 'nucleus']
    AREA_RECALCULATION_DATASETS = None
    AREA_RECALCULATION_SAMPLE_KEYS = None
    AREA_RECALCULATION_OUTPUT_ROOT = TILE_FILTER_OUTPUT_DIR / 'recalculated_area_manifests'

    def area_recalculation_sample_key(sample):
        return f'{sample.dataset}/{sample.sample}'

    def select_area_recalculation_samples(samples):
        selected = list(samples)
        available_datasets = {sample.dataset for sample in selected}
        if AREA_RECALCULATION_DATASETS is not None:
            requested_datasets = set(AREA_RECALCULATION_DATASETS)
            missing_datasets = sorted(requested_datasets - available_datasets)
            if missing_datasets:
                raise ValueError(f'Unknown AREA_RECALCULATION_DATASETS: {missing_datasets}')
            selected = [sample for sample in selected if sample.dataset in requested_datasets]
        if AREA_RECALCULATION_SAMPLE_KEYS is not None:
            requested_keys = set(AREA_RECALCULATION_SAMPLE_KEYS)
            available_keys = {area_recalculation_sample_key(sample) for sample in samples}
            missing_keys = sorted(requested_keys - available_keys)
            if missing_keys:
                raise ValueError(f'Unknown AREA_RECALCULATION_SAMPLE_KEYS: {missing_keys}')
            selected = [sample for sample in selected if area_recalculation_sample_key(sample) in requested_keys]
            excluded_by_dataset = sorted(requested_keys - {area_recalculation_sample_key(sample) for sample in selected})
            if excluded_by_dataset:
                raise ValueError(f'AREA_RECALCULATION_SAMPLE_KEYS contains sample(s) outside AREA_RECALCULATION_DATASETS: {excluded_by_dataset}')
        return selected
    samples_for_filtering = samples
    filter_inventory = inventory.copy()
    area_recalculation_samples = select_area_recalculation_samples(samples)
    if not area_recalculation_samples:
        raise ValueError('No samples selected for area recalculation.')
    area_recalculation_selection = pd.DataFrame(({'dataset': sample.dataset, 'sample': sample.sample, 'sample_key': area_recalculation_sample_key(sample)} for sample in area_recalculation_samples))
    display(Markdown('### Area recalculation selection'))
    display(area_recalculation_selection.groupby('dataset', as_index=False).agg(samples=('sample', 'size')))
    area_recalculation_summary, recalculated_subset = tile_filter_review.recalculate_boundary_area_metrics_for_samples(area_recalculation_samples, output_root=AREA_RECALCULATION_OUTPUT_ROOT, area_kinds=AREA_RECALCULATION_KINDS)
    TILE_FILTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    area_recalculation_summary_path = TILE_FILTER_OUTPUT_DIR / 'recalculated_area_summary.csv'
    area_recalculation_summary.to_csv(area_recalculation_summary_path, index=False)
    recalculated_by_key = {area_recalculation_sample_key(sample): sample for sample in recalculated_subset}
    samples_for_filtering = [recalculated_by_key.get(area_recalculation_sample_key(sample), sample) for sample in samples]
    filter_inventory = tile_filter_review.inventory_table(samples_for_filtering)
    filter_inventory['sample_key'] = filter_inventory['dataset'].astype(str) + '/' + filter_inventory['sample'].astype(str)
    filter_inventory_path = TILE_FILTER_OUTPUT_DIR / 'tile_filter_active_inventory.csv'
    filter_inventory.to_csv(filter_inventory_path, index=False)
    print(f'Area recalculation summary saved to: {area_recalculation_summary_path}')
    print(f'Active inventory saved to: {filter_inventory_path}')
    display(area_recalculation_summary)
    display(filter_inventory)
    ACTIVE_TILE_MANIFEST_SOURCE = 'recalculated'
    ACTIVE_TILE_MANIFEST_SOURCE_BY_DATASET = {}
    ACTIVE_TILE_MANIFEST_SOURCE_BY_SAMPLE_KEY = {}
    ACTIVE_TILE_MANIFEST_STRICT = True
    RECALCULATED_AREA_MANIFEST_ROOT = TILE_FILTER_OUTPUT_DIR / 'recalculated_area_manifests'
    samples_for_filtering, active_manifest_summary = tile_filter_review.select_active_tile_filter_samples(samples, manifest_source=ACTIVE_TILE_MANIFEST_SOURCE, recalculated_root=RECALCULATED_AREA_MANIFEST_ROOT, source_by_dataset=ACTIVE_TILE_MANIFEST_SOURCE_BY_DATASET, source_by_sample_key=ACTIVE_TILE_MANIFEST_SOURCE_BY_SAMPLE_KEY, strict=ACTIVE_TILE_MANIFEST_STRICT)
    RUN_HE_TISSUE_FRACTION_CALCULATION = True
    OVERWRITE_HE_TISSUE_FRACTIONS = False
    HE_TISSUE_MIN_MASK_PIXELS_PER_TILE = 16
    HE_TISSUE_FRACTION_MANIFEST_ROOT = TILE_FILTER_OUTPUT_DIR / 'he_tissue_fraction_manifests'
    he_tissue_summary, samples_for_filtering = tile_filter_review.calculate_he_tissue_fraction_for_samples(samples_for_filtering, output_root=HE_TISSUE_FRACTION_MANIFEST_ROOT, overwrite=OVERWRITE_HE_TISSUE_FRACTIONS, min_mask_pixels_per_tile=HE_TISSUE_MIN_MASK_PIXELS_PER_TILE)
    he_tissue_summary_path = TILE_FILTER_OUTPUT_DIR / 'he_tissue_fraction_calculation_summary.csv'
    he_tissue_summary.to_csv(he_tissue_summary_path, index=False)
    he_tissue_failures = he_tissue_summary.loc[he_tissue_summary['status'].eq('error')]
    if not he_tissue_failures.empty:
        display(he_tissue_failures)
        raise RuntimeError('H&E tissue-fraction calculation failed for one or more samples.')
    print(f'H&E tissue-fraction summary saved to: {he_tissue_summary_path}')
    display(he_tissue_summary[['dataset', 'sample', 'status', 'n_tiles', 'n_tiles_with_he_tissue_fraction', 'he_tissue_mask_method', 'he_tissue_darkness_otsu_threshold', 'he_tissue_chroma_otsu_threshold', 'he_tissue_mask_level', 'he_tissue_mask_pixels_per_tile', 'median_he_tissue_fraction', 'error']])
    filter_inventory = tile_filter_review.inventory_table(samples_for_filtering)
    filter_inventory['sample_key'] = filter_inventory['dataset'].astype(str) + '/' + filter_inventory['sample'].astype(str)
    filter_inventory_path = TILE_FILTER_OUTPUT_DIR / 'tile_filter_active_inventory.csv'
    TILE_FILTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filter_inventory.to_csv(filter_inventory_path, index=False)
    tile_table = tile_filter_review.load_combined_tile_filter_table(samples_for_filtering)
    print(f'{len(tile_table):,} nonempty tile rows loaded from {len(samples_for_filtering):,} active samples.')
    print(f'Active manifest inventory saved to: {filter_inventory_path}')
    TILE_FILTER_AREA_FRACTION_COLUMN = 'recalculated_cell_area_fraction'
    TILE_FILTER_AREA_FRACTION_COLUMN_BY_DATASET = {}
    TILE_FILTER_AREA_FRACTION_COLUMN_BY_SAMPLE_KEY = {}
    threshold_tile_table = tile_filter_review.combined_table_with_selected_area_fraction(tile_table, default=TILE_FILTER_AREA_FRACTION_COLUMN, by_dataset=TILE_FILTER_AREA_FRACTION_COLUMN_BY_DATASET, by_sample_key=TILE_FILTER_AREA_FRACTION_COLUMN_BY_SAMPLE_KEY)
    AUTOMATIC_CUTOFF_N_MADS = 3.0
    AUTOMATIC_CUTOFF_MIN_FINITE = 20
    AUTOMATIC_MANUAL_FALLBACKS = {'n_cells': 1, 'filter_area_fraction': 0.0, 'nucleus_area_fraction': 0.0, 'he_tissue_fraction': 0.6}
    APPLY_AUTOMATIC_NUCLEUS_AREA_CUTOFF = False
    RUN_AUTOMATIC_FILTERING = True
    AUTOMATIC_SELECT_DATASETS = None
    AUTOMATIC_SELECT_SAMPLE_KEYS = None
    SAVE_AUTOMATIC_QUADRANT_EXAMPLE_PLOTS = False
    DRAW_CELL_OVERLAYS_ON_AUTOMATIC_EXAMPLES = True
    MATERIALIZE_AUTOMATIC_TILES = True
    AUTOMATIC_MATERIALIZE_MODE = 'symlink'
    AUTOMATIC_MATERIALIZED_ROOT = TILE_FILTER_OUTPUT_DIR / 'selected_tiles_data_adaptive'
    REQUIRE_HE_TISSUE_FOR_AUTOMATIC_CUTOFF = True
    adaptive_cutoffs, adaptive_cutoff_diagnostics = tile_filter_review.estimate_data_adaptive_cutoffs(threshold_tile_table, n_mads=AUTOMATIC_CUTOFF_N_MADS, min_finite=AUTOMATIC_CUTOFF_MIN_FINITE, manual_fallbacks=AUTOMATIC_MANUAL_FALLBACKS, apply_nucleus_area_fraction=APPLY_AUTOMATIC_NUCLEUS_AREA_CUTOFF)
    adaptive_cutoffs_path = TILE_FILTER_OUTPUT_DIR / 'data_adaptive_cutoff_recommendations.csv'
    adaptive_diagnostics_path = TILE_FILTER_OUTPUT_DIR / 'data_adaptive_cutoff_diagnostics.csv'
    adaptive_cutoffs.to_csv(adaptive_cutoffs_path, index=False)
    adaptive_cutoff_diagnostics.to_csv(adaptive_diagnostics_path, index=False)
    automatic_samples = list(samples_for_filtering)
    if AUTOMATIC_SELECT_DATASETS is not None:
        wanted_datasets = set(AUTOMATIC_SELECT_DATASETS)
        automatic_samples = [sample for sample in automatic_samples if sample.dataset in wanted_datasets]
    if AUTOMATIC_SELECT_SAMPLE_KEYS is not None:
        wanted_keys = set(AUTOMATIC_SELECT_SAMPLE_KEYS)
        automatic_samples = [sample for sample in automatic_samples if tile_filter_review.tile_filter_sample_key(sample) in wanted_keys]
    if not automatic_samples:
        raise ValueError('No samples match the automatic-filter dataset/sample selection.')
    automatic_keys = {tile_filter_review.tile_filter_sample_key(sample) for sample in automatic_samples}
    selected_adaptive_cutoffs = adaptive_cutoffs.loc[adaptive_cutoffs['sample_key'].isin(automatic_keys)].copy()
    adaptive_filter_summary = tile_filter_review.save_adaptive_cutoff_outputs(automatic_samples, selected_adaptive_cutoffs, output_root=TILE_FILTER_OUTPUT_DIR, tile_size_px=None, area_fraction_column=TILE_FILTER_AREA_FRACTION_COLUMN, area_fraction_column_by_dataset=TILE_FILTER_AREA_FRACTION_COLUMN_BY_DATASET, area_fraction_column_by_sample_key=TILE_FILTER_AREA_FRACTION_COLUMN_BY_SAMPLE_KEY, materialize_tiles=MATERIALIZE_AUTOMATIC_TILES, materialize_mode=AUTOMATIC_MATERIALIZE_MODE, materialized_root=AUTOMATIC_MATERIALIZED_ROOT, examples_per_quadrant=3, max_scatter_points=50000, save_quadrant_examples=SAVE_AUTOMATIC_QUADRANT_EXAMPLE_PLOTS, draw_example_overlays=DRAW_CELL_OVERLAYS_ON_AUTOMATIC_EXAMPLES, require_nucleus_area_fraction=False, require_he_tissue_fraction=REQUIRE_HE_TISSUE_FOR_AUTOMATIC_CUTOFF)
    display(adaptive_filter_summary)
    print(f"Adaptive filter summary: {TILE_FILTER_OUTPUT_DIR / 'all_adaptive_cutoff_summary.csv'}")
    return write_selection_manifest(ROOT, samples)

def write_selection_manifest(root, samples):
    import pandas as pd
    from xenium_pipeline import tile_filter_review
    from xenium_pipeline.workflow_output import display, FileLink
    TILE_FILTER_OUTPUT_DIR = Path(root) / 'tile_filter_cutoff_tests'
    adaptive_summary_path = TILE_FILTER_OUTPUT_DIR / 'all_adaptive_cutoff_summary.csv'
    final_selection_manifest_path = TILE_FILTER_OUTPUT_DIR / 'final_automatic_tile_selection_manifest.csv'
    if not adaptive_summary_path.is_file():
        raise FileNotFoundError(f'Missing automatic cutoff summary: {adaptive_summary_path}. Run tile selection for all samples before creating the manifest.')
    saved_adaptive_summary = pd.read_csv(adaptive_summary_path)
    required_final_columns = {'dataset', 'sample', 'area_fraction_filter_column', 'min_cells_per_tile', 'min_cell_area_fraction', 'min_nucleus_area_fraction', 'min_he_tissue_fraction', 'n_nonempty_tiles_in_manifest', 'n_passing_nonempty_tiles', 'n_materialized_tiles', 'n_missing_tile_files', 'materialized_tile_dir', 'materialized_tile_manifest', 'passing_tile_ids_path'}
    missing_final_columns = required_final_columns - set(saved_adaptive_summary.columns)
    if missing_final_columns:
        raise ValueError(f'Automatic cutoff summary is missing final-manifest columns: {sorted(missing_final_columns)}')
    if saved_adaptive_summary.empty:
        raise ValueError('Automatic cutoff summary is empty.')
    saved_adaptive_summary = saved_adaptive_summary.copy()
    saved_adaptive_summary['sample_key'] = saved_adaptive_summary['dataset'].astype(str) + '/' + saved_adaptive_summary['sample'].astype(str)
    duplicate_final_samples = saved_adaptive_summary.loc[saved_adaptive_summary['sample_key'].duplicated(keep=False), 'sample_key'].tolist()
    if duplicate_final_samples:
        raise ValueError(f'Duplicate samples in automatic cutoff summary: {duplicate_final_samples}')
    expected_sample_keys = {tile_filter_review.tile_filter_sample_key(sample) for sample in samples}
    observed_sample_keys = set(saved_adaptive_summary['sample_key'])
    missing_final_samples = sorted(expected_sample_keys - observed_sample_keys)
    unexpected_final_samples = sorted(observed_sample_keys - expected_sample_keys)
    if missing_final_samples or unexpected_final_samples:
        raise ValueError(f'Automatic cutoff summary does not match the discovered samples. Missing: {missing_final_samples}; unexpected: {unexpected_final_samples}')
    numeric_final_columns = ['n_nonempty_tiles_in_manifest', 'n_passing_nonempty_tiles', 'n_materialized_tiles', 'n_missing_tile_files']
    for column in numeric_final_columns:
        saved_adaptive_summary[column] = pd.to_numeric(saved_adaptive_summary[column], errors='raise').astype(int)
    count_mismatches = saved_adaptive_summary.loc[saved_adaptive_summary['n_passing_nonempty_tiles'].ne(saved_adaptive_summary['n_materialized_tiles'])]
    if not count_mismatches.empty:
        display(count_mismatches[['dataset', 'sample', 'n_passing_nonempty_tiles', 'n_materialized_tiles']])
        raise ValueError('Selected and materialized tile counts do not match.')
    missing_tile_rows = saved_adaptive_summary.loc[saved_adaptive_summary['n_missing_tile_files'].ne(0)]
    if not missing_tile_rows.empty:
        display(missing_tile_rows[['dataset', 'sample', 'n_missing_tile_files']])
        raise ValueError('At least one materialized tile source file was missing.')
    path_columns = ['materialized_tile_dir', 'materialized_tile_manifest', 'passing_tile_ids_path']
    for column in path_columns:
        saved_adaptive_summary[column] = saved_adaptive_summary[column].fillna('').astype(str)
    blank_path_rows = saved_adaptive_summary.loc[saved_adaptive_summary[path_columns].eq('').any(axis=1)]
    if not blank_path_rows.empty:
        display(blank_path_rows[['dataset', 'sample', *path_columns]])
        raise ValueError('Final materialization paths are blank for one or more samples.')
    missing_directories = saved_adaptive_summary.loc[~saved_adaptive_summary['materialized_tile_dir'].map(lambda value: Path(value).is_dir())]
    missing_manifests = saved_adaptive_summary.loc[~saved_adaptive_summary['materialized_tile_manifest'].map(lambda value: Path(value).is_file())]
    missing_passing_tile_lists = saved_adaptive_summary.loc[~saved_adaptive_summary['passing_tile_ids_path'].map(lambda value: Path(value).is_file())]
    if not missing_directories.empty or not missing_manifests.empty or (not missing_passing_tile_lists.empty):
        display(pd.concat([missing_directories[['dataset', 'sample', 'materialized_tile_dir']], missing_manifests[['dataset', 'sample', 'materialized_tile_manifest']], missing_passing_tile_lists[['dataset', 'sample', 'passing_tile_ids_path']]], ignore_index=True, sort=False))
        raise FileNotFoundError('At least one saved tile directory, materialization manifest, or passing-tile CSV is missing.')
    final_selection_manifest = saved_adaptive_summary[['dataset', 'sample', 'sample_key', 'area_fraction_filter_column', 'min_cells_per_tile', 'min_cell_area_fraction', 'min_nucleus_area_fraction', 'min_he_tissue_fraction', 'n_nonempty_tiles_in_manifest', 'n_materialized_tiles', 'materialized_tile_dir', 'materialized_tile_manifest', 'passing_tile_ids_path']].rename(columns={'n_nonempty_tiles_in_manifest': 'n_nonempty_tiles', 'n_materialized_tiles': 'n_final_selected_tiles'})
    final_selection_manifest['retained_tile_fraction'] = final_selection_manifest['n_final_selected_tiles'] / final_selection_manifest['n_nonempty_tiles']
    final_selection_manifest = final_selection_manifest[['dataset', 'sample', 'sample_key', 'area_fraction_filter_column', 'min_cells_per_tile', 'min_cell_area_fraction', 'min_nucleus_area_fraction', 'min_he_tissue_fraction', 'n_nonempty_tiles', 'n_final_selected_tiles', 'retained_tile_fraction', 'materialized_tile_dir', 'materialized_tile_manifest', 'passing_tile_ids_path']].sort_values(['dataset', 'sample']).reset_index(drop=True)
    final_selection_manifest.to_csv(final_selection_manifest_path, index=False)
    print(f'Final automatic tile-selection manifest saved to: {final_selection_manifest_path}')
    print(f"{len(final_selection_manifest):,} samples; {final_selection_manifest['n_nonempty_tiles'].sum():,} nonempty tiles; {final_selection_manifest['n_final_selected_tiles'].sum():,} final tiles.")
    display(final_selection_manifest)
    return final_selection_manifest

def prepare_markers(root):
    from collections import defaultdict
    from itertools import combinations
    from pathlib import Path
    import importlib
    import json
    import re
    import sys
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from xenium_pipeline.workflow_output import Image, display
    ROOT = Path(root).expanduser().resolve()
    from xenium_pipeline import dataset_config, gene_panel_utils
    pd.set_option('display.max_columns', 120)
    pd.set_option('display.max_colwidth', 180)

    def show_saved_plot(path: Path) -> None:
        """Display a saved PNG reliably across notebook frontends."""
        display(Image(filename=str(path)))
    print(f'Root directory: {ROOT}')
    INCLUDE_BILOUS = True
    DATASET_NAMES = None
    OUTPUT_DIR = ROOT / 'gene_panel_review'
    WRITE_OUTPUTS = True
    configs = dataset_config.dataset_configs()
    selected_dataset_names = list(DATASET_NAMES or configs.keys())
    selected_configs = {name: configs[name] for name in selected_dataset_names}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_table = pd.DataFrame([{'dataset': name, 'display_name': config.display_name, 'root': str(config.root_path(ROOT)), 'pipeline': config.pipeline, 'cell_labels': config.cell_labels} for name, config in selected_configs.items()])
    display(dataset_table)
    print(f'Review output directory: {OUTPUT_DIR}')

    def natural_key(value: object) -> tuple[object, ...]:
        return gene_panel_utils.natural_key(value)

    def normalize_name(value: object) -> str:
        return re.sub('[^a-z0-9]+', '', str(value).lower())

    def root_panel_files(root: Path) -> list[Path]:
        return sorted(root.glob('*gene_panel.json'), key=lambda path: natural_key(path.name))

    def root_gene_group_files(root: Path) -> list[Path]:
        patterns = ['*gene_groups.csv', '*Input_gene_groups.csv']
        files: list[Path] = []
        for pattern in patterns:
            files.extend(sorted(root.glob(pattern), key=lambda path: natural_key(path.name)))
        return list(dict.fromkeys(files))

    def unique_paths(paths: list[Path]) -> list[Path]:
        return list(dict.fromkeys(paths))

    def feature_matrix_files_for_sample(dataset: str, root: Path, sample_dir: Path, sample_name: str) -> list[Path]:
        candidate_dirs = [sample_dir]
        if dataset == 'Bilous_2026':
            root_sample_dir = root / sample_name
            if root_sample_dir.is_dir():
                candidate_dirs.append(root_sample_dir)
            gsm_match = re.search('(GSM\\d+)', sample_name)
            if gsm_match:
                gsm = gsm_match.group(1)
                gsm_dir = root / gsm
                if gsm_dir.is_dir():
                    candidate_dirs.append(gsm_dir)
                candidate_dirs.extend((path for path in sorted(root.glob(f'*{gsm}*'), key=lambda path: natural_key(path.name)) if path.is_dir()))
        matrix_paths: list[Path] = []
        for directory in unique_paths(candidate_dirs):
            if not directory.is_dir():
                continue
            matrix_paths.extend(sorted(directory.rglob('cell_feature_matrix.zarr.zip'), key=lambda path: natural_key(path.name)))
            matrix_paths.extend(sorted(directory.rglob('*cell_feature_matrix.h5'), key=lambda path: natural_key(path.name)))
            matrix_paths.extend(sorted(directory.rglob('cell_feature_matrix.h5'), key=lambda path: natural_key(path.name)))
        return unique_paths(matrix_paths)

    def feature_matrix_genes(path: Path) -> pd.DataFrame:
        features, error = gene_panel_utils.feature_metadata(path)
        if error:
            raise ValueError(error)
        if features.empty:
            return features
        features = features.copy()
        features['gene_upper'] = features['gene_upper'].astype(str).str.upper()
        features = features.loc[features['gene_upper'].ne('')].copy()
        if 'is_gene_expression' in features.columns:
            features = features.loc[features['is_gene_expression'].fillna(False)].copy()
        return features

    def panel_genes(path: Path) -> pd.DataFrame:
        panel = gene_panel_utils.read_gene_panel_json(path)
        if panel.empty:
            return panel
        panel = panel.copy()
        panel['gene_upper'] = panel['gene_upper'].astype(str).str.upper()
        panel = panel.loc[panel['gene_upper'].ne('')].copy()
        if 'target_descriptor' in panel.columns:
            is_gene = panel['target_descriptor'].fillna('').astype(str).str.lower().isin(['', 'gene'])
            panel = panel.loc[is_gene].copy()
        return panel

    def marker_group_rows(path: Path) -> pd.DataFrame:
        groups = gene_panel_utils.load_gene_groups(path)
        groups = groups.copy()
        groups['gene_upper'] = groups['gene_upper'].astype(str).str.upper()
        return groups

    def files_have_same_gene_set(files: list[Path], loader) -> bool:
        if len(files) <= 1:
            return True
        sets = []
        for path in files:
            try:
                table = loader(path)
                sets.append(set(table['gene_upper'].dropna().astype(str)))
            except Exception:
                return False
        return all((item == sets[0] for item in sets[1:]))

    def resolve_shared_10x_file(sample_name: str, files: list[Path]) -> list[Path]:
        if not files:
            return []
        is_addon_sample = 'with_addon' in sample_name.lower()
        addon = [path for path in files if 'addon' in path.name.lower()]
        base = [path for path in files if 'addon' not in path.name.lower()]
        if is_addon_sample and addon:
            return [addon[0]]
        if not is_addon_sample and base:
            return [base[0]]
        return []

    def resolve_sample_files(*, dataset: str, root: Path, sample_dir: Path, root_files: list[Path], suffix: str, same_set_loader) -> tuple[list[Path], str]:
        sample_name = sample_dir.name
        sample_norm = normalize_name(sample_name)
        exact = root / f'{sample_name}{suffix}'
        if exact.is_file():
            return ([exact], 'root_exact_sample_name')
        in_sample = sorted(sample_dir.rglob(f'*{suffix}'), key=lambda path: natural_key(path.name))
        if in_sample:
            return (in_sample, 'sample_folder')
        matching = [path for path in root_files if sample_norm and sample_norm in normalize_name(path.stem)]
        if matching:
            return (matching, 'root_name_contains_sample')
        if dataset == '10x':
            selected = resolve_shared_10x_file(sample_name, root_files)
            if selected:
                return (selected, 'root_10x_base_addon_rule')
        if not root_files:
            return ([], 'no_root_files')
        if len(root_files) == 1:
            return ([root_files[0]], 'single_root_file_shared_by_dataset')
        if files_have_same_gene_set(root_files, same_set_loader):
            return (root_files[:1], 'root_files_equivalent_gene_sets')
        return ([], 'unresolved')

    def discover_dataset_samples(config) -> list[Path]:
        root = config.root_path(ROOT)
        if not root.is_dir():
            return []
        if config.name == 'Bilous_2026':
            root_sample_dirs = [path for path in sorted(root.iterdir(), key=lambda path: natural_key(path.name)) if path.is_dir() and re.match('^B[\\w_]*_GSM\\d+$', path.name)]
            if root_sample_dirs:
                return root_sample_dirs
            pipeline_summary = config.output_path(ROOT) / 'pipeline_summary.csv'
            if pipeline_summary.is_file():
                table = pd.read_csv(pipeline_summary, dtype=str).fillna('')
                if 'sample' in table.columns:
                    samples = [sample for sample in table['sample'].drop_duplicates().tolist() if sample]
                    return [root / sample if (root / sample).is_dir() else config.output_path(ROOT) / sample for sample in sorted(samples, key=natural_key)]
        return gene_panel_utils.discover_sample_dirs(root)
    summary_rows: list[dict[str, object]] = []
    panel_gene_rows: list[dict[str, object]] = []
    marker_rows: list[dict[str, object]] = []
    read_error_rows: list[dict[str, object]] = []
    sample_gene_sets: dict[tuple[str, str], set[str]] = {}
    for dataset, config in selected_configs.items():
        root = config.root_path(ROOT)
        sample_dirs = discover_dataset_samples(config)
        panels = root_panel_files(root)
        groups = root_gene_group_files(root)
        if not sample_dirs:
            summary_rows.append({'dataset': dataset, 'sample': '', 'sample_dir': '', 'status': 'missing_or_empty_dataset_root', 'panel_resolution': 'none', 'marker_group_resolution': 'none', 'n_panel_files': 0, 'n_feature_matrix_files': 0, 'n_panel_targets': 0, 'n_negative_control_targets': 0, 'n_panel_genes': 0, 'n_matrix_gene_features': 0, 'n_assay_genes': 0, 'n_marker_group_files': 0, 'n_marker_group_genes': 0, 'n_marker_groups': 0, 'panel_files': '', 'feature_matrix_files': '', 'marker_group_files': '', 'read_errors': 'No sample directories discovered'})
            continue
        for sample_dir in sample_dirs:
            sample = sample_dir.name
            panel_paths, panel_resolution = resolve_sample_files(dataset=dataset, root=root, sample_dir=sample_dir, root_files=panels, suffix='_gene_panel.json', same_set_loader=panel_genes)
            group_paths, group_resolution = resolve_sample_files(dataset=dataset, root=root, sample_dir=sample_dir, root_files=groups, suffix='_gene_groups.csv', same_set_loader=marker_group_rows)
            feature_matrix_paths = feature_matrix_files_for_sample(dataset, root, sample_dir, sample)
            read_errors: list[str] = []
            sample_genes: set[str] = set()
            panel_gene_set: set[str] = set()
            matrix_gene_set: set[str] = set()
            panel_names: set[str] = set()
            panel_target_count = 0
            panel_negative_control_count = 0
            matrix_feature_count = 0
            for panel_path in panel_paths:
                try:
                    raw_panel = gene_panel_utils.read_gene_panel_json(panel_path)
                    panel_target_count += len(raw_panel)
                    if 'target_descriptor' in raw_panel.columns:
                        descriptors = raw_panel['target_descriptor'].fillna('').astype(str).str.lower()
                        panel_negative_control_count += int(descriptors.str.contains('control').sum())
                    panel = panel_genes(panel_path)
                    panel_names.update(panel.get('source_name', pd.Series(dtype=str)).dropna().astype(str).loc[lambda s: s.ne('')])
                    for row in panel.to_dict('records'):
                        gene = str(row.get('gene_upper', '')).upper()
                        if not gene:
                            continue
                        panel_gene_set.add(gene)
                        sample_genes.add(gene)
                        panel_gene_rows.append({'dataset': dataset, 'sample': sample, 'source_kind': 'gene_panel_json', 'gene': gene, 'gene_display': row.get('gene', ''), 'gene_id': row.get('gene_id', ''), 'panel_file': str(panel_path.relative_to(ROOT)), 'source_name': row.get('source_name', ''), 'n_codewords': row.get('n_codewords', pd.NA)})
                except Exception as error:
                    message = f'{panel_path.name}: {type(error).__name__}: {error}'
                    read_errors.append(message)
                    read_error_rows.append({'dataset': dataset, 'sample': sample, 'path': str(panel_path), 'error': message})
            for matrix_path in feature_matrix_paths:
                try:
                    features = feature_matrix_genes(matrix_path)
                    matrix_feature_count += len(features)
                    for row in features.to_dict('records'):
                        gene = str(row.get('gene_upper', '')).upper()
                        if not gene:
                            continue
                        matrix_gene_set.add(gene)
                        sample_genes.add(gene)
                        panel_gene_rows.append({'dataset': dataset, 'sample': sample, 'source_kind': 'feature_matrix', 'gene': gene, 'gene_display': row.get('gene', ''), 'gene_id': row.get('gene', ''), 'panel_file': str(matrix_path.relative_to(ROOT)), 'source_name': row.get('feature_type', ''), 'n_codewords': pd.NA})
                except Exception as error:
                    message = f'{matrix_path.name}: {type(error).__name__}: {error}'
                    read_errors.append(message)
                    read_error_rows.append({'dataset': dataset, 'sample': sample, 'path': str(matrix_path), 'error': message})
            group_gene_set: set[str] = set()
            group_names: set[str] = set()
            for group_path in group_paths:
                try:
                    group_table = marker_group_rows(group_path)
                    for row in group_table.to_dict('records'):
                        gene = str(row.get('gene_upper', '')).upper()
                        group = str(row.get('group', ''))
                        if not gene or not group:
                            continue
                        group_gene_set.add(gene)
                        group_names.add(group)
                        marker_rows.append({'dataset': dataset, 'sample': sample, 'gene': gene, 'gene_display': row.get('gene', ''), 'group': group, 'grouping_column': row.get('grouping_column', ''), 'marker_group_file': str(group_path.relative_to(ROOT)), 'in_assay': gene in sample_genes})
                except Exception as error:
                    message = f'{group_path.name}: {type(error).__name__}: {error}'
                    read_errors.append(message)
                    read_error_rows.append({'dataset': dataset, 'sample': sample, 'path': str(group_path), 'error': message})
            sample_gene_sets[dataset, sample] = sample_genes
            if (panel_paths or feature_matrix_paths) and (not sample_genes):
                status = 'gene_sources_found_but_no_genes'
            elif sample_genes:
                status = 'ok'
            else:
                status = 'missing_panel_or_matrix'
            summary_rows.append({'dataset': dataset, 'sample': sample, 'sample_dir': str(sample_dir.relative_to(ROOT)), 'status': status, 'panel_resolution': panel_resolution, 'marker_group_resolution': group_resolution, 'n_panel_files': len(panel_paths), 'n_feature_matrix_files': len(feature_matrix_paths), 'n_panel_targets': panel_target_count, 'n_negative_control_targets': panel_negative_control_count, 'n_panel_genes': len(panel_gene_set), 'n_matrix_gene_features': len(matrix_gene_set), 'n_assay_genes': len(sample_genes), 'panel_names': '; '.join(sorted(panel_names)), 'n_marker_group_files': len(group_paths), 'n_marker_group_genes': len(group_gene_set), 'n_marker_groups': len(group_names), 'n_marker_group_genes_in_assay': len(group_gene_set & sample_genes), 'marker_group_gene_coverage_fraction': len(group_gene_set & sample_genes) / len(group_gene_set) if group_gene_set else np.nan, 'panel_files': '; '.join((str(path.relative_to(ROOT)) for path in panel_paths)), 'feature_matrix_files': '; '.join((str(path.relative_to(ROOT)) for path in feature_matrix_paths)), 'marker_group_files': '; '.join((str(path.relative_to(ROOT)) for path in group_paths)), 'read_errors': '; '.join(read_errors)})
    sample_panel_summary = pd.DataFrame(summary_rows)
    panel_gene_table = pd.DataFrame(panel_gene_rows).drop_duplicates() if panel_gene_rows else pd.DataFrame()
    marker_gene_table = pd.DataFrame(marker_rows).drop_duplicates() if marker_rows else pd.DataFrame()
    read_errors = pd.DataFrame(read_error_rows, columns=['dataset', 'sample', 'path', 'error'])
    if sample_panel_summary.empty:
        dataset_status_counts = pd.DataFrame(columns=['dataset', 'status', 'n_samples'])
        dataset_sample_counts = pd.DataFrame(columns=['dataset', 'n_discovered_samples', 'n_samples_with_gene_source', 'n_samples_missing_gene_source'])
    else:
        dataset_status_counts = sample_panel_summary.groupby(['dataset', 'status'], dropna=False).size().reset_index(name='n_samples').sort_values(['dataset', 'status']).reset_index(drop=True)
        dataset_sample_counts = sample_panel_summary.groupby('dataset', as_index=False).agg(n_discovered_samples=('sample', lambda values: values.replace('', pd.NA).dropna().nunique()), n_samples_with_gene_source=('status', lambda values: int(values.eq('ok').sum())), n_samples_missing_gene_source=('status', lambda values: int(values.ne('ok').sum()))).sort_values('dataset').reset_index(drop=True)
    bilous_summary = sample_panel_summary.loc[sample_panel_summary.get('dataset', pd.Series(dtype=str)).eq('Bilous_2026'), ['dataset', 'sample', 'status', 'panel_resolution', 'marker_group_resolution', 'n_feature_matrix_files', 'n_assay_genes', 'read_errors']].copy() if not sample_panel_summary.empty else pd.DataFrame(columns=['dataset', 'sample', 'status', 'panel_resolution', 'marker_group_resolution', 'n_feature_matrix_files', 'n_assay_genes', 'read_errors'])
    if marker_gene_table.empty:
        marker_group_coverage = pd.DataFrame(columns=['dataset', 'sample', 'group', 'grouping_column', 'marker_group_file', 'n_markers_listed', 'n_markers_in_assay', 'n_markers_missing', 'marker_coverage_fraction', 'has_incomplete_marker_coverage', 'has_fewer_than_two_markers_in_assay', 'marker_group_review_status', 'markers_missing', 'markers_found'])
    else:
        coverage_rows = []
        group_columns = ['dataset', 'sample', 'group', 'grouping_column', 'marker_group_file']
        for keys, group in marker_gene_table.groupby(group_columns, dropna=False, sort=True):
            dataset, sample, group_name, grouping_column, marker_group_file = keys
            listed = set(group['gene'].dropna().astype(str))
            found = set(group.loc[group['in_assay'].fillna(False), 'gene'].dropna().astype(str))
            missing = listed - found
            coverage_rows.append({'dataset': dataset, 'sample': sample, 'group': group_name, 'grouping_column': grouping_column, 'marker_group_file': marker_group_file, 'n_markers_listed': len(listed), 'n_markers_in_assay': len(found), 'n_markers_missing': len(missing), 'marker_coverage_fraction': len(found) / len(listed) if listed else np.nan, 'markers_missing': ', '.join(sorted(missing)), 'markers_found': ', '.join(sorted(found))})
        marker_group_coverage = pd.DataFrame(coverage_rows)
    if marker_group_coverage.empty:
        marker_group_coverage_status_counts = pd.DataFrame(columns=['marker_group_review_status', 'n_rows', 'fraction_of_marker_group_rows'])
        marker_group_caution_rows = marker_group_coverage.copy()
    else:
        marker_group_coverage['n_markers_listed'] = pd.to_numeric(marker_group_coverage['n_markers_listed'], errors='coerce').fillna(0).astype(int)
        marker_group_coverage['n_markers_in_assay'] = pd.to_numeric(marker_group_coverage['n_markers_in_assay'], errors='coerce').fillna(0).astype(int)
        marker_group_coverage['n_markers_missing'] = pd.to_numeric(marker_group_coverage['n_markers_missing'], errors='coerce').fillna(0).astype(int)
        marker_group_coverage['marker_coverage_fraction'] = pd.to_numeric(marker_group_coverage['marker_coverage_fraction'], errors='coerce')
        marker_group_coverage['has_incomplete_marker_coverage'] = marker_group_coverage['n_markers_missing'].gt(0)
        marker_group_coverage['has_fewer_than_two_markers_in_assay'] = marker_group_coverage['n_markers_in_assay'].lt(2)

        def marker_group_review_status(row: pd.Series) -> str:
            if int(row['n_markers_in_assay']) == 0:
                return 'no_markers_in_assay'
            if bool(row['has_incomplete_marker_coverage']):
                return 'incomplete_marker_coverage'
            if bool(row['has_fewer_than_two_markers_in_assay']):
                return 'single_marker_evidence'
            return 'available_for_marker_scoring'
        marker_group_coverage['marker_group_review_status'] = marker_group_coverage.apply(marker_group_review_status, axis=1)
        marker_group_coverage_status_counts = marker_group_coverage['marker_group_review_status'].value_counts(dropna=False).rename_axis('marker_group_review_status').reset_index(name='n_rows').assign(fraction_of_marker_group_rows=lambda table: table['n_rows'] / len(marker_group_coverage))
        marker_group_caution_rows = marker_group_coverage.loc[marker_group_coverage['marker_group_review_status'].ne('available_for_marker_scoring')].sort_values(['marker_group_review_status', 'dataset', 'sample', 'group'])
    if marker_gene_table.empty:
        all_marker_datasets = []
        gene_group_annotation_by_dataset = pd.DataFrame(columns=['gene', 'dataset', 'groups', 'n_groups', 'samples', 'marker_group_files', 'n_rows'])
        gene_group_annotation_matrix = pd.DataFrame()
        gene_group_annotation_conflicts = pd.DataFrame(columns=['gene', 'n_datasets_annotated', 'n_unique_dataset_group_assignments', 'n_marker_datasets_missing_annotation', 'missing_annotation_datasets', 'has_conflicting_dataset_annotations', 'has_multiple_groups_within_dataset', 'has_dataset_annotation_presence_difference', 'has_any_annotation_difference', 'datasets_annotated', 'groups_by_dataset', 'unique_group_assignments'])
        gene_group_annotation_differences = gene_group_annotation_conflicts.copy()
    else:
        annotation_rows = marker_gene_table.copy()
        annotation_rows['gene'] = annotation_rows['gene'].astype(str).str.upper().str.strip()
        annotation_rows['group'] = annotation_rows['group'].astype(str).str.strip()
        annotation_rows = annotation_rows.loc[annotation_rows['gene'].ne('') & annotation_rows['group'].ne('')].copy()
        all_marker_datasets = sorted(annotation_rows['dataset'].dropna().astype(str).unique())

        def join_unique(values: pd.Series) -> str:
            return '; '.join(sorted({str(value) for value in values.dropna().astype(str) if str(value)}))
        gene_group_annotation_by_dataset = annotation_rows.groupby(['gene', 'dataset'], as_index=False).agg(groups=('group', join_unique), n_groups=('group', lambda values: len({str(value) for value in values.dropna().astype(str) if str(value)})), samples=('sample', join_unique), marker_group_files=('marker_group_file', join_unique), n_rows=('group', 'size')).sort_values(['gene', 'dataset']).reset_index(drop=True)
        gene_group_annotation_matrix = gene_group_annotation_by_dataset.pivot_table(index='gene', columns='dataset', values='groups', aggfunc='first', fill_value='').sort_index(axis=0).sort_index(axis=1)
        conflict_rows: list[dict[str, object]] = []
        for gene, group in gene_group_annotation_by_dataset.groupby('gene', sort=True):
            dataset_map = {row.dataset: row.groups for row in group.itertuples(index=False)}
            unique_assignments = sorted({value for value in dataset_map.values() if value})
            missing_datasets = sorted(set(all_marker_datasets) - set(dataset_map))
            has_label_conflict = len(unique_assignments) > 1
            has_presence_difference = 0 < len(dataset_map) < len(all_marker_datasets)
            conflict_rows.append({'gene': gene, 'n_datasets_annotated': len(dataset_map), 'n_unique_dataset_group_assignments': len(unique_assignments), 'n_marker_datasets_missing_annotation': len(missing_datasets), 'missing_annotation_datasets': '; '.join(missing_datasets), 'has_conflicting_dataset_annotations': has_label_conflict, 'has_multiple_groups_within_dataset': bool(group['n_groups'].gt(1).any()), 'has_dataset_annotation_presence_difference': has_presence_difference, 'has_any_annotation_difference': has_label_conflict or has_presence_difference, 'datasets_annotated': '; '.join(sorted(dataset_map)), 'groups_by_dataset': ' | '.join((f'{dataset}: {dataset_map[dataset]}' for dataset in sorted(dataset_map))), 'unique_group_assignments': ' | '.join(unique_assignments)})
        gene_group_annotation_conflicts = pd.DataFrame(conflict_rows).sort_values(['has_conflicting_dataset_annotations', 'n_unique_dataset_group_assignments', 'gene'], ascending=[False, False, True])
        gene_group_annotation_differences = pd.DataFrame(conflict_rows).sort_values(['has_any_annotation_difference', 'has_conflicting_dataset_annotations', 'n_marker_datasets_missing_annotation', 'gene'], ascending=[False, False, False, True])
    n_marker_annotated_genes = gene_group_annotation_conflicts.shape[0]
    n_group_label_conflicts = int(gene_group_annotation_conflicts['has_conflicting_dataset_annotations'].sum()) if not gene_group_annotation_conflicts.empty else 0
    n_presence_differences = int(gene_group_annotation_differences['has_dataset_annotation_presence_difference'].sum()) if not gene_group_annotation_differences.empty else 0
    author_annotation_summary = pd.DataFrame([{'n_marker_providing_datasets': len(all_marker_datasets), 'marker_providing_datasets': '; '.join(all_marker_datasets), 'n_marker_annotated_genes': n_marker_annotated_genes, 'n_genes_with_conflicting_group_labels': n_group_label_conflicts, 'n_genes_annotated_in_some_marker_datasets_only': n_presence_differences}])
    conflict_view = gene_group_annotation_conflicts.loc[gene_group_annotation_conflicts['has_conflicting_dataset_annotations'].fillna(False).astype(bool)] if not gene_group_annotation_conflicts.empty else gene_group_annotation_conflicts
    difference_view = gene_group_annotation_differences.loc[gene_group_annotation_differences['has_any_annotation_difference'].fillna(False).astype(bool)] if not gene_group_annotation_differences.empty else gene_group_annotation_differences
    display(gene_group_annotation_by_dataset.head(20))

    def join_nonempty(values: pd.Series) -> str:
        return '; '.join(sorted({str(value).strip() for value in values.dropna().astype(str) if str(value).strip()}))

    def join_semicolon_gene_lists(values: pd.Series) -> str:
        genes: set[str] = set()
        for value in values.dropna().astype(str):
            genes.update((part.strip() for part in value.split(';') if part.strip()))
        return '; '.join(sorted(genes, key=natural_key))
    panel_implied_marker_groups_columns = ['dataset', 'sample', 'group', 'has_author_marker_group_file', 'group_in_author_marker_file', 'evidence_type', 'n_matching_marker_genes', 'matching_marker_genes', 'reference_datasets', 'reference_marker_group_files']
    panel_implied_marker_group_summary_columns = ['dataset', 'sample', 'status', 'has_author_marker_group_file', 'n_author_marker_groups', 'n_panel_implied_marker_groups', 'n_author_groups_also_implied', 'n_panel_implied_groups_not_in_author_file', 'n_author_groups_not_implied_by_panel', 'author_marker_groups', 'panel_implied_marker_groups', 'panel_implied_groups_not_in_author_file', 'author_groups_not_implied_by_panel', 'evidence_note']
    panel_implied_marker_group_by_dataset_columns = ['dataset', 'n_samples', 'n_samples_with_author_marker_groups', 'n_samples_with_panel_implied_groups', 'n_author_marker_groups', 'n_panel_implied_marker_groups', 'author_marker_groups', 'panel_implied_marker_groups']
    if marker_gene_table.empty or sample_panel_summary.empty:
        marker_reference = pd.DataFrame(columns=['gene', 'group', 'dataset', 'sample', 'marker_group_file'])
        panel_implied_marker_groups = pd.DataFrame(columns=panel_implied_marker_groups_columns)
        panel_implied_marker_group_summary = pd.DataFrame(columns=panel_implied_marker_group_summary_columns)
        panel_implied_marker_group_by_dataset = pd.DataFrame(columns=panel_implied_marker_group_by_dataset_columns)
    else:
        marker_reference = marker_gene_table.copy()
        marker_reference['gene'] = marker_reference['gene'].astype(str).str.upper().str.strip()
        marker_reference['group'] = marker_reference['group'].astype(str).str.strip()
        marker_reference = marker_reference.loc[marker_reference['gene'].ne('') & marker_reference['group'].ne('')].copy()
        reference_rows_by_gene: dict[str, list[dict[str, object]]] = defaultdict(list)
        author_groups_by_sample: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in marker_reference.to_dict('records'):
            gene = str(row.get('gene', '')).upper().strip()
            group = str(row.get('group', '')).strip()
            dataset = str(row.get('dataset', '')).strip()
            sample = str(row.get('sample', '')).strip()
            marker_file = str(row.get('marker_group_file', '')).strip()
            if not gene or not group:
                continue
            reference_rows_by_gene[gene].append({'group': group, 'dataset': dataset, 'sample': sample, 'marker_group_file': marker_file})
            author_groups_by_sample[dataset, sample].add(group)
        implied_rows: list[dict[str, object]] = []
        implied_summary_rows: list[dict[str, object]] = []
        for sample_row in sample_panel_summary.sort_values(['dataset', 'sample']).itertuples(index=False):
            dataset = str(sample_row.dataset)
            sample = str(sample_row.sample)
            status = str(sample_row.status)
            genes = {str(gene).upper().strip() for gene in sample_gene_sets.get((dataset, sample), set()) if str(gene).strip()}
            author_groups = set(author_groups_by_sample.get((dataset, sample), set()))
            has_author_marker_group_file = bool(author_groups)
            group_to_genes: dict[str, set[str]] = defaultdict(set)
            group_to_reference_datasets: dict[str, set[str]] = defaultdict(set)
            group_to_reference_files: dict[str, set[str]] = defaultdict(set)
            for gene in sorted(genes, key=natural_key):
                for reference_row in reference_rows_by_gene.get(gene, []):
                    group = str(reference_row['group'])
                    group_to_genes[group].add(gene)
                    if reference_row.get('dataset'):
                        group_to_reference_datasets[group].add(str(reference_row['dataset']))
                    if reference_row.get('marker_group_file'):
                        group_to_reference_files[group].add(str(reference_row['marker_group_file']))
            implied_groups = set(group_to_genes)
            for group in sorted(implied_groups, key=natural_key):
                group_in_author_file = group in author_groups
                if group_in_author_file:
                    evidence_type = 'author_group_and_panel_gene_match'
                elif has_author_marker_group_file:
                    evidence_type = 'reference_implied_extra_group'
                else:
                    evidence_type = 'reference_implied_no_author_group_file'
                implied_rows.append({'dataset': dataset, 'sample': sample, 'group': group, 'has_author_marker_group_file': has_author_marker_group_file, 'group_in_author_marker_file': group_in_author_file, 'evidence_type': evidence_type, 'n_matching_marker_genes': len(group_to_genes[group]), 'matching_marker_genes': '; '.join(sorted(group_to_genes[group], key=natural_key)), 'reference_datasets': '; '.join(sorted(group_to_reference_datasets[group], key=natural_key)), 'reference_marker_group_files': '; '.join(sorted(group_to_reference_files[group], key=natural_key))})
            groups_not_in_author = implied_groups - author_groups
            author_groups_not_implied = author_groups - implied_groups
            if not genes:
                evidence_note = 'No resolved panel or matrix genes, so no panel-implied groups can be inferred.'
            elif has_author_marker_group_file:
                evidence_note = 'Author marker groups are available; panel-implied groups add a cross-dataset reference check.'
            else:
                evidence_note = 'No author marker-group file for this sample; groups are inferred from panel genes matched to other datasets marker files.'
            implied_summary_rows.append({'dataset': dataset, 'sample': sample, 'status': status, 'has_author_marker_group_file': has_author_marker_group_file, 'n_author_marker_groups': len(author_groups), 'n_panel_implied_marker_groups': len(implied_groups), 'n_author_groups_also_implied': len(author_groups & implied_groups), 'n_panel_implied_groups_not_in_author_file': len(groups_not_in_author), 'n_author_groups_not_implied_by_panel': len(author_groups_not_implied), 'author_marker_groups': '; '.join(sorted(author_groups, key=natural_key)), 'panel_implied_marker_groups': '; '.join(sorted(implied_groups, key=natural_key)), 'panel_implied_groups_not_in_author_file': '; '.join(sorted(groups_not_in_author, key=natural_key)), 'author_groups_not_implied_by_panel': '; '.join(sorted(author_groups_not_implied, key=natural_key)), 'evidence_note': evidence_note})
        panel_implied_marker_groups = pd.DataFrame(implied_rows, columns=panel_implied_marker_groups_columns)
        panel_implied_marker_group_summary = pd.DataFrame(implied_summary_rows, columns=panel_implied_marker_group_summary_columns)
        if panel_implied_marker_group_summary.empty:
            panel_implied_marker_group_by_dataset = pd.DataFrame(columns=panel_implied_marker_group_by_dataset_columns)
        else:
            dataset_summary_rows: list[dict[str, object]] = []
            for dataset, group in panel_implied_marker_group_summary.groupby('dataset', sort=True):
                author_groups = set()
                implied_groups = set()
                for value in group['author_marker_groups'].dropna().astype(str):
                    author_groups.update((part.strip() for part in value.split(';') if part.strip()))
                for value in group['panel_implied_marker_groups'].dropna().astype(str):
                    implied_groups.update((part.strip() for part in value.split(';') if part.strip()))
                dataset_summary_rows.append({'dataset': dataset, 'n_samples': int(group['sample'].nunique()), 'n_samples_with_author_marker_groups': int(group['has_author_marker_group_file'].fillna(False).astype(bool).sum()), 'n_samples_with_panel_implied_groups': int(group['n_panel_implied_marker_groups'].gt(0).sum()), 'n_author_marker_groups': len(author_groups), 'n_panel_implied_marker_groups': len(implied_groups), 'author_marker_groups': '; '.join(sorted(author_groups, key=natural_key)), 'panel_implied_marker_groups': '; '.join(sorted(implied_groups, key=natural_key))})
            panel_implied_marker_group_by_dataset = pd.DataFrame(dataset_summary_rows, columns=panel_implied_marker_group_by_dataset_columns).sort_values('dataset').reset_index(drop=True)
    sample_implied_preview_columns = ['dataset', 'sample', 'has_author_marker_group_file', 'n_author_marker_groups', 'n_panel_implied_marker_groups', 'n_panel_implied_groups_not_in_author_file', 'n_author_groups_not_implied_by_panel', 'evidence_note']
    panel_implied_marker_group_summary_view = panel_implied_marker_group_summary[sample_implied_preview_columns].copy()
    no_author_implied = panel_implied_marker_group_summary.loc[~panel_implied_marker_group_summary['has_author_marker_group_file'].fillna(False).astype(bool), ['dataset', 'sample', 'n_panel_implied_marker_groups', 'panel_implied_marker_groups']].copy()
    if panel_implied_marker_groups.empty:
        panel_implied_marker_group_evidence_view = panel_implied_marker_groups.copy()
        panel_implied_extra_groups_in_author_datasets = pd.DataFrame(columns=['dataset', 'group', 'n_samples', 'matching_marker_genes', 'reference_datasets', 'reference_marker_group_files'])
    else:
        panel_implied_marker_group_evidence_view = panel_implied_marker_groups.sort_values(['dataset', 'sample', 'evidence_type', 'group'], key=lambda col: col.map(natural_key) if col.name in {'dataset', 'sample', 'group'} else col)
        extra_group_rows = panel_implied_marker_groups.loc[panel_implied_marker_groups['has_author_marker_group_file'].fillna(False).astype(bool) & ~panel_implied_marker_groups['group_in_author_marker_file'].fillna(False).astype(bool)].copy()
        if extra_group_rows.empty:
            panel_implied_extra_groups_in_author_datasets = pd.DataFrame(columns=['dataset', 'group', 'n_samples', 'matching_marker_genes', 'reference_datasets', 'reference_marker_group_files'])
        else:
            panel_implied_extra_groups_in_author_datasets = extra_group_rows.groupby(['dataset', 'group'], as_index=False).agg(n_samples=('sample', 'nunique'), matching_marker_genes=('matching_marker_genes', join_semicolon_gene_lists), reference_datasets=('reference_datasets', join_nonempty), reference_marker_group_files=('reference_marker_group_files', join_nonempty)).sort_values(['dataset', 'group'], key=lambda col: col.map(natural_key) if col.name in {'dataset', 'group'} else col).reset_index(drop=True)
    reference_genes_by_group: dict[str, set[str]] = defaultdict(set)
    if not marker_reference.empty:
        for row in marker_reference.itertuples(index=False):
            reference_genes_by_group[str(row.group)].add(str(row.gene).upper().strip())
    coverage_rows: list[dict[str, object]] = []
    for row in panel_implied_marker_groups.itertuples(index=False):
        reference_genes = reference_genes_by_group.get(str(row.group), set())
        matching_genes = {part.strip().upper() for part in str(row.matching_marker_genes).split(';') if part.strip()}
        missing_genes = reference_genes - matching_genes
        n_reference = len(reference_genes)
        n_matching = len(matching_genes)
        coverage_fraction = n_matching / n_reference if n_reference else np.nan
        if n_matching == 0:
            status = 'no_reference_markers_in_assay'
        elif n_matching == 1:
            status = 'single_marker_evidence'
        elif n_reference and n_matching < n_reference:
            status = 'partial_reference_marker_coverage'
        else:
            status = 'available_reference_marker_coverage'
        coverage_rows.append({'dataset': row.dataset, 'sample': row.sample, 'group': row.group, 'evidence_type': row.evidence_type, 'has_author_marker_group_file': row.has_author_marker_group_file, 'group_in_author_marker_file': row.group_in_author_marker_file, 'n_reference_marker_genes': n_reference, 'n_matching_marker_genes': n_matching, 'n_reference_markers_missing': len(missing_genes), 'reference_marker_coverage_fraction': coverage_fraction, 'matching_marker_genes': '; '.join(sorted(matching_genes, key=natural_key)), 'reference_markers_missing': '; '.join(sorted(missing_genes, key=natural_key)), 'reference_datasets': row.reference_datasets, 'reference_marker_group_files': row.reference_marker_group_files, 'panel_implied_marker_coverage_status': status})
    panel_implied_marker_group_coverage = pd.DataFrame(coverage_rows)
    if panel_implied_marker_group_coverage.empty:
        panel_implied_marker_group_coverage_summary = pd.DataFrame(columns=['dataset', 'evidence_type', 'panel_implied_marker_coverage_status', 'n_rows', 'median_reference_marker_coverage_fraction'])
    else:
        panel_implied_marker_group_coverage_summary = panel_implied_marker_group_coverage.groupby(['dataset', 'evidence_type', 'panel_implied_marker_coverage_status'], as_index=False).agg(n_rows=('group', 'size'), median_reference_marker_coverage_fraction=('reference_marker_coverage_fraction', 'median')).sort_values(['dataset', 'evidence_type', 'panel_implied_marker_coverage_status']).reset_index(drop=True)
    display(panel_implied_marker_group_by_dataset)
    outputs = {'sample_panel_summary.csv': sample_panel_summary, 'panel_gene_table.csv': panel_gene_table, 'marker_gene_table.csv': marker_gene_table, 'marker_group_coverage.csv': marker_group_coverage, 'gene_group_annotation_by_dataset.csv': gene_group_annotation_by_dataset, 'panel_implied_marker_groups.csv': panel_implied_marker_groups, 'panel_implied_marker_group_coverage.csv': panel_implied_marker_group_coverage, 'read_errors.csv': read_errors}
    for name, table in outputs.items():
        table.to_csv(OUTPUT_DIR / name, index=False)
    print(f'{len(sample_panel_summary)} samples; {len(marker_gene_table)} marker rows.')
    return outputs
