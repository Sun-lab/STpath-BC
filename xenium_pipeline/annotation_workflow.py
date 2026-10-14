from __future__ import annotations
from pathlib import Path
import os
CODE_ROOT = Path(__file__).resolve().parents[1]
from xenium_pipeline.storage import data_path, output_path, epithelial_roots
import pandas as pd
from xenium_pipeline.cell_identity_evidence import collapse_label
from xenium_pipeline.automatic_cluster_annotation import cell_label_masks
from xenium_pipeline.cluster_review import MISSING_CLUSTER_ASSIGNMENT_LABELS
from xenium_pipeline.uncertain_reclustering import author_label_usable_mask
from xenium_pipeline.final_cell_labels import pptx_final_broad_label
RECLUSTER_KEY = 'recluster_r0_7'
BROAD_NAMES = {'tumor cells': 'Tumor Cells', 'normal epithelial cells': 'Normal Epithelial Cells', 't cells': 'T Cells', 'nk cells': 'T Cells', 'b cells': 'B Cells', 'myeloid / other immune': 'pan-APCs', 'pan-apcs': 'pan-APCs', 'stromal cells': 'Stromal Cells', 'smooth muscle cells': 'Smooth Muscle Cells', 'uncertain': 'Uncertain'}

def broad_type(refined, saved_broad=''):
    refined = '' if pd.isna(refined) else str(refined).strip()
    saved_broad = '' if pd.isna(saved_broad) else str(saved_broad).strip()
    if not refined and (not saved_broad):
        return ''
    mapped = BROAD_NAMES.get(refined.casefold())
    if mapped is None:
        mapped = BROAD_NAMES.get(collapse_label(refined)[0].casefold(), 'Uncertain')
    if mapped == 'Uncertain':
        return BROAD_NAMES.get(saved_broad.casefold(), 'Uncertain')
    return mapped

def select_final_labels(labels, dataset):
    result = labels.copy()
    result['final_refined_label'] = 'Uncertain'
    result['final_broad_label'] = 'Uncertain'
    result['final_label_source'] = 'unresolved'

    def take(eligible, refined, broad, source):
        use = result['final_label_source'].eq('unresolved') & eligible
        result.loc[use, 'final_refined_label'] = refined[use]
        result.loc[use, 'final_broad_label'] = broad[use].map(pptx_final_broad_label)
        result.loc[use, 'final_label_source'] = source
    take(cell_label_masks(result['initial_cluster_refined_label'])[1], result['initial_cluster_refined_label'], result['initial_cluster_broad_label'], 'certain_initial_cluster')
    if dataset in {'Janesick_2023', 'Janesick_2025'}:
        take(author_label_usable_mask(result['author_refined_label']), result['author_refined_label'], result['author_broad_label'], 'author')
    else:
        raw = result['automatic_cluster_label'].fillna('').str.strip()
        confidence = result['automatic_label_confidence'].fillna('').str.strip().str.casefold()
        present = ~result[RECLUSTER_KEY].fillna('').str.strip().str.casefold().isin(MISSING_CLUSTER_ASSIGNMENT_LABELS | {'<na>'})
        certain = present & cell_label_masks(raw)[1] & confidence.isin({'moderate', 'high'})
        take(certain, raw, raw.map({label: broad_type(label) for label in raw.unique()}), 'certain_recluster')
        manual = result['manual_selected_broad_label'].fillna('')
        manual = manual.map({label: broad_type('', label) for label in manual.unique()})
        accepted = present & result['manual_decision_status'].fillna('').str.strip().str.casefold().eq('accepted')
        accepted &= ~manual.isin({'', 'Uncertain'})
        take(accepted, pd.Series('Uncertain', index=result.index), manual, 'manual_recluster')
    return result

def export_authors(root):
    from pathlib import Path
    import pandas as pd
    from xenium_pipeline import uncertain_reclustering
    ROOT = Path(root).expanduser().resolve()
    OUTPUT_DIR = ROOT / 'annotation_author_comparison'
    AUTHOR_PARQUET_LABEL_COLUMNS = {}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(ROOT / 'gene_panel_review/sample_panel_summary.csv', dtype=str, keep_default_na=False)
    registry = registry.loc[registry['status'].eq('ok'), ['dataset', 'sample', 'sample_dir']].drop_duplicates()
    parts, sources = ([], [])
    for row in registry.itertuples(index=False):
        source = uncertain_reclustering.discover_author_source(ROOT, row.dataset, row.sample, row.sample_dir, parquet_label_columns=AUTHOR_PARQUET_LABEL_COLUMNS)
        source.update(dataset=row.dataset, sample=row.sample)
        sources.append(source)
        if source['author_source_status'] == 'invalid_columns':
            raise ValueError(f'Invalid author-label columns: {row.dataset}/{row.sample}')
        labels = uncertain_reclustering.load_author_labels(source, ROOT)
        if labels.empty:
            continue
        labels = labels.rename(columns={'author_label': 'author_refined_label'})
        labels.insert(0, 'sample', row.sample)
        labels.insert(0, 'dataset', row.dataset)
        parts.append(labels)
    columns = ['dataset', 'sample', 'cell_id', 'author_raw_label', 'author_refined_label']
    author_labels = pd.concat(parts, ignore_index=True)[columns] if parts else pd.DataFrame(columns=columns)
    if author_labels.duplicated(['dataset', 'sample', 'cell_id']).any():
        raise ValueError('Duplicate author cell identities.')
    path = OUTPUT_DIR / 'cell_identity_evidence.csv.gz'
    temporary = path.with_name(path.name + '.tmp')
    try:
        author_labels.to_csv(temporary, index=False, compression='gzip')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    pd.DataFrame(sources).to_csv(OUTPUT_DIR / 'author_source_index.csv', index=False)
    print(f'{len(author_labels):,} author-labeled cells exported.')
    return OUTPUT_DIR

def compute_evidence(root):
    from pathlib import Path
    import importlib
    import sys
    import pandas as pd
    from xenium_pipeline.workflow_output import HTML, Image, display
    ROOT = Path(root).expanduser().resolve()
    from xenium_pipeline import cluster_review, signature_scores
    pd.set_option('display.max_rows', 80)
    pd.set_option('display.max_columns', 120)
    pd.set_option('display.max_colwidth', 160)
    OUTPUT_DIR = ROOT / 'cell_clustering'
    ANNOTATION_DIR = ROOT / 'cell_annotation'
    GENE_PANEL_REVIEW_DIR = ROOT / 'gene_panel_review'
    DEFAULT_CLUSTER_KEY = 'leiden_r1_3'
    LEIDEN_REVIEW_CLUSTER_KEYS = ('leiden_r1_3',)
    CLUSTERING_SOURCES = ('our_expression',)
    RUN_LEIDEN_MARKER_GENE_DISCOVERY = True
    SKIP_EXISTING_MARKER_TABLES = True
    WRITE_OUTPUTS = True
    MARKER_NORMALIZATION_TARGET = 100000
    MARKER_LFC_PSEUDOCOUNT = 0.1
    MARKER_MIN_FRACTION_EXPRESSING = 0.05
    MARKER_MIN_FRACTION_DIFFERENCE = 0.05
    MARKER_MIN_LOG2_FOLD_CHANGE = 0.25
    MARKER_TOP_N_GENES_PER_CLUSTER = 25
    RUN_WILCOXON_FOR_MARKERS = True
    MAX_CELLS_FOR_WILCOXON = 100000
    MIN_CELL_ID_MATCH_FRACTION = 0.8
    RANDOM_STATE = 1729
    RUN_UCELL_SIGNATURE_SENSITIVITY = True
    SIGNATURE_SENSITIVITY_SAMPLE_KEYS = None
    MIN_SIGNATURE_GENES = 3
    UCELL_MAX_RANK = None
    UCELL_CHUNK_SIZE = 5000
    marker_discovery_results = cluster_review.run_cluster_source_expression_marker_discovery(ROOT, output_dir=OUTPUT_DIR, annotation_dir=ANNOTATION_DIR, gene_panel_review_dir=GENE_PANEL_REVIEW_DIR, our_cluster_keys=LEIDEN_REVIEW_CLUSTER_KEYS, clustering_sources=CLUSTERING_SOURCES, skip_existing=SKIP_EXISTING_MARKER_TABLES, normalization_target=MARKER_NORMALIZATION_TARGET, lfc_pseudocount=MARKER_LFC_PSEUDOCOUNT, min_fraction_expressing=MARKER_MIN_FRACTION_EXPRESSING, min_fraction_difference=MARKER_MIN_FRACTION_DIFFERENCE, min_log2_fold_change=MARKER_MIN_LOG2_FOLD_CHANGE, top_n=MARKER_TOP_N_GENES_PER_CLUSTER, run_wilcoxon=RUN_WILCOXON_FOR_MARKERS, max_cells_for_wilcoxon=MAX_CELLS_FOR_WILCOXON, min_cell_id_match_fraction=MIN_CELL_ID_MATCH_FRACTION, random_state=RANDOM_STATE, write_outputs=WRITE_OUTPUTS)
    display(marker_discovery_results['marker_discovery_summary'][['dataset', 'sample', 'cluster_key', 'status']])
    automatic_candidate_results = cluster_review.build_cluster_source_top_two_gene_hint_candidates(ROOT, output_dir=OUTPUT_DIR, annotation_dir=ANNOTATION_DIR, gene_panel_review_dir=GENE_PANEL_REVIEW_DIR, our_cluster_keys=LEIDEN_REVIEW_CLUSTER_KEYS, clustering_sources=CLUSTERING_SOURCES, write_outputs=WRITE_OUTPUTS)
    print(automatic_candidate_results['output_path'])
    signature_scores_results = signature_scores.run_signature_evidence(ROOT, output_dir=OUTPUT_DIR, annotation_dir=ANNOTATION_DIR, gene_panel_review_dir=GENE_PANEL_REVIEW_DIR, our_cluster_keys=LEIDEN_REVIEW_CLUSTER_KEYS, clustering_sources=CLUSTERING_SOURCES, sample_keys=SIGNATURE_SENSITIVITY_SAMPLE_KEYS, run_ucell=RUN_UCELL_SIGNATURE_SENSITIVITY, min_signature_genes=MIN_SIGNATURE_GENES, ucell_max_rank=UCELL_MAX_RANK, ucell_chunk_size=UCELL_CHUNK_SIZE, random_state=RANDOM_STATE, min_cell_id_match_fraction=MIN_CELL_ID_MATCH_FRACTION, write_outputs=WRITE_OUTPUTS)
    display(signature_scores_results['run_summary'][['dataset', 'sample', 'method', 'status']])
    return OUTPUT_DIR

def annotate_initial(root):
    from pathlib import Path
    import importlib
    import sys
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from xenium_pipeline.workflow_output import FileLink, clear_output, display
    ROOT = Path(root).expanduser().resolve()
    from xenium_pipeline import automatic_cluster_annotation
    pd.set_option('display.max_columns', 80)
    pd.set_option('display.max_colwidth', 140)
    pd.set_option('display.max_rows', 100)
    GENE_PANEL_REVIEW_DIR = ROOT / 'gene_panel_review'
    CELL_ANNOTATION_DIR = ROOT / 'cell_annotation'
    CELL_CLUSTERING_DIR = ROOT / 'cell_clustering'
    OUTPUT_DIR = ROOT / 'automatic_cluster_annotation'
    FIGURE_DIR = OUTPUT_DIR / 'figures'
    WRITE_OUTPUTS = True
    OVERWRITE_EXISTING = False
    WRITE_CELL_LEVEL_OUTPUTS = True
    DEFAULT_LEIDEN_RESOLUTION = 1.3
    SHOW_FULL_SCORE_DISTRIBUTIONS = True
    MIN_CLUSTER_CELLS = 20
    MIN_ASSIGNED_SCORE_FRACTION = 0.2
    MIN_TOP_SCORE_LABEL_FRACTION = 0.5
    MIN_SCORE_LABEL_FRACTION_GAP = 0.15
    MIN_SIGNATURE_COVERAGE = 0.5
    MIN_SIGNATURE_SCORE_GAP = 0.02
    MIN_SIGNATURE_POSITIVE_CELL_FRACTION = 0.1
    MIN_CONSENSUS_METHODS = 2
    UNCERTAIN_LABEL = 'Uncertain'
    CONFIG = automatic_cluster_annotation.AutomaticAnnotationConfig(uncertain_label=UNCERTAIN_LABEL, min_cluster_cells=MIN_CLUSTER_CELLS, min_assigned_score_fraction=MIN_ASSIGNED_SCORE_FRACTION, min_top_score_label_fraction=MIN_TOP_SCORE_LABEL_FRACTION, min_score_label_fraction_gap=MIN_SCORE_LABEL_FRACTION_GAP, min_signature_coverage=MIN_SIGNATURE_COVERAGE, min_signature_score_gap=MIN_SIGNATURE_SCORE_GAP, min_signature_positive_cell_fraction=MIN_SIGNATURE_POSITIVE_CELL_FRACTION, min_consensus_methods=MIN_CONSENSUS_METHODS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    audit = automatic_cluster_annotation.audit_required_inputs(ROOT, annotation_dir=CELL_ANNOTATION_DIR, clustering_dir=CELL_CLUSTERING_DIR, gene_panel_review_dir=GENE_PANEL_REVIEW_DIR, leiden_keys=(), default_leiden_resolution=DEFAULT_LEIDEN_RESOLUTION)
    file_checks, sample_checks, sample_table = (audit[name] for name in ['file_checks', 'sample_checks', 'sample_table'])
    if sample_table.empty:
        raise FileNotFoundError('Sample inventory is unavailable; check score and cluster summary files.')
    selection = sample_table[['dataset', 'sample', 'selected_resolution', 'cluster_key']].copy()
    selection.to_csv(OUTPUT_DIR / 'selected_leiden_resolutions.csv', index=False)
    file_checks.to_csv(OUTPUT_DIR / 'required_global_file_audit.csv', index=False)
    sample_checks.to_csv(OUTPUT_DIR / 'required_sample_file_audit.csv', index=False)
    display(selection.groupby(['dataset', 'cluster_key']).size().rename('n_samples').reset_index())
    if file_checks['status'].ne('ready').any() or sample_checks['status'].ne('ready').any():
        print('Missing inputs are listed in required_global_file_audit.csv and required_sample_file_audit.csv.')
        raise FileNotFoundError('Resolve missing r1.3 inputs before continuing.')
    signature_top_two = automatic_cluster_annotation.select_leiden_evidence(pd.read_csv(data_path(ROOT, 'signature_top_two'), dtype={'cluster': str}), selection)
    signature_top_two = signature_top_two.loc[signature_top_two['method'].eq('ucell') & signature_top_two['granularity'].eq('refined')]
    gene_hint_candidates = automatic_cluster_annotation.select_leiden_evidence(pd.read_csv(CELL_CLUSTERING_DIR / 'combined_cluster_source_top_two_gene_hint_candidates.csv', dtype={'cluster': str}, low_memory=False), selection)
    keys = ['dataset', 'sample', 'cluster_key']
    evidence_audit = selection.copy()
    for evidence, column in [(signature_top_two, 'n_signature_clusters'), (gene_hint_candidates, 'n_gene_hint_clusters')]:
        counts = evidence.groupby(keys)['cluster'].nunique().rename(column).reset_index()
        evidence_audit = evidence_audit.merge(counts, on=keys, how='left', validate='one_to_one')
        evidence_audit[column] = evidence_audit[column].fillna(0).astype(int)
    evidence_audit['status'] = np.where(evidence_audit[['n_signature_clusters', 'n_gene_hint_clusters']].gt(0).all(axis=1), 'ready', 'missing_selected_evidence')
    evidence_audit.to_csv(OUTPUT_DIR / 'selected_evidence_audit.csv', index=False)
    display(evidence_audit.groupby(['dataset', 'status']).size().rename('n_samples').reset_index())
    if evidence_audit['status'].ne('ready').any():
        raise ValueError('Generate cluster evidence for the missing selected partitions; see selected_evidence_audit.csv.')
    cluster_score_vote_summary, cluster_score_label_composition = automatic_cluster_annotation.build_cluster_score_vote_tables(ROOT, sample_table, annotation_dir=CELL_ANNOTATION_DIR, clustering_dir=CELL_CLUSTERING_DIR, uncertain_label=UNCERTAIN_LABEL)
    vote_coverage_summary = cluster_score_vote_summary.groupby(['clustering_source', 'cluster_key'], dropna=False).agg(n_clusters=('cluster', 'size'), median_cluster_cells=('n_cluster_cells', 'median'), median_assigned_score_fraction=('assigned_score_fraction', 'median'), median_top_score_label_fraction=('score_vote_candidate_1_fraction', 'median'), median_score_label_fraction_gap=('score_vote_fraction_gap', 'median')).reset_index()
    display(vote_coverage_summary.round(3))
    cluster_score_vote_summary.to_csv(OUTPUT_DIR / 'cluster_marker_score_vote_summary.csv', index=False)
    cluster_score_label_composition.to_csv(OUTPUT_DIR / 'cluster_marker_score_label_composition.csv.gz', index=False, compression='gzip')
    automatic_cluster_labels = automatic_cluster_annotation.assign_automatic_cluster_labels(cluster_score_vote_summary, signature_top_two, gene_hint_candidates, config=CONFIG, signature_method='ucell', signature_granularity='refined')
    automatic_label_summary = automatic_cluster_labels.groupby(['clustering_source', 'cluster_key', 'automatic_label_confidence'], dropna=False).size().rename('n_clusters').reset_index()
    automatic_label_summary['fraction_within_partition_type'] = automatic_label_summary['n_clusters'] / automatic_label_summary.groupby(['clustering_source', 'cluster_key'])['n_clusters'].transform('sum')
    display(automatic_label_summary.round(3))
    automatic_cluster_labels.to_csv(OUTPUT_DIR / 'automatic_cluster_labels.csv', index=False)
    automatic_label_summary.to_csv(OUTPUT_DIR / 'automatic_cluster_label_summary.csv', index=False)
    confidence_order = ['high', 'moderate', 'uncertain']
    plot_table = automatic_label_summary.pivot_table(index=['clustering_source', 'cluster_key'], columns='automatic_label_confidence', values='fraction_within_partition_type', fill_value=0).reindex(columns=confidence_order, fill_value=0)
    colors = {'high': '#2A9D8F', 'moderate': '#E9C46A', 'uncertain': '#A6A6A6'}
    fig, axis = plt.subplots(figsize=(8, 4.5))
    left = np.zeros(len(plot_table))
    for confidence in confidence_order:
        values = plot_table[confidence].to_numpy()
        axis.barh(range(len(plot_table)), values, left=left, label=confidence.title(), color=colors[confidence])
        left += values
    axis.set_yticks(range(len(plot_table)), [f'{source}: {key}' for source, key in plot_table.index])
    axis.set_xlim(0, 1)
    axis.set_xlabel('Fraction of clusters')
    axis.set_title('Automatic cluster-label confidence')
    axis.legend(frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0))
    fig.savefig(FIGURE_DIR / 'automatic_cluster_label_confidence.png', dpi=180, bbox_inches='tight')
    plt.close()
    cell_annotation_set_summaries, cell_output_index_rows = ([], [])
    targets = [OUTPUT_DIR / row.dataset / row.sample / 'cell_automatic_annotations.csv.gz' for row in selection.itertuples(index=False)]
    if WRITE_OUTPUTS and WRITE_CELL_LEVEL_OUTPUTS and (not OVERWRITE_EXISTING):
        existing = [path for path in targets if path.exists()]
        if existing:
            raise FileExistsError(f'Output already exists: {existing[0]}; set OVERWRITE_EXISTING=True to regenerate.')
    for index, row in enumerate(selection.itertuples(index=False), 1):
        dataset, sample, key = (row.dataset, row.sample, row.cluster_key)
        print(f'[{index}/{len(selection)}] {dataset} / {sample}: {key}', flush=True)
        marker_labels = pd.read_csv(CELL_ANNOTATION_DIR / dataset / sample / 'cell_marker_labels.csv.gz', dtype={'cell_id': str}, keep_default_na=False)
        assignments = pd.read_csv(CELL_CLUSTERING_DIR / dataset / sample / 'cell_clusters.csv.gz', usecols=['cell_id', key], dtype=str, keep_default_na=False)
        cells = automatic_cluster_annotation.propagate_selected_leiden_labels(marker_labels, assignments, automatic_cluster_labels, dataset=dataset, sample=sample, cluster_key=key)
        for label_column in ['marker_score_label', 'initial_cluster_refined_label']:
            counts = cells.groupby(['dataset', 'sample', label_column], dropna=False).size().rename('n_cells').reset_index()
            counts = counts.rename(columns={label_column: 'cell_type'})
            counts['annotation_set'] = label_column
            counts['fraction_of_sample'] = counts['n_cells'] / len(cells)
            cell_annotation_set_summaries.append(counts)
        path = OUTPUT_DIR / dataset / sample / 'cell_automatic_annotations.csv.gz'
        if WRITE_OUTPUTS and WRITE_CELL_LEVEL_OUTPUTS:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + '.tmp')
            try:
                cells.to_csv(temporary, index=False, compression='gzip')
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        cell_output_index_rows.append(dict(dataset=dataset, sample=sample, cluster_key=key, selected_resolution=row.selected_resolution, n_cells=len(cells), cell_output_path=str(path), written=bool(WRITE_OUTPUTS and WRITE_CELL_LEVEL_OUTPUTS)))
    cell_annotation_set_summary = pd.concat(cell_annotation_set_summaries, ignore_index=True)
    cell_output_index = pd.DataFrame(cell_output_index_rows)
    cell_annotation_set_summary.to_csv(OUTPUT_DIR / 'cell_annotation_set_summary.csv', index=False)
    cell_output_index.to_csv(OUTPUT_DIR / 'cell_automatic_annotation_index.csv', index=False)
    availability = cell_annotation_set_summary.copy()
    availability['n_biologically_labeled_cells'] = availability['n_cells'] * automatic_cluster_annotation.cell_label_masks(availability['cell_type'])[1].astype(int)
    availability = availability.groupby('annotation_set').agg(n_cells=('n_cells', 'sum'), n_biologically_labeled_cells=('n_biologically_labeled_cells', 'sum')).reset_index()
    availability['biologically_labeled_fraction'] = availability['n_biologically_labeled_cells'] / availability['n_cells']
    display(availability.round(3))
    availability.to_csv(OUTPUT_DIR / 'annotation_set_availability.csv', index=False)
    for name in ['selected_leiden_resolutions.csv', 'automatic_cluster_labels.csv', 'cell_automatic_annotation_index.csv']:
        display(FileLink(str(OUTPUT_DIR / name)))
    return OUTPUT_DIR

def recluster_uncertain(root):
    from pathlib import Path
    import importlib
    import sys
    import numpy as np
    import pandas as pd
    from xenium_pipeline.workflow_output import Image, Markdown, display
    ROOT = Path(root).expanduser().resolve()
    from xenium_pipeline import uncertain_reclustering
    from xenium_pipeline.cell_identity_evidence import natural_key
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_colwidth', 140)
    print(f'Repository root: {ROOT}')
    OUTPUT_DIR = ROOT / 'uncertain_cell_reclustering'
    CONSOLIDATED_AUTHOR_PATH = ROOT / 'annotation_author_comparison' / 'cell_identity_evidence.csv.gz'
    AUTHOR_PARQUET_LABEL_COLUMNS: dict[str, str] = {}
    AUTHOR_UNUSABLE_LABELS = ['Unlabeled', 'Missing', 'Unknown', 'Unassigned', 'Uncertain']
    SELECTED_STUDIES: list[str] | None = None
    WRITE_AUDIT_OUTPUTS = True
    OVERWRITE_AUDIT_OUTPUTS = True
    RUN_RECLUSTERING = True
    RECLUSTER_MODE = 'balanced_sketch'
    NORMALIZATION_TARGET = 10000
    N_PCS = 20
    N_NEIGHBORS = 20
    LEIDEN_RESOLUTIONS = (0.7,)
    RANDOM_STATE = 1729
    RUN_UMAP = True
    MAX_FULL_CELLS_PER_STUDY = 300000
    MAX_SKETCH_CELLS_PER_STUDY = 250000
    MIN_SKETCH_CELLS_PER_STRATUM = 100
    MAX_SKETCH_CELLS_PER_STRATUM = 10000
    PROJECTION_NEIGHBORS = 15
    PROJECTION_CHUNK_SIZE = 50000
    MIN_PROJECTION_VOTE_FRACTION = 0.5
    MIN_PROJECTION_VOTE_MARGIN = 0.1
    MAX_UMAP_PLOT_CELLS = 100000
    MIN_CELLS_FOR_SAMPLE_PRESENCE = 20
    BROAD_MIN_SAMPLE_FRACTION = 0.5
    BROAD_MAX_JS_DIVERGENCE = 0.1
    DOMINATED_MIN_CLUSTER_FRACTION = 0.8
    DOMINATED_MIN_ENRICHMENT = 2.0
    WRITE_MODEL_CACHE = True
    OVERWRITE_EXISTING = False
    SKIP_COMPLETED = True
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Output folder: {OUTPUT_DIR}')
    print(f'Audit-only run: {not RUN_RECLUSTERING}')
    dependency_rows = []
    for module_name in ('numpy', 'pandas', 'matplotlib', 'scipy', 'h5py', 'zarr', 'anndata', 'scanpy', 'igraph', 'leidenalg', 'pynndescent'):
        try:
            module = importlib.import_module(module_name)
            dependency_rows.append({'module': module_name, 'available': True, 'version': getattr(module, '__version__', ''), 'required_for': 'audit and reclustering' if module_name in {'numpy', 'pandas'} else 'reclustering/figures'})
        except Exception as error:
            dependency_rows.append({'module': module_name, 'available': False, 'version': '', 'required_for': 'audit and reclustering' if module_name in {'numpy', 'pandas'} else 'reclustering/figures', 'error': f'{type(error).__name__}: {error}'})
    dependency_status = pd.DataFrame(dependency_rows)
    display(dependency_status)
    input_inventory = uncertain_reclustering.build_reclustering_inventory(ROOT, consolidated_author_path=CONSOLIDATED_AUTHOR_PATH, selected_studies=SELECTED_STUDIES, parquet_label_columns=AUTHOR_PARQUET_LABEL_COLUMNS)
    inventory_columns = ['dataset', 'sample', 'has_automatic_annotations', 'has_author_labels', 'author_source_type', 'has_feature_matrix', 'ready_for_hierarchy', 'ready_for_reclustering', 'missing_automatic_columns']
    display(input_inventory[inventory_columns])
    inventory_by_study = input_inventory.groupby('dataset', as_index=False).agg(n_samples=('sample', 'nunique'), n_with_automatic_annotations=('has_automatic_annotations', 'sum'), n_with_author_labels=('has_author_labels', 'sum'), n_with_feature_matrix=('has_feature_matrix', 'sum'), n_ready_for_hierarchy=('ready_for_hierarchy', 'sum'), n_ready_for_reclustering=('ready_for_reclustering', 'sum'))
    display(inventory_by_study)
    hierarchy_counts_by_sample, hierarchy_counts_by_study, hierarchy_input_audit = uncertain_reclustering.run_hierarchy_audit(input_inventory, ROOT, OUTPUT_DIR, unusable_author_labels=AUTHOR_UNUSABLE_LABELS, write_outputs=WRITE_AUDIT_OUTPUTS, overwrite_existing=OVERWRITE_AUDIT_OUTPUTS)
    count_columns = ['n_total_cells', 'n_certain_leiden_label', 'n_author_label_rescue', 'n_reclustering_target', 'n_qc_excluded']
    display(hierarchy_counts_by_study[['dataset', *count_columns]])
    display(hierarchy_counts_by_sample[['dataset', 'sample', *count_columns]])
    failed_hierarchy = hierarchy_input_audit.loc[~hierarchy_input_audit['hierarchy_status'].astype(str).eq('success')]
    if not failed_hierarchy.empty:
        display(failed_hierarchy[['dataset', 'sample', 'hierarchy_status', 'error']])
    predicate_columns = ['dataset', 'sample', 'n_cells', 'n_has_original_leiden_r1_3_cluster', 'n_biological_r1_3_automatic_label', 'n_moderate_or_high_r1_3_confidence', 'n_certain_leiden_label', 'r1_3_confidence_values']
    hierarchy_predicate_diagnostics = hierarchy_input_audit.loc[hierarchy_input_audit['hierarchy_status'].astype(str).eq('success'), predicate_columns].copy()
    display(hierarchy_predicate_diagnostics)
    reclustering_config = uncertain_reclustering.ReclusteringConfig(mode=RECLUSTER_MODE, normalization_target=NORMALIZATION_TARGET, n_pcs=N_PCS, n_neighbors=N_NEIGHBORS, leiden_resolutions=tuple(LEIDEN_RESOLUTIONS), random_state=RANDOM_STATE, run_umap=RUN_UMAP, max_full_cells_per_study=MAX_FULL_CELLS_PER_STUDY, max_sketch_cells_per_study=MAX_SKETCH_CELLS_PER_STUDY, min_sketch_cells_per_stratum=MIN_SKETCH_CELLS_PER_STRATUM, max_sketch_cells_per_stratum=MAX_SKETCH_CELLS_PER_STRATUM, projection_neighbors=PROJECTION_NEIGHBORS, projection_chunk_size=PROJECTION_CHUNK_SIZE, min_projection_vote_fraction=MIN_PROJECTION_VOTE_FRACTION, min_projection_vote_margin=MIN_PROJECTION_VOTE_MARGIN, max_umap_plot_cells=MAX_UMAP_PLOT_CELLS, write_model_cache=WRITE_MODEL_CACHE, overwrite_existing=OVERWRITE_EXISTING, skip_completed=SKIP_COMPLETED)
    if not WRITE_AUDIT_OUTPUTS:
        raise ValueError('WRITE_AUDIT_OUTPUTS must be True before reclustering.')
    unavailable_dependencies = dependency_status.loc[~dependency_status['available'] & dependency_status['module'].isin({'matplotlib', 'scipy', 'h5py', 'zarr', 'anndata', 'scanpy', 'igraph', 'leidenalg', 'pynndescent'}), 'module'].tolist()
    if unavailable_dependencies:
        raise ModuleNotFoundError('Install reclustering dependencies before running: ' + ', '.join(unavailable_dependencies))
    reclustering_run_summary = uncertain_reclustering.run_selected_studies(input_inventory, OUTPUT_DIR, ROOT, OUTPUT_DIR, reclustering_config)
    display(reclustering_run_summary)
    cluster_summary_parts = []
    sample_composition_parts = []
    figure_index_parts = []
    for study in sorted(input_inventory['dataset'].astype(str).unique(), key=natural_key):
        study_dir = OUTPUT_DIR / uncertain_reclustering.safe_name(study)
        cluster_path = study_dir / 'cluster_summary.csv'
        composition_path = study_dir / 'cluster_sample_composition.csv'
        figure_path = study_dir / 'umap_figure_index.csv'
        if cluster_path.is_file():
            cluster_summary_parts.append(pd.read_csv(cluster_path))
        if composition_path.is_file():
            sample_composition_parts.append(pd.read_csv(composition_path))
        if figure_path.is_file():
            figure_index_parts.append(pd.read_csv(figure_path))
    combined_cluster_summary = pd.concat(cluster_summary_parts, ignore_index=True) if cluster_summary_parts else pd.DataFrame()
    combined_sample_composition = pd.concat(sample_composition_parts, ignore_index=True) if sample_composition_parts else pd.DataFrame()
    umap_figure_index = pd.concat(figure_index_parts, ignore_index=True) if figure_index_parts else pd.DataFrame(columns=['dataset', 'resolution', 'cluster_key', 'figure_path'])
    if combined_cluster_summary.empty:
        print('No study-level reclustering outputs are available yet.')
    else:
        cluster_count_overview = combined_cluster_summary.groupby(['dataset', 'resolution'], as_index=False).agg(n_clusters=('cluster', 'nunique'), n_target_cells=('n_cells', 'sum'), max_dominant_sample_fraction=('dominant_sample_fraction', 'max'), median_sample_composition_entropy=('normalized_sample_composition_entropy', 'median'))
        display(cluster_count_overview)
    if not umap_figure_index.empty:
        link_lines = ['**Saved UMAP figures:**']
        for row in umap_figure_index.itertuples(index=False):
            path = Path(str(row.figure_path))
            if not path.is_absolute():
                path = ROOT / path
            label = f'{row.dataset} | {row.cluster_key}'
            link_lines.append(f'- [{label}]({path.as_posix()})')
        display(Markdown('\n'.join(link_lines)))
    MIN_CELLS_FOR_SAMPLE_PRESENCE = MIN_CELLS_FOR_SAMPLE_PRESENCE
    BROAD_MIN_SAMPLE_FRACTION = BROAD_MIN_SAMPLE_FRACTION
    BROAD_MAX_JS_DIVERGENCE = BROAD_MAX_JS_DIVERGENCE
    DOMINATED_MIN_CLUSTER_FRACTION = DOMINATED_MIN_CLUSTER_FRACTION
    DOMINATED_MIN_ENRICHMENT = DOMINATED_MIN_ENRICHMENT
    specificity_summary_parts = []
    specificity_composition_parts = []
    for study in sorted(input_inventory['dataset'].astype(str).unique(), key=natural_key):
        assignment_path = OUTPUT_DIR / uncertain_reclustering.safe_name(study) / 'study_recluster_assignments.csv.gz'
        if not assignment_path.is_file():
            continue
        assignment_columns = list(pd.read_csv(assignment_path, nrows=0).columns)
        cluster_keys = [uncertain_reclustering.cluster_key_for_resolution(resolution).replace('leiden_', 'recluster_') for resolution in LEIDEN_RESOLUTIONS]
        available_keys = [key for key in cluster_keys if key in assignment_columns]
        assignments = pd.read_csv(assignment_path, usecols=['dataset', 'sample', *available_keys], dtype={key: 'string' for key in available_keys})
        for cluster_key in available_keys:
            summary, composition = uncertain_reclustering.summarize_cluster_sample_specificity(assignments, cluster_key, min_cells_for_presence=MIN_CELLS_FOR_SAMPLE_PRESENCE, broad_min_sample_fraction=BROAD_MIN_SAMPLE_FRACTION, broad_max_js_divergence=BROAD_MAX_JS_DIVERGENCE, dominated_min_cluster_fraction=DOMINATED_MIN_CLUSTER_FRACTION, dominated_min_enrichment=DOMINATED_MIN_ENRICHMENT)
            specificity_summary_parts.append(summary)
            specificity_composition_parts.append(composition)
    cluster_sample_specificity = pd.concat(specificity_summary_parts, ignore_index=True) if specificity_summary_parts else pd.DataFrame()
    cluster_sample_composition_enrichment = pd.concat(specificity_composition_parts, ignore_index=True) if specificity_composition_parts else pd.DataFrame()
    if not cluster_sample_specificity.empty:
        cluster_sample_specificity.to_csv(OUTPUT_DIR / 'cluster_sample_specificity.csv', index=False)
        cluster_sample_composition_enrichment.to_csv(OUTPUT_DIR / 'cluster_sample_composition_enrichment.csv', index=False)
        specificity_category_summary = cluster_sample_specificity.groupby(['dataset', 'cluster_key', 'sample_specificity_category'], dropna=False).size().rename('n_clusters').reset_index()
        display(specificity_category_summary)
    else:
        print('No completed reclustering assignments are available for sample-specificity analysis.')
    return reclustering_run_summary

def annotate_reclusters(root, *, run=True):
    from pathlib import Path
    import importlib
    import sys
    import numpy as np
    import pandas as pd
    from xenium_pipeline.workflow_output import Markdown, display
    ROOT = Path(root).expanduser().resolve()
    from xenium_pipeline import automatic_cluster_annotation, reclustered_cluster_annotation
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_colwidth', 140)
    print(f'Repository root: {ROOT}')
    RECLUSTERING_DIR = ROOT / 'uncertain_cell_reclustering'
    OUTPUT_DIR = ROOT / 'reclustered_cluster_annotation'
    REVIEW_DECISIONS_PATH = OUTPUT_DIR / 'manual_recluster_decisions.csv'
    MARKER_SCORE_REVIEW_DIR = ROOT / 'cell_annotation'
    GENE_PANEL_REVIEW_DIR = ROOT / 'gene_panel_review'
    DEFAULT_RESOLUTION = 0.7
    SELECTED_RESOLUTION_BY_STUDY: dict[str, float] = {}
    SELECTED_STUDIES: list[str] | None = None
    RUN_ANNOTATION = run
    OVERWRITE_EXISTING = False
    SKIP_COMPLETED = True
    MAX_UNRESOLVED_UMAP_CELLS = 100000
    NORMALIZATION_TARGET = 100000
    LFC_PSEUDOCOUNT = 0.1
    MIN_FRACTION_EXPRESSING = 0.05
    MIN_FRACTION_DIFFERENCE = 0.05
    MIN_LOG2_FOLD_CHANGE = 0.25
    TOP_N_MARKERS = 25
    RUN_WILCOXON = True
    MAX_CELLS_FOR_WILCOXON = 100000
    RANDOM_STATE = 1729
    MIN_SIGNATURE_GENES = 3
    UCELL_MAX_RANK = None
    UCELL_CHUNK_SIZE = 10000
    MIN_CLUSTER_CELLS = 20
    MIN_ASSIGNED_SCORE_FRACTION = 0.2
    MIN_TOP_SCORE_LABEL_FRACTION = 0.5
    MIN_SCORE_LABEL_FRACTION_GAP = 0.15
    MIN_SIGNATURE_COVERAGE = 0.5
    MIN_SIGNATURE_SCORE_GAP = 0.02
    MIN_SIGNATURE_POSITIVE_CELL_FRACTION = 0.1
    MIN_CONSENSUS_METHODS = 2
    UNCERTAIN_LABEL = 'Uncertain'
    ANNOTATION_CONFIG = automatic_cluster_annotation.AutomaticAnnotationConfig(uncertain_label=UNCERTAIN_LABEL, min_cluster_cells=MIN_CLUSTER_CELLS, min_assigned_score_fraction=MIN_ASSIGNED_SCORE_FRACTION, min_top_score_label_fraction=MIN_TOP_SCORE_LABEL_FRACTION, min_score_label_fraction_gap=MIN_SCORE_LABEL_FRACTION_GAP, min_signature_coverage=MIN_SIGNATURE_COVERAGE, min_signature_score_gap=MIN_SIGNATURE_SCORE_GAP, min_signature_positive_cell_fraction=MIN_SIGNATURE_POSITIVE_CELL_FRACTION, min_consensus_methods=MIN_CONSENSUS_METHODS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Audit-only run: {not RUN_ANNOTATION}')
    annotation_inventory, sample_inventory = reclustered_cluster_annotation.build_annotation_inventory(ROOT, reclustering_dir=RECLUSTERING_DIR, gene_panel_review_dir=GENE_PANEL_REVIEW_DIR, selected_resolution_by_study=SELECTED_RESOLUTION_BY_STUDY, default_resolution=DEFAULT_RESOLUTION, selected_studies=SELECTED_STUDIES)
    annotation_inventory.to_csv(OUTPUT_DIR / 'annotation_input_audit.csv', index=False)
    display(annotation_inventory)
    failed_audit = annotation_inventory.loc[~annotation_inventory['ready_for_annotation'].astype(bool)]
    if not failed_audit.empty:
        display(failed_audit[['dataset', 'cluster_key', 'missing_assignment_columns', 'missing_matrix_samples', 'missing_gene_review_files']])
    selected_specificity_parts = []
    selected_composition_parts = []
    for row in annotation_inventory.loc[annotation_inventory['has_recluster_assignments']].itertuples(index=False):
        assignment_path = Path(str(row.assignment_path))
        if not assignment_path.is_absolute():
            assignment_path = ROOT / assignment_path
        header = list(pd.read_csv(assignment_path, nrows=0).columns)
        if row.cluster_key not in header:
            continue
        assignments = pd.read_csv(assignment_path, usecols=['dataset', 'sample', row.cluster_key], dtype={row.cluster_key: 'string'})
        summary, composition = reclustered_cluster_annotation.uncertain_reclustering.summarize_cluster_sample_specificity(assignments, row.cluster_key)
        selected_specificity_parts.append(summary)
        selected_composition_parts.append(composition)
    selected_cluster_sample_specificity = pd.concat(selected_specificity_parts, ignore_index=True) if selected_specificity_parts else pd.DataFrame()
    selected_cluster_sample_composition = pd.concat(selected_composition_parts, ignore_index=True) if selected_composition_parts else pd.DataFrame()
    if not selected_cluster_sample_specificity.empty:
        selected_cluster_sample_specificity.to_csv(OUTPUT_DIR / 'selected_cluster_sample_specificity.csv', index=False)
        display(selected_cluster_sample_specificity.groupby(['dataset', 'cluster_key', 'sample_specificity_category']).size().rename('n_clusters').reset_index())
    if RUN_ANNOTATION:
        annotation_run_summary = reclustered_cluster_annotation.run_annotation_inventory(annotation_inventory, sample_inventory, root=ROOT, output_dir=OUTPUT_DIR, gene_panel_review_dir=GENE_PANEL_REVIEW_DIR, annotation_config=ANNOTATION_CONFIG, normalization_target=NORMALIZATION_TARGET, lfc_pseudocount=LFC_PSEUDOCOUNT, min_fraction_expressing=MIN_FRACTION_EXPRESSING, min_fraction_difference=MIN_FRACTION_DIFFERENCE, min_log2_fold_change=MIN_LOG2_FOLD_CHANGE, top_n_markers=TOP_N_MARKERS, run_wilcoxon=RUN_WILCOXON, max_cells_for_wilcoxon=MAX_CELLS_FOR_WILCOXON, min_signature_genes=MIN_SIGNATURE_GENES, ucell_max_rank=UCELL_MAX_RANK, ucell_chunk_size=UCELL_CHUNK_SIZE, random_state=RANDOM_STATE, overwrite_existing=OVERWRITE_EXISTING, skip_completed=SKIP_COMPLETED)
    else:
        run_summary_path = OUTPUT_DIR / 'annotation_run_summary.csv'
        annotation_run_summary = pd.read_csv(run_summary_path) if run_summary_path.is_file() else pd.DataFrame()
        print('Annotation was not run because RUN_ANNOTATION is False.')
    display(annotation_run_summary)
    automatic_label_parts = []
    for row in annotation_inventory.itertuples(index=False):
        path = OUTPUT_DIR / reclustered_cluster_annotation.safe_name(str(row.dataset)) / str(row.cluster_key) / 'automatic_recluster_labels.csv'
        if path.is_file():
            automatic_label_parts.append(pd.read_csv(path, dtype={'cluster': 'string'}, low_memory=False))
    combined_automatic_recluster_labels = pd.concat(automatic_label_parts, ignore_index=True) if automatic_label_parts else pd.DataFrame()
    if not combined_automatic_recluster_labels.empty:
        combined_automatic_recluster_labels.to_csv(OUTPUT_DIR / 'combined_automatic_recluster_labels.csv', index=False)
        display(combined_automatic_recluster_labels.groupby(['dataset', 'cluster_key', 'automatic_label_confidence', 'sample_specificity_category'], dropna=False).size().rename('n_clusters').reset_index())
    else:
        print('No automatic recluster labels are available yet.')
    candidate_parts = []
    for row in annotation_inventory.itertuples(index=False):
        candidate_path = OUTPUT_DIR / reclustered_cluster_annotation.safe_name(str(row.dataset)) / reclustered_cluster_annotation.safe_name(str(row.cluster_key)) / 'unresolved_cluster_candidates.csv'
        if candidate_path.is_file():
            candidate_parts.append(pd.read_csv(candidate_path, dtype={'cluster': 'string'}, low_memory=False))
    combined_unresolved_cluster_candidates = pd.concat(candidate_parts, ignore_index=True) if candidate_parts else pd.DataFrame()
    combined_candidate_path = OUTPUT_DIR / 'combined_unresolved_cluster_candidates.csv'
    combined_unresolved_cluster_candidates, unresolved_cluster_review = reclustered_cluster_annotation.route_unresolved_cluster_review(combined_unresolved_cluster_candidates)
    review_path = OUTPUT_DIR / 'unresolved_cluster_review.csv'
    unresolved_cluster_review.to_csv(review_path, index=False)
    if candidate_parts:
        combined_unresolved_cluster_candidates.to_csv(combined_candidate_path, index=False)
    if not combined_unresolved_cluster_candidates.empty:
        leading_unresolved_candidates = combined_unresolved_cluster_candidates.loc[pd.to_numeric(combined_unresolved_cluster_candidates['candidate_rank'], errors='coerce').eq(1)].copy()
        review_class_summary = unresolved_cluster_review.groupby(['dataset', 'cluster_key', 'review_class'], dropna=False).agg(n_unresolved_clusters=('cluster', 'nunique'), n_cells=('n_cluster_cells', 'sum')).reset_index()
        display(review_class_summary)
        print(f'Cluster review pathways: {review_path}')
        print(f'Complete unresolved-candidate table: {combined_candidate_path}')
    else:
        leading_unresolved_candidates = pd.DataFrame()
        print('No unresolved-cluster candidate files are available yet.')
    import importlib
    from xenium_pipeline.workflow_output import FileLink
    saved_partitions = []
    for cell_path in sorted(OUTPUT_DIR.glob('*/*/cell_recluster_annotations.csv.gz')):
        study = cell_path.parent.parent.name
        selected_key = reclustered_cluster_annotation.recluster_key(SELECTED_RESOLUTION_BY_STUDY.get(study, DEFAULT_RESOLUTION))
        if SELECTED_STUDIES is not None and study not in SELECTED_STUDIES:
            continue
        if cell_path.parent.name == selected_key:
            saved_partitions.append({'dataset': study, 'cluster_key': selected_key})
    if not REVIEW_DECISIONS_PATH.is_file():
        print(f'Optional manual CSV absent: {REVIEW_DECISIONS_PATH}; using automatic consensus only.')
    if not saved_partitions:
        print(f'No selected automatic per-cell files found under {OUTPUT_DIR}.')
    else:
        reviewed_cell_index, reviewed_cell_counts = reclustered_cluster_annotation.export_reviewed_recluster_cells(OUTPUT_DIR, REVIEW_DECISIONS_PATH, pd.DataFrame(saved_partitions))
        display(reviewed_cell_counts.groupby(['dataset', 'cluster_key', 'recluster_status'], dropna=False)['n_cells'].sum().reset_index())
        for name in ['reviewed_cell_annotation_index.csv', 'reviewed_cell_label_status_counts.csv']:
            display(FileLink(str(OUTPUT_DIR / name)))
        print('Reviewed per-cell file locations are listed in reviewed_cell_annotation_index.csv.')
    if combined_unresolved_cluster_candidates.empty:
        unresolved_cluster_review = pd.DataFrame()
    return {name: value for name, value in [('OUTPUT_DIR', OUTPUT_DIR), ('RECLUSTERING_DIR', RECLUSTERING_DIR), ('REVIEW_DECISIONS_PATH', REVIEW_DECISIONS_PATH), ('MARKER_SCORE_REVIEW_DIR', MARKER_SCORE_REVIEW_DIR), ('MAX_UNRESOLVED_UMAP_CELLS', MAX_UNRESOLVED_UMAP_CELLS), ('RANDOM_STATE', RANDOM_STATE), ('combined_unresolved_cluster_candidates', combined_unresolved_cluster_candidates), ('leading_unresolved_candidates', leading_unresolved_candidates), ('unresolved_cluster_review', unresolved_cluster_review)]}

def finalize_labels(root):
    from pathlib import Path
    import pandas as pd
    from xenium_pipeline.cell_identity_evidence import clean_author_refined_label, collapse_label
    from xenium_pipeline.automatic_cluster_annotation import cell_label_masks
    from xenium_pipeline.cluster_review import MISSING_CLUSTER_ASSIGNMENT_LABELS
    from xenium_pipeline.uncertain_reclustering import author_label_usable_mask
    from xenium_pipeline.final_cell_labels import PPTX_BROAD_COLORS, pptx_final_broad_label
    ROOT = Path(root).expanduser().resolve()
    INITIAL_DIR = ROOT / 'automatic_cluster_annotation'
    INITIAL_CLUSTER_KEY = 'leiden_r1_3'
    RECLUSTER_DIR = ROOT / 'reclustered_cluster_annotation'
    RECLUSTER_KEY = 'recluster_r0_7'
    AUTHOR_DIR = ROOT / 'uncertain_cell_reclustering'
    OUTPUT_DIR = output_path(ROOT, 'reference_labels')
    ids = ['dataset', 'sample', 'cell_id']
    initial_columns = ['initial_cluster_refined_label', 'initial_cluster_broad_label']
    recluster_columns = ['recluster_refined_label', 'recluster_broad_label']
    recluster_evidence = [RECLUSTER_KEY, 'automatic_cluster_label', 'automatic_label_confidence', 'manual_decision_status', 'manual_selected_broad_label']
    author_datasets = {'Janesick_2023', 'Janesick_2025'}
    samples = pd.read_csv(INITIAL_DIR / 'cell_automatic_annotation_index.csv', usecols=['dataset', 'sample'], dtype=str)
    for dataset, study_samples in samples.groupby('dataset', sort=False):
        reviewed = None
        if dataset not in author_datasets:
            reviewed = pd.read_csv(RECLUSTER_DIR / dataset / RECLUSTER_KEY / 'cell_recluster_annotations_reviewed.csv.gz', usecols=ids + recluster_columns + recluster_evidence, dtype=str, keep_default_na=False)
        for sample in study_samples['sample']:
            label_column = f'{INITIAL_CLUSTER_KEY}_automatic_label'
            confidence_column = f'{INITIAL_CLUSTER_KEY}_automatic_confidence'
            initial = pd.read_csv(INITIAL_DIR / dataset / sample / 'cell_automatic_annotations.csv.gz', usecols=ids + [INITIAL_CLUSTER_KEY, label_column, confidence_column], dtype=str, keep_default_na=False)
            labels = initial[ids].copy()
            raw_label = initial[label_column].str.strip()
            confidence = initial[confidence_column].str.strip().str.casefold()
            has_cluster = ~initial[INITIAL_CLUSTER_KEY].str.strip().str.casefold().isin(MISSING_CLUSTER_ASSIGNMENT_LABELS)
            available, biological = cell_label_masks(raw_label)
            available &= ~raw_label.str.casefold().isin({'missing', 'none', 'nan', '<na>'})
            certain = has_cluster & available & biological & confidence.isin({'moderate', 'high'})
            labels['initial_cluster_refined_label'] = ''
            labels.loc[has_cluster & available, 'initial_cluster_refined_label'] = 'Uncertain'
            labels.loc[certain, 'initial_cluster_refined_label'] = raw_label[certain]
            labels['initial_cluster_broad_label'] = ''
            if dataset in author_datasets:
                author = pd.read_csv(AUTHOR_DIR / dataset / sample / 'cell_label_hierarchy.csv.gz', usecols=ids + ['author_label'], dtype=str, keep_default_na=False)
                author['author_refined_label'] = author['author_label'].map(clean_author_refined_label)
                unavailable = author['author_refined_label'].str.strip().str.casefold().isin({'', 'missing', 'none', 'nan', '<na>', 'not available'})
                author.loc[unavailable, 'author_refined_label'] = ''
                author['author_broad_label'] = author['author_refined_label'].map({label: broad_type(label) for label in author['author_refined_label'].unique()})
                author_columns = ['author_refined_label', 'author_broad_label']
                labels = labels.merge(author[ids + author_columns], on=ids, how='left', validate='one_to_one')
                labels = labels[ids + author_columns + initial_columns]
            else:
                labels = labels.merge(reviewed, on=ids, how='left', validate='one_to_one')
                labels = labels[ids + initial_columns + recluster_columns + recluster_evidence]
            labels = labels.fillna('')
            for broad_column in ['initial_cluster_broad_label', 'author_broad_label' if dataset in author_datasets else 'recluster_broad_label']:
                refined_column = broad_column.replace('_broad_label', '_refined_label')
                pairs = labels[[refined_column, broad_column]].fillna('').drop_duplicates()
                mapping = {(refined, broad): broad_type(refined, broad) for refined, broad in pairs.itertuples(index=False, name=None)}
                labels[broad_column] = [mapping[refined, broad] for refined, broad in labels[[refined_column, broad_column]].fillna('').itertuples(index=False, name=None)]
            labels = select_final_labels(labels, dataset)
            source_columns = author_columns + initial_columns if dataset in author_datasets else initial_columns + recluster_columns
            labels = labels[ids + source_columns + ['final_refined_label', 'final_broad_label', 'final_label_source']]
            path = OUTPUT_DIR / dataset / sample / 'final_cell_labels.csv.gz'
            path.parent.mkdir(parents=True, exist_ok=True)
            labels.to_csv(path, index=False, compression='gzip')
            mapping = labels.groupby(ids[:2] + ['final_refined_label', 'final_broad_label', 'final_label_source'], dropna=False).size().rename('n_cells').reset_index()
            mapping['training_target'] = mapping['final_broad_label'].replace({'B Cells': 'pan-APC + B', 'pan-APCs': 'pan-APC + B', 'Uncertain': 'None'})
            mapping['denominator_only'] = mapping['final_broad_label'].eq('Uncertain')
            mapping.to_csv(path.with_name('final_refined_to_target_mapping.csv'), index=False)
            print(f'{dataset} / {sample}: {len(labels):,} cells -> {path}')
    plot_final_labels(OUTPUT_DIR)
    return OUTPUT_DIR

def plot_final_labels(output_dir):
    from pathlib import Path
    OUTPUT_DIR = Path(output_dir)
    ROOT = OUTPUT_DIR.parent
    from collections import Counter
    from pathlib import Path
    from io import BytesIO
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, StrMethodFormatter
    from textwrap import fill
    from xenium_pipeline.workflow_output import Image, FileLink, display
    OUTPUT_DIR = Path(OUTPUT_DIR)
    from xenium_pipeline.final_cell_labels import PPTX_BROAD_COLORS
    BROAD_COLORS = PPTX_BROAD_COLORS.copy()
    distribution_rows = []
    label_paths = sorted(OUTPUT_DIR.glob('*/*/final_cell_labels.csv.gz'))
    for path in label_paths:
        counts = Counter()
        for chunk in pd.read_csv(path, usecols=['final_broad_label'], dtype=str, keep_default_na=False, chunksize=200000):
            counts.update(chunk['final_broad_label'].str.strip().replace('', 'Uncertain'))
        unexpected = set(counts) - set(BROAD_COLORS)
        if unexpected:
            raise ValueError(f'Unexpected broad labels in {path}: {sorted(unexpected)}')
        total = sum(counts.values())
        if total == 0:
            print(f'No cells to plot in {path}')
            continue
        for broad in BROAD_COLORS:
            distribution_rows.append({'dataset': path.parent.parent.name, 'sample': path.parent.name, 'broad_cell_type': broad, 'n_cells': counts[broad], 'n_sample_cells': total, 'fraction_of_sample': counts[broad] / total})
    broad_distribution = pd.DataFrame(distribution_rows)
    if broad_distribution.empty:
        print(f'No final-label cells found under {OUTPUT_DIR}. Generate cell labels first.')
    else:
        distribution_path = OUTPUT_DIR / 'final_broad_cell_type_distribution.csv'
        broad_distribution.to_csv(distribution_path, index=False)
        figure_dir = OUTPUT_DIR / 'figures'
        figure_dir.mkdir(parents=True, exist_ok=True)
        for dataset, rows in broad_distribution.groupby('dataset', sort=False):
            counts = rows.pivot(index='sample', columns='broad_cell_type', values='n_cells')
            counts = counts.reindex(columns=list(BROAD_COLORS)).sort_index()
            fractions = counts.div(counts.sum(axis=1), axis=0)
            y = np.arange(len(counts))
            figure, axes = plt.subplots(1, 2, sharey=True, figsize=(16, max(4.2, 0.62 * len(counts) + 2.4)))
            for axis, values, title in zip(axes, [counts, fractions * 100], ['Cell counts', 'Sample composition']):
                left = np.zeros(len(counts))
                for broad, color in BROAD_COLORS.items():
                    axis.barh(y, values[broad], left=left, color=color, height=0.7, label=broad)
                    left += values[broad].to_numpy()
                axis.set_title(title)
                axis.set_axisbelow(True)
                axis.grid(axis='x', alpha=0.2)
                axis.spines[['top', 'right']].set_visible(False)
            axes[0].set_yticks(y)
            axes[0].set_yticklabels([fill(sample, 46) for sample in counts.index], fontsize=9)
            axes[0].invert_yaxis()
            axes[0].set_xlabel('Number of cells')
            axes[0].xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
            axes[1].set_xlim(0, 100)
            axes[1].set_xlabel('Percentage of all sample cells')
            axes[1].xaxis.set_major_formatter(PercentFormatter(xmax=100))
            figure.suptitle(f'{dataset}: final broad cell types', fontsize=13)
            handles, legend_labels = axes[0].get_legend_handles_labels()
            figure.legend(handles, legend_labels, loc='lower center', ncol=4, frameon=False, fontsize=9)
            figure.tight_layout(rect=(0, 0.1, 1, 0.95))
            figure_path = figure_dir / f'{dataset}_final_broad_distribution.png'
            figure.savefig(figure_path, dpi=180, bbox_inches='tight')
            buffer = BytesIO()
            figure.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
            display(Image(data=buffer.getvalue()))
            plt.close(figure)
            display(FileLink(str(figure_path)))
        display(FileLink(str(distribution_path)))

def export_reference_targets(root):
    from pathlib import Path
    import importlib
    from xenium_pipeline.workflow_output import Image, FileLink, display
    from xenium_pipeline import retained_final_labels
    from xenium_pipeline.final_cell_labels import PPTX_BROAD_COLORS
    RETAINED_ROOT = Path(Path(root)).resolve()
    RETAINED_LABEL_DIR = data_path(RETAINED_ROOT, 'reference_labels')
    RETAINED_SELECTION_MANIFEST = RETAINED_ROOT / 'tile_filter_cutoff_tests' / 'final_automatic_tile_selection_manifest.csv'
    RETAINED_ANNOTATION_DIR = output_path(RETAINED_ROOT, 'reference_annotations')
    RETAINED_OUTPUT_DIR = RETAINED_LABEL_DIR / 'retained_tiles'
    retained_broad_distribution, retained_label_audit = retained_final_labels.summarize_retained_final_labels(RETAINED_ROOT, RETAINED_LABEL_DIR, RETAINED_SELECTION_MANIFEST, annotation_dir=RETAINED_ANNOTATION_DIR, broad_colors=PPTX_BROAD_COLORS)
    RETAINED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    retained_audit_path = RETAINED_OUTPUT_DIR / 'retained_final_label_membership_audit.csv'
    retained_label_audit.to_csv(retained_audit_path, index=False)
    display(FileLink(str(retained_audit_path)))
    if retained_label_audit['status'].eq('failed').any():
        raise ValueError(f"{retained_label_audit['status'].eq('failed').sum()} sample(s) failed validation. See status/error in {retained_audit_path}. No figures were generated in this run; any existing retained-tile figures are from a previous run.")
    if retained_broad_distribution.empty:
        print('No retained cells are available for plotting; see the saved audit.')
    else:
        retained_distribution_path = RETAINED_OUTPUT_DIR / 'retained_final_broad_cell_type_distribution.csv'
        retained_broad_distribution.to_csv(retained_distribution_path, index=False)
        display(FileLink(str(retained_distribution_path)))
        retained_figure_paths = retained_final_labels.plot_retained_distribution(retained_broad_distribution, RETAINED_OUTPUT_DIR / 'figures', broad_colors=PPTX_BROAD_COLORS)
        for retained_figure_path in retained_figure_paths:
            display(Image(filename=str(retained_figure_path), width=1200))
            display(FileLink(str(retained_figure_path)))
        print(f"{len(retained_figure_paths)} study figure(s) saved under {RETAINED_OUTPUT_DIR / 'figures'}")
    return RETAINED_ANNOTATION_DIR

def apply_epithelial_review(root):
    from pathlib import Path
    import importlib
    import pandas as pd
    from xenium_pipeline.workflow_output import display, Image, FileLink
    from xenium_pipeline import cell_label_review
    ROOT = Path(root).expanduser().resolve()
    BASELINE_DIR = data_path(ROOT, 'reference_labels')
    SAMPLE_INDEX = ROOT / 'automatic_cluster_annotation' / 'cell_automatic_annotation_index.csv'
    ANNOTATION_DIR = ROOT / 'epithelial_cluster_annotation'
    MANUAL_DECISIONS = ROOT / 'epithelial_morphology_review' / 'manual_cell_review' / 'manual_epithelial_decisions.csv'
    JANESICK_AUTHOR_WORKBOOK = ROOT / 'Janesick_2023' / 'Cell_Barcode_Type_Matrices.xlsx'
    CLUSTER_KEY = 'epithelial_r0_7'
    OUTPUT_DIR = output_path(ROOT, 'epithelial_labels')
    REVIEWED_ANNOTATION_DIR = OUTPUT_DIR / 'final_cell_annotations'
    RUN_RETAINED_TILE_EXPORT = True
    if not JANESICK_AUTHOR_WORKBOOK.is_file():
        raise FileNotFoundError(f'Missing original author workbook: {JANESICK_AUTHOR_WORKBOOK}')
    cluster_plan = cell_label_review.load_cluster_updates(ANNOTATION_DIR / 'all_cluster_annotations.csv', MANUAL_DECISIONS, CLUSTER_KEY)
    inventory = cell_label_review.input_inventory(BASELINE_DIR, ANNOTATION_DIR, cluster_plan, SAMPLE_INDEX)
    print(f"Clusters: {len(cluster_plan):,}; cells targeted: {cluster_plan.n_cells.sum():,}; affected samples: {cluster_plan[['dataset', 'sample']].drop_duplicates().shape[0]}")
    print(f'Reference label files found: {inventory.baseline_exists.sum()}/{len(inventory)}')
    display(inventory.groupby(['dataset', 'status']).size().rename('n_samples').reset_index())
    print(f'Baseline: {BASELINE_DIR}')
    print(f'Cell-to-cluster assignments: {ANNOTATION_DIR}/<dataset>/<sample>/cell_annotations.csv.gz')
    if not inventory.status.eq('ready').all():
        print('Provide reference labels and per-cell epithelial assignments before applying review.')
    plan_summary = cluster_plan.groupby(['update_source', 'updated_broad_label'], sort=False).agg(n_clusters=('cluster', 'size'), n_cells=('n_cells', 'sum')).reset_index()
    display(plan_summary)
    manual_rows = cluster_plan.he_manual_label.ne('')
    print(f'Manual decisions: {manual_rows.sum()}; expression-only clusters: {(~manual_rows).sum()}')
    print(f"Manual decisions with alignment unconfirmed: {(manual_rows & cluster_plan.he_alignment_confirmed.str.lower().eq('false')).sum()}")
    update_summary = cell_label_review.export_updated_labels(BASELINE_DIR, ANNOTATION_DIR, MANUAL_DECISIONS, OUTPUT_DIR, CLUSTER_KEY, sample_index=SAMPLE_INDEX, janesick_author_workbook=JANESICK_AUTHOR_WORKBOOK)
    display(update_summary.groupby('dataset')[['n_cells', 'n_updated_clusters', 'n_updated_cells', 'n_changed_labels', 'n_final_uncertain']].sum())
    display(update_summary.loc[update_summary.dataset.isin(['Janesick_2023', 'Janesick_2025'])].groupby('dataset')[['n_janesick_epithelial_override_cells', 'n_janesick_uncertain_fallback_cells', 'n_janesick_author_labeled_cells', 'n_janesick_author_missing_cells']].sum())
    print(f'Updated labels: {OUTPUT_DIR}/<dataset>/<sample>/final_cell_labels.csv.gz')
    print(f'Reference labels: {BASELINE_DIR}')
    for name in ['epithelial_update_summary.csv', 'cluster_update_plan.csv', 'input_inventory.csv', 'broad_label_transitions.csv', 'janesick_author_label_mapping.csv', 'update_manifest.json']:
        display(FileLink(str(OUTPUT_DIR / name)))
    distribution_path = OUTPUT_DIR / 'final_broad_cell_type_distribution.csv'
    distribution = pd.read_csv(distribution_path)
    for path in cell_label_review.plot_distributions(distribution, OUTPUT_DIR / 'figures'):
        display(Image(filename=str(path), width=1200))
        display(FileLink(str(path)))
    display(FileLink(str(distribution_path)))
    from xenium_pipeline import retained_final_labels
    selection_manifest = ROOT / 'tile_filter_cutoff_tests' / 'final_automatic_tile_selection_manifest.csv'
    retained_dir = OUTPUT_DIR / 'retained_tiles'
    retained, audit = retained_final_labels.summarize_retained_final_labels(ROOT, OUTPUT_DIR, selection_manifest, annotation_dir=REVIEWED_ANNOTATION_DIR)
    retained_dir.mkdir(parents=True, exist_ok=True)
    audit_path = retained_dir / 'retained_final_label_membership_audit.csv'
    audit.to_csv(audit_path, index=False)
    display(FileLink(str(audit_path)))
    if audit.status.eq('failed').any():
        raise ValueError('Retained-tile validation failed; see the saved audit. Existing figures may be stale.')
    if not retained.empty:
        retained.to_csv(retained_dir / 'retained_final_broad_cell_type_distribution.csv', index=False)
        for path in retained_final_labels.plot_retained_distribution(retained, retained_dir / 'figures'):
            display(Image(filename=str(path), width=1200))
            display(FileLink(str(path)))
        print(f'Reviewed training annotations: {REVIEWED_ANNOTATION_DIR}/<dataset>/<sample>/cell_annotations.csv.gz')
    return OUTPUT_DIR

def launch_recluster_review(root):
    from xenium_pipeline import reclustered_cluster_annotation
    import numpy as np
    import pandas as pd
    from IPython.display import display, Markdown
    state = annotate_reclusters(root, run=False)
    OUTPUT_DIR = state['OUTPUT_DIR']
    RECLUSTERING_DIR = state['RECLUSTERING_DIR']
    REVIEW_DECISIONS_PATH = state['REVIEW_DECISIONS_PATH']
    MARKER_SCORE_REVIEW_DIR = state['MARKER_SCORE_REVIEW_DIR']
    MAX_UNRESOLVED_UMAP_CELLS = state['MAX_UNRESOLVED_UMAP_CELLS']
    RANDOM_STATE = state['RANDOM_STATE']
    combined_unresolved_cluster_candidates = state['combined_unresolved_cluster_candidates']
    leading_unresolved_candidates = state['leading_unresolved_candidates']
    unresolved_cluster_review = state['unresolved_cluster_review']
    if not leading_unresolved_candidates.empty:
        try:
            import ipywidgets as widgets
            import matplotlib.pyplot as plt
            from io import BytesIO
            from matplotlib.lines import Line2D
            from textwrap import fill
            from IPython.display import Image
            unresolved_index = leading_unresolved_candidates[['dataset', 'cluster_key', 'cluster']].drop_duplicates()
            unresolved_options = [(f'{row.dataset} | {row.cluster_key} | cluster {row.cluster}', f'{row.dataset}|||{row.cluster_key}|||{row.cluster}') for row in unresolved_index.itertuples(index=False)]
            unresolved_dropdown = widgets.Dropdown(options=unresolved_options, description='Cluster:', layout=widgets.Layout(width='70%'))
            review_class_dropdown = widgets.Dropdown(options=[('All unresolved', 'all'), ('Likely epithelial', 'likely_epithelial'), ('Likely other', 'likely_other'), ('Mixed or insufficient', 'mixed_or_insufficient')], description='Review:')
            unresolved_output = widgets.Output()
            umap_assignment_cache = {}
            marker_score_review_cache = {}
            decision_label = widgets.Dropdown(options=reclustered_cluster_annotation.REVIEW_BROAD_LABELS, value='Uncertain', description='Your label:', layout=widgets.Layout(width='90%'))
            decision_status = widgets.Dropdown(options=['needs_review', 'tentative', 'accepted'], description='Status:')
            decision_notes = widgets.Textarea(description='Notes:', layout=widgets.Layout(width='90%', height='90px'))
            save_decision_button = widgets.Button(description='Save decision', icon='save', button_style='success')
            decision_message = widgets.Output()
            decision_state = {'key': None, 'baseline': None, 'drafts': {}}

            def decision_values():
                return {'selected_label': decision_label.value, 'decision_status': decision_status.value, 'notes': decision_notes.value}

            def load_selected_decision():
                old_key = decision_state['key']
                if old_key is not None and decision_values() != decision_state['baseline']:
                    decision_state['drafts'][old_key] = decision_values()
                elif old_key is not None:
                    decision_state['drafts'].pop(old_key, None)
                decision_state['key'] = None
                save_decision_button.disabled = True
                if unresolved_dropdown.value is None:
                    decision_label.value = 'Uncertain'
                    decision_notes.value = ''
                    decision_status.value = 'needs_review'
                    with decision_message:
                        decision_message.clear_output(wait=True)
                        print('Select a cluster to record a decision.')
                    return
                key = tuple(unresolved_dropdown.value.split('|||', 2))
                try:
                    saved = reclustered_cluster_annotation.load_recluster_review_decisions(REVIEW_DECISIONS_PATH)
                except (OSError, ValueError) as error:
                    with decision_message:
                        decision_message.clear_output(wait=True)
                        print(f'Decision CSV could not be loaded; saving is disabled: {error}')
                    return
                saved_mask = pd.Series(True, index=saved.index)
                for column, value in zip(['dataset', 'cluster_key', 'cluster'], key):
                    saved_mask &= saved[column].eq(value)
                rows = saved.loc[saved_mask]
                values = {'selected_label': 'Uncertain', 'decision_status': 'needs_review', 'notes': ''}
                if not rows.empty:
                    values.update(rows.iloc[0][list(values)].to_dict())
                unsupported_label = None
                if values['selected_label'] not in decision_label.options:
                    unsupported_label = values['selected_label']
                    values.update(selected_label='Uncertain', decision_status='needs_review')
                baseline = values.copy()
                draft = decision_state['drafts'].get(key)
                if draft is not None:
                    values.update(draft)
                decision_label.value = values['selected_label']
                decision_status.value = values['decision_status']
                decision_notes.value = values['notes']
                decision_state.update(key=key, baseline=baseline)
                save_decision_button.disabled = False
                with decision_message:
                    decision_message.clear_output(wait=True)
                    if unsupported_label is not None:
                        print(f'Previously saved label: {unsupported_label!r}. Select a broad class; the CSV is unchanged until Save decision.')
                    if draft is not None:
                        print('Unsaved draft restored. Click Save decision to write it to CSV.')
                    elif not rows.empty:
                        print(f"Loaded saved decision ({rows.iloc[0]['updated_at_utc']}).")
                    else:
                        print('No decision saved for this cluster yet.')
                    print(f'Decision CSV: {REVIEW_DECISIONS_PATH}')

            def save_selected_decision(_button):
                with decision_message:
                    decision_message.clear_output(wait=True)
                    key = decision_state['key']
                    if key is None or unresolved_dropdown.value != '|||'.join(key):
                        print('Select a cluster before saving.')
                        return
                    study, cluster_key, cluster = key
                    context = unresolved_cluster_review.loc[unresolved_cluster_review['dataset'].astype(str).eq(study) & unresolved_cluster_review['cluster_key'].astype(str).eq(cluster_key) & unresolved_cluster_review['cluster'].astype(str).eq(cluster)].iloc[0]
                    try:
                        table = reclustered_cluster_annotation.save_recluster_review_decision(REVIEW_DECISIONS_PATH, {'dataset': study, 'cluster_key': cluster_key, 'cluster': cluster, 'review_class': context['review_class'], **decision_values()})
                    except (OSError, ValueError) as error:
                        print(f'Decision was not saved: {error}')
                        return
                    decision_state['baseline'] = decision_values()
                    decision_state['drafts'].pop(key, None)
                    print(f'Saved {study} / {cluster_key} / cluster {cluster}.')
                    print('Export recluster labels to apply saved decisions.')
                    print(f'Decision CSV: {REVIEW_DECISIONS_PATH} ({len(table):,} saved decisions)')
            save_decision_button.on_click(save_selected_decision)

            def load_study_umap(study, cluster_key):
                cache_key = (study, cluster_key)
                if cache_key in umap_assignment_cache:
                    return (umap_assignment_cache[cache_key], '')
                assignment_path = RECLUSTERING_DIR / reclustered_cluster_annotation.safe_name(study) / 'study_recluster_assignments.csv.gz'
                if not assignment_path.is_file():
                    return (pd.DataFrame(), f'UMAP assignment file not found: {assignment_path}')
                required_columns = {'sample', 'umap_1', 'umap_2', cluster_key}
                available_columns = set(pd.read_csv(assignment_path, nrows=0).columns)
                missing_columns = sorted(required_columns - available_columns)
                if missing_columns:
                    return (pd.DataFrame(), f"UMAP cannot be displayed because the assignment file lacks: {', '.join(missing_columns)}")
                table = pd.read_csv(assignment_path, usecols=['sample', 'umap_1', 'umap_2', cluster_key], dtype={'sample': 'string', cluster_key: 'string'}, low_memory=False).dropna(subset=['umap_1', 'umap_2'])
                table[cluster_key] = table[cluster_key].astype(str)
                table['sample'] = table['sample'].fillna('Missing sample').astype(str)
                umap_assignment_cache[cache_key] = table
                return (table, '')

            def show_marker_score_distribution(study, cluster_key, cluster):
                cache_key = (study, cluster_key, str(cluster), str(MARKER_SCORE_REVIEW_DIR))
                try:
                    if cache_key not in marker_score_review_cache:
                        assignment_path = RECLUSTERING_DIR / reclustered_cluster_annotation.safe_name(study) / 'study_recluster_assignments.csv.gz'
                        marker_score_review_cache.clear()
                        marker_score_review_cache[cache_key] = reclustered_cluster_annotation.load_recluster_marker_score_review(assignment_path, MARKER_SCORE_REVIEW_DIR, study, cluster_key, str(cluster))
                    scores, totals = marker_score_review_cache[cache_key]
                    figure, coverage = reclustered_cluster_annotation.plot_recluster_marker_scores(scores, totals, cluster_key, cluster)
                    display(coverage)
                    if figure is None:
                        print('Full score rows were loaded, but this cluster has no finite marker scores.')
                        print('This is missing score coverage, not evidence of absent marker expression.')
                        return
                    folder = OUTPUT_DIR / reclustered_cluster_annotation.safe_name(study) / cluster_key / 'marker_score_review'
                    folder.mkdir(parents=True, exist_ok=True)
                    basename = f'cluster_{reclustered_cluster_annotation.safe_name(str(cluster))}'
                    figure_path = folder / f'{basename}_score_distributions.png'
                    coverage_path = folder / f'{basename}_score_coverage.csv'
                    figure.savefig(figure_path, dpi=180, bbox_inches='tight')
                    coverage.to_csv(coverage_path, index=False)
                    display_review_figure(figure)
                    from IPython.display import FileLink
                    display(FileLink(str(figure_path)))
                    print(f'Coverage table: {coverage_path}')
                    print('All assigned cluster cells included; no subsampling. Scores are not probabilities or independent validation.')
                except (OSError, ValueError, KeyError) as error:
                    print(f'Marker-score distribution unavailable: {error}')
                    print(f'Full marker-score matrices are required under: {MARKER_SCORE_REVIEW_DIR}')
                    print('Run the score command to generate the full marker-score matrices.')
                    print('Other manual-review evidence remains available.')

            def selected_cluster_umap(study, cluster_key, cluster):
                table, error = load_study_umap(study, cluster_key)
                if table.empty:
                    return (None, error or 'No cells with saved UMAP coordinates are available.')
                selected = table.loc[table[cluster_key].eq(str(cluster))]
                if selected.empty:
                    return (None, f'Cluster {cluster} has no saved UMAP coordinates.')
                background = table.loc[~table[cluster_key].eq(str(cluster))]
                max_cells = max(int(MAX_UNRESOLVED_UMAP_CELLS), 1)
                selected_plot = selected.sample(max_cells, random_state=RANDOM_STATE) if len(selected) > max_cells else selected
                remaining = max_cells - len(selected_plot)
                background_plot = background.sample(remaining, random_state=RANDOM_STATE) if remaining > 0 and len(background) > remaining else background.iloc[:remaining]
                plot = pd.concat([background_plot, selected_plot]).sample(frac=1, random_state=RANDOM_STATE)
                samples = sorted(table['sample'].unique(), key=reclustered_cluster_annotation.natural_key)
                palette = plt.get_cmap('tab20' if len(samples) <= 20 else 'gist_rainbow', len(samples))
                sample_colors = {sample: palette(index) for index, sample in enumerate(samples)}
                plotted_counts = plot['sample'].value_counts()
                sample_legend_labels = [fill(f'{sample} (n={plotted_counts.get(sample, 0):,} plotted)', width=48) for sample in samples]
                legend_columns = min(2, len(samples))
                legend_rows = (len(samples) + legend_columns - 1) // legend_columns
                legend_height = 0.5 + legend_rows * max((label.count('\n') + 1 for label in sample_legend_labels)) * 0.18
                figure, (axis, sample_axis) = plt.subplots(1, 2, figsize=(12, 5.8 + legend_height), sharex=True, sharey=True)
                if not background_plot.empty:
                    axis.scatter(background_plot['umap_1'], background_plot['umap_2'], s=2, c='#A8A8A8', alpha=0.65, linewidths=0, rasterized=True, label='Other clusters')
                axis.scatter(selected_plot['umap_1'], selected_plot['umap_2'], s=4, c='#C7352A', alpha=0.85, linewidths=0, rasterized=True, label=f'Cluster {cluster}')
                center_x = selected_plot['umap_1'].median()
                center_y = selected_plot['umap_2'].median()
                axis.annotate(str(cluster), (center_x, center_y), xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold', color='#8F1D17')
                axis.set(title=f'Selected cluster {cluster}', xlabel='UMAP 1', ylabel='UMAP 2')
                axis.legend(frameon=False, markerscale=3, loc='best')
                sample_axis.scatter(plot['umap_1'], plot['umap_2'], c=[sample_colors[sample] for sample in plot['sample']], s=3, alpha=0.75, linewidths=0, rasterized=True)
                sample_axis.set(title='Colored by sample', xlabel='UMAP 1')
                for current_axis in (axis, sample_axis):
                    for spine in ('top', 'right'):
                        current_axis.spines[spine].set_visible(False)
                for coordinate, setter in [('umap_1', axis.set_xlim), ('umap_2', axis.set_ylim)]:
                    low, high = (table[coordinate].min(), table[coordinate].max())
                    padding = max((high - low) * 0.04, 0.01)
                    setter(low - padding, high + padding)
                handles = [Line2D([], [], marker='o', linestyle='', color=sample_colors[sample], markersize=5) for sample in samples]
                figure.legend(handles, sample_legend_labels, loc='lower center', ncol=legend_columns, frameon=False, fontsize=8, title='Sample')
                figure.suptitle(f'{study} | {cluster_key}')
                figure.tight_layout(rect=(0, legend_height / (5.8 + legend_height), 1, 0.95))
                note = f'UMAP includes {len(table):,} directly clustered cells with coordinates; {len(selected):,} belong to cluster {cluster}. {len(background_plot) + len(selected_plot):,} points are displayed.'
                return (figure, note)

            def display_review_figure(figure):
                try:
                    buffer = BytesIO()
                    figure.savefig(buffer, format='png', dpi=140, bbox_inches='tight')
                    display(Image(data=buffer.getvalue()))
                finally:
                    plt.close(figure)

            def show_sample_composition(study, cluster_key, cluster):
                composition_path = OUTPUT_DIR / reclustered_cluster_annotation.safe_name(study) / reclustered_cluster_annotation.safe_name(cluster_key) / 'cluster_sample_composition.csv'
                if not composition_path.is_file():
                    print(f'Sample composition is unavailable: {composition_path}')
                    return
                composition = pd.read_csv(composition_path, dtype={'cluster': 'string'})
                composition = composition.loc[composition['cluster'].eq(str(cluster))].sort_values('fraction_of_cluster', ascending=True)
                if composition.empty:
                    print('No saved sample-composition rows for this cluster.')
                    return
                figure, axes = plt.subplots(1, 2, figsize=(11, max(3, len(composition) * 0.28)), sharey=True)
                positions = np.arange(len(composition))
                axes[0].barh(positions, composition['fraction_of_cluster'], color='#247D99')
                axes[0].set_yticks(positions)
                axes[0].set_yticklabels(composition['sample'])
                axes[0].set(xlabel='Fraction of cluster cells', xlim=(0, 1))
                axes[1].barh(positions, composition['sample_enrichment'], color='#80538C')
                axes[1].axvline(1, color='#555555', linestyle='--', linewidth=1)
                axes[1].set(xlabel='Enrichment relative to study targets')
                figure.suptitle(f'{study} | cluster {cluster} | sample composition')
                figure.tight_layout()
                display_review_figure(figure)

            def show_unresolved_cluster_candidates(*_args):
                load_selected_decision()
                if unresolved_dropdown.value is None:
                    with unresolved_output:
                        unresolved_output.clear_output(wait=True)
                        print('No unresolved clusters in this review class.')
                    return
                study, cluster_key, cluster = unresolved_dropdown.value.split('|||', 2)
                candidates = combined_unresolved_cluster_candidates.loc[combined_unresolved_cluster_candidates['dataset'].astype(str).eq(study) & combined_unresolved_cluster_candidates['cluster_key'].astype(str).eq(cluster_key) & combined_unresolved_cluster_candidates['cluster'].astype(str).eq(cluster)].sort_values('candidate_rank')
                marker_path = OUTPUT_DIR / reclustered_cluster_annotation.safe_name(study) / reclustered_cluster_annotation.safe_name(cluster_key) / 'expression_markers_top_with_gene_hints.csv'
                markers = pd.read_csv(marker_path, dtype={'cluster': 'string'}, low_memory=False) if marker_path.is_file() else pd.DataFrame()
                if not markers.empty:
                    markers = markers.loc[markers['cluster'].astype(str).eq(cluster)].head(15)
                candidate_columns = ['candidate_rank', 'candidate_label', 'candidate_review_lineage', 'candidate_evidence_tier', 'n_methods_nominating', 'n_qualifying_methods_nominating', 'nominating_methods', 'qualifying_methods', 'marker_score_candidate_fraction', 'marker_score_fraction_gap', 'assigned_score_fraction', 'signature_median_score', 'signature_score_gap', 'signature_coverage', 'signature_positive_cell_fraction', 'gene_hint_support_fraction', 'gene_hint_support_gap', 'gene_hint_n_supporting_genes', 'gene_hint_supporting_genes', 'sample_specificity_category', 'dominant_sample', 'dominant_sample_enrichment', 'jensen_shannon_divergence_from_study_baseline', 'automatic_label_reason']
                marker_columns = ['rank_for_annotation', 'gene', 'log2_fold_change', 'fraction_expressing_cluster', 'fraction_expressing_rest', 'fraction_expressing_difference', 'wilcoxon_pval_adj', 'gene_marker_groups', 'gene_marker_evidence_type']
                with unresolved_output:
                    unresolved_output.clear_output(wait=True)
                    context = unresolved_cluster_review.loc[unresolved_cluster_review['dataset'].astype(str).eq(study) & unresolved_cluster_review['cluster_key'].astype(str).eq(cluster_key) & unresolved_cluster_review['cluster'].astype(str).eq(cluster)]
                    display(Markdown('**Review pathway and sample-sharing context**'))
                    display(context[['review_class', 'provisional_review_label', 'review_reason', 'n_cluster_cells', 'sample_specificity_category', 'dominant_sample', 'dominant_sample_enrichment']].reset_index(drop=True))
                    display(Markdown('**Ranked label candidates**'))
                    display(candidates[[column for column in candidate_columns if column in candidates]])
                    display(Markdown('**Marker-score distributions for this cluster**'))
                    show_marker_score_distribution(study, cluster_key, cluster)
                    display(Markdown('**UMAP location of the selected unresolved cluster**'))
                    umap_figure, umap_note = selected_cluster_umap(study, cluster_key, cluster)
                    if umap_figure is None:
                        print(umap_note)
                    else:
                        display_review_figure(umap_figure)
                        print(umap_note)
                    show_sample_composition(study, cluster_key, cluster)
                    display(Markdown('**Top differential-expression markers**'))
                    if markers.empty:
                        print(f'No marker table found for {study} / {cluster_key} / cluster {cluster}.')
                    else:
                        display(markers[[column for column in marker_columns if column in markers]])

            def filter_unresolved_clusters(*_args):
                selected_class = review_class_dropdown.value
                review_subset = unresolved_cluster_review
                if selected_class != 'all':
                    review_subset = review_subset.loc[review_subset['review_class'].eq(selected_class)]
                unresolved_dropdown.unobserve(show_unresolved_cluster_candidates, names='value')
                unresolved_dropdown.options = [(f'{row.dataset} | {row.cluster_key} | cluster {row.cluster}', f'{row.dataset}|||{row.cluster_key}|||{row.cluster}') for row in review_subset.itertuples(index=False)]
                unresolved_dropdown.observe(show_unresolved_cluster_candidates, names='value')
                show_unresolved_cluster_candidates()
            review_class_dropdown.observe(filter_unresolved_clusters, names='value')
            unresolved_dropdown.observe(show_unresolved_cluster_candidates, names='value')
            show_unresolved_cluster_candidates()
            display(widgets.VBox([review_class_dropdown, unresolved_dropdown, decision_label, decision_status, decision_notes, save_decision_button, decision_message, unresolved_output]))
        except ImportError as error:
            print(f'Interactive cluster review dependency unavailable: {error}')
            print('Review the saved unresolved-candidate and top-marker CSV files.')
    else:
        print('No unresolved candidate clusters are available.')

def recluster_epithelial(root, datasets=None):
    from xenium_pipeline import epithelial_reclustering as er, epithelial_marker_review as em
    root = Path(root).expanduser().resolve()
    config = er.EpithelialConfig(screening_genes=('EPCAM', 'KRT8', 'KRT18', 'KRT19', 'KRT5', 'KRT14', 'KRT17'), min_screening_genes=2, min_counts_per_cell=10, min_genes_per_cell=5, min_cells_per_gene=10, normalization_target=10000, n_pcs=20, n_neighbors=20, resolutions=(0.7,), leiden_iterations=2, random_state=1729, run_umap=True, max_umap_plot_cells=100000, overwrite=False)
    selected = tuple(datasets or ('10x', 'Liu_2025', 'Bilous_2026'))
    if set(selected) - {'10x', 'Liu_2025', 'Bilous_2026'}:
        raise ValueError('Epithelial reclustering uses 10x, Liu_2025, and Bilous_2026.')
    summaries = []
    for dataset in selected:
        output = output_path(root, 'epithelial_clustering')
        inventory = er.build_inventory(root, data_path(root, 'reference_labels'), (dataset,), ())
        if inventory.empty:
            raise FileNotFoundError(f'No epithelial input samples for {dataset}')
        output.mkdir(parents=True, exist_ok=True)
        inventory.to_csv(output / f'{dataset}_input_inventory.csv', index=False)
        summary = er.run_all(inventory, output, config)
        if summary.status.eq('failed').any():
            raise RuntimeError(f'Epithelial clustering failed; see {output}')
        summaries.append(summary)
    for output in dict.fromkeys((output_path(root, 'epithelial_clustering') for d in selected)):
        marker = em.run_marker_review(output, root / 'gene_panel_review', em.MarkerConfig(min_cluster_cells=20, min_fraction_expressing=0.1, min_fraction_difference=0.1, min_log2_fold_change=0.5, lfc_pseudocount=0.1, fdr=0.05, top_n=15, overwrite=False))
        lineage = em.run_lineage_review(output, root / 'gene_panel_review', min_detected=2)
        if marker.status.eq('failed').any() or lineage.status.eq('failed').any():
            raise RuntimeError(f'Epithelial evidence failed; see {output}')
    return pd.concat(summaries, ignore_index=True)

def compute_epithelial_morphology(root):
    from xenium_pipeline import epithelial_morphology as morphology
    root = Path(root).expanduser().resolve()
    roots = epithelial_roots(root)
    output = root / 'epithelial_morphology_review'
    inventory = morphology.discover_inputs(root, roots, cache_path=output / 'input_inventory.csv')
    if inventory.empty:
        raise FileNotFoundError('No saved epithelial checkpoints found.')
    result = morphology.run_all(inventory, output, morphology.MorphologyConfig(cluster_key='epithelial_r0_7'))
    if result.status.eq('failed').any():
        raise RuntimeError(f"Morphology evidence failed; see {output / 'run_summary.csv'}")
    return result
