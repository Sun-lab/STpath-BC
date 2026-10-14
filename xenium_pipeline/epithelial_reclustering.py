"""Independent per-sample epithelial reclustering, without malignancy assignment."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import pandas as pd

from . import cell_clustering, cluster_review
from .gene_panel_utils import is_gene_expression


DATASETS = ('10x', 'Bilous_2026', 'Liu_2025')
LABEL_COLUMNS = (
    'initial_cluster_refined_label', 'initial_cluster_broad_label',
    'recluster_refined_label', 'recluster_broad_label',
    'final_refined_label', 'final_broad_label',
)


def non_gene_features(adata):
    names = adata.var.get('gene_symbol', pd.Series(adata.var_names, index=adata.var_names))
    types = adata.var.get('feature_type', pd.Series(None, index=adata.var_names, dtype=object))
    return [str(name) for kind, name in zip(types, names) if not is_gene_expression(kind, name)]


def audit_saved_features(root):
    """Inspect checkpoint feature metadata without loading expression matrices."""
    import anndata as ad
    rows = []
    for path in sorted(Path(root).glob('*/*/epithelial_expression.h5ad')):
        if path.parent.parent.name not in DATASETS:
            continue
        data = ad.read_h5ad(path, backed='r')
        try:
            excluded = non_gene_features(data)
            rows.append(dict(dataset=path.parent.parent.name, sample=path.parent.name,
                             n_features=data.n_vars, n_non_gene_features=len(excluded),
                             requires_repair=bool(excluded), checkpoint_path=str(path)))
        finally:
            data.file.close()
    return pd.DataFrame(rows, columns=['dataset', 'sample', 'n_features', 'n_non_gene_features',
                                       'requires_repair', 'checkpoint_path'])


@dataclass(frozen=True)
class EpithelialConfig:
    screening_genes: tuple[str, ...] = ('EPCAM', 'KRT8', 'KRT18', 'KRT19', 'KRT5', 'KRT14', 'KRT17')
    min_screening_genes: int = 2
    min_counts_per_cell: int = 10
    min_genes_per_cell: int = 5
    min_cells_per_gene: int = 10
    normalization_target: float = 10_000
    n_pcs: int = 20
    n_neighbors: int = 20
    resolutions: tuple[float, ...] = (0.3, 0.7, 1.3)
    random_state: int = 1729
    leiden_iterations: int = 2
    run_umap: bool = True
    max_umap_plot_cells: int = 100_000
    overwrite: bool = False


def validate_config(config):
    for key in ('min_screening_genes', 'min_counts_per_cell', 'min_genes_per_cell',
                'min_cells_per_gene', 'normalization_target', 'n_neighbors',
                'max_umap_plot_cells', 'leiden_iterations'):
        if not np.isfinite(getattr(config, key)) or getattr(config, key) <= 0:
            raise ValueError(f'{key} must be positive.')
    if config.n_pcs < 2:
        raise ValueError('n_pcs must be at least 2.')
    genes = [str(g).strip().upper() for g in config.screening_genes]
    if len(set(genes)) != len(genes) or '' in genes or len(genes) < config.min_screening_genes:
        raise ValueError('Screening genes must be unique, nonblank, and cover min_screening_genes.')
    if not config.resolutions or any(not np.isfinite(r) or r <= 0 for r in config.resolutions):
        raise ValueError('At least one positive Leiden resolution is required.')
    if len({cluster_key(r) for r in config.resolutions}) != len(config.resolutions):
        raise ValueError('Leiden resolutions must have unique output keys.')


def cluster_key(resolution):
    return cell_clustering.cluster_key_for_resolution(resolution).replace('leiden_', 'epithelial_')


def epithelial_label_mask(values):
    labels = values.fillna('').astype(str).str.replace('_', ' ', regex=False).str.casefold()
    return labels.str.contains(
        r'epithel|glandular|breast cancer|\btumou?r cells?\b|\bdcis\b|\binvasive (?:cancer|carcinoma)',
        regex=True,
    )


def select_epithelial_candidates(labels, detected_genes, minimum):
    """Inclusive selection; broad certainty is used only to limit expression rescue."""
    evidence = labels.reindex(columns=LABEL_COLUMNS, fill_value='')
    label_match = evidence.apply(epithelial_label_mask).any(axis=1)
    broad = labels['final_broad_label'].fillna('').astype(str).str.strip().str.casefold()
    unresolved = broad.isin({'', 'uncertain', 'unknown', 'unassigned', 'unlabeled', 'missing', 'nan', 'none'})
    rescued = ~label_match & unresolved & (np.asarray(detected_genes) >= minimum)
    reasons = np.select([label_match, rescued], ['epithelial_related_label', 'unresolved_epithelial_expression'],
                        default='not_selected')
    return pd.DataFrame({'epithelial_label_evidence': label_match.to_numpy(),
                         'n_screening_genes_detected': np.asarray(detected_genes, dtype=int),
                         'selected_for_reclustering': (label_match | rescued).to_numpy(),
                         'selection_reason': reasons}, index=labels.index)


def align_labels_to_matrix(labels, matrix_ids):
    if labels['cell_id'].isna().any() or labels['cell_id'].str.strip().eq('').any() or labels['cell_id'].duplicated().any():
        raise ValueError('Final-label cell IDs must be nonblank and unique within the sample.')
    matrix_ids = pd.Index(matrix_ids).astype(str)
    if matrix_ids.has_duplicates or (matrix_ids.str.strip() == '').any():
        raise ValueError('Expression matrix cell IDs must be nonblank and unique.')
    if labels.empty:
        raise ValueError('The final-label file is empty.')
    source = labels[['cell_id']].copy()
    source['label_row'] = np.arange(len(source)).astype(str)
    mapped, diagnostics = cluster_review.match_cluster_source_cell_ids(
        source, cell_id_columns=['cell_id'], cluster_column='label_row', cluster_key='label_row',
        matrix_cell_ids=matrix_ids,
    )
    if len(mapped) != len(labels) or not mapped['cell_id'].isin(matrix_ids).all():
        raise ValueError('Some final-label cells do not match the expression matrix. Check sample and matrix versions.')
    aligned = labels.iloc[mapped['label_row'].astype(int)].copy()
    aligned['cell_id'] = mapped['cell_id'].to_numpy()
    aligned['has_saved_label_row'] = True
    aligned = aligned.set_index('cell_id').reindex(matrix_ids)
    aligned['has_saved_label_row'] = aligned['has_saved_label_row'].fillna(False).astype(bool)
    for column in LABEL_COLUMNS + ('final_label_source',):
        if column not in aligned:
            aligned[column] = ''
        aligned[column] = aligned[column].fillna('').astype(str)
    return aligned, diagnostics['cell_id_match_method']


def build_inventory(root, label_dir, datasets=DATASETS, samples=()):
    """Check files without performing the biological diagnostic the user declined."""
    if set(datasets) - set(DATASETS):
        raise ValueError(f'This workflow is limited to {DATASETS}; Janesick is excluded.')
    root, label_dir = Path(root).resolve(), Path(label_dir)
    registry = cell_clustering.build_clustering_input_table(root, dataset_names=datasets,
                                                          sample_order='dataset_sample')
    if samples:
        requested = set(samples)
        known = set(zip(registry.dataset, registry['sample']))
        if requested - known:
            raise ValueError(f'Requested dataset/sample pairs absent from registry: {sorted(requested - known)}')
        registry = registry.loc[[tuple(k) in requested for k in registry[['dataset', 'sample']].to_numpy()]].copy()
    if registry.empty or registry.duplicated(['dataset', 'sample']).any():
        raise ValueError('Input registry must contain unique dataset/sample rows.')
    rows = []
    for row in registry.itertuples(index=False):
        matrix = cell_clustering.resolve_existing_path(row.matrix_path, root)
        path = label_dir / row.dataset / row.sample / 'final_cell_labels.csv.gz'
        missing = []
        if matrix is None:
            missing.append('expression matrix')
        if not path.is_file():
            missing.append('final_cell_labels.csv.gz')
        else:
            columns = set(pd.read_csv(path, nrows=0).columns)
            missing.extend(sorted({'dataset', 'sample', 'cell_id', 'final_refined_label',
                                   'final_broad_label', 'final_label_source'} - columns))
        rows.append(dict(dataset=row.dataset, sample=row.sample, matrix_path=str(matrix or ''),
                         label_path=str(path), ready=not missing, missing='; '.join(missing)))
    return pd.DataFrame(rows)


def preparation_fingerprint(matrix_path, label_path, config):
    parameters = asdict(config)
    for key in ('resolutions', 'leiden_iterations', 'run_umap', 'max_umap_plot_cells', 'overwrite'):
        parameters.pop(key)
    inputs = [{ 'path': str(p.resolve()), 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
              for p in (matrix_path, label_path)]
    return hashlib.sha256(json.dumps({'version': 1, 'inputs': inputs, 'parameters': parameters},
                                    sort_keys=True).encode()).hexdigest()


def save_checkpoint(adata, path):
    temporary = path.with_name(path.stem + '.tmp.h5ad')
    adata.write_h5ad(temporary)
    temporary.replace(path)


def prepare_sample(matrix_path, label_path, dataset, sample, output_dir, config):
    """Select, retain QC reasons, normalize, then construct the epithelial graph."""
    from scipy import sparse
    import scanpy as sc

    log = cell_clustering.log_step
    log(f'{dataset} / {sample}: loading expression matrix')
    adata = cell_clustering.load_feature_matrix_as_adata(matrix_path)
    if sparse.issparse(adata.X):
        adata.X = adata.X.tocsr()
        adata.X.sum_duplicates()
        adata.X.eliminate_zeros()
    labels = pd.read_csv(label_path, dtype=str, keep_default_na=False,
                         usecols=lambda c: c in {'dataset', 'sample', 'cell_id', 'final_label_source', *LABEL_COLUMNS})
    if not (labels.dataset.eq(dataset) & labels['sample'].eq(sample)).all():
        raise ValueError('Final-label file contains another dataset/sample.')
    labels, method = align_labels_to_matrix(labels, adata.obs_names)
    symbols = adata.var.get('gene_symbol', pd.Series(adata.var_names, index=adata.var_names)).astype(str).str.upper()
    present = sorted(set(symbols) & {g.upper() for g in config.screening_genes})
    if len(present) < config.min_screening_genes:
        raise ValueError(f'Only {present} screening genes are measured; adjust the documented screening rule explicitly.')
    detected = np.zeros(adata.n_obs, dtype=int)
    for gene in present:
        # Count distinct gene symbols, not duplicated probe/features.
        detected += np.asarray(adata.X[:, np.flatnonzero(symbols.eq(gene))].sum(axis=1)).ravel() > 0
    audit = select_epithelial_candidates(labels, detected, config.min_screening_genes)
    totals = np.asarray(adata.X.sum(axis=1)).ravel()
    genes = np.asarray((adata.X > 0).sum(axis=1)).ravel()
    audit['total_counts'] = totals
    audit['n_genes_by_counts'] = genes
    low_counts, low_genes = totals < config.min_counts_per_cell, genes < config.min_genes_per_cell
    audit['exclusion_reason'] = np.select(
        [~audit.selected_for_reclustering.to_numpy(), low_counts & low_genes, low_counts, low_genes],
        ['not_epithelial_candidate', 'low_counts;low_genes', 'low_counts', 'low_genes'], default='')
    keep = audit.selected_for_reclustering.to_numpy() & ~low_counts & ~low_genes
    audit['used_for_reclustering'] = keep
    audit['has_saved_label_row'] = labels.has_saved_label_row.to_numpy()
    audit['final_refined_label'] = labels.final_refined_label.to_numpy()
    audit['final_broad_label'] = labels.final_broad_label.to_numpy()
    audit.insert(0, 'cell_id', adata.obs_names.astype(str))
    audit.insert(0, 'sample', sample)
    audit.insert(0, 'dataset', dataset)
    audit.to_csv(output_dir / 'epithelial_cell_selection.csv.gz', index=False)
    log(f'{dataset} / {sample}: {int(keep.sum()):,} cells selected after QC; no clustering subsampling')
    if keep.sum() < 4:
        raise ValueError('Fewer than four epithelial cells passed selection and QC; see the saved cell selection audit.')
    adata = adata[keep].copy()
    gene_cells = np.asarray((adata.X > 0).sum(axis=0)).ravel()
    gene_keep = gene_cells >= config.min_cells_per_gene
    pd.DataFrame({'gene': symbols.to_numpy(), 'n_epithelial_cells_detected': gene_cells,
                  'used_for_reclustering': gene_keep}).to_csv(output_dir / 'epithelial_gene_filter.csv', index=False)
    adata = adata[:, gene_keep].copy()
    if adata.n_vars < 3:
        raise ValueError('Fewer than three genes passed the epithelial-cell gene filter.')
    nonzero = np.asarray(adata.X.sum(axis=1)).ravel() > 0
    if not nonzero.all():
        affected = audit['cell_id'].isin(adata.obs_names[~nonzero])
        audit.loc[affected, 'used_for_reclustering'] = False
        audit.loc[affected, 'exclusion_reason'] = 'no_counts_after_gene_filter'
        audit.to_csv(output_dir / 'epithelial_cell_selection.csv.gz', index=False)
        adata = adata[nonzero].copy()
    if adata.n_obs < 4:
        raise ValueError('Fewer than four cells remain after gene filtering.')
    for column in LABEL_COLUMNS + ('final_label_source',):
        adata.obs[column] = labels.loc[adata.obs_names, column].to_numpy()
    adata.obs['selection_reason'] = audit.set_index('cell_id').loc[adata.obs_names, 'selection_reason'].to_numpy()
    adata.obs['total_counts'] = audit.set_index('cell_id').loc[adata.obs_names, 'total_counts'].to_numpy()
    adata.obs['n_genes_by_counts'] = audit.set_index('cell_id').loc[adata.obs_names, 'n_genes_by_counts'].to_numpy()
    adata.layers['counts'] = adata.X.copy()
    log(f'{dataset} / {sample}: normalizing {adata.n_obs:,} cells x {adata.n_vars:,} genes; log1p')
    sc.pp.normalize_total(adata, target_sum=config.normalization_target)
    sc.pp.log1p(adata)
    n_pcs = min(config.n_pcs, adata.n_obs - 1, adata.n_vars - 1)
    log(f'{dataset} / {sample}: PCA ({n_pcs} components)')
    sc.pp.pca(adata, n_comps=n_pcs, svd_solver='arpack', random_state=config.random_state)
    log(f'{dataset} / {sample}: constructing within-sample neighbor graph')
    sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=min(config.n_neighbors, adata.n_obs - 1),
                    random_state=config.random_state)
    adata.uns['epithelial_selection'] = dict(n_matrix_cells=len(audit),
        n_candidates=int(audit.selected_for_reclustering.sum()),
        n_clustered_cells=adata.n_obs, screening_genes_present=present, cell_id_match_method=method)
    return adata


def run_sample(row, output_root, config):
    import scanpy as sc
    import anndata as ad

    validate_config(config)
    dataset, sample = str(row['dataset']), str(row['sample'])
    if dataset not in DATASETS:
        raise ValueError(f'{dataset} is excluded from epithelial reclustering.')
    log = cell_clustering.log_step
    start = perf_counter()
    output_dir = Path(output_root) / dataset / sample
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path, label_path = Path(row['matrix_path']), Path(row['label_path'])
    fingerprint = preparation_fingerprint(matrix_path, label_path, config)
    checkpoint = output_dir / 'epithelial_expression.h5ad'
    if checkpoint.is_file() and not config.overwrite:
        log(f'{dataset} / {sample}: loading graph checkpoint')
        adata = ad.read_h5ad(checkpoint)
        excluded = non_gene_features(adata)
        if excluded:
            raise ValueError(f'Checkpoint contains {len(excluded)} non-gene features (e.g. {excluded[:3]}). '
                             'Rebuild this sample from its original matrix in a separate gene-only output folder; '
                             'the saved graph must not be reused.')
        if adata.uns.get('preparation_fingerprint') != fingerprint:
            raise ValueError('Inputs or preparation parameters changed. Use a new OUTPUT_DIR or explicitly set OVERWRITE_EXISTING=True.')
        if 'connectivities' not in adata.obsp or 'X_pca' not in adata.obsm:
            raise ValueError('Checkpoint is incomplete; set OVERWRITE_EXISTING=True to rebuild this sample.')
    else:
        adata = prepare_sample(matrix_path, label_path, dataset, sample, output_dir, config)
        adata.uns['preparation_fingerprint'] = fingerprint
        save_checkpoint(adata, checkpoint)
        log(f'{dataset} / {sample}: graph checkpoint saved')
    if config.run_umap and 'X_umap' not in adata.obsm:
        log(f'{dataset} / {sample}: UMAP started')
        sc.tl.umap(adata, random_state=config.random_state)
        save_checkpoint(adata, checkpoint)
        log(f'{dataset} / {sample}: UMAP saved')
    for resolution in config.resolutions:
        key = cluster_key(resolution)
        signature = f'leidenalg;iterations={config.leiden_iterations};seed={config.random_state}'
        if key in adata.obs and adata.uns.get(f'{key}_settings') == signature:
            log(f'{dataset} / {sample}: reuse {key}')
            continue
        log(f'{dataset} / {sample}: {key} started')
        resolution_start = perf_counter()
        sc.tl.leiden(adata, resolution=float(resolution), key_added=key, random_state=config.random_state,
                     n_iterations=config.leiden_iterations)
        adata.uns[f'{key}_settings'] = signature
        save_checkpoint(adata, checkpoint)
        log(f'{dataset} / {sample}: {key} saved ({adata.obs[key].nunique()} clusters; {perf_counter() - resolution_start:,.1f}s)')
    keys = [cluster_key(r) for r in config.resolutions]
    assignments = adata.obs[[*LABEL_COLUMNS, 'final_label_source', 'selection_reason',
                             'total_counts', 'n_genes_by_counts', *keys]].copy()
    assignments.insert(0, 'cell_id', adata.obs_names.astype(str))
    assignments.insert(0, 'sample', sample)
    assignments.insert(0, 'dataset', dataset)
    if config.run_umap:
        assignments['umap_1'], assignments['umap_2'] = adata.obsm['X_umap'].T
    assignment_path = output_dir / 'epithelial_cell_clusters.csv.gz'
    assignments.to_csv(assignment_path, index=False)
    summary_rows, figures = [], []
    for resolution, key in zip(config.resolutions, keys):
        counts = assignments[key].value_counts()
        summary_rows.extend(dict(dataset=dataset, sample=sample, resolution=resolution, cluster_key=key,
                                 cluster=str(c), n_cells=int(n), fraction_of_epithelial_cells=n / len(assignments))
                            for c, n in counts.items())
        if config.run_umap:
            figure_path = output_dir / 'figures' / f'{key}_umap.png'
            plot_config = cell_clustering.ClusteringConfig(max_umap_plot_cells=config.max_umap_plot_cells,
                                                          random_state=config.random_state)
            created = cell_clustering.plot_cluster_umap(assignments, key, figure_path, plot_config)
            if created is None:
                raise RuntimeError('UMAP figure could not be generated; checkpoint was retained.')
            figures.append(dict(dataset=dataset, sample=sample, cluster_key=key, figure_path=str(created)))
    pd.DataFrame(summary_rows).to_csv(output_dir / 'epithelial_cluster_summary.csv', index=False)
    pd.DataFrame(figures, columns=['dataset', 'sample', 'cluster_key', 'figure_path']).to_csv(output_dir / 'umap_figure_index.csv', index=False)
    log(f'{dataset} / {sample}: complete ({perf_counter() - start:,.1f}s)')
    return dict(dataset=dataset, sample=sample, status='success', error='', n_clustered_cells=len(assignments),
                n_genes=adata.n_vars, elapsed_seconds=round(perf_counter() - start, 1), assignment_path=str(assignment_path))


def run_all(inventory, output_root, config):
    validate_config(config)
    if not inventory.ready.all():
        raise FileNotFoundError('Resolve missing files in the saved input inventory before running.')
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, row in enumerate(inventory.to_dict('records'), 1):
        cell_clustering.log_step(f'[{i}/{len(inventory)}] Epithelial reclustering {row["dataset"]} / {row["sample"]}')
        try:
            rows.append(run_sample(row, output_root, config))
        except Exception as error:
            cell_clustering.log_step(f'{row["dataset"]} / {row["sample"]}: failed: {error}')
            rows.append(dict(dataset=row['dataset'], sample=row['sample'], status='failed', error=str(error)))
        pd.DataFrame(rows).to_csv(output_root / 'epithelial_reclustering_run_summary.csv', index=False)
    return pd.DataFrame(rows)
