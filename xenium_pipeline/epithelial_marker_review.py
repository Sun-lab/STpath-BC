"""Marker discovery and expression review of saved epithelial-only partitions."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import cluster_review
from .cell_clustering import log_step, safe_name
from .epithelial_reclustering import DATASETS, non_gene_features
from .gene_panel_utils import natural_key


PROGRAM_GROUPS = {
    'Epithelial': 'Epithelial cells',
    'Glandular/luminal': 'Breast glandular cells',
    'Myoepithelial': 'Breast myoepithelial cells',
    'Tumor-associated': 'Breast cancer',
}
PROLIFERATION_GENES = ('MKI67', 'TOP2A', 'CENPF', 'PCLAF')
LINEAGE_GROUPS = {
    'Fibroblast/stromal': 'Fibroblasts',
    'Myeloid/macrophage': 'Macrophages',
    'T cells': 'T cells',
    'B cells': 'B cells',
    'Endothelial': 'Endothelial cells',
    'Smooth muscle': 'Smooth muscle cells',
}
# Small explicit detection panels complement the inherited broad dictionaries.
CORE_LINEAGE_GENES = {
    'Epithelial': ('EPCAM', 'KRT8', 'KRT18', 'KRT19', 'KRT5', 'KRT14', 'KRT17'),
    'Fibroblast/stromal': ('LUM', 'DCN', 'COL1A1', 'COL1A2', 'PDGFRA'),
    'Myeloid/macrophage': ('FCER1G', 'CD68', 'C1QA', 'C1QC', 'LYZ'),
    'T cells': ('CD3E', 'CD3G', 'TRAC'),
    'B cells': ('MS4A1', 'CD79A', 'CD79B', 'CD19'),
    'Endothelial': ('PECAM1', 'VWF', 'EMCN', 'CDH5'),
    'Smooth muscle': ('MYH11', 'ACTA2', 'TAGLN', 'ACTG2'),
}


@dataclass(frozen=True)
class MarkerConfig:
    min_cluster_cells: int = 20
    min_fraction_expressing: float = 0.10
    min_fraction_difference: float = 0.10
    min_log2_fold_change: float = 0.5
    lfc_pseudocount: float = 0.1
    fdr: float = 0.05
    top_n: int = 15
    overwrite: bool = False


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def saved_samples(root):
    return [p for p in sorted(Path(root).glob('*/*/epithelial_expression.h5ad'))
            if p.parent.parent.name in DATASETS]


def saved_keys(adata):
    import re
    return sorted([c for c in adata.obs if re.fullmatch(r'epithelial_r\d+(?:_\d+)?', c)], key=natural_key)


def load_programs(review_dir, dataset, sample, *, include_lineages=False):
    """Prefer direct sample groups; fall back to sample-specific panel-implied groups."""
    direct_path = Path(review_dir) / 'marker_gene_table.csv'
    implied_path = Path(review_dir) / 'panel_implied_marker_groups.csv'
    if not direct_path.is_file() and not implied_path.is_file():
        raise FileNotFoundError(f'No notebook-14 marker relationship tables under {review_dir}')
    programs, origins = {}, {}
    direct = pd.read_csv(direct_path, keep_default_na=False) if direct_path.is_file() else pd.DataFrame()
    implied = pd.read_csv(implied_path, keep_default_na=False) if implied_path.is_file() else pd.DataFrame()
    for table in (direct, implied):
        if not table.empty:
            table.drop(table.index[~(table['dataset'].eq(dataset) & table['sample'].eq(sample))], inplace=True)
    groups = {**PROGRAM_GROUPS, **(LINEAGE_GROUPS if include_lineages else {})}
    for program, group in groups.items():
        genes = []
        if not direct.empty:
            genes = direct.loc[direct['group'].eq(group), 'gene'].astype(str).tolist()
        if genes:
            origins[program] = 'direct_sample_group'
        elif not implied.empty:
            rows = implied.loc[implied['group'].eq(group), 'matching_marker_genes']
            genes = [g.strip() for value in rows.astype(str) for g in value.split(';') if g.strip()]
            origins[program] = 'panel_implied_sample_group' if genes else 'missing_group'
        else:
            origins[program] = 'missing_group'
        programs[program] = sorted({g.strip().upper() for g in genes if g.strip()})
    programs['Proliferation'] = list(PROLIFERATION_GENES)
    origins['Proliferation'] = 'explicit_exploratory_gene_list'
    return programs, origins


def normalized_target(adata):
    """Infer the original library-size target from the saved log1p matrix."""
    from scipy import sparse
    values = adata.X.copy()
    if sparse.issparse(values):
        values.data = np.expm1(values.data)
    else:
        values = np.expm1(values)
    totals = np.asarray(values.sum(axis=1)).ravel()
    target = float(np.median(totals))
    if target <= 0 or not np.isfinite(totals).all() or not np.allclose(totals, target, rtol=2e-4):
        raise ValueError('Expected saved library-normalized log1p expression from 06h Sections 1-3.')
    return target


def program_scores(adata, programs, origins):
    """Mean log1p expression over available distinct genes; missing is not zero."""
    symbols = adata.var.get('gene_symbol', pd.Series(adata.var_names, index=adata.var_names)).astype(str).str.upper()
    scores = pd.DataFrame({'cell_id': adata.obs_names.astype(str)})
    coverage = []
    for program, requested in programs.items():
        available = [g for g in requested if g in set(symbols)]
        missing = sorted(set(requested) - set(available))
        weights = np.zeros(adata.n_vars)
        for gene in available:
            indices = np.flatnonzero(symbols.eq(gene))
            # Each distinct gene has equal weight, even with duplicate features.
            weights[indices] = 1 / (len(indices) * len(available))
        scores[program] = np.asarray(adata.X @ weights).ravel() if available else np.nan
        coverage.append(dict(program=program, source=origins[program], n_requested_genes=len(requested),
                             n_used_genes=len(available), used_genes='; '.join(available),
                             unavailable_genes='; '.join(missing)))
    return scores, pd.DataFrame(coverage)


def discover_markers(adata, dataset, sample, key, config):
    import anndata as ad
    import scanpy as sc

    excluded = non_gene_features(adata)
    if excluded:
        raise ValueError(f'Checkpoint includes {len(excluded)} non-gene features. Rebuild gene-only clustering '
                         'from the original matrix before new DE. Existing outputs remain available for provisional review.')

    if 'counts' not in adata.layers:
        raise ValueError('The saved raw counts layer is required for expression metrics.')
    labels = adata.obs[key].astype(str)
    sizes = labels.value_counts()
    valid = sizes.index[(sizes >= config.min_cluster_cells) & ((len(labels) - sizes) >= config.min_cluster_cells)].tolist()
    statuses = pd.DataFrame([dict(dataset=dataset, sample=sample, cluster_key=key, cluster=str(c),
                                 n_cluster_cells=int(n), n_rest_cells=int(len(labels)-n),
                                 test_status='tested' if c in valid else 'too_few_cluster_or_rest_cells',
                                 n_genes_tested=adata.n_vars if c in valid else 0)
                            for c, n in sizes.items()])
    columns = ['dataset', 'sample', 'cluster_key', 'cluster', 'var_name', 'gene', 'n_cluster_cells',
               'n_rest_cells', 'mean_normalized_expression_cluster', 'mean_normalized_expression_rest',
               'fraction_expressing_cluster', 'fraction_expressing_rest', 'fraction_expressing_difference',
               'log2_fold_change', 'marker_rank_score', 'wilcoxon_score', 'p_value', 'p_value_adj',
               'passes_marker_filter']
    if not valid:
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=columns + ['review_rank']), statuses
    raw = ad.AnnData(X=adata.layers['counts'].copy(), obs=pd.DataFrame(index=adata.obs_names), var=adata.var.copy())
    from scipy import sparse
    if sparse.issparse(raw.X):
        raw.X = raw.X.tocsr()
        raw.X.eliminate_zeros()
    metrics = cluster_review.calculate_cluster_expression_marker_metrics(
        raw, labels, dataset=dataset, sample=sample, cluster_key=key,
        normalization_target=normalized_target(adata), lfc_pseudocount=config.lfc_pseudocount,
        min_fraction_expressing=config.min_fraction_expressing,
        min_fraction_difference=config.min_fraction_difference, min_log2_fold_change=config.min_log2_fold_change,
    )
    metrics = metrics.loc[metrics.cluster.isin(valid)].copy()
    test = ad.AnnData(X=adata.X, obs=pd.DataFrame(index=adata.obs_names), var=adata.var.copy())
    test.obs[key] = pd.Categorical(labels.to_numpy())
    sc.tl.rank_genes_groups(test, groupby=key, groups=valid, reference='rest', use_raw=False,
                           method='wilcoxon', corr_method='benjamini-hochberg', tie_correct=True,
                           n_genes=adata.n_vars)
    ranked = sc.get.rank_genes_groups_df(test, group=None).rename(columns={
        'group': 'cluster', 'names': 'var_name', 'scores': 'wilcoxon_score',
        'pvals': 'p_value', 'pvals_adj': 'p_value_adj'})
    ranked['cluster'], ranked['var_name'] = ranked['cluster'].astype(str), ranked['var_name'].astype(str)
    metrics = metrics.merge(ranked[['cluster', 'var_name', 'wilcoxon_score', 'p_value', 'p_value_adj']],
                            on=['cluster', 'var_name'], validate='one_to_one')
    if len(metrics) != len(valid) * adata.n_vars:
        raise ValueError('Wilcoxon output does not cover every tested cluster/gene pair.')
    metrics['passes_marker_filter'] = metrics.passes_basic_marker_filter & metrics.p_value_adj.le(config.fdr)
    metrics = metrics[columns].sort_values(['cluster', 'marker_rank_score', 'log2_fold_change', 'gene'],
                                          ascending=[True, False, False, True]).reset_index(drop=True)
    top = metrics.loc[metrics.passes_marker_filter].groupby('cluster', sort=False).head(config.top_n).copy()
    top['review_rank'] = top.groupby('cluster').cumcount() + 1
    return metrics, top, statuses


def validate_config(config):
    if config.min_cluster_cells < 2 or config.top_n < 1 or config.lfc_pseudocount <= 0:
        raise ValueError('Require min_cluster_cells >= 2, top_n >= 1 and positive lfc_pseudocount.')
    if not 0 < config.fdr <= 1 or not 0 <= config.min_fraction_expressing <= 1 or not 0 <= config.min_fraction_difference <= 1:
        raise ValueError('Invalid FDR or expression-fraction threshold.')
    if not np.isfinite(config.min_log2_fold_change):
        raise ValueError('min_log2_fold_change must be finite.')


def run_marker_review(root, review_dir, config=MarkerConfig()):
    import anndata as ad
    validate_config(config)
    root = Path(root)
    paths = saved_samples(root)
    if not paths:
        raise FileNotFoundError(f'No saved epithelial_expression.h5ad files under {root}')
    records = []
    parameters = asdict(config)
    parameters.pop('overwrite')
    for i, path in enumerate(paths, 1):
        dataset, sample = path.parent.parent.name, path.parent.name
        log_step(f'[{i}/{len(paths)}] Epithelial expression review: {dataset} / {sample}')
        try:
            source_hash = file_hash(path)
            adata = ad.read_h5ad(path)
            if non_gene_features(adata):
                raise ValueError('Checkpoint contains non-gene features; rebuild gene-only clustering in a '
                                 'separate folder. Recompute expression-only checkpoints before marker review.')
            keys = saved_keys(adata)
            if not keys:
                raise ValueError('Checkpoint contains no saved epithelial resolution columns.')
            out = path.parent / 'expression_review'
            out.mkdir(exist_ok=True)
            programs, origins = load_programs(review_dir, dataset, sample)
            program_signature = signature(dict(version=1, source_hash=source_hash, programs=programs, origins=origins))
            program_meta = out / 'program_metadata.json'
            score_path, coverage_path = out / 'cell_program_scores.csv.gz', out / 'program_coverage.csv'
            program_cached = (not config.overwrite and program_meta.is_file() and score_path.is_file() and coverage_path.is_file()
                              and json.loads(program_meta.read_text()).get('signature') == program_signature)
            if not program_cached:
                program_meta.unlink(missing_ok=True)
                scores, coverage = program_scores(adata, programs, origins)
                scores.to_csv(score_path, index=False)
                coverage.to_csv(coverage_path, index=False)
                program_meta.write_text(json.dumps(dict(signature=program_signature, source_hash=source_hash,
                                                        programs=programs, origins=origins), indent=2))
            for key in keys:
                log_step(f'{dataset} / {sample}: markers for {key}')
                target = out / key
                target.mkdir(exist_ok=True)
                fingerprint = signature(dict(version=1, source_hash=source_hash, key=key, parameters=parameters))
                meta = target / 'marker_metadata.json'
                outputs = [target / n for n in ('all_marker_genes.csv.gz', 'top_marker_genes.csv', 'cluster_test_status.csv')]
                cached = (not config.overwrite and meta.is_file() and all(p.is_file() for p in outputs)
                          and json.loads(meta.read_text()).get('signature') == fingerprint)
                if not cached:
                    meta.unlink(missing_ok=True)
                    markers, top, status = discover_markers(adata, dataset, sample, key, config)
                    markers.to_csv(outputs[0], index=False)
                    top.to_csv(outputs[1], index=False)
                    status.to_csv(outputs[2], index=False)
                    meta.write_text(json.dumps(dict(signature=fingerprint, source_hash=source_hash, parameters=parameters), indent=2))
                records.append(dict(dataset=dataset, sample=sample, cluster_key=key,
                                    status='reused' if cached else 'success', error='', output_dir=str(target)))
                pd.DataFrame(records).to_csv(root / 'epithelial_marker_review_run_summary.csv', index=False)
        except Exception as error:
            log_step(f'{dataset} / {sample}: marker review failed: {error}')
            records.append(dict(dataset=dataset, sample=sample, cluster_key='', status='failed', error=str(error), output_dir=''))
        pd.DataFrame(records).to_csv(root / 'epithelial_marker_review_run_summary.csv', index=False)
    return pd.DataFrame(records)


def load_review_sample(path):
    """Load one sample, rejecting stale expression-review outputs."""
    import anndata as ad
    path = Path(path)
    directory = path.parent / 'lineage_review'
    if not (directory / 'program_metadata.json').is_file():
        directory = path.parent / 'expression_review'
    metadata = json.loads((directory / 'program_metadata.json').read_text())
    source_hash = file_hash(path)
    if metadata['source_hash'] != source_hash:
        raise ValueError('Expression checkpoint changed. Recompute marker evidence before reviewing it.')
    adata = ad.read_h5ad(path)
    scores = pd.read_csv(directory / 'cell_program_scores.csv.gz', dtype={'cell_id': str}).set_index('cell_id')
    if scores.index.has_duplicates or set(scores.index) != set(adata.obs_names):
        raise ValueError('Program scores do not match the saved epithelial cells.')
    return adata, scores.reindex(adata.obs_names), pd.read_csv(directory / 'program_coverage.csv', keep_default_na=False), source_hash


def core_detection(adata, min_detected=2):
    """Count distinct raw-positive core genes; insufficient coverage remains missing."""
    if min_detected < 1:
        raise ValueError('min_detected must be positive.')
    names = adata.var.get('gene_symbol', pd.Series(adata.var_names, index=adata.var_names)).astype(str).str.upper()
    detected, coverage = {}, []
    for program, genes in CORE_LINEAGE_GENES.items():
        available = [g for g in genes if names.eq(g).any()]
        counts = np.zeros(adata.n_obs, dtype=int)
        for gene in available:
            counts += np.asarray((adata.layers['counts'][:, np.flatnonzero(names.eq(gene))] > 0).sum(axis=1)).ravel() > 0
        detected[program] = counts >= min_detected if len(available) >= min_detected else None
        coverage.append(dict(program=program, n_available_core_genes=len(available),
                             available_core_genes='; '.join(available), min_detected_genes=min_detected))
    return detected, pd.DataFrame(coverage)


def summarize_lineage_evidence(adata, scores, *, min_detected=2):
    detected, coverage = core_detection(adata, min_detected)
    flags = pd.DataFrame(index=adata.obs_names)
    epithelial = detected['Epithelial']
    for program, mask in detected.items():
        if program == 'Epithelial':
            continue
        valid = mask is not None and epithelial is not None
        flags[program + '|epithelial_only'] = (epithelial & ~mask) if valid else np.nan
        flags[program + '|lineage_only'] = (~epithelial & mask) if valid else np.nan
        flags[program + '|both'] = (epithelial & mask) if valid else np.nan
        flags[program + '|neither'] = (~epithelial & ~mask) if valid else np.nan
    rows = []
    for key in saved_keys(adata):
        labels = adata.obs[key].astype(str)
        fractions = flags.groupby(labels).mean()
        sizes = labels.value_counts()
        medians = scores.set_index('cell_id').reindex(adata.obs_names).groupby(labels).median()
        for cluster, values in fractions.iterrows():
            for program in LINEAGE_GROUPS:
                cov = coverage.loc[coverage.program.eq(program)].iloc[0]
                rows.append(dict(cluster_key=key, cluster=cluster, n_cells=int(sizes[cluster]), program=program,
                                 n_available_core_genes=int(cov.n_available_core_genes),
                                 min_detected_genes=min_detected,
                                 **{f'fraction_{state}': values[program+'|'+state]
                                    for state in ('epithelial_only', 'lineage_only', 'both', 'neither')},
                                 median_program_score=medians.loc[cluster, program]))
    return pd.DataFrame(rows), coverage


def run_lineage_review(root, review_dir, *, min_detected=2):
    """Add evidence without changing expression checkpoints, DE, or labels."""
    import anndata as ad
    root = Path(root)
    records = []
    paths = saved_samples(root)
    if not paths:
        raise FileNotFoundError(f'No saved epithelial checkpoints under {root}')
    for i, path in enumerate(paths, 1):
        dataset, sample = path.parent.parent.name, path.parent.name
        log_step(f'[{i}/{len(paths)}] Lineage evidence: {dataset} / {sample}')
        try:
            source_hash = file_hash(path)
            programs, origins = load_programs(review_dir, dataset, sample, include_lineages=True)
            expected = signature(dict(version=1, source_hash=source_hash, programs=programs, origins=origins,
                                      core_genes=CORE_LINEAGE_GENES, min_detected=min_detected))
            out = path.parent / 'lineage_review'
            out.mkdir(exist_ok=True)
            meta = out / 'program_metadata.json'
            outputs = ['cell_program_scores.csv.gz', 'program_coverage.csv', 'cluster_lineage_evidence.csv',
                       'core_gene_coverage.csv']
            cached = (meta.is_file() and all((out / f).is_file() for f in outputs)
                      and json.loads(meta.read_text()).get('signature') == expected)
            if not cached:
                data = ad.read_h5ad(path)
                excluded = non_gene_features(data)
                scores, coverage = program_scores(data, programs, origins)
                evidence, core = summarize_lineage_evidence(data, scores, min_detected=min_detected)
                evidence.insert(0, 'sample', sample)
                evidence.insert(0, 'dataset', dataset)
                evidence['n_non_gene_features'] = len(excluded)
                evidence['review_status'] = 'provisional_controls_present' if excluded else 'gene_only_checkpoint'
                meta.unlink(missing_ok=True)
                scores.to_csv(out / outputs[0], index=False)
                coverage.to_csv(out / outputs[1], index=False)
                evidence.to_csv(out / outputs[2], index=False)
                core.to_csv(out / outputs[3], index=False)
                meta.write_text(json.dumps(dict(signature=expected, source_hash=source_hash, programs=programs,
                                                origins=origins, core_genes=CORE_LINEAGE_GENES,
                                                min_detected=min_detected, n_non_gene_features=len(excluded)), indent=2))
                del data, scores
            metadata = json.loads(meta.read_text())
            records.append(dict(dataset=dataset, sample=sample, status='reused' if cached else 'success',
                                n_non_gene_features=metadata['n_non_gene_features'], error='', output_dir=str(out)))
        except Exception as error:
            records.append(dict(dataset=dataset, sample=sample, status='failed', error=str(error), output_dir=''))
        pd.DataFrame(records).to_csv(root / 'epithelial_lineage_review_run_summary.csv', index=False)
    return pd.DataFrame(records)


def plot_review(adata, scores, coverage, key, cluster, output_dir, *, max_dot_genes=8, histogram_bins=30,
                max_umap_cells=100_000, min_detected=2):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    selected = adata.obs[key].astype(str).eq(str(cluster)).to_numpy()
    if not selected.any():
        raise ValueError(f'Cluster {cluster} is absent from {key}')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    symbols = adata.var.get('gene_symbol', pd.Series(adata.var_names, index=adata.var_names)).astype(str).str.upper()
    programs = coverage.program.tolist()
    if max_dot_genes < 1 or histogram_bins < 2 or max_umap_cells < 2:
        raise ValueError('Plot limits must be positive (histogram/UMAP at least 2).')
    dot_rows = []
    for row in coverage.itertuples(index=False):
        genes = [g.strip() for g in row.used_genes.split(';') if g.strip()]
        candidates = []
        for gene in genes:
            cols = np.flatnonzero(symbols.eq(gene))
            values = np.asarray(adata.X[:, cols].mean(axis=1)).ravel()
            detected = np.asarray((adata.layers['counts'][:, cols] > 0).sum(axis=1)).ravel() > 0
            mean_in = float(values[selected].mean())
            mean_out = float(values[~selected].mean()) if (~selected).any() else np.nan
            candidates.append(dict(program=row.program, gene=gene, mean_in=mean_in, mean_out=mean_out,
                                   fraction_in=float(detected[selected].mean()),
                                   fraction_out=float(detected[~selected].mean()) if (~selected).any() else np.nan))
        candidates.sort(key=lambda r: (-abs(r['mean_in'] - r['mean_out']) if np.isfinite(r['mean_out']) else -r['mean_in'], r['gene']))
        dot_rows.extend(candidates[:max_dot_genes])
    dot_table = pd.DataFrame(dot_rows, columns=['program', 'gene', 'mean_in', 'mean_out', 'fraction_in', 'fraction_out'])
    max_rows = max([sum(r['program'] == p for r in dot_rows) for p in programs] + [1])
    n_rows = (len(programs) + 2) // 2
    fig, axes = plt.subplots(n_rows, 2, squeeze=False,
                             figsize=(13, max(6.5, n_rows * (.35 * max_rows + 1.2))), constrained_layout=True)
    maximum = max([r[k] for r in dot_rows for k in ('mean_in', 'mean_out') if np.isfinite(r[k])] + [1e-6])
    norm = Normalize(0, maximum)
    for axis, program in zip(axes.flat, programs):
        entries = [r for r in dot_rows if r['program'] == program]
        axis.set_title(program)
        if not entries:
            axis.text(.5, .5, 'No available program genes', ha='center', transform=axis.transAxes)
            axis.set_axis_off()
            continue
        for x, suffix in enumerate(('in', 'out')):
            axis.scatter(np.repeat(x, len(entries)), np.arange(len(entries)),
                         s=[250 * r[f'fraction_{suffix}'] for r in entries],
                         c=[r[f'mean_{suffix}'] for r in entries], cmap='viridis', norm=norm)
        axis.set_yticks(range(len(entries)), [r['gene'] for r in entries])
        axis.set_ylim(len(entries) - .5, -.5)
        axis.set_xticks([0, 1], ['Selected cluster', 'Other candidates'])
        axis.set_xlim(-.5, 1.5)
        axis.spines[['top', 'right']].set_visible(False)
    legend = axes.flat[-1]
    for axis in list(axes.flat)[len(programs):]:
        axis.set_axis_off()
    legend.set_axis_off()
    handles = [legend.scatter([], [], s=250*f, color='gray', label=f'{f:.0%}') for f in (.1, .5, 1)]
    legend.legend(handles=handles, title='Fraction expressing', loc='center', frameon=False)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=list(axes.flat), shrink=.6,
                 label='Mean normalized log1p expression')
    fig.suptitle(f'{key} / cluster {cluster}: program genes', fontsize=13)
    paths.append(out / 'program_gene_dotplots.png')
    fig.savefig(paths[-1], dpi=160, bbox_inches='tight')
    plt.close(fig)
    dot_table.to_csv(out / 'displayed_program_gene_statistics.csv', index=False)

    fig, axes = plt.subplots(n_rows, 2, squeeze=False, figsize=(13, 3.3 * n_rows), constrained_layout=True)
    for axis, program in zip(axes.flat, programs):
        values = scores[program].to_numpy(dtype=float)
        valid = np.isfinite(values)
        axis.set_title(program)
        if not valid.any():
            axis.text(.5, .5, 'No score: no available genes', ha='center', transform=axis.transAxes)
            axis.set_axis_off()
            continue
        edges = np.histogram_bin_edges(values[valid], bins=histogram_bins)
        for mask, color, label in ((selected & valid, '#C64A50', 'Selected cluster'), (~selected & valid, '#5A6772', 'Other candidates')):
            n = int(mask.sum())
            if n:
                counts, _ = np.histogram(values[mask], bins=edges)
                axis.stairs(counts / n, edges, color=color, label=f'{label} (n={n:,})', linewidth=1.5)
        axis.set_xlabel('Mean program log1p expression')
        axis.set_ylabel('Fraction of cells per bin')
        axis.legend(fontsize=8, frameon=False)
        axis.spines[['top', 'right']].set_visible(False)
    for axis in list(axes.flat)[len(programs):]:
        axis.set_axis_off()
    fig.suptitle(f'{key} / cluster {cluster}: all-cell program distributions', fontsize=13)
    paths.append(out / 'program_score_distributions.png')
    fig.savefig(paths[-1], dpi=160, bbox_inches='tight')
    plt.close(fig)

    if 'X_umap' in adata.obsm:
        rng = np.random.default_rng(1729)
        fig, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
        for mask, color, label in ((~selected, '#D1D1D1', 'Other candidates'), (selected, '#C64A50', f'Cluster {cluster}')):
            indices = np.flatnonzero(mask)
            if len(indices) > max_umap_cells // 2:
                indices = rng.choice(indices, max_umap_cells // 2, replace=False)
            axis.scatter(*adata.obsm['X_umap'][indices].T, color=color, s=2, linewidths=0, label=label)
        axis.set(xlabel='UMAP 1', ylabel='UMAP 2', title=f'{key} / cluster {cluster}')
        axis.legend(frameon=False)
        paths.append(out / 'selected_cluster_umap.png')
        fig.savefig(paths[-1], dpi=160, bbox_inches='tight')
        plt.close(fig)

    detected, core = core_detection(adata, min_detected)
    epithelial = detected['Epithelial']
    states = ('epithelial_only', 'lineage_only', 'both', 'neither')
    rows = []
    fig, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    for program in LINEAGE_GROUPS:
        lineage = detected[program]
        valid = epithelial is not None and lineage is not None
        masks = (epithelial & ~lineage, ~epithelial & lineage, epithelial & lineage,
                 ~epithelial & ~lineage) if valid else (None,) * 4
        for population, subset in (('Cluster', selected), ('Other', ~selected)):
            row = dict(program=program, population=population, n_cells=int(subset.sum()),
                       coverage_sufficient=valid, min_detected_genes=min_detected)
            for state, mask in zip(states, masks):
                row['fraction_'+state] = float(mask[subset].mean()) if valid and subset.any() else np.nan
            rows.append(row)
    coexpression = pd.DataFrame(rows)
    left = np.zeros(len(rows))
    colors = ('#3577A6', '#DA8F32', '#A35A98', '#D2D2D2')
    for state, color in zip(states, colors):
        values = coexpression['fraction_'+state].fillna(0).to_numpy()
        axis.barh(np.arange(len(rows)), values, left=left, color=color, label=state.replace('_', ' '))
        left += values
    for i, row in coexpression.iterrows():
        if not row.coverage_sufficient:
            axis.text(.02, i, 'Insufficient core-gene coverage', va='center')
    axis.set_yticks(np.arange(len(rows)), [f'{r.program} / {r.population}' for r in coexpression.itertuples()])
    axis.invert_yaxis()
    axis.set(xlim=(0, 1), xlabel='Fraction of cells in the named population',
             title=f'{key} / cluster {cluster}: core-gene co-detection (at least {min_detected} per panel)')
    axis.legend(loc='upper center', bbox_to_anchor=(.5, -.10), ncol=4, frameon=False)
    paths.append(out / 'core_gene_coexpression.png')
    fig.savefig(paths[-1], dpi=160, bbox_inches='tight')
    plt.close(fig)
    coexpression.to_csv(out / 'core_gene_coexpression.csv', index=False)
    core.to_csv(out / 'core_gene_coverage.csv', index=False)
    return paths
