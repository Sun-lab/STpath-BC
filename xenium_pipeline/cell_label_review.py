"""Final labels from expression, H&E review, and author annotations."""

from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from .cluster_review import match_cluster_source_cell_ids
from .epithelial_marker_review import file_hash
from .retained_final_labels import BROAD_COLORS


FINAL_COLUMNS = ['final_refined_label', 'final_broad_label', 'final_label_source']
KEYS = ['dataset', 'sample', 'cluster_key', 'cluster']
PROPOSAL_COLUMNS = ['proposed_refined_label', 'proposed_broad_label',
                    'lineage_confidence', 'malignancy_status', 'decision_status',
                    'annotation_reason']
EPITHELIAL_UNRESOLVED = 'Epithelial, normal/tumor unresolved'
MANUAL_LABELS = {
    'Tumor': ('Tumor epithelial cells', 'Tumor Cells'),
    'Normal epithelial': ('Normal epithelial cells', 'Normal Epithelial Cells'),
    'Uncertain': ('Uncertain', 'Uncertain'),
}


def _require(table, columns, name):
    missing = set(columns) - set(table)
    if missing:
        raise ValueError(f'{name} missing columns: {sorted(missing)}')


def _unique(table, keys, name):
    _require(table, keys, name)
    if table[keys].isna().any().any() or table[keys].astype(str).apply(
        lambda column: column.str.strip().eq('')).any().any() or table.duplicated(keys).any():
        raise ValueError(f'{name} has blank or duplicate identities: {keys}')


def load_cluster_updates(annotation_path, decision_path, cluster_key='epithelial_r0_7'):
    """Manual decisions override proposals; explicit uncertainty is propagated."""
    proposals = pd.read_csv(annotation_path, dtype=str, keep_default_na=False)
    _require(proposals, KEYS + PROPOSAL_COLUMNS + ['n_cells', 'checkpoint_sha256'], 'Proposals')
    proposals = proposals.loc[proposals.cluster_key.eq(cluster_key)].copy()
    if proposals.empty:
        raise ValueError(f'No proposals at {cluster_key}')
    _unique(proposals, KEYS, 'Proposals')
    if set(proposals.dataset) - {'10x', 'Bilous_2026', 'Liu_2025'}:
        raise ValueError('Epithelial updates are restricted to 10x, Bilous_2026, and Liu_2025.')
    unknown = set(proposals.proposed_broad_label) - set(BROAD_COLORS) - {EPITHELIAL_UNRESOLVED}
    if unknown:
        raise ValueError(f'Unmapped proposed broad labels: {sorted(unknown)}')
    counts = pd.to_numeric(proposals.n_cells, errors='raise')
    if (counts <= 0).any() or counts.mod(1).ne(0).any():
        raise ValueError('Proposal n_cells must be positive integers.')
    proposals['n_cells'] = counts.astype(int)
    manual = pd.read_csv(decision_path, dtype=str, keep_default_na=False)
    _require(manual, KEYS + ['manual_label', 'alignment_confirmed', 'notes'], 'Manual decisions')
    _unique(manual, KEYS, 'Manual decisions')
    if not manual.cluster_key.eq(cluster_key).all():
        raise ValueError('Manual decisions include another resolution; select its matching inputs explicitly.')
    if set(manual.manual_label) - set(MANUAL_LABELS):
        raise ValueError('Unknown manual_label in the saved decisions.')
    if not manual.alignment_confirmed.str.lower().isin({'true', 'false'}).all():
        raise ValueError('alignment_confirmed must be True or False.')
    outside = manual[KEYS].merge(proposals[KEYS], how='left', on=KEYS, indicator=True)
    if outside._merge.eq('left_only').any():
        raise ValueError('Manual decisions reference clusters absent from the proposals.')
    manual = manual.rename(columns={c: 'he_' + c for c in manual if c not in KEYS})
    plan = proposals.merge(manual, on=KEYS, how='left', validate='one_to_one').fillna('')
    plan['updated_refined_label'] = plan.proposed_refined_label
    plan['updated_broad_label'] = plan.proposed_broad_label.replace(EPITHELIAL_UNRESOLVED, 'Uncertain')
    plan.loc[plan.proposed_broad_label.eq('Uncertain'), 'updated_refined_label'] = 'Uncertain'
    plan['update_source'] = 'epithelial_expression_proposal'
    plan['update_status'] = 'provisional_expression_annotation'
    biological = ~plan.updated_broad_label.eq('Uncertain')
    if not plan.loc[biological, 'lineage_confidence'].isin({'moderate', 'high'}).all():
        raise ValueError('A biological expression label lacks moderate/high lineage confidence.')
    for label, (refined, broad) in MANUAL_LABELS.items():
        use = plan.he_manual_label.eq(label)
        plan.loc[use, ['updated_refined_label', 'updated_broad_label']] = [refined, broad]
        plan.loc[use, 'update_source'] = 'epithelial_manual_he_review'
        plan.loc[use, 'update_status'] = 'provisional_he_review'
    return plan


def input_inventory(baseline_dir, annotation_dir, plan, sample_index=None):
    baseline_dir, annotation_dir = Path(baseline_dir), Path(annotation_dir)
    samples = set(map(tuple, plan[['dataset', 'sample']].drop_duplicates().to_numpy()))
    samples |= {(p.parent.parent.name, p.parent.name)
                for p in baseline_dir.glob('*/*/final_cell_labels.csv.gz')}
    if sample_index is not None:
        expected = pd.read_csv(sample_index, dtype=str, keep_default_na=False)
        _unique(expected, ['dataset', 'sample'], 'final-label sample index')
        samples |= set(map(tuple, expected[['dataset', 'sample']].to_numpy()))
    rows = []
    for dataset, sample in sorted(samples):
        selected = plan.loc[plan.dataset.eq(dataset) & plan['sample'].eq(sample)]
        baseline = baseline_dir / dataset / sample / 'final_cell_labels.csv.gz'
        cells = annotation_dir / dataset / sample / 'cell_annotations.csv.gz'
        ready = baseline.is_file() and (selected.empty or cells.is_file())
        rows.append(dict(dataset=dataset, sample=sample, n_update_clusters=len(selected),
                         n_expected_update_cells=int(selected.n_cells.sum()),
                         baseline_path=str(baseline), baseline_exists=baseline.is_file(),
                         epithelial_cells_path=str(cells) if len(selected) else '',
                         epithelial_cells_exists=cells.is_file() if len(selected) else False,
                         status='ready' if ready else 'missing_input'))
    return pd.DataFrame(rows)


def load_cell_updates(annotation_dir, sample_plan):
    """Validate the saved per-cell proposals against the complete cluster table."""
    identities = sample_plan[['dataset', 'sample']].drop_duplicates()
    if len(identities) != 1:
        raise ValueError('Provide one sample plan.')
    dataset, sample = identities.iloc[0]
    path = Path(annotation_dir) / dataset / sample / 'cell_annotations.csv.gz'
    cells = pd.read_csv(path, dtype=str, keep_default_na=False)
    _require(cells, KEYS + ['cell_id'] + PROPOSAL_COLUMNS, 'Cell proposals')
    _unique(cells, ['cell_id'], 'Cell proposals')
    if not cells.dataset.eq(dataset).all() or not cells['sample'].eq(sample).all():
        raise ValueError(f'Cell proposal identifiers disagree with {path}')
    counts = cells.groupby(KEYS).size().sort_index()
    expected = sample_plan.set_index(KEYS).n_cells.sort_index()
    if not counts.index.equals(expected.index) or not (counts.to_numpy() == expected.to_numpy()).all():
        raise ValueError(f'Cluster membership/counts disagree with proposals: {path}')
    recorded = cells[KEYS + PROPOSAL_COLUMNS].drop_duplicates()
    check = recorded.merge(sample_plan[KEYS + PROPOSAL_COLUMNS], on=KEYS,
                           validate='one_to_one', suffixes=('_cell', '_cluster'))
    if any(check[c + '_cell'].ne(check[c + '_cluster']).any() for c in PROPOSAL_COLUMNS):
        raise ValueError(f'Saved cell and cluster proposals disagree: {path}')
    return cells[KEYS + ['cell_id']].merge(sample_plan, on=KEYS, validate='many_to_one')


def apply_cell_updates(baseline, updates):
    """Keep baseline rows/order and evidence, replacing only targeted final labels."""
    _unique(baseline, ['cell_id'], 'Baseline')
    _require(baseline, ['dataset', 'sample'] + FINAL_COLUMNS, 'Baseline')
    if baseline.empty or len(baseline[['dataset', 'sample']].drop_duplicates()) != 1:
        raise ValueError('Baseline must contain exactly one nonempty sample.')
    if any(c.startswith(('pre_epithelial_', 'epithelial_update_')) for c in baseline):
        raise ValueError('Use the original final-label baseline, not an already updated table.')
    result = baseline.copy()
    for column in FINAL_COLUMNS:
        result['pre_epithelial_' + column] = result[column]
    result['epithelial_update_applied'] = False
    result['epithelial_update_source_cell_id'] = ''
    if updates.empty:
        return result, 'not_applicable'
    _unique(updates, ['cell_id'], 'Cell updates')
    identity = baseline.iloc[0]
    if identity.dataset not in {'10x', 'Bilous_2026', 'Liu_2025'} or not (
        updates.dataset.eq(identity.dataset) & updates['sample'].eq(identity['sample'])).all():
        raise ValueError('Updates cannot change Janesick or cross sample boundaries.')
    source = updates[['cell_id']].copy()
    source['update_row'] = range(len(updates))
    matched, diagnostic = match_cluster_source_cell_ids(
        source, cell_id_columns=['cell_id'], cluster_column='update_row',
        cluster_key='update_row', matrix_cell_ids=baseline.cell_id)
    if len(matched) != len(updates) or not matched.cell_id.isin(baseline.cell_id).all():
        raise ValueError('Epithelial cells are absent from the final-label baseline; no cells may be dropped.')
    aligned = updates.iloc[matched.update_row.astype(int)].copy()
    aligned.index = matched.cell_id.to_numpy()
    mask = result.cell_id.isin(aligned.index)
    ordered = aligned.reindex(result.loc[mask, 'cell_id'])
    for final, updated in zip(FINAL_COLUMNS, ['updated_refined_label', 'updated_broad_label', 'update_source']):
        result.loc[mask, final] = ordered[updated].to_numpy()
    result.loc[mask, 'epithelial_update_applied'] = True
    result.loc[mask, 'epithelial_update_source_cell_id'] = ordered.cell_id.to_numpy()
    evidence = ['cluster_key', 'cluster', 'checkpoint_sha256', 'update_status'] + PROPOSAL_COLUMNS
    evidence += [c for c in updates if c.startswith('he_')]
    for column in evidence:
        result['epithelial_update_' + column] = ''
        result.loc[mask, 'epithelial_update_' + column] = ordered[column].to_numpy()
    return result, diagnostic['cell_id_match_method']


def apply_janesick_author_override(result, author=None):
    """Use author labels for epithelial or uncertain initial labels in both studies."""
    from .epithelial_reclustering import epithelial_label_mask
    from .uncertain_reclustering import author_label_usable_mask
    from .automatic_cluster_annotation import cell_label_masks

    result = result.copy()
    result['janesick_author_override_target'] = False
    result['janesick_author_override_status'] = 'not_applicable'
    result['janesick_author_override_reason'] = 'not_applicable'
    if not result.dataset.isin({'Janesick_2023', 'Janesick_2025'}).all():
        return result
    initial = ['initial_cluster_refined_label', 'initial_cluster_broad_label']
    _require(result, initial, 'Janesick baseline')
    epithelial = result[initial].apply(epithelial_label_mask).any(axis=1)
    uncertain = ~cell_label_masks(result.initial_cluster_refined_label)[1]
    uncertain |= result.initial_cluster_broad_label.fillna('').str.strip().str.casefold().isin(
        {'', 'uncertain', 'unknown', 'missing', 'unassigned', 'unlabeled', 'nan', 'none', '<na>'})
    target = epithelial | uncertain
    result.loc[target, 'janesick_author_override_reason'] = 'initial_uncertain_or_missing'
    result.loc[epithelial, 'janesick_author_override_reason'] = 'initial_epithelial'
    if author is None:
        _require(result, ['author_refined_label', 'author_broad_label'], 'Janesick baseline')
        refined = result.author_refined_label
        if target.any() and refined.fillna('').str.strip().str.casefold().isin({'', 'nan', 'none', '<na>'}).all():
            raise ValueError(f'{result.dataset.iloc[0]} / {result["sample"].iloc[0]}: all saved author labels are blank. '
                             'Restore the original author-label source before exporting; no version was published.')
        usable = target & author_label_usable_mask(refined)
        broad = result.author_broad_label.fillna('').str.strip().replace('', 'Uncertain')
    else:
        from .janesick_author_labels import match_author_labels
        aligned = match_author_labels(author, result.cell_id)
        aligned.index = result.index
        for column in aligned:
            result['janesick_author_source_' + column] = aligned[column]
        refined = aligned.author_refined_label
        broad = aligned.author_broad_label.replace('', 'Uncertain')
        usable = target & aligned.author_label_usable
    if set(broad[usable]) - set(BROAD_COLORS):
        raise ValueError('Janesick author broad labels contain unknown PI classes.')
    result.loc[target, 'janesick_author_override_target'] = True
    result.loc[target, 'janesick_author_override_status'] = 'author_missing_or_unusable'
    result.loc[target, ['final_refined_label', 'final_broad_label']] = 'Uncertain'
    result.loc[target, 'final_label_source'] = 'janesick_author_missing'
    result.loc[usable, 'final_refined_label'] = refined[usable]
    result.loc[usable, 'final_broad_label'] = broad[usable]
    result.loc[usable, 'final_label_source'] = 'janesick_author'
    result.loc[usable, 'janesick_author_override_status'] = 'author_label_used'
    if author is not None:
        result.loc[target & ~aligned.author_workbook_matched, 'janesick_author_override_status'] = 'author_workbook_unmatched'
    return result


def export_updated_labels(baseline_dir, annotation_dir, decision_path, output_dir,
                          cluster_key='epithelial_r0_7', sample_index=None,
                          janesick_author_workbook=None):
    """Publish a new complete version only after every sample passes validation."""
    baseline_dir, annotation_dir, decision_path, output_dir = map(
        Path, (baseline_dir, annotation_dir, decision_path, output_dir))
    baseline_dir, output_dir = baseline_dir.resolve(), output_dir.resolve()
    if output_dir.exists() or output_dir.is_relative_to(baseline_dir) or baseline_dir.is_relative_to(output_dir):
        raise FileExistsError('Choose a new output directory separate from the original final-label folder.')
    proposal_path = annotation_dir / 'all_cluster_annotations.csv'
    hashes = {str(p.resolve()): file_hash(p) for p in (proposal_path, decision_path)}
    if sample_index is not None:
        hashes[str(Path(sample_index).resolve())] = file_hash(sample_index)
    authors = {}
    if janesick_author_workbook is not None:
        from .janesick_author_labels import load_workbook
        hashes[str(Path(janesick_author_workbook).resolve())] = file_hash(janesick_author_workbook)
        authors = load_workbook(janesick_author_workbook)
    plan = load_cluster_updates(proposal_path, decision_path, cluster_key)
    inventory = input_inventory(baseline_dir, annotation_dir, plan, sample_index)
    if not inventory.status.eq('ready').all():
        raise FileNotFoundError('Required baseline/epithelial files are missing. Check the input inventory.')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent, prefix='.updated-final-labels-') as tmp:
        stage = Path(tmp) / 'labels'
        stage.mkdir()
        if authors:
            vocabulary = []
            for sample, table in authors.items():
                group = table.groupby(['author_raw_label', 'author_refined_label', 'author_broad_label',
                                       'author_label_usable', 'author_workbook_sheet'], dropna=False).size().rename('n_author_cells').reset_index()
                group.insert(0, 'sample', sample)
                vocabulary.append(group)
            pd.concat(vocabulary).to_csv(stage / 'janesick_author_label_mapping.csv', index=False)
        summaries, distributions, transitions = [], [], []
        for i, item in enumerate(inventory.itertuples(index=False), 1):
            print(f'[{i}/{len(inventory)}] {item.dataset} / {item.sample}', flush=True)
            path = Path(item.baseline_path)
            baseline_hash = file_hash(path)
            baseline = pd.read_csv(path, dtype=str, keep_default_na=False)
            if not baseline.dataset.eq(item.dataset).all() or not baseline['sample'].eq(item.sample).all():
                raise ValueError(f'Baseline identifiers disagree with folder: {path}')
            sample_plan = plan.loc[plan.dataset.eq(item.dataset) & plan['sample'].eq(item.sample)]
            cell_hash = file_hash(item.epithelial_cells_path) if len(sample_plan) else ''
            updates = load_cell_updates(annotation_dir, sample_plan) if len(sample_plan) else pd.DataFrame()
            result, method = apply_cell_updates(baseline, updates)
            author = None
            if item.dataset == 'Janesick_2023' and janesick_author_workbook is not None:
                author = authors[item.sample]
            result = apply_janesick_author_override(result, author)
            destination = stage / item.dataset / item.sample / 'final_cell_labels.csv.gz'
            destination.parent.mkdir(parents=True)
            result.to_csv(destination, index=False)
            if file_hash(path) != baseline_hash or (cell_hash and file_hash(item.epithelial_cells_path) != cell_hash):
                raise ValueError('An input changed during export; retry with stable inputs.')
            changed = result[FINAL_COLUMNS[:2]].ne(baseline[FINAL_COLUMNS[:2]]).any(axis=1)
            summaries.append(dict(dataset=item.dataset, sample=item.sample, n_cells=len(result),
                                  n_updated_clusters=len(sample_plan),
                                  n_updated_cells=len(updates) + int(result.janesick_author_override_target.sum()),
                                  n_epithelial_recluster_updated_cells=len(updates),
                                  n_janesick_author_target_cells=int(result.janesick_author_override_target.sum()),
                                  n_janesick_epithelial_override_cells=int(result.janesick_author_override_reason.eq('initial_epithelial').sum()),
                                  n_janesick_uncertain_fallback_cells=int(result.janesick_author_override_reason.eq('initial_uncertain_or_missing').sum()),
                                  n_janesick_author_labeled_cells=int(result.janesick_author_override_status.eq('author_label_used').sum()),
                                  n_janesick_author_missing_cells=int(result.janesick_author_override_status.isin(['author_missing_or_unusable', 'author_workbook_unmatched']).sum()),
                                  n_janesick_author_unmatched_cells=int(result.janesick_author_override_status.eq('author_workbook_unmatched').sum()),
                                  n_changed_labels=int(changed.sum()),
                                  n_final_uncertain=int(result.final_broad_label.eq('Uncertain').sum()),
                                  cell_id_match_method=method, baseline_sha256=baseline_hash,
                                  epithelial_cells_sha256=cell_hash))
            counts = result.final_broad_label.value_counts()
            if set(counts.index) - set(BROAD_COLORS):
                raise ValueError(f'Unknown final broad labels in {path}')
            for broad in BROAD_COLORS:
                distributions.append(dict(dataset=item.dataset, sample=item.sample, broad_cell_type=broad,
                                          n_cells=int(counts.get(broad, 0)), n_sample_cells=len(result),
                                          fraction_of_sample=counts.get(broad, 0) / len(result)))
            transition = result.groupby(['pre_epithelial_final_broad_label', 'final_broad_label']).size().rename('n_cells').reset_index()
            transition['dataset'], transition['sample'] = item.dataset, item.sample
            transitions.append(transition)
        if any(file_hash(p) != digest for p, digest in hashes.items()):
            raise ValueError('Cluster annotations or manual decisions changed during export.')
        summary = pd.DataFrame(summaries)
        summary.to_csv(stage / 'epithelial_update_summary.csv', index=False)
        pd.DataFrame(distributions).to_csv(stage / 'final_broad_cell_type_distribution.csv', index=False)
        pd.concat(transitions).to_csv(stage / 'broad_label_transitions.csv', index=False)
        plan.to_csv(stage / 'cluster_update_plan.csv', index=False)
        inventory.to_csv(stage / 'input_inventory.csv', index=False)
        (stage / 'manual_epithelial_decisions_snapshot.csv').write_bytes(decision_path.read_bytes())
        (stage / 'update_manifest.json').write_text(json.dumps(dict(
            created_at_utc=datetime.now(timezone.utc).isoformat(), cluster_key=cluster_key,
            baseline_directory=str(baseline_dir), input_sha256=hashes,
            n_clusters=len(plan), n_updated_cells=int(summary.n_updated_cells.sum()),
            annotation_status='provisional; confidence and alignment flags retained',
            policy=('manual including Uncertain > expression proposal for all saved epithelial clusters; '
                    'both Janesick datasets use author labels for initially epithelial or uncertain/missing cells; '
                    'missing authors become Uncertain; other certain non-epithelial cells unchanged'),
        ), indent=2) + '\n')
        stage.rename(output_dir)
    return summary


def plot_distributions(distribution, output_dir):
    """Match final-label workflow's per-study counts and percentage plots for all sample cells."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter, StrMethodFormatter
    from textwrap import fill

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for dataset, rows in distribution.groupby('dataset', sort=False):
        counts = rows.pivot(index='sample', columns='broad_cell_type', values='n_cells')
        counts = counts.reindex(columns=list(BROAD_COLORS), fill_value=0).sort_index()
        percentages = counts.div(counts.sum(axis=1), axis=0) * 100
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=(16, max(4.2, .62 * len(counts) + 2.4)))
        for ax, values, title in zip(axes, [counts, percentages], ['Cell counts', 'Sample composition']):
            left = np.zeros(len(counts))
            for broad, color in BROAD_COLORS.items():
                ax.barh(np.arange(len(counts)), values[broad], left=left, color=color, label=broad)
                left += values[broad].to_numpy()
            ax.set_title(title)
            ax.spines[['top', 'right']].set_visible(False)
        axes[0].set_yticks(np.arange(len(counts)))
        axes[0].set_yticklabels([fill(s, 46) for s in counts.index], fontsize=9)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Number of cells')
        axes[0].xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        axes[1].set_xlim(0, 100)
        axes[1].set_xlabel('Percentage of all sample cells')
        axes[1].xaxis.set_major_formatter(PercentFormatter(xmax=100))
        fig.suptitle(f'{dataset}: updated final broad cell types')
        fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=4, frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, .10, 1, .95))
        path = output_dir / f'{dataset}_updated_final_broad_distribution.png'
        fig.savefig(path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        paths.append(path)
    return paths
