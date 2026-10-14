"""Annotate saved epithelial clusters; preserve source data and final labels."""
from pathlib import Path
from datetime import datetime, timezone
import html
import json
import sys

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem
from scipy.stats import hypergeom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from xenium_pipeline.epithelial_marker_review import CORE_LINEAGE_GENES, file_hash

KEY = 'epithelial_r0_7'
CONFIG = dict(resolution=KEY, min_de_genes=3, program_fdr=0.05,
              epithelial_majority=0.50, epithelial_low=0.30,
              alternative_conflict=0.30, dominant_core=0.50,
              high_epithelial=0.80, high_alternative_max=0.20,
              min_phenotype_de_genes=2, umap_display_max_cells=100000)
EPI_BROAD = 'Epithelial, normal/tumor unresolved'
BROAD = {'Fibroblast/stromal': 'Stromal Cells', 'Myeloid/macrophage': 'pan-APCs',
         'T cells': 'T Cells', 'B cells': 'B Cells', 'Endothelial': 'Stromal Cells',
         'Smooth muscle': 'Smooth Muscle Cells'}
PALETTE = {EPI_BROAD: '#287AA4', 'Uncertain': '#989898', 'Stromal Cells': '#38915D',
           'pan-APCs': '#B66090', 'T Cells': '#B59A20', 'B Cells': '#784896',
           'Smooth Muscle Cells': '#C96845'}
IDS = ['dataset', 'sample', 'cluster_key', 'cluster']
LABELS = ['proposed_refined_label', 'proposed_broad_label', 'lineage_confidence',
          'malignancy_status', 'decision_status', 'annotation_reason']


def bh(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.minimum.accumulate((values[order] * len(values) /
                                     np.arange(1, len(values) + 1))[::-1])[::-1]
    result = np.empty(len(values))
    result[order] = np.minimum(adjusted, 1)
    return result


def decide(core, de_programs, de_genes):
    """Conservative lineage calls; expression states never establish malignancy."""
    e = core['Epithelial']
    alternatives = [p for p in CORE_LINEAGE_GENES if p != 'Epithelial']
    strong = [p for p in alternatives if de_programs.loc[p, 'qualifies_de_program']]
    core_conflicts = [p for p in alternatives if core[p] >= CONFIG['dominant_core']]
    candidates = list(dict.fromkeys(strong + core_conflicts))
    basal = sorted(de_genes & {'KRT5', 'KRT14', 'KRT6B', 'KRT17'})
    contractile = sorted(de_genes & {'ACTA2', 'ACTG2', 'MYH11', 'MYLK', 'TAGLN'})
    myo = e >= CONFIG['epithelial_majority'] and len(basal) >= 2 and len(contractile) >= 2
    # Epithelial plus contractile expression can form one coherent phenotype.
    conflicts = [p for p in candidates if not (myo and p == 'Smooth muscle')]
    if conflicts:
        if len(conflicts) == 1 and e < CONFIG['epithelial_low']:
            p = conflicts[0]
            competing = max([core[x] for x in alternatives if x != p], default=0)
            if core[p] >= CONFIG['dominant_core'] and competing < CONFIG['alternative_conflict'] and p in strong:
                return p + ' cells', BROAD[p], 'moderate', 'not_assessed_non_epithelial', \
                    f'{p} DE enrichment and majority core detection; epithelial core below 30%.'
            return p + ' candidate; lineage unresolved', 'Uncertain', 'low', 'not_assessed_lineage_unresolved', \
                f'{p} is a candidate, but core detection or competing-lineage criteria prevent a single-lineage call.'
        names = (['Epithelial'] if e >= CONFIG['epithelial_low'] else []) + conflicts
        return 'Mixed ' + ' + '.join(names) + ' evidence', 'Uncertain', 'low', 'not_assessed_lineage_unresolved', \
            'Multiple lineage signals; cluster-level summaries cannot distinguish admixture, coexpression, or contamination.'
    if e >= CONFIG['epithelial_majority']:
        max_other = max(core[p] for p in alternatives)
        confidence = 'high' if e >= CONFIG['high_epithelial'] and max_other < CONFIG['high_alternative_max'] else 'moderate'
        if myo:
            label = 'Basal/myoepithelial-like epithelial cells'
            confidence = 'moderate'
        elif len(basal) >= 2:
            label = 'Basal-like epithelial cells'
        elif de_programs.loc['Glandular/luminal', 'qualifies_phenotype_program']:
            label = 'Glandular/luminal-like epithelial cells'
        else:
            label = 'Epithelial cells'
        if len(de_genes & {'MKI67', 'TOP2A', 'CENPF', 'PCLAF'}) >= 2:
            label = 'Proliferative ' + label[0].lower() + label[1:]
        return label, EPI_BROAD, confidence, 'normal_vs_tumor_unresolved', \
            'Majority epithelial core detection without a qualifying competing lineage; phenotype comes from DE evidence.'
    if e >= CONFIG['epithelial_low'] and max(core[p] for p in alternatives) < CONFIG['alternative_conflict']:
        return 'Epithelial candidate; weak core detection', 'Uncertain', 'low', 'normal_vs_tumor_unresolved', \
            'Epithelial core detection is 30-50%; no strong competing lineage, but majority support is absent.'
    return 'Unresolved lineage', 'Uncertain', 'low', 'not_assessed_lineage_unresolved', \
        'No lineage meets the combined DE and core-detection rules.'


def self_check():
    assert np.allclose(bh([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04])
    names = list(CORE_LINEAGE_GENES) + ['Glandular/luminal', 'Myoepithelial', 'Tumor-associated', 'Proliferation']
    table = pd.DataFrame(False, index=names, columns=['qualifies_de_program', 'qualifies_phenotype_program'])
    core = dict.fromkeys(CORE_LINEAGE_GENES, 0.0)
    core['Epithelial'] = .9
    assert decide(core, table, set())[1:3] == (EPI_BROAD, 'high')
    table.loc['Myeloid/macrophage', 'qualifies_de_program'] = True
    core['Myeloid/macrophage'] = .8
    assert decide(core, table, set())[1] == 'Uncertain'
    core['Epithelial'] = .1
    assert decide(core, table, set())[1] == 'pan-APCs'
    table.loc['Myeloid/macrophage', 'qualifies_de_program'] = False
    table.loc['Smooth muscle', 'qualifies_de_program'] = True
    core.update({'Epithelial': .9, 'Myeloid/macrophage': 0, 'Smooth muscle': .8})
    assert decide(core, table, {'KRT5', 'KRT14', 'MYH11', 'MYLK'})[0].startswith('Basal/myoepithelial')


def run(root, output_dir=None):
    ROOT = Path(root).expanduser().resolve()
    OUT = Path(output_dir) if output_dir else ROOT / 'epithelial_cluster_annotation'
    OUT.mkdir(parents=True, exist_ok=True)
    from xenium_pipeline.storage import epithelial_roots
    ROOTS = epithelial_roots(ROOT)
    self_check()
    if (OUT / 'run_manifest.json').exists():
        raise FileExistsError('Completed review exists. Preserve it; choose a new output folder for another version.')
    paths = [p for dataset, checkpoint_root in ROOTS.items()
             for p in sorted((checkpoint_root / dataset).glob('*/epithelial_expression.h5ad'))]
    if not paths:
        raise FileNotFoundError('No saved epithelial checkpoints found.')
    morphology_path = ROOT / 'epithelial_morphology_review/assistant_review/cluster_metrics.csv'
    if morphology_path.is_file():
        morphology = pd.read_csv(morphology_path, dtype={'cluster':str})
    else:
        morphology = summarize_morphology(ROOT, paths)
        morphology_path.parent.mkdir(parents=True, exist_ok=True)
        morphology.to_csv(morphology_path, index=False)
    assert not morphology.duplicated(['dataset', 'sample', 'cluster']).any()
    expected_samples = set(map(tuple, morphology[['dataset', 'sample']].drop_duplicates().to_numpy()))
    assert {(p.parent.parent.name, p.parent.name) for p in paths} == expected_samples
    rows, program_rows, inventory, records = [], [], [], []
    for index, checkpoint in enumerate(paths, 1):
        dataset, sample = checkpoint.parent.parent.name, checkpoint.parent.name
        folder = checkpoint.parent
        print(f'[{index}/{len(paths)}] {dataset} / {sample}: checking saved evidence', flush=True)
        line_dir, de_dir = folder / 'lineage_review', folder / 'expression_review' / KEY
        lm = json.loads((line_dir / 'program_metadata.json').read_text())
        dm = json.loads((de_dir / 'marker_metadata.json').read_text())
        digest = file_hash(checkpoint)
        assert digest == lm['source_hash'] == dm['source_hash'], f'Stale marker/lineage cache: {sample}'
        assert lm['n_non_gene_features'] == 0 and lm['min_detected'] == 2
        with h5py.File(checkpoint, 'r') as handle:
            obs, var = read_elem(handle['obs']), read_elem(handle['var'])
            coordinates = np.asarray(handle['obsm/X_umap'])
        assert var.feature_type.eq('Gene Expression').all()
        assert obs.index.is_unique and not (obs.index.astype(str).str.strip() == '').any()
        groups = obs[KEY].astype(str)
        assert groups.notna().all() and not groups.isin(['nan', '', '<NA>']).any()
        genes = set(var.gene_symbol.astype(str).str.upper())
        lineage = pd.read_csv(line_dir / 'cluster_lineage_evidence.csv', dtype={'cluster': str})
        lineage = lineage.loc[lineage.cluster_key.eq(KEY)].copy()
        assert not lineage.duplicated(['cluster', 'program']).any()
        assert lineage.n_non_gene_features.eq(0).all()
        core_cov = pd.read_csv(line_dir / 'core_gene_coverage.csv')
        assert core_cov.n_available_core_genes.ge(2).all()
        de = pd.read_csv(de_dir / 'all_marker_genes.csv.gz', dtype={'cluster': str})
        assert set(de.cluster) == set(groups)
        assert de.passes_marker_filter.isin([True, False]).all()
        de['gene'] = de.gene.str.upper()
        assert set(de.gene) <= genes
        sample_morph = morphology.loc[morphology.dataset.eq(dataset) & morphology['sample'].eq(sample)].set_index('cluster')
        assert set(sample_morph.index) == set(groups)
        morph_cells = pd.read_csv(ROOT / 'epithelial_morphology_review' / dataset / sample / 'cell_morphology_evidence.csv.gz',
                                  usecols=['cell_id', 'cluster'], dtype=str).set_index('cell_id')
        assert morph_cells.index.is_unique and set(morph_cells.index) == set(obs.index.astype(str))
        assert np.array_equal(morph_cells.reindex(obs.index.astype(str)).cluster.to_numpy(), groups.to_numpy())
        del morph_cells
        local = []
        for cluster in sorted(groups.unique(), key=lambda x: int(x)):
            n = int(groups.eq(cluster).sum())
            evidence = lineage.loc[lineage.cluster.eq(cluster)].set_index('program')
            assert len(evidence) == len(CORE_LINEAGE_GENES) - 1 and evidence.n_cells.eq(n).all()
            assert int(sample_morph.loc[cluster, 'n_cells']) == n
            ef = evidence.fraction_epithelial_only + evidence.fraction_both
            assert np.allclose(ef, ef.iloc[0])
            core = {'Epithelial': float(ef.iloc[0]), **(evidence.fraction_lineage_only + evidence.fraction_both).to_dict()}
            selected = de.loc[de.cluster.eq(cluster) & de.passes_marker_filter].sort_values('marker_rank_score', ascending=False)
            marker_genes = set(selected.gene)
            entries = []
            for program, dictionary in lm['programs'].items():
                requested = set(g.upper() for g in dictionary) | set(CORE_LINEAGE_GENES.get(program, ()))
                available = requested & genes
                hits = available & marker_genes
                p = float(hypergeom.sf(len(hits) - 1, len(genes), len(available), len(marker_genes))) if available and marker_genes else 1.0
                entries.append(dict(dataset=dataset, sample=sample, cluster_key=KEY, cluster=cluster,
                                    program=program, dictionary_source=lm['origins'][program],
                                    n_tested_genes=len(genes), n_de_genes=len(marker_genes),
                                    n_available_program_genes=len(available), n_de_program_genes=len(hits),
                                    de_program_genes='; '.join(selected.loc[selected.gene.isin(hits), 'gene'].drop_duplicates()),
                                    enrichment_p_value=p, fraction_core_detected=core.get(program, np.nan)))
            pt = pd.DataFrame(entries)
            pt['enrichment_q_value'] = bh(pt.enrichment_p_value)
            pt['qualifies_de_program'] = pt.n_de_program_genes.ge(CONFIG['min_de_genes']) & pt.enrichment_q_value.le(CONFIG['program_fdr'])
            pt['qualifies_phenotype_program'] = pt.n_de_program_genes.ge(CONFIG['min_phenotype_de_genes']) & pt.enrichment_q_value.le(CONFIG['program_fdr'])
            pt = pt.sort_values(['enrichment_q_value', 'n_de_program_genes', 'program'], ascending=[True, False, True])
            pt['candidate_rank'] = np.arange(1, len(pt) + 1)
            program_rows.extend(pt.to_dict('records'))
            refined, broad, confidence, malignancy, reason = decide(core, pt.set_index('program'), marker_genes)
            top15 = '; '.join(selected.gene.head(15))
            phenotype = pt.loc[pt.program.eq('Tumor-associated')].iloc[0]
            rec = dict(dataset=dataset, sample=sample, cluster_key=KEY, cluster=cluster, n_cells=n,
                       proposed_refined_label=refined, proposed_broad_label=broad, lineage_confidence=confidence,
                       malignancy_status=malignancy, decision_status='proposed_not_accepted', annotation_reason=reason,
                       n_de_genes=len(marker_genes), top_15_de_genes=top15,
                       proliferation_de_genes='; '.join(sorted(marker_genes & {'MKI67', 'TOP2A', 'CENPF', 'PCLAF'})),
                       tumor_associated_de_genes=phenotype.de_program_genes,
                       tumor_associated_program_enriched=bool(phenotype.qualifies_de_program),
                       ranked_program_candidates='; '.join(pt.loc[pt.n_de_program_genes.gt(0), 'program'].head(3)),
                       checkpoint_sha256=digest, sample_sharing_status='not_assessable_sample_specific_partition',
                       he_evidence_status='not_used_alignment_unverified',
                       **{'fraction_core_detected:' + p: value for p, value in core.items()})
            for metric in ['polygon_coverage', 'polygon_nuclear_area_um2', 'polygon_eccentricity', 'area_ratio',
                           'eccentricity_shift', 'density_ratio']:
                rec[metric] = sample_morph.loc[cluster, metric]
            rec['morphology_note'] = (f'Polygon area {rec["area_ratio"]:.2f}x sample median; '
                                      f'eccentricity shift {rec["eccentricity_shift"]:+.3f}. Supporting context, not a malignancy rule.')
            local.append(rec)
        table = pd.DataFrame(local)
        destination = OUT / dataset / sample
        destination.mkdir(parents=True, exist_ok=True)
        table.to_csv(destination / 'cluster_annotations.csv', index=False)
        cells = pd.DataFrame({'cell_id': obs.index.astype(str), 'cluster': groups.to_numpy()})
        cells = cells.merge(table[IDS + LABELS], on='cluster', how='left', validate='many_to_one')
        assert len(cells) == len(obs) and cells.proposed_refined_label.notna().all()
        assert cells.groupby('cluster').size().to_dict() == table.set_index('cluster').n_cells.to_dict()
        cells = cells[['dataset', 'sample', 'cell_id', 'cluster_key', 'cluster'] + LABELS]
        cells.to_csv(destination / 'cell_annotations.csv.gz', index=False)
        show = np.arange(len(cells))
        if len(show) > CONFIG['umap_display_max_cells']:
            show = np.sort(np.random.default_rng(0).choice(show, CONFIG['umap_display_max_cells'], replace=False))
        assert len(coordinates) == len(cells) and np.isfinite(coordinates).all()
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
        code = pd.Categorical(groups, categories=table.cluster).codes
        axes[0].scatter(coordinates[show, 0], coordinates[show, 1], c=code[show], cmap='tab20', s=1, rasterized=True)
        for cluster in table.cluster:
            xy = np.median(coordinates[groups.eq(cluster)], axis=0)
            axes[0].text(*xy, cluster, fontsize=9, ha='center', bbox=dict(facecolor='white', alpha=.8, edgecolor='none', pad=1))
        axes[1].scatter(coordinates[show, 0], coordinates[show, 1],
                        c=cells.proposed_broad_label.map(PALETTE).iloc[show], s=1, rasterized=True)
        for label in PALETTE:
            if cells.proposed_broad_label.eq(label).any():
                axes[1].scatter([], [], color=PALETTE[label], label=label, s=16)
        axes[1].legend(loc='upper center', bbox_to_anchor=(.5, -.1), fontsize=8)
        for ax in axes:
            ax.set(xlabel='Saved UMAP 1', ylabel='Saved UMAP 2')
        axes[0].set_title('Existing epithelial r0.7 cluster IDs')
        axes[1].set_title('Proposed broad label; not accepted final labels')
        fig.suptitle(f'{dataset} / {sample}\n{len(cells):,} cells; {len(show):,} displayed', fontsize=11)
        fig.savefig(destination / 'annotation_umap.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        rows.extend(local)
        inventory.append(dict(dataset=dataset, sample=sample, checkpoint_path=str(checkpoint.relative_to(ROOT)),
                              checkpoint_sha256=digest, n_cells=len(cells), n_clusters=len(table),
                              marker_cache_hash_matches=True, lineage_cache_hash_matches=True,
                              morphology_membership_matches=True, every_cell_labeled=True,
                              n_umap_display_cells=len(show)))
        records.append((dataset, sample, table))
        pd.DataFrame(inventory).to_csv(OUT / 'input_audit.csv', index=False)
        print(f'  Saved {len(table)} cluster proposals and {len(cells):,} cell rows.', flush=True)
    combined, programs = pd.DataFrame(rows), pd.DataFrame(program_rows)
    assert len(combined) == len(morphology) and combined.n_cells.sum() == morphology.n_cells.sum()
    assert not combined.duplicated(IDS).any()
    combined.to_csv(OUT / 'all_cluster_annotations.csv', index=False)
    programs.to_csv(OUT / 'cluster_program_evidence.csv', index=False)
    distribution = combined.groupby(['dataset', 'sample', 'proposed_broad_label'], as_index=False).agg(n_clusters=('cluster', 'size'), n_cells=('n_cells', 'sum'))
    distribution['fraction_sample_cells'] = distribution.n_cells / distribution.groupby(['dataset', 'sample']).n_cells.transform('sum')
    distribution.to_csv(OUT / 'annotation_distribution_by_sample.csv', index=False)
    totals = combined.groupby(['proposed_broad_label', 'lineage_confidence'], as_index=False).agg(n_clusters=('cluster', 'size'), n_cells=('n_cells', 'sum'))
    totals.to_csv(OUT / 'annotation_summary.csv', index=False)
    pivot = distribution.pivot(index=['dataset', 'sample'], columns='proposed_broad_label', values='fraction_sample_cells').fillna(0)
    labels = [p for p in PALETTE if p in pivot]
    fig, ax = plt.subplots(figsize=(13, 10), constrained_layout=True)
    left = np.zeros(len(pivot))
    for label in labels:
        values = pivot[label].to_numpy() * 100
        ax.barh(np.arange(len(pivot)), values, left=left, label=label, color=PALETTE[label])
        left += values
    ax.set_yticks(np.arange(len(pivot)), [d + ' / ' + s.replace('Xenium_V1_FFPE_Human_Breast_', '') for d, s in pivot.index], fontsize=8)
    ax.invert_yaxis()
    ax.set(xlim=(0, 100), xlabel='Percentage of epithelial-reclustering input cells', title='Proposed cluster labels propagated to cells')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.06), ncol=2, fontsize=8)
    fig.savefig(OUT / 'annotation_distribution_by_sample.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    build_review(records, totals, OUT)
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), config=CONFIG,
                    annotation_type='cluster_level_rule_based_proposal', n_samples=len(paths),
                    n_clusters=len(combined), n_cells=int(combined.n_cells.sum()),
                    input_roots={d: str(p.relative_to(ROOT)) for d, p in ROOTS.items()},
                    input_audit_sha256=file_hash(OUT / 'input_audit.csv'),
                    script_sha256=file_hash(Path(__file__)), final_labels_modified=False,
                    new_clustering_run=False, sample_sharing_used=False)
    (OUT / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(totals.to_string(index=False))
    print(f'Complete: {len(paths)} samples, {len(combined)} clusters, {combined.n_cells.sum():,} cells.', flush=True)


def build_review(records, totals, OUT):
    esc = lambda value: html.escape(str(value))
    sections = []
    for dataset, sample, table in records:
        relative = f'{dataset}/{sample}'
        details = []
        for r in table.to_dict('records'):
            core = '; '.join(f'{p}: {r["fraction_core_detected:" + p]:.1%}' for p in CORE_LINEAGE_GENES)
            details.append(f'<details><summary>Cluster {esc(r["cluster"])} | {r["n_cells"]:,} cells | {esc(r["proposed_refined_label"])}</summary>'
                           f'<p><b>Broad:</b> {esc(r["proposed_broad_label"])}. <b>Lineage confidence:</b> {esc(r["lineage_confidence"])}. '
                           f'<b>Malignancy:</b> {esc(r["malignancy_status"])}.</p>'
                           f'<p>{esc(r["annotation_reason"])}</p><p><b>Top 15 qualifying DE genes:</b> {esc(r["top_15_de_genes"] or "None pass the saved filters")}</p>'
                           f'<p><b>Core-positive fractions:</b> {esc(core)}</p><p><b>Ranked program hints:</b> {esc(r["ranked_program_candidates"] or "None")}</p>'
                           f'<p><b>Morphology:</b> {esc(r["morphology_note"])}</p>'
                           f'<p><b>Tumor-associated dictionary DE genes:</b> {esc(r["tumor_associated_de_genes"] or "None")}. This dictionary overlap does not establish malignancy.</p></details>')
        sections.append(f'<section><h2>{esc(dataset)} / {esc(sample)}</h2><p>{len(table)} clusters; {table.n_cells.sum():,} cells. '
                        f'<a href="{relative}/cell_annotations.csv.gz">Per-cell proposals</a> | '
                        f'<a href="{relative}/cluster_annotations.csv">Cluster proposals</a></p>'
                        f'<a href="{relative}/annotation_umap.png"><img loading="lazy" src="{relative}/annotation_umap.png" alt="Saved UMAP and proposed labels"></a>' + ''.join(details) + '</section>')
    options = ''.join(f'<option value="{i}">{esc(d)} / {esc(s)}</option>' for i, (d, s, _) in enumerate(records))
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Epithelial Cluster Annotation Review</title><style>
body{font:16px system-ui;color:#222;margin:24px auto;padding:0 20px;max-width:1200px;line-height:1.55}h1{font-size:26px}h2{font-size:20px;overflow-wrap:anywhere}img{max-width:100%;height:auto}select{max-width:100%;padding:8px;font:inherit}details{border-top:1px solid #ccc;padding:12px 0}summary{cursor:pointer;font-weight:600;overflow-wrap:anywhere}p{overflow-wrap:anywhere}section{margin-top:24px}table{border-collapse:collapse}td,th{padding:6px 12px;border-bottom:1px solid #ddd;text-align:left}a{color:#126c94}</style>
<h1>Epithelial Cluster Annotation Review</h1>
<p>All 182 sample-specific clusters at resolution 0.7. These are proposed cluster identities, propagated to every member cell; existing final labels are unchanged.</p>
<p><b>Confidence refers to lineage support, not tumor status or per-cell accuracy.</b> Normal versus tumor remains unresolved for epithelial cells. Mixed labels describe competing evidence, not confirmed doublets.</p>
<p><a href="README.md">Methods and all column meanings</a> | <a href="all_cluster_annotations.csv">All cluster proposals</a> | <a href="cluster_program_evidence.csv">Detailed program evidence</a></p>
<h2>Overall Summary</h2>''' + totals.to_html(index=False, border=0) + '''
<p>proposed_broad_label: candidate broad identity; lineage_confidence: heuristic strength of lineage support; n_clusters: cluster count; n_cells: number of member cells inheriting these proposals.</p>
<img src="annotation_distribution_by_sample.png" alt="Cell-weighted annotation distributions per sample">
<p>Bars include all cells supplied to epithelial reclustering. Uncertain includes mixed-lineage and weak-evidence clusters. The epithelial category retains normal/tumor uncertainty.</p>
<label for="sample">Sample: </label><select id="sample">''' + options + '</select>' + ''.join(sections) + '''
<script>const sections=[...document.querySelectorAll('section')]; const sample=document.querySelector('#sample');
function render(){sections.forEach((s,i)=>s.hidden=i!==Number(sample.value));} sample.addEventListener('change',render);render();</script></html>'''
    (OUT / 'review.html').write_text(page)




def summarize_morphology(root, checkpoint_paths):
    """Aggregate saved polygons and density using the accepted sample references."""
    root = Path(root)
    metrics = ['polygon_nuclear_area_um2', 'polygon_major_axis_um', 'polygon_minor_axis_um',
        'polygon_eccentricity', 'polygon_solidity', 'polygon_circularity',
        'reported_nucleus_area_um2', 'local_density_per_mm2', 'neighbor_nuclear_area_cv']
    rows = []
    for checkpoint in checkpoint_paths:
        checkpoint = Path(checkpoint)
        dataset, sample = checkpoint.parent.parent.name, checkpoint.parent.name
        cells = pd.read_csv(root / 'epithelial_morphology_review' / dataset / sample / 'cell_morphology_evidence.csv.gz',
            usecols=['cluster', 'polygon_qc', 'mask_qc', *metrics], dtype={'cluster':str})
        reference = cells[metrics].median()
        for cluster, group in cells.groupby('cluster', sort=False):
            medians = group[metrics].median()
            rows.append(dict(dataset=dataset, sample=sample, cluster=cluster,
                n_cells=len(group), fraction_sample=len(group) / len(cells),
                polygon_coverage=group.polygon_qc.eq('ok').mean(), mask_coverage=group.mask_qc.eq('ok').mean(),
                area_ratio=medians.polygon_nuclear_area_um2 / reference.polygon_nuclear_area_um2,
                density_ratio=medians.local_density_per_mm2 / reference.local_density_per_mm2,
                eccentricity_shift=medians.polygon_eccentricity - reference.polygon_eccentricity,
                neighbor_cv_shift=medians.neighbor_nuclear_area_cv - reference.neighbor_nuclear_area_cv,
                **medians.to_dict()))
    return pd.DataFrame(rows)
