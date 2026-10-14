"""Command-line entry points for the accepted Xenium workflow."""
from pathlib import Path
import argparse
import importlib
import os
import subprocess
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('10x', 'Janesick_2023', 'Janesick_2025', 'Liu_2025', 'Bilous_2026')
TARGETS = ('tumor_cells', 'stromal_cells', 'normal_epithelial_cells', 't_cells', 'pan_apc_b')
ACTIONS = {
    'prepare': ('he', 'tiles', 'select', 'markers'),
    'annotate': ('score', 'authors', 'cluster', 'evidence', 'initial', 'uncertain',
                 'recluster-labels', 'reference-labels', 'reference-targets', 'epithelial', 'morphology', 'epithelial-proposals', 'epithelial-labels'),
    'models': ('features', 'targets', 'train', 'metrics', 'crc-combined'),
    'tcga': ('audit', 'count', 'groups', 'predict', 'cohort', 'resolution'),
    'analysis': ('importance', 'transfer', 'survival', 'spatial', 'reference'),
}

def run_module(name, arguments):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(CODE_ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    return subprocess.run([sys.executable, '-m', 'xenium_pipeline.' + name, *map(str, arguments)], env=env, check=True).returncode


def main(stage, argv=None):
    parser = argparse.ArgumentParser(description=f'STpath-BC {stage}: ' + ', '.join(ACTIONS[stage]))
    parser.add_argument('action', choices=ACTIONS[stage])
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('XENIUM_ROOT', CODE_ROOT)))
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--cell-type', choices=TARGETS)
    if stage == 'models':
        parser.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')
        parser.add_argument('--models', nargs='+')
        parser.add_argument('--batch-size', type=int, default=8)
        parser.add_argument('--workers', type=int, default=0)
    args, extra = parser.parse_known_args(argv)
    if extra and extra[0] == '--':
        extra = extra[1:]
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f'Data root does not exist: {root}')
    os.environ['XENIUM_ROOT'] = str(root)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    action = args.action
    delegated = (stage == 'tcga' and action != 'resolution') or (stage == 'models' and action in {'train', 'crc-combined'}) or (stage == 'prepare' and action in {'he', 'tiles'})
    if extra and not delegated:
        parser.error('Unrecognized arguments: ' + ' '.join(extra))
    if args.dataset and not ((stage == 'prepare' and action in {'he', 'tiles'}) or (stage == 'annotate' and action in {'score', 'cluster', 'epithelial'})):
        parser.error('--dataset applies to he, tiles, score, cluster, and epithelial.')
    if args.cell_type and not (stage == 'models' and action in {'targets', 'train'}):
        parser.error('--cell-type applies to targets and train.')
    if stage == 'prepare':
        if action in {'he', 'tiles'}:
            if not args.dataset:
                parser.error('Provide --dataset for he or tiles.')
            return run_module('prepare_he' if action == 'he' else 'create_tiles', ['--dataset', args.dataset, *extra])
        module = importlib.import_module('xenium_pipeline.preparation_workflow')
        return getattr(module, {'select':'select_tiles', 'markers':'prepare_markers'}[action])(root)
    if stage == 'annotate':
        if action == 'score':
            return run_module('all_sample_annotation', ['--root', root, '--write-full-score-matrix', '--skip-completed', *(['--datasets', args.dataset] if args.dataset else [])])
        if action == 'cluster':
            return run_module('cell_clustering', ['--root', root, '--min-counts-per-cell', 10, '--min-genes-per-cell', 5,
                '--min-cells-per-gene', 10, '--n-pcs', 20, '--n-neighbors', 20,
                '--leiden-resolutions', '1.3', '--default-resolution', '1.3', '--full-sample-clustering',
                *(['--datasets', args.dataset] if args.dataset else [])])
        if action == 'epithelial-proposals':
            return importlib.import_module('xenium_pipeline.epithelial_proposals').run(root)
        module = importlib.import_module('xenium_pipeline.annotation_workflow')
        if action == 'epithelial':
            return module.recluster_epithelial(root, [args.dataset] if args.dataset else None)
        functions = {'authors':'export_authors', 'evidence':'compute_evidence', 'initial':'annotate_initial',
            'uncertain':'recluster_uncertain', 'recluster-labels':'annotate_reclusters',
            'reference-labels':'finalize_labels', 'morphology':'compute_epithelial_morphology', 'reference-targets':'export_reference_targets', 'epithelial-labels':'apply_epithelial_review'}
        return getattr(module, functions[action])(root)
    if stage == 'models':
        if action == 'crc-combined':
            return run_module('train_crc_fixed_combined', ['--project-root', root, *extra])
        module = importlib.import_module('xenium_pipeline.model_workflow')
        if action == 'features':
            return module.extract_features(root, device=args.device, models=args.models, batch_size=args.batch_size, workers=args.workers)
        if action in {'targets', 'train'}:
            if not args.cell_type:
                parser.error('Provide --cell-type for targets or train.')
            return module.prepare_targets(root, args.cell_type) if action == 'targets' else module.train_target(root, args.cell_type, extra)
        return module.compare_models(root)
    if stage == 'tcga':
        if action == 'resolution':
            return importlib.import_module('xenium_pipeline.tcga_workflow').compare_resolution(root)
        if action == 'audit':
            return importlib.import_module('xenium_pipeline.tcga_workflow').audit_headers(root, extra)
        modules = {'count':'tcga_brca_count_batch',
            'groups':'tcga_brca_group_prediction', 'predict':'tcga_brca_prediction', 'cohort':'tcga_brca_cohort'}
        return run_module(modules[action], extra)
    module = importlib.import_module('xenium_pipeline.analysis_workflow')
    functions = {'importance':'compare_importance', 'transfer':'evaluate_transfer', 'survival':'analyze_survival',
        'spatial':'validate_spatial', 'reference':'generate_reference_figures'}
    return getattr(module, functions[action])(root)
