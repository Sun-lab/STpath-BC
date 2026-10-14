"""Paths and checkpoint bindings for the Xenium workflow."""
from pathlib import Path
import json

CODE_ROOT = Path(__file__).resolve().parents[1]

def load_layout(root=CODE_ROOT):
    file = Path(root) / 'storage_layout.json'
    if not file.is_file():
        file = CODE_ROOT / 'storage_layout.json'
    return json.loads(file.read_text())

def output_path(root, name):
    value = load_layout(root)['paths'][name]
    if isinstance(value, list):
        return [Path(root) / item for item in value]
    return Path(root) / value

def data_path(root, name):
    root = Path(root)
    path = output_path(root, name)
    if isinstance(path, list) or path.exists():
        return path
    value = load_layout(root).get('input_paths', {}).get(name)
    return root / value if value else path

def epithelial_roots(root):
    root = Path(root)
    layout = load_layout(root)
    canonical = output_path(root, 'epithelial_clustering')
    return {dataset: canonical if (canonical / dataset).is_dir()
            else root / layout['input_paths'][dataset + '_epithelial']
            for dataset in ('10x', 'Liu_2025', 'Bilous_2026')}

def model_runs(root=CODE_ROOT):
    return {spec['cell_type']: spec['run_id'] for spec in load_layout(root)['targets'].values()}

def clinical_file(run, filename):
    run = Path(run)
    canonical = run / 'survival_models' / filename
    if canonical.is_file():
        return canonical
    layout = load_layout()
    return run / layout['clinical_directory'] / layout['clinical_files'].get(filename, filename)

def curated_table(root, filename):
    candidates = (Path(root) / 'xgboost_training' / filename,
                  Path(root) / 'tables' / filename,
                  CODE_ROOT / 'tables' / filename)
    return next((path for path in candidates if path.is_file()), candidates[-1])
