"""Validate H&E inputs, write accepted tile manifests, and run alignment QC."""
from __future__ import annotations
import argparse
import importlib
import sys
from pathlib import Path
import pandas as pd
from .dataset_config import PROJECT_ROOT, get_dataset_config

def _import_dataset_module(script_path: Path):
    return importlib.import_module('xenium_pipeline.' + script_path.stem)

def prepare_standard_dataset(dataset: str, *, sample_names: list[str] | None=None, run_alignment_qc: bool=True, write_manifest: bool=True) -> dict[str, pd.DataFrame]:
    from . import standard_xenium
    config = get_dataset_config(dataset)
    root = config.root_path(PROJECT_ROOT)
    validation, _tiles = standard_xenium.run_all_samples(root=root, sample_names=sample_names, validate_only=True, selective_archive_extract=config.selective_archive_extract, require_alignment=config.require_alignment, require_cell_groups=config.require_cell_groups, missing_cell_label=config.missing_cell_label, cell_groups_label_source=config.cell_groups_label_source, cells_output_suffix=config.cells_output_suffix, completion_marker_suffix=config.completion_marker_suffix, skip_completed_before_validation=config.skip_completed_before_validation)
    outputs: dict[str, pd.DataFrame] = {'validation': validation}
    if run_alignment_qc:
        alignment_qc = standard_xenium.run_alignment_qc(root, sample_names=sample_names, selective_archive_extract=config.selective_archive_extract, require_alignment=config.require_alignment, require_cell_groups=config.require_cell_groups)
        outputs['alignment_qc'] = alignment_qc
    if write_manifest:
        manifest = standard_xenium.write_tile_input_manifest(root, sample_names)
        outputs['tile_input_manifest'] = manifest
    return outputs

def prepare_bilous_dataset(*, sample_names: list[str] | None=None, write_manifest: bool=True, transform_dir: str | Path | None=None) -> dict[str, pd.DataFrame]:
    config = get_dataset_config('Bilous_2026')
    root = config.root_path(PROJECT_ROOT)
    module = _import_dataset_module(config.script_path())
    mapping_validation = module.validate_mapping_inputs(root, sample_names=sample_names, transform_dir=transform_dir)
    outputs: dict[str, pd.DataFrame] = {'mapping_validation': mapping_validation}
    if write_manifest:
        manifest = module.write_tile_input_manifest(root, sample_names=sample_names, output_dir=config.output_path(PROJECT_ROOT), transform_dir=transform_dir)
        outputs['tile_input_manifest'] = manifest
    return outputs

def prepare_dataset(dataset: str, *, sample_names: list[str] | None=None, run_alignment_qc: bool=True, write_manifest: bool=True, transform_dir: str | Path | None=None) -> dict[str, pd.DataFrame]:
    config = get_dataset_config(dataset)
    if config.pipeline == 'standard_xenium':
        return prepare_standard_dataset(dataset, sample_names=sample_names, run_alignment_qc=run_alignment_qc, write_manifest=write_manifest)
    if dataset == 'Bilous_2026':
        return prepare_bilous_dataset(sample_names=sample_names, write_manifest=write_manifest, transform_dir=transform_dir)
    raise ValueError(f'Unsupported prepare_he dataset: {dataset}')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run scriptable H&E preparation handoff checks.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--samples', nargs='+')
    parser.add_argument('--skip-alignment-qc', action='store_true')
    parser.add_argument('--skip-manifest', action='store_true')
    parser.add_argument('--transform-dir', type=Path)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    outputs = prepare_dataset(args.dataset, sample_names=args.samples, run_alignment_qc=not args.skip_alignment_qc, write_manifest=not args.skip_manifest, transform_dir=args.transform_dir)
    for name, table in outputs.items():
        print(f'{name}: {len(table):,} rows')
        if not table.empty:
            print(table.head())
    invalid = outputs.get('validation', outputs.get('mapping_validation', pd.DataFrame()))
    if not invalid.empty and 'status' in invalid.columns:
        return int(invalid['status'].astype(str).isin({'invalid', 'failed'}).any())
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
