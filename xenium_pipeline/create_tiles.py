"""Dataset-aware command wrapper for tile creation."""
from __future__ import annotations
import argparse
import importlib
import sys
from pathlib import Path
import pandas as pd
from .dataset_config import PROJECT_ROOT, get_dataset_config

def _import_dataset_module(script_path: Path):
    return importlib.import_module('xenium_pipeline.' + script_path.stem)

def create_tiles_for_dataset(dataset: str, *, sample_names: list[str] | None=None, input_manifest: str | Path | None=None, overwrite: bool=False, reprocess_completed: bool=False, validate_only: bool=False, tile_qc_only: bool=False, recreate_qc_plots: bool=False, skip_qc_plots: bool=False, fail_fast: bool=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = get_dataset_config(dataset)
    root = config.root_path(PROJECT_ROOT)
    if config.pipeline == 'standard_xenium':
        from . import standard_xenium
        return standard_xenium.run_all_samples(root=root, sample_names=sample_names, input_manifest=input_manifest, overwrite=overwrite, skip_processed=not reprocess_completed, validate_only=validate_only, tile_qc_only=tile_qc_only, recreate_qc_plots=recreate_qc_plots, make_qc_plots=not skip_qc_plots, fail_fast=fail_fast, **config.standard_run_kwargs())
    if dataset == 'Bilous_2026':
        module = _import_dataset_module(config.script_path())
        manifest = input_manifest
        if manifest is None:
            default_manifest = config.output_path(PROJECT_ROOT) / 'tile_input_manifest.csv'
            manifest = default_manifest if default_manifest.is_file() else None
        return module.run_all_samples(root, sample_names=sample_names, input_manifest=manifest, output_dir=config.output_path(PROJECT_ROOT), transform_dir=config.transform_path(PROJECT_ROOT), tile_size=config.tile_size_px, min_cells_per_tile=config.min_cells_per_tile, preview_count=config.preview_count, annotation_alpha=config.annotation_alpha, overwrite=overwrite, skip_processed=not reprocess_completed, make_qc_plots=not skip_qc_plots, recreate_qc_plots=recreate_qc_plots, validate_only=validate_only, fail_fast=fail_fast)
    raise ValueError(f'Unsupported tile dataset: {dataset}')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create H&E tiles for one registered dataset.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--samples', nargs='+')
    parser.add_argument('--input-manifest', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--reprocess-completed', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--tile-qc-only', action='store_true')
    parser.add_argument('--recreate-qc-plots', action='store_true')
    parser.add_argument('--skip-qc-plots', action='store_true')
    parser.add_argument('--fail-fast', action='store_true')
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    summary, tiles = create_tiles_for_dataset(args.dataset, sample_names=args.samples, input_manifest=args.input_manifest, overwrite=args.overwrite, reprocess_completed=args.reprocess_completed, validate_only=args.validate_only, tile_qc_only=args.tile_qc_only, recreate_qc_plots=args.recreate_qc_plots, skip_qc_plots=args.skip_qc_plots, fail_fast=args.fail_fast)
    print(f'summary rows: {len(summary):,}')
    if not summary.empty:
        print(summary.head())
    print(f'tile rows: {len(tiles):,}')
    failed = False
    if not summary.empty and 'status' in summary.columns:
        failed = summary['status'].astype(str).isin({'failed', 'invalid'}).any()
    return int(failed)
if __name__ == '__main__':
    raise SystemExit(main())
