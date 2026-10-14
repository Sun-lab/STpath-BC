from __future__ import annotations

from pathlib import Path
import os

CODE_ROOT = Path(__file__).resolve().parents[1]
from xenium_pipeline.storage import data_path

def compare_resolution(root):
    from pathlib import Path
    from datetime import datetime, timezone
    import json
    import sys
    PROJECT_ROOT = Path(root).expanduser().resolve()
    INPUT_DIR = Path(str(os.environ.get('STPATH_TCGA_TIFF_ROOT', root / 'data/TCGA_BRCA_HE_filtered_tiff')))
    MANIFEST = data_path(PROJECT_ROOT, 'resolution_manifest')
    FIELD_OF_VIEW_UM = 70.0
    import pandas as pd
    from xenium_pipeline.workflow_output import display, Image as DisplayImage
    from xenium_pipeline.tcga_brca_resolution_comparison import run_comparison, image_layout
    manifest = json.loads(MANIFEST.read_text())
    selection = []
    for item in manifest['slides']:
        matched = image_layout(item['matched'], item['slide_id'])
        highest = image_layout(item['highest'], item['slide_id'])
        selection.append({'slide_id': item['slide_id'], 'inferred_tiff_um_per_pixel': matched['mpp_x'], 'highest_source_um_per_pixel': highest['mpp_x'], 'approx_tiff_crop_pixels': round(FIELD_OF_VIEW_UM / matched['mpp_x']), 'approx_source_crop_pixels': round(FIELD_OF_VIEW_UM / highest['mpp_x'])})
    display(pd.DataFrame(selection))
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    OUTPUT_DIR = PROJECT_ROOT / 'tcga_brca_resolution_comparison' / run_id
    summary = run_comparison(INPUT_DIR, MANIFEST, OUTPUT_DIR, field_um=FIELD_OF_VIEW_UM)
    print(json.dumps(summary, indent=2))
    display(pd.DataFrame(summary['slides']))
    comparison_path = OUTPUT_DIR / 'crop_comparison.csv'
    if comparison_path.stat().st_size:
        comparisons = pd.read_csv(comparison_path)
        display(comparisons[['slide_id', 'anchor', 'tiff_mpp_x', 'highest_mpp_x', 'tiff_field_width_um', 'highest_field_width_um', 'tiff_vs_public_mae', 'tiff_vs_public_max_difference', 'tiff_vs_public_exact_fraction', 'crop_tissue_fraction']])
    else:
        print('No complete crop comparisons; inspect the slide errors above.')
    for result in summary['slides']:
        print(result['slide_id'], result['status'])
        folder = OUTPUT_DIR / result['slide_id']
        if (folder / 'overview.png').exists():
            display(DisplayImage(filename=str(folder / 'overview.png'), width=700))
        if (folder / 'comparison.png').exists():
            display(DisplayImage(filename=str(folder / 'comparison.png'), width=960))
        if result['error']:
            print(result['error'])
    print(f'Results: {OUTPUT_DIR}')
    print(f"Status: {summary['status']}; locations: {summary['n_locations']}/9")
    for filename in ('run_summary.json', 'crop_comparison.csv', 'slide_results.json'):
        print(OUTPUT_DIR / filename)
    return OUTPUT_DIR


def audit_headers(root, argv=()):
    import argparse
    from xenium_pipeline.tcga_brca_header_audit import run_header_audit
    root = Path(root).expanduser().resolve()
    parser = argparse.ArgumentParser(description='Audit TCGA-BRCA TIFF headers and source spacing.')
    parser.add_argument('--input-dir', type=Path, default=Path(os.environ.get('STPATH_TCGA_TIFF_ROOT', root / 'data/TCGA_BRCA_HE_filtered_tiff')))
    parser.add_argument('--candidates', type=Path, default=root / 'tcga_brca_metadata/cohort_resolution_candidates.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--skip-spacing', action='store_true')
    parser.add_argument('--metadata-cache', type=Path)
    args = parser.parse_args(argv)
    result = run_header_audit(args.input_dir, args.candidates, args.output_dir,
        fetch_spacing=not args.skip_spacing, metadata_cache=args.metadata_cache)
    if result['status'] != 'complete':
        raise RuntimeError(f'Incomplete header audit: {args.output_dir}')
    return result
