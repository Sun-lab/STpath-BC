"""Training-compatible prediction on selected current TCGA BRCA TIFFs."""
from __future__ import annotations
from .storage import model_runs
import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from . import stpath_xgboost as sx
from .hf_auth import ensure_hf_token
from .he_tissue import MASK_METHOD_VERSION, color_tissue_mask_rgb
from .tcga_brca_images import apply_audited_spacing, image_array, inspect_slide, rgb_region, select_slides, read_tissue_thumbnail, eligible_physical_grid
RECIPE = 'stpath_brca_training_compatible_v1'
DEFAULT_SLIDES = ('TCGA-4H-AAAK-01Z-00-DX1', 'TCGA-A8-A081-01Z-00-DX1', 'TCGA-AO-A03L-01Z-00-DX1')
DEFAULT_MODEL_RUNS = model_runs()

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def check_model_exports(model_dirs, *, nthread=None):
    """Require each explicitly selected final target; never fall back to smoke/folds."""
    if not model_dirs:
        raise ValueError('Select at least one final model target')
    records = []
    probe = {name: np.zeros((1, width), np.float32) for name, width in sx.EXPECTED_DIMS.items()}
    for target, folder in model_dirs.items():
        folder = Path(folder).resolve()
        if 'checkpoints' in folder.parts or 'smoke' in folder.parts:
            raise ValueError(f'Use the exported final target directory: {folder}')
        paths = [folder / 'complete.json', folder / 'model_metadata/Combined.json', folder / 'models/Combined.ubj']
        missing = [str(p) for p in paths if not p.is_file() or not p.stat().st_size]
        if missing:
            raise FileNotFoundError(f'Final export incomplete for {target}: {missing}. Finalize the trained model before prediction.')
        complete, metadata = (json.loads(p.read_text()) for p in paths[:2])
        if complete.get('status') != 'complete' or complete.get('cell_type') != target or metadata.get('cell_type') != target:
            raise ValueError(f'Target/completion mismatch for {target}: {folder}')
        for completion_key, metadata_key in (('training_config_fingerprint', 'training_config_fingerprint'), ('tile_id_checksum', 'training_tile_id_checksum'), ('n_tiles', 'n_training_tiles')):
            if complete.get(completion_key) is None or complete[completion_key] != metadata.get(metadata_key):
                raise ValueError(f'Final export and completion disagree on {completion_key}: {folder}')
        sx.predict_saved_models(folder, probe, extraction_recipe=RECIPE, calibrate=False, model_names=('Combined',), nthread=nthread)
        records.append({'cell_type': target, 'target_dir': str(folder), 'n_training_tiles': metadata['n_training_tiles'], 'n_training_individuals': len(metadata.get('individual_ids', [])), 'input_dimension': metadata['input_dimension'], 'file_sha256': {str(p.relative_to(folder)): sha256(p) for p in paths}})
    return records

def systematic_sample(eligible, maximum, seed):
    """Equal-probability systematic sample of a row-major eligible grid."""
    eligible = np.asarray(eligible, dtype=np.int64)
    if maximum < 1:
        raise ValueError('maximum must be positive')
    n = min(len(eligible), maximum)
    if not n:
        return eligible
    positions = np.floor((np.arange(n) + np.random.default_rng(seed).random()) * len(eligible) / n).astype(int)
    return eligible[positions]

def prepare_tiles(input_dir, audit_csv, slide_ids, output_dir, *, sizes_um=(70, 95), max_tiles=256, min_tissue=0.6, seed=42):
    """Save tissue-positive native crops; max_tiles=None selects all for one slide."""
    sizes_um = tuple((float(size) for size in sizes_um))
    if not 1 <= len(sizes_um) <= 2 or len(set(sizes_um)) != len(sizes_um) or any((not math.isfinite(s) or s <= 0 for s in sizes_um)):
        raise ValueError('Choose one or two distinct positive physical crop sizes')
    if max_tiles is not None and (not 1 <= max_tiles <= 1024):
        raise ValueError('Choose 1–1024 sampled tiles, or max_tiles=None for all eligible tiles')
    if not math.isfinite(min_tissue) or not 0 < min_tissue <= 1:
        raise ValueError('Tissue cutoff must be in (0,1]')
    input_dir, output_dir = (Path(input_dir).resolve(), Path(output_dir).resolve())
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError('Keep pilot outputs outside the source TIFF directory')
    if max_tiles is None and len(slide_ids) != 1:
        raise ValueError('All-tile inference requires exactly one slide')
    selected = select_slides(input_dir, slide_ids, max_slides=3)
    with Path(audit_csv).open(newline='') as stream:
        audit = list(csv.DictReader(stream))
    by_id = {row['slide_id']: row for row in audit}
    if len(by_id) != len(audit):
        raise ValueError('Audit contains duplicate slide IDs')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Use a new empty pilot folder, or load its existing manifest: {output_dir}')
    metadata_rows = []
    for path in selected:
        metadata = inspect_slide(path)
        apply_audited_spacing(metadata, by_id[path.stem])
        if metadata['orientation'] != 1:
            raise ValueError(f'Non-default TIFF orientation: {path}')
        metadata_rows.append(metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, summaries = ([], [])
    for path, metadata in zip(selected, metadata_rows):
        start = time.monotonic()
        width, height = (metadata['width_px'], metadata['height_px'])
        thumb = read_tissue_thumbnail(path, metadata)
        mask, _, _ = color_tissue_mask_rgb(thumb)
        with image_array(path, metadata['series_index'], 0) as (array, axes, _reader):
            for size in sizes_um:
                candidate = f'physical_um_{size:g}'
                xs, ys, fractions, _full, eligible = eligible_physical_grid(metadata, mask, size, min_tissue)
                sample_seed = int.from_bytes(hashlib.sha256(f'{seed}/{path.stem}/{candidate}'.encode()).digest()[:8], 'big')
                chosen = eligible if max_tiles is None else systematic_sample(eligible, max_tiles, sample_seed)
                if not len(chosen):
                    raise ValueError(f'No full tiles pass tissue cutoff: {path.stem} {candidate}')
                folder = output_dir / 'tiles' / path.stem / candidate
                folder.mkdir(parents=True)
                overview = Image.fromarray(thumb)
                draw = ImageDraw.Draw(overview)
                for crop_number, index in enumerate(chosen, 1):
                    row, col = np.unravel_index(index, fractions.shape)
                    x0, x1, y0, y1 = (int(xs[col]), int(xs[col + 1]), int(ys[row]), int(ys[row + 1]))
                    tile_id = f'{path.stem}__{candidate}__r{row + 1}_c{col + 1}'
                    tile_path = folder / f'r{row + 1}_c{col + 1}.png'
                    Image.fromarray(rgb_region(array, axes, (x0, y0, x1, y1))).save(tile_path)
                    rows.append({'tile_id': tile_id, 'slide_id': path.stem, 'patient_id': metadata['patient_id'], 'candidate': candidate, 'nominal_size_um': size, 'tile_row': int(row + 1), 'tile_col': int(col + 1), 'x0_px': x0, 'y0_px': y0, 'x1_px': x1, 'y1_px': y1, 'tile_width_um': (x1 - x0) * metadata['tiling_mpp_x'], 'tile_height_um': (y1 - y0) * metadata['tiling_mpp_y'], 'tissue_fraction_thumbnail_estimate': float(fractions[row, col]), 'materialized_tile_path': str(tile_path), 'tile_sha256': sha256(tile_path)})
                    cx, cy = ((x0 + x1) / 2 * thumb.shape[1] / width, (y0 + y1) / 2 * thumb.shape[0] / height)
                    draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill='yellow', outline='black')
                    if max_tiles is None and (crop_number % 256 == 0 or crop_number == len(chosen)):
                        print(f'Cropping {path.stem} {candidate}: {crop_number:,}/{len(chosen):,}', flush=True)
                overview.save(output_dir / f'{path.stem}__{candidate}__sampling.png')
                summaries.append({'slide_id': path.stem, 'candidate': candidate, 'n_eligible_full_tiles_estimate': len(eligible), 'n_sampled_tiles': len(chosen), 'tile_width_px': int(xs[1]), 'tile_height_px': int(ys[1])})
                print(f'{path.stem} {candidate}: {len(chosen)} / {len(eligible):,} eligible full tiles', flush=True)
        metadata['tiling_seconds'] = time.monotonic() - start
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / 'sampled_tiles.csv', index=False)
    pd.DataFrame(summaries).to_csv(output_dir / 'sampling_summary.csv', index=False)
    write_json(output_dir / 'tile_preparation.json', {'status': 'complete', 'recipe': RECIPE, 'source_metadata': metadata_rows, 'audit_csv': str(Path(audit_csv).resolve()), 'audit_csv_sha256': sha256(audit_csv), 'sizes_um': list(sizes_um), 'max_tiles_per_slide_size': max_tiles, 'min_tissue_fraction': min_tissue, 'seed': seed, 'sampling': 'all eligible full tiles from each size-specific grid; no subsampling' if max_tiles is None else 'equal-probability random-start systematic sampling over eligible row-major tiles; sizes sampled independently', 'partial_edges': 'excluded', 'mask_method': MASK_METHOD_VERSION, 'manifest_sha256': sha256(output_dir / 'sampled_tiles.csv')})
    return table

def load_prepared_tiles(output_dir):
    """Resume only unchanged sampled crops from a completed preparation."""
    output_dir = Path(output_dir)
    required = [output_dir / 'tile_preparation.json', output_dir / 'sampled_tiles.csv']
    if any((not path.is_file() for path in required)):
        raise FileNotFoundError(f'No completed tile preparation in {output_dir}. Prepare tiles or provide a completed tile-preparation folder.')
    meta = json.loads((output_dir / 'tile_preparation.json').read_text())
    if meta['status'] != 'complete' or meta['manifest_sha256'] != sha256(output_dir / 'sampled_tiles.csv'):
        raise ValueError('Sample manifest changed; use a new pilot folder')
    table = pd.read_csv(output_dir / 'sampled_tiles.csv')
    if table.empty or table.tile_id.duplicated().any():
        raise ValueError('Empty or duplicate sampled tile IDs')
    for row in table.itertuples():
        if sha256(row.materialized_tile_path) != row.tile_sha256:
            raise ValueError(f'Sampled crop changed: {row.tile_id}')
    return table

def prepare_or_resume_tiles(input_dir, audit_csv, slide_ids, output_dir, *, sizes_um=(70, 95), max_tiles=256, min_tissue=0.6, seed=42):
    """Reuse only a completed preparation with matching sampling settings."""
    preparation = Path(output_dir) / 'tile_preparation.json'
    if not preparation.is_file():
        return prepare_tiles(input_dir, audit_csv, slide_ids, output_dir, sizes_um=sizes_um, max_tiles=max_tiles, min_tissue=min_tissue, seed=seed)
    saved = json.loads(preparation.read_text())
    if saved.get('sampling_mode') == 'matched_regions_v1':
        raise ValueError('This folder contains matched regions; resume with --matched-regions')
    requested = {'sizes_um': list(sizes_um), 'max_tiles_per_slide_size': max_tiles, 'min_tissue_fraction': min_tissue, 'seed': seed, 'audit_csv_sha256': sha256(audit_csv)}
    if any((saved.get(key) != value for key, value in requested.items())):
        raise ValueError('Sampling settings differ from this saved run; use a new output folder')
    if sorted((item['slide_id'] for item in saved['source_metadata'])) != sorted((Path(name).stem for name in slide_ids)):
        raise ValueError('Selected slides differ from this saved run; use a new output folder')
    if any((Path(item['source_path']).parent.resolve() != Path(input_dir).resolve() for item in saved['source_metadata'])):
        raise ValueError('Input TIFF directory differs from this saved run')
    for item in saved['source_metadata']:
        source_stat = Path(item['source_path']).stat()
        if (source_stat.st_size, source_stat.st_mtime_ns) != (item['file_size_bytes'], item['source_mtime_ns']):
            raise ValueError(f"Source TIFF changed since tile preparation: {item['slide_id']}; use a new output folder")
    print(f'Reusing prepared tiles: {output_dir}', flush=True)
    return load_prepared_tiles(output_dir)

def training_encoder_revisions(feature_root):
    """Read provenance without loading training embedding arrays into RAM."""
    import torch
    from .stpath_foundation import MODEL_SOURCES
    result = {}
    for name in sx.MODEL_NAMES:
        paths = sorted((Path(feature_root) / name).glob('*/*.pt'))
        if not paths:
            raise FileNotFoundError(f'Training feature shards missing: {Path(feature_root) / name}')
        revisions, missing = (set(), 0)
        for path in paths:
            payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            if payload.get('recipe_version') != RECIPE or payload.get('model_name') != name or payload.get('model_source') != MODEL_SOURCES[name]:
                raise ValueError(f'Training encoder provenance mismatch: {path}')
            revision = payload.get('resolved_hf_revision')
            if revision:
                revisions.add(revision)
            else:
                missing += 1
            del payload
        if len(revisions) > 1:
            raise ValueError(f'Training shards use multiple {name} weight revisions: {sorted(revisions)}')
        result[name] = {'recorded_revision': next(iter(revisions), None), 'n_shards': len(paths), 'n_missing_revision': missing}
    return result

def extract_features(output_dir, training_revisions, *, batch_size=32, device_name='cpu'):
    """Reuse validated arrays; preserve later encoders when one model fails to load."""
    output_dir = Path(output_dir)
    table = load_prepared_tiles(output_dir)
    import torch
    import torchvision
    import timm
    from torch.utils.data import DataLoader
    from . import stpath_foundation as foundation
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    device = torch.device(device_name)
    if device.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA is unavailable. Run in a GPU allocation with the encoder environment, or explicitly choose cpu.')
    features_dir = output_dir / 'features'
    features_dir.mkdir(exist_ok=True)
    identity = {'manifest_sha256': sha256(output_dir / 'sampled_tiles.csv'), 'recipe': RECIPE, 'recipe_code_sha256': sha256(Path(foundation.__file__)), 'versions': {'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__), 'timm': str(timm.__version__)}}
    timings, group_timings = ([], [])
    load_errors = []
    error_path = output_dir / 'encoder_load_errors.json'
    error_path.unlink(missing_ok=True)
    for name in sx.MODEL_NAMES:
        destination = features_dir / f'{name}.npz'
        meta_path = features_dir / f'{name}.json'
        expected_revision = training_revisions[name]['recorded_revision']
        if destination.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if any((meta.get(k) != v for k, v in identity.items())) or meta['training_revision'] != training_revisions[name]:
                raise ValueError(f'Feature cache identity changed: {destination}; use a new pilot folder')
            _read_features(destination, meta, table.tile_id.tolist(), sx.EXPECTED_DIMS[name])
            timings.append({**meta['timing'], 'reused': True})
            group_timings.extend(meta['group_timings'])
            print(f'Reusing validated {name} features', flush=True)
            continue
        revision = foundation.resolve_hf_revision(name)
        if expected_revision and revision != expected_revision:
            raise ValueError(f'{name}: currently resolved revision {revision} differs from training {expected_revision}. Use the training weight revision before continuing.')
        if training_revisions[name]['n_missing_revision']:
            print(f'{name}: some training shards lack weight revision metadata; complete weight identity cannot be verified.', flush=True)
        model = None
        start = time.monotonic()
        try:
            try:
                model, transform = foundation.build_model(name, device)
            except Exception as exc:
                load_errors.append({'encoder': name, 'error_type': type(exc).__name__, 'error': str(exc)})
                (output_dir / 'prediction_summary.json').unlink(missing_ok=True)
                write_json(error_path, load_errors)
                print(f'ERROR loading {name}: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
                print(f'Skipping {name} for this attempt; continuing with the next encoder.', flush=True)
                continue
            if foundation.resolve_hf_revision(name) != revision:
                raise ValueError(f'{name}: upstream revision changed during model loading')
            loaded_seconds = time.monotonic() - start
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            chunks, observed = ([], [])
            model_group_timings = []
            encode_start = time.monotonic()
            with torch.inference_mode():
                for (slide, candidate), group in table.groupby(['slide_id', 'candidate'], sort=False):
                    group_start = time.monotonic()
                    group_processed, next_report = (0, 256)
                    loader = DataLoader(foundation.ManifestTileDataset(group, transform), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == 'cuda')
                    for images, tile_ids in loader:
                        embeddings = foundation.encode_batch(name, model, images.to(device))
                        if embeddings.ndim != 2 or embeddings.shape[1] != sx.EXPECTED_DIMS[name]:
                            raise ValueError(f'Unexpected {name} feature shape: {tuple(embeddings.shape)}')
                        chunks.append(embeddings.detach().float().cpu().numpy())
                        observed.extend(tile_ids)
                        group_processed += len(tile_ids)
                        if group_processed >= next_report and group_processed < len(group):
                            print(f'{name} {slide} {candidate}: {group_processed:,}/{len(group):,} tiles ({len(observed):,}/{len(table):,} total)', flush=True)
                            next_report = (group_processed // 256 + 1) * 256
                    seconds = time.monotonic() - group_start
                    model_group_timings.append({'encoder': name, 'slide_id': slide, 'candidate': candidate, 'n_tiles': len(group), 'encoding_seconds': seconds, 'tiles_per_second': len(group) / seconds})
                    print(f'{name}: {len(observed):,}/{len(table):,} tiles', flush=True)
            matrix = np.concatenate(chunks)
            if observed != table.tile_id.tolist() or not np.isfinite(matrix).all():
                raise ValueError(f'Invalid or misordered {name} embeddings')
            encoding_seconds = time.monotonic() - encode_start
            temporary = destination.with_suffix('.tmp.npz')
            np.savez(temporary, embeddings=matrix, tile_ids=np.asarray(observed, dtype=str))
            temporary.replace(destination)
            timing = {'encoder': name, 'n_tiles': len(table), 'load_seconds': loaded_seconds, 'encoding_seconds': encoding_seconds, 'tiles_per_second': len(table) / encoding_seconds, 'peak_gpu_gib': torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == 'cuda' else None}
            write_json(meta_path, {**identity, 'model_name': name, 'resolved_hf_revision': revision, 'training_revision': training_revisions[name], 'timing': timing, 'group_timings': model_group_timings, 'device': str(device), 'batch_size': batch_size, 'array_sha256': sha256(destination)})
            timings.append({**timing, 'reused': False})
            group_timings.extend(model_group_timings)
            del loader, chunks, matrix, embeddings, images
        finally:
            if model is not None:
                del model
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    pd.DataFrame(timings).to_csv(output_dir / 'encoder_timing.csv', index=False)
    pd.DataFrame(group_timings).to_csv(output_dir / 'encoder_timing_by_slide_crop.csv', index=False)
    if load_errors:
        failed = ', '.join((error['encoder'] for error in load_errors))
        raise RuntimeError(f'Feature extraction incomplete: failed to load {failed}. Successful encoder features are saved; Combined predictions require all encoders. See {error_path}')
    return pd.DataFrame(timings)

def _read_features(path, metadata, tile_ids, width):
    if metadata['array_sha256'] != sha256(path):
        raise ValueError(f'Feature array changed: {path}')
    with np.load(path, allow_pickle=False) as payload:
        matrix = payload['embeddings']
        if payload['tile_ids'].tolist() != tile_ids or matrix.shape != (len(tile_ids), width) or matrix.dtype != np.float32 or (not np.isfinite(matrix).all()):
            raise ValueError(f'Invalid feature shape, values, dtype or tile order: {path}')
    return matrix

def predict_and_summarize(output_dir, model_dirs, *, nthread=None):
    """Preserve raw Combined predictions, including out-of-range values."""
    output_dir = Path(output_dir)
    (output_dir / 'prediction_summary.json').unlink(missing_ok=True)
    exports = check_model_exports(model_dirs, nthread=nthread)
    table = load_prepared_tiles(output_dir)
    preparation = json.loads((output_dir / 'tile_preparation.json').read_text())
    all_tiles = preparation['max_tiles_per_slide_size'] is None
    matched = preparation.get('sampling_mode') == 'matched_regions_v1'
    matrices = {}
    for name in sx.MODEL_NAMES:
        meta = json.loads((output_dir / 'features' / f'{name}.json').read_text())
        if meta['manifest_sha256'] != sha256(output_dir / 'sampled_tiles.csv') or meta['recipe'] != RECIPE or meta['model_name'] != name:
            raise ValueError(f'Feature provenance mismatch: {name}')
        matrices[name] = _read_features(output_dir / 'features' / f'{name}.npz', meta, table.tile_id.tolist(), sx.EXPECTED_DIMS[name])
    output, summaries, timings = ([], [], [])
    for target, folder in model_dirs.items():
        start = time.monotonic()
        raw = sx.predict_saved_models(folder, matrices, extraction_recipe=RECIPE, calibrate=False, model_names=('Combined',), nthread=nthread)['predicted_raw__Combined'].to_numpy()
        if not np.isfinite(raw).all():
            raise ValueError(f'Non-finite predictions: {target}')
        frame = table.drop(columns=['materialized_tile_path', 'tile_sha256']).copy()
        frame['cell_type'], frame['predicted_raw'] = (target, raw)
        frame['outside_0_1'] = (raw < 0) | (raw > 1)
        output.append(frame)
        timings.append({'cell_type': target, 'n_tiles': len(frame), 'prediction_seconds': time.monotonic() - start})
        for (slide, candidate), group in frame.groupby(['slide_id', 'candidate'], sort=False):
            values = group.predicted_raw.to_numpy()
            summaries.append({'slide_id': slide, 'patient_id': group.patient_id.iloc[0], 'candidate': candidate, 'cell_type': target, 'n_sampled_tiles': len(values), 'sample_mean': float(values.mean()), 'sample_median': float(np.median(values)), 'p05': float(np.quantile(values, 0.05)), 'p95': float(np.quantile(values, 0.95)), 'min': float(values.min()), 'max': float(values.max()), 'fraction_outside_0_1': float(group.outside_0_1.mean())})
    predictions, summary = (pd.concat(output, ignore_index=True), pd.DataFrame(summaries))
    predictions.to_csv(output_dir / 'tile_predictions_raw.csv.gz', index=False)
    summary.to_csv(output_dir / 'sampled_slide_summary.csv', index=False)
    pd.DataFrame(timings).to_csv(output_dir / 'prediction_timing.csv', index=False)
    sizes = sorted(table.nominal_size_um.unique())
    candidates = [f'physical_um_{size:g}' for size in sizes]
    comparison = summary.pivot(index=['slide_id', 'cell_type'], columns='candidate', values='sample_mean')
    if len(candidates) == 2:
        comparison['larger_minus_smaller'] = comparison[candidates[1]] - comparison[candidates[0]]
    comparison.reset_index().to_csv(output_dir / 'crop_size_comparison.csv', index=False)
    interpretation = 'All eligible full tiles from each size-specific grid; no subsampling. Means are tile averages, not cell-count-weighted whole-slide fractions. Tissue filtering and partial-edge exclusion can change coverage across sizes. No accuracy estimate or survival analysis.' if all_tiles else 'Technical and crop-size sensitivity pilot. Means describe sampled eligible tiles, not established whole-slide cell fractions. Sizes sampled independently; differences include sampling variability. No accuracy estimate or survival analysis.'
    if matched:
        from .tcga_brca_matched import INTERPRETATION, summarize_matched_predictions
        _, summary, comparison = summarize_matched_predictions(output_dir, predictions, list(model_dirs))
        interpretation = INTERPRETATION
    feature_provenance = {name: json.loads((output_dir / 'features' / f'{name}.json').read_text()) for name in sx.MODEL_NAMES}
    write_json(output_dir / 'prediction_summary.json', {'status': 'complete', 'n_slides': int(table.slide_id.nunique()), 'n_crops': len(table), 'sizes_um': [float(s) for s in sizes], 'cell_types': list(model_dirs), 'model_exports': exports, 'calibration': False, 'clipping': False, 'across_cell_type_normalization': False, 'feature_provenance': feature_provenance, 'manifest_sha256': sha256(output_dir / 'sampled_tiles.csv'), 'tile_selection': 'matched_regions' if matched else 'all_eligible' if all_tiles else 'sampled', 'interpretation': interpretation})
    return (predictions, summary, comparison if matched else comparison.reset_index())

def save_distribution_plots(output_dir):
    """Write the notebook's crop-size comparisons without a graphical display."""
    output_dir = Path(output_dir)
    preparation = json.loads((output_dir / 'tile_preparation.json').read_text())
    if preparation.get('sampling_mode') == 'matched_regions_v1':
        from .tcga_brca_matched import save_matched_plots
        save_matched_plots(output_dir)
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    table = pd.read_csv(output_dir / 'tile_predictions_raw.csv.gz')
    preparation = json.loads((output_dir / 'tile_preparation.json').read_text())
    population = 'eligible' if preparation['max_tiles_per_slide_size'] is None else 'sampled'
    slides = table.slide_id.drop_duplicates().tolist()
    figure_dir = output_dir / 'figures'
    figure_dir.mkdir(exist_ok=True)
    for target, target_rows in table.groupby('cell_type', sort=False):
        fig, axes = plt.subplots(1, len(slides), figsize=(5 * len(slides), 3.8), squeeze=False)
        for ax, slide in zip(axes[0], slides):
            for size, group in target_rows.loc[target_rows.slide_id == slide].groupby('nominal_size_um'):
                values = np.sort(group.predicted_raw.to_numpy())
                ax.step(values, np.arange(1, len(values) + 1) / len(values), where='post', label=f'{size:g} µm')
            ax.set(title=slide.replace('-01Z-00-DX1', ''), xlabel='Raw predicted proportion', ylabel=f'Fraction of {population} tiles')
            ax.grid(alpha=0.2)
            ax.legend()
        fig.suptitle(target)
        fig.tight_layout()
        fig.savefig(figure_dir / f'{sx.safe_name(target)}__prediction_distributions.png', dpi=160)
        plt.close(fig)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(os.environ.get('XENIUM_ROOT', str(Path(__file__).resolve().parents[1]))))
    parser.add_argument('--input-dir', type=Path, default=Path(os.environ.get('STPATH_TCGA_TIFF_ROOT', str(Path(os.environ.get('XENIUM_ROOT', Path(__file__).resolve().parents[1])) / 'data/TCGA_BRCA_HE_filtered_tiff'))))
    parser.add_argument('--audit-csv', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True, help='Stable run folder; repeat the same command to resume')
    parser.add_argument('--training-feature-root', type=Path)
    parser.add_argument('--model-dirs-json', type=Path, help='Optional JSON mapping cell-type names to explicit final target directories')
    parser.add_argument('--slides', nargs='+', default=list(DEFAULT_SLIDES))
    parser.add_argument('--sizes-um', nargs='+', type=float, help='One or two independent sizes; matched mode uses 50/75/100/150 µm')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--max-tiles', type=int, default=None)
    selection.add_argument('--all-tiles', action='store_true', help='Process every eligible full tile; requires one --slides ID')
    selection.add_argument('--matched-regions', action='store_true', help='Compare 50/75/100/150 µm tiles in shared 300 µm parent regions')
    parser.add_argument('--regions-per-slide', type=int, help='Parent regions per slide in matched mode (default 32)')
    parser.add_argument('--min-tissue', type=float, default=0.6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    args = parser.parse_args(argv)
    if args.max_tiles is None:
        args.max_tiles = 256
    if args.matched_regions:
        from .tcga_brca_matched import SIZES_UM
        if args.sizes_um is not None and tuple(args.sizes_um) != SIZES_UM:
            parser.error('--matched-regions requires sizes 50 75 100 150')
        args.sizes_um = SIZES_UM
        args.regions_per_slide = 32 if args.regions_per_slide is None else args.regions_per_slide
        if not 1 <= args.regions_per_slide <= 1024:
            parser.error('--regions-per-slide must be between 1 and 1024')
    else:
        if args.regions_per_slide is not None:
            parser.error('--regions-per-slide requires --matched-regions')
        args.sizes_um = (70, 95) if args.sizes_um is None else args.sizes_um
    if not 1 <= len(args.sizes_um) <= (4 if args.matched_regions else 2) or len(set(args.sizes_um)) != len(args.sizes_um) or any((not math.isfinite(size) or size <= 0 for size in args.sizes_um)):
        parser.error('--sizes-um requires one or two distinct positive finite sizes')
    if args.all_tiles:
        if len(args.slides) != 1:
            parser.error('--all-tiles requires exactly one --slides ID')
        args.max_tiles = None
    return args

def main(argv=None):
    args = parse_args(argv)
    root = args.project_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    audit = args.audit_csv or root / 'tcga_brca_header_audit/20260923T195233_983546Z/tiff_resolution_audit.csv'
    features = args.training_feature_root or root / 'foundation_features' / RECIPE
    models = json.loads(args.model_dirs_json.read_text()) if args.model_dirs_json else {target: root / 'xgboost_training/runs' / run / sx.safe_name(target) for target, run in DEFAULT_MODEL_RUNS.items()}
    allocated = int(os.environ.get('SLURM_CPUS_PER_TASK', sx.available_cpu_count() or 1))
    if not 1 <= args.threads <= allocated:
        raise ValueError(f'threads must be between 1 and the allocated CPU count ({allocated})')
    print(f'Python: {sys.executable}\nDevice: {args.device}; CPU threads: {args.threads}\nOutput: {output}', flush=True)
    ensure_hf_token()
    import torch
    import matplotlib
    from .stpath_foundation import require_hf_token
    require_hf_token()
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA requested but unavailable in this allocation; request a GPU or use --device cpu')
    if args.device == 'cuda':
        gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
        print(f'GPU: {gpu.name}; memory: {gpu.total_memory / 1024 ** 3:.1f} GiB', flush=True)
    exports = check_model_exports(models, nthread=args.threads)
    if args.matched_regions:
        from .tcga_brca_matched import prepare_or_resume_matched_tiles
        prepare_or_resume_matched_tiles(args.input_dir, audit, args.slides, output, max_regions=args.regions_per_slide, min_tissue=args.min_tissue, seed=args.seed)
    else:
        prepare_or_resume_tiles(args.input_dir, audit, args.slides, output, sizes_um=args.sizes_um, max_tiles=args.max_tiles, min_tissue=args.min_tissue, seed=args.seed)
    write_json(output / 'selected_model_exports.json', exports)
    revisions = training_encoder_revisions(features)
    write_json(output / 'training_encoder_revisions.json', revisions)
    extract_features(output, revisions, batch_size=args.batch_size, device_name=args.device)
    predict_and_summarize(output, models, nthread=args.threads)
    save_distribution_plots(output)
    print(f"Prediction pilot complete: {output / 'prediction_summary.json'}", flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
