"""Manual H&E review of saved unresolved epithelial clusters; no label inference."""

from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from . import epithelial_morphology as morphology


KEYS = ['dataset', 'sample', 'cluster_key', 'cluster']
LABELS = ['Uncertain', 'Normal epithelial', 'Tumor']
UNRESOLVED = 'Epithelial, normal/tumor unresolved'


def load_review_targets(path, cluster_key):
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = set(KEYS + ['proposed_broad_label'])
    if not required.issubset(table):
        raise ValueError(f'Annotation table needs columns: {sorted(required)}')
    table = table.loc[table.proposed_broad_label.eq(UNRESOLVED) & table.cluster_key.eq(cluster_key)].copy()
    if table.duplicated(KEYS).any():
        raise ValueError('Duplicate cluster identities in the annotation table.')
    if table.empty:
        raise ValueError(f'No unresolved epithelial clusters at {cluster_key} in {path}')
    return table


def select_examples(clusters, aligned, cluster_key, cluster, image_shape, *, count=20, seed=0):
    """Sample unique cells uniformly from eligible saved image coordinates."""
    if count < 1:
        raise ValueError('Example count must be positive.')
    morphology.validate_ids(clusters)
    required = {'he_x_px', 'he_y_px'}
    if not required.issubset(aligned):
        raise ValueError('The saved aligned-cell table needs he_x_px and he_y_px from the preparation pipeline.')
    selected = clusters.loc[clusters[cluster_key].astype(str).eq(str(cluster)), ['cell_id']]
    if selected.empty:
        raise ValueError(f'Cluster {cluster} has no cells in the saved {cluster_key} assignments.')
    matched, method = morphology.match_rows(aligned, selected.cell_id)
    x = pd.to_numeric(matched.he_x_px, errors='coerce')
    y = pd.to_numeric(matched.he_y_px, errors='coerce')
    matched['center_source'] = 'cell_centroid'
    if {'nucleus_he_x_px', 'nucleus_he_y_px'}.issubset(matched):
        nx = pd.to_numeric(matched.nucleus_he_x_px, errors='coerce')
        ny = pd.to_numeric(matched.nucleus_he_y_px, errors='coerce')
        use = np.isfinite(nx) & np.isfinite(ny)
        x, y = x.where(~use, nx), y.where(~use, ny)
        matched.loc[use, 'center_source'] = 'nucleus_centroid'
    height, width = image_shape
    valid = np.isfinite(x) & np.isfinite(y) & x.ge(0) & x.lt(width) & y.ge(0) & y.lt(height)
    matched['center_x_px'], matched['center_y_px'] = x, y
    eligible = matched.loc[valid].sort_values('cell_id').reset_index(drop=True)
    if eligible.empty:
        raise ValueError('No matched cells have finite coordinates inside the recorded H&E image.')
    examples = eligible.sample(min(count, len(eligible)), random_state=seed).reset_index(drop=True)
    return examples, dict(n_cluster_cells=len(selected), n_coordinate_matched=len(matched),
                          n_image_eligible=len(eligible), n_examples=len(examples),
                          cell_id_match_method=method)


def load_example_boundaries(row, aligned, target_ids, tolerance_px=1.0):
    """Map exported Xenium polygons onto the same H&E as the saved coordinates."""
    from .tile_utils import apply_affine
    from shapely.geometry import Polygon

    ids = [str(value) for value in target_ids]
    outlines = {kind: {cell_id: [] for cell_id in ids} for kind in ['cell', 'nucleus']}
    status = {kind: {cell_id: 'missing_boundary_file' for cell_id in ids} for kind in outlines}
    detail = dict(boundary_transform_source='', boundary_transform_error_px=None, boundary_transform_issue='')
    try:
        if not row.get('cells') or not Path(row['cells']).is_file():
            raise FileNotFoundError('Source cell coordinates are needed to verify the boundary transform.')
        source = morphology.read_table(row['cells']).rename(columns={
            'cell_centroid_x': 'x_centroid', 'cell_centroid_y': 'y_centroid'})
        inferred, residual = morphology.infer_he_affine(source, aligned, tolerance=tolerance_px)
        affine = inferred
        detail.update(boundary_transform_source='verified_saved_coordinate_affine',
                      boundary_transform_error_px=residual)
        # Explorer matrices map H&E pixels to Xenium pixels. Polygons are in
        # Xenium microns, so invert the matrix and apply the pixel-size conversion once.
        recorded, recorded_source = None, ''
        for key in ['alignment_csv', 'manual_alignment_json']:
            if not row.get(key) or not Path(row[key]).is_file():
                continue
            if key == 'alignment_csv':
                recorded = np.linalg.inv(np.loadtxt(row[key], delimiter=','))
            else:
                recorded = np.asarray(json.loads(Path(row[key]).read_text())[
                    'matrix_xenium_full_pixel_to_he_full_pixel'], dtype=float)
            recorded_source = key
            break
        if recorded is not None:
            pixel_size = pd.to_numeric(row.get('pixel_size_um'), errors='coerce')
            if not np.isfinite(pixel_size) or pixel_size <= 0:
                raise ValueError('Recorded transform needs a verified positive pixel_size_um from the image metadata or alignment inputs.')
            if recorded.shape != (3, 3) or not np.isfinite(recorded).all() or not np.allclose(recorded[2], [0, 0, 1]):
                raise ValueError('Recorded transform is not a finite 3x3 affine matrix.')
            affine = recorded @ np.diag([1 / pixel_size, 1 / pixel_size, 1])
            paired, _ = morphology.match_rows(aligned, source.cell_id)
            check = source[['cell_id', 'x_centroid', 'y_centroid']].merge(
                paired[['cell_id', 'he_x_px', 'he_y_px']], on='cell_id', validate='one_to_one')
            values = check.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').to_numpy(float)
            values = values[np.isfinite(values).all(axis=1)]
            error = float(np.linalg.norm(apply_affine(values[:, :2], affine) - values[:, 2:], axis=1).max())
            if error > tolerance_px:
                raise ValueError(f'Recorded transform disagrees with saved H&E coordinates by {error:.3f}px.')
            detail.update(boundary_transform_source=recorded_source, boundary_transform_error_px=error)
        source['_boundary_cell_id'] = source.cell_id
        matched, _ = morphology.match_rows(source, ids)
        id_map = dict(zip(matched._boundary_cell_id.astype(str), matched.cell_id.astype(str)))
    except (ValueError, KeyError, OSError, TypeError, np.linalg.LinAlgError) as error:
        detail['boundary_transform_issue'] = str(error)
        for kind in status:
            status[kind] = dict.fromkeys(ids, 'boundary_transform_unavailable')
        return outlines, status, detail

    for kind, key in [('cell', 'cell_boundaries'), ('nucleus', 'boundaries')]:
        if not row.get(key) or not Path(row[key]).is_file():
            continue
        try:
            boundaries = morphology.read_table(row[key])
            if not {'cell_id', 'vertex_x', 'vertex_y'}.issubset(boundaries):
                raise ValueError('Boundary table needs cell_id, vertex_x, vertex_y.')
            boundaries = boundaries.loc[boundaries.cell_id.isin(id_map)].copy()
            status[kind] = dict.fromkeys(ids, 'no_polygon_for_cell')
            for raw_id, group in boundaries.groupby('cell_id', sort=False):
                cell_id = id_map[str(raw_id)]
                rings = group.groupby('label_id', sort=False, dropna=False) if 'label_id' in group else [(None, group)]
                cell_outlines = []
                for _, ring in rings:
                    if 'vertex_index' in ring:
                        ring = ring.sort_values('vertex_index', kind='stable')
                    xy = ring[['vertex_x', 'vertex_y']].apply(pd.to_numeric, errors='coerce').to_numpy(float)
                    if len(xy) < 3 or not np.isfinite(xy).all() or not Polygon(xy).is_valid or Polygon(xy).area <= 0:
                        cell_outlines = []
                        status[kind][cell_id] = 'invalid_polygon'
                        break
                    cell_outlines.append(apply_affine(xy, affine))
                if cell_outlines:
                    outlines[kind][cell_id] = cell_outlines
                    status[kind][cell_id] = 'ok'
        except (ValueError, KeyError, OSError, TypeError) as error:
            status[kind] = dict.fromkeys(ids, 'unreadable_boundary_file')
            detail[kind + '_boundary_issue'] = str(error)
    return outlines, status, detail


def _gallery_request(row, cluster_key, cluster, output_root, count, crop_size_px, seed):
    """Identify a gallery from settings and source metadata without loading source data."""
    if crop_size_px < 32 or count < 1:
        raise ValueError('crop_size_px must be at least 32 and count must be positive.')
    for key in ['he_image', 'aligned_cells', 'clusters']:
        if not row.get(key) or not Path(row[key]).is_file():
            raise FileNotFoundError(f'{key} is missing for {row["dataset"]} / {row["sample"]}: {row.get(key, "")}')
    identity = dict(dataset=str(row['dataset']), sample=str(row['sample']),
                    cluster_key=str(cluster_key), cluster=str(cluster))
    slug = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    dest = Path(output_root) / 'galleries' / slug
    inputs = {}
    for key in ['he_image', 'aligned_cells', 'clusters', 'tile_input_manifest', 'cells',
                'cell_boundaries', 'boundaries', 'alignment_csv', 'manual_alignment_json']:
        path = Path(row[key]) if row.get(key) else None
        if path is not None and path.is_file():
            stat = path.stat()
            inputs[key] = [str(path.resolve()), stat.st_size, stat.st_mtime_ns]
    signature = hashlib.sha256(json.dumps(dict(version=2, identity=identity, inputs=inputs,
        pixel_size_um=str(row.get('pixel_size_um', '')),
        count=count, crop_size_px=crop_size_px, seed=seed), sort_keys=True).encode()).hexdigest()
    # Keep evidence linked by earlier manual decisions when the overlay inputs change.
    dest = dest / signature[:12]
    return identity, inputs, signature, dest


def _read_saved_gallery(dest, signature):
    meta_path, image_path, table_path = dest / 'gallery.json', dest / 'cell_examples.png', dest / 'cell_examples.csv'
    if all(p.is_file() and p.stat().st_size > 0 for p in [meta_path, image_path, table_path]):
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            return None
        if meta.get('signature') == signature:
            meta['image_path'] = str(image_path.resolve())
            meta['examples_path'] = str(table_path.resolve())
            return meta
    return None


def load_saved_cell_gallery(row, cluster_key, cluster, output_root, *, count=20, crop_size_px=256, seed=0):
    """Read saved metadata only; never open H&E, cell tables, or boundary tables."""
    _, _, signature, dest = _gallery_request(row, cluster_key, cluster, output_root, count, crop_size_px, seed)
    meta = _read_saved_gallery(dest, signature)
    if meta is None:
        raise FileNotFoundError(
            f'No complete saved gallery matches count={count}, crop_size_px={crop_size_px}, seed={seed}, '
            'and the current source files. Generate or resume the epithelial galleries.')
    return meta


def create_cell_gallery(row, cluster_key, cluster, output_root, *, count=20, crop_size_px=256, seed=0):
    """Render a missing gallery, or reuse its complete matching saved version."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    import tifffile
    import zarr
    from .tile_utils import extract_rgb_tile, get_hw, resolve_zarr_array

    identity, inputs, signature, dest = _gallery_request(row, cluster_key, cluster, output_root, count, crop_size_px, seed)
    saved = _read_saved_gallery(dest, signature)
    if saved is not None:
        return saved
    dest.mkdir(parents=True, exist_ok=True)
    meta_path, image_path, table_path = dest / 'gallery.json', dest / 'cell_examples.png', dest / 'cell_examples.csv'
    meta_path.unlink(missing_ok=True)
    clusters = pd.read_csv(row['clusters'], usecols=['cell_id', cluster_key], dtype=str, keep_default_na=False)
    columns = pd.read_csv(row['aligned_cells'], nrows=0).columns
    keep = [c for c in ['cell_id', 'he_x_px', 'he_y_px', 'nucleus_he_x_px', 'nucleus_he_y_px'] if c in columns]
    aligned = pd.read_csv(row['aligned_cells'], usecols=keep, dtype={'cell_id': str}, keep_default_na=False)
    with ExitStack() as stack:
        tif = stack.enter_context(tifffile.TiffFile(row['he_image']))
        store = tif.series[0].levels[0].aszarr()
        stack.callback(store.close)
        image = resolve_zarr_array(zarr.open(store, mode='r'))
        height, width = get_hw(image.shape)
        examples, counts = select_examples(clusters, aligned, cluster_key, cluster, (height, width), count=count, seed=seed)
        outlines, boundary_status, transform_detail = load_example_boundaries(row, aligned, examples.cell_id)
        fig, axes = plt.subplots(int(np.ceil(len(examples) / 4)), 4,
                                 figsize=(16, 4 * int(np.ceil(len(examples) / 4))), squeeze=False)
        try:
            for ax in axes.flat:
                ax.set_axis_off()
            records = []
            for index, (ax, cell) in enumerate(zip(axes.flat, examples.to_dict('records')), 1):
                x, y = cell['center_x_px'], cell['center_y_px']
                x0 = max(0, min(int(x) - crop_size_px // 2, width - crop_size_px))
                y0 = max(0, min(int(y) - crop_size_px // 2, height - crop_size_px))
                x1, y1 = min(width, x0 + crop_size_px), min(height, y0 + crop_size_px)
                crop = extract_rgb_tile(image, x0, y0, x1, y1)
                ax.imshow(crop)
                cell_id = str(cell['cell_id'])
                missing = []
                for kind, color in [('cell', '#ffff00'), ('nucleus', '#00ffff')]:
                    for points in outlines[kind][cell_id]:
                        ax.add_patch(Polygon(points - [x0, y0], closed=True, fill=False,
                                             edgecolor=color, linewidth=.8))
                    if boundary_status[kind][cell_id] != 'ok':
                        missing.append(f'{kind}: unavailable')
                ax.set(xlim=(-.5, x1 - x0 - .5), ylim=(y1 - y0 - .5, -.5))
                ax.set_title(f'{index}. {cell_id}\n' + ('; '.join(missing) if missing else 'Cell + nucleus outlines'), fontsize=9)
                records.append(dict(**identity, example_index=index, cell_id=cell['cell_id'],
                    center_source=cell['center_source'], center_x_px=x, center_y_px=y,
                    cell_boundary_status=boundary_status['cell'][cell_id],
                    nucleus_boundary_status=boundary_status['nucleus'][cell_id],
                    crop_x0_px=x0, crop_y0_px=y0, crop_x1_px=x1, crop_y1_px=y1,
                    he_image=str(row['he_image']), aligned_cells=str(row['aligned_cells']), seed=seed))
            fig.suptitle(f'{row["dataset"]} / {row["sample"]}\n{cluster_key}, cluster {cluster}: '
                         f'{len(examples)} sampled cells; yellow: cell boundary; cyan: nuclear boundary', fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, .95))
            fig.savefig(image_path, dpi=140, bbox_inches='tight')
        finally:
            plt.close(fig)
    pd.DataFrame(records).to_csv(table_path, index=False)
    meta = dict(**identity, **counts, **transform_detail, signature=signature, inputs=inputs,
                n_cell_outlines=sum(v == 'ok' for v in boundary_status['cell'].values()),
                n_nuclear_outlines=sum(v == 'ok' for v in boundary_status['nucleus'].values()),
                image_path=str(image_path.resolve()), examples_path=str(table_path.resolve()),
                he_image_source=str(row.get('he_image_source', 'unknown')),
                crop_size_px=crop_size_px, seed=seed)
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def export_cell_galleries(inventory, targets, output_root, *, count=28, crop_size_px=384, seed=0):
    """Generate every requested cluster gallery with restartable per-cluster checkpoints."""
    if inventory.duplicated(['dataset', 'sample']).any() or targets.duplicated(KEYS).any():
        raise ValueError('Duplicate inventory samples or target cluster identities.')
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / 'cell_gallery_index.csv'
    lookup = {(str(r['dataset']), str(r['sample'])): r for r in inventory.to_dict('records')}
    records = []
    for index, target in enumerate(targets.to_dict('records'), 1):
        identity = {key: str(target[key]) for key in KEYS}
        print(f'[{index}/{len(targets)}] {identity["dataset"]} / {identity["sample"]} / '
              f'{identity["cluster_key"]} / cluster {identity["cluster"]}', flush=True)
        started = time.monotonic()
        record = dict(**identity, requested_examples=count, crop_size_px=crop_size_px, seed=seed)
        try:
            row = lookup.get((identity['dataset'], identity['sample']))
            if row is None:
                raise FileNotFoundError('Sample absent from the input inventory. Refresh the input inventory.')
            _, _, signature, dest = _gallery_request(row, identity['cluster_key'], identity['cluster'],
                                                    output_root, count, crop_size_px, seed)
            cached = _read_saved_gallery(dest, signature)
            meta = cached if cached is not None else create_cell_gallery(
                row, identity['cluster_key'], identity['cluster'], output_root,
                count=count, crop_size_px=crop_size_px, seed=seed)
            outline_status = ('complete' if meta['n_cell_outlines'] == meta['n_nuclear_outlines'] == meta['n_examples']
                              else 'none' if meta['n_cell_outlines'] == meta['n_nuclear_outlines'] == 0 else 'partial')
            record.update(status='reused' if cached is not None else 'generated', outline_status=outline_status,
                          error='', **{key: meta[key] for key in ['n_examples', 'n_cluster_cells',
                          'n_coordinate_matched', 'n_image_eligible', 'n_cell_outlines', 'n_nuclear_outlines',
                          'image_path', 'examples_path', 'signature']})
            record['boundary_issue'] = '; '.join(str(meta.get(k, '')) for k in [
                'boundary_transform_issue', 'cell_boundary_issue', 'nucleus_boundary_issue'] if meta.get(k))
        except Exception as error:
            record.update(status='failed', outline_status='unavailable', error=str(error))
        record['elapsed_seconds'] = round(time.monotonic() - started, 2)
        records.append(record)
        # Persist after each cluster; an interruption does not discard finished PNGs.
        temporary = index_path.with_suffix('.tmp')
        pd.DataFrame(records).to_csv(temporary, index=False)
        temporary.replace(index_path)
        print(f'  {record["status"]}; outlines={record["outline_status"]}; '
              f'{record["elapsed_seconds"]:.1f}s' + (f'; {record["error"]}' if record['error'] else ''), flush=True)
    print(f'Gallery index: {index_path}', flush=True)
    return pd.DataFrame(records)


def save_decision(path, identity, label, notes, *, alignment_confirmed, gallery):
    if label not in LABELS:
        raise ValueError('Choose Normal epithelial, Tumor, or Uncertain.')
    if any(str(identity[k]) != str(gallery.get(k)) for k in KEYS):
        raise ValueError('The displayed gallery belongs to another cluster. Generate the selected gallery first.')
    if label != 'Uncertain' and not alignment_confirmed:
        raise ValueError('Confirm that highlighted cells match the tissue before assigning a biological label.')
    if label != 'Uncertain' and gallery.get('n_cell_outlines', 0) == 0:
        raise ValueError('No selected-cell outlines are available; retain Uncertain until correspondence can be reviewed.')
    path = Path(path)
    existing = pd.read_csv(path, dtype=str, keep_default_na=False) if path.is_file() else pd.DataFrame(columns=KEYS)
    if existing.duplicated(KEYS).any():
        raise ValueError('Duplicate manual decisions; resolve them before saving.')
    same = pd.Series(True, index=existing.index)
    for key in KEYS:
        same &= existing[key].eq(str(identity[key]))
    record = dict(**{k: str(identity[k]) for k in KEYS}, manual_label=label, notes=str(notes),
                  alignment_confirmed=bool(alignment_confirmed), n_examples=gallery['n_examples'],
                  gallery_signature=gallery['signature'], examples_csv=gallery['examples_path'],
                  gallery_png=gallery['image_path'], updated_at_utc=datetime.now(timezone.utc).isoformat())
    result = pd.concat([existing.loc[~same], pd.DataFrame([record])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    result.to_csv(temporary, index=False)
    temporary.replace(path)
    return record


def launch_image_reviewer(inventory, targets, output_root, *, count=20, crop_size_px=256, seed=0):
    import ipywidgets as widgets
    from IPython.display import display, clear_output, Image, FileLink
    from .gene_panel_utils import natural_key

    output_root = Path(output_root)
    decisions_path = output_root / 'manual_epithelial_decisions.csv'
    if inventory.duplicated(['dataset', 'sample']).any():
        raise ValueError('Duplicate sample rows in image inventory.')
    lookup = {(str(r['dataset']), str(r['sample'])): r for r in inventory.to_dict('records')}
    samples = list(targets[['dataset', 'sample']].drop_duplicates().itertuples(index=False, name=None))
    sample = widgets.Dropdown(options=[(' / '.join(p), p) for p in samples], description='Sample:', layout=widgets.Layout(width='95%'))
    cluster = widgets.Dropdown(description='Cluster:')
    generate = widgets.Button(description=f'Show saved {count}-cell PNG', icon='image')
    label = widgets.Dropdown(options=LABELS, description='Decision:')
    notes = widgets.Textarea(description='Notes:', layout=widgets.Layout(width='95%'))
    checked = widgets.Checkbox(description='Cell outlines match the H&E tissue', indent=False)
    save = widgets.Button(description='Save decision', icon='save', disabled=True)
    output, status = widgets.Output(), widgets.Output()
    state = {}

    def identity():
        dataset, name = sample.value
        target = targets.loc[targets.dataset.eq(dataset) & targets['sample'].eq(name) & targets.cluster.eq(cluster.value)]
        return target.iloc[0].to_dict()

    def reset(*_):
        state.clear()
        save.disabled = True
        label.value, notes.value, checked.value = 'Uncertain', '', False
        with output:
            clear_output(wait=False)
        with status:
            clear_output(wait=False)
        if cluster.value is None:
            return
        target = identity()
        if decisions_path.is_file():
            saved = pd.read_csv(decisions_path, dtype=str, keep_default_na=False)
            selected = pd.Series(True, index=saved.index)
            for key in KEYS:
                selected &= saved[key].eq(str(target[key]))
            if selected.any():
                old = saved.loc[selected].iloc[0]
                label.value, notes.value = old.manual_label, old.notes
                with status:
                    print('Saved decision loaded. View the current gallery and confirm alignment again before updating.')

    def load_sample(*_):
        dataset, name = sample.value
        cluster.options = sorted(targets.loc[targets.dataset.eq(dataset) & targets['sample'].eq(name), 'cluster'].tolist(), key=natural_key)
        reset()

    def render(_):
        save.disabled = True
        state.clear()
        checked.value = False
        with output:
            clear_output(wait=True)
            try:
                target = identity()
                row = lookup.get(sample.value)
                if row is None:
                    raise FileNotFoundError('Sample absent from the input inventory. Refresh the input inventory.')
                print(f'{target["dataset"]} / {target["sample"]} / {target["cluster_key"]} / cluster {target["cluster"]}')
                print('Top DE genes:', target.get('top_15_de_genes', '') or 'Unavailable')
                for key, meaning in [('polygon_nuclear_area_um2', 'Median polygon nuclear area (um^2)'),
                                     ('polygon_eccentricity', 'Median polygon eccentricity (0 round, 1 elongated)')]:
                    if target.get(key):
                        print(f'{meaning}: {target[key]}')
                print('H&E:', row.get('he_image', 'Missing'))
                print('Coordinates:', row.get('aligned_cells', 'Missing'))
                print('Image source:', row.get('he_image_source', 'unknown'))
                meta = load_saved_cell_gallery(row, target['cluster_key'], target['cluster'], output_root,
                                           count=count, crop_size_px=crop_size_px, seed=seed)
                print(f'Cluster cells: {meta["n_cluster_cells"]:,}; matched coordinates: {meta["n_coordinate_matched"]:,}; '
                      f'eligible inside image: {meta["n_image_eligible"]:,}; shown: {meta["n_examples"]}.')
                print(f'Outlined examples: cells {meta["n_cell_outlines"]}/{meta["n_examples"]}; '
                      f'nuclei {meta["n_nuclear_outlines"]}/{meta["n_examples"]}.')
                print('Boundary transform:', meta['boundary_transform_source'] or 'unavailable')
                if meta.get('boundary_transform_error_px') is not None:
                    print(f'Coordinate reproduction error: {meta["boundary_transform_error_px"]:.4f}px (not registration accuracy).')
                for key in ['boundary_transform_issue', 'cell_boundary_issue', 'nucleus_boundary_issue']:
                    if meta.get(key):
                        print(f'{key}: {meta[key]}')
                display(Image(filename=meta['image_path']))
                display(FileLink(meta['image_path']), FileLink(meta['examples_path']))
                state['gallery'] = meta
                save.disabled = False
            except Exception as error:
                print(f'Gallery unavailable: {error}')

    def store(_):
        with status:
            clear_output(wait=True)
            try:
                save_decision(decisions_path, identity(), label.value, notes.value,
                              alignment_confirmed=checked.value, gallery=state['gallery'])
                print(f'Saved: {decisions_path}. Existing final labels were not changed.')
            except Exception as error:
                print(f'Decision not saved: {error}')

    sample.observe(load_sample, names='value')
    cluster.observe(reset, names='value')
    generate.on_click(render)
    save.on_click(store)
    load_sample()
    ui = widgets.VBox([sample, widgets.HBox([cluster, generate]), output, label, checked, notes, save, status])
    display(ui)
    print(f'{len(targets)} unresolved clusters. Manual decisions: {decisions_path}')
    return ui
