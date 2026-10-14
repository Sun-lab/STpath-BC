"""Descriptive nuclear morphology for saved epithelial clusters; no label inference."""

from contextlib import ExitStack
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import fnmatch
import os
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MorphologyConfig:
    cluster_key: str = 'epithelial_r0_7'
    neighbors: int = 10
    block_size: int = 2048
    halo: int = 128
    use_masks: bool = True
    use_polygons: bool = True
    use_he: bool = False
    alignment_tolerance_px: float = 1.0
    overwrite: bool = False


METRICS = {
    'reported_nucleus_area_um2': 'XOA-reported nuclear area (square microns).',
    'nuclear_area_um2': 'Area of the complete single-component nuclear mask (square microns).',
    'major_axis_um': 'Major axis length of the equivalent second-moment ellipse (microns).',
    'minor_axis_um': 'Minor axis length of the equivalent second-moment ellipse (microns).',
    'eccentricity': 'Equivalent-ellipse eccentricity: 0 circular, approaching 1 elongated.',
    'solidity': 'Mask area divided by its convex-hull area.',
    'circularity': '4*pi*area / Crofton perimeter squared; pixel discretization can give values above 1.',
    'nearest_neighbor_um': 'Distance to the nearest other source-cell centroid (microns).',
    'neighbor_radius_um': 'Distance to the kth other source-cell centroid; k is neighbors_used.',
    'local_density_per_mm2': 'k / (pi*r_k^2), converted from per square micron to per square mm.',
    'neighbor_nuclear_area_cv': 'Population SD / mean of positive reported nuclear areas in the k neighbors.',
    'hematoxylin_mean': 'Mean fixed HED color-deconvolution hematoxylin value in the aligned nuclear mask; uncalibrated.',
    'hematoxylin_sd': 'Population SD of those hematoxylin values; intensity heterogeneity, not a full texture signature.',
}

PAIRED_SHAPE_METRICS = ('nuclear_area_um2', 'major_axis_um', 'minor_axis_um',
                       'eccentricity', 'solidity', 'circularity')
POLYGON_METRICS = {
    'polygon_nuclear_area_um2': 'Area enclosed by the simplified nuclear polygon (square microns).',
    'polygon_perimeter_um': 'Sum of nuclear-polygon edge lengths (microns).',
    'polygon_major_axis_um': 'Major axis of the polygon area-moment-equivalent ellipse (microns).',
    'polygon_minor_axis_um': 'Minor axis of the polygon area-moment-equivalent ellipse (microns).',
    'polygon_eccentricity': 'Polygon area-moment ellipse eccentricity: 0 circular, approaching 1 elongated.',
    'polygon_solidity': 'Nuclear-polygon area divided by its convex-hull area.',
    'polygon_circularity': '4*pi*polygon area / polygon perimeter squared; not the mask Crofton estimator.',
}
METRICS.update(POLYGON_METRICS)


def read_table(path):
    path = Path(path)
    if path.name.endswith('.parquet.gz'):
        with gzip.open(path, 'rb') as stream:
            table = pd.read_parquet(stream)
    elif path.suffix == '.parquet':
        table = pd.read_parquet(path)
    else:
        table = pd.read_csv(path, dtype={'cell_id': str}, keep_default_na=False)
    if 'cell_id' in table:
        table['cell_id'] = table['cell_id'].astype(str).str.strip()
    return table


def validate_ids(table):
    if 'cell_id' not in table or table.cell_id.isna().any() or table.cell_id.astype(str).str.strip().eq('').any():
        raise ValueError('Missing or blank cell IDs.')
    if table.cell_id.duplicated().any():
        raise ValueError('Duplicate cell IDs; row-order matching is not allowed.')


def match_rows(source, target_ids):
    from .cluster_review import match_cluster_source_cell_ids
    validate_ids(source)
    validate_ids(pd.DataFrame({'cell_id': list(target_ids)}))
    source = source.reset_index(drop=True)
    ids = source[['cell_id']].assign(_source_row=np.arange(len(source)).astype(str))
    mapped, audit = match_cluster_source_cell_ids(
        ids, cell_id_columns=['cell_id'], cluster_column='_source_row',
        cluster_key='_source_row', matrix_cell_ids=pd.Index(target_ids))
    mapped = mapped.loc[mapped.cell_id.isin(target_ids)].copy()
    result = source.iloc[mapped._source_row.astype(int)].copy().reset_index(drop=True)
    result['cell_id'] = mapped.cell_id.to_numpy()
    return result, audit['cell_id_match_method']


def _candidate_files(directories, *, recursive=False):
    """Index once; never descend into tiles, generated figures, or Zarr chunks."""
    files = set()
    skip = {'he_tiles', 'tiles', 'retained_tiles', 'training_tiles', 'features',
            'feature_extraction', 'figures', 'analysis', '__pycache__'}
    directories = list(dict.fromkeys(Path(p).resolve() for p in directories))
    if recursive:
        directories = [p for p in directories if not any(parent != p and parent in p.parents for parent in directories)]
    for directory in directories:
        if not directory.is_dir():
            continue
        if not recursive:
            with os.scandir(directory) as entries:
                files.update(Path(e.path) for e in entries if e.is_file())
            continue
        for parent, names, filenames in os.walk(directory, followlinks=False):
            names[:] = [name for name in names if not (
                name.startswith('.') or name.lower() in skip or name.lower().endswith(('_tiles', '.zarr'))
                or name.lower().startswith(('he_tiles_', 'tiles_')))]
            files.update(Path(parent) / name for name in filenames)
    return files


def _unique_path(files, patterns):
    for pattern in patterns:
        matches = sorted(p for p in files if fnmatch.fnmatchcase(p.name, pattern))
        if len(matches) == 1:
            return str(matches[0]), ''
        if len(matches) > 1:
            return '', f'Ambiguous {pattern}: {len(matches)} candidates; set an explicit override.'
    return '', ''


def discover_inputs(root, roots_by_dataset, overrides=None, *, cache_path=None,
                    refresh=False, inspect_archives=False):
    """Discover all saved epithelial samples, with cheap reuse and bounded I/O."""
    root = Path(root).resolve()
    overrides = overrides or {}
    cached = {}
    if cache_path and Path(cache_path).is_file() and not refresh:
        previous = pd.read_csv(cache_path, dtype=str, keep_default_na=False)
        if {'dataset', 'sample', 'discovery_signature', 'discovery_status'}.issubset(previous):
            if previous.duplicated(['dataset', 'sample']).any():
                raise ValueError('Cached input inventory contains duplicate samples; refresh discovery.')
            cached = {(row['dataset'], row['sample']): row for row in previous.to_dict('records')}
    rows = []
    for dataset, checkpoint_root in roots_by_dataset.items():
        output = root / dataset / ('tile_outputs' if dataset == 'Bilous_2026' else '')
        manifest = output / 'tile_input_manifest.csv'
        manifest_stamp = [manifest.stat().st_size, manifest.stat().st_mtime_ns] if manifest.is_file() else None
        manifest_samples, manifest_error = None, ''
        partitions = sorted((Path(checkpoint_root) / dataset).glob('*/epithelial_cell_clusters.csv.gz'))
        for index, clusters in enumerate(partitions, 1):
            started = time.monotonic()
            sample = clusters.parent.name
            print(f'Inputs [{index}/{len(partitions)}] {dataset} / {sample}', flush=True)
            directories = list(dict.fromkeys([root / dataset / sample, output / sample]))
            row = dict(dataset=dataset, sample=sample, clusters=str(clusters.resolve()),
                       checkpoint=str(clusters.with_name('epithelial_expression.h5ad').resolve()),
                       pixel_size_um='', alignment_reviewed=False, tile_input_manifest='',
                       alignment_csv='', manual_alignment_json='', he_image_source='file_discovery')
            notes = []
            patterns = {
                'cells': ['cells.parquet', '*_cells.parquet.gz', 'cells.csv.gz', '*_cells.csv.gz'],
                'masks': ['cells.zarr.zip', 'cell_segmentation.zarr.zip'],
                'boundaries': ['nucleus_boundaries.parquet', '*nucleus_boundaries.parquet.gz',
                               'nucleus_boundaries.csv.gz'],
                'cell_boundaries': ['cell_boundaries.parquet', '*cell_boundaries.parquet.gz',
                                    'cell_boundaries.csv.gz', '*cell_boundaries.csv.gz', 'cell_boundaries.csv'],
                'aligned_cells': [f'{sample}_cells_with_annotations.csv',
                                  f'{sample}_cells_with_cluster_labels.csv', f'{sample}_cells_aligned_to_he.csv'],
                'he_image': ['*he_image.ome.tif', '*he_unaligned_image.ome.tif'],
            }
            override = overrides.get((dataset, sample), {})
            signature = hashlib.sha256(json.dumps(
                dict(version=4, root=str(root), clusters=str(clusters.resolve()),
                     manifest=manifest_stamp, overrides=override, inspect_archives=inspect_archives),
                sort_keys=True, default=str).encode()).hexdigest()
            old = cached.get((dataset, sample), {})
            reuse = (old.get('discovery_signature') == signature and old.get('cells')
                     and all(k in old for k in patterns)
                     and all(not old.get(k) or Path(old[k]).is_file() for k in patterns))
            if reuse:
                row = old.copy()
                row['discovery_status'] = 'cached'
            else:
                row.update({key: '' for key in patterns})
                if manifest_samples is None:
                    manifest_samples = {}
                    if manifest.is_file():
                        from .standard_xenium import discover_samples_from_manifest
                        try:
                            manifest_samples = {p.name: p for p in discover_samples_from_manifest(manifest)}
                        except (ValueError, FileNotFoundError, KeyError) as error:
                            manifest_error = str(error)
                if manifest_error:
                    notes.append(f'Tile manifest: {manifest_error}')
                paths = manifest_samples.get(sample)
                if paths is not None:
                    row['tile_input_manifest'] = str(manifest.resolve())
                    row['he_image_source'] = 'notebook_01_manifest'
                    row['alignment_csv'] = str(getattr(paths, 'alignment', None) or '')
                    row['manual_alignment_json'] = str(getattr(paths, 'manual_alignment', None) or '')
                    for key, value in [('cells', paths.cell_table), ('he_image', paths.he_image),
                                       ('boundaries', paths.nucleus_boundaries),
                                       ('cell_boundaries', getattr(paths, 'cell_boundaries', None))]:
                        # Preserve a missing recorded image path instead of substituting
                        # a different (potentially prealigned or cropped) H&E image.
                        if value and (value.is_file() or key == 'he_image'):
                            row[key] = str(value.resolve())
                    if not paths.he_image:
                        notes.append('Tile input manifest has no H&E image; supply an explicit override.')
                    for suffix in ['_cells_with_annotations.csv', '_cells_with_cluster_labels.csv',
                                   '_cells_aligned_to_he.csv']:
                        aligned = paths.base / f'{sample}{suffix}'
                        if aligned.is_file():
                            row['aligned_cells'] = str(aligned.resolve())
                            break
                    if paths.pixel_size_um:
                        row['pixel_size_um'] = paths.pixel_size_um
                    directories = list(dict.fromkeys([*directories, paths.base, paths.outs]))
                row.update(override)
                if 'he_image' in override:
                    row['he_image_source'] = 'explicit_override'
                wanted = [key for key in patterns if not row[key] and key not in override
                          and not (key == 'he_image' and paths is not None)]
                files = _candidate_files(directories) if wanted or inspect_archives else set()
                unresolved = []
                for key in wanted:
                    row[key], note = _unique_path(files, patterns[key])
                    if note:
                        notes.append(note)
                    elif not row[key]:
                        unresolved.append(key)
                if unresolved:
                    # A single pruned scan supplies every still-missing file type.
                    files.update(_candidate_files(directories, recursive=True))
                    for key in unresolved:
                        row[key], note = _unique_path(files, patterns[key])
                        if note:
                            notes.append(note)
                row['discovery_status'] = 'scanned'
            archives = []
            if not reuse:
                if inspect_archives and (not row['cells'] or not row['masks']):
                    print('  inspecting outer ZIP indexes for missing cells/masks', flush=True)
                    for archive in sorted(p for p in files if p.name.endswith('outs.zip')):
                        try:
                            with zipfile.ZipFile(archive) as bundle:
                                members = [n for n in bundle.namelist()
                                           if Path(n).name in {'cells.zarr.zip', 'cell_segmentation.zarr.zip', 'cells.parquet'}]
                            archives.extend(f'{archive}::{name}' for name in members)
                        except (OSError, zipfile.BadZipFile) as error:
                            notes.append(str(error))
                row['archive_members'] = '; '.join(archives)
            for key in patterns:
                value = str(row[key]).strip()
                if value and not Path(value).is_absolute():
                    row[key] = str(root / value)
            row['basic_ready'] = bool(row['cells'] and Path(row['cells']).is_file())
            row['mask_file_present'] = bool(row['masks'] and Path(row['masks']).is_file())
            row['he_files_present'] = all(row[k] and Path(row[k]).is_file() for k in ['he_image', 'aligned_cells'])
            if not reuse:
                missing_paths = [f'{key}: specified path does not exist' for key in patterns
                                 if row[key] and not Path(row[key]).is_file()]
                row['reason'] = '; '.join(notes + missing_paths + ([] if row['basic_ready'] else ['Missing source cell table.']))
            row['discovery_signature'] = signature
            row['discovery_seconds'] = time.monotonic() - started
            print(f'  {row["discovery_status"]}; cells={row["basic_ready"]}, masks={row["mask_file_present"]}; '
                  f'{row["discovery_seconds"]:.2f}s', flush=True)
            rows.append(row)
    if not rows:
        raise FileNotFoundError('No saved epithelial_cell_clusters.csv.gz files in the selected roots.')
    return pd.DataFrame(rows)


def neighborhood_metrics(cells, k=10):
    """Use all valid source-cell centroids, including non-epithelial cells."""
    from scipy.spatial import cKDTree
    validate_ids(cells)
    if k < 1:
        raise ValueError('neighbors must be positive.')
    xy = cells[['x_centroid', 'y_centroid']].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    valid = np.isfinite(xy).all(axis=1)
    output = pd.DataFrame({'cell_id': cells.cell_id})
    for key in ['nearest_neighbor_um', 'neighbor_radius_um', 'local_density_per_mm2',
                'neighbor_nuclear_area_cv', 'neighbors_used', 'neighbors_with_nuclear_area']:
        output[key] = np.nan
    output['neighborhood_qc'] = 'invalid_coordinates'
    points = xy[valid]
    if len(points) < 2:
        output.loc[valid, 'neighborhood_qc'] = 'too_few_source_cells'
        return output
    used = min(k, len(points) - 1)
    distances, indices = cKDTree(points).query(points, k=used + 1)
    # Remove self explicitly; coincident centroids can alter tie order.
    keep = indices != np.arange(len(points))[:, None]
    order = np.argsort(~keep, axis=1, kind='stable')[:, :used]
    indices = np.take_along_axis(indices, order, axis=1)
    distances = np.take_along_axis(distances, order, axis=1)
    radii = distances.max(axis=1)
    nearest = distances.min(axis=1)
    density = np.divide(used * 1e6, np.pi * radii ** 2, out=np.full(len(points), np.nan), where=radii > 0)
    area = pd.to_numeric(cells.get('nucleus_area', pd.Series(np.nan, index=cells.index)), errors='coerce').to_numpy()[valid]
    neighbor_area = area[indices]
    good = np.isfinite(neighbor_area) & (neighbor_area > 0)
    count = good.sum(axis=1)
    mean = np.divide(np.where(good, neighbor_area, 0).sum(axis=1), count,
                     out=np.full(len(points), np.nan), where=count > 0)
    variance = np.divide(np.where(good, (neighbor_area - mean[:, None]) ** 2, 0).sum(axis=1), count,
                         out=np.full(len(points), np.nan), where=count > 1)
    cv = np.sqrt(variance) / mean
    edge_distance = np.minimum(points - points.min(axis=0), points.max(axis=0) - points).min(axis=1)
    qc = np.where(edge_distance < radii, 'bounding_box_edge', 'ok_unchecked_tissue_edge')
    qc[nearest == 0] = 'coincident_centroids'
    output.loc[valid, ['nearest_neighbor_um', 'neighbor_radius_um', 'local_density_per_mm2',
                       'neighbor_nuclear_area_cv', 'neighbors_used', 'neighbors_with_nuclear_area']] = np.column_stack(
                           [nearest, radii, density, cv, np.full(len(points), used), count])
    output.loc[valid, 'neighborhood_qc'] = qc
    return output


def infer_he_affine(cells, aligned, tolerance=1.0):
    """Recover the recorded affine from matching coordinate tables, not from biology."""
    aligned, _ = match_rows(aligned, cells.cell_id)
    pair = cells[['cell_id', 'x_centroid', 'y_centroid']].merge(
        aligned[['cell_id', 'he_x_px', 'he_y_px']], on='cell_id', validate='one_to_one')
    values = pair.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < 6:
        raise ValueError('At least six non-collinear matched coordinates are required for H&E alignment.')
    x = np.column_stack([values[:, :2], np.ones(len(values))])
    matrix, _, rank, _ = np.linalg.lstsq(x, values[:, 2:], rcond=None)
    errors = np.linalg.norm(x @ matrix - values[:, 2:], axis=1)
    if rank < 3 or errors.max() > tolerance:
        raise ValueError(f'Coordinate tables do not define one consistent affine: max error {errors.max():.3f}px.')
    return np.vstack([matrix.T, [0, 0, 1]]), float(errors.max())


def measure_polygons(boundaries, target_ids, *, nucleus_counts=None, progress=None):
    """Measure simplified polygons in microns, preserving vertex order and cell IDs."""
    from shapely.geometry import Polygon
    required = {'cell_id', 'vertex_x', 'vertex_y'}
    if not required.issubset(boundaries):
        raise ValueError(f'Nuclear boundaries require columns {sorted(required)}.')
    if boundaries.cell_id.isna().any() or boundaries.cell_id.astype(str).str.strip().eq('').any():
        raise ValueError('Nuclear boundary rows contain missing cell IDs.')
    boundaries = boundaries.copy()
    boundaries['cell_id'] = boundaries.cell_id.astype(str).str.strip()
    ids = boundaries[['cell_id']].drop_duplicates().assign(_boundary_id=lambda d: d.cell_id)
    ids, _ = match_rows(ids, target_ids)
    id_map = ids.set_index('_boundary_id').cell_id.to_dict()
    boundaries = boundaries.loc[boundaries.cell_id.isin(id_map)].reset_index(drop=True)
    xy = boundaries[['vertex_x', 'vertex_y']].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    label_ids = boundaries.label_id.to_numpy() if 'label_id' in boundaries else None
    groups = boundaries.groupby('cell_id', sort=False).indices
    rows = []
    for done, (source_id, indices) in enumerate(groups.items(), 1):
        points = xy[indices]
        row = dict(cell_id=id_map[source_id], polygon_qc='ok', polygon_n_vertices=0)
        count = pd.to_numeric(nucleus_counts.get(source_id, np.nan), errors='coerce') if nucleus_counts is not None else np.nan
        if np.isfinite(count) and count != 1:
            row['polygon_qc'] = 'multiple_nuclei' if count > 1 else 'no_nucleus'
        elif label_ids is not None and pd.isna(label_ids[indices]).any():
            row['polygon_qc'] = 'missing_nucleus_label_id'
        elif label_ids is not None and len(pd.unique(label_ids[indices])) != 1:
            row['polygon_qc'] = 'multiple_nucleus_polygons'
        elif not np.isfinite(points).all():
            row['polygon_qc'] = 'nonfinite_vertices'
        else:
            if len(points) > 1 and np.array_equal(points[0], points[-1]):
                points = points[:-1]
            row['polygon_n_vertices'] = len(points)
            if len(points) < 3:
                row['polygon_qc'] = 'too_few_vertices'
            elif np.any(np.all(points[1:] == points[0], axis=1)):
                row['polygon_qc'] = 'multiple_boundary_rings'
            else:
                polygon = Polygon(points)
                if not polygon.is_valid or polygon.area <= 0:
                    row['polygon_qc'] = 'invalid_or_degenerate_polygon'
                else:
                    # Integrate area moments over the polygon, not over its vertices.
                    # Center coordinates first to avoid cancellation far from the origin.
                    centered = points - np.array([polygon.centroid.x, polygon.centroid.y])
                    x, y = centered.T
                    nx, ny = np.roll(centered, -1, axis=0).T
                    cross = x * ny - nx * y
                    signed_area = cross.sum() / 2
                    xx = np.sum((x*x + x*nx + nx*nx) * cross) / (12 * signed_area)
                    yy = np.sum((y*y + y*ny + ny*ny) * cross) / (12 * signed_area)
                    xy_moment = np.sum((2*x*y + x*ny + nx*y + 2*nx*ny) * cross) / (24 * signed_area)
                    eigenvalues = np.linalg.eigvalsh([[xx, xy_moment], [xy_moment, yy]])
                    if eigenvalues[0] <= 0 or not np.isfinite(eigenvalues).all():
                        row['polygon_qc'] = 'degenerate_area_moments'
                    else:
                        row.update(polygon_nuclear_area_um2=polygon.area,
                                   polygon_perimeter_um=polygon.length,
                                   polygon_major_axis_um=4 * np.sqrt(eigenvalues[1]),
                                   polygon_minor_axis_um=4 * np.sqrt(eigenvalues[0]),
                                   polygon_eccentricity=np.sqrt(max(0., 1 - eigenvalues[0] / eigenvalues[1])),
                                   polygon_solidity=polygon.area / polygon.convex_hull.area,
                                   polygon_circularity=4 * np.pi * polygon.area / polygon.length**2)
        rows.append(row)
        if progress and (done == 1 or done % 10000 == 0 or done == len(groups)):
            progress(done, len(groups))
    return pd.DataFrame(rows, columns=['cell_id', 'polygon_qc', 'polygon_n_vertices', *POLYGON_METRICS])


def open_nuclear_masks(path, boundaries='', pixel_size_um=None):
    """Return store/group/mapping/geometry; never guess modern nucleus-to-cell indices."""
    import zarr
    from .standard_xenium import xenium_int_cell_id_to_str
    store = zarr.storage.ZipStore(str(path), mode='r')
    try:
        root = zarr.open_group(store=store, mode='r')
        masks = root['masks/0']
        if len(masks.shape) != 2:
            raise ValueError('Nuclear mask must be a 2D label image.')
        if boundaries and Path(boundaries).is_file():
            boundary = read_table(boundaries)
        else:
            boundary = pd.DataFrame()
        if {'cell_id', 'label_id'}.issubset(boundary):
            mapping = boundary[['cell_id', 'label_id']].drop_duplicates().copy()
        elif 'seg_mask_value' in root and 'cell_id' in root:
            mapping = pd.DataFrame({'cell_id': xenium_int_cell_id_to_str(root['cell_id'][:]),
                                    'label_id': np.asarray(root['seg_mask_value'][:]).ravel()})
        elif (root.attrs.get('major_version') == 4 and 'cell_id' in root
              and root['cell_id'].ndim == 1 and 'polygon_num_vertices' in root):
            # Before XOA 1.3, cell_id was the numeric segmentation mask value.
            # Restrict this rule to the XOA 1.x layout; nuclear IDs differ in XOA 2.x.
            ids = np.asarray(root['cell_id'][:])
            if not np.issubdtype(ids.dtype, np.integer):
                raise ValueError('XOA 1.x mask cell_id values must be integers.')
            if len(ids) != root.attrs.get('number_cells'):
                raise ValueError('XOA 1.x mask cell_id count differs from number_cells.')
            mapping = pd.DataFrame({'cell_id': ids.astype(str), 'label_id': ids})
        else:
            raise ValueError('No explicit nucleus label_id-to-cell mapping. Supply a boundary table with label_id; modern nucleus IDs are not assumed equal to cell indices.')
        numeric_labels = pd.to_numeric(mapping.label_id, errors='raise')
        if not np.isfinite(numeric_labels).all() or (numeric_labels % 1 != 0).any():
            raise ValueError('Nucleus mask IDs must be finite integers.')
        mapping['label_id'] = numeric_labels.astype(np.int64)
        if mapping.label_id.le(0).any() or mapping.label_id.duplicated().any():
            raise ValueError('Nucleus mask IDs must be positive and map to exactly one cell.')
        if 'masks/homogeneous_transform' in root:
            transform = np.asarray(root['masks/homogeneous_transform'][:], dtype=float)
            if transform.shape != (4, 4) or not np.allclose(transform[3], [0, 0, 0, 1]) or not np.allclose(transform[:2, 2], 0):
                raise ValueError('Unsupported physical-to-mask transform.')
            physical_to_mask = transform[np.ix_([0, 1, 3], [0, 1, 3])]
            mask_to_um = np.linalg.inv(physical_to_mask)
            linear = mask_to_um[:2, :2]
            if not np.allclose(linear, np.eye(2) * linear[0, 0]) or linear[0, 0] <= 0:
                raise ValueError('Shape measurement currently requires isotropic, axis-aligned mask pixels.')
            spacing = float(linear[0, 0])
        else:
            spacing = float(pixel_size_um)
            if not np.isfinite(spacing) or spacing <= 0:
                raise ValueError('Mask pixel size is unavailable; provide pixel_size_um explicitly.')
            mask_to_um = np.diag([spacing, spacing, 1.0])
        return store, masks, mapping, spacing, mask_to_um
    except Exception:
        store.close()
        raise


def measure_masks(mask, wanted, spacing, *, block_size=2048, halo=128, intensity=None, progress=None):
    """Measure complete objects in haloed blocks; incomplete objects stay unmeasured."""
    from skimage.measure import regionprops, label
    if block_size < 16 or halo < 1 or spacing <= 0:
        raise ValueError('Positive spacing/halo and block_size >= 16 are required.')
    height, width = mask.shape
    seen, rows, pixel_counts = set(), [], {}
    total = ((height + block_size - 1) // block_size) * ((width + block_size - 1) // block_size)
    done = 0
    wanted = np.array(sorted(set(wanted)), dtype=np.int64)
    columns = ['label_id', 'mask_qc', 'nuclear_centroid_x_px', 'nuclear_centroid_y_px',
               'nuclear_area_um2', 'major_axis_um', 'minor_axis_um', 'eccentricity',
               'solidity', 'circularity', 'hematoxylin_mean', 'hematoxylin_sd', 'he_qc']
    if not len(wanted):
        return pd.DataFrame(columns=columns)
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            y0, x0 = max(0, y - halo), max(0, x - halo)
            y1, x1 = min(height, y + block_size + halo), min(width, x + block_size + halo)
            block = np.asarray(mask[y0:y1, x0:x1])
            block = np.where(np.isin(block, wanted), block, 0)
            core = block[y-y0:min(y+block_size, height)-y0, x-x0:min(x+block_size, width)-x0]
            core_ids, counts = np.unique(core, return_counts=True)
            for object_id, count in zip(core_ids, counts):
                if object_id:
                    pixel_counts[int(object_id)] = pixel_counts.get(int(object_id), 0) + int(count)
            ids = np.unique(block)
            # Include zero even in a block entirely occupied by one object.
            ids = np.union1d(ids, [0])
            compact = np.searchsorted(ids, block).astype(np.int32)
            for prop in regionprops(compact):
                object_id = int(ids[prop.label])
                if object_id in seen:
                    continue
                cy, cx = np.asarray(prop.centroid) + [y0, x0]
                if not (y <= cy < min(y + block_size, height) and x <= cx < min(x + block_size, width)):
                    continue
                r0, c0, r1, c1 = prop.bbox
                if (r0 == 0 and y0 > 0) or (c0 == 0 and x0 > 0) or (r1 == block.shape[0] and y1 < height) or (c1 == block.shape[1] and x1 < width):
                    continue
                seen.add(object_id)
                edge = r0 + y0 == 0 or c0 + x0 == 0 or r1 + y0 == height or c1 + x0 == width
                components = int(label(prop.image, connectivity=2).max())
                status = 'image_edge' if edge else ('disconnected_mask' if components != 1 else 'ok')
                row = dict(label_id=object_id, mask_qc=status, nuclear_centroid_x_px=cx, nuclear_centroid_y_px=cy,
                           _measured_pixels=int(prop.area))
                if status == 'ok':
                    row.update(nuclear_area_um2=float(prop.area * spacing ** 2),
                               major_axis_um=float(prop.axis_major_length * spacing),
                               minor_axis_um=float(prop.axis_minor_length * spacing),
                               eccentricity=float(prop.eccentricity), solidity=float(prop.solidity),
                               circularity=float(4 * np.pi * prop.area / prop.perimeter_crofton ** 2) if prop.perimeter_crofton else np.nan)
                    if intensity is not None:
                        row.update(intensity(prop.image, x0 + c0, y0 + r0))
                rows.append(row)
            done += 1
            if progress:
                progress(done, total, len(rows))
    # A label may have a distant disconnected fragment outside the halo. Verify
    # each measured object's pixel count against the entire streamed mask.
    for row in rows:
        if row.pop('_measured_pixels') != pixel_counts[row['label_id']]:
            row['mask_qc'] = 'incomplete_or_disconnected_global_mask'
            for metric in ['nuclear_area_um2', 'major_axis_um', 'minor_axis_um', 'eccentricity',
                           'solidity', 'circularity', 'hematoxylin_mean', 'hematoxylin_sd']:
                row[metric] = np.nan
            row['he_qc'] = 'invalid_nuclear_mask'
    return pd.DataFrame(rows, columns=columns)


def he_intensity_reader(image, mask_to_he):
    from skimage.color import rgb2hed
    from .tile_utils import apply_affine, extract_rgb_tile, get_hw
    if np.dtype(image.dtype) != np.uint8 or len(image.shape) != 3 or not (image.shape[-1] == 3 or image.shape[0] == 3):
        raise ValueError('H&E intensity currently requires an RGB uint8 TIFF; no per-crop intensity rescaling is applied.')
    height, width = get_hw(image.shape)

    def measure(binary, x0, y0):
        y, x = np.nonzero(binary)
        xy = apply_affine(np.column_stack([x + x0, y + y0]), mask_to_he)
        xy = np.rint(xy).astype(int)
        if (xy < 0).any() or (xy[:, 0] >= width).any() or (xy[:, 1] >= height).any():
            return {'he_qc': 'outside_he_image'}
        xy = np.unique(xy, axis=0)
        xmin, ymin = xy.min(axis=0)
        xmax, ymax = xy.max(axis=0) + 1
        if (xmax - xmin) * (ymax - ymin) > 4_000_000:
            return {'he_qc': 'unexpected_transform_extent'}
        crop = extract_rgb_tile(image, xmin, ymin, xmax, ymax)
        stain = rgb2hed(crop)[..., 0][xy[:, 1] - ymin, xy[:, 0] - xmin]
        return {'hematoxylin_mean': float(stain.mean()), 'hematoxylin_sd': float(stain.std()), 'he_qc': 'ok'}
    return measure


def _signature(row, config):
    files = {}
    for key in ['clusters', 'cells', 'masks', 'boundaries', 'he_image', 'aligned_cells']:
        path = Path(str(row.get(key, '')))
        if str(row.get(key, '')) and path.is_file():
            stat = path.stat()
            files[key] = [str(path.resolve()), stat.st_size, stat.st_mtime_ns]
    payload = {'schema': 3, 'config': {k: v for k, v in asdict(config).items() if k != 'overwrite'},
               'files': files, 'pixel_size_um': str(row.get('pixel_size_um', '')),
               'alignment_reviewed': str(row.get('alignment_reviewed', False))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def run_sample(row, output_root, config=MorphologyConfig()):
    from . import tile_utils
    started = time.monotonic()
    output = Path(output_root) / row['dataset'] / row['sample']
    output.mkdir(parents=True, exist_ok=True)
    fingerprint, metadata = _signature(row, config)
    meta_path = output / 'run_metadata.json'
    evidence_path = output / 'cell_morphology_evidence.csv.gz'
    summary_path = output / 'cluster_morphology_summary.csv'
    comparison_path = output / 'polygon_mask_comparison.csv'
    if all(path.is_file() for path in [meta_path, evidence_path, summary_path, comparison_path]) and not config.overwrite:
        previous = json.loads(meta_path.read_text())
        if previous.get('signature') == fingerprint:
            return {**previous['audit'], 'status': 'reused', 'elapsed_seconds': time.monotonic() - started}
        raise ValueError('Saved morphology inputs/settings changed. Set overwrite=True to recompute this sample.')
    cells = read_table(row['cells'])
    validate_ids(cells)
    target = read_table(row['clusters'])
    validate_ids(target)
    if config.cluster_key not in target:
        raise KeyError(f'Missing saved resolution {config.cluster_key}')
    if any(not target[key].eq(row[key]).all() for key in ['dataset', 'sample']):
        raise ValueError('Cluster rows belong to another dataset/sample.')
    target = target[['dataset', 'sample', 'cell_id', config.cluster_key] +
                    [c for c in ['umap_1', 'umap_2'] if c in target]].rename(columns={config.cluster_key: 'cluster'})
    target['cluster_key'] = config.cluster_key
    spatial = neighborhood_metrics(cells, config.neighbors)
    source = cells[['cell_id']].copy()
    source['reported_nucleus_area_um2'] = pd.to_numeric(cells.get('nucleus_area', pd.Series(np.nan, index=cells.index)), errors='coerce')
    source.loc[~np.isfinite(source.reported_nucleus_area_um2) | source.reported_nucleus_area_um2.le(0), 'reported_nucleus_area_um2'] = np.nan
    source = source.merge(spatial, on='cell_id', validate='one_to_one')
    source, id_method = match_rows(source, target.cell_id)
    evidence = target.merge(source, on='cell_id', how='left', validate='one_to_one', indicator=True)
    evidence['source_cell_status'] = np.where(evidence.pop('_merge').eq('both'), 'matched', 'missing_source_cell')
    evidence['polygon_qc'] = 'not_requested' if not config.use_polygons else 'boundaries_unavailable'
    polygon_reason = ''
    if config.use_polygons:
        if row.get('boundaries') and Path(row['boundaries']).is_file():
            try:
                print('  measuring nuclear boundary polygons', flush=True)
                counts = cells.set_index('cell_id').nucleus_count if 'nucleus_count' in cells else None
                polygons = measure_polygons(
                    read_table(row['boundaries']), target.cell_id, nucleus_counts=counts,
                    progress=lambda done, total: print(f'  polygons {done:,}/{total:,}', flush=True))
                evidence = evidence.drop(columns='polygon_qc').merge(polygons, on='cell_id', how='left', validate='one_to_one')
                evidence['polygon_qc'] = evidence.polygon_qc.fillna('no_boundary_for_cell')
            except (ValueError, KeyError, OSError, TypeError, ImportError) as error:
                polygon_reason = str(error)
                evidence['polygon_qc'] = 'polygon_processing_unavailable'
        else:
            polygon_reason = 'Nuclear boundary file missing.'
    evidence['polygon_detail'] = polygon_reason
    evidence['mask_qc'] = 'not_requested' if not config.use_masks else 'mask_unavailable'
    evidence['he_qc'] = 'not_requested' if not config.use_he else 'not_measured'
    mask_reason = 'Mask file missing.' if config.use_masks and not row.get('masks') else ''
    he_reason = ''
    if config.use_masks and row.get('masks') and Path(row['masks']).is_file():
        with ExitStack() as stack:
            try:
                store, masks, mapping, spacing, mask_to_um = open_nuclear_masks(row['masks'], row.get('boundaries', ''), row.get('pixel_size_um'))
                stack.callback(store.close)
                multiple = mapping.groupby('cell_id').size()
                single = mapping.loc[mapping.cell_id.map(multiple).eq(1)].copy()
                if 'nucleus_count' in cells:
                    counts = cells.set_index('cell_id').nucleus_count
                    single = single.loc[pd.to_numeric(single.cell_id.map(counts), errors='coerce').fillna(1).eq(1)]
                single, _ = match_rows(single, target.cell_id)
                intensity = None
                if config.use_he:
                    try:
                        if str(row.get('alignment_reviewed', False)).lower() not in {'true', '1', 'yes'}:
                            raise ValueError('Set alignment_reviewed=True only after reviewing H&E overlays.')
                        affine, error = infer_he_affine(cells, read_table(row['aligned_cells']), config.alignment_tolerance_px)
                        import tifffile
                        import zarr
                        tif = stack.enter_context(tifffile.TiffFile(row['he_image']))
                        image_store = tif.series[0].levels[0].aszarr()
                        stack.callback(image_store.close)
                        image = tile_utils.resolve_zarr_array(zarr.open(image_store, mode='r'))
                        intensity = he_intensity_reader(image, affine @ mask_to_um)
                    except (ValueError, KeyError, OSError) as error:
                        he_reason = str(error)
                def progress(done, total, count):
                    if done == 1 or done % 25 == 0 or done == total:
                        print(f'  mask blocks {done}/{total}; {count:,} nuclei processed', flush=True)
                measured = measure_masks(masks, single.label_id, spacing, block_size=config.block_size,
                                         halo=config.halo, intensity=intensity, progress=progress)
                measured = single.merge(measured, on='label_id', how='left', validate='one_to_one')
                measured['mask_qc'] = measured.mask_qc.fillna('no_complete_mask_in_blocks')
                evidence = evidence.drop(columns=['mask_qc', 'he_qc']).merge(measured, on='cell_id', how='left', validate='one_to_one')
                evidence['mask_qc'] = evidence.mask_qc.fillna('no_single_nucleus_mapping')
                evidence['he_qc'] = evidence.he_qc.fillna('not_measured' if config.use_he else 'not_requested')
            except (ValueError, KeyError, OSError, TypeError) as error:
                mask_reason = str(error)
                evidence['mask_qc'] = 'mask_processing_unavailable'
    for metric in METRICS:
        if metric not in evidence:
            evidence[metric] = np.nan
    if 'polygon_n_vertices' not in evidence:
        evidence['polygon_n_vertices'] = np.nan
    evidence['mask_detail'] = mask_reason
    evidence['he_detail'] = he_reason
    summary = summarize_clusters(evidence)
    temporary = evidence_path.with_name(evidence_path.name + '.tmp')
    evidence.to_csv(temporary, index=False, compression='gzip')
    temporary.replace(evidence_path)
    summary.to_csv(summary_path, index=False)
    summarize_polygon_mask_pairs(evidence).to_csv(comparison_path, index=False)
    audit = dict(dataset=row['dataset'], sample=row['sample'], status='success', n_target_cells=len(evidence),
                 n_source_matched=int(evidence.source_cell_status.eq('matched').sum()),
                 n_mask_measured=int(evidence.mask_qc.eq('ok').sum()), n_he_measured=int(evidence.he_qc.eq('ok').sum()),
                 n_polygon_measured=int(evidence.polygon_qc.eq('ok').sum()), polygon_detail=polygon_reason,
                 cell_id_match_method=id_method, mask_detail=mask_reason, he_detail=he_reason,
                 evidence_path=str(evidence_path), elapsed_seconds=time.monotonic() - started)
    import importlib.metadata
    metadata['versions'] = {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-image']}
    try:
        metadata['versions']['shapely'] = importlib.metadata.version('shapely')
    except importlib.metadata.PackageNotFoundError:
        metadata['versions']['shapely'] = 'not_installed'
    meta_path.write_text(json.dumps(dict(signature=fingerprint, audit=audit, **metadata), indent=2))
    return audit


def summarize_clusters(evidence):
    rows = []
    keys = ['dataset', 'sample', 'cluster_key', 'cluster']
    for values, group in evidence.groupby(keys, sort=False, dropna=False):
        for metric in METRICS:
            data = pd.to_numeric(group.get(metric, pd.Series(dtype=float)), errors='coerce')
            data = data[np.isfinite(data)]
            rows.append(dict(zip(keys, values), metric=metric, n_cluster_cells=len(group), n_measured=len(data),
                             fraction_measured=len(data) / len(group), median=data.median(),
                             q25=data.quantile(.25), q75=data.quantile(.75)))
    return pd.DataFrame(rows)


def summarize_polygon_mask_pairs(evidence):
    """Compare like metrics on the same cells, separately for each saved cluster."""
    keys = ['dataset', 'sample', 'cluster_key', 'cluster']
    columns = [*keys, 'metric', 'n_cluster_cells', 'n_paired_cells', 'fraction_paired',
               'median_mask', 'median_polygon', 'median_difference', 'median_absolute_difference']
    rows = []
    for values, group in evidence.groupby(keys, sort=False, dropna=False):
        for metric in PAIRED_SHAPE_METRICS:
            mask = pd.to_numeric(group.get(metric, pd.Series(np.nan, index=group.index)), errors='coerce')
            polygon = pd.to_numeric(group.get('polygon_' + metric, pd.Series(np.nan, index=group.index)), errors='coerce')
            paired = np.isfinite(mask) & np.isfinite(polygon)
            mask, polygon = mask[paired], polygon[paired]
            difference = polygon - mask
            rows.append(dict(zip(keys, values), metric=metric, n_cluster_cells=len(group),
                             n_paired_cells=len(mask), fraction_paired=len(mask) / len(group),
                             median_mask=mask.median(), median_polygon=polygon.median(),
                             median_difference=difference.median(), median_absolute_difference=difference.abs().median()))
    return pd.DataFrame(rows, columns=columns)


def run_all(inventory, output_root, config=MorphologyConfig()):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    audits = []
    for index, row in enumerate(inventory.to_dict('records'), 1):
        print(f'[{index}/{len(inventory)}] {row["dataset"]} / {row["sample"]}', flush=True)
        try:
            if not row.get('cells') or not Path(row['cells']).is_file():
                raise FileNotFoundError('Missing source cells table; check input_inventory.csv.')
            audits.append(run_sample(row, output_root, config))
        except Exception as error:
            audits.append(dict(dataset=row['dataset'], sample=row['sample'], status='failed', error=str(error)))
            print(f'  failed: {error}', flush=True)
        pd.DataFrame(audits).to_csv(output_root / 'run_summary.csv', index=False)
    return pd.DataFrame(audits)


def plot_sample(evidence, output_dir):
    import matplotlib.pyplot as plt
    from .gene_panel_utils import natural_key
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [m for m in METRICS if m in evidence and evidence[m].notna().any()]
    clusters = sorted(evidence.cluster.astype(str).unique(), key=natural_key)
    paths = []
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(9, 5))
        for cluster in clusters:
            values = np.sort(evidence.loc[evidence.cluster.astype(str).eq(cluster), metric].dropna().to_numpy(float))
            if len(values):
                # Plot decimation only; quantiles and source files use every measured cell.
                indices = np.unique(np.linspace(0, len(values) - 1, min(len(values), 2000)).astype(int))
                ax.plot(values[indices], (indices + 1) / len(values), label=f'{cluster} (n={len(values):,})')
        ax.set(xlabel=metric, ylabel='Cumulative fraction of measured cells',
               title=f'{evidence.dataset.iloc[0]} / {evidence["sample"].iloc[0]}')
        ax.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        fig.tight_layout()
        path = output_dir / f'{metric}.png'
        fig.savefig(path, dpi=160, bbox_inches='tight')
        plt.close(fig)
        paths.append(path)
    return paths


def alignment_preview(row, output_dir, n_regions=6, region_size_px=512):
    """Review H&E correspondence without requiring cancer recognition."""
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    import tifffile
    import zarr
    from .tile_utils import apply_affine, extract_rgb_tile, get_hw, resolve_zarr_array
    if n_regions < 1 or region_size_px < 32:
        raise ValueError('Use positive n_regions and region_size_px >= 32.')
    cells = read_table(row['cells'])
    aligned = read_table(row['aligned_cells'])
    affine, residual = infer_he_affine(cells, aligned)
    aligned, _ = match_rows(aligned, read_table(row['clusters']).cell_id)
    xy = aligned[['he_x_px', 'he_y_px']].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    with ExitStack() as stack:
        tif = stack.enter_context(tifffile.TiffFile(row['he_image']))
        image_store = tif.series[0].levels[0].aszarr()
        stack.callback(image_store.close)
        image = resolve_zarr_array(zarr.open(image_store, mode='r'))
        height, width = get_hw(image.shape)
        valid = np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
        xy = xy[valid]
        if not len(xy):
            raise ValueError('No epithelial centroids fall inside H&E.')
        # Farthest-point selection distributes examples spatially, without using labels.
        chosen = [int(np.argmin(np.linalg.norm(xy - np.median(xy, axis=0), axis=1)))]
        distance = np.full(len(xy), np.inf)
        for _ in range(min(n_regions, len(xy)) - 1):
            distance = np.minimum(distance, np.linalg.norm(xy - xy[chosen[-1]], axis=1))
            chosen.append(int(np.argmax(distance)))
        masks, mask_to_he = None, None
        overlay_note = 'centroid-only overlay'
        if row.get('masks') and Path(row['masks']).is_file():
            try:
                store, masks, _, _, mask_to_um = open_nuclear_masks(row['masks'], row.get('boundaries', ''), row.get('pixel_size_um'))
                stack.callback(store.close)
                mask_to_he = affine @ mask_to_um
                overlay_note = 'cyan: native nuclear-mask boundaries'
            except (ValueError, OSError, KeyError, TypeError) as error:
                overlay_note += f'; masks unavailable: {error}'
        fig, axes = plt.subplots(int(np.ceil(len(chosen) / 3)), 3, figsize=(15, 5 * int(np.ceil(len(chosen) / 3))), squeeze=False)
        for ax in axes.flat:
            ax.set_axis_off()
        for ax, index in zip(axes.flat, chosen):
            cx, cy = xy[index]
            x0, y0 = max(0, int(cx) - region_size_px // 2), max(0, int(cy) - region_size_px // 2)
            x1, y1 = min(width, x0 + region_size_px), min(height, y0 + region_size_px)
            crop = extract_rgb_tile(image, x0, y0, x1, y1)
            ax.imshow(crop)
            inside = (xy[:, 0] >= x0) & (xy[:, 0] < x1) & (xy[:, 1] >= y0) & (xy[:, 1] < y1)
            ax.scatter(xy[inside, 0] - x0, xy[inside, 1] - y0, s=18, facecolors='none', edgecolors='yellow', linewidths=.7)
            if masks is not None:
                corners = apply_affine(np.array([[x0, y0], [x0, y1], [x1, y0], [x1, y1]]), np.linalg.inv(mask_to_he))
                mx0, my0 = np.maximum(0, np.floor(corners.min(axis=0)).astype(int) - 2)
                mx1, my1 = np.minimum([masks.shape[1], masks.shape[0]], np.ceil(corners.max(axis=0)).astype(int) + 2)
                if mx1 > mx0 and my1 > my0 and (mx1 - mx0) * (my1 - my0) <= 8_000_000:
                    by, bx = np.nonzero(find_boundaries(np.asarray(masks[my0:my1, mx0:mx1])))
                    step = max(1, len(bx) // 15000)
                    points = apply_affine(np.column_stack([bx[::step] + mx0, by[::step] + my0]), mask_to_he)
                    ax.scatter(points[:, 0] - x0, points[:, 1] - y0, s=.3, color='cyan', alpha=.5)
            ax.set(xlim=(0, x1 - x0), ylim=(y1 - y0, 0), title=f'H&E x={int(cx):,}, y={int(cy):,}')
        fig.suptitle(f'{row["dataset"]} / {row["sample"]}\nYellow: epithelial cell centroids; {overlay_note}\nAffine reproduction error: {residual:.4f}px (not an independent registration accuracy estimate)', fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .92))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / 'alignment_preview.png'
        fig.savefig(path, dpi=130, bbox_inches='tight')
        plt.close(fig)
    return path


def launch_reviewer(inventory, output_root, max_umap_points=100000):
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from io import BytesIO
    from IPython.display import display, clear_output, Markdown, Image
    from .gene_panel_utils import natural_key
    options = []
    lookup = {}
    for row in inventory.to_dict('records'):
        path = Path(output_root) / row['dataset'] / row['sample'] / 'cell_morphology_evidence.csv.gz'
        if path.is_file():
            name = f'{row["dataset"]} / {row["sample"]}'
            options.append(name)
            lookup[name] = (row, path)
    if not options:
        print(f'No saved morphology evidence under {output_root}. Run the measurement section on the server first.')
        return None
    sample = widgets.Dropdown(options=options, description='Sample:', layout=widgets.Layout(width='95%'))
    cluster = widgets.Dropdown(description='Cluster:')
    metric = widgets.Dropdown(options=list(METRICS), description='Feature:', layout=widgets.Layout(width='70%'))
    output = widgets.Output()
    state = {}

    def render(*_):
        if state.get('loading') or 'table' not in state or cluster.value is None:
            return
        with output:
            clear_output(wait=True)
            table = state['table']
            chosen = table.cluster.astype(str).eq(cluster.value)
            values = pd.to_numeric(table[metric.value], errors='coerce')
            measured = values.notna()
            display(Markdown(f'**{METRICS[metric.value]}**\n\nCluster {cluster.value}: {int((chosen & measured).sum()):,} / {int(chosen.sum()):,} cells measured. Missing values are excluded, not replaced with zero.'))
            paired_metric = metric.value.removeprefix('polygon_')
            compare = paired_metric in PAIRED_SHAPE_METRICS
            fig, axes = plt.subplots(1, 3 if compare else 2, figsize=(18 if compare else 13, 4.5))
            for name in sorted(table.cluster.astype(str).unique(), key=natural_key):
                group = np.sort(values[table.cluster.astype(str).eq(name) & measured])
                if len(group):
                    indices = np.unique(np.linspace(0, len(group) - 1, min(len(group), 2000)).astype(int))
                    selected = name == cluster.value
                    axes[0].plot(group[indices], (indices + 1) / len(group), color='#C64A50' if selected else '#777777',
                                 alpha=1 if selected else .3, linewidth=2 if selected else 1, label=name if selected else None)
            axes[0].set(xlabel=metric.value, ylabel='Cumulative fraction', title='Selected cluster (red) versus other clusters (gray)')
            if {'umap_1', 'umap_2'}.issubset(table) and measured.any():
                valid = table.loc[measured].copy()
                if len(valid) > max_umap_points:
                    valid = valid.sample(max_umap_points, random_state=0)
                dots = axes[1].scatter(valid.umap_1, valid.umap_2, c=valid[metric.value], s=1, cmap='viridis', rasterized=True)
                fig.colorbar(dots, ax=axes[1], label=metric.value)
                selected = table.loc[chosen]
                if len(selected) > 5000:
                    selected = selected.sample(5000, random_state=0)
                axes[1].scatter(selected.umap_1, selected.umap_2, s=4, facecolors='none', edgecolors='#C64A50', alpha=.15)
                axes[1].set(title='Saved expression UMAP; selected cluster outlined', xlabel='UMAP 1', ylabel='UMAP 2')
            else:
                axes[1].text(.5, .5, 'No measured UMAP values available', ha='center', transform=axes[1].transAxes)
            if compare:
                pair = table.loc[chosen, [paired_metric, 'polygon_' + paired_metric]]
                pair = pair.loc[np.isfinite(pair).all(axis=1)]
                if len(pair):
                    density = axes[2].hexbin(pair.iloc[:, 0], pair.iloc[:, 1], gridsize=40,
                                            mincnt=1, bins='log', cmap='viridis')
                    fig.colorbar(density, ax=axes[2], label='Paired cells per bin (log color scale)')
                    low, high = pair.to_numpy().min(), pair.to_numpy().max()
                    padding = max((high - low) * .05, abs(high) * .01, .01)
                    limits = (low - padding, high + padding)
                    axes[2].plot(limits, limits, '--', color='#C64A50', linewidth=1)
                    axes[2].set(xlim=limits, ylim=limits, aspect='equal')
                else:
                    axes[2].text(.5, .5, 'No paired polygon/mask measurements', ha='center',
                                 transform=axes[2].transAxes)
                axes[2].set(xlabel=f'Mask: {paired_metric}', ylabel=f'Polygon: {paired_metric}',
                            title=f'Cluster {cluster.value}: {len(pair):,} paired cells\nDashed line: equal values')
            fig.tight_layout()
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=130, bbox_inches='tight')
            display(Image(data=buffer.getvalue()))
            plt.close(fig)
            row, _path = lookup[sample.value]
            marker_path = Path(row['clusters']).parent / 'expression_review' / table.cluster_key.iloc[0] / 'top_marker_genes.csv'
            if marker_path.is_file():
                markers = pd.read_csv(marker_path, dtype={'cluster': str})
                markers = markers.loc[markers.cluster.eq(cluster.value)].head(15)
                gene_col = 'gene' if 'gene' in markers else 'names'
                genes = ', '.join(markers[gene_col].astype(str)) if gene_col in markers else ''
                display(Markdown(f'**Saved top DE genes:** {genes or "No qualifying markers saved for this cluster."}'))
            else:
                display(Markdown('**DE evidence:** No saved top-marker file at this resolution.'))
            print(f'Complete per-cell evidence: {_path}')
    def load_sample(*_):
        state['loading'] = True
        row, path = lookup[sample.value]
        table = read_table(path)
        for key in METRICS:
            table[key] = pd.to_numeric(table.get(key, pd.Series(np.nan, index=table.index)), errors='coerce')
        state['table'] = table
        cluster.options = sorted(table.cluster.astype(str).unique(), key=natural_key)
        cluster.value = cluster.options[0] if cluster.options else None
        state['loading'] = False
        render()
    sample.observe(load_sample, names='value')
    cluster.observe(render, names='value')
    metric.observe(render, names='value')
    panel = widgets.VBox([sample, widgets.HBox([cluster, metric]), output])
    display(panel)
    load_sample()
    return panel
