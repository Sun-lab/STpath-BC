"""TCGA-BRCA image metadata, pixel access, and physical tile grids."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import os
from xml.etree import ElementTree
import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
from .he_tissue import MASK_METHOD_VERSION, color_tissue_mask_rgb
from .image_metadata_manifest import get_hw
DEFAULT_INPUT = Path(os.environ.get('STPATH_TCGA_TIFF_ROOT', str(Path(os.environ.get('XENIUM_ROOT', Path(__file__).resolve().parents[1])) / 'data/TCGA_BRCA_HE_filtered_tiff')))
READ_LIMIT = 128 * 1024 ** 2

def physical_pixel_size(ome_xml: str | None, description: str, series_index: int) -> dict:
    """Trust explicit microscopy metadata, retaining units and provenance."""
    result = {'mpp_x': None, 'mpp_y': None, 'mpp_source': 'unknown'}
    if ome_xml:
        try:
            root = ElementTree.fromstring(ome_xml)
            images = root.findall('{*}Image')
            pixels = images[series_index].find('{*}Pixels')
            factors = {'µm': 1.0, 'μm': 1.0, 'um': 1.0, 'nm': 0.001, 'mm': 1000.0, 'm': 1000000.0}
            values = []
            for axis in ('X', 'Y'):
                unit = pixels.attrib.get(f'PhysicalSize{axis}Unit', 'µm')
                value = float(pixels.attrib[f'PhysicalSize{axis}']) * factors[unit]
                if not math.isfinite(value) or value <= 0:
                    raise ValueError('Physical pixel sizes must be positive and finite')
                values.append(value)
            return {'mpp_x': values[0], 'mpp_y': values[1], 'mpp_source': 'OME PhysicalSizeX/Y converted to µm'}
        except (ElementTree.ParseError, AttributeError, IndexError, KeyError, ValueError):
            result['mpp_source'] = 'unknown: incomplete, invalid, or unsupported OME physical units'
    if 'Aperio' in description:
        match = re.search('(?:^|\\|)\\s*MPP\\s*=\\s*([0-9.eE+-]+)', description)
        if match:
            value = float(match.group(1))
            if math.isfinite(value) and value > 0:
                return {'mpp_x': value, 'mpp_y': value, 'mpp_source': 'Aperio MPP (isotropic)'}
    return result

def validate_pixel_data_bounds(level) -> None:
    """Check declared strip/tile extents against their file, without decoding."""
    for page in level.pages:
        file_size = page.parent.filehandle.size
        for offset, byte_count in zip(page.dataoffsets, page.databytecounts):
            if offset and byte_count and (int(offset) + int(byte_count) > file_size):
                raise ValueError(f'TIFF pixel data extends beyond end of file: declared end {int(offset) + int(byte_count):,} bytes; file size {file_size:,} bytes. The TIFF is incomplete or has invalid offsets/byte counts; restore a valid source image.')

def inspect_slide(path: Path) -> dict:
    with tifffile.TiffFile(path) as tif:
        eligible = [(i, s) for i, s in enumerate(tif.series) if s.axes in ('YXS', 'SYX', 'YX')]
        if not eligible:
            raise ValueError('No supported 2D image series; inspect TIFF axes before choosing a plane')
        index, series = max(eligible, key=lambda item: math.prod(get_hw(item[1].shape)))
        for level in series.levels:
            validate_pixel_data_bounds(level)
        page = series.pages[0]
        height, width = get_hw(series.shape)
        description = page.description or ''
        tags = {name: str(page.tags[name].value) for name in ('XResolution', 'YResolution', 'ResolutionUnit', 'Software', 'Orientation') if name in page.tags}
        orientation = int(page.tags['Orientation'].value) if 'Orientation' in page.tags else 1
        return {'slide_id': path.stem, 'patient_id': '-'.join(path.stem.split('-')[:3]), 'source_path': str(path.resolve()), 'file_size_bytes': path.stat().st_size, 'source_mtime_ns': path.stat().st_mtime_ns, 'series_index': index, 'n_series': len(tif.series), 'axes': series.axes, 'width_px': width, 'height_px': height, 'dtype': str(series.dtype), 'photometric': page.photometric.name, 'orientation': orientation, 'compression': page.compression.name, 'is_tiled': bool(page.is_tiled), 'tile_width': int(page.tilewidth), 'tile_height': int(page.tilelength), 'rows_per_strip': int(page.rowsperstrip), 'levels': [{'shape': list(level.shape), 'axes': level.axes} for level in series.levels], 'tiff_resolution_tags_unverified': tags, 'description_excerpt': description[:500], **physical_pixel_size(tif.ome_metadata, description, index)}

@contextmanager
def image_array(path: Path, series_index: int, level_index: int):
    """Use memory mapping or TIFF-backed Zarr; cap individual decode segments."""
    with tifffile.TiffFile(path) as tif:
        level = tif.series[series_index].levels[level_index]
        validate_pixel_data_bounds(level)
        page = level.pages[0]
        if level.dtype != np.dtype('uint8') or page.photometric.name != 'RGB':
            raise ValueError('Preview requires uint8 RGB H&E; metadata saved without intensity conversion')
        if level.axes not in ('YXS', 'SYX') or level.shape[level.axes.index('S')] != 3:
            raise ValueError('Preview requires three RGB channels; inspect extra channels explicitly')
        try:
            array = tifffile.memmap(path, series=series_index, level=level_index, mode='r')
        except ValueError:
            array = None
        if array is not None:
            try:
                yield (array, level.axes, 'memory_map')
            finally:
                array._mmap.close()
            return
        height, width = get_hw(level.shape)
        segment_pixels = page.tilewidth * page.tilelength if page.is_tiled else width * min(page.rowsperstrip, height)
        if segment_pixels * page.samplesperpixel * level.dtype.itemsize > READ_LIMIT:
            raise ValueError('TIFF decode segment exceeds 128 MiB; use a smaller-strip/tiled derivative for this pilot')
        if math.prod(level.shape) * level.dtype.itemsize <= READ_LIMIT:
            yield (level.asarray(maxworkers=1), level.axes, 'bounded_array')
            return
        try:
            import zarr
        except ImportError as exc:
            raise RuntimeError('Large compressed TIFF requires compatible tifffile and zarr packages') from exc
        try:
            store = level.aszarr(maxworkers=1)
        except (ImportError, ValueError) as exc:
            raise RuntimeError('TIFF/Zarr region reader unavailable; check compatible versions (tested: tifffile 2025.6.11, zarr 3.0.8)') from exc
        try:
            array = zarr.open(store, mode='r')
            yield (array, level.axes, 'tiff_zarr')
        finally:
            store.close()

def rgb_region(array, axes: str, bounds: tuple[int, int, int, int], step: int=1) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    if math.ceil((x1 - x0) / step) * math.ceil((y1 - y0) / step) * 3 > READ_LIMIT:
        raise ValueError("Requested crop exceeds the pilot's 128 MiB allocation limit")
    if axes == 'YXS':
        return np.array(array[y0:y1:step, x0:x1:step, :], copy=True)
    return np.moveaxis(np.array(array[:, y0:y1:step, x0:x1:step], copy=True), 0, -1)

def grid_edges(width: int, height: int, recipe: str, size: float, mpp_x=None, mpp_y=None):
    if recipe == 'physical_um':
        if not mpp_x or not mpp_y:
            raise ValueError('Physical tiling requires verified X and Y pixel sizes')
        tile_w, tile_h = (max(1, round(size / mpp_x)), max(1, round(size / mpp_y)))
    elif recipe == 'native_px':
        tile_w = tile_h = int(size)
    else:
        if recipe == 'stpath_short_edge':
            cols, rows = (int(size), round(size * height / width)) if width <= height else (round(size * width / height), int(size))
        elif recipe == 'stpath_square_grid':
            cols = rows = int(size)
        else:
            raise ValueError(f'Unknown tiling recipe: {recipe}')
        if cols > width or rows > height:
            raise ValueError('Grid has more divisions than image pixels')
        return (np.r_[np.arange(cols) * (width // cols), width], np.r_[np.arange(rows) * (height // rows), height])
    return (np.r_[np.arange(0, width, tile_w), width], np.r_[np.arange(0, height, tile_h), height])

def mask_fractions(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray, width: int, height: int) -> np.ndarray:
    """Approximate grid tissue coverage from the thumbnail; never a cell fraction."""
    if (len(xs) - 1) * (len(ys) - 1) > 2000000:
        raise ValueError('Candidate exceeds two million tiles; increase tile size for this pilot')
    mh, mw = mask.shape
    x0 = np.clip(np.floor(xs[:-1] * mw / width).astype(int), 0, mw - 1)
    x1 = np.clip(np.ceil(xs[1:] * mw / width).astype(int), 1, mw)
    y0 = np.clip(np.floor(ys[:-1] * mh / height).astype(int), 0, mh - 1)
    y1 = np.clip(np.ceil(ys[1:] * mh / height).astype(int), 1, mh)
    integral = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    sums = integral[y1[:, None], x1] - integral[y0[:, None], x1] - integral[y1[:, None], x0] + integral[y0[:, None], x0]
    return sums / ((y1 - y0)[:, None] * (x1 - x0))

def read_tissue_thumbnail(path: Path, metadata: dict) -> np.ndarray:
    """Use the cohort reader and thumbnail scale for tissue eligibility."""
    levels = [i for i, level in enumerate(metadata['levels']) if max(get_hw(tuple(level['shape']))) >= 1024]
    level_index = levels[-1] if levels else 0
    height, width = get_hw(tuple(metadata['levels'][level_index]['shape']))
    with image_array(path, metadata['series_index'], level_index) as (array, axes, _):
        return rgb_region(array, axes, (0, 0, width, height), max(1, math.ceil(max(width, height) / 1536)))

def eligible_physical_grid(metadata: dict, mask: np.ndarray, size: float, min_tissue: float):
    """Return the physical grid and row-major indices of full, tissue-positive tiles."""
    if not math.isfinite(size) or size <= 0 or (not math.isfinite(min_tissue)) or (not 0 < min_tissue <= 1):
        raise ValueError('Crop size must be positive and tissue cutoff must be in (0, 1]')
    width, height = (metadata['width_px'], metadata['height_px'])
    mx, my = (metadata['tiling_mpp_x'], metadata['tiling_mpp_y'])
    xs, ys = grid_edges(width, height, 'physical_um', size, mx, my)
    fractions = mask_fractions(mask, xs, ys, width, height)
    full = (np.diff(ys)[:, None] == max(1, round(size / my))) & (np.diff(xs)[None, :] == max(1, round(size / mx)))
    eligible = np.flatnonzero(full & (fractions >= min_tissue))
    return (xs, ys, fractions, full, eligible)

def sample_anchors(mask: np.ndarray) -> list[tuple[int, int]]:
    tissue = np.flatnonzero(mask)
    background = np.flatnonzero(~mask)
    chosen = []
    if len(tissue):
        count = 5 if len(background) else 6
        indices = (np.linspace(0.1, 0.9, count) * (len(tissue) - 1)).astype(int)
        chosen.extend(tissue[indices].tolist())
    if len(background):
        indices = (np.linspace(0.1, 0.9, 6 - len(chosen)) * (len(background) - 1)).astype(int)
        chosen.extend(background[indices].tolist())
    return [(int(index % mask.shape[1]), int(index // mask.shape[1])) for index in dict.fromkeys(chosen)]

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    columns = list(dict.fromkeys((key for row in rows for key in row)))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})

def apply_audited_spacing(metadata: dict, audit_row: dict) -> None:
    """Use an unchanged TIFF's audited scale for tiling, retaining header metadata."""
    if metadata['slide_id'] != audit_row['slide_id']:
        raise ValueError('TIFF and audit slide IDs differ')
    if audit_row['match_status'] != 'matched_dimensions' or audit_row['spacing_status'] not in {'dicom_xy_verified_conversion_pending', 'dicom_xy_verified_conversion_verified'} or int(audit_row['n_matching_instances']) != 1:
        raise ValueError('Physical tiling requires an unambiguous audit with verified DICOM X/Y spacing')
    for key in ('width_px', 'height_px', 'file_size_bytes', 'source_mtime_ns', 'orientation', 'series_index'):
        if metadata[key] != int(audit_row[key]):
            raise ValueError(f'TIFF {key} differs from the audit; repeat the header audit')
    x, y = (float(audit_row[f'inferred_tiff_mpp_{axis}']) for axis in ('x', 'y'))
    if not all((math.isfinite(v) and v > 0 for v in (x, y))):
        raise ValueError('Audited physical spacing must be positive and finite')
    for axis, value in (('x', x), ('y', y)):
        if metadata.get(f'mpp_{axis}') is not None and (not math.isclose(metadata[f'mpp_{axis}'], value, rel_tol=0.001)):
            raise ValueError('Embedded TIFF scale conflicts with the audited scale')
    scale_source = 'verified native DICOM-to-TIFF conversion with embedded physical spacing' if audit_row['spacing_status'] == 'dicom_xy_verified_conversion_verified' else 'inferred from exact slide-ID/dimension match to verified DICOM spacing'
    metadata.update(tiling_mpp_x=x, tiling_mpp_y=y, tiling_scale_source=scale_source, conversion_provenance=audit_row['conversion_provenance'], matched_SOPInstanceUID=audit_row['matched_SOPInstanceUID'], dicom_metadata_url=audit_row['dicom_metadata_url'])

def select_slides(input_dir: Path, slide_ids: list[str] | None, max_slides: int) -> list[Path]:
    if not 1 <= max_slides <= 5:
        raise ValueError('This pilot accepts 1–5 slides; default is 3')
    files = sorted((path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in ('.tif', '.tiff')))
    if slide_ids:
        selected = [path for path in files if path.stem in slide_ids or path.name in slide_ids]
        missing = [name for name in slide_ids if not any((name in (p.stem, p.name) for p in selected))]
        if missing:
            raise FileNotFoundError(f'Requested slides not found: {missing}')
        if len(selected) > max_slides:
            raise ValueError('Requested slide list exceeds max_slides')
        return selected
    by_patient = {}
    for path in files:
        by_patient.setdefault('-'.join(path.stem.split('-')[:3]), path)
    files = list(by_patient.values())
    if not files:
        raise FileNotFoundError(f'No .tif/.tiff images found in {input_dir}')
    indices = np.linspace(0, len(files) - 1, min(max_slides, len(files))).round().astype(int)
    return [files[i] for i in indices]
