#!/usr/bin/env python3
"""Extract native-resolution H&E images from CZI or NDPI whole-slide files.

CZI files may contain several mosaic scenes. Each selected scene is written as
one tiled BigTIFF/OME-TIFF. NDPI files are written from pyramid level 0, which
is the highest-resolution OpenSlide level.

The extractor streams one tile at a time and therefore does not load a whole
slide into memory. Completed output files are skipped unless --overwrite is
specified. Gzip-wrapped source files are decompressed once and reused.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Iterable
import xml.etree.ElementTree as ET


def require_common_dependencies():
    try:
        import numpy as np
        import tifffile
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise ImportError(
            "Missing extraction dependencies. Install them with:\n"
            "python -m pip install -r requirements.txt"
        ) from exc
    return np, tifffile, tqdm


def strip_known_suffixes(path: Path) -> str:
    """Return a stable output stem for .czi[.gz] and .ndpi[.gz] files."""
    name = path.name
    if name.lower().endswith(".gz"):
        name = name[:-3]
    for suffix in (".czi", ".ndpi"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def source_kind(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".czi") or name.endswith(".czi.gz"):
        return "czi"
    if name.endswith(".ndpi") or name.endswith(".ndpi.gz"):
        return "ndpi"
    raise ValueError(f"Unsupported H&E image format: {path}")


def ensure_uncompressed(path: Path, tqdm: Any) -> tuple[Path, bool]:
    """Decompress a gzip-wrapped slide beside its archive and reuse it later."""
    path = path.resolve()
    if path.suffix.lower() != ".gz":
        return path, False

    output = path.with_suffix("")
    if output.exists() and output.stat().st_size > 0:
        print(f"Reusing decompressed source: {output}")
        return output, False

    temporary = output.with_name(output.name + ".partial")
    temporary.unlink(missing_ok=True)
    compressed_size = path.stat().st_size

    print(f"Decompressing: {path.name}")
    try:
        with path.open("rb") as raw_source:
            with gzip.GzipFile(fileobj=raw_source, mode="rb") as source:
                with temporary.open("wb") as destination:
                    with tqdm(
                        total=compressed_size,
                        unit="B",
                        unit_scale=True,
                        desc="Decompress",
                    ) as progress:
                        previous_position = 0
                        while True:
                            chunk = source.read(64 * 1024**2)
                            if not chunk:
                                break
                            destination.write(chunk)
                            current_position = raw_source.tell()
                            progress.update(current_position - previous_position)
                            previous_position = current_position
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return output, True


def normalize_rgb_tile(image: Any, pixel_type: str, np: Any) -> Any:
    """Convert reader-specific tile shapes and BGR color into uint8 RGB."""
    tile = np.asarray(image)
    while tile.ndim > 3 and tile.shape[0] == 1:
        tile = tile[0]

    if tile.ndim == 2:
        tile = np.repeat(tile[..., None], 3, axis=-1)
    elif tile.ndim == 3 and tile.shape[-1] >= 3:
        tile = tile[..., :3]
        if str(pixel_type).lower().startswith("bgr"):
            tile = tile[..., ::-1]
    else:
        raise ValueError(f"Unexpected tile shape: {tile.shape}")

    if tile.dtype != np.uint8:
        if np.issubdtype(tile.dtype, np.integer):
            maximum = np.iinfo(tile.dtype).max
            tile = np.rint(tile.astype(np.float32) * (255.0 / maximum))
        else:
            finite = tile[np.isfinite(tile)]
            if finite.size == 0:
                tile = np.zeros(tile.shape, dtype=np.uint8)
            else:
                low, high = np.percentile(finite, [0.1, 99.9])
                tile = (tile.astype(np.float32) - low) * (
                    255.0 / max(high - low, 1e-12)
                )
        tile = np.clip(tile, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(tile)


def parse_czi_pixel_sizes_um(czi: Any) -> tuple[float | None, float | None]:
    """Read X/Y physical pixel sizes from CZI XML; values are stored in meters."""
    metadata = czi.meta
    if isinstance(metadata, str):
        root = ET.fromstring(metadata)
    else:
        root = metadata

    values: dict[str, float] = {}
    for distance in root.findall(".//Distance"):
        axis = distance.attrib.get("Id")
        value = distance.findtext("Value")
        if axis in {"X", "Y"} and value is not None:
            values[axis] = float(value) / 1e-6
    return values.get("X"), values.get("Y")


def parse_czi_scene_names(czi: Any) -> dict[int, str]:
    metadata = czi.meta
    if isinstance(metadata, str):
        root = ET.fromstring(metadata)
    else:
        root = metadata
    names: dict[int, str] = {}
    for scene in root.findall(".//Scenes/Scene"):
        try:
            index = int(scene.attrib["Index"])
        except (KeyError, ValueError):
            continue
        names[index] = scene.attrib.get("Name", f"scene_{index}")
    return names


def tile_positions(width: int, height: int, tile_size: int):
    for y in range(0, height, tile_size):
        tile_height = min(tile_size, height - y)
        for x in range(0, width, tile_size):
            tile_width = min(tile_size, width - x)
            yield x, y, tile_width, tile_height


def write_tiled_ome_tiff(
    output: Path,
    width: int,
    height: int,
    tile_reader: Callable[[int, int, int, int], Any],
    tile_size: int,
    compression: str,
    physical_size_x_um: float | None,
    physical_size_y_um: float | None,
    description: str,
    overwrite: bool,
    np: Any,
    tifffile: Any,
    tqdm: Any,
) -> str:
    """Stream RGB tiles into an atomic, tiled BigTIFF/OME-TIFF write."""
    if output.exists() and not overwrite:
        print(f"Skipping completed output: {output}")
        return "skipped"

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    temporary.unlink(missing_ok=True)

    total_tiles = math.ceil(width / tile_size) * math.ceil(height / tile_size)
    metadata: dict[str, Any] = {
        "axes": "YXS",
        "Description": description,
    }
    if physical_size_x_um is not None:
        metadata["PhysicalSizeX"] = physical_size_x_um
        metadata["PhysicalSizeXUnit"] = "µm"
    if physical_size_y_um is not None:
        metadata["PhysicalSizeY"] = physical_size_y_um
        metadata["PhysicalSizeYUnit"] = "µm"

    write_options: dict[str, Any] = {
        "shape": (height, width, 3),
        "dtype": np.uint8,
        "photometric": "rgb",
        "tile": (tile_size, tile_size),
        "metadata": metadata,
    }
    if compression != "none":
        write_options["compression"] = compression
        if compression == "deflate":
            write_options["compressionargs"] = {"level": 6}

    def tiles() -> Iterable[Any]:
        with tqdm(
            total=total_tiles,
            unit="tile",
            desc=output.name,
        ) as progress:
            for x, y, tile_width, tile_height in tile_positions(
                width, height, tile_size
            ):
                tile = tile_reader(x, y, tile_width, tile_height)
                progress.update(1)
                yield tile

    try:
        with tifffile.TiffWriter(temporary, bigtiff=True, ome=True) as writer:
            writer.write(tiles(), **write_options)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    print(f"Saved: {output}")
    return "written"


def inspect_czi(path: Path) -> tuple[Any, dict[int, Any], dict[int, str]]:
    try:
        from aicspylibczi import CziFile
    except ImportError as exc:
        raise ImportError(
            "CZI extraction requires aicspylibczi. Install dependencies with:\n"
            "python -m pip install -r requirements.txt"
        ) from exc

    czi = CziFile(path)
    if not czi.is_mosaic():
        raise ValueError(
            f"This extractor currently expects a mosaic CZI file: {path}"
        )
    boxes = czi.get_all_mosaic_scene_bounding_boxes()
    return czi, boxes, parse_czi_scene_names(czi)


def extract_czi(
    source: Path,
    original_source: Path,
    output_dir: Path,
    selected_scenes: list[int] | None,
    tile_size: int,
    compression: str,
    overwrite: bool,
    dry_run: bool,
    np: Any,
    tifffile: Any,
    tqdm: Any,
) -> list[dict[str, Any]]:
    czi, scene_boxes, scene_names = inspect_czi(source)
    pixel_size_x, pixel_size_y = parse_czi_pixel_sizes_um(czi)
    pixel_type = str(czi.pixel_type)

    if selected_scenes is None:
        scenes = sorted(scene_boxes)
    else:
        unknown = sorted(set(selected_scenes) - set(scene_boxes))
        if unknown:
            raise ValueError(
                f"Requested scenes {unknown} are unavailable; "
                f"available scenes are {sorted(scene_boxes)}"
            )
        scenes = selected_scenes

    stem = strip_known_suffixes(original_source)
    records: list[dict[str, Any]] = []
    for scene_index in scenes:
        box = scene_boxes[scene_index]
        scene_name = scene_names.get(scene_index, f"scene_{scene_index}")
        output = output_dir / f"{stem}_scene_{scene_index:02d}.ome.tif"
        record: dict[str, Any] = {
            "source": str(original_source),
            "decompressed_source": str(source),
            "format": "CZI",
            "scene_index": scene_index,
            "scene_name": scene_name,
            "source_bbox": {"x": box.x, "y": box.y, "width": box.w, "height": box.h},
            "width": box.w,
            "height": box.h,
            "pixel_type": pixel_type,
            "physical_size_x_um": pixel_size_x,
            "physical_size_y_um": pixel_size_y,
            "output": str(output),
        }
        print(
            f"CZI scene {scene_index} ({scene_name}): "
            f"{box.w:,} × {box.h:,} px -> {output.name}"
        )

        if dry_run:
            record["status"] = "planned"
            records.append(record)
            continue

        def read_tile(
            x: int,
            y: int,
            width: int,
            height: int,
            *,
            _box: Any = box,
        ) -> Any:
            region = (_box.x + x, _box.y + y, width, height)
            image = czi.read_mosaic(
                C=0,
                scale_factor=1.0,
                region=region,
                background_color=(1.0, 1.0, 1.0),
            )
            tile = normalize_rgb_tile(image, pixel_type, np)
            if tile.shape[:2] != (height, width):
                raise ValueError(
                    f"CZI returned tile {tile.shape} for requested "
                    f"region {(x, y, width, height)}"
                )
            return tile

        record["status"] = write_tiled_ome_tiff(
            output=output,
            width=box.w,
            height=box.h,
            tile_reader=read_tile,
            tile_size=tile_size,
            compression=compression,
            physical_size_x_um=pixel_size_x,
            physical_size_y_um=pixel_size_y,
            description=(
                f"Native-resolution H&E extracted from {original_source.name}; "
                f"CZI scene {scene_index} ({scene_name}); source bounding box "
                f"x={box.x}, y={box.y}, width={box.w}, height={box.h}."
            ),
            overwrite=overwrite,
            np=np,
            tifffile=tifffile,
            tqdm=tqdm,
        )
        records.append(record)

    return records


def open_ndpi(path: Path):
    try:
        import openslide
    except (ImportError, OSError) as exc:
        raise ImportError(
            "OpenSlide is unavailable in this environment"
        ) from exc
    return openslide, openslide.OpenSlide(str(path))


def optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def rational_as_float(value: Any) -> float | None:
    try:
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            return float(value.numerator) / float(value.denominator)
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return float(value[0]) / float(value[1])
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def tiff_pixel_sizes_um(page: Any) -> tuple[float | None, float | None]:
    """Convert TIFF X/Y resolution tags into micrometers per pixel."""
    x_tag = page.tags.get("XResolution")
    y_tag = page.tags.get("YResolution")
    unit_tag = page.tags.get("ResolutionUnit")
    if x_tag is None or y_tag is None or unit_tag is None:
        return None, None

    x_resolution = rational_as_float(x_tag.value)
    y_resolution = rational_as_float(y_tag.value)
    try:
        resolution_unit = int(unit_tag.value)
    except (TypeError, ValueError):
        return None, None

    # TIFF ResolutionUnit: 2 = inch, 3 = centimeter.
    micrometers_per_unit = {2: 25_400.0, 3: 10_000.0}.get(resolution_unit)
    if micrometers_per_unit is None:
        return None, None
    pixel_size_x = (
        micrometers_per_unit / x_resolution if x_resolution else None
    )
    pixel_size_y = (
        micrometers_per_unit / y_resolution if y_resolution else None
    )
    return pixel_size_x, pixel_size_y


def zarr_array_from_root(root: Any) -> Any:
    if hasattr(root, "shape"):
        return root
    arrays = []

    def visit(node: Any) -> None:
        if hasattr(node, "shape"):
            arrays.append(node)
        elif hasattr(node, "keys"):
            for key in node.keys():
                visit(node[key])

    visit(root)
    if not arrays:
        raise KeyError("No array found in the TIFF Zarr store")
    return max(arrays, key=lambda array: math.prod(sorted(array.shape)[-2:]))


def read_yx_region_from_zarr(
    array: Any,
    axes: str,
    x: int,
    y: int,
    width: int,
    height: int,
    np: Any,
) -> Any:
    """Read a Y/X region and return RGB regardless of TIFF axis ordering."""
    if len(axes) != array.ndim or "Y" not in axes or "X" not in axes:
        raise ValueError(
            f"Cannot match TIFF axes {axes!r} to Zarr shape {array.shape}"
        )

    selection = []
    retained_axes = []
    for axis, size in zip(axes, array.shape):
        if axis == "Y":
            selection.append(slice(y, y + height))
            retained_axes.append(axis)
        elif axis == "X":
            selection.append(slice(x, x + width))
            retained_axes.append(axis)
        elif axis in {"S", "C", "A"} and size in {3, 4}:
            selection.append(slice(None))
            retained_axes.append(axis)
        else:
            selection.append(0)

    image = np.asarray(array[tuple(selection)])
    y_axis = retained_axes.index("Y")
    x_axis = retained_axes.index("X")
    sample_axes = [
        retained_axes.index(axis)
        for axis in ("S", "C", "A")
        if axis in retained_axes
    ]
    if sample_axes:
        image = np.moveaxis(image, (y_axis, x_axis, sample_axes[0]), (0, 1, 2))
    else:
        image = np.moveaxis(image, (y_axis, x_axis), (0, 1))
    return normalize_rgb_tile(image, "rgb", np)


def extract_ndpi_with_tifffile(
    source: Path,
    original_source: Path,
    output_dir: Path,
    tile_size: int,
    compression: str,
    overwrite: bool,
    dry_run: bool,
    crop_to_bounds: bool,
    np: Any,
    tifffile: Any,
    tqdm: Any,
) -> list[dict[str, Any]]:
    """NDPI fallback for clusters where the OpenSlide binary is unavailable."""
    try:
        import zarr
    except ImportError as exc:
        raise ImportError(
            "The TIFF-based NDPI fallback requires zarr<3 and imagecodecs. "
            "Install requirements.txt and restart the kernel."
        ) from exc

    if crop_to_bounds:
        print(
            "The TIFF NDPI fallback cannot read OpenSlide non-empty bounds; "
            "preserving the complete level-0 canvas."
        )

    tif = tifffile.TiffFile(source)
    store = None
    try:
        levels = []
        for series in tif.series:
            for level in getattr(series, "levels", [series]):
                axes = level.axes
                shape = tuple(level.shape)
                if "Y" in axes and "X" in axes and len(shape) == len(axes):
                    area = shape[axes.index("Y")] * shape[axes.index("X")]
                    levels.append((area, level))
        if not levels:
            raise ValueError(f"No spatial image series found in NDPI: {source}")

        _, level = max(levels, key=lambda item: item[0])
        axes = level.axes
        store = level.aszarr()
        array = zarr_array_from_root(zarr.open(store, mode="r"))
        if len(axes) != array.ndim:
            raise ValueError(
                f"NDPI series axes {axes!r} do not match array shape {array.shape}"
            )
        width = int(array.shape[axes.index("X")])
        height = int(array.shape[axes.index("Y")])
        page = level.pages[0]
        pixel_size_x, pixel_size_y = tiff_pixel_sizes_um(page)

        # Cache one full-width output band. This prevents a strip-organized NDPI
        # from decoding the same very wide JPEG strip once for every X tile.
        maximum_band_bytes = 256 * 1024**2
        maximum_band_rows = max(
            16,
            (maximum_band_bytes // max(width * 3, 1) // 16) * 16,
        )
        effective_tile_size = min(tile_size, maximum_band_rows)
        if effective_tile_size != tile_size:
            print(
                f"Reducing NDPI tile size from {tile_size} to "
                f"{effective_tile_size} to keep the source band cache below 256 MB."
            )

        stem = strip_known_suffixes(original_source)
        output = output_dir / f"{stem}.ome.tif"
        print(
            f"NDPI highest-resolution TIFF series: {width:,} × {height:,} px "
            f"-> {output.name}"
        )
        record: dict[str, Any] = {
            "source": str(original_source),
            "decompressed_source": str(source),
            "format": "NDPI",
            "reader": "tifffile-zarr",
            "level": 0,
            "source_axes": axes,
            "source_bbox": {"x": 0, "y": 0, "width": width, "height": height},
            "width": width,
            "height": height,
            "physical_size_x_um": pixel_size_x,
            "physical_size_y_um": pixel_size_y,
            "output": str(output),
        }
        if dry_run:
            record["status"] = "planned"
            return [record]

        band_cache: dict[str, Any] = {
            "y": None,
            "height": None,
            "image": None,
        }

        def read_tile(x: int, y: int, tile_width: int, tile_height: int) -> Any:
            if band_cache["y"] != y or band_cache["height"] != tile_height:
                band_cache["image"] = read_yx_region_from_zarr(
                    array=array,
                    axes=axes,
                    x=0,
                    y=y,
                    width=width,
                    height=tile_height,
                    np=np,
                )
                band_cache["y"] = y
                band_cache["height"] = tile_height
            return np.ascontiguousarray(
                band_cache["image"][:, x : x + tile_width]
            )

        record["status"] = write_tiled_ome_tiff(
            output=output,
            width=width,
            height=height,
            tile_reader=read_tile,
            tile_size=effective_tile_size,
            compression=compression,
            physical_size_x_um=pixel_size_x,
            physical_size_y_um=pixel_size_y,
            description=(
                f"Native-resolution H&E extracted from {original_source.name}; "
                "largest NDPI TIFF series (highest resolution); complete canvas."
            ),
            overwrite=overwrite,
            np=np,
            tifffile=tifffile,
            tqdm=tqdm,
        )
        return [record]
    finally:
        if store is not None and hasattr(store, "close"):
            store.close()
        tif.close()


def extract_ndpi(
    source: Path,
    original_source: Path,
    output_dir: Path,
    tile_size: int,
    compression: str,
    overwrite: bool,
    dry_run: bool,
    crop_to_bounds: bool,
    np: Any,
    tifffile: Any,
    tqdm: Any,
) -> list[dict[str, Any]]:
    try:
        openslide, slide = open_ndpi(source)
    except ImportError as exc:
        print(f"OpenSlide unavailable ({exc}); using the TIFF/Zarr NDPI fallback.")
        return extract_ndpi_with_tifffile(
            source=source,
            original_source=original_source,
            output_dir=output_dir,
            tile_size=tile_size,
            compression=compression,
            overwrite=overwrite,
            dry_run=dry_run,
            crop_to_bounds=crop_to_bounds,
            np=np,
            tifffile=tifffile,
            tqdm=tqdm,
        )
    try:
        full_width, full_height = slide.dimensions
        x0 = 0
        y0 = 0
        width = full_width
        height = full_height

        if crop_to_bounds:
            properties = slide.properties
            try:
                x0 = int(properties[openslide.PROPERTY_NAME_BOUNDS_X])
                y0 = int(properties[openslide.PROPERTY_NAME_BOUNDS_Y])
                width = int(properties[openslide.PROPERTY_NAME_BOUNDS_WIDTH])
                height = int(properties[openslide.PROPERTY_NAME_BOUNDS_HEIGHT])
            except KeyError:
                print("NDPI has no non-empty bounds metadata; using the full level-0 canvas.")

        pixel_size_x = optional_float(
            slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        )
        pixel_size_y = optional_float(
            slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
        )
        stem = strip_known_suffixes(original_source)
        output = output_dir / f"{stem}.ome.tif"

        print(
            f"NDPI level 0: {width:,} × {height:,} px "
            f"(origin {x0:,}, {y0:,}) -> {output.name}"
        )
        record: dict[str, Any] = {
            "source": str(original_source),
            "decompressed_source": str(source),
            "format": "NDPI",
            "level": 0,
            "source_bbox": {"x": x0, "y": y0, "width": width, "height": height},
            "full_level_0_width": full_width,
            "full_level_0_height": full_height,
            "width": width,
            "height": height,
            "physical_size_x_um": pixel_size_x,
            "physical_size_y_um": pixel_size_y,
            "output": str(output),
        }
        if dry_run:
            record["status"] = "planned"
            return [record]

        from PIL import Image

        def read_tile(x: int, y: int, tile_width: int, tile_height: int) -> Any:
            rgba = slide.read_region(
                (x0 + x, y0 + y),
                0,
                (tile_width, tile_height),
            ).convert("RGBA")
            # OpenSlide uses transparent pixels outside the scanned area. Composite
            # them onto white so empty slide background does not become black.
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            rgb = Image.alpha_composite(white, rgba).convert("RGB")
            return np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))

        record["status"] = write_tiled_ome_tiff(
            output=output,
            width=width,
            height=height,
            tile_reader=read_tile,
            tile_size=tile_size,
            compression=compression,
            physical_size_x_um=pixel_size_x,
            physical_size_y_um=pixel_size_y,
            description=(
                f"Native-resolution H&E extracted from {original_source.name}; "
                f"NDPI pyramid level 0; source bounding box x={x0}, y={y0}, "
                f"width={width}, height={height}."
            ),
            overwrite=overwrite,
            np=np,
            tifffile=tifffile,
            tqdm=tqdm,
        )
        return [record]
    finally:
        slide.close()


def extract_file(
    input_path: Path,
    output_dir: Path | None = None,
    selected_scenes: list[int] | None = None,
    tile_size: int = 1024,
    compression: str = "deflate",
    overwrite: bool = False,
    dry_run: bool = False,
    crop_ndpi_to_bounds: bool = False,
    remove_decompressed: bool = False,
) -> list[dict[str, Any]]:
    np, tifffile, tqdm = require_common_dependencies()
    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if tile_size <= 0 or tile_size % 16:
        raise ValueError("--tile-size must be a positive multiple of 16")
    if compression not in {"none", "deflate"}:
        raise ValueError("compression must be 'none' or 'deflate'")

    kind = source_kind(input_path)
    if dry_run and input_path.suffix.lower() == ".gz":
        uncompressed_candidate = input_path.with_suffix("")
        if not uncompressed_candidate.exists():
            print(
                "Dry run still needs a decompressed slide to inspect dimensions; "
                "the source will be decompressed but no image output will be written."
            )

    source, created_uncompressed = ensure_uncompressed(input_path, tqdm)
    destination = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else input_path.parent / "full_resolution_he"
    )

    completed_successfully = False
    try:
        if kind == "czi":
            records = extract_czi(
                source=source,
                original_source=input_path,
                output_dir=destination,
                selected_scenes=selected_scenes,
                tile_size=tile_size,
                compression=compression,
                overwrite=overwrite,
                dry_run=dry_run,
                np=np,
                tifffile=tifffile,
                tqdm=tqdm,
            )
        else:
            if selected_scenes is not None:
                raise ValueError("--scene applies only to CZI files")
            records = extract_ndpi(
                source=source,
                original_source=input_path,
                output_dir=destination,
                tile_size=tile_size,
                compression=compression,
                overwrite=overwrite,
                dry_run=dry_run,
                crop_to_bounds=crop_ndpi_to_bounds,
                np=np,
                tifffile=tifffile,
                tqdm=tqdm,
            )

        if not dry_run:
            destination.mkdir(parents=True, exist_ok=True)
            manifest_path = destination / (
                f"{strip_known_suffixes(input_path)}_extraction_manifest.json"
            )
            temporary_manifest = manifest_path.with_name(
                manifest_path.name + ".partial"
            )
            with temporary_manifest.open("w", encoding="utf-8") as handle:
                json.dump(records, handle, indent=2)
                handle.write("\n")
            temporary_manifest.replace(manifest_path)
            print(f"Manifest: {manifest_path}")
        completed_successfully = True
        return records
    finally:
        if (
            remove_decompressed
            and created_uncompressed
            and completed_successfully
        ):
            source.unlink(missing_ok=True)
            print(f"Removed temporary decompressed source: {source}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract full-resolution H&E images from a CZI or NDPI file into "
            "streamed, tiled BigTIFF/OME-TIFF output."
        )
    )
    parser.add_argument("input", type=Path, help="Input .czi[.gz] or .ndpi[.gz]")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: INPUT_PARENT/full_resolution_he)",
    )
    parser.add_argument(
        "--scene",
        type=int,
        nargs="+",
        default=None,
        help="CZI scene index or indices to extract (default: all scenes)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Output tile width and height; must be a multiple of 16 (default: 1024)",
    )
    parser.add_argument(
        "--compression",
        choices=["deflate", "none"],
        default="deflate",
        help="Lossless output compression (default: deflate)",
    )
    parser.add_argument(
        "--crop-ndpi-to-bounds",
        action="store_true",
        help="Crop NDPI to vendor non-empty bounds when available; default preserves the full level-0 canvas",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect native dimensions and planned outputs without writing images",
    )
    parser.add_argument(
        "--remove-decompressed",
        action="store_true",
        help="Remove a source decompressed by this run after successful extraction",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        extract_file(
            input_path=args.input,
            output_dir=args.output_dir,
            selected_scenes=args.scene,
            tile_size=args.tile_size,
            compression=args.compression,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            crop_ndpi_to_bounds=args.crop_ndpi_to_bounds,
            remove_decompressed=args.remove_decompressed,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
