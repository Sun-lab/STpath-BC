"""Cache raw per-sample image metadata in ``dataset_manifest.json``.

The review notebooks need dataset facts such as H&E image dimensions and
physical pixel size. Those facts should come from OME XML metadata in the
downloaded/raw images when the full project data are available, then be cached
in the manifest so later notebook renders can show exactly what was read.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

def ome_physical_pixel_size(ome_xml: str | None) -> tuple[float | None, float | None]:
    if not ome_xml:
        return None, None
    try:
        root = ElementTree.fromstring(ome_xml)
        namespace = {"ome": root.tag.split("}")[0].strip("{")}
        pixels = root.find(".//ome:Pixels", namespace)
        if pixels is None:
            return None, None
        x = pixels.attrib.get("PhysicalSizeX")
        y = pixels.attrib.get("PhysicalSizeY")
        return (float(x) if x else None, float(y) if y else None)
    except (ElementTree.ParseError, TypeError, ValueError):
        return None, None


def get_hw(shape: tuple[int, ...]) -> tuple[int, int]:
    """Return height and width for common grayscale/RGB image array layouts."""
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    if len(shape) == 3 and shape[-1] in (3, 4):
        return int(shape[0]), int(shape[1])
    if len(shape) == 3 and shape[0] in (3, 4):
        return int(shape[1]), int(shape[2])
    raise ValueError(f"Unexpected image shape: {shape}")


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _single_pixel_size(x: Any, y: Any) -> float | None:
    x_value = _finite_float(x)
    y_value = _finite_float(y)
    values = [value for value in (x_value, y_value) if value is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def _project_relative(path: Path | None, project_root: Path) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except Exception:
        return str(path)


def _resolve_existing_path(value: Any, project_root: Path) -> Path | None:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return None
    path = Path(text).expanduser()
    candidates = [path if path.is_absolute() else project_root / path]
    if path.is_absolute():
        parts = path.parts
        if "Xenium" in parts:
            candidates.append(project_root.joinpath(*parts[parts.index("Xenium") + 1 :]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_experiment_pixel_size(path: Path | None) -> float | None:
    if path is None or not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            experiment = json.load(handle)
        return _finite_float(experiment.get("pixel_size"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def read_he_ome_metadata(path: Path) -> dict[str, float | int | None]:
    try:
        import tifffile
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Reading raw H&E OME-TIFF metadata requires tifffile in the notebook environment."
        ) from error
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        level = series.levels[0] if getattr(series, "levels", None) else series
        height, width = get_hw(tuple(level.shape))
        physical_x, physical_y = ome_physical_pixel_size(tif.ome_metadata)
    # Do not fall back to TIFF resolution tags, tile manifests, alignment files,
    # or any other generated values. Missing OME PhysicalSizeX/Y stays blank.
    return {
        "he_width_px": width,
        "he_height_px": height,
        "raw_he_um_per_px_x": physical_x,
        "raw_he_um_per_px_y": physical_y,
        "raw_he_um_per_px": _single_pixel_size(physical_x, physical_y),
    }


def _find_he_image(sample_dir: Path) -> Path | None:
    name = sample_dir.name
    candidates = [
        sample_dir / f"{name}_he_image.ome.tif",
        sample_dir / f"{name}_he_unaligned_image.ome.tif",
        sample_dir / f"{name}_he_image.ome.tiff",
        sample_dir / f"{name}_he_unaligned_image.ome.tiff",
    ]
    candidates.extend(sorted(sample_dir.glob("*he*image*.ome.tif")))
    candidates.extend(sorted(sample_dir.glob("*he*image*.ome.tiff")))
    return next((path for path in dict.fromkeys(candidates) if path.is_file()), None)


def _find_experiment(sample_dir: Path) -> Path | None:
    name = sample_dir.name
    candidates = [
        sample_dir / f"{name}_outs" / "experiment.xenium",
        sample_dir / f"{name}_xe_outs" / "experiment.xenium",
        sample_dir / "outs" / "experiment.xenium",
        sample_dir / "xe_outs" / "experiment.xenium",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(sample_dir.rglob("experiment.xenium")) if sample_dir.is_dir() else []
    return matches[0] if matches else None


def _metadata_record(
    *,
    dataset: str,
    sample: str,
    he_image: Path,
    he_metadata: dict[str, float | int | None],
    xenium_x: float | None,
    xenium_y: float | None,
    source_files: list[Path],
    project_root: Path,
    metadata_source: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "sample": sample,
        "he_image": _project_relative(he_image, project_root),
        "he_width_px": he_metadata["he_width_px"],
        "he_height_px": he_metadata["he_height_px"],
        "raw_he_um_per_px": he_metadata["raw_he_um_per_px"],
        "raw_he_um_per_px_x": he_metadata["raw_he_um_per_px_x"],
        "raw_he_um_per_px_y": he_metadata["raw_he_um_per_px_y"],
        "xenium_or_morphology_um_per_px": _single_pixel_size(xenium_x, xenium_y),
        "xenium_or_morphology_um_per_px_x": xenium_x,
        "xenium_or_morphology_um_per_px_y": xenium_y,
        "metadata_source": metadata_source,
        "source_files": [
            _project_relative(path, project_root)
            for path in dict.fromkeys(source_files)
            if path is not None
        ],
    }
    return {key: value for key, value in record.items() if value is not None}


def collect_standard_dataset_image_metadata(
    cfg: dict[str, Any],
    project_root: Path,
) -> list[dict[str, Any]]:
    root = project_root / cfg["root"]
    if not root.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for sample_dir in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        he_image = _find_he_image(sample_dir)
        if he_image is None:
            continue
        experiment_path = _find_experiment(sample_dir)
        try:
            he_metadata = read_he_ome_metadata(he_image)
        except Exception:
            continue
        xenium_pixel_size = _read_experiment_pixel_size(experiment_path)
        records.append(
            _metadata_record(
                dataset=cfg["name"],
                sample=sample_dir.name,
                he_image=he_image,
                he_metadata=he_metadata,
                xenium_x=xenium_pixel_size,
                xenium_y=xenium_pixel_size,
                source_files=[path for path in (he_image, experiment_path) if path is not None],
                project_root=project_root,
                metadata_source="raw H&E OME-TIFF metadata plus experiment.xenium",
            )
        )
    return sorted(records, key=lambda item: str(item["sample"]))


def collect_bilous_dataset_image_metadata(
    cfg: dict[str, Any],
    project_root: Path,
) -> list[dict[str, Any]]:
    output_root = project_root / cfg["output_dir"]
    input_manifest = output_root / "tile_input_manifest.csv"
    validation = output_root / "input_validation_summary.csv"
    rows = _read_csv_rows(input_manifest) or _read_csv_rows(validation)
    records: list[dict[str, Any]] = []
    for row in rows:
        sample = str(row.get("sample", "")).strip()
        if not sample:
            continue
        he_image = _resolve_existing_path(row.get("he_image") or row.get("he_path"), project_root)
        if he_image is None:
            continue
        try:
            he_metadata = read_he_ome_metadata(he_image)
        except Exception:
            continue
        xenium_x, xenium_y = None, None
        records.append(
            _metadata_record(
                dataset=cfg["name"],
                sample=sample,
                he_image=he_image,
                he_metadata=he_metadata,
                xenium_x=xenium_x,
                xenium_y=xenium_y,
                source_files=[he_image],
                project_root=project_root,
                metadata_source="raw Bilous H&E OME-TIFF metadata",
            )
        )
    return sorted(records, key=lambda item: str(item["sample"]))


def collect_dataset_image_metadata(
    cfg: dict[str, Any],
    project_root: Path,
) -> list[dict[str, Any]]:
    pipeline = cfg.get("pipeline")
    if pipeline == "standard_xenium":
        return collect_standard_dataset_image_metadata(cfg, project_root)
    if pipeline == "bilous_landmark":
        return collect_bilous_dataset_image_metadata(cfg, project_root)
    return []


def refresh_manifest_image_metadata(
    manifest_path: str | Path,
    *,
    project_root: str | Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    project_root = Path(project_root or manifest_path.parent).resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    changed = False
    sample_records = 0
    datasets_with_metadata = 0
    for cfg in manifest.get("datasets", []):
        records = collect_dataset_image_metadata(cfg, project_root)
        if not records:
            continue
        datasets_with_metadata += 1
        sample_records += len(records)
        if cfg.get("sample_image_metadata") != records:
            cfg["sample_image_metadata"] = records
            changed = True
    if changed:
        manifest["schema_version"] = max(int(manifest.get("schema_version", 1)), 2)
        if write:
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    return {
        "changed": changed,
        "written": bool(changed and write),
        "datasets_with_metadata": datasets_with_metadata,
        "sample_records": sample_records,
        "manifest_path": str(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read raw H&E image metadata and cache it in dataset_manifest.json.",
    )
    parser.add_argument(
        "--manifest",
        default="dataset_manifest.json",
        help="Path to dataset_manifest.json.",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root. Defaults to the manifest parent directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect available files and report what would be written without modifying the manifest.",
    )
    args = parser.parse_args()
    summary = refresh_manifest_image_metadata(
        args.manifest,
        project_root=args.project_root,
        write=not args.dry_run,
    )
    action = "would update" if args.dry_run and summary["changed"] else "updated" if summary["written"] else "no update"
    print(
        f"{action}: {summary['sample_records']} sample image metadata records "
        f"across {summary['datasets_with_metadata']} datasets"
    )


if __name__ == "__main__":
    main()
