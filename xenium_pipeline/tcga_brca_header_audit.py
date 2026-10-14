"""Match stored BRCA TIFF headers to IDC image levels without reading pixels."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from .tcga_brca_images import inspect_slide, write_csv


DICOMWEB = "https://proxy.imaging.datacommons.cancer.gov/current/viewer-only-no-downloads-see-tinyurl-dot-com-slash-3j3d9jyp/dicomWeb"


def match_header(header: dict, slide: dict) -> dict:
    """Require exact identity and dimensions; never choose a nearest pyramid level."""
    result = {
        "match_status": "no_dimension_match", "n_matching_instances": 0,
        "spacing_status": "unresolved", "matched_SOPInstanceUID": None,
        "inferred_tiff_mpp_x": None, "inferred_tiff_mpp_y": None,
        "conversion_provenance": "unverified", "error": "",
    }
    if header["slide_id"] != slide["slide_id"]:
        raise ValueError("TIFF slide ID differs from the requested slide")
    matches = [c for c in slide["candidate_instances"]
               if c["ContainerIdentifier"] == header["slide_id"]
               and "VOLUME" in c["ImageType"]
               and c["TotalPixelMatrixColumns"] == header["width_px"]
               and c["TotalPixelMatrixRows"] == header["height_px"]]
    result["n_matching_instances"] = len(matches)
    if header["orientation"] != 1:
        result["match_status"] = "orientation_review_required"
    elif len(matches) > 1:
        result["match_status"] = "ambiguous_dimension_match"
    elif matches:
        c = matches[0]
        result.update(
            match_status="matched_dimensions", spacing_status="index_row_only",
            matched_SOPInstanceUID=c["SOPInstanceUID"],
            matched_StudyInstanceUID=c["StudyInstanceUID"],
            matched_SeriesInstanceUID=c["SeriesInstanceUID"],
            idc_index_row_mpp=c["PixelSpacing_0"] * 1000,
            matched_image_type=c["ImageType"],
        )
    return result


def dicom_spacing(metadata: list, row: dict) -> tuple[float, float]:
    """Validate the original DICOM record and convert row/column mm to Y/X µm."""
    if len(metadata) != 1:
        raise ValueError("Expected one DICOM instance")
    d = metadata[0]

    def value(tag):
        return d[tag]["Value"][0]

    expected = {
        "00080018": row["matched_SOPInstanceUID"],
        "0020000D": row["matched_StudyInstanceUID"],
        "0020000E": row["matched_SeriesInstanceUID"],
        "00400512": row["slide_id"],
        "00480006": row["width_px"], "00480007": row["height_px"],
    }
    for tag, wanted in expected.items():
        if value(tag) != wanted:
            raise ValueError(f"DICOM identity/dimension mismatch in tag {tag}")
    if "VOLUME" not in d["00080008"]["Value"]:
        raise ValueError("Matched DICOM is not a volume image")
    measures = value("52009229")["00289110"]["Value"][0]
    spacing = measures["00280030"]["Value"]
    if len(spacing) != 2:
        raise ValueError("DICOM must contain row and column spacing")
    y, x = [float(v) * 1000 for v in spacing]
    if not all(math.isfinite(v) and v > 0 for v in (x, y)):
        raise ValueError("DICOM spacing must be positive and finite")
    return x, y


def fetch_dicom_metadata(row: dict, cache_dir: Path) -> tuple[list, str]:
    ids = [row[f"matched_{kind}InstanceUID"] for kind in ("Study", "Series", "SOP")]
    if not all(re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", uid) for uid in ids):
        raise ValueError("Invalid DICOM UID")
    study, series, instance = ids
    url = f"{DICOMWEB}/studies/{study}/series/{series}/instances/{instance}/metadata"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{instance}.json"
    if path.exists():
        metadata = json.loads(path.read_text())
    else:
        request = Request(url, headers={"Accept": "application/dicom+json"})
        with urlopen(request, timeout=15) as response:
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("DICOM metadata exceeds the 2 MB metadata-only limit")
        metadata = json.loads(raw)
        dicom_spacing(metadata, row)
        path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata, url


def run_header_audit(input_dir: Path, candidates_path: Path, output_dir: Path, *,
                     fetch_spacing: bool = True, metadata_cache: Path | None = None) -> dict:
    """Audit every requested TIFF; keep failures and unresolved scale explicit."""
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    candidates_path = Path(candidates_path).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"TIFF directory not accessible: {input_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("Audit outputs must be outside the source TIFF directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use a new empty output directory: {output_dir}")
    manifest = json.loads(candidates_path.read_text())
    slides = manifest["slides"]
    if not slides or len({s["slide_id"] for s in slides}) != len(slides):
        raise ValueError("Candidate manifest must contain unique, nonempty slide IDs")
    for slide in slides:
        name = slide["filename"]
        if Path(name).name != name or Path(name).stem != slide["slide_id"]:
            raise ValueError(f"Invalid input filename: {name}")
    metadata_cache = Path(metadata_cache or output_dir.parent / "dicom_metadata_cache").resolve()
    if metadata_cache == input_dir or input_dir in metadata_cache.parents:
        raise ValueError("Metadata cache must be outside the source TIFF directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, slide in enumerate(slides, 1):
        path = input_dir / slide["filename"]
        row = {"slide_id": slide["slide_id"], "filename": slide["filename"],
               "patient_id": slide["patient_id"], "source_path": str(path),
               "match_status": "header_error", "spacing_status": "unresolved",
               "inferred_tiff_mpp_x": None, "inferred_tiff_mpp_y": None,
               "conversion_provenance": "unverified", "error": ""}
        try:
            row.update(inspect_slide(path))
            row.update(match_header(row, slide))
        except FileNotFoundError as exc:
            row.update(match_status="missing_file", error=str(exc))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if i % 25 == 0 or i == len(slides):
            print(f"Headers: {i}/{len(slides)}", flush=True)

    # Save the complete header scan before any network-dependent work.
    def save_rows():
        write_csv(output_dir / "tiff_resolution_audit.csv", rows)
        (output_dir / "tiff_resolution_audit.json").write_text(json.dumps(rows, indent=2) + "\n")

    save_rows()
    consecutive_failures = 0
    for i, row in enumerate(rows, 1):
        if row["match_status"] != "matched_dimensions" or not fetch_spacing:
            continue
        if consecutive_failures >= 3:
            row.update(spacing_status="metadata_fetch_skipped", error="Stopped after three consecutive metadata failures; header match preserved")
            continue
        try:
            metadata, url = fetch_dicom_metadata(row, metadata_cache)
            x, y = dicom_spacing(metadata, row)
            row.update(dicom_mpp_x=x, dicom_mpp_y=y, dicom_metadata_url=url)
            if any(row.get(f"mpp_{axis}") is not None and
                   not math.isclose(row[f"mpp_{axis}"], value, rel_tol=0.001)
                   for axis, value in (("x", x), ("y", y))):
                row["spacing_status"] = "header_dicom_spacing_conflict"
                row["error"] = "Header and DICOM spacing differ by more than 0.1%; review provenance"
            else:
                row.update(spacing_status="dicom_xy_verified_conversion_pending",
                           inferred_tiff_mpp_x=x, inferred_tiff_mpp_y=y,
                           inferred_image_width_mm=row["width_px"]*x/1000,
                           inferred_image_height_mm=row["height_px"]*y/1000,
                           crop_256_width_um=256*x, crop_256_height_um=256*y)
            consecutive_failures = 0
        except Exception as exc:
            consecutive_failures += 1
            row.update(spacing_status="metadata_fetch_error", error=f"{type(exc).__name__}: {exc}")
            print(f"Metadata lookup failed for {row['slide_id']}: {exc}", flush=True)
        if i % 25 == 0:
            save_rows()
            print(f"Resolution lookup: {i}/{len(rows)}", flush=True)
    save_rows()
    expected_names = {s["filename"] for s in slides}
    extra = sorted(p.name for p in input_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in {".tif", ".tiff"} and p.name not in expected_names)
    header_complete = all(r["match_status"] == "matched_dimensions" for r in rows)
    spacing_complete = all(r["spacing_status"] == "dicom_xy_verified_conversion_pending" for r in rows)
    summary = {
        "status": "complete" if header_complete and (spacing_complete or not fetch_spacing) else "partial",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir), "output_dir": str(output_dir),
        "candidates_path": str(candidates_path),
        "candidates_sha256": hashlib.sha256(candidates_path.read_bytes()).hexdigest(),
        "source_versions": manifest.get("source_versions"),
        "n_requested_slides": len(rows), "n_individuals": len({r['patient_id'] for r in rows}),
        "match_counts": dict(Counter(r["match_status"] for r in rows)),
        "spacing_counts": dict(Counter(r["spacing_status"] for r in rows)),
        "fetch_spacing": fetch_spacing, "metadata_cache": str(metadata_cache),
        "unlisted_tiff_files": extra,
        "interpretation": "Header and public metadata audit only. No image pixels read or TIFFs changed. Exact ID/dimension matches support inferred TIFF spacing; conversion provenance remains unverified. Physical image dimensions describe the image canvas, not tissue area. Crop-size selection and inference are separate steps.",
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Audit {summary['status']}: {output_dir}", flush=True)
    return summary
