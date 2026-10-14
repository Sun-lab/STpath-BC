#!/usr/bin/env python3
"""Batch-extract every GSE311609 breast H&E CZI/NDPI under a directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from xenium_pipeline.bilous_extract import extract_file, require_common_dependencies


SUPPORTED_ENDINGS = (".czi", ".czi.gz", ".ndpi", ".ndpi.gz")


def canonical_source_key(path: Path) -> str:
    resolved = str(path.resolve())
    return resolved[:-3] if resolved.lower().endswith(".gz") else resolved


def source_preference(path: Path) -> tuple[bool, bool]:
    """Prefer decompressed sources, then real files instead of sample symlinks."""
    return path.suffix.lower() != ".gz", not path.is_symlink()


def find_breast_he_files(root: Path) -> list[Path]:
    """Find each physical slide once.

    In the expected Bilous_2026 layout, ``images/`` is authoritative and the
    sample directories contain symlinks back to it. Restricting discovery to
    that directory prevents repeated extraction and ignores accidental slide
    copies in a mismatched sample directory. A recursive fallback supports
    layouts without a central image store.
    """
    central_images = root / "images"
    search_root = central_images if central_images.is_dir() else root
    candidates = [
        path
        for path in search_root.rglob("GSE311609_hne_breast_*")
        if path.is_file() and path.name.lower().endswith(SUPPORTED_ENDINGS)
    ]
    preferred: dict[str, Path] = {}
    for path in sorted(candidates):
        key = canonical_source_key(path)
        current = preferred.get(key)
        if current is None or source_preference(path) > source_preference(current):
            preferred[key] = path
    return sorted(preferred.values())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively find and extract all GSE311609 breast H&E CZI and "
            "NDPI files. Completed OME-TIFF outputs are skipped by default."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help="Directory to search recursively (default: current directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Central output directory; default creates full_resolution_he beside each source",
    )
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument(
        "--compression",
        choices=["deflate", "none"],
        default="deflate",
    )
    parser.add_argument("--crop-ndpi-to-bounds", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--remove-decompressed", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: search root is not a directory: {root}", file=sys.stderr)
        return 1

    files = find_breast_he_files(root)
    if not files:
        print(f"No GSE311609 breast H&E CZI/NDPI files found under {root}")
        return 0

    _, _, tqdm = require_common_dependencies()
    if (root / "images").is_dir():
        print(f"Using authoritative image store: {root / 'images'}")
    print(f"Found {len(files)} unique H&E source file(s):")
    for path in files:
        print(" -", path)

    completed = 0
    failed = 0
    all_records = []
    for source in tqdm(files, desc="H&E source files", unit="file"):
        print(f"\nProcessing: {source}")
        try:
            records = extract_file(
                input_path=source,
                output_dir=args.output_dir,
                tile_size=args.tile_size,
                compression=args.compression,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                crop_ndpi_to_bounds=args.crop_ndpi_to_bounds,
                remove_decompressed=args.remove_decompressed,
            )
            all_records.extend(records)
            completed += 1
        except Exception as exc:
            failed += 1
            print(f"FAILED: {source}\n  {type(exc).__name__}: {exc}", file=sys.stderr)

    written = sum(record.get("status") == "written" for record in all_records)
    skipped = sum(record.get("status") == "skipped" for record in all_records)
    planned = sum(record.get("status") == "planned" for record in all_records)
    print(
        "\nSummary:\n"
        f"  source files completed: {completed}\n"
        f"  source files failed:    {failed}\n"
        f"  images written:         {written}\n"
        f"  images skipped:         {skipped}\n"
        f"  images planned:         {planned}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
