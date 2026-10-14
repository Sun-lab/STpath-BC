"""Gene-panel and marker-group exploration helpers for Xenium datasets."""

from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def decode_text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def natural_key(value: object) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    )


def is_gene_expression(feature_type: object, feature_name: object) -> bool:
    feature_type_text = decode_text(feature_type).strip().lower()
    feature_name_text = decode_text(feature_name).strip().lower()
    # Xenium exports call unused decoding features "Blank Codeword".
    controls = ("control", "unassigned", "deprecated", "blank", "codeword")
    if any(
        token in feature_type_text or token in feature_name_text
        for token in controls
    ):
        return False
    if feature_type_text in {"", "none", "nan"}:
        return True  # Older exports omit feature types; still reject named controls.
    return re.sub(r"[\s_-]+", "", feature_type_text) == "geneexpression"


def load_gene_groups(path: str | Path) -> pd.DataFrame:
    """Read marker-gene group CSVs into ``gene, group, grouping_column`` rows."""
    path = Path(path)
    raw = pd.read_csv(path, header=None, dtype=str, comment="#")
    raw = raw.dropna(how="all").reset_index(drop=True)
    if raw.empty or raw.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in {path}")

    first = [str(value).strip().lower() for value in raw.iloc[0].fillna("")]
    has_header = first[0] == "gene" and first[1] == "group"
    if has_header:
        headers = [
            str(value).strip() or f"group_{index}"
            for index, value in enumerate(raw.iloc[0])
        ]
        table = raw.iloc[1:].copy()
        table.columns = headers
        gene_column = headers[0]
        group_columns = headers[1:]
    else:
        table = raw.copy()
        gene_column = "gene"
        group_columns = [
            "group" if index == 1 else f"group_{index}"
            for index in range(1, raw.shape[1])
        ]
        table.columns = [gene_column, *group_columns]

    pieces: list[pd.DataFrame] = []
    for group_column in group_columns:
        piece = table[[gene_column, group_column]].rename(
            columns={gene_column: "gene", group_column: "group"}
        )
        piece["grouping_column"] = group_column
        pieces.append(piece)
    groups = pd.concat(pieces, ignore_index=True)
    groups["gene"] = groups["gene"].astype("string").str.strip()
    groups["group"] = groups["group"].astype("string").str.strip()
    groups = groups.dropna(subset=["gene", "group"])
    groups = groups.loc[(groups["gene"] != "") & (groups["group"] != "")]
    groups["gene_upper"] = groups["gene"].str.upper()
    return groups.drop_duplicates().reset_index(drop=True)


def read_gene_panel_json(path: str | Path) -> pd.DataFrame:
    """Parse a 10x ``gene_panel.json`` file into one row per target."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload = payload.get("payload", payload) if isinstance(payload, dict) else {}
    targets = payload.get("targets", []) if isinstance(payload, dict) else []
    rows: list[dict[str, object]] = []
    for target in targets:
        target_type = target.get("type", {}) if isinstance(target, dict) else {}
        data = target_type.get("data", {}) if isinstance(target_type, dict) else {}
        source = target.get("source", {}) if isinstance(target, dict) else {}
        identity = source.get("identity", {}) if isinstance(source, dict) else {}
        info = target.get("info", {}) if isinstance(target, dict) else {}
        gene = data.get("name") or data.get("gene") or data.get("id")
        rows.append(
            {
                "path": str(path),
                "target_descriptor": target_type.get("descriptor", ""),
                "gene": str(gene) if gene is not None else "",
                "gene_upper": str(gene).upper() if gene is not None else "",
                "gene_id": data.get("id", ""),
                "source_category": source.get("category", ""),
                "source_name": identity.get("name", ""),
                "source_version": identity.get("version", ""),
                "gene_coverage": info.get("gene_coverage", pd.NA),
                "n_codewords": len(target.get("codewords", [])),
            }
        )
    return pd.DataFrame(rows)


def zarr_feature_metadata(path: str | Path) -> pd.DataFrame:
    """Return feature metadata from ``cell_feature_matrix.zarr.zip``."""
    try:
        import zarr
        from zarr.storage import ZipStore
    except ModuleNotFoundError as error:  # pragma: no cover - environment-specific.
        raise ModuleNotFoundError(
            "Reading cell_feature_matrix.zarr.zip requires zarr"
        ) from error

    path = Path(path)
    with ZipStore(str(path), mode="r") as store:
        root = zarr.open_group(store=store, mode="r")
        matrix = root["cell_features"] if "cell_features" in root else root
        number_features = len(matrix["indptr"]) - 1
        attrs = dict(root.attrs)
        attrs.update(dict(matrix.attrs))
        names_raw = attrs.get("feature_keys") or attrs.get("feature_names")
        if names_raw is None:
            raise KeyError(f"No feature_keys metadata in {path}")
        names = [decode_text(value) for value in names_raw]
        types_raw = attrs.get("feature_types")
        if types_raw is None or len(types_raw) != len(names):
            types = ["Gene Expression"] * len(names)
        else:
            types = [decode_text(value) for value in types_raw]
    if len(names) != number_features:
        raise ValueError(
            f"{path} reports {number_features} features but metadata has {len(names)} names"
        )
    return pd.DataFrame(
        {
            "path": str(path),
            "gene": names,
            "gene_upper": [name.upper() for name in names],
            "feature_type": types,
            "is_gene_expression": [
                is_gene_expression(feature_type, name)
                for name, feature_type in zip(names, types)
            ],
        }
    )


def h5_feature_metadata(path: str | Path) -> pd.DataFrame:
    """Return feature metadata from a 10x-style ``cell_feature_matrix.h5``."""
    try:
        import h5py
    except ModuleNotFoundError as error:  # pragma: no cover - environment-specific.
        raise ModuleNotFoundError("Reading cell_feature_matrix.h5 requires h5py") from error

    path = Path(path)
    with h5py.File(path, "r") as handle:
        group = handle["matrix"]["features"]
        names_key = "name" if "name" in group else "id"
        names = [decode_text(value) for value in group[names_key][:]]
        if "feature_type" in group:
            types = [decode_text(value) for value in group["feature_type"][:]]
        else:
            types = ["Gene Expression"] * len(names)
    return pd.DataFrame(
        {
            "path": str(path),
            "gene": names,
            "gene_upper": [name.upper() for name in names],
            "feature_type": types,
            "is_gene_expression": [
                is_gene_expression(feature_type, name)
                for name, feature_type in zip(names, types)
            ],
        }
    )


def discover_sample_dirs(root: str | Path, sample_names: Iterable[str] | None = None) -> list[Path]:
    root = Path(root).expanduser().resolve()
    requested = set(sample_names or [])
    if requested:
        return [root / sample for sample in sorted(requested, key=natural_key)]
    ignored = {
        "__pycache__",
        "he_tiles",
        "manual_alignment_reviews",
        "manual_alignment_from_mapping",
        "mapping_previews",
        "mapped_side_by_side",
        "tile_outputs",
    }
    return [
        path
        for path in sorted(root.iterdir(), key=lambda value: natural_key(value.name))
        if path.is_dir() and not path.name.startswith(".") and path.name not in ignored
    ]


def sample_relevant_files(sample_dir: Path, root: Path) -> dict[str, list[Path]]:
    panel_jsons = sorted(sample_dir.rglob("gene_panel.json"))
    panel_jsons.extend(sorted(sample_dir.rglob("*_gene_panel.json")))
    panel_jsons = list(dict.fromkeys(panel_jsons))

    marker_groups = sorted(sample_dir.glob("*gene_groups.csv"))
    marker_groups.extend(sorted(sample_dir.glob("*Input_gene_groups.csv")))
    root_marker_groups = sorted(root.glob("*gene_groups.csv")) + sorted(
        root.glob("*Input_gene_groups.csv")
    )
    root_marker_groups = list(dict.fromkeys(root_marker_groups))
    if not marker_groups and len(root_marker_groups) == 1:
        marker_groups.extend(root_marker_groups)
    marker_groups = list(dict.fromkeys(marker_groups))

    matrices = sorted(sample_dir.rglob("cell_feature_matrix.zarr.zip"))
    matrices.extend(sorted(sample_dir.rglob("*cell_feature_matrix.h5")))
    matrices.extend(sorted(sample_dir.rglob("cell_feature_matrix.h5")))
    matrices = list(dict.fromkeys(matrices))

    return {
        "panel_jsons": panel_jsons,
        "marker_groups": marker_groups,
        "feature_matrices": matrices,
    }


def feature_metadata(path: Path) -> tuple[pd.DataFrame, str]:
    try:
        if path.name.endswith(".zarr.zip"):
            return zarr_feature_metadata(path), ""
        if path.suffix == ".h5":
            return h5_feature_metadata(path), ""
        return pd.DataFrame(), f"Unsupported matrix format: {path.name}"
    except Exception as error:
        return pd.DataFrame(), f"{type(error).__name__}: {error}"


def explore_gene_panels(
    root: str | Path,
    sample_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Explore panel sources, feature matrices, and marker groups by sample."""
    root = Path(root).expanduser().resolve()
    summary_rows: list[dict[str, object]] = []
    gene_rows: list[pd.DataFrame] = []
    group_coverage_rows: list[dict[str, object]] = []
    sample_gene_sets: dict[str, set[str]] = {}

    for sample_dir in discover_sample_dirs(root, sample_names):
        if not sample_dir.is_dir():
            summary_rows.append(
                {
                    "sample": sample_dir.name,
                    "sample_dir": str(sample_dir),
                    "status": "missing_sample_dir",
                    "read_errors": "sample directory does not exist",
                }
            )
            continue

        files = sample_relevant_files(sample_dir, root)
        read_errors: list[str] = []
        sample_genes: list[pd.DataFrame] = []
        panel_names: set[str] = set()

        for panel_path in files["panel_jsons"]:
            try:
                panel = read_gene_panel_json(panel_path)
                if not panel.empty:
                    panel["sample"] = sample_dir.name
                    panel["source_kind"] = "gene_panel_json"
                    sample_genes.append(panel)
                    panel_names.update(
                        panel["source_name"].dropna().astype(str).loc[lambda s: s != ""]
                    )
            except Exception as error:
                read_errors.append(f"{panel_path.name}: {type(error).__name__}: {error}")

        feature_tables: list[pd.DataFrame] = []
        for matrix_path in files["feature_matrices"]:
            table, error = feature_metadata(matrix_path)
            if error:
                read_errors.append(f"{matrix_path.name}: {error}")
                continue
            if not table.empty:
                table["sample"] = sample_dir.name
                table["source_kind"] = "feature_matrix"
                feature_tables.append(table)
                sample_genes.append(table)

        group_tables: list[pd.DataFrame] = []
        for group_path in files["marker_groups"]:
            try:
                groups = load_gene_groups(group_path)
                groups["path"] = str(group_path)
                groups["sample"] = sample_dir.name
                groups["source_kind"] = "marker_group_csv"
                group_tables.append(groups)
                sample_genes.append(groups)
            except Exception as error:
                read_errors.append(f"{group_path.name}: {type(error).__name__}: {error}")

        genes = (
            pd.concat(sample_genes, ignore_index=True, sort=False)
            if sample_genes
            else pd.DataFrame()
        )
        if not genes.empty:
            gene_rows.append(genes)
            assay_genes = set(
                genes.loc[
                    genes["source_kind"].isin(["gene_panel_json", "feature_matrix"]),
                    "gene_upper",
                ]
                .dropna()
                .astype(str)
            )
            if not assay_genes:
                assay_genes = set(genes["gene_upper"].dropna().astype(str))
            sample_gene_sets[sample_dir.name] = {gene for gene in assay_genes if gene}
        else:
            sample_gene_sets[sample_dir.name] = set()

        assay_gene_set = sample_gene_sets[sample_dir.name]
        for groups in group_tables:
            for group_name, group in groups.groupby("group", sort=True):
                listed = set(group["gene_upper"].dropna().astype(str))
                found = listed & assay_gene_set if assay_gene_set else set()
                group_coverage_rows.append(
                    {
                        "sample": sample_dir.name,
                        "group": group_name,
                        "grouping_column": ", ".join(
                            sorted(group["grouping_column"].astype(str).unique())
                        ),
                        "n_markers_listed": len(listed),
                        "n_markers_in_assay": len(found),
                        "marker_coverage_fraction": (
                            len(found) / len(listed) if listed else np.nan
                        ),
                        "markers_missing": ", ".join(sorted(listed - found)),
                    }
                )

        panel_gene_count = 0
        panel_target_count = 0
        matrix_feature_count = 0
        matrix_gene_count = 0
        if not genes.empty:
            panel_rows = genes.loc[genes["source_kind"] == "gene_panel_json"]
            matrix_rows = genes.loc[genes["source_kind"] == "feature_matrix"]
            panel_target_count = len(panel_rows)
            if not panel_rows.empty and "target_descriptor" in panel_rows.columns:
                panel_gene_count = int(
                    panel_rows.loc[panel_rows["target_descriptor"].eq("gene"), "gene_upper"]
                    .replace("", pd.NA)
                    .dropna()
                    .nunique()
                )
            matrix_feature_count = len(matrix_rows)
            if "is_gene_expression" in matrix_rows:
                matrix_gene_count = int(
                    matrix_rows.loc[
                        matrix_rows["is_gene_expression"].fillna(False),
                        "gene_upper",
                    ]
                    .replace("", pd.NA)
                    .dropna()
                    .nunique()
                )
            else:
                matrix_gene_count = int(
                    matrix_rows["gene_upper"].replace("", pd.NA).dropna().nunique()
                )

        marker_group_genes = (
            int(pd.concat(group_tables, ignore_index=True)["gene_upper"].nunique())
            if group_tables
            else 0
        )
        marker_group_count = (
            int(pd.concat(group_tables, ignore_index=True)["group"].nunique())
            if group_tables
            else 0
        )
        any_relevant_files = any(files.values())
        if sample_genes:
            status = "ok"
        elif any_relevant_files:
            status = "files_found_but_not_readable"
        else:
            status = "no_panel_or_matrix_found"
        summary_rows.append(
            {
                "sample": sample_dir.name,
                "sample_dir": str(sample_dir),
                "status": status,
                "n_gene_panel_jsons": len(files["panel_jsons"]),
                "n_panel_targets": panel_target_count,
                "n_panel_genes": panel_gene_count,
                "panel_names": "; ".join(sorted(panel_names)),
                "n_feature_matrices": len(files["feature_matrices"]),
                "n_matrix_features": matrix_feature_count,
                "n_matrix_gene_features": matrix_gene_count,
                "n_marker_group_files": len(files["marker_groups"]),
                "n_marker_group_genes": marker_group_genes,
                "n_marker_groups": marker_group_count,
                "n_assay_or_marker_genes_union": len(sample_gene_sets[sample_dir.name]),
                "read_errors": "; ".join(read_errors),
            }
        )

    summary = pd.DataFrame(summary_rows)
    genes = (
        pd.concat(gene_rows, ignore_index=True, sort=False)
        if gene_rows
        else pd.DataFrame(
            columns=[
                "sample",
                "source_kind",
                "path",
                "gene",
                "gene_upper",
                "target_descriptor",
                "gene_id",
                "feature_type",
                "group",
                "grouping_column",
            ]
        )
    )
    overlap_rows: list[dict[str, object]] = []
    for sample_a, sample_b in combinations(sorted(sample_gene_sets, key=natural_key), 2):
        genes_a = sample_gene_sets[sample_a]
        genes_b = sample_gene_sets[sample_b]
        union = genes_a | genes_b
        intersection = genes_a & genes_b
        overlap_rows.append(
            {
                "sample_a": sample_a,
                "sample_b": sample_b,
                "n_genes_a": len(genes_a),
                "n_genes_b": len(genes_b),
                "n_overlap": len(intersection),
                "n_union": len(union),
                "jaccard": len(intersection) / len(union) if union else np.nan,
                "genes_only_a": ", ".join(sorted(genes_a - genes_b)),
                "genes_only_b": ", ".join(sorted(genes_b - genes_a)),
            }
        )
    overlap = pd.DataFrame(
        overlap_rows,
        columns=[
            "sample_a",
            "sample_b",
            "n_genes_a",
            "n_genes_b",
            "n_overlap",
            "n_union",
            "jaccard",
            "genes_only_a",
            "genes_only_b",
        ],
    )
    group_coverage = pd.DataFrame(
        group_coverage_rows,
        columns=[
            "sample",
            "group",
            "grouping_column",
            "n_markers_listed",
            "n_markers_in_assay",
            "marker_coverage_fraction",
            "markers_missing",
        ],
    )
    return summary, genes, overlap, group_coverage


def write_gene_panel_exploration(
    root: str | Path,
    sample_names: list[str] | None = None,
    *,
    output_prefix: str = "gene_panel",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run panel exploration and save CSV artifacts under ``root``."""
    root = Path(root).expanduser().resolve()
    summary, genes, overlap, group_coverage = explore_gene_panels(root, sample_names)
    summary.to_csv(root / f"{output_prefix}_summary.csv", index=False)
    genes.to_csv(root / f"{output_prefix}_genes.csv", index=False)
    overlap.to_csv(root / f"{output_prefix}_pairwise_overlap.csv", index=False)
    group_coverage.to_csv(root / f"{output_prefix}_marker_group_coverage.csv", index=False)
    return summary, genes, overlap, group_coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore Xenium gene panels by sample.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--samples", nargs="+")
    parser.add_argument("--output-prefix", default="gene_panel")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, _, overlap, group_coverage = write_gene_panel_exploration(
        args.root,
        args.samples,
        output_prefix=args.output_prefix,
    )
    print(summary)
    if not overlap.empty:
        print("\nPairwise overlap:")
        print(overlap[["sample_a", "sample_b", "n_overlap", "n_union", "jaccard"]])
    if not group_coverage.empty:
        print("\nMarker-group coverage:")
        print(
            group_coverage[
                [
                    "sample",
                    "group",
                    "n_markers_listed",
                    "n_markers_in_assay",
                    "marker_coverage_fraction",
                ]
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
