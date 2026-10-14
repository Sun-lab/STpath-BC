"""Expression-based marker scoring and cell annotation for Xenium samples."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - minimal local runtimes.
    def tqdm(iterable=None, *args, **kwargs):  # type: ignore[no-redef]
        return iterable if iterable is not None else ()

try:
    from .gene_panel_utils import (
        discover_sample_dirs,
        is_gene_expression,
        load_gene_groups,
        natural_key,
        sample_relevant_files,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from xenium_pipeline.gene_panel_utils import (  # type: ignore[no-redef]
        discover_sample_dirs,
        is_gene_expression,
        load_gene_groups,
        natural_key,
        sample_relevant_files,
    )


DEFAULT_BREAST_MARKERS: dict[str, list[str]] = {
    "Tumor_or_epithelial": [
        "EPCAM",
        "KRT8",
        "KRT18",
        "KRT19",
        "KRT7",
        "TACSTD2",
        "MUC1",
        "ERBB2",
        "ESR1",
        "PGR",
    ],
    "Basal_or_myoepithelial": ["KRT5", "KRT14", "KRT15", "TP63", "MYLK", "ACTA2"],
    "Fibroblast_or_stromal": [
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "DCN",
        "LUM",
        "PDGFRA",
        "FAP",
        "POSTN",
        "THY1",
    ],
    "Immune": [
        "PTPRC",
        "CD3D",
        "CD3E",
        "CD2",
        "CD79A",
        "MS4A1",
        "CD68",
        "LYZ",
        "LST1",
        "NKG7",
        "GNLY",
        "MZB1",
    ],
    "Endothelial": ["PECAM1", "VWF", "KDR", "FLT1", "CDH5", "ENG", "CLDN5", "EMCN"],
    "Pericyte_or_smooth_muscle": [
        "RGS5",
        "PDGFRB",
        "CSPG4",
        "MCAM",
        "NOTCH3",
        "TAGLN",
        "MYH11",
        "MYL9",
    ],
    "Adipocyte": ["ADIPOQ", "FABP4", "PLIN1", "LPL", "LEP"],
}


@dataclass(frozen=True)
class AnnotationConfig:
    marker_sets: dict[str, list[str]]
    normalization_target: float = 10_000.0
    min_score: float = 0.25
    min_margin: float = 0.10
    min_markers: int = 1
    uncertain_label: str = "Uncertain"
    overwrite: bool = False
    export_cell_groups: bool = False
    use_sample_marker_groups: bool = True


def decode_text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def xenium_int_cell_id_to_str(cell_id_array: np.ndarray) -> list[str]:
    cell_ids = np.asarray(cell_id_array)
    if cell_ids.ndim == 1:
        prefixes = cell_ids
        suffixes = np.ones(len(cell_ids), dtype=np.uint32)
    elif cell_ids.ndim == 2 and cell_ids.shape[1] == 2:
        prefixes = cell_ids[:, 0]
        suffixes = cell_ids[:, 1]
    elif cell_ids.ndim == 2 and cell_ids.shape[0] == 2:
        prefixes = cell_ids[0]
        suffixes = cell_ids[1]
    else:
        raise ValueError(f"Unsupported cells.zarr cell_id shape: {cell_ids.shape}")

    output: list[str] = []
    for prefix, suffix in zip(prefixes, suffixes):
        hexadecimal = f"{int(prefix):08x}"
        letters = "".join(chr(ord("a") + int(character, 16)) for character in hexadecimal)
        output.append(f"{letters}-{int(suffix)}")
    return output


def load_marker_sets(
    *,
    markers_json: str | Path | None = None,
    gene_groups_path: str | Path | None = None,
) -> dict[str, list[str]]:
    if markers_json is not None:
        payload = json.loads(Path(markers_json).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Marker JSON must map labels to marker-gene lists")
        return {
            str(label): sorted({str(gene).upper() for gene in genes})
            for label, genes in payload.items()
        }
    if gene_groups_path is not None:
        groups = load_gene_groups(gene_groups_path)
        return {
            str(group): sorted(set(table["gene_upper"].astype(str)))
            for group, table in groups.groupby("group", sort=True)
        }
    return {label: [gene.upper() for gene in genes] for label, genes in DEFAULT_BREAST_MARKERS.items()}


def find_feature_matrix(sample_dir: Path) -> Path | None:
    candidates = sorted(sample_dir.rglob("cell_feature_matrix.zarr.zip"))
    candidates.extend(sorted(sample_dir.rglob("*cell_feature_matrix.h5")))
    candidates.extend(sorted(sample_dir.rglob("cell_feature_matrix.h5")))
    return candidates[0] if candidates else None


def find_cells_csv(sample_dir: Path) -> Path | None:
    patterns = [
        "*_cells_with_annotations.csv",
        "*_cells_aligned_to_he.csv",
        "*_cells_with_cluster_labels.csv",
        "*cells.csv",
        "*cells.csv.gz",
    ]
    for pattern in patterns:
        matches = sorted(sample_dir.rglob(pattern))
        if matches:
            return matches[0]
    return None


def read_optional_cells_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    table = pd.read_csv(path)
    if "barcode" in table.columns and "cell_id" not in table.columns:
        table = table.rename(columns={"barcode": "cell_id"})
    if "cell_id" not in table.columns:
        return None
    table["cell_id"] = table["cell_id"].astype(str)
    return table


def cell_ids_from_cells_zarr(sample_dir: Path) -> list[str] | None:
    matches = sorted(sample_dir.rglob("cells.zarr.zip"))
    if not matches:
        return None
    try:
        import zarr
        from zarr.storage import ZipStore
    except ModuleNotFoundError as error:  # pragma: no cover - environment-specific.
        raise ModuleNotFoundError("Reading cells.zarr.zip requires zarr") from error

    with ZipStore(str(matches[0]), mode="r") as store:
        root = zarr.open_group(store=store, mode="r")
        if "cell_id" not in root:
            return None
        return xenium_int_cell_id_to_str(root["cell_id"][:])


def zarr_feature_metadata(matrix: object, root: object, number_features: int) -> tuple[list[str], list[str]]:
    attrs = dict(root.attrs)
    attrs.update(dict(matrix.attrs))
    names_raw = attrs.get("feature_keys") or attrs.get("feature_names")
    if names_raw is None:
        raise KeyError("cell_feature_matrix.zarr.zip has no feature_keys metadata")
    names = [decode_text(value) for value in names_raw]
    if len(names) != number_features:
        raise ValueError(
            f"Feature metadata has {len(names)} names but matrix has {number_features} rows"
        )
    types_raw = attrs.get("feature_types")
    if types_raw is None or len(types_raw) != len(names):
        types = ["Gene Expression"] * number_features
    else:
        types = [decode_text(value) for value in types_raw]
    return names, types


def score_zarr_marker_sets(
    matrix_path: Path,
    cell_ids: list[str],
    marker_sets: dict[str, list[str]],
    normalization_target: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        import zarr
        from zarr.storage import ZipStore
    except ModuleNotFoundError as error:  # pragma: no cover - environment-specific.
        raise ModuleNotFoundError("Scoring cell_feature_matrix.zarr.zip requires zarr") from error

    with ZipStore(str(matrix_path), mode="r") as store:
        root = zarr.open_group(store=store, mode="r")
        matrix = root["cell_features"] if "cell_features" in root else root
        data = matrix["data"]
        indices = matrix["indices"]
        indptr = np.asarray(matrix["indptr"][:], dtype=np.int64)
        number_features = len(indptr) - 1
        feature_names, feature_types = zarr_feature_metadata(matrix, root, number_features)
        matrix_number_cells = int(
            matrix.attrs.get("number_cells", root.attrs.get("number_cells", len(cell_ids)))
        )
        if matrix_number_cells != len(cell_ids):
            raise ValueError(
                f"Cell-id table has {len(cell_ids):,} cells but matrix reports "
                f"{matrix_number_cells:,}; row order cannot be assumed"
            )

        gene_features = [
            index
            for index, (name, feature_type) in enumerate(zip(feature_names, feature_types))
            if is_gene_expression(feature_type, name)
        ]
        totals = np.zeros(len(cell_ids), dtype=np.float64)
        for feature_index in tqdm(gene_features, desc="Computing cell totals", unit="gene"):
            start, end = int(indptr[feature_index]), int(indptr[feature_index + 1])
            if end <= start:
                continue
            cell_indices = np.asarray(indices[start:end], dtype=np.int64)
            values = np.asarray(data[start:end], dtype=np.float64)
            np.add.at(totals, cell_indices, values)
        totals[totals == 0] = 1.0

        feature_lookup: dict[str, list[int]] = {}
        for feature_index in gene_features:
            feature_lookup.setdefault(feature_names[feature_index].upper(), []).append(feature_index)

        labels = sorted(marker_sets, key=natural_key)
        scores_array = np.zeros((len(cell_ids), len(labels)), dtype=np.float32)
        detected_array = np.zeros((len(cell_ids), len(labels)), dtype=np.uint16)
        coverage_rows: list[dict[str, object]] = []
        for label_index, label in enumerate(tqdm(labels, desc="Scoring marker sets", unit="set")):
            requested = sorted({gene.upper() for gene in marker_sets[label]})
            found = [gene for gene in requested if gene in feature_lookup]
            group_score = np.zeros(len(cell_ids), dtype=np.float32)
            group_detected = np.zeros(len(cell_ids), dtype=np.uint16)
            for gene in found:
                detected_cells: list[np.ndarray] = []
                for feature_index in feature_lookup[gene]:
                    start, end = int(indptr[feature_index]), int(indptr[feature_index + 1])
                    if end <= start:
                        continue
                    cell_indices = np.asarray(indices[start:end], dtype=np.int64)
                    values = np.asarray(data[start:end], dtype=np.float64)
                    normalized = np.log1p(values * normalization_target / totals[cell_indices])
                    np.add.at(group_score, cell_indices, normalized.astype(np.float32))
                    detected_cells.append(cell_indices)
                if detected_cells:
                    group_detected[np.unique(np.concatenate(detected_cells))] += 1
            if found:
                group_score /= len(found)
            scores_array[:, label_index] = group_score
            detected_array[:, label_index] = group_detected
            coverage_rows.append(
                {
                    "group": label,
                    "n_markers_listed": len(requested),
                    "n_markers_found": len(found),
                    "markers_found": ", ".join(found),
                    "markers_missing": ", ".join(sorted(set(requested) - set(found))),
                }
            )

    scores = pd.DataFrame(scores_array, index=cell_ids, columns=labels)
    detected = pd.DataFrame(detected_array, index=cell_ids, columns=labels)
    coverage = pd.DataFrame(coverage_rows)
    return scores, detected, coverage


def score_h5_marker_sets(
    matrix_path: Path,
    marker_sets: dict[str, list[str]],
    normalization_target: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        import h5py
        from scipy import sparse
    except ModuleNotFoundError as error:  # pragma: no cover - environment-specific.
        raise ModuleNotFoundError(
            "Scoring cell_feature_matrix.h5 requires h5py and scipy"
        ) from error

    with h5py.File(matrix_path, "r") as handle:
        group = handle["matrix"]
        data = group["data"][:]
        indices = group["indices"][:]
        indptr = group["indptr"][:]
        shape = tuple(int(value) for value in group["shape"][:])
        features = group["features"]
        feature_key = "name" if "name" in features else "id"
        feature_names = [decode_text(value) for value in features[feature_key][:]]
        if "feature_type" in features:
            feature_types = [decode_text(value) for value in features["feature_type"][:]]
        else:
            feature_types = ["Gene Expression"] * len(feature_names)
        cell_ids = [decode_text(value) for value in group["barcodes"][:]]

    gene_by_cell = sparse.csc_matrix((data, indices, indptr), shape=shape)
    cell_by_gene = gene_by_cell.T.tocsr()
    gene_columns = [
        index
        for index, (name, feature_type) in enumerate(zip(feature_names, feature_types))
        if is_gene_expression(feature_type, name)
    ]
    totals = np.asarray(cell_by_gene[:, gene_columns].sum(axis=1)).ravel().astype(float)
    scale = np.divide(
        normalization_target,
        totals,
        out=np.zeros_like(totals, dtype=float),
        where=totals > 0,
    )
    normalized = cell_by_gene.multiply(scale[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)

    feature_lookup: dict[str, list[int]] = {}
    for feature_index in gene_columns:
        feature_lookup.setdefault(feature_names[feature_index].upper(), []).append(feature_index)

    labels = sorted(marker_sets, key=natural_key)
    score_tables: dict[str, np.ndarray] = {}
    detected_tables: dict[str, np.ndarray] = {}
    coverage_rows: list[dict[str, object]] = []
    for label in labels:
        requested = sorted({gene.upper() for gene in marker_sets[label]})
        found = [gene for gene in requested if gene in feature_lookup]
        columns = [index for gene in found for index in feature_lookup[gene]]
        if columns:
            score_tables[label] = np.asarray(normalized[:, columns].mean(axis=1)).ravel()
            detected_tables[label] = np.asarray((cell_by_gene[:, columns] > 0).sum(axis=1)).ravel()
        else:
            score_tables[label] = np.zeros(cell_by_gene.shape[0], dtype=float)
            detected_tables[label] = np.zeros(cell_by_gene.shape[0], dtype=np.uint16)
        coverage_rows.append(
            {
                "group": label,
                "n_markers_listed": len(requested),
                "n_markers_found": len(found),
                "markers_found": ", ".join(found),
                "markers_missing": ", ".join(sorted(set(requested) - set(found))),
            }
        )

    scores = pd.DataFrame(score_tables, index=cell_ids)
    detected = pd.DataFrame(detected_tables, index=cell_ids).astype(np.uint16)
    coverage = pd.DataFrame(coverage_rows)
    return scores, detected, coverage


def assign_cells_from_marker_scores(
    scores: pd.DataFrame,
    detected_markers: pd.DataFrame,
    *,
    min_score: float,
    min_margin: float,
    min_markers: int,
    uncertain_label: str,
) -> pd.DataFrame:
    if scores.empty or scores.shape[1] == 0:
        raise ValueError("No marker-set scores are available")
    if not scores.index.equals(detected_markers.index) or list(scores.columns) != list(detected_markers.columns):
        raise ValueError("Score and detected-marker tables are not aligned")
    score_values = scores.to_numpy(dtype=np.float32, copy=False)
    detected_values = detected_markers.to_numpy(dtype=np.uint16, copy=False)
    row_index = np.arange(len(scores))
    top_index = np.argmax(score_values, axis=1)
    top_score = score_values[row_index, top_index]
    if scores.shape[1] > 1:
        second_index = np.argpartition(score_values, -2, axis=1)[:, -2]
        second_score = score_values[row_index, second_index]
        swap = second_score > top_score
        corrected_top = np.where(swap, second_index, top_index)
        corrected_second = np.where(swap, top_index, second_index)
        top_index, second_index = corrected_top, corrected_second
        top_score = score_values[row_index, top_index]
        second_score = score_values[row_index, second_index]
    else:
        second_index = np.zeros(len(scores), dtype=np.int64)
        second_score = np.zeros(len(scores), dtype=np.float32)

    top_markers = detected_values[row_index, top_index]
    margin = top_score - second_score
    labels = np.asarray(scores.columns, dtype=object)
    accepted = (
        (top_score >= min_score)
        & (margin >= min_margin)
        & (top_markers >= min_markers)
    )
    assigned = np.where(accepted, labels[top_index], uncertain_label)
    return pd.DataFrame(
        {
            "cell_id": scores.index.astype(str),
            "group": assigned,
            "top_marker_group": labels[top_index],
            "top_score": top_score,
            "runner_up": np.where(scores.shape[1] > 1, labels[second_index], ""),
            "runner_up_score": second_score,
            "score_margin": margin,
            "n_top_markers_detected": top_markers,
            "passed_score": top_score >= min_score,
            "passed_margin": margin >= min_margin,
            "passed_marker_count": top_markers >= min_markers,
        }
    )


def summarize_annotations(assignments: pd.DataFrame) -> pd.DataFrame:
    summary = (
        assignments.groupby("group", dropna=False)
        .agg(
            n_cells=("cell_id", "size"),
            median_top_score=("top_score", "median"),
            median_margin=("score_margin", "median"),
            median_markers=("n_top_markers_detected", "median"),
        )
        .reset_index()
        .sort_values("n_cells", ascending=False)
    )
    summary["fraction"] = summary["n_cells"] / summary["n_cells"].sum()
    return summary


def annotate_sample(
    sample_dir: Path,
    root: Path,
    config: AnnotationConfig,
    *,
    marker_groups_path: str | Path | None = None,
) -> dict[str, object]:
    matrix_path = find_feature_matrix(sample_dir)
    if matrix_path is None:
        raise FileNotFoundError(f"No cell_feature_matrix.zarr.zip or .h5 found in {sample_dir}")

    effective_marker_groups_path = marker_groups_path
    if effective_marker_groups_path is None and config.use_sample_marker_groups:
        marker_files = sample_relevant_files(sample_dir, root)["marker_groups"]
        effective_marker_groups_path = marker_files[0] if marker_files else None
    marker_sets = (
        load_marker_sets(gene_groups_path=effective_marker_groups_path)
        if effective_marker_groups_path is not None
        else config.marker_sets
    )

    if matrix_path.name.endswith(".zarr.zip"):
        cell_ids = cell_ids_from_cells_zarr(sample_dir)
        if cell_ids is None:
            cells = read_optional_cells_csv(find_cells_csv(sample_dir))
            if cells is None:
                raise FileNotFoundError(
                    "Zarr matrix annotation requires cells.zarr.zip or a cell CSV "
                    "with cell_id in matrix order"
                )
            cell_ids = cells["cell_id"].astype(str).tolist()
        scores, detected, coverage = score_zarr_marker_sets(
            matrix_path,
            cell_ids,
            marker_sets,
            config.normalization_target,
        )
    else:
        scores, detected, coverage = score_h5_marker_sets(
            matrix_path,
            marker_sets,
            config.normalization_target,
        )

    assignments = assign_cells_from_marker_scores(
        scores,
        detected,
        min_score=config.min_score,
        min_margin=config.min_margin,
        min_markers=config.min_markers,
        uncertain_label=config.uncertain_label,
    )
    summary = summarize_annotations(assignments)
    details = assignments.copy()
    details = details.join(scores.add_prefix("score__"), on="cell_id")
    details = details.join(detected.add_prefix("detected__"), on="cell_id")

    prefix = sample_dir.name
    detail_path = sample_dir / f"{prefix}_expression_cell_annotation_details.csv.gz"
    summary_path = sample_dir / f"{prefix}_expression_cell_annotation_summary.csv"
    coverage_path = sample_dir / f"{prefix}_expression_marker_coverage.csv"
    for path in (detail_path, summary_path, coverage_path):
        if path.exists() and not config.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}")
    details.to_csv(detail_path, index=False, compression="gzip")
    summary.to_csv(summary_path, index=False)
    coverage.to_csv(coverage_path, index=False)

    cell_groups_path = ""
    if config.export_cell_groups:
        output = sample_dir / f"{prefix}_cell_groups.csv"
        if output.exists() and not config.overwrite:
            raise FileExistsError(f"Refusing to overwrite {output}")
        assignments[["cell_id", "group"]].to_csv(output, index=False)
        cell_groups_path = str(output)

    n_assigned = int((assignments["group"] != config.uncertain_label).sum())
    return {
        "sample": sample_dir.name,
        "status": "success",
        "matrix_path": str(matrix_path),
        "marker_source": str(effective_marker_groups_path or "configured_marker_sets"),
        "details_path": str(detail_path),
        "summary_path": str(summary_path),
        "coverage_path": str(coverage_path),
        "cell_groups_path": cell_groups_path,
        "n_cells": len(assignments),
        "n_marker_groups": len(scores.columns),
        "n_assigned_cells": n_assigned,
        "assigned_fraction": n_assigned / len(assignments) if len(assignments) else np.nan,
        "error": "",
    }


def run_expression_annotation(
    root: str | Path,
    sample_names: list[str] | None = None,
    *,
    markers_json: str | Path | None = None,
    marker_groups_path: str | Path | None = None,
    normalization_target: float = 10_000.0,
    min_score: float = 0.25,
    min_margin: float = 0.10,
    min_markers: int = 1,
    uncertain_label: str = "Uncertain",
    overwrite: bool = False,
    export_cell_groups: bool = False,
) -> pd.DataFrame:
    root = Path(root).expanduser().resolve()
    config = AnnotationConfig(
        marker_sets=load_marker_sets(markers_json=markers_json),
        normalization_target=normalization_target,
        min_score=min_score,
        min_margin=min_margin,
        min_markers=min_markers,
        uncertain_label=uncertain_label,
        overwrite=overwrite,
        export_cell_groups=export_cell_groups,
        use_sample_marker_groups=markers_json is None and marker_groups_path is None,
    )
    rows: list[dict[str, object]] = []
    for sample_dir in discover_sample_dirs(root, sample_names):
        try:
            rows.append(
                annotate_sample(
                    sample_dir,
                    root,
                    config,
                    marker_groups_path=marker_groups_path,
                )
            )
            print(f"[expression annotation] {sample_dir.name}: success")
        except Exception as error:
            rows.append(
                {
                    "sample": sample_dir.name,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"[expression annotation] {sample_dir.name}: FAILED: {error}")
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "expression_annotation_summary.csv", index=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate Xenium cells from expression marker scores."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--samples", nargs="+")
    parser.add_argument("--markers-json", type=Path)
    parser.add_argument("--marker-groups", type=Path)
    parser.add_argument("--normalization-target", type=float, default=10_000.0)
    parser.add_argument("--min-score", type=float, default=0.25)
    parser.add_argument("--min-margin", type=float, default=0.10)
    parser.add_argument("--min-markers", type=int, default=1)
    parser.add_argument("--uncertain-label", default="Uncertain")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--export-cell-groups",
        action="store_true",
        help="Also write <sample>_cell_groups.csv for the tiling pipeline.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_expression_annotation(
        args.root,
        sample_names=args.samples,
        markers_json=args.markers_json,
        marker_groups_path=args.marker_groups,
        normalization_target=args.normalization_target,
        min_score=args.min_score,
        min_margin=args.min_margin,
        min_markers=args.min_markers,
        uncertain_label=args.uncertain_label,
        overwrite=args.overwrite,
        export_cell_groups=args.export_cell_groups,
    )
    print(summary)
    return int(summary["status"].eq("failed").any()) if not summary.empty else 0


if __name__ == "__main__":
    raise SystemExit(main())
