"""Source-proportion-weighted spatial summaries from calibrated TCGA tile predictions."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from . import brca_reference_calibration as calibration

CELL_TYPES = {
    "Tumor Cells": "tumor_cells", "Stromal Cells": "stromal_cells",
    "pan-APC + B": "pan_apc_b", "T Cells": "t_cells",
    "Normal Epithelial Cells": "normal_epithelial_cells",
}
THRESHOLDS = dict(tumor_cells=.15, stromal_cells=.30, pan_apc_b=.30, t_cells=.30)
GEOMETRY = ["tile_row", "tile_col", "x0_px", "y0_px", "x1_px", "y1_px", "tile_width_um", "tile_height_um"]
RECIPE = "xenium_brca_source_weighted_nearest_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def tile_matrix(table, slide_id, crop_um, expected_tiles):
    """Validate the long export before aligning geometry with all five scores."""
    required = ["tile_id", "slide_id", "patient_id", "nominal_size_um", "cell_type", "predicted_raw", *GEOMETRY]
    missing = set(required)-set(table)
    if missing:
        raise ValueError(f"Missing tile columns: {sorted(missing)}")
    if table[required].isna().any().any():
        raise ValueError("Missing scores, identifiers or coordinates in tile predictions")
    if not table.slide_id.eq(slide_id).all() or not table.patient_id.eq(slide_id[:12]).all():
        raise ValueError("Tile export slide/patient ID mismatch")
    if not np.isclose(table.nominal_size_um, crop_um, rtol=0, atol=1e-8).all():
        raise ValueError("Tile export crop-size mismatch")
    if set(table.cell_type) != set(CELL_TYPES):
        raise ValueError("Spatial analysis requires exactly the five completed cell-type predictions")
    if table.duplicated(["tile_id", "cell_type"]).any():
        raise ValueError("Duplicate tile/cell-type predictions")
    grouped = table.groupby("tile_id", sort=False)
    if grouped[GEOMETRY].nunique(dropna=False).ne(1).any().any():
        raise ValueError("Cell types disagree on tile coordinates")
    geometry = grouped[GEOMETRY].first()
    raw = table.pivot(index="tile_id", columns="cell_type", values="predicted_raw").reindex(geometry.index)
    raw = raw[list(CELL_TYPES)].rename(columns=CELL_TYPES)
    if len(raw) != expected_tiles or raw.isna().any().any():
        raise ValueError("Tile count or cell-type completeness disagrees with completed slide summary")
    if not np.isfinite(raw.to_numpy()).all() or not np.isfinite(geometry.to_numpy()).all():
        raise ValueError("Non-finite tile scores or coordinates")
    if geometry.duplicated(["tile_row", "tile_col"]).any():
        raise ValueError("Duplicate grid coordinates with different tile IDs")
    grid = geometry[["tile_col", "tile_row"]].to_numpy()
    if (grid < 1).any() or not np.equal(grid, np.floor(grid)).all():
        raise ValueError("Grid row/column must be positive integers")
    widths = geometry.x1_px-geometry.x0_px
    heights = geometry.y1_px-geometry.y0_px
    if geometry[["x0_px", "y0_px"]].lt(0).any().any():
        raise ValueError("Negative pixel coordinates")
    if (widths <= 0).any() or (heights <= 0).any() or geometry[["tile_width_um", "tile_height_um"]].le(0).any().any():
        raise ValueError("Tile dimensions must be positive")
    mpp_x = geometry.tile_width_um/widths
    mpp_y = geometry.tile_height_um/heights
    if not np.allclose(mpp_x, mpp_x.iloc[0], rtol=1e-7) or not np.allclose(mpp_y, mpp_y.iloc[0], rtol=1e-7):
        raise ValueError("Inconsistent physical pixel spacing within slide")
    physical = np.column_stack(((geometry.x0_px+geometry.x1_px)/2*mpp_x,
                                (geometry.y0_px+geometry.y1_px)/2*mpp_y))
    return raw, {"grid": grid, "um": physical}


def prepare_scores(raw, calibration_bundle=None):
    """Apply the selected score policy before five-type normalization."""
    if calibration_bundle is None:
        clipped = raw.clip(0, 1)
    else:
        clipped = pd.DataFrame({feature: calibration.apply(calibration_bundle["tables"][target], raw[feature])
                               for target, feature in CELL_TYPES.items()}, index=raw.index)
    total = clipped.sum(axis=1)
    valid = total.gt(0)
    normalized = clipped.loc[valid].div(total.loc[valid], axis=0)
    qc = dict(n_tiles=len(raw), n_zero_sum_tiles=int((~valid).sum()), n_valid_tiles=int(valid.sum()),
              n_clipped_values=int(((raw < 0)|(raw > 1)).sum().sum()))
    if calibration_bundle is not None:
        qc["n_raw_values_outside_0_1"] = qc["n_clipped_values"]
        qc["n_clipped_values"] = 0  # The fitted reference knots already lie in [0, 1].
    return normalized, valid.to_numpy(), qc


def weighted_distances(coordinates, scores, thresholds=None):
    """For every source tile, nearest-target Euclidean distance × source score.

    Multiple labels and self-matches are allowed. Absent categories yield missing
    summaries, never fabricated large distances or zero. SD uses ddof=0.
    """
    thresholds = THRESHOLDS if thresholds is None else thresholds
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.shape != (len(scores), 2) or not np.isfinite(coordinates).all():
        raise ValueError("Coordinates must be finite N×2 aligned with score rows")
    if not set(thresholds) <= set(scores):
        raise ValueError("Threshold cell types are missing from scores")
    values = scores.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Prepared spatial scores must be finite and within [0,1]")
    if not all(np.isfinite(t) and 0 <= t <= 1 for t in thresholds.values()):
        raise ValueError("Thresholds must be finite and within [0,1]")
    masks = {name: scores[name].to_numpy() > value for name, value in thresholds.items()}
    trees = {name: cKDTree(coordinates[mask]) for name, mask in masks.items() if mask.any()}
    result, qc = {}, []
    for source, source_mask in masks.items():
        for target, target_mask in masks.items():
            if source == target:
                continue
            prefix = f"{source}_to_{target}"
            if source_mask.any() and target_mask.any():
                nearest = trees[target].query(coordinates[source_mask], k=1, workers=1)[0]
                distances = nearest*scores.loc[source_mask, source].to_numpy()
                q25, q75 = np.quantile(distances, [.25, .75], method="linear")
                stats = dict(mean=float(distances.mean()), std=float(distances.std(ddof=0)),
                             median=float(np.median(distances)), iqr=float(q75-q25))
                state = "ok"
            else:
                stats = dict.fromkeys(["mean", "std", "median", "iqr"], None)
                state = "no_source_tiles" if not source_mask.any() else "no_target_tiles"
            result.update({f"{prefix}_{name}": value for name, value in stats.items()})
            qc.append(dict(source=source, target=target, n_source_tiles=int(source_mask.sum()),
                           n_target_tiles=int(target_mask.sum()), status=state,
                           n_overlap=int((source_mask & target_mask).sum()),
                           overlap_fraction=(float((source_mask & target_mask).sum()/source_mask.sum())
                                             if source_mask.any() else None),
                           zero_fraction=(float((nearest == 0).mean()) if state == "ok" else None)))
    return result, qc


def compute_cohort(slides, output, cache_dir, *, score_policy="clip_normalize_raw", thresholds=None,
                   calibration_bundle=None, model_project_root=None, cache_thresholds=None):
    """Process one slide at a time; reuse hash-verified per-slide small summaries."""
    if score_policy not in {"clip_normalize_raw", "reference_quantile_normalize"}:
        raise ValueError("Unknown spatial score policy")
    if (score_policy == "reference_quantile_normalize") != (calibration_bundle is not None):
        raise ValueError("Reference quantile normalization requires its calibration bundle; raw mode takes none")
    thresholds = dict(THRESHOLDS if thresholds is None else thresholds)
    if set(thresholds) != set(THRESHOLDS):
        raise ValueError("Use the four BRCA spatial cell types")
    cache_thresholds = list(cache_thresholds or [])
    for other in cache_thresholds:
        if set(other) != set(THRESHOLDS) or not all(np.isfinite(t) and 0 <= t <= 1 for t in other.values()):
            raise ValueError("Invalid additional cache thresholds")
    required = ["slide_id", "patient_id", "crop_size_um", "source_run", "cell_type", "n_sampled_tiles", "sample_mean"]
    if not set(required) <= set(slides):
        raise ValueError("Pass the selected slide_predictions table")
    if slides.duplicated(["slide_id", "cell_type"]).any():
        raise ValueError("Select one result per slide and cell type")
    selected = slides[[x for x in required if x not in {"cell_type", "sample_mean"}]].drop_duplicates()
    if selected.slide_id.duplicated().any() or selected.empty:
        raise ValueError("Selected slide summaries disagree or selection is empty")
    if not slides.groupby("slide_id").cell_type.apply(lambda x: set(x) == set(CELL_TYPES)).all():
        raise ValueError("Each selected slide must have all five cell types")
    output, cache_dir = Path(output), Path(cache_dir)
    output.mkdir(parents=True, exist_ok=False)
    cache_dir.mkdir(parents=True, exist_ok=True)
    method = dict(recipe=RECIPE, code_sha256=sha256(__file__), score_policy=score_policy,
                  thresholds=thresholds, threshold_comparison="strict >", std_ddof=0,
                  iqr_method="linear/type7", self_matches=True, missing_category="NA, no imputation")
    if calibration_bundle is not None:
        method.update(calibration_recipe=calibration_bundle["recipe"],
                      calibration_sha256=calibration.fingerprint(calibration_bundle),
                      calibration_code_sha256=sha256(calibration.__file__))
    frames = {"grid": [], "um": []}
    audits, pair_rows, calibrated_rows = [], [], []
    for number, row in enumerate(selected.itertuples(index=False), 1):
        folder = Path(row.source_run)/"slides"/row.slide_id
        source = folder/"tile_predictions_raw.csv.gz"
        summary_path = folder/"prediction_summary.json"
        preparation_path = folder/"tile_preparation.json"
        summary = json.loads(summary_path.read_text())
        preparation = json.loads(preparation_path.read_text())
        if (preparation.get("status") != "complete" or "max_tiles_per_slide_size" not in preparation
            or preparation["max_tiles_per_slide_size"] is not None
            or not summary.get("manifest_sha256")
            or summary["manifest_sha256"] != preparation.get("manifest_sha256")
            or preparation.get("sizes_um") != [row.crop_size_um]):
            raise ValueError(f"Not a complete all-tile preparation matching predictions: {folder}")
        # Older completed exports omit tile_selection; the all-tile preparation
        # marker above is authoritative, as checked by the prediction coverage validator.
        if (summary.get("status") != "complete" or summary.get("tile_selection", "all_eligible") != "all_eligible"
            or summary.get("n_slides") != 1 or summary.get("n_crops") != row.n_sampled_tiles
            or set(summary.get("cell_types", [])) != set(CELL_TYPES)
            or summary.get("sizes_um") != [row.crop_size_um]):
            raise ValueError(f"Not a complete all-tile single-slide prediction: {folder}")
        if any(summary.get(x) is not False for x in ["calibration", "clipping", "across_cell_type_normalization"]):
            raise ValueError(f"Expected raw predictions, but score preprocessing flags differ: {folder}")
        if calibration_bundle is not None:
            calibration.verify_models(calibration_bundle, summary, project_root=model_project_root)
        identity = dict(method=method, prediction_csv_sha256=sha256(source),
                        prediction_summary_sha256=sha256(summary_path), slide_id=row.slide_id,
                        tile_preparation_sha256=sha256(preparation_path),
                        crop_size_um=float(row.crop_size_um), source_run=str(row.source_run))
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cached_path = cache_dir/(key+".json")
        additional = []
        for other in cache_thresholds:
            other_identity = dict(identity, method=dict(method, thresholds=dict(other)))
            other_key = hashlib.sha256(json.dumps(other_identity, sort_keys=True).encode()).hexdigest()
            additional.append((dict(other), other_identity, cache_dir/(other_key+".json")))
        if cached_path.is_file() and all(path.is_file() for _, _, path in additional):
            saved = json.loads(cached_path.read_text())
            if saved.get("identity") != identity:
                raise ValueError(f"Spatial cache identity mismatch: {cached_path}")
            reused = True
        else:
            table = pd.read_csv(source, usecols=["tile_id", "slide_id", "patient_id", "nominal_size_um",
                                               "cell_type", "predicted_raw", *GEOMETRY])
            raw, coordinates = tile_matrix(table, row.slide_id, row.crop_size_um, row.n_sampled_tiles)
            processed, keep, qc = prepare_scores(raw, calibration_bundle)
            if calibration_bundle is not None and processed.empty:
                raise ValueError(f"No positive-total calibrated tiles: {row.slide_id}")
            del table
            unit_summaries, pair_qc = {}, None
            for unit in frames:
                unit_summaries[unit], pair_qc = weighted_distances(coordinates[unit][keep], processed, thresholds)
            saved = dict(identity=identity, qc=qc, summaries=unit_summaries, pair_qc=pair_qc,
                         raw_means=raw.mean().to_dict())
            if calibration_bundle is not None:
                saved["normalized_means"] = processed.mean().to_dict()
                saved["calibrated_means"] = {feature: float(calibration.apply(
                    calibration_bundle["tables"][target], raw[feature]).mean())
                    for target, feature in CELL_TYPES.items()}
            write_json(cached_path, saved)
            # Same validated tiles, preprocessing and means; only labels differ.
            for other, other_identity, other_path in additional:
                if other_path.is_file():
                    if json.loads(other_path.read_text()).get("identity") != other_identity:
                        raise ValueError(f"Additional spatial cache identity mismatch: {other_path}")
                    continue
                other_summaries, other_qc = {}, None
                for unit in frames:
                    other_summaries[unit], other_qc = weighted_distances(coordinates[unit][keep], processed, other)
                write_json(other_path, dict(saved, identity=other_identity,
                           summaries=other_summaries, pair_qc=other_qc))
            reused = False
        expected_means = slides.loc[slides.slide_id.eq(row.slide_id)].set_index("cell_type").sample_mean
        if not np.allclose([saved["raw_means"][feature] for feature in CELL_TYPES.values()],
                           expected_means.reindex(CELL_TYPES).to_numpy(), rtol=1e-6, atol=1e-8):
            raise ValueError(f"Tile prediction means disagree with the selected slide summary: {row.slide_id}")
        base = dict(slide_id=row.slide_id, patient_id=row.patient_id,
                    crop_size_um=float(row.crop_size_um), source_run=str(row.source_run))
        if calibration_bundle is not None:
            for target, feature in CELL_TYPES.items():
                calibrated_rows.append(dict(**base, cell_type=target,
                    n_sampled_tiles=row.n_sampled_tiles, n_valid_tiles=saved["qc"]["n_valid_tiles"],
                    sample_mean=saved["normalized_means"][feature], fraction_outside_0_1=0.,
                    raw_sample_mean=saved["raw_means"][feature],
                    calibrated_unnormalized_mean=saved["calibrated_means"][feature],
                    score_policy=score_policy, calibration_sha256=method["calibration_sha256"]))
        for unit in frames:
            frames[unit].append({**base, **saved["summaries"][unit]})
        audits.append(dict(**base, **saved["qc"], cache_reused=reused,
                           prediction_csv=str(source), prediction_csv_sha256=identity["prediction_csv_sha256"],
                           prediction_summary_sha256=identity["prediction_summary_sha256"],
                           tile_preparation_sha256=identity["tile_preparation_sha256"]))
        pair_rows.extend(dict(**base, **item) for item in saved["pair_qc"])
        print(f"Spatial {number}/{len(selected)}: {row.slide_id} | {row.crop_size_um:g} um | "
              f"{saved['qc']['n_valid_tiles']:,} tiles | {'cached' if reused else 'computed'}", flush=True)
    pd.DataFrame(audits).to_csv(output/"spatial_input_audit.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(output/"spatial_pair_status.csv", index=False)
    outputs = {}
    if calibration_bundle is not None:
        means_path = output / "slide_predictions_calibrated.csv"
        pd.DataFrame(calibrated_rows).to_csv(means_path, index=False)
        write_json(output / "calibration_bundle.json", calibration_bundle)
        outputs["slide_predictions"] = means_path
    for unit, records in frames.items():
        csv_path = output/f"spatial_distances_{unit}.csv"
        pd.DataFrame(records).to_csv(csv_path, index=False)
        metadata = dict(prediction_source="current_xenium_predictions", distance_units=unit, method=method,
            calibration=("None: raw Xenium scores clipped to [0,1], then normalized across all five types per tile"
                         if calibration_bundle is None else
                         "Xenium LOIO reference quantile maps, clipping to [0,1], then five-type tile normalization"),
            coordinates="tile row/column" if unit == "grid" else "pixel centers multiplied by audited physical pixel spacing",
            interpretation="Source-score-weighted nearest-tile distances; not segmented cell-to-cell distances",
            manuscript_reproduction="Source-weighted formula retained; thresholds are recorded in method; score preprocessing and physical units are explicit adaptations",
            n_slides=len(selected), crop_sizes_um=sorted(selected.crop_size_um.unique().tolist()),
            csv_sha256=sha256(csv_path), input_audit_sha256=sha256(output/"spatial_input_audit.csv"))
        metadata_path = output/f"spatial_distances_{unit}.json"
        write_json(metadata_path, metadata)
        outputs[unit] = dict(csv=csv_path, metadata=metadata_path)
    return outputs
