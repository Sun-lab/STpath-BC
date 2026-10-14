"""Reference STpath quantile mapping fitted to Xenium Combined LOIO predictions.

The reference's duplicate quantile knots and np.interp behavior are intentional.
Target rescaling produces enrichment scores, not measured absolute fractions.
"""
from pathlib import Path
import argparse
import hashlib
import json
import tempfile

import numpy as np
import pandas as pd

RECIPE = "stpath_bc_quantile_99p5_max_100_v1"
REFERENCE_COMMIT = "86336ae4b12eb954667049889940bc10e2e4bac7"
REFERENCE_URL = ("https://github.com/Sun-lab/STpath-BC/blob/" + REFERENCE_COMMIT +
                 "/Workflow/StepC_Feature_Extraction_and_Train_Models/BRCA_XGBoost_Prediction.py")
TARGETS = {"Tumor Cells": "Tumor", "Stromal Cells": "Stromal",
           "pan-APC + B": "pan_APC", "T Cells": "T_cell",
           "Normal Epithelial Cells": "Normal_Epithelial"}
SCALED_TARGETS = {"T_cell", "pan_APC", "Normal_Epithelial"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(bundle):
    return hashlib.sha256(json.dumps(bundle, sort_keys=True, allow_nan=False).encode()).hexdigest()


def fit(y_true, y_pred, cell_type):
    """Match the reference function on finite inputs; reject undefined scaling."""
    if cell_type not in TARGETS:
        raise ValueError(f"Unknown calibration cell type: {cell_type}")
    truth, pred = np.asarray(y_true, dtype=float).ravel(), np.asarray(y_pred, dtype=float).ravel()
    if not len(truth) or truth.shape != pred.shape or not np.isfinite([truth, pred]).all():
        raise ValueError("Calibration requires aligned, nonempty, finite prediction/target pairs")
    if ((truth < 0) | (truth > 1)).any():
        raise ValueError("Measured calibration proportions must lie in [0, 1]")
    info = dict(needs_preprocessing=TARGETS[cell_type] in SCALED_TARGETS)
    if info["needs_preprocessing"]:
        threshold = float(np.percentile(truth, 99.5))
        keep = truth < threshold
        if not keep.any() or truth[keep].max() <= 0:
            raise ValueError(f"{cell_type}: reference 99.5% filtering leaves no positive scaling maximum")
        scale = float(truth[keep].max())
        info.update(outlier_threshold=threshold, scale_factor=scale, max_in_remaining=scale,
                    removed_count=int((~keep).sum()), original_samples=len(truth),
                    filtered_samples=int(keep.sum()), outlier_percentile=99.5)
        truth, pred = truth[keep] / scale, pred[keep]
    levels = np.linspace(0, 100, 100)
    return dict(cell_type=cell_type, reference_cell_type=TARGETS[cell_type],
                calibration_method="quantile_mapping", scaling_info=info,
                quantile_levels=levels.tolist(),
                y_pred_quantiles=np.percentile(pred, levels).tolist(),
                y_true_quantiles=np.percentile(truth, levels).tolist())


def apply(table, values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite predictions passed to calibration")
    # Do not merge tied knots: the upstream routine passes all 100 to np.interp.
    return np.clip(np.interp(values, table["y_pred_quantiles"], table["y_true_quantiles"]), 0, 1)


def load_bundle(path):
    bundle = json.loads(Path(path).read_text())
    if (bundle.get("recipe") != RECIPE or set(bundle.get("tables", {})) != set(TARGETS)
        or set(bundle.get("model_contracts", {})) != set(TARGETS)):
        raise ValueError("Expected a complete five-target reference calibration bundle")
    for target, table in bundle["tables"].items():
        if table.get("cell_type") != target or table.get("reference_cell_type") != TARGETS[target]:
            raise ValueError("Calibration target mapping mismatch")
        knots = np.asarray([table["y_pred_quantiles"], table["y_true_quantiles"]], dtype=float)
        if knots.shape != (2, 100) or not np.isfinite(knots).all() or (np.diff(knots, axis=1) < 0).any():
            raise ValueError("Invalid reference quantile knots")
        if not np.array_equal(table["quantile_levels"], np.linspace(0, 100, 100)):
            raise ValueError("Reference calibration requires 100 evenly spaced percentile levels")
    return bundle


def load_or_build_bundle(path, reference_dir):
    """Load a saved bundle, or build a missing one from the existing LOIO export."""
    path, reference_dir = Path(path), Path(reference_dir)
    if path.exists():
        return load_bundle(path)
    if path.name != "calibration_bundle.json":
        raise ValueError("Use calibration_bundle.json as the generated bundle filename")
    predictions = reference_dir / "paired_predictions.csv.gz"
    provenance = reference_dir / "provenance.json"
    missing = [str(p) for p in (predictions, provenance) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Calibration bundle is missing: {path}\n"
            "Copy the saved reference_calibration_v1 folder to the server, or provide these "
            "LOIO inputs to build it:\n" + "\n".join(missing))
    if path.parent.exists() and any(path.parent.iterdir()):
        raise FileExistsError(f"Incomplete calibration folder contains files: {path.parent}. "
                              "Use a new CALIBRATION_BUNDLE folder or copy the complete saved bundle.")
    path.parent.parent.mkdir(parents=True, exist_ok=True)
    print(f"Building missing calibration bundle from {reference_dir}", flush=True)
    # Publish the complete directory together; an interrupted fit leaves no partial bundle.
    with tempfile.TemporaryDirectory(prefix=".calibration-build-", dir=path.parent.parent) as temporary:
        staged = Path(temporary) / "bundle"
        build_bundle(predictions, provenance, staged)
        staged.replace(path.parent)
    return load_bundle(path)


def verify_models(bundle, prediction_summary, *, project_root=None):
    """Bind calibration targets to the original final-model export metadata.

    Fingerprints intentionally reject the newly trained CRC fixed-feature models;
    those require their own matching LOIO references and TCGA predictions.
    """
    exports = prediction_summary.get("model_exports", [])
    if len(exports) != len(TARGETS) or {v["cell_type"] for v in exports} != set(TARGETS):
        raise ValueError("Calibration requires the five original model export identities")
    for export in exports:
        target = export["cell_type"]
        path = Path(export["target_dir"]) / "model_metadata/Combined.json"
        if project_root is not None and not path.is_file():
            local_root = Path(project_root).resolve()
            if not path.is_relative_to(local_root):
                if "xgboost_training" not in path.parts:
                    raise ValueError(f"{target}: model metadata is outside the supplied project")
                path = local_root / Path(*path.parts[path.parts.index("xgboost_training"):])
        if sha256(path) != export["file_sha256"].get("model_metadata/Combined.json"):
            raise ValueError(f"{target}: saved prediction/model metadata hash mismatch")
        metadata = json.loads(path.read_text())
        expected = bundle["model_contracts"][target]
        if any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError(f"{target}: calibration reference does not match the prediction model")


def build_bundle(predictions, provenance, output):
    """Fit production maps and individual-excluded diagnostics from a complete export."""
    predictions, provenance, output = Path(predictions), Path(provenance), Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new calibration output directory: {output}")
    source = json.loads(provenance.read_text())
    if (source.get("status") != "complete" or source.get("prediction_scale") != "raw"
        or source.get("n_missing_folds") != 0 or source.get("exclude_individuals")):
        raise ValueError("Use complete, uncalibrated, all-individual Xenium LOIO predictions")
    cols = ["tile_id", "individual_id", "cell_type", "outer_fold", "observed_proportion", "predicted_xenium_oof"]
    data = pd.read_csv(predictions, usecols=cols, float_precision="round_trip")
    if data[cols].isna().any().any() or data.duplicated(["tile_id", "cell_type"]).any():
        raise ValueError("Missing or duplicate LOIO rows")
    if set(data.cell_type) != set(TARGETS) or len(data) != source.get("n_paired_rows"):
        raise ValueError("LOIO target/count mismatch with source provenance")
    sources = {v["visium_target"]: v for v in source["target_sources"]}
    tables, contracts, diagnostics, counts = {}, {}, [], []
    for target, reference_name in TARGETS.items():
        part = data.loc[data.cell_type.eq(target)]
        meta = sources[reference_name]["run_metadata"]
        if len(part) != meta["n_tiles"] or part.individual_id.nunique() < 3:
            raise ValueError(f"{target}: incomplete tile or individual coverage")
        if (part.groupby("individual_id").outer_fold.nunique().ne(1).any()
            or part.groupby("outer_fold").individual_id.nunique().ne(1).any()):
            raise ValueError(f"{target}: invalid LOIO individual/fold mapping")
        contracts[target] = dict(cell_type=target, model_type="Combined",
            training_config_fingerprint=meta["training_config_fingerprint"],
            training_tile_id_checksum=meta["tile_id_checksum"], n_training_tiles=meta["n_tiles"],
            extraction_recipe=source["feature_recipe"], individual_ids=sorted(part.individual_id.astype(str).unique()))
        tables[target] = fit(part.observed_proportion, part.predicted_xenium_oof, target)
        info = tables[target]["scaling_info"]
        counts.append(dict(cell_type=target, n_tiles=len(part), n_individuals=part.individual_id.nunique(),
                           **info))
        for individual in part.individual_id.unique():
            train, test = part.loc[part.individual_id.ne(individual)], part.loc[part.individual_id.eq(individual)]
            entry = dict(cell_type=target, held_out_individual=individual, n_tiles=len(test),
                         n_calibration_tiles=len(train), status="ok")
            try:
                heldout_map = fit(train.observed_proportion, train.predicted_xenium_oof, target)
                calibrated = apply(heldout_map, test.predicted_xenium_oof)
                processed_truth = np.clip(test.observed_proportion.to_numpy() /
                    heldout_map["scaling_info"].get("scale_factor", 1), 0, 1)
                raw = test.predicted_xenium_oof.to_numpy()
                entry.update(raw_mae_measured=float(np.abs(raw-test.observed_proportion).mean()),
                    calibrated_mae_measured=float(np.abs(calibrated-test.observed_proportion).mean()),
                    raw_mae_reference_scale=float(np.abs(raw-processed_truth).mean()),
                    calibrated_mae_reference_scale=float(np.abs(calibrated-processed_truth).mean()),
                    scale_factor=heldout_map["scaling_info"].get("scale_factor", 1))
            except ValueError as exc:
                entry.update(status="not_estimable", note=str(exc))
            diagnostics.append(entry)
        print(f"Calibration: {target} | {len(part):,} tiles | {part.individual_id.nunique()} individuals", flush=True)
    bundle = dict(recipe=RECIPE, reference_url=REFERENCE_URL, reference_commit=REFERENCE_COMMIT,
        calibration_code_sha256=sha256(__file__),
        model_type="Combined", source_prediction_sha256=sha256(predictions),
        source_provenance_sha256=sha256(provenance), model_contracts=contracts, tables=tables,
        fitting_scope="Pooled raw Xenium Combined LOIO predictions; all included individuals",
        diagnostics="Each calibration fit excludes its evaluated individual. Base LOIO models are reused; this is not nested model refitting.",
        interpretation="Reference-scaled enrichment scores; not absolute measured cell fractions")
    output.mkdir(parents=True, exist_ok=False)
    (output / "calibration_bundle.json").write_text(json.dumps(bundle, indent=2, allow_nan=False)+"\n")
    pd.DataFrame(counts).to_csv(output / "calibration_summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "individual_excluded_diagnostics.csv", index=False)
    return output / "calibration_bundle.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(build_bundle(args.predictions, args.provenance, args.output_dir))
