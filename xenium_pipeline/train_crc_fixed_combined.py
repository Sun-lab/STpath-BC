"""Combined LOIO with the fixed, fold-averaged CRC gain selection procedure.

Source checkpoints are read only. Outputs live under crc_fixed_selection_runs.
Selection is shared across outer folds and is not fully nested.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import os
import re

import numpy as np
import pandas as pd

from . import stpath_xgboost as sx

CRC_COMMIT = "dc7bfcf75c65488e6d7951af74504325e14fd883"
SELECTION_METHOD = "crc_loio_mean_gain_union_zero_fill_top30pct_count"


def rank_fold_gains(gains: list[dict[str, float]], width: int) -> pd.Series:
    """Match CRC concat/fillna(0)/mean/sort, including pandas tie ordering."""
    if not gains:
        raise ValueError("No fold gains available")
    for fold in gains:
        for key, value in fold.items():
            if not re.fullmatch(r"f\d+", key) or not 0 <= int(key[1:]) < width:
                raise ValueError(f"Invalid gain feature: {key}")
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid gain: {key}={value}")
    table = pd.concat(
        [pd.DataFrame.from_dict(fold, orient="index", columns=[i])
         for i, fold in enumerate(gains)], axis=1,
    )
    return table.fillna(0).mean(axis=1).sort_values(ascending=False)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection_fingerprint(provenance):
    scientific = {key: value for key, value in provenance.items() if key != "selection_sha256"}
    scientific["training_config"] = sx.configuration_fingerprint(sx.TrainingConfig(**provenance["training_config"]))
    return hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()


def prepare_selection(targets, cell_type, source, output, config):
    """Audit every expected partition and freeze one selection per encoder."""
    signature = sx.training_data_fingerprint(targets, cell_type, config)
    folds = sx.leave_one_group_out(targets.individual_id.astype(str))
    gains = {name: [] for name in sx.MODEL_NAMES}
    source_records, coverage = [], []
    widths = {}
    xgb = sx._load_xgboost()
    for number, (heldout, train, test) in enumerate(folds, 1):
        fold = source / f"outer_fold_{number:03d}__heldout_{sx.safe_name(heldout)}"
        skipped = fold / "skipped.json"
        if skipped.is_file() and not (fold / "complete.json").is_file():
            metadata = json.loads(skipped.read_text())
            if (metadata.get("checkpoint_signature") != signature
                    or metadata.get("status") != "skipped_all_zero"
                    or metadata.get("outer_fold") != number
                    or metadata.get("n_test_rows") != len(test)
                    or metadata.get("held_out_individual") != heldout
                    or metadata.get("test_indices_checksum") != sx._array_checksum(test.astype(np.int64))
                    or not np.all(targets[f"proportion__{cell_type}"].to_numpy()[test] == 0)):
                raise ValueError(f"Incompatible skipped fold: {skipped}")
            coverage.extend({"outer_fold": number, "held_out_individual": heldout,
                             "encoder": name, "status": "skipped_all_zero", "error": ""}
                            for name in sx.MODEL_NAMES)
            continue
        for name in sx.MODEL_NAMES:
            model = fold / "outer" / "models" / f"{name}.ubj"
            metadata_path = fold / "outer" / "model_metadata" / f"{name}.json"
            row = {"outer_fold": number, "held_out_individual": heldout,
                   "encoder": name, "status": "missing", "error": ""}
            try:
                if not model.is_file() or not metadata_path.is_file():
                    raise FileNotFoundError(f"Missing model or metadata: {model}")
                metadata = json.loads(metadata_path.read_text())
                expected = {
                    "checkpoint_signature": signature, "model_type": name,
                    "training_config_fingerprint": sx.configuration_fingerprint(config),
                    "n_train_rows": len(train), "n_test_rows": len(test),
                    "train_indices_checksum": sx._array_checksum(train.astype(np.int64)),
                    "test_indices_checksum": sx._array_checksum(test.astype(np.int64)),
                }
                mismatch = [key for key, value in expected.items() if metadata.get(key) != value]
                if mismatch:
                    raise ValueError(f"Incompatible {metadata_path}: {mismatch}")
                booster = xgb.Booster()
                booster.load_model(str(model))
                width = int(booster.num_features())
                if width != metadata["input_dimension"] or widths.get(name, width) != width:
                    raise ValueError(f"Inconsistent feature dimension: {model}")
                widths[name] = width
                fold_gains = booster.get_score(importance_type="gain")
                rank_fold_gains([fold_gains], width)
                gains[name].append(fold_gains)
                source_records.append({"outer_fold": number, "encoder": name,
                                       "model": str(model), "sha256": file_hash(model),
                                       "gain_dictionary": fold_gains})
                row["status"] = "available"
            except (OSError, ValueError, KeyError) as exc:
                row["error"] = str(exc)
                row["status"] = "missing" if isinstance(exc, FileNotFoundError) else "invalid"
            coverage.append(row)
    coverage = pd.DataFrame(coverage)
    sx._atomic_csv(coverage, output / "source_fold_coverage.csv")
    unresolved = coverage[~coverage.status.isin(["available", "skipped_all_zero"])]
    if not unresolved.empty:
        raise ValueError(f"{len(unresolved)} missing/invalid encoder folds; see {output / 'source_fold_coverage.csv'}")
    selected, ranking_rows = {}, []
    for name in sx.MODEL_NAMES:
        if not gains[name]:
            raise ValueError(f"No available source folds for {name}")
        ranked = rank_fold_gains(gains[name], widths[name])
        count = round(0.30 * len(ranked))
        if count == 0:
            raise ValueError(f"CRC selects zero {name} features from {len(ranked)} ranked keys")
        selected[name] = [int(key[1:]) for key in ranked.index[:count]]
        ranking_rows.extend({"encoder": name, "feature_name": key, "feature_index": int(key[1:]),
                             "rank": rank, "mean_gain": float(value), "selected": rank <= count,
                             "n_source_folds": len(gains[name])}
                            for rank, (key, value) in enumerate(ranked.items(), 1))
    provenance = {
        "selection_method": SELECTION_METHOD, "crc_source_commit": CRC_COMMIT,
        "cell_type": cell_type, "source_checkpoint_root": str(source),
        "source_training_data_fingerprint": signature,
        "training_config": asdict(config), "source_models": source_records,
        "source_fold_coverage": coverage.to_dict(orient="records"),
        "foundation_feature_dimensions": widths, "selected_feature_indices": selected,
        "evaluation_note": "Gain averaged across single-encoder LOIO folds; fixed selection shared by every Combined fold. Not fully nested feature selection.",
    }
    # Runtime thread changes do not change selection or invalidate resumes.
    selection_hash = selection_fingerprint(provenance)
    provenance["selection_sha256"] = selection_hash
    saved = output / "fixed_selection.json"
    if saved.is_file() and json.loads(saved.read_text())["selection_sha256"] != selection_hash:
        raise ValueError("Source models or selection changed; choose a fresh run-id")
    sx._atomic_json(provenance, saved)
    sx._atomic_csv(pd.DataFrame(ranking_rows), output / "feature_ranking.csv")
    return provenance, coverage


def evaluate_combined(matrices, targets, cell_type, config, provenance, coverage, output,
                      *, outer_fold=None, finalize_only=False, recipe="stpath_brca_training_compatible_v1"):
    widths = sx.validate_feature_matrices(matrices, len(targets), model_names=sx.MODEL_NAMES)
    if widths != provenance["foundation_feature_dimensions"]:
        raise ValueError("Embedding dimensions differ from source models")
    selections = provenance["selected_feature_indices"]
    target = targets[f"proportion__{cell_type}"].to_numpy(dtype=float)
    if not np.isfinite(target).all() or (target < 0).any() or (target > 1).any():
        raise ValueError("Target proportions must be finite and in [0, 1]")
    folds = sx.leave_one_group_out(targets.individual_id.astype(str))
    if outer_fold is not None and not 1 <= outer_fold <= len(folds):
        raise ValueError(f"outer-fold must be between 1 and {len(folds)}")
    signature = provenance["selection_sha256"]
    statuses, tables, metrics = [], [], []
    for number, (heldout, train, test) in enumerate(folds, 1):
        state = "skipped_all_zero" if coverage.loc[coverage.outer_fold.eq(number), "status"].eq("skipped_all_zero").all() else "pending"
        folder = output / "checkpoints" / f"outer_fold_{number:03d}__heldout_{sx.safe_name(heldout)}"
        metadata = folder / "model_metadata" / "Combined.json"
        should_train = not finalize_only and (outer_fold is None or outer_fold == number)
        if state != "skipped_all_zero" and (should_train or (outer_fold is None and metadata.is_file())):
            prediction, _, _ = sx._fit_or_load_checkpointed_model(
                "Combined", sx.concatenate_selected_features(matrices, selections, train),
                target[train], sx.concatenate_selected_features(matrices, selections, test), config,
                train_indices=train, test_indices=test, checkpoint_dir=folder,
                checkpoint_signature=signature, overwrite=False, select_features=False,
                context=f"{cell_type} CRC fixed Combined fold {number}/{len(folds)}",
            )
            table = targets.iloc[test][["tile_id", "dataset", "sample", "individual_id"]].copy()
            table["cell_type"], table["outer_fold"] = cell_type, number
            table["observed_proportion"] = target[test]
            table["predicted_raw__Combined"] = prediction
            rows, baseline = sx._metric_rows_for_fold(cell_type, number, heldout, target, train, test,
                                                     {"Combined": prediction}, {}, calibrate=False)
            table["predicted_raw__TrainingMeanBaseline"] = baseline
            tables.append(table)
            metrics.extend(rows)
            state = "complete"
        statuses.append({"outer_fold": number, "held_out_individual": heldout,
                         "status": state, "n_test_rows": len(test)})
    status = pd.DataFrame(statuses)
    # Array workers write only their own status; aggregation is a separate step.
    if outer_fold is not None:
        sx._atomic_csv(status.loc[status.outer_fold.eq(outer_fold)],
                       output / "checkpoints" / f"fold_{outer_fold:03d}_status.csv")
        return
    sx._atomic_csv(status, output / "fold_status.csv")
    if tables:
        predictions = pd.concat(tables, ignore_index=True)
        sx._atomic_csv(predictions, output / "loio_predictions.csv.gz")
        metric_table = pd.DataFrame(metrics)
        sx._atomic_csv(metric_table, output / "fold_metrics.csv")
        sx._atomic_csv(sx.summarize_fold_metrics(metric_table), output / "metric_summary.csv")
    if status.status.eq("pending").any():
        raise ValueError("Unresolved Combined folds; see fold_status.csv")
    all_rows = np.arange(len(targets))
    combined = sx.concatenate_selected_features(matrices, selections, all_rows)
    sx._fit_or_load_checkpointed_model(
        "Combined", combined, target, combined, config, train_indices=all_rows,
        test_indices=all_rows, checkpoint_dir=output / "checkpoints" / "final",
        checkpoint_signature=signature, overwrite=False, select_features=False,
        context=f"{cell_type} CRC fixed full-cohort Combined",
    )
    sx._atomic_copy(output / "checkpoints" / "final" / "models" / "Combined.ubj",
                    output / "models" / "Combined.ubj")
    sx._atomic_json({"cell_type": cell_type, "model_type": "Combined",
                     "selection_method": SELECTION_METHOD, "selection_sha256": signature,
                     "input_dimension": combined.shape[1], "foundation_feature_dimensions": widths,
                     "selected_feature_indices": selections, "encoder_order": list(sx.MODEL_NAMES),
                     "extraction_recipe": recipe, "xgboost_parameters": sx.xgboost_parameters(config),
                     "num_boost_round": config.num_boost_round,
                     "training_tile_id_checksum": sx.tile_id_checksum(targets.tile_id),
                     "evaluation_note": provenance["evaluation_note"]},
                    output / "model_metadata" / "Combined.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(os.environ.get("XENIUM_ROOT", Path(__file__).resolve().parents[1])))
    parser.add_argument("--reference-run-dir", type=Path, required=True)
    parser.add_argument("--target-table", type=Path, required=True)
    parser.add_argument("--cell-type", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--feature-recipe", default="stpath_brca_training_compatible_v1")
    parser.add_argument("--nthread", type=int, default=5)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--outer-fold", type=int)
    mode.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args(argv)
    root = args.project_root.expanduser().resolve()
    if sx.safe_name(args.run_id) != args.run_id:
        raise ValueError("run-id must be a simple directory name")
    reference = sx.resolve_project_path(args.reference_run_dir, root).resolve()
    source = reference / sx.safe_name(args.cell_type) / "checkpoints"
    original = sx.TrainingConfig(**json.loads((reference / "run_metadata.json").read_text())["training_config"])
    if original.calibrate or original.num_boost_round != 800 or original.seed != 42 or original.feature_fraction != 0.30:
        raise ValueError("Reference must use raw regression, 800 rounds, seed 42, fraction 0.30")
    if original.resnet50_only:
        raise ValueError("Reference must contain the five pathology encoders")
    targets, parse_mode = sx.load_checkpoint_compatible_targets(
        sx.resolve_project_path(args.target_table, root), args.cell_type, source, original)
    output = (root / "xgboost_training" / "crc_fixed_selection_runs" / args.run_id / sx.safe_name(args.cell_type)).resolve()
    if output == reference or reference in output.parents or output in reference.parents:
        raise ValueError("Output must be separate from reference")
    config = replace(original, runtime_nthread=args.nthread,
                     combined_runtime_nthread=args.nthread, parallel_model_jobs=1)
    sx.validate_cpu_budget(config)
    selection_path = output / "fixed_selection.json"
    if args.prepare_only or not selection_path.is_file():
        if args.outer_fold is not None or args.finalize_only:
            raise ValueError("Run --prepare-only once before array training or finalization")
        provenance, coverage = prepare_selection(targets, args.cell_type, source, output, original)
    else:
        provenance = json.loads(selection_path.read_text())
        if selection_fingerprint(provenance) != provenance["selection_sha256"]:
            raise ValueError("Fixed selection provenance failed its integrity check")
        if (provenance["source_training_data_fingerprint"] != sx.training_data_fingerprint(targets, args.cell_type, original)
                or provenance["source_checkpoint_root"] != str(source)):
            raise ValueError("Prepared selection does not match the reference and targets")
        coverage = pd.DataFrame(provenance["source_fold_coverage"])
    sx.log(f"Fixed selection prepared ({parse_mode}); source coverage:\n{coverage.to_string(index=False)}")
    if args.prepare_only:
        return 0
    feature_root = sx.resolve_project_path(args.feature_root, root) if args.feature_root else root / "foundation_features" / args.feature_recipe
    matrices = sx.load_feature_matrices(targets, feature_root, expected_recipe_version=args.feature_recipe,
                                       model_names=sx.MODEL_NAMES, show_progress=True)
    evaluate_combined(matrices, targets, args.cell_type, config, provenance, coverage, output,
                      outer_fold=args.outer_fold, finalize_only=args.finalize_only, recipe=args.feature_recipe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
