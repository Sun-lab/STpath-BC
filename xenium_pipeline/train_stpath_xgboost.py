"""Command-line runner for restartable STpath-style XGBoost training."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
import os
from pathlib import Path
from typing import Sequence

from . import stpath_xgboost as sx


DEFAULT_PROJECT_ROOT = Path(
    os.environ.get("XENIUM_ROOT", str(Path(__file__).resolve().parents[1]))
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one restartable STpath-style XGBoost cell-type model."
    )
    parser.add_argument("--cell-type", required=True)
    parser.add_argument("--target-table", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument(
        "--feature-recipe", default="stpath_brca_training_compatible_v1"
    )
    parser.add_argument("--parallel-model-jobs", type=int)
    parser.add_argument("--checkpoint-nthread", type=int, default=5)
    parser.add_argument("--nthread", type=int)
    parser.add_argument("--combined-nthread", type=int)
    encoders = parser.add_mutually_exclusive_group()
    encoders.add_argument("--resnet50-only", action="store_true",
                          help="Load and train only ResNet50, excluding Combined.")
    parser.add_argument("--reference-run-dir", type=Path,
                        help="Reuse a completed run’s scientific configuration and exact target-table parsing.")
    encoders.add_argument("--include-resnet50", action="store_true",
                        help="Train the ImageNet ResNet50 baseline alongside the five pathology encoders and Combined.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-all-zero-test-folds", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--outer-fold",
        type=int,
        help="Train exactly one LOIO fold and write only its checkpoint directory.",
    )
    mode.add_argument(
        "--finalize-only",
        action="store_true",
        help="Require resolved folds, then aggregate outputs and train final models.",
    )
    return parser.parse_args(argv)


def build_training_config(args: argparse.Namespace) -> sx.TrainingConfig:
    available = sx.available_cpu_count()
    if args.parallel_model_jobs is None:
        args.parallel_model_jobs = 1
    max_jobs = 1 if args.resnet50_only else (len(sx.ENCODER_NAMES) if args.include_resnet50 else len(sx.MODEL_NAMES))
    if not 1 <= args.parallel_model_jobs <= max_jobs:
        raise ValueError(
            f"parallel model jobs must be between 1 and {max_jobs}"
        )
    if available is None and (args.nthread is None or args.combined_nthread is None):
        raise RuntimeError(
            "CPU allocation could not be detected; pass --nthread and --combined-nthread"
        )
    runtime_nthread = (
        args.nthread
        if args.nthread is not None
        else max(1, int(available) // args.parallel_model_jobs)
    )
    combined_nthread = (
        args.combined_nthread
        if args.combined_nthread is not None
        else int(available)
    )
    config = sx.TrainingConfig(
        resnet50_only=args.resnet50_only,
        calibrate=False,
        include_resnet50=args.include_resnet50,
        nthread=args.checkpoint_nthread,
        runtime_nthread=runtime_nthread,
        combined_runtime_nthread=combined_nthread,
        parallel_model_jobs=args.parallel_model_jobs,
    )
    if args.reference_run_dir is not None:
        root = args.project_root.expanduser().resolve()
        reference = sx.resolve_project_path(args.reference_run_dir, root)
        metadata = json.loads((reference / "run_metadata.json").read_text())
        original = sx.TrainingConfig(**metadata["training_config"])
        config = replace(original, include_resnet50=args.include_resnet50,
                         resnet50_only=args.resnet50_only,
                         parallel_model_jobs=args.parallel_model_jobs,
                         runtime_nthread=runtime_nthread,
                         combined_runtime_nthread=combined_nthread)
    return config


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    target_path = sx.resolve_project_path(args.target_table, project_root)
    feature_root = (
        sx.resolve_project_path(args.feature_root, project_root)
        if args.feature_root is not None
        else project_root / "foundation_features" / args.feature_recipe
    )
    run_root = project_root / "xgboost_training" / "runs" / args.run_id
    checkpoint_root = run_root / sx.safe_name(args.cell_type) / "checkpoints"
    config = build_training_config(args)
    sx.validate_cpu_budget(config)

    sx.log(
        f"starting {args.cell_type}; models={list(config.model_names)}; encoder stage="
        f"{config.parallel_model_jobs} jobs x {config.effective_nthread} threads; "
        f"Combined={config.effective_combined_nthread} threads"
    )
    if args.reference_run_dir is not None:
        reference = sx.resolve_project_path(args.reference_run_dir, project_root)
        if reference == run_root:
            raise ValueError("Use a fresh run-id separate from the completed reference run")
        original = sx.TrainingConfig(**json.loads(
            (reference / "run_metadata.json").read_text()
        )["training_config"])
        reference_checkpoints = reference / sx.safe_name(args.cell_type) / "checkpoints"
        if not any(reference_checkpoints.glob("outer_fold_*/complete.json")):
            raise ValueError("Reference run has no completed outer folds for this cell type")
        tile_targets, parse_mode = sx.load_checkpoint_compatible_targets(
            target_path, args.cell_type, reference_checkpoints, original
        )
        if args.outer_fold is not None:
            reference_folds = list(reference_checkpoints.glob(
                f"outer_fold_{args.outer_fold:03d}__heldout_*"
            ))
            if len(reference_folds) != 1:
                raise ValueError("Reference run does not uniquely identify the requested fold")
            reference_fold = reference_folds[0]
            if (reference_fold / "complete.json").is_file():
                # Include a zero-target fold when the original comparison evaluated it.
                args.include_all_zero_test_folds = True
            elif not (reference_fold / "skipped.json").is_file():
                raise ValueError("Requested reference fold is neither complete nor intentionally skipped")
    else:
        tile_targets, parse_mode = sx.load_checkpoint_compatible_targets(
            target_path, args.cell_type, checkpoint_root, config
        )
    sx.log(f"loaded {len(tile_targets):,} tile targets using {parse_mode} parsing")
    eligibility = sx.summarize_target_eligibility(
        tile_targets, [args.cell_type], config
    )
    print(eligibility.to_string(index=False), flush=True)
    if not bool(eligibility.loc[0, "eligible"]):
        sx.log(f"target is ineligible: {eligibility.loc[0, 'exclusion_reason']}")
        return 2

    if args.outer_fold is not None or args.finalize_only:
        _metrics, _predictions, fold_status = sx.collect_completed_fold_metrics(
            tile_targets,
            args.cell_type,
            checkpoint_root,
            config,
            outer_folds=(args.outer_fold,) if args.outer_fold is not None else None,
        )

    if args.outer_fold is not None:
        selected = fold_status.loc[fold_status["outer_fold"].eq(args.outer_fold)]
        if selected.empty:
            raise ValueError(
                f"outer fold {args.outer_fold} does not exist; "
                f"available folds are 1-{len(fold_status)}"
            )
        current_status = str(selected.iloc[0]["status"])
        if current_status == "invalid":
            raise ValueError(str(selected.iloc[0]["error"]))
        if current_status == "complete" and not args.overwrite:
            sx.log(f"outer fold {args.outer_fold} is already complete")
            return 0
        if (
            current_status == "skipped_all_zero"
            and not args.include_all_zero_test_folds
            and not args.overwrite
        ):
            sx.log(f"outer fold {args.outer_fold} is already intentionally skipped")
            return 0
        if (
            not args.include_all_zero_test_folds
            and sx.skip_outer_fold_if_all_zero(
                tile_targets,
                args.cell_type,
                args.outer_fold,
                checkpoint_root,
                config,
            )
        ):
            return 0

    if args.finalize_only:
        accepted = {"complete"}
        if not args.include_all_zero_test_folds:
            accepted.add("skipped_all_zero")
        unresolved = fold_status.loc[~fold_status["status"].isin(accepted)]
        if not unresolved.empty:
            details = ", ".join(
                f"{int(row.outer_fold)} ({row.status})"
                for row in unresolved.itertuples(index=False)
            )
            sx.log(f"finalization stopped; unresolved outer folds: {details}")
            return 3

    sx.log(f"loading foundation features from {feature_root}")
    feature_matrices = sx.load_feature_matrices(
        tile_targets,
        feature_root,
        expected_recipe_version=args.feature_recipe,
        show_progress=True,
        model_names=config.encoder_names,
    )
    sx.log("foundation feature loading completed")

    if args.outer_fold is not None:
        evaluation = sx.evaluate_cell_type(
            feature_matrices,
            tile_targets,
            args.cell_type,
            config,
            checkpoint_dir=checkpoint_root,
            overwrite=args.overwrite,
            skip_all_zero_test_folds=not args.include_all_zero_test_folds,
            outer_folds=(args.outer_fold,),
        )
        print(evaluation["fold_status"].to_string(index=False), flush=True)
        return 0

    summary = sx.run_training(
        feature_matrices,
        tile_targets,
        eligibility,
        run_root,
        config,
        overwrite=args.overwrite,
        skip_all_zero_test_folds=not args.include_all_zero_test_folds,
        finalize_with_skipped_folds=args.finalize_only,
        force_finalize=args.finalize_only,
    )
    print(summary.to_string(index=False), flush=True)
    failed = summary["status"].isin({"failed", "ineligible"}).any()
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
