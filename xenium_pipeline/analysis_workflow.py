from __future__ import annotations
from pathlib import Path
import os
CODE_ROOT = Path(__file__).resolve().parents[1]
from xenium_pipeline.storage import load_layout, data_path, curated_table

def compare_importance(root):
    from pathlib import Path
    import os
    import pandas as pd
    from xenium_pipeline.workflow_output import display, Image
    from xenium_pipeline import feature_importance_comparison as importance
    ROOT = Path(root).expanduser().resolve()
    OUTPUT = ROOT / 'feature_importance_comparison/manuscript_figure6b_xenium'
    COAD_RESULTS = Path(os.environ.get('STPATH_COAD_VISIUM_RESULTS', ROOT / 'external_models/COAD'))
    TUMOR_MODELS = ROOT / 'xgboost_training/runs' / load_layout(ROOT)['targets']['tumor_cells']['run_id'] / 'Tumor_Cells'
    REFERENCE_PATHS = {encoder: COAD_RESULTS / f'Cancer Cells_{encoder}_individual_level_ratio100/xgboost_results_individual_level.pt' for encoder in importance.FIGURE7_ENCODERS}
    display(pd.DataFrame([dict(encoder=encoder, path=str(path), available=path.is_file()) for encoder, path in REFERENCE_PATHS.items()]))
    import json
    import hashlib
    import numpy as np
    OUTPUT.mkdir(parents=True, exist_ok=True)
    xenium, xenium_folds = importance.load_xenium_figure7(TUMOR_MODELS, encoders=importance.FIGURE7_ENCODERS, aggregation='present_keys')
    parts, inputs = ([], [])
    for encoder, path in REFERENCE_PATHS.items():
        coad, folds = importance.load_figure7_reference(path, encoder, aggregation='present_keys')
        parts.append(pd.DataFrame(dict(encoder=encoder, feature_index=np.arange(len(coad)), brca_xenium_percent=xenium[encoder], coad_visium_percent=coad)))
        inputs.append(dict(encoder=encoder, path=str(path), n_folds=folds, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    profiles = pd.concat(parts, ignore_index=True)
    metrics = importance.plot_figure7_scatter(profiles, OUTPUT, encoders=importance.FIGURE7_ENCODERS, views=['coad_visium_vs_brca_xenium'], manuscript_panel=True, save_pdf=True)
    profiles.to_csv(OUTPUT / 'figure6b_feature_profiles.csv.gz', index=False)
    metrics.to_csv(OUTPUT / 'figure6b_correlations.csv', index=False)
    (OUTPUT / 'figure6b_provenance.json').write_text(json.dumps(dict(aggregation='present_keys', full_embedding_vectors=True, encoders=list(importance.FIGURE7_ENCODERS), xenium_models=str(TUMOR_MODELS), xenium_folds=xenium_folds, coad_inputs=inputs, x_dataset='BRCA Xenium', y_dataset='COAD Visium', x_cutoff_percent=1.0, y_cutoff_percent=0.5), indent=2) + '\n')
    display(metrics)
    display(Image(filename=str(OUTPUT / 'Figure_6B.png')))
    return OUTPUT

def evaluate_transfer(root):
    from pathlib import Path
    import os
    import sys
    import importlib
    ROOT = Path(root).expanduser().resolve()
    import numpy as np
    import pandas as pd
    from xenium_pipeline.workflow_output import display, Image
    from xenium_pipeline import cross_platform_evaluation as cx
    FEATURE_ROOT = ROOT / 'foundation_features' / cx.RECIPE
    VISIUM_MODEL_DIR = ROOT / 'cross_platform_evaluation' / 'visium_models' / cx.VISIUM_COMMIT
    VISIUM_WORKBOOK = ROOT / 'preprint' / 'media-4.xlsx'
    OUTPUT = ROOT / 'cross_platform_evaluation' / 'xenium_transfer'
    RUN_EVALUATION = True
    DOWNLOAD_MISSING_VISIUM_MODELS = True
    REQUIRE_COMPLETE_FOLDS = True
    NTHREAD = 4
    BOOTSTRAP_REPLICATES = 2000
    SEED = 42
    EXCLUDE_INDIVIDUALS = []
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print('Project:', ROOT)
    print('Results:', OUTPUT)
    availability = cx.audit_inputs(ROOT, FEATURE_ROOT, VISIUM_MODEL_DIR, VISIUM_WORKBOOK)
    availability.to_csv(OUTPUT / 'input_availability.csv', index=False)
    display(availability.drop(columns=[col for col in availability if col.endswith('_path')]))
    mapping_path = curated_table(ROOT, 'sample_individual_mapping.csv')
    if mapping_path.is_file():
        mapping = pd.read_csv(mapping_path)
        display(mapping.groupby('dataset', sort=False).agg(samples=('sample', 'size'), groups=('individual_id', 'nunique')))
        display(mapping.groupby(['individual_id_source', 'review_status'], dropna=False).size().rename('n_samples').reset_index())
        mapping.to_csv(OUTPUT / 'individual_mapping_snapshot.csv', index=False)
    else:
        print('Missing mapping:', mapping_path)
    required = ['targets_available', 'run_metadata_available', 'checkpoints_available', 'feature_root_available']
    inputs_ready = bool(availability[required].all().all() and mapping_path.is_file())
    if not inputs_ready:
        print('Production inputs are missing at ROOT. Set ROOT/FEATURE_ROOT to the cluster copy before evaluation.')
        display(availability.loc[~availability[required].all(axis=1), ['cell_type', 'targets_path', 'run_metadata_path', 'checkpoints_path']])
    results = None
    if not inputs_ready:
        raise FileNotFoundError('Required production inputs are missing; inspect the inventory above.')
    results = cx.run_evaluation(ROOT, FEATURE_ROOT, VISIUM_MODEL_DIR, VISIUM_WORKBOOK, OUTPUT, require_complete_folds=REQUIRE_COMPLETE_FOLDS, download_missing=DOWNLOAD_MISSING_VISIUM_MODELS, nthread=NTHREAD, exclude_individuals=EXCLUDE_INDIVIDUALS, bootstrap_replicates=BOOTSTRAP_REPLICATES, seed=SEED)
    display(results['fold_coverage'].groupby(['cell_type', 'status'], sort=False).size().rename('n_folds').reset_index())
    display(results['paired_coverage'].groupby('cell_type', sort=False).agg(n_individuals=('individual_id', 'nunique'), n_tiles=('n_tiles', 'sum')))
    if results is not None:
        columns = ['cell_type', 'model', 'n_individuals', 'n_tiles', 'mean_mae_pp', 'mean_rmse_pp', 'mean_bias_pp', 'mean_pearson', 'n_defined_pearson', 'mean_spearman', 'n_defined_spearman']
        display(results['summary_metrics'][columns].round(4))
        display(results['paired_differences'].round(4))
        display(results['individual_metrics'][['cell_type', 'individual_id', 'model', 'n', 'n_positive_tiles', 'mae_pp', 'pearson', 'correlation_status', 'outside_unit_interval_fraction']].round(4))
    if results is not None:
        figure_paths = cx.plot_evaluation(results, OUTPUT)
        for path in figure_paths:
            if path.suffix == '.png':
                display(Image(filename=str(path)))
    import json
    saved_run = OUTPUT / 'provenance.json'
    if not saved_run.is_file():
        raise FileNotFoundError('Missing provenance.json. Run the transfer evaluation or provide its saved output folder.')
    run_status = json.loads(saved_run.read_text())['status']
    if run_status not in {'complete', 'partial'}:
        raise ValueError(f'Saved evaluation status is {run_status!r}; finish the evaluation before plotting these files.')
    print('Saved evaluation status:', run_status)
    if run_status == 'partial':
        print('Partial evaluation: the figure includes only the available matched tiles/groups.')
    scatter_columns = ['cell_type', 'tile_id', 'observed_proportion', 'predicted_visium', 'predicted_xenium_oof']
    scatter_data = pd.read_csv(OUTPUT / 'paired_predictions.csv.gz', usecols=scatter_columns)
    scatter_paths = cx.plot_tile_scatter(scatter_data, OUTPUT)
    for path in scatter_paths:
        print('Saved:', path)
        if path.suffix == '.png':
            display(Image(filename=str(path)))
    if results is not None:
        by_study = results['individual_metrics'].groupby(['dataset', 'cell_type', 'model'], sort=False).agg(n_individuals=('individual_id', 'nunique'), n_tiles=('n', 'sum'), mean_mae_pp=('mae_pp', 'mean'), mean_rmse_pp=('rmse_pp', 'mean'), mean_pearson=('pearson', 'mean'), n_defined_pearson=('pearson', 'count'), mean_spearman=('spearman', 'mean'), mean_bias_pp=('bias_pp', 'mean')).reset_index()
        by_study.to_csv(OUTPUT / 'study_stratified_metrics.csv', index=False)
        display(by_study.round(4))
    return OUTPUT

def analyze_survival(root):
    from pathlib import Path
    from datetime import datetime, timezone
    import os
    PROJECT_ROOT = Path(root).expanduser().resolve()
    PREDICTION_ROOT = PROJECT_ROOT / 'tcga_brca_prediction'
    COVERAGE_EXPECTED_COHORT_DIR = data_path(PROJECT_ROOT, 'prediction_cohort')
    COVERAGE_RUN_DIRS = data_path(PROJECT_ROOT, 'prediction_runs')
    COMPATIBILITY_REPORT = data_path(PROJECT_ROOT, 'prediction_audit')
    CLINICAL_CSV = PROJECT_ROOT / 'tcga_brca_survival_inputs' / 'clinical_survival.csv'
    SUBTYPE_CSV = PROJECT_ROOT / 'tcga_brca_survival_inputs' / 'pam50_subtypes.csv'
    INPUT_PROVENANCE = PROJECT_ROOT / 'tcga_brca_survival_inputs' / 'input_provenance.json'
    CLINICAL_COLUMNS = dict(id='Sample_ID', age='age_at_initial_pathologic_diagnosis', stage='ajcc_pathologic_tumor_stage', PFI_time='PFI.time', PFI_event='PFI', OS_time='OS.time', OS_event='OS', subtype=None)
    SUBTYPE_COLUMNS = dict(id='patient', subtype='subtype')
    CELL_TYPES = ['T Cells', 'Stromal Cells', 'pan-APC + B', 'Tumor Cells', 'Normal Epithelial Cells']
    FEATURE_COLUMNS = ['t_cells', 'stromal_cells', 'pan_apc_b', 'tumor_cells', 'normal_epithelial_cells']
    ENDPOINTS = ['PFI', 'OS']
    SLIDE_AGGREGATION = 'equal_slide'
    PRIMARY_TUMOR_ONLY = True
    ALLOW_PARTIAL_COHORT = False
    HR_INCREMENT = 0.1
    OUTPUT_DIR = PROJECT_ROOT / 'tcga_brca_survival' / ('brca_reference_calibrated_75um_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    print('Coverage runs:', *COVERAGE_RUN_DIRS, sep='\n  ')
    print('Survival input: first valid completed result per slide, in the run order above.')
    print('Results:', OUTPUT_DIR)
    CALIBRATION_BUNDLE = PROJECT_ROOT / 'tcga_brca_survival_inputs/reference_calibration_v1/calibration_bundle.json'
    SPATIAL_SCORE_POLICY = 'reference_quantile_normalize'
    SPATIAL_DISTANCE_UNITS = 'um'
    SPATIAL_THRESHOLDS = dict(tumor_cells=0.2, stromal_cells=0.3, pan_apc_b=0.3, t_cells=0.3)
    SPATIAL_CACHE_DIR = PROJECT_ROOT / 'tcga_brca_reference_calibration_cache'
    RUN_SPATIAL_THRESHOLD_SENSITIVITY = True
    SPATIAL_SENSITIVITY_THRESHOLDS = {'tumor_15pct_immune_30pct_sensitivity': dict(SPATIAL_THRESHOLDS, tumor_cells=0.15)}
    import hashlib
    import importlib
    import importlib.metadata
    import json
    import shutil
    import sys
    import numpy as np
    import pandas as pd
    from xenium_pipeline.workflow_output import display
    from xenium_pipeline import brca_survival_models as models
    from xenium_pipeline import brca_proportion_survival as proportions
    print('Python', sys.version.split()[0], '| lifelines', importlib.metadata.version('lifelines'))
    from functools import partial
    from xenium_pipeline.brca_survival_workflow import NA_VALUES, STAGES, SUBTYPES, GROUPS, read_table, read_json, sha256, require_columns, patient_barcode, number, collapse_patients, stage_group, subtype_group, write_table as _write_table
    write_table = partial(_write_table, output_dir=OUTPUT_DIR)
    from xenium_pipeline.brca_survival_workflow import read_compatibility_report, check_prediction_coverage, load_cohort as _load_cohort
    load_cohort = partial(_load_cohort, output_dir=OUTPUT_DIR)
    from xenium_pipeline.brca_survival_workflow import aggregate_slides, load_clinical, endpoint_membership, tertiles
    OUTPUT_DIR = Path(OUTPUT_DIR)
    if OUTPUT_DIR.exists():
        raise FileExistsError('OUTPUT_DIR already exists; rerun the command to create a new result folder.')
    OUTPUT_DIR.mkdir(parents=True)
    coverage_check = check_prediction_coverage(COVERAGE_EXPECTED_COHORT_DIR, COVERAGE_RUN_DIRS, CELL_TYPES, COMPATIBILITY_REPORT)
    for key, filename in [('coverage', 'prediction_coverage.csv'), ('missing', 'slides_missing_predictions.csv'), ('by_run', 'prediction_coverage_by_run.csv')]:
        write_table(coverage_check[key], filename)
    (OUTPUT_DIR / 'prediction_coverage_summary.json').write_text(json.dumps(coverage_check['summary'], indent=2, allow_nan=False) + '\n')
    display(pd.DataFrame([{k: coverage_check['summary'][k] for k in ['n_expected_slides', 'n_slides_with_predictions', 'n_slides_missing_predictions', 'all_slides_have_predictions']}]))
    covered_slides = coverage_check['coverage'].query("status == 'complete'")
    display(covered_slides.groupby(['crop_size_um', 'source_run']).size().rename('slides').reset_index())
    if coverage_check['missing'].empty:
        print('Every expected slide has a validated completed prediction.')
    else:
        display(coverage_check['missing'][['slide_id', 'detail']])
        print('Review slides_missing_predictions.csv for the full missing-slide list.')
    print('Coverage reports:', OUTPUT_DIR.resolve())
    (OUTPUT_DIR / 'figures').mkdir(exist_ok=True)
    cohort = load_cohort(coverage_check, CELL_TYPES, ALLOW_PARTIAL_COHORT)
    display(cohort['status'].status.value_counts().rename('slides'))
    slides = cohort['slides'].copy()
    slides['sample_type'] = slides.slide_id.str[13:15]
    excluded = slides.loc[PRIMARY_TUMOR_ONLY & slides.sample_type.ne('01'), ['slide_id', 'patient_id', 'sample_type']]
    write_table(excluded.drop_duplicates(), 'excluded_nonprimary_slides.csv')
    if PRIMARY_TUMOR_ONLY:
        slides = slides.loc[slides.sample_type.eq('01')].copy()
    if slides.empty:
        raise ValueError('No primary-tumor slides remain.')
    raw_slides = slides.copy()
    write_table(raw_slides, 'slide_predictions_raw.csv')
    if SLIDE_AGGREGATION != 'equal_slide':
        raise ValueError('This calibrated analysis uses equal_slide aggregation.')
    import importlib
    from xenium_pipeline import brca_reference_calibration as calibration
    from xenium_pipeline import brca_spatial as spatial
    calibration_bundle = calibration.load_or_build_bundle(CALIBRATION_BUNDLE, PROJECT_ROOT / 'tcga_brca_survival_inputs/tile_loio_reference')
    calibrated_spatial = spatial.compute_cohort(raw_slides, OUTPUT_DIR / 'calibrated_spatial_main', SPATIAL_CACHE_DIR, score_policy=SPATIAL_SCORE_POLICY, thresholds=SPATIAL_THRESHOLDS, calibration_bundle=calibration_bundle, model_project_root=PROJECT_ROOT, cache_thresholds=[dict(SPATIAL_THRESHOLDS, tumor_cells=0.15)])
    slides = pd.read_csv(calibrated_spatial['slide_predictions'])
    write_table(slides, 'slide_predictions.csv')
    crop_counts = slides[['slide_id', 'patient_id', 'crop_size_um', 'source_run']].drop_duplicates().groupby(['crop_size_um', 'source_run'], as_index=False).agg(n_slides=('slide_id', 'nunique'), n_patients=('patient_id', 'nunique'))
    write_table(crop_counts, 'survival_crop_size_counts.csv')
    display(crop_counts)
    patient_long = aggregate_slides(slides, SLIDE_AGGREGATION)
    write_table(patient_long, 'patient_predictions_long.csv')
    patients = patient_long.pivot(index='patient_id', columns='cell_type', values='score')[CELL_TYPES]
    patients.columns = FEATURE_COLUMNS
    patients = patients.reset_index()
    if patients.patient_id.duplicated().any() or patients[FEATURE_COLUMNS].isna().any().any():
        raise ValueError('Incomplete or duplicate patient predictions.')
    patient_info = patient_long.loc[patient_long.cell_type.eq(CELL_TYPES[0]), ['patient_id', 'n_slides', 'n_tiles', 'slide_ids', 'crop_sizes_um', 'mixed_crop_sizes', 'source_runs']]
    patients = patients.merge(patient_info, on='patient_id', validate='one_to_one')
    display(pd.DataFrame([dict(n_slides=slides.slide_id.nunique(), n_patients=len(patients), patients_with_multiple_slides=int(patients.n_slides.gt(1).sum()))]))
    display(patients.loc[patients.n_slides.gt(1), ['patient_id', 'n_slides', 'n_tiles', 'slide_ids', 'crop_sizes_um', 'mixed_crop_sizes', 'source_runs']])
    qc_rows = []
    for cell_type in CELL_TYPES:
        s, p = (slides.loc[slides.cell_type.eq(cell_type)], patient_long.loc[patient_long.cell_type.eq(cell_type)])
        qc_rows.append(dict(cell_type=cell_type, n_slides=len(s), n_tiles=int(s.n_sampled_tiles.sum()), fraction_tiles_outside_0_1=np.average(s.fraction_outside_0_1, weights=s.n_sampled_tiles), n_patients_outside_0_1=int((~p.score.between(0, 1)).sum()), minimum=p.score.min(), maximum=p.score.max()))
    score_qc = pd.DataFrame(qc_rows)
    write_table(score_qc, 'prediction_range_qc.csv')
    display(score_qc)
    clinical = load_clinical(CLINICAL_CSV, CLINICAL_COLUMNS, SUBTYPE_CSV, SUBTYPE_COLUMNS)
    write_table(clinical, 'clinical_patient_table.csv')
    dat = patients.merge(clinical, on='patient_id', how='left', validate='one_to_one', indicator=True)
    dat['clinical_matched'] = dat.pop('_merge').eq('both')
    write_table(dat[['patient_id', 'clinical_matched']], 'clinical_matching.csv')
    write_table(dat, 'patient_analysis_table.csv')
    display(dat[['stage_raw', 'stage']].drop_duplicates())
    display(dat.subtype.value_counts(dropna=False))
    EXTENDED_OUTPUT_DIR = OUTPUT_DIR
    DISTANCE_INPUT_CSV = calibrated_spatial[SPATIAL_DISTANCE_UNITS]['csv']
    DISTANCE_METADATA_JSON = calibrated_spatial[SPATIAL_DISTANCE_UNITS]['metadata']
    data, distance_audit, distance_metadata = models.load_distances(dat.copy(), slides, DISTANCE_INPUT_CSV, DISTANCE_METADATA_JSON)
    if distance_metadata['method'].get('calibration_sha256') != calibration.fingerprint(calibration_bundle):
        raise ValueError('Spatial inputs use a different calibration map.')
    if distance_metadata['method'].get('thresholds') != SPATIAL_THRESHOLDS:
        raise ValueError('Distance metadata cutoffs differ from the selected final method.')
    distance_audit.to_csv(OUTPUT_DIR / 'distance_input_matching.csv', index=False)
    results = models.run_survival_models(data, OUTPUT_DIR / 'survival_models', distance_metadata=distance_metadata, show=True)
    display(results['cox_status'])
    display(results['km_results'])
    proportion_results = proportions.prepare(results['output'], OUTPUT_DIR / 'proportion_models',
        patient_table=OUTPUT_DIR / 'patient_analysis_table.csv')
    display(proportion_results[0])
    display(proportion_results[-1])
    from xenium_pipeline import brca_spatial as spatial
    threshold_rows, distance_km_tables, distance_cox_tables = ([], [], [])
    threshold_sets = {'tumor_20pct_immune_30pct_main': SPATIAL_THRESHOLDS, **SPATIAL_SENSITIVITY_THRESHOLDS}
    for threshold_set, cutoffs in threshold_sets.items():
        if threshold_set == 'tumor_20pct_immune_30pct_main':
            result = results
            distance_csv, method_json = (DISTANCE_INPUT_CSV, DISTANCE_METADATA_JSON)
        else:
            sensitivity_dir = EXTENDED_OUTPUT_DIR / 'spatial_threshold_sensitivity' / threshold_set
            print('Spatial threshold set:', threshold_set, cutoffs, flush=True)
            outputs = spatial.compute_cohort(raw_slides, sensitivity_dir / 'spatial', SPATIAL_CACHE_DIR, score_policy=SPATIAL_SCORE_POLICY, thresholds=cutoffs, calibration_bundle=calibration_bundle, model_project_root=PROJECT_ROOT)
            distance_csv = outputs[SPATIAL_DISTANCE_UNITS]['csv']
            method_json = outputs[SPATIAL_DISTANCE_UNITS]['metadata']
            sensitivity_data, matching, method = models.load_distances(dat.copy(), slides, distance_csv, method_json)
            matching.to_csv(sensitivity_dir / 'distance_input_matching.csv', index=False)
            result = models.run_survival_models(sensitivity_data, sensitivity_dir / 'survival_models', distance_metadata=method, show=False)
        if distance_csv is None:
            raise ValueError('Threshold sensitivity requires the main spatial inputs and models plots first.')
        threshold_rows.append(dict(threshold_set=threshold_set, **cutoffs, distance_csv=str(distance_csv), metadata_json=str(method_json), analysis_dir=str(result['output'])))
        km = result['km_results'].loc[result['km_results'].family.eq('median_distances')].copy()
        km.insert(0, 'threshold_set', threshold_set)
        distance_km_tables.append(km)
        for filename in ('cox_model_status.csv', 'cox_coefficients.csv'):
            table = pd.read_csv(result['output'] / filename)
            table = table.loc[table.model_id.str.contains('_to_', regex=False)].copy()
            table.insert(0, 'threshold_set', threshold_set)
            table.insert(1, 'table_kind', filename)
            distance_cox_tables.append(table)
    comparison_dir = EXTENDED_OUTPUT_DIR / 'spatial_threshold_sensitivity'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    threshold_table = pd.DataFrame(threshold_rows)
    threshold_table.to_csv(comparison_dir / 'threshold_sets.csv', index=False)
    km_comparison = pd.concat(distance_km_tables, ignore_index=True)
    km_comparison.to_csv(comparison_dir / 'distance_KM_comparison.csv', index=False)
    pd.concat(distance_cox_tables, ignore_index=True).to_csv(comparison_dir / 'distance_cox_comparison.csv', index=False)
    display(threshold_table)
    display(km_comparison)
    print('Spatial cutoff sensitivity outputs:', comparison_dir)
    input_paths = [Path(p) for p in [CLINICAL_CSV, SUBTYPE_CSV, CALIBRATION_BUNDLE] if p is not None]
    write_table(pd.DataFrame([dict(path=str(p), sha256=sha256(p)) for p in input_paths]), 'analysis_input_hashes.csv')
    if Path(INPUT_PROVENANCE).is_file():
        shutil.copyfile(INPUT_PROVENANCE, OUTPUT_DIR / 'source_input_provenance.json')
    run_config = dict(prediction_coverage=coverage_check['summary'], cell_types=CELL_TYPES, prediction_cutoffs=SPATIAL_THRESHOLDS, sensitivity_cutoffs=SPATIAL_SENSITIVITY_THRESHOLDS, score_policy=SPATIAL_SCORE_POLICY, patient_aggregation=SLIDE_AGGREGATION, calibration=calibration_bundle['recipe'], clipping=True, across_cell_type_normalization=True, calibration_fingerprint=calibration.fingerprint(calibration_bundle), distance_units=SPATIAL_DISTANCE_UNITS, normal_epithelial_distances=False, adjustment=['age', 'stage', 'PAM50 including Normal-like'], proportion_HR_increment=0.1, distance_HR_increment=1.0, proportion_cohorts='Endpoint-available unadjusted; clinical-complete adjusted', distance_cohort='Stage/subtype eligible; complete-case endpoint cohort', proportion_cutpoints='Type-7 quantiles of all scored patients, fixed across endpoints', distance_cutpoints='Type-7 quantiles of stage/subtype-eligible patients, fixed across endpoints', spatial_models='Continuous and median Low/High; 12 directed pairs; source-weighted distance; self-matches allowed', calibration_code_sha256=sha256(CODE_ROOT / 'xenium_pipeline/brca_reference_calibration.py'), spatial_code_sha256=sha256(CODE_ROOT / 'xenium_pipeline/brca_spatial.py'))
    (OUTPUT_DIR / 'run_config.json').write_text(json.dumps(run_config, indent=2) + '\n')
    print('Final clinical results:', OUTPUT_DIR)
    print('Pass this run directory to Scripts_for_Analysis/FigureS15_S16.py for S15 and S16.')
    return OUTPUT_DIR

def validate_spatial(root):
    from pathlib import Path
    from datetime import datetime, timezone
    import os
    import json
    import pandas as pd
    from xenium_pipeline.workflow_output import display
    from xenium_pipeline import brca_local_validation as validation
    from xenium_pipeline import brca_reference_calibration as calibration
    PROJECT_ROOT = Path(root).expanduser().resolve()
    CALIBRATED_RESULTS = Path(os.environ.get('STPATH_BRCA_ANALYSIS_RUN', data_path(PROJECT_ROOT, 'clinical_run')))
    OUTPUT_DIR = PROJECT_ROOT / 'tcga_brca_local_validation' / ('final_calibrated_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    LOIO_INPUT = PROJECT_ROOT / 'tcga_brca_survival_inputs/tile_loio_reference/paired_predictions.csv.gz'
    CACHE_DIR = PROJECT_ROOT / 'tcga_brca_reference_calibration_cache'
    BUNDLE = calibration.load_bundle(PROJECT_ROOT / 'tcga_brca_survival_inputs/reference_calibration_v1/calibration_bundle.json')
    SPATIAL_RESULTS = CALIBRATED_RESULTS / 'calibrated_spatial_main'
    if not SPATIAL_RESULTS.is_dir():
        SPATIAL_RESULTS = CALIBRATED_RESULTS / 'spatial'
    slides = validation.local_slides(PROJECT_ROOT, SPATIAL_RESULTS / 'slide_predictions_calibrated.csv')
    print('Final calibrated validation:', OUTPUT_DIR)
    loio = validation.loio_validation(LOIO_INPUT, OUTPUT_DIR / 'xenium_loio_spatial', calibrated=True)
    display(loio['summary'][['threshold_set', 'source', 'target', 'metric', 'n_paired', 'MAE_grid', 'bias_grid', 'spearman']])
    qc = validation.calibrated_cache_qc(slides, CALIBRATED_RESULTS, CACHE_DIR, BUNDLE, OUTPUT_DIR / 'tcga_spatial_qc', spatial_dir=SPATIAL_RESULTS)
    display(qc['summary'].loc[qc['summary'].metric.eq('median'), ['threshold_set', 'source', 'target', 'n_valid', 'n_zero', 'fraction_zero', 'n_unique', 'median_source_overlap']])
    validation.plot_results(qc['summary'], loio['summary'], OUTPUT_DIR)
    return OUTPUT_DIR

def generate_reference_figures(root):
    from pathlib import Path
    import importlib
    import json
    import pandas as pd
    from xenium_pipeline.workflow_output import display, Image, FileLink
    from xenium_pipeline import supplementary_s9 as s9
    S9_ROOT = Path(root).expanduser().resolve()
    S9_TRAINING_SOURCES = {'reference': (data_path(S9_ROOT, 'reference_labels'), data_path(S9_ROOT, 'reference_annotations')), 'epithelial': (data_path(S9_ROOT, 'epithelial_labels'), data_path(S9_ROOT, 'epithelial_annotations'))}
    S9_TARGET_SOURCES = {spec['cell_type']: spec['annotation_source'] for spec in load_layout(S9_ROOT)['targets'].values()}
    S9_DISPLAY_SOURCE = 'epithelial'
    S9_SMOOTH_MUSCLE_AS_UNCERTAIN = True
    S9_LABEL_DIR, S9_ANNOTATION_DIR = S9_TRAINING_SOURCES[S9_DISPLAY_SOURCE]
    S9_COHORT_PATH = CODE_ROOT / 'tables/cohort_samples.csv'
    S9_TARGET_REFERENCE_PATHS = [S9_ROOT / 'xgboost_training' / f"tile_targets_{spec['stem']}.csv.gz" for spec in load_layout(S9_ROOT)['targets'].values()]
    S9_OUTPUT_DIR = S9_ROOT / 'supplementary_figures/S9'
    S9_B_DATASETS = ['10x', 'Liu_2025', 'Janesick_2023', 'Janesick_2025', 'Bilous_2026']
    S9_B_TARGETS = list(s9.TARGET_COLORS)[:-1]
    S9_B_RULES = dict(side_tiles=10, min_retained_fraction=0.75, min_cells=100, featured_range=(0.25, 0.6), other_min_fraction=0.1, min_other_types=2, preferred_other_fraction=0.05)
    S9_B_ALLOW_CACHED_REASSIGNMENT = True
    S9_B_REUSE_LOCATIONS = True
    S9_B_SEARCH_DIR = data_path(S9_ROOT, 'reference_crops')
    S9_B_FEATURED_SAMPLE_PREFERENCES = {'Stromal Cells': ('Liu_2025', 'Xenium_V1_FFPE_Human_Breast_IDC_Big_2')}
    S9_B_HE_STYLES = {('Janesick_2025', 'Human_Breast_Biomarkers_S1_Mid'): dict(he_grid_style=dict(color='white', alpha=0.9), scale_bar_color='white', scale_bar_um=100), ('Bilous_2026', 'B5_GSM9509171'): dict(he_grid_style=dict(color='white', alpha=0.9), scale_bar_color='white', scale_bar_um=100)}
    S9_B_ROIS = {}
    S9_B_IMAGE_PATHS = {}
    S9_B_CELL_TABLE_PATHS = {}
    S9_B_BOUNDARY_PATHS = {}
    S9_B_COORDINATE_PATHS = {}
    S9_tiles, S9_cohort = s9.load_training_references(S9_TARGET_REFERENCE_PATHS, S9_COHORT_PATH)
    if len(S9_cohort) != 35 or len(S9_tiles) != 300904:
        raise ValueError('The manuscript requires 35 samples and 300,904 retained tiles.')
    if set(S9_cohort.dataset) != set(S9_B_DATASETS):
        raise ValueError('The S9B dataset list must match Table S2.')
    S9_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    S9_SAMPLE_LABEL_DIRS = {}
    S9_MODEL_AUDITS = {}

    def s9_load_cells(dataset, sample):
        cells, audit, model_audit = s9.load_training_cells(S9_TRAINING_SOURCES, S9_TARGET_SOURCES, S9_tiles, dataset, sample, display_source=S9_DISPLAY_SOURCE)
        snapshot_paths = dict(zip(S9_TARGET_SOURCES, map(str, S9_TARGET_REFERENCE_PATHS)))
        model_audit['target_snapshot_path'] = model_audit.model_target.map(snapshot_paths)
        S9_MODEL_AUDITS[dataset, sample] = model_audit
        S9_SAMPLE_LABEL_DIRS[dataset, sample] = Path(audit['label_dir'])
        return (cells, audit)
    s9_load_figure_cells = s9_load_cells
    print(f'S9 output: {S9_OUTPUT_DIR}')
    print(f'S9 bars, polygons and tile fractions use epithelial annotations: {S9_LABEL_DIR}')
    print('Model targets are validated against their annotation sources.')
    S9_composition_parts, S9_audits = ([], [])
    S9_MODEL_AUDITS.clear()
    S9_SAMPLE_LABEL_DIRS.clear()
    for row in S9_cohort.itertuples(index=False):
        print(f'S9 updated labels and target-specific checks: {row.dataset} / {row.sample}', flush=True)
        try:
            cells, audit = s9_load_cells(row.dataset, row.sample)
        except (ValueError, FileNotFoundError) as error:
            S9_audits.append(dict(dataset=row.dataset, sample=row.sample, status='failed', error=str(error)))
            continue
        audit['status'] = 'ok'
        part = s9.composition_table(cells).assign(dataset=row.dataset, sample=row.sample)
        S9_composition_parts.append(part)
        S9_audits.append(audit)
        del cells
    pd.DataFrame(S9_audits).to_csv(S9_OUTPUT_DIR / 'Figure_S9_annotation_audit.csv', index=False)
    if S9_MODEL_AUDITS:
        pd.concat(S9_MODEL_AUDITS.values(), ignore_index=True).to_csv(S9_OUTPUT_DIR / 'Figure_S9_model_target_audit.csv', index=False)
    S9_failures = [a for a in S9_audits if a.get('status') == 'failed']
    if S9_failures:
        for failure in S9_failures:
            print(f"\nFAILED: {failure['dataset']} / {failure['sample']}\n{failure['error']}", flush=True)
        first = S9_failures[0]
        raise ValueError(f"{len(S9_failures)} sample(s) failed training-source validation. First failure: {first['dataset']} / {first['sample']}\n{first['error']}\nFull audit: {S9_OUTPUT_DIR / 'Figure_S9_annotation_audit.csv'}. No new figures generated.")
    S9_composition = pd.concat(S9_composition_parts, ignore_index=True)
    S9_composition.to_csv(S9_OUTPUT_DIR / 'Figure_S9A_sample_composition.csv', index=False)
    pd.DataFrame(S9_audits).to_csv(S9_OUTPUT_DIR / 'Figure_S9_annotation_audit.csv', index=False)
    S9_metadata = dict(training_sources={v: list(map(str, p)) for v, p in S9_TRAINING_SOURCES.items()}, display_annotation_source=S9_DISPLAY_SOURCE, figure_display_mapping={'Smooth Muscle Cells': 'Uncertain'} if S9_SMOOTH_MUSCLE_AS_UNCERTAIN else {}, figure_display_composition='Figure_S9A_display_composition.csv', display_label_dir=str(S9_LABEL_DIR), model_target_annotation_sources=S9_TARGET_SOURCES, target_reference_paths=list(map(str, S9_TARGET_REFERENCE_PATHS)), reference_deck='xgboost_LOIO_results.pptx', cohort_path=str(S9_COHORT_PATH), n_samples=len(S9_cohort), n_retained_tiles=len(S9_tiles), denominator='All cells assigned to retained tiles; no five-target renormalization', validation_scope='Each model checked against its own source; updated display differences recorded', annotation_audit='Figure_S9_annotation_audit.csv', model_target_audit='Figure_S9_model_target_audit.csv', spatial_datasets=S9_B_DATASETS, spatial_targets=S9_B_TARGETS, spatial_selection_rules=S9_B_RULES)
    (S9_OUTPUT_DIR / 'Figure_S9_provenance.json').write_text(json.dumps(S9_metadata, indent=2))
    s9.figure_composition(S9_composition, smooth_muscle_as_uncertain=S9_SMOOTH_MUSCLE_AS_UNCERTAIN).to_csv(S9_OUTPUT_DIR / 'Figure_S9A_display_composition.csv', index=False)
    for path in s9.plot_composition(S9_composition, S9_cohort, S9_OUTPUT_DIR, smooth_muscle_as_uncertain=S9_SMOOTH_MUSCLE_AS_UNCERTAIN):
        if path.suffix == '.png':
            display(Image(filename=str(path), width=1200))
        display(FileLink(str(path)))
    display(FileLink(str(S9_OUTPUT_DIR / 'Figure_S9A_sample_composition.csv')))
    display(FileLink(str(S9_OUTPUT_DIR / 'Figure_S9_model_target_audit.csv')))
    from xenium_pipeline import tile_filter_review
    inventory = {(s.dataset, s.sample): s for s in tile_filter_review.discover_tile_filter_samples(S9_ROOT)}
    saved = pd.read_csv(S9_B_SEARCH_DIR / 'Figure_S9B_selected_examples.csv')
    if len(saved) != 5 or saved.dataset.nunique() != 5:
        raise ValueError('Need the five previously selected S9B crop coordinates.')
    saved['featured_target'] = saved.featured_target.replace({'pan-APCs': 'pan-APC + B'})
    saved_targets = set(saved.featured_target)
    if saved_targets != set(S9_B_TARGETS) or set(saved.dataset) != set(S9_B_DATASETS):
        raise ValueError('Saved locations must cover each dataset and featured target once.')
    candidate_path = S9_B_SEARCH_DIR / 'Figure_S9B_candidates.csv'
    locations = saved[['dataset', 'sample', 'featured_target', 'x0', 'y0', 'x1', 'y1']].copy()
    if candidate_path.is_file():
        old_candidates = pd.read_csv(candidate_path)
        old_candidates['featured_target'] = old_candidates.featured_target.replace({'pan-APCs': 'pan-APC + B'})
        for target, (dataset, sample) in S9_B_FEATURED_SAMPLE_PREFERENCES.items():
            extra = old_candidates.loc[old_candidates.featured_target.eq(target) & old_candidates.dataset.eq(dataset) & old_candidates['sample'].eq(sample), locations.columns]
            locations = pd.concat([locations, extra], ignore_index=True).drop_duplicates()

    def s9_recheck_cached_locations(sample_locations_table):
        parts, audits = ([], [])
        for (dataset, sample), sample_locations in sample_locations_table.groupby(['dataset', 'sample'], sort=False):
            cells, _ = s9_load_figure_cells(dataset, sample)
            geometry = S9_cohort.loc[S9_cohort.dataset.eq(dataset) & S9_cohort['sample'].eq(sample)].iloc[0]
            if int(geometry.tile_width_px) != int(geometry.tile_height_px):
                raise ValueError('Panel B requires square training tiles.')
            candidates, audit = s9.revalidate_saved_regions(cells, sample_locations, int(geometry.tile_width_px), S9_B_RULES)
            parts.append(candidates)
            audits.append(audit)
            del cells
        nonempty_parts = [part for part in parts if not part.empty]
        candidates = pd.concat(nonempty_parts, ignore_index=True) if nonempty_parts else parts[0].iloc[:0].copy()
        return (candidates, pd.concat(audits, ignore_index=True))
    S9_candidates, S9_candidate_audit = s9_recheck_cached_locations(locations)
    qualifying_pairs = set(S9_candidates[['dataset', 'featured_target']].itertuples(index=False, name=None))
    missing_pairs = set(saved[['dataset', 'featured_target']].itertuples(index=False, name=None)) - qualifying_pairs
    if missing_pairs and candidate_path.is_file():
        cache_keys = ['dataset', 'sample', 'featured_target', 'x0', 'y0', 'x1', 'y1']
        seen = set(S9_candidate_audit[cache_keys].itertuples(index=False, name=None))
        extra_locations = old_candidates.loc[[pair in missing_pairs for pair in old_candidates[['dataset', 'featured_target']].itertuples(index=False, name=None)], cache_keys].drop_duplicates()
        extra_locations = extra_locations.merge(S9_cohort[['dataset', 'sample']], on=['dataset', 'sample'], how='inner', validate='many_to_one')
        extra_locations = extra_locations.loc[[key not in seen for key in extra_locations[cache_keys].itertuples(index=False, name=None)]]
        if not extra_locations.empty:
            print(f'Rechecking {len(extra_locations)} additional saved crop location(s) for failed pairs; no window search.')
            extra_candidates, extra_audit = s9_recheck_cached_locations(extra_locations)
            if not extra_candidates.empty:
                S9_candidates = pd.concat([S9_candidates, extra_candidates], ignore_index=True) if not S9_candidates.empty else extra_candidates
            S9_candidate_audit = pd.concat([S9_candidate_audit, extra_audit], ignore_index=True)
    S9_reassigned_cached_pairs = False
    qualifying_pairs = set(S9_candidates[['dataset', 'featured_target']].itertuples(index=False, name=None))
    remaining_pairs = set(saved[['dataset', 'featured_target']].itertuples(index=False, name=None)) - qualifying_pairs
    if remaining_pairs and S9_B_ALLOW_CACHED_REASSIGNMENT:
        roi_keys = ['dataset', 'sample', 'x0', 'y0', 'x1', 'y1']
        all_locations = locations[roi_keys].copy()
        if candidate_path.is_file():
            all_locations = pd.concat([all_locations, old_candidates[roi_keys]], ignore_index=True)
        all_locations = all_locations.drop_duplicates().merge(S9_cohort[['dataset', 'sample']], on=['dataset', 'sample'], how='inner', validate='many_to_one')
        already_counted = set(S9_candidate_audit[roi_keys].itertuples(index=False, name=None))
        new_locations = all_locations.loc[[key not in already_counted for key in all_locations[roi_keys].itertuples(index=False, name=None)]].copy()
        if not new_locations.empty:
            new_locations['featured_target'] = 'Tumor Cells'
            print(f'Recounting {len(new_locations)} additional saved coordinates for reassignment; no new windows or model audits.')
            _, new_region_counts = s9_recheck_cached_locations(new_locations)
            S9_candidate_audit = pd.concat([S9_candidate_audit, new_region_counts], ignore_index=True)
        S9_candidates, S9_candidate_audit = s9.evaluate_updated_region_counts(S9_candidate_audit, S9_B_RULES)
        S9_reassigned_cached_pairs = True
        pair_availability = S9_candidates.groupby(['dataset', 'featured_target']).size().unstack(fill_value=0).reindex(index=S9_B_DATASETS, columns=S9_B_TARGETS, fill_value=0).fillna(0).astype(int)
        pair_availability.to_csv(S9_OUTPUT_DIR / 'Figure_S9B_cached_pair_availability.csv')
        display(pair_availability)
    revalidation_path = S9_OUTPUT_DIR / 'Figure_S9B_location_revalidation_audit.csv'
    S9_candidate_audit.to_csv(revalidation_path, index=False)
    failed_locations = S9_candidate_audit.loc[~S9_candidate_audit.qualifies]
    if not failed_locations.empty:
        print(failed_locations[['dataset', 'sample', 'featured_target', 'featured_fraction', 'n_cells', 'retained_tile_fraction', 'n_other_types_ge_min', 'failed_rules']].to_string(index=False))
        print(f'Failed rule details: {revalidation_path}')
    if S9_reassigned_cached_pairs:
        try:
            chosen_examples, cached_location_audit = s9.assign_revalidated_cached_examples(S9_candidates, saved, S9_B_DATASETS, S9_B_TARGETS)
        except ValueError as error:
            normal = S9_candidate_audit.loc[S9_candidate_audit.featured_target.eq('Normal Epithelial Cells')].sort_values('featured_fraction', ascending=False).head(10)
            raise ValueError('No complete five-dataset/five-target assignment among the rechecked saved coordinates. See Figure_S9B_cached_pair_availability.csv and Figure_S9B_location_revalidation_audit.csv. The annotation audit and full window search were not repeated. Highest cached Normal fractions:\n' + normal[['dataset', 'sample', 'featured_fraction', 'n_other_types_ge_min', 'failed_rules']].to_string(index=False)) from error
    else:
        chosen_examples, cached_location_audit = s9.select_revalidated_saved_examples(saved, S9_candidates, S9_candidate_audit)
    cached_location_audit['selection_phase'] = 'cached reassignment' if S9_reassigned_cached_pairs else 'cached fallback before explicit sample preferences'
    cached_location_audit.to_csv(S9_OUTPUT_DIR / 'Figure_S9B_cached_location_selection_audit.csv', index=False)
    display(cached_location_audit)
    print('Rechecked saved crop locations with epithelial labels; full search skipped.')
    search_source = S9_B_SEARCH_DIR
    if not S9_B_REUSE_LOCATIONS:
        chosen_examples = s9.assign_dataset_examples(S9_candidates, S9_B_DATASETS, S9_B_TARGETS)
    applicable_preferences = {target: preference for target, preference in S9_B_FEATURED_SAMPLE_PREFERENCES.items() if chosen_examples.loc[chosen_examples.featured_target.eq(target), 'dataset'].iloc[0] == preference[0]}
    chosen_examples, sample_preference_audit = s9.prefer_featured_samples(S9_candidates, chosen_examples, applicable_preferences)
    for target, (dataset, sample) in S9_B_FEATURED_SAMPLE_PREFERENCES.items():
        if target not in applicable_preferences:
            selected = chosen_examples.loc[chosen_examples.featured_target.eq(target)].iloc[0]
            sample_preference_audit = pd.concat([sample_preference_audit, pd.DataFrame([dict(featured_target=target, requested_dataset=dataset, requested_sample=sample, qualifying_saved_candidate=bool((S9_candidates.dataset.eq(dataset) & S9_candidates['sample'].eq(sample) & S9_candidates.featured_target.eq(target)).any()), selected_sample=selected['sample'], status=f'featured target reassigned to {selected.dataset}; preferred sample not used')])], ignore_index=True)
    sample_preference_audit.to_csv(S9_OUTPUT_DIR / 'Figure_S9B_sample_preference_audit.csv', index=False)
    display(sample_preference_audit)
    chosen_examples.to_csv(S9_OUTPUT_DIR / 'Figure_S9B_selected_examples.csv', index=False)
    display(chosen_examples[['dataset', 'sample', 'featured_target', 'featured_fraction', 'min_other_fraction', 'all_other_types_ge_preferred']])
    region_metadata = []
    for i, example in enumerate(chosen_examples.itertuples(index=False), 1):
        key = (example.dataset, example.sample)
        if key not in inventory:
            raise FileNotFoundError(f'No original tiling inventory for {key}.')
        cells, _ = s9_load_figure_cells(*key)
        geometry = S9_cohort.loc[S9_cohort.dataset.eq(key[0]) & S9_cohort['sample'].eq(key[1])].iloc[0]
        tile_size_px = int(geometry.tile_width_px)
        example_dir = S9_OUTPUT_DIR / 'polygon_composition_examples' / example.dataset / example.sample
        example_dir.mkdir(parents=True, exist_ok=True)
        auto_roi = tuple((int(getattr(example, c)) for c in ['x0', 'y0', 'x1', 'y1']))
        roi = s9.spatial_roi(cells, tile_size_px, S9_B_ROIS.get(key, auto_roi))
        selection_method = 'explicit region after H&E review' if key in S9_B_ROIS else 'saved crop coordinates; epithelial composition revalidated'
        for path in s9.export_spatial_example(S9_ROOT, cells, inventory[key], tile_size_px, example_dir, roi=roi, pixel_size_um=float(geometry.effective_pixel_size_um), image_path=S9_B_IMAGE_PATHS.get(key), coordinate_path=S9_B_COORDINATE_PATHS.get(key), side_tiles=S9_B_RULES['side_tiles'], selection_method=selection_method, featured_target=example.featured_target, composition_rules=S9_B_RULES, cell_table_path=S9_B_CELL_TABLE_PATHS.get(key), boundary_path=S9_B_BOUNDARY_PATHS.get(key), label_dir=S9_SAMPLE_LABEL_DIRS[key], annotation_source='epithelial', smooth_muscle_as_uncertain=S9_SMOOTH_MUSCLE_AS_UNCERTAIN, **S9_B_HE_STYLES.get(key, {})):
            if path.suffix == '.png':
                display(Image(filename=str(path), width=1400))
            display(FileLink(str(path)))
        metadata_path = example_dir / 'Figure_S9B_metadata.json'
        metadata = json.loads(metadata_path.read_text())
        metadata.update(cached_dataset_target_reassignment=bool(S9_reassigned_cached_pairs), selection_input_dir=str(search_source), reused_crop_locations=S9_B_REUSE_LOCATIONS, example_index=i, requested_side_tiles=S9_B_RULES['side_tiles'], candidate_quality_score=float(example.quality_score), candidate_retained_tile_fraction=float(example.retained_tile_fraction), candidate_all_other_types_ge_preferred=bool(example.all_other_types_ge_preferred))
        metadata_path.write_text(json.dumps(metadata, indent=2))
        region_metadata.append(metadata)
        polygon_info = metadata['polygon_metadata']
        print(f"All annotated region cell polygons: {polygon_info['n_cells_with_polygons']:,}/{polygon_info['n_region_cells']:,}; affine error {polygon_info['affine_max_error_px']:.4g}px")
        annotation_info = metadata['annotation_metadata']
        print(f"Annotated cells outside retained tiles: {annotation_info['n_annotated_region_cells_outside_retained_tiles']:,}")
        print(f"{key}: {example.featured_target} {metadata['featured_fraction']:.1%}; {metadata['n_retained_region_cells']:,} retained cells; ROI {metadata['roi_he_pixels']}")
        for suffix in ['cells.csv.gz', 'annotated_cells.csv.gz', 'tile_proportions.csv', 'cell_polygons.csv.gz', 'polygon_audit.csv', 'metadata.json']:
            display(FileLink(str(example_dir / f'Figure_S9B_{suffix}')))
        del cells
    summary_path = S9_OUTPUT_DIR / 'Figure_S9B_examples_summary.json'
    summary_path.write_text(json.dumps(region_metadata, indent=2))
    display(FileLink(str(summary_path)))
    from pathlib import Path
    import json
    import zipfile
    S9_FIGURE_DIR = Path(str(root / 'supplementary_figures/S9'))
    S9_DOWNLOAD_ZIP = S9_FIGURE_DIR / 'Figure_S9_source_panels.zip'
    summary_path = S9_FIGURE_DIR / 'Figure_S9B_examples_summary.json'
    examples = json.loads(summary_path.read_text())
    if len(examples) != 5 or len({(e['dataset'], e['sample']) for e in examples}) != 5:
        raise ValueError('Expected five completed S9B examples in the summary.')
    files = [(S9_FIGURE_DIR / f'Figure_S9A.{ext}', f'S9A/Figure_S9A.{ext}') for ext in ('pdf', 'png')]
    files.append((summary_path, 'S9B/Figure_S9B_examples_summary.json'))
    for name in ('Figure_S9A_sample_composition.csv', 'Figure_S9A_display_composition.csv', 'Figure_S9_annotation_audit.csv', 'Figure_S9_model_target_audit.csv', 'Figure_S9_provenance.json'):
        files.append((S9_FIGURE_DIR / name, f'provenance/{name}'))
    for name in ('Figure_S9B_location_revalidation_audit.csv', 'Figure_S9B_cached_pair_availability.csv', 'Figure_S9B_cached_location_selection_audit.csv', 'Figure_S9B_sample_preference_audit.csv', 'Figure_S9B_selected_examples.csv'):
        path = S9_FIGURE_DIR / name
        if path.is_file():
            files.append((path, f'provenance/{name}'))
    for i, example in enumerate(examples, 1):
        folder = S9_FIGURE_DIR / 'polygon_composition_examples' / example['dataset'] / example['sample']
        label = example['featured_target'].replace(' ', '_').replace('/', '_')
        for filename in ('Figure_S9B.pdf', 'Figure_S9B.png', 'Figure_S9B_metadata.json'):
            files.append((folder / filename, f"S9B/{i:02d}_{label}/{example['dataset']}__{example['sample']}/{filename}"))
    missing = [str(path) for path, _ in files if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing generated outputs:\n' + '\n'.join(missing))
    if not S9_DOWNLOAD_ZIP.parent.is_dir():
        raise FileNotFoundError(f'Output folder is unavailable: {S9_DOWNLOAD_ZIP.parent}')
    temporary_zip = S9_DOWNLOAD_ZIP.with_suffix('.tmp.zip')
    with zipfile.ZipFile(temporary_zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path, archive_name in files:
            archive.write(path, archive_name)
    with zipfile.ZipFile(temporary_zip) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP integrity check failed.')
    temporary_zip.replace(S9_DOWNLOAD_ZIP)
    print(f'Created {S9_DOWNLOAD_ZIP} ({S9_DOWNLOAD_ZIP.stat().st_size / 1024 ** 2:.1f} MB)')
    print('Contains S9A and the five completed S9B examples as PDF/PNG, with their metadata.')
    return S9_OUTPUT_DIR
