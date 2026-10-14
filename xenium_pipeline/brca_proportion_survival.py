"""Five-cell-type proportion analyses using endpoint-specific clinical eligibility."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from lifelines.statistics import multivariate_logrank_test
from scipy.stats import false_discovery_control
from xenium_pipeline import brca_survival_models as models
from xenium_pipeline.storage import clinical_file

FEATURES = ['tumor_cells', 'stromal_cells', 'pan_apc_b', 't_cells', 'normal_epithelial_cells']
NAMES = ['Tumor Cells', 'Stromal Cells', 'pan-APC + B', 'T Cells', 'Normal Epithelial Cells']
COHORT_POLICY = 'endpoint_available_unadjusted_clinical_complete_adjusted'


def proportion_cohorts(data):
    return {endpoint: {
        'unadjusted': models.endpoint_rows(data, endpoint, FEATURES),
        'adjusted': models.endpoint_rows(data, endpoint, ['age', 'stage', 'subtype', *FEATURES]),
    } for endpoint in ['PFI', 'OS']}


def prepare(model_dir, output, *, patient_table=None):
    model_dir, output = Path(model_dir), Path(output)
    patient_table = Path(patient_table) if patient_table is not None else model_dir.parent / 'patient_analysis_table.csv'
    full = pd.read_csv(patient_table)
    if not full.patient_id.is_unique or full[FEATURES].isna().any().any():
        raise ValueError('Provide unique patients with all five calibrated prediction scores.')
    inputs = {name: clinical_file(model_dir.parent, name) for name in
              ['analysis_table.csv', 'cox_patient_ids.csv', 'analysis_config.json']}
    selected = pd.read_csv(inputs['analysis_table.csv'])
    reference = selected.set_index('patient_id')[FEATURES]
    scores = full.set_index('patient_id').reindex(reference.index)[FEATURES]
    if not np.allclose(scores, reference, rtol=0, atol=1e-12):
        raise ValueError('Full-cohort predictions differ from the saved distance analysis.')
    saved_members = pd.read_csv(inputs['cox_patient_ids.csv'])
    cohorts = proportion_cohorts(full)
    fits, rows, membership = {}, [], []
    for endpoint, adjusted_events in [('PFI', 62), ('OS', 60)]:
        adjusted = cohorts[endpoint]['adjusted']
        reference_id = f'{endpoint}__adjusted_distance__pan_apc_b_to_t_cells_median_bin'
        if set(adjusted.patient_id) != set(saved_members.loc[saved_members.model_id.eq(reference_id), 'patient_id']):
            raise ValueError(f'{endpoint}: adjusted proportion and distance cohorts differ.')
        for model, adjusters in [('unadjusted', []), ('adjusted', ['age', 'subtype', 'stage'])]:
            cohort = cohorts[endpoint][model]
            expected = (424, 66) if model == 'unadjusted' else (413, adjusted_events)
            observed = (len(cohort), int(cohort[f'{endpoint}_event'].sum()))
            if observed != expected:
                raise ValueError(f'{endpoint}/{model}: expected patients/events {expected}, found {observed}.')
            membership.extend(dict(patient_id=p, endpoint=endpoint, model=model) for p in cohort.patient_id)
            for feature, name in zip(FEATURES, NAMES):
                fit = models.cox(cohort, endpoint, [feature], adjust=adjusters, increments={feature: .1})
                model_id = f'{endpoint}__{model}__{feature}'
                fits[model_id] = fit
                if fit['summary']['status'] != 'ok':
                    raise ValueError(f'{model_id}: {fit["summary"]}')
                estimate = fit['coefficients'].loc[fit['coefficients'].variable.eq(feature)].iloc[0]
                rows.append(dict(endpoint=endpoint, model=model, cell_type=name,
                    **{key: fit['summary'][key] for key in ['n_patients', 'n_events', 'n_parameters', 'events_per_parameter', 'status']},
                    **{key: float(estimate[key]) for key in ['increment', 'HR', 'lower95', 'upper95', 'p_value']},
                    formula=' + '.join([feature, *adjusters])))
    props = pd.DataFrame(rows)
    props['q_BH'] = props.groupby(['endpoint', 'model']).p_value.transform(
        lambda p: false_discovery_control(p.to_numpy()))

    cut_rows, groups, km_rows = [], full[['patient_id']].copy(), []
    for kind, suffix in [('median', '_median_group'), ('tertiles', '_tertile_group')]:
        for feature, name in zip(FEATURES, NAMES):
            group, cuts, status = models.quantile_groups(full[feature], kind)
            if status != 'ok':
                raise ValueError(f'{feature}/{kind}: {status}')
            groups[feature + suffix] = group.astype(str).replace({'Medium': 'Middle'})
            cut_rows.append(dict(feature=feature, cell_type=name, kind=kind,
                cutpoints=json.dumps(cuts.tolist()), n_cutpoint_patients=len(full)))
            for endpoint in ['PFI', 'OS']:
                cohort = cohorts[endpoint]['unadjusted']
                labels = groups.set_index('patient_id').loc[cohort.patient_id, feature + suffix]
                test = multivariate_logrank_test(cohort[f'{endpoint}_time'], labels.to_numpy(), cohort[f'{endpoint}_event'])
                km_rows.append(dict(endpoint=endpoint, cell_type=name, comparison=kind,
                    n_patients=len(cohort), n_events=int(cohort[f'{endpoint}_event'].sum()),
                    p_value=float(test.p_value), status='ok'))
    km = pd.DataFrame(km_rows)
    km['q_BH'] = km.groupby(['endpoint', 'comparison']).p_value.transform(
        lambda p: false_discovery_control(p.to_numpy()))
    cuts = pd.DataFrame(cut_rows)
    output.mkdir(parents=True, exist_ok=True)
    props.to_csv(output / 'cox_results.csv', index=False)
    models.save_cox(fits, output, 'proportion_cox')
    pd.DataFrame(membership).to_csv(output / 'endpoint_cohort_membership.csv', index=False)
    km.to_csv(output / 'km_logrank_results.csv', index=False)
    cuts.to_csv(output / 'proportion_cutpoints.csv', index=False)
    groups.to_csv(output / 'proportion_group_membership.csv', index=False)
    sources = [patient_table, *inputs.values(), Path(__file__).resolve()]
    metadata = dict(source_run=model_dir.parent.name, cohort_policy=COHORT_POLICY,
        n_cutpoint_patients=len(full), endpoint_cohorts={endpoint: {
            model: dict(n_patients=len(frame), n_events=int(frame[f'{endpoint}_event'].sum()))
            for model, frame in by_model.items()} for endpoint, by_model in cohorts.items()},
        proportion_HR_increment=.1, adjustment=['age', 'AJCC stage', 'PAM50 including Normal-like'],
        source_table_preserved=True, saved_distance_models_refitted=False, proportion_models_fitted=20,
        cutpoint_rule='Type-7 quantiles on all scored patients; fixed across endpoints',
        BH_rule='Five cell types separately per endpoint and Cox model set or KM grouping method',
        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (output / 'analysis_provenance.json').write_text(json.dumps(metadata, indent=2) + chr(10))
    return props, full, groups, cuts, km
