"""Patient-level Cox models and Kaplan–Meier analyses."""
from itertools import permutations
from pathlib import Path
import hashlib
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.exceptions import ConvergenceError, ConvergenceWarning
from lifelines.statistics import multivariate_logrank_test
from scipy.stats import false_discovery_control


PROPORTIONS = ["tumor_cells", "stromal_cells", "pan_apc_b", "t_cells"]
DISTANCES = [f"{a}_to_{b}_{stat}" for a, b in permutations(PROPORTIONS, 2)
             for stat in ["mean", "std", "median", "iqr"]]
MEDIANS = [c for c in DISTANCES if c.endswith("_median")]
LEVELS = {"stage": ["I", "II", "III", "IV"],
          "subtype": ["Basal", "Her2", "LumA", "LumB", "Normal"]}
LABELS = dict(tumor_cells="Tumor", stromal_cells="Stromal", pan_apc_b="pan-APC + B",
              t_cells="T cells", normal_epithelial_cells="Normal epithelial", age="Age")


def label(name):
    clinical = {"stage": "Stage", "subtype": "PAM50 subtype", "age": "Age", "age_quart": "Age quartiles"}
    if name in clinical:
        return clinical[name]
    for key in sorted(LABELS, key=len, reverse=True):
        name = name.replace(key, LABELS[key])
    return name.replace("_to_", " → ").replace("_", " ")


def validate_data(data):
    required = ["patient_id", "age", "stage", "subtype", *PROPORTIONS,
                "PFI_time", "PFI_event", "OS_time", "OS_event"]
    missing = set(required) - set(data)
    if missing:
        raise ValueError(f"Missing analysis columns: {sorted(missing)}")
    if data.patient_id.isna().any() or data.patient_id.duplicated().any():
        raise ValueError("Use exactly one observation per patient.")
    for col, levels in LEVELS.items():
        if not data[col].dropna().isin(levels).all():
            raise ValueError(f"Unexpected {col} categories.")
    for endpoint in ["PFI", "OS"]:
        if not data[f"{endpoint}_event"].dropna().isin([0, 1]).all():
            raise ValueError(f"{endpoint}_event must be 0 or 1.")
    numeric = ["age", *PROPORTIONS, "PFI_time", "PFI_event", "OS_time", "OS_event"]
    if np.isinf(data[numeric].to_numpy(dtype=float)).any():
        raise ValueError("Infinite clinical values or prediction scores.")


def endpoint_rows(data, endpoint, variables=()):
    columns = list(dict.fromkeys([f"{endpoint}_time", f"{endpoint}_event", *variables]))
    keep = data[columns].notna().all(axis=1) & data[f"{endpoint}_time"].gt(0)
    if "clinical_matched" in data:
        keep &= data.clinical_matched
    return data.loc[keep].copy()


def quantile_groups(values, kind):
    """R type-7 quantiles and right-closed cuts; never split tied values by rank."""
    values = pd.Series(values)
    probs, levels = {"tertiles": ([1/3, 2/3], ["Low", "Medium", "High"]),
                     "median": ([.5], ["Low", "High"]),
                     "quartiles": ([0, .25, .5, .75, 1], ["Q1", "Q2", "Q3", "Q4"])}[kind]
    # R quantile(type=7) uses 1-based ranks. pandas/numpy's percentile conversion
    # can place a cut one ULP below an observed boundary, changing cut() groups.
    ordered = np.sort(values.dropna().to_numpy(dtype=float))
    if len(ordered):
        rank = 1 + (len(ordered)-1)*np.asarray(probs)
        low, high = np.floor(rank).astype(int)-1, np.ceil(rank).astype(int)-1
        weight = rank-np.floor(rank)
        cuts = (1-weight)*ordered[low] + weight*ordered[high]
    else:
        cuts = np.full(len(probs), np.nan)
    status = "ok"
    if not np.isfinite(cuts).all() or len(np.unique(cuts)) != len(cuts):
        group = pd.Series(pd.Categorical([None]*len(values), categories=levels), index=values.index)
        status = "tied_or_missing_cutpoints"
    else:
        bounds = cuts if kind == "quartiles" else [-np.inf, *cuts, np.inf]
        group = pd.cut(values, bounds, labels=levels, right=True, include_lowest=True)
        if group.value_counts().eq(0).any():
            status = "empty_group"
    return group, cuts, status


def cox(data, endpoint, predictors, *, adjust=(), strata=None, penalizer=0, increments=None):
    """Fit an explicit design; smaller Newton steps avoid observed PFI overshoot.

    Statistical models and penalties are unchanged; full warnings are retained.
    """
    variables = list(dict.fromkeys([*predictors, *adjust, *([strata] if strata else [])]))
    frame = endpoint_rows(data, endpoint, variables)
    design = pd.DataFrame({"time": frame[f"{endpoint}_time"], "event": frame[f"{endpoint}_event"]})
    increments = increments or {}
    owners, references = {}, {}
    constant = []
    for variable in variables:
        if variable == strata:
            design[variable] = frame[variable].astype(str)
            continue
        x = frame[variable]
        if variable in LEVELS or isinstance(x.dtype, pd.CategoricalDtype):
            levels = LEVELS.get(variable, list(x.cat.categories) if isinstance(x.dtype, pd.CategoricalDtype) else [])
            observed = [level for level in levels if x.eq(level).any()]
            references[variable] = observed[0] if observed else None
            if len(observed) < 2:
                constant.append(variable)
            for level in observed[1:]:
                term = f"{variable}[{level}]"
                design[term] = x.eq(level).astype(float)
                owners[term] = variable
        else:
            design[variable] = pd.to_numeric(x, errors="raise") / increments.get(variable, 1)
            owners[variable] = variable
            if x.nunique() < 2:
                constant.append(variable)
    base = dict(endpoint=endpoint, predictors=" + ".join(predictors), adjusters=" + ".join(adjust),
                strata=strata or "", penalizer=penalizer, n_patients=len(frame),
                n_events=int(frame[f"{endpoint}_event"].sum()), n_parameters=len(owners),
                events_per_parameter=float(design.event.sum()/len(owners)) if owners else np.nan,
                reference_levels=json.dumps(references), status="not_estimable", note="")
    members = frame[["patient_id"]].copy()
    empty = pd.DataFrame(columns=["term", "variable", "HR", "lower95", "upper95", "p_value", "increment"])
    diagnostic = pd.DataFrame()
    def failed(note):
        return dict(summary=dict(base, note=str(note)), coefficients=empty, diagnostics=diagnostic, patients=members)
    if constant:
        return failed("Constant requested predictors: " + ", ".join(constant))
    if len(frame) < 3 or design.event.sum() == 0 or not owners:
        return failed("Insufficient patients, events, or predictors")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = CoxPHFitter(penalizer=penalizer, l1_ratio=0).fit(
                design, duration_col="time", event_col="event", strata=[strata] if strata else None,
                fit_options=dict(step_size=0.50, precision=1e-9, r_precision=1e-10, max_steps=100))
        diagnostic = fit.summary.rename_axis("term").reset_index()
        diagnostic["variable"] = diagnostic.term.map(owners)
        issues = [str(w.message) for w in caught
                  if issubclass(w.category, (ConvergenceWarning, RuntimeWarning))]
        if issues:
            return failed("; ".join(dict.fromkeys(issues)))
        coefficients = diagnostic.rename(columns={"exp(coef)": "HR", "exp(coef) lower 95%": "lower95",
                                                   "exp(coef) upper 95%": "upper95", "p": "p_value"}).copy()
        if not np.isfinite(coefficients[["HR", "lower95", "upper95", "p_value"]]).all().all():
            return failed("Non-finite coefficient inference")
        coefficients["increment"] = coefficients.variable.map(lambda x: increments.get(x, 1))
        base.update(status="ok", c_index_apparent=float(fit.concordance_index_))
        return dict(summary=base, coefficients=coefficients, diagnostics=diagnostic, patients=members)
    except (ConvergenceError, ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
        return failed(exc)


def save_cox(fits, output, prefix):
    output = Path(output)
    summaries, coefficients, diagnostics, membership = [], [], [], []
    for model_id, result in fits.items():
        summaries.append(dict(model_id=model_id, **result["summary"]))
        for rows, name in [(coefficients, "coefficients"), (diagnostics, "diagnostics"), (membership, "patients")]:
            if not result[name].empty:
                rows.append(result[name].assign(model_id=model_id))
    status = pd.DataFrame(summaries)
    status.to_csv(output / f"{prefix}_model_status.csv", index=False)
    for rows, suffix in [(coefficients, "coefficients"), (diagnostics, "diagnostic_estimates"), (membership, "patient_ids")]:
        table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["model_id"])
        table.to_csv(output / f"{prefix}_{suffix}.csv", index=False)
    return status


def clinical_diagnostics(data):
    """Counts in both endpoint-eligible and full-adjustment complete-case samples."""
    rows = []
    for endpoint in ["PFI", "OS"]:
        for sample, variables in [("endpoint_eligible", []), ("full_adjustment", ["age", "stage", "subtype"])]:
            frame = endpoint_rows(data, endpoint, variables)
            for variable in ["stage", "subtype", "crop_sizes_um"]:
                if variable not in frame:
                    continue
                for level, part in frame.groupby(variable, observed=True, dropna=False):
                    rows.append(dict(endpoint=endpoint, sample=sample, variable=variable, level=level,
                        n_patients=len(part), n_events=int(part[f"{endpoint}_event"].sum()),
                        n_censored=int(part[f"{endpoint}_event"].eq(0).sum())))
    return pd.DataFrame(rows)


def load_distances(data, slides, path, metadata_path):
    """Require all 48 recomputed columns, matching selected slide/crop/run keys."""
    meta = json.loads(Path(metadata_path).read_text())
    if meta.get("csv_sha256"):
        with Path(path).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != meta["csv_sha256"]:
                raise ValueError("Distance CSV hash disagrees with its metadata")
    for field in ["prediction_source", "distance_units", "method", "calibration"]:
        if not meta.get(field):
            raise ValueError(f"Distance metadata needs {field}.")
    if meta["prediction_source"] != "current_xenium_predictions":
        raise ValueError("Distance inputs require Xenium prediction provenance.")
    if meta["distance_units"] not in {"grid", "um"}:
        raise ValueError("distance_units must be grid or um.")
    if meta["distance_units"] == "grid" and slides.crop_size_um.nunique() > 1:
        raise ValueError("Grid distances from different crop sizes cannot be pooled. Analyze one crop size at a time.")
    table = pd.read_csv(path)
    keys = ["slide_id", "crop_size_um", "source_run"]
    missing = set([*keys, *DISTANCES]) - set(table)
    if missing:
        raise ValueError(f"Missing distance columns: {sorted(missing)}")
    if table[keys].isna().any().any() or table.duplicated(keys).any():
        raise ValueError("Distance input has missing or duplicate slide/crop/run keys.")
    values = table[DISTANCES].apply(pd.to_numeric, errors="raise")
    if np.isinf(values).any().any() or values.lt(0).any().any():
        raise ValueError("Distances must be nonnegative finite values or missing when a category is absent.")
    table[DISTANCES] = values
    expected = slides[[*keys, "patient_id"]].drop_duplicates()
    if expected.slide_id.duplicated().any():
        raise ValueError("Select one prediction result per slide before loading distances.")
    matched = expected.merge(table[[*keys, *DISTANCES]], on=keys, how="left", validate="one_to_one", indicator=True)
    audit = matched[keys + ["patient_id", "_merge"]].rename(columns={"_merge": "distance_input_match"})
    audit["n_missing_distance_features"] = matched[DISTANCES].isna().sum(axis=1)
    # A patient's mean is defined only if each selected slide has that feature.
    grouped = matched.groupby("patient_id")[DISTANCES]
    means = grouped.mean().where(grouped.count().eq(matched.groupby("patient_id").size(), axis=0))
    return data.merge(means, on="patient_id", how="left", validate="one_to_one"), audit, meta


def _save(fig, output, stem, show):
    for extension in ["png", "pdf"]:
        fig.savefig(Path(output)/"figures"/f"{stem}.{extension}", bbox_inches="tight", dpi=160)
    if show:
        from IPython.display import Image, display
        display(Image(filename=str(Path(output)/"figures"/f"{stem}.png")))
    plt.close(fig)


def _km(ax, data, endpoint, variable):
    frame = endpoint_rows(data, endpoint, [variable])
    row = dict(variable=variable, endpoint=endpoint, n_patients=len(frame),
               n_events=int(frame[f"{endpoint}_event"].sum()), p_value=np.nan, status="insufficient_groups_or_events")
    curves, risks = [], []
    if frame.empty:
        ax.text(.5, .5, "No eligible observations", ha="center", transform=ax.transAxes)
        ax.set_title(label(variable))
        return row, curves, risks
    levels = LEVELS.get(variable)
    if levels is None:
        levels = list(frame[variable].cat.categories)
    if frame[variable].nunique() >= 2 and row["n_events"]:
        test = multivariate_logrank_test(frame[f"{endpoint}_time"], frame[variable], frame[f"{endpoint}_event"])
        row.update(p_value=float(test.p_value), status="ok")
    ticks = np.linspace(0, frame[f"{endpoint}_time"].max(), 6)
    for level in levels:
        part = frame.loc[frame[variable].eq(level)]
        if part.empty:
            continue
        fit = KaplanMeierFitter().fit(part[f"{endpoint}_time"], part[f"{endpoint}_event"], label=f"{level} (n={len(part)})")
        fit.plot_survival_function(ax=ax, ci_show=False, show_censors=True)
        curves.extend(dict(variable=variable, endpoint=endpoint, group=level, time_days=t, survival_probability=v)
                      for t, v in fit.survival_function_.iloc[:, 0].items())
        risks.extend(dict(variable=variable, endpoint=endpoint, group=level, time_days=t,
                          n_at_risk=int(part[f"{endpoint}_time"].ge(t).sum())) for t in ticks)
    ax.set(title=f"{label(variable)}\nLog-rank p={row['p_value']:.3g}", xlabel="Time (days)",
           ylabel="Progression-free interval probability" if endpoint == "PFI" else "Overall survival probability", ylim=(0, 1.02))
    ax.legend(fontsize=8)
    return row, curves, risks


def figures(data, medians, output, show=True):
    """Export the selected directed-distance PFI curves and median cutpoints."""
    data = data.copy()
    cuts, rows, curves, risks = [], [], [], []
    for variable in medians:
        groups, thresholds, status = quantile_groups(data[variable], 'median')
        data[variable + '_bin'] = groups
        cuts.append(dict(variable=variable, kind='median', cutpoints=json.dumps(thresholds.tolist()),
                         status=status, counts=json.dumps({str(k): int(v) for k, v in groups.value_counts(sort=False).items()})))
    pd.DataFrame(cuts).to_csv(Path(output) / 'cutpoints.csv', index=False)
    eligible = endpoint_rows(data, 'PFI', ['age', 'stage', 'subtype'])
    fig, axes = plt.subplots(4, 3, figsize=(16, 17), layout='constrained')
    for ax, variable in zip(axes.flat, medians):
        row, c, r = _km(ax, eligible, 'PFI', variable + '_bin')
        rows.append(row); curves.extend(c); risks.extend(r)
    _save(fig, output, 'distance_PFI_KM', show)
    tests = pd.DataFrame(rows)
    tests['family'] = 'median_distances'
    tests['q_BH_supplementary'] = false_discovery_control(tests.p_value.fillna(1).to_numpy())
    tests.loc[tests.p_value.isna(), 'q_BH_supplementary'] = np.nan
    tests.to_csv(Path(output) / 'KM_logrank.csv', index=False)
    pd.DataFrame(curves).to_csv(Path(output) / 'KM_curves.csv', index=False)
    pd.DataFrame(risks).to_csv(Path(output) / 'KM_risk_tables.csv', index=False)
    return data, tests


def run_survival_models(data, output, *, distance_metadata=None, show=True):
    """Fit unadjusted and clinically adjusted continuous and binary distance models."""
    validate_data(data)
    if set(DISTANCES) - set(data) or not distance_metadata:
        raise ValueError('Distance analyses require all 48 summaries and explicit method metadata.')
    membership = data[['patient_id']].copy()
    membership['included'] = data.stage.notna() & data.subtype.notna()
    membership['reason'] = np.select([data.stage.isna(), data.subtype.isna()],
                                    ['missing_stage', 'missing_subtype'], default='included')
    frame = data.loc[membership.included].copy()
    if frame.empty:
        raise ValueError('No patients have complete stage and subtype data.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'figures').mkdir()
    membership.to_csv(output / 'cohort_membership.csv', index=False)
    frame, km = figures(frame, MEDIANS, output, show)
    frame.to_csv(output / 'analysis_table.csv', index=False)
    variables = [*MEDIANS, *[variable + '_bin' for variable in MEDIANS]]
    fits = {}
    for endpoint in ['PFI', 'OS']:
        eligible = endpoint_rows(frame, endpoint, ['age', 'stage', 'subtype'])
        for variable in variables:
            fits[f'{endpoint}__marginal__{variable}'] = cox(eligible, endpoint, [variable])
            fits[f'{endpoint}__adjusted_distance__{variable}'] = cox(
                eligible, endpoint, [variable], adjust=['age', 'subtype', 'stage'])
    status = save_cox(fits, output, 'cox')
    inventory = pd.DataFrame([
        dict(analysis='Directed median-distance PFI Kaplan-Meier comparisons', status='run'),
        dict(analysis='Unadjusted and clinically adjusted continuous/binary PFI and OS distance models', status='run')])
    inventory.to_csv(output / 'analysis_inventory.csv', index=False)
    (output / 'analysis_config.json').write_text(json.dumps(dict(
        patient_aggregation='Equal slide means of slide distance medians',
        complete_stage_and_subtype_filter=True, exclude_nonpositive_followup=True,
        subtype_reference='Basal', stage_reference='I', continuous_HR_increment=1,
        median_cutpoints='Type-7 quantiles of stage/subtype-eligible patients, fixed across endpoints',
        KM_BH='12 directed PFI median-distance tests', cox_ties='Efron',
        distance_metadata=distance_metadata), indent=2) + chr(10))
    return dict(inventory=inventory, cox_status=status, km_results=km, output=output)
