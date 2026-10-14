"""Generate the final Supplementary Figures S15 and S16."""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

import argparse
import os

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from xenium_pipeline.storage import data_path, clinical_file
from xenium_pipeline.brca_proportion_survival import COHORT_POLICY, prepare as prepare_proportions

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--project-root", type=Path, default=Path(os.environ.get("XENIUM_ROOT", CODE_ROOT)))
parser.add_argument("--run-dir", type=Path, help="Completed final clinical run directory")
parser.add_argument("--output-dir", type=Path)
parser.add_argument("--proportion-dir", type=Path, help="Saved results for the selected unadjusted and adjusted cohorts")
parser.add_argument("--patient-table", type=Path, help="Full calibrated patient table before clinical filtering")
args = parser.parse_args()
ROOT = args.project_root.expanduser().resolve()
RUN = (args.run_dir or data_path(ROOT, "clinical_run")).expanduser().resolve()
PROPORTION_MODELS = (args.proportion_dir or RUN / 'proportion_models').expanduser().resolve()
provenance = PROPORTION_MODELS / 'analysis_provenance.json'
PROPORTIONS_PREPARED = not provenance.is_file() or json.loads(provenance.read_text()).get('cohort_policy') != COHORT_POLICY
if PROPORTIONS_PREPARED:
    patients = args.patient_table or RUN / 'patient_analysis_table.csv'
    if not Path(patients).is_file():
        patients = data_path(ROOT, 'clinical_patient_table')
    prepare_proportions(RUN / 'survival_models', PROPORTION_MODELS, patient_table=patients)
OUT = (args.output_dir or ROOT / "supplementary_figures/BRCA").expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=True)
MAIN = OUT
LABELS = {"tumor_cells": "Tumor", "stromal_cells": "Stroma",
          "pan_apc_b": "pan-APC", "t_cells": "T cells"}
PAIRS = [(s, t) for s in LABELS for t in LABELS if s != t]
FEATURES = [f"{s}_to_{t}_median" for s, t in PAIRS]
PAIR_NAMES = [f"{LABELS[s]} → {LABELS[t]}" for s, t in PAIRS]
PROP_NAMES = ["Tumor Cells", "Stromal Cells", "pan-APC + B", "T Cells", "Normal Epithelial Cells"]
DISPLAY_PROP = ["Tumor", "Stroma", "pan-APC", "T cells", "Normal epithelial"]
COLORS = ["#4477AA", "#CC6677"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "font.weight": "bold",
                     "axes.labelweight": "bold", "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
coeff = pd.read_csv(clinical_file(RUN, "cox_coefficients.csv"))
status = pd.read_csv(clinical_file(RUN, "cox_model_status.csv")).set_index("model_id")
# Reuse the validated run for figure regeneration; analysis is prepared separately.
props = pd.read_csv(PROPORTION_MODELS / "cox_results.csv")
data = pd.read_csv(clinical_file(RUN, "analysis_table.csv"))
curves = pd.read_csv(clinical_file(RUN, "KM_curves.csv"))
tests = pd.read_csv(clinical_file(RUN, "KM_logrank.csv"))
export = []
outputs = []
risk_export = []


def distances(endpoint, family, binary=False):
    features = [v + "_bin" if binary else v for v in FEATURES]
    rows = []
    for feature in features:
        key = f"{endpoint}__{family}__{feature}"
        row = coeff.loc[coeff.model_id.eq(key) & coeff.variable.eq(feature)]
        assert len(row) == 1 and status.loc[key, "status"] == "ok", key
        assert status.loc[key, "n_patients"] == 413
        assert status.loc[key, "n_events"] == (62 if endpoint == "PFI" else 60)
        assert row.iloc[0].increment == 1
        rows.append(row.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def save(fig, base):
    if base.name not in {"Figure_S15", "Figure_S16"}:
        plt.close(fig)
        return
    from matplotlib.text import Text
    for text in fig.findobj(Text):
        text.set_fontweight("bold")
    for suffix in ("png", "pdf", "svg"):
        path = base.with_suffix("." + suffix)
        fig.savefig(path, dpi=300, facecolor="white")
        outputs.append(str(path))
    plt.close(fig)


def interval(ax, rows, positions, color, marker="o", label=None):
    assert np.isfinite(rows[["HR", "lower95", "upper95"]]).all().all()
    hr, lo, hi = (rows[c].to_numpy() for c in ["HR", "lower95", "upper95"])
    assert ((lo > 0) & (lo <= hr) & (hr <= hi)).all()
    ax.errorbar(hr, positions, xerr=np.vstack([hr - lo, hi - hr]),
                fmt=marker, color=color, markersize=4.5, linewidth=1.25,
                capsize=2.5, label=label)


def format_axis(ax, names, limits, ticks):
    ax.axvline(1, linestyle="--", color="#888888", linewidth=1)
    ax.set(xscale="log", xlim=limits, yticks=np.arange(len(names)),
           yticklabels=names, ylim=(len(names) - .5, -.5))
    ax.set_xticks(ticks, labels=[f"{v:g}" for v in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.grid(axis="x", color="#e5e5e5", linewidth=.6)
    ax.tick_params(axis="y", length=0)


def record(rows, figure, endpoint, model):
    part = rows.copy()
    part["figure"] = figure
    part["endpoint"] = endpoint
    part["plot_model"] = model
    export.append(part)


# S15: proportions, continuous distances, and median-based distances in three rows.
fig, axes = plt.subplots(3, 2, figsize=(14.4, 16.6))
fig.subplots_adjust(left=.16, right=.985, top=.895, bottom=.065, wspace=.65, hspace=.48)
for col, endpoint in enumerate(["PFI", "OS"]):
    for model, offset, color, marker in [("unadjusted", -.13, COLORS[0], "o"),
                                          ("adjusted", .13, COLORS[1], "s")]:
        p = props.loc[props.endpoint.eq(endpoint) & props.model.eq(model)].set_index("cell_type").loc[PROP_NAMES].reset_index()
        assert len(p) == 5 and p.status.eq("ok").all() and p.increment.eq(.1).all()
        assert p.n_patients.eq(424 if model == 'unadjusted' else 413).all()
        assert p.n_events.eq(66 if model == 'unadjusted' else 62 if endpoint == 'PFI' else 60).all()
        interval(axes[0, col], p, np.arange(5) + offset, color, marker, model.capitalize())
        record(p, "S15_proportions", endpoint, model)
        ds = distances(endpoint, "marginal" if model == "unadjusted" else "adjusted_distance")
        interval(axes[1, col], ds, np.arange(12) + offset, color, marker, model.capitalize())
        record(ds, "S15_continuous_distances", endpoint, model)
    format_axis(axes[0, col], DISPLAY_PROP, (.04, 3.3), [.05, .1, .5, 1, 2, 3])
    format_axis(axes[1, col], PAIR_NAMES, (.945, 1.024), [.95, .97, .99, 1, 1.02])
    axes[0, col].tick_params(labelsize=12)
    axes[1, col].tick_params(labelsize=12)
    axes[0, col].set_title(f"{endpoint}: cell-type proportions", loc="left", fontsize=15)
    axes[1, col].set_title(f"{endpoint}: continuous distances", loc="left", fontsize=15)
    axes[0, col].set_xlabel("HR per 10-percentage-point\nincrease", fontsize=14, labelpad=8)
    axes[1, col].set_xlabel("HR per 1 source-proportion-\nweighted µm increase", fontsize=14, labelpad=8)
# Median-based distance models form the bottom row.
for col, endpoint in enumerate(["PFI", "OS"]):
    ax = axes[2, col]
    for family, offset, color, marker, label in [
        ("marginal", -.13, COLORS[0], "o", "Unadjusted"),
        ("adjusted_distance", .13, COLORS[1], "s", "Adjusted")]:
        ds = distances(endpoint, family, binary=True)
        interval(ax, ds, np.arange(12) + offset, color, marker, label)
        record(ds, "S15_binary_distances", endpoint, family)
    format_axis(ax, PAIR_NAMES, (.18, 7.5), [.2, .5, 1, 2, 5])
    ax.tick_params(labelsize=12)
    ax.set_title(f"{endpoint}: median-based distances", loc="left", fontsize=15)
    ax.set_xlabel("HR: high vs low median distance", fontsize=14, labelpad=8)
fig.suptitle("TCGA-BRCA: cell-type proportions and spatial-distance Cox models", fontsize=19, y=.982)
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
           bbox_to_anchor=(.55, .958), ncol=2, frameon=False, fontsize=15)
save(fig, OUT / "Figure_S15")


# S16: all 12 binary PFI distance comparisons in one figure, using saved groups and curves.
for number, indices in [(16, range(12))]:
    fig = plt.figure(figsize=(18.4, 20.4))
    grid = fig.add_gridspec(4, 3, left=.065, right=.98, top=.89, bottom=.035, hspace=.40, wspace=.32)
    for panel, index in enumerate(indices):
        feature = FEATURES[index]; variable = feature + "_bin"
        subgrid = grid[panel // 3, panel % 3].subgridspec(2, 1, height_ratios=[4.4, 1.2], hspace=.43)
        ax = fig.add_subplot(subgrid[0]); risk = fig.add_subplot(subgrid[1])
        eligible = data.loc[data[["PFI_time", "PFI_event", variable]].notna().all(axis=1) & data.PFI_time.gt(0) & data.clinical_matched].copy()
        test = tests.loc[tests.variable.eq(variable) & tests.endpoint.eq("PFI")].iloc[0]
        assert len(eligible) == test.n_patients == 413 and eligible.PFI_event.sum() == test.n_events == 62
        assert np.all(eligible[variable].eq("High").to_numpy() == eligible[feature].gt(data[feature].median()).to_numpy())
        for group, color in zip(["Low", "High"], COLORS):
            c = curves.loc[curves.variable.eq(variable) & curves.endpoint.eq("PFI") & curves.group.eq(group)]
            assert len(c) > 0 and c.survival_probability.between(0, 1).all()
            part = eligible.loc[eligible[variable].eq(group)]
            ax.step(c.time_days / 365.25, c.survival_probability, where="post", color=color, linewidth=1.65)
            # Verify the saved curve independently against the event/censor data.
            km = KaplanMeierFitter().fit(part.PFI_time, part.PFI_event)
            assert np.allclose(km.predict(c.time_days).to_numpy(), c.survival_probability.to_numpy(), atol=1e-12)
            censor_times = km.event_table.index[km.event_table.censored.gt(0)]
            visible = censor_times[censor_times <= 15 * 365.25]
            ax.plot(visible / 365.25, km.predict(visible), "+", color=color, markersize=3.2, markeredgewidth=.6)
            for year in [0, 5, 10, 15]:
                count = int(part.PFI_time.ge(year * 365.25).sum())
                risk.text(year, .36 if group == "Low" else .02, str(count), color=color, ha="center", va="center", fontsize=14)
                risk_export.append(dict(figure=f"S{number}", variable=variable, group=group, years=year, n_at_risk=count))
        ax.set(xlim=(0, 15), ylim=(0, 1.025), xticks=[0, 5, 10, 15], yticks=[0, .25, .5, .75, 1], xlabel="Years", ylabel="PFI probability")
        ax.tick_params(labelsize=13)
        ax.xaxis.label.set_size(13)
        ax.yaxis.label.set_size(13)
        ax.set_title(PAIR_NAMES[index], loc="left", fontsize=15)
        ax.text(.04, .055, f"Log-rank p = {test.p_value:.3g}\nBH q = {test.q_BH_supplementary:.3g}", transform=ax.transAxes, fontsize=12)
        ax.grid(color="#eeeeee", linewidth=.6)
        risk.set(xlim=(-4.5, 15.5), ylim=(-.20, 1))
        risk.axis("off")
        risk.text(-4.5, .82, "At risk", fontsize=14)
        for year in [0, 5, 10, 15]: risk.text(year, .82, str(year), ha="center", fontsize=14, color="#666666")
        risk.text(-4.5, .36, "Low", color=COLORS[0], fontsize=14, va="center")
        risk.text(-4.5, .02, "High", color=COLORS[1], fontsize=14, va="center")
    fig.suptitle("TCGA-BRCA: PFI by directed median distance", fontsize=18, y=.975)
    fig.legend([Line2D([0], [0], color=c, lw=2) for c in COLORS], ["Low (≤ median)", "High (> median)"],
               loc="upper center", bbox_to_anchor=(.55, .956), ncol=2, frameon=False, fontsize=13)
    save(fig, OUT / f"Figure_S{number}")

pd.concat(export, ignore_index=True).to_csv(MAIN / "BRCA_survival_figure_estimates.csv", index=False)
pd.DataFrame(risk_export).to_csv(MAIN / "BRCA_distance_KM_numbers_at_risk.csv", index=False)
sources = [clinical_file(RUN, name) for name in ["cox_coefficients.csv", "cox_model_status.csv",
    "analysis_table.csv", "KM_curves.csv", "KM_logrank.csv", "cox_patient_ids.csv",
    "cutpoints.csv", "analysis_config.json"]]
sources += list(PROPORTION_MODELS.glob("*.csv")) + [PROPORTION_MODELS/"analysis_provenance.json", Path(__file__).resolve(),
    CODE_ROOT/"xenium_pipeline/brca_proportion_survival.py"]
(MAIN / "BRCA_survival_figure_provenance.json").write_text(json.dumps(dict(
    input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    output_files=outputs, cutoffs=dict(tumor=.20, stroma=.30, t_cells=.30, pan_apc_b=.30),
    distance_HR_increment=1, proportion_HR_increment=.10, KM_display_max_years=15,
    models_refitted=PROPORTIONS_PREPARED, proportion_models_fitted=20 if PROPORTIONS_PREPARED else 0,
    proportion_models_reused=not PROPORTIONS_PREPARED, distance_models_refitted=False,
    source_run=RUN.name, derived_proportion_results=str(PROPORTION_MODELS),
    distance_endpoint_cohorts={"PFI":{"n_patients":413,"n_events":62}, "OS":{"n_patients":413,"n_events":60}},
    proportion_endpoint_cohorts=json.loads(provenance.read_text())['endpoint_cohorts'],
    distance_cutpoint_patients=416, proportion_cutpoint_patients=json.loads(provenance.read_text())['n_cutpoint_patients'],
    KM_saved_curves_verified=True), indent=2) + "\n")
print(f"Generated {len(outputs)} figure files; all saved KM curves verified against patient data.")
