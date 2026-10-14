"""Reproduce Figure 6C from the supplied aggregate figure data."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd

LABELS = {"tumor_cells": "Tumor", "stromal_cells": "Stroma",
          "pan_apc_b": "pan-APC", "t_cells": "T cells"}
PAIRS = [(s, t) for s in LABELS for t in LABELS if s != t]
FEATURES = [f"{s}_to_{t}_median" for s, t in PAIRS]
PAIR_NAMES = [f"{LABELS[s]} → {LABELS[t]}" for s, t in PAIRS]
COX_COLORS = ["#4477AA", "#CC6677"]
KM_COLORS = ["#009E73", "#E69F00"]
FILENAME = "Figure_6C_BRCA_PFI_survival"


def validate_inputs(estimates, curves, censoring, risk, metadata):
    expected = {(feature, metric, model) for feature in FEATURES
                for metric in ["continuous", "binary"] for model in ["unadjusted", "adjusted"]}
    keys = list(zip(estimates.feature, estimates.metric, estimates.model))
    if len(keys) != 48 or set(keys) != expected:
        raise ValueError("Cox data must contain exactly 12 distances × 2 metrics × 2 models.")
    values = estimates[["HR", "lower95", "upper95", "p_value"]]
    if not np.isfinite(values).all().all() or not (
        estimates.lower95.gt(0) & estimates.HR.ge(estimates.lower95)
        & estimates.HR.le(estimates.upper95) & estimates.p_value.between(0, 1)
    ).all():
        raise ValueError("Invalid Cox estimates, confidence intervals, or p-values.")
    if not estimates.n_patients.eq(metadata["n_patients"]).all() or not estimates.n_events.eq(metadata["n_events"]).all():
        raise ValueError("Cox cohort counts disagree with metadata.")
    for frame, name in [(curves, "curves"), (censoring, "censoring"), (risk, "risk")]:
        if set(frame.group) != {"Low", "High"}:
            raise ValueError(f"{name} must contain Low and High groups.")
    for frame in [curves, censoring]:
        if not np.isfinite(frame[["time_days", "survival_probability"]]).all().all() or not (
            frame.time_days.ge(0) & frame.survival_probability.between(0, 1)
        ).all():
            raise ValueError("Invalid KM times or probabilities.")
    for group in ["Low", "High"]:
        c = curves.loc[curves.group.eq(group)]
        if c.time_days.duplicated().any() or not c.time_days.is_monotonic_increasing or not c.survival_probability.is_monotonic_decreasing:
            raise ValueError("KM curve points must be ordered, unique, and non-increasing.")
        r = risk.loc[risk.group.eq(group)].sort_values("years")
        if r.years.tolist() != [0, 5, 10, 15] or not r.n_at_risk.is_monotonic_decreasing or not (
            r.n_at_risk.ge(0) & r.n_at_risk.eq(r.n_at_risk.round())
        ).all():
            raise ValueError("Invalid numbers-at-risk table.")
    if risk.loc[risk.years.eq(0), "n_at_risk"].sum() != metadata["n_patients"]:
        raise ValueError("Initial risk counts do not add up to the cohort size.")
    if not 0 <= metadata["logrank_p"] <= metadata["logrank_q_BH"] <= 1:
        raise ValueError("Invalid log-rank p-value or BH q-value.")


def render_figure(estimates, curves, censoring, risk, metadata, width=10.6, height=6.8):
    """Return the figure; risk-table text is included in panel-height alignment."""
    validate_inputs(estimates, curves, censoring, risk, metadata)
    if not np.isfinite([width, height]).all() or width <= 0 or height <= 0:
        raise ValueError("Figure dimensions must be positive finite inches.")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(width, height))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.55, 2.65, 2.35],
                           left=.025, right=.985, top=.83, bottom=.115, wspace=.22)
    labels, cox = [fig.add_subplot(grid[0, col]) for col in range(2)]
    labels.set(xlim=(0, 1), ylim=(11.5, -.5)); labels.axis("off")
    for i, name in enumerate(PAIR_NAMES):
        labels.text(1, i, name, ha="right", va="center", fontsize=9.5)
    titles = []
    for ax, metric, limits, ticks, title, xlabel in [
        (cox, "binary", (.2, 5.5), [.25, .5, 1, 2, 4], "Cox models", "HR: high vs low")]:
        for model, offset, color, marker, label in [
            ("unadjusted", -.17, COX_COLORS[0], "o", "Unadjusted"),
            ("adjusted", .17, COX_COLORS[1], "s", "Adjusted")]:
            rows = estimates.loc[estimates.metric.eq(metric) & estimates.model.eq(model)].set_index("feature").loc[FEATURES]
            hr, lo, hi = (rows[col].to_numpy() for col in ["HR", "lower95", "upper95"])
            if lo.min() <= limits[0] or hi.max() >= limits[1]:
                raise ValueError("A confidence interval would be clipped by the published axis limits.")
            ax.errorbar(hr, np.arange(12) + offset, xerr=np.vstack([hr - lo, hi - hr]),
                        fmt=marker, color=color, markersize=4.5, linewidth=1.25, capsize=2.5, label=label)
        ax.axvline(1, linestyle="--", color="#888888", linewidth=1)
        ax.set(xscale="log", xlim=limits, yticks=np.arange(12), yticklabels=[""] * 12, ylim=(11.5, -.5))
        ax.set_xticks(ticks, labels=[f"{v:g}" for v in ticks])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.grid(axis="x", color="#e5e5e5", linewidth=.6)
        ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", labelsize=9)
        titles.append(ax.set_title(title, loc="left", fontsize=10))
        ax.set_xlabel(xlabel, fontsize=9)
    kmgrid = grid[0, 2].subgridspec(2, 1, height_ratios=[4.5, 1.25], hspace=.60)
    km_ax = fig.add_subplot(kmgrid[0]); km_risk = fig.add_subplot(kmgrid[1])
    for group, color in zip(["Low", "High"], KM_COLORS):
        c = curves.loc[curves.group.eq(group)]
        km_ax.step(c.time_days / 365.25, c.survival_probability, where="post", color=color, linewidth=1.4, label=group)
        marks = censoring.loc[censoring.group.eq(group) & censoring.time_days.le(15 * 365.25)]
        km_ax.plot(marks.time_days / 365.25, marks.survival_probability, "+", color=color, markersize=3, markeredgewidth=.5)
        for row in risk.loc[risk.group.eq(group)].itertuples():
            km_risk.text(row.years, .36 if group == "Low" else .02, str(int(row.n_at_risk)),
                         color=color, ha="center", va="center", fontsize=9)
    km_ax.set(xlim=(0, 15), ylim=(0, 1.025), xticks=[0, 5, 10, 15], yticks=[0, .5, 1],
              xlabel="Years", ylabel="PFI probability")
    km_ax.tick_params(labelsize=9)
    km_ax.xaxis.label.set_size(9); km_ax.yaxis.label.set_size(9)
    titles.append(km_ax.set_title("pan-APC → T cells", loc="left", fontsize=10))
    km_ax.text(.045, .065, f"Log-rank p = {metadata['logrank_p']:.3g}\nBH q = {metadata['logrank_q_BH']:.3g}",
               transform=km_ax.transAxes, fontsize=9)
    km_ax.legend(loc="upper right", frameon=False, fontsize=9, handlelength=1.2)
    km_ax.grid(color="#eeeeee", linewidth=.5)
    km_risk.set(xlim=(-5.5, 15.7), ylim=(-.23, 1)); km_risk.axis("off")
    km_risk.text(-5.5, .82, "At risk", fontsize=9)
    for year in [0, 5, 10, 15]: km_risk.text(year, .82, str(year), ha="center", fontsize=9, color="#666666")
    km_risk.text(-5.5, .36, "Low", color=KM_COLORS[0], fontsize=9, va="center")
    km_risk.text(-5.5, .02, "High", color=KM_COLORS[1], fontsize=9, va="center")
    fig.suptitle("TCGA-BRCA spatial distances and PFI", x=.51, y=.965, fontsize=13)
    fig.legend(*cox.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.46, .925), ncol=2, frameon=False, fontsize=9)
    for text in fig.findobj(match=Text):
        text.set_fontsize(text.get_fontsize() * 1.2)
        text.set_fontweight("bold")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Keep 0.18 inches between the KM x-label and the risk-table header.
    risk_top = max(text.get_window_extent(renderer).y1 for text in km_risk.texts)
    target_top = km_ax.xaxis.label.get_window_extent(renderer).y0 - .18 * fig.dpi
    box = km_risk.get_position()
    shift = (target_top - risk_top) / fig.bbox.height
    km_risk.set_position([box.x0, box.y0 + shift, box.width, box.height])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    km_bottom = min(text.get_window_extent(renderer).y0 for text in km_risk.texts)
    for forest in [cox]:
        box = forest.get_position()
        shift = (km_bottom - forest.xaxis.label.get_window_extent(renderer).y0) / fig.bbox.height
        forest.set_position([box.x0, box.y0 + shift, box.width, box.height - shift])
    box = cox.get_position(); label_box = labels.get_position()
    labels.set_position([label_box.x0, box.y0, box.x0 - .12 / width - label_box.x0, box.height])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bottoms = [cox.xaxis.label.get_window_extent(renderer).y0, km_bottom]
    tops = [title.get_window_extent(renderer).y1 for title in titles]
    if max(bottoms) - min(bottoms) >= 1 or max(tops) - min(tops) >= 1:
        raise RuntimeError("Complete panel heights failed to align.")
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", type=Path, default=root / "tables")
    parser.add_argument("--output-dir", type=Path, default=root / "figures")
    parser.add_argument("--width", type=float, default=10.6, help="Width in inches")
    parser.add_argument("--height", type=float, default=6.8, help="Height in inches")
    args = parser.parse_args(argv)
    metadata = json.loads((args.data_dir / "figure_6c_metadata.json").read_text())
    for filename, expected in metadata["data_sha256"].items():
        actual = hashlib.sha256((args.data_dir / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Input checksum mismatch: {filename}")
    frames = [pd.read_csv(args.data_dir / ("figure_6c_" + name), float_precision="round_trip") for name in
              ["cox_estimates.csv", "km_curves.csv", "km_censoring.csv", "numbers_at_risk.csv"]]
    fig = render_figure(*frames, metadata, args.width, args.height)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "svg", "png"]:
        path = args.output_dir / f"{FILENAME}.{ext}"
        fig.savefig(path, dpi=300, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
