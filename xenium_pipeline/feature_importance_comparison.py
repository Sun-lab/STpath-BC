"""Final full-vector importance panels and frozen Visium comparator inputs."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import urllib.request
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .stpath_xgboost import EXPECTED_DIMS as ALL_ENCODER_DIMS, MODEL_NAMES, select_top_features

# Feature-importance comparisons use the five pathology encoders only.
EXPECTED_DIMS = {name: ALL_ENCODER_DIMS[name] for name in MODEL_NAMES}


from .storage import load_layout
TARGET_KEYS = {'Tumor':'tumor_cells', 'Normal_Epithelial':'normal_epithelial_cells',
               'T_cell':'t_cells', 'Stromal':'stromal_cells', 'pan_APC':'pan_apc_b'}
TARGETS = {alias:(spec['cell_type'], spec['run_id'] + '/' + re.sub(r'[^A-Za-z0-9._-]+', '_', spec['cell_type']).strip('_'))
           for alias, key in TARGET_KEYS.items() for spec in [load_layout()['targets'][key]]}
VISIUM_SHA256 = "9ecd5c69b6c192d33053c27732c9a5d84966d6d6601827a7b8c30a8000414da5"
VISIUM_URL = (
    "https://github.com/Sun-lab/STpath-software/blob/"
    "fcf867babfdf9ea768b9c6ddce9daef7ef87f372/BRCA/"
    "XGBoost_Models/Important_Features_BRCA.xlsx"
)
VISIUM_DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/Sun-lab/STpath-software/"
    "fcf867babfdf9ea768b9c6ddce9daef7ef87f372/BRCA/"
    "XGBoost_Models/Important_Features_BRCA.xlsx"
)
COAD_DOWNLOAD_URL = VISIUM_DOWNLOAD_URL.replace("BRCA", "COAD")
# The public COAD workbook has spaces in sheet names; its cell values match media-3.xlsx.
COAD_SHA256 = (
    "1a62b2c55775246ac6919d5a8a17f9e11e62c5d94fe3af0f63a8d66d7186e01a",
    "85cbe2106773820e6a4e730932f148b16593460a044701ffb5c2c12f4153ce8a",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def load_visium(path, *, cohort="BRCA"):
    """Fetch the verified workbook if absent; keep unreported dimensions missing."""
    import openpyxl

    if cohort not in {"BRCA", "COAD"}:
        raise ValueError(f"Unknown Visium cohort: {cohort}")
    url = VISIUM_DOWNLOAD_URL if cohort == "BRCA" else COAD_DOWNLOAD_URL
    checksums = (VISIUM_SHA256,) if cohort == "BRCA" else COAD_SHA256
    path = Path(path)
    if not path.exists():
        print(f"Downloading the manuscript's Visium feature rankings to {path}", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read()
        except OSError as exc:
            raise FileNotFoundError(
                f"Missing Visium workbook: {path}. Automatic download failed: {exc}. "
                f"Download {url} on a machine with internet access "
                f"and copy it to {path}."
            ) from exc
        if hashlib.sha256(content).hexdigest() not in checksums:
            raise ValueError("Downloaded Visium workbook differs from the verified manuscript supplement.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as temporary:
            pending = Path(temporary.name)
            try:
                temporary.write(content)
                temporary.close()
                pending.replace(path)
            finally:
                pending.unlink(missing_ok=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() not in checksums:
        raise ValueError("Visium workbook differs from the verified manuscript supplement.")
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = []
    try:
        sheets = {target: target for target in TARGETS} if cohort == "BRCA" else {
            "Tumor": "Cancer_Cells" if "Cancer_Cells" in workbook.sheetnames else "Cancer Cells",
        }
        for target, sheet in sheets.items():
            sheet_rows = list(workbook[sheet].values)
            header = list(sheet_rows[0])
            for encoder in MODEL_NAMES:
                fi = header.index(f"{encoder}_Features")
                gi = header.index(f"{encoder}_Importance")
                pairs = [(row[fi], row[gi]) for row in sheet_rows[1:] if row[fi] is not None]
                indices = [int(name.removeprefix("f")) for name, _ in pairs]
                gains = np.array([gain for _, gain in pairs], dtype=float)
                width = EXPECTED_DIMS[encoder]
                if (len(pairs) != round(0.30 * width) or len(set(indices)) != len(indices)
                        or min(indices) < 0 or max(indices) >= width
                        or not np.isfinite(gains).all() or (gains < 0).any()
                        or (np.diff(gains) > 0).any()):
                    raise ValueError(f"Invalid published ranking: {target}/{encoder}")
                for rank, (index, gain) in enumerate(zip(indices, gains), 1):
                    rows.append(dict(visium_target=target, encoder=encoder,
                                     feature_index=index, published_rank=rank, gain=gain))
    finally:
        workbook.close()
    return pd.DataFrame(rows)


def load_gain(checkpoint, encoder, completion):
    checkpoint = Path(checkpoint)
    metadata = read_json(checkpoint / "model_metadata" / f"{encoder}.json")
    if (metadata["checkpoint_signature"] != completion["training_data_fingerprint"]
            or metadata["training_config_fingerprint"] != completion["training_config_fingerprint"]
            or metadata["model_type"] != encoder
            or metadata["input_dimension"] != EXPECTED_DIMS[encoder]):
        raise ValueError(f"Incompatible gain checkpoint: {checkpoint}/{encoder}")
    with np.load(checkpoint / "results" / f"{encoder}.npz", allow_pickle=False) as saved:
        gains = saved["gains"].copy()
        selected = saved["selections"].copy()
    if (gains.shape != (EXPECTED_DIMS[encoder],) or not np.isfinite(gains).all()
            or (gains < 0).any() or gains.sum() <= 0
            or not np.array_equal(selected, select_top_features(gains))):
        raise ValueError(f"Invalid gain vector or selection: {checkpoint}/{encoder}")
    return gains


def load_xenium_target(directory, label):
    directory = Path(directory)
    complete = read_json(directory / "complete.json")
    metadata = read_json(directory / "model_metadata/Combined.json")
    if (complete.get("status") != "complete" or complete["cell_type"] != label
            or metadata["cell_type"] != label or metadata["num_boost_round"] != 800
            or metadata["extraction_recipe"] != "stpath_brca_training_compatible_v1"
            or metadata["training_config_fingerprint"] != complete["training_config_fingerprint"]
            or metadata["training_tile_id_checksum"] != complete["tile_id_checksum"]
            or metadata["n_training_tiles"] != complete["n_tiles"]
            or metadata["foundation_feature_dimensions"] != EXPECTED_DIMS
            or "smoke" in directory.parts):
        raise ValueError(f"Expected completed production training export: {directory}")
    checkpoints = directory / "checkpoints"
    final = {enc: load_gain(checkpoints / "final", enc, complete) for enc in MODEL_NAMES}
    for enc, gains in final.items():
        if not np.array_equal(select_top_features(gains), metadata["selected_feature_indices"][enc]):
            raise ValueError(f"Final selection metadata mismatch: {directory}/{enc}")
    folds, coverage, seen = [], [], set()
    for fold in sorted(checkpoints.glob("outer_fold_*")):
        marker = fold / "complete.json"
        if not marker.exists():
            continue
        info = read_json(marker)
        person = info["held_out_individual"]
        if (info["checkpoint_signature"] != complete["training_data_fingerprint"]
                or person not in metadata["individual_ids"] or person in seen):
            raise ValueError(f"Incompatible or duplicate fold: {fold}")
        seen.add(person)
        vectors = {enc: load_gain(fold / "outer", enc, complete) for enc in MODEL_NAMES}
        folds.append((person, vectors))
    for person in metadata["individual_ids"]:
        coverage.append(dict(xenium_target=label, individual_id=person,
                             gain_available=person in seen))
    return final, folds, coverage, metadata


def spearman(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Spearman inputs must be observed finite values.")
    a, b = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def top_indices(gains, k):
    return np.lexsort((np.arange(len(gains)), -np.asarray(gains)))[:k]


FIGURE7_ENCODERS = ("UNI2h", "Virchow2")
# The Xenium cutoff extends the manuscript's BRCA cutoff to the second breast dataset.
FIGURE7_THRESHOLDS = {"brca_visium": 1.0, "coad_visium": 0.5, "brca_xenium": 1.0}
FULL_GAIN_AGGREGATIONS = {'present_keys': 'mean gain over folds containing each feature key; normalize the full vector to 100%'}


def figure7_percent(fold_importances, encoder, *, aggregation="present_keys"):
    """Normalize complete fold gains; the default matches Figure 7's present-key mean."""
    if aggregation not in FULL_GAIN_AGGREGATIONS:
        raise ValueError(f"Unknown gain aggregation: {aggregation}")
    if not isinstance(fold_importances, (list, tuple)) or not fold_importances:
        raise ValueError("Expected a nonempty list of per-fold gain dictionaries.")
    width = EXPECTED_DIMS[encoder]
    totals, counts = np.zeros(width), np.zeros(width, dtype=int)
    for fold in fold_importances:
        if not isinstance(fold, dict):
            raise ValueError("Each fold must be a feature-name-to-gain dictionary.")
        for name, gain in fold.items():
            if not isinstance(name, str) or not name.startswith("f") or not name[1:].isdigit():
                raise ValueError(f"Invalid feature name: {name!r}")
            index, value = int(name[1:]), float(gain)
            if name != f"f{index}" or not 0 <= index < width or not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {encoder} importance: {name}={gain}")
            totals[index] += value
            counts[index] += 1  # A present key with gain=0 still contributes to the denominator.
    means = np.divide(totals, counts, out=np.zeros(width), where=counts > 0)
    if means.sum() <= 0:
        raise ValueError("Cannot normalize an all-zero importance profile.")
    return 100 * means / means.sum()


def load_figure7_reference(path, encoder, *, aggregation="present_keys"):
    """Read an author's trusted .pt results, or its feature_importances JSON export."""
    path = Path(path)
    if path.suffix == ".json":
        data = read_json(path)
    elif path.suffix == ".pt":
        import torch

        # Original result files contain NumPy arrays as well as the gain dictionaries.
        data = torch.load(path, map_location="cpu", weights_only=False)
    else:
        raise ValueError("Use the full per-fold .pt results or a .json export of feature_importances.")
    if not isinstance(data, dict) or "feature_importances" not in data:
        raise ValueError(f"Expected a feature_importances list in the full reference results: {path}")
    folds = data["feature_importances"]
    profile = figure7_percent(folds, encoder, aggregation=aggregation)
    # Author uploads include fold boosters: check both the input width and saved gains.
    model_dir = path.parent / "models"
    if model_dir.is_dir():
        import xgboost as xgb

        models = sorted(model_dir.glob("xgboost_model_leave_*_out.model"))
        if len(models) != len(folds):
            raise ValueError(f"Reference fold-count mismatch: {path}: {len(folds)} gain dictionaries, {len(models)} models")
        model_gains = []
        for model_path in models:
            model = xgb.Booster()
            model.load_model(str(model_path))
            if model.num_features() != EXPECTED_DIMS[encoder]:
                raise ValueError(f"Expected {EXPECTED_DIMS[encoder]} input dimensions: {model_path}")
            model_gains.append(model.get_score(importance_type="gain"))
        from_models = figure7_percent(model_gains, encoder, aggregation=aggregation)
        if not np.allclose(profile, from_models, rtol=1e-5, atol=1e-8):
            raise ValueError(f"Reference result gains disagree with the saved fold models: {path}")
    return profile, len(folds)


def load_xenium_figure7(directory, *, encoders=FIGURE7_ENCODERS, aggregation="present_keys"):
    """Read full tumor fold gains; recover exact booster key presence when required."""
    directory = Path(directory)
    _, folds, coverage, _ = load_xenium_target(directory, "Tumor Cells")
    if not coverage or not all(row["gain_available"] for row in coverage):
        raise ValueError("Figure 7 comparison requires all expected Xenium tumor LOIO folds to be complete.")
    if aggregation not in FULL_GAIN_AGGREGATIONS:
        raise ValueError(f"Unknown gain aggregation: {aggregation}")
    import xgboost as xgb

    expected = dict(folds)
    sparse = {enc: [] for enc in encoders}
    for marker in sorted((directory / "checkpoints").glob("outer_fold_*/complete.json")):
        person = read_json(marker)["held_out_individual"]
        for enc in encoders:
            model = xgb.Booster()
            model.load_model(str(marker.parent / "outer/models" / f"{enc}.ubj"))
            scores = model.get_score(importance_type="gain")
            # Validate names and values before indexing; verify the booster against saved gains.
            figure7_percent([scores], enc)
            dense = np.zeros(EXPECTED_DIMS[enc])
            for name, value in scores.items():
                dense[int(name[1:])] = value
            if model.num_features() != len(dense) or not np.allclose(dense, expected[person][enc], rtol=1e-6, atol=1e-12):
                raise ValueError(f"Booster and saved gains disagree: {marker.parent}/{enc}")
            sparse[enc].append(scores)
    return {enc: figure7_percent(sparse[enc], enc) for enc in encoders}, len(folds)


def _plot_full_figure7(groups, output_dir, *, save_pdf, views=None, manuscript_panel=False):
    """Extend the published Figure 7B panel design to every available dataset pair."""
    import matplotlib.pyplot as plt
    try:
        from adjustText import adjust_text
    except ImportError as exc:
        raise ImportError("Figure 7B uses the authors' label-placement package. Install it with: pip install adjustText") from exc

    encoders = [enc for enc in (*FIGURE7_ENCODERS, "Virchow", "ProvGigapath", "Conch") if enc in groups]
    has_coad = "coad_visium_percent" in groups[encoders[0]]
    if not has_coad:
        raise ValueError("The final manuscript figure requires a COAD Visium profile.")
    pairs = [("coad_visium_vs_brca_xenium", "brca_xenium", "coad_visium")]
    if views is not None:
        unknown = set(views).difference(view for view, _, _ in pairs)
        if unknown or not views:
            raise ValueError(f"Unavailable Figure 7B views: {sorted(unknown)}")
        pairs = [pair for pair in pairs if pair[0] in views]
    labels = {"brca_visium": "BRCA Visium", "coad_visium": "COAD Visium", "brca_xenium": "BRCA Xenium"}
    names = {"UNI2h": "UNI2-h", "ProvGigapath": "Prov-GigaPath", "Conch": "CONCH"}
    # Scale the author's 10-inch panel typography and marker areas to 6-inch panels.
    scale = .6
    legend_options = dict(prop={"size": 8.5, "weight": "bold"}, framealpha=.9)
    metrics = []
    for view, x_dataset, y_dataset in pairs:
        ncols = min(2, len(encoders))
        nrows = (len(encoders) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False)
        if not manuscript_panel:
            fig.suptitle(f"{labels[x_dataset]} versus {labels[y_dataset]}", fontsize=16, fontweight="bold", y=.995)
        else:
            fig.text(.015, .98, "B", fontsize=22, fontweight="bold", va="top")
        fig.subplots_adjust(left=.09, right=.98, bottom=.06, top=.94, hspace=.27, wspace=.27)
        for unused in axes.ravel()[len(encoders):]:
            unused.set_visible(False)
        label_sets = []
        for ax, enc in zip(axes.ravel(), encoders):
            table = groups[enc]
            x, y = (table[f"{dataset}_percent"].to_numpy() for dataset in (x_dataset, y_dataset))
            x_cut, y_cut = FIGURE7_THRESHOLDS[x_dataset], FIGURE7_THRESHOLDS[y_dataset]
            x_high, y_high = x > x_cut, y > y_cut
            masks = [~x_high & ~y_high, x_high & ~y_high, ~x_high & y_high, x_high & y_high]
            descriptions = ["Other features",
                            f"Important features (>{x_cut:.1f}%) for {labels[x_dataset]}",
                            f"Important features (>{y_cut:.1f}%) for {labels[y_dataset]}",
                            "Important features for both datasets"]
            for mask, color, size, alpha, zorder, description in zip(
                    masks, ("grey", "blue", "red", "purple"), (50, 80, 80, 120),
                    (.5, .8, .8, .9), (1, 2, 2, 3), descriptions):
                if mask.any():
                    ax.scatter(x[mask], y[mask], c=color, marker="o", s=size * scale**2,
                               alpha=alpha, zorder=zorder, label=description,
                               **({"edgecolor": "black", "linewidth": 1.5 * scale} if color == "purple" else {}))
            texts = [ax.text(x[pos], y[pos], f"f{int(table.feature_index.iloc[pos])}",
                             fontsize=(14 if manuscript_panel else 18) * scale, fontweight="bold", ha="center", va="bottom")
                     for pos in np.flatnonzero(x_high | y_high)]
            maximum = max(x.max(), y.max())
            ax.plot([0, maximum], [0, maximum], "k--", alpha=.5, linewidth=2 * scale, label="y=x", zorder=0)
            ax.set(xlim=(-.05 * maximum, 1.05 * maximum), ylim=(-.05 * maximum, 1.05 * maximum))
            ax.set_xlabel(f"{labels[x_dataset]} Tumor Cells Feature Importance (%)" if manuscript_panel else f"{labels[x_dataset]} Tumor Feature Importance (%)", fontsize=22 * scale, fontweight="bold")
            ax.set_ylabel(f"{labels[y_dataset]} Tumor Cells Feature Importance (%)" if manuscript_panel else f"{labels[y_dataset]} Tumor Feature Importance (%)", fontsize=22 * scale, fontweight="bold")
            ax.set_title(f"{names.get(enc, enc)} (N features = {len(table)})", fontsize=24 * scale, fontweight="bold")
            ax.tick_params(axis="both", labelsize=20 * scale, width=2 * scale, length=6 * scale)
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontweight("bold")
            ax.grid(True, alpha=.3, linestyle="--")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2 * scale)
                spine.set_color("black")
            # Start in the published corner, then avoid covering highlighted points.
            legend = ax.legend(loc="upper left", **legend_options)
            label_sets.append((ax, texts, legend, x, y, x_high | y_high))
            metrics.append(dict(view=view, encoder=enc, n_features=len(table), x_dataset=labels[x_dataset],
                                y_dataset=labels[y_dataset], spearman_all_dimensions=spearman(x, y),
                                x_threshold_percent=x_cut, y_threshold_percent=y_cut,
                                n_other=int(masks[0].sum()), n_x_only=int(masks[1].sum()),
                                n_y_only=int(masks[2].sum()), n_both=int(masks[3].sum())))
        fig.canvas.draw()
        for ax, texts, legend, x, y, high in label_sets:
            points = ax.transData.transform(np.column_stack([x[high], y[high]]))
            def legend_overlap():
                box = legend.get_window_extent().padded(3)
                return (sum(box.contains(*point) for point in points)
                        + sum(box.overlaps(text.get_window_extent()) for text in texts))

            best_location = "upper left"
            best_overlap = legend_overlap()
            if best_overlap:
                for location in ("upper right", "lower right"):
                    legend = ax.legend(loc=location, **legend_options)
                    fig.canvas.draw()
                    overlap = legend_overlap()
                    if overlap < best_overlap:
                        best_location, best_overlap = location, overlap
                    if best_overlap == 0:
                        break
                legend = ax.legend(loc=best_location, **legend_options)
                fig.canvas.draw()
                if best_overlap:
                    # Reserve an empty top band when every candidate corner contains data or labels.
                    maximum = max(x.max(), y.max())
                    ax.set(xlim=(-.05 * maximum, 1.4 * maximum),
                           ylim=(-.05 * maximum, 1.4 * maximum))
                    legend = ax.legend(loc="upper left", **legend_options)
                    fig.canvas.draw()
            if texts:
                # No arrows, as in the manuscript; repeatable placement for coincident labels.
                state = np.random.get_state()
                try:
                    np.random.seed(0)
                    legend_box = legend.get_window_extent().transformed(ax.transData.inverted())
                    adjust_text(texts, objects=[legend_box], ax=ax, iter_lim=500,
                                force_text=(.2, .4), force_pull=(.01, .01), expand=(1.05, 1.2),
                                prevent_crossings=False)
                    # Separate residual collisions pinned against the lower axis.
                    placed = [legend.get_window_extent()]
                    for text in sorted(texts, key=lambda item: item.get_position()[1]):
                        for _ in range(len(texts)):
                            box = text.get_window_extent().padded(1)
                            if not any(box.overlaps(other) for other in placed):
                                break
                            step = box.height + 2
                            if box.y1 + step > ax.get_window_extent().y1:
                                break
                            position = ax.transData.transform(text.get_position())
                            text.set_position(ax.transData.inverted().transform(position + [0, step]))
                        placed.append(text.get_window_extent().padded(1))
                finally:
                    np.random.set_state(state)
        for suffix in (("png", "pdf") if save_pdf else ("png",)):
            fig.savefig(output_dir / f"Figure_6B.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return pd.DataFrame(metrics)


def plot_figure7_scatter(profiles, output_dir, *, encoders=None,
                         aggregation="present_keys", save_pdf=True, views=None, manuscript_panel=False):
    """Plot the final full-vector BRCA Xenium versus COAD Visium panels."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    has_xenium = "brca_xenium_percent" in profiles.columns
    has_coad = "coad_visium_percent" in profiles.columns
    if not has_xenium:
        raise ValueError("Full Figure 7 comparison requires Xenium importance.")
    columns = []
    if has_coad:
        columns.append("coad_visium_percent")
    if has_xenium:
        columns.append("brca_xenium_percent")
    encoders = tuple(encoders) if encoders is not None else FIGURE7_ENCODERS
    if not encoders or len(set(encoders)) != len(encoders) or any(enc not in MODEL_NAMES for enc in encoders):
        raise ValueError("Select distinct supported encoders.")
    groups = {}
    for enc in encoders:
        table = profiles.loc[profiles.encoder == enc].sort_values("feature_index")
        values = table[list(columns)].to_numpy(dtype=float)
        ids = table.feature_index.to_numpy()
        valid_ids = (len(ids) >= 2 and len(set(ids)) == len(ids) and np.equal(ids, np.floor(ids)).all()
                     and (ids >= 0).all() and (ids < EXPECTED_DIMS[enc]).all())
        if (not valid_ids or not np.array_equal(ids, np.arange(EXPECTED_DIMS[enc]))
                or not np.isfinite(values).all() or (values < 0).any()
                or not np.allclose(values.sum(axis=0), 100)):
            raise ValueError(f"{enc}: all dimensions and full-vector percentages are required.")
        groups[enc] = table
    return _plot_full_figure7(groups, output_dir, save_pdf=save_pdf,
                              views=views, manuscript_panel=manuscript_panel)
