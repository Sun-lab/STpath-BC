"""Compare four crop sizes on the same sampled 300 µm tissue regions."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from . import tcga_brca_prediction as prediction
from .tcga_brca_images import (
    apply_audited_spacing, grid_edges, image_array, inspect_slide, rgb_region, select_slides,
)
from .he_tissue import MASK_METHOD_VERSION, color_tissue_mask_rgb
from .image_metadata_manifest import get_hw

SIZES_UM = (50, 75, 100, 150)
PARENT_UM = 300
MODE = "matched_regions_v1"
INTERPRETATION = (
    "Equal-weight child-tile means within the same 300 um parent regions. "
    "Each size covers the entire parent without overlaps or gaps; integer pixel rounding is recorded. "
    "Parents and every child at all four sizes must pass the tissue threshold. "
    "This describes crop-size consistency in selected tissue-rich regions, not accuracy or "
    "cell-count-weighted whole-slide proportions. Slide summaries are descriptive; regions "
    "within a slide are not independent biological replicates."
)


def child_boxes(bounds, size):
    """Partition the same integer-pixel rectangle exactly at every crop size."""
    n = PARENT_UM // size
    x0, y0, x1, y1 = bounds
    xs = np.rint(np.linspace(x0, x1, n + 1)).astype(int)
    ys = np.rint(np.linspace(y0, y1, n + 1)).astype(int)
    if np.any(np.diff(xs) < 1) or np.any(np.diff(ys) < 1):
        raise ValueError("Parent has too few native pixels for the requested child grids")
    return [(r + 1, c + 1, int(xs[c]), int(ys[r]), int(xs[c + 1]), int(ys[r + 1]))
            for r in range(n) for c in range(n)]


def box_tissue_fractions(integral, width, height, boxes):
    """Use one thumbnail integral image for all candidate-parent QC."""
    mh, mw = np.array(integral.shape) - 1
    boxes = np.asarray(boxes)
    x0 = np.clip(np.floor(boxes[:, 0] * mw / width).astype(int), 0, mw - 1)
    y0 = np.clip(np.floor(boxes[:, 1] * mh / height).astype(int), 0, mh - 1)
    x1 = np.clip(np.ceil(boxes[:, 2] * mw / width).astype(int), 1, mw)
    y1 = np.clip(np.ceil(boxes[:, 3] * mh / height).astype(int), 1, mh)
    return (integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]) / ((y1-y0)*(x1-x0))


def prepare_or_resume_matched_tiles(input_dir, audit_csv, slide_ids, output_dir, *,
                                    max_regions=32, min_tissue=.60, seed=42):
    """Select parent regions once, then retain complete four-size sets only."""
    if not isinstance(max_regions, int) or not 1 <= max_regions <= 1024:
        raise ValueError("max_regions must be an integer between 1 and 1024")
    if not math.isfinite(min_tissue) or not 0 < min_tissue <= 1:
        raise ValueError("min_tissue must be in (0,1]")
    source, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    audit_csv = Path(audit_csv).resolve()
    if output == source or source in output.parents:
        raise ValueError("Keep comparison outputs outside the TIFF directory")
    selected = select_slides(source, slide_ids, max_slides=3)
    settings = dict(sampling_mode=MODE, parent_size_um=PARENT_UM, sizes_um=list(SIZES_UM),
                    max_regions_per_slide=max_regions, min_tissue_fraction=min_tissue, seed=seed,
                    mask_method=MASK_METHOD_VERSION, audit_csv_sha256=prediction.sha256(audit_csv))
    marker = output / "tile_preparation.json"
    if marker.is_file():
        saved = json.loads(marker.read_text())
        if any(saved.get(k) != v for k, v in settings.items()):
            raise ValueError("Matched-region settings differ; use a new output folder")
        if [Path(m["source_path"]).resolve() for m in saved["source_metadata"]] != selected:
            raise ValueError("Matched-region source selection differs; use a new output folder")
        for item in saved["source_metadata"]:
            stat = Path(item["source_path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (item["file_size_bytes"], item["source_mtime_ns"]):
                raise ValueError("Source TIFF changed since matched-region preparation")
        if prediction.sha256(output / "matched_regions.csv") != saved["regions_sha256"]:
            raise ValueError("Matched-region manifest changed")
        table = prediction.load_prepared_tiles(output)
        validate_matched_manifest(table)
        print(f"Reusing matched-region tiles: {output}", flush=True)
        return table
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new empty matched-comparison folder: {output}")
    with audit_csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    audit = {r["slide_id"]: r for r in rows}
    if len(rows) != len(audit):
        raise ValueError("Duplicate audit slide IDs")
    metadata_rows = []
    for path in selected:
        metadata = inspect_slide(path)
        apply_audited_spacing(metadata, audit[path.stem])
        if metadata["orientation"] != 1:
            raise ValueError(f"Non-default TIFF orientation: {path}")
        metadata_rows.append(metadata)
    output.mkdir(parents=True, exist_ok=True)
    tiles, parents, summaries = [], [], []
    for path, metadata in zip(selected, metadata_rows):
        width, height = metadata["width_px"], metadata["height_px"]
        mpp_x, mpp_y = metadata["tiling_mpp_x"], metadata["tiling_mpp_y"]
        levels = [i for i, level in enumerate(metadata["levels"]) if max(get_hw(tuple(level["shape"]))) >= 1024]
        level = levels[-1] if levels else 0
        lh, lw = get_hw(tuple(metadata["levels"][level]["shape"]))
        with image_array(path, metadata["series_index"], level) as (pixels, axes, _):
            thumb = rgb_region(pixels, axes, (0, 0, lw, lh), max(1, math.ceil(max(lw, lh) / 1536)))
        mask, _, _ = color_tissue_mask_rgb(thumb)
        integral = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        xs, ys = grid_edges(width, height, "physical_um", PARENT_UM, mpp_x, mpp_y)
        parent_width, parent_height = max(1, round(PARENT_UM/mpp_x)), max(1, round(PARENT_UM/mpp_y))
        eligible, n_full, n_parent_pass = [], 0, 0
        for r in range(len(ys) - 1):
            for c in range(len(xs) - 1):
                bounds = tuple(map(int, (xs[c], ys[r], xs[c+1], ys[r+1])))
                if bounds[2]-bounds[0] != parent_width or bounds[3]-bounds[1] != parent_height:
                    continue
                n_full += 1
                parent_fraction = float(box_tissue_fractions(integral, width, height, [bounds])[0])
                if parent_fraction < min_tissue:
                    continue
                n_parent_pass += 1
                children = [box[2:] for size in SIZES_UM for box in child_boxes(bounds, size)]
                minimum = float(box_tissue_fractions(integral, width, height, children).min())
                if minimum >= min_tissue:
                    eligible.append((r + 1, c + 1, bounds, parent_fraction, minimum))
        sample_seed = int.from_bytes(hashlib.sha256(f"{seed}/{path.stem}/{MODE}".encode()).digest()[:8], "big")
        chosen = prediction.systematic_sample(np.arange(len(eligible)), max_regions, sample_seed)
        if not len(chosen):
            raise ValueError(f"No complete matched regions pass tissue QC: {path.stem}; inspect the mask and tissue threshold")
        print(f"{path.stem}: {len(chosen)}/{len(eligible)} matched regions; "
              f"{n_full} full parents, {n_parent_pass} pass parent QC", flush=True)
        overview = Image.fromarray(thumb)
        draw = ImageDraw.Draw(overview)
        with image_array(path, metadata["series_index"], 0) as (pixels, axes, _):
            for number, index in enumerate(chosen, 1):
                pr, pc, bounds, parent_fraction, minimum = eligible[index]
                parent_id = f"{path.stem}__parent300__r{pr}_c{pc}"
                common = dict(slide_id=path.stem, patient_id=metadata["patient_id"],
                              parent_region_id=parent_id, parent_row=pr, parent_col=pc,
                              parent_x0_px=bounds[0], parent_y0_px=bounds[1], parent_x1_px=bounds[2], parent_y1_px=bounds[3])
                parents.append(dict(common, parent_width_um=(bounds[2]-bounds[0])*mpp_x,
                                    parent_height_um=(bounds[3]-bounds[1])*mpp_y,
                                    parent_tissue_fraction=parent_fraction, minimum_child_tissue_fraction=minimum))
                draw.rectangle((bounds[0]*thumb.shape[1]/width, bounds[1]*thumb.shape[0]/height,
                                bounds[2]*thumb.shape[1]/width, bounds[3]*thumb.shape[0]/height), outline="yellow", width=1)
                for size in SIZES_UM:
                    candidate = f"physical_um_{size}"
                    folder = output / "tiles" / path.stem / candidate / f"parent_r{pr}_c{pc}"
                    folder.mkdir(parents=True)
                    boxes = child_boxes(bounds, size)
                    fractions = box_tissue_fractions(integral, width, height, [b[2:] for b in boxes])
                    for (cr, cc, x0, y0, x1, y1), fraction in zip(boxes, fractions):
                        tile_id = f"{parent_id}__{candidate}__r{cr}_c{cc}"
                        tile_path = folder / f"r{cr}_c{cc}.png"
                        Image.fromarray(rgb_region(pixels, axes, (x0, y0, x1, y1))).save(tile_path)
                        tiles.append(dict(common, tile_id=tile_id, candidate=candidate, nominal_size_um=size,
                                          child_row=cr, child_col=cc, x0_px=x0, y0_px=y0, x1_px=x1, y1_px=y1,
                                          tile_width_um=(x1-x0)*mpp_x, tile_height_um=(y1-y0)*mpp_y,
                                          tissue_fraction_thumbnail_estimate=float(fraction),
                                          materialized_tile_path=str(tile_path), tile_sha256=prediction.sha256(tile_path)))
                if number % 8 == 0 or number == len(chosen):
                    print(f"Cropping {path.stem}: {number}/{len(chosen)} matched regions", flush=True)
        overview.save(output / f"{path.stem}__matched_regions.png")
        for size in SIZES_UM:
            summaries.append(dict(slide_id=path.stem, candidate=f"physical_um_{size}",
                                  n_full_parent_regions=n_full, n_parent_tissue_pass=n_parent_pass,
                                  n_eligible_matched_regions=len(eligible), n_selected_regions=len(chosen),
                                  n_children_per_region=(PARENT_UM//size)**2,
                                  n_sampled_tiles=len(chosen)*(PARENT_UM//size)**2))
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (metadata["file_size_bytes"], metadata["source_mtime_ns"]):
            raise ValueError(f"Source TIFF changed while cropping: {path}")
    # Encoder extraction groups by slide/candidate: the manifest must use that same order.
    table = pd.DataFrame(tiles).sort_values(["slide_id", "nominal_size_um", "parent_row", "parent_col", "child_row", "child_col"])
    validate_matched_manifest(table)
    table.to_csv(output / "sampled_tiles.csv", index=False)
    pd.DataFrame(parents).to_csv(output / "matched_regions.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "sampling_summary.csv", index=False)
    prediction.write_json(marker, dict(settings, status="complete", recipe=prediction.RECIPE,
                                      source_metadata=metadata_rows, audit_csv=str(audit_csv),
                                      max_tiles_per_slide_size=max_regions*36,
                                      partial_edges="excluded", sampling="random-start systematic sample of eligible parent regions; all children retained",
                                      tissue_qc="parent and every child at every size must pass; otherwise reject whole parent",
                                      aggregation="equal-weight child-tile mean within each parent and size",
                                      manifest_sha256=prediction.sha256(output / "sampled_tiles.csv"),
                                      regions_sha256=prediction.sha256(output / "matched_regions.csv")))
    return table.reset_index(drop=True)


def validate_matched_manifest(table):
    """Reject incomplete sets or geometries that do not partition the same parent."""
    if table.empty or table.tile_id.duplicated().any():
        raise ValueError("Empty or duplicate matched tiles")
    parent_keys = ["parent_x0_px", "parent_y0_px", "parent_x1_px", "parent_y1_px"]
    for (_, parent), group in table.groupby(["slide_id", "parent_region_id"], sort=False):
        if len(group[parent_keys].drop_duplicates()) != 1 or set(group.nominal_size_um) != set(SIZES_UM):
            raise ValueError(f"Incomplete or inconsistent matched region: {parent}")
        bounds = tuple(group.iloc[0][parent_keys].astype(int))
        for size, children in group.groupby("nominal_size_um", sort=False):
            actual = list(children.sort_values(["child_row", "child_col"])[
                ["child_row", "child_col", "x0_px", "y0_px", "x1_px", "y1_px"]].itertuples(index=False, name=None))
            if actual != child_boxes(bounds, int(size)) or not children.candidate.eq(f"physical_um_{int(size)}").all():
                raise ValueError(f"Incomplete or inconsistent child grid: {parent}, {size} um")


def summarize_matched_predictions(output_dir, predictions, cell_types):
    """Aggregate only complete paired regions, retaining raw out-of-range values."""
    output = Path(output_dir)
    marker = output / "matched_comparison_summary.json"
    marker.unlink(missing_ok=True)
    meta = json.loads((output / "tile_preparation.json").read_text())
    if meta.get("sampling_mode") != MODE or prediction.sha256(output / "sampled_tiles.csv") != meta["manifest_sha256"]:
        raise ValueError("Matched comparison requires its unchanged prepared manifest")
    tiles = pd.read_csv(output / "sampled_tiles.csv")
    validate_matched_manifest(tiles)
    if (predictions.duplicated(["tile_id", "cell_type"]).any() or
            set(predictions.cell_type) != set(cell_types) or not np.isfinite(predictions.predicted_raw).all()):
        raise ValueError("Invalid matched prediction targets, duplicate rows or non-finite values")
    region_tables = []
    for cell_type in cell_types:
        values = predictions.loc[predictions.cell_type == cell_type, ["tile_id", "predicted_raw"]]
        if len(values) != len(tiles) or set(values.tile_id) != set(tiles.tile_id):
            raise ValueError(f"Incomplete matched predictions: {cell_type}")
        joined = tiles.merge(values, on="tile_id", validate="one_to_one")
        region = joined.groupby(["slide_id", "patient_id", "parent_region_id", "nominal_size_um"], sort=False).agg(
            mean_predicted_raw=("predicted_raw", "mean"), n_child_tiles=("tile_id", "size"))
        region["cell_type"] = cell_type
        region_tables.append(region.reset_index())
    regions = pd.concat(region_tables, ignore_index=True)
    paired = regions.pivot(index=["slide_id", "patient_id", "parent_region_id", "cell_type"],
                           columns="nominal_size_um", values="mean_predicted_raw")
    paired.columns = [f"mean_{int(s)}um" for s in paired.columns]
    paired = paired.reset_index()
    summaries = []
    for size in SIZES_UM:
        paired[f"difference_{size}um_minus_150um"] = paired[f"mean_{size}um"] - paired.mean_150um
        for (slide, target), group in paired.groupby(["slide_id", "cell_type"], sort=False):
            delta = group[f"difference_{size}um_minus_150um"].to_numpy()
            values, reference = group[f"mean_{size}um"].to_numpy(), group.mean_150um.to_numpy()
            summaries.append(dict(slide_id=slide, cell_type=target, size_um=size, n_regions=len(group),
                                  mean_predicted_raw=float(values.mean()), mean_difference_vs_150=float(delta.mean()),
                                  mean_absolute_difference_vs_150=float(np.abs(delta).mean()),
                                  rmse_vs_150=float(np.sqrt(np.mean(delta**2))),
                                  pearson_r_vs_150=float(np.corrcoef(values, reference)[0, 1])
                                  if len(group) > 1 and np.std(values) > 0 and np.std(reference) > 0 else None))
    files = {"matched_region_predictions.csv": regions, "matched_region_comparisons.csv": paired,
             "matched_slide_summary.csv": pd.DataFrame(summaries)}
    for name, frame in files.items():
        frame.to_csv(output / name, index=False)
    prediction.write_json(marker, dict(status="complete", parent_size_um=PARENT_UM, sizes_um=list(SIZES_UM),
                                      reference_size_um=150, n_regions=int(tiles.parent_region_id.nunique()),
                                      cell_types=list(cell_types), interpretation=INTERPRETATION,
                                      file_sha256={name: prediction.sha256(output / name) for name in
                                                   [*files, "tile_predictions_raw.csv.gz", "tile_preparation.json"]}))
    return regions, pd.DataFrame(summaries), paired


def save_matched_plots(output_dir):
    """Paired scatterplots and separate difference histograms for every crop size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    output = Path(output_dir)
    record = json.loads((output / "matched_comparison_summary.json").read_text())
    completion = json.loads((output / "prediction_summary.json").read_text())
    if completion.get("status") != "complete" or completion.get("tile_selection") != "matched_regions":
        raise ValueError("Finish matched-region prediction before plotting")
    if record["status"] != "complete" or any(prediction.sha256(output / name) != checksum
                                            for name, checksum in record["file_sha256"].items()):
        raise ValueError("Matched comparison outputs changed or are incomplete")
    pairs = pd.read_csv(output / "matched_region_comparisons.csv")
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    colors = {50: "#0072B2", 75: "#D55E00", 100: "#009E73"}
    for target, rows in pairs.groupby("cell_type", sort=False):
        slides = rows.slide_id.drop_duplicates().tolist()
        scatter, axes = plt.subplots(len(slides), 3, figsize=(13.5, 3.8*len(slides)), squeeze=False)
        differences, hist_axes = plt.subplots(len(slides), 3, figsize=(13.5, 3.5*len(slides)), squeeze=False)
        for i, slide in enumerate(slides):
            group = rows.loc[rows.slide_id == slide]
            values = group[[f"mean_{s}um" for s in SIZES_UM]].to_numpy()
            pad = max(.01, float(np.ptp(values))*.05)
            limits = (float(values.min())-pad, float(values.max())+pad)
            deltas = group[[f"difference_{s}um_minus_150um" for s in colors]].to_numpy()*100
            span = max(float(np.abs(deltas).max())*1.05, .1)
            bins = np.linspace(-span, span, 17)
            peak_count = max(np.histogram(deltas[:, j], bins=bins)[0].max() for j in range(len(colors)))
            for j, size in enumerate(colors):
                ax = axes[i, j]
                delta = group[f"difference_{size}um_minus_150um"].to_numpy()*100
                ax.scatter(group.mean_150um, group[f"mean_{size}um"], s=20, alpha=.7, color=colors[size])
                ax.plot(limits, limits, color="gray", linestyle="--", linewidth=1)
                ax.set(xlim=limits, ylim=limits, xlabel="150 µm: parent-region mean", ylabel=f"{size} µm: parent-region mean",
                       title=f"{slide}\n{size} vs 150 µm; n={len(group)}")
                ax.text(.04, .96, f"Mean Δ: {delta.mean():+.2f} pp\nMean |Δ|: {np.abs(delta).mean():.2f} pp",
                        transform=ax.transAxes, va="top", fontsize=9)
                ax.set_aspect("equal", adjustable="box")
                ax.grid(alpha=.15)
                hist = hist_axes[i, j]
                hist.hist(delta, bins=bins, color=colors[size], alpha=.85, edgecolor="white", linewidth=.7)
                hist.axvline(0, color="#555555", linestyle="--", linewidth=1)
                hist.set(xlim=(-span, span), ylim=(0, max(2, peak_count*1.3)),
                         title=f"{slide}\n{size} − 150 µm; n={len(group)}",
                         xlabel="Difference (percentage points)", ylabel="Parent regions" if j == 0 else "")
                hist.text(.97, .96, f"Mean Δ: {delta.mean():+.2f} pp", ha="right", va="top",
                          transform=hist.transAxes, fontsize=9,
                          bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=2))
                hist.yaxis.set_major_locator(MaxNLocator(integer=True))
                hist.set_axisbelow(True)
                hist.grid(axis="y", alpha=.15)
        for fig, suffix in ((scatter, "matched_scatter"), (differences, "matched_differences")):
            fig.suptitle(f"{target} — matched 300 µm regions")
            fig.tight_layout(rect=(0, 0, 1, .97))
            for ext in ("png", "pdf"):
                fig.savefig(figure_dir / f"{prediction.sx.safe_name(target)}__{suffix}.{ext}", dpi=160)
            plt.close(fig)
