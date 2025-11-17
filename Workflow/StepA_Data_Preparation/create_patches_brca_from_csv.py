#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BRCA Visium: Create image patches from CSV metadata (per spot), with optional Otsu segmentation

Features
--------
- Select specific sample with --sid (or row with --row-idx)
- Optional output root override with --output-dir (keeps per-sample subfolder)
- Otsu tissue segmentation (unless --disable-otsu), then prune regions with < --min-region-spots
- White-background filter (skip if bright-pixel proportion > --white-thresh, default 0.6)
- Visualization:
    * Dots = kept spots colored by region
    * Crosses = out-of-tissue (different colors for reasons)
- Save out-of-tissue patches into <sid>_out_of_tissue/{in_tissue0,no_region,high_white}/
- Per-sample summary CSV + per-spot table with 'tissue_region' and 'oot_reason'
"""

import os
import gzip
import argparse
import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # allow very large WSIs

# ------------------------- Utilities -------------------------
def pil_to_rgb_numpy(pil_img: Image.Image) -> np.ndarray:
    """
    Return an (H, W, 3) uint8 array.
    - If RGBA: alpha-composite onto white, then drop A.
    - If P/LA/L: convert to RGB.
    - Else: convert to RGB if needed.
    """
    mode = pil_img.mode
    if mode == "RGBA":
        bg = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
        pil_img = Image.alpha_composite(bg, pil_img).convert("RGB")
    elif mode in ("LA", "P", "L"):
        pil_img = pil_img.convert("RGB")
    elif mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return np.array(pil_img)


def _read_barcodes_any(path):
    """Accept file.tsv(.gz) or a directory containing barcodes.tsv(.gz). Return list[str] or None."""
    if path is None or (isinstance(path, float) and np.isnan(path)):
        return None
    candidates = []
    if os.path.isdir(path):
        candidates += [os.path.join(path, "barcodes.tsv"),
                       os.path.join(path, "barcodes.tsv.gz")]
    else:
        path = str(path)
        candidates.append(path)
        if path.endswith(".tsv") and os.path.exists(path + ".gz"):
            candidates.append(path + ".gz")
    for p in candidates:
        if os.path.exists(p):
            if p.endswith(".gz"):
                df = pd.read_csv(gzip.open(p, 'rt'), sep='\t', header=None, index_col=None)
            else:
                df = pd.read_csv(p, sep='\t', header=None, index_col=None)
            if df.shape[1] == 1:
                return df[0].astype(str).tolist()
            else:
                return [str(x) for x in df[0].tolist() if str(x) != 'nan']
    return None


def load_spatial_with_indices(spatial_path, barcode_col, pxl_x_col, pxl_y_col,
                              array_row_col=None, array_col_col=None, in_tissue_col=None):
    """
    Read a generic spatial CSV and rename selected columns to a standard schema.
    Required: 'barcode','pxl_x','pxl_y'
    Optional: 'array_row','array_col','in_tissue'
    Column indices are 1-based in the metadata CSV.
    """
    sp = pd.read_csv(spatial_path, header=0, index_col=None)
    sp.columns = [f'col_{i+1}' for i in range(sp.shape[1])]
    ren = {
        f'col_{int(barcode_col)}': 'barcode',
        f'col_{int(pxl_x_col)}': 'pxl_x',
        f'col_{int(pxl_y_col)}': 'pxl_y'
    }
    if array_row_col is not None and not pd.isna(array_row_col):
        ren[f'col_{int(array_row_col)}'] = 'array_row'
    if array_col_col is not None and not pd.isna(array_col_col):
        ren[f'col_{int(array_col_col)}'] = 'array_col'
    if in_tissue_col is not None and not pd.isna(in_tissue_col):
        ren[f'col_{int(in_tissue_col)}'] = 'in_tissue'
    sp = sp.rename(columns=ren)

    keep = ['barcode', 'pxl_x', 'pxl_y']
    for c in ['array_row', 'array_col', 'in_tissue']:
        if c in sp.columns:
            keep.append(c)
    sp = sp[keep].dropna()

    sp['barcode'] = sp['barcode'].astype(str)
    sp['pxl_x'] = sp['pxl_x'].astype(float)
    sp['pxl_y'] = sp['pxl_y'].astype(float)
    if 'array_row' in sp.columns: sp['array_row'] = sp['array_row'].astype(int)
    if 'array_col' in sp.columns: sp['array_col'] = sp['array_col'].astype(int)
    if 'in_tissue' in sp.columns: sp['in_tissue'] = sp['in_tissue'].astype(int)
    return sp


def median_neighbor_radius(sp, default=None):
    """
    Estimate patch radius from neighbor spacing using array grid if available;
    otherwise return 'default' or 64 px.
    """
    if 'array_row' in sp.columns and 'array_col' in sp.columns:
        dy = []
        for r in range(sp['array_row'].min(), sp['array_row'].max()):
            a = sp[sp['array_row'] == r].sort_values('array_col')['pxl_y'].to_numpy()
            b = sp[sp['array_row'] == r+1].sort_values('array_col')['pxl_y'].to_numpy()
            m = min(len(a), len(b))
            if m > 0:
                dy.extend(list(np.abs(a[:m] - b[:m])))
        dx = []
        for c in range(sp['array_col'].min(), sp['array_col'].max()-1):
            a = sp[sp['array_col'] == c].sort_values('array_row')['pxl_x'].to_numpy()
            b = sp[sp['array_col'] == c+2].sort_values('array_row')['pxl_x'].to_numpy()
            m = min(len(a), len(b))
            if m > 0:
                dx.extend(list(np.abs(a[:m] - b[:m])))
        vals = [v for v in dx + dy if np.isfinite(v)]
        if len(vals) > 0:
            return float(np.median(vals) / 2.0)
    if default is not None:
        return float(default)
    return 64.0


# ------------------------- Visualization -------------------------
def save_regions_viz(image_np, labels, spots_df, out_png, downsample=4):
    """
    Save an overview PNG:
      Left: Otsu region labels
      Right: WSI + spot scatter: dots (kept) colored by region; crosses for OOT reasons
        - blue  x : in_tissue=0
        - orange x: no_region
        - red   x: high_white
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Downsample for speed
    if downsample and downsample > 1:
        img_ds = image_np[::downsample, ::downsample]
        lab_ds = labels[::downsample, ::downsample] if labels is not None else None
        scale = 1.0 / downsample
        xs = spots_df["pxl_x"].to_numpy() * scale
        ys = spots_df["pxl_y"].to_numpy() * scale
    else:
        img_ds = image_np
        lab_ds = labels
        xs = spots_df["pxl_x"].to_numpy()
        ys = spots_df["pxl_y"].to_numpy()

    regs = spots_df["tissue_region"].to_numpy()
    oot = spots_df["oot_reason"].fillna("").to_numpy()

    fig = plt.figure(figsize=(13, 6))

    # Left
    ax1 = plt.subplot(1, 2, 1)
    if lab_ds is not None:
        ax1.imshow(lab_ds, interpolation="nearest")
        ax1.set_title("Otsu region labels")
    else:
        ax1.imshow(np.zeros(img_ds.shape[:2]), interpolation="nearest")
        ax1.set_title("No Otsu (single region)")
    ax1.axis("off")

    # Right
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(img_ds, interpolation="nearest")

    # Dots: kept (no OOT) and inside regions
    keep_mask = (spots_df["tissue_region"] >= 0) & (spots_df["oot_reason"].isna())
    if keep_mask.any():
        ax2.scatter(xs[keep_mask], ys[keep_mask], c=regs[keep_mask], s=8, alpha=0.9)

    # Crosses by OOT reason
    reason_colors = {"in_tissue0":"blue", "no_region":"orange", "high_white":"red"}
    for reason, color in reason_colors.items():
        m = (oot == reason)
        if np.any(m):
            ax2.scatter(xs[m], ys[m], marker='x', s=36, linewidths=1.6, c=color, alpha=0.95)

    ax2.set_title("Spots by region (dots) & out-of-tissue (crosses)")
    ax2.axis("off")

    # Legend
    legend_elems = [
        Line2D([0],[0], marker='o', color='w',
               label='Kept (dots, colored by region)',
               markerfacecolor='gray', markersize=6, markeredgewidth=0),
        Line2D([0],[0], marker='x', color='blue',   label='OOT: in_tissue=0', markersize=8, linewidth=1.6),
        Line2D([0],[0], marker='x', color='orange', label='OOT: no_region',   markersize=8, linewidth=1.6),
        Line2D([0],[0], marker='x', color='red',    label='OOT: high_white',  markersize=8, linewidth=1.6),
    ]
    ax2.legend(handles=legend_elems, loc='lower right', frameon=True, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------- Main per-row logic -------------------------
def process_row(row, args):
    # Identify sample id column: prefer 'sid', fallback to 'sample_id' or 'slide'
    if 'sid' in row.index:
        sample_id = str(row['sid'])
    elif 'sample_id' in row.index:
        sample_id = str(row['sample_id'])
    elif 'slide' in row.index:
        sample_id = str(row['slide'])
    else:
        raise ValueError("CSV must include 'sid' or 'sample_id' (or 'slide') column.")

    image_path   = str(row['image_path'])
    spatial_path = str(row['spatial_path'])

    # Determine output directory (with optional override of root)
    if 'output_dir' in row.index and pd.notna(row['output_dir']):
        orig_output_dir = str(row['output_dir'])
    elif 'patch_dir' in row.index and pd.notna(row['patch_dir']):
        orig_output_dir = str(row['patch_dir'])
    else:
        raise ValueError("CSV must include 'output_dir' or 'patch_dir' column.")

    if args.output_dir:
        sample_subdir = os.path.basename(os.path.normpath(orig_output_dir))
        output_dir = os.path.join(args.output_dir, sample_subdir)
        print(f"[Override output] {orig_output_dir} -> {output_dir}")
    else:
        output_dir = orig_output_dir
    os.makedirs(output_dir, exist_ok=True)

    # column indices (1-based)
    barcode_col   = int(row['barcode_col'])
    pxl_x_col     = int(row['pxl_x_col'])
    pxl_y_col     = int(row['pxl_y_col'])
    array_row_col = row.get('array_row_col', None)
    array_col_col = row.get('array_col_col', None)
    in_tissue_col = row.get('in_tissue_col', None)
    barcodes_path = row.get('barcodes_path', None)

    sp = load_spatial_with_indices(
        spatial_path, barcode_col, pxl_x_col, pxl_y_col,
        array_row_col=array_row_col, array_col_col=array_col_col, in_tissue_col=in_tissue_col
    )
    if 'in_tissue' in sp.columns:
        sp['in_tissue'] = sp['in_tissue'].astype(int)

    # optional barcode filter
    barcodes = _read_barcodes_any(barcodes_path) if barcodes_path and str(barcodes_path) != 'nan' else None
    if barcodes is not None:
        sp = sp[sp['barcode'].astype(str).isin(barcodes)].copy()

    # Load image safely to RGB
    img = Image.open(image_path)
    img_np = pil_to_rgb_numpy(img)
    H, W = img_np.shape[:2]

    # Otsu segmentation + region labeling
    if args.disable_otsu:
        sp['tissue_region'] = 0
        valid_regions = [0]
        labels = None
    else:
        from skimage.color import rgb2gray
        from skimage.filters import threshold_otsu
        from skimage.morphology import closing, remove_small_objects, dilation, erosion, disk
        from skimage import measure

        gray = rgb2gray(img_np)
        thresh = threshold_otsu(gray)
        binary = gray < thresh
        cleaned = closing(binary, disk(15))
        cleaned = remove_small_objects(cleaned, min_size=2000)
        cleaned = dilation(cleaned, disk(10))
        cleaned = erosion(cleaned, disk(8))
        labels = measure.label(cleaned)

        rid = []
        for _, s in sp.iterrows():
            x = int(round(s['pxl_x'])); y = int(round(s['pxl_y']))
            rid.append(int(labels[y, x]) if (0 <= x < W and 0 <= y < H) else -1)
        sp['tissue_region'] = rid

        # prune tiny or background regions
        region_counts = sp[sp['tissue_region'] >= 0]['tissue_region'].value_counts()
        small = set([r for r, c in region_counts.items() if c < args.min_region_spots])
        sp.loc[sp['tissue_region'].isin(small), 'tissue_region'] = -1
        sp.loc[sp['tissue_region'] == 0, 'tissue_region'] = -1

        kept = sorted(sp[sp['tissue_region'] >= 0]['tissue_region'].unique())
        mapping = {old: i for i, old in enumerate(kept)}
        sp.loc[sp['tissue_region'] >= 0, 'tissue_region'] = sp.loc[sp['tissue_region'] >= 0, 'tissue_region'].map(mapping)
        valid_regions = [int(r) for r in sorted(sp['tissue_region'].unique()) if r >= 0]

    # Out-of-tissue tagging, priority: in_tissue0 > no_region > high_white
    sp['oot_reason'] = pd.Series([None] * len(sp), index=sp.index, dtype=object)
    if 'in_tissue' in sp.columns:
        sp.loc[sp['in_tissue'] == 0, 'oot_reason'] = 'in_tissue0'
    sp.loc[(sp['oot_reason'].isna()) & (sp['tissue_region'] < 0), 'oot_reason'] = 'no_region'

    # Radius
    radius = None
    if 'diameter' in row.index and pd.notna(row['diameter']):
        try:
            d = float(row['diameter'])
            if d > 0:
                radius = float(args.factor) * d / 2.0
        except Exception:
            radius = None
    if radius is None:
        radius = median_neighbor_radius(sp, default=args.default_radius)

    # Helper to crop a patch (returns None if invalid box)
    def crop_patch(cx, cy, r):
        x0 = max(0, int(round(cx - r)))
        y0 = max(0, int(round(cy - r)))
        x1 = min(W, int(round(cx + r)))
        y1 = min(H, int(round(cy + r)))
        if x1 <= x0 or y1 <= y0:
            return None
        return img_np[y0:y1, x0:x1]

    # Prepare directories
    oot_root = os.path.join(output_dir, f"{sample_id}_out_of_tissue")
    oot_dirs = {
        "in_tissue0": os.path.join(oot_root, "in_tissue0"),
        "no_region":  os.path.join(oot_root, "no_region"),
        "high_white": os.path.join(oot_root, "high_white"),
    }
    for d in oot_dirs.values():
        os.makedirs(d, exist_ok=True)

    ext = '.tif' if args.output_format == 'tiff' else '.jpg'
    total_saved = 0
    skipped_white = 0
    spots_in_kept = 0

    # Save kept regions (or flat)
    for rid in (valid_regions if len(valid_regions) > 0 else [0]):
        if args.flat_output:
            save_dir = os.path.join(output_dir, f"{sample_id}")
            os.makedirs(save_dir, exist_ok=True)
            name_suffix = ""
        else:
            save_dir = os.path.join(output_dir, f"{sample_id}_region{rid}")
            os.makedirs(save_dir, exist_ok=True)
            name_suffix = f"_region{rid}"

        sub_idx = sp.index[sp['tissue_region'] == rid] if 'tissue_region' in sp.columns else sp.index
        spots_in_kept += len(sub_idx)

        for idx in sub_idx:
            # If already OOT due to in_tissue0, send to OOT folder and skip kept
            if pd.notna(sp.at[idx, 'oot_reason']) and sp.at[idx, 'oot_reason'] in ('in_tissue0',):
                srow = sp.loc[idx]
                crop = crop_patch(float(srow['pxl_x']), float(srow['pxl_y']), radius)
                if crop is not None:
                    out_fn = os.path.join(oot_dirs['in_tissue0'], f"{srow['barcode']}_{sample_id}_in_tissue0{ext}")
                    Image.fromarray(crop).save(out_fn)
                continue

            srow = sp.loc[idx]
            crop = crop_patch(float(srow['pxl_x']), float(srow['pxl_y']), radius)
            if crop is None:
                continue

            # White filter for kept; if fails -> mark high_white and save to OOT
            if not args.no_white_filter:
                gray = crop.mean(axis=2) / 255.0 if crop.ndim == 3 else (crop / 255.0)
                white_ratio = float((gray > args.white_cutoff).mean())
                if white_ratio > args.white_thresh:
                    sp.at[idx, 'oot_reason'] = 'high_white'
                    out_fn = os.path.join(oot_dirs['high_white'], f"{srow['barcode']}_{sample_id}_high_white{ext}")
                    Image.fromarray(crop).save(out_fn)
                    skipped_white += 1
                    continue

            out_fn = os.path.join(save_dir, f"{srow['barcode']}_{sample_id}{name_suffix}{ext}")
            Image.fromarray(crop).save(out_fn)
            total_saved += 1

    # Save OOT 'no_region' patches (those never entered kept loop)
    no_reg_idx = sp.index[sp['oot_reason'] == 'no_region']
    for idx in no_reg_idx:
        srow = sp.loc[idx]
        crop = crop_patch(float(srow['pxl_x']), float(srow['pxl_y']), radius)
        if crop is not None:
            out_fn = os.path.join(oot_dirs['no_region'], f"{srow['barcode']}_{sample_id}_no_region{ext}")
            Image.fromarray(crop).save(out_fn)

    # Visualization
    viz_path = ""
    if (not args.disable_otsu) and args.viz_regions:
        viz_path = os.path.join(output_dir, f"{sample_id}_regions_overview.png")
        try:
            save_regions_viz(img_np, labels, sp, viz_path, downsample=args.viz_downsample)
            print(f"[{sample_id}] saved regions visualization -> {viz_path}")
        except Exception as e:
            print(f"[{sample_id}] WARN: failed to save regions viz: {e}")
            viz_path = ""

    # Summary
    n_in_tissue0 = int((sp['oot_reason'] == 'in_tissue0').sum()) if 'in_tissue' in sp.columns else 0
    n_no_region  = int((sp['oot_reason'] == 'no_region').sum())
    n_high_white = int((sp['oot_reason'] == 'high_white').sum())

    print(f"[{sample_id}] regions={valid_regions}  radius={radius:.2f}  patches={total_saved}  "
          f"skipped_white={skipped_white} (cutoff={args.white_cutoff}, thresh={args.white_thresh}) "
          f"oot_counts: in_tissue0={n_in_tissue0}, no_region={n_no_region}, high_white={n_high_white}")

    summary = pd.DataFrame([{
        "sample_id": sample_id,
        "radius_pixels": float(radius),
        "n_regions_kept": int(len(valid_regions)),
        "regions_kept": ";".join(str(x) for x in valid_regions),
        "spots_in_kept_regions": int(spots_in_kept),
        "patches_saved": int(total_saved),
        "patches_skipped_white": int(skipped_white),
        "oot_in_tissue0": n_in_tissue0,
        "oot_no_region": n_no_region,
        "oot_high_white": n_high_white,
        "white_cutoff": float(args.white_cutoff),
        "white_thresh": float(args.white_thresh),
        "viz_path": viz_path,
        "oot_dir": oot_root,
        "output_dir_final": output_dir
    }])
    summary.to_csv(os.path.join(output_dir, f"{sample_id}_summary.csv"), index=False)

    # Save per-spot table including oot_reason
    extra = [c for c in ['array_row','array_col','in_tissue'] if c in sp.columns]
    cols = ['barcode','pxl_x','pxl_y'] + extra + ['tissue_region','oot_reason']
    sp.to_csv(os.path.join(output_dir, f"{sample_id}_spots_coordinates_with_regions.csv"),
              columns=cols, index=False)


# ------------------------- CLI -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True, help="Metadata CSV for patch creation")
    ap.add_argument("--sid", type=str, default=None,
                    help="Run only one sample by ID (matches 'sid' or 'sample_id' or 'slide'). Overrides --row-idx.")
    ap.add_argument("--row-idx", type=int, default=None,
                    help="Optional row index to run from CSV (0-based).")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="OPTIONAL: override root output dir from CSV; keeps per-sample subfolder.")
    ap.add_argument("--output-format", choices=["jpg","tiff"], default="tiff")

    # Segmentation / regions
    ap.add_argument("--disable-otsu", action="store_true",
                    help="Skip Otsu; treat all spots as a single region (region 0).")
    ap.add_argument("--min-region-spots", type=int, default=30,
                    help="Prune Otsu regions with fewer than this many spots.")
    ap.add_argument("--flat-output", action="store_true",
                    help="Save kept patches into <output_dir>/<sid>/ (no region suffix/folders).")

    # Patch geometry
    ap.add_argument("--factor", type=float, default=1.1,
                    help="If 'diameter' is present in CSV, radius = factor*diameter/2.")
    ap.add_argument("--default-radius", type=float, default=64.0,
                    help="Fallback pixel radius if grid-based estimate is unavailable.")

    # White-background filter
    ap.add_argument("--white-cutoff", type=float, default=0.9,
                    help="Grayscale cutoff in [0,1] defining 'white' pixels.")
    ap.add_argument("--white-thresh", type=float, default=0.6,
                    help="Skip kept patch if white proportion exceeds this.")
    ap.add_argument("--no-white-filter", action="store_true",
                    help="Disable white-background filtering.")

    # Visualization
    ap.add_argument("--viz-regions", action="store_true", help="Save regions overview PNG")
    ap.add_argument("--viz-downsample", type=int, default=4, help="Downsample factor for viz")

    args = ap.parse_args()

    df = pd.read_csv(args.input_csv).dropna(subset=['image_path']).reset_index(drop=True)

    # Select rows
    if args.sid is not None:
        idcol = 'sid' if 'sid' in df.columns else ('sample_id' if 'sample_id' in df.columns else ('slide' if 'slide' in df.columns else None))
        if idcol is None:
            raise ValueError("Neither 'sid' nor 'sample_id' nor 'slide' column present in CSV, cannot use --sid.")
        df = df[df[idcol].astype(str) == str(args.sid)].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No row matched --sid '{args.sid}' in column '{idcol}'.")
    elif args.row_idx is not None:
        if args.row_idx < 0 or args.row_idx >= len(df):
            raise ValueError(f"--row-idx {args.row_idx} out of range (0..{len(df)-1}).")
        df = df.iloc[[args.row_idx]].reset_index(drop=True)
    # else: run all rows in CSV

    for _, row in df.iterrows():
        process_row(row, args)


if __name__ == "__main__":
    main()
