"""Assemble independent final label sources from saved 06e/06f outputs."""

from pathlib import Path

import pandas as pd

from .automatic_cluster_annotation import cell_label_masks
from .cell_clustering import safe_name
from .cell_identity_evidence import clean_author_refined_label, collapse_label
from .reclustered_cluster_annotation import propagate_recluster_labels, recluster_key
from .uncertain_reclustering import author_label_usable_mask


AUTHOR_DATASETS = {"Janesick_2023", "Janesick_2025"}
IDS = ["dataset", "sample", "cell_id"]
INITIAL_COLUMNS = ["leiden_r1_3", "leiden_r1_3_automatic_label", "leiden_r1_3_automatic_confidence"]
HIERARCHY_COLUMNS = IDS + INITIAL_COLUMNS + ["hierarchy_layer"]
RECLUSTER_COLUMNS = IDS + ["automatic_cluster_label", "automatic_label_confidence"]

# xgboost_LOIO_results.pptx, composition slides 8–12: eight source annotation classes.
PPTX_BROAD_COLORS = {
    'Tumor Cells': '#C64A50', 'Normal Epithelial Cells': '#4B8DBD',
    'T Cells': '#459B69', 'B Cells': '#D5AC30', 'pan-APCs': '#9368AC',
    'Stromal Cells': '#49A8A0', 'Smooth Muscle Cells': '#DB8848', 'Uncertain': '#B4B4B4',
}


def pptx_final_broad_label(broad):
    """Canonicalize the eight LOIO annotation classes without pooling source labels."""
    text = '' if pd.isna(broad) else str(broad).strip()
    mapping = {name.casefold(): name for name in PPTX_BROAD_COLORS}
    mapping.update({'nk cells': 'T Cells', '': 'Uncertain'})
    if text.casefold() not in mapping:
        raise ValueError(f'Unknown source broad label for PPTX grouping: {text!r}')
    return mapping[text.casefold()]


def read_cells(path, columns):
    """Keep identifiers as strings, including leading zeros and literal NA."""
    table = pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False)
    if table[IDS].eq("").any().any() or table.duplicated(IDS).any():
        raise ValueError(f"Empty or duplicated cell keys in {path}")
    return table


def audit_inputs(samples, hierarchy_dir, recluster_dir, default_resolution=1.3, resolutions=None):
    """Check every expected sample, including files absent from this machine."""
    resolutions = resolutions or {}
    if samples.empty or samples.duplicated(["dataset", "sample"]).any():
        raise ValueError("Sample registry must be nonempty with unique dataset/sample pairs")
    rows = []
    for row in samples[["dataset", "sample"]].itertuples(index=False):
        key = recluster_key(resolutions.get(row.dataset, default_resolution))
        hierarchy = Path(hierarchy_dir) / safe_name(row.dataset) / safe_name(row.sample) / "cell_label_hierarchy.csv.gz"
        recluster = Path(recluster_dir) / safe_name(row.dataset) / key / "cell_recluster_annotations.csv.gz"
        requirements = [(hierarchy, HIERARCHY_COLUMNS + (["author_label"] if row.dataset in AUTHOR_DATASETS else []))]
        if row.dataset not in AUTHOR_DATASETS:
            requirements.append((recluster, RECLUSTER_COLUMNS + [key]))
        problems = []
        for path, required in requirements:
            if not path.is_file():
                problems.append(f"Missing file: {path}")
                continue
            try:
                missing = sorted(set(required) - set(pd.read_csv(path, nrows=0).columns))
                if missing:
                    problems.append(f"{path}: missing columns {missing}")
            except Exception as exc:
                problems.append(f"Cannot read {path}: {exc}")
        rows.append(dict(dataset=row.dataset, sample=row.sample,
                         label_sources="author + initial_cluster" if row.dataset in AUTHOR_DATASETS else "initial_cluster + recluster",
                         cluster_key=key if row.dataset not in AUTHOR_DATASETS else "",
                         hierarchy_path=str(hierarchy),
                         recluster_path=str(recluster) if row.dataset not in AUTHOR_DATASETS else "",
                         status="blocked" if problems else "ready", details="; ".join(problems)))
    return pd.DataFrame(rows)


def broad_labels(labels):
    mapping = {label: collapse_label(label)[0] for label in labels.unique()}
    mapping["Uncertain"] = "Uncertain"
    mapping[""] = ""
    return labels.map(mapping)


def certain_mask(cluster, labels, confidence):
    present = ~cluster.str.strip().str.lower().isin({"", "nan", "none", "unassigned", "missing", "<na>"})
    return present & cell_label_masks(labels)[1] & confidence.str.strip().str.lower().isin({"moderate", "high"})


def assemble_sample(hierarchy, reclusters=None, decisions=None, cluster_key="recluster_r0_7"):
    """Return exactly four annotation columns plus keys, and a separate audit."""
    if hierarchy.empty or hierarchy[["dataset", "sample"]].drop_duplicates().shape[0] != 1:
        raise ValueError("Expected one nonempty sample hierarchy")
    missing = set(HIERARCHY_COLUMNS) - set(hierarchy)
    if missing:
        raise ValueError(f"Hierarchy is missing {sorted(missing)}")
    h = hierarchy.copy().fillna("").astype(str).reset_index(drop=True)
    if h.duplicated(IDS).any() or h[IDS].eq("").any().any():
        raise ValueError("Hierarchy contains empty or duplicate cell keys")
    dataset, sample = h.loc[0, ["dataset", "sample"]]
    out = h[IDS].copy()
    audit = h[IDS + ["hierarchy_layer"] + INITIAL_COLUMNS].copy()
    certain = certain_mask(h[INITIAL_COLUMNS[0]], h[INITIAL_COLUMNS[1]], h[INITIAL_COLUMNS[2]])
    out["initial_cluster_refined_label"] = h[INITIAL_COLUMNS[1]].where(certain, "Uncertain")
    out["initial_cluster_broad_label"] = broad_labels(out["initial_cluster_refined_label"])
    audit["initial_cluster_status"] = "not_certain"
    audit.loc[certain, "initial_cluster_status"] = "certain"

    if dataset in AUTHOR_DATASETS:
        author = h["author_label"].map(clean_author_refined_label)
        usable = author_label_usable_mask(author)
        out["author_refined_label"] = author.where(usable, "Uncertain")
        out["author_broad_label"] = broad_labels(out["author_refined_label"])
        audit["author_saved_label"] = h["author_label"]
        audit["author_status"] = usable.map({True: "provided", False: "missing_or_unusable"})
        return out[IDS + ["author_refined_label", "author_broad_label", "initial_cluster_refined_label", "initial_cluster_broad_label"]], audit

    if reclusters is None:
        raise ValueError("A saved 06f recluster result is required for this dataset")
    r = reclusters.copy().fillna("").astype(str)
    r = r.loc[r["dataset"].eq(dataset) & r["sample"].eq(sample), RECLUSTER_COLUMNS + [cluster_key]]
    if r.duplicated(IDS).any():
        raise ValueError("Duplicate recluster cell keys")
    if not set(r["cell_id"]).issubset(set(h["cell_id"])):
        raise ValueError("Recluster cells are absent from the saved hierarchy; check input versions")
    merged = h.merge(r, on=IDS, how="left", validate="one_to_one", indicator=True)
    matched = merged.pop("_merge").eq("both")
    merged = merged.fillna("")
    target = h["hierarchy_layer"].eq("reclustering_target")
    if (matched & ~target).any():
        raise ValueError("Recluster results include non-target cells; check hierarchy/input versions")
    reviewed = propagate_recluster_labels(merged, decisions if decisions is not None else pd.DataFrame(), cluster_key)
    out["recluster_refined_label"] = ""
    out.loc[target, "recluster_refined_label"] = "Uncertain"
    out["recluster_broad_label"] = broad_labels(out["recluster_refined_label"])
    for column in ["recluster_refined_label", "recluster_broad_label"]:
        out.loc[matched, column] = reviewed.loc[matched, column]
    audit["recluster_key"] = cluster_key
    audit["recluster_cluster"] = merged[cluster_key]
    audit["recluster_automatic_label"] = merged["automatic_cluster_label"]
    audit["recluster_automatic_confidence"] = merged["automatic_label_confidence"]
    audit["recluster_status"] = "not_targeted"
    audit.loc[target, "recluster_status"] = "target_without_saved_result"
    audit.loc[matched, "recluster_status"] = reviewed.loc[matched, "recluster_status"]
    audit["manual_decision_status"] = ""
    audit["manual_selected_broad_label"] = ""
    for column in ["manual_decision_status", "manual_selected_broad_label"]:
        audit.loc[matched, column] = reviewed.loc[matched, column]
    return out, audit


def label_counts(labels):
    """Long-form counts, retaining blank/not-targeted values in denominators."""
    parts = []
    for column in labels.columns.difference(IDS):
        counts = labels.groupby(["dataset", "sample", column], dropna=False).size().rename("n_cells").reset_index()
        counts = counts.rename(columns={column: "label"})
        counts["label_column"] = column
        counts["fraction_of_sample"] = counts["n_cells"] / len(labels)
        parts.append(counts)
    return pd.concat(parts, ignore_index=True)
