"""Local spatial QC, held-out grid-distance agreement, and Cox optimizer diagnosis.

Reads transferred raw outputs only. Original prediction and survival exports stay
unchanged. Measured Xenium fractions retain their original denominator.
"""
from itertools import permutations
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, false_discovery_control

from . import brca_spatial as spatial

FEATURES = list(spatial.CELL_TYPES.values())
SPATIAL_TYPES = list(spatial.THRESHOLDS)
SETS = {
    "tumor_20pct_immune_30pct_main": dict(tumor_cells=.20, stromal_cells=.30, t_cells=.30, pan_apc_b=.30),
    "tumor_15pct_immune_30pct_sensitivity": dict(tumor_cells=.15, stromal_cells=.30, t_cells=.30, pan_apc_b=.30),
}
TRUTH = dict(spatial.THRESHOLDS)


def distance_qc(coordinates, scores, cutoffs, queries=None):
    """All twelve directed pairs; preserve allowed within-tile matches."""
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.shape != (len(scores), 2) or not np.isfinite(coordinates).all():
        raise ValueError("Finite, aligned coordinates required")
    if scores[SPATIAL_TYPES].isna().any().any() or not np.isfinite(scores[SPATIAL_TYPES]).all().all():
        raise ValueError("Missing/nonfinite scores")
    if not scores[SPATIAL_TYPES].ge(0).all().all() or not scores[SPATIAL_TYPES].le(1).all().all():
        raise ValueError("Scores outside [0,1]")
    if set(cutoffs) != set(SPATIAL_TYPES) or any(not 0 <= t <= 1 for t in cutoffs.values()):
        raise ValueError("Expected four finite spatial cutoffs")
    queries = {} if queries is None else queries  # Reuse fixed tumor/stromal targets across sets.
    masks = {f: scores[f].to_numpy() > cutoffs[f] for f in SPATIAL_TYPES}
    rows = []
    for source, target in permutations(SPATIAL_TYPES, 2):
        a, b = masks[source], masks[target]
        row = dict(source=source, target=target, n_source=int(a.sum()), n_target=int(b.sum()),
                   n_overlap=int((a & b).sum()), overlap_fraction=np.nan, zero_fraction=np.nan,
                   mean=np.nan, median=np.nan, std=np.nan, iqr=np.nan, unweighted_mean=np.nan,
                   status="no_source_tiles" if not a.any() else "no_target_tiles")
        if a.any() and b.any():
            key = (target, cutoffs[target])
            if key not in queries:
                queries[key] = cKDTree(coordinates[b]).query(coordinates, workers=1)[0]
            nearest = queries[key][a]
            weighted = nearest * scores.loc[a, source].to_numpy()
            q25, q75 = np.quantile(weighted, [.25, .75])
            row.update(status="ok", overlap_fraction=float((a & b).sum()/a.sum()),
                       zero_fraction=float((nearest == 0).mean()), mean=float(weighted.mean()),
                       median=float(np.median(weighted)), std=float(weighted.std()), iqr=float(q75-q25),
                       unweighted_mean=float(nearest.mean()))
        rows.append(row)
    return pd.DataFrame(rows)


def local_slides(project, source):
    project = Path(project)
    slides = pd.read_csv(source)
    roots = {}
    for value in slides.source_run.unique():
        parts = Path(value).parts
        i = parts.index("tcga_brca_prediction")
        relative = Path(*parts[i:])
        if len(relative.parts) != 2 or '..' in relative.parts:
            raise ValueError("Unexpected prediction root")
        roots[value] = str(project / relative)
    slides["server_source_run"] = slides.source_run
    slides["source_run"] = slides.source_run.map(roots)
    if slides.duplicated(["slide_id", "cell_type"]).any():
        raise ValueError("Duplicate selected slide/type")
    return slides


def export_qc(result, output, inputs):
    result.to_csv(output/"slide_distance_qc.csv", index=False)
    pd.DataFrame(inputs).to_csv(output/"input_audit.csv", index=False)
    # Equal slide means per patient; any absent category keeps the summary missing.
    metrics = ["mean", "median", "std", "iqr", "zero_fraction", "overlap_fraction", "unweighted_mean"]
    keys = ["threshold_set", "source", "target", "patient_id"]
    group = result.groupby(keys, sort=False)
    patient = group[metrics].mean().where(group[metrics].count().eq(group.size(), axis=0)).reset_index()
    patient.to_csv(output/"patient_distance_qc.csv", index=False)
    summary = []
    for key, tab in patient.groupby(keys[:-1]):
        for metric in ("mean", "median"):
            values = tab[metric].dropna()
            summary.append(dict(zip(keys[:-1], key), metric=metric, n_patients=len(tab), n_valid=len(values),
                n_zero=int(values.eq(0).sum()), fraction_zero=float(values.eq(0).mean()),
                n_unique=values.nunique(), population_median=float(values.median()),
                median_source_overlap=float(tab.overlap_fraction.median())))
    summary = pd.DataFrame(summary)
    summary.to_csv(output/"distance_variation_summary.csv", index=False)
    return dict(slides=result, patients=patient, summary=summary)


def loio_validation(source, output, *, calibrated=False):
    """Same exported grid support, fixed measured enrichment, all held-out samples."""
    output=Path(output); output.mkdir(parents=True, exist_ok=False)
    table=pd.read_csv(source, usecols=["tile_id", "dataset", "sample", "individual_id", "cell_type",
                                     "observed_proportion", "predicted_xenium_oof"])
    if table.duplicated(["tile_id", "cell_type"]).any() or set(table.cell_type) != set(spatial.CELL_TYPES):
        raise ValueError("LOIO tile/type alignment differs")
    tile_meta=table.groupby("tile_id")[["dataset", "sample", "individual_id"]].nunique()
    if tile_meta.ne(1).any().any():raise ValueError("LOIO tile identities disagree")
    records=[]; heldout_maps={}; proportion_rows=[]
    for key, tab in table.groupby(["dataset", "sample", "individual_id"], sort=True):
        truth=tab.pivot(index="tile_id", columns="cell_type", values="observed_proportion").rename(columns=spatial.CELL_TYPES)[FEATURES]
        raw=tab.pivot(index="tile_id", columns="cell_type", values="predicted_xenium_oof").rename(columns=spatial.CELL_TYPES)[FEATURES].reindex(truth.index)
        if truth.isna().any().any() or not truth.ge(0).all().all() or not truth.le(1).all().all():
            raise ValueError("Invalid measured original fractions")
        if raw.isna().any().any() or not np.isfinite(raw).all().all():
            raise ValueError("Invalid LOIO predictions")
        xy=truth.index.to_series().str.extract(r"_row-(\d+)_col-(\d+)$")
        if xy.isna().any().any():raise ValueError("Grid coordinates unavailable in LOIO tile IDs")
        xy=xy.astype(int).to_numpy()[:, ::-1]
        if len(np.unique(xy, axis=0)) != len(xy):raise ValueError("Duplicate LOIO coordinates within sample")
        if calibrated:
            from . import brca_reference_calibration as calibration
            if key[2] not in heldout_maps:
                train = table.loc[table.individual_id.ne(key[2])]
                heldout_maps[key[2]] = {target: calibration.fit(
                    part.observed_proportion, part.predicted_xenium_oof, target)
                    for target, part in train.groupby("cell_type")}
            pred, keep, audit=spatial.prepare_scores(raw, dict(tables=heldout_maps[key[2]]))
            baseline, baseline_keep, _ = spatial.prepare_scores(raw)
            shared = truth.index[keep & baseline_keep]
            for feature in FEATURES:
                proportion_rows.append(dict(individual_id=key[2],sample=key[1],feature=feature,n_shared_tiles=len(shared),
                    raw_normalized_MAE=float((baseline.loc[shared,feature]-truth.loc[shared,feature]).abs().mean()),
                    calibrated_normalized_MAE=float((pred.loc[shared,feature]-truth.loc[shared,feature]).abs().mean())))
        else:
            pred, keep, audit=spatial.prepare_scores(raw)
        reference=distance_qc(xy[keep], truth.loc[keep], TRUTH).add_prefix("observed_")
        reference=reference.rename(columns={"observed_source":"source", "observed_target":"target"})
        queries={}
        for name, cutoffs in SETS.items():
            estimate=distance_qc(xy[keep], pred, cutoffs, queries)
            compared=estimate.merge(reference, on=["source", "target"], validate="one_to_one")
            compared["threshold_set"]=name
            compared["dataset"], compared["sample"], compared["individual_id"]=key
            compared["n_support_tiles"]=int(keep.sum()); compared["n_excluded_zero_total"]=int((~keep).sum())
            records.append(compared)
        print("LOIO distances:", key[2], key[1], f"{keep.sum():,} grid tiles", flush=True)
    result=pd.concat(records, ignore_index=True)
    result.to_csv(output/"sample_distance_agreement.csv", index=False)
    metrics=["mean", "median", "observed_mean", "observed_median"]
    keys=["threshold_set", "source", "target", "individual_id"]
    group=result.groupby(keys)
    individuals=group[metrics].mean().where(group[metrics].count().eq(group.size(),axis=0)).reset_index()
    individuals.to_csv(output/"individual_distance_agreement.csv", index=False)
    summary=[]
    for key, tab in individuals.groupby(keys[:-1]):
        for metric in ("mean", "median"):
            valid=tab[[metric,"observed_"+metric]].dropna()
            difference=valid[metric]-valid["observed_"+metric]
            rho=(spearmanr(valid[metric],valid["observed_"+metric]).statistic
                 if len(valid)>2 and valid.nunique().min()>1 else np.nan)
            summary.append(dict(zip(keys[:-1],key), metric=metric, n_individuals=len(tab), n_paired=len(valid),
                MAE_grid=float(difference.abs().mean()), bias_grid=float(difference.mean()), spearman=float(rho),
                predicted_zero_fraction=float(valid[metric].eq(0).mean()),
                observed_zero_fraction=float(valid["observed_"+metric].eq(0).mean())))
    summary=pd.DataFrame(summary);summary.to_csv(output/"distance_agreement_summary.csv", index=False)
    if calibrated:
        pd.DataFrame(proportion_rows).to_csv(output/"sample_proportion_agreement.csv",index=False)
    spatial.write_json(output/"validation_method.json",dict(validation_code_sha256=spatial.sha256(__file__), calibrated=calibrated,
        calibration_scope=("Map fitted on other individuals' reused base LOIO predictions; not nested model refitting" if calibrated else "none"), source_sha256=spatial.sha256(source),
        prediction_policy=("Individual-excluded quantile mapping then normalize five scores" if calibrated else "clip then normalize five scores"), observed_policy="original measured cell-count fractions; no normalization",
        observed_cutoffs=TRUTH, prediction_cutoffs=SETS, units="grid spacing; physical tile footprints unverified",
        population="same exported tile support within each held-out sample, equal sample means within individuals",
        missing="No source/target stays missing; complete sample coverage required per individual",
        interpretation="Descriptive validation, conditional on already reviewed LOIO predictions; not independent external validation"))
    return dict(samples=result,individuals=individuals,summary=summary)


def plot_results(tcga, loio, output):
    """Outcome-free QC plots; save standalone scientific figures."""
    import matplotlib.pyplot as plt
    output=Path(output);(output/"figures").mkdir(parents=True,exist_ok=True)
    paths=[]
    def save(fig,name):
        for suffix in ("png","pdf"):
            path=output/"figures"/(name+"."+suffix)
            fig.savefig(path,dpi=180,bbox_inches="tight")
            if suffix=="png":paths.append(path)
        plt.close(fig)
    for metric in ("mean","median"):
        table=tcga.loc[tcga.metric.eq(metric)].copy()
        table["pair"]=table.source+" → "+table.target
        matrix=table.pivot(index="pair",columns="threshold_set",values="fraction_zero").reindex(columns=list(SETS))
        fig,ax=plt.subplots(figsize=(8,7),layout="constrained")
        im=ax.imshow(matrix.to_numpy(),vmin=0,vmax=1,cmap="YlOrRd")
        ax.set(xticks=range(len(SETS)),xticklabels=["Tumor >20% main", "Tumor >15% sensitivity"],yticks=range(len(matrix)),yticklabels=matrix.index,
               xlabel="Tumor cutoff; stroma/T/pan-APC >30%",title=f"TCGA: patients with zero {metric} weighted distance")
        for i in range(len(matrix)):
            for j in range(len(SETS)):ax.text(j,i,f"{100*matrix.iloc[i,j]:.1f}%",ha="center",va="center",color="white" if matrix.iloc[i,j]>.6 else "black")
        fig.colorbar(im,ax=ax,label="Fraction among patients with valid distances")
        save(fig,"tcga_zero_"+metric)
    for metric in ("mean","median"):
        table=loio.loc[loio.metric.eq(metric)].copy();table["pair"]=table.source+" → "+table.target
        fig,axes=plt.subplots(1,2,figsize=(14,7),layout="constrained")
        for ax,value,title in zip(axes,["MAE_grid","spearman"],["Absolute error (grid units)","Between-individual Spearman correlation"]):
            matrix=table.pivot(index="pair",columns="threshold_set",values=value).reindex(columns=list(SETS))
            im=ax.imshow(matrix.to_numpy(),cmap="viridis" if value=="MAE_grid" else "RdBu_r",
                         **({} if value=="MAE_grid" else dict(vmin=-1,vmax=1)))
            ax.set(xticks=range(len(SETS)),xticklabels=["Tumor >20% main", "Tumor >15% sensitivity"],yticks=range(len(matrix)),yticklabels=matrix.index,title=title)
            for i in range(len(matrix)):
                for j in range(len(SETS)):
                    v=matrix.iloc[i,j]
                    ax.text(j,i,"NA" if pd.isna(v) else f"{v:.2f}",ha="center",va="center",color="black",
                            bbox=dict(facecolor="white",alpha=.65,edgecolor="none",pad=.4))
            fig.colorbar(im,ax=ax,shrink=.7)
        fig.suptitle(f"Xenium LOIO {metric} distance | fixed measured enrichment reference")
        save(fig,"loio_agreement_"+metric)
    return paths


def calibrated_cache_qc(slides, calibrated_results, cache_dir, bundle, output, *, spatial_dir=None):
    """Validate the selected main/sensitivity cache sets without parsing tile CSVs."""
    from . import brca_reference_calibration as calibration
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    spatial_dir = Path(spatial_dir) if spatial_dir is not None else Path(calibrated_results)/"calibrated_spatial_main"
    audit=pd.read_csv(spatial_dir/"spatial_input_audit.csv").set_index("slide_id")
    selected=slides[["slide_id","patient_id","source_run","crop_size_um"]].drop_duplicates()
    if selected.slide_id.duplicated().any() or set(selected.slide_id)!=set(audit.index):
        raise ValueError("Calibrated cache/slide scope mismatch")
    records=[];inputs=[]
    for r in selected.itertuples(index=False):
        folder=Path(r.source_run)/"slides"/r.slide_id
        source=folder/"tile_predictions_raw.csv.gz"
        digest=spatial.sha256(source)
        if digest!=audit.loc[r.slide_id,"prediction_csv_sha256"]:raise ValueError("Raw predictions changed after calibration")
        summary=json.loads((folder/"prediction_summary.json").read_text())
        calibration.verify_models(bundle,summary,project_root=Path(r.source_run).parents[1])
        for name,cutoffs in SETS.items():
            method=dict(recipe=spatial.RECIPE,code_sha256=spatial.sha256(spatial.__file__),
                score_policy="reference_quantile_normalize",thresholds=cutoffs,threshold_comparison="strict >",
                std_ddof=0,iqr_method="linear/type7",self_matches=True,missing_category="NA, no imputation",
                calibration_recipe=bundle["recipe"],calibration_sha256=calibration.fingerprint(bundle),
                calibration_code_sha256=spatial.sha256(calibration.__file__))
            identity=dict(method=method,prediction_csv_sha256=digest,
                prediction_summary_sha256=spatial.sha256(folder/"prediction_summary.json"),slide_id=r.slide_id,
                tile_preparation_sha256=spatial.sha256(folder/"tile_preparation.json"),
                crop_size_um=float(r.crop_size_um),source_run=str(r.source_run))
            key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            saved=json.loads((Path(cache_dir)/(key+".json")).read_text())
            if saved["identity"]!=identity:raise ValueError("Calibrated cache identity mismatch")
            for pair in saved["pair_qc"]:
                prefix=pair["source"]+"_to_"+pair["target"]
                row=dict(slide_id=r.slide_id,patient_id=r.patient_id,threshold_set=name,
                    source=pair["source"],target=pair["target"],n_source=pair["n_source_tiles"],n_target=pair["n_target_tiles"],
                    n_overlap=pair["n_overlap"],overlap_fraction=pair["overlap_fraction"],zero_fraction=pair["zero_fraction"],
                    status=pair["status"],unweighted_mean=np.nan)
                row.update({stat:saved["summaries"]["um"][prefix+"_"+stat] for stat in ("mean","median","std","iqr")})
                records.append(row)
        inputs.append(dict(slide_id=r.slide_id,path=str(source),sha256=digest,**saved["qc"]))
    result=pd.DataFrame(records)
    print("Validated calibrated QC caches for",len(selected),"slides",flush=True)
    return export_qc(result,output,inputs)
