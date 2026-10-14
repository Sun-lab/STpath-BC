"""Shared cohort validation, clinical inputs, and KM plots for BRCA notebooks."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from IPython.display import display

NA_VALUES = ["", "NA", "NaN", "[Not Available]", "[Not Applicable]", "[Unknown]"]
STAGES = ["I", "II", "III", "IV"]
SUBTYPES = ["LumA", "LumB", "Basal", "Her2", "Normal"]
GROUPS = ["Low", "Middle", "High"]


def read_table(path):
    return pd.read_csv(path, na_values=NA_VALUES)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_table(frame, name, *, output_dir):
    frame.to_csv(Path(output_dir) / name, index=False)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_columns(frame, columns, label):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns: {sorted(missing)}")


def patient_barcode(values):
    values = pd.Series(values, dtype="string").str.strip().str.upper()
    valid = values.str.match(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}($|[-.])", na=False)
    if not valid.all():
        raise ValueError(f"Invalid TCGA identifier: {values[~valid].head().tolist()}")
    return values.str[:12]


def number(values, label):
    result = pd.to_numeric(values, errors="coerce")
    bad = values.notna() & (result.isna() | ~np.isfinite(result))
    if bad.any():
        raise ValueError(f"Non-numeric/non-finite {label}: {values[bad].head().tolist()}")
    return result


def collapse_patients(frame):
    require_columns(frame, ["patient_id"], "clinical table")
    conflicts = frame.groupby("patient_id").nunique(dropna=True).gt(1).any(axis=1)
    if conflicts.any():
        raise ValueError(f"Conflicting clinical/subtype records for {conflicts[conflicts].index.tolist()}")
    return frame.groupby("patient_id", as_index=False, sort=True).first()


def stage_group(values):
    key = values.astype("string").str.strip().str.upper().str.replace(r"[ _-]", "", regex=True)
    key = key.str.replace(r"^STAGE", "", regex=True)
    mapping = {s + suffix: s for s in STAGES for suffix in ["", "A", "B", "C"]}
    mapping.pop("IC")
    return pd.Categorical(key.map(mapping), categories=STAGES)


def subtype_group(values):
    key = values.astype("string").str.strip().str.lower().str.replace(r"[ _.\-]", "", regex=True)
    mapping = dict(luma="LumA", luminala="LumA", lumb="LumB", luminalb="LumB", basal="Basal",
                   basallike="Basal", her2="Her2", her2enriched="Her2", normal="Normal", normallike="Normal")
    bad = key.notna() & ~key.isin(mapping)
    if bad.any():
        raise ValueError(f"Unrecognized PAM50 subtype: {values[bad].unique().tolist()}")
    return pd.Categorical(key.map(mapping), categories=SUBTYPES)


def inspect_cohort(root, targets, crop_um=None):
    """Validate each requested slide and return its status, including incomplete/invalid outputs."""
    root = Path(root)
    cfg_path = root / "cohort_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError("Each prediction run folder must contain cohort_config.json. Use a cohort run folder, not a pilot folder.")
    cfg = read_json(cfg_path)
    if crop_um is None:
        crop_um = float(cfg["crop_size_um"])
    if not np.isfinite(crop_um) or crop_um <= 0:
        raise ValueError("Invalid crop size in cohort configuration.")
    if not np.isclose(cfg["crop_size_um"], crop_um) or cfg["tile_selection"] != "all_eligible":
        raise ValueError("The cohort crop size or all-tile setting differs from the configuration.")
    ids = cfg["slides"]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Empty/duplicate cohort slide IDs.")
    patient_barcode(ids)
    if any(not re.fullmatch(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[0-9]{2}[A-Z]-[0-9]{2}-DX[0-9]+", s) for s in ids):
        raise ValueError("Unexpected diagnostic slide ID in cohort.")
    candidate = f"physical_um_{crop_um:g}"
    statuses, rows, signatures, files = [], [], [], [cfg_path]
    for slide_id in ids:
        folder = root / "slides" / slide_id
        needed = [folder / name for name in ("sampled_slide_summary.csv", "sampling_summary.csv",
                                             "prediction_summary.json", "tile_preparation.json")]
        missing = [p.name for p in needed if not p.is_file()]
        if missing:
            statuses.append(dict(slide_id=slide_id, status="incomplete", detail="; ".join(missing)))
            continue
        try:
            pred, prep = read_json(needed[2]), read_json(needed[3])
            if pred.get("status") != "complete" or prep.get("status") != "complete":
                raise ValueError("Incomplete completion marker")
            if "max_tiles_per_slide_size" not in prep or prep["max_tiles_per_slide_size"] is not None:
                raise ValueError("Subsampled tiles")
            if pred.get("tile_selection", "all_eligible") != "all_eligible":
                raise ValueError("Subsampled predictions")
            if prep["audit_csv_sha256"] != cfg["audit_sha256"] or not np.isclose(prep["min_tissue_fraction"], cfg["min_tissue"]):
                raise ValueError("Audit/tissue threshold mismatch")
            if any(len(v["sizes_um"]) != 1 or not np.isclose(v["sizes_um"][0], crop_um) for v in (pred, prep)):
                raise ValueError("Mixed crop sizes")
            if pred["manifest_sha256"] != prep["manifest_sha256"]:
                raise ValueError("Prediction/preparation manifest mismatch")
            if any(pred.get(k) is not False for k in ("calibration", "clipping", "across_cell_type_normalization")):
                raise ValueError("Expected raw, unclipped, unnormalized predictions")
            if set(pred["cell_types"]) != set(targets):
                raise ValueError("Different cell types; review CELL_TYPES explicitly")
            for file in (needed[0], needed[1], needed[3]):
                recorded = pred.get("output_sha256", {}).get(file.name)
                if recorded and sha256(file) != recorded:
                    raise ValueError(f"Changed merged output: {file.name}")
            tab, coverage = read_table(needed[0]), read_table(needed[1])
            require_columns(tab, ["slide_id", "patient_id", "candidate", "cell_type", "n_sampled_tiles",
                                  "sample_mean", "fraction_outside_0_1"], needed[0])
            require_columns(coverage, ["slide_id", "candidate", "n_sampled_tiles", "n_eligible_full_tiles_estimate"], needed[1])
            if (len(tab) != len(targets) or tab.cell_type.duplicated().any() or set(tab.cell_type) != set(targets)
                    or not tab.slide_id.eq(slide_id).all() or not tab.candidate.eq(candidate).all()
                    or not tab.patient_id.eq(slide_id[:12]).all()):
                raise ValueError("Slide/candidate/target identity mismatch")
            for col in ["sample_mean", "n_sampled_tiles", "fraction_outside_0_1"]:
                tab[col] = number(tab[col], col)
            if (tab[["sample_mean", "n_sampled_tiles", "fraction_outside_0_1"]].isna().any().any()
                    or not tab.n_sampled_tiles.ge(1).all() or not tab.n_sampled_tiles.mod(1).eq(0).all()
                    or tab.n_sampled_tiles.nunique() != 1 or not tab.fraction_outside_0_1.between(0, 1).all()):
                raise ValueError("Invalid means/tile counts/range fractions")
            n_tiles = tab.n_sampled_tiles.iloc[0]
            if (len(coverage) != 1 or coverage.slide_id.iloc[0] != slide_id or coverage.candidate.iloc[0] != candidate
                    or coverage.n_sampled_tiles.iloc[0] != n_tiles or coverage.n_eligible_full_tiles_estimate.iloc[0] != n_tiles
                    or pred["n_crops"] != n_tiles or pred["n_slides"] != 1):
                raise ValueError("Full-tile coverage/count mismatch")
            exports = pred["model_exports"]
            if len(exports) != len(targets) or {e["cell_type"] for e in exports} != set(targets):
                raise ValueError("Model export metadata missing")
            models = []
            for m in sorted(exports, key=lambda m: m["cell_type"]):
                if not m["file_sha256"].get("models/Combined.ubj"):
                    raise ValueError("Missing Combined model hash")
                models.append(dict(cell_type=m["cell_type"], hashes=m["file_sha256"]))
            encoders = pred["feature_provenance"]
            if set(encoders) != {"UNI2h", "Virchow", "Virchow2", "ProvGigapath", "Conch"}:
                raise ValueError("Missing foundation encoder provenance")
            fingerprint = {}
            for name, encoder in sorted(encoders.items()):
                if not encoder.get("recipe_code_sha256") or not encoder.get("resolved_hf_revision"):
                    raise ValueError("Missing encoder revision/recipe")
                fingerprint[name] = {k: encoder.get(k) for k in ("recipe", "recipe_code_sha256", "resolved_hf_revision")}
            signature = dict(models=models, encoders=fingerprint, recipe=prep.get("recipe"))
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            statuses.append(dict(slide_id=slide_id, status="invalid", detail=str(exc)))
        else:
            statuses.append(dict(slide_id=slide_id, status="complete", detail=""))
            rows.append(tab)
            signatures.append(signature)
            files.extend(needed)
    return dict(slides=pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
                status=pd.DataFrame(statuses), config=cfg, signatures=signatures, files=files)


def read_compatibility_report(path, expected_path, roots):
    """Bind the server audit to the two unchanged configs and cached outputs."""
    path = Path(path).expanduser().resolve()
    report = read_json(path)
    if report["old_cohort_config_sha256"] != sha256(expected_path):
        raise ValueError("Cached cohort configuration differs from the compatibility report.")
    references = [root for root in roots if (root / "cohort_config.json").is_file()
                  and sha256(root / "cohort_config.json") == report["current_cohort_config_sha256"]]
    if len(references) != 1:
        raise ValueError("Compatibility report must match exactly one current prediction run.")
    current = read_json(references[0] / "cohort_config.json")
    expected = read_json(expected_path)["slides"]
    remaining = set(expected) - set(current["slides"])
    records = {row["slide_id"]: row for row in report["slides"]}
    if (len(records) != len(report["slides"]) or set(records) != remaining
            or report["n_expected_slides"] != len(expected)
            or report["n_upgraded_slides"] != len(current["slides"])
            or report["current_audit_sha256"] != current["audit_sha256"]):
        raise ValueError("Compatibility report has inconsistent slide scope or resolution audit.")
    required = {"sampled_tiles.csv", "tile_predictions_raw.csv.gz", "sampled_slide_summary.csv",
                "sampling_summary.csv", "tile_preparation.json", "prediction_summary.json"}
    verified, rejected = set(), {}
    for sid, record in records.items():
        if record["status"] != "compatible":
            rejected[sid] = "; ".join(record.get("reasons", [])) or "Older result was not verified compatible"
            continue
        hashes = record["retained_sha256"]
        if record.get("reasons") or set(hashes) != required:
            raise ValueError(f"Incomplete compatibility evidence for {sid}")
        folder = expected_path.parent / "slides" / sid
        try:
            if any(sha256(folder / name) != digest for name, digest in hashes.items()):
                raise ValueError("Retained result changed since server audit")
        except (OSError, ValueError) as exc:
            rejected[sid] = str(exc)
        else:
            verified.add(sid)
    return dict(path=path, report=report, reference_root=str(references[0]),
                cached_root=str(expected_path.parent.resolve()), verified=verified, rejected=rejected)


def check_prediction_coverage(expected_root, run_dirs, targets, compatibility_report=None):
    """One row per original slide, using the first validated complete result across runs."""
    expected_path = Path(expected_root) / "cohort_config.json"
    expected = read_json(expected_path)["slides"]
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("The expected full cohort has empty/duplicate slide IDs.")
    patient_barcode(expected)
    roots = list(dict.fromkeys(Path(p).expanduser().resolve() for p in run_dirs))
    if not roots:
        raise ValueError("Set COVERAGE_RUN_DIRS to the prediction run folders to check.")
    compatibility = (read_compatibility_report(compatibility_report, expected_path, roots)
                     if compatibility_report is not None else None)
    records, inspected_runs = [], {}
    for priority, root in enumerate(roots):
        crop_um, inspected, run_error = np.nan, None, ""
        try:
            inspected = inspect_cohort(root, targets)
            crop_um = float(inspected["config"]["crop_size_um"])
            inspected_runs[str(root)] = inspected
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            run_error = str(exc)
        statuses = inspected["status"].set_index("slide_id") if inspected else None
        counts = (inspected["slides"].groupby("slide_id").n_sampled_tiles.first()
                  if inspected is not None and not inspected["slides"].empty else pd.Series(dtype=float))
        for slide_id in expected:
            if run_error:
                status, detail = "unavailable_run", run_error
            elif slide_id not in statuses.index:
                status, detail = "not_requested", "Slide is not in this run's cohort configuration"
            else:
                status, detail = statuses.loc[slide_id, ["status", "detail"]]
            if compatibility and str(root) == compatibility["cached_root"]:
                if slide_id not in compatibility["verified"]:
                    status = "excluded_source"
                    detail = compatibility["rejected"].get(slide_id,
                        "Use the current upgraded image; cached input is outside the compatibility audit")
            records.append(dict(slide_id=slide_id, run_priority=priority, source_run=str(root),
                crop_size_um=crop_um, status=status, detail=detail,
                n_tiles=counts.get(slide_id, np.nan),
                prediction_summary=str(root / "slides" / slide_id / "prediction_summary.json")))
    by_run = pd.DataFrame(records)
    rows = []
    for slide_id, attempts in by_run.groupby("slide_id", sort=False):
        complete = attempts.loc[attempts.status.eq("complete")].sort_values("run_priority")
        chosen = complete.iloc[0] if not complete.empty else None
        issues = attempts.loc[~attempts.status.isin(["complete", "not_requested"])]
        detail = "; ".join(f"{Path(r.source_run).name}: {r.status} ({r.detail})"
                           for r in issues.itertuples())
        rows.append(dict(slide_id=slide_id, patient_id=slide_id[:12],
            status="complete" if chosen is not None else "missing_prediction",
            source_run=chosen.source_run if chosen is not None else "",
            crop_size_um=chosen.crop_size_um if chosen is not None else np.nan,
            n_tiles=chosen.n_tiles if chosen is not None else np.nan,
            prediction_summary=chosen.prediction_summary if chosen is not None else "",
            n_complete_runs=len(complete),
            detail="" if chosen is not None else detail or "Slide was not requested in any checked run"))
    coverage = pd.DataFrame(rows)
    missing = coverage.loc[coverage.status.ne("complete")].copy()
    summary = dict(expected_cohort_config=str(expected_path.resolve()),
        expected_config_sha256=sha256(expected_path), checked_runs=list(map(str, roots)),
        checked_utc=datetime.now(timezone.utc).isoformat(),
        n_expected_slides=len(coverage), n_slides_with_predictions=len(coverage) - len(missing),
        n_slides_missing_predictions=len(missing), all_slides_have_predictions=missing.empty,
        validation="Completion/preparation markers, all target summary rows, full-tile counts, and per-slide provenance; no Slurm/log or full tile-table scan",
        selection_rule="First validated completed result in COVERAGE_RUN_DIRS order")
    if compatibility:
        summary["compatibility_report"] = str(compatibility["path"])
        summary["compatibility_report_sha256"] = sha256(compatibility["path"])
        summary["n_compatible_slides"] = len(compatibility["verified"])
    return dict(coverage=coverage, by_run=by_run, missing=missing, summary=summary,
                inspected_runs=inspected_runs, compatibility=compatibility)


def load_cohort(coverage_check, targets, allow_partial=False, *, output_dir):
    """Use the coverage snapshot: one validated result per slide, in run preference order."""
    status = coverage_check["coverage"].copy()
    write_table(status, "slide_completion.csv", output_dir=output_dir)
    missing_ids = set(status.loc[status.status.ne("complete"), "slide_id"])
    attempts = coverage_check["by_run"]
    if (attempts.slide_id.isin(missing_ids) & attempts.status.eq("invalid")).any():
        raise ValueError("Invalid slide outputs with no valid fallback; see prediction_coverage_by_run.csv.")
    if missing_ids and not allow_partial:
        raise ValueError("Some slides have incomplete predictions across all runs; see slide_completion.csv. Enable ALLOW_PARTIAL_COHORT for an exploratory run.")
    selected = status.loc[status.status.eq("complete")]
    if selected.empty:
        raise ValueError("No completed slides across the configured runs.")
    rows, signatures, files, configs = [], [], [], []
    expected_path = Path(coverage_check["summary"]["expected_cohort_config"])
    files.append(expected_path)
    compatibility = coverage_check.get("compatibility")
    if compatibility:
        files.append(compatibility["path"])
    for root, choice in selected.groupby("source_run", sort=False):
        inspected = coverage_check["inspected_runs"][root]
        config = inspected["config"]
        settings = {k: config[k] for k in ("min_tissue", "tile_selection")}
        if configs and settings != {k: configs[0][k] for k in settings}:
            raise ValueError("Selected runs used different tissue-selection settings.")
        if compatibility:
            report = compatibility["report"]
            if root == compatibility["cached_root"]:
                if not set(choice.slide_id) <= compatibility["verified"]:
                    raise ValueError("Selected older slides lack verified compatibility evidence.")
                if sha256(Path(root) / "cohort_config.json") != report["old_cohort_config_sha256"]:
                    raise ValueError("Cached cohort configuration changed after compatibility validation.")
            elif config["audit_sha256"] != report["current_audit_sha256"]:
                raise ValueError("Current predictions used a different resolution audit.")
        elif configs and config["audit_sha256"] != configs[0]["audit_sha256"]:
            raise ValueError("Selected runs used different resolution audits; provide a verified compatibility report.")
        configs.append(config)
        chosen_ids = set(choice.slide_id)
        tab = inspected["slides"].loc[inspected["slides"].slide_id.isin(chosen_ids)].copy()
        tab = tab.merge(choice[["slide_id", "source_run", "crop_size_um"]], on="slide_id", validate="many_to_one")
        rows.append(tab)
        complete_ids = inspected["status"].loc[inspected["status"].status.eq("complete"), "slide_id"]
        signatures.extend(signature for slide_id, signature in zip(complete_ids, inspected["signatures"])
                          if slide_id in chosen_ids)
        files.extend(p for p in inspected["files"] if p.name == "cohort_config.json" or p.parent.name in chosen_ids)
    slides = pd.concat(rows, ignore_index=True)
    if (slides.duplicated(["slide_id", "cell_type"]).any()
            or len(slides) != len(selected) * len(targets) or len(signatures) != len(selected)):
        raise ValueError("Duplicate or incomplete selected slide predictions.")
    if any(signature != signatures[0] for signature in signatures):
        raise ValueError("Selected slides used different model exports or encoder recipes/revisions; analyze runs separately.")
    files = list(dict.fromkeys(files))
    write_table(pd.DataFrame([dict(path=str(p), sha256=sha256(p)) for p in files]),
                "prediction_input_hashes.csv", output_dir=output_dir)
    return dict(slides=slides, status=status, configs=configs, signatures=signatures, files=files)


def aggregate_slides(slides, method="equal_slide"):
    if method not in {"equal_slide", "tile_weighted"}:
        raise ValueError("Unknown slide aggregation method")
    if slides.duplicated(["slide_id", "cell_type"]).any():
        raise ValueError("Each slide must contribute exactly one prediction per cell type.")
    if method == "tile_weighted" and "crop_size_um" in slides and slides.crop_size_um.nunique() > 1:
        raise ValueError("Tile weighting requires one crop size; use equal_slide for mixed crop sizes.")
    rows = []
    for (patient, cell_type), group in slides.groupby(["patient_id", "cell_type"], sort=True):
        weights = group.n_sampled_tiles if method == "tile_weighted" else None
        rows.append(dict(patient_id=patient, cell_type=cell_type, score=np.average(group.sample_mean, weights=weights),
                         n_slides=len(group), n_tiles=int(group.n_sampled_tiles.sum()), slide_ids=";".join(sorted(group.slide_id))))
        if "crop_size_um" in group:
            rows[-1].update(crop_sizes_um=";".join(f"{v:g}" for v in sorted(group.crop_size_um.unique())),
                            mixed_crop_sizes=bool(group.crop_size_um.nunique() > 1),
                            source_runs=";".join(sorted(group.source_run.unique())))
    return pd.DataFrame(rows)


def load_clinical(path, columns, subtype_path=None, subtype_columns=None):
    raw = read_table(path)
    required = ["id", "age", "stage", "PFI_time", "PFI_event", "OS_time", "OS_event"]
    require_columns(raw, [columns[k] for k in required], path)
    frame = pd.DataFrame({key: raw[columns[key]] for key in required if key != "id"})
    frame.insert(0, "patient_id", patient_barcode(raw[columns["id"]]))
    for key in ["age", "PFI_time", "PFI_event", "OS_time", "OS_event"]:
        frame[key] = number(frame[key], key)
    for key in ["PFI_event", "OS_event"]:
        if not frame[key].dropna().isin([0, 1]).all():
            raise ValueError(f"{key} must be 0=censored, 1=event.")
    if not frame.age.dropna().between(0, 120, inclusive="right").all():
        raise ValueError("Age must be in years; review age mapping/units.")
    if columns.get("subtype") is not None:
        require_columns(raw, [columns["subtype"]], path)
        frame["subtype"] = subtype_group(raw[columns["subtype"]])
    frame = collapse_patients(frame)
    frame["stage_raw"] = frame.stage
    frame["stage"] = stage_group(frame.stage)
    if subtype_path is not None:
        if columns.get("subtype") is not None:
            raise ValueError("Use one explicit subtype source: clinical column OR SUBTYPE_CSV.")
        source = read_table(subtype_path)
        require_columns(source, list(subtype_columns.values()), subtype_path)
        subtypes = pd.DataFrame(dict(patient_id=patient_barcode(source[subtype_columns["id"]]),
                                     subtype=subtype_group(source[subtype_columns["subtype"]])))
        frame = frame.merge(collapse_patients(subtypes), on="patient_id", how="left", validate="one_to_one")
    if "subtype" not in frame:
        frame["subtype"] = np.nan
    frame["subtype"] = subtype_group(frame.subtype)
    return frame


def endpoint_membership(data, endpoint):
    time, event = data[f"{endpoint}_time"], data[f"{endpoint}_event"]
    reason = np.select([~data.clinical_matched, time.isna() | event.isna(), time.le(0)],
                       ["no_clinical_match", "missing_endpoint", "nonpositive_followup"], default="included")
    adjusted = np.where((reason == "included") & data[["age", "stage", "subtype"]].isna().any(axis=1),
                        "missing_adjustment_covariate", reason)
    return pd.DataFrame(dict(patient_id=data.patient_id, endpoint=endpoint,
                             unadjusted_status=reason, adjusted_status=adjusted), index=data.index)


def tertiles(values):
    values = pd.Series(values)
    cuts = values.quantile([1/3, 2/3], interpolation="linear").to_numpy()
    if not np.isfinite(cuts).all() or cuts[0] == cuts[1]:
        return dict(cuts=cuts, group=pd.Series(pd.Categorical([None]*len(values), categories=GROUPS), index=values.index),
                    status="tied_cutpoints")
    group = pd.cut(values, [-np.inf, *cuts, np.inf], labels=GROUPS, right=True)
    return dict(cuts=cuts, group=group, status="empty_tertile" if group.value_counts().eq(0).any() else "ok")


def save_figure(fig, stem, show=True, *, output_dir):
    for ext in ["png", "pdf"]:
        fig.savefig(Path(output_dir) / "figures" / f"{stem}.{ext}", bbox_inches="tight")
    if show:
        from IPython.display import Image
        display(Image(filename=str(Path(output_dir) / "figures" / f"{stem}.png")))
    plt.close(fig)


def make_km(data, feature, endpoint, groups):
    membership = endpoint_membership(data, endpoint)
    keep = membership.unadjusted_status.eq("included") & groups.notna()
    frame = data.loc[keep].copy()
    frame["group"] = groups.loc[keep]
    frame["time"], frame["event"] = frame[f"{endpoint}_time"], frame[f"{endpoint}_event"]
    if len(frame) < 3 or frame.group.nunique() < 2 or frame.event.sum() == 0:
        return None
    test = multivariate_logrank_test(frame.time, frame.group, frame.event)
    ticks = MaxNLocator(nbins=5).tick_values(0, frame.time.max())
    ticks = ticks[(ticks >= 0) & (ticks <= frame.time.max())]
    curves, risk = {}, []
    for group in GROUPS:
        subset = frame.loc[frame.group.eq(group)]
        if subset.empty:
            continue
        curves[group] = KaplanMeierFitter().fit(subset.time, subset.event, label=f"{group} (n={len(subset)})")
        for time in ticks:
            risk.append(dict(endpoint=endpoint, cell_type=feature, group=group, time_days=time,
                             n_at_risk=int(subset.time.ge(time).sum())))
    return dict(curves=curves, data=frame, p=float(test.p_value), risk=pd.DataFrame(risk),
                ticks=ticks, cell_type=feature, endpoint=endpoint)


def draw_km(km, q=np.nan):
    colors = dict(Low="#0072B2", Middle="#E69F00", High="#009E73")
    fig, (ax, risk_ax) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                     gridspec_kw=dict(height_ratios=[3.5, 1.3]), layout="constrained")
    for group, curve in km["curves"].items():
        curve.plot_survival_function(ax=ax, ci_show=False, show_censors=True, color=colors[group],
                                     censor_styles=dict(marker="+", ms=5), linewidth=2)
    ylabel = "Progression-free interval probability" if km["endpoint"] == "PFI" else "Overall survival probability"
    ax.set(xlabel="", ylabel=ylabel, ylim=(0, 1.03), xlim=(0, km["data"].time.max()),
           title=f"{km['cell_type']} | {km['endpoint']}\nLog-rank p = {km['p']:.3g}; BH q = {q:.3g}")
    ax.legend(loc="lower left", frameon=False)
    levels = list(km["curves"])
    for i, group in enumerate(levels):
        subset = km["risk"].loc[km["risk"].group.eq(group)]
        for row in subset.itertuples():
            risk_ax.text(row.time_days, len(levels)-i, str(row.n_at_risk), ha="center", va="center", fontsize=10)
    risk_ax.set(yticks=range(1, len(levels)+1), yticklabels=levels[::-1], xticks=km["ticks"],
                ylim=(0.4, len(levels)+0.6), xlabel="Time (days)")
    risk_ax.set_title("Number at risk", loc="left", fontsize=10)
    risk_ax.spines[["left", "top", "right"]].set_visible(False)
    risk_ax.tick_params(axis="y", length=0, pad=16)
    for tick, group in zip(risk_ax.get_yticklabels(), levels[::-1]):
        tick.set_color(colors[group])
    return fig
