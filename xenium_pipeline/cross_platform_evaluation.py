"""Compare frozen Visium BRCA regressors with validated Xenium LOIO predictions."""
from __future__ import annotations
from .storage import model_runs
import hashlib
import json
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from . import stpath_xgboost as sx
from .feature_importance_comparison import TARGETS, VISIUM_DOWNLOAD_URL, load_visium
RECIPE = 'stpath_brca_training_compatible_v1'
VISIUM_COMMIT = 'fcf867babfdf9ea768b9c6ddce9daef7ef87f372'
VISIUM_BLOBS = {'Normal_Epithelial': '3aef34ef5ed2f167a7a1cd0d5cf82f8000c33707', 'Stromal': '7ee277310ec7cb78505db0b1be2c57d2ccd07432', 'T_cell': 'ab3ccba691920ac35a8b808917be5ae6a3875ae0', 'Tumor': '467ad437b3f35b1cee450543b3ef322a98d29911', 'pan_APC': '62201ea51cfb6acfa8934005d0f3373fde71a1a6'}
from .feature_importance_comparison import TARGET_KEYS
from .storage import load_layout, curated_table
TARGET_FILES = {alias: f"tile_targets_{load_layout()['targets'][key]['stem']}.csv.gz" for alias, key in TARGET_KEYS.items()}
KEYS = ['tile_id', 'dataset', 'sample', 'individual_id']
PREDICTIONS = {'Visium frozen': 'predicted_visium', 'Xenium LOIO': 'predicted_xenium_oof', 'Training mean': 'predicted_training_mean'}

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def model_filename(target):
    return f'xgboost_model_{target}_Combined_external_prediction.model'

def verify_visium_model(path, target):
    path = Path(path)
    digest = hashlib.sha1(f'blob {path.stat().st_size}\x00'.encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != VISIUM_BLOBS[target]:
        raise ValueError(f'Model differs from the pinned public Visium release: {path}')

def audit_inputs(root, feature_root, model_dir, workbook):
    """A lightweight inventory; the evaluation performs the content checks."""
    root, feature_root, model_dir = map(Path, (root, feature_root, model_dir))
    rows = []
    for target, (label, relative) in TARGETS.items():
        directory = root / 'xgboost_training/runs' / relative
        paths = {'targets': root / 'xgboost_training' / TARGET_FILES[target], 'run_metadata': directory.parent / 'run_metadata.json', 'checkpoints': directory / 'checkpoints', 'visium_model': model_dir / model_filename(target)}
        rows.append({'visium_target': target, 'cell_type': label, **{f'{key}_available': path.is_dir() if key == 'checkpoints' else path.is_file() for key, path in paths.items()}, 'feature_root_available': feature_root.is_dir(), 'workbook_available': Path(workbook).is_file(), **{f'{key}_path': str(path) for key, path in paths.items()}})
    return pd.DataFrame(rows)

def load_xenium_oof(root):
    """Recover predictions from committed outer folds, never final-fit models."""
    root = Path(root)
    mapping = pd.read_csv(curated_table(root, 'sample_individual_mapping.csv'))
    targets, predictions, statuses, sources = ({}, [], [], [])
    canonical = None
    for target, (label, relative) in TARGETS.items():
        directory = root / 'xgboost_training/runs' / relative
        metadata_path = directory.parent / 'run_metadata.json'
        metadata = json.loads(metadata_path.read_text())
        config = sx.TrainingConfig(**metadata['training_config'])
        if config.num_boost_round != 800:
            raise ValueError(f'Expected the production 800-round run: {directory}')
        target_path = root / 'xgboost_training' / TARGET_FILES[target]
        table, parse_mode = sx.load_checkpoint_compatible_targets(target_path, label, directory / 'checkpoints', config)
        if table[KEYS].isna().any().any() or table.tile_id.duplicated().any():
            raise ValueError(f'Missing identities or duplicate tiles: {target_path}')
        table[KEYS] = table[KEYS].astype(str)
        checksum = sx.tile_id_checksum(table.tile_id)
        if metadata.get('tile_id_checksum') != checksum or metadata.get('training_config_fingerprint') != sx.configuration_fingerprint(config) or metadata.get('n_tiles') != len(table):
            raise ValueError(f'Target table/configuration differs from the saved run: {directory}')
        mapped = sx.validate_individual_mapping(table.drop(columns='individual_id'), mapping)
        if not np.array_equal(mapped.individual_id.astype(str), table.individual_id):
            raise ValueError('Current individual mapping differs from the training split. Refit LOIO after correcting groups.')
        counts = table[f'count__{label}'].to_numpy(dtype=float)
        totals = table.n_cells_total.to_numpy(dtype=float)
        if not np.isfinite(totals).all() or (totals <= 0).any() or (not np.isfinite(counts).all()) or (counts < 0).any() or (counts > totals).any() or (not np.allclose(counts / totals, table[f'proportion__{label}'], rtol=0, atol=1e-12)):
            raise ValueError(f'Invalid all-cell denominator/reference counts: {target_path}')
        identities = table[KEYS + ['n_cells_total']].set_index('tile_id').sort_index()
        if canonical is not None and (not identities.equals(canonical)):
            raise ValueError('Targets disagree on tile coverage, sample/group identity, or total-cell denominator.')
        canonical = identities
        _, oof, status = sx.collect_completed_fold_metrics(table, label, directory / 'checkpoints', config)
        status.insert(0, 'visium_target', target)
        status.insert(1, 'cell_type', label)
        statuses.append(status)
        predictions.append(oof[KEYS + ['cell_type', 'outer_fold', 'observed_proportion', 'predicted_raw__Combined', 'predicted_raw__TrainingMeanBaseline']].rename(columns={'predicted_raw__Combined': 'predicted_xenium_oof', 'predicted_raw__TrainingMeanBaseline': 'predicted_training_mean'}).assign(visium_target=target))
        targets[target] = table
        sources.append(dict(visium_target=target, path=str(target_path), sha256=sha256(target_path), parse_mode=parse_mode, run_metadata=metadata, training_data_fingerprint=sx.training_data_fingerprint(table, label, config)))
    return (targets, pd.concat(predictions, ignore_index=True), pd.concat(statuses, ignore_index=True), sources)

def load_visium_models(model_dir, workbook, *, download_missing=True, nthread=4):
    """Use the pinned published workbook order and five frozen Combined boosters."""
    import xgboost as xgb
    model_dir, workbook = (Path(model_dir), Path(workbook))
    if not workbook.is_file() and (not download_missing):
        raise FileNotFoundError(f'Missing published feature workbook: {workbook}')
    rankings = load_visium(workbook)
    models, selections, sources = ({}, {}, [])
    for target in TARGETS:
        path = model_dir / model_filename(target)
        url = VISIUM_DOWNLOAD_URL.rsplit('/', 1)[0] + '/' + path.name
        if not path.is_file():
            if not download_missing:
                raise FileNotFoundError(f'Missing frozen Visium model: {path}')
            path.parent.mkdir(parents=True, exist_ok=True)
            sx.log(f'Downloading frozen Visium model: {target}')
            with urllib.request.urlopen(url, timeout=60) as response:
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                    pending = Path(stream.name)
                    try:
                        for chunk in iter(lambda: response.read(1024 * 1024), b''):
                            stream.write(chunk)
                        stream.close()
                        verify_visium_model(pending, target)
                        probe = xgb.Booster()
                        probe.load_model(pending)
                        pending.replace(path)
                    finally:
                        pending.unlink(missing_ok=True)
        verify_visium_model(path, target)
        booster = xgb.Booster(params={'nthread': nthread})
        booster.load_model(path)
        booster.set_param({'nthread': nthread})
        selected = {name: rankings.loc[(rankings.visium_target == target) & (rankings.encoder == name)].sort_values('published_rank').feature_index.to_numpy(dtype=int) for name in sx.MODEL_NAMES}
        if booster.num_features() != sum(map(len, selected.values())):
            raise ValueError(f'Published workbook/model width mismatch: {path}')
        models[target], selections[target] = (booster, selected)
        sources.append(dict(visium_target=target, path=str(path), sha256=sha256(path), source_url=url, git_blob=VISIUM_BLOBS[target], n_features=booster.num_features(), selected_indices={key: value.tolist() for key, value in selected.items()}))
    return (models, selections, sources)

def predict_visium(tile_table, feature_root, models, selections, *, recipe=RECIPE):
    """Load one sample at a time; preserve tile identities and workbook column order."""
    import xgboost as xgb
    predictions, feature_sources = ([], [])
    for (dataset, sample), table in tile_table.groupby(['dataset', 'sample'], sort=False):
        sx.log(f'Visium prediction: {dataset}/{sample} ({len(table):,} tiles)')
        matrices = sx.load_feature_matrices(table, feature_root, expected_recipe_version=recipe)
        result = table[KEYS].copy()
        for target, booster in models.items():
            selected = np.concatenate([matrices[name][:, selections[target][name]] for name in sx.MODEL_NAMES], axis=1)
            values = booster.predict(xgb.DMatrix(selected))
            if values.shape != (len(table),) or not np.isfinite(values).all():
                raise ValueError(f'Invalid Visium predictions: {target}/{sample}')
            result[target] = values
        predictions.append(result)
        for name in sx.MODEL_NAMES:
            path = sx.feature_shard_path(feature_root, name, dataset, sample)
            feature_sources.append(dict(dataset=dataset, sample=sample, encoder=name, path=str(path), bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns, recipe=recipe, n_tiles=len(table), tile_id_checksum=sx.tile_id_checksum(table.tile_id)))
        del matrices, selected
    return (pd.concat(predictions, ignore_index=True), pd.DataFrame(feature_sources))

def pair_predictions(oof, visium):
    """Require one prediction from each model for every retained OOF row."""
    if oof.duplicated(['visium_target', 'tile_id']).any() or visium.tile_id.duplicated().any():
        raise ValueError('Duplicate prediction identities')
    rows = []
    for target, table in oof.groupby('visium_target', sort=False):
        paired = table.merge(visium[KEYS + [target]].rename(columns={target: 'predicted_visium'}), on=KEYS, how='left', validate='one_to_one')
        columns = ['observed_proportion', *PREDICTIONS.values()]
        if not np.isfinite(paired[columns].to_numpy(dtype=float)).all():
            raise ValueError(f'Missing or nonfinite matched predictions for {target}')
        if not paired.observed_proportion.between(0, 1).all():
            raise ValueError('Observed proportions must be in [0, 1]')
        rows.append(paired)
    if not rows:
        raise ValueError('No complete Xenium outer folds are available')
    return pd.concat(rows, ignore_index=True)

def calculate_metrics(paired, *, bootstrap_replicates=2000, seed=42):
    """Equal-weight group summaries and descriptive paired group-bootstrap intervals."""
    rows, calibration = ([], [])
    for (target, label, individual), table in paired.groupby(['visium_target', 'cell_type', 'individual_id'], sort=False):
        truth = table.observed_proportion.to_numpy(dtype=float)
        for model, column in PREDICTIONS.items():
            prediction = table[column].to_numpy(dtype=float)
            metrics = sx.regression_metrics(truth, prediction)
            slope = float(np.mean((prediction - prediction.mean()) * (truth - truth.mean())) / np.var(prediction)) if np.ptp(prediction) > 0 else np.nan
            rows.append(dict(visium_target=target, cell_type=label, individual_id=individual, dataset=';'.join(sorted(table.dataset.unique())), model=model, **metrics, mae_pp=100 * metrics['mae'], rmse_pp=100 * metrics['rmse'], bias_pp=100 * float(np.mean(prediction - truth)), calibration_slope=slope, calibration_intercept=float(truth.mean() - slope * prediction.mean()), n_positive_tiles=int((truth > 0).sum()), outside_unit_interval_fraction=float(((prediction < 0) | (prediction > 1)).mean())))
    individual_metrics = pd.DataFrame(rows)
    summaries = []
    for (label, model), table in individual_metrics.groupby(['cell_type', 'model'], sort=False):
        record = dict(cell_type=label, model=model, n_individuals=len(table), n_tiles=int(table.n.sum()))
        for metric in ['mae_pp', 'rmse_pp', 'bias_pp', 'pearson', 'spearman']:
            record[f'mean_{metric}'] = table[metric].mean()
            record[f'sd_{metric}'] = table[metric].std()
            record[f'n_defined_{metric}'] = int(table[metric].notna().sum())
        summaries.append(record)
    rng, differences = (np.random.default_rng(seed), [])
    if bootstrap_replicates < 1:
        raise ValueError('bootstrap_replicates must be positive')
    for label, table in individual_metrics.groupby('cell_type', sort=False):
        for metric in ['mae_pp', 'rmse_pp', 'pearson', 'spearman']:
            wide = table.pivot(index='individual_id', columns='model', values=metric)
            delta = (wide['Visium frozen'] - wide['Xenium LOIO']).dropna().to_numpy()
            bounds = [np.nan, np.nan]
            if len(delta) > 1:
                means = rng.choice(delta, size=(bootstrap_replicates, len(delta)), replace=True).mean(axis=1)
                bounds = np.quantile(means, [0.025, 0.975])
            differences.append(dict(cell_type=label, metric=metric, n_paired_individuals=len(delta), mean_visium_minus_xenium=float(delta.mean()) if len(delta) else np.nan, descriptive_lower_95=bounds[0], descriptive_upper_95=bounds[1]))
    for label, table in paired.groupby('cell_type', sort=False):
        for model, column in PREDICTIONS.items():
            bins = pd.cut(table[column], [-np.inf, *np.linspace(0, 1, 11), np.inf], include_lowest=True)
            grouped = table.groupby(bins, observed=True)
            for interval, chunk in grouped:
                calibration.append(dict(cell_type=label, model=model, prediction_bin=str(interval), n_tiles=len(chunk), mean_predicted=chunk[column].mean(), mean_observed=chunk.observed_proportion.mean()))
    return {'individual_metrics': individual_metrics, 'summary_metrics': pd.DataFrame(summaries), 'paired_differences': pd.DataFrame(differences), 'calibration_bins': pd.DataFrame(calibration)}

def run_evaluation(root, feature_root, model_dir, workbook, output_dir, *, require_complete_folds=True, download_missing=True, nthread=4, exclude_individuals=(), bootstrap_replicates=2000, seed=42):
    """Existing-cohort evaluation only. Training checkpoints remain read-only."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = output_dir / 'provenance.json'
    provenance = dict(status='running', started_at_utc=datetime.now(timezone.utc).isoformat(), evaluation='existing Xenium cohort; frozen Visium versus Xenium LOIO', visium_commit=VISIUM_COMMIT, feature_recipe=RECIPE, require_complete_folds=require_complete_folds, prediction_scale='raw', exclude_individuals=sorted(set(exclude_individuals)), bootstrap_replicates=bootstrap_replicates, seed=seed, notes=['Group identities and overlap with Visium training donors require source verification.', 'References use Xenium tile cell counts including unresolved cells in the denominator.', 'Class mapping is broad; absolute errors include annotation/denominator differences.', 'Current Xenium embeddings are reused. Physical target-footprint equivalence is unverified.', 'Project ProvGigapath uses Resize(256, bicubic)/CenterCrop(224); the released predictor uses Resize(224).', 'Exact original Visium encoder revisions are unverified; this is transfer on project embeddings.', 'No quantile fitting, cross-class normalization, or clipping is applied.', 'Bootstrap intervals describe between-group variation conditional on saved predictions; LOIO fits share training data.'])
    sx._atomic_json(provenance, provenance_path)
    try:
        targets, oof, status, target_sources = load_xenium_oof(root)
        exclusions = set(map(str, exclude_individuals))
        canonical = next(iter(targets.values()))
        unknown = exclusions.difference(canonical.individual_id)
        if unknown:
            raise ValueError(f'Unknown excluded individuals: {sorted(unknown)}')
        status['excluded_from_evaluation'] = status.held_out_individual.isin(exclusions)
        sx._atomic_csv(status, output_dir / 'fold_coverage.csv')
        active_status = status.loc[~status.excluded_from_evaluation]
        if active_status.empty:
            raise ValueError('No evaluation individuals remain')
        if active_status.status.eq('invalid').any():
            raise ValueError('Invalid Xenium checkpoints; inspect fold_coverage.csv')
        incomplete = active_status.loc[active_status.status.ne('complete')]
        if require_complete_folds and (not incomplete.empty):
            raise ValueError('Incomplete/skipped Xenium folds; inspect fold_coverage.csv and finish those outer folds. Set REQUIRE_COMPLETE_FOLDS=False only for explicitly partial results.')
        oof = oof.loc[~oof.individual_id.isin(exclusions)].copy()
        if oof.empty:
            raise ValueError('No complete Xenium outer folds are available')
        models, selections, model_sources = load_visium_models(model_dir, workbook, download_missing=download_missing, nthread=nthread)
        visium, features = predict_visium(canonical, feature_root, models, selections)
        paired = pair_predictions(oof, visium)
        tables = calculate_metrics(paired, bootstrap_replicates=bootstrap_replicates, seed=seed)
        tables.update(paired_predictions=paired, fold_coverage=status, feature_inputs=features)
        coverage = paired.groupby(['cell_type', 'individual_id'], sort=False).size().rename('n_tiles').reset_index()
        tables['paired_coverage'] = coverage
        for name, table in tables.items():
            suffix = '.csv.gz' if name == 'paired_predictions' else '.csv'
            sx._atomic_csv(table, output_dir / (name + suffix))
        provenance.update(status='complete' if incomplete.empty else 'partial', completed_at_utc=datetime.now(timezone.utc).isoformat(), target_sources=target_sources, visium_models=model_sources, workbook_sha256=sha256(workbook), mapping_sha256=sha256(curated_table(root, 'sample_individual_mapping.csv')), n_missing_folds=len(incomplete), n_paired_rows=len(paired))
        sx._atomic_json(provenance, provenance_path)
        return tables
    except Exception as exc:
        provenance.update(status='failed', error=str(exc))
        sx._atomic_json(provenance, provenance_path)
        raise

def plot_tile_scatter(paired, output_dir):
    """Plot both model estimates against the same observed value for every tile."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    columns = ['observed_proportion', 'predicted_visium', 'predicted_xenium_oof']
    required = ['cell_type', 'tile_id', *columns]
    missing = set(required).difference(paired.columns)
    if missing:
        raise ValueError(f'Missing scatterplot columns: {sorted(missing)}')
    if paired.empty or paired[required].isna().any().any():
        raise ValueError('Scatterplots require nonempty matched tiles with complete values.')
    if paired.duplicated(['cell_type', 'tile_id']).any():
        raise ValueError('Duplicate target/tile rows in scatterplot input.')
    values = paired[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not paired.observed_proportion.between(0, 1).all():
        raise ValueError('Scatterplots require finite estimates and observed proportions in [0, 1].')
    labels = list(paired.cell_type.unique())
    models = [('predicted_visium', 'Visium pretrained model', '#D55E00'), ('predicted_xenium_oof', 'Xenium LOIO model', '#0072B2')]
    limits = (min(-0.05, float(values.min()) - 0.025), max(1.05, float(values.max()) + 0.025))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(labels), figsize=(4 * len(labels), 9.5), sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.13, top=0.8, wspace=0.12, hspace=0.37)
    for col, label in enumerate(labels):
        group = paired.loc[paired.cell_type.eq(label)]
        observed = group.observed_proportion.to_numpy(dtype=float)
        for row, (column, model, color) in enumerate(models):
            ax = axes[row, col]
            ax.scatter(group[column].to_numpy(dtype=float), observed, s=1, alpha=0.06, color=color, linewidths=0, rasterized=True)
            ax.plot([0, 1], [0, 1], '--', color='0.2', lw=1.1, zorder=3)
            ax.set(xlim=limits, ylim=limits, aspect='equal')
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.tick_params(labelsize=10)
            ax.grid(alpha=0.15)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
            if row == 0:
                ax.set_title(f'{label}\n{len(group):,} matched tiles', fontsize=12, pad=12)
    fig.suptitle('Tile-level agreement with Xenium reference proportions', fontsize=20, y=0.98)
    fig.text(0.53, 0.933, 'Both rows use the same observed proportions and matched tiles for each cell type.', ha='center', fontsize=12, color='0.25')
    for row, (_, model, color) in enumerate(models):
        fig.text(0.53, 0.88 if row == 0 else 0.46, model, ha='center', fontsize=15, color=color)
    fig.supylabel('Observed Xenium cell proportion', x=0.014, fontsize=14)
    fig.supxlabel('Model-estimated cell proportion', y=0.065, fontsize=14)
    fig.text(0.53, 0.02, 'One dot per tile. Dashed line: estimated = observed. All tiles and raw estimates retained.', ha='center', fontsize=11, color='0.25')
    paths = []
    for suffix in ['png', 'pdf']:
        path = output_dir / f'tile_level_scatter.{suffix}'
        fig.savefig(path, dpi=190, bbox_inches='tight', facecolor='white')
        paths.append(path)
    plt.close(fig)
    return paths

def plot_evaluation(tables, output_dir):
    """Save paired group errors/correlations and pooled calibration diagnostics."""
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    metrics = tables['individual_metrics']
    labels = list(metrics.cell_type.unique())
    colors = {'Visium frozen': '#D55E00', 'Xenium LOIO': '#0072B2', 'Training mean': '#999999'}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), layout='constrained')
    for ax, metric, ylabel in zip(axes, ['mae_pp', 'pearson'], ['MAE (percentage points)', 'Pearson correlation']):
        for i, label in enumerate(labels):
            table = metrics.loc[metrics.cell_type.eq(label)]
            wide = table.pivot(index='individual_id', columns='model', values=metric)
            for _, row in wide.iterrows():
                ax.plot([i - 0.2, i], [row['Visium frozen'], row['Xenium LOIO']], color='0.75', lw=0.7, zorder=1)
            for offset, (model, color) in zip([-0.2, 0, 0.2], colors.items()):
                values = wide[model].dropna()
                ax.scatter(np.full(len(values), i + offset), values, color=color, s=15, alpha=0.65, label=model if i == 0 else None, zorder=2)
                if len(values):
                    ax.plot([i + offset - 0.065, i + offset + 0.065], [values.mean()] * 2, color='black', lw=2)
        ax.set_xticks(range(len(labels)), [label.replace(' Cells', '').replace(' ', '\n') for label in labels])
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.2)
    axes[0].legend(fontsize=9)
    fig.suptitle('Existing Xenium cohort: matched held-out groups (black bars: group means)')
    paths = []
    for suffix in ['png', 'pdf']:
        path = output_dir / f'paired_group_performance.{suffix}'
        fig.savefig(path, dpi=170, bbox_inches='tight')
        paths.append(path)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 4), squeeze=False, layout='constrained')
    for ax, label in zip(axes[0], labels):
        ax.plot([0, 1], [0, 1], '--', color='0.5', lw=1)
        for model, color in colors.items():
            table = tables['calibration_bins']
            table = table.loc[table.cell_type.eq(label) & table.model.eq(model)]
            ax.plot(table.mean_predicted, table.mean_observed, 'o-', color=color, label=model, ms=4)
        ax.set(title=label, xlabel='Mean prediction in bin', ylabel='Mean Xenium reference')
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Pooled calibration diagnostics; no calibration fitted to evaluation labels')
    for suffix in ['png', 'pdf']:
        path = output_dir / f'calibration_diagnostics.{suffix}'
        fig.savefig(path, dpi=170, bbox_inches='tight')
        paths.append(path)
    plt.close(fig)
    return paths
