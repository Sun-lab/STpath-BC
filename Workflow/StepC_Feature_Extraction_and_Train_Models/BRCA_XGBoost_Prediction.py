"""
STPath-BRCA: XGBoost Model Training and Prediction for Breast Cancer
======================================================================

This script trains XGBoost models for predicting cell type proportions in breast cancer
H&E patches using features extracted from multiple foundation models.

Author: Saishi Cui
Date: December 2025

Purpose: Train and evaluate XGBoost models for cell type proportion prediction using
both tile-level and individual-level cross-validation strategies. Supports feature
selection and hyperparameter optimization.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import xgboost as xgb
from datetime import datetime
from scipy import stats
from scipy.optimize import minimize
from matplotlib.patches import Patch
from umap import UMAP
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from joblib import Parallel, delayed
import pickle


### Calibration Functions
def calibrate_predictions(y_true, y_pred, cell_type, n_quantiles=100, outlier_percentile=99.5):
    """
    Calibrate model predictions using global quantile mapping with linear interpolation.
    """
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    y_true_original = np.asarray(y_true).flatten()
    y_pred_original = np.asarray(y_pred).flatten()
    
    print(f"  Original data: {len(y_true_original)} samples")
    print(f"  Pred range: [{y_pred_original.min():.4f}, {y_pred_original.max():.4f}]")
    print(f"  True range: [{y_true_original.min():.4f}, {y_true_original.max():.4f}]")
    
    # Cell types that need preprocessing for BRCA
    needs_preprocessing = cell_type in ["T.cell", "Myeloid", "Other.Epithelial", "B.cell", "Plasma"]
    
    if needs_preprocessing:
        print(f"\n  Preprocessing TRUE proportions for {cell_type}:")
        outlier_threshold = np.percentile(y_true_original, outlier_percentile)
        mask = y_true_original < outlier_threshold
        removed_count = np.sum(~mask)
        
        print(f"     Outlier removal: Remove top {100-outlier_percentile}%")
        print(f"        Threshold (99.5th percentile): {outlier_threshold:.4f}")
        print(f"        Removed {removed_count} outliers")
        
        y_true_filtered = y_true_original[mask]
        y_pred_filtered = y_pred_original[mask]
        
        max_in_remaining = y_true_filtered.max()
        scale_factor = max_in_remaining
        y_true_scaled = y_true_filtered / scale_factor
        
        print(f"     Scaling: max in remaining 99.5% data = {max_in_remaining:.4f}")
        print(f"        Scale factor = {scale_factor:.4f}")
        
        y_true_for_calibration = y_true_scaled
        y_pred_for_calibration = y_pred_filtered
        
        scaling_info = {
            'needs_preprocessing': True,
            'outlier_threshold': outlier_threshold,
            'scale_factor': scale_factor,
            'max_in_remaining': max_in_remaining,
            'removed_count': removed_count,
            'original_samples': len(y_true_original),
            'filtered_samples': len(y_true_scaled),
            'outlier_percentile': outlier_percentile
        }
    else:
        print(f"  No preprocessing for {cell_type}")
        y_true_for_calibration = y_true_original
        y_pred_for_calibration = y_pred_original
        scaling_info = {'needs_preprocessing': False}
    
    print(f"\n  Using global quantile mapping with {n_quantiles} quantiles")
    
    quantile_levels = np.linspace(0, 100, n_quantiles)
    y_pred_quantiles = np.percentile(y_pred_for_calibration, quantile_levels)
    y_true_quantiles = np.percentile(y_true_for_calibration, quantile_levels)
    
    lookup_table = {
        'y_pred_quantiles': y_pred_quantiles,
        'y_true_quantiles': y_true_quantiles,
        'quantile_levels': quantile_levels,
        'scaling_info': scaling_info,
        'calibration_method': 'quantile_mapping'
    }
    
    calibrated_predictions = np.interp(y_pred_original, y_pred_quantiles, y_true_quantiles)
    calibrated_predictions = np.clip(calibrated_predictions, 0, 1)
    
    print(f"  Calibration completed")
    
    return lookup_table, calibrated_predictions, y_true_for_calibration


def plot_calibration_scatter(y_true, y_pred_raw, y_pred_calibrated, y_true_processed,
                            calibration_table=None, dummy=None, output_path=None, model_name=None, cell_type=None):
    """Create calibration comparison plots."""
    scaling_info = calibration_table.get('scaling_info', {'needs_preprocessing': False})
    needs_preprocessing = scaling_info.get('needs_preprocessing', False)
    
    if needs_preprocessing:
        outlier_threshold = scaling_info['outlier_threshold']
        mask = y_true <= outlier_threshold
        y_pred_filtered = y_pred_raw[mask]
        y_true_filtered = y_true[mask]
        scale_factor = scaling_info['scale_factor']
        y_true_scaled = y_true_filtered / scale_factor
    else:
        y_pred_filtered = y_pred_raw
        y_true_scaled = y_true
        y_true_filtered = y_true
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    x_line = np.array([0, 1])
    
    ### Left: Raw predicted vs Raw true
    ax = axes[0]
    mae1 = mean_absolute_error(y_true, y_pred_raw)
    rmse1 = np.sqrt(mean_squared_error(y_true, y_pred_raw))
    spearman1, _ = spearmanr(y_true, y_pred_raw)
    pearson1, _ = pearsonr(y_true, y_pred_raw)
    slope1, intercept1, r1, _, _ = stats.linregress(y_true, y_pred_raw)
    
    ax.scatter(y_true, y_pred_raw, alpha=0.4, s=10, color='blue', edgecolor='none')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Identity')
    ax.plot(x_line, intercept1 + slope1 * x_line, 'r-', linewidth=2, alpha=0.7, label='Fit')
    
    textstr1 = f'MAE: {mae1:.4f}\nRMSE: {rmse1:.4f}\n'
    textstr1 += f'Spearman: {spearman1:.4f}\nPearson: {pearson1:.4f}\n'
    textstr1 += f'Slope: {slope1:.4f}\nR²: {r1**2:.4f}'
    ax.text(0.05, 0.95, textstr1, transform=ax.transAxes, fontsize=11, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Raw True Proportion', fontsize=13, fontweight='bold')
    ax.set_ylabel('Raw Predicted Proportion', fontsize=13, fontweight='bold')
    ax.set_title(f'(A) Raw Predicted vs Raw True\n{model_name} - {cell_type}', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='lower right')
    
    ### Middle: Raw predicted vs Calibrated true
    ax = axes[1]
    mae2 = mean_absolute_error(y_true_scaled, y_pred_filtered)
    rmse2 = np.sqrt(mean_squared_error(y_true_scaled, y_pred_filtered))
    spearman2, _ = spearmanr(y_true_scaled, y_pred_filtered)
    pearson2, _ = pearsonr(y_true_scaled, y_pred_filtered)
    slope2, intercept2, r2, _, _ = stats.linregress(y_true_scaled, y_pred_filtered)
    
    ax.scatter(y_true_scaled, y_pred_filtered, alpha=0.4, s=10, color='green', edgecolor='none')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Identity')
    ax.plot(x_line, intercept2 + slope2 * x_line, 'r-', linewidth=2, alpha=0.7, label='Fit')
    
    textstr2 = f'MAE: {mae2:.4f}\nRMSE: {rmse2:.4f}\n'
    textstr2 += f'Spearman: {spearman2:.4f}\nPearson: {pearson2:.4f}\n'
    textstr2 += f'Slope: {slope2:.4f}\nR²: {r2**2:.4f}'
    ax.text(0.05, 0.95, textstr2, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax.set_xlabel('Calibrated True Proportion', fontsize=13, fontweight='bold')
    ax.set_ylabel('Raw Predicted Proportion', fontsize=13, fontweight='bold')
    ax.set_title(f'(B) Raw Predicted vs Calibrated True\n{model_name} - {cell_type}', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='lower right')
    
    ### Right: Quantile Mapping Curve
    ax = axes[2]
    if calibration_table is not None:
        y_pred_quantiles = calibration_table['y_pred_quantiles']
        y_true_quantiles = calibration_table['y_true_quantiles']
        
        ax.plot(y_pred_quantiles, y_true_quantiles, 'b-o', linewidth=2, markersize=4, 
                alpha=0.7, label='Quantile Mapping')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Identity')
        
        ax.set_xlabel('Raw Predicted Proportion', fontsize=13, fontweight='bold')
        ax.set_ylabel('Calibrated Predicted Proportion', fontsize=13, fontweight='bold')
        ax.set_title(f'(C) Quantile Mapping Curve\n{model_name} - {cell_type}', fontsize=14, fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='lower right')
    
    plt.suptitle(f'Calibration Analysis: {cell_type}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Calibration plot saved: {output_path}")


### XGBoost Training Functions
def train_xgboost_tile_level(
    input_dir='Breast_Cancer_HE_patches/Training_features',
    cell_type="Cancer.Epithelial",
    model_name='UNI2h',
    output_dir=None,
    params=None,
    num_boost_round=500,
    seed=42,
    if_combined=False
):
    """Tile-level 5-fold cross-validation"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_dir is None:
        output_dir = f'Breast_Cancer_HE_patches/xgboost_prediction/{cell_type}_{model_name}_tile_level'
    os.makedirs(output_dir, exist_ok=True)
    
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'n_jobs': -1,
            'eta': 0.01,
            'gamma': 0,
            'min_child_weight': 5,
            'colsample_bytree': 0.05,
            'subsample': 0.5,
            'alpha': 0.1,
            'lambda': 0.01,
            'max_depth': 12,
            'seed': seed
        }
        print("Using XGBoost with optimized CPU acceleration")
    
    if if_combined:
        UNI2h_data = torch.load(input_dir + f"/{cell_type}_training_precomputed_features_UNI2h.pt")
        Virchow_data = torch.load(input_dir + f"/{cell_type}_training_precomputed_features_Virchow.pt")
        Virchow2_data = torch.load(input_dir + f"/{cell_type}_training_precomputed_features_Virchow2.pt")
        ProvGigapath_data = torch.load(input_dir + f"/{cell_type}_training_precomputed_features_ProvGigapath.pt")
        Conch_data = torch.load(input_dir + f"/{cell_type}_training_precomputed_features_Conch.pt")

        features = torch.cat([UNI2h_data['embeddings'], Virchow_data['embeddings'], 
                             Virchow2_data['embeddings'], ProvGigapath_data['embeddings'], 
                             Conch_data['embeddings']], dim=1)
        data = UNI2h_data
        labels = data['celltype_proportions']
    else:
        feature_name = f'{cell_type}_training_precomputed_features_{model_name}.pt'
        data_path = input_dir + "/" + feature_name
        print(f"Loading data: {data_path}")
        data = torch.load(data_path)
        features = data['embeddings']
        labels = data['celltype_proportions']

    tile_ids = data.get('tile_ids', [])
    print(f"Dataset contains: {len(tile_ids)} tiles")
    
    tile_indices = list(range(len(tile_ids)))
    np.random.shuffle(tile_indices)
    fold_size = len(tile_indices) // 5
    fold_indices = []
    for i in range(5):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < 4 else len(tile_indices)
        test_indices = tile_indices[start_idx:end_idx]
        train_indices = [idx for idx in tile_indices if idx not in test_indices]
        fold_indices.append((train_indices, test_indices))
    
    all_predictions = np.zeros_like(labels)
    all_importances = []
    is_test = np.zeros(len(labels), dtype=bool)
    fold_metrics = []
    
    for fold_idx, (train_indices, test_indices) in enumerate(fold_indices):
        print(f"\nStarting Tile-level fold {fold_idx+1}/5 prediction...")
        is_test[test_indices] = True
        
        X_train, y_train = features[train_indices], labels[train_indices]
        X_test, y_test = features[test_indices], labels[test_indices]
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        model = xgb.train(
            params, dtrain, num_boost_round=num_boost_round,
            evals=[(dtrain, 'train')], verbose_eval=100
        )
        
        importance = model.get_score(importance_type='gain')
        all_importances.append(importance)
        
        predictions = model.predict(dtest)
        all_predictions[test_indices] = predictions
        
        fold_mae = mean_absolute_error(y_test, predictions)
        fold_rmse = np.sqrt(mean_squared_error(y_test, predictions))
        fold_spearman, _ = spearmanr(y_test, predictions)
        fold_pearson, _ = pearsonr(y_test, predictions)
        
        fold_metrics.append({
            'Fold': fold_idx + 1,
            'MAE': fold_mae, 'RMSE': fold_rmse,
            'Spearman': fold_spearman, 'Pearson': fold_pearson,
            'n_test_samples': len(test_indices)
        })
        
        print(f"Fold {fold_idx+1} metrics: MAE={fold_mae:.4f}, RMSE={fold_rmse:.4f}")
        
        model_dir = os.path.join(output_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        model.save_model(os.path.join(model_dir, f'xgboost_model_fold_{fold_idx+1}.model'))
    
    fold_metrics_df = pd.DataFrame(fold_metrics)
    overall_mae = mean_absolute_error(labels, all_predictions)
    overall_rmse = np.sqrt(mean_squared_error(labels, all_predictions))
    overall_spearman, _ = spearmanr(labels, all_predictions)
    overall_pearson, _ = pearsonr(labels, all_predictions)
    
    print(f"\nOverall metrics: MAE={overall_mae:.4f}, RMSE={overall_rmse:.4f}, Spearman={overall_spearman:.4f}")
    
    results = {
        'predictions': all_predictions, 'labels': labels, 'tile_ids': tile_ids,
        'is_test': is_test, 'feature_importances': all_importances,
        'params': params, 'model_name': model_name, 'cell_type': cell_type,
        'metrics': {'overall_mae': overall_mae, 'overall_rmse': overall_rmse,
                   'overall_spearman': overall_spearman, 'overall_pearson': overall_pearson},
        'fold_metrics': fold_metrics
    }
    
    results_path = os.path.join(output_dir, 'xgboost_results_tile_level.pt')
    torch.save(results, results_path)
    print(f"Results saved to: {results_path}")
    
    return results


def train_xgboost_individual_level(
    input_dir='Breast_Cancer_HE_patches/Training_features',
    cell_type="Cancer.Epithelial",
    model_name='UNI2h',
    output_dir=None,
    params=None,
    num_boost_round=500,
    seed=42,
    if_combined=False,
    training_data_ratio=1.0,
    save_files=True
):
    """Individual-level leave-one-out training with individual-level testing"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if save_files:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if output_dir is None:
            ratio_str = f"ratio{int(training_data_ratio*100)}"
            output_dir = f'Breast_Cancer_HE_patches/xgboost_prediction/{cell_type}_{model_name}_individual_level_{ratio_str}'
        os.makedirs(output_dir, exist_ok=True)
    
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'n_jobs': -1,
            'eta': 0.01,
            'gamma': 0,
            'min_child_weight': 5,
            'colsample_bytree': 0.05,
            'subsample': 0.5,
            'alpha': 0.1,
            'lambda': 0.01,
            'max_depth': 12,
            'seed': seed
        }
        print("Using XGBoost with optimized CPU acceleration")
    
    if if_combined:
        important_features_path = f"/Users/scui2/Desktop/Breast_Cancer_HE_patches/xgboost_prediction/important_features_{cell_type}.pkl"
        important_features = pickle.load(open(important_features_path, "rb"))
        
        UNI2h_data = torch.load(input_dir + f"/{cell_type}_training_precomputed_features_UNI2h.pt")
        Virchow_data = torch.load(input_dir + f"/{cell_type}_training_precomputed_features_Virchow.pt")
        Virchow2_data = torch.load(input_dir + f"/{cell_type}_training_precomputed_features_Virchow2.pt")
        ProvGigapath_data = torch.load(input_dir + f"/{cell_type}_training_precomputed_features_ProvGigapath.pt")
        Conch_data = torch.load(input_dir + f"/{cell_type}_training_precomputed_features_Conch.pt")

        UNI2h_important_indices = [int(f.replace('f', '')) for f in important_features["UNI2h"]]
        Virchow_important_indices = [int(f.replace('f', '')) for f in important_features["Virchow"]]
        Virchow2_important_indices = [int(f.replace('f', '')) for f in important_features["Virchow2"]]
        ProvGigapath_important_indices = [int(f.replace('f', '')) for f in important_features["ProvGigapath"]]
        Conch_important_indices = [int(f.replace('f', '')) for f in important_features["Conch"]]
        
        UNI2h_selected = UNI2h_data['embeddings'][:, UNI2h_important_indices]
        Virchow_selected = Virchow_data['embeddings'][:, Virchow_important_indices]
        Virchow2_selected = Virchow2_data['embeddings'][:, Virchow2_important_indices]
        ProvGigapath_selected = ProvGigapath_data['embeddings'][:, ProvGigapath_important_indices]
        Conch_selected = Conch_data['embeddings'][:, Conch_important_indices]

        features = torch.cat([UNI2h_selected, Virchow_selected, Virchow2_selected, ProvGigapath_selected, Conch_selected], dim=1)
        data = UNI2h_data
        labels = data['celltype_proportions']
        print(f"Using important features: Total selected features: {features.shape[1]}")
    else:
        feature_name = f'{cell_type}_training_precomputed_features_{model_name}.pt'
        data_path = input_dir + "/" + feature_name
        print(f"Loading data: {data_path}")
        data = torch.load(data_path)
        features = data['embeddings']
        labels = data['celltype_proportions']

    tile_ids = data.get('tile_ids', [])
    individual_ids = data.get('individual_ids', [])
    sample_ids = data.get('sample_ids', [])
    
    unique_individuals = sorted(list(set(individual_ids)))
    print(f"Dataset contains: {len(unique_individuals)} individuals")
    
    all_predictions = np.zeros_like(labels)
    all_importances = []
    is_test = np.zeros(len(labels), dtype=bool)
    fold_metrics = []
    
    for ind_idx, ind in enumerate(unique_individuals):
        print(f"\nLeaving individual {ind} out ({ind_idx+1}/{len(unique_individuals)})...")
        
        individual_mask = np.array([iid == ind for iid in individual_ids])
        train_mask = np.array([iid != ind for iid in individual_ids])
        all_train_indices = np.where(train_mask)[0]
        
        if training_data_ratio < 1.0:
            np.random.seed(seed + ind_idx)
            n_samples = int(len(all_train_indices) * training_data_ratio)
            train_indices = np.random.choice(all_train_indices, size=n_samples, replace=False)
            train_indices = np.sort(train_indices)
        else:
            train_indices = all_train_indices
        
        X_train, y_train = features[train_indices], labels[train_indices]
        print(f"Training samples: {len(X_train)} ({training_data_ratio*100:.1f}%)")
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        model = xgb.train(
            params, dtrain, num_boost_round=num_boost_round,
            evals=[(dtrain, 'train')], verbose_eval=100
        )
        
        importance = model.get_score(importance_type='gain')
        all_importances.append(importance)
        
        if save_files:
            model_dir = os.path.join(output_dir, 'models')
            os.makedirs(model_dir, exist_ok=True)
            model.save_model(os.path.join(model_dir, f'xgboost_model_leave_{ind}_out.model'))
        
        test_mask = np.array([iid == ind for iid in individual_ids])
        test_indices = np.where(test_mask)[0]
        is_test[test_indices] = True
        
        X_test, y_test = features[test_indices], labels[test_indices]
        dtest = xgb.DMatrix(X_test, label=y_test)
        predictions = model.predict(dtest)
        all_predictions[test_indices] = predictions
        
        individual_mae = mean_absolute_error(y_test, predictions)
        individual_rmse = np.sqrt(mean_squared_error(y_test, predictions))
        individual_spearman, _ = spearmanr(y_test, predictions)
        individual_pearson, _ = pearsonr(y_test, predictions)
        
        fold_metrics.append({
            'Fold': ind, 'Individual': ind,
            'MAE': individual_mae, 'RMSE': individual_rmse,
            'Spearman': individual_spearman, 'Pearson': individual_pearson,
            'n_test_samples': len(test_indices)
        })
        
        print(f"    Individual {ind}: MAE={individual_mae:.4f}, Spearman={individual_spearman:.4f}")
    
    fold_metrics_df = pd.DataFrame(fold_metrics)
    average_metrics = {
        'MAE': fold_metrics_df['MAE'].mean(),
        'RMSE': fold_metrics_df['RMSE'].mean(),
        'Spearman': fold_metrics_df['Spearman'].mean(),
        'Pearson': fold_metrics_df['Pearson'].mean(),
        'MAE_std': fold_metrics_df['MAE'].std(),
        'Spearman_std': fold_metrics_df['Spearman'].std()
    }
    
    print(f"\nAverage metrics: MAE={average_metrics['MAE']:.4f} ± {average_metrics['MAE_std']:.4f}")
    
    overall_mae = mean_absolute_error(labels, all_predictions)
    overall_rmse = np.sqrt(mean_squared_error(labels, all_predictions))
    overall_spearman, _ = spearmanr(labels, all_predictions)
    overall_pearson, _ = pearsonr(labels, all_predictions)
    
    print(f"\nOverall metrics (raw): MAE={overall_mae:.4f}, Spearman={overall_spearman:.4f}")
    
    print(f"\nPerforming cross-model calibration...")
    calibration_table, calibrated_predictions, labels_processed = calibrate_predictions(labels, all_predictions, cell_type)
    
    cal_mae = mean_absolute_error(labels, calibrated_predictions)
    cal_spearman, _ = spearmanr(labels, calibrated_predictions)
    print(f"\nCalibrated metrics: MAE={cal_mae:.4f}, Spearman={cal_spearman:.4f}")
    
    if save_files:
        fold_metrics_df.to_csv(os.path.join(output_dir, 'individual_metrics.csv'), index=False)
        
        tile_predictions_df = pd.DataFrame({
            'tile_id': tile_ids, 'sample_id': sample_ids, 'individual_id': individual_ids,
            'deconvoluted_proportion': labels,
            'predicted_proportion_raw': all_predictions,
            'predicted_proportion_calibrated': calibrated_predictions
        })
        tile_predictions_df.to_csv(os.path.join(output_dir, 'tile_level_predictions_detailed.csv'), index=False)
    
    results = {
        'predictions': all_predictions, 'predictions_calibrated': calibrated_predictions,
        'labels': labels, 'tile_ids': tile_ids, 'individual_ids': individual_ids,
        'sample_ids': sample_ids, 'feature_importances': all_importances,
        'params': params, 'model_name': model_name, 'cell_type': cell_type,
        'metrics': {'overall_mae': overall_mae, 'overall_spearman': overall_spearman},
        'calibration': {'lookup_table': calibration_table, 'calibrated_mae': cal_mae},
        'individual_metrics': fold_metrics, 'average_metrics': average_metrics
    }
    
    if save_files and output_dir:
        results_path = os.path.join(output_dir, 'xgboost_results_individual_level.pt')
        torch.save(results, results_path)
        print(f"Results saved to: {results_path}")
    
    return results


def train_xgboost_for_external_prediction(
    input_dir='Breast_Cancer_HE_patches/Training_features',
    cell_type="Cancer.Epithelial",
    model_name='Combined',
    output_dir=None,
    params=None,
    num_boost_round=800,
    seed=42,
    if_combined=True
):
    """Train XGBoost model on all available data for external prediction"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if output_dir is None:
        output_dir = f'Breast_Cancer_HE_patches/xgboost_prediction/{cell_type}_{model_name}_external_prediction'
    os.makedirs(output_dir, exist_ok=True)
    
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'n_jobs': -1,
            'eta': 0.01,
            'gamma': 0,
            'min_child_weight': 5,
            'colsample_bytree': 0.05,
            'subsample': 0.5,
            'alpha': 0.1,
            'lambda': 0.01,
            'max_depth': 12,
            'seed': seed
        }
        print("Using optimized XGBoost parameters for external prediction")
    
    if if_combined:
        important_features_path = f"/Users/scui2/Desktop/Breast_Cancer_HE_patches/xgboost_prediction/important_features_{cell_type}.pkl"
        important_features = pickle.load(open(important_features_path, "rb"))
        
        print("Loading feature data...")
        UNI2h_data = torch.load(input_dir + f"/{cell_type}_training_precomputed_features_UNI2h.pt")
        Virchow_data = torch.load(input_dir + f"/{cell_type}_training_precomputed_features_Virchow.pt")
        Virchow2_data = torch.load(input_dir + f"/{cell_type}_training_precomputed_features_Virchow2.pt")
        ProvGigapath_data = torch.load(input_dir + f"/{cell_type}_training_precomputed_features_ProvGigapath.pt")
        Conch_data = torch.load(input_dir + f"/{cell_type}_training_precomputed_features_Conch.pt")

        UNI2h_important_indices = [int(f.replace('f', '')) for f in important_features["UNI2h"]]
        Virchow_important_indices = [int(f.replace('f', '')) for f in important_features["Virchow"]]
        Virchow2_important_indices = [int(f.replace('f', '')) for f in important_features["Virchow2"]]
        ProvGigapath_important_indices = [int(f.replace('f', '')) for f in important_features["ProvGigapath"]]
        Conch_important_indices = [int(f.replace('f', '')) for f in important_features["Conch"]]
        
        UNI2h_selected = UNI2h_data['embeddings'][:, UNI2h_important_indices]
        Virchow_selected = Virchow_data['embeddings'][:, Virchow_important_indices]
        Virchow2_selected = Virchow2_data['embeddings'][:, Virchow2_important_indices]
        ProvGigapath_selected = ProvGigapath_data['embeddings'][:, ProvGigapath_important_indices]
        Conch_selected = Conch_data['embeddings'][:, Conch_important_indices]

        features = torch.cat([UNI2h_selected, Virchow_selected, Virchow2_selected, ProvGigapath_selected, Conch_selected], dim=1)
        data = UNI2h_data
        labels = data['celltype_proportions']
        print(f"Total selected features: {features.shape[1]}")
    else:
        feature_name = f'{cell_type}_training_precomputed_features_{model_name}.pt'
        data_path = input_dir + "/" + feature_name
        print(f"Loading data: {data_path}")
        data = torch.load(data_path)
        features = data['embeddings']
        labels = data['celltype_proportions']

    print(f"Training data: {features.shape[0]} samples, {features.shape[1]} features")
    
    print(f"\nTraining XGBoost model on all {features.shape[0]} samples...")
    dtrain = xgb.DMatrix(features, label=labels)
    
    model = xgb.train(
        params, dtrain, num_boost_round=num_boost_round,
        evals=[(dtrain, 'train')], verbose_eval=100
    )
    
    model_path = os.path.join(output_dir, f'xgboost_model_{cell_type.replace(".", "_")}_{model_name}_external_prediction.model')
    model.save_model(model_path)
    print(f"Model saved to: {model_path}")
    
    feature_info = {
        'cell_type': cell_type, 'model_name': model_name, 'if_combined': if_combined,
        'total_features': features.shape[1], 'training_samples': features.shape[0],
        'params': params, 'num_boost_round': num_boost_round, 'seed': seed
    }
    
    info_path = os.path.join(output_dir, 'model_info.pt')
    torch.save(feature_info, info_path)
    
    print(f"\nExternal prediction model training completed!")
    return {'model_path': model_path, 'info_path': info_path, 'output_dir': output_dir}


if __name__ == "__main__":
    
    ### BRCA Cell Types
    CELL_TYPES = ["Cancer.Epithelial", "Other.Epithelial", "CAFs", "Myeloid", "Plasma", "B.cell", "T.cell", "Endothelial"]
    BASE_OUTPUT_DIR = '/Users/scui2/Desktop/Breast_Cancer_HE_patches/xgboost_prediction_with_calibration'
    INPUT_DIR = '/Users/scui2/Desktop/Breast_Cancer_HE_patches/Training_features'
    
    print(f"\nSTAGE 1: XGBoost Model Training")
    print(f"Cell Types: {CELL_TYPES}")
    
    ### Train models for all cell types
    for idx, cell_type in enumerate(CELL_TYPES):
        print(f"\nTraining Cell Type {idx+1}/{len(CELL_TYPES)}: {cell_type}")
        
        cell_type_folder = cell_type.replace(".", "_").replace("/", "_")
        output_dir = f"{BASE_OUTPUT_DIR}/{cell_type_folder}_Combined_individual_level"
        
        results = train_xgboost_individual_level(
            input_dir=INPUT_DIR,
            cell_type=cell_type,
            model_name='Combined',
            output_dir=output_dir,
            params=None,
            num_boost_round=800,
            seed=42,
            if_combined=True,
            training_data_ratio=1.0,
            save_files=True
        )
        
        print(f"\nCompleted: {cell_type}")
        print(f"   Raw MAE: {results['metrics']['overall_mae']:.4f}")
        print(f"   Calibrated MAE: {results['calibration']['calibrated_mae']:.4f}")
    
    print(f"\nSTAGE 1 COMPLETED: All models trained!")

