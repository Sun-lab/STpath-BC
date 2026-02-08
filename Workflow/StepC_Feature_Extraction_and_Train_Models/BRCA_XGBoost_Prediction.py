"""
STPath-BRCA: XGBoost Model Training for Breast Cancer
==========================================================================

This script trains XGBoost models for predicting cell type proportions in breast cancer
H&E patches using features extracted from multiple foundation models.

Updated to use 11 common intersected samples (samples present across all foundation models).

Author: Saishi Cui
Date: February 2026

Cell Types (BRCA):
- **Tumor Cells** - Malignant epithelial cells
- **Stromal Cells** - Fibroblasts and cancer-associated fibroblasts (CAFs)
- **Normal Epithelial Cells** - Non-malignant epithelial cells
- **T Cells** - T lymphocytes (CD4+ and CD8+ T cells)
- **pan-APC Cells** - B cells and Myeloid cells combined (antigen-presenting cells)

Models:
- Individual foundation models: ResNet50, Conch, ProvGigapath, UNI2h, Virchow, Virchow2
- Combined: Selected important features from all 5 foundation models (excluding ResNet50)
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
import pickle
from scipy.optimize import minimize


# =============================================================================
# Configuration
# =============================================================================

# Input/Output Paths
INPUT_DIR = '/BRCA_training/BRCA_training_features_11samples'
BASE_OUTPUT_DIR = '//BRCA_XGBoost_Results/New_common_samples'
IMPORTANT_FEATURES_DIR = '/BRCA_XGBoost_Results/New_common_samples'

# Cell types for BRCA (using simplified names matching Figure 7)
CELL_TYPES = ["Tumor", "Stromal", "pan_APC", "T_cell", "Normal_Epithelial"]

# Models to train
INDIVIDUAL_MODELS = ['ResNet50', 'Conch', 'ProvGigapath', 'UNI2h', 'Virchow', 'Virchow2']
COMBINED_MODEL = 'Combined'

# XGBoost parameters (optimized for BRCA)
DEFAULT_PARAMS = {
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
    'seed': 42
}

NUM_BOOST_ROUND = 800


# =============================================================================
# Helper Functions
# =============================================================================

def get_pt_filename(cell_type, model_name):
    """Get the .pt filename for loading features"""
    # File format: {cell_type}_training_precomputed_features_{model_name}.pt
    return f"{cell_type}_training_precomputed_features_{model_name}.pt"


def load_important_features(cell_type):
    """Load important features for combined model"""
    pkl_path = os.path.join(IMPORTANT_FEATURES_DIR, f"important_features_{cell_type}.pkl")
    
    if not os.path.exists(pkl_path):
        print(f"Warning: Important features file not found: {pkl_path}")
        print("Using all features for combined model.")
        return None
    
    with open(pkl_path, 'rb') as f:
        important_features = pickle.load(f)
    
    return important_features


def calibrate_predictions(y_true, y_pred, cell_type, n_quantiles=100, outlier_percentile=99.5):
    """
    Calibrate model predictions using global quantile mapping with linear interpolation.
    For certain cell types (T_cell, pan_APC, Normal_Epithelial),
    first preprocess true proportions by removing outliers and scaling.
    
    Preprocessing steps (for specific cell types):
    1. Remove outliers: Keep only values <= outlier_percentile (default 99.5%)
    2. Scale: Divide all values by the max (99.5th percentile value)
    3. Apply quantile mapping on processed data
    
    This method:
    1. Computes n_quantiles percentiles of both y_pred and y_true_processed
    2. Creates a lookup table mapping y_pred quantiles to y_true_processed quantiles
    3. For new predictions, uses linear interpolation (np.interp) between quantile points
    
    Args:
        y_true: True cell type proportions (deconvoluted) - can be torch tensor or numpy array
        y_pred: Raw model predictions - can be torch tensor or numpy array
        cell_type: Name of cell type (determines if preprocessing is needed)
        n_quantiles: Number of quantiles to store (default: 100, i.e., percentiles)
        outlier_percentile: Percentile threshold for outlier removal (default: 99.5)
    
    Returns:
        tuple: (lookup_table, calibrated_predictions, y_true_processed)
        - lookup_table: Dict with quantile info and scaling parameters
        - calibrated_predictions: Calibrated values for current data
        - y_true_processed: Processed true values (for plotting)
    """
    
    # Convert torch tensors to numpy if needed
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    # Ensure numpy arrays
    y_true_original = np.asarray(y_true).flatten()
    y_pred_original = np.asarray(y_pred).flatten()
    
    print(f"  Original data: {len(y_true_original)} samples")
    print(f"  Pred range: [{y_pred_original.min():.4f}, {y_pred_original.max():.4f}]")
    print(f"  True range: [{y_true_original.min():.4f}, {y_true_original.max():.4f}]")
    
    # Check if this cell type needs true proportion preprocessing
    # Mapping BRCA cell types to preprocessing needs (similar to COAD)
    needs_preprocessing = cell_type in ["T_cell", "pan_APC", "Normal_Epithelial"]
    
    if needs_preprocessing:
        print(f"\n  🔧 Preprocessing TRUE proportions for {cell_type}:")
        
        # Step 1: Remove top 0.5% outliers (keep bottom 99.5%)
        outlier_threshold = np.percentile(y_true_original, outlier_percentile)
        mask = y_true_original < outlier_threshold  # Strict inequality to remove exactly top 0.5%
        removed_count = np.sum(~mask)
        
        print(f"     ① Outlier removal: Remove top {100-outlier_percentile}%")
        print(f"        Threshold (99.5th percentile): {outlier_threshold:.4f}")
        print(f"        Removed {removed_count} outliers (≥{outlier_threshold:.4f})")
        
        # Apply mask to both arrays
        y_true_filtered = y_true_original[mask]
        y_pred_filtered = y_pred_original[mask]
        
        # Step 2: Find max value in the REMAINING 99.5% data and use it as scaling factor
        max_in_remaining = y_true_filtered.max()
        scale_factor = max_in_remaining
        y_true_scaled = y_true_filtered / scale_factor
        
        print(f"     ② Scaling: max in remaining 99.5% data = {max_in_remaining:.4f}")
        print(f"        Scale factor = {scale_factor:.4f}")
        print(f"        Scaled TRUE range: [{y_true_scaled.min():.4f}, {y_true_scaled.max():.4f}]")
        print(f"        Samples after preprocessing: {len(y_true_scaled)}")
        
        # Use processed data for calibration
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
        print(f"  ℹ️  No preprocessing for {cell_type}")
        y_true_for_calibration = y_true_original
        y_pred_for_calibration = y_pred_original
        scaling_info = {'needs_preprocessing': False}
    
    print(f"\n  Using global quantile mapping with {n_quantiles} quantiles")
    
    # Compute quantiles
    quantile_levels = np.linspace(0, 100, n_quantiles)
    y_pred_quantiles = np.percentile(y_pred_for_calibration, quantile_levels)
    y_true_quantiles = np.percentile(y_true_for_calibration, quantile_levels)
    
    # Create lookup table
    lookup_table = {
        'y_pred_quantiles': y_pred_quantiles,
        'y_true_quantiles': y_true_quantiles,
        'quantile_levels': quantile_levels,
        'scaling_info': scaling_info,
        'calibration_method': 'quantile_mapping'
    }
    
    print(f"  Lookup table:")
    print(f"    y_pred range: [{y_pred_quantiles[0]:.4f}, {y_pred_quantiles[-1]:.4f}]")
    print(f"    y_true range: [{y_true_quantiles[0]:.4f}, {y_true_quantiles[-1]:.4f}]")
    
    # Apply calibration to ORIGINAL full y_pred (with linear interpolation)
    calibrated_predictions = np.interp(y_pred_original, y_pred_quantiles, y_true_quantiles)
    calibrated_predictions = np.clip(calibrated_predictions, 0, 1)
    
    n_clipped = np.sum((calibrated_predictions == 0) | (calibrated_predictions == 1))
    if n_clipped > 0:
        print(f"  Clipped: {n_clipped} values ({n_clipped/len(y_pred_original)*100:.1f}%)")
    
    print(f"  ✅ Calibration completed")
    
    return lookup_table, calibrated_predictions, y_true_for_calibration


def plot_calibration_scatter(y_true, y_pred_raw, y_pred_calibrated, y_true_processed,
                            calibration_table=None, dummy=None, output_path=None, model_name=None, cell_type=None):
    """
    Create 1x3 plots:
    - Left: Raw predicted vs Raw true
    - Middle: Raw predicted vs Calibrated true (scaled/processed)
    - Right: Calibration curve (quantile mapping visualization)
    
    Args:
        y_true: Original true cell type proportions
        y_pred_raw: Raw model predictions
        y_pred_calibrated: Calibrated predictions
        y_true_processed: Processed true proportions (scaled/filtered for certain cell types)
        calibration_table: The lookup table with quantile info
        dummy: Placeholder for compatibility
        output_path: Path to save the plot
        model_name: Name of the model
        cell_type: Cell type name
    """
    
    # Handle processed data
    scaling_info = calibration_table.get('scaling_info', {'needs_preprocessing': False})
    needs_preprocessing = scaling_info.get('needs_preprocessing', False)
    
    if needs_preprocessing:
        # Need to filter y_pred and y_true to match y_true_processed
        outlier_threshold = scaling_info['outlier_threshold']
        mask = y_true <= outlier_threshold
        y_pred_filtered = y_pred_raw[mask]
        y_true_filtered = y_true[mask]
        
        # Scale true values
        scale_factor = scaling_info['scale_factor']
        y_true_scaled = y_true_filtered / scale_factor
    else:
        y_pred_filtered = y_pred_raw
        y_true_scaled = y_true
        y_true_filtered = y_true
    
    # Create 1x3 subplot layout
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    x_line = np.array([0, 1])
    
    ##########################################################################
    ### Left: Raw predicted vs Raw true
    ##########################################################################
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
    
    ##########################################################################
    ### Middle: Raw predicted vs Calibrated (scaled) true
    ##########################################################################
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
    if needs_preprocessing:
        textstr2 += f'\n\n✂️ Outliers removed\n📏 Scaled by {scaling_info["scale_factor"]:.4f}'
    ax.text(0.05, 0.95, textstr2, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax.set_xlabel('Calibrated True Proportion', fontsize=13, fontweight='bold')
    ax.set_ylabel('Raw Predicted Proportion', fontsize=13, fontweight='bold')
    ax.set_title(f'(B) Raw Predicted vs Calibrated True\n{model_name} - {cell_type}', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='lower right')
    
    ##########################################################################
    ### Right: Quantile Mapping Curve (100 points connected by lines)
    ##########################################################################
    ax = axes[2]
    
    if calibration_table is not None:
        # Get the quantile points
        y_pred_quantiles = calibration_table['y_pred_quantiles']
        y_true_quantiles = calibration_table['y_true_quantiles']
        
        # Plot the quantile mapping as a line connecting points
        ax.plot(y_pred_quantiles, y_true_quantiles, 'b-o', linewidth=2, markersize=4, 
                alpha=0.7, label='Quantile Mapping')
        
        # Plot identity line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Identity')
        
        # Add text info
        n_quantiles = len(y_pred_quantiles)
        textstr_qm = f'Quantile Mapping\n'
        textstr_qm += f'{n_quantiles} quantiles\n'
        textstr_qm += f'Linear interpolation\n\n'
        textstr_qm += f'y_pred range:\n[{y_pred_quantiles[0]:.4f}, {y_pred_quantiles[-1]:.4f}]\n\n'
        textstr_qm += f'y_true range:\n[{y_true_quantiles[0]:.4f}, {y_true_quantiles[-1]:.4f}]'
        
        if scaling_info.get('needs_preprocessing', False):
            textstr_qm += f'\n\n✂️ Preprocessing applied'
        
        ax.text(0.05, 0.95, textstr_qm, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax.set_xlabel('Raw Predicted Proportion', fontsize=13, fontweight='bold')
        ax.set_ylabel('Calibrated Predicted Proportion', fontsize=13, fontweight='bold')
        ax.set_title(f'(C) Quantile Mapping Curve\n{model_name} - {cell_type}', fontsize=14, fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='lower right')
    else:
        ax.text(0.5, 0.5, 'Calibration table not available', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
    
    plt.suptitle(f'Calibration Analysis: {cell_type}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Calibration plot (1x3) saved: {output_path}")


# =============================================================================
# Training Functions
# =============================================================================

def train_xgboost_individual_level(
    input_dir,
    cell_type,
    model_name,
    output_dir,
    params=None,
    num_boost_round=800,
    seed=42,
    if_combined=False,
    save_files=True
):
    """
    Train XGBoost with individual-level leave-one-out cross-validation.
    
    Args:
        input_dir: Directory containing precomputed features (.pt files)
        cell_type: Cell type name (display name, e.g., "Tumor", "pan_APC")
        model_name: Model name (e.g., "UNI2h", "Combined")
        output_dir: Directory to save results
        params: XGBoost parameters (uses DEFAULT_PARAMS if None)
        num_boost_round: Number of boosting rounds
        seed: Random seed
        if_combined: Whether to use combined model with important features
        save_files: Whether to save output files
    
    Returns:
        Dictionary containing predictions, metrics, and other results
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if save_files:
        os.makedirs(output_dir, exist_ok=True)
    
    if params is None:
        params = DEFAULT_PARAMS.copy()
        params['seed'] = seed
        print("Using optimized XGBoost parameters")
    
    # Load features
    if if_combined:
        print(f"\nLoading combined features for {cell_type}...")
        
        # Load important features
        important_features = load_important_features(cell_type)
        
        if important_features is None:
            # Use all features if important features not available
            print("Loading all features from 5 foundation models...")
            models_to_load = ['UNI2h', 'Virchow', 'Virchow2', 'ProvGigapath', 'Conch']
            all_embeddings = []
            
            for idx, fm in enumerate(models_to_load):
                pt_filename = get_pt_filename(cell_type, fm)
                pt_path = os.path.join(input_dir, pt_filename)
                
                if not os.path.exists(pt_path):
                    raise FileNotFoundError(f"Feature file not found: {pt_path}")
                
                fm_data = torch.load(pt_path, weights_only=False)
                all_embeddings.append(fm_data['embeddings'])
                
                if idx == 0:
                    data = fm_data  # Use first model's data for labels, IDs, etc.
            
            features = torch.cat(all_embeddings, dim=1)
            print(f"Total features: {features.shape[1]}")
            
        else:
            # Use selected important features
            print("Loading selected important features from 5 foundation models...")
            models_to_load = ['UNI2h', 'Virchow', 'Virchow2', 'ProvGigapath', 'Conch']
            selected_embeddings = []
            
            for idx, fm in enumerate(models_to_load):
                pt_filename = get_pt_filename(cell_type, fm)
                pt_path = os.path.join(input_dir, pt_filename)
                
                if not os.path.exists(pt_path):
                    raise FileNotFoundError(f"Feature file not found: {pt_path}")
                
                fm_data = torch.load(pt_path, weights_only=False)
                
                # Get important feature indices for this model
                if fm in important_features:
                    important_indices = [int(f.replace('f', '')) for f in important_features[fm]]
                    selected = fm_data['embeddings'][:, important_indices]
                    selected_embeddings.append(selected)
                    print(f"  {fm}: {len(important_indices)} features selected")
                else:
                    print(f"  Warning: No important features found for {fm}, using all")
                    selected_embeddings.append(fm_data['embeddings'])
                
                if idx == 0:
                    data = fm_data
            
            features = torch.cat(selected_embeddings, dim=1)
            print(f"Total selected features: {features.shape[1]}")
        
        labels = data['celltype_proportions']
        
    else:
        # Load single foundation model features
        pt_filename = get_pt_filename(cell_type, model_name)
        data_path = os.path.join(input_dir, pt_filename)
        
        print(f"\nLoading features: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Feature file not found: {data_path}")
        
        data = torch.load(data_path, weights_only=False)
        features = data['embeddings']
        labels = data['celltype_proportions']
    
    # Get tile and individual IDs
    # sample_ids format: SAMPLEID_BARCODE (e.g., "10xGenomics_06092021_CCACGGAGCCATAAGA-1")
    # We need to extract sample (individual) from the full tile ID
    sample_ids_raw = data.get('sample_ids', [])
    
    tile_ids = []
    individual_ids = []
    sample_ids = []
    
    for sid in sample_ids_raw:
        # Split by last underscore to separate barcode
        # Format: SampleID_Barcode where barcode contains "-1" or similar
        parts = sid.rsplit('_', 1)  # Split from right, only once
        
        if len(parts) == 2 and '-' in parts[1]:
            # parts[0] is the sample ID, parts[1] is the barcode
            sample_id = parts[0]  # e.g., "10xGenomics_06092021"
            tile_id = sid  # Full string as tile ID
        else:
            # Fallback: try to find the barcode pattern
            # Look for the last part that contains a hyphen (barcode pattern)
            all_parts = sid.split('_')
            for i in range(len(all_parts)-1, -1, -1):
                if '-' in all_parts[i]:
                    sample_id = '_'.join(all_parts[:i])
                    tile_id = sid
                    break
            else:
                # No barcode found, use full string
                sample_id = sid
                tile_id = sid
        
        sample_ids.append(sample_id)
        individual_ids.append(sample_id)  # In this dataset, sample = individual
        tile_ids.append(tile_id)
    
    unique_individuals = sorted(list(set(individual_ids)))
    n_individuals = len(unique_individuals)
    n_tiles = len(tile_ids)
    
    print(f"\nDataset info:")
    print(f"  Individuals: {n_individuals}")
    print(f"  Total tiles: {n_tiles}")
    print(f"  Features per tile: {features.shape[1]}")
    
    # Initialize storage
    all_predictions = np.zeros(len(labels))
    all_importances = []
    is_test = np.zeros(len(labels), dtype=bool)
    fold_metrics = []
    
    # Leave-one-individual-out cross-validation
    for ind_idx, ind in enumerate(unique_individuals):
        print(f"\n{'='*70}")
        print(f"Leave-one-out: {ind} ({ind_idx+1}/{n_individuals})")
        print(f"{'='*70}")
        
        # Split train/test
        train_mask = np.array([iid != ind for iid in individual_ids])
        test_mask = np.array([iid == ind for iid in individual_ids])
        
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]
        
        X_train = features[train_indices]
        y_train = labels[train_indices]
        X_test = features[test_indices]
        y_test = labels[test_indices]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Convert to numpy if needed
        if torch.is_tensor(X_train):
            X_train = X_train.numpy()
        if torch.is_tensor(y_train):
            y_train = y_train.numpy()
        if torch.is_tensor(X_test):
            X_test = X_test.numpy()
        if torch.is_tensor(y_test):
            y_test = y_test.numpy()
        
        # Train model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, 'train')],
            verbose_eval=100
        )
        
        # Get feature importance
        importance = model.get_score(importance_type='gain')
        all_importances.append(importance)
        
        # Save model
        if save_files:
            model_dir = os.path.join(output_dir, 'models')
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f'model_leave_{ind}_out.model')
            model.save_model(model_path)
        
        # Predict on test set
        dtest = xgb.DMatrix(X_test)
        predictions = model.predict(dtest)
        
        all_predictions[test_indices] = predictions
        is_test[test_indices] = True
        
        # Calculate metrics for this fold
        fold_mae = mean_absolute_error(y_test, predictions)
        fold_rmse = np.sqrt(mean_squared_error(y_test, predictions))
        fold_spearman, _ = spearmanr(y_test, predictions)
        fold_pearson, _ = pearsonr(y_test, predictions)
        
        fold_metrics.append({
            'Fold': ind_idx + 1,
            'Individual': ind,
            'MAE': fold_mae,
            'RMSE': fold_rmse,
            'Spearman': fold_spearman,
            'Pearson': fold_pearson,
            'n_test_samples': len(test_indices)
        })
        
        print(f"  MAE: {fold_mae:.4f}")
        print(f"  RMSE: {fold_rmse:.4f}")
        print(f"  Spearman: {fold_spearman:.4f}")
        print(f"  Pearson: {fold_pearson:.4f}")
    
    # Calculate overall metrics
    fold_metrics_df = pd.DataFrame(fold_metrics)
    
    average_metrics = {
        'MAE': fold_metrics_df['MAE'].mean(),
        'RMSE': fold_metrics_df['RMSE'].mean(),
        'Spearman': fold_metrics_df['Spearman'].mean(),
        'Pearson': fold_metrics_df['Pearson'].mean(),
        'MAE_std': fold_metrics_df['MAE'].std(),
        'RMSE_std': fold_metrics_df['RMSE'].std(),
        'Spearman_std': fold_metrics_df['Spearman'].std(),
        'Pearson_std': fold_metrics_df['Pearson'].std()
    }
    
    print(f"\n{'='*70}")
    print("Average individual-level metrics:")
    print(f"  MAE: {average_metrics['MAE']:.4f} ± {average_metrics['MAE_std']:.4f}")
    print(f"  RMSE: {average_metrics['RMSE']:.4f} ± {average_metrics['RMSE_std']:.4f}")
    print(f"  Spearman: {average_metrics['Spearman']:.4f} ± {average_metrics['Spearman_std']:.4f}")
    print(f"  Pearson: {average_metrics['Pearson']:.4f} ± {average_metrics['Pearson_std']:.4f}")
    
    # Overall metrics (all test predictions)
    if torch.is_tensor(labels):
        labels_np = labels.numpy()
    else:
        labels_np = labels
    
    overall_mae = mean_absolute_error(labels_np, all_predictions)
    overall_rmse = np.sqrt(mean_squared_error(labels_np, all_predictions))
    overall_spearman, _ = spearmanr(labels_np, all_predictions)
    overall_pearson, _ = pearsonr(labels_np, all_predictions)
    
    print(f"\nOverall metrics (all tiles):")
    print(f"  MAE: {overall_mae:.4f}")
    print(f"  RMSE: {overall_rmse:.4f}")
    print(f"  Spearman: {overall_spearman:.4f}")
    print(f"  Pearson: {overall_pearson:.4f}")
    print(f"{'='*70}\n")
    
    # Save results
    if save_files:
        # Save fold metrics
        metrics_csv_path = os.path.join(output_dir, 'individual_metrics_individual_level.csv')
        fold_metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Saved: {metrics_csv_path}")
        
        # Save tile-level predictions
        tile_predictions_df = pd.DataFrame({
            'tile_id': tile_ids,
            'sample_id': sample_ids,
            'individual_id': individual_ids,
            'deconvoluted_proportion': labels_np,
            'predicted_proportion': all_predictions
        })
        tile_csv_path = os.path.join(output_dir, 'tile_level_predictions.csv')
        tile_predictions_df.to_csv(tile_csv_path, index=False)
        print(f"Saved: {tile_csv_path}")
        
        # Create scatter plot
        plt.figure(figsize=(10, 10))
        plt.scatter(labels_np, all_predictions, alpha=0.5, s=20)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
        
        slope, intercept, r_value, _, _ = stats.linregress(labels_np, all_predictions)
        plt.plot([0, 1], [intercept, intercept + slope], 'r-', alpha=0.7)
        
        text_str = f'MAE: {overall_mae:.4f}\n'
        text_str += f'RMSE: {overall_rmse:.4f}\n'
        text_str += f'Spearman: {overall_spearman:.4f}\n'
        text_str += f'Pearson: {overall_pearson:.4f}\n'
        text_str += f'R²: {r_value**2:.4f}\n'
        text_str += f'Slope: {slope:.4f}\n'
        text_str += f'n: {len(labels_np):,}'
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', bbox=props)
        
        plt.xlabel('True Proportion', fontsize=14, fontweight='bold')
        plt.ylabel('Predicted Proportion', fontsize=14, fontweight='bold')
        plt.title(f'{cell_type} - {model_name}\nIndividual-level Leave-One-Out CV',
                 fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'overall_performance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot_path}")
    
    # Prepare results dictionary
    results = {
        'predictions': all_predictions,
        'labels': labels_np,
        'tile_ids': tile_ids,
        'individual_ids': individual_ids,
        'sample_ids': sample_ids,
        'is_test': is_test,
        'feature_importances': all_importances,
        'params': params,
        'model_name': model_name,
        'cell_type': cell_type,
        'level': 'individual',
        'timestamp': timestamp,
        'output_dir': output_dir if save_files else None,
        'metrics': {
            'overall_mae': overall_mae,
            'overall_rmse': overall_rmse,
            'overall_spearman': overall_spearman,
            'overall_pearson': overall_pearson,
        },
        'individual_metrics': fold_metrics,
        'average_metrics': average_metrics
    }
    
    if save_files and output_dir:
        results_path = os.path.join(output_dir, 'xgboost_results_individual_level.pt')
        torch.save(results, results_path)
        print(f"Saved: {results_path}")
    
    return results


def train_xgboost_for_external_prediction(
    input_dir,
    cell_type,
    model_name='Combined',
    output_dir=None,
    params=None,
    num_boost_round=800,
    seed=42,
    if_combined=True
):
    """
    Train XGBoost model on all available data for external prediction
    No cross-validation, just train on everything and save the final model
    
    This function:
    1. Loads all common intersected samples
    2. Trains a single XGBoost model on ALL data (no train/validation/test split)
    3. Applies calibration using quantile mapping
    4. Saves the model and calibration parameters for external use
    
    Args:
        input_dir: Feature input directory
        cell_type: Cell type for prediction (display name like "Tumor", "pan_APC")
        model_name: Model name for output directory naming
        output_dir: Output directory, if None it will be automatically generated
        params: XGBoost parameters, if None default optimized parameters will be used
        num_boost_round: Number of training rounds
        seed: Random seed
        if_combined: Whether to use combined features from multiple models (should be True)
    
    Returns:
        dict: Training results with model path and calibration info
    """
    
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set output directory
    if output_dir is None:
        output_dir = f'/Users/scui2/ST/BRCA_XGBoost_Results/External_Prediction_Models/{cell_type}_{model_name}_external_prediction'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Training External Prediction Model: {cell_type}")
    print(f"{'='*80}\n")
    
    # Set fixed optimized XGBoost parameters
    if params is None:
        # Use the same optimized parameters as in individual-level training
        params = DEFAULT_PARAMS.copy()
        params['seed'] = seed
        print("✅ Using optimized XGBoost parameters")
    
    # Load feature data
    if if_combined:
        print(f"Loading combined features for {cell_type}...")
        
        # Load important features
        important_features = load_important_features(cell_type)
        
        if important_features is None:
            # Use all features if important features not available
            print("Loading all features from 5 foundation models...")
            models_to_load = ['UNI2h', 'Virchow', 'Virchow2', 'ProvGigapath', 'Conch']
            all_embeddings = []
            
            for idx, fm in enumerate(models_to_load):
                pt_filename = get_pt_filename(cell_type, fm)
                pt_path = os.path.join(input_dir, pt_filename)
                
                if not os.path.exists(pt_path):
                    raise FileNotFoundError(f"Feature file not found: {pt_path}")
                
                fm_data = torch.load(pt_path, weights_only=False)
                all_embeddings.append(fm_data['embeddings'])
                
                if idx == 0:
                    data = fm_data  # Use first model's data for labels, IDs, etc.
            
            features = torch.cat(all_embeddings, dim=1)
            print(f"Total features: {features.shape[1]}")
            
        else:
            # Use selected important features
            print("Loading selected important features from 5 foundation models...")
            models_to_load = ['UNI2h', 'Virchow', 'Virchow2', 'ProvGigapath', 'Conch']
            selected_embeddings = []
            
            for idx, fm in enumerate(models_to_load):
                pt_filename = get_pt_filename(cell_type, fm)
                pt_path = os.path.join(input_dir, pt_filename)
                
                if not os.path.exists(pt_path):
                    raise FileNotFoundError(f"Feature file not found: {pt_path}")
                
                fm_data = torch.load(pt_path, weights_only=False)
                
                # Get important feature indices for this model
                if fm in important_features:
                    important_indices = [int(f.replace('f', '')) for f in important_features[fm]]
                    selected = fm_data['embeddings'][:, important_indices]
                    selected_embeddings.append(selected)
                    print(f"  {fm}: {len(important_indices)} features selected")
                else:
                    print(f"  Warning: No important features found for {fm}, using all")
                    selected_embeddings.append(fm_data['embeddings'])
                
                if idx == 0:
                    data = fm_data
            
            features = torch.cat(selected_embeddings, dim=1)
            print(f"Total selected features: {features.shape[1]}")
        
        labels = data['celltype_proportions']
        
    else:
        # Load single foundation model features
        pt_filename = get_pt_filename(cell_type, model_name)
        data_path = os.path.join(input_dir, pt_filename)
        
        print(f"Loading features: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Feature file not found: {data_path}")
        
        data = torch.load(data_path, weights_only=False)
        features = data['embeddings']
        labels = data['celltype_proportions']
        print(f"Total features: {features.shape[1]}")
    
    # Get IDs
    # Extract tile_ids, individual_ids, and sample_ids from sample_ids_raw
    sample_ids_raw = data.get('sample_ids', [])
    
    tile_ids = []
    individual_ids = []
    sample_ids = []
    
    for sid in sample_ids_raw:
        parts = sid.split('_')
        if len(parts) >= 3:
            # Extract: Study_Date as sample, Study as individual, full string as tile
            sample_id = '_'.join(parts[:2])  # e.g., "10xGenomics_01062023"
            individual_id = parts[0]  # e.g., "10xGenomics"
            tile_id = sid  # Full barcode as tile ID
        else:
            # Fallback
            sample_id = sid
            individual_id = sid
            tile_id = sid
        
        sample_ids.append(sample_id)
        individual_ids.append(individual_id)
        tile_ids.append(tile_id)
    
    print(f"\nTraining data: {features.shape[0]} samples, {features.shape[1]} features")
    print(f"Target: {cell_type} proportions")
    print(f"Unique individuals: {len(set(individual_ids))}")
    print(f"Unique samples: {len(set(sample_ids))}")
    
    # Convert to numpy if needed
    if torch.is_tensor(features):
        features_np = features.numpy()
    else:
        features_np = features
    
    if torch.is_tensor(labels):
        labels_np = labels.numpy()
    else:
        labels_np = labels
    
    # Train XGBoost model on ALL data for final model
    print(f"\n🚀 Training final XGBoost model on all {features.shape[0]} samples...")
    dtrain_full = xgb.DMatrix(features_np, label=labels_np)
    
    # Train final model
    final_model = xgb.train(
        params, 
        dtrain_full, 
        num_boost_round=num_boost_round,
        evals=[(dtrain_full, 'train')],
        verbose_eval=100
    )
    
    # Generate predictions for calibration using Leave-One-Individual-Out CV
    print(f"\n{'='*80}")
    print(f"Generating predictions for calibration using Leave-One-Individual-Out CV")
    print(f"{'='*80}\n")
    
    unique_individuals = sorted(set(individual_ids))
    print(f"Total unique individuals: {len(unique_individuals)}")
    
    # Store LOIO predictions
    loio_predictions = np.zeros(len(labels_np))
    
    for indi in tqdm(unique_individuals, desc="Leave-One-Individual-Out CV"):
        # Get indices for this individual
        test_mask = np.array([ind == indi for ind in individual_ids])
        train_mask = ~test_mask
        
        # Split data
        X_train = features_np[train_mask]
        y_train = labels_np[train_mask]
        X_test = features_np[test_mask]
        
        # Train model without this individual
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        
        model_loio = xgb.train(
            params, 
            dtrain, 
            num_boost_round=num_boost_round,
            verbose_eval=False
        )
        
        # Predict on held-out individual
        preds = model_loio.predict(dtest)
        loio_predictions[test_mask] = preds
    
    print(f"✅ Leave-One-Individual-Out predictions generated for all {len(labels_np)} tiles")
    
    # Calculate raw metrics using LOIO predictions
    raw_mae = mean_absolute_error(labels_np, loio_predictions)
    raw_rmse = np.sqrt(mean_squared_error(labels_np, loio_predictions))
    raw_spearman, _ = spearmanr(labels_np, loio_predictions)
    raw_pearson, _ = pearsonr(labels_np, loio_predictions)
    
    print(f"\nRaw LOIO prediction metrics:")
    print(f"  MAE: {raw_mae:.4f}")
    print(f"  RMSE: {raw_rmse:.4f}")
    print(f"  Spearman: {raw_spearman:.4f}")
    print(f"  Pearson: {raw_pearson:.4f}")
    
    # Perform calibration using LOIO predictions
    print(f"\n{'='*80}")
    print(f"Performing calibration based on LOIO predictions")
    print(f"{'='*80}\n")
    
    calibration_table, calibrated_predictions, labels_processed = calibrate_predictions(
        labels_np, loio_predictions, cell_type
    )
    
    # Calculate calibrated metrics
    cal_mae = mean_absolute_error(labels_np, calibrated_predictions)
    cal_rmse = np.sqrt(mean_squared_error(labels_np, calibrated_predictions))
    cal_spearman, _ = spearmanr(labels_np, calibrated_predictions)
    cal_pearson, _ = pearsonr(labels_np, calibrated_predictions)
    
    print(f"\nCalibrated prediction metrics:")
    print(f"  MAE: {cal_mae:.4f}")
    print(f"  RMSE: {cal_rmse:.4f}")
    print(f"  Spearman: {cal_spearman:.4f}")
    print(f"  Pearson: {cal_pearson:.4f}")
    print(f"  Improvement: MAE Δ={raw_mae-cal_mae:+.4f}, RMSE Δ={raw_rmse-cal_rmse:+.4f}")
    
    # Save the final model (trained on all data)
    model_path = os.path.join(output_dir, f'xgboost_model_{cell_type}_external.model')
    final_model.save_model(model_path)
    print(f"\n✅ Final model saved to: {model_path}")
    
    # Save calibration parameters
    calibration_data = {
        'lookup_table': calibration_table,
        'raw_mae': raw_mae,
        'raw_rmse': raw_rmse,
        'raw_spearman': raw_spearman,
        'raw_pearson': raw_pearson,
        'calibrated_mae': cal_mae,
        'calibrated_rmse': cal_rmse,
        'calibrated_spearman': cal_spearman,
        'calibrated_pearson': cal_pearson,
        'cell_type': cell_type
    }
    
    calib_params_path = os.path.join(output_dir, 'calibration_parameters.pkl')
    with open(calib_params_path, 'wb') as f:
        pickle.dump(calibration_data, f)
    print(f"✅ Calibration parameters saved: {calib_params_path}")
    
    # Save feature information for external prediction
    feature_info = {
        'cell_type': cell_type,
        'model_name': model_name,
        'if_combined': if_combined,
        'total_features': features.shape[1],
        'training_samples': features.shape[0],
        'params': params,
        'num_boost_round': num_boost_round,
        'seed': seed,
        'important_features_file': f"important_features_{cell_type}.pkl" if if_combined else None
    }
    
    # Save feature info
    info_path = os.path.join(output_dir, 'model_info.pt')
    torch.save(feature_info, info_path)
    print(f"✅ Model info saved to: {info_path}")
    
    # Create and save calibration comparison plot (using LOIO predictions)
    print(f"\n📊 Creating calibration comparison plot...")
    calibration_plot_path = os.path.join(output_dir, 'calibration_comparison.png')
    plot_calibration_scatter(
        y_true=labels_np,
        y_pred_raw=loio_predictions,  # Use LOIO predictions instead of training predictions
        y_pred_calibrated=calibrated_predictions,
        y_true_processed=labels_processed,
        calibration_table=calibration_table,
        dummy=None,
        output_path=calibration_plot_path,
        model_name=model_name,
        cell_type=cell_type
    )
    
    # Save tile-level predictions CSV
    print(f"\n💾 Saving tile-level predictions CSV...")
    tile_predictions_df = pd.DataFrame({
        'tile_id': tile_ids,
        'sample_id': sample_ids,
        'individual_id': individual_ids,
        'deconvoluted_proportion': labels_np,
        'predicted_proportion_raw': loio_predictions,  # Use LOIO predictions
        'predicted_proportion_calibrated': calibrated_predictions
    })
    
    tile_predictions_csv_path = os.path.join(output_dir, 'tile_level_predictions_detailed.csv')
    tile_predictions_df.to_csv(tile_predictions_csv_path, index=False)
    print(f"✅ Tile-level predictions saved to: {tile_predictions_csv_path}")
    print(f"  Total tiles: {len(tile_predictions_df):,}")
    
    # Create scatter plot: y_true vs y_calibrated
    print(f"\n📈 Creating y_true vs y_calibrated scatter plot...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Scatter plot
    ax.scatter(labels_np, calibrated_predictions, alpha=0.5, s=15, color='blue', edgecolor='none')
    
    # Identity line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Identity (y=x)')
    
    # Fit line
    slope, intercept, r_value, p_value, std_err = stats.linregress(labels_np, calibrated_predictions)
    x_line = np.array([0, 1])
    ax.plot(x_line, intercept + slope * x_line, 'r-', linewidth=2, alpha=0.7, label='Fit line')
    
    # Add text box with metrics
    textstr = f'MAE: {cal_mae:.4f}\n'
    textstr += f'RMSE: {cal_rmse:.4f}\n'
    textstr += f'Spearman: {cal_spearman:.4f}\n'
    textstr += f'Pearson: {cal_pearson:.4f}\n'
    textstr += f'Slope: {slope:.4f}\n'
    textstr += f'R²: {r_value**2:.4f}\n'
    textstr += f'n: {len(labels_np):,}'
    
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Labels and styling
    ax.set_xlabel('True Cell Type Proportion (y_true)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Calibrated Predicted Proportion (y_calibrated)', fontsize=14, fontweight='bold')
    ax.set_title(f'External Prediction Model - Calibrated\n{cell_type} - {model_name}', 
                 fontsize=16, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='lower right')
    ax.tick_params(labelsize=12, width=2)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    plt.tight_layout()
    
    output_scatter = os.path.join(output_dir, 'y_true_vs_y_calibrated_scatter.png')
    plt.savefig(output_scatter, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Scatter plot saved: {output_scatter}")
    
    # Return results
    results = {
        'model_path': model_path,
        'calibration_path': calib_params_path,
        'info_path': info_path,
        'output_dir': output_dir,
        'feature_info': feature_info,
        'calibration_data': calibration_data,
        'raw_metrics': {
            'mae': raw_mae,
            'rmse': raw_rmse,
            'spearman': raw_spearman,
            'pearson': raw_pearson
        },
        'calibrated_metrics': {
            'mae': cal_mae,
            'rmse': cal_rmse,
            'spearman': cal_spearman,
            'pearson': cal_pearson
        },
        'training_completed': True
    }
    
    print(f"\n🎉 External prediction model training completed!")
    print(f"📁 Output directory: {output_dir}")
    print(f"🤖 Model file: {model_path}")
    print(f"🔧 Calibration file: {calib_params_path}")
    print(f"📋 Info file: {info_path}")
    print(f"{'='*80}\n")
    
    return results


# =============================================================================
# Main Training Script
# =============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("STPath-BRCA: XGBoost External Prediction Model Training")
    print("="*80)
    print(f"\nCell Types: {CELL_TYPES}")
    print(f"Model: Combined (5 Foundation Models with Important Features)")
    print(f"\nInput Directory: {INPUT_DIR}")
    print("="*80 + "\n")
    
    # =================================================================
    # Train External Prediction Models (Combined, All Data, with Calibration)
    # =================================================================
    
    print("\n" + "#"*80)
    print("Training External Prediction Models")
    print("Using ALL common intersected samples (no CV)")
    print("With quantile mapping calibration")
    print("#"*80 + "\n")
    
    external_results = []
    
    for cell_type in CELL_TYPES:
        print(f"\n{'*'*80}")
        print(f"Cell Type: {cell_type}")
        print(f"{'*'*80}\n")
        
        try:
            results = train_xgboost_for_external_prediction(
                input_dir=INPUT_DIR,
                cell_type=cell_type,
                model_name='Combined',
                output_dir=None,  # Will use default path
                params=None,  # Use default optimized params
                num_boost_round=NUM_BOOST_ROUND,
                seed=42,
                if_combined=True
            )
            
            external_results.append({
                'cell_type': cell_type,
                'model_name': 'Combined',
                'raw_MAE': results['raw_metrics']['mae'],
                'raw_RMSE': results['raw_metrics']['rmse'],
                'raw_Spearman': results['raw_metrics']['spearman'],
                'raw_Pearson': results['raw_metrics']['pearson'],
                'calibrated_MAE': results['calibrated_metrics']['mae'],
                'calibrated_RMSE': results['calibrated_metrics']['rmse'],
                'calibrated_Spearman': results['calibrated_metrics']['spearman'],
                'calibrated_Pearson': results['calibrated_metrics']['pearson'],
                'MAE_improvement': results['raw_metrics']['mae'] - results['calibrated_metrics']['mae'],
                'model_path': results['model_path']
            })
            
            print(f"\n✓ Completed: {cell_type}")
            print(f"  Raw MAE: {results['raw_metrics']['mae']:.4f}")
            print(f"  Calibrated MAE: {results['calibrated_metrics']['mae']:.4f}")
            print(f"  Improvement: {results['raw_metrics']['mae'] - results['calibrated_metrics']['mae']:.4f}")
            
        except Exception as e:
            print(f"\n✗ Error training {cell_type}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # =================================================================
    # Summary
    # =================================================================
    
    print("\n\n" + "="*80)
    print("EXTERNAL PREDICTION MODEL TRAINING SUMMARY")
    print("="*80 + "\n")
    
    if len(external_results) > 0:
        summary_df = pd.DataFrame(external_results)
        print(summary_df.to_string(index=False))
        
        # Save summary
        summary_path = '/ST/BRCA_XGBoost_Results/External_Prediction_Models/external_training_summary.csv'
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")
    else:
        print("No models were successfully trained.")
    
    print("\n" + "="*80)
    print("✓ ALL EXTERNAL PREDICTION MODEL TRAINING COMPLETED!")
    print("="*80 + "\n")
