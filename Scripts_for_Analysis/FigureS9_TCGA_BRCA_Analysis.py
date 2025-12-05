"""
FigureS9 TCGA BRCA Analysis Pipeline
=====================================
Process TCGA BRCA whole slide images to predict cell type proportions and classify tiles

Pipeline:
1. WSI Processing: Cut images into tiles, extract features, predict proportions
2. Classification: Classify tiles using quantile-based thresholds

Input: .tif whole slide image path
Output: 
1. CSV file with tile-level predictions (8 cell types)
2. Hexagonal heatmap for Cancer.Epithelial visualization
3. Classification CSV
4. Hexagonal visualization of tile classifications

BRCA Cell Types (8):
- Cancer.Epithelial
- Other.Epithelial
- CAFs
- Myeloid
- Plasma
- B.cell
- T.cell
- Endothelial

Author: Saishi Cui
Date: December 2025
"""

import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))

import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
from timm.layers import SwiGLUPacked
from conch.open_clip_custom import create_model_from_pretrained
from huggingface_hub import login
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import shutil
import pickle
import glob


def login_to_huggingface(hf_token):
    """Login to HuggingFace once before processing multiple WSIs"""
    print(f"\n{'='*80}")
    print("Logging into HuggingFace...")
    print(f"{'='*80}\n")
    login(token=hf_token)
    print("HuggingFace login successful!\n")


def load_foundation_models(device=None):
    """Load all 5 foundation models once before processing multiple WSIs"""
    print(f"\n{'='*80}")
    print("Loading Foundation Models")
    print(f"{'='*80}\n")
    
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Apple Silicon)")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    
    models = {}
    
    standard_transform = transforms.Compose([
        transforms.Resize(224), 
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    ### Load UNI2h
    print("  Loading UNI2h...")
    timm_kwargs = {
        'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
        'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667*2,
        'num_classes': 0, 'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU,
        'reg_tokens': 8, 'dynamic_img_size': True
    }
    model_UNI2h = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    model_UNI2h.eval()
    model_UNI2h = model_UNI2h.to(device)
    models['UNI2h'] = {'model': model_UNI2h, 'transform': standard_transform}
    
    ### Load Virchow
    print("  Loading Virchow...")
    model_Virchow = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, 
                                     mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model_Virchow.eval()
    model_Virchow = model_Virchow.to(device)
    models['Virchow'] = {'model': model_Virchow, 'transform': standard_transform}
    
    ### Load Virchow2
    print("  Loading Virchow2...")
    model_Virchow2 = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True,
                                      mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model_Virchow2.eval()
    model_Virchow2 = model_Virchow2.to(device)
    models['Virchow2'] = {'model': model_Virchow2, 'transform': standard_transform}
    
    ### Load ProvGigapath
    print("  Loading ProvGigapath...")
    model_ProvGigapath = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    model_ProvGigapath.eval()
    model_ProvGigapath = model_ProvGigapath.to(device)
    models['ProvGigapath'] = {'model': model_ProvGigapath, 'transform': standard_transform}
    
    ### Load Conch
    print("  Loading Conch...")
    model_Conch, transform_Conch = create_model_from_pretrained(
        'conch_ViT-B-16', "hf_hub:MahmoodLab/conch"
    )
    model_Conch.eval()
    model_Conch = model_Conch.to(device)
    models['Conch'] = {'model': model_Conch, 'transform': transform_Conch}
    
    print(f"\nAll models loaded and ready on {device}!\n")
    return models


class TileDataset(Dataset):
    """Dataset class for loading tiles"""
    def __init__(self, tile_paths, transform=None):
        self.tile_paths = tile_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.tile_paths)
    
    def __getitem__(self, idx):
        tile_path = self.tile_paths[idx]
        tile_name = os.path.basename(tile_path).replace('.tif', '')
        image = Image.open(tile_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, tile_name


def _extract_for_one_model(model_name, model, transform, device, tiles_folder, batch_size=32):
    """Extract features for a single foundation model"""
    all_tiles = [f for f in os.listdir(tiles_folder) if f.endswith('.tif')]
    valid_tiles = [f for f in all_tiles if '_White' not in f]
    valid_tile_paths = [os.path.join(tiles_folder, f) for f in valid_tiles]
    
    if len(valid_tile_paths) == 0:
        print(f"  No valid tiles found for {model_name}")
        return {}
    
    dataset = TileDataset(valid_tile_paths, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    embeddings_list = []
    tile_names_list = []
    
    with torch.no_grad():
        for images, tile_names in tqdm(loader, desc=f"  Extracting {model_name} features"):
            images = images.to(device)
            
            if model_name == 'Conch':
                embeddings = model.encode_image(images, proj_contrast=False, normalize=False)
            elif model_name in ['Virchow', 'Virchow2']:
                embeddings = model(images)
                embeddings = torch.cat([embeddings[:,0], embeddings[:,5:].mean(1)], dim=-1)
            else:
                embeddings = model(images)
            
            embeddings_list.append(embeddings.cpu())
            tile_names_list.extend(tile_names)
    
    all_embeddings = torch.cat(embeddings_list, 0)
    
    features_dict = {}
    for i, tile_name in enumerate(tile_names_list):
        features_dict[tile_name] = all_embeddings[i].tolist()
    
    print(f"  {model_name}: Extracted features for {len(features_dict)} tiles")
    return features_dict


def _predict_cell_type_proportions(final_features, models_dir, important_features_excel, calibration_dir, 
                                   if_calibration=True, normalize=False):
    """Predict cell type proportions using trained XGBoost models with optional calibration"""
    
    ### BRCA has 8 cell types
    cell_types = [
        "Cancer.Epithelial",
        "Other.Epithelial",
        "CAFs",
        "Myeloid",
        "Plasma",
        "B.cell",
        "T.cell",
        "Endothelial"
    ]
    
    ### Load important features from Excel
    print("Loading important features from Excel...")
    important_features_dict = {}
    
    for cell_type in cell_types:
        sheet_name = cell_type
        print(f"  Loading sheet: {sheet_name}")
        
        try:
            df = pd.read_excel(important_features_excel, sheet_name=sheet_name)
        except Exception as e:
            print(f"    Warning: Could not load sheet '{sheet_name}': {e}")
            continue
        
        important_features_dict[cell_type] = {}
        model_order = ['UNI2h', 'Virchow', 'Virchow2', 'ProvGigapath', 'Conch']
        
        for model_name in model_order:
            feature_col = f'{model_name}_Features'
            if feature_col in df.columns:
                features_list = df[feature_col].dropna().tolist()
                indices = []
                for f in features_list:
                    f_str = str(f)
                    if f_str.startswith('f'):
                        try:
                            indices.append(int(f_str.replace('f', '')))
                        except ValueError:
                            continue
                important_features_dict[cell_type][model_name] = indices
            else:
                important_features_dict[cell_type][model_name] = []
    
    ### Load XGBoost models
    print("\nLoading XGBoost models...")
    models = {}
    for cell_type in cell_types:
        model_filename = f"xgboost_model_{cell_type}_Combined_external_prediction.model"
        model_path = os.path.join(models_dir, model_filename)
        
        if os.path.exists(model_path):
            models[cell_type] = xgb.Booster()
            models[cell_type].load_model(model_path)
            print(f"  Loaded: {cell_type}")
        else:
            print(f"  Warning: Model not found for {cell_type}: {model_path}")
    
    ### Load calibration parameters
    calibration_params = {}
    if if_calibration:
        print("\nLoading calibration parameters...")
        for cell_type in cell_types:
            cell_type_clean = cell_type.replace(".", "_")
            pkl_filename = f"{cell_type_clean}_calibration_parameters.pkl"
            pkl_path = os.path.join(calibration_dir, pkl_filename)
            
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    calibration_params[cell_type] = pickle.load(f)
                print(f"  Loaded calibration: {cell_type}")
            else:
                print(f"  Warning: Calibration file not found for {cell_type}")
    
    ### Prepare results
    results = []
    tile_names = sorted(final_features.keys(), key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))
    
    print(f"\nMaking predictions for {len(tile_names)} tiles...")
    
    for tile_name in tqdm(tile_names, desc="Predicting"):
        tile_features = final_features[tile_name]
        is_white = tile_features is None
        
        if is_white:
            result_row = {'tile_name': f"{tile_name}_White"}
            for cell_type in cell_types:
                result_row[cell_type] = np.nan
        else:
            if any(f is None for f in tile_features):
                result_row = {'tile_name': f"{tile_name}_White"}
                for cell_type in cell_types:
                    result_row[cell_type] = np.nan
            else:
                result_row = {'tile_name': tile_name}
                predictions = {}
                
                for cell_type in cell_types:
                    if cell_type not in models or cell_type not in important_features_dict:
                        predictions[cell_type] = np.nan
                        continue
                    
                    model_order = ['UNI2h', 'Virchow', 'Virchow2', 'ProvGigapath', 'Conch']
                    selected_features = []
                    
                    for i, model_name in enumerate(model_order):
                        embedding = np.array(tile_features[i])
                        indices = important_features_dict[cell_type][model_name]
                        
                        if len(indices) > 0:
                            selected = embedding[indices]
                            selected_features.append(selected)
                    
                    if len(selected_features) > 0:
                        combined_features = np.concatenate(selected_features)
                        dmatrix = xgb.DMatrix(combined_features.reshape(1, -1))
                        raw_prediction = models[cell_type].predict(dmatrix)[0]
                        
                        if if_calibration and cell_type in calibration_params and 'lookup_table' in calibration_params[cell_type]:
                            lookup = calibration_params[cell_type]['lookup_table']
                            y_pred_quantiles = lookup['y_pred_quantiles']
                            y_true_quantiles = lookup['y_true_quantiles']
                            calibrated_prediction = np.interp(raw_prediction, y_pred_quantiles, y_true_quantiles)
                            calibrated_prediction = np.clip(calibrated_prediction, 0, 1)
                            predictions[cell_type] = calibrated_prediction
                        else:
                            predictions[cell_type] = raw_prediction
                    else:
                        predictions[cell_type] = np.nan
                
                for cell_type in cell_types:
                    result_row[cell_type] = predictions.get(cell_type, np.nan)
        
        results.append(result_row)
    
    df_results = pd.DataFrame(results)
    
    if normalize:
        print(f"\nNormalizing predictions...")
        for idx, row in df_results.iterrows():
            if not pd.isna(row[cell_types[0]]):
                total = sum(row[cell_type] for cell_type in cell_types)
                if total > 0:
                    for cell_type in cell_types:
                        df_results.at[idx, cell_type] = row[cell_type] / total
    
    valid_tiles = df_results[cell_types[0]].notna().sum()
    white_tiles = df_results[cell_types[0]].isna().sum()
    
    print(f"\nPrediction completed!")
    print(f"   Total tiles: {len(df_results)}")
    print(f"   Valid predictions: {valid_tiles}")
    print(f"   White tiles (NaN): {white_tiles}")
    
    return df_results


def _visualize_hexagonal_heatmap(predictions_df, cell_type, output_path):
    """Visualize cell type proportion predictions as a hexagonal heatmap"""
    
    print(f"\nCreating hexagonal heatmap for: {cell_type}")
    
    data_list = []
    for idx, row in predictions_df.iterrows():
        tile_name = row['tile_name']
        
        try:
            parts = tile_name.split('-')
            tile_row = int(parts[0])
            tile_col = int(parts[1])
        except (ValueError, IndexError):
            continue
        
        proportion = row[cell_type]
        
        data_list.append({
            'tile_name': tile_name,
            'row': tile_row,
            'col': tile_col,
            'x': tile_col,
            'y': tile_row,
            'proportion': proportion
        })
    
    df = pd.DataFrame(data_list)
    valid_df = df[df['proportion'].notna()].copy()
    
    if len(valid_df) == 0:
        print("No valid tiles to visualize")
        return
    
    valid_df['proportion_pct'] = valid_df['proportion'] * 100
    max_proportion_pct = valid_df['proportion_pct'].max()
    scale_ceiling = np.ceil(max_proportion_pct / 10) * 10
    
    print(f"  Total tiles: {len(df)}")
    print(f"  Valid tiles: {len(valid_df)}")
    print(f"  Proportion range: {valid_df['proportion_pct'].min():.2f}% - {valid_df['proportion_pct'].max():.2f}%")
    
    max_row = df['row'].max()
    max_col = df['col'].max()
    estimated_grid_size = max(max_row, max_col)
    
    base_size = 150
    size_factor = (40 / estimated_grid_size) ** 1.5
    marker_size = max(20, int(base_size * size_factor))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(
        x=valid_df['x'],
        y=valid_df['y'],
        c=valid_df['proportion_pct'],
        cmap='coolwarm',
        marker='h',
        s=marker_size,
        edgecolors='black',
        linewidth=0.5,
        vmin=0,
        vmax=scale_ceiling
    )
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.ax.tick_params(labelsize=20, colors='black', width=3)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_color('black')
    
    ax.invert_yaxis()
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(5)
        spine.set_color('black')
    
    plt.title('')
    plt.xlabel('')
    plt.ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.05)
    print(f"\nHeatmap saved to: {output_path}")
    plt.close()


def process_wsi_for_cell_type_prediction(
    wsi_path,
    models_dict,
    models_dir,
    important_features_excel,
    calibration_dir,
    output_predictions_dir,
    output_heatmaps_dir,
    temp_tiles_folder="/tmp/wsi_tiles",
    white_threshold=220,
    cut_size=70
):
    """Complete pipeline: Cut WSI -> Extract features -> Predict -> Visualize"""
    
    print(f"\n{'='*80}")
    print(f"Processing WSI: {os.path.basename(wsi_path)}")
    print(f"{'='*80}\n")
    
    wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
    
    os.makedirs(output_predictions_dir, exist_ok=True)
    os.makedirs(output_heatmaps_dir, exist_ok=True)
    os.makedirs(temp_tiles_folder, exist_ok=True)
    
    ### Step 1: Cut image into tiles
    print(f"Step 1: Cutting WSI into tiles")
    
    Image.MAX_IMAGE_PIXELS = None
    img_pil = Image.open(wsi_path)
    img_width, img_height = img_pil.size
    print(f"Image dimensions: {img_width} x {img_height}")
    
    if img_width <= img_height:
        grid_cols = cut_size
        aspect_ratio = img_height / img_width
        grid_rows = int(np.round(cut_size * aspect_ratio))
    else:
        grid_rows = cut_size
        aspect_ratio = img_width / img_height
        grid_cols = int(np.round(cut_size * aspect_ratio))
    
    total_tiles = grid_rows * grid_cols
    print(f"Grid: {grid_rows} rows x {grid_cols} cols = {total_tiles} tiles")
    
    tile_height = img_height // grid_rows
    tile_width = img_width // grid_cols
    
    valid_count = 0
    white_count = 0
    
    print("Creating tiles...")
    for row in tqdm(range(grid_rows), desc="Processing rows"):
        for col in range(grid_cols):
            y_start = row * tile_height
            y_end = (row + 1) * tile_height if row < grid_rows - 1 else img_height
            x_start = col * tile_width
            x_end = (col + 1) * tile_width if col < grid_cols - 1 else img_width
            
            tile_pil = img_pil.crop((x_start, y_start, x_end, y_end))
            tile_np = np.array(tile_pil)
            if len(tile_np.shape) == 3:
                gray = np.mean(tile_np, axis=2)
            else:
                gray = tile_np
            mean_intensity = np.mean(gray)
            
            tile_name = f"{row+1}-{col+1}"
            if mean_intensity > white_threshold:
                tile_name += "_White"
                white_count += 1
            else:
                valid_count += 1
            
            output_path = os.path.join(temp_tiles_folder, f"{tile_name}.tif")
            tile_pil.save(output_path)
    
    img_pil.close()
    
    print(f"\nTile creation completed! Valid: {valid_count}, White: {white_count}\n")
    
    ### Step 2: Extract features
    print(f"Step 2: Extracting features from tiles")
    
    first_model = list(models_dict.values())[0]['model']
    device = next(first_model.parameters()).device
    
    model_order = ['UNI2h', 'Virchow', 'Virchow2', 'ProvGigapath', 'Conch']
    all_features = {}
    
    for model_name in model_order:
        print(f"\nExtracting features using {model_name}...")
        model = models_dict[model_name]['model']
        transform = models_dict[model_name]['transform']
        features = _extract_for_one_model(model_name, model, transform, device, temp_tiles_folder)
        all_features[model_name] = features
    
    print("\nOrganizing features by tile position...")
    final_features = {}
    
    for row in range(1, grid_rows + 1):
        for col in range(1, grid_cols + 1):
            tile_name = f"{row}-{col}"
            tile_name_white = f"{row}-{col}_White"
            
            white_tile_path = os.path.join(temp_tiles_folder, f"{tile_name_white}.tif")
            if os.path.exists(white_tile_path):
                final_features[tile_name] = None
            else:
                features_list = []
                for model_name in model_order:
                    if tile_name in all_features[model_name]:
                        features_list.append(all_features[model_name][tile_name])
                    else:
                        features_list.append(None)
                final_features[tile_name] = features_list
    
    ### Step 3: Predict
    print(f"\nStep 3: Predicting cell type proportions")
    
    predictions_df = _predict_cell_type_proportions(
        final_features, models_dir, important_features_excel, calibration_dir, 
        if_calibration=True, normalize=False
    )
    
    predictions_csv_path = os.path.join(output_predictions_dir, f"{wsi_name}_predictions.csv")
    predictions_df.to_csv(predictions_csv_path, index=False)
    print(f"\nPredictions saved to: {predictions_csv_path}")
    
    ### Step 4: Visualize Cancer.Epithelial
    print(f"\nStep 4: Creating Cancer.Epithelial hexagonal heatmap")
    
    heatmap_path = os.path.join(output_heatmaps_dir, f"{wsi_name}_CancerEpithelial_heatmap.png")
    _visualize_hexagonal_heatmap(predictions_df, "Cancer.Epithelial", heatmap_path)
    
    ### Clean up
    if os.path.exists(temp_tiles_folder):
        shutil.rmtree(temp_tiles_folder)
        print(f"\nTemporary tiles deleted")
    
    summary = {
        'wsi_name': wsi_name,
        'total_tiles': total_tiles,
        'valid_tiles': valid_count,
        'white_tiles': white_count,
        'grid_rows': grid_rows,
        'grid_cols': grid_cols,
        'predictions_csv': predictions_csv_path,
        'heatmap_png': heatmap_path
    }
    
    print(f"\nWSI Processing Completed: {wsi_name}")
    return summary


### Example usage
if __name__ == "__main__":
    
    ### Configuration for BRCA
    HF_TOKEN = "YOUR_HUGGINGFACE_TOKEN_HERE"
    MODELS_DIR = "/Users/scui2/Desktop/STPath_BRCA/STPath_Software/BRCA/XGBoost_Models"
    IMPORTANT_FEATURES_EXCEL = "/Users/scui2/Desktop/STPath_BRCA/STPath_Software/BRCA/XGBoost_Models/Important_Features_BRCA.xlsx"
    CALIBRATION_DIR = "/Users/scui2/Desktop/STPath_BRCA/STPath_Software/BRCA/XGBoost_Models"
    OUTPUT_PREDICTIONS_DIR = "/Users/scui2/Desktop/TCGA/TCGA_BRCA/TCGA_BRCA_Predictions_CSV"
    OUTPUT_HEATMAPS_DIR = "/Users/scui2/Desktop/TCGA/TCGA_BRCA/TCGA_CancerEpithelial_Heatmaps"
    TEMP_TILES_FOLDER = "/Users/scui2/Desktop/TCGA/TCGA_BRCA/TCGA_BRCA_Tiles"
    
    ### Step 1: Login to HuggingFace
    login_to_huggingface(HF_TOKEN)
    
    ### Step 2: Load foundation models
    models_dict = load_foundation_models()
    
    ### Step 3: Process all WSIs
    wsi_folder = "/Users/scui2/Desktop/TCGA/TCGA_BRCA/TCGA_BRCA_HE_filtered_tiff"
    all_wsi_files = sorted(glob.glob(os.path.join(wsi_folder, "*.tiff")))
    
    processed_csv_files = glob.glob(os.path.join(OUTPUT_PREDICTIONS_DIR, "*_predictions.csv"))
    processed_names = set([os.path.basename(f).replace('_predictions.csv', '') for f in processed_csv_files])
    
    unprocessed_wsi_files = []
    for wsi_path in all_wsi_files:
        wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
        if wsi_name not in processed_names:
            unprocessed_wsi_files.append(wsi_path)
    
    print(f"\nTotal WSI files: {len(all_wsi_files)}")
    print(f"Already processed: {len(processed_names)}")
    print(f"Remaining to process: {len(unprocessed_wsi_files)}\n")
    
    if len(unprocessed_wsi_files) == 0:
        print("All WSI files have been processed!")
    else:
        for i, wsi_path in enumerate(unprocessed_wsi_files, 1):
            print(f"\nProcessing WSI {i}/{len(unprocessed_wsi_files)}: {os.path.basename(wsi_path)}")
            
            try:
                summary = process_wsi_for_cell_type_prediction(
                    wsi_path=wsi_path,
                    models_dict=models_dict,
                    models_dir=MODELS_DIR,
                    important_features_excel=IMPORTANT_FEATURES_EXCEL,
                    calibration_dir=CALIBRATION_DIR,
                    output_predictions_dir=OUTPUT_PREDICTIONS_DIR,
                    output_heatmaps_dir=OUTPUT_HEATMAPS_DIR,
                    temp_tiles_folder=TEMP_TILES_FOLDER,
                    white_threshold=220,
                    cut_size=70
                )
                print(f"\nSuccessfully processed {os.path.basename(wsi_path)}")
            except Exception as e:
                print(f"\nError processing {os.path.basename(wsi_path)}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\nFinished processing {len(unprocessed_wsi_files)} remaining WSI files!")

