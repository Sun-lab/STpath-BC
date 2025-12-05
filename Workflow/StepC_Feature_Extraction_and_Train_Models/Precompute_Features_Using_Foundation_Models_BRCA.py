"""
STPath-BRCA: Feature Extraction using Foundation Models for Breast Cancer
============================================================================

This script extracts features from histopathology patches using multiple foundation models
including Conch, ProvGigapath, UNI2h, Virchow, and Virchow2 for breast cancer analysis.

Author: Saishi Cui
Date: December 2025

Purpose: Extract and precompute features from breast cancer H&E patches using various
foundation models to enable downstream cell type proportion prediction.
"""

import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from huggingface_hub import login
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
import os
from torchvision import models
from pathlib import Path
from timm.layers import SwiGLUPacked
import re
from conch.open_clip_custom import create_model_from_pretrained


### Utility Functions
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def load_Conch_model():
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token)
    model, transform = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token=hf_token)
    return model, transform


def load_UNI2h_model():
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token) 
    timm_kwargs = {
    'img_size': 224, 
    'patch_size': 14, 
    'depth': 24,
    'num_heads': 24,
    'init_values': 1e-5, 
    'embed_dim': 1536,
    'mlp_ratio': 2.66667*2,
    'num_classes': 0, 
    'no_embed_class': True,
    'mlp_layer': timm.layers.SwiGLUPacked, 
    'act_layer': torch.nn.SiLU, 
    'reg_tokens': 8, 
    'dynamic_img_size': True
    }
    model_UNI2h = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    return model_UNI2h


def load_ProvGigapath_model():
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token)
    tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    return tile_encoder


def load_Virchow_model():
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token)
    model_Virchow = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model_Virchow = model_Virchow.eval()
    return model_Virchow


def load_Virchow2_model():
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token)
    model_Virchow2 = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model_Virchow2 = model_Virchow2.eval()
    return model_Virchow2


### Dataset Class for BRCA
class ImageDataset_training(Dataset):
    def __init__(self, img_dir, csv_file, celltype, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.metadata_df = pd.read_csv(csv_file, index_col=0)
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
        self.img_to_label = dict(zip(self.metadata_df.index, self.metadata_df[celltype]))

    def extract_identifier_individual(self, text):
        pattern = r'.*?-1_(.*?)(?:_region\d+)?$'
        match = re.match(pattern, text)
        if match:
            extracted = match.group(1)
            if re.match(r'^\d', extracted):
                return extracted.split('_')[0]
            else:
                return extracted
        return None

    def extract_identifier_sample(self, text):
        pattern = r'.*?-1_(.*?)(?:_region\d+)?$'
        match = re.match(pattern, text)
        if match:
            extracted = match.group(1)        
            return extracted
        return None

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        tile_file = self.image_files[idx]
        tile_id = tile_file.split('.')[0]
        sample_id = self.extract_identifier_sample(tile_id)
        individual_id = self.extract_identifier_individual(tile_id)
        label = self.img_to_label.get(tile_id, 0.0)
        img_path = os.path.join(self.img_dir, tile_file)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32), tile_id, sample_id, individual_id


### Feature Extraction Functions
def create_training_features_Conch(cell_types_list):
    batch_size = 32
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log("use CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log("use MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        log("use CPU")
    
    model, transform = load_Conch_model()
    model = model.to(device)
    model.eval()

    for cell_type in cell_types_list:
        print(cell_type)
        img_dir = f"/Users/scui2/Desktop/Breast_Cancer_HE_patches/Training_tiles/{cell_type}"
        csv_file = f"/Users/scui2/Desktop/Breast_Cancer_HE_patches/Training_celltype_proportion/{cell_type}_celltype_proportion_combined.csv"
        output_file = f"/Users/scui2/Desktop/Breast_Cancer_HE_patches/Training_features/{cell_type}_training_precomputed_features_Conch.pt"
        if len(os.listdir(img_dir)) == 0:
            log(f"the image directory for {cell_type} is empty, skip the precomputation")
            continue

        log("create the image dataset...")
        dataset = ImageDataset_training(img_dir, csv_file, cell_type, transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        log(f"start to precompute the features for {len(dataset)} images...")
        embeddings_list = []
        celltype_proportions_list = []
        tile_ids_list = []
        sample_ids_list = []
        individual_ids_list = []
        with torch.no_grad():
            for images, celltype_prop, tile_id, sample_id, individual_id in tqdm(loader, desc="calculate the features"):
                images = images.to(device)  
                embeddings = model.encode_image(images, proj_contrast=False, normalize=False)
                embeddings_list.append(embeddings.cpu())
                celltype_proportions_list.append(celltype_prop)
                tile_ids_list.extend(tile_id)
                sample_ids_list.extend(sample_id)
                individual_ids_list.extend(individual_id)

        embeddings = torch.cat(embeddings_list, 0)
        celltype_proportions = torch.cat(celltype_proportions_list, 0)
        log(f"the features are calculated, the number of features: {len(embeddings)}")
        
        log(f"save the features to: {output_file}")
        torch.save({
            'embeddings': embeddings,
            'celltype_proportions': celltype_proportions,
            'tile_ids': tile_ids_list,
            'sample_ids': sample_ids_list,
            'individual_ids': individual_ids_list,
        }, output_file)
        log("the precomputation is completed, the features are saved")


def create_training_features_UNI2h(cell_types_list):
    batch_size = 32
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log("use CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log("use MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        log("use CPU")
    
    tile_encoder = load_UNI2h_model()
    tile_encoder = tile_encoder.to(device)
    tile_encoder.eval()
    
    transform = transforms.Compose([
    transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])

    for cell_type in cell_types_list:
        print(cell_type)
        img_dir = f"/Users/scui2/Desktop/Breast_Cancer_HE_patches/Training_tiles/{cell_type}"
        csv_file = f"/Users/scui2/Desktop/Breast_Cancer_HE_patches/Training_celltype_proportion/{cell_type}_celltype_proportion_combined.csv"
        output_file = f"/Users/scui2/Desktop/Breast_Cancer_HE_patches/Training_features/{cell_type}_training_precomputed_features_UNI2h.pt"
        if len(os.listdir(img_dir)) == 0:
            log(f"the image directory for {cell_type} is empty, skip the precomputation")
            continue

        log("create the image dataset...")
        dataset = ImageDataset_training(img_dir, csv_file, cell_type, transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        log(f"start to precompute the features for {len(dataset)} images...")
        embeddings_list = []
        celltype_proportions_list = []
        tile_ids_list = []
        sample_ids_list = []
        individual_ids_list = []
        with torch.no_grad():
            for images, celltype_prop, tile_id, sample_id, individual_id in tqdm(loader, desc="calculate the features"):
                images = images.to(device)
                embeddings = tile_encoder(images)
                embeddings_list.append(embeddings.cpu())
                celltype_proportions_list.append(celltype_prop)
                tile_ids_list.extend(tile_id)
                sample_ids_list.extend(sample_id)
                individual_ids_list.extend(individual_id)

        embeddings = torch.cat(embeddings_list, 0)
        celltype_proportions = torch.cat(celltype_proportions_list, 0)
        log(f"the features are calculated, the number of features: {len(embeddings)}")
        
        log(f"save the features to: {output_file}")
        torch.save({
            'embeddings': embeddings,
            'celltype_proportions': celltype_proportions,
            'tile_ids': tile_ids_list,
            'sample_ids': sample_ids_list,
            'individual_ids': individual_ids_list,
        }, output_file)
        log("the precomputation is completed, the features are saved")


def create_training_features_ProvGigapath(cell_types_list):
    batch_size = 32
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log("use CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log("use MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        log("use CPU")
    
    tile_encoder = load_ProvGigapath_model()
    tile_encoder = tile_encoder.to(device)
    tile_encoder.eval()
    
    transform = transforms.Compose(
    [transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])

    for cell_type in cell_types_list:
        print(cell_type)
        img_dir = f"/Users/scui2/Desktop/Breast_Cancer_HE_patches/Training_tiles/{cell_type}"
        csv_file = f"/Users/scui2/Desktop/Breast_Cancer_HE_patches/Training_celltype_proportion/{cell_type}_celltype_proportion_combined.csv"
        output_file = f"/Users/scui2/Desktop/Breast_Cancer_HE_patches/Training_features/{cell_type}_training_precomputed_features_ProvGigapath.pt"
        if len(os.listdir(img_dir)) == 0:
            log(f"the image directory for {cell_type} is empty, skip the precomputation")
            continue
    
        log("create the image dataset...")
        dataset = ImageDataset_training(img_dir, csv_file, cell_type, transform)  
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        log(f"start to precompute the features for {len(dataset)} images...")
        embeddings_list = []
        celltype_proportions_list = []
        tile_ids_list = []
        sample_ids_list = []
        individual_ids_list = []
        with torch.no_grad():
            for images, celltype_prop, tile_id, sample_id, individual_id in tqdm(loader, desc="calculate the features"):
                images = images.to(device)
                embeddings = tile_encoder(images)
                embeddings_list.append(embeddings.cpu())
                celltype_proportions_list.append(celltype_prop)
                tile_ids_list.extend(tile_id)
                sample_ids_list.extend(sample_id)
                individual_ids_list.extend(individual_id)

        embeddings = torch.cat(embeddings_list, 0)
        celltype_proportions = torch.cat(celltype_proportions_list, 0)
        log(f"the features are calculated, the number of features: {len(embeddings)}")
        
        log(f"save the features to: {output_file}")
        torch.save({
            'embeddings': embeddings,
            'celltype_proportions': celltype_proportions,
            'tile_ids': tile_ids_list,
            'sample_ids': sample_ids_list,
            'individual_ids': individual_ids_list,
        }, output_file)
        log("the precomputation is completed, the features are saved")


def create_training_features_Virchow(cell_types_list):
    batch_size = 32
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log("use CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log("use MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        log("use CPU")
    
    tile_encoder = load_Virchow_model()
    tile_encoder = tile_encoder.to(device)
    tile_encoder.eval()
    
    transform = transforms.Compose([
    transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])

    for cell_type in cell_types_list:
        print(cell_type)
        img_dir = f"/Users/scui2/Desktop/Breast_Cancer_HE_patches/Training_tiles/{cell_type}"
        csv_file = f"/Users/scui2/Desktop/Breast_Cancer_HE_patches/Training_celltype_proportion/{cell_type}_celltype_proportion_combined.csv"
        output_file = f"/Users/scui2/Desktop/Breast_Cancer_HE_patches/Training_features/{cell_type}_training_precomputed_features_Virchow.pt"
        if len(os.listdir(img_dir)) == 0:
            log(f"the image directory for {cell_type} is empty, skip the precomputation")
            continue

        log("create the image dataset...")
        dataset = ImageDataset_training(img_dir, csv_file, cell_type, transform)
        print(f"Total dataset size: {len(dataset)}")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        log(f"start to precompute the features for {len(dataset)} images...")
        embeddings_list = []
        celltype_proportions_list = []
        tile_ids_list = []
        sample_ids_list = []
        individual_ids_list = []
        with torch.no_grad():
            for images, celltype_prop, tile_id, sample_id, individual_id in tqdm(loader, desc="calculate the features"):
                images = images.to(device)
                embeddings = tile_encoder(images)
                embeddings_stacked = torch.cat([embeddings[:,0], embeddings[:,5:].mean(1)], dim=-1)  
                embeddings_list.append(embeddings_stacked.cpu())
                celltype_proportions_list.append(celltype_prop)
                tile_ids_list.extend(tile_id)
                sample_ids_list.extend(sample_id)
                individual_ids_list.extend(individual_id)

        embeddings = torch.cat(embeddings_list, 0)
        celltype_proportions = torch.cat(celltype_proportions_list, 0)
        log(f"the features are calculated, the number of features: {len(embeddings)}")
        
        log(f"save the features to: {output_file}")
        torch.save({
            'embeddings': embeddings,
            'celltype_proportions': celltype_proportions,
            'tile_ids': tile_ids_list,
            'sample_ids': sample_ids_list,
            'individual_ids': individual_ids_list,
        }, output_file)
        log("the precomputation is completed, the features are saved")


def create_training_features_Virchow2(cell_types_list):
    batch_size = 32
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log("use CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log("use MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        log("use CPU")
    
    tile_encoder = load_Virchow2_model()
    tile_encoder = tile_encoder.to(device)
    tile_encoder.eval()
    
    transform = transforms.Compose([
    transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])

    for cell_type in cell_types_list:
        print(cell_type)
        img_dir = f"/Users/scui2/Desktop/Breast_Cancer_HE_patches/Training_tiles/{cell_type}"
        csv_file = f"/Users/scui2/Desktop/Breast_Cancer_HE_patches/Training_celltype_proportion/{cell_type}_celltype_proportion_combined.csv"
        output_file = f"/Users/scui2/Desktop/Breast_Cancer_HE_patches/Training_features/{cell_type}_training_precomputed_features_Virchow2.pt"
        if len(os.listdir(img_dir)) == 0:
            log(f"the image directory for {cell_type} is empty, skip the precomputation")
            continue

        log("create the image dataset...")
        dataset = ImageDataset_training(img_dir, csv_file, cell_type, transform)
        print(f"Total dataset size: {len(dataset)}")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        log(f"start to precompute the features for {len(dataset)} images...")
        embeddings_list = []
        celltype_proportions_list = []
        tile_ids_list = []
        sample_ids_list = []
        individual_ids_list = []
        with torch.no_grad():
            for images, celltype_prop, tile_id, sample_id, individual_id in tqdm(loader, desc="calculate the features"):
                images = images.to(device)
                embeddings = tile_encoder(images)
                embeddings_stacked = torch.cat([embeddings[:,0], embeddings[:,5:].mean(1)], dim=-1)  
                embeddings_list.append(embeddings_stacked.cpu())
                celltype_proportions_list.append(celltype_prop)
                tile_ids_list.extend(tile_id)
                sample_ids_list.extend(sample_id)
                individual_ids_list.extend(individual_id)

        embeddings = torch.cat(embeddings_list, 0)
        celltype_proportions = torch.cat(celltype_proportions_list, 0)
        log(f"the features are calculated, the number of features: {len(embeddings)}")
        
        log(f"save the features to: {output_file}")
        torch.save({
            'embeddings': embeddings,
            'celltype_proportions': celltype_proportions,
            'tile_ids': tile_ids_list,
            'sample_ids': sample_ids_list,
            'individual_ids': individual_ids_list,
        }, output_file)
        log("the precomputation is completed, the features are saved")


def create_training_features_ResNet50(cell_types_list):
    batch_size = 32
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log("use CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log("use MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        log("use CPU")
    
    log("load the ImageNet pre-trained ResNet50 model...")
    model = models.resnet50(pretrained=True)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for cell_type in cell_types_list:
        print(cell_type)
        img_dir = f"/Users/scui2/Desktop/Breast_Cancer_HE_patches/Training_tiles/{cell_type}"
        csv_file = f"/Users/scui2/Desktop/Breast_Cancer_HE_patches/Training_celltype_proportion/{cell_type}_celltype_proportion_combined.csv"
        output_file = f"/Users/scui2/Desktop/Breast_Cancer_HE_patches/Training_features/{cell_type}_training_precomputed_features_ResNet50.pt"
        if len(os.listdir(img_dir)) == 0:
            log(f"the image directory for {cell_type} is empty, skip the precomputation")
            continue
        
        log(f"create the image dataset for {cell_type}...")
        dataset = ImageDataset_training(img_dir, csv_file, cell_type, transform)
        log(f"the dataset contains {len(dataset)} images")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        log(f"start to precompute the features for {len(dataset)} images...")
        embeddings_list = []
        celltype_proportions_list = []
        tile_ids_list = []
        sample_ids_list = []
        individual_ids_list = []
        
        with torch.no_grad():
            for images, celltype_prop, tile_id, sample_id, individual_id in tqdm(loader, desc="Computing ResNet50 features"):
                images = images.to(device)
                features = feature_extractor(images)
                features = features.squeeze(-1).squeeze(-1)
                embeddings_list.append(features.cpu())
                celltype_proportions_list.append(celltype_prop)
                tile_ids_list.extend(tile_id)
                sample_ids_list.extend(sample_id)
                individual_ids_list.extend(individual_id)
                
        embeddings = torch.cat(embeddings_list, 0)
        celltype_proportions = torch.cat(celltype_proportions_list, 0)
        log(f"ResNet50 feature calculation completed, the number of features: {len(embeddings)}, the dimension of the features: {embeddings.shape[1]}")
        
        log(f"save the features to: {output_file}")
        torch.save({
            'embeddings': embeddings,
            'celltype_proportions': celltype_proportions,
            'tile_ids': tile_ids_list,
            'sample_ids': sample_ids_list,
            'individual_ids': individual_ids_list,
        }, output_file)
        log(f"{cell_type} ResNet50 feature extraction completed")


if __name__ == "__main__":
    # BRCA cell types
    cell_types_list = ["Cancer.Epithelial", "Other.Epithelial", "CAFs", "Myeloid", "Plasma", "B.cell", "T.cell", "Endothelial"]
    create_training_features_Conch(cell_types_list = cell_types_list)
    create_training_features_ResNet50(cell_types_list = cell_types_list)
    create_training_features_ProvGigapath(cell_types_list = cell_types_list)
    create_training_features_UNI2h(cell_types_list = cell_types_list)
    create_training_features_Virchow(cell_types_list = cell_types_list)
    create_training_features_Virchow2(cell_types_list = cell_types_list)

