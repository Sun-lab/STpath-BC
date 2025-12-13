# STPath-BRCA: Spatial Transcriptomics Pathology for Breast Cancer

A comprehensive computational framework for predicting cell type proportions in breast cancer H&E images using multiple foundation models and machine learning approaches.

## Overview

The framework integrates multiple histopathology foundation models with XGBoost classifiers to predict spatial distributions of cell types in breast cancer tissues, validated against spatial transcriptomics data.

## Key Features

- **Multi-modal Foundation Models**: Integration of 6 state-of-the-art histopathology foundation models (ResNet50, Conch, UNI2-h, ProvGigaPath, Virchow, Virchow2)
- **Cell Type Deconvolution**: Spatial transcriptomics-guided cell type prediction for 8 major cell populations
- **TCGA Analysis**: Comprehensive survival analysis and clinical correlation studies
- **Cross-validation Framework**: Leave-one-individual-out (LOIO) validation strategy
- **Visualization Tools**: Hexagonal heatmaps, UMAP embeddings, and soft segmentation

## Cell Type Classification

The framework predicts proportions for 8 merged cell types:

1. **Cancer Epithelial Cells**:
2. **CAFs/PVL**
3. **Normal Epithelial Cells**
4. **T Cells**
5. **B Cells**
6. **Myeloid cells**
7. **Plasma Cells**
8. **Endothelial Cells**

## Repository Structure

```
STPath-BRCA/
├── Scripts_for_Analysis/          # Scripts for generating manuscript figures
│
├── Workflow/                      # Complete analysis workflow
│   ├── StepA_Data_Preparation/
│   │
│   ├── StepB_Cell_Type_Deconvolution/
│   │
│   ├── StepC_Feature_Extraction_and_Train_Models/
│   │
│   └── StepD_TCGA_Data_Preparation/
│
├── Figures/                       # Manuscript figures and supplementary figures
├── config_template.yaml           # Configuration template
├── README.md                      # This file
└── LICENSE                        # License information
```

## Workflow Overview

### Step A: Data Preparation
- Prepare spatial transcriptomics data for CARD deconvolution
- Extract H&E image patches at matched spatial locations
- Preprocess single-cell reference data for CARD deconvolution

### Step B: Cell Type Deconvolution
- Run CARD deconvolution using single-cell reference data
- Validate deconvolution results against known tissue regions
- Generate visualization of cell type spatial distributions

### Step C: Feature Extraction & Model Training
- Extract features using 6 foundation models
- Train XGBoost models with LOIO cross-validation
- Evaluate model performance and feature importance
- Generate calibrated predictions

### Step D: TCGA Data Processing
- Convert TCGA whole slide images from DCM to TIFF format
- Process TCGA-COAD cohort for validation studies
- Calculate distance metrics between cell types
- Perform survival analysis

## Requirements

### Software Dependencies
```
Python 3.8+
R 4.0+ (for CARD deconvolution)

Python packages:
- torch
- timm
- conch
- huggingface_hub
- xgboost
- numpy
- pandas
- matplotlib
- seaborn
- scanpy
- lifelines
- scipy
- scikit-learn
- umap-learn
- tqdm
- pillow
- opencv-python

R packages:
- CARD
- Seurat
```

### Hardware Requirements
- **Recommended**: GPU with 16GB+ VRAM (for foundation model feature extraction)
- **Minimum**: CPU with 32GB RAM
- **Storage**: 100GB+ for models, data, and intermediate results

### Data Requirements
1. **Spatial Transcriptomics Data**: Spot-level gene expression and spatial coordinates
2. **Single-cell Reference**: Annotated scRNA-seq data for CARD deconvolution
3. **H&E Images**: Whole slide images or tissue regions
4. **TCGA Data** (optional): For validation and survival analysis

## Getting Started

### 1. Environment Setup

```bash
# Create conda environment
conda create -n stpath python=3.9
conda activate stpath

# Install Python dependencies
pip install torch torchvision torchaudio
pip install timm transformers huggingface_hub
pip install xgboost scikit-learn
pip install numpy pandas matplotlib seaborn
pip install scanpy lifelines
pip install umap-learn tqdm pillow opencv-python
```

### 2. Configuration

Copy and edit the configuration template:
```bash
cp config_template.yaml config.yaml
```

Edit paths and parameters according to your data location and computing resources.

## Key Analyses

### Foundation Model Comparison
- Systematic comparison of 6 foundation models (Figure 4)
- Feature importance analysis across models
- Model complementarity assessment

### Spatial Analysis
- Cell type distance calculations using weighted minimum distance
- Hard classification based on quantile thresholds
- Hexagonal heatmap visualization

### Survival Analysis
- Cox proportional hazards models
- Kaplan-Meier survival curves
- Integration of cell type proportions, spatial metrics, and clinical variables

### Gene Expression Prediction
- Correlation between histopathology features and gene expression
- Cell type-specific marker gene analysis
- Partial correlation controlling for cell type composition

## Model Training Strategy

### Leave-One-Individual-Out (LOIO) Cross-Validation
- Each individual (patient) is held out once as test set
- Models trained on all other individuals
- Ensures generalization across patients
- Prevents overfitting to individual-specific patterns

### Calibration
- Quantile-based calibration for improved prediction accuracy
- Cell type-specific outlier handling
- Preservation of rank ordering while adjusting scale

### Feature Selection
- Top 30% most important features selected per model
- Combined feature set from multiple foundation models
- Reduces dimensionality while maintaining predictive power

## Output Files

### Model Predictions
- CSV files with patch-level predictions
- JSON files with metadata and overall proportions
- Feature importance scores

### Visualizations
- UMAP embeddings colored by various factors
- Hexagonal heatmaps for spatial distributions
- Violin plots for cell type proportions
- Scatter plots for model comparisons
- Kaplan-Meier survival curves

### Statistical Results
- Cox regression tables
- Model performance metrics (Spearman correlation, MAE, RMSE)
- Feature importance rankings

## Performance Metrics

### Model Evaluation
- **Spearman Correlation**: Primary metric for proportion prediction
- **Mean Absolute Error (MAE)**: Absolute prediction error
- **Root Mean Squared Error (RMSE)**: Squared error magnitude

## Computational Resources

### Processing Time
- **Feature Extraction**: ~2-5 minutes per WSI (GPU)
- **Model Training**: ~30-60 minutes per cell type (LOIO)
- **Prediction**: ~1-3 minutes per WSI
- **TCGA Analysis**: ~10-20 minutes per sample

### Memory Requirements
- **Feature Extraction**: 16GB GPU VRAM recommended
- **Model Training**: 32GB RAM minimum
- **Large WSI Processing**: 64GB RAM recommended

## Citation

If you use this code in your research, please cite:

```
[Citation information to be added upon publication]
```

## Related Repositories

- **STPath-Software**: Standalone prediction tool for clinical use (separate repository)
- **COAD Analysis**: Breast cancer application scripts (see `/Users/scui2/Desktop/BRCA_github/`)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, issues, or collaborations, please open an issue on GitHub or contact the authors.

## Acknowledgments

- Foundation models: Conch, UNI, ProvGigaPath, Virchow, Virchow2
- CARD deconvolution method
- TCGA consortium for data access
- All data contributors and collaborators

---

**Last Updated**: December 2025
