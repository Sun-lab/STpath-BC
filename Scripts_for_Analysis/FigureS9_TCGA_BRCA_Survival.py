"""
FigureS9 TCGA BRCA Survival Analysis
=====================================
Merge distance metrics with clinical data for survival analysis

BRCA Cell Types for Distance Analysis (8):
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

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

print(f"\n{'='*80}")
print(f"TCGA BRCA Survival Analysis - Data Preparation")
print(f"{'='*80}\n")

### ========================================================================
### Step 1: Load Clinical Data from TCGA-CDR Excel
### ========================================================================

print("Step 1: Loading clinical data from TCGA-CDR...")

clinical_excel = "/Users/scui2/Desktop/Breast_Cancer_HE_patches/TCGA-CDR-SupplementalTableS1.xlsx"
df_clinical = pd.read_excel(clinical_excel, sheet_name=0)

print(f"  Loaded clinical data: {df_clinical.shape[0]} rows × {df_clinical.shape[1]} columns")

clinical_cols = [
    'bcr_patient_barcode',
    'type',
    'age_at_initial_pathologic_diagnosis',
    'gender',
    'ajcc_pathologic_tumor_stage',
    'PFI.time',
    "PFI",
    'OS.time',
    "OS"
]

available_cols = [col for col in clinical_cols if col in df_clinical.columns]
missing_cols = [col for col in clinical_cols if col not in df_clinical.columns]

if missing_cols:
    print(f"  Warning: Missing columns: {missing_cols}")

df_clinical_subset = df_clinical[available_cols].copy()

# Filter for BRCA samples only
if 'type' in df_clinical_subset.columns:
    df_clinical_subset = df_clinical_subset[df_clinical_subset['type'] == 'BRCA'].copy()
    print(f"  Filtered for BRCA: {len(df_clinical_subset)} samples")

print(f"  Clinical variables extracted: {list(df_clinical_subset.columns)}")


### ========================================================================
### Step 2: Load Distance Metrics
### ========================================================================

print(f"\nStep 2: Loading distance metrics...")

distance_csv = "/Users/scui2/Desktop/TCGA/TCGA_BRCA/TCGA_BRCA_Weighted_Distances/Distance_Summary_All_Samples.csv"
df_distance = pd.read_csv(distance_csv)

print(f"  Loaded distance data: {df_distance.shape[0]} samples × {df_distance.shape[1]} columns")

df_distance['patient_barcode'] = df_distance['Sample_ID'].apply(
    lambda x: '-'.join(x.split('-')[:3])
)

print(f"  Extracted patient barcodes from Sample_ID")
print(f"  Example: {df_distance['Sample_ID'].iloc[0]} -> {df_distance['patient_barcode'].iloc[0]}")


### ========================================================================
### Step 3: Impute Missing Distance Values with Column Maximum
### ========================================================================

print(f"\nStep 3: Imputing missing values in distance metrics...")

distance_cols = [col for col in df_distance.columns if col not in ['Sample_ID', 'patient_barcode']]

missing_summary = df_distance[distance_cols].isnull().sum()
missing_cols = missing_summary[missing_summary > 0]

if len(missing_cols) > 0:
    print(f"  Found missing values in {len(missing_cols)} columns:")
    for col, count in missing_cols.items():
        print(f"    {col}: {count} missing values")
    
    print(f"\n  Imputing missing values with column maximum...")
    for col in missing_cols.index:
        col_max = df_distance[col].max()
        n_missing = df_distance[col].isnull().sum()
        df_distance[col].fillna(col_max, inplace=True)
        print(f"    {col}: imputed {n_missing} values with {col_max:.6f}")
    
    print(f"\n  Imputation complete!")
else:
    print(f"  No missing values found in distance metrics")


### ========================================================================
### Step 4: Merge Clinical and Distance Data
### ========================================================================

print(f"\nStep 4: Merging clinical and distance data...")

df_merged = df_distance.merge(
    df_clinical_subset,
    left_on='patient_barcode',
    right_on='bcr_patient_barcode',
    how='inner'
)

print(f"  Merged data: {df_merged.shape[0]} samples × {df_merged.shape[1]} columns")
print(f"  Samples with both distance and clinical data: {len(df_merged)}")

duplicate_patients = df_merged['patient_barcode'].value_counts()
duplicates = duplicate_patients[duplicate_patients > 1]
if len(duplicates) > 0:
    print(f"  Warning: {len(duplicates)} patients have multiple slides")

if 'bcr_patient_barcode' in df_merged.columns:
    df_merged = df_merged.drop(columns=['bcr_patient_barcode'])


### ========================================================================
### Step 5: Data Cleaning
### ========================================================================

print(f"\nStep 5: Data cleaning...")

if 'type' in df_merged.columns:
    df_merged = df_merged.drop(columns=['type'])
    print(f"  Dropped 'type' column")

if 'patient_barcode' in df_merged.columns:
    df_merged = df_merged.drop(columns=['patient_barcode'])
    print(f"  Dropped 'patient_barcode' column")

if 'gender' in df_merged.columns:
    print(f"\n  Gender distribution:")
    print(df_merged['gender'].value_counts().to_string().replace('\n', '\n    '))

if 'age_at_initial_pathologic_diagnosis' in df_merged.columns:
    print(f"\n  Age statistics:")
    print(f"    Mean: {df_merged['age_at_initial_pathologic_diagnosis'].mean():.1f}")
    print(f"    Median: {df_merged['age_at_initial_pathologic_diagnosis'].median():.1f}")
    print(f"    Range: [{df_merged['age_at_initial_pathologic_diagnosis'].min():.0f}, {df_merged['age_at_initial_pathologic_diagnosis'].max():.0f}]")

if 'ajcc_pathologic_tumor_stage' in df_merged.columns:
    print(f"\n  Stage distribution (before cleaning):")
    print(df_merged['ajcc_pathologic_tumor_stage'].value_counts().to_string().replace('\n', '\n    '))


### ========================================================================
### Step 6: Save Merged and Cleaned Data
### ========================================================================

print(f"\n{'='*80}")
print(f"Step 6: Saving merged and cleaned data...")
print(f"{'='*80}\n")

output_dir = "/Users/scui2/Desktop/TCGA/TCGA_BRCA/TCGA_BRCA_Survival_Analysis"
os.makedirs(output_dir, exist_ok=True)

output_csv = os.path.join(output_dir, "BRCA_Distance_Clinical_Merged.csv")
df_merged.to_csv(output_csv, index=False)

print(f"Merged data saved: {output_csv}")
print(f"   Shape: {df_merged.shape[0]} samples × {df_merged.shape[1]} columns")
print(f"\nFinal dataset summary:")
print(f"  Samples: {len(df_merged)}")
print(f"  Distance metrics: {len(distance_cols)}")
print(f"  Clinical variables: age, gender, stage, PFI, PFI.time, OS, OS.time")

print(f"\nMissing values in clinical variables:")
clinical_vars = ['age_at_initial_pathologic_diagnosis', 'gender', 'ajcc_pathologic_tumor_stage', 'PFI', 'PFI.time', 'OS', 'OS.time']
for col in clinical_vars:
    if col in df_merged.columns:
        missing_count = df_merged[col].isna().sum()
        missing_pct = (missing_count / len(df_merged)) * 100
        print(f"  {col:40s}: {missing_count:4d} ({missing_pct:5.1f}%)")

print(f"\n{'='*80}")
print(f"Data preparation complete!")
print(f"{'='*80}\n")


### ========================================================================
### Step 7: Joint Cox Proportional Hazards Model (PFI)
### ========================================================================

print(f"\n{'#'*80}")
print(f"Step 7: Joint Cox Proportional Hazards Model (PFI Endpoint)")
print(f"{'#'*80}\n")

from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

print(f"Preparing joint Cox model with:")
print(f"  - Proportion variables (8 BRCA cell types)")
print(f"  - Median distance metrics")
print(f"  - Age")
print(f"  - Gender\n")

### BRCA has 8 cell types
proportion_vars = [
    'Cancer.Epithelial_overall_proportion',
    'Other.Epithelial_overall_proportion',
    'CAFs_overall_proportion',
    'Myeloid_overall_proportion',
    'Plasma_overall_proportion',
    'B.cell_overall_proportion',
    'T.cell_overall_proportion',
    'Endothelial_overall_proportion'
]

### Directional pairs for distance (using key cell types)
directional_pairs = [
    ('Cancer.Epithelial', 'T.cell'),
    ('Cancer.Epithelial', 'CAFs'),
    ('Cancer.Epithelial', 'Myeloid'),
    ('T.cell', 'Cancer.Epithelial'),
    ('T.cell', 'CAFs'),
    ('T.cell', 'Myeloid'),
    ('CAFs', 'Cancer.Epithelial'),
    ('CAFs', 'T.cell'),
    ('CAFs', 'Myeloid'),
    ('Myeloid', 'Cancer.Epithelial'),
    ('Myeloid', 'T.cell'),
    ('Myeloid', 'CAFs')
]

median_distance_cols = [f'{source}_to_{target}_median' for source, target in directional_pairs]

clinical_vars = ['age_at_initial_pathologic_diagnosis', 'gender']

joint_model_vars = proportion_vars + median_distance_cols + clinical_vars + ['PFI', 'PFI.time']

available_vars = [var for var in joint_model_vars if var in df_merged.columns]
missing_vars = [var for var in joint_model_vars if var not in df_merged.columns]

if missing_vars:
    print(f"Warning: Missing variables: {missing_vars}\n")

df_joint = df_merged[available_vars].copy()

print(f"Variables in joint model:")
print(f"  Proportion variables: {len([v for v in proportion_vars if v in available_vars])}")
print(f"  Median distance variables: {len([v for v in median_distance_cols if v in available_vars])}")
print(f"  Clinical variables: {len([v for v in clinical_vars if v in available_vars])}")

print(f"\nBefore removing missing values: {len(df_joint)} samples")
df_joint = df_joint.dropna(subset=['PFI', 'PFI.time'])
print(f"After removing missing PFI/PFI.time: {len(df_joint)} samples")

df_joint = df_joint.dropna()
print(f"After removing all missing values: {len(df_joint)} samples")

df_joint_encoded = pd.get_dummies(df_joint, columns=['gender'], drop_first=False)

if 'gender_MALE' in df_joint_encoded.columns:
    df_joint_encoded = df_joint_encoded.drop(columns=['gender_MALE'])
    print(f"\nGender encoding: MALE as reference category")

print(f"\nFinal joint model dataset:")
print(f"  Samples: {len(df_joint_encoded)}")
print(f"  Variables: {df_joint_encoded.shape[1]}")

df_cox_joint = df_joint_encoded.copy()
df_cox_joint = df_cox_joint.rename(columns={'PFI.time': 'duration', 'PFI': 'event'})

median_distance_cols_present = [col for col in median_distance_cols if col in df_cox_joint.columns]
print(f"\nScaling {len(median_distance_cols_present)} median distance variables by ×10...")
for col in median_distance_cols_present:
    df_cox_joint[col] = df_cox_joint[col] * 10

print(f"\nFitting joint Cox PH model...")

cph_joint = CoxPHFitter(penalizer=0.0)
cph_joint.fit(df_cox_joint, duration_col='duration', event_col='event', show_progress=False)

print(f"Joint Cox PH model fitted successfully!")

print(f"\n{'='*80}")
print(f"Joint Cox PH Model Summary (PFI Endpoint)")
print(f"Proportions + Median Distances + Age + Gender")
print(f"{'='*80}\n")

print(cph_joint.summary)

summary_csv = os.path.join(output_dir, "Joint_Cox_PH_Model_PFI.csv")
cph_joint.summary.to_csv(summary_csv)
print(f"\nJoint Cox PH summary saved: {summary_csv}")

print(f"\n{'='*80}")
print(f"Testing Proportional Hazards Assumption")
print(f"{'='*80}\n")

try:
    ph_test = proportional_hazard_test(cph_joint, df_cox_joint, time_transform='rank')
    print(ph_test)
    
    ph_test_csv = os.path.join(output_dir, "Joint_Cox_PH_Assumption_Test_PFI.csv")
    ph_test.to_csv(ph_test_csv)
    print(f"\nPH assumption test saved: {ph_test_csv}")
except Exception as e:
    print(f"Could not perform PH assumption test: {e}")

print(f"\n{'='*80}")
print(f"Key Statistics")
print(f"{'='*80}\n")

significant_vars = cph_joint.summary[cph_joint.summary['p'] < 0.05]
print(f"Significant variables (p < 0.05): {len(significant_vars)}")
if len(significant_vars) > 0:
    print(f"\n{significant_vars[['coef', 'exp(coef)', 'p']]}")

print(f"\nConcordance Index: {cph_joint.concordance_index_:.4f}")

print(f"\n{'='*80}")
print(f"Joint Cox PH Analysis Complete (PFI)!")
print(f"{'='*80}\n")


### ========================================================================
### Step 8: Marginal Cox Regression (Univariate) for Each Variable (PFI)
### ========================================================================

print(f"\n{'#'*80}")
print(f"Step 8: Marginal Cox Regression (PFI Endpoint)")
print(f"{'#'*80}\n")

print(f"Running marginal (univariate) Cox regression for:")
print(f"  - Proportion variables (8 BRCA cell types)")
print(f"  - Median distance variables")
print(f"  - Age")
print(f"  - Gender\n")

marginal_vars = proportion_vars + median_distance_cols + clinical_vars + ['PFI', 'PFI.time']

df_marginal = df_merged[[var for var in marginal_vars if var in df_merged.columns]].copy()

df_marginal = df_marginal.dropna(subset=['PFI', 'PFI.time'])
df_marginal = df_marginal.dropna()

print(f"Marginal analysis dataset: {len(df_marginal)} samples")

df_marginal_encoded = pd.get_dummies(df_marginal, columns=['gender'], drop_first=False)

if 'gender_MALE' in df_marginal_encoded.columns:
    df_marginal_encoded = df_marginal_encoded.drop(columns=['gender_MALE'])
    print(f"Gender encoding: MALE as reference\n")

df_marginal_cox = df_marginal_encoded.copy()
df_marginal_cox = df_marginal_cox.rename(columns={'PFI.time': 'duration', 'PFI': 'event'})

covariate_cols = [col for col in df_marginal_cox.columns if col not in ['duration', 'event']]

print(f"Running marginal Cox regression for {len(covariate_cols)} variables...")

marginal_results = []

for covariate in tqdm(covariate_cols, desc="Marginal Cox (PFI)"):
    try:
        df_temp = df_marginal_cox[['duration', 'event', covariate]].copy()
        
        cph_marginal = CoxPHFitter(penalizer=0.0)
        cph_marginal.fit(df_temp, duration_col='duration', event_col='event', show_progress=False)
        
        summary = cph_marginal.summary
        
        result = {
            'variable': covariate,
            'coef': summary.loc[covariate, 'coef'],
            'exp(coef)': summary.loc[covariate, 'exp(coef)'],
            'se(coef)': summary.loc[covariate, 'se(coef)'],
            'coef_lower_95': summary.loc[covariate, 'coef lower 95%'],
            'coef_upper_95': summary.loc[covariate, 'coef upper 95%'],
            'exp(coef)_lower_95': summary.loc[covariate, 'exp(coef) lower 95%'],
            'exp(coef)_upper_95': summary.loc[covariate, 'exp(coef) upper 95%'],
            'z': summary.loc[covariate, 'z'],
            'p': summary.loc[covariate, 'p'],
            'concordance_index': cph_marginal.concordance_index_,
            'n_events': int(df_temp['event'].sum()),
            'n_samples': len(df_temp)
        }
        
        marginal_results.append(result)
        
    except Exception as e:
        print(f"  Error fitting {covariate}: {e}")
        continue

df_marginal_results = pd.DataFrame(marginal_results)

df_marginal_results = df_marginal_results.sort_values('p')

print(f"\nMarginal Cox regression completed for {len(df_marginal_results)} variables")

significant_marginal = df_marginal_results[df_marginal_results['p'] < 0.05]
print(f"\nSignificant variables (p < 0.05): {len(significant_marginal)}")

if len(significant_marginal) > 0:
    print(f"\nTop significant variables:")
    print(significant_marginal[['variable', 'coef', 'exp(coef)', 'p', 'concordance_index']].head(10).to_string(index=False))

marginal_csv = os.path.join(output_dir, "Marginal_Cox_PFI.csv")
df_marginal_results.to_csv(marginal_csv, index=False)

print(f"\nMarginal Cox results saved: {marginal_csv}")
print(f"   Total variables: {len(df_marginal_results)}")
print(f"   Significant (p<0.05): {len(significant_marginal)}")
print(f"   Significant (p<0.01): {len(df_marginal_results[df_marginal_results['p'] < 0.01])}")

print(f"\n{'='*80}")
print(f"Marginal Analysis Complete (PFI)!")
print(f"{'='*80}\n")


### ========================================================================
### Final Summary
### ========================================================================

print(f"\n{'#'*80}")
print(f"ALL SURVIVAL ANALYSES COMPLETE!")
print(f"{'#'*80}\n")

print(f"Summary of outputs saved to: {output_dir}")
print(f"\nFiles created:")
print(f"  1. BRCA_Distance_Clinical_Merged.csv")
print(f"  2. Joint_Cox_PH_Model_PFI.csv")
print(f"  3. Joint_Cox_PH_Assumption_Test_PFI.csv")
print(f"  4. Marginal_Cox_PFI.csv")

print(f"\nAnalysis Summary:")
print(f"  Endpoint: PFI (Progression-Free Interval)")
print(f"  Joint Model: Proportions + Median Distances + Age + Gender")
print(f"  Marginal Models: Each variable tested individually")

print(f"\n{'='*80}\n")

