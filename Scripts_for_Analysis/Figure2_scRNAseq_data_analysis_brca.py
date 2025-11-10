import os
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from scipy import sparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import re
import matplotlib.patches as patches


# ---- paths ----
base = "Mo_2024"
out_file = os.path.join(base, "Mo_2024_merged.h5ad")

# ---- discover sample folders (each contains matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz) ----
sample_dirs = sorted([
    os.path.join(base, d) for d in os.listdir(base)
    if os.path.isdir(os.path.join(base, d))
       and os.path.exists(os.path.join(base, d, "matrix.mtx.gz"))
       and os.path.exists(os.path.join(base, d, "features.tsv.gz"))
       and os.path.exists(os.path.join(base, d, "barcodes.tsv.gz"))
])

adata_list = []
for d in sample_dirs:
    sample_id = os.path.basename(d)

    # Prefer gene symbols; fall back to gene IDs if symbols are missing/duplicated
    try:
        adata = sc.read_10x_mtx(d, var_names="gene_symbols", make_unique=True)
        # If symbols are blank or mostly duplicated, switch to IDs
        if adata.var_names.isnull().any() or adata.var_names.duplicated().mean() > 0.05:
            raise ValueError("Too many missing/duplicated gene symbols")
    except Exception:
        adata = sc.read_10x_mtx(d, var_names="gene_ids", make_unique=True)

    # add sample id
    adata.obs["sample_id"] = sample_id

    # prefix cell barcodes to ensure global uniqueness
    adata.obs_names = [f"{sample_id}_{bc}" for bc in adata.obs_names]

    # keep raw counts in a named layer
    adata.layers["counts"] = adata.X.copy()

    # annotate mt genes (human = "^MT-"; mouse = "^mt-")
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")

    adata_list.append(adata)
    print(f"Loaded {sample_id}: {adata.n_obs} cells, {adata.n_vars} genes")

# ---- concatenate (outer-join genes) ----
adata = ad.concat(
    adata_list,
    join="outer",
    label="sample_id",
    keys=[a.obs["sample_id"].unique()[0] for a in adata_list],
    fill_value=0,
    merge="same"  # only merge obs/var columns if identical names
)

# Ensure sparse after concat
if not isinstance(adata.X, (np.ndarray,)):
    pass
else:
    # reconvert dense to sparse if large (rare with recent anndata versions)
    from scipy import sparse
    adata.X = sparse.csr_matrix(adata.X)

# Keep a clean counts layer after union join
if "counts" not in adata.layers:
    adata.layers["counts"] = adata.X.copy()
else:
    # After outer-join, some layers may be missing -> enforce counts layer shape
    adata.layers["counts"] = sparse.csr_matrix(adata.X)

adata.write_h5ad(out_file, compression="lzf")
print(f"Saved merged AnnData to: {out_file}")

# Load barcodes
barcodes_path = Path(base, "count_matrix_barcodes.tsv")
keep_df = pd.read_csv(barcodes_path, sep="\t", header=None, names=["barcode"], dtype=str)
keep = keep_df["barcode"].str.strip().unique()

# Strip the "scRNA_" prefix from each cell name in keep
keep_clean = pd.Index([
    s.replace("scRNA_", "", 1).replace("_", "-", 1)  # remove prefix, replace first _
    for s in keep
])

# Then filter directly
mask = adata.obs_names.isin(keep_clean)
print(f"{mask.sum():,} cells will be kept out of {adata.n_obs:,} total.")

adata_filt = adata[mask].copy()
print(f"Filtered AnnData: {adata_filt.n_obs:,} cells × {adata_filt.n_vars:,} genes")

print("Example cleaned names:", keep_clean[:5].to_list())
print("Example matching obs_names:", adata_filt.obs_names[:5].to_list())


sig_df = pd.read_csv(Path(base, "signature_genes_mo.csv"))
print(sig_df.head())

must_keep = sig_df['x'].astype(str).str.strip().unique().tolist()
print(f"{len(must_keep)} signature genes loaded.")


scData_Recluster = adata_filt.copy()

# filter genes
# sc.pp.filter_genes(scData_Recluster, min_cells= np.round(0.01*scData_Recluster.n_obs,0))  # Make sure for each gene at least 226 cells express the gene

# calculate quality control metrics
scData_Recluster.var["mt"] = scData_Recluster.var_names.str.upper().str.startswith("MT-")
sc.pp.calculate_qc_metrics(scData_Recluster, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

# normalize and log transform
sc.pp.normalize_total(scData_Recluster, target_sum=1e4)
sc.pp.log1p(scData_Recluster)

# compute HVGs without subsetting
sc.pp.highly_variable_genes(scData_Recluster,
                            min_mean=0.0125,
                            max_mean=3,
                            min_disp=0.5,
                            subset=False)
# ensure key genes stay marked as HVGs if present
present = [g for g in must_keep if g in scData_Recluster.var_names]
print(f"{len(present)} signature genes found in this dataset.")
scData_Recluster.var.loc[present, "highly_variable"] = True

# subset while keeping those must-keep genes
scData_Recluster = scData_Recluster[:, scData_Recluster.var["highly_variable"]].copy()

# sc.pp.highly_variable_genes(scData_Recluster, min_mean=0.0125, max_mean=3, min_disp=0.5)
# scData_Recluster = scData_Recluster[:, scData_Recluster.var.highly_variable]
# scData_Recluster_before_scale = scData_Recluster.copy()

# scale data
sc.pp.scale(scData_Recluster, max_value=10)

# PCA
sc.tl.pca(scData_Recluster, svd_solver='arpack')


# calculate neighbors graph
sc.pp.neighbors(scData_Recluster, n_neighbors=10, n_pcs=40)

# use Leiden algorithm for clustering
sc.tl.leiden(scData_Recluster, resolution=0.5)

# calculate UMAP
sc.tl.umap(scData_Recluster)

# plot UMAP, color by clustering results, and add labels
# Set global font settings for bold text

# Set figure size
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.figsize'] = [8, 8]

sc.settings.figdir = 'Visual/Sup/'

# Create the UMAP plot
fig, ax = plt.subplots(figsize=(10, 8))
sc.pl.umap(
    scData_Recluster,
    color=['leiden'],
    title='Leiden Clustering of scRNAseq data',
    size=25,  # Larger point size
    alpha=0.7,  # Adjust point transparency
    legend_loc='on data',  # Label on the plot
    legend_fontsize=18,  # Larger label font size
    legend_fontweight='bold',  # Adjust label font weight
    show=False,  # Don't show yet, we'll customize it
    ax=ax
)

# Customize the plot
ax.set_title('Leiden Clustering of scRNAseq data', fontsize=24, fontweight='bold', pad=20)

# Make tick labels larger and bold
ax.tick_params(axis='both', which='major', labelsize=16, width=3, length=6)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

# Make axis labels larger and bold
ax.set_xlabel('UMAP1', fontsize=20, fontweight='bold')
ax.set_ylabel('UMAP2', fontsize=20, fontweight='bold')

# Make borders thicker and black
for spine in ax.spines.values():
    spine.set_linewidth(5)
    spine.set_color('black')

plt.tight_layout()
plt.savefig('Visual/Sup/Sup_scRNAseq-Leiden_UMAP.png', 
            dpi=600, bbox_inches='tight')
plt.show()

p = Path(base, "metadata.csv")
if not p.exists():
    print("File not found at /mnt/data/metadata.csv")
else:
    try:
        dfm = pd.read_csv(p)
    except Exception:
        dfm = pd.read_csv(p, sep=",", engine="python")
    print(f"Loaded metadata.csv with shape {dfm.shape} and columns: {list(dfm.columns)}")

ID_COL = dfm.columns[0]   # assuming first column is the cell ID
print("Using ID column:", ID_COL)


id_clean = (
    dfm[ID_COL].astype(str).str.strip()
        .str.replace("^scRNA_", "", n=1, regex=True)
        .str.replace("_", "-", n=1)
)

# Set cleaned IDs as the index, drop the ID column from the data
md = dfm.copy()
md.index = id_clean
if ID_COL in md.columns:
    md = md.drop(columns=[ID_COL])

if md.index.duplicated().any():
    md = md[~md.index.duplicated(keep="first")]

print(f"Metadata rows: {md.shape[0]}, unique index: {~md.index.duplicated().any()}")

# Align to adata and join
common = scData_Recluster.obs.index.intersection(md.index)
print(f"Will join metadata to {len(common):,} cells in adata.")

scData_Recluster.obs = scData_Recluster.obs.join(md.add_prefix("meta_"), how="left")

print(scData_Recluster.obs.filter(regex="^meta_", axis=1).head())



# Create the UMAP plot for Cell Type
fig, ax = plt.subplots(figsize=(10, 8))
sc.pl.umap(
    scData_Recluster,
    color=['meta_cell_type_author'],
    title='Previous Cell Type Annotation of scRNAseq data',
    size=25,  # Larger point size
    alpha=0.7,  # Adjust point transparency
    legend_loc='on data',  # Label on the plot
    legend_fontsize=18,  # Larger label font size
    legend_fontweight='bold',  # Adjust label font weight
    show=False,  # Don't show yet, we'll customize it
    ax=ax
)

# Customize the plot
ax.set_title('Previous Cell Type Annotation of scRNAseq data', fontsize=24, fontweight='bold', pad=20)

# Make tick labels larger and bold
ax.tick_params(axis='both', which='major', labelsize=16, width=3, length=6)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

# Make axis labels larger and bold
ax.set_xlabel('UMAP1', fontsize=20, fontweight='bold')
ax.set_ylabel('UMAP2', fontsize=20, fontweight='bold')

# Make borders thicker and black
for spine in ax.spines.values():
    spine.set_linewidth(5)
    spine.set_color('black')

plt.tight_layout()
plt.savefig('Visual/Sup/Sup_scRNAseq-Previous_Ann_UMAP.png', 
            dpi=600, bbox_inches='tight')
plt.show()



# Core T-cell / CD4 / CD8 markers
markers = {
    "CD4_core":  ["CD4", "IL7R", "CCR7", "SELL", "TCF7"],
    "CD8_core":  ["CD8A", "CD8B", "GZMB", "PRF1", "NKG7"],
    "T_cell":    ["CD3D", "CD3E", "CD3G", "TRAC", "PTPRC"],
    # optional helpers
    "cytotoxic": ["GZMA", "GZMB", "GZMK", "PRF1", "NKG7", "IFNG"],
    "naive_T":   ["CCR7", "IL7R", "TCF7", "LEF1", "SELL"],
}


varnames = pd.Index(scData_Recluster.var_names.astype(str))

def find_genes(gene_list):
    found = []
    missing = []
    # try exact symbol match (case-insensitive), else try regex (beginning match)
    for g in gene_list:
        # exact (case-insensitive)
        exact = varnames.str.upper() == g.upper()
        if exact.any():
            found.extend(varnames[exact].tolist())
            continue
        # prefix regex (helps if e.g., Ensembl IDs have a SYMBOL column later; here we only have names)
        # skip prefix regex when names are Ensembl IDs; if your var has a SYMBOL column, see below
        missing.append(g)
    return sorted(set(found)), sorted(set(missing))

summary = {}
for k, genes in markers.items():
    f, m = find_genes(genes)
    summary[k] = {"found": f, "missing": m, "n_found": len(f), "n_missing": len(m)}

# pretty print
for k, d in summary.items():
    print(f"[{k}]  found: {d['n_found']}  missing: {d['n_missing']}")
    if d["found"]:
        print("  ✓", ", ".join(d["found"]))
    if d["missing"]:
        print("  ✗", ", ".join(d["missing"]))


# plot UMAP, color by CD8B gene expression

fig, ax = plt.subplots(figsize=(10, 8))
sc.pl.umap(
    scData_Recluster,
    color=['CD8B'],  # Use gene names to color the UMAP
    title='CD8B',
    size=25,  # Larger point size
    alpha=0.9,  # Adjust point transparency
    legend_loc='on data',  # Label on the plot
    legend_fontsize=18,  # Larger label font size
    legend_fontweight='bold',  # Adjust label font weight
    show=False,  # Don't show yet, we'll customize it
    ax=ax
)

# Customize the plot
ax.set_title('CD8B', fontsize=24, fontweight='bold', pad=20)


# Make tick labels larger and bold
ax.tick_params(axis='both', which='major', labelsize=16, width=3, length=6)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

# Make axis labels larger and bold
ax.set_xlabel('UMAP1', fontsize=20, fontweight='bold')
ax.set_ylabel('UMAP2', fontsize=20, fontweight='bold')

# Make borders thicker and black
for spine in ax.spines.values():
    spine.set_linewidth(5)
    spine.set_color('black')

# Add colorbar title
cbar = plt.gcf().axes[-1]  # Get the colorbar
cbar.set_ylabel('Log-Normalized Expression', fontsize=20, fontweight='bold', color='black')
cbar.tick_params(labelsize=16, colors='black', width=3)
for label in cbar.get_yticklabels():
    label.set_fontweight('bold')

plt.tight_layout()
plt.savefig('Visual/Sup/Sup_scRNAseq-CD8B_UMAP.png', 
            dpi=600, bbox_inches='tight')
plt.show()


# plot UMAP, color by CD4 gene expression

fig, ax = plt.subplots(figsize=(10, 8))
sc.pl.umap(
    scData_Recluster,
    color=['CD4'],  # Use gene names to color the UMAP
    title='CD4',
    size=25,  # Larger point size
    alpha=0.9,  # Adjust point transparency
    legend_loc='on data',  # Label on the plot
    legend_fontsize=18,  # Larger label font size
    legend_fontweight='bold',  # Adjust label font weight
    show=False,  # Don't show yet, we'll customize it
    ax=ax
)

# Customize the plot
ax.set_title('CD4', fontsize=24, fontweight='bold', pad=20)


# Make tick labels larger and bold
ax.tick_params(axis='both', which='major', labelsize=16, width=3, length=6)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

# Make axis labels larger and bold
ax.set_xlabel('UMAP1', fontsize=20, fontweight='bold')
ax.set_ylabel('UMAP2', fontsize=20, fontweight='bold')

# Make borders thicker and black
for spine in ax.spines.values():
    spine.set_linewidth(5)
    spine.set_color('black')

# Add colorbar title
cbar = plt.gcf().axes[-1]  # Get the colorbar
cbar.set_ylabel('Log-Normalized Expression', fontsize=20, fontweight='bold', color='black')
cbar.tick_params(labelsize=16, colors='black', width=3)
for label in cbar.get_yticklabels():
    label.set_fontweight('bold')

plt.tight_layout()
plt.savefig('Visual/Sup/Sup_scRNAseq-CD4_UMAP.png', 
            dpi=600, bbox_inches='tight')
plt.show()


# plot UMAP, color by CD3D gene expression

fig, ax = plt.subplots(figsize=(10, 8))
sc.pl.umap(
    scData_Recluster,
    color=['CD3D'],  # Use gene names to color the UMAP
    title='CD3D',
    size=25,  # Larger point size
    alpha=0.9,  # Adjust point transparency
    legend_loc='on data',  # Label on the plot
    legend_fontsize=18,  # Larger label font size
    legend_fontweight='bold',  # Adjust label font weight
    show=False,  # Don't show yet, we'll customize it
    ax=ax
)

# Customize the plot
ax.set_title('CD3D', fontsize=24, fontweight='bold', pad=20)


# Make tick labels larger and bold
ax.tick_params(axis='both', which='major', labelsize=16, width=3, length=6)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

# Make axis labels larger and bold
ax.set_xlabel('UMAP1', fontsize=20, fontweight='bold')
ax.set_ylabel('UMAP2', fontsize=20, fontweight='bold')

# Make borders thicker and black
for spine in ax.spines.values():
    spine.set_linewidth(5)
    spine.set_color('black')

# Add colorbar title
cbar = plt.gcf().axes[-1]  # Get the colorbar
cbar.set_ylabel('Log-Normalized Expression', fontsize=20, fontweight='bold', color='black')
cbar.tick_params(labelsize=16, colors='black', width=3)
for label in cbar.get_yticklabels():
    label.set_fontweight('bold')

plt.tight_layout()
plt.savefig('Visual/Sup/Sup_scRNAseq-CD3D_UMAP.png', 
            dpi=600, bbox_inches='tight')
plt.show()



# plot UMAP, color by CCR7 gene expression

fig, ax = plt.subplots(figsize=(10, 8))
sc.pl.umap(
    scData_Recluster,
    color=['CCR7'],  # Use gene names to color the UMAP
    title='CCR7',
    size=25,  # Larger point size
    alpha=0.9,  # Adjust point transparency
    legend_loc='on data',  # Label on the plot
    legend_fontsize=18,  # Larger label font size
    legend_fontweight='bold',  # Adjust label font weight
    show=False,  # Don't show yet, we'll customize it
    ax=ax
)

# Customize the plot
ax.set_title('CCR7', fontsize=24, fontweight='bold', pad=20)


# Make tick labels larger and bold
ax.tick_params(axis='both', which='major', labelsize=16, width=3, length=6)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

# Make axis labels larger and bold
ax.set_xlabel('UMAP1', fontsize=20, fontweight='bold')
ax.set_ylabel('UMAP2', fontsize=20, fontweight='bold')

# Make borders thicker and black
for spine in ax.spines.values():
    spine.set_linewidth(5)
    spine.set_color('black')

# Add colorbar title
cbar = plt.gcf().axes[-1]  # Get the colorbar
cbar.set_ylabel('Log-Normalized Expression', fontsize=20, fontweight='bold', color='black')
cbar.tick_params(labelsize=16, colors='black', width=3)
for label in cbar.get_yticklabels():
    label.set_fontweight('bold')

plt.tight_layout()
plt.savefig('Visual/Sup/Sup_scRNAseq-CCR7_UMAP.png', 
            dpi=600, bbox_inches='tight')
plt.show()



# create cross-tabulation tables
leiden_labels = scData_Recluster.obs['leiden']
previous_labels = scData_Recluster.obs['meta_cell_type_author']
sample_id = scData_Recluster.obs['sample_id']

crosstab_ReCluster_VS_preLabeled = pd.crosstab(leiden_labels, previous_labels, rownames=['Leiden Cluster'], colnames=['Previous Label'])
crosstab_ReCluster_VS_sample = pd.crosstab(leiden_labels, sample_id, rownames=['Leiden Cluster'], colnames=['Sample ID']).iloc[[7,16,17,18, 3, 15, 10],]
kept_sample_id_list = crosstab_ReCluster_VS_sample.sum(axis=0).sort_values(ascending=False)[0:39].index.tolist()
crosstab_ReCluster_VS_sample = crosstab_ReCluster_VS_sample.loc[:,kept_sample_id_list]


cell_type = "Cancer.Epithelial"
cell_type_order = ["Cancer.Epithelial", "Other.Epithelial", "CAFs/PVL", "Myeloid", "Plasma", "B.cell", "T.cell", "Endothelial"]

Region_ID = "Mo_2024_HTA12_129_9_region0"
st_dir = "Mo_2024_HTA12_129_9_region0/Mo_2024_HTA12_129_9_filtered_feature_bc_matrix"
try:
    adata_st = sc.read_10x_mtx(st_dir, var_names="gene_symbols", make_unique=True)
    # if too many duplicates/empties, fall back
    if adata.var_names.isnull().any() or adata.var_names.duplicated().mean() > 0.05:
        raise ValueError("Gene symbol quality insufficient, use IDs")
except Exception:
    adata_st = sc.read_10x_mtx(st_dir, var_names="gene_ids", make_unique=True)



# make sure counts are sparse CSR
if not sparse.issparse(adata_st.X):
    adata_st.X = sparse.csr_matrix(adata_st.X)

# prefix spot ids with Region_ID to ensure uniqueness downstream
# adata_st.obs_names = [f"{Region_ID}_{bc}" for bc in adata_st.obs_names]

# store raw counts in a layer for clarity (optional)
adata_st.layers["counts"] = adata_st.X.copy()

# annotate mitochondrial (optional; human)
adata_st.var["mt"] = adata_st.var_names.str.upper().str.startswith("MT-")

# ---- 2) build expression_df (genes x spots) from raw counts ----
# CARD typically expects genes as rows, spots as columns, raw integer counts (or at least unlogged)
counts = adata_st.layers.get("counts", adata_st.X)
# transpose: genes (rows) × spots (cols)
expression_df = pd.DataFrame.sparse.from_spmatrix(
    counts.T, index=adata_st.var_names, columns=adata_st.obs_names
)

Celltype_proportion_df = pd.read_csv("Mo_2024_HTA12_129_9_region0/Proportion_Mo_2024_HTA12_129_9_region0_cell_type_group.csv", index_col=0)
Celltype_proportion_df = Celltype_proportion_df[cell_type_order]
Celltype_proportion_df

B_matrix_df = pd.read_csv("Mo_2024_HTA12_129_9_region0/Mo_2024_HTA12_129_9_region0_B_Matrix.csv", index_col=0)
B_matrix_df = B_matrix_df[cell_type_order]
B_matrix_df


kept_barcode = Celltype_proportion_df.index.tolist()
expression_df = expression_df[kept_barcode]
expression_df


# per-spot library size (columns, not rows)
library_size = expression_df.sum(axis=0)

# scale factors; use 1.0 for proportions or 1e4 for CPM-like
target_sum = 1e4
scale = (target_sum / library_size).replace([np.inf, np.nan], 0.0)

# multiply (avoid pandas sparse division bug)
normalized_df = expression_df.mul(scale, axis=1)



cell_type = "Cancer.Epithelial"
markers_list = {
    "B.cell": [
        "CD19","MS4A1","CD79A","CD79B","BANK1",
        "CD69","IGHG1","JCHAIN"
    ],
    "T.cell": [
        "CD3D","CD3E","FOXP3","CD8A","CD8B","NKG7","CD69",
        "FCGR3A","SELL","CD4","CCR7","IL7R","TCF7","LEF1"
    ],
    "Plasma": [
        "IGHG1","JCHAIN","MZB1",
        "CD79A","TNFRSF17","IGKC","IGLC3","IGLC2","IGHA1",
        "IGHG3","IGHG4","IGHA2","FCRL5"
    ],
    "Myeloid": [
        "CD86","FCGR3A","CD68","CD14","CSF1R","IL1B",
        "MRC1","CD163","MSR1","IFI30","SPP1","APOE","LYZ","APOC1",
        "CD1C","CD74"
    ],
    "CAFs/PVL": ["COL1A1","COL3A1","COL1A2","POSTN","PDGFRB","RGS5","GJA4","SFRP2","DCN","DES"],
    "Endothelial": ["VWF","PECAM1","KDR","FLT1","SOX17","ACKR1","RGS5","GJA4","FABP4","LDB2"],
    "Cancer.Epithelial": ["EPCAM","KRT19","KRT8","KRT18","KRT5","KRT14"],
    "Other.Epithelial": ["KRT5","KRT14","EPCAM","KRT19","KRT8","KRT18","AMBP","MUC1"]
}

genes_in_norm = set(normalized_df.index.astype(str))
genes_in_B    = set(B_matrix_df.index.astype(str))

marker_genes_clean_dict = {}
for grp, genes in markers_list.items():
    # preserve order; drop dups; keep only genes present in both sources
    cleaned = [g for g in dict.fromkeys(genes) if (g in genes_in_norm) and (g in genes_in_B)]
    marker_genes_clean_dict[grp] = cleaned


# ---------------------------------------------------
# 2) Per-spot relative marker expression per group
#     normalized_df is genes x spots → select rows, transpose to spots x genes
#     Divide by B reference for that group (gene-wise), then mean across genes
# ---------------------------------------------------
group_scores = {}
eps = 1e-12  # avoid divide-by-zero if any B reference is 0

for grp, genes in marker_genes_clean_dict.items():
    if not genes:
        # no genes -> score is NaN; fill later
        group_scores[grp] = pd.Series(np.nan, index=normalized_df.columns)
        continue
    # spot x selected genes
    sub_spot_by_gene = normalized_df.loc[genes].T  # spots x genes
    # gene-wise reference from B (Series indexed by genes)
    if grp not in B_matrix_df.columns:
        # If your B-matrix uses slightly different names, map them here as needed.
        raise KeyError(f"Column '{grp}' not found in B_matrix_df.columns.")
    ref = B_matrix_df.loc[genes, grp].astype(float).replace(0.0, eps)
    # divide spot expression by reference (broadcast over columns), mean across genes → per-spot score
    group_scores[grp] = sub_spot_by_gene.div(ref, axis=1).mean(axis=1)

marker_genes_expression_df_grouped = pd.DataFrame(group_scores)  # spots x 8 groups


# Normalize within spot to get relative fractions (sum to 1 per spot)
relative_marker_genes_expression_df_grouped = marker_genes_expression_df_grouped.div(
    marker_genes_expression_df_grouped.sum(axis=1).replace(0, np.nan), axis=0
).fillna(0.0)
relative_marker_genes_expression_df_grouped


Celltype_proportion_df_grouped = Celltype_proportion_df[
    ["Cancer.Epithelial", "Other.Epithelial", "CAFs/PVL", 
     "Myeloid", "Plasma", "B.cell", "T.cell", "Endothelial"]
].copy()

focus_celltype_percent = (Celltype_proportion_df_grouped[cell_type]*100).round(4).values.tolist()
focus_relative_marker_genes_expression = np.round(
    relative_marker_genes_expression_df_grouped[cell_type].to_numpy(dtype=float) * 100, 4
).tolist()


# According to 10x Genomics Spatial format specification
colnames = [
    "barcode",         # 1st column: spot barcode
    "in_tissue",       # 2nd: 1 if spot is under tissue, 0 if not
    "array_row",       # 3rd: row position on the array grid
    "array_col",       # 4th: column position on the array grid
    "pxl_row_in_fullres",  # 5th: pixel row coordinate (in high-res image)
    "pxl_col_in_fullres"   # 6th: pixel column coordinate (in high-res image)
]

Spatial_location_df = pd.read_csv(
    "Mo_2024_HTA12_129_9_region0/Mo_2024_HTA12_129_9_spatial/tissue_positions_list.csv",
    header=None, names=colnames
)

# Make sure Spatial_location_df has barcode as the index and x/y columns
if "barcode" in Spatial_location_df.columns:
    Spatial_loc = Spatial_location_df.set_index("barcode")
else:
    Spatial_loc = Spatial_location_df.copy()  # already indexed by barcode

# standardize coordinate column names if needed
rename_map = {
    "pxl_col_in_fullres": "x",
    "pxl_row_in_fullres": "y",
    "xcoord": "x",
    "ycoord": "y"
}
Spatial_loc = Spatial_loc.rename(columns={k: v for k, v in rename_map.items() if k in Spatial_loc.columns})
assert {"x", "y"}.issubset(Spatial_loc.columns), "Spatial_location_df must contain x and y columns."

print(Spatial_loc.head())

if isinstance(focus_relative_marker_genes_expression, pd.Series):
    focus_expr = focus_relative_marker_genes_expression.astype(float)
else:
    # try to use the original index source used to create the list
    # first choice: relative_marker_genes_expression_df_grouped index
    if "relative_marker_genes_expression_df_grouped" in globals():
        idx = relative_marker_genes_expression_df_grouped.index
    # fallback: Celltype_proportion_df index (often aligns with spots)
    elif "Celltype_proportion_df" in globals():
        idx = Celltype_proportion_df.index
    else:
        raise ValueError("Need a barcode index to pair with focus_relative_marker_genes_expression list.")
    focus_expr = pd.Series(focus_relative_marker_genes_expression, index=idx, dtype=float)

if isinstance(focus_celltype_percent, pd.Series):
    focus_prop = focus_celltype_percent.astype(float)
else:
    # use same index as expression to keep alignment
    focus_prop = pd.Series(focus_celltype_percent, index=focus_expr.index, dtype=float)


# 2) Keep only barcodes present in BOTH (inner join by index)
common = Spatial_loc.index.intersection(focus_expr.index)
common

showcase_df = pd.concat(
    [
        focus_prop.reindex(common).rename("Proportion"),
        focus_expr.reindex(common).rename("Expression"),
        Spatial_loc.loc[common, ["x", "y"]],
    ],
    axis=1,
).dropna(subset=["x", "y"])


print(f"Spatial barcodes: {Spatial_loc.shape[0]:,} | focus barcodes: {focus_expr.shape[0]:,} | intersection: {showcase_df.shape[0]:,}")
showcase_df.head()



plt.figure(figsize=(14.5, 11.5))
ax = plt.gca()

# create the scatter plot with hexagon markers
plt.scatter(
    x=showcase_df['x'],
    y=showcase_df['y'],
    c=showcase_df['Proportion'],
    cmap='coolwarm',  # the color from light to dark
    marker='h',  # hexagon marker
    s=100,  # marker size
    edgecolors='black',
    linewidth=0.5
)

# Add colorbar
cbar = plt.colorbar()
cbar.set_label('Tumor cell proportion (%)', fontsize=25, fontweight='bold', color='black')
cbar.ax.tick_params(labelsize=18, colors='black', width=3)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_color('black')

# Invert the y axis to match the image coordinate system
ax.invert_yaxis()

# Set all four borders as thick black lines
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['top'].set_linewidth(5)
ax.spines['right'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
ax.spines['bottom'].set_linewidth(5)
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

# Remove title and axis labels
plt.title('')
plt.xlabel('')
plt.ylabel('')

# Remove ticks
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(f"Visual/Mo_2024_HTA12_129_9_region0_cancer.epithelial_proportion.png", dpi=600, bbox_inches='tight')
# Show the figure
plt.show()



# Set the figure size
plt.figure(figsize=(14.5, 11.5))
ax = plt.gca()

# Create the scatter plot with hexagon markers
plt.scatter(
    x=showcase_df['x'],
    y=showcase_df['y'],
    c=showcase_df['Expression'],
    cmap='coolwarm',  # the color from light to dark
    marker='h',  # hexagon marker
    s=100,  # marker size
    edgecolors='black',
    linewidth=0.5
)

# Add colorbar
cbar = plt.colorbar()
cbar.set_label('Relative Markers GE (%)', fontsize=25, fontweight='bold', color='black')
cbar.ax.tick_params(labelsize=18, colors='black', width=3)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_color('black')

# Invert the y axis to match the image coordinate system
ax.invert_yaxis()

# Set all four borders as thick black lines
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['top'].set_linewidth(5)
ax.spines['right'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
ax.spines['bottom'].set_linewidth(5)
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

# Remove title and axis labels
plt.title('')
plt.xlabel('')
plt.ylabel('')

# Remove ticks
ax.set_xticks([])
ax.set_yticks([])

plt.savefig(f"Visual/Mo_2024_HTA12_129_9_region0_Cancer_marker_genes_relative_expression.png", dpi=600, bbox_inches='tight')
plt.show()


### Figure 2 (C)
Celltype_proportion_df_mean_total = pd.read_csv("samples_train_mean.csv", index_col=0)
melted_df = Celltype_proportion_df_mean_total

# standardize coordinate column names if needed
rename_map = {
    "celltype": "Cell Type",
    "mean": "Proportion",
    "sid_region": "Sample"
}
melted_df = melted_df.rename(columns={k: v for k, v in rename_map.items() if k in melted_df.columns})

mean_proportions = melted_df.groupby('Cell Type')['Proportion'].mean().reset_index()
std_proportions = melted_df.groupby('Cell Type')['Proportion'].std().reset_index()
# Merge mean and std
mean_proportions['std'] = std_proportions['Proportion']

# Convert the 'Cell Type' column to a categorical type and set the order
melted_df['Cell Type'] = pd.Categorical(melted_df['Cell Type'], categories=cell_type_order, ordered=True)
mean_proportions['Cell Type'] = pd.Categorical(mean_proportions['Cell Type'], categories=cell_type_order, ordered=True)
mean_proportions = mean_proportions.sort_values('Cell Type')


palette= {
    'Cancer.Epithelial': '#E41A1C',               # red
    'Other.Epithelial': '#377EB8',    # blue
    'T.cell': '#4DAF4A',              # green
    'B.cell': '#F781BF',            # pink
    'Myeloid': '#A65628',          # brown
    'Plasma': '#FF7F00',                 # orange
    'CAFs/PVL': '#FFFF33',              # yellow
    'Endothelial': '#999999' # gray
                  }

# Create bar plot with error bars and scatter points
plt.figure(figsize=(10, 10))  # reduce figure width to make bars closer

# define bar swidth for consistent spacing
bar_width = 0.4
    
# draw the bar plot
bars = sns.barplot(
        data=mean_proportions,
        x='Cell Type',
        y='Proportion',
        palette=palette,
        edgecolor='black',  # set the rectangle border to black
        linewidth=6,  # the width of the rectangle border - bold
        errorbar=None,  # disable the default error bar
        width=bar_width  # make bars narrower
    )

# draw the error bar on each bar
for i, (mean, std) in enumerate(zip(mean_proportions['Proportion'], mean_proportions['std'])):
    # draw the error bar
    plt.plot(
        [i, i],  # x 
        [mean, mean + 0.5*std],  # y 
        color='black',  # the color of the error bar
        linewidth=5  # the width of the error bar - same thickness as border
    )
    # draw the error bar cap with width matching bar width
    cap_half_width = bar_width / 2
    plt.plot(
        [i - cap_half_width, i + cap_half_width],  # x coordinates matching bar width
        [mean + 0.5*std, mean + 0.5*std],  # y 
        color='black',  # the color of the error bar cap
        linewidth=5  # the width of the error bar cap - same thickness as border
    )

# draw the scatter plot on the bar plot
dot_size = 10

sns.stripplot(
        data=melted_df,
        x='Cell Type',
        y='Proportion',
        color='black',  # the color of the scatter plot
        jitter= True,  # enable the jitter
        size= dot_size,  # the size of the scatter plot
        alpha=0.7  # opaque, same as scatter plot
    )


# add the horizontal dashed line - grid lines same as scatter plot
plt.grid(True, axis='y', linestyle='--', linewidth=4, color='gray', alpha=0.7)


# set the style of the coordinate axis
plt.gca().xaxis.set_tick_params(labelsize=24, width=3, rotation=90)  # increase font size
plt.gca().yaxis.set_tick_params(labelsize=20, width=3)  # increase font size
plt.gca().tick_params(axis='both', which='both', color='black', direction='out')  # set the style of the tick

# Make x-axis labels bold and black
for label in plt.gca().get_xticklabels():
    label.set_fontweight('bold')
    label.set_color('black')

# Make y-axis labels bold and black
for label in plt.gca().get_yticklabels():
    label.set_fontweight('bold')
    label.set_color('black')

# Set left, bottom, and right borders as thick black lines
ax = plt.gca()
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['top'].set_linewidth(5)
ax.spines['right'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
ax.spines['bottom'].set_linewidth(5)
ax.spines['top'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

# remove the horizontal coordinate axis title
plt.xlabel(None)

plt.ylabel("Proportion (%)", fontsize=22, fontweight='bold', color='black', labelpad=10)
# remove the title
plt.title("Deconvoluted Proportion", fontsize=22, fontweight='bold', color='black', pad=12)

ax = plt.gca()
ax.add_patch(
    patches.Rectangle(
        (0, 1.01),          # x,y position (start just above plot)
        1, 0.1,            # width=100% of axis, height of box
        transform=ax.transAxes,
        color='#D9F0F6',    # light cyan background
        zorder=-1,          # draw behind text
        clip_on=False
    )
)

# re-draw title so it’s above the box
plt.title("Deconvoluted Proportion", fontsize=22, fontweight='bold', color='black', pad=12)


# show the figure
plt.tight_layout(rect=[0, 0, 1, 1])  # leave more top margin
plt.ylim(0, 1)  
plt.savefig('Visual/CARD_Results_Vis_Combined_Ungrouped.png', dpi=600)






