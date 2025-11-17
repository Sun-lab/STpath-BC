library(Seurat)
library(dplyr)
library(ggplot2)
library(patchwork)
library(SeuratDisk)

sc_dir <- "/fh/working/sun_w/zsui/st2image_data/brca_scRNAseq_datasets"
# sc_dir <- "/Volumes/sun_w/zsui/st2image_data/brca_scRNAseq_datasets"

# Create a Seurat object for all samples -------------------------------

dir <- file.path(sc_dir, "Wu_2021")
features <- read.delim(file.path(dir, "features.tsv.gz"), header = FALSE)
exprs_data <- Read10X(data.dir = dir, gene.column = 1)

# Create a Seurat object
seurat_Wu <- CreateSeuratObject(counts = exprs_data, project = "Wu_2021")

# Ensure metadata is atomic
seurat_Wu@meta.data <- seurat_Wu@meta.data %>%
  mutate(across(everything(), as.character))

# Store the Seurat object
DefaultAssay(seurat_Wu) <- "RNA"

save(seurat_Wu, file = file.path(dir, "scRNAseq_seurat_Wu_2021.rda"))

# Perform clustering -------------------------------

seurat_Wu <- NormalizeData(seurat_Wu)
seurat_Wu <- FindVariableFeatures(seurat_Wu, selection.method = "vst", nfeatures = 2000)
top10 <- head(VariableFeatures(seurat_Wu), 10)
plot1 <- VariableFeaturePlot(seurat_Wu)
plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)

all.genes <- rownames(seurat_Wu)

seurat_Wu <- ScaleData(seurat_Wu, features = all.genes)
seurat_Wu <- RunPCA(seurat_Wu, features = VariableFeatures(object = seurat_Wu))
print(seurat_Wu[["pca"]], dims = 1:5, nfeatures = 5)
VizDimLoadings(seurat_Wu, dims = 1:2, reduction = "pca")
DimPlot(seurat_Wu, reduction = "pca") + NoLegend()
DimHeatmap(seurat_Wu, dims = 1, cells = 500, balanced = TRUE)
DimHeatmap(seurat_Wu, dims = 1:15, cells = 500, balanced = TRUE)

seurat_Wu <- FindNeighbors(seurat_Wu, dims = 1:30)
seurat_Wu <- FindClusters(seurat_Wu, resolution = 0.5)
seurat_Wu <- RunUMAP(seurat_Wu, dims = 1:30)

# Add authors' annotations 
metadata_wu <- read.csv(file.path(dir, "metadata.csv"), row.names = 1)
metadata <- seurat_Wu@meta.data
metadata <- bind_cols(metadata, metadata_wu[, 4:ncol(metadata_wu)])
seurat_Wu@meta.data <- metadata
saveRDS(seurat_Wu, file = file.path(dir, "seurat_Wu_clustered.rds"))

# Check the marker genes ------------------------------------------------

seurat_Wu <- readRDS(file.path(dir, "seurat_Wu_clustered.rds"))

metadata <- seurat_Wu@meta.data
table(metadata$seurat_clusters, metadata$celltype_major)

pdf(file.path(dir, "dimplots.pdf"), width = 15, height = 15)
p1 <- DimPlot(seurat_Wu, reduction = "umap", label = TRUE) + 
  ggtitle("Cluster") +
  theme(legend.position = "none")
p2 <- DimPlot(seurat_Wu, reduction = "umap", group.by = "orig.ident") + 
  ggtitle("Sample") +
  theme(legend.position = "bottom")
p3 <- DimPlot(seurat_Wu, reduction = "umap", group.by = "celltype_major") + 
  ggtitle("Major Cell Type (Wu et al.)") +
  theme(legend.position = "bottom")
p4 <- DimPlot(seurat_Wu, reduction = "umap", group.by = "celltype_minor") + 
  ggtitle("Minor Cell Type (Wu et al.)") +
  guides(color=guide_legend(ncol=3,byrow=TRUE,
                            override.aes = list(size=3))) + 
  theme(legend.position = "bottom")
(p1+p2)/(p3+p4)
dev.off()

# Check the markers
markers_list <- list(
  Tcells = c("CD3D", "CD3E", "CD8A", "CD8B", "CD4", "FOXP3"),
  Bcells = c("CD19", "MS4A1", "CD79A", "IGHM", "IGHG1", "JCHAIN", "MZB1"),
  NKcells = c("NCAM1", "NKG7", "FCGR3A", "PRF1"),
  Macrophages = c("IL1B", "LYZ", "CD68", "CXCL10", "MRC1", "CD163", "FABP5", "NR4A1"),
  DCs = c("LAMP3", "CLEC9A", "CD1C", "CCR7"),
  CAFs = c("COL1A1", "PDGFRB", "POSTN", "SPARC", "CXCL12", "IL6", "AREG", "CCL11"),
  PVLs = c("ACTA2", "MCAM", "RGS5", "ALDH1A1", "THY1"),
  Endothelial = c("VWF", "PECAM1"),
  Epithelial = c("EPCAM", "KRT5", "KRT14", "KRT19", "AREG", "MKI67", "NR4A1")
)

pdf(file.path(dir, "featureplot_markers_Wu_2021.pdf"), width = 15, height = 10)
for (marker_set in names(markers_list)) {
  markers <- markers_list[[marker_set]]
  plots <- FeaturePlot(seurat_Wu, features = markers, combine = FALSE, ncol = 3,
                       label = TRUE)
  plots <- lapply(seq_along(plots), function(i) {
    plots[[i]] + ggtitle(markers[i])
  })
  final_plot <- wrap_plots(plots) +
    plot_annotation(
      title = paste("FeaturePlot -", marker_set),
      theme = theme(
        plot.title = element_text(size = 20, face = "bold", hjust = 0.5) 
      )
    )
  print(final_plot)
}
dev.off()

# Annotate clusters based on the marker genes
cluster_ids <- c(
  "0" = "T Cells",
  "1" = "T Cells CD8+, NK Cells",
  "2" = "Endothelial, CAFs, Macrophages, PVLs",
  "3" = "T Cells CD4+, NK Cells, Macrophages",
  "4" = "CAFs, PVLs",
  "5" = "CAFs, PVLs",
  "6" = "T Cells CD8+, NK Cells",
  "9" = "B Cells",
  "10" = "T Cell Regs",
  "12" = "Normal Epithelial",
  "14" = "T Cells CD4+, NK Cells, Macrophages",
  "16" = "Normal Epithelial",
  "18" = "Plasma B Cells",
  "19" = "T Cells CD8+, NK Cells, Normal Epithelial",
  "22" = "PVLs, CAFs, Normal Epithelial",
  "24" = "CAFs",
  "25" = "Plasma B Cells",
  "27" = "Plasma B Cells",
  "29" = "T Cells",
  "30" = "Endothelial, CAFs, Macrophages, PVLs",
  "31" = "T Cells CD4+, NK Cells, Macrophages",
  "32" = "Plasma B Cells",
  "35" = "Endothelial, CAFs, Macrophages")

seurat_Wu$cell_type <- as.character(seurat_Wu$seurat_clusters) 
seurat_Wu$cell_type <- ifelse(seurat_Wu$cell_type %in% names(cluster_ids),
                              cluster_ids[seurat_Wu$cell_type], 
                              "Cancer Epithelial (Unknown)")
seurat_Wu$cell_type <- factor(seurat_Wu$cell_type)
table(seurat_Wu$cell_type)

pdf(file.path(dir, "dimplots_Wu_2021.pdf"), width = 22, height = 18)
p1 <- DimPlot(seurat_Wu, reduction = "umap", group.by = "cell_type", 
              label = TRUE, label.size = 3) + 
  ggtitle("Cell Type Defined by Markers") +
  guides(color=guide_legend(ncol=2, byrow=F,
                            override.aes = list(size=3))) + 
  theme(legend.position = "bottom")
p2 <- DimPlot(seurat_Wu, reduction = "umap", group.by = "celltype_major",
              label = TRUE, label.size = 3) + 
  ggtitle("Major Cell Type Defined by Wu et al.") +
  guides(color=guide_legend(ncol=4, byrow=F,
                            override.aes = list(size=3))) + 
  theme(legend.position = "bottom")
p3 <- DimPlot(seurat_Wu, reduction = "umap", group.by = "celltype_minor") + 
  ggtitle("Minor Cell Type Defined by Wu et al.") +
  guides(color=guide_legend(ncol=3,byrow=F,
                            override.aes = list(size=3))) + 
  theme(legend.position = "bottom")
p4 <- DimPlot(seurat_Wu, reduction = "umap", label = TRUE) + 
  ggtitle("Cluster") +
  theme(legend.position = "none")
p5 <- DimPlot(seurat_Wu, reduction = "umap", group.by = "orig.ident") + 
  ggtitle("Sample") +
  theme(legend.position = "bottom")
p6 <- DimPlot(seurat_Wu, reduction = "umap", group.by = "subtype") + 
  ggtitle("Cancer Subtype") +
  theme(legend.position = "bottom")
(p4+p5+p6)/(p1+p2+p3)
dev.off()

# Filter cells with high confidence ---------------------------------------

# > table(metadata$seurat_clusters, metadata$celltype_major)
# B-cells  CAFs Cancer Epithelial Endothelial Myeloid Normal Epithelial Plasmablasts   PVL T-cells
# 0       11     0                 2           1       3                 3            2     1   13889
# 1        0     0                 0           0       2                 0            0     0   11654
# 2        3     2                 1        6991      31                 0            1     2      18
# 3       27    19                 4          20    6720                 0            2     3      25
# 4        0  6207                13           1       0                 0            1    38       7
# 5        0   309                22           3       1                 0            2  5356       8
# 6       13     4                 6           3       2                 2            5     2    4158
# 7        0     0              4009           1       0                 5            3     5       3
# 8        0     1              4004           0       1                 2            0     0       4
# 9     3040     0                 0           0      43                 0            6     0      34
# 10       0     1                 0           1       5                 0            0     0    2833
# 11      19    19              2399           1       8                 0            2     1     353
# 12       0     0              1319           0       0              1191            8     0      27
# 13       0     0              2338           0       0                 3            0     0       1
# 14      25     0                 0           0    2179                 0            0     0       2
# 15       0     0              2170           0       2                 3            0     0       4
# 16       2     0                10           0       1              2041            1     0       6
# 17       0     0              1711           0       0                12            8     0       5
# 18      17     0                 0           0       1                 0         1649     0       9
# 19       5     1                 5          21      26                 0           12     1    1383
# 20       0     0              1177           1       0                 0            2     0       8
# 21       0     0              1167           1       0                 0            0     0       1
# 22       0     0                 2           0       0              1085            2     1       3
# 23       3     0              1023           0       0                 2            7     0       2
# 24       0     7              1027           0       1                 0            0     2       0
# 25      29     0                 0           0       0                 0         1004     0       4
# 26       0     0              1031           0       0                 2            0     0       0
# 27      10     1                 1           0       1                 0          807     0      20
# 28       0     0               812           0       0                 1            0     0       0
# 29       0     0                 1           0       0                 0            0     0     597
# 30       0     0                 1         448       0                 0            0    11       0
# 31       1     1                 0           0     344                 0            0     0       5
# 32       1     0                 0           0     302                 0            0     0       0
# 33       0     0               210           0       0                 3            0     0       0
# 34       0     0                 1           0       1                 0            0     0     150
# 35       0     1                 1         112       1                 0            0     0       1
# 36       0     0                22           0       0                 0            0     0       0
#
# T-cells: 0, 1, 6, 10, 19, 29, 34, 
# B cells: 9, 
# CAF: 4, 
# Cancer: 7, 8, 11, 13, 15, 17, 20, 21, 23, 24, 26, 28, 33, 36 (we may not want to use 36 since it is too small).
# Endothelial: 2, 30,
# Myeloid: 3, 14, 31, 32. 
# Normal Epithelial: 16, 22,
# Plasmablasts: 18, 25, 27,
# PVL: 5. 
# We need to recluster the T cells, myeloid, and cells in cluster 12.

sc_dir <- "/fh/working/sun_w/zsui/st2image_data/brca_scRNAseq_datasets"
dir <- file.path(sc_dir, "Wu_2021")

seurat_Wu <- readRDS(file.path(dir, "seurat_Wu_clustered.rds"))
View(seurat_Wu@meta.data)

seurat_Wu$seurat_clusters_init <- seurat_Wu$seurat_clusters

cluster_celltype <- list(
  Endothelial = c(2, 30, 35),
  CAFs = c(4),
  PVL = c(5),
  `B-cells` = c(9),
  `T-cells` = c(0, 1, 6, 10, 19, 29, 34),
  Myeloid = c(3, 14, 31, 32),
  `Normal Epithelial` = c(16, 22),
  Plasmablasts = c(18, 25, 27),
  `Cancer Epithelial` = c(7, 8, 11, 13, 15, 17, 20, 21, 23, 24, 26, 28, 33, 36)
)

seurat_Wu$celltype_init <- NA
for (ct in names(cluster_celltype)) {
  clusters <- cluster_celltype[[ct]]
  seurat_Wu$celltype_init[seurat_Wu$seurat_clusters_init %in% clusters] <- ct
}
seurat_Wu$celltype_init <- ifelse(is.na(seurat_Wu$celltype_init), as.character(seurat_Wu$seurat_clusters_init), seurat_Wu$celltype_init)
View(seurat_Wu@meta.data)

# Recluster Cluster 12 
clusters_to_recluster <- c(12)

# Subset Seurat object
seurat_Wu_12 <- subset(seurat_Wu, idents = clusters_to_recluster)
View(seurat_Wu_12@meta.data)

# Normalize and scale data
seurat_Wu_12 <- NormalizeData(seurat_Wu_12)
seurat_Wu_12 <- FindVariableFeatures(seurat_Wu_12, selection.method = "vst")
seurat_Wu_12 <- ScaleData(seurat_Wu_12)

# Perform PCA
seurat_Wu_12 <- RunPCA(seurat_Wu_12, npcs = 50)

# Run UMAP for visualization
seurat_Wu_12 <- RunUMAP(seurat_Wu_12, dims = 1:50)

# Find neighbors and clusters
seurat_Wu_12 <- FindNeighbors(seurat_Wu_12, dims = 1:50)

resolutions <- seq(0.01, 0.2, 0.01)
for (res in resolutions) {
  seurat_Wu_12 <- FindClusters(seurat_Wu_12, resolution = res)
}

saveRDS(seurat_Wu_12, file = file.path(dir, "seurat_Wu_12.rds"))


pdf(file.path(dir, "dimplots_Wu_2021_cluster12.pdf"), width = 20, height = 15)
p_list <- list()
for (res in resolutions) {
  p_list[[as.character(res)]] <- DimPlot(seurat_Wu_12, reduction = "umap", label = TRUE,
                                         pt.size = 0.1, group.by = paste0("RNA_snn_res.", res)) +
    ggtitle(paste0("Resolution = ", res)) +
    theme(legend.position = "none")
  # DimPlot(seurat_Wu_12, reduction = "umap", label = TRUE,
  #         pt.size = 0.5, group.by = paste0("RNA_snn_res.", res)) +
  #   ggtitle(paste0("Resolution = ", res)) +
  #   theme(legend.position = "none")
}
(p_list[[1]] + p_list[[2]] + p_list[[3]] + p_list[[4]] + p_list[[5]] + 
    p_list[[6]] + p_list[[7]] + p_list[[8]] + p_list[[9]] + p_list[[10]] +
    p_list[[11]] + p_list[[12]] + p_list[[13]] + p_list[[14]] + p_list[[15]] +
    p_list[[16]] + p_list[[17]] + p_list[[18]] + p_list[[19]] + p_list[[20]] ) 
dev.off()

# Check the markers
markers_list <- list(
  Tcells = c("CD3D", "CD3E", "CD8A", "CD8B", "CD4", "FOXP3"),
  Bcells = c("CD19", "MS4A1", "CD79A", "IGHM", "IGHG1", "JCHAIN", "MZB1"),
  NKcells = c("NCAM1", "NKG7", "FCGR3A", "PRF1"),
  Macrophages = c("IL1B", "LYZ", "CD68", "CXCL10", "MRC1", "CD163", "FABP5", "NR4A1"),
  DCs = c("LAMP3", "CLEC9A", "CD1C", "CCR7"),
  CAFs = c("COL1A1", "PDGFRB", "POSTN", "SPARC", "CXCL12", "IL6", "AREG", "CCL11"),
  PVLs = c("ACTA2", "MCAM", "RGS5", "ALDH1A1", "THY1"),
  Endothelial = c("VWF", "PECAM1"),
  Epithelial = c("EPCAM", "KRT5", "KRT14", "KRT19", "AREG", "MKI67", "NR4A1")
)

pdf(file.path(dir, "featureplot_markers_Wu_2021_cluster12.pdf"), width = 15, height = 10)
for (marker_set in names(markers_list)) {
  markers <- markers_list[[marker_set]]
  plots <- FeaturePlot(seurat_Wu_12, features = markers, combine = FALSE, ncol = 3,
                       pt.size = 0.01, label = FALSE )
  plots <- lapply(seq_along(plots), function(i) {
    plots[[i]] + ggtitle(markers[i])
  })
  final_plot <- wrap_plots(plots) +
    plot_annotation(
      title = paste("FeaturePlot -", marker_set),
      theme = theme(
        plot.title = element_text(size = 20, face = "bold", hjust = 0.5) 
      )
    )
  print(final_plot)
}
dev.off()

#### Choose resolution = 0.13
p_sample <- DimPlot(seurat_Wu_12, reduction = "umap", label = F,
                    pt.size = 0.1, group.by ="orig.ident") +
  theme(legend.position = "none")

p_list[["0.13"]] + p_sample

recluster_celltype <- list(
  `T-cells` = c(8),
  `Normal Epithelial` = c(0, 3),
  `Cancer Epithelial` = c(1, 2, 5, 4, 6, 7, 9)
  
)

seurat_Wu_12$celltype_recluster <- NA
for (ct in names(recluster_celltype)) {
  clusters <- recluster_celltype[[ct]]
  seurat_Wu_12$celltype_recluster[seurat_Wu_12$RNA_snn_res.0.13 %in% clusters] <- ct
}

View(seurat_Wu_12@meta.data)

# Add the new annotation of cluster 12 to the overall seurat object
metadata_all <- seurat_Wu@meta.data
metadata_12 <- seurat_Wu_12@meta.data
metadata_all[rownames(metadata_12), "celltype_recluster"] <- metadata_12$celltype_recluster
metadata_all$celltype_recluster <- ifelse(is.na(metadata_all$celltype_recluster), metadata_all$celltype_init, metadata_all$celltype_recluster)
seurat_Wu$celltype_init <- metadata_all$celltype_recluster
View(seurat_Wu@meta.data)


# We want to keep the cells whose cluster membership match to one cell type provided by authors. 

# Select only the relevant cells with pure identity
cells_selected <- rownames(seurat_Wu@meta.data)[seurat_Wu$celltype_init == seurat_Wu$celltype_major]

seurat_Wu_pure <- subset(seurat_Wu, cells = cells_selected)

saveRDS(seurat_Wu_pure, file = file.path(dir, "seurat_Wu_pure.rds"))


# Refine clustering each cell type ------------------------------------------

seurat_Wu <- readRDS(file.path(dir, "seurat_Wu_pure.rds"))
celltypes <- unique(seurat_Wu$celltype_init)

resolutions <- seq(0.02, 0.2, 0.02)
for (ct in celltypes) {
  print(ct)
  cells <- rownames(seurat_Wu@meta.data)[seurat_Wu$celltype_init == ct]
  seurat_Wu_ct <- subset(seurat_Wu, cells = cells)

  # Normalize and scale data
  seurat_Wu_ct_recluster <- NormalizeData(seurat_Wu_ct)
  seurat_Wu_ct_recluster <- FindVariableFeatures(seurat_Wu_ct_recluster, selection.method = "vst")
  seurat_Wu_ct_recluster <- ScaleData(seurat_Wu_ct_recluster)
  
  # Perform PCA
  seurat_Wu_ct_recluster <- RunPCA(seurat_Wu_ct_recluster, npcs = 50)
  
  # Run UMAP for visualization
  seurat_Wu_ct_recluster <- RunUMAP(seurat_Wu_ct_recluster, dims = 1:50)
  
  # Find neighbors and clusters
  seurat_Wu_ct_recluster <- FindNeighbors(seurat_Wu_ct_recluster, dims = 1:50)
  
  for (res in resolutions) {
    seurat_Wu_ct_recluster <- FindClusters(seurat_Wu_ct_recluster, resolution = res)
  }
  
  saveRDS(seurat_Wu_ct_recluster, file = file.path(dir, paste0("seurat_Wu_", ct, ".rds")))
}

pdf(file.path(dir, paste0("dimplots_Wu_2021_ct_sample.pdf")), width = 20, height = 15)
p_list <- list()

for (ct in celltypes) {
  seurat_obj <- readRDS(file.path(dir, paste0("seurat_Wu_", ct, ".rds")))
  p_list[[ct]] <- DimPlot(seurat_obj, reduction = "umap", label = F,
                          pt.size = 0.1, group.by = "orig.ident") +
    ggtitle(ct) +
    theme(legend.position = "none")
}
(p_list[[1]] + p_list[[2]] + p_list[[3]] + p_list[[4]] + p_list[[5]] + 
    p_list[[6]] + p_list[[7]] + p_list[[8]] + p_list[[9]]) 
dev.off()


# Examine markers

markers_list <- list(
  Tcells = c(
    # General T Cell Markers
    "CD3D", "CD3E", "CD3G", "CD3",  # CD3 markers for all T cells
    # Helper T Cells (CD4+ T Cells)
    "CD4", "FOXP3", "IL2RA", "CD25", "CTLA4", "THY1",  # "THY1" is an additional marker for T cells
    # Cytotoxic T Cells (CD8+ T Cells)
    "CD8A", "CD8B", "GZMB", "PRF1", "IFNG", "NKG7",  # "NKG7" also present in cytotoxic T cells
    # Exhausted T Cells
    "PDCD1", "LAG3", "HAVCR2", "TOX", "CTLA4",  # "CTLA4" is also an exhaustion marker
    # Tissue-Resident Memory T Cells (TRM)
    "ITGAE", "CXCR6", "CD69"  # CD69 is another marker for TRM cells
  ),
  Bcells = c(
    # General B Cell Markers
    "CD19", "CD20", "MS4A1", "CD79A", "CD79B", "PAX5", 
    # Activated B Cells
    "CD86", "CD40", "HLA-DRA", "HLA-DRB1", "HLA-DQB1", "ICOS",  # "ICOS" for activated B cells
    # Memory B Cells
    "CD27", "CD38", "BANK1", "CD80",  # "CD80" also found on activated B cells
    # Plasma Cells (Differentiated B Cells)
    "PRDM1", "XBP1", "IGHG1", "JCHAIN", "MZB1",
    # Naïve and Early-Stage B Cells
    "IGHM", "CD43"  
  ),
  NKcells = c(
    "NCAM1", "NKG7", "AREG", "FCGR3A", "PRF1", "NR4A1", 
    # Additional NK cell markers
    "NCR1", "NKp46", "NCR3", "NKp30", "KLRD1", "CD16", "CD94", "KIRs", "GZMB"  # "NCR1" (NKp46), "NCR3" (NKp30), "KLRD1" (CD94) 
  ),
  Macrophages = c(
    # General Macrophage Markers
    "CD68", "CD14", "CSF1R", 
    # M1 (Pro-inflammatory) Macrophages
    "NOS2", "TNF", "IL1B", "IL6", "CD80", "CD86", "HLA-DRA", "IRF5",  
    # M2 (Anti-inflammatory) Macrophages
    "MRC1", "CD206", "CD163", "IL10", "TGFB1", "ARG1", "CHI3L1", "MSR1",
    "IFI30", "HLA-DR", "FABP5", "STAT6" 
  ),
  Monocytes = c(
    "CD14", "CD16", "CD11B", "CSF1R", "CCR2", "CX3CR1", "HLA-DR", 
    "CD80", "CD86", "MHC II", "TNF", "IL1B"
  ),
  DCs = c("LAMP3", "CLEC9A", "CD1C", "CCR7", "CD11C", "HLA-DR"),  
  CAFs = c(
    "TIMP1",  "PDGFRB", "SPARC", "FAP", 
    # iCAFs
    "CXCL12", "IL6", "CCL11", "CXCL14", "FN1", "ALDH1A1", 
    # mCAFs
    "COL1A1", "COL3A", "COL3A1",
    # myCAFs
    "ACTA2", "POSTN", "TAGLN"),
  PVLs = c("ACTA2", "MCAM", "RGS5", "ALDH1A1", "THY1", "ENG", "COL6A3", "CXCL3",
           "MYH11", "PDGFRB", "CNN1"),  # "ENG" for perivascular cells
  Endothelial = c("VWF", "PECAM1", "CD31", "KDR", "VEGFA", "VEGFB", "VEGFC",
                  "VEGFR2", "FLT1", "NRP1", "CD34", "SOX17", "GJA4", "ACKR1",
                  "SELP", "PDPN", "LYVE1", "PROX1", "ACKR4", "CLDN11"), 
  Epithelial = c("EPCAM", "KRT5", "KRT14", "KRT19", "MKI67", "KRT8", "KRT18") 
)


pdf(file.path(dir, "featureplot_markers_Wu_2021_pure.pdf"), width = 15, height = 10)
for (marker_set in names(markers_list)) {
  markers <- markers_list[[marker_set]]
  plots <- FeaturePlot(seurat_Wu, features = markers, combine = FALSE, ncol = 3,
                       pt.size = 0.01, label = FALSE )
  plots <- lapply(seq_along(plots), function(i) {
    plots[[i]] + ggtitle(markers[i])
  })
  final_plot <- wrap_plots(plots) +
    plot_annotation(
      title = paste("FeaturePlot -", marker_set),
      theme = theme(
        plot.title = element_text(size = 20, face = "bold", hjust = 0.5) 
      )
    )
  print(final_plot)
}
dev.off()


DimPlot(seurat_Wu, reduction = "umap", group.by = "celltype_major", label = FALSE) + 
  ggtitle("Cell Type") 


# Endothelial
ct <- "Endothelial"
seurat_obj <- readRDS(file.path(dir, paste0("seurat_Wu_", ct, ".rds")))
pdf(file.path(dir, paste0("dimplots_Wu_2021_", ct, ".pdf")), width = 20, height = 15)
p_list <- list()
for (res in resolutions) {
  p_list[[as.character(res)]] <- DimPlot(seurat_obj, reduction = "umap", label = TRUE,
                                         pt.size = 0.1, group.by = paste0("RNA_snn_res.", res)) +
    ggtitle(paste0("Resolution = ", res)) +
    theme(legend.position = "none")
}
(p_list[[1]] + p_list[[2]] + p_list[[3]] + p_list[[4]] + p_list[[5]] + 
    p_list[[6]] + p_list[[7]] + p_list[[8]] + p_list[[9]] + p_list[[10]]) 
dev.off()

pdf(file.path(dir, paste0("featureplot_markers_Wu_2021_", ct, ".pdf")), width = 15, height = 10)
for (marker_set in names(markers_list)) {
  markers <- unique(markers_list[[marker_set]])
  plots <- FeaturePlot(seurat_obj, features = markers, combine = FALSE, ncol = 3,
                       pt.size = 0.01, label = FALSE )
  final_plot <- wrap_plots(plots) +
    plot_annotation(
      title = paste("FeaturePlot -", marker_set),
      theme = theme(
        plot.title = element_text(size = 20, face = "bold", hjust = 0.5) 
      )
    )
  print(final_plot)
}
dev.off()


# CAFs
ct <- "CAFs"
seurat_obj <- readRDS(file.path(dir, paste0("seurat_Wu_", ct, ".rds")))

pdf(file.path(dir, paste0("dimplots_Wu_2021_", ct, ".pdf")), width = 20, height = 15)
p_list <- list()
for (res in resolutions) {
  p_list[[as.character(res)]] <- DimPlot(seurat_obj, reduction = "umap", label = TRUE,
                                         pt.size = 0.1, group.by = paste0("RNA_snn_res.", res)) +
    ggtitle(paste0("Resolution = ", res)) +
    theme(legend.position = "none")
}
(p_list[[1]] + p_list[[2]] + p_list[[3]] + p_list[[4]] + p_list[[5]] + 
    p_list[[6]] + p_list[[7]] + p_list[[8]] + p_list[[9]] + p_list[[10]]) 
dev.off()

pdf(file.path(dir, paste0("featureplot_markers_Wu_2021_", ct, ".pdf")), width = 15, height = 10)
for (marker_set in names(markers_list)) {
  markers <- unique(markers_list[[marker_set]])
  plots <- FeaturePlot(seurat_obj, features = markers, combine = FALSE, ncol = 3,
                       pt.size = 0.01, label = FALSE )
  final_plot <- wrap_plots(plots) +
    plot_annotation(
      title = paste("FeaturePlot -", marker_set),
      theme = theme(
        plot.title = element_text(size = 20, face = "bold", hjust = 0.5) 
      )
    )
  print(final_plot)
}
dev.off()

# PVL
ct <- "PVL"
seurat_obj <- readRDS(file.path(dir, paste0("seurat_Wu_", ct, ".rds")))

pdf(file.path(dir, paste0("dimplots_Wu_2021_", ct, ".pdf")), width = 20, height = 15)
p_list <- list()
for (res in resolutions) {
  p_list[[as.character(res)]] <- DimPlot(seurat_obj, reduction = "umap", label = TRUE,
                                         pt.size = 0.1, group.by = paste0("RNA_snn_res.", res)) +
    ggtitle(paste0("Resolution = ", res)) +
    theme(legend.position = "none")
}
(p_list[[1]] + p_list[[2]] + p_list[[3]] + p_list[[4]] + p_list[[5]] + 
    p_list[[6]] + p_list[[7]] + p_list[[8]] + p_list[[9]] + p_list[[10]]) 
dev.off()

pdf(file.path(dir, paste0("featureplot_markers_Wu_2021_", ct, ".pdf")), width = 15, height = 10)
for (marker_set in names(markers_list)) {
  markers <- unique(markers_list[[marker_set]])
  plots <- FeaturePlot(seurat_obj, features = markers, combine = FALSE, ncol = 3,
                       pt.size = 0.01, label = FALSE )
  final_plot <- wrap_plots(plots) +
    plot_annotation(
      title = paste("FeaturePlot -", marker_set),
      theme = theme(
        plot.title = element_text(size = 20, face = "bold", hjust = 0.5) 
      )
    )
  print(final_plot)
}
dev.off()

# B-cells
ct <- "B-cells"
seurat_obj <- readRDS(file.path(dir, paste0("seurat_Wu_", ct, ".rds")))

pdf(file.path(dir, paste0("dimplots_Wu_2021_", ct, ".pdf")), width = 20, height = 15)
p_list <- list()
for (res in resolutions) {
  p_list[[as.character(res)]] <- DimPlot(seurat_obj, reduction = "umap", label = TRUE,
                                         pt.size = 0.1, group.by = paste0("RNA_snn_res.", res)) +
    ggtitle(paste0("Resolution = ", res)) +
    theme(legend.position = "none")
}
(p_list[[1]] + p_list[[2]] + p_list[[3]] + p_list[[4]] + p_list[[5]] + 
    p_list[[6]] + p_list[[7]] + p_list[[8]] + p_list[[9]] + p_list[[10]]) 
dev.off()

pdf(file.path(dir, paste0("featureplot_markers_Wu_2021_", ct, ".pdf")), width = 15, height = 10)
for (marker_set in names(markers_list)) {
  markers <- unique(markers_list[[marker_set]])
  plots <- FeaturePlot(seurat_obj, features = markers, combine = FALSE, ncol = 3,
                       pt.size = 0.01, label = FALSE )
  final_plot <- wrap_plots(plots) +
    plot_annotation(
      title = paste("FeaturePlot -", marker_set),
      theme = theme(
        plot.title = element_text(size = 20, face = "bold", hjust = 0.5) 
      )
    )
  print(final_plot)
}
dev.off()

# T-cells
ct <- "T-cells"
seurat_obj <- readRDS(file.path(dir, paste0("seurat_Wu_", ct, ".rds")))

pdf(file.path(dir, paste0("dimplots_Wu_2021_", ct, ".pdf")), width = 20, height = 15)
p_list <- list()
for (res in resolutions) {
  p_list[[as.character(res)]] <- DimPlot(seurat_obj, reduction = "umap", label = TRUE,
                                         pt.size = 0.1, group.by = paste0("RNA_snn_res.", res)) +
    ggtitle(paste0("Resolution = ", res)) +
    theme(legend.position = "none")
}
(p_list[[1]] + p_list[[2]] + p_list[[3]] + p_list[[4]] + p_list[[5]] + 
    p_list[[6]] + p_list[[7]] + p_list[[8]] + p_list[[9]] + p_list[[10]]) 
dev.off()

pdf(file.path(dir, paste0("featureplot_markers_Wu_2021_", ct, ".pdf")), width = 15, height = 10)
for (marker_set in names(markers_list)) {
  markers <- unique(markers_list[[marker_set]])
  plots <- FeaturePlot(seurat_obj, features = markers, combine = FALSE, ncol = 3,
                       pt.size = 0.01, label = FALSE )
  final_plot <- wrap_plots(plots) +
    plot_annotation(
      title = paste("FeaturePlot -", marker_set),
      theme = theme(
        plot.title = element_text(size = 20, face = "bold", hjust = 0.5) 
      )
    )
  print(final_plot)
}
dev.off()


# Myeloid
ct <- "Myeloid"
seurat_obj <- readRDS(file.path(dir, paste0("seurat_Wu_", ct, ".rds")))

pdf(file.path(dir, paste0("dimplots_Wu_2021_", ct, ".pdf")), width = 20, height = 15)
p_list <- list()
for (res in resolutions) {
  p_list[[as.character(res)]] <- DimPlot(seurat_obj, reduction = "umap", label = TRUE,
                                         pt.size = 0.1, group.by = paste0("RNA_snn_res.", res)) +
    ggtitle(paste0("Resolution = ", res)) +
    theme(legend.position = "none")
}
(p_list[[1]] + p_list[[2]] + p_list[[3]] + p_list[[4]] + p_list[[5]] + 
    p_list[[6]] + p_list[[7]] + p_list[[8]] + p_list[[9]] + p_list[[10]]) 
dev.off()

pdf(file.path(dir, paste0("featureplot_markers_Wu_2021_", ct, ".pdf")), width = 15, height = 10)
for (marker_set in names(markers_list)) {
  markers <- unique(markers_list[[marker_set]])
  plots <- FeaturePlot(seurat_obj, features = markers, combine = FALSE, ncol = 3,
                       pt.size = 0.01, label = FALSE )
  final_plot <- wrap_plots(plots) +
    plot_annotation(
      title = paste("FeaturePlot -", marker_set),
      theme = theme(
        plot.title = element_text(size = 20, face = "bold", hjust = 0.5) 
      )
    )
  print(final_plot)
}
dev.off()


# Normal Epithelial
ct <- "Normal Epithelial"
seurat_obj <- readRDS(file.path(dir, paste0("seurat_Wu_", ct, ".rds")))

pdf(file.path(dir, paste0("dimplots_Wu_2021_", ct, ".pdf")), width = 20, height = 15)
p_list <- list()
for (res in resolutions) {
  p_list[[as.character(res)]] <- DimPlot(seurat_obj, reduction = "umap", label = TRUE,
                                         pt.size = 0.1, group.by = paste0("RNA_snn_res.", res)) +
    ggtitle(paste0("Resolution = ", res)) +
    theme(legend.position = "none")
}
(p_list[[1]] + p_list[[2]] + p_list[[3]] + p_list[[4]] + p_list[[5]] + 
    p_list[[6]] + p_list[[7]] + p_list[[8]] + p_list[[9]] + p_list[[10]]) 
dev.off()

pdf(file.path(dir, paste0("featureplot_markers_Wu_2021_", ct, ".pdf")), width = 15, height = 10)
for (marker_set in names(markers_list)) {
  markers <- unique(markers_list[[marker_set]])
  plots <- FeaturePlot(seurat_obj, features = markers, combine = FALSE, ncol = 3,
                       pt.size = 0.01, label = FALSE )
  final_plot <- wrap_plots(plots) +
    plot_annotation(
      title = paste("FeaturePlot -", marker_set),
      theme = theme(
        plot.title = element_text(size = 20, face = "bold", hjust = 0.5) 
      )
    )
  print(final_plot)
}
dev.off()


# Plasmablasts
ct <- "Plasmablasts"
seurat_obj <- readRDS(file.path(dir, paste0("seurat_Wu_", ct, ".rds")))

pdf(file.path(dir, paste0("dimplots_Wu_2021_", ct, ".pdf")), width = 20, height = 15)
p_list <- list()
for (res in resolutions) {
  p_list[[as.character(res)]] <- DimPlot(seurat_obj, reduction = "umap", label = TRUE,
                                         pt.size = 0.1, group.by = paste0("RNA_snn_res.", res)) +
    ggtitle(paste0("Resolution = ", res)) +
    theme(legend.position = "none")
}
(p_list[[1]] + p_list[[2]] + p_list[[3]] + p_list[[4]] + p_list[[5]] + 
    p_list[[6]] + p_list[[7]] + p_list[[8]] + p_list[[9]] + p_list[[10]]) 
dev.off()

pdf(file.path(dir, paste0("featureplot_markers_Wu_2021_", ct, ".pdf")), width = 15, height = 10)
for (marker_set in names(markers_list)) {
  markers <- unique(markers_list[[marker_set]])
  plots <- FeaturePlot(seurat_obj, features = markers, combine = FALSE, ncol = 3,
                       pt.size = 0.01, label = FALSE )
  final_plot <- wrap_plots(plots) +
    plot_annotation(
      title = paste("FeaturePlot -", marker_set),
      theme = theme(
        plot.title = element_text(size = 20, face = "bold", hjust = 0.5) 
      )
    )
  print(final_plot)
}
dev.off()


# Cancer Epithelial
ct <- "Cancer Epithelial"
seurat_obj <- readRDS(file.path(dir, paste0("seurat_Wu_", ct, ".rds")))
pdf(file.path(dir, paste0("dimplots_Wu_2021_", ct, ".pdf")), width = 20, height = 15)
p_list <- list()
for (res in resolutions) {
  p_list[[as.character(res)]] <- DimPlot(seurat_obj, reduction = "umap", label = TRUE,
                                         pt.size = 0.1, group.by = paste0("RNA_snn_res.", res)) +
    ggtitle(paste0("Resolution = ", res)) +
    theme(legend.position = "none")
}
(p_list[[1]] + p_list[[2]] + p_list[[3]] + p_list[[4]] + p_list[[5]] + 
    p_list[[6]] + p_list[[7]] + p_list[[8]] + p_list[[9]] + p_list[[10]]) 
dev.off()

pdf(file.path(dir, paste0("featureplot_markers_Wu_2021_", ct, ".pdf")), width = 15, height = 10)
for (marker_set in names(markers_list)) {
  markers <- unique(markers_list[[marker_set]])
  plots <- FeaturePlot(seurat_obj, features = markers, combine = FALSE, ncol = 3,
                       pt.size = 0.01, label = FALSE )
  final_plot <- wrap_plots(plots) +
    plot_annotation(
      title = paste("FeaturePlot -", marker_set),
      theme = theme(
        plot.title = element_text(size = 20, face = "bold", hjust = 0.5) 
      )
    )
  print(final_plot)
}
dev.off()


all_markers <- unique(stack(markers_list)$values)
celltypes <- c("Cancer Epithelial", "Normal Epithelial", "T-cells", "B-cells", 
               "Myeloid", "Plasmablasts", "PVL", "CAFs", "Endothelial")

pdf(file.path(dir, paste0("featureplot_markers_Wu_2021_all.pdf")), width = 25, height = 15)
p_list <- list()

# Read Seurat objects and store feature plots
for (ct in celltypes) {
  seurat_obj <- readRDS(file.path(dir, paste0("seurat_Wu_", ct, ".rds")))
  
  # Filter markers present in the dataset
  valid_markers <- all_markers[all_markers %in% rownames(seurat_obj)]
  
  if (length(valid_markers) > 0) {
    plots <- FeaturePlot(seurat_obj, features = valid_markers, combine = FALSE, 
                         pt.size = 0.01, label = FALSE)
    
    # Extract marker names directly from the plot titles
    names(plots) <- sapply(plots, function(p) p$labels$title)  
    
    p_list[[ct]] <- plots
  } else {
    p_list[[ct]] <- NULL
  }
}

num_markers <- length(valid_markers)
markers_per_page <- 5

library(gridExtra)
library(grid)

for (i in seq(1, num_markers, by = markers_per_page)) {
  marker_subset <- valid_markers[i:min(i + markers_per_page - 1, num_markers)]
  plot_list <- list()
  
  for (marker in marker_subset) {
    row_plots <- lapply(celltypes, function(ct) {
      if (!is.null(p_list[[ct]]) && marker %in% names(p_list[[ct]])) {
        p_list[[ct]][[marker]] +  # Use marker name directly instead of index
          ggtitle(ct)  + # Add cell type title 
          theme(plot.title = element_text(size = 11))
      } else {
        ggplot() + theme_void() + ggtitle(paste(ct, "- No Data"))  # Placeholder if missing
      }
    })
    
    # Arrange the row with the marker name as a title
    row_plot <- arrangeGrob(
      grobs = row_plots, 
      nrow = 1, 
      top = textGrob(marker, gp = gpar(fontsize = 15, fontface = "bold"))
    )
    
    plot_list <- append(plot_list, list(row_plot))
  }
  
  # Arrange 4 marker rows per page
  grid.arrange(grobs = plot_list, ncol = 1)
}

dev.off()


# Investigate copy number -------------------------------------------------

sc_dir <- "/fh/working/sun_w/zsui/st2image_data/brca_scRNAseq_datasets"
sc_dir <- "/Volumes/sun_w/zsui/st2image_data/brca_scRNAseq_datasets"

dir <- file.path(sc_dir, "Wu_2021")

seurat_Wu <- readRDS(file = file.path(dir, "seurat_Wu_pure.rds"))

library(biomaRt)

ensembl <- useEnsembl(biomart = "genes", dataset = "hsapiens_gene_ensembl")

get_genes_by_chromosome <- function(chr_number) {
  # Retrieve all genes for the specified chromosome
  genes <- getBM(
    attributes = c("hgnc_symbol", "ensembl_gene_id", "chromosome_name", "start_position", "end_position", "band"),
    filters = "chromosome_name",
    values = as.character(chr_number),
    mart = ensembl
  )
  
  # Split into p-arm and q-arm genes
  genes_p <- subset(genes, grepl("^p", genes$band))
  genes_q <- subset(genes, grepl("^q", genes$band))
  
  return(list(p_arm = genes_p, q_arm = genes_q))
}

# Get genes for chromosomes 1, 8, 16,17
genes_chr1 <- get_genes_by_chromosome(1)
genes_chr8 <- get_genes_by_chromosome(8)
genes_chr16 <- get_genes_by_chromosome(16)
genes_chr17 <- get_genes_by_chromosome(17)

# Function to compute CNV scores
# add a small constant (epsilon) 
compute_cnv_score <- function(seurat_obj, genes_gain, genes_loss, epsilon = 1e-6) {
  # Filter out empty or NA gene names
  genes_gain <- genes_gain[genes_gain != "" & !is.na(genes_gain)]
  genes_loss <- genes_loss[genes_loss != "" & !is.na(genes_loss)]
  
  # Fetch gene expression matrix for valid genes
  expr_gain <- as.matrix(FetchData(seurat_obj, vars = genes_gain, layer = "data"))
  expr_loss <- as.matrix(FetchData(seurat_obj, vars = genes_loss, layer = "data"))
  
  # Compute mean expression
  expr_gain.mean <- rowMeans(expr_gain, na.rm = TRUE) + epsilon
  expr_loss.mean <- rowMeans(expr_loss, na.rm = TRUE) + epsilon
  
  # Compute summed expression
  expr_gain.sum <- rowSums(expr_gain, na.rm = TRUE) + epsilon
  expr_loss.sum <- rowSums(expr_loss, na.rm = TRUE) + epsilon
  
  # Compute mean of nonzero values
  expr_gain_nonzero.mean <- rowMeans(expr_gain > 0, na.rm = TRUE) + epsilon
  expr_loss_nonzero.mean <- rowMeans(expr_loss > 0, na.rm = TRUE) + epsilon
  
  # Compute CNV scores
  cnv_score.mean <- expr_gain.mean / expr_loss.mean
  cnv_score.sum <- expr_gain.sum / expr_loss.sum
  cnv_score_nonzero.mean <- expr_gain_nonzero.mean / expr_loss_nonzero.mean
  
  return(list(
    cnv_score.mean = cnv_score.mean,
    cnv_score.sum = cnv_score.sum,
    cnv_score.mean.log10 = log10(cnv_score.mean),
    cnv_score.sum.log10 = log10(cnv_score.sum),
    cnv_score_nonzero.mean = cnv_score_nonzero.mean,
    cnv_score_nonzero.mean.log10 = log10(cnv_score_nonzero.mean)
  ))
}

# Compute CNV scores for each chromosome alteration
cnv_1p1q <- compute_cnv_score(seurat_Wu, genes_gain = genes_chr1$q_arm$hgnc_symbol, genes_loss = genes_chr1$p_arm$hgnc_symbol)
cnv_8p8q <- compute_cnv_score(seurat_Wu, genes_gain = genes_chr8$q_arm$hgnc_symbol, genes_loss = genes_chr8$p_arm$hgnc_symbol)
cnv_16p16q <- compute_cnv_score(seurat_Wu, genes_gain = genes_chr16$p_arm$hgnc_symbol, genes_loss = genes_chr16$q_arm$hgnc_symbol)
cnv_17p17q <- compute_cnv_score(seurat_Wu, genes_gain = genes_chr17$q_arm$hgnc_symbol, genes_loss = genes_chr17$p_arm$hgnc_symbol)


cnv_4chrs <- compute_cnv_score(seurat_Wu, 
                               genes_gain = c(genes_chr1$q_arm$hgnc_symbol,
                                              genes_chr8$q_arm$hgnc_symbol,
                                              genes_chr16$p_arm$hgnc_symbol,
                                              genes_chr17$q_arm$hgnc_symbol), 
                               genes_loss = c(genes_chr1$p_arm$hgnc_symbol,
                                              genes_chr8$p_arm$hgnc_symbol,
                                              genes_chr16$q_arm$hgnc_symbol,
                                              genes_chr17$p_arm$hgnc_symbol))

seurat_Wu$CNV_4chrs_mean <- cnv_4chrs$cnv_score.mean
seurat_Wu$CNV_4chrs_mean.log10 <- cnv_4chrs$cnv_score.mean.log10

summary(seurat_Wu$CNV_4chrs_mean.log10 )

plot_df <- seurat_Wu@meta.data[, c("celltype_major", "CNV_4chrs_mean.log10")]

msd <- plot_df %>% 
  group_by(celltype_major) %>%
  summarise(mean=mean(CNV_4chrs_mean.log10), 
            sd=sd(CNV_4chrs_mean.log10))

ggplot(plot_df, aes(x = CNV_4chrs_mean.log10)) +
  geom_density() +
  facet_wrap(~celltype_major, nrow = 3)+
  ggtitle("log10(mean(gain)/mean(loss))") +
  geom_vline(data=msd, aes(xintercept=mean), color = "red",
             linetype="dashed") + 
  theme(legend.position = "None")


# Add CNV scores to Seurat object
seurat_Wu$CNV_1p1q_mean <- cnv_1p1q$cnv_score.mean
seurat_Wu$CNV_1p1q_sum <- cnv_1p1q$cnv_score.sum
seurat_Wu$CNV_1p1q_nonzero_mean <- cnv_1p1q$cnv_score_nonzero.mean
seurat_Wu$CNV_8p8q_mean <- cnv_8p8q$cnv_score.mean
seurat_Wu$CNV_8p8q_sum <- cnv_8p8q$cnv_score.sum
seurat_Wu$CNV_8p8q_nonzero_mean <- cnv_8p8q$cnv_score_nonzero.mean
seurat_Wu$CNV_16p16q_mean <- cnv_16p16q$cnv_score.mean
seurat_Wu$CNV_16p16q_sum <- cnv_16p16q$cnv_score.sum
seurat_Wu$CNV_16p16q_nonzero_mean <- cnv_16p16q$cnv_score_nonzero.mean
seurat_Wu$CNV_17p17q_mean <- cnv_17p17q$cnv_score.mean
seurat_Wu$CNV_17p17q_sum <- cnv_17p17q$cnv_score.sum
seurat_Wu$CNV_17p17q_nonzero_mean <- cnv_17p17q$cnv_score_nonzero.mean

seurat_Wu$CNV_1p1q_mean.log10 <- cnv_1p1q$cnv_score.mean.log10
seurat_Wu$CNV_1p1q_sum.log10 <- cnv_1p1q$cnv_score.sum.log10
seurat_Wu$CNV_1p1q_nonzero_mean.log10 <- cnv_1p1q$cnv_score_nonzero.mean.log10
seurat_Wu$CNV_8p8q_mean.log10 <- cnv_8p8q$cnv_score.mean.log10
seurat_Wu$CNV_8p8q_sum.log10 <- cnv_8p8q$cnv_score.sum.log10
seurat_Wu$CNV_8p8q_nonzero_mean.log10 <- cnv_8p8q$cnv_score_nonzero.mean.log10
seurat_Wu$CNV_16p16q_mean.log10 <- cnv_16p16q$cnv_score.mean.log10
seurat_Wu$CNV_16p16q_sum.log10 <- cnv_16p16q$cnv_score.sum.log10
seurat_Wu$CNV_16p16q_nonzero_mean.log10 <- cnv_16p16q$cnv_score_nonzero.mean.log10
seurat_Wu$CNV_17p17q_mean.log10 <- cnv_17p17q$cnv_score.mean.log10
seurat_Wu$CNV_17p17q_sum.log10 <- cnv_17p17q$cnv_score.sum.log10
seurat_Wu$CNV_17p17q_nonzero_mean.log10 <- cnv_17p17q$cnv_score_nonzero.mean.log10

saveRDS(seurat_Wu, file = file.path(dir, "seurat_Wu_clustered_cnv.rds"))


library(copykat)
sample_count = seurat_Wu@assays$RNA$counts
sample_mat = as.matrix(sample_count)

copykat.test <- copykat(rawmat=sample_mat, 
                        id.type="S", 
                        cell.line = "no",
                        ngene.chr=5, 
                        win.size=25, 
                        KS.cut=0.1, 
                        distance="euclidean",
                        output.seg="FALSE", 
                        plot.genes="TRUE", 
                        genome="hg20")

saveRDS(copykat.test, 
        file=paste0("copykat_", cur_sample, ".rds"))




# Further investigate epithelial cells ------------------------------------

sc_dir <- "/fh/working/sun_w/zsui/st2image_data/brca_scRNAseq_datasets"
dir <- file.path(sc_dir, "Wu_2021")

seurat_Wu <- readRDS(file = file.path(dir, "seurat_Wu_pure.rds"))
Idents(seurat_Wu) <- "celltype_major"

# seurat for epithelial cancers
seurat_Wu_epi <- subset(seurat_Wu, idents = c("Cancer Epithelial", "Normal Epithelial"))
View(seurat_Wu_epi@meta.data)

resolutions <- seq(0.02, 0.2, 0.02)

# Normalize and scale data
seurat_Wu_epi <- NormalizeData(seurat_Wu_epi)
seurat_Wu_epi <- FindVariableFeatures(seurat_Wu_epi, selection.method = "vst")
seurat_Wu_epi <- ScaleData(seurat_Wu_epi)

# Perform PCA
seurat_Wu_epi <- RunPCA(seurat_Wu_epi, npcs = 50)

# Run UMAP for visualization
seurat_Wu_epi <- RunUMAP(seurat_Wu_epi, dims = 1:50)

# Find neighbors and clusters
seurat_Wu_epi <- FindNeighbors(seurat_Wu_epi, dims = 1:50)

for (res in resolutions) {
  seurat_Wu_epi <- FindClusters(seurat_Wu_epi, resolution = res)
}

saveRDS(seurat_Wu_epi, file = file.path(dir, paste0("seurat_Wu_epithelial.rds")))

# Visualize
pdf(file.path(dir, paste0("dimplots_Wu_2021_epithelial.pdf")), width = 12, height = 4)

p1 <- DimPlot(seurat_Wu_epi, reduction = "umap", label = F,
        pt.size = 0.1, group.by = "orig.ident") +
  theme(legend.position = "none")

p2 <- DimPlot(seurat_Wu_epi, reduction = "umap", label = F,
        pt.size = 0.1, group.by = "subtype") +
  theme(legend.position = "none")

p3 <- DimPlot(seurat_Wu_epi, reduction = "umap", label = F,
        pt.size = 0.1, group.by = "celltype_major") +
  theme(legend.position = "none")
p1+p2+p3
dev.off()


# Identify DE genes

Idents(seurat_Wu_epi) <- "celltype_major"
markers_epi <- FindAllMarkers(seurat_Wu_epi, only.pos = TRUE)

markers_epi <- markers_epi %>%
  group_by(cluster) %>%
  dplyr::filter(avg_log2FC > 1)



# Examine markers
markers <- c("EPCAM", "KRT5", "KRT14", "KRT19", "MKI67", "KRT8", "KRT18", 
             "CNN1", "CCDC8", "NGFR", "COL17A1", "CALML3", "MYH11", "DLK2", "MT1M",
             "PROL1", "PTN", "OXTR", "STAC2", 
             "SERPINB3", "LIN28B", "SPRR1B", "FGA", "LINC00221", "LGALS12", "LINC01285",
             "IGLV7-46", "ISL2", "MAGEA3", "CSAG1", "FGG", "BCAR4"
             )

pdf(file.path(dir, "featureplot_markers_Wu_epithelial.pdf"), width = 15, height = 10)
FeaturePlot(seurat_Wu_epi, features = markers, combine = T, ncol = 6, label = FALSE )
dev.off()






# Also check whether each cancer patient has their own tumor cluster. 
# The most important task we have is how to handle tumor when deconvoluting data. 
# We also need to check the estimation of cell type proportion versus the marker genes. 
# We may incorporate marker genes in the estimation. 
# For tumor, we need to use copy number information using the typical copy number 
# changes we have identified.



# Recluster tumor cells
# Normalize and scale data
seurat_Wu_tumors_recluster <- NormalizeData(seurat_Wu_tumors)
seurat_Wu_tumors_recluster <- FindVariableFeatures(seurat_Wu_tumors_recluster, selection.method = "vst")
seurat_Wu_tumors_recluster <- ScaleData(seurat_Wu_tumors_recluster)

# Perform PCA
seurat_Wu_tumors_recluster <- RunPCA(seurat_Wu_tumors_recluster, npcs = 50)

# Run UMAP for visualization
seurat_Wu_tumors_recluster <- RunUMAP(seurat_Wu_tumors_recluster, dims = 1:50)

# Find neighbors and clusters
seurat_Wu_tumors_recluster <- FindNeighbors(seurat_Wu_tumors_recluster, dims = 1:50)

seurat_Wu_tumors_recluster <- FindClusters(seurat_Wu_tumors_recluster, resolution = 0.005)
seurat_Wu_tumors_recluster <- FindClusters(seurat_Wu_tumors_recluster, resolution = 0.01)
seurat_Wu_tumors_recluster <- FindClusters(seurat_Wu_tumors_recluster, resolution = 0.02)
seurat_Wu_tumors_recluster <- FindClusters(seurat_Wu_tumors_recluster, resolution = 0.03)
seurat_Wu_tumors_recluster <- FindClusters(seurat_Wu_tumors_recluster, resolution = 0.04)
seurat_Wu_tumors_recluster <- FindClusters(seurat_Wu_tumors_recluster, resolution = 0.05)
seurat_Wu_tumors_recluster <- FindClusters(seurat_Wu_tumors_recluster, resolution = 0.1)
seurat_Wu_tumors_recluster <- FindClusters(seurat_Wu_tumors_recluster, resolution = 0.2)
seurat_Wu_tumors_recluster <- FindClusters(seurat_Wu_tumors_recluster, resolution = 0.3)

table( seurat_Wu_tumors_recluster$orig.ident, 
       seurat_Wu_tumors_recluster$RNA_snn_res.0.005)
table( seurat_Wu_tumors_recluster$orig.ident, 
       seurat_Wu_tumors_recluster$RNA_snn_res.0.01)
table( seurat_Wu_tumors_recluster$orig.ident, 
       seurat_Wu_tumors_recluster$RNA_snn_res.0.02)
table( seurat_Wu_tumors_recluster$orig.ident, 
       seurat_Wu_tumors_recluster$RNA_snn_res.0.03)
table( seurat_Wu_tumors_recluster$orig.ident, 
       seurat_Wu_tumors_recluster$RNA_snn_res.0.04)
table( seurat_Wu_tumors_recluster$orig.ident, 
       seurat_Wu_tumors_recluster$RNA_snn_res.0.05)
table( seurat_Wu_tumors_recluster$orig.ident, 
       seurat_Wu_tumors_recluster$RNA_snn_res.0.1)
table( seurat_Wu_tumors_recluster$orig.ident, 
       seurat_Wu_tumors_recluster$RNA_snn_res.0.2)
table( seurat_Wu_tumors_recluster$orig.ident, 
       seurat_Wu_tumors_recluster$RNA_snn_res.0.3)


plot_df <- seurat_Wu_tumors_recluster@meta.data[, c("RNA_snn_res.0.1",
                                                    "CNV_1p1q_nonzero_mean.log10", "CNV_8p8q_nonzero_mean.log10", 
                                                    "CNV_16p16q_nonzero_mean.log10", "CNV_17p17q_nonzero_mean.log10")]
plot_df_long <- reshape::melt(plot_df, 
                              id.vars = "RNA_snn_res.0.1", 
                              variable.name = "ratio")

ggplot(plot_df_long, aes(x = variable, y = value)) +
  geom_boxplot() +
  facet_wrap(~RNA_snn_res.0.1, nrow = 3)








table( seurat_Wu_tumors$orig.ident, 
       seurat_Wu_tumors$seurat_clusters_init)

table( seurat_Wu$orig.ident, 
       seurat_Wu$celltype_major)




# Create heatmap of pesudo-bulk expression in the cluster ----------------
library(dplyr)
library(tibble)
library(Matrix)
library(Matrix.utils)
library(pheatmap)
library(RColorBrewer)

seurat_Wu <- seurat_Wu_pure

# Step 1: Extract raw counts and metadata
counts <- GetAssayData(seurat_Wu, layer = "counts")  # Raw counts
clusters <- seurat_Wu$seurat_clusters  # Cluster assignments
# Check dimensions
length(clusters)  # Should match ncol(counts)
ncol(counts)
names(clusters)
all.equal(colnames(counts), names(clusters))

# Step 2: Sum counts per cluster (Pseudobulk)
counts <- as(counts, "CsparseMatrix")
cluster_matrix <- model.matrix(~ 0 + factor(clusters)) 
colnames(cluster_matrix) <- levels(factor(clusters)) 
pseudobulk_counts <- counts %*% cluster_matrix  # (genes x cells) * (cells x clusters) -> (genes x clusters)
pseudobulk_counts <- Matrix(pseudobulk_counts, sparse = TRUE)
dim(pseudobulk_counts)
saveRDS(pseudobulk_counts, file = file.path(dir, "pseudobulk_counts.rds"))

# Step 3: Normalize pseudobulk expression (log1p TPM)
pseudobulk_norm <- sweep(pseudobulk_counts, 2, colSums(pseudobulk_counts), "/") * 10000  # Normalize
pseudobulk_norm <- log1p(pseudobulk_norm)  # Log transform
pseudobulk_norm <- Matrix(pseudobulk_norm, sparse = TRUE)

# Step 4: Extract marker gene expression from pseudobulk
marker_df <- stack(markers_list)
colnames(marker_df) <- c("Gene", "Cell_Type")
valid_markers <- marker_df$Gene[marker_df$Gene %in% rownames(pseudobulk_norm)]
valid_markers <- unique(valid_markers)
pseudobulk_markers <- pseudobulk_norm[valid_markers, ]
dim(pseudobulk_markers)

# Step 5: Create annotation dataframe for rows (gene-cell type mapping)
annotation_row <- data.frame(Cell_Type = marker_df$Cell_Type[match(rownames(pseudobulk_markers), marker_df$Gene)])
rownames(annotation_row) <- rownames(pseudobulk_markers)
annotation_row <- annotation_row[order(annotation_row$Cell_Type), , drop = FALSE]

# Step 6: Extract metadata for column annotations (cell types per cluster)
metadata <- seurat_Wu@meta.data

annotation_col <- metadata %>%
  select(seurat_clusters, celltype_init) %>%  # Select relevant columns
  distinct() %>%  # Keep unique cluster-cell_type pairs
  remove_rownames() %>%
  column_to_rownames("seurat_clusters")  # Set clusters as row names

colnames(annotation_col) <- "Annotation"
annotation_col <- annotation_col[order(annotation_col$Annotation), , drop = FALSE]
annotation_col <- annotation_col[match(colnames(pseudobulk_markers), rownames(annotation_col)), , drop = FALSE]

# Step 7: Assign colors (after sorting)
n_ct <- length(unique(annotation_row$Cell_Type))
cell_type_colors <- brewer.pal(min(n_ct, 12), "Set3")  # Choose color palette
names(cell_type_colors) <- sort(unique(annotation_row$Cell_Type))  # Assign sorted names to colors

n_ann <- length(unique(annotation_col$Annotation))
cluster_colors <- setNames(colorRampPalette(brewer.pal(12, "Paired"))(n_ann), sort(unique(annotation_col$Annotation)))

annotation_colors <- list(
  Cell_Type = cell_type_colors,
  Annotation = cluster_colors
)

# Step 8: Generate the heatmap with sorted annotations
pdf(file.path(dir, "pseudobulk_heatmap_Wu_2021.pdf"), width = 15, height = 8)

pheatmap(pseudobulk_markers, 
         cluster_rows = TRUE, 
         cluster_cols = TRUE, 
         scale = "row", 
         annotation_row = annotation_row,  # Annotate genes with sorted cell type
         annotation_col = annotation_col,  # Annotate clusters with sorted cell type
         annotation_colors = annotation_colors,  # Assign colors
         show_rownames = TRUE, 
         show_colnames = TRUE, 
         color = colorRampPalette(c("blue", "white", "red"))(100),
         main = "Pseudobulk Marker Gene Expression (Sorted)")
dev.off()







# compare cnv of tumor vs non-tumor ---------------------------------------

plot_df <- seurat_Wu_pure@meta.data[, c("celltype_major",
                                        "CNV_1p1q_mean.log10", "CNV_8p8q_mean.log10", 
                                        "CNV_16p16q_mean.log10", "CNV_17p17q_mean.log10")]



plot_df_long <- reshape::melt(plot_df, 
                              id.vars = "celltype_major")
plot_df_long$celltype <- ifelse(plot_df_long$celltype_major == "Cancer Epithelial", "Tumor", "Non-tumor")


ggplot(plot_df_long, aes(x = value, color = celltype_major, fill = celltype_major)) +
  geom_density() +
  facet_grid(celltype_major~variable)+
  ggtitle("log10(mean(gain)/mean(loss))") +
  theme(legend.position = "None")

ggsave(file.path(dir, "cnv_mean_ratios_log10_9ct.pdf"), width = 12, height = 12)

ggplot(plot_df_long, aes(x = value, color = celltype, fill = celltype)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~variable, nrow = 2) +
  ggtitle("log10(mean(gain)/mean(loss))")

ggsave(file.path(dir, "cnv_mean_ratios_log10_2ct.pdf"), width = 8, height = 6)

celltypes <- unique(plot_df$celltype_major)

pdf(file.path(dir, "cnv_mean_ratios_log10_cor_mat_9ct.pdf"), width = 8, height = 9)
library(plot.matrix)
par(mfrow = c(3,3))
for(ct in celltypes) {
  cor_mtx <- cor(plot_df[plot_df$celltype_major == ct, 2:5])
  colnames(cor_mtx) <- rownames(cor_mtx) <- paste0("Chr", c(1,8, 16, 17))
  
  min_val <- -1
  max_val <- 1
  lim <- max(abs(min_val), abs(max_val))  # Take the largest absolute value
  breaks <- seq(-lim, lim, length.out = 101)  # Symmetric around zero
  color_palette <- colorRampPalette(c("blue", "white", "red"))(100)
  
  par(mar=c(1, 2, 7, 1))
  plot(cor_mtx, digits = 3, 
       axis.col = list(side = 3, cex.axis = 0.8), 
       axis.row = list(cex.axis = 0.8),
       cex = 0.8,
       na.print = FALSE,
       na.cell = FALSE,
       col = color_palette, 
       breaks = breaks,
       key = NULL, 
       main = paste0("Correlation of log10(mean(gain)/mean(loss))\n ", ct),
       cex.main = 1)
  
  }

dev.off()



ggplot(plot_df, aes(x= CNV_1p1q_mean.log10, CNV_8p8q_mean.log10, color = celltype_major)) +
  geom_point()



# Final genes for deconvolution ------------------------------------------------------

markers_list <- list(
  `B cells` = c(
    # Primary Markers: 
    "CD19","MS4A1", "CD79A", "CD79B","BANK1",
    # Secondary Markers: 
    "CD69", "IGHG1", "JCHAIN"
  ),
  `T cells` = c(
    # Primary Markers: 
    "CD3D", "CD3E", "FOXP3", "CD8A", "CD8B", "NKG7", "CD69", 
    # Secondary Markers: 
    "FCGR3A"
  ),
  Plasmablast = c(
    # Primary Markers: 
    "IGHG1", "JCHAIN", "MZB1", 
    # Secondary Markers: 
    "CD79A" 
  ),
  Myeloid = c(
    # Primary Markers:
    "CD86", "FCGR3A", "CD68", "CD14", "CSF1R", "IL1B", 
    "MRC1", "CD163", "MSR1", "IFI30" 
  ),
  CAFs = c(
    # Primary Markers: 
    "COL1A1", "COL3A1", "POSTN", 
    # Secondary Markers: 
    "PDGFRB" 
  ),
  PVL = c(
    # Primary Markers: 
    "PDGFRB", "RGS5", "GJA4", 
    # Secondary Markers: 
    "COL1A1", "COL3A1", "POSTN" 
  ), 
  Endothelial = c(
    # Primary Markers: 
    "VWF", "PECAM1", "KDR", "FLT1", "SOX17", "ACKR1", 
    # Secondary Markers:
    "RGS5", "GJA4"
  ), 
  `Cancer Epithelial` = c(
    # Primary Markers: 
    "EPCAM", "KRT19", "KRT8", "KRT18", 
    # Secondary Markers: 
    "KRT5", "KRT14" 
  ),
  `Normal Epithelial` = c(
    # Primary Markers: 
    "KRT5", "KRT14", 
    # Secondary Markers:
    "EPCAM", "KRT19", "KRT8", "KRT18"
  )
)

markers_final <- stack(markers_list)

# Identify DE genes

sc_dir <- "/fh/working/sun_w/zsui/st2image_data/brca_scRNAseq_datasets"
dir <- file.path(sc_dir, "Wu_2021")

seurat_Wu <- readRDS(file.path(dir, "seurat_Wu_pure.rds"))
celltypes <- unique(seurat_Wu$celltype_major)

Idents(seurat_Wu) <- "celltype_major"
Idents(seurat_Wu)

markers_identified <- FindAllMarkers(seurat_Wu, only.pos = TRUE)
markers_identified <- markers_identified %>%
  group_by(cluster) %>%
  dplyr::filter(avg_log2FC > 1)

saveRDS(markers_identified, file = file.path(dir, "markers_identified.rds"))

# Get reference with the markers and DE genes
seurat_card_ref <- subset(seurat_Wu, features = genes_deconvolution)

# save sparse matrix
ref_dir <- file.path(sc_dir, "CARD_reference", "Wu_2021_markers")
count_matrix <- seurat_card_ref[["RNA"]]$counts
Matrix::writeMM(obj = count_matrix, file = file.path(ref_dir, "count_matrix_sparse.mtx"))

# save genes and cells names and metadata
write(x = rownames(count_matrix), file = file.path(ref_dir, "count_matrix_genes.tsv"))
write(x = colnames(count_matrix), file = file.path(ref_dir, "count_matrix_barcodes.tsv"))

metadata <- seurat_card_ref@meta.data
write.csv(x = metadata, file = file.path(ref_dir, "metadata.csv"))


# Choose the top 50 DE genes

markers_identified <- readRDS(file.path(dir, "markers_identified.rds"))

top50_markers_identified <- markers_identified[!markers_identified$gene %in% markers_final$values, ] %>%
  group_by(cluster) %>%
  dplyr::filter(pct.1 > 0.25) %>%
  arrange(desc(avg_log2FC)) %>%
  slice_head(n = 50) %>%
  ungroup()

length(unique(top50_markers_identified$gene))

genes_deconvolution <- unique(c(markers_final$values, top50_markers_identified$gene))

seurat_Wu <- readRDS(file = file.path(dir, "seurat_Wu_pure.rds"))

# Get reference with the markers and DE genes
seurat_card_ref <- subset(seurat_Wu, features = genes_deconvolution)

# save sparse matrix
ref_dir <- file.path(sc_dir, "CARD_reference", "Wu_2021_markers_DEtop50")
count_matrix <- seurat_card_ref[["RNA"]]$counts
Matrix::writeMM(obj = count_matrix, file = file.path(ref_dir, "count_matrix_sparse.mtx"))

# save genes and cells names and metadata
write(x = rownames(count_matrix), file = file.path(ref_dir, "count_matrix_genes.tsv"))
write(x = colnames(count_matrix), file = file.path(ref_dir, "count_matrix_barcodes.tsv"))

metadata <- seurat_card_ref@meta.data
write.csv(x = metadata, file = file.path(ref_dir, "metadata.csv"))



# Similarity between normal and cancer epithelia --------------------------

seurat_Wu_pure <- readRDS(file = file.path(dir, "seurat_Wu_pure.rds"))

Idents(seurat_Wu_pure) <- "celltype_major"

# Subset the Seurat object by identity
seurat_normal <- subset(seurat_Wu_pure, idents = "Normal Epithelial")
seurat_cancer <- subset(seurat_Wu_pure, idents = "Cancer Epithelial")

marker_genes <- c("EPCAM", "KRT19", "KRT8", "KRT18", "KRT5", "KRT14")

avg_normal <- AverageExpression(seurat_normal, features = marker_genes, assays = "RNA")$RNA
avg_cancer <- AverageExpression(seurat_cancer, features = marker_genes, assays = "RNA")$RNA

df_epithelial <- data.frame(
  gene = marker_genes,
  normal = avg_normal[marker_genes, 1],
  cancer = avg_cancer[marker_genes, 1]
)

df_long <- reshape2::melt(df_epithelial, id.vars = "gene", variable.name = "Cell Type", value.name = "Expression")

ggplot(df_long, aes(x = gene, y = Expression, fill = `Cell Type`)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Marker Gene Expression: Normal Epithelial vs. Cancer Epithelial",
       y = "Log-Normalized Expression", x = "Gene") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

cor(avg_normal[marker_genes, 1], avg_cancer[marker_genes, 1])


avg_normal_by_sample <- AverageExpression(
  seurat_normal,
  features = marker_genes,
  assays = "RNA",
  group.by = "orig.ident"
)$RNA

avg_cancer_by_sample <- AverageExpression(
  seurat_cancer,
  features = marker_genes,
  assays = "RNA",
  group.by = "orig.ident"
)$RNA

df_normal <- as.data.frame(avg_normal_by_sample)
df_normal$gene <- rownames(df_normal)
df_normal_long <- reshape2::melt(df_normal, id.vars = "gene", variable.name = "sample", value.name = "expression")
df_normal_long$`Cell Type` <- "Normal Epithelial"

df_cancer <- as.data.frame(avg_cancer_by_sample)
df_cancer$gene <- rownames(df_cancer)
df_cancer_long <- reshape2::melt(df_cancer, id.vars = "gene", variable.name = "sample", value.name = "expression")
df_cancer_long$`Cell Type` <- "Cancer Epithelial"
df_epithelial_by_sample <- rbind(df_normal_long, df_cancer_long)

library(ggplot2)

ggplot(df_epithelial_by_sample, aes(x = sample, y = expression, fill = `Cell Type`)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~ gene, scales = "free") +
  theme_minimal() +
  labs(title = "Marker Gene Expression by Sample",
       x = "Gene", y = "Log-normalized Expression") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))



# Match samples present in both
common_samples <- intersect(colnames(avg_normal_by_sample), colnames(avg_cancer_by_sample))

cor_results <- sapply(common_samples, function(s) {
  cor(avg_normal_by_sample[marker_genes, s], avg_cancer_by_sample[marker_genes, s], method = "pearson")
})

cor_df <- data.frame(sample = names(cor_results), correlation = cor_results)










