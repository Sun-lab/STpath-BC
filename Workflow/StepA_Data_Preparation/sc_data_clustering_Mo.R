library(Seurat)
library(dplyr)
library(ggplot2)
library(patchwork)
library(SeuratDisk)
library(glmGamPoi)

# The raw scRNA-seq count data contains too many cells, we will do sample filtering. 
# > counts <- table(seurat_Mo$orig.ident)
# > counts
# 
# HT027B1-S1R1                    HT029B1-S1PC                    HT035B1-S1PA                    HT036B1-S1PE                    HT036B1-S2PG 
# 1073807                         2706439                         1855232                         1254907                         1241824 
# HT062B1-S1PA                    HT062B1-S1PB                    HT062B1-S1R1                    HT065B1-S1H1            HT065B1-S1H1A2A1Z1B1 
# 1455201                         1218697                         1242087                         1726406                         1992309 
# HT065B1-S1H7                    HT067B1-S1H2                    HT067B1-S1H5                    HT068B1-S1H1                    HT068B1-S1H2 
# 1521471                         1657583                         1392982                         1519913                         1186431 
# HT077B1-S1H1                    HT077B1-S1H3                    HT077B1-S1H7                    HT084B1-S1H8                    HT088B1-S1H4 
# 1655804                         1714562                         2058586                         1045936                         1312632 
# HT103B1-S1H4                    HT103B1-S1H7                    HT105B1-N1K1                    HT105B1-S1H1                    HT105B1-S1H3 
# 1258553                         1320405                          908113                          838875                         1084349 
# HT110B1-M1                    HT110B1-S1H1                    HT110B1-S1H4          HT110B1-S1H4A2K1G1Z1B1                    HT128B1-S1H3 
# 902377                          949136                         6794880                         1236300                         1904768 
# HT128B1-S1H4                    HT137B1-S1H7                    HT141B1-S1H1                    HT154B1-S1H1                    HT154B1-S1H3 
# 1217405                         2417691                         1556638                         2075027                         2007558 
# HT154B1-S1H5                    HT163B1-S1H2                    HT171B1-S1H1                    HT171B1-S1H8                    HT206B1-S1H4 
# 1359118                         1868535                         2983582                         2387208                         1756855 
# HT214B1-S1H2                    HT217B1-S1H1      HT243B1-H3A2Fc1N1Z1_1Bn1_1                    HT262B1-S1H3                    HT265B1-S1H4 
# 1952539                         2475020                         6794880                         1070884                         2509950 
# HT339B2-S1H2A3Y1Nd1_1Z1_1Bmn1_1    HT365B1-S1H1A3Y1N1Z1_1Bmn1_1    HT397B1-S1H4A4Y1N1Z1_1Bmn1_1    HT425B1-S1H1A3Y1N1Z1_1Bmn1_1    HT486B1-S1H1A3Y1N1Z1_1Bmn1_1 
# 707424                          704816                          725928                          688139                          674326 
# > summary(as.numeric(counts))
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# 674326 1109870 1424092 1719282 1940596 6794880 


sc_dir <- "/fh/working/sun_w/zsui/st2image_data/brca_scRNAseq_datasets"
dir <- file.path(sc_dir, "Mo_2024")
patient_dirs <- list.dirs(dir, recursive = F)
patient_ids <- basename(patient_dirs)

seurat_list_Mo <- list()
error_messages <- list()  # List to store error messages
for (pd1 in patient_dirs) {
  tryCatch({
    p1 <- basename(pd1)
    if(p1 %in% names(seurat_list_Mo)) {
      next
    }
    cat("Processing patient", p1, "\n")
    
    # Load the expression data
    features <- read.delim(file.path(pd1, "features.tsv.gz"), header = FALSE)
    if (ncol(features) == 1) {
      exprs_data <- Read10X(data.dir = pd1, gene.column = 1)
    } else {
      # Load the data
      exprs_data_result <- Read10X(data.dir = pd1)
      
      # Check if the result is a matrix or a list
      if (is.list(exprs_data_result)) {
        # Extract the "Gene Expression" matrix if it exists
        if ("Gene Expression" %in% names(exprs_data_result)) {
          exprs_data <- exprs_data_result[["Gene Expression"]]
        } else {
          stop("Expected 'Gene Expression' data type not found in 10X data.")
        }
      } else {
        # If not a list, it's already a matrix
        exprs_data <- exprs_data_result
      }
    }
    # Create a Seurat object
    p1_seurat <- CreateSeuratObject(counts = exprs_data, project = p1,
                                    min.cells = 3, min.features = 500)
    p1_seurat <- RenameCells(p1_seurat, new.names = paste0(p1, "_", colnames(p1_seurat)))
    nrow(p1_seurat@meta.data)
    
    # Ensure metadata is atomic
    p1_seurat@meta.data <- p1_seurat@meta.data %>%
      mutate(across(everything(), as.character))
    
    # Store the Seurat object
    DefaultAssay(p1_seurat) <- "RNA"
    seurat_list_Mo[[p1]] <- p1_seurat
    
  }, error = function(e) {
    error_messages[[p1]] <- e$message
  })
}
save(seurat_list_Mo, file = file.path(dir, "sc_seurat_list_Mo_2024_min.features.500.rda"))

# Assess the raw data
load(file.path(dir, "sc_seurat_list_Mo_2024_min.features.500.rda"))

metadata <- read.delim(file=file.path(dir, "metadata.tsv"))

gene_list <- list()
gene_meta <- data.frame()
for(s1 in names(seurat_list_Mo)) {
  meta1 <- metadata[grepl(s1,  metadata$Filename), "Filename"]
  parent_dir <- strsplit(meta1[1], "/")[[1]][1]
  n.genes <- nrow(seurat_list_Mo[[s1]]@assays$RNA$counts)
  n.sample <- ncol(seurat_list_Mo[[s1]]@assays$RNA$counts)
  gene_meta <- rbind(gene_meta, c(s1, parent_dir, n.genes, n.sample))
  gene_list[[s1]] <- rownames(seurat_list_Mo[[s1]]@assays$RNA$counts)
}

colnames(gene_meta) <- c("sample", "parent_dir", "n.genes", "n.cells")

unique(gene_meta$parent_dir)

gene_list_atac <- gene_list[gene_meta[gene_meta$parent_dir == "scsnRNA-Seq_level_3_atac_tumor",  "sample"]]
genes_atac <- Reduce(intersect, gene_list_atac)
length(genes_atac)

gene_list_BRCA23 <- gene_list[gene_meta[gene_meta$parent_dir == "scsnRNA-Seq_level_3_BRCA23",  "sample"]]
genes_BRCA23 <- Reduce(intersect, gene_list_BRCA23)
length(genes_BRCA23)

gene_list_spatial_clone <- gene_list[gene_meta[gene_meta$parent_dir == "snRNA-Seq_level_3-spatial_clone",  "sample"]]
genes_spatial_clone <- Reduce(intersect, gene_list_spatial_clone)
length(genes_spatial_clone)

common_genes <- Reduce(intersect, gene_list)
length(common_genes)

# Check the markers
markers_list <- list(
  TumorCore = c("CKB", "VEGFA", "RPS4X", "RPL22", "RPL4"),
  TumorBoundary = c("HSP90AB1", "LDHB", "TUBA1B", "NDUFA4", "TOMM40", 
                    "ENO1", "TMSB10", "ISG15"),
  Tcells = c(
    # General T Cell Markers
    "CD3D", "CD3E", "CD3G", "CD3",
    # Helper T Cells (CD4+ T Cells)
    "CD4", "FOXP3", "IL2RA", "CD25", "CTLA4",
    # Cytotoxic T Cells (CD8+ T Cells)
    "CD8", "CD8A", "CD8B", "GZMB", "PRF1", "IFNG",
    # Exhausted T Cells
    "PDCD1", "LAG3", "HAVCR2", "TOX",
    # Tissue-Resident Memory T Cells (TRM)
    "ITGAE", "CXCR6"),
  Bcells = c(
    # General B Cell Markers
    "CD19", "CD20", "MS4A1", "CD79A", "CD79B", "PAX5", 
    
    # Activated B Cells
    "CD86", "CD40", "HLA-DRA", "HLA-DRB1", "HLA-DQB1",
    
    # Memory B Cells
    "CD27", "CD38", "BANK1",
    
    # Plasma Cells (Differentiated B Cells)
    "PRDM1", "XBP1", "IGHG1", "JCHAIN", "MZB1",
    
    # Naïve and Early-Stage B Cells
    "IGHM"),
  NKcells = c("NCAM1", "NKG7", "AREG", "FCGR3A", "PRF1", "NR4A1"),
  Macrophages = c(
    # General Macrophage Markers
    "CD68", "CD14", "CSF1R",
    
    # M1 (Pro-inflammatory) Macrophages
    "NOS2", "TNF", "IL1B", "IL6", "CD80", "CD86", "HLA-DRA",
    
    # M2 (Anti-inflammatory) Macrophages
    "MRC1", "CD206", "CD163", "IL10", "TGFB1", "ARG1", "CHI3L1", "MSR1",
    
    "IFI30", "HLA-DR", "FABP5"),
  DCs = c("LAMP3", "CLEC9A", "CD1C", "CCR7"),
  CAFs = c("FN1", "TIMP1", "COL1A1", "PDGFRB", "POSTN", "SPARC", "CXCL12", "IL6", "CCL11"),
  PVLs = c("ACTA2", "MCAM", "RGS5", "ALDH1A1", "THY1"),
  Endothelial = c("VWF", "PECAM1"),
  Epithelial = c("EPCAM", "KRT5", "KRT14", "KRT19", "MKI67")
)

# Create a new list containing only markers that exist in common_genes
filtered_markers <- lapply(markers_list, function(markers) {
  intersect(markers, common_genes)
})

filtered_markers_atac <- lapply(markers_list, function(markers) {
  intersect(markers, genes_atac)
})

filtered_markers_BRCA23 <- lapply(markers_list, function(markers) {
  intersect(markers, genes_BRCA23)
})

filtered_markers_spatial_clone <- lapply(markers_list, function(markers) {
  intersect(markers, genes_spatial_clone)
})


# Extract and concatenate counts across all layers
combined_counts <- do.call(cbind, lapply(seurat_list_Mo, function(x) GetAssayData(x, layer = "counts")[common_genes, ]))

# Create a new Seurat object from the combined count matrix
seurat_Mo <- CreateSeuratObject(counts = combined_counts, project = "Mo_2024")

# Merge metadata
metadata_list <- lapply(seurat_list_Mo, function(x) x@meta.data)
combined_metadata <- do.call(rbind, metadata_list)

# Ensure rownames in metadata match the new Seurat object
rownames(combined_metadata) <- colnames(seurat_Mo)
seurat_Mo@meta.data <- combined_metadata

# Set default assay
DefaultAssay(seurat_Mo) <- "RNA"

# Save the combined Seurat object
saveRDS(seurat_Mo, file = file.path(dir, "seurat_combined_Mo_2024_min.features.500.rds"))
seurat_Mo <- readRDS(file.path(dir, "seurat_combined_Mo_2024_min.features.500.rds"))


# Step 1: Normalize the Data
seurat_Mo <- NormalizeData(seurat_Mo)
saveRDS(seurat_Mo, file = file.path(dir, "seurat_Mo_2024_min.features.500_normalized.rds"))

# Step 2: Find Variable Features
seurat_Mo <- FindVariableFeatures(seurat_Mo, selection.method = "vst", nfeatures = 2000)
saveRDS(seurat_Mo, file = file.path(dir, "seurat_Mo_2024_min.features.500_hvgs.rds"))

# Step 3: Scale the Data
# seurat_Mo <- ScaleData(seurat_Mo, do.center = TRUE, do.scale = TRUE)
### Error: cannot allocate vector of size 1281.0 Gb
### Scaling all genes can be memory-intensive, so instead, scale only the most variable features:
seurat_Mo <- ScaleData(seurat_Mo, features = VariableFeatures(seurat_Mo), do.center = TRUE, do.scale = TRUE)
saveRDS(seurat_Mo, file = file.path(dir, "seurat_Mo_2024_min.features.500_scaled.rds"))

# Step 4: Perform PCA
seurat_Mo<- readRDS(file.path(dir, "seurat_Mo_2024_min.features.500_scaled.rds"))
seurat_Mo <- RunPCA(seurat_Mo, npcs = 50)
saveRDS(seurat_Mo, file = file.path(dir, "seurat_Mo_2024_min.features.500_pca.rds"))

# Step 5: Find Neighbors and Clusters
seurat_Mo <- FindNeighbors(seurat_Mo, dims = 1:40)

resolutions <- c(0.2, 0.3, 0.5, 0.8, 1, 1.2)
for (res in resolutions) {
  print(res)
  if (paste0("RNA_snn_res.", res) %in% colnames(seurat_Mo@meta.data)) {
    next
  }
  seurat_Mo <- FindClusters(seurat_Mo, resolution = res)
}

# Step 6: Run UMAP for Visualization
seurat_Mo <- RunUMAP(seurat_Mo, dims = 1:40)
saveRDS(seurat_Mo, file = file.path(dir, "seurat_Mo_2024_min.features.500_clustered.rds"))

# Step 7: Visualize Clusters

# Add parent directory to the metadata of the seurat object

metadata <- read.delim(file=file.path(dir, "metadata.tsv"))
parent_dir_mapping <- data.frame(
  filename = sapply(strsplit(metadata[, "Filename"], "/"), "[[", 2),
  parent_dir = sapply(strsplit(metadata[, "Filename"], "/"), "[[", 1)
)

parent_dir_mapping$filename <- gsub("-features.tsv.gz", "", parent_dir_mapping$filename)
parent_dir_mapping$filename <- gsub("-matrix.mtx.gz", "", parent_dir_mapping$filename)
parent_dir_mapping$filename <- gsub("-barcodes.tsv.gz", "", parent_dir_mapping$filename)

parent_dir_mapping <- unique(parent_dir_mapping)

# Add parent dir to the seurat object
metadata <- seurat_Mo@meta.data
metadata$subject <- sapply(strsplit(metadata$orig.ident, "-"), "[[", 1)
metadata$X <- rownames(metadata)

metadata <- merge(metadata, parent_dir_mapping, by.x = "orig.ident", by.y = "filename")
rownames(metadata) <- metadata$X 

seurat_Mo@meta.data <- metadata

pdf(file.path(dir, "sample_vs_cluster_Mo_2024.pdf"), width = 25, height = 15)
p_list <- list()

for (res in resolutions) {
  count_df <- as.data.frame(table(metadata$orig.ident, metadata[, paste0("RNA_snn_res.", res) ]))
  colnames(count_df) <- c("Sample", "Cluster", "Count")
  count_df$Cluster <- as.character(count_df$Cluster)
  p_list[[as.character(res)]] <- ggplot(count_df, aes(x = Sample, y = Count, fill = Sample)) +
    geom_bar(stat = "identity") +
    facet_wrap(~ as.numeric(Cluster), scales = "free_y", ncol = 8) +
    theme_minimal() +
    theme(legend.position = "bottom",
          axis.text.x=element_blank()) +
    labs(x = "Sample", y = "Count", title = paste0("Cell Count by Sample and Cluster (Resolution = ", res, ")"))
}

(p_list[[1]] + p_list[[2]] + p_list[[3]] ) / (  p_list[[4]] + p_list[[5]] +  p_list[[6]] ) + 
  plot_layout(guides = "collect") & 
  theme(legend.position = "bottom",
        legend.text = element_text(size = 8),
        legend.title = element_blank())

dev.off()


pdf(file.path(dir, "subject_vs_cluster_Mo_2024.pdf"), width = 25, height = 15)
p_list <- list()

for (res in resolutions) {
  count_df <- as.data.frame(table(metadata$subject, metadata[, paste0("RNA_snn_res.", res) ]))
  colnames(count_df) <- c("Sample", "Cluster", "Count")
  count_df$Cluster <- as.character(count_df$Cluster)
  p_list[[as.character(res)]] <- ggplot(count_df, aes(x = Sample, y = Count, fill = Sample)) +
    geom_bar(stat = "identity") +
    facet_wrap(~ as.numeric(Cluster), scales = "free_y", ncol = 8) +
    theme_minimal() +
    theme(legend.position = "bottom",
          axis.text.x=element_blank()) +
    labs(x = "Subject", y = "Count", title = paste0("Cell Count by Subject and Cluster (Resolution = ", res, ")"))
}

(p_list[[1]] + p_list[[2]] + p_list[[3]] ) / (  p_list[[4]] + p_list[[5]] +  p_list[[6]] ) + 
  plot_layout(guides = "collect") & 
  theme(legend.position = "bottom",
        legend.text = element_text(size = 8),
        legend.title = element_blank())

dev.off()


pdf(file.path(dir, "dimplots_source_Mo_2024.pdf"), width = 10, height = 10)
DimPlot(seurat_Mo, reduction = "umap", label = F, group.by = "parent_dir") +
  ggtitle("Source of scRNA-seq data") +
  theme(legend.position = "bottom")
dev.off()



pdf(file.path(dir, "dimplots_sample_Mo_2024.pdf"), width = 15, height = 15)
p1 <- DimPlot(seurat_Mo, reduction = "umap", label = TRUE, 
              pt.size = 0.5, group.by = "RNA_snn_res.0.2") +
  ggtitle("RNA_snn_res.0.2") +
  theme(legend.position = "none")
p2 <- DimPlot(seurat_Mo, reduction = "umap", label = TRUE, 
              pt.size = 0.5, group.by = "RNA_snn_res.0.3") +
  ggtitle("RNA_snn_res.0.3") +
  theme(legend.position = "none")
p3 <- DimPlot(seurat_Mo, reduction = "umap", label = TRUE, 
              pt.size = 0.5, group.by = "RNA_snn_res.0.5") +
  ggtitle("RNA_snn_res.0.5") +
  theme(legend.position = "none")
p4 <- DimPlot(seurat_Mo, reduction = "umap", group.by = "orig.ident") + 
  ggtitle("Sample") +
  theme(legend.position = "bottom")
(p1+p2) / (p3+p4)
dev.off()



# Check the markers
markers_list <- list(
  TumorCore = c("CKB", "VEGFA", "RPS4X", "RPL22", "RPL4"),
  TumorBoundary = c("HSP90AB1", "LDHB", "TUBA1B", "NDUFA4", "TOMM40", 
                    "ENO1", "TMSB10", "ISG15"),
  Tcells = c(
    # General T Cell Markers
    "CD3D", "CD3E", "CD3G", "CD3",
    # Helper T Cells (CD4+ T Cells)
    "CD4", "FOXP3", "IL2RA", "CD25", "CTLA4",
    # Cytotoxic T Cells (CD8+ T Cells)
    "CD8", "CD8A", "CD8B", "GZMB", "PRF1", "IFNG",
    # Exhausted T Cells
    "PDCD1", "LAG3", "HAVCR2", "TOX",
    # Tissue-Resident Memory T Cells (TRM)
    "ITGAE", "CXCR6"),
  Bcells = c(
    # General B Cell Markers
    "CD19", "CD20", "MS4A1", "CD79A", "CD79B", "PAX5", 
    
    # Activated B Cells
    "CD86", "CD40", "HLA-DRA", "HLA-DRB1", "HLA-DQB1",
    
    # Memory B Cells
    "CD27", "CD38", "BANK1",
    
    # Plasma Cells (Differentiated B Cells)
    "PRDM1", "XBP1", "IGHG1", "JCHAIN", "MZB1",
    
    # Naïve and Early-Stage B Cells
    "IGHM"),
  NKcells = c("NCAM1", "NKG7", "AREG", "FCGR3A", "PRF1", "NR4A1"),
  Macrophages = c(
    # General Macrophage Markers
    "CD68", "CD14", "CSF1R",
    
    # M1 (Pro-inflammatory) Macrophages
    "NOS2", "TNF", "IL1B", "IL6", "CD80", "CD86", "HLA-DRA",
    
    # M2 (Anti-inflammatory) Macrophages
    "MRC1", "CD206", "CD163", "IL10", "TGFB1", "ARG1", "CHI3L1", "MSR1",
    
    "IFI30", "HLA-DR", "FABP5"),
  DCs = c("LAMP3", "CLEC9A", "CD1C", "CCR7"),
  CAFs = c("FN1", "TIMP1", "COL1A1", "PDGFRB", "POSTN", "SPARC", "CXCL12", "IL6", "CCL11"),
  PVLs = c("ACTA2", "MCAM", "RGS5", "ALDH1A1", "THY1"),
  Endothelial = c("VWF", "PECAM1"),
  Epithelial = c("EPCAM", "KRT5", "KRT14", "KRT19", "MKI67")
)


pdf(file.path(dir, "featureplot_markers_Mo_2024.pdf"), width = 15, height = 8)
for (marker_set in names(markers_list)) {
  markers <- markers_list[[marker_set]]
  plots <- FeaturePlot(seurat_Mo, features = markers, combine = FALSE, ncol = 3,
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



# Investigate BRAC23 ------------------------------------------------------

load(file.path(dir, "sc_seurat_list_Mo_2024_min.features.500.rda"))

# Extract BRCA23
metadata <- read.delim(file=file.path(dir, "metadata.tsv"))
parent_dir_mapping <- data.frame(
  filename = sapply(strsplit(metadata[, "Filename"], "/"), "[[", 2),
  parent_dir = sapply(strsplit(metadata[, "Filename"], "/"), "[[", 1)
)

parent_dir_mapping$filename <- gsub("-features.tsv.gz", "", parent_dir_mapping$filename)
parent_dir_mapping$filename <- gsub("-matrix.mtx.gz", "", parent_dir_mapping$filename)
parent_dir_mapping$filename <- gsub("-barcodes.tsv.gz", "", parent_dir_mapping$filename)

parent_dir_mapping <- unique(parent_dir_mapping)
seurat_list_Mo_BRCA23 <- seurat_list_Mo[parent_dir_mapping[parent_dir_mapping$parent_dir == "scsnRNA-Seq_level_3_BRCA23", "filename"]]
seurat_list_Mo_BRCA23 <- seurat_list_Mo_BRCA23[!sapply(seurat_list_Mo_BRCA23, is.null)]

# Extract and concatenate counts across all layers
combined_counts <- do.call(cbind, lapply(seurat_list_Mo_BRCA23, function(x) GetAssayData(x, layer = "counts")))


# Create a new Seurat object from the combined count matrix
seurat_Mo <- CreateSeuratObject(counts = combined_counts, project = "Mo_2024")

# Merge metadata
metadata_list <- lapply(seurat_list_Mo, function(x) x@meta.data)
combined_metadata <- do.call(rbind, metadata_list)

# Ensure rownames in metadata match the new Seurat object
rownames(combined_metadata) <- colnames(seurat_Mo)
seurat_Mo@meta.data <- combined_metadata

# Set default assay
DefaultAssay(seurat_Mo) <- "RNA"

# Save the combined Seurat object
saveRDS(seurat_Mo, file = file.path(dir, "seurat_combined_Mo_2024_min.features.500.rds"))



# Add parent dir to the seurat object
metadata <- seurat_Mo@meta.data
metadata$subject <- sapply(strsplit(metadata$orig.ident, "-"), "[[", 1)
metadata$X <- rownames(metadata)

metadata <- merge(metadata, parent_dir_mapping, by.x = "orig.ident", by.y = "filename")
rownames(metadata) <- metadata$X 

seurat_Mo@meta.data <- metadata

Idents(seurat_Mo) <- "parent_dir"
seurat_Mo$parent_dir
seurat_Mo_BRCA23 <- subset(seurat_Mo, idents = levels(Idents(seurat_Mo))[1])
View(seurat_Mo_BRCA23@meta.data)

# Process 
seurat_Mo_BRCA23 <- NormalizeData(seurat_Mo_BRCA23)
seurat_Mo_BRCA23 <- FindVariableFeatures(seurat_Mo_BRCA23, selection.method = "vst")
seurat_Mo_BRCA23 <- ScaleData(seurat_Mo_BRCA23)
seurat_Mo_BRCA23 <- RunPCA(seurat_Mo_BRCA23, npcs = 50)
seurat_Mo_BRCA23 <- RunUMAP(seurat_Mo_BRCA23, dims = 1:50)
seurat_Mo_BRCA23 <- FindNeighbors(seurat_Mo_BRCA23, dims = 1:50)
resolutions <- seq(0.1, 0.5, 0.1)
for (res in resolutions) {
  seurat_Mo_BRCA23 <- FindClusters(seurat_Mo_BRCA23, resolution = res)
}


pdf(file.path(dir, "featureplot_markers_Mo_2024_BRCA23.pdf"), width = 15, height = 8)
for (marker_set in names(markers_list)) {
  markers <- markers_list[[marker_set]]
  plots <- FeaturePlot(seurat_Mo_BRCA23, features = markers, combine = FALSE, ncol = 3,
                       label = TRUE)
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





# Investigate copy number -------------------------------------------------

sc_dir <- "/fh/working/sun_w/zsui/st2image_data/brca_scRNAseq_datasets"
dir <- file.path(sc_dir, "Mo_2024")

seurat_Mo <- readRDS(file = file.path(dir, "seurat_Mo_2024_min.features.500_clustered.rds"))

# Load required package
library(biomaRt)

# Connect to Ensembl
ensembl <- useEnsembl(biomart = "genes", dataset = "hsapiens_gene_ensembl")

# Function to retrieve genes from a given chromosome and separate into p/q arms
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
# add a small constant (epsilon) to the denominator
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
  
  # Compute CNV scores
  cnv_score.mean <- expr_gain.mean / expr_loss.mean
  cnv_score.sum <- expr_gain.sum / expr_loss.sum
  
  return(list(cnv_score.mean = cnv_score.mean, cnv_score.sum = cnv_score.sum,
              cnv_score.mean.log10 = log10(cnv_score.mean), cnv_score.sum.log10 = log10(cnv_score.sum)))
}

# Compute CNV scores for each chromosome alteration
cnv_1p1q <- compute_cnv_score(seurat_Mo, genes_gain = genes_chr1$q_arm$hgnc_symbol, genes_loss = genes_chr1$p_arm$hgnc_symbol)
cnv_8p8q <- compute_cnv_score(seurat_Mo, genes_gain = genes_chr8$q_arm$hgnc_symbol, genes_loss = genes_chr8$p_arm$hgnc_symbol)
cnv_16p16q <- compute_cnv_score(seurat_Mo, genes_gain = genes_chr16$p_arm$hgnc_symbol, genes_loss = genes_chr16$q_arm$hgnc_symbol)
cnv_17p17q <- compute_cnv_score(seurat_Mo, genes_gain = genes_chr17$q_arm$hgnc_symbol, genes_loss = genes_chr17$p_arm$hgnc_symbol)

# Add CNV scores to Seurat object
seurat_Mo$CNV_1p1q_mean <- cnv_1p1q$cnv_score.mean
seurat_Mo$CNV_1p1q_sum <- cnv_1p1q$cnv_score.sum
seurat_Mo$CNV_8p8q_mean <- cnv_8p8q$cnv_score.mean
seurat_Mo$CNV_8p8q_sum <- cnv_8p8q$cnv_score.sum
seurat_Mo$CNV_16p16q_mean <- cnv_16p16q$cnv_score.mean
seurat_Mo$CNV_16p16q_sum <- cnv_16p16q$cnv_score.sum
seurat_Mo$CNV_17p17q_mean <- cnv_17p17q$cnv_score.mean
seurat_Mo$CNV_17p17q_sum <- cnv_17p17q$cnv_score.sum

seurat_Mo$CNV_1p1q_mean.log10 <- cnv_1p1q$cnv_score.mean.log10
seurat_Mo$CNV_1p1q_sum.log10 <- cnv_1p1q$cnv_score.sum.log10
seurat_Mo$CNV_8p8q_mean.log10 <- cnv_8p8q$cnv_score.mean.log10
seurat_Mo$CNV_8p8q_sum.log10 <- cnv_8p8q$cnv_score.sum.log10
seurat_Mo$CNV_16p16q_mean.log10 <- cnv_16p16q$cnv_score.mean.log10
seurat_Mo$CNV_16p16q_sum.log10 <- cnv_16p16q$cnv_score.sum.log10
seurat_Mo$CNV_17p17q_mean.log10 <- cnv_17p17q$cnv_score.mean.log10
seurat_Mo$CNV_17p17q_sum.log10 <- cnv_17p17q$cnv_score.sum.log10


boxplot(seurat_Mo$CNV_1p1q_mean)
boxplot(seurat_Mo$CNV_1p1q_sum)
boxplot(seurat_Mo$CNV_8p8q_mean)
boxplot(seurat_Mo$CNV_8p8q_sum)
boxplot(seurat_Mo$CNV_16p16q_mean)
boxplot(seurat_Mo$CNV_16p16q_sum)
boxplot(seurat_Mo$CNV_17p17q_mean)
boxplot(seurat_Mo$CNV_17p17q_sum)

boxplot(seurat_Mo$CNV_1p1q_mean.log10)
boxplot(seurat_Mo$CNV_1p1q_sum.log10)
boxplot(seurat_Mo$CNV_8p8q_mean.log10)
boxplot(seurat_Mo$CNV_8p8q_sum.log10)
boxplot(seurat_Mo$CNV_16p16q_mean.log10)
boxplot(seurat_Mo$CNV_16p16q_sum.log10)
boxplot(seurat_Mo$CNV_17p17q_mean.log10)
boxplot(seurat_Mo$CNV_17p17q_sum.log10)
View(seurat_Mo@meta.data)

FeaturePlot(seurat_Mo, features = c("CNV_1p1q_mean.log10", "CNV_8p8q_mean.log10", 
                                    "CNV_16p16q_mean.log10", "CNV_17p17q_mean.log10"),
            cols = c("lightgray", "blue"), reduction = "umap")

FeaturePlot(seurat_Mo, features = c("CNV_1p1q_sum.log10", "CNV_8p8q_sum.log10", "CNV_16p16q_sum.log10", "CNV_17p17q_sum.log10"),
            cols = c("lightgray", "red"), reduction = "umap")



pdf(file.path(dir, "featureplot_cnv_Mo_2024.pdf"), width = 15, height = 8)

for (marker_set in names(markers_list)) {
  markers <- markers_list[[marker_set]]
  plots <- FeaturePlot(seurat_Mo, features = markers, combine = FALSE, ncol = 3,
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










### Still had the limited memory error.

### Therefore, instead of using all cells, we only randomly subsample 10% of cells from each sample and run the analysis.

seurat_Mo <- readRDS(file.path(dir, "seurat_combined_Mo_2024.rds"))

# Create a table of counts
counts <- table(seurat_Mo$orig.ident)
counts
summary(as.numeric(counts))

set.seed(123)  
samples <- unique(seurat_Mo$orig.ident)
selected_cells <- c()
for (sample in samples) {
  sample_cells <- Cells(seurat_Mo)[seurat_Mo$orig.ident == sample]
  num_to_sample <- max(1, round(length(sample_cells) * 0.10))  
  selected_cells <- c(selected_cells, sample(sample_cells, num_to_sample))
}

seurat_Mo <- subset(seurat_Mo, cells = selected_cells)
saveRDS(seurat_Mo, file = file.path(dir, "seurat_Mo_2024_downsampled_10percent.rds"))

seurat_Mo <- NormalizeData(seurat_Mo)
saveRDS(seurat_Mo, file = file.path(dir, "seurat_Mo_2024_downsampled_10percent_normalized.rds"))

seurat_Mo <- FindVariableFeatures(seurat_Mo, selection.method = "vst", nfeatures = 2000)
saveRDS(seurat_Mo, file = file.path(dir, "seurat_Mo_2024_downsampled_10percent_hvgs.rds"))

seurat_Mo <- ScaleData(seurat_Mo, do.center = TRUE, do.scale = TRUE)
saveRDS(seurat_Mo, file = file.path(dir, "seurat_Mo_2024_downsampled_10percent_scaled.rds"))

# Step 4: Perform PCA
seurat_Mo <- RunPCA(seurat_Mo, npcs = 50, verbose = FALSE)
saveRDS(seurat_Mo, file = file.path(dir, "seurat_Mo_2024_downsampled_10percent_pca.rds"))

# Step 5: Find Neighbors and Clusters
library(future)
options(future.globals.maxSize = 20 * 1024^3)  # Increase to 20GB if possible
seurat_Mo <- FindNeighbors(seurat_Mo, dims = 1:10)
seurat_Mo <- FindClusters(seurat_Mo, resolution = 0.5)  # Adjust resolution if needed

# Step 6: Run UMAP for Visualization
seurat_Mo <- RunUMAP(seurat_Mo, dims = 1:50)
saveRDS(seurat_Mo, file = file.path(dir, "seurat_Mo_2024_downsampled_10percent_clustered.rds"))


# Step 7: Visualize Clusters
DimPlot(seurat_Mo, reduction = "umap", label = TRUE, pt.size = 0.5)

# Step 8: Identify Marker Genes for Each Cluster
markers <- FindAllMarkers(seurat_Mo, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)

# Step 9: Save Marker Genes
write.csv(markers, file = file.path(dir, "marker_genes.csv"), row.names = FALSE)

# Step 10: Visualize Top Marker Genes
top_markers <- markers %>% group_by(cluster) %>% top_n(n = 10, wt = avg_log2FC)
DoHeatmap(seurat_Mo, features = top_markers$gene) + NoLegend()

# seurat_Mo <- Reduce(function(x, y) merge(x, y), seurat_list_Mo)
# seurat_Mo@meta.data <- seurat_Mo@meta.data %>%
#   mutate(across(everything(), ~ ifelse(is.null(.), NA, .)))
# 
# rownames(seurat_Mo@meta.data) <- make.names(rownames(seurat_Mo@meta.data), unique = TRUE)
# 
# save(seurat_Mo, file = file.path(dir, "scRNAseq_seurat_Mo_2024.rda"))
# 
# SaveH5Seurat(seurat_Mo, filename = file.path(dir, "scRNAseq_seurat_Mo_2024.h5seurat"), overwrite = TRUE)


load(file.path(dir, "scRNAseq_seurat_Mo_2024.rda"))
seurat_Mo <- NormalizeData(seurat_Mo)
save(seurat_Mo, file = file.path(dir, "scRNAseq_seurat_Mo_2024_normalized.rda"))

load(file.path(dir, "scRNAseq_seurat_Mo_2024_normalized.rda"))
seurat_Mo <- FindVariableFeatures(seurat_Mo, selection.method = "vst", nfeatures = 2000)
save(seurat_Mo, file = file.path(dir, "scRNAseq_seurat_Mo_2024_hvgs.rda"))

top10 <- head(VariableFeatures(seurat_Mo), 10)
plot1 <- VariableFeaturePlot(seurat_Mo)
plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
plot2

all.genes <- rownames(seurat_Mo)
seurat_Mo <- ScaleData(seurat_Mo, features = all.genes)

seurat_Mo <- RunPCA(seurat_Mo, features = VariableFeatures(object = seurat_Mo))
print(seurat_Mo[["pca"]], dims = 1:5, nfeatures = 5)
VizDimLoadings(seurat_Mo, dims = 1:2, reduction = "pca")
DimPlot(seurat_Mo, reduction = "pca") + NoLegend()
DimHeatmap(seurat_Mo, dims = 1, cells = 500, balanced = TRUE)
DimHeatmap(seurat_Mo, dims = 1:15, cells = 500, balanced = TRUE)

seurat_Mo <- FindNeighbors(seurat_Mo, dims = 1:30)
seurat_Mo <- FindClusters(seurat_Mo, resolution = 0.5)

seurat_Mo <- RunUMAP(seurat_Mo, dims = 1:30)
DimPlot(seurat_Mo, reduction = "umap")
DimPlot(seurat_Mo, reduction = "umap", group.by = "orig.ident")

save(seurat_Mo, file = file.path(dir, "seurat_Wu_umap.rda"))

# find markers for every cluster compared to all remaining cells, report only the positive
# ones
seurat_Mo.markers <- FindAllMarkers(seurat_Mo, only.pos = TRUE)
seurat_Mo.markers <- seurat_Mo.markers %>%
  group_by(cluster) %>%
  dplyr::filter(avg_log2FC > 1)

save(seurat_Mo.markers, file = file.path(dir, "seurat_Mo_identified_markers.rda"))

markers <- c("CD3D", "CD3E", "CD8A", "CD8B", "CD4", 
             "FOXP3", "CD19", "MS4A1", "CD79A", 
             "IGHM", "IGHG1", "JCHAIN", "MZB1", 
             "NCAM1" , "NKG7" , "AREG") 
pdf(file.path(dir, "violinplot_markers.pdf"), width = 20, height = 20)
VlnPlot(seurat_Wu, features = markers)
dev.off()

pdf(file.path(dir, "featureplot_markers.pdf"), width = 25, height = 25)
FeaturePlot(seurat_Wu, features = markers)
dev.off()





# Ensure unique cell names and preprocess each Seurat object
for (i in seq_along(seurat_list)) {
  
  
  
  # Normalize data
  seurat_list[[i]] <- NormalizeData(seurat_list[[i]])
  # Find variable features
  seurat_list[[i]] <- FindVariableFeatures(seurat_list[[i]], selection.method = "vst", nfeatures = 2000)
  
  
}




# Remove invalid Seurat objects
valid_objects <- sapply(seurat_list, function(obj) {
  !is.null(dim(obj)) &&
    length(VariableFeatures(obj)) > 0 &&
    DefaultAssay(obj) == "RNA" &&
    nrow(obj@assays$RNA@layers$data) > 0
})
seurat_list <- seurat_list[valid_objects]

# Integration across slides -----------------------------------------------

anchors_list <- list()
load(file.path(st_dir, "anchors_list.rda"))
error_messages <- list()  # List to store error messages
datasets_interested <- c("10xGenomics", "Wu_2021", "Mo_2024", "Bassiouni_2023") 
for (dataset in datasets_interested) {
  tryCatch({
    if(dataset %in% names(anchors_list)) {
      next
    }
    cat(dataset, "\n")
    anchors_list[[dataset]] <- FindIntegrationAnchors(
      object.list = seurat_list[startsWith(names(seurat_list), dataset)], 
      dims = 1:20
    )
  }, error = function(e) {
    error_messages[[dataset]] <- e$message
  })
}

# After the loop, you can inspect the recorded error messages
print(error_messages)

save(anchors_list, file = file.path(st_dir, "anchors_list.rda"))



integrated_data_list <- list()
for (dataset in names(anchors_list)) {
  integrated_data_list[[dataset]] <- IntegrateData(anchorset = anchors_list[[dataset]])
}
save(integrated_data_list, file = file.path(st_dir, "integrated_data_list.rda"))

# Scale and normalize the integrated data
umap_list <- list()
for (dataset in names(integrated_data_list)) {
  integrated_data <- ScaleData(integrated_data_list[[dataset]])
  integrated_data <- RunPCA(integrated_data)
  integrated_data <- RunUMAP(integrated_data, dims = 1:30)
  umap_list[[dataset]] <- integrated_data
}
save(umap_list, file = file.path(st_dir, "umap_list.rda"))


DimPlot(umap_list[[5]], group.by = "orig.ident")



# rds ---------------------------------------------------------------------

sc_dir <- "/fh/working/sun_w/zsui/st2image_data/brca_scRNAseq_datasets"
dir <- file.path(sc_dir, "Mo_2024_rds")

filenames <- list.files(dir, recursive = F)
filenames <- filenames[endsWith(filenames, ".rds")]

HTAN_objs <- list()
metadata <- data.frame()
for (f1 in filenames) {
  print(f1)
  fn1 <- gsub(".rds", "", f1) 
  obj <- readRDS(file = file.path(dir, f1))
  meta1 <- obj@meta.data
  meta1$barcode <- rownames(meta1)
  meta1$filename <- f1
  metadata <- bind_rows(metadata, meta1)
  HTAN_objs[[fn1]] <- obj
}

saveRDS(HTAN_objs, file = file.path("seurat_list_all.rds"))

seurat_objs_meta <- unique(metadata[, c("filename", "Piece_ID", "Sample", "orig.ident", "Chemistry", "Clinical_subtype" )])
write.csv(seurat_objs_meta, file = file.path(dir, "metadata_seurat.csv"))

metadata_scRNA <- seurat_objs_meta[seurat_objs_meta$Chemistry == "scRNA", ]
metadata_scRNA <- na.omit(metadata_scRNA)

seurat_list_scRNA <- HTAN_objs[metadata_scRNA$orig.ident]
saveRDS(seurat_list_scRNA, file = file.path("seurat_list_scRNAseq.rds"))


gene_list <- list()
gene_meta <- data.frame()
for(s1 in names(seurat_list_scRNA)) {
  print(s1)
  n.genes <- nrow(seurat_list_scRNA[[s1]]@assays$RNA$counts)
  n.sample <- ncol(seurat_list_scRNA[[s1]]@assays$RNA$counts)
  gene_meta <- rbind(gene_meta, c(s1, n.genes, n.sample))
  gene_list[[s1]] <- rownames(seurat_list_scRNA[[s1]]@assays$RNA$counts)
}

colnames(gene_meta) <- c("sample", "n.genes", "n.cells")

common_genes <- Reduce(intersect, gene_list)

# Merge Seurat objects
seurat_scRNA <- merge(x = seurat_list_scRNA[[1]], y = seurat_list_scRNA[2:length(seurat_list_scRNA)])
saveRDS(seurat_scRNA, file = file.path(dir, "seurat_scRNA.rds"))

# seurat_scRNA <- readRDS(file.path(dir, "seurat_scRNA.rds"))
metadata <- seurat_scRNA@meta.data
View(seurat_scRNA@meta.data)

map <- unique(seurat_scRNA@meta.data[, c("seurat_clusters", "cell_type_broad")])

celltypes <- unique(seurat_scRNA$cell_type_broad)


options(future.globals.maxSize = 100 * 1024^3)  # 100 GB 
seurat_scRNA <- SCTransform(object = seurat_scRNA, 
                            vars.to.regress = c('nCount_RNA', 'percent.mito'),
                            variable.features.n = 3000)


seurat_scRNA <- RunPCA(seurat_scRNA, verbose = FALSE, npcs = 30)
seurat_scRNA <- RunUMAP(seurat_scRNA, dims = 1:30)

seurat_scRNA <- FindNeighbors(seurat_scRNA, dims = 1:30)
seurat_scRNA <- FindClusters(seurat_scRNA, resolution = 0.5)

saveRDS(seurat_scRNA, file.path(dir, "seurat_scRNA_clustered.rds"))


pdf(file = file.path(dir, "dimplots_Mo_2024.pdf"),height=20,
    width=20)
p1 <- DimPlot(seurat_scRNA, group.by = "orig.ident") + ggtitle("Sample") +
  theme(legend.position = "none")
p2 <- DimPlot(seurat_scRNA) + ggtitle("Seurat Cluster")

p3 <- DimPlot(seurat_scRNA, group.by = "cell_type_broad") + ggtitle("Broad Cell Type")
p4 <- DimPlot(seurat_scRNA, group.by = "cell_type_specific2") + ggtitle("Specific Cell Type")

(p1+p2) / (p3+p4) &
  theme(legend.position = "bottom")

dev.off()


# Merge with Wu_2021 ------------------------------------------------------

sc_dir <- "/fh/working/sun_w/zsui/st2image_data/brca_scRNAseq_datasets"
dir_Mo <- file.path(sc_dir, "Mo_2024_rds")
dir_Wu <- file.path(sc_dir, "Wu_2021")
options(future.globals.maxSize = 300 * 1024^3)  # 200 GB 

seurat_Mo <- readRDS(file.path(dir_Mo, "seurat_scRNA_clustered.rds"))
seurat_Wu <- readRDS(file.path(dir_Wu, "seurat_Wu_pure.rds"))

seurat_Mo$dataset <- "Mo_2024"
seurat_Wu$dataset <- "Wu_2021"

# Merge for easier handling and then split by dataset
seurat_merged <- merge(seurat_Mo, seurat_Wu)
DefaultAssay(seurat_merged) <- "RNA"


# Merge two sc datasets, without integration ------------------------------

seurat_merged <- SCTransform(object = seurat_merged, 
                             vars.to.regress = c('nCount_RNA', 'percent.mito'),
                             variable.features.n = 3000)

saveRDS(seurat_merged, file.path(sc_dir, "seurat_merged_Mo_Wu_normalized.rds"))

seurat_merged <- RunPCA(seurat_merged, verbose = T, npcs = 30)
seurat_merged <- RunUMAP(seurat_merged, dims = 1:30)

seurat_merged <- FindNeighbors(seurat_merged, dims = 1:30)
seurat_merged <- FindClusters(seurat_merged, resolution = 0.5)

saveRDS(seurat_merged, file.path(sc_dir, "seurat_merged_Mo_Wu_clustered.rds"))

seurat_merged <- readRDS(file.path(sc_dir, "seurat_merged_Mo_Wu_clustered.rds"))


# Visualization -----------------------------------------------------------

seurat_merged <- readRDS(file.path(sc_dir, "seurat_merged_Mo_Wu_clustered.rds"))

seurat_merged$cell_type_broad_author <- ifelse(seurat_merged$dataset == "Mo_2024", 
                                               seurat_merged$cell_type_broad,
                                               seurat_merged$celltype_major) 

seurat_merged$cancer_subtype <- ifelse(seurat_merged$dataset == "Mo_2024", 
                                       seurat_merged$Clinical_subtype,
                                       seurat_merged$subtype) 

seurat_merged$Sample <- ifelse(seurat_merged$dataset == "Mo_2024", 
                               seurat_merged$Sample,
                               seurat_merged$orig.ident) 

seurat_merged$cancer_subtype <- gsub("Her2+", "HER2+", seurat_merged$cancer_subtype)
seurat_merged$cancer_subtype <- gsub("HER2\\+\\+", "HER2+", seurat_merged$cancer_subtype)

metadata <- seurat_merged@meta.data

pdf(file = file.path(sc_dir, "dimplots_merged_Mo_Wu_identity.pdf"), height=12,
    width=12)
p1 <- DimPlot(seurat_merged, group.by = "dataset") + ggtitle("Dataset") +
  theme(legend.position = "bottom") 

p2 <- DimPlot(seurat_merged, group.by = "Sample") + ggtitle("Sample") +
  theme(legend.position = "none")

p3 <- DimPlot(seurat_merged, group.by = "cancer_subtype") + ggtitle("Cancer Subtype") +
  theme(legend.position = "bottom")

p4 <- DimPlot(seurat_merged, label = T) + ggtitle("Seurat Cluster") +
  theme(legend.position = "none")

(p1 + p2) / (p3 + p4)

dev.off()


celltype_groups <- list(
  `T cells` = c("T-cells", "T/NK", "Treg"),
  `B cells` = c("B", "B-cells"),
  `Plasma` = c("Plasma", "Plasmablasts"),
  `Cancer Epithelial` = c("Tumor", "Cancer Epithelial"),
  `CAFs` = c("CAF", "CAFs", "cCAF", "mCAF", "vCAF"),
  `PVL` = c("PVL"),
  `DCs` = c("cDC1", "cDC2", "pDC", "DC"),
  `Endothelial` = c("Endothelial"),
  `Myeloid` = c("Mast", "Mono_Macro", "Myeloid"), 
  `Other Epithelial` = c("Normal Epithelial", "Basal_progenitor", "Luminal_progenitor",
                         "Luminal_mature")
)



p1 <- DimPlot(seurat_merged, group.by = "cell_type_broad_author") + ggtitle("Broad Cell Type (Both Studies)") +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(ncol = 3))

p2 <- DimPlot(seurat_merged, group.by = "celltype_major") + ggtitle("Wu_2021") +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(ncol = 3))

p3 <- DimPlot(seurat_merged, group.by = "cell_type_broad") + ggtitle("Mo_2024") +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(ncol = 3))

p_row1 <-  plot_spacer() + p1 + p2 + p3 +  plot_spacer()
p_row1 <- p_row1 + plot_layout(ncol = 5)

# p_list <- list()
# Idents(seurat_merged) <- "cell_type_broad_author"
# for (group in names(celltype_groups)) {
#   
#   cells <- WhichCells(seurat_merged, idents = celltype_groups[[group]])
#   
#   p1 <- DimPlot(object = subset(seurat_merged, cells = cells),
#                 group.by = "cell_type_broad_author") +
#     ggtitle(group) +
#     theme(legend.position = "bottom") +
#     guides(color = guide_legend(ncol = 3, override.aes = list(size=2)))
#   
#   p2 <- DimPlot(object = subset(seurat_merged, cells = cells),
#                 group.by = "cell_type_broad_author",
#                 split.by = "dataset") +
#     ggtitle(group) +
#     theme(legend.position = "bottom") +
#     guides(color = guide_legend(ncol = 3, override.aes = list(size=2)))
#   
#   p_list[[group]] <- p1 + p2 + plot_layout(widths = c(1, 2))
# }
# 
# # Arrange subsequent rows with 5 groups per row
# row_groups <- split(p_list, ceiling(seq_along(p_list)/5))
# plot_rows <- lapply(row_groups, function(row) {
#   do.call(wrap_plots, c(row, list(ncol = 5)))
# })
# 
# # Combine all rows into final layout
# final_plot <- p_row1 / plot_rows[[1]] / plot_rows[[2]] +
#   plot_layout(heights = c(2, 1, 1))
# 
# # Output to PDF
# pdf(file = file.path(sc_dir, "dimplots_merged_Mo_Wu_celltype.pdf"), height = 15, width = 25)
# print(final_plot)
# dev.off()

p_list <- list()
Idents(seurat_merged) <- "cell_type_broad_author"
for (group in names(celltype_groups)) {
  
  cells <- WhichCells(seurat_merged, idents = celltype_groups[[group]])
  
  p1 <- DimPlot(object = subset(seurat_merged, cells = cells),
                group.by = "cell_type_broad_author") +
    ggtitle(group) +
    theme(legend.position = "bottom") +
    guides(color = guide_legend(ncol = 3, override.aes = list(size=2)))
  
  p_list[[group]] <- p1
}

p_row2 <-  p_list[[1]] + p_list[[2]] + p_list[[3]] + p_list[[4]] + p_list[[5]] + plot_layout(ncol = 5)
p_row3 <-  p_list[[6]] + p_list[[7]] + p_list[[8]] + p_list[[9]] + p_list[[10]] + plot_layout(ncol = 5)

pdf(file = file.path(sc_dir, "dimplots_merged_Mo_Wu_celltype.pdf"), height = 15, width = 25)
p_row1 / p_row2 / p_row3 + plot_layout(heights = c(2, 1, 1))
dev.off()


clusters <- levels(seurat_merged$SCT_snn_res.0.5)
Idents(seurat_merged) <- "SCT_snn_res.0.5"
p_list <- list()
for (i in clusters) {
  cells <- WhichCells(seurat_merged, idents = i)
  p_list[[as.character(i)]] <- DimPlot(subset(seurat_merged, cells = cells),
                                       group.by = "cell_type_broad_author", split.by = "dataset") +
    theme(legend.position = "bottom") +
    ggtitle(paste0("Cluster ", i))
}

pdf(file = file.path(sc_dir, "dimplots_merged_Mo_Wu_cluster.pdf"), height=60,
    width=40)

wrap_plots(p_list, ncol = 7) & 
  guides(color = guide_legend(ncol = 2, override.aes = list(size=1)))

dev.off()







# Merge two sc datasets, with integration ---------------------------------

# Split merged object by dataset
split_seurat <- SplitObject(seurat_merged, split.by = "dataset")

# Normalize each dataset with SCTransform individually
split_seurat <- lapply(split_seurat, function(obj) {
  SCTransform(obj, vars.to.regress = c('nCount_RNA', 'percent.mito'),
              variable.features.n = 3000,
              verbose = FALSE)
})

saveRDS(split_seurat, file.path(sc_dir, "seurat_merged_normalized.rds"))

# Check rowname consistency across objects (prevent integration errors)
all_genes <- lapply(split_seurat, rownames)
lengths(all_genes)  # should be equal
all(sapply(all_genes[-1], function(x) identical(x, all_genes[[1]])))  # should be TRUE

common_genes <- Reduce(intersect, all_genes)

# Trim all objects to shared genes
split_seurat <- lapply(split_seurat, function(obj) obj[common_genes, ])

# Select integration features
features <- SelectIntegrationFeatures(object.list = split_seurat, nfeatures = 3000)

# Prepare objects for integration using SCTransform
split_seurat <- PrepSCTIntegration(object.list = split_seurat, anchor.features = features)

# Find anchors for integration
anchors <- FindIntegrationAnchors(
  object.list = split_seurat,
  normalization.method = "SCT",
  anchor.features = features,
  verbose = TRUE
)

# Review number of anchors retained
print(anchors)

# Integrate data using SCT normalization
seurat_integrated <- IntegrateData(
  anchorset = anchors,
  normalization.method = "SCT",
  verbose = TRUE
)

saveRDS(seurat_integrated, file = file.path(sc_dir, "seurat_integrated_Mo_Wu.rds"))

# Set the default assay to "integrated" for downstream steps
DefaultAssay(seurat_integrated) <- "integrated"

# Run PCA
seurat_integrated <- RunPCA(seurat_integrated, verbose = FALSE)

# Run UMAP for visualization (using top 30 PCs as a default)
seurat_integrated <- RunUMAP(seurat_integrated, dims = 1:30, reduction = "pca")

# Clustering
seurat_integrated <- FindNeighbors(seurat_integrated, dims = 1:30)
seurat_integrated <- FindClusters(seurat_integrated, resolution = 0.5)  # adjust resolution as needed

# Save the fully integrated Seurat object
saveRDS(seurat_integrated, file.path(sc_dir, "seurat_integrated_Mo_Wu_clustered.rds"))



# Visualization -----------------------------------------------------------

seurat_integrated <- readRDS(file.path(sc_dir, "seurat_integrated_Mo_Wu_clustered.rds"))

seurat_integrated$Sample <- ifelse(seurat_integrated$dataset == "Mo_2024", 
                                   seurat_integrated$Sample,
                                   seurat_integrated$orig.ident) 

seurat_integrated$cell_type_broad_author <- ifelse(seurat_integrated$dataset == "Mo_2024", 
                                                   seurat_integrated$cell_type_broad,
                                                   seurat_integrated$celltype_major) 

seurat_integrated$cancer_subtype <- ifelse(seurat_integrated$dataset == "Mo_2024", 
                                           seurat_integrated$Clinical_subtype,
                                           seurat_integrated$subtype) 

seurat_integrated$cancer_subtype <- gsub("Her2+", "HER2+", seurat_integrated$cancer_subtype)
seurat_integrated$cancer_subtype <- gsub("HER2\\+\\+", "HER2+", seurat_integrated$cancer_subtype)


metadata <- seurat_integrated@meta.data

pdf(file = file.path(sc_dir, "dimplots_integrated_Mo_Wu_identity.pdf"), height=12,
    width=12)
p1 <- DimPlot(seurat_integrated, group.by = "dataset") + ggtitle("Dataset") +
  theme(legend.position = "bottom") 

p2 <- DimPlot(seurat_integrated, group.by = "Sample") + ggtitle("Sample") +
  theme(legend.position = "none")

p3 <- DimPlot(seurat_integrated, group.by = "cancer_subtype") + ggtitle("Cancer Subtype") +
  theme(legend.position = "bottom")

p4 <- DimPlot(seurat_integrated, label = T) + ggtitle("Seurat Cluster") +
  theme(legend.position = "none")

(p1 + p2) / (p3 + p4)

dev.off()


celltype_groups <- list(
  `T cells` = c("T-cells", "T/NK", "Treg"),
  `B cells` = c("B", "B-cells"),
  `Plasma` = c("Plasma", "Plasmablasts"),
  `Cancer Epithelial` = c("Tumor", "Cancer Epithelial"),
  `CAFs` = c("CAF", "CAFs", "cCAF", "mCAF", "vCAF"),
  `PVL` = c("PVL"),
  `DCs` = c("cDC1", "cDC2", "pDC", "DC"),
  `Endothelial` = c("Endothelial"),
  `Myeloid` = c("Mast", "Mono_Macro", "Myeloid"), 
  `Other Epithelial` = c("Normal Epithelial", "Basal_progenitor", "Luminal_progenitor",
                         "Luminal_mature")
)


pdf(file = file.path(sc_dir, "dimplots_integrated_Mo_Wu_celltype.pdf"), height=15,
    width=25)

p1 <- DimPlot(seurat_integrated, group.by = "cell_type_broad_author") + ggtitle("Broad Cell Type (Both Studies)") +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(ncol = 3))

p2 <- DimPlot(seurat_integrated, group.by = "celltype_major") + ggtitle("Wu_2021") +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(ncol = 3))

p3 <- DimPlot(seurat_integrated, group.by = "cell_type_broad") + ggtitle("Mo_2024") +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(ncol = 3))

p_row1 <-  plot_spacer() + p1 + p2 + p3 +  plot_spacer()
p_row1 <- p_row1 + plot_layout(ncol = 5)

p_list <- list()
Idents(seurat_integrated) <- "cell_type_broad_author"
for (group in names(celltype_groups)) {
  
  cells <- WhichCells(seurat_integrated, idents = celltype_groups[[group]])
  
  p1 <- DimPlot(object = subset(seurat_integrated, cells = cells),
                group.by = "cell_type_broad_author",
                split.by = "dataset") +
    ggtitle(group) +
    theme(legend.position = "bottom") +
    guides(color = guide_legend(ncol = 3, override.aes = list(size=2)))
  
  p_list[[group]] <- p1
}

p_row2 <-  p_list[[1]] + p_list[[2]] + p_list[[3]] + p_list[[4]] + p_list[[5]] + plot_layout(ncol = 5)
p_row3 <-  p_list[[6]] + p_list[[7]] + p_list[[8]] + p_list[[9]] + p_list[[10]] + plot_layout(ncol = 5)

p_row1 / p_row2 / p_row3 

dev.off()



# Visualize epithelial cells
pdf(file = file.path(sc_dir, "dimplots_Mo_Wu_epithelial.pdf"), height=15,
    width=25)
Idents(seurat_Wu) <- "celltype_major"
cells <- WhichCells(seurat_Wu, idents = "Cancer Epithelial")
p1 <- DimPlot(subset(seurat_Wu, cells = cells), group.by = "orig.ident", label = T,
              label.size = 1) +
  theme(legend.position = "bottom") +
  ggtitle("Wu_2021: Cancer Epithelial")

cells <- WhichCells(seurat_Wu, idents = "Normal Epithelial")
p11 <- DimPlot(subset(seurat_Wu, cells = cells), group.by = "orig.ident") +
  theme(legend.position = "bottom") +
  ggtitle("Wu_2021: Normal Epithelial")

Idents(seurat_Mo) <- "cell_type_broad"
cells <- WhichCells(seurat_Mo, idents = "Tumor")
p2 <- DimPlot(subset(seurat_Mo, cells = cells), group.by = "Sample", label = T,
              label.size = 1) +
  theme(legend.position = "bottom") +
  ggtitle("Mo_2024: Tumor")

cells <- WhichCells(seurat_Mo, idents = "Basal_progenitor")
p3 <- DimPlot(subset(seurat_Mo, cells = cells), group.by = "Sample") +
  theme(legend.position = "bottom") +
  ggtitle("Mo_2024: Basal_progenitor")

cells <- WhichCells(seurat_Mo, idents = "Luminal_progenitor")
p4 <- DimPlot(subset(seurat_Mo, cells = cells), group.by = "Sample") +
  theme(legend.position = "bottom") +
  ggtitle("Mo_2024: Luminal_progenitor")

cells <- WhichCells(seurat_Mo, idents = "Luminal_mature")
p5 <- DimPlot(subset(seurat_Mo, cells = cells), group.by = "Sample") +
  theme(legend.position = "bottom") +
  ggtitle("Mo_2024: Luminal_mature")

(p1 + p11 + p2 ) / (p3 + p4 + p5)

dev.off()



clusters <- levels(seurat_integrated$integrated_snn_res.0.5)
Idents(seurat_integrated) <- "integrated_snn_res.0.5"
p_list <- list()
for (i in clusters) {
  cells <- WhichCells(seurat_integrated, idents = i)
  p_list[[as.character(i)]] <- DimPlot(subset(seurat_integrated, cells = cells),
                                       group.by = "cell_type_broad_author", split.by = "dataset") +
    theme(legend.position = "bottom") +
    ggtitle(paste0("Cluster ", i))
}

pdf(file = file.path(sc_dir, "dimplots_integrated_Mo_Wu_cluster.pdf"), height=60,
    width=40)

wrap_plots(p_list, ncol = 7) & 
  guides(color = guide_legend(ncol = 2, override.aes = list(size=1)))

dev.off()



# Exclude low-confidence cells --------------------------------------------

seurat_integrated <- readRDS(file.path(sc_dir, "seurat_integrated_Mo_Wu_clustered.rds"))

seurat_integrated$Sample <- ifelse(seurat_integrated$dataset == "Mo_2024", 
                                   seurat_integrated$Sample,
                                   seurat_integrated$orig.ident) 

seurat_integrated$cell_type_broad_author <- ifelse(seurat_integrated$dataset == "Mo_2024", 
                                                   seurat_integrated$cell_type_broad,
                                                   seurat_integrated$celltype_major) 

seurat_integrated$cancer_subtype <- ifelse(seurat_integrated$dataset == "Mo_2024", 
                                           seurat_integrated$Clinical_subtype,
                                           seurat_integrated$subtype) 

seurat_integrated$cancer_subtype <- gsub("Her2+", "HER2+", seurat_integrated$cancer_subtype)
seurat_integrated$cancer_subtype <- gsub("HER2\\+\\+", "HER2+", seurat_integrated$cancer_subtype)

metadata <- seurat_integrated@meta.data


# Annotate clusters
table(metadata$cell_type_broad_author, metadata$integrated_snn_res.0.5)

# Preliminary
cls_ct <- list(
  "0" = c("T/NK", "T−cells", "Treg"),
  "1" = c("cDC1", "cDC2", "Mono_Macro", "Myeloid"),
  "2" = c("T-cells", "T/NK", "Treg"),
  "3" = c("Endothelial", "Mono_Macro"),
  "4" = c("T-cells", "T/NK", "Treg", "T−cells"), 
  "5" = c("Tumor", "Plasma", "mCAF", "Mono_Macro", "Plasma", "T/NK", "Tumor", "vCAF"),
  "6" = c("Cancer Epithelial", "Normal Epithelial", "Plasma", "Tumor"),
  "7" = c("CAF", "cCAF", "mCAF", "PVL","vCAF"),
  "8" = c(),
  "9" = c("CAF", "CAFs", "mCAF"),
  "10" = c("Tumor"),
  "11" = c("Cancer Epithelial", "Luminal_mature", "Luminal_progenitor", "Normal Epithelial", "Tumor"),
  "12" = c("T−cells", "T/NK"),
  "13" = c("B", "B-cells"),
  "14" = c("Cancer Epithelial", "Tumor"),
  "15" = c("T−cells", "T/NK", "Treg", "Tumor"),
  "16" = c("Cancer Epithelial", "Luminal_mature", "Luminal_progenitor", "Normal Epithelial", "Tumor"),
  "17" = c("CAF", "CAFs", "Cancer Epithelial", "mCAF", "Tumor", "vCAF"),
  "18" = c("T−cells", "T/NK"),
  "19" = c("Cancer Epithelial", "Luminal_mature", "Tumor"),
  "20" = c("Cancer Epithelial"),
  "21" = c("Cancer Epithelial", "Luminal_mature", "Luminal_progenitor", "Mast", "Tumor"),
  "22" = c("Plasmablasts", ),
  "23" = c("Cancer Epithelial", "Luminal_mature", "Tumor"),
  "24" = c("Cancer Epithelial"),
  "25" = c("Plasma", "Plasmablasts", "Tumor"),
  "26" = c("T−cells", "T/NK"),
  "27" = c("CAF", "CAFs", "cCAF", "mCAF"),
  "28" = c("T/NK"),
  "29" = c("Tumor"),
  "30" = c("Cancer Epithelial", "cDC1", "cDC2", "mCAF", "Mono_Macro", "Tumor"),
  "31" = c("Basal_progenitor", "Luminal_mature", "Normal Epithelial"),
  "32" = c("Cancer Epithelial"),
  "33" = c("T/NK", "Tumor"),
  "34" = c("Mast"),
  "35" = c("DC", "Myeloid", "pDC"),
  "36" = c("Cancer Epithelial", "Tumor"),
  "37" = c("Tumor"),
  "38" = c("Cancer Epithelial", "Tumor"),
  "39" = c("Myeloid", "Tumor"),
  "40" = c("B"),
  "41" = c("Tumor"),
  "42" = c("Endothelial")
)

cls_ct <- list(
  "0" = c("T/NK", "T−cells", "Treg"),
  "1" = c("cDC1", "cDC2", "Mono_Macro", "Myeloid"),
  "2" = c("T-cells", "T/NK", "Treg"),
  "3" = c("Endothelial"),
  "4" = c("T-cells", "T/NK", "Treg"), 
  "5" = c("Cancer Epithelial", "Tumor", "Plasma", "mCAF", "Mono_Macro", "Plasma", "T/NK", "T-cells", "Treg", "Tumor", "vCAF"),
  "6" = c("Cancer Epithelial", "Normal Epithelial", "Plasma", "Tumor"),
  "7" = c("CAF", "cCAF", "mCAF", "PVL","vCAF"),
  "8" = c("Tumor", "Cancer Epithelial", "Endothelial", "Plasma", "mCAF", "Luminal_progenitor", "CAF", "CAFs"),
  "9" = c("CAF", "CAFs", "mCAF"),
  "10" = c("Tumor", "Cancer Epithelial"),
  "11" = c("Cancer Epithelial", "Luminal_mature", "Luminal_progenitor", "Normal Epithelial", "Tumor"),
  "12" = c("T−cells", "T/NK"),
  "13" = c("B", "B-cells"),
  "14" = c("Cancer Epithelial", "Tumor"),
  "15" = c("T−cells", "T/NK", "Treg"),
  "16" = c("Cancer Epithelial", "Luminal_mature", "Luminal_progenitor", "Normal Epithelial", "Tumor"),
  "17" = c("CAF", "CAFs", "Cancer Epithelial", "mCAF", "Tumor", "vCAF"),
  "18" = c("T−cells", "T/NK", "Treg"),
  "19" = c("Cancer Epithelial", "Luminal_mature", "Tumor"),
  "20" = c("Cancer Epithelial"),
  "21" = c("Cancer Epithelial", "Luminal_mature", "Luminal_progenitor", "Tumor"),
  "22" = c("Plasmablasts"),
  "23" = c("Cancer Epithelial", "Luminal_mature", "Tumor"),
  "24" = c("Cancer Epithelial"),
  "25" = c("Plasma", "Plasmablasts", "Tumor"),
  "26" = c("T−cells", "T/NK", "Treg"),
  "27" = c("CAF", "CAFs", "cCAF", "mCAF"),
  "28" = c("T/NK", "Treg"),
  "29" = c("Tumor"),
  "30" = c("Cancer Epithelial", "cDC1", "cDC2", "mCAF", "Mono_Macro", "Myeloid", "Tumor", "vCAF"),
  "31" = c("Basal_progenitor", "Luminal_mature", "Normal Epithelial"),
  "32" = c("Cancer Epithelial"),
  "33" = c("T/NK", "Tumor", "T-cells"),
  "34" = c("Mast"),
  "35" = c("DC", "Myeloid", "pDC"),
  "36" = c("Cancer Epithelial", "Tumor"),
  "37" = c("Tumor"),
  "38" = c("Cancer Epithelial", "Tumor"),
  "39" = c("Myeloid", "Tumor"),
  "40" = c("B"),
  "41" = c("Tumor"),
  "42" = c("Endothelial")
)

cls_ct_df <- stack(cls_ct)

# We want to keep the cells whose cluster membership match to one cell type provided by authors. 
cells_selected <- c()
for (cls in names(cls_ct)) {
  ct <- cls_ct[[cls]]
  cells_selected <- c(cells_selected, 
                      seurat_integrated@meta.data %>%
                        subset(integrated_snn_res.0.5 == cls) %>%
                        subset(cell_type_broad_author %in% ct) %>%
                        row.names()
  )
}

seurat_integrated_pure <- subset(seurat_integrated, cells = cells_selected)
table(seurat_integrated_pure$cell_type_broad_author, seurat_integrated_pure$integrated_snn_res.0.5)


# Recluster Filtered Cells, without integration ---------------------------

# Recluster filtered cells
seurat_merged_pure <- SCTransform(object = seurat_integrated_pure, 
                                  vars.to.regress = c('nCount_RNA', 'percent.mito'),
                                  variable.features.n = 3000)

seurat_merged_pure <- RunPCA(seurat_merged_pure, verbose = T, npcs = 30)
seurat_merged_pure <- RunUMAP(seurat_merged_pure, dims = 1:30)

seurat_merged_pure <- FindNeighbors(seurat_merged_pure, dims = 1:30)
seurat_merged_pure <- FindClusters(seurat_merged_pure, resolution = 0.5)

DimPlot(seurat_merged_pure)

saveRDS(seurat_merged_pure, file.path(sc_dir, "seurat_merged_Mo_Wu_pure_clustered.rds"))

# Visualization -----------------------------------------------------------

seurat_merged_pure <- readRDS(file.path(sc_dir, "seurat_merged_Mo_Wu_pure_clustered.rds"))

seurat_merged_pure$Sample <- ifelse(seurat_merged_pure$dataset == "Mo_2024", 
                                    seurat_merged_pure$Sample,
                                    seurat_merged_pure$orig.ident) 

seurat_merged_pure$cell_type_broad_author <- ifelse(seurat_merged_pure$dataset == "Mo_2024", 
                                                    seurat_merged_pure$cell_type_broad,
                                                    seurat_merged_pure$celltype_major) 

seurat_merged_pure$cancer_subtype <- ifelse(seurat_merged_pure$dataset == "Mo_2024", 
                                            seurat_merged_pure$Clinical_subtype,
                                            seurat_merged_pure$subtype) 

seurat_merged_pure$cancer_subtype <- gsub("Her2+", "HER2+", seurat_merged_pure$cancer_subtype)
seurat_merged_pure$cancer_subtype <- gsub("HER2\\+\\+", "HER2+", seurat_merged_pure$cancer_subtype)

metadata <- seurat_merged_pure@meta.data

pdf(file = file.path(sc_dir, "dimplots_merged_Mo_Wu_pure_identity.pdf"), height=12,
    width=12)
p1 <- DimPlot(seurat_merged_pure, group.by = "dataset") + ggtitle("Dataset") +
  theme(legend.position = "bottom") 

p2 <- DimPlot(seurat_merged_pure, group.by = "Sample") + ggtitle("Sample") +
  theme(legend.position = "none")

p3 <- DimPlot(seurat_merged_pure, group.by = "cancer_subtype") + ggtitle("Cancer Subtype") +
  theme(legend.position = "bottom")

p4 <- DimPlot(seurat_merged_pure, label = T) + ggtitle("Seurat Cluster") +
  theme(legend.position = "none")

(p1 + p2) / (p3 + p4)

dev.off()


celltype_groups <- list(
  `T cells` = c("T-cells", "T/NK", "Treg"),
  `B cells` = c("B", "B-cells"),
  `Plasma` = c("Plasma", "Plasmablasts"),
  `Cancer Epithelial` = c("Tumor", "Cancer Epithelial"),
  `CAFs` = c("CAF", "CAFs", "cCAF", "mCAF", "vCAF"),
  `PVL` = c("PVL"),
  `DCs` = c("cDC1", "cDC2", "pDC", "DC"),
  `Endothelial` = c("Endothelial"),
  `Myeloid` = c("Mast", "Mono_Macro", "Myeloid"), 
  `Other Epithelial` = c("Normal Epithelial", "Basal_progenitor", "Luminal_progenitor",
                         "Luminal_mature")
)


pdf(file = file.path(sc_dir, "dimplots_merged_Mo_Wu_pure_celltype.pdf"), height=15,
    width=25)

p1 <- DimPlot(seurat_merged_pure, group.by = "cell_type_broad_author") + ggtitle("Broad Cell Type (Both Studies)") +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(ncol = 3))

p2 <- DimPlot(seurat_merged_pure, group.by = "celltype_major") + ggtitle("Wu_2021") +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(ncol = 3))

p3 <- DimPlot(seurat_merged_pure, group.by = "cell_type_broad") + ggtitle("Mo_2024") +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(ncol = 3))

p_row1 <-  plot_spacer() + p1 + p2 + p3 +  plot_spacer()
p_row1 <- p_row1 + plot_layout(ncol = 5)

p_list <- list()
Idents(seurat_merged_pure) <- "cell_type_broad_author"
for (group in names(celltype_groups)) {
  
  cells <- WhichCells(seurat_merged_pure, idents = celltype_groups[[group]])
  
  p1 <- DimPlot(object = subset(seurat_merged_pure, cells = cells),
                group.by = "cell_type_broad_author",
                split.by = "dataset") +
    ggtitle(group) +
    theme(legend.position = "bottom") +
    guides(color = guide_legend(ncol = 3, override.aes = list(size=2)))
  
  p_list[[group]] <- p1
}

p_row2 <-  p_list[[1]] + p_list[[2]] + p_list[[3]] + p_list[[4]] + p_list[[5]] + plot_layout(ncol = 5)
p_row3 <-  p_list[[6]] + p_list[[7]] + p_list[[8]] + p_list[[9]] + p_list[[10]] + plot_layout(ncol = 5)

p_row1 / p_row2 / p_row3 

dev.off()

clusters <- levels(seurat_merged_pure$seurat_clusters)
Idents(seurat_merged_pure) <- "integrated_snn_res.0.5"
p_list <- list()
for (i in clusters) {
  cells <- WhichCells(seurat_merged_pure, idents = i)
  p_list[[as.character(i)]] <- DimPlot(subset(seurat_merged_pure, cells = cells),
                                       group.by = "cell_type_broad_author", split.by = "dataset") +
    theme(legend.position = "bottom") +
    ggtitle(paste0("Cluster ", i))
}

pdf(file = file.path(sc_dir, "dimplots_merged_Mo_Wu_pure_cluster.pdf"), height=60,
    width=40)

wrap_plots(p_list, ncol = 7) & 
  guides(color = guide_legend(ncol = 2, override.aes = list(size=1)))

dev.off()

# Recluster Filtered Cells, with integration ------------------------------

# Split merged object by dataset
split_seurat <- SplitObject(seurat_merged_pure, split.by = "dataset")

# Normalize each dataset with SCTransform individually
split_seurat <- lapply(split_seurat, function(obj) {
  SCTransform(obj, vars.to.regress = c('nCount_RNA', 'percent.mito'),
              variable.features.n = 3000,
              verbose = FALSE)
})

# Check rowname consistency across objects (prevent integration errors)
all_genes <- lapply(split_seurat, rownames)
lengths(all_genes)  # should be equal
all(sapply(all_genes[-1], function(x) identical(x, all_genes[[1]])))  # should be TRUE

common_genes <- Reduce(intersect, all_genes)

# Trim all objects to shared genes
split_seurat <- lapply(split_seurat, function(obj) obj[common_genes, ])

# Select integration features
features <- SelectIntegrationFeatures(object.list = split_seurat, nfeatures = 3000)

# Prepare objects for integration using SCTransform
split_seurat <- PrepSCTIntegration(object.list = split_seurat, anchor.features = features)

# Find anchors for integration
anchors <- FindIntegrationAnchors(
  object.list = split_seurat,
  normalization.method = "SCT",
  anchor.features = features,
  verbose = TRUE
)

# Review number of anchors retained
print(anchors)

# Integrate data using SCT normalization
seurat_integrated_pure <- IntegrateData(
  anchorset = anchors,
  normalization.method = "SCT",
  verbose = TRUE
)

saveRDS(seurat_integrated_pure, file = file.path(sc_dir, "seurat_integrated_Mo_Wu_pure.rds"))

# Set the default assay to "integrated" for downstream steps
DefaultAssay(seurat_integrated) <- "integrated"

# Run PCA
seurat_integrated <- RunPCA(seurat_integrated, verbose = FALSE)

# Run UMAP for visualization (using top 30 PCs as a default)
seurat_integrated <- RunUMAP(seurat_integrated, dims = 1:30, reduction = "pca")

# Clustering
seurat_integrated <- FindNeighbors(seurat_integrated, dims = 1:30)
seurat_integrated <- FindClusters(seurat_integrated, resolution = 0.5)  # adjust resolution as needed

# Save the fully integrated Seurat object
saveRDS(seurat_integrated, file.path(sc_dir, "seurat_integrated_Mo_Wu_pure_clustered.rds"))


# Visualization -----------------------------------------------------------

seurat_integrated_pure <- readRDS(file.path(sc_dir, "seurat_integrated_Mo_Wu_pure_clustered.rds"))

seurat_integrated_pure$Sample <- ifelse(seurat_integrated_pure$dataset == "Mo_2024", 
                                        seurat_integrated_pure$Sample,
                                        seurat_integrated_pure$orig.ident) 

seurat_integrated_pure$cell_type_broad_author <- ifelse(seurat_integrated_pure$dataset == "Mo_2024", 
                                                        seurat_integrated_pure$cell_type_broad,
                                                        seurat_integrated_pure$celltype_major) 

seurat_integrated_pure$cancer_subtype <- ifelse(seurat_integrated_pure$dataset == "Mo_2024", 
                                                seurat_integrated_pure$Clinical_subtype,
                                                seurat_integrated_pure$subtype) 

seurat_integrated_pure$cancer_subtype <- gsub("Her2+", "HER2+", seurat_integrated_pure$cancer_subtype)
seurat_integrated_pure$cancer_subtype <- gsub("HER2\\+\\+", "HER2+", seurat_integrated_pure$cancer_subtype)

metadata <- seurat_integrated_pure@meta.data

pdf(file = file.path(sc_dir, "dimplots_integrated_Mo_Wu_pure_identity.pdf"), height=12,
    width=12)
p1 <- DimPlot(seurat_integrated_pure, group.by = "dataset") + ggtitle("Dataset") +
  theme(legend.position = "bottom") 

p2 <- DimPlot(seurat_integrated_pure, group.by = "Sample") + ggtitle("Sample") +
  theme(legend.position = "none")

p3 <- DimPlot(seurat_integrated_pure, group.by = "cancer_subtype") + ggtitle("Cancer Subtype") +
  theme(legend.position = "bottom")

p4 <- DimPlot(seurat_integrated_pure, label = T) + ggtitle("Seurat Cluster") +
  theme(legend.position = "none")

(p1 + p2) / (p3 + p4)

dev.off()


celltype_groups <- list(
  `T cells` = c("T-cells", "T/NK", "Treg"),
  `B cells` = c("B", "B-cells"),
  `Plasma` = c("Plasma", "Plasmablasts"),
  `Cancer Epithelial` = c("Tumor", "Cancer Epithelial"),
  `CAFs` = c("CAF", "CAFs", "cCAF", "mCAF", "vCAF"),
  `PVL` = c("PVL"),
  `DCs` = c("cDC1", "cDC2", "pDC", "DC"),
  `Endothelial` = c("Endothelial"),
  `Myeloid` = c("Mast", "Mono_Macro", "Myeloid"), 
  `Other Epithelial` = c("Normal Epithelial", "Basal_progenitor", "Luminal_progenitor",
                         "Luminal_mature")
)


pdf(file = file.path(sc_dir, "dimplots_integrated_Mo_Wu_pure_celltype.pdf"), height=15,
    width=25)

p1 <- DimPlot(seurat_integrated_pure, group.by = "cell_type_broad_author") + ggtitle("Broad Cell Type (Both Studies)") +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(ncol = 3))

p2 <- DimPlot(seurat_integrated_pure, group.by = "celltype_major") + ggtitle("Wu_2021") +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(ncol = 3))

p3 <- DimPlot(seurat_integrated_pure, group.by = "cell_type_broad") + ggtitle("Mo_2024") +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(ncol = 3))

p_row1 <-  plot_spacer() + p1 + p2 + p3 +  plot_spacer()
p_row1 <- p_row1 + plot_layout(ncol = 5)

p_list <- list()
Idents(seurat_integrated_pure) <- "cell_type_broad_author"
for (group in names(celltype_groups)) {
  
  cells <- WhichCells(seurat_integrated_pure, idents = celltype_groups[[group]])
  
  p1 <- DimPlot(object = subset(seurat_integrated_pure, cells = cells),
                group.by = "cell_type_broad_author",
                split.by = "dataset") +
    ggtitle(group) +
    theme(legend.position = "bottom") +
    guides(color = guide_legend(ncol = 3, override.aes = list(size=2)))
  
  p_list[[group]] <- p1
}

p_row2 <-  p_list[[1]] + p_list[[2]] + p_list[[3]] + p_list[[4]] + p_list[[5]] + plot_layout(ncol = 5)
p_row3 <-  p_list[[6]] + p_list[[7]] + p_list[[8]] + p_list[[9]] + p_list[[10]] + plot_layout(ncol = 5)

p_row1 / p_row2 / p_row3 

dev.off()

clusters <- levels(seurat_integrated_pure$seurat_clusters)
Idents(seurat_integrated_pure) <- "integrated_snn_res.0.5"
p_list <- list()
for (i in clusters) {
  cells <- WhichCells(seurat_integrated_pure, idents = i)
  p_list[[as.character(i)]] <- DimPlot(subset(seurat_integrated_pure, cells = cells),
                                       group.by = "cell_type_broad_author", split.by = "dataset") +
    theme(legend.position = "bottom") +
    ggtitle(paste0("Cluster ", i))
}

pdf(file = file.path(sc_dir, "dimplots_integrated_Mo_Wu_pure_cluster.pdf"), height=60,
    width=40)

wrap_plots(p_list, ncol = 7) & 
  guides(color = guide_legend(ncol = 2, override.aes = list(size=1)))

dev.off()



# Further filter the cells based on broad cell type group. ----------------

seurat_integrated_pure <- readRDS(file.path(sc_dir, "seurat_integrated_Mo_Wu_pure_clustered.rds"))

seurat_integrated_pure$Sample <- ifelse(seurat_integrated_pure$dataset == "Mo_2024", 
                                        seurat_integrated_pure$Sample,
                                        seurat_integrated_pure$orig.ident) 

seurat_integrated_pure$cell_type_broad_author <- ifelse(seurat_integrated_pure$dataset == "Mo_2024", 
                                                        seurat_integrated_pure$cell_type_broad,
                                                        seurat_integrated_pure$celltype_major) 

seurat_integrated_pure$cancer_subtype <- ifelse(seurat_integrated_pure$dataset == "Mo_2024", 
                                                seurat_integrated_pure$Clinical_subtype,
                                                seurat_integrated_pure$subtype) 

seurat_integrated_pure$cancer_subtype <- gsub("Her2+", "HER2+", seurat_integrated_pure$cancer_subtype)
seurat_integrated_pure$cancer_subtype <- gsub("HER2\\+\\+", "HER2+", seurat_integrated_pure$cancer_subtype)

celltype_groups <- list(
  `T.cell` = c("T-cells", "T/NK", "Treg"),
  `B.cell` = c("B", "B-cells"),
  `Plasma` = c("Plasma"), 
  `Plasmablasts` = c("Plasmablasts"),
  `Cancer.Epithelial` = c("Tumor", "Cancer Epithelial"),
  `CAFs/PVL` = c("PVL", "CAF", "CAFs", "cCAF", "mCAF", "vCAF"),
  `Endothelial` = c("Endothelial"),
  `Myeloid` = c("Mast", "Mono_Macro", "Myeloid", "cDC1", "cDC2", "pDC", "DC"), 
  `Other.Epithelial` = c("Normal Epithelial", "Basal_progenitor", "Luminal_progenitor",
                         "Luminal_mature")
)

celltype_map <- setNames(
  rep(names(celltype_groups), lengths(celltype_groups)),
  unlist(celltype_groups)
)

seurat_integrated_pure$cell_type_group <- unname(celltype_map[seurat_integrated_pure$cell_type_broad_author])

table(seurat_integrated_pure$cell_type_group, seurat_integrated_pure$integrated_snn_res.0.5)

ct_cls <- list(
  "B.cell" = c(11),
  "CAFs/PVL" = c(5,6,8,15,25,32),
  "Cancer.Epithelial" = c(5,7,9,10,12,13,16,17,18,20,22,23,26,28,29,30,32,34,35,36,37,38,39,40),
  "Endothelial" = c(4,41),
  "Myeloid" = c(0,5,31,33,36,38),
  "Other.Epithelial" = c(9,12,16,27,29),
  "Plasma" = c(5,9,24,30),
  "Plasmablasts" = c(21),
  "T.cell" = c(1,2,3,5,14,19)
)

ct_cls_df <- stack(ct_cls)
cells_selected <- c()
for (ct in names(ct_cls)) {
  cls <- ct_cls[[ct]]
  cells_selected <- c(cells_selected, 
                      seurat_integrated_pure@meta.data %>%
                        subset(integrated_snn_res.0.5 %in% cls) %>%
                        subset(cell_type_group == ct) %>%
                        row.names()
  )
}

seurat_integrated_pure <- subset(seurat_integrated_pure, cells = cells_selected)
table(seurat_integrated_pure$cell_type_group, seurat_integrated_pure$integrated_snn_res.0.5)



p1 <- DimPlot(seurat_integrated_pure, group.by = "cell_type_group") + ggtitle("Broad Cell Type (Both Studies)") +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(ncol = 3))

p2 <- DimPlot(seurat_integrated_pure, group.by = "integrated_snn_res.0.5", split.by = "cell_type_group", 
              ncol = 3, label = T) + ggtitle("Broad Cell Type (Both Studies)") +
  theme(legend.position = "None")

pdf(file = file.path(sc_dir, "dimplots_integrated_Mo_Wu_pure_celltype_group_final.pdf"), height=20,
    width=20)

p1/p2

dev.off()

clusters <- levels(seurat_integrated_pure$integrated_snn_res.0.5)
Idents(seurat_integrated_pure) <- "integrated_snn_res.0.5"
p_list <- list()
for (i in clusters) {
  cells <- WhichCells(seurat_integrated_pure, idents = i)
  p_list[[as.character(i)]] <- DimPlot(subset(seurat_integrated_pure, cells = cells),
                                       group.by = "cell_type_group") +
    theme(legend.position = "bottom") +
    ggtitle(paste0("Cluster ", i))
}

pdf(file = file.path(sc_dir, "dimplots_integrated_Mo_Wu_pure_cluster_group_final.pdf"), height=40,
    width=40)

wrap_plots(p_list, ncol = 7) & 
  guides(color = guide_legend(ncol = 2, override.aes = list(size=1)))

dev.off()

split_seurat <- SplitObject(seurat_integrated_pure, split.by = "dataset")

seurat_Wu_pure <- split_seurat[["Wu_2021"]]  
seurat_Mo_pure <- split_seurat[["Mo_2024"]] 

seurat_Wu_pure <- SCTransform(object = seurat_Wu_pure, 
                              vars.to.regress = c('nCount_RNA', 'percent.mito'),
                              variable.features.n = 3000)

seurat_Mo_pure <- SCTransform(object = seurat_Mo_pure, 
                              vars.to.regress = c('nCount_RNA', 'percent.mito'),
                              variable.features.n = 3000)

saveRDS(seurat_Wu_pure, file.path(sc_dir, "seurat_Wu_pure_clustered.rds"))
saveRDS(seurat_Mo_pure, file.path(sc_dir, "seurat_Mo_pure_clustered.rds"))

# save sparse matrix
ref_dir <- file.path(sc_dir, "CARD_reference", "Mo_2024_filtered")
count_matrix <- seurat_Mo_pure[["RNA"]]$counts
Matrix::writeMM(obj = count_matrix, file = file.path(ref_dir, "count_matrix_sparse.mtx"))

# save genes and cells names and metadata
write(x = rownames(count_matrix), file = file.path(ref_dir, "count_matrix_genes.tsv"))
write(x = colnames(count_matrix), file = file.path(ref_dir, "count_matrix_barcodes.tsv"))
metadata <- seurat_Mo_pure@meta.data
write.csv(x = metadata, file = file.path(ref_dir, "metadata.csv"))



# save sparse matrix
ref_dir <- file.path(sc_dir, "CARD_reference", "Wu_2021_filtered")
count_matrix <- seurat_Wu_pure[["RNA"]]$counts
Matrix::writeMM(obj = count_matrix, file = file.path(ref_dir, "count_matrix_sparse.mtx"))

# save genes and cells names and metadata
write(x = rownames(count_matrix), file = file.path(ref_dir, "count_matrix_genes.tsv"))
write(x = colnames(count_matrix), file = file.path(ref_dir, "count_matrix_barcodes.tsv"))
metadata <- seurat_Wu_pure@meta.data
write.csv(x = metadata, file = file.path(ref_dir, "metadata.csv"))


# Generate reference datasets for CARD ------------------------------------





# # Calculate for Wu 2021 data
# chr_pseudo_expr_Wu <- data.frame()
# 
# for (i in 1:length(genes_chrs)) {
#   genes_chr1 <- genes_chrs[[i]]
#   p_genes <- intersect(genes_chr1$p_arm$hgnc_symbol, rownames(seurat_Wu_pure))
#   q_genes <- intersect(genes_chr1$q_arm$hgnc_symbol, rownames(seurat_Wu_pure))
#   
#   if (length(p_genes) != 0){
#     # count 
#     counts_submatrix_p <- GetAssayData(seurat_Wu_pure, slot = "counts")[p_genes, , drop = FALSE]
#     count_sum_p <- colSums(counts_submatrix_p)
#     count_mean_p <- colMeans(counts_submatrix_p)
#     
#     # normalized expression
#     sct_expr_submatrix_p <- GetAssayData(seurat_Wu_pure, assay = "SCT", slot = "data")[p_genes, , drop = FALSE]
#     expr_sum_p <- colSums(sct_expr_submatrix_p)
#     expr_mean_p <- colMeans(sct_expr_submatrix_p)
#   } else {
#     count_sum_p <- count_mean_p <- expr_sum_p <- expr_mean_p <- NA
#   }
#   
#   if (length(q_genes) != 0){
#     counts_submatrix_q <- GetAssayData(seurat_Wu_pure, slot = "counts")[q_genes, , drop = FALSE]
#     count_sum_q <- colSums(counts_submatrix_q)
#     count_mean_q <- colMeans(counts_submatrix_q)
#     
#     sct_expr_submatrix_q <- GetAssayData(seurat_Wu_pure, assay = "SCT", slot = "data")[q_genes, , drop = FALSE]
#     expr_sum_q <- colSums(sct_expr_submatrix_q)
#     expr_mean_q <- colMeans(sct_expr_submatrix_q)
#   } else {
#     count_sum_q <- count_mean_q <- expr_sum_q <- expr_mean_q <- NA
#   }
#   
#   chr_pseudo_expr_Wu <- rbind(chr_pseudo_expr_Wu, data.frame(
#     dataset = "Wu_2021",
#     X = colnames(seurat_Wu_pure),
#     chromosome = paste0("chr", i),
#     count_sum_p = count_sum_p,
#     count_mean_p = count_mean_p,
#     expr_sum_p = expr_sum_p,
#     expr_mean_p = expr_mean_p,
#     count_sum_q = count_sum_q,
#     count_mean_q = count_mean_q,
#     expr_sum_q = expr_sum_q,
#     expr_mean_q = expr_mean_q)
#   )
# }
# 
# saveRDS(chr_pseudo_expr_Wu, file = file.path(sc_dir, "CARD_reference", paste0("chr_pseudo_expr_Wu_pure.rds")))
# 
# 
# # Calculate for Mo 2024 dataset
# chr_pseudo_expr_Mo <- data.frame()
# 
# for (i in 1:length(genes_chrs)) {
#   genes_chr1 <- genes_chrs[[i]]
#   p_genes <- intersect(genes_chr1$p_arm$hgnc_symbol, rownames(seurat_Mo_pure))
#   q_genes <- intersect(genes_chr1$q_arm$hgnc_symbol, rownames(seurat_Mo_pure))
#   
#   if (length(p_genes) != 0){
#     # count 
#     counts_submatrix_p <- GetAssayData(seurat_Mo_pure, slot = "counts")[p_genes, , drop = FALSE]
#     count_sum_p <- colSums(counts_submatrix_p)
#     count_mean_p <- colMeans(counts_submatrix_p)
#     
#     # normalized expression
#     sct_expr_submatrix_p <- GetAssayData(seurat_Mo_pure, assay = "SCT", slot = "data")[p_genes, , drop = FALSE]
#     expr_sum_p <- colSums(sct_expr_submatrix_p)
#     expr_mean_p <- colMeans(sct_expr_submatrix_p)
#   } else {
#     count_sum_p <- count_mean_p <- expr_sum_p <- expr_mean_p <- NA
#   }
#   
#   if (length(q_genes) != 0){
#     counts_submatrix_q <- GetAssayData(seurat_Mo_pure, slot = "counts")[q_genes, , drop = FALSE]
#     count_sum_q <- colSums(counts_submatrix_q)
#     count_mean_q <- colMeans(counts_submatrix_q)
#     
#     sct_expr_submatrix_q <- GetAssayData(seurat_Mo_pure, assay = "SCT", slot = "data")[q_genes, , drop = FALSE]
#     expr_sum_q <- colSums(sct_expr_submatrix_q)
#     expr_mean_q <- colMeans(sct_expr_submatrix_q)
#   } else {
#     count_sum_q <- count_mean_q <- expr_sum_q <- expr_mean_q <- NA
#   }
#   
#   chr_pseudo_expr_Mo <- rbind(chr_pseudo_expr_Mo, data.frame(
#     dataset = "Mo_2024",
#     X = colnames(seurat_Mo_pure),
#     chromosome = paste0("chr", i),
#     count_sum_p = count_sum_p,
#     count_mean_p = count_mean_p,
#     expr_sum_p = expr_sum_p,
#     expr_mean_p = expr_mean_p,
#     count_sum_q = count_sum_q,
#     count_mean_q = count_mean_q,
#     expr_sum_q = expr_sum_q,
#     expr_mean_q = expr_mean_q)
#   )
# }
#  
# saveRDS(chr_pseudo_expr_Mo, file = file.path(sc_dir, "CARD_reference", paste0("chr_pseudo_expr_Mo_pure.rds")))
# 
# # Generate psuedo-bulk for each cell type.
# Idents(seurat_Wu_pure) <- seurat_Wu_pure[["cell_type_group"]][, 1]
# counts_submatrix <- GetAssayData(seurat_Wu_pure, slot = "counts")[signature_genes_union, , drop = FALSE]
# 
# # Aggregate (average) expression per gene per group
# library(Matrix)
# pseudobulk_matrix <- sapply(levels(Idents(seurat_Wu_pure)), function(group) {
#   cells_in_group <- WhichCells(seurat_Wu_pure, idents = group)
#   if (length(cells_in_group) > 1) {
#     Matrix::rowMeans(sct_data[, cells_in_group, drop = FALSE])
#   } else {
#     sct_data[, cells_in_group]  # Single cell fallback
#   }
# })
# saveRDS(pseudobulk_matrix, file = file.path(sc_dir, "CARD_reference", paste0("B_matrix_Wu_all_genes.rds")))
# 
# pseudobulk_matrix_signature <- pseudobulk_matrix[signature_genes, ]
# saveRDS(pseudobulk_matrix_signature, file = file.path(sc_dir,"CARD_reference", paste0("B_matrix_Wu_signature_genes.rds")))
# 
# 
# Idents(seurat_Mo_pure) <- seurat_Mo_pure[["cell_type_group"]][, 1]
# sct_data <- GetAssayData(seurat_Mo_pure, assay = "SCT", slot = "data")
# 
# # Aggregate (average) expression per gene per group
# pseudobulk_matrix <- sapply(levels(Idents(seurat_Mo_pure)), function(group) {
#   cells_in_group <- WhichCells(seurat_Mo_pure, idents = group)
#   if (length(cells_in_group) > 1) {
#     Matrix::rowMeans(sct_data[, cells_in_group, drop = FALSE])
#   } else {
#     sct_data[, cells_in_group]  # Single cell fallback
#   }
# })
# saveRDS(pseudobulk_matrix, file = file.path(sc_dir, "CARD_reference", paste0("B_matrix_Mo_all_genes.rds")))
# 
# pseudobulk_matrix_signature <- pseudobulk_matrix[signature_genes, ]
# saveRDS(pseudobulk_matrix_signature, file = file.path(sc_dir, "CARD_reference", paste0("B_matrix_Mo_signature_genes.rds")))
# 
# 
# sc_dir <- "/fh/working/sun_w/zsui/st2image_data/brca_scRNAseq_datasets"
# B_Wu <- readRDS(file = file.path(sc_dir,"CARD_reference", paste0("B_matrix_Wu_signature_genes.rds")))
# B_Mo <- readRDS(file = file.path(sc_dir,"CARD_reference", paste0("B_matrix_Mo_signature_genes.rds")))
# 
# common_genes <- intersect(rownames(B_Wu), rownames(B_Mo))
# ct_wu <- colnames(B_Wu)
# ct_mo <- colnames(B_Mo)
# 
# cor_matrix_full <- cor(B_Wu[common_genes, ct_wu], B_Mo[common_genes, ct_mo])
# cor_matrix_full <- cor_matrix_full[sort(rownames(cor_matrix_full)), sort(colnames(cor_matrix_full))]
# 
# red_gradient <- colorRampPalette(c("white", "pink", "red"))(100)
# number_labels <- round(cor_matrix_full, 2)
# 
# pdf(file.path(sc_dir, "CARD_reference", "B_Wu_vs_B_Mo.pdf"))
# pheatmap(
#   cor_matrix_full,
#   name = "Correlation",
#   cluster_rows = FALSE,
#   cluster_cols = FALSE,
#   color = red_gradient,
#   na_col = "gray90",
#   display_numbers = number_labels,
#   number_color = "black",
#   fontsize_number = 10,
#   main = "Cross-Correlation: B_Wu (rows) vs B_Mo (cols)",
#   breaks = seq(0, 1, length.out = 101)
# )
# dev.off()








