
# Setup -------------------------------------------------------------------

suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratDisk)
  library(glmGamPoi)
  library(dplyr)
  library(tidyr)
  library(tidyverse)
  library(Matrix)
  library(ggplot2)
  library(patchwork)
  library(biomaRt)
  library(pheatmap)
  library(stringr)
  library(car)
})

set.seed(1)
options(future.globals.maxSize = 300 * 1024^3)  # 300 GB

sc_dir <- "/fh/working/sun_w/zsui/st2image_data/brca_scRNAseq_datasets"
out_dir <- file.path(sc_dir, "CARD_reference")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# In "/fh/working/sun_w/zsui/st2image_data/brca_scRNAseq_datasets/":
# + "seurat_merged_Mo_Wu_pure_clustered.rds": Two SC datasets merged without integration
# + "seurat_integrated_Mo_Wu_pure_clustered.rds": Two SC datasets merged with integration
# + "seurat_Wu_pure_clustered.rds":
# + "seurat_Mo_pure_clustered.rds":

seurat_Wu_pure <- readRDS(file.path(sc_dir, "seurat_Wu_pure_clustered.rds"))
seurat_Mo_pure <- readRDS(file.path(sc_dir, "seurat_Mo_pure_clustered.rds"))

# Ensure required columns exist
stopifnot("cell_type_group" %in% colnames(seurat_Wu_pure@meta.data))
stopifnot("cell_type_group" %in% colnames(seurat_Mo_pure@meta.data))

# For improved per-sample paths we need a sample column.
# If not present, fall back to per-ident (keeps original behavior).
if (!"Sample" %in% colnames(seurat_Wu_pure@meta.data)) {
  seurat_Wu_pure$Sample <- seurat_Wu_pure$orig.ident
}
if (!"Sample" %in% colnames(seurat_Mo_pure@meta.data)) {
  seurat_Mo_pure$Sample <- seurat_Mo_pure$orig.ident
}

# Generate reference datasets for default CARD ------------------------------------

# Wu
ref_dir <- file.path(sc_dir, "CARD_reference", "Wu_2021_filtered")
count_matrix <- seurat_Wu_pure[["RNA"]]$counts
Matrix::writeMM(obj = count_matrix, file = file.path(ref_dir, "count_matrix_sparse.mtx"))
write(x = rownames(count_matrix), file = file.path(ref_dir, "count_matrix_genes.tsv"))
write(x = colnames(count_matrix), file = file.path(ref_dir, "count_matrix_barcodes.tsv"))
metadata <- seurat_Wu_pure@meta.data
write.csv(x = metadata, file = file.path(ref_dir, "metadata.csv"))

# Mo
ref_dir <- file.path(sc_dir, "CARD_reference", "Mo_2024_filtered")
count_matrix <- seurat_Mo_pure[["RNA"]]$counts
Matrix::writeMM(obj = count_matrix, file = file.path(ref_dir, "count_matrix_sparse.mtx"))
write(x = rownames(count_matrix), file = file.path(ref_dir, "count_matrix_genes.tsv"))
write(x = colnames(count_matrix), file = file.path(ref_dir, "count_matrix_barcodes.tsv"))
metadata <- seurat_Mo_pure@meta.data
write.csv(x = metadata, file = file.path(ref_dir, "metadata.csv"))


# Generate B matrices for custom CARD -------------------------------------


# + Identify signature genes ----------------------------------------------

# Curated Mo markers
Mo_marker_genes <- c(
  "ALDH1A3","ADGRE1","ACKR1","ACTA2","APOE","APOC1","AOC3","ATP1B2","ABCC9","AMBP",
  "ANKRD30A","AZGP1","AGR2","ANIN","AR","AKT1","BANK1","BST2","BTLA","BIRC5","CCL5",
  "CCL4","CCR7","CALD1","CPA3","CD1C","CD3D","CD3E","CD4","CD7","CD8A","CD8B","CD9",
  "CD14","CD19","CD28","CD44","CD68","CD69","CD74","CD79A","CD79B","CD86","CD96",
  "CD160","CD163","CD244","CDH11","CDCA3","CDCA8","COL1A1","COL1A2","COL3A1","COL6A1",
  "COL6A2","C1QA","C1QB","C1QC","CEP55","CIDEA","CEBPA","CSF1R","CILP","CPE","CFD",
  "CLEC4C","CTLA4","CAV1","CCNA2","CDH1","DCN","DES","DST","DPP4","EPCAM","EMP1","EBF1",
  "ERBB4","ENPP3","ERBB2","ESR1","EOMES","FABP4","FAM64A","FBLN1","FBLN2","FCGR3A",
  "FCGR1A","FOXP3","FYN","FN1","FLT1","FST","FCRL5","FCHSD2","FASLG","GNLY","GSN","GZMA",
  "GZMB","GZMH","GZMK","GJA4","GPR183","HIGD1B","HPGDS","HAVCR2","HLA-DPB1","HLA-DRA",
  "HLA-DPA1","HLA-DRB1","IGLC2","IGLC3","IGHG1","IGFBP7","IGHA1","IGHG3","IGHG4","IGHA2",
  "IGKC","IFI30","IFNG","ITGAM","ITGB1","ITGBL1","IL1A","IL1B","IL2RA","IL2RB","IL3RA",
  "IL7R","IRF7","IKZF2","ICOS","JCHAIN","KRT5","KRT8","KRT14","KRT16","KRT17","KRT18",
  "KRT19","KLRF1","KIR2DL3","KIT","LEF1","LRRC15","LTF","LUM","LY6D","LRRC8A","LYZ",
  "LAG3","LAMP1","LAMP2","LEP","LILRA4","LGALS3","MEF2C","MS4A1","MZB1","MGP","MCAM",
  "MRC1","MSR1","MFAP5","MYL9","MMP7","MIR205HG","MT1X","MUC1","MUCL1","MS4A2","MKI67",
  "MELK","MYC","NKG7","NCAM1","NDUFA4L2","NRP1","NR3C1","NUF2","KLRB1","KLRD1","KDR",
  "KCNJ8","KIF15","OLFML3","OXTR","PECAM1","PDGFRB","POSTN","PTN","PDGFRA","PRF1","PDPN",
  "PRKG1","PNPLA2","PGR","PBK","PTGDS","P2RX5","PDCD1","PPARG","RGS5","RNASE4","RNASE1",
  "S100A4","SELL","SELENOP","SCGB2A2","SFRP2","SLPI","SVEP1","SOX17","SPP1","SMOC2",
  "SLC20A1","STC2","SLC7A10","SLC36A2","SPARC","SPC25","SHOX2","SKA1","THY1","TCF7",
  "TCF4","TAGLN","TIMP1","TPSB2","TPSAB1","TPX2","TIGIT","TLR2","TLR7","TLR9","TTK",
  "TOP2A","TNFRSF17","TNFRSF18","VWF","VSIR","XCL1","XCL2"
)

identify_markers <- function(obj_Mo, obj_Wu, curated = Mo_marker_genes, top_n = 150,
                             min.pct = 0.25, logfc.threshold = 1, use_roc = FALSE) {
  Idents(obj_Mo) <- "cell_type_group"
  Idents(obj_Wu) <- "cell_type_group"
  test.use <- if (use_roc) "roc" else "wilcox"
  m_Mo <- FindAllMarkers(obj_Mo, only.pos = TRUE, min.pct = min.pct,
                         logfc.threshold = logfc.threshold, test.use = test.use)
  m_Wu <- FindAllMarkers(obj_Wu, only.pos = TRUE, min.pct = min.pct,
                         logfc.threshold = logfc.threshold, test.use = test.use)
  
  top_Mo <- m_Mo %>% group_by(cluster) %>% top_n(n = top_n, wt = avg_log2FC) %>% 
    ungroup() %>%
    split(.$cluster)
  top_Mo <- lapply(top_Mo, function(ct) unique(c(ct$gene, curated)))
  
  top_Wu <- m_Wu %>% group_by(cluster) %>% top_n(n = top_n, wt = avg_log2FC) %>% 
    ungroup() %>%
    split(.$cluster)
  top_Wu <- lapply(top_Wu, function(ct) unique(ct$gene))
  
  common_cts <- intersect(names(top_Mo), names(top_Wu))
  
  sig_consensus <- lapply(common_cts, function(ct) {
    intersect(top_Mo[[ct]], top_Wu[[ct]])
  }) 
  names(sig_consensus) <- common_cts
  
  # Consensus 
  sig_consensus[["Plasmablasts"]] <- top_Wu$Plasmablasts
  sig_consensus[["Plasma"]] <- top_Mo$Plasma
  
  list(Mo = top_Mo, Wu = top_Wu, Consensus = sig_consensus)
}

markers_identified <- identify_markers(seurat_Mo_pure, seurat_Wu_pure)
signature_genes <- markers_identified$Consensus
signature_genes <- stack(signature_genes)
signature_genes <- unique(signature_genes$values)

saveRDS(markers_identified, file = file.path(sc_dir, "CARD_reference", "markers_identified.rds"))
saveRDS(signature_genes, file = file.path(sc_dir, "CARD_reference", "signature_genes.rds"))

# + Pseudobulk with count data ---------------------------------------------

# Function to compute pseudobulk per group
get_pseudobulk_expression <- function(
    seurat_obj,
    gene_set,
    group_col = "cell_type_group",
    scale_factor = 1e4  # same as NormalizeData default
    ) {
  # 1. Filter valid genes
  if (!group_col %in% names(seurat_obj@meta.data)) {
    stop("Error: The specified 'group_col' was not found in the Seurat object's metadata.")
  }
  
  gene_set_present <- intersect(gene_set, rownames(seurat_obj))
  if (length(gene_set_present) == 0) {
    stop("Error: None of the specified genes were found in the Seurat object.")
  }
  
  # 2. Set cell identities to group_col
  Idents(seurat_obj) <- seurat_obj[[group_col]][,1]
  
  # 3. Extract raw counts
  counts_mat <- GetAssayData(seurat_obj, slot = "counts")[gene_set_present, , drop = FALSE]
  
  # 4. Compute pseudobulk: genes x groups (Sum counts for each group)
  pseudobulk_raw <- sapply(levels(Idents(seurat_obj)), function(group) {
    cells_in_group <- WhichCells(seurat_obj, idents = group)
    Matrix::rowSums(counts_mat[, cells_in_group, drop = FALSE])
  })
  
  pseudobulk_raw <- as.matrix(pseudobulk_raw)
  rownames(pseudobulk_raw) <- gene_set_present
  colnames(pseudobulk_raw) <- levels(Idents(seurat_obj))
  
  # 5. Normalize using Counts Per Million (CPM)
  # CPM = (Gene Counts / Total Counts in Group) * 1,000,000
  col_sums <- colSums(pseudobulk_raw)
  col_sums[col_sums == 0] <- 1
  pseudobulk_cpm <- sweep(pseudobulk_raw, 2, col_sums, FUN = "/") * 1e6
  
  # 6. Log-normalized pseudobulk (like Seurat)
  # Average the expression of the log-normalized data
  pseudobulk_log <- sweep(pseudobulk_raw, 2, col_sums, FUN = "/") * scale_factor
  pseudobulk_log <- log1p(pseudobulk_log) 
  
  # 7. SCTransform residuals pseudobulk
  if (!"SCT" %in% Assays(seurat_obj)) {
    message("SCTransform not found, running now...")
    seurat_obj <- SCTransform(seurat_obj, verbose = FALSE)
  }
  # Average the expression of the SCT-normalized data
  sct_data <- GetAssayData(seurat_obj, slot = "scale.data", assay = "SCT")
  gene_set_present2 <- intersect(rownames(sct_data), gene_set_present)
  sct_data <- sct_data[gene_set_present2, , drop = FALSE]

  pseudobulk_sct <- sapply(levels(Idents(seurat_obj)), function(group) {
    cells_in_group <- WhichCells(seurat_obj, idents = group)
    Matrix::rowMeans(sct_data[, cells_in_group, drop = FALSE])
  })

  pseudobulk_sct <- as.matrix(pseudobulk_sct)
  rownames(pseudobulk_sct) <- gene_set_present2
  colnames(pseudobulk_sct) <- levels(Idents(seurat_obj))
  
  return(list(
    Raw = pseudobulk_raw,     # raw summed counts
    CPM = pseudobulk_cpm,     # counts per million
    LogNorm = pseudobulk_log, # log-normalized (like Seurat)
    SCT = pseudobulk_sct      # average Pearson residuals
  ))
}

get_pseudobulk_expression_per_sample <- function(
    seurat_obj,
    gene_set,
    group_col = "cell_type_group",
    sample_col = "Sample",
    scale_factor = 1e4   # K=1e4 per Seurat default
) {
  if (!all(c(group_col, sample_col) %in% colnames(seurat_obj@meta.data)))
    stop("Required columns not found in meta.data")
  
  # Work on present genes only
  gene_set_present <- intersect(gene_set, rownames(seurat_obj))
  if (length(gene_set_present) == 0)
    stop("None of the specified genes were found in the Seurat object.")
  
  # Convenience aliases
  meta <- seurat_obj@meta.data
  counts <- GetAssayData(seurat_obj, slot = "counts")[gene_set_present, , drop = FALSE]
  
  # Split cells by (group, sample)
  split_keys <- interaction(meta[[group_col]], meta[[sample_col]], drop = TRUE)
  cell_index <- split(colnames(seurat_obj), split_keys)
  
  # Per-sample aggregates
  raw_ls <- list(); cpm_ls <- list(); log_ls <- list()
  for (key in names(cell_index)) {
    cells <- cell_index[[key]]
    if (length(cells) == 0) next
    grp <- as.character(meta[cells[1], group_col])
    smp <- as.character(meta[cells[1], sample_col])
    
    sub_counts <- counts[, cells, drop = FALSE]
    # Per-sample raw sums
    Y_gcs <- Matrix::rowSums(sub_counts)  # Y_{g,c,s}
    T_cs  <- sum(Y_gcs)                    # T_{c,s}
    
    # CPM per sample and LogNorm per-sample (log1p counts-per-10k)
    CPM_gcs <- if (T_cs > 0) (Y_gcs / T_cs) * 1e6 else rep(0, length(Y_gcs))
    LOG_gcs <- if (T_cs > 0) log1p((Y_gcs / T_cs) * scale_factor) else rep(0, length(Y_gcs))
    
    raw_ls[[length(raw_ls)+1]] <- data.frame(gene = names(Y_gcs), group = grp, sample = smp, value = as.numeric(Y_gcs))
    cpm_ls[[length(cpm_ls)+1]] <- data.frame(gene = names(CPM_gcs), group = grp, sample = smp, value = as.numeric(CPM_gcs))
    log_ls[[length(log_ls)+1]] <- data.frame(gene = names(LOG_gcs), group = grp, sample = smp, value = as.numeric(LOG_gcs))
  }
  
  # Bind + average across samples within group
  agg_fun <- function(df) {
    df %>%
      group_by(gene, group) %>%
      summarise(value = mean(value, na.rm = TRUE), .groups = "drop") %>%
      pivot_wider(names_from = group, values_from = value) %>%
      column_to_rownames("gene") %>%
      as.matrix()
  }
  
  Raw     <- agg_fun(bind_rows(raw_ls))
  CPM     <- agg_fun(bind_rows(cpm_ls))
  LogNorm <- agg_fun(bind_rows(log_ls))
  
  # SCT per-sample means of residuals, then average
  if (!"SCT" %in% Assays(seurat_obj)) {
    message("SCTransform not found, running now...")
    seurat_obj <- SCTransform(seurat_obj, verbose = FALSE)
  }
  sct <- GetAssayData(seurat_obj, slot = "scale.data", assay = "SCT")
  sct <- sct[intersect(rownames(sct), gene_set_present), , drop = FALSE]
  
  sct_ls <- list()
  for (key in names(cell_index)) {
    cells <- cell_index[[key]]
    if (length(cells) == 0) next
    grp <- as.character(meta[cells[1], group_col])
    smp <- as.character(meta[cells[1], sample_col])
    m <- Matrix::rowMeans(sct[, cells, drop = FALSE])
    sct_ls[[length(sct_ls)+1]] <- data.frame(gene = names(m), group = grp, sample = smp, value = as.numeric(m))
  }
  SCT <- agg_fun(bind_rows(sct_ls))
  
  list(Raw = Raw, CPM = CPM, LogNorm = LogNorm, SCT = SCT)
}

# Compute pseudobulk
B_pb_Wu_orig <- get_pseudobulk_expression(seurat_obj = seurat_Wu_pure, gene_set = signature_genes)
B_pb_Mo_orig <- get_pseudobulk_expression(seurat_obj = seurat_Mo_pure, gene_set = signature_genes)

B_pb_Wu  <- get_pseudobulk_expression_per_sample(seurat_obj = seurat_Wu_pure, gene_set = signature_genes)
B_pb_Mo  <- get_pseudobulk_expression_per_sample(seurat_obj = seurat_Mo_pure, gene_set = signature_genes)


# + Generate pseudo-genes for chromosomal arms ------------------------------

get_all_chrom_arm_genes <- function(chrs = 1:22) {
  # Connect to the Ensembl human genes database
  ensembl <- biomaRt::useMart(biomart = "ensembl", dataset = "hsapiens_gene_ensembl")
  
  # Get all genes for the specified chromosomes with their cytoband location
  all_genes <- getBM(
    attributes = c("hgnc_symbol", "ensembl_gene_id", "chromosome_name", "start_position", "end_position", "band"),
    filters = "chromosome_name",
    values = as.character(chrs),
    mart = ensembl
  ) %>%
    # Filter out genes without an official HGNC symbol or a defined cytoband
    dplyr::filter(hgnc_symbol != "", !is.na(band))
  
  # Initialize an empty list to store genes for each chromosome arm
  arm_gene_list <- list()
  # Loop through each chromosome and arm (p and q)
  for (chr in as.character(chrs)) {
    for (arm in c("p", "q")) {
      arm_label <- paste0("chr", chr, "_", arm)
      genes <- all_genes %>%
        dplyr::filter(chromosome_name == chr, grepl(paste0("^", arm), band)) %>%
        pull(hgnc_symbol) %>%
        unique()
      if (length(genes) > 0) {
        arm_gene_list[[arm_label]] <- genes
      }
    }
  }
  
  return(arm_gene_list)
}

get_informative_arm_genes <- function(seurat_obj, genes_chrs, group_col = "cell_type_group", min_pct = 0.1) {
  
  # Set the identity of the Seurat object to the specified grouping column
  Idents(seurat_obj) <- factor(seurat_obj@meta.data[[group_col]])
  
  # if (length(VariableFeatures(seurat_obj)) == 0) {
  #   stop("Error: Run FindVariableFeatures() first.")
  # }
  # informative_global <- VariableFeatures(seurat_obj)
  
  # Get the raw count data from the Seurat object
  counts <- GetAssayData(seurat_obj, slot = "counts")

  out_list <- list() 
  
  for (arm_label in names(genes_chrs)) {
    # 1. Intersect arm genes with variable features first
    # arm_genes <- intersect(genes_chrs[[arm_label]], informative_global)
    # arm_genes <- intersect(arm_genes, rownames(counts)) # Also keep only genes in counts
    
    # Get the list of genes for the current arm
    arm_genes <- genes_chrs[[arm_label]]
    # Keep only the arm genes that are present in the count matrix
    arm_genes <- intersect(arm_genes, rownames(counts))
    
    if (length(arm_genes) == 0) next
    
    # 2. For each group, check detection
    # Compute per-group detection: keep genes detected in any group >= min_pct
    per_group_detection <- sapply(levels(Idents(seurat_obj)), function(group) {
      cells <- WhichCells(seurat_obj, idents = group)
      # Calculate the proportion of cells with >0 expression for each gene
      gene_detect <- Matrix::rowSums(counts[arm_genes, cells, drop = FALSE] > 0) / length(cells)
      return(gene_detect)
    })
    # Identify genes that are detected in at least one group above the `min_pct` threshold
    if (length(arm_genes) == 1) {
      if (any(per_group_detection >= min_pct)) {
        keep_genes <- arm_genes
      } else {
        keep_genes <- NULL
      }
    } else{
      keep_genes <- rownames(per_group_detection)[apply(per_group_detection, 1, function(x) any(x >= min_pct))]
    }
   
    if (length(keep_genes) >= 1) {
      out_list[[arm_label]] <- keep_genes
    }
  }
  return(out_list)
}


# # ---- Pooled-by-type arms 
# get_chrom_pseudogene_expression <- function(
#     seurat_obj,
#     chrs = 1:22,
#     group_col = "cell_type_group",
#     min_pct = 0.1,
#     scale_factor = 1e4
# ) {
#   # 1. Get all chromosomal arm genes 
#   genes_chrs <- get_all_chrom_arm_genes(chrs = chrs)
#   
#   # 2. Get informative genes for each arm
#   arm_features <- get_informative_arm_genes(seurat_obj, genes_chrs, group_col, min_pct)
#   if (length(arm_features) == 0) stop("No informative arms found.")
#   
#   Idents(seurat_obj) <- factor(seurat_obj@meta.data[[group_col]])
#   
#   # 3. CPM-like method (raw counts)
#   counts <- GetAssayData(seurat_obj, slot = "counts")
#   out_list_cpm <- list()
#   
#   for (arm in names(arm_features)) {
#     genes <- arm_features[[arm]]
#     if (length(genes) == 0) next
#     
#     for (group in levels(Idents(seurat_obj))) {
#       cells <- WhichCells(seurat_obj, idents = group)
#       counts_sub <- counts[, cells, drop = FALSE]
#       
#       arm_sum <- sum(Matrix::rowSums(counts_sub[genes, , drop = FALSE]))
#       total_counts <- sum(counts_sub)
#       arm_cpm <- (arm_sum / total_counts) * 1e6
#       
#       out_list_cpm[[length(out_list_cpm) + 1]] <- data.frame(
#         Arm = arm,
#         cell_type_group = group,
#         Expression = arm_cpm
#       )
#     }
#   }
#   
#   pseudo_df_cpm <- do.call(rbind, out_list_cpm) %>%
#     tidyr::pivot_wider(names_from = cell_type_group, values_from = Expression) %>%
#     tibble::column_to_rownames("Arm") %>%
#     as.matrix()
#   
#   # 4. LogNorm 
#   pseudo_df_lognorm <- log1p((pseudo_df_cpm / 1e6) * scale_factor)
#   
#   # 5. SCT method (average Pearson residuals)
#   sct_data <- GetAssayData(seurat_obj, slot = "scale.data", assay = "SCT")
#   out_list_sct <- list()
#   
#   for (arm in names(arm_features)) {
#     genes <- arm_features[[arm]]
#     if (length(genes) == 0) next
#     
#     for (group in levels(Idents(seurat_obj))) {
#       cells <- WhichCells(seurat_obj, idents = group)
#       sct_sub <- sct_data[genes, cells, drop = FALSE]
#       
#       pseudo_sct <- Matrix::colMeans(sct_sub)
#       pseudo_sct_mean <- mean(pseudo_sct, na.rm = TRUE)
#       
#       out_list_sct[[length(out_list_sct) + 1]] <- data.frame(
#         Arm = arm,
#         cell_type_group = group,
#         Expression = pseudo_sct_mean
#       )
#     }
#   }
#   
#   pseudo_df_sct <- do.call(rbind, out_list_sct) %>%
#     tidyr::pivot_wider(names_from = cell_type_group, values_from = Expression) %>%
#     tibble::column_to_rownames("Arm") %>%
#     as.matrix()
#   
#   # seurat_obj <- AddModuleScore(
#   #   object = seurat_obj,
#   #   features = arm_features,
#   #   assay = "SCT",
#   #   name = "ArmScore"
#   # )
#   # 
#   # score_col_names <- paste0("ArmScore", 1:length(arm_features))
#   # 
#   # pseudo_df_modscore <- seurat_obj@meta.data %>%
#   #   group_by(.data[[group_col]]) %>%
#   #   summarise(across(all_of(score_col_names), mean, na.rm = TRUE)) %>%
#   #   tidyr::pivot_longer(-!!sym(group_col), names_to = "Arm", values_to = "Expression") %>%
#   #   tidyr::pivot_wider(names_from = !!sym(group_col), values_from = Expression) %>%
#   #   as.data.frame() %>%
#   #   tibble::column_to_rownames("Arm") %>%
#   #   as.matrix()
#   # 
#   # rownames(pseudo_df_modscore) <- names(arm_features)
#   
#   return(list(
#     CPM = pseudo_df_cpm,
#     LogNorm = pseudo_df_lognorm,
#     SCT = pseudo_df_sct,
#     InformativeGenes = arm_features
#   ))
# }
# 

# ---- per-sample arms + log-ratios + z-score 

get_chrom_arm_expression <- function(
    seurat_obj, chrs = 1:22, group_col = "cell_type_group", sample_col = "Sample",
    min_detect_prop = 0.10, scale_factor = 1e4, epsilon = 1e-3, zscore = TRUE
) {
  # --- Step 1: Get chromosome arm gene lists and filter for informative genes ---
  
  # Retrieve all genes for each chromosome arm
  genes_chrs <- get_all_chrom_arm_genes(chrs)
  
  # Filter for genes that are consistently expressed (informative)
  arm_features <- get_informative_arm_genes(seurat_obj, genes_chrs, group_col, min_pct = min_detect_prop)
  if (length(arm_features) == 0) stop("No informative arms found. Try lowering `min_detect_prop`.")
  
  # --- Step 2: Aggregate expression per sample and cell group ---
  
  # Helper function to calculate aggregated expression for each arm per sample and group
  .arm_features_per_sample <- function(seurat_obj, arm_genes, group_col, sample_col, scale_factor) {
    # Extract metadata columns for grouping and samples
    meta <- seurat_obj@meta.data[, c(group_col, sample_col), drop = FALSE]
    Idents(seurat_obj) <- factor(meta[[group_col]])
    counts <- GetAssayData(seurat_obj, slot = "counts")
    
    # Create an index to split cells by group and sample
    split_idx <- split(seq_len(nrow(meta)), interaction(meta[[group_col]], meta[[sample_col]], drop = TRUE))
    CPM_ls <- list(); LOG_ls <- list()
    
    # Loop through each sample-group combination
    for (key in names(split_idx)) {
      idx <- split_idx[[key]]
      grp <- unique(meta[[group_col]][idx])
      smp <- unique(meta[[sample_col]][idx])
      if (!identical(key, paste0(grp, ".", smp))) stop("Error")
      
      cells <- colnames(seurat_obj)[idx]
      sub_counts <- counts[, cells, drop = FALSE]
      total <- sum(sub_counts); if (total == 0) next
      
      # Calculate total counts for each gene
      gene_sums <- Matrix::rowSums(sub_counts)
      
      # For each chromosome arm, sum the counts of its informative genes
      for (arm in names(arm_genes)) {
        genes <- intersect(arm_genes[[arm]], names(gene_sums))
        if (length(genes) == 0) next
        arm_sum <- sum(gene_sums[genes])
        CPM_ls[[length(CPM_ls)+1]] <- data.frame(Arm = arm, group = grp, sample = smp, Value = (arm_sum / total) * 1e6)
        LOG_ls[[length(LOG_ls)+1]] <- data.frame(Arm = arm, group = grp, sample = smp, Value = log1p((arm_sum / total) * scale_factor))
      }
    }
    
    # Check if SCTransform has been run, and if not, run it
    if (!"SCT" %in% Assays(seurat_obj)) seurat_obj <- SCTransform(seurat_obj, verbose = FALSE)
    sct <- GetAssayData(seurat_obj, slot = "scale.data", assay = "SCT")
    SCT_ls <- list()
    # Loop through each sample-group combination to calculate aggregated SCT values
    for (key in names(split_idx)) {
      idx <- split_idx[[key]]
      grp <- unique(meta[[group_col]][idx])
      smp <- unique(meta[[sample_col]][idx])
      if (!identical(key, paste0(grp, ".", smp))) stop("Error")
      
      cells <- colnames(seurat_obj)[idx]
      
      for (arm in names(arm_genes)) {
        genes <- intersect(arm_genes[[arm]], rownames(sct))
        if (length(genes) == 0) next
        # Aggregate SCT values by taking the mean of cell-level means for that arm
        m <- Matrix::colMeans(sct[genes, cells, drop = FALSE])
        SCT_ls[[length(SCT_ls)+1]] <- data.frame(Arm = arm, group = grp, sample = smp, Value = mean(m, na.rm = TRUE))
      }
    }
    
    list(
      CPM = bind_rows(CPM_ls),
      LogNorm = bind_rows(LOG_ls),
      SCT = bind_rows(SCT_ls)
    )
  }
  
  per_sample <- .arm_features_per_sample(seurat_obj, arm_genes = arm_features,
                                         group_col = group_col, sample_col = sample_col,
                                         scale_factor = scale_factor)
  
  # --- Step 3: Aggregate per cell type group and calculate log-ratios ---
  
  # Helper function to aggregate sample-level data into a cell-type-level matrix
  .aggregate_arm_features <- function(arm_df) {
    arm_df %>%
      group_by(Arm, group) %>%
      summarise(Value = mean(Value, na.rm = TRUE), .groups = "drop") %>%
      pivot_wider(names_from = group, values_from = Value) %>%
      column_to_rownames("Arm") %>%
      as.matrix()
  }
  
  # Create matrices for each normalization method
  B_cpm <- .aggregate_arm_features(per_sample$CPM)
  B_log <- .aggregate_arm_features(per_sample$LogNorm)
  B_sct <- .aggregate_arm_features(per_sample$SCT)
  
  # Helper function to compute log-ratios of p- and q-arms for each chromosome
  .make_arm_logratios <- function(arm_mat, epsilon) {
    arms <- rownames(arm_mat)
    p_arms <- grep("_p$", arms, value = TRUE)
    q_arms <- grep("_q$", arms, value = TRUE)
    # Find chromosomes that have both p and q arms
    chrs <- intersect(sub("_p$","", p_arms), sub("_q$","", q_arms))
    if (length(chrs) == 0) return(NULL)
    
    LR <- matrix(NA_real_, nrow = length(chrs), ncol = ncol(arm_mat),
                 dimnames = list(chrs, colnames(arm_mat)))
    # Calculate the log-ratio for each chromosome
    for (chr in chrs) {
      p_row <- paste0(chr, "_p"); q_row <- paste0(chr, "_q")
      LR[chr, ] <- log((arm_mat[p_row, ] + epsilon) / (arm_mat[q_row, ] + epsilon))
    }
    LR
  }
  
  # Calculate log-ratios for each normalization method
  LR_cpm <- .make_arm_logratios(B_cpm, epsilon)
  LR_log <- .make_arm_logratios(B_log, epsilon)
  LR_sct <- .make_arm_logratios(B_sct, epsilon)
  
  # --- Step 4: Optional Z-score scaling and collinearity diagnostics ---
  
  # Helper function to standardize (Z-score) rows of a matrix
  .standardize_rows <- function(mat) t(scale(t(mat)))
  
  if (zscore) {
    B_cpm <- .standardize_rows(B_cpm)
    B_log <- .standardize_rows(B_log)
    B_sct <- .standardize_rows(B_sct)
    if (!is.null(LR_cpm)) LR_cpm <- .standardize_rows(LR_cpm)
    if (!is.null(LR_log)) LR_log <- .standardize_rows(LR_log)
    if (!is.null(LR_sct)) LR_sct <- .standardize_rows(LR_sct)
  }
  
  
  list(
    CPM = B_cpm,
    LogNorm = B_log,
    SCT = B_sct,
    LR = list(CPM = LR_cpm, LogNorm = LR_log, SCT = LR_sct),
    InformativeGenes = arm_features
  )
  
}


# # Compute ARMS: ORIGINAL vs IMPROVED
# B_chr_Wu_orig <- get_chrom_pseudogene_expression(seurat_Wu_pure)
# B_chr_Mo_orig <- get_chrom_pseudogene_expression(seurat_Mo_pure)

B_chr_Wu  <- get_chrom_arm_expression(seurat_Wu_pure, zscore = FALSE)
B_chr_Mo  <- get_chrom_arm_expression(seurat_Mo_pure, zscore = FALSE)



# + Augment gene-level B matrix with chromosomal arm features -------------

augment <- function(
    B_genes, B_arms,
    strategy = c("harmonize", "rowstandardize", "variance_match", "weight"),
    # strategy-specific knobs:
    alpha = 1,                           # for strategy = "weight"
    target_sd = c("mean", "median", "value"),  # for "variance_match"
    target_sd_value = NULL,              # numeric, used when target_sd = "value"
    center_all = TRUE, scale_all = TRUE  # for "rowstandardize"
) {
  stopifnot(is.matrix(B_genes), is.matrix(B_arms))
  strategy <- match.arg(strategy)
  
  # --- helper: ensure same column order (CTs) ---
  if (!identical(colnames(B_genes), colnames(B_arms))) {
    common <- intersect(colnames(B_genes), colnames(B_arms))
    if (length(common) == 0) stop("No overlapping columns between B_genes and B_arms.")
    B_genes <- B_genes[, common, drop = FALSE]
    B_arms  <- B_arms[,  common, drop = FALSE]
  }
  
  # --- helper: rbind without duplicate feature rows ---
  .rbind_no_dupes <- function(A, B) {
    extra <- setdiff(rownames(B), rownames(A))
    rbind(A, B[extra, , drop = FALSE])
  }
  
  # --- outputs ---
  audit <- list(strategy = strategy, notes = character(0))
  
  if (strategy == "harmonize") {
    # Assumes B_arms already in the same units as B_genes (CPM / LogNorm / SCT)
    out <- .rbind_no_dupes(B_genes, B_arms)
    audit$notes <- c(audit$notes, "Harmonized units assumed; no rescaling applied.")
    
  } else if (strategy == "rowstandardize") {
    # Standardize each row (gene or arm): mean=0, sd=1 across CTs
    B_full <- .rbind_no_dupes(B_genes, B_arms)
    out <- t(scale(t(B_full), center = center_all, scale = scale_all))
    audit$notes <- c(audit$notes, sprintf("Row-standardized all features (center=%s, scale=%s).",
                                          center_all, scale_all))
    
  } else if (strategy == "variance_match") {
    # Rescale each arm row so its sd equals target gene sd (mean/median/value)
    # 1) pick target SD from genes
    gsds <- apply(B_genes, 1, sd, na.rm = TRUE)
    gsds <- gsds[is.finite(gsds) & gsds > 0]
    if (length(gsds) == 0) stop("All gene rows have zero/NA variance.")
    
    target_sd <- match.arg(target_sd)
    tgt <- switch(target_sd,
                  mean   = mean(gsds),
                  median = median(gsds),
                  value  = {
                    if (is.null(target_sd_value) || !is.finite(target_sd_value) || target_sd_value <= 0)
                      stop("Provide a positive numeric target_sd_value when target_sd='value'.")
                    target_sd_value
                  })
    
    # 2) rescale arms to target SD row-wise
    asds <- apply(B_arms, 1, sd, na.rm = TRUE)
    ok   <- is.finite(asds) & asds > 0
    scale_fac <- rep(NA_real_, length(asds)); names(scale_fac) <- names(asds)
    scale_fac[ok] <- tgt / asds[ok]
    B_arms_rescaled <- B_arms
    B_arms_rescaled[ok, ] <- B_arms[ok, , drop = FALSE] * scale_fac[ok]
    
    n_rescaled <- sum(ok)
    n_skipped  <- sum(!ok)
    out <- .rbind_no_dupes(B_genes, B_arms_rescaled)
    audit$notes <- c(
      audit$notes,
      sprintf("Variance-matched arms to target SD = %.3f (%s). Rescaled %d arm rows; %d skipped (zero/NA sd).",
              tgt, target_sd, n_rescaled, n_skipped)
    )
    
  } else if (strategy == "weight") {
    # Multiply all arm rows by alpha, then bind
    if (!is.finite(alpha)) stop("alpha must be finite.")
    out <- .rbind_no_dupes(B_genes, B_arms * alpha)
    audit$notes <- c(audit$notes, sprintf("Applied scalar weight alpha = %.3f to all arm rows.", alpha))
  }
  
  list(B_aug = out, audit = audit)
}

# 1) Harmonize (arms already CPM/log/SCT like genes)
res1 <- augment(B_pb_Wu$CPM, B_chr_Wu$LR$CPM, strategy = "harmonize")
cor1 <- cor(res1$B_aug)

# 2) Equalize influence for correlation (row z-score everything)
res2 <- augment(B_pb_Wu$CPM, B_chr_Wu$LR$CPM, strategy = "rowstandardize")
cor2 <- cor(res2$B_aug)

# 3) Variance matching (arms' row-SD -> mean gene row-SD)
res3 <- augment(B_pb_Wu$SCT, B_chr_Wu$SCT, strategy = "variance_match", target_sd = "mean")
cor3 <- cor(res3$B_aug)

# 4) Weighted arms (e.g., 5x)
res4 <- augment(B_pb_Wu$CPM, B_chr_Wu$CPM, strategy = "weight", alpha = 1)
cor4 <- cor(res4$B_aug)













pheatmap::pheatmap(B_chr_Mo_ratio$CPM,
         cluster_rows = FALSE,
         cluster_cols = FALSE,
         scale = "row",
         main = "Chromosomal Arm CPM Heatmap")

pheatmap::pheatmap(B_chr_Mo_ratio$LogNorm,
                   cluster_rows = FALSE,
                   cluster_cols = FALSE,
                   scale = "row",
                   main = "Chromosomal Arm LogNorm Heatmap")

pheatmap::pheatmap(B_chr_Mo_ratio$SCT,
                   cluster_rows = FALSE,
                   cluster_cols = FALSE,
                   scale = "row",
                   main = "Chromosomal Arm SCT Heatmap")


# + Construct B matrix ----------------------------------------------------

align_cols <- function(A, B) {
  common <- intersect(colnames(A), colnames(B))
  if (length(common) == 0) stop("No overlapping columns between matrices.")
  list(A = A[, common, drop = FALSE], B = B[, common, drop = FALSE])
}

rbind_no_dupes <- function(A, B) {
  extra <- setdiff(rownames(B), rownames(A))
  rbind(A, B[extra, , drop = FALSE])
}

make_chrom_info_df <- function(informative_list) {
  # informative_list: named list of character vectors (genes), names = "chrX_p/q"
  df <- utils::stack(informative_list)
  colnames(df) <- c("gene", "chr_arm")
  df
}

.safe_augment <- function(B_genes, B_arms, augment_fun = NULL, ...) {
  # Align columns
  ab <- align_cols(B_genes, B_arms)
  Bg <- ab$A; Ba <- ab$B
  if (is.null(augment_fun)) {
    rbind_no_dupes(Bg, Ba)
  } else {
    res <- do.call(augment_fun, c(list(B_genes = Bg, B_arms = Ba), list(...)))
    if (is.list(res) && !is.null(res$B_aug)) res$B_aug else res
  }
}

.default_keymap <- c(CPM = "CPM", Log = "LogNorm", SCT = "SCT")

assemble_B_list <- function(
    B_pb,                 # list with CPM, LogNorm, SCT (gene-level)
    B_chr,                # list with CPM, LogNorm, SCT (chromosomal features)
    informative_list,     # named list of genes per arm
    keymap = .default_keymap,
    augment_fun = NULL,   # e.g., augment
    ...                   # augment_fun params (strategy, alpha, etc.)
) {
  chrom_df <- make_chrom_info_df(informative_list)
  
  out <- list()
  for (out_key in names(keymap)) {
    in_key <- keymap[[out_key]]
    if (is.null(B_pb[[in_key]]) || is.null(B_chr[[in_key]])) {
      stop(sprintf("Missing block: B_pb[['%s']] or B_chr[['%s']]", in_key, in_key))
    }
    B_aug <- .safe_augment(B_pb[[in_key]], B_chr[[in_key]], augment_fun = augment_fun, ...)
    out[[out_key]] <- list(B = B_aug, ChromInformativeGenes = chrom_df)
  }
  out
}

ChromInformativeGenes_Wu <- B_chr_Wu$InformativeGenes
ChromInformativeGenes_Mo <- B_chr_Mo$InformativeGenes

B_Wu_list <- assemble_B_list(
  B_pb  = B_pb_Wu,
  B_chr = B_chr_Wu,
  informative_list = ChromInformativeGenes_Wu
  # augment_fun = augment, strategy = "harmonize"   # <- optional
)

B_Mo_list <- assemble_B_list(
  B_pb  = B_pb_Mo,
  B_chr = B_chr_Mo,
  informative_list = ChromInformativeGenes_Mo
  # augment_fun = augment, strategy = "harmonize"
)

ChromInformativeGenes_Wu_ratio <- B_chr_Wu$InformativeGenes  # same mapping of genes->arms
ChromInformativeGenes_Mo_ratio <- B_chr_Mo$InformativeGenes

B_Wu_ratio_list <- assemble_B_list(
  B_pb  = B_pb_Wu,
  B_chr = B_chr_Wu$LR,                    # <- ratio blocks
  informative_list = ChromInformativeGenes_Wu_ratio
  # augment_fun = augment, strategy = "harmonize"
)

B_Mo_ratio_list <- assemble_B_list(
  B_pb  = B_pb_Mo,
  B_chr = B_chr_Mo$LR,
  informative_list = ChromInformativeGenes_Mo_ratio
  # augment_fun = augment, strategy = "harmonize"
)


saveRDS(B_Wu_list, file = file.path(sc_dir, "CARD_reference", "B_Wu_list.rds"))
saveRDS(B_Mo_list, file = file.path(sc_dir, "CARD_reference", "B_Mo_list.rds"))

saveRDS(B_Wu_ratio_list, file = file.path(sc_dir, "CARD_reference", "B_Wu_ratio_list.rds"))
saveRDS(B_Mo_ratio_list, file = file.path(sc_dir, "CARD_reference", "B_Mo_ratio_list.rds"))


# Investigate cancer epithelial -------------------------------------------

seurat_integrated <- readRDS(file.path(sc_dir, "seurat_merged_Mo_Wu_pure_clustered.rds"))

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

# Generate correlation heatmaps -------------------------------------------

plot_corr_triplet <- function(B_Wu_list, B_Mo_list, tag, out_dir_pdf) {
  types <- names(B_Wu_list)
  blue_red_gradient <- colorRampPalette(c("blue","white","red"))(100)
  
  for (type1 in types) {
    B_Wu <- B_Wu_list[[type1]]$B
    B_Mo <- B_Mo_list[[type1]]$B
    
    # With arms (as given)
    {
      ct_wu <- colnames(B_Wu); ct_mo <- colnames(B_Mo)
      common_genes <- intersect(rownames(B_Wu), rownames(B_Mo))
      cor_Wu <- cor(B_Wu[rownames(B_Wu), ])
      cor_Mo <- cor(B_Mo[rownames(B_Mo), ])
      num_Wu <- round(cor_Wu, 2); num_Mo <- round(cor_Mo, 2)
      pdf(file.path(out_dir_pdf, paste0("corr_within_B_Wu_",type1,"_",tag,"_WITH_arms.pdf")))
      pheatmap::pheatmap(cor_Wu, name="Correlation", cluster_rows=FALSE, cluster_cols=FALSE,
                         color=blue_red_gradient, na_col="gray90", display_numbers=num_Wu,
                         number_color="black", fontsize_number=10,
                         main=paste0("Within-B Correlation: Wu (",type1,") WITH arms"),
                         breaks=seq(-1,1,length.out=101))
      dev.off()
      pdf(file.path(out_dir_pdf, paste0("corr_within_B_Mo_",type1,"_",tag,"_WITH_arms.pdf")))
      pheatmap::pheatmap(cor_Mo, name="Correlation", cluster_rows=FALSE, cluster_cols=FALSE,
                         color=blue_red_gradient, na_col="gray90", display_numbers=num_Mo,
                         number_color="black", fontsize_number=10,
                         main=paste0("Within-B Correlation: Mo (",type1,") WITH arms"),
                         breaks=seq(-1,1,length.out=101))
      dev.off()
      
      cor_full <- cor(B_Wu[common_genes, ct_wu], B_Mo[common_genes, ct_mo])
      number_labels <- round(cor_full, 2)
      pdf(file.path(out_dir_pdf, paste0("corr_cross_B_Wu_Mo_",type1,"_",tag,"_WITH_arms.pdf")))
      pheatmap::pheatmap(cor_full, name="Correlation", cluster_rows=FALSE, cluster_cols=FALSE,
                         color=blue_red_gradient, na_col="gray90", display_numbers=number_labels,
                         number_color="black", fontsize_number=10,
                         main=paste0("Cross-B: Wu vs Mo (",type1,") WITH arms"),
                         breaks=seq(-1,1,length.out=101))
      dev.off()
    }
    
    # Without arms (filter chr rows)
    {
      B_Wu_nochr <- B_Wu[!str_detect(rownames(B_Wu), "^chr[0-9]+_"), ]
      B_Mo_nochr <- B_Mo[!str_detect(rownames(B_Mo), "^chr[0-9]+_"), ]
      ct_wu <- colnames(B_Wu_nochr); ct_mo <- colnames(B_Mo_nochr)
      common_genes <- intersect(rownames(B_Wu_nochr), rownames(B_Mo_nochr))
      cor_Wu <- cor(B_Wu_nochr[rownames(B_Wu_nochr), ])
      cor_Mo <- cor(B_Mo_nochr[rownames(B_Mo_nochr), ])
      num_Wu <- round(cor_Wu, 2); num_Mo <- round(cor_Mo, 2)
      pdf(file.path(out_dir_pdf, paste0("corr_within_B_Wu_",type1,"_",tag,"_NO_arms.pdf")))
      pheatmap::pheatmap(cor_Wu, name="Correlation", cluster_rows=FALSE, cluster_cols=FALSE,
                         color=blue_red_gradient, na_col="gray90", display_numbers=num_Wu,
                         number_color="black", fontsize_number=10,
                         main=paste0("Within-B Correlation: Wu (",type1,") NO arms"),
                         breaks=seq(-1,1,length.out=101))
      dev.off()
      pdf(file.path(out_dir_pdf, paste0("corr_within_B_Mo_",type1,"_",tag,"_NO_arms.pdf")))
      pheatmap::pheatmap(cor_Mo, name="Correlation", cluster_rows=FALSE, cluster_cols=FALSE,
                         color=blue_red_gradient, na_col="gray90", display_numbers=num_Mo,
                         number_color="black", fontsize_number=10,
                         main=paste0("Within-B Correlation: Mo (",type1,") NO arms"),
                         breaks=seq(-1,1,length.out=101))
      dev.off()
      
      cor_full <- cor(B_Wu_nochr[common_genes, ct_wu], B_Mo_nochr[common_genes, ct_mo])
      number_labels <- round(cor_full, 2)
      pdf(file.path(out_dir_pdf, paste0("corr_cross_B_Wu_Mo_",type1,"_",tag,"_NO_arms.pdf")))
      pheatmap::pheatmap(cor_full, name="Correlation", cluster_rows=FALSE, cluster_cols=FALSE,
                         color=blue_red_gradient, na_col="gray90", display_numbers=number_labels,
                         number_color="black", fontsize_number=10,
                         main=paste0("Cross-B: Wu vs Mo (",type1,") NO arms"),
                         breaks=seq(-1,1,length.out=101))
      dev.off()
    }
  }
}

plot_corr_triplet(B_Wu_list_imp,  B_Mo_list_imp,  tag = "IMPROVED")

B_Wu_list <- readRDS(file = file.path(sc_dir,"CARD_reference", "B_Wu_list.rds"))
B_Mo_list <- readRDS(file = file.path(sc_dir,"CARD_reference", "B_Mo_list.rds"))

types <- names(B_Wu_list)

# Heatmap color palette
blue_red_gradient <- colorRampPalette(c("blue", "white", "red"))(100)

# + With Augmented Chromosomal Pseudogenes --------------------------------

for (type1 in types) {
  B_Wu <- B_Wu_list[[type1]]$B
  B_Mo <- B_Mo_list[[type1]]$B
  
  ct_wu <- colnames(B_Wu)
  ct_mo <- colnames(B_Mo)
  
  genes_wu <- rownames(B_Wu)
  genes_mo <- rownames(B_Mo)
  common_genes <- intersect(genes_wu, genes_mo)
  
  # Wu-only correlation
  cor_matrix_Wu <- cor(B_Wu[genes_wu, ])
  cor_matrix_Wu <- cor_matrix_Wu[sort(rownames(cor_matrix_Wu)), sort(colnames(cor_matrix_Wu))]
  
  # Mo-only correlation
  cor_matrix_Mo <- cor(B_Mo[genes_mo, ])
  cor_matrix_Mo <- cor_matrix_Mo[sort(rownames(cor_matrix_Mo)), sort(colnames(cor_matrix_Mo))]
  
  # Round correlation values for annotation
  num_labels_Wu <- round(cor_matrix_Wu, 2)
  num_labels_Mo <- round(cor_matrix_Mo, 2)
  
  # Plot: Wu within-correlation
  pdf(file.path(sc_dir, "CARD_reference", paste0("corr_within_B_Wu_",type1, "_with_chr.pdf")))
  pheatmap::pheatmap(
    cor_matrix_Wu,
    name = "Correlation",
    cluster_rows = FALSE,
    cluster_cols = FALSE,
    color = blue_red_gradient,
    na_col = "gray90",
    display_numbers = num_labels_Wu,
    number_color = "black",
    fontsize_number = 10,
    main = paste0("Within-B Matrix Correlation: Wu\n", type1, "-Normalized B Matrix\nw/ Chromosomal Pseudogenes"),
    breaks = seq(-1, 1, length.out = 101)
  )
  dev.off()
  
  # Plot: Mo within-correlation
  pdf(file.path(sc_dir, "CARD_reference", paste0("corr_within_B_Mo_",type1, "_with_chr.pdf")))
  pheatmap::pheatmap(
    cor_matrix_Mo,
    name = "Correlation",
    cluster_rows = FALSE,
    cluster_cols = FALSE,
    color = blue_red_gradient,
    na_col = "gray90",
    display_numbers = num_labels_Mo,
    number_color = "black",
    fontsize_number = 10,
    main = paste0("Within-B Matrix Correlation: Mo\n", type1, "-Normalized B Matrix\nw/ Chromosomal Pseudogenes"),
    breaks = seq(-1, 1, length.out = 101)
  )
  dev.off()
  
  
  cor_matrix_full <- cor(B_Wu[common_genes, ct_wu], B_Mo[common_genes, ct_mo])
  cor_matrix_full <- cor_matrix_full[sort(rownames(cor_matrix_full)), sort(colnames(cor_matrix_full))]
  number_labels <- round(cor_matrix_full, 2)
  
  pdf(file.path(sc_dir, "CARD_reference", paste0("corr_cross_B_Wu_Mo_", type1, "_with_chr.pdf")))
  pheatmap::pheatmap(
    cor_matrix_full,
    name = "Correlation",
    cluster_rows = FALSE,
    cluster_cols = FALSE,
    color = blue_red_gradient,
    na_col = "gray90",
    display_numbers = number_labels,
    number_color = "black",
    fontsize_number = 10,
    main = paste0("Cross-B Matrix Correlation: B_Wu (rows) vs B_Mo (cols)\n", type1, "-Normalized B Matrices\nw/ Chromosomal Pseudogenes"),
    breaks = seq(-1, 1, length.out = 101)
  )
  dev.off()
}

# + Without Augmented Chromosomal Pseudogenes -----------------------------

for (type1 in types) {
  B_Wu <- B_Wu_list[[type1]]$B
  B_Mo <- B_Mo_list[[type1]]$B
  
  B_Wu <- B_Wu[!str_detect(rownames(B_Wu), "^chr[0-9]+_"), ]
  B_Mo <- B_Mo[!str_detect(rownames(B_Mo), "^chr[0-9]+_"), ]
  
  ct_wu <- colnames(B_Wu)
  ct_mo <- colnames(B_Mo)
  
  genes_wu <- rownames(B_Wu)
  genes_mo <- rownames(B_Mo)
  common_genes <- intersect(genes_wu, genes_mo)
  
  # Wu-only correlation
  cor_matrix_Wu <- cor(B_Wu[genes_wu, ])
  cor_matrix_Wu <- cor_matrix_Wu[sort(rownames(cor_matrix_Wu)), sort(colnames(cor_matrix_Wu))]
  
  # Mo-only correlation
  cor_matrix_Mo <- cor(B_Mo[genes_mo, ])
  cor_matrix_Mo <- cor_matrix_Mo[sort(rownames(cor_matrix_Mo)), sort(colnames(cor_matrix_Mo))]
  
  # Round correlation values for annotation
  num_labels_Wu <- round(cor_matrix_Wu, 2)
  num_labels_Mo <- round(cor_matrix_Mo, 2)
  
  # Plot: Wu within-correlation
  pdf(file.path(sc_dir, "CARD_reference", paste0("corr_within_B_Wu_",type1, "_without_chr.pdf")))
  pheatmap::pheatmap(
    cor_matrix_Wu,
    name = "Correlation",
    cluster_rows = FALSE,
    cluster_cols = FALSE,
    color = blue_red_gradient,
    na_col = "gray90",
    display_numbers = num_labels_Wu,
    number_color = "black",
    fontsize_number = 10,
    main = paste0("Within-B Matrix Correlation: Wu\n", type1, "-Normalized B Matrix\nw/o Chromosomal Pseudogenes"),
    breaks = seq(-1, 1, length.out = 101)
  )
  dev.off()
  
  # Plot: Mo within-correlation
  pdf(file.path(sc_dir, "CARD_reference", paste0("corr_within_B_Mo_",type1, "_without_chr.pdf")))
  pheatmap::pheatmap(
    cor_matrix_Mo,
    name = "Correlation",
    cluster_rows = FALSE,
    cluster_cols = FALSE,
    color = blue_red_gradient,
    na_col = "gray90",
    display_numbers = num_labels_Mo,
    number_color = "black",
    fontsize_number = 10,
    main = paste0("Within-B Matrix Correlation: Mo\n", type1, "-Normalized B Matrix\nw/o Chromosomal Pseudogenes"),
    breaks = seq(-1, 1, length.out = 101)
  )
  dev.off()
  
  
  cor_matrix_full <- cor(B_Wu[common_genes, ct_wu], B_Mo[common_genes, ct_mo])
  cor_matrix_full <- cor_matrix_full[sort(rownames(cor_matrix_full)), sort(colnames(cor_matrix_full))]
  number_labels <- round(cor_matrix_full, 2)
  
  pdf(file.path(sc_dir, "CARD_reference", paste0("corr_cross_B_Wu_Mo_", type1, "_without_chr.pdf")))
  pheatmap::pheatmap(
    cor_matrix_full,
    name = "Correlation",
    cluster_rows = FALSE,
    cluster_cols = FALSE,
    color = blue_red_gradient,
    na_col = "gray90",
    display_numbers = number_labels,
    number_color = "black",
    fontsize_number = 10,
    main = paste0("Cross-B Matrix Correlation: B_Wu (rows) vs B_Mo (cols)\n", type1, "-Normalized B Matrices\nw/o Chromosomal Pseudogenes"),
    breaks = seq(-1, 1, length.out = 101)
  )
  dev.off()
}


# + With Augmented Chromosomal Ratio Pseudogenes ----------------------------

for (type1 in types) {
  B_Wu <- B_Wu_ratio_list[[type1]]$B
  B_Mo <- B_Mo_ratio_list[[type1]]$B
  
  ct_wu <- colnames(B_Wu)
  ct_mo <- colnames(B_Mo)
  
  genes_wu <- rownames(B_Wu)
  genes_mo <- rownames(B_Mo)
  common_genes <- intersect(genes_wu, genes_mo)
  
  # Wu-only correlation
  cor_matrix_Wu <- cor(B_Wu[genes_wu, ])
  cor_matrix_Wu <- cor_matrix_Wu[sort(rownames(cor_matrix_Wu)), sort(colnames(cor_matrix_Wu))]
  
  # Mo-only correlation
  cor_matrix_Mo <- cor(B_Mo[genes_mo, ])
  cor_matrix_Mo <- cor_matrix_Mo[sort(rownames(cor_matrix_Mo)), sort(colnames(cor_matrix_Mo))]
  
  # Round correlation values for annotation
  num_labels_Wu <- round(cor_matrix_Wu, 2)
  num_labels_Mo <- round(cor_matrix_Mo, 2)
  
  # Plot: Wu within-correlation
  pdf(file.path(sc_dir, "CARD_reference", paste0("corr_within_B_Wu_",type1, "_with_chr_ratio.pdf")))
  pheatmap::pheatmap(
    cor_matrix_Wu,
    name = "Correlation",
    cluster_rows = FALSE,
    cluster_cols = FALSE,
    color = blue_red_gradient,
    na_col = "gray90",
    display_numbers = num_labels_Wu,
    number_color = "black",
    fontsize_number = 10,
    main = paste0("Within-B Matrix Correlation: Wu\n", type1, "-Normalized B Matrix\nw/ Chromosomal Ratio Pseudogenes"),
    breaks = seq(-1, 1, length.out = 101)
  )
  dev.off()
  
  # Plot: Mo within-correlation
  pdf(file.path(sc_dir, "CARD_reference", paste0("corr_within_B_Mo_",type1, "_with_chr_ratio.pdf")))
  pheatmap::pheatmap(
    cor_matrix_Mo,
    name = "Correlation",
    cluster_rows = FALSE,
    cluster_cols = FALSE,
    color = blue_red_gradient,
    na_col = "gray90",
    display_numbers = num_labels_Mo,
    number_color = "black",
    fontsize_number = 10,
    main = paste0("Within-B Matrix Correlation: Mo\n", type1, "-Normalized B Matrix\nw/ Chromosomal Ratio Pseudogenes"),
    breaks = seq(-1, 1, length.out = 101)
  )
  dev.off()
  
  
  cor_matrix_full <- cor(B_Wu[common_genes, ct_wu], B_Mo[common_genes, ct_mo])
  cor_matrix_full <- cor_matrix_full[sort(rownames(cor_matrix_full)), sort(colnames(cor_matrix_full))]
  number_labels <- round(cor_matrix_full, 2)
  
  pdf(file.path(sc_dir, "CARD_reference", paste0("corr_cross_B_Wu_Mo_", type1, "_with_chr_ratio.pdf")))
  pheatmap::pheatmap(
    cor_matrix_full,
    name = "Correlation",
    cluster_rows = FALSE,
    cluster_cols = FALSE,
    color = blue_red_gradient,
    na_col = "gray90",
    display_numbers = number_labels,
    number_color = "black",
    fontsize_number = 10,
    main = paste0("Cross-B Matrix Correlation: B_Wu (rows) vs B_Mo (cols)\n", type1, "-Normalized B Matrices\nw/ Chromosomal Ratio Pseudogenes"),
    breaks = seq(-1, 1, length.out = 101)
  )
  dev.off()
}








