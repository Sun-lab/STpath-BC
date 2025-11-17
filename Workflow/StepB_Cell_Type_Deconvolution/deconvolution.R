# Load necessary libraries
library(ggplot2)
library(data.table)
library(MuSiC)
library(CARD)
library(stringr)
library(Matrix)
library(readr)
library(tidyverse)
library(biomaRt)
library(gprofiler2)
library(org.Hs.eg.db)
library(EnsDb.Hsapiens.v79)
library(dplyr)
library(grid)

# Define functions --------------------------------------------------------

# Function to detect if the first row is a header
has_header <- function(file) {
  first_row <- read.csv(file, nrows = 1, header = FALSE)  # Read the first row without assuming a header
  all(sapply(first_row, is.character))  # Check if all elements are non-numeric
}

# -----------------------------------------------------------------------------#
# Convert gene IDs to gene symbols using multiple sources.
# Args:
#     gene_ids (vector): A vector of gene IDs to convert.
# Returns:
#     all_matches (data.frame): A data frame with the original gene IDs and the corresponding gene symbols from different sources.
# -----------------------------------------------------------------------------#
get_gene_symbols <- function(gene_ids) {
  # biomaRt
  gene_symbol_1 <- tryCatch({
    ensembl <- useEnsembl(biomart = "genes")
    ensembl <- useDataset(dataset = "hsapiens_gene_ensembl", mart = ensembl)
    res <- getBM(mart = ensembl, 
                 attributes = c("ensembl_gene_id", "hgnc_symbol"), 
                 filters = "ensembl_gene_id", 
                 values = gene_ids)
    colnames(res) <- c("id", "symbol.biomaRt")
    res$symbol.biomaRt <- ifelse(res$symbol.biomaRt == "", NA, res$symbol.biomaRt)
    res
  }, error = function(e) {
    message("biomaRt failed: ", e$message)
    NULL
  })
  
  # gprofiler2
  gene_symbol_2 <- gconvert(gene_ids,
                            organism = "hsapiens",
                            target = "ENTREZGENE",
                            filter_na = FALSE)[, c("input", "target")]
  colnames(gene_symbol_2) <- c("id", "symbol.gprofiler2")
  
  # org.Hs.eg.db
  symbols <- mapIds(org.Hs.eg.db, keys = gene_ids, keytype = "ENSEMBL", column = "SYMBOL")
  gene_symbol_3 <- data.frame(id = names(symbols), symbol.org.Hs.eg.db = symbols, stringsAsFactors = FALSE)
  
  # EnsDb.Hsapiens.v79
  gene_symbol_4 <- ensembldb::select(EnsDb.Hsapiens.v79, keys = gene_ids, keytype = "GENEID", columns = c("GENEID", "SYMBOL"))
  colnames(gene_symbol_4) <- c("id", "symbol.EnsDb.Hsapiens.v79")
  
  # Combine all results
  all_matches <- data.frame(id = gene_ids)
  
  if (!is.null(gene_symbol_1)) {
    all_matches <- left_join(all_matches, gene_symbol_1, by = "id")
  }
  all_matches <- all_matches %>%
    left_join(gene_symbol_2, by = "id") %>%
    left_join(gene_symbol_3, by = "id") %>%
    left_join(gene_symbol_4, by = "id") %>%
    pivot_longer(cols = -id, values_to = "symbol")
  
  all_matches <- unique(all_matches[, c("id", "symbol")])
  
  return(all_matches)
}

# get_gene_symbols <- function(gene_ids) {
#   library(biomaRt)
#   library(gprofiler2)
#   library(org.Hs.eg.db)
#   library(ensembldb)
#   library(EnsDb.Hsapiens.v79)
#   library(dplyr)
#   library(tidyr)
#   
#   # biomaRt
#   ensembl <- useEnsembl(biomart = "genes")
#   ensembl <- useDataset(dataset = "hsapiens_gene_ensembl", mart = ensembl)
#   gene_symbol_1 <- getBM(mart = ensembl, attributes = c("ensembl_gene_id", "hgnc_symbol"), 
#                          filters = "ensembl_gene_id", values = gene_ids)
#   colnames(gene_symbol_1) <- c("id", "symbol.biomaRt")
#   gene_symbol_1$symbol.biomaRt <- ifelse(gene_symbol_1$symbol.biomaRt == "", NA, gene_symbol_1$symbol.biomaRt)
#   
#   # gprofiler2
#   gene_symbol_2 <- gconvert(gene_ids,
#                             organism="hsapiens",
#                             target="ENTREZGENE",
#                             filter_na = FALSE)[, c("input", "target")]
#   colnames(gene_symbol_2) <- c("id", "symbol.gprofiler2")
#   
#   # org.Hs.eg.db
#   symbols <- mapIds(org.Hs.eg.db, keys = gene_ids, keytype = "ENSEMBL", column = "SYMBOL")
#   gene_symbol_3 <- data.frame(id = names(symbols), symbol.org.Hs.eg.db = symbols, stringsAsFactors = FALSE)
#   
#   # EnsDb.Hsapiens.v79
#   gene_symbol_4 <- ensembldb::select(EnsDb.Hsapiens.v79, keys= gene_ids, keytype = "GENEID", columns = c("GENEID","SYMBOL"))
#   colnames(gene_symbol_4) <- c("id", "symbol.EnsDb.Hsapiens.v79")
#   
#   # Initialize data frame to store results
#   all_matches <- data.frame(id = gene_ids)
#   
#   # Combine results from all sources
#   all_matches <- all_matches %>%
#     left_join(gene_symbol_1, by = "id") %>%
#     left_join(gene_symbol_2, by = "id") %>%
#     left_join(gene_symbol_3, by = "id") %>%
#     left_join(gene_symbol_4, by = "id") %>%
#     pivot_longer(2:5, values_to = "symbol")
#   
#   all_matches <- unique(all_matches[, c(1,3)])
#   
#   return(all_matches)
# }

get_ST_chrom_pseudogene_count <- function(
    st_count,
    informative_gene_set
) {
  # 1. Filter valid genes
  genes_present <- intersect(informative_gene_set$gene, rownames(st_count))
  if (length(genes_present) == 0) {
    stop("Error: None of the informative genes were found in the ST count data.")
  }
  
  gene_set_present <- informative_gene_set[informative_gene_set$gene %in% genes_present, ]
  chr_arms <- unique(gene_set_present$chr_arm)
  
  # 2. For each chrom arm, sum counts across spots
  pseudogene_list <- lapply(chr_arms, function(arm) {
    arm_genes <- gene_set_present$gene[gene_set_present$chr_arm == arm]
    arm_genes <- intersect(arm_genes, rownames(st_count))
    if (length(arm_genes) == 0) return(NULL)
    
    arm_sum <- Matrix::colSums(st_count[arm_genes, , drop = FALSE])
    return(arm_sum)
  })
  
  names(pseudogene_list) <- chr_arms
  pseudo_counts <- do.call(rbind, pseudogene_list)  # rows = arms, cols = spots
  
  # Relative expression: divide by read depth
  spot_total <- Matrix::colSums(st_count)
  spot_total[spot_total == 0] <- 1  # prevent div by zero
  
  pseudo_frac <- sweep(pseudo_counts, 2, spot_total, FUN = "/")
  
  return(pseudo_df)
  
}


CARD_deconvolution_custom <- function(CARD_object = NULL, B_custom = NULL, ct.select = NULL, 
                                      spatial_count = NULL, spatial_location = NULL, seed=1234) {
  
  if (is.null(B_custom) && (!is.null(CARD_object))) {
    ct.select = CARD_object@info_parameters$ct.select
    ct.varname = CARD_object@info_parameters$ct.varname
    sample.varname = CARD_object@info_parameters$sample.varname
    cat(paste0("## create reference matrix from scRNASeq...\n"))
    sc_eset = CARD_object@sc_eset
    Basis_ref = CARD::createscRef(sc_eset, ct.select, ct.varname, sample.varname)
    Basis = Basis_ref$basis
    spatial_count = CARD_object@spatial_countMat
    spatial_location = CARD_object@spatial_location
  } else if (!is.null(B_custom)) {
    Basis = B_custom
  }
  
  Basis = Basis[,colnames(Basis) %in% ct.select]
  Basis = Basis[,match(ct.select,colnames(Basis))]
  commonGene = intersect(rownames(spatial_count),rownames(Basis))
  #### remove mitochondrial and ribosomal genes
  common  = commonGene[!(commonGene %in% commonGene[grep("mt-",commonGene)])]
  # cat(paste0("## Select Informative Genes! ...\n"))
  # common = selectInfo(Basis,sc_eset,commonGene,ct.select,ct.varname)
  Xinput = spatial_count
  rm(spatial_count)
  B = Basis
  rm(Basis)
  
  ##### match the common gene names
  Xinput = Xinput[order(rownames(Xinput)),]
  B = B[order(rownames(B)),]
  B = B[rownames(B) %in% common,]
  Xinput = Xinput[rownames(Xinput) %in% common,]
  ##### filter out non expressed genes or cells again
  Xinput = Xinput[rowSums(Xinput) > 0,]
  Xinput = Xinput[,colSums(Xinput) > 0]
  ##### normalize count data
  colsumvec = colSums(Xinput)
  Xinput_norm = sweep(Xinput,2,colsumvec,"/")
  B = B[rownames(B) %in% rownames(Xinput_norm),]    
  B = B[match(rownames(Xinput_norm),rownames(B)),]
  #### spatial location
  spatial_location = spatial_location[rownames(spatial_location) %in% colnames(Xinput_norm),]
  spatial_location = spatial_location[match(colnames(Xinput_norm),rownames(spatial_location)),]
  
  ##### normalize the coordinates without changing the shape and relative position
  norm_cords = spatial_location[ ,c("x","y")]
  norm_cords$x = norm_cords$x - min(norm_cords$x)
  norm_cords$y = norm_cords$y - min(norm_cords$y)
  scaleFactor = max(norm_cords$x,norm_cords$y)
  norm_cords$x = norm_cords$x / scaleFactor
  norm_cords$y = norm_cords$y / scaleFactor
  ##### initialize the proportion matrix
  ED <- fields::rdist(as.matrix(norm_cords))##Euclidean distance matrix
  cat(paste0("## Deconvolution Starts! ...\n"))
  set.seed(seed)
  Vint1 = as.matrix(MCMCpack::rdirichlet(ncol(Xinput_norm), rep(10,ncol(B))))
  colnames(Vint1) = colnames(B)
  rownames(Vint1) = colnames(Xinput_norm)
  b = rep(0,length(ct.select))
  ###### parameters that need to be set
  isigma = 0.1 ####construct Gaussian kernel with the default scale /length parameter to be 0.1
  epsilon = 1e-04  #### convergence epsion 
  phi = c(0.01,0.1,0.3,0.5,0.7,0.9,0.99) #### grided values for phi
  kernel_mat <- exp(-ED^2 / (2 * isigma^2))
  diag(kernel_mat) <- 0
  rm(ED)
  rm(Xinput)
  rm(norm_cords)
  gc()
  ###### scale the Xinput_norm and B to speed up the convergence. 
  mean_X = mean(Xinput_norm)
  mean_B = mean(B)
  Xinput_norm = Xinput_norm * 1e-01 / mean_X
  B = B * 1e-01 / mean_B
  gc()
  ResList = list()
  Obj = c()
  for(iphi in 1:length(phi)){
    res = CARD::CARDref(
      XinputIn = as.matrix(Xinput_norm),
      UIn = as.matrix(B),
      WIn = kernel_mat, 
      phiIn = phi[iphi],
      max_iterIn =1000,
      epsilonIn = epsilon,
      initV = Vint1,
      initb = rep(0,ncol(B)),
      initSigma_e2 = 0.1, 
      initLambda = rep(10,length(ct.select)))
    rownames(res$V) = colnames(Xinput_norm)
    colnames(res$V) = colnames(B)
    ResList[[iphi]] = res
    Obj = c(Obj,res$Obj)
  }
  Optimal = which(Obj == max(Obj))
  Optimal = Optimal[length(Optimal)] #### just in case if there are two equal objective function values
  OptimalPhi = phi[Optimal]
  OptimalRes = ResList[[Optimal]]
  cat(paste0("## Deconvolution Finish! ...\n"))
  
  CARD_result <- list()
  CARD_result[["phi"]] <- OptimalPhi
  CARD_result[["Proportion_CARD"]] <- sweep(OptimalRes$V,1,rowSums(OptimalRes$V),"/")
  CARD_result[["algorithm_matrix"]] = list(B = B * mean_B / 1e-01, Xinput_norm = Xinput_norm * mean_X / 1e-01, Res = OptimalRes)
  CARD_result[["spatial_location"]] <- spatial_location
  
  return(CARD_result)
}



# CARD_deconvolution_custom <- function(CARD_object = NULL, B_custom = NULL, ct.select = NULL, 
#                                       spatial_count = NULL, spatial_location = NULL, seed=1234) {
#   
#   if (is.null(B_custom) && (!is.null(CARD_object))) {
#     ct.select = CARD_object@info_parameters$ct.select
#     ct.varname = CARD_object@info_parameters$ct.varname
#     sample.varname = CARD_object@info_parameters$sample.varname
#     cat(paste0("## create reference matrix from scRNASeq...\n"))
#     sc_eset = CARD_object@sc_eset
#     Basis_ref = CARD::createscRef(sc_eset, ct.select, ct.varname, sample.varname)
#     Basis = Basis_ref$basis
#     Basis = Basis[,colnames(Basis) %in% ct.select]
#     Basis = Basis[,match(ct.select,colnames(Basis))]
#     spatial_count = CARD_object@spatial_countMat
#     commonGene = intersect(rownames(spatial_count), rownames(Basis))
#     B = Basis
#     rm(Basis)
#   } else if (!is.null(B_custom)) {
#     B = B_custom
#   }
#   
#   # Subset and order B to match ct.select
#   B = B[, colnames(B) %in% ct.select, drop = FALSE]
#   B = B[, match(ct.select, colnames(B)), drop = FALSE]
#   
#   # Match genes between B and spatial count
#   Xinput = spatial_count
#   common_genes = intersect(rownames(Xinput), rownames(B))
#   common_genes = common_genes[!(common_genes %in% grep("mt-", common_genes, value = TRUE))]
#   
#   B = B[common_genes, , drop = FALSE]
#   Xinput = Xinput[common_genes, , drop = FALSE]
#   
#   # Filter out low-quality rows and columns
#   Xinput = Xinput[rowSums(Xinput) > 0, , drop = FALSE]
#   Xinput = Xinput[, colSums(Xinput) > 0, drop = FALSE]
#   
#   # Normalize spatial count data
#   colsumvec = colSums(Xinput)
#   Xinput_norm = sweep(Xinput, 2, colsumvec, "/")
#   
#   # Ensure matching order of genes
#   B = B[rownames(B) %in% rownames(Xinput_norm), , drop = FALSE]
#   B = B[match(rownames(Xinput_norm), rownames(B)), , drop = FALSE]
#   
#   # Handle spatial coordinates
#   spatial_location = spatial_location[rownames(spatial_location) %in% colnames(Xinput_norm), , drop = FALSE]
#   spatial_location = spatial_location[match(colnames(Xinput_norm), rownames(spatial_location)), , drop = FALSE]
# 
#   ##### normalize the coordinates without changing the shape and relative position
#   norm_cords = spatial_location[, c("x", "y")]
#   norm_cords$x = norm_cords$x - min(norm_cords$x)
#   norm_cords$y = norm_cords$y - min(norm_cords$y)
#   scaleFactor = max(norm_cords$x, norm_cords$y)
#   norm_cords$x = norm_cords$x / scaleFactor
#   norm_cords$y = norm_cords$y / scaleFactor
#   
#   # Distance and kernel
#   ED <- fields::rdist(as.matrix(norm_cords))
#   
#   cat(paste0("## Deconvolution Starts! ...\n"))
#   set.seed(seed)
#   Vint1 = as.matrix(MCMCpack::rdirichlet(ncol(Xinput_norm), rep(10, ncol(B))))
#   colnames(Vint1) = colnames(B)
#   rownames(Vint1) = colnames(Xinput_norm)
#   b = rep(0,length(ct.select))
#   
#   ###### parameters that need to be set
#   isigma = 0.1
#   epsilon = 1e-04
#   phi = c(0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99)
#   kernel_mat <- exp(-ED^2 / (2 * isigma^2))
#   diag(kernel_mat) <- 0
#   gc()
#   
#   # Scale inputs
#   mean_X = mean(Xinput_norm)
#   mean_B = mean(B)
#   Xinput_norm = Xinput_norm * 1e-01 / mean_X
#   B = B * 1e-01 / mean_B
#   gc()
#   
#   ResList = list()
#   Obj = numeric(length(phi))
#   
#   for (iphi in seq_along(phi)) {
#     res = CARD::CARDref(
#       XinputIn = as.matrix(Xinput_norm),
#       UIn = as.matrix(B),
#       WIn = kernel_mat, 
#       phiIn = phi[iphi],
#       max_iterIn = 1000,
#       epsilonIn = epsilon,
#       initV = Vint1,
#       initb = rep(0, ncol(B)),
#       initSigma_e2 = 0.1, 
#       initLambda = rep(10, length(ct.select))
#     )
#     rownames(res$V) = colnames(Xinput_norm)
#     colnames(res$V) = colnames(B)
#     ResList[[iphi]] = res
#     Obj[iphi] = res$Obj
#   }
#   
#   Optimal = which.max(Obj)
#   Optimal = Optimal[length(Optimal)] 
#   OptimalPhi = phi[Optimal]
#   OptimalRes = ResList[[Optimal]]
#   
#   CARD_result <- list()
#   CARD_result[["phi"]] <- OptimalPhi
#   CARD_result[["Proportion_CARD"]] <- sweep(OptimalRes$V, 1, rowSums(OptimalRes$V), "/")
#   CARD_result[["algorithm_matrix"]] = list(
#     B = B * mean_B / 1e-01,
#     Xinput_norm = Xinput_norm * mean_X / 1e-01,
#     Res = OptimalRes
#   )
#   CARD_result[["spatial_location"]] <- spatial_location
# 
#   return(CARD_result)
# }

.card_skip <- function(...) {
  structure(list(message = paste0(...), call = NULL),
            class = c("card_skip", "error", "condition"))
}

# Use this instead of .stopif:
.stopif_next <- function(ok, ...) {
  if (!ok) stop(.card_skip(...))
} 

.stopif <- function(ok, ...) if (!ok) stop(..., call. = FALSE)
.catln  <- function(...) cat(paste0(..., "\n"))

# -----------------------------------------------------------------------------#
# Perform cell type deconvolution on spatial transcriptomics data using reference scRNA-seq data.
# Args:
#     st_meta (data.frame): Metadata for spatial transcriptomics samples. Contains sample_id, subtype, count_matrix_path, features_path, and barcodes_path.
#     spatial_location (data.frame): Spatial location information for ST spots, with row names being barcodes, having one column of row position and one of column position.
#     output_dir (str): Directory to save the deconvolution results (optional).
# Outputs:
#     Saves the following files to the output directory:
#     - CARD_obj_<sampleID>.rds: Deconvoluted object for each sample.
#     - CARD_alg_matrix_<sampleID>.rds: Algorithm matrix for each sample.
#     - Proportion_<sampleID>.csv: Cell type proportions for each spot in the sample.
# -----------------------------------------------------------------------------#
run_deconvolution <- function(sc_count = NULL, sc_meta = NULL, 
                              st_meta, spatial_location,
                              stdata_list = NULL, 
                              B_custom = NULL,
                              use_selectInfo = T,
                              output_dir = NULL, 
                              match.subtype = TRUE, match.slide = FALSE,
                              cell_type_res = "celltype_major", 
                              subtypes = c("ER+", "HER2+", "TNBC"), 
                              save.CARDobj = F,
                              save.amat = F,
                              overwrite = FALSE,
                              seed = 1234,
                              file_suffix = "") {

  .stopif(all(c("sid","count_matrix_path","features_path","barcodes_path") %in% colnames(st_meta)),
          "st_meta must contain columns: sid, count_matrix_path, features_path, barcodes_path")
  .stopif(all(c("spot_id","sid","x","y") %in% colnames(spatial_location)),
          "spatial_location must contain columns: spot_id, sid, x, y")
  
  
  for (i in 1:nrow(st_meta)) {
    s1 <- st_meta[i, "sid", drop = TRUE]
    
    outdir <- if ("proportion_dir" %in% colnames(st_meta)) {
      st_meta[i, "proportion_dir", drop = TRUE]
    } else {
      .stopif(!is.null(output_dir), "No output directory is provided.")
      output_dir
    }
    dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
    
    # -------- filenames now include optional suffix (e.g., _region0)
    sfx <- if (is.null(file_suffix) || identical(file_suffix, "")) "" else file_suffix
    prop_fp <- file.path(outdir, sprintf("Proportion_%s%s_%s.csv", s1, sfx, cell_type_res))
    card_fp <- file.path(outdir, sprintf("CARD_obj_%s%s_%s.rds", s1, sfx, cell_type_res))
    amat_fp <- file.path(outdir, sprintf("CARD_alg_matrix_%s%s_%s.rds", s1, sfx, cell_type_res))
    
    .catln("=================================================")
    if (!overwrite && file.exists(prop_fp)) {
      .catln(s1, " has been deconvoluted previously. (use overwrite=TRUE to recompute)")
      next
    } else {
      .catln(s1)
      .catln("Output directory: ", outdir)
    }
    
    # Load and prepare spatial transcriptomics count data
    st_path <- st_meta[i, "count_matrix_path", drop = TRUE]
    
    stdata_s <- NULL
    if (!is.null(stdata_list)) {
      stdata_s <- stdata_list[[s1]]
    } 
    
    if (is.null(stdata_s)) {
      stdata_s <- tryCatch({
        Matrix::readMM(st_path)
      }, error = function(e) {
        tryCatch({
          read.csv(file = st_path, sep = "\t") %>%
            remove_rownames %>% column_to_rownames(var="X")
        }, error = function(e) {
          tryCatch({
            read.csv(file = st_path) %>%
              remove_rownames %>% column_to_rownames(var="X")
          }, error = function(e){
            # If both attempts fail, print an error message
            cat("Both readMM and read.csv failed to read the file:", st_path, "\n")
          })
        })
      })
    }
    
    # Check if the matrix was successfully read
    if (is.null(stdata_s)) {
      cat("Failed to read the matrix from the file:", st_path, "\n")
    } else {
      cat("Count matrix successfully read.\n")
    }
    
    # Check if the rowname and colname exist
    if(is.null(colnames(stdata_s)) || is.null(rownames(stdata_s))){
      barcode <- read.table(file = gzfile(st_meta[i, "barcodes_path"]),
                            sep = '\t', header = FALSE)
      barcode_names <- barcode$V1
      feature <- read.table(file = gzfile(st_meta[i, "features_path"]),
                            sep = '\t', header = FALSE)
      feature <- as.data.frame(lapply(feature, function(x) gsub("GRCh38_", "", x)))
      if (ncol(feature) == 1){
        feature_names <- feature[, 1]
      } else {
        feature_names <- feature[, sapply(feature, function(column) !("Gene Expression" %in% column) && !any(grepl("^ENSG", column)))]
      }
      
      # Assign to the dimension with same length.
      if (ncol(stdata_s) == length(barcode_names) && nrow(stdata_s) == length(feature_names)) {
        colnames(stdata_s) <- barcode_names
        rownames(stdata_s) <- feature_names
      } else if (nrow(stdata_s) == length(barcode_names) && ncol(stdata_s) == length(feature_names)) {
        rownames(stdata_s) <- barcode_names
        colnames(stdata_s) <- feature_names
      } else {
        cat("The lengths of the barcode or feature names do not match the dimensions of the matrix.\n")
      }
    }
    
    # Verify the names are assigned correctly
    if (is.null(colnames(stdata_s)) || is.null(rownames(stdata_s))) {
      stop("Names are not assigned to the expression matrix.")
    } else {
      cat("Names are assigned to the expression matrix.\n")
    }
    
    stdata_s <- as(stdata_s, "sparseMatrix")
    stdata_s <- as(stdata_s, "CsparseMatrix")
    # if rows happen to be spot IDs, transpose
    if(any(rownames(stdata_s) %in% spatial_location[,"spot_id"])){
      stdata_s <- t(stdata_s)
    }
    # drop ambiguous rows if present
    st_count <- stdata_s[!startsWith(rownames(stdata_s), "X__ambiguous"), ]
    
    # Check if gene ID of ST data need to be converted to gene symbol to match with the reference
    if (is.null(B_custom)) {
      sc_gene <- data.frame(symbol.sc = rownames(sc_count))
    } else {
      sc_gene <- data.frame(symbol.sc = rownames(B_custom))
    }
    
    if (any(startsWith(rownames(st_count), "ENSG"))) {
      # Retrieve the gene symbols for ST data
      symbols_st <- get_gene_symbols(rownames(st_count))
      colnames(symbols_st) <- c("id.st", "symbol.st")
      
      # Match symbols between scRNA-seq and ST data
      symbols_matched <- symbols_st[symbols_st$symbol.st %in% sc_gene$symbol.sc, ]
      colnames(symbols_matched) <- c("id.st", "symbol.sc")
      # ST genes matched with more than one reference gene
      id_multi <- symbols_matched$id.st[duplicated(symbols_matched$id.st)]
      # Reference genes matched with more than one gene in ST
      symbol_multi <- symbols_matched$symbol.sc[duplicated(symbols_matched$symbol.sc)]
      # Extract ST genes that match with a common reference gene
      id_common_match <- unlist(symbols_matched[symbols_matched$symbol.sc %in% symbol_multi, "id.st"])
      # Remove ST genes that share the same gene symbol in reference and ST genes that match with more than one reference gene
      genes_bijective <- symbols_matched[!symbols_matched$id.st %in% unique(c(id_multi, id_common_match)), ]
      # Convert ST gene ID to gene symbol
      st_count <- st_count[genes_bijective$id.st, ]
      rownames(st_count) <- genes_bijective$symbol.sc
    }
    
    # Prepare spatial location
    spatial_location_i <- spatial_location %>%
      dplyr::filter(spot_id %in% colnames(st_count) & sid == s1 & in_tissue == 1) %>%
      column_to_rownames("spot_id") %>%
      dplyr::select(x,y)

    if(nrow(spatial_location_i) > ncol(st_count)){
      spatial_location_i <- spatial_location_i[colnames(st_count), ]
    } else {
      st_count <- st_count[, rownames(spatial_location_i)]
    }
    
    table(rownames(spatial_location_i) == colnames(st_count))
    
    # ========================================================================== 
    # Route A: build B from scRNA-seq (no B_custom provided)
    # ========================================================================== 
    if (is.null(B_custom)) {
      .stopif(!is.null(sc_count) && !is.null(sc_meta),
              "sc_count and sc_meta must be provided when B_custom is NULL")
      
      # Select scRNA-seq cells of the same subtype and/or same slide
      if (match.slide) { # If use the matched scRNA-seq; no need to match on subtype
        .stopif("orig.ident" %in% colnames(sc_meta), "sc_meta must have 'orig.ident' when match.slide=TRUE")
        if (!s1 %in% sc_meta$orig.ident) { 
          warning(cat("Deconvolution failed for", s1, ": No matched scRNA-seq reference found."))
          next 
        }
        cells2use <- which(sc_meta$orig.ident == s1)
        subtype_used <- NA_character_
      } else { # If not match on slide
        subtype_used <- if ("subtype" %in% colnames(st_meta)) st_meta[i, "subtype"] else NA
        
        if (match.subtype && !is.na(subtype_used) && "subtype" %in% colnames(sc_meta) && subtype_used %in% subtypes) {
          cells2use <- which(sc_meta$subtype == subtype_used)
          .catln("Subtype: ", subtype_used)
        } else {
          cells2use <- seq_len(nrow(sc_meta))
          .catln("Subtype not specified.")
        }
        
      }
      
      # Subset scRNA-seq reference
      sc_count_i <- sc_count[, cells2use, drop = FALSE]
      sc_meta_i  <- sc_meta[cells2use, , drop = FALSE]
      
      # Create CARD object
      .catln("Create CARD object.")
      .catln("  sc dims: ", nrow(sc_count_i), " x ", ncol(sc_count_i))
      .catln("  st dims: ", nrow(st_count),   " x ", ncol(st_count))
      
      common_genes <- intersect(rownames(sc_count_i), rownames(st_count))
      
      if (length(common_genes) > 0){
        cat(length(common_genes), "genes shared by the scRNA-seq reference data and the ST data. \n")
      } else {
        warning("No common genes between the scRNA-seq reference and the ST data.\n")
        next
      }
      
      sc_count_i <- sc_count_i[common_genes, , drop = FALSE]
      st_count <- st_count[common_genes, , drop = FALSE]
      
      sc_expressed <- sum(Matrix::rowSums(sc_count_i) != 0)
      st_expressed <- sum(Matrix::rowSums(st_count)   != 0)
      
      if(sc_expressed == 0 ) {
        warning("Deconvolution failed: All the common genes have zero counts in all cells of the scRNA-seq reference. \n")
        next
      } else {
        cat(sc_expressed, "of the common genes are expressed in at least one cell of the scRNA-seq reference. \n")
      }
      if(st_expressed == 0 ) {
        warning("Deconvolution failed: all the common genes have zero counts in all spots of the ST data. \n")
        next
      } else {
        cat(st_expressed, "of the common genes are expressed in at least one spot of the ST data. \n")
      }
      
      CARD_obj <- CARD::createCARDObject(
        sc_count = sc_count_i,
        sc_meta = sc_meta_i,
        spatial_count = st_count,
        spatial_location = spatial_location_i,
        ct.varname = cell_type_res,
        ct.select = unique(sc_meta_i[[cell_type_res]]),
        sample.varname = "orig.ident",
        minCountGene = 100,
        minCountSpot = 5
      )
      gc()
      
      # Run deconvolution
      .catln("start CARD deconvolution.")
      if (!use_selectInfo) {
        res_list <- CARD_deconvolution_custom(
          CARD_object = CARD_obj,
          B_custom = NULL,
          seed = seed
        )
        .catln("finish CARD deconvolution.")
        
        if (save.CARDobj) saveRDS(res_list, file = card_fp)
        if (save.amat)    saveRDS(res_list[["algorithm_matrix"]], file = amat_fp)
        prop <- as.data.frame(res_list[["Proportion_CARD"]])
        prop$sid <- s1
        utils::write.csv(prop, prop_fp, row.names = TRUE)
        
      } else {
        CARD_obj <- CARD::CARD_deconvolution(CARD_object = CARD_obj)
        .catln("finish CARD deconvolution.")
        if (save.CARDobj) saveRDS(CARD_obj, file = card_fp)
        if (save.amat)    saveRDS(CARD_obj@algorithm_matrix, file = amat_fp)
        prop <- as.data.frame(CARD_obj@Proportion_CARD)
        prop$sid <- s1
        utils::write.csv(prop, prop_fp, row.names = TRUE)
      }
      
      # ========================================================================== 
      # Route B: use provided custom B (genes x cell types)
      # ========================================================================== 
    } else {
      .catln("dim(st_count): ", paste(dim(st_count), collapse = " x "))
      
      common_genes <- intersect(rownames(B_custom), rownames(st_count))
      
      if (length(common_genes) > 0){
        cat(length(common_genes), "genes shared by the custom B matrix and the ST data. \n")
      } else {
        warning("No common genes between the custom B matrix and the ST data.\n")
        next
      }
      
      st_count2 <- st_count[common_genes, , drop = FALSE]
      B_custom2 <- B_custom[common_genes, , drop = FALSE]
      
      st_expressed <- sum(Matrix::rowSums(st_count2) != 0)
      if(st_expressed == 0 ) {
        warning("Deconvolution failed: all the common genes have zero counts in all spots of the ST data. \n")
        next
      } else {
        cat(st_expressed, "of the common genes are expressed in at least one spot of the ST data. \n")
      }
      
      .catln("start CARD (custom B) deconvolution.")
      res_list <- CARD_deconvolution_custom(
        CARD_object = NULL,
        B_custom = B_custom2,
        ct.select = colnames(B_custom2),
        spatial_count = st_count2,
        spatial_location = spatial_location_i,
        seed = seed
      )
      .catln("finish CARD deconvolution.")
      
      
      if (save.CARDobj) saveRDS(res_list, file = card_fp)
      if (save.amat)    saveRDS(res_list[["algorithm_matrix"]], file = amat_fp)
      prop <- as.data.frame(res_list[["Proportion_CARD"]])
      prop$sid <- s1
      utils::write.csv(prop, prop_fp, row.names = TRUE)

    }
  }
}


# -----------------------------------------------------------------------------#
# Visualize the cell type proportions and spatial distribution for each sample.
# Args:
#     st_meta (data.frame): Metadata for spatial transcriptomics samples.
#     output_dir (str): Directory to save the visualizations.
#     ct.visualize (vector): Cell types of interest to visualize.
#     width (numeric): Width of the output plots.
#     pie_radius (numeric): Radius of the pie charts in the plots.
#     point_size (numeric): Size of the points in the proportion plots.
#     x_reverse (bool): Whether to reverse the x-axis.
#     y_reverse (bool): Whether to reverse the y-axis.
#     xy_flip (bool): Whether to flip the x and y axes.
# Outputs:
#     Saves the following plots to the output directory:
#     - cell_type_proportion_<sampleID>.jpg: Visualization of the cell type proportions for all spots.
#     - specific_type_proportion_<sampleID>.jpg: Visualization of the spatial distribution of the cell type proportions.
# -----------------------------------------------------------------------------#
visualize_proportions <- function(st_meta, output_dir, ct.visualize = NULL, 
                                  width = 10, point_size = 2, pie_radius = 1,
                                  x_reverse = F, y_reverse = F, xy_flip = F) {
  colors = c(`B-cells` = "#2a5fbd",
             CAFs = "#7ddffa",
             `Cancer Epithelial` = "#117d30",
             Endothelial = "#71f25a", 
             Myeloid = "#ebc857",
             `Normal Epithelial` ="#D39200",
             Plasmablasts = "#F8766D",
             PVL = "#DB72FB",
             `T-cells` = "#bd2a84")
  cell_type_res = "celltype_major"
  
  for(i in 1:nrow(st_meta)){
    s1 = st_meta[i, "sid"]
    # Load deconvoluted object
    CARD_obj <- readRDS(file.path(output_dir, sprintf("/CARD_obj_%s_%s.rds", s1, cell_type_res)))
    
    h_w_ratio <-  (max(CARD_obj@spatial_location$y)-min(CARD_obj@spatial_location$y)) / (max(CARD_obj@spatial_location$x)-min(CARD_obj@spatial_location$x))
    # Get the proportion of cell types of each cell
    prop <- as.data.frame(CARD_obj@Proportion_CARD)
    
    # Visualization of the proportions of all spots
    p1 <- CARD.visualize.pie(proportion = CARD_obj@Proportion_CARD, 
                             spatial_location = CARD_obj@spatial_location, 
                             colors = colors,
                             radius = pie_radius)
    
    # Visualization of the spatial distribution of the cell type proportion
    p2 <- CARD.visualize.prop(
      proportion = CARD_obj@Proportion_CARD,        
      spatial_location = CARD_obj@spatial_location, 
      ct.visualize = ct.visualize,                 
      colors = c("lightblue","lightyellow","red"), 
      NumCols = 3,
      pointSize = point_size) 
    
    if(x_reverse){
      p1 <- p1 + scale_x_reverse()
      p2 <- p2 + scale_x_reverse()
    }
    if(y_reverse){
      p1 <- p1 + scale_y_reverse()
      p2 <- p2 + scale_y_reverse()
    }
    if(xy_flip){
      p1 <- p1 + coord_flip()
      p2 <- p2 + coord_flip()
      h_w_ratio <- 1/h_w_ratio
    }
    
    # Save the figures
    
    jpeg(file.path(output_dir, paste0("cell_type_proportion_", s1, ".jpg")), 
         width = width, height = width*h_w_ratio, units = 'in', res = 300)
    print(p1)
    dev.off()
    
    jpeg(file.path(output_dir, paste0("specific_type_proportion_", s1, ".jpg")), 
         width = width, height = width*h_w_ratio, units = 'in', res = 300)
    print(p2)
    dev.off()
    
    gc()
  }
}


