# Load necessary libraries
library(ggplot2)
library(ggforce)
library(ggnewscale)
library(ggpubr)
library(patchwork)
library(data.table)
library(Matrix)
library(readr)
library(stringr)
library(tidyverse)
library(dplyr)
library(tidyr)
library(grid)
library(purrr)
library(tibble)

options(future.globals.maxSize = 20 * 1024^3) 

source("deconvolution.R")

# Set directories ---------------------------------------------------------

st_dir <- "/fh/working/sun_w/zsui/st2image_data"
sc_dir <- "/fh/working/sun_w/zsui/st2image_data/brca_scRNAseq_datasets/CARD_reference"

# Cell Types and Markers --------------------------------------------------

celltypes <- c(
  "T.cell",
  "Endothelial",
  "CAFs/PVL",
  "Cancer.Epithelial",
  "Other.Epithelial",
  "Myeloid",
  "B.cell",
  "Plasma"
  # "Plasmablasts",
  # "PlasmaMerged"
)


subtype_colors <- c(
  "ER+" = "orange",
  "HER2+" = "#377EB8",
  "TNBC" = "#4DAF4A",
  "Unknown" = "gray"
)

colors = c(T.cell = "#2a5fbd",
           B.cell = "#7ddffa",
           Myeloid = "#117d30",
           PlasmaMerged = "#71f25a",
           `CAFs/PVL` = "#ebc857",
           Endothelial ="#D39200",
           Other.Epithelial = "#F8766D",
           Cancer.Epithelial = "#DB72FB")

markers_list <- list(
  `B.cell` = c(
    # Primary Markers: 
    "CD19","MS4A1", "CD79A", "CD79B","BANK1",
    # Secondary Markers: 
    "CD69", "IGHG1", "JCHAIN"
  ),
  `T.cell` = c(
    # Primary Markers: 
    "CD3D", "CD3E", "FOXP3", "CD8A", "CD8B", "NKG7", "CD69", 
    # Secondary Markers: 
    "FCGR3A", "SELL", "CD4", "CCR7", "IL7R", "TCF7", "LEF1"
  ),
  PlasmaMerged = c(
    # Primary Markers: 
    "IGHG1", "JCHAIN", "MZB1", 
    # Secondary Markers: 
    "CD79A", "TNFRSF17", "IGKC", "IGLC3", "IGLC2", "IGHA1",
    "IGHG3", "IGHG4", "IGHA2", "FCRL5"
  ),
  Myeloid = c(
    # Primary Markers:
    "CD86", "FCGR3A", "CD68", "CD14", "CSF1R", "IL1B", 
    "MRC1", "CD163", "MSR1", "IFI30", "SPP1", "APOE", "LYZ", "APOC1",
    "CD1C", "CD74"
  ),
  `CAFs/PVL` = c(
    "COL1A1", "COL3A1", "COL1A2", "POSTN", "PDGFRB", "RGS5", "GJA4", "SFRP2", "DCN", "DES"
  ),
  Endothelial = c(
    # Primary Markers: 
    "VWF", "PECAM1", "KDR", "FLT1", "SOX17", "ACKR1", 
    # Secondary Markers:
    "RGS5", "GJA4", "FABP4", "LDB2"
  ), 
  `Cancer.Epithelial` = c(
    # Primary Markers: 
    "EPCAM", "KRT19", "KRT8", "KRT18", 
    # Secondary Markers: 
    "KRT5", "KRT14"
  ),
  `Other.Epithelial` = c(
    # Primary Markers: 
    "KRT5", "KRT14", 
    # Secondary Markers:
    "EPCAM", "KRT19", "KRT8", "KRT18", "AMBP", "MUC1"
  )
)

# Create a lookup table of valid marker-celltype pairs
marker_celltype_df <- stack(markers_list)
colnames(marker_celltype_df) <- c( "marker", "celltype")

markers_unique <- unique(marker_celltype_df$marker)
markers_identified <- readRDS(file = file.path(sc_dir, "markers_identified.rds"))
markers_ct_B <- markers_identified$Consensus
markers_B <- unique(stack(markers_ct_B)$value)

get_legend <- function(myplot) {
  tmp <- ggplotGrob(myplot)
  legend_index <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  tmp$grobs[[legend_index]]
}


# Load metadata for the ST data -------------------------------------------

st_subtype <- read.csv(file.path(st_dir, "subtypes_brca_visium.csv"))
st_meta <- read.csv(file.path(st_dir, "metadata_brca_visium.csv"), row.names = 1)
st_meta <- merge(st_meta, st_subtype[, -1], by = "slide")
st_meta$sid <- st_meta$slide
studies <- unique(st_meta$dataset)

# use only slides that have tiles/patches available
valid_image_samples <- list.files(file.path(st_dir, "tiles_tiff"), recursive = FALSE, full.names = FALSE)
st_meta <- st_meta %>% dplyr::filter(sid %in% valid_image_samples)

st_meta$patch_dir <- gsub("patches", "tiles_tiff", st_meta$patch_dir)

# Create spatial location data
spatial <- data.frame()
for (i in 1:nrow(st_meta)) {
  sp_file <- st_meta[i, "spatial_path"]
  header_flag <- has_header(sp_file)
  sp <- read.csv(sp_file, header = header_flag, sep = ',')  
  colnames(sp) <- c("spot_id", "in_tissue", 
                    "array_row", "array_col", 
                    "y","x")
  sp$sid <- st_meta[i, "slide"]
  spatial <- rbind(spatial, sp)
}


# Load per-sample region labels and join to spatial expects files ---------
#  - <patch_dir>/<sid>_spots_coordinates_with_regions.csv (created by create_patches_brca_from_csv.py)
region_csv_path <- function(patch_dir, sid) {
  file.path(patch_dir, paste0(sid, "_spots_coordinates_with_regions.csv"))
}

regions_tbl <- map_dfr(seq_len(nrow(st_meta)), function(i) {
  sid <- st_meta$sid[i]
  pdir <- st_meta$patch_dir[i]
  p <- region_csv_path(pdir, sid)
  
  if (!file.exists(p)) {
    warning("Region CSV not found for ", sid, ": ", p)
    return(tibble())
  }
  
  r <- read.csv(p, stringsAsFactors = FALSE)
  
  # Expect columns: barcode, pxl_x, pxl_y, (optional array_row/array_col/in_tissue), tissue_region, oot_reason
  r <- r %>%
    dplyr::rename(spot_id = barcode) %>%
    dplyr::mutate(sid = sid)
  r
})

if (nrow(regions_tbl) == 0) stop("No region tables were found; please check 'patch_dir' and filenames.")

# join region info to spatial (by sid + spot_id)
spatial_regions <- spatial %>%
  left_join(regions_tbl %>% dplyr::select(sid, spot_id, tissue_region, oot_reason),
            by = c("sid", "spot_id"))

# Keep only kept regions (tissue_region >= 0)
spatial_kept <- spatial_regions %>% dplyr::filter(!is.na(tissue_region) & tissue_region >= 0)


# Expand st_meta: one row per (sid, region) -------------------------------
#   - add sid_region (e.g., "10xGenomics_06232020-S1_region0")
#   - set per-region output folder under proportion_dir

st_meta_regions <- spatial_kept %>%
  distinct(sid, tissue_region) %>%
  arrange(sid, tissue_region) %>%
  left_join(st_meta, by = c("sid" = "sid"))

if (nrow(st_meta_regions) == 0) stop("No (sid, region) combinations found after filtering.")

st_meta_regions <- st_meta_regions %>%
  mutate(
    sid_region = paste0(sid, "_region", tissue_region),
    region_id = tissue_region,
    # each original proportion_dir gets a per-region subfolder
    proportion_dir = "/fh/working/sun_w/zsui/st2image_data/CARD_results_ref_B_matrix_regions",
    patch_dir = file.path(patch_dir, sid_region)
  )

# helper: subset spatial to a particular (sid, region)
spatial_for <- function(sid, region_id) {
  spatial_kept %>%
    dplyr::filter(sid == !!sid, tissue_region == !!region_id) %>%
    # CARD expects columns: spot_id, x, y, in_tissue, array_row, array_col, sid
    dplyr::select(spot_id, tissue_region, in_tissue, array_row, array_col, y, x, sid) %>%
    mutate(sid = paste0(sid, "_region", tissue_region))
}


# Load stdata_list and single-cell references -----------------------------

stdata_list <- readRDS(file.path(st_dir, "CARD_analysis_ref_B_matrix", "stdata_list_all.rds"))

stdata_regions_list <- list()
for (s1 in unique(st_meta_regions$sid)) {
  stdata_s <- stdata_list[[s1]]
  fn <- file.path("/fh/working/sun_w/zsui/st2image_data/tiles_tiff", s1,
                  paste0(s1, "_spots_coordinates_with_regions.csv"))
  spots_regions <- read.csv(fn) %>%
    dplyr::filter(tissue_region >= 0) %>%
    dplyr::select(c(barcode, tissue_region))  %>%
    split(.$tissue_region)

  for (i in seq_len(length(spots_regions))) {
    region <- paste0(s1, "_region", names(spots_regions)[i])
    spots <- spots_regions[[1]]
    stdata_regions_list[[region]] <- stdata_s[, unique(spots$barcode), drop = T]
  }
}

saveRDS(stdata_regions_list,
        file = file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", "stdata_regions_list_all.rds"))

stdata_regions_list <- readRDS(file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", "stdata_regions_list_all.rds"))



# ---- Wu_2021_filtered counts
sc_dir_wu <- file.path(sc_dir, "Wu_2021_filtered")
sc_count_wu <- readMM(file.path(sc_dir_wu, "count_matrix_sparse.mtx"))
sc_meta_wu  <- read.csv(file = file.path(sc_dir_wu, "metadata.csv"))
rownames(sc_meta_wu) <- sc_meta_wu$X
sc_gene_wu  <- read.csv(file = file.path(sc_dir_wu, "count_matrix_genes.tsv"), sep = "\t", header = FALSE)
colnames(sc_gene_wu) <- "symbol.sc"
sc_bar_wu <- read.csv(file = file.path(sc_dir_wu, "count_matrix_barcodes.tsv"), sep = "\t", header = FALSE)
colnames(sc_count_wu) <- sc_bar_wu$V1
rownames(sc_count_wu) <- sc_gene_wu$symbol.sc
sc_count_wu <- as(sc_count_wu, "CsparseMatrix")

# ---- Mo_2024_filtered counts
sc_dir_mo <- file.path(sc_dir, "Mo_2024_filtered")
sc_count_mo <- readMM(file.path(sc_dir_mo, "count_matrix_sparse.mtx"))
sc_meta_mo <- read.csv(file = file.path(sc_dir_mo, "metadata.csv"))
rownames(sc_meta_mo) <- sc_meta_mo$X
sc_gene_mo  <- read.csv(file = file.path(sc_dir_mo, "count_matrix_genes.tsv"), sep = "\t", header = FALSE)
colnames(sc_gene_mo) <- "symbol.sc"
sc_bar_mo <- read.csv(file = file.path(sc_dir_mo, "count_matrix_barcodes.tsv"), sep = "\t", header = FALSE)
colnames(sc_count_mo) <- sc_bar_mo$V1
rownames(sc_count_mo) <- sc_gene_mo$symbol.sc
sc_count_mo <- as(sc_count_mo, "CsparseMatrix")

# ---- B matrices (Wu & Mo)
B_Wu_list <- readRDS(file = file.path(sc_dir, "B_Wu_list.rds"))
B_Mo_list <- readRDS(file = file.path(sc_dir, "B_Mo_list.rds"))

# Run four deconvolution variants, per region -----------------------------
run_region_batch <- function(st_meta_region_df,
                             label,
                             sc_count = NULL, sc_meta = NULL,
                             B_custom = NULL,
                             cell_type_res = "cell_type_group",
                             use_selectInfo = TRUE,
                             overwrite = F,
                             base_root = "/fh/working/sun_w/zsui/st2image_data/CARD_results_ref_B_matrix_regions",
                             seed = 20251009) {
  
  # one folder per variant (all slides/regions for this variant will go here)
  out_root <- file.path(base_root, label)
  dir.create(out_root, recursive = TRUE, showWarnings = FALSE)
  
  err_rows <- list()
  
  for (i in seq_len(nrow(st_meta_region_df))) {
    sid <- st_meta_region_df$sid[i]
    region_id <- st_meta_region_df$region_id[i]
    dataset <- st_meta_region_df$dataset[i]
    dir0 <- st_meta_region_df$dir[i]
    exprs_dir <- st_meta_region_df$exprs_dir[i]
    spatial_dir <- st_meta_region_df$spatial_dir[i]
    image_path  <- st_meta_region_df$image_path[i]
    diameter <- st_meta_region_df$diameter[i]
    spatial_path<- st_meta_region_df$spatial_path[i]
    patch_dir <- st_meta_region_df$patch_dir[i]
    barcodes_path <- st_meta_region_df$barcodes_path[i]
    count_matrix_path <- st_meta_region_df$count_matrix_path[i]
    features_path <- st_meta_region_df$features_path[i]
    
    spatial_region <- spatial_for(sid, region_id)
    
    message(sprintf("\n[Deconv] %s  region=%s  -> %s", sid, region_id, out_root))
    
    st_meta_row <- tibble::tibble(
      sid = paste0(sid, "_region",region_id),
      slide = sid, dataset = dataset, dir = dir0,
      exprs_dir = exprs_dir, spatial_dir = spatial_dir,
      image_path = image_path, diameter = diameter,
      spatial_path = spatial_path, patch_dir = patch_dir,
      barcodes_path = barcodes_path, count_matrix_path = count_matrix_path,
      features_path = features_path, proportion_dir = out_root
    )
    
    tryCatch({
      run_deconvolution(
        sc_count = sc_count,
        sc_meta = sc_meta,
        st_meta = st_meta_row,
        spatial_location = spatial_region,
        stdata_list = stdata_regions_list,
        B_custom = B_custom,
        cell_type_res = cell_type_res,
        save.CARDobj = TRUE,  
        save.amat = TRUE,
        match.slide = FALSE,
        match.subtype = FALSE,
        use_selectInfo = use_selectInfo,
        # file_suffix  = paste0("_region", region_id),
        overwrite = overwrite,
        seed = seed
      )
    },
    error = function(e) {
      msg <- conditionMessage(e)
      warning(sprintf("[ERROR] variant=%s sid=%s region=%s :: %s", label, sid, region_id, msg))
      err_rows[[length(err_rows) + 1]] <<- tibble::tibble(
        timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
        reference = label,
        sid = sid,
        region_id = region_id,
        out_dir = out_root,
        error = msg
      )
    })
  }
  
  # write per-variant error log (append if exists)
  err_df <- dplyr::bind_rows(err_rows)
  if (nrow(err_df) > 0) {
    per_variant_err_fp <- file.path(out_root, "_errors.csv")
    if (file.exists(per_variant_err_fp)) {
      old <- tryCatch(readr::read_csv(per_variant_err_fp, show_col_types = FALSE),
                      error = function(...) NULL)
      if (!is.null(old) && nrow(old) > 0) err_df <- dplyr::bind_rows(old, err_df)
    }
    readr::write_csv(err_df, per_variant_err_fp)
  }
  return(err_df)
}

all_errors <- list()

cat("==== Wu_2021_filtered per region ====\n")
all_errors[["Wu_counts"]] <- run_region_batch(
  st_meta_region_df = st_meta_regions,
  label = "CARD_proportions_ref_Wu_filtered",
  sc_count = sc_count_wu,
  sc_meta = sc_meta_wu,
  B_custom = NULL,
  cell_type_res = "cell_type_group",
  use_selectInfo = TRUE,
  overwrite = FALSE,
  seed = 20251009
)

cat("==== Mo_2024_filtered per region ====\n")
all_errors[["Mo_counts"]] <- run_region_batch(
  st_meta_region_df = st_meta_regions,
  label = "CARD_proportions_ref_Mo_filtered",
  sc_count = sc_count_mo,
  sc_meta  = sc_meta_mo,
  B_custom = NULL,
  cell_type_res = "cell_type_group",
  use_selectInfo = TRUE,
  overwrite = FALSE,
  seed = 20251009
)

cat("==== B_count_Wu (signature genes) per region ====\n")
signature_genes_wu <- intersect(rownames(B_Wu_list$CPM$B), rownames(sc_count_wu))
sc_count_wu_sig <- sc_count_wu[signature_genes_wu, , drop = FALSE]
all_errors[["Bcount_Wu"]] <- run_region_batch(
  st_meta_region_df = st_meta_regions,
  label = "CARD_proportions_B_count_Wu",
  sc_count = sc_count_wu_sig,
  sc_meta  = sc_meta_wu,
  B_custom = NULL,
  cell_type_res = "cell_type_group",
  use_selectInfo = FALSE,
  overwrite = FALSE,
  seed = 20251009
)

cat("==== B_count_Mo (signature genes) per region ====\n")
signature_genes_mo <- intersect(rownames(B_Mo_list$CPM$B), rownames(sc_count_mo))
sc_count_mo_sig <- sc_count_mo[signature_genes_mo, , drop = FALSE]
all_errors[["Bcount_Mo"]] <- run_region_batch(
  st_meta_region_df = st_meta_regions,
  label = "CARD_proportions_B_count_Mo",
  sc_count = sc_count_mo_sig,
  sc_meta  = sc_meta_mo,
  B_custom = NULL,
  cell_type_res = "cell_type_group",
  use_selectInfo = FALSE,
  overwrite = FALSE,
  seed = 20251009
)

cat("==== B_matrix_Wu (two types) per region ====\n")
wu_type_errors <- list()
for (type1 in names(B_Wu_list)[1:2]) {
  B_custom <- B_Wu_list[[type1]]$B
  B_custom <- B_custom[!stringr::str_detect(rownames(B_custom), "^chr[0-9]+_"), , drop = FALSE]
  wu_type_errors[[type1]] <- run_region_batch(
    st_meta_region_df = st_meta_regions,
    label = paste0("CARD_proportions_B_Wu_", type1),
    B_custom = B_custom,
    sc_count = NULL,
    sc_meta  = NULL,
    cell_type_res = "cell_type_group",
    use_selectInfo = FALSE,
    overwrite = FALSE,
    seed = 20251009
  )
}
all_errors[["Bmatrix_Wu"]] <- dplyr::bind_rows(wu_type_errors)

cat("==== B_matrix_Mo (two types) per region ====\n")
mo_type_errors <- list()
for (type1 in names(B_Mo_list)[1:2]) {
  B_custom <- B_Mo_list[[type1]]$B
  B_custom <- B_custom[!stringr::str_detect(rownames(B_custom), "^chr[0-9]+_"), , drop = FALSE]
  mo_type_errors[[type1]] <- run_region_batch(
    st_meta_region_df = st_meta_regions,
    label = paste0("CARD_proportions_B_Mo_", type1),
    B_custom = B_custom,
    sc_count = NULL,
    sc_meta  = NULL,
    cell_type_res = "cell_type_group",
    use_selectInfo = FALSE,
    overwrite = FALSE,
    seed = 20251009
  )
}
all_errors[["Bmatrix_Mo"]] <- dplyr::bind_rows(mo_type_errors)

cat("==== All per-region deconvolution jobs completed ====\n")

# ---- Global error log (aggregated across all variants)
errors_all <- dplyr::bind_rows(all_errors)
if (nrow(errors_all) > 0) {
  global_err_fp <- file.path("/fh/working/sun_w/zsui/st2image_data/CARD_results_ref_B_matrix_regions",
                             paste0("_errors_all_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv"))
  readr::write_csv(errors_all, global_err_fp)
  message(sprintf("[Summary] Wrote global error log: %s", global_err_fp))
} else {
  message("[Summary] No errors recorded across all variants.")
}


# Obtain proportions by different references ------------------------------

# Set directories
sc_dir <- "/fh/working/sun_w/zsui/st2image_data/brca_scRNAseq_datasets/CARD_reference"
st_dir <- "/fh/working/sun_w/zsui/st2image_data"

dir_ref_B_Mo <- list(naive = "/fh/working/sun_w/zsui/st2image_data/CARD_results_ref_B_matrix_regions/CARD_proportions_ref_Mo_filtered",
                     B_count = "/fh/working/sun_w/zsui/st2image_data/CARD_results_ref_B_matrix_regions/CARD_proportions_B_count_Mo",
                     B_CPM = "/fh/working/sun_w/zsui/st2image_data/CARD_results_ref_B_matrix_regions/CARD_proportions_B_Mo_CPM",
                     B_Log = "/fh/working/sun_w/zsui/st2image_data/CARD_results_ref_B_matrix_regions/CARD_proportions_B_Mo_Log"
)
dir_ref_B_Wu <- list(naive = "/fh/working/sun_w/zsui/st2image_data/CARD_results_ref_B_matrix_regions/CARD_proportions_ref_Wu_filtered",
                     B_count = "/fh/working/sun_w/zsui/st2image_data/CARD_results_ref_B_matrix_regions/CARD_proportions_B_count_Wu",
                     B_CPM = "/fh/working/sun_w/zsui/st2image_data/CARD_results_ref_B_matrix_regions/CARD_proportions_B_Wu_CPM",
                     B_Log = "/fh/working/sun_w/zsui/st2image_data/CARD_results_ref_B_matrix_regions/CARD_proportions_B_Wu_Log"
)


samples <- st_meta_regions[, "sid_region"]
types <- names(dir_ref_B_Mo)

card_proportions <- data.frame()
missing_files <- data.frame(reference = character(),
                            type = character(),
                            sid_region = character(),
                            file_path = character(),
                            stringsAsFactors = FALSE)

for (type1 in types) {
  for (s1 in samples) {
    # --- Mo_2024 reference ---
    file_path1 <- file.path(dir_ref_B_Mo[[type1]], paste0("Proportion_", s1, "_cell_type_group.csv"))
    
    if (file.exists(file_path1)) {
      prop_ref1 <- read.csv(file_path1, check.names = FALSE)
      colnames(prop_ref1)[1] <- "X"
      prop_ref1$sid_region <- s1
      prop_ref1$sid <- gsub("_region[0-9]+", "", s1)
      prop_ref1$reference.1 <- "Mo_2024"
      prop_ref1$reference.2 <- type1
    } else {
      message("File not found: ", file_path1)
      missing_files <- rbind(missing_files, data.frame(
        reference = "Mo_2024",
        type = type1,
        sid_region = s1,
        file_path = file_path1
      ))
      prop_ref1 <- NULL
    }
    
    # --- Wu_2021 reference ---
    file_path2 <- file.path(dir_ref_B_Wu[[type1]], paste0("Proportion_", s1, "_cell_type_group.csv"))
    
    if (file.exists(file_path2)) {
      prop_ref2 <- read.csv(file_path2, check.names = FALSE)
      colnames(prop_ref2)[1] <- "X"
      prop_ref2$sid_region <- s1
      prop_ref2$sid <- gsub("_region[0-9]+", "", s1)
      prop_ref2$reference.1 <- "Wu_2021"
      prop_ref2$reference.2 <- type1
    } else {
      message("File not found: ", file_path2)
      missing_files <- rbind(missing_files, data.frame(
        reference = "Wu_2021",
        type = type1,
        sid_region = s1,
        file_path = file_path2
      ))
      prop_ref2 <- NULL
    }
    
    card_proportions <- bind_rows(card_proportions, bind_rows(prop_ref1, prop_ref2))
  }
}

write.csv(card_proportions,
          file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", 
                    "CARD_proportions_all_Wu_Mo.csv"),
          row.names = FALSE)

write.csv(missing_files,
          file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", 
                    "CARD_missing_files_Wu_Mo.csv"),
          row.names = FALSE)

# Compare different references --------------------------------------------

# Load ST data and CARD proportions
stdata_list <- readRDS(file.path(st_dir, "CARD_analysis_ref_B_matrix", "stdata_list_all.rds"))
stdata_regions_list <- readRDS(file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", "stdata_regions_list_all.rds"))


card_proportions <- read.csv(file.path(st_dir, "CARD_analysis_ref_B_matrix_regions",
                                       "CARD_proportions_all_Wu_Mo.csv"),
                             # row.names = 1, 
                             check.names = F)

card_proportions_long <- card_proportions %>%
  mutate(PlasmaMerged = ifelse(reference.1 == "Mo_2024", Plasma, Plasmablasts)) %>%
  dplyr::select(sid, sid_region, reference.1, reference.2, X, all_of(celltypes)) %>%
  pivot_longer(cols = all_of(celltypes), names_to = "celltype", values_to = "proportion") %>%
  tidyr::unite(reference, reference.1, reference.2, sep = "_", remove = FALSE)

card_proportions_wide <- card_proportions_long %>%
  dplyr::select(-reference) %>%
  pivot_wider(names_from = c(reference.1, reference.2), values_from = proportion)






card_proportions_final <- card_proportions %>%
  dplyr::filter(reference.1 == "Mo_2024", reference.2 == "B_count") %>%
  dplyr::select(sid, sid_region, X, all_of(celltypes)) %>%
  pivot_longer(cols = all_of(celltypes), names_to = "celltype", values_to = "proportion")
  

samples_train <- read.csv("samples.csv") %>%
  dplyr::rename(sid = Individual,
         celltype = Cell_type) %>%
  left_join(card_proportions_final)


samples_train_mean <- samples_train %>%
  dplyr::group_by(sid_region, celltype) %>%
  summarise(mean = mean(proportion)) %>%
  ungroup()

write.csv(samples_train_mean, "samples_train_mean.csv")
# + Boxplot for proportions -----------------------------------------------

pdf(file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", paste0("CARD_props_boxplot_compare_refs.pdf")),
    height = 15, width = 12)

for(study in studies){
  plot_prop <- card_proportions_long[startsWith(card_proportions_long$sid, study), ]
  
  sid_list <- unique(plot_prop$sid)
  sid_chunks <- split(sid_list, ceiling(seq_along(sid_list) / 7))
  
  for (i in seq_along(sid_chunks)) {
    current_sids <- sid_chunks[[i]]
    n_sids <- length(current_sids)
    n_empty <- 7 - n_sids
    
    # Main plot
    p_main <- list()
    for (sid in current_sids) {
      plot_data <- plot_prop[plot_prop$sid == sid, ] 
      plot_data$sid <- gsub(paste0(study, "_"), "", plot_data$sid )
      
      p_main[[sid]] <- ggplot(plot_data, aes(y = proportion, color = reference)) +
        geom_boxplot() +
        facet_grid(sid ~ celltype, scales = "free_y") +
        theme_bw() +
        theme(legend.position = "bottom") +
        labs(y = "CARD Proportion")
    }
    
    legend <- get_legend(p_main[[1]])
    
    empty_plots <- replicate(n_empty, ggplot() + theme_void(), simplify = FALSE)
    all_plots <- c(p_main, empty_plots)
    
    # Combine the 6 plots vertically
    plot_stack <- all_plots[[1]] / all_plots[[2]] / all_plots[[3]] / all_plots[[4]] / all_plots[[5]] / all_plots[[6]]  / all_plots[[7]] & 
      theme(
        legend.position = "none",
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        strip.text.y = element_text(size = 6)
      )
    
    # Add title and legend
    patch <- (plot_stack / wrap_elements(full = legend)) + 
      plot_annotation(
        title = paste0(study, " (Page ", i, " of ", length(sid_chunks), ")"),
        theme = theme(plot.title = element_text(size = 15))
      ) +
      plot_layout(
        guides = 'collect',
        axes = "collect",
        heights = unit(c(rep(1, 7), 0.1), rep('inches', 8))
      )
    
    # Print the final result
    print(patch)
    
  }
}

dev.off()

table(card_proportions_long$sid_region, card_proportions_long$reference)

# Compare with individual marker gene's expression ---------------------------------

references <- unique(card_proportions_long$reference)

sc_dir <- "/fh/working/sun_w/zsui/st2image_data/brca_scRNAseq_datasets/CARD_reference"

for (s1 in unique(card_proportions_long$sid_region)) {
  message("Processing: ", s1)
  
  # if (file.exists(file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", paste0("gene_correlations_", s1, ".rds")))) {
  #   next
  # }
  
  # 1. Prepare 'props_s1_dt' using data.table, proportion data
  props_s1_df <- card_proportions_long[card_proportions_long$sid_region == s1, ] 
  
  props_s1_df <- props_s1_df %>%
    mutate(celltype = case_when(
      reference.1 == "Wu_2021" & celltype == "PlasmaMerged" ~ "Plasmablasts",
      reference.1 == "Mo_2024" & celltype == "PlasmaMerged" ~ "Plasma",
      TRUE ~ celltype
    )) %>%
    distinct() %>%
    dplyr::filter(!is.na(proportion))
  
  if (nrow(props_s1_df) == 0) {
    message("  Skipping ", s1, ": props_s1_df is empty.")
    all_samples_correlation_results[[s1]] <- NULL # Or an empty data.table
    next
  }
  props_s1_dt <- as.data.table(props_s1_df)
  props_s1_dt[, X := as.character(X)]
  setkey(props_s1_dt, X)
  
  # 2. Prepare 'expr_long_combined_dt', expression data 
  stdata_s <- stdata_regions_list[[s1]] # raw count matrix: genes x spots
  markers_plot <- intersect(rownames(stdata_s), markers_B)
  
  gene_size <- rowSums(stdata_s) # total expression per gene across all spots.
  gene_size[gene_size == 0] <- 1
  
  library_size <- colSums(stdata_s) # total expression (library size) per spot across all genes.
  library_size[library_size == 0] <- 1
  
  expr_matrices <- list(
    raw = stdata_s[markers_plot, unique(props_s1_dt$X), drop = FALSE],
    gene_norm = sweep(stdata_s, 1, gene_size, "/")[markers_plot, unique(props_s1_dt$X), drop = FALSE], # each gene’s profile across spots sums to 1.
    lib_norm = sweep(stdata_s, 2, library_size[colnames(stdata_s)], FUN = "/")[markers_plot, unique(props_s1_dt$X), drop = FALSE]
  )
  
  expr_list_dt <- vector("list", length(expr_matrices))
  names(expr_list_dt) <- names(expr_matrices) 
  
  for (expr_name in names(expr_matrices)) {
    mat <- expr_matrices[[expr_name]]
    # Ensure mat is a standard R matrix for as.data.table to work as expected with keep.rownames
    dt_wide <- as.data.table(as.matrix(mat), keep.rownames = "marker")
    # Melt to long format. variable.factor=FALSE ensures 'X' (spot IDs) are characters, not factors.
    dt_long <- melt(dt_wide, 
                    id.vars = "marker", 
                    variable.name = "X", 
                    value.name = "Expression", 
                    variable.factor = FALSE, # Important for consistent 'X' type
                    verbose = FALSE) 
    dt_long[, expr_type := expr_name]
    expr_list_dt[[expr_name]] <- dt_long
  }
  expr_long_combined_dt <- rbindlist(expr_list_dt, use.names = TRUE, fill = FALSE)
  setkey(expr_long_combined_dt, X)
  
  # 3. Perform the data.table join
  df_merged_dt <- expr_long_combined_dt[props_s1_dt, on = "X", allow.cartesian = TRUE, nomatch = 0]
  
  if (nrow(df_merged_dt) == 0) {
    message("  Skipping ", s1, ": df_merged_dt is empty after join (no common spots 'X' or other issue).")
    all_samples_correlation_results[[s1]] <- NULL
    next
  }
  
  # Ensure Expression and proportion are numeric for correlation
  df_merged_dt[, Expression := as.numeric(Expression)]
  df_merged_dt[, proportion := as.numeric(proportion)]
  
  # 4. Calculate correlations using data.table's efficient group-by
  map_dt <- rbindlist(
    lapply(names(markers_ct_B), function(ct) 
      data.table(celltype = ct, marker = markers_ct_B[[ct]])
    ),
    use.names = TRUE, fill = TRUE
  )
  setkey(map_dt, marker, celltype)
  
  df_valid <- map_dt[df_merged_dt, on = c("marker", "celltype"), nomatch = 0]
  
  sample_results_dt <- df_valid[, {
    complete_obs <- complete.cases(Expression, proportion)
    n_complete  <- sum(complete_obs)
    has_var <- n_complete >= 2 &&
      var(Expression[complete_obs], na.rm = FALSE) > 0 &&
      var(proportion[complete_obs], na.rm = FALSE) > 0
    
    pearson_val <- if (has_var) { 
      cor(Expression, proportion, use = "pairwise.complete.obs", method = "pearson") 
      } else NA_real_ 
    
    spearman_val <- if (has_var) {
      cor(Expression, proportion, use = "pairwise.complete.obs", method = "spearman") 
      } else NA_real_
    
    pearson_p <- if (has_var && n_complete >= 3) {
      tryCatch(cor.test(Expression, proportion, method = "pearson")$p.value, error = function(e) NA_real_)
    } else NA_real_
    
    spearman_p <- if (has_var && n_complete >= 3) {
      tryCatch(cor.test(Expression, proportion, method = "spearman")$p.value, error = function(e) NA_real_)
    } else NA_real_
    
    .(pearson = pearson_val, pearson_p = pearson_p,
      spearman = spearman_val, spearman_p = spearman_p,
      n_spots = n_complete)
  }, by = .(marker, reference, celltype, expr_type)] # Grouping variables
  
  sample_results_dt[, sid := s1] # Add sample ID
  saveRDS(sample_results_dt, file = file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", paste0("gene_correlations_", s1, ".rds")))
  rm(sample_results_dt)
}

all_samples_correlations <- list()
for (s1 in unique(card_proportions_long$sid_region)) {
  message("Processing: ", s1)
  all_samples_correlations[[s1]] <- readRDS(file = file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", paste0("gene_correlations_", s1, ".rds")))
}
correlation_results_df <- do.call(bind_rows, all_samples_correlations)

saveRDS(correlation_results_df, file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", "correlation_results_regions_all_Wu_Mo.rds"))

########################################

correlation_results_df <- readRDS(
  file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", "correlation_results_regions_all_Wu_Mo.rds"))

corr_individual <- correlation_results_df %>%
  dplyr::filter(!is.na(pearson) | !is.na(spearman)) %>%
  dplyr::filter(pearson_p < 0.05 | spearman_p < 0.05)
  
# Compute summary statistics per marker
summary_corr <- corr_individual %>%
  group_by(celltype, reference, marker, expr_type) %>%
  summarise(
    n_samples = n(),
    avg_corr.pearson = mean(pearson),
    prop_positive.pearson = mean(pearson > 0),
    avg_corr.spearman = mean(spearman),
    prop_positive.spearman = mean(spearman > 0),
    .groups = "drop"
  )

summary_corr_long <- summary_corr %>%
  pivot_longer(c(prop_positive.pearson, prop_positive.spearman))

ggplot(summary_corr_long, aes(x = celltype, y = value, color = name)) +
  geom_boxplot() +
  facet_grid(expr_type~reference)

# Filter markers with >70% positive correlation
filtered_markers <- summary_corr %>%
  dplyr::filter(prop_positive.pearson > 0.7, prop_positive.spearman > 0.7)

# Step 3: For each cell type, select top markers sorted by average correlation
good_markers_pearson <- filtered_markers %>%
  dplyr::filter(!startsWith(marker, "HLA")) %>%
  group_by(celltype, expr_type) %>%
  arrange(desc(avg_corr.pearson), .by_group = TRUE) %>%
  distinct(celltype, marker, .keep_all = TRUE) %>%
  ungroup()

good_markers_spearman <- filtered_markers %>%
  dplyr::filter(!startsWith(marker, "HLA")) %>%
  group_by(celltype, expr_type) %>%
  arrange(desc(avg_corr.spearman), .by_group = TRUE) %>%
  distinct(celltype, marker, .keep_all = TRUE) %>%
  ungroup()

top_n_markers_pearson <- good_markers_pearson %>%
  group_by(celltype, expr_type) %>%
  slice_head(n = 10) %>%  
  ungroup() %>%
  dplyr::select(celltype, n_samples, marker, expr_type)

top_n_markers_spearman <- good_markers_spearman %>%
  group_by(celltype, expr_type) %>%
  slice_head(n = 10) %>%  
  ungroup() %>%
  dplyr::select(celltype, n_samples, marker, expr_type)

top_marker_list_pearson <- top_n_markers_pearson %>%
  group_by(expr_type) %>%
  group_split() %>%
  setNames(unique(top_n_markers_pearson$expr_type))

top_marker_list_spearman <- top_n_markers_spearman %>%
  group_by(expr_type) %>%
  group_split() %>%
  setNames(unique(top_n_markers_spearman$expr_type))

all_top_markers_pearson <- top_n_markers_pearson %>%
  group_by(celltype) %>%
  distinct(marker)

all_top_markers_spearman <- top_n_markers_spearman %>%
  group_by(celltype) %>%
  distinct(marker)

all_top_markers <- inner_join(all_top_markers_pearson, all_top_markers_spearman)

top_markers_list <- all_top_markers %>%
  split(.$celltype)

top_markers <- unique(all_top_markers$marker)
corr_top_markers <- corr_individual %>%
  inner_join(all_top_markers)

# # Visualize (TO BE UPDATED)
# for (s1 in unique(card_proportions_long$sid)) {
#   # s1 <- st_meta_regions[i, "sid"]
#   message("Processing: ", s1)
#   
#   stdata_s <- stdata_list[[s1]] # raw count matrix: genes x spots
#   
#   pdf(file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", 
#                 paste0("marker_expr_vs_celltype_proportion_", s1, ".pdf")),
#       height = 6, width = 10)
#   
#   for (celltype in celltypes) {
#     expr_type = "linear_norm"
#     markers_plot <- top_marker_list[[expr_type]]
#     markers_plot <- unlist(markers_plot[markers_plot$celltype == celltype, "marker"])
#     
#     # 1. Create all expression versions and pivot them into one long data frame
#     stdata_s_subset <- stdata_s[markers_plot, , drop = FALSE]
#     
#     all_gene_col_sums <- colSums(stdata_s) 
#     
#     expr_matrices <- list(
#       raw = stdata_s_subset,
#       log1p = log1p(stdata_s_subset),
#       linear_norm = sweep(stdata_s_subset, 2, all_gene_col_sums[colnames(stdata_s_subset)], FUN = "/") * 1e4
#     )
#     
#     expr_list_dt <- vector("list", length(expr_matrices))
#     names(expr_list_dt) <- names(expr_matrices) 
#     
#     for (expr_name in names(expr_matrices)) {
#       mat <- expr_matrices[[expr_name]]
#       # Ensure mat is a standard R matrix for as.data.table to work as expected with keep.rownames
#       dt_wide <- as.data.table(as.matrix(mat), keep.rownames = "marker")
#       # Melt to long format. variable.factor=FALSE ensures 'X' (spot IDs) are characters, not factors.
#       dt_long <- melt(dt_wide, 
#                       id.vars = "marker", 
#                       variable.name = "X", 
#                       value.name = "Expression", 
#                       variable.factor = FALSE, # Important for consistent 'X' type
#                       verbose = FALSE) 
#       dt_long[, expr_type := expr_name]
#       expr_list_dt[[expr_name]] <- dt_long
#     }
#     expr_long_combined_dt <- rbindlist(expr_list_dt, use.names = TRUE, fill = FALSE)
#     setkey(expr_long_combined_dt, X)
#     
#     # 2. Prepare 'props_s1_dt' using data.table
#     props_s1_df <- card_proportions_long[card_proportions_long$sid == s1, ] 
#     
#     props_s1_dt <- as.data.table(props_s1_df)
#     props_s1_dt[, X := as.character(X)]
#     setkey(props_s1_dt, X)
#     
#     # 3. Perform the data.table join
#     df_merged_dt <- expr_long_combined_dt[props_s1_dt, on = "X", allow.cartesian = TRUE, nomatch = 0L]
#     df_merged_dt[, Expression := as.numeric(Expression)]
#     df_merged_dt[, proportion := as.numeric(proportion)]
#     
#     df_plot <- df_merged_dt
#     
#     p <- ggplot(df_plot, aes(x = proportion, y = Expression)) +
#       geom_point(alpha = 0.4) +
#       geom_smooth(method = "lm", se = FALSE, color = "blue", linetype = "dashed") +
#       stat_cor(method = "pearson", label.x.npc = "left", label.y.npc = "top", size = 3, color = "blue") +
#       facet_grid(expr_type ~ marker, scales = "free_y") +
#       labs(
#         title = paste0("Marker Expression vs. ", celltype, " Proportion"),
#         subtitle = paste0("Slide: ", s1),
#         x = paste(celltype, "proportion"),
#         y = "Expression"
#       ) +
#       theme_bw()
#     
#     print(p)
#   }
#   dev.off()
# }


# Compare with pseudo-marker gene's expression ---------------------------------

## Helper to compute per-spot relative marker expression using B-matrix reference means
rel_marker_expr <- function(genes, st_exprs, B_matrix, ref_cts) {
  genes <- intersect(genes, rownames(st_exprs))             # genes present in ST
  genes <- intersect(genes, rownames(B_matrix))               # genes present in B
  if (length(genes) == 0) {
    v <- rep(NA_real_, ncol(st_exprs))
    names(v) <- colnames(st_exprs)
    return(v)
  }
  ref_cts <- intersect(ref_cts, colnames(B_matrix))
  if (length(ref_cts) == 0) {
    v <- rep(NA_real_, ncol(st_exprs))
    names(v) <- colnames(st_exprs)
    return(v)
  }
  
  # mean ref expression per gene across CTs (vector length = |genes|)
  ref_means <- rowMeans(B_matrix[genes, ref_cts, drop = FALSE], na.rm = TRUE)
  ref_means[ref_means == 0] <- NA
  
  # divide each gene row in st_exprs by its ref mean, then average across genes per spot
  st_sub <- st_exprs[genes, , drop = FALSE]
  adj <- sweep(st_sub, 1, ref_means[rownames(st_sub)], "/")
  out  <- colMeans(adj, na.rm = TRUE)                    # vector length = spots
  names(out) <- colnames(st_exprs)
  out
}

B_Mo_list <- readRDS(file = file.path(sc_dir, "B_Mo_list.rds"))
B_Wu_list <- readRDS(file = file.path(sc_dir, "B_Wu_list.rds"))

celltypes <- c(
  "T.cell",
  "Endothelial",
  "CAFs/PVL",
  "Cancer.Epithelial",
  "Other.Epithelial",
  "Myeloid",
  "B.cell",
  "Plasma",
  "Plasmablasts",
  "PlasmaMerged"
)

relative_marker_exprs_list <- readRDS(file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", "relative_marker_exprs_list.rds"))


relative_marker_exprs_list <- list()
for (s1 in unique(card_proportions_long$sid_region)) {
  # s1 <- st_meta_regions[i, "sid_region"]
  # print(s1)
  
  # sid <- st_meta_regions[i, "sid"]
  
  # B matrices
  paths <- list(
    Mo_2024_naive = file.path("/fh/working/sun_w/zsui/st2image_data/CARD_results_ref_B_matrix_regions/CARD_proportions_ref_Mo_filtered",
                              paste0("CARD_alg_matrix_", s1, "_cell_type_group.rds")),
    Wu_2021_naive = file.path("/fh/working/sun_w/zsui/st2image_data/CARD_results_ref_B_matrix_regions/CARD_proportions_ref_Wu_filtered",
                              paste0("CARD_alg_matrix_", s1, "_cell_type_group.rds")),
    Mo_2024_B_count = file.path("/fh/working/sun_w/zsui/st2image_data/CARD_results_ref_B_matrix_regions/CARD_proportions_B_count_Mo",
                                paste0("CARD_alg_matrix_", s1, "_cell_type_group.rds")),
    Wu_2021_B_count = file.path("/fh/working/sun_w/zsui/st2image_data/CARD_results_ref_B_matrix_regions/CARD_proportions_B_count_Wu",
                                paste0("CARD_alg_matrix_", s1, "_cell_type_group.rds"))
  )
  
  # Safely read if file exists
  safe_read_B <- function(path) {
    if (file.exists(path)) {
      obj <- readRDS(path)
      if (!is.null(obj[["B"]])) return(obj[["B"]])
    }
    NULL
  }
  
  # Build refs list
  refs <- list(
    Mo_2024_naive = safe_read_B(paths$Mo_2024_naive),
    Wu_2021_naive = safe_read_B(paths$Wu_2021_naive),
    Mo_2024_B_count = safe_read_B(paths$Mo_2024_B_count),
    Wu_2021_B_count = safe_read_B(paths$Wu_2021_B_count),
    Mo_2024_B_CPM = B_Mo_list$CPM[["B"]],
    Wu_2021_B_CPM = B_Wu_list$CPM[["B"]],
    Mo_2024_B_Log = B_Mo_list$Log[["B"]],
    Wu_2021_B_Log = B_Wu_list$Log[["B"]]
  )
  
  # Remove NULL entries
  refs <- refs[!sapply(refs, is.null)]
  
  found_refs <- names(refs)[!sapply(refs, is.null)]
  message("Successfully loaded: ", paste(found_refs, collapse = ", "))
  
  stdata_s <- stdata_regions_list[[s1]]
  # gene_size <- rowSums(stdata_s) # total expression per gene across all spots.
  # gene_size[gene_size == 0] <- 1
  
  library_size <- colSums(stdata_s) # total expression (library size) per spot across all genes.
  library_size[library_size == 0] <- 1
  
  expr_matrices <- list(
    raw = stdata_s,
    # gene_norm = sweep(stdata_s, 1, gene_size, "/"), # each gene’s profile across spots sums to 1.
    lib_norm = sweep(stdata_s, 2, library_size[colnames(stdata_s)], FUN = "/")
  )
  
  # --- 2. Calculate reference-normalized pseudo-bulk expression ---
  exist <- relative_marker_exprs_list[[s1]]
  relative_expr_list <- list()
  for (ref_nm in names(refs)) {
    B_matrix = refs[[ref_nm]]
    cts <- intersect(celltypes, colnames(B_matrix))
    # 
    # exist_ref <- exist %>%
    #   dplyr::filter(reference == ref_nm)
    
    # cts <- cts[!cts %in% colnames(exist_ref)]
    
    # build one tibble per cell type, then join by X
    rst_list <- map(cts, function(ct) {
      markers <- tryCatch(
        top_markers_list[[ct]] %>% dplyr::pull(marker) %>% unique(),
        error = function(e) character(0)
      )
      
      if (length(markers) == 0) {
        return(tibble(X = character(0), !!ct := numeric(0)))
      }
      
      expr1 <- rel_marker_expr(
        genes  = markers,
        st_exprs  = expr_matrices[["lib_norm"]],
        B_matrix  = B_matrix,
        ref_cts   = cts
      )
      
      tibble(X = names(expr1), !!ct := as.numeric(expr1))
    })
    
    rst_expr <- Reduce(function(x, y) dplyr::full_join(x, y, by = "X"), rst_list) %>%
      dplyr::arrange(X) %>%
      tibble::column_to_rownames(var = "X")
    
    ## Row-normalize to get relative marker expression per spot
    row_sums <- rowSums(rst_expr, na.rm = TRUE)
    row_sums[row_sums == 0] <- NA
    relative_expr <- sweep(rst_expr, 1, row_sums + 1e-8, "/") %>%
      tibble::rownames_to_column("X") %>%
      mutate(reference = ref_nm)
    relative_expr_list[[ref_nm]] <- relative_expr
  }
  
  relative_expr_s1 <- do.call(bind_rows, relative_expr_list)
  relative_marker_exprs_list[[s1]] <- relative_expr_s1
}

saveRDS(relative_marker_exprs_list, 
        file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", "relative_marker_exprs_list.rds"))

relative_marker_exprs_list <- readRDS(file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", "relative_marker_exprs_list.rds"))

correlations_pseudo_list <- list()
for (s1 in unique(card_proportions_long$sid_region)) {
  # s1 <- st_meta_regions[i, "sid"]
  print(s1)
  
  props_s1_df <- card_proportions_long[card_proportions_long$sid_region == s1, ] 
  props_s1_df <- props_s1_df %>%
    mutate(celltype = case_when(
      reference.1 == "Wu_2021" & celltype == "PlasmaMerged" ~ "Plasmablasts",
      reference.1 == "Mo_2024" & celltype == "PlasmaMerged" ~ "Plasma",
      TRUE ~ celltype
    )) %>%
    dplyr::select(X, reference, celltype, proportion) %>%
    mutate(X = as.character(X)) %>%
    distinct()
  
  relative_expr_s1 <- relative_marker_exprs_list[[s1]]
  
  rel_long_s1 <- relative_expr_s1 %>%
    pivot_longer(cols = -c(X, reference),
                 names_to = "celltype",
                 values_to = "rel_expr") %>%
    mutate(X = as.character(X))
  
  
  df_join <- inner_join(rel_long_s1, props_s1_df,
                        by = c("X","reference","celltype")) %>%
    na.omit()
  
  cor_by_ct_ref <- df_join %>%
    group_by(reference, celltype) %>%
    summarize(
      n = sum(complete.cases(rel_expr, proportion)),
      var_expr = var(rel_expr, na.rm = TRUE),
      var_prop = var(proportion, na.rm = TRUE),
      pearson = if (n >= 2 && var_expr > 0 && var_prop > 0)
        cor(rel_expr, proportion, method = "pearson", use = "complete.obs") else NA_real_,
      spearman = if (n >= 2 && var_expr > 0 && var_prop > 0)
        cor(rel_expr, proportion, method = "spearman", use = "complete.obs") else NA_real_,
      pearson_p = if (n >= 3 && !is.na(pearson))
        tryCatch(cor.test(rel_expr, proportion, method = "pearson")$p.value,
                 error = function(e) NA_real_) else NA_real_,
      spearman_p = if (n >= 3 && !is.na(spearman))
        tryCatch(cor.test(rel_expr, proportion, method = "spearman")$p.value,
                 error = function(e) NA_real_) else NA_real_
      , .groups = "drop") %>%
    mutate(sid = s1, .before = 1)
  
  correlations_pseudo_list[[s1]] <- cor_by_ct_ref
}

correlation_results_pseudo <- dplyr::bind_rows(correlations_pseudo_list)

saveRDS(correlation_results_pseudo, 
        file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", "correlation_results_pseudogenes_regions.rds"))


# HERE --------------------------------------------------------------------

# Visualization
correlation_results_pseudo <- readRDS(file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", "correlation_results_pseudogenes.rds"))

summ <- correlation_results_pseudo %>%
  na.omit() %>%
  distinct(sid, reference, celltype, n)

poor_regions <- summ %>% 
  dplyr::filter(n < 30) %>%
  distinct(sid) %>%
  pull(sid)

poor_regions_corr <- correlation_results_pseudo %>%
  dplyr::filter(sid %in% poor_regions)

corr_pseudo <- correlation_results_pseudo %>%
  dplyr::filter(n >= 30)

corr_pseudo_long <- corr_pseudo %>%
  dplyr::rename(
    pearson.corr = pearson,
    spearman.corr = spearman,
    pearson.p = pearson_p,
    spearman.p = spearman_p
  ) %>%
  pivot_longer(
    cols = c(pearson.corr, spearman.corr, pearson.p, spearman.p),
    names_to = c("method", "stat"),
    names_sep = "\\.",
    values_to = "value"
  ) %>%
  pivot_wider(
    names_from = stat,
    values_from = value
  ) 

corr_pseudo_sig <- corr_pseudo_long %>%
  dplyr::filter(p < 0.05)

avg_corr <- corr_pseudo %>%
  group_by(reference, celltype) %>%
  summarise(
    mean_pearson  = mean(pearson,  na.rm = TRUE),
    mean_spearman = mean(spearman, na.rm = TRUE),
    n_samples = dplyr::n(),
    .groups = "drop"
  )

p_heat_pearson <- ggplot(avg_corr, aes(x = celltype, y = reference, fill = mean_pearson)) +
  geom_tile(color = "gray80") +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                       midpoint = 0, limits = c(-1, 1), na.value = "gray90") +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        strip.text = element_text(face = "bold")) +
  labs(title = "Average Pearson correlations across samples",
       x = "Cell type", y = "Reference", fill = "Pearson")

p_heat_spearman <- ggplot(avg_corr, aes(x = celltype, y = reference, fill = mean_spearman)) +
  geom_tile(color = "gray80") +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                       midpoint = 0, limits = c(-1, 1), na.value = "gray90") +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        strip.text = element_text(face = "bold")) +
  labs(title = "Average Spearman correlations across samples",
       x = "Cell type", y = "Reference", fill = "Spearman")

print(p_heat_pearson)
print(p_heat_spearman)

p_sample_heat_pearson <- ggplot(corr_pseudo,
                                aes(x = celltype, y = sid, fill = pearson)) +
  geom_tile(color = "gray80") +
  facet_wrap(~ reference, nrow = 1) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                       midpoint = 0, limits = c(-1, 1), na.value = "gray90") +
  theme_minimal(base_size = 8) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text.y = element_text(size = 6)) +
  labs(title = "Pearson correlation per sample",
       x = "Cell type", y = "Sample (sid_region)", fill = "Pearson")

p_sample_heat_spearman <- ggplot(corr_pseudo,
                                 aes(x = celltype, y = sid, fill = spearman)) +
  geom_tile(color = "gray80") +
  facet_wrap(~ reference, nrow = 1) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                       midpoint = 0, limits = c(-1, 1), na.value = "gray90") +
  theme_minimal(base_size = 8) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.text.y = element_text(size = 6)) +
  labs(title = "Spearman correlation per sample",
       x = "Cell type", y = "Sample (sid_region)", fill = "Spearman")

pdf(file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", "correlation_heatmaps_per_region.pdf"),
    height = 13, width = 10)
print(p_sample_heat_pearson)
print(p_sample_heat_spearman)
dev.off()

# corr_plot <- corr_pseudo_sig %>%
#   mutate(celltype_plot = case_when(
#     celltype %in% c("Plasma", "Plasmablasts") ~ "PlasmaMerged",
#     TRUE ~ celltype
#   )) %>%
#   group_by(sid, reference, celltype_plot) %>%
#   summarise(
#     pearson  = mean(pearson,  na.rm = TRUE),
#     spearman = mean(spearman, na.rm = TRUE),
#     .groups = "drop"
#   )
# 
# corr_wide_plot_pearson <- corr_plot %>%
#   dplyr::select(sid, reference, celltype_plot, pearson) %>%
#   pivot_wider(names_from = reference, values_from = pearson)

corr_plot <- corr_pseudo_sig %>%
  mutate(celltype_plot = case_when(
    celltype %in% c("Plasma", "Plasmablasts") ~ "PlasmaMerged",
    TRUE ~ celltype
  )) 






# Pivot wide for Spearman
corr_wide_plot_spearman <- corr_plot %>%
  dplyr::select(sid, n, reference, celltype_plot, spearman) %>%
  pivot_wider(names_from = reference, values_from = spearman)




# Example pair to compare (change as needed)
ref_x <- "Mo_2024_B_CPM"
ref_y <- "Wu_2021_B_CPM"

p_scatter_P <- ggplot(corr_wide_plot_pearson,
                      aes_string(x = ref_x, y = ref_y, color = "celltype_plot")) +
  geom_point(alpha = 0.7, size = 2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray40") +
  facet_wrap(~ celltype_plot) +
  theme_bw() +
  labs(title = paste("Sample-level Pearson correlations:", ref_x, "vs", ref_y),
       x = paste(ref_x, "correlation"),
       y = paste(ref_y, "correlation"),
       color = "Cell type")
print(p_scatter_P)


p_scatter_S <- ggplot(corr_wide_plot_spearman,
                      aes_string(x = ref_x, y = ref_y, color = "celltype_plot")) +
  geom_point(alpha = 0.7, size = 2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray40") +
  facet_wrap(~ celltype_plot) +
  theme_bw() +
  labs(title = paste("Sample-level Spearman correlations:", ref_x, "vs", ref_y),
       x = paste(ref_x, "correlation"),
       y = paste(ref_y, "correlation"),
       color = "Cell type")
print(p_scatter_S)


corr_long_P <- corr_wide_plot_pearson %>%
  pivot_longer(
    cols = starts_with("Mo_2024_") | starts_with("Wu_2021_"),
    names_to = "reference", values_to = "corr"
  ) %>%
  mutate(
    study = case_when(
      grepl("^Mo_2024_", reference) ~ "Mo",
      grepl("^Wu_2021_", reference) ~ "Wu",
      TRUE ~ NA_character_
    ),
    type = case_when(
      grepl("^Mo_2024_B_", reference) ~ sub("^Mo_2024_B_", "", reference),
      grepl("^Wu_2021_B_", reference) ~ sub("^Wu_2021_B_", "", reference),
      grepl("^Mo_2024_",  reference)  ~ sub("^Mo_2024_",  "", reference),
      grepl("^Wu_2021_",  reference)  ~ sub("^Wu_2021_",  "", reference),
      TRUE ~ NA_character_
    )
  )

corr_pairs_P <- corr_long_P %>%
  dplyr::select(sid, celltype_plot, type, study, corr) %>%
  pivot_wider(names_from = study, values_from = corr)  # columns "Mo" and "Wu"

pairwise_summary_P <- corr_pairs_P %>%
  group_by(type, celltype_plot) %>%
  summarise(
    pearson = cor(Mo, Wu, use = "complete.obs", method = "pearson"),
    spearman = cor(Mo, Wu, use = "complete.obs", method = "spearman"),
    n = sum(complete.cases(Mo, Wu)),
    .groups = "drop"
  )
print(pairwise_summary_P)

# Repeat using Spearman metric table
corr_long_S <- corr_wide_plot_spearman %>%
  pivot_longer(
    cols = starts_with("Mo_2024_") | starts_with("Wu_2021_"),
    names_to = "reference", values_to = "corr"
  ) %>%
  mutate(
    study = case_when(
      grepl("^Mo_2024_", reference) ~ "Mo",
      grepl("^Wu_2021_", reference) ~ "Wu",
      TRUE ~ NA_character_
    ),
    type = case_when(
      grepl("^Mo_2024_B_", reference) ~ sub("^Mo_2024_B_", "", reference),
      grepl("^Wu_2021_B_", reference) ~ sub("^Wu_2021_B_", "", reference),
      grepl("^Mo_2024_",  reference)  ~ sub("^Mo_2024_",  "", reference),
      grepl("^Wu_2021_",  reference)  ~ sub("^Wu_2021_",  "", reference),
      TRUE ~ NA_character_
    )
  )

corr_pairs_S <- corr_long_S %>%
  dplyr::select(sid, celltype_plot, type, study, corr) %>%
  pivot_wider(names_from = study, values_from = corr)

pairwise_summary_S <- corr_pairs_S %>%
  group_by(type, celltype_plot) %>%
  summarise(
    pearson  = cor(Mo, Wu, use = "complete.obs", method = "pearson"),
    spearman = cor(Mo, Wu, use = "complete.obs", method = "spearman"),
    n = sum(complete.cases(Mo, Wu)),
    .groups = "drop"
  )
print(pairwise_summary_S)


###################################

thr = 0.5
corr_plot_wide <- corr_plot %>%
  dplyr::filter(p < 0.05) %>%
  mutate(
    study = case_when(
      grepl("^Mo_2024_", reference) ~ "Mo",
      grepl("^Wu_2021_", reference) ~ "Wu",
      TRUE ~ NA_character_
    ),
    type = case_when(
      grepl("^Mo_2024_B_", reference) ~ sub("^Mo_2024_B_", "", reference),
      grepl("^Wu_2021_B_", reference) ~ sub("^Wu_2021_B_", "", reference),
      grepl("^Mo_2024_",  reference)  ~ sub("^Mo_2024_",  "", reference),
      grepl("^Wu_2021_",  reference)  ~ sub("^Wu_2021_",  "", reference),
      TRUE ~ NA_character_
    )
  ) %>%
  dplyr::select(-c(reference, p, var_expr, var_prop, celltype)) %>%
  pivot_wider(names_from = c(study), values_from = corr)
  
corr_plot_wide_pearson <- corr_plot_wide %>%
  dplyr::filter(method == "pearson")

corr_plot_wide_spearman <- corr_plot_wide %>%
  dplyr::filter(method == "spearman")

# ---------- PEARSON ----------
# Count points with Mo >= thr AND Wu >= thr per facet
counts_P <- corr_plot_wide_pearson %>%
  dplyr::mutate(hit = (Mo >= thr) & (Wu >= thr)) %>%
  dplyr::group_by(type, celltype_plot) %>%
  dplyr::summarise(n_both = sum(hit, na.rm = TRUE), .groups = "drop") %>%
  dplyr::mutate(label = paste0("Corr>=", thr, ": ", n_both))

p_pairs_P <- ggplot(corr_plot_wide_pearson, aes(x = Mo, y = Wu, color = celltype_plot)) +
  geom_point(alpha = 0.6, size = 1.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray40") +
  geom_hline(yintercept = thr, color = "red", linetype = "dashed") +
  geom_vline(xintercept = thr, color = "red", linetype = "dashed") +
  facet_grid(type ~ celltype_plot) +
  coord_equal() +
  theme_bw() +
  labs(
    title = paste0("Mo vs Wu agreement (Pearson) -  threshold = ", thr),
    x = "Mo", y = "Wu", color = "Cell type"
  ) +
  geom_text(
    data = counts_P,
    aes(x = -Inf, y = -Inf, label = label),
    inherit.aes = FALSE,
    hjust = -0.1, vjust = -0.5, size = 3.2
  )
print(p_pairs_P)

counts_S <- corr_plot_wide_spearman %>%
  dplyr::mutate(hit = (Mo >= thr) & (Wu >= thr)) %>%
  dplyr::group_by(type, celltype_plot) %>%
  dplyr::summarise(n_both = sum(hit, na.rm = TRUE), .groups = "drop") %>%
  dplyr::mutate(label = paste0("Corr>=", thr, ": ", n_both))

p_pairs_S <- ggplot(corr_plot_wide_spearman, aes(x = Mo, y = Wu, color = celltype_plot)) +
  geom_point(alpha = 0.6, size = 1.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray40") +
  geom_hline(yintercept = thr, color = "red", linetype = "dashed") +
  geom_vline(xintercept = thr, color = "red", linetype = "dashed") +
  facet_grid(type ~ celltype_plot) +
  coord_equal() +
  theme_bw() +
  labs(
    title = paste0("Mo vs Wu agreement (Spearman)  - threshold = ", thr),
    x = "Mo", y = "Wu", color = "Cell type"
  ) +
  geom_text(
    data = counts_S,
    aes(x = -Inf, y = -Inf, label = label),
    inherit.aes = FALSE,
    hjust = -0.1, vjust = -0.5, size = 3.2
  )
print(p_pairs_S)

pdf(file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", 
              "correlation_scatterplots_compare_Wu_vs_Mo_regions.pdf"),
    height = 10, width = 10)
p_pairs_P / p_pairs_S &
  theme(legend.position = "none")
dev.off()


corr_good_P <- corr_plot_wide_pearson %>%
  dplyr::filter((Mo >= thr) & (Wu >= thr)) %>%
  mutate(better_ref = ifelse(is.na(Mo), "Wu_2021", 
                             ifelse(is.na(Wu), "Mo_2024", 
                                    ifelse(Mo > Wu, "Mo_2024", "Wu_2021")))) %>%
  rowwise() %>%
  mutate(better_corr = max(Mo, Wu, na.rm = T)) %>%
  dplyr::select(-c(Wu, Mo)) 


corr_good_S <- corr_plot_wide_spearman %>%
  dplyr::filter((Mo >= thr) & (Wu >= thr)) %>%
  mutate(better_ref = ifelse(is.na(Mo), "Wu_2021", 
                             ifelse(is.na(Wu), "Mo_2024", 
                                    ifelse(Mo > Wu, "Mo_2024", "Wu_2021")))) %>%
  rowwise() %>%
  mutate(better_corr = max(Mo, Wu, na.rm = T)) %>%
  dplyr::select(-c(Wu, Mo)) 


corr_best_P <- corr_good_P %>%
  group_by(sid, celltype_plot) %>%
  arrange(desc(better_corr)) %>%
  slice_head(n = 1) %>%
  mutate(
    type = ifelse(
    type %in% c("count", "CPM", "Log"), 
    paste0("B_", type),
    type
  ),
  reference = paste0(better_ref, "_", type)) %>%
  dplyr::select(-c(better_ref, type))

  
corr_best_S <- corr_good_S %>%
  group_by(sid, celltype_plot) %>%
  arrange(desc(better_corr)) %>%
  slice_head(n = 1) %>%
  mutate(
    type = ifelse(
      type %in% c("count", "CPM", "Log"), 
      paste0("B_", type),
      type
    ),
    reference = paste0(better_ref, "_", type)) %>%
  dplyr::select(-c(better_ref, type))

corr_best <- bind_rows(corr_best_S, corr_best_P)

good_samples <- corr_best %>%
  # distinct(sid, celltype_plot) %>%
  split(.$celltype_plot)

good_refs <- list()
for (ct in names(good_samples)) {
  good <- good_samples[[ct]] %>%
    ungroup() %>%
    dplyr::select(sid, celltype_plot, reference, method, better_corr) %>%
    distinct(sid, celltype_plot, reference) 
  
  more_than <- good %>%
    count(sid) %>%
    dplyr::filter(n > 1) %>%
    pull(sid)
  
  ref_unique <- good %>%
    dplyr::filter(!sid %in% more_than) 
  
  more_than_ref <- good %>% 
    dplyr::filter(sid %in% more_than)
  
  more_than_corr <- corr_plot %>%
    inner_join(more_than_ref) 
  
  for (i in more_than) {
    more_than_i <- more_than_corr %>%
      dplyr::filter(sid == i)
    
    refs <- unique(more_than_i$reference)
    
    poor_ref <- more_than_i %>%
      dplyr::filter(corr < thr) %>%
      pull(reference)
    
    if (length(poor_ref) == 1) {
      ref_unique <- rbind(ref_unique,
                          tibble(sid = i, 
                                 celltype_plot = ct, 
                                 reference = refs[!refs %in% poor_ref])
                          )
    } else if (length(poor_ref) == 0) {
      tmp <- more_than_i %>%
        dplyr::select(sid, celltype_plot, reference, method, corr) %>%
        pivot_wider(names_from = reference, values_from = corr) 
      
      tmp$diff <- unlist(abs(tmp[, refs[1]] - tmp[, refs[2]])) 
      
      higher <- tmp %>%
        arrange(diff) %>%
        slice_tail(n=1) %>%
        dplyr::select(all_of(refs))
      
      ref_unique <- rbind(ref_unique,
                          tibble(sid = i, 
                                 celltype_plot = ct, 
                                 reference = names(higher)[which.max(higher)])
      )
    }
  }
  good_refs[[ct]] <- ref_unique
}



card_proportions_cpm <- card_proportions %>%
  dplyr::filter(reference.2 == "B_CPM") 



tile_root_dir <- "/fh/working/sun_w/zsui/st2image_data/tiles_tiff"

card_prop_cts <- list()
for (ct in names(good_refs)){
  ref <- good_refs[[ct]] %>%
    dplyr::rename(sid_region = sid,
                  celltype = celltype_plot)
  
  props <- card_proportions_long %>%
    inner_join(ref) %>%
    dplyr::rename(!!ct := proportion) %>%
    dplyr::select(sid, sid_region, X, all_of(ct), reference) %>%
    mutate(X = paste0(X, "_", sid_region),
           tile_path = file.path(tile_root_dir, sid, sid_region, paste0(X, ".tif")))
  
  corrs <- corr_plot %>%
    mutate(celltype = celltype_plot,
           sid_region = sid) %>%
    inner_join(ref) %>%
    dplyr::select(sid_region, method, corr) %>%
    pivot_wider(values_from = corr, names_from = method, names_prefix = "corr.")
  
  props <- props %>%
    inner_join(corrs)
  
  colnames(props) <- gsub("/", ".", colnames(props))
  ct <- gsub("/", ".", ct)
  write.csv(props, file = file.path(tile_root_dir, paste0("input_proportions_", ct, ".csv")))
  card_prop_cts[[ct]] <- props
}
saveRDS(card_prop_cts, file = file.path(tile_root_dir, "card_prop_cts_input.rds"))


lapply(card_prop_cts, function(df) length(unique(df$sid)))









# Select "CPM" reference
corr_pairs_P$corr <- "Pearson"
corr_pairs_S$corr <- "Spearman"
corr_pairs <- rbind(corr_pairs_P, corr_pairs_S) %>%
  mutate(better_ref = ifelse(Mo > Wu, "Mo_2024", "Wu_2021"))  

thr = 0.5

good_samples_P <- corr_pairs_P %>%
  dplyr::filter(type == "CPM") %>%
  dplyr::filter((Mo >= thr) & (Wu >= thr)) %>%
  split(.$celltype_plot)

good_samples_S <- corr_pairs_S %>%
  dplyr::filter(type == "CPM") %>%
  dplyr::filter((Mo >= thr) & (Wu >= thr)) %>%
  split(.$celltype_plot)

good_samples <- list()
for (ct in names(good_samples_P)) {
  good_samples[[ct]] <- unique(c(good_samples_S[[ct]]$sid, good_samples_P[[ct]]$sid))
}

good_samples <- list()
for (ct in names(good_samples_P)) {
  good_samples[[ct]] <- intersect(good_samples_S[[ct]]$sid, good_samples_P[[ct]]$sid)
}

good_samples <- list()
for (ct in names(good_samples_P)) {
  good_samples[[ct]] <- good_samples_S[[ct]]$sid
}

############

card_proportions_cpm <- card_proportions %>%
  dplyr::filter(reference.2 == "B_CPM") 

good_samples <- good_samples_S

card_prop_cts <- list()
for (ct in celltypes){
  samples <- good_samples[[ct]]
  
  props <- card_proportions_cpm %>%
    dplyr::select(sid_region, X, reference.1, all_of(ct)) %>%
    dplyr::filter(sid_region %in% samples) %>%
    dplyr::rename(reference = reference.1) 
  
  corrs <- corr_pairs %>%
    dplyr::filter(sid %in% samples, type == "CPM", celltype_plot == ct) 
  
  references <- corrs %>% 
    distinct(all_of(ct), sid, corr, better_ref) %>%
    pivot_wider(values_from = better_ref, names_from = corr)
  
  same_refs <- references %>%
    dplyr::filter(Pearson == Spearman)
  
  diff_refs <- references %>%
    dplyr::filter(Pearson != Spearman) %>%
    pull(sid) 
  
  diff_corrs <- corrs %>%
    dplyr::filter(sid %in% diff_refs) 
  
  references <- corrs %>%
    distinct(sid, better_ref) %>%
    dplyr::rename(reference = better_ref) 
  
  props <- props %>%
    inner_join(references) %>%
    # rowwise() %>%
    mutate(X = paste0(sid, "_", X)) %>%
    dplyr::select(X, all_of(ct), sid, reference)
  
  # colnames(props) <- gsub("/", ".", colnames(props))
  # ct <- gsub("/", ".", ct)
  # write.csv(props, file = file.path(st_dir, "images_celltypes", paste0("input_proportions_", ct, ".csv")))
  card_prop_cts[[ct]] <- props
}



# HERE: POOR --------------------------------------------------------------

samples <- c("Mo_2024_HTA12_246_3", "Mo_2024_HTA12_254_8")

props <- card_proportions_cpm %>%
  dplyr::select(sid, X, reference.1, all_of(celltypes)) %>%
  dplyr::filter(sid %in% samples) %>%
  dplyr::rename(reference = reference.1) 

corrs <- corr_pairs %>%
  dplyr::filter(sid %in% samples, type == "CPM", celltype_plot == ct,
                corr == "Spearman") 

references <- corrs %>%
  distinct(sid, better_ref) %>%
  dplyr::rename(reference = better_ref) 

props <- props %>%
  inner_join(references) %>%
  # rowwise() %>%
  mutate(X = paste0(sid, "_", X)) %>%
  dplyr::select(X, all_of(celltypes), sid, reference)



dir_ref_B_Mo <- list(naive = "/fh/working/sun_w/zsui/st2image_data/CARD_results_ref_B_matrix_regions/CARD_proportions_ref_Mo_filtered",
                     B_count = "/fh/working/sun_w/zsui/st2image_data/CARD_results_ref_B_matrix_regions/CARD_proportions_B_count_Mo",
                     B_CPM = "/fh/working/sun_w/zsui/st2image_data/CARD_results_ref_B_matrix_regions/CARD_proportions_B_Mo_CPM",
                     B_Log = "/fh/working/sun_w/zsui/st2image_data/CARD_results_ref_B_matrix_regions/CARD_proportions_B_Mo_Log"
                     # SCT = "/fh/working/sun_w/zsui/st2image_data/CARD_results_ref_B_matrix_regions/CARD_proportions_B_Mo_SCT"
)
dir_ref_B_Wu <- list(naive = "/fh/working/sun_w/zsui/st2image_data/CARD_results_ref_B_matrix_regions/CARD_proportions_ref_Wu_filtered",
                     B_count = "/fh/working/sun_w/zsui/st2image_data/CARD_results_ref_B_matrix_regions/CARD_proportions_B_count_Wu",
                     B_CPM = "/fh/working/sun_w/zsui/st2image_data/CARD_results_ref_B_matrix_regions/CARD_proportions_B_Wu_CPM",
                     B_Log = "/fh/working/sun_w/zsui/st2image_data/CARD_results_ref_B_matrix_regions/CARD_proportions_B_Wu_Log"
                     # SCT = "/fh/working/sun_w/zsui/st2image_data/CARD_results_ref_B_matrix_regions/CARD_proportions_B_Wu_SCT"
)

colors = c(T.cell = "#2a5fbd",
           B.cell = "#7ddffa",
           Myeloid = "#117d30",
           PlasmaMerged = "#71f25a",
           `CAFs/PVL` = "#ebc857",
           Endothelial ="#D39200",
           Other.Epithelial = "#F8766D",
           Cancer.Epithelial = "#DB72FB")

cell_type_res = "cell_type_group"

celltypes <- names(colors)

for (s1 in samples) {
  prop <- props %>%
    dplyr::filter(sid == s1) 
  
  output_dir <- ifelse (unique(prop$reference) == "Mo_2024", 
                        "/fh/working/sun_w/zsui/st2image_data/CARD_results_ref_B_matrix_regions/CARD_proportions_B_Mo_CPM",
                        "/fh/working/sun_w/zsui/st2image_data/CARD_results_ref_B_matrix_regions/CARD_proportions_B_Wu_CPM")
  
  spot_id <- gsub(paste0(s1, "_"), "", prop$X)
  spatial_location <- spatial %>%
    dplyr::filter(spot_id %in% spot_id & sid == s1, in_tissue == 1) %>%
    mutate(spot_id = paste0(s1, "_", spot_id)) %>%
    column_to_rownames("spot_id") %>%
    dplyr::select(x,y)
  
  h_w_ratio <- (max(spatial_location_i$y)-min(spatial_location_i$y)) / (max(spatial_location_i$x)-min(spatial_location_i$x))
  
  # Visualization of the proportions of all spots
  prop <- prop %>%
    column_to_rownames("X") %>%
    dplyr::select(all_of(celltypes))
  prop <- prop[rownames(spatial_location), ]
  
  # p1 <- CARD.visualize.pie(proportion = prop, 
  #                          spatial_location = spatial_location, 
  #                          colors = colors,
  #                          radius = 1)
  
  # Visualization of the spatial distribution of the cell type proportion
  CARD.visualize.prop(
    proportion = prop,        
    spatial_location = spatial_location, 
    ct.visualize = "Cancer.Epithelial",                 
    colors = c("lightyellow","orange","red"), 
    NumCols = 3,
    pointSize = 1) + 
    # scale_x_reverse() +
    scale_y_reverse() +
    labs(title = s1)

}

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
# if( nrow(spatial_location_i) > ncol(st_count) ){
#   spatial_location_i <- spatial_location_i[colnames(st_count), ]
# } else {
#   st_count <- st_count[, rownames(spatial_location_i)]
# }
# 
# table(rownames(spatial_location_i) == colnames(st_count))


visualize_proportions(st_meta_regions = st_meta_regions %>% dplyr::filter(slide %in% samples),
                      output_dir = output_dir,
                      ct.visualize = c("Cancer Epithelial"),
                      width = 10,
                      point_size = 1,
                      pie_radius = 0.5,
                      y_reverse = T,
                      x_reverse = F,
                      xy_flip = T)

# prop_ref1 <- read.csv(file.path(dir_ref_B_Mo[[type1]], paste0("Proportion_", s1, "_cell_type_group.csv")), check.names = F)
# colnames(prop_ref1)[1] <- "X"
# prop_ref1$reference.1 <- "Mo_2024"
# prop_ref1$reference.2 <- type1
# prop_ref1$reference.3 <- "w/o chr"
# 
# prop_ref2 <- read.csv(file.path(dir_ref_B_Wu[[type1]], paste0("Proportion_", s1, "_cell_type_group.csv")), check.names = F)
# colnames(prop_ref2)[1] <- "X"
# prop_ref2$reference.1 <- "Wu_2021"
# prop_ref2$reference.2 <- type1
# prop_ref2$reference.3 <- "w/o chr"
# 
# card_proportions <- rbind(card_proportions, bind_rows(prop_ref1, prop_ref2))





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





# 
# colnames(props) <- gsub("/", ".", colnames(props))
# ct <- gsub("/", ".", ct)
# write.csv(props, file = file.path(st_dir, "images_celltypes", paste0("input_proportions_", ct, ".csv")))
# card_prop_cts[[ct]] <- props
# 





saveRDS(card_prop_cts, file = file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", "card_proportions_input.rds") )

card_prop_cts <- readRDS(file = file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", "card_proportions_input.rds") )


sapply(card_prop_cts, function(obj) length(unique(obj$sid)))

#####

good_samples_PT <- good_samples_P[["T.cell"]]

good_samples_ST <- good_samples_S[["T.cell"]]


all_samples <- unique(c(good_samples_ST$sid, good_samples_PT$sid))
common_samples <- intersect(good_samples_ST$sid, good_samples_PT$sid)
samples_T_inconsist <- all_samples[!all_samples %in% common_samples]

p_list <- list()
for (s1 in samples_T_inconsist) {
 
  relative_expr_s1 <- relative_marker_exprs_list[[s1]]
  
  rel_long_s1 <- relative_expr_s1 %>%
    pivot_longer(cols = -c(X, reference),
                 names_to = "celltype",
                 values_to = "rel_expr") %>%
    mutate(X = as.character(X)) %>%
    dplyr::filter(celltype == "T.cell")
  
  props_s1_df <- card_proportions_long[card_proportions_long$sid == s1, ] 
  
  props_s1_df <- props_s1_df %>%
    dplyr::filter(reference.2 == "B_count") %>%
    mutate(celltype = case_when(
      reference.1 == "Wu_2021" & celltype == "PlasmaMerged" ~ "Plasmablasts",
      reference.1 == "Mo_2024" & celltype == "PlasmaMerged" ~ "Plasma",
      TRUE ~ celltype
    )) %>%
    dplyr::select(X, reference, celltype, proportion) %>%
    mutate(X = as.character(X)) %>%
    mutate(reference = gsub("_w/o chr", "", reference)) %>%
    dplyr::filter(celltype == "T.cell")
  
  df_join <- inner_join(rel_long_s1, props_s1_df,
                        by = c("X","reference","celltype")) %>%
    na.omit()
  
  fmt_num <- function(x) ifelse(is.na(x), "NA", formatC(x, digits = 3, format = "f"))
  fmt_p <- function(p) ifelse(is.na(p), "NA", ifelse(p < 1e-3, formatC(p, format = "e", digits = 2), formatC(p, format = "f", digits = 3)))
  
  cor_by_ct_ref <- df_join %>%
    dplyr::group_by(reference, celltype) %>%
    dplyr::summarize(
      n        = sum(stats::complete.cases(rel_expr, proportion)),
      var_expr = stats::var(rel_expr, na.rm = TRUE),
      var_prop = stats::var(proportion, na.rm = TRUE),
      pearson  = if (n >= 2 && var_expr > 0 && var_prop > 0)
        stats::cor(rel_expr, proportion, method = "pearson") else NA_real_,
      spearman = if (n >= 2 && var_expr > 0 && var_prop > 0)
        stats::cor(rel_expr, proportion, method = "spearman") else NA_real_,
      pearson_p  = if (n >= 3 && !is.na(pearson))
        tryCatch(stats::cor.test(rel_expr, proportion, method = "pearson")$p.value,
                 error = function(e) NA_real_) else NA_real_,
      spearman_p = if (n >= 3 && !is.na(spearman))
        tryCatch(stats::cor.test(rel_expr, proportion, method = "spearman")$p.value,
                 error = function(e) NA_real_) else NA_real_,
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      sid = s1,
      label = paste0(
        "Pearson corr = ", fmt_num(pearson), " (p=", fmt_p(pearson_p), ")\n",
        "Spearman corr = ", fmt_num(spearman), " (p=", fmt_p(spearman_p), ")\n",
        "n = ", n
      )
    )
  
  # plot with per-reference panels and correlation text pinned to top-left of each
  p1 <- ggplot(df_join, aes(x = rel_expr, y = proportion)) +
    geom_point(alpha = 0.7, size = 1.6) +
    geom_smooth(method = "lm", se = FALSE, linewidth = 0.6) +
    facet_wrap(~ reference) +
    # place the text at the top-left of each facet
    geom_text(
      data = cor_by_ct_ref,
      aes(x = -Inf, y = Inf, label = label),
      hjust = -0.05, vjust = 1.1, size = 3.2,
      inherit.aes = FALSE
    ) +
    labs(
      title = paste0("T cell (", s1, ")"),
      x = "Relative marker expression",
      y = "Proportion"
    ) +
    theme_bw() +
    theme(
      strip.background = element_rect(fill = "grey95", color = NA),
      panel.grid.minor = element_blank()
    )

  p_list[[s1]] <- p1
}

combined_plot <- wrap_plots(p_list, ncol = 3)

ggsave("T_cell.pdf", width = 16, height = 12)

####


good_samples_PT <- good_samples_P[["Endothelial"]]

good_samples_ST <- good_samples_S[["Endothelial"]]


all_samples <- unique(c(good_samples_ST$sid, good_samples_PT$sid))
common_samples <- intersect(good_samples_ST$sid, good_samples_PT$sid)
samples_endo_inconsist <- all_samples[!all_samples %in% common_samples][sample(1:60, 12, )]

p_list <- list()
for (s1 in samples_endo_inconsist) {
  
  relative_expr_s1 <- relative_marker_exprs_list[[s1]]
  
  rel_long_s1 <- relative_expr_s1 %>%
    pivot_longer(cols = -c(X, reference),
                 names_to = "celltype",
                 values_to = "rel_expr") %>%
    mutate(X = as.character(X)) %>%
    dplyr::filter(celltype == "Endothelial")
  
  props_s1_df <- card_proportions_long[card_proportions_long$sid == s1, ] 
  
  props_s1_df <- props_s1_df %>%
    dplyr::filter(reference.2 == "B_count") %>%
    mutate(celltype = case_when(
      reference.1 == "Wu_2021" & celltype == "PlasmaMerged" ~ "Plasmablasts",
      reference.1 == "Mo_2024" & celltype == "PlasmaMerged" ~ "Plasma",
      TRUE ~ celltype
    )) %>%
    dplyr::select(X, reference, celltype, proportion) %>%
    mutate(X = as.character(X)) %>%
    mutate(reference = gsub("_w/o chr", "", reference)) %>%
    dplyr::filter(celltype == "Endothelial")
  
  df_join <- inner_join(rel_long_s1, props_s1_df,
                        by = c("X","reference","celltype")) %>%
    na.omit()
  
  fmt_num <- function(x) ifelse(is.na(x), "NA", formatC(x, digits = 3, format = "f"))
  fmt_p <- function(p) ifelse(is.na(p), "NA", ifelse(p < 1e-3, formatC(p, format = "e", digits = 2), formatC(p, format = "f", digits = 3)))
  
  cor_by_ct_ref <- df_join %>%
    dplyr::group_by(reference, celltype) %>%
    dplyr::summarize(
      n        = sum(stats::complete.cases(rel_expr, proportion)),
      var_expr = stats::var(rel_expr, na.rm = TRUE),
      var_prop = stats::var(proportion, na.rm = TRUE),
      pearson  = if (n >= 2 && var_expr > 0 && var_prop > 0)
        stats::cor(rel_expr, proportion, method = "pearson") else NA_real_,
      spearman = if (n >= 2 && var_expr > 0 && var_prop > 0)
        stats::cor(rel_expr, proportion, method = "spearman") else NA_real_,
      pearson_p  = if (n >= 3 && !is.na(pearson))
        tryCatch(stats::cor.test(rel_expr, proportion, method = "pearson")$p.value,
                 error = function(e) NA_real_) else NA_real_,
      spearman_p = if (n >= 3 && !is.na(spearman))
        tryCatch(stats::cor.test(rel_expr, proportion, method = "spearman")$p.value,
                 error = function(e) NA_real_) else NA_real_,
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      sid = s1,
      label = paste0(
        "Pearson corr = ", fmt_num(pearson), " (p=", fmt_p(pearson_p), ")\n",
        "Spearman corr = ", fmt_num(spearman), " (p=", fmt_p(spearman_p), ")\n",
        "n = ", n
      )
    )
  
  # plot with per-reference panels and correlation text pinned to top-left of each
  p1 <- ggplot(df_join, aes(x = rel_expr, y = proportion)) +
    geom_point(alpha = 0.7, size = 1.6) +
    geom_smooth(method = "lm", se = FALSE, linewidth = 0.6) +
    facet_wrap(~ reference) +
    # place the text at the top-left of each facet
    geom_text(
      data = cor_by_ct_ref,
      aes(x = -Inf, y = Inf, label = label),
      hjust = -0.05, vjust = 1.1, size = 3.2,
      inherit.aes = FALSE
    ) +
    labs(
      title = paste0("Endothelial (", s1, ")"),
      x = "Relative marker expression",
      y = "Proportion"
    ) +
    theme_bw() +
    theme(
      strip.background = element_rect(fill = "grey95", color = NA),
      panel.grid.minor = element_blank()
    )
  
  p_list[[s1]] <- p1
}

combined_plot <- wrap_plots(p_list, ncol = 3)

ggsave("endothelial.pdf", width = 16, height = 12)




###################################

correlation_results <- readRDS(file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", "correlation_results.rds"))
correlation_results_pseudo <- readRDS(file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", "correlation_results_pseudogenes.rds"))
correlation_results_pseudo1 <- readRDS(file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", "correlation_results_pseudogenes_robust.rds"))
correlation_results_pseudo2 <- readRDS(file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", "correlation_results_pseudogenes2.rds"))

all_data <- list()
for (ref in references) {
  corr_df <- correlation_results[correlation_results$reference == ref, ]
  
  filtered_corr_df <- corr_df %>%
    inner_join(marker_celltype_df, by = c("celltype", "marker"))
  
  agg_corr_df <- filtered_corr_df %>%
    group_by(sid, celltype) %>%
    summarize(mean_corr = mean(correlation, na.rm = TRUE), .groups = "drop") %>%
    mutate(reference = ref)  # Add reference label
  
  all_data[[ref]] <- agg_corr_df
}

# Combine all into a single data.frame
combined_df <- bind_rows(all_data)
combined_df <- merge(combined_df, st_subtype %>% dplyr::select(slide, subtype), 
                     by.x = "sid", by.y = "slide")

combined_df$subtype <- ifelse(!combined_df$subtype %in% c("ER+", "HER2+", "TNBC"),
                              "Unknown", combined_df$subtype)
combined_df <- combined_df %>%
  arrange(subtype, sid) %>%
  mutate(sid = factor(sid, levels = unique(sid)))


#####
correlation_results_pseudo <- merge(correlation_results_pseudo,
                                    st_subtype %>% dplyr::select(slide, subtype), 
                                    by.x = "sid", by.y = "slide")

combined_df_pseudo <- correlation_results_pseudo %>%
  rowwise() %>%
  mutate(marker = gsub(paste0("_", celltype), "", marker)) %>%
  ungroup()

combined_df_pseudo$subtype <- ifelse(!combined_df_pseudo$subtype %in% c("ER+", "HER2+", "TNBC"),
                                     "Unknown", combined_df_pseudo$subtype)

combined_df_pseudo <- combined_df_pseudo %>%
  arrange(subtype, sid) %>%
  mutate(sid = factor(sid, levels = unique(sid)))

pdf(file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", "correlation_heatmaps_average_marker_exprs.pdf"),
    height = 8, width = 15)

corr_min <- -1
corr_max <- 1

p1 <- ggplot(combined_df_pseudo[combined_df_pseudo$marker == "pseudo_mean", ], 
             aes(x = celltype, y = sid)) +
  geom_tile(aes(fill = correlation)) +
  scale_fill_gradient2(
    low = "blue", mid = "white", high = "red", midpoint = 0,
    limits = c(corr_min, corr_max),
    name = "Correlation"
  ) +
  new_scale_fill() +
  geom_tile(aes(x = 0, fill = subtype), width = 0.4) +
  scale_fill_manual(values = subtype_colors, name = "Subtype") +
  facet_wrap(~ reference, nrow = 1) +
  theme_minimal(base_size = 10) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold"),
    axis.text.y = element_text(size = 6)
  ) +
  labs(
    title = "Correlation of Pseudo-Marker Expression with Cell Type Proportions",
    subtitle = "Based on mean log-normalized expression of marker genes",
    x = "Cell Type", 
    y = "Spatial Transcriptomics Slide"
  )

p2 <- ggplot(combined_df_pseudo[combined_df_pseudo$marker == "prop_pseudo_mean", ], 
             aes(x = celltype, y = sid)) +
  geom_tile(aes(fill = correlation)) +
  scale_fill_gradient2(
    low = "blue", mid = "white", high = "red", midpoint = 0,
    limits = c(corr_min, corr_max),
    name = "Correlation"
  ) +
  new_scale_fill() +
  geom_tile(aes(x = 0, fill = subtype), width = 0.4) +
  scale_fill_manual(values = subtype_colors, name = "Subtype") +
  facet_wrap(~ reference, nrow = 1) +
  theme_minimal(base_size = 10) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold"),
    axis.text.y = element_text(size = 6)
  ) +
  labs(
    title = "Correlation of Proportional Pseudo-Marker Expression with Cell Type Proportions",
    subtitle = "Pseudo-marker expression normalized by total pseudo-expression across cell types",
    x = "Cell Type", 
    y = "Spatial Transcriptomics Slide"
  )

p1 + p2 + plot_layout(guides = "collect") & theme(legend.position = "right")

dev.off()

# Boxplot for raw pseudo-expression correlations
p_pseudo_box_raw <- ggplot(
  combined_df_pseudo[combined_df_pseudo$marker == "pseudo_mean", ],
  aes(x = reference, y = correlation, fill = reference)
) +
  geom_boxplot(width = 0.6, outlier.size = 0.8) +
  facet_wrap(~ celltype, ncol = 4, scales = "free_y") +
  scale_fill_brewer(palette = "Set1") +
  theme_bw(base_size = 10) +
  labs(
    title = "Correlation of Pseudo-Marker Expression with Cell Type Proportion",
    subtitle = "Using mean log-normalized expression of marker genes",
    x = "Deconvolution Method",
    y = "Pearson Correlation"
  ) +
  theme(
    strip.text = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"
  )

# Boxplot for proportional pseudo-expression correlations
p_pseudo_box_prop <- ggplot(
  combined_df_pseudo[combined_df_pseudo$marker == "prop_pseudo_mean", ],
  aes(x = reference, y = correlation, fill = reference)
) +
  geom_boxplot(width = 0.6, outlier.size = 0.8) +
  facet_wrap(~ celltype, ncol = 4, scales = "free_y") +
  scale_fill_brewer(palette = "Set1") +
  theme_bw(base_size = 10) +
  labs(
    title = "Correlation of Proportional Pseudo-Marker Expression with Cell Type Proportion",
    subtitle = "Pseudo-marker expression normalized by total pseudo-expression across cell types",
    x = "Deconvolution Method",
    y = "Pearson Correlation"
  ) +
  theme(
    strip.text = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"
  )

# Save to PDF
pdf(file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", "correlation_boxplots_average_marker_exprs.pdf"),
    width = 7, height = 8)

p_pseudo_box_raw / p_pseudo_box_prop

dev.off()

#####
correlation_results_pseudo1 <- merge(correlation_results_pseudo1,
                                    st_subtype %>% dplyr::select(slide, subtype), 
                                    by.x = "sid", by.y = "slide")

combined_df_pseudo <- correlation_results_pseudo1 %>%
  rowwise() %>%
  mutate(marker = gsub(paste0("_", celltype), "", marker)) %>%
  ungroup()

combined_df_pseudo$subtype <- ifelse(!combined_df_pseudo$subtype %in% c("ER+", "HER2+", "TNBC"),
                                     "Unknown", combined_df_pseudo$subtype)

combined_df_pseudo <- combined_df_pseudo %>%
  arrange(subtype, sid) %>%
  mutate(sid = factor(sid, levels = unique(sid)))

pdf(file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", "correlation_heatmaps_average_marker_exprs_robust.pdf"),
    height = 8, width = 15)

corr_min <- -1
corr_max <- 1

p1 <- ggplot(combined_df_pseudo[combined_df_pseudo$marker == "pseudo_mean", ], 
             aes(x = celltype, y = sid)) +
  geom_tile(aes(fill = correlation)) +
  scale_fill_gradient2(
    low = "blue", mid = "white", high = "red", midpoint = 0,
    limits = c(corr_min, corr_max),
    name = "Correlation"
  ) +
  new_scale_fill() +
  geom_tile(aes(x = 0, fill = subtype), width = 0.4) +
  scale_fill_manual(values = subtype_colors, name = "Subtype") +
  facet_wrap(~ reference, nrow = 1) +
  theme_minimal(base_size = 10) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold"),
    axis.text.y = element_text(size = 6)
  ) +
  labs(
    title = "Correlation of Pseudo-Marker Expression with Cell Type Proportions",
    subtitle = "Based on mean log-normalized expression of marker genes",
    x = "Cell Type", 
    y = "Spatial Transcriptomics Slide"
  )

p2 <- ggplot(combined_df_pseudo[combined_df_pseudo$marker == "prop_pseudo_mean", ], 
             aes(x = celltype, y = sid)) +
  geom_tile(aes(fill = correlation)) +
  scale_fill_gradient2(
    low = "blue", mid = "white", high = "red", midpoint = 0,
    limits = c(corr_min, corr_max),
    name = "Correlation"
  ) +
  new_scale_fill() +
  geom_tile(aes(x = 0, fill = subtype), width = 0.4) +
  scale_fill_manual(values = subtype_colors, name = "Subtype") +
  facet_wrap(~ reference, nrow = 1) +
  theme_minimal(base_size = 10) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold"),
    axis.text.y = element_text(size = 6)
  ) +
  labs(
    title = "Correlation of Proportional Pseudo-Marker Expression with Cell Type Proportions",
    subtitle = "Pseudo-marker expression normalized by total pseudo-expression across cell types",
    x = "Cell Type", 
    y = "Spatial Transcriptomics Slide"
  )

p1 + p2 + plot_layout(guides = "collect") & theme(legend.position = "right")

dev.off()

# Boxplot for raw pseudo-expression correlations
p_pseudo_box_raw <- ggplot(
  combined_df_pseudo[combined_df_pseudo$marker == "pseudo_mean", ],
  aes(x = reference, y = correlation, fill = reference)
) +
  geom_boxplot(width = 0.6, outlier.size = 0.8) +
  facet_wrap(~ celltype, ncol = 4, scales = "free_y") +
  scale_fill_brewer(palette = "Set1") +
  theme_bw(base_size = 10) +
  labs(
    title = "Correlation of Pseudo-Marker Expression with Cell Type Proportion",
    subtitle = "Using mean log-normalized expression of marker genes",
    x = "Deconvolution Method",
    y = "Pearson Correlation"
  ) +
  theme(
    strip.text = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"
  )

# Boxplot for proportional pseudo-expression correlations
p_pseudo_box_prop <- ggplot(
  combined_df_pseudo[combined_df_pseudo$marker == "prop_pseudo_mean", ],
  aes(x = reference, y = correlation, fill = reference)
) +
  geom_boxplot(width = 0.6, outlier.size = 0.8) +
  facet_wrap(~ celltype, ncol = 4, scales = "free_y") +
  scale_fill_brewer(palette = "Set1") +
  theme_bw(base_size = 10) +
  labs(
    title = "Correlation of Proportional Pseudo-Marker Expression with Cell Type Proportion",
    subtitle = "Pseudo-marker expression normalized by total pseudo-expression across cell types",
    x = "Deconvolution Method",
    y = "Pearson Correlation"
  ) +
  theme(
    strip.text = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"
  )

# Save to PDF
pdf(file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", "correlation_boxplots_average_marker_exprs_robust.pdf"),
    width = 7, height = 8)

p_pseudo_box_raw / p_pseudo_box_prop

dev.off()

#####
correlation_results_pseudo2 <- merge(correlation_results_pseudo2,
                                    st_subtype %>% dplyr::select(slide, subtype), 
                                    by.x = "sid", by.y = "slide")

combined_df_pseudo <- correlation_results_pseudo2 %>%
  rowwise() %>%
  mutate(marker = gsub(paste0("_", celltype), "", marker)) %>%
  ungroup()

combined_df_pseudo$subtype <- ifelse(!combined_df_pseudo$subtype %in% c("ER+", "HER2+", "TNBC"),
                                     "Unknown", combined_df_pseudo$subtype)

combined_df_pseudo <- combined_df_pseudo %>%
  arrange(subtype, sid) %>%
  mutate(sid = factor(sid, levels = unique(sid)))

pdf(file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", "correlation_heatmaps_average_marker_exprs2.pdf"),
    height = 8, width = 16)

corr_min <- -1
corr_max <- 1

p1 <- ggplot(combined_df_pseudo[combined_df_pseudo$marker == "pseudo_ref_norm_mean", ], 
             aes(x = celltype, y = sid)) +
  geom_tile(aes(fill = correlation)) +
  scale_fill_gradient2(
    low = "blue", mid = "white", high = "red", midpoint = 0,
    limits = c(corr_min, corr_max),
    name = "Correlation"
  ) +
  new_scale_fill() +
  geom_tile(aes(x = 0, fill = subtype), width = 0.4) +
  scale_fill_manual(values = subtype_colors, name = "Subtype") +
  facet_wrap(~reference, nrow = 1) +
  theme_minimal(base_size = 10) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold"),
    axis.text.y = element_text(size = 6)
  ) +
  labs(
    title = "Correlation: Reference-Adjusted Mean Expression vs. Cell Type Proportions",
    subtitle = "Pseudo-bulk score: Mean of (library-norm. ST expression / B-matrix ref. value) per marker",
    x = "Cell Type", 
    y = "Spatial Transcriptomics Slide"
  )

p2 <- ggplot(combined_df_pseudo[combined_df_pseudo$marker == "prop_ref_norm_mean", ], 
             aes(x = celltype, y = sid)) +
  geom_tile(aes(fill = correlation)) +
  scale_fill_gradient2(
    low = "blue", mid = "white", high = "red", midpoint = 0,
    limits = c(corr_min, corr_max),
    name = "Correlation"
  ) +
  new_scale_fill() +
  geom_tile(aes(x = 0, fill = subtype), width = 0.4) +
  scale_fill_manual(values = subtype_colors, name = "Subtype") +
  facet_wrap(~ reference, nrow = 1) +
  theme_minimal(base_size = 10) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold"),
    axis.text.y = element_text(size = 6)
  ) +
  labs(
    title = "Correlation: Proportional Reference-Adjusted Mean Expression vs. Cell Type Proportions",
    subtitle = "Score based on reference-adjusted mean expression, normalized by sum of scores from all cell types per spot",
    x = "Cell Type", 
    y = "Spatial Transcriptomics Slide"
  )

p1 + p2 + plot_layout(guides = "collect") & theme(legend.position = "right")

dev.off()

# Boxplot for raw pseudo-expression correlations
p_pseudo_box_raw <- ggplot(
  combined_df_pseudo[combined_df_pseudo$marker == "pseudo_ref_norm_mean", ],
  aes(x = reference, y = correlation, fill = reference)
) +
  geom_boxplot(width = 0.6, outlier.size = 0.8) +
  facet_wrap(~ celltype, ncol = 4, scales = "free_y") +
  scale_fill_brewer(palette = "Set1") +
  theme_bw(base_size = 10) +
  labs(
    title = "Correlation: Reference-Adjusted Mean Expression vs. Cell Type Proportions",
    subtitle = "Pseudo-bulk score: Mean of (library-norm. ST expression / B-matrix ref. value) per marker",
    x = "Deconvolution Method",
    y = "Pearson Correlation"
  ) +
  theme(
    strip.text = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"
  )

# Boxplot for proportional pseudo-expression correlations
p_pseudo_box_prop <- ggplot(
  combined_df_pseudo[combined_df_pseudo$marker == "prop_ref_norm_mean", ],
  aes(x = reference, y = correlation, fill = reference)
) +
  geom_boxplot(width = 0.6, outlier.size = 0.8) +
  facet_wrap(~ celltype, ncol = 4, scales = "free_y") +
  scale_fill_brewer(palette = "Set1") +
  theme_bw(base_size = 10) +
  labs(
    title = "Correlation: Proportional Reference-Adjusted Mean Expression vs. Cell Type Proportions",
    subtitle = "Score based on reference-adjusted mean expression, normalized by sum of scores from all cell types per spot",
    x = "Deconvolution Method",
    y = "Pearson Correlation"
  ) +
  theme(
    strip.text = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"
  )

# Save to PDF
pdf(file.path(st_dir, "CARD_analysis_ref_B_matrix_regions", "correlation_boxplots_average_marker_exprs2.pdf"),
    width = 8, height = 8)

p_pseudo_box_raw / p_pseudo_box_prop

dev.off()


# Examine two samples -----------------------------------------------------

slides <- c("Mo_2024_HTA12_249_7", "Mo_2024_HTA12_129_9")


marker <- "CD3D"
p_list <- list()
for (s1 in slides) {
  card_prop1 <- card_proportions[card_proportions$sid == s1, ]
  exprs1 <- as.data.frame(stdata_list[[s1]][marker, ]) %>%
    rownames_to_column()
  colnames(exprs1) <- c("X", marker)
  plot1 <- merge(exprs1, card_prop1, by = "X")

  p_list[[s1]] <- ggplot(plot1, aes(x = T.cell, y = CD3D)) +
    geom_point() +
    labs(title = s1, x = "Prop of T.cell", y = paste0("Expression of ", marker))
}

wrap_plots(p_list)

# Examine pure spots ------------------------------------------------------

pure_spot_list <- list()
for (ct in celltypes) {
  subset_df <- card_proportions %>%
    dplyr::select(X, sid, reference, ct) 
  
  subset_wide <- subset_df %>%
    pivot_wider(names_from = reference, values_from = ct)
  
  pure_subset <- subset_wide %>%
    dplyr::filter(
      !is.na(Wu_B_signature),
      !is.na(Mo_B_signature),
      Wu_B_signature > 0.9,
      Mo_B_signature > 0.9
    )
  
  pure_spot_list[[ct]] <- pure_subset
}


B_Wu <- readRDS(file = file.path(sc_dir, paste0("B_matrix_Wu_signature_genes_union_count.rds")))
B_Mo <- readRDS(file = file.path(sc_dir, paste0("B_matrix_Mo_signature_genes_union_count.rds")))

common_genes <- intersect(rownames(B_Wu), rownames(B_Mo))
B_Wu <- B_Wu[common_genes, ]
B_Mo <- B_Mo[common_genes, ]

alphas <- seq(0, 1, by = 0.05)
alpha_results <- data.frame()
expr_cache <- list()
mse_curve_all <- list()  # store curves per cell type
for (celltype in names(pure_spot_list)) {
  pure_df <- pure_spot_list[[celltype]]
  if (nrow(pure_df) == 0) next
  sid_list <- split(pure_df$X, pure_df$sid)
  
  expr_list <- list()
  for (sid in names(sid_list)) {
    if (!sid %in% names(expr_cache)) {
      i <- which(st_meta_regions$slide == sid)
      
      stdata_s <- tryCatch({
        readMM(st_meta_regions[i, "count_matrix_path"])
      }, error = function(e) {
        # If an error occurs, try to read the matrix using read.csv
        tryCatch({
          read.csv(file = st_meta_regions[i, "count_matrix_path"], sep = "\t") %>%
            remove_rownames %>% column_to_rownames(var="X")
        }, error = function(e) {
          tryCatch({
            read.csv(file = st_meta_regions[i, "count_matrix_path"]) %>%
              remove_rownames %>% column_to_rownames(var="X")
          }, error = function(e){
            # If both attempts fail, print an error message
            cat("Both readMM and read.csv failed to read the file:", st_meta_regions[i, "count_matrix_path"], "\n")
          })
        })
      })
      
      if (is.null(stdata_s)) {
        cat("Failed to read the matrix from the file:", st_meta_regions[i, "count_matrix_path"], "\n")
      } else {
        cat("Count matrix successfully read.\n")
      }
      
      if(is.null(colnames(stdata_s)) || is.null(rownames(stdata_s))){
        barcode <- read.table(file = gzfile(st_meta_regions[i, "barcodes_path"]),
                              sep = '\t', header = FALSE)
        barcode_names <- barcode$V1
        feature <- read.table(file = gzfile(st_meta_regions[i, "features_path"]),
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
      
      if (is.null(colnames(stdata_s)) || is.null(rownames(stdata_s))) {
        stop("Names are not assigned to the expression matrix.")
      } else {
        cat("Names are assigned to the expression matrix.\n")
      }
      
      stdata_s <- as(stdata_s, "sparseMatrix")
      stdata_s <- as(stdata_s, "CsparseMatrix")
      
      # Force all matrices to have the same gene set in the same order
      genes <- common_genes[common_genes %in% rownames(stdata_s)]
      missing_genes <- setdiff(common_genes, rownames(stdata_s))
      
      expr_part <- stdata_s[genes, , drop = FALSE]
      
      if (length(missing_genes) > 0) {
        # Add zero-filled rows for missing genes
        zero_mat <- Matrix::Matrix(0, nrow = length(missing_genes), ncol = ncol(expr_part),
                                   dimnames = list(missing_genes, colnames(expr_part)))
        expr_part <- rbind(expr_part, zero_mat)
      }
      
      expr_part <- expr_part[common_genes, , drop = FALSE]
      
      expr_cache[[sid]] <- expr_part
    }
    
    spot_ids <- sid_list[[sid]]
    expr_list[[sid]] <- expr_cache[[sid]][, spot_ids, drop = FALSE]
  }
  
  expr_st <- do.call(cbind, expr_list)  # gene × spot matrix
  
  # Use the celltype_map to resolve actual source columns
  type_Mo <- ifelse(length(celltype_map[[celltype]]) == 2,
                    "Plasma", celltype_map[[celltype]])
  
  type_Wu <- ifelse(length(celltype_map[[celltype]]) == 2,
                    "Plasmablasts", celltype_map[[celltype]])
  
  errors <- sapply(alphas, function(a) {
    B_alpha <- a * B_Wu[, type_Wu] + (1 - a) * B_Mo[, type_Mo]
    y_hat <- outer(B_alpha, rep(1, ncol(expr_st)))
    mean((expr_st - y_hat)^2)
  })
  
  mse_curve_all[[celltype]] <- data.frame(
    alpha = alphas,
    mse = errors,
    celltype = celltype
  )
  best_alpha <- alphas[which.min(errors)]
  alpha_results <- rbind(alpha_results, data.frame(
    celltype = celltype,
    best_alpha = best_alpha,
    min_mse = min(errors)
  ))
}

mse_df <- do.call(rbind, mse_curve_all)

ggplot(mse_df, aes(x = alpha, y = mse, color = celltype)) +
  geom_line(size = 1) +
  facet_wrap(~ celltype, scales = "free_y") +
  labs(
    title = "MSE vs. alpha for Each Cell Type",
    x = "alpha (Weight on B_Wu)",
    y = "Mean Squared Error"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")







# COMPARE -----------------------------------------------------------------

card_proportions_regions <- read.csv(file.path(st_dir, "CARD_analysis_ref_B_matrix_regions",
                                       "CARD_proportions_all_Wu_Mo.csv"),
                             row.names = 1, check.names = F)

card_proportions_old <- read.csv(file.path(st_dir, "CARD_analysis_ref_B_matrix",
                                           "CARD_proportions_all_Wu_Mo_without_chr.csv"),
                                 row.names = 1, check.names = F)

card_proportions_regions$type <- "region"
card_proportions_old$type <- "old"

card_proportions_compare <- bind_rows(card_proportions_old, card_proportions_regions) %>%
  dplyr::select(-c(reference.3, sid_region))

card_compare_long <- card_proportions_compare %>%
  pivot_longer(all_of(celltypes)) %>%
  na.omit()

card_compare_long <- card_compare_long %>%
  mutate(reference = paste0(reference.1, "_", reference.2),
         X = paste0(sid, "_", X)) %>%
  dplyr::select(-c(reference.1, reference.2)) %>%
  pivot_wider(names_from = type, values_from = value)


card_compare_long <- card_compare_long %>%
  na.omit()

card_compare_list <- card_compare_long %>%
  split(.$reference)

for (ref in names(card_compare_list)) {
  png(paste0(ref, ".png"), width = 10, height = 10, units = "in", res = 300)
  ggplot(card_compare_list[[ref]],
         aes(x = old, y = region, color = sid)) +
    geom_point(size = 0.5, alpha = 0.2) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray40") +  # diagonal line
    geom_smooth(method = "lm", se = FALSE, color = "black", linewidth = 0.6) +       # linear fit
    facet_wrap(~name, nrow = 3) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
    theme_bw() +
    theme(legend.position = "none")
  dev.off()
}


outdir <- "png_out"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

nm <- names(card_compare_list)
if (is.null(nm)) nm <- paste0("plot_", seq_along(card_compare_list))  # fallback names

for (i in seq_along(card_compare_list)) {
  ref <- gsub("[^[:alnum:]_\\-]+", "_", nm[i])  # sanitize name for filename
  f <- file.path(outdir, paste0(ref, ".png"))
  png(f, width = 10, height = 10, units = "in", res = 300)
  
  p <- ggplot(card_compare_list[[i]],
              aes(x = old, y = region, color = sid)) +
    geom_point(size = 0.5, alpha = 0.2) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray40") +
    geom_smooth(method = "lm", se = FALSE, color = "black", linewidth = 0.6) +
    facet_wrap(~name, nrow = 3) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
    theme_bw() +
    theme(legend.position = "none")
  
  print(p)   # <-- crucial in loops
  dev.off()
  message("Saved: ", f)
}




library(ggplot2)
library(dplyr)

outdir <- "png_out"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

nm <- names(card_compare_list)
if (is.null(nm)) nm <- paste0("plot_", seq_along(card_compare_list))  # fallback names

for (i in seq_along(card_compare_list)) {
  ref <- nm[i]
  dat <- card_compare_list[[i]]
  
  # --- add counts per facet ---
  dat_counts <- dat %>%
    group_by(name) %>%
    mutate(n_spots = n()) %>%
    ungroup() %>%
    mutate(name_count = paste0(name, " (n=", n_spots, ")"))
  
  f <- file.path(outdir, paste0(gsub("[^[:alnum:]_\\-]+", "_", ref), ".png"))
  png(f, width = 8, height = 8, units = "in", res = 300)
  
  p <- ggplot(dat_counts,
              aes(x = old, y = region, color = sid)) +
    geom_point(size = 0.5, alpha = 0.2) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray40") +
    geom_smooth(method = "lm", se = FALSE, color = "black", linewidth = 0.6) +
    facet_wrap(~name_count, nrow = 3) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
    ggtitle(ref) +
    theme_bw() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      legend.position = "none"
    )
  
  print(p)
  dev.off()
  message("Saved: ", f)
}

