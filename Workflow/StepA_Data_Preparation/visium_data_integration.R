library(Seurat)
library(dplyr)

st_dir <- "/fh/working/sun_w/zsui/st2image_data/brca_visium_datasets"
read_csv_with_header_check <- function(file_path) {
  # Read the first line of the file
  first_line <- readLines(file_path, n = 1)
  
  # Split the first line into columns
  first_row <- strsplit(first_line, split = ",")[[1]]
  
  # Check if the first column name is "barcode"
  has_header <- tolower(first_row[1]) == "barcode"
  
  # Read the file with the determined header
  data <- read.csv(file = file_path, header = has_header)
  
  return(data)
}


# Create individual Seurat objects ----------------------------------------

metadata <- read.csv("/fh/working/sun_w/zsui/st2image_data/metadata_brca_visium.csv")

seurat_list <- list()

for (i in 1:nrow(metadata)) {
  # Load the expression data
  features <- read.delim(file.path(metadata[i, "exprs_dir"], "features.tsv.gz"), header = FALSE)
  if (ncol(features) == 1) {
    exprs_data <- Read10X(data.dir = metadata[i, "exprs_dir"], gene.column = 1)
  } else {
    # Load the data
    exprs_data_result <- Read10X(data.dir = metadata[i, "exprs_dir"])
    
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
  
  # Load spatial metadata
  spatial_metadata <- read_csv_with_header_check(metadata[i, "spatial_path"])
  colnames(spatial_metadata) <- c("barcode", "tissue", "x", "y", "image_x", "image_y")
  spatial_metadata[, 2:ncol(spatial_metadata)] <- mutate_all(spatial_metadata[, 2:ncol(spatial_metadata)], function(x) as.numeric(as.character(x)))    
  # Filter spatial metadata to include only tissue-associated barcodes
  spatial_metadata <- spatial_metadata %>%
    filter(tissue == 1) %>%
    filter(barcode %in% colnames(exprs_data)) 
  if (any(!is.numeric(spatial_metadata$x)) || any(!is.numeric(spatial_metadata$y))) {
    stop("Spatial metadata contains non-numeric values in x or y columns.")
  }
  
  # Filter the expression matrix to include only barcodes in spatial_metadata
  filtered_matrix <- exprs_data[, colnames(exprs_data) %in% spatial_metadata$barcode]
  
  # Create a Seurat object
  slide_seurat <- CreateSeuratObject(counts = filtered_matrix, project = metadata[i, "slide"])
  # Add spatial metadata to the Seurat object
  slide_seurat <- AddMetaData(slide_seurat, metadata = spatial_metadata)
  # Add spatial coordinates to the Seurat object
  spatial_coords <- as.matrix(spatial_metadata[, c("x", "y")])
  rownames(spatial_coords) <- spatial_metadata$barcode
  spatial_coords <- spatial_coords[rownames(spatial_coords) %in% colnames(slide_seurat), , drop = FALSE]
  colnames(spatial_coords) <- c("spatial_1", "spatial_2")
  # Add spatial coordinates as a DimReduc object
  slide_seurat[["spatial"]] <- CreateDimReducObject(
    embeddings = spatial_coords,
    key = "spatial_",
    assay = DefaultAssay(slide_seurat)
  )
  # Store the Seurat object
  seurat_list[[metadata[i, "slide"]]] <- slide_seurat
}

save(seurat_list, file = file.path(st_dir, "visium_seurat_list.rda"))
load(file.path(st_dir, "visium_seurat_list.rda"))

# Ensure unique cell names and preprocess each Seurat object
for (i in seq_along(seurat_list)) {
  cat("Processing object", i, "\n")
  
  # Ensure unique cell names
  seurat_list[[i]] <- RenameCells(seurat_list[[i]], new.names = paste0(names(seurat_list)[i], "_", colnames(seurat_list[[i]])))
  # Ensure metadata is atomic
  seurat_list[[i]]@meta.data <- seurat_list[[i]]@meta.data %>%
    mutate(across(everything(), as.character))
  # Normalize data
  seurat_list[[i]] <- NormalizeData(seurat_list[[i]])
  # Find variable features
  seurat_list[[i]] <- FindVariableFeatures(seurat_list[[i]], selection.method = "vst", nfeatures = 2000)

  DefaultAssay(seurat_list[[i]]) <- "RNA"
}

save(seurat_list, file = file.path(st_dir, "visium_seurat_list2.rda"))

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


