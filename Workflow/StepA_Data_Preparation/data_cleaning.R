
st_dir <- "/fh/working/sun_w/zsui/st2image_data/brca_visium_datasets"
# Get the list of dataset names
datasets <- list.dirs(path = st_dir, full.names = FALSE, recursive = FALSE)
# Get the list of dataset directories (folder names with the full path)
dataset_dirs <- list.dirs(path = st_dir, full.names = TRUE, recursive = FALSE)

# Organize to the same structure ------------------------------------------

# Iterate through each dataset directory
# Organize files in each dataset such that each slide has a subfolder and its files are moved
# into corresponding subfolder based on slide IDs extracted from the filenames
# for (dir in dataset_dirs) {
#   # List files within the current directory that match the pattern *_scalefactors_json.json
#   fns <- list.files(dir, "*scalefactors_json.json")
#   fns <- gsub("-scalefactors_json.json", "_scalefactors_json.json", fns)
#   # Extract the slide names by removing the suffix "_scalefactors_json.json"
#   slides <- gsub("_scalefactors_json\\.json|-scalefactors_json\\.json", "", fns)
#   # # Create new folder paths by appending slide IDs to the directory path
#   # new_fns <- file.path(dir,
#   #                      paste0(gsub("/fh/working/sun_w/zsui/st2image_data_visium/", "", dir),
#   #                             "_",
#   #                             slides))
#   # # Create directories for each slide ID
#   # sapply(new_fns, function(dir) {
#   #   if (!dir.exists(dir)) {
#   #     dir.create(dir, recursive = TRUE) # Create the directory, including any parent directories if needed
#   #     message(paste("Created:", dir))
#   #   } else {
#   #     message(paste("Already exists:", dir))
#   #   }
#   # })
# 
#   # List all files in the current directory
#   files <- list.files(path = dir, full.names = TRUE)
# 
#   sapply(slides, function(slide_id) {
#     # Create a folder name by replacing "_" with "-" in the slide ID
#     folder_name <- gsub("_", "-", slide_id)
#     folder_path <- file.path(dir, paste0(gsub("/fh/working/sun_w/zsui/st2image_data_visium/", "", dir),
#                                          "_",
#                                          slide_id))
# 
#     # Create the directory if it doesn't already exist
#     if (!dir.exists(folder_path)) {
#       dir.create(folder_path , recursive = TRUE)
#       message(paste("Created directory:", folder_path))
#     }
# 
#     # Identify files that match the slide ID (with either "-" or "_")
#     matching_files <- files[grepl(paste0("^", slide_id, "[-_]"), basename(files))]
# 
#     # Move and rename matching files to the corresponding directory
#     sapply(matching_files, function(file) {
#       # Replace "_" with "-" in the slide ID portion of the filename
#       new_filename <- gsub(paste0("^", slide_id), gsub("_", "-", slide_id), basename(file))
#       target_path <- file.path(folder_path, new_filename)
# 
#       # Move and rename the file
#       file.rename(file, target_path)
#       message(paste("Moved and renamed:", basename(file), "to", target_path))
#     })
#   })
# 
#   # # Move files to the corresponding new directories
#   # sapply(files, function(file) {
#   #   # Extract the slide ID from the file name (everything before the first underscore)
#   #   slide_id <- sub("_.*", "", basename(file))
#   #   # Find the corresponding folder based on the slide ID
#   #   folder <- new_fns[grep(slide_id, new_fns)]
#   #   # If exactly one matching folder is found, move the file to that folder
#   #   if (length(folder) == 1) {
#   #     file.rename(file, file.path(folder, basename(file)))
#   #     message(paste("Moved:", basename(file), "to", folder))
#   #   } else {
#   #     # If no matching folder or multiple matches are found, log a message
#   #     message(paste("No matching folder for:", basename(file)))
#   #   }
#   # })
# }


# Create directories for expression data ----------------------------------

library(rhdf5)
library(Matrix)

metadata <- data.frame()
for (dir in dataset_dirs) {
  dataset <- gsub(paste0(st_dir, "/"), "", dir)
  slides <- list.files(path = dir, pattern = paste0("^", dataset),
                       full.names = FALSE, recursive = FALSE,
                       include.dirs = TRUE)
  slide_dirs <- file.path(dir, slides)
  meta <- data.frame(dataset = dataset,
                     slide = slides,
                     dir = slide_dirs)
  meta$exprs_dir <- file.path(meta$dir, paste0(meta$slide, "_filtered_feature_bc_matrix"))
  metadata <- rbind(metadata, meta)
  for (i in 1:nrow(meta)){
    dir1 <- meta[i, "dir"]
    s1 <- meta[i, "slide"]
    count_matrix_dir <- meta[i, "exprs_dir"]
    if(dir.exists(count_matrix_dir)){
      files <- list.files(path = count_matrix_dir, full.names = TRUE)
      # Rename files
      sapply(files, function(file) {
        file_name <- basename(file)
        # Generate the new file name by removing everything before "matrix", "features", or "barcodes"
        new_file_name <- gsub(".*?(matrix|features|barcodes)", "\\1", file_name)
        # Define the new file path
        new_file_path <- file.path(count_matrix_dir, new_file_name)
        # Rename the file
        if (file.rename(file, new_file_path)) {
          message(paste("Renamed:", file_name, "to", new_file_name))
        } else {
          message(paste("Failed to rename:", file_name))
        }
      })
    } else {
      dir.create(count_matrix_dir, recursive = TRUE)
      files_to_move <- list.files(
        path = dir1, 
        pattern = "\\.(mtx|tsv|mtx\\.gz|tsv\\.gz)$", 
        full.names = TRUE, 
        recursive = TRUE
      )
      
      if (length(files_to_move) > 0){
        # Move and rename each file
        sapply(files_to_move, function(file) {
          file_name <- basename(file) # Extract the file name
          new_file_name <- gsub(".*?(matrix|features|barcodes)", "\\1", file_name)
          destination_path <- file.path(count_matrix_dir, new_file_name)
          if (file.rename(file, destination_path)) {
            message(paste("Moved and renamed:", file_name, "to", destination_path))
          } else {
            message(paste("Failed to move:", file_name))
          }
        })
      } else {
        h5_file <- list.files(path = dir1, pattern = "\\.h5$", full.names = TRUE)
        h5_data <- H5Fopen(h5_file)
        barcodes <- h5read(h5_data, "matrix/barcodes")
        features <- h5read(h5_data, "matrix/features/name")
        data <- h5read(h5_data, "matrix/data")
        indices <- h5read(h5_data, "matrix/indices")
        indptr <- h5read(h5_data, "matrix/indptr")
        shape <- h5read(h5_data, "matrix/shape")
        
        H5Fclose(h5_data)
        
        matrix <- sparseMatrix(
          i = indices + 1,          # Convert to 1-based index
          p = indptr,
          x = data,
          dims = c(shape[1], shape[2])
        )
        
        writeMM(matrix, file.path(count_matrix_dir, "matrix.mtx"))
        write.table(barcodes, file.path(count_matrix_dir, "barcodes.tsv"),
                    sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
        write.table(features, file.path(count_matrix_dir, "features.tsv"), sep = "\t", 
                    row.names = FALSE, col.names = FALSE, quote = FALSE)
        
      }
    }
  }
}


# Create directories for spatial information ------------------------------
metadata$spatial_dir <-  file.path(metadata$dir, paste0(metadata$slide, "_spatial"))
for (i in 1:nrow(metadata)) {
  s1 <- metadata[i, "slide"]
  dir1 <- metadata[i, "dir"]
  sp_dir <- metadata[i, "spatial_dir"]
  
  # Check if there is a folder containing "spatial" in dir1
  spatial_folder <- list.dirs(path = dir1, full.names = TRUE, recursive = FALSE)
  spatial_folder <- spatial_folder[grep("spatial", basename(spatial_folder), ignore.case = TRUE)]
  
  if (length(spatial_folder) == 1) {
    # Rename the folder to sp_dir
    if (file.rename(spatial_folder, sp_dir)) {
      message(paste("Renamed spatial folder to:", sp_dir))
    } else {
      stop(paste("Failed to rename spatial folder:", spatial_folder))
    }
  } else if (length(spatial_folder) > 1) {
    stop("Multiple spatial folders found. Manual resolution needed.")
  }
  
  # Check if the spatial directory does not exist
  if (!dir.exists(sp_dir)) {
    dir.create(sp_dir, recursive = TRUE)
    target_suffixes <- c(
      "tissue_positions_list.csv",
      "tissue_positions.csv",
      "tissue_lowres_image.png",
      "tissue_hires_image.png",
      "scalefactors_json.json",
      "detected_tissue_image.jpg",
      "aligned_fiducials.jpg"
    )
    files_to_move <- list.files(
      path = dir1,
      pattern = paste0(target_suffixes, collapse = "|"),
      full.names = TRUE,
      recursive = TRUE
    )
    sapply(files_to_move, function(file) {
      file_name <- basename(file)
      new_file_name <- sub(
        ".*?(tissue_positions_list\\.csv|tissue_positions\\.csv|tissue_lowres_image\\.png|tissue_hires_image\\.png|scalefactors_json\\.json|detected_tissue_image\\.jpg|aligned_fiducials\\.jpg)$",
        "\\1",
        file_name
      )
      destination_path <- file.path(sp_dir, new_file_name)
      if (file.rename(file, destination_path)) {
        message(paste("Moved and renamed:", file_name, "to", new_file_name))
      } else {
        message(paste("Failed to move:", file_name))
      }
    })
  }
}


# HTAN data ---------------------------------------------------------------
# 
# metadata_htan <- read.csv(file.path(st_dir, "Mo_2024", "metadata_HTAN_download.tsv"), sep = "\t")
# metadata_htan <- metadata_htan[grepl("Breast", metadata_htan$Organ, ignore.case = TRUE), ]
# fn_id_matching <- metadata_htan[, c("Filename", "Biospecimen", "File.Format")]
# fn_id_matching$fn <- sapply(strsplit(metadata_htan$Filename, "/"), "[[", 2)
# fn_id_matching <- fn_id_matching[fn_id_matching$File.Format %in% c("tiff", "json"), ]
# fn_id_matching$old_fn <- gsub("-scalefactors_json.json", "", fn_id_matching$fn)
# fn_id_matching$old_fn <- paste0("Mo_2024_", fn_id_matching$old_fn)
# folder_fn2id <- fn_id_matching[fn_id_matching$File.Format == "json", c("old_fn", "Biospecimen")]
# folder_fn2id$new_fn <- paste0("Mo_2024_", folder_fn2id$Biospecimen)
# 
# # Iterate over each row of the dataframe
# apply(folder_fn2id, 1, function(row) {
#   old_fn <- row["old_fn"]
#   new_fn <- row["new_fn"]
# 
#   # Define the old and new folder paths
#   old_folder <- file.path(file.path(st_dir, "Mo_2024"), old_fn)
#   new_folder <- file.path(file.path(st_dir, "Mo_2024"), new_fn)
# 
#   if (dir.exists(old_folder)) {
#     file.rename(old_folder, new_folder)
#     old_spatial <- file.path(new_folder, paste0(old_fn, "_spatial"))
#     new_spatial <- file.path(new_folder, paste0(new_fn, "_spatial"))
# 
#     old_filtered <- file.path(new_folder, paste0(old_fn, "_filtered_feature_bc_matrix"))
#     new_filtered <- file.path(new_folder, paste0(new_fn, "_filtered_feature_bc_matrix"))
# 
#     if (dir.exists(old_spatial)) {
#       file.rename(old_spatial, new_spatial)
#     } else {
#       message(paste("Subfolder not found:", old_spatial))
#     }
# 
#     if (dir.exists(old_filtered)) {
#       file.rename(old_filtered, new_filtered)
#     } else {
#       message(paste("Subfolder not found:", old_filtered))
#     }
# 
#     message(paste("Renamed:", old_fn, "to", new_fn))
#   } else {
#     message(paste("Folder not found:", old_folder))
#   }
# })
# 
# image_fn2id <- fn_id_matching[fn_id_matching$File.Format == "tiff", c("fn", "Biospecimen")]
# image_fn2id$old_fn <- file.path(st_dir, "Mo_2024", image_fn2id$fn)
# image_fn2id$new_fn <- file.path(st_dir, "Mo_2024", paste0("Mo_2024_", image_fn2id$Biospecimen), paste0("Mo_2024_", image_fn2id$Biospecimen, "_image.tif"))
# 
# # Iterate over each row of the dataframe
# apply(image_fn2id, 1, function(row) {
#   old_path <- as.character(row["old_fn"])
#   new_path <- as.character(row["new_fn"])
# 
#   # Check if the old path exists
#   if (file.exists(old_path)) {
#     # Rename the file or folder
#     if (file.rename(old_path, new_path)) {
#       message(paste("Renamed:", old_path, "to", new_path))
#     } else {
#       message(paste("Failed to rename:", old_path))
#     }
#   } else {
#     message(paste("Path not found:", old_path))
#   }
# })


# 
# # Bassiouni_2023 -----------------------------------------------------------
# 
# metadata1 <- metadata[metadata$dataset == "Bassiouni_2023",]
# metadata1$new_fn <- paste0(metadata1$slide, "_image.png")
# metadata1$old_fn <- sapply(strsplit(metadata1$slide, "_"), "[[", 3)
# metadata1$old_fn <- paste0("GSE210616_", metadata1$old_fn, ".png")
# 
# # Iterate over each row of the dataframe
# apply(metadata1, 1, function(row) {
#   old_fn <- row["old_fn"]
#   new_fn <- row["new_fn"]
# 
#   # Define the old and new folder paths
#   old_path <- file.path("/fh/working/sun_w/zsui/STpath/code/STpath pipeline", new_fn)
#   new_path <- file.path(file.path(st_dir, "Bassiouni_2023", row['slide']), new_fn)
# 
#   # Check if the old path exists
#   if (file.exists(old_path)) {
#     # Rename the file or folder
#     if (file.rename(old_path, new_path)) {
#       message(paste("Renamed:", old_path, "to", new_path))
#     } else {
#       message(paste("Failed to rename:", old_path))
#     }
#   } else {
#     message(paste("Path not found:", old_path))
#   }
# })
# 


# Create directories for full resolution WSIs ------------------------------
for (i in 1:nrow(metadata)) {
  s1 <- metadata[i, "slide"]
  dir1 <- metadata[i, "dir"]
  
  # Check if there is a file ending with "jpg", "tif", or "png" in dir1
  image_files <- list.files(
    path = dir1,
    pattern = "\\.(jpg|tif|png)$", # Match .jpg, .tif, or .png files
    full.names = TRUE,
    recursive = FALSE
  )
  
  if (length(image_files) == 1) {
    # Dynamically set the correct extension for the new file name
    file_extension <- tools::file_ext(image_files)
    new_image_file <- file.path(dir1, paste0(s1, "_image.", file_extension))
    
    # Rename the file
    if (file.rename(image_files, new_image_file)) {
      message(paste("Renamed image file to:", new_image_file))
    } else {
      stop(paste("Failed to rename image file:", image_files))
    }
    
    # Update metadata with the new image path
    metadata[i, "image_path"] <- new_image_file
  } else if (length(image_files) > 1) {
    stop("Multiple image files found. Manual resolution needed.")
  } else {
    message("No image file found in directory:", dir1)
  }
}

metadata$diameter <- NA  # To store paths of .json files
metadata$spatial_path <- NA  # To store paths of .csv files containing "tissue_positions"
for (i in 1:nrow(metadata)) {
  sp_dir <- metadata$spatial_dir[i]
  if (dir.exists(sp_dir)) {
    files <- list.files(path = sp_dir, full.names = TRUE)
    json_file <- files[endsWith(files, ".json")]
    if (length(json_file) == 1) {
      metadata$diameter[i] <- json_file[1]
    }
    csv_file <- files[grepl("tissue_positions", files) & endsWith(files, ".csv")]
    if (length(csv_file) == 1) {
      metadata$spatial_path[i] <- csv_file[1] 
    }
  } else {
    message(paste("Directory does not exist:", sp_dir))
  }
}

# metadata1 <- read.csv(file.path(st_dir, "metadata_brca_visium.csv"))
# subtypes <- metadata1[, c("slide", "subtype")]
# metadata <- merge(metadata, subtypes, by = "slide")

metadata$patch_dir <- file.path("/fh/working/sun_w/zsui/st2image_data/patches", metadata$slide)
metadata$proportion_dir <- file.path(st_dir, "CARD_proportions")
metadata$barcodes_path <- file.path(metadata$exprs_dir, "barcodes.tsv")
metadata$count_matrix_path <- file.path(metadata$exprs_dir, "matrix.mtx")
metadata$features_path <- file.path(metadata$exprs_dir, "features.tsv")
write.csv(metadata, file = "/fh/working/sun_w/zsui/st2image_data/metadata_brca_visium.csv",
          row.names = F)

metadata <- read.csv("/fh/working/sun_w/zsui/st2image_data/metadata_brca_visium.csv")

metadata$sample_id <- metadata$slide
metadata$output_dir <- metadata$patch_dir
metadata$barcode_col <- 1
metadata$array_row_col <- 3
metadata$array_col_col <- 4
metadata$pxl_x_col <- 6
metadata$pxl_y_col <- 5
metadata$in_tissue_col <- 2

write.csv(metadata[, c("sample_id", "image_path", "diameter", "spatial_path", 
                       "barcodes_path", "output_dir", "barcode_col", "in_tissue_col",
                       "array_row_col", "array_col_col", "pxl_x_col", "pxl_y_col")], 
          file = "/fh/working/sun_w/zsui/st2image_data/create_patches_input_brca_visium_all.csv",
          row.names = F)






# scRNA-seq ---------------------------------------------------------------

sc_dir <- "/fh/working/sun_w/zsui/st2image_data/brca_scRNAseq_datasets"


# HTAN --------------------------------------------------------------------

dir <- file.path(sc_dir, "Mo_2024")
# Organize files in the folder such that each patient has a subfolder and its files are moved
# into corresponding subfolder based on IDs extracted from the filenames
# List files within the current directory that match the pattern *barcodes.tsv.gz
fns <- list.files(dir, "*barcodes.tsv.gz")
# Extract the slide names by removing the suffix "-barcodes.tsv.gz"
patient_ids <- gsub("-barcodes\\.tsv\\.gz", "", fns)

# List all files in the current directory
files <- list.files(path = dir, full.names = TRUE)

sapply(patient_ids, function(p1) {
  # Create a folder name by replacing "_" with "-" in the slide ID
  folder_name <- p1
  folder_path <- file.path(dir, p1)
  
  # Create the directory if it doesn't already exist
  if (!dir.exists(folder_path)) {
    dir.create(folder_path , recursive = TRUE)
    message(paste("Created directory:", folder_path))
  }
  
  # Identify files that match the slide ID (with either "-" or "_")
  matching_files <- files[grepl(paste0("^", p1, "[-_]"), basename(files))]
  
  # Move and rename matching files to the corresponding directory
  sapply(matching_files, function(file) {
    target_path <- file.path(folder_path, basename(file))
    
    # Move and rename the file
    file.rename(file, target_path)
    message(paste("Moved and renamed:", basename(file), "to", target_path))
  })
})

patient_dirs <- list.dirs(dir, recursive = F)
for (pd1 in patient_dirs) {
  p1 <- basename(pd1)
  old_fns <- list.files(pd1) 
  new_fns <- gsub(paste0(p1, "-"), "", old_fns)
  file.rename(file.path(pd1, old_fns[1]), file.path(pd1, new_fns[1]))
  file.rename(file.path(pd1, old_fns[2]), file.path(pd1, new_fns[2]))
  file.rename(file.path(pd1, old_fns[3]), file.path(pd1, new_fns[3]))
}
patient_ids <- basename(patient_dirs)


# HTAN (Old) --------------------------------------------------------------

metadata_htan <- read.csv(file.path(sc_dir, "Mo_2024", "metadata.tsv"), sep = "\t")
metadata_htan <- metadata_htan[grepl("Breast", metadata_htan$Organ, ignore.case = TRUE), ]
fn_id_matching <- metadata_htan[, c("Filename", "Biospecimen")]
fn_id_matching$fn <- sapply(strsplit(metadata_htan$Filename, "/"), "[[", 2)
fn_id_matching$old_fn <- gsub(".gz", "", fn_id_matching$fn)
folder_fn2id <- fn_id_matching[, c("old_fn", "Biospecimen")]
folder_fn2id$slide <- paste0("Mo_2024_", folder_fn2id$Biospecimen)
folder_fn2id$new_fn <- paste0(folder_fn2id$slide, paste0("_", sapply(strsplit(folder_fn2id$old_fn, "-"), "[[", 3)))
folder_fn2id$new_fn1 <- file.path(paste0("Mo_2024_", folder_fn2id$Biospecimen), folder_fn2id$new_fn)

# Iterate over each row of the dataframe
apply(folder_fn2id, 1, function(row) {
  old_fn <- row["old_fn"]
  new_fn <- row["new_fn1"]
  
  # Define the old and new folder paths
  old_path <- file.path(file.path(sc_dir, "Mo_2024"), old_fn)
  new_path <- file.path(file.path(sc_dir, "Mo_2024"), new_fn)
  
  # Check if the old path exists
  if (file.exists(old_path)) {
    # Rename the file 
    dir.create(file.path(file.path(sc_dir, "Mo_2024"), row["slide"]), showWarnings = FALSE)
    if (file.rename(old_path, new_path)) {
      message(paste("Renamed:", old_path, "to", new_path))
    } else {
      message(paste("Failed to rename:", old_path))
    }
  } else {
    message(paste("Path not found:", old_path))
  }
})


folder_fn2id$sc_dir <- file.path(sc_dir, "Mo_2024", folder_fn2id$slide)
meta_sc_htan <- unique(folder_fn2id[, c("slide", "sc_dir")])


metadata <- read.csv("/fh/working/sun_w/zsui/st2image_data/metadata_brca_visium.csv")
metadata1 <- merge(metadata, meta_sc_htan, by = "slide", all = TRUE)

metadata1 <- metadata1[!is.na(metadata1$dataset), ]
