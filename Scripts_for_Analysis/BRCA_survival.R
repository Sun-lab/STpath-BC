library(tidyverse)  
library(survival)
library(survminer)
library(glmnet)
library(GGally)
library(RColorBrewer)
library(TCGAbiolinks)
library(UCSCXenaTools)
library(ComplexHeatmap) # Required for Heatmap functions
library(circlize)       # Required for colorRamp2


## Read and preprocess main BRCA dataset --------------------------------------

df <- read.csv("BRCA_Distance_Clinical_Merged.csv")

df <- df %>%
  separate(
    col   = Sample_ID,
    into  = c("Project", "TSS", "Participant", "SampleType_Combined", "Portion", "SlideID"),
    sep   = "-",
    remove = FALSE,
    fill  = "right"
  ) %>%
  mutate(
    patient = paste("TCGA", TSS, Participant, sep = "-"),
    Stage_Group = case_when(
      ajcc_pathologic_tumor_stage %in%
        c("Stage I", "Stage IA", "Stage IB") ~ "Stage_I",
      ajcc_pathologic_tumor_stage %in%
        c("Stage II", "Stage IIA", "Stage IIB") ~ "Stage_II",
      ajcc_pathologic_tumor_stage %in%
        c("Stage III", "Stage IIIA", "Stage IIIB", "Stage IIIC") ~ "Stage_III",
      ajcc_pathologic_tumor_stage %in%
        c("Stage IV") ~ "Stage_IV",
      TRUE ~ NA_character_
    ),
    Stage_Group = factor(Stage_Group,
                         levels = c("Stage_I","Stage_II","Stage_III","Stage_IV"))
  ) %>%
  filter(!is.na(Stage_Group)) %>%
  select(-ajcc_pathologic_tumor_stage) %>%
  rename(Age = age_at_initial_pathologic_diagnosis)

subjects <- unique(df$Participant)
samples  <- paste0(unique(df$patient), "-01")


## Subtype Processing ----------------------------------------

# --- Xena Hub ---
xe <- XenaGenerate(subset = XenaHostNames == "tcgaHub")

xe_brca <- xe %>%
  XenaFilter(
    filterDatasets = "BRCA_clinicalMatrix",
    filterCohorts  = "TCGA Breast Invasive Carcinoma (BRCA)"
  )

query_brca  <- XenaQuery(xe_brca)
xe_download <- XenaDownload(query_brca, trans_slash = TRUE, force = TRUE)
subtypes_xe <- XenaPrepare(xe_download)

subtypes_xe <- subtypes_xe %>%
  filter(sampleID %in% samples) %>%
  select(sampleID, bcr_patient_barcode, PAM50Call_RNAseq, PAM50_mRNA_nature2012)


# --- PanCancer Atlas ---

subtypes_pan <- PanCancerAtlas_subtypes()
brca_subtypes <- subset(subtypes_pan, cancer.type == "BRCA")

brca_subtypes <- brca_subtypes %>%
  separate(
    pan.samplesID,
    into = c("Project", "TSS", "Participant", "Rest"),
    sep = "-",
    remove = FALSE,
    extra = "merge",
    fill = "right"
  ) %>%
  select(pan.samplesID, TSS, Participant, Subtype_Selected) %>%
  mutate(Subtype_Selected = gsub("BRCA.", "", Subtype_Selected),
         patient = paste("TCGA", TSS, Participant, sep = "-")) %>%
  filter(Participant %in% subjects)


# --- Original Marker Paper ---
brca_orig <- TCGAquery_subtype(tumor = "BRCA")
brca_orig <- brca_orig %>%
  separate(
    col = patient,
    into = c("Project", "TSS", "Participant"),
    sep = "-",
    remove = FALSE,
    fill = "right"
  ) %>% 
  select(patient, TSS, Participant, BRCA_Subtype_PAM50) %>%
  distinct(patient, .keep_all = TRUE)  %>%
  filter(Participant %in% subjects)

# --- Merging Subtypes ---
brca_subtypes <- brca_subtypes %>%
  left_join(brca_orig) %>%
  mutate(match = BRCA_Subtype_PAM50 == Subtype_Selected)

# --- Resolve multiple subtypes per patient ---
subtypes_per_TSS_P <- brca_subtypes %>%
  group_by(patient) %>%
  summarise(
    n_aliquots = n(),
    Subtypes_all = {
      v <- c(Subtype_Selected, BRCA_Subtype_PAM50)
      v <- unique(na.omit(v))
      paste(sort(v), collapse = "; ")
    },
    n_subtypes = {
      v <- c(Subtype_Selected, BRCA_Subtype_PAM50)
      length(unique(na.omit(v)))
    },
    .groups = "drop"
  )

brca_subtypes <- brca_subtypes %>%
  left_join(subtypes_per_TSS_P, by = "patient") %>%
  mutate(
    sample_id = str_extract(pan.samplesID, "^[^-]+-[^-]+-[^-]+-([0-9]{2})") %>%
      str_extract("[0-9]{2}")
  )

# Handle duplicates
subtypes_dup <- brca_subtypes %>% filter(n_subtypes > 1)
primary_codes <- sprintf("%02d", 1:9) # "01", "02", ..., "09"

subtypes_keep <- subtypes_dup %>%
  group_by(Participant) %>%
  mutate(has_01_09 = any(sample_id %in% primary_codes)) %>%
  ungroup() %>%
  filter((has_01_09 & sample_id %in% primary_codes) | (!has_01_09)) %>%
  select(-has_01_09) %>%
  group_by(Participant) %>%
  mutate(
    n_subtypes = n_distinct(Subtype_Selected, na.rm = TRUE),
    Subtypes_all = paste(sort(unique(Subtype_Selected)), collapse = "; ")
  ) %>%
  ungroup()

brca_subtypes <- brca_subtypes %>%
  filter(n_subtypes == 1) %>%
  rbind(subtypes_keep)

## --- Merge Xena + PanCancer + main df ---

subtypes_all <- brca_subtypes %>%
  left_join(subtypes_xe, by = c("patient" = "bcr_patient_barcode")) %>%
  mutate(PAM50Call_RNAseq = ifelse(is.na(PAM50Call_RNAseq), Subtypes_all, PAM50Call_RNAseq)) %>%
  distinct(Participant, patient, sampleID, PAM50Call_RNAseq) %>%
  rename(subtype = PAM50Call_RNAseq)

df <- subtypes_all %>% left_join(df)

## --- Check duplicated subjects ----

dup_subjects <- df %>%
  group_by(Participant) %>%
  summarise(
    n_IDs = n_distinct(Sample_ID),
    .groups = "drop"
  ) %>%
  filter(n_IDs > 1)

df %>%
  filter(Participant %in% dup_subjects$Participant)


# Variable Setup (Numeric & Cont.) ----------------------------------------

# Define Continuous Variables
cont_vars <- names(df)[sapply(df, is.numeric)]
cont_vars <- setdiff(cont_vars, c("PFI.time", "OS.time", "PFI", "OS"))

length(cont_vars)
cont_vars

df_long_cont <- df %>%
  select(all_of(cont_vars)) %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value")

# Distance & Proportion Patterns
distance_vars <- grep(".*_to_.*_(mean|std|median|iqr)$", names(df), value = TRUE)
dist_median <- grep(".*_to_.*_median$", names(df), value = TRUE)
dist_iqr <- grep(".*_to_.*_iqr$", names(df), value = TRUE)
dist_mean <- grep(".*_to_.*_mean$", names(df), value = TRUE)
dist_std  <- grep(".*_to_.*_std$",  names(df), value = TRUE)

prop_vars <- sort(grep("_overall_proportion$", colnames(df), value = TRUE))
rowSums(df[,  prop_vars])

demo_vars <- c("Age", "subtype")
strata_vars <- c("Stage_Group")



# Visualizations ----------------------------------------------------------

#### Proportions: KM tertiles + histograms  ####

# Histogram with cutpoints
cutpoints <- map_dfr(prop_vars, function(v) {
  x <- df[[v]]
  qs <- quantile(x, probs = c(1/3, 2/3), na.rm = TRUE)
  tibble(
    variable = v, cut1 = qs[1], cut2 = qs[2],
    n_low = sum(x < qs[1], na.rm = TRUE),
    n_median = sum(x >= qs[1] & x < qs[2], na.rm = TRUE),
    n_high = sum(x >= qs[2], na.rm = TRUE)
  )
})

cutpoints

p_prop_hist <- ggplot(
  df_long_cont %>% filter(variable %in% prop_vars), 
  aes(x = value)) +
  geom_histogram(bins = 40, fill = "grey30", color = "white") +
  facet_wrap(~ variable, scales = "free", ncol = 4) +
  theme_bw(base_size = 11) +
  geom_vline(aes(xintercept = cut1), data = cutpoints, linetype = "dashed", color = "blue") +
  geom_vline(aes(xintercept = cut2), data = cutpoints, linetype = "dashed", color = "blue") +
  labs(title = "Histograms: Overall Proportion", x = "Value", y = "Count")

ggsave("BRCA_histograms_proportions.jpg", p_prop_hist, height = 3, width = 10)

# Create tertile groups for proportions
df_tert <- df
prop_tert_vars <- paste0(prop_vars, "_tert")
p_prop_list <- list()

for (v in prop_vars) {
  qs <- quantile(df_tert[[v]], probs = c(1/3, 2/3), na.rm = TRUE)
  new_name <- paste0(v, "_tert")
  df_tert[[new_name]] <- cut(
    df_tert[[v]],
    breaks = c(-Inf, qs[1], qs[2], Inf),
    labels = c("Low", "Medium", "High"),
    include.lowest = TRUE
  )
}

# KM PFI by tertile groups for proportions
for (var in prop_tert_vars) {
  f <- paste0("Surv(PFI.time, PFI) ~ ", var)
  fit <- survfit(as.formula(f), data = df_tert)
  base_name <- gsub("_overall_proportion_tert$", "", var)
  
  p1 <- ggsurvplot(
    fit, 
    data = df_tert, 
    pval = TRUE,
    palette = c("green", "blue", "orange", "red"),
    risk.table = FALSE,
    legend.title = base_name,
    legend.labs = levels(df_tert[[var]]),
    xlab = "Time (days)", 
    ylab = "Progression-Free Survival Probability",
    ggtheme = theme_bw(base_size = 18)
  )
  p_prop_list[[var]] <- p1
}

combined_prop <- ggarrange(plotlist = lapply(p_prop_list[prop_tert_vars],
                                             function(x) x$plot), 
                           nrow = 1) +
  plot_annotation(title = "KM: PFI by Overall Proportion Groups", 
                  theme = theme(plot.title = element_text(hjust = 0.5, size = 22, face = "bold")))

ggsave("BRCA_KM_PFI_by_Overall_Proportion_Groups.png", combined_prop, width = 20, height = 6, dpi = 300)

p_all <- p_prop_hist / combined_prop
ggsave("BRCA_histogram_KM_PFI_by_Overall_Proportion_Groups.png", p_all, width = 20, height = 12, dpi = 300)



#### Distance: KM binary cut at median + histograms ####

# Histogram with binary cutpoints at median
dist_median_cuts <- map_dfr(dist_median, function(v) {
  x <- df[[v]]
  m <- median(x, na.rm = TRUE)
  tibble(
    variable = v, cutpoint_median = m,
    n_low = sum(x <= m, na.rm = TRUE), 
    n_high = sum(x > m, na.rm = TRUE)
  )
})

dist_median_cuts

p_dist_hist <- ggplot(
  df_long_cont %>% filter(variable %in% dist_median), 
  aes(x = value)) +
  geom_histogram(bins = 40, fill = "grey30", color = "white") +
  facet_wrap(~ variable, scales = "free", ncol = 3) +
  theme_bw(base_size = 11) +
  geom_vline(aes(xintercept = cutpoint_median),
             data = dist_median_cuts, linetype = "dashed", color = "blue") +
  labs(title = "Histograms: Median Distance", x = "Value", y = "Count")

ggsave("BRCA_histograms_median_distance.jpg", p_dist_hist, width = 10, height = 10)

# Create binary groups for median distance
df <- df %>%
  mutate(across(all_of(dist_median),
                ~ cut(.x, breaks = c(-Inf, median(.x, na.rm = TRUE), Inf),
                      labels = c("Low", "High"), include.lowest = TRUE),
                .names = "{.col}_bin"
  ))


# KM by median distance (Low vs High) 
distance_bin_vars <- paste0(dist_median, "_bin")
p_dist_median_list <- list()

for (var in distance_bin_vars) {
  f <- paste0("Surv(PFI.time, PFI) ~ ", var)
  fit <- survfit(as.formula(f), data = df)
  base_name <- gsub("_(mean|median|overall_proportion)_bin$", "", var)
  
  p1 <- ggsurvplot(
    fit, 
    data = df,
    pval = TRUE,
    palette = c("green", "blue", "orange", "red"),
    risk.table = FALSE, 
    legend.title = base_name,
    legend.labs = levels(df[[var]]),
    xlab = "Time (days)",
    ylab = "Progression-Free Survival Probability",
    ggtheme = theme_bw(base_size = 18)
  )
  p_dist_median_list[[var]] <- p1
}

combined_median <- ggarrange(
  plotlist = lapply(p_dist_median_list, function(x) x$plot),
  ncol = 3, nrow = ceiling(length(distance_bin_vars) / 3)
) +
  plot_annotation(title = "KM: PFI by Median Distance Groups",
                  theme = theme(plot.title = element_text(hjust = 0.5, size = 30, face = "bold")))

ggsave("BRCA_KM_PFI_by_Distance_All_Median_Groups.png", combined_median, width = 16, height = 24, dpi = 300)



#### Age: KM quartiles + histogram ####
# Histogram
p_age_hist <- ggplot(
  df_long_cont %>% filter(variable == "Age"), 
  aes(x = value)) +
  geom_histogram(bins = 40, fill = "grey30", color = "white") +
  facet_wrap(~ variable, scales = "free", ncol = 3) +
  theme_bw(base_size = 11) +
  labs(title = "Histograms: Age", x = "Value", y = "Count")

ggsave("BRCA_histograms_age.jpg", p_age_hist, width = 3, height = 3)

# Create quartiles for age
age_cuts <- quantile(df$Age, probs = c(0, 0.25, 0.50, 0.75, 1), na.rm = TRUE)
age_cuts
df <- df %>%
  mutate(Age_quart = cut(Age, 
                         breaks = age_cuts, 
                         include.lowest = TRUE,
                         labels = c("Q1", "Q2", "Q3", "Q4")))
df$Age_quart <- factor(df$Age_quart)

table(df$Age_quart)

# KM by age quartiles
fit_age <- survfit(Surv(PFI.time, PFI) ~ Age_quart, data = df)

png(paste0("BRCA_KM_PFI_by_age_quart.png"), width = 500, height = 500)
p1 <- ggsurvplot(
  fit_age,
  data = df,
  pval = TRUE, 
  palette = c("green", "blue", "orange", "red"),
  legend.title = "Age Quartiles", 
  legend.labs = levels(df[["Age_quart"]]),
  xlab = "Time (days)", 
  ylab = "Progression-Free Survival Probability",
  ggtheme = theme_bw(base_size = 18), 
  combine = TRUE)

print(p1)
dev.off()


#### Stage, Subtype: KM  ####

# KM by Stage
table(df$Stage_Group)
fit_pfi_stage <- survfit(Surv(PFI.time, PFI) ~ Stage_Group, data=df)

png("BRCA_KM_PFI_by_Stage.png", width = 500, height = 500)
ggsurvplot(
  fit_pfi_stage,
  data=df,
  palette=c("green","blue","orange","red"),
  pval=TRUE,
  conf.int=FALSE,
  legend.title="",
  legend.labs = levels(df$Stage_Group),
  risk.table=FALSE,
  xlab="Time (days)",
  ylab="Progression-Free Survival Probability",
  ggtheme=theme_bw(base_size=18),
  combine = TRUE
)
dev.off()

# KM by Subtype
df$subtype <- factor(df$subtype, 
                     levels = c("Basal", "Her2", "LumA", "LumB", "Normal"),
                     labels = c("Basal", "Her2", "LumA", "LumB", "Normal"))
table(df$subtype)
fit_pfi_subtype <- survfit(Surv(PFI.time, PFI) ~ subtype, data=df)

png("BRCA_KM_PFI_by_Subtype.png", width = 500, height = 500)
ggsurvplot(
  fit_pfi_subtype,
  data=df,
  palette=c("green","blue","orange","red", "purple"),
  pval=TRUE,
  conf.int=FALSE,
  legend.title="",
  legend.labs = levels(df$subtype),
  risk.table=FALSE,
  xlab="Time (days)",
  ylab="Progression-Free Survival Probability",
  ggtheme=theme_bw(base_size=18),
  combine = TRUE
)
dev.off()


# Correlation & Heatmaps --------------------------------------------------

#### Overall correlation heatmap ####

# Correlation matrix over distance + proportions
all_vars <- c(distance_vars, prop_vars)
cor_all <- cor(df[, all_vars],
               use = "pairwise.complete.obs",
               method = "pearson")

# Create heatmap
distance_stem <- sub("_(mean|median|std|iqr)$", "", distance_vars)
cell_from <- sub("_to_.*", "", distance_stem)
cell_to <- sub(".*_to_", "", distance_stem)

pair_id <- ifelse(cell_from < cell_to,
                  paste(cell_from, cell_to, sep = "_"),
                  paste(cell_to, cell_from, sep = "_"))
prop_pair_id <- rep("Overall_Proportion", length(prop_vars))

# Combined in the same order as cor_all rows/cols
pair_vec <- c(pair_id, prop_pair_id)
pair_fac <- factor(pair_vec, levels = unique(pair_vec))
pair_levels <- levels(pair_fac)
n_pairs <- length(pair_levels)
pair_colors <- structure(brewer.pal(min(n_pairs, 8), "Set2"), names = pair_levels)

top_ann <- HeatmapAnnotation(
  Pair = pair_fac,
  col  = list(Pair = pair_colors),
  show_annotation_name = FALSE,
  annotation_name_gp   = gpar(fontsize = 0),
  show_legend = TRUE,
  annotation_legend_param = list(
    title_gp  = gpar(fontsize = 8, fontface = "bold"),
    labels_gp = gpar(fontsize = 6)
  )
)

right_ann <- rowAnnotation(
  Pair = pair_fac,
  col = list(Pair = pair_colors),
  show_legend = FALSE,
  show_annotation_name = FALSE
)

ht <- Heatmap(
  cor_all,
  name = "corr",
  col = colorRamp2(c(-1, 0, 1), c("blue", "white", "red")),
  na_col = "white",
  cell_fun = NULL,
  
  # Block structure
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  row_split = pair_fac,
  column_split = pair_fac,
  
  heatmap_width = unit(18, "cm"),
  heatmap_height = unit(18, "cm"),
  row_title  = NULL,
  column_title = NULL,
  row_gap = unit(1, "mm"),
  column_gap = unit(1, "mm"),
  rect_gp = gpar(col = NA),
  border = FALSE,
  show_row_dend = FALSE,
  show_column_dend = FALSE,
  
  show_row_names = TRUE,
  show_column_names = TRUE,
  row_names_side = "left",
  column_names_rot = 90,
  row_names_gp = gpar(fontsize = 6),
  column_names_gp = gpar(fontsize = 6),
  
  heatmap_legend_param = list(
    title = "corr",
    at = c(-1, -0.5, 0, 0.5, 1),
    color_bar = "continuous"
  ),
  
  top_annotation = top_ann,
  right_annotation = right_ann
)

pdf("BRCA_corr_all_heatmap.pdf", width = 14, height = 10)
draw(
  ht,
  padding = unit(c(5, 5, 5, 5), "mm"),
  column_title = "Correlation: Distances + Proportions (Undirected Pair Blocks)",
  column_title_gp = gpar(fontsize = 12, fontface = "bold")
)
dev.off()

# GGpairs
outdir <- "BRCA_Directional_GGpairs_Facets"
dir.create(outdir, showWarnings = FALSE)

dir_stem <- sub("_(mean|median|std|iqr)$", "", distance_vars)
dir_groups <- split(distance_vars, dir_stem)

for (pair in names(dir_groups)) {
  
  vars <- dir_groups[[pair]]
  
  df_sub <- df[, vars, drop = FALSE]
  df_sub <- df_sub[, sapply(df_sub, is.numeric), drop = FALSE]
  
  if (ncol(df_sub) < 2) next
  
  colnames(df_sub) <- sub(".*_(mean|median|std|iqr)$", "\\1", vars)
  
  p <- ggpairs(
    df_sub,
    title = paste("Directional Pair:", pair),
    progress = FALSE
  )
  
  ggsave(file.path(outdir, paste0(pair, "_ggpairs.png")),
         p, width = 5, height = 5, dpi = 180)
}



# Marginal Cox Models (PFI) -----------------------------------------------

cox_covars <- c(demo_vars, dist_median, distance_bin_vars, prop_vars)
margin_cox_vars <- c(strata_vars, cox_covars)

marginal_pfi_results <- list()

for (v in margin_cox_vars) {
  formula <- as.formula(paste0("Surv(PFI.time, PFI) ~ ", v))
  fit <- coxph(formula, data=df)
  
  s <- summary(fit)
  
  marginal_pfi_results[[v]] <- data.frame(
    variable = v,
    variable1 = rownames(s$coefficients),
    HR = s$coefficients[,"exp(coef)"],
    lower95 = s$conf.int[,"lower .95"],
    upper95 = s$conf.int[,"upper .95"],
    pval = s$coefficients[,"Pr(>|z|)"],
    c_index = s$concordance[1]
  )
}

marginal_pfi_df <- bind_rows(marginal_pfi_results) %>%
  # arrange(pval)  %>%
  # mutate(signif = pval <= 0.05) %>%
  remove_rownames()

print(marginal_pfi_df)

write.csv(marginal_pfi_df, "BRCA_marginal_coxph_pfi_results.csv")


# Marginal Cox Models (OS) ------------------------------------------------


marginal_os_results <- list()

for (v in margin_cox_vars) {
  formula <- as.formula(paste0("Surv(OS.time, OS) ~ ", v))
  fit <- coxph(formula, data=df)
  
  s <- summary(fit)
  
  marginal_os_results[[v]] <- data.frame(
    variable = v,
    variable1 = rownames(s$coefficients),
    HR = s$coefficients[,"exp(coef)"],
    lower95 = s$conf.int[,"lower .95"],
    upper95 = s$conf.int[,"upper .95"],
    pval = s$coefficients[,"Pr(>|z|)"],
    c_index = s$concordance[1]
  )
}

marginal_os_df <- bind_rows(marginal_os_results) %>%
  # arrange(pval) %>%
  # mutate(signif = pval <= 0.05) %>%
  remove_rownames() 

print(marginal_os_df)

write_csv(marginal_os_df, "BRCA_marginal_coxph_os_results.csv")


# Adjusted Cox Models (PFI) ---------------------------------------

adjusted_vars <- c(dist_median, distance_bin_vars)

adjusted_pfi_results <- list()

for (v in adjusted_vars) {
  
  fml <- as.formula(
    paste0(
      "Surv(PFI.time, PFI) ~ ",
      v,
      " + Age + subtype + Stage_Group"
    )
  )
  
  fit <- coxph(fml, df)
  s <- summary(fit)
  
  adjusted_pfi_results[[v]] <- data.frame(
    variable = v,
    variable1 = rownames(s$coefficients),
    HR = s$coefficients[,"exp(coef)"],
    lower95 = s$conf.int[,"lower .95"],
    upper95 = s$conf.int[,"upper .95"],
    pval = s$coefficients[,"Pr(>|z|)"],
    c_index = s$concordance[1]
  )
}


adjusted_pfi_df <- bind_rows(adjusted_pfi_results) %>%
  # arrange(pval) %>%
  remove_rownames() %>%
  mutate(signif = pval <= 0.05) %>%
  filter(variable1 %in% c(dist_median, paste0(dist_median, "_binHigh")))

print(adjusted_pfi_df)

write_csv(adjusted_pfi_df, "BRCA_adjusted_marginal_coxph_pfi_results.csv")


# Adjusted Cox Models (OS) ---------------------------------------
adjusted_os_results <- list()

for (v in adjusted_vars) {
  
  fml <- as.formula(
    paste0(
      "Surv(OS.time, OS) ~ ",
      v,
      " + Age + subtype + Stage_Group"
    )
  )
  
  fit <- coxph(fml, df)
  s <- summary(fit)
  
  adjusted_os_results[[v]] <- data.frame(
    variable = v,
    variable1 = rownames(s$coefficients),
    HR = s$coefficients[,"exp(coef)"],
    lower95 = s$conf.int[,"lower .95"],
    upper95 = s$conf.int[,"upper .95"],
    pval = s$coefficients[,"Pr(>|z|)"],
    c_index = s$concordance[1]
  )
}


adjusted_os_df <- bind_rows(adjusted_os_results) %>%
  remove_rownames() %>%
  mutate(signif = pval <= 0.05) %>%
  filter(variable1 %in% c(dist_median, paste0(dist_median, "_binHigh")))

print(adjusted_os_df)

write_csv(adjusted_os_df, "BRCA_adjusted_marginal_coxph_os_results.csv")



# Joint Cox Models (PFI) ------------------------------------------------

joint_distance_vars <- c(distance_center_vars, distance_spread_vars)
celltype_covars <- c(joint_distance_vars, prop_used)

# Always include stage group (factor)
joint_formula_pfi <- as.formula(
  paste(
    "Surv(PFI.time, PFI) ~ ",
    paste(cox_covars, collapse = " + "),
    "+ subtype + strata(Stage_Group)"
  )
)


cox_joint_pfi <- coxph(joint_formula_pfi, data = df)
s <- summary(cox_joint_pfi)

joint_pfi <- data.frame(
  variable = rownames(s$coefficients),
  HR = s$coefficients[,"exp(coef)"],
  lower95 = s$conf.int[,"lower .95"],
  upper95 = s$conf.int[,"upper .95"],
  pval = s$coefficients[,"Pr(>|z|)"],
  c_index = s$concordance[1]
)


write.csv(joint_pfi,
          "BRCA_joint_coxph_pfi_results.csv")



# Joint Cox Models (OS) ------------------------------------------------

# Always include stage group (factor)
joint_formula_os <- as.formula(
  paste(
    "Surv(OS.time, OS) ~ ",
    paste(cox_covars, collapse = " + "),
    "+ subtype + strata(Stage_Group)"
  )
)

cox_joint_os <- coxph(joint_formula_os, data = df)
s <- summary(cox_joint_os)

joint_os <- data.frame(
  variable = rownames(s$coefficients),
  HR = s$coefficients[,"exp(coef)"],
  lower95 = s$conf.int[,"lower .95"],
  upper95 = s$conf.int[,"upper .95"],
  pval = s$coefficients[,"Pr(>|z|)"]
)


write.csv(joint_os,
          "BRCA_joint_coxph_os_results.csv")


