# STpath-BC: Xenium Breast Cancer Models

H&E tiles, Xenium annotations, encoder features, XGBoost models, and TCGA-BRCA analyses.
The workflow and analysis organization follows [STpath-CRC](https://github.com/Sun-lab/STpath-CRC).

```text
Workflow/          Preparation, annotation, training, and TCGA steps (01–08)
Scripts_for_Analysis/ Figure 6, S9, S15, S16, and analysis commands
xenium_pipeline/   Shared code and Bilous preparation helpers
jobs/              One generic Slurm launcher
tables/            Curated inputs and numerical figure data
tests/             Core scientific checks
```

## Local setup

Use Python 3.11. Run these commands from this checkout (macOS/Linux):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install 'git+https://github.com/mahmoodlab/CONCH.git'
export XENIUM_ROOT=/absolute/path/to/Xenium
export STPATH_TCGA_TIFF_ROOT=/absolute/path/to/TCGA_BRCA_HE_filtered_tiff
```

CONCH is installed from its [upstream repository](https://github.com/mahmoodlab/CONCH#installation).

The checkout contains code and curated figure inputs. Put raw datasets, alignment
transforms, accepted annotation decisions, embeddings, models, and clinical inputs
under `XENIUM_ROOT`. Dataset folders are defined in `dataset_manifest.json`; result
and checkpoint paths are defined in `storage_layout.json`. Both resolve relative
to `XENIUM_ROOT`, which can be separate from the checkout.

CPU execution is supported. For an NVIDIA GPU, install the appropriate PyTorch
build and choose `--device cuda`. Training feature extraction also accepts
`--device mps` on Apple Silicon; use CPU if an encoder does not support MPS.
Large encoders and whole-slide cohorts need substantial RAM, disk space, and time.

Pathology encoders require access to their gated Hugging Face model repositories.
After access is granted, save a login once:

```bash
python Workflow/08_predict_tcga.py cohort -- login
```

Alternatively, provide `HF_TOKEN` in your environment. ResNet50 and regeneration
of Figure 6 from the included tables do not require gated model access.

## Local workflow

| Stage | Steps | Work |
| --- | --- | --- |
| A: Data Preparation | 01–03 | Prepare data and review H&E alignment |
| B: Cell Annotation | 04–06 | Annotate cells and save manual decisions |
| C: Features and Models | 07 | Extract features, build targets, train, and compare models |
| D: TCGA Preparation | 08 | Prepare and predict TCGA-BRCA |
| Analysis | Figure scripts | Run analyses and generate figures |

```bash
python Workflow/01_prepare_data.py he --dataset 10x
python Workflow/01_prepare_data.py tiles --dataset 10x
python Workflow/01_prepare_data.py select
python Workflow/01_prepare_data.py markers
jupyter lab Workflow/
```

Use steps 01–08 as the stage order; finish any required alignment before extracting
tiles. The alignment and annotation-review notebooks save manual decisions used
by later scripts. Repeat preparation for the datasets listed by `--help`.
After annotation is complete:

```bash
python Workflow/07_fit_models.py features --device cpu --batch-size 1 --workers 0
python Workflow/07_fit_models.py targets --cell-type tumor_cells
python Workflow/07_fit_models.py train --cell-type tumor_cells -- \
  --parallel-model-jobs 1 --nthread 1 --combined-nthread 1
```

Repeat `targets` and `train` for all five cell types. To include the ResNet50
comparator in a new training run, append `--include-resnet50` to the training
command. Training defaults to one model job and detects the available CPU budget;
the explicit one-thread settings above also work on smaller machines.

Extract Bilous images with
`python -m xenium_pipeline.bilous_extract_batch /path/to/Bilous_2026`.

For local TCGA prediction, use exported final models, training feature metadata,
and a verified TIFF resolution audit. This processes all eligible 75 µm tiles
for one slide without Slurm:

```bash
python Workflow/08_predict_tcga.py cohort -- local \
  --slides TCGA-4H-AAAK-01Z-00-DX1 \
  --input-dir "$STPATH_TCGA_TIFF_ROOT" \
  --audit-csv /absolute/path/to/tiff_resolution_audit.csv \
  --output-dir "$XENIUM_ROOT/tcga_brca_prediction/local_75um" \
  --crop-size-um 75 --device cpu --batch-size 1 --threads 1 \
  --include-completed-75um
```

Replace the slide ID with your TIFF filename or stem. Use `--all-slides` instead
of `--slides ...` to process a cohort sequentially. Repeat the same command to
resume; completed slides are reused. Add `--dry-run` to inspect the selected
manifest without inference. A local NVIDIA GPU can use `--device cuda`.
Optional `--model-dirs-json` and `--training-feature-root` select relocated exports.

Run each script with `--help`; forwarded runner options use `ACTION -- --help`.
Initial clustering uses r1.3; uncertain and epithelial reclustering use r0.7.
Normal-epithelial all-zero folds are included.

Run the checks with `python -m unittest discover -s tests`.

## Figures and survival analysis

| Figures | Code |
| --- | --- |
| 6A, 6B, 6C | `Scripts_for_Analysis/Figure6_ABC.py` |
| S9A and S9B | `Scripts_for_Analysis/FigureS9_AB.py` |
| S15, S16 | `Scripts_for_Analysis/FigureS15_S16.py` |

Figure 6 can be regenerated locally from the included tables:

```bash
python Scripts_for_Analysis/Figure6_ABC.py --output-dir figures
```

For S9, use the accepted annotation and tile inputs to generate source panels,
then assemble the extracted package:

```bash
python Scripts_for_Analysis/Run_Analyses.py reference
python Scripts_for_Analysis/FigureS9_AB.py /absolute/path/to/extracted_S9_package
```

With the full prediction and clinical inputs configured in `storage_layout.json`:

```bash
python Scripts_for_Analysis/Run_Analyses.py survival
python Scripts_for_Analysis/FigureS15_S16.py \
  --run-dir /absolute/path/to/completed_clinical_run \
  --output-dir supplementary_figures
```

Use the completed run directory printed by the survival command, or an existing
completed run. S15/S16 accept `--patient-table` for its full calibrated patient table. Unadjusted proportion
analyses use 424 patients; adjusted proportion and distance analyses use 413.

## Optional Slurm execution

Activate your Python environment before submitting. `jobs/run_job.sh` runs any
Python workflow command; account, partition, resources, and environment-module
setup are supplied by the user. TCGA submission commands use the same Python
interpreter as the submitting process, or the path set in `STPATH_PYTHON`.

```bash
export STPATH_SLURM_ACCOUNT=your_account
export STPATH_SLURM_PARTITION=your_gpu_partition
python Workflow/08_predict_tcga.py cohort -- submit \
  --all-slides --audit-csv /absolute/path/to/tiff_resolution_audit.csv \
  --output-dir "$XENIUM_ROOT/tcga_brca_prediction/cluster_75um" \
  --cpus 8 --mem 64G --time 2-00:00:00
```

Account and partition are optional when the cluster supplies defaults; they also
accept `--account` and `--partition` flags. `--gpus 0` runs the slide jobs on CPU.
Add `--dry-run` to inspect submission commands before scheduling.

| TCGA execution | Work assignment |
| --- | --- |
| `cohort -- submit` | One complete slide per job |
| `count -- submit`, then `groups -- submit` | Count tiles, balance tile ranges across GPU jobs, then merge per slide |

Grouped prediction requires the training-matched encoder weights in the local
Hugging Face cache. Its CPU merge resources use `--merge-cpus`, `--merge-mem`,
and `--merge-time`. All submission commands accept resource and cluster flags.

For other stages, submit the generic launcher directly, for example:

```bash
sbatch --account="$STPATH_SLURM_ACCOUNT" --partition=your_cpu_partition \
  --cpus-per-task=4 --mem=32G --time=1-00:00:00 \
  jobs/run_job.sh Workflow/07_fit_models.py train --cell-type tumor_cells -- \
  --parallel-model-jobs 1 --nthread 4 --combined-nthread 4
```

For arrays, a quoted `'{task}'` argument is replaced with `SLURM_ARRAY_TASK_ID`
inside the job, for example `--outer-fold '{task}'` or `--sample-index '{task}'`.

[License](LICENSE)
