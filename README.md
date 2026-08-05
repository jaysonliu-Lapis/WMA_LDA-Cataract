# PROP for cataract comorbidity classification

This repository packages the exact real-data cohort, ResNet-50 features, and R implementation used for the PROP cataract experiment. PROP is a model-averaging discriminant classifier: it screens variables, builds random-subspace DSDA base learners, learns simplex-constrained ensemble weights by quadratic programming, and classifies using the weighted discriminant score.

## Study task

The reproduced task is patient-level binary classification on the ODIR-5K training split:

| Label | Definition | Count |
| --- | --- | ---: |
| `0` | Patient has cataract (`C=1`) and no other patient-level disease flag | 146 |
| `1` | Patient has cataract (`C=1`) and at least one of `D/G/A/H/M/O` | 66 |

Each patient contributes the **right-eye image** only. Features were extracted with ImageNet-pretrained `torchvision` ResNet-50 (`IMAGENET1K_V2`), with the final fully connected layer replaced by an identity map. This produces a frozen 2,048-dimensional global-average-pooling representation.

The reported evaluation is 100 repeated stratified 2:1 train/test splits, using seeds 714--813. Every split has 141 training samples (97 class 0, 44 class 1) and 71 test samples (49 class 0, 22 class 1).

> Important: `C`, `D`, `G`, `A`, `H`, `M`, and `O` originate from ODIR's patient-level labels, while the input is a single right-eye image. This repository therefore supports the patient-level task stated above. It must not be described as a same-eye image-level co-disease dataset without rebuilding labels from same-eye annotations.

## Repository layout

```text
data/
  images/right/                 # 212 right-eye images actually used
  processed/
    cataract_metadata.csv       # all original ODIR fields for the 212 selected patients
    cataract_labels.csv         # model labels and covariates
    features_resnet50.csv       # 212 x (ID, filename, 2048 features)
    features_resnet50.npy       # 212 x 2048 float32 matrix
    dataset_manifest.json
    SHA256SUMS
src/PROP.R                      # reusable PROP functions
scripts/run_prop_experiment.R   # reproducible PROP experiment
scripts/package_used_data.py    # rebuilds this exact data package from the source project
results/published/              # original 100-repetition PROP result CSV
docs/DATASET_CARD.md            # variables, provenance, and redistribution note
```

## Run PROP

Install the required R packages once:

```powershell
& 'D:/R-4.4.1/bin/Rscript.exe' scripts/install_r_dependencies.R
```

Then run the experiment with the R installation that contains `data.table`, `quadprog`, and `TULIP`:

```powershell
& 'D:/R-4.4.1/bin/Rscript.exe' scripts/run_prop_experiment.R
```

The default run writes a new result CSV under `results/reproduced/`. It does not overwrite the archived PROP output. Useful options are:

```powershell
& 'D:/R-4.4.1/bin/Rscript.exe' scripts/run_prop_experiment.R --reps=5
& 'D:/R-4.4.1/bin/Rscript.exe' scripts/run_prop_experiment.R --screening=l1lr
& 'D:/R-4.4.1/bin/Rscript.exe' scripts/run_prop_experiment.R --transform=zscore
```

For Windows, the same command is wrapped by:

```powershell
.\scripts\run_reproduction.ps1 -Reps 100
```

## Rebuild the packaged data

The repository already contains the exact data package used for the reported study. To recreate it from the original project structure, use:

```powershell
& 'D:/anaconda3/python.exe' scripts/package_used_data.py --source-root D:/path/to/original-project
```

The packager validates the selected cohort, feature dimensions, image filenames, and output checksums.

## Archived PROP result

The archived PROP result is **27.00% +/- 3.65%** test error over 100 splits. The original per-repetition PROP output is in [results/published/PROP_ResNet50_raw_ttest_100_reps.csv](results/published/PROP_ResNet50_raw_ttest_100_reps.csv).

## Data availability and permission

The 212-image data package and derived ResNet-50 features are supplied through the collaborating hospital project for the associated paper. The project maintainer has confirmed that this subset, its metadata, and the derived features may be included in this public research repository. See [DATA_PERMISSION.md](DATA_PERMISSION.md) for the release statement.

The data provenance follows the ODIR-5K / Ocular Disease Recognition study data structure. Please cite the associated paper and the original ODIR-5K source when using this repository.

## Citation

Please cite the ODIR-5K source dataset and the paper associated with this repository. PROP uses DSDA base learners from Mai, Zou, and Yuan (2012), *Biometrika*, "A direct approach to sparse discriminant analysis in ultra-high dimensions." The repository code is released under the MIT License.
