# PROP for cataract comorbidity classification

This repository packages the exact real-data cohort, ResNet-50 features, and R implementation used for the PROP cataract experiment. PROP is a model-averaging discriminant classifier: it screens variables, builds random-subspace DSDA base learners, learns simplex-constrained ensemble weights by quadratic programming, and classifies using the weighted discriminant score.

## Quick start (clone -> classification)

1. **Clone** the repository. The 2048-d ResNet-50 features and labels are
   already included under `data/processed/`, so **no Python/PyTorch install is
   needed** for classification.
2. **Install the R dependencies** once:

   ```powershell
   & 'D:/R-4.4.1/bin/Rscript.exe' scripts/install_r_dependencies.R
   ```

3. **Prepare the feature CSV** to classify, with the same schema as
   `data/processed/features_resnet50.csv` (`ID, filename, f0001 ... f2048`;
   `ID`/`filename` are optional), e.g. `new_features.csv`.
4. **Run classification**:

   ```powershell
   & 'D:/R-4.4.1/bin/Rscript.exe' scripts/run_prop_classify.R --input=new_features.csv
   ```

5. **Read the console output**: one row per sample, `ID filename Class`,
   where `Class` is **1** (cataract only) or **2** (cataract + comorbidity).

To regenerate the features from the images instead (requires PyTorch), see
[Feature extraction](#feature-extraction-optional).

To classify **your own eye images** (right-eye images -> features -> class), see
[Classify your own eye images](#classify-your-own-eye-images).

## Study task

The reproduced task is patient-level binary classification on the cataract_patient_dataset training split:

| Label | Definition | Count |
| --- | --- | ---: |
| `0` | Patient has cataract (`C=1`) and no other patient-level disease flag | 146 |
| `1` | Patient has cataract (`C=1`) and at least one of `D/G/A/H/M/O` | 66 |

Each patient contributes the **right-eye image** only. Features were extracted with ImageNet-pretrained `torchvision` ResNet-50 (`IMAGENET1K_V2`), with the final fully connected layer replaced by an identity map. This produces a frozen 2,048-dimensional global-average-pooling representation.

The reported evaluation is 100 repeated stratified 2:1 train/test splits, using seeds 714--813. Every split has 141 training samples (97 class 0, 44 class 1) and 71 test samples (49 class 0, 22 class 1).

> Important: `C`, `D`, `G`, `A`, `H`, `M`, and `O` originate from cataract_patient_dataset's patient-level labels, while the input is a single right-eye image. This repository therefore supports the patient-level task stated above. It must not be described as a same-eye image-level co-disease dataset without rebuilding labels from same-eye annotations.

## Repository layout

```text
data/
  images/right/                 # 212 right-eye images actually used (Patient_1.jpg ... Patient_212.jpg)
  processed/
    cataract_metadata.csv       # all original cataract_patient_dataset fields for the 212 selected patients
    cataract_labels.csv         # model labels and covariates
    features_resnet50.csv       # 212 x (ID, filename, 2048 features)
    features_resnet50.npy       # 212 x 2048 float32 matrix
    dataset_manifest.json
    SHA256SUMS
src/PROP.R                      # reusable PROP functions
scripts/run_prop_classify.R    # train PROP and classify feature vectors (console only)
scripts/extract_features.py   # (optional) regenerate 2048-d features from data/images/right with PyTorch
scripts/package_used_data.py    # rebuilds this exact data package from the source project
docs/DATASET_CARD.md            # variables, provenance, and redistribution note
```

## Feature extraction (optional)

The repository ships the extracted features in
`data/processed/features_resnet50.csv` and `data/processed/features_resnet50.npy`,
so the R classification pipeline does **not** require Python or PyTorch. Use
this script only to regenerate the features from the images:

```powershell
python scripts/extract_features.py
```

- Input: `data/images/right/` (`Patient_1.jpg` ... `Patient_212.jpg`)
- Output: overwrites `data/processed/features_resnet50.csv` and
  `data/processed/features_resnet50.npy`, then verifies the row order against
  `data/processed/cataract_labels.csv`
- Requirements (only for this optional step): Python 3, `torch`, `torchvision`
  (ResNet-50 `IMAGENET1K_V2`), `Pillow`, `pandas`, `numpy`, `tqdm`

### Classify your own eye images

1. Put your own **right-eye** images in a folder (e.g. `my_images/`), named
   `Patient_<number>.jpg` (for example `Patient_1.jpg`, `Patient_2.jpg`).
2. Extract their 2048-d features:

   ```powershell
   python scripts/extract_features.py --images-dir my_images --output-csv my_features.csv --output-npy my_features.npy
   ```

3. Classify them with the model trained on the packaged 212 samples:

   ```powershell
   & 'D:/R-4.4.1/bin/Rscript.exe' scripts/run_prop_classify.R --input=my_features.csv
   ```

4. Read the console output: `ID filename Class`, where `Class` is **1**
   (cataract only) or **2** (cataract + comorbidity).

When `--images-dir`/`--output-csv` are the packaged defaults, the script also
verifies the row order against `data/processed/cataract_labels.csv`; for custom
folders/outputs that check is skipped automatically.

## Train and classify with PROP

Install the required R packages once:

```powershell
& 'D:/R-4.4.1/bin/Rscript.exe' scripts/install_r_dependencies.R
```

### Function API (recommended for custom train/test splits)

Load the methods and call the two functions directly:

```r
source("src/PROP.R")

# 1) Train on your own training set -> fitted object with weights
model <- prop_train(X_train, y_train)   # y_train must be 1/2 (add 1L if your labels are 0/1)

# 2) Classify new feature rows -> class labels 1 or 2
pred <- prop_predict(model, X_new)
```

`prop_train()` returns the fitted PROP object (including the learned ensemble
`weights`); `prop_predict()` uses that object to classify new 2048-dimensional
feature rows and returns class 1 or 2. A fixed seed (default 2026) makes
training reproducible.

### Prepare your own train/test data

`scripts/run_prop_classify.R` also defines `prepare_prop_data()`, a hook that
loads the packaged training data (and optionally an input CSV) so you can make
your own split and call `prop_train()` / `prop_predict()` directly:

```r
source("scripts/run_prop_classify.R")     # defines prepare_prop_data() and loads src/PROP.R
d <- prepare_prop_data(input = "new_features.csv", train_ratio = 0.7)
X_train <- d$X_train                       # training features (70%)
y_train <- d$y_train                       # training labels as 1/2
X_test  <- d$X_test                        # held-out test features (30%), NULL if train_ratio is NULL
X_new   <- d$X_new                         # features to classify (input file, or test set if input is NULL)
```

`prepare_prop_data()` returns a list with `X_train`, `y_train`, `X_test`, `y_test`,
`X_new`, `id_col`, `fn_col`, `test_id_col`, `test_fn_col`, and `feature_columns`.
- `train_ratio = NULL` (default): all packaged data is used for training.
- `train_ratio = 0.7` (any value in (0, 1)): the packaged data is stratified by
  class into 70% train / 30% test. If `input` is `NULL`, `X_new` becomes the
  held-out test set so you can immediately evaluate `prop_predict()`.
You can also ignore the split and do your own on `X_train`/`y_train`, then call
`prop_train()` and `prop_predict()`.

### Command line (train on the packaged data, classify a CSV)

```powershell
& 'D:/R-4.4.1/bin/Rscript.exe' scripts/run_prop_classify.R --input=new_features.csv
```

Options: `--train-ratio=0.7`, `--screening=l1lr`, `--transform=zscore|rank_int`, `--seed=123`.
The input CSV must use the same schema as `data/processed/features_resnet50.csv`
(`ID, filename, f0001 ... f2048`; `ID`/`filename` are optional). If `--input` is
omitted, the script classifies the packaged training features themselves.
Results are printed to the console only; no files are written.

## Rebuild the packaged data

The repository already contains the exact data package used for the reported study. To recreate it from the original project structure, use:

```powershell
& 'D:/anaconda3/python.exe' scripts/package_used_data.py --source-root D:/path/to/original-project
```

The packager validates the selected cohort, feature dimensions, image filenames, and output checksums.

## Data availability and permission

The 212-image data package and derived ResNet-50 features are supplied through the collaborating hospital project for the associated paper. The project maintainer has confirmed that this subset, its metadata, and the derived features may be included in this public research repository. See [DATA_PERMISSION.md](DATA_PERMISSION.md) for the release statement.

The data provenance follows the cataract_patient_dataset study data structure. Please cite the associated paper and the original cataract_patient_dataset source when using this repository.

## Citation

Please cite the cataract_patient_dataset source dataset and the paper associated with this repository. PROP uses DSDA base learners from Mai, Zou, and Yuan (2012), *Biometrika*, "A direct approach to sparse discriminant analysis in ultra-high dimensions." The repository code is released under the MIT License.
